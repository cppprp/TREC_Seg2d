"""
train.py
--------
Offline training script for the 2D UNet on tilted plankton slices.

Saves a model.ckpt that predict.py can load directly without any changes.

Usage (local):
    python train.py --patches ./patches --project ./project_2d --epochs 50

Usage (HPC / SLURM):
    See the SLURM template at the bottom of this file.

All parameter defaults live in config.py — edit that file to change them
without touching the CLI or this script.
"""

import argparse
import copy
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

import wandb

from config  import DEFAULT_CONFIG
from dataset import FlatSliceDataset, TiltedSliceDataset
from metrics import compute_dice, compute_loss
from model   import build_model, ema_update


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    project_path = Path(args.project)
    project_path.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    def _disable_augment(ds):
        if isinstance(ds, ConcatDataset):
            for sub in ds.datasets:
                _disable_augment(sub)
        else:
            ds.augment    = False
            ds.transforms = None

    datasets = []
    if args.patches:
        datasets.append(TiltedSliceDataset(
            patches_dir      = args.patches,
            slices_per_patch = args.slices_per_patch,
            output_size      = args.input_size,
            n_channels       = args.n_channels,
            channel_spacing  = args.channel_spacing,
            max_tilt_deg     = args.max_tilt,
            augment          = True,
            preload          = True,
            norm_min         = args.norm_min,
            norm_max         = args.norm_max,
            intensity_aug    = args.intensity_aug,
            aug_brightness   = args.aug_brightness,
            aug_contrast     = args.aug_contrast,
            aug_gamma        = args.aug_gamma,
            aug_noise_std    = args.aug_noise_std,
        ))
    if args.patches_2d:
        datasets.append(FlatSliceDataset(
            patches_dir       = args.patches_2d,
            patches_per_image = args.patches_per_image,
            output_size       = args.input_size,
            augment           = True,
            preload           = True,
            norm_min          = args.norm_min,
            norm_max          = args.norm_max,
            intensity_aug     = args.intensity_aug,
            aug_brightness    = args.aug_brightness,
            aug_contrast      = args.aug_contrast,
            aug_gamma         = args.aug_gamma,
            aug_noise_std     = args.aug_noise_std,
        ))
    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # 80/20 train/val split
    n_val   = max(1, int(len(full_dataset) * 0.2))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds_tmp = random_split(full_dataset, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(42))

    # Val needs its own dataset object with augmentation disabled.
    val_dataset = copy.deepcopy(full_dataset)
    _disable_augment(val_dataset)
    val_ds = Subset(val_dataset, val_ds_tmp.indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # --- Model ---
    model, model_ema, num_classes = build_model(
        project_path = project_path,
        n_channels   = args.n_channels,
        architecture = args.architecture,
        encoder      = args.encoder,
        pretrained   = args.pretrained,
        reset        = args.reset,
    )

    model.to(device)
    model_ema.to(device)

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type in ('cuda', 'mps')))

    # LR scheduler: cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # --- WandB ---
    if args.wandb:
        wandb.init(
            project = args.wandb_project,
            name    = args.wandb_run,
            config  = vars(args),
        )

    # --- Training loop ---
    model_path       = project_path / 'model.ckpt'
    best_val_dice    = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        train_loss = 0.0
        n_batches  = 0

        for images, targets, weights in train_loader:
            images  = images.to(device,  non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device.type, enabled=(device.type in ('cuda', 'mps'))):
                preds = model(images)

            loss = compute_loss(preds.float(), targets, weights,
                                loss=args.loss,
                                weight_a=args.loss_weight_a,
                                weight_b=args.loss_weight_b)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            ema_update(model, model_ema, decay=args.ema_decay)

            if torch.isfinite(loss):
                train_loss += loss.item()
                n_batches  += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        # --- Validation ---
        model_ema.eval()
        val_loss      = 0.0
        val_dice_fg   = 0.0
        val_dice_bd   = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, targets, weights in val_loader:
                images  = images.to(device,  non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)

                with torch.autocast(device.type, enabled=(device.type in ('cuda', 'mps'))):
                    preds = model_ema(images)

                loss = compute_loss(preds.float(), targets, weights,
                                    loss=args.loss,
                                    weight_a=args.loss_weight_a,
                                    weight_b=args.loss_weight_b)

                dice_metrics   = compute_dice(preds, targets)
                val_loss      += loss.item() if torch.isfinite(loss) else 0.0
                val_dice_fg   += dice_metrics['dice_foreground']
                val_dice_bd   += dice_metrics['dice_boundary']
                n_val_batches += 1

        val_loss    /= max(n_val_batches, 1)
        val_dice_fg /= max(n_val_batches, 1)
        val_dice_bd /= max(n_val_batches, 1)

        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"dice_fg={val_dice_fg:.3f}  "
              f"dice_bd={val_dice_bd:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({elapsed:.1f}s)")

        if args.wandb:
            wandb.log({
                'epoch':           epoch,
                'train_loss':      train_loss,
                'val_loss':        val_loss,
                'dice_foreground': val_dice_fg,
                'dice_boundary':   val_dice_bd,
                'lr':              scheduler.get_last_lr()[0],
            })

        # --- Save best checkpoint ---
        val_dice_mean = (val_dice_fg + val_dice_bd) / 2.0
        if val_dice_mean > best_val_dice:
            best_val_dice    = val_dice_mean
            patience_counter = 0
            torch.save({
                'num_classes': num_classes,
                'model':       model_ema,   # loaded directly by predict.py
                'epoch':       epoch,
                'val_dice_fg': val_dice_fg,
                'val_dice_bd': val_dice_bd,
            }, model_path)
            print(f"  Saved best model (dice_mean={val_dice_mean:.3f})")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    print(f"\nDone. Best val dice (mean): {best_val_dice:.3f}")
    print(f"Model saved to: {model_path}")
    print(f"\nTo run inference, point predict_all_volumes() at: {project_path}")

    if args.wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    cfg = DEFAULT_CONFIG
    p = argparse.ArgumentParser(description='Train 2D UNet on tilted plankton slices')

    # Paths
    p.add_argument('--patches',    default=cfg.patches,    help='Root folder of annotated 3D patches (optional if --patches_2d is set)')
    p.add_argument('--patches_2d', default=cfg.patches_2d, help='Root folder of annotated 2D images (optional if --patches is set)')
    p.add_argument('--project',    required=True,           help='Output folder (model.ckpt saved here)')

    # Data
    p.add_argument('--slices_per_patch',  type=int,   default=cfg.slices_per_patch)
    p.add_argument('--patches_per_image', type=int,   default=cfg.patches_per_image)
    p.add_argument('--input_size',        type=int,   default=cfg.input_size)
    p.add_argument('--n_channels',        type=int,   default=cfg.n_channels)
    p.add_argument('--channel_spacing',   type=float, default=cfg.channel_spacing)
    p.add_argument('--max_tilt',          type=float, default=cfg.max_tilt)
    p.add_argument('--norm_min',          type=float, default=cfg.norm_min,
                   help='Fixed intensity lower bound — values below are clipped to 0')
    p.add_argument('--norm_max',          type=float, default=cfg.norm_max,
                   help='Fixed intensity upper bound — values above are clipped to 1')

    # Intensity augmentation (training image only; off for validation)
    p.add_argument('--intensity_aug', action=argparse.BooleanOptionalAction,
                   default=cfg.intensity_aug,
                   help='Enable photometric augmentation (--no-intensity_aug to disable)')
    p.add_argument('--aug_brightness',    type=float, default=cfg.aug_brightness,
                   help='Max additive brightness shift (0 disables)')
    p.add_argument('--aug_contrast',      type=float, default=cfg.aug_contrast,
                   help='Max contrast scale deviation (0 disables)')
    p.add_argument('--aug_gamma',         type=float, default=cfg.aug_gamma,
                   help='Max gamma deviation (0 disables)')
    p.add_argument('--aug_noise_std',     type=float, default=cfg.aug_noise_std,
                   help='Additive Gaussian noise std-dev (0 disables)')

    # Model
    p.add_argument('--architecture', default=cfg.architecture)
    p.add_argument('--encoder',      default=cfg.encoder)
    p.add_argument('--pretrained',   action='store_true', default=cfg.pretrained)
    p.add_argument('--reset',        action='store_true', default=cfg.reset,
                   help='Ignore existing checkpoint, start fresh')

    # Training
    p.add_argument('--epochs',       type=int,   default=cfg.epochs)
    p.add_argument('--batch_size',   type=int,   default=cfg.batch_size)
    p.add_argument('--lr',           type=float, default=cfg.lr)
    p.add_argument('--weight_decay', type=float, default=cfg.weight_decay)
    p.add_argument('--ema_decay',    type=float, default=cfg.ema_decay)
    p.add_argument('--loss',          default=cfg.loss,
                   help='Loss function: dice, ce, iou, mcc, dice_ce, iou_ce, mcc_ce')
    p.add_argument('--loss_weight_a', type=float, default=cfg.loss_weight_a,
                   help='Weight of the first component in combined losses')
    p.add_argument('--loss_weight_b', type=float, default=cfg.loss_weight_b,
                   help='Weight of the second component in combined losses')
    p.add_argument('--patience',     type=int,   default=cfg.patience,
                   help='Early stopping patience (0=disabled)')
    p.add_argument('--workers',      type=int,   default=cfg.workers)

    # Logging
    p.add_argument('--wandb',         action='store_true', default=cfg.wandb)
    p.add_argument('--wandb_project', default=cfg.wandb_project)
    p.add_argument('--wandb_run',     default=cfg.wandb_run)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not args.patches and not args.patches_2d:
        import sys; sys.exit("Error: at least one of --patches or --patches_2d must be provided")
    if args.patches_2d and args.n_channels != 1:
        import sys; sys.exit("Error: --patches_2d requires --n_channels 1 (2.5D mode has no meaning for 2D images)")
    train(args)
