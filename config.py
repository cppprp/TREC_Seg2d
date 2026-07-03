"""
config.py
---------
All tunable training parameters in one place.
Edit this file to change defaults — no need to touch train.py or the CLI.

Pass --config is not required; train.py reads DEFAULT_CONFIG automatically.
Any value can still be overridden on the command line, e.g.:
    python train.py --patches ./data --epochs 100 --lr 3e-4
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:

    # ── Paths ─────────────────────────────────────────────────────────────────
    patches:    str = ''             # root folder of annotated 3D patches (optional)
    patches_2d: str = ''             # root folder of annotated 2D images  (optional)
    project:    str = './project_2d' # output folder — model.ckpt is saved here

    # ── Data ──────────────────────────────────────────────────────────────────
    slices_per_patch:  int   = 30   # virtual samples drawn from each 3D patch per epoch
    patches_per_image: int   = 30   # random crops drawn from each 2D image per epoch
    input_size:       int   = 256   # slice H and W in pixels
    n_channels:       int   = 1     # 1 = single 2D slice,  3 = 2.5D triplet
    channel_spacing:  float = 1.0   # voxel gap between 2.5D channels
    max_tilt:         float = 90.0  # max tilt from XY plane in degrees (90 = fully random)

    # ── Model ─────────────────────────────────────────────────────────────────
    architecture: str  = 'U-Net'    # UNet2D architecture string passed to smp
    encoder:      str  = 'resnet34' # encoder backbone (any smp-compatible name)
    pretrained:   bool = True       # initialise encoder with ImageNet weights
    reset:        bool = False      # ignore existing checkpoint and start fresh

    # ── Optimiser ─────────────────────────────────────────────────────────────
    epochs:       int   = 50
    batch_size:   int   = 8
    lr:           float = 1e-4
    weight_decay: float = 1e-4
    ema_decay:    float = 0.99      # exponential moving average decay for model weights

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss:          str   = 'dice_ce' # loss function: dice, ce, iou, mcc, dice_ce, iou_ce, mcc_ce
    loss_weight_a: float = 0.5       # weight of the first  component in combined losses
    loss_weight_b: float = 0.5       # weight of the second component in combined losses

    # ── Early stopping ────────────────────────────────────────────────────────
    patience: int = 15              # epochs without val-dice improvement (0 = disabled)

    # ── DataLoader ────────────────────────────────────────────────────────────
    workers: int = 4                # number of DataLoader worker processes

    # ── Weights & Biases logging ──────────────────────────────────────────────
    wandb:         bool = False
    wandb_project: str  = 'plankton_2d'
    wandb_run:     str  = None      # run name shown in W&B (None = auto-generated)


# Single instance used as the source of all defaults in train.py
DEFAULT_CONFIG = TrainConfig()
