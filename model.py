"""
model.py
--------
Model definition, loss combiner, and EMA helper for the 2D UNet trainer.

UNet2D wraps segmentation_models_pytorch and applies softmax in forward(),
so all downstream losses and metrics receive probabilities in [0, 1].
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from metrics import compute_loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UNet2D(nn.Module):
    """
    A 2D UNet model with various architectures built by segmentation_models_pytorch.
    forward() returns softmax probabilities, not logits.
    """

    def __init__(self,
                 num_channels=1, num_classes=2,
                 architecture='U-Net',
                 encoder_name='resnet34',
                 pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        encoder_weights = 'imagenet' if pretrained else None
        model_builder = {
            'U-Net':       smp.Unet,
            'U-Net++':     smp.UnetPlusPlus,
            'FPN':         smp.FPN,
            'PSPNet':      smp.PSPNet,
            'DeepLabV3':   smp.DeepLabV3,
            'DeepLabV3+':  smp.DeepLabV3Plus,
            'LinkNet':     smp.Linknet,
            'MA-Net':      smp.MAnet,
            'PAN':         smp.PAN,
            'UPerNet':     smp.UPerNet,
            'Segformer':   smp.Segformer,
        }[architecture]
        self.model = model_builder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=num_channels,
            classes=num_classes,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))

    def set_num_classes(self, num_classes):
        """
        Update the number of output classes, changing only the head and
        preserving existing weights for the classes that remain.
        """
        if num_classes == self.num_classes:
            return
        head   = self.model.segmentation_head
        layers = list(head.children())
        last   = layers[-1]
        new_conv = nn.Conv2d(
            in_channels=last.in_channels,
            out_channels=num_classes,
            kernel_size=last.kernel_size,
            stride=last.stride,
            padding=last.padding,
            bias=(last.bias is not None),
        )
        nn.init.xavier_uniform_(new_conv.weight)
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
        old_classes = min(self.num_classes, num_classes)
        new_conv.weight.data[:old_classes] = last.weight.data[:old_classes]
        if last.bias is not None:
            new_conv.bias.data[:old_classes] = last.bias.data[:old_classes]
        layers[-1] = new_conv
        self.model.segmentation_head = nn.Sequential(*layers)
        self.num_classes = num_classes


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(model: nn.Module, model_ema: nn.Module,
               decay: float = 0.99) -> None:
    for (_, v), (_, v_ema) in zip(model.state_dict().items(),
                                   model_ema.state_dict().items()):
        if torch.is_floating_point(v_ema):
            v_ema.mul_(decay).add_(v, alpha=1.0 - decay)
        else:
            v_ema.copy_(v)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    project_path: str | Path,
    n_channels:   int,
    architecture: str,
    encoder:      str,
    pretrained:   bool,
    reset:        bool = False,
) -> tuple[nn.Module, nn.Module, int]:
    """
    Load an existing checkpoint from project_path/model.ckpt, or create a
    fresh model if none exists or reset=True.

    Returns:
        model      : main model (train mode, gradients enabled)
        model_ema  : EMA copy  (eval mode, gradients disabled)
        num_classes: number of output channels (always 2: foreground + boundary)
    """
    model_path = Path(project_path) / 'model.ckpt'

    if model_path.exists() and not reset:
        checkpoint  = torch.load(model_path, weights_only=False)
        model       = checkpoint['model']
        num_classes = checkpoint['num_classes']
        print(f"Resuming from {model_path}  (num_classes={num_classes})")
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        num_classes = 2  # foreground + boundary
        model = UNet2D(
            num_channels = n_channels,
            num_classes  = num_classes,
            architecture = architecture,
            encoder_name = encoder,
            pretrained   = pretrained,
        )
        print(f"New model: {architecture} / {encoder}  "
              f"in_channels={n_channels}  num_classes={num_classes}")

    model_ema = copy.deepcopy(model).eval()
    for p in model_ema.parameters():
        p.requires_grad_(False)

    return model, model_ema, num_classes
