"""HC18 U-Net segmentation model architecture and loss function.

Implements the 3-channel EfficientNet-B4 U-Net architecture and compound
Dice + BCE loss for fetal head circumference and biparietal diameter segmentation.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

ENCODER_NAME = "efficientnet-b4"
DEFAULT_IN_CHANNELS = 3
DEFAULT_CLASSES = 1


def build_segmentation_model(
    encoder_name: str = ENCODER_NAME,
    encoder_weights: str | None = "imagenet",
    in_channels: int = DEFAULT_IN_CHANNELS,
    classes: int = DEFAULT_CLASSES,
) -> nn.Module:
    """Build U-Net segmentation network with specified encoder.

    Args:
        encoder_name: Backbone encoder name (default: 'efficientnet-b4').
        encoder_weights: Pretrained weights (e.g. 'imagenet' or None).
        in_channels: Number of input image channels (default: 3).
        classes: Number of output segmentation masks (default: 1).

    Returns:
        smp.Unet model instance.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


class HC18Loss(nn.Module):
    """Compound segmentation loss: DiceLoss (binary) + BCEWithLogitsLoss."""

    def __init__(self, mode: str = "binary"):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode=mode)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute compound Dice + BCE loss.

        Args:
            preds: Raw model logits of shape (B, 1, H, W).
            targets: Binary target mask of shape (B, 1, H, W).

        Returns:
            Scalar loss tensor.
        """
        return self.dice_loss(preds, targets) + self.bce_loss(preds, targets)
