import torch
import segmentation_models_pytorch as smp
from fetal import config

_DEFAULT = object()


def build_model(encoder_weights=_DEFAULT):
    if encoder_weights is _DEFAULT:
        encoder_weights = None if config.SMOKE else "imagenet"
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=4,
    )


def decode(logits):
    B, C, H, W = logits.shape
    p = torch.softmax(logits.float().reshape(B, C, -1), dim=-1)
    pm = p.reshape(B, C, H, W)
    xs = torch.arange(W, device=logits.device, dtype=torch.float32)
    ys = torch.arange(H, device=logits.device, dtype=torch.float32)
    return torch.stack([(pm.sum(2) * xs).sum(-1), (pm.sum(3) * ys).sum(-1)], -1), p
