# FIELD: AC
# MODEL: EfficientNet-B4 Landmark U-Net
# STEP: model_definition
# USED BY: pipeline imports across work/scripts and work/tests

import torch
import segmentation_models_pytorch as smp
from fetal import config

_DEFAULT = object()


def build_model(encoder_weights=_DEFAULT, encoder_name="efficientnet-b4"):
    if encoder_weights is _DEFAULT:
        encoder_weights = None if config.SMOKE else (True if encoder_name.startswith("tu-") else "imagenet")
    elif encoder_name.startswith("tu-") and encoder_weights == "imagenet":
        encoder_weights = True
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=4,
    )
    if encoder_weights:
        print(f"pretrained weights loaded (encoder={encoder_name}, encoder_weights={encoder_weights})")
    return model


def load_pretrained_encoder(model, checkpoint_path, verbose=True):
    """
    Load encoder weights from a saved checkpoint (e.g. FASSD U-Net) into the model.
    Strict reporting: returns and prints how many encoder keys matched.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")

    model_sd = model.state_dict()
    model_encoder_keys = [k for k in model_sd.keys() if k.startswith("encoder.")]
    total_encoder_keys = len(model_encoder_keys)

    matched_dict = {}
    for k in model_encoder_keys:
        if k in state_dict and state_dict[k].shape == model_sd[k].shape:
            matched_dict[k] = state_dict[k]

    missing, unexpected = model.load_state_dict(matched_dict, strict=False)
    if verbose:
        print(
            f"[init-encoder] Loaded {len(matched_dict)} / {total_encoder_keys} "
            f"encoder keys from {checkpoint_path}"
        )
    return len(matched_dict), total_encoder_keys


def decode(logits):
    B, C, H, W = logits.shape
    p = torch.softmax(logits.float().reshape(B, C, -1), dim=-1)
    pm = p.reshape(B, C, H, W)
    xs = torch.arange(W, device=logits.device, dtype=torch.float32)
    ys = torch.arange(H, device=logits.device, dtype=torch.float32)
    return torch.stack([(pm.sum(2) * xs).sum(-1), (pm.sum(3) * ys).sum(-1)], -1), p
