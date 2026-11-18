"""src/fetal/train/ema.py

Exponential Moving Average (EMA) for PyTorch model weights.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Maintains moving average of model parameters.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.backup: dict[str, torch.Tensor] = {}

    @property
    def shadow(self) -> dict[str, torch.Tensor]:
        return self.ema_model.state_dict()

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_sd = model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            model_v = model_sd[k]
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)

    def apply_shadow(self, model: nn.Module):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema_model.state_dict())

    def restore(self, model: nn.Module):
        if self.backup:
            model.load_state_dict(self.backup)
            self.backup = {}

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        self.ema_model.to(device)
        return self
