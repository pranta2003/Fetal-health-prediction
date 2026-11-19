"""src/fetal/train/ema.py

Exponential Moving Average (EMA) for PyTorch model weights.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Maintains moving average of model parameters with warmup and buffer copying.
    """

    def __init__(self, model: nn.Module, decay: float = 0.99, warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.backup: dict[str, torch.Tensor] = {}

    @property
    def shadow(self) -> dict[str, torch.Tensor]:
        return self.ema_model.state_dict()

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step)) if self.warmup else self.decay

        model_params = dict(model.named_parameters())
        for k, ema_p in self.ema_model.named_parameters():
            if k in model_params:
                ema_p.data.mul_(d).add_(model_params[k].data, alpha=1.0 - d)

        # Copy BatchNorm statistics and any buffers directly
        model_buffers = dict(model.named_buffers())
        for k, ema_b in self.ema_model.named_buffers():
            if k in model_buffers:
                ema_b.copy_(model_buffers[k])

    def apply_shadow(self, model: nn.Module):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema_model.state_dict())

    def restore(self, model: nn.Module):
        if self.backup:
            model.load_state_dict(self.backup)
            self.backup = {}

    def state_dict(self) -> dict:
        sd = self.ema_model.state_dict()
        sd["_ema_step"] = self.step
        return sd

    def load_state_dict(self, state_dict: dict):
        sd = dict(state_dict)
        if "_ema_step" in sd:
            self.step = int(sd.pop("_ema_step"))
        self.ema_model.load_state_dict(sd)

    def to(self, device):
        self.ema_model.to(device)
        return self
