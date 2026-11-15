"""Femur landmark model definitions and peak decoders for ablation study.

Row A / C: U-Net EfficientNet-B4 predicting 1 heatmap channel with 2 peaks.
Row B: Direct coordinate regression (ConvNeXt-Tiny predicting 4 coordinates).
Includes subpixel peak decoder, order-invariant loss functions, and flip TTA.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fetal import config

_DEFAULT = object()


def build_femur_heatmap_model(encoder_weights: Any = _DEFAULT, classes: int = 1) -> nn.Module:
    """Build U-Net with EfficientNet-B4 backbone and heatmap output channel(s)."""
    import segmentation_models_pytorch as smp

    if encoder_weights is _DEFAULT:
        encoder_weights = None if config.SMOKE else "imagenet"
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
    )


class FemurHRNetHeatmap(nn.Module):
    """HRNet-W32 model with convolutional heatmap prediction head."""

    def __init__(self, pretrained: bool = True, classes: int = 1) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model("hrnet_w32", pretrained=pretrained, features_only=True)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        # feat 0 is the highest resolution branch: shape (B, 64, H/2, W/2)
        h = self.head(feats[0])
        return F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)


def build_femur_hrnet_model(encoder_weights: Any = _DEFAULT, classes: int = 1) -> nn.Module:
    """Build HRNet-W32 backbone with heatmap prediction head."""
    pretrained = (encoder_weights is not None) if encoder_weights is not _DEFAULT else (not config.SMOKE)
    return FemurHRNetHeatmap(pretrained=pretrained, classes=classes)


def build_femur_regression_model(weights: Any = _DEFAULT) -> nn.Module:
    """Build ConvNeXt-Tiny direct coordinate regression model predicting 4 numbers [x1, y1, x2, y2]."""
    import torchvision.models as tvm

    if weights is _DEFAULT:
        w = None if config.SMOKE else tvm.ConvNeXt_Tiny_Weights.DEFAULT
    else:
        w = weights
    model = tvm.convnext_tiny(weights=w)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 4)
    return model


def generate_femur_heatmap(pts: torch.Tensor, S: int, sigma: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate dual-Gaussian heatmap mixture for two femur endpoints.

    Parameters
    ----------
    pts : torch.Tensor
        (B, 2, 2) endpoint coordinates in (x, y) space.
    S : int
        Canvas resolution (e.g. 384 or 512).
    sigma : float
        Gaussian standard deviation in pixels.

    Returns
    -------
    prob_dist : torch.Tensor
        (B, 1, S, S) normalized probability distribution (sums to 1).
    unnorm_map : torch.Tensor
        (B, 1, S, S) unnormalized Gaussian peak map with max 1.0 per peak.
    """
    B = pts.shape[0]
    device = pts.device
    xs = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, 1, S)
    ys = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, S, 1)

    p1_x, p1_y = pts[:, 0, 0:1, None, None], pts[:, 0, 1:2, None, None]
    p2_x, p2_y = pts[:, 1, 0:1, None, None], pts[:, 1, 1:2, None, None]

    g1 = torch.exp(-((xs - p1_x) ** 2 + (ys - p1_y) ** 2) / (2.0 * sigma ** 2))
    g2 = torch.exp(-((xs - p2_x) ** 2 + (ys - p2_y) ** 2) / (2.0 * sigma ** 2))

    unnorm = torch.clamp(g1 + g2, 0.0, 1.0)
    flat = unnorm.view(B, 1, -1) + 1e-12
    prob = (flat / flat.sum(-1, keepdim=True)).view(B, 1, S, S)
    return prob, unnorm


def decode_femur_peaks(
    heatmap: torch.Tensor,
    S: int | None = None,
    k: int = 9,
    r: int = 8,
    min_dist: float = 8.0,
) -> torch.Tensor:
    """Decode top-2 peak coordinates from single-channel heatmap with sub-pixel refinement.

    Parameters
    ----------
    heatmap : torch.Tensor
        (B, 1, H, W) heatmap logits or probability map.
    S : int | None
        Canvas dimension.
    k : int
        Local max pooling kernel size.
    r : int
        Sub-pixel window radius.
    min_dist : float
        Minimum spatial separation between the two detected peaks in pixels.

    Returns
    -------
    torch.Tensor
        (B, 2, 2) decoded coordinates in (x, y) order on [0, W] x [0, H].
    """
    B, C, H, W = heatmap.shape
    assert C == 1, f"Expected 1 channel, got {C}"
    if S is None:
        S = H

    # Convert logits to probability distribution if needed
    if (heatmap < 0).any() or (heatmap > 1.01).any():
        flat = torch.softmax(heatmap.float().reshape(B, -1), dim=-1)
        prob = flat.reshape(B, 1, H, W)
    else:
        prob = heatmap.float()

    hmax = F.max_pool2d(prob, kernel_size=k, stride=1, padding=k // 2)
    peaks_mask = (prob == hmax)

    batch_pts = []
    for b in range(B):
        p_b = prob[b, 0]
        mask_b = peaks_mask[b, 0]
        coords = torch.nonzero(mask_b)  # (N, 2) in (y, x)

        if len(coords) >= 2:
            vals = p_b[coords[:, 0], coords[:, 1]]
            top_count = min(len(coords), 16)
            top_idx = torch.topk(vals, top_count).indices
            top_coords = coords[top_idx]

            p1 = top_coords[0]
            p2 = None
            for cand in top_coords[1:]:
                dist = torch.hypot((cand[0] - p1[0]).float(), (cand[1] - p1[1]).float())
                if dist >= min_dist:
                    p2 = cand
                    break
            if p2 is None:
                p2 = top_coords[1]
            chosen = [p1, p2]
        else:
            # Robust fallback: global argmax and second max outside exclusion radius
            flat_b = p_b.flatten()
            idx1 = flat_b.argmax()
            y1, x1 = idx1 // W, idx1 % W
            p1 = torch.stack([y1, x1])

            p_b_masked = p_b.clone()
            y_min, y_max = max(0, int(y1 - min_dist)), min(H, int(y1 + min_dist + 1))
            x_min, x_max = max(0, int(x1 - min_dist)), min(W, int(x1 + min_dist + 1))
            p_b_masked[y_min:y_max, x_min:x_max] = -1.0
            idx2 = p_b_masked.flatten().argmax()
            y2, x2 = idx2 // W, idx2 % W
            p2 = torch.stack([y2, x2])
            chosen = [p1, p2]

        # Sub-pixel center-of-mass refinement around chosen local peaks
        refined = []
        for cy, cx in chosen:
            cy_i, cx_i = int(cy), int(cx)
            y_min, y_max = max(0, cy_i - r), min(H, cy_i + r + 1)
            x_min, x_max = max(0, cx_i - r), min(W, cx_i + r + 1)
            win = p_b[y_min:y_max, x_min:x_max]
            w_sum = win.sum() + 1e-12
            wx = torch.arange(x_min, x_max, dtype=torch.float32, device=p_b.device)
            wy = torch.arange(y_min, y_max, dtype=torch.float32, device=p_b.device)
            sub_x = (win.sum(0) * wx).sum() / w_sum
            sub_y = (win.sum(1) * wy).sum() / w_sum
            refined.append(torch.stack([sub_x, sub_y]))

        batch_pts.append(torch.stack(refined))

    return torch.stack(batch_pts)


def decode_femur_peaks_dark(
    heatmap: torch.Tensor,
    S: int | None = None,
    k: int = 9,
    min_dist: float = 8.0,
) -> torch.Tensor:
    """Decode top-2 peak coordinates using DARK-style (Distribution-Aware Coordinate Representation) 2D Taylor expansion.

    Parameters
    ----------
    heatmap : torch.Tensor
        (B, 1, H, W) heatmap logits or probability map.
    S : int | None
        Canvas dimension.
    k : int
        Local max pooling kernel size.
    min_dist : float
        Minimum spatial separation between the two detected peaks in pixels.

    Returns
    -------
    torch.Tensor
        (B, 2, 2) sub-pixel decoded coordinates in (x, y) space.
    """
    B, C, H, W = heatmap.shape
    assert C == 1, f"Expected 1 channel, got {C}"
    if S is None:
        S = H

    if (heatmap < 0).any() or (heatmap > 1.01).any():
        flat = torch.softmax(heatmap.float().reshape(B, -1), dim=-1)
        prob = flat.reshape(B, 1, H, W)
    else:
        prob = heatmap.float()

    log_p = torch.log(prob + 1e-12)
    k_blur = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=heatmap.device).view(1, 1, 3, 3) / 16.0
    log_p_smooth = F.conv2d(log_p, k_blur, padding=1)

    hmax = F.max_pool2d(prob, kernel_size=k, stride=1, padding=k // 2)
    peaks_mask = (prob == hmax)

    batch_pts = []
    for b in range(B):
        p_b = prob[b, 0]
        lp_b = log_p_smooth[b, 0]
        mask_b = peaks_mask[b, 0]
        coords = torch.nonzero(mask_b)

        if len(coords) >= 2:
            vals = p_b[coords[:, 0], coords[:, 1]]
            top_count = min(len(coords), 16)
            top_idx = torch.topk(vals, top_count).indices
            top_coords = coords[top_idx]
            p1 = top_coords[0]
            p2 = None
            for cand in top_coords[1:]:
                dist = torch.hypot((cand[0] - p1[0]).float(), (cand[1] - p1[1]).float())
                if dist >= min_dist:
                    p2 = cand
                    break
            if p2 is None:
                p2 = top_coords[1]
            chosen = [p1, p2]
        else:
            flat_b = p_b.flatten()
            idx1 = flat_b.argmax()
            y1, x1 = idx1 // W, idx1 % W
            p1 = torch.stack([y1, x1])
            p_b_masked = p_b.clone()
            y_min, y_max = max(0, int(y1 - min_dist)), min(H, int(y1 + min_dist + 1))
            x_min, x_max = max(0, int(x1 - min_dist)), min(W, int(x1 + min_dist + 1))
            p_b_masked[y_min:y_max, x_min:x_max] = -1.0
            idx2 = p_b_masked.flatten().argmax()
            y2, x2 = idx2 // W, idx2 % W
            p2 = torch.stack([y2, x2])
            chosen = [p1, p2]

        refined = []
        for cy, cx in chosen:
            y, x = int(cy), int(cx)
            if 1 <= y < H - 1 and 1 <= x < W - 1:
                # 1st derivatives
                dx = 0.5 * (lp_b[y, x + 1] - lp_b[y, x - 1])
                dy = 0.5 * (lp_b[y + 1, x] - lp_b[y - 1, x])
                # 2nd derivatives (Hessian)
                dxx = lp_b[y, x + 1] - 2.0 * lp_b[y, x] + lp_b[y, x - 1]
                dyy = lp_b[y + 1, x] - 2.0 * lp_b[y, x] + lp_b[y - 1, x]
                dxy = 0.25 * (lp_b[y + 1, x + 1] - lp_b[y + 1, x - 1] - lp_b[y - 1, x + 1] + lp_b[y - 1, x - 1])

                det = dxx * dyy - dxy * dxy
                if abs(det) > 1e-6 and dxx < 0 and dyy < 0:
                    shift_x = -(dx * dyy - dy * dxy) / det
                    shift_y = -(dy * dxx - dx * dxy) / det
                    shift_x = torch.clamp(shift_x, -0.5, 0.5)
                    shift_y = torch.clamp(shift_y, -0.5, 0.5)
                else:
                    shift_x = -dx / torch.clamp(dxx, max=-1e-4) if dxx < 0 else torch.tensor(0.0, device=lp_b.device)
                    shift_y = -dy / torch.clamp(dyy, max=-1e-4) if dyy < 0 else torch.tensor(0.0, device=lp_b.device)
                    shift_x = torch.clamp(shift_x, -0.5, 0.5)
                    shift_y = torch.clamp(shift_y, -0.5, 0.5)
                refined.append(torch.stack([float(x) + shift_x, float(y) + shift_y]))
            else:
                refined.append(torch.tensor([float(x), float(y)], device=lp_b.device))
        batch_pts.append(torch.stack(refined))

    return torch.stack(batch_pts)


def soft_argmax_2d(logits: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Compute 2D spatial soft-argmax (integral coordinates) for each channel in logits.

    Parameters
    ----------
    logits : torch.Tensor
        (B, C, H, W) raw model logits.
    beta : float
        Softmax temperature parameter.

    Returns
    -------
    torch.Tensor
        (B, C, 2) sub-pixel coordinates in (x, y) space on [0, W-1] x [0, H-1].
    """
    B, C, H, W = logits.shape
    device = logits.device
    flat = logits.view(B, C, -1)
    prob = torch.softmax(flat * beta, dim=-1).view(B, C, H, W)

    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    ys = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)

    pred_x = (prob * xs).sum(dim=(-2, -1))
    pred_y = (prob * ys).sum(dim=(-2, -1))
    return torch.stack([pred_x, pred_y], dim=-1)


def generate_femur_boneline(pts: torch.Tensor, S: int, sigma: float = 3.0) -> torch.Tensor:
    """Generate thin Gaussian bone line segment connecting the two femur endpoints.

    Parameters
    ----------
    pts : torch.Tensor
        (B, 2, 2) endpoint coordinates in (x, y) space.
    S : int
        Canvas resolution.
    sigma : float
        Gaussian falloff width perpendicular to the segment.

    Returns
    -------
    torch.Tensor
        (B, 1, S, S) target bone line heatmap with values in [0, 1].
    """
    B = pts.shape[0]
    device = pts.device
    xs = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, 1, S)
    ys = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, S, 1)

    p1_x, p1_y = pts[:, 0, 0:1, None, None], pts[:, 0, 1:2, None, None]
    p2_x, p2_y = pts[:, 1, 0:1, None, None], pts[:, 1, 1:2, None, None]

    vx = p2_x - p1_x
    vy = p2_y - p1_y
    l2 = vx ** 2 + vy ** 2 + 1e-8

    t = torch.clamp(((xs - p1_x) * vx + (ys - p1_y) * vy) / l2, 0.0, 1.0)
    qx = p1_x + t * vx
    qy = p1_y + t * vy

    d2 = (xs - qx) ** 2 + (ys - qy) ** 2
    return torch.exp(-d2 / (2.0 * sigma ** 2))


def femur_boneline_loss(
    logits: torch.Tensor,
    gt_pts: torch.Tensor,
    S: int,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Compute binary cross-entropy loss between predicted bone line channel and Gaussian bone line target."""
    target = generate_femur_boneline(gt_pts, S, sigma=sigma)
    return F.binary_cross_entropy_with_logits(logits, target)


def match_endpoints(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find permutation of pred (B, 2, 2) that minimizes distance to gt (B, 2, 2).

    Parameters
    ----------
    pred : torch.Tensor
        (B, 2, 2) predicted endpoints.
    gt : torch.Tensor
        (B, 2, 2) ground truth endpoints.

    Returns
    -------
    matched_pred : torch.Tensor
        (B, 2, 2) predicted endpoints aligned to gt.
    order : torch.Tensor
        (B, 2) permutation indices.
    """
    B = pred.shape[0]
    # Ordering 0: [0, 1]
    d0 = (pred - gt).norm(dim=-1).sum(-1)
    # Ordering 1: [1, 0]
    pred_swapped = torch.stack([pred[:, 1], pred[:, 0]], dim=1)
    d1 = (pred_swapped - gt).norm(dim=-1).sum(-1)

    use_swap = (d1 < d0)
    matched = torch.where(use_swap[:, None, None], pred_swapped, pred)
    order = torch.where(use_swap[:, None], torch.tensor([1, 0], device=pred.device), torch.tensor([0, 1], device=pred.device))
    return matched, order


def femur_order_invariant_loss(pred_pts: torch.Tensor, gt_pts: torch.Tensor) -> torch.Tensor:
    """Compute order-invariant coordinate loss: min over the two endpoint orderings.

    Parameters
    ----------
    pred_pts : torch.Tensor
        (B, 2, 2) or (B, 4) predicted coordinates.
    gt_pts : torch.Tensor
        (B, 2, 2) or (B, 4) ground truth coordinates.

    Returns
    -------
    torch.Tensor
        Scalar order-invariant loss (L1 / distance normalized).
    """
    if pred_pts.dim() == 2 and pred_pts.shape[1] == 4:
        pred = pred_pts.view(-1, 2, 2)
    else:
        pred = pred_pts

    if gt_pts.dim() == 2 and gt_pts.shape[1] == 4:
        gt = gt_pts.view(-1, 2, 2)
    else:
        gt = gt_pts

    # Ordering 1: P1 -> G1, P2 -> G2
    d1 = (pred[:, 0] - gt[:, 0]).norm(dim=-1) + (pred[:, 1] - gt[:, 1]).norm(dim=-1)
    # Ordering 2: P1 -> G2, P2 -> G1
    d2 = (pred[:, 0] - gt[:, 2 - 1]).norm(dim=-1) + (pred[:, 1] - gt[:, 0]).norm(dim=-1)

    return torch.minimum(d1, d2).mean() / 10.0


def femur_heatmap_loss(
    logits: torch.Tensor,
    gt_pts: torch.Tensor,
    S: int,
    sigma: float = 4.0,
) -> torch.Tensor:
    """Compute JS divergence loss and BCE supervision between single heatmap and dual Gaussian.

    Parameters
    ----------
    logits : torch.Tensor
        (B, 1, S, S) raw model output logits.
    gt_pts : torch.Tensor
        (B, 2, 2) ground truth endpoints.
    S : int
        Canvas resolution.
    sigma : float
        Gaussian regularizer width.

    Returns
    -------
    torch.Tensor
        Total heatmap loss.
    """
    B = logits.shape[0]
    scaled_sigma = sigma * S / 384.0
    prob_target, unnorm_target = generate_femur_heatmap(gt_pts, S, scaled_sigma)

    p = torch.softmax(logits.float().reshape(B, -1), dim=-1)
    q = prob_target.reshape(B, -1) + 1e-12

    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + 1e-12) - torch.log(m + 1e-12))).sum(-1)
    kl_qm = (q * (torch.log(q) - torch.log(m + 1e-12))).sum(-1)
    js = (0.5 * kl_pm + 0.5 * kl_qm).mean()

    # Auxiliary BCE loss directly supervising heatmap response
    bce = F.binary_cross_entropy_with_logits(logits, unnorm_target)
    return js + bce


@torch.no_grad()
def predict_femur_batch(
    model: nn.Module,
    x: torch.Tensor,
    model_type: str = "heatmap",
    decode_type: str = "standard",
    min_dist: float = 8.0,
    tta: bool = True,
    device: str = "cpu",
    S: int = 384,
) -> torch.Tensor:
    """Predict 2 femur endpoints for a batch of images with optional horizontal flip TTA.

    Parameters
    ----------
    model : nn.Module
        Trained model (heatmap U-Net/HRNet, direct regression, integral, or boneline).
    x : torch.Tensor
        (B, 3, S, S) input image tensor.
    model_type : str
        'heatmap', 'regression', 'integral', or 'boneline'.
    decode_type : str
        'standard' (local max + center-of-mass) or 'dark' (2D Taylor expansion).
    min_dist : float
        Minimum spatial separation between detected peaks in pixels.
    tta : bool
        If True, applies horizontal flip test-time augmentation.
    device : str
        Compute device.
    S : int
        Canvas resolution.

    Returns
    -------
    torch.Tensor
        (B, 2, 2) predicted endpoints in (x, y) coordinates.
    """
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    x = x.to(device)

    if model_type == "heatmap":
        with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            lo0 = model(x)
            if tta:
                lo1 = model(torch.flip(x, dims=[3]))
                lo1_back = torch.flip(lo1, dims=[3])
                lo = 0.5 * (lo0 + lo1_back)
            else:
                lo = lo0
        if decode_type == "dark":
            return decode_femur_peaks_dark(lo, S=S, min_dist=min_dist)
        return decode_femur_peaks(lo, S=S, min_dist=min_dist)

    elif model_type == "boneline":
        with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            lo0 = model(x)
            if tta:
                lo1 = model(torch.flip(x, dims=[3]))
                lo1_back = torch.flip(lo1, dims=[3])
                lo = 0.5 * (lo0 + lo1_back)
            else:
                lo = lo0
        hm0 = lo[:, 0:1]
        return decode_femur_peaks(hm0, S=S, min_dist=min_dist)

    elif model_type == "integral":
        with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            lo0 = model(x)
            pts0 = soft_argmax_2d(lo0)
            if not tta:
                return pts0
            lo1 = model(torch.flip(x, dims=[3]))
            pts1 = soft_argmax_2d(lo1)
            pts1_back = pts1.clone()
            pts1_back[..., 0] = (x.shape[-1] - 1) - pts1_back[..., 0]
            matched_1, _ = match_endpoints(pts1_back, pts0)
            return 0.5 * (pts0 + matched_1)

    elif model_type == "regression":
        with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            out0 = model(x).view(-1, 2, 2)
            if not tta:
                return out0
            out1 = model(torch.flip(x, dims=[3])).view(-1, 2, 2)
            out1_back = out1.clone()
            out1_back[..., 0] = x.shape[-1] - 1 - out1_back[..., 0]
            matched_1, _ = match_endpoints(out1_back, out0)
            return 0.5 * (out0 + matched_1)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
