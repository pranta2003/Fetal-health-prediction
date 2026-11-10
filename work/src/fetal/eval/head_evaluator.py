"""Evaluation utilities for Head landmark regression models.

Computes BPD and OFD caliper errors (mm), signed biases, NME (same formula as AC NME),
and Ramanujan approximate HC reference comparison. Supports 3-seed median ensembling
and patient-level bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fetal import config
from fetal.data.head_dataset import HeadDS, ramanujan_hc
from fetal.models.head_landmark import decode_head

PERMS = []
for sp in (0, 1):
    for s0 in (0, 1):
        for s1 in (0, 1):
            a = [1, 0] if s0 else [0, 1]
            b = [3, 2] if s1 else [2, 3]
            PERMS.append(b + a if sp else a + b)
PERMS_T = torch.tensor(PERMS, dtype=torch.long)


def match(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find permutation of pred (B, 4, 2) that minimizes distance to gt (B, 4, 2)."""
    B = pred.shape[0]
    P = PERMS_T.to(pred.device)
    pp = pred[:, P]  # (B, 8, 4, 2)
    d = (pp - gt[:, None]).norm(dim=-1).mean(-1)  # (B, 8)
    idx = d.argmin(1).detach()
    return pp[torch.arange(B, device=pred.device), idx], P[idx]


def lengths_mm(pts: torch.Tensor, mmpp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate OFD (pair 1: pts 0-1) and BPD (pair 2: pts 2-3) lengths in mm."""
    ofd_mm = (pts[:, 1] - pts[:, 0]).norm(dim=-1) * mmpp
    bpd_mm = (pts[:, 3] - pts[:, 2]).norm(dim=-1) * mmpp
    return ofd_mm, bpd_mm


@torch.no_grad()
def predict_batch(model: torch.nn.Module, x: torch.Tensor, tta: bool = True, device: str = "cpu") -> torch.Tensor:
    """Predict 4 keypoints for batch with optional horizontal flip TTA."""
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
        lo0 = model(x)
        lo1 = model(torch.flip(x, dims=[3])) if tta else None
    p0, _ = decode_head(lo0)
    if not tta:
        return p0
    p1, _ = decode_head(lo1)
    p1 = p1.clone()
    p1[..., 0] = x.shape[-1] - 1 - p1[..., 0]
    m1, _ = match(p1, p0)
    return (p0 + m1) / 2.0


def metrics_rows(pred: torch.Tensor, gt: torch.Tensor, mmpp: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute per-image metrics matching AC NME and Ramanujan HC formulations."""
    pm, _ = match(pred, gt)
    d = (pm - gt).norm(dim=-1)  # (B, 4) in px

    # OFD is pair 1 (pts 0-1), BPD is pair 2 (pts 2-3)
    tl = (gt[:, 1] - gt[:, 0]).norm(dim=-1)  # OFD gt length in px
    al = (gt[:, 3] - gt[:, 2]).norm(dim=-1)  # BPD gt length in px

    ofd_p, bpd_p = lengths_mm(pm, mmpp)
    ofd_t, bpd_t = lengths_mm(gt, mmpp)

    hc_p = ramanujan_hc(ofd_p / 2.0, bpd_p / 2.0)
    hc_t = ramanujan_hc(ofd_t / 2.0, bpd_t / 2.0)

    nme_ofd = d[:, :2].mean(-1) / tl
    nme_bpd = d[:, 2:].mean(-1) / al
    nme = 0.5 * (nme_ofd + nme_bpd)

    return {
        "ofd_true": ofd_t,
        "ofd_pred": ofd_p,
        "ofd_err": (ofd_p - ofd_t).abs(),
        "ofd_signed": ofd_p - ofd_t,
        "bpd_true": bpd_t,
        "bpd_pred": bpd_p,
        "bpd_err": (bpd_p - bpd_t).abs(),
        "bpd_signed": bpd_p - bpd_t,
        "hc_true": hc_t,
        "hc_pred": hc_p,
        "hc_err": (hc_p - hc_t).abs(),
        "hc_signed": hc_p - hc_t,
        "nme_ofd": nme_ofd,
        "nme_bpd": nme_bpd,
        "nme": nme,
        "lm_mm": d.mean(-1) * mmpp,
    }


def run_eval(
    model: torch.nn.Module,
    df: pd.DataFrame,
    S: int = 384,
    tta: bool = True,
    bs: int = 8,
    num_workers: int = 0,
    device: str = "cpu",
) -> pd.DataFrame:
    """Evaluate a single model on a dataset."""
    ds = HeadDS(df, S=S, train=False, strong=False, inpaint=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    model.eval()
    rows = []
    for x, gt, mmpp, idx in dl:
        x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
        pred = predict_batch(model, x, tta=tta, device=device)
        m = metrics_rows(pred, gt, mmpp)
        for j in range(len(idx)):
            r = df.iloc[int(idx[j])]
            row_dict = {k: float(v[j].cpu().item()) for k, v in m.items()}
            row_dict.update({
                "image_name": r["image_name"],
                "patient": r["patient"],
                "source": r["source"],
                "official_split": r.get("official_split", ""),
                "px_to_mm_rate": float(r["px_to_mm_rate"]),
            })
            rows.append(row_dict)
    return pd.DataFrame(rows)


@torch.no_grad()
def eval_ensemble(
    models: list[torch.nn.Module],
    df: pd.DataFrame,
    S: int = 384,
    bs: int = 8,
    num_workers: int = 0,
    device: str = "cpu",
) -> pd.DataFrame:
    """Evaluate median ensemble of multiple models with flip TTA."""
    for m in models:
        m.eval()
    ds = HeadDS(df, S=S, train=False, strong=False, inpaint=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    rows = []
    for x, gt, mmpp, idx in dl:
        x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
        preds = [predict_batch(m, x, tta=True, device=device) for m in models]
        ref = preds[0]
        stack = torch.stack([ref] + [match(p, ref)[0] for p in preds[1:]])
        pred = stack.median(0).values
        m_ = metrics_rows(pred, gt, mmpp)
        for j in range(len(idx)):
            r = df.iloc[int(idx[j])]
            row_dict = {k: float(v[j].cpu().item()) for k, v in m_.items()}
            row_dict.update({
                "image_name": r["image_name"],
                "patient": r["patient"],
                "source": r["source"],
                "official_split": r.get("official_split", ""),
                "px_to_mm_rate": float(r["px_to_mm_rate"]),
            })
            rows.append(row_dict)
    return pd.DataFrame(rows)


def patient_level_bootstrap_ci(
    df: pd.DataFrame,
    val_col: str,
    subject_col: str = "patient",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Compute 1000-iteration patient-level bootstrap 95% confidence intervals."""
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    unique_subjs = np.array(df[subject_col].unique())
    n_subjs = len(unique_subjs)
    if n_subjs <= 1:
        val = float(df[val_col].mean())
        return {"mean": [val, val], "mae": [val, val], "bias": [val, val]}

    subj_indices = {s: df.index[df[subject_col] == s].to_numpy() for s in unique_subjs}

    boot_mean = []
    boot_mae = []
    boot_bias = []

    vals = df[val_col].to_numpy()
    abs_vals = np.abs(vals)

    for _ in range(n_boot):
        sampled_subjs = rng.choice(unique_subjs, size=n_subjs, replace=True)
        idx_list = [subj_indices[s] for s in sampled_subjs]
        all_idx = np.concatenate(idx_list)
        b_vals = vals[all_idx]
        b_abs = abs_vals[all_idx]

        boot_mean.append(float(np.mean(b_vals)))
        boot_mae.append(float(np.mean(b_abs)))
        boot_bias.append(float(np.mean(b_vals)))

    return {
        "mean": [float(np.percentile(boot_mean, 2.5)), float(np.percentile(boot_mean, 97.5))],
        "mae": [float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5))],
        "bias": [float(np.percentile(boot_bias, 2.5)), float(np.percentile(boot_bias, 97.5))],
    }


def compute_cohort_metrics(df: pd.DataFrame, group_name: str, subject_col: str = "patient") -> dict:
    """Compute summary metrics and bootstrap 95% CIs for a cohort."""
    n_images = len(df)
    n_patients = df[subject_col].nunique()

    # Absolute errors & biases
    bpd_mae = float(df["bpd_err"].mean())
    bpd_bias = float(df["bpd_signed"].mean())
    bpd_ci = patient_level_bootstrap_ci(df, "bpd_signed", subject_col=subject_col)

    ofd_mae = float(df["ofd_err"].mean())
    ofd_bias = float(df["ofd_signed"].mean())
    ofd_ci = patient_level_bootstrap_ci(df, "ofd_signed", subject_col=subject_col)

    hc_mae = float(df["hc_err"].mean())
    hc_bias = float(df["hc_signed"].mean())
    hc_ci = patient_level_bootstrap_ci(df, "hc_signed", subject_col=subject_col)

    bpd_nme_mean = float(df["nme_bpd"].mean())
    bpd_nme_sd = float(df["nme_bpd"].std())
    ofd_nme_mean = float(df["nme_ofd"].mean())
    ofd_nme_sd = float(df["nme_ofd"].std())
    nme_mean = float(df["nme"].mean())
    nme_sd = float(df["nme"].std())

    return {
        "group": group_name,
        "n_images": n_images,
        "n_patients": n_patients,
        "bpd_mae": bpd_mae,
        "bpd_mae_ci": bpd_ci["mae"],
        "bpd_bias": bpd_bias,
        "bpd_bias_ci": bpd_ci["bias"],
        "ofd_mae": ofd_mae,
        "ofd_mae_ci": ofd_ci["mae"],
        "ofd_bias": ofd_bias,
        "ofd_bias_ci": ofd_ci["bias"],
        "hc_mae": hc_mae,
        "hc_mae_ci": hc_ci["mae"],
        "hc_bias": hc_bias,
        "hc_bias_ci": hc_ci["bias"],
        "bpd_nme_mean": bpd_nme_mean,
        "bpd_nme_sd": bpd_nme_sd,
        "ofd_nme_mean": ofd_nme_mean,
        "ofd_nme_sd": ofd_nme_sd,
        "nme_mean": nme_mean,
        "nme_sd": nme_sd,
    }
