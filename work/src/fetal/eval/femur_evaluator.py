"""Evaluation utilities for Femur (FL) landmark regression models.

Computes femur length (FL) in mm, absolute errors, signed biases, percentage errors,
landmark NME (matching the AC NME and dataset paper formulation), top-10 share of
squared error, patient-level bootstrap 95% confidence intervals, and 3-seed median ensembling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from fetal import config
from fetal.models.femur_landmark import match_endpoints, predict_femur_batch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_image(img: np.ndarray, S: int) -> tuple[np.ndarray, float, int, int]:
    """Resize and pad image with zero border to S x S square canvas."""
    h, w = img.shape[:2]
    scale = S / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = np.zeros((S, S, 3), np.uint8)
    px, py = (S - nw) // 2, (S - nh) // 2
    resized = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    if resized.ndim == 2:
        resized = np.stack([resized] * 3, axis=-1)
    elif resized.shape[2] == 1:
        resized = np.repeat(resized, 3, axis=2)
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, scale, px, py


class FemurDS(Dataset):
    """PyTorch Dataset for femur images and endpoint annotations."""

    def __init__(
        self,
        df: pd.DataFrame,
        S: int = 384,
        is_train: bool = False,
        strong_aug: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.S = S
        self.is_train = is_train
        self.strong_aug = strong_aug

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        canvas, scale, px, py = letterbox_image(img, self.S)

        # Raw pixel coordinates
        p1_x, p1_y = float(row["fl_1_x"]), float(row["fl_1_y"])
        p2_x, p2_y = float(row["fl_2_x"]), float(row["fl_2_y"])

        # Map to letterbox coordinates
        p1 = np.array([p1_x * scale + px, p1_y * scale + py], dtype=np.float32)
        p2 = np.array([p2_x * scale + px, p2_y * scale + py], dtype=np.float32)
        pts = np.stack([p1, p2], axis=0)  # (2, 2) in (x, y)

        # Training augmentation
        if self.is_train:
            rng = np.random.RandomState()
            # Random horizontal flip
            if rng.rand() < 0.5:
                canvas = np.fliplr(canvas).copy()
                pts[:, 0] = (self.S - 1) - pts[:, 0]
            # Random slight rotation and scaling
            if self.strong_aug and rng.rand() < 0.5:
                ang = rng.uniform(-15, 15)
                sc = rng.uniform(0.92, 1.08)
                M = cv2.getRotationMatrix2D((self.S / 2, self.S / 2), ang, sc)
                canvas = cv2.warpAffine(canvas, M, (self.S, self.S), borderValue=(0, 0, 0))
                pts = pts @ M[:, :2].T + M[:, 2]
                pts = np.clip(pts, 0, self.S - 1)

        # Normalization
        norm_img = ((canvas.astype(np.float32) / 255.0) - MEAN) / STD
        x = torch.from_numpy(norm_img.transpose(2, 0, 1).astype(np.float32))
        gt = torch.from_numpy(pts.astype(np.float32))

        # mm per pixel in letterbox canvas
        rate = float(row["px_to_mm_rate"]) if "px_to_mm_rate" in row and pd.notna(row["px_to_mm_rate"]) else float("nan")
        mmpp = torch.tensor(rate / scale, dtype=torch.float32)

        return x, gt, mmpp, idx


def compute_femur_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mmpp: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-image FL biometry and NME metrics.

    Parameters
    ----------
    pred : torch.Tensor
        (B, 2, 2) predicted endpoints.
    gt : torch.Tensor
        (B, 2, 2) ground truth endpoints.
    mmpp : torch.Tensor
        (B,) mm per pixel in canvas space.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary of per-image metric tensors.
    """
    matched, _ = match_endpoints(pred, gt)
    d = (matched - gt).norm(dim=-1)  # (B, 2) endpoint error in px

    # FL in canvas pixels
    fl_px_true = (gt[:, 1] - gt[:, 0]).norm(dim=-1)
    fl_px_pred = (matched[:, 1] - matched[:, 0]).norm(dim=-1)

    # FL in mm
    fl_true = fl_px_true * mmpp
    fl_pred = fl_px_pred * mmpp

    fl_err = (fl_pred - fl_true).abs()
    fl_signed = fl_pred - fl_true
    pct_err = (fl_err / torch.clamp(fl_true, min=1e-4)) * 100.0

    # NME defined exactly as AC NME: mean endpoint error normalized by gt length
    nme = d.mean(-1) / torch.clamp(fl_px_true, min=1e-4)

    return {
        "fl_true": fl_true,
        "fl_pred": fl_pred,
        "fl_err": fl_err,
        "fl_signed": fl_signed,
        "fl_pct_err": pct_err,
        "nme": nme,
        "lm_px": d.mean(-1),
        "lm_mm": d.mean(-1) * mmpp,
    }


def evaluate_femur_dataframe(
    model: torch.nn.Module,
    df: pd.DataFrame,
    model_type: str = "heatmap",
    S: int = 384,
    tta: bool = True,
    bs: int = 16,
    device: str = "cpu",
    num_workers: int = 0,
) -> pd.DataFrame:
    """Evaluate model on a DataFrame and return per-image predictions and metrics."""
    ds = FemurDS(df, S=S, is_train=False, strong_aug=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    model.eval()
    records = []

    with torch.no_grad():
        for x, gt, mmpp, idx_t in dl:
            x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
            pred = predict_femur_batch(model, x, model_type=model_type, tta=tta, device=device, S=S)
            m = compute_femur_metrics(pred, gt, mmpp)

            # Map predictions back to original image space
            scale_batch = S / np.maximum(
                df.iloc[idx_t.cpu().numpy()]["image_name"].apply(lambda _: 800),  # fallback
                df.iloc[idx_t.cpu().numpy()]["image_name"].apply(lambda _: 800),
            )

            for j in range(len(idx_t)):
                orig_idx = int(idx_t[j].item())
                orig_row = df.iloc[orig_idx]
                records.append({
                    "image_name": orig_row["image_name"],
                    "patient": orig_row["patient"],
                    "source": orig_row["source"],
                    "official_split": orig_row["official_split"],
                    "px_to_mm_rate": float(orig_row["px_to_mm_rate"]),
                    "mmpp": float(mmpp[j].item()),
                    "fl_true": float(m["fl_true"][j].item()),
                    "fl_pred": float(m["fl_pred"][j].item()),
                    "fl_err": float(m["fl_err"][j].item()),
                    "fl_signed": float(m["fl_signed"][j].item()),
                    "fl_pct_err": float(m["fl_pct_err"][j].item()),
                    "nme": float(m["nme"][j].item()),
                    "lm_mm": float(m["lm_mm"][j].item()),
                    "gt_p1_x": float(gt[j, 0, 0].item()),
                    "gt_p1_y": float(gt[j, 0, 1].item()),
                    "gt_p2_x": float(gt[j, 1, 0].item()),
                    "gt_p2_y": float(gt[j, 1, 1].item()),
                    "pred_p1_x": float(pred[j, 0, 0].item()),
                    "pred_p1_y": float(pred[j, 0, 1].item()),
                    "pred_p2_x": float(pred[j, 1, 0].item()),
                    "pred_p2_y": float(pred[j, 1, 1].item()),
                })

    return pd.DataFrame(records)


def patient_bootstrap_ci(
    df: pd.DataFrame,
    metric_col: str,
    agg_func: str = "mean",
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute patient-level clustered bootstrap 95% confidence interval."""
    patients = np.array(sorted(df["patient"].unique()))
    n_pats = len(patients)
    if n_pats < 2:
        val = float(df[metric_col].mean()) if agg_func == "mean" else float(df[metric_col].median())
        return val, val

    # Group dataframe indices by patient for fast lookup
    grouped = df.groupby("patient").indices
    rng = np.random.RandomState(seed)

    boot_stats = []
    for _ in range(n_resamples):
        sample_pats = rng.choice(patients, size=n_pats, replace=True)
        sampled_indices = np.concatenate([grouped[p] for p in sample_pats])
        sample_vals = df[metric_col].iloc[sampled_indices].to_numpy()

        if agg_func == "mean":
            stat = float(np.mean(sample_vals))
        elif agg_func == "median":
            stat = float(np.median(sample_vals))
        elif agg_func == "rmse":
            stat = float(np.sqrt(np.mean(sample_vals ** 2)))
        else:
            raise ValueError(f"Unknown agg_func: {agg_func}")
        boot_stats.append(stat)

    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def summarize_femur_metrics(df: pd.DataFrame, label: str = "eval") -> dict[str, Any]:
    """Compute comprehensive metric summary with bootstrap 95% CIs and top-10 error share."""
    valid_df = df[df["fl_true"].notna() & df["fl_pred"].notna()].copy()
    n_img = len(valid_df)
    n_subj = valid_df["patient"].nunique()

    fl_err = valid_df["fl_err"].to_numpy()
    fl_signed = valid_df["fl_signed"].to_numpy()
    fl_pct = valid_df["fl_pct_err"].to_numpy()
    nme = valid_df["nme"].to_numpy()

    mae = float(np.mean(fl_err))
    rmse = float(np.sqrt(np.mean(fl_err ** 2)))
    med = float(np.median(fl_err))
    bias = float(np.mean(fl_signed))
    pct_err = float(np.mean(fl_pct))
    nme_mean = float(np.mean(nme))
    nme_sd = float(np.std(nme))

    # Top-10 share of squared error
    sq_err = fl_err ** 2
    if len(sq_err) > 0 and sq_err.sum() > 0:
        top10_share = float(np.sort(sq_err)[-10:].sum() / sq_err.sum())
    else:
        top10_share = 0.0

    mae_ci = list(patient_bootstrap_ci(valid_df, "fl_err", "mean"))
    rmse_ci = list(patient_bootstrap_ci(valid_df, "fl_err", "rmse"))
    bias_ci = list(patient_bootstrap_ci(valid_df, "fl_signed", "mean"))
    med_ci = list(patient_bootstrap_ci(valid_df, "fl_err", "median"))
    pct_ci = list(patient_bootstrap_ci(valid_df, "fl_pct_err", "mean"))

    return {
        "label": label,
        "n_images": n_img,
        "n_patients": n_subj,
        "fl_mae": mae,
        "fl_mae_ci": mae_ci,
        "fl_rmse": rmse,
        "fl_rmse_ci": rmse_ci,
        "fl_median": med,
        "fl_median_ci": med_ci,
        "fl_bias": bias,
        "fl_bias_ci": bias_ci,
        "fl_pct_error": pct_err,
        "fl_pct_error_ci": pct_ci,
        "fl_nme_mean": nme_mean,
        "fl_nme_sd": nme_sd,
        "top10_sq_err_share": top10_share,
    }


def ensemble_3seed_predictions(
    df42: pd.DataFrame,
    df43: pd.DataFrame,
    df44: pd.DataFrame,
) -> pd.DataFrame:
    """Combine 3 seeds into median ensemble by aligning endpoint ordering and taking coordinate median."""
    assert len(df42) == len(df43) == len(df44)

    p42 = torch.tensor(df42[["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))
    p43 = torch.tensor(df43[["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))
    p44 = torch.tensor(df44[["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))

    # Align seed 43 and seed 44 to seed 42 ordering
    p43_aligned, _ = match_endpoints(p43, p42)
    p44_aligned, _ = match_endpoints(p44, p42)

    stacked = torch.stack([p42, p43_aligned, p44_aligned], dim=0)  # (3, N, 2, 2)
    ens_pts = torch.median(stacked, dim=0).values  # (N, 2, 2)

    gt_pts = torch.tensor(df42[["gt_p1_x", "gt_p1_y", "gt_p2_x", "gt_p2_y"]].to_numpy().reshape(-1, 2, 2))
    mmpp = torch.tensor(df42["mmpp"].to_numpy())

    m = compute_femur_metrics(ens_pts, gt_pts, mmpp)

    res_df = df42.copy()
    res_df["pred_p1_x"] = ens_pts[:, 0, 0].numpy()
    res_df["pred_p1_y"] = ens_pts[:, 0, 1].numpy()
    res_df["pred_p2_x"] = ens_pts[:, 1, 0].numpy()
    res_df["pred_p2_y"] = ens_pts[:, 1, 1].numpy()
    res_df["fl_pred"] = m["fl_pred"].numpy()
    res_df["fl_err"] = m["fl_err"].numpy()
    res_df["fl_signed"] = m["fl_signed"].numpy()
    res_df["fl_pct_err"] = m["fl_pct_err"].numpy()
    res_df["nme"] = m["nme"].numpy()
    res_df["lm_mm"] = m["lm_mm"].numpy()

    return res_df
