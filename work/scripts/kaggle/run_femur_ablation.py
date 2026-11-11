"""Self-contained Kaggle runner for Femur Model Ablation Study (Rows A, B, C; seeds 42, 43, 44).

Trains and evaluates 9 models on Kaggle GPU T4:
  Row A: U-Net EfficientNet-B4, 384 letterbox, 1-channel dual-peak heatmap, flip TTA
  Row B: ConvNeXt-Tiny, 384 letterbox, direct coordinate regression, order-invariant loss
  Row C: U-Net EfficientNet-B4, 512 letterbox, 1-channel dual-peak heatmap, flip TTA

Computes per-seed and 3-seed median ensemble metrics on 348 test images with patient bootstrap 95% CIs,
generates markdown table and json, and executes benchmark notebook via nbconvert.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 1. Install SMP if missing
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Installing segmentation-models-pytorch==0.5.0...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-warn-script-location", "segmentation-models-pytorch==0.5.0"])
    import segmentation_models_pytorch as smp

import torchvision.models as tvm

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIGMA = 4.0


# ---------------------------------------------------------------------------
# Data Loading and Splitting
# ---------------------------------------------------------------------------
def patient_id_from_name(name: str, src: str) -> str:
    name_str = str(name)
    if src == "FP":
        m = re.search(r"(Patient\d+)", name_str)
        if not m:
            raise ValueError(f"Could not extract FP patient ID from {name_str}")
        return f"FP_{m.group(1)}"
    elif src == "UCL":
        m = re.match(r"(\d+)_", name_str)
        if not m:
            raise ValueError(f"Could not extract UCL patient ID from {name_str}")
        return f"UCL_{m.group(1)}"
    raise ValueError(f"Unknown source: {src}")


def load_femur_source(src: str, ann_dir: Path, img_dir: Path) -> pd.DataFrame:
    ann_path = ann_dir / "Femur.csv"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing {ann_path}")
    df = pd.read_csv(ann_path)
    if src == "UCL":
        df = df[~df["image_name"].str.startswith("125_")].copy()
    df = df.reset_index(drop=True)

    df["source"] = src
    df["image_path"] = df["image_name"].apply(lambda f: str(img_dir / f))
    df["patient"] = df["image_name"].apply(lambda f: patient_id_from_name(f, src))

    te_path = ann_dir / "Femur_Test.csv"
    tr_path = ann_dir / "Femur_Train.csv"
    if te_path.is_file():
        te_names = set(pd.read_csv(te_path)["image_name"].astype(str))
        df["official_split"] = df["image_name"].astype(str).apply(lambda f: "test" if f in te_names else "train")
    elif tr_path.is_file():
        tr_names = set(pd.read_csv(tr_path)["image_name"].astype(str))
        df["official_split"] = df["image_name"].astype(str).apply(lambda f: "train" if f in tr_names else "test")
    else:
        df["official_split"] = df["Split"].astype(str).str.lower()

    df["fl_length_mm"] = [
        float(np.hypot(r.fl_2_x - r.fl_1_x, r.fl_2_y - r.fl_1_y) * r.px_to_mm_rate)
        if pd.notna(r.px_to_mm_rate) else float("nan")
        for r in df.itertuples()
    ]
    cols = [
        "image_name", "image_path", "source", "patient", "official_split",
        "px_to_mm_rate", "fl_length_mm", "fl_1_x", "fl_1_y", "fl_2_x", "fl_2_y"
    ]
    return df[cols]


def build_femur_data_from_root(data_root: Path) -> pd.DataFrame:
    fp = load_femur_source("FP", data_root / "FP" / "annotations", data_root / "FP" / "images" / "Femur")
    ucl = load_femur_source("UCL", data_root / "UCL" / "annotations" / "px_to_mm", data_root / "UCL" / "images" / "Femur")
    df = pd.concat([fp, ucl], ignore_index=True)
    df = df[df["px_to_mm_rate"].notna()].reset_index(drop=True)
    return df


def split_femur_data(df: pd.DataFrame, dev_fraction: float = 0.12, seed: int = 0):
    train_all = df[df["official_split"] == "train"]
    pats = np.array(sorted(train_all["patient"].unique()))
    rng = np.random.RandomState(seed)
    dev_count = max(1, int(round(dev_fraction * len(pats))))
    dev_p = set(rng.permutation(pats)[:dev_count])

    tr = train_all[~train_all["patient"].isin(dev_p)].reset_index(drop=True)
    dv = train_all[train_all["patient"].isin(dev_p)].reset_index(drop=True)
    te = df[df["official_split"] == "test"].reset_index(drop=True)

    assert not (set(tr["patient"]) & set(dv["patient"])), "Train/dev patient overlap!"
    assert not (set(tr["patient"]) & set(te["patient"])), "Train/test patient overlap!"
    assert not (set(dv["patient"]) & set(te["patient"])), "Dev/test patient overlap!"
    return tr, dv, te


# ---------------------------------------------------------------------------
# Dataset & Letterboxing
# ---------------------------------------------------------------------------
def letterbox_image(img: np.ndarray, S: int):
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


class FemurKaggleDS(Dataset):
    def __init__(self, df: pd.DataFrame, S: int = 384, is_train: bool = False, strong_aug: bool = False):
        self.df = df.reset_index(drop=True)
        self.S = S
        self.is_train = is_train
        self.strong_aug = strong_aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Missing {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        canvas, scale, px, py = letterbox_image(img, self.S)

        p1_x, p1_y = float(row["fl_1_x"]), float(row["fl_1_y"])
        p2_x, p2_y = float(row["fl_2_x"]), float(row["fl_2_y"])

        p1 = np.array([p1_x * scale + px, p1_y * scale + py], dtype=np.float32)
        p2 = np.array([p2_x * scale + px, p2_y * scale + py], dtype=np.float32)
        pts = np.stack([p1, p2], axis=0)

        if self.is_train:
            rng = np.random.RandomState()
            if rng.rand() < 0.5:
                canvas = np.fliplr(canvas).copy()
                pts[:, 0] = (self.S - 1) - pts[:, 0]
            if self.strong_aug and rng.rand() < 0.5:
                ang = rng.uniform(-15, 15)
                sc = rng.uniform(0.92, 1.08)
                M = cv2.getRotationMatrix2D((self.S / 2, self.S / 2), ang, sc)
                canvas = cv2.warpAffine(canvas, M, (self.S, self.S), borderValue=(0, 0, 0))
                pts = pts @ M[:, :2].T + M[:, 2]
                pts = np.clip(pts, 0, self.S - 1)

        norm_img = ((canvas.astype(np.float32) / 255.0) - MEAN) / STD
        x = torch.from_numpy(norm_img.transpose(2, 0, 1).astype(np.float32))
        gt = torch.from_numpy(pts.astype(np.float32))
        rate = float(row["px_to_mm_rate"])
        mmpp = torch.tensor(rate / scale, dtype=torch.float32)
        return x, gt, mmpp, idx


# ---------------------------------------------------------------------------
# Models, Peak Decoder & Losses
# ---------------------------------------------------------------------------
def build_femur_heatmap_model(encoder_weights="imagenet"):
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )


def build_femur_regression_model(pretrained=True):
    w = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = tvm.convnext_tiny(weights=w)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 4)
    return model


def generate_femur_heatmap(pts: torch.Tensor, S: int, sigma: float = 4.0):
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


def decode_femur_peaks(heatmap: torch.Tensor, S: int | None = None, k: int = 9, r: int = 8, min_dist: float = 8.0):
    B, C, H, W = heatmap.shape
    if S is None:
        S = H

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


def match_endpoints(pred: torch.Tensor, gt: torch.Tensor):
    d0 = (pred - gt).norm(dim=-1).sum(-1)
    pred_swapped = torch.stack([pred[:, 1], pred[:, 0]], dim=1)
    d1 = (pred_swapped - gt).norm(dim=-1).sum(-1)
    use_swap = (d1 < d0)
    matched = torch.where(use_swap[:, None, None], pred_swapped, pred)
    order = torch.where(use_swap[:, None], torch.tensor([1, 0], device=pred.device), torch.tensor([0, 1], device=pred.device))
    return matched, order


def femur_order_invariant_loss(pred_pts: torch.Tensor, gt_pts: torch.Tensor):
    if pred_pts.dim() == 2:
        pred = pred_pts.view(-1, 2, 2)
    else:
        pred = pred_pts
    if gt_pts.dim() == 2:
        gt = gt_pts.view(-1, 2, 2)
    else:
        gt = gt_pts

    d1 = (pred[:, 0] - gt[:, 0]).norm(dim=-1) + (pred[:, 1] - gt[:, 1]).norm(dim=-1)
    d2 = (pred[:, 0] - gt[:, 1]).norm(dim=-1) + (pred[:, 1] - gt[:, 0]).norm(dim=-1)
    return torch.minimum(d1, d2).mean() / 10.0


def femur_heatmap_loss(logits: torch.Tensor, gt_pts: torch.Tensor, S: int, sigma: float = 4.0):
    B = logits.shape[0]
    scaled_sigma = sigma * S / 384.0
    prob_target, unnorm_target = generate_femur_heatmap(gt_pts, S, scaled_sigma)

    p = torch.softmax(logits.float().reshape(B, -1), dim=-1)
    q = prob_target.reshape(B, -1) + 1e-12

    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + 1e-12) - torch.log(m + 1e-12))).sum(-1)
    kl_qm = (q * (torch.log(q) - torch.log(m + 1e-12))).sum(-1)
    js = (0.5 * kl_pm + 0.5 * kl_qm).mean()
    bce = F.binary_cross_entropy_with_logits(logits, unnorm_target)
    return js + bce


@torch.no_grad()
def predict_batch(model, x, model_type="heatmap", tta=True, device="cpu", S=384):
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
        return decode_femur_peaks(lo, S=S)

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


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, mmpp: torch.Tensor):
    matched, _ = match_endpoints(pred, gt)
    d = (matched - gt).norm(dim=-1)

    fl_px_true = (gt[:, 1] - gt[:, 0]).norm(dim=-1)
    fl_px_pred = (matched[:, 1] - matched[:, 0]).norm(dim=-1)

    fl_true = fl_px_true * mmpp
    fl_pred = fl_px_pred * mmpp

    fl_err = (fl_pred - fl_true).abs()
    fl_signed = fl_pred - fl_true
    pct_err = (fl_err / torch.clamp(fl_true, min=1e-4)) * 100.0
    nme = d.mean(-1) / torch.clamp(fl_px_true, min=1e-4)

    return {
        "fl_true": fl_true, "fl_pred": fl_pred, "fl_err": fl_err,
        "fl_signed": fl_signed, "fl_pct_err": pct_err, "nme": nme,
        "lm_mm": d.mean(-1) * mmpp,
    }


def evaluate_dataframe(model, df, model_type="heatmap", S=384, tta=True, bs=16, device="cpu"):
    ds = FemurKaggleDS(df, S=S, is_train=False, strong_aug=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    model.eval()
    records = []

    with torch.no_grad():
        for x, gt, mmpp, idx_t in dl:
            x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
            pred = predict_batch(model, x, model_type=model_type, tta=tta, device=device, S=S)
            m = compute_metrics(pred, gt, mmpp)

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


def patient_bootstrap_ci(df: pd.DataFrame, metric_col: str, agg_func: str = "mean", n_resamples: int = 1000, seed: int = 42):
    patients = np.array(sorted(df["patient"].unique()))
    n_pats = len(patients)
    if n_pats < 2:
        val = float(df[metric_col].mean()) if agg_func == "mean" else float(df[metric_col].median())
        return val, val

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


def summarize_metrics(df: pd.DataFrame, label: str = "eval"):
    valid_df = df[df["fl_true"].notna() & df["fl_pred"].notna()].copy()
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

    sq_err = fl_err ** 2
    top10_share = float(np.sort(sq_err)[-10:].sum() / sq_err.sum()) if len(sq_err) > 0 and sq_err.sum() > 0 else 0.0

    mae_ci = list(patient_bootstrap_ci(valid_df, "fl_err", "mean"))
    rmse_ci = list(patient_bootstrap_ci(valid_df, "fl_err", "rmse"))
    bias_ci = list(patient_bootstrap_ci(valid_df, "fl_signed", "mean"))
    med_ci = list(patient_bootstrap_ci(valid_df, "fl_err", "median"))
    pct_ci = list(patient_bootstrap_ci(valid_df, "fl_pct_err", "mean"))

    return {
        "label": label,
        "n_images": len(valid_df),
        "n_patients": valid_df["patient"].nunique(),
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


def ensemble_3seeds(df42: pd.DataFrame, df43: pd.DataFrame, df44: pd.DataFrame):
    assert len(df42) == len(df43) == len(df44)

    p42 = torch.tensor(df42[["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))
    p43 = torch.tensor(df43[["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))
    p44 = torch.tensor(df44[["pred_p1_x", "pred_p1_y", "pred_p2_x", "pred_p2_y"]].to_numpy().reshape(-1, 2, 2))

    p43_aligned, _ = match_endpoints(p43, p42)
    p44_aligned, _ = match_endpoints(p44, p42)

    stacked = torch.stack([p42, p43_aligned, p44_aligned], dim=0)
    ens_pts = torch.median(stacked, dim=0).values

    gt_pts = torch.tensor(df42[["gt_p1_x", "gt_p1_y", "gt_p2_x", "gt_p2_y"]].to_numpy().reshape(-1, 2, 2))
    mmpp = torch.tensor(df42["mmpp"].to_numpy())

    m = compute_metrics(ens_pts, gt_pts, mmpp)

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


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def train_single_model(
    row_key: str,
    seed: int,
    tr_df: pd.DataFrame,
    dv_df: pd.DataFrame,
    te_df: pd.DataFrame,
    S: int,
    model_type: str,
    device: str = "cuda",
    max_epochs: int = 35,
    patience: int = 8,
    batch_size: int = 8,
    lr: float = 2e-4,
    out_dir: Path = Path("ckpts"),
):
    print(f"\n>>> Starting {row_key} (Seed {seed}, S={S}, model={model_type})")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / f"{row_key}_s{seed}_best.pt"

    if model_type == "heatmap":
        model = build_femur_heatmap_model(encoder_weights="imagenet").to(device)
    elif model_type == "regression":
        model = build_femur_regression_model(pretrained=True).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler(device, enabled=("cuda" in str(device)))

    ds_tr = FemurKaggleDS(tr_df, S=S, is_train=True, strong_aug=True)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    best_val_score = float("inf")
    bad_epochs = 0
    t0_train = time.time()
    epochs_trained = 0

    for ep in range(max_epochs):
        t0_ep = time.time()
        model.train()
        train_loss = 0.0

        for x, gt, mmpp, _ in dl_tr:
            x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=("cuda" in str(device))):
                if model_type == "heatmap":
                    logits = model(x)
                    loss = femur_heatmap_loss(logits, gt, S=S)
                else:
                    pred_coords = model(x).view(-1, 2, 2)
                    loss = femur_order_invariant_loss(pred_coords, gt)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()

        epochs_trained = ep + 1
        avg_loss = train_loss / len(dl_tr)

        # Validation on the 65 validation images only
        val_res = evaluate_dataframe(model, dv_df, model_type=model_type, S=S, tta=False, device=device)
        val_sum = summarize_metrics(val_res, "val")
        val_score = val_sum["fl_mae"]

        sched.step(val_score)

        improved = val_score < best_val_score
        if improved:
            best_val_score = val_score
            bad_epochs = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            bad_epochs += 1

        ep_sec = time.time() - t0_ep
        star = " *" if improved else ""
        print(f"[{row_key} s{seed}] Ep {ep+1:02d}/{max_epochs} | loss: {avg_loss:.4f} | val MAE: {val_score:.3f} mm, NME: {val_sum['fl_nme_mean']:.4f} | {ep_sec:.1f}s{star}")

        if bad_epochs >= patience:
            print(f"[{row_key} s{seed}] Early stopping triggered at epoch {ep+1}")
            break

    train_time_sec = time.time() - t0_train
    print(f"[{row_key} s{seed}] Training finished in {train_time_sec:.1f}s ({epochs_trained} epochs). Loading best checkpoint...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    # Final evaluation on 348 test images with flip TTA
    test_res_df = evaluate_dataframe(model, te_df, model_type=model_type, S=S, tta=True, device=device)
    metrics_all = summarize_metrics(test_res_df, "test_all")
    metrics_fp = summarize_metrics(test_res_df[test_res_df["source"] == "FP"], "test_fp")
    metrics_ucl = summarize_metrics(test_res_df[test_res_df["source"] == "UCL"], "test_ucl")

    metrics_all["epochs"] = epochs_trained
    metrics_all["train_time_sec"] = train_time_sec
    metrics_all["fp_mae"] = metrics_fp["fl_mae"]
    metrics_all["ucl_mae"] = metrics_ucl["fl_mae"]

    return model, test_res_df, metrics_all, metrics_fp, metrics_ucl


# ---------------------------------------------------------------------------
# Main Execution Function
# ---------------------------------------------------------------------------
# Format Markdown Table
# ---------------------------------------------------------------------------
def format_ablation_markdown_table(results: dict) -> str:
    """Format comprehensive markdown table comparing rows A, B, and C."""
    lines = [
        "# Femur Model Ablation Study (FP + UCL Femur Test Set, N=348 images / 266 patients)",
        "",
        "> Reference FL: Euclidean distance between annotated endpoints * px_to_mm_rate.",
        "> Protocol: Train on 451 images, early stop on 65 validation images (0 patient overlap).",
        "> Evaluation: 348 official test images (323 FP / 261 patients, 25 UCL / 5 patients [tiny]).",
        "> 95% CIs: Patient-level bootstrap with 1,000 resamples.",
        "> NME: Mean Euclidean landmark error normalized by ground-truth femur length.",
        "",
        "| Approach | Row / Model | Resolution | Epochs | Train Time (s) | FL MAE [95% CI] (mm) | FL RMSE [95% CI] (mm) | FL Median [95% CI] (mm) | % Error [95% CI] | FL NME (mean+-SD) | Top-10 SqErr Share (%) | FP MAE (mm) | UCL MAE (mm, tiny) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    row_names = [
        ("Row A (U-Net EffNet-B4 1-ch Heatmap, flip TTA)", "row_A"),
        ("Row B (ConvNeXt-Tiny Direct Reg, Order-Inv)", "row_B"),
        ("Row C (U-Net EffNet-B4 512px Heatmap, flip TTA)", "row_C"),
    ]

    for display_name, row_key in row_names:
        if row_key not in results:
            continue
        row_data = results[row_key]
        res_str = f"{row_data['resolution']} px"
        m_sd = row_data["seeds_mean_sd"]
        ens = row_data["ensemble_all"]
        ens_fp = row_data["ensemble_fp"]
        ens_ucl = row_data["ensemble_ucl"]

        # Seeds Mean +- SD row
        ep_str = f"{m_sd['epochs_mean']:.1f} +- {m_sd['epochs_sd']:.1f}"
        time_str = f"{m_sd['time_mean']:.1f} +- {m_sd['time_sd']:.1f}"
        mae_ms = f"{m_sd['fl_mae_mean']:.3f} +- {m_sd['fl_mae_sd']:.3f}"
        rmse_ms = f"{m_sd['fl_rmse_mean']:.3f} +- {m_sd['fl_rmse_sd']:.3f}"
        med_ms = f"{m_sd['fl_med_mean']:.3f} +- {m_sd['fl_med_sd']:.3f}"
        pct_ms = f"{m_sd['fl_pct_mean']:.2f}% +- {m_sd['fl_pct_sd']:.2f}%"
        nme_ms = f"{m_sd['fl_nme_mean']:.3f} +- {m_sd['fl_nme_sd']:.3f}"
        top10_ms = f"{m_sd['top10_mean']*100:.1f}% +- {m_sd['top10_sd']*100:.1f}%"
        fp_ms = f"{m_sd['fp_mae_mean']:.3f}"
        ucl_ms = f"{m_sd['ucl_mae_mean']:.3f}"

        lines.append(
            f"| {display_name} | *Seeds Mean +- SD* | {res_str} | {ep_str} | {time_str} | "
            f"{mae_ms} | {rmse_ms} | {med_ms} | {pct_ms} | {nme_ms} | {top10_ms} | {fp_ms} | {ucl_ms} |"
        )

        # 3-Seed Median Ensemble row
        e_mae = f"**{ens['fl_mae']:.3f}** [{ens['fl_mae_ci'][0]:.3f}, {ens['fl_mae_ci'][1]:.3f}]"
        e_rmse = f"**{ens['fl_rmse']:.3f}** [{ens['fl_rmse_ci'][0]:.3f}, {ens['fl_rmse_ci'][1]:.3f}]"
        e_med = f"**{ens['fl_median']:.3f}** [{ens['fl_median_ci'][0]:.3f}, {ens['fl_median_ci'][1]:.3f}]"
        e_pct = f"**{ens['fl_pct_error']:.2f}%** [{ens['fl_pct_error_ci'][0]:.2f}%, {ens['fl_pct_error_ci'][1]:.2f}%]"
        e_nme = f"**{ens['fl_nme_mean']:.3f} +- {ens['fl_nme_sd']:.3f}**"
        e_top10 = f"**{ens['top10_sq_err_share']*100:.1f}%**"
        e_fp = f"**{ens_fp['fl_mae']:.3f}**"
        e_ucl = f"**{ens_ucl['fl_mae']:.3f}**"

        lines.append(
            f"| {display_name} | **3-Seed Median Ens** | {res_str} | - | - | "
            f"{e_mae} | {e_rmse} | {e_med} | {e_pct} | {e_nme} | {e_top10} | {e_fp} | {e_ucl} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def run_ablation_experiment():
    print("=" * 80)
    print("FEMUR MODEL ABLATION EXPERIMENT (Rows A, B, C; Seeds 42, 43, 44)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Locate dataset dynamically
    raw_root = None
    if Path("/kaggle/input").is_dir():
        input_dirs = list(Path("/kaggle/input").iterdir())
        print(f"Listing /kaggle/input: {[p.name for p in input_dirs]}")
        # Search for femur_data.zip
        for p in Path("/kaggle/input").rglob("*.zip"):
            if "femur" in p.name.lower():
                raw_root = p.parent
                print(f"Found zip at: {p}")
                break
        # Search for uncompressed FP/Femur.csv where root has FP and UCL
        if raw_root is None:
            for p in Path("/kaggle/input").rglob("Femur.csv"):
                for parent in p.parents:
                    if (parent / "FP").is_dir() and (parent / "UCL").is_dir():
                        raw_root = parent
                        print(f"Found dataset root at: {raw_root}")
                        break
                if raw_root is not None:
                    break

    if raw_root is None:
        for cand in [Path("C:/fetal-data"), Path("../data"), Path("/tmp/femur_data")]:
            if cand.is_dir():
                raw_root = cand
                break

    if raw_root is None:
        raise FileNotFoundError(f"Could not find femur dataset in /kaggle/input or local paths.")

    # If zipped, unpack
    zip_candidates = list(raw_root.glob("*.zip"))
    work_data = Path("/tmp/femur_data")
    if zip_candidates:
        print(f"Unpacking {zip_candidates[0]} to {work_data}...")
        work_data.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_candidates[0], "r") as zf:
            zf.extractall(work_data)
        data_dir = work_data
    elif (raw_root / "FP").is_dir():
        data_dir = raw_root
    else:
        raise FileNotFoundError(f"No FP dataset folder or zip found in {raw_root}")

    print(f"Using data directory: {data_dir}")

    # Build dataset
    df = build_femur_data_from_root(data_dir)
    print(f"Total calibrated femur images: {len(df)}")
    assert len(df) == 864, f"Expected 864 calibrated femur images, got {len(df)}"

    tr_df, dv_df, te_df = split_femur_data(df, dev_fraction=0.12, seed=0)
    print(f"Splits: Train = {len(tr_df)} ({tr_df['patient'].nunique()} pats), Val = {len(dv_df)} ({dv_df['patient'].nunique()} pats), Test = {len(te_df)} ({te_df['patient'].nunique()} pats)")
    assert len(tr_df) == 451
    assert len(dv_df) == 65
    assert len(te_df) == 348

    out_dir = Path("/kaggle/working/out") if Path("/kaggle").is_dir() else Path("docs/results/femur_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("/kaggle/working/ckpts") if Path("/kaggle").is_dir() else Path("ckpts_femur")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows_config = [
        ("row_A", "Row A (EffNet-B4 1-ch Heatmap, 384px)", 384, "heatmap"),
        ("row_B", "Row B (ConvNeXt-Tiny Direct Reg, 384px)", 384, "regression"),
        ("row_C", "Row C (EffNet-B4 1-ch Heatmap, 512px)", 512, "heatmap"),
    ]

    seeds = [42, 43, 44]
    all_results = {}
    pred_dfs_by_row = {}

    for row_key, row_name, S, model_type in rows_config:
        print("\n" + "#" * 70)
        print(f"STARTING {row_name}")
        print("#" * 70)

        seed_dfs = {}
        seed_metrics = {}

        for seed in seeds:
            _, test_pred_df, m_all, m_fp, m_ucl = train_single_model(
                row_key=row_key,
                seed=seed,
                tr_df=tr_df,
                dv_df=dv_df,
                te_df=te_df,
                S=S,
                model_type=model_type,
                device=device,
                max_epochs=35,
                patience=8,
                batch_size=8,
                lr=2e-4,
                out_dir=ckpt_dir,
            )
            seed_dfs[seed] = test_pred_df
            seed_metrics[f"seed_{seed}"] = {
                "all": m_all,
                "fp": m_fp,
                "ucl": m_ucl,
            }

        # Compute 3-seed Mean +- SD
        mae_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_mae"] for s in seeds]
        rmse_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_rmse"] for s in seeds]
        med_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_median"] for s in seeds]
        pct_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_pct_error"] for s in seeds]
        nme_vals = [seed_metrics[f"seed_{s}"]["all"]["fl_nme_mean"] for s in seeds]
        top10_vals = [seed_metrics[f"seed_{s}"]["all"]["top10_sq_err_share"] for s in seeds]
        ep_vals = [seed_metrics[f"seed_{s}"]["all"]["epochs"] for s in seeds]
        time_vals = [seed_metrics[f"seed_{s}"]["all"]["train_time_sec"] for s in seeds]
        fp_maes = [seed_metrics[f"seed_{s}"]["fp"]["fl_mae"] for s in seeds]
        ucl_maes = [seed_metrics[f"seed_{s}"]["ucl"]["fl_mae"] for s in seeds]

        mean_sd_stats = {
            "fl_mae_mean": float(np.mean(mae_vals)),
            "fl_mae_sd": float(np.std(mae_vals)),
            "fl_rmse_mean": float(np.mean(rmse_vals)),
            "fl_rmse_sd": float(np.std(rmse_vals)),
            "fl_med_mean": float(np.mean(med_vals)),
            "fl_med_sd": float(np.std(med_vals)),
            "fl_pct_mean": float(np.mean(pct_vals)),
            "fl_pct_sd": float(np.std(pct_vals)),
            "fl_nme_mean": float(np.mean(nme_vals)),
            "fl_nme_sd": float(np.std(nme_vals)),
            "top10_mean": float(np.mean(top10_vals)),
            "top10_sd": float(np.std(top10_vals)),
            "epochs_mean": float(np.mean(ep_vals)),
            "epochs_sd": float(np.std(ep_vals)),
            "time_mean": float(np.mean(time_vals)),
            "time_sd": float(np.std(time_vals)),
            "fp_mae_mean": float(np.mean(fp_maes)),
            "ucl_mae_mean": float(np.mean(ucl_maes)),
        }

        # 3-Seed Median Ensemble
        ens_df = ensemble_3seeds(seed_dfs[42], seed_dfs[43], seed_dfs[44])
        pred_dfs_by_row[row_key] = ens_df

        ens_all = summarize_metrics(ens_df, f"{row_key}_ens_all")
        ens_fp = summarize_metrics(ens_df[ens_df["source"] == "FP"], f"{row_key}_ens_fp")
        ens_ucl = summarize_metrics(ens_df[ens_df["source"] == "UCL"], f"{row_key}_ens_ucl")

        all_results[row_key] = {
            "name": row_name,
            "resolution": S,
            "model_type": model_type,
            "seeds": seed_metrics,
            "seeds_mean_sd": mean_sd_stats,
            "ensemble_all": ens_all,
            "ensemble_fp": ens_fp,
            "ensemble_ucl": ens_ucl,
        }

    # Format Markdown Table
    md_table = format_ablation_markdown_table(all_results)
    print("\n" + "=" * 80)
    print("FINAL ABLATION TABLE:")
    print("=" * 80)
    print(md_table)

    # Save Markdown Table
    table_path = out_dir / "femur_ablation_table.md"
    with open(table_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(md_table)
    print(f"Saved {table_path}")

    # Save JSON results
    json_path = out_dir / "femur_ablation_results.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {json_path}")

    # Save combined predictions CSV (Row A ensemble by default, plus all rows)
    combined_csv_path = out_dir / "femur_predictions.csv"
    ens_row_a = pred_dfs_by_row["row_A"].copy()
    ens_row_a["row_A_fl_pred"] = pred_dfs_by_row["row_A"]["fl_pred"]
    ens_row_a["row_B_fl_pred"] = pred_dfs_by_row["row_B"]["fl_pred"]
    ens_row_a["row_C_fl_pred"] = pred_dfs_by_row["row_C"]["fl_pred"]
    ens_row_a.to_csv(combined_csv_path, index=False, lineterminator="\n")
    print(f"Saved {combined_csv_path}")

    # Generate and execute benchmark notebook
    nb_dir = Path("/kaggle/working") if Path("/kaggle").is_dir() else Path("notebooks/benchmarks")
    nb_file = nb_dir / "femur_ablation.ipynb"

    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Femur Model Ablation Study: Heatmap vs Coordinate Regression\n\n"
            "Evaluates 3 architectural approaches on the official FP+UCL Femur test set (N=348 images / 266 patients):\n"
            "- **Row A**: Heatmap landmarks (U-Net EfficientNet-B4, 384px letterbox, 1-channel dual-peak, flip TTA)\n"
            "- **Row B**: Direct coordinate regression (ConvNeXt-Tiny, 384px letterbox, order-invariant loss)\n"
            "- **Row C**: Heatmap landmarks (U-Net EfficientNet-B4, 512px letterbox, 1-channel dual-peak, flip TTA)\n\n"
            "Each row trained with 3 random seeds (42, 43, 44). Early stopping on 65 patient-grouped validation images.\n"
            "Patient-level bootstrap 95% confidence intervals computed with 1,000 resamples."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Markdown, display\n\n"
            "results_dir = Path('/kaggle/working/out')\n"
            "if not results_dir.is_dir():\n"
            "    results_dir = Path('../../docs/results/femur_ablation')\n"
            "if not results_dir.is_dir():\n"
            "    results_dir = Path('docs/results/femur_ablation')\n\n"
            "table_path = results_dir / 'femur_ablation_table.md'\n"
            "if table_path.is_file():\n"
            "    display(Markdown(table_path.read_text(encoding='utf-8')))\n"
            "else:\n"
            "    print('Table not found at', table_path)"
        ),
        nbf.v4.new_code_cell(
            "json_path = results_dir / 'femur_ablation_results.json'\n"
            "if json_path.is_file():\n"
            "    with open(json_path, 'r', encoding='utf-8') as f:\n"
            "        results = json.load(f)\n"
            "    print('Loaded ablation results for keys:', list(results.keys()))\n"
            "    for rk in ['row_A', 'row_B', 'row_C']:\n"
            "        if rk in results:\n"
            "            ens = results[rk]['ensemble_all']\n"
            "            ms = results[rk]['seeds_mean_sd']\n"
            "            print(f'=== {results[rk][\"name\"]} ===')\n"
            "            print(f'  Seeds Mean: MAE={ms[\"fl_mae_mean\"]:.3f} +- {ms[\"fl_mae_sd\"]:.3f} mm, NME={ms[\"fl_nme_mean\"]:.3f} +- {ms[\"fl_nme_sd\"]:.3f}')\n"
            "            print(f'  Ensemble:   MAE={ens[\"fl_mae\"]:.3f} [{ens[\"fl_mae_ci\"][0]:.3f}, {ens[\"fl_mae_ci\"][1]:.3f}] mm, NME={ens[\"fl_nme_mean\"]:.3f} +- {ens[\"fl_nme_sd\"]:.3f}')\n"
            "            print(f'  Top-10 SqErr Share: {ens[\"top10_sq_err_share\"]*100:.1f}%')\n"
            "else:\n"
            "    print('Results JSON not found at', json_path)"
        ),
        nbf.v4.new_code_cell(
            "csv_path = results_dir / 'femur_predictions.csv'\n"
            "if csv_path.is_file():\n"
            "    df = pd.read_csv(csv_path)\n"
            "    print(f'Total predictions: {len(df)} images across {df[\"patient\"].nunique()} patients')\n"
            "    print(f'Source counts: {dict(df[\"source\"].value_counts())}')\n"
            "    assert len(df) == 348, f'Expected 348 test images, got {len(df)}'\n"
            "    print('Verification check passed: 348 test predictions confirmed.')\n"
            "else:\n"
            "    print('Predictions CSV not found at', csv_path)"
        ),
    ]
    with open(nb_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook at {nb_file}. Executing via nbconvert...")

    executed = False
    for cmd in [
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb_file)],
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb_file)],
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb_file)],
    ]:
        try:
            print(f"Trying notebook execution with: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            executed = True
            break
        except Exception as e:
            print(f"nbconvert attempt failed: {e}")
            continue

    shutil.copy2(nb_file, out_dir / "femur_ablation.ipynb")
    if executed:
        print(f"Successfully executed and saved {out_dir / 'femur_ablation.ipynb'}")
    else:
        print(f"Saved notebook to {out_dir / 'femur_ablation.ipynb'} (without execution)")

    print("\nAll Femur Ablation tasks completed successfully!")


if __name__ == "__main__":
    run_ablation_experiment()
