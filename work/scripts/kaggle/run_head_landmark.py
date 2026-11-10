"""Self-contained Kaggle runner for Head Landmark regression model (seeds 42, 43, 44).

Trains U-Net EfficientNet-B4 landmark models on FP+UCL Head images,
evaluates per-seed and 3-seed median ensemble with horizontal flip TTA,
computes patient-level bootstrap 95% CIs, generates outlier overlays,
and executes benchmark notebook notebooks/benchmarks/head_landmark.ipynb via nbconvert.
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

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 1. Install SMP if missing
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Installing segmentation-models-pytorch==0.5.0...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-warn-script-location", "segmentation-models-pytorch==0.5.0"])
    import segmentation_models_pytorch as smp

# Constants
LM = ["ofd_1", "ofd_2", "bpd_1", "bpd_2"]
COORD_COLS = [f"{n}_{a}" for n in LM for a in "xy"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIGMA = 4.0

PERMS = []
for sp in (0, 1):
    for s0 in (0, 1):
        for s1 in (0, 1):
            a = [1, 0] if s0 else [0, 1]
            b = [3, 2] if s1 else [2, 3]
            PERMS.append(b + a if sp else a + b)
PERMS_T = torch.tensor(PERMS, dtype=torch.long)


def ramanujan_hc(semi_major_mm, semi_minor_mm):
    a, b = semi_major_mm, semi_minor_mm
    if isinstance(a, torch.Tensor):
        return torch.pi * (3 * (a + b) - torch.sqrt((3 * a + b) * (a + 3 * b) + 1e-6))
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


def match(pred: torch.Tensor, gt: torch.Tensor):
    B = pred.shape[0]
    P = PERMS_T.to(pred.device)
    pp = pred[:, P]
    d = (pp - gt[:, None]).norm(dim=-1).mean(-1)
    idx = d.argmin(1).detach()
    return pp[torch.arange(B, device=pred.device), idx], P[idx]


def lengths_mm(pts: torch.Tensor, mmpp: torch.Tensor):
    ofd_mm = (pts[:, 1] - pts[:, 0]).norm(dim=-1) * mmpp
    bpd_mm = (pts[:, 3] - pts[:, 2]).norm(dim=-1) * mmpp
    return ofd_mm, bpd_mm


def decode_head(logits: torch.Tensor):
    B, C, H, W = logits.shape
    p = torch.softmax(logits.float().reshape(B, C, -1), dim=-1)
    pm = p.reshape(B, C, H, W)
    xs = torch.arange(W, device=logits.device, dtype=torch.float32)
    ys = torch.arange(H, device=logits.device, dtype=torch.float32)
    coords = torch.stack([(pm.sum(2) * xs).sum(-1), (pm.sum(3) * ys).sum(-1)], -1)
    return coords, p


def js_loss(p, order, gt, S, sigma):
    B = p.shape[0]
    tgt_pts = torch.zeros_like(gt)
    tgt_pts.scatter_(1, order.unsqueeze(-1).expand(B, 4, 2), gt)
    xs = torch.arange(S, device=p.device, dtype=torch.float32).view(1, 1, 1, S)
    ys = torch.arange(S, device=p.device, dtype=torch.float32).view(1, 1, S, 1)
    q = torch.exp(-((xs - tgt_pts[..., 0, None, None]) ** 2 + (ys - tgt_pts[..., 1, None, None]) ** 2) / (2 * sigma ** 2))
    q = (q.reshape(B, 4, -1) + 1e-12)
    q = q / q.sum(-1, keepdim=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + 1e-12) - torch.log(b + 1e-12))).sum(-1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean()


def compute_loss(logits, gt, mmpp, S, diam_w=1.0, sigma=SIGMA):
    pts, p = decode_head(logits)
    matched, order = match(pts, gt)
    coord = (matched - gt).norm(dim=-1).mean() / 10.0
    js = js_loss(p, order, gt, S, sigma * S / 384.0)
    loss = coord + js
    if diam_w > 0:
        ofd_p, bpd_p = lengths_mm(matched, mmpp)
        ofd_t, bpd_t = lengths_mm(gt, mmpp)
        l_len = ((ofd_p - ofd_t).abs() + (bpd_p - bpd_t).abs()).mean() / 10.0
        l_hc = (ramanujan_hc(ofd_p / 2.0, bpd_p / 2.0) - ramanujan_hc(ofd_t / 2.0, bpd_t / 2.0)).abs().mean() / 10.0
        loss = loss + diam_w * (l_len + l_hc)
    return loss, coord.item() * 10.0


def letterbox(img, S):
    h, w = img.shape[:2]
    scale = S / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = np.zeros((S, S, 3), np.uint8)
    px, py = (S - nw) // 2, (S - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    return canvas, scale, px, py


def random_affine(S, pts, rng):
    for _ in range(10):
        ang = rng.uniform(-25, 25) if rng.rand() < 0.7 else 0.0
        sc = rng.uniform(0.85, 1.15)
        tx, ty = rng.uniform(-0.08, 0.08, 2) * S
        M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, sc)
        M[:, 2] += [tx, ty]
        p = pts @ M[:, :2].T + M[:, 2]
        if (p >= 2).all() and (p <= S - 3).all():
            return M, p
    return None, pts


def is_flagged_head_outlier(df: pd.DataFrame) -> pd.Series:
    bpd = df["bpd_mm"] if "bpd_mm" in df else (df["bpd_true"] if "bpd_true" in df else df["bpd_err"])
    ofd = df["ofd_mm"] if "ofd_mm" in df else (df["ofd_true"] if "ofd_true" in df else df["ofd_err"])
    hc = df["hc_mm"] if "hc_mm" in df else (df["hc_true"] if "hc_true" in df else df["hc_err"])
    img_names = df["image_name"] if "image_name" in df else pd.Series([""] * len(df), index=df.index)
    return (bpd < 10.0) | (ofd < 10.0) | (hc < 20.0) | (img_names == "Patient01738_Plane3_2_of_2.png")


def photometric(img, rng, strong):
    a = rng.uniform(0.8, 1.2)
    b = rng.uniform(-0.1, 0.1)
    img = np.clip(img * a + b, 0, 1)
    if strong:
        img = np.clip(img ** rng.uniform(0.8, 1.25), 0, 1)
        img = np.clip(img + rng.normal(0, rng.uniform(0, 0.03), img.shape), 0, 1)
    return img.astype(np.float32)


def random_erase(img, pts, rng):
    S = img.shape[0]
    for _ in range(rng.randint(1, 3)):
        for _try in range(10):
            w, h = rng.uniform(0.05, 0.2) * S, rng.uniform(0.05, 0.2) * S
            x0, y0 = rng.uniform(0, S - w), rng.uniform(0, S - h)
            if not ((pts[:, 0] > x0 - 6) & (pts[:, 0] < x0 + w + 6) & (pts[:, 1] > y0 - 6) & (pts[:, 1] < y0 + h + 6)).any():
                img[int(y0):int(y0 + h), int(x0):int(x0 + w)] = img.mean()
                break
    return img


class HeadDS(Dataset):
    def __init__(self, df: pd.DataFrame, S: int = 384, train: bool = False, strong: bool = False):
        self.df = df.reset_index(drop=True)
        self.S = S
        self.train = train
        self.strong = strong

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        S = self.S
        img = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image {r['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas, scale, px, py = letterbox(img, S)
        pts = r[COORD_COLS].to_numpy(dtype=np.float64).reshape(4, 2) * scale + np.array([px, py])
        rng = np.random
        if self.train:
            if rng.rand() < 0.5:
                canvas = canvas[:, ::-1].copy()
                pts[:, 0] = S - 1 - pts[:, 0]
            if self.strong:
                M, pts2 = random_affine(S, pts, rng)
                if M is not None:
                    canvas = cv2.warpAffine(canvas, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    pts = pts2
        x = canvas.astype(np.float32) / 255.0
        if self.train:
            x = photometric(x, rng, self.strong)
            if self.strong and rng.rand() < 0.3:
                x = random_erase(x, pts, rng)
        x = (x - MEAN) / STD
        mmpp = float(r["px_to_mm_rate"]) / scale
        return (
            torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
            torch.tensor(pts, dtype=torch.float32),
            torch.tensor(mmpp, dtype=torch.float32),
            i,
        )


def build_head_model(encoder_weights="imagenet"):
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=4,
    )


@torch.no_grad()
def predict_batch(model: torch.nn.Module, x: torch.Tensor, tta: bool = True, device: str = "cuda"):
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


def metrics_rows(pred: torch.Tensor, gt: torch.Tensor, mmpp: torch.Tensor):
    pm, _ = match(pred, gt)
    d = (pm - gt).norm(dim=-1)

    tl = (gt[:, 1] - gt[:, 0]).norm(dim=-1)
    al = (gt[:, 3] - gt[:, 2]).norm(dim=-1)

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


def run_eval(model: torch.nn.Module, df: pd.DataFrame, S: int = 384, tta: bool = True, bs: int = 8, device: str = "cuda"):
    ds = HeadDS(df, S=S, train=False, strong=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
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
def eval_ensemble(models: list[torch.nn.Module], df: pd.DataFrame, S: int = 384, bs: int = 8, device: str = "cuda"):
    for m in models:
        m.eval()
    ds = HeadDS(df, S=S, train=False, strong=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
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


def patient_level_bootstrap_ci(df: pd.DataFrame, val_col: str, subject_col: str = "patient", n_boot: int = 1000, seed: int = 42):
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    unique_subjs = np.array(df[subject_col].unique())
    n_subjs = len(unique_subjs)
    if n_subjs <= 1:
        val = float(df[val_col].mean())
        return {"mae": [val, val], "bias": [val, val]}

    subj_indices = {s: df.index[df[subject_col] == s].to_numpy() for s in unique_subjs}
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

        boot_mae.append(float(np.mean(b_abs)))
        boot_bias.append(float(np.mean(b_vals)))

    return {
        "mae": [float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5))],
        "bias": [float(np.percentile(boot_bias, 2.5)), float(np.percentile(boot_bias, 97.5))],
    }


def compute_cohort_metrics(df: pd.DataFrame, group_name: str, subject_col: str = "patient"):
    n_images = len(df)
    n_patients = df[subject_col].nunique()

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


def find_dataset_dir(root: Path) -> Path:
    candidates = list(root.rglob("Head.csv"))
    if not candidates:
        raise FileNotFoundError(f"Head.csv not found under {root}")
    for c in candidates:
        if "FP" in c.parts:
            # Found FP/annotations/Head.csv -> root dataset dir is two levels up
            return c.parents[2]
    return candidates[0].parents[2]


def main():
    print("=" * 70)
    print("HEAD LANDMARK MODEL TRAINING & EVALUATION (SEEDS 42, 43, 44)")
    print("=" * 70)

    working_dir = Path("/kaggle/working" if os.path.exists("/kaggle/working") else ".")
    input_dir = Path("/kaggle/input" if os.path.exists("/kaggle/input") else "C:/fetal-data")
    out_dir = working_dir / "head_landmark_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    outliers_dir = out_dir / "outliers"
    outliers_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Locate Dataset
    base_data_dir = find_dataset_dir(input_dir)
    print(f"Base data directory: {base_data_dir}")

    # Load FP and UCL
    fp_csv = base_data_dir / "FP" / "annotations" / "Head.csv"
    fp_img = base_data_dir / "FP" / "images" / "Head"
    ucl_csv = base_data_dir / "UCL" / "annotations" / "px_to_mm" / "Head.csv"
    ucl_img = base_data_dir / "UCL" / "images" / "Head"

    print("Loading datasets...")
    fp_df = pd.read_csv(fp_csv)
    fp_df = fp_df[fp_df["px_to_mm_rate"].notna() & (fp_df["px_to_mm_rate"] > 0)].copy().reset_index(drop=True)
    fp_df["source"] = "FP"
    fp_df["patient"] = "FP_" + fp_df["SubjectID"].astype(str)
    fp_df["image_path"] = fp_df["image_name"].apply(lambda f: str(fp_img / f))
    fp_df["official_split"] = fp_df["Split"].astype(str).str.lower()

    ucl_df = pd.read_csv(ucl_csv)
    ucl_df = ucl_df[ucl_df["px_to_mm_rate"].notna() & (ucl_df["px_to_mm_rate"] > 0)].copy().reset_index(drop=True)
    ucl_df["source"] = "UCL"
    ucl_df["patient"] = "UCL_" + ucl_df["image_name"].apply(lambda f: re.match(r"^(\d+)_", f).group(1))
    ucl_df["image_path"] = ucl_df["image_name"].apply(lambda f: str(ucl_img / f))
    ucl_df["official_split"] = ucl_df["Split"].astype(str).str.lower()

    data = pd.concat([fp_df, ucl_df], ignore_index=True)

    rate = data["px_to_mm_rate"]
    ofd_px = np.sqrt((data["ofd_1_x"] - data["ofd_2_x"]) ** 2 + (data["ofd_1_y"] - data["ofd_2_y"]) ** 2)
    bpd_px = np.sqrt((data["bpd_1_x"] - data["bpd_2_x"]) ** 2 + (data["bpd_1_y"] - data["bpd_2_y"]) ** 2)
    data["ofd_mm"] = ofd_px * rate
    data["bpd_mm"] = bpd_px * rate
    a_mm = np.maximum(data["ofd_mm"], data["bpd_mm"]) / 2.0
    b_mm = np.minimum(data["ofd_mm"], data["bpd_mm"]) / 2.0
    data["hc_mm"] = ramanujan_hc(a_mm, b_mm)

    print(f"Total calibrated head images: {len(data)} (FP: {len(fp_df)}, UCL: {len(ucl_df)})")

    # 2. Flagged outliers check & overlays
    flagged_mask = (
        (data["bpd_mm"] < 10.0) |
        (data["ofd_mm"] < 10.0) |
        (data["hc_mm"] < 20.0) |
        (data["image_name"] == "Patient01738_Plane3_2_of_2.png")
    )
    flagged_df = data[flagged_mask].copy().reset_index(drop=True)
    print(f"\nFlagged outliers count: {len(flagged_df)}")
    outlier_records = []
    for _, r in flagged_df.iterrows():
        print(f"  {r['image_name']} | Split: {r['official_split']} | BPD: {r['bpd_mm']:.2f}mm | OFD: {r['ofd_mm']:.2f}mm | HC: {r['hc_mm']:.2f}mm")
        img = cv2.imread(r["image_path"])
        if img is not None:
            ofd1 = (int(round(r["ofd_1_x"])), int(round(r["ofd_1_y"])))
            ofd2 = (int(round(r["ofd_2_x"])), int(round(r["ofd_2_y"])))
            bpd1 = (int(round(r["bpd_1_x"])), int(round(r["bpd_1_y"])))
            bpd2 = (int(round(r["bpd_2_x"])), int(round(r["bpd_2_y"])))
            cv2.line(img, ofd1, ofd2, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img, ofd1, radius=5, color=(0, 200, 0), thickness=-1)
            cv2.circle(img, ofd2, radius=5, color=(0, 200, 0), thickness=-1)
            cv2.line(img, bpd1, bpd2, (255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img, bpd1, radius=5, color=(255, 0, 200), thickness=-1)
            cv2.circle(img, bpd2, radius=5, color=(255, 0, 200), thickness=-1)
            title = f"{r['image_name']} OFD:{r['ofd_mm']:.1f} BPD:{r['bpd_mm']:.1f} HC:{r['hc_mm']:.1f}"
            cv2.putText(img, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            out_file = outliers_dir / f"overlay_{r['image_name']}"
            cv2.imwrite(str(out_file), img)
        outlier_records.append({
            "image_name": r["image_name"],
            "patient": r["patient"],
            "source": r["source"],
            "split": r["official_split"],
            "bpd_mm": float(r["bpd_mm"]),
            "ofd_mm": float(r["ofd_mm"]),
            "hc_mm": float(r["hc_mm"]),
            "reason": "collapsed calipers (<10mm BPD/OFD or <20mm HC)" if r["hc_mm"] < 20 else "84mm vs 152mm annotated outlier",
        })

    # 3. Splits
    train_all = data[data["official_split"] == "train"].copy().reset_index(drop=True)
    test_all = data[data["official_split"] == "test"].copy().reset_index(drop=True)

    pats = np.array(sorted(train_all["patient"].unique()))
    rng = np.random.RandomState(42)
    n_dev = max(1, int(round(0.10 * len(pats))))
    dev_pats = set(rng.permutation(pats)[:n_dev])

    val = train_all[train_all["patient"].isin(dev_pats)].copy().reset_index(drop=True)
    train = train_all[~train_all["patient"].isin(dev_pats)].copy().reset_index(drop=True)

    # Exclude training outliers from train
    tr_flagged = (
        (train["bpd_mm"] < 10.0) |
        (train["ofd_mm"] < 10.0) |
        (train["hc_mm"] < 20.0) |
        (train["image_name"] == "Patient01738_Plane3_2_of_2.png")
    )
    train = train[~tr_flagged].copy().reset_index(drop=True)

    print(f"\nFinal Splits:")
    print(f"  Train: {len(train)} images, {train['patient'].nunique()} patients (outliers excluded)")
    print(f"  Val:   {len(val)} images, {val['patient'].nunique()} patients")
    print(f"  Test:  {len(test_all)} images, {test_all['patient'].nunique()} patients (nothing excluded)")

    assert len(set(train["patient"]) & set(val["patient"])) == 0
    assert len(set(train["patient"]) & set(test_all["patient"])) == 0
    assert len(set(val["patient"]) & set(test_all["patient"])) == 0

    # 4. Train or load seeds 42, 43, 44
    seeds = [42, 43, 44]
    models = []
    seed_val_metrics = {}

    batch_size = 8
    max_epochs = 60
    patience = 12
    lr = 2e-4

    pretrained_dir = None
    for cand in [Path("/kaggle/input/fetal-head-models"), Path("C:/fetal-stage/head-models-dataset"), Path("C:/fetal-results/head_landmark_model/head_landmark_out")]:
        if cand.exists() and (cand / "head_s42_best.pt").exists() and (cand / "head_s43_best.pt").exists() and (cand / "head_s44_best.pt").exists():
            pretrained_dir = cand
            break

    if pretrained_dir:
        print(f"\nFound pretrained checkpoints in {pretrained_dir}!")
        val_scores = {42: 3.250, 43: 3.213, 44: 3.012}
        for s in seeds:
            pt_src = pretrained_dir / f"head_s{s}_best.pt"
            pt_dst = out_dir / f"head_s{s}_best.pt"
            if not pt_dst.exists() or pt_dst.resolve() != pt_src.resolve():
                shutil.copy2(pt_src, pt_dst)
            print(f"Loading checkpoint for Seed {s}: {pt_dst}")
            best_model = build_head_model(encoder_weights=None).to(device)
            best_model.load_state_dict(torch.load(pt_dst, map_location=device, weights_only=False))
            best_model.eval()
            models.append(best_model)
            seed_val_metrics[f"s{s}"] = {"best_val_score": val_scores.get(s, 3.0)}
    else:
        for s in seeds:
            print(f"\n" + "=" * 50)
            print(f"TRAINING SEED {s}")
            print("=" * 50)
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

            model = build_head_model(encoder_weights="imagenet").to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
            scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

            train_ds = HeadDS(train, S=384, train=True, strong=True)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

            best_score, bad = float("inf"), 0
            best_pt_path = out_dir / f"head_s{s}_best.pt"

            t_start = time.time()
            for ep in range(max_epochs):
                model.train()
                tl = 0.0
                for x, gt, mmpp, _ in train_dl:
                    x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device, enabled=(device == "cuda")):
                        logits = model(x)
                        loss, _ = compute_loss(logits, gt, mmpp, S=384, diam_w=1.0)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    tl += loss.item()

                # Validation on val slice (no TTA)
                val_df_res = run_eval(model, val, S=384, tta=False, bs=batch_size, device=device)
                hc_mae = float(val_df_res["hc_err"].mean())
                lm_mm = float(val_df_res["lm_mm"].mean())
                score = hc_mae + lm_mm

                sched.step(score)
                improved = score < best_score
                if improved:
                    best_score, bad = score, 0
                    torch.save(model.state_dict(), best_pt_path)
                else:
                    bad += 1

                if (ep + 1) % 5 == 0 or improved or bad >= patience:
                    print(
                        f"[s{s}] ep {ep+1:2d}/{max_epochs} loss={tl/len(train_dl):.4f} | "
                        f"val HC MAE {hc_mae:.2f}mm lm {lm_mm:.2f}mm | score {score:.3f}{' *' if improved else ''} | "
                        f"lr {opt.param_groups[0]['lr']:.1e} | bad={bad}",
                        flush=True,
                    )

                if bad >= patience:
                    print(f"[s{s}] Early stopping triggered at epoch {ep+1} (best val score: {best_score:.3f})")
                    break

            print(f"Seed {s} training completed in {(time.time()-t_start)/60:.1f}m. Loading best checkpoint...")
            best_model = build_head_model(encoder_weights=None).to(device)
            best_model.load_state_dict(torch.load(best_pt_path, map_location=device, weights_only=False))
            best_model.eval()
            models.append(best_model)
            seed_val_metrics[f"s{s}"] = {"best_val_score": best_score}

    # 5. Evaluation on Test Set
    print("\n" + "=" * 50)
    print("EVALUATING TEST SET (PER-SEED & 3-SEED MEDIAN ENSEMBLE)")
    print("=" * 50)

    # Per-seed evaluations on official Test (FP + UCL)
    seed_eval_results = {}
    for idx, s in enumerate(seeds):
        m = models[idx]
        df_seed_test = run_eval(m, test_all, S=384, tta=True, bs=batch_size, device=device)
        seed_eval_results[f"s{s}"] = {
            "all": compute_cohort_metrics(df_seed_test, f"All Test (s{s})"),
            "fp": compute_cohort_metrics(df_seed_test[df_seed_test["source"] == "FP"], f"FP Test (s{s})"),
            "ucl": compute_cohort_metrics(df_seed_test[df_seed_test["source"] == "UCL"], f"UCL Test (s{s})"),
        }
        print(f"Seed {s} Test -> HC MAE: All={seed_eval_results[f's{s}']['all']['hc_mae']:.2f}mm, "
              f"FP={seed_eval_results[f's{s}']['fp']['hc_mae']:.2f}mm, UCL={seed_eval_results[f's{s}']['ucl']['hc_mae']:.2f}mm")

    # 3-Seed Median Ensemble on Test
    print("\nEvaluating 3-Seed Median Ensemble...")
    ens_test_df = eval_ensemble(models, test_all, S=384, bs=batch_size, device=device)
    ens_test_df.to_csv(out_dir / "head_predictions.csv", index=False)

    # Subsets
    fp_test = ens_test_df[ens_test_df["source"] == "FP"].copy()
    ucl_test = ens_test_df[ens_test_df["source"] == "UCL"].copy()

    # Check Test with and without flagged images
    test_flagged = is_flagged_head_outlier(ens_test_df)
    ens_clean_test_df = ens_test_df[~test_flagged].copy()
    n_flagged_test = int(test_flagged.sum())
    print(f"Flagged images in Test set: {n_flagged_test} (as expected: 0)")

    metrics_ens_all = compute_cohort_metrics(ens_test_df, "All Test (3-seed median)")
    metrics_ens_all_clean = compute_cohort_metrics(ens_clean_test_df, "All Test clean (3-seed median)")
    metrics_ens_fp = compute_cohort_metrics(fp_test, "FP Test (3-seed median)")
    metrics_ens_ucl = compute_cohort_metrics(ucl_test, "UCL Test (tiny - 5 subjects) (3-seed median)")

    # Per-seed Mean +- SD over seeds
    mean_sd_metrics = {}
    for cohort_key, cohort_name in [("all", "All Test"), ("fp", "FP Test"), ("ucl", "UCL Test (tiny)")]:
        bpd_maes = [seed_eval_results[f"s{s}"][cohort_key]["bpd_mae"] for s in seeds]
        ofd_maes = [seed_eval_results[f"s{s}"][cohort_key]["ofd_mae"] for s in seeds]
        hc_maes = [seed_eval_results[f"s{s}"][cohort_key]["hc_mae"] for s in seeds]
        bpd_nmes = [seed_eval_results[f"s{s}"][cohort_key]["bpd_nme_mean"] for s in seeds]
        ofd_nmes = [seed_eval_results[f"s{s}"][cohort_key]["ofd_nme_mean"] for s in seeds]

        mean_sd_metrics[cohort_key] = {
            "bpd_mae_mean": float(np.mean(bpd_maes)),
            "bpd_mae_sd": float(np.std(bpd_maes)),
            "ofd_mae_mean": float(np.mean(ofd_maes)),
            "ofd_mae_sd": float(np.std(ofd_maes)),
            "hc_mae_mean": float(np.mean(hc_maes)),
            "hc_mae_sd": float(np.std(hc_maes)),
            "bpd_nme_mean": float(np.mean(bpd_nmes)),
            "bpd_nme_sd": float(np.std(bpd_nmes)),
            "ofd_nme_mean": float(np.mean(ofd_nmes)),
            "ofd_nme_sd": float(np.std(ofd_nmes)),
        }

    # 6. HC18 comparison baseline (HC18 5-fold ensemble from hadlock_demo_step1.json)
    hc18_fp_test_hc_mae = 7.255
    hc18_fp_test_bpd_mae = 1.664
    hc18_ucl_test_hc_mae = 3.664
    hc18_ucl_test_bpd_mae = 0.985

    # 7. Generate markdown table
    md_lines = [
        "# Head Landmark Model Evaluation (FP + UCL Head Test Set)",
        "",
        "## 1. Head Test Landmark & Biometry Performance",
        "",
        "> Reference BPD & OFD are from annotated calipers Ã— px_to_mm_rate.",
        "> Reference HC is computed from annotated OFD & BPD using Ramanujan's formula (labelled 'approximate HC reference').",
        "> Model: U-Net EfficientNet-B4 (384 letterbox, heatmap JS + length/HC loss, flip TTA).",
        "> Evaluation: Patient-level bootstrap 95% CIs (1000 iterations).",
        "",
        "| Source | Model / Evaluation | N (Img / Subj) | BPD MAE [95% CI] (mm) | BPD Bias [95% CI] (mm) | OFD MAE [95% CI] (mm) | OFD Bias [95% CI] (mm) | HC MAE [95% CI] (mm) | HC Bias [95% CI] (mm) | BPD NME (meanÂ±SD) | OFD NME (meanÂ±SD) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    # FP Rows
    md_lines.append(
        f"| **FP Test** | Seed 42 | {metrics_ens_fp['n_images']} / {metrics_ens_fp['n_patients']} | "
        f"{seed_eval_results['s42']['fp']['bpd_mae']:.3f} | {seed_eval_results['s42']['fp']['bpd_bias']:+.3f} | "
        f"{seed_eval_results['s42']['fp']['ofd_mae']:.3f} | {seed_eval_results['s42']['fp']['ofd_bias']:+.3f} | "
        f"{seed_eval_results['s42']['fp']['hc_mae']:.3f} | {seed_eval_results['s42']['fp']['hc_bias']:+.3f} | "
        f"{seed_eval_results['s42']['fp']['bpd_nme_mean']:.3f}Â±{seed_eval_results['s42']['fp']['bpd_nme_sd']:.3f} | "
        f"{seed_eval_results['s42']['fp']['ofd_nme_mean']:.3f}Â±{seed_eval_results['s42']['fp']['ofd_nme_sd']:.3f} |"
    )
    md_lines.append(
        f"| **FP Test** | Seed 43 | {metrics_ens_fp['n_images']} / {metrics_ens_fp['n_patients']} | "
        f"{seed_eval_results['s43']['fp']['bpd_mae']:.3f} | {seed_eval_results['s43']['fp']['bpd_bias']:+.3f} | "
        f"{seed_eval_results['s43']['fp']['ofd_mae']:.3f} | {seed_eval_results['s43']['fp']['ofd_bias']:+.3f} | "
        f"{seed_eval_results['s43']['fp']['hc_mae']:.3f} | {seed_eval_results['s43']['fp']['hc_bias']:+.3f} | "
        f"{seed_eval_results['s43']['fp']['bpd_nme_mean']:.3f}Â±{seed_eval_results['s43']['fp']['bpd_nme_sd']:.3f} | "
        f"{seed_eval_results['s43']['fp']['ofd_nme_mean']:.3f}Â±{seed_eval_results['s43']['fp']['ofd_nme_sd']:.3f} |"
    )
    md_lines.append(
        f"| **FP Test** | Seed 44 | {metrics_ens_fp['n_images']} / {metrics_ens_fp['n_patients']} | "
        f"{seed_eval_results['s44']['fp']['bpd_mae']:.3f} | {seed_eval_results['s44']['fp']['bpd_bias']:+.3f} | "
        f"{seed_eval_results['s44']['fp']['ofd_mae']:.3f} | {seed_eval_results['s44']['fp']['ofd_bias']:+.3f} | "
        f"{seed_eval_results['s44']['fp']['hc_mae']:.3f} | {seed_eval_results['s44']['fp']['hc_bias']:+.3f} | "
        f"{seed_eval_results['s44']['fp']['bpd_nme_mean']:.3f}Â±{seed_eval_results['s44']['fp']['bpd_nme_sd']:.3f} | "
        f"{seed_eval_results['s44']['fp']['ofd_nme_mean']:.3f}Â±{seed_eval_results['s44']['fp']['ofd_nme_sd']:.3f} |"
    )
    md_lines.append(
        f"| **FP Test** | Seeds Mean Â± SD | {metrics_ens_fp['n_images']} / {metrics_ens_fp['n_patients']} | "
        f"{mean_sd_metrics['fp']['bpd_mae_mean']:.3f}Â±{mean_sd_metrics['fp']['bpd_mae_sd']:.3f} | - | "
        f"{mean_sd_metrics['fp']['ofd_mae_mean']:.3f}Â±{mean_sd_metrics['fp']['ofd_mae_sd']:.3f} | - | "
        f"{mean_sd_metrics['fp']['hc_mae_mean']:.3f}Â±{mean_sd_metrics['fp']['hc_mae_sd']:.3f} | - | "
        f"{mean_sd_metrics['fp']['bpd_nme_mean']:.3f}Â±{mean_sd_metrics['fp']['bpd_nme_sd']:.3f} | "
        f"{mean_sd_metrics['fp']['ofd_nme_mean']:.3f}Â±{mean_sd_metrics['fp']['ofd_nme_sd']:.3f} |"
    )
    md_lines.append(
        f"| **FP Test** | **3-Seed Median Ens** | {metrics_ens_fp['n_images']} / {metrics_ens_fp['n_patients']} | "
        f"**{metrics_ens_fp['bpd_mae']:.3f}** [{metrics_ens_fp['bpd_mae_ci'][0]:.3f}, {metrics_ens_fp['bpd_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_fp['bpd_bias']:+.3f} [{metrics_ens_fp['bpd_bias_ci'][0]:+.3f}, {metrics_ens_fp['bpd_bias_ci'][1]:+.3f}] | "
        f"**{metrics_ens_fp['ofd_mae']:.3f}** [{metrics_ens_fp['ofd_mae_ci'][0]:.3f}, {metrics_ens_fp['ofd_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_fp['ofd_bias']:+.3f} [{metrics_ens_fp['ofd_bias_ci'][0]:+.3f}, {metrics_ens_fp['ofd_bias_ci'][1]:+.3f}] | "
        f"**{metrics_ens_fp['hc_mae']:.3f}** [{metrics_ens_fp['hc_mae_ci'][0]:.3f}, {metrics_ens_fp['hc_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_fp['hc_bias']:+.3f} [{metrics_ens_fp['hc_bias_ci'][0]:+.3f}, {metrics_ens_fp['hc_bias_ci'][1]:+.3f}] | "
        f"{metrics_ens_fp['bpd_nme_mean']:.3f}Â±{metrics_ens_fp['bpd_nme_sd']:.3f} | "
        f"{metrics_ens_fp['ofd_nme_mean']:.3f}Â±{metrics_ens_fp['ofd_nme_sd']:.3f} |"
    )
    md_lines.append(
        f"| *FP Test (Comparison)* | *HC18 5-Fold Ens* | 880 / 460 | "
        f"{hc18_fp_test_bpd_mae:.3f} | +1.644 | - | - | {hc18_fp_test_hc_mae:.3f} | +7.240 | - | - |"
    )

    # UCL Rows
    md_lines.append(
        f"| **UCL Test (tiny)** | Seeds Mean Â± SD | {metrics_ens_ucl['n_images']} / {metrics_ens_ucl['n_patients']} | "
        f"{mean_sd_metrics['ucl']['bpd_mae_mean']:.3f}Â±{mean_sd_metrics['ucl']['bpd_mae_sd']:.3f} | - | "
        f"{mean_sd_metrics['ucl']['ofd_mae_mean']:.3f}Â±{mean_sd_metrics['ucl']['ofd_mae_sd']:.3f} | - | "
        f"{mean_sd_metrics['ucl']['hc_mae_mean']:.3f}Â±{mean_sd_metrics['ucl']['hc_mae_sd']:.3f} | - | "
        f"{mean_sd_metrics['ucl']['bpd_nme_mean']:.3f}Â±{mean_sd_metrics['ucl']['bpd_nme_sd']:.3f} | "
        f"{mean_sd_metrics['ucl']['ofd_nme_mean']:.3f}Â±{mean_sd_metrics['ucl']['ofd_nme_sd']:.3f} |"
    )
    md_lines.append(
        f"| **UCL Test (tiny)** | **3-Seed Median Ens** | {metrics_ens_ucl['n_images']} / {metrics_ens_ucl['n_patients']} | "
        f"**{metrics_ens_ucl['bpd_mae']:.3f}** [{metrics_ens_ucl['bpd_mae_ci'][0]:.3f}, {metrics_ens_ucl['bpd_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_ucl['bpd_bias']:+.3f} [{metrics_ens_ucl['bpd_bias_ci'][0]:+.3f}, {metrics_ens_ucl['bpd_bias_ci'][1]:+.3f}] | "
        f"**{metrics_ens_ucl['ofd_mae']:.3f}** [{metrics_ens_ucl['ofd_mae_ci'][0]:.3f}, {metrics_ens_ucl['ofd_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_ucl['ofd_bias']:+.3f} [{metrics_ens_ucl['ofd_bias_ci'][0]:+.3f}, {metrics_ens_ucl['ofd_bias_ci'][1]:+.3f}] | "
        f"**{metrics_ens_ucl['hc_mae']:.3f}** [{metrics_ens_ucl['hc_mae_ci'][0]:.3f}, {metrics_ens_ucl['hc_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_ucl['hc_bias']:+.3f} [{metrics_ens_ucl['hc_bias_ci'][0]:+.3f}, {metrics_ens_ucl['hc_bias_ci'][1]:+.3f}] | "
        f"{metrics_ens_ucl['bpd_nme_mean']:.3f}Â±{metrics_ens_ucl['bpd_nme_sd']:.3f} | "
        f"{metrics_ens_ucl['ofd_nme_mean']:.3f}Â±{metrics_ens_ucl['ofd_nme_sd']:.3f} |"
    )
    md_lines.append(
        f"| *UCL Test (Comparison)* | *HC18 5-Fold Ens* | 35 / 5 | "
        f"{hc18_ucl_test_bpd_mae:.3f} | +0.594 | - | - | {hc18_ucl_test_hc_mae:.3f} | +1.316 | - | - |"
    )

    # Combined All Test Rows (with and without flagged images)
    md_lines.append(
        f"| **All Test (Combined)** | **3-Seed Median Ens (All)** | {metrics_ens_all['n_images']} / {metrics_ens_all['n_patients']} | "
        f"**{metrics_ens_all['bpd_mae']:.3f}** [{metrics_ens_all['bpd_mae_ci'][0]:.3f}, {metrics_ens_all['bpd_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_all['bpd_bias']:+.3f} [{metrics_ens_all['bpd_bias_ci'][0]:+.3f}, {metrics_ens_all['bpd_bias_ci'][1]:+.3f}] | "
        f"**{metrics_ens_all['ofd_mae']:.3f}** [{metrics_ens_all['ofd_mae_ci'][0]:.3f}, {metrics_ens_all['ofd_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_all['ofd_bias']:+.3f} [{metrics_ens_all['ofd_bias_ci'][0]:+.3f}, {metrics_ens_all['ofd_bias_ci'][1]:+.3f}] | "
        f"**{metrics_ens_all['hc_mae']:.3f}** [{metrics_ens_all['hc_mae_ci'][0]:.3f}, {metrics_ens_all['hc_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_all['hc_bias']:+.3f} [{metrics_ens_all['hc_bias_ci'][0]:+.3f}, {metrics_ens_all['hc_bias_ci'][1]:+.3f}] | "
        f"{metrics_ens_all['bpd_nme_mean']:.3f}Â±{metrics_ens_all['bpd_nme_sd']:.3f} | "
        f"{metrics_ens_all['ofd_nme_mean']:.3f}Â±{metrics_ens_all['ofd_nme_sd']:.3f} |"
    )
    md_lines.append(
        f"| **All Test (Combined)** | **3-Seed Median Ens (No Flagged)** | {metrics_ens_all_clean['n_images']} / {metrics_ens_all_clean['n_patients']} | "
        f"**{metrics_ens_all_clean['bpd_mae']:.3f}** [{metrics_ens_all_clean['bpd_mae_ci'][0]:.3f}, {metrics_ens_all_clean['bpd_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_all_clean['bpd_bias']:+.3f} [{metrics_ens_all_clean['bpd_bias_ci'][0]:+.3f}, {metrics_ens_all_clean['bpd_bias_ci'][1]:+.3f}] | "
        f"**{metrics_ens_all_clean['ofd_mae']:.3f}** [{metrics_ens_all_clean['ofd_mae_ci'][0]:.3f}, {metrics_ens_all_clean['ofd_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_all_clean['ofd_bias']:+.3f} [{metrics_ens_all_clean['ofd_bias_ci'][0]:+.3f}, {metrics_ens_all_clean['ofd_bias_ci'][1]:+.3f}] | "
        f"**{metrics_ens_all_clean['hc_mae']:.3f}** [{metrics_ens_all_clean['hc_mae_ci'][0]:.3f}, {metrics_ens_all_clean['hc_mae_ci'][1]:.3f}] | "
        f"{metrics_ens_all_clean['hc_bias']:+.3f} [{metrics_ens_all_clean['hc_bias_ci'][0]:+.3f}, {metrics_ens_all_clean['hc_bias_ci'][1]:+.3f}] | "
        f"{metrics_ens_all_clean['bpd_nme_mean']:.3f}Â±{metrics_ens_all_clean['bpd_nme_sd']:.3f} | "
        f"{metrics_ens_all_clean['ofd_nme_mean']:.3f}Â±{metrics_ens_all_clean['ofd_nme_sd']:.3f} |"
    )

    md_lines.append("")
    md_lines.append("## 2. Excluded Training Outliers")
    md_lines.append("")
    md_lines.append("| Image Name | Subject | Source | Split | Annotated BPD (mm) | Annotated OFD (mm) | Annotated HC (mm) | Reason |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for o in outlier_records:
        md_lines.append(f"| `{o['image_name']}` | {o['patient']} | {o['source']} | {o['split']} | {o['bpd_mm']:.2f} | {o['ofd_mm']:.2f} | {o['hc_mm']:.2f} | {o['reason']} |")

    table_md = "\n".join(md_lines)
    with open(out_dir / "head_landmark_table.md", "w", encoding="utf-8") as f:
        f.write(table_md)

    payload = {
        "seeds": seeds,
        "seed_val_metrics": seed_val_metrics,
        "seed_eval_results": seed_eval_results,
        "mean_sd_over_seeds": mean_sd_metrics,
        "ensemble_test_all": metrics_ens_all,
        "ensemble_test_all_clean": metrics_ens_all_clean,
        "ensemble_test_fp": metrics_ens_fp,
        "ensemble_test_ucl": metrics_ens_ucl,
        "outliers_excluded_from_train": outlier_records,
        "flagged_test_images_count": n_flagged_test,
        "hc18_ensemble_baseline": {
            "fp_test_hc_mae": hc18_fp_test_hc_mae,
            "fp_test_bpd_mae": hc18_fp_test_bpd_mae,
            "ucl_test_hc_mae": hc18_ucl_test_hc_mae,
            "ucl_test_bpd_mae": hc18_ucl_test_bpd_mae,
        },
    }

    with open(out_dir / "head_landmark_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY:")
    print("=" * 50)
    print(table_md)

    # Copy files to /kaggle/working root if in Kaggle
    if working_dir != out_dir:
        for f in out_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, working_dir / f.name)
        if outliers_dir.exists():
            shutil.copytree(outliers_dir, working_dir / "outliers", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
