"""Evaluation and biometric measurement for HC18 fetal head biometry.

Reuses robust ellipse fitting to compute Head Circumference (HC) and Biparietal
Diameter (BPD) from segmentation probability maps and ground truth masks.
"""

from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

from fetal.eval.ellipse_fitting import fit_ellipse_robust, hc_bpd_from_ellipse_params

IMG_SIZE = 384


def measure_from_prob(
    prob_map: np.ndarray,
    mask_tensor: torch.Tensor,
    pixel_size: float,
    scale: float,
) -> tuple[float, float] | None:
    """Measure HC and BPD absolute errors from probability map and target mask.

    Args:
        prob_map: 2D probability map array in [0, 1].
        mask_tensor: Ground truth mask tensor of shape (1, H, W) or (H, W).
        pixel_size: Image pixel size in mm/pixel.
        scale: Resize/letterbox scale factor used to normalize coordinates.

    Returns:
        Tuple of (abs(pred_hc - true_hc), abs(pred_bpd - true_bpd)), or None if
        contour cannot be fit.
    """
    binary = (prob_map > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None
    (_, _), (minor_px, major_px), _ = fit_ellipse_robust(largest)
    pred_hc, pred_bpd = hc_bpd_from_ellipse_params(
        (major_px / 2) / scale * pixel_size, (minor_px / 2) / scale * pixel_size
    )

    if mask_tensor.ndim == 3:
        mask_arr = mask_tensor[0].cpu().numpy()
    else:
        mask_arr = mask_tensor.cpu().numpy()
    true_mask = (mask_arr * 255).astype(np.uint8)
    true_contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not true_contours:
        return None
    true_largest = max(true_contours, key=cv2.contourArea)
    if len(true_largest) < 5:
        return None
    (_, _), (t_minor, t_major), _ = fit_ellipse_robust(true_largest)
    true_hc, true_bpd = hc_bpd_from_ellipse_params(
        (t_major / 2) / scale * pixel_size, (t_minor / 2) / scale * pixel_size
    )
    return abs(pred_hc - true_hc), abs(pred_bpd - true_bpd)


def predict_prob_with_tta(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Average prediction on the original image and horizontal flip.

    Args:
        model: Trained segmentation model.
        img_tensor: Input image tensor (3, H, W).
        device: Device to run inference on.

    Returns:
        2D numpy array of probabilities (H, W).
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    img_batch = img_tensor.unsqueeze(0).to(device)
    flipped_batch = torch.flip(img_batch, dims=[3])

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            prob_orig = torch.sigmoid(model(img_batch)).float().cpu().numpy()[0, 0]
            prob_flipped = torch.sigmoid(model(flipped_batch)).float().cpu().numpy()[0, 0]

    prob_flipped_back = np.flip(prob_flipped, axis=1)
    return (prob_orig + prob_flipped_back) / 2.0


def predict_native_prob(
    model: torch.nn.Module,
    image_pil: Image.Image,
    target_size: int = IMG_SIZE,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Run TTA prediction and project back to original image coordinates."""
    from torchvision import transforms

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    orig_w, orig_h = image_pil.size
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image_pil.resize((new_w, new_h))
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = norm(transforms.ToTensor()(canvas)).unsqueeze(0).to(device)
    flipped_tensor = torch.flip(img_tensor, dims=[3])

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            prob_orig = torch.sigmoid(model(img_tensor)).float().cpu().numpy()[0, 0]
            prob_flip = torch.sigmoid(model(flipped_tensor)).float().cpu().numpy()[0, 0]

    prob_flip_back = np.flip(prob_flip, axis=1)
    prob_avg = (prob_orig + prob_flip_back) / 2.0

    cropped = prob_avg[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    native_prob = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return native_prob


def load_raw_filled_mask(row: dict | Sequence, image_dir: str) -> np.ndarray:
    """Load annotation mask image and fill contour if valid."""
    filename = row["filename"] if isinstance(row, dict) else row.filename
    mask_filename = filename.replace(".png", "_Annotation.png")
    mask_path = os.path.join(image_dir, mask_filename)
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        return np.zeros((100, 100), dtype=np.uint8)
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(raw_mask)
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(raw_mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    return filled


def evaluate_solo_with_tta(
    model: torch.nn.Module,
    dataset,
    indices: Sequence[int],
    device: torch.device | str | None = None,
) -> tuple[list[float], list[float]]:
    """Single model, TTA-averaged, evaluated on given indices."""
    model.eval()
    hc_errors: list[float] = []
    bpd_errors: list[float] = []
    for idx in indices:
        img_tensor, mask_tensor, pixel_size, scale = dataset[idx]
        prob_map = predict_prob_with_tta(model, img_tensor, device=device)
        result = measure_from_prob(prob_map, mask_tensor, pixel_size, scale)
        if result is None:
            continue
        hc_err, bpd_err = result
        hc_errors.append(hc_err)
        bpd_errors.append(bpd_err)
    return hc_errors, bpd_errors


def evaluate_true_oof(
    fold_models: Sequence[torch.nn.Module],
    fold_val_indices: Sequence[Sequence[int]],
    dataset,
    device: torch.device | str | None = None,
) -> tuple[list[float], list[float]]:
    """True out-of-fold evaluation across all folds."""
    hc_errors: list[float] = []
    bpd_errors: list[float] = []
    for fold_idx, val_idx in enumerate(fold_val_indices):
        model = fold_models[fold_idx]
        model.eval()
        for idx in val_idx:
            img_tensor, mask_tensor, pixel_size, scale = dataset[idx]
            prob_map = predict_prob_with_tta(model, img_tensor, device=device)
            result = measure_from_prob(prob_map, mask_tensor, pixel_size, scale)
            if result is None:
                continue
            hc_err, bpd_err = result
            hc_errors.append(hc_err)
            bpd_errors.append(bpd_err)
    return hc_errors, bpd_errors


def compute_metrics(
    hc_errors: Sequence[float],
    bpd_errors: Sequence[float],
) -> dict[str, float]:
    """Compute MAE and RMSE metrics for HC and BPD."""
    hc_arr = np.array(hc_errors, dtype=np.float64)
    bpd_arr = np.array(bpd_errors, dtype=np.float64)
    return {
        "hc_mae": float(np.mean(hc_arr)) if len(hc_arr) > 0 else 0.0,
        "hc_rmse": float(np.sqrt(np.mean(hc_arr**2))) if len(hc_arr) > 0 else 0.0,
        "bpd_mae": float(np.mean(bpd_arr)) if len(bpd_arr) > 0 else 0.0,
        "bpd_rmse": float(np.sqrt(np.mean(bpd_arr**2))) if len(bpd_arr) > 0 else 0.0,
        "n_samples": int(len(hc_arr)),
    }


def measure_biometrics_from_prob(
    prob_map: np.ndarray,
    mask_tensor: torch.Tensor,
    pixel_size: float,
    scale: float,
) -> dict[str, float] | None:
    """Measure true and predicted HC and BPD in mm from probability map and target mask.

    Returns:
        Dict with keys: true_hc, pred_hc, abs_err_hc, true_bpd, pred_bpd, abs_err_bpd;
        or None if ellipse fitting fails on either mask.
    """
    binary = (prob_map > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None
    (_, _), (minor_px, major_px), _ = fit_ellipse_robust(largest)
    pred_hc, pred_bpd = hc_bpd_from_ellipse_params(
        (major_px / 2) / scale * pixel_size, (minor_px / 2) / scale * pixel_size
    )

    if mask_tensor.ndim == 3:
        mask_arr = mask_tensor[0].cpu().numpy()
    else:
        mask_arr = mask_tensor.cpu().numpy()
    true_mask = (mask_arr * 255).astype(np.uint8)
    true_contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not true_contours:
        return None
    true_largest = max(true_contours, key=cv2.contourArea)
    if len(true_largest) < 5:
        return None
    (_, _), (t_minor, t_major), _ = fit_ellipse_robust(true_largest)
    true_hc, true_bpd = hc_bpd_from_ellipse_params(
        (t_major / 2) / scale * pixel_size, (t_minor / 2) / scale * pixel_size
    )

    return {
        "true_hc": float(true_hc),
        "pred_hc": float(pred_hc),
        "abs_err_hc": float(abs(pred_hc - true_hc)),
        "true_bpd": float(true_bpd),
        "pred_bpd": float(pred_bpd),
        "abs_err_bpd": float(abs(pred_bpd - true_bpd)),
    }


def fit_submission_ellipse(
    prob_map_native: np.ndarray,
    pixel_size: float,
    filename: str,
) -> dict[str, float | str]:
    """Fit ellipse in native image coordinates and return Grand Challenge submission format.

    Args:
        prob_map_native: 2D probability map in native pixel dimensions (H, W).
        pixel_size: Millimeters per pixel.
        filename: Image filename.

    Returns:
        Dict with keys: filename, center_x_mm, center_y_mm, semi_axes_a_mm, semi_axes_b_mm, angle_rad.
    """
    binary = (prob_map_native > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {
            "filename": filename,
            "center_x_mm": 0.0,
            "center_y_mm": 0.0,
            "semi_axes_a_mm": 0.0,
            "semi_axes_b_mm": 0.0,
            "angle_rad": 0.0,
        }
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return {
            "filename": filename,
            "center_x_mm": 0.0,
            "center_y_mm": 0.0,
            "semi_axes_a_mm": 0.0,
            "semi_axes_b_mm": 0.0,
            "angle_rad": 0.0,
        }
    (cx_px, cy_px), (axis1_px, axis2_px), angle_deg = fit_ellipse_robust(largest)
    semi_major_mm = (max(axis1_px, axis2_px) / 2.0) * pixel_size
    semi_minor_mm = (min(axis1_px, axis2_px) / 2.0) * pixel_size
    center_x_mm = cx_px * pixel_size
    center_y_mm = cy_px * pixel_size
    angle_rad = float(np.deg2rad(angle_deg))

    return {
        "filename": filename,
        "center_x_mm": float(center_x_mm),
        "center_y_mm": float(center_y_mm),
        "semi_axes_a_mm": float(semi_major_mm),
        "semi_axes_b_mm": float(semi_minor_mm),
        "angle_rad": float(angle_rad),
    }


def compute_tercile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze error breakdown by true-HC terciles ('head-size proxy for gestational age').

    Args:
        df: DataFrame with true_HC_mm, abs_err_HC, abs_err_BPD.

    Returns:
        DataFrame with columns: tercile, n_samples, min_true_hc_mm, max_true_hc_mm,
        hc_mae, hc_rmse, bpd_mae, bpd_rmse.
    """
    required_cols = {"true_HC_mm", "abs_err_HC", "abs_err_BPD"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for tercile analysis: {missing}")

    df_copy = df.copy()
    df_copy["tercile"] = pd.qcut(
        df_copy["true_HC_mm"],
        q=3,
        labels=["Small (< T1)", "Medium (T1-T2)", "Large (> T2)"],
    )

    rows = []
    for label, grp in df_copy.groupby("tercile", observed=False):
        if len(grp) == 0:
            continue
        hc_errs = grp["abs_err_HC"].to_numpy(dtype=np.float64)
        bpd_errs = grp["abs_err_BPD"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "tercile": str(label),
                "n_samples": int(len(grp)),
                "min_true_hc_mm": float(grp["true_HC_mm"].min()),
                "max_true_hc_mm": float(grp["true_HC_mm"].max()),
                "hc_mae": float(np.mean(hc_errs)),
                "hc_rmse": float(np.sqrt(np.mean(hc_errs**2))),
                "bpd_mae": float(np.mean(bpd_errs)),
                "bpd_rmse": float(np.sqrt(np.mean(bpd_errs**2))),
            }
        )
    return pd.DataFrame(rows)


def compute_patient_bootstrap_ci(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute patient-level bootstrap 95% confidence interval for pooled HC MAE.

    Args:
        df: DataFrame with patient_id and abs_err_HC.
        n_boot: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Tuple of (ci_lower_2.5%, ci_upper_97.5%).
    """
    required_cols = {"patient_id", "abs_err_HC"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for patient bootstrap: {missing}")

    unique_patients = df["patient_id"].unique()
    pt_groups = df.groupby("patient_id")["abs_err_HC"].apply(lambda s: s.to_numpy(dtype=np.float64)).to_dict()

    rng = np.random.RandomState(seed)
    n_patients = len(unique_patients)
    boot_maes: list[float] = []

    for _ in range(n_boot):
        sampled_pts = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_errs = np.concatenate([pt_groups[p] for p in sampled_pts])
        boot_maes.append(float(np.mean(sampled_errs)))

    ci_low = float(np.percentile(boot_maes, 2.5))
    ci_high = float(np.percentile(boot_maes, 97.5))
    return ci_low, ci_high

