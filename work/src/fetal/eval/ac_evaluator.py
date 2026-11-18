import math
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fetal import config
from fetal.data.ac_dataset import AbdomenDS
from fetal.models.ac_landmark import decode
from fetal.eval.ac_geometry import match, lengths_mm, ac_t


@torch.no_grad()
def predict_batch(model, x, tta, device=None):
    if device is None:
        device = config.DEVICE
    with torch.autocast(device_type=device, enabled=(device == "cuda")):
        lo0 = model(x)
        lo1 = model(torch.flip(x, dims=[3])) if tta else None
    p0, _ = decode(lo0)
    if not tta:
        return p0
    p1, _ = decode(lo1)
    p1 = p1.clone()
    p1[..., 0] = x.shape[-1] - 1 - p1[..., 0]
    m1, _ = match(p1, p0)
    return (p0 + m1) / 2


def metrics_rows(pred, gt, mmpp):
    pm, _ = match(pred, gt)
    d = (pm - gt).norm(dim=-1)                                   # px per landmark
    tp, ap = lengths_mm(pm, mmpp)
    tt, at = lengths_mm(gt, mmpp)
    acp, act = ac_t(tp, ap), ac_t(tt, at)
    tl, al = (gt[:, 1] - gt[:, 0]).norm(dim=-1), (gt[:, 3] - gt[:, 2]).norm(dim=-1)
    return dict(ac_true=act, ac_pred=acp, ac_err=(acp - act).abs(), ac_signed=acp - act,
                tad_err=(tp - tt).abs(), apad_err=(ap - at).abs(), lm_mm=d.mean(-1) * mmpp,
                nme=0.5 * (d[:, :2].mean(-1) / tl + d[:, 2:].mean(-1) / al))


def run_eval(model, df, cfg, tta, bs=8, num_workers=None, device=None):
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    if device is None:
        device = config.DEVICE
    ds = AbdomenDS(df, cfg["img"], False, False, cfg["inpaint"])
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    model.eval()
    rows = []
    for x, gt, mmpp, idx in dl:
        x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
        pred = predict_batch(model, x, tta, device=device)
        m = metrics_rows(pred, gt, mmpp)
        for j in range(len(idx)):
            rows.append({k: float(v[j]) for k, v in m.items()} | {"image_name": df.iloc[int(idx[j])]["image_name"],
                         "source": df.iloc[int(idx[j])]["source"]})
    return pd.DataFrame(rows)


def summarize(r):
    return dict(n=len(r), ac_mae=r.ac_err.mean(), ac_rmse=math.sqrt((r.ac_err ** 2).mean()), ac_median=r.ac_err.median(),
                ac_bias=r.ac_signed.mean(), tad_mae=r.tad_err.mean(), apad_mae=r.apad_err.mean(),
                lm_mm=r.lm_mm.mean(), nme=r.nme.mean(), over10=float((r.ac_err > 10).mean()))


def compute_metrics(y_true, y_pred):
    import numpy as np
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = np.abs(y_pred - y_true)
    sq_err = (y_pred - y_true) ** 2
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    med = float(np.median(err))
    nme = float(np.mean(err / np.maximum(y_true, 1e-6)))
    gt10_rate = float(np.mean(err > 10.0) * 100.0)
    gt10_count = int(np.sum(err > 10.0))
    return {
        "n_samples": len(y_true),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "median": round(med, 4),
        "nme": round(nme, 4),
        "gt10_rate_pct": round(gt10_rate, 2),
        "gt10_count": gt10_count,
    }


def patient_level_paired_bootstrap(
    merged: pd.DataFrame,
    n_bootstraps: int = 5000,
    seed: int = 42,
) -> dict:
    """Compute patient-level cluster paired bootstrap for metric deltas (B - A).

    Args:
        merged: DataFrame with ['patient', 'ac_true', 'ac_pred_a', 'ac_pred_b']
        n_bootstraps: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        dict with point differences, 95% CIs, and empirical p-values.
    """
    import numpy as np

    patients = np.array(sorted(merged["patient"].unique()))
    n_patients = len(patients)

    patient_data = {}
    for p, g in merged.groupby("patient"):
        yt = g["ac_true"].to_numpy(dtype=np.float64)
        ya = g["ac_pred_a"].to_numpy(dtype=np.float64)
        yb = g["ac_pred_b"].to_numpy(dtype=np.float64)
        patient_data[str(p)] = {
            "yt": yt,
            "err_a": np.abs(ya - yt),
            "err_b": np.abs(yb - yt),
            "sq_a": (ya - yt) ** 2,
            "sq_b": (yb - yt) ** 2,
        }

    obs_mae_a = np.mean(np.concatenate([patient_data[p]["err_a"] for p in patients]))
    obs_mae_b = np.mean(np.concatenate([patient_data[p]["err_b"] for p in patients]))
    obs_delta_mae = float(obs_mae_b - obs_mae_a)

    obs_rmse_a = np.sqrt(np.mean(np.concatenate([patient_data[p]["sq_a"] for p in patients])))
    obs_rmse_b = np.sqrt(np.mean(np.concatenate([patient_data[p]["sq_b"] for p in patients])))
    obs_delta_rmse = float(obs_rmse_b - obs_rmse_a)

    rng = np.random.default_rng(seed)
    boot_delta_mae = np.empty(n_bootstraps, dtype=np.float64)
    boot_delta_rmse = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        sampled_p = rng.choice(patients, size=n_patients, replace=True)
        err_a_concat = np.concatenate([patient_data[p]["err_a"] for p in sampled_p])
        err_b_concat = np.concatenate([patient_data[p]["err_b"] for p in sampled_p])
        sq_a_concat = np.concatenate([patient_data[p]["sq_a"] for p in sampled_p])
        sq_b_concat = np.concatenate([patient_data[p]["sq_b"] for p in sampled_p])

        b_mae_a = np.mean(err_a_concat)
        b_mae_b = np.mean(err_b_concat)
        boot_delta_mae[b] = b_mae_b - b_mae_a

        b_rmse_a = np.sqrt(np.mean(sq_a_concat))
        b_rmse_b = np.sqrt(np.mean(sq_b_concat))
        boot_delta_rmse[b] = b_rmse_b - b_rmse_a

    ci_mae_low = float(np.percentile(boot_delta_mae, 2.5))
    ci_mae_high = float(np.percentile(boot_delta_mae, 97.5))
    ci_rmse_low = float(np.percentile(boot_delta_rmse, 2.5))
    ci_rmse_high = float(np.percentile(boot_delta_rmse, 97.5))

    p_mae = float(2.0 * min(np.mean(boot_delta_mae <= 0.0), np.mean(boot_delta_mae >= 0.0)))
    p_mae = min(1.0, p_mae)
    p_rmse = float(2.0 * min(np.mean(boot_delta_rmse <= 0.0), np.mean(boot_delta_rmse >= 0.0)))
    p_rmse = min(1.0, p_rmse)

    return {
        "n_patients": int(n_patients),
        "n_images": int(len(merged)),
        "delta_mae": round(obs_delta_mae, 4),
        "ci95_delta_mae": [round(ci_mae_low, 4), round(ci_mae_high, 4)],
        "p_val_mae": round(p_mae, 4),
        "delta_rmse": round(obs_delta_rmse, 4),
        "ci95_delta_rmse": [round(ci_rmse_low, 4), round(ci_rmse_high, 4)],
        "p_val_rmse": round(p_rmse, 4),
    }


