# FIELD: AC
# MODEL: EfficientNet-B4 Landmark U-Net
# STEP: dataset_loading
# USED BY: pipeline imports across work/scripts and work/tests

import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fetal import config
from fetal.data.transforms import (
    MEAN,
    STD,
    letterbox,
    photometric,
    random_affine,
    random_erase,
    read_rgb,
)

LM = ["tad_1", "tad_2", "apad_1", "apad_2"]
COORD_COLS = [f"{n}_{a}" for n in LM for a in "xy"]
DEV_FRACTION = 0.12


def read_names(path):
    if not os.path.isfile(path):
        return None
    d = pd.read_csv(path)
    return set(d["image_name" if "image_name" in d.columns else d.columns[0]].astype(str))


def load_source(src, ann_dir, img_dir):
    df = pd.read_csv(f"{ann_dir}/Abdomen.csv")
    if src == "UCL":
        df = df[~df["image_name"].str.startswith("125_")]
    df = df.reset_index(drop=True)
    df["source"] = src
    df["image_path"] = df["image_name"].apply(lambda f: f"{img_dir}/{f}")
    if src == "FP":
        df["patient"] = "FP_" + df["image_name"].apply(lambda f: re.search(r"(Patient\d+)", f).group(1))
    else:
        df["patient"] = "UCL_" + df["image_name"].apply(lambda f: re.match(r"(\d+)_", f).group(1))
    te, tr = read_names(f"{ann_dir}/Abdomen_Test.csv"), read_names(f"{ann_dir}/Abdomen_Train.csv")
    if te is not None:
        df["official_split"] = df["image_name"].astype(str).apply(lambda f: "test" if f in te else "train")
    elif tr is not None:
        df["official_split"] = df["image_name"].astype(str).apply(lambda f: "train" if f in tr else "test")
    else:
        raise RuntimeError(f"{src}: no Abdomen_Train.csv / Abdomen_Test.csv found in {ann_dir}")
    return df[["image_name", "image_path", "source", "patient", "official_split", "px_to_mm_rate"] + COORD_COLS]


def build_data(base=None):
    if base is None:
        base = config.DATA_DIR
    fp = load_source("FP", f"{base}/FP/annotations", f"{base}/FP/images/Abdomen")
    ucl = load_source("UCL", f"{base}/UCL/annotations/px_to_mm", f"{base}/UCL/images/Abdomen")
    data = pd.concat([fp, ucl], ignore_index=True)
    missing = [p for p in data["image_path"] if not os.path.isfile(p)]
    assert not missing, f"{len(missing)} image files missing, e.g. {missing[:3]}"
    assert data["px_to_mm_rate"].notna().all()
    return data


def split_data(data, dev_fraction=DEV_FRACTION):
    train_all = data[data.official_split == "train"]
    pats = np.array(sorted(train_all.patient.unique()))
    rng = np.random.RandomState(0)
    dev_p = set(rng.permutation(pats)[: max(1, int(round(dev_fraction * len(pats))))])
    tr = train_all[~train_all.patient.isin(dev_p)].reset_index(drop=True)
    dv = train_all[train_all.patient.isin(dev_p)].reset_index(drop=True)
    te = data[data.official_split == "test"].reset_index(drop=True)
    assert not (set(tr.patient) & set(dv.patient)) and not (set(tr.patient) & set(te.patient)), "patient overlap!"
    return tr, dv, te


def load_ac_folds(folds_csv=None, base=None):
    """
    Load the fixed 5-fold assignment and merge with dataset annotations.

    Returns DataFrame with official train images (642 rows), patient, fold (1..5),
    pixel-to-mm rate, and landmark coordinate columns.
    """
    if folds_csv is None:
        folds_csv = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "docs",
            "results",
            "ac_campaign",
            "folds_v1.csv",
        )
    if not os.path.isfile(folds_csv):
        raise FileNotFoundError(f"Folds CSV not found: {folds_csv}")

    folds_df = pd.read_csv(folds_csv)
    data = build_data(base=base)
    train_data = data[data.official_split == "train"].copy()

    merged = pd.merge(
        train_data,
        folds_df[["filename", "fold"]],
        left_on="image_name",
        right_on="filename",
        how="inner",
    )
    assert len(merged) == len(folds_df), (
        f"Mismatch after merging folds: expected {len(folds_df)}, got {len(merged)}"
    )
    return merged


def get_fold_data(df, fold, inner_val_fraction=DEV_FRACTION, seed=42):
    """
    Partition 5-fold training set for fold K cross-validation.

    - te: Fold K images (out-of-fold validation/test)
    - tr: Remaining 4 folds minus a patient-grouped inner validation holdout
    - dv: Patient-grouped inner validation holdout from the 4 training folds (used for epoch selection)

    Asserts zero patient leakage between tr, dv, and te.
    """
    te = df[df.fold == fold].reset_index(drop=True)
    remaining = df[df.fold != fold].reset_index(drop=True)

    pats = np.array(sorted(remaining.patient.unique()))
    rng = np.random.RandomState(seed + fold)
    n_dev_pats = max(1, int(round(inner_val_fraction * len(pats))))
    dev_p = set(rng.permutation(pats)[:n_dev_pats])

    tr = remaining[~remaining.patient.isin(dev_p)].reset_index(drop=True)
    dv = remaining[remaining.patient.isin(dev_p)].reset_index(drop=True)

    # Strict leakage checks
    assert not (set(tr.patient) & set(dv.patient)), "Leakage between inner-train and inner-val!"
    assert not (set(tr.patient) & set(te.patient)), "Leakage between inner-train and outer-fold!"
    assert not (set(dv.patient) & set(te.patient)), "Leakage between inner-val and outer-fold!"

    return tr, dv, te


class AbdomenDS(Dataset):
    def __init__(self, df, S, train, strong, inpaint):
        self.df, self.S, self.train, self.strong, self.inpaint = df.reset_index(drop=True), S, train, strong, inpaint

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        S = self.S
        img = read_rgb(r["image_path"], self.inpaint)
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
        mmpp = float(r["px_to_mm_rate"]) / scale   # mm per pixel in the S x S training image (scale-aware)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.tensor(pts, dtype=torch.float32),
                torch.tensor(mmpp, dtype=torch.float32), i)

