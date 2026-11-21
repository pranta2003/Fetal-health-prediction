# AC sub-model v2 - DSNT-style landmark regression for TAD/APAD (FP + UCL pooled)
# v1 (sigmoid heatmaps + MSE) collapsed to "predict zeros". v2 decodes coordinates with a
# spatial softmax and trains on the coordinates directly, so that collapse is impossible.

import os, gc, shutil, time
import numpy as np
import pandas as pd
import cv2
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from PIL import Image

SMOKE_TEST = False   # True = few-minute sanity run on a tiny subset. Set False for the real run.

BASE = "D:/Ishan_390/nor/fetal-project/data/raw/FETAL_BIOMETRY_ABDOMEN"
MODEL_DIR = "C:/fetal-project/models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
IMG_SIZE = 384
BATCH_SIZE = 4
ENCODER_NAME = "efficientnet-b4"
N_FOLDS = 5
SIGMA = 4.0          # width of the Gaussian the JS regulariser pulls each heatmap toward
JS_WEIGHT = 1.0
LR = 1e-4
MAX_EPOCHS = 5 if SMOKE_TEST else 100
PATIENCE = 10
RUN_TAG = "smoke" if SMOKE_TEST else "full"
CKPT_DIR = f"{MODEL_DIR}/ac_dsnt_{RUN_TAG}"

if SMOKE_TEST and os.path.isdir(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)          # smoke runs always start fresh
os.makedirs(CKPT_DIR, exist_ok=True)
torch.backends.cudnn.benchmark = True
print(f"Using device: {DEVICE} | SMOKE_TEST={SMOKE_TEST} | checkpoints -> {CKPT_DIR}")

# ==== NUMPY HELPERS BEGIN ====
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LANDMARK_NAMES = ["tad_1", "tad_2", "apad_1", "apad_2"]
# Allowed re-labelings of the 4 points: swap endpoints inside a pair, and/or swap the two pairs.
PERMS_NP = np.array([[0, 1, 2, 3], [1, 0, 2, 3], [0, 1, 3, 2], [1, 0, 3, 2],
                     [2, 3, 0, 1], [3, 2, 0, 1], [2, 3, 1, 0], [3, 2, 1, 0]])


def letterbox_transform(image_pil, target_size=IMG_SIZE):
    w, h = image_pil.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image_pil.resize((new_w, new_h))
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def match_points_np(ref, other):
    """Re-order `other` (4,2) with the allowed permutation that best matches `ref` (4,2)."""
    costs = [np.linalg.norm(ref - other[p], axis=1).sum() for p in PERMS_NP]
    return other[PERMS_NP[int(np.argmin(costs))]]


def augment_sample(arr, pts, rng=None):
    """arr: HxWx3 uint8 (letterboxed). pts: (4,2) pixel coords in the same frame.
    Returns float32 image in [0,1] and the transformed points."""
    rng = rng if rng is not None else np.random.default_rng()
    pts = np.asarray(pts, dtype=np.float32).copy()
    if rng.random() < 0.5:                                   # horizontal flip
        arr = arr[:, ::-1]
        pts[:, 0] = (IMG_SIZE - 1) - pts[:, 0]
    arr = np.ascontiguousarray(arr)
    if rng.random() < 0.7:                                   # rotation / scale / shift
        c = (IMG_SIZE - 1) / 2.0
        for _ in range(5):
            M = cv2.getRotationMatrix2D((c, c), rng.uniform(-20, 20), rng.uniform(0.9, 1.1))
            M[:, 2] += rng.uniform(-0.04, 0.04, 2) * IMG_SIZE
            new = pts @ M[:, :2].T + M[:, 2]
            if (new >= 2).all() and (new <= IMG_SIZE - 3).all():   # keep every landmark inside
                arr = cv2.warpAffine(arr, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                pts = new.astype(np.float32)
                break
    img = arr.astype(np.float32) / 255.0
    if rng.random() < 0.8:                                   # gamma / contrast / brightness
        img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
        img = np.clip(img * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08), 0, 1)
    return img, pts


def normalize_chw(img01):
    return np.ascontiguousarray(((img01 - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)


def ac_from_diameters(d1_mm, d2_mm):
    a, b = d1_mm / 2.0, d2_mm / 2.0      # diameters -> semi-axes (same function as HC18)
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


def diameters_mm(pts, rate):
    return (np.linalg.norm(pts[0] - pts[1]) * rate, np.linalg.norm(pts[2] - pts[3]) * rate)
# ==== NUMPY HELPERS END ====


# ---------------------------------------------------------------------------
# Step 1: pooled dataframe (patient ids come from the FILENAME, not the SubjectID column)
# ---------------------------------------------------------------------------
fp_df = pd.read_csv(f"{BASE}/FP/annotations/Abdomen.csv")
fp_df["image_path"] = fp_df["image_name"].apply(lambda f: f"{BASE}/FP/images/Abdomen/{f}")
fp_df["subject_id"] = "FP_" + fp_df["image_name"].str.extract(r"(Patient\d+)")[0]

ucl_df = pd.read_csv(f"{BASE}/UCL/annotations/px_to_mm/Abdomen.csv")
ucl_df = ucl_df[~ucl_df["image_name"].str.startswith("125_")].reset_index(drop=True)
ucl_df["image_path"] = ucl_df["image_name"].apply(lambda f: f"{BASE}/UCL/images/Abdomen/{f}")
ucl_df["subject_id"] = "UCL_" + ucl_df["image_name"].str.extract(r"^(\d+)_")[0]

assert fp_df["subject_id"].notna().all() and ucl_df["subject_id"].notna().all(), "subject_id extraction failed"
KEEP_COLS = ["image_path", "subject_id"] + [f"{n}_{c}" for n in LANDMARK_NAMES for c in "xy"] + ["px_to_mm_rate"]
data = pd.concat([fp_df[KEEP_COLS], ucl_df[KEEP_COLS]], ignore_index=True)
assert data["image_path"].apply(os.path.isfile).all(), "Some image paths don't exist."
assert data["px_to_mm_rate"].notna().all(), "Some rows have no calibration."
print(f"Full pooled dataset: {len(data)} images, {data['subject_id'].nunique()} patients")

if SMOKE_TEST:   # tiny subset, NOT used for the real run
    fp_part = data[data["subject_id"].str.startswith("FP_")].sample(120, random_state=0)
    ucl_part = data[data["subject_id"].str.startswith("UCL_")].sample(40, random_state=0)
    data = pd.concat([fp_part, ucl_part]).reset_index(drop=True)
    print(f"SMOKE subset: {len(data)} images")

# NOTE: row order of `data` decides which patient lands in which fold. Don't change it between sessions.
CONFIG = dict(img_size=IMG_SIZE, encoder=ENCODER_NAME, sigma=SIGMA, js_weight=JS_WEIGHT,
              n_images=len(data), n_folds=N_FOLDS, smoke=SMOKE_TEST)


class AbdomenLandmarkDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lb, scale, pad_x, pad_y = letterbox_transform(Image.open(row["image_path"]).convert("RGB"))
        arr = np.asarray(lb)
        pts = np.array([[row[f"{n}_x"] * scale + pad_x, row[f"{n}_y"] * scale + pad_y]
                        for n in LANDMARK_NAMES], dtype=np.float32)
        if self.augment:
            img01, pts = augment_sample(arr, pts)
        else:
            img01 = arr.astype(np.float32) / 255.0
        return torch.from_numpy(normalize_chw(img01)), torch.from_numpy(pts)


# ---------------------------------------------------------------------------
# Step 2: model + DSNT-style decoding and losses (all float32)
# ---------------------------------------------------------------------------
def build_landmark_model():
    return smp.Unet(encoder_name=ENCODER_NAME, encoder_weights="imagenet", in_channels=3, classes=4)


PERMS_T = torch.from_numpy(PERMS_NP).long().to(DEVICE)
XG = torch.arange(IMG_SIZE, dtype=torch.float32, device=DEVICE).repeat(IMG_SIZE)               # x of flat index
YG = torch.arange(IMG_SIZE, dtype=torch.float32, device=DEVICE).repeat_interleave(IMG_SIZE)    # y of flat index


def decode_heatmaps(logits):
    """logits (B,4,H,W) -> spatial-softmax probabilities (B,4,HW) and expected (x,y) in pixels (B,4,2)."""
    B, K, H, W = logits.shape
    p = torch.softmax(logits.float().reshape(B, K, H * W), dim=-1)
    coords = torch.stack([(p * XG).sum(-1), (p * YG).sum(-1)], dim=-1)
    return coords, p


def match_gt(pred, gt):
    """For each sample pick the allowed re-labeling of gt (B,4,2) that best matches pred (B,4,2)."""
    gt_perm = gt[:, PERMS_T]                                              # (B,8,4,2)
    d = torch.sqrt(((pred[:, None] - gt_perm) ** 2).sum(-1) + 1e-8)       # (B,8,4)
    best = d.sum(-1).argmin(dim=1)
    return gt_perm[torch.arange(pred.shape[0], device=pred.device), best]  # (B,4,2)


def gaussian_targets(gt):
    gx, gy = gt[..., 0:1], gt[..., 1:2]                                   # (B,4,1)
    q = torch.exp(-((XG - gx) ** 2 + (YG - gy) ** 2) / (2 * SIGMA ** 2))  # (B,4,HW)
    return q / q.sum(-1, keepdim=True).clamp_min(1e-12)


def js_div(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def compute_losses(logits, gt):
    coords, p = decode_heatmaps(logits)
    matched = match_gt(coords.detach(), gt)
    dist = torch.sqrt(((coords - matched) ** 2).sum(-1) + 1e-8)           # (B,4) pixels
    coord_loss = (dist / ((IMG_SIZE - 1) / 2.0)).mean()
    js = js_div(p, gaussian_targets(matched)).mean()
    return coord_loss + JS_WEIGHT * js, dist.detach().mean().item()


def run_validation(model, loader):
    model.eval()
    tot_loss = tot_px = base_px = n = 0.0
    center = (IMG_SIZE - 1) / 2.0
    with torch.no_grad():
        for images, gts in loader:
            images, gts = images.to(DEVICE, non_blocking=True), gts.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logits = model(images)
            loss, px = compute_losses(logits, gts)
            b = images.shape[0]
            tot_loss += loss.item() * b
            tot_px += px * b
            base_px += (gts - center).norm(dim=-1).mean().item() * b   # "all points at the centre" baseline
            n += b
    return tot_loss / n, tot_px / n, base_px / n


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)     # a cutoff mid-write can no longer corrupt the checkpoint


def train_one_fold(train_loader, val_loader, fold_num):
    last_p = f"{CKPT_DIR}/fold{fold_num}_last.pt"
    best_p = f"{CKPT_DIR}/fold{fold_num}_best.pt"
    model = build_landmark_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    start_epoch, best_px, no_improve = 0, float("inf"), 0

    if os.path.isfile(last_p):
        ck = torch.load(last_p, map_location=DEVICE)
        if ck["config"] != CONFIG:
            raise RuntimeError(f"Checkpoint was made with a different config. Delete the files in {CKPT_DIR}.")
        model.load_state_dict(ck["model_state"]); optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"]); scaler.load_state_dict(ck["scaler_state"])
        start_epoch, best_px, no_improve = ck["epoch"] + 1, ck["best_px"], ck["no_improve"]
        print(f"  Resumed fold {fold_num} at epoch {start_epoch} (best val landmark error {best_px:.2f}px)")

    for epoch in range(start_epoch, MAX_EPOCHS):
        if no_improve >= PATIENCE:
            break
        t0 = time.time()
        model.train()
        loss_sum = px_sum = n = 0.0
        for images, gts in train_loader:
            images, gts = images.to(DEVICE, non_blocking=True), gts.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logits = model(images)
            loss, px = compute_losses(logits, gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            b = images.shape[0]
            loss_sum += loss.item() * b; px_sum += px * b; n += b
        val_loss, val_px, base_px = run_validation(model, val_loader)
        scheduler.step(val_px)
        print(f"  Fold {fold_num} | Ep {epoch+1:3d}/{MAX_EPOCHS} | train_loss={loss_sum/n:.4f} train_px={px_sum/n:6.2f} | "
              f"val_loss={val_loss:.4f} val_px={val_px:6.2f} (centre-baseline {base_px:.1f}) | {time.time()-t0:.0f}s | no_improve={no_improve}")
        if val_px < best_px - 1e-3:
            best_px, no_improve = val_px, 0
            atomic_save({"model_state": model.state_dict(), "val_px": val_px, "epoch": epoch}, best_p)
        else:
            no_improve += 1
        atomic_save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                     "epoch": epoch, "best_px": best_px, "no_improve": no_improve, "config": CONFIG}, last_p)

    model.load_state_dict(torch.load(best_p, map_location=DEVICE)["model_state"])
    return model, best_px


# ---------------------------------------------------------------------------
# Step 3: evaluation - flip-TTA with label matching, native resolution, metrics in mm
# ---------------------------------------------------------------------------
def predict_points_tta(model, image_pil):
    lb, scale, pad_x, pad_y = letterbox_transform(image_pil)
    x = torch.from_numpy(normalize_chw(np.asarray(lb).astype(np.float32) / 255.0)).unsqueeze(0).to(DEVICE)
    batch = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            logits = model(batch)
        coords, _ = decode_heatmaps(logits)
    c = coords.cpu().numpy()
    c_orig, c_flip = c[0], c[1].copy()
    c_flip[:, 0] = (IMG_SIZE - 1) - c_flip[:, 0]            # undo the flip
    c_flip = match_points_np(c_orig, c_flip)                # channels may be labelled differently -> match first
    c_avg = 0.5 * (c_orig + c_flip)
    return (c_avg - np.array([pad_x, pad_y], dtype=np.float32)) / scale   # native pixel coordinates


def evaluate_oof(fold_models, fold_val_indices, df):
    rows = []
    for model, val_idx in zip(fold_models, fold_val_indices):
        for idx in val_idx:
            row = df.iloc[idx]
            pred = predict_points_tta(model, Image.open(row["image_path"]).convert("RGB"))
            gt = np.array([[row[f"{n}_x"], row[f"{n}_y"]] for n in LANDMARK_NAMES], dtype=np.float32)
            rate = row["px_to_mm_rate"]
            pd1, pd2 = diameters_mm(pred, rate)
            gd1, gd2 = diameters_mm(gt, rate)
            p_sorted, g_sorted = np.sort([pd1, pd2]), np.sort([gd1, gd2])
            rows.append(dict(source=row["subject_id"].split("_")[0],
                             ac_true=ac_from_diameters(gd1, gd2), ac_pred=ac_from_diameters(pd1, pd2),
                             major_err=abs(p_sorted[1] - g_sorted[1]), minor_err=abs(p_sorted[0] - g_sorted[0]),
                             lm_err_mm=np.linalg.norm(gt - match_points_np(gt, pred), axis=1).mean() * rate))
    res = pd.DataFrame(rows)
    err = (res["ac_pred"] - res["ac_true"]).abs()
    base = (res["ac_true"] - res["ac_true"].mean()).abs().mean()
    print(f"n = {len(res)} images")
    print(f"AC  -> MAE {err.mean():.2f} mm | RMSE {np.sqrt((err**2).mean()):.2f} mm | mean abs % error {(err/res['ac_true']*100).mean():.2f}%")
    print(f"       (reference: always predicting the mean AC gives MAE {base:.2f} mm)")
    print(f"Larger diameter MAE {res['major_err'].mean():.2f} mm | smaller diameter MAE {res['minor_err'].mean():.2f} mm")
    print(f"Mean landmark error {res['lm_err_mm'].mean():.2f} mm")
    for src in sorted(res["source"].unique()):
        m = res["source"] == src
        print(f"  {src}: n={m.sum()}, AC MAE {err[m].mean():.2f} mm")
    return res


# ---------------------------------------------------------------------------
# Step 4: fold loop (resumable; skips folds that finished in an earlier session)
# ---------------------------------------------------------------------------
gkf = GroupKFold(n_splits=N_FOLDS)
folds_to_run = [0] if SMOKE_TEST else list(range(N_FOLDS))
fold_models, fold_val_indices = [], []

for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(data)), groups=data["subject_id"].values)):
    if fold not in folds_to_run:
        continue
    fold_num = fold + 1
    final_p = f"{CKPT_DIR}/fold{fold_num}_final.pt"
    print(f"\n===== Fold {fold_num}/{N_FOLDS} | train patients {data.iloc[train_idx]['subject_id'].nunique()}, "
          f"val patients {data.iloc[val_idx]['subject_id'].nunique()} =====")
    fold_val_indices.append(val_idx)

    if os.path.isfile(final_p):
        ck = torch.load(final_p, map_location=DEVICE)
        if ck.get("config") != CONFIG or not ck.get("completed"):
            raise RuntimeError(f"{final_p} does not match this run. Delete the files in {CKPT_DIR}.")
        print(f"  Already completed (best val landmark error {ck['best_px']:.2f}px) - loading.")
        model = build_landmark_model().to(DEVICE)
        model.load_state_dict(ck["model_state"])
        fold_models.append(model)
        continue

    train_loader = DataLoader(Subset(AbdomenLandmarkDataset(data, augment=True), train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(AbdomenLandmarkDataset(data, augment=False), val_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    model, best_px = train_one_fold(train_loader, val_loader, fold_num)
    atomic_save({"model_state": model.state_dict(), "best_px": best_px, "completed": True, "config": CONFIG}, final_p)
    fold_models.append(model)
    del train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()

print("\n===== Out-of-fold evaluation (patient-disjoint, flip-TTA, native resolution, mm) =====")
oof_results = evaluate_oof(fold_models, fold_val_indices, data)
oof_results.to_csv(f"{CKPT_DIR}/oof_results.csv", index=False)
