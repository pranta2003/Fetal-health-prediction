# Original reference code extracted verbatim from Capstone_fetalDataset.ipynb.
# Source:
# - Notebook: Capstone_fetalDataset.ipynb
# - Cell 32: 5-fold EfficientNet-B4 U-Net training, TTA inference, and ellipse fitting
# - Cell 34: Corrected native-resolution evaluation using CSV ground truth

# ==============================================================================
# Cell 32 Functions & Definitions
# ==============================================================================
#Folding with Gaint!!!

#k-fold structure, but resolution 384, efficientnet-b4 encoder, TTA + fitEllipseAMS at eval time,
#and correct OOF scoring (per-image, using only that image's held-out fold model).



# Using K-Fold(5) validation on HC18 — upgraded: higher res, efficientnet-b4, AMP, TTA, fitEllipseAMS, true OOF scoring

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from PIL import Image
import os
import time

DRIVE_ROOT = "/content/drive/MyDrive/fetal-health-project/data/raw/HC18"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
IMG_SIZE = 384          # up from 224 — more spatial precision for HC/BPD measurement
BATCH_SIZE = 8          # lower than before because of bigger image + bigger encoder; drop to 4 if you hit OOM
ENCODER_NAME = "efficientnet-b4"

print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: No GPU detected. Go to Runtime > Change runtime type > GPU, "
          "then reconnect and rerun this cell, or this will be extremely slow.")


class LetterboxResize:
    def __init__(self, target_size=IMG_SIZE):
        self.target_size = target_size

    def __call__(self, image):
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = image.resize((new_w, new_h))
        canvas = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas, scale, paste_x, paste_y


def hc_bpd_from_ellipse_params(semi_major_mm, semi_minor_mm):
    a, b = semi_major_mm, semi_minor_mm
    hc = np.pi * (3*(a+b) - np.sqrt((3*a+b)*(a+3*b)))
    bpd = 2 * min(a, b)
    return hc, bpd


def fit_ellipse_robust(contour):
    """Prefer fitEllipseAMS (more robust to noisy mask boundaries); fall back to fitEllipse."""
    try:
        return cv2.fitEllipseAMS(contour)
    except cv2.error:
        return cv2.fitEllipse(contour)


class HC18SegDataset(Dataset):
    def __init__(self, csv_path, image_dir, augment=False, mask_size=IMG_SIZE):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.augment = augment
        self.letterbox = LetterboxResize(mask_size)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        mask_filename = row["filename"].replace(".png", "_Annotation.png")
        mask_path = os.path.join(self.image_dir, mask_filename)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(raw_mask)
        cv2.drawContours(filled_mask, [largest], -1, 255, thickness=cv2.FILLED)
        mask_pil = Image.fromarray(filled_mask).convert("RGB")

        img_letterboxed, scale, pad_x, pad_y = self.letterbox(image)
        mask_letterboxed, _, _, _ = self.letterbox(mask_pil)

        if self.augment and np.random.rand() > 0.5:
            img_letterboxed = img_letterboxed.transpose(Image.FLIP_LEFT_RIGHT)
            mask_letterboxed = mask_letterboxed.transpose(Image.FLIP_LEFT_RIGHT)

        img_tensor = self.normalize(transforms.ToTensor()(img_letterboxed))
        mask_tensor = transforms.ToTensor()(mask_letterboxed)[0:1]

        return img_tensor, mask_tensor, row["pixel size(mm)"], scale


def build_segmentation_model():
    return smp.Unet(encoder_name=ENCODER_NAME, encoder_weights="imagenet", in_channels=3, classes=1)


def train_one_fold(train_loader, val_loader, fold_num, max_epochs=100, lr=1e-4, patience=8):
    model = build_segmentation_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    dice_loss_fn = smp.losses.DiceLoss(mode="binary")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(max_epochs):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        for images, masks, _, _ in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                preds = model(images)
                loss = dice_loss_fn(preds, masks) + bce_loss_fn(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item()
        train_loss_avg = train_loss_sum / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    preds = model(images)
                    val_loss += (dice_loss_fn(preds, masks) + bce_loss_fn(preds, masks)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        print(f"  Fold {fold_num} | Epoch {epoch+1:3d}/{max_epochs} | "
              f"train_loss={train_loss_avg:.4f} val_loss={val_loss:.4f} | "
              f"{epoch_time:.1f}s | no_improve={epochs_no_improve}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"  Early stopping fold {fold_num} at epoch {epoch+1}")
            break

    print(f"Fold {fold_num}: best val_loss = {best_val_loss:.4f} (stopped at epoch {epoch+1})")
    model.load_state_dict(best_state)
    return model


def predict_prob_with_tta(model, img_tensor):
    """Average prediction on the original image and its horizontal flip (flipped back before averaging)."""
    img_batch = img_tensor.unsqueeze(0).to(DEVICE)
    flipped_batch = torch.flip(img_batch, dims=[3])  # flip width axis

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            prob_orig = torch.sigmoid(model(img_batch)).float().cpu().numpy()[0, 0]
            prob_flipped = torch.sigmoid(model(flipped_batch)).float().cpu().numpy()[0, 0]
    prob_flipped_back = np.flip(prob_flipped, axis=1)  # undo the flip on the prediction
    return (prob_orig + prob_flipped_back) / 2.0


def measure_from_prob(prob_map, mask_tensor, pixel_size, scale):
    binary = (prob_map > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None
    (_, _), (minor_px, major_px), _ = fit_ellipse_robust(largest)
    pred_hc, pred_bpd = hc_bpd_from_ellipse_params(
        (major_px/2)/scale*pixel_size, (minor_px/2)/scale*pixel_size
    )

    true_mask = (mask_tensor[0].numpy() * 255).astype(np.uint8)
    true_contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    true_largest = max(true_contours, key=cv2.contourArea)
    (_, _), (t_minor, t_major), _ = fit_ellipse_robust(true_largest)
    true_hc, true_bpd = hc_bpd_from_ellipse_params(
        (t_major/2)/scale*pixel_size, (t_minor/2)/scale*pixel_size
    )
    return abs(pred_hc - true_hc), abs(pred_bpd - true_bpd)


def evaluate_solo_with_tta(model, dataset, indices):
    """Single model, TTA-averaged, evaluated on given indices (used for each fold's own held-out set)."""
    model.eval()
    hc_errors, bpd_errors = [], []
    for idx in indices:
        img_tensor, mask_tensor, pixel_size, scale = dataset[idx]
        prob_map = predict_prob_with_tta(model, img_tensor)
        result = measure_from_prob(prob_map, mask_tensor, pixel_size, scale)
        if result is None:
            continue
        hc_err, bpd_err = result
        hc_errors.append(hc_err)
        bpd_errors.append(bpd_err)
    return hc_errors, bpd_errors


def evaluate_true_oof(fold_models, fold_val_indices, dataset):
    """
    The ONLY leak-free 'full dataset' number: for every image, score it using
    ONLY the fold model that held it out during training (never averaging in
    models that trained on that image). This is what you report as your final MAE.
    """
    hc_errors, bpd_errors = [], []
    for fold_idx, val_idx in enumerate(fold_val_indices):
        model = fold_models[fold_idx]
        model.eval()
        for idx in val_idx:
            img_tensor, mask_tensor, pixel_size, scale = dataset[idx]
            prob_map = predict_prob_with_tta(model, img_tensor)
            result = measure_from_prob(prob_map, mask_tensor, pixel_size, scale)
            if result is None:
                continue
            hc_err, bpd_err = result
            hc_errors.append(hc_err)
            bpd_errors.append(bpd_err)
    return hc_errors, bpd_errors




# ==============================================================================
# Cell 34 Functions & Definitions
# ==============================================================================

HC_COL = "head circumference (mm)"

def load_raw_filled_mask(row, image_dir=None):
    if image_dir is None:
        image_dir = f"{DRIVE_ROOT}/training_set"
    mask_filename = row["filename"].replace(".png", "_Annotation.png")
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


def predict_native_prob(model, image_pil, target_size=IMG_SIZE):
    orig_w, orig_h = image_pil.size
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image_pil.resize((new_w, new_h))
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    from torchvision import transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = norm(transforms.ToTensor()(canvas)).unsqueeze(0).to(DEVICE)
    flipped_tensor = torch.flip(img_tensor, dims=[3])

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            prob_orig = torch.sigmoid(model(img_tensor)).float().cpu().numpy()[0, 0]
            prob_flip = torch.sigmoid(model(flipped_tensor)).float().cpu().numpy()[0, 0]
    prob_flip_back = np.flip(prob_flip, axis=1)
    prob_avg = (prob_orig + prob_flip_back) / 2.0

    cropped = prob_avg[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    native_prob = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return native_prob


if __name__ == "__main__":
    # ---- Setup: 5-fold cross-validation ----

    val_dataset_ref = HC18SegDataset(
        csv_path=f"{DRIVE_ROOT}/training_set_pixel_size_and_HC.csv",
        image_dir=f"{DRIVE_ROOT}/training_set", augment=False
    )
    train_dataset_ref = HC18SegDataset(
        csv_path=f"{DRIVE_ROOT}/training_set_pixel_size_and_HC.csv",
        image_dir=f"{DRIVE_ROOT}/training_set", augment=True
    )

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    all_indices = np.arange(len(val_dataset_ref))

    fold_models = []
    fold_val_indices = []
    all_hc_errors_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")
        train_subset = Subset(train_dataset_ref, train_idx)
        val_subset = Subset(val_dataset_ref, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = train_one_fold(train_loader, val_loader, fold+1)
        fold_models.append(model)
        fold_val_indices.append(val_idx)

        # Solo fold performance, now with TTA
        hc_err, bpd_err = evaluate_solo_with_tta(model, val_dataset_ref, val_idx)
        print(f"Fold {fold+1} solo (TTA) -> HC MAE: {np.mean(hc_err):.2f}mm, BPD MAE: {np.mean(bpd_err):.2f}mm")
        all_hc_errors_per_fold.append(np.mean(hc_err))

        torch.save(model.state_dict(),
                   f"/content/drive/MyDrive/fetal-health-project/models/hc18_seg_fold{fold+1}_eff_b4_384.pt")

    print(f"\n===== Cross-validation summary =====")
    print(f"Per-fold HC MAE: {[f'{x:.2f}' for x in all_hc_errors_per_fold]}")
    print(f"Mean across folds: {np.mean(all_hc_errors_per_fold):.2f}mm ± {np.std(all_hc_errors_per_fold):.2f}mm")

    # ---- The number to actually report: true out-of-fold MAE over the whole dataset ----
    print("\n===== TRUE out-of-fold evaluation (no leakage, whole dataset, TTA) =====")
    hc_errors_oof, bpd_errors_oof = evaluate_true_oof(fold_models, fold_val_indices, val_dataset_ref)
    print(f"[OOF] HC  -> MAE: {np.mean(hc_errors_oof):.2f}mm, RMSE: {np.sqrt(np.mean(np.array(hc_errors_oof)**2)):.2f}mm")
    print(f"[OOF] BPD -> MAE: {np.mean(bpd_errors_oof):.2f}mm, RMSE: {np.sqrt(np.mean(np.array(bpd_errors_oof)**2)):.2f}mm")
