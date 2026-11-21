# AC sub-model v3 - landmark regression (spatial-softmax) with optional diameter/AC loss, strong augmentation
# and burned-in-marker removal. Evaluated on the OFFICIAL Train/Test split (no leakage):
#   train on official-train patients minus a 12% patient hold-out ("dev", used only for early stopping / model choice),
#   report ONCE on the official test images.
# Several configs run one after the other; each is resumable (kill it any time, run again, it continues).
# Run (lab PC, venv active):   python -u ac_v3.py        Colab: paste in one cell (after pip install segmentation-models-pytorch==0.5.0)
import os, re, sys, time, math, json, random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

try:  # Colab only
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception:
    pass

IS_WIN = os.name == "nt"
BASE = os.environ.get("DATA_BASE", "D:/Ishan_390/nor/fetal-project/data/raw/FETAL_BIOMETRY_ABDOMEN" if IS_WIN
                      else "/content/drive/MyDrive/fetal-health-project/data/raw/FETAL_BIOMETRY_ABDOMEN")
OUT = os.environ.get("OUT_DIR", "C:/fetal-project/models/ac_v3" if IS_WIN
                     else "/content/drive/MyDrive/fetal-health-project/models/ac_v3")
SMOKE = os.environ.get("AC_SMOKE") == "1"           # my own sandbox test only
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if IS_WIN else 2
MAX_EPOCHS = 2 if SMOKE else 80
PATIENCE = 12
BATCH = 4
LR = 2e-4
SIGMA = 4.0          # heatmap regulariser width (px at 384; scaled with image size)
DEV_FRACTION = 0.12

# configs, run in this order. "base" ~ the old recipe (flip + brightness only, no diameter loss, markers kept)
RUNS = [
    dict(name="base",     img=384, diam_w=0.0, strong_aug=False, inpaint=False),
    dict(name="v3_full",  img=384, diam_w=1.0, strong_aug=True,  inpaint=True),
]
if SMOKE:
    for r in RUNS:
        r["img"] = 128

LM = ["tad_1", "tad_2", "apad_1", "apad_2"]
COORD_COLS = [f"{n}_{a}" for n in LM for a in "xy"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PERMS = []   # 8 allowed relabelings: swap ends inside a pair, swap the two pairs
for sp in (0, 1):
    for s0 in (0, 1):
        for s1 in (0, 1):
            a = [1, 0] if s0 else [0, 1]
            b = [3, 2] if s1 else [2, 3]
            PERMS.append(b + a if sp else a + b)
PERMS_T = torch.tensor(PERMS, dtype=torch.long)


# ------------------------------------------------------------------------------------------------ data
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


def build_data():
    fp = load_source("FP", f"{BASE}/FP/annotations", f"{BASE}/FP/images/Abdomen")
    ucl = load_source("UCL", f"{BASE}/UCL/annotations/px_to_mm", f"{BASE}/UCL/images/Abdomen")
    data = pd.concat([fp, ucl], ignore_index=True)
    missing = [p for p in data["image_path"] if not os.path.isfile(p)]
    assert not missing, f"{len(missing)} image files missing, e.g. {missing[:3]}"
    assert data["px_to_mm_rate"].notna().all()
    return data


def split_data(data):
    train_all = data[data.official_split == "train"]
    pats = np.array(sorted(train_all.patient.unique()))
    rng = np.random.RandomState(0)
    dev_p = set(rng.permutation(pats)[: max(1, int(round(DEV_FRACTION * len(pats))))])
    tr = train_all[~train_all.patient.isin(dev_p)].reset_index(drop=True)
    dv = train_all[train_all.patient.isin(dev_p)].reset_index(drop=True)
    te = data[data.official_split == "test"].reset_index(drop=True)
    assert not (set(tr.patient) & set(dv.patient)) and not (set(tr.patient) & set(te.patient)), "patient overlap!"
    return tr, dv, te


# ------------------------------------------------------------------------------------------------ image ops
def remove_yellow(img):
    """burned-in yellow calipers / labels -> inpaint (applied identically in train and test)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 18) & (hsv[..., 0] <= 38) & (hsv[..., 1] >= 100) & (hsv[..., 2] >= 120)).astype(np.uint8)
    if m.sum() < 20:
        return img
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)


def read_rgb(path, inpaint):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return remove_yellow(img) if inpaint else img


def letterbox(img, S):
    h, w = img.shape[:2]
    scale = S / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = np.zeros((S, S, 3), np.uint8)
    px, py = (S - nw) // 2, (S - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
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


# ------------------------------------------------------------------------------------------------ model / loss
def build_model():
    return smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None if SMOKE else "imagenet", in_channels=3, classes=4)


def decode(logits):
    B, C, H, W = logits.shape
    p = torch.softmax(logits.float().reshape(B, C, -1), dim=-1)
    pm = p.reshape(B, C, H, W)
    xs = torch.arange(W, device=logits.device, dtype=torch.float32)
    ys = torch.arange(H, device=logits.device, dtype=torch.float32)
    return torch.stack([(pm.sum(2) * xs).sum(-1), (pm.sum(3) * ys).sum(-1)], -1), p


def match(pred, gt):
    """pred, gt (B,4,2). returns pred reordered to gt's labelling (differentiable gather) and the order used"""
    B = pred.shape[0]
    P = PERMS_T.to(pred.device)
    pp = pred[:, P]                                            # (B,8,4,2)
    d = (pp - gt[:, None]).norm(dim=-1).mean(-1)               # (B,8)
    idx = d.argmin(1).detach()
    return pp[torch.arange(B, device=pred.device), idx], P[idx]


def ac_t(tad_mm, apad_mm):
    a, b = tad_mm / 2, apad_mm / 2
    return math.pi * (3 * (a + b) - torch.sqrt((3 * a + b) * (a + 3 * b) + 1e-6))


def lengths_mm(pts, mmpp):
    return (pts[:, 1] - pts[:, 0]).norm(dim=-1) * mmpp, (pts[:, 3] - pts[:, 2]).norm(dim=-1) * mmpp


def js_loss(p, order, gt, S, sigma):
    B = p.shape[0]
    tgt_pts = torch.zeros_like(gt)
    tgt_pts.scatter_(1, order.unsqueeze(-1).expand(B, 4, 2), gt)     # target for pred channel order[b,i] is gt[b,i]
    xs = torch.arange(S, device=p.device, dtype=torch.float32).view(1, 1, 1, S)
    ys = torch.arange(S, device=p.device, dtype=torch.float32).view(1, 1, S, 1)
    q = torch.exp(-((xs - tgt_pts[..., 0, None, None]) ** 2 + (ys - tgt_pts[..., 1, None, None]) ** 2) / (2 * sigma ** 2))
    q = (q.reshape(B, 4, -1) + 1e-12)
    q = q / q.sum(-1, keepdim=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + 1e-12) - torch.log(b + 1e-12))).sum(-1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean()


def compute_loss(logits, gt, mmpp, S, diam_w):
    pts, p = decode(logits)
    matched, order = match(pts, gt)
    coord = (matched - gt).norm(dim=-1).mean() / 10.0
    js = js_loss(p, order, gt, S, SIGMA * S / 384.0)
    loss = coord + js
    if diam_w > 0:
        tp, ap = lengths_mm(matched, mmpp)
        tt, at = lengths_mm(gt, mmpp)
        l_len = ((tp - tt).abs() + (ap - at).abs()).mean() / 10.0
        l_ac = (ac_t(tp, ap) - ac_t(tt, at)).abs().mean() / 10.0
        loss = loss + diam_w * (l_len + l_ac)
    return loss, coord.item() * 10.0


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def predict_batch(model, x, tta):
    with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
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


def run_eval(model, df, cfg, tta, bs=8):
    ds = AbdomenDS(df, cfg["img"], False, False, cfg["inpaint"])
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    rows = []
    for x, gt, mmpp, idx in dl:
        x, gt, mmpp = x.to(DEVICE), gt.to(DEVICE), mmpp.to(DEVICE)
        pred = predict_batch(model, x, tta)
        m = metrics_rows(pred, gt, mmpp)
        for j in range(len(idx)):
            rows.append({k: float(v[j]) for k, v in m.items()} | {"image_name": df.iloc[int(idx[j])]["image_name"],
                         "source": df.iloc[int(idx[j])]["source"]})
    return pd.DataFrame(rows)


def summarize(r):
    return dict(n=len(r), ac_mae=r.ac_err.mean(), ac_rmse=math.sqrt((r.ac_err ** 2).mean()), ac_median=r.ac_err.median(),
                ac_bias=r.ac_signed.mean(), tad_mae=r.tad_err.mean(), apad_mae=r.apad_err.mean(),
                lm_mm=r.lm_mm.mean(), nme=r.nme.mean(), over10=float((r.ac_err > 10).mean()))


# ------------------------------------------------------------------------------------------------ training
def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_one(cfg, tr, dv, te):
    d = f"{OUT}/{cfg['name']}"
    os.makedirs(d, exist_ok=True)
    res_path = f"{d}/results.json"
    if os.path.isfile(res_path):
        print(f"[{cfg['name']}] already finished -> skipping")
        return json.load(open(res_path))
    S = cfg["img"]
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    model = build_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == "cuda"))
    start, best, bad = 0, float("inf"), 0
    last = f"{d}/last.pt"
    if os.path.isfile(last):
        ck = torch.load(last, map_location=DEVICE, weights_only=False)
        assert ck["cfg"] == cfg, f"checkpoint config differs from {cfg} - delete {d} to start over"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"[{cfg['name']}] resumed at epoch {start}, best dev score {best:.3f}")
    trd = tr.head(24) if SMOKE else tr
    dl = DataLoader(AbdomenDS(trd, S, True, cfg["strong_aug"], cfg["inpaint"]), batch_size=BATCH, shuffle=True,
                    num_workers=NUM_WORKERS, drop_last=True)
    for ep in range(start, MAX_EPOCHS):
        t0 = time.time()
        model.train()
        tl = 0.0
        for x, gt, mmpp, _ in dl:
            x, gt, mmpp = x.to(DEVICE), gt.to(DEVICE), mmpp.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                logits = model(x)
            loss, _ = compute_loss(logits, gt, mmpp, S, cfg["diam_w"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl += loss.item()
        r = run_eval(model, dv.head(16) if SMOKE else dv, cfg, tta=False)
        s = summarize(r)
        score = s["ac_mae"] + s["lm_mm"]
        sched.step(score)
        improved = score < best
        if improved:
            best, bad = score, 0
            atomic_save(model.state_dict(), f"{d}/best.pt")
        else:
            bad += 1
        print(f"[{cfg['name']}] ep {ep+1:3d}/{MAX_EPOCHS} loss={tl/len(dl):.4f} | dev AC MAE {s['ac_mae']:.2f}mm "
              f"lm {s['lm_mm']:.2f}mm bias {s['ac_bias']:+.2f} | score {score:.3f}{' *' if improved else ''} | "
              f"lr {opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s | bad={bad}", flush=True)
        atomic_save(dict(cfg=cfg, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                         scaler=scaler.state_dict(), epoch=ep, best=best, bad=bad), last)
        if bad >= PATIENCE:
            print(f"[{cfg['name']}] early stop")
            break
    model.load_state_dict(torch.load(f"{d}/best.pt", map_location=DEVICE))
    out = {"cfg": cfg}
    te_use = te.head(16) if SMOKE else te
    for tta in (False, True):
        r = run_eval(model, te_use, cfg, tta=tta)
        out["test_tta" if tta else "test"] = summarize(r)
        if tta:
            r.to_csv(f"{d}/test_predictions.csv", index=False)
            out["test_tta_FP_only"] = summarize(r[r.source == "FP"]) if (r.source == "FP").any() else None
            out["test_tta_UCL_only"] = summarize(r[r.source == "UCL"]) if (r.source == "UCL").any() else None
    out["dev"] = summarize(run_eval(model, dv.head(16) if SMOKE else dv, cfg, tta=True))
    json.dump(out, open(res_path, "w"), indent=1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"device={DEVICE} torch {torch.__version__} smp {smp.__version__} | out={OUT} | smoke={SMOKE}")
    data = build_data()
    tr, dv, te = split_data(data)
    print(f"images: train {len(tr)} ({tr.patient.nunique()} pts) | dev {len(dv)} ({dv.patient.nunique()} pts) | "
          f"official test {len(te)} ({te.patient.nunique()} pts)")
    results = [run_one(cfg, tr, dv, te) for cfg in RUNS]
    print("\n" + "=" * 110 + "\nOFFICIAL TEST SET (flip-TTA, mm)   [lower is better]")
    print(f"{'run':10s} {'AC MAE':>7s} {'RMSE':>6s} {'median':>7s} {'bias':>7s} {'TAD':>6s} {'APAD':>6s} {'NME':>7s} {'>10mm':>6s} | FP-only AC MAE")
    for r in results:
        s = r["test_tta"]; f = r.get("test_tta_FP_only") or {}
        print(f"{r['cfg']['name']:10s} {s['ac_mae']:7.2f} {s['ac_rmse']:6.2f} {s['ac_median']:7.2f} {s['ac_bias']:+7.2f} "
              f"{s['tad_mae']:6.2f} {s['apad_mae']:6.2f} {s['nme']:7.4f} {100*s['over10']:5.1f}% | {f.get('ac_mae', float('nan')):.2f}")
    print("DONE. Send me this table (and results.json of each run if you like).")


if __name__ == "__main__":
    main()