# AC v3 - overnight round 2: 5 seeds of the winning recipe (each with a DIFFERENT dev hold-out), one 512-px run,
# then the 5-model ensemble on the official test set. Needs ac_v3.py in the SAME folder (it imports it; change nothing in it).
# Reports mean +- SD over seeds (honest run-to-run spread), the ensemble, and per-diameter NME as mean +- SD
# (the format used in the multicentre benchmark paper, FP->FP: TAD 0.08+-0.06, APAD 0.08+-0.06; M-C->M-C: 0.08 / 0.07).
# Resumable: stop it any time, run it again.
#   python -u ac_v3_ens.py 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath ens_log.txt -Append
import os, json, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ac_v3 as A

OUT_ENS = A.OUT + "_ens"
SEEDS = [1, 2] if A.SMOKE else [1, 2, 3, 4, 5]
RECIPE = dict(img=128 if A.SMOKE else 384, diam_w=1.0, strong_aug=True, inpaint=True)
CFGS = [dict(name=f"s{s}", seed=s, batch=A.BATCH, **RECIPE) for s in SEEDS]
CFG_512 = dict(name="res512", seed=1, batch=2, img=160 if A.SMOKE else 512, diam_w=1.0, strong_aug=True, inpaint=True)


def split_seed(data, seed):
    train_all = data[data.official_split == "train"]
    pats = np.array(sorted(train_all.patient.unique()))
    rng = np.random.RandomState(seed)
    dev_p = set(rng.permutation(pats)[: max(1, int(round(A.DEV_FRACTION * len(pats))))])
    tr = train_all[~train_all.patient.isin(dev_p)].reset_index(drop=True)
    dv = train_all[train_all.patient.isin(dev_p)].reset_index(drop=True)
    te = data[data.official_split == "test"].reset_index(drop=True)
    assert not (set(tr.patient) & set(dv.patient)) and not (set(tr.patient) & set(te.patient)), "patient overlap!"
    return tr, dv, te


def train_one(cfg, data):
    d = f"{OUT_ENS}/{cfg['name']}"
    os.makedirs(d, exist_ok=True)
    if os.path.isfile(f"{d}/DONE"):
        print(f"[{cfg['name']}] finished earlier -> skipping")
        return d
    S, seed = cfg["img"], cfg["seed"]
    tr, dv, _ = split_seed(data, seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = A.build_model().to(A.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=A.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler(A.DEVICE, enabled=(A.DEVICE == "cuda"))
    start, best, bad = 0, float("inf"), 0
    last = f"{d}/last.pt"
    if os.path.isfile(last):
        ck = torch.load(last, map_location=A.DEVICE, weights_only=False)
        assert ck["cfg"] == cfg, f"checkpoint config differs - delete {d} to start over"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"[{cfg['name']}] resumed at epoch {start}, best dev score {best:.3f}")
    trd = tr.head(24) if A.SMOKE else tr
    dl = DataLoader(A.AbdomenDS(trd, S, True, cfg["strong_aug"], cfg["inpaint"]), batch_size=cfg["batch"], shuffle=True,
                    num_workers=A.NUM_WORKERS, drop_last=True)
    for ep in range(start, A.MAX_EPOCHS):
        t0 = time.time()
        model.train()
        tl = 0.0
        for x, gt, mmpp, _ in dl:
            x, gt, mmpp = x.to(A.DEVICE), gt.to(A.DEVICE), mmpp.to(A.DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=A.DEVICE, enabled=(A.DEVICE == "cuda")):
                logits = model(x)
            loss, _ = A.compute_loss(logits, gt, mmpp, S, cfg["diam_w"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl += loss.item()
        s = A.summarize(A.run_eval(model, dv.head(16) if A.SMOKE else dv, cfg, tta=False))
        score = s["ac_mae"] + s["lm_mm"]
        sched.step(score)
        improved = score < best
        if improved:
            best, bad = score, 0
            A.atomic_save(model.state_dict(), f"{d}/best.pt")
        else:
            bad += 1
        print(f"[{cfg['name']}] ep {ep+1:3d}/{A.MAX_EPOCHS} loss={tl/len(dl):.4f} | dev AC MAE {s['ac_mae']:.2f}mm lm {s['lm_mm']:.2f}mm "
              f"| score {score:.3f}{' *' if improved else ''} | {time.time()-t0:.0f}s | bad={bad}", flush=True)
        A.atomic_save(dict(cfg=cfg, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                           scaler=scaler.state_dict(), epoch=ep, best=best, bad=bad), last)
        if bad >= A.PATIENCE:
            print(f"[{cfg['name']}] early stop")
            break
    open(f"{d}/DONE", "w").write("ok")
    return d


@torch.no_grad()
def eval_models(dirs, S, te, inpaint, agg="mean"):
    models = []
    for d in dirs:
        m = A.build_model().to(A.DEVICE)
        m.load_state_dict(torch.load(f"{d}/best.pt", map_location=A.DEVICE))
        models.append(m.eval())
    ds = A.AbdomenDS(te, S, False, False, inpaint)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=A.NUM_WORKERS)
    rows = []
    for x, gt, mmpp, idx in dl:
        x, gt, mmpp = x.to(A.DEVICE), gt.to(A.DEVICE), mmpp.to(A.DEVICE)
        preds = [A.predict_batch(m, x, True) for m in models]
        ref = preds[0]
        stack = torch.stack([ref] + [A.match(p, ref)[0] for p in preds[1:]])
        pred = stack.mean(0) if agg == "mean" else stack.median(0).values
        m_ = A.metrics_rows(pred, gt, mmpp)
        pm, _ = A.match(pred, gt)
        dd = (pm - gt).norm(dim=-1)
        tl, al = (gt[:, 1] - gt[:, 0]).norm(dim=-1), (gt[:, 3] - gt[:, 2]).norm(dim=-1)
        m_["nme_tad"], m_["nme_apad"] = dd[:, :2].mean(-1) / tl, dd[:, 2:].mean(-1) / al
        for j in range(len(idx)):
            r = te.iloc[int(idx[j])]
            rows.append({k: float(v[j]) for k, v in m_.items()} | {"image_name": r["image_name"], "source": r["source"]})
    return pd.DataFrame(rows)


def summ(r):
    s = A.summarize(r)
    s |= dict(tad_nme=r.nme_tad.mean(), tad_nme_sd=r.nme_tad.std(), apad_nme=r.nme_apad.mean(), apad_nme_sd=r.nme_apad.std(),
              ac_pct=float((100 * r.ac_err / r.ac_true).mean()))
    return s


def main():
    os.makedirs(OUT_ENS, exist_ok=True)
    print(f"device={A.DEVICE} | out={OUT_ENS} | seeds={SEEDS} | smoke={A.SMOKE}")
    data = A.build_data()
    _, _, te = split_seed(data, 1)
    te_use = te.head(16) if A.SMOKE else te
    print(f"official test: {len(te)} images ({te.patient.nunique()} patients)")
    dirs = [train_one(c, data) for c in CFGS]
    d512 = train_one(CFG_512, data)

    res = {}
    for c, d in zip(CFGS, dirs):
        res[c["name"]] = summ(eval_models([d], c["img"], te_use, True))
    res["res512"] = summ(eval_models([d512], CFG_512["img"], te_use, True))
    r_mean = eval_models(dirs, RECIPE["img"], te_use, True, "mean")
    r_med = eval_models(dirs, RECIPE["img"], te_use, True, "median")
    res["ENS_mean"], res["ENS_median"] = summ(r_mean), summ(r_med)
    r_mean.to_csv(f"{OUT_ENS}/ensemble_test_predictions.csv", index=False)
    json.dump(res, open(f"{OUT_ENS}/results.json", "w"), indent=1, default=float)

    print("\n" + "=" * 118 + "\nOFFICIAL TEST SET, flip-TTA, mm (lower is better)")
    print(f"{'model':11s} {'AC MAE':>7s} {'RMSE':>6s} {'median':>7s} {'bias':>7s} {'AC %':>6s} {'>10mm':>6s} | {'TAD NME':>13s} {'APAD NME':>13s}")
    for k, s in res.items():
        print(f"{k:11s} {s['ac_mae']:7.2f} {s['ac_rmse']:6.2f} {s['ac_median']:7.2f} {s['ac_bias']:+7.2f} {s['ac_pct']:6.2f} "
              f"{100*s['over10']:5.1f}% | {s['tad_nme']:.3f}+-{s['tad_nme_sd']:.3f} {s['apad_nme']:.3f}+-{s['apad_nme_sd']:.3f}")
    mae = np.array([res[c['name']]['ac_mae'] for c in CFGS])
    print(f"\nsingle 384-px seeds: AC MAE {mae.mean():.2f} +- {mae.std(ddof=1) if len(mae) > 1 else float('nan'):.2f} mm "
          f"(min {mae.min():.2f}, max {mae.max():.2f})")
    print("paper (BiometryNet/HRNet-W18, NME mean+-SD): FP->FP TAD 0.08+-0.06 APAD 0.08+-0.06 | M-C->M-C TAD 0.08+-0.08 APAD 0.07+-0.08")
    print("DONE. Send me this table.")


if __name__ == "__main__":
    main()