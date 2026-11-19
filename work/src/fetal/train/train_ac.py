import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fetal import config
from fetal.models.ac_landmark import build_model, load_pretrained_encoder
from fetal.models.losses import compute_loss
from fetal.data.ac_dataset import AbdomenDS
from fetal.eval.ac_evaluator import run_eval, summarize
from fetal.train.checkpoint import atomic_save
from fetal.train.constants import BATCH, LR, MAX_EPOCHS, PATIENCE
from fetal.train.ema import ModelEMA


def run_one(cfg, tr, dv, te, out_dir=None, max_epochs=None, patience=None):
    """
    Train a single AC landmark regression model based on config recipe.
    Supports optional --ema, --loss {default, wing, smooth_l1}, --init-encoder PATH,
    --epochs, --patience, and --seed while preserving default v3_full behavior.
    """
    if out_dir is None:
        out_dir = config.AC_V3_DIR
    if max_epochs is None:
        max_epochs = cfg.get("max_epochs", 2 if config.SMOKE else MAX_EPOCHS)
    if patience is None:
        patience = cfg.get("patience", PATIENCE)

    is_smoke = config.SMOKE or cfg.get("smoke", False)
    seed = cfg.get("seed", 0)
    use_ema = cfg.get("ema", False)
    loss_type = cfg.get("loss_type", "default")
    init_encoder_path = cfg.get("init_encoder", None)

    d = os.path.join(out_dir, cfg["name"])
    os.makedirs(d, exist_ok=True)
    res_path = os.path.join(d, "results.json")
    if os.path.isfile(res_path):
        print(f"[{cfg['name']}] already finished -> skipping")
        with open(res_path, "r", encoding="utf-8") as f:
            return json.load(f)
    S = cfg["img"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model().to(config.DEVICE)

    if init_encoder_path:
        load_pretrained_encoder(model, init_encoder_path, verbose=True)

    ema = ModelEMA(model, decay=0.99, warmup=True).to(config.DEVICE) if use_ema else None

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler(config.DEVICE, enabled=(config.DEVICE == "cuda"))
    start, best, bad = 0, float("inf"), 0
    last = os.path.join(d, "last.pt")
    if os.path.isfile(last):
        ck = torch.load(last, map_location=config.DEVICE, weights_only=False)
        assert ck["cfg"] == cfg, f"checkpoint config differs from {cfg} - delete {d} to start over"
        model.load_state_dict(ck["model"])
        if use_ema and "ema_model" in ck:
            ema.load_state_dict(ck["ema_model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"[{cfg['name']}] resumed at epoch {start}, best dev score {best:.3f}")
    trd = tr.head(24) if is_smoke else tr
    dl = DataLoader(
        AbdomenDS(trd, S, True, cfg["strong_aug"], cfg["inpaint"]),
        batch_size=BATCH,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )
    for ep in range(start, max_epochs):
        t0 = time.time()
        model.train()
        tl = 0.0
        for x, gt, mmpp, _ in dl:
            x, gt, mmpp = x.to(config.DEVICE), gt.to(config.DEVICE), mmpp.to(config.DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=config.DEVICE, enabled=(config.DEVICE == "cuda")):
                logits = model(x)
            loss, _ = compute_loss(logits, gt, mmpp, S, cfg["diam_w"], loss_type=loss_type)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if use_ema:
                ema.update(model)
            tl += loss.item()

        eval_model = ema.ema_model if use_ema else model
        r = run_eval(
            eval_model,
            dv.head(16) if is_smoke else dv,
            cfg,
            tta=False,
            num_workers=config.NUM_WORKERS,
            device=config.DEVICE,
        )
        s = summarize(r)
        score = s["ac_mae"] + s["lm_mm"]
        sched.step(score)
        improved = score < best
        if improved:
            best, bad = score, 0
            atomic_save(eval_model.state_dict(), os.path.join(d, "best.pt"))
        else:
            bad += 1
        print(
            f"[{cfg['name']}] ep {ep+1:3d}/{max_epochs} loss={tl/len(dl):.4f} | dev AC MAE {s['ac_mae']:.2f}mm "
            f"lm {s['lm_mm']:.2f}mm bias {s['ac_bias']:+.2f} | score {score:.3f}{' *' if improved else ''} | "
            f"lr {opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s | bad={bad}",
            flush=True,
        )
        last_ckpt = dict(
            cfg=cfg,
            model=model.state_dict(),
            opt=opt.state_dict(),
            sched=sched.state_dict(),
            scaler=scaler.state_dict(),
            epoch=ep,
            best=best,
            bad=bad,
        )
        if use_ema:
            last_ckpt["ema_model"] = ema.state_dict()
        atomic_save(last_ckpt, last)
        if bad >= patience:
            print(f"[{cfg['name']}] early stop")
            break

    best_ckpt = torch.load(os.path.join(d, "best.pt"), map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt)
    out = {"cfg": cfg}
    te_use = te.head(16) if is_smoke else te
    for tta in (False, True):
        r = run_eval(
            model,
            te_use,
            cfg,
            tta=tta,
            num_workers=config.NUM_WORKERS,
            device=config.DEVICE,
        )
        out["test_tta" if tta else "test"] = summarize(r)
        if tta:
            # Save test predictions (or fold OOF predictions)
            pred_csv_name = "oof_predictions.csv" if cfg.get("fold") is not None else "test_predictions.csv"
            r.to_csv(os.path.join(d, pred_csv_name), index=False)
            if "source" in r.columns:
                out["test_tta_FP_only"] = summarize(r[r.source == "FP"]) if (r.source == "FP").any() else None
                out["test_tta_UCL_only"] = summarize(r[r.source == "UCL"]) if (r.source == "UCL").any() else None
    out["dev"] = summarize(
        run_eval(
            model,
            dv.head(16) if is_smoke else dv,
            cfg,
            tta=True,
            num_workers=config.NUM_WORKERS,
            device=config.DEVICE,
        )
    )
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out
