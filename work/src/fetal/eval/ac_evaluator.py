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

