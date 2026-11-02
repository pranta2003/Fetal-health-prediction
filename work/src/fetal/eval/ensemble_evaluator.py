import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fetal import config
from fetal.data.ac_dataset import AbdomenDS
from fetal.models.ac_landmark import build_model
from fetal.eval.ac_geometry import match
from fetal.eval.ac_evaluator import predict_batch, metrics_rows, summarize


@torch.no_grad()
def eval_ensemble(models, te, S=384, inpaint=True, agg="median", bs=8, num_workers=None, device=None):
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    if device is None:
        device = config.DEVICE

    for m in models:
        m.eval()

    ds = AbdomenDS(te, S, False, False, inpaint)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    rows = []
    for x, gt, mmpp, idx in dl:
        x, gt, mmpp = x.to(device), gt.to(device), mmpp.to(device)
        preds = [predict_batch(m, x, True, device=device) for m in models]
        ref = preds[0]
        stack = torch.stack([ref] + [match(p, ref)[0] for p in preds[1:]])
        pred = stack.mean(0) if agg == "mean" else stack.median(0).values
        m_ = metrics_rows(pred, gt, mmpp)
        pm, _ = match(pred, gt)
        dd = (pm - gt).norm(dim=-1)
        tl, al = (gt[:, 1] - gt[:, 0]).norm(dim=-1), (gt[:, 3] - gt[:, 2]).norm(dim=-1)
        m_["nme_tad"], m_["nme_apad"] = dd[:, :2].mean(-1) / tl, dd[:, 2:].mean(-1) / al
        for j in range(len(idx)):
            r = te.iloc[int(idx[j])]
            rows.append({k: float(v[j]) for k, v in m_.items()} | {"image_name": r["image_name"], "source": r["source"]})
    return pd.DataFrame(rows)


def summarize_ensemble(r):
    s = summarize(r)
    s |= dict(
        tad_nme=r.nme_tad.mean(),
        tad_nme_sd=r.nme_tad.std(),
        apad_nme=r.nme_apad.mean(),
        apad_nme_sd=r.nme_apad.std(),
        ac_pct=float((100 * r.ac_err / r.ac_true).mean()),
    )
    return s


def load_ensemble_models(checkpoint_paths_or_dirs, device=None):
    if device is None:
        device = config.DEVICE
    models = []
    for p in checkpoint_paths_or_dirs:
        ckpt_file = p if p.endswith(".pt") else os.path.join(p, "best.pt")
        m = build_model().to(device)
        m.load_state_dict(torch.load(ckpt_file, map_location=device))
        models.append(m.eval())
    return models

