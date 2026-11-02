import torch
from fetal.models.ac_landmark import decode
from fetal.eval.ac_geometry import match, lengths_mm, ac_t

SIGMA = 4.0  # heatmap regulariser width (px at 384; scaled with image size)


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


def compute_loss(logits, gt, mmpp, S, diam_w, sigma=SIGMA):
    pts, p = decode(logits)
    matched, order = match(pts, gt)
    coord = (matched - gt).norm(dim=-1).mean() / 10.0
    js = js_loss(p, order, gt, S, sigma * S / 384.0)
    loss = coord + js
    if diam_w > 0:
        tp, ap = lengths_mm(matched, mmpp)
        tt, at = lengths_mm(gt, mmpp)
        l_len = ((tp - tt).abs() + (ap - at).abs()).mean() / 10.0
        l_ac = (ac_t(tp, ap) - ac_t(tt, at)).abs().mean() / 10.0
        loss = loss + diam_w * (l_len + l_ac)
    return loss, coord.item() * 10.0

