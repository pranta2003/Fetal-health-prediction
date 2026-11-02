import math
import torch

PERMS = []   # 8 allowed relabelings: swap ends inside a pair, swap the two pairs
for sp in (0, 1):
    for s0 in (0, 1):
        for s1 in (0, 1):
            a = [1, 0] if s0 else [0, 1]
            b = [3, 2] if s1 else [2, 3]
            PERMS.append(b + a if sp else a + b)
PERMS_T = torch.tensor(PERMS, dtype=torch.long)


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

