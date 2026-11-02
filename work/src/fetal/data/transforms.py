import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
    canvas[py:py + nh, px:px + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
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

