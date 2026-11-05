"""ac_infer.py: Single-image inference API for abdominal circumference (AC) biometry."""

import os
from pathlib import Path
from typing import Sequence
import numpy as np
import torch

from fetal import config
from fetal.data.ac_dataset import build_data
from fetal.data.transforms import MEAN, STD, letterbox, read_rgb
from fetal.eval.ac_geometry import ac_t, lengths_mm, match
from fetal.eval.ac_evaluator import predict_batch
from fetal.eval.ensemble_evaluator import load_ensemble_models

_ANN_LOOKUP = None


def _get_px_to_mm_lookup():
    """Lazy loader for px_to_mm_rate lookup dictionary from standard annotations."""
    global _ANN_LOOKUP
    if _ANN_LOOKUP is None:
        try:
            df = build_data(config.DATA_DIR)
            _ANN_LOOKUP = {}
            for _, r in df.iterrows():
                rate = float(r["px_to_mm_rate"])
                _ANN_LOOKUP[r["image_name"]] = rate
                _ANN_LOOKUP[Path(r["image_path"]).as_posix()] = rate
                _ANN_LOOKUP[str(Path(r["image_path"]).resolve())] = rate
        except Exception:
            _ANN_LOOKUP = {}
    return _ANN_LOOKUP


def predict_ac(
    image_path: str | Path,
    ckpt_paths: Sequence[str | Path] = config.AC_ENS_CKPTS,
    flip_tta: bool = True,
    mm_per_pixel: float | None = None,
    S: int = 384,
    inpaint: bool = True,
    models: list | None = None,
    device: str | None = None,
) -> dict:
    """Predict AC, TAD, and APAD biometry in millimeters for a single ultrasound image.

    Parameters
    ----------
    image_path : str or Path
        Path to ultrasound image file.
    ckpt_paths : list of str/Path, optional
        Checkpoint file paths for model ensemble (default: config.AC_ENS_CKPTS).
    flip_tta : bool, optional
        Whether to perform horizontal flip test-time augmentation (default: True).
    mm_per_pixel : float, optional
        Physical pixel spacing (mm/px). If None, looked up from dataset annotations.
    S : int, optional
        Image square dimension for network input (default: 384).
    inpaint : bool, optional
        Whether to inpaint text annotations/callipers during load (default: True).
    models : list of nn.Module, optional
        Preloaded models to avoid reloading from disk on repeated calls.
    device : str, optional
        Computation device (default: config.DEVICE).

    Returns
    -------
    dict with keys 'ac_mm', 'tad_mm', 'apad_mm'.
    """
    image_path = str(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if device is None:
        device = config.DEVICE

    # 1. Resolve physical pixel spacing (mm/px)
    if mm_per_pixel is None:
        lookup = _get_px_to_mm_lookup()
        img_name = Path(image_path).name
        norm_path = Path(image_path).as_posix()
        resolved_path = str(Path(image_path).resolve())

        if img_name in lookup:
            mm_per_pixel = lookup[img_name]
        elif norm_path in lookup:
            mm_per_pixel = lookup[norm_path]
        elif resolved_path in lookup:
            mm_per_pixel = lookup[resolved_path]
        else:
            raise ValueError(
                f"mm_per_pixel was not provided and image '{img_name}' was not found in annotations."
            )
    else:
        mm_per_pixel = float(mm_per_pixel)

    # 2. Image letterbox preprocessing (reusing existing transforms)
    img = read_rgb(image_path, inpaint=inpaint)
    canvas, scale, px, py = letterbox(img, S)
    x = canvas.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    tensor_x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).float().to(device)

    # mm per pixel in the S x S letterboxed image
    mmpp_tensor = torch.tensor([mm_per_pixel / scale], dtype=torch.float32, device=device)

    # 3. Model loading or reuse
    if models is None:
        models = load_ensemble_models(ckpt_paths, device=device)

    # 4. Predict with flip-TTA and median ensemble
    preds = [predict_batch(m, tensor_x, tta=flip_tta, device=device) for m in models]
    if len(models) == 1:
        pred = preds[0]
    else:
        ref = preds[0]
        stack = torch.stack([ref] + [match(p, ref)[0] for p in preds[1:]])
        pred = stack.median(0).values

    # 5. Measure lengths and Ramanujan ellipse circumference
    tp, ap = lengths_mm(pred, mmpp_tensor)
    ac = ac_t(tp, ap)

    return {
        "ac_mm": float(ac[0].item()),
        "tad_mm": float(tp[0].item()),
        "apad_mm": float(ap[0].item()),
    }
