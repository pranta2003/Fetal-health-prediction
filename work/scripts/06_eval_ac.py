import argparse
import os
import sys
from pathlib import Path
import torch

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal import config
from fetal.data.ac_dataset import build_data, split_data
from fetal.models.ac_landmark import build_model
from fetal.eval.ac_evaluator import run_eval, summarize
from fetal.eval.ensemble_evaluator import load_ensemble_models, eval_ensemble, summarize_ensemble


def parse_args():
    parser = argparse.ArgumentParser(
        description="06_eval_ac: Evaluate existing AC checkpoints (v3_full and 4-model ensemble) on official test set."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=config.DATA_DIR,
        help=f"Dataset root directory (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--v3-full-ckpt",
        type=str,
        default=config.AC_V3_FULL_CKPT,
        help=f"Path to v3_full checkpoint (default: {config.AC_V3_FULL_CKPT})",
    )
    parser.add_argument(
        "--ens-ckpts",
        nargs="+",
        default=config.AC_ENS_CKPTS,
        help=f"Paths to ensemble seed checkpoints (default: {config.AC_ENS_CKPTS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("06_eval_ac: Evaluating AC checkpoints on official test set")
    print(f"  Device   : {config.DEVICE}")
    print(f"  Data dir : {args.data_dir}")
    print("=" * 70)

    data = build_data(args.data_dir)
    _, _, te = split_data(data)
    print(f"\n[Data] Official test set loaded: {len(te)} images ({te.patient.nunique()} patients)")

    # 1. Evaluate v3_full
    print(f"\n--- 1. Evaluating Single Model: v3_full ---")
    print(f"Checkpoint: {args.v3_full_ckpt}")
    if not os.path.isfile(args.v3_full_ckpt):
        raise FileNotFoundError(f"v3_full checkpoint not found: {args.v3_full_ckpt}")

    model = build_model(encoder_weights=None).to(config.DEVICE)
    model.load_state_dict(torch.load(args.v3_full_ckpt, map_location=config.DEVICE))
    model.eval()

    cfg = dict(name="v3_full", img=384, diam_w=1.0, strong_aug=True, inpaint=True)
    res_df = run_eval(
        model,
        te,
        cfg,
        tta=True,
        bs=8,
        num_workers=config.NUM_WORKERS,
        device=config.DEVICE,
    )
    s_single = summarize(res_df)

    print(f"\n[RESULT: v3_full (flip-TTA)]")
    print(f"  AC MAE       : {s_single['ac_mae']:.4f} mm  (Baseline: 4.3951 mm)")
    print(f"  AC RMSE      : {s_single['ac_rmse']:.4f} mm  (Baseline: 6.1540 mm)")
    print(f"  AC Median    : {s_single['ac_median']:.4f} mm")
    print(f"  AC Bias      : {s_single['ac_bias']:+.4f} mm")
    print(f"  TAD MAE      : {s_single['tad_mae']:.4f} mm")
    print(f"  APAD MAE     : {s_single['apad_mae']:.4f} mm")
    print(f"  NME          : {s_single['nme']:.4f}")
    print(f"  >10mm Outlier: {100*s_single['over10']:.2f}%")

    # 2. Evaluate 4-model ensemble (s1-s4 median)
    print(f"\n--- 2. Evaluating 4-Model Ensemble: s1-s4 (Median Aggregation) ---")
    for ckpt in args.ens_ckpts:
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Ensemble seed checkpoint not found: {ckpt}")
        print(f"  Seed ckpt: {ckpt}")

    seed_dirs = [os.path.dirname(ckpt) for ckpt in args.ens_ckpts]
    models = load_ensemble_models(seed_dirs, device=config.DEVICE)
    r_ens = eval_ensemble(
        models,
        te,
        S=384,
        inpaint=True,
        agg="median",
        bs=8,
        num_workers=config.NUM_WORKERS,
        device=config.DEVICE,
    )
    s_ens = summarize_ensemble(r_ens)

    print(f"\n[RESULT: 4-Model Ensemble Median (flip-TTA)]")
    print(f"  AC MAE       : {s_ens['ac_mae']:.4f} mm  (Baseline: 4.3426 mm)")
    print(f"  AC RMSE      : {s_ens['ac_rmse']:.4f} mm  (Baseline: 6.0305 mm)")
    print(f"  AC Median    : {s_ens['ac_median']:.4f} mm")
    print(f"  AC Bias      : {s_ens['ac_bias']:+.4f} mm")
    print(f"  TAD MAE      : {s_ens['tad_mae']:.4f} mm")
    print(f"  APAD MAE     : {s_ens['apad_mae']:.4f} mm")
    print(f"  NME          : {s_ens['nme']:.4f}")
    print(f"  >10mm Outlier: {100*s_ens['over10']:.2f}%")

    # Verification checks
    diff_v3 = abs(s_single["ac_mae"] - 4.3951)
    diff_ens = abs(s_ens["ac_mae"] - 4.3426)
    print("\n" + "=" * 70)
    print(f"Verification Check against published baselines:")
    print(f"  v3_full  AC MAE diff: {diff_v3:.6f} mm (Pass <= 0.002 mm: {diff_v3 <= 0.002})")
    print(f"  Ensemble AC MAE diff: {diff_ens:.6f} mm (Pass <= 0.002 mm: {diff_ens <= 0.002})")
    print("=" * 70)

    assert diff_v3 <= 0.002, f"v3_full AC MAE differs by {diff_v3:.6f} mm"
    assert diff_ens <= 0.002, f"Ensemble AC MAE differs by {diff_ens:.6f} mm"
    print("\nALL EVALUATION CHECKS PASSED.")


if __name__ == "__main__":
    main()

