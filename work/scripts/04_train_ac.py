import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal import config
from fetal.data.ac_dataset import build_data, split_data
from fetal.train.train_ac import run_one
from fetal.train.recipes import get_v3_runs


import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal import config
from fetal.data.ac_dataset import build_data, split_data, load_ac_folds, get_fold_data
from fetal.train.train_ac import run_one
from fetal.train.recipes import get_v3_runs
from fetal.eval.ac_evaluator import summarize


def pool_oof_results(run_dir: str | Path, n_folds: int = 5) -> dict | None:
    """
    Check if all n_folds out-of-fold prediction CSVs exist, pool them,
    compute consolidated metrics, and write oof_predictions_pooled.csv & oof_summary.json.
    """
    run_path = Path(run_dir)
    fold_csvs = []
    for f in range(1, n_folds + 1):
        fold_csv = run_path / f"fold_{f}" / "oof_predictions.csv"
        if not fold_csv.is_file():
            # Check alternative naming
            fold_csv = run_path / f"fold_{f}" / "test_predictions.csv"
        if fold_csv.is_file():
            fold_csvs.append(fold_csv)

    if len(fold_csvs) != n_folds:
        return None

    dfs = [pd.read_csv(p) for p in fold_csvs]
    pooled_df = pd.concat(dfs, ignore_index=True)
    pooled_path = run_path / "oof_predictions_pooled.csv"
    pooled_df.to_csv(pooled_path, index=False)

    summary = summarize(pooled_df)
    summary_path = run_path / "oof_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("\n" + "=" * 70)
    print(f"POOLED 5-FOLD OUT-OF-FOLD RESULTS ({len(pooled_df)} images)")
    print("=" * 70)
    print(f"  AC MAE       : {summary['ac_mae']:.4f} mm")
    print(f"  AC RMSE      : {summary['ac_rmse']:.4f} mm")
    print(f"  AC Median    : {summary['ac_median']:.4f} mm")
    print(f"  AC Bias      : {summary['ac_bias']:+.4f} mm")
    print(f"  TAD MAE      : {summary['tad_mae']:.4f} mm")
    print(f"  APAD MAE     : {summary['apad_mae']:.4f} mm")
    print(f"  NME          : {summary['nme']:.4f}")
    print(f"  >10mm Outlier: {100 * summary['over10']:.2f}%")
    print("=" * 70 + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="04_train_ac: Train AC landmark regression models (single split or 5-fold CV)."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=config.AC_V3_DIR,
        help=f"Directory to save outputs and checkpoints (default: {config.AC_V3_DIR})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=config.SMOKE,
        help="Run smoke mode (2 epochs on tiny subset, default from AC_SMOKE env)",
    )
    # Campaign Step 1 Options
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help="Fold index (1..5) or 'all' to train on 5-fold CV splits. If None, runs default v3 split.",
    )
    parser.add_argument(
        "--folds-csv",
        type=str,
        default=None,
        help="Path to folds_v1.csv (default: docs/results/ac_campaign/folds_v1.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for fold inner holdout and model initialization (default: 42 for folds, 0 for legacy)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Max epochs to train (default: 80, or 2 if smoke)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default: 12)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="default",
        choices=["default", "l2", "smooth_l1", "smooth-l1", "wing"],
        help="Loss for landmark coordinate offsets: 'default' (Euclidean), 'smooth_l1', 'wing'",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
        help="Enable Exponential Moving Average (EMA) on model weights for validation/eval",
    )
    parser.add_argument(
        "--init-encoder",
        type=str,
        default=None,
        help="Path to pretrained U-Net effb4 checkpoint (e.g. FASSD Gray best_model.pt)",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="v3_full",
        choices=["v3_full", "base"],
        help="Base recipe hyperparameter preset (default: v3_full)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Standard Legacy Mode (No --fold specified)
    if args.fold is None:
        print(f"[04_train_ac] device={config.DEVICE} | out_dir={args.out_dir} | smoke={args.smoke}")
        data = build_data(config.DATA_DIR)
        tr, dv, te = split_data(data)
        print(
            f"[04_train_ac] images: train {len(tr)} ({tr.patient.nunique()} pts) | "
            f"dev {len(dv)} ({dv.patient.nunique()} pts) | official test {len(te)} ({te.patient.nunique()} pts)"
        )

        runs = get_v3_runs(smoke=args.smoke)
        results = []
        for cfg in runs:
            # Propagate optional overrides if specified
            if args.epochs is not None:
                cfg["max_epochs"] = args.epochs
            if args.patience is not None:
                cfg["patience"] = args.patience
            if args.loss != "default":
                cfg["loss_type"] = args.loss
            if args.ema:
                cfg["ema"] = True
            if args.init_encoder:
                cfg["init_encoder"] = args.init_encoder
            if args.seed is not None:
                cfg["seed"] = args.seed

            print(f"\n[04_train_ac] Starting recipe: {cfg['name']} (img={cfg['img']}, diam_w={cfg['diam_w']})")
            r = run_one(cfg, tr, dv, te, out_dir=args.out_dir)
            results.append(r)

        print("\n[04_train_ac] Finished all recipes successfully.")
        return

    # 2. 5-Fold Cross Validation Mode (--fold 1..5 or 'all')
    seed = 42 if args.seed is None else args.seed
    folds_df = load_ac_folds(args.folds_csv, base=config.DATA_DIR)
    print(f"[04_train_ac] Loaded folds ({len(folds_df)} images, {folds_df.patient.nunique()} patients, seed={seed})")

    if args.fold.lower() == "all":
        folds_to_run = list(range(1, 6))
    else:
        folds_to_run = [int(args.fold)]

    base_cfg = {
        "img": 128 if args.smoke else 384,
        "diam_w": 1.0 if args.recipe == "v3_full" else 0.0,
        "strong_aug": True if args.recipe == "v3_full" else False,
        "inpaint": True if args.recipe == "v3_full" else False,
        "ema": args.ema,
        "loss_type": args.loss,
        "init_encoder": args.init_encoder,
        "seed": seed,
        "smoke": args.smoke,
    }
    if args.epochs is not None:
        base_cfg["max_epochs"] = args.epochs
    if args.patience is not None:
        base_cfg["patience"] = args.patience

    for fold_k in folds_to_run:
        print(f"\n" + "=" * 60)
        print(f"[04_train_ac] Running Fold {fold_k}/5 (seed={seed}, loss={args.loss}, ema={args.ema})")
        print("=" * 60)
        tr, dv, te = get_fold_data(folds_df, fold=fold_k, seed=seed)
        print(
            f"  Inner-train : {len(tr):3d} images ({tr.patient.nunique():3d} patients)\n"
            f"  Inner-val   : {len(dv):3d} images ({dv.patient.nunique():3d} patients) [epoch selection]\n"
            f"  Out-of-fold : {len(te):3d} images ({te.patient.nunique():3d} patients) [fold {fold_k}]"
        )

        fold_cfg = dict(base_cfg)
        fold_cfg["name"] = f"fold_{fold_k}"
        fold_cfg["fold"] = fold_k

        run_one(fold_cfg, tr, dv, te, out_dir=args.out_dir)

    # Pool results if all 5 folds are present
    pool_oof_results(args.out_dir, n_folds=5)


if __name__ == "__main__":
    main()

