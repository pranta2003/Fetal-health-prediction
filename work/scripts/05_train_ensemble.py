import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal import config
from fetal.data.ac_dataset import build_data
from fetal.train.train_ensemble import train_one, split_seed
from fetal.train.recipes import (
    get_seeds,
    get_skip_512,
    get_ensemble_recipe,
    get_ensemble_cfgs,
    get_cfg_512,
)
from fetal.eval.ensemble_evaluator import load_ensemble_models, eval_ensemble, summarize_ensemble


def parse_args():
    parser = argparse.ArgumentParser(
        description="05_train_ensemble: Train AC ensemble seeds, optional 512-px run, and evaluate ensemble."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=config.AC_ENS_DIR,
        help=f"Directory to save outputs and checkpoints (default: {config.AC_ENS_DIR})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=config.SMOKE,
        help="Run smoke mode (2 epochs on tiny subset, default from AC_SMOKE env)",
    )
    parser.add_argument(
        "--skip-512",
        action="store_true",
        default=get_skip_512(),
        help="Skip the 512-px training run (default from ENS_SKIP_512 env)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = get_seeds(smoke=args.smoke)
    recipe = get_ensemble_recipe(smoke=args.smoke)
    cfgs = get_ensemble_cfgs(smoke=args.smoke, seeds=seeds)
    cfg_512 = get_cfg_512(smoke=args.smoke)

    print(
        f"[05_train_ensemble] device={config.DEVICE} | out={args.out_dir} | "
        f"seeds={seeds} | smoke={args.smoke} | skip_512={args.skip_512}"
    )

    data = build_data(config.DATA_DIR)
    _, _, te = split_seed(data, 1)
    te_use = te.head(16) if args.smoke else te
    print(f"[05_train_ensemble] official test: {len(te)} images ({te.patient.nunique()} patients)")

    dirs = [train_one(c, data, out_dir=args.out_dir) for c in cfgs]
    d512 = None if args.skip_512 else train_one(cfg_512, data, out_dir=args.out_dir)

    res = {}
    for c, d in zip(cfgs, dirs):
        m = load_ensemble_models([d], device=config.DEVICE)
        r = eval_ensemble(
            m,
            te_use,
            S=c["img"],
            inpaint=True,
            agg="mean",
            bs=8,
            num_workers=config.NUM_WORKERS,
            device=config.DEVICE,
        )
        res[c["name"]] = summarize_ensemble(r)

    if d512 is not None:
        m512 = load_ensemble_models([d512], device=config.DEVICE)
        r512 = eval_ensemble(
            m512,
            te_use,
            S=cfg_512["img"],
            inpaint=True,
            agg="mean",
            bs=8,
            num_workers=config.NUM_WORKERS,
            device=config.DEVICE,
        )
        res["res512"] = summarize_ensemble(r512)

    # Evaluate ensemble across all seed directories
    models = load_ensemble_models(dirs, device=config.DEVICE)
    r_mean = eval_ensemble(
        models,
        te_use,
        S=recipe["img"],
        inpaint=True,
        agg="mean",
        bs=8,
        num_workers=config.NUM_WORKERS,
        device=config.DEVICE,
    )
    r_med = eval_ensemble(
        models,
        te_use,
        S=recipe["img"],
        inpaint=True,
        agg="median",
        bs=8,
        num_workers=config.NUM_WORKERS,
        device=config.DEVICE,
    )

    res["ENS_mean"] = summarize_ensemble(r_mean)
    res["ENS_median"] = summarize_ensemble(r_med)

    r_mean.to_csv(os.path.join(args.out_dir, "ensemble_test_predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)

    print("\n" + "=" * 118 + "\nOFFICIAL TEST SET, flip-TTA, mm (lower is better)")
    print(
        f"{'model':11s} {'AC MAE':>7s} {'RMSE':>6s} {'median':>7s} {'bias':>7s} "
        f"{'AC %':>6s} {'>10mm':>6s} | {'TAD NME':>13s} {'APAD NME':>13s}"
    )
    for k, s in res.items():
        print(
            f"{k:11s} {s['ac_mae']:7.2f} {s['ac_rmse']:6.2f} {s['ac_median']:7.2f} {s['ac_bias']:+7.2f} "
            f"{s['ac_pct']:6.2f} {100*s['over10']:5.1f}% | {s['tad_nme']:.3f}+-{s['tad_nme_sd']:.3f} "
            f"{s['apad_nme']:.3f}+-{s['apad_nme_sd']:.3f}"
        )

    mae = np.array([res[c['name']]['ac_mae'] for c in cfgs])
    print(
        f"\nsingle {recipe['img']}-px seeds: AC MAE {mae.mean():.2f} +- "
        f"{mae.std(ddof=1) if len(mae) > 1 else float('nan'):.2f} mm (min {mae.min():.2f}, max {mae.max():.2f})"
    )
    print("paper (BiometryNet/HRNet-W18, NME mean+-SD): FP->FP TAD 0.08+-0.06 APAD 0.08+-0.06 | M-C->M-C TAD 0.08+-0.08 APAD 0.07+-0.08")
    print("DONE. Send me this table.")


if __name__ == "__main__":
    main()

