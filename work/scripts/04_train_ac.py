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


def parse_args():
    parser = argparse.ArgumentParser(
        description="04_train_ac: Train AC landmark regression models (base and v3_full)."
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
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
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
        print(f"\n[04_train_ac] Starting recipe: {cfg['name']} (img={cfg['img']}, diam_w={cfg['diam_w']})")
        r = run_one(cfg, tr, dv, te, out_dir=args.out_dir)
        results.append(r)

    print("\n[04_train_ac] Finished all recipes successfully.")


if __name__ == "__main__":
    main()

