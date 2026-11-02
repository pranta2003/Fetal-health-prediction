import os
import sys
import json
import subprocess
import pytest
import torch

from fetal import config
from fetal.data.ac_dataset import build_data, split_data
from fetal.eval.ac_evaluator import run_eval, summarize
from fetal.models.ac_landmark import build_model
from fetal.train.train_ac import run_one
from fetal.train.train_ensemble import train_one, split_seed


@pytest.mark.slow
def test_run_one_training_equivalence(tmp_path):
    """
    Test 2: AC_SMOKE=1 training equivalence between reference ac_v3_original.py and new fetal.train.train_ac.
    24 train images, 16 dev, 2 epochs, CPU, num_workers=0.
    Compares dev and test_tta ac_mae, lm_mm, nme within 1e-4.
    """
    if not os.path.isdir(config.DATA_DIR):
        pytest.skip(f"Data directory missing: {config.DATA_DIR}")

    tmp_orig = (tmp_path / "orig").as_posix()
    tmp_new = (tmp_path / "new").as_posix()

    cfg = dict(name="v3_smoke_run", img=128, diam_w=1.0, strong_aug=True, inpaint=True)

    # Run original run_one via subprocess to honour import-time SMOKE flags
    script_orig = f"""
import os, sys
os.environ['AC_SMOKE'] = '1'
os.environ['DATA_BASE'] = r'{config.DATA_DIR}'
os.environ['OUT_DIR'] = r'{tmp_orig}'
sys.path.insert(0, r'{os.path.join(config.REPO_ROOT, "tests", "reference")}')
import ac_v3_original as orig
orig.DEVICE = 'cpu'
orig.NUM_WORKERS = 0
orig.BASE = r'{config.DATA_DIR}'
orig.OUT = r'{tmp_orig}'
data = orig.build_data()
tr, dv, te = orig.split_data(data)
cfg = {cfg}
res = orig.run_one(cfg, tr, dv, te)
"""
    p_orig = subprocess.run(
        [sys.executable, "-c", script_orig],
        capture_output=True,
        text=True,
    )
    assert p_orig.returncode == 0, f"Original run_one failed:\n{p_orig.stderr}"

    # Run new run_one via subprocess to ensure exact isolated environment
    script_new = f"""
import os, sys
os.environ['AC_SMOKE'] = '1'
os.environ['FETAL_DATA_DIR'] = r'{config.DATA_DIR}'
os.environ['FETAL_MODEL_DIR'] = r'{config.MODEL_DIR}'
sys.path.insert(0, r'{os.path.join(config.REPO_ROOT, "src")}')
from fetal import config
from fetal.data.ac_dataset import build_data, split_data
from fetal.train.train_ac import run_one
data = build_data(config.DATA_DIR)
tr, dv, te = split_data(data)
cfg = {cfg}
res = run_one(cfg, tr, dv, te, out_dir=r'{tmp_new}')
"""
    p_new = subprocess.run(
        [sys.executable, "-c", script_new],
        capture_output=True,
        text=True,
    )
    assert p_new.returncode == 0, f"New run_one failed:\n{p_new.stderr}"

    orig_json = os.path.join(tmp_orig, cfg["name"], "results.json")
    new_json = os.path.join(tmp_new, cfg["name"], "results.json")

    assert os.path.isfile(orig_json), f"Original results.json missing at {orig_json}"
    assert os.path.isfile(new_json), f"New results.json missing at {new_json}"

    with open(orig_json, "r", encoding="utf-8") as f:
        res_orig = json.load(f)
    with open(new_json, "r", encoding="utf-8") as f:
        res_new = json.load(f)

    print("\n[TRAIN EQUIVALENCE: run_one]")
    for split in ["dev", "test_tta"]:
        print(f"  Split: {split}")
        for metric in ["ac_mae", "lm_mm", "nme"]:
            val_orig = res_orig[split][metric]
            val_new = res_new[split][metric]
            diff = abs(val_orig - val_new)
            print(f"    {metric:8s}: orig={val_orig:.6f}, new={val_new:.6f}, diff={diff:.8f}")
            assert diff <= 1e-4, (
                f"run_one mismatch on {split} {metric}: orig={val_orig}, new={val_new}, diff={diff}"
            )


@pytest.mark.slow
def test_train_one_ensemble_equivalence(tmp_path):
    """
    Test 3: AC_SMOKE=1 training equivalence between reference ac_v3_ens_original.py and fetal.train.train_ensemble.
    Seed 1, 2 epochs, CPU, num_workers=0.
    Evaluates best.pt on smoke dev set and compares results.
    """
    if not os.path.isdir(config.DATA_DIR):
        pytest.skip(f"Data directory missing: {config.DATA_DIR}")

    tmp_orig = (tmp_path / "orig_ens").as_posix()
    tmp_new = (tmp_path / "new_ens").as_posix()

    cfg = dict(name="s1", seed=1, batch=4, img=128, diam_w=1.0, strong_aug=True, inpaint=True)

    # Run original train_one via subprocess
    script_orig = f"""
import os, sys
os.environ['AC_SMOKE'] = '1'
os.environ['DATA_BASE'] = r'{config.DATA_DIR}'
os.environ['OUT_DIR'] = r'{tmp_orig}'
sys.path.insert(0, r'{os.path.join(config.REPO_ROOT, "tests", "reference")}')
import ac_v3 as A
import ac_v3_ens_original as orig_ens
A.DEVICE = 'cpu'
A.NUM_WORKERS = 0
A.BASE = r'{config.DATA_DIR}'
orig_ens.OUT_ENS = r'{tmp_orig}'
data = A.build_data()
cfg = {cfg}
d = orig_ens.train_one(cfg, data)
"""
    p_orig = subprocess.run(
        [sys.executable, "-c", script_orig],
        capture_output=True,
        text=True,
    )
    assert p_orig.returncode == 0, f"Original train_one failed:\n{p_orig.stderr}"

    # Run new train_one via subprocess
    script_new = f"""
import os, sys
os.environ['AC_SMOKE'] = '1'
os.environ['FETAL_DATA_DIR'] = r'{config.DATA_DIR}'
os.environ['FETAL_MODEL_DIR'] = r'{config.MODEL_DIR}'
sys.path.insert(0, r'{os.path.join(config.REPO_ROOT, "src")}')
from fetal import config
from fetal.data.ac_dataset import build_data
from fetal.train.train_ensemble import train_one
data = build_data(config.DATA_DIR)
cfg = {cfg}
d = train_one(cfg, data, out_dir=r'{tmp_new}')
"""
    p_new = subprocess.run(
        [sys.executable, "-c", script_new],
        capture_output=True,
        text=True,
    )
    assert p_new.returncode == 0, f"New train_one failed:\n{p_new.stderr}"

    orig_best = os.path.join(tmp_orig, cfg["name"], "best.pt")
    new_best = os.path.join(tmp_new, cfg["name"], "best.pt")
    assert os.path.isfile(orig_best), f"Original best.pt missing at {orig_best}"
    assert os.path.isfile(new_best), f"New best.pt missing at {new_best}"

    # Evaluate both best.pt models on the smoke dev set
    data = build_data(config.DATA_DIR)
    _, dv, _ = split_seed(data, seed=1)
    dv_smoke = dv.head(16)

    m_orig = build_model(encoder_weights=None).to("cpu")
    m_orig.load_state_dict(torch.load(orig_best, map_location="cpu"))
    res_orig = summarize(run_eval(m_orig, dv_smoke, cfg, tta=False, num_workers=0, device="cpu"))

    m_new = build_model(encoder_weights=None).to("cpu")
    m_new.load_state_dict(torch.load(new_best, map_location="cpu"))
    res_new = summarize(run_eval(m_new, dv_smoke, cfg, tta=False, num_workers=0, device="cpu"))

    print("\n[TRAIN EQUIVALENCE: train_one]")
    for metric in ["ac_mae", "lm_mm", "nme"]:
        val_orig = res_orig[metric]
        val_new = res_new[metric]
        diff = abs(val_orig - val_new)
        print(f"  {metric:8s}: orig={val_orig:.6f}, new={val_new:.6f}, diff={diff:.8f}")
        assert diff <= 1e-4, (
            f"train_one mismatch on dev {metric}: orig={val_orig}, new={val_new}, diff={diff}"
        )


@pytest.mark.slow
def test_training_resume_and_skip(tmp_path):
    """
    Test 4: Verify checkpoint resume (reads last.pt) and skip of completed runs.
    """
    if not os.path.isdir(config.DATA_DIR):
        pytest.skip(f"Data directory missing: {config.DATA_DIR}")

    out_dir = (tmp_path / "resume_test").as_posix()
    cfg = dict(name="resume_run", img=128, diam_w=1.0, strong_aug=True, inpaint=True, smoke=True)

    data = build_data(config.DATA_DIR)
    tr, dv, te = split_data(data)

    # Step 1: Run with max_epochs = 1 (runs epoch 0)
    res_ep1 = run_one(cfg, tr, dv, te, out_dir=out_dir, max_epochs=1)

    d = os.path.join(out_dir, cfg["name"])
    last_pt = os.path.join(d, "last.pt")
    res_json = os.path.join(d, "results.json")

    assert os.path.isfile(last_pt), "last.pt was not saved after epoch 1"
    assert os.path.isfile(res_json), "results.json was not saved after completion of 1 epoch"

    ck1 = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert ck1["epoch"] == 0, f"Expected last.pt epoch=0, got {ck1['epoch']}"

    # Remove results.json to simulate an interrupted run before final evaluation
    os.remove(res_json)

    # Step 2: Resume with max_epochs = 2 (reads last.pt, starts from epoch 1)
    res_ep2 = run_one(cfg, tr, dv, te, out_dir=out_dir, max_epochs=2)

    assert os.path.isfile(res_json), "results.json was not created after resuming to epoch 2"
    ck2 = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert ck2["epoch"] == 1, f"Expected last.pt epoch=1 after resume, got {ck2['epoch']}"

    # Step 3: Run again when results.json exists -> must skip immediately
    mtime_before = os.path.getmtime(res_json)
    res_skipped = run_one(cfg, tr, dv, te, out_dir=out_dir, max_epochs=2)
    mtime_after = os.path.getmtime(res_json)

    assert mtime_before == mtime_after, "run_one did not skip finished run with existing results.json"
    assert res_skipped == res_ep2, "Skipped run output differs from saved results.json"

    # Step 4: Test train_one skip on DONE flag
    d_ens = os.path.join(out_dir, "s_done")
    os.makedirs(d_ens, exist_ok=True)
    with open(os.path.join(d_ens, "DONE"), "w", encoding="utf-8") as f:
        f.write("ok")

    cfg_ens = dict(name="s_done", seed=1, batch=4, img=128, diam_w=1.0, strong_aug=True, inpaint=True, smoke=True)
    ret_d = train_one(cfg_ens, data, out_dir=out_dir, max_epochs=1)
    assert ret_d == d_ens, "train_one did not skip run with existing DONE flag"
