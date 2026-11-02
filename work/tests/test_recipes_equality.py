import os
import copy
import pytest

from fetal.train import recipes
import ac_v3_original as orig_ac
import ac_v3_ens_original as orig_ens


def test_v3_runs_equality_non_smoke():
    expected = [
        dict(name="base", img=384, diam_w=0.0, strong_aug=False, inpaint=False),
        dict(name="v3_full", img=384, diam_w=1.0, strong_aug=True, inpaint=True),
    ]
    actual = recipes.get_v3_runs(smoke=False)
    assert actual == expected


def test_v3_runs_equality_smoke():
    expected = [
        dict(name="base", img=128, diam_w=0.0, strong_aug=False, inpaint=False),
        dict(name="v3_full", img=128, diam_w=1.0, strong_aug=True, inpaint=True),
    ]
    actual = recipes.get_v3_runs(smoke=True)
    assert actual == expected


def test_ensemble_recipes_equality_non_smoke():
    expected_recipe = dict(img=384, diam_w=1.0, strong_aug=True, inpaint=True)
    assert recipes.get_ensemble_recipe(smoke=False) == expected_recipe

    expected_seeds = [1, 2, 3, 4, 5]
    assert recipes.get_seeds(smoke=False) == expected_seeds

    expected_cfgs = [
        dict(name=f"s{s}", seed=s, batch=4, img=384, diam_w=1.0, strong_aug=True, inpaint=True)
        for s in expected_seeds
    ]
    assert recipes.get_ensemble_cfgs(smoke=False, seeds=expected_seeds) == expected_cfgs

    expected_512 = dict(
        name="res512",
        seed=1,
        batch=2,
        img=512,
        diam_w=1.0,
        strong_aug=True,
        inpaint=True,
    )
    assert recipes.get_cfg_512(smoke=False) == expected_512


def test_ensemble_recipes_equality_smoke():
    expected_recipe = dict(img=128, diam_w=1.0, strong_aug=True, inpaint=True)
    assert recipes.get_ensemble_recipe(smoke=True) == expected_recipe

    expected_seeds = [1, 2]
    assert recipes.get_seeds(smoke=True) == expected_seeds

    expected_cfgs = [
        dict(name=f"s{s}", seed=s, batch=4, img=128, diam_w=1.0, strong_aug=True, inpaint=True)
        for s in expected_seeds
    ]
    assert recipes.get_ensemble_cfgs(smoke=True, seeds=expected_seeds) == expected_cfgs

    expected_512 = dict(
        name="res512",
        seed=1,
        batch=2,
        img=160,
        diam_w=1.0,
        strong_aug=True,
        inpaint=True,
    )
    assert recipes.get_cfg_512(smoke=True) == expected_512


def test_recipe_equality_against_reference_modules():
    # Verify non-smoke matches reference modules
    orig_ac_runs = [
        dict(name="base", img=384, diam_w=0.0, strong_aug=False, inpaint=False),
        dict(name="v3_full", img=384, diam_w=1.0, strong_aug=True, inpaint=True),
    ]
    assert recipes.get_v3_runs(smoke=False) == orig_ac_runs

    # Verify smoke matches reference modules
    orig_ac_runs_smoke = [
        dict(name="base", img=128, diam_w=0.0, strong_aug=False, inpaint=False),
        dict(name="v3_full", img=128, diam_w=1.0, strong_aug=True, inpaint=True),
    ]
    assert recipes.get_v3_runs(smoke=True) == orig_ac_runs_smoke


def test_recipe_env_switches(monkeypatch):
    monkeypatch.setenv("ENS_SEEDS", "1, 4")
    assert recipes.get_seeds(smoke=False) == [1, 4]
    cfgs = recipes.get_ensemble_cfgs(smoke=False)
    assert [c["seed"] for c in cfgs] == [1, 4]

    monkeypatch.setenv("ENS_SKIP_512", "1")
    assert recipes.get_skip_512() is True

    monkeypatch.setenv("ENS_SKIP_512", "0")
    assert recipes.get_skip_512() is False

