import os
from fetal import config
from fetal.train.constants import BATCH


def get_v3_runs(smoke=None):
    if smoke is None:
        smoke = config.SMOKE
    img_size = 128 if smoke else 384
    return [
        dict(name="base", img=img_size, diam_w=0.0, strong_aug=False, inpaint=False),
        dict(name="v3_full", img=img_size, diam_w=1.0, strong_aug=True, inpaint=True),
    ]


def get_seeds(smoke=None):
    if smoke is None:
        smoke = config.SMOKE
    env_seeds = os.environ.get("ENS_SEEDS")
    if env_seeds:
        return [int(s.strip()) for s in env_seeds.split(",") if s.strip()]
    return [1, 2] if smoke else [1, 2, 3, 4, 5]


def get_skip_512():
    return os.environ.get("ENS_SKIP_512", "0").lower() in ("1", "true", "yes")


def get_ensemble_recipe(smoke=None):
    if smoke is None:
        smoke = config.SMOKE
    return dict(img=128 if smoke else 384, diam_w=1.0, strong_aug=True, inpaint=True)


def get_ensemble_cfgs(smoke=None, seeds=None):
    if smoke is None:
        smoke = config.SMOKE
    if seeds is None:
        seeds = get_seeds(smoke=smoke)
    recipe = get_ensemble_recipe(smoke=smoke)
    return [dict(name=f"s{s}", seed=s, batch=BATCH, **recipe) for s in seeds]


def get_cfg_512(smoke=None):
    if smoke is None:
        smoke = config.SMOKE
    return dict(
        name="res512",
        seed=1,
        batch=2,
        img=160 if smoke else 512,
        diam_w=1.0,
        strong_aug=True,
        inpaint=True,
    )


# Module-level defaults evaluated at import time
RUNS = get_v3_runs()
SEEDS = get_seeds()
RECIPE = get_ensemble_recipe()
CFGS = get_ensemble_cfgs()
CFG_512 = get_cfg_512()

