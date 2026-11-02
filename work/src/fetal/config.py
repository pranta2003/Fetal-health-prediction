import os
import re
from pathlib import Path
import yaml
import torch

IS_WIN = os.name == "nt"

# Repository root (parent of src/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _expand_env_vars(val: str) -> str:
    """
    Expand ${VAR:-default} or ${VAR} syntax in configuration strings.
    If FETAL_DATA_DIR or FETAL_MODEL_DIR are not in os.environ, platform-appropriate defaults win
    (Windows: C:/fetal-data, C:/fetal-models; Linux/Colab: /content/fetal-data, /content/fetal-models).
    """
    pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

    def _replace(match):
        env_var = match.group(1)
        yaml_default = match.group(2) if match.group(2) is not None else ""
        if env_var == "FETAL_DATA_DIR":
            platform_default = "C:/fetal-data" if IS_WIN else "/content/fetal-data"
            return os.environ.get(env_var, yaml_default or platform_default)
        elif env_var == "FETAL_MODEL_DIR":
            platform_default = "C:/fetal-models" if IS_WIN else "/content/fetal-models"
            return os.environ.get(env_var, yaml_default or platform_default)
        return os.environ.get(env_var, yaml_default)

    return pattern.sub(_replace, val)


def _process_config_dict(d: dict) -> dict:
    processed = {}
    for k, v in d.items():
        if isinstance(v, dict):
            processed[k] = _process_config_dict(v)
        elif isinstance(v, str):
            processed[k] = _expand_env_vars(v)
        else:
            processed[k] = v
    return processed


def load_paths_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = REPO_ROOT / "configs" / "paths.yaml"
    else:
        config_path = Path(config_path)

    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = _process_config_dict(cfg)
    else:
        cfg = {}

    return cfg


_PATHS_CFG = load_paths_config()

# Data paths (single source of truth from configs/paths.yaml)
_data_cfg = _PATHS_CFG.get("data", {})
DATA_DIR = _data_cfg.get("root_dir", "C:/fetal-data" if IS_WIN else "/content/fetal-data")
FP_ANNOTATIONS = _data_cfg.get("fp_annotations", f"{DATA_DIR}/FP/annotations")
FP_IMAGES = _data_cfg.get("fp_images", f"{DATA_DIR}/FP/images/Abdomen")
UCL_ANNOTATIONS = _data_cfg.get("ucl_annotations", f"{DATA_DIR}/UCL/annotations/px_to_mm")
UCL_IMAGES = _data_cfg.get("ucl_images", f"{DATA_DIR}/UCL/images/Abdomen")
HC18_DIR = _data_cfg.get("hc18_dir", f"{DATA_DIR}/HC18")
TABULAR_DIR = _data_cfg.get("tabular_dir", f"{DATA_DIR}/cdc")

# Model paths (single source of truth from configs/paths.yaml)
_models_cfg = _PATHS_CFG.get("models", {})
MODEL_DIR = _models_cfg.get("root_dir", "C:/fetal-models" if IS_WIN else "/content/fetal-models")
AC_V3_DIR = _models_cfg.get("ac_v3_dir", f"{MODEL_DIR}/ac_v3")
AC_V3_FULL_CKPT = _models_cfg.get("ac_v3_full", f"{MODEL_DIR}/ac_v3/v3_full/best.pt")
AC_V3_BASE_CKPT = _models_cfg.get("ac_v3_base", f"{MODEL_DIR}/ac_v3/base/best.pt")
AC_ENS_DIR = _models_cfg.get("ac_ensemble_dir", f"{MODEL_DIR}/ac_v3_ens")
AC_ENS_CKPTS = [
    _models_cfg.get("ac_ens_s1", f"{AC_ENS_DIR}/s1/best.pt"),
    _models_cfg.get("ac_ens_s2", f"{AC_ENS_DIR}/s2/best.pt"),
    _models_cfg.get("ac_ens_s3", f"{AC_ENS_DIR}/s3/best.pt"),
    _models_cfg.get("ac_ens_s4", f"{AC_ENS_DIR}/s4/best.pt"),
]
HC18_MODELS_DIR = _models_cfg.get("hc18_dir", f"{MODEL_DIR}/hc18")
CLINICAL_XGB_CKPT = _models_cfg.get("clinical_xgb", f"{MODEL_DIR}/stream3_clinical_xgb.json")

# Runtime environment settings
_env_cfg = _PATHS_CFG.get("environment", {})
_cfg_device = _env_cfg.get("default_device", "auto")
if _cfg_device == "cuda":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif _cfg_device == "cpu":
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = _env_cfg.get("num_workers_win", 0) if IS_WIN else _env_cfg.get("num_workers_posix", 2)
SMOKE = os.environ.get("AC_SMOKE") == "1"
