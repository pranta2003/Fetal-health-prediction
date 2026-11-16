"""FASSD fetal abdominal structures segmentation module."""

from .fassd_train import (
    FASSDKaggleDataset,
    FASSDMultiLabelLoss,
    ModelEMA,
    build_fassd_model,
    compute_structure_metrics,
    evaluate_dataset,
    train_fassd,
)

from .fassd_postproc import apply_morphology, apply_postprocessing, run_patient_paired_bootstrap
from .fassd_ensemble import (
    average_probabilities,
    select_candidate_models,
    evaluate_ensemble_slice,
)

__all__ = [
    "FASSDKaggleDataset",
    "FASSDMultiLabelLoss",
    "ModelEMA",
    "build_fassd_model",
    "compute_structure_metrics",
    "evaluate_dataset",
    "train_fassd",
    "apply_morphology",
    "apply_postprocessing",
    "run_patient_paired_bootstrap",
    "average_probabilities",
    "select_candidate_models",
    "evaluate_ensemble_slice",
]
