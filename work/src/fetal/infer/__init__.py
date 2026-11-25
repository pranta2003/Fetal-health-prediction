# FIELD: AC
# MODEL: EfficientNet-B4 Landmark U-Net
# STEP: inference
# USED BY: pipeline imports across work/scripts and work/tests

"""Inference modules for multimodal fetal health assessment."""

from fetal.infer.ac_infer import predict_ac

__all__ = ["predict_ac"]
