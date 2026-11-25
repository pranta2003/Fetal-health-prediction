# FIELD: General
# MODEL: None
# STEP: training
# USED BY: pipeline imports across work/scripts and work/tests

from fetal import config

BATCH = 4
LR = 2e-4
MAX_EPOCHS = 2 if config.SMOKE else 80
PATIENCE = 12
DEV_FRACTION = 0.12
