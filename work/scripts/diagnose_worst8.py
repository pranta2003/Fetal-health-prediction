# FIELD: General
# MODEL: None
# STEP: utility
# USED BY: work/docs/CODE_MAP.md

import pandas as pd
from pathlib import Path

oof = pd.read_csv(r'E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\ac_baseline_s42\ac-baseline-s42\oof_predictions_pooled.csv')
worst = oof.sort_values('ac_err', ascending=False).head(8)
for _, r in worst.iterrows():
    print(f"{r['image_name']}: source={r['source']}, ac_err={r['ac_err']:.2f}, tad_err={r['tad_err']:.2f}, apad_err={r['apad_err']:.2f}, ac_true={r['ac_true']:.2f}, ac_pred={r['ac_pred']:.2f}")
