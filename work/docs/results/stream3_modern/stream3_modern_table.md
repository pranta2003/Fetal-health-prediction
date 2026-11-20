# Stream 3 Modern Tabular Campaign Results (E1 Protocol)

- **Protocol**: Temporal split (train <= 2021 [N=20142], test >= 2022 [N=6477])
- **Features**: 15 clinical & pre-pregnancy variables (zero outcome-time leakage)
- **Hyperparameter Tuning**: Optuna (2 trials)

| Model | Test ROC-AUC | 95% Bootstrap CI | Description |
|---|:---:|:---:|---|
| **Tuned XGBoost** | **0.7233** | [0.7094, 0.7328] | Depth, lr, subsample tuned |
| **Tuned LightGBM** | **0.7289** | [0.7163, 0.7375] | Leaves, depth, lr tuned |
| **Tuned CatBoost** | **0.7153** | [0.7013, 0.7256] | Depth, l2_reg, lr tuned |
| **Tuned Logistic Regression** | **0.6658** | [0.6543, 0.6749] | Standardized, C=0.05 |
| **Calibrated GBM+LR Blend** | **0.7298** | [0.7168, 0.7378] | `0.90*lightgbm + 0.10*LR` + Isotonic Calibration |
