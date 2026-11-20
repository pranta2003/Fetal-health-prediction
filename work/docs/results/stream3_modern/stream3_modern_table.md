# Stream 3 Modern Tabular Campaign Results (E1 Protocol)

- **Protocol**: Temporal split (train <= 2021 [N=20142], test >= 2022 [N=6477])
- **Features**: 15 clinical & pre-pregnancy variables (zero outcome-time leakage)
- **Hyperparameter Tuning**: Optuna (2 trials)

| Model | Test ROC-AUC | 95% Bootstrap CI | Description |
|---|:---:|:---:|---|
| **Tuned XGBoost** | **0.7226** | [0.7084, 0.7320] | Depth, lr, subsample tuned |
| **Tuned LightGBM** | **0.7301** | [0.7169, 0.7396] | Leaves, depth, lr tuned |
| **Tuned CatBoost** | **0.7296** | [0.7172, 0.7381] | Depth, l2_reg, lr tuned |
| **Tuned Logistic Regression** | **0.6658** | [0.6543, 0.6749] | Standardized, C=0.05 |
| **Calibrated GBM+LR Blend** | **0.7301** | [0.7173, 0.7389] | `0.85*lightgbm + 0.15*LR` + Isotonic Calibration |
