# Stream 3 Modern Tabular Campaign Results (E1 Protocol)

- **Protocol**: Temporal split (train <= 2021 [N=402282], test >= 2022 [N=130110])
- **Features**: 15 clinical & pre-pregnancy variables (zero outcome-time leakage)
- **Hyperparameter Tuning**: Optuna (40 trials)

| Model | Test ROC-AUC | 95% Bootstrap CI | Description |
|---|:---:|:---:|---|
| **Tuned XGBoost** | **0.7382** | [0.7349, 0.7412] | Depth, lr, subsample tuned |
| **Tuned LightGBM** | **0.7383** | [0.7351, 0.7412] | Leaves, depth, lr tuned |
| **Tuned CatBoost** | **0.7383** | [0.7350, 0.7414] | Depth, l2_reg, lr tuned |
| **Tuned Logistic Regression** | **0.6659** | [0.6624, 0.6691] | Standardized, C=0.01 |
| **Calibrated GBM+LR Blend** | **0.7382** | [0.7350, 0.7412] | `0.90*lightgbm + 0.10*LR` + Isotonic Calibration |
