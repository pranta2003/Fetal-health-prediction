# Stream 3 Clinical Risk Model Benchmark

Comparison of model families under the exact same protocol: CDC natality/stillbirth data, temporal split (Train <= 2021, Test >= 2022), no post-delivery leakage.

| Setting | Model | Train Rows | Test Rows | Test AUC | 95% Bootstrap CI | Fit Time (s) |
|:---|:---|---:|---:|---:|:---:|---:|
| E1 (Primary, >=20 wk) | Logistic Regression | 402,282 | 130,110 | **0.6659** | [0.6629, 0.6691] | 8.5 |
| E1 (Primary, >=20 wk) | Random Forest | 402,282 | 130,110 | **0.7288** | [0.7263, 0.7324] | 36.2 |
| E1 (Primary, >=20 wk) | LightGBM | 402,282 | 130,110 | **0.7380** | [0.7357, 0.7413] | 9.3 |
| E1 (Primary, >=20 wk) | Our XGBoost | 402,282 | 130,110 | **0.7381** | [0.7358, 0.7413] | 17.5 |
| E4 (Clinical-Only, >=20 wk) | Logistic Regression | 402,282 | 130,110 | **0.6585** | [0.6557, 0.6617] | 4.7 |
| E4 (Clinical-Only, >=20 wk) | Random Forest | 402,282 | 130,110 | **0.7034** | [0.7004, 0.7068] | 30.9 |
| E4 (Clinical-Only, >=20 wk) | LightGBM | 402,282 | 130,110 | **0.7070** | [0.7041, 0.7101] | 9.4 |
| E4 (Clinical-Only, >=20 wk) | Our XGBoost | 402,282 | 130,110 | **0.7070** | [0.7040, 0.7101] | 16.6 |
