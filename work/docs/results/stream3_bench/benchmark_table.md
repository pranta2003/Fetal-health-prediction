# Stream 3 Clinical Risk Model Benchmark

Comparison of model families under the exact same protocol: CDC natality/stillbirth data, temporal split (Train <= 2021, Test >= 2022), no post-delivery leakage.

| Setting | Model | Train Rows | Test Rows | Test AUC | 95% Bootstrap CI | Fit Time (s) |
|:---|:---|---:|---:|---:|:---:|---:|
| E1 (Primary, >=20 wk) | Logistic Regression | 402,282 | 130,110 | **0.6659** | [0.6629, 0.6691] | 5.7 |
| E1 (Primary, >=20 wk) | Random Forest | 402,282 | 130,110 | **0.7322** | [0.7297, 0.7354] | 50.5 |
| E1 (Primary, >=20 wk) | LightGBM | 402,282 | 130,110 | **0.7380** | [0.7357, 0.7413] | 10.8 |
| E1 (Primary, >=20 wk) | Our XGBoost | 402,282 | 130,110 | **0.7381** | [0.7358, 0.7413] | 22.6 |
| E4 (Clinical-Only, >=20 wk) | Logistic Regression | 402,282 | 130,110 | **0.6585** | [0.6557, 0.6617] | 4.8 |
| E4 (Clinical-Only, >=20 wk) | Random Forest | 402,282 | 130,110 | **0.7043** | [0.7013, 0.7075] | 41.8 |
| E4 (Clinical-Only, >=20 wk) | LightGBM | 402,282 | 130,110 | **0.7070** | [0.7041, 0.7101] | 12.0 |
| E4 (Clinical-Only, >=20 wk) | Our XGBoost | 402,282 | 130,110 | **0.7070** | [0.7040, 0.7101] | 19.5 |
