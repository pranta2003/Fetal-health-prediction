# Growth-Bridged Integration of Ultrasound Biometry and Maternal Clinical Data for Explainable Fetal Health and Stillbirth Risk Prediction

Capstone project - BUBT CSE - Six-layer, three-stream architecture for fetal health / stillbirth risk prediction.

## Project structure

```
data/
  raw/            <- put original downloaded datasets here (not committed to git - see .gitignore)
  processed/       <- cleaned/preprocessed data goes here
src/
  stream1_image/   <- Deep Image Model: head (ConvNeXt-Tiny), abdomen (nnU-Net), femur (ConvNeXt-Tiny)
  stream2_growth/  <- Growth Centile Model + Hadlock formula / growth percentile logic (Layer 3)
  stream3_clinical/<- Clinical Risk Model (XGBoost) on CDC data
  fusion/          <- Transformer-based cross-attention fusion (Layer 4)
  xai/             <- SHAP + Grad-CAM explainability (Layer 5)
notebooks/         <- exploratory Kaggle/Colab notebooks
models/            <- saved trained model weights (not committed to git)
```

## Build order

1. Stream 3 (Clinical Risk Model) - CDC tabular data, XGBoost. **STATUS: pipeline scaffolded and verified working on synthetic data.**
2. Layer 3 (Hadlock formula + growth percentile) - pure calculation, no training.
3. Stream 1 - head sub-model (HC18 -> ConvNeXt-Tiny regression)
4. Stream 1 - abdomen sub-model (FASSD -> nnU-Net segmentation)
5. Stream 1 - femur sub-model (FETAL_PLANES_DB / FPUS23 interim, pending BabyNet++ supplementary dataset)
6. Stream 2 (Growth Centile Model) - trained on Layer 3 outputs
7. Layer 4 (Transformer cross-attention fusion of all 3 streams)
8. Layer 5 (SHAP + Grad-CAM explainability)
9. Layer 6 (final risk output + clinical decision-support report / dashboard)

## Datasets needed (download separately, not included in repo)

| Dataset | Where to get it | Goes in |
|---|---|---|
| CDC/NCHS Fetal Death Public Use File | cdc.gov/nchs/data_access/vitalstatsonline.htm | data/raw/cdc_fetal_death.csv |
| HC18 | zenodo.org/records/1327317 | data/raw/hc18/ |
| FASSD | data.mendeley.com/datasets/4gcpm9dsc3/1 | data/raw/fassd/ |
| FETAL_PLANES_DB | zenodo.org/records/3904280 | data/raw/fetal_planes_db/ |
| FPUS23 | github.com/bharathprabakaran/FPUS23 | data/raw/fpus23/ |
| Supplementary Femur+GA (BabyNet++) | pending - Dr. Szymon Plotka | data/raw/babynet_femur_ga.csv |

## Platform notes

- Image model training (Stream 1) -> run on Kaggle Notebooks (free GPU quota, dataset hosting)
- Clinical model (Stream 3) -> run locally, CPU is enough
- Colab -> backup when Kaggle GPU quota runs out

## Setup

```bash
pip install -r requirements.txt
```
