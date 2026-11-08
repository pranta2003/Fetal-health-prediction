# HC18 Historical Approach Ladder (Grouped Fold 1)

Evaluation on the exact 206 held-out fold-1 images (`make_grouped_folds`, seed 42) with inner 10% patient-isolation validation.

| Row | Approach | Model / Backbone | Input Res | Target Type | Fold 1 HC MAE (mm) | Fold 1 HC RMSE (mm) | Fold 1 BPD MAE (mm) | Fold 1 BPD RMSE (mm) | Epochs | Train Time (s) |
|:---:|:---|:---|:---:|:---|---:|---:|---:|---:|---:|---:|
| A | Direct continuous scalar regression | ConvNeXt-Tiny | 224x224 (direct) | Row A: Direct Regression | **10.6728** | 15.9672 | **3.2616** | 4.7343 | 25 | 695.4 |
| B | Ellipse parameter regression (Huber loss) | ConvNeXt-Tiny | 224x224 (letterbox) | Row B: Geometry Approach v3 | **11.5964** | 16.1775 | **3.2198** | 4.5392 | 20 | 307.4 |
| C | Segmentation + shared-encoder ellipse regression | ResNet-34 | 224x224 (direct) | Row C: Multi-Task U-Net | **10.5104** | 15.5098 | **2.9474** | 4.3227 | 52 | 548.1 |
| D | Segmentation mask + robust native ellipse fit | ResNet-34 | 224x224 (letterbox) | Row D: ResNet-34 U-Net | **1.9246** | 2.4351 | **0.6003** | 0.8316 | 48 | 495.6 |
| E | High-res segmentation + TTA + fitEllipseAMS | EfficientNet-B4 | 384x384 (letterbox) | Row E: EfficientNet-B4 U-Net 384 | **2.4817** | 5.0996 | **0.6982** | 0.9200 | - | - |
