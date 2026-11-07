# Capstone Historical Notebook Results

Source notebook: `Capstone_fetalDataset_history.ipynb`

Extracted 14 cells containing metric lines (MAE, RMSE, AUC, NME) verbatim from execution outputs.

## Cell 8

- **Nearest Heading**: `#Stram - 3 value check and AUC check from tabular dataset`
- **Extracted Lines**:

```text
AUC-ROC: 0.9684
```

## Cell 19

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
Best val_loss achieved: 124.6242 (RMSE ~ 11.2 mm combined)
```

## Cell 21

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
HC  -> MAE: 10.22mm, RMSE: 16.07mm
BPD -> MAE: 3.03mm, RMSE: 4.75mm
```

## Cell 22

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
HC  -> MAE: 10.96mm, RMSE: 16.40mm
BPD -> MAE: 3.14mm, RMSE: 4.71mm
```

## Cell 23

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
[GEOMETRY APPROACH] HC  -> MAE: 9.22mm, RMSE: 12.11mm
[GEOMETRY APPROACH] BPD -> MAE: 2.61mm, RMSE: 3.41mm
```

## Cell 25

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
[MULTI-TASK APPROACH] HC  -> MAE: 10.81mm, RMSE: 14.29mm
[MULTI-TASK APPROACH] BPD -> MAE: 2.89mm, RMSE: 4.03mm
```

## Cell 26

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
[SEGMENTATION APPROACH] HC  -> MAE: 34.75mm, RMSE: 37.50mm
[SEGMENTATION APPROACH] BPD -> MAE: 10.83mm, RMSE: 12.57mm
```

## Cell 27

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
[GEOMETRY APPROACH v3] HC  -> MAE: 8.79mm, RMSE: 12.99mm
[GEOMETRY APPROACH v3] BPD -> MAE: 2.67mm, RMSE: 3.87mm
```

## Cell 31

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
Fold 1 solo -> HC MAE: 2.86mm, BPD MAE: 0.87mm
Fold 2 solo -> HC MAE: 2.82mm, BPD MAE: 0.72mm
Fold 3 solo -> HC MAE: 2.70mm, BPD MAE: 0.89mm
Fold 4 solo -> HC MAE: 2.70mm, BPD MAE: 0.86mm
Fold 5 solo -> HC MAE: 3.15mm, BPD MAE: 0.97mm
Per-fold HC MAE: ['2.86', '2.82', '2.70', '2.70', '3.15']
[5-FOLD ENSEMBLE] HC  -> MAE: 2.31mm, RMSE: 2.49mm
[5-FOLD ENSEMBLE] BPD -> MAE: 0.68mm, RMSE: 0.73mm
```

## Cell 32

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
Fold 1 solo (TTA) -> HC MAE: 2.16mm, BPD MAE: 0.75mm
Fold 2 solo (TTA) -> HC MAE: 2.24mm, BPD MAE: 0.67mm
Fold 3 solo (TTA) -> HC MAE: 1.92mm, BPD MAE: 0.64mm
Fold 4 solo (TTA) -> HC MAE: 2.14mm, BPD MAE: 0.76mm
Fold 5 solo (TTA) -> HC MAE: 2.26mm, BPD MAE: 0.71mm
Per-fold HC MAE: ['2.16', '2.24', '1.92', '2.14', '2.26']
[OOF] HC  -> MAE: 2.14mm, RMSE: 2.87mm
[OOF] BPD -> MAE: 0.71mm, RMSE: 0.93mm
```

## Cell 34

- **Nearest Heading**: `# HC18 dataset`
- **Extracted Lines**:

```text
HC  -> MAE: 2.06mm, RMSE: 2.82mm
BPD -> MAE: 0.66mm, RMSE: 0.91mm
```

## Cell 51

- **Nearest Heading**: `#From this part , work start on Stream 2 , FASSD and FP , UCL Dataset . All the best to me !`
- **Extracted Lines**:

```text
TAD  -> MAE: 57.98mm, RMSE: 62.02mm
APAD -> MAE: 57.46mm, RMSE: 61.50mm
AC   -> MAE: 182.60mm, RMSE: 195.13mm
```

## Cell 58

- **Nearest Heading**: `##Checking some defect image in Abdomen dataset`
- **Extracted Lines**:

```text
  AC  abs err mm   n= 789 mean=  5.18 median=  2.98 rmse=  9.07 p90= 11.17 p95= 17.52 p99= 38.92 max=  76.20
  TAD abs err mm   n= 789 mean=  1.88 median=  1.05 rmse=  3.67 p90=  4.10 p95=  5.90 p99= 15.00 max=  40.32
  APAD abs err mm  n= 789 mean=  2.17 median=  1.30 rmse=  3.70 p90=  4.87 p95=  7.07 p99= 14.91 max=  30.01
  landmark err mm  n= 789 mean=  4.64 median=  3.22 rmse=  6.37 p90=  9.84 p95= 13.09 p99= 20.42 max=  43.63
  AC  err %        n= 789 mean=  2.60 median=  1.70 rmse=  4.09 p90=  5.27 p95=  7.81 p99= 15.57 max=  26.36
  AC MAE 5.18 mm | RMSE 9.07 mm | median 2.98 mm
  NME (landmark err / true diameter): TAD 0.0743 | APAD 0.0784 | mean 0.0764
    FP                           AC mm: n= 693 mean=  5.08 median=  3.07 rmse=  8.59 p90= 11.15 p95= 16.70 p99= 34.20 max=  76.20 | NME 0.0802
    UCL                          AC mm: n=  96 mean=  5.91 median=  2.48 rmse= 11.99 p90= 11.15 p95= 25.15 p99= 46.84 max=  69.45 | NME 0.0483
    test                         AC mm: n= 147 mean=  4.92 median=  2.92 rmse=  7.68 p90= 11.04 p95= 15.00 p99= 27.40 max=  45.01 | NME 0.0738
    train                        AC mm: n= 642 mean=  5.24 median=  3.00 rmse=  9.36 p90= 11.17 p95= 18.68 p99= 40.01 max=  76.20 | NME 0.0769
    NA                           AC mm: n= 789 mean=  5.18 median=  2.98 rmse=  9.07 p90= 11.17 p95= 17.52 p99= 38.92 max=  76.20 | NME 0.0764
    rate_low                     AC mm: n= 263 mean=  3.47 median=  2.19 rmse=  5.69 p90=  6.96 p95= 10.89 p99= 24.49 max=  34.20 | NME 0.0783
    rate_mid                     AC mm: n= 278 mean=  3.96 median=  2.53 rmse=  6.41 p90=  7.71 p95= 10.28 p99= 27.69 max=  41.64 | NME 0.0727
    rate_high                    AC mm: n= 248 mean=  8.36 median=  5.21 rmse= 13.47 p90= 17.77 p95= 26.32 p99= 54.20 max=  76.20 | NME 0.0784
    (148.176, 161.781]           AC mm: n= 197 mean=  4.05 median=  2.61 rmse=  6.53 p90=  8.00 p95= 11.63 p99= 23.36 max=  41.64 | NME 0.0792
    (161.781, 244.238]           AC mm: n= 197 mean=  4.15 median=  2.71 rmse=  6.40 p90=  7.98 p95= 10.68 p99= 27.15 max=  30.84 | NME 0.0690
    (244.238, 391.075]           AC mm: n= 197 mean=  9.54 median=  5.64 rmse= 14.98 p90= 19.94 p95= 32.31 p99= 61.10 max=  76.20 | NME 0.0798
    (68.615, 148.176]            AC mm: n= 198 mean=  2.99 median=  1.89 rmse=  4.65 p90=  6.32 p95=  8.19 p99= 20.62 max=  25.74 | NME 0.0773
    no_marker                    AC mm: n= 722 mean=  5.06 median=  3.07 rmse=  8.50 p90= 11.04 p95= 16.58 p99= 34.10 max=  76.20 | NME 0.0807
    yellow_marker                AC mm: n=  67 mean=  6.46 median=  2.20 rmse= 13.82 p90= 19.33 p95= 30.50 p99= 53.75 max=  69.45 | NME 0.0298
    1                            AC mm: n= 158 mean=  5.35 median=  3.37 rmse=  8.80 p90= 13.37 p95= 20.66 p99= 33.05 max=  45.01 | NME 0.0751
    2                            AC mm: n= 158 mean=  5.09 median=  3.04 rmse=  9.26 p90= 10.38 p95= 13.81 p99= 35.68 max=  69.45 | NME 0.0729
    3                            AC mm: n= 158 mean=  5.66 median=  2.74 rmse= 10.78 p90= 12.36 p95= 18.92 p99= 43.37 max=  76.20 | NME 0.0763
    4                            AC mm: n= 158 mean=  4.91 median=  2.99 rmse=  7.70 p90= 10.45 p95= 13.79 p99= 32.28 max=  38.65 | NME 0.0784
    5                            AC mm: n= 157 mean=  4.88 median=  2.75 rmse=  8.54 p90= 11.13 p95= 17.27 p99= 29.12 max=  60.76 | NME 0.0791
```

## Cell 75

- **Nearest Heading**: `##Checking s5 results after completing kaggle dataset upload`
- **Extracted Lines**:

```text
s1         AC MAE 4.96 RMSE 7.33 med 3.14 bias +0.96 >10mm 12.2% | TAD 0.078+-0.063 APAD 0.081+-0.064
s2         AC MAE 4.32 RMSE 6.05 med 2.73 bias -0.99 >10mm 8.8% | TAD 0.077+-0.060 APAD 0.077+-0.062
s3         AC MAE 4.54 RMSE 6.59 med 3.37 bias -0.74 >10mm 10.2% | TAD 0.079+-0.060 APAD 0.079+-0.063
s4         AC MAE 4.49 RMSE 6.27 med 3.64 bias -0.49 >10mm 8.2% | TAD 0.076+-0.052 APAD 0.077+-0.057
res512     AC MAE 5.02 RMSE 8.01 med 3.32 bias -1.53 >10mm 11.6% | TAD 0.075+-0.054 APAD 0.081+-0.066
ENS_mean   AC MAE 4.47 RMSE 6.31 med 3.01 bias -0.53 >10mm 9.5% | TAD 0.075+-0.060 APAD 0.075+-0.062
ENS_median AC MAE 4.34 RMSE 6.03 med 3.12 bias -0.60 >10mm 6.8% | TAD 0.076+-0.060 APAD 0.075+-0.061
s1         AC MAE 4.96 RMSE 7.33 med 3.14 bias +0.96 >10mm 12.2% | TAD 0.078+-0.063 APAD 0.081+-0.064
s2         AC MAE 4.32 RMSE 6.05 med 2.73 bias -0.99 >10mm 8.8% | TAD 0.077+-0.060 APAD 0.077+-0.062
s3         AC MAE 4.54 RMSE 6.59 med 3.37 bias -0.74 >10mm 10.2% | TAD 0.079+-0.060 APAD 0.079+-0.063
s4         AC MAE 4.49 RMSE 6.27 med 3.64 bias -0.49 >10mm 8.2% | TAD 0.076+-0.052 APAD 0.077+-0.057
res512     AC MAE 5.02 RMSE 8.01 med 3.32 bias -1.53 >10mm 11.6% | TAD 0.075+-0.054 APAD 0.081+-0.066
ENS_mean   AC MAE 4.47 RMSE 6.31 med 3.01 bias -0.53 >10mm 9.5% | TAD 0.075+-0.060 APAD 0.075+-0.062
ENS_median AC MAE 4.34 RMSE 6.03 med 3.12 bias -0.60 >10mm 6.8% | TAD 0.076+-0.060 APAD 0.075+-0.061
```

