# Fetal Clinical & Imaging Dataset Inventory

This document provides an exhaustive, definitive audit of all datasets utilized across the fetal health project, spanning tabular clinical records (CDC NVSS) and ultrasound imaging modalities (HC18, Abdominal Circumference, Femur Length, Head Landmark Biometry, and FASSD multi-organ segmentation). All numbers and file splits are computed directly from real files on disk and verified against source pipelines.

Accompanying machine-readable audit: [`docs/results/data_inventory.json`](file:///C:/fetal-github-clone/work/docs/results/data_inventory.json).

---

## 1. CDC NVSS Vital Statistics Tabular (Stillbirth Risk Modelling)

### Provenance and Raw Data Architecture
- **Raw Source Files**: 16 annual data files (2016–2023) from the CDC National Vital Statistics System (NVSS):
  - **Natality Cohort (Live Births)**: 8 files (`2016_NATAL.csv` through `2023_NATAL.csv`), totaling **29,950,846** live birth records (2016: 3,956,112; 2017: 3,864,754; 2018: 3,801,534; 2019: 3,757,582; 2020: 3,619,826; 2021: 3,669,928; 2022: 3,676,029; 2023: 3,605,081).
  - **Fetal Death Cohort (Stillbirths)**: 8 files (`2016_FETAL_COD.csv` through `2023_FETAL_COD.csv`), totaling **358,175** fetal deaths (`OUTCOME = 1`).
  - *Location of Raw Files*: These 16 raw files (~25 GB) were processed in cloud storage (`Google Drive: /content/drive/MyDrive/fetal-health-project/data/raw/`) and are **not hosted on the local laptop disk**.
- **Modelling File Construction**:
  - The local modelling dataset `C:\fetal-data\cdc\clinical_training_data.csv` was constructed by drawing a random subsample of approximately 45,000 live births per annual natality cohort (`seed = 42`, total 359,961 live births) combined with **100% of all recorded fetal deaths** (358,175 stillbirths) from 2016 to 2023.
  - The combined records were shuffled (`seed = 42`) into a unified table of **718,136 rows and 80 columns** (live:stillbirth ratio $\approx 1:1$).

### Table 1: CDC Vital Statistics Cohort Breakdown

| Metric / Stratum | Cohort Filter | Live Births (`OUTCOME=0`) | Stillbirths (`OUTCOME=1`) | Total Records | Full File Path |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Complete Modelling Dataset** | Unfiltered | 359,961 (50.12%) | 358,175 (49.88%) | **718,136** | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Year 2016** | Unfiltered | 45,000 | 51,391 | 96,391 | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Year 2017** | Unfiltered | 45,000 | 49,170 | 94,170 | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Year 2018** | Unfiltered | 44,991 | 47,676 | 92,667 | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Year 2019** | Unfiltered | 44,997 | 46,007 | 91,004 | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Year 2020** | Unfiltered | 44,994 | 41,816 | 86,810 | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Year 2021** | Unfiltered | 44,993 | 42,428 | 87,421 | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Year 2022** | Unfiltered | 44,995 | 40,113 | 85,108 | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Year 2023** | Unfiltered | 44,991 | 39,574 | 84,565 | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **$\ge$ 20 Weeks Viability Filter** | `COMBGEST >= 20` | 359,843 (66.04%) | 185,077 (33.96%) | **544,920** | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Train Set ($\le$ 2021)** | `COMBGEST >= 20` | 269,888 (65.51%) | 142,105 (34.49%) | **411,993** | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Test Set ($\ge$ 2022)** | `COMBGEST >= 20` | 89,955 (67.67%) | 42,972 (32.33%) | **132,927** | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Deduplicated Filtered Cohort** | `load_clean` ($\ge 20$ wk) | 359,567 (67.54%) | 172,825 (32.46%) | **532,392** | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Deduplicated Train ($\le$ 2021)** | `load_clean` ($\ge 20$ wk) | 269,678 (67.04%) | 132,604 (32.96%) | **402,282** | `C:\fetal-data\cdc\clinical_training_data.csv` |
| **Deduplicated Test ($\ge$ 2022)** | `load_clean` ($\ge 20$ wk) | 89,889 (69.09%) | 40,221 (30.91%) | **130,110** | `C:\fetal-data\cdc\clinical_training_data.csv` |

### Feature Space & Column Removal Rationale (80 Total Columns)
From the 80 columns present in `clinical_training_data.csv`, exactly **15 feature columns** are selected for the primary clinical risk model (E1), and **65 columns are removed**.

- **Primary Model E1 Features (15 Features)**:
  - **5 Numeric Features**:
    1. `MAGER` (Maternal age in years)
    2. `BMI` (Pre-pregnancy body mass index, $kg/m^2$)
    3. `PRECARE` (Month prenatal care began; 0 = no prenatal care)
    4. `PRIORDEAD` (Number of prior fetal losses / stillbirths)
    5. `PRIORLIVE` (Number of prior live births)
  - **10 Categorical Features**:
    1. `RF_GDIAB` (Gestational diabetes: Y/N)
    2. `RF_GHYPE` (Gestational hypertension: Y/N)
    3. `RF_EHYPE` (Pre-pregnancy / chronic hypertension: Y/N)
    4. `RF_CESAR` (Previous cesarean delivery: Y/N)
    5. `RF_ARTEC` (Assisted reproductive technology / IVF: Y/N)
    6. `RF_INFTR` (Infertility treatment used: Y/N)
    7. `RF_FEDRG` (Fertility-enhancing drugs used: Y/N)
    8. `CIG_REC` (Maternal tobacco smoking recode: Y/N)
    9. `DPLURAL` (Plurality / multiple gestation: 1=Single, 2=Twin, 3+=Higher order)
    10. `SEX` (Infant sex: M/F)

- **Breakdown of 65 Removed Columns**:
  1. **Target and Partition Variables (2 columns)**:
     - `OUTCOME`: Prediction label (0 = live birth, 1 = fetal death / stillbirth).
     - `YEAR`: Calendar year of registration, reserved strictly for temporal hold-out partitioning ($\le 2021$ train vs. $\ge 2022$ test).
  2. **Data Leakage & Post-Outcome Variables (5 columns)**:
     - `COMBGEST`, `OEGest_Comb`, `OBGEST_FLG`, `DBWT`, `BWTR4`.
     - *Reason*: Measured at or after delivery. Using gestational age or birth weight directly to predict antepartum fetal death causes severe target leakage, as fetal deaths frequently occur preterm with restricted birth weight.
  3. **Labor, Delivery, Facility, and Attendant Variables (13 columns)**:
     - `ATTEND`, `BFACIL`, `BFACIL3`, `DLMP_MM`, `DLMP_YY`, `DMETH_REC`, `RDMETH_REC`, `ME_PRES`, `ME_ROUT`, `ME_TRIAL`, `MM_AICU`, `MM_RUPT`, `RESTATUS`.
     - *Reason*: Intrapartum and hospital delivery characteristics only observed during delivery; inapplicable for antepartum risk screening.
  4. **Reporting, Collection, and Imputation Flags (23 columns)**:
     - `F_CIGS_0`, `F_CIGS_1`, `F_CIGS_2`, `F_CIGS_3`, `F_MEDUC`, `F_MM_AICU`, `F_MPCB`, `F_M_HT`, `F_PWGT`, `F_RF_CESAR`, `F_RF_GDIAB`, `F_RF_GHYPER`, `F_RF_NCESAR`, `F_RF_PDIAB`, `F_RF_PHYPER`, `F_TOBACO`, `F_WIC`, `MAGE_IMPFLG`, `MAGE_REPFLG`, `IMP_PLUR`, `IMP_SEX`, `MRACEIMP`, `FAGERPT_FLG`.
     - *Reason*: Data administrative artifacts indicating whether a state reported or imputed a field; leak administrative processing rather than clinical pathophysiology.
  5. **Paternal Characteristics (2 columns)**:
     - `FAGECOMB`, `FAGEREC11`.
     - *Reason*: High rate of missingness on vital records ($>15\%$) and potential socio-demographic bias.
  6. **Redundant Recodes & Granular Bins (16 columns)**:
     - `BMI_R`, `CIG_0`, `CIG_1`, `CIG_2`, `CIG_3`, `MAGER9`, `MAGER14`, `M_Ht_In`, `PWgt_R`, `LBO_REC`, `ILLB_R`, `ILLB_R11`, `RF_CESARN`, `MRACE15`, `MRACEHISP`, `MHISPX`.
     - *Reason*: Direct collinear transformations or coarse binned variants of continuous variables already represented by primary numerical features (`BMI`, `MAGER`, `CIG_REC`).
  7. **Socioeconomic & Demographic Fairness Exclusions (4 columns)**:
     - `MEDUC` (Maternal education), `WIC` (WIC nutritional assistance), `MBSTATE_REC` (Maternal birth state/nativity), `MRACE6` (Maternal race/ethnicity).
     - *Reason*: Excluded from clinical baseline models to prevent algorithmic bias and discriminatory proxy disparities across protected demographic classes.

---

## 2. HC18 Grand Challenge (Head Circumference Biometry)

- **Source / Provenance**: HC18 Grand Challenge (van den Heuvel et al., *Medical Image Analysis*, 2018), Radboud University Medical Center, Nijmegen, Netherlands. Acquired on Voluson E8/E10 machines across all trimesters (10–40 weeks).
- **Partitioning**: 999 training images with ground truth ellipses; 335 official held-out test images evaluated through the Grand Challenge portal.

### Table 2: HC18 Dataset Specification

| Cohort | Images | Patients | Annotations / Ground Truth | Full File Path |
| :--- | :---: | :---: | :--- | :--- |
| **Training Set** | 999 | 806 | Ellipse parameters ($x, y, a, b, \alpha$), pixel size (mm), HC (mm) | `C:\fetal-data\HC18\training_set_pixel_size_and_HC.csv` |
| ↳ *Fold 0* | 206 (val) / 793 (train) | 166 (val) / 640 (train) | Patient-grouped 5-fold cross-validation split | `C:\fetal-github-clone\work\docs\results\hc18_nnunet\pooled_eval_5fold.json` |
| ↳ *Fold 1* | 199 (val) / 800 (train) | 160 (val) / 646 (train) | Patient-grouped 5-fold cross-validation split | `C:\fetal-github-clone\work\docs\results\hc18_nnunet\pooled_eval_5fold.json` |
| ↳ *Fold 2* | 208 (val) / 791 (train) | 160 (val) / 646 (train) | Patient-grouped 5-fold cross-validation split | `C:\fetal-github-clone\work\docs\results\hc18_nnunet\pooled_eval_5fold.json` |
| ↳ *Fold 3* | 194 (val) / 805 (train) | 160 (val) / 646 (train) | Patient-grouped 5-fold cross-validation split | `C:\fetal-github-clone\work\docs\results\hc18_nnunet\pooled_eval_5fold.json` |
| ↳ *Fold 4* | 192 (val) / 807 (train) | 160 (val) / 646 (train) | Patient-grouped 5-fold cross-validation split | `C:\fetal-github-clone\work\docs\results\hc18_nnunet\pooled_eval_5fold.json` |
| **Test Set** | 335 | 335 | Pixel size (mm); ground-truth ellipses held out by Challenge | `C:\fetal-data\HC18\test_set_pixel_size.csv` |
| **Total HC18** | **1,334** | **1,141** | 999 Train + 335 Challenge Test | `C:\fetal-data\HC18` |

---

## 3. Fetal Abdominal Circumference (AC)

- **Source / Provenance**:
  - **FP Cohort**: Fetal Planes Challenge dataset (Burgos-Artizzu et al., 2020), Hospital Clínic / BCNatal, Barcelona, Spain. Zenodo open access.
  - **UCL Cohort**: University College London Hospital fetal ultrasound cohort (Gimenez et al., 2024), London, UK. Zenodo open access.
- **Exclusion Audit**: Literature reports 823 images (FP: 693, UCL: 130). 31 UCL images lack physical calibration factors (`px_to_mm_rate`), and 3 images (Patient00125) have corrupted multi-point annotation layouts, leaving 792 calibrated images.
- **Partitioning**: 642 images across 501 patients form the 5-fold cross-validation development set (`folds_v1.csv`); 147 images form the held-out test set.

### Table 3: Fetal Abdominal Circumference Cohort Breakdown

| Cohort | FP Images | UCL Images | Total Images | Unique Patients | Full File Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw Available Annotations** | 693 | 130 | 823 | 648 | `C:\fetal-data\FP\annotations\Abdomen.csv` / `C:\fetal-data\UCL\annotations\px_to_mm\Abdomen.csv` |
| **Calibrated Usable Cohort** | 693 | 99 | 792 | 623 | `C:\fetal-data\FP\annotations\Abdomen.csv` / `C:\fetal-data\UCL\annotations\px_to_mm\Abdomen.csv` |
| **Training / CV Cohort** | 569 | 73 | **642** | **501** | `C:\fetal-github-clone\work\docs\results\ac_campaign\folds_v1.csv` |
| ↳ *Fold 1* | 126 | 16 | 142 | 101 | `C:\fetal-github-clone\work\docs\results\ac_campaign\folds_v1.csv` |
| ↳ *Fold 2* | 107 | 14 | 121 | 100 | `C:\fetal-github-clone\work\docs\results\ac_campaign\folds_v1.csv` |
| ↳ *Fold 3* | 110 | 14 | 124 | 100 | `C:\fetal-github-clone\work\docs\results\ac_campaign\folds_v1.csv` |
| ↳ *Fold 4* | 107 | 14 | 121 | 100 | `C:\fetal-github-clone\work\docs\results\ac_campaign\folds_v1.csv` |
| ↳ *Fold 5* | 119 | 15 | 134 | 100 | `C:\fetal-github-clone\work\docs\results\ac_campaign\folds_v1.csv` |
| **Held-Out Test Set** | 124 | 23 | **147** | 122 | `C:\fetal-data\FP\annotations\Abdomen_Test.csv` / `C:\fetal-data\UCL\annotations\px_to_mm\Abdomen_Test.csv` |
| **Total Calibrated AC** | **693** | **96** | **789** | **623** | `C:\fetal-data\FP\annotations` / `C:\fetal-data\UCL\annotations\px_to_mm` |

---

## 4. Fetal Femur Length (FL)

- **Source / Provenance**:
  - **FP Cohort**: Fetal Planes Challenge dataset (Burgos-Artizzu et al., 2020), Plane 5 (Femur diaphysis).
  - **UCL Cohort**: University College London Hospital fetal ultrasound cohort, Femur landmarks.
- **Exclusion Audit**: Literature reports 895 images (FP: 760, UCL: 135). 28 UCL images lack physical calibration factors (`px_to_mm_rate`), and 3 FP images (`Patient00888_Plane5_1_of_1.png`, `Patient01319_Plane5_1_of_1.png`, `Patient01423_Plane5_1_of_1.png`) lack millimeter calibration in `Femur.csv`, yielding **864 clean calibrated images** across 658 patients.

### Table 4: Fetal Femur Length Cohort Breakdown

| Split | FP Images | UCL Images | Total Images | Unique Patients | Full File Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Published / Raw Total** | 760 | 135 | 895 | 689 | `C:\fetal-data\FP\annotations\Femur.csv` / `C:\fetal-data\UCL\annotations\px_to_mm\Femur.csv` |
| **Calibrated Images** | 757 | 107 | 864 | 658 | `C:\fetal-github-clone\work\docs\results\femur\femur_data_followup.json` |
| **Train Set** | 382 | 69 | **451** | **345** | `C:\fetal-github-clone\work\docs\results\femur\femur_data_followup.json` |
| **Validation Set** | 52 | 13 | **65** | **47** | `C:\fetal-github-clone\work\docs\results\femur\femur_data_followup.json` |
| **Test Set** | 323 | 25 | **348** | **266** | `C:\fetal-github-clone\work\docs\results\femur\femur_data_followup.json` |
| **Total Clean Femur** | **757** | **107** | **864** | **658** | `C:\fetal-github-clone\work\docs\results\femur\femur_data_followup.json` |

---

## 5. Fetal Head Landmark Biometry (BPD & OFD Calipers)

- **Source / Provenance**:
  - **FP Cohort**: Fetal Planes Challenge (Burgos-Artizzu et al., 2020), Trans-thalamic and trans-ventricular standard cranial planes.
  - **UCL Cohort**: University College London Hospital fetal head caliper landmarks.
- **Calibration Audit**: Total published/raw images = 1,796 (FP: 1,637, UCL: 159). 33 UCL images lack millimeter calibration factors, leaving **1,763 calibrated images**.

### Table 5: Fetal Head Landmark Cohort Breakdown

| Split | FP Images | UCL Images | Total Images | Unique Patients | Full File Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw Annotations** | 1,637 | 159 | 1,796 | 940 | `C:\fetal-data\FP\annotations\Head.csv` / `C:\fetal-data\UCL\annotations\px_to_mm\Head.csv` |
| **Calibrated Usable Total** | 1,637 | 126 | **1,763** | **940** | `C:\fetal-data\FP\annotations\Head.csv` / `C:\fetal-data\UCL\annotations\px_to_mm\Head.csv` |
| **Train Set** | 757 (449 pts) | 91 (26 pts) | **848** | **475** | `C:\fetal-data\FP\annotations\Head_Train.csv` / `C:\fetal-data\UCL\annotations\px_to_mm\Head_Train.csv` |
| **Test Set** | 880 (460 pts) | 35 (5 pts) | **915** | **465** | `C:\fetal-data\FP\annotations\Head_Test.csv` / `C:\fetal-data\UCL\annotations\px_to_mm\Head_Test.csv` |
| **Total Calibrated Head** | **1,637** | **126** | **1,763** | **940** | `C:\fetal-data\FP\annotations` / `C:\fetal-data\UCL\annotations\px_to_mm` |

---

## 6. FASSD (Fetal Abdominal Structures Segmentation Dataset)

- **Source / Provenance**: FASSD Dataset (Xu et al., *Scientific Data*, 2023; Kaggle `prantakumerpandit390/fassd-fetal-abdomen-v2`), multi-center ultrasound segmentation covering 4 distinct abdominal anatomical structures: **artery**, **liver**, **stomach bubble**, and **umbilical vein**.
- **Images and Arrays**: Exactly 1,588 PNG ultrasound images and 1,588 matching NumPy `.npy` 4-channel binary masks across 169 patients.

### Table 6: FASSD Dataset Splits & Anatomy

| Split Partition | Images / Masks | Unique Patients | Target Segmentations | Full File Path |
| :--- | :---: | :---: | :--- | :--- |
| **Split v1 - Train** | 1,100 | 118 | Artery, Liver, Stomach, Vein | `C:\fetal-github-clone\work\docs\results\fassd\split_v1.csv` |
| **Split v1 - Validation** | 236 | 25 | Artery, Liver, Stomach, Vein | `C:\fetal-github-clone\work\docs\results\fassd\split_v1.csv` |
| **Split v1 - Test** | 252 | 26 | Artery, Liver, Stomach, Vein | `C:\fetal-github-clone\work\docs\results\fassd\split_v1.csv` |
| **Split v2 - Train** | 1,063 | 118 | Artery, Liver, Stomach, Vein | `C:\fetal-github-clone\work\docs\results\fassd\split_v2.csv` |
| **Split v2 - Validation** | 254 | 26 | Artery, Liver, Stomach, Vein | `C:\fetal-github-clone\work\docs\results\fassd\split_v2.csv` |
| **Split v2 - Test** | 271 | 25 | Artery, Liver, Stomach, Vein | `C:\fetal-github-clone\work\docs\results\fassd\split_v2.csv` |
| **Total FASSD** | **1,588** | **169** | Complete 4-organ anatomical masks | `C:\fetal-github-clone\work\docs\results\fassd\fassd_audit.json` |

---

## 7. Master File Paths & Disk Footprint Reference

| Modality / Target | Primary Data Files & Audit Sources (Full Absolute Local Paths) | Local Directory | Size on Disk (MB) | Size on Disk (GB) |
| :--- | :--- | :--- | :---: | :---: |
| **CDC Tabular** | `C:\fetal-data\cdc\clinical_training_data.csv`<br>`C:\fetal-github-clone\work\docs\results\data_inventory.json` | `C:\fetal-data\cdc\` | 172.95 MB | 0.169 GB |
| **HC18 Challenge** | `C:\fetal-data\HC18\training_set_pixel_size_and_HC.csv`<br>`C:\fetal-data\HC18\test_set_pixel_size.csv`<br>`C:\fetal-github-clone\work\docs\results\hc18_nnunet\pooled_eval_5fold.json` | `C:\fetal-data\HC18\` | 336.66 MB | 0.329 GB |
| **Abdominal Circumference** | `C:\fetal-data\FP\annotations\Abdomen.csv`<br>`C:\fetal-data\UCL\annotations\px_to_mm\Abdomen.csv`<br>`C:\fetal-github-clone\work\docs\results\ac_campaign\folds_v1.csv` | `C:\fetal-data\FP\`<br>`C:\fetal-data\UCL\` | 527.25 MB<br>109.60 MB | 0.515 GB<br>0.107 GB |
| **Femur Length** | `C:\fetal-data\FP\annotations\Femur.csv`<br>`C:\fetal-data\UCL\annotations\px_to_mm\Femur.csv`<br>`C:\fetal-github-clone\work\docs\results\femur\femur_data_followup.json` | Shared FP/UCL images;<br>`C:\fetal-data\kaggle_femur_dataset\` | 146.97 MB | 0.144 GB |
| **Head Landmarks** | `C:\fetal-data\FP\annotations\Head.csv`<br>`C:\fetal-data\FP\annotations\Head_Train.csv`<br>`C:\fetal-data\FP\annotations\Head_Test.csv`<br>`C:\fetal-data\UCL\annotations\px_to_mm\Head.csv`<br>`C:\fetal-data\UCL\annotations\px_to_mm\Head_Train.csv`<br>`C:\fetal-data\UCL\annotations\px_to_mm\Head_Test.csv` | Shared FP/UCL images;<br>`C:\fetal-data\kaggle_head_dataset\` | 332.51 MB | 0.325 GB |
| **FASSD Segmentation** | `C:\fetal-github-clone\work\docs\results\fassd\fassd_audit.json`<br>`C:\fetal-github-clone\work\docs\results\fassd\split_v1.csv`<br>`C:\fetal-github-clone\work\docs\results\fassd\split_v2.csv` | Cloud hosted on Kaggle (`fassd-fetal-abdomen-v2`) | ~1.2 GB (Cloud) | 0.00 GB (Local) |
| **nnU-Net Datasets** | `C:\fetal-data\nnunet_ac\Dataset502_AC\`<br>`C:\fetal-data\nnUNet_raw\Dataset501_HC18\` | `C:\fetal-data\nnunet_ac\`<br>`C:\fetal-data\nnUNet_raw\` | 151.22 MB<br>133.48 MB | 0.148 GB<br>0.130 GB |
