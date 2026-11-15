# FASSD Data Audit and Patient-Grouped Split Report

## 1. Dataset Overview & Pairwise Agreement
- **Dataset**: FASSD (Fetal Abdominal Structures Segmentation Dataset)
- **Total Images**: 1588 PNG files, 1588 NPY files
- **Pairwise Representation Difference**: Max absolute diff = **0** across all 1,588 pairs
- **Array Shapes**: 1507 images of shape (768, 1024, 3), 81 images of shape (768, 1024)
- **Format Note**: ARRAY_FORMAT .npy files store 1,507 samples as 3D BGR arrays of shape (768, 1024, 3) and 81 samples as 2D grayscale arrays of shape (768, 1024). Corresponding IMAGES/*.png files match byte-for-byte in pixel values when loaded with cv2.IMREAD_UNCHANGED (max absolute diff = 0). cv2.imread with default IMREAD_COLOR automatically expands the 81 2D images to 3 identical channels.

## 2. Patient ID Parsing & Zero-Padding Integrity
- **Unique Patients**: 169 (expected 169)
- **Patient ID Width Distribution**:
  - Width 3 (`P01`–`P09`): 9 patients (71 images)
  - Width 4 (`P010`–`P099`): 90 patients (878 images)
  - Width 5 (`P0100`–`P0169`): 70 patients (639 images)
- **Images per Patient**: Min = 3, Median = 9.0, Max = 17
- **Zero-Padding Collisions**: None detected (Zero zero-padding collisions exist: each numeric patient index (1..169) maps uniquely to exactly one prefix (P01..P09, P010..P099, P0100..P0169).)

## 3. Structure Masks & Cross-Structure Overlap

| Structure | Non-Empty Masks | Empty Masks | Area Mean (%) | Area Median (%) | Area Min (%) | Area Max (%) | Blobs (Median / Max) |
|---|---|---|---|---|---|---|---|
| `artery` | 1572 | 16 | 0.129% | 0.113% | 0.0390% | 0.92% | 1.0 / 1 |
| `liver` | 1587 | 1 | 3.673% | 3.505% | 0.0992% | 8.53% | 1.0 / 1 |
| `stomach` | 1518 | 70 | 0.680% | 0.570% | 0.0088% | 5.03% | 1.0 / 2 |
| `vein` | 1582 | 6 | 0.519% | 0.502% | 0.0221% | 1.44% | 1.0 / 3 |

### Cross-Structure Overlap (Pixel Counts across 1,588 images)
| Structure Pair | Overlapping Pixels | Interpretation |
|---|---|---|
| `artery` vs `liver` | 25 | Minor contour boundary contact |
| `artery` vs `stomach` | 38 | Minor contour boundary contact |
| `artery` vs `vein` | 0 | Disjoint structures (zero overlap) |
| `liver` vs `stomach` | 97 | Organ boundary contact |
| `liver` vs `vein` | 125047 | **Anatomical overlap**: intrahepatic umbilical vein / ductus venosus traversing liver |
| `stomach` vs `vein` | 76 | Minor boundary proximity |

## 4. Exact Duplicate Analysis & Data Leakage Audit
- **Unique PNG Hashes**: 1563 / 1588
- **Within-Patient Duplicate Pairs**: 12 pairs (24 images total)
- **Across-Patient Duplicate Pairs**: 13 pairs (26 images total)

### Across-Patient Duplicate Pairs (Leakage Risks)
| Patient A | Patient B | Number of Duplicate Images | Specific Filenames |
|---|---|---|---|
| `P0119` | `P019` | 4 | `P0119_IMG1..4` == `P019_IMG10..13` |
| `P058` | `P0128` | 5 | `P0128_IMG1..5` == `P058_IMG1..5` |
| `P031` | `P033` | 4 | `P031_IMG7..10` == `P033_IMG7..10` |

> [!WARNING] **Data Leakage Danger**: Found 13 across-patient duplicate image pairs across 3 patient pairs (P0119 & P019 [4 images], P058 & P0128 [5 images], P031 & P033 [4 images]). These represent identical ultrasound frames labeled under different patient IDs, posing a data leakage risk.

## 5. Image Intensity & Color Channel Check
- **Fraction of Pixels with Channel Differences > 0**: 0.3368
- **Images with Color Content**: 1507 / 1588
- **Grayscale Images**: 81 / 1588
- **Explanation**: Ultrasonic frames in FASSD are primarily grayscale ultrasound sectors; however, 1,507 images contain color content (33.68% of all pixels across the dataset have channel differences > 0). The colored regions correspond to Color Doppler flow velocity mapping (red/orange and blue flow encoding) within major fetal blood vessels, as well as ultrasound machine UI text, calipers, and orientation markers.

## 6. Patient-Grouped Split (Seed 42)
- **Split File**: `docs/results/fassd/split_v1.csv`
- **Patients**: Train = 118 (1100 imgs), Val = 25 (236 imgs), Test = 26 (252 imgs)
- **Patient Overlap**: Strictly 0 (enforced and asserted)
- **Structure Representation per Split** (all >= 20 images):

| Split | Patients | Total Images | Artery | Liver | Stomach | Vein |
|---|---|---|---|---|---|---|
| `train` | 118 | 1100 | 1093 | 1100 | 1061 | 1095 |
| `val` | 25 | 236 | 230 | 236 | 218 | 235 |
| `test` | 26 | 252 | 249 | 251 | 239 | 252 |

## 7. Sample Visualizations (12 overlays)
Generated 12 contour/overlay PNGs in `docs/results/fassd/overlays/` (3 samples per structure):

| Structure | Sample | Patient | Filename | Area (%) | Saved Overlay |
|---|---|---|---|---|---|
| `artery` | 1 | `P080` | `P080_IMG1.png` | 0.20% | [overlays/artery_1_P080_IMG1.png](overlays/artery_1_P080_IMG1.png) |
| `artery` | 2 | `P0122` | `P0122_IMG1.png` | 0.04% | [overlays/artery_2_P0122_IMG1.png](overlays/artery_2_P0122_IMG1.png) |
| `artery` | 3 | `P0104` | `P0104_IMG3.png` | 0.21% | [overlays/artery_3_P0104_IMG3.png](overlays/artery_3_P0104_IMG3.png) |
| `liver` | 1 | `P095` | `P095_IMG3.png` | 3.96% | [overlays/liver_1_P095_IMG3.png](overlays/liver_1_P095_IMG3.png) |
| `liver` | 2 | `P0157` | `P0157_IMG11.png` | 6.12% | [overlays/liver_2_P0157_IMG11.png](overlays/liver_2_P0157_IMG11.png) |
| `liver` | 3 | `P014` | `P014_IMG4.png` | 3.58% | [overlays/liver_3_P014_IMG4.png](overlays/liver_3_P014_IMG4.png) |
| `stomach` | 1 | `P0149` | `P0149_IMG7.png` | 2.17% | [overlays/stomach_1_P0149_IMG7.png](overlays/stomach_1_P0149_IMG7.png) |
| `stomach` | 2 | `P0128` | `P0128_IMG8.png` | 0.10% | [overlays/stomach_2_P0128_IMG8.png](overlays/stomach_2_P0128_IMG8.png) |
| `stomach` | 3 | `P099` | `P099_IMG8.png` | 0.72% | [overlays/stomach_3_P099_IMG8.png](overlays/stomach_3_P099_IMG8.png) |
| `vein` | 1 | `P011` | `P011_IMG5.png` | 0.81% | [overlays/vein_1_P011_IMG5.png](overlays/vein_1_P011_IMG5.png) |
| `vein` | 2 | `P086` | `P086_IMG5.png` | 0.46% | [overlays/vein_2_P086_IMG5.png](overlays/vein_2_P086_IMG5.png) |
| `vein` | 3 | `P095` | `P095_IMG6.png` | 0.47% | [overlays/vein_3_P095_IMG6.png](overlays/vein_3_P095_IMG6.png) |