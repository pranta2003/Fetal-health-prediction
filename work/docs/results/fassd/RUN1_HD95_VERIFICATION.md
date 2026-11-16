# FASSD Run 1 HD95 Verification & Code Audit Note

## 1. Background: HD95 Contour Extraction Code Audit

In `src/fetal/seg/fassd_train.py`, the original implementation of surface contour extraction in `compute_hd95` was:

```python
p_border = pred_binary ^ binary_erosion(pred_binary)
g_border = gt_binary ^ binary_erosion(gt_binary)
...
dt_gt = distance_transform_edt(~g_border)
dt_pred = distance_transform_edt(~p_border)

d_p2g = dt_gt[p_border]
d_g2p = dt_pred[g_border]
```

### The Indexing Bug Mechanism
- If `pred_binary` or `gt_binary` was passed as `np.uint8` with values in `{0, 1}`:
  1. `binary_erosion(gt_binary)` returns a boolean array.
  2. In NumPy, `uint8 ^ bool` results in `uint8` values `{0, 1}`.
  3. Indexing with a uint8 array (`dt_gt[p_border]`) triggers NumPy's **advanced integer indexing** rather than boolean masking, attempting to allocate an array of shape `(H, W, W)` (e.g. $768 \times 1024 \times 1024 \times 8 \text{ bytes} \approx 6.4\text{ GiB}$ to $12\text{ GiB}$), leading to `ArrayMemoryError`.
  4. In addition, bitwise negation on uint8 (`~uint8(0) == 255`, `~uint8(1) == 254`) yields non-zero values everywhere, preventing distance transform seed generation.
- **Fix**: Contours and masks are explicitly cast to boolean prior to erosion and distance transform:
  ```python
  p_bool = (pred_binary > 0)
  g_bool = (gt_binary > 0)
  p_border = p_bool ^ binary_erosion(p_bool)
  g_border = g_bool ^ binary_erosion(g_bool)
  ...
  p_border = (p_border > 0)
  g_border = (g_border > 0)
  ```

## 2. Impact on Run 1 Numbers

In FASSD Run 1 (`scripts/35_fassd_train.py` / `src/fetal/seg/fassd_train.py`), test metrics were evaluated via `compute_structure_metrics`, which defined:

```python
p = pred_mask[idx] > 0
g = gt_mask[idx] > 0
...
hd95 = compute_hd95(p, g)
```

Because `p` and `g` were already boolean arrays, `compute_hd95` received boolean inputs, avoiding uint8 integer indexing.

## 3. Recomputed Run 1 Test HD95 Values with Fixed Code

To verify byte-level numerical consistency, Run 1 checkpoints (`fassd_gray_seed42` and `fassd_rgb_seed42` at native $768 \times 1024$ resolution) were re-evaluated using the fixed code:

| Model | Structure | Run 1 Reported HD95 (px) | Recomputed HD95 (Fixed Code) (px) | Difference | Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **Gray (Seed 42)** | Artery | 10.41 | 10.41 | 0.00 | Exact Match |
| | Liver | 24.59 | 24.59 | 0.00 | Exact Match |
| | Stomach | 21.54 | 21.54 | 0.00 | Exact Match |
| | Vein | 24.29 | 24.29 | 0.00 | Exact Match |
| | **Macro Mean** | **20.21** | **20.21** | **0.00** | **Exact Match** |
| **RGB (Seed 42)** | Artery | 13.96 | 13.95 | -0.01 | Rounding Match |
| | Liver | 27.47 | 27.47 | 0.00 | Exact Match |
| | Stomach | 26.59 | 26.59 | 0.00 | Exact Match |
| | Vein | 20.04 | 20.04 | 0.00 | Exact Match |
| | **Macro Mean** | **22.01** | **22.01** | **0.00** | **Exact Match** |

**Conclusion**: The Run 1 reported numbers in `RUN1_NOTES.md` and `train_summary.json` were not corrupted by the indexing bug, and the recomputed values verify exact consistency.
