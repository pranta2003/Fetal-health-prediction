# AC Modern Encoders Serial Queue

**Campaign Goal**: Evaluate modern pretrained `smp.Unet` encoders serially on `folds_v1.csv` (642 images, 501 patients), keeping the exact `ac-baseline-s42` `v3_full` recipe (flip-TTA, max 80 epochs, patience 12, seed 42), varying only the encoder and learning rate.

## Encoder Compatibility & Parameter Counts (384×384, ImageNet Pretrained)

| Order | Kernel Slug | Encoder Name | LR | Params | 2-img 384px CPU Step | Compatibility | Status |
|---|---|---|---|---:|---:|---|---|
| 1 | `ac-convnext-lr1e4-s42` | `tu-convnext_tiny` | `1e-4` | 31,929,188 | 15.93s | PASS | **DONE** (4.8551 mm OOF MAE, NEGATIVE RESULT vs 4-seed +0.4523 mm) |
| 2 | `ac-convnextv2-s42` | `tu-convnextv2_tiny` | `1e-4` | 31,975,556 | 16.29s | PASS | QUEUED (Local only) |
| 3 | `ac-tf-efficientnetv2-s-s42` | `tu-tf_efficientnetv2_s` | `2e-4` | 22,093,908 | 8.40s | PASS | QUEUED (Local only) |
| 4 | `ac-repvit-s42` | `tu-repvit_m1_1` | `1e-4` | 10,908,564 | 4.44s | PASS | **DONE** (5.7041 mm OOF MAE, NEGATIVE RESULT vs 4-seed +1.3013 mm; pushed out of order) |
| 5 | `ac-fastvit-s42` | `tu-fastvit_sa12` | `1e-4` | 13,543,172 | 4.70s | PASS | QUEUED (Local only) |
| 6 | `ac-mambaout-s42` | `tu-mambaout_tiny` | `1e-4` | 26,573,156 | 6.99s | PASS | QUEUED (Local only) |
| 7 | *(skipped)* | `tu-coatnet_0_rw_224` | `1e-4` | — | — | **FAIL** (`RuntimeError: The size of tensor a (576) must match the size of tensor b (196) at non-singleton dimension 3` at 384×384) | SKIPPED |
| 8 | `ac-mit-b2-s42` | `mit_b2` | `1e-4` | 27,477,444 | 5.99s | PASS | QUEUED (Local only) |

> Baseline reference: `ac-baseline-s42` (`efficientnet-b4`, `lr=2e-4`, 20,226,124 params) achieves 4.7744 mm OOF MAE (4-seed avg: 4.4028 mm).
