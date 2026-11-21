# Heterogeneous Blend Study (Folds 0, 1, 2 Validation, N=613, 484 Patients)
1. **nnU-Net Alone (measfix)**: Pooled MAE = 1.9495 mm, Bias = -0.1588 mm.
2. **EfficientNet-B4 Alone**: Pooled MAE = 1.9914 mm (raw), 2.0940 mm (measfix).
3. **ResNet-34 Row D Alone**: Pooled MAE = 2.2295 mm (raw), 2.2071 mm (measfix).
4. **Blend nnU-Net (measfix) + EffNet (raw)**: Pooled MAE = 1.8540 mm, Bias = -0.0016 mm.
5. **Bootstrap vs nnU-Net Alone (EffNet raw)**: Delta = -0.0955 mm, 95% CI [-0.1869, 0.0178], p = 0.0888 (NOT significant).
6. **Blend nnU-Net (measfix) + EffNet (measfix)**: Pooled MAE = 1.8966 mm, Delta = -0.0529 mm, 95% CI [-0.1453, 0.0604], p = 0.3232.
7. **Blend nnU-Net (measfix) + Row D (raw)**: Pooled MAE = 1.9591 mm (Delta = +0.0096 mm, p = 0.9116).
8. **Ellipse Parameter Blend (iv)**: STOPPED — per-image ellipse parameters (center, semi-axes, angle) are not saved in OOF prediction files.
9. **Conclusion**: Heterogeneous blending with existing U-Nets does NOT statistically improve over nnU-Net alone (p > 0.05). nnU-Net alone is superior.
