# Neural Operator Tutorials

> Data-driven learning — surrogate models that approximate PDE solutions via supervised learning

---

## Overview

Neural Operators are surrogate models that learn PDE solutions from data, enabling fast inference. They require labeled data (input-output pairs) but achieve millisecond-level inference after training.

---

## Tutorials

| # | Model | Core Mechanism | Script |
|---|-------|---------------|--------|
| 1 | [FNO - Darcy Flow](fno/darcy_flow/) | Fourier transform (frequency domain) | `fno_darcy.py` |
| 2 | [FNO - Navier-Stokes](fno/navier_stokes/) | Fourier transform (time-dependent flow) | `fno_navier_stokes.py` |
| 3 | [FNO - Heat Conduction](fno/heat_conduction/) | Fourier transform (heat conduction) | `fno_heatconduction.py` |
| 4 | [AFNO - Darcy Flow](afno/) | Adaptive FNO (frequency + Attention) | `afno_darcy.py` |
| 5 | [Transolver - Darcy Flow](transolver/) | Physics Attention (slicing) | `transolver_darcy.py` |
| 6 | [U-Net - Darcy Flow](unet/) | 3D CNN Encoder-Decoder (skip connections) | `unet_darcy.py` |
| 7 | [SRRN - Super Resolution](srrn/) | Sub-Pixel Conv (resolution upscaling) | `srrn_superres.py` |

---

## Recommended Learning Order

1. **FNO - Darcy Flow** — Neural Operator introduction (fastest and most accurate)
2. **U-Net - Darcy Flow** — CNN-based (intuitive structure)
3. **Transolver - Darcy Flow** — Transformer-based (Physics Attention)
4. **AFNO - Darcy Flow** — FNO + Attention (composite structure)
5. **SRRN - Super Resolution** — Resolution enhancement (different paradigm)
6. **FNO - Heat Conduction** — Different PDE application

---

## Key Features by Model

| Feature | FNO | AFNO | Transolver | U-Net | SRRN |
|---------|-----|------|------------|-------|------|
| **Compute domain** | Frequency | Freq+Attention | Token | Spatial | Spatial |
| **Resolution independence** | ✓ | ✓ | Partial | ✓ | ✗ |
| **Irregular mesh** | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Skip connections** | Bypass | Bypass | Residual | U-Net | Residual |
| **Training speed** | Fast | Medium | Medium | Fast | Slow |
| **Memory** | Medium | Medium | High | Low | Medium |

---

## Performance Comparison

Darcy Flow problem (32×32 grid, 200 epochs):

| Model | Parameters | Training Time | Final Loss |
|-------|-----------|---------------|------------|
| FNO | 2.4M | 21s | 8.68×10⁻⁴ |
| U-Net | 417K | 127s | 7.14×10⁻³ |
| Transolver | 354K | 126s | 1.47×10⁻² |

> See [comparisons/](../comparisons/) for detailed comparison.
