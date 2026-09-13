# Neural Operator Tutorials

> Data-driven learning — surrogate models that approximate PDE solutions via supervised learning

---

## Overview

Neural Operators are surrogate models that learn PDE solutions from data, enabling fast inference. They require labeled data (input-output pairs) but achieve millisecond-level inference after training.

---

## Tutorials

| # | Model | Key Mechanism | Script |
|---|------|---------------|--------|
| 1 | [FNO - Darcy Flow](fno/darcy_flow/) | Fourier transform (frequency domain) | `fno_darcy.py` |
| 2 | [FNO - Navier-Stokes](fno/navier_stokes/) | Fourier transform (time-dependent flow) | `fno_navier_stokes.py` |
| 3 | [FNO - Heat Conduction](fno/heat_conduction/) | Fourier transform (heat conduction) | `fno_heatconduction.py` |
| 4 | [FNO - Allen-Cahn](fno/allen_cahn/) | Fourier transform (phase separation) | `allen_cahn.py` |
| 5 | [FNO - Wave Equation](fno/wave/) | Fourier transform (2nd-order time) | `wave.py` |
| 6 | [AFNO - Darcy Flow](afno/) | Adaptive FNO (frequency + attention) | `afno_darcy.py` |
| 7 | [Transolver - Darcy Flow](transolver/) | Physics Attention (slicing) | `transolver_darcy.py` |
| 8 | [U-Net - Darcy Flow](unet/) | 3D CNN Encoder-Decoder (skip connections) | `unet_darcy.py` |
| 9 | [SRRN - Super Resolution](srrn/) | Sub-Pixel Conv (resolution upscaling) | `srrn_superres.py` |
| 10 | [DeepONet - Burgers](deeponet/) | Branch-Trunk (operator learning) | `deeponet_burgers.py` |
| 11 | [PINO - Darcy Flow](pino/) | FNO + PDE residual (hybrid) | `pino_darcy.py` |
| 12 | [FNO - Zero-Shot Resolution](fno/zero_shot/) | Train 32×32, test 64/128 | `zero_shot.py` |
| 13 | [PI-DeepONet](pi_deeponet/) | DeepONet + PDE residual (less data, physics-constrained) | `pi_deeponet.py` |

---

## Recommended Learning Order

1. **FNO - Darcy Flow** — Neural Operator introduction (fastest and most accurate)
2. **U-Net - Darcy Flow** — CNN-based (intuitive structure)
3. **Transolver - Darcy Flow** — Transformer-based (Physics Attention)
4. **AFNO - Darcy Flow** — FNO + Attention (composite structure)
5. **SRRN - Super Resolution** — Resolution enhancement (different paradigm)
6. **FNO - Heat Conduction** — Apply to different PDE
7. **DeepONet - Burgers** — Operator learning (function-to-function mapping)
8. **FNO - Allen-Cahn** — Time evolution (auto-regressive rollout)
9. **FNO - Wave Equation** — 2nd-order time (2-channel input)
10. **PINO - Darcy Flow** — Physics-informed neural operator (hybrid)
11. **FNO - Zero-Shot** — Resolution generalization (train low, test high)

---

## Key Features by Model

| Feature | FNO | AFNO | Transolver | U-Net | SRRN | DeepONet | PINO |
|---------|-----|------|------------|-------|------|----------|------|
| **Compute domain** | Frequency | Freq+Attn | Token | Spatial | Spatial | Sensor+Coord | Freq+PDE |
| **Resolution independence** | ✓ | ✓ | Partial | ✓ | ✗ | ✓ | ✓ |
| **Irregular mesh** | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| **Skip connection** | Bypass | Bypass | Residual | U-Net | Residual | None | Bypass |
| **Training speed** | Fast | Medium | Medium | Fast | Slow | Medium | Fast |
| **Memory** | Medium | Medium | High | Low | Medium | Low | Medium |
| **Parameterized** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |

---

## Performance Comparison

Darcy Flow problem (32×32 grid, 200 epochs):

| Model | Parameters | Training Time | Final Loss |
|-------|------------|---------------|------------|
| FNO | 2.4M | 21s | 8.68×10⁻⁴ |
| U-Net | 417K | 127s | 7.14×10⁻³ |
| Transolver | 354K | 126s | 1.47×10⁻² |

> For detailed comparison, see [comparisons/](../comparisons/).
