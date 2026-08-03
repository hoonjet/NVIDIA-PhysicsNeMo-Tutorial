# PINN Tutorial: Burgers Equation (English)

## Overview
This tutorial solves the 1D Viscous Burgers Equation using a Physics-Informed Neural Network (PINN). The Burgers equation is the classic PINN benchmark from Raissi et al. (2019), featuring a shock wave that sharpens over time.

## Equation
```
u_t + u * u_x = (nu / pi) * u_xx
```
- Domain: x ∈ [-1, 1], t ∈ [0, 1]
- Viscosity: nu/pi = 0.01/π ≈ 0.00318
- BC: u(-1, t) = u(1, t) = 0
- IC: u(x, 0) = -sin(πx)

## Model Architecture
- **Network**: 2 → 64 → 64 → 64 → 64 → 1 (Fully Connected, Tanh)
- **Parameters**: 12,737
- **Optimizer**: Adam (lr=1e-3, StepLR)
- **Epochs**: 5,000

## Key Results
| Metric | Value |
|--------|-------|
| Training time | 80.4s |
| Final loss | 8.07×10⁻⁴ |
| IC loss | 2.98×10⁻⁴ |
| BC loss | 4.81×10⁻⁶ |
| PDE loss | 5.04×10⁻⁴ |

## Key Observations
1. The shock wave forms near x=0 and sharpens over time
2. PINN captures the shock but smooths it (viscosity effect)
3. No training data needed — only the PDE and boundary/initial conditions
4. Solution is continuous (can evaluate at any point in the domain)

## How to Run
```cmd
E:\physicsnemo_env\Scripts\python.exe burgers.py
```

## Output Files
- `results/burgers_result.png` — Solution snapshots at different times
- `results/burgers_loss.png` — 3D surface, contour, and loss curve
