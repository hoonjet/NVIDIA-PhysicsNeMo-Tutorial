# FNO Tutorial: Navier-Stokes Equation (Vorticity Form) — English

## Overview
This tutorial trains a Fourier Neural Operator (FNO) to predict the time evolution of 2D vorticity fields governed by the Navier-Stokes equations. Unlike the steady-state Darcy problem, this is a **time-dependent** PDE where the model learns the operator w(t) → w(t+dt).

## Equation
```
w_t + u * w_x + v * w_y = nu * (w_xx + w_yy)
u_x + v_y = 0  (incompressibility)
```
- Grid: 32×32 (periodic BCs)
- Time steps: 10 (dt=0.1)
- Viscosity: nu=0.01, decay=1e-3

## Data Generation
Synthetic vorticity data is generated using a **pseudo-spectral method** with:
- Random smoothed initial conditions
- Explicit Euler time stepping in Fourier space
- 2/3 dealiasing rule
- 30 samples × 10 time steps = 270 training pairs

## Model Architecture
- **FNO2d**: 4 Fourier layers, modes=8×8, width=20
- **Parameters**: 209,337
- **Optimizer**: Adam (lr=1e-3, StepLR)
- **Epochs**: 300, Batch size: 16

## Key Results
| Metric | Value |
|--------|-------|
| Training time | 70.6s |
| Final train loss | 1.42×10⁻⁵ |
| Final test loss | 2.32×10⁻⁵ |
| 10-step rollout L2 error | 1.24% |

## Key Observations
1. FNO learns the time evolution operator w(t) → w(t+dt)
2. Autoregressive rollout maintains stability over 10 steps
3. Unlike Darcy (steady), this is a time-dependent problem
4. Vorticity field shows turbulent decay over time

## How to Run
```cmd
E:\physicsnemo_env\Scripts\python.exe fno_navier_stokes.py
```

## Output Files
- `results/fno_ns_loss.png` — Training & test loss curves
- `results/fno_ns_result.png` — Vorticity evolution: Ground Truth vs FNO
- `results/fno_ns_error.png` — Absolute error maps
