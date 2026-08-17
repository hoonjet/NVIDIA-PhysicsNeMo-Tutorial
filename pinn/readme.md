# PINN (Physics-Informed Neural Networks) Tutorials

> Equation-based learning — trains without labeled data using PDE residual loss

---

## Overview

PINN injects physical laws (PDEs) directly into the neural network loss function, enabling learning of PDE solutions without labeled data. It uses PhysicsNeMo's `FullyConnected` model and PyTorch `autograd`.

---

## Tutorials

| # | Tutorial | Equation | Script |
|---|----------|----------|--------|
| 1 | [Burgers Equation](burgers/) | Burgers (shock wave) | `burgers.py` |
| 2 | [Lid-Driven Cavity 2D](lid_driven_cavity/) | Navier-Stokes | `pinn_ldc2d.py` |
| 3 | [Conjugate Heat Transfer 2D](conjugate_heat_transfer/) | Heat transfer + flow | `pinn_cht2d.py` |
| 4 | [Electrostatics](electrostatics/) | Poisson equation | `pinn_electrostatics.py` |
| 5 | [Plane Stress](plane_stress/) | Linear elasticity | `pinn_planestress.py` |
| 6 | [Inverse Problem](inverse_problem/) | Inverse problem (parameter estimation) | `inverse_pinn.py` |
| 7 | [Adaptive Sampling (RAR)](adaptive_sampling/) | Adaptive collocation (2D Poisson) | `adaptive_sampling.py` |
| 8 | [Reaction-Diffusion](reaction_diffusion/) | Gray-Scott (multi-variable coupled PDE) | `reaction_diffusion.py` |

---

## Recommended Learning Order

1. **Burgers Equation** — PINN basics (shock wave, classic benchmark from Raissi et al. 2019)
2. **Lid-Driven Cavity 2D** — Basic flow (Navier-Stokes)
3. **Conjugate Heat Transfer 2D** — Multi-physics (flow + heat transfer)
4. **Electrostatics** — Simple PDE (Poisson, PINN beginner)
5. **Plane Stress** — Structural mechanics (linear elasticity)
6. **Inverse Problem** — Inverse problem (forward → inverse)
7. **Adaptive Sampling (RAR)** — Advanced technique (adaptive collocation, transferable skill)
8. **Reaction-Diffusion** — Multi-variable coupled PDE (Gray-Scott, Turing patterns)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Training data** | Not required (learns from equations only) |
| **Core mechanism** | Autograd (automatic differentiation, 2nd-order derivatives) |
| **Loss function** | PDE residual + boundary conditions + initial conditions |
| **Resolution independence** | ✓ (continuous function) |
| **Irregular mesh** | ✓ (point-based) |
| **Inverse problems** | ✓ (natural extension) |
| **Memory** | Low |
| **Training speed** | Slow (point-wise computation) |
