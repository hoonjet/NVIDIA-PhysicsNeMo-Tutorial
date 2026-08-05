# DeepONet Tutorial: Burgers Equation

## Overview
This tutorial trains a **Deep Operator Network (DeepONet)** to learn the solution operator of the 1D Viscous Burgers Equation. Unlike PINN (which solves a single PDE instance using equation residuals) or FNO (which maps a fixed-grid field to another fixed-grid field), DeepONet learns a **mapping between function spaces**: given any initial condition function u₀(x), predict the solution u(x, t) at any continuous query point.

## Equation
```
u_t + u * u_x = (nu / pi) * u_xx
```
- Domain: x ∈ [-1, 1], t ∈ [0, 1]
- Viscosity: nu/pi = 0.01/π ≈ 0.00318
- Boundary conditions: periodic
- Initial conditions: random smooth functions (sum of sinusoids)

## What Makes DeepONet Different?

| Feature | PINN | FNO | **DeepONet** |
|---------|------|-----|-------------|
| Learning paradigm | Equation-based | Data-driven (fixed grid) | **Data-driven (operator)** |
| Input | Coordinates (x, t) | Full field on grid | **Function at sensors + query point** |
| Output | Single value u(x, t) | Full field on grid | **Single value u(x, t)** |
| Generalization | One PDE instance | Fixed resolution | **Any IC from same family** |
| Query points | Continuous | Discrete grid | **Continuous** |
| Training data | Not needed | Fixed input-output pairs | **Function evaluations at sensors** |

## Architecture

### Branch-Net (encodes input function)
- Input: u₀(x) evaluated at m=100 sensor points
- Layers: 100 → 64 → 64 → 64 (Tanh activation)
- Output: p=64 dimensional latent vector

### Trunk-Net (encodes query coordinates)
- Input: (x, t) query point
- Layers: 2 → 64 → 64 → 64 (Tanh activation)
- Output: p=64 dimensional latent vector

### Output
- Dot product of branch and trunk outputs + bias
- `u(x,t) = Σᵢ branchᵢ(u₀) × trunkᵢ(x,t) + bias`

## Data Generation
- **80 training ICs** + **10 test ICs** (each a random sum of sinusoids)
- Solutions generated via finite difference with CFL-stable sub-stepping
- 128-point spatial grid, 21 time steps (t=0 to t=1)
- 200 random query points per IC → 16,000 training pairs

## Model Configuration
- **Parameters**: 17,985 (Branch: 12,480 / Trunk: 4,480 / Bias: 1)
- **Optimizer**: Adam (lr=1e-3, StepLR gamma=0.5 every 500 epochs)
- **Epochs**: 2,000
- **Batch size**: 512

## Key Results
| Metric | Value |
|--------|-------|
| Training time | ~120s |
| Final train loss | ~5×10⁻³ |
| Final test loss | ~5×10⁻³ |
| Relative L2 error (full trajectory) | ~0.05–0.10 |

## Key Observations
1. **Operator learning**: DeepONet learns the mapping u₀(x) → u(x,t), not a single solution
2. **Generalization**: One trained model predicts solutions for **unseen** initial conditions
3. **Continuous output**: Unlike FNO, DeepONet can predict at **any** (x, t) point, not just grid nodes
4. **Branch-Trunk decomposition**: Branch encodes the input function; Trunk encodes the query location; their dot product gives the prediction
5. **Theoretical basis**: Based on the Universal Approximation Theorem for operators (Chen & Chen 1995; Lu et al. 2021)

## How to Run
```cmd
E:\physicsnemo_env\Scripts\python.exe deeponet_burgers.py
```

## Output Files
- `results/deeponet_burgers_loss.png` — Training & test loss curves
- `results/deeponet_burgers_result.png` — Solution snapshots: Ground Truth vs DeepONet + error
- `results/deeponet_burgers_surface.png` — 3D surface comparison (truth, prediction, error)
- `results/deeponet_burgers_generalization.png` — Predictions across 5 different unseen ICs

## When to Use DeepONet (vs PINN / FNO)

| Scenario | Recommended Model |
|----------|------------------|
| Single PDE, no data, need physics | PINN |
| Fast inference on fixed grid, have data | FNO |
| **Multiple ICs / parametric PDE family** | **DeepONet** |
| **Irregular sensor placement** | **DeepONet** |
| **Continuous query at arbitrary points** | **DeepONet** |
| Need physics constraints + data | DeepONet + PINN (Physics-informed DeepONet) |

## References
- Lu Lu et al., "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators," Nature Machine Intelligence, 2021.
- Chen & Chen, "Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems," IEEE TNN, 1995.
