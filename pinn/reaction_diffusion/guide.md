# PINN Tutorial: Reaction-Diffusion (Gray-Scott Model) — English

## Overview
This tutorial trains a PINN to solve the **Gray-Scott model** — a system of **two coupled nonlinear PDEs** that produce emergent **Turing patterns** (spots, stripes, self-replicating structures). This is the **only tutorial in the repo that solves a multi-variable coupled PDE system**, where two chemical concentrations (u, v) interact through nonlinear reaction terms.

## Equations (Coupled System)
```
u_t = D_u * (u_xx + u_yy) - u*v² + F*(1 - u)
v_t = D_v * (v_xx + v_yy) + u*v² - (F + k)*v
```
- Domain: [0, 1] × [0, 1], t ∈ [0, 2]
- D_u = 2×10⁻⁵ (diffusion of u)
- D_v = 1×10⁻⁵ (diffusion of v)
- F = 0.025 (feed rate)
- k = 0.060 (kill rate)
- IC: u=1, v=0 everywhere except center square (u=0.5, v=0.25)
- BC: u=1, v=0 on domain boundary

## What Makes This Tutorial Unique?

| Feature | Other PINN Tutorials | **This Tutorial** |
|---------|---------------------|-------------------|
| Number of variables | 1 (single) | **2 (coupled u, v)** |
| Variable coupling | None or pressure-only | **Nonlinear reaction (u*v²)** |
| Nonlinearity type | Advective (u*u_x) | **Reaction (u*v²)** |
| Pattern formation | No | **Yes — Turing patterns** |
| Network output | 1 value | **2 values (u, v)** |
| Physics phenomenon | Shocks, fields | **Emergent self-organization** |

## Architecture
- **Network**: 3 → 64 → 64 → 64 → 64 → **2** (Fully Connected, Tanh)
- **Input**: (x, y, t)
- **Output**: (u, v) — both variables from a single network
- **Parameters**: 13,250
- **Optimizer**: Adam (lr=1e-3, StepLR gamma=0.5 every 3000 epochs)
- **Epochs**: 8,000

## Coupled PDE Residuals
Each equation's residual depends on **both** variables:
- `res_u = u_t - D_u·∇²u + u·v² - F·(1-u)` — contains v (via u·v²)
- `res_v = v_t - D_v·∇²v - u·v² + (F+k)·v` — contains u (via u·v²)

The autograd computation must differentiate **both** u and v with respect to (x, y, t), requiring careful graph management.

## Training Data
| Data Type | Count | Description |
|-----------|-------|-------------|
| IC points | 2,000 | Random points at t=0 with center perturbation |
| BC points | 1,000 | Boundary points (u=1, v=0) at random times |
| Collocation | 15,000 | Interior points for PDE residual (both equations) |

## Reference Solution
A finite difference solution (64×64 grid, 200 steps, explicit Euler, periodic BC) serves as ground truth for quantitative comparison.

## Key Results

| Metric | Value |
|--------|-------|
| Training time | ~180s |
| Final loss | ~1×10⁻⁴ |
| L2 error (u) at t=2.0 | ~0.05–0.15 |
| L2 error (v) at t=2.0 | ~0.10–0.25 |

> PINN captures the qualitative pattern formation. Exact accuracy depends on training duration and collocation density. Reaction-diffusion systems are challenging for PINNs due to the multi-scale nature (slow diffusion, fast reaction).

## Key Observations
1. **Multi-output network**: A single network predicts both u and v — the shared representation learns the coupling
2. **Nonlinear coupling**: The u·v² term appears in both equations with opposite signs, modeling predator-prey-like dynamics
3. **Turing instability**: Diffusion (destabilizing) + reaction (stabilizing) → emergent spatial patterns from uniform initial conditions
4. **Pattern formation**: Starting from a localized perturbation, the system self-organizes into spots/stripes — a hallmark of biological morphogenesis
5. **Challenging for PINN**: The very small diffusion coefficients (10⁻⁵) create sharp spatial gradients that are hard to resolve with collocation methods

## How to Run
```cmd
E:\physicsnemo_env\Scripts\python.exe reaction_diffusion.py
```

## Output Files
- `results/reaction_diffusion_loss.png` — Loss breakdown (Total, IC, BC, PDE)
- `results/reaction_diffusion_patterns.png` — Pattern evolution: u and v at t=0, 0.5, 1.0, 1.5, 2.0
- `results/reaction_diffusion_comparison.png` — PINN vs FD reference at t=2.0 (u, v, error)
- `results/reaction_diffusion_cross.png` — Cross-section comparison at y=0.5, t=2.0

## Applications of Reaction-Diffusion Systems

| Field | Application |
|------|-------------|
| **Biology** | Animal coat patterns (zebra stripes, leopard spots) |
| **Chemistry** | Belousov-Zhabotinsky reaction, chemical oscillations |
| **Ecology** | Predator-prey spatial dynamics, population spread |
| **Combustion** | Flame front propagation, autoignition |
| **Materials** | Crystal growth, phase separation |
| **Developmental biology** | Morphogen gradients, tissue patterning |

## Gray-Scott Parameter Space
Different (F, k) values produce different patterns:

| F | k | Pattern Type |
|---|---|-------------|
| 0.025 | 0.060 | Spots (used in this tutorial) |
| 0.014 | 0.045 | Stripes |
| 0.022 | 0.051 | Self-replicating spots |
| 0.030 | 0.062 | Mazes |
| 0.037 | 0.060 | Holes |

> Try changing F_FEED and K_KILL in the script to explore different pattern regimes!

## References
- Pearson, "Complex patterns in a simple system," Science, 1993.
- Gray & Scott, "Autocatalytic reactions in the isothermal, continuous stirred tank reactor," Chemical Engineering Science, 1984.
- Turing, "The chemical basis of morphogenesis," Philosophical Transactions of the Royal Society, 1952.
