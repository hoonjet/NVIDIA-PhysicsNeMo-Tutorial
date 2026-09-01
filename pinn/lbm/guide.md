# Lattice Boltzmann Method (LBM) PINN

> **Category**: `pinn/` — Mesoscopic / kinetic theory approach
> **Paradigm**: PINN with Boltzmann BGK equation (D2Q9 lattice)
> **Problem**: 2D Lid-Driven Cavity (same geometry as existing LDC PINN, different PDE)

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing LDC PINN | THIS (LBM PINN) |
|--------|------------------|------------------|
| **PDE** | Navier-Stokes (macroscopic) | Boltzmann BGK (mesoscopic) |
| **Variables** | u, v, p (3 fields) | f₀..f₈ (9 distribution functions) |
| **Scale** | Continuum (macro) | Kinetic theory (meso) |
| **Equations** | 1 vector PDE | 9 coupled advection-relaxation equations |
| **Physics** | Conservation laws | Particle distribution + collision |

### Key Difference: Mesoscopic vs Macroscopic
- **Navier-Stokes**: Solves for macroscopic velocity/pressure directly
- **Boltzmann BGK**: Solves for particle distribution functions; macroscopic fields are *derived* via moments
- LBM recovers Navier-Stokes in the continuum limit, but the mathematical structure is completely different

---

## 2. D2Q9 Lattice

The D2Q9 model uses 9 discrete velocity directions:

```
  f₆   f₂   f₅          (-1,1)  (0,1)  (1,1)
    \   |   /              \    |    /
     \  |  /                \   |   /
  f₃--f₀--f₁    velocities  (-1,0)--(0,0)--(1,0)
     /  |  \                /   |   \
    /   |   \              /    |    \
  f₇   f₄   f₈          (-1,-1) (0,-1) (1,-1)
```

| Index i | c_i | Weight w_i |
|---------|-----|-----------|
| 0 | (0, 0) | 4/9 |
| 1 | (1, 0) | 1/9 |
| 2 | (0, 1) | 1/9 |
| 3 | (-1, 0) | 1/9 |
| 4 | (0, -1) | 1/9 |
| 5 | (1, 1) | 1/36 |
| 6 | (-1, 1) | 1/36 |
| 7 | (-1, -1) | 1/36 |
| 8 | (1, -1) | 1/36 |

---

## 3. Boltzmann BGK Equation

### Steady-state BGK equation (what PINN solves):
```
c_i · ∇f_i = -(1/τ)(f_i - f_i^eq)
```

Where:
- `f_i(x, y)` — distribution function for velocity direction i
- `c_i` — discrete velocity vector
- `τ = 3ν + 0.5` — relaxation time (related to viscosity)
- `f_i^eq` — equilibrium distribution

### Equilibrium distribution:
```
f_i^eq = w_i · ρ · (1 + 3(c_i·u) + 4.5(c_i·u)² - 1.5|u|²)
```

### Macroscopic recovery (moments):
```
ρ = Σ f_i                    (density)
u = (Σ c_i · f_i) / ρ        (velocity)
```

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| [1] D2Q9 Lattice | 9 velocities, weights, relaxation time |
| [2] Reference Solver | Standard LBM solver (bounce-back BC, streaming, collision) |
| [3] PINN Model | 2→64→64→64→64→9 (predicts 9 distribution functions) |
| [4] Collocation | 2000 interior + 800 boundary points |
| [5] Loss | PDE (BGK residual) + BC (no-slip + lid) + Data (LBM reference) |
| [6] Training | 5000 epochs, Adam, λ_PDE=1, λ_BC=10, λ_DATA=5 |
| [7] Visualization | Flow field, streamlines, loss, D2Q9 concept, distributions |
| [8] Summary | Relative L2 error vs LBM solver |

---

## 5. How to Run

```cmd
cd E:\physicsnemo-tutorials\pinn\lbm
python lbm.py
```

Results saved to `results/`:
- `lbm_flow_comparison.png` — Reference vs PINN (ux, uy, |u|)
- `lbm_streamlines.png` — Streamline comparison
- `lbm_loss.png` — Training loss history
- `lbm_concept.png` — D2Q9 lattice diagram + equation comparison
- `lbm_distributions.png` — All 9 distribution functions f_i

---

## 6. vs. Existing LDC PINN (Navier-Stokes)

| Feature | LDC PINN (NS) | LBM PINN (BGK) |
|---------|---------------|----------------|
| **PDE** | Navier-Stokes | Boltzmann BGK |
| **Unknowns** | u, v, p (3) | f₀..f₈ (9) |
| **Scale** | Macroscopic | Mesoscopic |
| **Physics** | Continuum | Kinetic theory |
| **BC** | Velocity BC | Bounce-back / velocity BC |
| **Output** | u, v, p directly | f_i → moments → u, v, ρ |
| **Advantage** | Simpler, fewer unknowns | Natural for complex BC, multiphase |

---

## 7. Why LBM Matters

1. **Complex boundaries**: LBM handles complex geometry naturally (bounce-back)
2. **Multiphase flow**: LBM naturally models multiphase/multicomponent flow
3. **Parallelizable**: Local collision + streaming is highly parallel
4. **Compressible**: LBM is inherently compressible (unlike incompressible NS PINN)
5. **Non-continuum**: LBM works in rarefied gas regime where NS breaks down

---

## 8. References

- Qian et al., "Lattice BGK Models for Navier-Stokes Equation" (1992)
- Chen & Doolen, "Lattice Boltzmann Method for Fluid Flows" (1998)
- Krüger et al., "The Lattice Boltzmann Method" (2017, Springer textbook)
