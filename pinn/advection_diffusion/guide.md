# Advection-Diffusion PINN (Pollutant Transport)

> **Category**: `pinn/` — Linear transport, environmental engineering
> **Problem**: 2D Gaussian plume dispersion (wind + diffusion)

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing Burgers | **THIS (Advection-Diffusion)** |
|--------|-----------------|-------------------------------|
| **Nonlinearity** | Nonlinear (u·∇u) | **Linear (v·∇c, fixed v)** |
| **Regime control** | Fixed (viscosity only) | **Peclet number (Pe)** |
| **Feature** | Shock waves | **Gaussian plume** |
| **Application** | Fluid dynamics | **Environmental/pollutant** |

---

## 2. Physics

```
∂c/∂t + v·∇c = D·∇²c

v = (vx, vy) = (1.0, 0.0)  (wind)
D = 0.01  (diffusion)
Pe = v·L/D = 100  (advection-dominated)
```

### Peclet Number Regimes
- **Pe >> 1**: Advection dominates → plume moves with wind, narrow
- **Pe << 1**: Diffusion dominates → plume spreads isotropically, round
- **Pe ~ 1**: Mixed regime

### Analytical Solution (Gaussian Plume)
```
c(x,y,t) = (1/(4πDt)) · exp(-((x-vt-x₀)² + (y-y₀)²) / (4Dt))
```

---

## 3. How to Run

```cmd
cd E:\physicsnemo-tutorials\pinn\advection_diffusion
python advection_diffusion.py
```

---

## 4. Results

- `advdiff_snapshots.png` — Concentration field at 4 times (analytical vs PINN vs error)
- `advdiff_1d.png` — 1D cross-section showing plume evolution
- `advdiff_loss.png` — Training loss
- `advdiff_concept.png` — Physics explanation + Peclet regimes
