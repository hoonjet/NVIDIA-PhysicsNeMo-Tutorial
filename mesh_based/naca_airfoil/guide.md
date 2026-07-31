# NACA Airfoil Potential Flow Tutorial Guide

> **Tutorial file**: `tutorial_naca_airfoil.py`  
> **Result image**: `tutorial_results/naca_airfoil_3way_comparison.png`  
> **Date**: 2026-07-13

---

## 1. Overview

This tutorial solves **2D incompressible potential flow around a NACA 0012 airfoil** using three methods and compares them:

1. **Analytical Ground Truth**: Exact analytical solution via Joukowski transformation
2. **PINN**: Continuity equation + irrotational condition as loss functions, physics-based learning
3. **FNO**: Data-driven learning of SDF → velocity field mapping

### Fundamental Differences from Existing Tutorials

| Feature | Existing (Darcy/LDC) | This Tutorial (NACA) |
|:---|:---|:---|
| **Geometry** | Simple rectangular grid | NACA 0012 airfoil (curved surface) |
| **Governing equation** | Darcy/Navier-Stokes | Potential flow (Laplace equation) |
| **Geometry awareness** | Coordinates only | SDF (Signed Distance Function) |
| **Ground Truth** | Numerical solution (FDM) | Analytical solution (Joukowski transform) |
| **Comparison structure** | 2-way (PINN vs FNO) | 3-way (PINN vs FNO vs Analytical) |
| **Physical output** | Pressure/velocity | Velocity field, pressure coefficient Cp, streamlines |

---

## 2. Physics Background

### Potential Flow

Potential flow is incompressible + irrotational flow, where a velocity potential φ exists:

- **Velocity**: u = ∂φ/∂x, v = ∂φ/∂y
- **Governing equation**: ∇²φ = 0 (Laplace equation)
- **Equivalent conditions**: Continuity equation (∂u/∂x + ∂v/∂y = 0) + irrotational condition (∂u/∂y - ∂v/∂x = 0)

### Joukowski Transformation

Transforms potential flow around a cylinder (which has an analytical solution) to an airfoil shape:

1. **Flow around cylinder**: Complex potential W(ζ) = U(ζ + R²/ζ) + iΓ/(2π) ln(ζ)
2. **Joukowski transformation**: z = ζ + c²/ζ (cylinder → airfoil)
3. **Kutta condition**: Circulation Γ = 4πRU sin(α) ensures smooth flow at trailing edge
4. **Result**: Exact velocity and pressure fields around the airfoil computed analytically

### SDF (Signed Distance Function)

- A scalar field representing the distance to the airfoil surface
- Outside: positive, inside: negative, surface: zero
- Used as network input to recognize geometry

---

## 3. Tutorial Configuration

### Data
- **Problem**: NACA 0012 airfoil, potential flow
- **Grid**: 64×64, domain [-1.5, 2.5] × [-1.5, 1.5]
- **Flow conditions**: U∞ = 1.0, AoA = 5°
- **Ground Truth**: Joukowski transformation (analytical)

### FNO Settings
```python
FNO(
    in_channels=5,       # SDF + x + y + sin(α) + cos(α)
    out_channels=2,       # u, v
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
)
# Training: 50 samples (AoA -10° ~ +10°), 200 epochs
```

### PINN Settings
```python
PINN_Airfoil(
    in_dim=3,    # x, y, SDF
    hidden=128, layers=6,
    out_dim=2,   # directly predict u, v
)
# Loss: continuity + irrotational + far-field BC + surface no-penetration + supervised
# Training: 3000 epochs, LR scheduler (step=1000, γ=0.5)
```

---

## 4. Experimental Results

### Performance Summary

| Metric | FNO | PINN | Analytical GT |
|:---|:---:|:---:|:---:|
| **Parameters** | 2,365,314 | 99,842 | — |
| **Training time** | 20s | 93s | Instant |
| **Training data** | 50 samples (Joukowski GT) | 988 supervised points + physics | — |
| **L2 error** | **8.99%** | 107.26% | 0% (reference) |
| **Physics constraint** | Implicit (learned from data) | Explicit (PDE loss) | Exact |
| **Grid required** | Yes | No | — |

### PINN Analysis

PINN loss breakdown by component:

| Loss Term | Initial | Final | Meaning |
|:---|:---:|:---:|:---|
| **Continuity equation** | 9.44×10⁻⁶ | 2.06×10⁻⁴ | ∂u/∂x + ∂v/∂y = 0 |
| **Far-field BC** | 1.08×10⁰ | 1.65×10⁻⁶ | u=Ucos(α), v=Usin(α) |
| **Supervised** | 1.12×10⁰ | 2.59×10⁻³ | Matching Joukowski GT points |
| **Total loss** | 1.30×10¹ | 5.95×10⁻³ | — |

**Why PINN struggles**:

1. **Complex geometry**: The curved surface of NACA 0012, especially the large velocity gradient at the leading edge, is difficult for PINN to learn
2. **Localized characteristics**: Fits well at supervised points (988) but fails to generalize between them
3. **Continuity convergence**: Continuity loss plateaus at 2×10⁻⁴ → PDE not fully satisfied
4. **SDF input limitation**: SDF alone is insufficient to learn the complex velocity field near the surface

### FNO Analysis

FNO learns Joukowski solutions for 50 different angles of attack, achieving 8.99% L2 error:
- **Frequency domain learning**: FNO's FFT-based computation effectively captures the complex velocity field around the airfoil
- **Generalization**: Training on -10° to +10° range works well at 5°
- **Fast training**: Converges in 20 seconds

---

## 5. Lessons from 3-way Comparison

### Methodology Comparison

| Aspect | FNO | PINN | Analytical |
|:---|:---|:---|:---|
| **Accuracy** | High (8.99%) | Low (107%) | Perfect |
| **Speed** | Fast (20s) | Slow (93s) | Instant |
| **Data required** | Essential (50 samples) | Minimal (988 points) | None |
| **Physics laws** | Implicit learning | Explicit constraint | Exact |
| **Geometry generalization** | Possible via SDF input | Possible via SDF input | Recompute per geometry |
| **Scalability** | Retrain for new geometry | Retrain for new geometry | Re-derive for new geometry |

### Key Insights

1. **Data-driven vs Physics-based**: FNO is fast and accurate when data is available, but PINN attempts to learn from physics laws alone (this tutorial uses some supervised data)

2. **PINN limitations on complex geometry**: PINN works well on simple rectangles (LDC) but struggles with curved geometry (airfoil). This is an active research topic in PINN

3. **Value of analytical solution**: Joukowski transformation provides exact Ground Truth without numerical methods, useful for validating deep learning models

4. **Role of SDF**: Both models use SDF as input to recognize airfoil geometry. This is a standard technique for handling complex geometries

---

## 6. Resource Comparison (vs existing tutorials)

| Resource | FNO (Darcy) | FNO (NACA) | PINN (LDC) | PINN (NACA) |
|:---|:---:|:---:|:---:|:---:|
| **Parameters** | 2.4M | 2.4M | 17K | 100K |
| **Training time** | 21s | 20s | 61s | 93s |
| **L2 error** | 5.4% | 9.0% | N/A | 107% |
| **Geometry** | Rectangle | Airfoil | Rectangle | Airfoil |
| **Ground Truth** | Numerical | Analytical | Numerical | Analytical |

---

## 7. Key Parameter Tuning Guide

### FNO
| Parameter | Default | Effect |
|:---|:---|:---|
| `num_fno_modes` | 12 | Larger captures higher frequencies, more memory |
| `latent_channels` | 32 | Larger = more expressiveness |
| `num_fno_layers` | 4 | Deeper = learns more complex patterns |
| Training data count | 50 | More = better generalization |

### PINN
| Parameter | Default | Effect |
|:---|:---|:---|
| `hidden` | 128 | Larger = more expressiveness |
| `layers` | 6 | Deeper = approximates more complex functions |
| `N_INTERIOR` | 5000 | More = better PDE satisfaction |
| `N_SUP` | 1000 | Supervised points, more = more accurate |
| Loss weights | 1/0.1/10/5/2 | BC > PDE > supervised priority |

---

## 8. Conclusion

This tutorial provides a **3-way comparison on non-standard geometry (airfoil)**:

1. **FNO** achieves good performance with 8.99% error using data-driven approach
2. **PINN** struggles with complex geometry (107% error) — a known limitation
3. **Joukowski transformation** provides exact Ground Truth without numerical methods

**Lesson**: PINN is powerful for simple geometries, but for complex geometries (airfoils, turbines, etc.), data-driven methods (FNO) may be more practical. However, PINN research is actively progressing and can be improved with better architectures and training strategies.
