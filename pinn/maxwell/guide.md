# Maxwell's Equations PINN (Electromagnetic Wave Propagation)

> **Category**: `pinn/` — Dynamic electromagnetics, vector field
> **Paradigm**: PINN with Maxwell's curl equations (TE mode)
> **Problem**: 2D EM wave propagation through a dielectric interface

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing Electrostatics | Existing Helmholtz | **THIS (Maxwell)** |
|--------|------------------------|--------------------|--------------------|
| **Time** | Static (∂/∂t=0) | Frequency-domain | **Time-domain** |
| **Field** | Scalar (φ) | Complex scalar | **Vector (Ez, Hx, Hy)** |
| **Equations** | Poisson | Helmholtz | **Maxwell curl (3 coupled)** |
| **Material** | Single | Single | **Dielectric interface** |
| **Physics** | Electrostatics | Acoustics | **EM wave + refraction** |

### Key Difference: Vector Field + Dynamic + Multi-Material
- **Electrostatics**: scalar potential, no time, single material
- **Maxwell**: 3-component vector field (E, H), time-varying, material interface with refraction

---

## 2. Physics: Maxwell's Equations (TE Mode)

### Problem Setup
```
  Vacuum (ε_r=1)        Dielectric (ε_r=4)
  ┌─────────────┐──────┬─────────────┐
  │             │      │             │
  │  Incident → │      │ → Refracted │
  │  ← Reflected│      │             │
  │             │ x=0.5│             │
  └─────────────┘──────┴─────────────┘
```

### Maxwell's Curl Equations (TE mode, 2D)
```
∂Ez/∂t = (1/ε)(∂Hy/∂x - ∂Hx/∂y)    (Faraday's law)
∂Hx/∂t = -(1/μ)(∂Ez/∂y)              (Ampere's law, x)
∂Hy/∂t = (1/μ)(∂Ez/∂x)               (Ampere's law, y)
```

Where:
- `ε = ε_r · ε_0` (permittivity, varies by material)
- `μ = μ_0` (permeability, constant)
- TE mode: E-field transverse (z-component), H-field in-plane (x, y)

### Boundary Conditions
- **x=0**: Ez = cos(ωt) (source plane wave)
- **Interface (x=0.5)**: Continuity of Ez, Hy (implicit in PDE)

### Fresnel Coefficients (Analytical)
```
r = (√ε₁ - √ε₂) / (√ε₁ + √ε₂)     (reflection)
t = 2√ε₁ / (√ε₁ + √ε₂)              (transmission)
```

---

## 3. PINN Architecture

```
Input: (x, y, t)
         ↓
    ┌─────────────┐
    │  Linear(3,64) │
    │  Tanh         │
    │  Linear(64,64)│
    │  Tanh         │
    │  Linear(64,64)│
    │  Tanh         │
    │  Linear(64,64)│
    │  Tanh         │
    │  Linear(64,3) │
    └─────────────┘
         ↓
Output: (Ez, Hx, Hy) — 3 vector components
```

---

## 4. Loss Functions

| Loss | Equation | Weight |
|------|---------|--------|
| **PDE** | 3 Maxwell curl residuals | λ=1.0 |
| **BC** | Ez = cos(ωt) at x=0 | λ=10.0 |
| **Data** | Analytical (Fresnel) at 1000 points | λ=5.0 |

---

## 5. Code Structure

| Section | Description |
|---------|-------------|
| [1] Parameters | ε_r1=1, ε_r2=4, ω=2π, interface at x=0.5 |
| [2] Analytical | Fresnel coefficients (normal incidence) |
| [3] PINN Model | 3→64→64→64→64→3 (predicts Ez, Hx, Hy) |
| [4] Collocation | 5000 interior + 500 BC + 1000 reference |
| [5] Loss | Maxwell PDE + BC + Data |
| [6] Training | 5000 epochs, Adam |
| [7] Visualization | Snapshots, 1D cross-section, loss, concept |
| [8] Summary | Relative L2 error for Ez, Hy |

---

## 6. How to Run

```cmd
cd E:\physicsnemo-tutorials\pinn\maxwell
python maxwell.py
```

Results saved to `results/`:
- `maxwell_snapshots.png` — Ez field at 4 time snapshots (analytical vs PINN vs error)
- `maxwell_1d.png` — 1D cross-section (Ez, Hx, Hy) at y=0.5
- `maxwell_loss.png` — Training loss history
- `maxwell_concept.png` — Setup diagram + equation comparison

---

## 7. Why Maxwell Matters

1. **Antenna design**: Radiation patterns, impedance matching
2. **Photonics**: Waveguides, photonic crystals, metamaterials
3. **RF/microwave**: Circuit analysis, EM compatibility
4. **Optics**: Refraction, reflection, polarization
5. **Digital twins**: Real-time EM field monitoring

---

## 8. References

- Jackson, "Classical Electrodynamics" (1999)
- Taflove & Hagness, "Computational Electrodynamics: FDTD" (2005)
- Chen et al., "Physics-Informed Neural Networks for Maxwell's Equations" (2022)
