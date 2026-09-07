# Helmholtz Equation PINN (Acoustic Wave Scattering)

> **Category**: `pinn/` — Frequency-domain, complex-valued field
> **Paradigm**: PINN with Helmholtz equation (complex field via Re/Im split)
> **Problem**: 2D acoustic scattering from a sound-hard cylinder

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing Wave (FNO) | Existing Electrostatics | **THIS (Helmholtz)** |
|--------|---------------------|------------------------|----------------------|
| **Domain** | Time-domain | Static | **Frequency-domain** |
| **Field** | Real-valued | Real-valued | **Complex (Re + Im)** |
| **BC** | Dirichlet/Neumann | Dirichlet/Neumann | **Sommerfeld radiation** |
| **Scattering** | ✗ | ✗ | **✓ (cylinder scatterer)** |
| **PDE** | u_tt = c²∇²u | ∇²u = f | **∇²u + k²u = 0** |

### Key Difference: Complex-Valued Frequency-Domain
- **Time-domain wave**: u(x,y,t) — real-valued, evolves in time
- **Helmholtz**: u(x,y,ω) — complex-valued, single frequency, steady-state
- The complex field encodes both amplitude and phase of the acoustic wave

---

## 2. Physics: Acoustic Scattering

### Problem Setup
```
                    Sommerfeld BC (r=R)
                    ┌─────────────────┐
                    │                 │
  Incident wave    │    Annulus      │
  → → → → → →     │   (a < r < R)   │
  exp(ikx)         │                 │
                    │    ┌───┐       │
                    │    │ ⌀ │ ← Cylinder (r=a)
                    │    └───┘       │
                    │  sound-hard    │
                    └─────────────────┘
```

### Helmholtz Equation
```
∇²u + k²u = 0  (in annulus: a < r < R)
```

Where:
- `u = u_incident + u_scattered` (total field)
- `u_incident = exp(ikx)` (plane wave)
- `k = 2π/λ` (wavenumber)

### Boundary Conditions

**Cylinder surface (r = a):** Sound-hard (Neumann)
```
∂u/∂n = 0  (no normal velocity)
```

**Outer boundary (r = R):** Sommerfeld radiation condition
```
∂u_s/∂r - iku_s = 0  (outgoing waves only, no reflection)
```

### Complex Field Split
```
u = u_r + i·u_i

→ ∇²u_r + k²u_r = 0  (real part)
→ ∇²u_i + k²u_i = 0  (imaginary part)
```

---

## 3. Analytical Solution

For a sound-hard cylinder, the analytical solution uses Bessel/Hankel functions:

```
u_total = u_inc + u_sc

u_inc = exp(ikr cos θ)

u_sc = Σ_n A_n H_n(kr) exp(inθ)

A_n = -(-i)^n · J_n'(ka) / H_n'(ka)
```

Where:
- `J_n` = Bessel function of first kind
- `H_n` = Hankel function of first kind (outgoing wave)
- `'` = derivative w.r.t. argument

This is used as reference data for PINN training/validation.

---

## 4. PINN Architecture

```
Input: (x, y)
         ↓
    ┌─────────────┐
    │  Linear(2,64) │
    │  Tanh         │
    │  Linear(64,64)│
    │  Tanh         │
    │  Linear(64,64)│
    │  Tanh         │
    │  Linear(64,64)│
    │  Tanh         │
    │  Linear(64,2) │
    └─────────────┘
         ↓
Output: (u_real, u_imag)
```

The network predicts the **total field** (incident + scattered). The incident field is known analytically, so the scattered field is obtained by subtraction.

---

## 5. Loss Functions

| Loss | Equation | Weight |
|------|---------|--------|
| **PDE** | ∇²u_r + k²u_r = 0, ∇²u_i + k²u_i = 0 | λ=1.0 |
| **Cylinder BC** | ∂u/∂n = 0 on r=a | λ=10.0 |
| **Sommerfeld BC** | ∂u/∂r - iku = ik(cosθ-1)u_inc on r=R | λ=5.0 |
| **Data** | Match analytical solution at 500 points | λ=2.0 |

### Sommerfeld BC Derivation
For total field u = u_inc + u_sc:
```
∂u/∂r - ik·u = ∂u_inc/∂r - ik·u_inc  (since u_sc satisfies Sommerfeld)
             = ik(cosθ - 1) · u_inc
```

Split into real/imaginary:
```
Re: ∂u_r/∂r + k·u_i = k(cosθ-1)·(-u_inc_i)
Im: ∂u_i/∂r - k·u_r = k(cosθ-1)·(u_inc_r)
```

---

## 6. Code Structure

| Section | Description |
|---------|-------------|
| [1] Parameters | Wavenumber k, cylinder radius, domain size |
| [2] Analytical | Bessel/Hankel series solution (validation) |
| [3] PINN Model | 2→64→64→64→64→2 (predicts Re + Im) |
| [4] Collocation | 3000 interior + 300 cylinder + 300 outer |
| [5] Loss | PDE + Cylinder BC + Sommerfeld BC + Data |
| [6] Training | 5000 epochs, Adam, step LR decay |
| [7] Visualization | Re/Im/Magnitude comparison, loss, concept |
| [8] Summary | Relative L2 error vs analytical |

---

## 7. How to Run

```cmd
cd E:\physicsnemo-tutorials\pinn\helmholtz
python helmholtz.py
```

Results saved to `results/`:
- `helmholtz_real.png` — Real part: analytical vs PINN vs error
- `helmholtz_imag.png` — Imaginary part: analytical vs PINN vs error
- `helmholtz_magnitude.png` — Field magnitude |u|: analytical vs PINN vs error
- `helmholtz_loss.png` — Training loss history
- `helmholtz_concept.png` — Scattering setup + equation comparison

---

## 8. Why Helmholtz Matters

1. **Frequency-domain analysis**: Many engineering problems are steady-state (no time evolution needed)
2. **Complex fields**: Amplitude + phase information (not just real values)
3. **Scattering problems**: Radar, sonar, nondestructive evaluation, antenna design
4. **Radiation BC**: Sommerfeld condition is fundamental for open-boundary problems
5. **Eigenvalue problems**: Helmholtz with source term → resonance/cavity analysis

---

## 9. References

- Colton & Monk, "An Integral Equation Method for Acoustic Scattering" (1987)
- Jin, "The Finite Element Method in Electromagnetics" (2015)
- Chen et al., "Physics-Informed Neural Networks for Helmholtz Equations" (2022)
