# FSI PINN (Fluid-Structure Interaction)

> **Category**: `pinn/` — Multi-physics coupling (fluid + solid)
> **Problem**: Channel flow over elastic beam

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing CHT | **THIS (FSI)** |
|--------|-------------|----------------|
| **Coupling** | Heat flux + temperature | **Velocity + force** |
| **Interface** | Stationary | **Deformable** |
| **Physics** | Fluid heat + solid heat | **Fluid flow + solid elasticity** |
| **Models** | Single PINN | **Two coupled PINNs** |

---

## 2. Physics

```
Fluid (Stokes):   0 = -∇p + μ∇²u,  ∇·u = 0
Solid (Elastic):  ∇·σ = 0,  σ = λ·tr(ε)·I + 2μ·ε

Interface coupling:
  1. Velocity continuity:  u_fluid = u_solid
  2. Force balance:        σ_fluid·n = σ_solid·n
```

- **Fluid**: μ=0.1, inflow U=1.0
- **Solid**: E=10, ν=0.3 (linear elasticity)
- Two separate PINNs trained jointly

---

## 3. How to Run

```cmd
cd E:\physicsnemo-tutorials\pinn\fsi
python fsi.py
```

---

## 4. Results

- `fsi_result.png` — Fluid velocity, pressure, solid displacement, streamlines
- `fsi_loss.png` — Training loss (fluid + solid + interface + BCs)
- `fsi_concept.png` — FSI vs CHT comparison, applications
