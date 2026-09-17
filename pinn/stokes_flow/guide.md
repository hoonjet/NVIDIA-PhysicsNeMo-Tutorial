# Stokes Flow PINN (Creeping Flow)

> **Category**: `pinn/` — Low Reynolds number, viscous-dominated flow
> **Problem**: 2D lid-driven cavity at Re→0

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing LDC (Re=100) | **THIS (Stokes, Re→0)** |
|--------|---------------------|------------------------|
| **Inertia** | u·∇u included | **Removed (linear)** |
| **Equation** | Nonlinear | **Linear** |
| **Vortex** | Corner vortices | **Symmetric primary** |
| **Application** | General CFD | **Microfluidics, bio** |

---

## 2. Physics

```
0 = -∇p + μ∇²u    (momentum, no inertia)
0 = ∇·u           (continuity)
```

- **μ** = 1.0 (dynamic viscosity)
- Re → 0 (creeping flow regime)
- BC: Top wall moves (u=1), others fixed

---

## 3. How to Run

```cmd
cd E:\physicsnemo-tutorials\pinn\stokes_flow
python stokes_flow.py
```

---

## 4. Results

- `stokes_result.png` — U, V, P, streamlines
- `stokes_loss.png` — Training loss
- `stokes_concept.png` — Stokes vs Navier-Stokes comparison
