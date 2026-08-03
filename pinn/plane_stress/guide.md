# PINN Plane Stress Tutorial Guide: Solid Mechanics with Physics-Informed Neural Networks

## 1. Overview

This tutorial demonstrates **2D plane stress analysis** — a fundamental solid mechanics problem — using a Physics-Informed Neural Network (PINN). Unlike the existing PhysicsNeMo tutorials which focus on fluid dynamics (Navier-Stokes, Darcy flow), this tutorial addresses **structural mechanics**, a core CAE discipline that is not included in PhysicsNeMo by default.

### What is Plane Stress Analysis?

Plane stress analysis is a 2D simplification of 3D elasticity where the stress components in the out-of-plane direction (σ_zz, τ_xz, τ_yz) are assumed to be zero. This is valid for thin plates loaded in their plane.

**Applications:**
- Thin-walled structures (plates, shells)
- Bracket design and analysis
- Pressure vessel walls
- Sheet metal components
- MEMS devices

### How Does It Differ from CFD?

| Aspect | CFD (Existing Tutorials) | Solid Mechanics (This Tutorial) |
|--------|--------------------------|--------------------------------|
| **Governing PDE** | Navier-Stokes (momentum + continuity) | Equilibrium (∂σ/∂x = 0) |
| **Unknown field** | Velocity (u, v), pressure (p) | Displacement (u, v) |
| **Constitutive law** | Newtonian fluid (τ = μ∇u) | Hooke's law (σ = D·ε) |
| **Time dependence** | Often transient | Static (no time) |
| **Derivative chain** | 1st order (velocity → stress) | 2nd order (displacement → strain → stress → equilibrium) |
| **Boundary conditions** | Velocity/pressure | Displacement (Dirichlet) / traction (Neumann) |

---

## 2. Physics Background

### 2.1 Kinematics: Strain-Displacement Relations

In small-deformation elasticity, the strain tensor is the symmetric gradient of the displacement field:

```
ε_xx = ∂u/∂x          (normal strain in x)
ε_yy = ∂v/∂y          (normal strain in y)
γ_xy = ∂u/∂y + ∂v/∂x  (engineering shear strain)
```

**Physical meaning:**
- **ε_xx**: How much a line element originally in the x-direction stretches per unit length
- **ε_yy**: Same for y-direction
- **γ_xy**: Change in angle between originally perpendicular line elements

**Key assumption**: Small deformations — the strain-displacement relations are linearized. For large deformations, we would need the Green-Lagrange strain (nonlinear).

### 2.2 Constitutive Law: Hooke's Law (Plane Stress)

For a **linear elastic, isotropic material** under plane stress (σ_zz = 0):

```
σ_xx = E/(1-ν²) × (ε_xx + ν × ε_yy)
σ_yy = E/(1-ν²) × (ε_yy + ν × ε_xx)
τ_xy = G × γ_xy = E/(2(1+ν)) × γ_xy
```

where:
- **E** (Young's modulus): Material stiffness — how much stress is needed for a given strain
- **ν** (Poisson's ratio): Lateral contraction — when you pull in x, how much does it contract in y (typically 0.3 for metals)
- **G** (Shear modulus): Resistance to shear deformation

**Matrix form:**
```
[σ_xx]   [C11  C12  0 ] [ε_xx]
[σ_yy] = [C12  C11  0 ] [ε_yy]
[τ_xy]   [0    0    C33] [γ_xy]
```

where C11 = E/(1-ν²), C12 = νE/(1-ν²), C33 = E/(2(1+ν))

### 2.3 Equilibrium Equations

The fundamental PDE of solid mechanics is **force equilibrium**. For a static body with no body forces:

```
∂σ_xx/∂x + ∂τ_xy/∂y = 0   (x-direction equilibrium)
∂τ_xy/∂x + ∂σ_yy/∂y = 0   (y-direction equilibrium)
```

**Physical meaning**: The net force on any infinitesimal element must be zero (Newton's second law with zero acceleration).

**Derivation**: Consider a 2D element of size dx × dy. The stress varies across the element. The net force in x is:
```
(σ_xx + ∂σ_xx/∂x · dx)·dy - σ_xx·dy + (τ_xy + ∂τ_xy/∂y · dy)·dx - τ_xy·dx = 0
```
Simplifying: ∂σ_xx/∂x + ∂τ_xy/∂y = 0

### 2.4 Boundary Conditions

**Dirichlet (essential) BC**: Prescribed displacement
```
u = ū  on  Γ_D  (e.g., fixed support: u = v = 0)
```

**Neumann (natural) BC**: Prescribed traction
```
t = σ · n  on  Γ_N  (e.g., applied load, free surface)
```

where n is the outward unit normal and t is the traction vector.

For the cantilever beam:
- **Left end (x=0)**: u = 0, v = 0 (fixed support)
- **Right end (x=L)**: Applied downward load (traction)
- **Top/bottom (y=0, y=H)**: Free surface (traction = 0)

### 2.5 Euler-Bernoulli Beam Theory (Validation)

For a cantilever beam with end load P, the analytical deflection is:

```
v(x) = P·x²·(3L - x) / (6·E·I)
```

where I = H³/12 is the second moment of area for a rectangular cross-section.

**Assumptions**:
- Plane sections remain plane (Euler-Bernoulli hypothesis)
- Small deformations
- Linear elastic material
- Shear deformation neglected (slender beam: L >> H)

This provides a reference solution to validate the PINN results.

---

## 3. PINN Formulation for Solid Mechanics

### 3.1 Network Architecture

```
Input: (x, y) → MLP (6 layers, 64 neurons, tanh) → Output: (u, v)
```

The network predicts the displacement field directly. The `tanh` activation is chosen because:
1. **Smoothness**: Infinitely differentiable (needed for 2nd-order PDE)
2. **Bounded**: Prevents displacement blow-up
3. **Zero-centered**: Helps with convergence

### 3.2 Derivative Chain

The PINN computes the full physics chain using automatic differentiation:

```
Network output: u(x,y), v(x,y)
    ↓ 1st derivative (autograd)
Strain: ε_xx = ∂u/∂x, ε_yy = ∂v/∂y, γ_xy = ∂u/∂y + ∂v/∂x
    ↓ Constitutive law (Hooke's law)
Stress: σ_xx, σ_yy, τ_xy
    ↓ 1st derivative of stress (autograd)
Equilibrium residual: ∂σ_xx/∂x + ∂τ_xy/∂y, ∂τ_xy/∂x + ∂σ_yy/∂y
```

This requires **second-order derivatives** of the network output (displacement → strain → stress → equilibrium), which is more demanding than CFD PINNs that typically need only first-order derivatives.

### 3.3 Loss Function

```
L = λ_pde · L_pde + λ_left · L_left + λ_right · L_right + λ_free · L_free
```

| Loss Term | Equation | Physical Meaning |
|-----------|----------|-----------------|
| L_pde | ∂σ_xx/∂x + ∂τ_xy/∂y = 0 | Interior equilibrium |
| L_left | u = 0, v = 0 at x=0 | Fixed support |
| L_right | v = v_target at x=L | Applied load (weak form) |
| L_free | σ_yy = 0, τ_xy = 0 at y=0,H | Free surface |

**Loss weights** (λ) balance the competing objectives. Dirichlet BCs typically need higher weights because they are enforced as hard constraints.

---

## 4. Comparison with CFD PINN

### 4.1 Derivative Order

| | CFD (Lid-Driven Cavity) | Solid Mechanics (Plane Stress) |
|---|---|---|
| **1st derivative** | Velocity gradient → stress | Displacement gradient → strain |
| **2nd derivative** | Velocity Laplacian → viscosity | Strain gradient → equilibrium |
| **Total autograd depth** | 2nd order | 2nd order (but through constitutive law) |

### 4.2 Constitutive Law

- **CFD**: Newtonian fluid: τ = μ(∇u + ∇uᵀ) — stress is directly proportional to velocity gradient
- **Solid**: Hooke's law: σ = D·ε — stress is proportional to strain, which is the gradient of displacement

The key difference is that in solids, the "field" (displacement) is one derivative removed from the PDE variable (stress), requiring an extra layer of differentiation.

### 4.3 Boundary Conditions

- **CFD**: Typically velocity (Dirichlet) or pressure (Neumann)
- **Solid**: Displacement (Dirichlet) or traction (Neumann) — traction involves stress, which is already a derivative of the field

---

## 5. Results Interpretation

### 5.1 Displacement Field

- **u(x,y)**: Horizontal displacement — should be zero at the fixed end and small elsewhere
- **v(x,y)**: Vertical displacement — should show beam bending (maximum at the free end)

### 5.2 Stress Fields

- **σ_xx**: Bending stress — positive (tension) at the top, negative (compression) at the bottom near the fixed end
- **σ_yy**: Should be near zero (free surface condition)
- **τ_xy**: Shear stress — carries the vertical load, maximum near the neutral axis

### 5.3 Validation

The PINN deflection at y = H/2 is compared with the Euler-Bernoulli analytical solution. Discrepancies arise from:
1. **Shear deformation**: Euler-Bernoulli neglects shear; PINN captures it
2. **Poisson effect**: 2D elasticity includes lateral contraction; beam theory doesn't
3. **Stress concentration**: Near the fixed end, stress is singular (not captured by beam theory)

---

## 6. Extensions

1. **Plane Strain**: Change constitutive matrix for thick bodies (σ_zz ≠ 0)
2. **Nonlinear Material**: Plasticity, hyperelasticity (replace Hooke's law)
3. **Large Deformation**: Use Green-Lagrange strain and 2nd PK stress
4. **Contact**: Add contact constraints between bodies
5. **Dynamics**: Add inertia term (ρ ü) for transient analysis
6. **Thermoelasticity**: Add thermal strain (ε_thermal = α·ΔT)

---

## 7. Key Takeaways

1. **Solid mechanics** uses equilibrium equations (∂σ/∂x = 0), not Navier-Stokes
2. **Displacement → strain → stress → equilibrium** is the derivative chain in elasticity
3. **Hooke's law** (σ = D·ε) is the constitutive relation for linear elastic materials
4. **Plane stress** assumes σ_zz = 0 (thin plates); **plane strain** assumes ε_zz = 0 (thick bodies)
5. **PINN for solids** requires 2nd-order autograd through the constitutive law
6. **Euler-Bernoulli beam theory** provides an analytical benchmark for validation
