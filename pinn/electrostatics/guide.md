# PINN Electrostatics Tutorial Guide: Electromagnetic Analysis with Physics-Informed Neural Networks

## 1. Overview

This tutorial demonstrates **2D electrostatic analysis** using a Physics-Informed Neural Network (PINN). This is an **electromagnetics** problem — a core CAE discipline — that is not available in PhysicsNeMo by default. The existing PhysicsNeMo tutorials focus on fluid dynamics (Navier-Stokes, Darcy flow), while this tutorial addresses Maxwell's equations in the static limit.

### What is Electrostatic Analysis?

Electrostatic analysis computes the electric potential and field distribution produced by electric charges in a domain. It is the simplest form of electromagnetic simulation, dealing with charges at rest (no time-varying fields).

**Applications:**
- Capacitor design (energy storage, dielectric optimization)
- PCB/electronics shielding (EMI/EMC analysis)
- High-voltage insulation design
- Particle accelerators (beam optics)
- Electrostatic precipitators (pollution control)
- MEMS devices (electrostatic actuators)
- Lightning protection design
- Sensor design (capacitive sensors)

### How Does It Differ from CFD?

| Aspect | CFD (Existing Tutorials) | Electrostatics (This Tutorial) |
|--------|--------------------------|-------------------------------|
| **Governing PDE** | Navier-Stokes (momentum + continuity) | Poisson: ∇²φ = -ρ/ε₀ |
| **Unknown field** | Velocity (u,v), pressure (p) | Potential φ (scalar) |
| **Source term** | Body force, pressure gradient | Charge density ρ |
| **Flux law** | Newtonian fluid (τ = μ∇u) | Gauss's law (∇·E = ρ/ε₀) |
| **Field relation** | Velocity → stress (1st derivative) | E = -∇φ (1st derivative) |
| **PDE order** | 2nd order (viscous term) | 2nd order (Laplacian) |
| **Time dependence** | Often transient | Static (no time) |

---

## 2. Physics Background

### 2.1 Coulomb's Law

The fundamental force law of electrostatics, discovered by Charles-Augustin de Coulomb (1785):

```
F = (1 / 4πε₀) × q₁q₂ / r²
```

where:
- **F**: Force between charges (N)
- **q₁, q₂**: Electric charges (C)
- **r**: Distance between charges (m)
- **ε₀**: Permittivity of free space = 8.854 × 10⁻¹² F/m

**Key features:**
- Inverse-square law (like gravity)
- Like charges repel (F > 0), opposite charges attract (F < 0)
- Acts along the line connecting charges
- Superposition: total force = vector sum of individual forces

### 2.2 Electric Field

The electric field is defined as the force per unit charge:

```
E = F / q_test = (1 / 4πε₀) × q / r²  (radial from point charge)
```

**Physical meaning**: The electric field at a point is the force that a unit positive test charge would experience if placed at that point. It is a vector field — it has both magnitude and direction at every point in space.

- **Positive charge**: E points radially outward
- **Negative charge**: E points radially inward
- **Units**: V/m (Volts per meter) or N/C (Newtons per Coulomb)

### 2.3 Gauss's Law

Gauss's law relates the electric field to the charge distribution:

**Integral form:**
```
∮ E · dA = Q_enc / ε₀
```
The total electric flux through a closed surface equals the enclosed charge divided by ε₀.

**Differential form:**
```
∇·E = ρ / ε₀
```
The divergence of E at a point equals the charge density at that point divided by ε₀.

**Physical meaning**: Electric field lines originate from positive charges and terminate on negative charges. The "strength" of the source/sink is proportional to the charge density.

### 2.4 Electric Potential

The electric potential φ is defined as the work done per unit charge to move a test charge from a reference point to the field point:

```
E = -∇φ
```

**Physical meaning**: The electric field points in the direction of **decreasing** potential. Charges "fall" from high potential to low potential, just as masses fall from high elevation to low elevation in a gravitational field.

**Units**: Volts (V) = Joules per Coulomb (J/C)

**Why use potential?**
1. **Scalar field**: φ is a scalar (one number per point), while E is a vector (two numbers in 2D)
2. **Superposition**: Potentials add algebraically (no vector addition needed)
3. **Energy**: Potential energy = qφ (simple multiplication)
4. **Measurement**: Voltage is easier to measure than electric field

### 2.5 Poisson's and Laplace's Equations

Combining Gauss's law (∇·E = ρ/ε₀) with the potential definition (E = -∇φ):

```
∇·(-∇φ) = ρ/ε₀
-∇²φ = ρ/ε₀
∇²φ = -ρ/ε₀   (Poisson's equation)
```

**In regions with no charge (ρ = 0):**
```
∇²φ = 0   (Laplace's equation)
```

**In 2D Cartesian coordinates:**
```
∂²φ/∂x² + ∂²φ/∂y² = -ρ/ε₀
```

This is the PDE that the PINN must satisfy. It is an **elliptic** equation (like the steady-state heat equation), meaning:
- Information propagates instantly throughout the domain
- Boundary conditions determine the solution everywhere
- No time dependence, no wave propagation

### 2.6 Boundary Conditions

**Dirichlet (prescribed potential):**
```
φ = V₀  on  Γ_D
```
Example: Conductor at fixed voltage (e.g., grounded wall: φ = 0)

**Neumann (prescribed field/charge):**
```
-∂φ/∂n = E_n  on  Γ_N
```
Example: Known surface charge density (σ_s = ε₀ E_n), or symmetry boundary (∂φ/∂n = 0)

**Robin (mixed):**
```
∂φ/∂n + αφ = β  on  Γ_R
```
Example: Surface impedance boundary condition

### 2.7 Method of Images (Analytical Solution)

For a point charge in a grounded conducting box, the **method of images** provides an exact analytical solution:

**Principle**: Replace the conducting boundaries with **image charges** outside the domain such that the boundary condition (φ = 0 on walls) is automatically satisfied.

For a charge q at (x₀, y₀) in a box [0, Lx] × [0, Ly]:
- Image charges at positions (±x₀ + 2iLx, ±y₀ + 2jLy) for i, j = 0, ±1, ±2, ...
- Alternating signs to satisfy φ = 0 on all walls

```
φ(x,y) = Σ (-1)^(i+j) × q / (4πε₀ × r_ij)
```

This provides a reference solution to validate the PINN results.

---

## 3. PINN Formulation for Electrostatics

### 3.1 Network Architecture

```
Input: (x, y) → MLP (6 layers, 64 neurons, tanh) → Output: φ
```

The network predicts the **electric potential** directly. The `tanh` activation is chosen because:
1. **Smoothness**: Infinitely differentiable (needed for 2nd-order PDE)
2. **Bounded**: Prevents potential blow-up
3. **Zero-centered**: Helps with convergence

### 3.2 Derivative Chain

The PINN computes the full physics chain using automatic differentiation:

```
Network output: φ(x,y)
    ↓ 1st derivative (autograd)
Electric field: E = -∇φ  →  Ex = -∂φ/∂x, Ey = -∂φ/∂y
    ↓ 2nd derivative (autograd)
Laplacian: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²
    ↓ Poisson equation
Residual: R = ∇²φ + ρ/ε₀  (should be zero)
```

This requires **second-order derivatives** of the network output, similar to the CFD PINN tutorials.

### 3.3 Charge Regularization

A point charge has infinite charge density (delta function), which is problematic for neural network training. We regularize it with a **Gaussian distribution**:

```
ρ(x,y) = q / (2πσ²) × exp(-((x-x₀)² + (y-y₀)²) / (2σ²))
```

This:
- Integrates to q (total charge is preserved)
- Is smooth (infinitely differentiable)
- Approaches a delta function as σ → 0
- Avoids numerical singularity in training

### 3.4 Loss Function

```
L = λ_pde · L_pde + λ_bc · L_bc
```

| Loss Term | Equation | Physical Meaning |
|-----------|----------|-----------------|
| L_pde | ∇²φ + ρ/ε₀ = 0 | Poisson equation (interior) |
| L_bc | φ = 0 on walls | Grounded conductor (boundary) |

**Loss weights**: λ_bc = 50 (Dirichlet BC needs strong enforcement), λ_pde = 1.

---

## 4. Comparison with Other Tutorials

### 4.1 vs. CFD PINN (Lid-Driven Cavity)

| | CFD (Lid-Driven Cavity) | Electrostatics (This Tutorial) |
|---|---|---|
| **PDE** | Navier-Stokes (nonlinear) | Poisson (linear) |
| **Unknowns** | u, v, p (3 fields) | φ (1 field) |
| **Nonlinearity** | Convective term (u·∇u) | None (linear PDE) |
| **Derivatives** | 1st (advection) + 2nd (viscous) | 2nd (Laplacian only) |
| **Source** | None (driven by BC) | Charge density ρ |
| **Difficulty** | Harder (nonlinear coupling) | Easier (linear, scalar) |

### 4.2 vs. Heat Conduction (FNO)

| | Heat Conduction (FNO) | Electrostatics (PINN) |
|---|---|---|
| **PDE** | ∇·(k∇T) + Q = 0 | ∇²φ + ρ/ε₀ = 0 |
| **Method** | FNO (operator learning) | PINN (function approximation) |
| **Input** | Field (k, Q) | Coordinates (x, y) |
| **Output** | T field | φ at point |
| **Training** | Supervised (needs data) | Physics-informed (no data) |
| **Generalization** | Any (k, Q) from distribution | One specific problem |

The electrostatics and heat conduction equations are mathematically identical (both Poisson equations), but the solution methods differ:
- **FNO**: Learns the operator for many different source terms
- **PINN**: Solves one specific problem using physics constraints

---

## 5. Results Interpretation

### 5.1 Electric Potential φ(x,y)

- **Peak**: At the charge location (x₀, y₀) — maximum potential
- **Boundary**: φ = 0 on all walls (grounded)
- **Shape**: Radially symmetric near the charge, distorted by the box
- **Sign**: Positive for positive charge (q > 0)

### 5.2 Electric Field E = -∇φ

- **Direction**: Radially outward from positive charge
- **Magnitude**: Decreases as 1/r from the charge (Coulomb's law)
- **Boundary**: Field lines perpendicular to conductor surface
- **Vectors**: Point from charge toward walls (down the potential gradient)

### 5.3 Validation with Method of Images

The PINN potential is compared with the analytical solution (method of images):
- **Agreement**: Should be good away from the charge
- **Error**: Concentrated near the charge (singularity region)
- **Boundary**: Should be exactly zero (Dirichlet BC enforced)

---

## 6. Extensions

1. **Multiple Charges**: Add multiple Gaussian charge distributions
2. **Dielectric Materials**: ∇·(ε∇φ) = -ρ (variable permittivity)
3. **Conductors**: Floating conductors (φ = const, unknown value)
4. **3D Electrostatics**: Extend to ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z² = -ρ/ε₀
5. **Magnetostatics**: ∇²A = -μJ (vector potential for magnetic fields)
6. **Time-Harmonic**: ∇²φ - με ∂²φ/∂t² = -ρ/ε₀ (wave equation)
7. **Full Maxwell**: Coupled E and H fields with time dependence

---

## 7. Key Takeaways

1. **Electrostatics** is governed by Poisson's equation: ∇²φ = -ρ/ε₀
2. **Electric field** is the negative gradient of potential: E = -∇φ
3. **Gauss's law** (∇·E = ρ/ε₀) is the differential form of charge conservation
4. **Method of images** provides analytical solutions for charges near conductors
5. **Charge regularization** (Gaussian) is needed to avoid singularities in PINN training
6. **Electrostatics** is mathematically equivalent to steady-state heat conduction (both Poisson equations)
7. **PINN for electrostatics** is simpler than CFD because the PDE is linear and scalar
