# FNO Heat Conduction Tutorial Guide: Thermal Analysis with Fourier Neural Operator

## 1. Overview

This tutorial demonstrates **2D steady-state heat conduction** using PhysicsNeMo's **FNO (Fourier Neural Operator)**. This is a thermal analysis problem — a core CAE discipline — that is not available in PhysicsNeMo by default. The existing PhysicsNeMo tutorials focus on fluid dynamics (Darcy flow, lid-driven cavity), while this tutorial addresses heat transfer.

### What is Heat Conduction?

Heat conduction is the transfer of thermal energy through a material without bulk fluid motion. It occurs in solids, stationary fluids, and is governed by Fourier's law and the heat equation.

**Applications:**
- Electronics cooling (heat sinks, PCB thermal management)
- Building insulation (wall, window thermal analysis)
- Manufacturing (welding, casting, heat treatment)
- Nuclear reactor design
- Battery thermal management
- Aerospace thermal protection systems

### How Does It Differ from CFD?

| Aspect | CFD (Darcy Flow) | Heat Conduction (This Tutorial) |
|--------|-----------------|--------------------------------|
| **Governing PDE** | Darcy: ∇·(k∇p) = 0 | Poisson: ∇·(k∇T) + Q = 0 |
| **Unknown field** | Pressure p (scalar) | Temperature T (scalar) |
| **Physical meaning** | Fluid flow through porous media | Heat flow through solid |
| **Source term** | None | Heat source Q (W/m³) |
| **Flux law** | Darcy: v = -(k/μ)∇p | Fourier: q = -k∇T |
| **Boundary conditions** | Pressure/velocity | Temperature/heat flux |

---

## 2. Physics Background

### 2.1 Fourier's Law of Heat Conduction

The fundamental law of heat conduction, discovered by Joseph Fourier (1822):

```
q = -k ∇T
```

where:
- **q** = heat flux vector (W/m²) — heat flow per unit area per unit time
- **k** = thermal conductivity (W/m·K) — material's ability to conduct heat
- **∇T** = temperature gradient (K/m)

**Components in 2D:**
```
q_x = -k ∂T/∂x
q_y = -k ∂T/∂y
```

**Physical meaning**: Heat flows from hot regions to cold regions (down the temperature gradient). The negative sign ensures this direction. The rate of heat flow is proportional to the temperature difference and the material's conductivity.

**Material properties:**
| Material | k (W/m·K) | Type |
|----------|-----------|------|
| Copper | 401 | High conductor |
| Aluminum | 237 | High conductor |
| Steel | 50 | Moderate conductor |
| Water | 0.6 | Poor conductor |
| Air | 0.026 | Insulator |
| Insulation | 0.04 | Insulator |

### 2.2 Energy Conservation (Heat Equation)

The heat equation is derived from energy conservation on an infinitesimal element:

```
Rate of energy in - Rate of energy out + Generation = Rate of energy storage
```

For **steady-state** (no time dependence) with no storage:

```
∇·(k∇T) + Q = 0
```

where Q is the volumetric heat source (W/m³).

**Derivation**: Consider a control volume dx × dy. The net heat entering by conduction is:
```
-∂q_x/∂x · dx · dy - ∂q_y/∂y · dy · dx = (∂/∂x(k ∂T/∂x) + ∂/∂y(k ∂T/∂y)) · dx · dy
```

Adding the heat source Q · dx · dy and setting the sum to zero (steady state):
```
∂/∂x(k ∂T/∂x) + ∂/∂y(k ∂T/∂y) + Q = 0
```

### 2.3 Constant vs. Variable Conductivity

**Constant k**: The equation simplifies to the Poisson equation:
```
k ∇²T + Q = 0  →  ∇²T = -Q/k
```

**Variable k(x,y)**: The full form must be used:
```
∂/∂x(k(x,y) ∂T/∂x) + ∂/∂y(k(x,y) ∂T/∂y) + Q = 0
```

This is important for composite materials, functionally graded materials, or when temperature-dependent conductivity matters.

### 2.4 Boundary Conditions

**Dirichlet (prescribed temperature)**:
```
T = T₀  on  Γ_D
```
Example: A surface in contact with a known temperature reservoir.

**Neumann (prescribed heat flux)**:
```
-k ∂T/∂n = q₀  on  Γ_N
```
Example: A heating element supplying known heat flux. Insulated boundary: q₀ = 0.

**Robin (convection)**:
```
-k ∂T/∂n = h(T - T_∞)  on  Γ_R
```
Example: Surface exposed to fluid at temperature T_∞ with heat transfer coefficient h.

### 2.5 Analytical Solution (Simple Case)

For constant k, no source Q, with T = T_hot at x=0 and T = T_cold at x=1:

```
T(x,y) = T_hot + (T_cold - T_hot) · x
```

This is a linear temperature profile — the simplest heat conduction solution. The FNO tutorial uses more complex cases with variable k and heat sources.

---

## 3. FNO for Heat Conduction

### 3.1 Why FNO for Thermal Analysis?

Traditional thermal analysis uses:
- **Finite Difference Method (FDM)**: Discretize domain, solve linear system
- **Finite Element Method (FEM)**: Mesh-based, handles complex geometry
- **Finite Volume Method (FVM)**: Conservation-based, common in CFD

These methods require solving a large linear system for each new set of inputs (k, Q, BCs). For parametric studies (many different k, Q combinations), this is expensive.

**FNO advantage**: After training, FNO predicts the temperature field for any new (k, Q) input in a **single forward pass** — no iterative solver needed.

### 3.2 FNO Architecture

```
Input: [k(x,y), Q(x,y)]  →  Lift  →  Fourier Layers  →  Project  →  T(x,y)
         2 channels              32 ch      4 layers          1 channel
```

**Fourier Layer Operations:**
1. **FFT**: Transform spatial features to frequency domain
2. **Spectral convolution**: Multiply low-frequency modes by learnable weights
3. **Inverse FFT**: Transform back to spatial domain
4. **Skip connection**: Add local (1×1 conv) path for high frequencies

**Key parameter: n_modes** — number of Fourier modes retained. More modes = finer details but more parameters.

### 3.3 Operator Learning vs. Function Approximation

| | Standard NN (PINN) | FNO |
|---|---|---|
| **Learns** | One solution function | A family of solutions (operator) |
| **Input** | Coordinates (x, y) | Entire field (k, Q) |
| **Output** | T at that point | T field everywhere |
| **Generalization** | One specific problem | Any (k, Q) from same distribution |
| **Physics** | PDE residual (autograd) | Data-driven (supervised) |

---

## 4. Tutorial Code Walkthrough

### Step 1: Data Generation

```python
inputs, outputs = generate_heat_data(n_samples=200, resolution=32)
```

For each sample:
1. Generate random conductivity field k(x,y) using sinusoidal patterns
2. Generate random Gaussian heat source Q(x,y)
3. Solve ∇·(k∇T) + Q = 0 using finite differences (Gauss-Seidel, 2000 iterations)
4. Store (k, Q) as input, T as output

### Step 2: FNO Model

```python
model = FNO(
    in_channels=2,      # [k, Q]
    out_channels=1,     # [T]
    n_modes=(8, 8),    # 8 Fourier modes in each direction
    hidden_channels=32, # width
    n_layers=4,         # depth
)
```

### Step 3: Training

```python
pred = model(inputs)  # FNO forward pass
loss = MSE(pred, true_T)  # Supervised loss
```

FNO is trained as a supervised operator: given (k, Q), predict T. No PDE residual is needed during training (unlike PINN).

### Step 4: Heat Flux Computation

After predicting T, compute heat flux using Fourier's law:
```python
q_x = -k * ∂T/∂x  (numerical gradient)
q_y = -k * ∂T/∂y
```

---

## 5. Results Interpretation

### 5.1 Temperature Field

- **Left boundary**: T = 100°C (hot)
- **Right boundary**: T = 0°C (cold)
- **Interior**: Temperature decreases from left to right, modified by:
  - Conductivity variations (high k → more uniform T)
  - Heat sources (Q > 0 → local temperature increase)

### 5.2 Heat Flux

- **Direction**: From hot (left) to cold (right)
- **Magnitude**: Higher in high-conductivity regions
- **Vectors**: Should point predominantly rightward, with deviations around heat sources

### 5.3 FNO vs. Finite Difference

| Metric | Finite Difference | FNO |
|--------|-------------------|-----|
| **Solve time** | ~2000 iterations | 1 forward pass |
| **Accuracy** | Exact (converged) | Approximate (trained) |
| **Generalization** | One problem at a time | Any (k, Q) from training distribution |
| **Memory** | O(N²) for matrix | O(params) for network |

---

## 6. Extensions

1. **Transient Heat Conduction**: Add time dimension: ρc ∂T/∂t = ∇·(k∇T) + Q
2. **Temperature-dependent k**: k = k(T) → nonlinear equation
3. **Convection BCs**: Robin boundary conditions (h, T_∞)
4. **3D Heat Conduction**: Extend to volumetric domains
5. **Coupled Thermal-Structural**: Thermal expansion → stress analysis
6. **Multi-physics**: Conjugate heat transfer (solid + fluid)

---

## 7. Key Takeaways

1. **Heat conduction** is governed by ∇·(k∇T) + Q = 0 (Poisson equation for constant k)
2. **Fourier's law** (q = -k∇T) is the constitutive relation for heat flux
3. **FNO** learns the operator (k, Q) → T, enabling instant prediction for new inputs
4. **Variable conductivity** requires the full divergence form ∇·(k∇T)
5. **Thermal analysis** is a core CAE discipline distinct from fluid dynamics
6. **Operator learning** (FNO) is fundamentally different from function approximation (PINN)
