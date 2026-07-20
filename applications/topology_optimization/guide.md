# TopoDiff Tutorial Guide: Topology Optimization via Diffusion Model

## 1. Overview

This tutorial demonstrates **topology optimization** using PhysicsNeMo's **TopoDiff** model — a diffusion-based generative model that produces optimal structural designs given boundary conditions and loads.

### What is Topology Optimization?

Topology optimization is a structural design method that determines the **optimal material distribution** within a design domain. Unlike shape optimization (which modifies boundaries) or sizing optimization (which changes cross-sections), topology optimization can create entirely new structural layouts — including holes, branches, and complex internal architecture.

**Applications:**
- Aerospace: lightweight aircraft brackets, satellite components
- Automotive: crashworthy structures, lightweight chassis
- Civil: bridge designs, building connections
- Biomedical: bone implants, prosthetics
- Additive Manufacturing: 3D-printed optimized parts

### How Does It Differ from Traditional CAE?

| Aspect | Traditional (SIMP/MMA) | TopoDiff (ML-based) |
|--------|----------------------|---------------------|
| **Method** | Iterative FEM + gradient descent | Single forward pass |
| **Time** | Minutes to hours | Milliseconds |
| **Input** | Boundary conditions + loads | Same (encoded as images) |
| **Output** | Density field (0-1) | Same |
| **Iterations** | 100-500 FEM solves | 0 (pre-trained) |
| **Physics** | Full FEM compliance evaluation | Learned from data |

---

## 2. Physics Background

### 2.1 Structural Equilibrium

The fundamental equation of structural mechanics is:

```
K(x) U = F
```

where:
- **K(x)**: Global stiffness matrix (depends on material distribution x)
- **U**: Displacement vector (unknown)
- **F**: External force vector (known)

The stiffness matrix is assembled from element stiffness matrices:

```
K_e = x_e^p × E_0 × B^T × D_0 × B × A_e
```

where:
- `x_e`: density of element e (design variable, 0 to 1)
- `p`: SIMP penalization exponent (typically 3.0)
- `E_0`: Young's modulus of solid material
- `B`: strain-displacement matrix
- `D_0`: constitutive matrix (plane stress or plane strain)
- `A_e`: element area

### 2.2 SIMP Penalization

The SIMP (Solid Isotropic Material with Penalization) method uses:

```
E_e = x_e^p × E_0
```

- `p = 1`: No penalization (intermediate densities allowed → "gray" design)
- `p = 3`: Standard penalization (pushes toward 0/1 → "black-and-white" design)
- `p > 4`: Over-penalization (may cause premature convergence)

The penalization ensures that intermediate densities are energetically unfavorable, producing manufacturable designs with clear solid/void regions.

### 2.3 Compliance Minimization

The optimization objective is to minimize **compliance** (maximize stiffness):

```
minimize:  C(x) = F^T U = U^T K(x) U
```

Compliance is the work done by external forces = strain energy stored in the structure. Minimizing compliance means the structure deforms less under load (stiffer).

**Constraint**: Volume fraction
```
sum(x_e) / V_total ≤ V_frac  (e.g., 0.5 = use 50% of material)
```

### 2.4 Cantilever Beam Problem

The cantilever beam is a benchmark problem in topology optimization:

```
    Fixed wall          Free end
    +---------+         ↓ F (downward force)
    |         |
    |         |
    +---------+
    ←←←←←←←←← (fixed in x and y)
```

- **Left edge**: Fixed (Dirichlet boundary condition, u = v = 0)
- **Right edge**: Point load applied (Neumann boundary condition)
- **Design domain**: 2D rectangular region
- **Optimal topology**: Typically a truss-like structure connecting the fixed wall to the load point

The optimal design transfers load efficiently from the application point to the support, using the minimum amount of material.

---

## 3. Diffusion Model Background

### 3.1 What is a Diffusion Model?

A diffusion model is a **generative model** that learns to create data by reversing a noising process:

**Forward Process (Training):**
```
x_t = √(ᾱ_t) × x_0 + √(1 - ᾱ_t) × ε
```
where:
- `x_0`: clean data (optimal topology)
- `x_t`: noisy data at timestep t
- `ᾱ_t`: cumulative noise schedule
- `ε`: standard Gaussian noise

**Reverse Process (Sampling):**
```
x_{t-1} = (1/√α_t) × [x_t - (β_t/√(1-ᾱ_t)) × ε_θ(x_t, t, c)] + σ_t × z
```
where:
- `ε_θ`: neural network (TopoDiff) predicting noise
- `c`: conditioning (boundary conditions)
- `z`: random noise (for stochasticity)

### 3.2 TopoDiff Architecture

TopoDiff uses the **ADM (Ablated Diffusion Model)** architecture:

```
Input: [noisy_topology, conditions] → U-Net → predicted_noise
```

**U-Net Components:**
1. **Encoder**: Progressive downsampling with residual blocks
   - Level 0: 32×32, 64 channels
   - Level 1: 16×16, 128 channels (with self-attention)
   - Level 2: 8×8, 128 channels (with self-attention)
   - Level 3: 4×4, 128 channels

2. **Decoder**: Progressive upsampling with skip connections
   - Mirror of encoder, concatenating skip features

3. **Time embedding**: Sinusoidal positional encoding of diffusion timestep

4. **Conditioning**: Boundary conditions concatenated with noisy input

### 3.3 Why Diffusion for Topology Optimization?

1. **Multi-modality**: Multiple valid topologies exist for the same BCs — diffusion models naturally capture this distribution
2. **Diversity**: Can generate different designs for the same problem (useful for design exploration)
3. **Physics-aware**: Conditioning on boundary conditions ensures generated designs respect the problem setup
4. **Speed**: After training, generation is much faster than iterative FEM optimization

---

## 4. Tutorial Code Walkthrough

### Step 1: Data Generation

Since we don't have pre-computed FEM optimization results, we generate **synthetic topologies** that mimic cantilever beam optimal designs:

```python
topologies, conditions = generate_cantilever_topologies(n_samples=200, resolution=32)
```

- **Topologies**: Binary images (0=void, 1=solid) with truss-like patterns
- **Conditions**: 3-channel images [fixed_mask, load_x, load_y]

In production, replace this with real SIMP-optimized topologies from tools like:
- TopOpt (MATLAB)
- TOUGH (Python)
- CalculiX with optimization plugin

### Step 2: Model Creation

```python
model = TopoDiff(
    img_resolution=32,
    in_channels=1,      # noisy topology
    out_channels=1,      # predicted noise
    model_channels=64,   # U-Net width
    channel_mult=[1, 2, 2, 2],  # 4 resolution levels
    attn_resolutions=[16, 8],   # self-attention at 16x16 and 8x8
)
```

### Step 3: Diffusion Training

```python
diffusion = Diffusion(n_steps=200, min_beta=1e-4, max_beta=0.02)
loss = diffusion.train_loss(model, x0, conditions)
```

The training loss is:
```
L = MSE(ε_θ(x_t, t, c), ε)
```
where ε is the actual noise added during forward diffusion.

### Step 4: Sampling (Generation)

```python
x = sample_topology(model, diffusion, condition, n_steps=50)
```

Reverse diffusion: start from noise, iteratively denoise using the trained model.

---

## 5. Results Interpretation

### Expected Output

The tutorial produces a 4-row figure:
1. **Row 1**: Fixed wall boundary condition (red = fixed)
2. **Row 2**: Load condition (blue = downward force)
3. **Row 3**: Ground truth topology (from training data)
4. **Row 4**: Generated topology (from diffusion model)

### Quality Metrics

- **Volume fraction**: Should be close to 50% (target)
- **Connectivity**: Material should connect fixed wall to load point
- **Truss-like structure**: Should show load-bearing members
- **Black-and-white**: Minimal intermediate densities

### Limitations of Synthetic Data

Since we use synthetic (not FEM-optimized) data:
- Generated topologies won't be truly optimal
- Patterns may not satisfy physics constraints
- For real applications, use FEM-generated training data

---

## 6. Extensions and Next Steps

1. **Real FEM Data**: Generate training data using SIMP optimization
2. **Higher Resolution**: Scale to 64×64 or 128×128
3. **Physics-Informed Loss**: Add compliance evaluation to training loss
4. **3D Topology**: Extend to volumetric designs
5. **Multi-load Cases**: Condition on multiple load scenarios
6. **Manufacturing Constraints**: Add constraints for minimum member size, symmetry

---

## 7. Key Takeaways

1. **Topology optimization** is a structural CAE method for finding optimal material layouts
2. **TopoDiff** uses diffusion models to generate designs instantly (vs. iterative FEM)
3. **Diffusion models** learn to reverse a noising process, enabling diverse design generation
4. **Conditioning** on boundary conditions ensures generated designs respect the problem setup
5. **SIMP penalization** (E = x^p × E_0) is the physics behind traditional topology optimization
