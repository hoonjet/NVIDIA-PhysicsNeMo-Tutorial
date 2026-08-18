# FNO — Allen-Cahn Equation (Phase Separation)

> **Category**: `neural_operators/fno/allen_cahn/` — Nonlinear Reaction-Diffusion  
> **Paradigm**: Auto-regressive time-dependent PDE prediction  
> **Model**: 1D FNO with multi-step rollout

---

## 1. What Makes This Tutorial Unique?

This tutorial covers the **Allen-Cahn equation** — the **third benchmark PDE** from the original FNO paper. The FNO paper (Li et al., ICLR 2021) used three PDEs to validate FNO:

| Benchmark PDE | Equation | In This Repo? |
|---------------|----------|:-------------:|
| Darcy Flow | `-∇·(k∇p) = f` | ✅ (6 tutorials) |
| Navier-Stokes | `ω_t + u·∇ω = ν∇²ω` | ✅ (1 tutorial) |
| **Allen-Cahn** | `u_t = ε²·u_xx + u - u³` | **✅ This tutorial** |

### vs. All Other Tutorials

| Feature | Darcy | Navier-Stokes | Heat | Burgers | **Allen-Cahn** |
|---------|:------:|:-------------:|:----:|:-------:|:--------------:|
| **Time-dependent** | ✗ | ✓ | ✓ | ✓ | **✓** |
| **Nonlinear** | ✗ | ✓ (advection) | ✗ | ✓ (advection) | **✓ (reaction)** |
| **Reaction term** | ✗ | ✗ | ✗ | ✗ | **✓ (u - u³)** |
| **Phase separation** | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Stiff (sharp interfaces)** | ✗ | ✗ | ✗ | Partial | **✓** |

The key differentiator is the **nonlinear reaction term** `u - u³`, which creates **phase separation**: the solution evolves from random noise into structured domains of +1 and -1. No other tutorial covers this phenomenon.

---

## 2. The FNO Original Paper

> **Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A.**  
> "Fourier Neural Operator for Parametric Partial Differential Equations."  
> *International Conference on Learning Representations (ICLR), 2021.*  
> arXiv: https://arxiv.org/abs/2010.08895

This paper introduced the **Fourier Neural Operator (FNO)**, the foundation of all FNO tutorials in this repository. The paper validated FNO on three benchmark PDEs:

1. **Burgers Equation** (Section 5.1): 1D nonlinear advection-diffusion
2. **Darcy Flow** (Section 5.2): 2D steady-state elliptic PDE
3. **Allen-Cahn Equation** (Section 5.3): 1D nonlinear reaction-diffusion

The Allen-Cahn benchmark in the paper used:
- Equation: `u_t = 0.0001·u_xx + 5·u³ - 5·u` (equivalent form)
- Grid: 1024 spatial points
- Task: Auto-regressive prediction from t=0 to t=T

This tutorial implements the same equation with slightly different parameters (ε=0.01) suitable for the Quadro P4000 GPU.

### Why This Paper Matters

- **Foundation**: This is THE paper that introduced FNO. All FNO tutorials in this repo derive from it.
- **PhysicsNeMo**: NVIDIA's PhysicsNeMo implements FNO based on this paper's architecture.
- **Benchmarks**: The three PDEs (Darcy, NS, Allen-Cahn) are the standard benchmarks for Neural Operator research.
- **Completeness**: With this tutorial, all three FNO paper benchmarks are now covered in this repository.

---

## 3. Problem: Allen-Cahn Equation

```
u_t = ε²·u_xx + u - u³

Domain: x ∈ [0, 1], periodic boundary conditions
ε = 0.01 (small → sharp interfaces)
```

### 3.1 Physical Meaning

The Allen-Cahn equation models **phase separation** — a process where two phases (states) of a material separate from a mixed state:

- **Alloy solidification**: Two metals in a molten state separate as they solidify
- **Binary fluid**: Two immiscible liquids (oil/water) demix over time
- **Pattern formation**: Random initial noise evolves into structured domains

### 3.2 The Two Terms

**Diffusion term** (`ε²·u_xx`):
- Smooths interfaces between phases
- Small ε → sharp interfaces (steep gradients)
- This is the "stiff" part requiring small time steps

**Reaction term** (`u - u³`):
- Nonlinear: drives u toward +1 or -1
- u=0 is **unstable** (any perturbation grows)
- u=±1 are **stable equilibria**
- Creates phase separation: regions converge to +1 or -1

### 3.3 Phase Separation Process

```
t=0:  Random noise around u=0 (unstable)
       ↓
t=5:  Domains start forming (+1 and -1 regions)
       ↓
t=10: Sharp interfaces between domains
       ↓
t=20: Coarsening — small domains merge into larger ones
```

---

## 4. Method: Auto-Regressive FNO

### 4.1 One-Step Prediction
```
Input:  u(x, t)     [128 points]
Output: u(x, t+1)   [128 points]
```
FNO learns the mapping `u(t) → u(t+1)`. Training uses all consecutive pairs.

### 4.2 Multi-Step Rollout
```
u(0) → FNO → u(1) → FNO → u(2) → FNO → ... → u(T)
```
Feed each prediction back as input. This is **auto-regressive** prediction.

### 4.3 Error Accumulation
- One-step error: small (teacher-forced, always uses true input)
- Rollout error: grows with steps (each prediction's error feeds into next step)
- This is the key challenge of auto-regressive prediction

---

## 5. Code Structure

| Section | Description |
|---------|-------------|
| `[1] Data Generation` | ETDRK4 spectral method for Allen-Cahn (stiff PDE) |
| `[2] FNO 1D` | 1D spectral convolution + Fourier layers |
| `[3] Training` | Auto-regressive: u(t)→u(t+1), 300 epochs |
| `[4] Rollout` | Multi-step auto-regressive prediction + error analysis |
| `[5] Visualization` | Loss, phase separation, space-time, error accumulation, patterns, equation explanation |

---

## 6. Key Results

### 6.1 Phase Separation (Visible in Space-Time)
- Ground truth shows clear +1/-1 domain formation
- FNO rollout prediction captures the domain structure
- Error concentrates at interfaces (sharp gradients)

### 6.2 Error Accumulation
- One-step error (teacher-forced): low and stable
- Rollout error: grows with steps (compounding)
- FNO maintains reasonable accuracy for ~10-15 steps, then degrades

### 6.3 Diverse Patterns
- Each sample produces a unique phase separation pattern
- FNO generalizes across different initial conditions

---

## 7. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\neural_operators\fno\allen_cahn
python allen_cahn.py
```

### Result Files
| File | Description |
|------|-------------|
| `results/allen_cahn_loss.png` | Training & test loss |
| `results/allen_cahn_phase.png` | Phase separation at t=0, 10, 20 |
| `results/allen_cahn_result.png` | Space-time: ground truth vs FNO + error + rollout error |
| `results/allen_cahn_patterns.png` | 6 diverse phase separation patterns |
| `results/allen_cahn_explanation.png` | Equation explanation (diffusion + reaction terms) |

---

## 8. Key Concepts Learned

1. **Nonlinear Reaction-Diffusion**: The `u - u³` term is fundamentally different from advection (Navier-Stokes) or pure diffusion (Heat). It creates two stable equilibria and drives phase separation.

2. **Phase Separation**: Random initial noise evolves into structured +1/-1 domains. This is a universal phenomenon in materials science, fluid dynamics, and biology.

3. **Stiff PDE**: Small ε creates sharp interfaces (steep gradients). The data generation uses ETDRK4 (Exponential Time Differencing) for stability — a specialized method for stiff PDEs.

4. **Auto-Regressive Prediction**: FNO learns one-step mapping `u(t)→u(t+1)`. Multi-step prediction is done by feeding predictions back (rollout). Error accumulates with each step.

5. **Error Accumulation**: The key challenge of auto-regressive prediction. One-step error is small, but rollout error grows because each prediction's error feeds into the next step's input.

6. **FNO Paper Benchmark**: This completes the three benchmark PDEs from Li et al. (ICLR 2021). Darcy, Navier-Stokes, and Allen-Cahn are now all covered.

---

## 9. Comparison with Other Tutorials

| Feature | FNO-NS | FNO-Heat | PINN-Burgers | PINN-ReactionDiff | **This Tutorial** |
|---------|:------:|:--------:|:------------:|:-----------------:|:-----------------:|
| **PDE type** | Advection-diffusion | Pure diffusion | Advection-diffusion | Reaction-diffusion | **Reaction-diffusion** |
| **Nonlinearity** | Linear (vorticity) | Linear | Nonlinear (u·u_x) | Nonlinear | **Nonlinear (u³)** |
| **Phase separation** | ✗ | ✗ | ✗ | ✓ (Gray-Scott) | **✓ (Allen-Cahn)** |
| **Architecture** | FNO 2D | FNO 2D | PINN (MLP) | PINN (MLP) | **FNO 1D** |
| **Auto-regressive** | ✓ | ✓ | ✗ | ✗ | **✓** |
| **Rollout analysis** | ✗ | ✗ | ✗ | ✗ | **✓** |

### vs. PINN Reaction-Diffusion (Gray-Scott)
- PINN-RD uses MLP architecture (point-by-point, no spectral processing)
- This tutorial uses FNO (spectral, global receptive field)
- PINN-RD solves the PDE directly (equation-based); this tutorial learns from data (data-driven)
- PINN-RD doesn't do auto-regressive rollout; this tutorial analyzes error accumulation

---

## 10. Extensions

- **2D Allen-Cahn**: Extend to 2D spatial domain (more complex patterns)
- **Different ε**: Larger ε → smoother interfaces (easier); smaller ε → sharper (harder)
- **Pushforward trick**: Train with multi-step loss to reduce rollout error
- **Noise injection**: Add noise to FNO input during training for robust rollout
- **Long-time integration**: More time steps (100+) to test FNO's long-range prediction

---

## 11. References

1. **Li, Z., et al.** "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021.*  
   arXiv: https://arxiv.org/abs/2010.08895  
   *(The original FNO paper. Allen-Cahn is one of three benchmark PDEs.)*

2. **Allen, S.M. & Cahn, J.W.** "A microscopic theory for antiphase boundary motion and its application to antiphase domain coarsening." *Acta Metallurgica, 27(6), 1085-1095, 1979.*  
   *(The original Allen-Cahn equation paper.)*

3. **Cox, S.M. & Matthews, P.C.** "Exponential time differencing for stiff systems." *Journal of Computational Physics, 176(2), 430-455, 2002.*  
   *(ETDRK4 method used for data generation in this tutorial.)*

4. **Kovachki, N., et al.** "Neural Operator: Learning Maps Between Function Spaces." arXiv:2108.08481, 2021.  
   *(Comprehensive Neural Operator survey.)*
