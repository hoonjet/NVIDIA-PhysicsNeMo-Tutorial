# FNO — 2D Wave Equation (Electromagnetics/Acoustics)

> **Category**: `neural_operators/fno/wave/` — 2nd-order time-dependent PDE  
> **Paradigm**: Wave propagation, reflection, and interference  
> **Model**: 2D FNO with 2-channel input (for 2nd-order time derivative)

---

## 1. What Makes This Tutorial Unique?

This is the **ONLY tutorial covering a 2nd-order time derivative PDE**. Every other time-dependent tutorial uses 1st-order:

| Tutorial | Time Derivative | Order |
|----------|:--------------:|:-----:|
| Navier-Stokes | `u_t` | 1st |
| Heat Conduction | `u_t` | 1st |
| Burgers | `u_t` | 1st |
| Allen-Cahn | `u_t` | 1st |
| **Wave Equation** | **`u_tt`** | **2nd** |

The 2nd-order time derivative creates fundamentally different dynamics:
- **Oscillation** (not decay or growth)
- **Energy conservation** (no dissipation)
- **Wave propagation** (disturbance travels at speed c)
- **Reflection** (waves bounce off boundaries)
- **Interference** (multiple waves superpose)

### Key Architectural Difference
- 1st-order PDE: input = `[u(t)]` → predict `u(t+1)` (1 channel)
- **2nd-order PDE**: input = `[u(t), u(t-1)]` → predict `u(t+1)` (**2 channels**)

This is because the 2nd-order derivative `u_tt` requires two time levels to compute.

---

## 2. The FNO Original Paper

> **Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A.**  
> "Fourier Neural Operator for Parametric Partial Differential Equations."  
> *International Conference on Learning Representations (ICLR), 2021.*  
> arXiv: https://arxiv.org/abs/2010.08895

This paper introduced the **Fourier Neural Operator (FNO)**. While the paper's three benchmarks were Darcy, Navier-Stokes, and Allen-Cahn, the wave equation is a natural extension that demonstrates FNO's ability to handle:

1. **2nd-order time derivatives** (not in the original paper's benchmarks)
2. **Oscillatory solutions** (high-frequency content in Fourier domain)
3. **Energy conservation** (no dissipation to dampen errors)

The wave equation is a standard test problem in computational physics and is closely related to the Helmholtz equation (frequency-domain wave equation).

### Why This Paper Matters

- **Foundation**: This is THE paper that introduced FNO. All FNO tutorials in this repo derive from it.
- **PhysicsNeMo**: NVIDIA's PhysicsNeMo implements FNO based on this paper's architecture.
- **Extension**: While the paper didn't use the wave equation as a benchmark, it demonstrates FNO's generality for any PDE with a Fourier-representable solution.

---

## 3. Problem: 2D Wave Equation

```
u_tt = c²·(u_xx + u_yy)

Domain: [0, 1] × [0, 1]
Boundary: u = 0 (Dirichlet → wave reflection)
Initial: Gaussian pulse at random location, u_t = 0 (at rest)
c = 1.0 (wave speed)
```

### 3.1 Physical Meaning

The wave equation describes **propagation of disturbances** through a medium:

- **Electromagnetics**: Maxwell's equations reduce to the wave equation for E and B fields
- **Acoustics**: Sound pressure waves in air/water
- **Seismology**: Earthquake waves (P-waves, S-waves)
- **Optics**: Light propagation in free space
- **Vibrations**: String, membrane, and plate vibrations

### 3.2 2nd-Order Time Derivative

The key difference from all other tutorials:

**1st-order** (Allen-Cahn, Heat, NS, Burgers):
```
u_t = f(u)     →  u(t+1) = u(t) + dt·f(u)
Input: [u(t)] → predict u(t+1)    (1 channel)
```

**2nd-order** (Wave):
```
u_tt = f(u)    →  u(t+1) = 2·u(t) - u(t-1) + dt²·f(u)
Input: [u(t), u(t-1)] → predict u(t+1)    (2 channels)
```

The 2nd-order derivative means:
- The system **remembers** its previous state (needs 2 time levels)
- Solutions **oscillate** (sinusoidal in time)
- Energy is **conserved** (no damping)

### 3.3 Wave Phenomena Visible in This Tutorial

```
t=0:   Gaussian pulse (compact disturbance)
        ↓
t=5:   Circular wavefront expands outward
        ↓
t=10:  Wave reaches boundary
        ↓
t=15:  Reflection from boundaries (wave bounces back)
        ↓
t=25:  Multiple reflections create interference patterns
        ↓
t=35:  Complex interference (standing wave-like patterns)
```

---

## 4. Method: 2-Channel FNO with Auto-Regressive Rollout

### 4.1 Input Format
```
Channel 1: u(x, y, t)      (current state)
Channel 2: u(x, y, t-1)    (previous state)
           ↓
         FNO 2D
           ↓
Output:   u(x, y, t+1)     (next state)
```

### 4.2 Multi-Step Rollout
```
u(0), u(1) → FNO → u(2) → FNO → u(3) → FNO → ... → u(T)
     ↑                    ↑
     feed back as u(t-1)  feed back as u(t)
```

### 4.3 Error Accumulation
- One-step error: small (teacher-forced)
- Rollout error: grows with steps (compounding)
- Wave equation is especially challenging: no dissipation to dampen errors

---

## 5. Code Structure

| Section | Description |
|---------|-------------|
| `[1] Data Generation` | 2D FDM with CFL-stable time stepping, Gaussian pulse IC |
| `[2] FNO 2D` | 2-channel input, 4 spectral layers, 12 Fourier modes |
| `[3] Training` | 250 epochs, 2-channel input [u(t), u(t-1)] → u(t+1) |
| `[4] Rollout` | Auto-regressive from u(0), u(1), error analysis |
| `[5] Visualization` | Loss, propagation snapshots, GT vs FNO, rollout error, explanation |

---

## 6. Key Results

### 6.1 Wave Propagation (Visible)
- t=0: Compact Gaussian pulse
- t=5-10: Circular wavefront expands
- t=15+: Reflection from boundaries
- t=25+: Interference patterns

### 6.2 FNO Captures Wave Dynamics
- Ground truth and FNO rollout show matching wavefronts
- Error concentrates at wavefronts (sharp gradients)
- FNO maintains reasonable accuracy for ~15-20 steps

### 6.3 Error Accumulation
- One-step error: low and stable
- Rollout error: grows steadily (no dissipation to dampen)
- Wave equation is harder than dissipative PDEs (Heat, Allen-Cahn)

---

## 7. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\neural_operators\fno\wave
python wave.py
```

### Result Files
| File | Description |
|------|-------------|
| `results/wave_loss.png` | Training & test loss |
| `results/wave_propagation.png` | Wave propagation snapshots (t=0 to t=35) |
| `results/wave_result.png` | Ground truth vs FNO vs error (4 time steps) |
| `results/wave_rollout_error.png` | Rollout error accumulation |
| `results/wave_explanation.png` | Equation explanation (2nd-order, wave phenomena) |

---

## 8. Key Concepts Learned

1. **2nd-Order Time Derivative**: The wave equation uses `u_tt` (not `u_t`). This requires two previous time steps as input and creates oscillatory dynamics. All other tutorials use 1st-order.

2. **2-Channel Input**: For 2nd-order PDEs, FNO needs `[u(t), u(t-1)]` as input (2 channels). This is a fundamental architectural difference from 1st-order PDE tutorials.

3. **Wave Propagation**: An initial disturbance travels at speed c in all directions. The wavefront is circular (in 2D with uniform medium).

4. **Reflection**: Dirichlet boundaries (u=0) reflect waves. The reflected wave has opposite sign (phase inversion). Multiple reflections create complex interference.

5. **Energy Conservation**: Unlike dissipative PDEs (Heat, Allen-Cahn), the wave equation conserves energy. This means errors don't decay — they persist and accumulate.

6. **CFL Condition**: The data generation uses FDM with CFL stability condition: `c·dt/dx ≤ 1/√2` (2D). This ensures numerical stability.

---

## 9. Comparison with Other Tutorials

| Feature | Allen-Cahn | Navier-Stokes | Heat | **Wave** |
|---------|:----------:|:-------------:|:----:|:--------:|
| **Time order** | 1st | 1st | 1st | **2nd** |
| **Input channels** | 1 | 1 | 1 | **2** |
| **Dynamics** | Dissipative | Advection | Dissipative | **Oscillatory** |
| **Energy** | Decreases | Decreases | Decreases | **Conserved** |
| **Reflection** | ✗ | ✗ | ✗ | **✓** |
| **Interference** | ✗ | ✗ | ✗ | **✓** |
| **Spatial dim** | 1D | 2D | 2D | **2D** |

### vs. Allen-Cahn (closest time-dependent tutorial)
- Allen-Cahn: 1st-order, 1D, dissipative (phase separation), 1-channel input
- Wave: 2nd-order, 2D, conservative (oscillation), 2-channel input
- Allen-Cahn errors decay (dissipation); Wave errors persist (conservation)

---

## 10. Extensions

- **Absorbing boundaries**: Use PML (Perfectly Matched Layer) to prevent reflection
- **Variable wave speed**: c(x,y) — heterogeneous medium (seismology)
- **3D wave equation**: Extend to 3D (more memory intensive)
- **Helmholtz equation**: Frequency-domain wave equation (steady-state)
- **Multi-source**: Multiple Gaussian pulses (interference from start)
- **Different boundary types**: Neumann (free boundary → no phase inversion)

---

## 11. References

1. **Li, Z., et al.** "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021.*  
   arXiv: https://arxiv.org/abs/2010.08895  
   *(The original FNO paper. Foundation for all FNO tutorials in this repo.)*

2. **Kovachki, N., et al.** "Neural Operator: Learning Maps Between Function Spaces." arXiv:2108.08481, 2021.  
   *(Comprehensive Neural Operator survey.)*

3. **Courant, R., Friedrichs, K., & Lewy, H.** "Über die partiellen Differenzengleichungen der mathematischen Physik." *Mathematische Annalen, 100(1), 32-74, 1928.*  
   *(The CFL condition — stability requirement for wave equation FDM.)*

4. **Taflove, A. & Hagness, S.** "Computational Electrodynamics: The Finite-Difference Time-Domain Method." *Artech House, 2005.*  
   *(FDTD method for Maxwell's equations — the wave equation in electromagnetics.)*
