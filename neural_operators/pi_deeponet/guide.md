# PI-DeepONet (Physics-Informed DeepONet)

> **Category**: `neural_operators/` — Operator learning + physics constraint
> **Paradigm**: DeepONet + PDE residual loss
> **Problem**: 1D Burgers equation operator learning (u₀ → u)

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing DeepONet | Existing PINO | **THIS (PI-DeepONet)** |
|--------|-------------------|---------------|------------------------|
| **Backbone** | DeepONet (branch-trunk) | FNO | **DeepONet** |
| **Loss** | Data-only | Data + PDE | **Data + PDE** |
| **Grid** | Sensor-based (irregular) | Grid (regular) | **Sensor-based** |
| **Resolution** | Independent | Fixed | **Independent** |
| **Data need** | High | Medium | **Low (PDE helps)** |

### Key Difference: Physics + Operator Learning
- **DeepONet** (existing): learns operator from data only — needs lots of data, may violate PDE
- **PINO** (existing): FNO + PDE — grid-based, fixed resolution
- **PI-DeepONet** (this): DeepONet + PDE — sensor-based, resolution-independent, less data needed

---

## 2. Physics: 1D Burgers Equation

```
∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

IC: u(x, 0) = u₀(x)  (varies — operator input)
BC: u(-1, t) = u(1, t) = 0
```

**Operator**: G: u₀(x) → u(x, t=T)

---

## 3. Architecture

```
Branch (encodes IC):              Trunk (encodes query):
  u₀(sensors) [50]                  (x, t) [2]
       ↓                               ↓
  Linear(50, 64) + Tanh           Linear(2, 64) + Tanh
  Linear(64, 64) + Tanh           Linear(64, 64) + Tanh
  Linear(64, 32)                  Linear(64, 32)
       ↓                               ↓
       b [32]                          t [32]
       └────────── · ──────────────────┘
                     ↓
              u(x, t) = Σ_k b_k · t_k
```

---

## 4. Loss Functions

| Loss | Equation | Weight |
|------|---------|--------|
| **Data** | MSE(u_pred, u_true) at t=T | λ=1.0 |
| **PDE** | ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0 | λ=0.5 |

### PDE Residual via Autograd
```
u_x = ∂u/∂x  (autograd)
u_t = ∂u/∂t  (autograd)
u_xx = ∂²u/∂x²  (autograd, 2nd order)

residual = u_t + u·u_x - ν·u_xx
```

---

## 5. Comparison: Data-only vs PI-DeepONet

Both models have identical architecture. The only difference is the loss:
- **Data-only**: `Loss = MSE(u_pred, u_true)`
- **PI-DeepONet**: `Loss = λ_data·MSE + λ_pde·|PDE|²`

With only 200 training samples, PI-DeepONet should outperform data-only.

---

## 6. How to Run

```cmd
cd E:\physicsnemo-tutorials\neural_operators\pi_deeponet
python pi_deeponet.py
```

Results:
- `pi_deeponet_prediction.png` — Prediction comparison (2 test samples)
- `pi_deeponet_loss.png` — Training loss comparison
- `pi_deeponet_concept.png` — Architecture + error bar chart

---

## 7. References

- Lu et al., "DeepONet: Learning nonlinear operators" (2021)
- Wang et al., "When and why PINN fail to train" (2022)
- Li et al., "Physics-Informed DeepONet for PDEs" (2023)
