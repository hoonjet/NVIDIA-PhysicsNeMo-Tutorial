# PINN Transfer Learning: Burgers → Sine-Gordon

> **Category**: `transfer_learning/` — Cross-PDE knowledge transfer
> **Paradigm**: Transfer network weights between different PDEs
> **Model**: FC network (shared architecture, different loss functions)

---

## 1. What Makes This Tutorial Unique?

| Aspect | FNO Transfer (existing) | PINN Transfer (THIS) |
|--------|------------------------|---------------------|
| **PDE** | Same (Darcy), different params | **Different** (Burgers → Sine-Gordon) |
| **Learning** | Data-driven (MSE) | Equation-based (PDE residual) |
| **What transfers** | Encoder weights | Network weights |
| **Loss function** | Same (MSE) | **Changes** (Burgers residual → Sine-Gordon residual) |
| **PDE type** | Same (elliptic) | Different (hyperbolic → nonlinear wave) |

### Key Difference: Loss Function Changes
- **FNO Transfer**: Same loss (MSE), same PDE, different parameters
- **PINN Transfer**: Loss function itself changes — Burgers residual → Sine-Gordon residual
- This is fundamentally harder: the network must adapt to a different equation

---

## 2. Source & Target PDEs

### Source: Burgers Equation (hyperbolic, dissipative)
```
u_t + u·u_x = ν·u_xx
```
- 1st order in time
- Dissipative (shock waves form)
- Domain: x∈[-1,1], t∈[0,1]

### Target: Sine-Gordon Equation (nonlinear wave, soliton)
```
u_tt - u_xx + sin(u) = 0
```
- 2nd order in time
- Conservative (solitons)
- Domain: x∈[-5,5], t∈[0,5]
- Analytical solution: kink soliton u(x,t) = 4·atan(exp(x))

---

## 3. Three Transfer Strategies

| Strategy | Init | Frozen | Trainable | LR |
|----------|------|--------|-----------|-----|
| **Scratch** | Random | None | All | 1e-3 |
| **Freeze** | Burgers | First 3 layers | Last 2 + output | 1e-3 |
| **Full FT** | Burgers | None | All | 5e-4 |

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| [1] Problem | Source (Burgers) → Target (Sine-Gordon) |
| [2] Network | FC(2→50→50→50→50→50→1), Tanh |
| [3] Source PDE | Burgers residual + BC/IC |
| [4] Target PDE | Sine-Gordon residual + BC/IC (kink soliton) |
| [5] Phase 1 | Train on Burgers (3000 epochs) |
| [6] Phase 2 | Transfer to Sine-Gordon (3 strategies, 3000 epochs each) |
| [7] Visualization | Loss, solutions, summary, concept |
| [8] Summary | Strategy comparison |

---

## 5. How to Run

```cmd
cd E:\physicsnemo-tutorials\transfer_learning\pinn_transfer
python pinn_transfer.py
```

Results saved to `results/`:
- `pinn_transfer_source_loss.png` — Burgers training loss
- `pinn_transfer_target_loss.png` — Sine-Gordon loss (3 strategies)
- `pinn_transfer_solutions.png` — Kink soliton predictions
- `pinn_transfer_summary.png` — Final loss & time comparison
- `pinn_transfer_explanation.png` — Concept comparison with FNO transfer

---

## 6. vs. FNO Transfer (Existing Tutorial)

| Feature | FNO Transfer | PINN Transfer |
|---------|-------------|---------------|
| **PDE** | Same (Darcy) | Different (Burgers → Sine-Gordon) |
| **Loss** | Same (MSE) | Changes (PDE residual) |
| **What transfers** | Encoder features | Network weights |
| **Adaptation** | Fine-tune decoder | Fine-tune all/freeze layers |
| **Difficulty** | Easier (same PDE) | Harder (different PDE) |

---

## 7. References

- Raissi et al., "Physics-Informed Neural Networks" (2019)
- Transfer learning: Pan & Yang, "A Survey on Transfer Learning" (2010)
- Sine-Gordon solitons: Drazin, "Solitons: an Introduction" (1989)
