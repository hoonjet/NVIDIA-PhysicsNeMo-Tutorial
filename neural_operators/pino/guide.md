# Physics-Informed Neural Operator (PINO)

> **Category**: `neural_operators/pino/` — Hybrid Data + Physics  
> **Paradigm**: FNO architecture + PDE residual loss  
> **Model**: Fourier Neural Operator with hybrid loss

---

## 1. What Makes This Tutorial Unique?

This is the **only tutorial that combines both paradigms**: Neural Operator architecture (FNO) + Physics-Informed loss (PDE residual). Every other tutorial is EITHER data-driven OR equation-based — never both.

| Aspect | PINN | FNO / U-Net | **PINO** |
|--------|:----:|:-----------:|:--------:|
| **Architecture** | MLP | Neural Operator | **Neural Operator** |
| **Loss** | PDE residual only | Data MSE only | **Data MSE + PDE residual** |
| **Needs data?** | No | Yes | **Less data needed** |
| **Spectral processing** | ✗ | ✓ | **✓** |
| **Physics-constrained** | ✓ | ✗ | **✓** |

### vs. FNO (Data-Only)
- FNO learns purely from data — no physics constraint
- PINO adds PDE residual loss → predictions must satisfy physics laws
- Result: better accuracy, lower PDE residual, works with less data

### vs. PINN (Equation-Only)
- PINN uses MLP architecture — no spectral processing, slow for high-dim
- PINO uses FNO's Fourier layers — spectral efficiency, global receptive field
- Result: faster convergence, better for 2D/3D PDEs

---

## 2. Problem: 2D Darcy Flow

```
-∇·(k∇p) = 1,   p = 0 on boundary

Input:  Permeability k(x)  [32×32]
Output: Pressure p(x)  [32×32]
```

### Two Training Modes Compared
1. **Pure Data FNO**: `L = MSE(pred, true)` — standard supervised learning
2. **PINO**: `L = MSE(pred, true) + λ · PDE_residual²` — hybrid

---

## 3. Method: Hybrid Loss

### 3.1 Data Loss (Supervised)
```
L_data = MSE(model(k), p_true)
```
Standard supervised loss — same as FNO.

### 3.2 Physics Loss (PDE Residual)
```
residual = -∇·(k∇p_pred) - 1
L_phys = mean(residual²)
```
The PDE residual is computed via differentiable finite differences on the model's output. This is the "Physics-Informed" component — the model must produce predictions that satisfy the Darcy equation.

### 3.3 Hybrid Loss
```
L = L_data + λ · L_phys
```
- `λ` (lambda) controls the physics constraint strength
- Too small: physics has no effect (degrades to pure FNO)
- Too large: physics dominates, data fitting suffers
- This tutorial uses λ = 0.1

### 3.4 Why PINO Works Better
- **Data loss** ensures predictions match ground truth
- **Physics loss** ensures predictions satisfy PDE laws
- Together: predictions are both accurate AND physically consistent
- With less data, physics loss compensates for missing data

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| `[1] Data Generation` | Darcy flow (spectral permeability + FD solver) |
| `[2] FNO Architecture` | SpectralConv2d + Fourier layers |
| `[3] PDE Residual` | Differentiable Darcy residual via finite differences |
| `[4] Training` | Pure Data FNO vs PINO (200 epochs each) |
| `[5] Evaluation` | L2 error + PDE residual comparison |
| `[6] Low-Data Test` | 50% less data — PINO's key advantage |
| `[7] Visualization` | Loss, physics loss, prediction, PDE residual, low-data bar chart |

---

## 5. Key Results

### 5.1 Full Data (200 samples)
- PINO achieves **lower L2 error** than pure data FNO
- PINO's **PDE residual is much lower** — predictions are more physically consistent

### 5.2 Low Data (100 samples, 50% less)
- PINO's advantage **grows** when data is scarce
- Physics loss compensates for missing data
- This is PINO's key practical advantage

### 5.3 Physics Loss During Training
- PINO's PDE residual decreases during training
- Pure data FNO has no physics constraint → higher PDE residual

---

## 6. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\neural_operators\pino
python pino_darcy.py
```

### Result Files
| File | Description |
|------|-------------|
| `results/pino_loss.png` | Training/test loss: Data FNO vs PINO |
| `results/pino_physics_loss.png` | PINO PDE residual during training |
| `results/pino_result.png` | Prediction + error + PDE residual comparison |
| `results/pino_lowdata.png` | Bar chart: full data vs low data |

---

## 7. Key Concepts Learned

1. **Hybrid Loss**: `L = L_data + λ · L_phys`. The same model architecture (FNO) can be trained with or without physics. Adding physics is just adding a loss term.

2. **Differentiable Physics**: The PDE residual is computed via differentiable finite differences. Gradients flow through the residual back to the model weights — no adjoint method needed.

3. **Data Efficiency**: Physics loss provides "free" supervision signal. When data is expensive (CFD simulations), PINO needs fewer samples to achieve the same accuracy.

4. **Physics Consistency**: Even when data FNO's L2 error is acceptable, its PDE residual can be high (predictions violate physics). PINO ensures both accuracy AND consistency.

5. **Bridge Paradigm**: PINO connects PINN (equation-based) and Neural Operator (data-driven). It shows that these are not competing approaches but complementary components of a single framework.

6. **Lambda Tuning**: The physics weight λ is a hyperparameter. Too small → no physics effect. Too large → physics dominates, data fitting suffers. λ = 0.1 is a good starting point.

---

## 8. Comparison with Other Tutorials

| Feature | PINN (Burgers) | FNO (Darcy) | **PINO** |
|---------|:--------------:|:-----------:|:--------:|
| **Architecture** | MLP | Fourier layers | **Fourier layers** |
| **Loss** | PDE residual | Data MSE | **Data MSE + PDE residual** |
| **Data needed** | 0 | 200+ | **100+ (less)** |
| **Spectral** | ✗ | ✓ | **✓** |
| **Physics** | ✓ | ✗ | **✓** |
| **Best of both** | ✗ | ✗ | **✓** |

---

## 9. Extensions

- **Pure physics PINO**: Set λ very high, remove data loss entirely → FNO trained like PINN
- **Multi-physics PINO**: Add multiple PDE residuals (e.g., Navier-Stokes: momentum + continuity)
- **Adaptive λ**: Schedule λ during training (start low, increase gradually)
- **PINO for inverse problems**: Use PDE residual to estimate unknown parameters
- **3D PINO**: Extend to 3D Fourier layers for volumetric PDEs

---

## 10. References

- Li et al., "Physics-Informed Neural Operator for Learning PDEs," arXiv 2111.03794, 2021
- Li et al., "Fourier Neural Operator for Parametric PDEs," ICLR 2021
- Raissi et al., "Physics-Informed Neural Networks," JCP 2019
- Lu et al., "DeepONet: Learning Nonlinear Operators for PDEs," arXiv 1910.03193, 2020
