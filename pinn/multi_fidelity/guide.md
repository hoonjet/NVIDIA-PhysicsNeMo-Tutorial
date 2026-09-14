# Multi-Fidelity PINN

> **Category**: `pinn/` — Multi-fidelity data fusion
> **Paradigm**: Low-fidelity net + Correction net (simultaneous training)
> **Problem**: 2D Poisson with mixed-fidelity data sources

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing Transfer Learning | **THIS (Multi-Fidelity)** |
|--------|---------------------------|--------------------------|
| **Training** | Sequential (pre-train → fine-tune) | **Simultaneous (2 nets)** |
| **Grid** | Same grid, same PDE | **Different fidelity (16×16 vs 64×64)** |
| **Correction** | None | **Correction net (learns gap)** |
| **Data** | Single fidelity | **Multi-fidelity (500 low + 20 high)** |
| **Goal** | Adapt to new PDE | **Fuse cheap + expensive data** |

### Key Difference: Multi-Fidelity Data Fusion
- **Transfer learning**: sequential, same grid, no correction
- **Multi-Fidelity PINN**: simultaneous, different grids, correction net bridges the gap

---

## 2. Physics: 2D Poisson Equation

```
∇²u = f(x, y)  on [0,1]×[0,1]
BC: u = 0 on boundary

f(x,y) = A · sin(k₁·π·x) · sin(k₂·π·y)
```

**Parameter family**: (A, k₁, k₂) varies → generates solution family

---

## 3. Architecture

```
         Low-Fidelity Net                    Correction Net
         
  (A, k₁, k₂) [3]                      (A, k₁, k₂) [3]
       ↓                                       ↓
  Linear(3, 64) + Tanh                  Linear(3, 128) + Tanh
  Linear(64, 64) + Tanh                 Linear(128, 128) + Tanh
  Linear(64, 256)                       Linear(128, 4096)
       ↓                                       ↓
  u_low [16×16]                         correction [64×64]
       ↓                                       ↓
  Upsample (bilinear)                          ↓
       ↓                                       ↓
  u_low_up [64×64]  ───────── + ───────  correction [64×64]
                                ↓
                         u_high [64×64]
```

**Key idea**: `u_high = Upsample(u_low) + correction`

---

## 4. Data Strategy

| Fidelity | Grid | Samples | Cost | Purpose |
|----------|------|---------|------|---------|
| Low | 16×16 (256) | 500 | Cheap | Learn general trend |
| High | 64×64 (4096) | 20 | Expensive | Learn fine details |

---

## 5. Loss Functions

| Loss | Equation | Weight |
|------|---------|--------|
| **Low-fidelity** | MSE(u_low_pred, u_low_data) | λ=1.0 |
| **High-fidelity** | MSE(u_high_pred, u_high_data) | λ=5.0 |

---

## 6. How to Run

```cmd
cd E:\physicsnemo-tutorials\pinn\multi_fidelity
python multi_fidelity.py
```

Results:
- `mf_prediction.png` — True vs prediction vs error (4 test samples)
- `mf_decomposition.png` — Low-fidelity + correction decomposition
- `mf_loss.png` — Training loss (total, low, high)
- `mf_concept.png` — Architecture + cost vs accuracy

---

## 7. Why Multi-Fidelity Matters

1. **Cost reduction**: 500 cheap + 20 expensive ≈ 520 total (vs 500 expensive)
2. **Accuracy**: Better than low-only, comparable to high-only with less data
3. **Engineering**: Real-world has mixed-fidelity data (coarse CFD + fine experiments)
4. **Scalability**: Add more fidelity levels (low → medium → high)
5. **Digital twins**: Real-time low-fid + periodic high-fid calibration

---

## 8. References

- Perdikaris et al., "Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling" (2017)
- Meng & Karniadakis, "A composite neural network for high-dimensional PDEs" (2020)
- Howard et al., "Multi-fidelity modeling for physical systems" (2023)
