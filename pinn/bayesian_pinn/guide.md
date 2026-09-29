# Bayesian PINN

> **Category**: `pinn/` — Uncertainty-aware PINN
> **Problem**: 1D Burgers equation with epistemic uncertainty estimation

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing PINNs | Existing UQ (Ensemble/MC) | **THIS (Bayesian PINN)** |
|--------|----------------|--------------------------|--------------------------|
| **Physics** | PDE residual | None (data-driven) | **PDE residual + UQ** |
| **Uncertainty** | None | Yes (data-based) | **Yes (physics-aware)** |
| **Training data** | Not needed | Required | **Not needed** |
| **Method** | Deterministic | N models / dropout | **MC Dropout on PINN** |

This is the **only tutorial** that combines:
- **Physics-informed** learning (PDE + BC + IC, no labeled data)
- **Uncertainty quantification** (MC Dropout, epistemic uncertainty)

---

## 2. Method: MC Dropout for Bayesian PINN

```
Training:   Standard PINN training (PDE + BC + IC loss)
            Dropout layers active during training (regularization)

Inference:  Keep dropout ON (model.train() mode)
            Run T=50 stochastic forward passes
            Mean = best estimate   Std = epistemic uncertainty
```

### Key Insight
Uncertainty correlates with PDE residual — the model is **more uncertain where physics is less satisfied**, making this a *physics-aware* uncertainty estimate.

---

## 3. How to Run

```cmd
cd E:\physicsnemo-tutorials\pinn\bayesian_pinn
python bayesian_pinn.py
```

Trains Bayesian PINN (5000 epochs) + Deterministic PINN (5000 epochs) for comparison.

---

## 4. Results

- **`bayesian_pinn_overview.png`** — 3 panels: mean prediction, uncertainty (std), |PDE residual|
- **`bayesian_pinn_snapshots.png`** — Solution at t=0.1, 0.5, 0.9 with 95% confidence intervals
- **`bayesian_pinn_loss.png`** — Training loss curve
- **`bayesian_pinn_correlation.png`** — Uncertainty vs PDE residual scatter (Pearson r)
- **`bayesian_pinn_mc_distribution.png`** — MC sample histogram at shock point
- **`bayesian_vs_deterministic.png`** — Bayesian mean vs deterministic prediction
