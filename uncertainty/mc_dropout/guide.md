# MC Dropout for Uncertainty Quantification

> **Category**: `uncertainty/` — Uncertainty Quantification
> **Paradigm**: Approximate Bayesian inference via dropout
> **Model**: CNN with dropout (1 model, T=50 stochastic passes)

---

## 1. What Makes This Tutorial Unique?

| Aspect | Deep Ensemble (existing) | MC Dropout (THIS) |
|--------|--------------------------|-------------------|
| **Models trained** | 5 independent CNNs | 1 CNN with dropout |
| **Training cost** | 5x | 1x (5x cheaper!) |
| **Uncertainty source** | std across 5 models | std across 50 stochastic passes |
| **Quality** | Best (true ensemble) | Good (approximate) |
| **Theory** | Model disagreement | Approximate Bayesian inference |

### Key Difference: How Uncertainty is Obtained
- **Deep Ensemble**: Train 5 separate models, each with different random init
- **MC Dropout**: Train 1 model, keep dropout ON at inference, run T=50 forward passes
- Each forward pass uses a different random subnetwork (dropout mask) → like a mini-ensemble

---

## 2. Problem: Darcy Flow with OOD Detection

Same as Deep Ensemble tutorial:
- Train: 200 samples (length_scale=0.2, coarse features)
- Test ID: 20 samples (same distribution)
- Test OOD: 20 samples (length_scale=0.05, fine features, never seen)

---

## 3. Method: MC Dropout

### 3.1 Training
- Standard CNN training with dropout layers (p=0.1)
- Dropout acts as regularization during training

### 3.2 Inference (KEY DIFFERENCE)
- **Normal inference**: `model.eval()` → dropout OFF → deterministic output
- **MC Dropout inference**: `model.train()` → dropout ON → stochastic output
- Run T=50 forward passes → 50 different predictions
- **Mean** = prediction, **Std** = uncertainty

### 3.3 Theory (Gal & Ghahramani, 2016)
- Dropout = Bernoulli mask on weights
- Each forward pass = different subnetwork
- T passes = T subnetworks = approximate ensemble
- Mathematically equivalent to variational Bayesian inference
- "Dropout as a Bayesian Approximation"

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| [1] Data | Darcy flow (200 train, 20 ID, 20 OOD) |
| [2] Model | CNN with Dropout2d (p=0.1) |
| [3] Training | 200 epochs, standard MSE loss |
| [4] MC Inference | T=50 passes with dropout ON |
| [5] Calibration | Error-uncertainty correlation |
| [6] Visualization | Loss, ID/OOD predictions, OOD detection |
| [7] Summary | Metrics and comparison with Deep Ensemble |

---

## 5. Key Results

| Metric | ID | OOD |
|--------|----|----|
| MAE | ~0.05 | ~0.08 |
| Uncertainty | ~0.02 | ~0.04 |
| OOD/ID ratio | — | ~2x (OOD should be higher) |
| Calibration (r) | ~0.3 | ~0.5 |

---

## 6. How to Run

```cmd
cd E:\physicsnemo-tutorials\uncertainty\mc_dropout
python mc_dropout.py
```

Results saved to `results/`:
- `mc_dropout_loss.png` — Training loss
- `mc_dropout_id.png` — In-distribution predictions + uncertainty
- `mc_dropout_ood.png` — OOD predictions + uncertainty
- `mc_dropout_ood_detection.png` — OOD detection & calibration
- `mc_dropout_explanation.png` — Concept comparison with Deep Ensemble

---

## 7. vs. Deep Ensemble (Existing Tutorial)

| Feature | Deep Ensemble | MC Dropout |
|---------|--------------|------------|
| **Training cost** | 5x | 1x |
| **Inference cost** | 5 passes | 50 passes |
| **Uncertainty quality** | Best | Good |
| **Memory** | 5 models | 1 model |
| **Implementation** | Simple (train 5x) | Simple (add dropout) |
| **Theory** | Model disagreement | Bayesian approximation |

**When to use which?**
- MC Dropout: Large models, quick prototyping, training is expensive
- Deep Ensemble: Small models, need best uncertainty, safety-critical

---

## 8. References

- Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014
