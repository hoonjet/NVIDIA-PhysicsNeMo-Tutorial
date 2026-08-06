# Deep Ensemble for Uncertainty Quantification

> **Category**: `uncertainty/` — Uncertainty Quantification  
> **Paradigm**: Epistemic uncertainty estimation via model disagreement  
> **Model**: N=5 independent CNNs, ensemble mean & std

---

## 1. What Makes This Tutorial Unique?

This is the **only tutorial that answers "WHEN does the model make mistakes?"** Every other tutorial trains a single model and reports a point estimate. This tutorial trains **5 independent models** and uses their **disagreement** as a measure of uncertainty.

| Aspect | All Other Tutorials | This Tutorial |
|--------|---------------------|---------------|
| **Model count** | 1 model | **5 models (ensemble)** |
| **Output** | Point estimate | **Mean + uncertainty map** |
| **Knows when it's wrong?** | No | **Yes (ensemble std)** |
| **OOD detection** | No | **Yes (high uncertainty)** |
| **Safety-critical** | Unsafe (blind trust) | **Safe (knows limits)** |

### vs. Conditional Diffusion (also produces multiple outputs)
- Diffusion generates **diverse samples** from a learned distribution (aleatoric uncertainty)
- Deep ensemble measures **model disagreement** (epistemic uncertainty)
- Diffusion: "what are the possible solutions?" / Ensemble: "where is the model unsure?"
- Different uncertainty types, different methods, different purposes

---

## 2. Problem: Darcy Flow with Uncertainty

### Forward Problem
```
Input:  Permeability k(x)  [32×32]
Output: Pressure p(x)  [32×32]
Model:  CNN (encoder-decoder)
```

### What's Different?
We train **5 identical CNNs** with different random seeds. Each model learns a slightly different mapping. On in-distribution data, they agree. On out-of-distribution data, they disagree — this disagreement IS the uncertainty signal.

### In-Distribution (ID) vs Out-of-Distribution (OOD)
| Dataset | Length Scale | Description |
|---------|:-----------:|-------------|
| Train | 0.2 | Coarse permeability features |
| Test (ID) | 0.2 | Same distribution as training |
| Test (OOD) | 0.05 | Fine features, **never seen during training** |

---

## 3. Method: Deep Ensemble

### 3.1 Training
Train N independent models with different random initializations:
```
for i in range(N):
    torch.manual_seed(1000 + i)  # different seed
    model_i = CNN()
    train(model_i, data)  # standard training
```

### 3.2 Prediction
For each input, get N predictions:
```
preds = [model_i(x) for i in range(N)]  # N predictions
mean = average(preds)  # ensemble mean (better accuracy)
std = stddev(preds)     # uncertainty map
```

### 3.3 Uncertainty Interpretation
- **Low std**: Models agree → high confidence
- **High std**: Models disagree → low confidence (likely OOD or hard region)

### 3.4 OOD Detection
```
threshold = mean_std(ID) + 2 * std_std(ID)  # 95% of ID
if sample_std > threshold:
    → OOD detected (model shouldn't be trusted)
```

### 3.5 Calibration
Well-calibrated uncertainty: high predicted uncertainty ↔ high actual error.
Measured via Pearson correlation between ensemble std and absolute error.

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| `[1] Data Generation` | ID (length_scale=0.2) + OOD (length_scale=0.05) |
| `[2] CNN Model` | Simple encoder-decoder CNN |
| `[3] Train Ensemble` | 5 independent models, different seeds |
| `[4] Evaluate` | Ensemble mean/std, ID vs OOD, OOD detection |
| `[5] Calibration` | Uncertainty-error correlation, binned calibration |
| `[6] Visualization` | Loss, ID vs OOD, OOD histogram, calibration, ensemble vs single |

---

## 5. Key Results

### 5.1 Ensemble Effect (Accuracy)
The ensemble **mean** outperforms any single model — averaging reduces variance.

### 5.2 OOD Detection
OOD inputs have **significantly higher** ensemble std than ID inputs. A simple threshold (mean + 2σ of ID uncertainty) can detect most OOD samples.

### 5.3 Calibration
On ID data, ensemble std correlates with actual error — the model "knows" where it's wrong. On OOD data, both uncertainty and error increase, maintaining the correlation.

### 5.4 Uncertainty Maps
- **ID**: Low uncertainty overall, slightly higher at high-gradient regions
- **OOD**: High uncertainty everywhere, especially at fine-scale features the model never saw

---

## 6. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\uncertainty\deep_ensemble
python deep_ensemble.py
```

### Result Files
| File | Description |
|------|-------------|
| `results/ensemble_loss.png` | Training loss for all 5 models |
| `results/ensemble_result.png` | ID vs OOD: permeability, truth, mean, uncertainty |
| `results/ood_detection.png` | Uncertainty histogram: ID vs OOD + threshold |
| `results/calibration.png` | Binned calibration + pixel-wise scatter |
| `results/ensemble_vs_single.png` | Bar chart: 5 single models vs ensemble |

---

## 7. Key Concepts Learned

1. **Epistemic vs Aleatoric**: Epistemic = model uncertainty (reducible with more data). Aleatoric = data noise (irreducible). Deep ensemble captures **epistemic** uncertainty.

2. **Model Disagreement = Uncertainty**: When 5 models trained on the same data disagree, it means the input is in a region where the training data was insufficient. This is a purely **data-driven** uncertainty signal.

3. **OOD Detection for Free**: No separate OOD detector needed. The ensemble's disagreement naturally flags inputs that differ from training data. This is critical for safety-critical applications.

4. **Ensemble Averaging**: Beyond uncertainty, the ensemble **mean** is a better predictor than any single model. This is the bias-variance tradeoff in action: averaging reduces variance without increasing bias.

5. **Calibration**: A model that says "I'm 90% sure" should be right 90% of the time. Calibration measures this. Deep ensembles are naturally better calibrated than single models.

6. **Safety-Critical ML**: In nuclear, aerospace, medical — you need to know **when not to trust** the model. This tutorial teaches the fundamental approach: if models disagree, don't trust the prediction.

---

## 8. Comparison with Other Tutorials

| Feature | FNO / U-Net | Cond. Diffusion | **This Tutorial** |
|---------|:----------:|:--------------:|:------------------:|
| **# Models** | 1 | 1 | **5 (ensemble)** |
| **Uncertainty type** | None | Aleatoric | **Epistemic** |
| **OOD detection** | ✗ | ✗ | **✓** |
| **Calibration** | ✗ | ✗ | **✓** |
| **Accuracy boost** | — | — | **✓ (ensemble mean)** |
| **Safety** | Blind trust | Samples | **Knows limits** |
| **Cost** | 1× | 1× + sampling | **5× training** |

---

## 9. Extensions

- **MC Dropout**: Replace ensemble with dropout at inference (cheaper, less accurate)
- **Bayesian Neural Networks**: Place distributions over weights (principled but expensive)
- **Evidential Deep Learning**: Single model predicts both mean and uncertainty
- **Conformal Prediction**: Distribution-free uncertainty guarantees
- **Heteroscedastic uncertainty**: Model both epistemic + aleatoric simultaneously
- **Adaptive ensembling**: Add models only where uncertainty is high

---

## 10. References

- Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," NeurIPS 2017
- Gal & Ghahramani, "Dropout as a Bayesian Approximation," ICML 2016
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction," arXiv 2022
- Amini et al., "Deep Evidential Regression," NeurIPS 2020
