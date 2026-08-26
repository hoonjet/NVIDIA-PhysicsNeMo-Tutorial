# Uncertainty Quantification Tutorials

> Know when the model is wrong — safety-critical ML for physics

---

## Overview

Uncertainty Quantification (UQ) tutorials answer the question: **"WHEN does the model make mistakes?"** All other tutorials train a single model and report a point estimate. UQ tutorials train multiple models and use their **disagreement** to quantify uncertainty, detect out-of-distribution inputs, and provide calibrated confidence estimates.

---

## Tutorials

| # | Tutorial | Method | Script |
|---|----------|--------|--------|
| 1 | [Deep Ensemble](deep_ensemble/) | N=5 independent CNNs, ensemble std | `deep_ensemble.py` |
| 2 | [MC Dropout](mc_dropout/) | 1 CNN with dropout, T=50 stochastic passes | `mc_dropout.py` |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Paradigm** | Uncertainty estimation (model disagreement) |
| **Uncertainty type** | Epistemic (reducible with more data) |
| **OOD detection** | ✓ (high uncertainty on unseen inputs) |
| **Calibration** | ✓ (uncertainty correlates with error) |
| **Accuracy boost** | ✓ (ensemble mean > single model) |
| **Safety** | ✓ (knows when not to trust) |

---

## vs. Other Categories

| Feature | PINN / Neural Operators | **Uncertainty** |
|---------|:----------------------:|:---------------:|
| **# Models** | 1 | **5 (ensemble)** |
| **Knows when wrong?** | No | **Yes** |
| **OOD detection** | ✗ | **✓** |
| **Calibration** | ✗ | **✓** |
| **Safety-critical** | Unsafe | **Safe** |
