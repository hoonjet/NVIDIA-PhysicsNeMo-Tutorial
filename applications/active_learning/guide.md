# Active Learning for PDE Surrogate Models

> **Category**: `applications/` — Data-efficient surrogate training
> **Paradigm**: Uncertainty-based selective sampling (active learning loop)
> **Model**: CNN with MC Dropout (surrogate + uncertainty estimator)

---

## 1. What Makes This Tutorial Unique?

| Aspect | Topology Optimization (existing) | Active Learning (THIS) |
|--------|----------------------------------|----------------------|
| **Goal** | Find optimal design | Efficient surrogate training |
| **Method** | Generative model | Uncertainty-based sampling |
| **Data** | Pre-computed solutions | Selectively queried from oracle |
| **Key question** | "What's the best design?" | "Which samples to simulate next?" |
| **Cost optimization** | Design quality | Simulation budget |

### Key Difference: What's Being Optimized
- **Topology Optimization**: Optimizes the **design** (material distribution)
- **Active Learning**: Optimizes the **data collection** (which simulations to run)
- Active learning minimizes expensive simulation calls by smart sample selection

---

## 2. Problem: Darcy Flow Surrogate

Train a CNN surrogate for Darcy flow with minimal simulation budget.

- **Oracle**: FDM solver (expensive — takes time per simulation)
- **Pool**: 200 unlabeled candidate permeability fields
- **Budget**: Start with 20, add 10 per round, 8 rounds = 100 total
- **Goal**: Achieve low test MAE with fewer simulations than random sampling

---

## 3. Method: Active Learning Loop

```
1. Train surrogate on labeled data
2. Predict on unlabeled pool (MC Dropout for uncertainty)
3. Select most uncertain samples (acquisition function)
4. Query oracle (FDM solver) for selected samples
5. Add to training set → retrain
6. Repeat
```

### Acquisition Function: Uncertainty Sampling
- Compute MC Dropout uncertainty (std across T=20 passes) for each pool sample
- Select top-N samples with highest uncertainty
- "Where is the model most unsure?" → simulate those

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| [1] Problem | Darcy surrogate, 20 initial + 10/round × 8 rounds |
| [2] Data | 200 pool, 50 test (FDM oracle) |
| [3] Model | CNN with MC Dropout (p=0.1) |
| [4] AL Loop | Train → predict → select → simulate → retrain |
| [5] Visualization | AL vs random, predictions, concept |
| [6] Summary | Sample efficiency comparison |

---

## 5. How to Run

```cmd
cd E:\physicsnemo-tutorials\applications\active_learning
python active_learning.py
```

Results saved to `results/`:
- `active_learning_comparison.png` — AL vs random (MAE & uncertainty)
- `active_learning_rounds.png` — Per-round MAE comparison
- `active_learning_predictions.png` — Final predictions
- `active_learning_explanation.png` — Concept comparison

---

## 6. vs. Topology Optimization (Existing Tutorial)

| Feature | Topology Optimization | Active Learning |
|---------|----------------------|-----------------|
| **What's optimized** | Design (material) | Data collection |
| **Method** | Generative model | Uncertainty sampling |
| **Oracle** | Not used | FDM solver (expensive) |
| **Key metric** | Design quality | Sample efficiency |
| **Real-world** | Structural design | Simulation budget |

---

## 7. References

- Settles, "Active Learning Literature Survey" (2009)
- Gal et al., "Deep Bayesian Active Learning with Image Data" (2017)
- Ren et al., "A Meta-Learning Approach for Active Learning" (2020)
