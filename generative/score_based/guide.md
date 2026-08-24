# Score-Based Generative Model for PDE Solution Generation

> **Category**: `generative/` — Generative AI for Physics
> **Paradigm**: Continuous-time SDE (score matching)
> **Model**: ScoreNet (U-Net with time embedding)

---

## 1. What Makes This Tutorial Unique?

This tutorial uses a **Score-Based Generative Model** with continuous-time SDEs, which is fundamentally different from the existing DDPM tutorial.

| Aspect | DDPM (existing) | Score-Based (THIS) |
|--------|-----------------|-------------------|
| **Time** | Discrete (t=1,...,T) | Continuous (t∈[0,1]) |
| **Noise schedule** | Discrete betas | Continuous sigma(t) |
| **What's predicted** | Noise ε | Score ∇log p(x) |
| **Sampling** | Reverse Markov chain | Reverse SDE or prob. flow ODE |
| **Steps** | Fixed (200) | Flexible (any number) |
| **Theory** | Discrete Markov chain | Continuous SDE/ODE |

### Key Difference: Score Function vs Noise Prediction
- **DDPM**: Predicts the noise ε that was added, then removes it
- **Score-Based**: Predicts the score function s(x,t) = ∇log p_t(x), the gradient of log probability density
- The score tells you "which direction to move x to increase its probability" — a more fundamental quantity

---

## 2. Problem: Stochastic Darcy Flow

Same PDE as the DDPM tutorial:
```
-∇·(k(x) ∇p) = 1   on [0,1]²
p = 0               on boundary
```

The permeability k(x) is a random Gaussian field → solution p(x) has a distribution.

---

## 3. Method: Score-Based Generative Model

### 3.1 Score Function
```
s(x, t) = ∇_x log p_t(x)
```
- Gradient of log probability density
- Points toward higher probability regions
- For Gaussian noise: s = -ε / σ(t) (analytical)

### 3.2 Forward SDE (Variance Exploding)
```
x_t = x_0 + σ(t) · ε,   ε ~ N(0, I)
```
- σ(t) = σ_min · (σ_max/σ_min)^t (exponential schedule)
- σ_min=0.01 (almost no noise), σ_max=50 (pure noise)

### 3.3 Training: Denoising Score Matching
```
Loss = E_t [ σ²(t) · ||s_θ(x_t, t, c) - (-ε/σ(t)) ||² ]
```
- Sample random time t ~ U(0,1)
- Perturb data: x_t = x_0 + σ(t)·ε
- Train network to predict score: s_θ ≈ -ε/σ(t)
- Weight by σ²(t) for importance sampling

### 3.4 Sampling: Two Methods

**Reverse SDE** (stochastic):
```
dx = -σ²(t)·s(x,t)·dt - σ(t)·dw
```
- Adds noise during denoising → more diverse samples

**Probability Flow ODE** (deterministic):
```
dx = -0.5·σ²(t)·s(x,t)·dt
```
- No noise → deterministic, faster, less diverse

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| [0] Setup | Device, seeds, results directory |
| [1] Data | Generate stochastic Darcy flow (200 train, 20 test) |
| [2] ScoreNet | U-Net with time embedding (2ch input, 1ch output) |
| [3] SDE | Variance Exploding: σ(t) = σ_min·(σ_max/σ_min)^t |
| [4] Training | Score matching, 300 epochs, cosine LR |
| [5] Sampling | Reverse SDE (200 steps) + Prob. flow ODE (100 steps) |
| [6] Visualization | Loss, samples, error, concept explanation |
| [7] Summary | Metrics and key observations |

---

## 5. Key Results

| Metric | Reverse SDE | Prob. Flow ODE |
|--------|-------------|----------------|
| Mean MAE | ~0.05 | ~0.06 |
| Uncertainty (std) | Higher | Lower |
| Diversity | More diverse | Less diverse |
| Speed | Slower (200 steps) | Faster (100 steps) |

---

## 6. How to Run

```cmd
cd E:\physicsnemo-tutorials\generative\score_based
python score_based.py
```

Results saved to `results/`:
- `score_based_loss.png` — Training/test loss curves
- `score_based_samples.png` — Generated samples (SDE + ODE)
- `score_based_error.png` — Mean prediction error
- `score_based_explanation.png` — Concept comparison with DDPM

---

## 7. Parameter Experimentation

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SIGMA_MIN` | 0.01 | Smaller = less residual noise |
| `SIGMA_MAX` | 50.0 | Larger = more diverse samples |
| `N_STEPS_SDE` | 200 | More steps = better quality, slower |
| `N_STEPS_ODE` | 100 | Fewer steps OK for ODE (deterministic) |
| `N_SAMPLES` | 5 | More samples = better uncertainty estimate |
| `EPOCHS` | 300 | More epochs = better score estimation |

---

## 8. vs. DDPM (Existing Tutorial)

| Feature | DDPM | Score-Based |
|---------|------|-------------|
| **Theoretical framework** | Discrete Markov chain | Continuous SDE |
| **Generality** | Special case of score-based | General framework |
| **Sampling flexibility** | Fixed steps | Any number of steps |
| **Deterministic option** | No (DDIM is separate) | Yes (prob. flow ODE) |
| **What's learned** | Noise ε | Score ∇log p(x) |
| **Implementation** | Simpler | Slightly more complex |

**When to use which?**
- DDPM: Simpler, well-tested, good enough for most cases
- Score-Based: When you need continuous theory, flexible sampling, or deterministic ODE sampling

---

## 9. References

- Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021
- Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution", NeurIPS 2019
- Score matching: Hyvärinen (2005)
