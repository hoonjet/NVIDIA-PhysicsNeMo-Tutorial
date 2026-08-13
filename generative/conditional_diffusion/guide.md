# Conditional Diffusion for PDE Solution Generation

> **Category**: `generative/` — Generative AI for Physics  
> **Paradigm**: Generative modeling (distribution learning, not deterministic prediction)  
> **Model**: Conditional DDPM (Denoising Diffusion Probabilistic Model) with U-Net

---

## 1. What Makes This Tutorial Unique?

This is the **only generative model** in the entire tutorial repository. Every other tutorial is **deterministic**: given an input, it predicts exactly one output. This tutorial learns a **distribution** of solutions and **generates diverse samples** from it.

| Aspect | All Other Tutorials | This Tutorial |
|--------|-------------------|---------------|
| **Input → Output** | 1 → 1 (deterministic) | 1 → N (stochastic generation) |
| **What it learns** | Point estimate | Conditional distribution p(solution \| input) |
| **Inference** | Single forward pass | 200-step reverse diffusion |
| **Output** | One solution | Multiple plausible solutions |
| **Uncertainty** | Not available | Ensemble std = uncertainty map |

### vs. Topology Optimization (also uses Diffusion)
- Topology Optimization generates **design shapes** (binary masks for structural design)
- This tutorial generates **PDE solutions** (pressure fields for stochastic Darcy flow)
- Different conditioning, different output space, different purpose

---

## 2. Problem: Stochastic Darcy Flow

### Governing Equation
```
-∇·(k(x) ∇p) = 1   on [0,1]²
p = 0               on boundary
```

### What's Stochastic?
The permeability field `k(x)` is a **random Gaussian field** — each sample is different. Even with the same boundary conditions, different `k(x)` produces different pressure fields `p(x)`.

### Why Generative?
In real-world subsurface flow (groundwater, oil reservoir), we **don't know** the exact `k(x)`. We have measurements at a few wells and a statistical model. The solution `p(x)` is therefore **uncertain** — it has a distribution, not a single value.

A generative model can:
1. Learn the conditional distribution `p(pressure | permeability)` from training data
2. Generate **multiple plausible pressure fields** for a given permeability
3. Quantify **solution uncertainty** via ensemble statistics

---

## 3. Method: Conditional DDPM

### 3.1 Forward Diffusion (Training)
Gradually add Gaussian noise to the clean pressure field over T=200 timesteps:

```
x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)
```

where `ᾱ_t = ∏(1-β_i)` is the cumulative product of noise schedule.

### 3.2 Noise Prediction Network
A **conditional U-Net** learns to predict the noise `ε` given:
- Noisy pressure field `x_t`
- Permeability field `k(x)` (conditioning)
- Timestep `t` (positional embedding)

```
ε_pred = UNet(x_t, k, t)
```

### 3.3 Reverse Diffusion (Sampling)
Starting from pure noise `x_T ~ N(0, I)`, iteratively denoise:

```
x_{t-1} = (1/√α_t) · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_pred) + σ_t · z
```

After 200 steps, the output is a **sample** from the learned distribution.

### 3.4 Conditioning
The permeability field `k(x)` is concatenated with the noisy pressure as a second channel:
```
UNet input: [x_t (noisy pressure), k (permeability)]  → 2 channels
```

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| `[1] Data Generation` | Random permeability via spectral method + FD solver for ground truth |
| `[2] Noise Schedule` | Linear β schedule, precompute ᾱ_t terms |
| `[3] Conditional U-Net` | DoubleConv + Down/Up blocks + timestep embedding |
| `[4] Training` | Sample random (x_0, t, ε), compute MSE loss on noise prediction |
| `[5] Sampling` | 200-step reverse diffusion to generate diverse solutions |
| `[6] Visualization` | Loss curve, generated samples, ensemble stats, reverse process |

---

## 5. Key Results

### 5.1 Diversity of Generated Samples
For a single permeability field, the model generates **8 different pressure fields** — each is a plausible solution, but they differ in details. This is impossible with deterministic models (FNO, U-Net, etc.).

### 5.2 Ensemble Statistics
- **Ensemble mean**: Averages out high-frequency noise, approximates the expected solution
- **Ensemble std**: Reveals where the solution is most uncertain (typically high-gradient regions)

### 5.3 Reverse Diffusion Process
Visualized at 4 timesteps (T, 3T/4, T/2, 0), showing how pure noise gradually transforms into a structured PDE solution.

---

## 6. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\generative\conditional_diffusion
python conditional_diffusion.py
```

### Expected Output
```
[1] Generating stochastic Darcy data...
  Train: 200 samples, k shape: (200, 32, 32), p shape: (200, 32, 32)
[2] DDPM config: 200 timesteps, beta: [0.0001, 0.02]
[3] Conditional U-Net parameters: ~500K
[4] Training DDPM (300 epochs, batch=32)...
  Epoch    0/300 | Loss: 8.5e-01 | Time: 0.0s
  ...
  Epoch  299/300 | Loss: 2.1e-03 | Time: ~120s
[5] Generating samples via reverse diffusion...
  Generated 8 samples for 1 permeability field
```

### Result Files
| File | Description |
|------|-------------|
| `results/diffusion_loss.png` | Training loss curve |
| `results/diffusion_result.png` | Generated samples + ensemble stats + scatter |
| `results/diffusion_process.png` | Reverse diffusion visualization (4 timesteps) |
| `results/diffusion_multitest.png` | 3 test cases: permeability, truth, gen mean, error |

---

## 7. Key Concepts Learned

1. **Generative vs Discriminative**: All other tutorials learn p(output|input) as a point estimate. This tutorial learns the full distribution.

2. **Forward/Reverse Process**: Forward diffusion destroys structure (add noise); reverse diffusion creates structure (denoise). The network learns the reverse process.

3. **Score Matching**: Predicting noise ε is equivalent to estimating the score ∇log p(x_t), the gradient of the data distribution.

4. **Conditioning**: The permeability field guides generation — without it, the model would generate random pressure fields unrelated to the input.

5. **Stochastic PDEs**: When PDE coefficients are random, the solution is a random field. Generative models naturally handle this; deterministic models cannot.

6. **Uncertainty Quantification**: Ensemble std provides a **free** uncertainty estimate — no additional training needed, just multiple samples.

---

## 8. Comparison with Other Tutorials

| Feature | FNO / U-Net | TopoDiff | **This Tutorial** |
|---------|:----------:|:--------:|:------------------:|
| **Paradigm** | Deterministic | Generative | **Generative** |
| **Output** | 1 solution | 1 design | **N solutions** |
| **What's generated** | Nothing | Design shape | **PDE solution** |
| **Conditioning** | Input field | Load/constraint | **Permeability field** |
| **Uncertainty** | ✗ | ✗ | **✓ (ensemble std)** |
| **Diffusion steps** | 0 | ~100 | **200** |
| **Physics** | Darcy/NS/Heat | Topology | **Stochastic Darcy** |

---

## 9. Extensions

- **Classifier-free guidance**: Train with 10% dropout on conditioning for better generation quality
- **Latent diffusion**: Diffuse in a compressed latent space (VAE encoder) for larger grids
- **Score-based SDE**: Replace discrete DDPM with continuous SDE for better theoretical properties
- **Multi-resolution**: Hierarchical diffusion at multiple grid resolutions
- **Physics-informed loss**: Add PDE residual loss to the denoiser for physics consistency

---

## 10. References

- Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020
- Song et al., "Score-Based Generative Modeling through SDEs," ICLR 2021
- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis," NeurIPS 2021
- Rozdeba et al., "Generative AI for Subsurface Flow," arXiv 2024
