# PINN Tutorial: Adaptive Sampling (RAR) — English

## Overview
This tutorial introduces **Residual-based Adaptive Refinement (RAR)** — an advanced PINN training strategy that dynamically redistributes collocation points to regions where the PDE residual is largest. All other PINN tutorials in this repo use **static, uniform random** collocation points. This is the only tutorial that teaches **adaptive sampling**, a transferable skill applicable to any PINN problem.

The problem is a 2D Poisson equation with a **sharp Gaussian source** that creates a localized peak — exactly the type of feature that static sampling struggles to resolve.

## Equation
```
-Laplacian(u) = f(x, y)
u = 0 on boundary (Dirichlet)
```
- Domain: [0, 1] × [0, 1]
- Source: f(x,y) = 10 · exp(-((x-0.5)² + (y-0.5)²) / (2·0.05²))
- The sharp Gaussian (σ=0.05) creates a localized peak at the center

## What Makes This Tutorial Unique?

| Feature | Other PINN Tutorials | **This Tutorial** |
|---------|---------------------|-------------------|
| Collocation sampling | Static uniform random | **Adaptive (RAR)** |
| Point redistribution | Never (fixed) | **Every 500 epochs** |
| Sampling guide | None | **PDE residual magnitude** |
| Clustering | None | **k-means for spatial spreading** |
| Comparison | None | **Side-by-side: Static vs Adaptive** |

## RAR Algorithm

```
1. Train PINN with initial uniform collocation points for N epochs
2. Evaluate PDE residual on a large candidate pool (10,000 points)
3. Weight each candidate by residual² (high residual = high probability)
4. Sample 5×N points from candidates using residual-weighted probabilities
5. Apply k-means clustering to get N well-distributed cluster centers
6. Replace collocation points with new cluster centers
7. Repeat from step 1
```

**Why k-means?** Pure residual-weighted sampling would cluster all points in one spot. k-means ensures points are spread within high-residual regions.

## Model Architecture
- **Network**: 2 → 64 → 64 → 64 → 64 → 1 (Fully Connected, Tanh)
- **Parameters**: 12,737
- **Optimizer**: Adam (lr=1e-3, StepLR gamma=0.5 every 2000 epochs)
- **Epochs**: 5,000
- **Collocation points**: 2,000 (identical budget for both static and adaptive)
- **Boundary points**: 400
- **Candidate pool**: 10,000
- **RAR interval**: Every 500 epochs (9 redistribution steps)

## Reference Solution
A fine finite difference solution (101×101 grid) is computed via direct sparse solve as ground truth for quantitative comparison.

## Key Results

| Metric | Static (Uniform) | Adaptive (RAR) | Improvement |
|--------|-----------------|----------------|-------------|
| Training time | ~60s | ~80s | — |
| Final total loss | ~1×10⁻³ | ~5×10⁻⁴ | 2× lower |
| Final PDE loss | ~1×10⁻³ | ~5×10⁻⁴ | 2× lower |
| Relative L2 error | ~0.05 | ~0.02 | **2–3× better** |
| Max absolute error | ~0.03 | ~0.01 | **2–3× better** |

> Exact values depend on hardware; the relative improvement is consistent.

## Key Observations
1. **RAR redistributes collocation points to high-residual regions** — points concentrate near the sharp Gaussian source over time
2. **Same compute budget, lower error** — adaptive sampling achieves 2–3× better L2 and max error with the same number of collocation points and epochs
3. **Improvement is localized** — the biggest error reduction occurs near the sharp source, exactly where static sampling is weakest
4. **k-means prevents clustering** — without it, all points would collapse to a single high-residual spot
5. **Transferable technique** — RAR can be applied to any existing PINN tutorial (Burgers, Navier-Stokes, Heat Transfer, etc.) to improve accuracy

## How to Run
```cmd
E:\physicsnemo_env\Scripts\python.exe adaptive_sampling.py
```

> **Note**: Requires `scikit-learn` for k-means clustering. Install with:
> ```cmd
> pip install scikit-learn
> ```

## Output Files
- `results/adaptive_sampling_loss.png` — Loss curves: Static vs Adaptive (with RAR markers)
- `results/adaptive_sampling_result.png` — Solution & error maps: Reference, Static, Adaptive
- `results/adaptive_sampling_points.png` — Collocation point redistribution over training
- `results/adaptive_sampling_cross.png` — Cross-section comparison through source center

## When to Use Adaptive Sampling

| Scenario | Use RAR? |
|----------|----------|
| Smooth solutions, no localized features | Static is fine |
| **Sharp gradients / discontinuities** | **Yes — big improvement** |
| **Localized source terms** | **Yes — essential** |
| **Boundary layers** | **Yes** |
| Quick prototyping | Static (simpler) |
| Production accuracy | **RAR or similar** |

## References
- Wu et al., "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks," Computer Methods in Applied Mechanics and Engineering, 2023.
- Lu et al., "DeepXDE: A deep learning library for solving differential equations," SIAM Review, 2021.
- Nabian et al., "Efficient training of physics-informed neural networks via importance sampling," Computer-Aided Civil and Infrastructure Engineering, 2021.
