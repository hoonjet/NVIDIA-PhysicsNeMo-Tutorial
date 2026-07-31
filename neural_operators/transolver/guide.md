# PhysicsNeMo Transolver Tutorial Guide (Detailed Guide for Beginners)

> **Date**: 2026-07-10  
> **Environment**: PhysicsNeMo 1.3.0, PyTorch 2.7.1+cu118, CUDA (Quadro P4000 8GB)  
> **Location**: `E:\physicsnemo_env`

---

## Table of Contents

1. [Tutorial Overview](#1-tutorial-overview)
2. [What is Transolver? (Beginner-Friendly Explanation)](#2-what-is-transolver-beginner-friendly-explanation)
3. [Understanding Physics Attention](#3-understanding-physics-attention)
4. [How to Run the Tutorial](#4-how-to-run-the-tutorial)
5. [Step-by-Step Code Explanation](#5-step-by-step-code-explanation)
6. [Interpreting Results](#6-interpreting-results)
7. [PINN vs FNO vs Transolver Comparison](#7-pinn-vs-fno-vs-transolver-comparison)
8. [Parameter Experimentation Guide](#8-parameter-experimentation-guide)
9. [Troubleshooting (FAQ)](#9-troubleshooting-faq)

---

## 1. Tutorial Overview

### What Did We Learn?

In this tutorial, we used the **Transolver** model to solve the **Darcy Flow** (flow in porous media) problem.

| Item | Details |
|------|---------|
| **Problem** | Darcy Flow (permeability field → pressure field prediction) |
| **Governing equation** | Darcy equation: -div(k·grad(p)) = f |
| **Learning method** | Data-driven (supervised learning, different from PINN) |
| **Neural network** | PhysicsNeMo Transolver (4 layers, 128 hidden, 8 heads, 32 slices) |
| **Parameters** | 1,831,073 |
| **Training time** | ~5,324 seconds (GPU: Quadro P4000) |
| **Final loss** | 0.0108 (converged from 1.033) |
| **Result** | Pressure field prediction successful (error < 0.25) |

### Files Used

| File | Description |
|------|-------------|
| `tutorial_transolver_darcy.py` | Main tutorial script (Transolver implementation) |
| `tutorial_results/transolver_darcy_result.png` | Prediction visualization (input, target, prediction, error) |
| `tutorial_results/transolver_darcy_loss.png` | Training loss curve |

---

## 2. What is Transolver? (Beginner-Friendly Explanation)

### 2.1 What is Transolver?

**Transolver** is a model that adapts the Transformer architecture for PDE (partial differential equation) problems.

- **Trans**former + S**olver** = **Transolver**
- Replaces standard transformer attention with **Physics Attention**
- Optimized structure for learning PDE solutions

### 2.2 Why Not Standard Transformer?

Standard transformers (GPT, BERT, etc. used in NLP) have **all tokens attend to each other**.

```
Standard transformer attention:
  Token1 → Token2  ✓
  Token1 → Token3  ✓
  Token1 → Token4  ✓
  ... (all pairs computed)
  
  N tokens → N×N = N² computations
  32×32 grid = 1024 tokens → 1,048,576 computations!
```

For PDE problems, computation explodes as grid size increases. 256×256 grid = 65,536 tokens → 4.2 billion computations!

### 2.3 Transolver's Solution: "Slicing"

Instead of all tokens attending to each other, Transolver **learns a small number of "slices"** and only attends between slices.

```
Transolver Physics Attention:
  1024 tokens → compress to 32 slices
  Attention only between 32 slices
  Distribute results back to 1024 tokens
  
  N tokens, S slices → N×S computations
  1024 tokens, 32 slices → 32,768 computations (32× reduction!)
```

### 2.4 Key Components of Transolver

```
[Input] → [Position Embedding] → [Preprocessing MLP] → [Transolver Block × N] → [Output]
                                                        │
                                                        ├── LayerNorm
                                                        ├── Physics Attention (slicing!)
                                                        ├── Residual Connection
                                                        ├── LayerNorm
                                                        ├── MLP
                                                        └── Residual Connection
```

---

## 3. Understanding Physics Attention

### 3.1 What is a Slice?

A slice is a **learned spatial cluster**. The model learns "tokens in this region have similar physical properties" and creates groups.

Analogy:
- **Standard attention**: All students in a classroom talk to each other (100 × 100 = 10,000 conversations)
- **Physics Attention**: Divide classroom into 10 groups, only group leaders talk (100 → 10 groups → 10×10 = 100 conversations)

### 3.2 How Physics Attention Works

```
1. Projection
   Input tokens → project to slice space
   "Which slice does each token belong to?" (learned)

2. Slice Token Computation
   Weighted average of tokens in each slice → representative value
   "What is the representative feature of this slice?"

3. Slice Attention
   Attention only between slices (S×S, very efficient!)
   "How does slice A's information affect slice B?"

4. Deslicing
   Map attention results back to token space
   "Each token receives its slice's attention result"
```

### 3.3 Structured vs Unstructured Mesh

Transolver supports two data formats:

| Format | Description | Use Case |
|--------|-------------|----------|
| **Structured mesh** | Regular grid (image-like) | 2D grid, 3D voxel |
| **Unstructured mesh** | Irregular points (CFD mesh) | Complex geometry, FEM mesh |

This tutorial uses **structured 2D mesh** (32×32 grid).

---

## 4. How to Run the Tutorial

### Method 1: Run from Command Line

```cmd
cd E:\physicsnemo_env
Scripts\activate
python tutorial_transolver_darcy.py
```

### Method 2: Run from VS Code

1. Open `E:\physicsnemo_env\tutorial_transolver_darcy.py` in VS Code
2. Set Python interpreter to `E:\physicsnemo_env\Scripts\python.exe`
3. Run with `F5` or `Ctrl+F5`

### View Results

```cmd
start E:\physicsnemo_env\tutorial_results\transolver_darcy_result.png
start E:\physicsnemo_env\tutorial_results\transolver_darcy_loss.png
```

### Prerequisite: Transformer Engine Mock

This tutorial includes mock modules to run without the `transformer_engine` package:
- `E:\physicsnemo_env\lib\site-packages\transformer_engine\__init__.py`
- `E:\physicsnemo_env\lib\site-packages\transformer_engine\pytorch.py`

This mock enables pure PyTorch operation in `use_te=False` mode.

---

## 5. Step-by-Step Code Explanation

### 5.1 Overall Structure (7 Steps)

```
tutorial_transolver_darcy.py
|
+-- [0] Environment Setup    - Check GPU, fix random seeds
+-- [1] Data Generation      - Darcy Flow synthetic data (permeability → pressure)
+-- [2] Model Creation       - Create Transolver neural network
+-- [3] Training Setup       - Configure optimizer, epochs
+-- [4] Training Loop        - 500 epochs of iterative training
+-- [5] Resolution Test      - Explain model characteristics
+-- [6] Visualization         - Plot prediction results
+-- [7] Summary              - Compare PINN/FNO/Transolver
```

### 5.2 [1] Data Generation

```python
N_TRAIN = 200      # Training samples
N_GRID = 32        # Grid resolution (32x32)
```

Uses the same Darcy Flow synthetic data as the FNO tutorial. The difference is **data format**:

| Model | Data Format | Description |
|-------|-------------|-------------|
| FNO | `[B, C, H, W]` | channels-first (PyTorch standard) |
| Transolver | `[B, H, W, C]` | channels-last (transformer standard) |

```python
# FNO format: [B, 1, 32, 32]
k_train = generate_darcy_data(...)  # [200, 1, 32, 32]

# Convert to Transolver format: [B, H, W, C] = [200, 32, 32, 1]
k_train_transolver = k_train.permute(0, 2, 3, 1).contiguous()
```

### 5.3 [2] Transolver Model Creation

```python
model = Transolver(
    functional_dim=1,           # Input channels (permeability 1)
    out_dim=1,                   # Output channels (pressure 1)
    n_layers=4,                  # Number of Transolver blocks
    n_hidden=128,                # Hidden dimension (model width)
    n_head=8,                    # Number of attention heads
    dropout=0.0,                 # Dropout (deterministic learning)
    act="gelu",                  # GELU activation
    mlp_ratio=4,                 # MLP expansion ratio
    slice_num=32,                # Number of slices (key parameter!)
    unified_pos=True,            # Use unified position encoding
    ref=8,                       # Reference grid size (8x8)
    structured_shape=(32, 32),   # 2D structured grid
    use_te=False,                # Pure PyTorch mode
)
```

#### Key Parameter Explanation

**functional_dim (input channels)**:
- Number of physical variables. Excludes position embedding.
- Darcy: 1 (permeability), Navier-Stokes: 3 (u, v, p)

**n_hidden (hidden dimension)**:
- Transformer "width". Larger = more expressiveness but more memory.
- Must be divisible by n_head (128 / 8 = 16 ✓)

**slice_num (number of slices)** ⭐:
- **The most important parameter!**
- Determines how many spatial clusters each layer learns
- Smaller: faster but less expressiveness
- Larger: more expressiveness but slower
- "How many regions will the model divide space into?"

**unified_pos (unified position encoding)**:
- Automatically generates position info for structured grids
- Uses ref×ref reference grid to encode each point's position
- True: structured grid, False: unstructured mesh (provide embedding directly)

**structured_shape (structured shape)**:
- (H, W): 2D grid → convolution-based slice projection
- (H, W, D): 3D voxel → 3D convolution-based
- None: unstructured mesh → linear projection-based

**use_te (Transformer Engine)**:
- True: Use NVIDIA Transformer Engine (optimized, requires TE installation)
- False: Pure PyTorch (slower but no dependencies)

### 5.4 [4] Training Loop

```python
for epoch in range(500):
    optimizer.zero_grad()
    
    # Forward pass: [B, H, W, C] → [B, H, W, C]
    pred = model(k_train_transolver)
    
    # MSE loss (unlike PINN, no PDE residual!)
    loss = loss_fn(pred, p_train_transolver)
    
    loss.backward()
    optimizer.step()
```

Key difference from PINN:
- **PINN**: PDE residual + boundary condition loss (requires 2nd-order autograd)
- **Transolver**: Simple MSE loss (supervised learning, no autograd needed)

### 5.5 [5] Resolution Characteristics

FNO uses Fourier modes making it **resolution-independent** (train on 32×32 → infer on 64×64).

Transolver's structured mesh mode uses convolutions, so it has **fixed resolution**. However, unstructured mesh mode (`structured_shape=None`) can handle arbitrary point counts.

| Feature | FNO | Transolver (structured) | Transolver (unstructured) |
|---------|-----|------------------------|--------------------------|
| Resolution independence | ✓ | ✗ | ✓ (variable point count) |
| Irregular mesh | ✗ | ✗ | ✓ |
| Complex geometry | ✗ | ✗ | ✓ |

---

## 6. Interpreting Results

### 6.1 Training Log

```
Epoch   0/500 | Loss: 1.033244e+00 | Time: 14.2s    (initial)
Epoch  50/500 | Loss: 2.989372e-01 | Time: 523.6s   (rapid decrease)
Epoch 100/500 | Loss: 2.335838e-01 | Time: 1019.9s   (plateau)
Epoch 200/500 | Loss: 2.072675e-02 | Time: 2098.2s  (sharp decrease!)
Epoch 350/500 | Loss: 1.448573e-02 | Time: 3790.9s  (gradual decrease)
Epoch 499/500 | Loss: 1.081579e-02 | Time: 5323.7s  (converged)
```

### 6.2 Loss Curve Analysis (transolver_darcy_loss.png)

Loss converged in 3 stages:
1. **0~100 epochs**: 1.03 → 0.23 (rapid initial decrease, plateau exists)
2. **100~200 epochs**: 0.23 → 0.02 (sharp decrease, core learning phase)
3. **200~500 epochs**: 0.02 → 0.01 (gradual fine-tuning)

### 6.3 Prediction Results (transolver_darcy_result.png)

| Panel | Observation | Physical Meaning |
|-------|-------------|------------------|
| **Input (permeability k)** | Random pattern, range 0.4~0.9 | Irregular permeability distribution in porous media |
| **Target (pressure p)** | High pressure center (1.75), low outside (0.0) | Fluid generated at center spreads outward |
| **Prediction** | Nearly identical to target | Transolver accurately learned pressure distribution |
| **Error** | Mostly < 0.10, max 0.25 at boundary | Slight error at boundaries, accurate overall |

**Conclusion**: Transolver accurately predicted the Darcy Flow pressure field. Predictions are visually indistinguishable from the target.

---

## 7. PINN vs FNO vs Transolver Comparison

| Feature | PINN | FNO | Transolver |
|---------|------|-----|------------|
| **Learning method** | Equation-based | Data-driven | Data-driven |
| **Training data needed** | Not needed | Needed | Needed |
| **Core mechanism** | Autograd | Fourier transform | Physics Attention |
| **Resolution independence** | ✓ (continuous function) | ✓ (Fourier modes) | ✗ (structured), ✓ (unstructured) |
| **Irregular mesh** | ✓ | ✗ | ✓ |
| **Complex geometry** | Good | Low | Very good |
| **Memory usage** | Low | Medium | High |
| **Training speed** | Slow | Fast | Medium |
| **Inference speed** | Slow (per-point) | Fast (grid) | Fast (grid) |
| **PhysicsNeMo model** | FullyConnected | FNO | Transolver |

### When to Use What?

| Situation | Recommended Model | Reason |
|-----------|-------------------|--------|
| No labeled data, only equations | **PINN** | Learn from equations only |
| Regular grid, fast surrogate needed | **FNO** | Fast and resolution-independent |
| Complex geometry, irregular mesh | **Transolver** | Supports irregular mesh, handles complex geometry |
| Inverse problems (parameter estimation) | **PINN** | Naturally handles inverse problems |
| Multi-physics, transfer learning | **Transolver** | Transformer flexibility |

---

## 8. Parameter Experimentation Guide

### 8.1 Changing Slice Count (Most Important!)

```python
# Experiment with accuracy-speed tradeoff via slice count
model = Transolver(
    slice_num=8,    # Few: fast but less accurate
    slice_num=32,   # Default: balanced
    slice_num=64,   # Many: more accurate but slower
    slice_num=128,  # Very many: best accuracy, lowest speed
)
```

### 8.2 Changing Model Size

```python
# Larger model for better accuracy
model = Transolver(
    n_layers=6,      # 4 → 6 (deeper)
    n_hidden=256,    # 128 → 256 (wider)
    n_head=8,        # 256 / 8 = 32 dim/head
)
```

### 8.3 Unstructured Mesh Mode (Irregular Points)

```python
# Irregular mesh mode: structured_shape=None, unified_pos=False
model = Transolver(
    functional_dim=1,
    out_dim=1,
    embedding_dim=2,          # 2D coordinate embedding (x, y)
    structured_shape=None,     # Unstructured mesh!
    unified_pos=False,         # Don't use unified position encoding
    use_te=False,
)

# Input: [B, N_points, C] + [B, N_points, embedding_dim]
# N_points is variable (works at different resolutions!)
```

### 8.4 Learning Rate Scheduling

```python
# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=500, eta_min=1e-5
)

for epoch in range(EPOCHS):
    # ... training ...
    scheduler.step()
```

---

## 9. Troubleshooting (FAQ)

### Q: "No module named 'transformer_engine'" error

**A**: Check that mock transformer_engine modules are installed:
```
E:\physicsnemo_env\lib\site-packages\transformer_engine\__init__.py
E:\physicsnemo_env\lib\site-packages\transformer_engine\pytorch.py
```
With these files, you can run in pure PyTorch mode with `use_te=False`.

### Q: CUDA Out of Memory error

**A**: Reduce model size:
```python
model = Transolver(
    n_hidden=64,      # 128 → 64
    n_layers=2,       # 4 → 2
    slice_num=16,     # 32 → 16
)
```

### Q: Training is too slow

**A**: Try the following:
1. Decrease `n_hidden` (128 → 64)
2. Decrease `n_layers` (4 → 2)
3. Decrease `slice_num` (32 → 16)
4. Decrease epochs (500 → 200)

### Q: Why is it slower than FNO?

**A**: Transolver has additional computation in Physics Attention's slice projection/deslicing. However, it provides more features like irregular mesh support. Use FNO for fast inference on regular grids.

### Q: Can I use it on irregular meshes (CFD grids)?

**A**: Yes! Set `structured_shape=None`:
```python
model = Transolver(
    functional_dim=3,        # u, v, p
    out_dim=3,
    embedding_dim=3,         # 3D coordinates (x, y, z)
    structured_shape=None,   # Unstructured mesh
    unified_pos=False,
    use_te=False,
)

# Input: [B, N_nodes, 3] + embedding [B, N_nodes, 3]
# N_nodes is arbitrary (CFD mesh node count)
```

### Q: Want to save the trained model

**A**: Add this code:
```python
# Save
torch.save(model.state_dict(), "E:/physicsnemo_env/tutorial_results/transolver_model.pth")

# Load
model = Transolver(functional_dim=1, out_dim=1, ...)
model.load_state_dict(torch.load("E:/physicsnemo_env/tutorial_results/transolver_model.pth"))
model.eval()
```

---

## Summary

Through this tutorial, we learned:

1. **Transolver architecture**: Transformer + Physics Attention for learning PDE solutions
2. **Physics Attention principle**: Slicing reduces O(N²) → O(N×S) computation
3. **Structured 2D mesh**: Convolution-based slice projection captures spatial structure
4. **Unified position encoding**: Position info generation via reference grid for structured data
5. **use_te=False mode**: Run in pure PyTorch without Transformer Engine
6. **PINN/FNO/Transolver comparison**: Strengths/weaknesses and application scenarios of each model
7. **Darcy Flow prediction**: Achieved loss 1.03 → 0.01 with 500 epochs

### Core Value of Transolver

- **Irregular mesh support**: Handles complex geometry that FNO cannot
- **Efficient attention**: Slicing makes it practical even for large PDE grids
- **Flexibility**: Supports both structured/unstructured, 2D/3D
- **Physical understanding**: Physics Attention learns spatial physical structure
