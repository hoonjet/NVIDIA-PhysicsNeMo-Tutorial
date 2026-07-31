"""
PhysicsNeMo Tutorial: Transolver for Darcy Flow (Data-Driven Learning)
=====================================================================
This tutorial demonstrates the Transolver model - a transformer-based neural
operator that uses "Physics Attention" instead of standard self-attention.

Transolver is PhysicsNeMo's cutting-edge architecture for learning PDE solutions:
  - PINN: learns from physics equations (no data needed, slow per-point)
  - FNO:  learns from data, operates in Fourier frequency domain
  - Transolver: learns from data, uses transformer + physics attention

Problem: Darcy Flow (flow through porous media)
  - Input:  permeability field (how easily fluid flows through each point)
  - Output: pressure field (solution of Darcy's equation)

Key Transolver concepts:
  1. Physics Attention: Instead of standard O(N^2) self-attention (too expensive
     for PDE grids with thousands of points), Transolver uses "slicing" to
     project all tokens onto a small number of learned "slices" (e.g., 32).
     Attention is computed among slices only -> O(N * S) instead of O(N^2).

  2. Structured Mesh Support: For 2D grid data, Transolver uses convolution-based
     projections to capture spatial structure before attention.

  3. Unified Position Encoding: For structured grids, a reference grid is used
     to create positional embeddings that capture spatial relationships.

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_transolver_darcy.py

==========================================================================
[What Makes Transolver Different from FNO and PINN?]
==========================================================================

PINN (FullyConnected):
  - Equation-based: learns by minimizing PDE residuals
  - Point-by-point: input (x,y) -> output (u,v,p)
  - No training data needed, but slow inference
  - Good for: inverse problems, sparse data scenarios

FNO (Fourier Neural Operator):
  - Data-driven: learns from input-output pairs
  - Frequency domain: Fourier transform -> mode truncation -> inverse
  - Resolution-independent (train 32x32, infer 64x64)
  - Good for: fast surrogate models, resolution transfer

Transolver (Transformer + Physics Attention):
  - Data-driven: learns from input-output pairs
  - Spatial domain: uses attention mechanism with learned "slices"
  - Can handle both structured grids AND irregular meshes
  - Good for: complex geometries, multi-physics, transfer learning

  Transolver internal pipeline:
    1. Input grid -> Position embedding (unified position or custom)
    2. Preprocess MLP: project to hidden dimension
    3. For each Transolver block:
       a. LayerNorm -> Physics Attention -> residual connection
       b. LayerNorm -> MLP -> residual connection
    4. Physics Attention internals:
       i.   Project input onto "slices" (learned spatial clusters)
       ii.  Compute attention AMONG slices (not among all tokens!)
       iii. Project attention outputs back to token space
    5. Final layer: project to output dimension

  The "slicing" mechanism is the key innovation:
    - Standard transformer: every token attends to every other token -> O(N^2)
    - Transolver: tokens are grouped into S slices, attention is among slices -> O(N*S)
    - For N=1024 tokens, S=32 slices: 1024^2=1M vs 1024*32=33K attention operations
==========================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import physicsnemo
from physicsnemo.models.transolver import Transolver
import time
import os

# ============================================================================
# [0] Environment Setup
# ============================================================================
print("=" * 70)
print("  PhysicsNeMo Transolver Tutorial: Darcy Flow (Data-Driven)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  PyTorch version:    {torch.__version__}")
print(f"  PhysicsNeMo version: {physicsnemo.__version__}")
print(f"  Device:             {device}")
if torch.cuda.is_available():
    print(f"  GPU:                {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU Memory:         {gpu_mem:.1f} GB")
print("=" * 70)
print()

torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# [1] Generate Synthetic Training Data (Darcy Flow)
# ============================================================================
# Same Darcy flow problem as the FNO tutorial:
#   Darcy's equation: -div(k * grad(p)) = f
#   k = permeability (input: how easily fluid flows)
#   p = pressure (output: what we want to predict)
#
# Data format for Transolver (structured 2D):
#   Input:  [batch, H, W, C]  (permeability field, image-like)
#   Output: [batch, H, W, C]  (pressure field, image-like)
#
# Note: Transolver uses [B, H, W, C] format (channels-last),
#   while FNO uses [B, C, H, W] format (channels-first, like PyTorch convs)

print("[1/7] Generating synthetic training data...")

N_TRAIN = 200      # Number of training samples
N_GRID = 32        # Grid resolution for training (32x32)

def generate_darcy_data(n_samples, grid_size, device):
    """
    Generate synthetic Darcy flow data.

    Input (permeability k): random Gaussian random field (smooth patterns)
    Output (pressure p): approximate solution (smoothed + modulated version)

    In practice, you would replace this with actual PDE solver output.
    """
    # Generate random permeability fields using Gaussian smoothing
    raw_noise = torch.randn(n_samples, 1, grid_size, grid_size, device=device)

    # Smooth the noise to create realistic permeability fields
    kernel_size = 5
    k_field = torch.nn.functional.avg_pool2d(
        raw_noise, kernel_size, stride=1, padding=kernel_size // 2
    )
    # Normalize to positive values (permeability must be positive)
    k_field = torch.sigmoid(k_field * 2) + 0.1  # range: [0.1, 1.1]

    # Generate approximate pressure field (synthetic "solution")
    p_smooth = torch.nn.functional.avg_pool2d(
        k_field, kernel_size, stride=1, padding=kernel_size // 2
    )

    # Add spatial structure (sine waves at different frequencies)
    x_coords = torch.linspace(0, 1, grid_size, device=device).view(1, 1, 1, -1)
    y_coords = torch.linspace(0, 1, grid_size, device=device).view(1, 1, -1, 1)
    spatial_mod = torch.sin(np.pi * x_coords) * torch.sin(np.pi * y_coords)

    # Pressure = smoothed permeability modulated by spatial structure
    p_field = p_smooth * spatial_mod * 3.0

    return k_field, p_field

# Generate training data
k_train, p_train = generate_darcy_data(N_TRAIN, N_GRID, device)
print(f"      Training data shape: input {k_train.shape}, output {p_train.shape}")
print(f"      (batch={N_TRAIN}, channels=1, grid={N_GRID}x{N_GRID})")

# Transolver expects [B, H, W, C] format (channels-last)
# FNO uses [B, C, H, W] (channels-first)
# We need to convert: [B, 1, H, W] -> [B, H, W, 1]
k_train_transolver = k_train.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 1]
p_train_transolver = p_train.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 1]
print(f"      Transolver format:  input {k_train_transolver.shape}, output {p_train_transolver.shape}")


# ============================================================================
# [2] Create Transolver Model (PhysicsNeMo's Transformer for PDEs!)
# ============================================================================
print("\n[2/7] Creating Transolver model (PhysicsNeMo built-in)...")

# ==========================================================================
# Transolver Architecture Explained:
# ==========================================================================
#
# Key parameters:
#
# functional_dim (int):
#   The number of input channels (physical variables).
#   For Darcy flow: 1 (permeability field)
#   This does NOT include position embeddings - those are added internally.
#
# out_dim (int):
#   The number of output channels.
#   For Darcy flow: 1 (pressure field)
#
# embedding_dim (int):
#   Dimension of position embeddings. Required when unified_pos=False.
#   For irregular meshes, this is the dimension of your custom position encoding.
#   When unified_pos=True, this is computed automatically from ref.
#
# n_layers (int, default=4):
#   Number of Transolver blocks (transformer encoder layers with physics attention).
#   More layers = more expressive power, but slower and more memory.
#
# n_hidden (int, default=256):
#   Hidden dimension of the transformer.
#   This is the "width" of the model. Larger = more capacity.
#   Must be divisible by n_head.
#
# n_head (int, default=8):
#   Number of attention heads in each physics attention layer.
#   Multi-head attention allows the model to attend to different aspects simultaneously.
#
# slice_num (int, default=32):
#   THE KEY PARAMETER! Number of "slices" in physics attention.
#   - Each slice is a learned spatial cluster
#   - Attention is computed among slices, not among all tokens
#   - Smaller slice_num = faster but less expressive
#   - Larger slice_num = more expressive but slower
#   - Think of it as: "how many spatial regions should the model attend to?"
#
# unified_pos (bool, default=False):
#   Whether to use unified positional encoding (for structured grids).
#   When True, a reference grid is used to create position embeddings.
#   Only available for structured data (2D or 3D grids).
#
# ref (int, default=8):
#   Reference grid size for unified position encoding.
#   Creates an ref x ref reference grid for position computation.
#   Only used when unified_pos=True.
#
# structured_shape (tuple, default=None):
#   Shape of the structured grid. For 2D: (H, W). For 3D: (H, W, D).
#   When provided, enables structured mesh physics attention (uses convolutions).
#   When None, uses irregular mesh attention (pure linear projections).
#
# use_te (bool, default=True):
#   Whether to use Transformer Engine (NVIDIA's optimized transformer library).
#   We set this to False to use pure PyTorch (no TE dependency needed).
# ==========================================================================

model = Transolver(
    functional_dim=1,           # Input: 1 channel (permeability field k)
    out_dim=1,                   # Output: 1 channel (pressure field p)
    n_layers=4,                  # 4 Transolver blocks
    n_hidden=128,                # Hidden dimension (smaller for tutorial speed)
    n_head=8,                    # 8 attention heads (128 / 8 = 16 dim per head)
    dropout=0.0,                 # No dropout for deterministic training
    act="gelu",                  # GELU activation (standard for transformers)
    mlp_ratio=4,                 # MLP expansion ratio (standard for transformers)
    slice_num=32,                # 32 learned slices (key physics attention param!)
    unified_pos=True,            # Use unified position encoding (structured grid)
    ref=8,                       # 8x8 reference grid for position encoding
    structured_shape=(N_GRID, N_GRID),  # 32x32 structured grid
    use_te=False,                # Use pure PyTorch (no Transformer Engine needed)
).to(device)

# Print model info
n_params = sum(p.numel() for p in model.parameters())
print(f"      Transolver parameters: {n_params:,}")
print(f"      Architecture: 4 layers, 128 hidden dim, 8 heads, 32 slices")
print(f"      Position encoding: unified (8x8 reference grid)")
print(f"      Backend: Pure PyTorch (use_te=False)")


# ============================================================================
# [3] Training Setup
# ============================================================================
print("\n[3/7] Setting up training...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 500

# Transolver uses standard supervised learning (like FNO, unlike PINN)
# Loss = MSE between predicted and target pressure fields
loss_fn = nn.MSELoss()


# ============================================================================
# [4] Training Loop
# ============================================================================
print(f"\n[4/7] Training Transolver ({EPOCHS} epochs)...")
print(f"      Note: This is standard supervised learning, not PDE-based like PINN")
print()

start_time = time.time()
loss_history = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # Transolver forward pass: [B, H, W, C] -> [B, H, W, C]
    # The model internally:
    #   1. Adds unified position embeddings to the input
    #   2. Projects to hidden dimension via preprocess MLP
    #   3. Passes through 4 Transolver blocks (physics attention + MLP)
    #   4. Final block projects to output dimension
    pred = model(k_train_transolver)  # [N, 32, 32, 1] -> [N, 32, 32, 1]

    # Standard MSE loss (no PDE residuals, no autograd differentiation!)
    loss = loss_fn(pred, p_train_transolver)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:5d}/{EPOCHS} | "
              f"Loss: {loss.item():.6e} | "
              f"Time: {elapsed:.1f}s")

elapsed_total = time.time() - start_time
print(f"\n  Training complete! Total time: {elapsed_total:.1f}s")
print(f"  Final loss: {loss.item():.6e}")


# ============================================================================
# [5] Test on Different Resolution (Transolver's Flexibility!)
# ============================================================================
print("\n[5/7] Testing on different resolution...")

# Unlike FNO (which is naturally resolution-independent via Fourier modes),
# Transolver's structured mesh attention uses convolutions with fixed kernel sizes.
# However, Transolver can still handle different resolutions by reshaping.
# The key advantage of Transolver is its ability to handle IRREGULAR meshes,
# which FNO cannot do at all!

model.eval()
with torch.no_grad():
    # Test on training resolution (32x32)
    pred_32 = model(k_train_transolver[:5])
    loss_32 = loss_fn(pred_32, p_train_transolver[:5])

    # Test on higher resolution (64x64) - need a new model for structured mesh
    # because the structured_shape is fixed. But Transolver with irregular mesh
    # mode can handle variable resolutions!
    k_test_64, p_test_64 = generate_darcy_data(5, 64, device)
    k_test_64_t = k_test_64.permute(0, 2, 3, 1).contiguous()
    p_test_64_t = p_test_64.permute(0, 2, 3, 1).contiguous()

    # For structured mesh Transolver, we can't directly infer on different
    # resolution because the conv layers expect the same spatial shape.
    # This is a key difference from FNO!
    # To handle variable resolution, use irregular mesh mode (structured_shape=None)
    print(f"  32x32  (training res):  Loss = {loss_32.item():.6e}")
    print(f"\n  Note: Transolver with structured_shape=(32,32) is fixed to 32x32.")
    print(f"  For variable resolution, use irregular mesh mode (structured_shape=None)")
    print(f"  or create a new model with the target resolution's structured_shape.")
    print(f"\n  Transolver's key advantage over FNO: handles IRREGULAR MESHES!")
    print(f"  FNO requires regular grids; Transolver works on any point cloud.")


# ============================================================================
# [6] Visualization
# ============================================================================
print("\n[6/7] Visualizing results...")

output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    # Get one sample for visualization
    k_sample = k_train[0:1].cpu()                    # [1, 1, 32, 32] (channels-first)
    p_sample = p_train[0:1].cpu()                    # [1, 1, 32, 32]
    pred_sample = model(k_train_transolver[0:1]).cpu()  # [1, 32, 32, 1]
    pred_sample = pred_sample.permute(0, 3, 1, 2)   # [1, 1, 32, 32] (back to channels-first)

# Create comparison figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

titles = ['Input: k (permeability)\n32x32',
          'Target: p (pressure, true)\n32x32',
          'Transolver Prediction\n32x32',
          'Error |pred - target|\n32x32']

data_list = [k_sample[0, 0].numpy(),
              p_sample[0, 0].numpy(),
              pred_sample[0, 0].numpy(),
              np.abs(pred_sample[0, 0].numpy() - p_sample[0, 0].numpy())]

for i, (ax, title, data) in enumerate(zip(axes, titles, data_list)):
    if i == 3:
        im = ax.imshow(data, cmap='hot', interpolation='bilinear')
    else:
        im = ax.imshow(data, cmap='viridis', interpolation='bilinear')
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(
    f'Transolver: Darcy Flow (Data-Driven, PhysicsNeMo {physicsnemo.__version__})\n'
    f'4 layers, 128 hidden, 8 heads, 32 slices, {n_params:,} parameters\n'
    f'Physics Attention with unified position encoding (8x8 ref grid)',
    fontsize=13, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(output_dir, "transolver_darcy_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Result image saved: {fig_path}")

# Loss curve
fig2, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(loss_history, linewidth=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('Transolver Training Loss History', fontsize=14)
ax.grid(True, alpha=0.3)
fig2_path = os.path.join(output_dir, "transolver_darcy_loss.png")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Loss curve saved: {fig2_path}")

plt.close('all')


# ============================================================================
# [7] Summary
# ============================================================================
print("\n[7/7] Tutorial complete!")
print("=" * 70)
print("  Transolver vs FNO vs PINN Comparison:")
print("-" * 70)
print(f"  {'Feature':<28} {'PINN':<18} {'FNO':<18} {'Transolver':<18}")
print(f"  {'-'*28} {'-'*18} {'-'*18} {'-'*18}")
print(f"  {'Learning type':<28} {'Equation-based':<18} {'Data-driven':<18} {'Data-driven':<18}")
print(f"  {'Training data needed':<28} {'No':<18} {'Yes':<18} {'Yes':<18}")
print(f"  {'Core mechanism':<28} {'Autograd':<18} {'Fourier':<18} {'Attention':<18}")
print(f"  {'Attention type':<28} {'N/A':<18} {'N/A':<18} {'Physics Attn':<18}")
print(f"  {'Resolution independent':<28} {'Yes':<18} {'Yes':<18} {'No (fixed)':<18}")
print(f"  {'Irregular mesh support':<28} {'Yes':<18} {'No':<18} {'Yes!':<18}")
print(f"  {'Complex geometry':<28} {'Good':<18} {'Poor':<18} {'Excellent':<18}")
print(f"  {'Memory usage':<28} {'Low':<18} {'Medium':<18} {'High':<18}")
print(f"  {'Training speed':<28} {'Slow':<18} {'Fast':<18} {'Medium':<18}")
print(f"  {'PhysicsNeMo model':<28} {'FullyConnected':<18} {'FNO':<18} {'Transolver':<18}")
print("-" * 70)
print()
print("  Key Transolver concepts demonstrated:")
print("    1. Physics Attention: O(N*S) instead of O(N^2) standard attention")
print("       N=tokens (1024 for 32x32), S=slices (32) -> 33K vs 1M operations")
print("    2. Unified Position Encoding: structured grid positions via ref grid")
print("    3. Structured Mesh 2D: convolution-based slice projection")
print("    4. use_te=False: pure PyTorch mode (no Transformer Engine needed)")
print()
print("  When to use Transolver vs FNO:")
print("    - Regular grid, need resolution transfer -> FNO")
print("    - Irregular mesh, complex geometry -> Transolver")
print("    - Need both? Transolver with structured_shape=None (irregular mode)")
print()
print("  Result files:")
print(f"    - {fig_path}")
print(f"    - {fig2_path}")
print("=" * 70)
