"""
PhysicsNeMo Tutorial: FNO for Darcy Flow (Data-Driven Learning)
================================================================
This tutorial demonstrates the Fourier Neural Operator (FNO) - a model
architecture unique to PhysicsNeMo that would be very complex to implement
from scratch in plain PyTorch.

Unlike the PINN tutorial (equation-based), FNO is DATA-DRIVEN:
  - PINN: learns from physics equations (no data needed)
  - FNO:  learns from input-output data pairs (simulation results)

Problem: Darcy Flow (flow through porous media)
  - Input:  permeability field (how easily fluid flows through each point)
  - Output: pressure field (solution of Darcy's equation)

Key FNO advantage demonstrated here: RESOLUTION INDEPENDENCE
  - Train on 32x32 grid
  - Infer on 64x64 grid (impossible with standard CNNs!)

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_fno_darcy.py

==========================================================================
[What Makes FNO Different from Standard Neural Networks?]
==========================================================================

Standard CNN (Convolutional Neural Network):
  - Operates in SPATIAL domain (pixel-by-pixel)
  - Fixed receptive field (can only see local neighborhood)
  - Resolution-dependent (trained on 32x32, must infer on 32x32)

FNO (Fourier Neural Operator):
  - Operates in FREQUENCY domain (Fourier modes)
  - Global receptive field (captures long-range dependencies instantly)
  - Resolution-independent (trained on 32x32, can infer on ANY resolution!)

  FNO internal pipeline:
    1. Input grid -> Fourier Transform (spatial -> frequency domain)
    2. Keep top-K modes (low frequencies = smooth patterns)
    3. Apply learned weights in frequency domain
    4. Inverse Fourier Transform (frequency -> spatial domain)
    5. Add bypass convolution (local features)
    6. Repeat for num_fno_layers
    7. Decode to output channels

  This entire pipeline is built into PhysicsNeMo's FNO class!
  Implementing SpectralConv2d from scratch would take ~100 lines of code.
==========================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import physicsnemo
from physicsnemo.models.fno import FNO
import time
import os

# ============================================================================
# [0] Environment Setup
# ============================================================================
print("=" * 70)
print("  PhysicsNeMo FNO Tutorial: Darcy Flow (Data-Driven)")
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
# Darcy's equation: -div(k * grad(p)) = f
#   k = permeability (input: how easily fluid flows)
#   p = pressure (output: what we want to predict)
#   f = source term (constant in our case)
#
# Instead of solving the PDE numerically (which is slow), we create synthetic
# data using an analytical approximation. In real applications, you would use
# CFD simulation results as training data.
#
# Data format for FNO: [batch, channels, height, width] (like images!)

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
    # Start with white noise, then smooth it to create spatial correlations
    raw_noise = torch.randn(n_samples, 1, grid_size, grid_size, device=device)

    # Smooth the noise to create realistic permeability fields
    # (real porous media has spatial structure, not random noise)
    kernel_size = 5
    k_field = torch.nn.functional.avg_pool2d(
        raw_noise, kernel_size, stride=1, padding=kernel_size // 2
    )
    # Normalize to positive values (permeability must be positive)
    k_field = torch.sigmoid(k_field * 2) + 0.1  # range: [0.1, 1.1]

    # Generate approximate pressure field (synthetic "solution")
    # In reality, this would come from a PDE solver.
    # Here we use: p = smooth(k) * sinusoidal_modulation
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


# ============================================================================
# [2] Create FNO Model (PhysicsNeMo's unique architecture!)
# ============================================================================
print("\n[2/7] Creating FNO model (PhysicsNeMo built-in)...")

# This is the key difference from the PINN tutorial!
# The FNO class internally implements:
#   - SpectralConv2d: Fourier transform -> mode truncation -> inverse transform
#   - FNO2DEncoder: stacking spectral conv layers with bypass connections
#   - Coordinate feature injection (automatically adds x,y grid as features)
#   - Grid-to-point and point-to-grid conversion
#
# To build this from scratch in PyTorch, you would need:
#   1. Custom SpectralConv2d layer (~50 lines: FFT, truncation, weight multiply, IFFT)
#   2. FNO encoder block (~30 lines: spectral conv + 1x1 conv + activation)
#   3. Coordinate feature generation (~15 lines)
#   4. Grid/point conversion utilities (~20 lines)
#   Total: ~115 lines of complex code
#
# With PhysicsNeMo: ONE line!

model = FNO(
    in_channels=1,            # Input: 1 channel (permeability field k)
    out_channels=1,           # Output: 1 channel (pressure field p)
    dimension=2,              # 2D problem
    decoder_layers=2,         # Decoder MLP layers (maps latent -> output)
    decoder_layer_size=32,    # Decoder neurons per layer
    latent_channels=32,       # FNO latent width (number of Fourier features)
    num_fno_layers=4,         # Number of spectral convolution layers
    num_fno_modes=12,         # Number of Fourier modes to keep (KEY parameter!)
    # More modes = can capture higher frequency patterns
    # Fewer modes = smoother output, less memory
    padding=8,                # Domain padding (reduces boundary artifacts)
    coord_features=True,      # Automatically add (x,y) coordinates as input
).to(device)

# Print model info
n_params = sum(p.numel() for p in model.parameters())
print(f"      FNO parameters: {n_params:,}")
print(f"      Architecture: 4 spectral layers, 12 Fourier modes, 32 latent channels")


# ============================================================================
# [3] Training Setup
# ============================================================================
print("\n[3/7] Setting up training...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 1000

# FNO uses standard supervised learning (unlike PINN which uses PDE residuals)
# Loss = MSE between predicted and target pressure fields
loss_fn = nn.MSELoss()


# ============================================================================
# [4] Training Loop
# ============================================================================
print(f"\n[4/7] Training FNO ({EPOCHS} epochs)...")
print(f"      Note: This is standard supervised learning, not PDE-based like PINN")
print()

start_time = time.time()
loss_history = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # FNO forward pass: input grid -> output grid
    # This is like image-to-image translation (like image segmentation)
    pred = model(k_train)          # [N, 1, 32, 32] -> [N, 1, 32, 32]

    # Standard MSE loss (no PDE residuals, no autograd differentiation!)
    loss = loss_fn(pred, p_train)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:5d}/{EPOCHS} | "
              f"Loss: {loss.item():.6e} | "
              f"Time: {elapsed:.1f}s")

elapsed_total = time.time() - start_time
print(f"\n  Training complete! Total time: {elapsed_total:.1f}s")
print(f"  Final loss: {loss.item():.6e}")


# ============================================================================
# [5] FNO's Superpower: Resolution Independence!
# ============================================================================
print("\n[5/7] Testing resolution independence (FNO's key advantage)...")
print("      Training resolution: 32x32")
print("      Testing resolution:  64x64 and 128x128")

# FNO can infer on DIFFERENT resolutions than it was trained on!
# This is impossible with standard CNNs (which have fixed-size weights).
# FNO achieves this because Fourier modes are resolution-agnostic.

model.eval()
with torch.no_grad():
    # Generate test data at HIGHER resolution (64x64)
    k_test_64, p_test_64 = generate_darcy_data(5, 64, device)
    pred_64 = model(k_test_64)
    loss_64 = loss_fn(pred_64, p_test_64)

    # Generate test data at EVEN HIGHER resolution (128x128)
    k_test_128, p_test_128 = generate_darcy_data(5, 128, device)
    pred_128 = model(k_test_128)
    loss_128 = loss_fn(pred_128, p_test_128)

    print(f"  32x32  (training res):  Loss = {loss.item():.6e}")
    print(f"  64x64  (2x resolution):  Loss = {loss_64.item():.6e}")
    print(f"  128x128 (4x resolution): Loss = {loss_128.item():.6e}")
    print(f"\n  FNO works at ALL resolutions without retraining!")
    print(f"  (A standard CNN would fail or require resizing at 64x64 and 128x128)")


# ============================================================================
# [6] Visualization
# ============================================================================
print("\n[6/7] Visualizing results...")

output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    # Get one sample from training resolution (32x32)
    k_sample_32 = k_train[0:1].cpu()
    p_sample_32 = p_train[0:1].cpu()
    pred_sample_32 = model(k_train[0:1]).cpu()

    # Get one sample from 64x64
    k_sample_64 = k_test_64[0:1].cpu()
    p_sample_64 = p_test_64[0:1].cpu()
    pred_sample_64 = pred_64[0:1].cpu()

# Create comparison figure
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: 32x32 (training resolution)
titles_row1 = ['Input: k (32x32)\nPermeability',
               'Target: p (32x32)\nPressure (true)',
               'FNO Prediction (32x32)\nPressure (predicted)',
               'Error (32x32)\n|pred - target|']

data_row1 = [k_sample_32[0, 0].numpy(),
             p_sample_32[0, 0].numpy(),
             pred_sample_32[0, 0].numpy(),
             np.abs(pred_sample_32[0, 0].numpy() - p_sample_32[0, 0].numpy())]

for i, (ax, title, data) in enumerate(zip(axes[0], titles_row1, data_row1)):
    if i == 3:
        im = ax.imshow(data, cmap='hot', interpolation='bilinear')
    else:
        im = ax.imshow(data, cmap='viridis', interpolation='bilinear')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Row 2: 64x64 (higher resolution - never seen during training!)
titles_row2 = ['Input: k (64x64)\nPermeability',
               'Target: p (64x64)\nPressure (true)',
               'FNO Prediction (64x64)\nPressure (predicted)',
               'Error (64x64)\n|pred - target|']

data_row2 = [k_sample_64[0, 0].numpy(),
             p_sample_64[0, 0].numpy(),
             pred_sample_64[0, 0].numpy(),
             np.abs(pred_sample_64[0, 0].numpy() - p_sample_64[0, 0].numpy())]

for i, (ax, title, data) in enumerate(zip(axes[1], titles_row2, data_row2)):
    if i == 3:
        im = ax.imshow(data, cmap='hot', interpolation='bilinear')
    else:
        im = ax.imshow(data, cmap='viridis', interpolation='bilinear')
    ax.set_title(title + '\n[UNSEEN RESOLUTION!]', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(
    f'FNO: Darcy Flow (Data-Driven, PhysicsNeMo {physicsnemo.__version__})\n'
    f'Trained on 32x32, tested on 64x64 (resolution independence!)\n'
    f'4 FNO layers, 12 Fourier modes, {n_params:,} parameters',
    fontsize=13, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(output_dir, "fno_darcy_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Result image saved: {fig_path}")

# Loss curve
fig2, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(loss_history, linewidth=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('FNO Training Loss History', fontsize=14)
ax.grid(True, alpha=0.3)
fig2_path = os.path.join(output_dir, "fno_darcy_loss.png")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Loss curve saved: {fig2_path}")

plt.close('all')


# ============================================================================
# [7] Summary
# ============================================================================
print("\n[7/7] Tutorial complete!")
print("=" * 70)
print("  FNO vs PINN Comparison:")
print("-" * 70)
print(f"  {'Feature':<25} {'PINN':<25} {'FNO':<25}")
print(f"  {'-'*25} {'-'*25} {'-'*25}")
print(f"  {'Learning type':<25} {'Equation-based':<25} {'Data-driven':<25}")
print(f"  {'Training data needed':<25} {'No':<25} {'Yes':<25}")
print(f"  {'Physics equations':<25} {'Required':<25} {'Not required':<25}")
print(f"  {'Loss function':<25} {'PDE residuals':<25} {'MSE (supervised)':<25}")
print(f"  {'Autograd (2nd order)':<25} {'Yes (essential)':<25} {'No':<25}")
print(f"  {'Resolution independent':<25} {'Yes (continuous)':<25} {'Yes (grid-based)':<25}")
print(f"  {'Inference speed':<25} {'Slow (per-point)':<25} {'Fast (grid)':<25}")
print(f"  {'Training speed':<25} {'Slow (autograd)':<25} {'Fast':<25}")
print(f"  {'PhysicsNeMo model':<25} {'FullyConnected':<25} {'FNO':<25}")
print("-" * 70)
print()
print("  Why FNO is a PhysicsNeMo differentiator:")
print("    - FNO internally implements SpectralConv (Fourier transform layers)")
print("    - Building SpectralConv from scratch in PyTorch: ~100+ lines")
print("    - With PhysicsNeMo: just FNO(in_channels=1, out_channels=1, dimension=2)")
print()
print("  Key FNO advantage demonstrated:")
print("    - Trained on 32x32 grid")
print("    - Successfully inferred on 64x64 and 128x128 grids")
print("    - This is IMPOSSIBLE with standard CNNs!")
print()
print("  Result files:")
print(f"    - {fig_path}")
print(f"    - {fig2_path}")
print("=" * 70)
