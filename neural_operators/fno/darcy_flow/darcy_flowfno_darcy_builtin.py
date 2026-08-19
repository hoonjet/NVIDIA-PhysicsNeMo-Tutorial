"""
PhysicsNeMo Tutorial: FNO for Darcy Flow with Built-in GPU PDE Solver
=====================================================================
This tutorial uses PhysicsNeMo's built-in Darcy2D DataPipe to generate REAL
PDE solutions on the GPU, then trains an FNO model on that data.

Key difference from tutorial_fno_darcy.py:
  - Previous tutorial: synthetic data (Gaussian smoothing = fake "PDE solution")
  - This tutorial: REAL PDE solutions from PhysicsNeMo's GPU solver (NVIDIA Warp)

The Darcy2D DataPipe:
  1. Generates random permeability fields via Fourier series
  2. Thresholds to create piecewise constant permeability
  3. Solves the Darcy equation using a multi-grid Jacobi iterative method
  4. All on GPU via NVIDIA Warp kernels (not possible in plain PyTorch!)

This is PhysicsNeMo's key differentiator: model (FNO) + data pipeline (Darcy2D).

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_fno_darcybuiltin.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import physicsnemo
from physicsnemo.models.fno import FNO
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
import time
import os

# ============================================================================
# [0] Environment Setup
# ============================================================================
print("=" * 70)
print("  PhysicsNeMo FNO + Darcy2D DataPipe Tutorial")
print("  (Built-in GPU PDE Solver for Real Training Data)")
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
# [1] Create Darcy2D DataPipe (PhysicsNeMo's Built-in GPU PDE Solver)
# ============================================================================
# The Darcy2D DataPipe solves the 2D Darcy equation on the GPU:
#   -div(k * grad(p)) = f
#   k = permeability (random, piecewise constant)
#   p = pressure (solution obtained by multi-grid Jacobi iteration)
#
# This is REAL PDE data, not synthetic! The solver runs entirely on GPU
# using NVIDIA Warp kernels. Each call to the iterator generates a new
# batch with different random permeability fields.
#
# Resolution must be divisible by 2^(nr_multigrids-1).
# With nr_multigrids=4, resolution must be divisible by 8.

print("[1/7] Creating Darcy2D DataPipe (built-in GPU PDE solver)...")
print("      This datapipe generates REAL Darcy flow solutions on GPU!")
print()

RESOLUTION = 64          # 64x64 grid (divisible by 8 for 4-level multigrid)
BATCH_SIZE = 8           # Batch size per data generation (GPU memory limited)
N_TRAIN_BATCHES = 25     # Generate 25 batches -> 200 training samples
N_VAL_BATCHES = 5        # Generate 5 batches -> 40 validation samples
# Total: 200 train + 40 val = 240 real PDE solutions

# Create the datapipe with reduced iteration count for speed
# (default max_iterations=30000 is too slow for a tutorial)
darcy_datapipe = Darcy2D(
    resolution=RESOLUTION,
    batch_size=BATCH_SIZE,
    nr_permeability_freq=5,       # Number of Fourier frequencies for permeability
    max_permeability=2.0,          # Max permeability value
    min_permeability=0.5,          # Min permeability value
    max_iterations=5000,           # Reduced from 30000 for tutorial speed
    convergence_threshold=1e-6,    # Solver convergence threshold
    iterations_per_convergence_check=500,  # Check convergence every 500 iterations
    nr_multigrids=4,               # 4-level multigrid (resolution must be div by 8)
    device=device,
)

print(f"      Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"      Batch size: {BATCH_SIZE}")
print(f"      Train batches: {N_TRAIN_BATCHES} ({N_TRAIN_BATCHES * BATCH_SIZE} samples)")
print(f"      Val batches:   {N_VAL_BATCHES} ({N_VAL_BATCHES * BATCH_SIZE} samples)")
print(f"      Multigrid levels: 4 (GPU-accelerated Jacobi solver)")
print()


# ============================================================================
# [2] Generate Training Data from the PDE Solver
# ============================================================================
print("[2/7] Generating real PDE solutions using GPU solver...")
print("      (Each sample = random permeability -> solved Darcy equation)")
print()

start_time = time.time()
data_iter = iter(darcy_datapipe)

# Collect training data
k_train_list = []
p_train_list = []
for i in range(N_TRAIN_BATCHES):
    batch = next(data_iter)
    k_train_list.append(batch["permeability"].clone())
    p_train_list.append(batch["darcy"].clone())
    if (i + 1) % 5 == 0:
        elapsed = time.time() - start_time
        print(f"      Train batch {i+1}/{N_TRAIN_BATCHES} "
              f"({(i+1)*BATCH_SIZE} samples) | Time: {elapsed:.1f}s")

# Collect validation data
k_val_list = []
p_val_list = []
for i in range(N_VAL_BATCHES):
    batch = next(data_iter)
    k_val_list.append(batch["permeability"].clone())
    p_val_list.append(batch["darcy"].clone())

k_train = torch.cat(k_train_list, dim=0)  # [N_train, 1, 64, 64]
p_train = torch.cat(p_train_list, dim=0)  # [N_train, 1, 64, 64]
k_val = torch.cat(k_val_list, dim=0)      # [N_val, 1, 64, 64]
p_val = torch.cat(p_val_list, dim=0)      # [N_val, 1, 64, 64]

data_time = time.time() - start_time
print(f"\n      Data generation complete! Time: {data_time:.1f}s")
print(f"      Train: k={k_train.shape}, p={p_train.shape}")
print(f"      Val:   k={k_val.shape}, p={p_val.shape}")
print(f"      Permeability range: [{k_train.min():.3f}, {k_train.max():.3f}]")
print(f"      Pressure range:     [{p_train.min():.3f}, {p_train.max():.3f}]")

# Free the datapipe (GPU memory)
del darcy_datapipe, data_iter
torch.cuda.empty_cache()


# ============================================================================
# [3] Create FNO Model
# ============================================================================
print("\n[3/7] Creating FNO model (PhysicsNeMo built-in)...")

model = FNO(
    in_channels=1,            # Input: 1 channel (permeability field k)
    out_channels=1,           # Output: 1 channel (pressure field p)
    dimension=2,              # 2D problem
    decoder_layers=2,         # Decoder MLP layers
    decoder_layer_size=32,    # Decoder neurons per layer
    latent_channels=32,       # FNO latent width
    num_fno_layers=4,         # Number of spectral convolution layers
    num_fno_modes=12,         # Number of Fourier modes to keep
    padding=8,                # Domain padding
    coord_features=True,      # Add (x,y) coordinates as input features
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"      FNO parameters: {n_params:,}")
print(f"      Architecture: 4 spectral layers, 12 Fourier modes, 32 latent channels")


# ============================================================================
# [4] Training Setup
# ============================================================================
print("\n[4/7] Setting up training...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 500
loss_fn = nn.MSELoss()

# Use mini-batches for training
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 40

train_dataset = torch.utils.data.TensorDataset(k_train, p_train)
val_dataset = torch.utils.data.TensorDataset(k_val, p_val)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False
)


# ============================================================================
# [5] Training Loop
# ============================================================================
print(f"\n[5/7] Training FNO on real PDE data ({EPOCHS} epochs)...")
print()

start_time = time.time()
loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for k_batch, p_batch in train_loader:
        optimizer.zero_grad()
        pred = model(k_batch)
        loss = loss_fn(pred, p_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    loss_history.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for k_batch, p_batch in val_loader:
            pred = model(k_batch)
            val_loss += loss_fn(pred, p_batch).item()
    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:4d}/{EPOCHS} | "
              f"Train: {train_loss:.6e} | "
              f"Val: {val_loss:.6e} | "
              f"Time: {elapsed:.1f}s")

elapsed_total = time.time() - start_time
print(f"\n  Training complete! Total time: {elapsed_total:.1f}s")
print(f"  Final train loss: {train_loss:.6e}")
print(f"  Final val loss:   {val_loss:.6e}")

# Save model
output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "fno_darcy_builtin_model.pth")
torch.save(model.state_dict(), model_path)
print(f"  Model saved: {model_path}")


# ============================================================================
# [6] Visualization
# ============================================================================
print("\n[6/7] Visualizing results...")

model.eval()
with torch.no_grad():
    # Get a few validation samples for visualization
    k_vis = k_val[:4].cpu()
    p_vis = p_val[:4].cpu()
    pred_vis = model(k_val[:4]).cpu()

    # Compute relative L2 error for each sample
    errors = []
    for i in range(4):
        err = torch.norm(pred_vis[i] - p_vis[i]).item() / torch.norm(p_vis[i]).item()
        errors.append(err)

# Create figure: 4 rows x 3 columns (permeability, true pressure, predicted pressure)
fig, axes = plt.subplots(4, 3, figsize=(12, 14))

for i in range(4):
    # Permeability (input)
    ax = axes[i, 0]
    im = ax.imshow(k_vis[i, 0].numpy(), cmap='viridis', interpolation='bilinear')
    ax.set_title(f'Sample {i+1}: Input k (permeability)', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # True pressure (PDE solver output)
    ax = axes[i, 1]
    im = ax.imshow(p_vis[i, 0].numpy(), cmap='jet', interpolation='bilinear')
    ax.set_title(f'Sample {i+1}: True p (GPU PDE solver)', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # FNO prediction
    ax = axes[i, 2]
    im = ax.imshow(pred_vis[i, 0].numpy(), cmap='jet', interpolation='bilinear')
    ax.set_title(f'Sample {i+1}: FNO pred (rel.L2={errors[i]:.3f})', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(
    f'FNO + Darcy2D DataPipe: Real PDE Solutions\n'
    f'Darcy equation solved on GPU via NVIDIA Warp (multi-grid Jacobi)\n'
    f'PhysicsNeMo {physicsnemo.__version__} | {device} | '
    f'{k_train.shape[0]} train + {k_val.shape[0]} val samples | '
    f'{EPOCHS} epochs',
    fontsize=13, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(output_dir, "fno_darcy_builtin_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Result image saved: {fig_path}")

# Loss curve
fig2, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(loss_history, linewidth=1.0, label='Train loss', alpha=0.8)
ax.semilogy(val_loss_history, linewidth=1.0, label='Val loss', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('FNO Training Loss (Real Darcy PDE Data)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig2_path = os.path.join(output_dir, "fno_darcy_builtin_loss.png")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Loss curve saved: {fig2_path}")

plt.close('all')


# ============================================================================
# [7] Summary
# ============================================================================
print("\n[7/7] Tutorial complete!")
print("=" * 70)
print("  FNO + Darcy2D DataPipe Tutorial Summary:")
print(f"    - Data source: PhysicsNeMo Darcy2D DataPipe (GPU PDE solver)")
print(f"    - PDE solver: Multi-grid Jacobi (NVIDIA Warp kernels)")
print(f"    - Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"    - Train samples: {k_train.shape[0]} (real PDE solutions)")
print(f"    - Val samples: {k_val.shape[0]}")
print(f"    - Model: FNO (4 layers, 12 modes, {n_params:,} params)")
print(f"    - Training time: {elapsed_total:.1f}s")
print(f"    - Final train loss: {train_loss:.6e}")
print(f"    - Final val loss: {val_loss:.6e}")
print(f"    - Device: {device}")
print()
print("  Key difference from previous FNO tutorial:")
print("    Previous: Synthetic data (Gaussian smoothing = fake PDE solution)")
print("    This:     REAL PDE solutions from GPU solver (multi-grid Jacobi)")
print()
print("  Comparison of all tutorials:")
print(f"    {'Tutorial':<30} {'Data source':<25} {'Model':<20}")
print(f"    {'-'*30} {'-'*25} {'-'*20}")
print(f"    {'PINN LDC2D':<30} {'Equation (autograd)':<25} {'FullyConnected':<20}")
print(f"    {'PINN CHT2D':<30} {'Equation (autograd)':<25} {'FullyConnected':<20}")
print(f"    {'FNO Darcy (synthetic)':<30} {'Gaussian smoothing':<25} {'FNO':<20}")
print(f"    {'FNO Darcy (builtin)':<30} {'GPU PDE solver (Warp)':<25} {'FNO':<20}")
print()
print("  Result files:")
print(f"    - {fig_path}")
print(f"    - {fig2_path}")
print("=" * 70)
