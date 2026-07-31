"""
PhysicsNeMo Tutorial: U-Net for Darcy Flow (Data-Driven)
=========================================================
This tutorial demonstrates the U-Net model - a classic CNN architecture
with encoder-decoder structure and skip connections.

U-Net is fundamentally different from all previous models:
  - PINN: equation-based, point-by-point
  - FNO: data-driven, Fourier frequency domain
  - Transolver: data-driven, transformer + physics attention
  - MeshGraphNet: data-driven, graph message passing
  - U-Net: data-driven, CNN encoder-decoder + skip connections

Problem: Darcy Flow (same as FNO/Transolver tutorials)
  - Input:  permeability field (how easily fluid flows)
  - Output: pressure field (solution of Darcy's equation)

Note: PhysicsNeMo's U-Net uses 3D convolutions (Conv3d). We convert our
2D data to 3D by replicating across a depth dimension.

Key U-Net concepts:
  1. Encoder: progressively downsamples (conv + pool), captures context
  2. Decoder: progressively upsamples (deconv), enables precise localization
  3. Skip Connections: connect encoder features to decoder (U-shape!)
     - Preserves fine-grained details lost during downsampling
     - The key innovation that makes U-Net effective

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_unet_darcy.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import physicsnemo
from physicsnemo.models.unet import UNet
import time
import os

# ============================================================================
# [0] Environment Setup
# ============================================================================
print("=" * 70)
print("  PhysicsNeMo U-Net Tutorial: Darcy Flow (Data-Driven)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  PyTorch version:    {torch.__version__}")
print(f"  PhysicsNeMo version: {physicsnemo.__version__}")
print(f"  Device:             {device}")
if torch.cuda.is_available():
    print(f"  GPU:                {torch.cuda.get_device_name(0)}")
print("=" * 70)
print()

torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# [1] Generate Synthetic Training Data (Darcy Flow)
# ============================================================================
print("[1/7] Generating synthetic training data...")

N_TRAIN = 200
N_GRID = 32
DEPTH = 4  # 3D depth (PhysicsNeMo U-Net uses Conv3d, so we need 3D input)
# 2D data [B,1,32,32] -> 3D data [B,1,32,32,4] by replicating

def generate_darcy_data(n_samples, grid_size, device):
    """Generate synthetic Darcy flow data (same as FNO tutorial)."""
    raw_noise = torch.randn(n_samples, 1, grid_size, grid_size, device=device)
    kernel_size = 5
    k_field = torch.nn.functional.avg_pool2d(
        raw_noise, kernel_size, stride=1, padding=kernel_size // 2
    )
    k_field = torch.sigmoid(k_field * 2) + 0.1

    p_smooth = torch.nn.functional.avg_pool2d(
        k_field, kernel_size, stride=1, padding=kernel_size // 2
    )
    x_coords = torch.linspace(0, 1, grid_size, device=device).view(1, 1, 1, -1)
    y_coords = torch.linspace(0, 1, grid_size, device=device).view(1, 1, -1, 1)
    spatial_mod = torch.sin(np.pi * x_coords) * torch.sin(np.pi * y_coords)
    p_field = p_smooth * spatial_mod * 3.0
    return k_field, p_field

k_train_2d, p_train_2d = generate_darcy_data(N_TRAIN, N_GRID, device)

# Convert 2D -> 3D: [B, C, H, W] -> [B, C, H, W, D]
k_train = k_train_2d.unsqueeze(-1).expand(-1, -1, -1, -1, DEPTH).contiguous()
p_train = p_train_2d.unsqueeze(-1).expand(-1, -1, -1, -1, DEPTH).contiguous()
print(f"      2D data: {k_train_2d.shape} -> 3D data: {k_train.shape}")
print(f"      (depth={DEPTH} for 3D Conv, 2D data replicated)")


# ============================================================================
# [2] Create U-Net Model
# ============================================================================
print("\n[2/7] Creating U-Net model...")

# U-Net Architecture:
#   Encoder (downsampling):  Conv3d + Pool3d, captures context at multiple scales
#   Bottleneck:              deepest layer with most channels
#   Decoder (upsampling):    ConvTranspose3d, reconstructs spatial resolution
#   Skip Connections:        encoder features -> decoder (concatenation)
#
#   The "U" shape:
#     32x32x4 --conv--> 32x32x4 --pool--> 16x16x2 --conv--> 16x16x2 --pool--> 8x8x1
#        |                                                              |
#        |___________skip___________     _______________________________|
#                                   |   |
#                                   v   v
#     32x32x4 <--deconv-- 16x16x2 <--deconv-- 8x8x1 (bottleneck)
#
#   feature_map_channels: [32, 32, 64, 64] for model_depth=2
#   - Level 0: 32 channels (2 conv blocks)
#   - Level 1: 64 channels (2 conv blocks, bottleneck)

model = UNet(
    in_channels=1,              # Input: 1 channel (permeability)
    out_channels=1,             # Output: 1 channel (pressure)
    model_depth=2,              # 2 levels (shallow for tutorial speed)
    feature_map_channels=[32, 32, 64, 64],  # 2 levels x 2 conv blocks
    num_conv_blocks=2,          # 2 conv blocks per level
    conv_activation="relu",     # ReLU activation
    pooling_type="MaxPool3d",   # Max pooling for downsampling
    pool_size=2,                # Pool by factor of 2
    normalization="groupnorm",  # Group normalization
    gradient_checkpointing=False,  # Disable checkpointing (small model)
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"      U-Net parameters: {n_params:,}")
print(f"      Architecture: 2-level encoder-decoder, skip connections")
print(f"      Input: [B, 1, 32, 32, {DEPTH}] -> Output: [B, 1, 32, 32, {DEPTH}]")


# ============================================================================
# [3] Training Setup
# ============================================================================
print("\n[3/7] Setting up training...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 300
loss_fn = nn.MSELoss()


# ============================================================================
# [4] Training Loop
# ============================================================================
print(f"\n[4/7] Training U-Net ({EPOCHS} epochs)...")

start_time = time.time()
loss_history = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    pred = model(k_train)  # [N, 1, 32, 32, 4] -> [N, 1, 32, 32, 4]
    loss = loss_fn(pred, p_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 30 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:5d}/{EPOCHS} | Loss: {loss.item():.6e} | Time: {elapsed:.1f}s")

elapsed_total = time.time() - start_time
print(f"\n  Training complete! Total time: {elapsed_total:.1f}s")
print(f"  Final loss: {loss.item():.6e}")


# ============================================================================
# [5] Test on Different Resolution
# ============================================================================
print("\n[5/7] Testing resolution independence...")

model.eval()
with torch.no_grad():
    # Test on training resolution (32x32)
    pred_32 = model(k_train[:5])
    loss_32 = loss_fn(pred_32, p_train[:5])

    # Test on 64x64 (higher resolution)
    k_test_64_2d, p_test_64_2d = generate_darcy_data(5, 64, device)
    k_test_64 = k_test_64_2d.unsqueeze(-1).expand(-1, -1, -1, -1, DEPTH).contiguous()
    p_test_64 = p_test_64_2d.unsqueeze(-1).expand(-1, -1, -1, -1, DEPTH).contiguous()
    pred_64 = model(k_test_64)
    loss_64 = loss_fn(pred_64, p_test_64)

    print(f"  32x32 (training res): Loss = {loss_32.item():.6e}")
    print(f"  64x64 (2x resolution): Loss = {loss_64.item():.6e}")
    print(f"\n  Note: U-Net uses fixed-size convolutions, so it CAN process")
    print(f"  different resolutions (unlike FNO which is naturally resolution-independent).")
    print(f"  However, skip connections require matching encoder/decoder shapes,")
    print(f"  so the same model works at any resolution divisible by 2^(model_depth-1).")


# ============================================================================
# [6] Visualization
# ============================================================================
print("\n[6/7] Visualizing results...")

output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    k_sample = k_train_2d[0:1].cpu()
    p_sample = p_train_2d[0:1].cpu()
    pred_sample = model(k_train[0:1]).cpu()
    pred_sample_2d = pred_sample[0, 0, :, :, 0]  # Take first depth slice

    # Also get 64x64 prediction
    pred_64_sample = pred_64[0:1].cpu()
    pred_64_2d = pred_64_sample[0, 0, :, :, 0]
    k_64_sample = k_test_64_2d[0:1].cpu()
    p_64_sample = p_test_64_2d[0:1].cpu()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: 32x32
titles_r1 = ['Input: k (32x32)\nPermeability',
             'Target: p (32x32)\nPressure (true)',
             'U-Net Prediction (32x32)\nPressure (predicted)',
             'Error (32x32)\n|pred - target|']
data_r1 = [k_sample[0, 0].numpy(), p_sample[0, 0].numpy(),
           pred_sample_2d.numpy(), np.abs(pred_sample_2d.numpy() - p_sample[0, 0].numpy())]
for i, (ax, title, data) in enumerate(zip(axes[0], titles_r1, data_r1)):
    im = ax.imshow(data, cmap='hot' if i == 3 else 'viridis', interpolation='bilinear')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Row 2: 64x64
titles_r2 = ['Input: k (64x64)\nPermeability',
             'Target: p (64x64)\nPressure (true)',
             'U-Net Prediction (64x64)\nPressure (predicted)',
             'Error (64x64)\n|pred - target|']
data_r2 = [k_64_sample[0, 0].numpy(), p_64_sample[0, 0].numpy(),
           pred_64_2d.numpy(), np.abs(pred_64_2d.numpy() - p_64_sample[0, 0].numpy())]
for i, (ax, title, data) in enumerate(zip(axes[1], titles_r2, data_r2)):
    im = ax.imshow(data, cmap='hot' if i == 3 else 'viridis', interpolation='bilinear')
    ax.set_title(title + '\n[UNSEEN RESOLUTION!]', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(
    f'U-Net: Darcy Flow (Data-Driven, PhysicsNeMo {physicsnemo.__version__})\n'
    f'2-level encoder-decoder, skip connections, {n_params:,} parameters\n'
    f'Trained on 32x32, tested on 64x64 (resolution flexible!)',
    fontsize=13, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(output_dir, "unet_darcy_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Result image saved: {fig_path}")

# Loss curve
fig2, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(loss_history, linewidth=0.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('U-Net Training Loss History', fontsize=14)
ax.grid(True, alpha=0.3)
fig2_path = os.path.join(output_dir, "unet_darcy_loss.png")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Loss curve saved: {fig2_path}")
plt.close('all')


# ============================================================================
# [7] Summary
# ============================================================================
print("\n[7/7] Tutorial complete!")
print("=" * 70)
print("  5-Model Comparison: PINN vs FNO vs Transolver vs MGN vs UNet")
print("-" * 70)
print(f"  {'Feature':<24} {'PINN':<12} {'FNO':<12} {'Transolver':<12} {'MGN':<12} {'UNet':<12}")
print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'Learning type':<24} {'Equation':<12} {'Data':<12} {'Data':<12} {'Data':<12} {'Data':<12}")
print(f"  {'Data structure':<24} {'Points':<12} {'Grid':<12} {'Grid/Mesh':<12} {'Graph':<12} {'Grid':<12}")
print(f"  {'Core mechanism':<24} {'Autograd':<12} {'Fourier':<12} {'Attention':<12} {'MsgPass':<12} {'Conv':<12}")
print(f"  {'Skip connections':<24} {'No':<12} {'No':<12} {'Residual':<12} {'No':<12} {'Yes!':<12}")
print(f"  {'Multi-scale':<24} {'No':<12} {'Freq modes':<12} {'Slices':<12} {'Hops':<12} {'Pool/Up':<12}")
print(f"  {'Resolution flexible':<24} {'Yes':<12} {'Yes':<12} {'Partial':<12} {'Yes':<12} {'Yes':<12}")
print(f"  {'Irregular mesh':<24} {'Yes':<12} {'No':<12} {'Yes':<12} {'Yes':<12} {'No':<12}")
print(f"  {'Memory usage':<24} {'Low':<12} {'Medium':<12} {'High':<12} {'Medium':<12} {'Low':<12}")
print(f"  {'Training speed':<24} {'Slow':<12} {'Fast':<12} {'Medium':<12} {'Fast':<12} {'Fast':<12}")
print("-" * 70)
print()
print("  Key U-Net concepts demonstrated:")
print("    1. Encoder-Decoder: downsample (context) -> upsample (localization)")
print("    2. Skip Connections: preserve fine details lost during downsampling")
print("    3. Multi-scale: process features at 32x32 and 16x16 simultaneously")
print("    4. 3D Conv: PhysicsNeMo U-Net uses Conv3d (2D data replicated to 3D)")
print()
print("  When to use U-Net:")
print("    - Grid-based data (images, regular meshes)")
print("    - Need multi-scale feature extraction")
print("    - Want simple, fast, well-understood architecture")
print("    - Fine-grained output details matter (skip connections help!)")
print()
print("  Result files:")
print(f"    - {fig_path}")
print(f"    - {fig2_path}")
print("=" * 70)
