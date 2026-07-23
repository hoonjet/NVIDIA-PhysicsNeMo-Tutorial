"""
Darcy Flow comparison script: FNO vs Transolver vs U-Net
==========================================================
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import physicsnemo
from physicsnemo.models.fno import FNO
from physicsnemo.models.unet import UNet
import time
import os
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"PhysicsNeMo: {physicsnemo.__version__}\n")

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# [1] Generating a common data set
# ============================================================
N_TRAIN = 200
N_GRID = 32
DEPTH = 4  # U-Net용 3D depth

def generate_darcy_data(n_samples, grid_size, device):
    raw_noise = torch.randn(n_samples, 1, grid_size, grid_size, device=device)
    ks = 5
    k_field = torch.nn.functional.avg_pool2d(raw_noise, ks, stride=1, padding=ks//2)
    k_field = torch.sigmoid(k_field * 2) + 0.1
    p_smooth = torch.nn.functional.avg_pool2d(k_field, ks, stride=1, padding=ks//2)
    x_c = torch.linspace(0, 1, grid_size, device=device).view(1,1,1,-1)
    y_c = torch.linspace(0, 1, grid_size, device=device).view(1,1,-1,1)
    spatial = torch.sin(np.pi * x_c) * torch.sin(np.pi * y_c)
    p_field = p_smooth * spatial * 3.0
    return k_field, p_field

k_train_2d, p_train_2d = generate_darcy_data(N_TRAIN, N_GRID, device)
# U-Net용 3D 변환
k_train_3d = k_train_2d.unsqueeze(-1).expand(-1,-1,-1,-1,DEPTH).contiguous()
p_train_3d = p_train_2d.unsqueeze(-1).expand(-1,-1,-1,-1,DEPTH).contiguous()

# 테스트 데이터 (64x64)
k_test_64_2d, p_test_64_2d = generate_darcy_data(5, 64, device)
k_test_64_3d = k_test_64_2d.unsqueeze(-1).expand(-1,-1,-1,-1,DEPTH).contiguous()
p_test_64_3d = p_test_64_2d.unsqueeze(-1).expand(-1,-1,-1,-1,DEPTH).contiguous()

print(f"Training data: {N_TRAIN} samples, {N_GRID}x{N_GRID} grid")
print(f"Test data: 5 samples, 64x64 grid (unseen resolution)\n")

# ============================================================
# [2] Model Definition
# ============================================================
EPOCHS = 200
LR = 1e-3
loss_fn = nn.MSELoss()

# --- FNO ---
fno = FNO(
    in_channels=1, out_channels=1,
    dimension=2,
    decoder_layers=2, decoder_layer_size=32,
    latent_channels=32, num_fno_layers=4,
    num_fno_modes=12, padding=8,
    coord_features=True,
).to(device)

# --- Transolver ---
from physicsnemo.models.transolver import Transolver

# Transolver uses [B, H, W, C] format (channels-last)
k_train_trans = k_train_2d.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 1]
p_train_trans = p_train_2d.permute(0, 2, 3, 1).contiguous()
k_test_64_trans = k_test_64_2d.permute(0, 2, 3, 1).contiguous()
p_test_64_trans = p_test_64_2d.permute(0, 2, 3, 1).contiguous()

transolver = Transolver(
    functional_dim=1,           # Input channels
    out_dim=1,                  # Output channels
    n_layers=3,                 # 3 Transolver blocks (reduced for speed)
    n_hidden=64,                # Hidden dimension (reduced for speed)
    n_head=4,                   # 4 attention heads
    dropout=0.0,
    act="gelu",
    mlp_ratio=4,
    slice_num=16,               # 16 learned slices (reduced for speed)
    unified_pos=True,
    ref=8,                       # 8x8 reference grid
    structured_shape=(N_GRID, N_GRID),
    use_te=False,                # Pure PyTorch
).to(device)

# --- U-Net ---
unet = UNet(
    in_channels=1, out_channels=1,
    model_depth=2,
    feature_map_channels=[32, 32, 64, 64],
    num_conv_blocks=2,
    conv_activation="relu",
    pooling_type="MaxPool3d", pool_size=2,
    normalization="groupnorm",
    gradient_checkpointing=False,
).to(device)

models = {
    "FNO": (fno, k_train_2d, p_train_2d),
    "Transolver": (transolver, k_train_trans, p_train_trans),
    "U-Net": (unet, k_train_3d, p_train_3d),
}

for name, (m, _, _) in models.items():
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  {name:12s}: {n_params:>10,} parameters")

# ============================================================
# [3] Training
# ============================================================
results = {}

for name, (model, k_data, p_data) in models.items():
    print(f"\n{'='*60}")
    print(f"  Training {name} ({EPOCHS} epochs)...")
    print(f"{'='*60}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_history = []
    start = time.time()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(k_data)
        loss = loss_fn(pred, p_data)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - start
            print(f"  Epoch {epoch:4d}/{EPOCHS} | Loss: {loss.item():.6e} | {elapsed:.1f}s")
    
    train_time = time.time() - start
    final_loss = loss.item()
    
    # 평가
    model.eval()
    with torch.no_grad():
        # 32x32 (Resolution for traning)
        pred_32 = model(k_data[:5])
        loss_32 = loss_fn(pred_32, p_data[:5]).item()
        
        # 64x64 (Up-scaled resolution)
        if name == "U-Net":
            pred_64 = model(k_test_64_3d)
            loss_64 = loss_fn(pred_64, p_test_64_3d).item()
            pred_32_vis = pred_32[0, 0, :, :, 0].cpu().numpy()
            pred_64_vis = pred_64[0, 0, :, :, 0].cpu().numpy()
        elif name == "Transolver":
            # Transolver uses fixed structured_shape=(32,32), cannot handle 64x64
            try:
                pred_64 = model(k_test_64_trans)
                loss_64 = loss_fn(pred_64, p_test_64_trans).item()
                pred_64_vis = pred_64[0, :, :, 0].cpu().numpy()
            except RuntimeError:
                loss_64 = float('nan')
                pred_64_vis = np.full((64, 64), np.nan)
            pred_32_vis = pred_32[0, :, :, 0].cpu().numpy()
        else:
            pred_64 = model(k_test_64_2d)
            loss_64 = loss_fn(pred_64, p_test_64_2d).item()
            pred_32_vis = pred_32[0, 0].cpu().numpy()
            pred_64_vis = pred_64[0, 0].cpu().numpy()
    
    n_params = sum(p.numel() for p in model.parameters())
    results[name] = {
        "params": n_params,
        "train_time": train_time,
        "final_loss": final_loss,
        "loss_32": loss_32,
        "loss_64": loss_64,
        "loss_history": loss_history,
        "pred_32": pred_32_vis,
        "pred_64": pred_64_vis,
    }
    print(f"  Done! Final loss: {final_loss:.6e}, Time: {train_time:.1f}s")
    print(f"  32x32 loss: {loss_32:.6e}, 64x64 loss: {loss_64:.6e}")

# ============================================================
# [4] Comparison 1: learning curve
# ============================================================
output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"FNO": "#2196F3", "Transolver": "#FF9800", "U-Net": "#4CAF50"}
for name, r in results.items():
    ax.semilogy(r["loss_history"], label=f'{name} ({r["params"]:,} params)', 
                color=colors[name], linewidth=1.0, alpha=0.8)
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("MSE Loss (log scale)", fontsize=13)
ax.set_title("Darcy Flow: Training Loss Comparison\nFNO vs Transolver vs U-Net (32x32, 200 samples)", 
             fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, EPOCHS)
plt.tight_layout()
fig1_path = os.path.join(output_dir, "comparison_loss_curves.png")
plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
print(f"\nLoss curve comparison saved: {fig1_path}")

# ============================================================
# [5] Comparison 2: 32x32
# ============================================================
k_sample = k_train_2d[0, 0].cpu().numpy()
p_sample = p_train_2d[0, 0].cpu().numpy()

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# Row 1: Predection
axes[0, 0].imshow(k_sample, cmap="viridis")
axes[0, 0].set_title("Input: k (32x32)\nPermeability", fontsize=11)
axes[0, 1].imshow(p_sample, cmap="viridis")
axes[0, 1].set_title("Target: p (32x32)\nTrue Pressure", fontsize=11)
for i, (name, r) in enumerate(results.items()):
    axes[0, 2+i].imshow(r["pred_32"], cmap="viridis")
    axes[0, 2+i].set_title(f"{name} Prediction\n(32x32)", fontsize=11)

# Row 2: Error map
axes[1, 0].imshow(np.zeros_like(p_sample), cmap="hot", vmin=0, vmax=0.5)
axes[1, 0].set_title("(blank)", fontsize=11)
axes[1, 1].imshow(np.abs(results["FNO"]["pred_32"] - p_sample), cmap="hot", vmin=0, vmax=0.5)
axes[1, 1].set_title("FNO Error\n|pred - target|", fontsize=11)
axes[1, 2].imshow(np.abs(results["Transolver"]["pred_32"] - p_sample), cmap="hot", vmin=0, vmax=0.5)
axes[1, 2].set_title("Transolver Error\n|pred - target|", fontsize=11)
axes[1, 3].imshow(np.abs(results["U-Net"]["pred_32"] - p_sample), cmap="hot", vmin=0, vmax=0.5)
axes[1, 3].set_title("U-Net Error\n|pred - target|", fontsize=11)
axes[1, 4].imshow(np.zeros_like(p_sample), cmap="hot", vmin=0, vmax=0.5)
axes[1, 4].set_title("(blank)", fontsize=11)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Darcy Flow Prediction Comparison (32x32, training resolution)\n"
             "Row 1: Input / Target / FNO / Transolver / U-Net predictions\n"
             "Row 2: Error maps (darker = better)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig2_path = os.path.join(output_dir, "comparison_predictions_32.png")
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"Prediction comparison (32x32) saved: {fig2_path}")

# ============================================================
# [6]  Comparison 3: transition to 64x64 scale
# ============================================================
k_64_sample = k_test_64_2d[0, 0].cpu().numpy()
p_64_sample = p_test_64_2d[0, 0].cpu().numpy()

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

axes[0, 0].imshow(k_64_sample, cmap="viridis")
axes[0, 0].set_title("Input: k (64x64)\n[UNSEEN RES]", fontsize=11)
axes[0, 1].imshow(p_64_sample, cmap="viridis")
axes[0, 1].set_title("Target: p (64x64)\nTrue Pressure", fontsize=11)
for i, (name, r) in enumerate(results.items()):
    if np.isnan(r["pred_64"]).any():
        axes[0, 2+i].text(0.5, 0.5, "N/A\n(fixed grid)", ha="center", va="center", fontsize=14, transform=axes[0, 2+i].transAxes)
        axes[0, 2+i].set_title(f"{name} Prediction\n(64x64) - UNSUPPORTED", fontsize=11, color="red")
    else:
        axes[0, 2+i].imshow(r["pred_64"], cmap="viridis")
        axes[0, 2+i].set_title(f"{name} Prediction\n(64x64)", fontsize=11)

# Row 2: Error map
axes[1, 0].imshow(np.zeros_like(p_64_sample), cmap="hot", vmin=0, vmax=1.0)
axes[1, 0].set_title("(blank)", fontsize=11)
for i, (name, r) in enumerate(results.items()):
    if np.isnan(r["pred_64"]).any():
        axes[1, 1+i].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14, transform=axes[1, 1+i].transAxes)
        axes[1, 1+i].set_title(f"{name} Error\nN/A", fontsize=11, color="red")
    else:
        axes[1, 1+i].imshow(np.abs(r["pred_64"] - p_64_sample), cmap="hot", vmin=0, vmax=1.0)
        axes[1, 1+i].set_title(f"{name} Error\n(loss={r['loss_64']:.4f})", fontsize=11)
axes[1, 4].imshow(np.zeros_like(p_64_sample), cmap="hot", vmin=0, vmax=1.0)
axes[1, 4].set_title("(blank)", fontsize=11)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Darcy Flow: Zero-Shot Resolution Transfer (32x32 → 64x64)\n"
             "Row 1: Input / Target / FNO / Transolver / U-Net predictions at 64x64\n"
             "Row 2: Error maps (darker = better)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig3_path = os.path.join(output_dir, "comparison_predictions_64.png")
plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
print(f"Prediction comparison (64x64) saved: {fig3_path}")

plt.close("all")

# ============================================================
# [7] Summary for test reports
# ============================================================
print("\n" + "=" * 80)
print("  PERFORMANCE SUMMARY: FNO vs Transolver vs U-Net (Darcy Flow)")
print("=" * 80)
print(f"  {'Metric':<28} {'FNO':>16} {'Transolver':>16} {'U-Net':>16}")
print(f"  {'-'*28} {'-'*16} {'-'*16} {'-'*16}")
print(f"  {'Parameters':<28} {results['FNO']['params']:>16,} {results['Transolver']['params']:>16,} {results['U-Net']['params']:>16,}")
print(f"  {'Training time (s)':<28} {results['FNO']['train_time']:>16.1f} {results['Transolver']['train_time']:>16.1f} {results['U-Net']['train_time']:>16.1f}")
print(f"  {'Final loss (32x32)':<28} {results['FNO']['final_loss']:>16.6e} {results['Transolver']['final_loss']:>16.6e} {results['U-Net']['final_loss']:>16.6e}")
print(f"  {'Test loss (32x32)':<28} {results['FNO']['loss_32']:>16.6e} {results['Transolver']['loss_32']:>16.6e} {results['U-Net']['loss_32']:>16.6e}")
print(f"  {'Test loss (64x64)':<28} {results['FNO']['loss_64']:>16.6e} {results['Transolver']['loss_64']:>16.6e} {results['U-Net']['loss_64']:>16.6e}")
ratio_fno = results['FNO']['loss_64'] / results['FNO']['loss_32']
ratio_trans = results['Transolver']['loss_64'] / results['Transolver']['loss_32']
ratio_unet = results['U-Net']['loss_64'] / results['U-Net']['loss_32']
print(f"  {'Loss ratio (64/32)':<28} {ratio_fno:>16.2f}x {ratio_trans:>16.2f}x {ratio_unet:>16.2f}x")
print(f"  {'Resolution independent':<28} {'Yes':>16} {'Partial':>16} {'Degraded':>16}")
print("=" * 80)
