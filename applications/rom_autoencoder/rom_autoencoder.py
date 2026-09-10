"""
PhysicsNeMo Tutorial: Reduced-Order Model via Autoencoder
=========================================================
Compressing High-Dimensional PDE Solutions into Latent Space

Existing tutorials:
  - All are forward solvers (input → full field prediction)
  - No model compression / dimensionality reduction
  - No latent space analysis

THIS tutorial:
  - Autoencoder: compress 2D heat field (64×64=4096 dims) → latent (8 dims)
  - 512× compression ratio
  - Decoder reconstructs full field from latent
  - Latent space interpolation: morph between different boundary conditions
  - Latent space PCA: discover physical modes automatically

Key difference from existing tutorials:
  ┌──────────────────────┬──────────────────────────┐
  │ Existing              │ THIS (ROM Autoencoder)   │
  ├──────────────────────┼──────────────────────────┤
  │ Forward solver        │ Dimensionality reduction │
  │ Full field I/O        │ Compressed latent space  │
  │ No interpolation      │ Latent interpolation     │
  │ Single solution       │ Solution manifold        │
  │ No mode discovery     │ PCA on latent = POD-like  │
  └──────────────────────┴──────────────────────────┘

Physics: 2D Steady-State Heat Equation
  ∇·(k∇T) + Q = 0  on [0,1]×[0,1]
  BC: T=0 on left/right, T=T_top on top, T=0 on bottom
  Parameter: T_top varies (generates solution family)

Author: PhysicsNeMo Tutorial
Date: 2026-09-07
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo Tutorial: Reduced-Order Model (Autoencoder)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# [1] Data Generation: 2D Heat Equation Solver
# ============================================================
NX, NY = 64, 64
N_SAMPLES = 500  # Number of heat solutions (different T_top)

def solve_heat_2d(T_top, Q_source=0.0, nx=NX, ny=NY):
    """
    Solve 2D steady-state heat equation via finite difference.
    ∇²T = -Q/k  (Laplace/Poisson)
    BC: T=0 left/right/bottom, T=T_top on top
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    T = np.zeros((ny, nx), dtype=np.float32)
    # BC
    T[-1, :] = T_top  # top

    # Gauss-Seidel iteration
    for _ in range(2000):
        T_old = T.copy()
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                T[j, i] = 0.25 * (T[j, i+1] + T[j, i-1] +
                                   T[j+1, i] + T[j-1, i] - Q_source * dx * dy)
        # Re-apply BC
        T[-1, :] = T_top
        T[0, :] = 0.0
        T[:, 0] = 0.0
        T[:, -1] = 0.0
        if np.max(np.abs(T - T_old)) < 1e-6:
            break

    return T

print(f"\n[1] Generating {N_SAMPLES} heat equation solutions...")
print(f"  Grid: {NX}×{NY} = {NX*NY} dimensions")
t0 = time.time()

# Generate dataset: vary T_top from 50 to 200
T_top_values = np.linspace(50, 200, N_SAMPLES, dtype=np.float32)
dataset = np.zeros((N_SAMPLES, NY, NX), dtype=np.float32)

for idx, T_top in enumerate(T_top_values):
    dataset[idx] = solve_heat_2d(T_top)

print(f"  Generation time: {time.time()-t0:.1f}s")
print(f"  T_top range: [{T_top_values[0]:.0f}, {T_top_values[-1]:.0f}]")
print(f"  Temperature range: [{dataset.min():.2f}, {dataset.max():.2f}]")

# Normalize to [0, 1]
T_max = dataset.max()
dataset_norm = dataset / T_max

# Train/test split
N_TRAIN = 400
N_TEST = N_SAMPLES - N_TRAIN
indices = np.random.permutation(N_SAMPLES)
train_idx = indices[:N_TRAIN]
test_idx = indices[N_TRAIN:]

train_data = torch.tensor(dataset_norm[train_idx], dtype=torch.float32, device=device)
test_data = torch.tensor(dataset_norm[test_idx], dtype=torch.float32, device=device)
train_T_top = torch.tensor(T_top_values[train_idx], dtype=torch.float32, device=device)
test_T_top = torch.tensor(T_top_values[test_idx], dtype=torch.float32, device=device)

# Flatten for autoencoder
train_flat = train_data.view(N_TRAIN, -1)  # [N, 4096]
test_flat = test_data.view(N_TEST, -1)

print(f"  Train: {N_TRAIN}, Test: {N_TEST}")
print(f"  Input dim: {NX*NY} → Latent dim: 8 (compression: {NX*NY//8}×)")

# ============================================================
# [2] Autoencoder Model
# ============================================================
LATENT_DIM = 8

class HeatAutoencoder(nn.Module):
    """
    Autoencoder for 2D heat field compression.

    Encoder: 4096 → 512 → 128 → 32 → 8 (latent)
    Decoder: 8 → 32 → 128 → 512 → 4096 (reconstruction)

    The latent space captures the essential modes of the solution family.
    """
    def __init__(self, input_dim=NX*NY, latent_dim=LATENT_DIM):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

print(f"\n[2] Autoencoder Model:")
print(f"  Encoder: {NX*NY} → 512 → 128 → 32 → {LATENT_DIM}")
print(f"  Decoder: {LATENT_DIM} → 32 → 128 → 512 → {NX*NY}")
print(f"  Compression: {NX*NY}/{LATENT_DIM} = {NX*NY/LATENT_DIM:.0f}×")

# ============================================================
# [3] Training
# ============================================================
model = HeatAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

N_EPOCHS = 300
BATCH_SIZE = 32

print(f"\n[3] Training:")
print(f"  Epochs: {N_EPOCHS}, Batch: {BATCH_SIZE}")

loss_history = []
t_start = time.time()

for epoch in range(N_EPOCHS):
    model.train()
    perm = torch.randperm(N_TRAIN)
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, N_TRAIN, BATCH_SIZE):
        batch_idx = perm[i:i+BATCH_SIZE]
        batch = train_flat[batch_idx]

        recon, z = model(batch)
        loss = F.mse_loss(recon, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | Loss: {avg_loss:.6e} | Time: {time.time()-t_start:.1f}s")

print(f"\n  Training complete! Total time: {time.time()-t_start:.1f}s")

# ============================================================
# [4] Evaluation: Reconstruction Quality
# ============================================================
print(f"\n[4] Evaluation: Reconstruction Quality...")

model.eval()
with torch.no_grad():
    recon_test, z_test = model(test_flat)
    recon_train, z_train = model(train_flat)

    # MSE
    mse_test = F.mse_loss(recon_test, test_flat).item()
    mse_train = F.mse_loss(recon_train, train_flat).item()

    # Relative L2 error
    rel_err_test = torch.norm(recon_test - test_flat, dim=1) / (torch.norm(test_flat, dim=1) + 1e-10)
    rel_err_train = torch.norm(recon_train - train_flat, dim=1) / (torch.norm(train_flat, dim=1) + 1e-10)

print(f"  Train MSE: {mse_train:.6e}, Mean Rel L2: {rel_err_train.mean().item():.4f}")
print(f"  Test  MSE: {mse_test:.6e}, Mean Rel L2: {rel_err_test.mean().item():.4f}")

# ============================================================
# [5] Latent Space Analysis
# ============================================================
print(f"\n[5] Latent Space Analysis...")

# Get latent vectors for all samples
all_data = torch.tensor(dataset_norm, dtype=torch.float32, device=device).view(N_SAMPLES, -1)
with torch.no_grad():
    _, all_z = model(all_data)
all_z_np = all_z.cpu().numpy()

# PCA on latent space
from sklearn.decomposition import PCA
pca = PCA(n_components=LATENT_DIM)
z_pca = pca.fit_transform(all_z_np)

print(f"  Latent dim: {LATENT_DIM}")
print(f"  PCA explained variance ratio (top 5):")
for i in range(min(5, LATENT_DIM)):
    print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} ({pca.explained_variance_ratio_[i]*100:.2f}%)")
print(f"  Cumulative (top 3): {sum(pca.explained_variance_ratio_[:3])*100:.2f}%")

# ============================================================
# [6] Latent Interpolation
# ============================================================
print(f"\n[6] Latent Interpolation...")

# Pick two extreme samples
idx_low = 0   # T_top=50
idx_high = -1  # T_top=200

z_low = all_z[idx_low:idx_low+1]
z_high = all_z[idx_high:idx_high+1]

# Interpolate in latent space
n_interp = 5
interp_fields = []
for i in range(n_interp):
    alpha = i / (n_interp - 1)
    z_interp = (1 - alpha) * z_low + alpha * z_high
    with torch.no_grad():
        field_interp = model.decoder(z_interp)
    field_interp = field_interp.view(NY, NX).cpu().numpy() * T_max
    interp_fields.append(field_interp)

# Also get the actual solutions at interpolated T_top values
T_top_low = T_top_values[idx_low]
T_top_high = T_top_values[idx_high]
actual_fields = []
for i in range(n_interp):
    alpha = i / (n_interp - 1)
    T_top_interp = (1 - alpha) * T_top_low + alpha * T_top_high
    actual_fields.append(solve_heat_2d(T_top_interp))

# ============================================================
# [7] Visualization
# ============================================================
print(f"\n[7] Visualization...")

# --- Figure 1: Reconstruction comparison ---
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for col in range(4):
    idx = col * (N_TEST // 4)
    actual = test_flat[idx].view(NY, NX).cpu().numpy() * T_max
    recon = recon_test[idx].view(NY, NX).cpu().numpy() * T_max
    error = np.abs(actual - recon)

    axes[0, col].imshow(actual.T, origin='lower', cmap='hot', vmin=0, vmax=T_max)
    axes[0, col].set_title(f'Original (T_top={test_T_top[idx].item():.0f})', fontsize=11)
    axes[0, col].axis('off')

    axes[1, col].imshow(recon.T, origin='lower', cmap='hot', vmin=0, vmax=T_max)
    axes[1, col].set_title(f'Reconstruction', fontsize=11)
    axes[1, col].axis('off')

    im = axes[2, col].imshow(error.T, origin='lower', cmap='hot', vmin=0, vmax=T_max*0.05)
    axes[2, col].set_title(f'|Error| (max={error.max():.2f})', fontsize=11)
    axes[2, col].axis('off')

plt.colorbar(im, ax=axes[2, :], shrink=0.5, label='Temperature')
plt.suptitle(f'ROM Autoencoder: Reconstruction (Latent dim={LATENT_DIM}, Compression={NX*NY//LATENT_DIM}×)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rom_reconstruction.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: rom_reconstruction.png")

# --- Figure 2: Latent space PCA ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PC1 vs PC2, colored by T_top
ax = axes[0]
sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=T_top_values, cmap='viridis', s=20)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title('Latent Space (PC1 vs PC2)', fontsize=13, fontweight='bold')
plt.colorbar(sc, ax=ax, label='T_top')

# PC1 vs T_top
ax = axes[1]
ax.scatter(T_top_values, z_pca[:, 0], c='blue', s=20, alpha=0.6)
ax.set_xlabel('T_top', fontsize=12)
ax.set_ylabel('PC1', fontsize=12)
ax.set_title('PC1 vs Physical Parameter', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Explained variance
ax = axes[2]
ax.bar(range(1, LATENT_DIM+1), pca.explained_variance_ratio_, color='steelblue', alpha=0.7)
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance Ratio', fontsize=12)
ax.set_title('PCA Explained Variance', fontsize=13, fontweight='bold')
ax.set_xticks(range(1, LATENT_DIM+1))
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rom_latent_pca.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: rom_latent_pca.png")

# --- Figure 3: Latent interpolation ---
fig, axes = plt.subplots(2, n_interp, figsize=(5*n_interp, 10))

for i in range(n_interp):
    alpha = i / (n_interp - 1)
    T_top_interp = (1 - alpha) * T_top_low + alpha * T_top_high

    # Interpolated (from latent)
    ax = axes[0, i]
    ax.imshow(interp_fields[i].T, origin='lower', cmap='hot', vmin=0, vmax=T_max)
    ax.set_title(f'Latent interp\nT_top={T_top_interp:.0f}', fontsize=10)
    ax.axis('off')

    # Actual (from solver)
    ax = axes[1, i]
    ax.imshow(actual_fields[i].T, origin='lower', cmap='hot', vmin=0, vmax=T_max)
    ax.set_title(f'Actual solver\nT_top={T_top_interp:.0f}', fontsize=10)
    ax.axis('off')

axes[0, 0].set_ylabel('Latent Interp', fontsize=12, rotation=90, labelpad=10)
axes[1, 0].set_ylabel('Actual Solver', fontsize=12, rotation=90, labelpad=10)

plt.suptitle('Latent Space Interpolation: Morphing Between Solutions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rom_interpolation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: rom_interpolation.png")

# --- Figure 4: Training loss ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(loss_history, 'b-', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('ROM Autoencoder Training Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rom_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: rom_loss.png")

# --- Figure 5: Concept comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.axis('off')
concept_text = (
    "Existing Tutorials (Forward Solver):\n\n"
    "  Input (BC/geometry)\n"
    "       ↓\n"
    "  ┌─────────┐\n"
    "  │  Model  │  → Full field (4096 dims)\n"
    "  └─────────┘\n\n"
    "  No compression\n"
    "  No latent space\n"
    "  No interpolation\n\n"
    "THIS Tutorial (ROM Autoencoder):\n\n"
    "  Full field (4096 dims)\n"
    "       ↓\n"
    "  ┌─────────┐\n"
    "  │ Encoder │  → Latent (8 dims)\n"
    "  └─────────┘\n"
    "       ↓\n"
    "  ┌─────────┐\n"
    "  │ Decoder │  → Reconstructed field\n"
    "  └─────────┘\n\n"
    "  512× compression\n"
    "  Latent interpolation\n"
    "  Mode discovery (PCA)"
)
ax.text(0.05, 0.95, concept_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Forward Solver vs ROM Autoencoder', fontsize=13, fontweight='bold')

# Compression ratio bar chart
ax = axes[1]
methods = ['Full Field\n(No ROM)', 'POD\n(16 modes)', 'Autoencoder\n(8 latent)']
dims = [NX*NY, 16, LATENT_DIM]
colors = ['red', 'orange', 'green']
bars = ax.bar(methods, dims, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Dimensionality', fontsize=12)
ax.set_title('Dimensionality Reduction Comparison', fontsize=13, fontweight='bold')
ax.set_yscale('log')
for bar, dim in zip(bars, dims):
    ax.text(bar.get_x() + bar.get_width()/2, dim * 1.2, f'{dim}', 
            ha='center', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rom_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: rom_concept.png")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("[8] Summary")
print("=" * 70)

print(f"\n  Method: Reduced-Order Model via Autoencoder")
print(f"  Physics: 2D Steady-State Heat Equation")
print(f"  Full dim: {NX*NY} → Latent dim: {LATENT_DIM} (compression: {NX*NY//LATENT_DIM}×)")
print(f"")
print(f"  Reconstruction Error:")
print(f"    Train: MSE={mse_train:.6e}, Rel L2={rel_err_train.mean().item():.4f}")
print(f"    Test:  MSE={mse_test:.6e}, Rel L2={rel_err_test.mean().item():.4f}")
print(f"")
print(f"  PCA on Latent Space:")
print(f"    PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"    PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"    Top 3 cumulative: {sum(pca.explained_variance_ratio_[:3])*100:.2f}%")
print(f"")
print(f"  Key difference from existing tutorials:")
print(f"    Existing: Forward solver (input → full field)")
print(f"    THIS:     ROM autoencoder (compress → latent → reconstruct)")
print(f"")
print(f"  Results saved to: {RESULTS_DIR}")
print("=" * 70)
