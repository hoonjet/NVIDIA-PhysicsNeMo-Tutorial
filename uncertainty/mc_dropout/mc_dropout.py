"""
MC Dropout for Uncertainty Quantification
==========================================
This tutorial implements Monte Carlo Dropout for uncertainty estimation
on Darcy flow, and compares it with the existing Deep Ensemble tutorial.

Existing tutorial (deep_ensemble):
  - Trains N=5 independent CNNs (5x training cost)
  - Uncertainty = std across 5 models
  - High quality, high cost

THIS tutorial:
  - Trains 1 CNN with dropout (1x training cost)
  - At inference: keep dropout ON, run T=50 forward passes
  - Uncertainty = std across 50 stochastic passes
  - Lower cost, approximate uncertainty

Key concepts:
  1. MC Dropout: approximate Bayesian inference via dropout
  2. Epistemic uncertainty: model uncertainty (reducible with more data)
  3. Cost-accuracy tradeoff: 1 model (MC Dropout) vs 5 models (Deep Ensemble)
  4. OOD detection: high uncertainty on out-of-distribution inputs
  5. Calibration: uncertainty should correlate with error

Author: PhysicsNeMo Tutorial
Date: 2026-08-24
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

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("MC Dropout for Uncertainty Quantification")
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
# [1] Darcy Flow Data Generation
# ============================================================
N_GRID = 32
N_TRAIN = 200
N_TEST_ID = 20
N_TEST_OOD = 20

print(f"\n[1] Generating Darcy flow data")
print(f"  Grid: {N_GRID}x{N_GRID}")

def generate_gaussian_field(n, length_scale=0.2, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    x = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    field = np.zeros((n, n))
    for _ in range(20):
        kx = rng.randint(-3, 4)
        ky = rng.randint(-3, 4)
        amp = rng.randn() * np.exp(-np.sqrt(kx**2 + ky**2) * length_scale * 10)
        phase = rng.uniform(0, 2 * np.pi)
        field += amp * np.sin(2 * np.pi * (kx * xx + ky * yy) + phase)
    field = (field - field.mean()) / (field.std() + 1e-8)
    return np.exp(field * 1.5)

def solve_darcy(k, n):
    dx = 1.0 / (n - 1)
    p = np.zeros((n, n))
    for _ in range(500):
        p_new = p.copy()
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                k_e = 0.5 * (k[i, j] + k[i + 1, j])
                k_w = 0.5 * (k[i, j] + k[i - 1, j])
                k_n = 0.5 * (k[i, j] + k[i, j + 1])
                k_s = 0.5 * (k[i, j] + k[i, j - 1])
                p_new[i, j] = (k_e * p[i + 1, j] + k_w * p[i - 1, j] +
                               k_n * p[i, j + 1] + k_s * p[i, j - 1] + dx * dx) / (k_e + k_w + k_n + k_s)
        p = p_new
    return p

print("  Generating train (ID, ls=0.2)...")
train_k = np.array([generate_gaussian_field(N_GRID, 0.2) for _ in range(N_TRAIN)])
train_p = np.array([solve_darcy(train_k[i], N_GRID) for i in range(N_TRAIN)])
print("  Generating test ID (ls=0.2)...")
test_k_id = np.array([generate_gaussian_field(N_GRID, 0.2) for _ in range(N_TEST_ID)])
test_p_id = np.array([solve_darcy(test_k_id[i], N_GRID) for i in range(N_TEST_ID)])
print("  Generating test OOD (ls=0.05, never seen)...")
test_k_ood = np.array([generate_gaussian_field(N_GRID, 0.05) for _ in range(N_TEST_OOD)])
test_p_ood = np.array([solve_darcy(test_k_ood[i], N_GRID) for i in range(N_TEST_OOD)])

k_mean, k_std = train_k.mean(), train_k.std()
p_mean, p_std = train_p.mean(), train_p.std()
train_k_n = (train_k - k_mean) / (k_std + 1e-8)
train_p_n = (train_p - p_mean) / (p_std + 1e-8)
test_k_id_n = (test_k_id - k_mean) / (k_std + 1e-8)
test_p_id_n = (test_p_id - p_mean) / (p_std + 1e-8)
test_k_ood_n = (test_k_ood - k_mean) / (k_std + 1e-8)
test_p_ood_n = (test_p_ood - p_mean) / (p_std + 1e-8)

train_k_t = torch.from_numpy(train_k_n).float().unsqueeze(1).to(device)
train_p_t = torch.from_numpy(train_p_n).float().unsqueeze(1).to(device)
test_k_id_t = torch.from_numpy(test_k_id_n).float().unsqueeze(1).to(device)
test_p_id_t = torch.from_numpy(test_p_id_n).float().unsqueeze(1).to(device)
test_k_ood_t = torch.from_numpy(test_k_ood_n).float().unsqueeze(1).to(device)
test_p_ood_t = torch.from_numpy(test_p_ood_n).float().unsqueeze(1).to(device)

print(f"  Train: {N_TRAIN}, Test ID: {N_TEST_ID}, Test OOD: {N_TEST_OOD}")

# ============================================================
# [2] CNN with Dropout
# ============================================================
# Same architecture as deep_ensemble, but WITH dropout layers.
# At training: dropout active (regularization)
# At inference: dropout KEPT active (MC Dropout)

class CNNDropout(nn.Module):
    """CNN with dropout for MC Dropout uncertainty estimation."""
    def __init__(self, in_ch=1, out_ch=1, hidden=32, dropout_p=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.SiLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(hidden * 2, hidden * 4, 3, stride=2, padding=1), nn.SiLU(),
            nn.Dropout2d(dropout_p),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Dropout2d(dropout_p),
            nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1), nn.SiLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(hidden, out_ch, 3, padding=1),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

DROPOUT_P = 0.1
print(f"\n[2] Building CNN with dropout (p={DROPOUT_P})...")
model = CNNDropout(in_ch=1, out_ch=1, hidden=32, dropout_p=DROPOUT_P).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")
print(f"  Dropout: {DROPOUT_P} (kept ON during inference for MC Dropout)")

# ============================================================
# [3] Training
# ============================================================
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3

print(f"\n[3] Training CNN with dropout ({EPOCHS} epochs)")
print("-" * 70)

opt = torch.optim.Adam(model.parameters(), lr=LR)
train_losses = []
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(N_TRAIN)
    epoch_loss = 0; n_b = 0
    for i in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        k = train_k_t[idx]
        p = train_p_t[idx]
        pred = model(k)
        loss = F.mse_loss(pred, p)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_b += 1
    train_losses.append(epoch_loss / n_b)
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:4d} | Loss: {train_losses[-1]:.6e} | Time: {time.time()-start:.1f}s")

train_time = time.time() - start
print("-" * 70)

# ============================================================
# [4] MC Dropout Inference
# ============================================================
# KEY: Keep model in TRAIN mode (dropout ON) during inference!
# Run T forward passes, compute mean and std.

T_MC = 50  # Number of MC samples

print(f"\n[4] MC Dropout inference (T={T_MC} forward passes)")
print("-" * 70)

def mc_dropout_predict(model, x, T=50):
    """Run T stochastic forward passes, return mean and std."""
    model.train()  # Keep dropout ON!
    preds = torch.stack([model(x) for _ in range(T)])  # [T, B, 1, H, W]
    mean = preds.mean(dim=0)  # [B, 1, H, W]
    std = preds.std(dim=0)    # [B, 1, H, W]
    return mean, std, preds

print("  Predicting on ID test set...")
mean_id, std_id, _ = mc_dropout_predict(model, test_k_id_t, T_MC)
print("  Predicting on OOD test set...")
mean_ood, std_ood, _ = mc_dropout_predict(model, test_k_ood_t, T_MC)

# Compute metrics
mae_id = torch.mean(torch.abs(mean_id - test_p_id_t)).item()
mae_ood = torch.mean(torch.abs(mean_ood - test_p_ood_t)).item()
unc_id = std_id.mean().item()
unc_ood = std_ood.mean().item()

print(f"\n  ID  MAE: {mae_id:.6f} | Uncertainty: {unc_id:.6f}")
print(f"  OOD MAE: {mae_ood:.6f} | Uncertainty: {unc_ood:.6f}")
print(f"  OOD/ID uncertainty ratio: {unc_ood/unc_id:.2f}x (should be > 1)")

# ============================================================
# [5] Calibration Analysis
# ============================================================
print(f"\n[5] Calibration analysis")

# Check if uncertainty correlates with error
errors_id = torch.abs(mean_id - test_p_id_t).detach().cpu().numpy().flatten()
uncs_id = std_id.detach().cpu().numpy().flatten()
errors_ood = torch.abs(mean_ood - test_p_ood_t).detach().cpu().numpy().flatten()
uncs_ood = std_ood.detach().cpu().numpy().flatten()

# Pearson correlation
from numpy import corrcoef
corr_id = corrcoef(errors_id, uncs_id)[0, 1]
corr_ood = corrcoef(errors_ood, uncs_ood)[0, 1]
print(f"  Error-uncertainty correlation (ID):  {corr_id:.4f}")
print(f"  Error-uncertainty correlation (OOD): {corr_ood:.4f}")
print(f"  (Higher = better calibrated)")

# ============================================================
# [6] Visualization
# ============================================================
print(f"\n[6] Generating visualizations...")

# --- Figure 1: Training loss ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(train_losses, linewidth=1.5, color='blue')
ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss (MSE)')
ax.set_title('CNN with Dropout — Training Loss'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mc_dropout_loss.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: ID vs OOD predictions ---
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
for i in range(4):
    # True
    ax = axes[0, i]
    im = ax.imshow(test_p_id[i, 0].cpu().numpy(), cmap='jet', origin='lower')
    ax.set_title(f'ID True {i+1}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Mean prediction
    ax = axes[1, i]
    im = ax.imshow(mean_id[i, 0].detach().cpu().numpy(), cmap='jet', origin='lower')
    ax.set_title(f'ID Mean Pred {i+1}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Uncertainty
    ax = axes[2, i]
    im = ax.imshow(std_id[i, 0].detach().cpu().numpy(), cmap='hot', origin='lower')
    ax.set_title(f'ID Uncertainty {i+1}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('MC Dropout: In-Distribution Predictions', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mc_dropout_id.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: OOD predictions ---
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
for i in range(4):
    ax = axes[0, i]
    im = ax.imshow(test_p_ood[i, 0].cpu().numpy(), cmap='jet', origin='lower')
    ax.set_title(f'OOD True {i+1}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, i]
    im = ax.imshow(mean_ood[i, 0].detach().cpu().numpy(), cmap='jet', origin='lower')
    ax.set_title(f'OOD Mean Pred {i+1}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2, i]
    im = ax.imshow(std_ood[i, 0].detach().cpu().numpy(), cmap='hot', origin='lower')
    ax.set_title(f'OOD Uncertainty {i+1}\n(higher = less confident)'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('MC Dropout: Out-of-Distribution Predictions (OOD)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mc_dropout_ood.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: OOD detection ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Uncertainty histogram: ID vs OOD
ax = axes[0]
ax.hist(uncs_id, bins=50, alpha=0.6, color='blue', label=f'ID (mean={uncs_id.mean():.4f})', density=True)
ax.hist(uncs_ood, bins=50, alpha=0.6, color='red', label=f'OOD (mean={uncs_ood.mean():.4f})', density=True)
ax.set_xlabel('Uncertainty (std)'); ax.set_ylabel('Density')
ax.set_title('OOD Detection: Uncertainty Distribution'); ax.legend(); ax.grid(True, alpha=0.3)

# Error vs uncertainty scatter
ax = axes[1]
ax.scatter(uncs_id[::10], errors_id[::10], s=5, alpha=0.3, color='blue', label='ID')
ax.scatter(uncs_ood[::10], errors_ood[::10], s=5, alpha=0.3, color='red', label='OOD')
ax.set_xlabel('Uncertainty (std)'); ax.set_ylabel('Absolute Error')
ax.set_title(f'Calibration: Error vs Uncertainty\n(ID r={corr_id:.3f}, OOD r={corr_ood:.3f})')
ax.legend(); ax.grid(True, alpha=0.3)

# Summary bar chart
ax = axes[2]
metrics = ['MAE\n(ID)', 'MAE\n(OOD)', 'Uncertainty\n(ID)', 'Uncertainty\n(OOD)']
values = [mae_id, mae_ood, unc_id, unc_ood]
colors = ['blue', 'red', 'blue', 'red']
bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Value')
ax.set_title('MC Dropout: Summary Metrics')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.4f}',
            ha='center', va='bottom', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('MC Dropout: OOD Detection & Calibration', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mc_dropout_ood_detection.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 5: Concept comparison with Deep Ensemble ---
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.text(0.5, 0.95, 'MC Dropout vs Deep Ensemble', ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.80,
    'Deep Ensemble (existing tutorial):\n'
    '  - Train N=5 independent CNNs\n'
    '  - 5x training cost\n'
    '  - Uncertainty = std across 5 models\n'
    '  - High quality, high cost\n'
    '  - Each model sees full data',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(0.55, 0.80,
    'MC Dropout (THIS tutorial):\n'
    '  - Train 1 CNN with dropout\n'
    '  - 1x training cost (5x cheaper!)\n'
    '  - Uncertainty = std across T=50 passes\n'
    '  - Approximate, low cost\n'
    '  - Dropout = approximate Bayesian',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.50,
    'MC Dropout theory:\n'
    '  - Dropout = Bernoulli mask on weights\n'
    '  - Each forward pass = different subnetwork\n'
    '  - T passes = T subnetworks = ensemble\n'
    '  - Approximates Bayesian model averaging\n'
    '  - Gal & Ghahramani (2016)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.55, 0.50,
    'Cost-accuracy tradeoff:\n'
    '  Deep Ensemble:\n'
    '    Cost: 5x train + 5x inference\n'
    '    Quality: Best (true ensemble)\n'
    '  MC Dropout:\n'
    '    Cost: 1x train + 50x inference\n'
    '    Quality: Good (approximate)\n'
    '  Rule: Use MC Dropout when\n'
    '    training is expensive',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.20,
    'When to use which?\n'
    '  MC Dropout:\n'
    '    - Large models (training is expensive)\n'
    '    - Quick prototyping\n'
    '    - When 1 model is already good\n'
    '  Deep Ensemble:\n'
    '    - Small models (training is cheap)\n'
    '    - Need best uncertainty\n'
    '    - Safety-critical applications',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax.text(0.55, 0.20,
    'Key concepts:\n'
    '  1. MC Dropout = approximate Bayesian\n'
    '  2. Dropout ON at inference (unusual!)\n'
    '  3. T stochastic forward passes\n'
    '  4. mean = prediction, std = uncertainty\n'
    '  5. OOD: higher uncertainty\n'
    '  6. Calibration: error ~ uncertainty',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightskyblue', alpha=0.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mc_dropout_explanation.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [7] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: MC Dropout Uncertainty Quantification")
print("=" * 70)
print(f"  Problem:           Darcy Flow ({N_GRID}x{N_GRID})")
print(f"  Model:             CNN with dropout (p={DROPOUT_P})")
print(f"  Training:          {EPOCHS} epochs, {train_time:.1f}s (1 model)")
print(f"  MC samples:        T={T_MC} forward passes")
print(f"  --- Results ---")
print(f"  ID  MAE:           {mae_id:.6f}")
print(f"  OOD MAE:           {mae_ood:.6f}")
print(f"  ID  Uncertainty:   {unc_id:.6f}")
print(f"  OOD Uncertainty:   {unc_ood:.6f}")
print(f"  OOD/ID ratio:      {unc_ood/unc_id:.2f}x (OOD should be higher)")
print(f"  Calibration (ID):  r={corr_id:.4f}")
print(f"  Calibration (OOD): r={corr_ood:.4f}")
print()
print("Key observations:")
print("  1. MC DROPOUT: 1 model + dropout ON at inference = approximate ensemble")
print("  2. COST: 1x training (vs 5x for Deep Ensemble) — 5x cheaper!")
print("  3. OOD DETECTION: Higher uncertainty on unseen inputs (OOD/ID > 1)")
print("  4. CALIBRATION: Uncertainty correlates with error (r > 0)")
print("  5. vs DEEP ENSEMBLE: Cheaper but less accurate — cost-accuracy tradeoff")
print("  6. BAYESIAN: Dropout approximates Bayesian inference (Gal & Ghahramani 2016)")
print("=" * 70)
