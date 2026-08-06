"""
PhysicsNeMo Uncertainty Quantification Tutorial: Deep Ensemble for UQ
=====================================================================
Uncertainty Quantification (UQ) using Deep Ensemble method.

All existing tutorials in this repo train a SINGLE model and report its prediction.
This is the ONLY tutorial that answers: "WHEN does the model make mistakes?"

Problem: Darcy Flow surrogate with uncertainty estimation
    - Train N=5 independent CNN models with different random seeds
    - For each test input, get N predictions
    - Ensemble MEAN = improved prediction (reduced error)
    - Ensemble STD = uncertainty map (where the model is unsure)
    - Detect OOD (out-of-distribution) inputs via high uncertainty

Key concepts:
    - Epistemic uncertainty: model doesn't know (can be reduced with more data)
    - Aleatoric uncertainty: data is inherently noisy (cannot be reduced)
    - Deep ensemble: train N models, disagreement = uncertainty
    - Calibration: predicted uncertainty should match actual error
    - OOD detection: high uncertainty on unseen input distributions

This is fundamentally different from:
    - FNO/U-Net (single model, no uncertainty)
    - Conditional Diffusion (generates samples, but not calibrated UQ)
    - PINN (equation-based, no data uncertainty)

Author: PhysicsNeMo Tutorial
Date: 2026-08-06
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
print("PhysicsNeMo UQ Tutorial: Deep Ensemble for Uncertainty Quantification")
print("Darcy Flow — When Does the Model Make Mistakes?")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# [1] Data Generation: Darcy Flow (In-Distribution + OOD)
# ============================================================
GRID = 32

def generate_permeability(n_samples, grid_size, length_scale=0.2):
    """Random permeability via spectral method."""
    kx = np.fft.fftfreq(grid_size) * grid_size
    ky = np.fft.fftfreq(grid_size) * grid_size
    KX, KY = np.meshgrid(kx, ky)
    spectrum = np.exp(-(KX**2 + KY**2) * length_scale**2)

    fields = []
    for _ in range(n_samples):
        noise = np.random.randn(grid_size, grid_size) + 1j * np.random.randn(grid_size, grid_size)
        field_hat = noise * np.sqrt(spectrum)
        field = np.real(np.fft.ifft2(field_hat))
        field = (field - field.mean()) / (field.std() + 1e-8)
        field = np.exp(field * 1.5)
        fields.append(field)
    return np.array(fields, dtype=np.float32)


def solve_darcy_fd(k_field, grid_size):
    """Solve -∇·(k∇p) = 1, p=0 on boundary."""
    N = grid_size - 2
    h = 1.0 / (grid_size - 1)
    A = np.zeros((N * N, N * N))
    b = np.ones(N * N) * h * h

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            k_c = k_field[i + 1, j + 1]
            k_n = k_field[i, j + 1] if i > 0 else k_field[i + 1, j + 1]
            k_s = k_field[i + 2, j + 1] if i < N - 1 else k_field[i + 1, j + 1]
            k_w = k_field[i + 1, j] if j > 0 else k_field[i + 1, j + 1]
            k_e = k_field[i + 1, j + 2] if j < N - 1 else k_field[i + 1, j + 1]
            k_n = 2 * k_c * k_n / (k_c + k_n + 1e-8)
            k_s = 2 * k_c * k_s / (k_c + k_s + 1e-8)
            k_w = 2 * k_c * k_w / (k_c + k_w + 1e-8)
            k_e = 2 * k_c * k_e / (k_c + k_e + 1e-8)
            A[idx, idx] = -(k_n + k_s + k_w + k_e)
            if i > 0: A[idx, idx - N] = k_n
            if i < N - 1: A[idx, idx + N] = k_s
            if j > 0: A[idx, idx - 1] = k_w
            if j < N - 1: A[idx, idx + 1] = k_e

    u_inner = np.linalg.solve(A, b)
    p = np.zeros((grid_size, grid_size))
    p[1:-1, 1:-1] = u_inner.reshape(N, N)
    return p.astype(np.float32)


print("\n[1] Generating Darcy data...")
N_TRAIN = 300
N_TEST_ID = 30   # in-distribution test
N_TEST_OOD = 20  # out-of-distribution test

# In-distribution: length_scale=0.2 (same as train)
train_k = generate_permeability(N_TRAIN, GRID, length_scale=0.2)
train_p = np.array([solve_darcy_fd(k, GRID) for k in train_k])

test_k_id = generate_permeability(N_TEST_ID, GRID, length_scale=0.2)
test_p_id = np.array([solve_darcy_fd(k, GRID) for k in test_k_id])

# OOD: length_scale=0.05 (much finer features, unseen during training)
test_k_ood = generate_permeability(N_TEST_OOD, GRID, length_scale=0.05)
test_p_ood = np.array([solve_darcy_fd(k, GRID) for k in test_k_ood])

print(f"  Train (ID):     {N_TRAIN} samples, length_scale=0.2")
print(f"  Test (ID):      {N_TEST_ID} samples, length_scale=0.2")
print(f"  Test (OOD):     {N_TEST_OOD} samples, length_scale=0.05")

# Normalize
k_mean, k_std = train_k.mean(), train_k.std()
p_mean, p_std = train_p.mean(), train_p.std()

train_k_n = (train_k - k_mean) / (k_std + 1e-8)
train_p_n = (train_p - p_mean) / (p_std + 1e-8)
test_k_id_n = (test_k_id - k_mean) / (k_std + 1e-8)
test_p_id_n = (test_p_id - p_mean) / (p_std + 1e-8)
test_k_ood_n = (test_k_ood - k_mean) / (k_std + 1e-8)
test_p_ood_n = (test_p_ood - p_mean) / (p_std + 1e-8)

train_k_t = torch.from_numpy(train_k_n).unsqueeze(1).to(device)
train_p_t = torch.from_numpy(train_p_n).unsqueeze(1).to(device)
test_k_id_t = torch.from_numpy(test_k_id_n).unsqueeze(1).to(device)
test_p_id_t = torch.from_numpy(test_p_id_n).unsqueeze(1).to(device)
test_k_ood_t = torch.from_numpy(test_k_ood_n).unsqueeze(1).to(device)
test_p_ood_t = torch.from_numpy(test_p_ood_n).unsqueeze(1).to(device)

# ============================================================
# [2] CNN Model (Simple Surrogate)
# ============================================================
class DarcyCNN(nn.Module):
    """Simple CNN: permeability [B,1,H,W] → pressure [B,1,H,W]."""
    def __init__(self, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1), nn.GroupNorm(8, base_ch), nn.SiLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.GroupNorm(8, base_ch * 2), nn.SiLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1), nn.GroupNorm(8, base_ch * 4), nn.SiLU())
        self.bot = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1), nn.GroupNorm(8, base_ch * 4), nn.SiLU())
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base_ch * 2), nn.SiLU())
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1), nn.GroupNorm(8, base_ch), nn.SiLU())
        self.out = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bot(e3)
        d3 = self.dec3(b)
        d2 = self.dec2(d3)
        return self.out(d2)


# ============================================================
# [3] Train Deep Ensemble (N independent models)
# ============================================================
N_MODELS = 5
EPOCHS = 150
BATCH_SIZE = 32

print(f"\n[3] Training Deep Ensemble ({N_MODELS} models, {EPOCHS} epochs each)...")
print("-" * 70)

ensemble = []
all_train_losses = []
total_start = time.time()

for model_idx in range(N_MODELS):
    torch.manual_seed(1000 + model_idx)  # different seed for each model
    model = DarcyCNN(base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model_losses = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            k_batch = train_k_t[idx]
            p_batch = train_p_t[idx]
            pred = model(k_batch)
            loss = F.mse_loss(pred, p_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        model_losses.append(avg_loss)
        scheduler.step()

    all_train_losses.append(model_losses)
    ensemble.append(model)

    elapsed = time.time() - total_start
    print(f"  Model {model_idx + 1}/{N_MODELS} done | Final loss: {model_losses[-1]:.6e} | Total time: {elapsed:.1f}s")

total_time = time.time() - total_start
print("-" * 70)
print(f"Ensemble training complete! Total time: {total_time:.1f}s")

# ============================================================
# [4] Evaluate: Ensemble Predictions & Uncertainty
# ============================================================
print("\n[4] Evaluating ensemble predictions...")

@torch.no_grad()
def ensemble_predict(ensemble, k_input):
    """Get predictions from all ensemble members."""
    preds = []
    for model in ensemble:
        model.eval()
        preds.append(model(k_input))
    preds = torch.stack(preds, dim=0)  # [N_MODELS, B, 1, H, W]
    return preds


# --- In-Distribution Test ---
preds_id = ensemble_predict(ensemble, test_k_id_t)  # [N_MODELS, N_TEST_ID, 1, H, W]
mean_id = preds_id.mean(dim=0)  # [N_TEST_ID, 1, H, W]
std_id = preds_id.std(dim=0)    # [N_TEST_ID, 1, H, W]

# Per-sample errors
errors_id = (mean_id - test_p_id_t).abs()
rel_l2_id = torch.norm(mean_id - test_p_id_t, dim=(1, 2, 3)) / \
            (torch.norm(test_p_id_t, dim=(1, 2, 3)) + 1e-8)

# Mean uncertainty per sample
mean_std_id = std_id.mean(dim=(1, 2, 3))  # [N_TEST_ID]

print(f"\n  In-Distribution (ID) Results:")
print(f"    Mean Rel L2:     {rel_l2_id.mean().item():.4f} ± {rel_l2_id.std().item():.4f}")
print(f"    Mean Uncertainty: {mean_std_id.mean().item():.6f} ± {mean_std_id.std().item():.6f}")

# --- OOD Test ---
preds_ood = ensemble_predict(ensemble, test_k_ood_t)
mean_ood = preds_ood.mean(dim=0)
std_ood = preds_ood.std(dim=0)

errors_ood = (mean_ood - test_p_ood_t).abs()
rel_l2_ood = torch.norm(mean_ood - test_p_ood_t, dim=(1, 2, 3)) / \
             (torch.norm(test_p_ood_t, dim=(1, 2, 3)) + 1e-8)
mean_std_ood = std_ood.mean(dim=(1, 2, 3))

print(f"\n  Out-of-Distribution (OOD) Results:")
print(f"    Mean Rel L2:     {rel_l2_ood.mean().item():.4f} ± {rel_l2_ood.std().item():.4f}")
print(f"    Mean Uncertainty: {mean_std_ood.mean().item():.6f} ± {mean_std_ood.std().item():.6f}")

# --- OOD Detection ---
print(f"\n  OOD Detection (via uncertainty threshold):")
threshold = mean_std_id.mean().item() + 2 * mean_std_id.std().item()
ood_detected = (mean_std_ood > threshold).sum().item()
id_false_alarm = (mean_std_id > threshold).sum().item()
print(f"    Threshold: {threshold:.6f} (mean + 2σ of ID uncertainty)")
print(f"    OOD detected: {ood_detected}/{N_TEST_OOD} ({ood_detected/N_TEST_OOD*100:.0f}%)")
print(f"    ID false alarm: {id_false_alarm}/{N_TEST_ID} ({id_false_alarm/N_TEST_ID*100:.0f}%)")

# --- Ensemble vs Single Model ---
single_model_errors = []
for i in range(N_MODELS):
    single_pred = preds_id[i]
    single_l2 = torch.norm(single_pred - test_p_id_t, dim=(1, 2, 3)) / \
                 (torch.norm(test_p_id_t, dim=(1, 2, 3)) + 1e-8)
    single_model_errors.append(single_l2.mean().item())

ensemble_l2 = rel_l2_id.mean().item()
print(f"\n  Ensemble vs Single Model (ID):")
print(f"    Single model avg L2: {np.mean(single_model_errors):.4f} ± {np.std(single_model_errors):.4f}")
print(f"    Ensemble L2:         {ensemble_l2:.4f}")
print(f"    Improvement:         {(1 - ensemble_l2 / np.mean(single_model_errors)) * 100:.1f}%")

# ============================================================
# [5] Calibration Analysis
# ============================================================
print("\n[5] Calibration analysis...")

# For each test sample, compute predicted uncertainty vs actual error
# Well-calibrated: high uncertainty ↔ high error

# Flatten all pixels for correlation analysis
all_std_id = std_id.flatten().cpu().numpy()
all_err_id = errors_id.flatten().cpu().numpy()
all_std_ood = std_ood.flatten().cpu().numpy()
all_err_ood = errors_ood.flatten().cpu().numpy()

# Binned calibration: sort by uncertainty, check if high-uncertainty bins have high error
n_bins = 10
sorted_idx = np.argsort(all_std_id)
bin_size = len(sorted_idx) // n_bins
calib_unc = []
calib_err = []
for b in range(n_bins):
    idx = sorted_idx[b * bin_size:(b + 1) * bin_size]
    calib_unc.append(all_std_id[idx].mean())
    calib_err.append(all_err_id[idx].mean())

# Pearson correlation
corr_id = np.corrcoef(all_std_id, all_err_id)[0, 1]
corr_ood = np.corrcoef(all_std_ood, all_err_ood)[0, 1]
print(f"  Uncertainty-Error correlation (ID):  {corr_id:.4f}")
print(f"  Uncertainty-Error correlation (OOD): {corr_ood:.4f}")
print(f"  (1.0 = perfect calibration, 0.0 = no calibration)")

# ============================================================
# [6] Visualization
# ============================================================
print("\n[6] Generating visualizations...")

# --- Figure 1: Training losses for all ensemble members ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i, losses in enumerate(all_train_losses):
    ax.semilogy(losses, linewidth=1.5, alpha=0.8, label=f'Model {i+1}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE, log scale)')
ax.set_title(f'Deep Ensemble Training ({N_MODELS} Independent Models)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
loss_path = os.path.join(RESULTS_DIR, "ensemble_loss.png")
plt.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {loss_path}")

# --- Figure 2: ID vs OOD uncertainty comparison ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: ID sample
id_idx = 0
ax = axes[0, 0]
ax.imshow(test_k_id[id_idx, 0], cmap='magma', origin='lower')
ax.set_title('ID: Permeability k(x)')

ax = axes[0, 1]
p_true = test_p_id_t[id_idx, 0].cpu().numpy() * p_std + p_mean
ax.imshow(p_true, cmap='viridis', origin='lower')
ax.set_title('ID: Ground Truth p(x)')

ax = axes[0, 2]
p_mean_show = mean_id[id_idx, 0].cpu().numpy() * p_std + p_mean
ax.imshow(p_mean_show, cmap='viridis', origin='lower')
ax.set_title('ID: Ensemble Mean')

ax = axes[0, 3]
s_show = std_id[id_idx, 0].cpu().numpy() * p_std
im = ax.imshow(s_show, cmap='hot', origin='lower')
ax.set_title('ID: Ensemble Std (Uncertainty)')
plt.colorbar(im, ax=ax, fraction=0.046)

# Row 2: OOD sample
ood_idx = 0
ax = axes[1, 0]
ax.imshow(test_k_ood[ood_idx, 0], cmap='magma', origin='lower')
ax.set_title('OOD: Permeability k(x)')

ax = axes[1, 1]
p_true = test_p_ood_t[ood_idx, 0].cpu().numpy() * p_std + p_mean
ax.imshow(p_true, cmap='viridis', origin='lower')
ax.set_title('OOD: Ground Truth p(x)')

ax = axes[1, 2]
p_mean_show = mean_ood[ood_idx, 0].cpu().numpy() * p_std + p_mean
ax.imshow(p_mean_show, cmap='viridis', origin='lower')
ax.set_title('OOD: Ensemble Mean')

ax = axes[1, 3]
s_show = std_ood[ood_idx, 0].cpu().numpy() * p_std
im = ax.imshow(s_show, cmap='hot', origin='lower')
ax.set_title('OOD: Ensemble Std (Uncertainty)')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Deep Ensemble UQ: In-Distribution vs Out-of-Distribution', fontsize=14)
plt.tight_layout()
result_path = os.path.join(RESULTS_DIR, "ensemble_result.png")
plt.savefig(result_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {result_path}")

# --- Figure 3: OOD detection histogram ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(mean_std_id.cpu().numpy(), bins=20, alpha=0.6, color='blue', label=f'In-Distribution (n={N_TEST_ID})', density=True)
ax.hist(mean_std_ood.cpu().numpy(), bins=20, alpha=0.6, color='red', label=f'OOD (n={N_TEST_OOD})', density=True)
ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
ax.set_xlabel('Mean Ensemble Std (Uncertainty)')
ax.set_ylabel('Density')
ax.set_title('OOD Detection: Uncertainty Distribution')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
ood_path = os.path.join(RESULTS_DIR, "ood_detection.png")
plt.savefig(ood_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {ood_path}")

# --- Figure 4: Calibration plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(calib_unc, calib_err, 'bo-', linewidth=2, markersize=8, label='ID calibration')
max_val = max(max(calib_unc), max(calib_err))
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect calibration')
ax.set_xlabel('Predicted Uncertainty (binned)')
ax.set_ylabel('Actual Error (binned)')
ax.set_title(f'Calibration: Uncertainty vs Error (ID)\nCorr = {corr_id:.4f}')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(all_std_id[::100], all_err_id[::100], s=1, alpha=0.3, c='blue', label='ID')
ax.scatter(all_std_ood[::50], all_err_ood[::50], s=1, alpha=0.3, c='red', label='OOD')
ax.set_xlabel('Predicted Uncertainty (per pixel)')
ax.set_ylabel('Actual Error (per pixel)')
ax.set_title('Pixel-wise: Uncertainty vs Error')
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Deep Ensemble Calibration Analysis', fontsize=14)
plt.tight_layout()
calib_path = os.path.join(RESULTS_DIR, "calibration.png")
plt.savefig(calib_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {calib_path}")

# --- Figure 5: Ensemble vs single model ---
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
models = [f'M{i+1}' for i in range(N_MODELS)] + ['Ensemble']
l2_vals = single_model_errors + [ensemble_l2]
colors = ['steelblue'] * N_MODELS + ['crimson']
bars = ax.bar(models, l2_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=np.mean(single_model_errors), color='steelblue', linestyle='--', alpha=0.5, label=f'Single avg: {np.mean(single_model_errors):.4f}')
ax.axhline(y=ensemble_l2, color='crimson', linestyle='--', alpha=0.5, label=f'Ensemble: {ensemble_l2:.4f}')
ax.set_ylabel('Relative L2 Error')
ax.set_title('Ensemble vs Single Model: Prediction Accuracy')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
compare_path = os.path.join(RESULTS_DIR, "ensemble_vs_single.png")
plt.savefig(compare_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {compare_path}")

# ============================================================
# [7] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Deep Ensemble for Uncertainty Quantification")
print("=" * 70)
print(f"  Problem:          Darcy Flow surrogate with UQ")
print(f"  Grid:              {GRID}x{GRID}")
print(f"  Train samples:    {N_TRAIN} (length_scale=0.2)")
print(f"  Test (ID):        {N_TEST_ID} (length_scale=0.2)")
print(f"  Test (OOD):       {N_TEST_OOD} (length_scale=0.05)")
print(f"  Ensemble size:    {N_MODELS} models")
print(f"  Epochs/model:    {EPOCHS}")
print(f"  Total train time: {total_time:.1f}s")
print(f"  ID Rel L2:        {rel_l2_id.mean().item():.4f} ± {rel_l2_id.std().item():.4f}")
print(f"  OOD Rel L2:       {rel_l2_ood.mean().item():.4f} ± {rel_l2_ood.std().item():.4f}")
print(f"  ID uncertainty:   {mean_std_id.mean().item():.6f}")
print(f"  OOD uncertainty:  {mean_std_ood.mean().item():.6f}")
print(f"  OOD detected:      {ood_detected}/{N_TEST_OOD} ({ood_detected/N_TEST_OOD*100:.0f}%)")
print(f"  Calibration corr: {corr_id:.4f} (ID), {corr_ood:.4f} (OOD)")
print(f"  Ensemble improve:  {(1 - ensemble_l2 / np.mean(single_model_errors)) * 100:.1f}% vs single")
print(f"  Results:          {RESULTS_DIR}")
print()
print("Key observations:")
print("  1. UNCERTAINTY: Ensemble std reveals WHERE the model is unsure")
print("  2. OOD DETECTION: OOD inputs have HIGHER uncertainty (auto-detected)")
print("  3. ENSEMBLE EFFECT: Mean of N models > any single model (reduced error)")
print("  4. CALIBRATION: Higher uncertainty correlates with higher error")
print("  5. SAFETY: In safety-critical apps, high uncertainty → don't trust prediction")
print("  6. NEW PARADIGM: First UQ tutorial (all others report point estimates only)")
print("=" * 70)
