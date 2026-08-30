"""
Active Learning for PDE Surrogate Models
=========================================
This tutorial implements active learning (uncertainty-based sampling)
for efficient PDE surrogate training.

Existing tutorial (topology_optimization):
  - Design optimization (generative model for structural design)
  - Goal: find optimal material distribution

THIS tutorial:
  - Active learning (data-efficient surrogate training)
  - Goal: minimize simulation cost by smart sample selection
  - Which samples to simulate next for maximum information gain?

Key concepts:
  1. Active learning loop: train → predict uncertainty → select → simulate → retrain
  2. Acquisition functions: uncertainty sampling, expected improvement
  3. Sample efficiency: fewer simulations for same accuracy
  4. Oracle: expensive simulator (FDM solver) queried selectively
  5. Exploration vs exploitation: balance uncertain vs promising regions

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
print("Active Learning for PDE Surrogate Models")
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
# [1] Problem Setup
# ============================================================
# We want to train a surrogate model for Darcy flow.
# Each simulation (FDM solve) is "expensive" (takes time).
# Active learning: start with few samples, selectively add more.

N_GRID = 32
N_INITIAL = 20       # Start with 20 samples
N_PER_ROUND = 10     # Add 10 samples per round
N_ROUNDS = 8         # 8 rounds of active learning
N_POOL = 200          # Pool of unlabeled candidates
N_TEST = 50           # Test set

print(f"\n[1] Problem: Active learning for Darcy surrogate")
print(f"  Grid: {N_GRID}x{N_GRID}")
print(f"  Initial samples: {N_INITIAL}")
print(f"  Per round: {N_PER_ROUND} new samples")
print(f"  Rounds: {N_ROUNDS}")
print(f"  Pool: {N_POOL} unlabeled candidates")
print(f"  Total budget: {N_INITIAL + N_PER_ROUND * N_ROUNDS} simulations")

# ============================================================
# [2] Data Generation (Oracle = FDM solver)
# ============================================================
def generate_gaussian_field(n, length_scale=0.15, rng=None):
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

print(f"\n[2] Generating data pool and test set...")

# Generate pool (unlabeled candidates) and test set
rng = np.random.RandomState(42)
pool_k = np.array([generate_gaussian_field(N_GRID, 0.15, rng) for _ in range(N_POOL)])
test_k = np.array([generate_gaussian_field(N_GRID, 0.15, rng) for _ in range(N_TEST)])
test_p = np.array([solve_darcy(test_k[i], N_GRID) for i in range(N_TEST)])

# Normalize
k_mean, k_std = pool_k.mean(), pool_k.std()
p_mean, p_std = test_p.mean(), test_p.std()

pool_k_n = (pool_k - k_mean) / (k_std + 1e-8)
test_k_n = (test_k - k_mean) / (k_std + 1e-8)
test_p_n = (test_p - p_mean) / (p_std + 1e-8)

test_k_t = torch.from_numpy(test_k_n).float().unsqueeze(1).to(device)
test_p_t = torch.from_numpy(test_p_n).float().unsqueeze(1).to(device)

print(f"  Pool: {N_POOL}, Test: {N_TEST}")

# ============================================================
# [3] Surrogate Model (CNN with MC Dropout for uncertainty)
# ============================================================
class SurrogateCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, hidden=32, dropout_p=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.SiLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(hidden * 2, hidden * 4, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, out_ch, 3, padding=1),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

print(f"\n[3] Surrogate: CNN with MC Dropout (p=0.1)")

# ============================================================
# [4] Active Learning Loop
# ============================================================
EPOCHS_PER_ROUND = 100
T_MC = 20  # MC samples for uncertainty

print(f"\n[4] Active learning loop")
print("-" * 70)

# Initialize: randomly select N_INITIAL samples from pool
labeled_indices = list(range(N_INITIAL))
unlabeled_indices = list(range(N_INITIAL, N_POOL))

# Track metrics
al_history = {'n_samples': [], 'test_mae': [], 'test_unc': []}
random_history = {'n_samples': [], 'test_mae': [], 'test_unc': []}

def train_surrogate(k_train, p_train, epochs=100):
    """Train surrogate model."""
    model = SurrogateCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(k_train)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i+32]
            k = k_train[idx]
            p = p_train[idx]
            pred = model(k)
            loss = F.mse_loss(pred, p)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def mc_predict(model, x, T=20):
    """MC Dropout prediction: mean and std."""
    model.train()  # Keep dropout ON
    preds = torch.stack([model(x) for _ in range(T)])
    return preds.mean(dim=0), preds.std(dim=0)

def evaluate(model, k_test, p_test):
    """Evaluate on test set."""
    model.eval()
    with torch.no_grad():
        pred = model(k_test)
        mae = torch.mean(torch.abs(pred - p_test)).item()
    mean_pred, std_pred = mc_predict(model, k_test, T_MC)
    unc = std_pred.mean().item()
    return mae, unc

# --- Active Learning (uncertainty sampling) ---
print("\n  Active Learning (uncertainty sampling):")
for round_idx in range(N_ROUNDS + 1):
    # Prepare training data
    k_train = torch.from_numpy(pool_k_n[labeled_indices]).float().unsqueeze(1).to(device)
    p_train = torch.from_numpy(
        np.array([solve_darcy(pool_k[i], N_GRID) for i in labeled_indices])
    ).float().unsqueeze(1).to(device)
    p_train_n = (p_train - p_mean) / (p_std + 1e-8)

    # Train
    model = train_surrogate(k_train, p_train_n, EPOCHS_PER_ROUND)

    # Evaluate
    mae, unc = evaluate(model, test_k_t, test_p_t)
    al_history['n_samples'].append(len(labeled_indices))
    al_history['test_mae'].append(mae)
    al_history['test_unc'].append(unc)
    print(f"    Round {round_idx} | Samples: {len(labeled_indices):3d} | Test MAE: {mae:.6f} | Unc: {unc:.6f}")

    if round_idx == N_ROUNDS:
        break

    # Select next samples (uncertainty sampling)
    if len(unlabeled_indices) > 0:
        k_pool = torch.from_numpy(pool_k_n[unlabeled_indices]).float().unsqueeze(1).to(device)
        _, pool_unc = mc_predict(model, k_pool, T_MC)
        pool_unc_mean = pool_unc.view(len(unlabeled_indices), -1).mean(dim=1)
        # Select top-N most uncertain
        n_select = min(N_PER_ROUND, len(unlabeled_indices))
        top_indices = torch.argsort(pool_unc_mean, descending=True)[:n_select]
        selected = [unlabeled_indices[i] for i in top_indices.cpu().numpy()]
        labeled_indices.extend(selected)
        unlabeled_indices = [i for i in unlabeled_indices if i not in selected]

# --- Random sampling baseline ---
print("\n  Random sampling (baseline):")
random_labeled = list(range(N_INITIAL))
random_unlabeled = list(range(N_INITIAL, N_POOL))

for round_idx in range(N_ROUNDS + 1):
    k_train = torch.from_numpy(pool_k_n[random_labeled]).float().unsqueeze(1).to(device)
    p_train = torch.from_numpy(
        np.array([solve_darcy(pool_k[i], N_GRID) for i in random_labeled])
    ).float().unsqueeze(1).to(device)
    p_train_n = (p_train - p_mean) / (p_std + 1e-8)

    model = train_surrogate(k_train, p_train_n, EPOCHS_PER_ROUND)
    mae, unc = evaluate(model, test_k_t, test_p_t)
    random_history['n_samples'].append(len(random_labeled))
    random_history['test_mae'].append(mae)
    random_history['test_unc'].append(unc)
    print(f"    Round {round_idx} | Samples: {len(random_labeled):3d} | Test MAE: {mae:.6f} | Unc: {unc:.6f}")

    if round_idx == N_ROUNDS:
        break

    if len(random_unlabeled) > 0:
        n_select = min(N_PER_ROUND, len(random_unlabeled))
        selected = np.random.choice(random_unlabeled, n_select, replace=False).tolist()
        random_labeled.extend(selected)
        random_unlabeled = [i for i in random_unlabeled if i not in selected]

print("-" * 70)

# ============================================================
# [5] Visualization
# ============================================================
print(f"\n[5] Generating visualizations...")

# --- Figure 1: Active Learning vs Random — MAE ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(al_history['n_samples'], al_history['test_mae'], 'o-', color='red', linewidth=2, markersize=8, label='Active Learning')
ax.plot(random_history['n_samples'], random_history['test_mae'], 's--', color='blue', linewidth=2, markersize=8, label='Random Sampling')
ax.set_xlabel('Number of Labeled Samples'); ax.set_ylabel('Test MAE')
ax.set_title('Test MAE: Active Learning vs Random'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(al_history['n_samples'], al_history['test_unc'], 'o-', color='red', linewidth=2, markersize=8, label='Active Learning')
ax.plot(random_history['n_samples'], random_history['test_unc'], 's--', color='blue', linewidth=2, markersize=8, label='Random Sampling')
ax.set_xlabel('Number of Labeled Samples'); ax.set_ylabel('Test Uncertainty')
ax.set_title('Test Uncertainty: Active Learning vs Random'); ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Active Learning: Sample Efficiency', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "active_learning_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Sample selection visualization ---
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for round_idx in range(min(8, N_ROUNDS + 1)):
    ax = axes[round_idx // 4, round_idx % 4]
    # Show which samples were selected (by permeability field mean)
    n_samples = al_history['n_samples'][round_idx]
    ax.bar(['Active', 'Random'], [al_history['test_mae'][round_idx], random_history['test_mae'][round_idx]],
           color=['red', 'blue'], alpha=0.7, edgecolor='black')
    ax.set_title(f'Round {round_idx} ({n_samples} samples)', fontsize=10)
    ax.set_ylabel('Test MAE'); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Active Learning vs Random: MAE per Round', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "active_learning_rounds.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: Final predictions ---
model.eval()
with torch.no_grad():
    pred_final = model(test_k_t[:6])

fig, axes = plt.subplots(3, 6, figsize=(20, 10))
for i in range(6):
    ax = axes[0, i]
    im = ax.imshow(test_p[i], cmap='jet', origin='lower')
    ax.set_title(f'True {i+1}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, i]
    im = ax.imshow(pred_final[i, 0].cpu().numpy() * p_std + p_mean, cmap='jet', origin='lower')
    ax.set_title(f'Predicted {i+1}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2, i]
    diff = np.abs(pred_final[i, 0].cpu().numpy() * p_std + p_mean - test_p[i])
    im = ax.imshow(diff, cmap='hot', origin='lower')
    ax.set_title(f'|Error| {i+1}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle(f'Active Learning Final Predictions ({al_history["n_samples"][-1]} samples)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "active_learning_predictions.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: Concept explanation ---
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.text(0.5, 0.95, 'Active Learning for PDE Surrogates', ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.80,
    'Topology Optimization (existing tutorial):\n'
    '  - Goal: find optimal design\n'
    '  - Method: generative model\n'
    '  - Data: pre-computed optimal solutions\n'
    '  - No sample selection',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(0.55, 0.80,
    'Active Learning (THIS tutorial):\n'
    '  - Goal: efficient surrogate training\n'
    '  - Method: uncertainty-based sampling\n'
    '  - Data: selectively queried from oracle\n'
    '  - Smart sample selection',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.50,
    'Active learning loop:\n'
    '  1. Train surrogate on labeled data\n'
    '  2. Predict on unlabeled pool\n'
    '  3. Compute uncertainty (MC Dropout)\n'
    '  4. Select most uncertain samples\n'
    '  5. Query oracle (FDM solver)\n'
    '  6. Add to training set → retrain',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.55, 0.50,
    'Acquisition functions:\n'
    '  1. Uncertainty sampling:\n'
    '     select max std(pred)\n'
    '  2. Expected improvement:\n'
    '     E[max(f(x) - f*)]\n'
    '  3. Query-by-committee:\n'
    '     max disagreement\n'
    '  This tutorial: #1 (uncertainty)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.20,
    'Why active learning?\n'
    '  - Simulations are expensive\n'
    '  - Labeling costs time/money\n'
    '  - Smart selection = fewer sims\n'
    '  - Same accuracy, less data\n'
    '  - Critical for industry',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax.text(0.55, 0.20,
    'Key concepts:\n'
    '  1. Oracle = expensive simulator\n'
    '  2. Pool = unlabeled candidates\n'
    '  3. Acquisition = selection criterion\n'
    '  4. Exploration vs exploitation\n'
    '  5. Sample efficiency metric\n'
    '  6. MC Dropout for uncertainty',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightskyblue', alpha=0.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "active_learning_explanation.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [6] Summary
# ============================================================
final_al_mae = al_history['test_mae'][-1]
final_rand_mae = random_history['test_mae'][-1]
improvement = (final_rand_mae - final_al_mae) / final_rand_mae * 100

print("\n" + "=" * 70)
print("SUMMARY: Active Learning for PDE Surrogate")
print("=" * 70)
print(f"  Problem:           Darcy Flow ({N_GRID}x{N_GRID})")
print(f"  Initial samples:   {N_INITIAL}")
print(f"  Per round:         {N_PER_ROUND}")
print(f"  Rounds:            {N_ROUNDS}")
print(f"  Total samples:     {N_INITIAL + N_PER_ROUND * N_ROUNDS}")
print(f"  --- Final Results ---")
print(f"  Active Learning MAE: {final_al_mae:.6f}")
print(f"  Random Sampling MAE: {final_rand_mae:.6f}")
print(f"  Improvement:         {improvement:.1f}% (AL vs Random)")
print()
print("Key observations:")
print("  1. ACTIVE LEARNING: Selectively query oracle for most informative samples")
print("  2. UNCERTAINTY SAMPLING: Select samples where model is most uncertain")
print("  3. SAMPLE EFFICIENCY: Fewer simulations for same accuracy")
print("  4. vs TOPOLOGY OPT: Existing tutorial optimizes design; this optimizes data collection")
print("  5. ORACLE: FDM solver = expensive; active learning minimizes oracle calls")
print("  6. MC DROPOUT: Used for uncertainty estimation in acquisition function")
print("=" * 70)
