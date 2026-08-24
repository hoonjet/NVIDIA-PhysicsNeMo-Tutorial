"""
Score-Based Generative Model for Stochastic PDE Solutions
==========================================================
This tutorial implements a Score-Based Generative Model using
continuous-time SDEs (Song et al., 2021) for stochastic Darcy flow.

Existing tutorial (conditional_diffusion):
  - DDPM: discrete denoising steps (200 steps)
  - Noise schedule: beta_1, ..., beta_T (discrete)
  - Sampling: reverse Markov chain (discrete)

THIS tutorial:
  - Score-Based: continuous SDE (Variance Exploding)
  - Noise schedule: sigma(t) = sigma_min * (sigma_max/sigma_min)^t (continuous)
  - Score function: s(x, t) = ∇log p_t(x) learned by neural network
  - Sampling: reverse-time SDE or probability flow ODE (continuous)

Key concepts:
  1. Score function: ∇log p(x) — gradient of log probability density
  2. Forward SDE: add noise continuously (diffusion process)
  3. Reverse SDE: denoise using learned score function
  4. Score matching: train network to predict score at each noise level
  5. Probability flow ODE: deterministic alternative to reverse SDE

This is the ONLY tutorial that:
  - Uses continuous-time SDE (not discrete DDPM)
  - Learns score function directly (not noise prediction)
  - Supports both reverse SDE and probability flow ODE sampling
  - Compares score-based vs DDPM approaches

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
print("Score-Based Generative Model for Stochastic PDE Solutions")
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
# [1] Stochastic Darcy Flow Data Generation
# ============================================================
# Same PDE as conditional_diffusion tutorial, but we use it
# to demonstrate a DIFFERENT generative model (score-based SDE).

N_GRID = 32
N_TRAIN = 200
N_TEST = 20

print(f"\n[1] Generating stochastic Darcy flow data")
print(f"  Grid: {N_GRID}x{N_GRID}")
print(f"  Train: {N_TRAIN}, Test: {N_TEST}")

def generate_gaussian_field(n, length_scale=0.1, rng=None):
    """Generate random Gaussian field via spectral method."""
    if rng is None:
        rng = np.random.RandomState()
    x = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    # Spectral method: sum of random sinusoids
    field = np.zeros((n, n))
    n_modes = 20
    for _ in range(n_modes):
        kx = rng.randint(-3, 4)
        ky = rng.randint(-3, 4)
        amp = rng.randn() * np.exp(-np.sqrt(kx**2 + ky**2) * length_scale * 10)
        phase = rng.uniform(0, 2 * np.pi)
        field += amp * np.sin(2 * np.pi * (kx * xx + ky * yy) + phase)
    # Normalize and exponentiate for permeability
    field = (field - field.mean()) / (field.std() + 1e-8)
    k = np.exp(field * 1.5)  # Log-normal permeability
    return k

def solve_darcy(k, n):
    """Solve -div(k * grad(p)) = 1 with p=0 on boundary (FDM)."""
    dx = 1.0 / (n - 1)
    p = np.zeros((n, n))
    # Simple Jacobi iteration
    for _ in range(500):
        p_new = p.copy()
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                k_e = 0.5 * (k[i, j] + k[i + 1, j])
                k_w = 0.5 * (k[i, j] + k[i - 1, j])
                k_n = 0.5 * (k[i, j] + k[i, j + 1])
                k_s = 0.5 * (k[i, j] + k[i, j - 1])
                p_new[i, j] = (k_e * p[i + 1, j] + k_w * p[i - 1, j] +
                               k_n * p[i, j + 1] + k_s * p[i, j - 1] +
                               dx * dx) / (k_e + k_w + k_n + k_s)
        p = p_new
    return p

print("  Generating training data...")
train_k = np.array([generate_gaussian_field(N_GRID, 0.1) for _ in range(N_TRAIN)])
train_p = np.array([solve_darcy(train_k[i], N_GRID) for i in range(N_TRAIN)])
test_k = np.array([generate_gaussian_field(N_GRID, 0.1) for _ in range(N_TEST)])
test_p = np.array([solve_darcy(test_k[i], N_GRID) for i in range(N_TEST)])

# Normalize
p_mean, p_std = train_p.mean(), train_p.std()
train_p_n = (train_p - p_mean) / (p_std + 1e-8)
test_p_n = (test_p - p_mean) / (p_std + 1e-8)
k_mean, k_std = train_k.mean(), train_k.std()
train_k_n = (train_k - k_mean) / (k_std + 1e-8)
test_k_n = (test_k - k_mean) / (k_std + 1e-8)

train_p_t = torch.from_numpy(train_p_n).float().to(device)
train_k_t = torch.from_numpy(train_k_n).float().to(device)
test_p_t = torch.from_numpy(test_p_n).float().to(device)
test_k_t = torch.from_numpy(test_k_n).float().to(device)

print(f"  Train p shape: {train_p_t.shape}")
print(f"  p range: [{train_p_n.min():.3f}, {train_p_n.max():.3f}]")

# ============================================================
# [2] Score Network
# ============================================================
# The score network s(x, t, c) predicts the score function:
#   s ≈ ∇_x log p_t(x | c)
# where c is the conditioning (permeability field).
# Input: noisy solution x_t, time t, condition c
# Output: score ∇_x log p_t(x | c)

class ScoreNet(nn.Module):
    """
    U-Net based score network with time embedding.
    Input: [x_t (1ch), c (1ch)] = 2 channels, time t (scalar)
    Output: score (1 channel)
    """
    def __init__(self, in_ch=2, out_ch=1, hidden=64):
        super().__init__()
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden * 4)
        )
        # Encoder
        self.enc1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.enc2 = nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(hidden * 2, hidden * 4, 3, stride=2, padding=1)
        # Middle (with time modulation)
        self.mid = nn.Conv2d(hidden * 4, hidden * 4, 3, padding=1)
        self.mid_time = nn.Linear(hidden * 4, hidden * 4)
        # Decoder
        self.dec3 = nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(hidden, out_ch, 3, padding=1)
        # Skip connections
        self.skip3 = nn.Conv2d(hidden * 4, hidden * 2, 1)
        self.skip2 = nn.Conv2d(hidden * 2, hidden, 1)

    def forward(self, x, t, c):
        # x: [B, 1, H, W], t: [B], c: [B, 1, H, W]
        inp = torch.cat([x, c], dim=1)  # [B, 2, H, W]
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, hidden*4]
        # Encode
        h1 = F.silu(self.enc1(inp))           # [B, hidden, H, W]
        h2 = F.silu(self.enc2(h1))            # [B, hidden*2, H/2, W/2]
        h3 = F.silu(self.enc3(h2))            # [B, hidden*4, H/4, W/4]
        # Middle with time modulation
        h = self.mid(h3)
        h = h + self.mid_time(t_emb).unsqueeze(-1).unsqueeze(-1)  # Add time info
        h = F.silu(h)
        # Decode with skip connections
        d3 = F.silu(self.dec3(h) + self.skip3(h3))  # [B, hidden*2, H/2, W/2]
        d2 = F.silu(self.dec2(d3) + self.skip2(h2))  # [B, hidden, H, W]
        out = self.dec1(d2)                            # [B, out_ch, H, W]
        return out

print(f"\n[2] Building score network...")
score_net = ScoreNet(in_ch=2, out_ch=1, hidden=64).to(device)
n_params = sum(p.numel() for p in score_net.parameters())
print(f"  ScoreNet parameters: {n_params:,}")
print(f"  Input: [x_t (1ch), c (1ch)] + time t")
print(f"  Output: score ∇_x log p_t(x|c)")

# ============================================================
# [3] SDE Configuration (Variance Exploding)
# ============================================================
# Forward SDE: dx = sqrt(d[sigma^2(t)]/dt) * dw
#   => x_t = x_0 + sigma(t) * epsilon,  epsilon ~ N(0, I)
# Score: s(x, t) = -epsilon / sigma(t)  (analytical score for Gaussian)
# Reverse SDE: dx = -sigma^2(t) * s(x, t) dt - sigma(t) * dw

SIGMA_MIN = 0.01
SIGMA_MAX = 50.0

def sigma(t):
    """Continuous noise schedule: sigma(t) = sigma_min * (sigma_max/sigma_min)^t"""
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t

def sigma_dot(t):
    """Derivative of sigma(t) w.r.t. t"""
    return sigma(t) * np.log(SIGMA_MAX / SIGMA_MIN)

print(f"\n[3] SDE: Variance Exploding (VE)")
print(f"  sigma_min: {SIGMA_MIN}, sigma_max: {SIGMA_MAX}")
print(f"  sigma(t) = {SIGMA_MIN} * ({SIGMA_MAX}/{SIGMA_MIN})^t")
print(f"  Forward:  x_t = x_0 + sigma(t) * epsilon")
print(f"  Score:    s(x,t) = -epsilon / sigma(t)")
print(f"  Reverse:  dx = -sigma^2(t)*s(x,t)*dt - sigma(t)*dw")

# ============================================================
# [4] Training: Score Matching
# ============================================================
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3

print(f"\n[4] Training score network ({EPOCHS} epochs)")
print(f"    Loss: Denoising score matching")
print("-" * 70)

opt = torch.optim.Adam(score_net.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

train_losses, test_losses = [], []
start = time.time()

for epoch in range(EPOCHS):
    score_net.train()
    perm = torch.randperm(N_TRAIN)
    epoch_loss = 0; n_batches = 0

    for i in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        bs = len(idx)

        p_batch = train_p_t[idx]  # [bs, H, W]
        k_batch = train_k_t[idx]  # [bs, H, W]

        # Sample random time t ~ U(0, 1)
        t = torch.rand(bs, device=device)  # [bs]

        # Sample noise
        eps = torch.randn_like(p_batch)  # [bs, H, W]

        # Perturb: x_t = x_0 + sigma(t) * eps
        sig = sigma(t)  # [bs]
        sig = sig.view(bs, 1, 1)  # broadcast
        x_t = p_batch + sig * eps

        # Target score: s = -eps / sigma(t)
        target_score = -eps / sig

        # Predict score
        pred_score = score_net(
            x_t.unsqueeze(1),  # [bs, 1, H, W]
            t,                  # [bs]
            k_batch.unsqueeze(1)  # [bs, 1, H, W]
        ).squeeze(1)  # [bs, H, W]

        # Loss: weighted score matching
        # Weight: sigma^2(t) (importance weighting)
        loss = ((pred_score - target_score) ** 2 * sig.squeeze() ** 2).mean()

        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_batches += 1

    train_losses.append(epoch_loss / n_batches)
    sched.step()

    # Test
    score_net.eval()
    with torch.no_grad():
        t = torch.rand(N_TEST, device=device)
        eps = torch.randn_like(test_p_t)
        sig = sigma(t).view(-1, 1, 1)
        x_t = test_p_t + sig * eps
        target = -eps / sig
        pred = score_net(x_t.unsqueeze(1), t, test_k_t.unsqueeze(1)).squeeze(1)
        test_loss = ((pred - target) ** 2 * sig.squeeze() ** 2).mean().item()
    test_losses.append(test_loss)

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:4d} | Train: {train_losses[-1]:.6e} | Test: {test_loss:.6e} | Time: {time.time()-start:.1f}s")

train_time = time.time() - start
print("-" * 70)

# ============================================================
# [5] Sampling: Reverse SDE and Probability Flow ODE
# ============================================================
print(f"\n[5] Sampling from trained score model")
print("-" * 70)

N_SAMPLES = 5  # Generate 5 samples per test input
N_STEPS_SDE = 200   # Reverse SDE steps
N_STEPS_ODE = 100   # Probability flow ODE steps

@torch.no_grad()
def sample_reverse_sde(score_net, condition, n_steps, n_samples):
    """Sample using reverse-time SDE (Euler-Maruyama)."""
    B = condition.shape[0]
    H, W = condition.shape[-2], condition.shape[-1]
    # Start from pure noise: x(1) ~ N(0, sigma_max^2 * I)
    x = torch.randn(B, n_samples, H, W, device=device) * SIGMA_MAX
    x = x.view(B * n_samples, 1, H, W)
    cond = condition.unsqueeze(1).repeat(1, n_samples, 1, 1).view(B * n_samples, 1, H, W)

    dt = 1.0 / n_steps
    for i in range(n_steps, 0, -1):
        t = torch.full((B * n_samples,), i * dt, device=device)
        sig = sigma(t)  # [B*n_samples]
        # Score
        s = score_net(x, t, cond)  # [B*n_samples, 1, H, W]
        # Reverse SDE: dx = -sigma^2 * s * dt - sigma * sqrt(dt) * z
        z = torch.randn_like(x)
        x = x + sig.view(-1, 1, 1, 1) ** 2 * s * (-dt) + sig.view(-1, 1, 1, 1) * np.sqrt(dt) * z

    return x.view(B, n_samples, H, W)

@torch.no_grad()
def sample_probability_flow_ode(score_net, condition, n_steps, n_samples):
    """Sample using probability flow ODE (deterministic, Heun's method)."""
    B = condition.shape[0]
    H, W = condition.shape[-2], condition.shape[-1]
    x = torch.randn(B, n_samples, H, W, device=device) * SIGMA_MAX
    x = x.view(B * n_samples, 1, H, W)
    cond = condition.unsqueeze(1).repeat(1, n_samples, 1, 1).view(B * n_samples, 1, H, W)

    dt = 1.0 / n_steps
    for i in range(n_steps, 0, -1):
        t = torch.full((B * n_samples,), i * dt, device=device)
        sig = sigma(t)
        s = score_net(x, t, cond)
        # Probability flow ODE: dx = -0.5 * sigma^2 * s * dt
        x = x - 0.5 * sig.view(-1, 1, 1, 1) ** 2 * s * dt

    return x.view(B, n_samples, H, W)

print(f"  Generating {N_SAMPLES} samples per test input...")
print(f"  Method 1: Reverse SDE ({N_STEPS_SDE} steps)...")
samples_sde = sample_reverse_sde(score_net, test_p_t[:5], N_STEPS_SDE, N_SAMPLES)
print(f"  Method 2: Probability Flow ODE ({N_STEPS_ODE} steps)...")
samples_ode = sample_probability_flow_ode(score_net, test_p_t[:5], N_STEPS_ODE, N_SAMPLES)

# Denormalize
samples_sde = samples_sde * (p_std + 1e-8) + p_mean
samples_ode = samples_ode * (p_std + 1e-8) + p_mean
true_p = test_p[:5]

# Compute statistics
sde_mean = samples_sde.mean(dim=1)  # [5, H, W]
sde_std = samples_sde.std(dim=1)    # [5, H, W]
ode_mean = samples_ode.mean(dim=1)
ode_std = samples_ode.std(dim=1)

sde_mae = np.mean(np.abs(sde_mean.cpu().numpy() - true_p))
ode_mae = np.mean(np.abs(ode_mean.cpu().numpy() - true_p))
print(f"\n  SDE  mean MAE: {sde_mae:.6f}")
print(f"  ODE  mean MAE: {ode_mae:.6f}")
print(f"  SDE  avg uncertainty (std): {sde_std.mean().item():.6f}")
print(f"  ODE  avg uncertainty (std): {ode_std.mean().item():.6f}")

# ============================================================
# [6] Visualization
# ============================================================
print(f"\n[6] Generating visualizations...")

# --- Figure 1: Training loss ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(train_losses, linewidth=1.5, color='blue')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss')
ax1.set_title('Training Loss (Score Matching)'); ax1.grid(True, alpha=0.3)
ax2.semilogy(test_losses, linewidth=1.5, color='red')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Test Loss')
ax2.set_title('Test Loss'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_based_loss.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Generated samples (SDE) ---
fig, axes = plt.subplots(4, 6, figsize=(20, 13))
sample_idx = 0
for col in range(6):
    if col == 0:
        # True solution
        im = axes[0, col].imshow(true_p[sample_idx], cmap='jet', origin='lower')
        axes[0, col].set_title('True Solution', fontsize=11)
    else:
        # Generated samples
        im = axes[0, col].imshow(samples_sde[sample_idx, col-1].cpu().numpy(), cmap='jet', origin='lower')
        axes[0, col].set_title(f'SDE Sample {col}', fontsize=11)
    axes[0, col].axis('off')
    plt.colorbar(im, ax=axes[0, col], fraction=0.046)

# Row 2: ODE samples
for col in range(6):
    if col == 0:
        im = axes[1, col].imshow(true_p[sample_idx], cmap='jet', origin='lower')
        axes[1, col].set_title('True Solution', fontsize=11)
    else:
        im = axes[1, col].imshow(samples_ode[sample_idx, col-1].cpu().numpy(), cmap='jet', origin='lower')
        axes[1, col].set_title(f'ODE Sample {col}', fontsize=11)
    axes[1, col].axis('off')
    plt.colorbar(im, ax=axes[1, col], fraction=0.046)

# Row 3: Mean comparison
for col, (title, data) in enumerate([
    ("True", true_p[sample_idx]),
    ("SDE Mean", sde_mean[sample_idx].cpu().numpy()),
    ("ODE Mean", ode_mean[sample_idx].cpu().numpy()),
    ("SDE Std", sde_std[sample_idx].cpu().numpy()),
    ("ODE Std", ode_std[sample_idx].cpu().numpy()),
    ("|SDE-True|", np.abs(sde_mean[sample_idx].cpu().numpy() - true_p[sample_idx]))
]):
    cmap = 'jet' if 'Std' not in title else 'hot'
    im = axes[2, col].imshow(data, cmap=cmap, origin='lower')
    axes[2, col].set_title(title, fontsize=11)
    axes[2, col].axis('off')
    plt.colorbar(im, ax=axes[2, col], fraction=0.046)

# Row 4: Explanation
ax = axes[3, 0]
ax.text(0.5, 0.85, 'Score-Based Generative Model (SDE)', ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.60,
    'vs. DDPM (existing tutorial):\n'
    '  DDPM: discrete steps, predict noise\n'
    '  Score: continuous SDE, predict score\n'
    '  DDPM: fixed 200 steps\n'
    '  Score: any number of steps',
    fontsize=10, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.05, 0.25,
    'Key concepts:\n'
    '  Score function: s = ∇log p(x)\n'
    '  Forward SDE: add noise continuously\n'
    '  Reverse SDE: denoise using score\n'
    '  Prob. flow ODE: deterministic version',
    fontsize=10, transform=ax.transAxes, family='monospace', verticalalignment='top')
for col in range(6):
    axes[3, col].axis('off')

plt.suptitle('Score-Based Generative Model: Stochastic Darcy Flow', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_based_samples.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: SDE vs ODE comparison ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i in range(3):
    ax = axes[0, i]
    im = ax.imshow(true_p[i], cmap='jet', origin='lower')
    ax.set_title(f'True Solution {i+1}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, i]
    diff = np.abs(sde_mean[i].cpu().numpy() - true_p[i])
    im = ax.imshow(diff, cmap='hot', origin='lower')
    ax.set_title(f'|SDE Mean - True| {i+1}\nMAE={np.mean(diff):.4f}'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Score-Based SDE: Mean Prediction Error', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_based_error.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: Concept explanation ---
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.text(0.5, 0.95, 'Score-Based Generative Model vs DDPM', ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.80,
    'DDPM (existing tutorial):\n'
    '  - Discrete time: t = 1, 2, ..., T\n'
    '  - Noise schedule: beta_1, ..., beta_T\n'
    '  - Predict: noise epsilon\n'
    '  - Sample: reverse Markov chain\n'
    '  - Fixed number of steps (200)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(0.55, 0.80,
    'Score-Based (THIS tutorial):\n'
    '  - Continuous time: t in [0, 1]\n'
    '  - Noise: sigma(t) = sigma_min * (sigma_max/sigma_min)^t\n'
    '  - Predict: score s(x,t) = ∇log p_t(x)\n'
    '  - Sample: reverse SDE or prob. flow ODE\n'
    '  - Any number of steps (flexible)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.50,
    'Score function:\n'
    '  s(x, t) = ∇_x log p_t(x)\n'
    '  = direction of increasing probability\n'
    '  = "which way to move x to increase likelihood"\n'
    '  For Gaussian noise: s = -epsilon / sigma(t)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.55, 0.50,
    'Sampling methods:\n'
    '  1. Reverse SDE: stochastic, adds noise\n'
    '     dx = -sigma^2 * s * dt - sigma * dw\n'
    '  2. Probability flow ODE: deterministic\n'
    '     dx = -0.5 * sigma^2 * s * dt\n'
    '  SDE: more diverse samples\n'
    '  ODE: faster, deterministic',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.20,
    'Why score-based?\n'
    '  - More general framework (DDPM is special case)\n'
    '  - Continuous theory (SDE, ODE)\n'
    '  - Flexible sampling (any # steps)\n'
    '  - Score matching is well-studied',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax.text(0.55, 0.20,
    'Engineering relevance:\n'
    '  - Stochastic PDEs (uncertain parameters)\n'
    '  - Solution distribution (not just mean)\n'
    '  - Risk assessment (worst-case scenarios)\n'
    '  - Bayesian inverse problems',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightskyblue', alpha=0.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_based_explanation.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [7] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Score-Based Generative Model (SDE)")
print("=" * 70)
print(f"  Problem:           Stochastic Darcy Flow ({N_GRID}x{N_GRID})")
print(f"  Train samples:     {N_TRAIN}")
print(f"  Test samples:      {N_TEST}")
print(f"  Epochs:            {EPOCHS}")
print(f"  Training time:     {train_time:.1f}s")
print(f"  --- Results ---")
print(f"  SDE mean MAE:      {sde_mae:.6f}")
print(f"  ODE mean MAE:      {ode_mae:.6f}")
print(f"  SDE uncertainty:   {sde_std.mean().item():.6f}")
print(f"  ODE uncertainty:   {ode_std.mean().item():.6f}")
print()
print("Key observations:")
print("  1. SCORE FUNCTION: s = ∇log p(x) — gradient of log probability density")
print("  2. CONTINUOUS SDE: noise added continuously (not discrete like DDPM)")
print("  3. TWO SAMPLING METHODS: reverse SDE (stochastic) vs prob. flow ODE (deterministic)")
print("  4. vs DDPM: more general framework, flexible # steps, continuous theory")
print("  5. DIVERSITY: generates multiple plausible solutions (uncertainty quantification)")
print("  6. STOCHASTIC PDE: handles solution distribution (not just point estimate)")
print("=" * 70)
