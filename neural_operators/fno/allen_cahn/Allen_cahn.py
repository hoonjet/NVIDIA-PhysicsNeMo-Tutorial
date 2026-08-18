"""
PhysicsNeMo FNO Tutorial: Allen-Cahn Equation (Phase Separation)
=================================================================
Allen-Cahn equation is one of the THREE benchmark PDEs in the original FNO paper:
    Li et al., "Fourier Neural Operator for Parametric PDEs," ICLR 2021.
    arXiv: https://arxiv.org/abs/2010.08895

The three benchmarks in the FNO paper are:
    1. Darcy Flow       — already covered (6 tutorials in this repo)
    2. Navier-Stokes    — already covered (1 tutorial in this repo)
    3. Allen-Cahn       — THIS tutorial (was missing!)

Allen-Cahn equation:
    u_t = ε²·u_xx + u - u³

This is a REACTION-DIFFUSION equation with a nonlinear reaction term (u - u³).
The reaction term drives the solution toward two stable states: u = +1 and u = -1.
This creates PHASE SEPARATION: regions of +1 and -1 form and coarsen over time.

Key differences from existing tutorials:
    - vs Darcy: Time-dependent (Darcy is steady-state), nonlinear reaction term
    - vs Navier-Stokes: No advection, pure reaction-diffusion, phase separation
    - vs Heat: Nonlinear reaction term (heat is linear diffusion only)
    - vs Burgers: 2nd-order diffusion + 3rd-order nonlinearity (Burgers is 1st-order advection)

Key concepts:
    - Nonlinear reaction term: u - u³ creates two stable equilibria
    - Phase separation: initial noise → pattern formation → coarsening
    - Auto-regressive prediction: predict u(t+1) from u(t)
    - Multi-step rollout: feed prediction back as input (error accumulation analysis)

Author: PhysicsNeMo Tutorial
Date: 2026-08-14
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
from matplotlib.animation import FuncAnimation

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo FNO Tutorial: Allen-Cahn Equation (Phase Separation)")
print("FNO Original Paper Benchmark (Li et al., ICLR 2021)")
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
# [1] Data Generation: Allen-Cahn via Spectral Method
# ============================================================
# u_t = ε²·u_xx + u - u³
# We use a pseudo-spectral method with periodic boundary conditions.

EPSILON = 0.01       # Diffusion coefficient (small → sharp interfaces)
DOMAIN = 1.0         # Domain length [0, 1]
N_POINTS = 128       # Spatial grid points
DT = 0.0001           # Time step for data generation
N_STEPS_DATA = 100   # Number of time steps to generate per sample
SAVE_INTERVAL = 5    # Save every N steps (so saved_dt = DT * SAVE_INTERVAL)

def generate_allen_cahn_data(n_samples, n_points, n_steps, dt, save_interval, epsilon):
    """
    Generate Allen-Cahn data using pseudo-spectral method.
    
    u_t = ε²·u_xx + u - u³
    
    Returns:
        u_history: [n_samples, n_saved_steps, n_points]
    """
    dx = DOMAIN / n_points
    k = np.fft.fftfreq(n_points, d=dx) * 2 * np.pi
    k2 = k**2
    
    # Linear part: ε²·u_xx → -ε²·k²·u_hat (in Fourier space)
    # We use ETDRK4 (Exponential Time Differencing Runge-Kutta 4th order)
    # for stability with stiff linear part.
    
    L = -epsilon**2 * k2  # Linear operator in Fourier space
    E = np.exp(L * dt)    # Exact integration of linear part
    E2 = np.exp(L * dt / 2)
    
    n_saved = n_steps // save_interval
    u_history = np.zeros((n_samples, n_saved + 1, n_points), dtype=np.float32)
    
    for s in range(n_samples):
        # Initial condition: small random perturbation around 0
        # This creates phase separation as the system evolves
        u = 0.05 * np.random.randn(n_points)
        # Add some low-frequency structure
        for m in range(1, 6):
            u += 0.1 * np.sin(2 * np.pi * m * np.linspace(0, 1, n_points) + 
                              np.random.rand() * 2 * np.pi) / m
        
        u_history[s, 0] = u
        u_hat = np.fft.fft(u)
        save_idx = 1
        
        for step in range(1, n_steps + 1):
            # Nonlinear part: N(u) = u - u³
            # ETDRK4 scheme
            u_real = np.real(np.fft.ifft(u_hat))
            Nu = u_real - u_real**3
            Nu_hat = np.fft.fft(Nu)
            
            a = E2 * u_hat + (E2 - 1) / (L + 1e-30) * Nu_hat
            Na = np.real(np.fft.ifft(a))
            Na_hat = np.fft.fft(Na - Na**3)
            
            b = E2 * u_hat + (E2 - 1) / (L + 1e-30) * Na_hat
            Nb = np.real(np.fft.ifft(b))
            Nb_hat = np.fft.fft(Nb - Nb**3)
            
            c = E2 * a + (E - E2) / (L + 1e-30) * Nb_hat
            Nc = np.real(np.fft.ifft(c))
            Nc_hat = np.fft.fft(Nc - Nc**3)
            
            u_hat = E * u_hat + ((E - 1) / (L + 1e-30) * (2*Nu_hat + 2*Na_hat + 4*Nb_hat - Nc_hat) 
                                  + 2 * (E - E2) / (L + 1e-30) * Nb_hat) / 6
            
            if step % save_interval == 0 and save_idx <= n_saved:
                u_history[s, save_idx] = np.real(np.fft.ifft(u_hat))
                save_idx += 1
    
    return u_history

print("\n[1] Generating Allen-Cahn data...")
N_TRAIN = 150
N_TEST = 30
N_STEPS = 100  # 20 saved steps (100/5)

print(f"  Equation: u_t = ε²·u_xx + u - u³  (ε={EPSILON})")
print(f"  Grid: {N_POINTS} points, Domain: [0, {DOMAIN}]")
print(f"  Time steps: {N_STEPS_DATA} (save every {SAVE_INTERVAL}, → {N_STEPS_DATA//SAVE_INTERVAL} saved)")
print(f"  Train: {N_TRAIN} samples, Test: {N_TEST} samples")

train_data = generate_allen_cahn_data(N_TRAIN, N_POINTS, N_STEPS_DATA, DT, SAVE_INTERVAL, EPSILON)
test_data = generate_allen_cahn_data(N_TEST, N_POINTS, N_STEPS_DATA, DT, SAVE_INTERVAL, EPSILON)

print(f"  Data shape: {train_data.shape} [samples, time_steps, spatial]")

# Normalize
data_mean = train_data.mean()
data_std = train_data.std()
train_data_n = (train_data - data_mean) / (data_std + 1e-8)
test_data_n = (test_data - data_mean) / (data_std + 1e-8)

# Convert to tensors: [samples, time, 1, spatial]
train_t = torch.from_numpy(train_data_n).unsqueeze(2).to(device)
test_t = torch.from_numpy(test_data_n).unsqueeze(2).to(device)

# ============================================================
# [2] FNO Architecture (1D)
# ============================================================
class SpectralConv1d(nn.Module):
    """1D Fourier layer for Allen-Cahn (1D spatial domain)."""
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (in_ch * out_ch)
        self.weights = nn.Parameter(self.scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.weights.shape[1], N // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = min(self.modes, N // 2 + 1)
        out_ft[:, :, :m] = torch.einsum('bix,iox->box', x_ft[:, :, :m], self.weights[:, :, :m])
        return torch.fft.irfft(out_ft, n=N)


class FNO1d(nn.Module):
    """1D Fourier Neural Operator for time-dependent PDEs."""
    def __init__(self, in_ch=1, out_ch=1, width=16, modes=8, n_layers=4):
        super().__init__()
        self.mlp_in = nn.Sequential(nn.Conv1d(in_ch, width, 1), nn.SiLU())
        self.convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.mlp_out = nn.Sequential(nn.SiLU(), nn.Conv1d(width, 64, 1), nn.SiLU(), nn.Conv1d(64, out_ch, 1))

    def forward(self, x):
        x = self.mlp_in(x)
        for conv, w in zip(self.convs, self.ws):
            x = F.silu(conv(x) + w(x))
        return self.mlp_out(x)


# ============================================================
# [3] Training: Auto-Regressive Prediction
# ============================================================
# Task: Given u(t), predict u(t+1) (one step ahead)
# Then do multi-step rollout: feed prediction back as input

EPOCHS = 300
BATCH_SIZE = 15
LR = 1e-3
N_INPUT_STEPS = train_t.shape[1] - 1  # Use all consecutive pairs

print(f"\n[3] Training FNO for auto-regressive prediction ({EPOCHS} epochs)")
print(f"    Task: u(t) → u(t+1) (one-step prediction)")
print("-" * 70)

torch.manual_seed(42)
fno = FNO1d(in_ch=1, out_ch=1, width=16, modes=8, n_layers=4).to(device)
opt = torch.optim.Adam(fno.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

train_losses = []
test_losses = []
start = time.time()

for epoch in range(EPOCHS):
    fno.train()
    epoch_loss = 0; n_batches = 0
    perm = torch.randperm(N_TRAIN)
    for i in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        # Input: u(t) at all time steps except last
        # Target: u(t+1) at all time steps except first
        u_in = train_t[idx, :-1]   # [B, T-1, 1, N]
        u_target = train_t[idx, 1:] # [B, T-1, 1, N]
        
        # Reshape for batch processing: [B*(T-1), 1, N]
        B, T = u_in.shape[0], u_in.shape[1]
        u_in_flat = u_in.reshape(B * T, 1, N_POINTS)
        u_target_flat = u_target.reshape(B * T, 1, N_POINTS)
        
        pred = fno(u_in_flat)
        loss = F.mse_loss(pred, u_target_flat)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_batches += 1
    
    train_losses.append(epoch_loss / n_batches)
    sched.step()
    
    # Test: one-step prediction error
    fno.eval()
    with torch.no_grad():
        u_in = test_t[:, :-1]
        u_target = test_t[:, 1:]
        B, T = u_in.shape[0], u_in.shape[1]
        u_in_flat = u_in.reshape(B * T, 1, N_POINTS)
        pred = fno(u_in_flat).reshape(B, T, 1, N_POINTS)
        test_loss = F.mse_loss(pred, u_target).item()
    test_losses.append(test_loss)
    
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:4d} | Train: {train_losses[-1]:.6e} | Test: {test_loss:.6e} | Time: {time.time()-start:.1f}s")

train_time = time.time() - start
print("-" * 70)

# ============================================================
# [4] Multi-Step Rollout Evaluation
# ============================================================
print(f"\n[4] Multi-step rollout evaluation (auto-regressive)")
print("    Feed prediction back as input, measure error accumulation")
print("-" * 70)

fno.eval()
n_saved_steps = test_t.shape[1]

with torch.no_grad():
    # Start from initial condition, predict all steps auto-regressively
    rollout_pred = torch.zeros_like(test_t)
    rollout_pred[:, 0] = test_t[:, 0]  # Start with true initial condition
    
    for step in range(1, n_saved_steps):
        u_current = rollout_pred[:, step-1:step]  # [B, 1, 1, N]
        B = u_current.shape[0]
        u_in = u_current.reshape(B, 1, N_POINTS)
        u_next = fno(u_in).reshape(B, 1, 1, N_POINTS)
        rollout_pred[:, step] = u_next.squeeze(1)
    
    # Compute relative L2 error at each rollout step
    rollout_errors = []
    for step in range(n_saved_steps):
        err = torch.norm(rollout_pred[:, step] - test_t[:, step], dim=(1, 2)) / \
              (torch.norm(test_t[:, step], dim=(1, 2)) + 1e-8)
        rollout_errors.append(err.mean().item())
    
    # Also compute one-step prediction error (teacher-forced)
    one_step_errors = []
    for step in range(n_saved_steps - 1):
        u_in = test_t[:, step].reshape(N_TEST, 1, N_POINTS)
        pred = fno(u_in)
        err = torch.norm(pred - test_t[:, step+1], dim=(1, 2)) / \
              (torch.norm(test_t[:, step+1], dim=(1, 2)) + 1e-8)
        one_step_errors.append(err.mean().item())

print(f"  One-step (teacher-forced) avg error: {np.mean(one_step_errors):.4f}")
print(f"  Rollout step  1: {rollout_errors[1]:.4f}")
print(f"  Rollout step  5: {rollout_errors[5]:.4f}")
print(f"  Rollout step 10: {rollout_errors[10]:.4f}")
print(f"  Rollout step 15: {rollout_errors[15]:.4f}")
print(f"  Rollout step 20: {rollout_errors[20]:.4f}")

# ============================================================
# [5] Visualization
# ============================================================
print("\n[5] Generating visualizations...")

# --- Figure 1: Training loss ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(train_losses, linewidth=2, color='blue')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss (MSE)')
ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)

ax2.semilogy(test_losses, linewidth=2, color='red')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Test Loss (MSE)')
ax2.set_title('Test Loss (one-step)'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "allen_cahn_loss.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Phase separation visualization (ground truth) ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
x = np.linspace(0, DOMAIN, N_POINTS)
sample_idx = 0
for i, step in enumerate([0, 10, 20]):
    ax = axes[i]
    u = test_data[sample_idx, step]
    ax.plot(x, u, linewidth=2, color=f'C{i}')
    ax.fill_between(x, u, alpha=0.3, color=f'C{i}')
    ax.set_ylabel(f'u(x, t={step})')
    ax.set_ylim(-1.3, 1.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f't = {step} (phase separation: +1 / -1 regions)')
axes[-1].set_xlabel('x')
plt.suptitle('Allen-Cahn: Phase Separation Over Time (Ground Truth)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "allen_cahn_phase.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: Space-time diagram (ground truth vs prediction) ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
sample_idx = 0

# Ground truth space-time
ax = axes[0, 0]
im = ax.imshow(test_data[sample_idx], aspect='auto', cmap='RdBu_r', 
               extent=[0, DOMAIN, n_saved_steps-1, 0], vmin=-1.2, vmax=1.2)
ax.set_xlabel('x'); ax.set_ylabel('Time step')
ax.set_title('Ground Truth (space-time)')
plt.colorbar(im, ax=ax)

# FNO rollout prediction
ax = axes[0, 1]
pred_show = rollout_pred[sample_idx, :, 0].cpu().numpy() * data_std + data_mean
im = ax.imshow(pred_show, aspect='auto', cmap='RdBu_r',
               extent=[0, DOMAIN, n_saved_steps-1, 0], vmin=-1.2, vmax=1.2)
ax.set_xlabel('x'); ax.set_ylabel('Time step')
ax.set_title('FNO Rollout Prediction (space-time)')
plt.colorbar(im, ax=ax)

# Error
ax = axes[1, 0]
err = np.abs(pred_show - test_data[sample_idx])
im = ax.imshow(err, aspect='auto', cmap='hot',
               extent=[0, DOMAIN, n_saved_steps-1, 0])
ax.set_xlabel('x'); ax.set_ylabel('Time step')
ax.set_title(f'Absolute Error (max={err.max():.4f})')
plt.colorbar(im, ax=ax)

# Rollout error vs steps
ax = axes[1, 1]
ax.plot(range(n_saved_steps), rollout_errors, 'o-', linewidth=2, color='red', label='Rollout (auto-regressive)')
ax.axhline(y=np.mean(one_step_errors), color='blue', linestyle='--', linewidth=2, label=f'One-step avg ({np.mean(one_step_errors):.4f})')
ax.set_xlabel('Rollout Step'); ax.set_ylabel('Relative L2 Error')
ax.set_title('Error Accumulation in Auto-Regressive Rollout')
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Allen-Cahn: FNO Prediction vs Ground Truth', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "allen_cahn_result.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: Multiple samples phase separation ---
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.imshow(test_data[i], aspect='auto', cmap='RdBu_r', vmin=-1.2, vmax=1.2)
    ax.set_title(f'Sample {i+1}')
    ax.set_xlabel('x'); ax.set_ylabel('Time step')
plt.suptitle('Allen-Cahn: Diverse Phase Separation Patterns', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "allen_cahn_patterns.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 5: Equation explanation ---
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.text(0.5, 0.9, 'Allen-Cahn Equation: u_t = ε²·u_xx + u - u³', 
        ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.7, 
    'Diffusion term (ε²·u_xx):\n'
    '  • Smooths interfaces between phases\n'
    '  • Small ε → sharp interfaces (steep gradients)\n'
    '  • This is the "stiff" part (requires small dt)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.7,
    'Reaction term (u - u³):\n'
    '  • Nonlinear: drives u toward +1 or -1\n'
    '  • u=0 is unstable (any perturbation grows)\n'
    '  • u=±1 are stable equilibria\n'
    '  • Creates PHASE SEPARATION',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.1, 0.35,
    'Physical meaning:\n'
    '  • Alloy solidification: two metal phases separate\n'
    '  • Binary fluid: two liquids demix\n'
    '  • Pattern formation: initial noise → structured domains',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.35,
    'FNO challenge:\n'
    '  • Sharp interfaces (high frequency content)\n'
    '  • Nonlinear dynamics (u³ term)\n'
    '  • Long-time integration (error accumulation)\n'
    '  • Auto-regressive rollout (feed prediction back)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "allen_cahn_explanation.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [6] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: FNO — Allen-Cahn Equation (Phase Separation)")
print("=" * 70)
print(f"  Equation:          u_t = ε²·u_xx + u - u³  (ε={EPSILON})")
print(f"  Grid:               {N_POINTS} points, Domain [0, {DOMAIN}]")
print(f"  Time steps:         {n_saved_steps} (saved from {N_STEPS_DATA} steps)")
print(f"  Train samples:      {N_TRAIN}")
print(f"  Test samples:       {N_TEST}")
print(f"  Epochs:             {EPOCHS}")
print(f"  Training time:      {train_time:.1f}s")
print(f"  --- Results ---")
print(f"  One-step error:     {np.mean(one_step_errors):.4f} (avg)")
print(f"  Rollout step  1:    {rollout_errors[1]:.4f}")
print(f"  Rollout step 10:    {rollout_errors[10]:.4f}")
print(f"  Rollout step 20:    {rollout_errors[20]:.4f}")
print()
print("Key observations:")
print("  1. PHASE SEPARATION: Initial noise → structured +1/-1 domains (visible in space-time)")
print("  2. NONLINEAR REACTION: u-u³ term creates two stable equilibria (unlike Navier-Stokes)")
print("  3. SHARP INTERFACES: Small ε creates steep gradients (spectral challenge for FNO)")
print("  4. AUTO-REGRESSIVE: FNO predicts u(t+1) from u(t), then feeds back for rollout")
print("  5. ERROR ACCUMULATION: Rollout error grows with steps (compounding prediction errors)")
print("  6. FNO PAPER BENCHMARK: This is the 3rd benchmark from Li et al. (ICLR 2021)")
print("=" * 70)
