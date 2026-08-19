"""
PhysicsNeMo FNO Tutorial: 2D Wave Equation (Electromagnetics/Acoustics)
=======================================================================
Wave equation is a fundamental PDE with 2nd-order time derivative.
This is the ONLY tutorial covering a 2nd-order time derivative PDE.

Wave equation:
    u_tt = c²·(u_xx + u_yy)

All other time-dependent tutorials use 1st-order time derivatives:
    - Navier-Stokes:  ω_t + ... (1st order)
    - Heat:            u_t = ... (1st order)
    - Burgers:         u_t + ... (1st order)
    - Allen-Cahn:      u_t = ... (1st order)

The wave equation's 2nd-order time derivative creates fundamentally different
dynamics: oscillation, wave propagation, reflection, and interference.

Key concepts:
    - 2nd-order time derivative (u_tt, not u_t)
    - Wave propagation: initial disturbance travels at speed c
    - Reflection: waves bounce off boundaries
    - Interference: multiple waves superpose
    - Multi-step rollout: predict u(t+1) from u(t) and u(t-1)

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

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo FNO Tutorial: 2D Wave Equation (Electromagnetics/Acoustics)")
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
# [1] Data Generation: 2D Wave Equation via FDM
# ============================================================
# u_tt = c²·(u_xx + u_yy)
# Discretized with central differences (2nd-order in both space and time)
#
# For 2nd-order time PDE, we need TWO previous time steps:
#   u(t+1) = 2·u(t) - u(t-1) + c²·dt²·(u_xx + u_yy)
#
# FNO input: [u(t), u(t-1)] → predict u(t+1)
# This is different from 1st-order PDEs where input is just [u(t)]

WAVE_SPEED = 1.0      # c = 1.0
GRID = 64              # 64×64 spatial grid
DOMAIN = 1.0           # [0, 1] × [0, 1]
DT = 0.005             # Time step
N_STEPS_DATA = 40      # 40 time steps per sample
N_TRAIN = 200
N_TEST = 30

def generate_wave_data(n_samples, grid, n_steps, dt, c):
    """
    Generate 2D wave equation data using FDM.
    
    u_tt = c²·(u_xx + u_yy)
    
    Boundary: u = 0 (Dirichlet, fixed boundaries → reflection)
    Initial: Gaussian pulse at random location
    
    Returns:
        u_history: [n_samples, n_steps+1, grid, grid]
    """
    dx = DOMAIN / (grid - 1)
    # CFL condition: c·dt/dx ≤ 1/√2 for 2D stability
    courant = c * dt / dx
    assert courant < 1.0 / np.sqrt(2), f"CFL violated: {courant:.3f} >= {1/np.sqrt(2):.3f}"
    
    u_history = np.zeros((n_samples, n_steps + 1, grid, grid), dtype=np.float32)
    
    for s in range(n_samples):
        # Initial condition: Gaussian pulse at random location
        u_prev = np.zeros((grid, grid), dtype=np.float64)
        u_curr = np.zeros((grid, grid), dtype=np.float64)
        
        # Random Gaussian source
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        sigma = np.random.uniform(0.03, 0.08)
        amplitude = np.random.uniform(0.5, 1.5)
        
        x = np.linspace(0, 1, grid)
        X, Y = np.meshgrid(x, x, indexing='ij')
        u_curr = amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        u_prev = u_curr.copy()  # Start from rest (u_t = 0)
        
        u_history[s, 0] = u_curr
        
        # FDM time stepping
        c2dt2 = (c * dt / dx)**2
        
        for step in range(1, n_steps + 1):
            u_next = np.zeros_like(u_curr)
            # Interior points: u_tt = c²·(u_xx + u_yy)
            u_next[1:-1, 1:-1] = (2 * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1]
                                   + c2dt2 * (u_curr[2:, 1:-1] - 2*u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]
                                             + u_curr[1:-1, 2:] - 2*u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]))
            # Boundary: u = 0 (Dirichlet → reflection)
            u_next[0, :] = 0; u_next[-1, :] = 0
            u_next[:, 0] = 0; u_next[:, -1] = 0
            
            u_prev = u_curr.copy()
            u_curr = u_next.copy()
            u_history[s, step] = u_curr
    
    return u_history

print("\n[1] Generating 2D Wave Equation data...")
print(f"  Equation: u_tt = c²·(u_xx + u_yy)  (c={WAVE_SPEED})")
print(f"  Grid: {GRID}×{GRID}, Domain: [0,1]×[0,1]")
print(f"  Time steps: {N_STEPS_DATA} (dt={DT})")
print(f"  Boundary: Dirichlet (u=0 → wave reflection)")
print(f"  Initial: Gaussian pulse at random location")
print(f"  Train: {N_TRAIN}, Test: {N_TEST}")

train_data = generate_wave_data(N_TRAIN, GRID, N_STEPS_DATA, DT, WAVE_SPEED)
test_data = generate_wave_data(N_TEST, GRID, N_STEPS_DATA, DT, WAVE_SPEED)
print(f"  Data shape: {train_data.shape} [samples, time, x, y]")

# Normalize
data_max = np.abs(train_data).max()
train_data_n = train_data / data_max
test_data_n = test_data / data_max

# Convert to tensors
train_t = torch.from_numpy(train_data_n).to(device)
test_t = torch.from_numpy(test_data_n).to(device)

# ============================================================
# [2] FNO Architecture (2D, 2-channel input for 2nd-order)
# ============================================================
class SpectralConv2d(nn.Module):
    """2D Fourier layer."""
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_ch * out_ch)
        self.weights = nn.Parameter(self.scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.weights.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum('bixy,ioxy->boxy', x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    """2D FNO for wave equation. Input: 2 channels [u(t), u(t-1)]."""
    def __init__(self, in_ch=2, out_ch=1, width=20, modes=12, n_layers=4):
        super().__init__()
        self.mlp_in = nn.Sequential(nn.Conv2d(in_ch, width, 1), nn.SiLU())
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.mlp_out = nn.Sequential(nn.SiLU(), nn.Conv2d(width, 64, 1), nn.SiLU(), nn.Conv2d(64, out_ch, 1))

    def forward(self, x):
        x = self.mlp_in(x)
        for conv, w in zip(self.convs, self.ws):
            x = F.silu(conv(x) + w(x))
        return self.mlp_out(x)


# ============================================================
# [3] Training
# ============================================================
# For 2nd-order PDE: input = [u(t), u(t-1)], target = u(t+1)
# This is the KEY difference from 1st-order PDEs (Allen-Cahn, NS, Heat)

EPOCHS = 250
BATCH_SIZE = 20
LR = 1e-3

print(f"\n[3] Training FNO for wave equation ({EPOCHS} epochs)")
print(f"    Input: [u(t), u(t-1)] → Output: u(t+1)  (2-channel input for 2nd-order PDE)")
print("-" * 70)

torch.manual_seed(42)
fno = FNO2d(in_ch=2, out_ch=1, width=20, modes=12, n_layers=4).to(device)
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
        # Input: [u(t), u(t-1)] for t=1..T-1
        # Target: u(t+1) for t=1..T-1
        u_curr = train_t[idx, 1:-1]   # u(t) for t=1..T-1
        u_prev = train_t[idx, :-2]    # u(t-1) for t=0..T-2
        u_next = train_t[idx, 2:]     # u(t+1) for t=2..T
        
        B, T = u_curr.shape[0], u_curr.shape[1]
        # Stack to 2-channel input: [B*T, 2, H, W]
        u_in = torch.stack([u_curr.reshape(B*T, GRID, GRID), u_prev.reshape(B*T, GRID, GRID)], dim=1)
        u_target = u_next.reshape(B*T, 1, GRID, GRID)
        
        pred = fno(u_in)
        loss = F.mse_loss(pred, u_target)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_batches += 1
    
    train_losses.append(epoch_loss / n_batches)
    sched.step()
    
    # Test
    fno.eval()
    with torch.no_grad():
        u_curr = test_t[:, 1:-1]
        u_prev = test_t[:, :-2]
        u_next = test_t[:, 2:]
        B, T = u_curr.shape[0], u_curr.shape[1]
        u_in = torch.stack([u_curr.reshape(B*T, GRID, GRID), u_prev.reshape(B*T, GRID, GRID)], dim=1)
        pred = fno(u_in).reshape(B, T, GRID, GRID)
        test_loss = F.mse_loss(pred, u_next).item()
    test_losses.append(test_loss)
    
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:4d} | Train: {train_losses[-1]:.6e} | Test: {test_loss:.6e} | Time: {time.time()-start:.1f}s")

train_time = time.time() - start
print("-" * 70)

# ============================================================
# [4] Multi-Step Rollout Evaluation
# ============================================================
print(f"\n[4] Multi-step rollout evaluation")
print("    Start from u(0) and u(1), predict all steps auto-regressively")
print("-" * 70)

fno.eval()
n_steps = test_t.shape[1]

with torch.no_grad():
    rollout_pred = torch.zeros_like(test_t)
    rollout_pred[:, 0] = test_t[:, 0]  # True u(0)
    rollout_pred[:, 1] = test_t[:, 1]  # True u(1)
    
    for step in range(1, n_steps - 1):
        u_curr = rollout_pred[:, step:step+1]     # [B, 1, H, W]
        u_prev = rollout_pred[:, step-1:step]     # [B, 1, H, W]
        u_in = torch.cat([u_curr, u_prev], dim=1)  # [B, 2, H, W]
        u_next = fno(u_in)                          # [B, 1, H, W]
        rollout_pred[:, step+1] = u_next.squeeze(1)
    
    # Relative L2 error at each step
    rollout_errors = []
    one_step_errors = []
    for step in range(n_steps):
        if step < n_steps - 1:
            # One-step (teacher-forced)
            u_curr = test_t[:, step:step+1]
            u_prev = test_t[:, step-1:step] if step > 0 else test_t[:, 0:1]
            u_in = torch.cat([u_curr, u_prev], dim=1)
            pred = fno(u_in)
            err1 = torch.norm(pred.squeeze(1) - test_t[:, step+1], dim=(1,2)) / \
                   (torch.norm(test_t[:, step+1], dim=(1,2)) + 1e-8)
            one_step_errors.append(err1.mean().item())
        
        # Rollout
        err_r = torch.norm(rollout_pred[:, step] - test_t[:, step], dim=(1,2)) / \
                (torch.norm(test_t[:, step], dim=(1,2)) + 1e-8)
        rollout_errors.append(err_r.mean().item())

print(f"  One-step (teacher-forced) avg error: {np.mean(one_step_errors):.4f}")
print(f"  Rollout step  5: {rollout_errors[5]:.4f}")
print(f"  Rollout step 10: {rollout_errors[10]:.4f}")
print(f"  Rollout step 20: {rollout_errors[20]:.4f}")
print(f"  Rollout step 30: {rollout_errors[30]:.4f}")
print(f"  Rollout step 40: {rollout_errors[40]:.4f}")

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
plt.savefig(os.path.join(RESULTS_DIR, "wave_loss.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Wave propagation snapshots (ground truth) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
sample_idx = 0
for i, step in enumerate([0, 5, 10, 15]):
    ax = axes[0, i]
    im = ax.imshow(test_data_n[sample_idx, step], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    ax.set_title(f't = {step}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if i == 0: ax.set_title('t = 0 (initial pulse)')
    plt.colorbar(im, ax=ax, fraction=0.046)
for i, step in enumerate([20, 25, 30, 35]):
    ax = axes[1, i]
    im = ax.imshow(test_data_n[sample_idx, step], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    ax.set_title(f't = {step}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if i == 3: ax.set_title('t = 35 (reflected waves)')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('2D Wave Equation: Propagation & Reflection (Ground Truth)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "wave_propagation.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: Ground truth vs FNO prediction ---
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
sample_idx = 0
for i, step in enumerate([5, 15, 25, 35]):
    # Ground truth
    ax = axes[0, i]
    im = ax.imshow(test_data_n[sample_idx, step], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    ax.set_title(f'Ground Truth (t={step})')
    plt.colorbar(im, ax=ax, fraction=0.046)
    # FNO rollout
    ax = axes[1, i]
    im = ax.imshow(rollout_pred[sample_idx, step].cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    ax.set_title(f'FNO Rollout (t={step})')
    plt.colorbar(im, ax=ax, fraction=0.046)
    # Error
    ax = axes[2, i]
    err = np.abs(rollout_pred[sample_idx, step].cpu().numpy() - test_data_n[sample_idx, step])
    im = ax.imshow(err, cmap='hot', origin='lower')
    ax.set_title(f'Error (t={step}, max={err.max():.4f})')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('2D Wave: Ground Truth vs FNO Rollout vs Error', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "wave_result.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: Rollout error ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(range(n_steps), rollout_errors, 'o-', linewidth=2, color='red', label='Rollout (auto-regressive)')
ax.axhline(y=np.mean(one_step_errors), color='blue', linestyle='--', linewidth=2, 
           label=f'One-step avg ({np.mean(one_step_errors):.4f})')
ax.set_xlabel('Rollout Step'); ax.set_ylabel('Relative L2 Error')
ax.set_title('Error Accumulation in Auto-Regressive Rollout (Wave Equation)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "wave_rollout_error.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 5: Equation explanation ---
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.text(0.5, 0.92, '2D Wave Equation: u_tt = c²·(u_xx + u_yy)', 
        ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.75,
    '2nd-order time derivative (u_tt):\n'
    '  • ALL other tutorials use 1st-order (u_t)\n'
    '  • Creates OSCILLATION (not decay/growth)\n'
    '  • Energy is conserved (no dissipation)\n'
    '  • Requires TWO previous steps: u(t), u(t-1)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.75,
    'Wave propagation:\n'
    '  • Initial pulse → circular wavefront\n'
    '  • Travels at speed c in all directions\n'
    '  • Reflects off boundaries (Dirichlet)\n'
    '  • Multiple reflections create interference',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.05, 0.40,
    'Physical applications:\n'
    '  • Electromagnetics (Maxwell equations)\n'
    '  • Acoustics (sound propagation)\n'
    '  • Seismology (earthquake waves)\n'
    '  • Optics (light propagation)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.40,
    'FNO challenge:\n'
    '  • 2-channel input: [u(t), u(t-1)]\n'
    '  • Oscillatory solution (high frequency)\n'
    '  • Long-time energy conservation\n'
    '  2D spatial domain (more complex than 1D)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.05, 0.10,
    'vs. Allen-Cahn (1st-order):\n'
    '  • Allen-Cahn: u_t = ... (needs 1 step)\n'
    '  • Wave: u_tt = ... (needs 2 steps)\n'
    '  • Allen-Cahn: dissipative (phase separation)\n'
    '  • Wave: conservative (oscillation)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "wave_explanation.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [6] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: FNO — 2D Wave Equation (Electromagnetics/Acoustics)")
print("=" * 70)
print(f"  Equation:          u_tt = c²·(u_xx + u_yy)  (c={WAVE_SPEED})")
print(f"  Grid:               {GRID}×{GRID}, Domain [0,1]×[0,1]")
print(f"  Time steps:         {n_steps}")
print(f"  Train samples:      {N_TRAIN}")
print(f"  Test samples:       {N_TEST}")
print(f"  Epochs:             {EPOCHS}")
print(f"  Training time:      {train_time:.1f}s")
print(f"  --- Results ---")
print(f"  One-step error:     {np.mean(one_step_errors):.4f} (avg)")
print(f"  Rollout step 10:    {rollout_errors[10]:.4f}")
print(f"  Rollout step 20:    {rollout_errors[20]:.4f}")
print(f"  Rollout step 40:    {rollout_errors[40]:.4f}")
print()
print("Key observations:")
print("  1. 2nd-ORDER TIME: Only tutorial with u_tt (all others use u_t)")
print("  2. 2-CHANNEL INPUT: [u(t), u(t-1)] needed for 2nd-order PDE")
print("  3. WAVE PROPAGATION: Gaussian pulse → circular wavefront (visible)")
print("  4. REFLECTION: Waves bounce off Dirichlet boundaries")
print("  5. ENERGY CONSERVATION: No dissipation (unlike Allen-Cahn/Heat)")
print("  6. ERROR ACCUMULATION: Rollout error grows (compounding)")
print("=" * 70)
