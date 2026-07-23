"""
PhysicsNeMo Comparison: PINN vs FNO on Burgers Equation
=========================================================
This script compares two fundamentally different approaches to solving
the same PDE (1D Viscous Burgers Equation):

1. PINN: Equation-based learning (no data needed, uses PDE residual)
2. FNO:  Data-driven learning (needs labeled data, supervised)

Both models solve:
    u_t + u * u_x = (nu / pi) * u_xx

The comparison reveals the trade-offs between:
- Data requirement (PINN: none, FNO: needs training data)
- Training speed (PINN: slow per epoch, FNO: fast per epoch)
- Accuracy (PINN: depends on PDE loss, FNO: depends on data quality)
- Generalization (PINN: continuous, FNO: fixed grid)

Author: PhysicsNeMo Tutorial
Date: 2026-07-20
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
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo Comparison: PINN vs FNO on Burgers Equation")
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
NU = 0.01 / np.pi
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
N_GRID = 64  # Grid resolution for comparison

# ============================================================
# [2] PINN Model (Equation-based)
# ============================================================
class PINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 1]):
        super().__init__()
        self.layers = layers
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        self.linears = nn.ModuleList(layer_list)
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        for i in range(len(self.layers) - 2):
            x = self.activation(self.linears[i](x))
        x = self.linears[-1](x)
        return x


def pde_residual(model, x, t):
    xt = torch.cat([x, t], dim=1).requires_grad_(True)
    u = model(xt)
    u_x = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 0:1]
    u_t = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    return u_t + u * u_x - NU * u_xx


# ============================================================
# [3] FNO Model (Data-driven)
# ============================================================
class SpectralConv1d(nn.Module):
    """1D Spectral Convolution (Fourier layer)"""
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))
    
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                              dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", 
            x_ft[:, :, :self.modes], self.weights)

        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    """1D Fourier Neural Operator"""
    def __init__(self, modes=8, width=32, n_layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(2, width)  # Input: (u, t)
        self.conv_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x, t_val):
        # x: [batch, 1, N] (vorticity at time t)
        # t_val: scalar time value
        batch, _, N = x.shape
        # Expand time to match spatial grid
        t_expanded = torch.full((batch, 1, N), t_val, device=x.device)
        # Concatenate: [batch, 2, N]
        x = torch.cat([x, t_expanded], dim=1)
        # Lift: [batch, 2, N] -> [batch, width, N]
        x = x.permute(0, 2, 1)  # [batch, N, 2]
        x = self.fc0(x)  # [batch, N, width]
        x = x.permute(0, 2, 1)  # [batch, width, N]
        # Fourier layers
        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)
        # Project: [batch, width, N] -> [batch, 1, N]
        x = x.permute(0, 2, 1)  # [batch, N, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)  # [batch, N, 1]
        x = x.permute(0, 2, 1)  # [batch, 1, N]
        return x


# ============================================================
# [4] Generate Reference Data (for FNO training)
# ============================================================
print("\n[1] Generating reference data for FNO training...")

def smooth_gaussian_1d(u, sigma=2.0):
    """Simple Gaussian smoothing using numpy convolution."""
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    # Pad with periodic boundary
    u_padded = np.concatenate([u[-radius:], u, u[:radius]])
    u_smooth = np.convolve(u_padded, kernel, mode='same')
    return u_smooth[radius:radius+len(u)]


def generate_burgers_data(n_samples=50, n_grid=64, n_times=20):
    """Generate Burgers equation data using finite difference with sub-stepping for stability."""
    dx = (X_MAX - X_MIN) / (n_grid - 1)
    dt_target = (T_MAX - T_MIN) / n_times
    x = np.linspace(X_MIN, X_MAX, n_grid)
    
    # CFL stability: dt < dx / max(|u|). Use sub-stepping.
    u_max = 0.5
    dt_stable = 0.4 * dx / u_max  # CFL condition
    n_substeps = max(1, int(np.ceil(dt_target / dt_stable)))
    dt = dt_target / n_substeps
    
    all_data = []
    for s in range(n_samples):
        # Random initial condition (smoothed with numpy)
        u = np.random.randn(n_grid) * 0.3
        u = smooth_gaussian_1d(u, sigma=2.0)
        u = u / (np.max(np.abs(u)) + 1e-8) * 0.5
        
        u_traj = [u.copy()]
        for step in range(n_times):
            for _ in range(n_substeps):
                # Finite difference: u_t + u*u_x = nu * u_xx
                u_x = np.zeros_like(u)
                u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
                u_x[0] = (u[1] - u[-1]) / (2 * dx)
                u_x[-1] = (u[0] - u[-2]) / (2 * dx)
                
                u_xx = np.zeros_like(u)
                u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
                u_xx[0] = (u[1] - 2*u[0] + u[-1]) / dx**2
                u_xx[-1] = (u[0] - 2*u[-1] + u[-2]) / dx**2
                
                u_new = u + dt * (-u * u_x + NU * u_xx)
                # Clip to prevent blowup
                u = np.clip(u_new, -2.0, 2.0)
            u_traj.append(u.copy())
        
        all_data.append(np.stack(u_traj, axis=0))
    
    return np.array(all_data, dtype=np.float32), x


w_data, x_grid = generate_burgers_data(n_samples=50, n_grid=N_GRID, n_times=20)
print(f"  Data shape: {w_data.shape} [samples, times, x]")
print(f"  Value range: [{w_data.min():.3f}, {w_data.max():.3f}]")


# ============================================================
# [5] Train PINN
# ============================================================
print("\n[2] Training PINN (equation-based, no data needed)...")

# Training data for PINN
N_IC = 200
N_BC = 100
N_F = 10000

x_ic = torch.linspace(X_MIN, X_MAX, N_IC).reshape(-1, 1)
t_ic = torch.zeros(N_IC, 1)
u_ic = -torch.sin(np.pi * x_ic)

t_bc = torch.linspace(T_MIN, T_MAX, N_BC // 2).reshape(-1, 1)
x_bc_left = torch.full_like(t_bc, X_MIN)
x_bc_right = torch.full_like(t_bc, X_MAX)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
t_bc_all = torch.cat([t_bc, t_bc], dim=0)
u_bc = torch.zeros_like(x_bc)

x_f = torch.rand(N_F, 1) * (X_MAX - X_MIN) + X_MIN
t_f = torch.rand(N_F, 1) * (T_MAX - T_MIN) + T_MIN

x_ic, t_ic, u_ic = x_ic.to(device), t_ic.to(device), u_ic.to(device)
x_bc, t_bc_all, u_bc = x_bc.to(device), t_bc_all.to(device), u_bc.to(device)
x_f, t_f = x_f.to(device), t_f.to(device)

pinn_model = PINN().to(device)
pinn_params = sum(p.numel() for p in pinn_model.parameters())
pinn_optimizer = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)
pinn_scheduler = torch.optim.lr_scheduler.StepLR(pinn_optimizer, step_size=2000, gamma=0.9)

PINN_EPOCHS = 5000
pinn_losses = []
pinn_start = time.time()

for epoch in range(PINN_EPOCHS):
    pinn_optimizer.zero_grad()
    
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    pred_ic = pinn_model(xt_ic)
    loss_ic = torch.mean((pred_ic - u_ic) ** 2)
    
    xt_bc = torch.cat([x_bc, t_bc_all], dim=1)
    pred_bc = pinn_model(xt_bc)
    loss_bc = torch.mean((pred_bc - u_bc) ** 2)
    
    residual = pde_residual(pinn_model, x_f, t_f)
    loss_pde = torch.mean(residual ** 2)
    
    loss = loss_ic + loss_bc + loss_pde
    loss.backward()
    pinn_optimizer.step()
    pinn_scheduler.step()
    pinn_losses.append(loss.item())
    
    if epoch % 1000 == 0 or epoch == PINN_EPOCHS - 1:
        print(f"  PINN Epoch {epoch:5d}/{PINN_EPOCHS} | Loss: {loss.item():.6e}")

pinn_time = time.time() - pinn_start
print(f"  PINN training time: {pinn_time:.1f}s")
print(f"  PINN final loss: {pinn_losses[-1]:.6e}")

# ============================================================
# [6] Train FNO
# ============================================================
print("\n[3] Training FNO (data-driven, needs labeled data)...")

# Prepare FNO training data: u(t) -> u(t+dt)
n_samples, n_times, n_grid = w_data.shape
inputs_fno = w_data[:, :-1, :].reshape(-1, 1, n_grid)  # [samples*(times-1), 1, N]
outputs_fno = w_data[:, 1:, :].reshape(-1, 1, n_grid)
times_fno = np.tile(np.linspace(0, 1, n_times)[:-1], n_samples)  # time for each pair

n_total_fno = inputs_fno.shape[0]
n_train_fno = int(n_total_fno * 0.8)

indices_fno = np.random.permutation(n_total_fno)
train_idx_fno = indices_fno[:n_train_fno]
test_idx_fno = indices_fno[n_train_fno:]

x_train_fno = torch.from_numpy(inputs_fno[train_idx_fno]).to(device)
y_train_fno = torch.from_numpy(outputs_fno[train_idx_fno]).to(device)
t_train_fno = torch.from_numpy(times_fno[train_idx_fno]).to(device)

fno_model = FNO1d(modes=8, width=32, n_layers=4).to(device)
fno_params = sum(p.numel() for p in fno_model.parameters())
fno_optimizer = torch.optim.Adam(fno_model.parameters(), lr=1e-3, weight_decay=1e-5)
fno_scheduler = torch.optim.lr_scheduler.StepLR(fno_optimizer, step_size=100, gamma=0.5)

FNO_EPOCHS = 300
BATCH_SIZE = 16
fno_losses = []
fno_start = time.time()

for epoch in range(FNO_EPOCHS):
    fno_model.train()
    epoch_loss = 0
    n_batches = 0
    perm = torch.randperm(n_train_fno)
    
    for i in range(0, n_train_fno, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x_batch = x_train_fno[idx]
        y_batch = y_train_fno[idx]
        t_batch = t_train_fno[idx]
        
        # Process each sample (different time values)
        pred = torch.stack([fno_model(x_batch[j:j+1], t_batch[j]) for j in range(len(idx))])
        loss = F.mse_loss(pred, y_batch)
        
        fno_optimizer.zero_grad()
        loss.backward()
        fno_optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    fno_losses.append(epoch_loss / n_batches)
    fno_scheduler.step()
    
    if epoch % 50 == 0 or epoch == FNO_EPOCHS - 1:
        print(f"  FNO Epoch {epoch:4d}/{FNO_EPOCHS} | Loss: {fno_losses[-1]:.6e}")

fno_time = time.time() - fno_start
print(f"  FNO training time: {fno_time:.1f}s")
print(f"  FNO final loss: {fno_losses[-1]:.6e}")

# ============================================================
# [7] Compare on Same Grid
# ============================================================
print("\n[4] Comparing PINN and FNO on the same grid...")

# Create evaluation grid
x_eval = np.linspace(X_MIN, X_MAX, N_GRID)
t_eval = np.linspace(T_MIN, T_MAX, 20)

# PINN predictions (continuous, evaluate at any point)
pinn_pred = np.zeros((len(t_eval), N_GRID))
with torch.no_grad():
    for ti, t_val in enumerate(t_eval):
        xt = torch.tensor(np.stack([x_eval, np.full(N_GRID, t_val)], axis=1), dtype=torch.float32).to(device)
        pinn_pred[ti] = pinn_model(xt).cpu().numpy().flatten()

# FNO predictions (autoregressive from t=0)
# Use -sin(pi*x) as initial condition (same as PINN's IC)
u0 = -np.sin(np.pi * x_eval)
fno_pred = np.zeros((len(t_eval), N_GRID))
fno_pred[0] = u0

with torch.no_grad():
    current = torch.from_numpy(u0).float().reshape(1, 1, N_GRID).to(device)
    for ti in range(1, len(t_eval)):
        t_val = t_eval[ti]
        current = fno_model(current, torch.tensor(t_val, device=device))
        fno_pred[ti] = current.cpu().numpy().flatten()

# ============================================================
# [8] Visualization
# ============================================================
print("\n[5] Generating comparison visualizations...")

# --- Figure 1: Loss curves comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(pinn_losses, label='PINN', color='blue', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title(f'PINN Loss ({PINN_EPOCHS} epochs, {pinn_time:.0f}s)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(fno_losses, label='FNO', color='red', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title(f'FNO Loss ({FNO_EPOCHS} epochs, {fno_time:.0f}s)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Training Loss Comparison: PINN vs FNO', fontsize=14)
plt.tight_layout()
loss_path = os.path.join(RESULTS_DIR, "pinn_vs_fno_loss.png")
plt.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {loss_path}")

# --- Figure 2: Solution comparison at different times ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
snapshots = [0, 5, 10, 15, 18, 19]

for idx, snap in enumerate(snapshots):
    ax = axes[idx // 3, idx % 3]
    ax.plot(x_eval, pinn_pred[snap], 'b-', linewidth=2, label='PINN', alpha=0.8)
    ax.plot(x_eval, fno_pred[snap], 'r--', linewidth=2, label='FNO', alpha=0.8)
    if snap == 0:
        ax.plot(x_eval, -np.sin(np.pi * x_eval), 'k:', linewidth=1, alpha=0.5, label='IC')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(f't = {t_eval[snap]:.2f}')
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Burgers Equation: PINN vs FNO Solution Comparison', fontsize=14)
plt.tight_layout()
result_path = os.path.join(RESULTS_DIR, "pinn_vs_fno_result.png")
plt.savefig(result_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {result_path}")

# --- Figure 3: Summary bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Parameters comparison
ax = axes[0]
models = ['PINN', 'FNO']
params = [pinn_params, fno_params]
bars = ax.bar(models, params, color=['blue', 'red'], alpha=0.7)
ax.set_ylabel('Parameters')
ax.set_title('Model Parameters')
for bar, val in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(params), 
            f'{val:,}', ha='center', va='bottom', fontsize=11)

# Training time comparison
ax = axes[1]
times = [pinn_time, fno_time]
bars = ax.bar(models, times, color=['blue', 'red'], alpha=0.7)
ax.set_ylabel('Time (seconds)')
ax.set_title('Training Time')
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(times), 
            f'{val:.1f}s', ha='center', va='bottom', fontsize=11)

# Final loss comparison
ax = axes[2]
losses = [pinn_losses[-1], fno_losses[-1]]
bars = ax.bar(models, losses, color=['blue', 'red'], alpha=0.7)
ax.set_ylabel('Final Loss (log scale)')
ax.set_yscale('log')
ax.set_title('Final Training Loss')
for bar, val in zip(bars, losses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
            f'{val:.2e}', ha='center', va='bottom', fontsize=11)

plt.suptitle('PINN vs FNO: Quantitative Comparison', fontsize=14)
plt.tight_layout()
summary_path = os.path.join(RESULTS_DIR, "pinn_vs_fno_summary.png")
plt.savefig(summary_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {summary_path}")

# ============================================================
# [9] Summary
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY: PINN vs FNO on Burgers Equation")
print("=" * 70)
print(f"{'Metric':<25} {'PINN':<20} {'FNO':<20}")
print("-" * 65)
print(f"{'Approach':<25} {'Equation-based':<20} {'Data-driven':<20}")
print(f"{'Training data needed':<25} {'No':<20} {'Yes (50 samples)':<20}")
print(f"{'Parameters':<25} {pinn_params:<20,} {fno_params:<20,}")
print(f"{'Epochs':<25} {PINN_EPOCHS:<20} {FNO_EPOCHS:<20}")
print(f"{'Training time':<25} {pinn_time:<20.1f} {fno_time:<20.1f}")
print(f"{'Final loss':<25} {pinn_losses[-1]:<20.6e} {fno_losses[-1]:<20.6e}")
print(f"{'Time per epoch':<25} {pinn_time/PINN_EPOCHS:<20.4f} {fno_time/FNO_EPOCHS:<20.4f}")
print("-" * 65)
print()
print("Key takeaways:")
print("  1. PINN needs NO data but trains slower (autograd overhead)")
print("  2. FNO needs labeled data but trains faster (matrix ops)")
print("  3. PINN gives continuous solution (evaluate at any point)")
print("  4. FNO gives discrete solution (fixed grid resolution)")
print("  5. PINN is better for: inverse problems, unknown PDEs")
print("  6. FNO is better for: fast inference, many forward solves")
print("=" * 70)
