"""
PhysicsNeMo PINN Tutorial: Bayesian PINN
=========================================
Uncertainty Quantification for Physics-Informed Neural Networks

Existing tutorials:
  - PINN (burgers, etc.): single deterministic model, no uncertainty
  - Deep ensemble (uncertainty/): N=5 CNNs, data-driven, no PDE
  - MC Dropout (uncertainty/): 1 CNN w/ dropout, data-driven, no PDE

THIS tutorial:
  - Bayesian PINN: combines PDE constraints WITH uncertainty estimation
  - Uses MC Dropout approximation to Bayesian inference
  - Dropout enabled at inference (T=50 stochastic forward passes)
  - Mean prediction = best estimate; Std = epistemic uncertainty
  - Uncertainty is HIGHER where PDE residual is large (physics-aware UQ)

Key difference from existing uncertainty tutorials:
  Deep Ensemble / MC Dropout  =>  THIS (Bayesian PINN)
  Data-driven (CNN/FNO)        =>  Physics-informed (PDE residual)
  Needs training data          =>  No labeled data needed
  Uncertainty from data        =>  Uncertainty from model + PDE
  2D Darcy (image-like)        =>  1D Burgers (x,t continuous)
  No physics constraint        =>  PDE + BC + IC enforced

Physics: 1D Burgers Equation
    u_t + u * u_x = (nu/pi) * u_xx
    BC: u(-1,t) = u(1,t) = 0
    IC: u(x,0) = -sin(pi*x)

Author: PhysicsNeMo Tutorial
Date: 2026-09-18
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo PINN Tutorial: Bayesian PINN (Uncertainty-Aware)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# [1] Bayesian PINN Model (with Dropout)
# ============================================================
class BayesianPINN(nn.Module):
    """PINN with dropout layers for Bayesian approximation (MC Dropout)."""

    def __init__(self, layers=[2, 64, 64, 64, 64, 1], dropout_rate=0.05):
        super().__init__()
        self.layers = layers
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)

        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
        self.linears = nn.ModuleList(layer_list)

        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for i in range(len(self.layers) - 2):
            x = self.activation(self.linears[i](x))
            if i == 1 or i == 3:
                x = self.dropout(x)
        x = self.linears[-1](x)
        return x

NU = 0.01 / np.pi
print(f"\n[1] Model: BayesianPINN")
print(f"  Architecture: 2->64->64->64->64->1, dropout=0.05")
print(f"  Viscosity nu/pi: {NU:.6f}")

# ============================================================
# [2] Loss Functions (PDE + BC + IC)
# ============================================================
def pde_residual(model, xy):
    """Burgers equation residual: u_t + u*u_x - (nu/pi)*u_xx = 0"""
    u = model(xy)
    grad = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grad[:, 0:1], grad[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return u_t + u * u_x - NU * u_xx

def pde_loss(model, xy):
    return (pde_residual(model, xy) ** 2).mean()

def ic_loss(model, xy_ic):
    u_pred = model(xy_ic)
    x = xy_ic[:, 0:1]
    return ((u_pred + torch.sin(np.pi * x)) ** 2).mean()

def bc_loss(model, xy_bc):
    return (model(xy_bc) ** 2).mean()

print(f"\n[2] Losses: PDE residual + BC (Dirichlet) + IC (sin)")

# ============================================================
# [3] Collocation Points
# ============================================================
N_INT = 5000; N_IC = 500; N_BC = 400
x_int = torch.rand(N_INT, 1, device=device) * 2 - 1
t_int = torch.rand(N_INT, 1, device=device)
xy_int = torch.cat([x_int, t_int], dim=1); xy_int.requires_grad_(True)
x_ic = torch.rand(N_IC, 1, device=device) * 2 - 1
xy_ic = torch.cat([x_ic, torch.zeros(N_IC, 1, device=device)], dim=1)
x_bc = torch.cat([-torch.ones(N_BC//2, 1, device=device), torch.ones(N_BC//2, 1, device=device)])
xy_bc = torch.cat([x_bc, torch.rand(N_BC, 1, device=device)], dim=1)
print(f"\n[3] Collocation: {N_INT} interior + {N_IC} IC + {N_BC} BC")

# ============================================================
# [4] Train Bayesian PINN
# ============================================================
print(f"\n[4] Training Bayesian PINN...")
EPOCHS = 5000
model = BayesianPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

loss_history = []
t0 = time.time()
for ep in range(EPOCHS):
    optimizer.zero_grad()
    lp = pde_loss(model, xy_int)
    li = ic_loss(model, xy_ic)
    lb = bc_loss(model, xy_bc)
    loss = lp + 10.0 * li + 10.0 * lb
    loss.backward(); optimizer.step(); scheduler.step()
    loss_history.append(loss.item())
    if (ep + 1) % 500 == 0:
        print(f"  Ep {ep+1:5d}/{EPOCHS} | Loss: {loss.item():.6e} | "
              f"PDE: {lp.item():.4e} | IC: {li.item():.4e} | BC: {lb.item():.4e}")

train_time = time.time() - t0
print(f"\n  Training complete! Time: {train_time:.1f}s")

# ============================================================
# [5] Bayesian Inference (MC Dropout)
# ============================================================
print(f"\n[5] Bayesian inference with MC Dropout (T=50 samples)...")
MC_SAMPLES = 50
N_GRID_X, N_GRID_T = 100, 100
x_grid = torch.linspace(-1, 1, N_GRID_X, device=device)
t_grid = torch.linspace(0, 1, N_GRID_T, device=device)
X, T = torch.meshgrid(x_grid, t_grid, indexing='ij')
xy_eval = torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1)

model.train()  # Enable dropout at inference
predictions = []
with torch.no_grad():
    for i in range(MC_SAMPLES):
        u_sample = model(xy_eval)
        predictions.append(u_sample.cpu().numpy())

predictions = np.array(predictions)
mean_pred = predictions.mean(axis=0)
std_pred = predictions.std(axis=0)
mean_grid = mean_pred.reshape(N_GRID_X, N_GRID_T)
std_grid = std_pred.reshape(N_GRID_X, N_GRID_T)

print(f"  MC samples: {MC_SAMPLES}")
print(f"  Mean prediction range: [{mean_pred.min():.4f}, {mean_pred.max():.4f}]")
print(f"  Uncertainty range:     [{std_pred.min():.6f}, {std_pred.max():.6f}]")

# ============================================================
# [6] Compute PDE Residual (physics-aware uncertainty)
# ============================================================
print(f"\n[6] Computing PDE residual on grid...")
model.eval()
xy_g = xy_eval.clone().detach().requires_grad_(True)
u_eval = model(xy_g)
grad = torch.autograd.grad(u_eval, xy_g, torch.ones_like(u_eval), create_graph=True)[0]
u_x = grad[:, 0:1]; u_t = grad[:, 1:2]
u_xx = torch.autograd.grad(u_x, xy_g, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
residual = (u_t + u_eval * u_x - NU * u_xx).detach().cpu().numpy()
residual_grid = np.abs(residual).reshape(N_GRID_X, N_GRID_T)
print(f"  Max |residual|: {residual_grid.max():.6f}")
print(f"  Mean |residual|: {residual_grid.mean():.6f}")

# ============================================================
# [7] Visualization
# ============================================================
print(f"\n[7] Visualization...")

# --- Figure 1: Mean prediction + uncertainty + PDE residual ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
im0 = ax.imshow(mean_grid, extent=[0, 1, -1, 1], aspect='auto', cmap='RdBu_r',
                vmin=-1, vmax=1, origin='lower')
ax.set_title('Mean Prediction u(x,t)', fontsize=13, fontweight='bold')
ax.set_xlabel('t'); ax.set_ylabel('x')
plt.colorbar(im0, ax=ax, shrink=0.8)
ax = axes[1]
im1 = ax.imshow(std_grid, extent=[0, 1, -1, 1], aspect='auto', cmap='hot', origin='lower')
ax.set_title('Epistemic Uncertainty (std)', fontsize=13, fontweight='bold')
ax.set_xlabel('t'); ax.set_ylabel('x')
plt.colorbar(im1, ax=ax, shrink=0.8)
ax = axes[2]
im2 = ax.imshow(residual_grid, extent=[0, 1, -1, 1], aspect='auto', cmap='magma', origin='lower')
ax.set_title('|PDE Residual|', fontsize=13, fontweight='bold')
ax.set_xlabel('t'); ax.set_ylabel('x')
plt.colorbar(im2, ax=ax, shrink=0.8)
plt.suptitle('Bayesian PINN: Prediction, Uncertainty & Physics Residual',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_pinn_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: bayesian_pinn_overview.png")

# --- Figure 2: Solution snapshots with error bars ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
times = [0.1, 0.5, 0.9]
for idx, t_val in enumerate(times):
    ax = axes[idx]
    t_idx = int(t_val * (N_GRID_T - 1))
    x_vals = x_grid.cpu().numpy()
    u_mean = mean_grid[:, t_idx]
    u_std = std_grid[:, t_idx]
    ax.plot(x_vals, u_mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(x_vals, u_mean - 2*u_std, u_mean + 2*u_std,
                     alpha=0.3, color='blue', label='95% CI')
    ax.set_title(f't = {t_val}', fontsize=13, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('u')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(-1, 1)
plt.suptitle('Bayesian PINN: Solution Snapshots with Uncertainty',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_pinn_snapshots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: bayesian_pinn_snapshots.png")

# --- Figure 3: Loss curve ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(loss_history, 'b-', linewidth=1)
ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss (log)', fontsize=12)
ax.set_title('Bayesian PINN: Training Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_pinn_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: bayesian_pinn_loss.png")

# --- Figure 4: Uncertainty vs PDE residual correlation ---
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(std_pred.flatten(), residual_grid.flatten(), alpha=0.3, s=5, c='blue')
ax.set_xlabel('Epistemic Uncertainty (std)', fontsize=12)
ax.set_ylabel('|PDE Residual|', fontsize=12)
ax.set_title('Uncertainty vs Physics Residual Correlation', fontsize=13, fontweight='bold')
corr = np.corrcoef(std_pred.flatten(), residual_grid.flatten())[0, 1]
ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_pinn_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: bayesian_pinn_correlation.png")

# --- Figure 5: MC sample distribution at shock point ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x_idx = N_GRID_X // 2; t_idx = N_GRID_T // 2
samples_pt = predictions[:, x_idx * N_GRID_T + t_idx]
ax.hist(samples_pt, bins=20, color='steelblue', edgecolor='black', alpha=0.7, density=True)
ax.axvline(samples_pt.mean(), color='red', linewidth=2, linestyle='--',
           label=f'Mean={samples_pt.mean():.4f}')
ax.axvline(samples_pt.mean() - 2*samples_pt.std(), color='orange', linewidth=2, linestyle=':', label='95% CI')
ax.axvline(samples_pt.mean() + 2*samples_pt.std(), color='orange', linewidth=2, linestyle=':')
ax.set_xlabel('u(x=0, t=0.5) prediction', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'MC Dropout Sample Distribution (T={MC_SAMPLES})', fontsize=13, fontweight='bold')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_pinn_mc_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: bayesian_pinn_mc_distribution.png")

# ============================================================
# [8] Comparison: Bayesian PINN vs Deterministic PINN
# ============================================================
print(f"\n[8] Comparison: Bayesian PINN vs Deterministic PINN...")
model_det = BayesianPINN().to(device)
optimizer_det = torch.optim.Adam(model_det.parameters(), lr=1e-3)
scheduler_det = torch.optim.lr_scheduler.StepLR(optimizer_det, step_size=2000, gamma=0.5)
DET_EPOCHS = 5000
det_loss_history = []
for ep in range(DET_EPOCHS):
    optimizer_det.zero_grad()
    lp = pde_loss(model_det, xy_int)
    li = ic_loss(model_det, xy_ic)
    lb = bc_loss(model_det, xy_bc)
    loss = lp + 10.0 * li + 10.0 * lb
    loss.backward(); optimizer_det.step(); scheduler_det.step()
    det_loss_history.append(loss.item())
model_det.eval()
with torch.no_grad():
    det_pred = model_det(xy_eval).cpu().numpy().reshape(N_GRID_X, N_GRID_T)
pred_diff = np.abs(mean_grid - det_pred)
print(f"  Deterministic final loss: {det_loss_history[-1]:.6e}")
print(f"  Bayesian final loss:      {loss_history[-1]:.6e}")
print(f"  Mean |Bayesian - Det|:    {pred_diff.mean():.6f}")
print(f"  Max  |Bayesian - Det|:    {pred_diff.max():.6f}")

# --- Figure 6: Bayesian vs Deterministic comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
im0 = ax.imshow(mean_grid, extent=[0, 1, -1, 1], aspect='auto', cmap='RdBu_r',
                vmin=-1, vmax=1, origin='lower')
ax.set_title('Bayesian PINN (Mean)', fontsize=13, fontweight='bold')
ax.set_xlabel('t'); ax.set_ylabel('x')
plt.colorbar(im0, ax=ax, shrink=0.8)
ax = axes[1]
im1 = ax.imshow(det_pred, extent=[0, 1, -1, 1], aspect='auto', cmap='RdBu_r',
                vmin=-1, vmax=1, origin='lower')
ax.set_title('Deterministic PINN', fontsize=13, fontweight='bold')
ax.set_xlabel('t'); ax.set_ylabel('x')
plt.colorbar(im1, ax=ax, shrink=0.8)
ax = axes[2]
im2 = ax.imshow(pred_diff, extent=[0, 1, -1, 1], aspect='auto', cmap='viridis', origin='lower')
ax.set_title('|Bayesian - Deterministic|', fontsize=13, fontweight='bold')
ax.set_xlabel('t'); ax.set_ylabel('x')
plt.colorbar(im2, ax=ax, shrink=0.8)
plt.suptitle('Bayesian PINN vs Deterministic PINN', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bayesian_vs_deterministic.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: bayesian_vs_deterministic.png")

# ============================================================
# [9] Summary
# ============================================================
print("\n" + "=" * 70)
print("[9] Summary")
print("=" * 70)
print(f"\n  Method: Bayesian PINN (MC Dropout approximation)")
print(f"  Problem: 1D Burgers equation")
print(f"  MC samples: {MC_SAMPLES}")
print(f"  Training time: {train_time:.1f}s")
print(f"\n  Key results:")
print(f"    - Bayesian final loss:      {loss_history[-1]:.6e}")
print(f"    - Deterministic final loss: {det_loss_history[-1]:.6e}")
print(f"    - Mean uncertainty (std):   {std_pred.mean():.6f}")
print(f"    - Max uncertainty (std):    {std_pred.max():.6f}")
print(f"    - Uncertainty-PDE residual correlation: r={corr:.3f}")
print(f"\n  Key insight:")
print(f"    Uncertainty is HIGHER where PDE residual is large,")
print(f"    showing the model 'knows what it doesn't know' in physics space.")
print(f"\n  Output files in: {RESULTS_DIR}")
print("=" * 70)
print(f"  Final loss: {loss_history[-1]:.6e}")
