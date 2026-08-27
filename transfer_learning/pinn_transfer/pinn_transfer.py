"""
PINN Transfer Learning: Burgers → Sine-Gordon
=============================================
This tutorial implements transfer learning for PINNs, transferring
knowledge from one PDE (Burgers) to another (Sine-Gordon).

Existing tutorial (transfer_fno):
  - FNO transfer: Darcy coarse → fine (same PDE, different parameters)
  - Data-driven: freeze/fine-tune encoder weights
  - Same loss function (MSE)

THIS tutorial:
  - PINN transfer: Burgers → Sine-Gordon (DIFFERENT PDE)
  - Equation-based: transfer network weights, change loss function
  - The PDE residual loss itself changes (different equation)

Key concepts:
  1. Cross-PDE transfer: Burgers (hyperbolic) → Sine-Gordon (nonlinear wave)
  2. Loss function transfer: PDE residual changes, but network structure stays
  3. Three strategies: scratch, freeze-encoder, full fine-tune
  4. Data efficiency: less training needed with pre-trained weights
  5. Feature reuse: low-frequency features (smoothness, gradients) transfer

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
print("PINN Transfer Learning: Burgers → Sine-Gordon")
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
# Source PDE: Burgers equation (hyperbolic, shock wave)
#   u_t + u * u_x = nu * u_xx
#
# Target PDE: Sine-Gordon equation (nonlinear wave, soliton)
#   u_tt - u_xx + sin(u) = 0
#
# Both are 1D nonlinear PDEs, but fundamentally different:
#   - Burgers: 1st order in time, dissipative (shock waves)
#   - Sine-Gordon: 2nd order in time, conservative (solitons)
#
# Transfer: Can a network trained on Burgers help solve Sine-Gordon?

print(f"\n[1] Problem: PINN Transfer Learning")
print(f"  Source: Burgers equation (u_t + u*u_x = nu*u_xx)")
print(f"  Target: Sine-Gordon equation (u_tt - u_xx + sin(u) = 0)")
print(f"  Question: Can Burgers-trained network help solve Sine-Gordon?")

# ============================================================
# [2] Neural Network (shared architecture)
# ============================================================
class PINNNet(nn.Module):
    """Fully connected network for both Burgers and Sine-Gordon."""
    def __init__(self, in_dim=2, out_dim=1, hidden=50, layers=5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden))
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, out_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return self.out(x)

print(f"\n[2] Network: FC(2→50→50→50→50→50→1), Tanh activation")

# ============================================================
# [3] Source PDE: Burgers Equation
# ============================================================
NU_BURGERS = 0.01 / np.pi  # Viscosity

def burgers_residual(model, x, t):
    """Burgers: u_t + u*u_x = nu*u_xx"""
    xt = torch.cat([x, t], dim=1)
    u = model(xt)
    # 1st derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # 2nd derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    # Residual
    return u_t + u * u_x - NU_BURGERS * u_xx

def burgers_bc(model):
    """Boundary: u(-1,t) = u(1,t) = 0, IC: u(x,0) = -sin(pi*x)"""
    # IC
    x_ic = torch.linspace(-1, 1, 100, device=device).reshape(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(np.pi * x_ic)
    pred_ic = model(torch.cat([x_ic, t_ic], dim=1))
    loss_ic = F.mse_loss(pred_ic, u_ic)
    # BC
    t_bc = torch.linspace(0, 1, 50, device=device).reshape(-1, 1)
    x_left = -torch.ones_like(t_bc)
    x_right = torch.ones_like(t_bc)
    loss_bc = F.mse_loss(model(torch.cat([x_left, t_bc], dim=1)), torch.zeros_like(t_bc)) + \
              F.mse_loss(model(torch.cat([x_right, t_bc], dim=1)), torch.zeros_like(t_bc))
    return loss_ic + loss_bc

# ============================================================
# [4] Target PDE: Sine-Gordon Equation
# ============================================================
def sine_gordon_residual(model, x, t):
    """Sine-Gordon: u_tt - u_xx + sin(u) = 0"""
    xt = torch.cat([x, t], dim=1)
    u = model(xt)
    # 1st derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # 2nd derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    # Residual
    return u_tt - u_xx + torch.sin(u)

def sine_gordon_bc(model):
    """BC: u(-5,t) = u(5,t) = 0, IC: u(x,0) = 4*atan(exp(x)) (kink soliton)"""
    # IC: Sine-Gordon kink soliton: u(x,0) = 4*atan(exp(x))
    x_ic = torch.linspace(-5, 5, 100, device=device).reshape(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = 4 * torch.atan(torch.exp(x_ic))
    pred_ic = model(torch.cat([x_ic, t_ic], dim=1))
    loss_ic = F.mse_loss(pred_ic, u_ic)
    # IC velocity: u_t(x,0) = 0 (static kink)
    t_ic2 = torch.zeros_like(x_ic)
    u_pred = model(torch.cat([x_ic, t_ic2], dim=1))
    u_t_ic = torch.autograd.grad(u_pred, t_ic2, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    loss_ut = F.mse_loss(u_t_ic, torch.zeros_like(u_t_ic))
    # BC
    t_bc = torch.linspace(0, 5, 50, device=device).reshape(-1, 1)
    x_left = -5 * torch.ones_like(t_bc)
    x_right = 5 * torch.ones_like(t_bc)
    loss_bc = F.mse_loss(model(torch.cat([x_left, t_bc], dim=1)), torch.zeros_like(t_bc)) + \
              F.mse_loss(model(torch.cat([x_right, t_bc], dim=1)), torch.zeros_like(t_bc))
    return loss_ic + loss_ut + loss_bc

# ============================================================
# [5] Phase 1: Train on Source PDE (Burgers)
# ============================================================
EPOCHS_SOURCE = 3000
EPOCHS_TARGET = 3000
N_INTERIOR = 2000
LR = 1e-3

print(f"\n[5] Phase 1: Train on Burgers equation ({EPOCHS_SOURCE} epochs)")
print("-" * 70)

source_model = PINNNet(in_dim=2, out_dim=1, hidden=50, layers=5).to(device)
opt = torch.optim.Adam(source_model.parameters(), lr=LR)
source_losses = []
start = time.time()

for epoch in range(EPOCHS_SOURCE):
    source_model.train()
    # Interior points
    x = torch.rand(N_INTERIOR, 1, device=device) * 2 - 1  # [-1, 1]
    t = torch.rand(N_INTERIOR, 1, device=device)  # [0, 1]
    x.requires_grad_(True); t.requires_grad_(True)
    # PDE loss
    res = burgers_residual(source_model, x, t)
    loss_pde = torch.mean(res ** 2)
    # BC/IC loss
    loss_bc = burgers_bc(source_model)
    loss = loss_pde + 10.0 * loss_bc
    opt.zero_grad(); loss.backward(); opt.step()
    source_losses.append(loss.item())
    if epoch % 500 == 0 or epoch == EPOCHS_SOURCE - 1:
        print(f"  Epoch {epoch:4d} | Loss: {loss.item():.6e} | PDE: {loss_pde.item():.6e} | BC: {loss_bc.item():.6e} | Time: {time.time()-start:.1f}s")

source_time = time.time() - start
print("-" * 70)
print(f"  Source training time: {source_time:.1f}s")

# ============================================================
# [6] Phase 2: Transfer to Target PDE (Sine-Gordon)
# ============================================================
# Three strategies:
# 1. Scratch: random init, train on Sine-Gordon from scratch
# 2. Freeze: load Burgers weights, freeze first 3 layers, fine-tune last 2
# 3. Full FT: load Burgers weights, fine-tune all layers

print(f"\n[6] Phase 2: Transfer to Sine-Gordon ({EPOCHS_TARGET} epochs)")
print("-" * 70)

strategies = ['scratch', 'freeze', 'full_ft']
results = {}

for strategy in strategies:
    print(f"\n  Strategy: {strategy}")

    # Create model
    model = PINNNet(in_dim=2, out_dim=1, hidden=50, layers=5).to(device)

    if strategy == 'scratch':
        # Random init
        lr = 1e-3
    elif strategy == 'freeze':
        # Load source weights, freeze first 3 layers
        model.load_state_dict(source_model.state_dict())
        for i, layer in enumerate(model.layers):
            if i < 3:
                for p in layer.parameters():
                    p.requires_grad = False
        lr = 1e-3
    elif strategy == 'full_ft':
        # Load source weights, fine-tune all
        model.load_state_dict(source_model.state_dict())
        lr = 5e-4  # Lower LR for fine-tuning

    # Optimizer (only non-frozen params)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    losses = []
    start = time.time()
    for epoch in range(EPOCHS_TARGET):
        model.train()
        # Interior points for Sine-Gordon: x in [-5, 5], t in [0, 5]
        x = torch.rand(N_INTERIOR, 1, device=device) * 10 - 5
        t = torch.rand(N_INTERIOR, 1, device=device) * 5
        x.requires_grad_(True); t.requires_grad_(True)
        res = sine_gordon_residual(model, x, t)
        loss_pde = torch.mean(res ** 2)
        loss_bc = sine_gordon_bc(model)
        loss = loss_pde + 10.0 * loss_bc
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if epoch % 500 == 0 or epoch == EPOCHS_TARGET - 1:
            print(f"    Epoch {epoch:4d} | Loss: {loss.item():.6e} | PDE: {loss_pde.item():.6e} | BC: {loss_bc.item():.6e} | Time: {time.time()-start:.1f}s")

    t_time = time.time() - start
    final_loss = losses[-1]
    results[strategy] = {'model': model, 'losses': losses, 'final_loss': final_loss, 'time': t_time}
    print(f"    Final loss: {final_loss:.6e}, Time: {t_time:.1f}s")

print("-" * 70)

# ============================================================
# [7] Visualization
# ============================================================
print(f"\n[7] Generating visualizations...")

# --- Figure 1: Source (Burgers) training loss ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(source_losses, linewidth=1.5, color='blue')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Phase 1: Source PDE (Burgers) Training Loss'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pinn_transfer_source_loss.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Target (Sine-Gordon) training loss comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
colors = {'scratch': 'red', 'freeze': 'green', 'full_ft': 'blue'}
labels = {'scratch': 'Scratch (random init)', 'freeze': 'Freeze (first 3 layers)', 'full_ft': 'Full Fine-Tune'}
for strategy in strategies:
    ax.semilogy(results[strategy]['losses'], linewidth=1.5, color=colors[strategy], label=labels[strategy], alpha=0.8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Phase 2: Target PDE (Sine-Gordon) Training Loss — Strategy Comparison')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pinn_transfer_target_loss.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: Sine-Gordon solutions ---
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
x_eval = torch.linspace(-5, 5, 100, device=device).reshape(-1, 1)
times = [0, 1, 2, 4]

for row, strategy in enumerate(strategies):
    model = results[strategy]['model']
    model.eval()
    for col, t_val in enumerate(times):
        t_eval = torch.full_like(x_eval, t_val)
        with torch.no_grad():
            u_pred = model(torch.cat([x_eval, t_eval], dim=1)).cpu().numpy()
        # Analytical kink soliton: u(x,t) = 4*atan(exp(x/sqrt(1-v^2) - vt))
        # For static kink (v=0): u = 4*atan(exp(x))
        u_exact = 4 * np.arctan(np.exp(x_eval.cpu().numpy()))

        ax = axes[row, col]
        ax.plot(x_eval.cpu().numpy(), u_exact, 'k--', linewidth=2, label='Exact')
        ax.plot(x_eval.cpu().numpy(), u_pred, 'r-', linewidth=1.5, label='Predicted')
        ax.set_title(f'{labels[strategy]}\nt={t_val}', fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('u')
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 5)

plt.suptitle('Sine-Gordon Kink Soliton: Transfer Learning Strategies', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pinn_transfer_solutions.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: Summary bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Final loss comparison
ax = axes[0]
final_losses = [results[s]['final_loss'] for s in strategies]
bars = ax.bar(labels.values(), final_losses, color=[colors[s] for s in strategies], alpha=0.7, edgecolor='black')
ax.set_ylabel('Final Loss')
ax.set_title('Final Loss Comparison (lower = better)')
for bar, val in zip(bars, final_losses):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.2e}',
            ha='center', va='bottom', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Training time comparison
ax = axes[1]
times = [results[s]['time'] for s in strategies]
bars = ax.bar(labels.values(), times, color=[colors[s] for s in strategies], alpha=0.7, edgecolor='black')
ax.set_ylabel('Training Time (s)')
ax.set_title('Training Time Comparison')
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.1f}s',
            ha='center', va='bottom', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('PINN Transfer Learning: Strategy Comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pinn_transfer_summary.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 5: Concept explanation ---
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.text(0.5, 0.95, 'PINN Transfer Learning: Burgers → Sine-Gordon', ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.80,
    'FNO Transfer (existing tutorial):\n'
    '  - Same PDE (Darcy), different parameters\n'
    '  - Data-driven (MSE loss)\n'
    '  - Freeze/fine-tune encoder weights\n'
    '  - Loss function stays the same',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(0.55, 0.80,
    'PINN Transfer (THIS tutorial):\n'
    '  - DIFFERENT PDE (Burgers → Sine-Gordon)\n'
    '  - Equation-based (PDE residual loss)\n'
    '  - Transfer network weights, change loss\n'
    '  - Loss function ITSELF changes!',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.50,
    'Source: Burgers equation\n'
    '  u_t + u*u_x = nu*u_xx\n'
    '  - 1st order in time\n'
    '  - Dissipative (shock waves)\n'
    '  - Hyperbolic',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.55, 0.50,
    'Target: Sine-Gordon equation\n'
    '  u_tt - u_xx + sin(u) = 0\n'
    '  - 2nd order in time\n'
    '  - Conservative (solitons)\n'
    '  - Nonlinear wave',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.20,
    'What transfers?\n'
    '  - Low-frequency features (smoothness)\n'
    '  - Gradient computation (autograd)\n'
    '  - Network capacity\n'
    '  - Tanh activation behavior',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax.text(0.55, 0.20,
    'Three strategies:\n'
    '  1. Scratch: random init (baseline)\n'
    '  2. Freeze: lock first 3 layers\n'
    '  3. Full FT: fine-tune all layers\n'
    '  Key: loss function changes!\n'
    '  (Burgers residual → Sine-Gordon residual)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightskyblue', alpha=0.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pinn_transfer_explanation.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: PINN Transfer Learning (Burgers → Sine-Gordon)")
print("=" * 70)
print(f"  Source PDE:        Burgers (u_t + u*u_x = nu*u_xx)")
print(f"  Target PDE:        Sine-Gordon (u_tt - u_xx + sin(u) = 0)")
print(f"  Source epochs:     {EPOCHS_SOURCE}")
print(f"  Target epochs:     {EPOCHS_TARGET}")
print(f"  Source train time: {source_time:.1f}s")
print(f"  --- Strategy Results ---")
for s in strategies:
    print(f"  {s:10s}: final_loss={results[s]['final_loss']:.6e}, time={results[s]['time']:.1f}s")
print()
print("Key observations:")
print("  1. CROSS-PDE TRANSFER: Burgers (hyperbolic) → Sine-Gordon (wave) — different PDE types")
print("  2. LOSS FUNCTION CHANGES: PDE residual changes (unlike FNO transfer where MSE stays)")
print("  3. THREE STRATEGIES: scratch (baseline), freeze (partial), full FT (all layers)")
print("  4. FEATURE REUSE: Low-frequency features (smoothness, gradients) transfer across PDEs")
print("  5. vs FNO TRANSFER: FNO transfers within same PDE; PINN transfers across different PDEs")
print("  6. DATA EFFICIENCY: Pre-trained network may converge faster (if transfer helps)")
print("=" * 70)
