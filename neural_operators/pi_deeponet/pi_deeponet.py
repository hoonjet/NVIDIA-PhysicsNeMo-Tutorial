"""
PhysicsNeMo Tutorial: Physics-Informed DeepONet (PI-DeepONet)
==============================================================
Operator Learning + PDE Residual = Less Data, More Physics

Existing tutorials:
  - DeepONet (data-only, branch-trunk, Burgers)
  - PINO (FNO + PDE residual, Darcy)
  - PINN (point-wise, no operator learning)

THIS tutorial:
  - DeepONet + PDE residual loss (operator learning + physics)
  - 1D Burgers equation: learn operator u₀(x) → u(x, t)
  - Compare: data-only vs data+physics (PI-DeepONet)
  - Show that PDE loss reduces data requirement

Key difference from existing tutorials:
  ┌──────────────────────┬──────────────────────────┐
  │ DeepONet (existing)   │ THIS (PI-DeepONet)      │
  ├──────────────────────┼──────────────────────────┤
  │ Data-only loss        │ Data + PDE residual loss │
  │ Needs lots of data    │ Less data needed         │
  │ No physics constraint │ Physics-constrained      │
  │ May violate PDE       │ PDE-satisfying           │
  └──────────────────────┴──────────────────────────┘

  ┌──────────────────────┬──────────────────────────┐
  │ PINO (existing)      │ THIS (PI-DeepONet)      │
  ├──────────────────────┼──────────────────────────┤
  │ FNO backbone          │ DeepONet backbone        │
  │ Grid-based (regular)  │ Sensor-based (irregular) │
  │ Fixed resolution      │ Resolution-independent   │
  └──────────────────────┴──────────────────────────┘

Physics: 1D Burgers Equation
  ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
  IC: u(x, 0) = u₀(x)  (varies — operator input)
  BC: u(-1, t) = u(1, t) = 0

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

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo Tutorial: PI-DeepONet (Physics-Informed DeepONet)")
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
# [1] Problem Parameters
# ============================================================
NU = 0.01 / np.pi  # Viscosity (Burgers)
N_SENSORS = 50     # Number of sensor points for branch input
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

print(f"\n[1] Problem Parameters:")
print(f"  Equation: 1D Burgers (∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²)")
print(f"  Viscosity ν = {NU:.6f}")
print(f"  Domain: x∈[{X_MIN}, {X_MAX}], t∈[{T_MIN}, {T_MAX}]")
print(f"  Sensors: {N_SENSORS} (branch input)")

# ============================================================
# [2] Data Generation (Burgers solver)
# ============================================================
def solve_burgers(u0, x, t, nu, nx=100):
    """Solve 1D Burgers via finite difference (explicit)."""
    dx = (x[-1] - x[0]) / (len(x) - 1)
    u = u0.copy()
    u_all = [u.copy()]
    dt = 0.0005
    n_steps = int(t / dt)
    for _ in range(n_steps):
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        u_x = np.zeros_like(u)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2*dx)
        u_new = u + dt * (-u * u_x + nu * u_xx)
        u_new[0] = 0; u_new[-1] = 0  # BC
        u = u_new
        u_all.append(u.copy())
    return u

def generate_initial_condition(x, n_modes=3):
    """Random smooth initial condition."""
    coeffs = np.random.randn(n_modes) * 0.5
    u0 = np.zeros_like(x)
    for k in range(1, n_modes + 1):
        u0 += coeffs[k-1] * np.sin(k * np.pi * x)
    return u0

print(f"\n[2] Generating training data...")

# Sensor points (branch input locations)
x_sensors = np.linspace(X_MIN, X_MAX, N_SENSORS, dtype=np.float32)

# Generate dataset
N_TRAIN = 200  # Small dataset (PI-DeepONet needs less)
N_TEST = 50
N_COLLOC = 2000  # Collocation points for PDE loss

# Training data
train_u0 = []
train_u_final = []
for i in range(N_TRAIN):
    u0 = generate_initial_condition(x_sensors)
    x_fine = np.linspace(X_MIN, X_MAX, 100, dtype=np.float32)
    u0_fine = generate_initial_condition(x_fine)
    u_final = solve_burgers(u0_fine, x_fine, T_MAX, NU)
    # Interpolate to sensor points
    u0_sensor = np.interp(x_sensors, x_fine, u0_fine).astype(np.float32)
    u_final_sensor = np.interp(x_sensors, x_fine, u_final).astype(np.float32)
    train_u0.append(u0_sensor)
    train_u_final.append(u_final_sensor)

train_u0 = torch.tensor(np.array(train_u0), dtype=torch.float32, device=device)
train_u_final = torch.tensor(np.array(train_u_final), dtype=torch.float32, device=device)

# Test data
test_u0 = []
test_u_final = []
for i in range(N_TEST):
    u0 = generate_initial_condition(x_sensors)
    x_fine = np.linspace(X_MIN, X_MAX, 100, dtype=np.float32)
    u0_fine = generate_initial_condition(x_fine)
    u_final = solve_burgers(u0_fine, x_fine, T_MAX, NU)
    u0_sensor = np.interp(x_sensors, x_fine, u0_fine).astype(np.float32)
    u_final_sensor = np.interp(x_sensors, x_fine, u_final).astype(np.float32)
    test_u0.append(u0_sensor)
    test_u_final.append(u_final_sensor)

test_u0 = torch.tensor(np.array(test_u0), dtype=torch.float32, device=device)
test_u_final = torch.tensor(np.array(test_u_final), dtype=torch.float32, device=device)

# Collocation points for PDE loss
x_colloc = torch.rand(N_COLLOC, 1, device=device) * (X_MAX - X_MIN) + X_MIN
t_colloc = torch.rand(N_COLLOC, 1, device=device) * (T_MAX - T_MIN) + T_MIN
xt_colloc = torch.cat([x_colloc, t_colloc], dim=1)
xt_colloc.requires_grad_(True)

print(f"  Train: {N_TRAIN}, Test: {N_TEST}, Collocation: {N_COLLOC}")

# ============================================================
# [3] PI-DeepONet Model
# ============================================================
class BranchNet(nn.Module):
    """Branch: maps initial condition (sensors) → latent."""
    def __init__(self, n_sensors=N_SENSORS, hidden=64, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent),
        )
    def forward(self, u0):
        return self.net(u0)

class TrunkNet(nn.Module):
    """Trunk: maps (x, t) → latent."""
    def __init__(self, hidden=64, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent),
        )
    def forward(self, xt):
        return self.net(xt)

class PIDeepONet(nn.Module):
    """
    Physics-Informed DeepONet.

    Output: u(x, t) = Σ_k branch_k(u₀) * trunk_k(x, t)

    PDE residual: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0
    """
    def __init__(self, n_sensors=N_SENSORS, latent=32):
        super().__init__()
        self.branch = BranchNet(n_sensors, latent=latent)
        self.trunk = TrunkNet(latent=latent)
        self.latent = latent

    def forward(self, u0, xt):
        """
        u0: [B, n_sensors] — initial condition at sensors
        xt: [N, 2] — query points (x, t)
        Returns: [B, N] — predicted u at each query point
        """
        b = self.branch(u0)  # [B, latent]
        t = self.trunk(xt)   # [N, latent]
        # Dot product: [B, N]
        u = torch.einsum('bl,nl->bn', b, t)
        return u

print(f"\n[3] PI-DeepONet Model:")
print(f"  Branch: {N_SENSORS} → 64 → 64 → 32")
print(f"  Trunk: 2 → 64 → 64 → 32")
print(f"  Output: u(x,t) = Σ branch_k(u₀) · trunk_k(x,t)")

# ============================================================
# [4] Loss Functions
# ============================================================
def compute_data_loss(model, u0_batch, u_target_batch, x_query):
    """Data loss: match known solution at t=T_MAX."""
    t_query = torch.full((x_query.shape[0], 1), T_MAX, device=device)
    xt_query = torch.cat([x_query, t_query], dim=1)
    u_pred = model(u0_batch, xt_query)  # [B, N]
    return F.mse_loss(u_pred, u_target_batch)

def compute_pde_loss(model, u0_batch, xt_colloc):
    """
    PDE residual: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0

    Need derivatives w.r.t. x and t at collocation points.
    """
    B = u0_batch.shape[0]
    N = xt_colloc.shape[0]

    # Repeat u0 for each collocation point
    u0_rep = u0_batch.unsqueeze(1).expand(B, N, N_SENSORS).reshape(B*N, N_SENSORS)
    xt_rep = xt_colloc.unsqueeze(0).expand(B, N, 2).reshape(B*N, 2)
    xt_rep.requires_grad_(True)

    u_pred = model(u0_rep, xt_rep)  # [B*N, 1] — need to reshape

    # Actually, let's do it per-sample for cleaner autograd
    # Use first sample for PDE loss (or average over batch)
    total_loss = 0.0
    n_samples = min(B, 10)  # Use subset for efficiency

    for i in range(n_samples):
        u0_i = u0_batch[i:i+1]  # [1, n_sensors]
        xt_i = xt_colloc  # [N, 2]
        xt_i.requires_grad_(True)

        u_i = model(u0_i, xt_i)  # [1, N] → [N]
        u_i = u_i.squeeze(0)  # [N]

        # Derivatives
        u_x = torch.autograd.grad(u_i.sum(), xt_i, create_graph=True, retain_graph=True)[0][:, 0]
        u_t = torch.autograd.grad(u_i.sum(), xt_i, create_graph=True, retain_graph=True)[0][:, 1]

        u_xx = torch.autograd.grad(u_x.sum(), xt_i, create_graph=True, retain_graph=True)[0][:, 0]

        residual = u_t + u_i * u_x - NU * u_xx
        total_loss += (residual**2).mean()

    return total_loss / n_samples

print(f"\n[4] Loss Functions:")
print(f"  Data: MSE at t=T_MAX (known solutions)")
print(f"  PDE: Burgers residual (∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0)")

# ============================================================
# [5] Training: Data-only vs PI-DeepONet
# ============================================================
N_EPOCHS = 3000
LR = 1e-3
BATCH_SIZE = 32

# Query points for data loss
x_query = torch.linspace(X_MIN, X_MAX, N_SENSORS, device=device).view(-1, 1)

# --- Model 1: Data-only DeepONet ---
print(f"\n[5a] Training Data-only DeepONet...")
model_data = PIDeepONet().to(device)
opt_data = torch.optim.Adam(model_data.parameters(), lr=LR)

loss_history_data = []
t_start = time.time()

for epoch in range(N_EPOCHS):
    model_data.train()
    perm = torch.randperm(N_TRAIN)
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        u0_batch = train_u0[idx]
        u_target = train_u_final[idx]

        loss = compute_data_loss(model_data, u0_batch, u_target, x_query)

        opt_data.zero_grad()
        loss.backward()
        opt_data.step()

        epoch_loss += loss.item()
        n_batches += 1

    loss_history_data.append(epoch_loss / n_batches)

    if (epoch + 1) % 500 == 0:
        print(f"  Epoch {epoch+1:4d}/{N_EPOCHS} | Loss: {loss_history_data[-1]:.6e} | Time: {time.time()-t_start:.1f}s")

print(f"  Done! Time: {time.time()-t_start:.1f}s")

# --- Model 2: PI-DeepONet (Data + PDE) ---
print(f"\n[5b] Training PI-DeepONet (Data + PDE)...")
model_pi = PIDeepONet().to(device)
opt_pi = torch.optim.Adam(model_pi.parameters(), lr=LR)

LAMBDA_DATA = 1.0
LAMBDA_PDE = 0.5

loss_history_pi = {'total': [], 'data': [], 'pde': []}
t_start = time.time()

for epoch in range(N_EPOCHS):
    model_pi.train()
    perm = torch.randperm(N_TRAIN)
    epoch_total = 0.0
    epoch_data = 0.0
    epoch_pde = 0.0
    n_batches = 0

    for i in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        u0_batch = train_u0[idx]
        u_target = train_u_final[idx]

        loss_d = compute_data_loss(model_pi, u0_batch, u_target, x_query)
        loss_p = compute_pde_loss(model_pi, u0_batch, xt_colloc)

        loss = LAMBDA_DATA * loss_d + LAMBDA_PDE * loss_p

        opt_pi.zero_grad()
        loss.backward()
        opt_pi.step()

        epoch_total += loss.item()
        epoch_data += loss_d.item()
        epoch_pde += loss_p.item()
        n_batches += 1

    loss_history_pi['total'].append(epoch_total / n_batches)
    loss_history_pi['data'].append(epoch_data / n_batches)
    loss_history_pi['pde'].append(epoch_pde / n_batches)

    if (epoch + 1) % 500 == 0:
        print(f"  Epoch {epoch+1:4d}/{N_EPOCHS} | "
              f"Total: {loss_history_pi['total'][-1]:.6e} | "
              f"Data: {loss_history_pi['data'][-1]:.6e} | "
              f"PDE: {loss_history_pi['pde'][-1]:.6e} | "
              f"Time: {time.time()-t_start:.1f}s")

print(f"  Done! Time: {time.time()-t_start:.1f}s")

# ============================================================
# [6] Evaluation
# ============================================================
print(f"\n[6] Evaluation...")

model_data.eval()
model_pi.eval()

with torch.no_grad():
    # Test: predict final solution
    pred_data = model_data(test_u0, torch.cat([x_query, torch.full_like(x_query, T_MAX)], dim=1))
    pred_pi = model_pi(test_u0, torch.cat([x_query, torch.full_like(x_query, T_MAX)], dim=1))

    # Relative L2 error
    err_data = torch.norm(pred_data - test_u_final) / torch.norm(test_u_final)
    err_pi = torch.norm(pred_pi - test_u_final) / torch.norm(test_u_final)

print(f"  Data-only DeepONet: Rel L2 = {err_data.item():.4f} ({err_data.item()*100:.2f}%)")
print(f"  PI-DeepONet:        Rel L2 = {err_pi.item():.4f} ({err_pi.item()*100:.2f}%)")

# ============================================================
# [7] Visualization
# ============================================================
print(f"\n[7] Visualization...")

# --- Figure 1: Prediction comparison ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for row in range(2):
    idx = row * 20
    u0 = test_u0[idx].cpu().numpy()
    u_true = test_u_final[idx].cpu().numpy()
    u_d = pred_data[idx].cpu().numpy()
    u_p = pred_pi[idx].cpu().numpy()

    axes[row, 0].plot(x_sensors, u0, 'b-o', markersize=3, label='u₀(x)')
    axes[row, 0].set_title(f'Initial Condition (sample {idx})', fontsize=12)
    axes[row, 0].set_xlabel('x'); axes[row, 0].set_ylabel('u₀')
    axes[row, 0].legend(); axes[row, 0].grid(True, alpha=0.3)

    axes[row, 1].plot(x_sensors, u_true, 'k-', linewidth=2, label='True')
    axes[row, 1].plot(x_sensors, u_d, 'r--', linewidth=2, label='Data-only')
    axes[row, 1].plot(x_sensors, u_p, 'g-.', linewidth=2, label='PI-DeepONet')
    axes[row, 1].set_title(f'Solution at t={T_MAX}', fontsize=12)
    axes[row, 1].set_xlabel('x'); axes[row, 1].set_ylabel('u')
    axes[row, 1].legend(); axes[row, 1].grid(True, alpha=0.3)

    err_d = np.abs(u_true - u_d)
    err_p = np.abs(u_true - u_p)
    axes[row, 2].plot(x_sensors, err_d, 'r-', linewidth=2, label='Data-only')
    axes[row, 2].plot(x_sensors, err_p, 'g-', linewidth=2, label='PI-DeepONet')
    axes[row, 2].set_title('Absolute Error', fontsize=12)
    axes[row, 2].set_xlabel('x'); axes[row, 2].set_ylabel('|Error|')
    axes[row, 2].legend(); axes[row, 2].grid(True, alpha=0.3)

plt.suptitle('PI-DeepONet vs Data-only DeepONet (Burgers)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pi_deeponet_prediction.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pi_deeponet_prediction.png")

# --- Figure 2: Loss comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.semilogy(loss_history_data, 'r-', linewidth=2, label='Data-only')
ax.semilogy(loss_history_pi['total'], 'g-', linewidth=2, label='PI-DeepONet (total)')
ax.semilogy(loss_history_pi['data'], 'g--', alpha=0.5, label='PI-DeepONet (data)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(loss_history_pi['pde'], 'b-', linewidth=2, label='PDE residual')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('PDE Loss', fontsize=12)
ax.set_title('PI-DeepONet PDE Residual Loss', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pi_deeponet_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pi_deeponet_loss.png")

# --- Figure 3: Concept ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.axis('off')
concept_text = (
    "DeepONet (existing, data-only):\n\n"
    "  u₀(sensors) ──→ Branch ──→ b\n"
    "                              ↓\n"
    "  (x, t) ──────→ Trunk ──→ t  → u = b·t\n"
    "                              ↓\n"
    "  Loss = MSE(u_pred, u_data)\n\n"
    "  ✗ Needs lots of data\n"
    "  ✗ May violate PDE\n\n"
    "PI-DeepONet (THIS):\n\n"
    "  u₀(sensors) ──→ Branch ──→ b\n"
    "                              ↓\n"
    "  (x, t) ──────→ Trunk ──→ t  → u = b·t\n"
    "                              ↓\n"
    "  Loss = λ_data·MSE + λ_pde·|PDE|²\n\n"
    "  ✓ Less data needed\n"
    "  ✓ PDE-constrained\n"
    "  ✓ Better generalization"
)
ax.text(0.05, 0.95, concept_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Data-only vs PI-DeepONet', fontsize=13, fontweight='bold')

ax = axes[1]
methods = ['Data-only\nDeepONet', 'PI-DeepONet\n(Data+PDE)']
errors = [err_data.item(), err_pi.item()]
colors = ['red', 'green']
bars = ax.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Relative L2 Error', fontsize=12)
ax.set_title('Test Error Comparison', fontsize=13, fontweight='bold')
for bar, err in zip(bars, errors):
    ax.text(bar.get_x() + bar.get_width()/2, err + 0.005, f'{err:.4f}',
            ha='center', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pi_deeponet_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pi_deeponet_concept.png")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("[8] Summary")
print("=" * 70)

print(f"\n  Method: PI-DeepONet (Physics-Informed DeepONet)")
print(f"  Physics: 1D Burgers equation")
print(f"  Operator: u₀(x) → u(x, t=T_MAX)")
print(f"  Training data: {N_TRAIN} samples (small dataset)")
print(f"")
print(f"  Test Relative L2 Error:")
print(f"    Data-only DeepONet: {err_data.item():.4f} ({err_data.item()*100:.2f}%)")
print(f"    PI-DeepONet:        {err_pi.item():.4f} ({err_pi.item()*100:.2f}%)")
improvement = (1 - err_pi.item() / err_data.item()) * 100 if err_data.item() > 0 else 0
print(f"    Improvement: {improvement:.1f}%")
print(f"")
print(f"  Key difference from existing tutorials:")
print(f"    DeepONet: data-only (needs lots of data)")
print(f"    PINO: FNO+PDE (grid-based, fixed resolution)")
print(f"    THIS: DeepONet+PDE (sensor-based, resolution-independent)")
print(f"")
print(f"  Results saved to: {RESULTS_DIR}")
print("=" * 70)
