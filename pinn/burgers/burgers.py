"""
PhysicsNeMo PINN Tutorial: Burgers Equation
=============================================
1D Viscous Burgers Equation:
    u_t + u * u_x = (nu / pi) * u_xx

Boundary conditions:
    u(-1, t) = u(1, t) = 0  (Dirichlet)
Initial condition:
    u(x, 0) = -sin(pi * x)

This is the classic PINN benchmark problem from Raissi et al. (2019).
The solution develops a shock wave that sharpens over time, making it
an excellent test of PINN's ability to capture discontinuities.

Author: PhysicsNeMo Tutorial
Date: 2026-07-20
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo PINN Tutorial: Burgers Equation")
print("=" * 70)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# [1] Neural Network Model (Fully Connected)
# ============================================================
class PINN(nn.Module):
    """
    Fully connected neural network for PINN.
    Input: (x, t) -> Output: u(x, t)
    
    Architecture: 2 -> 64 -> 64 -> 64 -> 64 -> 1
    """
    def __init__(self, layers=[2, 64, 64, 64, 64, 1]):
        super().__init__()
        self.layers = layers
        self.activation = nn.Tanh()
        
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        self.linears = nn.ModuleList(layer_list)
        
        # Xavier initialization
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: [N, 2] where columns are (x_coord, t_coord)
        for i in range(len(self.layers) - 2):
            x = self.activation(self.linears[i](x))
        x = self.linears[-1](x)  # No activation on output
        return x


# ============================================================
# [2] Problem Setup: Burgers Equation
# ============================================================
# Viscosity parameter (nu/pi in the equation)
NU = 0.01 / np.pi  # Small viscosity -> sharp shock

# Domain
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# ============================================================
# [3] Training Data
# ============================================================
N_IC = 200      # Initial condition points
N_BC = 100      # Boundary condition points (50 per boundary)
N_F = 10000     # Collocation points (PDE residual)

# --- Initial condition: u(x, 0) = -sin(pi * x) ---
x_ic = torch.linspace(X_MIN, X_MAX, N_IC).reshape(-1, 1)
t_ic = torch.zeros(N_IC, 1)
u_ic = -torch.sin(np.pi * x_ic)

# --- Boundary conditions: u(-1, t) = u(1, t) = 0 ---
t_bc = torch.linspace(T_MIN, T_MAX, N_BC // 2).reshape(-1, 1)
x_bc_left = torch.full_like(t_bc, X_MIN)
x_bc_right = torch.full_like(t_bc, X_MAX)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
t_bc_all = torch.cat([t_bc, t_bc], dim=0)
u_bc = torch.zeros_like(x_bc)

# --- Collocation points (random in domain) ---
x_f = torch.rand(N_F, 1) * (X_MAX - X_MIN) + X_MIN
t_f = torch.rand(N_F, 1) * (T_MAX - T_MIN) + T_MIN

# Move to device
x_ic = x_ic.to(device)
t_ic = t_ic.to(device)
u_ic = u_ic.to(device)
x_bc = x_bc.to(device)
t_bc_all = t_bc_all.to(device)
u_bc = u_bc.to(device)
x_f = x_f.to(device)
t_f = t_f.to(device)

print(f"\nTraining data:")
print(f"  Initial condition points: {N_IC}")
print(f"  Boundary condition points: {N_BC}")
print(f"  Collocation points: {N_F}")
print(f"  Viscosity (nu/pi): {NU:.6f}")

# ============================================================
# [4] Model and Optimizer
# ============================================================
model = PINN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")

# Use Adam with learning rate scheduling
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)

# ============================================================
# [5] Loss Functions
# ============================================================
def pde_residual(model, x, t):
    """
    Compute PDE residual: u_t + u * u_x - (nu/pi) * u_xx = 0
    
    Uses autograd for automatic differentiation.
    """
    xt = torch.cat([x, t], dim=1).requires_grad_(True)
    u = model(xt)
    
    # First derivatives
    u_x = torch.autograd.grad(
        u, xt, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    
    u_t = torch.autograd.grad(
        u, xt, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]
    
    # Second derivative
    u_xx = torch.autograd.grad(
        u_x, xt, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    
    # Burgers equation: u_t + u * u_x = (nu/pi) * u_xx
    residual = u_t + u * u_x - NU * u_xx
    return residual


def compute_loss(model):
    """Total loss = IC loss + BC loss + PDE loss"""
    # Initial condition loss
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    pred_ic = model(xt_ic)
    loss_ic = torch.mean((pred_ic - u_ic) ** 2)
    
    # Boundary condition loss
    xt_bc = torch.cat([x_bc, t_bc_all], dim=1)
    pred_bc = model(xt_bc)
    loss_bc = torch.mean((pred_bc - u_bc) ** 2)
    
    # PDE residual loss
    residual = pde_residual(model, x_f, t_f)
    loss_pde = torch.mean(residual ** 2)
    
    # Total loss with weights
    loss = loss_ic + loss_bc + loss_pde
    
    return loss, loss_ic, loss_bc, loss_pde


# ============================================================
# [6] Training Loop
# ============================================================
EPOCHS = 5000
print(f"\nStarting training ({EPOCHS} epochs)...")
print("-" * 70)

loss_history = []
start_time = time.time()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    loss, loss_ic, loss_bc, loss_pde = compute_loss(model)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if epoch % 500 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:5d}/{EPOCHS} | "
              f"Loss: {loss.item():.6e} | "
              f"IC: {loss_ic.item():.4e} | "
              f"BC: {loss_bc.item():.4e} | "
              f"PDE: {loss_pde.item():.4e} | "
              f"Time: {elapsed:.1f}s")

total_time = time.time() - start_time
print("-" * 70)
print(f"Training complete! Total time: {total_time:.1f}s")
print(f"Final loss: {loss_history[-1]:.6e}")

# ============================================================
# [7] Visualization
# ============================================================
print("\nGenerating visualizations...")

# Create fine grid for plotting
N_PLOT = 256
x_plot = np.linspace(X_MIN, X_MAX, N_PLOT)
t_plot = np.linspace(T_MIN, T_MAX, N_PLOT)
X, T = np.meshgrid(x_plot, t_plot)

# Predict on grid
XT = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)
with torch.no_grad():
    U_pred = model(XT).cpu().numpy().reshape(N_PLOT, N_PLOT)

# --- Figure 1: Solution surface + loss curve ---
fig = plt.figure(figsize=(18, 5))

# 3D surface plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, T, U_pred, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('u(x, t)')
ax1.set_title('PINN Solution: Burgers Equation')

# Contour plot
ax2 = fig.add_subplot(132)
cf = ax2.contourf(X, T, U_pred, levels=50, cmap='viridis')
plt.colorbar(cf, ax=ax2)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_title('Solution Contour')

# Loss curve
ax3 = fig.add_subplot(133)
ax3.semilogy(loss_history)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (log scale)')
ax3.set_title('Training Loss')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
loss_path = os.path.join(RESULTS_DIR, "burgers_loss.png")
plt.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {loss_path}")

# --- Figure 2: Time snapshots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
snapshots = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

for idx, t_val in enumerate(snapshots):
    ax = axes[idx // 3, idx % 3]
    
    # PINN prediction at time t_val
    x_line = torch.linspace(X_MIN, X_MAX, N_PLOT).reshape(-1, 1).to(device)
    t_line = torch.full((N_PLOT, 1), t_val).to(device)
    xt_line = torch.cat([x_line, t_line], dim=1)
    
    with torch.no_grad():
        u_line = model(xt_line).cpu().numpy().flatten()
    
    ax.plot(x_plot, u_line, 'b-', linewidth=2, label='PINN')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(f't = {t_val:.1f}')
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Mark initial condition at t=0
    if t_val == 0.0:
        ax.plot(x_plot, -np.sin(np.pi * x_plot), 'r--', linewidth=1, alpha=0.5, label='Exact IC')
        ax.legend()

plt.suptitle('Burgers Equation: PINN Solution at Different Times', fontsize=14)
plt.tight_layout()
result_path = os.path.join(RESULTS_DIR, "burgers_result.png")
plt.savefig(result_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {result_path}")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Burgers Equation PINN")
print("=" * 70)
print(f"  Equation:       u_t + u * u_x = (nu/pi) * u_xx")
print(f"  Viscosity:      nu/pi = {NU:.6f}")
print(f"  Network:        2 -> 64 -> 64 -> 64 -> 64 -> 1 (Tanh)")
print(f"  Parameters:     {n_params:,}")
print(f"  Epochs:         {EPOCHS}")
print(f"  Training time:  {total_time:.1f}s")
print(f"  Final loss:     {loss_history[-1]:.6e}")
print(f"  Results:        {RESULTS_DIR}")
print()
print("Key observations:")
print("  - The shock wave forms near x=0 and sharpens over time")
print("  - PINN captures the shock but may smooth it (viscosity effect)")
print("  - This is the classic PINN benchmark from Raissi et al. (2019)")
print("=" * 70)
