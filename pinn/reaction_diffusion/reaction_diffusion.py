"""
PhysicsNeMo PINN Tutorial: Reaction-Diffusion (Gray-Scott Model)
=================================================================
Multi-variable coupled PDE system with nonlinear reaction terms.

All existing PINN tutorials in this repo solve SINGLE-VARIABLE equations
(Burgers, Poisson, Navier-Stokes, etc.). This tutorial is the ONLY one
that solves a MULTI-VARIABLE COUPLED PDE SYSTEM, where two variables
interact through nonlinear reaction terms to produce emergent Turing
patterns (spots, stripes, self-replicating structures).

Gray-Scott Model:
    u_t = D_u * (u_xx + u_yy) - u*v^2 + F*(1 - u)
    v_t = D_v * (v_xx + v_yy) + u*v^2 - (F + k)*v

where:
    u, v  = two interacting chemical concentrations
    D_u, D_v = diffusion coefficients
    F      = feed rate
    k      = kill rate

Key concepts:
    - Multi-output neural network (predicts both u and v simultaneously)
    - Coupled PDE residuals: each equation depends on BOTH variables
    - Nonlinear reaction terms (u*v^2) create pattern formation
    - Turing instability: diffusion-driven pattern emergence
    - Initial condition: localized perturbation that evolves into patterns

This is fundamentally different from existing PINN tutorials:
    - Burgers: 1 variable, 1D, advective nonlinearity
    - Poisson: 1 variable, 2D, linear elliptic
    - Navier-Stokes: coupled but via pressure (not reaction terms)
    - THIS: 2 variables, 2D+time, nonlinear coupling, pattern formation

Author: PhysicsNeMo Tutorial
Date: 2026-07-31
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
print("PhysicsNeMo PINN Tutorial: Reaction-Diffusion (Gray-Scott)")
print("Multi-Variable Coupled PDE with Turing Pattern Formation")
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
# [1] Problem Setup: Gray-Scott Model
# ============================================================
# Domain: [0, 1] x [0, 1], t in [0, T_MAX]
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 2.0

# Gray-Scott parameters (spots pattern)
DU = 2.0e-5    # Diffusion coefficient for u
DV = 1.0e-5    # Diffusion coefficient for v
F_FEED = 0.025  # Feed rate
K_KILL = 0.060  # Kill rate

print(f"\nGray-Scott parameters:")
print(f"  D_u = {DU}, D_v = {DV}")
print(f"  F (feed) = {F_FEED}, k (kill) = {K_KILL}")
print(f"  Domain: [{X_MIN},{X_MAX}] x [{Y_MIN},{Y_MAX}], t in [{T_MIN},{T_MAX}]")

# ============================================================
# [2] Reference Solution (Finite Difference)
# ============================================================
def gray_scott_fd(n_grid=64, n_steps=200, dt=0.01):
    """
    Compute reference Gray-Scott solution using finite difference.
    Uses explicit Euler with periodic boundary conditions.

    Initial condition: u=1 everywhere except a small square perturbation
    where u=0.5, v=0.25 (classic spot-generating IC).
    """
    dx = (X_MAX - X_MIN) / n_grid
    dy = (Y_MAX - Y_MIN) / n_grid

    # Initialize: u=1, v=0 everywhere
    u = np.ones((n_grid, n_grid), dtype=np.float64)
    v = np.zeros((n_grid, n_grid), dtype=np.float64)

    # Perturbation: small square in center
    r = n_grid // 8
    cx, cy = n_grid // 2, n_grid // 2
    u[cx-r:cx+r, cy-r:cy+r] = 0.50
    v[cx-r:cx+r, cy-r:cy+r] = 0.25

    # Add small noise
    u += np.random.randn(n_grid, n_grid) * 0.01
    v += np.random.randn(n_grid, n_grid) * 0.01
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    # Storage for snapshots
    save_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    snapshots_u = {0: u.copy()}
    snapshots_v = {0: v.copy()}

    for step in range(1, n_steps + 1):
        # Laplacian (periodic)
        u_xx = (np.roll(u, 1, axis=0) - 2 * u + np.roll(u, -1, axis=0)) / dx**2
        u_yy = (np.roll(u, 1, axis=1) - 2 * u + np.roll(u, -1, axis=1)) / dy**2
        v_xx = (np.roll(v, 1, axis=0) - 2 * v + np.roll(v, -1, axis=0)) / dx**2
        v_yy = (np.roll(v, 1, axis=1) - 2 * v + np.roll(v, -1, axis=1)) / dy**2

        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy

        # Reaction terms
        uvv = u * v * v

        # Gray-Scott equations
        u_new = u + dt * (DU * lap_u - uvv + F_FEED * (1 - u))
        v_new = v + dt * (DV * lap_v + uvv - (F_FEED + K_KILL) * v)

        u = np.clip(u_new, 0, 1)
        v = np.clip(v_new, 0, 1)

        if step in save_steps:
            snapshots_u[step] = u.copy()
            snapshots_v[step] = v.copy()

    return snapshots_u, snapshots_v, save_steps


print("\n[1] Computing reference solution (finite difference)...")
N_GRID = 64
N_STEPS = 200
DT_FD = (T_MAX - T_MIN) / N_STEPS

ref_u, ref_v, save_steps = gray_scott_fd(N_GRID, N_STEPS, DT_FD)
print(f"  Grid: {N_GRID}x{N_GRID}, Steps: {N_STEPS}, dt: {DT_FD:.4f}")
print(f"  Snapshots at steps: {save_steps}")

# ============================================================
# [3] PINN Model (Multi-Output)
# ============================================================
class ReactionDiffusionPINN(nn.Module):
    """
    Multi-output fully connected network for Gray-Scott model.
    Input: (x, y, t) -> Output: (u, v)

    Architecture: 3 -> 64 -> 64 -> 64 -> 64 -> 2
    The network outputs BOTH u and v simultaneously.
    """
    def __init__(self, layers=[3, 64, 64, 64, 64, 2]):
        super().__init__()
        self.layers = layers
        self.activation = nn.Tanh()
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
        return self.linears[-1](x)  # [batch, 2] -> (u, v)


# ============================================================
# [4] PDE Residuals (Coupled System)
# ============================================================
def compute_residuals(model, x, y, t):
    """
    Compute coupled PDE residuals for Gray-Scott model.

    Returns:
        res_u: u_t - D_u*(u_xx+u_yy) + u*v^2 - F*(1-u)
        res_v: v_t - D_v*(v_xx+v_yy) - u*v^2 + (F+k)*v
    """
    xyt = torch.cat([x, y, t], dim=1).requires_grad_(True)
    uv = model(xyt)  # [N, 2]
    u = uv[:, 0:1]
    v = uv[:, 1:2]

    # First derivatives (need u_t, v_t)
    grad_u = torch.autograd.grad(
        u, xyt, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]

    grad_v = torch.autograd.grad(
        v, xyt, grad_outputs=torch.ones_like(v),
        create_graph=True, retain_graph=True
    )[0]
    v_x = grad_v[:, 0:1]
    v_y = grad_v[:, 1:2]
    v_t = grad_v[:, 2:3]

    # Second derivatives
    u_xx = torch.autograd.grad(
        u_x, xyt, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, xyt, grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]

    v_xx = torch.autograd.grad(
        v_x, xyt, grad_outputs=torch.ones_like(v_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    v_yy = torch.autograd.grad(
        v_y, xyt, grad_outputs=torch.ones_like(v_y),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]

    # Laplacians
    lap_u = u_xx + u_yy
    lap_v = v_xx + v_yy

    # Reaction term (nonlinear coupling)
    uvv = u * v * v

    # Residuals
    res_u = u_t - DU * lap_u + uvv - F_FEED * (1 - u)
    res_v = v_t - DV * lap_v - uvv + (F_FEED + K_KILL) * v

    return res_u, res_v


# ============================================================
# [5] Training Data
# ============================================================
print("\n[2] Preparing training data...")

# --- Initial condition: t=0 ---
# u=1 everywhere except center square, v=0 except center
N_IC = 2000
x_ic = torch.rand(N_IC, 1) * (X_MAX - X_MIN) + X_MIN
y_ic = torch.rand(N_IC, 1) * (Y_MAX - Y_MIN) + Y_MIN
t_ic = torch.zeros(N_IC, 1)

# Center square perturbation
SQ_X_MIN, SQ_X_MAX = 0.375, 0.625  # 1/4 to 3/8... center square
SQ_Y_MIN, SQ_Y_MAX = 0.375, 0.625

u_ic = torch.ones(N_IC, 1)
v_ic = torch.zeros(N_IC, 1)
in_square = (x_ic >= SQ_X_MIN) & (x_ic <= SQ_X_MAX) & (y_ic >= SQ_Y_MIN) & (y_ic <= SQ_Y_MAX)
u_ic[in_square] = 0.5
v_ic[in_square] = 0.25
# Add small noise
u_ic += torch.randn_like(u_ic) * 0.01
v_ic += torch.randn_like(v_ic) * 0.01

# --- Boundary conditions: periodic (u, v continuous at boundaries) ---
# For simplicity, enforce Dirichlet: u=1, v=0 on domain boundary
# (This is an approximation; the FD reference uses periodic, but the
# perturbation is in the center far from boundaries, so boundary effect is minimal)
N_BC = 1000
# Sample on boundary
n_per_side = N_BC // 4
x_bc = torch.cat([
    torch.rand(n_per_side, 1) * (X_MAX - X_MIN),
    torch.rand(n_per_side, 1) * (X_MAX - X_MIN),
    torch.zeros(n_per_side, 1),
    torch.ones(n_per_side, 1),
])
y_bc = torch.cat([
    torch.zeros(n_per_side, 1),
    torch.ones(n_per_side, 1),
    torch.rand(n_per_side, 1) * (Y_MAX - Y_MIN),
    torch.rand(n_per_side, 1) * (Y_MAX - Y_MIN),
])
t_bc = torch.rand(N_BC, 1) * (T_MAX - T_MIN)

u_bc = torch.ones(N_BC, 1)
v_bc = torch.zeros(N_BC, 1)

# --- Collocation points (PDE residual) ---
N_F = 15000
x_f = torch.rand(N_F, 1) * (X_MAX - X_MIN) + X_MIN
y_f = torch.rand(N_F, 1) * (Y_MAX - Y_MIN) + Y_MIN
t_f = torch.rand(N_F, 1) * (T_MAX - T_MIN) + T_MIN

# Move to device
x_ic, y_ic, t_ic = x_ic.to(device), y_ic.to(device), t_ic.to(device)
u_ic, v_ic = u_ic.to(device), v_ic.to(device)
x_bc, y_bc, t_bc = x_bc.to(device), y_bc.to(device), t_bc.to(device)
u_bc, v_bc = u_bc.to(device), v_bc.to(device)
x_f, y_f, t_f = x_f.to(device), y_f.to(device), t_f.to(device)

print(f"  IC points: {N_IC}")
print(f"  BC points: {N_BC}")
print(f"  Collocation points: {N_F}")

# ============================================================
# [6] Model and Optimizer
# ============================================================
model = ReactionDiffusionPINN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")

# Loss weights (balance IC, BC, PDE)
W_IC = 1.0
W_BC = 1.0
W_PDE = 1.0

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

# ============================================================
# [7] Training Loop
# ============================================================
EPOCHS = 8000
print(f"\nStarting training ({EPOCHS} epochs)...")
print("-" * 70)

loss_history = []
loss_ic_history = []
loss_bc_history = []
loss_pde_history = []

start_time = time.time()

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # --- IC loss ---
    xyt_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
    uv_ic_pred = model(xyt_ic)
    u_ic_pred = uv_ic_pred[:, 0:1]
    v_ic_pred = uv_ic_pred[:, 1:2]
    loss_ic_u = torch.mean((u_ic_pred - u_ic) ** 2)
    loss_ic_v = torch.mean((v_ic_pred - v_ic) ** 2)
    loss_ic = loss_ic_u + loss_ic_v

    # --- BC loss ---
    xyt_bc = torch.cat([x_bc, y_bc, t_bc], dim=1)
    uv_bc_pred = model(xyt_bc)
    u_bc_pred = uv_bc_pred[:, 0:1]
    v_bc_pred = uv_bc_pred[:, 1:2]
    loss_bc_u = torch.mean((u_bc_pred - u_bc) ** 2)
    loss_bc_v = torch.mean((v_bc_pred - v_bc) ** 2)
    loss_bc = loss_bc_u + loss_bc_v

    # --- PDE residual loss (coupled) ---
    res_u, res_v = compute_residuals(model, x_f, y_f, t_f)
    loss_pde_u = torch.mean(res_u ** 2)
    loss_pde_v = torch.mean(res_v ** 2)
    loss_pde = loss_pde_u + loss_pde_v

    # Total loss
    loss = W_IC * loss_ic + W_BC * loss_bc + W_PDE * loss_pde
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())
    loss_ic_history.append(loss_ic.item())
    loss_bc_history.append(loss_bc.item())
    loss_pde_history.append(loss_pde.item())

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
# [8] Evaluation: Predict Pattern at Multiple Times
# ============================================================
print("\n[3] Evaluating pattern prediction...")

model.eval()
N_EVAL = 64
x_eval = np.linspace(X_MIN, X_MAX, N_EVAL)
y_eval = np.linspace(Y_MIN, Y_MAX, N_EVAL)
X_eval, Y_eval = np.meshgrid(x_eval, y_eval)

# Evaluate at several time steps
eval_times = [0.0, 0.5, 1.0, 1.5, 2.0]
pred_u_all = {}
pred_v_all = {}

for t_val in eval_times:
    xy_t = np.stack([
        X_eval.flatten(),
        Y_eval.flatten(),
        np.full(N_EVAL * N_EVAL, t_val)
    ], axis=1)
    xy_t_t = torch.from_numpy(xy_t).float().to(device)

    with torch.no_grad():
        uv_pred = model(xy_t_t).cpu().numpy()
    u_pred = uv_pred[:, 0].reshape(N_EVAL, N_EVAL)
    v_pred = uv_pred[:, 1].reshape(N_EVAL, N_EVAL)
    pred_u_all[t_val] = u_pred
    pred_v_all[t_val] = v_pred

# Compare with reference at t=T_MAX (step 200)
ref_u_final = ref_u[N_STEPS]
ref_v_final = ref_v[N_STEPS]
pred_u_final = pred_u_all[2.0]
pred_v_final = pred_v_all[2.0]

# Relative L2 errors
l2_u = np.linalg.norm(pred_u_final - ref_u_final) / (np.linalg.norm(ref_u_final) + 1e-8)
l2_v = np.linalg.norm(pred_v_final - ref_v_final) / (np.linalg.norm(ref_v_final) + 1e-8)
print(f"  L2 error (u) at t=2.0: {l2_u:.4f}")
print(f"  L2 error (v) at t=2.0: {l2_v:.4f}")

# ============================================================
# [9] Visualization
# ============================================================
print("\n[4] Generating visualizations...")

# --- Figure 1: Loss curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(loss_history, label='Total', linewidth=2)
ax.semilogy(loss_ic_history, label='IC', alpha=0.7)
ax.semilogy(loss_bc_history, label='BC', alpha=0.7)
ax.semilogy(loss_pde_history, label='PDE', alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Training Loss Breakdown')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(loss_pde_history, linewidth=2, color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('PDE Residual Loss (log scale)')
ax.set_title('PDE Residual Loss (Coupled System)')
ax.grid(True, alpha=0.3)

plt.suptitle('Reaction-Diffusion PINN: Loss Curves', fontsize=14)
plt.tight_layout()
loss_path = os.path.join(RESULTS_DIR, "reaction_diffusion_loss.png")
plt.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {loss_path}")

# --- Figure 2: Pattern evolution (u and v at different times) ---
fig, axes = plt.subplots(2, len(eval_times), figsize=(22, 9))

for idx, t_val in enumerate(eval_times):
    # u
    ax = axes[0, idx]
    im = ax.imshow(pred_u_all[t_val], extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                   origin='lower', cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'u(x,y,t={t_val:.1f})')
    ax.set_xlabel('x')
    if idx == 0:
        ax.set_ylabel('u concentration')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # v
    ax = axes[1, idx]
    im = ax.imshow(pred_v_all[t_val], extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                   origin='lower', cmap='plasma', vmin=0, vmax=0.5)
    ax.set_title(f'v(x,y,t={t_val:.1f})')
    ax.set_xlabel('x')
    if idx == 0:
        ax.set_ylabel('v concentration')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Turing Pattern Evolution: PINN Prediction', fontsize=14)
plt.tight_layout()
pattern_path = os.path.join(RESULTS_DIR, "reaction_diffusion_patterns.png")
plt.savefig(pattern_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {pattern_path}")

# --- Figure 3: Comparison with FD reference at t=2.0 ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# u comparison
ax = axes[0, 0]
im = ax.imshow(ref_u_final, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
               origin='lower', cmap='viridis', vmin=0, vmax=1)
ax.set_title(f'FD Reference: u (t=2.0)')
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 1]
im = ax.imshow(pred_u_final, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
               origin='lower', cmap='viridis', vmin=0, vmax=1)
ax.set_title(f'PINN Prediction: u (t=2.0)')
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 2]
err_u = np.abs(pred_u_final - ref_u_final)
im = ax.imshow(err_u, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
               origin='lower', cmap='hot')
ax.set_title(f'|Error|: u (L2={l2_u:.4f})')
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, fraction=0.046)

# v comparison
ax = axes[1, 0]
im = ax.imshow(ref_v_final, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
               origin='lower', cmap='plasma', vmin=0, vmax=0.5)
ax.set_title(f'FD Reference: v (t=2.0)')
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 1]
im = ax.imshow(pred_v_final, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
               origin='lower', cmap='plasma', vmin=0, vmax=0.5)
ax.set_title(f'PINN Prediction: v (t=2.0)')
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 2]
err_v = np.abs(pred_v_final - ref_v_final)
im = ax.imshow(err_v, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
               origin='lower', cmap='hot')
ax.set_title(f'|Error|: v (L2={l2_v:.4f})')
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Reaction-Diffusion: PINN vs FD Reference at t=2.0', fontsize=14)
plt.tight_layout()
compare_path = os.path.join(RESULTS_DIR, "reaction_diffusion_comparison.png")
plt.savefig(compare_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {compare_path}")

# --- Figure 4: Cross-sections through center ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
y_mid = N_EVAL // 2

# u cross-section
ax = axes[0]
ax.plot(x_eval, ref_u_final[y_mid, :], 'b-', linewidth=2, label='FD Reference')
ax.plot(x_eval, pred_u_final[y_mid, :], 'r--', linewidth=2, label='PINN')
ax.set_xlabel('x')
ax.set_ylabel('u(x, 0.5, 2.0)')
ax.set_title('u Cross-section at y=0.5, t=2.0')
ax.legend()
ax.grid(True, alpha=0.3)

# v cross-section
ax = axes[1]
ax.plot(x_eval, ref_v_final[y_mid, :], 'b-', linewidth=2, label='FD Reference')
ax.plot(x_eval, pred_v_final[y_mid, :], 'r--', linewidth=2, label='PINN')
ax.set_xlabel('x')
ax.set_ylabel('v(x, 0.5, 2.0)')
ax.set_title('v Cross-section at y=0.5, t=2.0')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Cross-section Comparison at y=0.5, t=2.0', fontsize=14)
plt.tight_layout()
cross_path = os.path.join(RESULTS_DIR, "reaction_diffusion_cross.png")
plt.savefig(cross_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {cross_path}")

# ============================================================
# [10] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: PINN for Reaction-Diffusion (Gray-Scott)")
print("=" * 70)
print(f"  Equation:       Gray-Scott Model (2 coupled variables)")
print(f"  u_t = D_u*Lap(u) - u*v^2 + F*(1-u)")
print(f"  v_t = D_v*Lap(v) + u*v^2 - (F+k)*v")
print(f"  Parameters:      D_u={DU}, D_v={DV}, F={F_FEED}, k={K_KILL}")
print(f"  Domain:          [{X_MIN},{X_MAX}]x[{Y_MIN},{Y_MAX}], t in [{T_MIN},{T_MAX}]")
print(f"  Network:         3 -> 64 -> 64 -> 64 -> 64 -> 2 (Tanh)")
print(f"  Outputs:         u(x,y,t) AND v(x,y,t) [MULTI-OUTPUT]")
print(f"  Parameters:      {n_params:,}")
print(f"  IC points:       {N_IC}")
print(f"  BC points:       {N_BC}")
print(f"  Collocation:     {N_F}")
print(f"  Epochs:          {EPOCHS}")
print(f"  Training time:   {total_time:.1f}s")
print(f"  Final loss:      {loss_history[-1]:.6e}")
print(f"  L2 error (u):    {l2_u:.4f}")
print(f"  L2 error (v):    {l2_v:.4f}")
print(f"  Results:         {RESULTS_DIR}")
print()
print("Key observations:")
print("  1. MULTI-VARIABLE: Network outputs BOTH u and v simultaneously")
print("  2. COUPLING: u equation has u*v^2 (depends on v); v equation too")
print("  3. NONLINEAR REACTION: u*v^2 term creates pattern formation")
print("  4. TURING INSTABILITY: Diffusion + reaction -> emergent patterns")
print("  5. This is the ONLY tutorial with coupled multi-variable PDEs")
print("  6. Applicable to: chemistry, biology, ecology, combustion")
print("=" * 70)
