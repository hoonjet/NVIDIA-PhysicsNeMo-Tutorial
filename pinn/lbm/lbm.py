"""
PhysicsNeMo PINN Tutorial: Lattice Boltzmann Method (LBM)
=========================================================
2D Lid-Driven Cavity via Lattice Boltzmann Equation (D2Q9)

Existing LDC PINN tutorial:
  - Solves Navier-Stokes:  u_t + (u·∇)u = -∇p/ρ + ν∇²u
  - Variables: macroscopic (u, v, p)
  - Eulerian PDE

THIS tutorial:
  - Solves Boltzmann Bhatnagar-Gross-Krook (BGK) equation:
      ∂f_i/∂t + c_i · ∇f_i = -(1/τ)(f_i - f_i^eq)
  - Variables: distribution functions f_i (i=0..8, D2Q9 lattice)
  - Mesoscopic / kinetic theory approach
  - Macroscopic (u, v, ρ) recovered via moments of f_i

Key difference from existing LDC PINN:
  ┌──────────────────────┬──────────────────────┐
  │ Existing LDC PINN     │ THIS (LBM PINN)      │
  ├──────────────────────┼──────────────────────┤
  │ Navier-Stokes (macro) │ Boltzmann BGK (meso) │
  │ Variables: u, v, p    │ Variables: f_0..f_8   │
  │ Continuum assumption  │ Kinetic theory       │
  │ Single PDE system     │ 9 coupled advection- │
  │                       │ relaxation equations  │
  └──────────────────────┴──────────────────────┘

D2Q9 Lattice:
  - 9 discrete velocities c_i = [(0,0), (1,0), (0,1), (-1,0), (0,-1),
    (1,1), (-1,1), (-1,-1), (1,-1)]
  - Equilibrium: f_i^eq = w_i * ρ * (1 + 3(c_i·u) + 4.5(c_i·u)² - 1.5|u|²)
  - Relaxation: τ = 3ν + 0.5 (relaxation time ↔ viscosity)

Author: PhysicsNeMo Tutorial
Date: 2026-09-01
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
print("PhysicsNeMo PINN Tutorial: Lattice Boltzmann Method (D2Q9)")
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
# [1] D2Q9 Lattice Constants
# ============================================================
# Discrete velocity set (D2Q9)
# Index:  0    1    2    3    4    5    6    7    8
# Dir:  rest  +x   +y   -x   -y  +x+y -x+y -x-y +x-y
CX = torch.tensor([0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=torch.float32, device=device)
CY = torch.tensor([0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=torch.float32, device=device)

# Weight coefficients
W = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
                 dtype=torch.float32, device=device)

# Opposite directions (for bounce-back BC)
OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long, device=device)

# Physical parameters
NU = 0.05          # Kinematic viscosity
TAU = 3.0 * NU + 0.5  # Relaxation time (τ = 3ν + 0.5 in lattice units)
U_LID = 0.1        # Lid velocity (lattice units)
OMEGA = 1.0 / TAU   # Relaxation rate ω = 1/τ

print(f"\n[1] D2Q9 Lattice Setup:")
print(f"  Viscosity ν = {NU}")
print(f"  Relaxation time τ = {TAU:.4f}")
print(f"  Relaxation rate ω = {OMEGA:.4f}")
print(f"  Lid velocity U = {U_LID}")
print(f"  Number of distribution functions: 9 (D2Q9)")

# ============================================================
# [2] Reference LBM Solver (FDM-style, for ground truth)
# ============================================================
def lbm_solver_reference(nx=32, ny=32, n_steps=500, u_lid=0.1, nu=0.05):
    """
    Standard LBM solver for 2D Lid-Driven Cavity.
    Used to generate reference data for PINN training/validation.
    """
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau

    # Initialize distribution functions (equilibrium with ρ=1, u=0)
    f = np.ones((nx, ny, 9), dtype=np.float32)
    for i in range(9):
        f[:, :, i] = W[i].cpu().numpy()

    # Macroscopic fields
    rho = np.ones((nx, ny), dtype=np.float32)
    ux = np.zeros((nx, ny), dtype=np.float32)
    uy = np.zeros((nx, ny), dtype=np.float32)

    cx = CX.cpu().numpy()
    cy = CY.cpu().numpy()

    for step in range(n_steps):
        # Compute macroscopic moments
        rho = f.sum(axis=2)
        ux = (f * cx).sum(axis=2) / rho
        uy = (f * cy).sum(axis=2) / rho

        # Bounce-back on walls (no-slip)
        # Left wall
        f[0, :, 1] = f[0, :, 3]  # +x gets -x
        f[0, :, 5] = f[0, :, 7]  # +x+y gets -x-y
        f[0, :, 8] = f[0, :, 6]  # +x-y gets -x+y
        # Right wall
        f[-1, :, 3] = f[-1, :, 1]
        f[-1, :, 7] = f[-1, :, 5]
        f[-1, :, 6] = f[-1, :, 8]
        # Bottom wall
        f[:, 0, 2] = f[:, 0, 4]
        f[:, 0, 5] = f[:, 0, 7]
        f[:, 0, 6] = f[:, 0, 8]
        # Top wall (moving lid: u = u_lid)
        f[:, -1, 4] = f[:, -1, 2]
        f[:, -1, 7] = f[:, -1, 5]
        f[:, -1, 8] = f[:, -1, 6]
        # Lid velocity correction (Zou-He boundary)
        rho_top = f[:, -1, 0] + f[:, -1, 1] + f[:, -1, 3] + \
                  2 * (f[:, -1, 5] + f[:, -1, 6] + f[:, -1, 7])
        # Simplified: just set lid velocity
        ux[:, -1] = u_lid

        # Compute equilibrium
        u_sq = ux**2 + uy**2
        feq = np.zeros_like(f)
        for i in range(9):
            cu = cx[i] * ux + cy[i] * uy
            feq[:, :, i] = W[i].cpu().numpy() * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)

        # Collision (BGK)
        f = f - omega * (f - feq)

        # Streaming (shift in velocity directions)
        f_new = np.zeros_like(f)
        for i in range(9):
            f_new[:, :, i] = np.roll(f[:, :, i], shift=(int(cx[i]), int(cy[i])), axis=(0, 1))
        f = f_new

    # Final macroscopic fields
    rho = f.sum(axis=2)
    ux = (f * cx).sum(axis=2) / rho
    uy = (f * cy).sum(axis=2) / rho

    return rho, ux, uy


print("\n[2] Generating reference LBM solution...")
t0 = time.time()
NX_REF, NY_REF = 32, 32
rho_ref, ux_ref, uy_ref = lbm_solver_reference(nx=NX_REF, ny=NY_REF, n_steps=500,
                                               u_lid=U_LID, nu=NU)
print(f"  Reference grid: {NX_REF}×{NY_REF}")
print(f"  Reference generation time: {time.time()-t0:.2f}s")
print(f"  ρ range: [{rho_ref.min():.4f}, {rho_ref.max():.4f}]")
print(f"  ux range: [{ux_ref.min():.4f}, {ux_ref.max():.4f}]")
print(f"  uy range: [{uy_ref.min():.4f}, {uy_ref.max():.4f}]")

# ============================================================
# [3] PINN Model for Distribution Functions
# ============================================================
class LBMPINN(nn.Module):
    """
    PINN that predicts 9 distribution functions f_i(x, y) for D2Q9.

    Input:  (x, y) — spatial coordinates
    Output: (f_0, f_1, ..., f_8) — 9 distribution functions

    The network learns the steady-state distribution that satisfies:
      1. Boltzmann BGK equation: c_i · ∇f_i = -(1/τ)(f_i - f_i^eq)
      2. Bounce-back BC on walls (no-slip)
      3. Lid velocity BC on top

    Architecture: 2 → 64 → 64 → 64 → 64 → 9
    """
    def __init__(self, layers=[2, 64, 64, 64, 64, 9]):
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
        # x: [N, 2] — (x_coord, y_coord)
        for i in range(len(self.layers) - 2):
            x = self.activation(self.linears[i](x))
        x = self.linears[-1](x)  # No activation on output (f can be any real)
        return x  # [N, 9]


def compute_equilibrium(f, rho, ux, uy):
    """
    Compute equilibrium distribution f_i^eq from macroscopic moments.

    f_i^eq = w_i * ρ * (1 + 3(c_i·u) + 4.5(c_i·u)² - 1.5|u|²)

    Args:
        f: [N, 9] distribution functions (used to compute moments)
    Returns:
        feq: [N, 9] equilibrium distributions
        rho: [N] density
        ux: [N] x-velocity
        uy: [N] y-velocity
    """
    # Macroscopic moments from f
    rho = f.sum(dim=1, keepdim=True)  # [N, 1]
    ux = (f * CX.unsqueeze(0)).sum(dim=1, keepdim=True) / rho  # [N, 1]
    uy = (f * CY.unsqueeze(0)).sum(dim=1, keepdim=True) / rho  # [N, 1]

    # Equilibrium
    u_sq = ux**2 + uy**2  # [N, 1]
    cu = CX.unsqueeze(0) * ux + CY.unsqueeze(0) * uy  # [N, 9]
    feq = W.unsqueeze(0) * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)  # [N, 9]

    return feq, rho, ux, uy


print(f"\n[3] PINN Model: LBMPINN")
print(f"  Architecture: 2 → 64 → 64 → 64 → 64 → 9")
print(f"  Input: (x, y)")
print(f"  Output: f_0, f_1, ..., f_8 (9 distribution functions)")

# ============================================================
# [4] Collocation Points & Boundary Conditions
# ============================================================
# Domain: [0, 1] × [0, 1]
N_INTERIOR = 2000
N_BOUNDARY = 200  # per wall

# Interior collocation points
x_interior = torch.rand(N_INTERIOR, 1, device=device)
y_interior = torch.rand(N_INTERIOR, 1, device=device)
xy_interior = torch.cat([x_interior, y_interior], dim=1)  # [N, 2]
xy_interior.requires_grad_(True)

# Boundary points
# Bottom wall (y=0, no-slip: u=v=0)
x_bottom = torch.rand(N_BOUNDARY, 1, device=device)
y_bottom = torch.zeros(N_BOUNDARY, 1, device=device)
xy_bottom = torch.cat([x_bottom, y_bottom], dim=1)

# Top wall (y=1, moving lid: u=U_LID, v=0)
x_top = torch.rand(N_BOUNDARY, 1, device=device)
y_top = torch.ones(N_BOUNDARY, 1, device=device)
xy_top = torch.cat([x_top, y_top], dim=1)

# Left wall (x=0, no-slip: u=v=0)
y_left = torch.rand(N_BOUNDARY, 1, device=device)
x_left = torch.zeros(N_BOUNDARY, 1, device=device)
xy_left = torch.cat([x_left, y_left], dim=1)

# Right wall (x=1, no-slip: u=v=0)
y_right = torch.rand(N_BOUNDARY, 1, device=device)
x_right = torch.ones(N_BOUNDARY, 1, device=device)
xy_right = torch.cat([x_right, y_right], dim=1)

# All boundary points
xy_bc = torch.cat([xy_bottom, xy_top, xy_left, xy_right], dim=0)  # [4*N_BOUNDARY, 2]

# Reference data points (from LBM solver)
x_ref = torch.linspace(0, 1, NX_REF, device=device).view(-1, 1)
y_ref = torch.linspace(0, 1, NY_REF, device=device).view(-1, 1)
X_ref, Y_ref = torch.meshgrid(x_ref.squeeze(), y_ref.squeeze(), indexing='ij')
xy_ref = torch.cat([X_ref.reshape(-1, 1), Y_ref.reshape(-1, 1)], dim=1)

# Reference macroscopic fields
ux_ref_t = torch.tensor(ux_ref.flatten(), dtype=torch.float32, device=device)
uy_ref_t = torch.tensor(uy_ref.flatten(), dtype=torch.float32, device=device)
rho_ref_t = torch.tensor(rho_ref.flatten(), dtype=torch.float32, device=device)

print(f"\n[4] Collocation Points:")
print(f"  Interior: {N_INTERIOR}")
print(f"  Boundary: {4*N_BOUNDARY} (4 walls × {N_BOUNDARY})")
print(f"  Reference data: {NX_REF*NY_REF}")

# ============================================================
# [5] Loss Functions
# ============================================================
def compute_pde_residual(model, xy):
    """
    Boltzmann BGK equation residual (steady-state):
      c_i · ∇f_i + (1/τ)(f_i - f_i^eq) = 0

    For each i: c_ix * df_i/dx + c_iy * df_i/dy = -(1/τ)(f_i - f_i^eq)
    """
    f = model(xy)  # [N, 9]
    feq, rho, ux, uy = compute_equilibrium(f, None, None, None)

    residual = torch.zeros_like(f)
    for i in range(9):
        # Gradient of f_i w.r.t. (x, y)
        df_i = torch.autograd.grad(
            f[:, i], xy,
            grad_outputs=torch.ones_like(f[:, i]),
            create_graph=True,
            retain_graph=True
        )[0]  # [N, 2]

        # c_i · ∇f_i
        c_dot_grad_f = CX[i] * df_i[:, 0] + CY[i] * df_i[:, 1]

        # BGK: c_i · ∇f_i = -(1/τ)(f_i - f_i^eq)
        residual[:, i] = c_dot_grad_f + OMEGA * (f[:, i] - feq[:, i])

    return residual


def compute_bc_loss(model, xy_bc, xy_top):
    """
    Boundary conditions:
    - No-slip walls (bottom, left, right): u = v = 0
    - Moving lid (top): u = U_LID, v = 0

    Implemented via macroscopic velocity constraints:
    At walls, the macroscopic velocity must match BC values.
    """
    # All boundary points
    f_bc = model(xy_bc)  # [N_bc, 9]
    _, _, ux_bc, uy_bc = compute_equilibrium(f_bc, None, None, None)

    # Bottom, left, right walls: u = v = 0
    n_per_wall = N_BOUNDARY
    ux_bottom = ux_bc[:n_per_wall]
    uy_bottom = uy_bc[:n_per_wall]
    ux_left = ux_bc[2*n_per_wall:3*n_per_wall]
    uy_left = uy_bc[2*n_per_wall:3*n_per_wall]
    ux_right = ux_bc[3*n_per_wall:4*n_per_wall]
    uy_right = uy_bc[3*n_per_wall:4*n_per_wall]

    bc_no_slip = (ux_bottom**2 + uy_bottom**2 +
                  ux_left**2 + uy_left**2 +
                  ux_right**2 + uy_right**2).mean()

    # Top wall: u = U_LID, v = 0
    f_top = model(xy_top)
    _, _, ux_top, uy_top = compute_equilibrium(f_top, None, None, None)
    bc_lid = ((ux_top - U_LID)**2 + uy_top**2).mean()

    return bc_no_slip + bc_lid


def compute_data_loss(model, xy_ref, ux_ref, uy_ref, rho_ref):
    """Data loss against LBM reference solution (macroscopic fields)."""
    f = model(xy_ref)
    feq, rho_pred, ux_pred, uy_pred = compute_equilibrium(f, None, None, None)

    loss_ux = ((ux_pred.squeeze() - ux_ref)**2).mean()
    loss_uy = ((uy_pred.squeeze() - uy_ref)**2).mean()
    loss_rho = ((rho_pred.squeeze() - rho_ref)**2).mean()

    return loss_ux + loss_uy + 0.1 * loss_rho


print(f"\n[5] Loss Functions:")
print(f"  PDE: Boltzmann BGK residual (9 equations)")
print(f"  BC:  No-slip walls + moving lid")
print(f"  Data: LBM reference (macroscopic u, v, ρ)")

# ============================================================
# [6] Training
# ============================================================
model = LBMPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

N_EPOCHS = 5000
LAMBDA_PDE = 1.0
LAMBDA_BC = 10.0
LAMBDA_DATA = 5.0

print(f"\n[6] Training:")
print(f"  Epochs: {N_EPOCHS}")
print(f"  λ_PDE={LAMBDA_PDE}, λ_BC={LAMBDA_BC}, λ_DATA={LAMBDA_DATA}")
print(f"  Optimizer: Adam (lr=1e-3, step decay @2000)")

loss_history = {'total': [], 'pde': [], 'bc': [], 'data': []}
t_start = time.time()

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()

    # PDE loss
    residual = compute_pde_residual(model, xy_interior)
    loss_pde = (residual**2).mean()

    # BC loss
    loss_bc = compute_bc_loss(model, xy_bc, xy_top)

    # Data loss
    loss_data = compute_data_loss(model, xy_ref, ux_ref_t, uy_ref_t, rho_ref_t)

    # Total loss
    loss = LAMBDA_PDE * loss_pde + LAMBDA_BC * loss_bc + LAMBDA_DATA * loss_data

    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history['total'].append(loss.item())
    loss_history['pde'].append(loss_pde.item())
    loss_history['bc'].append(loss_bc.item())
    loss_history['data'].append(loss_data.item())

    if (epoch + 1) % 500 == 0:
        elapsed = time.time() - t_start
        print(f"  Epoch {epoch+1:5d}/{N_EPOCHS} | "
              f"Total: {loss.item():.6e} | "
              f"PDE: {loss_pde.item():.6e} | "
              f"BC: {loss_bc.item():.6e} | "
              f"Data: {loss_data.item():.6e} | "
              f"Time: {elapsed:.1f}s")

print(f"\n  Training complete! Total time: {time.time()-t_start:.1f}s")

# ============================================================
# [7] Evaluation & Visualization
# ============================================================
print(f"\n[7] Evaluation & Visualization...")

# Predict on fine grid
NX_EVAL, NY_EVAL = 50, 50
x_eval = torch.linspace(0, 1, NX_EVAL, device=device).view(-1, 1)
y_eval = torch.linspace(0, 1, NY_EVAL, device=device).view(-1, 1)
X_eval, Y_eval = torch.meshgrid(x_eval.squeeze(), y_eval.squeeze(), indexing='ij')
xy_eval = torch.cat([X_eval.reshape(-1, 1), Y_eval.reshape(-1, 1)], dim=1)

model.eval()
with torch.no_grad():
    f_pred = model(xy_eval)
    feq_pred, rho_pred, ux_pred, uy_pred = compute_equilibrium(f_pred, None, None, None)

rho_pred = rho_pred.cpu().numpy().reshape(NX_EVAL, NY_EVAL)
ux_pred = ux_pred.cpu().numpy().reshape(NX_EVAL, NY_EVAL)
uy_pred = uy_pred.cpu().numpy().reshape(NX_EVAL, NY_EVAL)
speed_pred = np.sqrt(ux_pred**2 + uy_pred**2)

# --- Figure 1: Flow field comparison ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Reference
ax = axes[0, 0]
im = ax.imshow(ux_ref.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r',
               vmin=-U_LID, vmax=U_LID)
ax.set_title('Reference ux (LBM Solver)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[0, 1]
im = ax.imshow(uy_ref.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
ax.set_title('Reference uy (LBM Solver)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[0, 2]
speed_ref = np.sqrt(ux_ref**2 + uy_ref**2)
im = ax.imshow(speed_ref.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
ax.set_title('Reference |u| (LBM Solver)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

# PINN prediction
ax = axes[1, 0]
im = ax.imshow(ux_pred.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r',
               vmin=-U_LID, vmax=U_LID)
ax.set_title('PINN ux (Boltzmann BGK)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[1, 1]
im = ax.imshow(uy_pred.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
ax.set_title('PINN uy (Boltzmann BGK)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[1, 2]
im = ax.imshow(speed_pred.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
ax.set_title('PINN |u| (Boltzmann BGK)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

plt.suptitle('LBM PINN: 2D Lid-Driven Cavity (Boltzmann BGK vs LBM Solver)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'lbm_flow_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: lbm_flow_comparison.png")

# --- Figure 2: Streamlines ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
x_plot = np.linspace(0, 1, NX_REF)
y_plot = np.linspace(0, 1, NY_REF)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
ax.streamplot(X_plot, Y_plot, ux_ref.T, uy_ref.T, density=1.5, color='blue')
ax.set_title('Reference (LBM Solver)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax = axes[1]
x_plot2 = np.linspace(0, 1, NX_EVAL)
y_plot2 = np.linspace(0, 1, NY_EVAL)
X_plot2, Y_plot2 = np.meshgrid(x_plot2, y_plot2)
ax.streamplot(X_plot2, Y_plot2, ux_pred.T, uy_pred.T, density=1.5, color='red')
ax.set_title('PINN (Boltzmann BGK)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.suptitle('Streamlines: LBM Solver vs Boltzmann BGK PINN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'lbm_streamlines.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: lbm_streamlines.png")

# --- Figure 3: Loss history ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(loss_history['total'], label='Total', linewidth=2)
ax.semilogy(loss_history['pde'], label='PDE (BGK)', alpha=0.7)
ax.semilogy(loss_history['bc'], label='BC', alpha=0.7)
ax.semilogy(loss_history['data'], label='Data', alpha=0.7)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('LBM PINN Training Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'lbm_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: lbm_loss.png")

# --- Figure 4: D2Q9 lattice & concept ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# D2Q9 lattice diagram
ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
for i in range(9):
    cx_val = CX[i].item()
    cy_val = CY[i].item()
    ax.arrow(0, 0, cx_val*0.8, cy_val*0.8,
             head_width=0.08, head_length=0.05, fc='blue', ec='blue')
    ax.plot(cx_val, cy_val, 'ro', markersize=10)
    ax.annotate(f'f_{i}\n(w={W[i].item():.4f})',
                (cx_val*1.2, cy_val*1.2), fontsize=8, ha='center')
ax.set_title('D2Q9 Lattice Velocities', fontsize=13, fontweight='bold')
ax.set_xlabel('cx'); ax.set_ylabel('cy')
ax.grid(True, alpha=0.3)

# Concept comparison
ax = axes[1]
ax.axis('off')
concept_text = (
    "Boltzmann BGK Equation (Mesoscopic):\n\n"
    "  ∂f_i/∂t + c_i · ∇f_i = -(1/τ)(f_i - f_i^eq)\n\n"
    "  f_i  : distribution function (i=0..8)\n"
    "  c_i  : discrete velocity (D2Q9)\n"
    "  τ    : relaxation time (= 3ν + 0.5)\n"
    "  f_eq : equilibrium distribution\n\n"
    "Macroscopic recovery:\n"
    "  ρ = Σ f_i\n"
    "  u = (Σ c_i f_i) / ρ\n\n"
    "vs. Navier-Stokes (Macroscopic):\n\n"
    "  ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u\n\n"
    "  Variables: u, v, p (continuum)\n"
    "  No distribution functions"
)
ax.text(0.05, 0.95, concept_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Boltzmann BGK vs Navier-Stokes', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'lbm_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: lbm_concept.png")

# --- Figure 5: Distribution functions ---
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i in range(9):
    ax = axes[i // 3, i % 3]
    f_i = f_pred[:, i].cpu().numpy().reshape(NX_EVAL, NY_EVAL)
    im = ax.imshow(f_i.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax.set_title(f'f_{i} (c=({CX[i].item():.0f},{CY[i].item():.0f}), w={W[i].item():.4f})', fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, shrink=0.6)

plt.suptitle('D2Q9 Distribution Functions (PINN Prediction)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'lbm_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: lbm_distributions.png")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("[8] Summary")
print("=" * 70)

# Compute relative L2 error on reference grid
from scipy.interpolate import RegularGridInterpolator
interp_ux = RegularGridInterpolator((x_eval.squeeze().cpu().numpy(),
                                      y_eval.squeeze().cpu().numpy()),
                                     ux_pred, method='linear')
interp_uy = RegularGridInterpolator((x_eval.squeeze().cpu().numpy(),
                                      y_eval.squeeze().cpu().numpy()),
                                     uy_pred, method='linear')

pts = np.array([[x_ref[i].item(), y_ref[j].item()]
                for i in range(NX_REF) for j in range(NY_REF)])
ux_interp = interp_ux(pts).reshape(NX_REF, NY_REF)
uy_interp = interp_uy(pts).reshape(NX_REF, NY_REF)

rel_err_ux = np.linalg.norm(ux_interp - ux_ref) / (np.linalg.norm(ux_ref) + 1e-10)
rel_err_uy = np.linalg.norm(uy_interp - uy_ref) / (np.linalg.norm(uy_ref) + 1e-10)

print(f"\n  Method: Lattice Boltzmann Method (D2Q9) + PINN")
print(f"  PDE: Boltzmann BGK equation (mesoscopic/kinetic)")
print(f"  Variables: 9 distribution functions f_i(x,y)")
print(f"  Relaxation time τ = {TAU:.4f} (ν = {NU})")
print(f"")
print(f"  Relative L2 Error (vs LBM solver):")
print(f"    ux: {rel_err_ux:.4f} ({rel_err_ux*100:.2f}%)")
print(f"    uy: {rel_err_uy:.4f} ({rel_err_uy*100:.2f}%)")
print(f"")
print(f"  Key difference from existing LDC PINN:")
print(f"    Existing: Navier-Stokes (macroscopic u, v, p)")
print(f"    THIS:     Boltzmann BGK (mesoscopic f_0..f_8)")
print(f"")
print(f"  Results saved to: {RESULTS_DIR}")
print("=" * 70)
