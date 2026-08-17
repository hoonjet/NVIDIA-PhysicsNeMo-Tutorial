"""
PhysicsNeMo Tutorial: Inverse PINN — Parameter Discovery
=========================================================
This tutorial demonstrates INVERSE problem solving with PINN.

Previous PINN tutorials (LDC, CHT) solved FORWARD problems:
  - Given: PDE + parameters + BCs → Find: solution field

This tutorial solves an INVERSE problem:
  - Given: PDE + observation data → Find: unknown PDE parameter

Problem: Burgers' Equation with Unknown Viscosity
  PDE:  u_t + u * u_x = nu * u_xx
  - u(x,t): velocity field (what we solve for)
  - nu: kinematic viscosity (UNKNOWN — what we want to discover!)

  We generate "observation data" using the true viscosity (nu_true = 0.01/pi),
  then pretend we don't know nu and let the network learn it from the data.

Key difference from forward PINN:
  - Forward:  network parameters are trained, PDE params are fixed
  - Inverse:  network parameters AND PDE parameters are BOTH trained
              The PDE parameter (nu) becomes a learnable variable!

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_inverse_pinns.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import physicsnemo
from physicsnemo.models.mlp.fully_connected import FullyConnected
import time
import os

# ============================================================================
# [0] Environment Setup
# ============================================================================
print("=" * 70)
print("  PhysicsNeMo Tutorial: Inverse PINN (Parameter Discovery)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  PyTorch version:    {torch.__version__}")
print(f"  PhysicsNeMo version: {physicsnemo.__version__}")
print(f"  Device:             {device}")
if torch.cuda.is_available():
    print(f"  GPU:                {torch.cuda.get_device_name(0)}")
print("=" * 70)
print()

torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# [1] Problem Setup: Burgers' Equation with Unknown Viscosity
# ============================================================================
# Burgers' equation: u_t + u * u_x = nu * u_xx
#
# This is a fundamental PDE in fluid dynamics that models:
#   - Shock wave formation (when nu is small)
#   - Nonlinear advection + viscous diffusion
#
# The viscosity nu controls how "smooth" the solution is:
#   - Large nu:  smooth, diffusive (laminar flow)
#   - Small nu:  sharp gradients, shocks (turbulent flow)
#
# In this inverse problem:
#   - We have observation data (u at some points)
#   - We DON'T know nu
#   - We train the network to find BOTH u(x,t) AND nu

NU_TRUE = 0.01 / np.pi  # True viscosity (we pretend we don't know this)
# Domain: x in [-1, 1], t in [0, 1]
# IC: u(x, 0) = -sin(pi*x)  (standard Burgers IC)
# BC: u(-1, t) = u(1, t) = 0

print("[1/7] Problem: Inverse Burgers' Equation")
print(f"      PDE: u_t + u*u_x = nu * u_xx")
print(f"      True viscosity: nu_true = {NU_TRUE:.6f} (unknown to the network)")
print(f"      Goal: Discover nu from observation data")
print()


# ============================================================================
# [2] Generate Observation Data (Analytical Solution)
# ============================================================================
# We use the Cole-Hopf transformation to generate "exact" solutions.
# For small nu, the Burgers equation solution can be approximated.
# Here we use a simple numerical approximation for data generation.

print("[2/7] Generating observation data...")

def generate_burgers_data(n_obs=200, nu=NU_TRUE, device="cpu"):
    """
    Generate observation data for Burgers' equation.
    
    We use a simple finite-difference solver to generate "ground truth" data.
    In practice, this would come from experiments or high-fidelity simulations.
    """
    nx, nt = 100, 200
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Initialize: u(x, 0) = -sin(pi*x)
    u = -np.sin(np.pi * x)
    
    # Time-step using upwind + central difference
    u_history = np.zeros((nt, nx))
    u_history[0] = u
    
    for n in range(1, nt):
        u_new = u.copy()
        # Upwind for advection, central for diffusion
        for i in range(1, nx - 1):
            if u[i] > 0:
                adv = u[i] * (u[i] - u[i-1]) / dx
            else:
                adv = u[i] * (u[i+1] - u[i]) / dx
            diff = nu * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
            u_new[i] = u[i] + dt * (-adv + diff)
        # Boundary conditions
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new
        u_history[n] = u
    
    # Sample observation points (random subset)
    np.random.seed(123)
    obs_idx = np.random.choice(nt * nx, n_obs, replace=False)
    obs_t_idx = obs_idx // nx
    obs_x_idx = obs_idx % nx
    
    obs_x = x[obs_x_idx]
    obs_t = t[obs_t_idx]
    obs_u = u_history[obs_t_idx, obs_x_idx]
    
    # Convert to tensors
    obs_x = torch.tensor(obs_x, dtype=torch.float32, device=device).reshape(-1, 1)
    obs_t = torch.tensor(obs_t, dtype=torch.float32, device=device).reshape(-1, 1)
    obs_u = torch.tensor(obs_u, dtype=torch.float32, device=device).reshape(-1, 1)
    
    # Also return full grid for visualization
    X, T = np.meshgrid(x, t)
    U_full = u_history
    
    return obs_x, obs_t, obs_u, (X, T, U_full)

N_OBS = 300
obs_x, obs_t, obs_u, (X_grid, T_grid, U_full) = generate_burgers_data(
    N_OBS, NU_TRUE, device
)
print(f"      Observation points: {N_OBS}")
print(f"      Data range: u in [{obs_u.min():.3f}, {obs_u.max():.3f}]")
print()


# ============================================================================
# [3] Create Model + Learnable Viscosity Parameter
# ============================================================================
print("[3/7] Creating model (FullyConnected) + learnable viscosity nu...")

model = FullyConnected(
    in_features=2,        # Input: x, t
    out_features=1,       # Output: u (velocity)
    layer_size=50,        # Neurons per hidden layer
    num_layers=5,         # Number of hidden layers
    activation_fn="tanh", # Tanh (smooth, differentiable)
    weight_norm=True,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"      Network parameters: {n_params:,}")

# KEY: nu is a LEARNABLE PARAMETER (not a fixed constant!)
# We initialize it with a wrong guess and let the optimizer find the true value.
NU_INIT = 0.1  # Initial guess (deliberately wrong — true is 0.00318)
nu_param = torch.tensor([NU_INIT], dtype=torch.float32, device=device, requires_grad=True)
print(f"      Initial nu guess: {NU_INIT:.6f} (true: {NU_TRUE:.6f})")
print(f"      nu is a LEARNABLE PARAMETER (will be optimized!)")


# ============================================================================
# [4] Training Setup
# ============================================================================
print("\n[4/7] Setting up training...")

# Both network parameters AND nu are optimized together
optimizer = torch.optim.Adam(
    list(model.parameters()) + [nu_param], lr=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8000, eta_min=1e-5)
EPOCHS = 8000


# Generate collocation points for PDE residual
N_PDE = 5000
x_pde = (torch.rand(N_PDE, 1, device=device) * 2 - 1).requires_grad_(True)  # x in [-1, 1]
t_pde = (torch.rand(N_PDE, 1, device=device)).requires_grad_(True)           # t in [0, 1]

# IC points: t=0, u = -sin(pi*x)
N_IC = 200
x_ic = torch.linspace(-1, 1, N_IC, device=device).reshape(-1, 1)
t_ic = torch.zeros(N_IC, 1, device=device)
u_ic = -torch.sin(np.pi * x_ic)

# BC points: x=±1, u=0
N_BC = 100
t_bc = torch.rand(N_BC, 1, device=device)
x_bc_left = -torch.ones(N_BC, 1, device=device)
x_bc_right = torch.ones(N_BC, 1, device=device)


# ============================================================================
# [5] Training Loop
# ============================================================================
print(f"\n[5/7] Training Inverse PINN ({EPOCHS} epochs)...")
print(f"      Watching nu converge to true value {NU_TRUE:.6f}...")
print()

start_time = time.time()
loss_history = []
nu_history = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # --- PDE residual: u_t + u*u_x = nu * u_xx ---
    inputs = torch.cat([x_pde, t_pde], dim=1)
    u_pred = model(inputs)
    
    u_x = torch.autograd.grad(u_pred, x_pde, grad_outputs=torch.ones_like(u_pred),
                              create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u_pred, t_pde, grad_outputs=torch.ones_like(u_pred),
                              create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]
    
    # PDE residual (uses learnable nu_param!)
    res_pde = u_t + u_pred * u_x - nu_param * u_xx
    loss_pde = torch.mean(res_pde ** 2)
    
    # --- Observation data loss ---
    inputs_obs = torch.cat([obs_x, obs_t], dim=1)
    u_obs_pred = model(inputs_obs)
    loss_obs = torch.mean((u_obs_pred - obs_u) ** 2)
    
    # --- IC loss: u(x, 0) = -sin(pi*x) ---
    inputs_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(inputs_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)
    
    # --- BC loss: u(±1, t) = 0 ---
    inputs_bc_l = torch.cat([x_bc_left, t_bc], dim=1)
    inputs_bc_r = torch.cat([x_bc_right, t_bc], dim=1)
    loss_bc = torch.mean(model(inputs_bc_l) ** 2) + torch.mean(model(inputs_bc_r) ** 2)
    
    # Total loss
    loss = loss_pde + 10.0 * loss_obs + 5.0 * loss_ic + 5.0 * loss_bc
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())

    nu_history.append(nu_param.item())
    
    if epoch % 500 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        nu_err = abs(nu_param.item() - NU_TRUE) / NU_TRUE * 100
        print(f"  Epoch {epoch:5d}/{EPOCHS} | "
              f"Loss: {loss.item():.4e} | "
              f"nu: {nu_param.item():.6f} | "
              f"nu_err: {nu_err:.1f}% | "
              f"Time: {elapsed:.1f}s")

elapsed_total = time.time() - start_time
nu_final = nu_param.item()
nu_error = abs(nu_final - NU_TRUE) / NU_TRUE * 100
print(f"\n  Training complete! Time: {elapsed_total:.1f}s")
print(f"  Final loss: {loss.item():.6e}")
print(f"  Discovered nu: {nu_final:.6f} (true: {NU_TRUE:.6f}, error: {nu_error:.1f}%)")


# ============================================================================
# [6] Visualization
# ============================================================================
print("\n[6/7] Visualizing results...")

model.eval()
output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    # Predict full field
    x_line = torch.linspace(-1, 1, 100, device=device)
    t_line = torch.linspace(0, 1, 100, device=device)
    X_pred, T_pred = torch.meshgrid(x_line, t_line, indexing='xy')
    inputs = torch.cat([X_pred.reshape(-1, 1), T_pred.reshape(-1, 1)], dim=1)
    U_pred = model(inputs).reshape(100, 100).cpu().numpy()

# Create figure: 2x2
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig)

# 1. True solution (from FD solver)
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.pcolormesh(X_grid, T_grid, U_full, cmap='RdBu_r', shading='gouraud')
ax1.set_title('True u(x,t) (Finite Difference)', fontsize=12)
ax1.set_xlabel('x'); ax1.set_ylabel('t')
plt.colorbar(im1, ax=ax1, label='u')

# 2. PINN predicted solution
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.pcolormesh(X_pred.cpu().numpy(), T_pred.cpu().numpy(), U_pred, cmap='RdBu_r', shading='gouraud')
ax2.set_title('PINN u(x,t) (Inverse Solution)', fontsize=12)
ax2.set_xlabel('x'); ax2.set_ylabel('t')
plt.colorbar(im2, ax=ax2, label='u')

# 3. nu convergence history
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(nu_history, 'b-', linewidth=1.0, label='Predicted nu')
ax3.axhline(y=NU_TRUE, color='r', linestyle='--', linewidth=1.5, label=f'True nu={NU_TRUE:.6f}')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('nu (viscosity)')
ax3.set_title(f'Viscosity Discovery (final: {nu_final:.6f}, error: {nu_error:.1f}%)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, max(NU_INIT * 1.2, max(nu_history) * 1.2)])

# 4. Loss history
ax4 = fig.add_subplot(gs[1, 1])
ax4.semilogy(loss_history, 'g-', linewidth=0.5)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Total Loss (log)')
ax4.set_title('Training Loss History', fontsize=12)
ax4.grid(True, alpha=0.3)

plt.suptitle(
    f'Inverse PINN: Burgers\' Equation — Viscosity Discovery\n'
    f'PhysicsNeMo {physicsnemo.__version__} | {device} | '
    f'nu_true={NU_TRUE:.6f}, nu_pred={nu_final:.6f} ({nu_error:.1f}% error)',
    fontsize=13, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(output_dir, "inverse_pinn_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Result image saved: {fig_path}")
plt.close('all')


# ============================================================================
# [7] Summary
# ============================================================================
print("\n[7/7] Tutorial complete!")
print("=" * 70)
print("  Inverse PINN Tutorial Summary:")
print(f"    - Problem: Burgers' equation with unknown viscosity nu")
print(f"    - True viscosity:  nu = {NU_TRUE:.6f}")
print(f"    - Initial guess:   nu = {NU_INIT:.6f} (deliberately wrong)")
print(f"    - Discovered nu:   nu = {nu_final:.6f} (error: {nu_error:.1f}%)")
print(f"    - Observation points: {N_OBS}")
print(f"    - Network: FullyConnected (5 layers, 50 neurons, Tanh)")
print(f"    - Training: Adam, {EPOCHS} epochs, {elapsed_total:.1f}s")
print(f"    - Device: {device}")
print()
print("  Key concept — Forward vs Inverse PINN:")
print(f"    {'Aspect':<25} {'Forward PINN':<25} {'Inverse PINN':<25}")
print(f"    {'-'*25} {'-'*25} {'-'*25}")
print(f"    {'Known':<25} {'PDE + params + BCs':<25} {'PDE + obs data':<25}")
print(f"    {'Unknown':<25} {'Solution field':<25} {'Solution + PDE params':<25}")
print(f"    {'Learnable':<25} {'Network weights':<25} {'Network + PDE params':<25}")
print(f"    {'Example':<25} {'LDC, CHT tutorials':<25} {'This tutorial':<25}")
print()
print("  Comparison with previous tutorials:")
print(f"    {'Tutorial':<30} {'Type':<15} {'Model':<20}")
print(f"    {'-'*30} {'-'*15} {'-'*20}")
print(f"    {'PINN LDC2D':<30} {'Forward':<15} {'FullyConnected':<20}")
print(f"    {'PINN CHT2D':<30} {'Forward':<15} {'FullyConnected':<20}")
print(f"    {'FNO Darcy (synthetic)':<30} {'Data-driven':<15} {'FNO':<20}")
print(f"    {'FNO Darcy (builtin)':<30} {'Data-driven':<15} {'FNO':<20}")
print(f"    {'Inverse PINN (Burgers)':<30} {'Inverse':<15} {'FullyConnected':<20}")
print()
print("  Result files:")
print(f"    - {fig_path}")
print("=" * 70)
