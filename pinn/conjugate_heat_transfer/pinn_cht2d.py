"""
PhysicsNeMo Tutorial: 2D Conjugate Heat Transfer with PINN
============================================================
This tutorial solves a 2D Conjugate Heat Transfer (CHT) problem using PINN.

Conjugate Heat Transfer = heat transfer between a FLUID and a SOLID at their
interface. This is a fundamental problem in engineering:
  - Electronics cooling (chip + heat sink)
  - Engine thermal management (combustion gas + metal wall)
  - Heat exchangers (hot fluid + cold fluid separated by a wall)

Problem Setup:
  A hot fluid flows through the bottom channel (fluid domain).
  A solid wall sits above the fluid (solid domain).
  The top of the solid is cooled (T_cold).
  Heat transfers from hot fluid -> through the interface -> into the solid.

  +--------------------------------------+  y=1  (T_cold boundary)
  |          SOLID DOMAIN                |
  |    T_s: solid temperature            |       k_s = 2.0 (high conductivity)
  |    Heat equation: Laplacian(T_s) = 0 |       (e.g., metal wall)
  +--------------------------------------+  y=h  (fluid-solid interface)
  |          FLUID DOMAIN                |
  |    T_f: fluid temperature            |       k_f = 0.5 (low conductivity)
  |    u(x,y): Poiseuille flow velocity  |       (e.g., water/gas)
  |    Energy eq: u*dT/dx = a_f*Lap(T_f)|       a_f = thermal diffusivity
  +--------------------------------------+  y=0  (bottom wall, u=0)
  x=0 (T_hot inlet)                  x=L (outlet, dT/dx=0)

Interface conditions (y = h):
  1. Temperature continuity:  T_f(x, h) = T_s(x, h)
  2. Heat flux continuity:    k_f * dT_f/dy = k_s * dT_s/dy

Key difference from the LDC tutorial:
  - LDC: single domain (fluid only), Navier-Stokes equations
  - CHT: TWO domains (fluid + solid), different equations in each,
         coupled by interface conditions

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_pinn_cht2d.py
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
print("  PhysicsNeMo PINN Tutorial: 2D Conjugate Heat Transfer")
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
# [1] Problem Setup
# ============================================================================
# Physical parameters
K_F = 0.5         # Fluid thermal conductivity [W/(m*K)]
K_S = 2.0          # Solid thermal conductivity [W/(m*K)] (metal > fluid)
ALPHA_F = 0.01     # Fluid thermal diffusivity [m^2/s] (Pe~200, advection-dominated)


RHO_F = 1.0        # Fluid density [kg/m^3]
CP_F = 1.0         # Fluid specific heat [J/(kg*K)]
# Note: alpha_f = k_f / (rho_f * cp_f), here we set it directly for simplicity

# Domain parameters
L = 2.0            # Channel length (x: 0 to L)
H_FLUID = 0.5      # Fluid domain height (y: 0 to H_FLUID)
H_SOLID = 0.5      # Solid domain height (y: H_FLUID to H_FLUID + H_SOLID)
H_TOTAL = H_FLUID + H_SOLID  # Total height = 1.0

# Boundary conditions
T_HOT = 1.0        # Inlet fluid temperature (hot fluid enters)
T_COLD = 0.0       # Top solid temperature (cooled from above)

# Poiseuille flow velocity profile (parabolic):
#   u(x, y) = u_max * (1 - (2y/H_fluid - 1)^2)  for fluid domain
#   This is the analytical solution for pressure-driven flow in a channel.
#   We use it as a known velocity field (no need to solve Navier-Stokes).
U_MAX = 1.0        # Maximum flow velocity at channel center

def poiseuille_velocity(y, h_fluid=H_FLUID, u_max=U_MAX):
    """Compute Poiseuille flow velocity in the fluid domain.
    
    Parabolic velocity profile: u = u_max * (1 - (2y/h - 1)^2)
    This is the exact solution for steady pressure-driven flow between
    two parallel plates (no-slip at walls).
    
    Returns u (x-direction velocity). v (y-direction) = 0 for Poiseuille flow.
    """
    y_norm = 2.0 * y / h_fluid - 1.0  # Normalize to [-1, 1]
    u = u_max * (1.0 - y_norm ** 2)
    return u

print(f"[1/8] Problem: 2D Conjugate Heat Transfer")
print(f"      Fluid: k_f={K_F}, alpha_f={ALPHA_F}, domain y=[0, {H_FLUID}]")
print(f"      Solid: k_s={K_S}, domain y=[{H_FLUID}, {H_TOTAL}]")
print(f"      BC: T_hot={T_HOT} (inlet), T_cold={T_COLD} (top)")
print(f"      Interface at y={H_FLUID}")
print()


# ============================================================================
# [2] Create PhysicsNeMo Model
# ============================================================================
# Single neural network for BOTH domains.
# Input: (x, y) -> Output: T (temperature)
# The network learns to output different behaviors in fluid vs solid regions
# through the PDE loss (different equations in each domain).

print("[2/8] Creating PhysicsNeMo FullyConnected model...")

model = FullyConnected(
    in_features=2,        # Input: x, y coordinates
    out_features=1,       # Output: temperature T
    layer_size=80,        # Neurons per hidden layer (larger for better expressivity)
    num_layers=6,         # Number of hidden layers (deeper for complex CHT)
    activation_fn="tanh", # Tanh (smooth, differentiable for 2nd-order derivatives)
    weight_norm=True,     # Weight normalization for training stability
)

model = model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"      Model parameters: {n_params:,}")


# ============================================================================
# [3] Training Setup
# ============================================================================
print("\n[3/8] Setting up training...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS_ADAM = 3000   # Adam epochs (phase 1: fast convergence)
EPOCHS_LBFGS = 500   # L-BFGS steps (phase 2: fine-tuning to near-zero residual)
N_FLUID = 5000       # Interior points in fluid domain (more points for accuracy)
N_SOLID = 5000       # Interior points in solid domain
N_INTERFACE = 800    # Points on fluid-solid interface
N_BC = 300           # Points per boundary



# ============================================================================
# [4] Generate Sampling Points
# ============================================================================
print("[4/8] Generating training points...")

# --- Fluid domain points: x in [0, L], y in [0, H_FLUID] ---
x_fluid = torch.rand(N_FLUID, 1, device=device) * L
y_fluid = torch.rand(N_FLUID, 1, device=device) * H_FLUID
x_fluid.requires_grad_(True)
y_fluid.requires_grad_(True)

# --- Solid domain points: x in [0, L], y in [H_FLUID, H_TOTAL] ---
x_solid = torch.rand(N_SOLID, 1, device=device) * L
y_solid = torch.rand(N_SOLID, 1, device=device) * H_SOLID + H_FLUID
x_solid.requires_grad_(True)
y_solid.requires_grad_(True)

# --- Interface points: y = H_FLUID, x in [0, L] ---
x_iface = torch.rand(N_INTERFACE, 1, device=device) * L
y_iface = torch.full((N_INTERFACE, 1), H_FLUID, device=device)
x_iface.requires_grad_(True)
y_iface.requires_grad_(True)

# --- Boundary: Inlet (x=0), fluid domain ---
x_inlet = torch.zeros(N_BC, 1, device=device)
y_inlet = torch.rand(N_BC, 1, device=device) * H_FLUID

# --- Boundary: Outlet (x=L), fluid domain ---
x_outlet = torch.full((N_BC, 1), L, device=device)
y_outlet = torch.rand(N_BC, 1, device=device) * H_FLUID
x_outlet.requires_grad_(True)  # Need gradient for dT/dx = 0

# --- Boundary: Top (y=H_TOTAL), solid domain ---
x_top = torch.rand(N_BC, 1, device=device) * L
y_top = torch.full((N_BC, 1), H_TOTAL, device=device)

# --- Boundary: Bottom (y=0), fluid domain ---
x_bottom = torch.rand(N_BC, 1, device=device) * L
y_bottom = torch.zeros(N_BC, 1, device=device)

print(f"      Fluid points: {N_FLUID}, Solid points: {N_SOLID}")
print(f"      Interface points: {N_INTERFACE}, BC points/wall: {N_BC}")


# ============================================================================
# [5] Loss Functions
# ============================================================================
print("\n[5/8] Defining PINN loss functions...")

def predict_temperature(model, x, y):
    """Feed (x, y) into the network and return temperature T."""
    inputs = torch.cat([x, y], dim=1)
    outputs = model(inputs)
    T = outputs[:, 0:1]
    return T

def compute_fluid_pde_residual(model, x, y):
    """
    Fluid domain PDE: Energy equation (advection-diffusion)
    
      u * dT/dx + v * dT/dy = alpha_f * (d2T/dx2 + d2T/dy2)
    
    Since v = 0 (Poiseuille flow) and u is known:
      u * dT/dx - alpha_f * (d2T/dx2 + d2T/dy2) = 0  (residual)
    
    Returns the residual (should be close to 0 when PDE is satisfied).
    """
    T = predict_temperature(model, x, y)
    
    # 1st-order derivatives
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                               create_graph=True, retain_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T),
                               create_graph=True, retain_graph=True)[0]
    
    # 2nd-order derivatives
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                                create_graph=True, retain_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y),
                                create_graph=True, retain_graph=True)[0]
    
    # Poiseuille velocity (known analytically, no need to solve NS)
    u = poiseuille_velocity(y)
    
    # Residual: u*dT/dx - alpha_f*(d2T/dx2 + d2T/dy2) = 0
    residual = u * T_x - ALPHA_F * (T_xx + T_yy)
    return residual

def compute_solid_pde_residual(model, x, y):
    """
    Solid domain PDE: Steady-state heat conduction (Laplace equation)
    
      d2T_s/dx2 + d2T_s/dy2 = 0  (Laplace equation)
    
    This means: temperature distribution satisfies Laplace's equation in
    the solid (no heat generation, steady state).
    
    Returns the residual (should be close to 0).
    """
    T = predict_temperature(model, x, y)
    
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                               create_graph=True, retain_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T),
                               create_graph=True, retain_graph=True)[0]
    
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                                create_graph=True, retain_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y),
                                create_graph=True, retain_graph=True)[0]
    
    # Residual: d2T/dx2 + d2T/dy2 = 0
    residual = T_xx + T_yy
    return residual

def compute_interface_loss(model, x, y):
    """
    Interface conditions at y = H_FLUID:
    
    1. Temperature continuity: T_fluid(x, h) = T_solid(x, h)
       (Temperature must be the same on both sides of the interface)
    
    2. Heat flux continuity: k_f * dT_f/dy = k_s * dT_s/dy
       (Heat flux must be conserved across the interface)
       
       Note: We approximate this by evaluating the network slightly below
       and slightly above the interface to get fluid-side and solid-side
       gradients. This is a common PINN technique for interface problems.
    """
    # Temperature at interface (single network, so T is continuous by design)
    T_iface = predict_temperature(model, x, y)
    
    # Gradient from fluid side (evaluate just below interface)
    eps = 1e-3
    x_f = x.detach().clone().requires_grad_(True)
    y_f = (y - eps).detach().clone().requires_grad_(True)
    T_f = predict_temperature(model, x_f, y_f)
    T_f_y = torch.autograd.grad(T_f, y_f, grad_outputs=torch.ones_like(T_f),
                                 create_graph=True, retain_graph=True)[0]
    
    # Gradient from solid side (evaluate just above interface)
    x_s = x.detach().clone().requires_grad_(True)
    y_s = (y + eps).detach().clone().requires_grad_(True)
    T_s = predict_temperature(model, x_s, y_s)
    T_s_y = torch.autograd.grad(T_s, y_s, grad_outputs=torch.ones_like(T_s),
                                 create_graph=True, retain_graph=True)[0]
    
    # Loss 1: Temperature continuity (T should be smooth across interface)
    # Since we use a single network, T is already continuous.
    # But we enforce flux continuity:
    # k_f * dT_f/dy = k_s * dT_s/dy
    flux_residual = K_F * T_f_y - K_S * T_s_y
    
    return torch.mean(flux_residual ** 2)

def compute_boundary_loss(model):
    """
    Boundary conditions:
    1. Inlet (x=0, fluid): T = T_HOT
    2. Outlet (x=L, fluid): dT/dx = 0 (free outflow)
    3. Top (y=H_TOTAL, solid): T = T_COLD
    4. Bottom (y=0, fluid): dT/dy = 0 (insulated bottom or symmetry)
    """
    loss = torch.tensor(0.0, device=device)
    
    # 1. Inlet: T = T_HOT
    T_inlet = predict_temperature(model, x_inlet, y_inlet)
    loss += torch.mean((T_inlet - T_HOT) ** 2)
    
    # 2. Outlet: dT/dx = 0
    T_outlet = predict_temperature(model, x_outlet, y_outlet)
    T_outlet_x = torch.autograd.grad(T_outlet, x_outlet,
                                      grad_outputs=torch.ones_like(T_outlet),
                                      create_graph=True, retain_graph=True)[0]
    loss += torch.mean(T_outlet_x ** 2)
    
    # 3. Top: T = T_COLD
    T_top = predict_temperature(model, x_top, y_top)
    loss += torch.mean((T_top - T_COLD) ** 2)
    
    # 4. Bottom: dT/dy = 0 (insulated)
    x_bot = x_bottom.detach().clone().requires_grad_(True)
    y_bot = y_bottom.detach().clone().requires_grad_(True)
    T_bot = predict_temperature(model, x_bot, y_bot)
    T_bot_y = torch.autograd.grad(T_bot, y_bot,
                                   grad_outputs=torch.ones_like(T_bot),
                                   create_graph=True, retain_graph=True)[0]
    loss += torch.mean(T_bot_y ** 2)
    
    return loss

def total_loss(model):
    """
    Total loss = fluid PDE + solid PDE + interface + boundary conditions
    
    Weights are tuned to balance the different loss terms.
    """
    # PDE residuals
    res_fluid = compute_fluid_pde_residual(model, x_fluid, y_fluid)
    res_solid = compute_solid_pde_residual(model, x_solid, y_solid)
    
    loss_pde_fluid = torch.mean(res_fluid ** 2)
    loss_pde_solid = torch.mean(res_solid ** 2)
    
    # Interface conditions
    loss_interface = compute_interface_loss(model, x_iface, y_iface)
    
    # Boundary conditions
    loss_bc = compute_boundary_loss(model)
    
    # Total loss with weights
    total = (loss_pde_fluid + loss_pde_solid 
             + 10.0 * loss_interface 
             + 10.0 * loss_bc)
    
    return total, loss_pde_fluid, loss_pde_solid, loss_interface, loss_bc


# ============================================================================
# [6] Training Loop (Two-Phase: Adam + L-BFGS)
# ============================================================================
# Phase 1: Adam optimizer for fast initial convergence
# Phase 2: L-BFGS optimizer for fine-tuning to near-zero PDE residual
# This two-phase approach is standard practice in PINN training and
# dramatically improves solution quality (smoother fields, lower residual).

print(f"\n[6/8] Training PINN (Phase 1: Adam {EPOCHS_ADAM} + Phase 2: L-BFGS {EPOCHS_LBFGS})")
print()

start_time = time.time()
loss_history = []

# --- Phase 1: Adam with cosine annealing LR scheduler ---
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS_ADAM, eta_min=1e-5
)

print("  --- Phase 1: Adam (cosine annealing) ---")
for epoch in range(EPOCHS_ADAM):
    optimizer.zero_grad()
    
    loss, l_pf, l_ps, l_if, l_bc = total_loss(model)
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if epoch % 500 == 0 or epoch == EPOCHS_ADAM - 1:
        elapsed = time.time() - start_time
        print(f"  Adam  {epoch:5d}/{EPOCHS_ADAM} | "
              f"Total: {loss.item():.6e} | "
              f"PDE_f: {l_pf.item():.4e} | "
              f"PDE_s: {l_ps.item():.4e} | "
              f"IF: {l_if.item():.4e} | "
              f"BC: {l_bc.item():.4e} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {elapsed:.1f}s")

# --- Phase 2: L-BFGS for fine-tuning ---
# L-BFGS uses second-order quasi-Newton approximation to achieve
# much lower residuals than first-order methods like Adam.
print("\n  --- Phase 2: L-BFGS (fine-tuning) ---")

lbfgs_optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=EPOCHS_LBFGS,
    max_eval=EPOCHS_LBFGS,
    history_size=50,
    tolerance_grad=1e-9,
    tolerance_change=1e-11,
    line_search_fn="strong_wolfe",
)

lbfgs_step = [0]  # Use list for mutable closure variable

def lbfgs_closure():
    lbfgs_optimizer.zero_grad()
    loss, l_pf, l_ps, l_if, l_bc = total_loss(model)
    loss.backward()
    
    lbfgs_step[0] += 1
    if lbfgs_step[0] % 50 == 0 or lbfgs_step[0] == 1:
        elapsed = time.time() - start_time
        print(f"  LBFGS {lbfgs_step[0]:5d}/{EPOCHS_LBFGS} | "
              f"Total: {loss.item():.6e} | "
              f"PDE_f: {l_pf.item():.4e} | "
              f"PDE_s: {l_ps.item():.4e} | "
              f"IF: {l_if.item():.4e} | "
              f"BC: {l_bc.item():.4e} | "
              f"Time: {elapsed:.1f}s")
    
    loss_history.append(loss.item())
    return loss

lbfgs_optimizer.step(lbfgs_closure)

# Get final loss
loss, l_pf, l_ps, l_if, l_bc = total_loss(model)

elapsed_total = time.time() - start_time
print(f"\n  Training complete! Total time: {elapsed_total:.1f}s")
print(f"  Final loss (Adam): {loss_history[EPOCHS_ADAM-1]:.6e}")
print(f"  Final loss (L-BFGS): {loss.item():.6e}")


# Save model
output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "cht2d_model.pth")
torch.save(model.state_dict(), model_path)
print(f"  Model saved: {model_path}")


# ============================================================================
# [7] Visualization
# ============================================================================
print("\n[7/8] Visualizing results...")

model.eval()

# All visualization is done under no_grad (no differentiation needed)
with torch.no_grad():
    # Create grid for visualization
    N_GRID_X = 80
    N_GRID_Y = 50
    x_grid = torch.linspace(0, L, N_GRID_X, device=device)
    y_grid = torch.linspace(0, H_TOTAL, N_GRID_Y, device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    
    inputs = torch.cat([X_flat, Y_flat], dim=1)
    T_pred = model(inputs)
    T_grid = T_pred.reshape(N_GRID_X, N_GRID_Y).cpu().numpy()
    
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # Pre-compute temperature profiles for vertical slices
    profile_data = []
    x_slices = [0.5, 1.0, 1.5]
    for x_val in x_slices:
        y_line = torch.linspace(0, H_TOTAL, 200, device=device).reshape(-1, 1)
        x_line = torch.full_like(y_line, x_val)
        inputs_line = torch.cat([x_line, y_line], dim=1)
        T_line = model(inputs_line).cpu().numpy().flatten()
        y_np = y_line.cpu().numpy().flatten()
        profile_data.append((T_line, y_np, x_val))
    
    # Pre-compute interface temperature along x
    x_iface_plot = torch.linspace(0, L, 200, device=device).reshape(-1, 1)
    y_iface_plot = torch.full_like(x_iface_plot, H_FLUID)
    inputs_iface = torch.cat([x_iface_plot, y_iface_plot], dim=1)
    T_iface_plot = model(inputs_iface).cpu().numpy().flatten()
    x_iface_np = x_iface_plot.cpu().numpy().flatten()


# Create figure: 3 subplots
fig = plt.figure(figsize=(16, 5))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

# 1. Temperature field (full domain)
ax1 = fig.add_subplot(gs[0, 0])
# Transpose for correct orientation (x horizontal, y vertical)
im1 = ax1.pcolormesh(X_np, Y_np, T_grid.T, cmap='hot', shading='gouraud')
ax1.axhline(y=H_FLUID, color='cyan', linewidth=1.5, linestyle='--', label='Interface')
ax1.set_title('Temperature Field T(x, y)', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend(fontsize=8, loc='upper right')
plt.colorbar(im1, ax=ax1, label='T')

# 2. Temperature profiles at different x locations
ax2 = fig.add_subplot(gs[0, 1])
colors = ['red', 'green', 'blue']
for (T_line, y_np, x_val), color in zip(profile_data, colors):
    ax2.plot(T_line, y_np, color=color, label=f'x={x_val}', linewidth=1.5)


ax2.axhline(y=H_FLUID, color='gray', linewidth=1, linestyle='--', alpha=0.7)
ax2.axvline(x=T_HOT, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
ax2.axvline(x=T_COLD, color='blue', linewidth=0.5, linestyle=':', alpha=0.5)
ax2.set_title('Temperature Profiles (vertical slices)', fontsize=12)
ax2.set_xlabel('Temperature T')
ax2.set_ylabel('y')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Interface temperature along x (use pre-computed data from no_grad block)
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(x_iface_np, T_iface_plot, 'b-', linewidth=2, label='Interface T(x, h)')

ax3.axhline(y=T_HOT, color='red', linewidth=0.5, linestyle=':', label=f'T_hot={T_HOT}')
ax3.axhline(y=T_COLD, color='blue', linewidth=0.5, linestyle=':', label=f'T_cold={T_COLD}')
ax3.set_title('Interface Temperature Distribution', fontsize=12)
ax3.set_xlabel('x')
ax3.set_ylabel('T at interface (y=h)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.suptitle(
    f'PINN: 2D Conjugate Heat Transfer\n'
    f'Fluid (k_f={K_F}) + Solid (k_s={K_S}), Interface at y={H_FLUID}\n'
    f'PhysicsNeMo {physicsnemo.__version__} | {device} | Adam {EPOCHS_ADAM} + L-BFGS {EPOCHS_LBFGS}',

    fontsize=13, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(output_dir, "cht2d_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Result image saved: {fig_path}")

# Loss curve
fig2, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(loss_history, linewidth=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Loss (log scale)')
ax.set_title('CHT PINN Training Loss History', fontsize=14)
ax.grid(True, alpha=0.3)
fig2_path = os.path.join(output_dir, "cht2d_loss.png")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Loss curve saved: {fig2_path}")

plt.close('all')


# ============================================================================
# [8] Summary
# ============================================================================
print("\n[8/8] Tutorial complete!")
print("=" * 70)
print("  CHT PINN Tutorial Summary:")
print(f"    - Problem: 2D Conjugate Heat Transfer (fluid + solid)")
print(f"    - Fluid PDE: Advection-diffusion (Poiseuille flow + heat)")
print(f"    - Solid PDE: Laplace equation (steady heat conduction)")
print(f"    - Interface: Temperature + heat flux continuity")
print(f"    - Network: FullyConnected (6 layers, 80 neurons/layer, Tanh)")
print(f"    - Training: Adam({EPOCHS_ADAM}) + L-BFGS({EPOCHS_LBFGS})")
print(f"    - Training time: {elapsed_total:.1f}s")
print(f"    - Final loss: {loss.item():.6e}")

print(f"    - Device: {device}")
print()
print("  Key physics captured:")
print("    - Hot fluid enters at x=0 (T=1.0)")
print("    - Heat transfers from fluid to solid across interface")
print("    - Solid is cooled from top (T=0.0)")
print("    - Temperature is continuous at the interface")
print("    - Heat flux is conserved across the interface")
print()
print("  Comparison with previous tutorials:")
print("    - LDC PINN: single domain, Navier-Stokes (flow)")
print("    - FNO Darcy: data-driven, single domain (permeability->pressure)")
print("    - CHT PINN: TWO domains, coupled by interface conditions (heat)")
print()
print("  Result files:")
print(f"    - {fig_path}")
print(f"    - {fig2_path}")
print("=" * 70)
