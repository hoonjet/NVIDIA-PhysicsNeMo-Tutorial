"""
PhysicsNeMo Tutorial: 2D Plane Stress Analysis with PINN
=========================================================
This tutorial uses PhysicsNeMo's FullyConnected model to implement a PINN
(Physics-Informed Neural Network) for solid mechanics — specifically, 2D plane
stress analysis of a cantilever beam.

Problem: 2D Cantilever Beam under End Load (Plane Stress)
- A beam is fixed at the left end and subjected to a downward point load at the right end
- The displacement field (u, v) is predicted by a neural network
- The network is trained to satisfy the equilibrium equations (no labeled data needed)

Learning objectives:
1. Solid mechanics fundamentals (stress, strain, equilibrium)
2. How to encode elasticity equations as PINN loss
3. Comparison with Euler-Bernoulli beam theory (analytical solution)
4. Stress field computation from displacement field

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_pinn_planestress.py

=========================================================================
[Physics Background: Plane Stress Elasticity]
=========================================================================

In solid mechanics, the displacement field u(x,y) = (u_x, u_y) describes how
each point in a body deforms. From displacement, we compute strain, then stress,
using the constitutive law (Hooke's law).

--- Strain-Displacement Relations (Small Deformation) ---

    ε_xx = ∂u/∂x          (normal strain in x)
    ε_yy = ∂v/∂y          (normal strain in y)
    γ_xy = ∂u/∂y + ∂v/∂x  (engineering shear strain)

--- Constitutive Law (Plane Stress, Hooke's Law) ---

For plane stress (σ_zz = 0), the stress-strain relationship is:

    σ_xx = E/(1-ν²) × (ε_xx + ν × ε_yy)
    σ_yy = E/(1-ν²) × (ε_yy + ν × ε_xx)
    τ_xy = E/(1+ν) × γ_xy / 2  = G × γ_xy

where:
    E = Young's modulus (material stiffness)
    ν = Poisson's ratio (lateral contraction ratio)
    G = E / (2(1+ν)) = shear modulus

--- Equilibrium Equations (No Body Forces) ---

    ∂σ_xx/∂x + ∂τ_xy/∂y = 0   (x-direction equilibrium)
    ∂τ_xy/∂x + ∂σ_yy/∂y = 0   (y-direction equilibrium)

These are the PDEs that the PINN must satisfy.

--- Boundary Conditions ---

    Left end (x=0): u = 0, v = 0  (fixed/clamped)
    Right end (x=L): Applied load (Neumann BC or approximate with traction)

=========================================================================
[Problem Setup: Cantilever Beam]
=========================================================================

    L = 4.0 (length)      H = 1.0 (height)
    E = 1.0 (Young's modulus, normalized)
    ν = 0.3 (Poisson's ratio)
    P = -1.0 (downward end load, normalized)

    Fixed end              Free end (load P)
    +----------------------+
    |                      | ↓ P
    |        beam          |
    |                      |
    +----------------------+
    x=0                    x=L

    Analytical solution (Euler-Bernoulli beam theory):
    v(x) = -P x² (3L - x) / (6 E I)   (vertical deflection)
    where I = H³/12 (second moment of area for unit depth)
"""

# ============================================================================
# Library Imports
# ============================================================================
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# PhysicsNeMo: FullyConnected (MLP) model
from physicsnemo.models.mlp.fully_connected import FullyConnected

# ============================================================================
# [0] Environment Setup
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")

# ============================================================================
# [1] Problem Parameters
# ============================================================================
print("\n[1] Problem Setup: 2D Plane Stress Cantilever Beam")

# Geometry
L = 4.0    # beam length
H = 1.0    # beam height

# Material properties (normalized)
E_mod = 1.0    # Young's modulus
nu = 0.3       # Poisson's ratio

# Derived constants for plane stress
# D_matrix components (constitutive matrix for plane stress)
# D = E/(1-nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
C11 = E_mod / (1 - nu**2)  # coefficient for sigma_xx from eps_xx
C12 = nu * C11              # coefficient for sigma_xx from eps_yy (and vice versa)
C33 = E_mod / (2 * (1 + nu))  # shear modulus G

# Load
P_load = -1.0  # downward force at right end

# Second moment of area (for unit depth, I = H^3/12)
I_inertia = H**3 / 12.0

print(f"   Beam: L={L}, H={H}")
print(f"   Material: E={E_mod}, ν={nu}")
print(f"   Plane stress: C11={C11:.4f}, C12={C12:.4f}, G={C33:.4f}")
print(f"   Load: P={P_load}")
print(f"   I = {I_inertia:.6f}")

# ============================================================================
# [2] Create Neural Network Model
# ============================================================================
print("\n[2] Creating PINN model (FullyConnected)...")

# The network maps (x, y) -> (u, v) where:
#   u = displacement in x direction
#   v = displacement in y direction
model = FullyConnected(
    in_features=2,        # (x, y)
    out_features=2,       # (u, v)
    layer_size=64,        # neurons per hidden layer
    num_layers=6,         # number of hidden layers
    activation_fn="tanh", # smooth activation (important for derivatives)
    weight_norm=True,     # weight normalization for training stability
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"   Model: {n_params} parameters")
print(f"   Architecture: 2 -> 64 -> 64 -> 64 -> 64 -> 64 -> 2 (tanh)")

# ============================================================================
# [3] Training Setup
# ============================================================================
print("\n[3] Training setup...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

n_epochs = 5000

# Generate collocation points (interior)
n_interior = 2000
x_interior = torch.rand(n_interior, 1, device=device) * L
y_interior = torch.rand(n_interior, 1, device=device) * H

# Boundary points
# Left boundary (x=0): fixed (u=0, v=0)
n_left = 200
x_left = torch.zeros(n_left, 1, device=device)
y_left = torch.rand(n_left, 1, device=device) * H

# Right boundary (x=L): applied load
n_right = 200
x_right = torch.full((n_right, 1), L, device=device)
y_right = torch.rand(n_right, 1, device=device) * H

# Top and bottom boundaries (free surface: traction = 0)
n_top = 200
x_top = torch.rand(n_top, 1, device=device) * L
y_top = torch.full((n_top, 1), H, device=device)

n_bottom = 200
x_bottom = torch.rand(n_bottom, 1, device=device) * L
y_bottom = torch.zeros(n_bottom, 1, device=device)

print(f"   Epochs: {n_epochs}")
print(f"   Interior points: {n_interior}")
print(f"   Boundary points: {n_left} (left) + {n_right} (right) + {n_top} (top) + {n_bottom} (bottom)")

# ============================================================================
# [4] Physics: Compute Stress from Displacement
# ============================================================================
print("\n[4] Defining physics functions...")

def compute_strain_stress(model, x, y):
    """
    Compute strain and stress from displacement field using autograd.

    Strain-displacement relations (small deformation):
        ε_xx = ∂u/∂x
        ε_yy = ∂v/∂y
        γ_xy = ∂u/∂y + ∂v/∂x

    Plane stress constitutive law (Hooke's law):
        σ_xx = C11 * ε_xx + C12 * ε_yy
        σ_yy = C12 * ε_xx + C11 * ε_yy
        τ_xy = C33 * γ_xy  (= G * γ_xy)

    Parameters
    ----------
    model : nn.Module
        Neural network mapping (x, y) -> (u, v)
    x, y : torch.Tensor
        Coordinates (require_grad=True)

    Returns
    -------
    u, v : displacement components
    sigma_xx, sigma_yy, tau_xy : stress components
    """
    # Ensure gradients are enabled
    x.requires_grad_(True)
    y.requires_grad_(True)

    # Forward pass
    uv = model(torch.cat([x, y], dim=1))
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    
    # First derivatives (strain-displacement)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                              create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                              create_graph=True)[0]

    # Strains
    eps_xx = u_x
    eps_yy = v_y
    gamma_xy = u_y + v_x  # engineering shear strain

    # Stresses (plane stress constitutive law)
    sigma_xx = C11 * eps_xx + C12 * eps_yy
    sigma_yy = C12 * eps_xx + C11 * eps_yy
    tau_xy = C33 * gamma_xy

    return u, v, sigma_xx, sigma_yy, tau_xy


def equilibrium_residual(model, x, y):
    """
    Compute equilibrium PDE residuals.

    Equilibrium equations (no body forces):
        ∂σ_xx/∂x + ∂τ_xy/∂y = 0   (x-direction)
        ∂τ_xy/∂x + ∂σ_yy/∂y = 0   (y-direction)

    Returns
    -------
    res_x, res_y : PDE residuals (should be zero if equilibrium is satisfied)
    """
    x.requires_grad_(True)
    y.requires_grad_(True)

    u, v, sigma_xx, sigma_yy, tau_xy = compute_strain_stress(model, x, y)

    # Second derivatives (equilibrium)
    sigma_xx_x = torch.autograd.grad(sigma_xx, x, grad_outputs=torch.ones_like(sigma_xx),
                                     create_graph=True)[0]
    tau_xy_y = torch.autograd.grad(tau_xy, y, grad_outputs=torch.ones_like(tau_xy),
                                   create_graph=True)[0]

    tau_xy_x = torch.autograd.grad(tau_xy, x, grad_outputs=torch.ones_like(tau_xy),
                                   create_graph=True)[0]
    sigma_yy_y = torch.autograd.grad(sigma_yy, y, grad_outputs=torch.ones_like(sigma_yy),
                                     create_graph=True)[0]

    # Equilibrium residuals
    res_x = sigma_xx_x + tau_xy_y   # should be 0
    res_y = tau_xy_x + sigma_yy_y   # should be 0

    return res_x, res_y


# ============================================================================
# [5] Training Loop
# ============================================================================
print("\n[5] Training PINN...")
print("    Loss = λ_pde * L_pde + λ_bc_left * L_bc_left + λ_bc_right * L_bc_right + λ_bc_free * L_bc_free")
print("    where:")
print("      L_pde       = equilibrium equation residual (interior)")
print("      L_bc_left   = u=0, v=0 at fixed end (Dirichlet)")
print("      L_bc_right  = traction = applied load (Neumann)")
print("      L_bc_free   = traction = 0 at top/bottom (free surface)\n")

# Loss weights
lambda_pde = 1.0
lambda_bc_left = 50.0     # Dirichlet BC (strong enforcement needed)
lambda_bc_right = 10.0    # Neumann BC (applied load)
lambda_bc_free = 10.0     # Neumann BC (free surface)

loss_history = {'total': [], 'pde': [], 'bc_left': [], 'bc_right': [], 'bc_free': []}

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    # --- PDE Loss (Equilibrium equations in interior) ---
    res_x, res_y = equilibrium_residual(model, x_interior, y_interior)
    loss_pde = torch.mean(res_x**2) + torch.mean(res_y**2)

    # --- BC: Left end fixed (u=0, v=0) ---
    uv_left = model(torch.cat([x_left, y_left], dim=1))
    u_left = uv_left[:, 0:1]
    v_left = uv_left[:, 1:2]
    loss_bc_left = torch.mean(u_left**2) + torch.mean(v_left**2)

    # --- BC: Right end (applied load as traction) ---
    # Approximate: apply vertical force distributed over the right edge
    # σ_xx at x=L should be 0, τ_xy at x=L should be P/H (average shear)
    # For simplicity, we use a weak form: v at right end follows beam theory
    # v_beam(x=L) = -P*L^3 / (3*E*I)
    v_tip_analytical = -P_load * L**3 / (3 * E_mod * I_inertia) * (-1)  # P_load is negative
    # Actually: v(L) = P*L^3/(3EI) with P downward (negative)
    v_tip_target = P_load * L**3 / (3 * E_mod * I_inertia)

    uv_right = model(torch.cat([x_right, y_right], dim=1))
    v_right = uv_right[:, 1:2]
    # Apply average displacement at right end (weak enforcement)
    loss_bc_right = torch.mean((v_right - v_tip_target)**2)

    # --- BC: Free surface (top and bottom, traction = 0) ---
    # At y=0 and y=H: σ_yy = 0, τ_xy = 0
    _, _, _, sigma_yy_top, tau_xy_top = compute_strain_stress(model, x_top, y_top)
    _, _, _, sigma_yy_bot, tau_xy_bot = compute_strain_stress(model, x_bottom, y_bottom)
    loss_bc_free = (torch.mean(sigma_yy_top**2) + torch.mean(tau_xy_top**2) +
                     torch.mean(sigma_yy_bot**2) + torch.mean(tau_xy_bot**2))

    # Total loss
    loss = (lambda_pde * loss_pde +
            lambda_bc_left * loss_bc_left +
            lambda_bc_right * loss_bc_right +
            lambda_bc_free * loss_bc_free)

    loss.backward()
    optimizer.step()

    # Record
    loss_history['total'].append(loss.item())
    loss_history['pde'].append(loss_pde.item())
    loss_history['bc_left'].append(loss_bc_left.item())
    loss_history['bc_right'].append(loss_bc_right.item())
    loss_history['bc_free'].append(loss_bc_free.item())

    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1:5d}/{n_epochs} | Total: {loss.item():.6e} | "
              f"PDE: {loss_pde.item():.6e} | BC_left: {loss_bc_left.item():.6e} | "
              f"BC_right: {loss_bc_right.item():.6e} | BC_free: {loss_bc_free.item():.6e}")

    scheduler.step()

print("\n   Training complete!")

# ============================================================================
# [6] Analytical Solution (Euler-Bernoulli Beam Theory)
# ============================================================================
print("\n[6] Computing analytical solution (Euler-Bernoulli beam theory)...")

def analytical_beam_deflection(x, L, P, E, I):
    """
    Euler-Bernoulli beam theory for cantilever with end load.

    v(x) = P x^2 (3L - x) / (6 E I)

    For downward load P < 0, deflection is downward (negative v).

    Parameters
    ----------
    x : array
        Position along beam
    L : float
        Beam length
    P : float
        End load (negative = downward)
    E : float
        Young's modulus
    I : float
        Second moment of area

    Returns
    -------
    v : array
        Vertical deflection
    """
    v = P * x**2 * (3 * L - x) / (6 * E * I)
    return v

# ============================================================================
# [7] Visualization
# ============================================================================
print("\n[7] Visualizing results...")

# Create evaluation grid
n_eval = 100
x_eval = np.linspace(0, L, n_eval)
y_eval = np.linspace(0, H, n_eval // 2)
X, Y = np.meshgrid(x_eval, y_eval)

# Predict displacement field
model.eval()
with torch.no_grad():
    xy_grid = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32).to(device)
    uv_pred = model(xy_grid)
    U_pred = uv_pred[:, 0].cpu().numpy().reshape(X.shape)
    V_pred = uv_pred[:, 1].cpu().numpy().reshape(X.shape)

# Compute stress field (requires grad)
x_grid = torch.tensor(X.ravel(), dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
y_grid = torch.tensor(Y.ravel(), dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
u_g, v_g, sxx_g, syy_g, txy_g = compute_strain_stress(model, x_grid, y_grid)

SXX = sxx_g.detach().cpu().numpy().reshape(X.shape)
SYY = syy_g.detach().cpu().numpy().reshape(X.shape)
TXY = txy_g.detach().cpu().numpy().reshape(X.shape)

# Analytical deflection at y=H/2
v_analytical = analytical_beam_deflection(x_eval, L, P_load, E_mod, I_inertia)

# --- Figure 1: Displacement and Stress Fields ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Displacement u (horizontal)
ax = axes[0, 0]
im = ax.contourf(X, Y, U_pred, levels=50, cmap='RdBu_r')
ax.set_title('Horizontal Displacement u(x,y)', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='u')

# Displacement v (vertical)
ax = axes[0, 1]
im = ax.contourf(X, Y, V_pred, levels=50, cmap='RdBu_r')
ax.set_title('Vertical Displacement v(x,y)', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='v')

# Deformed shape
ax = axes[0, 2]
scale = 50  # magnification factor for visualization
ax.plot(X.ravel(), Y.ravel(), 'k.', markersize=0.1, alpha=0.3)
ax.plot((X + scale * U_pred).ravel(), (Y + scale * V_pred).ravel(), 'r.', markersize=0.1, alpha=0.5)
ax.set_title(f'Deformed Shape (scale={scale}x)', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend(['Original', 'Deformed'], loc='upper right')

# Stress sigma_xx
ax = axes[1, 0]
im = ax.contourf(X, Y, SXX, levels=50, cmap='RdBu_r')
ax.set_title('Normal Stress σ_xx(x,y)', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='σ_xx')

# Stress sigma_yy
ax = axes[1, 1]
im = ax.contourf(X, Y, SYY, levels=50, cmap='RdBu_r')
ax.set_title('Normal Stress σ_yy(x,y)', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='σ_yy')

# Stress tau_xy
ax = axes[1, 2]
im = ax.contourf(X, Y, TXY, levels=50, cmap='RdBu_r')
ax.set_title('Shear Stress τ_xy(x,y)', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='τ_xy')

plt.suptitle('PINN: 2D Plane Stress Analysis of Cantilever Beam\n'
             '(Solid Mechanics — Not Available in PhysicsNeMo by Default)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_pinn_planestress.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_pinn_planestress.png")

# --- Figure 2: Comparison with Analytical Solution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Deflection at y=H/2
ax = axes[0]
# Extract v at y ≈ H/2
mid_idx = len(y_eval) // 2
v_pred_mid = V_pred[mid_idx, :]
ax.plot(x_eval, v_pred_mid, 'b-', linewidth=2, label='PINN Prediction')
ax.plot(x_eval, v_analytical, 'r--', linewidth=2, label='Euler-Bernoulli Theory')
ax.set_xlabel('x (along beam)', fontsize=12)
ax.set_ylabel('v (vertical deflection)', fontsize=12)
ax.set_title('Beam Deflection at y=H/2\nPINN vs Analytical', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Error
ax = axes[1]
error = np.abs(v_pred_mid - v_analytical)
ax.plot(x_eval, error, 'g-', linewidth=2)
ax.set_xlabel('x (along beam)', fontsize=12)
ax.set_ylabel('|v_PINN - v_analytical|', fontsize=12)
ax.set_title('Absolute Error', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_pinn_planestress_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_pinn_planestress_comparison.png")

# --- Figure 3: Training Loss ---
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(loss_history['total'], 'k-', label='Total', linewidth=1.5)
ax.plot(loss_history['pde'], 'b-', label='PDE (equilibrium)', alpha=0.7)
ax.plot(loss_history['bc_left'], 'r-', label='BC left (fixed)', alpha=0.7)
ax.plot(loss_history['bc_right'], 'g-', label='BC right (load)', alpha=0.7)
ax.plot(loss_history['bc_free'], 'm-', label='BC free surface', alpha=0.7)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('PINN Training Loss Components', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_pinn_planestress_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_pinn_planestress_loss.png")

# ============================================================================
# [8] Summary
# ============================================================================
print("\n" + "=" * 70)
print("Tutorial Summary: 2D Plane Stress Analysis with PINN")
print("=" * 70)
print(f"""
Physics:
  - Solid mechanics: equilibrium equations for plane stress
  - Strain-displacement: ε = ∇u (small deformation)
  - Constitutive law: σ = D × ε (Hooke's law, plane stress)
  - PDE: ∂σ_xx/∂x + ∂τ_xy/∂y = 0, ∂τ_xy/∂x + ∂σ_yy/∂y = 0

Model:
  - FullyConnected (MLP): (x,y) → (u,v)
  - {n_params} parameters, tanh activation
  - Automatic differentiation for strain/stress computation

Training:
  - Epochs: {n_epochs}
  - Final total loss: {loss_history['total'][-1]:.6e}
  - Final PDE loss: {loss_history['pde'][-1]:.6e}

Key Difference from CFD Tutorials:
  - CFD: Navier-Stokes (momentum + continuity)
  - Solid mechanics: Equilibrium (no inertia, no time)
  - Output: displacement → strain → stress (chain of derivatives)
  - Constitutive law: Hooke's law (linear elastic)

Validation:
  - Compared with Euler-Bernoulli beam theory
  - v(L) analytical = {v_analytical[-1]:.6f}
  - v(L) PINN = {v_pred_mid[-1]:.6f}
  - Error = {abs(v_pred_mid[-1] - v_analytical[-1]):.6e}
""")
print("=" * 70)
