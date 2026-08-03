"""
PhysicsNeMo Tutorial: 2D Electrostatics with PINN
===================================================
This tutorial uses PhysicsNeMo's FullyConnected model to implement a PINN
for electromagnetics — specifically, 2D electrostatic analysis. This is an
electromagnetic CAE problem that is not available in PhysicsNeMo by default.

Problem: 2D Electrostatic Potential with Point Charges
- A grounded conducting box contains point charges
- The electric potential φ(x,y) is predicted by a neural network
- The network is trained to satisfy Poisson's equation (no labeled data needed)

Learning objectives:
1. Electromagnetics fundamentals (Coulomb's law, Poisson's equation)
2. How to encode Maxwell's equations as PINN loss
3. Electric field computation from potential
4. Comparison with analytical solution (method of images)

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_pinn_electrostatics.py

=========================================================================
[Physics Background: Electrostatics]
=========================================================================

Electrostatics deals with electric charges at rest. The fundamental equations
are derived from Maxwell's equations with no time variation (∂/∂t = 0).

--- Coulomb's Law (Force between charges) ---

    F = (1 / 4πε₀) × q₁q₂ / r²

where:
    ε₀ = 8.854 × 10⁻¹² F/m  (vacuum permittivity)
    q₁, q₂ = electric charges (Coulombs)
    r = distance between charges

--- Electric Field ---

    E = F / q = (1 / 4πε₀) × q / r²  (radial from positive charge)

The electric field is the force per unit charge. It points away from
positive charges and toward negative charges.

--- Gauss's Law (Differential Form) ---

    ∇·E = ρ / ε₀

where:
    E = electric field vector (V/m)
    ρ = charge density (C/m³)
    ε₀ = permittivity of free space

--- Electric Potential ---

    E = -∇φ

where φ(x,y) is the electric potential (Volts). The electric field points
in the direction of DECREASING potential.

Combining Gauss's law with E = -∇φ:

    ∇²φ = -ρ / ε₀   (Poisson's equation)

For regions with no charge (ρ = 0):
    ∇²φ = 0          (Laplace's equation)

--- Boundary Conditions ---

    Dirichlet: φ = V₀  (prescribed potential, e.g., conductor at fixed voltage)
    Neumann:   -∂φ/∂n = -σ_s / ε₀  (prescribed surface charge)

=========================================================================
[Problem Setup: Point Charge in Grounded Box]
=========================================================================

    φ = 0 (grounded walls)
    +------------------+
    |                  |
    |       + q       |  (point charge at center)
    |                  |
    +------------------+
    φ = 0 (grounded walls)

    Domain: [0, 1] × [0, 1]
    All walls: φ = 0 (grounded conductor)
    Point charge: q at (0.5, 0.5)

    Analytical solution: Method of images
    For a point charge q at (x₀, y₀) in a grounded box [0,a]×[0,b]:
    φ(x,y) = Σ (-1)^(i+j+k+l) × q / (4πε₀ × r_ijkl)

    where r_ijkl is the distance to the image charge at position:
    (±x₀ + 2ia, ±y₀ + 2jb) for i,j = 0, ±1, ±2, ...

    For simplicity, we use a regularized charge (avoid singularity):
    ρ(x,y) = q / (2πσ²) × exp(-r²/2σ²)  (Gaussian charge distribution)
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
print("\n[1] Problem Setup: 2D Electrostatics — Point Charge in Grounded Box")

# Domain
Lx = 1.0  # box width
Ly = 1.0  # box height

# Physical constants (normalized for simplicity)
# In SI: ε₀ = 8.854e-12 F/m, but we use ε₀ = 1 for normalized units
eps_0 = 1.0  # permittivity (normalized)

# Point charge
q_charge = 1.0  # charge magnitude (normalized)
x0, y0 = 0.5, 0.5  # charge location (center)

# Regularization (to avoid singularity at charge location)
sigma_charge = 0.03  # Gaussian width for charge distribution

print(f"   Domain: [{0}, {Lx}] × [{0}, {Ly}]")
print(f"   Permittivity: ε₀ = {eps_0} (normalized)")
print(f"   Charge: q = {q_charge} at ({x0}, {y0})")
print(f"   Regularization: σ = {sigma_charge}")
print(f"   BC: φ = 0 on all walls (grounded)")

# ============================================================================
# [2] Create Neural Network Model
# ============================================================================
print("\n[2] Creating PINN model (FullyConnected)...")

# The network maps (x, y) -> φ (electric potential)
model = FullyConnected(
    in_features=2,        # (x, y)
    out_features=1,       # φ (electric potential)
    layer_size=64,        # neurons per hidden layer
    num_layers=6,         # number of hidden layers
    activation_fn="tanh", # smooth activation (important for 2nd derivatives)
    weight_norm=True,     # weight normalization for training stability
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"   Model: {n_params} parameters")
print(f"   Architecture: 2 -> 64 -> 64 -> 64 -> 64 -> 64 -> 1 (tanh)")

# ============================================================================
# [3] Training Setup
# ============================================================================
print("\n[3] Training setup...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

n_epochs = 5000

# Generate collocation points (interior)
n_interior = 5000
x_interior = torch.rand(n_interior, 1, device=device) * Lx
y_interior = torch.rand(n_interior, 1, device=device) * Ly

# Boundary points (all walls: φ = 0)
n_boundary = 200
# Left wall
x_left = torch.zeros(n_boundary, 1, device=device)
y_left = torch.rand(n_boundary, 1, device=device) * Ly
# Right wall
x_right = torch.full((n_boundary, 1), Lx, device=device)
y_right = torch.rand(n_boundary, 1, device=device) * Ly
# Bottom wall
x_bottom = torch.rand(n_boundary, 1, device=device) * Lx
y_bottom = torch.zeros(n_boundary, 1, device=device)
# Top wall
x_top = torch.rand(n_boundary, 1, device=device) * Lx
y_top = torch.full((n_boundary, 1), Ly, device=device)

# Combine all boundary points
x_bc = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
y_bc = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)

print(f"   Epochs: {n_epochs}")
print(f"   Interior points: {n_interior}")
print(f"   Boundary points: {len(x_bc)} (all walls grounded)")

# ============================================================================
# [4] Physics: Poisson Equation Residual
# ============================================================================
print("\n[4] Defining physics functions...")

def charge_density(x, y, x0, y0, q, sigma):
    """
    Regularized charge density (Gaussian distribution).

    Instead of a point charge (delta function), we use a Gaussian:
        ρ(x,y) = q / (2πσ²) × exp(-((x-x0)² + (y-y0)²) / (2σ²))

    This avoids the singularity at the charge location and makes
    the problem well-posed for neural network training.

    Parameters
    ----------
    x, y : torch.Tensor
        Coordinates
    x0, y0 : float
        Charge center
    q : float
        Total charge
    sigma : float
        Gaussian width

    Returns
    -------
    rho : torch.Tensor
        Charge density at each point
    """
    r2 = (x - x0)**2 + (y - y0)**2
    rho = q / (2 * np.pi * sigma**2) * torch.exp(-r2 / (2 * sigma**2))
    return rho


def poisson_residual(model, x, y):
    """
    Compute Poisson equation residual: ∇²φ + ρ/ε₀ = 0

    The Laplacian of the potential equals minus the charge density
    divided by permittivity:
        ∂²φ/∂x² + ∂²φ/∂y² = -ρ/ε₀

    The residual is:
        R = ∂²φ/∂x² + ∂²φ/∂y² + ρ/ε₀

    If the PDE is satisfied, R = 0.

    Parameters
    ----------
    model : nn.Module
        Neural network mapping (x, y) -> φ
    x, y : torch.Tensor
        Coordinates (require_grad=True)

    Returns
    -------
    residual : torch.Tensor
        PDE residual (should be zero)
    phi : torch.Tensor
        Electric potential
    """
    x.requires_grad_(True)
    y.requires_grad_(True)

    # Forward pass: predict potential
    phi = model(torch.cat([x, y], dim=1))

    # First derivatives
    phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi),
                                create_graph=True)[0]
    phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi),
                                create_graph=True)[0]

    # Second derivatives (Laplacian)
    phi_xx = torch.autograd.grad(phi_x, x, grad_outputs=torch.ones_like(phi_x),
                                 create_graph=True)[0]
    phi_yy = torch.autograd.grad(phi_y, y, grad_outputs=torch.ones_like(phi_y),
                                 create_graph=True)[0]

    # Charge density
    rho = charge_density(x, y, x0, y0, q_charge, sigma_charge)

    # Poisson residual: ∇²φ + ρ/ε₀ = 0
    laplacian = phi_xx + phi_yy
    residual = laplacian + rho / eps_0

    return residual, phi


def compute_electric_field(model, x, y):
    """
    Compute electric field from potential: E = -∇φ

    Parameters
    ----------
    model : nn.Module
        Neural network
    x, y : torch.Tensor
        Coordinates (require_grad=True)

    Returns
    -------
    Ex, Ey : torch.Tensor
        Electric field components
    phi : torch.Tensor
        Electric potential
    """
    x.requires_grad_(True)
    y.requires_grad_(True)

    phi = model(torch.cat([x, y], dim=1))

    phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi),
                                create_graph=True)[0]
    phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi),
                                create_graph=True)[0]

    # E = -∇φ
    Ex = -phi_x
    Ey = -phi_y

    return Ex, Ey, phi


# ============================================================================
# [5] Training Loop
# ============================================================================
print("\n[5] Training PINN...")
print("    PDE: ∇²φ = -ρ/ε₀  (Poisson's equation)")
print("    BC:  φ = 0 on all walls (grounded conductor)")
print("    Loss = λ_pde * L_pde + λ_bc * L_bc\n")

# Loss weights
lambda_pde = 1.0
lambda_bc = 50.0  # Dirichlet BC needs strong enforcement

loss_history = {'total': [], 'pde': [], 'bc': []}

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    # --- PDE Loss (Poisson equation in interior) ---
    res, _ = poisson_residual(model, x_interior, y_interior)
    loss_pde = torch.mean(res**2)

    # --- BC Loss (φ = 0 on all walls) ---
    phi_bc = model(torch.cat([x_bc, y_bc], dim=1))
    loss_bc = torch.mean(phi_bc**2)

    # Total loss
    loss = lambda_pde * loss_pde + lambda_bc * loss_bc

    loss.backward()
    optimizer.step()

    # Record
    loss_history['total'].append(loss.item())
    loss_history['pde'].append(loss_pde.item())
    loss_history['bc'].append(loss_bc.item())

    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1:5d}/{n_epochs} | Total: {loss.item():.6e} | "
              f"PDE: {loss_pde.item():.6e} | BC: {loss_bc.item():.6e}")

    scheduler.step()

print("\n   Training complete!")

# ============================================================================
# [6] Analytical Solution (Method of Images)
# ============================================================================
print("\n[6] Computing analytical solution (method of images)...")

def analytical_potential(x, y, x0, y0, q, Lx, Ly, n_images=5):
    """
    Analytical potential for point charge in grounded box using method of images.

    For a point charge q at (x0, y0) in a grounded box [0, Lx] × [0, Ly],
    the potential is computed by summing contributions from image charges:

    φ(x,y) = Σ (-1)^(i+j) × q / (4πε₀ × r_ij)

    where image charges are at positions:
    (2iLx ± x0, 2jLy ± y0) for i, j = -n_images, ..., n_images

    The sign alternates to satisfy the boundary condition φ = 0 on walls.

    Parameters
    ----------
    x, y : float or array
        Evaluation points
    x0, y0 : float
        Charge location
    q : float
        Charge magnitude
    Lx, Ly : float
        Box dimensions
    n_images : int
        Number of image charge layers (higher = more accurate)

    Returns
    -------
    phi : array
        Electric potential
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    phi = np.zeros_like(x)

    for i in range(-n_images, n_images + 1):
        for j in range(-n_images, n_images + 1):
            # Image charge positions
            for sx in [1, -1]:
                for sy in [1, -1]:
                    xi = 2 * i * Lx + sx * x0
                    yj = 2 * j * Ly + sy * y0

                    r = np.sqrt((x - xi)**2 + (y - yj)**2)
                    r = np.maximum(r, 1e-10)  # avoid singularity

                    # Sign: (-1)^(i+j) for alternating images
                    sign = (-1)**(abs(i) + abs(j))
                    if sx < 0:
                        sign *= -1
                    if sy < 0:
                        sign *= -1

                    phi += sign * q / (4 * np.pi * 1.0 * r)

    return phi


# ============================================================================
# [7] Visualization
# ============================================================================
print("\n[7] Visualizing results...")

# Create evaluation grid
n_eval = 100
x_eval = np.linspace(0, Lx, n_eval)
y_eval = np.linspace(0, Ly, n_eval)
X, Y = np.meshgrid(x_eval, y_eval)

# Predict potential
model.eval()
with torch.no_grad():
    xy_grid = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32).to(device)
    phi_pred = model(xy_grid)
    PHI_pred = phi_pred[:, 0].cpu().numpy().reshape(X.shape)

# Compute electric field (requires grad)
x_grid = torch.tensor(X.ravel(), dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
y_grid = torch.tensor(Y.ravel(), dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
Ex_g, Ey_g, _ = compute_electric_field(model, x_grid, y_grid)

EX = Ex_g.detach().cpu().numpy().reshape(X.shape)
EY = Ey_g.detach().cpu().numpy().reshape(X.shape)

# Analytical solution
phi_analytical = analytical_potential(X, Y, x0, y0, q_charge, Lx, Ly, n_images=5)

# --- Figure 1: Potential and Electric Field ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# PINN potential
ax = axes[0, 0]
im = ax.contourf(X, Y, PHI_pred, levels=50, cmap='RdYlBu_r')
ax.plot(x0, y0, 'k+', markersize=15, markeredgewidth=3, label='Charge')
ax.set_title('PINN: Electric Potential φ(x,y)', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='φ (V)')
ax.legend(fontsize=10)

# Analytical potential
ax = axes[0, 1]
im = ax.contourf(X, Y, phi_analytical, levels=50, cmap='RdYlBu_r')
ax.plot(x0, y0, 'k+', markersize=15, markeredgewidth=3, label='Charge')
ax.set_title('Analytical: Electric Potential φ(x,y)\n(Method of Images)', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='φ (V)')
ax.legend(fontsize=10)

# Electric field magnitude
ax = axes[1, 0]
E_mag = np.sqrt(EX**2 + EY**2)
im = ax.contourf(X, Y, E_mag, levels=50, cmap='hot_r')
ax.plot(x0, y0, 'w+', markersize=15, markeredgewidth=3, label='Charge')
ax.set_title('Electric Field Magnitude |E|', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='|E| (V/m)')
ax.legend(fontsize=10)

# Electric field vectors
ax = axes[1, 1]
ax.contourf(X, Y, PHI_pred, levels=20, cmap='RdYlBu_r', alpha=0.5)
step = 4
ax.quiver(X[::step, ::step], Y[::step, ::step],
           EX[::step, ::step], EY[::step, ::step],
           color='black', scale=200, width=0.005)
ax.plot(x0, y0, 'r+', markersize=15, markeredgewidth=3, label='Charge')
ax.set_title('Electric Field Vectors E = -∇φ', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=10)

plt.suptitle('PINN: 2D Electrostatics — Point Charge in Grounded Box\n'
             '(Electromagnetics — Not Available in PhysicsNeMo by Default)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_pinn_electrostatics.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_pinn_electrostatics.png")

# --- Figure 2: Comparison with Analytical Solution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Potential along y = 0.5 (horizontal cut through charge)
ax = axes[0]
mid_row = n_eval // 2
ax.plot(x_eval, PHI_pred[mid_row, :], 'b-', linewidth=2, label='PINN')
ax.plot(x_eval, phi_analytical[mid_row, :], 'r--', linewidth=2, label='Analytical (Images)')
ax.set_xlabel('x (along y=0.5)', fontsize=12)
ax.set_ylabel('φ (V)', fontsize=12)
ax.set_title('Potential along y = 0.5', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Potential along x = 0.5 (vertical cut through charge)
ax = axes[1]
mid_col = n_eval // 2
ax.plot(y_eval, PHI_pred[:, mid_col], 'b-', linewidth=2, label='PINN')
ax.plot(y_eval, phi_analytical[:, mid_col], 'r--', linewidth=2, label='Analytical (Images)')
ax.set_xlabel('y (along x=0.5)', fontsize=12)
ax.set_ylabel('φ (V)', fontsize=12)
ax.set_title('Potential along x = 0.5', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Error map
ax = axes[2]
error = np.abs(PHI_pred - phi_analytical)
im = ax.contourf(X, Y, error, levels=50, cmap='hot_r')
ax.plot(x0, y0, 'w+', markersize=15, markeredgewidth=3)
ax.set_title('Absolute Error |φ_PINN - φ_analytical|', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='|error| (V)')

plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_pinn_electrostatics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_pinn_electrostatics_comparison.png")

# --- Figure 3: Training Loss ---
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(loss_history['total'], 'k-', label='Total', linewidth=1.5)
ax.plot(loss_history['pde'], 'b-', label='PDE (Poisson)', alpha=0.7)
ax.plot(loss_history['bc'], 'r-', label='BC (grounded walls)', alpha=0.7)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('PINN Training Loss: Electrostatics', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_pinn_electrostatics_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_pinn_electrostatics_loss.png")

# ============================================================================
# [8] Summary
# ============================================================================
print("\n" + "=" * 70)
print("Tutorial Summary: 2D Electrostatics with PINN")
print("=" * 70)
print(f"""
Physics:
  - Electrostatics: Poisson's equation ∇²φ = -ρ/ε₀
  - Electric field: E = -∇φ (from potential)
  - Gauss's law: ∇·E = ρ/ε₀ (differential form)
  - BC: φ = 0 on all walls (grounded conductor)

Model:
  - FullyConnected (MLP): (x,y) → φ
  - {n_params} parameters, tanh activation
  - 2nd-order autograd for Laplacian computation

Training:
  - Epochs: {n_epochs}
  - Final total loss: {loss_history['total'][-1]:.6e}
  - Final PDE loss: {loss_history['pde'][-1]:.6e}
  - Final BC loss: {loss_history['bc'][-1]:.6e}

Key Difference from CFD Tutorials:
  - CFD: Navier-Stokes (momentum + continuity)
  - EM: Poisson/Laplace equation (elliptic, scalar field)
  - Output: electric potential φ (scalar) → E = -∇φ
  - Source: charge density ρ (instead of force/pressure)

Validation:
  - Analytical: method of images (5 layers)
  - Max error: {np.max(error):.6e} V
  - Mean error: {np.mean(error):.6e} V
  - Error concentrated near charge (singularity region)
""")
print("=" * 70)
