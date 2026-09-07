"""
PhysicsNeMo PINN Tutorial: Helmholtz Equation (Acoustic Wave Scattering)
========================================================================
2D Acoustic Wave Scattering from a Cylinder

Existing tutorials cover:
  - Wave equation (FNO, time-domain, real-valued)
  - Electrostatics (Poisson, static, real-valued)
  - Burgers, Navier-Stokes, etc. (all real-valued, time-domain)

THIS tutorial:
  - Helmholtz equation (frequency-domain, COMPLEX-valued field)
  - Acoustic scattering: plane wave hits a sound-hard cylinder
  - Split into real (u_r) and imaginary (u_i) parts
  - Sommerfeld radiation condition (absorbing BC for outgoing waves)

Key difference from existing tutorials:
  ┌──────────────────────┬──────────────────────────┐
  │ Existing              │ THIS (Helmholtz)         │
  ├──────────────────────┼──────────────────────────┤
  │ Real-valued fields    │ Complex field (Re + Im)  │
  │ Time-domain           │ Frequency-domain         │
  │ Dirichlet/Neumann BC  │ Sommerfeld radiation BC  │
  │ No scattering         │ Scattering from object   │
  │ Single PDE            │ Coupled Re/Im equations  │
  └──────────────────────┴──────────────────────────┘

Helmholtz Equation:
  ∇²u + k²u = 0  (outside scatterer)

  where u = u_incident + u_scattered
    u_incident = exp(ikx)  (plane wave)
    u_scattered satisfies Sommerfeld radiation condition:
      lim(r→∞) r^{1/2}(∂u_s/∂r - iku_s) = 0

  Sound-hard cylinder BC: ∂u/∂n = 0 on cylinder surface

  Split into real/imaginary:
    ∇²u_r + k²u_r = 0
    ∇²u_i + k²u_i = 0
    BC: ∂u_r/∂n = -∂u_inc_r/∂n,  ∂u_i/∂n = -∂u_inc_i/∂n

Author: PhysicsNeMo Tutorial
Date: 2026-09-07
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo PINN Tutorial: Helmholtz Equation (Acoustic Scattering)")
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
K_WAVE = 2.0 * np.pi  # Wavenumber k = 2π/λ, λ=1 → k=2π
CYLINDER_R = 0.5       # Cylinder radius
DOMAIN_R = 3.0          # Outer boundary radius

print(f"\n[1] Problem Parameters:")
print(f"  Wavenumber k = {K_WAVE:.4f}")
print(f"  Wavelength λ = {2*np.pi/K_WAVE:.4f}")
print(f"  Cylinder radius a = {CYLINDER_R}")
print(f"  Domain radius R = {DOMAIN_R}")
print(f"  Incident wave: plane wave exp(ikx)")

# ============================================================
# [2] Analytical Solution (for validation)
# ============================================================
def analytical_solution(r, theta, k, a):
    """
    Analytical solution for plane wave scattering by sound-hard cylinder.

    Total field: u = u_inc + u_sc
    u_inc = exp(ik r cos θ)
    u_sc = Σ_n (-i)^n [J_n'(ka)/H_n'(ka)] H_n(kr) cos(nθ)

    where J_n = Bessel, H_n = Hankel, ' = derivative w.r.t. argument
    """
    from scipy.special import jv, hankel1, jvp, h1vp

    u_total = np.zeros_like(r, dtype=complex)

    # Incident wave
    u_inc = np.exp(1j * k * r * np.cos(theta))
    u_total = u_inc.copy()

    # Scattered field
    u_sc = np.zeros_like(r, dtype=complex)
    n_max = 30
    for n in range(-n_max, n_max + 1):
        # Coefficient: A_n = -(-i)^n * J_n'(ka) / H_n'(ka)
        Jn_prime = jvp(n, k * a)
        Hn_prime = h1vp(n, k * a)
        A_n = -((-1j)**n) * Jn_prime / Hn_prime
        u_sc += A_n * hankel1(n, k * r) * np.exp(1j * n * theta)

    u_total = u_inc + u_sc
    return u_total

print(f"\n[2] Analytical solution: Bessel/Hankel series (n_max=30)")

# ============================================================
# [3] PINN Model (Complex-valued via 2 outputs)
# ============================================================
class HelmholtzPINN(nn.Module):
    """
    PINN for Helmholtz equation with complex-valued field.

    Input: (x, y) — spatial coordinates
    Output: (u_r, u_i) — real and imaginary parts of total field

    Architecture: 2 → 64 → 64 → 64 → 64 → 2

    The network predicts the TOTAL field u = u_incident + u_scattered.
    We subtract the known incident field to get the scattered field.
    """
    def __init__(self, layers=[2, 64, 64, 64, 64, 2]):
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
        for i in range(len(self.layers) - 2):
            x = self.activation(self.linears[i](x))
        x = self.linears[-1](x)
        return x  # [N, 2] — (u_r, u_i)

print(f"\n[3] PINN Model: HelmholtzPINN")
print(f"  Architecture: 2 → 64 → 64 → 64 → 64 → 2")
print(f"  Input: (x, y)")
print(f"  Output: (u_real, u_imag) — complex field")

# ============================================================
# [4] Collocation Points & Boundary Conditions
# ============================================================
N_INTERIOR = 3000
N_CYLINDER = 300   # Points on cylinder surface
N_OUTER = 300      # Points on outer boundary

# Interior collocation points (annulus: a < r < R)
r_rand = torch.rand(N_INTERIOR, 1, device=device) * (DOMAIN_R - CYLINDER_R) + CYLINDER_R
theta_rand = torch.rand(N_INTERIOR, 1, device=device) * 2 * np.pi
x_interior = r_rand * torch.cos(theta_rand)
y_interior = r_rand * torch.sin(theta_rand)
xy_interior = torch.cat([x_interior, y_interior], dim=1)
xy_interior.requires_grad_(True)

# Cylinder surface (r = a, sound-hard BC: ∂u/∂n = 0)
theta_cyl = torch.linspace(0, 2 * np.pi, N_CYLINDER, device=device).view(-1, 1)
x_cyl = CYLINDER_R * torch.cos(theta_cyl)
y_cyl = CYLINDER_R * torch.sin(theta_cyl)
xy_cyl = torch.cat([x_cyl, y_cyl], dim=1)
xy_cyl.requires_grad_(True)

# Outer boundary (r = R, Sommerfeld radiation condition)
theta_out = torch.linspace(0, 2 * np.pi, N_OUTER, device=device).view(-1, 1)
x_out = DOMAIN_R * torch.cos(theta_out)
y_out = DOMAIN_R * torch.sin(theta_out)
xy_out = torch.cat([x_out, y_out], dim=1)
xy_out.requires_grad_(True)

# Reference data points (from analytical solution)
N_REF = 500
r_ref = torch.rand(N_REF, 1, device=device) * (DOMAIN_R - CYLINDER_R - 0.1) + CYLINDER_R + 0.05
theta_ref = torch.rand(N_REF, 1, device=device) * 2 * np.pi
x_ref = r_ref * torch.cos(theta_ref)
y_ref = r_ref * torch.sin(theta_ref)
xy_ref = torch.cat([x_ref, y_ref], dim=1)

# Analytical solution at reference points
r_np = r_ref.cpu().numpy().flatten()
theta_np = theta_ref.cpu().numpy().flatten()
u_analytical = analytical_solution(r_np, theta_np, K_WAVE, CYLINDER_R)
u_r_ref = torch.tensor(u_analytical.real, dtype=torch.float32, device=device).view(-1, 1)
u_i_ref = torch.tensor(u_analytical.imag, dtype=torch.float32, device=device).view(-1, 1)

print(f"\n[4] Collocation Points:")
print(f"  Interior (annulus): {N_INTERIOR}")
print(f"  Cylinder surface: {N_CYLINDER}")
print(f"  Outer boundary: {N_OUTER}")
print(f"  Reference data: {N_REF}")

# ============================================================
# [5] Loss Functions
# ============================================================
def compute_pde_residual(model, xy):
    """
    Helmholtz equation residual:
      ∇²u_r + k²u_r = 0
      ∇²u_i + k²u_i = 0

    The network predicts the TOTAL field (incident + scattered).
    Since the incident field exp(ikx) also satisfies Helmholtz,
    the total field satisfies ∇²u + k²u = 0 everywhere.
    """
    u = model(xy)  # [N, 2]
    u_r = u[:, 0:1]
    u_i = u[:, 1:2]

    # Compute Laplacian via autograd
    # First derivatives
    grad_r = torch.autograd.grad(u_r, xy, grad_outputs=torch.ones_like(u_r),
                                  create_graph=True, retain_graph=True)[0]  # [N, 2]
    grad_i = torch.autograd.grad(u_i, xy, grad_outputs=torch.ones_like(u_i),
                                  create_graph=True, retain_graph=True)[0]  # [N, 2]

    # Second derivatives (Laplacian)
    u_r_xx = torch.autograd.grad(grad_r[:, 0], xy, grad_outputs=torch.ones_like(grad_r[:, 0]),
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
    u_r_yy = torch.autograd.grad(grad_r[:, 1], xy, grad_outputs=torch.ones_like(grad_r[:, 1]),
                                  create_graph=True, retain_graph=True)[0][:, 1:2]

    u_i_xx = torch.autograd.grad(grad_i[:, 0], xy, grad_outputs=torch.ones_like(grad_i[:, 0]),
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
    u_i_yy = torch.autograd.grad(grad_i[:, 1], xy, grad_outputs=torch.ones_like(grad_i[:, 1]),
                                  create_graph=True, retain_graph=True)[0][:, 1:2]

    # Residual: ∇²u + k²u = 0
    lap_r = u_r_xx + u_r_yy
    lap_i = u_i_xx + u_i_yy
    residual_r = lap_r + K_WAVE**2 * u_r
    residual_i = lap_i + K_WAVE**2 * u_i

    return residual_r, residual_i


def compute_cylinder_bc(model, xy_cyl):
    """
    Sound-hard cylinder BC: ∂u/∂n = 0 (Neumann)
    Normal direction on cylinder = radial direction (outward from origin).
    ∂u/∂r = 0 at r = a.

    Since u = u_inc + u_sc, and ∂u_inc/∂r ≠ 0,
    we need ∂u_sc/∂r = -∂u_inc/∂r on cylinder.
    But the network predicts total u, so ∂u/∂r = 0.
    """
    u = model(xy_cyl)
    u_r = u[:, 0:1]
    u_i = u[:, 1:2]

    # Radial derivative
    grad_r = torch.autograd.grad(u_r, xy_cyl, grad_outputs=torch.ones_like(u_r),
                                  create_graph=True, retain_graph=True)[0]
    grad_i = torch.autograd.grad(u_i, xy_cyl, grad_outputs=torch.ones_like(u_i),
                                  create_graph=True, retain_graph=True)[0]

    # Normal direction (radial): n = (x, y) / |r|
    x = xy_cyl[:, 0:1]
    y = xy_cyl[:, 1:2]
    r = torch.sqrt(x**2 + y**2 + 1e-10)
    nx = x / r
    ny = y / r

    # ∂u/∂n = grad · n
    du_r_dn = grad_r[:, 0:1] * nx + grad_r[:, 1:2] * ny
    du_i_dn = grad_i[:, 0:1] * nx + grad_i[:, 1:2] * ny

    return (du_r_dn**2 + du_i_dn**2).mean()


def compute_outer_bc(model, xy_out):
    """
    Sommerfeld radiation condition (approximate):
      ∂u_s/∂r - iku_s ≈ 0  at r = R

    For total field u = u_inc + u_sc:
      ∂u/∂r - ik*u = ∂u_inc/∂r - ik*u_inc  (since u_sc satisfies Sommerfeld)
      ∂u_inc/∂r = ik cos(θ) exp(ikx) = ik cos(θ) u_inc
      So: ∂u/∂r - ik*u = ik(cos(θ) - 1) * u_inc

    We enforce: ∂u/∂r - ik*u = ik(cos(θ) - 1) * u_inc
    """
    u = model(xy_out)
    u_r = u[:, 0:1]
    u_i = u[:, 1:2]

    # Radial derivative
    grad_r = torch.autograd.grad(u_r, xy_out, grad_outputs=torch.ones_like(u_r),
                                  create_graph=True, retain_graph=True)[0]
    grad_i = torch.autograd.grad(u_i, xy_out, grad_outputs=torch.ones_like(u_i),
                                  create_graph=True, retain_graph=True)[0]

    x = xy_out[:, 0:1]
    y = xy_out[:, 1:2]
    r = torch.sqrt(x**2 + y**2 + 1e-10)
    nx = x / r
    ny = y / r

    du_r_dr = grad_r[:, 0:1] * nx + grad_r[:, 1:2] * ny
    du_i_dr = grad_i[:, 0:1] * nx + grad_i[:, 1:2] * ny

    # Incident field at outer boundary
    u_inc_r = torch.cos(K_WAVE * x)  # Re(exp(ikx))
    u_inc_i = torch.sin(K_WAVE * x)  # Im(exp(ikx))

    # RHS: ∂u_inc/∂r - ik*u_inc = ik(cos θ - 1) u_inc
    cos_theta = x / r
    rhs_r = K_WAVE * (cos_theta - 1) * (-u_inc_i)  # Re(ik(cosθ-1) * u_inc)
    rhs_i = K_WAVE * (cos_theta - 1) * u_inc_r     # Im(ik(cosθ-1) * u_inc)

    # LHS: ∂u/∂r - ik*u
    lhs_r = du_r_dr + K_WAVE * u_i   # Re(∂u/∂r - ik*u) = ∂u_r/∂r + k*u_i
    lhs_i = du_i_dr - K_WAVE * u_r   # Im(∂u/∂r - ik*u) = ∂u_i/∂r - k*u_r

    residual_r = lhs_r - rhs_r
    residual_i = lhs_i - rhs_i

    return (residual_r**2 + residual_i**2).mean()


def compute_data_loss(model, xy_ref, u_r_ref, u_i_ref):
    """Data loss against analytical solution."""
    u = model(xy_ref)
    loss_r = ((u[:, 0:1] - u_r_ref)**2).mean()
    loss_i = ((u[:, 1:2] - u_i_ref)**2).mean()
    return loss_r + loss_i

print(f"\n[5] Loss Functions:")
print(f"  PDE: Helmholtz (∇²u + k²u = 0, Re + Im)")
print(f"  BC1: Sound-hard cylinder (∂u/∂n = 0)")
print(f"  BC2: Sommerfeld radiation (∂u/∂r - iku = ...)")
print(f"  Data: Analytical solution (Bessel/Hankel)")

# ============================================================
# [6] Training
# ============================================================
model = HelmholtzPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

N_EPOCHS = 5000
LAMBDA_PDE = 1.0
LAMBDA_CYL = 10.0
LAMBDA_OUT = 5.0
LAMBDA_DATA = 2.0

print(f"\n[6] Training:")
print(f"  Epochs: {N_EPOCHS}")
print(f"  λ_PDE={LAMBDA_PDE}, λ_CYL={LAMBDA_CYL}, λ_OUT={LAMBDA_OUT}, λ_DATA={LAMBDA_DATA}")

loss_history = {'total': [], 'pde': [], 'cyl': [], 'out': [], 'data': []}
t_start = time.time()

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()

    # PDE loss
    res_r, res_i = compute_pde_residual(model, xy_interior)
    loss_pde = (res_r**2 + res_i**2).mean()

    # Cylinder BC
    loss_cyl = compute_cylinder_bc(model, xy_cyl)

    # Outer BC (Sommerfeld)
    loss_out = compute_outer_bc(model, xy_out)

    # Data loss
    loss_data = compute_data_loss(model, xy_ref, u_r_ref, u_i_ref)

    # Total
    loss = (LAMBDA_PDE * loss_pde + LAMBDA_CYL * loss_cyl +
            LAMBDA_OUT * loss_out + LAMBDA_DATA * loss_data)

    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history['total'].append(loss.item())
    loss_history['pde'].append(loss_pde.item())
    loss_history['cyl'].append(loss_cyl.item())
    loss_history['out'].append(loss_out.item())
    loss_history['data'].append(loss_data.item())

    if (epoch + 1) % 500 == 0:
        elapsed = time.time() - t_start
        print(f"  Epoch {epoch+1:5d}/{N_EPOCHS} | "
              f"Total: {loss.item():.6e} | "
              f"PDE: {loss_pde.item():.6e} | "
              f"Cyl: {loss_cyl.item():.6e} | "
              f"Out: {loss_out.item():.6e} | "
              f"Data: {loss_data.item():.6e} | "
              f"Time: {elapsed:.1f}s")

print(f"\n  Training complete! Total time: {time.time()-t_start:.1f}s")

# ============================================================
# [7] Evaluation & Visualization
# ============================================================
print(f"\n[7] Evaluation & Visualization...")

# Predict on fine grid (Cartesian, excluding cylinder)
NX_EVAL, NY_EVAL = 100, 100
x_eval = torch.linspace(-DOMAIN_R, DOMAIN_R, NX_EVAL, device=device)
y_eval = torch.linspace(-DOMAIN_R, DOMAIN_R, NY_EVAL, device=device)
X_eval, Y_eval = torch.meshgrid(x_eval, y_eval, indexing='ij')
xy_eval = torch.cat([X_eval.reshape(-1, 1), Y_eval.reshape(-1, 1)], dim=1)

# Mask out cylinder interior
r_eval = torch.sqrt(X_eval**2 + Y_eval**2)
mask = r_eval > CYLINDER_R

model.eval()
with torch.no_grad():
    u_pred = model(xy_eval)
    u_r_pred = u_pred[:, 0].cpu().numpy().reshape(NX_EVAL, NY_EVAL)
    u_i_pred = u_pred[:, 1].cpu().numpy().reshape(NX_EVAL, NY_EVAL)

# Mask cylinder
u_r_pred_masked = np.where(mask.cpu().numpy(), u_r_pred, np.nan)
u_i_pred_masked = np.where(mask.cpu().numpy(), u_i_pred, np.nan)
u_mag_pred = np.sqrt(u_r_pred_masked**2 + u_i_pred_masked**2)

# Analytical solution on same grid
r_np = r_eval.cpu().numpy()
theta_np = np.arctan2(Y_eval.cpu().numpy(), X_eval.cpu().numpy())
u_analytical_grid = analytical_solution(r_np, theta_np, K_WAVE, CYLINDER_R)
u_r_ana = u_analytical_grid.real
u_i_ana = u_analytical_grid.imag
u_r_ana_masked = np.where(mask.cpu().numpy(), u_r_ana, np.nan)
u_i_ana_masked = np.where(mask.cpu().numpy(), u_i_ana, np.nan)
u_mag_ana = np.sqrt(u_r_ana_masked**2 + u_i_ana_masked**2)

# --- Figure 1: Real part comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
im = ax.imshow(u_r_ana_masked.T, origin='lower', extent=[-DOMAIN_R, DOMAIN_R, -DOMAIN_R, DOMAIN_R],
               cmap='RdBu_r', vmin=-1.5, vmax=1.5)
ax.add_patch(Circle((0, 0), CYLINDER_R, color='black', fill=True))
ax.set_title('Analytical Re(u)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[1]
im = ax.imshow(u_r_pred_masked.T, origin='lower', extent=[-DOMAIN_R, DOMAIN_R, -DOMAIN_R, DOMAIN_R],
               cmap='RdBu_r', vmin=-1.5, vmax=1.5)
ax.add_patch(Circle((0, 0), CYLINDER_R, color='black', fill=True))
ax.set_title('PINN Re(u)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[2]
error = np.abs(u_r_ana_masked - u_r_pred_masked)
im = ax.imshow(error.T, origin='lower', extent=[-DOMAIN_R, DOMAIN_R, -DOMAIN_R, DOMAIN_R],
               cmap='hot', vmin=0, vmax=0.3)
ax.add_patch(Circle((0, 0), CYLINDER_R, color='black', fill=True))
ax.set_title('|Re(u) Error|', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

plt.suptitle('Helmholtz PINN: Real Part of Acoustic Field', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'helmholtz_real.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: helmholtz_real.png")

# --- Figure 2: Imaginary part comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
im = ax.imshow(u_i_ana_masked.T, origin='lower', extent=[-DOMAIN_R, DOMAIN_R, -DOMAIN_R, DOMAIN_R],
               cmap='RdBu_r', vmin=-1.5, vmax=1.5)
ax.add_patch(Circle((0, 0), CYLINDER_R, color='black', fill=True))
ax.set_title('Analytical Im(u)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[1]
im = ax.imshow(u_i_pred_masked.T, origin='lower', extent=[-DOMAIN_R, DOMAIN_R, -DOMAIN_R, DOMAIN_R],
               cmap='RdBu_r', vmin=-1.5, vmax=1.5)
ax.add_patch(Circle((0, 0), CYLINDER_R, color='black', fill=True))
ax.set_title('PINN Im(u)', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[2]
error = np.abs(u_i_ana_masked - u_i_pred_masked)
im = ax.imshow(error.T, origin='lower', extent=[-DOMAIN_R, DOMAIN_R, -DOMAIN_R, DOMAIN_R],
               cmap='hot', vmin=0, vmax=0.3)
ax.add_patch(Circle((0, 0), CYLINDER_R, color='black', fill=True))
ax.set_title('|Im(u) Error|', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

plt.suptitle('Helmholtz PINN: Imaginary Part of Acoustic Field', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'helmholtz_imag.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: helmholtz_imag.png")

# --- Figure 3: Magnitude comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
im = ax.imshow(u_mag_ana.T, origin='lower', extent=[-DOMAIN_R, DOMAIN_R, -DOMAIN_R, DOMAIN_R],
               cmap='viridis', vmin=0, vmax=2.5)
ax.add_patch(Circle((0, 0), CYLINDER_R, color='black', fill=True))
ax.set_title('Analytical |u|', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[1]
im = ax.imshow(u_mag_pred.T, origin='lower', extent=[-DOMAIN_R, DOMAIN_R, -DOMAIN_R, DOMAIN_R],
               cmap='viridis', vmin=0, vmax=2.5)
ax.add_patch(Circle((0, 0), CYLINDER_R, color='black', fill=True))
ax.set_title('PINN |u|', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

ax = axes[2]
error = np.abs(u_mag_ana - u_mag_pred)
im = ax.imshow(error.T, origin='lower', extent=[-DOMAIN_R, DOMAIN_R, -DOMAIN_R, DOMAIN_R],
               cmap='hot', vmin=0, vmax=0.3)
ax.add_patch(Circle((0, 0), CYLINDER_R, color='black', fill=True))
ax.set_title('||u| Error|', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, shrink=0.6)

plt.suptitle('Helmholtz PINN: Field Magnitude |u|', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'helmholtz_magnitude.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: helmholtz_magnitude.png")

# --- Figure 4: Loss history ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(loss_history['total'], label='Total', linewidth=2)
ax.semilogy(loss_history['pde'], label='PDE (Helmholtz)', alpha=0.7)
ax.semilogy(loss_history['cyl'], label='Cylinder BC', alpha=0.7)
ax.semilogy(loss_history['out'], label='Sommerfeld BC', alpha=0.7)
ax.semilogy(loss_history['data'], label='Data', alpha=0.7)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Helmholtz PINN Training Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'helmholtz_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: helmholtz_loss.png")

# --- Figure 5: Concept diagram ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(-DOMAIN_R - 0.5, DOMAIN_R + 0.5)
ax.set_ylim(-DOMAIN_R - 0.5, DOMAIN_R + 0.5)
# Cylinder
ax.add_patch(Circle((0, 0), CYLINDER_R, color='gray', fill=True, alpha=0.8))
# Incident wave arrows
for y_arrow in np.linspace(-2, 2, 8):
    ax.annotate('', xy=(0.2, y_arrow), xytext=(-2.5, y_arrow),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.set_title('Acoustic Scattering Setup', fontsize=13, fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.text(-2.5, 2.5, 'Incident wave\n→ → →', fontsize=11, color='blue')
ax.text(0, 0, 'Cylinder\n(sound-hard)', fontsize=9, ha='center', va='center', color='white')
# Outer boundary
ax.add_patch(Circle((0, 0), DOMAIN_R, color='red', fill=False, linewidth=1.5, linestyle='--'))
ax.text(2.5, 2.5, 'Sommerfeld\nBC', fontsize=9, color='red')

ax = axes[1]
ax.axis('off')
concept_text = (
    "Helmholtz Equation (Frequency-Domain):\n\n"
    "  ∇²u + k²u = 0\n\n"
    "  u = u_incident + u_scattered\n"
    "  u_incident = exp(ikx)  (plane wave)\n\n"
    "Complex field: u = u_r + i·u_i\n\n"
    "  ∇²u_r + k²u_r = 0\n"
    "  ∇²u_i + k²u_i = 0\n\n"
    "Boundary Conditions:\n"
    "  Cylinder: ∂u/∂n = 0  (sound-hard)\n"
    "  Outer: ∂u_s/∂r - iku_s = 0  (Sommerfeld)\n\n"
    "vs. Existing Wave Equation (FNO):\n\n"
    "  Time-domain: u_tt = c²∇²u\n"
    "  Real-valued only\n"
    "  No scattering, no radiation BC"
)
ax.text(0.05, 0.95, concept_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Helmholtz vs Wave Equation', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'helmholtz_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: helmholtz_concept.png")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("[8] Summary")
print("=" * 70)

# Compute relative L2 error
mask_flat = mask.cpu().numpy().flatten()
rel_err_r = np.linalg.norm(u_r_pred.flatten()[mask_flat] - u_r_ana.flatten()[mask_flat]) / \
            (np.linalg.norm(u_r_ana.flatten()[mask_flat]) + 1e-10)
rel_err_i = np.linalg.norm(u_i_pred.flatten()[mask_flat] - u_i_ana.flatten()[mask_flat]) / \
            (np.linalg.norm(u_i_ana.flatten()[mask_flat]) + 1e-10)
rel_err_mag = np.linalg.norm(np.sqrt(u_r_pred.flatten()[mask_flat]**2 + u_i_pred.flatten()[mask_flat]**2) -
                              np.sqrt(u_r_ana.flatten()[mask_flat]**2 + u_i_ana.flatten()[mask_flat]**2)) / \
               (np.linalg.norm(np.sqrt(u_r_ana.flatten()[mask_flat]**2 + u_i_ana.flatten()[mask_flat]**2)) + 1e-10)

print(f"\n  Method: PINN for Helmholtz Equation")
print(f"  Physics: Acoustic wave scattering (frequency-domain)")
print(f"  Field: Complex-valued (Re + Im)")
print(f"  Wavenumber k = {K_WAVE:.4f}")
print(f"  Scatterer: Sound-hard cylinder (r={CYLINDER_R})")
print(f"")
print(f"  Relative L2 Error (vs analytical):")
print(f"    Re(u): {rel_err_r:.4f} ({rel_err_r*100:.2f}%)")
print(f"    Im(u): {rel_err_i:.4f} ({rel_err_i*100:.2f}%)")
print(f"    |u|:   {rel_err_mag:.4f} ({rel_err_mag*100:.2f}%)")
print(f"")
print(f"  Key difference from existing tutorials:")
print(f"    Existing: Real-valued, time-domain, Dirichlet/Neumann BC")
print(f"    THIS:     Complex-valued, frequency-domain, Sommerfeld BC")
print(f"")
print(f"  Results saved to: {RESULTS_DIR}")
print("=" * 70)
