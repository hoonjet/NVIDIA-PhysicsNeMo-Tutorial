"""
PhysicsNeMo PINN Tutorial: Maxwell's Equations (Electromagnetic Wave Propagation)
=================================================================================
2D Transverse Electric (TE) Wave in a Dielectric Slab

Existing tutorials cover:
  - Electrostatics (Poisson, static, scalar potential)
  - Helmholtz (frequency-domain, complex, acoustic)
  - Wave equation (FNO, time-domain, real, scalar)

THIS tutorial:
  - Maxwell's equations (dynamic electromagnetics, VECTOR field)
  - TE mode: Ez, Hx, Hy (3 coupled vector components)
  - Dielectric slab interface (refraction, Snell's law)
  - Time-domain, real-valued, but VECTOR (not scalar)

Key difference from existing tutorials:
  ┌──────────────────────┬──────────────────────────┐
  │ Electrostatics        │ THIS (Maxwell)           │
  ├──────────────────────┼──────────────────────────┤
  │ Static (∂/∂t = 0)    │ Dynamic (time-varying)   │
  │ Scalar potential φ    │ Vector (E, H) — 3 comps │
  │ Poisson equation       │ Maxwell curl equations   │
  │ No wave propagation    │ Wave propagation + refraction │
  │ Single material        │ Dielectric interface    │
  └──────────────────────┴──────────────────────────┘

Maxwell's Equations (TE mode, 2D):
  ∂Ez/∂t = (1/ε)(∂Hy/∂x - ∂Hx/∂y)
  ∂Hx/∂t = -(1/μ)(∂Ez/∂y)
  ∂Hy/∂t = (1/μ)(∂Ez/∂x)

  where ε = ε_r * ε_0 (permittivity, varies with material)
        μ = μ_0 (permeability, constant)

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

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo PINN Tutorial: Maxwell's Equations (EM Wave)")
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
EPS_R_1 = 1.0       # Relative permittivity (left region, vacuum)
EPS_R_2 = 4.0       # Relative permittivity (right region, dielectric)
MU_0 = 1.0          # Permeability (normalized)
C0 = 1.0 / np.sqrt(MU_0 * EPS_R_1)  # Speed of light in vacuum (normalized)
OMEGA = 2.0 * np.pi  # Angular frequency
K0 = OMEGA / C0     # Wavenumber in vacuum
SLAB_X = 0.5        # Dielectric slab interface position

print(f"\n[1] Problem Parameters:")
print(f"  ε_r1 = {EPS_R_1} (vacuum), ε_r2 = {EPS_R_2} (dielectric)")
print(f"  μ₀ = {MU_0} (normalized)")
print(f"  c₀ = {C0:.4f} (speed of light)")
print(f"  ω = {OMEGA:.4f}, k₀ = {K0:.4f}")
print(f"  Slab interface at x = {SLAB_X}")
print(f"  Wavelength λ = {2*np.pi/K0:.4f}")

# ============================================================
# [2] Analytical Solution (Plane wave + refraction)
# ============================================================
def analytical_solution(x, y, t, eps_r1, eps_r2, omega, slab_x):
    """
    Analytical solution: plane wave hitting dielectric interface.
    Left (x < slab_x): incident + reflected wave
    Right (x >= slab_x): transmitted wave

    Ez = E0 * exp(i(kx - ωt)) for each component
    """
    k1 = omega * np.sqrt(eps_r1)  # wavenumber in region 1
    k2 = omega * np.sqrt(eps_r2)  # wavenumber in region 2

    # Normal incidence (simplification: 1D-like, no y-dependence)
    # Fresnel coefficients
    r = (np.sqrt(eps_r1) - np.sqrt(eps_r2)) / (np.sqrt(eps_r1) + np.sqrt(eps_r2))
    t_coeff = 2 * np.sqrt(eps_r1) / (np.sqrt(eps_r1) + np.sqrt(eps_r2))

    E0 = 1.0

    Ez = np.zeros_like(x, dtype=np.float32)
    Hx = np.zeros_like(x, dtype=np.float32)
    Hy = np.zeros_like(x, dtype=np.float32)

    # Region 1: incident + reflected
    mask1 = x < slab_x
    Ez[mask1] = E0 * np.cos(k1 * x[mask1] - omega * t) + \
                r * E0 * np.cos(-k1 * x[mask1] - omega * t)
    Hy[mask1] = (E0 / MU_0 / omega * k1) * np.cos(k1 * x[mask1] - omega * t) - \
                (r * E0 / MU_0 / omega * k1) * np.cos(-k1 * x[mask1] - omega * t)
    Hx[mask1] = 0.0  # TE mode, normal incidence

    # Region 2: transmitted
    mask2 = x >= slab_x
    Ez[mask2] = t_coeff * E0 * np.cos(k2 * (x[mask2] - slab_x) - omega * t)
    Hy[mask2] = (t_coeff * E0 / MU_0 / omega * k2) * np.cos(k2 * (x[mask2] - slab_x) - omega * t)
    Hx[mask2] = 0.0

    return Ez, Hx, Hy

print(f"\n[2] Analytical solution: Fresnel coefficients (normal incidence)")

# ============================================================
# [3] PINN Model (Vector field: Ez, Hx, Hy)
# ============================================================
class MaxwellPINN(nn.Module):
    """
    PINN for Maxwell's equations with vector field output.

    Input: (x, y, t) — spatial + temporal coordinates
    Output: (Ez, Hx, Hy) — 3 vector components (TE mode)

    Architecture: 3 → 64 → 64 → 64 → 64 → 3
    """
    def __init__(self, layers=[3, 64, 64, 64, 64, 3]):
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
        return x  # [N, 3] — (Ez, Hx, Hy)

print(f"\n[3] PINN Model: MaxwellPINN")
print(f"  Architecture: 3 → 64 → 64 → 64 → 64 → 3")
print(f"  Input: (x, y, t)")
print(f"  Output: (Ez, Hx, Hy) — vector field (TE mode)")

# ============================================================
# [4] Collocation Points
# ============================================================
N_INTERIOR = 5000
N_BOUNDARY = 500
N_INITIAL = 500

L_X, R_X = 0.0, 1.0
B_Y, T_Y = 0.0, 1.0
T_START, T_END = 0.0, 2.0

# Interior collocation
x_int = torch.rand(N_INTERIOR, 1, device=device) * (R_X - L_X) + L_X
y_int = torch.rand(N_INTERIOR, 1, device=device) * (T_Y - B_Y) + B_Y
t_int = torch.rand(N_INTERIOR, 1, device=device) * (T_END - T_START) + T_START
xyt_int = torch.cat([x_int, y_int, t_int], dim=1)
xyt_int.requires_grad_(True)

# Boundary (x=0, Ez = cos(ωt))
t_bc = torch.rand(N_BOUNDARY, 1, device=device) * (T_END - T_START) + T_START
y_bc = torch.rand(N_BOUNDARY, 1, device=device) * (T_Y - B_Y) + B_Y
x_bc = torch.zeros(N_BOUNDARY, 1, device=device)
xyt_bc = torch.cat([x_bc, y_bc, t_bc], dim=1)

# Initial condition (t=0)
x_ic = torch.rand(N_INITIAL, 1, device=device) * (R_X - L_X) + L_X
y_ic = torch.rand(N_INITIAL, 1, device=device) * (T_Y - B_Y) + B_Y
t_ic = torch.zeros(N_INITIAL, 1, device=device)
xyt_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)

# Reference data
N_REF = 1000
x_ref = torch.rand(N_REF, 1, device=device) * (R_X - L_X) + L_X
y_ref = torch.rand(N_REF, 1, device=device) * (T_Y - B_Y) + B_Y
t_ref = torch.rand(N_REF, 1, device=device) * (T_END - T_START) + T_START
xyt_ref = torch.cat([x_ref, y_ref, t_ref], dim=1)

# Analytical at reference points
x_np = x_ref.cpu().numpy().flatten()
y_np = y_ref.cpu().numpy().flatten()
t_np = t_ref.cpu().numpy().flatten()
Ez_ref_np, Hx_ref_np, Hy_ref_np = analytical_solution(x_np, y_np, t_np, EPS_R_1, EPS_R_2, OMEGA, SLAB_X)
Ez_ref = torch.tensor(Ez_ref_np, dtype=torch.float32, device=device).view(-1, 1)
Hx_ref = torch.tensor(Hx_ref_np, dtype=torch.float32, device=device).view(-1, 1)
Hy_ref = torch.tensor(Hy_ref_np, dtype=torch.float32, device=device).view(-1, 1)

print(f"\n[4] Collocation Points:")
print(f"  Interior: {N_INTERIOR}")
print(f"  Boundary (x=0): {N_BOUNDARY}")
print(f"  Initial (t=0): {N_INITIAL}")
print(f"  Reference data: {N_REF}")

# ============================================================
# [5] Loss Functions
# ============================================================
def get_eps(x):
    """Spatially varying permittivity."""
    return torch.where(x < SLAB_X, 
                       torch.tensor(EPS_R_1, device=device, dtype=torch.float32),
                       torch.tensor(EPS_R_2, device=device, dtype=torch.float32))

def compute_maxwell_residual(model, xyt):
    """
    Maxwell's equations (TE mode):
      ∂Ez/∂t = (1/ε)(∂Hy/∂x - ∂Hx/∂y)
      ∂Hx/∂t = -(1/μ)(∂Ez/∂y)
      ∂Hy/∂t = (1/μ)(∂Ez/∂x)
    """
    u = model(xyt)  # [N, 3]
    Ez = u[:, 0:1]
    Hx = u[:, 1:2]
    Hy = u[:, 2:3]

    x = xyt[:, 0:1]

    # First derivatives
    dEz = torch.autograd.grad(Ez, xyt, grad_outputs=torch.ones_like(Ez),
                               create_graph=True, retain_graph=True)[0]
    dHx = torch.autograd.grad(Hx, xyt, grad_outputs=torch.ones_like(Hx),
                               create_graph=True, retain_graph=True)[0]
    dHy = torch.autograd.grad(Hy, xyt, grad_outputs=torch.ones_like(Hy),
                               create_graph=True, retain_graph=True)[0]

    Ez_x, Ez_y, Ez_t = dEz[:, 0:1], dEz[:, 1:2], dEz[:, 2:3]
    Hx_x, Hx_y, Hx_t = dHx[:, 0:1], dHx[:, 1:2], dHx[:, 2:3]
    Hy_x, Hy_y, Hy_t = dHy[:, 0:1], dHy[:, 1:2], dHy[:, 2:3]

    eps = get_eps(x)

    # Maxwell residuals
    r1 = Ez_t - (1.0 / eps) * (Hy_x - Hx_y)   # Faraday
    r2 = Hx_t + (1.0 / MU_0) * Ez_y             # Ampere (Hx)
    r3 = Hy_t - (1.0 / MU_0) * Ez_x             # Ampere (Hy)

    return (r1**2 + r2**2 + r3**2).mean()

def compute_bc_loss(model, xyt_bc):
    """Boundary: Ez = cos(ωt) at x=0."""
    u = model(xyt_bc)
    Ez = u[:, 0:1]
    t = xyt_bc[:, 2:3]
    Ez_target = torch.cos(OMEGA * t)
    return ((Ez - Ez_target)**2).mean()

def compute_data_loss(model, xyt_ref, Ez_ref, Hx_ref, Hy_ref):
    """Data loss against analytical solution."""
    u = model(xyt_ref)
    loss = ((u[:, 0:1] - Ez_ref)**2).mean() + \
           ((u[:, 1:2] - Hx_ref)**2).mean() + \
           ((u[:, 2:3] - Hy_ref)**2).mean()
    return loss

print(f"\n[5] Loss Functions:")
print(f"  PDE: Maxwell's curl equations (3 coupled residuals)")
print(f"  BC: Ez = cos(ωt) at x=0")
print(f"  Data: Analytical (Fresnel)")

# ============================================================
# [6] Training
# ============================================================
model = MaxwellPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

N_EPOCHS = 5000
LAMBDA_PDE = 1.0
LAMBDA_BC = 10.0
LAMBDA_DATA = 5.0

print(f"\n[6] Training:")
print(f"  Epochs: {N_EPOCHS}")
print(f"  λ_PDE={LAMBDA_PDE}, λ_BC={LAMBDA_BC}, λ_DATA={LAMBDA_DATA}")

loss_history = {'total': [], 'pde': [], 'bc': [], 'data': []}
t_start = time.time()

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()

    loss_pde = compute_maxwell_residual(model, xyt_int)
    loss_bc = compute_bc_loss(model, xyt_bc)
    loss_data = compute_data_loss(model, xyt_ref, Ez_ref, Hx_ref, Hy_ref)

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

# Evaluate at fixed time snapshots
NX_EVAL, NY_EVAL = 100, 50
x_eval = torch.linspace(L_X, R_X, NX_EVAL, device=device)
y_eval = torch.linspace(B_Y, T_Y, NY_EVAL, device=device)
X_eval, Y_eval = torch.meshgrid(x_eval, y_eval, indexing='ij')

snapshots_t = [0.0, 0.5, 1.0, 1.5]

fig, axes = plt.subplots(len(snapshots_t), 3, figsize=(18, 5 * len(snapshots_t)))

for row, t_snap in enumerate(snapshots_t):
    T_eval = torch.full((NX_EVAL * NY_EVAL, 1), t_snap, device=device)
    xy_eval = torch.cat([X_eval.reshape(-1, 1), Y_eval.reshape(-1, 1), T_eval], dim=1)

    model.eval()
    with torch.no_grad():
        u_pred = model(xy_eval)
        Ez_pred = u_pred[:, 0].cpu().numpy().reshape(NX_EVAL, NY_EVAL)
        Hy_pred = u_pred[:, 2].cpu().numpy().reshape(NX_EVAL, NY_EVAL)

    # Analytical
    x_np = X_eval.cpu().numpy()
    y_np = Y_eval.cpu().numpy()
    t_np = np.full_like(x_np, t_snap)
    Ez_ana, _, Hy_ana = analytical_solution(x_np, y_np, t_np, EPS_R_1, EPS_R_2, OMEGA, SLAB_X)

    # Ez
    ax = axes[row, 0]
    im = ax.imshow(Ez_ana.T, origin='lower', extent=[L_X, R_X, B_Y, T_Y],
                   cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    ax.axvline(x=SLAB_X, color='green', linestyle='--', linewidth=1.5, label='Interface')
    ax.set_title(f'Ez Analytical (t={t_snap:.1f})', fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if row == 0:
        ax.legend(fontsize=8, loc='upper right')

    ax = axes[row, 1]
    im = ax.imshow(Ez_pred.T, origin='lower', extent=[L_X, R_X, B_Y, T_Y],
                   cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    ax.axvline(x=SLAB_X, color='green', linestyle='--', linewidth=1.5)
    ax.set_title(f'Ez PINN (t={t_snap:.1f})', fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axes[row, 2]
    error = np.abs(Ez_ana - Ez_pred)
    im = ax.imshow(error.T, origin='lower', extent=[L_X, R_X, B_Y, T_Y],
                   cmap='hot', vmin=0, vmax=0.2)
    ax.axvline(x=SLAB_X, color='green', linestyle='--', linewidth=1.5)
    ax.set_title(f'|Ez Error| (t={t_snap:.1f})', fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.suptitle("Maxwell PINN: Ez Field at Dielectric Interface (TE mode)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'maxwell_snapshots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: maxwell_snapshots.png")

# --- Figure 2: 1D cross-section (y=0.5) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

t_snap = 1.0
x_1d = torch.linspace(L_X, R_X, 200, device=device).view(-1, 1)
y_1d = torch.full_like(x_1d, 0.5)
t_1d = torch.full_like(x_1d, t_snap)
xy_1d = torch.cat([x_1d, y_1d, t_1d], dim=1)

model.eval()
with torch.no_grad():
    u_1d = model(xy_1d)
    Ez_1d_pred = u_1d[:, 0].cpu().numpy()
    Hx_1d_pred = u_1d[:, 1].cpu().numpy()
    Hy_1d_pred = u_1d[:, 2].cpu().numpy()

x_np = x_1d.cpu().numpy().flatten()
Ez_1d_ana, Hx_1d_ana, Hy_1d_ana = analytical_solution(
    x_np, np.full_like(x_np, 0.5), np.full_like(x_np, t_snap),
    EPS_R_1, EPS_R_2, OMEGA, SLAB_X)

ax = axes[0]
ax.plot(x_np, Ez_1d_ana, 'b-', linewidth=2, label='Analytical')
ax.plot(x_np, Ez_1d_pred, 'r--', linewidth=2, label='PINN')
ax.axvline(x=SLAB_X, color='green', linestyle=':', linewidth=1.5, label='Interface')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Ez', fontsize=12)
ax.set_title(f'Ez at y=0.5, t={t_snap}', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x_np, Hx_1d_ana, 'b-', linewidth=2, label='Analytical')
ax.plot(x_np, Hx_1d_pred, 'r--', linewidth=2, label='PINN')
ax.axvline(x=SLAB_X, color='green', linestyle=':', linewidth=1.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Hx', fontsize=12)
ax.set_title(f'Hx at y=0.5, t={t_snap}', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(x_np, Hy_1d_ana, 'b-', linewidth=2, label='Analytical')
ax.plot(x_np, Hy_1d_pred, 'r--', linewidth=2, label='PINN')
ax.axvline(x=SLAB_X, color='green', linestyle=':', linewidth=1.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Hy', fontsize=12)
ax.set_title(f'Hy at y=0.5, t={t_snap}', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Maxwell PINN: 1D Cross-Section (Vector Field)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'maxwell_1d.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: maxwell_1d.png")

# --- Figure 3: Loss history ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(loss_history['total'], label='Total', linewidth=2)
ax.semilogy(loss_history['pde'], label='PDE (Maxwell)', alpha=0.7)
ax.semilogy(loss_history['bc'], label='BC', alpha=0.7)
ax.semilogy(loss_history['data'], label='Data', alpha=0.7)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Maxwell PINN Training Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'maxwell_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: maxwell_loss.png")

# --- Figure 4: Concept ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
# Vacuum region
ax.axvspan(0, SLAB_X, alpha=0.1, color='blue', label='Vacuum (ε_r=1)')
# Dielectric region
ax.axvspan(SLAB_X, 1.0, alpha=0.1, color='red', label='Dielectric (ε_r=4)')
ax.axvline(x=SLAB_X, color='green', linestyle='--', linewidth=2, label='Interface')
# Wave arrows
for y_arrow in np.linspace(0.2, 0.8, 4):
    ax.annotate('', xy=(0.45, y_arrow), xytext=(0.05, y_arrow),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=(0.95, y_arrow), xytext=(0.55, y_arrow),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax.set_title('EM Wave at Dielectric Interface', fontsize=13, fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(fontsize=9, loc='upper right')
ax.text(0.25, 0.05, 'Incident +\nReflected', fontsize=9, ha='center', color='blue')
ax.text(0.75, 0.05, 'Transmitted\n(refracted)', fontsize=9, ha='center', color='red')

ax = axes[1]
ax.axis('off')
concept_text = (
    "Maxwell's Equations (TE mode, 2D):\n\n"
    "  ∂Ez/∂t = (1/ε)(∂Hy/∂x - ∂Hx/∂y)\n"
    "  ∂Hx/∂t = -(1/μ)(∂Ez/∂y)\n"
    "  ∂Hy/∂t = (1/μ)(∂Ez/∂x)\n\n"
    "Vector field: (Ez, Hx, Hy) — 3 components\n\n"
    "vs. Electrostatics (existing):\n\n"
    "  ∇²φ = -ρ/ε  (scalar, static)\n"
    "  No time derivative\n"
    "  Single material\n\n"
    "vs. Helmholtz (existing):\n\n"
    "  ∇²u + k²u = 0  (complex, freq.)\n"
    "  Scalar field\n"
    "  No vector components"
)
ax.text(0.05, 0.95, concept_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Maxwell vs Existing EM Tutorials', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'maxwell_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: maxwell_concept.png")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("[8] Summary")
print("=" * 70)

# Relative L2 error
rel_err_Ez = np.linalg.norm(Ez_1d_pred - Ez_1d_ana) / (np.linalg.norm(Ez_1d_ana) + 1e-10)
rel_err_Hy = np.linalg.norm(Hy_1d_pred - Hy_1d_ana) / (np.linalg.norm(Hy_1d_ana) + 1e-10)

print(f"\n  Method: PINN for Maxwell's Equations")
print(f"  Physics: EM wave propagation (TE mode, time-domain)")
print(f"  Field: Vector (Ez, Hx, Hy) — 3 coupled components")
print(f"  Material: Dielectric interface (ε_r: {EPS_R_1} → {EPS_R_2})")
print(f"")
print(f"  Relative L2 Error (1D, y=0.5, t=1.0):")
print(f"    Ez: {rel_err_Ez:.4f} ({rel_err_Ez*100:.2f}%)")
print(f"    Hy: {rel_err_Hy:.4f} ({rel_err_Hy*100:.2f}%)")
print(f"")
print(f"  Key difference from existing tutorials:")
print(f"    Electrostatics: scalar, static, single material")
print(f"    THIS:           vector, dynamic, dielectric interface")
print(f"")
print(f"  Results saved to: {RESULTS_DIR}")
print("=" * 70)
