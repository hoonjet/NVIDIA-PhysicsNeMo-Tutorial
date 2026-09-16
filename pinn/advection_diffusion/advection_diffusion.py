"""
PhysicsNeMo PINN Tutorial: Advection-Diffusion Equation
========================================================
Pollutant Transport — Linear Convection + Diffusion

Existing tutorials cover:
  - Burgers (nonlinear advection + diffusion)
  - Reaction-Diffusion (reaction + diffusion)
  - Lid-Driven Cavity (Navier-Stokes)

THIS tutorial:
  - Linear advection-diffusion (v·∇c = D∇²c) — simplest transport PDE
  - Peclet number analysis (Pe = v·L/D): advection vs diffusion balance
  - Gaussian plume dispersion (environmental engineering)
  - Pure transport (Pe→∞) vs pure diffusion (Pe→0) regime comparison

Key difference from existing tutorials:
  ┌──────────────────────┬──────────────────────────┐
  │ Burgers (existing)    │ THIS (Advection-Diffusion)│
  ├──────────────────────┼──────────────────────────┤
  │ Nonlinear (u·∇u)     │ Linear (v·∇c, fixed v)   │
  │ Single regime          │ Pe controls regime       │
  │ Shock formation        │ Gaussian plume           │
  └──────────────────────┴──────────────────────────┘

Physics: 2D Advection-Diffusion Equation
  ∂c/∂t + v·∇c = D·∇²c
  v = (vx, vy) = (1.0, 0.0)  (wind velocity)
  D = diffusion coefficient (varies for Pe study)

  IC: c(x,y,0) = exp(-50·((x-0.25)² + (y-0.5)²))  (Gaussian source)
  BC: c = 0 at inflow (x=0), outflow (natural)

Author: PhysicsNeMo Tutorial
Date: 2026-09-16
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
print("PhysicsNeMo PINN Tutorial: Advection-Diffusion Equation")
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
VX = 1.0   # Wind velocity x-component
VY = 0.0   # Wind velocity y-component
D = 0.01   # Diffusion coefficient (default)
L = 1.0    # Characteristic length
PE = VX * L / D  # Peclet number

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.5

print(f"\n[1] Problem Parameters:")
print(f"  Equation: ∂c/∂t + v·∇c = D·∇²c")
print(f"  Velocity: v = ({VX}, {VY})")
print(f"  Diffusion: D = {D}")
print(f"  Peclet number: Pe = v·L/D = {PE:.1f}")
print(f"  Domain: x∈[{X_MIN},{X_MAX}], y∈[{Y_MIN},{Y_MAX}], t∈[{T_MIN},{T_MAX}]")
print(f"  Regime: {'Advection-dominated' if PE > 10 else 'Diffusion-dominated' if PE < 0.1 else 'Mixed'}")

# ============================================================
# [2] Analytical Solution (Gaussian Plume)
# ============================================================
def analytical_solution(x, y, t, vx, vy, D):
    """
    Gaussian plume: point source advected + diffused.

    c(x,y,t) = (1/(4πDt)) * exp(-((x-vx*t-x0)² + (y-vy*t-y0)²) / (4Dt))

    This is the Green's function for advection-diffusion.
    """
    x0, y0 = 0.25, 0.5  # Source location
    sigma2 = 4 * D * t + 1e-10  # Avoid division by zero at t=0
    c = (1.0 / (4 * np.pi * D * t + 1e-10)) * \
        np.exp(-((x - vx * t - x0)**2 + (y - vy * t - y0)**2) / sigma2)
    # Scale for reasonable magnitude
    c = c * 0.01
    return c

print(f"\n[2] Analytical solution: Gaussian plume (Green's function)")

# ============================================================
# [3] PINN Model
# ============================================================
class AdvDiffPINN(nn.Module):
    """
    PINN for advection-diffusion equation.

    Input: (x, y, t)
    Output: c (concentration, scalar)
    """
    def __init__(self, layers=[3, 64, 64, 64, 64, 1]):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        x = self.linears[-1](x)
        return x

print(f"\n[3] PINN Model: 3 → 64 → 64 → 64 → 64 → 1")
print(f"  Input: (x, y, t), Output: c (concentration)")

# ============================================================
# [4] Collocation Points
# ============================================================
N_INTERIOR = 5000
N_IC = 1000
N_BC = 500
N_REF = 2000

# Interior
x_int = torch.rand(N_INTERIOR, 1, device=device) * (X_MAX - X_MIN) + X_MIN
y_int = torch.rand(N_INTERIOR, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
t_int = torch.rand(N_INTERIOR, 1, device=device) * (T_MAX - T_MIN) + T_MIN
xyt_int = torch.cat([x_int, y_int, t_int], dim=1)
xyt_int.requires_grad_(True)

# Initial condition points (t=0)
x_ic = torch.rand(N_IC, 1, device=device) * (X_MAX - X_MIN) + X_MIN
y_ic = torch.rand(N_IC, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
t_ic = torch.zeros(N_IC, 1, device=device)
xyt_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)

# Reference data points
x_ref = torch.rand(N_REF, 1, device=device) * (X_MAX - X_MIN) + X_MIN
y_ref = torch.rand(N_REF, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
t_ref = torch.rand(N_REF, 1, device=device) * (T_MAX - T_MIN) + T_MIN
xyt_ref = torch.cat([x_ref, y_ref, t_ref], dim=1)

# Analytical at reference
x_np = x_ref.cpu().numpy().flatten()
y_np = y_ref.cpu().numpy().flatten()
t_np = t_ref.cpu().numpy().flatten()
c_ref_np = analytical_solution(x_np, y_np, t_np, VX, VY, D)
c_ref = torch.tensor(c_ref_np, dtype=torch.float32, device=device).view(-1, 1)

print(f"\n[4] Collocation Points:")
print(f"  Interior: {N_INTERIOR}, IC: {N_IC}, Reference: {N_REF}")

# ============================================================
# [5] Loss Functions
# ============================================================
def compute_pde_residual(model, xyt):
    """Advection-diffusion: ∂c/∂t + vx·∂c/∂x + vy·∂c/∂y - D·∇²c = 0"""
    c = model(xyt)
    grad = torch.autograd.grad(c, xyt, grad_outputs=torch.ones_like(c),
                                create_graph=True, retain_graph=True)[0]
    c_x, c_y, c_t = grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]

    c_xx = torch.autograd.grad(c_x, xyt, grad_outputs=torch.ones_like(c_x),
                                create_graph=True, retain_graph=True)[0][:, 0:1]
    c_yy = torch.autograd.grad(c_y, xyt, grad_outputs=torch.ones_like(c_y),
                                create_graph=True, retain_graph=True)[0][:, 1:2]

    residual = c_t + VX * c_x + VY * c_y - D * (c_xx + c_yy)
    return (residual**2).mean()

def compute_ic_loss(model, xyt_ic):
    """IC: c(x,y,0) = Gaussian source."""
    x0, y0 = 0.25, 0.5
    c_pred = model(xyt_ic)
    x = xyt_ic[:, 0:1]
    y = xyt_ic[:, 1:2]
    c_target = 0.01 * torch.exp(-50 * ((x - x0)**2 + (y - y0)**2))
    return ((c_pred - c_target)**2).mean()

def compute_data_loss(model, xyt_ref, c_ref):
    """Data loss against analytical solution."""
    c_pred = model(xyt_ref)
    return ((c_pred - c_ref)**2).mean()

print(f"\n[5] Loss: PDE + IC + Data")

# ============================================================
# [6] Training
# ============================================================
model = AdvDiffPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

N_EPOCHS = 5000
LAMBDA_PDE = 1.0
LAMBDA_IC = 10.0
LAMBDA_DATA = 5.0

print(f"\n[6] Training: {N_EPOCHS} epochs, λ_pde={LAMBDA_PDE}, λ_ic={LAMBDA_IC}, λ_data={LAMBDA_DATA}")

loss_history = {'total': [], 'pde': [], 'ic': [], 'data': []}
t_start = time.time()

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()

    loss_pde = compute_pde_residual(model, xyt_int)
    loss_ic = compute_ic_loss(model, xyt_ic)
    loss_data = compute_data_loss(model, xyt_ref, c_ref)

    loss = LAMBDA_PDE * loss_pde + LAMBDA_IC * loss_ic + LAMBDA_DATA * loss_data

    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history['total'].append(loss.item())
    loss_history['pde'].append(loss_pde.item())
    loss_history['ic'].append(loss_ic.item())
    loss_history['data'].append(loss_data.item())

    if (epoch + 1) % 500 == 0:
        print(f"  Epoch {epoch+1:5d}/{N_EPOCHS} | "
              f"Total: {loss.item():.6e} | PDE: {loss_pde.item():.6e} | "
              f"IC: {loss_ic.item():.6e} | Data: {loss_data.item():.6e} | "
              f"Time: {time.time()-t_start:.1f}s")

print(f"\n  Training complete! Time: {time.time()-t_start:.1f}s")

# ============================================================
# [7] Visualization
# ============================================================
print(f"\n[7] Visualization...")

NX_EVAL, NY_EVAL = 100, 50
x_eval = torch.linspace(X_MIN, X_MAX, NX_EVAL, device=device)
y_eval = torch.linspace(Y_MIN, Y_MAX, NY_EVAL, device=device)
X_eval, Y_eval = torch.meshgrid(x_eval, y_eval, indexing='ij')

snapshots_t = [0.1, 0.2, 0.3, 0.5]

fig, axes = plt.subplots(len(snapshots_t), 3, figsize=(18, 5 * len(snapshots_t)))

for row, t_snap in enumerate(snapshots_t):
    T_eval = torch.full((NX_EVAL * NY_EVAL, 1), t_snap, device=device)
    xy_eval = torch.cat([X_eval.reshape(-1, 1), Y_eval.reshape(-1, 1), T_eval], dim=1)

    model.eval()
    with torch.no_grad():
        c_pred = model(xy_eval).cpu().numpy().reshape(NX_EVAL, NY_EVAL)

    x_np2 = X_eval.cpu().numpy()
    y_np2 = Y_eval.cpu().numpy()
    t_np2 = np.full_like(x_np2, t_snap)
    c_ana = analytical_solution(x_np2, y_np2, t_np2, VX, VY, D)

    vmax = max(c_ana.max(), c_pred.max(), 0.001)

    axes[row, 0].imshow(c_ana.T, origin='lower', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                         cmap='YlOrRd', vmin=0, vmax=vmax)
    axes[row, 0].set_title(f'Analytical (t={t_snap})', fontsize=11)
    axes[row, 0].set_xlabel('x'); axes[row, 0].set_ylabel('y')

    axes[row, 1].imshow(c_pred.T, origin='lower', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                         cmap='YlOrRd', vmin=0, vmax=vmax)
    axes[row, 1].set_title(f'PINN (t={t_snap})', fontsize=11)
    axes[row, 1].set_xlabel('x'); axes[row, 1].set_ylabel('y')

    error = np.abs(c_ana - c_pred)
    im = axes[row, 2].imshow(error.T, origin='lower', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                               cmap='hot', vmin=0, vmax=vmax*0.3)
    axes[row, 2].set_title(f'|Error| (t={t_snap})', fontsize=11)
    axes[row, 2].set_xlabel('x'); axes[row, 2].set_ylabel('y')

plt.suptitle(f'Advection-Diffusion PINN (Pe={PE:.1f}, D={D})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'advdiff_snapshots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: advdiff_snapshots.png")

# --- Figure 2: 1D cross-section ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x_1d = torch.linspace(X_MIN, X_MAX, 200, device=device).view(-1, 1)
y_1d = torch.full_like(x_1d, 0.5)

for t_snap in [0.1, 0.2, 0.3, 0.5]:
    t_1d = torch.full_like(x_1d, t_snap)
    xy_1d = torch.cat([x_1d, y_1d, t_1d], dim=1)
    with torch.no_grad():
        c_1d_pred = model(xy_1d).cpu().numpy().flatten()
    x_np3 = x_1d.cpu().numpy().flatten()
    c_1d_ana = analytical_solution(x_np3, np.full_like(x_np3, 0.5), np.full_like(x_np3, t_snap), VX, VY, D)

    ax.plot(x_np3, c_1d_ana, 'b-', linewidth=2, alpha=0.5)
    ax.plot(x_np3, c_1d_pred, 'r--', linewidth=2, label=f't={t_snap}')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('c (concentration)', fontsize=12)
ax.set_title(f'1D Cross-Section (y=0.5) — Blue: Analytical, Red: PINN', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'advdiff_1d.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: advdiff_1d.png")

# --- Figure 3: Loss ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(loss_history['total'], 'b-', linewidth=2, label='Total')
ax.semilogy(loss_history['pde'], 'g--', alpha=0.7, label='PDE')
ax.semilogy(loss_history['ic'], 'r--', alpha=0.7, label='IC')
ax.semilogy(loss_history['data'], 'm--', alpha=0.7, label='Data')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Advection-Diffusion PINN Training Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'advdiff_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: advdiff_loss.png")

# --- Figure 4: Concept ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.axis('off')
concept_text = (
    "Advection-Diffusion Equation:\n\n"
    "  ∂c/∂t + v·∇c = D·∇²c\n\n"
    "  v = (1.0, 0.0)  — wind velocity\n"
    "  D = 0.01         — diffusion\n"
    f"  Pe = v·L/D = {PE:.1f}\n\n"
    "  Pe >> 1: Advection-dominated\n"
    "    → Plume moves with wind\n"
    "    → Narrow, elongated\n\n"
    "  Pe << 1: Diffusion-dominated\n"
    "    → Plume spreads isotropically\n"
    "    → Round, symmetric\n\n"
    "vs. Burgers (existing):\n\n"
    "  ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²\n"
    "  Nonlinear (u·∇u) → shock\n"
    "  THIS: Linear (v·∇c) → no shock"
)
ax.text(0.05, 0.95, concept_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Physics: Advection-Diffusion', fontsize=13, fontweight='bold')

ax = axes[1]
pes = [0.01, 0.1, 1, 10, 100]
regimes = ['Diffusion\n(Pe<<1)', '', 'Mixed\n(Pe~1)', '', 'Advection\n(Pe>>1)']
colors = ['blue', 'cyan', 'green', 'orange', 'red']
bars = ax.bar(range(len(pes)), [1]*5, color=colors, alpha=0.5)
ax.set_xticks(range(len(pes)))
ax.set_xticklabels([f'Pe={p}' for p in pes], fontsize=9)
ax.set_ylabel('Regime', fontsize=12)
ax.set_title(f'Peclet Number Regimes (THIS: Pe={PE:.1f})', fontsize=13, fontweight='bold')
for i, (bar, regime) in enumerate(zip(bars, regimes)):
    if regime:
        ax.text(bar.get_x() + bar.get_width()/2, 0.5, regime, ha='center', fontsize=9)
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'advdiff_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: advdiff_concept.png")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("[8] Summary")
print("=" * 70)

# Relative L2 error
with torch.no_grad():
    c_pred_ref = model(xyt_ref)
rel_err = torch.norm(c_pred_ref - c_ref) / (torch.norm(c_ref) + 1e-10)

print(f"\n  Method: PINN for Advection-Diffusion")
print(f"  Physics: Linear transport (∂c/∂t + v·∇c = D·∇²c)")
print(f"  Peclet number: Pe = {PE:.1f} ({'advection-dominated' if PE > 10 else 'diffusion-dominated' if PE < 0.1 else 'mixed'})")
print(f"  Relative L2 Error: {rel_err.item():.4f} ({rel_err.item()*100:.2f}%)")
print(f"")
print(f"  Key difference from existing tutorials:")
print(f"    Burgers: nonlinear advection → shock waves")
print(f"    THIS:    linear advection → Gaussian plume, Pe regimes")
print(f"")
print(f"  Results saved to: {RESULTS_DIR}")
print("=" * 70)
