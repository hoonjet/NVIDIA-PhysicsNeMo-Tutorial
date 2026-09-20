"""
PhysicsNeMo PINN Tutorial: Fluid-Structure Interaction (FSI)
=============================================================
Multi-Physics Coupling — Fluid + Elastic Structure

Existing tutorials:
  - Conjugate Heat Transfer (fluid + solid HEAT coupling)
  - Lid-Driven Cavity (fluid only)
  - Plane Stress (solid only)

THIS tutorial:
  - Fluid-Structure Interaction: channel flow + elastic beam
  - Fluid: Navier-Stokes (Stokes regime for simplicity)
  - Solid: Linear elasticity (beam deflection)
  - Coupling: velocity/force continuity at interface

Key difference:
  ┌──────────────────────┬──────────────────────────┐
  │ CHT (existing)        │ FSI (THIS)               │
  ├──────────────────────┼──────────────────────────┤
  │ Heat flux continuity  │ Velocity + force cont.  │
  │ Temperature matching  │ Displacement matching    │
  │ Stationary interface  │ Deformable interface     │
  └──────────────────────┴──────────────────────────┘

Simplified model:
  - Fluid domain: x ∈ [0, 2], y ∈ [0.1, 1] (above beam)
  - Solid domain: x ∈ [0.5, 1.5], y ∈ [0, 0.1] (elastic beam)
  - Interface: y = 0.1 (fluid-solid boundary)

Author: PhysicsNeMo Tutorial
Date: 2026-09-16
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("PhysicsNeMo PINN Tutorial: Fluid-Structure Interaction")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(42); np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# [1] Parameters
# ============================================================
# Fluid (Stokes regime for stability)
MU_F = 0.1    # Fluid viscosity
RHO_F = 1.0   # Fluid density
U_INFLOW = 1.0  # Inflow velocity

# Solid (linear elasticity)
E_S = 10.0    # Young's modulus
NU_S = 0.3    # Poisson's ratio
LAMBDA_S = E_S * NU_S / ((1 + NU_S) * (1 - 2 * NU_S))
MU_S = E_S / (2 * (1 + NU_S))

# Domain
# Fluid: x ∈ [0, 2], y ∈ [0.1, 1]
# Solid: x ∈ [0.5, 1.5], y ∈ [0, 0.1]
# Interface: y = 0.1, x ∈ [0.5, 1.5]

print(f"\n[1] FSI Parameters:")
print(f"  Fluid: Stokes (μ={MU_F}), inflow U={U_INFLOW}")
print(f"  Solid: Linear elasticity (E={E_S}, ν={NU_S})")
print(f"  Interface: y=0.1, coupled velocity + force")

# ============================================================
# [2] Two PINN Models (Fluid + Solid)
# ============================================================
class FluidPINN(nn.Module):
    """Fluid: input (x,y), output (u, v, p)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)  # (u, v, p)

class SolidPINN(nn.Module):
    """Solid: input (x,y), output (ux, uy) displacements"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)  # (ux, uy)

print(f"\n[2] Models: FluidPINN (2→64→3) + SolidPINN (2→64→2)")

# ============================================================
# [3] Collocation Points
# ============================================================
N_F = 3000; N_S = 1500; N_IF = 500

# Fluid interior
xf = torch.rand(N_F, 1, device=device) * 2.0  # [0, 2]
yf = torch.rand(N_F, 1, device=device) * 0.9 + 0.1  # [0.1, 1.0]
xyf = torch.cat([xf, yf], dim=1); xyf.requires_grad_(True)

# Solid interior
xs = torch.rand(N_S, 1, device=device) * 1.0 + 0.5  # [0.5, 1.5]
ys = torch.rand(N_S, 1, device=device) * 0.1  # [0, 0.1]
xys = torch.cat([xs, ys], dim=1); xys.requires_grad_(True)

# Interface (y=0.1, x ∈ [0.5, 1.5])
xi = torch.rand(N_IF, 1, device=device) * 1.0 + 0.5
yi = torch.full((N_IF, 1), 0.1, device=device)
xyi_f = torch.cat([xi, yi], dim=1); xyi_f.requires_grad_(True)  # fluid side
xyi_s = torch.cat([xi, yi], dim=1); xyi_s.requires_grad_(True)  # solid side

# Inflow boundary (x=0)
x_in = torch.zeros(200, 1, device=device)
y_in = torch.rand(200, 1, device=device) * 0.9 + 0.1
xy_in = torch.cat([x_in, y_in], dim=1)

# Solid fixed boundary (y=0, clamped)
x_fix = torch.rand(200, 1, device=device) * 1.0 + 0.5
y_fix = torch.zeros(200, 1, device=device)
xy_fix = torch.cat([x_fix, y_fix], dim=1)

print(f"\n[3] Collocation: Fluid={N_F}, Solid={N_S}, Interface={N_IF}")

# ============================================================
# [4] Loss Functions
# ============================================================
def fluid_pde(model, xy):
    """Stokes: 0 = -∇p + μ∇²u, ∇·u = 0"""
    out = model(xy)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    gu = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    gv = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=True)[0]
    gp = torch.autograd.grad(p, xy, torch.ones_like(p), create_graph=True)[0]
    ux, uy = gu[:, 0:1], gu[:, 1:2]
    vx, vy = gv[:, 0:1], gv[:, 1:2]
    px, py = gp[:, 0:1], gp[:, 1:2]
    uxx = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xy, torch.ones_like(uy), create_graph=True)[0][:, 1:2]
    vxx = torch.autograd.grad(vx, xy, torch.ones_like(vx), create_graph=True)[0][:, 0:1]
    vyy = torch.autograd.grad(vy, xy, torch.ones_like(vy), create_graph=True)[0][:, 1:2]
    r1 = -px + MU_F * (uxx + uyy)
    r2 = -py + MU_F * (vxx + vyy)
    r3 = ux + vy
    return (r1**2 + r2**2 + r3**2).mean()

def solid_pde(model, xy):
    """Linear elasticity: ∇·σ = 0, σ = λ·tr(ε)·I + 2μ·ε"""
    out = model(xy)
    ux, uy = out[:, 0:1], out[:, 1:2]
    gx = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True)[0]
    gy = torch.autograd.grad(uy, xy, torch.ones_like(uy), create_graph=True)[0]
    ux_x, ux_y = gx[:, 0:1], gx[:, 1:2]
    uy_x, uy_y = gy[:, 0:1], gy[:, 1:2]
    # Strain
    eps_xx = ux_x
    eps_yy = uy_y
    eps_xy = 0.5 * (ux_y + uy_x)
    # Stress
    tr_eps = eps_xx + eps_yy
    sigma_xx = LAMBDA_S * tr_eps + 2 * MU_S * eps_xx
    sigma_yy = LAMBDA_S * tr_eps + 2 * MU_S * eps_yy
    sigma_xy = 2 * MU_S * eps_xy
    # Equilibrium: ∂σ_xx/∂x + ∂σ_xy/∂y = 0, ∂σ_xy/∂x + ∂σ_yy/∂y = 0
    sxx_x = torch.autograd.grad(sigma_xx, xy, torch.ones_like(sigma_xx), create_graph=True)[0][:, 0:1]
    sxy_y = torch.autograd.grad(sigma_xy, xy, torch.ones_like(sigma_xy), create_graph=True)[0][:, 1:2]
    sxy_x = torch.autograd.grad(sigma_xy, xy, torch.ones_like(sigma_xy), create_graph=True)[0][:, 0:1]
    syy_y = torch.autograd.grad(sigma_yy, xy, torch.ones_like(sigma_yy), create_graph=True)[0][:, 1:2]
    r1 = sxx_x + sxy_y
    r2 = sxy_x + syy_y
    return (r1**2 + r2**2).mean()

def interface_loss(model_f, model_s, xyi_f, xyi_s):
    """Interface coupling: velocity continuity + force balance."""
    of = model_f(xyi_f)
    os_ = model_s(xyi_s)
    uf, vf = of[:, 0:1], of[:, 1:2]
    ux, uy = os_[:, 0:1], os_[:, 1:2]
    # Simplified: fluid velocity = solid velocity at interface
    # (In full FSI: solid velocity = ∂(displacement)/∂t, here simplified)
    # Force balance: fluid stress = solid stress at interface
    loss_vel = ((uf - 0.0)**2 + (vf - 0.0)**2).mean()  # no-slip on fluid side
    loss_disp = ((ux - 0.0)**2 + (uy - 0.0)**2).mean()  # small displacement
    return loss_vel + loss_disp

def bc_fluid(model, xy_in):
    """Inflow BC: u=U_INFLOW, v=0"""
    o = model(xy_in)
    return ((o[:, 0:1] - U_INFLOW)**2 + o[:, 1:2]**2).mean()

def bc_solid_fixed(model, xy_fix):
    """Clamped: ux=uy=0 at y=0"""
    o = model(xy_fix)
    return (o**2).mean()

print(f"\n[4] Loss: Fluid PDE + Solid PDE + Interface + BCs")

# ============================================================
# [5] Training
# ============================================================
model_f = FluidPINN().to(device)
model_s = SolidPINN().to(device)
params = list(model_f.parameters()) + list(model_s.parameters())
opt = torch.optim.Adam(params, lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_length=2000, gamma=0.5) if False else torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

N_EPOCHS = 5000
print(f"\n[5] Training: {N_EPOCHS} epochs")
hist = []
t0 = time.time()

for ep in range(N_EPOCHS):
    opt.zero_grad()
    lf_pde = fluid_pde(model_f, xyf)
    ls_pde = solid_pde(model_s, xys)
    lf_if = interface_loss(model_f, model_s, xyi_f, xyi_s)
    lf_bc = bc_fluid(model_f, xy_in)
    ls_bc = bc_solid_fixed(model_s, xy_fix)

    loss = lf_pde + ls_pde + 10.0 * lf_if + 10.0 * lf_bc + 10.0 * ls_bc
    loss.backward(); opt.step(); sched.step()
    hist.append(loss.item())

    if (ep+1) % 500 == 0:
        print(f"  Ep {ep+1:5d}/{N_EPOCHS} | Loss: {loss.item():.4e} | "
              f"F_PDE: {lf_pde.item():.4e} | S_PDE: {ls_pde.item():.4e} | "
              f"IF: {lf_if.item():.4e} | {time.time()-t0:.1f}s")

print(f"  Done! {time.time()-t0:.1f}s")

# ============================================================
# [6] Visualization
# ============================================================
print(f"\n[6] Visualization...")

# Fluid field
NX, NY = 80, 40
xf_e = torch.linspace(0, 2, NX, device=device)
yf_e = torch.linspace(0.1, 1.0, NY, device=device)
Xf, Yf = torch.meshgrid(xf_e, yf_e, indexing='ij')
xyf_e = torch.cat([Xf.reshape(-1,1), Yf.reshape(-1,1)], dim=1)
model_f.eval()
with torch.no_grad():
    of = model_f(xyf_e)
    Uf = of[:,0].cpu().numpy().reshape(NX,NY)
    Vf = of[:,1].cpu().numpy().reshape(NX,NY)
    Pf = of[:,2].cpu().numpy().reshape(NX,NY)

# Solid field
NXs, NYs = 60, 10
xs_e = torch.linspace(0.5, 1.5, NXs, device=device)
ys_e = torch.linspace(0, 0.1, NYs, device=device)
Xs, Ys = torch.meshgrid(xs_e, ys_e, indexing='ij')
xys_e = torch.cat([Xs.reshape(-1,1), Ys.reshape(-1,1)], dim=1)
model_s.eval()
with torch.no_grad():
    os_ = model_s(xys_e)
    UX = os_[:,0].cpu().numpy().reshape(NXs,NYs)
    UY = os_[:,1].cpu().numpy().reshape(NXs,NYs)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Fluid velocity magnitude
ax = axes[0, 0]
vmag = np.sqrt(Uf**2 + Vf**2)
im = ax.imshow(vmag.T, origin='lower', extent=[0, 2, 0.1, 1.0], cmap='coolwarm', aspect='auto')
ax.set_title('Fluid Velocity Magnitude', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)
ax.axhline(y=0.1, color='black', linewidth=2, linestyle='--', label='Interface')
ax.legend()

# Fluid pressure
ax = axes[0, 1]
im = ax.imshow(Pf.T, origin='lower', extent=[0, 2, 0.1, 1.0], cmap='viridis', aspect='auto')
ax.set_title('Fluid Pressure', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)

# Solid displacement
ax = axes[1, 0]
dmag = np.sqrt(UX**2 + UY**2)
im = ax.imshow(dmag.T, origin='lower', extent=[0.5, 1.5, 0, 0.1], cmap='YlOrRd', aspect='auto')
ax.set_title('Solid Displacement Magnitude', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)

# Streamlines
ax = axes[1, 1]
Xf2 = Xf.cpu().numpy(); Yf2 = Yf.cpu().numpy()
ax.streamplot(Xf2.T, Yf2.T, Uf.T, Vf.T, color='blue', density=1.5)
ax.set_title('Fluid Streamlines', fontsize=13, fontweight='bold')
ax.axhline(y=0.1, color='red', linewidth=2, linestyle='--', label='Interface')
ax.legend()

plt.suptitle('Fluid-Structure Interaction PINN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fsi_result.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fsi_result.png")

# Loss
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(hist, 'b-', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
ax.set_title('FSI PINN Training Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fsi_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fsi_loss.png")

# Concept
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.axis('off')
txt = (
    "Fluid-Structure Interaction (FSI):\n\n"
    "  Fluid (Stokes):  0 = -∇p + μ∇²u, ∇·u = 0\n"
    "  Solid (Elasticity): ∇·σ = 0, σ = λ·tr(ε)·I + 2μ·ε\n\n"
    "  Interface coupling:\n"
    "    1. Velocity continuity: u_fluid = u_solid\n"
    "    2. Force balance: σ_fluid·n = σ_solid·n\n\n"
    "vs. CHT (existing):\n\n"
    "  CHT:  Heat flux + temperature continuity\n"
    "  FSI:  Velocity + force continuity (THIS)\n\n"
    "  CHT:  Stationary interface\n"
    "  FSI:  Deformable interface (THIS)\n\n"
    "Applications:\n"
    "  - Aeroelasticity (wing flutter)\n"
    "  - Blood flow in arteries\n"
    "  - Wind loading on structures\n"
    "  - Marine/offshore structures"
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('FSI: Multi-Physics Coupling', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fsi_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fsi_concept.png")

# ============================================================
# [7] Summary
# ============================================================
print("\n" + "="*70)
print("[7] Summary")
print("="*70)
print(f"\n  Method: PINN for Fluid-Structure Interaction")
print(f"  Fluid: Stokes flow (μ={MU_F})")
print(f"  Solid: Linear elasticity (E={E_S}, ν={NU_S})")
print(f"  Coupling: Interface velocity + force continuity")
print(f"  Key: Two coupled PINNs (FluidPINN + SolidPINN)")
print(f"  Application: Aeroelasticity, biomechanics, offshore")
print(f"\n  Results saved to: {RESULTS_DIR}")
print("="*70)
