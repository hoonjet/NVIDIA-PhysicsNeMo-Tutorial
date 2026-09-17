"""
PhysicsNeMo PINN Tutorial: Stokes Flow (Creeping Flow)
=======================================================
Low Reynolds Number — Viscosity-Dominated Flow

Existing tutorials:
  - Lid-Driven Cavity (Re=100, inertia + viscous)
  - Burgers (nonlinear advection)

THIS tutorial:
  - Stokes flow: Re → 0 (inertia terms REMOVED)
  - Governing: -∇p + μ∇²u = 0, ∇·u = 0 (LINEAR)
  - Application: microfluidics, lubrication, biological flows
  - Lid-driven cavity at Re→0 (no vortex breakdown)

Key difference:
  ┌──────────────────────┬──────────────────────────┐
  │ LDC (Re=100)          │ Stokes (Re→0)            │
  ├──────────────────────┼──────────────────────────┤
  │ ∂u/∂t + u·∇u = ...    │ 0 = -∇p + μ∇²u (no inertia)│
  │ Nonlinear              │ Linear                   │
  │ Corner vortices        │ Symmetric primary vortex │
  └──────────────────────┴──────────────────────────┘

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
print("PhysicsNeMo PINN Tutorial: Stokes Flow (Creeping Flow)")
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
MU = 1.0  # Dynamic viscosity
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0

print(f"\n[1] Stokes Flow Parameters:")
print(f"  Equation: -∇p + μ∇²u = 0, ∇·u = 0 (Re→0)")
print(f"  μ = {MU}, Re → 0 (creeping flow)")
print(f"  BC: Top wall u=1,v=0; Others u=0,v=0")

# ============================================================
# [2] PINN Model
# ============================================================
class StokesPINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 3]):
        super().__init__()
        self.act = nn.Tanh()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for m in self.layers:
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.act(self.layers[i](x))
        return self.layers[-1](x)  # (u, v, p)

print(f"\n[2] Model: 2 → 64 → 64 → 64 → 64 → 3 (u, v, p)")

# ============================================================
# [3] Collocation
# ============================================================
N_INT = 5000; N_BC = 2000
x_int = torch.rand(N_INT,1,device=device); y_int = torch.rand(N_INT,1,device=device)
xy_int = torch.cat([x_int,y_int],dim=1); xy_int.requires_grad_(True)

# Boundary: top (y=1, u=1), bottom (y=0, u=0), left (x=0, u=0), right (x=1, u=0)
n_per = N_BC//4
# Top
xt = torch.rand(n_per,1,device=device); yt = torch.ones(n_per,1,device=device)
xy_top = torch.cat([xt,yt],dim=1)
# Bottom
xb = torch.rand(n_per,1,device=device); yb = torch.zeros(n_per,1,device=device)
xy_bot = torch.cat([xb,yb],dim=1)
# Left
xl = torch.zeros(n_per,1,device=device); yl = torch.rand(n_per,1,device=device)
xy_left = torch.cat([xl,yl],dim=1)
# Right
xr = torch.ones(n_per,1,device=device); yr = torch.rand(n_per,1,device=device)
xy_right = torch.cat([xr,yr],dim=1)

print(f"\n[3] Collocation: {N_INT} interior, {N_BC} boundary")

# ============================================================
# [4] Loss
# ============================================================
def pde_loss(model, xy):
    u = model(xy)
    g = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    u_x,u_y = g[:,0:1],g[:,1:2]
    u_xx = torch.autograd.grad(u_x,xy,torch.ones_like(u_x),create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y,xy,torch.ones_like(u_y),create_graph=True)[0][:,1:2]
    v = u[:,1:2]; p = u[:,2:3]
    v_x = g[:,0:1]  # Actually need separate grad
    # Recompute properly
    uu = u[:,0:1]; vv = u[:,1:2]; pp = u[:,2:3]
    gu = torch.autograd.grad(uu,xy,torch.ones_like(uu),create_graph=True)[0]
    gv = torch.autograd.grad(vv,xy,torch.ones_like(vv),create_graph=True)[0]
    gp = torch.autograd.grad(pp,xy,torch.ones_like(pp),create_graph=True)[0]
    ux,uy = gu[:,0:1],gu[:,1:2]
    vx,vy = gv[:,0:1],gv[:,1:2]
    px,py = gp[:,0:1],gp[:,1:2]
    uxx = torch.autograd.grad(ux,xy,torch.ones_like(ux),create_graph=True)[0][:,0:1]
    uyy = torch.autograd.grad(uy,xy,torch.ones_like(uy),create_graph=True)[0][:,1:2]
    vxx = torch.autograd.grad(vx,xy,torch.ones_like(vx),create_graph=True)[0][:,0:1]
    vyy = torch.autograd.grad(vy,xy,torch.ones_like(vy),create_graph=True)[0][:,1:2]
    # Stokes: 0 = -dp/dx + μ∇²u, 0 = -dp/dy + μ∇²v, 0 = du/dx + dv/dy
    r1 = -px + MU*(uxx+uyy)
    r2 = -py + MU*(vxx+vyy)
    r3 = ux + vy  # continuity
    return (r1**2+r2**2+r3**2).mean()

def bc_loss(model, xy_top, xy_bot, xy_left, xy_right):
    l = 0.0
    # Top: u=1, v=0
    o = model(xy_top); l += ((o[:,0:1]-1)**2 + o[:,1:2]**2).mean()
    # Others: u=0, v=0
    for xy in [xy_bot, xy_left, xy_right]:
        o = model(xy); l += (o[:,0:2]**2).mean()
    return l

print(f"\n[4] Loss: Stokes PDE (linear) + BC")

# ============================================================
# [5] Training
# ============================================================
model = StokesPINN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
N_EPOCHS = 5000; L_PDE=1.0; L_BC=10.0
print(f"\n[5] Training: {N_EPOCHS} epochs")
hist = []
t0 = time.time()
for ep in range(N_EPOCHS):
    opt.zero_grad()
    lp = pde_loss(model, xy_int)
    lb = bc_loss(model, xy_top, xy_bot, xy_left, xy_right)
    loss = L_PDE*lp + L_BC*lb
    loss.backward(); opt.step(); sched.step()
    hist.append(loss.item())
    if (ep+1)%500==0:
        print(f"  Epoch {ep+1:5d}/{N_EPOCHS} | Loss: {loss.item():.6e} | PDE: {lp.item():.6e} | BC: {lb.item():.6e} | {time.time()-t0:.1f}s")
print(f"  Done! {time.time()-t0:.1f}s")

# ============================================================
# [6] Visualization
# ============================================================
print(f"\n[6] Visualization...")
NX,NY = 50,50
xe = torch.linspace(X_MIN,X_MAX,NX,device=device)
ye = torch.linspace(Y_MIN,Y_MAX,NY,device=device)
X,Y = torch.meshgrid(xe,ye,indexing='ij')
xy = torch.cat([X.reshape(-1,1),Y.reshape(-1,1)],dim=1)
model.eval()
with torch.no_grad():
    out = model(xy)
    U = out[:,0].cpu().numpy().reshape(NX,NY)
    V = out[:,1].cpu().numpy().reshape(NX,NY)
    P = out[:,2].cpu().numpy().reshape(NX,NY)

fig,axes = plt.subplots(2,2,figsize=(14,12))
im=axes[0,0].imshow(U.T,origin='lower',extent=[0,1,0,1],cmap='RdBu_r')
axes[0,0].set_title('Velocity U',fontsize=13,fontweight='bold'); plt.colorbar(im,ax=axes[0,0])
im=axes[0,1].imshow(V.T,origin='lower',extent=[0,1,0,1],cmap='RdBu_r')
axes[0,1].set_title('Velocity V',fontsize=13,fontweight='bold'); plt.colorbar(im,ax=axes[0,1])
im=axes[1,0].imshow(P.T,origin='lower',extent=[0,1,0,1],cmap='viridis')
axes[1,0].set_title('Pressure P',fontsize=13,fontweight='bold'); plt.colorbar(im,ax=axes[1,0])
X2=X.cpu().numpy(); Y2=Y.cpu().numpy()
axes[1,1].streamplot(X2.T,Y2.T,U.T,V.T,color='blue',density=1.5)
axes[1,1].set_title('Streamlines',fontsize=13,fontweight='bold')
plt.suptitle('Stokes Flow (Re→0): Lid-Driven Cavity',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,'stokes_result.png'),dpi=150,bbox_inches='tight')
plt.close()
print("  Saved: stokes_result.png")

fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.semilogy(hist,'b-',linewidth=2)
ax.set_xlabel('Epoch',fontsize=12); ax.set_ylabel('Loss',fontsize=12)
ax.set_title('Stokes Flow PINN Training Loss',fontsize=14,fontweight='bold')
ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,'stokes_loss.png'),dpi=150,bbox_inches='tight')
plt.close()
print("  Saved: stokes_loss.png")

# Concept
fig,ax = plt.subplots(1,1,figsize=(10,6))
ax.axis('off')
txt = (
    "Stokes Flow (Re → 0):\n\n"
    "  0 = -∇p + μ∇²u   (no inertia)\n"
    "  0 = ∇·u           (incompressible)\n\n"
    "vs. Navier-Stokes (existing, Re=100):\n\n"
    "  ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u\n"
    "  → Nonlinear (u·∇u)\n"
    "  → Corner vortices\n\n"
    "Stokes:\n"
    "  → Linear (no u·∇u term)\n"
    "  → Symmetric vortex\n"
    "  → Microfluidics, lubrication\n"
    "  → Biological flows (Re<<1)"
)
ax.text(0.05,0.95,txt,transform=ax.transAxes,fontsize=11,va='top',fontfamily='monospace',
        bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.8))
ax.set_title('Stokes vs Navier-Stokes',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,'stokes_concept.png'),dpi=150,bbox_inches='tight')
plt.close()
print("  Saved: stokes_concept.png")

# ============================================================
# [7] Summary
# ============================================================
print("\n"+"="*70)
print("[7] Summary")
print("="*70)
print(f"\n  Method: PINN for Stokes Flow")
print(f"  Physics: Re→0 (creeping/viscous flow)")
print(f"  Key: No inertia term (linear PDE)")
print(f"  Application: Microfluidics, lubrication, bio-fluids")
print(f"\n  Results saved to: {RESULTS_DIR}")
print("="*70)
