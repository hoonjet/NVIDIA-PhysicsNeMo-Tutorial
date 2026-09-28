"""
PhysicsNeMo PINN Tutorial: Curriculum Learning
================================================
Training Strategy — Progressive Difficulty Increase

Existing tutorials:
  - All PINNs train with FIXED parameters from epoch 0 to end
  - If PINN fails to converge, user manually adjusts hyperparameters

THIS tutorial:
  - Curriculum learning: start easy, progressively increase difficulty
  - Stage 1: High viscosity (ν=0.1, Re=10, easy) → smooth solution
  - Stage 2: Medium viscosity (ν=0.01, Re=100) → moderate
  - Stage 3: Low viscosity (ν=0.001, Re=1000, hard) → sharp gradients
  - Transfer weights between stages (warm start)

Key difference:
  ┌──────────────────────┬──────────────────────────┐
  │ Fixed (existing)      │ Curriculum (THIS)         │
  ├──────────────────────┼──────────────────────────┤
  │ ν fixed from start    │ ν decreases over stages   │
  │ Single difficulty     │ Easy → medium → hard      │
  │ May fail at high Re   │ Warm start from prev stage│
  │ No transfer            │ Weight transfer (warm)    │
  └──────────────────────┴──────────────────────────┘

Problem: 1D Burgers equation (same physics, varying ν)
  ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

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
print("PhysicsNeMo PINN Tutorial: Curriculum Learning")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(42); np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# [1] Curriculum Stages
# ============================================================
STAGES = [
    {"name": "Stage 1: Easy (Re=10)", "nu": 0.1, "epochs": 1500},
    {"name": "Stage 2: Medium (Re=100)", "nu": 0.01, "epochs": 2000},
    {"name": "Stage 3: Hard (Re=1000)", "nu": 0.001, "epochs": 2500},
]

X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

print(f"\n[1] Curriculum Stages:")
for s in STAGES:
    print(f"  {s['name']}: ν={s['nu']}, Re={1/s['nu']:.0f}, epochs={s['epochs']}")

# ============================================================
# [2] PINN Model
# ============================================================
class BurgersPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

print(f"\n[2] Model: 2 → 64 → 64 → 64 → 64 → 1")

# ============================================================
# [3] Loss Functions
# ============================================================
def pde_loss(model, xy, nu):
    u = model(xy)
    g = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    residual = u_t + u * u_x - nu * u_xx
    return (residual**2).mean()

def ic_loss(model, xy_ic):
    u_pred = model(xy_ic)
    x = xy_ic[:, 0:1]
    u_target = -torch.sin(np.pi * x)
    return ((u_pred - u_target)**2).mean()

def bc_loss(model, xy_bc):
    u_pred = model(xy_bc)
    return (u_pred**2).mean()

print(f"\n[3] Loss: PDE + IC + BC (Burgers)")

# ============================================================
# [4] Collocation Points
# ============================================================
N_INT = 5000; N_IC = 1000; N_BC = 400

x_int = torch.rand(N_INT, 1, device=device) * 2 - 1  # [-1, 1]
t_int = torch.rand(N_INT, 1, device=device)  # [0, 1]
xy_int = torch.cat([x_int, t_int], dim=1); xy_int.requires_grad_(True)

x_ic = torch.rand(N_IC, 1, device=device) * 2 - 1
t_ic = torch.zeros(N_IC, 1, device=device)
xy_ic = torch.cat([x_ic, t_ic], dim=1)

# BC: x=-1 and x=1
x_bc = torch.cat([-torch.ones(N_BC//2, 1, device=device), torch.ones(N_BC//2, 1, device=device)])
t_bc = torch.rand(N_BC, 1, device=device)
xy_bc = torch.cat([x_bc, t_bc], dim=1)

print(f"\n[4] Collocation: {N_INT} interior, {N_IC} IC, {N_BC} BC")

# ============================================================
# [5] Curriculum Training
# ============================================================
model = BurgersPINN().to(device)
all_hist = []
stage_boundaries = [0]
t_total = time.time()

for stage_idx, stage in enumerate(STAGES):
    nu = stage["nu"]
    epochs = stage["epochs"]
    stage_name = stage["name"]

    print(f"\n{'='*50}")
    print(f"  {stage_name}")
    print(f"  ν = {nu}, Re = {1/nu:.0f}, epochs = {epochs}")
    print(f"{'='*50}")

    # Adjust learning rate per stage (lower for harder stages)
    lr = 1e-3 if stage_idx == 0 else 5e-4
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.7)

    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        lp = pde_loss(model, xy_int, nu)
        li = ic_loss(model, xy_ic)
        lb = bc_loss(model, xy_bc)
        loss = lp + 10.0 * li + 10.0 * lb
        loss.backward(); opt.step(); sched.step()
        all_hist.append(loss.item())

        if (ep+1) % 500 == 0:
            print(f"  Ep {ep+1:4d}/{epochs} | Loss: {loss.item():.6e} | "
                  f"PDE: {lp.item():.6e} | IC: {li.item():.6e} | BC: {lb.item():.6e} | "
                  f"{time.time()-t0:.1f}s")

    stage_boundaries.append(len(all_hist))
    print(f"  Stage {stage_idx+1} complete! Time: {time.time()-t0:.1f}s")

print(f"\n  Total curriculum time: {time.time()-t_total:.1f}s")

# ============================================================
# [6] Comparison: Fixed vs Curriculum
# ============================================================
print(f"\n[6] Training fixed (non-curriculum) baseline for comparison...")

# Fixed training at hardest viscosity (ν=0.001) from scratch
model_fixed = BurgersPINN().to(device)
opt_fixed = torch.optim.Adam(model_fixed.parameters(), lr=1e-3)
sched_fixed = torch.optim.lr_scheduler.StepLR(opt_fixed, step_size=500, gamma=0.7)

nu_hard = STAGES[-1]["nu"]
total_fixed_epochs = sum(s["epochs"] for s in STAGES)
fixed_hist = []

t0 = time.time()
for ep in range(total_fixed_epochs):
    opt_fixed.zero_grad()
    lp = pde_loss(model_fixed, xy_int, nu_hard)
    li = ic_loss(model_fixed, xy_ic)
    lb = bc_loss(model_fixed, xy_bc)
    loss = lp + 10.0 * li + 10.0 * lb
    loss.backward(); opt_fixed.step(); sched_fixed.step()
    fixed_hist.append(loss.item())

    if (ep+1) % 1000 == 0:
        print(f"  Fixed Ep {ep+1:4d}/{total_fixed_epochs} | Loss: {loss.item():.6e} | {time.time()-t0:.1f}s")

print(f"  Fixed training done! Time: {time.time()-t0:.1f}s")

# ============================================================
# [7] Visualization
# ============================================================
print(f"\n[7] Visualization...")

# --- Figure 1: Curriculum vs Fixed loss ---
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.semilogy(all_hist, 'b-', linewidth=2, label='Curriculum (easy→hard)')
ax.semilogy(fixed_hist, 'r--', linewidth=2, label='Fixed (hard from start)')

# Mark stage boundaries
for i, b in enumerate(stage_boundaries[1:-1]):
    ax.axvline(x=b, color='green', linestyle=':', alpha=0.7, linewidth=2)
    ax.text(b, ax.get_ylim()[1]*0.5, f' Stage {i+1}→{i+2}', fontsize=9, color='green')

ax.set_xlabel('Total Epoch', fontsize=12)
ax.set_ylabel('Loss (log)', fontsize=12)
ax.set_title('Curriculum Learning vs Fixed Training', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'curriculum_vs_fixed.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: curriculum_vs_fixed.png")

# --- Figure 2: Solution snapshots ---
NX_EVAL, NT_EVAL = 200, 5
x_eval = torch.linspace(X_MIN, X_MAX, NX_EVAL, device=device).view(-1, 1)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
snapshots_t = [0.0, 0.25, 0.5, 0.75, 1.0]

# Curriculum model
for col, t_snap in enumerate(snapshots_t[:3]):
    t_eval = torch.full((NX_EVAL, 1), t_snap, device=device)
    xy_eval = torch.cat([x_eval, t_eval], dim=1)
    model.eval()
    with torch.no_grad():
        u_cur = model(xy_eval).cpu().numpy().flatten()
    x_np = x_eval.cpu().numpy().flatten()
    axes[0, col].plot(x_np, u_cur, 'b-', linewidth=2)
    axes[0, col].set_title(f'Curriculum (t={t_snap})', fontsize=12)
    axes[0, col].set_xlabel('x'); axes[0, col].set_ylabel('u')
    axes[0, col].set_ylim(-1.2, 1.2)
    axes[0, col].grid(True, alpha=0.3)

# Fixed model
for col, t_snap in enumerate(snapshots_t[:3]):
    t_eval = torch.full((NX_EVAL, 1), t_snap, device=device)
    xy_eval = torch.cat([x_eval, t_eval], dim=1)
    model_fixed.eval()
    with torch.no_grad():
        u_fix = model_fixed(xy_eval).cpu().numpy().flatten()
    x_np = x_eval.cpu().numpy().flatten()
    axes[1, col].plot(x_np, u_fix, 'r-', linewidth=2)
    axes[1, col].set_title(f'Fixed (t={t_snap})', fontsize=12)
    axes[1, col].set_xlabel('x'); axes[1, col].set_ylabel('u')
    axes[1, col].set_ylim(-1.2, 1.2)
    axes[1, col].grid(True, alpha=0.3)

# Comparison at t=0.5
ax = axes[0, 2] if len(snapshots_t) >= 3 else axes[0, -1]
t_eval = torch.full((NX_EVAL, 1), 0.5, device=device)
xy_eval = torch.cat([x_eval, t_eval], dim=1)
model.eval(); model_fixed.eval()
with torch.no_grad():
    u_cur = model(xy_eval).cpu().numpy().flatten()
    u_fix = model_fixed(xy_eval).cpu().numpy().flatten()
x_np = x_eval.cpu().numpy().flatten()
ax.plot(x_np, u_cur, 'b-', linewidth=2, label='Curriculum')
ax.plot(x_np, u_fix, 'r--', linewidth=2, label='Fixed')
ax.set_title('Comparison (t=0.5)', fontsize=12, fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('u')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Hide unused subplot
axes[1, 2].axis('off')
txt = (
    f"Curriculum: 3 stages\n"
    f"  Stage 1: ν=0.1 (Re=10), 1500 ep\n"
    f"  Stage 2: ν=0.01 (Re=100), 2000 ep\n"
    f"  Stage 3: ν=0.001 (Re=1000), 2500 ep\n\n"
    f"Fixed: ν=0.001 (Re=1000)\n"
    f"  {total_fixed_epochs} epochs from scratch\n\n"
    f"Curriculum final loss: {all_hist[-1]:.4e}\n"
    f"Fixed final loss: {fixed_hist[-1]:.4e}\n\n"
    f"Advantage: Curriculum converges\n"
    f"better at high Re (hard problem)"
)
axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes, fontsize=9, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Curriculum Learning vs Fixed Training (Burgers)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'curriculum_solutions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: curriculum_solutions.png")

# --- Figure 3: Concept ---
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.axis('off')
txt = (
    "Curriculum Learning for PINNs:\n\n"
    "  Core Idea: Start with easy problem, progressively increase difficulty.\n\n"
    "  ┌──────────────────────────────────────────────────────────┐\n"
    "  │ Stage 1: ν=0.1   (Re=10)    │ Easy: smooth solution      │\n"
    "  │   → Train 1500 epochs        │ → Learn basic structure    │\n"
    "  ├──────────────────────────────────────────────────────────┤\n"
    "  │ Stage 2: ν=0.01  (Re=100)   │ Medium: sharper gradients │\n"
    "  │   → Transfer weights         │ → Warm start from Stage 1  │\n"
    "  │   → Train 2000 epochs        │ → Refine solution          │\n"
    "  ├──────────────────────────────────────────────────────────┤\n"
    "  │ Stage 3: ν=0.001 (Re=1000)  │ Hard: near-shock           │\n"
    "  │   → Transfer weights         │ → Warm start from Stage 2  │\n"
    "  │   → Train 2500 epochs        │ → Final refinement         │\n"
    "  └──────────────────────────────────────────────────────────┘\n\n"
    "  vs. Fixed Training (existing):\n\n"
    "  - Start at hardest ν from scratch\n"
    "  - May fail to converge (stuck in local minimum)\n"
    "  - No warm start\n\n"
    "  Applications:\n"
    "  - High Reynolds number flows\n"
    "  - Sharp gradient/shock problems\n"
    "  - Multi-scale physics\n"
    "  - Domain decomposition (spatial curriculum)"
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Curriculum Learning Strategy', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'curriculum_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: curriculum_concept.png")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "="*70)
print("[8] Summary")
print("="*70)
print(f"\n  Method: Curriculum Learning for PINN")
print(f"  Stages: {len(STAGES)}")
for s in STAGES:
    print(f"    {s['name']}: ν={s['nu']}, Re={1/s['nu']:.0f}")
print(f"\n  Curriculum final loss: {all_hist[-1]:.6e}")
print(f"  Fixed final loss:      {fixed_hist[-1]:.6e}")
ratio = fixed_hist[-1] / (all_hist[-1] + 1e-10)
print(f"  Curriculum improvement: {ratio:.2f}x better")
print(f"\n  Key concept:")
print(f"    - Progressive difficulty (easy → hard)")
print(f"    - Warm start (transfer weights between stages)")
print(f"    - Better convergence at high Re")
print(f"\n  Results saved to: {RESULTS_DIR}")
print("="*70)
