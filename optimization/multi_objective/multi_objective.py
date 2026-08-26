"""
Multi-Objective Pareto Optimization for Airfoil Design
=======================================================
This tutorial implements multi-objective optimization using a
differentiable surrogate model to explore the Pareto front.

Existing tutorial (differentiable_design):
  - Single objective: minimize |Cp_pred - Cp_target|
  - One design variable: airfoil shape
  - One optimal solution

THIS tutorial:
  - Multi-objective: maximize lift, minimize drag, maximize thickness
  - Pareto front exploration via weighted sum method
  - Multiple optimal solutions (trade-off curve)

Key concepts:
  1. Pareto optimality: no single "best" — trade-offs between objectives
  2. Weighted sum method: scalarize multi-objective → single objective
  3. Pareto front: set of non-dominated solutions
  4. Surrogate backprop: gradient flows to INPUT (design variables)
  5. Trade-off analysis: visualize competing objectives

Author: PhysicsNeMo Tutorial
Date: 2026-08-24
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("Multi-Objective Pareto Optimization for Airfoil Design")
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
# [1] Problem Setup: Airfoil Design
# ============================================================
# Design variables: 16 control points defining airfoil shape
# Objectives:
#   1. Maximize lift coefficient (Cl) — aerodynamic performance
#   2. Minimize drag coefficient (Cd) — fuel efficiency
#   3. Maximize thickness — structural strength
#
# These objectives CONFLICT:
#   - Thicker airfoil → more lift but more drag
#   - Thinner airfoil → less drag but less lift and less strength
#
# We use a surrogate model to predict Cl, Cd, thickness from shape,
# then optimize shape via gradient backprop for different weight combinations.

N_POINTS = 16  # Control points defining airfoil shape
N_TRAIN = 500  # Training samples for surrogate

print(f"\n[1] Problem: Multi-objective airfoil design")
print(f"  Design variables: {N_POINTS} control points (airfoil shape)")
print(f"  Objectives: maximize Cl, minimize Cd, maximize thickness")
print(f"  Method: Pareto front via weighted sum + surrogate backprop")

# ============================================================
# [2] Generate Training Data for Surrogate
# ============================================================
# We create a simplified airfoil model:
# - Shape: y(x) = sum of basis functions with control point amplitudes
# - Cl ~ f(shape, angle_of_attack) — simplified potential flow
# - Cd ~ f(shape, thickness) — simplified drag model
# - Thickness = max(y_upper - y_lower)

def generate_airfoil_shape(control_points, n=64):
    """Generate airfoil surface from control points."""
    x = np.linspace(0, 1, n)
    # Use cosine spacing for better leading/trailing edge resolution
    x_cos = 0.5 * (1 - np.cos(np.pi * x))
    # Interpolate control points
    cp_x = np.linspace(0, 1, len(control_points))
    y_upper = np.interp(x_cos, cp_x, control_points)
    y_lower = -np.interp(x_cos, cp_x, control_points)
    return x_cos, y_upper, y_lower

def compute_aero_metrics(control_points, aoa=2.0):
    """Simplified aerodynamic metrics (surrogate for CFD)."""
    x, y_upper, y_lower = generate_airfoil_shape(control_points)
    thickness = np.max(y_upper - y_lower)
    camber = 0.5 * (y_upper + y_lower)
    # Simplified Cl: thin airfoil theory + camber effect
    cl = 2 * np.pi * np.radians(aoa) + 4 * np.mean(camber) * np.pi
    # Simplified Cd: form drag ~ thickness^2 + skin friction
    cd = 0.01 + 0.5 * thickness**2 + 0.005 * np.mean(y_upper**2)
    return cl, cd, thickness

print(f"\n[2] Generating {N_TRAIN} training samples for surrogate...")

# Generate random airfoil shapes
train_shapes = np.random.randn(N_TRAIN, N_POINTS) * 0.05 + 0.1
train_shapes[:, 0] = 0.0   # Leading edge: sharp
train_shapes[:, -1] = 0.0  # Trailing edge: sharp
train_shapes = np.clip(train_shapes, 0.001, 0.3)

# Compute metrics
train_cl = np.array([compute_aero_metrics(s)[0] for s in train_shapes])
train_cd = np.array([compute_aero_metrics(s)[1] for s in train_shapes])
train_th = np.array([compute_aero_metrics(s)[2] for s in train_shapes])

# Normalize
sh_mean, sh_std = train_shapes.mean(), train_shapes.std()
cl_mean, cl_std = train_cl.mean(), train_cl.std()
cd_mean, cd_std = train_cd.mean(), train_cd.std()
th_mean, th_std = train_th.mean(), train_th.std()

train_sh_n = (train_shapes - sh_mean) / (sh_std + 1e-8)
train_cl_n = (train_cl - cl_mean) / (cl_std + 1e-8)
train_cd_n = (train_cd - cd_mean) / (cd_std + 1e-8)
train_th_n = (train_th - th_mean) / (th_std + 1e-8)

train_sh_t = torch.from_numpy(train_sh_n).float().to(device)
train_cl_t = torch.from_numpy(train_cl_n).float().to(device)
train_cd_t = torch.from_numpy(train_cd_n).float().to(device)
train_th_t = torch.from_numpy(train_th_n).float().to(device)

print(f"  Cl range: [{train_cl.min():.4f}, {train_cl.max():.4f}]")
print(f"  Cd range: [{train_cd.min():.4f}, {train_cd.max():.4f}]")
print(f"  Thickness range: [{train_th.min():.4f}, {train_th.max():.4f}]")

# ============================================================
# [3] Surrogate Model
# ============================================================
# Input: airfoil shape (16 points)
# Output: Cl, Cd, thickness (3 values)

class SurrogateNet(nn.Module):
    """MLP surrogate: shape → [Cl, Cd, thickness]"""
    def __init__(self, in_dim=16, hidden=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

print(f"\n[3] Building surrogate model...")
surrogate = SurrogateNet(in_dim=N_POINTS, hidden=128, out_dim=3).to(device)
n_params = sum(p.numel() for p in surrogate.parameters())
print(f"  Surrogate parameters: {n_params:,}")
print(f"  Input: {N_POINTS} shape points → Output: [Cl, Cd, thickness]")

# ============================================================
# [4] Train Surrogate
# ============================================================
EPOCHS_SURR = 200
BATCH_SIZE = 64
LR = 1e-3

print(f"\n[4] Training surrogate ({EPOCHS_SURR} epochs)")
print("-" * 70)

opt = torch.optim.Adam(surrogate.parameters(), lr=LR)
surr_losses = []
start = time.time()

for epoch in range(EPOCHS_SURR):
    surrogate.train()
    perm = torch.randperm(N_TRAIN)
    epoch_loss = 0; n_b = 0
    for i in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        sh = train_sh_t[idx]
        targets = torch.stack([train_cl_t[idx], train_cd_t[idx], train_th_t[idx]], dim=1)
        preds = surrogate(sh)
        loss = F.mse_loss(preds, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_b += 1
    surr_losses.append(epoch_loss / n_b)
    if epoch % 50 == 0 or epoch == EPOCHS_SURR - 1:
        print(f"  Epoch {epoch:4d} | Loss: {surr_losses[-1]:.6e} | Time: {time.time()-start:.1f}s")

surr_time = time.time() - start
print("-" * 70)

# ============================================================
# [5] Multi-Objective Pareto Optimization
# ============================================================
# We optimize the airfoil shape for different weight combinations.
# Objectives (all minimized):
#   f1 = -Cl  (maximize lift → minimize negative lift)
#   f2 =  Cd  (minimize drag)
#   f3 = -th  (maximize thickness → minimize negative thickness)
#
# Weighted sum: L = w1*f1 + w2*f2 + w3*f3
# We sweep different weight combinations to trace the Pareto front.

print(f"\n[5] Multi-objective Pareto optimization")
print("-" * 70)

N_PARETO = 30  # Number of Pareto points
N_OPT_STEPS = 300
LR_OPT = 0.01

# Weight combinations: sweep w1 (lift weight) from 0 to 1
# w2 (drag) = (1-w1) * (1-w3), w3 (thickness) = (1-w1) * w3_frac
pareto_results = []

for i in range(N_PARETO):
    # Weight for lift vs drag trade-off
    w_lift = i / (N_PARETO - 1)  # 0 to 1
    w_drag = 1 - w_lift
    # Fixed thickness weight (small, as secondary objective)
    w_thick = 0.1

    # Normalize weights
    w_sum = w_lift + w_drag + w_thick
    w1, w2, w3 = w_lift/w_sum, w_drag/w_sum, w_thick/w_sum

    # Initialize design from mean shape
    design = torch.zeros(1, N_POINTS, device=device, requires_grad=True)
    opt_design = torch.optim.Adam([design], lr=LR_OPT)

    for step in range(N_OPT_STEPS):
        # Predict metrics
        preds = surrogate(design)  # [1, 3]
        cl_pred, cd_pred, th_pred = preds[0, 0], preds[0, 1], preds[0, 2]

        # Multi-objective loss (minimize)
        loss = w1 * (-cl_pred) + w2 * (cd_pred) + w3 * (-th_pred)

        opt_design.zero_grad()
        loss.backward()
        opt_design.step()

        # Clip design to feasible range
        with torch.no_grad():
            design.data = torch.clamp(design.data, -2.0, 2.0)
            design.data[:, 0] = 0.0   # Leading edge
            design.data[:, -1] = 0.0  # Trailing edge

    # Get final metrics (denormalized)
    with torch.no_grad():
        preds = surrogate(design)
        cl_final = preds[0, 0].item() * cl_std + cl_mean
        cd_final = preds[0, 1].item() * cd_std + cd_mean
        th_final = preds[0, 2].item() * th_std + th_mean
        shape_final = design[0].cpu().numpy() * sh_std + sh_mean

    pareto_results.append({
        'w_lift': w_lift, 'w_drag': w_drag, 'w_thick': w_thick,
        'cl': cl_final, 'cd': cd_final, 'th': th_final,
        'shape': shape_final
    })

    if i % 5 == 0:
        print(f"  Pareto {i:3d} | w_lift={w_lift:.2f} | Cl={cl_final:.4f} Cd={cd_final:.4f} th={th_final:.4f}")

print("-" * 70)

# ============================================================
# [6] Pareto Front Analysis
# ============================================================
print(f"\n[6] Pareto front analysis")

# Extract Pareto-optimal solutions (non-dominated)
cl_arr = np.array([r['cl'] for r in pareto_results])
cd_arr = np.array([r['cd'] for r in pareto_results])
th_arr = np.array([r['th'] for r in pareto_results])

# Non-dominated sorting (simplified: check if any solution dominates)
is_pareto = np.ones(N_PARETO, dtype=bool)
for i in range(N_PARETO):
    for j in range(N_PARETO):
        if i == j:
            continue
        # j dominates i if j is better or equal in all objectives and strictly better in at least one
        # Objectives: maximize Cl, minimize Cd, maximize th
        j_dominates = (cl_arr[j] >= cl_arr[i] and cd_arr[j] <= cd_arr[i] and th_arr[j] >= th_arr[i]
                       and (cl_arr[j] > cl_arr[i] or cd_arr[j] < cd_arr[i] or th_arr[j] > th_arr[i]))
        if j_dominates:
            is_pareto[i] = False
            break

n_pareto = is_pareto.sum()
print(f"  Pareto-optimal solutions: {n_pareto}/{N_PARETO}")

# ============================================================
# [7] Visualization
# ============================================================
print(f"\n[7] Generating visualizations...")

# --- Figure 1: Surrogate training loss ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(surr_losses, linewidth=1.5, color='blue')
ax.set_xlabel('Epoch'); ax.set_ylabel('Surrogate Loss (MSE)')
ax.set_title('Surrogate Model Training Loss'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "multi_obj_surrogate_loss.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Pareto front (3D) ---
fig = plt.figure(figsize=(18, 6))

# 3D Pareto front
ax1 = fig.add_subplot(131, projection='3d')
colors = ['red' if p else 'blue' for p in is_pareto]
sizes = [80 if p else 20 for p in is_pareto]
ax1.scatter(cl_arr, cd_arr, th_arr, c=colors, s=sizes, alpha=0.7)
ax1.set_xlabel('Cl (lift)'); ax1.set_ylabel('Cd (drag)'); ax1.set_zlabel('Thickness')
ax1.set_title('3D Pareto Front\n(red=Pareto optimal)')

# 2D: Cl vs Cd
ax2 = fig.add_subplot(132)
ax2.scatter(cl_arr[~is_pareto], cd_arr[~is_pareto], c='blue', s=20, alpha=0.5, label='Dominated')
ax2.scatter(cl_arr[is_pareto], cd_arr[is_pareto], c='red', s=80, alpha=0.8, label='Pareto optimal', edgecolors='black')
ax2.set_xlabel('Cl (lift coefficient)'); ax2.set_ylabel('Cd (drag coefficient)')
ax2.set_title('Pareto Front: Lift vs Drag'); ax2.legend(); ax2.grid(True, alpha=0.3)

# 2D: Cl vs Thickness
ax3 = fig.add_subplot(133)
ax3.scatter(cl_arr[~is_pareto], th_arr[~is_pareto], c='blue', s=20, alpha=0.5, label='Dominated')
ax3.scatter(cl_arr[is_pareto], th_arr[is_pareto], c='red', s=80, alpha=0.8, label='Pareto optimal', edgecolors='black')
ax3.set_xlabel('Cl (lift coefficient)'); ax3.set_ylabel('Thickness')
ax3.set_title('Pareto Front: Lift vs Thickness'); ax3.legend(); ax3.grid(True, alpha=0.3)

plt.suptitle('Multi-Objective Pareto Front: Airfoil Design', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "multi_obj_pareto_front.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: Airfoil shapes along Pareto front ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
pareto_indices = np.where(is_pareto)[0]
n_show = min(6, len(pareto_indices))
for idx in range(n_show):
    ax = axes[idx // 3, idx % 3]
    i = pareto_indices[idx]
    r = pareto_results[i]
    x, y_upper, y_lower = generate_airfoil_shape(r['shape'])
    ax.fill_between(x, y_lower, y_upper, alpha=0.3, color='skyblue')
    ax.plot(x, y_upper, 'b-', linewidth=2, label='Upper surface')
    ax.plot(x, y_lower, 'b-', linewidth=2, label='Lower surface')
    ax.set_title(f"w_lift={r['w_lift']:.2f}\nCl={r['cl']:.3f} Cd={r['cd']:.4f} th={r['th']:.3f}", fontsize=10)
    ax.set_aspect('equal'); ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

plt.suptitle('Airfoil Shapes Along Pareto Front', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "multi_obj_airfoils.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: Trade-off analysis ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Trade-off: Cl vs Cd (color = thickness)
sc1 = axes[0].scatter(cl_arr, cd_arr, c=th_arr, cmap='viridis', s=60, alpha=0.8)
axes[0].set_xlabel('Cl (lift)'); axes[0].set_ylabel('Cd (drag)')
axes[0].set_title('Trade-off: Lift vs Drag\n(color = thickness)')
plt.colorbar(sc1, ax=axes[0], label='Thickness')
axes[0].grid(True, alpha=0.3)

# Trade-off: Cl vs thickness (color = Cd)
sc2 = axes[1].scatter(cl_arr, th_arr, c=cd_arr, cmap='plasma', s=60, alpha=0.8)
axes[1].set_xlabel('Cl (lift)'); axes[1].set_ylabel('Thickness')
axes[1].set_title('Trade-off: Lift vs Thickness\n(color = Cd)')
plt.colorbar(sc2, ax=axes[1], label='Cd')
axes[1].grid(True, alpha=0.3)

# Weight sweep effect
w_lifts = [r['w_lift'] for r in pareto_results]
axes[2].plot(w_lifts, cl_arr / (cl_arr.max() + 1e-8), 'o-', label='Cl (normalized)', markersize=4)
axes[2].plot(w_lifts, cd_arr / (cd_arr.max() + 1e-8), 's-', label='Cd (normalized)', markersize=4)
axes[2].plot(w_lifts, th_arr / (th_arr.max() + 1e-8), '^-', label='Thickness (normalized)', markersize=4)
axes[2].set_xlabel('Weight on Lift (w_lift)'); axes[2].set_ylabel('Normalized Objective')
axes[2].set_title('Effect of Weight on Objectives'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.suptitle('Multi-Objective Trade-off Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "multi_obj_tradeoff.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 5: Concept explanation ---
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.text(0.5, 0.95, 'Multi-Objective Pareto Optimization', ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.80,
    'Single-Objective (existing tutorial):\n'
    '  - One objective: minimize |Cp - target|\n'
    '  - One optimal solution\n'
    '  - No trade-offs\n'
    '  - Gradient descent to single minimum',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(0.55, 0.80,
    'Multi-Objective (THIS tutorial):\n'
    '  - 3 objectives: max Cl, min Cd, max thickness\n'
    '  - Pareto front (multiple optima)\n'
    '  - Trade-offs between objectives\n'
    '  - Weighted sum sweep',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.50,
    'Pareto optimality:\n'
    '  Solution A dominates B if:\n'
    '    A is >= B in ALL objectives\n'
    '    A is > B in at least ONE\n'
    '  Pareto front = non-dominated set\n'
    '  No "best" — depends on preference',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.55, 0.50,
    'Weighted sum method:\n'
    '  L = w1*(-Cl) + w2*(Cd) + w3*(-th)\n'
    '  Sweep w1, w2, w3 to trace front\n'
    '  Each weight → one Pareto point\n'
    '  Convex fronts: exact\n'
    '  Non-convex: may miss points',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.05, 0.20,
    'Engineering relevance:\n'
    '  - Real design: multiple competing goals\n'
    '  - No single "best" design\n'
    '  - Designer chooses from Pareto front\n'
    '  - Examples: speed vs efficiency,\n'
    '    strength vs weight, cost vs performance',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax.text(0.55, 0.20,
    'Key concepts:\n'
    '  1. Pareto dominance\n'
    '  2. Non-dominated sorting\n'
    '  3. Weighted sum scalarization\n'
    '  4. Surrogate backprop to input\n'
    '  5. Trade-off visualization',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightskyblue', alpha=0.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "multi_obj_explanation.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Multi-Objective Pareto Optimization")
print("=" * 70)
print(f"  Problem:           Airfoil design (3 objectives)")
print(f"  Design variables:  {N_POINTS} control points")
print(f"  Objectives:        max Cl, min Cd, max thickness")
print(f"  Surrogate train:   {EPOCHS_SURR} epochs, {surr_time:.1f}s")
print(f"  Pareto points:     {N_PARETO} weight combinations")
print(f"  Pareto-optimal:    {n_pareto}/{N_PARETO} non-dominated")
print()
print("Key observations:")
print("  1. TRADE-OFFS: Higher lift → more drag; thicker → more drag but stronger")
print("  2. PARETO FRONT: Multiple optimal solutions, no single 'best'")
print("  3. WEIGHTED SUM: Sweep weights to trace Pareto front")
print("  4. SURROGATE BACKPROP: Gradient flows to INPUT (design variables)")
print("  5. vs SINGLE-OBJECTIVE: Existing tutorial finds ONE optimum; this finds a FRONT")
print("  6. ENGINEERING: Real design always has competing objectives")
print("=" * 70)
