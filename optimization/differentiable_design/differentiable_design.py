"""
PhysicsNeMo Optimization Tutorial: Differentiable Design Optimization
======================================================================
AI-based inverse design using gradient backpropagation through a surrogate model.

All existing tutorials in this repo are FORWARD problems: given input → predict output.
This is the ONLY tutorial that solves the INVERSE DESIGN problem: given desired
output → optimize input (design variables) via gradient descent.

Problem: Aerodynamic Shape Optimization
    - Surrogate model: CNN that maps airfoil shape → pressure coefficient (Cp)
    - Forward: shape → Cp prediction (standard supervised learning)
    - Inverse design: desired Cp → optimize shape via backprop through CNN
    - The shape is treated as a DIFFERENTIABLE INPUT and optimized with gradient descent

Key concepts:
    - Surrogate-based optimization: replace expensive CFD with fast neural net
    - Input gradient: backpropagate loss w.r.t. INPUT (not weights)
    - Projected gradient descent: constrain design variables to feasible range
    - Multi-objective: match target Cp + maintain shape smoothness
    - Differentiable physics: the entire pipeline is end-to-end differentiable

This is fundamentally different from:
    - PINN Inverse Problem (estimates scalar PDE parameters, not shape functions)
    - FNO/U-Net (forward prediction only, no design optimization)
    - Topology Optimization (uses diffusion model, not gradient-based inverse design)

Author: PhysicsNeMo Tutorial
Date: 2026-08-06
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

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo Optimization Tutorial: Differentiable Design Optimization")
print("Aerodynamic Shape Inverse Design via Surrogate Backpropagation")
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
# [1] Data Generation: Synthetic Airfoil Cp Database
# ============================================================
# We create a synthetic database of airfoil shapes and their Cp distributions.
# Shape: parameterized by NACA 4-digit coefficients + perturbations
# Cp: computed via a simplified panel-method-like analytical model

N_POINTS = 64  # surface points (half airfoil, upper surface)

def naca_thickness(x, t_max):
    """NACA 4-digit thickness distribution."""
    return 5 * t_max * (0.2969 * np.sqrt(x) - 0.1260 * x
                        - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

def generate_airfoil_shape(n_samples, n_points, t_max_range=(0.06, 0.18),
                           camber_range=(0.0, 0.05), perturb_scale=0.01):
    """
    Generate random airfoil upper surface shapes.
    Parameterization: NACA thickness + camber + smooth random perturbation.
    """
    x = np.linspace(0, 1, n_points)
    shapes = []
    for _ in range(n_samples):
        t_max = np.random.uniform(*t_max_range)
        camber = np.random.uniform(*camber_range)
        camber_pos = np.random.uniform(0.2, 0.8)

        # Thickness
        thickness = naca_thickness(x, t_max)
        # Camber
        camber_line = np.where(x < camber_pos,
                               camber / camber_pos**2 * (2 * camber_pos * x - x**2),
                               camber / (1 - camber_pos)**2 * ((1 - 2 * camber_pos) + 2 * camber_pos * x - x**2))
        # Smooth random perturbation (low-frequency)
        perturb_freq = np.random.randint(2, 5)
        perturb_phase = np.random.rand(perturb_freq) * 2 * np.pi
        perturb_amp = np.random.rand(perturb_freq) * perturb_scale
        perturb = np.zeros(n_points)
        for k in range(perturb_freq):
            perturb += perturb_amp[k] * np.sin(2 * np.pi * (k + 1) * x + perturb_phase[k])

        y = thickness + camber_line + perturb
        y[0] = 0.0  # leading edge
        y[-1] = 0.001  # trailing edge
        shapes.append(y)
    return np.array(shapes, dtype=np.float32)


def compute_cp_thin_airfoil(shape, n_points, alpha_deg=2.0):
    """
    Simplified Cp distribution using thin airfoil theory + thickness correction.
    This is a fast analytical surrogate for the "true" CFD solution.
    """
    alpha = np.radians(alpha_deg)
    x = np.linspace(0, 1, n_points)

    # Thin airfoil: Cp = -2 * d(y)/dx (simplified)
    dy = np.gradient(shape, x)
    cp = -2 * dy * np.cos(alpha) - 0.5 * shape * 10  # thickness effect

    # Leading edge suction peak
    cp[0] = -2.0
    # Smooth
    from scipy.ndimage import gaussian_filter1d
    cp = gaussian_filter1d(cp, sigma=2)

    # Add angle of attack effect
    cp = cp - 2 * np.pi * alpha * np.sqrt(1 - x)

    return cp.astype(np.float32)


print("\n[1] Generating synthetic airfoil database...")
N_TRAIN = 500
N_TEST = 50

# Check if scipy is available
try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("  (scipy not available, using numpy smoothing)")

if not HAS_SCIPY:
    # Override compute_cp to use numpy smoothing
    def compute_cp_thin_airfoil(shape, n_points, alpha_deg=2.0):
        alpha = np.radians(alpha_deg)
        x = np.linspace(0, 1, n_points)
        dy = np.gradient(shape, x)
        cp = -2 * dy * np.cos(alpha) - 0.5 * shape * 10
        cp[0] = -2.0
        # Simple moving average smoothing
        kernel = np.ones(5) / 5
        cp = np.convolve(cp, kernel, mode='same')
        cp = cp - 2 * np.pi * alpha * np.sqrt(np.maximum(1 - x, 0))
        return cp.astype(np.float32)

train_shapes = generate_airfoil_shape(N_TRAIN, N_POINTS)
train_cps = np.array([compute_cp_thin_airfoil(s, N_POINTS) for s in train_shapes])

test_shapes = generate_airfoil_shape(N_TEST, N_POINTS)
test_cps = np.array([compute_cp_thin_airfoil(s, N_POINTS) for s in test_shapes])

print(f"  Train: {N_TRAIN} samples, shape: {train_shapes.shape}, Cp: {train_cps.shape}")
print(f"  Test:  {N_TEST} samples")
print(f"  Shape range: [{train_shapes.min():.4f}, {train_shapes.max():.4f}]")
print(f"  Cp range:    [{train_cps.min():.4f}, {train_cps.max():.4f}]")

# Normalize
s_mean, s_std = train_shapes.mean(), train_shapes.std()
c_mean, c_std = train_cps.mean(), train_cps.std()

train_s_norm = (train_shapes - s_mean) / (s_std + 1e-8)
train_c_norm = (train_cps - c_mean) / (c_std + 1e-8)
test_s_norm = (test_shapes - s_mean) / (s_std + 1e-8)
test_c_norm = (test_cps - c_mean) / (c_std + 1e-8)

train_s_t = torch.from_numpy(train_s_norm).unsqueeze(1).to(device)  # [N, 1, L]
train_c_t = torch.from_numpy(train_c_norm).unsqueeze(1).to(device)
test_s_t = torch.from_numpy(test_s_norm).unsqueeze(1).to(device)
test_c_t = torch.from_numpy(test_c_norm).unsqueeze(1).to(device)

# ============================================================
# [2] Surrogate Model: 1D CNN (Shape → Cp)
# ============================================================
class SurrogateCNN(nn.Module):
    """
    1D CNN that maps airfoil shape → Cp distribution.
    Input:  [B, 1, N_POINTS]  (airfoil upper surface y-coordinates)
    Output: [B, 1, N_POINTS]  (pressure coefficient Cp)
    """
    def __init__(self, base_ch=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, base_ch, 7, padding=3),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv1d(base_ch, base_ch * 2, 5, stride=2, padding=2),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.Conv1d(base_ch * 2, base_ch * 4, 5, stride=2, padding=2),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.ConvTranspose1d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv1d(base_ch, 1, 7, padding=3),
        )

    def forward(self, x):
        # x: [B, 1, N_POINTS]
        h = self.encoder(x)
        out = self.decoder(h)
        # Ensure output length matches input
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear', align_corners=True)
        return out


surrogate = SurrogateCNN(base_ch=32).to(device)
n_params = sum(p.numel() for p in surrogate.parameters())
print(f"\n[2] Surrogate CNN parameters: {n_params:,}")

# ============================================================
# [3] Train Surrogate: Shape → Cp (Forward Model)
# ============================================================
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3

optimizer = torch.optim.Adam(surrogate.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\n[3] Training surrogate model ({EPOCHS} epochs)...")
print("-" * 70)

train_losses = []
test_losses = []
start_time = time.time()

for epoch in range(EPOCHS):
    surrogate.train()
    epoch_loss = 0
    n_batches = 0

    perm = torch.randperm(N_TRAIN)
    for i in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        s_batch = train_s_t[idx]
        c_batch = train_c_t[idx]

        pred = surrogate(s_batch)
        loss = F.mse_loss(pred, c_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)

    # Test
    surrogate.eval()
    with torch.no_grad():
        test_pred = surrogate(test_s_t)
        test_loss = F.mse_loss(test_pred, test_c_t).item()
    test_losses.append(test_loss)

    scheduler.step()

    if epoch % 40 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:4d}/{EPOCHS} | Train: {avg_loss:.6e} | Test: {test_loss:.6e} | Time: {elapsed:.1f}s")

total_train_time = time.time() - start_time
print("-" * 70)
print(f"Surrogate training complete! Time: {total_train_time:.1f}s")

# ============================================================
# [4] Inverse Design: Optimize Shape to Match Target Cp
# ============================================================
# Now we FREEZE the surrogate and optimize the INPUT (shape) via backprop.

print("\n[4] Inverse Design: Optimizing airfoil shape to match target Cp...")
print("-" * 70)

# Select a target Cp from test set
target_idx = 5
target_cp = test_c_t[target_idx:target_idx + 1]  # [1, 1, N_POINTS]
target_shape_truth = test_s_t[target_idx:target_idx + 1]  # ground truth shape

# Initialize design from a random shape (NOT the target)
init_shape = train_s_t[0:1].clone()

# Make shape a leaf tensor that requires gradient
design_shape = init_shape.clone().detach().requires_grad_(True)

# Freeze surrogate
for p in surrogate.parameters():
    p.requires_grad = False
surrogate.eval()

# Design optimization
DESIGN_LR = 0.05
DESIGN_STEPS = 500
design_optimizer = torch.optim.Adam([design_shape], lr=DESIGN_LR)

design_losses = []
shape_diffs = []
start_time = time.time()

for step in range(DESIGN_STEPS):
    design_optimizer.zero_grad()

    # Forward: shape → Cp
    pred_cp = surrogate(design_shape)

    # Loss 1: Match target Cp
    loss_match = F.mse_loss(pred_cp, target_cp)

    # Loss 2: Smoothness (penalize rough shapes)
    smoothness = torch.mean(torch.abs(torch.diff(design_shape, dim=-1)))

    # Loss 3: Leading/trailing edge constraint
    edge_loss = design_shape[:, :, 0]**2 + (design_shape[:, :, -1] - 0.0)**2

    loss = loss_match + 0.01 * smoothness + 0.1 * edge_loss
    loss.backward()
    design_optimizer.step()

    # Project to feasible range (non-negative thickness)
    with torch.no_grad():
        design_shape.clamp_(min=train_s_norm.min(), max=train_s_norm.max())

    design_losses.append(loss_match.item())

    # Track shape difference
    with torch.no_grad():
        shape_diff = F.mse_loss(design_shape, target_shape_truth).item()
    shape_diffs.append(shape_diff)

    if step % 100 == 0 or step == DESIGN_STEPS - 1:
        elapsed = time.time() - start_time
        print(f"Step {step:4d}/{DESIGN_STEPS} | Cp Loss: {loss_match.item():.6e} "
              f"| Shape Diff: {shape_diff:.6e} | Time: {elapsed:.1f}s")

design_time = time.time() - start_time
print("-" * 70)
print(f"Inverse design complete! Time: {design_time:.1f}s")

# ============================================================
# [5] Evaluate: Compare Designed Shape vs Ground Truth
# ============================================================
print("\n[5] Evaluating inverse design results...")

with torch.no_grad():
    # Designed shape's Cp
    designed_cp = surrogate(design_shape)

    # Denormalize
    designed_shape_dn = design_shape.squeeze().cpu().numpy() * s_std + s_mean
    target_shape_dn = target_shape_truth.squeeze().cpu().numpy() * s_std + s_mean
    init_shape_dn = init_shape.squeeze().cpu().numpy() * s_std + s_mean

    designed_cp_dn = designed_cp.squeeze().cpu().numpy() * c_std + c_mean
    target_cp_dn = target_cp.squeeze().cpu().numpy() * c_std + c_mean

    # Metrics
    cp_l2 = np.linalg.norm(designed_cp_dn - target_cp_dn) / (np.linalg.norm(target_cp_dn) + 1e-8)
    shape_l2 = np.linalg.norm(designed_shape_dn - target_shape_dn) / (np.linalg.norm(target_shape_dn) + 1e-8)

print(f"  Cp L2 error:     {cp_l2:.4f} ({cp_l2*100:.1f}%)")
print(f"  Shape L2 error:   {shape_l2:.4f} ({shape_l2*100:.1f}%)")
print(f"  Initial shape L2: {np.linalg.norm(init_shape_dn - target_shape_dn) / (np.linalg.norm(target_shape_dn) + 1e-8):.4f}")

# ============================================================
# [6] Visualization
# ============================================================
print("\n[6] Generating visualizations...")

# --- Figure 1: Surrogate training loss ---
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.semilogy(train_losses, label='Train', linewidth=1.5)
ax.semilogy(test_losses, label='Test', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE, log scale)')
ax.set_title('Surrogate Model Training (Shape → Cp)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
loss_path = os.path.join(RESULTS_DIR, "surrogate_loss.png")
plt.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {loss_path}")

# --- Figure 2: Inverse design convergence ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(design_losses, linewidth=1.5, color='blue')
ax1.set_xlabel('Optimization Step')
ax1.set_ylabel('Cp Match Loss (MSE)')
ax1.set_title('Inverse Design: Cp Matching Convergence')
ax1.grid(True, alpha=0.3)

ax2.semilogy(shape_diffs, linewidth=1.5, color='green')
ax2.set_xlabel('Optimization Step')
ax2.set_ylabel('Shape L2 Diff (to ground truth)')
ax2.set_title('Inverse Design: Shape Recovery Convergence')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
conv_path = os.path.join(RESULTS_DIR, "design_convergence.png")
plt.savefig(conv_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {conv_path}")

# --- Figure 3: Shape comparison ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Shape comparison
ax = axes[0, 0]
x = np.linspace(0, 1, N_POINTS)
ax.plot(x, init_shape_dn, 'b--', linewidth=2, label='Initial (random)', alpha=0.7)
ax.plot(x, designed_shape_dn, 'r-', linewidth=2.5, label='Designed (optimized)')
ax.plot(x, target_shape_dn, 'g-', linewidth=2, label='Target (ground truth)', alpha=0.8)
ax.set_xlabel('x (chord)')
ax.set_ylabel('y (thickness)')
ax.set_title('Airfoil Shape: Initial vs Designed vs Target')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Cp comparison
ax = axes[0, 1]
ax.plot(x, designed_cp_dn, 'r-', linewidth=2.5, label='Designed Cp')
ax.plot(x, target_cp_dn, 'g--', linewidth=2, label='Target Cp', alpha=0.8)
ax.set_xlabel('x (chord)')
ax.set_ylabel('Cp')
ax.set_title('Pressure Coefficient: Designed vs Target')
ax.legend(); ax.grid(True, alpha=0.3)
ax.invert_yaxis()  # Cp is negative on upper surface

# Shape error
ax = axes[1, 0]
shape_err = np.abs(designed_shape_dn - target_shape_dn)
ax.fill_between(x, 0, shape_err, alpha=0.5, color='red')
ax.set_xlabel('x (chord)')
ax.set_ylabel('|Designed - Target|')
ax.set_title('Shape Error Distribution')
ax.grid(True, alpha=0.3)

# Cp error
ax = axes[1, 1]
cp_err = np.abs(designed_cp_dn - target_cp_dn)
ax.fill_between(x, 0, cp_err, alpha=0.5, color='orange')
ax.set_xlabel('x (chord)')
ax.set_ylabel('|Designed Cp - Target Cp|')
ax.set_title('Cp Error Distribution')
ax.grid(True, alpha=0.3)

plt.suptitle('Differentiable Design Optimization: Inverse Airfoil Design', fontsize=14)
plt.tight_layout()
result_path = os.path.join(RESULTS_DIR, "design_result.png")
plt.savefig(result_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {result_path}")

# --- Figure 4: Multiple design targets ---
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
for tc in range(3):
    tgt_cp = test_c_t[tc:tc+1]
    tgt_shape = test_s_t[tc:tc+1]

    # Re-run design optimization
    d_shape = train_s_t[(tc+10):(tc+11)].clone().detach().requires_grad_(True)
    d_opt = torch.optim.Adam([d_shape], lr=DESIGN_LR)
    for step in range(300):
        d_opt.zero_grad()
        pred = surrogate(d_shape)
        loss = F.mse_loss(pred, tgt_cp) + 0.01 * torch.mean(torch.abs(torch.diff(d_shape, dim=-1)))
        loss.backward()
        d_opt.step()
        with torch.no_grad():
            d_shape.clamp_(min=train_s_norm.min(), max=train_s_norm.max())

    with torch.no_grad():
        d_shape_dn = d_shape.squeeze().cpu().numpy() * s_std + s_mean
        tgt_shape_dn = tgt_shape.squeeze().cpu().numpy() * s_std + s_mean
        d_cp_dn = surrogate(d_shape).squeeze().cpu().numpy() * c_std + c_mean
        tgt_cp_dn = tgt_cp.squeeze().cpu().numpy() * c_std + c_mean

    ax = axes[tc, 0]
    ax.plot(x, d_shape_dn, 'r-', linewidth=2, label='Designed')
    ax.plot(x, tgt_shape_dn, 'g--', linewidth=2, label='Target')
    ax.set_title(f'Case {tc}: Shape')
    ax.set_ylabel('y')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[tc, 1]
    ax.plot(x, d_cp_dn, 'r-', linewidth=2, label='Designed')
    ax.plot(x, tgt_cp_dn, 'g--', linewidth=2, label='Target')
    ax.set_title(f'Case {tc}: Cp')
    ax.invert_yaxis()
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[tc, 2]
    err = np.abs(d_shape_dn - tgt_shape_dn)
    ax.fill_between(x, 0, err, alpha=0.5, color='red')
    ax.set_title(f'Case {tc}: Shape Error')
    ax.grid(True, alpha=0.3)

plt.suptitle('Differentiable Design: 3 Inverse Design Cases', fontsize=14)
plt.tight_layout()
multi_path = os.path.join(RESULTS_DIR, "design_multitarget.png")
plt.savefig(multi_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {multi_path}")

# ============================================================
# [7] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Differentiable Design Optimization")
print("=" * 70)
print(f"  Problem:          Airfoil shape → Cp (forward), Cp → shape (inverse)")
print(f"  Surface points:   {N_POINTS}")
print(f"  Train samples:    {N_TRAIN}")
print(f"  Test samples:     {N_TEST}")
print(f"  Surrogate params: {n_params:,}")
print(f"  Surrogate epochs: {EPOCHS}")
print(f"  Surrogate time:   {total_train_time:.1f}s")
print(f"  Design steps:     {DESIGN_STEPS}")
print(f"  Design time:      {design_time:.1f}s")
print(f"  Cp L2 error:      {cp_l2:.4f} ({cp_l2*100:.1f}%)")
print(f"  Shape L2 error:   {shape_l2:.4f} ({shape_l2*100:.1f}%)")
print(f"  Results:          {RESULTS_DIR}")
print()
print("Key observations:")
print("  1. INVERSE DESIGN: Given target Cp → optimize shape (not forward prediction)")
print("  2. INPUT GRADIENT: Backprop w.r.t. INPUT (shape), not weights")
print("  3. SURROGATE-BASED: Replace expensive CFD with fast CNN for optimization")
print("  4. DIFFERENTIABLE: Entire pipeline is end-to-end differentiable")
print("  5. MULTI-OBJECTIVE: Cp matching + smoothness + edge constraints")
print("  6. NEW PARADIGM: First design optimization tutorial (all others are forward)")
print("=" * 70)
