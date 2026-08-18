"""
PhysicsNeMo DeepONet Tutorial: Burgers Equation
=================================================
Deep Operator Network (DeepONet) for 1D Viscous Burgers Equation:
    u_t + u * u_x = (nu / pi) * u_xx

Unlike PINN (equation-based) or FNO (fixed-grid data-driven), DeepONet
learns a mapping between FUNCTION SPACES: given an initial condition
function u0(x) (evaluated at m sensor points), predict the solution
u(x, t) at any query point (x, t).

Architecture:
    - Branch Net: encodes the input function u0(x) at m sensors -> latent vector
    - Trunk Net:  encodes the query coordinates (x, t) -> latent vector
    - Output:     dot product of branch and trunk outputs

This is fundamentally different from FNO (which maps a fixed-grid field
to another fixed-grid field). DeepONet can handle:
    - Different initial conditions (parametric PDE family)
    - Irregular sensor locations
    - Continuous query points (evaluate at any (x, t))

Key concepts:
    - Operator learning (function-to-function mapping)
    - Branch-Trunk decomposition (based on Universal Approximation Theorem
      for operators, Chen & Chen 1995; Lu et al. 2021)
    - Parametric PDE: multiple ICs -> one learned operator

Author: PhysicsNeMo Tutorial
Date: 2026-07-31
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo DeepONet Tutorial: Burgers Equation")
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
# [1] Problem Setup
# ============================================================
NU = 0.01 / np.pi  # Viscosity parameter
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# Sensor points for branch net (where u0 is evaluated)
M_SENSORS = 100
sensor_x = np.linspace(X_MIN, X_MAX, M_SENSORS)

# ============================================================
# [2] Data Generation: Multiple Initial Conditions
# ============================================================
def generate_ic(n_samples, n_grid):
    """
    Generate diverse initial conditions for Burgers equation.
    Each IC is a smooth random function (sum of sinusoids).
    """
    x = np.linspace(X_MIN, X_MAX, n_grid)
    ics = []
    for _ in range(n_samples):
        # Random sum of sinusoids with varying frequencies and amplitudes
        n_modes = np.random.randint(1, 5)
        u0 = np.zeros(n_grid)
        for _ in range(n_modes):
            freq = np.random.choice([1, 2, 3, 4])
            amp = np.random.uniform(0.2, 0.8) * np.random.choice([-1, 1])
            phase = np.random.uniform(0, 2 * np.pi)
            u0 += amp * np.sin(freq * np.pi * x + phase)
        ics.append(u0)
    return np.array(ics, dtype=np.float32)


def solve_burgers_fd(u0, n_grid, n_times, nu):
    """
    Solve Burgers equation using finite differences (for training data).
    u_t + u * u_x = nu * u_xx
    Periodic boundary conditions.
    """
    dx = (X_MAX - X_MIN) / (n_grid - 1)
    dt_target = (T_MAX - T_MIN) / n_times
    # CFL stability: sub-stepping
    u_max = max(np.max(np.abs(u0)), 0.5)
    dt_stable = 0.4 * dx / (u_max + 1e-8)
    n_sub = max(1, int(np.ceil(dt_target / dt_stable)))
    dt = dt_target / n_sub

    u = u0.copy()
    trajectory = [u.copy()]
    for _ in range(n_times):
        for _ in range(n_sub):
            u_x = np.zeros_like(u)
            u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            u_x[0] = (u[1] - u[-1]) / (2 * dx)
            u_x[-1] = (u[0] - u[-2]) / (2 * dx)

            u_xx = np.zeros_like(u)
            u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
            u_xx[0] = (u[1] - 2 * u[0] + u[-1]) / dx**2
            u_xx[-1] = (u[0] - 2 * u[-1] + u[-2]) / dx**2

            u = u + dt * (-u * u_x + nu * u_xx)
            u = np.clip(u, -5.0, 5.0)
        trajectory.append(u.copy())

    return np.array(trajectory, dtype=np.float32)


print("\n[1] Generating training data (multiple ICs)...")
N_GRID = 128
N_TIMES = 20
N_TRAIN_ICS = 80
N_TEST_ICS = 10

# Generate initial conditions
train_ics = generate_ic(N_TRAIN_ICS, N_GRID)
test_ics = generate_ic(N_TEST_ICS, N_GRID)

# Solve Burgers for each IC
print(f"  Solving {N_TRAIN_ICS} training ICs...")
train_solutions = np.array([solve_burgers_fd(ic, N_GRID, N_TIMES, NU) for ic in train_ics])
print(f"  Solving {N_TEST_ICS} test ICs...")
test_solutions = np.array([solve_burgers_fd(ic, N_GRID, N_TIMES, NU) for ic in test_ics])

print(f"  Train solutions shape: {train_solutions.shape} [ICs, times, x]")
print(f"  Test solutions shape:  {test_solutions.shape} [ICs, times, x]")

# ============================================================
# [3] Prepare DeepONet Training Data
# ============================================================
# Branch input: u0 evaluated at M_SENSORS points
# Trunk input:  (x, t) query coordinates
# Output:       u(x, t)

# Resample ICs to sensor points (linear interpolation)
x_fine = np.linspace(X_MIN, X_MAX, N_GRID)

def resample_to_sensors(u0_fine):
    """Resample from fine grid to sensor locations."""
    return np.interp(sensor_x, x_fine, u0_fine)

# Build training dataset: (u0_sensors, x_query, t_query) -> u(x_query, t_query)
print("\n[2] Building DeepONet training dataset...")

# Sample query points: for each IC, sample random (x, t) points
N_QUERIES_PER_IC = 200  # query points per IC

train_branch_list = []
train_trunk_list = []
train_output_list = []

for i in range(N_TRAIN_ICS):
    # Sensor values for this IC
    u0_sensors = resample_to_sensors(train_ics[i])  # [M_SENSORS]

    # Random query points in (x, t)
    x_q = np.random.uniform(X_MIN, X_MAX, N_QUERIES_PER_IC)
    t_idx = np.random.randint(0, N_TIMES + 1, N_QUERIES_PER_IC)
    t_q = t_idx / N_TIMES * (T_MAX - T_MIN)

    # Ground truth: interpolate solution at query points
    for j in range(N_QUERIES_PER_IC):
        u_val = np.interp(x_q[j], x_fine, train_solutions[i, t_idx[j]])
        train_branch_list.append(u0_sensors)
        train_trunk_list.append([x_q[j], t_q[j]])
        train_output_list.append(u_val)

train_branch = np.array(train_branch_list, dtype=np.float32)  # [N, M_SENSORS]
train_trunk = np.array(train_trunk_list, dtype=np.float32)    # [N, 2]
train_output = np.array(train_output_list, dtype=np.float32)   # [N]

print(f"  Training pairs: {len(train_output)}")

# Build test dataset similarly
test_branch_list = []
test_trunk_list = []
test_output_list = []

for i in range(N_TEST_ICS):
    u0_sensors = resample_to_sensors(test_ics[i])
    x_q = np.random.uniform(X_MIN, X_MAX, N_QUERIES_PER_IC)
    t_idx = np.random.randint(0, N_TIMES + 1, N_QUERIES_PER_IC)
    t_q = t_idx / N_TIMES * (T_MAX - T_MIN)
    for j in range(N_QUERIES_PER_IC):
        u_val = np.interp(x_q[j], x_fine, test_solutions[i, t_idx[j]])
        test_branch_list.append(u0_sensors)
        test_trunk_list.append([x_q[j], t_q[j]])
        test_output_list.append(u_val)

test_branch = np.array(test_branch_list, dtype=np.float32)
test_trunk = np.array(test_trunk_list, dtype=np.float32)
test_output = np.array(test_output_list, dtype=np.float32)
print(f"  Test pairs: {len(test_output)}")

# Convert to tensors
train_branch_t = torch.from_numpy(train_branch).to(device)
train_trunk_t = torch.from_numpy(train_trunk).to(device)
train_output_t = torch.from_numpy(train_output).reshape(-1, 1).to(device)
test_branch_t = torch.from_numpy(test_branch).to(device)
test_trunk_t = torch.from_numpy(test_trunk).to(device)
test_output_t = torch.from_numpy(test_output).reshape(-1, 1).to(device)

# ============================================================
# [4] DeepONet Model
# ============================================================
class BranchNet(nn.Module):
    """Encodes the input function u0(x) at sensor points."""
    def __init__(self, n_sensors, hidden=64, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class TrunkNet(nn.Module):
    """Encodes query coordinates (x, t)."""
    def __init__(self, input_dim=2, hidden=64, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    """
    Deep Operator Network: branch(x) . trunk(y) + bias.

    The output is the dot product of branch and trunk outputs,
    approximating the operator G(u0)(x, t).
    """
    def __init__(self, n_sensors, branch_hidden=64, trunk_hidden=64, p=64):
        super().__init__()
        self.branch = BranchNet(n_sensors, branch_hidden, p)
        self.trunk = TrunkNet(2, trunk_hidden, p)
        # Optional bias term
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0_sensors, xt_query):
        """
        Args:
            u0_sensors: [batch, n_sensors] -- function values at sensors
            xt_query:   [batch, 2]         -- query coordinates (x, t)
        Returns:
            [batch, 1] -- predicted u(x, t)
        """
        b = self.branch(u0_sensors)   # [batch, p]
        t = self.trunk(xt_query)       # [batch, p]
        # Dot product (element-wise then sum)
        out = torch.sum(b * t, dim=1, keepdim=True) + self.bias  # [batch, 1]
        return out


model = DeepONet(n_sensors=M_SENSORS, branch_hidden=64, trunk_hidden=64, p=64).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nDeepONet parameters: {n_params:,}")
print(f"  Branch: {sum(p.numel() for p in model.branch.parameters()):,}")
print(f"  Trunk:  {sum(p.numel() for p in model.trunk.parameters()):,}")

# ============================================================
# [5] Training
# ============================================================
EPOCHS = 2000
BATCH_SIZE = 512

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

n_train = len(train_output)
train_losses = []
test_losses = []

print(f"\nStarting training ({EPOCHS} epochs, batch={BATCH_SIZE})...")
print("-" * 70)

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    n_batches = 0

    perm = torch.randperm(n_train)
    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        b_batch = train_branch_t[idx]
        t_batch = train_trunk_t[idx]
        y_batch = train_output_t[idx]

        pred = model(b_batch, t_batch)
        loss = F.mse_loss(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    train_loss = epoch_loss / n_batches
    train_losses.append(train_loss)

    # Test loss
    model.eval()
    with torch.no_grad():
        test_pred = model(test_branch_t, test_trunk_t)
        test_loss = F.mse_loss(test_pred, test_output_t).item()
    test_losses.append(test_loss)

    scheduler.step()

    if epoch % 200 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:5d}/{EPOCHS} | "
              f"Train: {train_loss:.6e} | "
              f"Test: {test_loss:.6e} | "
              f"Time: {elapsed:.1f}s")

total_time = time.time() - start_time
print("-" * 70)
print(f"Training complete! Total time: {total_time:.1f}s")
print(f"Final train loss: {train_losses[-1]:.6e}")
print(f"Final test loss:  {test_losses[-1]:.6e}")

# ============================================================
# [6] Evaluation: Full Trajectory Prediction
# ============================================================
print("\n[3] Evaluating on full trajectory prediction...")

model.eval()
# Pick a test IC and predict the full solution at all (x, t) grid points
eval_ic_idx = 0
u0_sensors_eval = torch.from_numpy(
    resample_to_sensors(test_ics[eval_ic_idx])
).float().unsqueeze(0).to(device)  # [1, M_SENSORS]

# Query grid: all (x, t) combinations
x_eval = np.linspace(X_MIN, X_MAX, N_GRID)
t_eval = np.linspace(T_MIN, T_MAX, N_TIMES + 1)
X_eval, T_eval = np.meshgrid(x_eval, t_eval)

# Flatten for batch prediction
xt_query = np.stack([X_eval.flatten(), T_eval.flatten()], axis=1)
xt_query_t = torch.from_numpy(xt_query).float().to(device)

# Repeat sensor values for all query points
u0_repeated = u0_sensors_eval.repeat(xt_query_t.shape[0], 1)

with torch.no_grad():
    u_pred_flat = model(u0_repeated, xt_query_t).cpu().numpy().flatten()

u_pred = u_pred_flat.reshape(N_TIMES + 1, N_GRID)
u_truth = test_solutions[eval_ic_idx]  # [N_TIMES+1, N_GRID]

# Relative L2 error
rel_l2 = np.linalg.norm(u_pred - u_truth) / (np.linalg.norm(u_truth) + 1e-8)
print(f"  Relative L2 error (full trajectory): {rel_l2:.4f}")

# ============================================================
# [7] Visualization
# ============================================================
print("\n[4] Generating visualizations...")

# --- Figure 1: Loss curves ---
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.semilogy(train_losses, label='Train', linewidth=2)
ax.semilogy(test_losses, label='Test', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE, log scale)')
ax.set_title('DeepONet Burgers: Training & Test Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
loss_path = os.path.join(RESULTS_DIR, "deeponet_burgers_loss.png")
plt.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {loss_path}")

# --- Figure 2: Solution comparison (snapshots) ---
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
snapshots = [0, 5, 10, 20]

for idx, snap in enumerate(snapshots):
    # Ground truth
    ax = axes[0, idx]
    ax.plot(x_eval, u_truth[snap], 'b-', linewidth=2, label='Ground Truth')
    ax.plot(x_eval, u_pred[snap], 'r--', linewidth=2, label='DeepONet')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(f't = {snap / N_TIMES:.2f}')
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Error
    ax = axes[1, idx]
    error = np.abs(u_pred[snap] - u_truth[snap])
    ax.plot(x_eval, error, 'k-', linewidth=1.5)
    ax.fill_between(x_eval, 0, error, alpha=0.3, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('|Error|')
    ax.set_title(f'Absolute Error at t={snap / N_TIMES:.2f}')
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f'DeepONet Burgers: Prediction vs Ground Truth (Test IC #{eval_ic_idx}, '
    f'Rel L2={rel_l2:.4f})',
    fontsize=14
)
plt.tight_layout()
result_path = os.path.join(RESULTS_DIR, "deeponet_burgers_result.png")
plt.savefig(result_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {result_path}")

# --- Figure 3: 3D surface comparison ---
fig = plt.figure(figsize=(18, 5))

# Ground truth
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_eval, T_eval, u_truth, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('u')
ax1.set_title('Ground Truth')

# DeepONet prediction
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X_eval, T_eval, u_pred, cmap='viridis', alpha=0.8)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u')
ax2.set_title('DeepONet Prediction')

# Error
ax3 = fig.add_subplot(133, projection='3d')
error_full = np.abs(u_pred - u_truth)
ax3.plot_surface(X_eval, T_eval, error_full, cmap='hot', alpha=0.8)
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_zlabel('|Error|')
ax3.set_title('Absolute Error')

plt.suptitle('DeepONet Burgers: 3D Solution Comparison', fontsize=14)
plt.tight_layout()
surface_path = os.path.join(RESULTS_DIR, "deeponet_burgers_surface.png")
plt.savefig(surface_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {surface_path}")

# --- Figure 4: Multiple ICs generalization ---
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
t_show = 10  # t = 0.5

for idx in range(5):
    ic_idx = idx
    u0_s = torch.from_numpy(
        resample_to_sensors(test_ics[ic_idx])
    ).float().unsqueeze(0).to(device)

    x_q = np.linspace(X_MIN, X_MAX, N_GRID)
    t_q = np.full(N_GRID, t_show / N_TIMES)
    xt_q = np.stack([x_q, t_q], axis=1)
    xt_q_t = torch.from_numpy(xt_q).float().to(device)
    u0_rep = u0_s.repeat(N_GRID, 1)

    with torch.no_grad():
        u_p = model(u0_rep, xt_q_t).cpu().numpy().flatten()

    # Ground truth
    u_t = test_solutions[ic_idx, t_show]

    ax = axes[0, idx]
    ax.plot(x_q, u_t, 'b-', linewidth=2, label='Truth')
    ax.plot(x_q, u_p, 'r--', linewidth=2, label='DeepONet')
    ax.set_title(f'Test IC #{ic_idx} (t=0.5)')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9)

    # IC
    ax = axes[1, idx]
    ax.plot(x_fine, test_ics[ic_idx], 'g-', linewidth=2)
    ax.set_title(f'Initial Condition #{ic_idx}')
    ax.set_xlabel('x')
    ax.set_ylabel('u0(x)')
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)

plt.suptitle('DeepONet Generalization: 5 Different Initial Conditions', fontsize=14)
plt.tight_layout()
gen_path = os.path.join(RESULTS_DIR, "deeponet_burgers_generalization.png")
plt.savefig(gen_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {gen_path}")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: DeepONet Burgers Equation")
print("=" * 70)
print(f"  Equation:       u_t + u * u_x = (nu/pi) * u_xx")
print(f"  Viscosity:      nu/pi = {NU:.6f}")
print(f"  Training ICs:   {N_TRAIN_ICS} (each with {N_QUERIES_PER_IC} query points)")
print(f"  Test ICs:       {N_TEST_ICS}")
print(f"  Sensors:        {M_SENSORS} points (branch input)")
print(f"  Branch net:     {M_SENSORS} -> 64 -> 64 -> 64")
print(f"  Trunk net:      2 -> 64 -> 64 -> 64")
print(f"  Parameters:     {n_params:,}")
print(f"  Epochs:         {EPOCHS}")
print(f"  Training time:  {total_time:.1f}s")
print(f"  Final train loss: {train_losses[-1]:.6e}")
print(f"  Final test loss:  {test_losses[-1]:.6e}")
print(f"  Rel L2 (full traj): {rel_l2:.4f}")
print(f"  Results:        {RESULTS_DIR}")
print()
print("Key observations:")
print("  1. DeepONet learns the OPERATOR u0(x) -> u(x,t), not a single solution")
print("  2. One trained model generalizes to UNSEEN initial conditions")
print("  3. Branch net encodes IC at sensors; Trunk net encodes query (x,t)")
print("  4. Unlike FNO (fixed grid), DeepONet can predict at ANY continuous point")
print("  5. Unlike PINN (equation-based), DeepONet needs training data but is faster")
print("=" * 70)
