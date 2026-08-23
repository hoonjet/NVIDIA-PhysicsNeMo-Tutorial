"""
PhysicsNeMo GNN Tutorial: Time Evolution Rollout (Spring-Mass System)
======================================================================
This tutorial extends the existing MeshGraphNet tutorial from 1-step
prediction to MULTI-STEP AUTO-REGRESSIVE ROLLOUT.

Existing MeshGraphNet tutorial:
  - Input:  (position, velocity) at time t
  - Output: acceleration at time t
  - Single-step prediction only

THIS tutorial:
  - Input:  (position, velocity) at time t
  - Output: (position, velocity) at time t+1 (integrated)
  - Multi-step rollout: feed prediction back, predict T steps forward
  - Error accumulation analysis (key challenge of auto-regressive prediction)

Key concepts:
  1. Auto-regressive rollout: predict t+1 from t, then t+2 from predicted t+1, etc.
  2. Error accumulation: each step's error feeds into next step's input
  3. Teacher-forcing vs rollout: one-step error (low) vs rollout error (grows)
  4. Stability: long rollout may diverge (energy drift)

This is the ONLY tutorial that:
  - Performs multi-step GNN rollout on a mesh graph
  - Analyzes error accumulation in GNN time evolution
  - Compares one-step vs multi-step prediction error
  - Visualizes trajectory divergence over time

Author: PhysicsNeMo Tutorial
Date: 2026-08-20
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
from matplotlib.animation import FuncAnimation

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo GNN Tutorial: Time Evolution Rollout (Spring-Mass)")
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
# [1] Spring-Mass System Setup
# ============================================================
# Same system as existing MeshGraphNet tutorial, but we generate
# FULL TRAJECTORIES (time series) instead of single snapshots.

N_GRID_X = 5
N_GRID_Y = 4
N_NODES = N_GRID_X * N_GRID_Y  # 20 nodes
K_SPRING = 50.0
MASS = 1.0
REST_LENGTH = 1.0
DAMPING = 0.1
DT = 0.01           # Time step for simulation
N_STEPS = 50         # Steps per trajectory
N_TRAJECTORIES_TRAIN = 150
N_TRAJECTORIES_TEST = 20

print(f"\n[1] Spring-mass system setup")
print(f"  Grid: {N_GRID_X}x{N_GRID_Y} = {N_NODES} nodes")
print(f"  Spring constant: {K_SPRING}")
print(f"  Damping: {DAMPING}")
print(f"  Time step: {DT}, Steps per trajectory: {N_STEPS}")

def build_grid_graph(nx, ny):
    """Build bidirectional grid graph."""
    edges_src, edges_dst = [], []
    for i in range(nx):
        for j in range(ny):
            node_id = i * ny + j
            if i < nx - 1:
                neighbor = (i + 1) * ny + j
                edges_src += [node_id, neighbor]; edges_dst += [neighbor, node_id]
            if j < ny - 1:
                neighbor = i * ny + (j + 1)
                edges_src += [node_id, neighbor]; edges_dst += [neighbor, node_id]
            if i < nx - 1 and j < ny - 1:
                neighbor = (i + 1) * ny + (j + 1)
                edges_src += [node_id, neighbor]; edges_dst += [neighbor, node_id]
            if i < nx - 1 and j > 0:
                neighbor = (i + 1) * ny + (j - 1)
                edges_src += [node_id, neighbor]; edges_dst += [neighbor, node_id]
    return torch.tensor([edges_src, edges_dst], dtype=torch.long)

edge_index = build_grid_graph(N_GRID_X, N_GRID_Y)
N_EDGES = edge_index.shape[1]
print(f"  Edges: {N_EDGES} (bidirectional)")

def simulate_trajectory(n_steps, dt, nx, ny, k, mass, rest_len, damping):
    """Simulate one trajectory of the spring-mass system."""
    n_nodes = nx * ny
    # Random initial conditions
    xs = np.arange(nx, dtype=np.float64) * rest_len
    ys = np.arange(ny, dtype=np.float64) * rest_len
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    positions = np.stack([xx.flatten(), yy.flatten()], axis=1)
    positions += np.random.randn(n_nodes, 2) * 0.15
    velocities = np.random.randn(n_nodes, 2) * 0.5

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    trajectory = np.zeros((n_steps + 1, n_nodes, 4), dtype=np.float32)  # [x, y, vx, vy]
    trajectory[0, :, :2] = positions
    trajectory[0, :, 2:] = velocities

    for step in range(n_steps):
        # Compute forces
        forces = np.zeros_like(positions)
        for e in range(len(src)):
            s, d = src[e], dst[e]
            d_vec = positions[d] - positions[s]
            d_mag = np.linalg.norm(d_vec)
            if d_mag > 1e-8:
                f_mag = -k * (d_mag - rest_len)
                f_vec = f_mag * (d_vec / d_mag)
                forces[s] += f_vec

        forces -= damping * velocities
        accelerations = forces / mass

        # Semi-implicit Euler
        velocities += accelerations * dt
        positions += velocities * dt

        trajectory[step + 1, :, :2] = positions
        trajectory[step + 1, :, 2:] = velocities

    return trajectory

print(f"\n  Generating {N_TRAJECTORIES_TRAIN} train + {N_TRAJECTORIES_TEST} test trajectories...")
train_trajs = np.array([simulate_trajectory(N_STEPS, DT, N_GRID_X, N_GRID_Y, K_SPRING, MASS, REST_LENGTH, DAMPING)
                        for _ in range(N_TRAJECTORIES_TRAIN)])
test_trajs = np.array([simulate_trajectory(N_STEPS, DT, N_GRID_X, N_GRID_Y, K_SPRING, MASS, REST_LENGTH, DAMPING)
                       for _ in range(N_TRAJECTORIES_TEST)])
print(f"  Train shape: {train_trajs.shape} [trajectories, steps, nodes, 4]")
print(f"  Test shape:  {test_trajs.shape}")

# Normalize
traj_mean = train_trajs.mean()
traj_std = train_trajs.std()
train_trajs_n = (train_trajs - traj_mean) / (traj_std + 1e-8)
test_trajs_n = (test_trajs - traj_mean) / (traj_std + 1e-8)

train_t = torch.from_numpy(train_trajs_n).to(device)
test_t = torch.from_numpy(test_trajs_n).to(device)
edge_index = edge_index.to(device)

# ============================================================
# [2] GNN Model
# ============================================================
class GNNRollout(nn.Module):
    """
    GNN for time evolution: predicts next state from current state.
    Input:  node features [x, y, vx, vy] (4)
    Output: delta state [dx, dy, dvx, dvy] (4) — change in state
    """
    def __init__(self, node_in=4, edge_in=3, hidden=64, n_processor=6):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(node_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.edge_enc = nn.Sequential(nn.Linear(edge_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.edge_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden*3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_processor)])
        self.node_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_processor)])
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, node_in))

    def forward(self, node_feat, edge_feat, edge_idx):
        h_n = self.node_enc(node_feat)
        h_e = self.edge_enc(edge_feat)
        src, dst = edge_idx[0], edge_idx[1]
        for eb, nb in zip(self.edge_blocks, self.node_blocks):
            msg = torch.cat([h_e, h_n[src], h_n[dst]], dim=-1)
            h_e = h_e + eb(msg)
            agg = torch.zeros_like(h_n)
            agg.index_add_(0, dst, h_e)
            h_n = h_n + nb(torch.cat([h_n, agg], dim=-1))
        return self.decoder(h_n)

# Precompute edge features (same for all steps/trajectories)
src_np = edge_index[0].cpu().numpy()
dst_np = edge_index[1].cpu().numpy()
# Use mean positions for edge features (approximate)
mean_pos = train_trajs[:, 0, :, :2].mean(axis=0)  # [N, 2]
rel_pos = mean_pos[dst_np] - mean_pos[src_np]
dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)
edge_feat_base = np.concatenate([rel_pos, dist], axis=1).astype(np.float32)
edge_feat_base_t = torch.from_numpy(edge_feat_base).to(device)

print(f"\n[2] Building GNN model...")
model = GNNRollout(node_in=4, edge_in=3, hidden=64, n_processor=6).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: GNNRollout (6 processor blocks, 64 hidden)")
print(f"  Parameters: {n_params:,}")
print(f"  Input: [x, y, vx, vy] (4)")
print(f"  Output: [dx, dy, dvx, dvy] (delta state)")

# ============================================================
# [3] Training: One-Step Prediction
# ============================================================
EPOCHS = 400
BATCH_SIZE = 30
LR = 1e-3

print(f"\n[3] Training GNN for one-step prediction ({EPOCHS} epochs)")
print(f"    Task: state(t) → state(t+1)")
print("-" * 70)

opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.MSELoss()

train_losses, test_losses = [], []
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(N_TRAJECTORIES_TRAIN)
    epoch_loss = 0; n_batches = 0

    for i in range(0, N_TRAJECTORIES_TRAIN, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        bs = len(idx)

        # All consecutive pairs: state(t) → state(t+1)
        states_curr = train_t[idx, :-1]   # [bs, T-1, N, 4]
        states_next = train_t[idx, 1:]    # [bs, T-1, N, 4]
        B, T = states_curr.shape[0], states_curr.shape[1]

        # Flatten for batch processing
        node_in = states_curr.reshape(B * T, N_NODES, 4)
        # Repeat edge features for batch
        edge_in = edge_feat_base_t.unsqueeze(0).expand(B * T, -1, -1).reshape(-1, 3)
        # Batched edge_index
        ei_batch = torch.cat([edge_index + b * N_NODES for b in range(B * T)], dim=1)

        pred = model(node_in.reshape(-1, 4), edge_in, ei_batch)
        pred = pred.reshape(B, T, N_NODES, 4)
        # Predict delta: state(t+1) = state(t) + delta
        pred_state = states_curr + pred
        loss = loss_fn(pred_state, states_next)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_batches += 1

    train_losses.append(epoch_loss / n_batches)
    sched.step()

    # Test one-step
    model.eval()
    with torch.no_grad():
        sc = test_t[:, :-1]; sn = test_t[:, 1:]
        B, T = sc.shape[0], sc.shape[1]
        ni = sc.reshape(B * T, N_NODES, 4)
        ei = edge_feat_base_t.unsqueeze(0).expand(B * T, -1, -1).reshape(-1, 3)
        eib = torch.cat([edge_index + b * N_NODES for b in range(B * T)], dim=1)
        pred = model(ni.reshape(-1, 4), ei, eib).reshape(B, T, N_NODES, 4)
        test_loss = loss_fn(sc + pred, sn).item()
    test_losses.append(test_loss)

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:4d} | Train: {train_losses[-1]:.6e} | Test: {test_loss:.6e} | Time: {time.time()-start:.1f}s")

train_time = time.time() - start
print("-" * 70)

# ============================================================
# [4] Multi-Step Rollout Evaluation
# ============================================================
print(f"\n[4] Multi-step rollout evaluation")
print(f"    Auto-regressive: predict t+1 from t, feed back, repeat")
print("-" * 70)

model.eval()
n_steps = test_t.shape[1]

with torch.no_grad():
    # Rollout: start from true state(0), predict all steps
    rollout_pred = torch.zeros_like(test_t)
    rollout_pred[:, 0] = test_t[:, 0]  # True initial state

    for step in range(1, n_steps):
        curr = rollout_pred[:, step-1]  # [B, N, 4]
        B = curr.shape[0]
        ei = edge_feat_base_t.unsqueeze(0).expand(B, -1, -1).reshape(-1, 3)
        eib = torch.cat([edge_index + b * N_NODES for b in range(B)], dim=1)
        delta = model(curr.reshape(-1, 4), ei, eib).reshape(B, N_NODES, 4)
        rollout_pred[:, step] = curr + delta

    # Compute errors
    rollout_errors = []
    one_step_errors = []
    for step in range(n_steps):
        # Rollout error
        err_r = torch.norm(rollout_pred[:, step] - test_t[:, step], dim=(1, 2)) / \
                (torch.norm(test_t[:, step], dim=(1, 2)) + 1e-8)
        rollout_errors.append(err_r.mean().item())

        # One-step (teacher-forced) error
        if step < n_steps - 1:
            curr = test_t[:, step]
            B = curr.shape[0]
            ei = edge_feat_base_t.unsqueeze(0).expand(B, -1, -1).reshape(-1, 3)
            eib = torch.cat([edge_index + b * N_NODES for b in range(B)], dim=1)
            delta = model(curr.reshape(-1, 4), ei, eib).reshape(B, N_NODES, 4)
            pred_next = curr + delta
            err1 = torch.norm(pred_next - test_t[:, step+1], dim=(1, 2)) / \
                   (torch.norm(test_t[:, step+1], dim=(1, 2)) + 1e-8)
            one_step_errors.append(err1.mean().item())

print(f"  One-step (teacher-forced) avg error: {np.mean(one_step_errors):.4f}")
print(f"  Rollout step  5:  {rollout_errors[5]:.4f}")
print(f"  Rollout step 10:  {rollout_errors[10]:.4f}")
print(f"  Rollout step 20:  {rollout_errors[20]:.4f}")
print(f"  Rollout step 30:  {rollout_errors[30]:.4f}")
print(f"  Rollout step 50:  {rollout_errors[50]:.4f}")

# ============================================================
# [5] Visualization
# ============================================================
print(f"\n[5] Generating visualizations...")

# --- Figure 1: Training loss ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(train_losses, linewidth=1.5, color='blue')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss (MSE)')
ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)
ax2.semilogy(test_losses, linewidth=1.5, color='red')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Test Loss (MSE)')
ax2.set_title('Test Loss (one-step)'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_rollout_loss.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Trajectory comparison (node positions over time) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sample_idx = 0
steps_show = [0, 10, 20, 30, 40, 50]

for i, step in enumerate(steps_show):
    ax = axes[i // 3, i % 3]
    # True positions
    true_pos = test_trajs[sample_idx, step, :, :2]
    # Predicted positions (denormalized)
    pred_pos = rollout_pred[sample_idx, step, :, :2].cpu().numpy() * traj_std + traj_mean

    # Draw edges
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    for e in range(len(src)):
        ax.plot([true_pos[src[e], 0], true_pos[dst[e], 0]],
                [true_pos[src[e], 1], true_pos[dst[e], 1]], 'b-', alpha=0.3, linewidth=0.5)
        ax.plot([pred_pos[src[e], 0], pred_pos[dst[e], 0]],
                [pred_pos[src[e], 1], pred_pos[dst[e], 1]], 'r--', alpha=0.3, linewidth=0.5)

    ax.scatter(true_pos[:, 0], true_pos[:, 1], c='blue', s=50, zorder=5, label='True')
    ax.scatter(pred_pos[:, 0], pred_pos[:, 1], c='red', s=30, marker='x', zorder=5, label='GNN Rollout')
    ax.set_title(f'Step {step}\nRollout L2: {rollout_errors[step]*100:.1f}%')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.set_xlim(-1, N_GRID_X)
    ax.set_ylim(-1, N_GRID_Y)

plt.suptitle('GNN Rollout: True (blue) vs Predicted (red) Positions', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_rollout_trajectory.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: Error accumulation ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(range(n_steps), rollout_errors, 'o-', linewidth=2, color='red', label='Rollout (auto-regressive)', markersize=4)
ax.axhline(y=np.mean(one_step_errors), color='blue', linestyle='--', linewidth=2,
           label=f'One-step avg ({np.mean(one_step_errors):.4f})')
ax.set_xlabel('Rollout Step'); ax.set_ylabel('Relative L2 Error')
ax.set_title('Error Accumulation in GNN Auto-Regressive Rollout')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_rollout_error.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: Per-node trajectory comparison ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sample_idx = 0
node_indices = [0, 5, 10, 19]  # 4 nodes to track

for idx, node_id in enumerate(node_indices):
    ax = axes[idx // 2, idx % 2]
    true_x = test_trajs[sample_idx, :, node_id, 0]
    true_y = test_trajs[sample_idx, :, node_id, 1]
    pred_x = rollout_pred[sample_idx, :, node_id, 0].cpu().numpy() * traj_std + traj_mean
    pred_y = rollout_pred[sample_idx, :, node_id, 1].cpu().numpy() * traj_std + traj_mean

    ax.plot(true_x, true_y, 'b-', linewidth=2, label='True trajectory')
    ax.plot(pred_x, pred_y, 'r--', linewidth=1.5, label='GNN rollout')
    ax.plot(true_x[0], true_y[0], 'go', markersize=8, label='Start')
    ax.set_title(f'Node {node_id} Trajectory (x-y plane)')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.suptitle('GNN Rollout: Per-Node Trajectory Comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_rollout_nodes.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 5: Explanation ---
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.text(0.5, 0.92, 'GNN Time Evolution Rollout: Spring-Mass System',
        ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.75,
    'Auto-regressive rollout:\n'
    '  state(0) → GNN → state(1) → GNN → state(2) → ...\n'
    '  Each prediction feeds back as next input\n'
    '  Error accumulates with each step',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.75,
    'vs. existing MeshGraphNet tutorial:\n'
    '  MGN: 1-step (pos,vel → accel)\n'
    '  This: N-step (state(t) → state(t+1))\n'
    '  MGN: no rollout analysis\n'
    '  This: error accumulation analysis',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.05, 0.45,
    'Key challenge:\n'
    '  • One-step error: small (teacher-forced)\n'
    '  • Rollout error: grows with steps\n'
    '  • Compounding: each error feeds forward\n'
    '  • Long rollout may diverge (energy drift)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.45,
    'Mitigation strategies:\n'
    '  • Pushforward trick: train with multi-step loss\n'
    '  • Noise injection: add noise to input during training\n'
    '  • Energy conservation: add physics constraint\n'
    '  • Longer training: more epochs for stability',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.05, 0.15,
    'Connection to other tutorials:\n'
    '  • Allen-Cahn: same rollout concept (FNO)\n'
    '  • Wave: same rollout concept (FNO, 2-channel)\n'
    '  • This: rollout on GRAPH (not grid)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.15,
    'Engineering relevance:\n'
    '  • Structural dynamics: earthquake response\n'
    '  • CFD: unsteady flow prediction\n'
    '  • Vibration analysis: modal prediction\n'
    '  • Real-time simulation: fast surrogate',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_rollout_explanation.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [6] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: GNN — Time Evolution Rollout (Spring-Mass)")
print("=" * 70)
print(f"  System:             Spring-mass ({N_NODES} nodes, {N_EDGES} edges)")
print(f"  Trajectory length:  {N_STEPS} steps (dt={DT})")
print(f"  Train trajectories: {N_TRAJECTORIES_TRAIN}")
print(f"  Test trajectories:  {N_TRAJECTORIES_TEST}")
print(f"  Epochs:             {EPOCHS}")
print(f"  Training time:      {train_time:.1f}s")
print(f"  --- Results ---")
print(f"  One-step error:     {np.mean(one_step_errors):.4f} (avg)")
print(f"  Rollout step 10:    {rollout_errors[10]:.4f}")
print(f"  Rollout step 20:    {rollout_errors[20]:.4f}")
print(f"  Rollout step 50:    {rollout_errors[50]:.4f}")
print()
print("Key observations:")
print("  1. AUTO-REGRESSIVE: state(t) → GNN → state(t+1), feed back, repeat")
print("  2. ERROR ACCUMULATION: rollout error grows with steps (compounding)")
print("  3. ONE-STEP vs ROLLOUT: one-step is stable, rollout diverges over time")
print("  4. TRAJECTORY: GNN captures short-term dynamics, drifts at long times")
print("  5. vs MGN TUTORIAL: MGN does 1-step only; this does N-step rollout")
print("  6. GRAPH ROLLOUT: same concept as Allen-Cahn/Wave, but on graph (not grid)")
print("=" * 70)
