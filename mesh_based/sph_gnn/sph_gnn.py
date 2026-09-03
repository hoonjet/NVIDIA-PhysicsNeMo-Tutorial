"""
PhysicsNeMo GNN Tutorial: SPH (Smoothed Particle Hydrodynamics) + GNN
=====================================================================
2D Dam-Break Simulation with Lagrangian Particle Learning

Existing GNN tutorials:
  - GNN Beam: Fixed mesh, structural analysis (load → displacement)
  - GNN Rollout: Fixed mesh, time evolution (spring-mass system)
  - MeshGraphNet: Fixed mesh, 1-step prediction

THIS tutorial:
  - SPH particles: Nodes MOVE (Lagrangian framework)
  - Dynamic graph: Edges rebuilt every step (neighbors change)
  - Free surface: Particles can separate (no fixed topology)
  - Dam-break: Water column collapses under gravity

Key difference from existing GNN tutorials:
  ┌──────────────────────┬──────────────────────┐
  │ Existing GNN          │ THIS (SPH GNN)       │
  ├──────────────────────┼──────────────────────┤
  │ Fixed mesh (Eulerian) │ Moving particles     │
  │ Static graph          │ Dynamic graph        │
  │ Fixed topology        │ Free surface         │
  │ Structural / spring   │ Fluid (Navier-Stokes)│
  │ No particle transport │ Mass transport       │
  └──────────────────────┴──────────────────────┘

SPH Physics:
  - Each particle has: position (x,y), velocity (vx,vy), density (ρ)
  - Kernel function W(r, h): smoothing kernel with support radius h
  - Density: ρ_i = Σ_j m_j W(|x_i - x_j|, h)
  - Pressure: p = c²(ρ - ρ₀)  (equation of state)
  - Acceleration: a = -∇p/ρ + g

GNN learns: (position, velocity) → acceleration
  - Input:  node features (pos, vel) + edge features (relative pos)
  - Output:  acceleration (dv/dt)
  - Integrate: v_{t+1} = v_t + a·dt, x_{t+1} = x_t + v·dt

Author: PhysicsNeMo Tutorial
Date: 2026-09-01
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
from matplotlib.patches import Rectangle

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo GNN Tutorial: SPH (Dam-Break) — Lagrangian")
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
# [1] SPH Simulation Parameters
# ============================================================
# Domain
DOMAIN_X = 2.0       # Width
DOMAIN_Y = 1.0       # Height

# SPH parameters
H_SMOOTH = 0.08      # Smoothing length (kernel support radius)
PARTICLE_MASS = 1.0   # Particle mass
REST_DENSITY = 1000.0 # Reference density (ρ₀)
C_SOUND = 15.0        # Speed of sound (artificial, for incompressible limit)
GRAVITY = -9.81       # Gravity (y-direction)
DT = 0.0005           # Time step
NU_SPH = 0.5          # Artificial viscosity coefficient

# Dam-break initial condition
DAM_WIDTH = 0.5       # Initial water column width
DAM_HEIGHT = 0.8      # Initial water column height
PARTICLE_SPACING = 0.04

# Training
N_STEPS_TRAJ = 100    # Steps per trajectory
N_TRAJ_TRAIN = 80
N_TRAJ_TEST = 20

print(f"\n[1] SPH Parameters:")
print(f"  Domain: {DOMAIN_X} × {DOMAIN_Y}")
print(f"  Smoothing length h = {H_SMOOTH}")
print(f"  Sound speed c = {C_SOUND}")
print(f"  Gravity g = {GRAVITY}")
print(f"  dt = {DT}")
print(f"  Trajectory length: {N_STEPS_TRAJ} steps")
print(f"  Train trajectories: {N_TRAJ_TRAIN}, Test: {N_TRAJ_TEST}")

# ============================================================
# [2] SPH Kernel Functions
# ============================================================
def W_poly6(r, h):
    """Poly6 kernel: W(r, h) = (315/(64π h^9)) (h²-r²)³ for r ≤ h"""
    coeff = 315.0 / (64.0 * np.pi * h**9)
    q = h*h - r*r
    q = np.maximum(q, 0.0)
    return coeff * q**3

def gradW_spiky(dx, dy, r, h):
    """Spiky kernel gradient: ∇W = (-45/(π h^6)) (h-r)² * (dx/r, dy/r)"""
    coeff = -45.0 / (np.pi * h**6)
    q = np.maximum(h - r, 0.0)
    factor = coeff * q**2 / (r + 1e-10)
    return factor * dx, factor * dy

def W_viscosity(r, h):
    """Viscosity kernel: W(r,h) = (15/(2π h^3))(-r³/(2h³) + r²/h² + h/(2r) - 1)"""
    coeff = 15.0 / (2.0 * np.pi * h**3)
    q = r / h
    result = np.zeros_like(r)
    mask = (r > 1e-10) & (r <= h)
    result[mask] = coeff * (-q[mask]**3 / 2 + q[mask]**2 + 1/(2*q[mask]) - 1)
    return result

print(f"\n[2] SPH Kernels: Poly6 (density), Spiky (pressure gradient), Viscosity")

# ============================================================
# [3] SPH Solver (Reference / Data Generator)
# ============================================================
def init_dam_break():
    """Initialize dam-break: water column on the left."""
    particles = []
    x = PARTICLE_SPACING / 2
    while x < DAM_WIDTH:
        y = PARTICLE_SPACING / 2
        while y < DAM_HEIGHT:
            # Add small random perturbation
            px = x + np.random.uniform(-0.001, 0.001)
            py = y + np.random.uniform(-0.001, 0.001)
            particles.append([px, py, 0.0, 0.0])  # x, y, vx, vy
            y += PARTICLE_SPACING
        x += PARTICLE_SPACING
    return np.array(particles, dtype=np.float32)  # [N, 4]

def find_neighbors(positions, h):
    """Find all particle pairs within distance h."""
    n = len(positions)
    # Brute-force neighbor search (fine for tutorial-scale)
    diff = positions[:, None, :] - positions[None, :, :]  # [N, N, 2]
    dist = np.sqrt((diff**2).sum(axis=2))  # [N, N]
    mask = (dist < h) & (dist > 1e-10)
    return diff, dist, mask

def sph_step(particles, dt, h=H_SMOOTH):
    """
    One SPH time step.
    particles: [N, 4] — (x, y, vx, vy)
    Returns: [N, 4] — updated particles
    """
    n = len(particles)
    pos = particles[:, :2]  # [N, 2]
    vel = particles[:, 2:4]  # [N, 2]

    # Find neighbors
    diff, dist, mask = find_neighbors(pos, h)  # diff: [N,N,2], dist: [N,N]

    # Compute density: ρ_i = Σ_j m_j W(r_ij, h)
    W = W_poly6(dist, h)  # [N, N]
    W = W * mask  # zero out non-neighbors
    density = PARTICLE_MASS * W.sum(axis=1)  # [N]
    density = np.maximum(density, 1.0)  # avoid division by zero

    # Compute pressure: p = c²(ρ - ρ₀)
    pressure = C_SOUND**2 * (density - REST_DENSITY)  # [N]
    pressure = np.maximum(pressure, 0.0)  # no negative pressure

    # Compute pressure force: -∇p/ρ
    ax = np.zeros(n)
    ay = np.zeros(n)

    for i in range(n):
        neighbors = mask[i]
        if not neighbors.any():
            continue
        j_idx = np.where(neighbors)[0]
        for j in j_idx:
            r_ij = dist[i, j]
            dx_ij = diff[i, j, 0]
            dy_ij = diff[i, j, 1]
            gx, gy = gradW_spiky(dx_ij, dy_ij, r_ij, h)
            # Pressure force: -m_j * (p_i/ρ_i² + p_j/ρ_j²) * ∇W
            force_coeff = -PARTICLE_MASS * (pressure[i]/density[i]**2 + pressure[j]/density[j]**2)
            ax[i] += force_coeff * gx
            ay[i] += force_coeff * gy

            # Viscosity force
            vij_x = vel[j, 0] - vel[i, 0]
            vij_y = vel[j, 1] - vel[i, 1]
            W_v = W_viscosity(np.array([r_ij]), h)[0]
            visc_coeff = NU_SPH * PARTICLE_MASS * W_v / density[j]
            ax[i] += visc_coeff * vij_x
            ay[i] += visc_coeff * vij_y

    # Add gravity
    ay += GRAVITY

    # Integrate (semi-implicit Euler)
    new_vx = vel[:, 0] + ax * dt
    new_vy = vel[:, 1] + ay * dt
    new_x = pos[:, 0] + new_vx * dt
    new_y = pos[:, 1] + new_vy * dt

    # Boundary conditions (walls)
    # Left wall
    mask_l = new_x < PARTICLE_SPACING / 2
    new_x[mask_l] = PARTICLE_SPACING / 2
    new_vx[mask_l] *= -0.5  # damping
    # Right wall
    mask_r = new_x > DOMAIN_X - PARTICLE_SPACING / 2
    new_x[mask_r] = DOMAIN_X - PARTICLE_SPACING / 2
    new_vx[mask_r] *= -0.5
    # Bottom wall
    mask_b = new_y < PARTICLE_SPACING / 2
    new_y[mask_b] = PARTICLE_SPACING / 2
    new_vy[mask_b] *= -0.5
    # Top wall (optional, usually open)
    mask_t = new_y > DOMAIN_Y - PARTICLE_SPACING / 2
    new_y[mask_t] = DOMAIN_Y - PARTICLE_SPACING / 2
    new_vy[mask_t] *= -0.5

    result = np.stack([new_x, new_y, new_vx, new_vy], axis=1)
    return result.astype(np.float32)

def generate_trajectory(n_steps, add_noise=True):
    """Generate one SPH dam-break trajectory."""
    particles = init_dam_break()
    trajectory = [particles.copy()]
    for step in range(n_steps):
        particles = sph_step(particles, DT)
        trajectory.append(particles.copy())
    return np.array(trajectory)  # [T+1, N, 4]

print(f"\n[3] SPH Solver: Dam-break simulation")
print(f"  Generating {N_TRAJ_TRAIN + N_TRAJ_TEST} trajectories...")

# Generate training data
t0 = time.time()
train_trajectories = []
for i in range(N_TRAJ_TRAIN):
    traj = generate_trajectory(N_STEPS_TRAJ)
    train_trajectories.append(traj)
    if (i+1) % 20 == 0:
        print(f"  Train trajectory {i+1}/{N_TRAJ_TRAIN} ({time.time()-t0:.1f}s)")

# Generate test data
test_trajectories = []
for i in range(N_TRAJ_TEST):
    traj = generate_trajectory(N_STEPS_TRAJ)
    test_trajectories.append(traj)

print(f"  Data generation complete: {time.time()-t0:.1f}s")
n_particles = train_trajectories[0].shape[1]
print(f"  Particles per simulation: {n_particles}")

# ============================================================
# [4] GNN Model
# ============================================================
class SPHGNN(nn.Module):
    """
    Graph Neural Network for SPH particle simulation.

    Input:
      - Node features: [N, 4] (x, y, vx, vy)
      - Edge index: [2, E] (sender, receiver pairs)
      - Edge features: [E, 2] (relative position dx, dy)

    Output:
      - Node acceleration: [N, 2] (ax, ay)

    Architecture:
      1. Encoder: node/edge features → latent
      2. Message passing (3 rounds):
         - Aggregate messages from neighbors
         - Update node features
      3. Decoder: latent → acceleration
    """
    def __init__(self, node_dim=4, edge_dim=2, latent_dim=64, n_mp=3):
        super().__init__()
        self.n_mp = n_mp

        # Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            ) for _ in range(n_mp)
        ])
        self.node_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            ) for _ in range(n_mp)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 2),  # ax, ay
        )

    def forward(self, node_features, edge_index, edge_features):
        """
        node_features: [N, 4]
        edge_index: [2, E] — (sender, receiver)
        edge_features: [E, 2] — (dx, dy)
        """
        # Encode
        h_node = self.node_encoder(node_features)  # [N, latent]
        h_edge = self.edge_encoder(edge_features)  # [E, latent]

        # Message passing rounds
        for layer_idx in range(self.n_mp):
            senders = edge_index[0]  # [E]
            receivers = edge_index[1]  # [E]

            # Message: combine sender node + edge
            msg_input = torch.cat([h_node[senders], h_edge], dim=1)  # [E, 2*latent]
            messages = self.mp_layers[layer_idx](msg_input)  # [E, latent]

            # Aggregate: sum messages to receivers
            agg = torch.zeros_like(h_node)  # [N, latent]
            agg.index_add_(0, receivers, messages)

            # Update node
            update_input = torch.cat([h_node, agg], dim=1)  # [N, 2*latent]
            h_node = h_node + self.node_update[layer_idx](update_input)  # residual

        # Decode
        accel = self.decoder(h_node)  # [N, 2]
        return accel

def build_graph(positions, h=H_SMOOTH):
    """
    Build dynamic graph from particle positions.
    Returns edge_index [2, E] and edge_features [E, 2].
    """
    n = len(positions)
    pos = positions  # [N, 2]

    # Compute pairwise distances
    diff = pos[:, None, :] - pos[None, :, :]  # [N, N, 2]
    dist = torch.sqrt((diff**2).sum(dim=2) + 1e-10)  # [N, N]

    # Find neighbors (within h, excluding self)
    mask = (dist < h) & (dist > 1e-6)

    # Extract edges
    senders, receivers = torch.where(mask)

    # Edge features: relative position (from sender to receiver)
    edge_features = diff[senders, receivers]  # [E, 2]

    edge_index = torch.stack([senders, receivers])  # [2, E]

    return edge_index, edge_features

print(f"\n[4] GNN Model: SPHGNN")
print(f"  Architecture: Encoder → 3× MP → Decoder")
print(f"  Node features: (x, y, vx, vy) → 4")
print(f"  Edge features: (dx, dy) → 2")
print(f"  Latent dim: 64")
print(f"  Output: (ax, ay) → 2")

# ============================================================
# [5] Training
# ============================================================
model = SPHGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

N_EPOCHS = 60
BATCH_SIZE = 8  # trajectories per batch (sample random steps)

print(f"\n[5] Training:")
print(f"  Epochs: {N_EPOCHS}")
print(f"  Batch: {BATCH_SIZE} trajectories (random step sampling)")
print(f"  Optimizer: Adam (lr=1e-3)")

loss_history = []
t_start = time.time()

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    # Sample random batch of trajectories
    indices = np.random.choice(N_TRAJ_TRAIN, BATCH_SIZE, replace=False)

    for idx in indices:
        traj = train_trajectories[idx]  # [T+1, N, 4]

        # Sample random time step
        t = np.random.randint(0, N_STEPS_TRAJ)

        # Current state
        current = torch.tensor(traj[t], dtype=torch.float32, device=device)  # [N, 4]
        next_state = torch.tensor(traj[t+1], dtype=torch.float32, device=device)  # [N, 4]

        # Build graph from current positions
        pos = current[:, :2]
        edge_index, edge_features = build_graph(pos)

        # Predict acceleration
        pred_accel = model(current, edge_index, edge_features)  # [N, 2]

        # Target acceleration (from finite difference)
        target_accel = (next_state[:, 2:4] - current[:, 2:4]) / DT  # [N, 2]

        # Loss
        loss = F.mse_loss(pred_accel, target_accel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"Loss: {avg_loss:.6e} | "
              f"Time: {time.time()-t_start:.1f}s")

print(f"\n  Training complete! Total time: {time.time()-t_start:.1f}s")

# ============================================================
# [6] Rollout Evaluation (Auto-regressive)
# ============================================================
print(f"\n[6] Rollout Evaluation...")

def rollout(model, initial_state, n_steps, device):
    """Auto-regressive rollout: predict step-by-step."""
    model.eval()
    current = initial_state.clone()
    trajectory = [current.cpu().numpy().copy()]

    with torch.no_grad():
        for step in range(n_steps):
            pos = current[:, :2]
            edge_index, edge_features = build_graph(pos)
            accel = model(current, edge_index, edge_features)

            # Integrate (semi-implicit Euler)
            new_vel = current[:, 2:4] + accel * DT
            new_pos = current[:, :2] + new_vel * DT

            # Boundary conditions
            new_pos[:, 0] = torch.clamp(new_pos[:, 0], PARTICLE_SPACING/2, DOMAIN_X - PARTICLE_SPACING/2)
            new_pos[:, 1] = torch.clamp(new_pos[:, 1], PARTICLE_SPACING/2, DOMAIN_Y - PARTICLE_SPACING/2)
            # Damping at walls
            mask_l = new_pos[:, 0] <= PARTICLE_SPACING/2 + 1e-6
            mask_r = new_pos[:, 0] >= DOMAIN_X - PARTICLE_SPACING/2 - 1e-6
            mask_b = new_pos[:, 1] <= PARTICLE_SPACING/2 + 1e-6
            mask_t = new_pos[:, 1] >= DOMAIN_Y - PARTICLE_SPACING/2 - 1e-6
            new_vel[mask_l, 0] *= -0.5
            new_vel[mask_r, 0] *= -0.5
            new_vel[mask_b, 1] *= -0.5
            new_vel[mask_t, 1] *= -0.5

            current = torch.cat([new_pos, new_vel], dim=1)
            trajectory.append(current.cpu().numpy().copy())

    return np.array(trajectory)

# Run rollout on test trajectory
test_idx = 0
initial = torch.tensor(test_trajectories[test_idx][0], dtype=torch.float32, device=device)
pred_trajectory = rollout(model, initial, N_STEPS_TRAJ, device)
ref_trajectory = test_trajectories[test_idx]

# Compute position error over time
pos_errors = []
for t in range(min(len(pred_trajectory), len(ref_trajectory))):
    pred_pos = pred_trajectory[t][:, :2]
    ref_pos = ref_trajectory[t][:, :2]
    err = np.sqrt(((pred_pos - ref_pos)**2).sum(axis=1)).mean()
    pos_errors.append(err)

print(f"  Rollout steps: {len(pred_trajectory)-1}")
print(f"  Initial position error: {pos_errors[0]:.6f}")
print(f"  Final position error: {pos_errors[-1]:.6f}")

# ============================================================
# [7] Visualization
# ============================================================
print(f"\n[7] Visualization...")

# --- Figure 1: Dam-break snapshots (SPH vs GNN) ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
snapshots = [0, 20, 50, 100]

for col, step in enumerate(snapshots):
    if step >= len(ref_trajectory):
        step = len(ref_trajectory) - 1

    # Reference (SPH solver)
    ax = axes[0, col]
    ref_pos = ref_trajectory[step][:, :2]
    ref_vel = ref_trajectory[step][:, 2:4]
    speed = np.sqrt(ref_vel[:, 0]**2 + ref_vel[:, 1]**2)
    sc = ax.scatter(ref_pos[:, 0], ref_pos[:, 1], c=speed, cmap='viridis', s=15, vmin=0, vmax=3)
    ax.set_xlim(-0.05, DOMAIN_X + 0.05)
    ax.set_ylim(-0.05, DOMAIN_Y + 0.05)
    ax.set_aspect('equal')
    ax.set_title(f'Reference (SPH) t={step*DT:.3f}s', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.add_patch(Rectangle((0, 0), DOMAIN_X, DOMAIN_Y, fill=False, edgecolor='black', linewidth=2))
    if col == 0:
        plt.colorbar(sc, ax=ax, label='|v|', shrink=0.7)

    # GNN prediction
    ax = axes[1, col]
    if step < len(pred_trajectory):
        pred_pos = pred_trajectory[step][:, :2]
        pred_vel = pred_trajectory[step][:, 2:4]
        speed = np.sqrt(pred_vel[:, 0]**2 + pred_vel[:, 1]**2)
    else:
        pred_pos = pred_trajectory[-1][:, :2]
        speed = np.zeros(len(pred_pos))
    sc = ax.scatter(pred_pos[:, 0], pred_pos[:, 1], c=speed, cmap='viridis', s=15, vmin=0, vmax=3)
    ax.set_xlim(-0.05, DOMAIN_X + 0.05)
    ax.set_ylim(-0.05, DOMAIN_Y + 0.05)
    ax.set_aspect('equal')
    ax.set_title(f'GNN Prediction t={step*DT:.3f}s', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.add_patch(Rectangle((0, 0), DOMAIN_X, DOMAIN_Y, fill=False, edgecolor='black', linewidth=2))
    if col == 0:
        plt.colorbar(sc, ax=ax, label='|v|', shrink=0.7)

plt.suptitle('SPH Dam-Break: SPH Solver vs GNN (Lagrangian)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sph_dam_break.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sph_dam_break.png")

# --- Figure 2: Rollout error ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(np.arange(len(pos_errors)) * DT, pos_errors, 'b-', linewidth=2)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Mean Position Error', fontsize=12)
ax.set_title('GNN Rollout Error (Auto-regressive)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(['Position error (L2)'], fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sph_rollout_error.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sph_rollout_error.png")

# --- Figure 3: Training loss ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(loss_history, 'b-', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss (acceleration)', fontsize=12)
ax.set_title('SPH GNN Training Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sph_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sph_loss.png")

# --- Figure 4: Concept comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Eulerian (fixed mesh) vs Lagrangian (moving particles)
ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(-0.1, DOMAIN_X + 0.1)
ax.set_ylim(-0.1, DOMAIN_Y + 0.1)
# Draw fixed mesh (Eulerian)
for i in range(11):
    ax.plot([i * DOMAIN_X/10, i * DOMAIN_X/10], [0, DOMAIN_Y], 'b-', alpha=0.3, linewidth=0.5)
    ax.plot([0, DOMAIN_X], [i * DOMAIN_Y/10, i * DOMAIN_Y/10], 'b-', alpha=0.3, linewidth=0.5)
ax.set_title('Eulerian (Fixed Mesh)\nExisting GNN tutorials', fontsize=13, fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.text(0.5, -0.15, 'Nodes do NOT move\nStatic graph', transform=ax.transAxes,
        ha='center', fontsize=11, color='blue')

ax = axes[1]
ax.set_aspect('equal')
ax.set_xlim(-0.1, DOMAIN_X + 0.1)
ax.set_ylim(-0.1, DOMAIN_Y + 0.1)
# Draw moving particles (Lagrangian) at different times
colors = ['blue', 'green', 'red']
labels = ['t=0', 't=0.025s', 't=0.05s']
for idx, t_step in enumerate([0, 50, 100]):
    if t_step < len(ref_trajectory):
        pos = ref_trajectory[t_step][:, :2]
        ax.scatter(pos[:, 0], pos[:, 1], c=colors[idx], s=10, alpha=0.6, label=labels[idx])
ax.set_title('Lagrangian (Moving Particles)\nTHIS tutorial (SPH GNN)', fontsize=13, fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(fontsize=10)
ax.text(0.5, -0.15, 'Nodes MOVE with fluid\nDynamic graph (rebuilt each step)', transform=ax.transAxes,
        ha='center', fontsize=11, color='red')

plt.suptitle('Eulerian vs Lagrangian: Why SPH GNN is Different', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sph_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sph_concept.png")

# --- Figure 5: Dynamic graph visualization ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for col, t_step in enumerate([0, 50]):
    ax = axes[col]
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, DOMAIN_X + 0.1)
    ax.set_ylim(-0.1, DOMAIN_Y + 0.1)

    if t_step < len(ref_trajectory):
        pos = ref_trajectory[t_step][:, :2]
        ax.scatter(pos[:, 0], pos[:, 1], c='blue', s=20, zorder=2)

        # Draw edges (neighbors)
        pos_np = pos
        for i in range(min(len(pos_np), 50)):  # limit for clarity
            for j in range(i+1, len(pos_np)):
                r = np.sqrt(((pos_np[i] - pos_np[j])**2).sum())
                if r < H_SMOOTH:
                    ax.plot([pos_np[i, 0], pos_np[j, 0]],
                            [pos_np[i, 1], pos_np[j, 1]],
                            'r-', alpha=0.1, linewidth=0.3, zorder=1)

    ax.set_title(f'Graph at t={t_step*DT:.3f}s (h={H_SMOOTH})', fontsize=13, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.add_patch(Rectangle((0, 0), DOMAIN_X, DOMAIN_Y, fill=False, edgecolor='black', linewidth=2))

plt.suptitle('Dynamic Graph: Edges Rebuilt Every Step (Lagrangian)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sph_dynamic_graph.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sph_dynamic_graph.png")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "=" * 70)
print("[8] Summary")
print("=" * 70)

print(f"\n  Method: SPH (Smoothed Particle Hydrodynamics) + GNN")
print(f"  Framework: Lagrangian (particles move with fluid)")
print(f"  Problem: 2D Dam-Break (free surface flow)")
print(f"  Particles: {n_particles}")
print(f"  Graph: Dynamic (rebuilt every step, h={H_SMOOTH})")
print(f"")
print(f"  Rollout Error:")
print(f"    Initial: {pos_errors[0]:.6f}")
print(f"    Final:   {pos_errors[-1]:.6f}")
print(f"")
print(f"  Key difference from existing GNN tutorials:")
print(f"    GNN Beam/Rollout: Fixed mesh (Eulerian), static graph")
print(f"    THIS (SPH GNN):   Moving particles (Lagrangian), dynamic graph")
print(f"")
print(f"  Results saved to: {RESULTS_DIR}")
print("=" * 70)
