"""
PhysicsNeMo GNN Tutorial: Beam Structural Analysis (FEM Mesh)
==============================================================
This tutorial demonstrates GNN for structural analysis on a FEM mesh.

Problem: Cantilever beam under tip load
  - Beam mesh: triangular elements (FEM-style)
  - Input:  node positions (x, y), boundary conditions (fixed/support), loads
  - Output: node displacements (ux, uy) — displacement field

Key difference from existing tutorials:
  - PINN Plane Stress: equation-based, MLP, regular grid, no mesh structure
  - MeshGraphNet:      data-driven, GNN, spring-mass (not structural FEM)
  - THIS tutorial:     data-driven, GNN, FEM mesh, structural analysis

This is the ONLY tutorial that:
  1. Uses FEM mesh (triangular elements) for structural analysis
  2. Predicts displacement field from loads (not acceleration from velocity)
  3. Compares GNN prediction with analytical beam theory

Physics:
  - Cantilever beam: fixed at left, point load at right tip
  - Euler-Bernoulli beam theory: δ = PL³/(3EI) (analytical reference)
  - FEM: triangular elements, plane stress
  - GNN: learns load → displacement mapping on the mesh graph

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
from matplotlib.collections import PolyCollection
from scipy.spatial import Delaunay

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo GNN Tutorial: Beam Structural Analysis (FEM Mesh)")
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
# [1] Generate FEM Mesh: Cantilever Beam
# ============================================================
# Beam: L x H, fixed at x=0, point load at x=L (tip)
# Mesh: structured triangular mesh (Delaunay)

BEAM_L = 4.0       # Beam length
BEAM_H = 0.5       # Beam height
NX = 17            # Nodes in x
NY = 5             # Nodes in y
E_MOD = 200e3      # Young's modulus (steel-like, scaled)
NU = 0.3           # Poisson's ratio

print(f"\n[1] Generating FEM mesh for cantilever beam...")
print(f"  Beam: {BEAM_L} x {BEAM_H}")
print(f"  Mesh: {NX} x {NY} = {NX*NY} nodes")

def create_beam_mesh(nx, ny, L, H):
    """Create structured triangular mesh for a beam."""
    x = np.linspace(0, L, nx)
    y = np.linspace(-H/2, H/2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    nodes = np.stack([X.flatten(), Y.flatten()], axis=1)  # [N, 2]
    
    # Create triangular elements using Delaunay
    tri = Delaunay(nodes)
    elements = tri.simplices  # [E, 3]
    
    return nodes, elements

nodes, elements = create_beam_mesh(NX, NY, BEAM_L, BEAM_H)
N_NODES = nodes.shape[0]
N_ELEM = elements.shape[0]
print(f"  Nodes: {N_NODES}, Elements: {N_ELEM} (triangular)")

# Build graph edges from mesh connectivity
def mesh_to_edges(nodes, elements):
    """Convert FEM mesh to graph edges (node connectivity)."""
    edges = set()
    for elem in elements:
        for i in range(3):
            for j in range(i+1, 3):
                e = tuple(sorted([elem[i], elem[j]]))
                edges.add(e)
    # Make bidirectional
    edge_list = []
    for s, d in edges:
        edge_list.append([s, d])
        edge_list.append([d, s])
    return np.array(edge_list, dtype=np.int64)

edges = mesh_to_edges(nodes, elements)
N_EDGES = edges.shape[0]
print(f"  Graph edges: {N_EDGES} (bidirectional)")

edge_index = torch.from_numpy(edges.T).to(device)  # [2, E]

# ============================================================
# [2] Generate Training Data: FEM-like Analytical Solutions
# ============================================================
# For each sample: random tip load → compute displacement field
# We use Euler-Bernoulli beam theory as ground truth (simplified FEM)
# δ_x(x,y) = -P*y*(6*L*x - 3*x²)/(6*E*I)  (axial)
# δ_y(x,y) = P*x²*(3*L - x)/(6*E*I)        (deflection)
# where I = H³/12 (unit width)

I_BEAM = BEAM_H**3 / 12.0  # Second moment of area

def compute_beam_displacement(nodes, P, L, H, E, I):
    """
    Compute displacement field for cantilever beam under tip load P.
    Uses Euler-Bernoulli beam theory (plane stress approximation).
    
    Returns: [N, 2] (ux, uy) for each node
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    # Beam deflection (Euler-Bernoulli)
    # δ_y(x) = P*x²*(3L - x) / (6*E*I)
    uy = P * x**2 * (3*L - x) / (6 * E * I)
    
    # Axial displacement (due to bending rotation)
    # δ_x(x,y) = -P*y*(6*L*x - 3*x²) / (6*E*I)
    ux = -P * y * (6*L*x - 3*x**2) / (6 * E * I)
    
    return np.stack([ux, uy], axis=1)  # [N, 2]

def generate_beam_data(n_samples, nodes, L, H, E, I):
    """Generate training data with random tip loads."""
    all_displacements = []
    all_loads = []
    
    for s in range(n_samples):
        # Random tip load (both magnitude and direction)
        Px = np.random.uniform(-0.5, 0.5)  # Horizontal load
        Py = np.random.uniform(-2.0, 2.0)  # Vertical load (main)
        
        # Compute displacement for combined load (superposition)
        disp_x = compute_beam_displacement(nodes, Px, L, H, E, I)[:, 0]
        disp_y = compute_beam_displacement(nodes, Py, L, H, E, I)[:, 1]
        
        # Also add cross terms (small for slender beam)
        disp = np.stack([disp_x, disp_y], axis=1)
        
        all_displacements.append(disp)
        all_loads.append([Px, Py])
    
    return np.array(all_displacements, dtype=np.float32), np.array(all_loads, dtype=np.float32)

N_TRAIN = 200
N_TEST = 30

print(f"\n[2] Generating training data...")
print(f"  Load: random tip load (Px: [-0.5, 0.5], Py: [-2.0, 2.0])")
print(f"  Ground truth: Euler-Bernoulli beam theory")
print(f"  Train: {N_TRAIN} samples, Test: {N_TEST} samples")

train_disp, train_loads = generate_beam_data(N_TRAIN, nodes, BEAM_L, BEAM_H, E_MOD, I_BEAM)
test_disp, test_loads = generate_beam_data(N_TEST, nodes, BEAM_L, BEAM_H, E_MOD, I_BEAM)

print(f"  Displacement shape: {train_disp.shape} [samples, nodes, 2]")
print(f"  Max displacement: {np.abs(train_disp).max():.6f}")

# Normalize
disp_mean = train_disp.mean()
disp_std = train_disp.std()
train_disp_n = (train_disp - disp_mean) / (disp_std + 1e-8)
test_disp_n = (test_disp - disp_mean) / (disp_std + 1e-8)

# Convert to tensors
train_disp_t = torch.from_numpy(train_disp_n).to(device)
test_disp_t = torch.from_numpy(test_disp_n).to(device)
train_loads_t = torch.from_numpy(train_loads).to(device)
test_loads_t = torch.from_numpy(test_loads).to(device)

# ============================================================
# [3] GNN Architecture: MeshGraphNet for Structural Analysis
# ============================================================
# Node features: [x, y, is_fixed, load_x, load_y]
#   - x, y: node position
#   - is_fixed: 1.0 if fixed boundary, 0.0 otherwise
#   - load_x, load_y: applied load at this node (0 for most, nonzero at tip)
# Edge features: [rel_x, rel_y, distance]
# Output: [ux, uy] displacement per node

class MeshGraphNetSimple(nn.Module):
    """
    Simplified MeshGraphNet for structural analysis.
    Encoder-Processor-Decoder architecture with message passing.
    """
    def __init__(self, node_in_dim, edge_in_dim, out_dim, hidden=64, n_processor=6):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()
        )
        
        # Processor: message passing blocks
        self.edge_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden*3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_processor)
        ])
        self.node_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_processor)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, node_feat, edge_feat, edge_index, batch_size=1):
        """
        node_feat: [B*N, F_node]
        edge_feat: [B*E, F_edge]
        edge_index: [2, E] (shared across batch)
        """
        h_node = self.node_encoder(node_feat)  # [B*N, H]
        h_edge = self.edge_encoder(edge_feat)  # [B*E, H]
        
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]
        
        for eb, nb in zip(self.edge_blocks, self.node_blocks):
            # Edge update: concat(edge, src_node, dst_node)
            msg = torch.cat([h_edge, h_node[src], h_node[dst]], dim=-1)
            h_edge = h_edge + eb(msg)
            
            # Node update: aggregate edges + node
            # scatter_add messages to destination nodes
            agg = torch.zeros_like(h_node)
            agg.index_add_(0, dst, h_edge)
            h_node = h_node + nb(torch.cat([h_node, agg], dim=-1))
        
        return self.decoder(h_node)  # [B*N, out_dim]

# Prepare node and edge features
def prepare_features(nodes, loads, edge_index, n_samples):
    """Prepare node and edge features for all samples."""
    N = nodes.shape[0]
    E = edge_index.shape[1]
    
    # Node features: [x, y, is_fixed, load_x, load_y]
    # is_fixed: 1.0 for x=0 (left boundary), 0.0 otherwise
    is_fixed = (nodes[:, 0] < 1e-6).astype(np.float32)
    
    # Load is applied at the rightmost nodes (tip)
    is_tip = (nodes[:, 0] > BEAM_L - 1e-6).astype(np.float32)
    
    all_node_feat = []
    for s in range(n_samples):
        load_x = np.zeros(N, dtype=np.float32)
        load_y = np.zeros(N, dtype=np.float32)
        # Apply load at tip nodes
        load_x[is_tip > 0] = loads[s, 0] / max(is_tip.sum(), 1)
        load_y[is_tip > 0] = loads[s, 1] / max(is_tip.sum(), 1)
        
        nf = np.stack([nodes[:, 0], nodes[:, 1], is_fixed, load_x, load_y], axis=1)
        all_node_feat.append(nf)
    
    all_node_feat = np.array(all_node_feat, dtype=np.float32)  # [S, N, 5]
    
    # Edge features: [rel_x, rel_y, distance] (same for all samples)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    rel_pos = nodes[dst] - nodes[src]
    dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)
    edge_feat = np.concatenate([rel_pos, dist], axis=1).astype(np.float32)  # [E, 3]
    
    # Repeat for all samples
    all_edge_feat = np.tile(edge_feat, (n_samples, 1))  # [S*E, 3]
    
    return all_node_feat, all_edge_feat

print(f"\n[3] Building GNN model...")
train_node_feat, train_edge_feat = prepare_features(nodes, train_loads, edge_index, N_TRAIN)
test_node_feat, test_edge_feat = prepare_features(nodes, test_loads, edge_index, N_TEST)

train_node_feat_t = torch.from_numpy(train_node_feat).to(device)
train_edge_feat_t = torch.from_numpy(train_edge_feat).to(device)
test_node_feat_t = torch.from_numpy(test_node_feat).to(device)
test_edge_feat_t = torch.from_numpy(test_edge_feat).to(device)

model = MeshGraphNetSimple(
    node_in_dim=5, edge_in_dim=3, out_dim=2,
    hidden=64, n_processor=6
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: MeshGraphNetSimple (6 processor blocks, 64 hidden)")
print(f"  Parameters: {n_params:,}")
print(f"  Node features: [x, y, is_fixed, load_x, load_y] (5)")
print(f"  Edge features: [rel_x, rel_y, distance] (3)")
print(f"  Output: [ux, uy] displacement (2)")

# ============================================================
# [4] Training
# ============================================================
EPOCHS = 500
BATCH_SIZE = 20
LR = 1e-3

print(f"\n[4] Training GNN ({EPOCHS} epochs)")
print("-" * 70)

opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.MSELoss()

train_losses = []
test_losses = []
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(N_TRAIN)
    epoch_loss = 0; n_batches = 0
    
    for i in range(0, N_TRAIN, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        bs = len(idx)
        
        # Prepare batched graph
        node_feat = train_node_feat_t[idx].reshape(bs * N_NODES, 5)
        edge_feat = train_edge_feat_t[idx * N_EDGES:(idx + 1) * N_EDGES].reshape(bs * N_EDGES, 3)
        # Wait, edge_feat is [S*E, 3] but we need [bs*E, 3]
        # Actually edge_feat is tiled: [S, E, 3] flattened to [S*E, 3]
        # For batch idx, we need rows idx[0]*E to idx[-1]*E+E
        edge_feat = train_edge_feat_t.reshape(N_TRAIN, N_EDGES, 3)[idx].reshape(bs * N_EDGES, 3)
        
        # Batched edge_index: offset by N_NODES per sample
        edge_idx_batch = []
        for b in range(bs):
            edge_idx_batch.append(edge_index + b * N_NODES)
        edge_idx_batch = torch.cat(edge_idx_batch, dim=1)
        
        pred = model(node_feat, edge_feat, edge_idx_batch)
        pred = pred.reshape(bs, N_NODES, 2)
        
        target = train_disp_t[idx]
        loss = loss_fn(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_batches += 1
    
    train_losses.append(epoch_loss / n_batches)
    sched.step()
    
    # Test
    model.eval()
    with torch.no_grad():
        node_feat = test_node_feat_t.reshape(N_TEST * N_NODES, 5)
        edge_feat = test_edge_feat_t.reshape(N_TEST, N_EDGES, 3).reshape(N_TEST * N_EDGES, 3)
        edge_idx_batch = []
        for b in range(N_TEST):
            edge_idx_batch.append(edge_index + b * N_NODES)
        edge_idx_batch = torch.cat(edge_idx_batch, dim=1)
        pred = model(node_feat, edge_feat, edge_idx_batch).reshape(N_TEST, N_NODES, 2)
        test_loss = loss_fn(pred, test_disp_t).item()
    test_losses.append(test_loss)
    
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:4d} | Train: {train_losses[-1]:.6e} | Test: {test_loss:.6e} | Time: {time.time()-start:.1f}s")

train_time = time.time() - start
print("-" * 70)

# ============================================================
# [5] Evaluation
# ============================================================
print(f"\n[5] Evaluating on test data...")

model.eval()
with torch.no_grad():
    node_feat = test_node_feat_t.reshape(N_TEST * N_NODES, 5)
    edge_feat = test_edge_feat_t.reshape(N_TEST, N_EDGES, 3).reshape(N_TEST * N_EDGES, 3)
    edge_idx_batch = []
    for b in range(N_TEST):
        edge_idx_batch.append(edge_index + b * N_NODES)
    edge_idx_batch = torch.cat(edge_idx_batch, dim=1)
    preds = model(node_feat, edge_feat, edge_idx_batch).reshape(N_TEST, N_NODES, 2)

# Denormalize
preds_denorm = (preds.cpu().numpy() * disp_std + disp_mean)
test_disp_denorm = test_disp

# Relative L2 error
l2_errors = []
for i in range(N_TEST):
    err = np.linalg.norm(preds_denorm[i] - test_disp_denorm[i]) / (np.linalg.norm(test_disp_denorm[i]) + 1e-12)
    l2_errors.append(err)

l2_errors = np.array(l2_errors)
print(f"  Mean L2 error: {l2_errors.mean():.4f} ({l2_errors.mean()*100:.2f}%)")
print(f"  Min L2 error:  {l2_errors.min():.4f}")
print(f"  Max L2 error:  {l2_errors.max():.4f}")

# Tip deflection comparison (analytical vs GNN)
tip_idx = np.argmax(nodes[:, 0])  # Rightmost node
tip_disp_true = test_disp_denorm[:, tip_idx, 1]  # uy at tip
tip_disp_pred = preds_denorm[:, tip_idx, 1]

# Analytical: δ_tip = P*L³ / (3*E*I)
tip_loads = test_loads[:, 1]  # Py
tip_analytical = tip_loads * BEAM_L**3 / (3 * E_MOD * I_BEAM)

print(f"\n  Tip deflection comparison (sample 0):")
print(f"    Load Py = {test_loads[0, 1]:.4f}")
print(f"    Analytical: {tip_analytical[0]:.6f}")
print(f"    GNN pred:   {tip_disp_pred[0]:.6f}")
print(f"    Error:       {abs(tip_disp_pred[0] - tip_analytical[0]):.6f}")

# ============================================================
# [6] Visualization
# ============================================================
print(f"\n[6] Generating visualizations...")

# --- Figure 1: Training loss ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(train_losses, linewidth=1.5, color='blue')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss (MSE)')
ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)
ax2.semilogy(test_losses, linewidth=1.5, color='red')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Test Loss (MSE)')
ax2.set_title('Test Loss'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_beam_loss.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Deformed shape comparison ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
scale = 50  # Displacement magnification for visualization

for idx in range(6):
    ax = axes[idx // 3, idx % 3]
    
    # Original mesh
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color='gray', alpha=0.3, linewidth=0.5)
    
    # Deformed (true)
    disp_true = test_disp_denorm[idx]
    nodes_def_true = nodes + disp_true * scale
    ax.triplot(nodes_def_true[:, 0], nodes_def_true[:, 1], elements, color='blue', alpha=0.5, linewidth=1.0, label='True')
    
    # Deformed (GNN)
    disp_pred = preds_denorm[idx]
    nodes_def_pred = nodes + disp_pred * scale
    ax.triplot(nodes_def_pred[:, 0], nodes_def_pred[:, 1], elements, color='red', alpha=0.5, linewidth=1.0, linestyle='--', label='GNN')
    
    # Fixed boundary
    fixed_nodes = nodes[nodes[:, 0] < 1e-6]
    ax.scatter(fixed_nodes[:, 0], fixed_nodes[:, 1], c='black', s=30, marker='s', zorder=5, label='Fixed')
    
    # Load arrow
    tip_node = nodes[np.argmax(nodes[:, 0])]
    ax.annotate('', xy=(tip_node[0], tip_node[1] + test_loads[idx, 1] * 0.1),
                xytext=(tip_node[0], tip_node[1]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_title(f'Sample {idx+1}\nPy={test_loads[idx, 1]:.2f}, L2={l2_errors[idx]*100:.2f}%')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper left')

plt.suptitle(f'Cantilever Beam: GNN vs True (deformation x{scale})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_beam_deformation.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: Displacement field comparison ---
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
sample_idx = 0

# uy (vertical displacement) — the main component
for col, (data, title, cmap) in enumerate([
    (test_disp_denorm[sample_idx, :, 1], 'True uy', 'RdBu_r'),
    (preds_denorm[sample_idx, :, 1], 'GNN uy', 'RdBu_r'),
    (np.abs(preds_denorm[sample_idx, :, 1] - test_disp_denorm[sample_idx, :, 1]), 'Error', 'hot')
]):
    for row, comp in enumerate(['uy', 'ux']):
        if row == 0:
            d = data if col < 2 else data
        else:
            if col < 2:
                d = test_disp_denorm[sample_idx, :, 0] if col == 0 else preds_denorm[sample_idx, :, 0]
            else:
                d = np.abs(preds_denorm[sample_idx, :, 0] - test_disp_denorm[sample_idx, :, 0])
        
        ax = axes[row, col]
        sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=d, cmap=cmap, s=20)
        ax.triplot(nodes[:, 0], nodes[:, 1], elements, color='gray', alpha=0.2, linewidth=0.3)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f'{title} ({comp})')
        ax.set_aspect('equal')

# Row 3: tip deflection scatter
ax = axes[2, 0]
ax.scatter(tip_analytical, tip_disp_pred, c='blue', s=30, alpha=0.7)
lim = max(abs(tip_analytical).max(), abs(tip_disp_pred).max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.5)
ax.set_xlabel('Analytical Tip Deflection')
ax.set_ylabel('GNN Tip Deflection')
ax.set_title('Tip Deflection: Analytical vs GNN')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# L2 error distribution
ax = axes[2, 1]
ax.hist(l2_errors * 100, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('L2 Error (%)')
ax.set_ylabel('Count')
ax.set_title(f'L2 Error Distribution\n(mean: {l2_errors.mean()*100:.2f}%)')
ax.grid(True, alpha=0.3)

# Mesh visualization
ax = axes[2, 2]
ax.triplot(nodes[:, 0], nodes[:, 1], elements, color='steelblue', alpha=0.5, linewidth=0.5)
ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=10, zorder=5)
fixed_nodes = nodes[nodes[:, 0] < 1e-6]
ax.scatter(fixed_nodes[:, 0], fixed_nodes[:, 1], c='black', s=50, marker='s', zorder=6, label='Fixed')
tip_nodes = nodes[nodes[:, 0] > BEAM_L - 1e-6]
ax.scatter(tip_nodes[:, 0], tip_nodes[:, 1], c='green', s=50, marker='^', zorder=6, label='Loaded')
ax.set_title(f'FEM Mesh\n{N_NODES} nodes, {N_ELEM} elements')
ax.set_aspect('equal')
ax.legend(fontsize=8)

plt.suptitle('Cantilever Beam: GNN Structural Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_beam_result.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: Equation explanation ---
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.text(0.5, 0.92, 'Cantilever Beam: GNN Structural Analysis', 
        ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.75,
    'Problem:\n'
    '  • Cantilever beam (fixed at left, load at right)\n'
    '  • FEM mesh: triangular elements\n'
    '  • Input: node positions + BC + loads\n'
    '  • Output: displacement field (ux, uy)',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.75,
    'GNN Architecture:\n'
    '  • Node features: [x, y, is_fixed, load_x, load_y]\n'
    '  • Edge features: [rel_x, rel_y, distance]\n'
    '  • 6 message passing blocks (encoder-processor-decoder)\n'
    '  • Output: [ux, uy] per node',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.05, 0.45,
    'Analytical reference:\n'
    '  • Euler-Bernoulli beam theory\n'
    '  • δ_tip = P·L³ / (3·E·I)\n'
    '  • I = H³/12 (second moment of area)\n'
    '  • GNN learns this mapping from data',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.45,
    'vs. PINN Plane Stress:\n'
    '  • PINN: equation-based, MLP, regular grid\n'
    '  • GNN: data-driven, graph, FEM mesh\n'
    '  • PINN: no mesh structure\n'
    '  • GNN: mesh topology is explicit input',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.05, 0.15,
    'Key advantage:\n'
    '  • GNN handles ANY mesh topology\n'
    '  • Different beam shapes, holes, notches\n'
    '  • No need for regular grid\n'
    '  • Mesh connectivity is explicit input',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.text(0.55, 0.15,
    'Engineering workflow:\n'
    '  1. Generate FEM mesh (any shape)\n'
    '  2. Apply BCs and loads as node features\n'
    '  3. GNN predicts displacement field\n'
    '  4. Compare with FEM solver\n'
    '  5. Use as fast surrogate for design iteration',
    fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='top')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_beam_explanation.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [7] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: GNN — Beam Structural Analysis (FEM Mesh)")
print("=" * 70)
print(f"  Problem:            Cantilever beam, tip load")
print(f"  Beam:               {BEAM_L} x {BEAM_H}")
print(f"  Mesh:               {N_NODES} nodes, {N_ELEM} triangular elements, {N_EDGES} edges")
print(f"  Train samples:      {N_TRAIN}")
print(f"  Test samples:       {N_TEST}")
print(f"  Epochs:             {EPOCHS}")
print(f"  Training time:      {train_time:.1f}s")
print(f"  --- Results ---")
print(f"  Mean L2 error:      {l2_errors.mean()*100:.2f}%")
print(f"  Min L2 error:       {l2_errors.min()*100:.2f}%")
print(f"  Max L2 error:       {l2_errors.max()*100:.2f}%")
print()
print("Key observations:")
print("  1. FEM MESH: Triangular elements → graph (nodes + edges)")
print("  2. STRUCTURAL ANALYSIS: Load → displacement field (not acceleration)")
print("  3. BOUNDARY CONDITIONS: is_fixed as node feature (GNN learns BC)")
print("  4. ANALYTICAL VALIDATION: Euler-Bernoulli beam theory as ground truth")
print("  5. vs PINN: GNN uses mesh topology explicitly; PINN has no mesh structure")
print("  6. ENGINEERING: Fast surrogate for design iteration (no FEM solve needed)")
print("=" * 70)
