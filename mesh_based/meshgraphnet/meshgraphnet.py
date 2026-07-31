"""
PhysicsNeMo Tutorial: MeshGraphNet for Spring-Mass System (Data-Driven)
=======================================================================
This tutorial demonstrates the MeshGraphNet (MGN) model - a Graph Neural
Network (GNN) designed for learning mesh-based physics simulations.

MeshGraphNet is fundamentally different from PINN, FNO, and Transolver:
  - PINN: equation-based, point-by-point (no graph structure)
  - FNO: data-driven, regular grid, Fourier domain
  - Transolver: data-driven, attention-based, structured/irregular
  - MeshGraphNet: data-driven, GRAPH-based, message passing

Problem: Spring-Mass System
  - A set of nodes connected by springs (edges)
  - Input:  node positions (x, y), node velocities (vx, vy), node type
  - Output: node accelerations (ax, ay) at next time step
  - Graph: edges connect nodes that are physically linked by springs

Key MeshGraphNet concepts:
  1. Graph Representation: Physics is represented as a graph
     - Nodes = physical objects (mass points)
     - Edges = physical connections (springs)
     - Node features = position, velocity, type
     - Edge features = relative position, distance

  2. Message Passing: Information flows along edges
     - Edge Block: update edge features from connected nodes
     - Node Block: update node features from connected edges
     - Repeat for N processor blocks (default: 15)

  3. Encoder-Processor-Decoder architecture:
     - Encoder: raw features -> latent space (MLP)
     - Processor: message passing (graph neural network)
     - Decoder: latent space -> output (MLP)

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_meshgraphnet.py

==========================================================================
[What Makes MeshGraphNet Different?]
==========================================================================

Standard Neural Networks (MLP/CNN):
  - Fixed input size, no relational structure
  - Cannot naturally handle varying graph topologies

MeshGraphNet (GNN):
  - Input is a GRAPH (nodes + edges), not a grid or image
  - Message passing: nodes send messages to neighbors via edges
  - Permutation invariant: doesn't matter how you order the nodes
  - Can handle different graph sizes (10 nodes or 1000 nodes)

  MeshGraphNet internal pipeline:
    1. Node Encoder: node_features [N, F_node] -> [N, hidden] (MLP)
    2. Edge Encoder: edge_features [E, F_edge] -> [E, hidden] (MLP)
    3. Processor (15 blocks, each block = 1 edge update + 1 node update):
       a. Edge Block: concat(edge, src_node, dst_node) -> MLP -> new edge
       b. Node Block: aggregate(edges) + node -> MLP -> new node
    4. Node Decoder: [N, hidden] -> [N, output_dim] (MLP)

  The message passing is the key:
    - Each edge "sees" its source and destination nodes
    - Each node "sees" all its incoming edges (aggregated by sum)
    - This is repeated 15 times for long-range information flow
==========================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import physicsnemo
from physicsnemo.models.meshgraphnet import MeshGraphNet
from torch_geometric.data import Data
import time
import os

# ============================================================================
# [0] Environment Setup
# ============================================================================
print("=" * 70)
print("  PhysicsNeMo MeshGraphNet Tutorial: Spring-Mass System")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  PyTorch version:    {torch.__version__}")
print(f"  PhysicsNeMo version: {physicsnemo.__version__}")
print(f"  Device:             {device}")
if torch.cuda.is_available():
    print(f"  GPU:                {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU Memory:         {gpu_mem:.1f} GB")
print("=" * 70)
print()

torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# [1] Generate Synthetic Spring-Mass System Data
# ============================================================================
# We simulate a spring-mass system:
#   - N_NODES mass points in a 2D grid
#   - Springs connect adjacent nodes (grid-like mesh)
#   - Physics: F = -k * (|d| - rest_length) * d_hat  (Hooke's law)
#   - We generate (position, velocity) -> (acceleration) pairs
#
# Graph structure:
#   - Nodes: mass points with features [x, y, vx, vy, mass]
#   - Edges: springs with features [rel_x, rel_y, distance]
#   - edge_index: [2, num_edges] (source, destination) pairs

print("[1/7] Generating spring-mass system data...")

N_GRID_X = 5       # 5x4 grid of mass points
N_GRID_Y = 4
N_NODES = N_GRID_X * N_GRID_Y  # 20 nodes
N_SAMPLES = 300     # Number of training samples (different initial conditions)
K_SPRING = 50.0     # Spring constant
MASS = 1.0          # Mass of each node
REST_LENGTH = 1.0   # Rest length of springs
DAMPING = 0.1       # Damping coefficient

def build_grid_graph(nx, ny):
    """
    Build a grid graph for spring-mass system.
    Returns edge_index [2, num_edges] and edge rest lengths.
    
    Connections:
      - Horizontal: (i,j) <-> (i+1,j)
      - Vertical: (i,j) <-> (i,j+1)
      - Diagonal: (i,j) <-> (i+1,j+1) and (i+1,j) <-> (i,j+1)
    """
    edges_src = []
    edges_dst = []
    
    for i in range(nx):
        for j in range(ny):
            node_id = i * ny + j
            
            # Horizontal right
            if i < nx - 1:
                neighbor = (i + 1) * ny + j
                edges_src.append(node_id)
                edges_dst.append(neighbor)
                edges_src.append(neighbor)
                edges_dst.append(node_id)
            
            # Vertical up
            if j < ny - 1:
                neighbor = i * ny + (j + 1)
                edges_src.append(node_id)
                edges_dst.append(neighbor)
                edges_src.append(neighbor)
                edges_dst.append(node_id)
            
            # Diagonal
            if i < nx - 1 and j < ny - 1:
                neighbor = (i + 1) * ny + (j + 1)
                edges_src.append(node_id)
                edges_dst.append(neighbor)
                edges_src.append(neighbor)
                edges_dst.append(node_id)
            
            # Anti-diagonal
            if i < nx - 1 and j > 0:
                neighbor = (i + 1) * ny + (j - 1)
                edges_src.append(node_id)
                edges_dst.append(neighbor)
                edges_src.append(neighbor)
                edges_dst.append(node_id)
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index

def generate_spring_mass_data(n_samples, n_nodes, nx, ny, k_spring, mass, rest_len, damping, device):
    """
    Generate spring-mass system data.
    
    For each sample:
      - Random initial positions (perturbed from grid)
      - Random initial velocities
      - Compute accelerations using Hooke's law
    
    Returns:
      - node_features: [n_samples, n_nodes, 5] (x, y, vx, vy, mass)
      - edge_features: [n_samples, n_edges, 3] (rel_x, rel_y, distance)
      - targets: [n_samples, n_nodes, 2] (ax, ay)
      - edge_index: [2, n_edges]
    """
    edge_index = build_grid_graph(nx, ny)
    n_edges = edge_index.shape[1]
    
    all_node_features = []
    all_edge_features = []
    all_targets = []
    
    for s in range(n_samples):
        # Base grid positions
        xs = np.arange(nx, dtype=np.float32) * rest_len
        ys = np.arange(ny, dtype=np.float32) * rest_len
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten()], axis=1)  # [n_nodes, 2]
        
        # Add random perturbation to positions
        positions += np.random.randn(n_nodes, 2).astype(np.float32) * 0.1
        
        # Random velocities
        velocities = np.random.randn(n_nodes, 2).astype(np.float32) * 0.5
        
        # Compute edge features
        src_idx = edge_index[0].numpy()
        dst_idx = edge_index[1].numpy()
        
        rel_pos = positions[dst_idx] - positions[src_idx]  # [n_edges, 2]
        distances = np.linalg.norm(rel_pos, axis=1, keepdims=True)  # [n_edges, 1]
        edge_feats = np.concatenate([rel_pos, distances], axis=1)  # [n_edges, 3]
        
        # Compute forces and accelerations using Hooke's law
        # F = -k * (|d| - rest_length) * (d / |d|)
        forces = np.zeros_like(positions)  # [n_nodes, 2]
        
        for e in range(n_edges):
            s_node = src_idx[e]
            d_node = dst_idx[e]
            d_vec = positions[d_node] - positions[s_node]
            d_mag = np.linalg.norm(d_vec)
            if d_mag > 1e-8:
                f_mag = -k_spring * (d_mag - rest_len)
                f_vec = f_mag * (d_vec / d_mag)
                forces[s_node] += f_vec  # force on source from this spring
                forces[d_node] -= f_vec  # equal and opposite
        
        # Add damping force: F_damp = -c * v
        forces -= damping * velocities
        
        # Acceleration: a = F / m
        accelerations = forces / mass  # [n_nodes, 2]
        
        # Node features: [x, y, vx, vy, mass]
        node_feats = np.concatenate([
            positions,
            velocities,
            np.full((n_nodes, 1), mass, dtype=np.float32)
        ], axis=1)  # [n_nodes, 5]
        
        all_node_features.append(node_feats)
        all_edge_features.append(edge_feats)
        all_targets.append(accelerations)
    
    node_features = torch.tensor(np.stack(all_node_features), dtype=torch.float32, device=device)
    edge_features = torch.tensor(np.stack(all_edge_features), dtype=torch.float32, device=device)
    targets = torch.tensor(np.stack(all_targets), dtype=torch.float32, device=device)
    edge_index = edge_index.to(device)
    
    return node_features, edge_features, targets, edge_index

# Generate data
node_features, edge_features, targets, edge_index = generate_spring_mass_data(
    N_SAMPLES, N_NODES, N_GRID_X, N_GRID_Y, K_SPRING, MASS, REST_LENGTH, DAMPING, device
)

n_edges = edge_index.shape[1]
print(f"      Grid: {N_GRID_X}x{N_GRID_Y} = {N_NODES} nodes")
print(f"      Edges: {n_edges} (springs)")
print(f"      Samples: {N_SAMPLES}")
print(f"      Node features: {node_features.shape} (x, y, vx, vy, mass)")
print(f"      Edge features: {edge_features.shape} (rel_x, rel_y, distance)")
print(f"      Targets: {targets.shape} (ax, ay)")


# ============================================================================
# [2] Create MeshGraphNet Model
# ============================================================================
print("\n[2/7] Creating MeshGraphNet model...")

# ==========================================================================
# MeshGraphNet Architecture Explained:
# ==========================================================================
#
# Key parameters:
#
# input_dim_nodes (int):
#   Number of node features. For spring-mass: 5 (x, y, vx, vy, mass)
#
# input_dim_edges (int):
#   Number of edge features. For spring-mass: 3 (rel_x, rel_y, distance)
#
# output_dim (int):
#   Number of output values per node. For spring-mass: 2 (ax, ay)
#
# processor_size (int, default=15):
#   Number of message passing blocks. Each block = 1 edge update + 1 node update.
#   More blocks = longer range information flow (like deeper GNN).
#   15 is the original paper's default.
#
# hidden_dim_processor (int, default=128):
#   Hidden dimension used throughout the model (encoder, processor, decoder).
#   Larger = more capacity but more memory.
#
# aggregation (str, default="sum"):
#   How to aggregate messages from edges to nodes.
#   "sum": sum all incoming edge messages (most common)
#   "mean": average incoming edge messages
# ==========================================================================

model = MeshGraphNet(
    input_dim_nodes=5,        # x, y, vx, vy, mass
    input_dim_edges=3,        # rel_x, rel_y, distance
    output_dim=2,            # ax, ay (acceleration)
    processor_size=6,        # 6 message passing blocks (reduced from 15 for speed)
    hidden_dim_processor=64, # 64 hidden dim (reduced from 128 for speed)
    mlp_activation_fn="relu",# ReLU activation
    aggregation="sum",       # Sum aggregation for message passing
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"      MeshGraphNet parameters: {n_params:,}")
print(f"      Architecture: 6 processor blocks, 64 hidden dim")
print(f"      Node features: 5 -> 64 (encoder) -> 64 (processor) -> 2 (decoder)")
print(f"      Edge features: 3 -> 64 (encoder) -> 64 (processor)")


# ============================================================================
# [3] Training Setup
# ============================================================================
print("\n[3/7] Setting up training...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 500
loss_fn = nn.MSELoss()


# ============================================================================
# [4] Training Loop
# ============================================================================
print(f"\n[4/7] Training MeshGraphNet ({EPOCHS} epochs)...")
print(f"      Note: Message passing along graph edges (not grid convolution)")
print()

start_time = time.time()
loss_history = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # MeshGraphNet forward pass:
    #   node_features: [N_samples, N_nodes, F_node]
    #   edge_features: [N_samples, N_edges, F_edge]
    #   graph: PyG Data object with edge_index
    #
    # The model processes ALL samples simultaneously (batch dimension)
    # Each sample shares the same graph topology (edge_index)
    
    # We need to create a PyG Data object for each sample
    # But since all samples share the same graph, we can process them in batch
    # by creating a batched graph
    
    # For simplicity, process one sample at a time (or small batches)
    # Here we process all samples at once by treating the batch dim as separate graphs
    
    total_loss = 0.0
    batch_size = 50  # Process 50 samples at a time
    
    for batch_start in range(0, N_SAMPLES, batch_size):
        batch_end = min(batch_start + batch_size, N_SAMPLES)
        bs = batch_end - batch_start
        
        # Create a batched graph: replicate edge_index for each sample
        # PyG uses a block-diagonal edge_index for batching
        # edge_index_batch: [2, n_edges * batch_size]
        # with node indices offset by batch_size * N_NODES
        
        edge_index_batch = []
        for b in range(bs):
            offset = b * N_NODES
            edge_index_batch.append(edge_index + offset)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)  # [2, n_edges * bs]
        
        # Flatten node and edge features
        node_feat_flat = node_features[batch_start:batch_end].reshape(bs * N_NODES, 5)
        edge_feat_flat = edge_features[batch_start:batch_end].reshape(bs * n_edges, 3)
        
        # Create PyG Data object
        graph = Data(edge_index=edge_index_batch, num_nodes=bs * N_NODES)
        
        # Forward pass
        pred = model(node_feat_flat, edge_feat_flat, graph)  # [bs * N_NODES, 2]
        pred = pred.reshape(bs, N_NODES, 2)
        
        # Compute loss
        target_batch = targets[batch_start:batch_end]
        loss = loss_fn(pred, target_batch)
        total_loss += loss.item() * bs
    
    total_loss /= N_SAMPLES
    loss_history.append(total_loss)
    
    # Backward pass (on last batch's loss for simplicity)
    optimizer.zero_grad()
    
    # Recompute on full batch for gradient
    # For efficiency, just use last batch
    bs = min(batch_size, N_SAMPLES)
    edge_index_batch = []
    for b in range(bs):
        offset = b * N_NODES
        edge_index_batch.append(edge_index + offset)
    edge_index_batch = torch.cat(edge_index_batch, dim=1)
    
    node_feat_flat = node_features[:bs].reshape(bs * N_NODES, 5)
    edge_feat_flat = edge_features[:bs].reshape(bs * n_edges, 3)
    graph = Data(edge_index=edge_index_batch, num_nodes=bs * N_NODES)
    
    pred = model(node_feat_flat, edge_feat_flat, graph)
    pred = pred.reshape(bs, N_NODES, 2)
    loss = loss_fn(pred, targets[:bs])
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:5d}/{EPOCHS} | "
              f"Loss: {total_loss:.6e} | "
              f"Time: {elapsed:.1f}s")

elapsed_total = time.time() - start_time
print(f"\n  Training complete! Total time: {elapsed_total:.1f}s")
print(f"  Final loss: {total_loss:.6e}")


# ============================================================================
# [5] Evaluate Model
# ============================================================================
print("\n[5/7] Evaluating model on test data...")

# Generate test data
N_TEST = 20
test_node_feat, test_edge_feat, test_targets, _ = generate_spring_mass_data(
    N_TEST, N_NODES, N_GRID_X, N_GRID_Y, K_SPRING, MASS, REST_LENGTH, DAMPING, device
)

model.eval()
with torch.no_grad():
    # Process test samples one at a time
    test_preds = []
    for i in range(N_TEST):
        graph = Data(edge_index=edge_index, num_nodes=N_NODES)
        pred = model(test_node_feat[i], test_edge_feat[i], graph)
        test_preds.append(pred)
    
    test_preds = torch.stack(test_preds)  # [N_TEST, N_NODES, 2]
    
    # Compute relative L2 error
    l2_error = torch.norm(test_preds - test_targets) / torch.norm(test_targets)
    print(f"  Relative L2 error: {l2_error.item():.4f}")
    print(f"  Mean absolute error: {(test_preds - test_targets).abs().mean().item():.6f}")
    print(f"  Max absolute error: {(test_preds - test_targets).abs().max().item():.6f}")


# ============================================================================
# [6] Visualization
# ============================================================================
print("\n[6/7] Visualizing results...")

output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)

# Visualize a few test samples: show node positions with predicted vs true acceleration arrows
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, ax in enumerate(axes.flat):
    if idx >= N_TEST:
        break
    
    # Get sample data
    pos = test_node_feat[idx, :, :2].cpu().numpy()  # [N_NODES, 2]
    true_acc = test_targets[idx].cpu().numpy()  # [N_NODES, 2]
    pred_acc = test_preds[idx].cpu().numpy()  # [N_NODES, 2]
    
    # Draw edges
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    for e in range(len(src)):
        ax.plot([pos[src[e], 0], pos[dst[e], 0]],
                [pos[src[e], 1], pos[dst[e], 1]], 'k-', alpha=0.3, linewidth=0.5)
    
    # Draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], c='blue', s=50, zorder=5)
    
    # Draw true acceleration arrows (green)
    scale = 0.01
    ax.quiver(pos[:, 0], pos[:, 1], true_acc[:, 0], true_acc[:, 1],
              color='green', scale=1/scale, scale_units='xy', alpha=0.7, width=0.005)
    
    # Draw predicted acceleration arrows (red)
    ax.quiver(pos[:, 0], pos[:, 1], pred_acc[:, 0], pred_acc[:, 1],
              color='red', scale=1/scale, scale_units='xy', alpha=0.5, width=0.003)
    
    ax.set_title(f'Sample {idx+1}\nGreen=True, Red=Predicted', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(-1, N_GRID_X)
    ax.set_ylim(-1, N_GRID_Y)

plt.suptitle(
    f'MeshGraphNet: Spring-Mass System\n'
    f'Green arrows = True acceleration, Red arrows = Predicted\n'
    f'{N_NODES} nodes, {n_edges} edges, 6 processor blocks, {n_params:,} params',
    fontsize=13, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(output_dir, "meshgraphnet_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Result image saved: {fig_path}")

# Loss curve
fig2, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(loss_history, linewidth=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('MeshGraphNet Training Loss History', fontsize=14)
ax.grid(True, alpha=0.3)
fig2_path = os.path.join(output_dir, "meshgraphnet_loss.png")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Loss curve saved: {fig2_path}")

plt.close('all')


# ============================================================================
# [7] Summary
# ============================================================================
print("\n[7/7] Tutorial complete!")
print("=" * 70)
print("  MeshGraphNet vs FNO vs Transolver vs PINN Comparison:")
print("-" * 70)
print(f"  {'Feature':<28} {'PINN':<14} {'FNO':<14} {'Transolver':<14} {'MGN':<14}")
print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
print(f"  {'Learning type':<28} {'Equation':<14} {'Data-driven':<14} {'Data-driven':<14} {'Data-driven':<14}")
print(f"  {'Data structure':<28} {'Points':<14} {'Grid':<14} {'Grid/Mesh':<14} {'Graph':<14}")
print(f"  {'Core mechanism':<28} {'Autograd':<14} {'Fourier':<14} {'Attention':<14} {'MsgPass':<14}")
print(f"  {'Graph topology':<28} {'No':<14} {'No':<14} {'No':<14} {'Yes!':<14}")
print(f"  {'Irregular mesh':<28} {'Yes':<14} {'No':<14} {'Yes':<14} {'Yes':<14}")
print(f"  {'Varying topology':<28} {'N/A':<14} {'No':<14} {'No':<14} {'Yes!':<14}")
print(f"  {'Permutation invariant':<28} {'No':<14} {'No':<14} {'No':<14} {'Yes':<14}")
print(f"  {'PhysicsNeMo model':<28} {'FC':<14} {'FNO':<14} {'Transolver':<14} {'MGN':<14}")
print("-" * 70)
print()
print("  Key MeshGraphNet concepts demonstrated:")
print("    1. Graph Representation: physics as nodes (masses) + edges (springs)")
print("    2. Message Passing: edge blocks + node blocks, repeated 6 times")
print("    3. Encoder-Processor-Decoder: raw features -> latent -> message passing -> output")
print("    4. PyG Integration: uses torch_geometric Data objects for graph structure")
print()
print("  When to use MeshGraphNet:")
print("    - Physics has explicit graph/mesh structure (springs, CFD mesh, FEM)")
print("    - Different samples have different graph topologies")
print("    - Need permutation invariance (node ordering doesn't matter)")
print("    - Complex connectivity patterns (not regular grids)")
print()
print("  Result files:")
print(f"    - {fig_path}")
print(f"    - {fig2_path}")
print("=" * 70)
