# PhysicsNeMo MeshGraphNet Tutorial Guide (Detailed Guide for Beginners)

> **Date**: 2026-07-10  
> **Environment**: PhysicsNeMo 1.3.0, PyTorch 2.7.1+cu118, CUDA (Quadro P4000 8GB)  
> **Location**: `E:\physicsnemo_env`

---

## Table of Contents

1. [Tutorial Overview](#1-tutorial-overview)
2. [What is MeshGraphNet? (Beginner-Friendly Explanation)](#2-what-is-meshgraphnet-beginner-friendly-explanation)
3. [Understanding Message Passing](#3-understanding-message-passing)
4. [How to Run the Tutorial](#4-how-to-run-the-tutorial)
5. [Step-by-Step Code Explanation](#5-step-by-step-code-explanation)
6. [Interpreting Results](#6-interpreting-results)
7. [PINN vs FNO vs Transolver vs MGN Comparison](#7-pinn-vs-fno-vs-transolver-vs-mgn-comparison)
8. [Parameter Experimentation Guide](#8-parameter-experimentation-guide)
9. [Troubleshooting (FAQ)](#9-troubleshooting-faq)

---

## 1. Tutorial Overview

### What Did We Learn?

In this tutorial, we used the **MeshGraphNet (MGN)** model to solve a **spring-mass system** problem.

| Item | Details |
|------|---------|
| **Problem** | Spring-mass system (node position/velocity → acceleration prediction) |
| **Physics law** | Hooke's law (F = -k·Δx) + damping |
| **Learning method** | Data-driven (supervised learning) |
| **Data structure** | Graph (nodes = mass points, edges = springs) |
| **Neural network** | PhysicsNeMo MeshGraphNet (6 processor blocks, 64 hidden) |
| **Parameters** | 301,186 |
| **Training time** | ~51 seconds (GPU: Quadro P4000) |
| **Final loss** | 23.78 (converged from 2486) |
| **Relative L2 error** | 0.0957 (9.57%) |
| **Result** | Acceleration prediction successful (prediction ≈ actual) |

### Files Used

| File | Description |
|------|-------------|
| `tutorial_meshgraphnet.py` | Main tutorial script (MGN implementation) |
| `tutorial_results/meshgraphnet_result.png` | Prediction visualization (nodes, edges, acceleration arrows) |
| `tutorial_results/meshgraphnet_loss.png` | Training loss curve |

---

## 2. What is MeshGraphNet? (Beginner-Friendly Explanation)

### 2.1 What is MeshGraphNet?

**MeshGraphNet (MGN)** is a Graph Neural Network (GNN)-based model that directly learns the **mesh structure** of physics simulations.

- **Mesh** + **Graph** + **Net** = **MeshGraphNet**
- Represents physical systems as **graphs**: nodes = objects, edges = connections
- Information is transmitted through **Message Passing**

### 2.2 Why Graphs?

Limitations of existing models:

| Model | Data Structure | Limitation |
|-------|---------------|------------|
| PINN | Points (x, y) | Cannot represent node-to-node connections |
| FNO | Regular grid | Cannot handle irregular meshes |
| Transolver | Grid/mesh | Difficult to directly process graph topology |

**MeshGraphNet advantages**:
- Directly models **connection relationships** between nodes
- Naturally handles **irregular meshes** (CFD, FEM)
- Supports **variable graph structures** (different connections per sample)
- **Permutation invariance** (same result regardless of node ordering)

### 2.3 Spring-Mass System Example

```
    o---o---o---o---o     o = mass point (node)
    | \ | \ | \ | \ |     - = spring (edge)
    o---o---o---o---o     \ = diagonal spring
    | \ | \ | \ | \ |
    o---o---o---o---o
    | \ | \ | \ | \ |
    o---o---o---o---o

    5x4 grid = 20 nodes, 110 edges
```

Node features: position (x,y), velocity (vx,vy), mass (m)  
Edge features: relative position (rel_x, rel_y), distance  
Prediction target: acceleration (ax, ay) for each node

---

## 3. Understanding Message Passing

### 3.1 What is Message Passing?

**Message Passing** is the process of information flowing through a graph:

```
1. Edge collects information from its two connected nodes
   "What is the state of the mass points at both ends of this spring?"

2. Edge feature update
   "Update spring state based on both nodes' information"

3. Each node aggregates messages from all connected edges
   "Sum the forces from all springs connected to this mass point"

4. Node feature update
   "Update node state with aggregated messages"
```

Analogy:
- **Standard neural network**: Each student studies alone (no information exchange)
- **Message passing**: Friends exchange information and study together (6 repetitions = information reaches 6-hop friends)

### 3.2 MeshGraphNet's 3-Stage Structure

```
[Input] → [Encoder] → [Processor] → [Decoder] → [Output]
           (MLP)     (Message Passing)  (MLP)
                     6 blocks repeated
```

#### (1) Encoder
- Node encoder: raw node features [5] → latent space [64] (MLP)
- Edge encoder: raw edge features [3] → latent space [64] (MLP)
- "Transforms physical features into a format suitable for neural network processing"

#### (2) Processor - Core!
- 6 message passing blocks (original paper uses 15)
- Each block = 1 edge block + 1 node block
- **Edge block**: concat(edge, source_node, dest_node) → MLP → new edge features
- **Node block**: sum(incoming edges) + node → MLP → new node features
- "Information transfer mimicking physical interactions"

#### (3) Decoder
- Node decoder: latent space [64] → output [2] (MLP)
- "Transforms latent representation to physical output (acceleration)"

### 3.3 Effect of Message Passing

```
Block 1: Get info from directly connected neighbors (1-hop)
Block 2: Get info from neighbors of neighbors (2-hop)
Block 3: Get info from 3-hop nodes
...
Block 6: Information reaches 6-hop nodes

→ 6 message passing rounds spread information across the entire graph
```

---

## 4. How to Run the Tutorial

### Prerequisites: Package Installation

```cmd
cd E:\physicsnemo_env
Scripts\activate
pip install torch_geometric
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
```

### Run

```cmd
cd E:\physicsnemo_env
Scripts\activate
python tutorial_meshgraphnet.py
```

### View Results

```cmd
start E:\physicsnemo_env\tutorial_results\meshgraphnet_result.png
start E:\physicsnemo_env\tutorial_results\meshgraphnet_loss.png
```

---

## 5. Step-by-Step Code Explanation

### 5.1 Overall Structure (7 Steps)

```
tutorial_meshgraphnet.py
|
+-- [0] Environment Setup    - Check GPU, fix random seeds
+-- [1] Data Generation      - Synthetic spring-mass system data
+-- [2] Model Creation       - Create MeshGraphNet neural network
+-- [3] Training Setup       - Configure optimizer, epochs
+-- [4] Training Loop        - 500 epochs of iterative training
+-- [5] Model Evaluation     - Compute L2 error on test data
+-- [6] Visualization         - Plot acceleration arrows
+-- [7] Summary              - Compare 4 models
```

### 5.2 [1] Data Generation: Graph Construction

```python
N_GRID_X = 5       # 5x4 grid
N_GRID_Y = 4
N_NODES = 20        # 20 mass points
N_SAMPLES = 300     # 300 training samples
```

Graph construction function `build_grid_graph()`:
- Horizontal connections: (i,j) ↔ (i+1,j)
- Vertical connections: (i,j) ↔ (i,j+1)
- Diagonal connections: (i,j) ↔ (i+1,j+1)
- Anti-diagonal: (i+1,j) ↔ (i,j+1)

Result: 20 nodes, 110 edges (bidirectional)

**Data format**:
- Node features: `[N_samples, N_nodes, 5]` (x, y, vx, vy, mass)
- Edge features: `[N_samples, N_edges, 3]` (rel_x, rel_y, distance)
- Targets: `[N_samples, N_nodes, 2]` (ax, ay)
- edge_index: `[2, N_edges]` (source, destination pairs)

### 5.3 [2] MeshGraphNet Model Creation

```python
model = MeshGraphNet(
    input_dim_nodes=5,        # Number of node features (x, y, vx, vy, mass)
    input_dim_edges=3,        # Number of edge features (rel_x, rel_y, distance)
    output_dim=2,            # Output dimension (ax, ay)
    processor_size=6,        # Number of message passing blocks
    hidden_dim_processor=64, # Hidden dimension
    mlp_activation_fn="relu", # ReLU activation
    aggregation="sum",       # Message aggregation method
)
```

#### Key Parameter Explanation

**input_dim_nodes / input_dim_edges**:
- Number of node/edge features. Determined by physical variables.
- Spring-mass: 5 nodes, 3 edges
- CFD: 6 nodes (x,y,z,vx,vy,vz), 3 edges (rel_x,rel_y,rel_z)

**processor_size** ⭐:
- **Number of message passing blocks**. The most important parameter!
- More blocks: information travels farther, but slower
- Original paper: 15, tutorial: 6 (speed priority)
- "How many steps will information spread through the graph?"

**hidden_dim_processor**:
- Hidden dimension for all MLPs. Larger = more expressiveness.
- Original paper: 128, tutorial: 64 (memory savings)

**aggregation**:
- Edge → node message aggregation method
- "sum": Sum all edge messages (most common, similar to physical force summation)
- "mean": Average edge messages

### 5.4 [4] Training Loop: Graph Batching

```python
for epoch in range(500):
    # Graph batching: combine multiple samples into one large graph
    edge_index_batch = []
    for b in range(batch_size):
        offset = b * N_NODES
        edge_index_batch.append(edge_index + offset)
    edge_index_batch = torch.cat(edge_index_batch, dim=1)
    
    # Create PyG Data object
    graph = Data(edge_index=edge_index_batch, num_nodes=bs * N_NODES)
    
    # Forward pass
    pred = model(node_feat_flat, edge_feat_flat, graph)
    loss = loss_fn(pred, target)
    loss.backward()
    optimizer.step()
```

**What is graph batching?**
- Combining multiple graphs into one large graph
- Shifting node indices by offset to prevent overlap
- PyTorch Geometric handles this automatically (here implemented manually)

### 5.5 PyG Data Object

```python
from torch_geometric.data import Data

graph = Data(
    edge_index=edge_index,  # [2, num_edges] (source, destination)
    num_nodes=N_NODES,      # Number of nodes
)
```

PyG Data defines the graph structure:
- `edge_index`: Which nodes are connected to which
- `num_nodes`: Total number of nodes
- Node/edge features are passed to the model separately

---

## 6. Interpreting Results

### 6.1 Training Log

```
Epoch   0/500 | Loss: 2.486432e+03 | Time: 1.0s    (initial)
Epoch  50/500 | Loss: 2.489172e+02 | Time: 5.9s    (rapid decrease)
Epoch 100/500 | Loss: 6.062692e+01 | Time: 10.5s   (continuing decrease)
Epoch 200/500 | Loss: 3.169598e+01 | Time: 21.0s   (gradual decrease)
Epoch 400/500 | Loss: 2.157091e+01 | Time: 41.8s   (fine-tuning)
Epoch 499/500 | Loss: 2.377823e+01 | Time: 50.9s   (converged)
```

### 6.2 Loss Curve Analysis (meshgraphnet_loss.png)

- **0~100 epochs**: 2486 → 60 (sharp decrease, sawtooth pattern)
- **100~400 epochs**: 60 → 21 (gradual decrease)
- **400~500 epochs**: 21 → 24 (minor fluctuations, convergence)

The sawtooth pattern is due to mini-batch training, but the overall trend is decreasing.

### 6.3 Prediction Results (meshgraphnet_result.png)

| Element | Observation | Meaning |
|---------|-------------|---------|
| **Nodes (blue dots)** | 5x4 grid, slight deformation | Mass point positions |
| **Edges (gray lines)** | Connections between nodes | Spring connections |
| **Green arrows** | Actual acceleration | True values computed by Hooke's law |
| **Red arrows** | Predicted acceleration | MGN predictions |
| **Alignment** | Green ≈ Red (nearly identical) | Prediction is accurate |

**Conclusion**: MeshGraphNet accurately predicted the acceleration of the spring-mass system. The green (actual) and red (predicted) arrows are nearly indistinguishable.

### 6.4 Quantitative Evaluation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Relative L2 error | 0.0957 (9.57%) | Good (under 10%) |
| Mean absolute error | 3.42 | Small relative to acceleration range |
| Max absolute error | 21.42 | Large error at some nodes |

---

## 7. PINN vs FNO vs Transolver vs MGN Comparison

| Feature | PINN | FNO | Transolver | MGN |
|---------|------|-----|------------|-----|
| **Learning method** | Equation-based | Data-driven | Data-driven | Data-driven |
| **Data structure** | Points | Grid | Grid/mesh | **Graph** |
| **Core mechanism** | Autograd | Fourier transform | Physics Attention | **Message passing** |
| **Graph topology** | ✗ | ✗ | ✗ | **✓** |
| **Irregular mesh** | ✓ | ✗ | ✓ | ✓ |
| **Variable topology** | N/A | ✗ | ✗ | **✓** |
| **Permutation invariance** | ✗ | ✗ | ✗ | **✓** |
| **Resolution independent** | ✓ | ✓ | Partial | ✓ (variable node count) |
| **Memory** | Low | Medium | High | Medium |
| **Training speed** | Slow | Fast | Medium | Fast |

### When to Use What?

| Situation | Recommended Model | Reason |
|-----------|-------------------|--------|
| No labeled data, only equations | **PINN** | Learn from equations only |
| Regular grid, fast surrogate | **FNO** | Fast and resolution-independent |
| Complex geometry, attention-based | **Transolver** | Physics Attention for complex patterns |
| **Explicit graph/mesh structure** | **MGN** | Directly processes graph topology |
| **Different connections per sample** | **MGN** | Supports variable graph topology |
| CFD mesh, FEM grid | **MGN** | Naturally handles mesh structure |

---

## 8. Parameter Experimentation Guide

### 8.1 Changing Processor Size (Most Important!)

```python
# Change number of message passing blocks
model = MeshGraphNet(
    processor_size=3,    # Few: fast but only short-range info
    processor_size=6,    # Default: balanced
    processor_size=15,   # Original paper: long-range info, slow
    processor_size=30,   # Very deep: longest range, very slow
)
```

### 8.2 Changing Hidden Dimension

```python
# Larger model for better accuracy
model = MeshGraphNet(
    hidden_dim_processor=32,   # Small: fast but less accurate
    hidden_dim_processor=64,   # Default (tutorial)
    hidden_dim_processor=128,  # Original paper default
    hidden_dim_processor=256,  # Large: more accurate but more memory
)
```

### 8.3 Applying to Other Physical Systems

```python
# CFD mesh (3D)
model = MeshGraphNet(
    input_dim_nodes=6,    # x, y, z, vx, vy, vz
    input_dim_edges=3,    # rel_x, rel_y, rel_z
    output_dim=3,         # next time step vx, vy, vz
    processor_size=15,    # Original paper setting
    hidden_dim_processor=128,
)

# Graph extracted from CFD mesh
# edge_index = mesh.connectivity  # mesh connectivity info
```

### 8.4 Changing Aggregation Method

```python
model = MeshGraphNet(
    aggregation="sum",   # Sum (default, similar to physical force summation)
    aggregation="mean",  # Average (independent of node degree)
)
```

---

## 9. Troubleshooting (FAQ)

### Q: "No module named 'torch_geometric'" error

**A**: Install the package:
```cmd
cd E:\physicsnemo_env
Scripts\activate
pip install torch_geometric
```

### Q: "No module named 'torch_scatter'" error

**A**: Install the pre-built wheel:
```cmd
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
```

### Q: DGL-related warnings appear

**A**: Safe to ignore. PhysicsNeMo's MeshGraphNet works with PyTorch Geometric without DGL. DGL was used in previous versions; PyG is now the default.

### Q: CUDA Out of Memory error

**A**: Reduce model size or batch size:
```python
model = MeshGraphNet(
    hidden_dim_processor=32,  # 64 → 32
    processor_size=3,         # 6 → 3
)
batch_size = 20               # 50 → 20
```

### Q: Training doesn't converge

**A**: Try the following:
1. Adjust learning rate: `lr=5e-4` or `lr=2e-3`
2. Increase processor size: `processor_size=10`
3. Increase hidden dimension: `hidden_dim_processor=128`
4. Increase epochs: `EPOCHS=1000`

### Q: Want to use a different graph structure

**A**: Modify the `build_grid_graph()` function or extract directly from a CFD mesh:
```python
# Extract edge_index from CFD mesh
import meshio
mesh = meshio.read("mesh.vtk")
edge_index = extract_edges_from_mesh(mesh)  # custom function
```

### Q: Want to do time-dependent simulation

**A**: Implement rollout:
```python
# Repeat: predict state at t → state at t+1
for t in range(n_timesteps):
    pred = model(node_feat, edge_feat, graph)
    # Use pred to update next time step's velocity/position
    velocities += pred * dt
    positions += velocities * dt
    # Update node/edge features and predict again
```

---

## Summary

Through this tutorial, we learned:

1. **Graph representation**: Representing physical systems as node (mass point) + edge (spring) graphs
2. **Message passing principle**: Edge block + node block repetition to spread information across the graph
3. **Encoder-Processor-Decoder structure**: Raw features → latent space → message passing → output
4. **PyG integration**: Defining graph structure with torch_geometric Data objects
5. **Graph batching**: Combining multiple graphs into one large graph for efficient training
6. **4-model comparison**: Strengths/weaknesses and application scenarios of PINN/FNO/Transolver/MGN
7. **Spring-mass prediction**: Achieved L2 error of 9.57% with 500 epochs of training

### Core Value of MeshGraphNet

- **Graph topology processing**: Directly models physical connection structure
- **Variable graph support**: Can handle different connection structures per sample
- **Permutation invariance**: Predictions independent of node ordering
- **Physical message passing**: Information transfer through edges mimics physical interactions
- **CFD/FEM application**: Directly applicable to real engineering meshes
