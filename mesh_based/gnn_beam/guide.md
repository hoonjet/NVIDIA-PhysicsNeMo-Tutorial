# GNN — Beam Structural Analysis (FEM Mesh)

> **Category**: `mesh_based/gnn_beam/` — Structural analysis on FEM mesh  
> **Paradigm**: Data-driven GNN for displacement field prediction  
> **Model**: Simplified MeshGraphNet (encoder-processor-decoder)

---

## 1. What Makes This Tutorial Unique?

This is the **ONLY tutorial that uses a FEM mesh for structural analysis**. It bridges the gap between traditional FEM and machine learning:

| Feature | MeshGraphNet | PINN Plane Stress | **This Tutorial** |
|---------|:-----------:|:-----------------:|:-----------------:|
| **Mesh type** | Spring-mass grid | Regular grid | **FEM triangular mesh** |
| **Problem** | Spring-mass dynamics | Plane stress | **Cantilever beam** |
| **Input** | Position, velocity | Coordinates | **Load, BC, mesh** |
| **Output** | Acceleration | Stress field | **Displacement field** |
| **Ground truth** | Hooke's law | PDE residual | **Euler-Bernoulli theory** |
| **Mesh topology** | Grid graph | None | **FEM connectivity** |

### Key Differentiators
1. **FEM mesh**: Triangular elements (Delaunay) → graph (nodes + edges)
2. **Structural analysis**: Load → displacement (not acceleration → velocity)
3. **Boundary conditions as features**: `is_fixed` node feature lets GNN learn BCs
4. **Analytical validation**: Euler-Bernoulli beam theory as ground truth

---

## 2. Problem: Cantilever Beam

```
Beam: L × H = 4.0 × 0.5
Fixed: x = 0 (left boundary)
Load: Random tip load at x = L (right boundary)

Euler-Bernoulli beam theory:
  δ_y(x) = P·x²·(3L - x) / (6·E·I)     (deflection)
  δ_x(x,y) = -P·y·(6Lx - 3x²) / (6·E·I)  (axial)
  δ_tip = P·L³ / (3·E·I)                  (tip deflection)

where I = H³/12 (second moment of area)
```

### 2.1 Physical Setup

A cantilever beam is fixed at one end and free at the other. When a load is applied at the free end, the beam deflects. The amount of deflection depends on:
- **Load magnitude** (P)
- **Beam length** (L)
- **Material stiffness** (E, Young's modulus)
- **Cross-section** (I, second moment of area)

### 2.2 FEM Mesh

The beam is discretized into triangular elements using Delaunay triangulation:
- **Nodes**: Mass points with positions (x, y)
- **Elements**: Triangles connecting 3 nodes
- **Edges**: Graph connectivity from element adjacency

This is exactly how a real FEM solver represents the geometry — the GNN operates on this same structure.

---

## 3. GNN Architecture

### 3.1 Node Features
```
[x, y, is_fixed, load_x, load_y]
```
- `x, y`: Node position
- `is_fixed`: 1.0 if node is on the fixed boundary, 0.0 otherwise
- `load_x, load_y`: Applied load at this node (nonzero only at tip)

### 3.2 Edge Features
```
[rel_x, rel_y, distance]
```
- `rel_x, rel_y`: Relative position (dst - src)
- `distance`: Euclidean distance between connected nodes

### 3.3 Architecture: Encoder-Processor-Decoder

```
Node Encoder: [5] → [64] (MLP)
Edge Encoder: [3] → [64] (MLP)
    ↓
Processor (6 blocks):
  Edge Block: concat(edge, src_node, dst_node) → [192] → [64]
  Node Block: aggregate(edges) + node → [128] → [64]
    ↓
Decoder: [64] → [64] → [2] (ux, uy)
```

### 3.4 Message Passing

Each processor block performs:
1. **Edge update**: Each edge receives information from its source and destination nodes
2. **Node update**: Each node aggregates messages from all its edges

After 6 blocks, information has propagated 6 hops through the mesh — enough for the GNN to "feel" the boundary conditions and load distribution.

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| `[1] Mesh Generation` | Delaunay triangulation → nodes, elements, edges |
| `[2] Data Generation` | Euler-Bernoulli beam theory → displacement field |
| `[3] GNN Model` | MeshGraphNetSimple (encoder-processor-decoder) |
| `[4] Training` | 500 epochs, batch size 20, cosine LR |
| `[5] Evaluation` | L2 error, tip deflection vs analytical |
| `[6] Visualization` | Loss, deformation, displacement field, explanation |

---

## 5. Key Results

### 5.1 Deformation Visualization
- Original mesh (gray) vs true deformed (blue) vs GNN predicted (red)
- GNN accurately captures beam bending under various loads
- Deformation magnified ×50 for visibility

### 5.2 Tip Deflection Accuracy
- GNN tip deflection closely matches analytical Euler-Bernoulli solution
- Scatter plot shows 1:1 correlation between analytical and GNN predictions

### 5.3 L2 Error
- Mean L2 error: low (single-digit %)
- Error distribution is tight (consistent across different loads)

---

## 6. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\mesh_based\gnn_beam
python gnn_beam.py
```

### Result Files
| File | Description |
|------|-------------|
| `results/gnn_beam_loss.png` | Training & test loss |
| `results/gnn_beam_deformation.png` | 6 samples: original vs true vs GNN deformed |
| `results/gnn_beam_result.png` | Displacement field, tip deflection, error, mesh |
| `results/gnn_beam_explanation.png` | Problem setup and architecture explanation |

---

## 7. Key Concepts Learned

1. **FEM Mesh → Graph**: Triangular elements are converted to graph edges. Each element's 3 edges become bidirectional graph connections. This is the standard FEM-to-GNN conversion.

2. **BC as Node Features**: Instead of hard-coding boundary conditions (like in FEM), the GNN learns them. `is_fixed=1.0` tells the GNN "this node is fixed" — the GNN learns to predict zero displacement at fixed nodes.

3. **Load as Node Features**: The applied load is encoded as a node feature at the tip nodes. The GNN learns to propagate this load information through the mesh to predict the displacement field.

4. **Message Passing = Information Flow**: 6 processor blocks allow information to travel 6 hops. The load at the tip (right side) must "reach" the fixed boundary (left side) for the GNN to understand the beam's response. 6 hops is sufficient for this mesh size.

5. **Analytical Validation**: Euler-Bernoulli beam theory provides an exact analytical solution. This allows precise validation of GNN predictions — not just "looks right" but quantitatively correct.

6. **Surrogate Modeling**: Once trained, the GNN predicts displacement fields in milliseconds (vs seconds/minutes for FEM). This enables rapid design iteration: change load → instant prediction.

---

## 8. Comparison with Other Tutorials

### vs. MeshGraphNet (spring-mass)
- MGN: predicts acceleration from velocity (dynamics, 1-step)
- This: predicts displacement from load (statics, no time)
- MGN: regular grid graph
- This: FEM triangular mesh (Delaunay)

### vs. PINN Plane Stress
- PINN: equation-based (PDE residual loss), MLP, no mesh
- This: data-driven (supervised), GNN, FEM mesh
- PINN: boundary conditions hard-coded in loss function
- This: boundary conditions as node features (learned)

### vs. NACA Airfoil
- NACA: uses PINN + FNO (not GNN!) despite being in mesh_based/
- This: uses GNN on actual FEM mesh
- NACA: fluid dynamics (potential flow)
- This: structural mechanics (beam bending)

---

## 9. Extensions

- **Complex geometry**: L-shaped beam, beam with hole, bracket
- **3D mesh**: Tetrahedral elements (requires more memory)
- **Material nonlinearity**: Plastic deformation, hyperelasticity
- **Multi-load cases**: Train on multiple load patterns simultaneously
- **Mesh generalization**: Train on one mesh, test on different mesh topology
- **Stress prediction**: Extend output to include stress/strain fields
- **Dynamic analysis**: Time-dependent loading (combine with rollout tutorial)

---

## 10. References

1. **Pfaff, T., et al.** "Learning Mesh-Based Simulation with Graph Networks." *ICLR 2021.*  
   arXiv: https://arxiv.org/abs/2010.03409  
   *(MeshGraphNet original paper — basis for GNN-based mesh simulation.)*

2. **Sanchez-Gonzalez, A., et al.** "Learning to Simulate Complex Physics with Graph Networks." *ICML 2020.*  
   *(GNS architecture — predecessor to MeshGraphNet.)*

3. **Timoshenko, S. & Goodier, J.N.** "Theory of Elasticity." *McGraw-Hill, 1951.*  
   *(Euler-Bernoulli beam theory — analytical reference used in this tutorial.)*

4. **Li, Z., et al.** "Fourier Neural Operator for Parametric PDEs." *ICLR 2021.*  
   arXiv: https://arxiv.org/abs/2010.08895  
   *(FNO paper — for comparison with grid-based methods.)*
