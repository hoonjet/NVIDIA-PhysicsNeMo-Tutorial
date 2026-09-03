# Mesh-Based Learning Tutorials

> Irregular mesh / complex geometry processing — graph neural networks and mesh-based learning

---

## Overview

Mesh-based learning directly handles **irregular meshes** (CFD grids, FEM meshes) rather than regular grids. It can process complex geometries used in real industrial simulations.

---

## Tutorials

| # | Tutorial | Key Mechanism | Script |
|---|----------|---------------|--------|
| 1 | [MeshGraphNet](meshgraphnet/) | Graph neural network (message passing) | `meshgraphnet.py` |
| 2 | [NACA Airfoil](naca_airfoil/) | Aerodynamic analysis (flow field prediction) | `naca_airfoil.py` |
| 3 | [GNN Beam](gnn_beam/) | Structural analysis on FEM mesh (load → displacement) | `gnn_beam.py` |
| 4 | [GNN Rollout](gnn_rollout/) | Multi-step time evolution (auto-regressive rollout) | `gnn_rollout.py` |
| 5 | [SPH GNN](sph_gnn/) | Lagrangian particle simulation (dam-break, dynamic graph) | `sph_gnn.py` |

---

## Recommended Learning Order

1. **MeshGraphNet** — Graph neural network basics (mesh structure learning)
2. **NACA Airfoil** — Applied aerodynamics (flow field prediction)
3. **GNN Beam** — Structural analysis on FEM mesh (load → displacement, Euler-Bernoulli validation)
4. **GNN Rollout** — Time evolution with error accumulation (1-step → N-step rollout)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Data structure** | Graph (nodes + edges) |
| **Core mechanism** | Message passing (aggregate neighbor node info) |
| **Irregular mesh** | ✓ (area FNO cannot handle) |
| **Complex geometry** | ✓ (real CFD/FEM meshes) |
| **Resolution flexibility** | ✓ (variable node count) |
| **Multi-scale** | Hop-based |
