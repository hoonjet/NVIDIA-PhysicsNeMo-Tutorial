# Mesh-Based Learning Tutorials

> Irregular mesh / complex geometry processing — graph neural networks and mesh-based learning

---

## Overview

This folder contains tutorials for learning on irregular meshes and complex geometries. Unlike grid-based methods (FNO, U-Net), these approaches handle unstructured meshes commonly used in industrial CFD/FEA simulations.

---

## Tutorials

| # | Tutorial | Description | Script |
|---|----------|-------------|--------|
| 1 | [MeshGraphNet](meshgraphnet/) | Graph neural network for mesh-based learning | `meshgraphnet.py` |
| 2 | [NACA Airfoil](naca_airfoil/) | Aerodynamic analysis (flow field prediction) | `naca_airfoil.py` |

---

## Recommended Learning Order

1. **MeshGraphNet** — Graph neural network basics (mesh → graph conversion, message passing)
2. **NACA Airfoil** — Applied aerodynamics (airfoil geometry → flow field)

---

## Key Features

| Feature | MeshGraphNet | NACA Airfoil |
|---------|-------------|-------------|
| **Input** | Mesh graph | Airfoil coordinates |
| **Output** | Node features (velocity, pressure) | Flow field (u, v, p) |
| **Architecture** | Message Passing GNN | MLP + feature transform |
| **Irregular mesh** | ✓ | ✓ |
| **Resolution independence** | ✓ | ✓ |
| **Training data** | Required (mesh + labels) | Required (geometry + flow) |
| **Inference speed** | Fast (after training) | Fast (after training) |
