# SPH (Smoothed Particle Hydrodynamics) + GNN

> **Category**: `mesh_based/` — Lagrangian particle simulation
> **Paradigm**: GNN with dynamic graph (moving particles)
> **Problem**: 2D Dam-Break (free surface flow)

---

## 1. What Makes This Tutorial Unique?

| Aspect | GNN Beam (existing) | GNN Rollout (existing) | **SPH GNN (THIS)** |
|--------|---------------------|------------------------|---------------------|
| **Framework** | Eulerian (fixed mesh) | Eulerian (fixed mesh) | **Lagrangian (moving particles)** |
| **Graph** | Static | Static | **Dynamic (rebuilt every step)** |
| **Nodes** | Fixed positions | Fixed positions | **Move with fluid** |
| **Physics** | Structural (elastic) | Spring-mass | **Fluid (Navier-Stokes via SPH)** |
| **Free surface** | ✗ | ✗ | **✓ (particles separate)** |
| **Mass transport** | ✗ | ✗ | **✓ (particles carry mass)** |

### Key Difference: Eulerian vs Lagrangian

```
Eulerian (existing GNN tutorials):          Lagrangian (THIS tutorial):
┌─────────────────────┐                    •  •  •  •
│  o---o---o---o     │                   •  •  •  •  •
│  |   |   |   |     │     →             •  •  •  •
│  o---o---o---o     │                   •  •  •
│  |   |   |   |     │                    •  •
│  o---o---o---o     │
└─────────────────────┘
Nodes stay fixed                              Particles flow with fluid
Graph never changes                           Graph rebuilt every step
```

---

## 2. SPH Physics

### Core Idea
SPH represents fluid as discrete particles. Each particle carries mass, position, velocity, and density. Physical quantities are smoothed via kernel functions.

### Key Equations

**Density** (sum over neighbors):
```
ρ_i = Σ_j m_j W(|x_i - x_j|, h)
```

**Pressure** (equation of state):
```
p = c² (ρ - ρ₀)
```

**Acceleration** (pressure + viscosity + gravity):
```
a_i = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij + ν ∇²v + g
```

### SPH Kernels

| Kernel | Use | Formula |
|--------|-----|---------|
| Poly6 | Density | W(r,h) = (315/64πh⁹)(h²-r²)³ |
| Spiky | Pressure gradient | ∇W = (-45/πh⁶)(h-r)²(r̂) |
| Viscosity | Viscous force | W(r,h) = (15/2πh³)(...) |

---

## 3. GNN Architecture

### Input → Output
```
Node features: (x, y, vx, vy)  →  4 dims
Edge features: (dx, dy)         →  2 dims
                                    ↓
                          ┌─────────────────┐
                          │  Node Encoder    │
                          │  Edge Encoder    │
                          │  3× Message Pass │
                          │  Decoder         │
                          └─────────────────┘
                                    ↓
Output: (ax, ay)  →  2 dims (acceleration)
```

### Message Passing (3 rounds)
1. **Encode**: node/edge features → latent (64-dim)
2. **Message passing** (×3):
   - For each edge: message = MLP(sender_node, edge_feature)
   - Aggregate: sum messages to receiver
   - Update: node = node + MLP(node, aggregated_messages)
3. **Decode**: latent → acceleration

### Dynamic Graph Construction
- At each time step, compute pairwise distances between all particles
- Connect particles within smoothing length h
- Edge features = relative position (dx, dy)
- **Graph topology changes every step** (particles move)

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| [1] Parameters | Domain, SPH constants, dam-break geometry |
| [2] Kernels | Poly6, Spiky gradient, Viscosity |
| [3] SPH Solver | Reference solver (density, pressure, force, integration) |
| [4] GNN Model | Encoder → 3× MP → Decoder (latent=64) |
| [5] Training | 60 epochs, 8 traj/batch, random step sampling |
| [6] Rollout | Auto-regressive prediction (100 steps) |
| [7] Visualization | Snapshots, rollout error, loss, concept, dynamic graph |
| [8] Summary | Position error, key differences |

---

## 5. How to Run

```cmd
cd E:\physicsnemo-tutorials\mesh_based\sph_gnn
python sph_gnn.py
```

> **Note**: SPH solver is Python-based (no GPU acceleration for data generation). Data generation takes ~5-10 minutes. Training is GPU-accelerated.

Results saved to `results/`:
- `sph_dam_break.png` — SPH solver vs GNN at 4 time snapshots
- `sph_rollout_error.png` — Auto-regressive rollout position error
- `sph_loss.png` — Training loss
- `sph_concept.png` — Eulerian vs Lagrangian comparison
- `sph_dynamic_graph.png` — Graph topology at different times

---

## 6. vs. Existing GNN Tutorials

| Feature | GNN Beam | GNN Rollout | **SPH GNN** |
|---------|----------|-------------|-------------|
| **Framework** | Eulerian | Eulerian | **Lagrangian** |
| **Graph** | Static | Static | **Dynamic** |
| **Node positions** | Fixed | Fixed | **Moving** |
| **Physics** | Linear elasticity | Spring-mass | **Fluid (SPH)** |
| **Free surface** | ✗ | ✗ | **✓** |
| **BC** | Fixed nodes | Fixed nodes | **Wall bounce-back** |
| **Integration** | N/A | Euler | **Semi-implicit Euler** |
| **Rollout** | ✗ | ✓ (spring) | **✓ (fluid)** |

---

## 7. Why Lagrangian Matters

1. **Free surface tracking**: Particles naturally track interfaces (no level-set needed)
2. **Large deformation**: No mesh tangling (particles move freely)
3. **Multiphase**: Different particle types for different fluids
4. **Moving boundaries**: Particles interact with moving objects naturally
5. **Mass conservation**: Each particle carries fixed mass (exact conservation)

---

## 8. References

- Monaghan, "Smoothed Particle Hydrodynamics" (1992)
- Gingold & Monaghan, "Smoothed particle hydrodynamics: theory and application" (1977)
- Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph Networks" (2020)
- Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks" (2021)
