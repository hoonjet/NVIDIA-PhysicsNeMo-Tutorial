# GNN — Time Evolution Rollout (Spring-Mass System)

> **Category**: `mesh_based/gnn_rollout/` — Multi-step auto-regressive GNN prediction  
> **Paradigm**: Time evolution on graph with error accumulation analysis  
> **Model**: GNN with encoder-processor-decoder (delta-state prediction)

---

## 1. What Makes This Tutorial Unique?

This is the **ONLY tutorial that performs multi-step auto-regressive rollout on a graph**. It extends the existing MeshGraphNet tutorial from 1-step to N-step prediction:

| Feature | MeshGraphNet (existing) | **This Tutorial** |
|---------|:-----------------------:|:-----------------:|
| **Prediction** | 1-step (pos,vel → accel) | **N-step (state(t) → state(t+1))** |
| **Time evolution** | ✗ | **✓ (auto-regressive)** |
| **Rollout analysis** | ✗ | **✓ (error accumulation)** |
| **Trajectory** | Single snapshot | **Full trajectory (50 steps)** |
| **Error analysis** | L2 at one step | **L2 vs step (growth curve)** |

### Key Concept: Auto-Regressive Rollout
```
state(0) → GNN → state(1) → GNN → state(2) → ... → state(T)
              ↑                ↑
              feed back        feed back
```
Each prediction becomes the input for the next step. Errors compound — this is the fundamental challenge of time evolution with neural networks.

---

## 2. Problem: Spring-Mass Time Evolution

```
System: 5×4 grid of mass points connected by springs
  - 20 nodes, ~62 edges (bidirectional)
  - Spring constant: k = 50.0
  - Damping: c = 0.1
  - Time step: dt = 0.01

Physics: F = -k·(|d| - L₀)·d̂  (Hooke's law)
  Semi-implicit Euler integration:
    v(t+1) = v(t) + a(t)·dt
    x(t+1) = x(t) + v(t+1)·dt

GNN task:
  Input:  state(t) = [x, y, vx, vy] per node
  Output: Δstate = [dx, dy, dvx, dvy] per node
  state(t+1) = state(t) + Δstate
```

---

## 3. Key Concepts

### 3.1 One-Step vs Rollout
- **One-step (teacher-forced)**: Always use true state(t) to predict state(t+1). Error is small and stable.
- **Rollout (auto-regressive)**: Use predicted state(t) to predict state(t+1). Error accumulates.

### 3.2 Error Accumulation
```
Step  1: small error ε₁
Step  2: ε₁ + ε₂(ε₁) — error from step 1 affects step 2
Step  3: ε₁ + ε₂ + ε₃(ε₁, ε₂) — compounding
...
Step 50: potentially large divergence
```

### 3.3 Delta-State Prediction
Instead of predicting the full next state, the GNN predicts the **change** (delta):
```
state(t+1) = state(t) + GNN(state(t))
```
This is more stable than predicting the absolute next state, because:
- The delta is typically small (smooth dynamics)
- The network only needs to learn the "change", not the full state
- Residual learning is easier for the optimizer

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| `[1] System Setup` | Spring-mass grid, trajectory generation (FDM) |
| `[2] GNN Model` | Encoder-processor-decoder, delta-state output |
| `[3] Training` | One-step prediction, 400 epochs |
| `[4] Rollout` | Auto-regressive 50-step, error analysis |
| `[5] Visualization` | Loss, trajectory, error, per-node, explanation |

---

## 5. Key Results

### 5.1 Error Accumulation Curve
- One-step error: low and stable (teacher-forced)
- Rollout error: grows steadily with steps
- Growth rate depends on system damping (dissipation helps stability)

### 5.2 Trajectory Comparison
- Early steps (0-10): GNN rollout closely matches true trajectory
- Middle steps (20-30): visible drift begins
- Late steps (40-50): significant divergence (compounded error)

### 5.3 Per-Node Trajectories
- Individual node paths in x-y plane show the divergence clearly
- True trajectory (blue) vs GNN rollout (red dashed)
- Start point (green) is the same; end points diverge

---

## 6. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\mesh_based\gnn_rollout
python gnn_rollout.py
```

### Result Files
| File | Description |
|------|-------------|
| `results/gnn_rollout_loss.png` | Training & test loss |
| `results/gnn_rollout_trajectory.png` | 6 snapshots: true vs GNN positions |
| `results/gnn_rollout_error.png` | Error accumulation curve |
| `results/gnn_rollout_nodes.png` | Per-node trajectory (x-y plane) |
| `results/gnn_rollout_explanation.png` | Concept explanation |

---

## 7. Key Concepts Learned

1. **Auto-Regressive Rollout**: The core concept of time evolution with neural networks. Each prediction feeds back as input. This is how all time-dependent neural network simulations work (FNO, GNN, etc.).

2. **Error Accumulation**: The fundamental challenge. Unlike one-step prediction (teacher-forced), rollout errors compound. This is why long-term prediction is hard — even a small per-step error grows exponentially.

3. **Delta-State Prediction**: Predicting the change (Δstate) rather than the absolute state is more stable. This is analogous to residual connections in deep learning — the network learns the "residual" or "update".

4. **Teacher-Forcing vs Rollout**: Teacher-forcing (using true states) gives optimistic error estimates. Rollout (using predicted states) reveals the true performance. Always evaluate with rollout for time-dependent problems.

5. **Graph Rollout vs Grid Rollout**: Same concept as Allen-Cahn/Wave tutorials, but on a graph instead of a regular grid. The GNN processes the mesh topology at each step.

6. **Damping Helps Stability**: The spring-mass system has damping (c=0.1), which dissipates energy. This helps the rollout remain stable longer. Conservative systems (no damping, like the wave equation) are harder.

---

## 8. Comparison with Other Tutorials

### vs. MeshGraphNet (existing)
- MGN: 1-step, pos/vel → acceleration
- This: N-step, state → state (integrated)
- MGN: no time evolution
- This: full trajectory rollout with error analysis

### vs. Allen-Cahn (FNO)
- Allen-Cahn: rollout on regular 1D grid
- This: rollout on 2D graph (irregular connectivity)
- Both: same error accumulation concept
- Difference: FNO uses FFT; GNN uses message passing

### vs. Wave Equation (FNO)
- Wave: 2nd-order time (2-channel input)
- This: 1st-order time (1-channel state, delta prediction)
- Wave: conservative (no damping, harder rollout)
- This: dissipative (damping, easier rollout)

---

## 9. Extensions

- **Pushforward trick**: Train with multi-step loss (unroll during training)
- **Noise injection**: Add Gaussian noise to input during training (improves rollout stability)
- **Energy conservation loss**: Add physics constraint to prevent energy drift
- **Longer trajectories**: Increase N_STEPS (watch for divergence)
- **Different systems**: Beam dynamics, CFD mesh, cloth simulation
- **Variable time step**: Predict with variable dt (input dt as feature)
- **3D mesh**: Extend to 3D spring-mass or tetrahedral FEM mesh

---

## 10. References

1. **Pfaff, T., et al.** "Learning Mesh-Based Simulation with Graph Networks." *ICLR 2021.*  
   arXiv: https://arxiv.org/abs/2010.03409  
   *(MeshGraphNet — basis for the GNN architecture used here.)*

2. **Sanchez-Gonzalez, A., et al.** "Learning to Simulate Complex Physics with Graph Networks." *ICML 2020.*  
   *(GNS — introduced the rollout concept for GNN physics simulation.)*

3. **Brandstetter, B., et al.** "Message Passing Neural PDE Solvers." *ICLR 2022.*  
   *(Multi-step rollout and error analysis for GNN PDE solvers.)*

4. **Li, Z., et al.** "Fourier Neural Operator for Parametric PDEs." *ICLR 2021.*  
   arXiv: https://arxiv.org/abs/2010.08895  
   *(FNO — for comparison with grid-based rollout in Allen-Cahn/Wave tutorials.)*
