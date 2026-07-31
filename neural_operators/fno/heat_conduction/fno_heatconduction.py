"""
PhysicsNeMo Tutorial: 2D Heat Conduction with FNO
==================================================
This tutorial uses PhysicsNeMo's FNO (Fourier Neural Operator) to solve a
2D steady-state heat conduction problem — a thermal analysis CAE problem
that is not available in PhysicsNeMo by default.

Problem: 2D Steady-State Heat Conduction with Variable Conductivity
- Input:  Thermal conductivity field k(x,y), heat source Q(x,y), boundary temperatures
- Output: Temperature field T(x,y)

Learning objectives:
1. Heat conduction physics (Fourier's law, Poisson equation)
2. How to use FNO for thermal analysis
3. Generate synthetic thermal data with analytical solutions
4. Compare FNO prediction with exact solution

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_fno_heatconduction.py

=========================================================================
[Physics Background: Heat Conduction]
=========================================================================

Heat conduction is the transfer of thermal energy through a material without
bulk fluid motion. The governing equation depends on the regime:

--- Steady-State Heat Conduction (Poisson Equation) ---

    ∇·(k ∇T) + Q = 0

where:
    T = T(x,y)  = temperature field (unknown)
    k = k(x,y)  = thermal conductivity (material property, W/m·K)
    Q = Q(x,y)  = volumetric heat source (W/m³)

For constant conductivity (k = const), this simplifies to:
    k ∇²T + Q = 0  →  ∇²T = -Q/k  (Poisson equation)

--- Fourier's Law of Heat Conduction ---

    q = -k ∇T

where:
    q = (q_x, q_y) = heat flux vector (W/m²)
    q_x = -k ∂T/∂x  (heat flows from hot to cold)
    q_y = -k ∂T/∂y

The negative sign means heat flows DOWN the temperature gradient
(from high T to low T).

--- Boundary Conditions ---

    Dirichlet: T = T_boundary  (prescribed temperature)
    Neumann:   -k ∂T/∂n = q_boundary  (prescribed heat flux)
    Robin:     -k ∂T/∂n = h(T - T_ambient)  (convection)

=========================================================================
[Problem Setup: Heat Sink with Non-Uniform Source]
=========================================================================

    T_hot = 100°C (left boundary)
    +----------------------+
    |                      |  T_cold = 0°C (right boundary)
    |   Q(x,y) heat source |
    |   k(x,y) conductivity|
    |                      |  Insulated (top/bottom)
    +----------------------+

    Domain: [0, 1] × [0, 1]
    Left: T = 100 (hot)
    Right: T = 0 (cold)
    Top/Bottom: Insulated (q = 0)

    Variable conductivity: k(x,y) = k_base + k_var * pattern(x,y)
    Heat source: Q(x,y) = Q0 * exp(-r²/σ²)  (Gaussian source)

    Analytical solution (for constant k, no source):
    T(x,y) = T_hot * (1 - x/L)  (linear profile)
"""

# ============================================================================
# Library Imports
# ============================================================================
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# PhysicsNeMo: FNO model
from physicsnemo.models.fno.fno import FNO

# ============================================================================
# [0] Environment Setup
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")

# ============================================================================
# [1] Generate Synthetic Heat Conduction Data
# ============================================================================
print("\n[1] Generating synthetic heat conduction data...")

def generate_heat_data(n_samples=300, resolution=32):
    """
    Generate synthetic steady-state heat conduction data.

    For each sample:
    - Random thermal conductivity field k(x,y)
    - Random heat source Q(x,y)
    - Solve ∇·(k∇T) + Q = 0 with finite differences
    - Boundary: T=100 at x=0, T=0 at x=1, insulated top/bottom

    Parameters
    ----------
    n_samples : int
        Number of samples
    resolution : int
        Grid resolution

    Returns
    -------
    inputs : np.ndarray, shape (n_samples, 2, res, res)
        [conductivity k, heat source Q]
    outputs : np.ndarray, shape (n_samples, 1, res, res)
        [temperature T]
    """
    res = resolution
    inputs = np.zeros((n_samples, 2, res, res), dtype=np.float32)
    outputs = np.zeros((n_samples, 1, res, res), dtype=np.float32)

    dx = 1.0 / (res - 1)
    dy = 1.0 / (res - 1)

    for s in range(n_samples):
        # --- Random conductivity field ---
        # Base conductivity + spatial variation
        k_base = np.random.uniform(0.5, 2.0)
        # Add smooth random variation using low-frequency sinusoids
        x_coords = np.linspace(0, 1, res)
        y_coords = np.linspace(0, 1, res)
        X, Y = np.meshgrid(x_coords, y_coords)

        k_var = np.random.uniform(0.1, 0.5)
        freq_x = np.random.randint(1, 4)
        freq_y = np.random.randint(1, 4)
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)

        k_field = k_base + k_var * (np.sin(freq_x * np.pi * X + phase_x) *
                                      np.cos(freq_y * np.pi * Y + phase_y))
        k_field = np.clip(k_field, 0.1, 5.0)

        # --- Random heat source (Gaussian) ---
        Q0 = np.random.uniform(0, 50)
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        sigma = np.random.uniform(0.1, 0.3)
        Q_field = Q0 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

        inputs[s, 0] = k_field
        inputs[s, 1] = Q_field

        # --- Solve ∇·(k∇T) + Q = 0 with finite differences ---
        # Using Gauss-Seidel iteration
        T = np.zeros((res, res), dtype=np.float32)

        # Boundary conditions
        T[:, 0] = 100.0   # left: hot
        T[:, -1] = 0.0    # right: cold
        # Top and bottom: insulated (handled in FD stencil)

        # Iterative solver
        for iteration in range(2000):
            T_old = T.copy()
            for i in range(1, res - 1):
                for j in range(1, res - 1):
                    # ∇·(k∇T) using central differences
                    # d/dx(k * dT/dx) ≈ (k[i+1,j]*(T[i+1,j]-T[i,j]) - k[i-1,j]*(T[i,j]-T[i-1,j])) / dx^2
                    k_e = 0.5 * (k_field[j, i] + k_field[j, i+1])
                    k_w = 0.5 * (k_field[j, i-1] + k_field[j, i])
                    k_n = 0.5 * (k_field[j, i] + k_field[j+1, i])
                    k_s = 0.5 * (k_field[j-1, i] + k_field[j, i])

                    flux_x = k_e * (T[j, i+1] - T[j, i]) - k_w * (T[j, i] - T[j, i-1])
                    flux_y = k_n * (T[j+1, i] - T[j, i]) - k_s * (T[j, i] - T[j-1, i])

                    T[j, i] = (flux_x / dx**2 + flux_y / dy**2 + Q_field[j, i]) / \
                              ((k_e + k_w) / dx**2 + (k_n + k_s) / dy**2)

            # Insulated BC (top/bottom): dT/dn = 0
            T[0, :] = T[1, :]
            T[-1, :] = T[-2, :]

            # Check convergence
            if np.max(np.abs(T - T_old)) < 1e-4:
                break

        outputs[s, 0] = T

    return inputs, outputs


# Generate data (with progress indication)
print("   Solving finite difference heat equation for each sample...")
print("   (This may take a minute...)")
inputs, outputs = generate_heat_data(n_samples=200, resolution=32)
print(f"   Inputs shape: {inputs.shape}  [k, Q]")
print(f"   Outputs shape: {outputs.shape}  [T]")
print(f"   Temperature range: [{outputs.min():.1f}, {outputs.max():.1f}]")

# ============================================================================
# [2] Create FNO Model
# ============================================================================
print("\n[2] Creating FNO model...")

# FNO (Fourier Neural Operator) learns the operator mapping:
#   (k(x,y), Q(x,y)) → T(x,y)
#
# Key idea: Instead of learning point-to-point mapping, FNO learns in
# Fourier (frequency) space, capturing global spatial patterns efficiently.
#
# Architecture:
#   1. Lift: input (2 channels) → hidden (width channels)
#   2. Fourier layers: mix in frequency space (retain 'modes' frequencies)
#   3. Project: hidden → output (1 channel)

model = FNO(
    in_channels=2,      # [conductivity k, heat source Q]
    out_channels=1,     # [temperature T]
    spatial_dim=2,      # 2D problem
    n_modes=(8, 8),    # Fourier modes kept in each dimension
    hidden_channels=32, # width of hidden layers
    n_layers=4,         # number of Fourier layers
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"   FNO parameters: {n_params:,}")
print(f"   Architecture: 2ch → 32ch → [4 Fourier layers, 8 modes] → 1ch")

# ============================================================================
# [3] Training Setup
# ============================================================================
print("\n[3] Training setup...")

# Convert to tensors
inputs_tensor = torch.from_numpy(inputs).to(device)
outputs_tensor = torch.from_numpy(outputs).to(device)

# Train/test split
n_train = 160
n_test = 40
train_inputs = inputs_tensor[:n_train]
train_outputs = outputs_tensor[:n_train]
test_inputs = inputs_tensor[n_train:n_train + n_test]
test_outputs = outputs_tensor[n_train:n_train + n_test]

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

n_epochs = 100
batch_size = 16

print(f"   Train samples: {n_train}")
print(f"   Test samples: {n_test}")
print(f"   Epochs: {n_epochs}")
print(f"   Batch size: {batch_size}")

# ============================================================================
# [4] Training Loop
# ============================================================================
print("\n[4] Training FNO...")
print("    FNO learns the operator: (k, Q) → T")
print("    By mixing in Fourier space, it captures global heat flow patterns\n")

loss_history = []
test_loss_history = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    perm = torch.randperm(n_train)

    for batch_idx in range(0, n_train, batch_size):
        idx = perm[batch_idx:batch_idx + batch_size]
        x_batch = train_inputs[idx]
        y_batch = train_outputs[idx]

        pred = model(x_batch)
        loss = torch.nn.functional.mse_loss(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_pred = model(test_inputs)
        test_loss = torch.nn.functional.mse_loss(test_pred, test_outputs).item()
        test_loss_history.append(test_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1:4d}/{n_epochs} | Train: {avg_loss:.6f} | Test: {test_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

print("\n   Training complete!")

# ============================================================================
# [5] Evaluation and Visualization
# ============================================================================
print("\n[5] Evaluating and visualizing...")

model.eval()
with torch.no_grad():
    test_pred = model(test_inputs)

# Select 4 test samples for visualization
n_vis = 4
fig, axes = plt.subplots(4, n_vis, figsize=(4 * n_vis, 16))

for i in range(n_vis):
    # Row 1: Conductivity field
    ax = axes[0, i]
    k = test_inputs[i, 0].cpu().numpy()
    im = ax.imshow(k, cmap='hot', vmin=k.min(), vmax=k.max())
    ax.set_title(f'Conductivity k(x,y)\n(Sample {i+1})', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Heat source
    ax = axes[1, i]
    Q = test_inputs[i, 1].cpu().numpy()
    im = ax.imshow(Q, cmap='YlOrRd', vmin=0, vmax=Q.max())
    ax.set_title(f'Heat Source Q(x,y)', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 3: Ground truth temperature
    ax = axes[2, i]
    T_gt = test_outputs[i, 0].cpu().numpy()
    im = ax.imshow(T_gt, cmap='jet', vmin=0, vmax=100)
    ax.set_title(f'Ground Truth T(x,y)\n(Finite Difference)', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046, label='T (°C)')

    # Row 4: FNO prediction
    ax = axes[3, i]
    T_pred = test_pred[i, 0].cpu().numpy()
    im = ax.imshow(T_pred, cmap='jet', vmin=0, vmax=100)
    error = np.mean(np.abs(T_pred - T_gt))
    ax.set_title(f'FNO Prediction T(x,y)\n(MAE: {error:.2f}°C)', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046, label='T (°C)')

plt.suptitle('FNO: 2D Steady-State Heat Conduction\n'
             '(Thermal Analysis — Not Available in PhysicsNeMo by Default)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_fno_heatconduction.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_fno_heatconduction.png")

# --- Figure 2: Training Loss ---
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(loss_history, 'b-', label='Train Loss', linewidth=1.5)
ax.plot(test_loss_history, 'r-', label='Test Loss', linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('FNO Training: Heat Conduction', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_fno_heatconduction_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_fno_heatconduction_loss.png")

# --- Figure 3: Heat Flux Visualization ---
print("\n   Computing heat flux (Fourier's law: q = -k∇T)...")

# Compute heat flux for one sample
sample_idx = 0
k_sample = test_inputs[sample_idx, 0].cpu().numpy()
T_pred_sample = test_pred[sample_idx, 0].cpu().numpy()
T_gt_sample = test_outputs[sample_idx, 0].cpu().numpy()

# Gradients (central differences)
dT_dx = np.gradient(T_pred_sample, axis=1)
dT_dy = np.gradient(T_pred_sample, axis=0)

# Heat flux: q = -k * ∇T
qx = -k_sample * dT_dx
qy = -k_sample * dT_dy

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Temperature
ax = axes[0]
im = ax.imshow(T_pred_sample, cmap='jet', vmin=0, vmax=100)
ax.set_title('Temperature T(x,y)', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='T (°C)')

# Heat flux magnitude
ax = axes[1]
q_mag = np.sqrt(qx**2 + qy**2)
im = ax.imshow(q_mag, cmap='YlOrRd')
ax.set_title('Heat Flux Magnitude |q|', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='|q| (W/m²)')

# Heat flux vectors
ax = axes[2]
ax.imshow(T_pred_sample, cmap='jet', vmin=0, vmax=100, alpha=0.5)
# Subsample for clarity
step = 3
X, Y = np.meshgrid(np.arange(0, 32, step), np.arange(0, 32, step))
ax.quiver(X, Y, qx[::step, ::step], qy[::step, ::step],
           color='white', scale=500, width=0.005)
ax.set_title('Heat Flux Vectors q = -k∇T', fontsize=13, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.suptitle('Heat Flux Analysis (Fourier\'s Law)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_fno_heatconduction_flux.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_fno_heatconduction_flux.png")

# ============================================================================
# [6] Summary
# ============================================================================
print("\n" + "=" * 70)
print("Tutorial Summary: 2D Heat Conduction with FNO")
print("=" * 70)
print(f"""
Physics:
  - Steady-state heat conduction: ∇·(k∇T) + Q = 0
  - Fourier's law: q = -k∇T (heat flows down temperature gradient)
  - Variable conductivity k(x,y) and heat source Q(x,y)
  - BCs: Dirichlet (T=100 left, T=0 right), Neumann (insulated top/bottom)

Model:
  - FNO (Fourier Neural Operator): (k, Q) → T
  - {n_params:,} parameters
  - 4 Fourier layers, 8 modes per dimension

Training:
  - Epochs: {n_epochs}
  - Final train loss: {loss_history[-1]:.6f}
  - Final test loss: {test_loss_history[-1]:.6f}

Key Difference from CFD Tutorials:
  - CFD: Navier-Stokes (momentum + continuity)
  - Thermal: Poisson equation (elliptic, no time)
  - Input: material property (k) + source (Q) → operator learning
  - Output: scalar field (temperature), not vector field (velocity)

Validation:
  - Ground truth: finite difference solver (2000 iterations)
  - FNO predicts temperature in single forward pass
  - MAE on test set: {np.mean([np.mean(np.abs(test_pred[i,0].cpu().numpy() - test_outputs[i,0].cpu().numpy())) for i in range(n_test)]):.2f}°C
""")
print("=" * 70)
