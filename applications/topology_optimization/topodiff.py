"""
PhysicsNeMo Tutorial: Topology Optimization with TopoDiff (Diffusion Model)
============================================================================
This tutorial uses PhysicsNeMo's TopoDiff model to perform topology optimization
for structural design. TopoDiff is a diffusion-based generative model that
generates optimal material distributions given boundary conditions and loads.

Problem: 2D Cantilever Beam Topology Optimization
- Input:  Boundary conditions, load conditions (encoded as images)
- Output: Optimal material distribution (0 = void, 1 = solid)

Learning objectives:
1. Understand topology optimization fundamentals
2. How to use PhysicsNeMo's TopoDiff diffusion model
3. Generate synthetic training data for structural optimization
4. Train and sample from a diffusion model for design generation

How to run:
  cd E:\\physicsnemo_env
  Scripts\\activate
  python tutorial_topodiff.py

=========================================================================
[Physics Background: Topology Optimization]
=========================================================================

Topology optimization finds the best material layout within a design domain
that minimizes compliance (maximizes stiffness) subject to a volume constraint.

The classical formulation (SIMP - Solid Isotropic Material with Penalization):

    min   C(x) = U^T K(x) U        (compliance = strain energy)
    x     subject to:
          K(x) U = F               (equilibrium: stiffness * displacement = force)
          sum(x) / V <= V_frac     (volume fraction constraint)
          0 <= x_e <= 1            (density variables, 0=void, 1=solid)

where:
    x  = density field (design variable, per element)
    K  = global stiffness matrix (depends on density via SIMP penalization)
    U  = displacement vector
    F  = external force vector
    V  = total domain volume

SIMP penalization: E_e = x_e^p * E_0
    - p = penalization exponent (typically 3.0)
    - Forces intermediate densities toward 0 or 1 (black-and-white design)

Traditional solvers (SIMP, level-set, MMA) require hundreds of FEM iterations.
TopoDiff replaces this iterative process with a single forward pass of a
diffusion model, generating near-optimal designs instantly.

=========================================================================
[What is a Diffusion Model?]
=========================================================================

A diffusion model learns to generate data by:
1. Forward process: gradually add Gaussian noise to data (training data)
   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
2. Reverse process: learn to denoise (remove noise step by step)
   The neural network predicts the noise added at each step

For topology optimization:
- x_0 = optimal topology (material distribution)
- conditions = boundary conditions + loads (encoded as images)
- The model learns: given noisy topology + conditions, predict the noise

=========================================================================
[Problem Setup: Cantilever Beam]
=========================================================================

    Fixed wall          Free end (load applied here)
    +---------+         v (downward force)
    |         |
    |         |
    +---------+

    - Left edge is fixed (Dirichlet BC)
    - A downward point load is applied at the right edge
    - Design domain: 32x32 grid
    - Volume fraction: 50% (half of domain is material)
    - Goal: Find stiffest structure using only 50% of material
"""

# ============================================================================
# Library Imports
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# PhysicsNeMo TopoDiff model
from physicsnemo.models.topodiff.topodiff import TopoDiff
from physicsnemo.models.topodiff.diffusion import Diffusion

# ============================================================================
# [0] Environment Setup
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")

# ============================================================================
# [1] Generate Synthetic Topology Optimization Data
# ============================================================================
# Since we don't have pre-computed FEM optimization results, we generate
# synthetic data that mimics optimal topologies for cantilever beams.
# In production, this data would come from running SIMP/MMA optimization.

def generate_cantilever_topologies(n_samples=200, resolution=32):
    """
    Generate synthetic cantilever beam topologies.

    A real cantilever beam optimal topology typically looks like a truss
    structure connecting the fixed wall to the load point. We approximate
    this with parametric truss-like patterns.

    Parameters
    ----------
    n_samples : int
        Number of topology samples to generate
    resolution : int
        Grid resolution (resolution x resolution)

    Returns
    -------
    topologies : np.ndarray, shape (n_samples, 1, resolution, resolution)
        Binary material distributions (0=void, 1=solid)
    conditions : np.ndarray, shape (n_samples, 3, resolution, resolution)
        Boundary conditions: [fixed_mask, load_x, load_y]
    """
    topologies = np.zeros((n_samples, 1, resolution, resolution), dtype=np.float32)
    conditions = np.zeros((n_samples, 3, resolution, resolution), dtype=np.float32)

    for i in range(n_samples):
        # --- Boundary conditions ---
        # Left edge is fixed (Dirichlet BC)
        conditions[i, 0, :, 0:2] = 1.0  # fixed_mask

        # Load applied at right edge, varying vertical position
        load_y = np.random.randint(resolution // 4, 3 * resolution // 4)
        conditions[i, 1, load_y, -1] = 1.0   # load_x (horizontal, 0 here)
        conditions[i, 2, load_y, -1] = -1.0  # load_y (downward force)

        # --- Generate truss-like topology ---
        topo = np.zeros((resolution, resolution), dtype=np.float32)

        # Number of truss members (randomized for variety)
        n_members = np.random.randint(3, 7)

        for _ in range(n_members):
            # Start point: near left edge (fixed wall)
            x1 = np.random.randint(0, 3)
            y1 = np.random.randint(0, resolution)

            # End point: near load application point
            x2 = resolution - 1
            y2 = load_y + np.random.randint(-4, 5)
            y2 = np.clip(y2, 0, resolution - 1)

            # Draw thick line (truss member)
            steps = max(abs(x2 - x1), abs(y2 - y1)) * 2
            for s in range(steps + 1):
                t = s / max(steps, 1)
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                # Draw with thickness
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        cx, cy = px + dx, py + dy
                        if 0 <= cx < resolution and 0 <= cy < resolution:
                            topo[cy, cx] = 1.0

        # Add some cross-bracing
        for _ in range(n_members // 2):
            x1 = np.random.randint(0, resolution // 2)
            y1 = np.random.randint(0, resolution)
            x2 = np.random.randint(resolution // 2, resolution)
            y2 = np.random.randint(0, resolution)

            steps = max(abs(x2 - x1), abs(y2 - y1)) * 2
            for s in range(steps + 1):
                t = s / max(steps, 1)
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        cx, cy = px + dx, py + dy
                        if 0 <= cx < resolution and 0 <= cy < resolution:
                            topo[cy, cx] = 1.0

        # Ensure volume fraction ~50%
        target_volume = 0.5 * resolution * resolution
        current_volume = np.sum(topo)
        if current_volume > 0:
            # Adjust by randomly adding/removing material
            while np.sum(topo) > target_volume * 1.2:
                # Remove random solid elements
                solid_pts = np.argwhere(topo > 0)
                if len(solid_pts) == 0:
                    break
                idx = np.random.randint(len(solid_pts))
                topo[solid_pts[idx][0], solid_pts[idx][1]] = 0.0

            while np.sum(topo) < target_volume * 0.8:
                # Add random elements near existing solid
                solid_pts = np.argwhere(topo > 0)
                if len(solid_pts) == 0:
                    break
                idx = np.random.randint(len(solid_pts))
                py, px = solid_pts[idx]
                dy, dx = np.random.randint(-2, 3, size=2)
                cy, cx = py + dy, px + dx
                if 0 <= cx < resolution and 0 <= cy < resolution:
                    topo[cy, cx] = 1.0

        topologies[i, 0] = topo

    return topologies, conditions


print("\n[1] Generating synthetic topology optimization data...")
topologies, conditions = generate_cantilever_topologies(n_samples=200, resolution=32)
print(f"   Topologies shape: {topologies.shape}")
print(f"   Conditions shape: {conditions.shape}")
print(f"   Volume fraction (sample 0): {topologies[0].mean():.3f}")

# ============================================================================
# [2] Create TopoDiff Model
# ============================================================================
print("\n[2] Creating TopoDiff model...")

# TopoDiff is a U-Net based diffusion model
# - img_resolution: spatial resolution of the design domain
# - in_channels: 1 (noisy topology) + 3 (conditions) = 4 (concatenated in forward)
# - out_channels: 1 (predicted noise, same shape as topology)
# - model_channels: base width of the U-Net
# - channel_mult: multiplier at each resolution level
# - attn_resolutions: where to apply self-attention

model = TopoDiff(
    img_resolution=32,
    in_channels=4,       # 1 (noisy topology) + 3 (conditions: fixed_mask, load_x, load_y)
    out_channels=1,      # predicted noise
    label_dim=0,         # unconditional (no class labels)
    augment_dim=0,       # no augmentation
    model_channels=64,   # base channel width (reduced for CPU/memory)
    channel_mult=[1, 2, 2, 2],  # 4 resolution levels
    channel_mult_emb=4,
    num_blocks=2,        # residual blocks per level
    attn_resolutions=[16, 8],  # attention at 16x16 and 8x8
    dropout=0.1,
    label_dropout=0.0,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"   Model parameters: {n_params:,}")

# Create diffusion process handler
diffusion = Diffusion(
    n_steps=200,        # number of diffusion steps (reduced for speed)
    min_beta=1e-4,     # minimum noise variance
    max_beta=0.02,     # maximum noise variance
    device=device,
)

print(f"   Diffusion steps: {diffusion.n_steps}")

# ============================================================================
# [3] Training Setup
# ============================================================================
print("\n[3] Setting up training...")

# Convert data to tensors
topo_tensor = torch.from_numpy(topologies).to(device)
cond_tensor = torch.from_numpy(conditions).to(device)

# Optimizer: Adam with learning rate scheduling
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

n_epochs = 200
batch_size = 16
n_batches = len(topo_tensor) // batch_size

print(f"   Epochs: {n_epochs}")
print(f"   Batch size: {batch_size}")
print(f"   Batches per epoch: {n_batches}")

# ============================================================================
# [4] Training Loop
# ============================================================================
print("\n[4] Training TopoDiff diffusion model...")
print("    (Diffusion training: model learns to predict noise at each step)")
print("    Forward process: x_t = sqrt(a_bar_t)*x_0 + sqrt(1-a_bar_t)*noise")
print("    Loss: MSE(predicted_noise, actual_noise)\n")

loss_history = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0

    # Shuffle data
    perm = torch.randperm(len(topo_tensor))
    topo_shuffled = topo_tensor[perm]
    cond_shuffled = cond_tensor[perm]

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size

        x0 = topo_shuffled[start:end]  # clean topologies
        cons = cond_shuffled[start:end]  # boundary conditions

        # Compute diffusion training loss
        # The Diffusion.train_loss method:
        # 1. Samples random timesteps t
        # 2. Adds noise to x0 to get x_t (forward diffusion)
        # 3. Model predicts the noise given x_t, conditions, and t
        # 4. Returns MSE between predicted and actual noise
        loss = diffusion.train_loss(model, x0, cons)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

print("\n   Training complete!")

# ============================================================================
# [5] Sampling: Generate New Topologies
# ============================================================================
print("\n[5] Generating new topologies via reverse diffusion...")

model.eval()

@torch.no_grad()
def sample_topology(model, diffusion, condition, n_steps=None):
    """
    Generate a topology using the trained diffusion model.

    Reverse diffusion process:
    1. Start from pure noise x_T ~ N(0, I)
    2. For t = T, T-1, ..., 1:
       - Predict noise using model
       - Remove predicted noise to get x_{t-1}
    3. Return x_0 (generated topology)

    Parameters
    ----------
    model : TopoDiff
        Trained diffusion model
    diffusion : Diffusion
        Diffusion process handler
    condition : torch.Tensor, shape (1, 3, H, W)
        Boundary conditions [fixed_mask, load_x, load_y]
    n_steps : int, optional
        Number of sampling steps (default: diffusion.n_steps)

    Returns
    -------
    x : torch.Tensor, shape (1, 1, H, W)
        Generated topology (values in [0, 1])
    """
    if n_steps is None:
        n_steps = diffusion.n_steps

    b, c, h, w = condition.shape
    device = condition.device

    # Start from pure noise
    x = torch.randn(1, 1, h, w, device=device)

    # Reverse diffusion: gradually denoise
    for t in reversed(range(n_steps)):
        t_batch = torch.tensor([t], device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x, condition, t_batch)

        # DDPM sampling step
        beta_t = diffusion.betas[t]
        alpha_t = diffusion.alphas[t]
        alpha_bar_t = diffusion.alpha_bars[t]

        # Mean of p(x_{t-1} | x_t)
        mean = (1 / alpha_t.sqrt()) * (
            x - (beta_t / (1 - alpha_bar_t).sqrt()) * predicted_noise
        )

        if t > 0:
            # Add stochastic noise (except at t=0)
            noise = torch.randn_like(x)
            sigma = beta_t.sqrt()
            x = mean + sigma * noise
        else:
            x = mean

    # Clamp to [0, 1] and threshold to binary
    x = x.clamp(0, 1)
    return x


# Generate topologies for a few test conditions
n_test = 4
test_conditions = cond_tensor[:n_test]
generated_topologies = []

for i in range(n_test):
    print(f"   Generating topology {i+1}/{n_test}...")
    cond = test_conditions[i:i+1]
    gen = sample_topology(model, diffusion, cond, n_steps=50)  # fewer steps for speed
    generated_topologies.append(gen)

# ============================================================================
# [6] Visualization
# ============================================================================
print("\n[6] Visualizing results...")

fig, axes = plt.subplots(4, n_test, figsize=(4 * n_test, 16))

for i in range(n_test):
    # Row 1: Boundary conditions (fixed mask)
    ax = axes[0, i]
    fixed = test_conditions[i, 0].cpu().numpy()
    ax.imshow(fixed, cmap='RdGy_r', vmin=0, vmax=1)
    ax.set_title(f'BC: Fixed Wall\n(Sample {i+1})', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Row 2: Load conditions
    ax = axes[1, i]
    load = test_conditions[i, 2].cpu().numpy()  # load_y
    ax.imshow(load, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(f'Load Condition\n(Downward force)', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Row 3: Training data (ground truth)
    ax = axes[2, i]
    gt = topo_tensor[i, 0].cpu().numpy()
    ax.imshow(gt, cmap='binary', vmin=0, vmax=1)
    vol_frac = gt.mean()
    ax.set_title(f'Ground Truth Topology\n(Vol. frac: {vol_frac:.2f})', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Row 4: Generated topology
    ax = axes[3, i]
    gen = generated_topologies[i][0, 0].cpu().numpy()
    # Threshold to binary
    gen_binary = (gen > 0.5).astype(float)
    ax.imshow(gen_binary, cmap='binary', vmin=0, vmax=1)
    vol_frac_gen = gen_binary.mean()
    ax.set_title(f'Generated Topology\n(Vol. frac: {vol_frac_gen:.2f})', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.suptitle('TopoDiff: Topology Optimization via Diffusion Model\n'
             '(Cantilever Beam Design)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_topodiff.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_topodiff.png")

# ============================================================================
# [7] Training Loss Plot
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(loss_history, 'b-', linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Diffusion Loss (MSE)', fontsize=12)
ax.set_title('TopoDiff Training Loss', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('e:/physicsnemo_env/result_topodiff_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: result_topodiff_loss.png")

# ============================================================================
# [8] Summary
# ============================================================================
print("\n" + "=" * 70)
print("Tutorial Summary: Topology Optimization with TopoDiff")
print("=" * 70)
print(f"""
Physics:
  - Topology optimization finds optimal material distribution
  - Traditional: SIMP/MMA (hundreds of FEM iterations)
  - TopoDiff: Single forward pass of diffusion model

Model:
  - TopoDiff: U-Net with self-attention (ADM architecture)
  - Diffusion: 200 steps, linear beta schedule
  - Parameters: {n_params:,}

Training:
  - Epochs: {n_epochs}
  - Final loss: {loss_history[-1]:.6f}
  - Data: 200 synthetic cantilever topologies

Key Concepts:
  1. Forward diffusion: gradually add noise to topology
  2. Reverse diffusion: model learns to denoise
  3. Conditioning: boundary conditions guide generation
  4. Sampling: start from noise, denoise step by step

Next Steps:
  - Use real FEM optimization data for training
  - Increase resolution (64x64, 128x128)
  - Add physics-informed loss (compliance evaluation)
  - Compare with SIMP-optimized topologies
""")
print("=" * 70)
