"""
PhysicsNeMo Tutorial: Solving 2D Lid Driven Cavity Flow with PINN
=================================================================
This tutorial uses PhysicsNeMo 1.3.0's FullyConnected model to implement
a PINN (Physics-Informed Neural Network) using PyTorch autograd.

Problem: 2D flow inside a box (Lid Driven Cavity)
- The top wall moves to the right, driving the flow
- The flow is governed by the Navier-Stokes equations

Learning objectives:
1. How to use PhysicsNeMo models
2. PINN fundamentals (computing PDE residuals via automatic differentiation)
3. GPU-accelerated training
4. Result visualization

How to run:
  cd YourDirectory\\physicsnemo_env
  Scripts\\activate
  python tutorial_pinn_ldc2d.py

==========================================================================
[Understanding the Overall Flow of This Tutorial]
==========================================================================

This program consists of the following 8 steps:

  [0] Environment Setup   - Check for GPU, fix random seeds
  [1] Problem Setup       - Define the physics problem (equations, BCs)
  [2] Model Creation      - Create a neural network: input (x,y) -> output (u,v,p)
  [3] Training Setup      - Configure optimizer and number of epochs
  [4] Point Generation    - Generate random coordinate points for training
  [5] Loss Function       - Measure how close predictions are to the answer
  [6] Training Loop       - Iteratively train until the network finds the solution
  [7] Visualization       - Plot the trained network's predictions
  [8] Summary             - Print results

==========================================================================
[What is PINN in One Sentence?]
==========================================================================

  A technique that teaches physics equations to a neural network so that
  the network itself discovers a solution that obeys the laws of physics.

  - Standard deep learning: provide labeled data and learn patterns (supervised)
  - PINN: instead of labeled data, provide physics equations and train the
          network to find a solution that satisfies them (equation = solution constraint)

==========================================================================
[The Problem Solved in This Tutorial: Lid Driven Cavity Flow]
==========================================================================

  +----------------------> u = 1 (lid moves to the right)
  |                      |
  |     (vortex/swirl)   |   <- When the top wall moves,
  |                      |      a clockwise vortex forms inside
  |   (fluid circulates) |
  |                      |
  +----------------------+
   u = 0 (bottom is fixed)

  The x-axis points right, the y-axis points up.
  Box size: width 1, height 1 (0 <= x <= 1, 0 <= y <= 1)
"""

# ============================================================================
# Library Imports
# ============================================================================
# torch: PyTorch deep learning framework (neural networks, autodiff, GPU support)
#   - torch.autograd.grad: automatic differentiation function (core of PINN!)
#     Differentiates the network's output w.r.t. input to compute partial
#     derivatives (e.g., du/dx).
import torch
import torch.nn as nn

# numpy: numerical computing library (array operations, math functions)
import numpy as np

# matplotlib: plotting/visualization library (for result visualization)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# physicsnemo: NVIDIA's physics AI library
#   - FullyConnected: the most basic neural network architecture (MLP)
#     A structure of: input -> hidden layers (multiple) -> output
import physicsnemo
from physicsnemo.models.mlp.fully_connected import FullyConnected

# time: for measuring training time
import time
# os: for file/folder path handling
import os


# ============================================================================
# [0] Environment Setup
# ============================================================================
print("=" * 70)
print("  PhysicsNeMo PINN Tutorial: 2D Lid Driven Cavity Flow")
print("=" * 70)

# Use GPU (CUDA) if available, otherwise use CPU.
# GPUs have thousands of cores for parallel computation, greatly speeding up training.
#   - CPU: 4-16 cores, good at serial processing (training takes hours to days)
#   - GPU: thousands of cores, good at parallel processing (training takes minutes to hours)
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

# Set random seeds for reproducibility.
# Fixing the seed ensures random numbers are generated in the same order,
# so running the code multiple times produces identical results (essential for debugging).
torch.manual_seed(42)   # Fix PyTorch random seed
np.random.seed(42)      # Fix NumPy random seed


# ============================================================================
# [1] Problem Setup - What physics problem are we solving?
# ============================================================================
# The problem we are solving is "2D flow inside a box (Lid Driven Cavity Flow)".
#
# This flow is described by the following equations (Navier-Stokes equations):
#
#  (1) Continuity equation (mass conservation):
#      du/dx + dv/dy = 0
#      -> "what comes in = what goes out" (fluid doesn't appear or disappear)
#      -> u: x-direction velocity, v: y-direction velocity
#
#  (2) Momentum equation - x-direction (fluid version of Newton's F=ma):
#      u*du/dx + v*du/dy = -dp/dx + nu*(d2u/dx2 + d2u/dy2)
#      -> Left side: inertia term (force from fluid's momentum)
#      -> Right first: pressure term (pressure difference pushes the fluid)
#      -> Right second: viscous term (fluid's stickiness; larger nu = more sticky)
#
#  (3) Momentum equation - y-direction:
#      u*dv/dx + v*dv/dy = -dp/dy + nu*(d2v/dx2 + d2v/dy2)
#
#  Variable descriptions:
#    u, v = x, y components of velocity (values we want to find)
#    p = pressure (value we want to find)
#    nu = kinematic viscosity (fluid stickiness; water < honey < tar)
#    rho = density (mass per unit volume)
#
# Reynolds number (Re): an important dimensionless number characterizing the flow
#   Re = 1/nu (in this problem)
#   Small Re -> strong viscosity -> laminar flow (smooth)
#   Large Re -> weak viscosity -> near-turbulent (complex flow)

NU = 0.01       # Kinematic viscosity (Reynolds number = 1/nu = 100, laminar flow)
RHO = 1.0       # Density (similar to water)

# Domain (the space to solve in): square box [0,1] x [0,1]
#
# Boundary conditions (what the velocity should be at each of the four walls):
#
#   Top wall (y=1): u=1, v=0    -> lid moves to the right at velocity 1 [KEY]
#   Bottom wall (y=0): u=0, v=0 -> bottom is fixed (does not move)
#   Left wall (x=0): u=0, v=0   -> left wall is fixed
#   Right wall (x=1): u=0, v=0  -> right wall is fixed
#
# Why boundary conditions matter: the top wall moving alone causes
# flow to develop throughout the entire box (like stirring water in a cup).


# ============================================================================
# [2] Create PhysicsNeMo Model - Building the Neural Network (Artificial Brain)
# ============================================================================
print("[1/6] Creating PhysicsNeMo FullyConnected model...")

# Role of the neural network: take coordinates (x, y) as input and output
# the velocity and pressure at that location.
#
#   Input:  (x, y)     -> 2 values (spatial coordinates)
#   Output: (u, v, p)  -> 3 values (x-velocity, y-velocity, pressure)
#
# Network structure (FullyConnected = all neurons are connected to each other):
#
#   Input layer   Hidden1      Hidden2     ...   Hidden5      Output layer
#   (2 neurons)  (50 neurons) (50 neurons)       (50 neurons) (3 neurons)
#    x  -------> O -------> O -------> ... ---> O -------> u
#    y  -------> O -------> O -------> ... ---> O -------> v
#               O          O                   O       ---> p
#              ...        ...                 ...
#               O          O                   O
#             (50)        (50)                (50)
#
# Between each layer, an activation function (Tanh) adds nonlinearity.
# (Without activation functions, the network can only perform linear transformations)
#
# Why use Tanh?
#   - Tanh is a smooth curve in the range -1 to 1, differentiable and stable
#   - PINN requires 2nd-order derivatives, so a smoothly differentiable function is important
#   - ReLU is not differentiable at 0, making it unsuitable for PINNs
#
# weight_norm (weight normalization):
#   - A technique that makes training more stable
#   - Normalizes the magnitude of weights to prevent gradient explosion/vanishing

model = FullyConnected(
    in_features=2,       # Input: x, y (2 coordinates)
    out_features=3,      # Output: u, v, p (3 physical quantities)
    layer_size=50,       # Number of neurons per hidden layer (tuned for Quadro P4000 GPU)
    num_layers=5,        # Number of hidden layers (deeper = can learn more complex patterns)
    activation_fn="tanh",  # Tanh activation function (suitable for PINN)
    weight_norm=True,    # Weight normalization (improves training stability)
)
model = model.to(device)  # Move model to GPU (required for GPU computation)
print(f"      Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# Parameter count = total number of weights (w) and biases (b) the network must learn
# Larger values mean more expressive power, but harder training and more memory usage.


# ============================================================================
# [3] Training Setup - How to train the neural network?
# ============================================================================
print("[2/6] Setting up training...")

# Optimizer: the algorithm that decides how to update the network's weights
# Adam: the most widely used optimizer, stable and fast with adaptive learning rate
# lr (learning rate): how much to change the weights at each step
#   - Too large: overshoots the optimum (diverges)
#   - Too small: training is too slow
#   - 1e-3 (0.001): a commonly used value for PINNs
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Number of training iterations (Epochs)
# 1 epoch = one pass through the entire training data
# After 5000 iterations, the network gradually approaches the correct solution
EPOCHS = 5000

# Batch sizes (number of data points to train on at once)
N_INTERIOR = 5000   # Number of interior sampling points (where PDE is applied)
N_BOUNDARY = 200    # Number of sampling points per boundary/wall (where BCs are applied)


# ============================================================================
# [4] Generate Sampling Points - Create coordinate points for training
# ============================================================================
print("[3/6] Generating training points...")

# Why do we need "points"?
#
# PINN does not compute the entire space at once.
# Instead, it checks whether the equation is satisfied at multiple locations (points)
# within the space. It's like solving many problems on an exam paper.
#
#   - Interior points: points inside the box -> check if PDE (equation) is satisfied
#   - Boundary points: points on the box walls -> check if BCs are satisfied

# 4-1. Interior domain points (where the PDE is applied)
# Generate 5000 random points with x, y in [0, 1]
# torch.rand: generates uniform random numbers between 0 and 1
x_interior = torch.rand(N_INTERIOR, 1, device=device)
y_interior = torch.rand(N_INTERIOR, 1, device=device)

# requires_grad_(True): enables differentiation (autograd) for this variable
# PINN needs to differentiate w.r.t. the inputs (x, y), so this is essential!
# (Only with this enabled can torch.autograd.grad compute du/dx, etc.)
x_interior.requires_grad_(True)
y_interior.requires_grad_(True)

# 4-2. Boundary domain points (where boundary conditions are applied)
# Generate 200 points for each wall.
#
#   y=1 ---------------------  <- top wall (x_top, y_top): x is random, y=1
#   |                       |
#   |                       |  <- left/right walls: x is fixed (0 or 1), y is random
#   |                       |
#   y=0 ---------------------  <- bottom wall: x is random, y=0
#  x=0                     x=1

# Top wall (y=1): x is random in [0,1], y is always 1
x_top = torch.rand(N_BOUNDARY, 1, device=device)
y_top = torch.ones(N_BOUNDARY, 1, device=device)

# Bottom wall (y=0): x is random in [0,1], y is always 0
x_bottom = torch.rand(N_BOUNDARY, 1, device=device)
y_bottom = torch.zeros(N_BOUNDARY, 1, device=device)

# Left wall (x=0): x is always 0, y is random in [0,1]
x_left = torch.zeros(N_BOUNDARY, 1, device=device)
y_left = torch.rand(N_BOUNDARY, 1, device=device)

# Right wall (x=1): x is always 1, y is random in [0,1]
x_right = torch.ones(N_BOUNDARY, 1, device=device)
y_right = torch.rand(N_BOUNDARY, 1, device=device)


# ============================================================================
# [5] Define PINN Loss Function - "How close are we to the answer?"
# ============================================================================
# Loss: a number indicating how far the network's prediction is from the answer.
# The closer the loss is to 0, the closer we are to the correct solution.
# Training = the process of minimizing the loss (by adjusting weights little by little)

def compute_pde_residuals(x, y, model):
    """
    Compute the residuals of the Navier-Stokes equations.

    [What is a Residual?]
    When an equation is written as "left-hand side - right-hand side = 0",
    the residual is how far the computed result (using the network's predictions)
    deviates from zero.

    Residual = 0  -> equation is perfectly satisfied (correct!)
    Residual != 0 -> equation is not satisfied (more training needed)

    Core of PINN: automatic differentiation (autograd) computes partial derivatives.
    There is no need to manually compute du/dx, d2u/dx2, etc. --
    PyTorch does the differentiation for you!
    """
    # --- Step 1: Neural network prediction ---
    # Feed (x, y) coordinates into the network to get (u, v, p).
    # torch.cat: concatenate x and y horizontally into an [N, 2] matrix
    inputs = torch.cat([x, y], dim=1)  # [N, 1] + [N, 1] -> [N, 2]
    outputs = model(inputs)            # [N, 2] -> [N, 3] (through the network)
    u = outputs[:, 0:1]  # 1st output: x-component of velocity (u)
    v = outputs[:, 1:2]  # 2nd output: y-component of velocity (v)
    p = outputs[:, 2:3]  # 3rd output: pressure (p)
    # Note: we use slicing 0:1 instead of just 0 to keep the [N, 1] shape.
    #   Using 0 alone would produce [N], causing a dimension mismatch.

    # --- Step 2: Compute 1st-order partial derivatives (using autograd) ---
    # torch.autograd.grad(output, input): differentiates output w.r.t. input
    #
    # Example: u_x = du/dx (differentiate u with respect to x)
    #
    # Parameter descriptions:
    #   grad_outputs=torch.ones_like(u):
    #     Weights for the Vector-Jacobian Product.
    #     Required when differentiating a vector (u) rather than a scalar loss.
    #
    #   create_graph=True:
    #     Keeps the computation graph so that the 1st-order derivative result
    #     can be differentiated again.
    #     [IMPORTANT] If False, 2nd-order derivatives (d2u/dx2) cannot be computed!
    #
    #   retain_graph=True:
    #     Keeps the graph when differentiating the same computation graph
    #     multiple times. (Needed because we differentiate w.r.t. several variables.)
    #
    #   [0]:
    #     grad returns a (gradient,) tuple, so we extract the first element.

    # Partial derivatives of u: du/dx, du/dy
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    # Partial derivatives of v: dv/dx, dv/dy
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                               create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                               create_graph=True, retain_graph=True)[0]
    # Partial derivatives of p: dp/dx, dp/dy
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                               create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p),
                               create_graph=True, retain_graph=True)[0]

    # --- Step 3: Compute 2nd-order partial derivatives ---
    # Differentiate the 1st-order derivative results again to get 2nd-order derivatives.
    # Example: u_xx = d(du/dx)/dx = d2u/dx2
    # create_graph=True is set even though we won't differentiate further,
    # because the graph is needed later for backpropagation.

    # 2nd-order partial derivatives of u: d2u/dx2, d2u/dy2
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                create_graph=True, retain_graph=True)[0]
    # 2nd-order partial derivatives of v: d2v/dx2, d2v/dy2
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                                create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y),
                                create_graph=True, retain_graph=True)[0]

    # --- Step 4: Compute Navier-Stokes residuals ---
    # Now combine the derivative values to compute the equation residuals.
    # The residuals must be zero for the equations to be satisfied.

    # (1) Continuity equation residual: du/dx + dv/dy = 0
    #     -> continuity = u_x + v_y  (this should be close to 0)
    continuity = u_x + v_y

    # (2) Momentum equation (x-direction) residual:
    #     u*du/dx + v*du/dy + dp/dx - nu*(d2u/dx2 + d2u/dy2) = 0
    #     -> momentum_x = inertia + pressure - viscosity (should be close to 0)
    #
    #     u * u_x: inertia term (effect of fluid carrying momentum as it moves)
    #     p_x / RHO: pressure term (pressure gradient pushes the fluid; divided by rho)
    #     NU * (u_xx + u_yy): viscous term (fluid stickiness diffuses velocity)
    momentum_x = u * u_x + v * u_y + p_x / RHO - NU * (u_xx + u_yy)

    # (3) Momentum equation (y-direction) residual:
    #     u*dv/dx + v*dv/dy + dp/dy - nu*(d2v/dx2 + d2v/dy2) = 0
    momentum_y = u * v_x + v * v_y + p_y / RHO - NU * (v_xx + v_yy)

    return continuity, momentum_x, momentum_y, u, v, p


def compute_boundary_loss(model, x_b, y_b, u_target, v_target):
    """
    Compute boundary condition loss.

    Measures how different the network's prediction is from the target value
    at the boundary (walls). This is the same as the standard MSE
    (Mean Squared Error) loss used in deep learning.

    MSE = mean of (prediction - target)^2

    Example: on the top wall, u=1, v=0 is required.
    If the network predicts u=0.8, v=0.1, a loss is incurred.
    """
    # Network prediction
    inputs = torch.cat([x_b, y_b], dim=1)
    outputs = model(inputs)
    u_pred = outputs[:, 0:1]  # Network's predicted u
    v_pred = outputs[:, 1:2]  # Network's predicted v

    # MSE loss = mean of (prediction - target)^2
    loss_u = torch.mean((u_pred - u_target) ** 2)  # loss for u
    loss_v = torch.mean((v_pred - v_target) ** 2)  # loss for v
    return loss_u + loss_v  # Return the sum of both losses


def total_loss(model, x_int, y_int,
               x_top, y_top, x_bot, y_bot, x_left, y_left, x_right, y_right):
    """
    Total loss = PDE loss + (weight * boundary condition loss)

    [PINN Loss Function Structure]

      Total Loss = Loss_PDE + lambda * Loss_BC

      - Loss_PDE: Are the physics equations satisfied in the interior?
        -> mean of continuity^2 + momentum_x^2 + momentum_y^2
        -> "Does it follow the laws of physics?"

      - Loss_BC: Are the boundary conditions satisfied at the walls?
        -> mean of (predicted velocity - target velocity)^2
        -> "Does it respect the wall conditions?"

      - lambda: weight (10.0 in this code)
        -> Applies boundary conditions more strongly so they are exactly met at walls
        -> Too large: PDE is ignored; too small: wall conditions become loose
    """
    # PDE loss (interior domain)
    # Compute the Navier-Stokes residuals at interior points
    cont, mom_x, mom_y, _, _, _ = compute_pde_residuals(x_int, y_int, model)
    # Compute the mean squared (MSE) of each residual and sum them
    # The closer the residuals are to 0, the smaller the loss
    loss_pde = torch.mean(cont ** 2) + torch.mean(mom_x ** 2) + torch.mean(mom_y ** 2)

    # Boundary condition loss
    # Check that the network's prediction matches the target velocity at each wall

    # Top wall: u=1, v=0 (lid moves to the right)
    # torch.ones_like: tensor filled with 1s (target for u)
    # torch.zeros_like: tensor filled with 0s (target for v)
    loss_top = compute_boundary_loss(model, x_top, y_top,
                                      torch.ones_like(x_top), torch.zeros_like(y_top))
    # Bottom wall: u=0, v=0 (fixed)
    loss_bottom = compute_boundary_loss(model, x_bot, y_bot,
                                         torch.zeros_like(x_bot), torch.zeros_like(y_bot))
    # Left wall: u=0, v=0 (fixed)
    loss_left = compute_boundary_loss(model, x_left, y_left,
                                       torch.zeros_like(x_left), torch.zeros_like(y_left))
    # Right wall: u=0, v=0 (fixed)
    loss_right = compute_boundary_loss(model, x_right, y_right,
                                        torch.zeros_like(x_right), torch.zeros_like(y_right))

    # Sum the losses from all walls
    loss_bc = loss_top + loss_bottom + loss_left + loss_right

    # Total loss (apply a 10x weight to boundary conditions)
    # The weight is used because BCs must be exactly satisfied for a physically
    # meaningful solution.
    total = loss_pde + 10.0 * loss_bc

    return total, loss_pde, loss_bc


# ============================================================================
# [6] Training Loop - Iteratively train until the network finds the solution
# ============================================================================
# The training loop is the most important part of PINN.
# We repeatedly adjust the weights so that the network gradually approaches
# the correct solution.

print("[4/6] Starting training (GPU accelerated)...")
print(f"      - Epochs: {EPOCHS}")
print(f"      - Interior points: {N_INTERIOR}")
print(f"      - Boundary points/wall: {N_BOUNDARY}")
print()

# Record start time to measure training duration
start_time = time.time()
# List to record loss history (used later for plotting)
loss_history = []

# Training loop: repeat EPOCHS times
for epoch in range(EPOCHS):
    # --- Step 1: Zero the gradients ---
    # Clear the gradients computed in the previous epoch.
    # If not cleared, gradients accumulate and cause incorrect updates.
    optimizer.zero_grad()

    # --- Step 2: Compute loss ---
    # Calculate how far the current network's prediction is from the answer.
    # Loss = PDE residual loss + 10 * boundary condition loss
    loss, loss_pde, loss_bc = total_loss(
        model, x_interior, y_interior,
        x_top, y_top, x_bottom, y_bottom,
        x_left, y_left, x_right, y_right
    )

    # --- Step 3: Backpropagation ---
    # Differentiate the loss w.r.t. each weight to compute gradients.
    # The gradient tells us "in which direction and how much to change each weight
    # to reduce the loss."
    loss.backward()

    # --- Step 4: Update weights ---
    # The optimizer (Adam) uses the gradients to update the weights.
    # Roughly: weight = weight - learning_rate * gradient
    optimizer.step()

    # Record current loss (for plotting)
    # loss.item(): extract a Python float from the tensor
    loss_history.append(loss.item())

    # Print progress every 500 epochs or at the last epoch
    if epoch % 500 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:5d}/{EPOCHS} | "
              f"Total Loss: {loss.item():.6e} | "
              f"PDE: {loss_pde.item():.6e} | "
              f"BC: {loss_bc.item():.6e} | "
              f"Time: {elapsed:.1f}s")

# Compute total training time
elapsed_total = time.time() - start_time
print(f"\n  Training complete! Total time: {elapsed_total:.1f}s")
print(f"  Final loss: {loss.item():.6e}")

# Save the trained model (can be loaded later for reuse)
# .pth is PyTorch's model weight file format
output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)
model_save_path = os.path.join(output_dir, "ldc2d_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"  Model saved: {model_save_path}")


# ============================================================================
# [7] Visualization - Plot the trained network's predictions
# ============================================================================
print("\n[5/6] Visualizing results...")

# model.eval(): switch the model to evaluation mode
# (Some layers behave differently in training vs. evaluation mode)
model.eval()

# torch.no_grad(): disable gradient computation inside this block
# (Only visualization is needed, so no differentiation -> saves memory, speeds up)
with torch.no_grad():
    # Create a grid (for visualization)
    # Build a 50x50 grid to predict values over the entire space.
    # Use indexing='xy' for compatibility with matplotlib streamplot
    N_GRID = 50
    x_grid = torch.linspace(0, 1, N_GRID, device=device)  # x: 0~1 in 50 steps
    y_grid = torch.linspace(0, 1, N_GRID, device=device)  # y: 0~1 in 50 steps
    # meshgrid: create a 2D grid from two 1D arrays (all (x,y) combinations)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    # Flatten the 2D grid to 1D so it can be fed into the network
    X_flat = X.reshape(-1, 1)  # [2500, 1]
    Y_flat = Y.reshape(-1, 1)  # [2500, 1]

    # Model prediction: compute (u, v, p) for 2500 points at once
    inputs = torch.cat([X_flat, Y_flat], dim=1)
    outputs = model(inputs)
    # Reshape the 1D predictions back to 2D grid form
    # .cpu().numpy(): move GPU tensor to CPU and convert to NumPy array
    # (matplotlib can only handle NumPy arrays)
    U = outputs[:, 0].reshape(N_GRID, N_GRID).cpu().numpy()  # velocity x
    V = outputs[:, 1].reshape(N_GRID, N_GRID).cpu().numpy()  # velocity y
    P = outputs[:, 2].reshape(N_GRID, N_GRID).cpu().numpy()  # pressure

    X_np = X.cpu().numpy()  # grid x coordinates (NumPy)
    Y_np = Y.cpu().numpy()  # grid y coordinates (NumPy)

# --- Create figure: 4 subplots (arranged horizontally) ---
fig = plt.figure(figsize=(16, 5))
gs = GridSpec(1, 4, figure=fig)  # 1 row, 4 columns

# 1. Velocity U (x-component) - contourf: filled contour plot
ax1 = fig.add_subplot(gs[0, 0])
cf1 = ax1.contourf(X_np, Y_np, U, levels=50, cmap='RdBu_r')  # red-blue colormap
ax1.set_title('Velocity U (x-component)', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
plt.colorbar(cf1, ax=ax1)  # color bar (shows value range)

# 2. Velocity V (y-component)
ax2 = fig.add_subplot(gs[0, 1])
cf2 = ax2.contourf(X_np, Y_np, V, levels=50, cmap='RdBu_r')
ax2.set_title('Velocity V (y-component)', fontsize=12)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.colorbar(cf2, ax=ax2)

# 3. Pressure P
ax3 = fig.add_subplot(gs[0, 2])
cf3 = ax3.contourf(X_np, Y_np, P, levels=50, cmap='viridis')  # purple-yellow colormap
ax3.set_title('Pressure P', fontsize=12)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
plt.colorbar(cf3, ax=ax3)

# 4. Flow vector field (streamplot) - shows flow direction as lines
ax4 = fig.add_subplot(gs[0, 3])
speed = np.sqrt(U**2 + V**2)  # velocity magnitude = sqrt(u^2 + v^2)
strm = ax4.streamplot(X_np, Y_np, U, V, color=speed, cmap='coolwarm', density=1.5)
ax4.set_title('Flow Streamlines', fontsize=12)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
plt.colorbar(strm.lines, ax=ax4, label='Speed')

# Set overall title
plt.suptitle(f'PINN: 2D Lid Driven Cavity Flow (Re={1/NU:.0f}, nu={NU})\n'
             f'PhysicsNeMo {physicsnemo.__version__} | '
             f'PyTorch {torch.__version__} | '
             f'Device: {device}',
             fontsize=14, fontweight='bold')
plt.tight_layout()  # automatically adjust subplot spacing

# Save result image
fig_path = os.path.join(output_dir, "ldc2d_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')  # dpi: resolution
print(f"  Result image saved: {fig_path}")

# Loss curve figure (shows how training progressed)
fig2, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(loss_history, linewidth=0.5)  # semilogy: log scale on y-axis
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Loss (log scale)')
ax.set_title('Training Loss History', fontsize=14)
ax.grid(True, alpha=0.3)
fig2_path = os.path.join(output_dir, "ldc2d_loss.png")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Loss curve saved: {fig2_path}")

# Memory cleanup (close all open figures)
plt.close('all')


# ============================================================================
# [8] Summary - Print training results
# ============================================================================
print("\n[6/6] Tutorial complete!")
print("=" * 70)
print("  Training Summary:")
print(f"    - Problem: 2D Lid Driven Cavity Flow")
print(f"    - Equations: Navier-Stokes (steady-state, incompressible)")
print(f"    - Reynolds number: {1/NU:.0f}")
print(f"    - Network: FullyConnected (5 layers, 50 neurons/layer, Tanh)")
print(f"    - Training time: {elapsed_total:.1f}s")
print(f"    - Final loss: {loss.item():.6e}")
print(f"    - Device: {device}")
print()
print("  Result files:")
print(f"    - {fig_path}")
print(f"    - {fig2_path}")
print("=" * 70)
print()
print("  Next steps:")
print("    1. Open the result image to inspect the flow field")
print("    2. Increase EPOCHS for better accuracy (e.g., 10000)")
print("    3. Change NU to try different Reynolds numbers (e.g., 0.01->0.001)")
print("    4. Adjust layer_size or num_layers to change network size")
print("    5. Change the learning rate (lr) to adjust convergence speed")
print("=" * 70)
