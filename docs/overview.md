# PhysicsNeMo PINN Tutorial Guide (Detailed Guide for Beginners)

> **Date**: 2026-06-30  
> **Environment**: PhysicsNeMo 1.3.0, PyTorch 2.7.1+cu118, CUDA (Quadro P4000 8GB)  
> **Location**: `E:\physicsnemo_env`

---

## Table of Contents

1. [Tutorial Overview](#1-tutorial-overview)
2. [What is PINN? (Beginner-Friendly Explanation)](#2-what-is-pinn-beginner-friendly-explanation)
3. [Understanding Navier-Stokes Equations](#3-understanding-navier-stokes-equations)
4. [How to Run the Tutorial](#4-how-to-run-the-tutorial)
5. [Step-by-Step Code Explanation](#5-step-by-step-code-explanation)
6. [Interpreting Results](#6-interpreting-results)
7. [Parameter Experimentation Guide](#7-parameter-experimentation-guide)
8. [Next Learning Steps](#8-next-learning-steps)
9. [Troubleshooting (FAQ)](#9-troubleshooting-faq)

---

## 1. Tutorial Overview

### What Did We Learn?

In this tutorial, we used a **PINN (Physics-Informed Neural Network)** to solve the **2D Lid Driven Cavity Flow** problem (flow inside a box).

| Item | Details |
|------|---------|
| **Problem** | 2D flow inside a box (top wall moves to generate flow) |
| **Governing equation** | Navier-Stokes equations (incompressible, steady-state) |
| **Reynolds number** | 100 (kinematic viscosity nu = 0.01) |
| **Neural network** | PhysicsNeMo FullyConnected (5 layers, 50 neurons/layer, Tanh) |
| **Training time** | ~324 seconds (GPU: Quadro P4000) |
| **Final loss** | 0.153 (converged from 17.6) |
| **Result** | Successfully predicted physically accurate flow field |

### Files Used

| File | Description |
|------|-------------|
| `tutorial_pinn_ldc2d.py` | Main tutorial script (PINN implementation) |
| `tutorial_results/ldc2d_result.png` | Flow field visualization (U, V, P, Streamlines) |
| `tutorial_results/ldc2d_loss.png` | Training loss curve |
| `tutorial_results/ldc2d_model.pth` | Trained model weights |

---

## 2. What is PINN? (Beginner-Friendly Explanation)

### 2.1 Difference Between Standard Deep Learning and PINN

**Standard Deep Learning (Supervised Learning):**
- Give 10,000 cat photos with labels "this is a cat" for training
- The neural network finds patterns and classifies new photos
- Labeled data (ground truth) is required

**PINN (Physics-Informed Neural Network):**
- Instead of labeled data, provide **physics equations**
- Train the network to "find a solution that satisfies this equation"
- Can find solutions using only physics laws, without any labeled data

### 2.2 Core Idea of PINN

```
[Neural Network]         [Automatic Differentiation]     [Loss Function]
(x, y) coordinates  -->  Network predicts  -->  Check if equation  -->  Compute loss
  input               (u, v, p) output       is satisfied (0 = correct)
```

1. **Network approximates the solution**: Input coordinates (x, y), output velocity and pressure (u, v, p) at that location
2. **Automatic differentiation**: PyTorch automatically computes partial derivatives (du/dx, d²u/dx², etc.)
3. **Physics laws as loss**: The residual of the equation (how far from satisfying equation = 0) is used as loss
4. **Training**: Minimizing the loss makes the network find a solution that follows physics laws

### 2.3 Why PINN is Good

| Feature | Traditional CFD | PINN |
|---------|----------------|------|
| Mesh generation | Very difficult for complex geometries | Not needed (just place points) |
| High-dimensional problems | Computation explodes with dimensions | Handles relatively well |
| Inverse problems (parameter estimation) | Difficult | Naturally supported |
| Labeled data | Not needed (just need equations) | Not needed |

---

## 3. Understanding Navier-Stokes Equations

### 3.1 Fundamentals of Fluid Dynamics

The **Navier-Stokes equations** describe how fluids (water, air, etc.) flow.

These equations consist of 3 parts:

#### (1) Continuity Equation (Mass Conservation)

```
du/dx + dv/dy = 0
```

- Meaning: "What comes in = What goes out"
- Fluid doesn't suddenly appear or disappear
- Example: If water enters a pipe at 1 L/s, it must exit at 1 L/s

#### (2) Momentum Equation - x-direction (Fluid version of Newton's F=ma)

```
u * du/dx + v * du/dy = -dp/dx + nu * (d²u/dx² + d²u/dy²)
```

Left side (inertia term): Force from fluid momentum
Right side first (pressure term): Force from pressure differences pushing fluid
Right side second (viscosity term): Velocity spreading due to fluid stickiness (viscosity)

#### (3) Momentum Equation - y-direction

```
u * dv/dx + v * dv/dy = -dp/dy + nu * (d²v/dx² + d²v/dy²)
```

Same structure as x-direction but for y-direction.

### 3.2 Variable Explanation

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|---------|
| u | Velocity x-component | Right/left flow speed | Wind blowing east |
| v | Velocity y-component | Up/down flow speed | Air rising |
| p | Pressure | Force per unit area | Pressure inside a balloon |
| nu | Kinematic viscosity | Fluid stickiness | Water (small) < Honey (large) |
| rho | Density | Mass per unit volume | Air (light) < Water (heavy) |

### 3.3 Reynolds Number (Re)

The Reynolds number is a crucial parameter that determines flow characteristics:

```
Re = 1/nu (in this tutorial)
```

- **Low Re** (e.g., Re=10): Strong viscosity → Smooth laminar flow (like honey flowing)
- **High Re** (e.g., Re=1000): Weak viscosity → Complex turbulent flow (like a fast river)
- This tutorial: Re=100 (moderate laminar flow, good difficulty for PINN)

### 3.4 Lid Driven Cavity Flow Problem

Let's understand the problem visually:

```
   y=1 +----------------------+  <- Top wall moves right (u=1)
       |  ----> ----> ---->   |     (like wiping a cup with your hand)
       |                      |
       |    (clockwise        |
       |     vortex forms)    |     <- When the top wall moves,
       |                      |        a vortex forms inside
       |    fluid circulates |
       |                      |
   y=0 +----------------------+  <- Bottom is fixed (u=0, v=0)
      x=0                    x=1

  Left/right walls also fixed (u=0, v=0)
```

---

## 4. How to Run the Tutorial

### Method 1: Run from Command Line

```cmd
cd E:\physicsnemo_env
Scripts\activate
python tutorial_pinn_ldc2d.py
```

### Method 2: Run from VS Code

1. Open `E:\physicsnemo_env\tutorial_pinn_ldc2d.py` in VS Code
2. Set Python interpreter to `E:\physicsnemo_env\Scripts\python.exe`
3. Run with `F5` or `Ctrl+F5`

### View Results

```cmd
# Open result images
start E:\physicsnemo_env\tutorial_results\ldc2d_result.png
start E:\physicsnemo_env\tutorial_results\ldc2d_loss.png
```

---

## 5. Step-by-Step Code Explanation

### 5.1 Overall Structure (8 Steps)

```
tutorial_pinn_ldc2d.py
|
+-- [0] Environment Setup    - Check for GPU, fix random seeds
+-- [1] Problem Setup        - Define physics problem (equations, boundary conditions)
+-- [2] Model Creation       - Create neural network: input (x,y) → output (u,v,p)
+-- [3] Training Setup       - Configure optimizer, epochs
+-- [4] Point Generation     - Randomly generate coordinate points for training
+-- [5] Loss Function        - Compute how far from the correct answer
+-- [6] Training Loop        - Iteratively train until network finds the answer
+-- [7] Visualization         - Plot the trained network's predictions
+-- [8] Summary              - Print results
```

### 5.2 [0] Environment Setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- Uses GPU if available, otherwise CPU
- GPU has thousands of cores for parallel computation, making training 10-100x faster
- Quadro P4000 GPU is detected and automatically used

```python
torch.manual_seed(42)
np.random.seed(42)
```

- Fixing seeds ensures consistent results across runs
- Essential for debugging and result reproduction

### 5.3 [1] Problem Setup

```python
NU = 0.01    # Kinematic viscosity (Reynolds number = 100)
RHO = 1.0    # Density
```

- NU (kinematic viscosity): Fluid stickiness. 0.01 is similar to water
- RHO (density): 1.0 is similar to water density
- Reynolds number = 1/NU = 100, a moderate laminar flow

Boundary conditions (4 walls):
- Top wall (y=1): u=1, v=0 (lid moves right) - this drives the flow
- Other 3 walls: u=0, v=0 (fixed, no-slip)

### 5.4 [2] Neural Network Model Creation

```python
model = FullyConnected(
    in_features=2,        # Input: x, y (2 coordinates)
    out_features=3,       # Output: u, v, p (3 physical quantities)
    layer_size=50,        # Neurons per hidden layer
    num_layers=5,         # Number of hidden layers
    activation_fn="tanh", # Activation function
    weight_norm=True,     # Weight normalization
)
```

Network role: **Input coordinates (x, y), output velocity and pressure (u, v, p) at that location**

Network structure (FullyConnected = all neurons connected):

```
Input(2) -> Hidden1(50) -> Hidden2(50) -> ... -> Hidden5(50) -> Output(3)
  x,y        50 neurons      50 neurons             50 neurons      u,v,p
```

**Why use Tanh activation?**
- PINN requires 2nd-order derivatives (d²u/dx², etc.)
- Tanh is a smooth curve differentiable everywhere (range -1 to 1)
- ReLU is not differentiable at 0, making it unsuitable for PINNs

**What is weight_norm?**
- A technique to maintain consistent weight magnitudes
- Helps training be more stable
- Prevents gradient explosion (weights diverging) or vanishing (weights → 0)

### 5.5 [3] Training Setup

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 5000
N_INTERIOR = 5000
N_BOUNDARY = 200
```

- **Optimizer (Adam)**: Algorithm that decides how to update network weights
  - Adam is the most widely used optimizer, automatically adjusting learning rate
- **Learning rate (lr=1e-3=0.001)**: How much to change weights at each step
  - Too large: Overshoots and diverges
  - Too small: Training is too slow
  - 0.001 is a commonly used value for PINNs
- **EPOCHS (5000)**: How many times to iterate over all training data
- **N_INTERIOR (5000)**: Number of interior points to check equations
- **N_BOUNDARY (200)**: Number of boundary points per wall to check BCs

### 5.6 [4] Sampling Point Generation

PINN doesn't compute the entire space at once. Instead, it checks whether equations are satisfied at multiple locations (points) within the space. It's like solving multiple problems on an exam.

```python
# Interior region: 5000 points inside the box (for equation checking)
x_interior = torch.rand(N_INTERIOR, 1, device=device)
y_interior = torch.rand(N_INTERIOR, 1, device=device)
x_interior.requires_grad_(True)  # Enable differentiation (essential!)
y_interior.requires_grad_(True)
```

**Why requires_grad_(True) is important:**
- PINN needs to differentiate with respect to inputs (x, y) (compute du/dx)
- requires_grad must be True to use torch.autograd.grad for differentiation
- Without this setting, differentiation is impossible and PINN won't work

### 5.7 [5] Loss Function (Core of PINN)

The loss function consists of 3 parts:

#### (a) compute_pde_residuals: PDE Residual Computation

This function is the **most important part** of PINN.

**What is a Residual?** When the equation is written as "left side - right side = 0", the residual is how far the network's prediction deviates from 0. A residual of 0 means the equation is perfectly satisfied (correct answer!).

```python
# Step 1: Network prediction
inputs = torch.cat([x, y], dim=1)  # Combine x, y into [N, 2] matrix
outputs = model(inputs)            # Pass through network -> [N, 3] (u, v, p)
u = outputs[:, 0:1]  # Velocity x
v = outputs[:, 1:2]  # Velocity y
p = outputs[:, 2:3]  # Pressure
```

```python
# Step 2: 1st-order partial derivatives (automatic differentiation)
u_x = torch.autograd.grad(u, x, ...)  # du/dx
u_y = torch.autograd.grad(u, y, ...)  # du/dy
```

**torch.autograd.grad parameter explanation:**
- `grad_outputs=torch.ones_like(u)`: Weights needed for vector differentiation (always use ones)
- `create_graph=True`: Keep computation graph for 2nd-order derivatives (without this, d²u/dx² is impossible!)
- `retain_graph=True`: Keep graph when differentiating the same graph multiple times
- `[0]`: grad function returns a tuple, extract the first element

```python
# Step 3: 2nd-order partial derivatives (differentiate 1st-order results again)
u_xx = torch.autograd.grad(u_x, x, ...)  # d²u/dx²
u_yy = torch.autograd.grad(u_y, y, ...)  # d²u/dy²
```

```python
# Step 4: Navier-Stokes residual computation
continuity = u_x + v_y                                    # Continuity equation
momentum_x = u*u_x + v*u_y + p_x/RHO - NU*(u_xx + u_yy)  # x momentum
momentum_y = u*v_x + v*v_y + p_y/RHO - NU*(v_xx + v_yy)  # y momentum
```

Each residual approaching 0 means the equation is satisfied.

#### (b) compute_boundary_loss: Boundary Condition Loss

Computes how far the network's prediction is from target values at boundaries (walls). This is the same as standard MSE (Mean Squared Error) loss in deep learning.

```python
loss_u = torch.mean((u_pred - u_target) ** 2)  # Mean of (prediction - target)^2
loss_v = torch.mean((v_pred - v_target) ** 2)
return loss_u + loss_v
```

#### (c) total_loss: Total Loss

```python
total = loss_pde + 10.0 * loss_bc
```

- **Loss_PDE**: Are physics equations satisfied in the interior? ("Does it follow physics laws?")
- **Loss_BC**: Are boundary conditions satisfied at walls? ("Does it respect wall conditions?")
- **Weight 10.0**: Apply boundary conditions more strongly (must be accurate at walls for physically meaningful solutions)

### 5.8 [6] Training Loop

```python
for epoch in range(EPOCHS):
    optimizer.zero_grad()       # 1. Clear gradients
    loss, ... = total_loss(...) # 2. Compute loss
    loss.backward()             # 3. Backpropagation (compute gradients)
    optimizer.step()            # 4. Update weights
```

1. **Clear gradients**: Remove previous epoch's gradients (prevent accumulation)
2. **Compute loss**: Calculate how far current prediction is from the correct answer
3. **Backpropagation**: Differentiate loss with respect to each weight to compute gradients ("which direction to change weights to reduce loss")
4. **Update weights**: Optimizer adjusts weights using gradients (roughly: weight = weight - learning_rate * gradient)

Repeating this 5000 times makes the network gradually approach the correct answer.

### 5.9 [7] Visualization

- Create a 50×50 grid and predict values across the entire space
- Output 4 plots: Velocity U, Velocity V, Pressure P, Streamlines
- Save results as PNG files

### 5.10 [8] Summary

Print training results (time, loss, device, etc.) and guide next steps.

---

## 6. Interpreting Results

### 6.1 Training Log Interpretation

```
Epoch     0/5000 | Total Loss: 1.758319e+01 | PDE: 6.015871e-01 | BC: 1.698160e+00 | Time: 0.4s
```

- **Total Loss**: Total loss (PDE + 10*BC). Starts at 17.6
- **PDE**: Equation residual loss. Lower = better physics law compliance
- **BC**: Boundary condition loss. Lower = more accurate at walls
- **Time**: Elapsed time

Loss decreased steadily during training:
```
Epoch     0: 17.58  (initial, random weights)
Epoch   500: 0.874  (rapid decrease)
Epoch  2500: 0.233
Epoch  4999: 0.153  (converged - training successful!)
```

### 6.2 Flow Field Results (ldc2d_result.png)

| Subplot | Observation | Physical Meaning |
|---------|-------------|------------------|
| **Velocity U** | Max at top wall (+1.0), reverse flow at bottom (-0.125) | Lid moving right drives flow. Reverse flow at bottom |
| **Velocity V** | Upward on left (+0.20), downward on right (-0.12) | Clockwise circulation (up on left, down on right) |
| **Pressure P** | High pressure top-right, low pressure bottom-left | High where fluid hits wall (top-right), low where it leaves (bottom-left) |
| **Streamlines** | Large clockwise vortex in center, small vortices at bottom corners | Typical Re=100 flow pattern (primary vortex + corner vortices) |

**Conclusion**: PINN predicted a physically accurate flow field. Clockwise primary vortex, corner vortices, and pressure distribution are all correct.

### 6.3 Loss Curve (ldc2d_loss.png)

The loss curve is plotted on a log scale:
- Rapid initial decrease, gradually leveling off (typical training pattern)
- Stabilizes after ~3500 epochs, reaching convergence
- No divergence, stable convergence = training success

---

## 7. Parameter Experimentation Guide

Try changing these parameters in `tutorial_pinn_ldc2d.py`:

### 7.1 Improving Accuracy

```python
# More epochs for better accuracy
EPOCHS = 10000        # 5000 -> 10000

# Larger network for more expressiveness
model = FullyConnected(
    layer_size=100,    # 50 -> 100
    num_layers=7,      # 5 -> 7
)
```

### 7.2 Different Reynolds Numbers

```python
# Lower Reynolds number (laminar, easier problem)
NU = 0.1    # Re = 10

# Higher Reynolds number (near turbulent, harder problem)
NU = 0.001  # Re = 1000
```

### 7.3 Adjusting Learning Speed

```python
# Faster initial convergence
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # 1e-3 -> 1e-2

# More stable training
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # 1e-3 -> 5e-4
```

### 7.4 Adjusting Loss Weights

```python
# Apply boundary conditions more strongly
total = loss_pde + 50.0 * loss_bc   # 10.0 -> 50.0

# Apply PDE more strongly
total = 5.0 * loss_pde + loss_bc    # Increase PDE weight
```

---

## 8. Next Learning Steps

### Step 1: Deepen Current Tutorial (1-2 days)
- [ ] Increase EPOCHS to 10000 and check accuracy improvement
- [ ] Change NU to experiment with different Reynolds numbers
- [ ] Change network size (layer_size, num_layers)
- [ ] Add learning rate scheduler

### Step 2: Try Other PDE Problems (3-5 days)
- [ ] **Darcy Flow**: Simpler equation to establish PINN fundamentals
- [ ] **Helmholtz equation**: Wave-related problem
- [ ] **Heat transfer equation**: Temperature distribution prediction

### Step 3: PhysicsNeMo Advanced Features (1 week)
- [ ] **FNO (Fourier Neural Operator)**: Data-driven learning
- [ ] **Model save/load**: Reuse trained models
- [ ] **ONNX export**: Inference optimization

### Recommended Learning Resources

| Resource | URL | Description |
|----------|-----|-------------|
| PhysicsNeMo Official Docs | https://nvidia.github.io/PhysicsNeMo/ | API reference |
| PhysicsNeMo GitHub | https://github.com/NVIDIA/physicsnemo | Example code |
| PINN Original Paper | https://arxiv.org/abs/1711.10561 | Theoretical background |
| PyTorch Autograd Tutorial | https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html | Automatic differentiation basics |

---

## 9. Troubleshooting (FAQ)

### Q: CUDA Out of Memory error
**A**: Reduce batch size:
```python
N_INTERIOR = 2000  # 5000 -> 2000
N_BOUNDARY = 100   # 200 -> 100
```

### Q: Training doesn't converge
**A**: Try the following:
1. Lower learning rate: `lr=1e-4`
2. Increase network size: `layer_size=100, num_layers=7`
3. Increase epochs: `EPOCHS=10000`
4. Change activation function: `activation_fn="silu"`

### Q: Results look physically wrong
**A**: Adjust loss weights:
```python
# Apply boundary conditions more strongly
total = loss_pde + 100.0 * loss_bc
```

### Q: Want to run on CPU only
**A**: Modify this line:
```python
device = torch.device("cpu")  # "cuda" -> "cpu"
```
(However, training will be 10-100x slower)

### Q: Want to reuse the trained model
**A**: Load the saved model:
```python
model = FullyConnected(in_features=2, out_features=3, layer_size=50, num_layers=5)
model.load_state_dict(torch.load("E:/physicsnemo_env/tutorial_results/ldc2d_model.pth"))
model.eval()
```

---

## Summary

Through this tutorial, we learned:

1. **Using PhysicsNeMo models**: Create FullyConnected neural network and train on GPU
2. **Understanding PINN principles**: Compute PDE residuals using automatic differentiation (autograd)
3. **Navier-Stokes equations**: Implement continuity equation + momentum equations
4. **Boundary condition handling**: Apply velocity conditions at walls
5. **Result visualization**: Analyze velocity field, pressure field, streamlines
6. **GPU-accelerated training**: Completed training in 324 seconds on Quadro P4000
