# Differentiable Design Optimization

> **Category**: `optimization/` — AI-Based Design Optimization  
> **Paradigm**: Inverse design via gradient backpropagation through a surrogate model  
> **Model**: 1D CNN surrogate + input-space gradient descent

---

## 1. What Makes This Tutorial Unique?

This is the **only inverse design tutorial** in the repository. Every other tutorial solves the **forward problem**: given input → predict output. This tutorial solves the **inverse design problem**: given desired output → optimize input (design variables).

| Aspect | All Other Tutorials | This Tutorial |
|--------|---------------------|---------------|
| **Direction** | Forward (input → output) | **Inverse (output → input)** |
| **What's optimized** | Model weights | **Input (design shape)** |
| **Gradient flows to** | Network parameters | **Input tensor** |
| **Goal** | Predict physics | **Design shape to achieve target performance** |
| **Iteration** | Training epochs | **Design optimization steps** |

### vs. PINN Inverse Problem
- PINN Inverse Problem estimates **scalar PDE parameters** (e.g., viscosity ν) from data
- This tutorial optimizes a **shape function** (airfoil surface, 64 points) via gradient descent
- Different optimization target: scalar vs. function; different method: PDE residual vs. surrogate backprop

### vs. Topology Optimization
- Topology Optimization uses a **diffusion model** to generate designs (generative approach)
- This tutorial uses **gradient-based optimization** through a differentiable surrogate (analytical approach)
- Different methodology, different problem formulation

---

## 2. Problem: Aerodynamic Shape Optimization

### Forward Problem (Surrogate Training)
```
Input:  Airfoil shape y(x)  [64 surface points]
Output: Pressure coefficient Cp(x)  [64 points]
Model:  1D CNN (encoder-decoder)
```

### Inverse Design Problem
```
Given:  Target Cp distribution (desired aerodynamic performance)
Find:   Airfoil shape that produces this Cp
Method: Gradient descent on shape via backprop through frozen surrogate
```

### Why Surrogate-Based?
- Direct CFD simulation: minutes to hours per shape evaluation
- Surrogate CNN: milliseconds per evaluation
- Optimization requires hundreds of evaluations → surrogate makes it feasible
- The surrogate is **differentiable** → gradients flow from Cp loss back to shape input

---

## 3. Method: Input-Space Gradient Descent

### 3.1 Train Surrogate (Standard)
```
loss = MSE(CNN(shape), true_Cp)
gradient → update CNN weights (standard training)
```

### 3.2 Freeze Surrogate, Optimize Input
```
# Freeze model weights
for p in surrogate.parameters():
    p.requires_grad = False

# Make input a leaf tensor with gradient
design_shape = initial_shape.clone().requires_grad_(True)

# Optimize input
pred_cp = surrogate(design_shape)
loss = MSE(pred_cp, target_cp) + smoothness + edge_constraints
loss.backward()  # gradient flows to design_shape, NOT weights
optimizer.step()  # updates design_shape, NOT weights
```

### 3.3 Multi-Objective Loss
```
L = L_match + λ₁ · L_smooth + λ₂ · L_edge

L_match:  MSE(predicted Cp, target Cp) — primary objective
L_smooth: Mean|Δy| — penalize non-physical rough shapes
L_edge:   y(0)² + y(1)² — enforce closed leading/trailing edges
```

### 3.4 Projected Gradient Descent
After each step, clamp the design to a feasible range:
```
design_shape.clamp_(min=feasible_min, max=feasible_max)
```
This ensures the shape stays physically valid (non-negative thickness).

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| `[1] Data Generation` | NACA parameterization + thin airfoil theory for synthetic Cp |
| `[2] Surrogate CNN` | 1D encoder-decoder (Conv1d + ConvTranspose1d) |
| `[3] Train Surrogate` | Standard supervised learning: shape → Cp |
| `[4] Inverse Design` | Freeze model, optimize input via Adam, 500 steps |
| `[5] Evaluate` | Compare designed shape & Cp to ground truth |
| `[6] Visualization` | Loss, convergence, shape/Cp comparison, multi-target |

---

## 5. Key Results

### 5.1 Surrogate Accuracy
The CNN learns the shape → Cp mapping accurately, enabling fast gradient computation.

### 5.2 Inverse Design Convergence
- **Cp matching loss** decreases monotonically — the designed shape produces Cp closer to target
- **Shape L2 difference** also decreases — the optimizer recovers a shape similar to ground truth

### 5.3 Shape Recovery
Starting from a random initial shape, the optimizer modifies it to match the target Cp. The recovered shape is close (but not identical) to the ground truth — this is expected because the inverse problem is ill-posed (multiple shapes can produce similar Cp).

---

## 6. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\optimization\differentiable_design
python differentiable_design.py
```

### Result Files
| File | Description |
|------|-------------|
| `results/surrogate_loss.png` | Surrogate training/test loss |
| `results/design_convergence.png` | Cp matching + shape recovery convergence |
| `results/design_result.png` | Shape & Cp comparison (initial vs designed vs target) |
| `results/design_multitarget.png` | 3 inverse design cases |

---

## 7. Key Concepts Learned

1. **Input Gradient**: Standard training backprops to weights; inverse design backprops to **input**. The same `loss.backward()` call computes gradients for a different target.

2. **Surrogate-Based Optimization**: Replace expensive simulations with a fast, differentiable neural network. The key insight: if the surrogate is differentiable, you can optimize the **input** directly.

3. **Ill-Posedness**: The inverse design problem has multiple solutions (many shapes produce similar Cp). Regularization (smoothness, edge constraints) guides toward physically valid solutions.

4. **Projected Gradient**: After each gradient step, project to a feasible set. This handles constraints that gradients alone cannot enforce.

5. **Multi-Objective Optimization**: Real design problems have multiple objectives (match performance + maintain feasibility). Weighted sum is the simplest approach.

6. **End-to-End Differentiability**: The entire pipeline (shape → CNN → Cp → loss) is differentiable. No finite differences, no adjoint methods — just autograd.

---

## 8. Comparison with Other Tutorials

| Feature | PINN Inverse | TopoDiff | **This Tutorial** |
|---------|:----------:|:--------:|:------------------:|
| **What's optimized** | Scalar parameter | Design shape | **Design shape (function)** |
| **Method** | PDE residual + data | Diffusion model | **Gradient backprop** |
| **Surrogate** | PDE itself (PINN) | Diffusion U-Net | **CNN** |
| **Differentiable** | ✓ (autograd) | ✗ (sampling) | **✓ (autograd)** |
| **Constraints** | Boundary conditions | Volume fraction | **Smoothness + edges** |
| **Output** | Parameter value | Binary mask | **Continuous shape** |

---

## 9. Extensions

- **Adversarial regularization**: Use a discriminator to ensure designed shapes look realistic
- **Latent space optimization**: Optimize in a compressed latent space (VAE/GAN) for better feasibility
- **Multi-fidelity**: Combine cheap surrogate + expensive CFD for final validation
- **Topology optimization**: Extend from shape optimization to topology (add/remove material)
- **Reinforcement learning**: Replace gradient descent with RL for non-differentiable objectives
- **Active learning**: Retrain surrogate on newly designed shapes for better accuracy

---

## 10. References

- Chen et al., "B-Spline Neural Networks for Airfoil Design," AIAA Journal 2023
- Li et al., "Differentiable Physics Surrogates for Inverse Design," NeurIPS 2022
- Sun & Ma, "Surrogate-Based Aerodynamic Shape Optimization," Progress in Aerospace Sciences 2023
- Hoyer et al., "Neural Reparameterization Improves Structural Optimization," ICLR 2019
