# AI-Based Design Optimization Tutorials

> Inverse design — optimize inputs to achieve desired physical performance

---

## Overview

Optimization tutorials solve the **inverse design problem**: given a desired output (e.g., target pressure distribution), find the optimal input (e.g., airfoil shape). This is fundamentally different from all forward-prediction tutorials. The method uses gradient backpropagation through a differentiable surrogate model to optimize the **input** directly.

---

## Tutorials

| # | Tutorial | Method | Script |
|---|----------|--------|--------|
| 1 | [Differentiable Design](differentiable_design/) | Surrogate backprop (input gradient) | `differentiable_design.py` |
| 2 | [Multi-Objective Pareto](multi_objective/) | Weighted sum + Pareto front exploration | `multi_objective.py` |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Paradigm** | Inverse design (output → input) |
| **Core mechanism** | Gradient backprop w.r.t. INPUT |
| **What's optimized** | Design variables (shape, not weights) |
| **Surrogate** | Differentiable neural network |
| **Constraints** | Smoothness, edge conditions, feasibility |

---

## vs. Other Categories

| Feature | PINN / Neural Operators | **Optimization** |
|---------|:----------------------:|:----------------:|
| **Direction** | Forward (input → output) | **Inverse (output → input)** |
| **What's optimized** | Model weights | **Input (design shape)** |
| **Goal** | Predict physics | **Design for target performance** |
| **Gradient flows to** | Network parameters | **Input tensor** |
