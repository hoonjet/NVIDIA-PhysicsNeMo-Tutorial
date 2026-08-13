# Generative AI for Physics Tutorials

> Generative modeling — learn solution distributions and generate diverse samples

---

## Overview

Generative AI tutorials go beyond deterministic prediction. Instead of learning a single input → output mapping, these models learn the **distribution** of solutions and can **generate diverse samples** from it. This is essential for stochastic PDEs, uncertainty quantification, and creative design exploration.

---

## Tutorials

| # | Tutorial | Method | Script |
|---|----------|--------|--------|
| 1 | [Conditional Diffusion](conditional_diffusion/) | DDPM (200-step reverse diffusion) | `conditional_diffusion.py` |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Paradigm** | Generative (1 input → N outputs) |
| **Core mechanism** | Forward/reverse diffusion process |
| **Output** | Multiple plausible solutions |
| **Uncertainty** | Free (ensemble std of generated samples) |
| **Stochastic PDEs** | ✓ (naturally handles solution distributions) |

---

## vs. Other Categories

| Feature | PINN / Neural Operators | **Generative** |
|---------|:----------------------:|:--------------:|
| **Input → Output** | 1 → 1 (deterministic) | **1 → N (stochastic)** |
| **What it learns** | Point estimate | **Conditional distribution** |
| **Solution diversity** | ✗ | **✓** |
| **Stochastic PDEs** | ✗ | **✓** |
