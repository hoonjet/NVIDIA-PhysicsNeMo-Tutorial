# Application Tutorials

> Specific application domains — real-world engineering problems

---

## Overview

This folder contains tutorials for specific application domains such as structural optimization, design exploration, and other engineering problems that go beyond standard PDE solving.

---

## Tutorials

| # | Tutorial | Description | Script |
|---|----------|-------------|--------|
| 1 | [Topology Optimization](topology_optimization/) | Topology optimization using diffusion-based generative model | `topodiff.py` |
| 2 | [Active Learning](active_learning/) | Uncertainty-based selective sampling for efficient surrogate training | `active_learning.py` |

---

## Key Features

| Feature | Topology Optimization |
|---------|----------------------|
| **Problem** | Optimize material distribution under constraints |
| **Method** | Diffusion model (generative AI) |
| **Input** | Boundary conditions + loads |
| **Output** | Optimal material distribution |
| **Training data** | Required (optimized solutions from FEM) |
| **Inference** | Fast (generates design in seconds) |
