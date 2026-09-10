# Applications Tutorials

> Specific application domains — design optimization and industrial applications

---

## Overview

This folder contains tutorials applying PhysicsNeMo to specific industrial application domains. It covers using AI for real design/optimization problems beyond basic model training.

---

## Tutorials

| # | Tutorial | Description | Script |
|---|----------|-------------|--------|
| 1 | [Topology Optimization](topology_optimization/) | Topology optimization (Diffusion-based) | `topodiff.py` |
| 2 | [Active Learning](active_learning/) | Uncertainty-based selective sampling for efficient surrogate training | `active_learning.py` |
| 3 | [ROM Autoencoder](rom_autoencoder/) | Compress 4096-dim field → 8-dim latent (512× compression, PCA mode discovery) | `rom_autoencoder.py` |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Application area** | Design optimization, inverse design |
| **Learning method** | Data-driven (supervised learning) |
| **Core mechanism** | Diffusion model |
| **Input/Output** | Boundary conditions → optimal shape |
