# Utility Tutorials

> Practical ML workflow tools for PhysicsNeMo projects

---

## Overview

This directory contains utility tutorials that demonstrate practical ML workflow techniques:
configuration management, checkpoint/resume, and experiment tracking.

---

## Tutorials

| # | Tutorial | Description | Script |
|---|----------|-------------|--------|
| 1 | [Hydra Configuration](hydra_config/) | Config management with Hydra (YAML, CLI override, composition) | `hydra_config.py` |
| 2 | [Checkpoint Management](checkpoint_management/) | Save, resume, and manage training experiments | `checkpoint_management.py` |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Hydra Config** | YAML-based config, CLI override, multi-run sweeps, composition |
| **Checkpoint Mgmt** | Full state save (model + optimizer + scheduler), resume, best model tracking |
| **Metadata** | JSON experiment config + results for easy comparison |

---

## Recommended Learning Order

1. **Hydra Configuration** — Manage hyperparameters systematically (YAML, CLI, composition)
2. **Checkpoint Management** — Save/resume training, track best model, experiment metadata
