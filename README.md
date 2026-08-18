# PhysicsNeMo Tutorials

> Physics simulation AI model tutorials using NVIDIA PhysicsNeMo

---

## Overview

This repository provides tutorials for solving various physics simulation problems using the NVIDIA PhysicsNeMo framework. Each tutorial includes a Python script, detailed guides (Korean/English), and result images.

---

## Folder Structure

```
physicsnemo-tutorials/
│
├── docs/                  # Project-level documentation (installation, system inspection, guides)
├── pinn/                  # PINN (Physics-Informed Neural Network) tutorials
├── neural_operators/      # Neural Operator (FNO, AFNO, Transolver, U-Net, SRRN) tutorials
├── mesh_based/            # Mesh-based learning (MeshGraphNet, NACA Airfoil) tutorials
├── applications/          # Application domains (topology optimization, etc.)
├── comparisons/           # Multi-model performance comparisons
├── generative/            # Generative AI for physics (conditional diffusion)
├── optimization/          # AI-based design optimization (inverse design)
├── uncertainty/           # Uncertainty quantification (deep ensemble, OOD detection)
├── transfer_learning/     # Transfer learning (pre-train → fine-tune)
└── utils/                 # Utility scripts
```

---

## Tutorial List

### PINN (Physics-Informed Neural Networks)

Equation-based learning — trains without labeled data using PDE residual loss

| Tutorial | Description | Script |
|----------|-------------|--------|
| [Burgers Equation](pinn/burgers/) | 1D viscous Burgers (shock wave) | `burgers.py` |
| [Lid-Driven Cavity 2D](pinn/lid_driven_cavity/) | 2D box flow (Navier-Stokes) | `pinn_ldc2d.py` |
| [Conjugate Heat Transfer 2D](pinn/conjugate_heat_transfer/) | Solid-fluid heat transfer | `pinn_cht2d.py` |
| [Electrostatics](pinn/electrostatics/) | Electrostatics (Poisson equation) | `pinn_electrostatics.py` |
| [Plane Stress](pinn/plane_stress/) | Plane stress (linear elasticity) | `pinn_planestress.py` |
| [Inverse Problem](pinn/inverse_problem/) | Inverse problem (parameter estimation) | `inverse_pinn.py` |
| [Adaptive Sampling (RAR)](pinn/adaptive_sampling/) | Adaptive collocation (RAR, 2D Poisson) | `adaptive_sampling.py` |
| [Reaction-Diffusion](pinn/reaction_diffusion/) | Gray-Scott (multi-variable coupled PDE) | `reaction_diffusion.py` |

### Neural Operators

Data-driven learning — surrogate models that approximate PDE solutions via supervised learning

| Tutorial | Description | Script |
|----------|-------------|--------|
| [FNO - Darcy Flow](neural_operators/fno/darcy_flow/) | Permeability → pressure prediction | `fno_darcy.py` |
| [FNO - Navier-Stokes](neural_operators/fno/navier_stokes/) | Time-dependent flow (vorticity) | `fno_navier_stokes.py` |
| [FNO - Heat Conduction](neural_operators/fno/heat_conduction/) | Heat conduction solution | `fno_heatconduction.py` |
| [FNO - Allen-Cahn](neural_operators/fno/allen_cahn/) | Phase separation (FNO paper benchmark) | `allen_cahn.py` |
| [FNO - Wave Equation](neural_operators/fno/wave/) | 2D wave propagation (2nd-order time) | `wave.py` |
| [AFNO - Darcy Flow](neural_operators/afno/) | Adaptive FNO (Darcy) | `afno_darcy.py` |
| [Transolver - Darcy Flow](neural_operators/transolver/) | Physics Attention (Darcy) | `transolver_darcy.py` |
| [U-Net - Darcy Flow](neural_operators/unet/) | 3D CNN Encoder-Decoder (Darcy) | `unet_darcy.py` |
| [SRRN - Super Resolution](neural_operators/srrn/) | Low-res → high-res upscaling | `srrn_superres.py` |
| [DeepONet - Burgers](neural_operators/deeponet/) | Operator learning (branch-trunk) | `deeponet_burgers.py` |
| [PINO - Darcy Flow](neural_operators/pino/) | FNO + PDE residual (hybrid data + physics) | `pino_darcy.py` |
| [FNO - Zero-Shot Resolution](neural_operators/fno/zero_shot/) | Train 32×32, test 64/128 (no retrain) | `zero_shot.py` |

### Mesh-Based Learning

Irregular mesh / complex geometry processing

| Tutorial | Description | Script |
|----------|-------------|--------|
| [MeshGraphNet](mesh_based/meshgraphnet/) | Graph neural network (mesh learning) | `meshgraphnet.py` |
| [NACA Airfoil](mesh_based/naca_airfoil/) | Aerodynamic analysis (flow prediction) | `naca_airfoil.py` |

### Applications

Specific application domains

| Tutorial | Description | Script |
|----------|-------------|--------|
| [Topology Optimization](applications/topology_optimization/) | Topology optimization (Diffusion) | `topodiff.py` |

### Comparisons

Multi-model performance comparison

| Report | Models Compared | Script |
|--------|-----------------|--------|
| [Darcy Flow Model Comparison](comparisons/report_en.md) | FNO vs Transolver vs U-Net | `compare_darcy_models.py` |
| [PINN vs FNO Comparison](comparisons/report_pinn_vs_fno_en.md) | PINN vs FNO (Burgers) | `compare_pinn_vs_fno.py` |

### Generative AI

Generative modeling — learn solution distributions and generate diverse samples

| Tutorial | Description | Script |
|----------|-------------|--------|
| [Conditional Diffusion](generative/conditional_diffusion/) | DDPM for stochastic Darcy (1→N generation) | `conditional_diffusion.py` |

### Optimization

AI-based inverse design — optimize inputs to achieve desired performance

| Tutorial | Description | Script |
|----------|-------------|--------|
| [Differentiable Design](optimization/differentiable_design/) | Surrogate backprop for airfoil shape design | `differentiable_design.py` |

### Uncertainty Quantification

Know when the model is wrong — safety-critical ML for physics

| Tutorial | Description | Script |
|----------|-------------|--------|
| [Deep Ensemble](uncertainty/deep_ensemble/) | N=5 CNNs, OOD detection, calibration | `deep_ensemble.py` |

### Transfer Learning

Pre-train on abundant source data, fine-tune on scarce target data — the most practical ML technique

| Tutorial | Description | Script |
|----------|-------------|--------|
| [FNO Transfer Learning](transfer_learning/transfer_fno/) | Pre-train (coarse k) → fine-tune (fine k): freeze vs full FT | `transfer_fno.py` |

---

## Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU (Compute Capability 3.5+, 8GB+ VRAM recommended)
- **Python**: 3.10
- **PyTorch**: 2.x with CUDA 11.8
- **PhysicsNeMo**: 1.3.0

### Installation

For detailed installation instructions, see [docs/installation_manual_en.md](docs/installation_manual_en.md).

```cmd
:: Create and activate virtual environment
python -m venv physicsnemo_env
physicsnemo_env\Scripts\activate

:: Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install PhysicsNeMo
pip install nvidia-physicsnemo
```

### Running a Tutorial

```cmd
:: Example: Run FNO Darcy Flow tutorial
cd E:\physicsnemo-tutorials\neural_operators\fno\darcy_flow
python fno_darcy.py
```

---

## Tutorial Folder Structure

Each tutorial folder contains the following files:

```
tutorial_name/
├── script.py          # Python tutorial script
├── guide_en.md        # English detailed guide
├── guide_ko.md        # Korean detailed guide
└── results/           # Result images and model files
    ├── *_result.png
    ├── *_loss.png
    └── *.pth
```

### Adding a New Tutorial

1. Create a new subfolder under the appropriate group folder
2. Place the script, `guide_en.md`, and `guide_ko.md`
3. Save result images in a `results/` subfolder

---

## Language Support

- 🇺🇸 **English**

---

## Environment

| Component | Version |
|-----------|---------|
| OS | Windows 11 (64-bit) |
| GPU | NVIDIA Quadro P4000 (8GB) |
| Python | 3.10 |
| PyTorch | 2.7.1+cu118 |
| PhysicsNeMo | 1.3.0 |
| NVIDIA Driver | 582.70 |

---
