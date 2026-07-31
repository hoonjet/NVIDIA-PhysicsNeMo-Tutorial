# PhysicsNeMo U-Net Tutorial Guide

> **Date**: 2026-07-10  
> **Environment**: PhysicsNeMo 1.3.0, PyTorch 2.7.1+cu118, CUDA (Quadro P4000 8GB)

---

## 1. Tutorial Overview

| Item | Details |
|------|---------|
| **Problem** | Darcy Flow (permeability → pressure prediction) |
| **Learning method** | Data-driven (supervised learning) |
| **Neural network** | PhysicsNeMo U-Net (2-level, skip connections) |
| **Parameters** | 417,281 |
| **Training time** | ~146 seconds (300 epochs) |
| **Final loss** | 0.0065 (converged from 1.18) |
| **Resolution transfer** | Train on 32×32 → test on 64×64 possible |

### Files

| File | Description |
|------|-------------|
| `tutorial_unet_darcy.py` | Main tutorial script |
| `tutorial_results/unet_darcy_result.png` | Prediction results (32×32 + 64×64) |
| `tutorial_results/unet_darcy_loss.png` | Training loss curve |

---

## 2. What is U-Net?

U-Net is a CNN with **encoder-decoder + skip connection** structure. Originally developed for medical image segmentation, it is widely used as a PDE surrogate model.

### Core Structure

```
Input [1, 32, 32, 4]
  │
  ├─ Encoder Level 0: Conv3d(1→32) + Conv3d(32→32) → [32, 32, 32, 4]
  │    │
  │    └─ skip connection 0 ─────────────────────────┐
  │                                                  │
  ├─ MaxPool3d → [32, 16, 16, 2]                     │
  │    │                                             │
  ├─ Encoder Level 1: Conv3d(32→64) + Conv3d(64→64) │
  │    (Bottleneck) → [64, 16, 16, 2]               │
  │                                                  │
  ├─ ConvTranspose3d → [64, 32, 32, 4]              │
  │    │                                             │
  │    └─ skip connection 1 ──────────────────┐      │
  │                                           │      │
  ├─ concat(skip 0) ←─────────────────────────┼──────┘
  │    │                                      │
  ├─ Decoder: Conv3d(96→32) + Conv3d(32→32)  │
  │    │                                      │
  └─ Final Conv3d(32→1) → output [1, 32, 32, 4]
```

### 3 Key Concepts

1. **Encoder (downsampling)**: Conv + Pool to reduce resolution while capturing context
2. **Decoder (upsampling)**: ConvTranspose to restore resolution with precise position info
3. **Skip connections**: Directly pass encoder features to decoder → preserve details lost during downsampling

### Why "U" Shape?

The encoder goes down and decoder goes up, forming a U shape. Skip connections cross the symmetric structure horizontally.

---

## 3. How to Run

```cmd
cd E:\physicsnemo_env
Scripts\activate
python tutorial_unet_darcy.py
```

View results:
```cmd
start E:\physicsnemo_env\tutorial_results\unet_darcy_result.png
start E:\physicsnemo_env\tutorial_results\unet_darcy_loss.png
```

---

## 4. Key Code Parts

### Model Creation

```python
from physicsnemo.models.unet import UNet

model = UNet(
    in_channels=1,              # Input channels (permeability 1)
    out_channels=1,             # Output channels (pressure 1)
    model_depth=2,              # U-Net levels (2 = shallow model)
    feature_map_channels=[32, 32, 64, 64],  # Channels per level
    num_conv_blocks=2,          # Conv blocks per level
    conv_activation="relu",     # ReLU activation
    pooling_type="MaxPool3d",   # Max pooling
    pool_size=2,                # 2× downsampling
    normalization="groupnorm",  # Group normalization
    gradient_checkpointing=False,
)
```

### Key Parameters

| Parameter | Description | Tutorial Value |
|-----------|-------------|----------------|
| `model_depth` | U-Net levels (deeper = more downsampling) | 2 |
| `feature_map_channels` | Channels per level | [32, 32, 64, 64] |
| `num_conv_blocks` | Conv blocks per level | 2 |
| `pooling_type` | Downsampling method | MaxPool3d |
| `normalization` | Normalization method | groupnorm |

### 3D Conversion (Important)

PhysicsNeMo's U-Net uses **3D convolution (Conv3d)**. Convert 2D data to 3D:

```python
# 2D: [B, 1, 32, 32] → 3D: [B, 1, 32, 32, 4]
k_train = k_train_2d.unsqueeze(-1).expand(-1, -1, -1, -1, DEPTH).contiguous()
```

---

## 5. Results

### Training Loss

| Epoch | Loss | Notes |
|-------|------|-------|
| 0 | 1.178 | Initial |
| 30 | 0.029 | Rapid decrease |
| 150 | 0.0085 | Gradual decrease |
| 299 | 0.0065 | Converged |

### Resolution Transfer Test

| Resolution | Loss | Notes |
|------------|------|-------|
| 32×32 (trained) | 0.0055 | Accurate |
| 64×64 (untrained) | 0.081 | Accuracy decrease but works |

U-Net uses fixed-size convolutions, so it works at untrained resolutions. However, skip connection shapes must match, so resolution must be divisible by `2^(model_depth-1)`.

---

## 6. 5-Model Comparison

| Feature | PINN | FNO | Transolver | MGN | **U-Net** |
|---------|------|-----|------------|-----|-----------|
| Learning method | Equation | Data | Data | Data | **Data** |
| Data structure | Points | Grid | Grid/mesh | Graph | **Grid** |
| Core mechanism | Autograd | Fourier | Attention | Message passing | **Conv** |
| Skip connections | ✗ | ✗ | Residual | ✗ | **✓** |
| Multi-scale | ✗ | Frequency | Slices | Hops | **Pool/Up** |
| Resolution flexibility | ✓ | ✓ | Partial | ✓ | **✓** |
| Irregular mesh | ✓ | ✗ | ✓ | ✓ | **✗** |
| Memory | Low | Medium | High | Medium | **Low** |
| Training speed | Slow | Fast | Medium | Fast | **Fast** |

### When to Use U-Net?

- **Grid-based data** (images, regular meshes)
- **Multi-scale feature extraction** needed (downsampling/upsampling)
- **Simple and fast architecture** desired
- **Detail preservation** important (skip connections help)

---

## 7. Parameter Experimentation Guide

### Deeper Model

```python
model = UNet(
    model_depth=4,  # 2 → 4 (deeper U-Net)
    feature_map_channels=[32, 32, 64, 64, 128, 128, 256, 256],
)
```

### Attention Gates

```python
model = UNet(
    use_attn_gate=True,  # Apply attention to skip connections
    attn_decoder_feature_maps=[64, 32],
    attn_feature_map_channels=[32, 32],
    attn_intermediate_channels=16,
)
```

---

## Summary

Through the U-Net tutorial, we learned:

1. **Encoder-decoder structure**: Downsampling (context) → upsampling (position)
2. **Skip connections**: Preserve details lost during downsampling
3. **Multi-scale processing**: Simultaneous feature extraction at 32×32 and 16×16
4. **3D convolution**: PhysicsNeMo U-Net uses Conv3d (2D→3D conversion needed)
5. **Resolution flexibility**: Works at untrained resolutions (with performance decrease)
6. **5-model comparison**: Strengths/weaknesses of PINN/FNO/Transolver/MGN/UNet
