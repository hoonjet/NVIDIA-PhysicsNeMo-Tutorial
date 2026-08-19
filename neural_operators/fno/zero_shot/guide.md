# FNO Zero-Shot Resolution Generalization

> **Category**: `neural_operators/fno/zero_shot/` — Resolution Invariance  
> **Paradigm**: Train on one resolution, test on others WITHOUT retraining  
> **Model**: FNO (spectral) vs U-Net (convolutional)

---

## 1. What Makes This Tutorial Unique?

This is the **only tutorial that demonstrates FNO's resolution invariance** — its killer feature. All other FNO tutorials train and test on the same resolution. This tutorial trains on 32×32 and tests on 32×32, 64×64, and 128×128 with the **same model, same weights, no retraining**.

| Aspect | All Other FNO Tutorials | **This Tutorial** |
|--------|--------------------------|-------------------|
| **Train resolution** | 32×32 | 32×32 |
| **Test resolution** | 32×32 (same) | **32, 64, 128 (different!)** |
| **Retraining needed?** | N/A | **No — zero-shot** |
| **Comparison** | None | **FNO vs U-Net** |

### vs. SRRN (Super Resolution)
- SRRN needs a **separate model** to upscale low-res → high-res
- This tutorial uses the **same model** at any resolution — no separate upscaler
- SRRN: "train a model to convert resolutions" / FNO Zero-Shot: "same model works at any resolution"

### vs. U-Net
- U-Net's conv kernels are tied to spatial scale → **cannot generalize** to new resolutions
- FNO's Fourier weights are mode-based → **automatically adapt** to any resolution
- This tutorial directly compares both on the same task

---

## 2. Why FNO Is Resolution-Independent

### FNO: Spectral (Fourier) Layers
```
Input [H×W] → FFT → multiply low-frequency modes → IFFT → Output [H×W]
```
- Weights are stored as **Fourier modes** (e.g., 8×8 modes), not pixel kernels
- FFT/IFFT automatically adapt to any input size
- The same 8×8 modes work whether input is 32×32 or 128×128
- **Key insight**: the model learns a continuous operator in spectral space, not a discrete pixel mapping

### U-Net: Convolutional Layers
```
Input [H×W] → Conv2d (3×3 kernel) → ... → Output [H×W]
```
- Weights are **fixed-size kernels** (e.g., 3×3 pixels)
- Downsampling (stride=2) assumes specific spatial dimensions
- If input size changes, downsampling/upsampling dimensions mismatch
- **Cannot generalize**: the model is tied to the training resolution

---

## 3. Problem: Darcy Flow at Multiple Resolutions

```
-∇·(k∇p) = 1,   p = 0 on boundary

Train:  k(x) [32×32]  →  p(x) [32×32]
Test:   k(x) [32×32]  →  p(x) [32×32]   (same resolution)
        k(x) [64×64]  →  p(x) [64×64]   (2× higher, zero-shot)
        k(x) [128×128] → p(x) [128×128] (4× higher, zero-shot)
```

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| `[1] Data Generation` | Darcy at 32, 64, 128 resolutions |
| `[2] Architecture` | FNO (spectral) + U-Net (convolutional) for comparison |
| `[3] Training` | Both models trained on 32×32 only (200 epochs) |
| `[4] Zero-Shot Test` | Feed 32, 64, 128 inputs to both models — NO retraining |
| `[5] Visualization` | Loss, resolution bar chart, visual comparison, architecture explanation |

---

## 5. Key Results

### 5.1 FNO: Zero-Shot Success
- FNO trained on 32×32 produces accurate predictions on 64×64 and 128×128
- Error increases slightly at higher resolutions (expected — finer features are harder)
- But the model **never saw** these resolutions during training

### 5.2 U-Net: Zero-Shot Failure
- U-Net's downsampling/upsampling assumes fixed spatial dimensions
- At 64×64 and 128×128, the encoder-decoder dimensions mismatch
- U-Net either produces wrong-sized output or crashes entirely

### 5.3 Practical Implication
- **Train on cheap (low-res) data, deploy on expensive (high-res) data**
- No need to retrain when mesh resolution changes
- One model serves all resolutions — massive cost savings

---

## 6. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\neural_operators\fno\zero_shot
python zero_shot.py
```

### Result Files
| File | Description |
|------|-------------|
| `results/zero_shot_loss.png` | Training loss: FNO vs U-Net on 32×32 |
| `results/zero_shot_resolution.png` | Bar chart: L2 error at each resolution |
| `results/zero_shot_result.png` | Visual: input, truth, FNO prediction, error at 32/64/128 |
| `results/zero_shot_explanation.png` | Architecture comparison: why FNO works, U-Net doesn't |

---

## 7. Key Concepts Learned

1. **Spectral Discretization Invariance**: FNO operates in Fourier space. The learned weights are Fourier mode coefficients, not pixel-level kernels. FFT/IFFT handle any input size automatically.

2. **Zero-Shot Transfer**: No fine-tuning, no retraining. The same model weights produce valid predictions at any resolution. This is impossible for CNN-based models.

3. **Mode-Based vs Pixel-Based**: FNO weights are `[in_ch, out_ch, modes1, modes2]` (Fourier modes). U-Net weights are `[in_ch, out_ch, kernel_h, kernel_w]` (pixel kernels). Modes are resolution-independent; kernels are not.

4. **Cost Strategy**: Generate cheap low-resolution training data (fast CFD), train FNO, then deploy on high-resolution meshes. The model generalizes for free.

5. **FNO's Killer Feature**: This is the single most important advantage of FNO over all other architectures (PINN, U-Net, Transolver, DeepONet). No other model in this repository can do zero-shot resolution transfer.

6. **Limitation**: Zero-shot works best when the physics at different resolutions are similar (same PDE, same domain). If the physics changes qualitatively at higher resolution (e.g., new turbulence), zero-shot may degrade.

---

## 8. Comparison with Other Tutorials

| Feature | FNO (Darcy) | SRRN (Super-Res) | U-Net (Darcy) | **This Tutorial** |
|---------|:-----------:|:----------------:|:-------------:|:------------------:|
| **Train res** | 32×32 | 16×16 | 32×32 | **32×32** |
| **Test res** | 32×32 | 32×32 | 32×32 | **32, 64, 128** |
| **Retrain?** | N/A | Yes (separate model) | N/A | **No (zero-shot)** |
| **Architecture** | FNO | CNN | CNN | **FNO vs U-Net** |
| **Resolution invariant?** | Not shown | ✗ | ✗ | **✓ (demonstrated)** |

---

## 9. Extensions

- **Super-resolution via FNO**: Train on 16×16, zero-shot predict on 256×256
- **Irregular mesh transfer**: Train on structured grid, test on unstructured mesh
- **Cross-domain transfer**: Train on Darcy, zero-shot on Helmholtz (same operator structure)
- **Multi-resolution training**: Train on mixed resolutions simultaneously for better generalization
- **Mode count analysis**: How many Fourier modes are needed for zero-shot at 4× resolution?

---

## 10. References

- Li et al., "Fourier Neural Operator for Parametric PDEs," ICLR 2021 (Section 3.3: resolution invariance)
- Li et al., "Neural Operator: Graph Kernel Network for PDEs," ICLR 2023 (GNO, generalization analysis)
- Kovachki et al., "Neural Operator: Learning Maps Between Function Spaces," arXiv 2108.08481, 2021
