# Reduced-Order Model via Autoencoder

> **Category**: `applications/` — Model compression / dimensionality reduction
> **Paradigm**: Autoencoder (unsupervised learning)
> **Problem**: Compress 2D heat equation solutions (4096 dims → 8 dims)

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing Tutorials | **THIS (ROM Autoencoder)** |
|--------|-------------------|---------------------------|
| **Goal** | Forward solve (input → field) | **Compress field → latent → reconstruct** |
| **Output** | Full field (4096 dims) | **Latent vector (8 dims)** |
| **Compression** | None | **512× compression** |
| **Interpolation** | Not possible | **Latent space morphing** |
| **Mode discovery** | Not available | **PCA on latent = POD-like modes** |
| **Solution manifold** | Single solution | **Entire solution family** |

### Key Difference: Dimensionality Reduction
All existing tutorials are **forward solvers** — they map inputs to full fields. This tutorial is fundamentally different: it **compresses** high-dimensional PDE solutions into a tiny latent space, enabling:
- Fast storage/transmission (8 numbers instead of 4096)
- Interpolation between solutions (morphing)
- Automatic mode discovery (PCA on latent space)

---

## 2. Physics: 2D Steady-State Heat Equation

```
∇²T = 0  on [0,1]×[0,1]

BC: T = 0       (left, right, bottom)
    T = T_top   (top boundary)
```

**Parameter family**: Vary `T_top` from 50 to 200 → generates 500 different solutions.

This creates a **solution manifold** — a low-dimensional surface in the 4096-dimensional space that all solutions lie on.

---

## 3. Autoencoder Architecture

```
         Encoder                          Decoder
         
4096 ──→ 512 ──→ 128 ──→ 32 ──→ 8     8 ──→ 32 ──→ 128 ──→ 512 ──→ 4096
  │       │       │       │      │     │      │       │       │       │
  Linear  Linear  Linear  Linear       Linear  Linear  Linear  Linear
  SiLU    SiLU    SiLU    (z)          SiLU    SiLU    SiLU    (recon)
  
         ←─── Compression ───→         ←─── Reconstruction ───→
```

| Component | Input dim | Output dim | Activation |
|-----------|-----------|-----------|------------|
| Encoder L1 | 4096 | 512 | SiLU |
| Encoder L2 | 512 | 128 | SiLU |
| Encoder L3 | 128 | 32 | SiLU |
| Encoder L4 | 32 | **8 (latent)** | — |
| Decoder L1 | 8 | 32 | SiLU |
| Decoder L2 | 32 | 128 | SiLU |
| Decoder L3 | 128 | 512 | SiLU |
| Decoder L4 | 512 | **4096 (recon)** | — |

**Compression ratio**: 4096 / 8 = **512×**

---

## 4. Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-3) |
| Scheduler | StepLR (step=100, γ=0.5) |
| Epochs | 300 |
| Batch size | 32 |
| Loss | MSE (reconstruction) |
| Train/Test | 400 / 100 |

---

## 5. Latent Space Analysis

### PCA on Latent Vectors
After training, we extract latent vectors for all 500 samples and apply PCA:

- **PC1**: Dominant mode (correlates with T_top — the physical parameter)
- **PC2**: Secondary mode (captures subtle variations)
- **Top 3 PCs**: Typically >99% variance explained

This is analogous to **Proper Orthogonal Decomposition (POD)** — a classical ROM technique — but learned automatically by the autoencoder.

### Latent Interpolation
```
z_low (T_top=50) ──α=0──→ z_high (T_top=200)
                    ↓
    z_interp = (1-α)·z_low + α·z_high
                    ↓
    Decoder(z_interp) → Interpolated field
```

The interpolated field should match the actual solution at the interpolated T_top value.

---

## 6. Code Structure

| Section | Description |
|---------|-------------|
| [1] Data Generation | 500 heat solutions (Gauss-Seidel solver, T_top varies) |
| [2] Autoencoder Model | 4096→512→128→32→8 (encoder) + reverse (decoder) |
| [3] Training | 300 epochs, Adam, MSE loss |
| [4] Evaluation | Reconstruction MSE, relative L2 error |
| [5] Latent Analysis | PCA on latent vectors, explained variance |
| [6] Interpolation | Linear interp in latent space, compare to actual |
| [7] Visualization | Reconstruction, PCA, interpolation, loss, concept |
| [8] Summary | Compression ratio, error, PCA variance |

---

## 7. How to Run

```cmd
cd E:\physicsnemo-tutorials\applications\rom_autoencoder
python rom_autoencoder.py
```

Results saved to `results/`:
- `rom_reconstruction.png` — Original vs reconstruction vs error (4 test samples)
- `rom_latent_pca.png` — Latent space PCA (PC1 vs PC2, PC1 vs T_top, variance bar chart)
- `rom_interpolation.png` — Latent interpolation vs actual solver (5 steps)
- `rom_loss.png` — Training loss history
- `rom_concept.png` — Forward solver vs ROM comparison + dimensionality chart

---

## 8. Why ROM Matters

1. **Real-time applications**: Store 8 numbers instead of 4096 → instant loading
2. **Design exploration**: Interpolate in latent space to explore new designs
3. **Digital twins**: Compress sensor data for real-time monitoring
4. **Multi-fidelity**: Use latent as bridge between coarse and fine models
5. **Optimization**: Optimize in latent space (8 dims) instead of full space (4096 dims)

---

## 9. Comparison with Classical ROM

| Method | Classical POD | **Autoencoder ROM** |
|--------|--------------|---------------------|
| Basis | Linear (SVD) | **Nonlinear (neural net)** |
| Modes | Fixed (top-k SVD) | **Learned (latent dims)** |
| Nonlinearity | ✗ | **✓** |
| Interpolation | Linear | **Nonlinear (decoder)** |
| Data needed | Moderate | **More (for training)** |
| Speed | Fast | **Fast (after training)** |

---

## 10. References

- Hesthaven et al., "Reduced Basis Methods" (2016)
- Lee & Carlberg, "Model Reduction of Dynamical Systems" (2020)
- Champion et al., "Data-Driven Discovery of Coordinated Multi-Organ Physics" (2019)
