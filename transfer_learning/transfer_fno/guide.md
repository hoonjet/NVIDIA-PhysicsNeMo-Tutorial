# Transfer Learning: FNO Pre-train → Fine-tune

> **Category**: `transfer_learning/` — Pre-train on source, fine-tune on target  
> **Paradigm**: Knowledge transfer between domains  
> **Model**: FNO with freeze/fine-tune strategies

---

## 1. What Makes This Tutorial Unique?

This is the **only tutorial that demonstrates transfer learning**. Every other tutorial trains models from scratch. This tutorial pre-trains an FNO on an abundant source domain, then fine-tunes it on a scarce target domain.

| Aspect | All Other Tutorials | **This Tutorial** |
|--------|---------------------|-------------------|
| **Training** | From scratch | **Pre-train → Fine-tune** |
| **Source data** | N/A | **200 samples (coarse permeability)** |
| **Target data** | N/A | **30 samples (fine permeability)** |
| **Strategies** | 1 (scratch) | **3 (scratch, freeze, full FT)** |

### vs. FNO Zero-Shot
- Zero-Shot: same model, no adaptation at all
- Transfer Learning: **adapt** the pre-trained model to the new domain
- Zero-Shot works when domains are identical; Transfer Learning works when domains differ

### vs. PINO
- PINO adds physics loss to the same training data
- Transfer Learning uses a different source domain to pre-train
- Both improve data efficiency, but via completely different mechanisms

---

## 2. Problem: Cross-Domain Darcy Flow

```
Source domain:  Coarse permeability (length_scale=0.2) — 200 samples (abundant)
Target domain:  Fine permeability (length_scale=0.05) — 30 samples (scarce)

Same PDE: -∇·(k∇p) = 1, p=0 on boundary
Different k distribution: coarse (smooth) vs fine (rapidly varying)
```

### Why This Matters
In real engineering, you often have:
- Abundant data from a **cheap/simplified** simulation (coarse mesh, 2D, steady-state)
- Scarce data from an **expensive/detailed** simulation (fine mesh, 3D, transient)

Transfer learning bridges this gap: learn general features from cheap data, adapt to expensive data.

---

## 3. Three Strategies Compared

### Strategy 1: Scratch (Baseline)
```
Train FNO from scratch on 30 target samples
```
- No pre-training, no knowledge transfer
- Limited by small target dataset (30 samples)
- Serves as baseline to measure transfer learning benefit

### Strategy 2: Transfer (Freeze Encoder)
```
1. Pre-train FNO on 200 source samples (200 epochs)
2. Freeze spectral layers (encoder)
3. Fine-tune only the output MLP (decoder) on 30 target samples
```
- Encoder weights are locked → preserves learned features
- Only ~10% of parameters are trainable
- Fastest fine-tuning, prevents catastrophic forgetting
- Best when source and target are similar

### Strategy 3: Transfer (Full Fine-tune)
```
1. Pre-train FNO on 200 source samples (200 epochs)
2. Fine-tune ALL layers on 30 target samples with low LR (5e-4)
```
- All weights are adaptable
- Lower learning rate prevents destroying pre-trained features
- Best accuracy but risk of forgetting source knowledge
- Best when source and target differ moderately

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| `[1] Data` | Source (coarse, 200) + Target (fine, 30) + Test (fine, 30) |
| `[2] FNO` | Architecture with `freeze_encoder()` method |
| `[3] Scratch` | Train from scratch on target (baseline) |
| `[4] Pre-train` | Train on source domain (200 epochs) |
| `[5] Freeze` | Fine-tune decoder only (frozen encoder) |
| `[6] Full FT` | Fine-tune all layers with low LR |
| `[7] Evaluation` | Compare L2 errors across 3 strategies |
| `[8] Visualization` | Pre-train loss, fine-tune curves, bar chart, predictions, strategy diagram |

---

## 5. Key Results

### 5.1 Transfer Learning Beats Scratch
- With only 30 target samples, scratch training struggles
- Pre-trained models (freeze & full FT) achieve lower L2 error
- The benefit grows as target data decreases

### 5.2 Freeze vs Full Fine-tune
- **Freeze**: faster, fewer trainable params, prevents forgetting
- **Full FT**: more accurate, but needs careful LR tuning
- Trade-off: adaptation power vs risk of forgetting

### 5.3 Data Efficiency
- Transfer learning achieves comparable accuracy with **6× less target data**
- This is the key practical advantage in expensive simulation scenarios

---

## 6. Running the Tutorial

```cmd
cd E:\physicsnemo-tutorials\transfer_learning\transfer_fno
python transfer_fno.py
```

### Result Files
| File | Description |
|------|-------------|
| `results/transfer_pretrain.png` | Pre-training loss on source domain |
| `results/transfer_finetune.png` | Fine-tuning: train & test loss (3 strategies) |
| `results/transfer_comparison.png` | Bar chart: final L2 for 3 strategies |
| `results/transfer_result.png` | Prediction comparison + error maps |
| `results/transfer_strategies.png` | Visual explanation of 3 strategies |

---

## 7. Key Concepts Learned

1. **Pre-training**: Train on abundant source data to learn general features (spectral patterns, PDE structure). The encoder learns "how to process PDE inputs" — this transfers across domains.

2. **Freezing**: Set `requires_grad = False` for encoder weights. Only the decoder adapts. This is useful when target data is very scarce — you don't want to destroy pre-trained features with noisy gradients.

3. **Fine-tuning**: Use a **lower learning rate** (5e-4 vs 1e-3 for scratch). This makes small, careful adjustments to pre-trained weights rather than large random changes.

4. **Catastrophic Forgetting**: If you fine-tune all layers with high LR, the model "forgets" source domain knowledge. Freezing prevents this; low LR mitigates it.

5. **Domain Similarity**: Transfer learning works best when source and target share structure. Here, both are Darcy flow (same PDE), just different permeability distributions.

6. **Data Efficiency**: The practical value. CFD simulations are expensive. If you can pre-train on cheap data and fine-tune on a few expensive samples, you save enormous computation time.

---

## 8. Comparison with Other Tutorials

| Feature | FNO (Darcy) | FNO Zero-Shot | PINO | **This Tutorial** |
|---------|:-----------:|:-------------:|:----:|:------------------:|
| **Training** | Scratch | Scratch | Scratch + physics | **Pre-train + fine-tune** |
| **Source data** | N/A | N/A | Same domain | **Different domain** |
| **Adaptation** | None | None (zero-shot) | Physics loss | **Fine-tune** |
| **Data efficiency** | Standard | Standard | Less data (physics) | **Less target data (transfer)** |

---

## 9. Extensions

- **Progressive unfreezing**: Start with frozen encoder, gradually unfreeze layers
- **Layer-wise LR decay**: Different LR per layer (lower for encoder, higher for decoder)
- **Cross-PDE transfer**: Pre-train on Darcy, fine-tune on Helmholtz (different PDE, same operator structure)
- **Multi-source transfer**: Pre-train on multiple source domains simultaneously
- **Transfer + PINO**: Combine transfer learning (pre-train) with physics loss (PINO) for maximum data efficiency

---

## 10. References

- Yosinski et al., "How transferable are features in deep neural networks?," NeurIPS 2014
- Li et al., "Fourier Neural Operator for Parametric PDEs," ICLR 2021
- Pan & Yang, "A Survey on Transfer Learning," IEEE TKDE 2010
