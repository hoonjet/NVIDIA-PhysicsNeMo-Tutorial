# Transfer Learning

> Pre-train on abundant source data, fine-tune on scarce target data.

## Tutorials

| Tutorial | Problem | Key Concept |
|----------|---------|-------------|
| [transfer_fno](transfer_fno/) | Cross-domain Darcy Flow | FNO pre-train (coarse k) → fine-tune (fine k): freeze vs full FT |
| [pinn_transfer](pinn_transfer/) | Cross-PDE: Burgers → Sine-Gordon | PINN transfer (different PDE, loss function changes): scratch vs freeze vs full FT |

## Why This Category?

All other categories train models **from scratch**. Transfer learning is the most widely used ML technique in practice, yet no tutorial covered it. This category fills that gap.

## Key Concepts

- **Pre-training**: Learn general features on abundant source data
- **Freezing**: Lock encoder weights, only adapt decoder
- **Fine-tuning**: Adapt all weights with low learning rate
- **Data efficiency**: Achieve comparable accuracy with less target data
