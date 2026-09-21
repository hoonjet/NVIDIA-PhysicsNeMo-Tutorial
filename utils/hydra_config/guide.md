# Hydra Configuration System

> **Category**: `utils/` — Config-driven ML workflow
> **Problem**: Manage experiments via YAML, not hardcoded Python

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing tutorials | **THIS (Hydra Config)** |
|--------|-------------------|------------------------|
| **Parameter management** | Hardcoded in .py | **YAML config files** |
| **Experiment switching** | Edit Python code | **`--config-name=exp`** |
| **Hyperparameter sweep** | Manual runs | **`--multirun` auto sweep** |
| **Reproducibility** | Hard to track | **Config = full recipe** |

---

## 2. Concept

```
base_config.yaml          ← Default params (model, optimizer, training)
    + experiment overrides ← "large_model", "fast_lr", "high_re", etc.
    = composed config       ← Merged config for each experiment
```

### YAML Structure
```yaml
model:
  architecture: FullyConnected
  layer_size: 64
  num_layers: 4
optimizer:
  name: Adam
  lr: 1e-3
training:
  epochs: 3000
physics:
  nu: 0.01  # Re=100
```

### Usage (Hydra)
```bash
python train.py experiment=default
python train.py experiment=large_model
python train.py --multirun experiment=large_model,fast_lr,slow_lr
```

---

## 3. How to Run

```cmd
cd E:\physicsnemo-tutorials\utils\hydra_config
python hydra_config.py
```

Runs 6 experiments automatically and compares results.

---

## 4. Results

- `hydra_sweep_loss.png` — Loss curves for all 6 experiments
- `hydra_sweep_table.png` — Summary table (params, loss, time)
- `hydra_config_structure.png` — YAML structure and usage examples
- `hydra_best.png` — Best experiment loss curve
- `sweep_results.json` — Machine-readable summary
