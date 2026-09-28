# Checkpoint & Resume Management

> **Category**: `utils/` — Practical ML workflow
> **Problem**: Save, resume, and manage training experiments

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing tutorials | **THIS (Checkpoint Mgmt)** |
|--------|-------------------|---------------------------|
| **Save** | `torch.save(model)` at end | **Full state every N epochs** |
| **Resume** | Not supported | **Load and continue training** |
| **Best model** | Not tracked | **Auto-tracked (lowest loss)** |
| **Metadata** | None | **JSON config + results** |

Most tutorials in this repo save a single `.pth` file at the end of training.
This tutorial demonstrates a **systematic checkpoint management system** that
saves the full training state periodically and can resume from any checkpoint.

---

## 2. Checkpoint Structure

Each checkpoint stores the **complete training state**, not just model weights:

```python
checkpoint = {
    'epoch': 1500,
    'model_state_dict': {...},       # Model weights
    'optimizer_state_dict': {...},   # Adam momentum buffers
    'scheduler_state_dict': {...},   # LR scheduler state
    'loss': 0.000123,
    'loss_history': [...],           # Full loss curve
    'timestamp': '2026-09-16 07:38:00',
    'extra': {'lr': 0.0005}
}
```

### Saved Files

| File | Description |
|------|-------------|
| `checkpoint_epoch_00500.pth` | Periodic checkpoint (every 500 epochs) |
| `checkpoint_epoch_01000.pth` | Periodic checkpoint |
| `best_model.pth` | Best model (lowest loss) |
| `latest.pth` | Latest checkpoint (for easy resume) |
| `metadata.json` | Experiment config + results |

---

## 3. CheckpointManager API

### `save(model, optimizer, scheduler, epoch, loss, loss_history)`
Saves full training state. Automatically updates `best_model.pth` and `latest.pth`.

### `load(model, optimizer, scheduler, checkpoint_path)`
Restores model + optimizer + scheduler + epoch from a checkpoint file.

### `load_best(model)`
Loads the best model (lowest loss) for evaluation.

### `list_checkpoints()`
Returns all saved checkpoint files with sizes.

### `save_metadata(config, results)`
Saves experiment configuration and results as JSON.

---

## 4. How to Run

```cmd
cd E:\physicsnemo-tutorials\utils\checkpoint_management
python checkpoint_management.py
```

The script demonstrates the full workflow:
1. Train Burgers PINN for 2000 epochs (save every 500)
2. Simulate interruption — create new model, load from `latest.pth`
3. Resume training for 500 more epochs
4. Load best model for evaluation
5. List all saved checkpoints
6. Save metadata JSON

---

## 5. Results

- **`checkpoint_training.png`** — Loss curve with checkpoint markers (green ▲),
  resume point (red dashed line), and best model (red ★)
- **`checkpoint_structure.png`** — Checkpoint structure and feature explanation
- **`checkpoint_listing.png`** — Table of all saved checkpoint files with sizes

---

## 6. Key Takeaways

1. **Always save optimizer state** — Adam momentum buffers are needed for smooth resume
2. **Track best model separately** — the last checkpoint is not necessarily the best
3. **Save metadata as JSON** — makes experiment comparison easy without loading `.pth` files
4. **Use `latest.pth` convention** — simplifies resume logic (no need to search for latest epoch)


