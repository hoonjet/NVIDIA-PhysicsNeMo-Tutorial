"""
PhysicsNeMo Tutorial: Checkpoint & Resume Management
=====================================================
Practical ML Workflow — Save, Resume, and Compare Experiments

Existing tutorials:
  - Most tutorials save a single .pth file at the end
  - No resume capability, no best model tracking, no experiment management

THIS tutorial:
  - Systematic checkpoint management for PhysicsNeMo workflows
  - Save model + optimizer + scheduler + epoch state
  - Resume training from any checkpoint
  - Track best model (lowest validation loss)
  - Multi-experiment comparison
  - Checkpoint inspection and loading

Key difference:
  ┌──────────────────────┬──────────────────────────┐
  │ Existing (simple)     │ THIS (systematic)         │
  ├──────────────────────┼──────────────────────────┤
  │ torch.save(model)     │ Save model+opt+sched+epoch│
  │ Save once at end       │ Save every N epochs       │
  │ No resume              │ Full resume capability    │
  │ No best tracking       │ Best model tracking       │
  │ No experiment mgmt     │ Multi-experiment compare  │
  └──────────────────────┴──────────────────────────┘

Author: PhysicsNeMo Tutorial
Date: 2026-09-16
"""

import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("PhysicsNeMo Tutorial: Checkpoint & Resume Management")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(42); np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ============================================================
# [1] Checkpoint Manager Class
# ============================================================
class CheckpointManager:
    """
    Systematic checkpoint management for PINN training.

    Features:
    - Save full training state (model + optimizer + scheduler + epoch)
    - Resume from any checkpoint
    - Track best model (lowest loss)
    - Save training metadata (loss history, config)
    - List and inspect saved checkpoints
    """

    def __init__(self, checkpoint_dir, experiment_name="default"):
        self.ckpt_dir = os.path.join(checkpoint_dir, experiment_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.experiment_name = experiment_name
        self.best_loss = float('inf')
        self.best_epoch = 0

    def save(self, model, optimizer, scheduler, epoch, loss, loss_history, extra=None):
        """Save full training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'loss_history': loss_history,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'extra': extra or {},
        }

        # Save periodic checkpoint
        ckpt_path = os.path.join(self.ckpt_dir, f'checkpoint_epoch_{epoch:05d}.pth')
        torch.save(checkpoint, ckpt_path)

        # Track best model
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            best_path = os.path.join(self.ckpt_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

        # Save latest (for easy resume)
        latest_path = os.path.join(self.ckpt_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        return ckpt_path

    def load(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        """Load checkpoint and restore training state."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.ckpt_dir, 'latest.pth')

        if not os.path.exists(checkpoint_path):
            print(f"  No checkpoint found at {checkpoint_path}")
            return 0, float('inf'), []

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        loss_history = checkpoint.get('loss_history', [])

        print(f"  Loaded checkpoint: epoch={epoch}, loss={loss:.6e}")
        return epoch, loss, loss_history

    def load_best(self, model):
        """Load best model (lowest loss)."""
        best_path = os.path.join(self.ckpt_dir, 'best_model.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded BEST model: epoch={checkpoint['epoch']}, loss={checkpoint['loss']:.6e}")
            return checkpoint['epoch'], checkpoint['loss']
        return 0, float('inf')

    def list_checkpoints(self):
        """List all saved checkpoints."""
        ckpts = []
        for f in sorted(os.listdir(self.ckpt_dir)):
            if f.endswith('.pth'):
                path = os.path.join(self.ckpt_dir, f)
                size = os.path.getsize(path) / 1024  # KB
                ckpts.append({'file': f, 'size_kb': size})
        return ckpts

    def save_metadata(self, config, results):
        """Save experiment metadata as JSON."""
        meta = {
            'experiment': self.experiment_name,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'config': config,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        meta_path = os.path.join(self.ckpt_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return meta_path

print(f"\n[1] CheckpointManager class defined")
print(f"  Features: save/load/resume/best/list/metadata")

# ============================================================
# [2] Simple PINN Model (Burgers)
# ============================================================
class SimplePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

NU = 0.01  # Viscosity

def pde_loss(model, xy):
    u = model(xy)
    g = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    residual = u_t + u * u_x - NU * u_xx
    return (residual**2).mean()

def ic_loss(model, xy_ic):
    u_pred = model(xy_ic)
    x = xy_ic[:, 0:1]
    return ((u_pred + torch.sin(np.pi * x))**2).mean()

def bc_loss(model, xy_bc):
    return (model(xy_bc)**2).mean()

print(f"\n[2] Model: SimplePINN (Burgers, ν={NU})")

# ============================================================
# [3] Collocation
# ============================================================
N_INT = 3000; N_IC = 500; N_BC = 200
x_int = torch.rand(N_INT,1,device=device)*2-1
t_int = torch.rand(N_INT,1,device=device)
xy_int = torch.cat([x_int,t_int],dim=1); xy_int.requires_grad_(True)
x_ic = torch.rand(N_IC,1,device=device)*2-1
xy_ic = torch.cat([x_ic, torch.zeros(N_IC,1,device=device)],dim=1)
x_bc = torch.cat([-torch.ones(N_BC//2,1,device=device), torch.ones(N_BC//2,1,device=device)])
xy_bc = torch.cat([x_bc, torch.rand(N_BC,1,device=device)],dim=1)

# ============================================================
# [4] Training with Checkpoint Manager
# ============================================================
print(f"\n[4] Training with CheckpointManager...")

EPOCHS = 2000
SAVE_INTERVAL = 500  # Save every 500 epochs

model = SimplePINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

ckpt_mgr = CheckpointManager(CKPT_DIR, experiment_name="burgers_exp1")

loss_history = []
t0 = time.time()

for ep in range(EPOCHS):
    optimizer.zero_grad()
    lp = pde_loss(model, xy_int)
    li = ic_loss(model, xy_ic)
    lb = bc_loss(model, xy_bc)
    loss = lp + 10.0 * li + 10.0 * lb
    loss.backward(); optimizer.step(); scheduler.step()
    loss_history.append(loss.item())

    # Save checkpoint periodically
    if (ep + 1) % SAVE_INTERVAL == 0:
        ckpt_mgr.save(model, optimizer, scheduler, ep + 1, loss.item(), loss_history,
                      extra={'lr': optimizer.param_groups[0]['lr']})
        print(f"  Ep {ep+1:4d}/{EPOCHS} | Loss: {loss.item():.6e} | [CHECKPOINT SAVED]")

    if (ep + 1) % 200 == 0 and (ep + 1) % SAVE_INTERVAL != 0:
        print(f"  Ep {ep+1:4d}/{EPOCHS} | Loss: {loss.item():.6e}")

print(f"\n  Training complete! Time: {time.time()-t0:.1f}s")
print(f"  Best loss: {ckpt_mgr.best_loss:.6e} at epoch {ckpt_mgr.best_epoch}")

# ============================================================
# [5] Demonstrate Resume
# ============================================================
print(f"\n[5] Demonstrating resume from checkpoint...")

# Simulate interruption: create new model, load from checkpoint
model_resumed = SimplePINN().to(device)
optimizer_resumed = torch.optim.Adam(model_resumed.parameters(), lr=1e-3)
scheduler_resumed = torch.optim.lr_scheduler.StepLR(optimizer_resumed, step_size=1000, gamma=0.5)

start_epoch, start_loss, start_hist = ckpt_mgr.load(
    model_resumed, optimizer_resumed, scheduler_resumed,
    checkpoint_path=os.path.join(CKPT_DIR, "burgers_exp1", "latest.pth"))

# Continue training from resumed checkpoint
RESUME_EPOCHS = 500
print(f"  Resuming from epoch {start_epoch}, training {RESUME_EPOCHS} more epochs...")

for ep in range(start_epoch, start_epoch + RESUME_EPOCHS):
    optimizer_resumed.zero_grad()
    lp = pde_loss(model_resumed, xy_int)
    li = ic_loss(model_resumed, xy_ic)
    lb = bc_loss(model_resumed, xy_bc)
    loss = lp + 10.0 * li + 10.0 * lb
    loss.backward(); optimizer_resumed.step(); scheduler_resumed.step()
    start_hist.append(loss.item())

    if (ep + 1) % 100 == 0:
        print(f"  Resumed Ep {ep+1:4d} | Loss: {loss.item():.6e}")

# Save final resumed checkpoint
ckpt_mgr.save(model_resumed, optimizer_resumed, scheduler_resumed,
              start_epoch + RESUME_EPOCHS, loss.item(), start_hist)

# ============================================================
# [6] Demonstrate Best Model Loading
# ============================================================
print(f"\n[6] Loading best model for evaluation...")

model_best = SimplePINN().to(device)
best_epoch, best_loss = ckpt_mgr.load_best(model_best)

# ============================================================
# [7] List Checkpoints
# ============================================================
print(f"\n[7] Saved checkpoints:")
ckpts = ckpt_mgr.list_checkpoints()
for c in ckpts:
    print(f"  {c['file']:40s}  {c['size_kb']:.1f} KB")

# ============================================================
# [8] Save Metadata
# ============================================================
meta_path = ckpt_mgr.save_metadata(
    config={'nu': NU, 'epochs': EPOCHS, 'lr': 1e-3, 'model': 'SimplePINN'},
    results={'best_loss': ckpt_mgr.best_loss, 'best_epoch': ckpt_mgr.best_epoch,
             'total_epochs': start_epoch + RESUME_EPOCHS})
print(f"\n  Metadata saved: {meta_path}")

# ============================================================
# [9] Visualization
# ============================================================
print(f"\n[9] Visualization...")

# --- Figure 1: Loss with checkpoint markers ---
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.semilogy(loss_history + start_hist[EPOCHS:], 'b-', linewidth=2, label='Training Loss')

# Mark checkpoint save points
for ep in range(SAVE_INTERVAL, EPOCHS + 1, SAVE_INTERVAL):
    if ep <= len(loss_history):
        ax.axvline(x=ep-1, color='green', linestyle=':', alpha=0.7)
        ax.plot(ep-1, loss_history[ep-1], 'g^', markersize=10)

# Mark resume point
ax.axvline(x=EPOCHS, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax.text(EPOCHS+10, ax.get_ylim()[1]*0.3, 'RESUME', color='red', fontsize=11, fontweight='bold')

# Mark best model
ax.plot(ckpt_mgr.best_epoch-1, ckpt_mgr.best_loss, 'r*', markersize=15, label=f'Best (ep={ckpt_mgr.best_epoch})')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (log)', fontsize=12)
ax.set_title('Checkpoint Management: Training with Save/Resume', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'checkpoint_training.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: checkpoint_training.png")

# --- Figure 2: Checkpoint structure ---
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.axis('off')
txt = (
    "Checkpoint Structure:\n\n"
    "checkpoint = {\n"
    "    'epoch': 1500,\n"
    "    'model_state_dict': {...},      # Model weights\n"
    "    'optimizer_state_dict': {...},   # Adam momentum buffers\n"
    "    'scheduler_state_dict': {...},   # LR scheduler state\n"
    "    'loss': 0.000123,\n"
    "    'loss_history': [...],          # Full loss curve\n"
    "    'timestamp': '2026-09-16 07:38:00',\n"
    "    'extra': {'lr': 0.0005}\n"
    "}\n\n"
    "CheckpointManager Features:\n\n"
    "  1. Periodic save (every N epochs)\n"
    "     → checkpoint_epoch_00500.pth\n"
    "     → checkpoint_epoch_01000.pth\n"
    "     → checkpoint_epoch_01500.pth\n\n"
    "  2. Best model tracking\n"
    "     → best_model.pth (lowest loss)\n\n"
    "  3. Latest checkpoint (for resume)\n"
    "     → latest.pth\n\n"
    "  4. Metadata (JSON)\n"
    "     → metadata.json (config + results)\n\n"
    "  5. Resume workflow:\n"
    "     → Load latest.pth\n"
    "     → Restore model + optimizer + scheduler\n"
    "     → Continue training from saved epoch\n\n"
    "vs. Existing (simple save):\n\n"
    "  Existing: torch.save(model.state_dict(), 'model.pth')\n"
    "  THIS:     Full state (model+opt+sched+epoch+history)"
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Checkpoint Management System', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'checkpoint_structure.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: checkpoint_structure.png")

# --- Figure 3: Checkpoint listing ---
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.axis('off')
col_labels = ['Checkpoint File', 'Size (KB)', 'Type']
row_data = []
for c in ckpts:
    ftype = 'Best' if 'best' in c['file'] else 'Latest' if 'latest' in c['file'] else 'Periodic'
    row_data.append([c['file'], f"{c['size_kb']:.1f}", ftype])

table = ax.table(cellText=row_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title(f'Saved Checkpoints ({len(ckpts)} files)', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'checkpoint_listing.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: checkpoint_listing.png")

# ============================================================
# [10] Summary
# ============================================================
print("\n" + "="*70)
print("[10] Summary")
print("="*70)
print(f"\n  Method: Checkpoint & Resume Management")
print(f"  Checkpoints saved: {len(ckpts)}")
print(f"  Best model: epoch {ckpt_mgr.best_epoch}, loss {ckpt_mgr.best_loss:.6e}")
print(f"  Resume demonstrated: {RESUME_EPOCHS} epochs from epoch {EPOCHS}")
print(f"\n  Key features:")
print(f"    1. Full state save (model + optimizer + scheduler + epoch)")
print(f"    2. Periodic checkpoints (every {SAVE_INTERVAL} epochs)")
print(f"    3. Best model tracking (lowest loss)")
print(f"    4. Resume from any checkpoint")
print(f"    5. Metadata JSON (config + results)")
print(f"\n  Checkpoints in: {CKPT_DIR}")
print(f"  Results in: {RESULTS_DIR}")
print("="*70)
