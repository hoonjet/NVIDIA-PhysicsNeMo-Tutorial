"""
PhysicsNeMo Tutorial: Hydra Configuration System
==================================================
Config-Driven ML — Manage Models, Optimizers, and Training via YAML

Existing tutorials:
  - All 35+ tutorials use HARDCODED parameters in Python scripts
  - To change a parameter, you must edit the .py file

THIS tutorial:
  - Hydra-based YAML configuration for PhysicsNeMo workflows
  - Define model, optimizer, loss, training params in config files
  - Run experiments WITHOUT modifying Python code
  - Hyperparameter sweep with a single command
  - Config composition (base + experiment override)

Key difference:
  ┌──────────────────────┬──────────────────────────┐
  │ Existing (hardcoded)  │ THIS (Hydra config)      │
  ├──────────────────────┼──────────────────────────┤
  │ Edit .py to change    │ Edit YAML to change       │
  │ One config per script │ Compose: base + override │
  │ Manual sweep           │ Auto sweep (--multirun)   │
  └──────────────────────┴──────────────────────────┘

This tutorial demonstrates:
  1. YAML config structure (model, optimizer, training)
  2. Config composition (base + experiment override)
  3. Hydra decorator usage
  4. Hyperparameter sweep
  5. Results logging and comparison

Author: PhysicsNeMo Tutorial
Date: 2026-09-16
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo Tutorial: Hydra Configuration System")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(42); np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# [1] Simulated Hydra Config (Pure Python Dict)
# ============================================================
# In real PhysicsNeMo, Hydra reads YAML files via @hydra.main decorator.
# Here we simulate the same structure using Python dicts to avoid
# requiring OmegaConf/Hydra installation.

BASE_CONFIG = {
    "model": {
        "architecture": "FullyConnected",
        "in_features": 2,
        "out_features": 3,
        "layer_size": 64,
        "num_layers": 4,
        "activation": "tanh",
    },
    "optimizer": {
        "name": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
    },
    "scheduler": {
        "name": "StepLR",
        "step_size": 2000,
        "gamma": 0.5,
    },
    "training": {
        "epochs": 3000,
        "n_interior": 5000,
        "n_boundary": 2000,
        "lambda_pde": 1.0,
        "lambda_bc": 10.0,
    },
    "physics": {
        "nu": 0.01,  # viscosity (Re=100)
        "rho": 1.0,
    }
}

# Experiment overrides (like Hydra config groups)
EXPERIMENT_CONFIGS = {
    "default": {},  # Use base config as-is
    "large_model": {
        "model": {"layer_size": 128, "num_layers": 6},
    },
    "fast_lr": {
        "optimizer": {"lr": 5e-3},
    },
    "slow_lr": {
        "optimizer": {"lr": 5e-4},
    },
    "high_re": {
        "physics": {"nu": 0.001},  # Re=1000
    },
    "long_train": {
        "training": {"epochs": 5000},
    },
}

def deep_update(base, override):
    """Recursively merge override dict into base dict (like Hydra composition)."""
    result = {k: v for k, v in base.items()}
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result

def get_config(experiment_name="default"):
    """Compose base config with experiment override (like Hydra @hydra.main)."""
    override = EXPERIMENT_CONFIGS.get(experiment_name, {})
    return deep_update(BASE_CONFIG, override)

print(f"\n[1] Config System (simulated Hydra):")
print(f"  Base config: model={BASE_CONFIG['model']['architecture']}")
print(f"  Available experiments: {list(EXPERIMENT_CONFIGS.keys())}")

# ============================================================
# [2] Model Factory (from config)
# ============================================================
def build_model_from_config(cfg):
    """Build neural network from config dict (like PhysicsNeMo model factory)."""
    m = cfg["model"]
    layers = [m["in_features"]] + [m["layer_size"]] * m["num_layers"] + [m["out_features"]]
    activation_fn = nn.Tanh if m["activation"] == "tanh" else nn.SiLU

    class ConfigurablePINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linears = nn.ModuleList()
            for i in range(len(layers) - 1):
                self.linears.append(nn.Linear(layers[i], layers[i+1]))
            self.act = activation_fn()
            for lin in self.linears:
                nn.init.xavier_normal_(lin.weight)
                nn.init.zeros_(lin.bias)
        def forward(self, x):
            for i in range(len(self.linears) - 1):
                x = self.act(self.linears[i](x))
            return self.linears[-1](x)

    model = ConfigurablePINN()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params

print(f"\n[2] Model factory: build from config dict")

# ============================================================
# [3] Optimizer Factory (from config)
# ============================================================
def build_optimizer_from_config(cfg, model):
    """Build optimizer from config dict."""
    o = cfg["optimizer"]
    if o["name"] == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=o["lr"], weight_decay=o.get("weight_decay", 0))
    elif o["name"] == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {o['name']}")
    return opt

def build_scheduler_from_config(cfg, opt):
    """Build scheduler from config dict."""
    s = cfg["scheduler"]
    if s["name"] == "StepLR":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=s["step_size"], gamma=s["gamma"])
    elif s["name"] == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["training"]["epochs"])
    return None

print(f"[3] Optimizer + Scheduler factories")

# ============================================================
# [4] PINN Training (Lid-Driven Cavity, simplified)
# ============================================================
def train_pinn(cfg, experiment_name="default"):
    """Train PINN using config-driven approach."""
    print(f"\n{'='*50}")
    print(f"  Experiment: '{experiment_name}'")
    print(f"{'='*50}")

    # Build from config
    model, n_params = build_model_from_config(cfg)
    model = model.to(device)
    opt = build_optimizer_from_config(cfg, model)
    sched = build_scheduler_from_config(cfg, opt)

    t_cfg = cfg["training"]
    p_cfg = cfg["physics"]
    NU = p_cfg["nu"]; RHO = p_cfg["rho"]
    N_INT = t_cfg["n_interior"]; N_BC = t_cfg["n_boundary"]
    EPOCHS = t_cfg["epochs"]
    L_PDE = t_cfg["lambda_pde"]; L_BC = t_cfg["lambda_bc"]

    print(f"  Model: {n_params} params, layers={cfg['model']['num_layers']}, size={cfg['model']['layer_size']}")
    print(f"  Optimizer: {cfg['optimizer']['name']}, lr={cfg['optimizer']['lr']}")
    print(f"  Physics: nu={NU} (Re={1/NU:.0f})")
    print(f"  Training: {EPOCHS} epochs, N_int={N_INT}")

    # Collocation
    x_int = torch.rand(N_INT,1,device=device); y_int = torch.rand(N_INT,1,device=device)
    xy_int = torch.cat([x_int,y_int],dim=1); xy_int.requires_grad_(True)
    n_per = N_BC//4
    xy_top = torch.cat([torch.rand(n_per,1,device=device), torch.ones(n_per,1,device=device)],dim=1)
    xy_bot = torch.cat([torch.rand(n_per,1,device=device), torch.zeros(n_per,1,device=device)],dim=1)
    xy_left = torch.cat([torch.zeros(n_per,1,device=device), torch.rand(n_per,1,device=device)],dim=1)
    xy_right = torch.cat([torch.ones(n_per,1,device=device), torch.rand(n_per,1,device=device)],dim=1)

    def pde_loss(xy):
        out = model(xy)
        u,v,p = out[:,0:1],out[:,1:2],out[:,2:3]
        gu = torch.autograd.grad(u,xy,torch.ones_like(u),create_graph=True)[0]
        gv = torch.autograd.grad(v,xy,torch.ones_like(v),create_graph=True)[0]
        gp = torch.autograd.grad(p,xy,torch.ones_like(p),create_graph=True)[0]
        ux,uy = gu[:,0:1],gu[:,1:2]
        vx,vy = gv[:,0:1],gv[:,1:2]
        px,py = gp[:,0:1],gp[:,1:2]
        uxx = torch.autograd.grad(ux,xy,torch.ones_like(ux),create_graph=True)[0][:,0:1]
        uyy = torch.autograd.grad(uy,xy,torch.ones_like(uy),create_graph=True)[0][:,1:2]
        vxx = torch.autograd.grad(vx,xy,torch.ones_like(vx),create_graph=True)[0][:,0:1]
        vyy = torch.autograd.grad(vy,xy,torch.ones_like(vy),create_graph=True)[0][:,1:2]
        r1 = u*ux + v*uy + px/RHO - NU*(uxx+uyy)
        r2 = u*vx + v*vy + py/RHO - NU*(vxx+vyy)
        r3 = ux + vy
        return (r1**2+r2**2+r3**2).mean()

    def bc_loss():
        l = 0.0
        o = model(xy_top); l += ((o[:,0:1]-1)**2+o[:,1:2]**2).mean()
        for xy in [xy_bot,xy_left,xy_right]:
            o = model(xy); l += (o[:,0:2]**2).mean()
        return l

    hist = []
    t0 = time.time()
    for ep in range(EPOCHS):
        opt.zero_grad()
        lp = pde_loss(xy_int)
        lb = bc_loss()
        loss = L_PDE*lp + L_BC*lb
        loss.backward(); opt.step()
        if sched: sched.step()
        hist.append(loss.item())

    elapsed = time.time() - t0
    final_loss = hist[-1]
    print(f"  Final loss: {final_loss:.6e}, Time: {elapsed:.1f}s")

    return {
        "experiment": experiment_name,
        "config": cfg,
        "n_params": n_params,
        "final_loss": final_loss,
        "time": elapsed,
        "loss_history": hist,
    }

# ============================================================
# [5] Run Multiple Experiments (Simulated Sweep)
# ============================================================
print(f"\n[5] Running experiments (simulated Hydra sweep)...")

results = {}
for exp_name in ["default", "large_model", "fast_lr", "slow_lr", "high_re", "long_train"]:
    cfg = get_config(exp_name)
    results[exp_name] = train_pinn(cfg, exp_name)

# ============================================================
# [6] Visualization — Sweep Comparison
# ============================================================
print(f"\n[6] Visualization...")

# --- Figure 1: Loss comparison ---
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
for i, (name, res) in enumerate(results.items()):
    ax.semilogy(res["loss_history"], color=colors[i], linewidth=1.5, label=f'{name} (loss={res["final_loss"]:.2e})')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (log scale)', fontsize=12)
ax.set_title('Hydra Config Sweep: Loss Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'hydra_sweep_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: hydra_sweep_loss.png")

# --- Figure 2: Summary table ---
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.axis('off')
col_labels = ['Experiment', 'Model Size', 'LR', 'Re', 'Epochs', 'Params', 'Final Loss', 'Time (s)']
row_data = []
for name, res in results.items():
    cfg = res["config"]
    row = [
        name,
        f"{cfg['model']['layer_size']}×{cfg['model']['num_layers']}",
        f"{cfg['optimizer']['lr']:.0e}",
        f"{1/cfg['physics']['nu']:.0f}",
        cfg['training']['epochs'],
        f"{res['n_params']:,}",
        f"{res['final_loss']:.2e}",
        f"{res['time']:.1f}",
    ]
    row_data.append(row)

table = ax.table(cellText=row_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
# Color header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('Hydra Configuration Sweep Results', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'hydra_sweep_table.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: hydra_sweep_table.png")

# --- Figure 3: Config structure visualization ---
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.axis('off')
txt = (
    "Hydra Configuration Structure (YAML):\n\n"
    "# base_config.yaml\n"
    "model:\n"
    "  architecture: FullyConnected\n"
    "  in_features: 2\n"
    "  out_features: 3\n"
    "  layer_size: 64\n"
    "  num_layers: 4\n"
    "  activation: tanh\n\n"
    "optimizer:\n"
    "  name: Adam\n"
    "  lr: 1e-3\n"
    "  weight_decay: 0.0\n\n"
    "scheduler:\n"
    "  name: StepLR\n"
    "  step_size: 2000\n"
    "  gamma: 0.5\n\n"
    "training:\n"
    "  epochs: 3000\n"
    "  n_interior: 5000\n"
    "  n_boundary: 2000\n"
    "  lambda_pde: 1.0\n"
    "  lambda_bc: 10.0\n\n"
    "physics:\n"
    "  nu: 0.01  # Re = 100\n"
    "  rho: 1.0\n\n"
    "# Experiment overrides (config groups):\n\n"
    "# large_model.yaml\n"
    "#   model: {layer_size: 128, num_layers: 6}\n\n"
    "# fast_lr.yaml\n"
    "#   optimizer: {lr: 5e-3}\n\n"
    "# high_re.yaml\n"
    "#   physics: {nu: 0.001}  # Re=1000\n\n"
    "# Usage (Hydra):\n"
    "#   python train.py experiment=large_model\n"
    "#   python train.py experiment=fast_lr\n"
    "#   python train.py --multirun experiment=large_model,fast_lr,slow_lr"
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Hydra Config Structure & Usage', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'hydra_config_structure.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: hydra_config_structure.png")

# --- Figure 4: Best experiment ---
best_exp = min(results.items(), key=lambda x: x[1]["final_loss"])
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(best_exp[1]["loss_history"], 'b-', linewidth=2, label=f'Best: {best_exp[0]}')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title(f'Best Experiment: "{best_exp[0]}" (loss={best_exp[1]["final_loss"]:.2e})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'hydra_best.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: hydra_best.png")

# ============================================================
# [7] Save sweep results as JSON
# ============================================================
sweep_summary = {}
for name, res in results.items():
    sweep_summary[name] = {
        "n_params": res["n_params"],
        "final_loss": res["final_loss"],
        "time_seconds": res["time"],
        "config_summary": {
            "layer_size": res["config"]["model"]["layer_size"],
            "num_layers": res["config"]["model"]["num_layers"],
            "lr": res["config"]["optimizer"]["lr"],
            "epochs": res["config"]["training"]["epochs"],
            "nu": res["config"]["physics"]["nu"],
        }
    }

with open(os.path.join(RESULTS_DIR, "sweep_results.json"), "w") as f:
    json.dump(sweep_summary, f, indent=2)
print(f"\n  Saved: sweep_results.json")

# ============================================================
# [8] Summary
# ============================================================
print("\n" + "="*70)
print("[8] Summary")
print("="*70)
print(f"\n  Method: Hydra Configuration System for PhysicsNeMo")
print(f"  Experiments run: {len(results)}")
print(f"  Best experiment: '{best_exp[0]}' (loss={best_exp[1]['final_loss']:.2e})")
print(f"\n  Key concept:")
print(f"    - Config-driven (YAML) instead of hardcoded parameters")
print(f"    - Config composition: base + experiment override")
print(f"    - Hyperparameter sweep with single command")
print(f"    - Reproducible experiments")
print(f"\n  Results saved to: {RESULTS_DIR}")
print("="*70)
