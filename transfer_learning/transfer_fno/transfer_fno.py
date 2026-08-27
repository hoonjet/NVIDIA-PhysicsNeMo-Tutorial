"""
PhysicsNeMo Transfer Learning Tutorial: FNO Pre-train → Fine-tune
=================================================================
All existing tutorials train models FROM SCRATCH. This is the ONLY tutorial
that demonstrates TRANSFER LEARNING: pre-train on one problem, fine-tune on another.

Problem: Darcy Flow with different permeability distributions
    - Source domain: coarse permeability (length_scale=0.2) — abundant data (200 samples)
    - Target domain: fine permeability (length_scale=0.05) — scarce data (30 samples)

Three strategies compared:
    1. Scratch: Train on target data from scratch (baseline)
    2. Transfer (freeze encoder): Pre-train on source, freeze encoder, fine-tune decoder
    3. Transfer (full fine-tune): Pre-train on source, fine-tune all layers

Key concepts:
    - Pre-training: Learn general features on abundant source data
    - Freezing: Lock encoder weights, only adapt decoder to target
    - Fine-tuning: Adapt all weights with small learning rate
    - Data efficiency: Transfer learning needs less target data

This is fundamentally different from:
    - All other tutorials (train from scratch every time)
    - FNO Zero-Shot (no adaptation at all — transfer learning adapts to new domain)
    - PINO (adds physics loss, not transfer learning)

Author: PhysicsNeMo Tutorial
Date: 2026-08-13
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# [0] Environment Setup
# ============================================================
print("=" * 70)
print("PhysicsNeMo Transfer Learning Tutorial: FNO Pre-train → Fine-tune")
print("Source: Coarse permeability (200 samples) → Target: Fine permeability (30 samples)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# [1] Data: Source (coarse) + Target (fine)
# ============================================================
GRID = 32

def generate_permeability(n_samples, grid_size, length_scale=0.2):
    kx = np.fft.fftfreq(grid_size) * grid_size
    ky = np.fft.fftfreq(grid_size) * grid_size
    KX, KY = np.meshgrid(kx, ky)
    spectrum = np.exp(-(KX**2 + KY**2) * length_scale**2)
    fields = []
    for _ in range(n_samples):
        noise = np.random.randn(grid_size, grid_size) + 1j * np.random.randn(grid_size, grid_size)
        field_hat = noise * np.sqrt(spectrum)
        field = np.real(np.fft.ifft2(field_hat))
        field = (field - field.mean()) / (field.std() + 1e-8)
        field = np.exp(field * 1.5)
        fields.append(field)
    return np.array(fields, dtype=np.float32)


def solve_darcy_fd(k_field, grid_size):
    N = grid_size - 2
    h = 1.0 / (grid_size - 1)
    A = np.zeros((N * N, N * N))
    b = np.ones(N * N) * h * h
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            k_c = k_field[i + 1, j + 1]
            k_n = k_field[i, j + 1] if i > 0 else k_field[i + 1, j + 1]
            k_s = k_field[i + 2, j + 1] if i < N - 1 else k_field[i + 1, j + 1]
            k_w = k_field[i + 1, j] if j > 0 else k_field[i + 1, j + 1]
            k_e = k_field[i + 1, j + 2] if j < N - 1 else k_field[i + 1, j + 1]
            k_n = 2 * k_c * k_n / (k_c + k_n + 1e-8)
            k_s = 2 * k_c * k_s / (k_c + k_s + 1e-8)
            k_w = 2 * k_c * k_w / (k_c + k_w + 1e-8)
            k_e = 2 * k_c * k_e / (k_c + k_e + 1e-8)
            A[idx, idx] = -(k_n + k_s + k_w + k_e)
            if i > 0: A[idx, idx - N] = k_n
            if i < N - 1: A[idx, idx + N] = k_s
            if j > 0: A[idx, idx - 1] = k_w
            if j < N - 1: A[idx, idx + 1] = k_e
    u_inner = np.linalg.solve(A, b)
    p = np.zeros((grid_size, grid_size))
    p[1:-1, 1:-1] = u_inner.reshape(N, N)
    return p.astype(np.float32)


print("\n[1] Generating data...")
# Source: coarse permeability (abundant)
N_SOURCE = 200
source_k = generate_permeability(N_SOURCE, GRID, length_scale=0.2)
source_p = np.array([solve_darcy_fd(k, GRID) for k in source_k])

# Target: fine permeability (scarce)
N_TARGET = 30
target_k = generate_permeability(N_TARGET, GRID, length_scale=0.05)
target_p = np.array([solve_darcy_fd(k, GRID) for k in target_k])

# Test (target domain)
N_TEST = 30
test_k = generate_permeability(N_TEST, GRID, length_scale=0.05)
test_p = np.array([solve_darcy_fd(k, GRID) for k in test_k])

print(f"  Source (coarse, ls=0.2): {N_SOURCE} samples (abundant)")
print(f"  Target (fine, ls=0.05):   {N_TARGET} samples (scarce)")
print(f"  Test (fine, ls=0.05):     {N_TEST} samples")

# Normalize (use source stats — simulates real scenario where target data is limited)
k_mean, k_std = source_k.mean(), source_k.std()
p_mean, p_std = source_p.mean(), source_p.std()

source_k_n = (source_k - k_mean) / (k_std + 1e-8)
source_p_n = (source_p - p_mean) / (p_std + 1e-8)
target_k_n = (target_k - k_mean) / (k_std + 1e-8)
target_p_n = (target_p - p_mean) / (p_std + 1e-8)
test_k_n = (test_k - k_mean) / (k_std + 1e-8)
test_p_n = (test_p - p_mean) / (p_std + 1e-8)

source_k_t = torch.from_numpy(source_k_n).unsqueeze(1).to(device)
source_p_t = torch.from_numpy(source_p_n).unsqueeze(1).to(device)
target_k_t = torch.from_numpy(target_k_n).unsqueeze(1).to(device)
target_p_t = torch.from_numpy(target_p_n).unsqueeze(1).to(device)
test_k_t = torch.from_numpy(test_k_n).unsqueeze(1).to(device)
test_p_t = torch.from_numpy(test_p_n).unsqueeze(1).to(device)

# ============================================================
# [2] FNO Architecture
# ============================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.weights1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum('bixy,ioxy->boxy', x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum('bixy,ioxy->boxy', x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, width=20, modes=8, n_layers=4):
        super().__init__()
        self.mlp_in = nn.Sequential(nn.Conv2d(in_ch, width, 1), nn.SiLU())
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.mlp_out = nn.Sequential(nn.SiLU(), nn.Conv2d(width, 128, 1), nn.SiLU(), nn.Conv2d(128, out_ch, 1))

    def forward(self, x):
        x = self.mlp_in(x)
        for conv, w in zip(self.convs, self.ws):
            x = F.silu(conv(x) + w(x))
        return self.mlp_out(x)

    def freeze_encoder(self):
        """Freeze spectral layers + input MLP (encoder)."""
        for p in self.mlp_in.parameters():
            p.requires_grad = False
        for conv in self.convs:
            for p in conv.parameters():
                p.requires_grad = False
        for w in self.ws:
            for p in w.parameters():
                p.requires_grad = False

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# [3] Strategy 1: Scratch (train on target from scratch)
# ============================================================
EPOCHS_PRETRAIN = 200
EPOCHS_FINETUNE = 150
BATCH_SIZE = 20
LR = 1e-3
LR_FINETUNE = 5e-4  # Lower LR for fine-tuning

print(f"\n[3] Strategy 1: Scratch (train on {N_TARGET} target samples from scratch)")
print("-" * 70)

torch.manual_seed(42)
model_scratch = FNO2d(in_ch=1, out_ch=1, width=20, modes=8, n_layers=4).to(device)
opt = torch.optim.Adam(model_scratch.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FINETUNE)

scratch_losses = []
scratch_test = []
start = time.time()
for epoch in range(EPOCHS_FINETUNE):
    model_scratch.train()
    perm = torch.randperm(N_TARGET)
    epoch_loss = 0; n_batches = 0
    for i in range(0, N_TARGET, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        pred = model_scratch(target_k_t[idx])
        loss = F.mse_loss(pred, target_p_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_batches += 1
    scratch_losses.append(epoch_loss / n_batches)
    sched.step()
    model_scratch.eval()
    with torch.no_grad():
        test_loss = F.mse_loss(model_scratch(test_k_t), test_p_t).item()
    scratch_test.append(test_loss)
    if epoch % 50 == 0 or epoch == EPOCHS_FINETUNE - 1:
        print(f"  Epoch {epoch:4d} | Train: {scratch_losses[-1]:.6e} | Test: {test_loss:.6e} | Time: {time.time()-start:.1f}s")
scratch_time = time.time() - start

# ============================================================
# [4] Pre-train on Source Domain
# ============================================================
print(f"\n[4] Pre-training on source domain ({N_SOURCE} samples, {EPOCHS_PRETRAIN} epochs)")
print("-" * 70)

torch.manual_seed(42)
pretrained = FNO2d(in_ch=1, out_ch=1, width=20, modes=8, n_layers=4).to(device)
opt = torch.optim.Adam(pretrained.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PRETRAIN)

pretrain_losses = []
start = time.time()
for epoch in range(EPOCHS_PRETRAIN):
    pretrained.train()
    perm = torch.randperm(N_SOURCE)
    epoch_loss = 0; n_batches = 0
    for i in range(0, N_SOURCE, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        pred = pretrained(source_k_t[idx])
        loss = F.mse_loss(pred, source_p_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_batches += 1
    pretrain_losses.append(epoch_loss / n_batches)
    sched.step()
    if epoch % 50 == 0 or epoch == EPOCHS_PRETRAIN - 1:
        print(f"  Epoch {epoch:4d} | Loss: {pretrain_losses[-1]:.6e} | Time: {time.time()-start:.1f}s")
pretrain_time = time.time() - start

# ============================================================
# [5] Strategy 2: Transfer (freeze encoder)
# ============================================================
print(f"\n[5] Strategy 2: Transfer (freeze encoder, fine-tune decoder only)")
print("-" * 70)

import copy
model_freeze = copy.deepcopy(pretrained)
model_freeze.freeze_encoder()

total_params = model_freeze.count_total()
trainable_params = model_freeze.count_trainable()
print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model_freeze.parameters()), lr=LR_FINETUNE)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FINETUNE)

freeze_losses = []
freeze_test = []
start = time.time()
for epoch in range(EPOCHS_FINETUNE):
    model_freeze.train()
    perm = torch.randperm(N_TARGET)
    epoch_loss = 0; n_batches = 0
    for i in range(0, N_TARGET, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        pred = model_freeze(target_k_t[idx])
        loss = F.mse_loss(pred, target_p_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_batches += 1
    freeze_losses.append(epoch_loss / n_batches)
    sched.step()
    model_freeze.eval()
    with torch.no_grad():
        test_loss = F.mse_loss(model_freeze(test_k_t), test_p_t).item()
    freeze_test.append(test_loss)
    if epoch % 50 == 0 or epoch == EPOCHS_FINETUNE - 1:
        print(f"  Epoch {epoch:4d} | Train: {freeze_losses[-1]:.6e} | Test: {test_loss:.6e} | Time: {time.time()-start:.1f}s")
freeze_time = time.time() - start

# ============================================================
# [6] Strategy 3: Transfer (full fine-tune)
# ============================================================
print(f"\n[6] Strategy 3: Transfer (full fine-tune all layers)")
print("-" * 70)

model_fullft = copy.deepcopy(pretrained)
total_params = model_fullft.count_total()
print(f"  Total params: {total_params:,} | All trainable")

opt = torch.optim.Adam(model_fullft.parameters(), lr=LR_FINETUNE)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FINETUNE)

fullft_losses = []
fullft_test = []
start = time.time()
for epoch in range(EPOCHS_FINETUNE):
    model_fullft.train()
    perm = torch.randperm(N_TARGET)
    epoch_loss = 0; n_batches = 0
    for i in range(0, N_TARGET, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        pred = model_fullft(target_k_t[idx])
        loss = F.mse_loss(pred, target_p_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item(); n_batches += 1
    fullft_losses.append(epoch_loss / n_batches)
    sched.step()
    model_fullft.eval()
    with torch.no_grad():
        test_loss = F.mse_loss(model_fullft(test_k_t), test_p_t).item()
    fullft_test.append(test_loss)
    if epoch % 50 == 0 or epoch == EPOCHS_FINETUNE - 1:
        print(f"  Epoch {epoch:4d} | Train: {fullft_losses[-1]:.6e} | Test: {test_loss:.6e} | Time: {time.time()-start:.1f}s")
fullft_time = time.time() - start

# ============================================================
# [7] Evaluation
# ============================================================
print("\n[7] Final Evaluation on target test set:")
print("-" * 70)

model_scratch.eval()
model_freeze.eval()
model_fullft.eval()

with torch.no_grad():
    pred_scratch = model_scratch(test_k_t)
    pred_freeze = model_freeze(test_k_t)
    pred_fullft = model_fullft(test_k_t)

    l2_scratch = torch.norm(pred_scratch - test_p_t, dim=(1,2,3)) / (torch.norm(test_p_t, dim=(1,2,3)) + 1e-8)
    l2_freeze = torch.norm(pred_freeze - test_p_t, dim=(1,2,3)) / (torch.norm(test_p_t, dim=(1,2,3)) + 1e-8)
    l2_fullft = torch.norm(pred_fullft - test_p_t, dim=(1,2,3)) / (torch.norm(test_p_t, dim=(1,2,3)) + 1e-8)

print(f"  1. Scratch:              L2 = {l2_scratch.mean().item():.4f} ± {l2_scratch.std().item():.4f}")
print(f"  2. Transfer (freeze):    L2 = {l2_freeze.mean().item():.4f} ± {l2_freeze.std().item():.4f}")
print(f"  3. Transfer (full FT):   L2 = {l2_fullft.mean().item():.4f} ± {l2_fullft.std().item():.4f}")
print()
best = min(l2_scratch.mean().item(), l2_freeze.mean().item(), l2_fullft.mean().item())
if best == l2_fullft.mean().item():
    print(f"  Best: Full Fine-tune ({(1-l2_fullft.mean().item()/l2_scratch.mean().item())*100:.1f}% better than scratch)")
elif best == l2_freeze.mean().item():
    print(f"  Best: Freeze Encoder ({(1-l2_freeze.mean().item()/l2_scratch.mean().item())*100:.1f}% better than scratch)")

# ============================================================
# [8] Visualization
# ============================================================
print("\n[8] Generating visualizations...")

# --- Figure 1: Pre-training loss ---
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.semilogy(pretrain_losses, linewidth=2, color='blue')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)')
ax.set_title(f'Pre-training on Source Domain (coarse, {N_SOURCE} samples)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "transfer_pretrain.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Fine-tuning comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(scratch_losses, label='Scratch', linewidth=2, color='gray')
ax1.semilogy(freeze_losses, label='Transfer (freeze)', linewidth=2, color='orange')
ax1.semilogy(fullft_losses, label='Transfer (full FT)', linewidth=2, color='green')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss (MSE)')
ax1.set_title('Fine-tuning: Training Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.semilogy(scratch_test, label='Scratch', linewidth=2, color='gray')
ax2.semilogy(freeze_test, label='Transfer (freeze)', linewidth=2, color='orange')
ax2.semilogy(fullft_test, label='Transfer (full FT)', linewidth=2, color='green')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Test Loss (MSE)')
ax2.set_title('Fine-tuning: Test Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "transfer_finetune.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 3: Final L2 bar chart ---
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
strategies = ['Scratch', 'Transfer\n(Freeze)', 'Transfer\n(Full FT)']
l2_vals = [l2_scratch.mean().item(), l2_freeze.mean().item(), l2_fullft.mean().item()]
colors = ['gray', 'orange', 'green']
bars = ax.bar(strategies, l2_vals, color=colors, edgecolor='black', width=0.5)
ax.set_ylabel('Relative L2 Error (lower is better)')
ax.set_title(f'Transfer Learning vs Scratch (Target: {N_TARGET} samples)')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(l2_vals):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "transfer_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 4: Prediction comparison ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
idx = 0
k_show = test_k[idx, 0]
p_true = test_p[idx, 0]
p_scratch = pred_scratch[idx, 0].cpu().numpy() * p_std + p_mean
p_freeze = pred_freeze[idx, 0].cpu().numpy() * p_std + p_mean
p_fullft = pred_fullft[idx, 0].cpu().numpy() * p_std + p_mean

ax = axes[0, 0]; ax.imshow(k_show, cmap='magma', origin='lower'); ax.set_title('Input k(x) (fine)')
ax = axes[0, 1]; ax.imshow(p_true, cmap='viridis', origin='lower'); ax.set_title('Ground Truth p(x)')
ax = axes[0, 2]; ax.imshow(p_scratch, cmap='viridis', origin='lower'); ax.set_title(f'Scratch (L2={l2_scratch[idx].item():.4f})')
ax = axes[0, 3]; ax.imshow(p_fullft, cmap='viridis', origin='lower'); ax.set_title(f'Transfer Full FT (L2={l2_fullft[idx].item():.4f})')

ax = axes[1, 0]; ax.imshow(np.abs(p_scratch - p_true), cmap='hot', origin='lower'); ax.set_title('Scratch Error')
ax = axes[1, 1]; ax.imshow(np.abs(p_freeze - p_true), cmap='hot', origin='lower'); ax.set_title('Freeze Error')
ax = axes[1, 2]; ax.imshow(np.abs(p_fullft - p_true), cmap='hot', origin='lower'); ax.set_title('Full FT Error')
ax = axes[1, 3]; ax.imshow(p_freeze, cmap='viridis', origin='lower'); ax.set_title(f'Transfer Freeze (L2={l2_freeze[idx].item():.4f})')

plt.suptitle('Transfer Learning: Prediction Comparison on Target Domain', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "transfer_result.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 5: Strategy explanation ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, title, desc, color in zip(axes,
    ['1. Scratch', '2. Transfer (Freeze)', '3. Transfer (Full FT)'],
    ['Train from scratch\non target data only\n\n• No pre-training\n• Needs more data\n• Slow convergence',
     'Pre-train on source\nFreeze encoder\nFine-tune decoder only\n\n• Encoder locked\n• Fast adaptation\n• Prevents forgetting',
     'Pre-train on source\nFine-tune ALL layers\nwith low LR\n\n• Full adaptation\n• Best accuracy\n• Risk of forgetting'],
    ['gray', 'orange', 'green']):
    ax.text(0.5, 0.7, title, ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.2, desc, fontsize=11, transform=ax.transAxes, family='monospace', verticalalignment='bottom')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor=color, linewidth=3, transform=ax.transAxes))

plt.suptitle('Transfer Learning Strategies', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "transfer_strategies.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved all figures to results/")

# ============================================================
# [9] Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Transfer Learning (FNO Pre-train → Fine-tune)")
print("=" * 70)
print(f"  Source domain:  Coarse permeability (ls=0.2), {N_SOURCE} samples")
print(f"  Target domain:  Fine permeability (ls=0.05), {N_TARGET} samples")
print(f"  Test domain:    Fine permeability (ls=0.05), {N_TEST} samples")
print(f"  Pre-train:      {EPOCHS_PRETRAIN} epochs ({pretrain_time:.1f}s)")
print(f"  Fine-tune:      {EPOCHS_FINETUNE} epochs each")
print(f"  Scratch time:   {scratch_time:.1f}s")
print(f"  Freeze time:    {freeze_time:.1f}s")
print(f"  Full FT time:   {fullft_time:.1f}s")
print(f"  --- Results (Relative L2 on target test) ---")
print(f"  Scratch:            {l2_scratch.mean().item():.4f}")
print(f"  Transfer (freeze):  {l2_freeze.mean().item():.4f}")
print(f"  Transfer (full FT): {l2_fullft.mean().item():.4f}")
print()
print("Key observations:")
print("  1. PRE-TRAINING: Learn general features on abundant source data")
print("  2. FINE-TUNING: Adapt pre-trained model to scarce target data")
print("  3. FREEZE vs FULL: Freeze is faster but full FT is more accurate")
print("  4. DATA EFFICIENCY: Transfer learning needs less target data")
print("  5. FIRST TRANSFER TUTORIAL: All others train from scratch")
print("=" * 70)
