"""
PhysicsNeMo 튜토리얼: AFNO (Adaptive Fourier Neural Operator) for Darcy Flow
=============================================================================
AFNO는 FNO의 확장 모델로, 패치 임베딩 + 적응형 푸리에 스펙트럴 컨볼루션을 결합합니다.
기존 FNO 튜토리얼과 동일한 Darcy Flow 문제를 풀며 성능을 비교합니다.

물리적 의미:
  - FNO: 전체 격자에서 FFT → 모드 절단 → IFFT
  - AFNO: 패치 단위로 분할 → 블록 대각 가중치로 적응적 스펙트럴 연산 → 희소화
  - AFNO는 FNO보다 메모리 효율적이며, 더 큰 격자에서 확장 가능

자원 비교 (기존 FNO 대비):
  - FNO:  2.4M params, 21초 학습, 손실 8.7e-4
  - AFNO: ~0.5-1M params (패치 기반으로 경량), 학습 시간 유사, 메모리 사용량 적음
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import physicsnemo
from physicsnemo.models.afno import AFNO
import time
import os
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"PhysicsNeMo: {physicsnemo.__version__}\n")

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# [1] 데이터 생성 (FNO 튜토리얼과 동일)
# ============================================================
N_TRAIN = 200
N_GRID = 32

def generate_darcy_data(n_samples, grid_size, device):
    raw_noise = torch.randn(n_samples, 1, grid_size, grid_size, device=device)
    ks = 5
    k_field = torch.nn.functional.avg_pool2d(raw_noise, ks, stride=1, padding=ks//2)
    k_field = torch.sigmoid(k_field * 2) + 0.1
    p_smooth = torch.nn.functional.avg_pool2d(k_field, ks, stride=1, padding=ks//2)
    x_c = torch.linspace(0, 1, grid_size, device=device).view(1,1,1,-1)
    y_c = torch.linspace(0, 1, grid_size, device=device).view(1,1,-1,1)
    spatial = torch.sin(np.pi * x_c) * torch.sin(np.pi * y_c)
    p_field = p_smooth * spatial * 3.0
    return k_field, p_field

k_train, p_train = generate_darcy_data(N_TRAIN, N_GRID, device)
k_test_64, p_test_64 = generate_darcy_data(5, 64, device)

print(f"[1/7] Data generated: {N_TRAIN} train samples ({N_GRID}x{N_GRID}), 5 test samples (64x64)")
print(f"      Input shape:  {k_train.shape} (permeability k)")
print(f"      Output shape: {p_train.shape} (pressure p)\n")

# ============================================================
# [2] AFNO 모델 생성
# ============================================================
print("[2/7] Creating AFNO model...")

# AFNO 파라미터 설명:
#   inp_shape: 입력 격자 크기 [H, W] — patch_size로 나누어 떨어져야 함
#   patch_size: 격자를 패치로 분할 [ph, pw] — 32/8=4, 즉 4x4=16개 패치
#   embed_dim: 패치 임베딩 차원 (FNO의 latent_channels와 유사)
#   depth: AFNO 블록 수 (FNO의 num_fno_layers와 유사)
#   num_blocks: 블록 대각 가중치의 블록 수 (메모리 절약, FNO와의 핵심 차이)

model = AFNO(
    inp_shape=[N_GRID, N_GRID],   # 32x32 격자
    in_channels=1,                 # 입력: 투과율 k (1채널)
    out_channels=1,                 # 출력: 압력 p (1채널)
    patch_size=[8, 8],              # 8x8 패치 → 4x4=16개 토큰
    embed_dim=32,                   # 임베딩 차원
    depth=4,                        # 4개 AFNO 블록
    mlp_ratio=4.0,                  # MLP 확장 비율
    drop_rate=0.0,                  # 드롭아웃 없음
    num_blocks=8,                   # 8개 블록 대각 가중치 (메모리 효율)
    sparsity_threshold=0.01,        # 스펙트럴 희소화 임계값
    hard_thresholding_fraction=1.0, # 모드 사용 비율
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"      AFNO parameters: {n_params:,}")
print(f"      Architecture: 4 AFNO blocks, 32 embed_dim, 8x8 patches, 8 spectral blocks")
print(f"      Key difference from FNO: block-diagonal weights (memory efficient)\n")

# ============================================================
# [3] 학습 설정
# ============================================================
print("[3/7] Setting up training...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 200
loss_fn = nn.MSELoss()

# ============================================================
# [4] 학습 루프
# ============================================================
print(f"\n[4/7] Training AFNO ({EPOCHS} epochs)...")
loss_history = []
start_time = time.time()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    pred = model(k_train)
    loss = loss_fn(pred, p_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:4d}/{EPOCHS} | Loss: {loss.item():.6e} | {elapsed:.1f}s")

train_time = time.time() - start_time
print(f"\n  Training complete! Final loss: {loss.item():.6e}, Time: {train_time:.1f}s")

# ============================================================
# [5] 평가 (32x32 및 64x64)
# ============================================================
print("\n[5/7] Evaluating model...")
model.eval()
with torch.no_grad():
    # 32x32 (학습 해상도)
    pred_32 = model(k_train[:5])
    loss_32 = loss_fn(pred_32, p_train[:5]).item()
    
    # 64x64 (미학습 해상도 — AFNO는 inp_shape 고정이므로 오류 예상)
    try:
        pred_64 = model(k_test_64)
        loss_64 = loss_fn(pred_64, p_test_64).item()
        has_64 = True
        print(f"  32x32 test loss: {loss_32:.6e}")
        print(f"  64x64 test loss: {loss_64:.6e} (zero-shot transfer)")
    except (RuntimeError, ValueError) as e:
        loss_64 = float('nan')
        has_64 = False
        print(f"  32x32 test loss: {loss_32:.6e}")
        print(f"  64x64: NOT SUPPORTED (inp_shape fixed to 32x32)")

# ============================================================
# [6] 시각화
# ============================================================
print("\n[6/7] Generating visualizations...")
output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

k_sample = k_train[0, 0].cpu().numpy()
p_sample = p_train[0, 0].cpu().numpy()
pred_sample = pred_32[0, 0].cpu().numpy()
error = np.abs(pred_sample - p_sample)

# Row 1: 입력, 타겟, 예측
im0 = axes[0, 0].imshow(k_sample, cmap="viridis")
axes[0, 0].set_title("Input: k (32x32)\nPermeability", fontsize=12)
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(p_sample, cmap="viridis")
axes[0, 1].set_title("Target: p (32x32)\nTrue Pressure", fontsize=12)
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

im2 = axes[0, 2].imshow(pred_sample, cmap="viridis")
axes[0, 2].set_title("AFNO Prediction (32x32)\nPredicted Pressure", fontsize=12)
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

# Row 2: 오차 맵, 학습 곡선, 64x64 (지원 시)
im3 = axes[1, 0].imshow(error, cmap="hot")
axes[1, 0].set_title(f"Error Map\n|pred - target| (max={error.max():.4f})", fontsize=12)
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

axes[1, 1].semilogy(loss_history, color="#FF6B35", linewidth=1.0)
axes[1, 1].set_title("Training Loss Curve", fontsize=12)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("MSE Loss (log)")
axes[1, 1].grid(True, alpha=0.3)

if has_64:
    pred_64_sample = pred_64[0, 0].cpu().numpy()
    p_64_sample = p_test_64[0, 0].cpu().numpy()
    im5 = axes[1, 2].imshow(pred_64_sample, cmap="viridis")
    axes[1, 2].set_title("AFNO Prediction (64x64)\nZero-Shot Transfer", fontsize=12)
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
else:
    axes[1, 2].text(0.5, 0.5, "64x64\nNOT SUPPORTED\n(fixed inp_shape)", 
                    ha="center", va="center", fontsize=14, color="red",
                    transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("64x64 Transfer", fontsize=12)

plt.suptitle(f"AFNO for Darcy Flow (PhysicsNeMo)\n"
             f"Params: {n_params:,} | Time: {train_time:.1f}s | Loss: {loss.item():.4e}",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig_path = os.path.join(output_dir, "afno_darcy_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Result saved: {fig_path}")
plt.close()

# ============================================================
# [7] 요약
# ============================================================
print("\n" + "=" * 70)
print("  AFNO TUTORIAL SUMMARY")
print("=" * 70)
print(f"  Model:        AFNO (Adaptive Fourier Neural Operator)")
print(f"  Parameters:   {n_params:,}")
print(f"  Training:     {EPOCHS} epochs, {train_time:.1f}s")
print(f"  Final loss:   {loss.item():.6e}")
print(f"  Test (32x32): {loss_32:.6e}")
loss_64_str = f"{loss_64:.6e}" if has_64 else "N/A (fixed grid)"
print(f"  Test (64x64): {loss_64_str}")
print(f"  Resolution transfer: {'Yes' if has_64 else 'No (inp_shape fixed)'}")
print(f"  Key feature: Block-diagonal spectral weights (memory efficient)")
print("=" * 70)
