"""
PhysicsNeMo 튜토리얼: SRRN (Super-Resolution Residual Network) for Darcy Flow
=============================================================================
SRRN은 저해상도 PDE 해를 고해상도로 업스케일하는 초해상도 모델입니다.
32x32 압력장을 입력으로 받아 128x128 고해상도 압력장을 생성합니다.

물리적 의미:
  - 기존 FNO/U-Net: 투과율 k → 압력 p (해상도 유지)
  - SRRN: 저해상도 압력 p_low → 고해상도 압력 p_high (해상도 향상)
  - "저렴한 coarse mesh 시뮬레이션 → 딥러닝으로 fine mesh 보간" 패러다임
  - 실제 산업 응용: 빠른 저해상도 시뮬레이션 후 SRRN으로 정밀화

자원 비교 (기존 FNO 대비):
  - FNO:  2.4M params, 21초, 32→64 전이 시 손실 150x 증가
  - SRRN: ~0.5M params (잔차 블록 기반, 가벼움), 학습 2-5분, 32→128 직접 학습
  - SRRN은 업스케일링 자체가 목적이므로 고해상도에서 더 정확
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import physicsnemo
from physicsnemo.models.srrn import SRResNet
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
# [1] 데이터 생성: 저해상도(32x32) → 고해상도(128x128) 쌍
# ============================================================
N_TRAIN = 200
LR_GRID = 32    # 저해상도 격자
HR_GRID = 128   # 고해상도 격자 (4배 업스케일)
DEPTH = 1       # 3D 깊이 (SRRN은 3D Conv 사용, depth=1로 공간만 업스케일)

def generate_darcy_data(n_samples, grid_size, device):
    """Darcy Flow 합성 데이터 생성"""
    raw_noise = torch.randn(n_samples, 1, grid_size, grid_size, device=device)
    ks = 5
    k_field = torch.nn.functional.avg_pool2d(raw_noise, ks, stride=1, padding=ks//2)
    k_field = torch.sigmoid(k_field * 2) + 0.1
    p_smooth = torch.nn.functional.avg_pool2d(k_field, ks, stride=1, padding=ks//2)
    x_c = torch.linspace(0, 1, grid_size, device=device).view(1,1,1,-1)
    y_c = torch.linspace(0, 1, grid_size, device=device).view(1,1,-1,1)
    spatial = torch.sin(np.pi * x_c) * torch.sin(np.pi * y_c)
    p_field = p_smooth * spatial * 3.0
    return p_field

def to_3d(tensor_2d, depth=4):
    """2D [B,1,H,W] → 3D [B,1,D,H,W]"""
    return tensor_2d.unsqueeze(-1).expand(-1,-1,-1,-1,depth).contiguous()

def from_3d(tensor_3d):
    """3D [B,1,D,H,W] → 2D [B,1,H,W] (depth 차원 평균)"""
    return tensor_3d.mean(dim=-1)

# 고해상도 데이터 생성 (128x128)
p_hr_train = generate_darcy_data(N_TRAIN, HR_GRID, device)
p_hr_train_3d = to_3d(p_hr_train, DEPTH)

# 저해상도 데이터: 고해상도를 avg_pool로 다운샘플링 (32x32)
p_lr_train = torch.nn.functional.avg_pool2d(p_hr_train, kernel_size=4, stride=4)
p_lr_train_3d = to_3d(p_lr_train, DEPTH)

# 테스트 데이터
p_hr_test = generate_darcy_data(5, HR_GRID, device)
p_hr_test_3d = to_3d(p_hr_test, DEPTH)
p_lr_test = torch.nn.functional.avg_pool2d(p_hr_test, kernel_size=4, stride=4)
p_lr_test_3d = to_3d(p_lr_test, DEPTH)

print(f"[1/7] Data generated:")
print(f"      Training: {N_TRAIN} samples")
print(f"      Low-res:  {p_lr_train_3d.shape} (32x32x{DEPTH})")
print(f"      High-res: {p_hr_train_3d.shape} (128x128x{DEPTH})")
print(f"      Scaling factor: {HR_GRID // LR_GRID}x\n")

# ============================================================
# [2] SRRN 모델 생성
# ============================================================
print("[2/7] Creating SRRN (Super-Resolution Residual Network) model...")

# SRRN 파라미터 설명:
#   in_channels: 입력 채널 (저해상도 압력장, 1채널)
#   out_channels: 출력 채널 (고해상도 압력장, 1채널)
#   large_kernel_size: 첫/마지막 컨볼루션 커널 (기본 7)
#   small_kernel_size: 내부 컨볼루션 커널 (기본 3)
#   conv_layer_size: 잠재 채널 크기 (기본 32)
#   n_resid_blocks: 잔차 블록 수 (기본 8)
#   scaling_factor: 업스케일 비율 (2, 4, 8 중 선택) — 4배 업스케일

model = SRResNet(
    in_channels=1,
    out_channels=1,
    large_kernel_size=7,
    small_kernel_size=3,
    conv_layer_size=32,
    n_resid_blocks=4,          # 4개 잔차 블록 (학습 속도를 위해 감소)
    scaling_factor=4,           # 4배 업스케일 (32→128)
    activation_fn="prelu",
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"      SRRN parameters: {n_params:,}")
print(f"      Architecture: 4 residual blocks, 32 latent channels, 4x super-resolution")
print(f"      Input:  [B, 1, 32, 32, 4] (low-res 3D)")
print(f"      Output: [B, 1, 128, 128, 4] (high-res 3D)\n")

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
print(f"\n[4/7] Training SRRN ({EPOCHS} epochs)...")
loss_history = []
start_time = time.time()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    pred = model(p_lr_train_3d)
    loss = loss_fn(pred, p_hr_train_3d)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:4d}/{EPOCHS} | Loss: {loss.item():.6e} | {elapsed:.1f}s")

train_time = time.time() - start_time
print(f"\n  Training complete! Final loss: {loss.item():.6e}, Time: {train_time:.1f}s")

# ============================================================
# [5] 평가
# ============================================================
print("\n[5/7] Evaluating model...")
model.eval()
with torch.no_grad():
    # 테스트 셋 평가
    pred_test = model(p_lr_test_3d)
    loss_test = loss_fn(pred_test, p_hr_test_3d).item()
    
    # 2D로 변환 (depth 차원 평균)
    pred_test_2d = from_3d(pred_test)
    p_hr_test_2d = p_hr_test
    p_lr_test_2d = p_lr_test
    
    # 업샘플링 베이스라인 (bilinear)과 비교
    pred_bilinear = torch.nn.functional.interpolate(
        p_lr_test_2d, scale_factor=4, mode="bilinear", align_corners=False
    )
    loss_bilinear = loss_fn(pred_bilinear, p_hr_test_2d).item()
    
    print(f"  SRRN test loss:      {loss_test:.6e}")
    print(f"  Bilinear baseline:   {loss_bilinear:.6e}")
    print(f"  SRRN improvement:    {loss_bilinear/loss_test:.2f}x better than bilinear")

# ============================================================
# [6] 시각화
# ============================================================
print("\n[6/7] Generating visualizations...")
output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 첫 번째 테스트 샘플
lr_sample = p_lr_test_2d[0, 0].cpu().numpy()
hr_sample = p_hr_test_2d[0, 0].cpu().numpy()
srrn_sample = pred_test_2d[0, 0].cpu().numpy()
bilinear_sample = pred_bilinear[0, 0].cpu().numpy()

# Row 1: 저해상도 입력, 고해상도 타겟, SRRN 예측
im0 = axes[0, 0].imshow(lr_sample, cmap="viridis")
axes[0, 0].set_title(f"Input: Low-Res (32x32)\nCoarse Simulation", fontsize=12)
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(hr_sample, cmap="viridis")
axes[0, 1].set_title(f"Target: High-Res (128x128)\nFine Simulation", fontsize=12)
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

im2 = axes[0, 2].imshow(srrn_sample, cmap="viridis")
axes[0, 2].set_title(f"SRRN Prediction (128x128)\nSuper-Resolved", fontsize=12)
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

# Row 2: 오차 맵 (SRRN vs bilinear), 학습 곡선, 확대 비교
err_srrn = np.abs(srrn_sample - hr_sample)
err_bilinear = np.abs(bilinear_sample - hr_sample)

im3 = axes[1, 0].imshow(err_srrn, cmap="hot", vmin=0, vmax=max(err_srrn.max(), err_bilinear.max()))
axes[1, 0].set_title(f"SRRN Error\n(max={err_srrn.max():.4f})", fontsize=12)
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

im4 = axes[1, 1].imshow(err_bilinear, cmap="hot", vmin=0, vmax=max(err_srrn.max(), err_bilinear.max()))
axes[1, 1].set_title(f"Bilinear Error\n(max={err_bilinear.max():.4f})", fontsize=12)
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

axes[1, 2].semilogy(loss_history, color="#9C27B0", linewidth=1.0, label="SRRN")
axes[1, 2].axhline(y=loss_bilinear, color="gray", linestyle="--", label=f"Bilinear baseline")
axes[1, 2].set_title("Training Loss Curve", fontsize=12)
axes[1, 2].set_xlabel("Epoch")
axes[1, 2].set_ylabel("MSE Loss (log)")
axes[1, 2].legend(fontsize=10)
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle(f"SRRN Super-Resolution for Darcy Flow (PhysicsNeMo)\n"
             f"Params: {n_params:,} | Time: {train_time:.1f}s | "
             f"SRRN Loss: {loss_test:.4e} vs Bilinear: {loss_bilinear:.4e}",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig_path = os.path.join(output_dir, "srrn_superres_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Result saved: {fig_path}")
plt.close()

# ============================================================
# [7] 요약
# ============================================================
print("\n" + "=" * 70)
print("  SRRN SUPER-RESOLUTION TUTORIAL SUMMARY")
print("=" * 70)
print(f"  Model:          SRRN (Super-Resolution Residual Network)")
print(f"  Parameters:     {n_params:,}")
print(f"  Training:       {EPOCHS} epochs, {train_time:.1f}s")
print(f"  Final loss:     {loss.item():.6e}")
print(f"  Test loss:      {loss_test:.6e}")
print(f"  Bilinear loss:  {loss_bilinear:.6e}")
print(f"  Improvement:    {loss_bilinear/loss_test:.2f}x better than bilinear")
print(f"  Scaling:        32x32 → 128x128 (4x super-resolution)")
print(f"  Key feature:    Sub-pixel convolution for efficient upscaling")
print("=" * 70)
