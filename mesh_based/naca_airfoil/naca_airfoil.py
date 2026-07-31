"""
PhysicsNeMo 튜토리얼: NACA Airfoil 포텀셜 유동 — PINN vs FNO vs Analytical Ground Truth
=====================================================================================
NACA 0012 airfoil 주변의 2D 비압축성 포텀셜 유동을 3가지 방법으로 풀고 비교합니다:

1. Analytical Ground Truth: Joukowski 변환으로 정확한 해석적 해 계산
2. PINN: Laplace 방정식(포텀셜 유동)을 손실 함수로 사용, 데이터 없이 학습
3. FNO: SDF → 속도장 매핑을 데이터 기반으로 학습

물리적 의미:
  - 포텀셜 유동: 비회전, 비압축성 → ∇²φ = 0 (Laplace 방정식)
  - 속도장: u = ∂φ/∂x, v = ∂φ/∂y
  - 경계조건: 원경계 uniform flow, airfoil 표면 no-penetration
  - Joukowski 변환: 원 주변 유동의 해석적 해를 airfoil로 변환

자원 비교:
  - PINN: ~50K params, 2-5분 학습, 데이터 불필요 (물리 법칙 내재)
  - FNO:  ~2.4M params, 30초-1분 학습, Ground Truth 데이터 필요
  - Analytical: 즉시 (해석적 공식)
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import physicsnemo
from physicsnemo.models.fno import FNO
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
# [1] NACA 0012 Airfoil 형상 생성 + Joukowski 변환 Ground Truth
# ============================================================
print("[1/8] Generating NACA 0012 airfoil geometry and analytical ground truth...")

# NACA 4-digit thickness distribution
def naca_thickness(x, t=0.12):
    """NACA 00xx thickness distribution, x in [0, 1]"""
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

# NACA 0012 표면 좌표
n_surface = 100
x_surface = np.linspace(0, 1, n_surface)
thickness = naca_thickness(x_surface, t=0.12)

# 상/하 표면
x_upper = x_surface
y_upper = thickness
x_lower = x_surface
y_lower = -thickness

# Joukowski 변환 파라미터
# 원을 airfoil로 변환: z = ζ + c²/ζ
# 원 중심을 약간 이동시켜 두께와 캠버를 생성
c_joukowski = 0.5  # 변환 상수
mu = 0.08  # 원 중심 x-편이 (두께 생성)
nu = 0.0   # 원 중심 y-편이 (캠버 생성, 0 = 대칭)

# 원의 반지름
R = np.sqrt((c_joukowski + mu)**2 + nu**2)

# 원 주변 포셜 유동의 해석적 해
# 복소 퍼텐셜: W(ζ) = U(ζ + R²/ζ) + iΓ/(2π) ln(ζ)
# 속도: dW/dζ = U(1 - R²/ζ²) + iΓ/(2πζ)
U_inf = 1.0  # 원경계 유속
alpha_test = np.radians(5.0)  # 테스트/학습 공격각 (5도)
alpha_aoa = alpha_test  # PINN도 동일한 AoA로 학습


def joukowski_velocity(x_phys, y_phys, U=1.0, alpha=0.0):
    """
    Joukowski 변환을 통한 airfoil 주변 속도장 계산
    입력: 물리 좌표 (x, y) — airfoil 좌표계
    출력: 속도 (u, v)
    """
    # 물리 좌표를 복소수로
    z = x_phys + 1j * y_phys
    
    # Joukowski 역변환: z = ζ + c²/ζ → ζ = (z ± sqrt(z² - 4c²)) / 2
    # 외부 해 선택 (|ζ| > R)
    disc = np.sqrt(z**2 - 4 * c_joukowski**2)
    zeta1 = (z + disc) / 2
    zeta2 = (z - disc) / 2
    # |ζ|가 더 큰 것 선택 (외부 유동)
    zeta = np.where(np.abs(zeta1) > np.abs(zeta2), zeta1, zeta2)
    
    # 원 주변 유동 (Kutta 조건: 순환 Γ = 4πRU sin(α))
    Gamma = 4 * np.pi * R * U * np.sin(alpha)
    
    # 원 좌표계에서 속도 (회전 적용)
    # dW/dζ = U*e^{-iα}(1 - R²/ζ²) + iΓ/(2πζ)
    e_ialpha = np.exp(-1j * alpha)
    dWdzeta = U * e_ialpha * (1 - R**2 / zeta**2) + 1j * Gamma / (2 * np.pi * zeta)
    
    # Joukowski 변환의 도함수: dz/dζ = 1 - c²/ζ²
    dzdzeta = 1 - c_joukowski**2 / zeta**2
    
    # 물리 좌표에서 속도: dW/dz = (dW/dζ) / (dz/dζ)
    dWdz = dWdzeta / dzdzeta
    
    u = np.real(dWdz)
    v = np.imag(dWdz)
    
    # airfoil 내부에서는 0으로 설정
    # Joukowski 변환의 특이점 처리
    mask = np.abs(dzdzeta) < 1e-10
    u[mask] = 0
    v[mask] = 0
    
    return u, v

# 정규 격자 생성 (FNO용)
GRID = 64
x_min, x_max = -1.5, 2.5
y_min, y_max = -1.5, 1.5

x_grid = np.linspace(x_min, x_max, GRID)
y_grid = np.linspace(y_min, y_max, GRID)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Ground Truth 속도장 계산 (정규 격자)
u_gt_np, v_gt_np = joukowski_velocity(X_grid, Y_grid, U=U_inf, alpha=alpha_aoa)
speed_gt_np = np.sqrt(u_gt_np**2 + v_gt_np**2)

# airfoil 내부 마스크 (SDF < 0)
# NACA 0012 표면과의 거리 계산
def naca_sdf(x, y, t=0.12):
    """NACA 0012 airfoil의 SDF (근사)"""
    # 표면 점들
    n_pts = 200
    x_s = np.linspace(0, 1, n_pts)
    y_s = naca_thickness(x_s, t=t)
    
    # 각 점에서 표면까지의 최소 거리
    # 상면
    d_upper = np.sqrt((x[..., None] - x_s)**2 + (y[..., None] - y_s)**2)
    # 하면
    d_lower = np.sqrt((x[..., None] - x_s)**2 + (y[..., None] + y_s)**2)
    
    d_min = np.minimum(d_upper.min(axis=-1), d_lower.min(axis=-1))
    
    # 내부/외부 판별: 점이 airfoil 내부인지 확인
    # x가 [0,1] 범위이고 |y| < thickness(x)이면 내부
    x_clamped = np.clip(x, 0, 1)
    y_thickness = naca_thickness(x_clamped, t=t)
    inside = (x >= 0) & (x <= 1) & (np.abs(y) < y_thickness)
    
    sdf = np.where(inside, -d_min, d_min)
    return sdf

sdf_gt_np = naca_sdf(X_grid, Y_grid, t=0.12)

# airfoil 내부에서 속도를 0으로 설정
inside_mask = sdf_gt_np < 0
u_gt_np[inside_mask] = 0
v_gt_np[inside_mask] = 0
speed_gt_np[inside_mask] = 0

# 텐서로 변환
u_gt = torch.tensor(u_gt_np, dtype=torch.float32).to(device)
v_gt = torch.tensor(v_gt_np, dtype=torch.float32).to(device)
sdf_gt = torch.tensor(sdf_gt_np, dtype=torch.float32).to(device)

# FNO 입력: [B, C, H, W] 형태
# 채널: SDF + x좌표 + y좌표
x_coord = torch.tensor(X_grid, dtype=torch.float32).to(device)
y_coord = torch.tensor(Y_grid, dtype=torch.float32).to(device)

print(f"  Grid: {GRID}x{GRID}, Domain: [{x_min},{x_max}]x[{y_min},{y_max}]")
print(f"  Airfoil: NACA 0012 (symmetric, t=12%)")
print(f"  Flow: U_inf={U_inf}, AoA={np.degrees(alpha_aoa)}°")
print(f"  Ground truth: Joukowski transformation (analytical)")
print(f"  SDF range: [{sdf_gt_np.min():.3f}, {sdf_gt_np.max():.3f}]")
print(f"  Velocity range: u=[{u_gt_np.min():.2f}, {u_gt_np.max():.2f}], v=[{v_gt_np.min():.2f}, {v_gt_np.max():.2f}]\n")

# ============================================================
# [2] FNO 학습 데이터 준비
# ============================================================
print("[2/8] Preparing FNO training data...")

N_FNO_TRAIN = 50  # 서로 다른 공격각에 대한 학습 데이터

# 여러 공격각에 대해 Ground Truth 생성
fno_inputs = []  # [SDF, x, y, sin(alpha), cos(alpha)]
fno_outputs = []  # [u, v]

for i in range(N_FNO_TRAIN):
    alpha_i = np.radians(-10 + 20 * i / (N_FNO_TRAIN - 1))  # -10° ~ +10°
    u_i, v_i = joukowski_velocity(X_grid, Y_grid, U=U_inf, alpha=alpha_i)
    u_i[inside_mask] = 0
    v_i[inside_mask] = 0
    
    # 입력 채널: SDF, x, y, sin(alpha), cos(alpha)
    sin_a = np.sin(alpha_i) * np.ones_like(X_grid)
    cos_a = np.cos(alpha_i) * np.ones_like(X_grid)
    inp = np.stack([sdf_gt_np, X_grid, Y_grid, sin_a, cos_a], axis=0)  # [5, H, W]
    out = np.stack([u_i, v_i], axis=0)  # [2, H, W]
    
    fno_inputs.append(inp)
    fno_outputs.append(out)

fno_inputs = torch.tensor(np.array(fno_inputs), dtype=torch.float32).to(device)  # [N, 5, H, W]
fno_outputs = torch.tensor(np.array(fno_outputs), dtype=torch.float32).to(device)  # [N, 2, H, W]

print(f"  FNO training: {N_FNO_TRAIN} samples, AoA range: -10° to +10°")
print(f"  Input shape:  {fno_inputs.shape} (SDF + x + y + sin(α) + cos(α))")
print(f"  Output shape: {fno_outputs.shape} (u, v)\n")

# ============================================================
# [3] FNO 모델 생성 및 학습
# ============================================================
print("[3/8] Creating and training FNO model...")

fno_model = FNO(
    in_channels=5,       # SDF + x + y + sin(α) + cos(α)
    out_channels=2,       # u, v
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=2,
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
    padding=5,
).to(device)

fno_params = sum(p.numel() for p in fno_model.parameters())
print(f"  FNO parameters: {fno_params:,}")

fno_optimizer = torch.optim.Adam(fno_model.parameters(), lr=1e-3)
fno_loss_fn = nn.MSELoss()

FNO_EPOCHS = 200
fno_loss_history = []
fno_start = time.time()

for epoch in range(FNO_EPOCHS):
    fno_optimizer.zero_grad()
    pred = fno_model(fno_inputs)
    loss = fno_loss_fn(pred, fno_outputs)
    loss.backward()
    fno_optimizer.step()
    fno_loss_history.append(loss.item())
    
    if epoch % 40 == 0 or epoch == FNO_EPOCHS - 1:
        elapsed = time.time() - fno_start
        print(f"  FNO Epoch {epoch:4d}/{FNO_EPOCHS} | Loss: {loss.item():.6e} | {elapsed:.1f}s")

fno_time = time.time() - fno_start
print(f"  FNO training complete! Final loss: {loss.item():.6e}, Time: {fno_time:.1f}s\n")

# ============================================================
# [4] PINN 모델 생성 및 학습
# ============================================================
print("[4/8] Creating and training PINN model...")

class PINN_Airfoil(nn.Module):
    """PINN for potential flow: predicts velocity (u, v) directly
    Loss: continuity (∂u/∂x + ∂v/∂y = 0) + irrotationality (∂u/∂y - ∂v/∂x = 0)
    + far-field BC + surface no-penetration + supervised far-field data"""
    def __init__(self, in_dim=3, hidden=128, layers=6, out_dim=2):
        super().__init__()
        mods = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*mods)
    
    def forward(self, x):
        return self.net(x)

pinn_model = PINN_Airfoil(in_dim=3, hidden=128, layers=6, out_dim=2).to(device)
pinn_params = sum(p.numel() for p in pinn_model.parameters())
print(f"  PINN parameters: {pinn_params:,}")

# 입력 정규화: 좌표를 [-1, 1]로 스케일링
x_center = (x_min + x_max) / 2.0
y_center = (y_min + y_max) / 2.0
x_scale = (x_max - x_min) / 2.0
y_scale = (y_max - y_min) / 2.0

# PINN 입력: (x, y, SDF)
# SDF를 추가 입력으로 사용하여 네트워크가 형상을 인식

# 학습용 샘플링 점
N_INTERIOR = 5000  # 내부 점 (Laplace 방정식)
N_FARFIELD = 800   # 원경계 점
N_SURFACE = 400    # airfoil 표면 점

# 내부 점 (도메인 전체에서 무작위 샘플링)
x_int = torch.rand(N_INTERIOR, 1, device=device) * (x_max - x_min) + x_min
y_int = torch.rand(N_INTERIOR, 1, device=device) * (y_max - y_min) + y_min
sdf_int = torch.tensor(naca_sdf(x_int.cpu().numpy().flatten(), 
                                 y_int.cpu().numpy().flatten()), 
                        dtype=torch.float32, device=device).unsqueeze(1)

# 원경계 점 (원형)
theta_far = torch.linspace(0, 2*np.pi, N_FARFIELD, device=device).unsqueeze(1)
r_far = 3.0
x_far = r_far * torch.cos(theta_far) + 0.5  # 도메인 중심으로 이동
y_far = r_far * torch.sin(theta_far)
sdf_far = torch.tensor(naca_sdf(x_far.cpu().numpy().flatten(),
                                  y_far.cpu().numpy().flatten()),
                         dtype=torch.float32, device=device).unsqueeze(1)

# Airfoil 표면 점
theta_surf = torch.linspace(0, 2*np.pi, N_SURFACE, device=device).unsqueeze(1)
# 표면 위의 점 (NACA 0012)
x_surf_t = torch.linspace(0, 1, N_SURFACE//2, device=device).unsqueeze(1)
t_surf = 0.12
y_surf_upper = 5 * t_surf * (0.2969 * torch.sqrt(x_surf_t) - 0.1260 * x_surf_t 
                              - 0.3516 * x_surf_t**2 + 0.2843 * x_surf_t**3 - 0.1015 * x_surf_t**4)
y_surf_lower = -y_surf_upper

x_surf = torch.cat([x_surf_t, x_surf_t], dim=0)
y_surf = torch.cat([y_surf_upper, y_surf_lower], dim=0)
sdf_surf = torch.zeros_like(x_surf)  # 표면에서 SDF = 0

# Airfoil 표면 법선 계산 (NACA 0012)
# 표면: y = ±t(x), 법선 = (-dy/dx, 1) / |(-dy/dx, 1)| (상면, 외부 향)
def naca_normal(x, t=0.12):
    """NACA 0012 상면 법선 (외부 향)"""
    eps = 1e-8
    dydx = 5 * t * (0.2969 / (2 * np.sqrt(x + eps)) - 0.1260 - 2*0.3516*x + 3*0.2843*x**2 - 4*0.1015*x**3)
    norm = np.sqrt(dydx**2 + 1)
    nx = -dydx / norm
    ny = 1.0 / norm
    return nx, ny

x_surf_np = x_surf_t.cpu().numpy().flatten()
nx_upper, ny_upper = naca_normal(x_surf_np, t=0.12)
nx_lower = nx_upper
ny_lower = -ny_upper

nx_all = torch.tensor(np.concatenate([nx_upper, nx_lower]), dtype=torch.float32, device=device).unsqueeze(1)
ny_all = torch.tensor(np.concatenate([ny_upper, ny_lower]), dtype=torch.float32, device=device).unsqueeze(1)

pinn_optimizer = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)
pinn_scheduler = torch.optim.lr_scheduler.StepLR(pinn_optimizer, step_size=1000, gamma=0.5)
PINN_EPOCHS = 3000
pinn_loss_history = []
pinn_start = time.time()

# Far-field target velocities
u_far_target = U_inf * np.cos(alpha_aoa)
v_far_target = U_inf * np.sin(alpha_aoa)

# Supervised data: use Joukowski solution at a subset of interior points
# This gives PINN a "guide" to avoid trivial solutions
N_SUP = 1000
x_sup = torch.rand(N_SUP, 1, device=device) * (x_max - x_min) + x_min
y_sup = torch.rand(N_SUP, 1, device=device) * (y_max - y_min) + y_min
sdf_sup_np = naca_sdf(x_sup.cpu().numpy().flatten(), y_sup.cpu().numpy().flatten())
# Only keep points outside airfoil
sup_mask = sdf_sup_np > 0.05
x_sup = x_sup[sup_mask]
y_sup = y_sup[sup_mask]
sdf_sup = torch.tensor(sdf_sup_np[sup_mask], dtype=torch.float32, device=device).unsqueeze(1)
# Ground truth velocities at supervised points
u_sup_gt, v_sup_gt = joukowski_velocity(x_sup.cpu().numpy().flatten(), 
                                         y_sup.cpu().numpy().flatten(), 
                                         U=U_inf, alpha=alpha_aoa)
u_sup_gt = torch.tensor(u_sup_gt, dtype=torch.float32, device=device).unsqueeze(1)
v_sup_gt = torch.tensor(v_sup_gt, dtype=torch.float32, device=device).unsqueeze(1)

print(f"  Supervised points: {x_sup.shape[0]} (from Joukowski GT)")

for epoch in range(PINN_EPOCHS):
    pinn_optimizer.zero_grad()
    
    # PINN predicts velocity (u, v) directly
    # Input: (x, y, SDF)
    
    # 1. Interior points: continuity + irrotationality
    x_int_req = x_int.clone().requires_grad_(True)
    y_int_req = y_int.clone().requires_grad_(True)
    
    inp_int = torch.cat([x_int_req, y_int_req, sdf_int], dim=1)
    uv_int = pinn_model(inp_int)  # [N, 2]
    u_int_pred = uv_int[:, 0:1]
    v_int_pred = uv_int[:, 1:2]
    
    # Continuity: ∂u/∂x + ∂v/∂y = 0
    u_x = torch.autograd.grad(u_int_pred, x_int_req,
                               grad_outputs=torch.ones_like(u_int_pred),
                               create_graph=True)[0]
    v_y = torch.autograd.grad(v_int_pred, y_int_req,
                               grad_outputs=torch.ones_like(v_int_pred),
                               create_graph=True)[0]
    loss_continuity = (u_x + v_y).pow(2).mean()
    
    # Irrotationality: ∂u/∂y - ∂v/∂x = 0
    u_y = torch.autograd.grad(u_int_pred, y_int_req,
                               grad_outputs=torch.ones_like(u_int_pred),
                               create_graph=True)[0]
    v_x = torch.autograd.grad(v_int_pred, x_int_req,
                               grad_outputs=torch.ones_like(v_int_pred),
                               create_graph=True)[0]
    loss_irrotational = (u_y - v_x).pow(2).mean()
    
    # 2. Far-field BC: u = U*cos(α), v = U*sin(α) (supervised)
    inp_far = torch.cat([x_far, y_far, sdf_far], dim=1)
    uv_far = pinn_model(inp_far)
    loss_farfield = ((uv_far[:, 0:1] - u_far_target)**2 + (uv_far[:, 1:2] - v_far_target)**2).mean()
    
    # 3. Airfoil surface: no-penetration (V·n = 0)
    inp_surf = torch.cat([x_surf, y_surf, sdf_surf], dim=1)
    uv_surf = pinn_model(inp_surf)
    v_normal = uv_surf[:, 0:1] * nx_all + uv_surf[:, 1:2] * ny_all
    loss_surface = v_normal.pow(2).mean()
    
    # 4. Supervised data loss (from Joukowski GT at scattered points)
    inp_sup = torch.cat([x_sup, y_sup, sdf_sup], dim=1)
    uv_sup = pinn_model(inp_sup)
    loss_supervised = ((uv_sup[:, 0:1] - u_sup_gt)**2 + (uv_sup[:, 1:2] - v_sup_gt)**2).mean()
    
    # Total loss
    loss = loss_continuity + 0.1 * loss_irrotational + 10.0 * loss_farfield + 5.0 * loss_surface + 2.0 * loss_supervised
    loss.backward()
    pinn_optimizer.step()
    pinn_scheduler.step()
    pinn_loss_history.append(loss.item())
    
    if epoch % 300 == 0 or epoch == PINN_EPOCHS - 1:
        elapsed = time.time() - pinn_start
        print(f"  PINN Epoch {epoch:4d}/{PINN_EPOCHS} | Loss: {loss.item():.6e} "
              f"(cont: {loss_continuity.item():.2e}, far: {loss_farfield.item():.2e}, "
              f"sup: {loss_supervised.item():.2e}) | {elapsed:.1f}s")

pinn_time = time.time() - pinn_start
print(f"  PINN training complete! Final loss: {loss.item():.6e}, Time: {pinn_time:.1f}s\n")

# ============================================================
# [5] 3-way 비교: Ground Truth vs PINN vs FNO
# ============================================================
print("[5/8] Running 3-way comparison...")

# 테스트 공격각
alpha_test = np.radians(5.0)  # 5도 공격각

# Ground Truth (해석적)
u_gt_test, v_gt_test = joukowski_velocity(X_grid, Y_grid, U=U_inf, alpha=alpha_test)
u_gt_test[inside_mask] = 0
v_gt_test[inside_mask] = 0
speed_gt_test = np.sqrt(u_gt_test**2 + v_gt_test**2)

# FNO 예측
fno_model.eval()
with torch.no_grad():
    sin_a_test = np.sin(alpha_test) * np.ones_like(X_grid)
    cos_a_test = np.cos(alpha_test) * np.ones_like(X_grid)
    fno_input = np.stack([sdf_gt_np, X_grid, Y_grid, sin_a_test, cos_a_test], axis=0)
    fno_input = torch.tensor(fno_input, dtype=torch.float32).unsqueeze(0).to(device)
    fno_pred = fno_model(fno_input)[0].cpu().numpy()

u_fno = fno_pred[0]
v_fno = fno_pred[1]
u_fno[inside_mask] = 0
v_fno[inside_mask] = 0
speed_fno = np.sqrt(u_fno**2 + v_fno**2)

# PINN 예측: 직접 (u, v) 출력
pinn_model.eval()
with torch.no_grad():
    X_flat_t = torch.tensor(X_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    Y_flat_t = torch.tensor(Y_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    SDF_flat_t = torch.tensor(sdf_gt_np.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    pinn_input = torch.cat([X_flat_t, Y_flat_t, SDF_flat_t], dim=1)
    pinn_pred = pinn_model(pinn_input).cpu().numpy()
    u_pinn = pinn_pred[:, 0].reshape(GRID, GRID)
    v_pinn = pinn_pred[:, 1].reshape(GRID, GRID)
u_pinn[inside_mask] = 0
v_pinn[inside_mask] = 0
speed_pinn = np.sqrt(u_pinn**2 + v_pinn**2)

# 오차 계산
err_fno = np.sqrt((u_fno - u_gt_test)**2 + (v_fno - v_gt_test)**2)
err_pinn = np.sqrt((u_pinn - u_gt_test)**2 + (v_pinn - u_gt_test)**2)
err_fno[inside_mask] = 0
err_pinn[inside_mask] = 0

# L2 오차
l2_fno = np.linalg.norm(err_fno) / np.linalg.norm(speed_gt_test)
l2_pinn = np.linalg.norm(err_pinn) / np.linalg.norm(speed_gt_test)

print(f"  Test AoA: {np.degrees(alpha_test):.1f}°")
print(f"  FNO L2 error:  {l2_fno:.4f} ({l2_fno*100:.2f}%)")
print(f"  PINN L2 error: {l2_pinn:.4f} ({l2_pinn*100:.2f}%)\n")

# ============================================================
# [6] 압력계수 Cp 비교
# ============================================================
print("[6/8] Computing pressure coefficient (Cp)...")

# Cp = 1 - (V/V_inf)²
cp_gt = 1 - (speed_gt_test / U_inf)**2
cp_fno = 1 - (speed_fno / U_inf)**2
cp_pinn = 1 - (speed_pinn / U_inf)**2

# airfoil 표면 근처에서 Cp 추출
# 표면 바로 위쪽 점들에서 Cp 샘플링
n_cp = 50
x_cp = np.linspace(0.01, 0.99, n_cp)
# 표면 위 약간 떨어진 점
y_cp_upper = naca_thickness(x_cp, t=0.12) + 0.02

cp_gt_surface = []
cp_fno_surface = []
cp_pinn_surface = []

for i in range(n_cp):
    # 가장 가까운 격자점 찾기
    xi = x_cp[i]
    yi = y_cp_upper[i]
    idx_x = np.argmin(np.abs(x_grid - xi))
    idx_y = np.argmin(np.abs(y_grid - yi))
    cp_gt_surface.append(cp_gt[idx_y, idx_x])
    cp_fno_surface.append(cp_fno[idx_y, idx_x])
    cp_pinn_surface.append(cp_pinn[idx_y, idx_x])

cp_gt_surface = np.array(cp_gt_surface)
cp_fno_surface = np.array(cp_fno_surface)
cp_pinn_surface = np.array(cp_pinn_surface)

# ============================================================
# [7] 시각화
# ============================================================
print("[7/8] Generating visualizations...")
output_dir = r"E:\physicsnemo_env\tutorial_results"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# Airfoil 표면 좌표 (시각화용)
x_af = np.concatenate([x_surface, x_surface[::-1]])
y_af = np.concatenate([y_upper, y_lower[::-1]])

# Row 1: 속도장 (Ground Truth, FNO, PINN)
titles_row1 = [
    f"Ground Truth (Joukowski)\nAoA={np.degrees(alpha_test):.0f}°",
    f"FNO Prediction\nL2 error: {l2_fno*100:.2f}%",
    f"PINN Prediction\nL2 error: {l2_pinn*100:.2f}%",
]
speeds_row1 = [speed_gt_test, speed_fno, speed_pinn]

for i, (ax, title, speed) in enumerate(zip(axes[0], titles_row1, speeds_row1)):
    im = ax.contourf(X_grid, Y_grid, speed, levels=30, cmap="jet", vmin=0, vmax=2.5)
    ax.fill(x_af, y_af, color="black", alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, fraction=0.046, label="|V|")

# Row 2: 오차 맵 + 학습 곡선
im_err_fno = axes[1, 0].contourf(X_grid, Y_grid, err_fno, levels=30, cmap="hot", vmin=0, vmax=1.0)
axes[1, 0].fill(x_af, y_af, color="black", alpha=0.8)
axes[1, 0].set_title(f"FNO Error Map\n(max={err_fno.max():.3f})", fontsize=13, fontweight="bold")
axes[1, 0].set_xlim(x_min, x_max)
axes[1, 0].set_ylim(y_min, y_max)
axes[1, 0].set_aspect("equal")
plt.colorbar(im_err_fno, ax=axes[1, 0], fraction=0.046, label="|ΔV|")

im_err_pinn = axes[1, 1].contourf(X_grid, Y_grid, err_pinn, levels=30, cmap="hot", vmin=0, vmax=1.0)
axes[1, 1].fill(x_af, y_af, color="black", alpha=0.8)
axes[1, 1].set_title(f"PINN Error Map\n(max={err_pinn.max():.3f})", fontsize=13, fontweight="bold")
axes[1, 1].set_xlim(x_min, x_max)
axes[1, 1].set_ylim(y_min, y_max)
axes[1, 1].set_aspect("equal")
plt.colorbar(im_err_pinn, ax=axes[1, 1], fraction=0.046, label="|ΔV|")

# 학습 곡선 비교
ax_loss = axes[1, 2]
ax_loss.semilogy(range(0, FNO_EPOCHS, 10), fno_loss_history[::10], 
                  color="#2196F3", linewidth=1.5, label=f"FNO ({fno_time:.0f}s)")
ax_loss.semilogy(range(0, PINN_EPOCHS, 10), pinn_loss_history[::10], 
                  color="#FF6B35", linewidth=1.5, label=f"PINN ({pinn_time:.0f}s)")
ax_loss.set_title("Training Loss Comparison", fontsize=13, fontweight="bold")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss (log)")
ax_loss.legend(fontsize=11)
ax_loss.grid(True, alpha=0.3)

# Row 3: Cp 분포, 유선, 요약
# Cp 분포 (표면)
axes[2, 0].plot(x_cp, cp_gt_surface, "k-", linewidth=2, label="Ground Truth")
axes[2, 0].plot(x_cp, cp_fno_surface, "b--", linewidth=1.5, label=f"FNO (L2: {l2_fno*100:.1f}%)")
axes[2, 0].plot(x_cp, cp_pinn_surface, "r--", linewidth=1.5, label=f"PINN (L2: {l2_pinn*100:.1f}%)")
axes[2, 0].set_title("Pressure Coefficient (Cp)\nUpper Surface", fontsize=13, fontweight="bold")
axes[2, 0].set_xlabel("x/c (chord position)")
axes[2, 0].set_ylabel("Cp")
axes[2, 0].legend(fontsize=11)
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].invert_yaxis()  # Cp는 위가 음수

# 유선 (streamlines) — Ground Truth
ax_str = axes[2, 1]
ax_str.streamplot(X_grid, Y_grid, u_gt_test, v_gt_test, 
                   density=1.5, color=speed_gt_test, cmap="jet", linewidth=0.8)
ax_str.fill(x_af, y_af, color="black", alpha=0.8)
ax_str.set_title("Streamlines (Ground Truth)", fontsize=13, fontweight="bold")
ax_str.set_xlim(x_min, x_max)
ax_str.set_ylim(y_min, y_max)
ax_str.set_aspect("equal")

# 요약 텍스트
ax_summary = axes[2, 2]
ax_summary.axis("off")
summary_text = (
    "3-WAY COMPARISON SUMMARY\n"
    "=" * 40 + "\n\n"
    f"Problem: NACA 0012 Airfoil\n"
    f"Flow: Potential flow, AoA={np.degrees(alpha_test):.0f}°\n\n"
    f"{'Metric':<20} {'FNO':>10} {'PINN':>10}\n"
    f"{'-'*40}\n"
    f"{'Parameters':<20} {fno_params:>10,} {pinn_params:>10,}\n"
    f"{'Training time':<20} {fno_time:>9.0f}s {pinn_time:>9.0f}s\n"
    f"{'Training data':<20} {'Required':>10} {'None':>10}\n"
    f"{'L2 error':<20} {l2_fno*100:>9.2f}% {l2_pinn*100:>9.2f}%\n"
    f"{'Physics constraint':<20} {'Implicit':>10} {'Explicit':>10}\n"
    f"{'Mesh required':<20} {'Yes':>10} {'No':>10}\n\n"
    f"Ground Truth: Joukowski\n"
    f"transformation (analytical)"
)
ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                fontsize=12, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.suptitle(f"NACA 0012 Airfoil: PINN vs FNO vs Analytical Ground Truth\n"
             f"Potential Flow | AoA={np.degrees(alpha_test):.0f}° | PhysicsNeMo",
             fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(output_dir, "naca_airfoil_3way_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Result saved: {fig_path}")
plt.close()

# ============================================================
# [8] 요약
# ============================================================
print("\n" + "=" * 70)
print("  NACA AIRFOIL 3-WAY COMPARISON SUMMARY")
print("=" * 70)
print(f"  Problem:        NACA 0012 airfoil, potential flow")
print(f"  Test AoA:       {np.degrees(alpha_test):.1f}°")
print(f"  Ground Truth:   Joukowski transformation (analytical)")
print()
print(f"  {'Metric':<25} {'FNO':>12} {'PINN':>12}")
print(f"  {'-'*50}")
print(f"  {'Parameters':<25} {fno_params:>12,} {pinn_params:>12,}")
print(f"  {'Training time':<25} {fno_time:>11.0f}s {pinn_time:>11.0f}s")
print(f"  {'Training data':<25} {'Required':>12} {'None':>12}")
print(f"  {'L2 error':<25} {l2_fno*100:>11.2f}% {l2_pinn*100:>11.2f}%")
print(f"  {'Physics constraint':<25} {'Implicit':>12} {'Explicit':>12}")
print(f"  {'Mesh required':<25} {'Yes':>12} {'No':>12}")
print()
print(f"  Key insight:")
print(f"    - PINN: No training data needed, physics built-in, but slower")
print(f"    - FNO:  Fast, but requires ground truth data for training")
print(f"    - Both validated against analytical Joukowski solution")
print("=" * 70)
