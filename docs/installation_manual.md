# NVIDIA PhysicsNeMo Installation Manual

> This document provides step-by-step instructions for installing NVIDIA PhysicsNeMo on Windows.  
> It is written in detail so anyone can follow along without AI tools.

---

## 1. Prerequisites

### 1.1 Hardware Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| **GPU** | NVIDIA GPU (Compute Capability 3.5+) | NVIDIA Quadro P4000 or higher |
| **VRAM** | 4GB+ | 8GB+ |
| **RAM** | 16GB+ | 32GB+ |
| **Disk** | 20GB+ free space | 50GB+ free space |

### 1.2 Software Requirements

| Item | Requirement |
|------|-------------|
| **OS** | Windows 10/11 (64-bit) |
| **Python** | 3.10 recommended |
| **NVIDIA Driver** | 452.39+ (CUDA 11.8 support) |

### 1.3 GPU Compute Capability Check

Refer to the table below to check if your GPU supports CUDA 11.8:

| GPU Series | Compute Capability | CUDA 11.8 Support |
|------------|-------------------|-------------------|
| GeForce GTX 600/700 (Kepler) | 3.0/3.5 | ❌ (only 3.5+ supported) |
| GeForce GTX 700/900 (Maxwell) | 5.0/5.2 | ✅ |
| GeForce GTX 1000 (Pascal) | 6.0/6.1 | ✅ |
| GeForce RTX 2000 (Turing) | 7.5 | ✅ |
| GeForce RTX 3000 (Ampere) | 8.6 | ✅ |
| GeForce RTX 4000 (Ada Lovelace) | 8.9 | ✅ |
| Quadro P series (Pascal) | 6.0/6.1 | ✅ |
| Quadro RTX series (Turing) | 7.5 | ✅ |

> 💡 **Tip**: Any NVIDIA GPU with Compute Capability 3.5+ supports CUDA 11.8.

---

## 2. NVIDIA Driver Update

### 2.1 Check Current Driver Version

1. Press **Windows Key + R** to open the Run dialog.
2. Type `cmd` and press **Enter** to open Command Prompt.
3. Enter the following command:
   ```
   nvidia-smi
   ```
4. Check **Driver Version** and **CUDA Version** in the output.

> ⚠️ **Note**: If `nvidia-smi` is not recognized, the NVIDIA driver is not installed or not added to PATH.

### 2.2 Download and Install Driver

1. Go to https://www.nvidia.com/Download/index.aspx
2. Select your GPU:
   - **GeForce/RTX users**: Product Type → GeForce
   - **Quadro users**: Product Type → Quadro
3. Select Product Series, Product, OS and click **Search**.
4. Click **Download** to get the installer.
5. Run the downloaded file to install.
   - Installation type: **Express** or **Custom (clean install)** recommended
6. **Reboot** the computer after installation.

### 2.3 Verify Driver Installation

After reboot, check again in Command Prompt:
```
nvidia-smi
```
- Verify Driver Version is **452.39 or higher**
- Verify CUDA Version shows **12.x**

---

## 3. Python Installation

### 3.1 Download Python 3.10

1. Go to https://www.python.org/downloads/
2. Find Python 3.10.x and download it.
   - Select Windows installer (64-bit)
3. Run the installer.
4. ⚠️ **Important**: On the first installation screen, check **"Add Python 3.10 to PATH"**.
5. Click **Install Now** to install.

### 3.2 Verify Python Installation

Open Command Prompt:
```
python --version
```
If `Python 3.10.x` is output, installation is successful.

---

## 4. Create Python Virtual Environment

Virtual environments create independent Python environments per project, preventing package conflicts.

### 4.1 Create Virtual Environment

Open Command Prompt and run:

```
cd %USERPROFILE%\Desktop
python -m venv physicsnemo_env
```

> 💡 `physicsnemo_env` is the virtual environment name; you can change it to any name.

### 4.2 Activate Virtual Environment

```
physicsnemo_env\Scripts\activate
```

When activated, `(physicsnemo_env)` appears before the prompt.

> ⚠️ **Note**: You must re-activate the virtual environment each time you open a new Command Prompt.

### 4.3 Upgrade pip

```
python -m pip install --upgrade pip
```

---

## 5. Install PyTorch (CUDA 11.8)

### 5.1 Install PyTorch

With the virtual environment activated, run:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ **Note**: `cu118` means CUDA 11.8. For other CUDA versions, use the appropriate index URL:
> - CUDA 12.1: `https://download.pytorch.org/whl/cu121`
> - CUDA 12.4: `https://download.pytorch.org/whl/cu124`

### 5.2 Verify PyTorch Installation

```
python -c "import torch; print(torch.__version__)"
```

If output is in `2.x.x+cu118` format, it's normal.

### 5.3 Verify CUDA Recognition

```
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

- `CUDA available: True` → Normal (GPU recognized)
- `CUDA available: False` → Problem (check driver in Step 2)

---

## 6. Install NVIDIA PhysicsNeMo

### 6.1 Install PhysicsNeMo

With the virtual environment activated:

```
pip install nvidia-physicsnemo
```

### 6.2 Verify Installation

```
python -c "import physicsnemo; print(physicsnemo.__version__)"
```

If a version number is output, installation is complete.

---

## 7. Full Installation Verification Script

Save the following as `check_physicsnemo.py` and run it to verify the full installation:

```python
import physicsnemo
print("PhysicsNeMo version:", physicsnemo.__version__)

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("WARNING: CUDA is not available. Please check your NVIDIA driver.")
```

Run:
```
physicsnemo_env\Scripts\python.exe check_physicsnemo.py
```

---

## 8. Troubleshooting Guide

### 8.1 "CUDA initialization: The NVIDIA driver on your system is too old"

**Cause**: NVIDIA driver is outdated.

**Solution**: Update the driver following Step 2.

### 8.2 "nvidia-smi is not recognized"

**Cause**: NVIDIA driver is not installed or not in PATH.

**Solution**:
1. Install the NVIDIA driver.
2. Add `C:\Program Files\NVIDIA Corporation\NVSMI` to system PATH.

### 8.3 "No module named 'physicsnemo'"

**Cause**: Virtual environment is not activated or PhysicsNeMo is not installed.

**Solution**:
1. Activate virtual environment: `physicsnemo_env\Scripts\activate`
2. Install PhysicsNeMo: `pip install nvidia-physicsnemo`

### 8.4 "python command is not recognized"

**Cause**: Python is not in PATH.

**Solution**:
1. Reinstall Python and check "Add Python to PATH".
2. Or manually add Python installation path to system PATH.

### 8.5 SSL certificate error during pip install

**Cause**: SSL certificate issue due to corporate network proxy/firewall.

**Solution**:
```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org nvidia-physicsnemo
```

---

## 9. Summary: Full Installation Commands

Open Command Prompt and run the following in order:

```batch
:: 1. Create virtual environment
cd %USERPROFILE%\Desktop
python -m venv physicsnemo_env

:: 2. Activate virtual environment
physicsnemo_env\Scripts\activate

:: 3. Upgrade pip
python -m pip install --upgrade pip

:: 4. Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: 5. Install PhysicsNeMo
pip install nvidia-physicsnemo

:: 6. Verify installation
python -c "import physicsnemo; print('PhysicsNeMo:', physicsnemo.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 10. Reference Links

| Item | URL |
|------|-----|
| NVIDIA Driver Download | https://www.nvidia.com/Download/index.aspx |
| Python Download | https://www.python.org/downloads/ |
| PyTorch Installation Guide | https://pytorch.org/get-started/locally/ |
| PhysicsNeMo Official Docs | https://docs.nvidia.com/deeplearning/physicsnemo/ |
| PhysicsNeMo GitHub | https://github.com/NVIDIA/physicsnemo |
| CUDA Compatibility Table | https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/ |

---

*This document was written as of June 30, 2026. For the latest information, please refer to each official website.*
