# AlphaQCM 环境配置说明

本文档记录 **alphaqcm** 这一 conda 环境的配置逻辑与在**非本机**上的复现方式。环境最初并非在当前机器上构建，因此以文档形式固定其用途、依赖关系与迁移要点。

---

## 1. 环境用途

| 项目 | 说明 |
|------|------|
| **环境名** | `alphaqcm` |
| **用途** | 运行 AlphaQCM 相关脚本：QCM 层级（FQF/IQN/QR-DQN + QCM）与 DRL 基线 |
| **对应代码** | `fqf_iqn_qrdqn`、`qcm_config`、`train_qcm.py`、`train_qcm_csi300.py`、`train_qcm_csi500.py`、`train_drl_csi300.py` |
| **入口脚本** | `runqcm.sh`（需先激活本环境） |

---

## 2. 配置逻辑概览

- **Python**：固定为 **3.8**（conda 中为 `python=3.8.18`），与 pyqlib、gym、torch 等版本兼容。
- **包来源**：系统级/运行时用 **conda** 安装；项目级 Python 库用 **pip** 在 conda 环境中安装（`dependencies` 下的 `pip:` 列表）。
- **channels 顺序**：`bioconda` → `r` → `conda-forge` → `defaults`，保证优先从 conda-forge 等渠道解析，减少与 defaults 的冲突。
- **无 prefix**：`alphaqcm_env.yml` 中**不写死** `prefix`，便于在不同机器、不同用户下直接 `conda env create -f alphaqcm_env.yml`，环境会落在当前 conda 的默认 env 目录。

---

## 3. 依赖分组与用途

### 3.1 Conda 部分（基础运行时）

- **Python 3.8** 及解释器相关。
- **ipython / ipykernel / jupyter_***：交互与调试。
- **系统库**：如 `libgcc-ng`、`libstdcxx-ng`、`openssl`、`libffi`、`bzip2`、`xz`、`tk` 等，多为 Linux 下编译或运行所需（若在 Windows 上从零用 conda 建环境，conda 会按平台解析，部分包可能不可用或名称不同）。
- **pip / setuptools / wheel**：用于在环境中通过 pip 安装下方 Python 包。

### 3.2 Pip 部分（按功能分组）

| 类别 | 代表包 | 说明 |
|------|--------|------|
| **深度学习 / 强化学习** | `torch==1.13.1+cu116`，`torchaudio`，`torchvision`，`gym==0.26.2`，`gymnasium==0.29.1` | 训练 QCM/DRL 模型；当前指定 CUDA 11.6。 |
| **量化 / 数据** | `pyqlib==0.9.3`，`pandas`，`numpy`，`tables`，`baostock` | 行情与因子数据、qlib 初始化。 |
| **Alpha / 优化** | `cvxpy`，`scipy`，`scikit-learn`，`lightgbm`，`statsmodels`，`clarabel`，`osqp`，`qdldl`，`ecos`，`scs` | 组合优化、回归、目标等。 |
| **实验与日志** | `mlflow`，`tensorboard`，`sacred`，`loguru` | 实验记录与可视化。 |
| **工程与工具** | `pyyaml`，`fire`，`tqdm`，`joblib`，`dill`，`cloudpickle`，`hyperopt` | 配置解析、命令行、序列化、调参。 |
| **其他** | `flask`，`redis`，`pymongo`，`sqlalchemy`，`alembic`，`docker` 等 | 若项目或脚本中有服务、缓存、数据库、容器化会用到。 |

---

## 4. 关键版本与跨机注意点

- **PyTorch**：`torch==1.13.1+cu116`（及对应 torchaudio、torchvision）表示 **CUDA 11.6**。在另一台机器上若 CUDA 版本不同（如 11.8、12.x），需要改为对应版本或使用 `cu118`/`cu121` 等索引，否则可能无法使用 GPU 或报错。
- **PyQlib**：`0.9.3`，与数据路径、`alphagen_qlib.stock_data.initialize_qlib` 的默认路径（如 `~/.qlib/qlib_data/cn_data_2024h1`）配合使用；新机器需自备 qlib 数据并视情况改路径。
- **Python 3.8**：不建议随意升到 3.9+，部分包（尤其是带二进制扩展的）可能尚未兼容。
- **原始导出环境**：来自 **Linux**（如 conda 包带 `linux-64`、`gnu` 等）。在 **Windows** 上若直接 `conda env create -f alphaqcm_env.yml` 可能部分 conda 包无法安装；常见做法是：
  - 仅用 conda 创建 Python 3.8 环境，再根据下面的「最小复现」用 pip 安装主要依赖；或
  - 在 WSL / 虚拟机中按原 yml 完整复现。

---

## 5. 在新机器上的使用方式

### 5.1 直接使用 yml（适合 Linux / 同架构）

```bash
conda env create -f alphaqcm_env.yml
conda activate alphaqcm
```

若希望指定安装路径，可先修改 yml 或导出时加上 prefix，再创建：

```bash
# 示例：创建时指定 prefix（按需修改路径）
conda env create -f alphaqcm_env.yml --prefix /path/to/envs/alphaqcm
# 激活
conda activate /path/to/envs/alphaqcm
```

### 5.2 新机器 CUDA 与当前 yml 不一致时

- 先创建空环境：`conda create -n alphaqcm python=3.8 -y && conda activate alphaqcm`
- 到 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/) 选择对应 CUDA 的安装命令，例如 CUDA 11.8：
  ```bash
  pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 torchvision==0.14.1+cu117
  ```
- 再安装其余 pip 依赖：可从 `alphaqcm_env.yml` 中拷贝 `pip:` 下列表，**去掉** torch/torchaudio/torchvision 三行，保存为 `alphaqcm_pip_requirements.txt`，然后：
  ```bash
  pip install -r alphaqcm_pip_requirements.txt
  ```

### 5.3 运行前检查

- 在项目根目录下执行（或保证 `PYTHONPATH` 含项目根）。
- 使用 `runqcm.sh` 时脚本会检查 `CONDA_DEFAULT_ENV == alphaqcm`，否则会提示先激活环境。

---

## 6. 与 runqcm.sh 的关系

- **runqcm.sh** 假定已存在并激活 **alphaqcm** 环境，不负责创建环境。
- 环境配置的**权威来源**是 `alphaqcm_env.yml`；本文档（`alphaqcm_env.md`）只记录其**配置逻辑**与**跨机复现**方式，便于在「环境不是在这台机器上配的」的情况下，在别处正确重建或调整环境。
