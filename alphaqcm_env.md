# AlphaQCM 环境配置说明

本文档记录 **alphaqcm** 这一 conda 环境的配置逻辑与在**非本机**上的复现方式。环境最初并非在当前机器上构建，因此以文档形式固定其用途、依赖关系与迁移要点。

---

## 1. 环境用途

| 项目 | 说明 |
|------|------|
| **环境名** | 独立使用时为 `alphaqcm`；**推荐与 alphagen 共用** `alphagen_env`（见第 6 节） |
| **用途** | 运行 AlphaQCM 相关脚本：QCM 层级（FQF/IQN/QR-DQN + QCM）与 DRL 基线 |
| **对应代码** | `fqf_iqn_qrdqn`、`qcm_config`、`train_qcm.py`、`train_qcm_csi300.py`、`train_qcm_csi500.py`、`train_drl_csi300.py` |
| **入口脚本** | `runqcm.sh`（需先激活 conda 环境，默认 `alphagen_env`） |

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
- 使用 `runqcm.sh` 时脚本会检查 conda 环境（默认与 alphagen 共用 `alphagen_env`），否则会提示先激活环境。

---

## 6. 在现有 alphagen 环境上升级（推荐）

QCM 基于 alphagen，训练基模与 alphagen 一致，**可与原有 alphagen 共用同一 conda 环境**，无需单独建 `alphaqcm` 环境，从而避免两套环境带来的配置分裂与冲突。

### 6.1 为何可以共用

- **基模一致**：QCM 的 `train_qcm*` 与 alphagen 的 `scripts.rl` 等共用 `alphagen`、`alphagen_qlib`、同一套数据与 calculator。
- **依赖重叠**：两者都需要 `torch`、`gym`、`pyqlib`、`numpy`、`pandas`、`tensorboard` 等；QCM 额外需要的主要是配置解析（`pyyaml`）和可选 `gymnasium`，其余多为 alphagen 已有或可兼容版本。

### 6.2 可能冲突点与建议

| 项目 | alphagen（当前） | alphaqcm_env.yml（独立环境） | 在 alphagen 上升级时的建议 |
|------|------------------|------------------------------|-----------------------------|
| **torch** | 2.0.1（requirements.txt） | 1.13.1+cu116 | **保持现有 torch 不降级**。QCM 代码使用常规 PyTorch API，与 2.x 兼容；若遇兼容问题再考虑单独环境或按 5.2 换 CUDA 版。 |
| **numpy / pandas** | 1.21.1 / 1.2.4 | 1.23.5 / 1.5.3 | **可选升级**。若现有 alphagen 脚本无版本敏感逻辑，可逐步升级；若有报错再回退或仅装 QCM 增量包（不升级这两项）。 |
| **pyqlib** | 未锁版本 | 0.9.3 | 建议在升级时加上 `pyqlib==0.9.3`，与 QCM/qlib 数据路径行为一致。 |
| **stable_baselines3 / sb3_contrib** | 有（scripts/rl.py 用） | 无 | **保留**，QCM 不依赖 SB3，不影响 runqcm。 |

### 6.3 推荐操作：仅装 QCM 增量依赖

在**已激活的 alphagen 环境**下执行，不覆盖已有 torch/numpy/pandas 等，只补充 QCM 所需且当前可能缺失的包：

```bash
conda activate alphagen_env
pip install -r requirements_qcm_addon.txt
```

`requirements_qcm_addon.txt` 仅包含相对 alphagen 的**最小增量**（如 `pyyaml`、`gymnasium`、可选 `pyqlib==0.9.3` 等），不包含 torch/numpy/pandas，以降低冲突概率。若某次安装提示与现有包冲突，可按提示解决或暂时跳过该行。

### 6.4 若仍希望单独使用 alphaqcm 环境

- 保留 `conda env create -f alphaqcm_env.yml` 与 `conda activate alphaqcm` 的用法即可。
- 此时可将 `runqcm.sh` 中的 `TARGET_ENV` 改为 `alphaqcm`，与 5.3 的检查一致。

---

## 7. 与 runqcm.sh 的关系

- **runqcm.sh** 默认使用与 alphagen 相同的环境（`alphagen_env`），不负责创建环境；若采用「在 alphagen 上升级」方案，只需激活 `alphagen_env` 并可选安装 `requirements_qcm_addon.txt`。
- **职责边界**：`runqcm.sh` 与 `run.sh` 可独立运行，互不依赖。`run.sh` 用于 AlphaGen 主训练（`scripts/rl.py`），`runqcm.sh` 用于 AlphaQCM 入口（`train_qcm*.py` 与 `train_drl_csi300.py`）。
- **同脚本分层**：`runqcm.sh` 内部将 QCM 分支与 baseline 分支分层处理：QCM 使用 `qcm_config/*.yaml`，baseline（`drl_csi300`）使用 `config/*.yaml`，避免参数和配置串用。
- 环境配置的**权威来源**：独立环境用 `alphaqcm_env.yml`；在 alphagen 上升级时以现有 `requirements.txt` + `requirements_qcm_addon.txt` 为准。
- 本文档记录**配置逻辑**与**跨机/升级**方式，便于在非本机或与 alphagen 共用环境时正确复现与避免冲突。
