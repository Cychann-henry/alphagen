# alphagen 与 alphagen-qcm 差异比对

本文档对比 **alphagen**（主仓库基础库）与 **alphagen-qcm**（QCM 作业单独拷贝）的差异，便于统一以 alphagen 为基准时识别冲突与可选合并项。当前训练脚本（`train_qcm*.py`、`train_drl_csi300.py`）已统一使用 **alphagen** + **MseAlphaPool**，不依赖 alphagen-qcm 目录。

---

## 1. 目录结构差异

| 仅 alphagen 存在 | 仅 alphagen-qcm 存在 |
|------------------|----------------------|
| `data/parser.py` | `models/model.py` |
| `data/exception.py` | |
| `data/pool_update.py` | |
| `utils/maybe.py` | |
| `utils/misc.py` | |
| `utils/logging.py` | |
| `models/linear_alpha_pool.py` | |

- **alphagen**：抽象基类在 `alpha_pool.py`，具体实现（`LinearAlphaPool`、`MseAlphaPool` 等）在 `linear_alpha_pool.py`；另有 parser、exception、pool_update、maybe、misc、logging 等支撑。
- **alphagen-qcm**：在 `alpha_pool.py` 内同时定义 `AlphaPoolBase` 与具体类 `AlphaPool`（无 `linear_alpha_pool.py`）；多一个 `models/model.py`；无 parser/exception/pool_update/maybe/misc/logging。

---

## 2. 共同文件内容差异（需检阅）

### 2.1 `config.py`

| 项目 | alphagen | alphagen-qcm |
|------|----------|--------------|
| `MAX_EXPR_LENGTH` | 15 | **20** |
| `OPERATORS` | `List[Type[Operator]]` 类型注解 | 无类型注解 |
| `DELTA_TIMES` | [1, 5, 10, 20, 40] | **[10, 20, 30, 40, 50]** |

**冲突与建议**：  
- 表达式最大长度和 Delta 时间集合不同，会直接影响动作空间与可生成表达式。  
- 若希望 QCM 与主库行为一致，保持 alphagen 配置即可。  
- 若希望采用 QCM 的更长表达式或不同时间窗口，可在 alphagen 的 `config.py` 中改为与 alphagen-qcm 一致（或通过配置项区分场景）。

---

### 2.2 `models/alpha_pool.py`

| 项目 | alphagen | alphagen-qcm |
|------|----------|--------------|
| 内容 | 仅 **AlphaPoolBase** 抽象类（`state`、`to_json_dict`、`try_new_expr`、`test_ensemble`） | **AlphaPoolBase** + 具体类 **AlphaPool**（完整实现：`try_new_expr`、`_optimize`、`_calc_ics`、`force_load_exprs` 等） |
| 依赖 | 仅 `AlphaCalculator`、`Expression` | 另依赖 `alphagen_qlib.stock_data.StockData`、`batch_pearsonr`、`batch_spearmanr`、`masked_mean_std` |
| 基类 API | `state`、`to_json_dict` | `state`、**to_dict**（无 `to_json_dict`） |

**冲突与建议**：  
- 主仓库训练已统一使用 **alphagen** 的 `MseAlphaPool`（来自 `linear_alpha_pool.py`），与 alphagen-qcm 的 `AlphaPool` 是两套实现。  
- **不建议**用 alphagen-qcm 的 `alpha_pool.py` 覆盖 alphagen：会丢失 `LinearAlphaPool`/`MseAlphaPool` 体系，且与现有 `scripts/rl`、`gp` 等不兼容。  
- 若将来希望保留 QCM 那套池子逻辑，可在 alphagen 中新增一个模块（如 `alpha_pool_qcm.py`）把 alphagen-qcm 的 `AlphaPool` 迁入，并让 QCM 脚本显式从该模块导入，而不是替换现有 `alpha_pool`/`linear_alpha_pool`。

---

### 2.3 `data/calculator.py`

| 项目 | alphagen | alphagen-qcm |
|------|----------|--------------|
| 内容 | **AlphaCalculator** 抽象 + **TensorAlphaCalculator** 完整实现（含 `evaluate_alpha`、`make_ensemble_alpha`、`_calc_IC`、`_calc_rIC`、`calc_pool_all_ret_with_ir` 等） | 仅 **AlphaCalculator** 抽象接口（`calc_single_IC_ret`、`calc_mutual_IC`、`calc_pool_IC_ret`、`calc_pool_rIC_ret`） |
| 签名 | `calc_pool_IC_ret(exprs: Sequence[Expression], weights: Sequence[float])` | `calc_pool_IC_ret(exprs: List[Expression], weights: List[float])` |

**冲突与建议**：  
- 主仓库的 **alphagen_qlib** 的 `QLibStockDataCalculator` 继承自 alphagen 的 `TensorAlphaCalculator`，依赖其完整实现。  
- alphagen-qcm 的 calculator 只是接口子集，**不要**用 alphagen-qcm 的 `calculator.py` 覆盖 alphagen，否则会破坏现有计算链。

---

### 2.4 `rl/env/core.py`

| 项目 | alphagen | alphagen-qcm |
|------|----------|--------------|
| 导入 | `from alphagen.models.alpha_pool import AlphaPoolBase`<br>`from alphagen.models.linear_alpha_pool import LinearAlphaPool` | `from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool` |
| `__init__` | 有 `self.reset()` 调用 | **无** `self.reset()` |
| `step` 返回值 | `..., False, self._valid_action_types()`（第 4 个为 `truncated`） | `..., truncated=False  # Fk gymnasium, ...` |
| Tree 校验 | `validate_featured_expr()` | **validate_feature()** |

**冲突与建议**：  
- 保持使用 alphagen 的 core：依赖 `LinearAlphaPool`、与现有 wrapper 一致、带 `reset()` 初始化。  
- 若把 alphagen-qcm 的 core 覆盖进来，会因 `validate_feature` 与下面 data/tree 的命名不一致而需要同步改 tree；且会失去对 `LinearAlphaPool` 的引用。

---

### 2.5 `rl/env/wrapper.py`

| 项目 | alphagen | alphagen-qcm |
|------|----------|--------------|
| 导入 | 仅 `AlphaPoolBase` | `AlphaPoolBase, AlphaPool` |
| **subexprs** | 支持：`__init__(env, subexprs=None)`，动作空间含子表达式，`action_to_token` 含 `ExpressionToken(subexprs[i])` | **不支持**：无 subexprs，`__init__(env)` 仅接受 env |
| 动作空间 | `SIZE_ACTION + len(subexprs)`，含子表达式槽位 | `SIZE_ACTION = SIZE_ALL - SIZE_NULL`，无子表达式 |
| 动作→Token | 内置 `action_to_token`（0-based，先 OP、再 FEATURE、CONSTANT、DELTA_TIME、subexprs、SEP） | 独立函数 **action2token**（内部 `action_raw + 1` 再按 OFFSET_* 映射，无 subexprs） |
| `AlphaEnv` 工厂 | `AlphaEnv(pool, subexprs=None, **kwargs)` | `AlphaEnv(pool, **kwargs)` |

**冲突与建议**：  
- **动作空间与映射不一致**：同一动作下标在两套 wrapper 中可能对应不同 Token（有无 subexprs、0-based 与 +1 偏移）。fqf_iqn_qrdqn 与当前 train 脚本使用的是 **alphagen** 的 env（通过 `alphagen.rl.env.wrapper`），必须保持 alphagen 的 wrapper 不变。  
- **不要**用 alphagen-qcm 的 wrapper 覆盖，否则 SB3/scripts.rl 等依赖 subexprs 和现有动作布局的代码会出错。

---

### 2.6 `data/tree.py`

| 项目 | alphagen | alphagen-qcm |
|------|----------|--------------|
| 导入 | 含 `InvalidExpressionException`（来自 `data.exception`） | 无 exception 导入；文件末尾定义 **InvalidExpressionException** 于本文件 |
| `add_token` | 支持 **ExpressionToken**（子表达式入栈） | 仅支持 FeatureToken，**无 ExpressionToken** |
| `validate` | `(FeatureToken, ExpressionToken)` → **validate_featured_expr()** | `FeatureToken` → **validate_feature()** |
| 方法名 | **validate_featured_expr()** | **validate_feature()** |
| 异常类 | 在 `data/exception.py` | 在 tree.py 内定义 |

**冲突与建议**：  
- core 的 `_valid_action_types()` 在 alphagen 中调 `validate_featured_expr()`，在 alphagen-qcm 中调 `validate_feature()`；若混用会导致 AttributeError。  
- 统一以 alphagen 为准：保留 `validate_featured_expr`、ExpressionToken 与 `data/exception.py`，**不要**用 alphagen-qcm 的 tree 覆盖。

---

### 2.7 `data/expression.py`

| 项目 | alphagen | alphagen-qcm |
|------|----------|--------------|
| 导入 | 使用 `alphagen.utils.maybe` 的 Maybe/some/none | 无 maybe，直接使用内置类型 |
| `__add__` / `__sub__` 等 | 可能通过 Maybe 或统一逻辑处理 Expression/float | 显式 `isinstance(other, Expression)` 分支，非 Expression 则包成 `Constant(other)` |

**冲突与建议**：  
- 若 alphagen 中大量使用 Maybe，用 alphagen-qcm 的 expression 覆盖会报错（缺 maybe）。  
- 建议保持 alphagen 的 expression；若需要 QCM 的某种运算行为，可单独摘取逻辑合并到 alphagen，而不是整文件替换。

---

### 2.8 其他共同文件（差异较小或仅风格）

- **utils/random.py**：一致。  
- **utils/__init__.py**：需对比是否导出相同符号（如 `reseed_everything`）。  
- **utils/correlation.py**、**utils/pytorch_utils.py**：未逐行比对；若 QCM 有修复或优化，可单独 diff 后选择性合并。  
- **data/tokens.py**：未逐行比对；若 QCM 有新增 Token 类型或修改，需确认与现有 parser/tree 兼容。  
- **trade/base.py**、**trade/strategy.py**：未逐行比对；通常为底层交易类型，变更可能影响回测/实盘，建议按需 diff。  
- **rl/policy.py**：开头结构相似（PositionalEncoding、TransformerSharedNet）；若 QCM 有网络结构或接口改动，需与当前 RL 入口（如 scripts.rl）一起检阅。

---

## 3. 冲突汇总与推荐策略

| 冲突点 | 风险 | 建议 |
|--------|------|------|
| 使用 alphagen-qcm 的 `alpha_pool.py` 替换 alphagen | 高：丢失 LinearAlphaPool/MseAlphaPool，破坏现有脚本 | 不替换；QCM 训练继续用 alphagen 的 **MseAlphaPool**（已改好）。 |
| 使用 alphagen-qcm 的 `calculator.py` 替换 alphagen | 高：丢失 TensorAlphaCalculator，QLib 等依赖断裂 | 不替换。 |
| 使用 alphagen-qcm 的 `core.py` / `wrapper.py` 替换 alphagen | 高：动作空间与 subexprs 不一致，validate_feature 与 tree 不匹配 | 不替换；保持 alphagen 的 env。 |
| 使用 alphagen-qcm 的 `tree.py` 替换 alphagen | 高：失去 ExpressionToken、validate_featured_expr，core 报错 | 不替换。 |
| 使用 alphagen-qcm 的 `expression.py` 替换 alphagen | 中：可能缺 maybe，语义可能有细微差异 | 不整文件替换；有需要再局部合并。 |
| **仅**在 alphagen 的 `config.py` 中采纳 QCM 的 MAX_EXPR_LENGTH / DELTA_TIMES | 低 | 可选：若希望与 QCM 实验设置一致，可改 config，并跑一遍回归确认 env 与训练正常。 |

**推荐**：  
- 以 **alphagen** 为唯一基础库，**不**用 alphagen-qcm 做整体覆盖。  
- 当前 QCM 流程已统一：`train_qcm*.py`、`train_drl_csi300.py` 使用 **alphagen** + **MseAlphaPool** + **AlphaEnv**（alphagen 的 wrapper/core）。  
- alphagen-qcm 仅作**参考**：若需要其中某段逻辑（例如 AlphaPool 的 `force_load_exprs`、或 config 数值），再单独摘取并合并到 alphagen，并做测试。

---

## 4. 若希望将 alphagen-qcm 的 AlphaPool 纳入主库（可选）

若你希望保留 QCM 版「单文件 AlphaPool」的实现（例如与现有 MseAlphaPool 二选一使用），建议：

1. 在 alphagen 中新增 **`models/alpha_pool_qcm.py`**（或类似命名），把 alphagen-qcm 的 `AlphaPool` 类及其依赖（如对 calculator 的 `calc_pool_IC_ret` 等调用）迁入。  
2. 确保该模块**只依赖** alphagen 已有的 calculator 接口（与当前 `TensorAlphaCalculator` 一致），去掉对 `alphagen_qlib` 的直连（若可抽成接口）。  
3. 需要用到该池子的脚本显式写：`from alphagen.models.alpha_pool_qcm import AlphaPool`，而不改现有 `from alphagen.models.linear_alpha_pool import MseAlphaPool` 的用法。  
4. 在测试/回归中同时跑一轮现有 MseAlphaPool 流程与 AlphaPool 流程，确认无回归。

这样既不破坏现有 alphagen 结构，又能在主库内保留 QCM 的池子实现供你检阅与复用。

---

## 5. 当前对接方案：QCM 独立文件（已实现）

在保留 alphagen 基础逻辑**全部不变**的前提下，QCM 新增/冲突部分已拆成带 **qcm** 标志的独立文件，并与训练脚本对接：

| 文件 | 说明 |
|------|------|
| **alphagen/config_qcm.py** | QCM 配置：MAX_EXPR_LENGTH=20，DELTA_TIMES=[10,20,30,40,50]，OPERATORS/CONSTANTS 同构 |
| **alphagen/models/alpha_pool_qcm.py** | QCM 因子池实现 **AlphaPoolQcm**，仅依赖 AlphaCalculator 接口，实现 `to_json_dict`/`state` 与 alphagen 基类一致 |
| **alphagen/rl/env/core_qcm.py** | **AlphaEnvCoreQcm**：仅从 config_qcm 读 MAX_EXPR_LENGTH，其余与 core 一致 |
| **alphagen/rl/env/wrapper_qcm.py** | **AlphaEnvWrapperQcm** + **AlphaEnvQcm**：使用 config_qcm 与 core_qcm，无 subexprs |
| **alphagen/qcm.py** | 统一入口：`from alphagen.qcm import AlphaPoolQcm, AlphaEnvQcm` |

训练脚本（`train_qcm.py`、`train_qcm_csi300.py`、`train_qcm_csi500.py`）增加 **`--use-qcm-stack`**：

- **不加**：使用 alphagen 默认（MseAlphaPool + AlphaEnv），与现有行为一致。
- **加 `--use-qcm-stack`**：使用 AlphaPoolQcm + AlphaEnvQcm（config_qcm 的表达式长度与动作空间，无 subexprs）。

示例：

```bash
python train_qcm_csi300.py --model qrdqn --seed 0 --pool 20 --std-lam 1.0 --use-qcm-stack
```
