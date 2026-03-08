```mermaid
graph TD
    %% =======================
    %% 样式定义
    %% =======================
    classDef file_script fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef file_env fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef file_pool fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef reward_node fill:#fce4ec,stroke:#c2185b,stroke-width:3px,stroke-dasharray: 5 5;

    %% =======================
    %% 1. 主控脚本
    %% =======================
    %% 这里的 <br/> 是关键，它撑开了标题的高度，产生了适度的间距
    subgraph Script["文件: scripts/rl.py<br/> "]
        direction TB
        Agent["PPO Agent (MaskablePPO)"]:::file_script
        Callback["CustomCallback (每隔N步)"]:::file_script
        LLM["LLM Client (ChatSession)"]:::file_script
    end

    %% =======================
    %% 2. 环境交互
    %% =======================
    subgraph Env["文件: rl/env/wrapper.py<br/> "]
        direction TB
        WrapperAct["action(int) -> Token"]:::file_env
        Builder["ExpressionBuilder (堆栈构建)"]:::file_env
        CheckDone{"是否生成 SEP?"}:::file_env
    end

    %% =======================
    %% 3. 因子池与奖励
    %% =======================
    subgraph Pool["文件: linear_alpha_pool.py "]
        direction TB
        CalcIC["计算单因子 IC (Calculator)"]:::file_pool
        Optimize["optimize() 权重优化<br/>(目标: 最小化 MSE (Label) 或 最大化 ICIR)"]:::file_pool
        
        %% 重点：奖励函数定义
        CalcReward("计算奖励 Reward:<br/>(新组合IC - 旧组合IC) + 步长惩罚"):::reward_node
    end

    %% =======================
    %% 流程连接
    %% =======================
    
    %% Agent 逻辑
    Agent -- "输出动作 ID" --> WrapperAct
    WrapperAct -- "生成 Token 对象" --> Builder
    Builder --> CheckDone
    
    %% 循环分支
    CheckDone -- "否 (未完成)" --> StepPenalty("小额负奖励 (鼓励短公式)"):::reward_node
    StepPenalty --> Agent
    
    %% 完成分支
    CheckDone -- "是 (生成 Expression)" --> CalcIC
    
    %% 优化逻辑
    CalcIC --> Optimize
    Optimize -- "产生新的 Best IC" --> CalcReward
    
    %% 反馈
    CalcReward -- "最终 Reward" --> Agent
    
    %% LLM 旁路
    Agent -.-> Callback
    Callback -- "定期触发" --> LLM
    LLM -- "直接注入新因子" --> Optimize

    %% 隐藏不可见节点的样式，用于撑开布局
    classDef invisible fill:none,stroke:none,color:none;
```

### 核心逻辑说明

1.  **初始化 (Initialization)**:
    *   脚本首先初始化 Qlib 数据加载器 (`QLibStockDataCalculator`)，并将数据划分为训练集（2012-2021）和测试集（2022-2023）。
    *   构建 `LinearAlphaPool`（具体是 `MseAlphaPool`），如果指定了 `alphagpt_init`，则会从硬盘加载预设的因子。
    *   构建 Gym 环境 `AlphaEnv` 和 PPO 智能体。

2.  **训练循环 (Training Loop)**:
    *   `MaskablePPO` 控制 Agent 在环境中不断生成 Token，构建因子表达式。
    *   每当一个表达式构建完成，环境将其放入 `AlphaPool` 计算 IC（信息系数）并优化权重，返回 Reward。

3.  **回调系统 (Callback)**:
    *   这是 `rl.py` 的关键。通过 `CustomCallback`，在每次 PPO 收集完一批数据（Rollout End）后介入。
    *   它负责记录日志、在**样本外数据（2022-2023）**上验证当前因子池的表现，并保存检查点。

4.  **LLM 混合驱动 (LLM Interaction)**:
    *   如果开启了 `use_llm`，回调函数会检查步数（例如每 25,000 步）。
    *   满足条件时，会先**踢除**掉一部分由 RL 生成的效果较差的因子 (`drop_rl_n`)，腾出空间。
    *   然后调用 `ChatClient` 让大语言模型根据当前状态生成或修改因子，将结果直接注入 `AlphaPool`，实现 RL 与 LLM 的协同优化。