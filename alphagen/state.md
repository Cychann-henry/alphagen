```mermaid
stateDiagram-v2
    direction LR
    
    %% 定义状态
    state "初始状态\nStack: []" as Start
    state "压入特征\nStack: [Close]" as S1
    state "压入常数\nStack: [Close, 10]" as S2
    state "压入算子(规约)\nStack: [Mean(Close, 10)]" as S3
    state "提交(SEP)\nTree: Mean(Close, 10)" as Final

    %% 流程连线
    Start --> S1 : 动作: Feature(Close)
    note on link
        Builder.add_token()
        直接压栈
    end note

    S1 --> S2 : 动作: Constant(10)
    note on link
        Builder.add_token()
        直接压栈
    end note

    S2 --> S3 : 动作: Operator(Mean)
    note top of S3
        **核心逻辑触发**:
        Mean 需要2个参数:
        1. 弹出 10 (arg2)
        2. 弹出 Close (arg1)
        3. 组合: Mean(Close, 10)
        4. 结果压回栈顶
    end note

    S3 --> Final : 动作: SEP (结束符)
    note on link
        校验: len(stack) == 1
        校验: 根节点是因子而非参数
        导出最终对象
    end note
```