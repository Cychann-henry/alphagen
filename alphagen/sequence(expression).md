```mermaid
graph TD
    subgraph "1. RL Agent & Wrapper: 动作生成"
        direction LR
        A1["RL Agent 输出整数动作, e.g., 5"] --> A2(AlphaEnvWrapper);
        A2 -- "action_to_token(5)" --> A3["生成具体 Token 对象, e.g., FeatureToken('close')"];
    end

    subgraph "2. AlphaEnvCore: 驱动构建器"
        direction LR
        B1(AlphaEnvCore) -- "接收 Token" --> B2{"调用 self.builder.add_token(Token)"};
    end
    
    subgraph "3. ExpressionBuilder: 逆波兰式核心构建逻辑"
        direction TB
        C1["ExpressionBuilder 内部维护一个堆栈 Stack"]
        
        C2{"Token 类型是操作数吗?\n(Feature, Constant, DeltaTime)"}
        C3{"Token 类型是操作符吗?\n(Operator)"}
        
        C1 --> C2;
        C2 -- 是 --> C4["直接压入堆栈 Stack.push(Token)"];
        C2 -- 否 --> C3;
        
        C3 -- 是 --> C5["从堆栈弹出所需数量的参数"];
        C5 --> C6["用操作符将参数组合成一个新的子树 (Sub-Expression)"];
        C6 --> C7["将生成的新子树压回堆栈 Stack.push(Sub-Expression)"];
        
        C4 --> C8(("循环接收下一个 Token..."));
        C7 --> C8;
    end

    subgraph "4. 结束与产出"
        direction LR
        D1["接收到 SEP (结束) Token"] --> D2{"校验堆栈是否只剩一个根节点?"};
        D2 -- 是 --> D3["弹出该节点, 作为最终的 Expression 对象"];
        D2 -- 否 --> D4["构建失败"];
    end

    %% 连接各大模块
    A3 --> B1;
    B2 --> C1;
    C8 --> B2;
    B2 -- "当 Token 是 SEP 时" --> D1;

    %% 样式定义
    classDef rl fill:#ffe0b2,stroke:#e65100;
    classDef core fill:#e1f5fe,stroke:#0277bd;
    classDef builder fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef final fill:#fce4ec,stroke:#ad1457;

    class A1,A2,A3 rl;
    class B1,B2 core;
    class C1,C2,C3,C4,C5,C6,C7,C8 builder;
    class D1,D2,D3,D4 final;
```

### `Expression` 构建过程详解 (基于逆波兰式)

在 `alphagen` 中，一个因子表达式 (`Expression` 对象) 本质上是一棵树。这个构建过程巧妙地利用了**逆波兰式（Reverse Polish Notation, RPN）** 的思想，其核心是 `alphagen/data/tree.py` 中的 `ExpressionBuilder` 类。

#### 什么是 `Expression` 对象？
`Expression` 是一个抽象基类，代表一个可计算的因子公式。它可以是一个简单的叶子节点（如 `Feature('close')`），也可以是一个复杂的内部节点（如 `Add(Rank(close), Rank(open))`）。每个 `Expression` 对象都知道如何通过 `evaluate()` 方法计算自身的因子值。

---

#### 构建流程：一个具体的例子

假设 RL Agent 依次生成了代表 `close`, `10`, `Mean` 的动作序列。

1.  **起点: 空的 `ExpressionBuilder`**
    *   `ExpressionBuilder` 内部维护一个**堆栈 (Stack)**，此时为空。
    *   `Stack: []`

2.  **第一步: 接收 `FeatureToken('close')`**
    *   `AlphaEnvWrapper` 将 RL 的动作转换为 `FeatureToken('close')`。
    *   `AlphaEnvCore` 将此 `Token` 传递给 `ExpressionBuilder`。
    *   `ExpressionBuilder` 判断出这是一个**操作数** (Operand)。
    *   **操作**: 直接将其压入堆栈。
    *   `Stack: [Feature('close')]`

3.  **第二步: 接收 `ConstantToken(10)`**
    *   同上，`ConstantToken(10)` 是一个**操作数**。
    *   **操作**: 直接压入堆栈。
    *   `Stack: [Feature('close'), Constant(10)]`

4.  **第三步: 接收 `OperatorToken('Mean')`**
    *   `ExpressionBuilder` 判断出这是一个**操作符** (Operator)。`Mean` 是一个二元算子，需要两个参数。
    *   **操作 (规约/Reduce)**:
        1.  从堆栈中**弹出**两个元素：先弹出 `Constant(10)` (作为参数2)，再弹出 `Feature('close')` (作为参数1)。
        2.  用 `Mean` 操作符将这两个参数**组合**成一个新的 `Expression` 子树：`Mean(Feature('close'), Constant(10))`。
        3.  将这个新生成的**子树**压回堆栈。
    *   `Stack: [Mean(Feature('close'), Constant(10))]`

5.  **结束: 接收 `SEP_TOKEN`**
    *   `AlphaEnvCore` 接收到序列结束符。
    *   `ExpressionBuilder` 进行最终校验：
        1.  检查堆栈中是否**只剩下一个元素**。
        2.  检查这个元素是否是一个**合法、完整的表达式树**。
    *   校验通过后，`ExpressionBuilder` 将堆栈中这唯一的元素弹出，作为最终的构建结果返回。
    *   **产出**: `Mean(Feature('close'), Constant(10))` 这个 `Expression` 对象。

这个 `Expression` 对象随后会被送入 `AlphaPool` 进行评估。整个过程就像使用一个老式的堆栈计算器：先输入数字，再输入运算符