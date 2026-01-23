```mermaid
graph TD
    %% 定义节点样式
    classDef decision fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef terminator fill:#e0e0e0,stroke:#333,stroke-width:2px;
    classDef critical fill:#ffccbc,stroke:#d84315,stroke-width:4px;

    Start([输入: 新生成的 Expression]) --> CalcSingle[计算单因子 IC]
    
    CalcSingle --> CheckLow{IC < 阈值?}:::decision
    CheckLow -- 是 --> Reject([拒绝: Return 0]):::terminator
    CheckLow -- 否 --> CheckCorr[计算与池内因子的互相关性]:::process
    
    CheckCorr --> IsDuplicate{相关性 > 0.99?}:::decision
    IsDuplicate -- 是 --> Reject
    IsDuplicate -- 否 --> AddToPool[暂时加入 exprs 列表]:::process
    
    AddToPool --> Optimize[权重优化 optimize()]:::critical
    
    subgraph WeightOptimization [权重优化黑盒]
        direction TB
        OptStart[构建目标函数]
        Adam[运行优化算法]
        NewWeights[产出新权重向量]
        OptStart --> Adam --> NewWeights
    end
    
    Optimize --> WeightOptimization
    WeightOptimization --> CheckCapacity{池子满了吗?}:::decision
    
    CheckCapacity -- 否 --> CalcIncrement
    CheckCapacity -- 是 --> Pruning[末位淘汰]:::process
    
    Pruning --> FindWorst[找到权重绝对值最小的索引]
    FindWorst --> Remove[移除该因子]
    Remove --> CalcIncrement
    
    CalcIncrement[计算 IC 增量]:::process
    CalcIncrement --> Reward([输出 Reward 给 RL]):::terminator
```