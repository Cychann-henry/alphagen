# QCM 专用配置：与 alphagen 基础 config 隔离，仅 QCM 体系使用
# 使用方式：from alphagen.config_qcm import MAX_EXPR_LENGTH, OPERATORS, ...

from typing import Type, List
from alphagen.data.expression import *

# QCM: 表达式最大长度 20，DELTA_TIMES 仅 [10,20,30,40,50]
MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

OPERATORS: List[Type[Operator]] = [
    Abs, Log,
    Add, Sub, Mul, Div, Greater, Less,
    Ref, Mean, Sum, Std, Var,
    Max, Min, Med, Mad,
    Delta, WMA, EMA,
    Cov, Corr,
]

DELTA_TIMES = [10, 20, 30, 40, 50]
CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]
REWARD_PER_STEP = 0.
