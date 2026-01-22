from enum import IntEnum
from typing import Type
from alphagen_qlib.stock_data import FeatureType
from alphagen.data.expression import Operator, Expression


class SequenceIndicatorType(IntEnum):
    """
    序列指示器类型的枚举。
    用于在序列生成过程中标记开始或结束位置。
    """
    BEG = 0  # Beginning of Sequence (序列开始)
    SEP = 1  # Separator / End of Sequence (序列分隔/结束)


class Token:
    """
    所有 Token 的基类。
    Token 是表达式生成过程中的最小单位（词汇）。
    """
    def __repr__(self):
        return str(self)


class ConstantToken(Token):
    """
    常数 Token。
    用于表示数学常数（如 0.5, -1.0 等）。
    """
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def __str__(self): return str(self.constant)


class DeltaTimeToken(Token):
    """
    时间间隔 Token。
    用于表示时间序列操作的时间窗口大小（如 10天, 30天）。
    """
    def __init__(self, delta_time: int) -> None:
        self.delta_time = delta_time

    def __str__(self): return str(self.delta_time)


class FeatureToken(Token):
    """
    特征 Token。
    用于表示原始的基础行情数据特征（如 Open, Close, Volume 等）。
    """
    def __init__(self, feature: FeatureType) -> None:
        self.feature = feature

    def __str__(self): return '$' + self.feature.name.lower()


class OperatorToken(Token):
    """
    操作符 Token。
    用于表示数学运算或逻辑运算函数（如 Add, Mean, Ts_Rank 等）。
    """
    def __init__(self, operator: Type[Operator]) -> None:
        self.operator = operator

    def __str__(self): return self.operator.__name__


class SequenceIndicatorToken(Token):
    """
    序列指示器 Token。
    包装了 SequenceIndicatorType，用于控制生成的流程（开始/停止）。
    """
    def __init__(self, indicator: SequenceIndicatorType) -> None:
        self.indicator = indicator

    def __str__(self): return self.indicator.name


class ExpressionToken(Token):
    """
    表达式 Token。
    将一个完整的 Expression 对象封装为一个 Token，通常用于将复杂的子树视为单个节点。
    """
    def __init__(self, expr: Expression) -> None:
        self.expression = expr

    def __str__(self): return str(self.expression)


# 预定义的全局 Token 实例，用于标记序列的开始和结束
BEG_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.BEG)
SEP_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.SEP)
