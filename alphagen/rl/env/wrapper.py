from typing import Tuple, List, Optional
import gymnasium as gym
import numpy as np

from alphagen.config import *
from alphagen.data.tokens import *
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.rl.env.core import AlphaEnvCore

SIZE_NULL = 1
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FeatureType)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1
SIZE_ACTION = SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP


class AlphaEnvWrapper(gym.Wrapper):
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(
        self,
        env: AlphaEnvCore,
        subexprs: Optional[List[Expression]] = None
    ):
        """
        初始化环境包装器。
        将原始的 AlphaEnvCore 包装成符合标准 Gym 接口的环境，
        以便强化学习算法可以直接调用。定义了动作空间和观察空间。
        """
        super().__init__(env)
        self.subexprs = subexprs or []
        self.size_action = SIZE_ACTION + len(self.subexprs)
        self.action_space = gym.spaces.Discrete(self.size_action)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.size_action + SIZE_NULL - 1,
            shape=(MAX_EXPR_LENGTH, ),
            dtype=np.uint8
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        重置环境。
        初始化状态数组（全零），重置计数器，并调用底层环境的 reset 方法。
        """
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self.env.reset()
        return self.state, {}

    def step(self, action: int):
        """
        执行一步动作。
        将整数类型的动作转换为 Token，在底层环境中执行，
        然后更新包装器的状态（将动作记录到 token 序列中）。
        """
        _, reward, done, truncated, info = self.env.step(self.action(action))
        # 如果当前 episode 未结束，将当前动作记录到状态数组中并增加计数器
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        return self.state, self.reward(reward), done, truncated, info

    def action(self, action: int) -> Token:
        """
        将整数动作索引转换为对应的 Token 对象。
        """
        return self.action_to_token(action)

    def reward(self, reward: float) -> float:
        """
        计算奖励。
        在原始奖励的基础上加上每步的基础奖励（REWARD_PER_STEP），通常用于鼓励或惩罚长序列。
        """
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        """
        生成动作掩码。
        根据当前表达式构建的状态，返回一个布尔数组，指示哪些动作是合法的。
        用于在强化学习中屏蔽无效动作。
        """
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()

        offset = 0              # Operators
        for i in range(offset, offset + SIZE_OP):
            # 检查特定类型的操作符是否合法
            if valid['op'][OPERATORS[i - offset].category_type()]:
                res[i] = True
        offset += SIZE_OP
        # 检查特征 (Features) 是否合法
        if valid['select'][1]:  # Features
            res[offset:offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE
        # 检查常数 (Constants) 是否合法
        if valid['select'][2]:  # Constants
            res[offset:offset + SIZE_CONSTANT] = True
        offset += SIZE_CONSTANT
        # 检查时间间隔 (Delta time) 是否合法
        if valid['select'][3]:  # Delta time
            res[offset:offset + SIZE_DELTA_TIME] = True
        offset += SIZE_DELTA_TIME
        # 检查子表达式 (Sub-expressions) 是否合法
        if valid['select'][1]:  # Sub-expressions
            res[offset:offset + len(self.subexprs)] = True
        offset += len(self.subexprs)
        # 检查结束符 (SEP) 是否合法
        if valid['select'][4]:  # SEP
            res[offset] = True
        return res

    def action_to_token(self, action: int) -> Token:
        """
        根据动作索引空间映射，将整数 id 解码为具体的 Token 对象。
        """
        # 动作索引不能为负数
        if action < 0:
            raise ValueError
        # 如果索引在操作符范围内，返回对应的操作符 Token
        if action < SIZE_OP:
            return OperatorToken(OPERATORS[action])
        action -= SIZE_OP
        # 如果索引在特征范围内，返回对应的特征 Token
        if action < SIZE_FEATURE:
            return FeatureToken(FeatureType(action))
        action -= SIZE_FEATURE
        # 如果索引在常数范围内，返回对应的常数 Token
        if action < SIZE_CONSTANT:
            return ConstantToken(CONSTANTS[action])
        action -= SIZE_CONSTANT
        # 如果索引在时间间隔范围内，返回对应的时间间隔 Token
        if action < SIZE_DELTA_TIME:
            return DeltaTimeToken(DELTA_TIMES[action])
        action -= SIZE_DELTA_TIME
        # 如果索引在子表达式范围内，返回对应的表达式 Token
        if action < len(self.subexprs):
            return ExpressionToken(self.subexprs[action])
        action -= len(self.subexprs)
        # 如果索引为 0（剩余），则返回序列结束符 Token
        if action == 0:
            return SequenceIndicatorToken(SequenceIndicatorType.SEP)
        assert False


def AlphaEnv(pool: AlphaPoolBase, subexprs: Optional[List[Expression]] = None, **kwargs):
    """
    辅助函数，用于创建被包装过的 AlphaEnv 实例。
    """
    return AlphaEnvWrapper(AlphaEnvCore(pool=pool, **kwargs), subexprs=subexprs)
