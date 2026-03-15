# QCM 专用环境 Wrapper：使用 config_qcm 与 core_qcm，无 subexprs，与 alphagen 基础 wrapper 隔离
# 使用方式：from alphagen.rl.env.wrapper_qcm import AlphaEnvQcm

from typing import Tuple, List, Optional
import gymnasium as gym
import numpy as np

from alphagen.config_qcm import (
    MAX_EXPR_LENGTH,
    OPERATORS,
    DELTA_TIMES,
    CONSTANTS,
    REWARD_PER_STEP,
)
from alphagen.data.tokens import *
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.rl.env.core_qcm import AlphaEnvCoreQcm

# 与 alphagen wrapper 同构，但无 subexprs，尺寸由 config_qcm 决定
SIZE_NULL = 1
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FeatureType)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1
SIZE_ACTION = SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP


class AlphaEnvWrapperQcm(gym.Wrapper):
    """QCM 体系环境 Wrapper：动作空间仅 OP/Feature/Constant/DeltaTime/SEP，无子表达式槽位。"""
    state: np.ndarray
    env: AlphaEnvCoreQcm
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(self, env: AlphaEnvCoreQcm):
        super().__init__(env)
        self.size_action = SIZE_ACTION
        self.action_space = gym.spaces.Discrete(self.size_action)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.size_action + SIZE_NULL - 1,
            shape=(MAX_EXPR_LENGTH,),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self.env.reset()
        return self.state, {}

    def step(self, action: int):
        _, reward, done, truncated, info = self.env.step(self.action_to_token(action))
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        return self.state, self.reward(reward), done, truncated, info

    def action_to_token(self, action: int) -> Token:
        if action < 0:
            raise ValueError("action must be non-negative")
        if action < SIZE_OP:
            return OperatorToken(OPERATORS[action])
        action -= SIZE_OP
        if action < SIZE_FEATURE:
            return FeatureToken(FeatureType(action))
        action -= SIZE_FEATURE
        if action < SIZE_CONSTANT:
            return ConstantToken(CONSTANTS[action])
        action -= SIZE_CONSTANT
        if action < SIZE_DELTA_TIME:
            return DeltaTimeToken(DELTA_TIMES[action])
        action -= SIZE_DELTA_TIME
        if action == 0:
            return SequenceIndicatorToken(SequenceIndicatorType.SEP)
        raise ValueError("invalid action index")

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()
        offset = 0
        for i in range(offset, offset + SIZE_OP):
            if valid["op"][OPERATORS[i - offset].category_type()]:
                res[i] = True
        offset += SIZE_OP
        if valid["select"][1]:
            res[offset : offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE
        if valid["select"][2]:
            res[offset : offset + SIZE_CONSTANT] = True
        offset += SIZE_CONSTANT
        if valid["select"][3]:
            res[offset : offset + SIZE_DELTA_TIME] = True
        offset += SIZE_DELTA_TIME
        if valid["select"][4]:
            res[offset] = True
        return res


def AlphaEnvQcm(pool: AlphaPoolBase, subexprs: Optional[List] = None, **kwargs):
    """QCM 体系工厂：创建使用 config_qcm + core_qcm + wrapper_qcm 的环境（忽略 subexprs）。"""
    return AlphaEnvWrapperQcm(AlphaEnvCoreQcm(pool=pool, **kwargs))
