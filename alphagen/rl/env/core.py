from typing import Tuple, Optional
import gymnasium as gym
import math

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.models.linear_alpha_pool import LinearAlphaPool
from alphagen.utils import reseed_everything


class AlphaEnvCore(gym.Env):
    pool: AlphaPoolBase
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool

    def __init__(
        self,
        pool: AlphaPoolBase,
        device: torch.device = torch.device('cuda:0'),
        print_expr: bool = False
    ):
        """
        初始化环境。
        设置因子池、计算设备、以及是否打印生成的表达式。
        """
        super().__init__()

        self.pool = pool
        self._print_expr = print_expr
        self._device = device

        self.eval_cnt = 0

        self.render_mode = None
        self.reset()

    def reset(
        self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        """
        重置环境状态。
        初始化 Token 列表（以 BEG_TOKEN 开始）和表达式构建器，准备开始生成新的表达式。
        """
        reseed_everything(seed)
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        """
        环境的一步交互。
        接受一个 Token 动作，更新当前表达式状态。
        如果是结束符或达到最大长度，则进行评估并返回奖励；否则继续构建。
        """
        # 如果动作是序列结束符（SEP），则表示表达式构建完成
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            done = True
        # 如果当前表达式长度未达到最大限制，继续添加 Token
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = 0.0
        # 如果达到最大长度限制，强制结束
        else:
            done = True
            # 如果构建的表达式有效则进行评估，否则奖励为 -1
            reward = self._evaluate() if self._builder.is_valid() else -1.

        # 如果奖励是 NaN，则将其置为 0
        if math.isnan(reward):
            reward = 0.

        return self._tokens, reward, done, False, self._valid_action_types()

    def _evaluate(self):
        """
        评估当前构建的表达式。
        将 Token 序列转换为表达式树，尝试将其加入因子池，并返回由于该因子带来的性能提升（奖励）。
        处理潜在的数据越界错误。
        """
        expr: Expression = self._builder.get_tree()
        # 如果设置了打印表达式，则输出当前表达式
        if self._print_expr:
            print(expr)
        try:
            ret = self.pool.try_new_expr(expr)
            self.eval_cnt += 1
            return ret
        except OutOfDataRangeError:
            return 0.

    def _valid_action_types(self) -> dict:
        """
        获取当前状态下有效的动作类型。
        检查表达式构建器的状态（如是否需要操作符、操作数等），返回允许的 Token 类型结构，用于屏蔽无效动作。
        """
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_featured_expr()
        valid_stop = self._builder.is_valid()

        ret = {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
        return ret

    def valid_action_types(self) -> dict:
        """
        公开接口，返回有效的动作类型。
        """
        return self._valid_action_types()

    def render(self, mode='human'):
        pass
