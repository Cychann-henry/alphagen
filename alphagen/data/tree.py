from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Type

from alphagen.data.exception import InvalidExpressionException
from alphagen.data.expression import (
    BinaryOperator,
    Constant,
    DeltaTime,
    Expression,
    Feature,
    Operator,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)
from alphagen.data.tokens import *


class ExpressionBuilder:
    stack: List[Expression]

    def __init__(self, max_repeat_per_symbol: int = 2):
        self.stack = []
        self._max_repeat_per_symbol = max_repeat_per_symbol

    def _iter_exprs(self, exprs: Iterable[Expression]) -> Iterable[Expression]:
        for e in exprs:
            if isinstance(e, Expression) and not isinstance(e, DeltaTime):
                yield e

    def _collect_counts(self, expr: Expression, op_counter: Counter[str], feat_counter: Counter[str]) -> None:
        if isinstance(expr, Feature):
            feat_counter[expr._feature.name] += 1  # type: ignore[attr-defined]
            return
        if isinstance(expr, Operator):
            op_counter[type(expr).__name__] += 1
            for child in expr.operands:
                self._collect_counts(child, op_counter, feat_counter)
            return
        # Constant / other Expression nodes: nothing to count

    def _violates_repeat_limit(self, hypothetical_stack: List[Expression]) -> bool:
        op_counter: Counter[str] = Counter()
        feat_counter: Counter[str] = Counter()
        for e in self._iter_exprs(hypothetical_stack):
            self._collect_counts(e, op_counter, feat_counter)
        limit = self._max_repeat_per_symbol
        return any(v > limit for v in op_counter.values()) or any(v > limit for v in feat_counter.values())

    def get_tree(self) -> Expression:
        if len(self.stack) == 1:
            return self.stack[0]
        else:
            raise InvalidExpressionException(f"Expected only one tree, got {len(self.stack)}")

    def add_token(self, token: Token):
        if not self.validate(token):
            raise InvalidExpressionException(f"Token {token} not allowed here, stack: {self.stack}.")
        if isinstance(token, OperatorToken):
            n_args: int = token.operator.n_args()
            children = []
            for _ in range(n_args):
                children.append(self.stack.pop())
            self.stack.append(token.operator(*reversed(children)))  # type: ignore
        elif isinstance(token, ConstantToken):
            self.stack.append(Constant(token.constant))
        elif isinstance(token, DeltaTimeToken):
            self.stack.append(DeltaTime(token.delta_time))
        elif isinstance(token, FeatureToken):
            self.stack.append(Feature(token.feature))
        elif isinstance(token, ExpressionToken):
            self.stack.append(token.expression)
        else:
            assert False

    def is_valid(self) -> bool:
        return len(self.stack) == 1 and self.stack[0].is_featured

    def validate(self, token: Token) -> bool:
        if isinstance(token, OperatorToken):
            return self.validate_op(token.operator)
        elif isinstance(token, DeltaTimeToken):
            return self.validate_dt()
        elif isinstance(token, ConstantToken):
            return self.validate_const()
        elif isinstance(token, (FeatureToken, ExpressionToken)):
            if not self.validate_featured_expr():
                return False
            if isinstance(token, FeatureToken):
                hypothetical = self.stack + [Feature(token.feature)]
            else:
                hypothetical = self.stack + [token.expression]
            return not self._violates_repeat_limit(hypothetical)
        else:
            assert False

    def validate_op(self, op: Type[Operator]) -> bool:
        if len(self.stack) < op.n_args():
            return False

        if issubclass(op, UnaryOperator):
            if not self.stack[-1].is_featured:
                return False
        elif issubclass(op, BinaryOperator):
            if not self.stack[-1].is_featured and not self.stack[-2].is_featured:
                return False
            if (isinstance(self.stack[-1], DeltaTime) or
                    isinstance(self.stack[-2], DeltaTime)):
                return False
        elif issubclass(op, RollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured:
                return False
        elif issubclass(op, PairRollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured or not self.stack[-3].is_featured:
                return False
        else:
            assert False
        n_args = op.n_args()
        children = list(reversed(self.stack[-n_args:]))
        hypothetical_stack = self.stack[:-n_args] + [op(*children)]  # type: ignore[misc]
        return not self._violates_repeat_limit(hypothetical_stack)

    def validate_dt(self) -> bool:
        return len(self.stack) > 0 and self.stack[-1].is_featured

    def validate_const(self) -> bool:
        return len(self.stack) == 0 or self.stack[-1].is_featured

    def validate_featured_expr(self) -> bool:
        return not (len(self.stack) >= 1 and isinstance(self.stack[-1], DeltaTime))
