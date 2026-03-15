# QCM 专用因子池实现：独立于 alphagen.models.linear_alpha_pool，仅依赖 AlphaCalculator 接口
# 使用方式：from alphagen.models.alpha_pool_qcm import AlphaPoolQcm

from itertools import count
import math
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch

from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.models.alpha_pool import AlphaPoolBase


class AlphaPoolQcm(AlphaPoolBase):
    """QCM 体系使用的因子池实现，与 MseAlphaPool 并列，不修改 alphagen 原有逻辑。"""

    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(capacity, calculator, device)

        self.size: int = 0
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self.mutual_ics: np.ndarray = np.identity(capacity + 1)
        self.weights: np.ndarray = np.zeros(capacity + 1)
        self.best_ic_ret: float = -1.0

        self.ic_lower_bound = ic_lower_bound if ic_lower_bound is not None else -1.0
        self.l1_alpha = l1_alpha
        self.eval_cnt = 0

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "exprs": list(self.exprs[: self.size]),
            "ics_ret": list(self.single_ics[: self.size]),
            "weights": list(self.weights[: self.size]),
            "best_ic_ret": self.best_ic_ret,
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "exprs": [str(expr) for expr in self.exprs[: self.size]],
            "weights": list(self.weights[: self.size]),
        }

    def try_new_expr(self, expr: Expression) -> float:
        ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or (np.any(np.isnan(ic_mut)) if ic_mut else False):
            return 0.0

        self._add_factor(expr, ic_ret, ic_mut)
        if self.size > 1:
            new_weights = self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=500)
            worst_idx = np.argmin(np.abs(new_weights))
            if worst_idx != self.capacity:
                self.weights[: self.size] = new_weights
            self._pop()

        new_ic_ret = self.evaluate_ensemble()
        if new_ic_ret > self.best_ic_ret:
            self.best_ic_ret = new_ic_ret
        self.eval_cnt += 1
        return new_ic_ret

    def force_load_exprs(self, exprs: List[Expression]) -> None:
        """QCM 扩展：批量加载表达式入池（不参与 try_new_expr 的淘汰）。"""
        for expr in exprs:
            ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, ic_ret, ic_mut)
            assert self.size <= self.capacity
        self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=500)

    def _optimize(self, alpha: float, lr: float, n_iter: int) -> np.ndarray:
        if math.isclose(alpha, 0.0):
            return self._optimize_lstsq()

        ics_ret = torch.from_numpy(self.single_ics[: self.size]).to(self.device)
        ics_mut = torch.from_numpy(self.mutual_ics[: self.size, : self.size]).to(self.device)
        weights = torch.from_numpy(self.weights[: self.size]).to(self.device).requires_grad_(True)
        optim = torch.optim.Adam([weights], lr=lr)

        loss_ic_min = 1e9 + 7
        best_weights = weights.cpu().detach().numpy()
        iter_cnt = 0
        for it in count():
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()

            loss_l1 = torch.norm(weights, p=1)
            loss = loss_ic + alpha * loss_l1

            optim.zero_grad()
            loss.backward()
            optim.step()

            if loss_ic_min - loss_ic_curr > 1e-6:
                iter_cnt = 0
            else:
                iter_cnt += 1

            if loss_ic_curr < loss_ic_min:
                best_weights = weights.cpu().detach().numpy()
                loss_ic_min = loss_ic_curr

            if iter_cnt >= n_iter or it >= 10000:
                break

        return best_weights

    def _optimize_lstsq(self) -> np.ndarray:
        try:
            return np.linalg.lstsq(
                self.mutual_ics[: self.size, : self.size],
                self.single_ics[: self.size],
                rcond=None,
            )[0]
        except (np.linalg.LinAlgError, ValueError):
            return self.weights[: self.size].copy()

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        ic = calculator.calc_pool_IC_ret(
            list(self.exprs[: self.size]), list(self.weights[: self.size])
        )
        return (float(ic), float(ic))

    def evaluate_ensemble(self) -> float:
        return self.calculator.calc_pool_IC_ret(
            list(self.exprs[: self.size]), list(self.weights[: self.size])
        )

    @property
    def _under_thres_alpha(self) -> bool:
        if self.ic_lower_bound is None or self.size > 1:
            return False
        return self.size == 0 or abs(self.single_ics[0]) < self.ic_lower_bound

    def _calc_ics(
        self,
        expr: Expression,
        ic_mut_threshold: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[List[float]]]:
        single_ic = self.calculator.calc_single_IC_ret(expr)
        if not self._under_thres_alpha and single_ic < self.ic_lower_bound:
            return single_ic, None

        mutual_ics: List[float] = []
        for i in range(self.size):
            ex = self.exprs[i]
            if ex is None:
                continue
            mutual_ic = self.calculator.calc_mutual_IC(expr, ex)
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return single_ic, None
            mutual_ics.append(mutual_ic)

        return single_ic, mutual_ics

    def _add_factor(
        self,
        expr: Expression,
        ic_ret: float,
        ic_mut: List[float],
    ) -> None:
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.single_ics[n] = ic_ret
        for i in range(n):
            self.mutual_ics[i, n] = self.mutual_ics[n, i] = ic_mut[i]
        self.weights[n] = ic_ret
        self.size += 1

    def _pop(self) -> None:
        if self.size <= self.capacity:
            return
        idx = int(np.argmin(np.abs(self.weights)))
        self._swap_idx(idx, self.capacity)
        self.size = self.capacity

    def _swap_idx(self, i: int, j: int) -> None:
        if i == j:
            return
        self.exprs[i], self.exprs[j] = self.exprs[j], self.exprs[i]
        self.single_ics[i], self.single_ics[j] = self.single_ics[j], self.single_ics[i]
        self.mutual_ics[:, [i, j]] = self.mutual_ics[:, [j, i]]
        self.mutual_ics[[i, j], :] = self.mutual_ics[[j, i], :]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]
