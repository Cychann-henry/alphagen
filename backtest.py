from typing import Optional, TypeVar, Callable, Optional, Tuple
import os
import pickle
import warnings
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from qlib.contrib.strategy import TopkDropoutStrategy

from alphagen.data.expression import *
from alphagen.data.parser import parse_expression
from alphagen_generic.features import *
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.utils import load_alpha_pool_by_path


_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    """
    创建路径所需的父目录。
    """
    dir = os.path.dirname(path)
    # 如果父目录路径不为空，则确保该目录存在，如果不存在则创建
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    """
    将文本内容写入指定路径的文件中，会自动创建父目录。
    """
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    """
    将对象序列化并保存到文件（Pickle格式）。
    支持缓存机制：如果文件存在且不强制刷新，则不执行操作；否则调用 factory 生成对象并保存。
    """
    # 如果强制刷新缓存，或者文件不存在，则生成并保存对象
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


@dataclass
class BacktestResult(DataClassJsonMixin):
    """
    回测结果数据类，用于存储回测的关键性能指标。
    """
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 50,
        n_drop: Optional[int] = None,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        """
        初始化回测器。
        参数包括基准指数、持仓股票数量(Top K)、换仓数量(n_drop)、成交价格模式以及交易成本配置。
        """
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: Union[pd.Series, pd.DataFrame],
        output_prefix: Optional[str] = None
    ) -> Tuple[pd.DataFrame, BacktestResult]:
        """
        执行回测。
        根据给定的预测分数（prediction），运行 TopkDropout 策略，并生成回测报告。
        """
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            """
            内部函数：实际执行 Qlib 回测逻辑。
            last 参数用于控制回测结束日期（处理数据完整性问题）。
            """
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy = TopkDropoutStrategy(
                    signal=prediction,
                    topk=self._top_k,
                    n_drop=self._n_drop,
                    only_tradable=True,
                    forbid_all_trade_at_limit=True
                )
                executor = exec.SimulatorExecutor(
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            # 尝试正常执行回测
            portfolio_metric = backtest_impl()
        except IndexError:
            # 如果出现索引错误（通常是因为预测数据的最后一天在 Qlib 数据中不可用），则尝试回退一天重试
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)
        graph = report_graph(report, show_notebook=False)[0]
        
        # 如果提供了输出前缀，则保存报告、图表和 JSON 结果
        if output_prefix is not None:
            dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            write_all_text(output_prefix + "-result.json", result.to_json())
        return report, result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        """
        分析回测报告 dataframe。
        计算超额收益（Excess Return）和绝对收益的各项风险指标（如夏普比率、最大回撤等）。
        """
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )


if __name__ == "__main__":
    initialize_qlib("~/.qlib/qlib_data/cn_data")
    qlib_backtest = QlibBacktest(top_k=50, n_drop=5)
    data = StockData(
        instrument="csi300",
        start_time="2022-01-01",
        end_time="2023-06-30"
    )
    calc = QLibStockDataCalculator(data, None)

    def run_backtest(prefix: str, seed: int, exprs: List[Expression], weights: List[float]):
        """
        辅助函数：根据提供的因子表达式列表和权重，生成组合因子并运行回测，保存结果到指定路径。
        """
        df = data.make_dataframe(calc.make_ensemble_alpha(exprs, weights))
        qlib_backtest.run(df, output_prefix=f"out/backtests/50-5/{prefix}/{seed}")

    # 遍历 GP (遗传编程) 方法生成的因子池结果并进行回测
    for p in Path("out/gp").iterdir():
        seed = int(p.name)
        with open(p / "40.json") as f:
            report = json.load(f)
        state = report["res"]["res"]["pool_state"]
        run_backtest("gp", seed, [parse_expression(e) for e in state["exprs"]], state["weights"])
    
    exit(0)
    
    # 遍历其他实验结果目录进行回测（已被 exit(0) 截断，不会执行）
    for p in Path("out/results").iterdir():
        inst, size, seed, time, ver = p.name.split('_', 4)
        size, seed = int(size), int(seed)
        # 加上过滤条件：只处理 CSI300、池大小为 20、时间晚于 20240923 且不是 llm_d5 的结果
        if inst != "csi300" or size != 20 or time < "20240923" or ver == "llm_d5":
            continue
        exprs, weights = load_alpha_pool_by_path(str(p / "251904_steps_pool.json"))
        run_backtest(ver, seed, exprs, weights)
    
    # 遍历 LLM 交互实验的结果进行回测
    for p in Path("out/llm-tests/interaction").iterdir():
        # 只处理以 "v1" 开头的文件夹
        if not p.name.startswith("v1"):
            continue
        run = int(p.name[3])
        with open(p / "report.json") as f:
            report = json.load(f)
        state = report[-1]["pool_state"]
        run_backtest("pure_llm", run, [parse_expression(t[0]) for t in state], [t[1] for t in state])
