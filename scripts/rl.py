import json
import os
import shutil
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from openai import OpenAI
import fire

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.prompts.interaction import InterativeSession, DefaultInteraction


def read_alphagpt_init_pool(seed: int) -> List[Expression]:
    """
    读取由 AlphaGPT 预先生成的初始因子池。
    """
    DIR = "./out/llm-tests/interaction"
    parser = build_parser()
    # 遍历指定目录，查找匹配种子号的报告文件
    for path in Path(DIR).glob(f"v0_{seed}*"):
        with open(path / "report.json") as f:
            data = json.load(f)
            pool_state = data[-1]["pool_state"]
            # 解析并返回因子表达式列表
            return [parser.parse(expr) for expr, _ in pool_state]
    return []


def build_parser() -> ExpressionParser:
    """
    构建并配置一个表达式解析器。
    """
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        # 为一些非标准的操作符名称提供映射
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


def build_chat_client(log_dir: str) -> ChatClient:
    """
    构建并配置一个与大语言模型交互的客户端。
    """
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    return OpenAIClient(
        client=OpenAI(base_url="https://api.ai.cs.ac.cn/v1"),
        config=ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=logger
        )
    )


class CustomCallback(BaseCallback):
    """
    自定义的回调函数，用于在训练过程中执行特定操作，如保存模型、评估因子池、与LLM交互等。
    """
    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5,
        pool_capacity: int = 10,
        segments: Optional[List[Tuple[str, str]]] = None,
        train_end_date: str = ""
    ):
        """
        初始化回调函数。
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

        self.llm_use_count = 0
        self.last_llm_use = 0
        self.obj_history: List[Tuple[int, float]] = []
        self.llm_every_n_steps = llm_every_n_steps
        self.chat_session = chat_session
        self._drop_rl_n = drop_rl_n
        self.pool_capacity = pool_capacity
        self.segments = segments or []
        self.train_end_date = train_end_date

    def _on_step(self) -> bool:
        """
        在每个训练步骤后调用，返回 True 以继续训练。
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        在每次 rollout (数据收集) 结束后调用。
        """
        # 如果配置了聊天会话，则尝试调用 LLM
        if self.chat_session is not None:
            self._try_use_llm()

        # 记录因子池的各种状态指标
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        
        # 在测试集上评估当前因子池的表现
        n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calculator.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
        self.logger.record(f'test/ic_mean', ic_test_mean)
        self.logger.record(f'test/rank_ic_mean', rank_ic_test_mean)
        
        # 保存模型和因子池的快照
        self.save_checkpoint()

    def save_checkpoint(self):
        """
        保存模型和因子池状态到文件。
        """
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        # 如果详细模式开启，打印保存信息
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        
        # 修改 3: 文件名增加因子池容量后缀
        # 原来是: f'{path}_pool.json'
        # 现在变为: f'{path}_pool_{self.pool_capacity}.json'
        with open(f'{path}_pool_{self.pool_capacity}.json', 'w') as f:
            json.dump(self.pool.to_json_dict(), f)

    def show_pool_state(self):
        """
        在控制台打印当前因子池的详细状态。
        """
        state = self.pool.state
        print('---------------------------------------------')
        for i in range(self.pool.size):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    def _try_use_llm(self) -> None:
        """
        尝试调用大语言模型来优化因子池。
        """
        n_steps = self.num_timesteps
        # 判断是否达到了调用 LLM 的时间间隔
        if n_steps - self.last_llm_use < self.llm_every_n_steps:
            return
        self.last_llm_use = n_steps
        self.llm_use_count += 1
        
        assert self.chat_session is not None
        self.chat_session.client.reset()
        logger = self.chat_session.logger
        logger.debug(
            f"[Step: {n_steps}] Trying to invoke LLM (#{self.llm_use_count}): "
            f"IC={self.pool.best_ic_ret:.4f}, obj={self.pool.best_ic_ret:.4f}")

        try:
            # 在调用 LLM 前，先移除一部分由 RL 生成的、权重较低的因子
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            logger.warning(f"LLM invocation failed due to {type(e)}: {str(e)}")

    @property
    def pool(self) -> LinearAlphaPool:
        """
        获取环境中的因子池对象。
        """
        assert(isinstance(self.env_core.pool, LinearAlphaPool))
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        """
        获取底层的 AlphaEnvCore 环境实例。
        """
        return self.training_env.envs[0].unwrapped  # type: ignore


def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
    alphagpt_init: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 5,
    llm_replace_n: int = 3
):
    """
    运行单次完整的强化学习实验。
    """
    reseed_everything(seed)
    
    # 修改：将初始化路径指向 cn_data_2024h1，确保能读取到 2022-2023 年的数据
    initialize_qlib("~/.qlib/qlib_data/cn_data_2024h1")

    # 如果不使用 LLM，则替换的因子数量为 0
    llm_replace_n = 0 if not use_llm else llm_replace_n
    print(f"""[Main] Starting training process
    Seed: {seed}
    Instruments: {instruments}
    Pool capacity: {pool_capacity}
    Total Iteration Steps: {steps}
    AlphaGPT-Like Init-Only LLM Usage: {alphagpt_init}
    Use LLM: {use_llm}
    Invoke LLM every N steps: {llm_every_n_steps}
    Replace N alphas with LLM: {llm_replace_n}
    Drop N alphas before LLM: {drop_rl_n}""")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # tag = "rlv2" if llm_add_subexpr == 0 else f"afs{llm_add_subexpr}aar1-5"
    # 根据实验配置生成一个标签，用于区分不同的实验类型
    tag = (
        "agpt" if alphagpt_init else
        "rl" if not use_llm else
        f"llm_d{drop_rl_n}")
    name_prefix = f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{tag}"
    save_path = os.path.join("./out/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    def get_dataset(start: str, end: str) -> StockData:
        """辅助函数，用于获取指定时间段的股票数据。"""
        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            device=device
        )

    # ====== 动态计算训练和测试时间段 ======
    # 以当前日期往前推7天作为最后测试段的末尾，3个测试段各半年依次往前推导
    # 训练段从2012年开始到第一个测试段之前

    today = datetime.now().date()
    test_end = today - timedelta(days=7)                        # 最后测试段末尾
    test3_start = test_end - relativedelta(months=6) + timedelta(days=1)  # 第4段起始
    test2_end = test3_start - timedelta(days=1)                 # 第3段末尾
    test2_start = test2_end - relativedelta(months=6) + timedelta(days=1)  # 第3段起始
    test1_end = test2_start - timedelta(days=1)                 # 第2段末尾
    test1_start = test1_end - relativedelta(months=6) + timedelta(days=1)  # 第2段起始
    train_start = "2012-01-01"                                  # 训练段起始（固定）
    train_end = test1_start - timedelta(days=1)                 # 训练段末尾

    # 格式化日期为字符串
    train_start_str = train_start
    train_end_str = train_end.strftime("%Y-%m-%d")
    test1_start_str = test1_start.strftime("%Y-%m-%d")
    test1_end_str = test1_end.strftime("%Y-%m-%d")
    test2_start_str = test2_start.strftime("%Y-%m-%d")
    test2_end_str = test2_end.strftime("%Y-%m-%d")
    test3_start_str = test3_start.strftime("%Y-%m-%d")
    test3_end_str = test_end.strftime("%Y-%m-%d")

    print(f"[Main] 时间段划分:")
    print(f"  训练段:   {train_start_str} ~ {train_end_str}")
    print(f"  测试段1:  {test1_start_str} ~ {test1_end_str}")
    print(f"  测试段2:  {test2_start_str} ~ {test2_end_str}")
    print(f"  测试段3:  {test3_start_str} ~ {test3_end_str}")

    segments = [
        (train_start_str, train_end_str),
        (test1_start_str, test1_end_str),
        (test2_start_str, test2_end_str),
        (test3_start_str, test3_end_str)
    ]
    datasets = [get_dataset(*s) for s in segments]
    calculators = [QLibStockDataCalculator(d, target) for d in datasets]

    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        """辅助函数，用于构建一个因子池实例。"""
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device
        )
        # 如果提供了初始表达式，则强制加载它们
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    chat, inter, pool = None, None, build_pool([])
    # 如果使用 AlphaGPT 方式初始化，则读取预生成的因子池
    if alphagpt_init:
        pool = build_pool(read_alphagpt_init_pool(seed))
    # 如果使用 LLM 交互，则构建客户端和交互会话
    elif use_llm:
        chat = build_chat_client(save_path)
        inter = DefaultInteraction(
            build_parser(), chat, build_pool,
            calculator_train=calculators[0], calculators_test=calculators[1:],
            replace_k=llm_replace_n, forgetful=True
        )
        pool = inter.run()

    # 创建强化学习环境
    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=True
    )
    # 创建自定义回调
    checkpoint_callback = CustomCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n,
        pool_capacity=pool_capacity,
        segments=segments,
        train_end_date=train_end_str
    )
    # 创建并配置 PPO 模型
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log="./out/tensorboard",
        device=device,
        verbose=1,
    )
    # 开始训练
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=name_prefix,
    )

    # ====== 训练完成后，自动保存最终因子池到统一目录 ======
    latest_dir = os.path.join("./out/latest_factors")
    os.makedirs(latest_dir, exist_ok=True)

    # 找到当前实验目录中步数最大的因子池文件
    pool_files = [
        f for f in os.listdir(save_path)
        if f.endswith(f'_pool_{pool_capacity}.json')
    ]
    if pool_files:
        # 按步数排序，取最大的
        def extract_steps(filename: str) -> int:
            try:
                return int(filename.split('_steps_pool')[0])
            except (ValueError, IndexError):
                return 0
        pool_files.sort(key=extract_steps)
        latest_pool_file = pool_files[-1]
        final_steps = extract_steps(latest_pool_file)

        # 生成统一的标准文件名
        latest_pool_name = f"pool_{instruments}_{pool_capacity}_{test3_end_str}.json"
        src = os.path.join(save_path, latest_pool_file)
        dst = os.path.join(latest_dir, latest_pool_name)
        shutil.copy2(src, dst)

        # 保存元数据
        metadata = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "instruments": instruments,
            "pool_capacity": pool_capacity,
            "total_steps": final_steps,
            "seed": seed,
            "train_end_date": train_end_str,
            "test_end_date": test3_end_str,
            "segments": {
                "train": {"start": train_start_str, "end": train_end_str},
                "test1": {"start": test1_start_str, "end": test1_end_str},
                "test2": {"start": test2_start_str, "end": test2_end_str},
                "test3": {"start": test3_start_str, "end": test3_end_str}
            },
            "pool_file": latest_pool_name,
            "source_experiment": name_prefix,
            "source_path": os.path.abspath(os.path.join(save_path, latest_pool_file))
        }
        metadata_path = os.path.join(latest_dir, f"metadata_{instruments}_{pool_capacity}_{test3_end_str}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ 最终因子池已保存到: {dst}")
        print(f"📋 元数据已保存到: {metadata_path}")
        print(f"   - 训练结束日期: {train_end_str}")
        print(f"   - 测试结束日期: {test3_end_str}")
        print(f"   - 因子池容量: {pool_capacity}")
        print(f"   - 训练步数: {final_steps}")
        print(f"{'='*60}")
    else:
        print("\n⚠️ 未找到因子池文件，跳过自动保存。")


def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: str = "csi300",
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25000
):
    """
    :param random_seeds: Random seeds
    :param pool_capacity: Maximum size of the alpha pool
    :param instruments: Stock subset name
    :param alphagpt_init: Use an alpha set pre-generated by LLM as the initial pool
    :param use_llm: Enable LLM usage
    :param drop_rl_n: Drop n worst alphas before invoke the LLM
    :param steps: Total iteration steps
    :param llm_every_n_steps: Invoke LLM every n steps

    主函数，用于批量运行实验。

    :param random_seeds: 随机种子，可以是一个整数或一个整数元组，用于多次实验。
    :param pool_capacity: 因子池的最大容量。
    :param instruments: 股票池的名称 (例如 'csi300', 'csi500')。
    :param alphagpt_init: 是否使用由 LLM 预先生成的一组因子作为初始池。
    :param use_llm: 是否在训练过程中启用 LLM 进行交互式优化。
    :param drop_rl_n: 在调用 LLM 之前，从池中移除表现最差的 n 个由 RL 生成的因子。
    :param steps: 总的训练步数。如果为 None，则根据因子池容量使用默认值。
    :param llm_every_n_steps: 每隔 n 步调用一次 LLM。
    """
    
    # 如果只提供了一个整数种子，将其转换为元组以便迭代
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds, )
    
    # 根据因子池容量设置默认的训练步数
    default_steps = {
        10: 200_000,
        20: 250_000,
        50: 300_000,
        100: 350_000
    }
    
    # 遍历所有种子，运行实验
    for s in random_seeds:
        run_single_experiment(
            seed=s,
            instruments=instruments,
            pool_capacity=pool_capacity,
            steps=default_steps[int(pool_capacity)] if steps is None else int(steps),
            alphagpt_init=alphagpt_init,
            drop_rl_n=drop_rl_n,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps
        )


if __name__ == '__main__':
    # 使用 fire 库将 main 函数暴露
    fire.Fire(main)
