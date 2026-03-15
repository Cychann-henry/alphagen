import os
import yaml
import argparse
import torch
from datetime import datetime, timedelta

from fqf_iqn_qrdqn.agent import QRDQNAgent, IQNAgent, FQFAgent
from alphagen.data.expression import Feature, FeatureType, Ref, StockData
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv


def build_dynamic_windows():
    """Build train/valid/test windows ending one week before today."""
    today = datetime.now().date()
    test_end = today - timedelta(days=7)
    test_start = test_end - timedelta(days=730) + timedelta(days=1)
    valid_end = test_start - timedelta(days=1)
    valid_start = valid_end - timedelta(days=365) + timedelta(days=1)
    train_start = datetime(2010, 1, 1).date()
    train_end = valid_start - timedelta(days=1)
    return {
        'train_start': train_start.strftime('%Y-%m-%d'),
        'train_end': train_end.strftime('%Y-%m-%d'),
        'valid_start': valid_start.strftime('%Y-%m-%d'),
        'valid_end': valid_end.strftime('%Y-%m-%d'),
        'test_start': test_start.strftime('%Y-%m-%d'),
        'test_end': test_end.strftime('%Y-%m-%d')
    }


def run(args):

    # torch.cuda.set_device(args.cuda)
    config_path = os.path.join('config', f'{args.model}.yaml')

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    device = torch.device(f'cuda')
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    instruments: float = 'csi300'
    windows = build_dynamic_windows()

    print('[DRL-csi300] 时间段划分（动态）:')
    print(f"  训练段:   {windows['train_start']} ~ {windows['train_end']}")
    print(f"  验证段:   {windows['valid_start']} ~ {windows['valid_end']}")
    print(f"  测试段:   {windows['test_start']} ~ {windows['test_end']}")

    data_train = StockData(instrument=instruments,
                           start_time=windows['train_start'],
                           end_time=windows['train_end'])
    data_valid = StockData(instrument=instruments,
                           start_time=windows['valid_start'],
                           end_time=windows['valid_end'])
    data_test = StockData(instrument=instruments,
                          start_time=windows['test_start'],
                          end_time=windows['test_end'])
    train_calculator = QLibStockDataCalculator(data_train, target)
    valid_calculator = QLibStockDataCalculator(data_valid, target)
    test_calculator = QLibStockDataCalculator(data_test, target)
    train_pool = MseAlphaPool(capacity=args.pool,
                           calculator=train_calculator,
                           ic_lower_bound=None,
                           l1_alpha=5e-3)
    train_env = AlphaEnv(pool=train_pool, device=device, print_expr=True)

    # Specify the directory to log.
    name = args.model
    time = datetime.now().strftime("%Y%m%d-%H%M")
    if name == 'qrdqn':
        log_dir = os.path.join('AlphaQCM_data/csi300_logs',
                           f"pool_{args.pool}",
                           f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}")
    elif name == 'iqn':
        log_dir = os.path.join('AlphaQCM_data/csi300_logs',
                           f"pool_{args.pool}",
                           f"{name}-seed{args.seed}-{time}-N{config['K']}-lr{config['lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}")
    elif name == 'fqf':
        log_dir = os.path.join('AlphaQCM_data/csi300_logs',
                           f"pool_{args.pool}",
                           f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['quantile_lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}")

    # Create the agent and run.
    if name == 'qrdqn':
        agent = QRDQNAgent(env=train_env,
                           valid_calculator=valid_calculator,
                           test_calculator=test_calculator,
                           log_dir=log_dir,
                           seed=args.seed,
                           cuda=True, **config)
    elif name == 'iqn':
        agent = IQNAgent(env=train_env,
                         valid_calculator=valid_calculator,
                         test_calculator=test_calculator,
                         log_dir=log_dir,
                         seed=args.seed,
                         cuda=True, **config)
    elif name == 'fqf':
        agent = FQFAgent(env=train_env,
                         valid_calculator=valid_calculator,
                         test_calculator=test_calculator,
                         log_dir=log_dir,
                         seed=args.seed,
                         cuda=True, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qrdqn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pool', type=int, default=20)
    args = parser.parse_args()
    run(args)