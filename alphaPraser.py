import json
import math
from alphagen.data.expression import *
# 假设你在 alphagen 项目根目录下运行，且正确设置了 PYTHONPATH
# 你可能需要根据实际 import路径调整，例如: from alphagen.data.parser import ExpressionParser

def load_and_print_formulas(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    exprs = data['exprs']
    weights = data['weights']
    
    print(f"{'ID':<4} | {'Weight':<10} | {'Expression'}")
    print("-" * 80)
    
    for i, (expr_str, weight) in enumerate(zip(exprs, weights)):
        # 这里直接打印原始表达式，也可以尝试用 ExpressionParser 解析后打印 str(expr_obj)
        # 如果你有 alphagen 环境，expr_obj = ExpressionParser.parse(expr_str)
        print(f"{i+1:<4} | {weight:<10.4f} | {expr_str}")

# 使用方法
# filepath: /path/to/script.py
load_and_print_formulas('/home/user/XinhuiLu/alphagen/alphagen/out/results/csi300_20_0_20260127223925_rl/200704_steps_pool.json')