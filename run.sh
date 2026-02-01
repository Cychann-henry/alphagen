#!/bin/bash

# ================= 配置区域 =================
EXP_NAME="alphagen-20pool-50w"
# 你的环境名
TARGET_ENV="alphagen_env" 

# 训练参数
GPU_ID="1"
SEED=1
# 新增：因子池容量 (10, 20, 50, 100)
POOL_CAPACITY=20
# 设置训练步数
STEPS=500000

# 路径
# SAVE_PATH 和 LOG_PATH 在 rl.py 中硬编码，此处设置无效
# SAVE_PATH="./experiments/${EXP_NAME}"
# LOG_PATH="${SAVE_PATH}/tb_logs"
# 日志文件 (你的眼睛) - 此路径保持不变
OUTPUT_LOG="./experiments/${EXP_NAME}/training_log-${POOL_CAPACITY}-${STEPS}}yearto250930.txt"

# ===========================================

# 1. 检查目录
# 只需确保 nohup 日志的目录存在
if [ ! -d "$(dirname "$OUTPUT_LOG")" ]; then mkdir -p "$(dirname "$OUTPUT_LOG")"; fi

echo "----------------------------------------------------"
echo "🚀 AlphaGen 启动脚本 (Nohup版 - 无需Tmux)"
echo "----------------------------------------------------"

# 2. 检查环境
if [ "$CONDA_DEFAULT_ENV" != "$TARGET_ENV" ]; then
    echo "❌ 请先激活环境: conda activate $TARGET_ENV"
    exit 1
fi

echo "✅ 环境已激活，准备后台运行..."
echo "📄 实时日志将输出到: $OUTPUT_LOG"
echo "⚠️ 注意: 训练结果(模型, 因子池, Tensorboard)将输出到默认的 'out/' 目录。"

# 3. 构造核心命令
# 修改说明：
# 1. 将 --seed 改为 --random_seeds 以匹配 rl.py 定义
# 2. 增加 --pool_capacity 参数
CMD="python -m scripts.rl \
    --random_seeds $SEED \
    --pool_capacity $POOL_CAPACITY \
    --steps $STEPS"

# 4. 使用 nohup 后台启动
# > $OUTPUT_LOG 2>&1  意思是把所有输出（包括报错）都写入日志文件
# &                   意思是放入后台
export CUDA_VISIBLE_DEVICES=$GPU_ID
nohup $CMD > "$OUTPUT_LOG" 2>&1 &

# 获取刚才启动的进程 PID
PID=$!

echo "----------------------------------------------------"
echo "🎉 任务已在后台启动！(PID: $PID)"
echo "----------------------------------------------------"
echo "👀 【如何查看进度】(按 Ctrl+C 退出查看，不会打断任务):"
echo "   tail -f $OUTPUT_LOG"
echo ""
echo "🛑 【如何停止任务】:"
echo "   kill $PID"
echo "----------------------------------------------------"
