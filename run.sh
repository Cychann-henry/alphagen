#!/bin/bash

# ================= 配置区域 =================
EXP_NAME="run_01_nohup"
# 你的环境名
TARGET_ENV="alphagen_env" 

# 训练参数
GPU_ID="1"
SEED=1
# BATCH_SIZE 在 rl.py 中硬编码，此处设置无效
# BATCH_SIZE=4096
STEPS=200000

# 路径
# SAVE_PATH 和 LOG_PATH 在 rl.py 中硬编码，此处设置无效
# SAVE_PATH="./experiments/${EXP_NAME}"
# LOG_PATH="${SAVE_PATH}/tb_logs"
# 日志文件 (你的眼睛) - 此路径保持不变
OUTPUT_LOG="./experiments/${EXP_NAME}/training_log.txt"

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
# 只向 rl.py 传递它能识别的参数。
# 其他参数如 device, save_path, tensorboard_log 等在 rl.py 内部硬编码，
# 从命令行传递是无效的，并且会导致程序在最后崩溃。
CMD="python -m scripts.rl \
    --seed $SEED \
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
