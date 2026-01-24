#!/bin/bash

# ================= 配置区域 =================
EXP_NAME="run_01_nohup"
# 你的环境名
TARGET_ENV="alphagen_env" 

# 训练参数
GPU_ID="1"
SEED=1
BATCH_SIZE=4096
STEPS=2000000

# 路径
SAVE_PATH="./experiments/${EXP_NAME}"
LOG_PATH="${SAVE_PATH}/tb_logs"
# 日志文件 (你的眼睛)
OUTPUT_LOG="${SAVE_PATH}/training_log.txt"

# ===========================================

# 1. 检查目录
if [ ! -d "$SAVE_PATH" ]; then mkdir -p "$SAVE_PATH"; fi

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

# 3. 构造核心命令
CMD="python -m scripts.rl \
    --device cuda:0 \
    --seed $SEED \
    --save_path '$SAVE_PATH' \
    --tensorboard_log '$LOG_PATH' \
    --batch_size $BATCH_SIZE \
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
