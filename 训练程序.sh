#!/bin/bash

# ================= 配置区域 =================

# 1. 实验名称 (同时也会作为 Tmux 的会话名)
# 建议每次跑新实验改一下这里，比如 run_01, run_02
EXP_NAME="run_01_baseline"

# 2. 目标环境名称 (脚本会检查你是否在这个环境里)
TARGET_ENV="alphagen_env"

# 3. 指定显卡 (0, 1, 2, 3)
GPU_ID="0"

# 4. 训练参数
SEED=1
BATCH_SIZE=4096  # 4090 专属优化参数
STEPS=2000000    # 200万步

# 5. 路径设置
SAVE_PATH="./experiments/${EXP_NAME}"
LOG_PATH="${SAVE_PATH}/tb_logs"

# ================= 逻辑区域 (自动安检与启动) =================

echo "----------------------------------------------------"
echo "🤖 AlphaGen 自动训练管家"
echo "----------------------------------------------------"

# [安检 1] 检查是否安装了 tmux
if ! command -v tmux &> /dev/null; then
    echo "❌ 错误: 未找到 tmux 命令。"
    echo "💡 提示: 请确保你已激活环境，并且在该环境下安装了 tmux (conda install -c conda-forge tmux)"
    exit 1
fi

# [安检 2] 检查当前是否激活了正确的 Conda 环境
# 注意：因为你的 tmux 是装在环境里的，必须先激活才能用
if [ "$CONDA_DEFAULT_ENV" != "$TARGET_ENV" ]; then
    echo "❌ 错误: 当前环境是 ($CONDA_DEFAULT_ENV)，但需要在 ($TARGET_ENV) 下运行。"
    echo "💡 修正: 请执行 'conda activate $TARGET_ENV' 后再运行脚本。"
    exit 1
fi

# [安检 3] 检查是否已经有同名的 tmux 会话在跑 (防止重复启动)
tmux has-session -t "$EXP_NAME" 2>/dev/null
if [ $? == 0 ]; then
    echo "⚠️  警告: 已经存在名为 '$EXP_NAME' 的 tmux 会话！"
    echo "👉 若要查看，请运行: tmux attach -t $EXP_NAME"
    echo "👉 若要强制重跑，请先杀死它: tmux kill-session -t $EXP_NAME"
    exit 1
fi

echo "✅ 环境检查通过 ($CONDA_DEFAULT_ENV)"
echo "🚀 准备启动后台任务..."
echo "   - 会话名称: $EXP_NAME"
echo "   - 使用显卡: $GPU_ID"
echo "   - Batch Size: $BATCH_SIZE"

# ================= 核心操作 (Tmux 自动托管) =================

# 1. 创建一个新的后台 Tmux 会话 (Detached mode)
tmux new-session -d -s "$EXP_NAME"

# 2. 为了保险，在 Tmux 会话内部再次激活环境
# (因为 Tmux 启动时可能会重置为 base 环境)
tmux send-keys -t "$EXP_NAME" "source ~/miniconda3/etc/profile.d/conda.sh" C-m
# 如果你的 miniconda 安装路径不同，请修改上面这一行，或者直接依赖 ~/.bashrc
tmux send-keys -t "$EXP_NAME" "conda activate $TARGET_ENV" C-m

# 3. 构造训练命令
CMD="export CUDA_VISIBLE_DEVICES=$GPU_ID && python -m scripts.rl --device cuda:0 --seed $SEED --save_path '$SAVE_PATH' --tensorboard_log '$LOG_PATH' --batch_size $BATCH_SIZE --steps $STEPS"

# 4. 将命令发送到 Tmux 会话并执行
tmux send-keys -t "$EXP_NAME" "$CMD" C-m

echo "----------------------------------------------------"
echo "🎉 成功！任务已在后台运行。"
echo "----------------------------------------------------"
echo "👇 常用指令："
echo "   👁️  查看进度: tmux attach -t $EXP_NAME"
echo "   🏃 退出查看: 按 Ctrl+B 然后按 D (任务继续运行)"
echo "   📈 查看日志: tensorboard --logdir $LOG_PATH"
echo "----------------------------------------------------"