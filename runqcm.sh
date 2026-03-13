#!/bin/bash
# =============================================================================
# AlphaQCM 运行脚本 (QCM 层级)
# 用途：环境检查 → 选择数据集/模型 → 运行 train_qcm / train_qcm_csi300 / train_qcm_csi500
# 可选：运行 DRL 基线 train_drl_csi300（使用 config/ 下配置，非 qcm_config）
#
# 环境适配：
#   1) 创建 conda 环境: conda env create -f alphaqcm_env.yml
#   2) 激活环境: conda activate alphaqcm
#   3) 需在项目根目录执行（脚本会自动 cd 到根目录）；依赖 PYTHONPATH 已在本脚本中设置
#   4) Windows 下可用 Git Bash 或 WSL 运行本脚本
# =============================================================================

# ----------------------------- 配置区域 -----------------------------
# 目标 conda 环境（需先创建：conda env create -f alphaqcm_env.yml）
TARGET_ENV="alphaqcm"

# GPU
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

# 任务类型: qcm_all | qcm_csi300 | qcm_csi500 | drl_csi300
TASK="qcm_csi300"
# 模型: qrdqn | iqn | fqf
MODEL="qrdqn"
SEED=0
POOL=20
# QCM 专用（train_qcm*.py 使用，train_drl_csi300 忽略）
STD_LAM=1.0

# 日志目录（nohup 输出）
LOG_DIR="./AlphaQCM_data"
OUTPUT_LOG="${LOG_DIR}/runqcm_${TASK}_${MODEL}_seed${SEED}_$(date +%Y%m%d-%H%M).txt"
# ---------------------------------------------------------------------

# 切换到项目根目录（脚本所在目录的上级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 1. 环境检查
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$TARGET_ENV" ]; then
    echo "请先激活 QCM 环境: conda activate $TARGET_ENV"
    echo "若未创建环境，请先执行: conda env create -f alphaqcm_env.yml"
    exit 1
fi

# 2. 可选：设置 Python 路径，保证 alphagen/alphagen_qlib/fqf_iqn_qrdqn 可被导入
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# 3. 确保日志目录存在
mkdir -p "$LOG_DIR"

# 4. 根据 TASK 选择入口与参数
case "$TASK" in
    qcm_all)
        ENTRY="train_qcm.py"
        LOG_SUBDIR="alpha_logs"
        EXTRA_ARGS="--model $MODEL --seed $SEED --pool $POOL --std-lam $STD_LAM"
        ;;
    qcm_csi300)
        ENTRY="train_qcm_csi300.py"
        LOG_SUBDIR="csi300_logs"
        EXTRA_ARGS="--model $MODEL --seed $SEED --pool $POOL --std-lam $STD_LAM"
        ;;
    qcm_csi500)
        ENTRY="train_qcm_csi500.py"
        LOG_SUBDIR="csi500_logs"
        EXTRA_ARGS="--model $MODEL --seed $SEED --pool $POOL --std-lam $STD_LAM"
        ;;
    drl_csi300)
        # DRL 基线：使用 config/*.yaml，无 std_lam
        ENTRY="train_drl_csi300.py"
        LOG_SUBDIR="csi300_logs"
        EXTRA_ARGS="--model $MODEL --seed $SEED --pool $POOL"
        ;;
    *)
        echo "未知 TASK: $TASK"
        echo "可选: qcm_all | qcm_csi300 | qcm_csi500 | drl_csi300"
        exit 1
        ;;
esac

CMD="python $ENTRY $EXTRA_ARGS"
echo "----------------------------------------------------"
echo "AlphaQCM 运行脚本 (TASK=$TASK, MODEL=$MODEL)"
echo "----------------------------------------------------"
echo "环境: $CONDA_DEFAULT_ENV"
echo "命令: $CMD"
echo "日志: $OUTPUT_LOG"
echo "训练结果将写入: ${LOG_DIR}/${LOG_SUBDIR}/"
echo "----------------------------------------------------"

# 5. 后台运行（如需前台直接运行，可改为: exec $CMD）
export CUDA_VISIBLE_DEVICES=$GPU_ID
nohup $CMD > "$OUTPUT_LOG" 2>&1 &
PID=$!

echo "已后台启动 (PID: $PID)"
echo "查看进度: tail -f $OUTPUT_LOG"
echo "停止任务: kill $PID"
