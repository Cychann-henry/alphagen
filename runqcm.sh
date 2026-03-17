#!/bin/bash
# =============================================================================
# AlphaQCM 运行脚本（仅用于 AlphaQCM 体系）
# 用途：环境检查 -> 选择 QCM/baseline 任务 -> 运行 train_qcm* 或 train_drl_csi300
# 说明：本脚本与 run.sh 职责分离；run.sh 负责 AlphaGen（scripts/rl.py），runqcm.sh 负责 AlphaQCM 入口
#
# 环境适配：
#   与 alphagen 共用同一 conda 环境（推荐）：conda activate alphagen_env，可选 pip install -r requirements_qcm_addon.txt
#   或单独建 alphaqcm 环境：conda env create -f alphaqcm_env.yml && conda activate alphaqcm
#   需在项目根目录执行（脚本会自动 cd 到根目录）；依赖 PYTHONPATH 已在本脚本中设置
#   Windows 下可用 Git Bash 或 WSL 运行本脚本
# =============================================================================

# ----------------------------- 配置区域 -----------------------------
# 目标 conda 环境（与 alphagen 一致，推荐 alphagen_env；若用独立 QCM 环境则改为 alphaqcm）
TARGET_ENV="alphagen_env"

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
# 是否使用 QCM 独立体系（AlphaPoolQcm + AlphaEnvQcm + config_qcm）；留空则用默认 MseAlphaPool + AlphaEnv
USE_QCM_STACK=""
# 干跑开关：非空时仅打印命令，不启动训练（可临时用 DRY_RUN=1 ./runqcm.sh）
DRY_RUN="${DRY_RUN:-}"

# 日志目录（nohup 输出）
LOG_DIR="./AlphaQCM_data"
# ---------------------------------------------------------------------

# 切换到项目根目录（脚本所在目录的上级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 1. 环境检查
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$TARGET_ENV" ]; then
    echo "请先激活环境: conda activate $TARGET_ENV"
    echo "（与 alphagen 共用同一环境时，可选安装 QCM 增量依赖: pip install -r requirements_qcm_addon.txt）"
    exit 1
fi

# 2. 可选：设置 Python 路径，保证 alphagen/alphagen_qlib/fqf_iqn_qrdqn 可被导入
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# 3. 确保日志目录存在
mkdir -p "$LOG_DIR"

# 4. 基础校验
case "$TASK" in
    qcm_all|qcm_csi300|qcm_csi500|drl_csi300)
        ;;
    *)
        echo "未知 TASK: $TASK"
        echo "可选: qcm_all | qcm_csi300 | qcm_csi500 | drl_csi300"
        exit 1
        ;;
esac

case "$MODEL" in
    qrdqn|iqn|fqf)
        ;;
    *)
        echo "未知 MODEL: $MODEL"
        echo "可选: qrdqn | iqn | fqf"
        exit 1
        ;;
esac

# 5. 根据 TASK 选择入口与参数
case "$TASK" in
    qcm_all)
        ENTRY="train_qcm.py"
        LOG_SUBDIR="alpha_logs"
        STACK_ROLE="qcm"
        CONFIG_SOURCE="qcm_config/${MODEL}.yaml"
        EXTRA_ARGS="--model $MODEL --seed $SEED --pool $POOL --std-lam $STD_LAM"
        if [ -n "$USE_QCM_STACK" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --use-qcm-stack"
            STACK_TAG="qcmstack"
        else
            STACK_TAG="defaultstack"
        fi
        ;;
    qcm_csi300)
        ENTRY="train_qcm_csi300.py"
        LOG_SUBDIR="csi300_logs"
        STACK_ROLE="qcm"
        CONFIG_SOURCE="qcm_config/${MODEL}.yaml"
        EXTRA_ARGS="--model $MODEL --seed $SEED --pool $POOL --std-lam $STD_LAM"
        if [ -n "$USE_QCM_STACK" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --use-qcm-stack"
            STACK_TAG="qcmstack"
        else
            STACK_TAG="defaultstack"
        fi
        ;;
    qcm_csi500)
        ENTRY="train_qcm_csi500.py"
        LOG_SUBDIR="csi500_logs"
        STACK_ROLE="qcm"
        CONFIG_SOURCE="qcm_config/${MODEL}.yaml"
        EXTRA_ARGS="--model $MODEL --seed $SEED --pool $POOL --std-lam $STD_LAM"
        if [ -n "$USE_QCM_STACK" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --use-qcm-stack"
            STACK_TAG="qcmstack"
        else
            STACK_TAG="defaultstack"
        fi
        ;;
    drl_csi300)
        # DRL baseline：使用 config/*.yaml，忽略 QCM 专用变量
        ENTRY="train_drl_csi300.py"
        LOG_SUBDIR="csi300_logs"
        STACK_ROLE="baseline"
        STACK_TAG="baseline"
        CONFIG_SOURCE="config/${MODEL}.yaml"
        EXTRA_ARGS="--model $MODEL --seed $SEED --pool $POOL"
        if [ -n "$USE_QCM_STACK" ] || [ "$STD_LAM" != "1.0" ]; then
            echo "提示: TASK=drl_csi300 时将忽略 USE_QCM_STACK 和 STD_LAM"
        fi
        ;;
esac

OUTPUT_LOG="${LOG_DIR}/runqcm_${TASK}_${STACK_TAG}_${MODEL}_seed${SEED}_$(date +%Y%m%d-%H%M).txt"

CMD="python $ENTRY $EXTRA_ARGS"
echo "----------------------------------------------------"
echo "AlphaQCM 运行脚本 (TASK=$TASK, MODEL=$MODEL, STACK=$STACK_TAG)"
echo "----------------------------------------------------"
echo "环境: $CONDA_DEFAULT_ENV"
echo "脚本职责: runqcm.sh 仅负责编排 AlphaQCM 相关入口；run.sh 负责 AlphaGen 主训练"
echo "任务角色: $STACK_ROLE"
echo "配置来源: $CONFIG_SOURCE"
echo "命令: $CMD"
echo "日志: $OUTPUT_LOG"
echo "训练结果将写入: ${LOG_DIR}/${LOG_SUBDIR}/"
echo "----------------------------------------------------"

if [ -n "$DRY_RUN" ]; then
    echo "DRY_RUN 已开启：仅完成命令构造与参数检查，不启动后台训练。"
    exit 0
fi

# 6. 后台运行（如需前台直接运行，可改为: exec $CMD）
export CUDA_VISIBLE_DEVICES=$GPU_ID
nohup $CMD > "$OUTPUT_LOG" 2>&1 &
PID=$!

echo "已后台启动 (PID: $PID)"
echo "查看进度: tail -f $OUTPUT_LOG"
echo "停止任务: kill $PID"
