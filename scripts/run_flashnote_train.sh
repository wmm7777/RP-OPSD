#!/bin/bash
# flash_note summary RP-OPSD 自蒸馏训练启动脚本（2 epoch，每 0.2 epoch 存一次）
set -eo pipefail   # 不用 -u：source conda.sh 会触发 unbound 变量退出

PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
cd "$PROJECT_ROOT"

# 激活论文一致环境
source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate verl_opd_flashnote

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export WORLD_SIZE=1                              # run_rp_opsd.sh 依赖（trainer.nnodes=$WORLD_SIZE），单机8卡=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 跨机三坑防护（skill cross-machine-train-deploy；自蒸馏无教师，坑4/6 不适用）
export TMPDIR=/tmp/rp_opsd_flashnote             # 坑1：勿用 sshfs 路径，改本地 /tmp
mkdir -p "$TMPDIR"
export MASTER_PORT=29551                          # 坑2：换空闲端口防 rendezvous EADDRINUSE
export RP_OPSD_LAUNCHER_CHECKPOINT_DIR="$PROJECT_ROOT/outputs/flashnote_rp_opsd/checkpoints"
export RP_OPSD_LAUNCHER_ROLLOUT_DIR="$PROJECT_ROOT/outputs/flashnote_rp_opsd/rollouts"
mkdir -p "$RP_OPSD_LAUNCHER_CHECKPOINT_DIR" "$RP_OPSD_LAUNCHER_ROLLOUT_DIR"

LOG="$PROJECT_ROOT/.runtime/flashnote_summary/train.log"
mkdir -p "$(dirname "$LOG")"

echo "[start] $(date)  verl_opd_flashnote  2epoch save_freq=150  host=$(hostname)"
# 坑3：triton cache 单进程预热（目标机 HOME 本地、~/.triton 空，多 worker 冷编译竞态崩）
python -c "import triton; print('triton preheat:', triton.runtime.driver.active.get_current_target())" || echo "[warn] triton preheat failed, continue"
bash scripts/run_rp_opsd.sh 2>&1 | tee "$LOG"
