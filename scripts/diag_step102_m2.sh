#!/bin/bash
# 诊断脚本: 复现 step 102 崩溃, 抓堆栈. 不用 tee 覆盖, 用 >> append.
set -o pipefail   # 不用 -e/-u: -u 会因 conda activate.d 的 unbound LD_LIBRARY_PATH 退出, -e 会在 swift crash 时跳过 RC 记录

PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
cd "$PROJECT_ROOT"

source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate swift

export CUDA_HOME=/data1/meimei.wu/miniforge3/envs/verl_opd_flashnote
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export DS_BUILD_OPS=0

export TMPDIR=/dev/shm/diag_step102_m2
mkdir -p "$TMPDIR"
export HF_DATASETS_CACHE=/data4/wumeimei/meimei_tmp/hf_datasets_diag
mkdir -p "$HF_DATASETS_CACHE"
export TRITON_CACHE_DIR=/dev/shm/diag_step102_m2/.triton/cache
mkdir -p "$TRITON_CACHE_DIR"

# 调试环境变量
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export HF_HUB_DISABLE_TELEMETRY=1
ulimit -c unlimited 2>/dev/null

MODEL="/data4/wumeimei/download_models/Qwen3.5-9B"
DATA="$PROJECT_ROOT/.runtime/flashnote_summary/sft_gold_397b_final.jsonl"
OUT="$PROJECT_ROOT/outputs/diag_step102_m2"
NOW=$(date +%m%d_%H%M)
LOG="$PROJECT_ROOT/.runtime/flashnote_summary/diag_step102_${NOW}.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

# 4 卡 (4,5,6,7), 避开 GPU 0,1 (rui.ni) 和 GPU 2,3 (eval vllm)
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "[diag start] $(date)  NOW=$NOW  GPU=4,5,6,7  NPROC=4" | tee -a "$LOG"
echo "[diag config] bs=6 grad_acc=2 lr=1e-4 logging_steps=10 (与原 6 卡同, 仅 ranks 4≠6)" | tee -a "$LOG"
echo "[diag env] NCCL_DEBUG=WARN TORCH_DISTRIBUTED_DEBUG=DETAIL PYTHONFAULTHANDLER=1" | tee -a "$LOG"

# 用 >> append, 不覆盖. stdout+stderr 都进 log.
swift sft \
  --model "$MODEL" \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --dataset "$DATA" \
  --num_train_epochs 5 \
  --save_strategy steps \
  --save_steps 450 \
  --save_total_limit 2 \
  --max_length 6144 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --output_dir "$OUT" \
  --report_to tensorboard \
  >> "$LOG" 2>&1
RC=$?
echo "[diag end] $(date)  exit_code=$RC" >> "$LOG"
