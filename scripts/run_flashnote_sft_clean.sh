#!/bin/bash
# flash_note summary SFT 重训 — cleaned data (assistant 只含纯摘要)
# 原 sft_train.jsonl 全 72166 条 assistant 都把 teacher_prompt 拼前面,已清洗成 sft_train_clean.jsonl
# 参数与原 run_flashnote_sft.sh 完全一致,只改 DATA/OUT/LOG
set -eo pipefail

PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
cd "$PROJECT_ROOT"

source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate swift

export CUDA_HOME=/data1/meimei.wu/miniforge3/envs/verl_opd_flashnote
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export DS_BUILD_OPS=0
export TMPDIR=/data4/wumeimei/tmp
mkdir -p "$TMPDIR"

MODEL="/data4/wumeimei/download_models/Qwen3.5-9B"
DATA="$PROJECT_ROOT/.runtime/flashnote_summary/sft_train_clean_hr.jsonl"
OUT="$PROJECT_ROOT/outputs/flashnote_sft_clean"
LOG="$PROJECT_ROOT/.runtime/flashnote_summary/sft_train_clean.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29501

echo "[start] $(date)  swift SFT CLEAN  5epoch save376  model=$MODEL  data=$DATA"
swift sft \
  --model "$MODEL" \
  --tuner_type full \
  --dataset "$DATA" \
  --num_train_epochs 5 \
  --save_strategy steps \
  --save_steps 376 \
  --save_total_limit 12 \
  --save_only_model true \
  --max_length 6144 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.01 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 6 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --deepspeed zero2 \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"
