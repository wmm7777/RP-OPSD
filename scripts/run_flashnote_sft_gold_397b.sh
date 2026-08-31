#!/bin/bash
# flash_note summary SFT —— gen_gold_397b 数据集（397B 教师生成 gold）
# 机器3（rf-nlp-h20-3，10.162.52.31）启动
# 与 ori 唯一区别：gold 来源 = 397B 看原图生成（非训练集自带）
# prompt 用 parquet 4 语种全文翻译版（与 ori 完全一致），隔离 gold 变量
set -eo pipefail

PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
cd "$PROJECT_ROOT"

source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate swift

export CUDA_HOME=/data1/meimei.wu/miniforge3/envs/verl_opd_flashnote
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export DS_BUILD_OPS=0
export TMPDIR=/data1/meimei.wu/tmp
mkdir -p "$TMPDIR"
# triton cache 放本地盘, 避免旧 cache 损坏导致 gcc SIGABRT
export TRITON_CACHE_DIR=/data1/meimei.wu/.triton/cache
rm -rf "$TRITON_CACHE_DIR" 2>/dev/null
mkdir -p "$TRITON_CACHE_DIR"

MODEL="/data4/wumeimei/download_models/Qwen3.5-9B"
DATA="$PROJECT_ROOT/.runtime/flashnote_summary/sft_gold_397b_final.jsonl"
OUT="$PROJECT_ROOT/outputs/flashnote_sft_gold_397b"
LOG="$PROJECT_ROOT/.runtime/flashnote_summary/sft_train_gold_397b.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

# resume: 找最新 run 目录下最新的 checkpoint
RESUME_CKPT=""
LATEST_RUN=$(find "$OUT" -maxdepth 1 -name 'v*-*' -type d 2>/dev/null | xargs -I{} stat -c '%Y {}' {} 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [[ -n "$LATEST_RUN" ]]; then
  LATEST_CKPT=$(find "$LATEST_RUN" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1)
  if [[ -n "$LATEST_CKPT" ]]; then
    RESUME_CKPT="$LATEST_RUN/checkpoint-$LATEST_CKPT"
  fi
fi

export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "[start] $(date)  gen_gold_397b SFT  model=$MODEL  data=$DATA"
if [[ -n "$RESUME_CKPT" && -d "$RESUME_CKPT" ]]; then
  echo "[resume] from $RESUME_CKPT"
  swift sft \
    --model "$MODEL" \
    --tuner_type full \
    --dataset "$DATA" \
    --num_train_epochs 5 \
    --save_strategy steps \
    --save_steps 376 \
    --save_total_limit 12 \
    --max_length 6144 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.01 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 6 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --deepspeed zero2 \
    --output_dir "$OUT" \
    --resume_from_checkpoint "$RESUME_CKPT" \
    2>&1 | tee "$LOG"
else
  echo "[resume] no checkpoint found, fresh start"
  swift sft \
    --model "$MODEL" \
    --tuner_type full \
    --dataset "$DATA" \
    --num_train_epochs 5 \
    --save_strategy steps \
    --save_steps 376 \
    --save_total_limit 12 \
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
fi
