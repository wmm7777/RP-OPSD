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
export TMPDIR=/data4/wumeimei/tmp
mkdir -p "$TMPDIR"

MODEL="/data4/wumeimei/download_models/Qwen3.5-9B"
DATA="$PROJECT_ROOT/.runtime/flashnote_summary/sft_gold_397b_final.jsonl"
OUT="$PROJECT_ROOT/outputs/flashnote_sft_gold_397b"
LOG="$PROJECT_ROOT/.runtime/flashnote_summary/sft_train_gold_397b.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "[start] $(date)  gen_gold_397b SFT  model=$MODEL  data=$DATA"
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
