#!/bin/bash
# flash_note summary SFT —— gen_gold_397b 数据集, m2 重头新训
# 机器2 (rf-nlp-h20-2, 10.162.52.29) 启动
# 与 m3 sft_gold_397b 唯一区别: 输出目录带 _m2 标识, 从头训不 resume
set -eo pipefail

PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
cd "$PROJECT_ROOT"

source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate swift

export CUDA_HOME=/data1/meimei.wu/miniforge3/envs/verl_opd_flashnote
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export DS_BUILD_OPS=0
# TMPDIR 必须放 tmpfs: datasets 的 multiprocess.Manager 要 bind socket,
# /data4 xfs 上 socket bind 报 EPERM (Operation not permitted), 必须用 /dev/shm
export TMPDIR=/dev/shm/sft_gold_397b_m2
mkdir -p "$TMPDIR"
# HF datasets arrow cache 走 /data4 (常规文件 OK, 只有 socket bind 才挂)
export HF_DATASETS_CACHE=/data4/wumeimei/meimei_tmp/hf_datasets
mkdir -p "$HF_DATASETS_CACHE"
# triton cache 放 tmpfs, 清旧 cache 防 gcc SIGABRT
export TRITON_CACHE_DIR=/dev/shm/sft_gold_397b_m2/.triton/cache
rm -rf "$TRITON_CACHE_DIR" 2>/dev/null
mkdir -p "$TRITON_CACHE_DIR"

MODEL="/data4/wumeimei/download_models/Qwen3.5-9B"
DATA="$PROJECT_ROOT/.runtime/flashnote_summary/sft_gold_397b_final.jsonl"
OUT="$PROJECT_ROOT/outputs/flashnote_sft_gold_397b_m2"
LOG="$PROJECT_ROOT/.runtime/flashnote_summary/sft_train_gold_397b_m2.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "[start] $(date)  sft_gold_397b_m2 (fresh)  model=$MODEL  data=$DATA"
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
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --deepspeed zero2 \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"
