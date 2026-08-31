#!/bin/bash
# flash_note summary SFT —— gen_gold_397b 数据集, LoRA rank=64, m2
# 机器2 (rf-nlp-h20-2, 10.162.52.29) 启动
# 与 full SFT 区别: LoRA r=64, batch=2, grad_acc=6, 用 6 卡 (2-7), LR 1e-4
# 2026-08-31 4 次 step 102 OOM, 改 batch 6->2 + expandable_segments (flash_attn 未装在 swift env, 默认 sdpa 已走 torch native flash 后端)
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
# /data4 xfs 上 socket bind 报 EPERM, 必须用 /dev/shm
export TMPDIR=/dev/shm/sft_gold_397b_lora_r64_m2
mkdir -p "$TMPDIR"
# HF datasets arrow cache 走 /data4 (常规文件 OK)
export HF_DATASETS_CACHE=/data4/wumeimei/meimei_tmp/hf_datasets_lora_r64
mkdir -p "$HF_DATASETS_CACHE"
# triton cache 放 tmpfs, 清旧 cache 防 gcc SIGABRT
export TRITON_CACHE_DIR=/dev/shm/sft_gold_397b_lora_r64_m2/.triton/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf "$TRITON_CACHE_DIR" 2>/dev/null
mkdir -p "$TRITON_CACHE_DIR"
# 降低 PyTorch caching allocator fragmentation, step 102 OOM 修复之一

MODEL="/data4/wumeimei/download_models/Qwen3.5-9B"
DATA="$PROJECT_ROOT/.runtime/flashnote_summary/sft_gold_397b_final.jsonl"
OUT="$PROJECT_ROOT/outputs/flashnote_sft_gold_397b_lora_r64_m2"
LOG="$PROJECT_ROOT/.runtime/flashnote_summary/sft_train_gold_397b_lora_r64_m2.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

# 只用 6 卡 (2,3,4,5,6,7), 留 0,1 给潜在的其他任务
export NPROC_PER_NODE=6
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

echo "[start] $(date)  sft_gold_397b_lora_r64_m2  model=$MODEL  data=$DATA"
echo "[config] rank=64 alpha=128 batch=2 grad_acc=6 lr=1e-4 epochs=5 save_steps=450 expandable_segments=on"
swift sft \
  --model "$MODEL" \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --dataset "$DATA" \
  --num_train_epochs 5 \
  --save_strategy steps \
  --save_steps 450 \
  --save_total_limit 12 \
  --max_length 6144 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 6 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --output_dir "$OUT" \
  "$@" \
  2>&1 | tee "$LOG"
