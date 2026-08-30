#!/bin/bash
# flash_note summary SFT 对照实验启动脚本
# 全参数 SFT，gold summary 监督，2 epoch，每 0.2 epoch（≈150 步）存一次
# 对照 RP-OPSD：同 9B / 同 72k 采样 / 同半分辨率训练图 / 同原图评测，只差训练范式
# max_length 6144 与 RP-OPSD 6k 口径对齐（数据实测 prompt+response ~1945，6144 远够）
set -eo pipefail

PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
cd "$PROJECT_ROOT"

# swift 环境（多模态 SFT；SFT 不依赖 flash_attn/causal_conv1d/vllm rollout）
source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate swift

# swift 环境无完整 CUDA toolkit（仅 pip nvidia-cuda-runtime），借 verl_opd_flashnote 的 CUDA 12.9（含 nvcc/headers）
# 否则 deepspeed import 时 installed_cuda_version() 抛 MissingCUDAException
export CUDA_HOME=/data1/meimei.wu/miniforge3/envs/verl_opd_flashnote
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export DS_BUILD_OPS=0

MODEL="/data4/wumeimei/download_models/Qwen3.5-9B"
DATA="$PROJECT_ROOT/.runtime/flashnote_summary/sft_train.jsonl"
OUT="$PROJECT_ROOT/outputs/flashnote_sft"
LOG="$PROJECT_ROOT/.runtime/flashnote_summary/sft_train.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

# effective batch = per_device(2) × grad_accum(6) × 8 卡 = 96，与 RP-OPSD batch96 对齐
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "[start] $(date)  swift SFT  5epoch save150  model=$MODEL  data=$DATA"
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
