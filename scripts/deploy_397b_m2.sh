#!/bin/bash
# 机器2 部署 Qwen3.5-397B-A17B-FP8（多模态 MoE，线性+full attention 混合，FP8）
# 环境 verl_opd_flashnote（含 causal_conv1d + flashinfer，linear_attention 必需；swift 环境缺 causal_conv1d 不可用）
set -eo pipefail   # 不用 -u：source conda.sh 会触发 unbound 变量退出

source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate verl_opd_flashnote
# 系统 libstdc++ 旧（只到 CXXABI_1.3.13，缺 1.3.15），前置 env 新版 libstdc++.so.6.0.35（含 1.3.15/1.3.17）
export LD_LIBRARY_PATH=/data1/meimei.wu/miniforge3/envs/verl_opd_flashnote/lib:${LD_LIBRARY_PATH:-}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TMPDIR=/tmp/deploy397b_m2            # 坑1：跨机勿用 sshfs 路径
mkdir -p "$TMPDIR"
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1

# 坑3：triton 单进程预热
python -c "import triton; print('triton:', triton.runtime.driver.active.get_current_target())" || echo "[warn] triton preheat fail"

MODEL=/data4/wumeimei/download_models/Qwen3.5-397B-A17B-FP8
LOG=/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary/deploy_397b_m2.log
mkdir -p "$(dirname "$LOG")"

echo "[start] $(date)  deploy 397B-A17B-FP8  vllm TP=8  port 8000  host=$(hostname)"
# FP8 量化由 config.json 声明（quant_method=fp8），vllm 自动识别，不显式 --quantization 避免冲突
# 线性注意力 conv1d / MoE 512 专家 由 vllm + causal_conv1d 内核处理
vllm serve "$MODEL" \
  --served-model-name qwen397b \
  --tensor-parallel-size 8 \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 9216 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  2>&1 | tee "$LOG"
