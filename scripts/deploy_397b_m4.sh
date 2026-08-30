set -eo pipefail
source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate verl_opd_flashnote
export LD_LIBRARY_PATH=/data1/meimei.wu/miniforge3/envs/verl_opd_flashnote/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TMPDIR=/dev/shm/vllm_m4
mkdir -p "$TMPDIR"
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
# 2026-08-30 根因定位：m4 flashinfer cache (8-29 编译) 用错 GCC(系统 c++ 11.4 非 conda 13.4)，GDN TMA descriptor 运行时损坏 CUDA context → invalid resource handle。8-30 清 cache 重编译验证(走本脚本 conda activate 正确 PATH)。若重编译后 .so md5 = m3 的 71fc63bc... 且首前向不崩则根因确认。
# 不设 SKIP_GDN_WARMUP：profile 阶段的 _warmup_prefill_kernels 给 GDN prefill 内核预编译/autotune，跳过会首次前向报错。

python -c "import triton; print('triton:', triton.runtime.driver.active.get_current_target())" || echo "[warn] triton preheat fail"

MODEL=/data4/wumeimei/download_models/Qwen3.5-397B-A17B-FP8
LOG=/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary/deploy_397b_m4.log
mkdir -p "$(dirname "$LOG")"
echo "[start] $(date) deploy 397B on machine4 (local) TP=8 port 8002"
vllm serve "$MODEL" \
  --served-model-name qwen397b \
  --tensor-parallel-size 8 \
  --port 8002 \
  --host 0.0.0.0 \
  --max-model-len 9216 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 96 \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  2>&1 | tee "$LOG"
