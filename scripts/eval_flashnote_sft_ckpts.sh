#!/usr/bin/env bash
# flash_note summary MOS 评测 —— 支持三种实验 ori / sft / rp_opsd
#
# 每个实验的 checkpoint 列表按 TAG 遍历, 推理产物和 MOS report 都带实验前缀, 不互相覆盖:
#   推理:    /data4/wumeimei/flash_note/infer/infer_res_${EXP}/flashnote_${lang}_${EXP}_summary_9b_${TAG}.json
#   MOS:     eval_results/eval_report/${EXP}_summary_9b_${TAG}/<lang>/{title,summary}_mos_results.json
#
# 用法:
#   EXP=sft bash eval_flashnote_sft_ckpts.sh                 # SFT 各 epoch (默认 1.0~3.5)
#   EXP=sft EPOCHS="1.0 2.0" GPU=0 bash eval_flashnote_sft_ckpts.sh
#   EXP=ori bash eval_flashnote_sft_ckpts.sh                 # 原始 9B base (单次)
#   EXP=rp_opsd TAGS="step55 step275" bash eval_flashnote_sft_ckpts.sh  # verl 训练 merged ckpt
#
# 前置依赖:
#   - gemini API key (已写在 gemini_model.py 代码里)
#   - swift conda 环境 (vllm 0.19+ 多模态)
#   - test_data/excel/data_image{,s}_<lang>.xlsx + test_data/<lang>_image/*.jpg
#   - TMPDIR 必须设到 /data4 (根盘满会卡死 vllm)
set -uo pipefail

# ===== 实验配置 =====
EXP="${EXP:-sft}"   # ori | sft | rp_opsd

# 各实验的 checkpoint 根目录
declare -A CKPT_ROOTS=(
  [sft]="/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_sft/v0-20260829-230600"
  [ori]="/data4/wumeimei/download_models/Qwen3.5-9B"
  [rp_opsd]="/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_rp_opsd/merged"
)

# sft: TAG=epoch1.0 → checkpoint-752 (1 epoch = 752 step, save_steps=376)
declare -A SFT_EPOCH_STEP=(
  [epoch1.0]=752  [epoch1.5]=1128 [epoch2.0]=1504 [epoch2.5]=1880
  [epoch3.0]=2256 [epoch3.5]=2632 [epoch4.0]=3008 [epoch4.5]=3384 [epoch5.0]=3760
)

# 默认 TAG 列表 (按 EXP 选)
case "$EXP" in
  sft)     TAGS="${TAGS:-epoch1.0 epoch1.5 epoch2.0 epoch2.5 epoch3.0 epoch3.5}" ;;
  ori)     TAGS="${TAGS:-base}" ;;
  rp_opsd) TAGS="${TAGS:-step55}" ;;   # verl 训练完 merge 后的 step, 用户按需改
  *) echo "[error] EXP=$EXP 不支持 (ori/sft/rp_opsd)"; exit 1 ;;
esac

# checkpoint 路径解析: 给定 EXP + TAG 返回实际路径
resolve_ckpt() {
  local exp=$1 tag=$2
  local root="${CKPT_ROOTS[$exp]}"
  case "$exp" in
    sft)
      local step="${SFT_EPOCH_STEP[$tag]:-}"
      [[ -z "$step" ]] && { echo ""; return; }
      echo "$root/checkpoint-$step" ;;
    ori)
      echo "$root" ;;   # 整个目录就是模型, 无 checkpoint 子目录
    rp_opsd)
      echo "$root/$tag" ;;   # merged/step55 等
    *) echo "" ;;
  esac
}

# ===== 运行配置 =====
GPU="${GPU:-0}"
PORT="${PORT:-8005}"
VLLM_MODEL="flashnote-eval-${EXP}"
LANGS="${LANGS:-en fr ru zh}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate swift
ENV_PY="$(python -c 'import sys;print(sys.executable)')"

export CUDA_HOME="/data1/meimei.wu/miniforge3/envs/swift/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR="${TMPDIR:-/dev/shm}"; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export LD_LIBRARY_PATH="/data1/meimei.wu/miniforge3/envs/swift/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1

INFER_ROOT="/data4/wumeimei/flash_note/infer"
EVAL_ROOT="/data4/wumeimei/flash_note/auto_eval/evaluators"
INFER_DIR="$INFER_ROOT/infer_res_${EXP}"
LOG_DIR="$INFER_ROOT/logs"
mkdir -p "$INFER_DIR" "$LOG_DIR"

log()  { printf '[%s-eval] %s\n' "$EXP" "$*"; }
die()  { printf '[%s-eval] ERROR: %s\n' "$EXP" "$*" >&2; exit 1; }

start_vllm() {
  local ckpt=$1 port=$2
  local logf="$LOG_DIR/vllm_${EXP}_${port}_$(date +%m%d_%H%M).log"
  log "启动 vLLM  ckpt=$ckpt  GPU=$GPU  port=$port"
  log "  log: $logf"
  CUDA_VISIBLE_DEVICES="$GPU" nohup setsid "$ENV_PY" -m vllm.entrypoints.openai.api_server \
    --model "$ckpt" \
    --served-model-name "$VLLM_MODEL" \
    --host 0.0.0.0 --port "$port" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --gdn-prefill-backend triton \
    --trust-remote-code \
    > "$logf" 2>&1 &
  VLLM_PID=$!
  log "vLLM PID=$VLLM_PID  等待就绪 (最多 8 分钟)"
  for i in $(seq 1 96); do
    sleep 5
    if curl -s "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      log "vLLM 就绪 (${i}x5s)"; return 0
    fi
    kill -0 "$VLLM_PID" 2>/dev/null || { log "vLLM 进程退出"; tail -30 "$logf"; return 1; }
  done
  log "vLLM 8 分钟未就绪"; tail -30 "$logf"; return 1
}

stop_vllm() {
  local port=$1 pid
  pid=$(ss -ltnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | sort -u | head -1)
  [[ -z "$pid" ]] && { log "stop: 端口 $port 无进程"; return 0; }
  log "stop vLLM  pid=$pid"
  kill "$pid" 2>/dev/null || true
  sleep 3
  kill -0 "$pid" 2>/dev/null && { kill -9 "$pid" 2>/dev/null || true; }
}

run_infer() {
  local label=$1
  VLLM_API_URL="http://localhost:${PORT}/v1/chat/completions" \
  VLLM_MODEL="$VLLM_MODEL" \
  MAX_TOKENS=1024 \
  "$ENV_PY" "$INFER_ROOT/test_image_ts_qwen35_9b.py" $LANGS --label "$label"
}

run_mos() {
  local label=$1 lang=$2
  local infer_json="$INFER_DIR/flashnote_${lang}_${label}.json"
  [[ -f "$infer_json" ]] || { log "推理产物不存在: $infer_json  跳过 MOS"; return 1; }
  ( cd "$EVAL_ROOT" && "$ENV_PY" run_multilang_eval.py \
      --lang "$lang" \
      --input "$infer_json" \
      --model-label "$label" \
      --flash-title-key qwen35_title \
      --flash-summary-key qwen35_summary \
      --modes flash_summary_mos --concurrency 8 )
}

# ===== 主循环 =====
log "=========================================="
log " EXP=$EXP  TAGS=$TAGS"
log " GPU=$GPU  PORT=$PORT  LANGS=$LANGS"
log " 推理产物目录: $INFER_DIR"
log "=========================================="

for TAG in $TAGS; do
  LABEL="${EXP}_summary_9b_${TAG}"
  CKPT="$(resolve_ckpt "$EXP" "$TAG")"
  log "========== $EXP  TAG=$TAG  label=$LABEL  ckpt=$CKPT =========="
  [[ -z "$CKPT" ]] && { log "TAG=$TAG 无法解析 checkpoint 路径, 跳过"; continue; }
  [[ -d "$CKPT" ]] || { log "checkpoint 不存在: $CKPT  跳过 (训练可能还没到这步)"; continue; }

  if ! start_vllm "$CKPT" "$PORT"; then
    log "TAG=$TAG vLLM 启动失败, 跳过"
    stop_vllm "$PORT"
    continue
  fi

  if ! run_infer "$LABEL"; then
    log "TAG=$TAG 推理失败, 停 vLLM 进下一个"
    stop_vllm "$PORT"
    continue
  fi
  stop_vllm "$PORT"

  # 推理产物在 INFER_ROOT/infer_res_<MMDD>/ 下, 拷到统一 INFER_DIR
  DAILY_DIR="$INFER_ROOT/infer_res_$(date +%m%d)"
  for L in $LANGS; do
    src="$DAILY_DIR/flashnote_${L}_${LABEL}.json"
    [[ -f "$src" ]] && cp -f "$src" "$INFER_DIR/"
  done

  for L in $LANGS; do
    run_mos "$LABEL" "$L" || log "MOS 失败 $EXP/$TAG lang=$L"
  done
done

log "=========================================="
log "全部 $EXP 评测完成"
log "MOS report: $EVAL_ROOT/eval_results/eval_report/${EXP}_summary_9b_*/<lang>/"
log "汇总:"
log "  $ENV_PY $(dirname "$0")/collect_sft_eval_results.py --exp $EXP --eval-root $EVAL_ROOT/eval_results --tags '$TAGS' --langs '$LANGS'"
log "=========================================="
