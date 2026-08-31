#!/usr/bin/env bash
# 并行评测 sft_ori 的 8 个 epoch checkpoint, 每个 ckpt 占 1 张 GPU
# 8 个 eval 子进程并行, 总耗时 ≈ 单个 ckpt 耗时(~25min), 而非 8x 串行
set -uo pipefail

PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
cd "$PROJECT_ROOT"

LOG="/data4/wumeimei/flash_note/RP-OPSD/.runtime/eval_sft_ori_parallel.log"
mkdir -p "$(dirname "$LOG")"

TAGS=(epoch1.0 epoch1.5 epoch2.5 epoch3.0 epoch3.5 epoch4.0 epoch4.5 epoch5.0)
BASE_PORT=8005
BASE_GPU=0

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; echo "[$(date '+%F %T')] $*"; }

log "===== 并行 eval 启动: ${#TAGS[@]} 个 ckpt, 每个占 1 张 GPU ====="

PIDS=()
for i in "${!TAGS[@]}"; do
  TAG="${TAGS[$i]}"
  GPU=$((BASE_GPU + i))
  PORT=$((BASE_PORT + i))
  log "启动 TAG=$TAG  GPU=$GPU  PORT=$PORT"
  EXP=sft_ori TAGS="$TAG" GPU=$GPU PORT=$PORT CONCURRENCY=64 \
    bash scripts/eval_flashnote_sft_ckpts.sh >> "$LOG" 2>&1 &
  PIDS+=($!)
done

log "全部 ${#TAGS[@]} 个 eval 子进程已启动, 等待完成..."
log "PIDS: ${PIDS[*]}"

# 等全部完成
FAILED=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    log "TAG=${TAGS[$i]} eval 失败 (pid=${PIDS[$i]})"
    FAILED=$((FAILED+1))
  else
    log "TAG=${TAGS[$i]} eval 完成 (pid=${PIDS[$i]})"
  fi
done

log "===== 全部完成: $(( ${#TAGS[@]} - FAILED ))/${#TAGS[@]} 成功, $FAILED 失败 ====="
