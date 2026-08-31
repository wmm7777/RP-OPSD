#!/usr/bin/env bash
# 等 m2 sft_ori 训练结束后自动起 8 个 epoch (1.0/1.5/2.5/3.0/3.5/4.0/4.5/5.0) 的推理+评测
# 训练进程消失 + checkpoint-3600 落盘 = 训完信号
# 用法: nohup bash wait_and_eval_sft_ori.sh > wait_and_eval_sft_ori.log 2>&1 &
set -uo pipefail

CKPT_DIR="/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_sft_ori/v3-20260830-211922"
LOG="/data4/wumeimei/flash_note/RP-OPSD/.runtime/wait_and_eval_sft_ori.log"
TAGS="epoch1.0 epoch1.5 epoch2.5 epoch3.0 epoch3.5 epoch4.0 epoch4.5 epoch5.0"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; echo "[$(date '+%F %T')] $*"; }

mkdir -p "$(dirname "$LOG")"
log "===== wait_and_eval_sft_ori 启动 ====="
log "等待训练进程结束 + checkpoint-3600 落盘..."
log "目标 TAGS: $TAGS"

while true; do
  # 训练进程是否还在
  if pgrep -f 'swift.*sft_ori' >/dev/null 2>&1; then
    log "训练仍在进行, 等待 300s..."
    sleep 300
    continue
  fi

  # 进程已结束, 等 30s 让 checkpoint 落盘
  log "训练进程已结束, 等 30s 让 checkpoint 完全落盘..."
  sleep 30

  # 确认 checkpoint-3600 存在 (epoch5.0 = 训练最后一个保存的 ckpt)
  if [ -d "$CKPT_DIR/checkpoint-3600" ]; then
    log "checkpoint-3600 存在, 训练确认完成"
    break
  else
    log "checkpoint-3600 不存在! 可能训练异常中断. 列出现有 ckpt:"
    ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort >> "$LOG"
    log "仍继续起 eval (已存在的 ckpt 会跑, 不存在的会被 eval 脚本跳过)"
    break
  fi
done

# 起评测
log "===== 启动 eval: EXP=sft_ori TAGS=$TAGS GPU=0 PORT=8005 CONCURRENCY=64 ====="
cd /data4/wumeimei/flash_note/RP-OPSD
EXP=sft_ori TAGS="$TAGS" GPU=0 PORT=8005 CONCURRENCY=64 \
  bash scripts/eval_flashnote_sft_ckpts.sh >> "$LOG" 2>&1
log "===== eval 结束 ====="
