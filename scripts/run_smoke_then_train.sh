#!/bin/bash
# flash_note RP-OPSD triton cache 预热脚本（1 step）跑完自动续 2 epoch 完整训练
# 配置严格按 docs/flash_note_RP_OPSD.md §5.4.1 实测启动命令
set -eo pipefail

PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
cd "$PROJECT_ROOT"

# 环境（§5.4.1 完整 env vars，关键：PYTORCH_ALLOC_CONF=expandable_segments:True 防 triton do_bench OOM）
export PATH=/data1/meimei.wu/miniforge3/envs/verl_opd_flashnote/bin:$PATH
export CUDA_HOME=/data1/meimei.wu/miniforge3/envs/verl_opd_flashnote
export TMPDIR=/data4/wumeimei/meimei_tmp
export _RAY_TMPDIR=/data4/wumeimei/meimei_tmp
export RAY_TMPDIR=/data4/wumeimei/meimei_tmp
mkdir -p "$TMPDIR"
export WORLD_SIZE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=
export MASTER_PORT=29551
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# triton cache 持久化复用（预热一次后跨 run 复用，跳过 smoke 阶段）
export TRITON_CACHE_DIR=/data4/wumeimei/.triton/rp_opsd_flashnote
mkdir -p "$TRITON_CACHE_DIR"

source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh
conda activate verl_opd_flashnote

# === 直接 2 epoch 完整训练（triton cache 已预热，跨 run 复用）===
TRAIN_OUT="$PROJECT_ROOT/outputs/flashnote_train_v2"
export RP_OPSD_LAUNCHER_CHECKPOINT_DIR="$TRAIN_OUT/checkpoints"
export RP_OPSD_LAUNCHER_ROLLOUT_DIR="$TRAIN_OUT/rollouts"
mkdir -p "$RP_OPSD_LAUNCHER_CHECKPOINT_DIR" "$RP_OPSD_LAUNCHER_ROLLOUT_DIR" "$TRAIN_OUT/logs"

TRAIN_LOG="$TRAIN_OUT/logs/train.log"
echo "[train start] $(date)  2 epoch full training  host=$(hostname)"

bash scripts/run_rp_opsd.bak.sh \
  'data.train_files=["/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary/train.parquet"]' \
  'data.val_files=[]' \
  'data.image_key=images' \
  'actor_rollout_ref.model.path=/data4/wumeimei/download_models/Qwen3.5-9B' \
  'critic.model.path=/data4/wumeimei/download_models/Qwen3.5-9B' \
  'actor_rollout_ref.actor.self_distillation.teacher_image_key=teacher_images' \
  'actor_rollout_ref.actor.self_distillation.alpha=1.0' \
  '+actor_rollout_ref.model.override_config.attn_implementation=sdpa' \
  '+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton' \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model]' \
  'trainer.total_epochs=2' \
  'trainer.save_freq=150' \
  'trainer.test_freq=-1' \
  'trainer.max_actor_ckpt_to_keep=10' \
  "trainer.default_local_dir=$TRAIN_OUT/checkpoints" \
  "trainer.rollout_data_dir=$TRAIN_OUT/rollouts" \
  2>&1 | tee "$TRAIN_LOG"
TRAIN_RC=${PIPESTATUS[0]}
echo "[train done] $(date)  rc=$TRAIN_RC"
exit $TRAIN_RC
