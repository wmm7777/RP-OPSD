#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: train.sh --model-path DIR --output-dir DIR [options]

Options:
  --release-dir DIR       Portable Dataset4.0 release.
  --asset-root DIR        External root containing assets/student and assets/teacher.
  --env-dir DIR           Prepared Python 3.12 environment.
  --work-dir DIR          Local runtime data and rollout directory.
  --steps N               Training steps (default: 55).
  --smoke                 Run one step and validate RP-OPSD diagnostics.
  --reset-local-ray       Stop a pre-existing local Ray runtime before launch.

This command never deletes or overwrites an existing output directory.
Use a dedicated 8-GPU worker.
EOF
}

RUNTIME_ROOT="$(default_runtime_root)"
MODEL_PATH=""
OUTPUT_DIR=""
RELEASE_DIR="$(default_release_dir)"
ASSET_ROOT=""
ENV_DIR="${RUNTIME_ROOT}/venv"
WORK_DIR=""
TOTAL_STEPS="${RP_OPSD_TOTAL_STEPS}"
SMOKE=0
RESET_LOCAL_RAY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --release-dir) RELEASE_DIR=$2; shift 2 ;;
    --asset-root) ASSET_ROOT=$2; shift 2 ;;
    --env-dir) ENV_DIR=$2; shift 2 ;;
    --work-dir) WORK_DIR=$2; shift 2 ;;
    --steps) TOTAL_STEPS=$2; shift 2 ;;
    --smoke) SMOKE=1; TOTAL_STEPS=1; shift ;;
    --reset-local-ray) RESET_LOCAL_RAY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown train argument: $1" ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || die "--model-path is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
[[ "${TOTAL_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive integer"

MODEL_PATH="$(absolute_path "${MODEL_PATH}")"
OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR}")"
RELEASE_DIR="$(absolute_path "${RELEASE_DIR}")"
if [[ -n "${ASSET_ROOT}" ]]; then
  ASSET_ROOT="$(absolute_path "${ASSET_ROOT}")"
fi
ENV_DIR="$(absolute_path "${ENV_DIR}")"
if [[ -z "${WORK_DIR}" ]]; then
  RUN_TAG="$(basename "${OUTPUT_DIR}")"
  WORK_DIR="${TMPDIR:-/tmp}/rp-opsd-${RUN_TAG}"
fi
WORK_DIR="$(absolute_path "${WORK_DIR}")"

require_dir "${MODEL_PATH}"
require_dir "${RELEASE_DIR}"
require_file "${RELEASE_DIR}/train.parquet"
if [[ -n "${ASSET_ROOT}" ]]; then
  require_dir "${ASSET_ROOT}/assets/student"
  require_dir "${ASSET_ROOT}/assets/teacher"
elif [[ ! -f "${RELEASE_DIR}/dataset4_0_assets.tar.zst" ]]; then
  die "images are not bundled; pass --asset-root or use a locally generated release archive"
fi
require_empty_output "${OUTPUT_DIR}"

require_file "${ENV_DIR}/bin/python"

mkdir -p "${OUTPUT_DIR}/logs" "${WORK_DIR}"
DATA_DIR="${WORK_DIR}/data"
ROLLOUT_DIR="${WORK_DIR}/rollouts"
TRAIN_LOG="${OUTPUT_DIR}/logs/train.log"
VERIFY_LOG="${OUTPUT_DIR}/logs/verify.json"

materialize_args=(
  --release-dir "${RELEASE_DIR}"
  --output-dir "${DATA_DIR}"
)
[[ -n "${ASSET_ROOT}" ]] && materialize_args+=(--asset-root "${ASSET_ROOT}")
RUNTIME_PARQUET="$("${ENV_DIR}/bin/python" "${SCRIPT_DIR}/materialize_data.py" "${materialize_args[@]}")"

"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify.py" \
  --model-path "${MODEL_PATH}" \
  --data-parquet "${RUNTIME_PARQUET}" \
  --data-root "${DATA_DIR}" \
  --check-source \
  --check-runtime \
  --require-8-gpus \
  --output-json "${VERIFY_LOG}"

if [[ ${RESET_LOCAL_RAY} -eq 1 ]]; then
  log "stopping the local Ray runtime by explicit request"
  "${ENV_DIR}/bin/ray" stop --force || true
  unset RAY_ADDRESS
fi

source "${ENV_DIR}/bin/activate"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export PYTHONPATH="${RP_OPSD_PACKAGE_ROOT}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export RP_OPSD_LAUNCHER_CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
export RP_OPSD_LAUNCHER_ROLLOUT_DIR="${ROLLOUT_DIR}"
unset VLLM_ATTENTION_BACKEND

EXPERIMENT_NAME="rp_opsd_qwen35_9b_dataset4_bs96_$(
  [[ ${SMOKE} -eq 1 ]] && printf 'smoke1' || printf '55step'
)"
PROJECT_NAME="RP-OPSD-Reproduction"

log "model=${MODEL_PATH}"
log "data=${RUNTIME_PARQUET}"
log "output=${OUTPUT_DIR}"
log "steps=${TOTAL_STEPS} n=8 batch=96 lr=2e-6 warmup=10"
log "objective=teacher-selected Top-100 bias-corrected reverse KL"
log "student=physical half resolution teacher=original resolution EMA=0.05"

cd "${RP_OPSD_PACKAGE_ROOT}"
set +e
bash scripts/run_rp_opsd.bak.sh \
  "data.train_files=[\"${RUNTIME_PARQUET}\"]" \
  "data.val_files=[]" \
  "data.image_key=images" \
  "data.train_batch_size=${RP_OPSD_TRAIN_BATCH_SIZE}" \
  "data.max_prompt_length=${RP_OPSD_MAX_PROMPT_LENGTH}" \
  "data.max_response_length=${RP_OPSD_MAX_RESPONSE_LENGTH}" \
  "data.dataloader_num_workers=8" \
  "actor_rollout_ref.model.path=${MODEL_PATH}" \
  "critic.model.path=${MODEL_PATH}" \
  "actor_rollout_ref.actor.optim.lr=${RP_OPSD_LEARNING_RATE}" \
  "actor_rollout_ref.actor.optim.lr_warmup_steps=${RP_OPSD_WARMUP_STEPS}" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=${RP_OPSD_PPO_MINI_BATCH_SIZE}" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${RP_OPSD_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  "actor_rollout_ref.actor.use_dynamic_bsz=False" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${RP_OPSD_PPO_MAX_TOKEN_LEN_PER_GPU}" \
  "actor_rollout_ref.actor.policy_loss.loss_mode=vopd" \
  "actor_rollout_ref.actor.self_distillation.full_logit_distillation=True" \
  "actor_rollout_ref.actor.self_distillation.distillation_topk=${RP_OPSD_TOPK}" \
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=False" \
  "actor_rollout_ref.actor.self_distillation.distillation_objective=mopd_topk_reverse_kl" \
  "actor_rollout_ref.actor.self_distillation.distillation_topk_source=teacher" \
  "actor_rollout_ref.actor.self_distillation.alpha=${RP_OPSD_ALPHA}" \
  "actor_rollout_ref.actor.self_distillation.teacher_model_source=legacy" \
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema" \
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=${RP_OPSD_TEACHER_UPDATE_RATE}" \
  "actor_rollout_ref.actor.self_distillation.teacher_always_on=True" \
  "actor_rollout_ref.actor.self_distillation.teacher_image_key=teacher_images" \
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True" \
  "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False" \
  "actor_rollout_ref.actor.self_distillation.is_clip=2.0" \
  "actor_rollout_ref.rollout.n=${RP_OPSD_ROLLOUT_N}" \
  "actor_rollout_ref.rollout.temperature=1.0" \
  "actor_rollout_ref.rollout.top_p=1.0" \
  "actor_rollout_ref.rollout.top_k=-1" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
  "actor_rollout_ref.rollout.response_length=${RP_OPSD_MAX_RESPONSE_LENGTH}" \
  "actor_rollout_ref.rollout.max_model_len=$((RP_OPSD_MAX_PROMPT_LENGTH + RP_OPSD_MAX_RESPONSE_LENGTH))" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=$((RP_OPSD_MAX_PROMPT_LENGTH + RP_OPSD_MAX_RESPONSE_LENGTH))" \
  "actor_rollout_ref.rollout.agent.num_workers=8" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.norm_adv_by_std_in_grpo=False" \
  "algorithm.use_kl_in_reward=False" \
  "algorithm.rollout_correction.rollout_is=token" \
  "algorithm.rollout_correction.rollout_is_threshold=2.0" \
  "reward_model.enable=False" \
  "reward_model.use_reward_loop=False" \
  "trainer.project_name=${PROJECT_NAME}" \
  "trainer.group_name=${EXPERIMENT_NAME}" \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  "trainer.n_gpus_per_node=8" \
  "trainer.nnodes=1" \
  "trainer.total_epochs=1" \
  "trainer.total_training_steps=${TOTAL_STEPS}" \
  "trainer.save_freq=${TOTAL_STEPS}" \
  "trainer.test_freq=-1" \
  "trainer.max_actor_ckpt_to_keep=1" \
  "trainer.val_before_train=False" \
  "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints" \
  "trainer.rollout_data_dir=${ROLLOUT_DIR}" \
  2>&1 | tee "${TRAIN_LOG}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
[[ ${TRAIN_STATUS} -eq 0 ]] || die "training failed; see ${TRAIN_LOG}"

CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/global_step_${TOTAL_STEPS}"
require_dir "${CHECKPOINT_DIR}/actor"

if [[ ${SMOKE} -eq 1 ]]; then
  "${ENV_DIR}/bin/python" - "${TRAIN_LOG}" <<'PY'
import math
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
required = {
    "teacher_image_swap_fraction": r"teacher_image_swap_fraction:([0-9.eE+-]+)",
    "teacher_topk_mass": r"teacher_topk_mass_mean:([0-9.eE+-]+)",
    "bias_correction": r"mopd_bias_correction_mean:([0-9.eE+-]+)",
    "distillation_loss": r"raw_distillation_token_mean:([0-9.eE+-]+)",
}
values = {}
for name, pattern in required.items():
    matches = re.findall(pattern, text)
    if not matches:
        raise SystemExit(f"smoke metric missing: {name}")
    values[name] = float(matches[-1])
    if not math.isfinite(values[name]):
        raise SystemExit(f"smoke metric is non-finite: {name}={values[name]}")
if abs(values["teacher_image_swap_fraction"] - 1.0) > 1e-6:
    raise SystemExit(f"teacher image swap fraction is not 1: {values}")
if values["distillation_loss"] <= 0:
    raise SystemExit(f"distillation loss must be non-zero and positive: {values}")
print(values)
PY
fi

cat >"${OUTPUT_DIR}/run_summary.json" <<EOF
{
  "model": "$(basename "${MODEL_PATH}")",
  "data_release": "$(basename "${RELEASE_DIR}")",
  "source_manifest_sha256": "${RP_OPSD_SOURCE_MANIFEST_SHA256}",
  "checkpoint": "checkpoints/global_step_${TOTAL_STEPS}",
  "steps": ${TOTAL_STEPS},
  "smoke": $([[ ${SMOKE} -eq 1 ]] && printf true || printf false)
}
EOF

log "training complete: ${CHECKPOINT_DIR}"
