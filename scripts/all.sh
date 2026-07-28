#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5-9B base model}"
: "${JUDGE_MODEL_PATH:?Set JUDGE_MODEL_PATH to the Qwen3.5-9B judge}"
: "${EVAL_DATA_DIR:?Set EVAL_DATA_DIR to prepared evaluation data}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new reproduction run directory}"

OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR}")"
require_empty_output "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

RUNTIME_ROOT="${RP_OPSD_RUNTIME_ROOT:-${OUTPUT_DIR}/runtime}"
ENV_DIR="${ENV_DIR:-${RUNTIME_ROOT}/venv}"
RELEASE_DIR="${RELEASE_DIR:-$(default_release_dir)}"
ASSET_ROOT="${ASSET_ROOT:-}"

prepare_env_args=(
  --env-dir "${ENV_DIR}"
)
bash "${SCRIPT_DIR}/prepare_env.sh" "${prepare_env_args[@]}"

common_train_args=(
  --model-path "${MODEL_PATH}"
  --release-dir "${RELEASE_DIR}"
  --env-dir "${ENV_DIR}"
)
[[ -n "${ASSET_ROOT}" ]] && common_train_args+=(--asset-root "${ASSET_ROOT}")

bash "${SCRIPT_DIR}/train.sh" \
  "${common_train_args[@]}" \
  --output-dir "${OUTPUT_DIR}/smoke" \
  --work-dir "${TMPDIR:-/tmp}/rp-opsd-all-smoke-${USER:-user}" \
  --smoke

bash "${SCRIPT_DIR}/train.sh" \
  "${common_train_args[@]}" \
  --output-dir "${OUTPUT_DIR}/train" \
  --work-dir "${TMPDIR:-/tmp}/rp-opsd-all-train-${USER:-user}"

merge_args=(
  --checkpoint-dir "${OUTPUT_DIR}/train/checkpoints/global_step_${RP_OPSD_TOTAL_STEPS}"
  --output-dir "${OUTPUT_DIR}/merged"
  --env-dir "${ENV_DIR}"
)
bash "${SCRIPT_DIR}/merge.sh" "${merge_args[@]}"

eval_args=(
  --model-path "${OUTPUT_DIR}/merged"
  --judge-model-path "${JUDGE_MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}/eval"
  --eval-data-dir "${EVAL_DATA_DIR}"
  --env-dir "${ENV_DIR}"
)
[[ "${PREPARE_EVAL_DATA:-0}" == 1 ]] && eval_args+=(--prepare-data)
bash "${SCRIPT_DIR}/eval.sh" "${eval_args[@]}"

log "full reproduction complete: ${OUTPUT_DIR}"
