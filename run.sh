#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./run.sh COMMAND [arguments]

Commands:
  prepare-data  Prepare Dataset4.0 after its public release.
  prepare-env   Build the pinned Python 3.12 environment and install RP-OPSD.
  verify        Validate model, data, source, and optionally the runtime.
  smoke         Run one RP-OPSD training step and validate diagnostics.
  train         Run the complete 55-step training.
  merge         Merge FSDP actor shards into Hugging Face weights.
  eval          Run the canonical nine-benchmark, full-resolution evaluation.
  all           Run env, smoke, train, merge, and eval.

Run "./run.sh COMMAND --help" for command-specific options.

The "all" command is environment-driven:
  MODEL_PATH=/path/Qwen3.5-9B \
  JUDGE_MODEL_PATH=/path/Qwen3.5-9B \
  EVAL_DATA_DIR=/path/eval_data \
  ASSET_ROOT=/path/dataset4_0_asset_root \
  OUTPUT_DIR=/new/output \
  ./run.sh all
EOF
}

[[ $# -gt 0 ]] || { usage; exit 2; }
COMMAND=$1
shift

case "${COMMAND}" in
  prepare-data)
    exec python3 "${PACKAGE_ROOT}/scripts/prepare_data.py" "$@"
    ;;
  prepare-env)
    exec bash "${PACKAGE_ROOT}/scripts/prepare_env.sh" "$@"
    ;;
  verify)
    exec python3 "${PACKAGE_ROOT}/scripts/verify.py" "$@"
    ;;
  smoke)
    exec bash "${PACKAGE_ROOT}/scripts/train.sh" --smoke "$@"
    ;;
  train)
    exec bash "${PACKAGE_ROOT}/scripts/train.sh" "$@"
    ;;
  merge)
    exec bash "${PACKAGE_ROOT}/scripts/merge.sh" "$@"
    ;;
  eval)
    exec bash "${PACKAGE_ROOT}/scripts/eval.sh" "$@"
    ;;
  all)
    exec bash "${PACKAGE_ROOT}/scripts/all.sh" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    printf 'Unknown command: %s\n\n' "${COMMAND}" >&2
    usage >&2
    exit 2
    ;;
esac
