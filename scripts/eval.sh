#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: eval.sh --model-path DIR --judge-model-path DIR --output-dir DIR [options]

Options:
  --eval-data-dir DIR     Prepared benchmark evaluation data.
  --prepare-data          Download/prepare missing benchmark data.
  --env-dir DIR           Prepared reproduction environment.
  --benchmarks CSV        Override the canonical nine benchmarks.
  --gpu-ids CSV           Visible GPU IDs (default: 0,1,2,3,4,5,6,7).
  --target-port PORT      Target model API port (default: 8000).
  --judge-port PORT       Judge API port (default: 8001).
  --parallel-workers N    Concurrent requests (default: 256).
  --resume                Resume a partially completed evaluation directory.

The target model and judge run sequentially, so one 8-GPU worker is sufficient.
ZoomBench is rejected unconditionally.
EOF
}

RUNTIME_ROOT="$(default_runtime_root)"
MODEL_PATH=""
JUDGE_MODEL_PATH=""
OUTPUT_DIR=""
EVAL_DATA_DIR=""
ENV_DIR="${RUNTIME_ROOT}/venv"
BENCHMARKS="${RP_OPSD_EVAL_BENCHMARKS}"
GPU_IDS="0,1,2,3,4,5,6,7"
TARGET_PORT=8000
JUDGE_PORT=8001
PARALLEL_WORKERS=256
PREPARE_DATA=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH=$2; shift 2 ;;
    --judge-model-path) JUDGE_MODEL_PATH=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --eval-data-dir) EVAL_DATA_DIR=$2; shift 2 ;;
    --prepare-data) PREPARE_DATA=1; shift ;;
    --env-dir) ENV_DIR=$2; shift 2 ;;
    --benchmarks) BENCHMARKS=$2; shift 2 ;;
    --gpu-ids) GPU_IDS=$2; shift 2 ;;
    --target-port) TARGET_PORT=$2; shift 2 ;;
    --judge-port) JUDGE_PORT=$2; shift 2 ;;
    --parallel-workers) PARALLEL_WORKERS=$2; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown eval argument: $1" ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || die "--model-path is required"
[[ -n "${JUDGE_MODEL_PATH}" ]] || die "--judge-model-path is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
[[ "${TARGET_PORT}" =~ ^[0-9]+$ ]] || die "--target-port must be an integer"
[[ "${JUDGE_PORT}" =~ ^[0-9]+$ ]] || die "--judge-port must be an integer"
[[ "${PARALLEL_WORKERS}" =~ ^[1-9][0-9]*$ ]] || die "--parallel-workers must be positive"
[[ ",${BENCHMARKS}," != *",zoombench,"* ]] || die "ZoomBench is permanently excluded"
require_command curl
require_command setsid

MODEL_PATH="$(absolute_path "${MODEL_PATH}")"
JUDGE_MODEL_PATH="$(absolute_path "${JUDGE_MODEL_PATH}")"
OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR}")"
ENV_DIR="$(absolute_path "${ENV_DIR}")"
if [[ -z "${EVAL_DATA_DIR}" ]]; then
  EVAL_DATA_DIR="${RUNTIME_ROOT}/eval_data"
fi
EVAL_DATA_DIR="$(absolute_path "${EVAL_DATA_DIR}")"

require_dir "${MODEL_PATH}"
require_dir "${JUDGE_MODEL_PATH}"
require_file "${ENV_DIR}/bin/python"
if [[ ${RESUME} -eq 0 ]]; then
  require_empty_output "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}/logs" "${EVAL_DATA_DIR}"

"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify.py" --model-path "${MODEL_PATH}"
"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify.py" --model-path "${JUDGE_MODEL_PATH}"

benchmark_json_name() {
  case "$1" in
    vstar) printf 'vstar.json\n' ;;
    hrbench-4k) printf 'hr_bench_4k.json\n' ;;
    hrbench-8k) printf 'hr_bench_8k.json\n' ;;
    mme-realworld) printf 'MME_RealWorld.json\n' ;;
    mme-realworld-cn) printf 'MME_RealWorld_CN.json\n' ;;
    visualprobe) printf 'visualprobe.json\n' ;;
    mmvp) printf 'mmvp.json\n' ;;
    mmstar) printf 'mmstar.json\n' ;;
    pope) printf 'POPE.json\n' ;;
    *) die "unsupported canonical benchmark: $1" ;;
  esac
}

IFS=',' read -r -a BENCHMARK_ARRAY <<<"${BENCHMARKS}"
for index in "${!BENCHMARK_ARRAY[@]}"; do
  BENCHMARK_ARRAY[$index]="$(printf '%s' "${BENCHMARK_ARRAY[$index]}" | xargs)"
  [[ -n "${BENCHMARK_ARRAY[$index]}" ]] || die "empty benchmark in --benchmarks"
  JSON_NAME="$(benchmark_json_name "${BENCHMARK_ARRAY[$index]}")"
  if [[ ! -f "${EVAL_DATA_DIR}/${JSON_NAME}" ]]; then
    [[ ${PREPARE_DATA} -eq 1 ]] || die \
      "missing ${EVAL_DATA_DIR}/${JSON_NAME}; rerun with --prepare-data"
    "${ENV_DIR}/bin/python" "${RP_OPSD_PACKAGE_ROOT}/eval/prepare_data.py" \
      --benchmark "${BENCHMARK_ARRAY[$index]}" \
      --data_dir "${EVAL_DATA_DIR}"
  fi
done

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
TP_SIZE="${#GPU_ARRAY[@]}"
[[ ${TP_SIZE} -gt 0 ]] || die "--gpu-ids is empty"

SERVER_PID=""
stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

wait_for_server() {
  local port=$1
  local pid=$2
  local deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    if curl --silent --fail "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      return 0
    fi
    sleep 5
  done
  return 1
}

start_server() {
  local model_path=$1
  local served_name=$2
  local port=$3
  local log_path=$4
  local visible_gpus=$5
  local tensor_parallel_size=$6
  local max_model_len=${7:-}
  local -a serve_args=(
    serve "${model_path}"
    --host 127.0.0.1
    --port "${port}"
    --served-model-name "${served_name}"
    --tensor-parallel-size "${tensor_parallel_size}"
    --gpu-memory-utilization "${RP_OPSD_EVAL_GPU_MEMORY_UTILIZATION}"
    --trust-remote-code
    --disable-custom-all-reduce
  )
  [[ -n "${max_model_len}" ]] && serve_args+=(--max-model-len "${max_model_len}")
  log "starting vLLM service ${served_name} on ${visible_gpus}, TP=${tensor_parallel_size}"
  CUDA_VISIBLE_DEVICES="${visible_gpus}" setsid "${ENV_DIR}/bin/vllm" "${serve_args[@]}" \
    --disable-log-requests \
    >"${log_path}" 2>&1 &
  SERVER_PID=$!
  if ! wait_for_server "${port}" "${SERVER_PID}"; then
    tail -n 200 "${log_path}" >&2 || true
    die "vLLM service failed to become ready: ${served_name}"
  fi
}

MODEL_TAG="rp_opsd_qwen35_9b_seed${RP_OPSD_SEED}"
TARGET_SERVED_NAME="rp-opsd-qwen35-9b"
JUDGE_SERVED_NAME="qwen35-9b-judge"
RUN_DIR="${OUTPUT_DIR}/evaluation"
mkdir -p "${RUN_DIR}/model_answer" "${RUN_DIR}/judge" "${RUN_DIR}/scores"

start_server \
  "${MODEL_PATH}" \
  "${TARGET_SERVED_NAME}" \
  "${TARGET_PORT}" \
  "${OUTPUT_DIR}/logs/target_vllm.log" \
  "${GPU_IDS}" \
  "${TP_SIZE}"

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
  JSON_NAME="$(benchmark_json_name "${benchmark}")"
  (
    cd "${RUN_DIR}"
    "${ENV_DIR}/bin/python" "${RP_OPSD_PACKAGE_ROOT}/eval/infer.py" \
      --benchmark "${benchmark}" \
      --benchmark_json "${EVAL_DATA_DIR}/${JSON_NAME}" \
      --out_dir model_answer \
      --model_name "${MODEL_TAG}" \
      --seed "${RP_OPSD_SEED}" \
      --api_base "http://127.0.0.1:${TARGET_PORT}/v1" \
      --api_key EMPTY \
      --model_id "${TARGET_SERVED_NAME}" \
      --max_tokens "${RP_OPSD_EVAL_MAX_TOKENS}" \
      --top_p 1.0 \
      --max_retries 3 \
      --parallel_workers "${PARALLEL_WORKERS}" \
      --image_scale_divisor 1 \
      --enable_thinking False
  ) 2>&1 | tee "${OUTPUT_DIR}/logs/infer_${benchmark}.log"
done
stop_server

[[ ${TP_SIZE} -ge ${RP_OPSD_JUDGE_TP} ]] || die \
  "the canonical judge requires at least ${RP_OPSD_JUDGE_TP} GPUs"
JUDGE_GPU_IDS="$(IFS=,; printf '%s' "${GPU_ARRAY[*]:0:${RP_OPSD_JUDGE_TP}}")"
start_server \
  "${JUDGE_MODEL_PATH}" \
  "${JUDGE_SERVED_NAME}" \
  "${JUDGE_PORT}" \
  "${OUTPUT_DIR}/logs/judge_vllm.log" \
  "${JUDGE_GPU_IDS}" \
  "${RP_OPSD_JUDGE_TP}" \
  "${RP_OPSD_JUDGE_MAX_MODEL_LEN}"

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
  JSON_NAME="$(benchmark_json_name "${benchmark}")"
  (
    cd "${RUN_DIR}"
    "${ENV_DIR}/bin/python" "${RP_OPSD_PACKAGE_ROOT}/eval/judge_qwenlm.py" \
      --benchmark "${benchmark}" \
      --model "${MODEL_TAG}" \
      --api_base "http://127.0.0.1:${JUDGE_PORT}/v1" \
      --api_key EMPTY \
      --judge_model "${JUDGE_SERVED_NAME}" \
      --judge_max_tokens "${RP_OPSD_JUDGE_MAX_TOKENS}" \
      --judge_enable_thinking False
    "${ENV_DIR}/bin/python" "${RP_OPSD_PACKAGE_ROOT}/eval/cal_acc.py" \
      --benchmark "${benchmark}" \
      --judge_json "judge/${benchmark}/${MODEL_TAG}_answer.jsonl" \
      --benchmark_json "${EVAL_DATA_DIR}/${JSON_NAME}"
  ) 2>&1 | tee "${RUN_DIR}/scores/${benchmark}.log"
done
stop_server

"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/collect_metrics.py" \
  --run-dir "${RUN_DIR}" \
  --model-tag "${MODEL_TAG}" \
  --benchmarks "${BENCHMARKS}" \
  --output "${OUTPUT_DIR}/metrics.json"

cp "${RP_OPSD_PACKAGE_ROOT}/expected_metrics.json" "${OUTPUT_DIR}/historical_reference.json"
log "evaluation complete: ${OUTPUT_DIR}/metrics.json"
