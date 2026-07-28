#!/usr/bin/env bash
set -euo pipefail

RP_OPSD_PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../config/best.env
source "${RP_OPSD_PACKAGE_ROOT}/config/best.env"

log() {
  printf '[rp-opsd] %s\n' "$*"
}

die() {
  printf '[rp-opsd] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "required directory not found: $1"
}

absolute_path() {
  realpath -m "$1"
}

require_empty_output() {
  local output=$1
  if [[ -e "${output}" ]] && [[ -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    die "output directory is not empty: ${output}"
  fi
}

default_runtime_root() {
  printf '%s\n' "${RP_OPSD_RUNTIME_ROOT:-${RP_OPSD_PACKAGE_ROOT}/.runtime}"
}

default_release_dir() {
  printf '%s\n' "${RP_OPSD_DATA_RELEASE_DIR:-${RP_OPSD_PACKAGE_ROOT}/release/dataset4_0}"
}
