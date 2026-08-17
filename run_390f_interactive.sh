#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_SIM_PYTHON:-/home/eric/isaacsim/python.sh}"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "Isaac Sim Python launcher not found or not executable: ${ISAAC_PYTHON}" >&2
  exit 2
fi

cd "${REPOSITORY_ROOT}"
exec "${ISAAC_PYTHON}" isaac_loader/run_390f_v2.py --config configs/390f_v2_interactive.yaml "$@"
