#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /home/eric/isaacsim/python.sh "${PROJECT_ROOT}/run_390f_presentation_demo.py" "$@"
