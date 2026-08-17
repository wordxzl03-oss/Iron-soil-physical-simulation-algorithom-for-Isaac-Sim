#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ISAAC_ROOT_PATH=${ISAACSIM_ROOT:-/home/eric/isaacsim}
ISAAC_PYTHON="$ISAAC_ROOT_PATH/python.sh"

if [ ! -x "$ISAAC_PYTHON" ]; then
    echo "Isaac Sim Python launcher not found: $ISAAC_PYTHON" >&2
    echo "Set ISAACSIM_ROOT to the Isaac Sim 4.5 installation directory." >&2
    exit 1
fi

for SLOPE_DEG in 0 10 20; do
    "$ISAAC_PYTHON" \
        "$PROJECT_ROOT/isaac_loader/phase_cd_slope_drive_runtime.py" \
        --headless \
        --slope-deg "$SLOPE_DEG" \
        "$@"
done

echo "Phase C/D slope evidence: $PROJECT_ROOT/outputs/phase_cd_slope_summary.json"
