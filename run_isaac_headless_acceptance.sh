#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ISAAC_ROOT=${ISAAC_SIM_ROOT:-/home/eric/isaacsim}

if [ ! -x "$ISAAC_ROOT/python.sh" ]; then
    echo "Isaac Sim python.sh not found: $ISAAC_ROOT/python.sh" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
exec "$ISAAC_ROOT/python.sh" "$PROJECT_ROOT/isaac_loader/interactive_dig_demo.py" \
    --headless \
    --project-config "$PROJECT_ROOT/configs/project_25m.yaml" \
    --scoops 6 \
    --mesh-update-stride 4 \
    --hold-frames 0 \
    --output-dir "$PROJECT_ROOT/outputs/modular_25m_six_scoop"
