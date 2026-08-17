#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_EXECUTABLE=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
if [ ! -x "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(command -v python3)
fi

MPL_CACHE_DIR=${MESH_MPLCONFIGDIR:-/tmp/mesh-matplotlib-cache}
mkdir -p "$MPL_CACHE_DIR"
cd "$PROJECT_ROOT"

COMMON_ENV=(
    "MPLBACKEND=Agg"
    "MPLCONFIGDIR=$MPL_CACHE_DIR"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT"
)

# Keep the modular P0/Phase-A suite isolated from historical RL/OBJ tests so a
# failure identifies the affected subsystem and Isaac smoke scripts are never
# imported accidentally by broad discovery.
env "${COMMON_ENV[@]}" "$PYTHON_EXECUTABLE" \
    -m unittest discover -s tests -p 'test_*.py' -v

# Pytest collects both the legacy unittest.TestCase classes and the five
# function-style tests at repository root.  The explicit root glob excludes
# Isaac SimulationApp probes and smoke scripts under isaac_loader/ and tests/.
env "${COMMON_ENV[@]}" "$PYTHON_EXECUTABLE" \
    -m pytest -p no:cacheprovider -v "$PROJECT_ROOT"/test_*.py
