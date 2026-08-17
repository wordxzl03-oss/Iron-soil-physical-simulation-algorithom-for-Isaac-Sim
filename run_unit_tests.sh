#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_EXECUTABLE=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
if [ ! -x "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(command -v python3)
fi
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_EXECUTABLE" \
    -m unittest discover -s tests -p 'test_*.py' -v
