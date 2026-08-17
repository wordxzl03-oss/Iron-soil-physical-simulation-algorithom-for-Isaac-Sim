"""Launch the reproducible Phase-B runtime acceptance with Isaac Python.

This filename intentionally does not match ``test_*.py`` so ordinary unit-test
discovery never imports SimulationApp.
"""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "isaac_loader" / "phase_b_manual_demo.py"

sys.argv = [
    str(DEMO),
    "--headless",
    "--scripted",
    "--steps",
    "480",
    "--output",
    str(ROOT / "outputs" / "phase_b_manual_runtime.json"),
    "--capture-dir",
    str(ROOT / "outputs" / "phase_b_screenshots"),
]
runpy.run_path(str(DEMO), run_name="__main__")
