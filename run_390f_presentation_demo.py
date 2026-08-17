#!/usr/bin/env python3
"""Independent entry point for the production 390F presentation layer."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

import yaml


ROOT = Path(__file__).resolve().parent
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--presentation-config", default="configs/390f_presentation_demo.yaml")
PARSER.add_argument("--autostart", action="store_true", help="explicit recording/quality-gate mode")
PARSER.add_argument("--exit-after-hold", action="store_true", help="explicit automated-test exit")
PARSER.add_argument("--visual-blocker-check", action="store_true", help="stop 5-10 sim seconds after breakout")
PARSER.add_argument("--capture-evidence", action="store_true")
ARGS, UNKNOWN = PARSER.parse_known_args()

presentation_path = (ROOT / ARGS.presentation_config).resolve()
document = yaml.safe_load(presentation_path.read_text(encoding="utf-8"))
production_path = (ROOT / str(document["production_config"])).resolve()
runner = ROOT / "isaac_loader" / "run_390f_v2.py"

runner_args = [
    str(runner),
    "--config",
    str(production_path),
    "--no-headless",
    "--presentation-demo",
    "--final-presentation-config",
    str(presentation_path),
]
if ARGS.autostart:
    runner_args.append("--final-presentation-autostart")
if ARGS.exit_after_hold:
    runner_args.append("--final-presentation-exit-after-hold")
if ARGS.visual_blocker_check:
    runner_args.append("--final-presentation-blocker-check")
if ARGS.capture_evidence or ARGS.visual_blocker_check:
    runner_args.append("--final-presentation-capture")
sys.argv = runner_args + UNKNOWN
runpy.run_path(str(runner), run_name="__main__")
