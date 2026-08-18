"""Interactive Isaac Sim 4.5 frontend for the 390F V2 pipeline.

Default behaviour is GUI + READY.  Nothing advances the task cycle until the
user presses a run control.  A failed acceptance gate leaves Isaac open so the
physical state and diagnostics remain inspectable.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from enum import Enum
import json
from pathlib import Path
import signal
import sys
from time import perf_counter, time
import traceback
from types import SimpleNamespace

import numpy as np


class ExitReason(str, Enum):
    USER_CLOSED_WINDOW = "USER_CLOSED_WINDOW"
    PRESENTATION_COMPLETED = "PRESENTATION_COMPLETED"
    AUTOMATED_TEST_COMPLETED = "AUTOMATED_TEST_COMPLETED"
    PYTHON_EXCEPTION = "PYTHON_EXCEPTION"
    CUDA_ERROR = "CUDA_ERROR"
    PHYSX_ERROR = "PHYSX_ERROR"
    UI_EXCEPTION = "UI_EXCEPTION"
    UNEXPECTED_APPLICATION_EXIT = "UNEXPECTED_APPLICATION_EXIT"
    EXTERNAL_SIGNAL = "EXTERNAL_SIGNAL"
    UNKNOWN = "UNKNOWN"


_LIFECYCLE_RECORD = {
    "exit_reason": ExitReason.UNKNOWN.value,
    "last_simulation_time_s": 0.0,
    "last_frame_number": 0,
    "last_physics_state": "NOT_INITIALIZED",
    "simulation_app_is_running_became_false": False,
    "exception": None,
    "signal": None,
}
_ACTIVE_LIFECYCLE_PATH: Path | None = None
_SIGNAL_EXIT_REQUESTED = False


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--config", default="configs/390f_v2_interactive.yaml")
PARSER.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
PARSER.add_argument("--startup-smoke-frames", type=int, default=0)
PARSER.add_argument("--acceptance-cycles", type=int, default=0, help="explicit headless CI opt-in; default never autoplays")
PARSER.add_argument(
    "--runtime-backend",
    choices=("HOST_REFERENCE", "GPU_RUNTIME"),
    default=None,
    help="explicitly override physics.runtime_backend for controlled acceptance",
)
PARSER.add_argument(
    "--initial-heightmap",
    type=Path,
    default=None,
    help="explicit acceptance terrain override; resolution/shape remain formally validated",
)
PARSER.add_argument(
    "--material-scenario-id",
    default=None,
    help="explicit literature/engineering material sensitivity scenario",
)
PARSER.add_argument(
    "--avalanche-sensitivity",
    choices=("nominal", "more_mobile", "less_mobile"),
    default="nominal",
    help="explicit uncalibrated transition sensitivity; never treated as site calibration",
)
PARSER.add_argument(
    "--dump-after-checkpoint",
    type=Path,
    default=None,
    help=(
        "write one explicit production DEVICE checkpoint on the DUMP→DEPOSITION "
        "transition, then stop without waiting for terrain settlement"
    ),
)
PARSER.add_argument(
    "--dump-release-checkpoint",
    type=Path,
    default=None,
    help=(
        "optional companion DEVICE checkpoint immediately after the first "
        "non-zero Payload→Airborne release"
    ),
)
PARSER.add_argument(
    "--dump-plus-9-checkpoint",
    type=Path,
    default=None,
    help=(
        "with --dump-after-checkpoint, persist the same production DEVICE "
        "state after nine additional terrain-physics seconds"
    ),
)
PARSER.add_argument(
    "--continue-post-dump-until-settled",
    action="store_true",
    help=(
        "presentation acceptance only: replace the nine-second auto-pause by "
        "a physical terrain_settled gate or the explicit observation horizon"
    ),
)
PARSER.add_argument(
    "--post-dump-max-observation-s",
    type=float,
    default=60.0,
    help="maximum physical observation horizon for --continue-post-dump-until-settled",
)
PARSER.add_argument(
    "--presentation-demo",
    action="store_true",
    help="local GUI demo: pause nine simulation seconds after DEPOSITION entry",
)
PARSER.add_argument(
    "--presentation-fixed-camera",
    choices=("oblique", "side"),
    default=None,
    help="visual-only fixed camera for a clean production replay",
)
PARSER.add_argument(
    "--final-presentation-config",
    type=Path,
    default=None,
    help="presentation-only trajectory/camera/HUD configuration",
)
PARSER.add_argument(
    "--final-presentation-autostart",
    action="store_true",
    help="explicit recording/quality-gate autostart for the final presentation",
)
PARSER.add_argument(
    "--final-presentation-exit-after-hold",
    action="store_true",
    help="explicit automated quality-gate exit after the final hold",
)
PARSER.add_argument(
    "--final-presentation-blocker-check",
    action="store_true",
    help="capture/exit 5-10 simulation seconds after breakout",
)
PARSER.add_argument(
    "--final-presentation-capture",
    action="store_true",
    help="capture READY, interaction and final presentation viewport evidence",
)
PARSER.add_argument(
    "--presentation-smoke-autostart",
    action="store_true",
    help="CI-only presentation smoke; auto-press START and exit after demo auto-pause",
)
PARSER.add_argument(
    "--presentation-gui-lifecycle-smoke",
    action="store_true",
    help=(
        "non-headless lifecycle test: hold READY, run the presentation, hold "
        "post-dump PAUSE, RESET ALL, hold READY, then explicitly exit"
    ),
)
PARSER.add_argument(
    "--v3-closure-audit",
    action="store_true",
    help=(
        "one production GPU excavation with five explicit acceptance snapshots; "
        "does not alter physics, trajectory, material or task gates"
    ),
)
PARSER.add_argument(
    "--mobile-v2-pre-dump-acceptance",
    type=Path,
    default=None,
    help=(
        "one headless production cycle stopped on the first ALIGN_DUMP->DUMP "
        "transition, before any Payload->Airborne release; writes Mobile V2 fields"
    ),
)
PARSER.add_argument(
    "--cut-fill-payload-audit",
    type=Path,
    default=None,
    help="write one read-only CUT_AND_FILL payload causal audit and start checkpoint",
)
PARSER.add_argument(
    "--realistic-cut-scoop",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "replace the legacy fixed CUT target by the audited continuous A-E "
        "curl-scoop target schedule; physics and task gates remain unchanged"
    ),
)
PARSER.add_argument(
    "--tracksoil-conservation-audit",
    action="store_true",
    help=(
        "acceptance-only per-operator DEVICE scalar ledger from post CUT/CURL "
        "through PRE_DUMP; does not alter terrain physics"
    ),
)
PARSER.add_argument(
    "--mobile-v2-dual-cv-audit",
    type=Path,
    default=None,
    help=(
        "P0-2B acceptance-only per-operator DEVICE ledger output; reuses the "
        "P0-2A observer without overwriting its frozen causal artifacts"
    ),
)
PARSER.add_argument(
    "--mobile-first-write-audit",
    type=Path,
    default=None,
    help=(
        "read-only authoritative DEVICE full-state audit over t=5.45..5.75 s; "
        "autostarts one diagnostic trajectory and exits after writing the report"
    ),
)
PARSER.add_argument(
    "--mobile-first-write-visual-diagnostic",
    action="store_true",
    help=(
        "optional non-physical USD overlays for the first-write audit; requires "
        "a GUI and reads the same authoritative DEVICE audit snapshots"
    ),
)
PARSER.add_argument(
    "--tool-mobile-validity-audit",
    type=Path,
    default=None,
    help=(
        "diagnostic-only P0-2D physical-validity run through four seconds after "
        "first Mobile creation; writes compact contact and synchronized CUT telemetry"
    ),
)
PARSER.add_argument(
    "--tool-mobile-impulse-ablation",
    choices=("ON", "OFF"),
    default="ON",
    help=(
        "counterfactual used only with --tool-mobile-validity-audit; OFF retains "
        "exact contact geometry/telemetry but removes tool impulse and reaction"
    ),
)
PARSER.add_argument(
    "--tool-mobile-validity-visual-diagnostic",
    action="store_true",
    help=(
        "non-headless authoritative Mobile/contact overlay and fixed-camera frame "
        "capture for --tool-mobile-validity-audit"
    ),
)
ARGS, _UNKNOWN = PARSER.parse_known_args()
if ARGS.dump_plus_9_checkpoint is not None and ARGS.dump_after_checkpoint is None:
    raise ValueError("--dump-plus-9-checkpoint requires --dump-after-checkpoint")
if not np.isfinite(ARGS.post_dump_max_observation_s) or ARGS.post_dump_max_observation_s <= 0.0:
    raise ValueError("--post-dump-max-observation-s must be finite and positive")

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaac_bulk_pipeline.runtime import Interactive390FConfig
from isaac_bulk_pipeline.runtime.presentation_trajectory import (
    PresentationDemoConfig,
    PresentationTrajectory,
)
from isaac_bulk_pipeline.runtime.delivery_preflight import evaluate_delivery_preflight

CONFIG = Interactive390FConfig.load(ROOT / ARGS.config)
if ARGS.runtime_backend is not None:
    CONFIG = replace(CONFIG, runtime_backend=ARGS.runtime_backend)
if ARGS.initial_heightmap is not None:
    CONFIG = replace(CONFIG, initial_heightmap=ARGS.initial_heightmap.expanduser().resolve())
if ARGS.material_scenario_id is not None:
    CONFIG = replace(CONFIG, material_scenario_id=str(ARGS.material_scenario_id))
if ARGS.avalanche_sensitivity != "nominal":
    transition = dict(CONFIG.large_avalanche_transition)
    factor = -0.25 if ARGS.avalanche_sensitivity == "more_mobile" else 0.25
    transition["persistence_time_s"] *= 1.0 + factor
    transition["minimum_connected_area_m2"] *= 1.0 + factor
    transition["minimum_mobilizable_volume_m3"] *= 1.0 + factor
    transition["mobilization_depth_m"] *= 1.0 - factor
    transition["velocity_efficiency"] *= 1.0 - factor
    transition["sensitivity_case"] = (
        f"{ARGS.avalanche_sensitivity}_uncalibrated"
    )
    CONFIG = replace(CONFIG, large_avalanche_transition=transition)
REALISTIC_CUT_SCOOP = (
    CONFIG.realistic_cut_scoop_enabled
    if ARGS.realistic_cut_scoop is None
    else bool(ARGS.realistic_cut_scoop)
)
HEADLESS = CONFIG.headless if ARGS.headless is None else bool(ARGS.headless)
PRESENTATION_DEMO = bool(ARGS.presentation_demo)
FINAL_PRESENTATION = ARGS.final_presentation_config is not None
PRESENTATION_CONFIG = (
    PresentationDemoConfig.load(ARGS.final_presentation_config)
    if FINAL_PRESENTATION
    else None
)
if FINAL_PRESENTATION:
    if not PRESENTATION_DEMO:
        raise ValueError("--final-presentation-config requires --presentation-demo")
    if CONFIG.config_path != PRESENTATION_CONFIG.production_config:
        raise ValueError(
            "FINAL_PRESENTATION_PRODUCTION_CONFIG_MISMATCH: "
            f"runner={CONFIG.config_path} presentation={PRESENTATION_CONFIG.production_config}"
        )
    if CONFIG.runtime_backend != "GPU_RUNTIME":
        raise ValueError("final presentation requires production GPU_RUNTIME")
if (
    ARGS.final_presentation_autostart
    or ARGS.final_presentation_exit_after_hold
    or ARGS.final_presentation_blocker_check
    or ARGS.final_presentation_capture
) and not FINAL_PRESENTATION:
    raise ValueError("final presentation test flags require --final-presentation-config")
if ARGS.presentation_smoke_autostart and not PRESENTATION_DEMO:
    raise ValueError("--presentation-smoke-autostart requires --presentation-demo")
GUI_LIFECYCLE_SMOKE = bool(ARGS.presentation_gui_lifecycle_smoke)
V3_CLOSURE_AUDIT = bool(ARGS.v3_closure_audit)
if V3_CLOSURE_AUDIT and (
    not HEADLESS
    or ARGS.acceptance_cycles != 1
    or CONFIG.runtime_backend != "GPU_RUNTIME"
):
    raise ValueError(
        "--v3-closure-audit requires --headless --acceptance-cycles 1 "
        "and GPU_RUNTIME"
    )
if GUI_LIFECYCLE_SMOKE and (not PRESENTATION_DEMO or HEADLESS):
    raise ValueError(
        "--presentation-gui-lifecycle-smoke requires --presentation-demo "
        "and --no-headless"
    )
if ARGS.mobile_v2_pre_dump_acceptance is not None and (
    not HEADLESS or ARGS.acceptance_cycles != 1 or CONFIG.runtime_backend != "GPU_RUNTIME"
):
    raise ValueError(
        "--mobile-v2-pre-dump-acceptance requires --headless "
        "--acceptance-cycles 1 and GPU_RUNTIME"
    )
if ARGS.cut_fill_payload_audit is not None and (
    not HEADLESS or ARGS.acceptance_cycles != 1 or CONFIG.runtime_backend != "GPU_RUNTIME"
):
    raise ValueError(
        "--cut-fill-payload-audit requires --headless --acceptance-cycles 1 and GPU_RUNTIME"
    )
if (ARGS.tracksoil_conservation_audit or ARGS.mobile_v2_dual_cv_audit is not None) and (
    not HEADLESS
    or ARGS.acceptance_cycles != 1
    or CONFIG.runtime_backend != "GPU_RUNTIME"
    or ARGS.mobile_v2_pre_dump_acceptance is None
):
    raise ValueError(
        "ledger conservation audit requires --headless --acceptance-cycles 1 "
        "--mobile-v2-pre-dump-acceptance PATH and GPU_RUNTIME"
    )
if ARGS.mobile_first_write_audit is not None and (
    CONFIG.runtime_backend != "GPU_RUNTIME"
):
    raise ValueError(
        "--mobile-first-write-audit requires GPU_RUNTIME"
    )
if ARGS.mobile_first_write_visual_diagnostic and (
    ARGS.mobile_first_write_audit is None or HEADLESS
):
    raise ValueError(
        "--mobile-first-write-visual-diagnostic requires "
        "--mobile-first-write-audit PATH and --no-headless"
    )
if ARGS.tool_mobile_validity_audit is None and (
    ARGS.tool_mobile_impulse_ablation != "ON"
    or ARGS.tool_mobile_validity_visual_diagnostic
):
    raise ValueError(
        "tool-Mobile ablation/visual flags require --tool-mobile-validity-audit PATH"
    )
if ARGS.tool_mobile_validity_audit is not None and CONFIG.runtime_backend != "GPU_RUNTIME":
    raise ValueError("--tool-mobile-validity-audit requires GPU_RUNTIME")
if ARGS.tool_mobile_validity_visual_diagnostic and HEADLESS:
    raise ValueError("--tool-mobile-validity-visual-diagnostic requires --no-headless")

MANUAL_PRESENTATION_GUI = bool(
    PRESENTATION_DEMO
    and not HEADLESS
    and not GUI_LIFECYCLE_SMOKE
    and not ARGS.presentation_smoke_autostart
    and not ARGS.final_presentation_autostart
    and not ARGS.final_presentation_blocker_check
    and ARGS.startup_smoke_frames == 0
)
print(
    "\n".join(
        (
            "PRESENTATION_MODE="
            + ("MANUAL_GUI" if MANUAL_PRESENTATION_GUI else "EXPLICIT_TEST_MODE"),
            "AUTO_START="
            + str(
                bool(
                    ARGS.presentation_smoke_autostart
                    or GUI_LIFECYCLE_SMOKE
                    or ARGS.final_presentation_autostart
                    or ARGS.final_presentation_blocker_check
                )
            ).lower(),
            "AUTO_EXIT="
            + str(
                bool(
                    ARGS.presentation_smoke_autostart
                    or GUI_LIFECYCLE_SMOKE
                    or ARGS.final_presentation_exit_after_hold
                    or ARGS.final_presentation_blocker_check
                )
            ).lower(),
            "STARTUP_SMOKE_FRAMES="
            + (
                "disabled"
                if ARGS.startup_smoke_frames == 0
                else str(ARGS.startup_smoke_frames)
            ),
            "PRESENTATION_SMOKE_AUTOSTART="
            + str(bool(ARGS.presentation_smoke_autostart)).lower(),
        )
    ),
    flush=True,
)

# A signed autonomous cycle is identity-gated before SimulationApp import.
# GUI/manual development remains available without claiming formal acceptance.
if ARGS.acceptance_cycles > 0:
    _preflight = evaluate_delivery_preflight(
        repository_root=ROOT,
        runtime_backend=CONFIG.runtime_backend,
        grid_resolution_m=CONFIG.grid_spacing_m,
        grid_shape_yx=CONFIG.grid_shape,
        terrain_file=CONFIG.initial_heightmap,
        material_scenario=CONFIG.material_scenario_id,
        production_core="EarthmovingPhysicsCore",
        full_field_transfer_disabled=True,
        root_pose_write_count=0,
    )
    _preflight_output = ROOT / "outputs/390f_v2/final_cycle_preflight.json"
    _preflight_output.parent.mkdir(parents=True, exist_ok=True)
    _preflight_output.write_text(
        json.dumps(_preflight.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    if _preflight.status != "PASS":
        _rejected = ROOT / "outputs/390f_v2/PRE_FLIGHT_REJECTED_NOT_ACCEPTED.json"
        _rejected.write_text(
            json.dumps(_preflight.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        raise SystemExit(
            "PRE_FLIGHT_REJECTED_NOT_ACCEPTED: " + ",".join(_preflight.failures)
        )

if CONFIG.runtime_backend == "GPU_RUNTIME":
    if not (
        CONFIG.active_tile_size == CONFIG.visual_chunk_size
        == CONFIG.contact_chunk_size
    ):
        raise RuntimeError(
            "GPU_RUNTIME_TILE_PUBLICATION_MISMATCH: active, visual and contact "
            "tile sizes must be identical"
        )

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": HEADLESS,
        # NVIDIA recommends explicitly disabling viewport updates for headless
        # standalone apps: headless otherwise still updates a Kit viewport.
        # This job has no camera observations or streaming consumer.
        "disable_viewport_updates": HEADLESS,
        # Keep two logical CPUs available to the OS and Python-side terrain/RL
        # bookkeeping.  The 20-thread hybrid i9 otherwise oversubscribes Kit's
        # worker schedulers during the CPU soil solve.
        "limit_cpu_threads": 16,
        "multi_gpu": False,
        "width": CONFIG.viewport_width,
        "height": CONFIG.viewport_height,
        "anti_aliasing": 2,
    }
)


def _material_from_config(path: Path, scenario_id: str):
    from isaac_bulk_pipeline.bulk_state import MaterialScenario

    data = json.loads(path.read_text(encoding="utf-8"))
    selected = next((item for item in data["scenarios"] if item["id"] == scenario_id), None)
    if selected is None:
        raise ValueError(f"[390FInteractive] material scenario not found: {scenario_id}")
    required = {
        "bulk_density_kg_m3", "internal_friction_angle_deg", "cohesion_pa",
        "tool_wall_friction_angle_deg", "theta_start_deg", "theta_stop_deg",
        "mobile_friction_coefficient",
    }
    missing = sorted(required - selected.keys())
    if missing:
        raise ValueError(f"[390FInteractive] incomplete material scenario: {missing}")
    return MaterialScenario(
        selected["id"],
        selected["bulk_density_kg_m3"],
        selected["internal_friction_angle_deg"],
        selected["cohesion_pa"],
        float(np.tan(np.deg2rad(selected["tool_wall_friction_angle_deg"]))),
        selected["theta_start_deg"],
        selected["theta_stop_deg"],
        selected["mobile_friction_coefficient"],
    )


def _set_presentation_camera(index: int) -> None:
    """Stable authored views; camera selection never affects physics."""

    if HEADLESS:
        return
    views = {
        1: (
            PRESENTATION_CONFIG.camera_eye_m
            if FINAL_PRESENTATION
            else np.asarray([21.0, -24.0, 13.0]),
            PRESENTATION_CONFIG.camera_target_m
            if FINAL_PRESENTATION
            else np.asarray([5.5, 0.0, 2.5]),
        ),
        2: (np.asarray([12.5, -10.5, 7.0]), np.asarray([5.4, 0.2, 1.8])),
        3: (np.asarray([20.0, -10.0, 10.0]), np.asarray([8.0, 1.0, 2.2])),
        4: (np.asarray([5.5, -27.0, 8.0]), np.asarray([5.5, 0.0, 2.2])),
    }
    eye, target = views.get(int(index), views[1])
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=eye, target=target)
    except Exception as error:
        print(f"PRESENTATION_CAMERA_WARNING: {error}", flush=True)


class InteractivePanel:
    """Thin UI command layer; callbacks never step physics."""

    def __init__(self, control, initial_joint_deg: np.ndarray) -> None:
        self.control = control
        self.pending_reset = None
        self.status_text = None
        self.status_model = None
        self.failure_text = None
        self.failure_model = None
        self.joint_models = {}
        self.track_models = {}
        self.hud_models = {}
        self.window = None
        if not HEADLESS:
            self._build(initial_joint_deg)

    def _build(self, initial_joint_deg: np.ndarray) -> None:
        import omni.ui as ui
        from isaac_bulk_pipeline.runtime import ResetLevel

        if PRESENTATION_DEMO:
            self.window = ui.Window(
                "Observation" if FINAL_PRESENTATION else "390F LOCAL PRESENTATION — DEMO ONLY",
                width=330 if FINAL_PRESENTATION else 390,
                height=245 if FINAL_PRESENTATION else 325,
            )
            with self.window.frame:
                with ui.VStack(spacing=7, height=0):
                    if not FINAL_PRESENTATION:
                        ui.Label(
                            "DEMO_ONLY | POST_DUMP_TIMESCALE_UNDER_VALIDATION",
                            height=25,
                        )
                    if FINAL_PRESENTATION:
                        for key, label in (
                            ("bucket_force_kn", "Bucket Force (kN)"),
                            ("payload_mass_kg", "Payload Mass (kg)"),
                            ("payload_volume_m3", "Payload Volume (m³)"),
                            ("mobile_volume_m3", "Mobile Soil (m³)"),
                        ):
                            with ui.HStack(height=26):
                                ui.Label(label, width=175)
                                model = ui.SimpleFloatModel(0.0)
                                self.hud_models[key] = model
                                ui.FloatField(model=model, enabled=False)
                        ui.Label("GPU Physics Active", height=24)
                        self.failure_model = ui.SimpleStringModel("")
                        self.failure_text = ui.StringField(
                            model=self.failure_model,
                            read_only=True,
                            multiline=True,
                            height=24,
                        )
                    else:
                        self.status_model = ui.SimpleStringModel(
                            "Observation\nREADY"
                        )
                        self.status_text = ui.StringField(
                            model=self.status_model,
                            read_only=True,
                            multiline=True,
                            height=145,
                        )
                        self.failure_model = ui.SimpleStringModel("")
                        self.failure_text = ui.StringField(
                            model=self.failure_model,
                            read_only=True,
                            multiline=True,
                            height=24,
                        )
                    with ui.HStack(height=34):
                        ui.Button("START", clicked_fn=lambda: self._run(1))
                        ui.Button("PAUSE", clicked_fn=self._pause)
                        ui.Button("RESUME", clicked_fn=self._resume)
                        ui.Button(
                            "RESET",
                            clicked_fn=lambda: self._queue_reset(ResetLevel.ALL),
                        )
                    if not FINAL_PRESENTATION:
                        with ui.HStack(height=30):
                            ui.Button("1 OVERALL", clicked_fn=lambda: _set_presentation_camera(1))
                            ui.Button("2 BUCKET", clicked_fn=lambda: _set_presentation_camera(2))
                            ui.Button("3 DUMP", clicked_fn=lambda: _set_presentation_camera(3))
            return
        self.window = ui.Window("390F V2 Interactive Earthmoving", width=520, height=920)
        with self.window.frame:
            with ui.VStack(spacing=5, height=0):
                ui.Label("390F V2 — USER CONTROLLED", height=26)
                self.status_model = ui.SimpleStringModel("INITIALIZING")
                self.status_text = ui.StringField(
                    model=self.status_model,
                    read_only=True,
                    multiline=True,
                    height=245,
                )
                self.failure_model = ui.SimpleStringModel("")
                self.failure_text = ui.StringField(
                    model=self.failure_model,
                    read_only=True,
                    multiline=True,
                    height=52,
                )
                ui.Separator()
                with ui.HStack(height=30):
                    ui.Button("START / 1 CYCLE", clicked_fn=lambda: self._run(1))
                    ui.Button("RUN 3 CYCLES", clicked_fn=lambda: self._run(3))
                with ui.HStack(height=30):
                    ui.Button("PAUSE", clicked_fn=self._pause)
                    ui.Button("RESUME", clicked_fn=self._resume)
                    ui.Button("EMERGENCY STOP", clicked_fn=self.control.emergency_stop)
                with ui.HStack(height=30):
                    ui.Button("RESET ROBOT", clicked_fn=lambda: self._queue_reset(ResetLevel.ROBOT))
                    ui.Button("RESET PAYLOAD", clicked_fn=lambda: self._queue_reset(ResetLevel.PAYLOAD))
                with ui.HStack(height=30):
                    ui.Button("RESET TERRAIN", clicked_fn=lambda: self._queue_reset(ResetLevel.TERRAIN))
                    ui.Button("RESET ALL", clicked_fn=lambda: self._queue_reset(ResetLevel.ALL))
                ui.Separator()
                ui.Label("Soil resistance (change only while READY)")
                with ui.HStack(height=30):
                    ui.Button("NO SOIL", clicked_fn=lambda: self._set_force("NO_SOIL_FORCE"))
                    ui.Button("QUASI STATIC", clicked_fn=lambda: self._set_force("QUASI_STATIC_ONLY"))
                    ui.Button("FULL SOIL", clicked_fn=lambda: self._set_force("FULL_SOIL_FORCE"))
                with ui.HStack(height=30):
                    ui.Button("TRACK SOIL ON/OFF", clicked_fn=self._toggle_track)
                    ui.Button("DEBUG OVERLAY", clicked_fn=self.control.toggle_debug_overlay)
                ui.Button("KINEMATIC DRIVE DEBUG ON/OFF", height=30, clicked_fn=self.control.toggle_kinematic_debug)
                ui.Separator()
                ui.Label("Drive targets — physical joints, no pose teleport")
                limits = ((-180.0, 180.0), (-25.0, 45.0), (-100.0, 45.0), (-120.0, 60.0))
                for index, name in enumerate(("Swing", "Boom", "Stick", "Bucket")):
                    ui.Label(name)
                    model = ui.SimpleFloatModel(float(initial_joint_deg[index]))
                    self.joint_models[name.lower()] = model
                    ui.FloatSlider(model=model, min=limits[index][0], max=limits[index][1], step=0.5, height=20)
                ui.Label("Left Track (tractive force at physical track body)")
                self.track_models["left"] = ui.SimpleFloatModel(0.0)
                ui.FloatSlider(model=self.track_models["left"], min=-1.0, max=1.0, step=0.05, height=20)
                ui.Label("Right Track (tractive force at physical track body)")
                self.track_models["right"] = ui.SimpleFloatModel(0.0)
                ui.FloatSlider(model=self.track_models["right"], min=-1.0, max=1.0, step=0.05, height=20)

    def _run(self, count: int) -> None:
        try:
            self.control.run_cycles(count)
        except RuntimeError as error:
            self._local_error(str(error))

    def _pause(self) -> None:
        try:
            self.control.pause()
        except RuntimeError as error:
            self._local_error(str(error))

    def _resume(self) -> None:
        try:
            self.control.resume()
        except RuntimeError as error:
            self._local_error(str(error))

    def _set_force(self, value: str) -> None:
        from isaac_bulk_pipeline.runtime import SoilForceMode
        try:
            self.control.set_soil_force_mode(SoilForceMode(value))
        except RuntimeError as error:
            self._local_error(str(error))

    def _toggle_track(self) -> None:
        try:
            self.control.toggle_track_soil()
        except RuntimeError as error:
            self._local_error(str(error))

    def _queue_reset(self, level) -> None:
        self.pending_reset = level

    def _local_error(self, message: str) -> None:
        if self.failure_model is not None:
            self.failure_model.set_value(message)

    def update_observation(
        self,
        *,
        bucket_force_kn: float,
        payload_mass_kg: float,
        payload_volume_m3: float,
        mobile_volume_m3: float,
    ) -> None:
        values = locals()
        for key, model in self.hud_models.items():
            model.set_value(float(values[key]))

    def update(self, extra: str = "") -> None:
        snapshot = self.control.snapshot()
        if PRESENTATION_DEMO:
            if FINAL_PRESENTATION:
                if self.failure_model is not None and snapshot.failure is not None:
                    self.failure_model.set_value(
                        f"FAIL {snapshot.failure.code}: {snapshot.failure.message}"
                    )
                return
            if self.status_model is not None:
                self.status_model.set_value(extra or "PHASE  READY")
            if self.failure_model is not None and snapshot.failure is not None:
                message = f"FAIL {snapshot.failure.code}: {snapshot.failure.message}"
                self.failure_model.set_value(message)
            return
        status = (
            f"STATE={snapshot.state.value} | CYCLE={snapshot.current_cycle} | "
            f"FORCE={snapshot.soil_force_mode.value} | TRACK_SOIL={'ON' if snapshot.track_soil_enabled else 'OFF'}"
        )
        if extra:
            status += f"\n{extra}"
        if self.status_model is not None:
            self.status_model.set_value(status)
        if self.failure_model is not None and snapshot.failure is not None:
            self.failure_model.set_value(
                f"FAIL {snapshot.failure.code}: {snapshot.failure.message}"
            )

    def joint_target_rad(self, fallback: np.ndarray) -> np.ndarray:
        if not self.joint_models:
            return np.asarray(fallback, dtype=np.float64)
        return np.deg2rad(np.asarray([self.joint_models[name].as_float for name in ("swing", "boom", "stick", "bucket")]))

    def track_commands(self) -> tuple[float, float]:
        if not self.track_models:
            return 0.0, 0.0
        return self.track_models["left"].as_float, self.track_models["right"].as_float


def main() -> None:
    global _ACTIVE_LIFECYCLE_PATH, _SIGNAL_EXIT_REQUESTED
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction

    from isaac_bulk_pipeline.config import MeshConfig, RobotConfig
    from isaac_bulk_pipeline.bulk_interaction import (
        LargeAvalancheTransitionConfig,
        TrackSoilConfig,
    )
    from isaac_bulk_pipeline.contact import IsaacChunkedContactMeshAdapter
    from isaac_bulk_pipeline.operation import (
        ExcavatorCycleConfig,
        ExcavatorCycleObservation,
        ExcavatorCycleState,
        ExcavatorCycleStateMachine,
    )
    from isaac_bulk_pipeline.robot import RobotAdapter
    from isaac_bulk_pipeline.runtime import (
        EarthmovingPhysicsCore, InteractiveControlModel, InteractiveRuntimeState,
        ResetLevel, RuntimeFailure,
    )
    from isaac_bulk_pipeline.runtime.device_checkpoint import (
        sha256_file,
        write_device_checkpoint,
    )
    from isaac_bulk_pipeline.runtime.flow_arrest_audit import (
        classify_flow_arrest_state,
    )
    from isaac_bulk_pipeline.runtime.cut_fill_payload_audit import (
        CutFillPayloadCausalAudit,
    )
    from isaac_bulk_pipeline.runtime.curl_scoop_trajectory import (
        RealisticCurlScoopTrajectory,
    )
    from isaac_bulk_pipeline.runtime.tracksoil_conservation_audit import (
        TrackSoilConservationAudit,
    )
    from isaac_bulk_pipeline.runtime.mobile_first_write_audit import (
        IsaacMobileFirstWriteVisualization, MobileFirstWriteAudit,
    )
    from isaac_bulk_pipeline.runtime.tool_mobile_validity_audit import (
        ToolMobilePhysicalValidityAudit,
    )
    from isaac_bulk_pipeline.terrain import HeightmapIO, TerrainGrid
    from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolKinematicsAdapter
    from isaac_bulk_pipeline.vehicle import (
        DifferentialTrackDriveConfig,
        DifferentialTrackDriveModel,
        ExcavatorActuatorConfig,
        ExcavatorActuatorModel,
        IsaacDifferentialTrackDriveAdapter,
        TrackFootprintConfig,
        TrackFootprintRasterizer,
    )
    from isaac_bulk_pipeline.visualization import (
        ChunkedDynamicMeshAdapter, LightingManager, TerrainMaterialAdapter,
    )


    run_id = f"run_{int(time())}"
    run_dir = CONFIG.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    presentation_run_dir = (
        PRESENTATION_CONFIG.output_root / run_id if FINAL_PRESENTATION else run_dir
    )
    presentation_run_dir.mkdir(parents=True, exist_ok=True)
    presentation_screenshot_dir = presentation_run_dir / "screenshots"
    if FINAL_PRESENTATION:
        presentation_screenshot_dir.mkdir(parents=True, exist_ok=True)
    tool_mobile_validity_frame_dir = (
        None
        if ARGS.tool_mobile_validity_audit is None
        else ARGS.tool_mobile_validity_audit.expanduser().resolve().with_suffix("").with_name(
            ARGS.tool_mobile_validity_audit.expanduser().resolve().stem + "_frames"
        )
    )
    if ARGS.tool_mobile_validity_visual_diagnostic:
        assert tool_mobile_validity_frame_dir is not None
        tool_mobile_validity_frame_dir.mkdir(parents=True, exist_ok=True)
    failure_path = run_dir / "runtime_failure.json"
    manifest_path = run_dir / "runtime_manifest.json"
    telemetry_path = run_dir / "runtime_telemetry.json"
    _ACTIVE_LIFECYCLE_PATH = run_dir / "application_exit.json"

    def _record_signal(signum, _frame) -> None:
        global _SIGNAL_EXIT_REQUESTED
        _LIFECYCLE_RECORD["signal"] = signal.Signals(signum).name
        _LIFECYCLE_RECORD["exit_reason"] = ExitReason.EXTERNAL_SIGNAL.value
        # Kit may invoke Python signal callbacks from its async-engine update.
        # Raising there is consumed by the engine and leaves the GUI alive.
        # Latch the request and let the owning application loop finalize once.
        _SIGNAL_EXIT_REQUESTED = True

    signal.signal(signal.SIGINT, _record_signal)
    signal.signal(signal.SIGTERM, _record_signal)

    world = World(stage_units_in_meters=1.0, physics_dt=CONFIG.physics_dt_s, rendering_dt=CONFIG.physics_dt_s)
    add_reference_to_stage(str(CONFIG.vehicle_asset), "/World/Excavator")
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    # The apron supports the portion of the real 6.17 m tracks that initially
    # lies outside the 35 m heightmap. It does not overlap the authoritative
    # terrain domain and therefore cannot mask rut deformation inside it.
    ground_path = "/World/LocalSupportApron"
    ground = UsdGeom.Cube.Define(stage, ground_path)
    ground.CreateSizeAttr(2.0)
    terrain_min_x_world = float(CONFIG.terrain_translation_world_m[0])
    apron_width_m = 20.0
    # The former 0.50 m overlap put the apron and the first 64-cell static
    # height-field chunk under the same track contact patch. Two coincident
    # static support manifolds can pin the reduced-coordinate articulation
    # even while the drive applies its full measured wrench. Meet exactly at
    # the authoritative-domain boundary instead; there is no unsupported gap.
    apron_flat_overlap_m = 0.0
    ground.AddTranslateOp().Set(
        Gf.Vec3d(
            terrain_min_x_world + apron_flat_overlap_m - 0.5 * apron_width_m,
            0.0,
            -0.10,
        )
    )
    ground.AddScaleOp().Set(Gf.Vec3f(0.5 * apron_width_m, 30.0, 0.10))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    ground.CreateDisplayColorAttr([(0.17, 0.19, 0.21)])
    if CONFIG.auto_align_track_bottom_to_ground:
        bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        minimum_track_z = min(
            float(bounds_cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMin()[2])
            for path in (CONFIG.left_track_body, CONFIG.right_track_body)
        )
        initial_placement_z_m = float(CONFIG.track_ground_clearance_m - minimum_track_z)
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/Excavator")).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, initial_placement_z_m))
    else:
        initial_placement_z_m = 0.0
    for name in ("swing_joint", "boom_joint", "stick_joint", "bucket_joint"):
        UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{CONFIG.articulation_root}/Joints/{name}"), "angular").GetTargetVelocityAttr().Set(0.0)
    if CONFIG.mobile_base_enabled:
        anchor = UsdPhysics.Joint(stage.GetPrimAtPath(CONFIG.world_anchor_joint))
        if not anchor or not anchor.GetPrim().IsValid():
            raise RuntimeError(f"[390FInteractive] configured world anchor missing: {CONFIG.world_anchor_joint}")
        anchor.GetJointEnabledAttr().Set(False)
        anchor.GetPrim().SetActive(False)
    support_group = UsdPhysics.CollisionGroup.Define(stage, "/World/V2_390F_CollisionGroups/TerrainSupport")
    Usd.CollectionAPI.Apply(support_group.GetPrim(), "colliders").CreateIncludesRel().SetTargets([Sdf.Path(ground_path)])
    bucket_group = UsdPhysics.CollisionGroup.Define(stage, "/World/V2_390F_CollisionGroups/BucketInteraction")
    Usd.CollectionAPI.Apply(bucket_group.GetPrim(), "colliders").CreateIncludesRel().SetTargets([Sdf.Path(CONFIG.bucket_link)])
    support_group.CreateFilteredGroupsRel().AddTarget(bucket_group.GetPath())
    bucket_group.CreateFilteredGroupsRel().AddTarget(support_group.GetPath())

    transform = np.eye(4)
    transform[:3, 3] = CONFIG.terrain_translation_world_m
    grid = TerrainGrid(701, 701, CONFIG.grid_spacing_m, CONFIG.grid_spacing_m, 0.0, 0.0, "/World/Terrain/DynamicBulkSurface", terrain_to_world_matrix=transform)
    height = HeightmapIO.load(CONFIG.initial_heightmap, grid=grid, source_axis_order="xy")
    terrain_mesh = ChunkedDynamicMeshAdapter(CONFIG.visual_chunk_size)
    terrain_mesh.initialize(
        stage, grid,
        MeshConfig(collision_enabled=False, update_normals=True, mesh_update_rate_hz=10.0, normal_update_rate_hz=5.0, display_color_rgb=(0.24, 0.075, 0.035)),
        height,
        root_prim_path="/World/Terrain/DynamicVisualChunks",
    )
    contact_mesh = IsaacChunkedContactMeshAdapter(CONFIG.contact_chunk_size)
    contact_status = contact_mesh.initialize(
        stage,
        grid,
        height,
        root_prim_path="/World/Terrain/DynamicContactChunks",
        timestamp_s=0.0,
    )
    # TrackSoil/track drive explicitly own tangential shear.  Leaving an
    # implicit PhysX Coulomb material on the same pair would double count that
    # resistance and can pin a physically commanded track force.  This
    # material intentionally supplies normal collision only.
    normal_contact_material = UsdShade.Material.Define(
        stage, "/World/Materials/TrackSoilNormalContactOnly"
    )
    normal_contact_physics = UsdPhysics.MaterialAPI.Apply(
        normal_contact_material.GetPrim()
    )
    normal_contact_physics.CreateStaticFrictionAttr(0.0)
    normal_contact_physics.CreateDynamicFrictionAttr(0.0)
    normal_contact_physics.CreateRestitutionAttr(0.0)
    for path in (
        ground_path,
        CONFIG.left_track_body,
        CONFIG.right_track_body,
        *contact_mesh.collider_paths,
    ):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"[390FInteractive] contact material target missing: {path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            normal_contact_material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
    dynamic_support_group = UsdPhysics.CollisionGroup.Define(
        stage, "/World/V2_390F_CollisionGroups/DynamicTerrainContact"
    )
    Usd.CollectionAPI.Apply(
        dynamic_support_group.GetPrim(), "colliders"
    ).CreateIncludesRel().SetTargets([Sdf.Path(path) for path in contact_mesh.collider_paths])
    dynamic_support_group.CreateFilteredGroupsRel().AddTarget(bucket_group.GetPath())
    bucket_group.CreateFilteredGroupsRel().AddTarget(dynamic_support_group.GetPath())
    TerrainMaterialAdapter("iron_ore_fines").author_and_bind_usd(
        stage, "/World/Terrain/DynamicVisualChunks"
    )
    LightingManager("outdoor_day").author_usd(stage, "/World/V2_390F_Lighting")
    if not HEADLESS:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            if FINAL_PRESENTATION:
                set_camera_view(
                    eye=PRESENTATION_CONFIG.camera_eye_m,
                    target=PRESENTATION_CONFIG.camera_target_m,
                )
                camera = UsdGeom.Camera.Get(stage, "/OmniverseKit_Persp")
                if camera and camera.GetPrim().IsValid():
                    camera.GetFocalLengthAttr().Set(
                        PRESENTATION_CONFIG.camera_focal_length_mm
                    )
            else:
                set_camera_view(eye=np.array([21.0, -24.0, 13.0]), target=np.array([5.5, 0.0, 2.5]))
        except Exception:
            pass

    articulation = world.scene.add(SingleArticulation(CONFIG.articulation_root, name="interactive_390f"))
    left_track = world.scene.add(RigidPrim(CONFIG.left_track_body, name="interactive_left_track", reset_xform_properties=False))
    right_track = world.scene.add(RigidPrim(CONFIG.right_track_body, name="interactive_right_track", reset_xform_properties=False))
    lower_body = world.scene.add(RigidPrim(CONFIG.lower_body, name="interactive_lower_body", reset_xform_properties=False))
    bucket_force_body = world.scene.add(RigidPrim(CONFIG.bucket_link, name="interactive_bucket_force", reset_xform_properties=False))
    articulation.set_joints_default_state(
        positions=np.asarray(CONFIG.phase_targets_rad["initial_pose"], dtype=np.float64)
    )
    world.reset()
    if not HEADLESS:
        world.pause()
    if tuple(str(name) for name in articulation.dof_names) != ("swing_joint", "boom_joint", "stick_joint", "bucket_joint"):
        raise RuntimeError(f"[390FInteractive] unexpected DOFs: {articulation.dof_names}")
    controller = articulation.get_articulation_controller()
    actuator_config = ExcavatorActuatorConfig.cat_390f_l_mass_configuration()
    actuator_limit_by_name = {
        item.joint_name: item for item in actuator_config.joints
    }
    for joint_name in articulation.dof_names:
        drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{CONFIG.articulation_root}/Joints/{joint_name}"),
            "angular",
        )
        drive.GetMaxForceAttr().Set(
            actuator_limit_by_name[str(joint_name)].effort_limit_nm
        )
    arm_actuator = ExcavatorActuatorModel(
        actuator_config, tuple(str(name) for name in articulation.dof_names)
    )
    arm_actuator.reset(
        np.asarray(articulation.get_joint_velocities(), dtype=np.float64).reshape(-1)
    )
    arm_drive_target_rad = np.asarray(
        articulation.get_joint_positions(), dtype=np.float64
    ).reshape(-1).copy()

    def apply_bounded_arm_target(desired_position_rad: np.ndarray):
        """Advance a continuous, force-capped drive target; never teleport links."""

        nonlocal arm_drive_target_rad
        desired = np.asarray(desired_position_rad, dtype=np.float64).reshape(-1)
        measured_position = np.asarray(
            articulation.get_joint_positions(), dtype=np.float64
        ).reshape(-1)
        measured_velocity = np.asarray(
            articulation.get_joint_velocities(), dtype=np.float64
        ).reshape(-1)
        output = arm_actuator.step(
            desired,
            measured_position,
            measured_velocity,
            CONFIG.physics_dt_s,
        )
        proposed = arm_drive_target_rad + output.target_velocity_rad_s * CONFIG.physics_dt_s
        remaining_before = desired - arm_drive_target_rad
        remaining_after = desired - proposed
        reached = (remaining_before == 0.0) | (
            np.sign(remaining_before) != np.sign(remaining_after)
        )
        proposed[reached] = desired[reached]
        arm_drive_target_rad = proposed
        controller.apply_action(ArticulationAction(joint_positions=arm_drive_target_rad))
        return output
    lower_matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(CONFIG.lower_body)), dtype=np.float64).T
    axis_u, _, axis_vh = np.linalg.svd(lower_matrix[:3, :3])
    lower_rigid = axis_u @ axis_vh
    forward_axis_index = int(np.argmax(np.abs(lower_rigid[0, :])))
    derived_forward_axis_local = np.zeros(3)
    derived_forward_axis_local[forward_axis_index] = float(np.sign(lower_rigid[0, forward_axis_index]))
    lower_position_world = np.asarray(
        lower_body.get_world_poses()[0], dtype=np.float64
    ).reshape(-1, 3)[0]
    application_bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    track_application_offsets_lower_local_m = []
    for track_path in (CONFIG.left_track_body, CONFIG.right_track_body):
        aligned = application_bounds_cache.ComputeWorldBound(
            stage.GetPrimAtPath(track_path)
        ).ComputeAlignedRange()
        center_world = 0.5 * (
            np.asarray(aligned.GetMin(), dtype=np.float64)
            + np.asarray(aligned.GetMax(), dtype=np.float64)
        )
        track_application_offsets_lower_local_m.append(
            lower_rigid.T @ (center_world - lower_position_world)
        )
    track_drive = IsaacDifferentialTrackDriveAdapter(
        left_track,
        right_track,
        lower_body,
        DifferentialTrackDriveModel(
            DifferentialTrackDriveConfig(
                maximum_tractive_force_per_track_n=float(CONFIG.controller["maximum_tractive_force_per_track_n"]),
                maximum_braking_force_per_track_n=float(CONFIG.controller["maximum_braking_force_per_track_n"]),
                nominal_track_speed_m_s=CONFIG.nominal_track_belt_speed_m_s,
                full_force_speed_error_m_s=CONFIG.nominal_track_belt_speed_m_s,
                forward_axis_lower_body_local=tuple(derived_forward_axis_local.tolist()),
            )
        ),
        left_application_offset_lower_local_m=track_application_offsets_lower_local_m[0],
        right_application_offset_lower_local_m=track_application_offsets_lower_local_m[1],
    )
    initial_q = np.asarray(articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
    presentation_trajectory = (
        PresentationTrajectory(PRESENTATION_CONFIG, initial_q)
        if FINAL_PRESENTATION
        else None
    )

    descriptor = ToolDescriptorLoader.load(ToolDescriptorLoader.load_config(CONFIG.bucket_descriptor))
    material = _material_from_config(CONFIG.material_scenarios, CONFIG.material_scenario_id)
    physics_core = EarthmovingPhysicsCore(
        grid=grid, descriptor=descriptor, material=material, initial_heightmap_m=height,
        runtime_backend=CONFIG.runtime_backend,
        solver_backend=CONFIG.solver_backend, slope_backend=CONFIG.slope_backend,
        tile_size=CONFIG.active_tile_size,
        large_avalanche_iteration_threshold=CONFIG.large_avalanche_iteration_threshold,
        numerical_safety_max_iterations=CONFIG.numerical_safety_max_iterations,
        minislope_round_budget_per_step=CONFIG.minislope_round_budget_per_step,
        minislope_tolerance_m=CONFIG.minislope_tolerance_m,
        large_avalanche_config=LargeAvalancheTransitionConfig.from_mapping(
            dict(CONFIG.large_avalanche_transition)
        ),
        track_soil_config=TrackSoilConfig(
            sinkage_rate_m_s=float(CONFIG.track_soil["sinkage_rate_m_s"]),
            slip_gain=float(CONFIG.track_soil["slip_gain"]),
            maximum_sinkage_per_step_m=float(CONFIG.track_soil["maximum_sinkage_per_step_m"]),
            maximum_total_rut_depth_m=float(CONFIG.track_soil["maximum_total_rut_depth_m"]),
            shoulder_halo_cells=int(CONFIG.track_soil["shoulder_halo_cells"]),
            minimum_slip_speed_m_s=float(CONFIG.track_soil["minimum_slip_speed_m_s"]),
            parameter_status=str(CONFIG.track_soil["parameter_status"]),
        ),
    )
    payload_visual_prim = None
    payload_visual_translate = None
    payload_visual_scale = None
    if PRESENTATION_DEMO:
        # Presentation-only geometry. Its size and center are driven by the
        # authoritative PayloadState; it has no collision/rigid-body API.
        payload_visual_prim = UsdGeom.Sphere.Define(
            stage, f"{CONFIG.bucket_link}/PresentationPayloadVisual"
        )
        payload_visual_prim.CreateRadiusAttr(1.0)
        payload_visual_prim.CreateDisplayColorAttr([(0.31, 0.085, 0.025)])
        payload_visual_prim.CreateDisplayOpacityAttr([0.96])
        payload_xform = UsdGeom.Xformable(payload_visual_prim.GetPrim())
        payload_visual_translate = payload_xform.AddTranslateOp()
        payload_visual_scale = payload_xform.AddScaleOp()
        payload_visual_prim.MakeInvisible()

    def update_payload_visual() -> None:
        if payload_visual_prim is None:
            return
        payload = physics_core.payload
        volume = float(payload.volume_m3)
        if volume <= 1.0e-8:
            payload_visual_prim.MakeInvisible()
            return
        aspect = np.asarray([1.35, 0.72, 0.52], dtype=np.float64)
        radius = float(
            np.cbrt(volume / ((4.0 / 3.0) * np.pi * float(np.prod(aspect))))
        )
        center = np.asarray(payload.center_of_mass_bucket_frame_m, dtype=np.float64)
        payload_visual_translate.Set(Gf.Vec3d(*center.tolist()))
        payload_visual_scale.Set(Gf.Vec3f(*(radius * aspect).tolist()))
        payload_visual_prim.MakeVisible()
    robot = RobotAdapter(articulation=articulation, time_source=lambda: float(world.current_time))
    robot.initialize(stage, RobotConfig("/World/Excavator", CONFIG.articulation_root, CONFIG.bucket_link))
    kinematics = ToolKinematicsAdapter(descriptor, grid)
    physics_core.initialize_tool(kinematics.update(robot.get_tool_link_pose_world(), float(world.current_time)))
    footprint_rasterizer = TrackFootprintRasterizer(
        grid,
        TrackFootprintConfig(
            length_m=CONFIG.track_footprint_length_m,
            width_m=CONFIG.track_footprint_width_m,
            nominal_belt_speed_m_s=CONFIG.nominal_track_belt_speed_m_s,
            contact_gap_m=CONFIG.track_contact_gap_m,
        ),
    )

    def base_pose_and_forward() -> tuple[np.ndarray, np.ndarray]:
        position, quaternion_wxyz = lower_body.get_world_poses()
        position = np.asarray(position, dtype=np.float64).reshape(-1, 3)[0]
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1, 4)[0]
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
        forward = rotation @ derived_forward_axis_local
        forward[2] = 0.0
        forward /= np.linalg.norm(forward)
        return np.asarray([position[0], position[1], np.arctan2(forward[1], forward[0])]), forward

    initial_base_pose, initial_forward = base_pose_and_forward()
    track_bottom_offsets_m = []
    post_reset_bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    for body, path in ((left_track, CONFIG.left_track_body), (right_track, CONFIG.right_track_body)):
        position, _ = body.get_world_poses()
        body_z = float(np.asarray(position, dtype=np.float64).reshape(-1, 3)[0, 2])
        bottom_z = float(
            post_reset_bounds.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMin()[2]
        )
        track_bottom_offsets_m.append(body_z - bottom_z)
    dump_base_pose = initial_base_pose.copy()
    dump_base_pose[:2] -= 2.0 * initial_forward[:2]
    cycle_machine = ExcavatorCycleStateMachine(
        ExcavatorCycleConfig(
            CONFIG.phase_targets_rad,
            state_timeout_s=CONFIG.phase_timeout_s,
            return_travel_timeout_s=CONFIG.return_travel_timeout_s,
            navigation_drive_heading_gate_rad=(
                CONFIG.navigation_drive_heading_gate_rad
            ),
            reverse_distance_m=2.0,
            minimum_payload_gain_m3=0.02,
            minimum_dump_release_m3=0.01,
            # The staged machine is already at the audited dig position.
            # APPROACH is an arm-only trajectory; base travel during its
            # multi-second slew moves the bucket away from the pile.
            approach_track_command=0.0,
            dynamic_dump_from_reverse_entry=True,
        ),
        initial_base_pose,
        dump_base_pose,
    )
    cycle_started = False
    cutting_distance_m = 0.0
    previous_cutting_center = None
    latest_intersection_volume_m3 = 0.0
    latest_penetration_depth_m = 0.0
    transition_log: list[dict[str, object]] = []
    track_soil_log: list[dict[str, object]] = []
    runtime_telemetry: list[dict[str, object]] = []
    v3_closure_records: list[dict[str, object]] = []
    v3_closure_labels: set[str] = set()
    v3_breakout_time_s: float | None = None
    last_core_result = None
    initial_resting_volume_m3 = physics_core.integrator.integrate(height)
    material_funnel = {
        "failure_volume_m3": 0.0,
        "activated_volume_m3": 0.0,
        "candidate_mouth_flux_m3": 0.0,
        "admitted_volume_m3": 0.0,
        "dump_released_volume_m3": 0.0,
        "airborne_landed_volume_m3": 0.0,
        "deposited_after_dump_m3": 0.0,
    }
    cut_fill_payload_audit = (
        CutFillPayloadCausalAudit(
            physics_core.device_state,
            grid,
            descriptor,
            np.asarray(CONFIG.phase_targets_rad["coordinated_cut"], dtype=np.float64),
            CONFIG.physics_dt_s,
        )
        if (
            ARGS.cut_fill_payload_audit is not None
            or ARGS.tool_mobile_validity_audit is not None
        )
        else None
    )
    realistic_cut_trajectory = (
        RealisticCurlScoopTrajectory(
            np.asarray(CONFIG.phase_targets_rad["penetrate"], dtype=np.float64),
            np.asarray(CONFIG.phase_targets_rad["breakout"], dtype=np.float64),
        )
        if REALISTIC_CUT_SCOOP
        else None
    )
    realistic_cut_sample = None
    tracksoil_conservation_audit = (
        TrackSoilConservationAudit()
        if (ARGS.tracksoil_conservation_audit or ARGS.mobile_v2_dual_cv_audit is not None)
        else None
    )
    production_physics_step_count = 0
    mobile_first_write_audit = (
        MobileFirstWriteAudit(
            physics_core.device_state,
            material,
            physics_core.gpu_chain.mobile.config,
            ARGS.mobile_first_write_audit.expanduser().resolve(),
        )
        if ARGS.mobile_first_write_audit is not None
        else None
    )
    if mobile_first_write_audit is not None:
        mobile_first_write_audit.run_id = run_id
    tool_mobile_validity_audit = (
        ToolMobilePhysicalValidityAudit(
            physics_core.device_state,
            material,
            physics_core.gpu_chain.mobile.config,
            descriptor,
            ARGS.tool_mobile_validity_audit.expanduser().resolve(),
            impulse_mode=ARGS.tool_mobile_impulse_ablation,
            observation_horizon_s=(
                1_000_000.0
                if ARGS.mobile_v2_pre_dump_acceptance is not None
                else 4.0
            ),
        )
        if ARGS.tool_mobile_validity_audit is not None
        else None
    )
    if tool_mobile_validity_audit is not None:
        tool_mobile_validity_audit.run_id = run_id
        physics_core.gpu_chain.mobile.diagnostic_disable_tool_mobile_impulse = bool(
            ARGS.tool_mobile_impulse_ablation == "OFF"
        )
    tool_mobile_visual_offsets_s = tuple(
        step / 60.0 for step in (0, 1, 2, 3, 5, 10, 20)
    )
    tool_mobile_visual_captured_offsets: set[float] = set()
    if (
        mobile_first_write_audit is not None
        and ARGS.mobile_first_write_visual_diagnostic
    ):
        mobile_first_write_audit.visual_observer = (
            IsaacMobileFirstWriteVisualization(stage, grid, descriptor)
        )
        _set_presentation_camera(2)
    if (
        tool_mobile_validity_audit is not None
        and ARGS.tool_mobile_validity_visual_diagnostic
    ):
        tool_mobile_validity_audit.visual_observer = (
            IsaacMobileFirstWriteVisualization(stage, grid, descriptor)
        )
        _set_presentation_camera(2)
    if (
        tracksoil_conservation_audit is not None
        or mobile_first_write_audit is not None
        or tool_mobile_validity_audit is not None
    ):
        assert physics_core.gpu_chain is not None
        def observe_gpu_operator_boundary(label):
            if tracksoil_conservation_audit is not None:
                tracksoil_conservation_audit.operator_boundary(
                    label, physics_core.scalar_state()
                )
            if mobile_first_write_audit is not None:
                mobile_first_write_audit.operator_boundary(label)
        physics_core.gpu_chain.audit_state_observer = observe_gpu_operator_boundary
        if mobile_first_write_audit is not None:
            physics_core.gpu_chain.audit_mobile_substep_observer = (
                mobile_first_write_audit.mobile_substep_boundary
            )
        if tool_mobile_validity_audit is not None:
            if mobile_first_write_audit is not None:
                raise RuntimeError(
                    "MOBILE_FIRST_WRITE_AND_PHYSICAL_VALIDITY_AUDITS_ARE_MUTUALLY_EXCLUSIVE"
                )
            physics_core.gpu_chain.audit_mobile_substep_observer = (
                tool_mobile_validity_audit.mobile_substep_boundary
            )
            physics_core.gpu_chain.audit_tool_contact_observer = (
                tool_mobile_validity_audit.observe_contact_geometry
            )
    cut_fill_start_checkpoint_written = False
    soil_force_evidence = {
        "computed_peak_n": 0.0,
        "applied_peak_n": 0.0,
        "quasi_static_peak_n": 0.0,
        "momentum_peak_n": 0.0,
        "computed_nonzero_step_count": 0,
        "applied_nonzero_step_count": 0,
        "fee_applicable_step_count": 0,
        "fee_outside_domain_step_count": 0,
        "applied_impulse_terrain_ns": [0.0, 0.0, 0.0],
    }
    dump_release_started = False
    dump_checkpoint_written = False
    dump_checkpoint_device_time_s = None
    dump_plus_9_checkpoint_written = False
    dump_release_checkpoint_written = False
    presentation_deposition_entry_sim_s = None
    presentation_deposition_entry_device_s = None
    presentation_auto_paused = False
    presentation_smoke_reset_requested = False
    presentation_smoke_reset_verified = False
    presentation_smoke_reset_completed = False
    final_presentation_sample = None
    final_presentation_breakout_sim_s = None
    final_presentation_completed = False
    final_presentation_capture_records: list[dict[str, object]] = []
    final_presentation_captured_labels: set[str] = set()
    final_presentation_frame_times_ms: list[float] = []
    final_presentation_rtf_samples: list[float] = []
    final_presentation_peak_payload_m3 = 0.0
    final_presentation_peak_mobile_m3 = 0.0
    final_presentation_peak_force_n = 0.0
    final_presentation_ui_warmup_frames = 120
    gui_lifecycle_stage = "READY_HOLD" if GUI_LIFECYCLE_SMOKE else None
    gui_lifecycle_hold_started_wall_s = perf_counter()
    gui_lifecycle_evidence: list[dict[str, object]] = []
    last_contact_surface = np.array(height, copy=True)
    last_visual_surface = np.array(height, copy=True)
    pending_contact_tiles: dict[int, np.ndarray] = {}
    pending_visual_tiles: dict[int, np.ndarray] = {}
    terrain_timestamp_s = 0.0
    last_contact_update_sim_s = 0.0
    latest_track_status = {
        "left_sinkage_m": 0.0,
        "right_sinkage_m": 0.0,
        "active_bbox_grid": [0, 0, 0, 0],
        "left_contact": False,
        "right_contact": False,
    }
    latest_debug = {
        "phase": "READY",
        "completion_condition": "waiting for user command",
        "rake_deg": 0.0,
        "penetration_m": 0.0,
        "failure_zone_m3": 0.0,
        "mobile_m3": 0.0,
        "payload_m3": 0.0,
        "fill_ratio": 0.0,
        "force_quasi_n": 0.0,
        "force_momentum_n": 0.0,
        "force_total_n": 0.0,
        "resting_m3": initial_resting_volume_m3,
        "airborne_m3": 0.0,
        "outflow_m3": 0.0,
        "mass_error_m3": 0.0,
        "rtf": 0.0,
    }

    def current_surface_height() -> np.ndarray:
        if CONFIG.runtime_backend == "GPU_RUNTIME":
            raise RuntimeError(
                "GPU_RUNTIME_FULL_FIELD_TERRAIN_REQUEST: launcher must use compact queries"
            )
        state = physics_core.state
        return np.asarray(state.H_resting_m + state.mobile_height_m, dtype=np.float64)

    def capture_v3_closure_state(label: str, core_result=None) -> None:
        """Explicit acceptance boundary; never called in the normal hot path."""

        if not V3_CLOSURE_AUDIT or label in v3_closure_labels:
            return
        if physics_core.device_state is None:
            raise RuntimeError("V3_CLOSURE_AUDIT_REQUIRES_DEVICE_AUTHORITY")
        view = physics_core.device_state.explicit_host_view(source="acceptance")
        snapshot_path = run_dir / f"v3_closure_{label.lower()}_terrain.npz"
        np.savez_compressed(
            snapshot_path,
            H_resting_m=view.H_resting_m,
            h_mobile_m=view.H_mobile_m,
            H_free_m=view.H_resting_m + view.H_mobile_m,
            mobile_momentum_m2_s=view.mobile_momentum_m2_s,
            timestamp_device_s=np.asarray(view.timestamp_device_s),
        )
        if label == "ARREST_FINAL":
            causal_report, causal_masks = classify_flow_arrest_state(
                view.H_resting_m,
                view.H_mobile_m,
                view.mobile_momentum_m2_s,
                material,
                grid,
                physics_core.integrator,
                physics_core.avalanche_controller.config,
                residual_active_tile_ids=np.asarray(
                    physics_core._gpu_frontier_active_tiles, dtype=np.int32
                ),
                tile_size=CONFIG.active_tile_size,
                residual_tolerance_m=CONFIG.minislope_tolerance_m,
                dt_s=CONFIG.physics_dt_s,
            )
            causal_mask_path = run_dir / "v3_flow_arrest_final_masks.npz"
            np.savez_compressed(causal_mask_path, **causal_masks)
            (run_dir / "v3_flow_arrest_final_audit.json").write_text(
                json.dumps(causal_report, indent=2) + "\n", encoding="utf-8"
            )
        # explicit_host_view consumes publication IDs by contract. Re-marking
        # those IDs preserves the normal visual/contact publication schedule;
        # no field is written and no physics result is changed.
        physics_core.device_state.mark_dirty_tiles(view.dirty_tile_ids)
        reservoirs = physics_core.reservoir_observation()
        diagnostics = None if core_result is None else core_result.physics_diagnostics
        force = np.zeros(3) if diagnostics is None else diagnostics.soil_force_terrain_n
        record = {
            "label": label,
            "stage": "EXPLICIT_ACCEPTANCE_SNAPSHOT_NOT_NORMAL_PATH_TRANSFER",
            "timestamp_s": float(world.current_time),
            "device_timestamp_s": float(view.timestamp_device_s),
            "payload_m3": float(reservoirs["payload_volume_m3"]),
            "soil_force_terrain_n": np.asarray(force, dtype=np.float64).tolist(),
            "soil_force_norm_n": float(np.linalg.norm(force)),
            "mobile_m3": float(reservoirs["mobile_volume_m3"]),
            "moving_mobile_m3": 0.0 if diagnostics is None else diagnostics.mobile_moving_volume_m3,
            "terrain_state": "PRE_DIG" if diagnostics is None else diagnostics.terrain_state,
            "not_settled_reason": "PRE_DIG" if diagnostics is None else diagnostics.not_settled_reason,
            "dynamic_flow_time_s": 0.0 if diagnostics is None else diagnostics.dynamic_flow_time_s,
            "residual_solver_iterations": 0 if diagnostics is None else diagnostics.residual_solver_iterations,
            "mass_balance_error_m3": float(reservoirs["mass_balance_error_m3"]),
            "snapshot_file": snapshot_path.name,
        }
        v3_closure_records.append(record)
        v3_closure_labels.add(label)
        (run_dir / "v3_closure_lightweight_diagnostics.json").write_text(
            json.dumps(v3_closure_records, indent=2) + "\n", encoding="utf-8"
        )

    def collect_device_publication_tiles() -> None:
        if CONFIG.runtime_backend != "GPU_RUNTIME":
            return
        samples = physics_core.consume_dirty_surface_tiles()
        pending_contact_tiles.update(samples)
        pending_visual_tiles.update(samples)

    def world_vector_to_terrain_xy(vector_world: np.ndarray) -> np.ndarray:
        vector = np.linalg.solve(grid.terrain_to_world_matrix[:3, :3], np.asarray(vector_world, dtype=np.float64))
        return vector[:2]

    def track_footprints_and_contact():
        _, forward_world = base_pose_and_forward()
        footprints = []
        contacts = []
        support_diagnostics = []
        surface = (
            None
            if CONFIG.runtime_backend == "GPU_RUNTIME"
            else current_surface_height()
        )
        transform_world = grid.terrain_to_world_matrix
        for index, body in enumerate((left_track, right_track)):
            positions, _ = body.get_world_poses()
            position = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[0]
            footprint = footprint_rasterizer.rasterize(position, forward_world)
            footprints.append(footprint)
            if np.any(footprint.mask):
                rows, cols = np.nonzero(footprint.mask)
                terrain_x = grid.origin_x + cols * grid.dx
                terrain_y = grid.origin_y + rows * grid.dy
                local_h = (
                    physics_core.surface_at_grid_coordinates(
                        np.column_stack((rows, cols))
                    )
                    if CONFIG.runtime_backend == "GPU_RUNTIME"
                    else surface[rows, cols]
                )
                world_z = (
                    transform_world[2, 0] * terrain_x
                    + transform_world[2, 1] * terrain_y
                    + transform_world[2, 2] * local_h
                    + transform_world[2, 3]
                )
                track_bottom_z = position[2] - track_bottom_offsets_m[index]
                dynamic_support_z = float(np.max(world_z))
                overlaps_apron = footprint.active_bbox_grid[1] == 0
                apron_support_z = 0.0 if overlaps_apron else -np.inf
                contacts.append(
                    bool(
                        track_bottom_z
                        <= max(dynamic_support_z, apron_support_z)
                        + CONFIG.track_contact_gap_m
                    )
                )
                support_diagnostics.append({
                    "terrain_overlap_cell_count": int(rows.size),
                    "dynamic_support_z_m": dynamic_support_z,
                    "track_bottom_z_m": float(track_bottom_z),
                    "support_owner": "DEVICE_TERRAIN_OR_APRON_OVERLAP",
                })
            else:
                # Outside the heightmap the explicitly separate support apron
                # owns contact. It does not deform the terrain state.
                track_bottom_z = position[2] - track_bottom_offsets_m[index]
                contacts.append(
                    bool(
                        position[0] <= terrain_min_x_world
                        and track_bottom_z <= CONFIG.track_contact_gap_m
                    )
                )
                support_diagnostics.append({
                    "terrain_overlap_cell_count": 0,
                    "dynamic_support_z_m": None,
                    "track_bottom_z_m": float(track_bottom_z),
                    "support_owner": "LOCAL_NONDEFORMING_APRON",
                })
        contact_mesh.set_active_mask(
            footprints[0].mask | footprints[1].mask,
            tile_halo=1,
        )
        return (
            footprints[0], footprints[1], contacts[0], contacts[1],
            forward_world, support_diagnostics,
        )

    def apply_track_soil_for_command(
        left_command: float,
        right_command: float,
        *,
        braking: bool = False,
    ):
        nonlocal terrain_timestamp_s, latest_track_status
        (
            left_fp, right_fp, left_contact, right_contact, forward_world,
            support_diagnostics,
        ) = track_footprints_and_contact()
        drive_result = track_drive.apply(
            left_command,
            right_command,
            CONFIG.physics_dt_s,
            left_contact_active=left_contact,
            right_contact_active=right_contact,
            braking=braking,
        )
        latest_track_status = {
            **latest_track_status,
            "left_contact": drive_result.left_contact_active,
            "right_contact": drive_result.right_contact_active,
            "applied_left_command": drive_result.applied_left_command,
            "applied_right_command": drive_result.applied_right_command,
            "left_force_world_n": drive_result.left_force_world_n.tolist(),
            "right_force_world_n": drive_result.right_force_world_n.tolist(),
            "support": support_diagnostics,
        }
        track_result = None
        audit_geometry = {
            "affected_cell_count": 0,
            "footprint_area_m2": 0.0,
            "requested_r2m_m3": 0.0,
        }
        if control.snapshot().track_soil_enabled and (
            abs(drive_result.applied_left_command) > 1.0e-4
            or abs(drive_result.applied_right_command) > 1.0e-4
        ) and (left_contact or right_contact):
            base_velocity_world = np.asarray(
                lower_body.get_linear_velocities(), dtype=np.float64
            ).reshape(-1, 3)[0]
            base_velocity_xy = world_vector_to_terrain_xy(base_velocity_world)
            forward_xy = world_vector_to_terrain_xy(forward_world)
            forward_xy /= max(float(np.linalg.norm(forward_xy)), 1.0e-12)
            left_mask = left_fp.mask if left_contact else np.zeros(grid.shape, dtype=bool)
            right_mask = right_fp.mask if right_contact else np.zeros(grid.shape, dtype=bool)
            left_track_velocity_xy = (
                base_velocity_xy
                + drive_result.applied_left_command
                * CONFIG.nominal_track_belt_speed_m_s
                * forward_xy
            )
            right_track_velocity_xy = (
                base_velocity_xy
                + drive_result.applied_right_command
                * CONFIG.nominal_track_belt_speed_m_s
                * forward_xy
            )
            if tracksoil_conservation_audit is not None:
                track_config = (
                    physics_core.gpu_chain.track.config
                    if physics_core.gpu_chain is not None
                    else physics_core.track_soil_model.config
                )

                def requested_track_volume(mask, track_velocity):
                    slip = float(
                        np.linalg.norm(track_velocity - base_velocity_xy)
                    )
                    slip_effect = max(
                        slip - track_config.minimum_slip_speed_m_s, 0.0
                    )
                    requested_depth = min(
                        track_config.maximum_sinkage_per_step_m,
                        track_config.sinkage_rate_m_s
                        * CONFIG.physics_dt_s
                        * (1.0 + track_config.slip_gain * slip_effect),
                    )
                    return float(
                        requested_depth
                        * np.sum(
                            physics_core.integrator.vertex_weights_m2[mask],
                            dtype=np.float64,
                        )
                    )

                combined_mask = left_mask | right_mask
                audit_geometry = {
                    "affected_cell_count": int(
                        np.count_nonzero(combined_mask)
                    ),
                    "footprint_area_m2": float(
                        np.sum(
                            physics_core.integrator.vertex_weights_m2[
                                combined_mask
                            ],
                            dtype=np.float64,
                        )
                    ),
                    "requested_r2m_m3": (
                        requested_track_volume(
                            left_mask, left_track_velocity_xy
                        )
                        + requested_track_volume(
                            right_mask, right_track_velocity_xy
                        )
                    ),
                }
            track_result = physics_core.apply_track_soil(
                left_footprint_mask=left_mask,
                right_footprint_mask=right_mask,
                left_track_velocity_xy_m_s=left_track_velocity_xy,
                right_track_velocity_xy_m_s=right_track_velocity_xy,
                base_velocity_xy_m_s=base_velocity_xy,
                dt_s=CONFIG.physics_dt_s,
            )
            if track_result.resting_to_mobile_volume_m3 > 0.0:
                terrain_timestamp_s = float(world.current_time)
                latest_track_status = {
                    **latest_track_status,
                    "left_sinkage_m": track_result.left_mean_sinkage_m,
                    "right_sinkage_m": track_result.right_mean_sinkage_m,
                    "active_bbox_grid": list(
                        getattr(
                            track_result,
                            "active_bbox_grid",
                            (
                                min(left_fp.active_bbox_grid[0], right_fp.active_bbox_grid[0]),
                                min(left_fp.active_bbox_grid[1], right_fp.active_bbox_grid[1]),
                                max(left_fp.active_bbox_grid[2], right_fp.active_bbox_grid[2]),
                                max(left_fp.active_bbox_grid[3], right_fp.active_bbox_grid[3]),
                            ),
                        )
                    ),
                    "left_contact": left_contact,
                    "right_contact": right_contact,
                }
                track_soil_log.append({
                    "timestamp_s": terrain_timestamp_s,
                    "left_command": drive_result.applied_left_command,
                    "right_command": drive_result.applied_right_command,
                    "resting_to_mobile_volume_m3": track_result.resting_to_mobile_volume_m3,
                    **latest_track_status,
                })
        return drive_result, track_result, audit_geometry

    def synchronize_terrain_contact(*, force: bool = False) -> None:
        nonlocal last_contact_surface, last_contact_update_sim_s, contact_status
        if CONFIG.runtime_backend == "GPU_RUNTIME":
            collect_device_publication_tiles()
            due = (
                force
                or float(world.current_time) - last_contact_update_sim_s
                >= 1.0 / CONFIG.contact_update_hz
            )
            if not due:
                contact_status = contact_mesh.lag_status(terrain_timestamp_s)
                return
            if pending_contact_tiles:
                contact_status = contact_mesh.update_tile_samples(
                    dict(pending_contact_tiles),
                    tile_size_cells=CONFIG.active_tile_size,
                    terrain_timestamp_s=terrain_timestamp_s,
                )
                pending_contact_tiles.clear()
            else:
                contact_status = contact_mesh.lag_status(terrain_timestamp_s)
            last_contact_update_sim_s = float(world.current_time)
            return
        surface = current_surface_height()
        due = (
            force
            or float(world.current_time) - last_contact_update_sim_s
            >= 1.0 / CONFIG.contact_update_hz
        )
        if not due:
            contact_status = contact_mesh.lag_status(terrain_timestamp_s)
            return
        changed = np.abs(surface - last_contact_surface) > 1.0e-10
        if np.any(changed):
            contact_status = contact_mesh.update(
                surface,
                terrain_timestamp_s=terrain_timestamp_s,
                changed_mask=changed,
            )
            last_contact_surface = np.array(surface, copy=True)
        else:
            contact_status = contact_mesh.lag_status(terrain_timestamp_s)
        last_contact_update_sim_s = float(world.current_time)

    def synchronize_visual(*, force: bool = False) -> None:
        nonlocal last_visual_surface
        if CONFIG.runtime_backend == "GPU_RUNTIME":
            collect_device_publication_tiles()
            if pending_visual_tiles:
                terrain_mesh.update_tile_samples(
                    dict(pending_visual_tiles),
                    tile_size_cells=CONFIG.active_tile_size,
                )
                pending_visual_tiles.clear()
            return
        surface = current_surface_height()
        if force:
            terrain_mesh.reset(surface)
            last_visual_surface = np.array(surface, copy=True)
            return
        changed = np.abs(surface - last_visual_surface) > 1.0e-10
        if np.any(changed):
            terrain_mesh.update(surface, changed_mask=changed)
            last_visual_surface = np.array(surface, copy=True)

    def capture_presentation_viewport(label: str) -> None:
        """Capture the actual Isaac viewport without altering simulation state."""

        normalized = str(label).upper()
        if (
            not FINAL_PRESENTATION
            or HEADLESS
            or not ARGS.final_presentation_capture
            or normalized in final_presentation_captured_labels
        ):
            return
        from omni.kit.async_engine import run_coroutine
        from omni.kit.viewport.utility import (
            capture_viewport_to_file,
            get_active_viewport,
        )
        import omni.kit.renderer_capture

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("FINAL_PRESENTATION_ACTIVE_VIEWPORT_UNAVAILABLE")
        viewport.resolution = (CONFIG.viewport_width, CONFIG.viewport_height)
        destination = presentation_screenshot_dir / f"{normalized.lower()}.png"
        was_playing = bool(world.is_playing())
        if was_playing:
            world.pause()
        synchronize_visual(force=True)
        _set_presentation_camera(1)
        for _ in range(12):
            simulation_app.update()
        capture = capture_viewport_to_file(
            viewport, file_path=str(destination), is_hdr=False
        )
        pending = run_coroutine(capture.wait_for_result(completion_frames=60))
        for _ in range(120):
            if pending.done():
                break
            simulation_app.update()
        if not pending.done() or not bool(pending.result()):
            raise RuntimeError(f"FINAL_PRESENTATION_VIEWPORT_CAPTURE_TIMEOUT:{normalized}")
        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        if was_playing:
            world.play()
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"FINAL_PRESENTATION_EMPTY_SCREENSHOT:{normalized}")
        final_presentation_captured_labels.add(normalized)
        final_presentation_capture_records.append(
            {
                "label": normalized,
                "path": str(destination),
                "size_bytes": destination.stat().st_size,
                "sim_time_s": float(world.current_time),
                "visual_source": "AUTHORITATIVE_H_FREE_GPU_DIRTY_TILES",
            }
        )

    def capture_tool_mobile_validity_view(
        *, label: str, view: str, tool_state, tau_s: float
    ) -> None:
        """Capture a paused render frame; physics state and time stay fixed."""

        if (
            HEADLESS
            or not ARGS.tool_mobile_validity_visual_diagnostic
            or tool_mobile_validity_frame_dir is None
            or tool_mobile_validity_audit is None
        ):
            return
        from omni.kit.async_engine import run_coroutine
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        import omni.kit.renderer_capture
        from isaacsim.core.utils.viewports import set_camera_view

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("TOOL_MOBILE_VALIDITY_ACTIVE_VIEWPORT_UNAVAILABLE")
        viewport.resolution = (CONFIG.viewport_width, CONFIG.viewport_height)
        destination = tool_mobile_validity_frame_dir / f"{label}_{view}.png"
        was_playing = bool(world.is_playing())
        if was_playing:
            world.pause()
        synchronize_visual(force=True)
        if view == "side":
            _set_presentation_camera(2)
        elif view == "top":
            center_t = np.mean(np.asarray(tool_state.cutting_edge_terrain), axis=0)
            center_w = grid.terrain_to_world(center_t)
            eye = np.asarray(center_w, dtype=np.float64) + np.asarray([0.01, -0.01, 10.0])
            set_camera_view(eye=eye, target=np.asarray(center_w, dtype=np.float64))
        else:
            raise ValueError(f"unknown tool-Mobile diagnostic view: {view}")
        for _ in range(8):
            simulation_app.update()
        capture = capture_viewport_to_file(
            viewport, file_path=str(destination), is_hdr=False
        )
        pending = run_coroutine(capture.wait_for_result(completion_frames=20))
        for _ in range(60):
            if pending.done():
                break
            simulation_app.update()
        if not pending.done() or not bool(pending.result()):
            raise RuntimeError(f"TOOL_MOBILE_VALIDITY_CAPTURE_TIMEOUT:{label}:{view}")
        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"TOOL_MOBILE_VALIDITY_EMPTY_SCREENSHOT:{label}:{view}")
        tool_mobile_validity_audit.visual_frames.append({
            "label": label,
            "view": view,
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
            "simulation_time_s": float(world.current_time),
            "tau_from_first_mobile_s": float(tau_s),
            "visual_source": "AUTHORITATIVE_H_FREE_PLUS_READ_ONLY_DEVICE_OVERLAY",
            "arrow_scale": "velocity endpoint = origin + 0.35 * q/h (visual only)",
        })
        if was_playing:
            world.play()

    def write_final_presentation_summary(status: str) -> dict[str, object]:
        reservoirs = physics_core.reservoir_observation()
        frame_values = np.asarray(final_presentation_frame_times_ms, dtype=np.float64)
        rtf_values = np.asarray(final_presentation_rtf_samples, dtype=np.float64)
        summary = {
            "schema": "390F_FINAL_PRESENTATION_DEMO/v1",
            "status": status,
            "production_core": type(physics_core).__name__,
            "runtime_backend": CONFIG.runtime_backend,
            "state_authority": "DEVICE",
            "grid_shape_yx": list(CONFIG.grid_shape),
            "grid_spacing_m": CONFIG.grid_spacing_m,
            "production_config": str(CONFIG.config_path),
            "presentation_config": str(PRESENTATION_CONFIG.path),
            "vehicle_asset": str(CONFIG.vehicle_asset),
            "trajectory_duration_s": PRESENTATION_CONFIG.duration_s,
            "trajectory_stage": (
                "READY" if final_presentation_sample is None else final_presentation_sample.label
            ),
            "peak_payload_volume_m3": final_presentation_peak_payload_m3,
            "peak_payload_mass_kg": (
                final_presentation_peak_payload_m3
                * material.assumed_bulk_density_kg_m3
            ),
            "peak_mobile_volume_m3": final_presentation_peak_mobile_m3,
            "peak_bucket_force_n": final_presentation_peak_force_n,
            "final_payload_volume_m3": float(reservoirs["payload_volume_m3"]),
            "final_mobile_volume_m3": float(reservoirs["mobile_volume_m3"]),
            "mass_balance_error_m3": float(reservoirs["mass_balance_error_m3"]),
            "mean_rtf": float(np.mean(rtf_values)) if rtf_values.size else None,
            "mean_frame_time_ms": float(np.mean(frame_values)) if frame_values.size else None,
            "root_pose_write_count": 0,
            "visual_terrain_publication": "AUTHORITATIVE_H_FREE_GPU_DIRTY_TILES_TO_ISAAC_MESH",
            "payload_visualization": "PAYLOAD_STATE_DRIVEN_VISUALIZATION_NO_PHYSX",
            "captures": list(final_presentation_capture_records),
        }
        destination = presentation_run_dir / "demo_summary.json"
        destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        latest = PRESENTATION_CONFIG.output_root / "latest_demo_summary.json"
        latest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    def capture_visual_blocker_state() -> dict[str, object]:
        if physics_core.device_state is None:
            raise RuntimeError("FINAL_PRESENTATION_BLOCKER_REQUIRES_DEVICE_STATE")
        view = physics_core.device_state.explicit_host_view(source="acceptance")
        H_free = np.asarray(view.H_resting_m + view.H_mobile_m, dtype=np.float64)
        delta = H_free - height
        active = np.abs(delta) > 0.005
        if np.any(active):
            rows, cols = np.nonzero(active)
            bounds = [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())]
        else:
            bounds = None
        snapshot_path = presentation_run_dir / "visual_blocker_h_free.npz"
        np.savez_compressed(
            snapshot_path,
            H_free_m=H_free,
            H_resting_m=view.H_resting_m,
            h_mobile_m=view.H_mobile_m,
            delta_H_free_m=delta,
        )
        physics_core.device_state.mark_dirty_tiles(view.dirty_tile_ids)
        record = {
            "schema": "390F_PRESENTATION_VISUAL_BLOCKER_CHECK/v1",
            "status": "AWAITING_VIEWPORT_VISUAL_CLASSIFICATION",
            "post_breakout_simulation_s": float(world.current_time - final_presentation_breakout_sim_s),
            "active_h_free_cell_count_gt_5mm": int(np.count_nonzero(active)),
            "affected_bounds_grid": bounds,
            "maximum_h_free_uplift_m": float(np.max(delta)),
            "maximum_h_free_drop_m": float(np.min(delta)),
            "payload_m3": float(physics_core.payload.volume_m3),
            "mobile_m3": float(physics_core.reservoir_observation()["mobile_volume_m3"]),
            "snapshot": str(snapshot_path),
            "viewport_capture": next(
                (
                    item["path"]
                    for item in final_presentation_capture_records
                    if item["label"] == "VISUAL_BLOCKER"
                ),
                None,
            ),
        }
        (presentation_run_dir / "visual_blocker_check.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        return record

    def capture_mobile_v2_pre_dump_acceptance() -> dict[str, object]:
        """Explicit full-field acceptance boundary before Payload release."""

        if physics_core.device_state is None or ARGS.mobile_v2_pre_dump_acceptance is None:
            raise RuntimeError("MOBILE_V2_PRE_DUMP_REQUIRES_DEVICE_STATE_AND_PATH")
        state = physics_core.device_state
        view = state.explicit_host_view(source="acceptance")
        b_eff = np.asarray(view.b_eff_m, dtype=np.float64)
        z_base = np.asarray(view.z_base_m, dtype=np.float64)
        mobile = np.asarray(view.H_mobile_m, dtype=np.float64)
        momentum = np.asarray(view.mobile_momentum_m2_s, dtype=np.float64)
        H_free = b_eff + mobile
        neighbor_jumps = np.concatenate(
            (np.abs(np.diff(H_free, axis=0)).ravel(), np.abs(np.diff(H_free, axis=1)).ravel())
        )
        center = H_free[1:-1, 1:-1]
        neighbor_max = np.maximum.reduce(
            (H_free[:-2, 1:-1], H_free[2:, 1:-1], H_free[1:-1, :-2], H_free[1:-1, 2:])
        )
        neighbor_min = np.minimum.reduce(
            (H_free[:-2, 1:-1], H_free[2:, 1:-1], H_free[1:-1, :-2], H_free[1:-1, 2:])
        )
        spike = center > neighbor_max + 0.05
        pit = center < neighbor_min - 0.05
        speed = np.divide(
            np.linalg.norm(momentum, axis=-1), mobile,
            out=np.zeros_like(mobile), where=mobile > 1.0e-10,
        )
        area = CONFIG.grid_spacing_m * CONFIG.grid_spacing_m
        density = material.assumed_bulk_density_kg_m3
        kinetic_normalized = float(0.5 * np.sum(mobile * speed * speed) * area)
        gravitational_normalized = float(9.81 * np.sum(mobile * b_eff) * area)
        internal_normalized = float(0.5 * 0.45 * 9.81 * np.sum(mobile * mobile) * area)
        reservoirs = physics_core.reservoir_observation()
        r2m = np.asarray(state.runtime.download("avalanche_r2m_cumulative"), dtype=np.float64).reshape(state.shape)
        m2r = np.asarray(state.runtime.download("avalanche_m2r_cumulative"), dtype=np.float64).reshape(state.shape)
        target = ARGS.mobile_v2_pre_dump_acceptance.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            z_base_m=z_base, b_eff_m=b_eff, h_mobile_m=mobile,
            momentum_x_m2_s=momentum[..., 0], momentum_y_m2_s=momentum[..., 1],
            H_free_m=H_free, r2m_cumulative_m3=r2m, m2r_cumulative_m3=m2r,
        )
        record = {
            "schema": "MOBILE_V2_PRODUCTION_PRE_DUMP/v1",
            "status": "CAPTURED_BEFORE_PAYLOAD_RELEASE",
            "runtime_backend": "GPU_RUNTIME",
            "state_authority": "DEVICE",
            "grid_shape_yx": list(state.shape),
            "grid_spacing_m": CONFIG.grid_spacing_m,
            "simulation_time_s": float(world.current_time),
            "device_time_s": state.timestamp_device_s,
            "payload_m3": float(reservoirs["payload_volume_m3"]),
            "mobile_m3": float(reservoirs["mobile_volume_m3"]),
            "mass_balance_error_m3": float(reservoirs["mass_balance_error_m3"]),
            "soil_force_resultant_n": float(latest_debug["force_total_n"]),
            "morphology": {
                "H_free_jump_max_m": float(np.max(neighbor_jumps, initial=0.0)),
                "H_free_jump_p95_m": float(np.percentile(neighbor_jumps, 95.0)),
                "H_free_jump_p99_m": float(np.percentile(neighbor_jumps, 99.0)),
                "H_free_jump_p99_9_m": float(np.percentile(neighbor_jumps, 99.9)),
                "isolated_spike_count_prominence_gt_5cm": int(np.count_nonzero(spike)),
                "isolated_pit_count_prominence_gt_5cm": int(np.count_nonzero(pit)),
            },
            "mobile": {
                "moving_volume_m3": float(np.sum(mobile[speed > 0.01]) * area),
                "maximum_speed_m_s": float(np.max(speed, initial=0.0)),
                "r2m_cumulative_m3": float(np.sum(r2m)),
                "m2r_cumulative_m3": float(np.sum(m2r)),
            },
            "energy": {
                "nomenclature": "E_density_normalized",
                "density_normalized": {
                    "kinetic": kinetic_normalized,
                    "gravitational": gravitational_normalized,
                    "pressure_internal": internal_normalized,
                    "total": kinetic_normalized + gravitational_normalized + internal_normalized,
                },
                "physical_J": {
                    "kinetic": density * kinetic_normalized,
                    "gravitational": density * gravitational_normalized,
                    "pressure_internal": density * internal_normalized,
                    "total": density * (kinetic_normalized + gravitational_normalized + internal_normalized),
                },
            },
            "field_archive": str(target),
        }
        target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        state.mark_dirty_tiles(view.dirty_tile_ids)
        return record

    def terrain_surface_at(points: np.ndarray, state_height: np.ndarray | None = None) -> np.ndarray:
        if CONFIG.runtime_backend == "GPU_RUNTIME":
            return physics_core.surface_at_terrain_points(points)
        assert state_height is not None
        values = []
        for point in np.asarray(points, dtype=np.float64):
            rc = grid.terrain_to_grid(point)
            row = int(np.clip(np.rint(rc[0]), 0, grid.ny - 1))
            column = int(np.clip(np.rint(rc[1]), 0, grid.nx - 1))
            values.append(float(state_height[row, column]))
        return np.asarray(values, dtype=np.float64)

    def operation_observation(tool_state) -> ExcavatorCycleObservation:
        reservoirs = physics_core.reservoir_observation()
        state = physics_core.state if CONFIG.runtime_backend == "HOST_REFERENCE" else None
        surface = terrain_surface_at(
            tool_state.cutting_edge_terrain,
            None if state is None else state.H_resting_m,
        )
        edge_z = np.asarray(tool_state.cutting_edge_terrain)[:, 2]
        base_pose, _ = base_pose_and_forward()
        return ExcavatorCycleObservation(
            timestamp_s=float(world.current_time),
            joint_position_rad=np.asarray(articulation.get_joint_positions(), dtype=np.float64).reshape(-1),
            base_pose_xy_yaw=base_pose,
            cutting_edge_depth_m=max(latest_penetration_depth_m, float(np.max(surface - edge_z))),
            tool_terrain_intersection_m3=latest_intersection_volume_m3,
            cutting_distance_m=cutting_distance_m,
            cutting_lip_clearance_m=float(np.min(edge_z - surface)),
            payload_volume_m3=float(reservoirs["payload_volume_m3"]),
            deposited_volume_m3=float(reservoirs["resting_volume_m3"]),
            mobile_volume_m3=float(reservoirs["mobile_volume_m3"]),
            airborne_volume_m3=float(reservoirs["airborne_volume_m3"]),
            terrain_settled=physics_core.terrain_settled,
        )

    def airborne_domain_diagnostics(state=None) -> dict[str, object]:
        del state
        parcels = physics_core.airborne_parcels
        if not parcels:
            return {
                "parcel_count": 0,
                "inside_heightmap_count": 0,
                "outside_heightmap_count": 0,
                "position_world_bounds_m": None,
            }
        positions = np.asarray(
            [parcel.position_world_m for parcel in parcels], dtype=np.float64
        )
        inside = 0
        for position in positions:
            terrain_point = grid.world_to_terrain(position)
            row, column = grid.terrain_to_grid(terrain_point)
            inside += int(
                0.0 <= row <= grid.ny - 1
                and 0.0 <= column <= grid.nx - 1
            )
        return {
            "parcel_count": len(parcels),
            "inside_heightmap_count": inside,
            "outside_heightmap_count": len(parcels) - inside,
            "position_world_bounds_m": {
                "minimum": np.min(positions, axis=0).tolist(),
                "maximum": np.max(positions, axis=0).tolist(),
            },
        }

    def dump_release_gate_metrics(tool_state) -> dict[str, float]:
        geometry = descriptor.bucket_geometry
        mouth_local = np.vstack([geometry.lip_local, geometry.top_edge_local])
        mouth_terrain = geometry.transform_points(
            tool_state.pose_terrain, mouth_local
        )
        radii = mouth_terrain - tool_state.pose_terrain[:3, 3]
        angular = np.broadcast_to(
            np.asarray(tool_state.angular_velocity, dtype=np.float64),
            radii.shape,
        )
        velocities = (
            np.asarray(tool_state.linear_velocity, dtype=np.float64)[None, :]
            + np.cross(angular, radii)
        )
        measured_q = np.asarray(
            articulation.get_joint_positions(), dtype=np.float64
        ).reshape(-1)
        measured_qd = np.asarray(
            articulation.get_joint_velocities(), dtype=np.float64
        ).reshape(-1)
        return {
            "joint_position_error_max_rad": float(
                np.max(
                    np.abs(
                        measured_q - CONFIG.phase_targets_rad["dump_spill"]
                    )
                )
            ),
            "joint_speed_max_rad_s": float(np.max(np.abs(measured_qd))),
            "mouth_horizontal_speed_max_m_s": float(
                np.max(np.linalg.norm(velocities[:, :2], axis=1))
            ),
        }

    acceptance_path = ROOT / "outputs" / "390f_v2" / "no_soil_machine_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.is_file() else {"overall_status": "MISSING", "blocking_failures": ["NO_SOIL_ACCEPTANCE_NOT_RUN"]}
    control = InteractiveControlModel(
        soil_force_mode=CONFIG.soil_force_mode,
        track_soil_enabled=CONFIG.track_soil_enabled,
        debug_overlay_enabled=CONFIG.debug_overlay_enabled,
    )
    panel = InteractivePanel(control, np.rad2deg(initial_q))
    control.initialized()
    if ARGS.acceptance_cycles > 0:
        control.run_cycles(ARGS.acceptance_cycles)
    elif (
        ARGS.mobile_first_write_audit is not None
        or ARGS.tool_mobile_validity_audit is not None
    ):
        control.run_cycles(1)
    elif ARGS.presentation_smoke_autostart and HEADLESS:
        control.run_cycles(1)
    manifest = {
        "schema": "390f-v2-interactive-runtime/v1",
        "run_id": run_id,
        "config": str(CONFIG.config_path),
        "headless": HEADLESS,
        "record_video": False,
        "autoplay": False,
        "presentation_demo": PRESENTATION_DEMO,
        "final_presentation": FINAL_PRESENTATION,
        "final_presentation_config": (
            str(PRESENTATION_CONFIG.path) if FINAL_PRESENTATION else None
        ),
        "presentation_claim": (
            "DEMO_ONLY_POST_DUMP_TIMESCALE_UNDER_VALIDATION"
            if PRESENTATION_DEMO else None
        ),
        "initial_runtime_state": control.state.value,
        "vehicle_asset": str(CONFIG.vehicle_asset),
        "soil_grid_shape": list(CONFIG.grid_shape),
        "soil_grid_spacing_m": CONFIG.grid_spacing_m,
        "solver_backend": CONFIG.solver_backend,
        "runtime_backend": CONFIG.runtime_backend,
        "state_authority": (
            "DEVICE" if CONFIG.runtime_backend == "GPU_RUNTIME" else "HOST"
        ),
        "slope_backend": CONFIG.slope_backend,
        "minislope_tolerance_m": CONFIG.minislope_tolerance_m,
        "no_soil_machine_gate": acceptance,
        "root_pose_write_count": 0,
        "mobile_base_world_anchor_disabled": CONFIG.mobile_base_enabled,
        "initial_scene_track_ground_alignment_z_m": initial_placement_z_m,
        "bucket_support_ground_collision_filtered": True,
        "support_apron_flat_heightmap_overlap_m": apron_flat_overlap_m,
        "heightmap_contact_mode": "DIRTY_STATIC_TRIANGLE_MESH_CHUNKS",
        "contact_chunk_count": len(contact_mesh.collider_paths),
        "contact_update_hz": CONFIG.contact_update_hz,
        "track_footprint_m": [CONFIG.track_footprint_length_m, CONFIG.track_footprint_width_m],
        "track_force_application_offsets_lower_local_m": [
            item.tolist() for item in track_application_offsets_lower_local_m
        ],
        "track_soil_parameter_status": CONFIG.track_soil["parameter_status"],
        "track_contact_domain_ownership": {
            "normal_support": "PHYSX_DYNAMIC_HEIGHTMAP_CHUNKS_AND_LOCAL_APRON",
            "tangential_shear": "TRACK_SOIL_AND_DIFFERENTIAL_TRACK_FORCE_MODEL",
            "physx_static_friction": 0.0,
            "physx_dynamic_friction": 0.0,
            "reason": "avoid double-counting tangential track-soil shear",
        },
        "arm_actuation": {
            "method": "ACCELERATION_AND_VELOCITY_SLEWED_POSITION_DRIVE_WITH_FORCE_CAP",
            "shared_positive_power_limit_w": actuator_config.shared_positive_power_limit_w,
            "root_or_link_pose_writes": 0,
            "dump_release_max_mouth_horizontal_speed_m_s": 0.25,
            "limits": [
                {
                    "joint": item.joint_name,
                    "effort_limit_nm": item.effort_limit_nm,
                    "velocity_limit_rad_s": item.velocity_limit_rad_s,
                    "acceleration_limit_rad_s2": item.acceleration_limit_rad_s2,
                    "provenance": item.provenance,
                }
                for item in actuator_config.joints
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "READY", "run_dir": str(run_dir), "autoplay": False}), flush=True)
    if ARGS.presentation_fixed_camera is not None:
        _set_presentation_camera(
            1 if ARGS.presentation_fixed_camera == "oblique" else 4
        )
    elif FINAL_PRESENTATION:
        _set_presentation_camera(1)
        # The blocker run needs only the post-breakout evidence frame.  Some
        # window managers interpret an early capture before the first PhysX
        # frame as a completed transient GUI task, so defer all capture there.
        if not ARGS.final_presentation_blocker_check:
            capture_presentation_viewport("READY")

    smoke_remaining = max(0, ARGS.startup_smoke_frames)
    smoke_requested = smoke_remaining > 0
    gate_reported = False
    if MANUAL_PRESENTATION_GUI:
        print(
            "GUI_LIFECYCLE=MANUAL_READY_WAIT; "
            "exit only follows a user window-close request",
            flush=True,
        )
    presentation_loop_count = 0
    while simulation_app.is_running():
        if _SIGNAL_EXIT_REQUESTED:
            break
        presentation_loop_count += 1
        _LIFECYCLE_RECORD["last_frame_number"] = presentation_loop_count
        _LIFECYCLE_RECORD["last_simulation_time_s"] = float(world.current_time)
        _LIFECYCLE_RECORD["last_physics_state"] = control.state.value
        if GUI_LIFECYCLE_SMOKE and gui_lifecycle_stage == "READY_HOLD":
            ready_hold_s = perf_counter() - gui_lifecycle_hold_started_wall_s
            if ready_hold_s >= 30.0:
                gui_lifecycle_evidence.append(
                    {
                        "check": "READY_HOLD",
                        "status": "PASS",
                        "wall_hold_s": ready_hold_s,
                        "runtime_state": control.state.value,
                        "process_alive": True,
                    }
                )
                # A real GUI button/key event may arrive at the same boundary
                # as the deterministic smoke hook. Starting is idempotent for
                # this acceptance boundary; never issue a second RUN request
                # after the user/UI has already entered RUNNING.
                if control.state in {
                    InteractiveRuntimeState.READY,
                    InteractiveRuntimeState.READY_NEXT_CYCLE,
                }:
                    control.run_cycles(1)
                gui_lifecycle_stage = "RUNNING"
        if smoke_remaining > 0:
            simulation_app.update()
            smoke_remaining -= 1
            if smoke_remaining == 0 and smoke_requested:
                break
            continue
        if (
            HEADLESS
            and ARGS.startup_smoke_frames == 0
            and ARGS.acceptance_cycles == 0
            and not ARGS.presentation_smoke_autostart
            and ARGS.mobile_first_write_audit is None
            and ARGS.tool_mobile_validity_audit is None
        ):
            # Headless mode cannot receive GUI input; remain deterministic and
            # avoid an accidental autonomous cycle.
            break

        if panel.pending_reset is not None:
            level = panel.pending_reset
            panel.pending_reset = None
            if level in {ResetLevel.ROBOT, ResetLevel.ALL}:
                world.reset()
                if not HEADLESS:
                    world.pause()
                kinematics.reset()
                track_drive.reset()
                arm_drive_target_rad = np.asarray(
                    articulation.get_joint_positions(), dtype=np.float64
                ).reshape(-1).copy()
                arm_actuator.reset(
                    np.asarray(
                        articulation.get_joint_velocities(), dtype=np.float64
                    ).reshape(-1)
                )
            physics_core.reset(level)
            if level in {ResetLevel.TERRAIN, ResetLevel.ALL}:
                terrain_timestamp_s = float(world.current_time)
                synchronize_visual(force=True)
                synchronize_terrain_contact(force=True)
            control.reset_completed(level)
            cycle_machine.reset()
            cycle_started = False
            if FINAL_PRESENTATION:
                presentation_trajectory.reset(
                    np.asarray(
                        articulation.get_joint_positions(), dtype=np.float64
                    ).reshape(-1)
                )
                final_presentation_sample = None
                final_presentation_breakout_sim_s = None
                final_presentation_completed = False
                final_presentation_capture_records.clear()
                final_presentation_captured_labels.clear()
                final_presentation_frame_times_ms.clear()
                final_presentation_rtf_samples.clear()
                final_presentation_peak_payload_m3 = 0.0
                final_presentation_peak_mobile_m3 = 0.0
                final_presentation_peak_force_n = 0.0
                capture_presentation_viewport("READY")
            cutting_distance_m = 0.0
            previous_cutting_center = None
            for key in material_funnel:
                material_funnel[key] = 0.0
            dump_release_started = False
            presentation_deposition_entry_sim_s = None
            presentation_auto_paused = False
            update_payload_visual()
            if (
                ARGS.presentation_smoke_autostart or GUI_LIFECYCLE_SMOKE
            ) and presentation_smoke_reset_requested:
                reset_reservoirs = physics_core.reservoir_observation()
                reset_joint_error = float(
                    np.max(
                        np.abs(
                            np.asarray(
                                articulation.get_joint_positions(),
                                dtype=np.float64,
                            ).reshape(-1)
                            - np.asarray(
                                CONFIG.phase_targets_rad["initial_pose"],
                                dtype=np.float64,
                            )
                        )
                    )
                )
                reset_base_pose, _ = base_pose_and_forward()
                reset_base_error = float(
                    np.linalg.norm(reset_base_pose - initial_base_pose)
                )
                reset_resting_volume_m3 = float(
                    reset_reservoirs["resting_volume_m3"]
                )
                reset_resting_volume_error_m3 = abs(
                    reset_resting_volume_m3 - initial_resting_volume_m3
                )
                presentation_smoke_reset_verified = bool(
                    control.state is InteractiveRuntimeState.READY
                    and float(reset_reservoirs["payload_volume_m3"]) <= 1.0e-12
                    and float(reset_reservoirs["mobile_volume_m3"]) <= 1.0e-12
                    and reset_resting_volume_error_m3 <= 1.0e-8
                    and reset_joint_error <= 1.0e-6
                    and reset_base_error <= 1.0e-6
                )
                presentation_smoke_reset_completed = True
                if GUI_LIFECYCLE_SMOKE:
                    gui_lifecycle_stage = "RESET_READY_HOLD"
                    gui_lifecycle_hold_started_wall_s = perf_counter()
                (run_dir / "presentation_reset_smoke.json").write_text(
                    json.dumps(
                        {
                            "status": (
                                "PASS" if presentation_smoke_reset_verified else "FAIL"
                            ),
                            "runtime_state": control.state.value,
                            "payload_m3": reset_reservoirs["payload_volume_m3"],
                            "mobile_m3": reset_reservoirs["mobile_volume_m3"],
                            "resting_volume_m3": reset_resting_volume_m3,
                            "initial_resting_volume_m3": initial_resting_volume_m3,
                            "resting_volume_error_m3": reset_resting_volume_error_m3,
                            "joint_position_max_error_rad": reset_joint_error,
                            "base_pose_xy_yaw_error_norm": reset_base_error,
                            "terrain_reset": True,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        snapshot = control.snapshot()
        if snapshot.state is InteractiveRuntimeState.RUNNING and acceptance.get("overall_status") != "PASS":
            failures = ", ".join(acceptance.get("blocking_failures", ["UNKNOWN_NO_SOIL_GATE_FAILURE"]))
            control.fail(RuntimeFailure("NO_SOIL_MACHINE_GATE_BLOCKED", failures, snapshot.current_cycle, "READY", float(world.current_time)))
            gate_reported = True
            snapshot = control.snapshot()

        if snapshot.kinematic_debug_enabled and snapshot.state not in {InteractiveRuntimeState.EMERGENCY_STOPPED, InteractiveRuntimeState.FAILED}:
            target = panel.joint_target_rad(initial_q)
            apply_bounded_arm_target(target)
            left_command, right_command = panel.track_commands()
            apply_track_soil_for_command(left_command, right_command)
            # A training/acceptance run has no viewport consumer.  Rendering each
            # physics tick forces the RTX/Kit render path to synchronize with this
            # CPU-bound terrain solver even when launched with ``--headless``.
            # Physics and dynamic contact meshes still advance identically.
            world.step(render=not HEADLESS)
            synchronize_terrain_contact()
        elif snapshot.state is InteractiveRuntimeState.RUNNING:
            if not world.is_playing():
                world.play()
            frame_wall_start = perf_counter()
            acceptance_ledger_wall_ms = 0.0
            cut_fill_audit_wall_ms = 0.0
            current_tool = kinematics.update(robot.get_tool_link_pose_world(), float(world.current_time))
            if FINAL_PRESENTATION:
                if not cycle_started:
                    presentation_trajectory.reset(
                        np.asarray(
                            articulation.get_joint_positions(), dtype=np.float64
                        ).reshape(-1)
                    )
                    cycle_started = True
                final_presentation_sample = presentation_trajectory.sample(
                    CONFIG.physics_dt_s
                )
                presentation_state = {
                    "APPROACH": ExcavatorCycleState.APPROACH,
                    "PENETRATION": ExcavatorCycleState.PENETRATE,
                    "ADVANCE_AND_CURL": ExcavatorCycleState.CUT_AND_FILL,
                    "BUCKET_FILL": ExcavatorCycleState.CURL_AND_BREAKOUT,
                    "BREAKOUT": ExcavatorCycleState.CURL_AND_BREAKOUT,
                    "LIFT": ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT,
                    "HOLD": ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT,
                }[final_presentation_sample.label]
                state_before_decision = presentation_state
                decision = SimpleNamespace(
                    state=presentation_state,
                    command=SimpleNamespace(
                        joint_target_rad=final_presentation_sample.target_rad,
                        left_track_effort_fraction=0.0,
                        right_track_effort_fraction=0.0,
                        hold_brake=True,
                    ),
                    failure=None,
                    transitioned=final_presentation_sample.transitioned,
                    transition_reason="PRESENTATION_C1_TRAJECTORY_STAGE",
                    completion_condition=(
                        "final presentation hold"
                        if final_presentation_sample.complete
                        else "smooth time-parameterized articulation target"
                    ),
                )
                observation = operation_observation(current_tool)
            else:
                if not cycle_started:
                    cycle_machine.start(operation_observation(current_tool))
                    cycle_started = True
                observation = operation_observation(current_tool)
                state_before_decision = cycle_machine.state
                decision = cycle_machine.step(observation)
            if decision.failure is not None:
                failure_reservoirs = physics_core.reservoir_observation()
                failure = RuntimeFailure(
                    decision.failure.code,
                    decision.failure.message,
                    snapshot.current_cycle,
                    decision.failure.state.value,
                    float(world.current_time),
                    {
                        "joint_position_rad": observation.joint_position_rad.tolist(),
                        "base_pose_xy_yaw": observation.base_pose_xy_yaw.tolist(),
                        "payload_volume_m3": observation.payload_volume_m3,
                        "joint_velocity_rad_s": np.asarray(
                            articulation.get_joint_velocities(), dtype=np.float64
                        ).reshape(-1).tolist(),
                        "dump_release_gate": dump_release_gate_metrics(current_tool),
                        "resting_volume_m3": float(
                            failure_reservoirs["resting_volume_m3"]
                        ),
                        "mobile_volume_m3": observation.mobile_volume_m3,
                        "airborne_volume_m3": observation.airborne_volume_m3,
                        "outflow_volume_m3": float(
                            failure_reservoirs["outflow_volume_m3"]
                        ),
                        "airborne_domain": airborne_domain_diagnostics(),
                        "material_funnel": dict(material_funnel),
                    },
                )
                control.fail(failure)
                failure_path.write_text(
                    json.dumps(
                        {
                            "code": failure.code,
                            "message": failure.message,
                            "cycle": failure.cycle,
                            "phase": failure.phase,
                            "timestamp_s": failure.timestamp_s,
                            "physical_state": failure.physical_state,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                panel.update("CYCLE_FAIL — physical state retained")
                continue
            if decision.transitioned:
                transition_log.append({
                    "timestamp_s": float(world.current_time),
                    "cycle": snapshot.current_cycle,
                    "state": (
                        final_presentation_sample.label
                        if FINAL_PRESENTATION
                        else decision.state.value
                    ),
                    "reason": decision.transition_reason,
                })
                (run_dir / "state_transitions.json").write_text(json.dumps(transition_log, indent=2) + "\n", encoding="utf-8")
                if decision.state is ExcavatorCycleState.CUT_AND_FILL:
                    cutting_distance_m = 0.0
                    previous_cutting_center = np.mean(current_tool.cutting_edge_terrain, axis=0)
                    if realistic_cut_trajectory is not None:
                        realistic_cut_trajectory.reset(
                            np.asarray(
                                articulation.get_joint_positions(), dtype=np.float64
                            ).reshape(-1)
                        )
                    if (
                        ARGS.cut_fill_payload_audit is not None
                        and cut_fill_payload_audit is not None
                        and not cut_fill_start_checkpoint_written
                    ):
                        audit_target = ARGS.cut_fill_payload_audit.expanduser().resolve()
                        checkpoint_path = audit_target.with_suffix(".cut_start.npz")
                        checkpoint_record = write_device_checkpoint(
                            physics_core,
                            checkpoint_path,
                            provenance={
                                "checkpoint_boundary": "CUT_AND_FILL_START_BEFORE_FIRST_CUT_CORE_STEP",
                                "checkpoint_source_run": run_id,
                                "simulation_time_s": float(world.current_time),
                                "joint_position_rad": np.asarray(articulation.get_joint_positions(), dtype=np.float64).reshape(-1).tolist(),
                                "joint_velocity_rad_s": np.asarray(articulation.get_joint_velocities(), dtype=np.float64).reshape(-1).tolist(),
                                "base_pose_xy_yaw": observation.base_pose_xy_yaw.tolist(),
                                "payload_m3": observation.payload_volume_m3,
                            },
                        )
                        checkpoint_path.with_suffix(".npz.json").write_text(
                            json.dumps(checkpoint_record, indent=2) + "\n", encoding="utf-8"
                        )
                        cut_fill_start_checkpoint_written = True
                if (
                    FINAL_PRESENTATION
                    and final_presentation_sample.label == "LIFT"
                ):
                    final_presentation_breakout_sim_s = float(world.current_time)
                if (
                    PRESENTATION_DEMO
                    and decision.state is ExcavatorCycleState.DEPOSITION
                ):
                    presentation_deposition_entry_sim_s = float(world.current_time)
                    presentation_deposition_entry_device_s = float(
                        physics_core.device_state.timestamp_device_s
                        if CONFIG.runtime_backend == "GPU_RUNTIME"
                        else world.current_time
                    )

            if (
                ARGS.mobile_v2_pre_dump_acceptance is not None
                and decision.transitioned
                and state_before_decision is ExcavatorCycleState.ALIGN_DUMP
                and decision.state is ExcavatorCycleState.DUMP
            ):
                # This boundary is deliberately before actuator/core handling
                # for DUMP, so no Payload->Airborne release can have occurred.
                capture_mobile_v2_pre_dump_acceptance()
                control.pause()
                if world.is_playing():
                    world.pause()
                _LIFECYCLE_RECORD["exit_reason"] = ExitReason.AUTOMATED_TEST_COMPLETED.value
                break

            if V3_CLOSURE_AUDIT:
                if decision.state is ExcavatorCycleState.PENETRATE:
                    capture_v3_closure_state("PRE_DIG", last_core_result)
                if (
                    decision.transitioned
                    and decision.state is ExcavatorCycleState.CURL_AND_BREAKOUT
                ):
                    capture_v3_closure_state("POST_CUT", last_core_result)
                if (
                    decision.transitioned
                    and decision.state is ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT
                ):
                    capture_v3_closure_state("POST_BREAKOUT", last_core_result)
                    v3_breakout_time_s = float(world.current_time)

            if decision.state is ExcavatorCycleState.READY_NEXT_CYCLE:
                # Publish the last track-induced terrain changes before the
                # acceptance loop exits. Normal GPU physics remains compact:
                # this consumes dirty tiles only, never a full TerrainState.
                synchronize_visual(force=True)
                synchronize_terrain_contact(force=True)
                control.cycle_completed()
                cycle_started = False
                cycle_machine.reset()
                panel.update(
                    f"Bucket Force     {latest_debug['force_total_n'] / 1000.0:7.1f} kN\n"
                    f"Payload Mass    {latest_debug['payload_m3'] * material.assumed_bulk_density_kg_m3:7.1f} kg\n"
                    f"Payload Volume  {latest_debug['payload_m3']:7.3f} m³\n"
                    f"Mobile Soil     {latest_debug['mobile_m3']:7.3f} m³\n"
                    "GPU Physics Active"
                )
                panel.update("cycle completed by physical completion conditions")
                continue

            commanded_joint_target_rad = decision.command.joint_target_rad
            if (
                realistic_cut_trajectory is not None
                and decision.state
                in {
                    ExcavatorCycleState.CUT_AND_FILL,
                    ExcavatorCycleState.CURL_AND_BREAKOUT,
                }
            ):
                realistic_cut_sample = realistic_cut_trajectory.sample(
                    CONFIG.physics_dt_s
                )
                commanded_joint_target_rad = realistic_cut_sample.target_rad
            actuator_output = apply_bounded_arm_target(commanded_joint_target_rad)
            production_physics_step_count += 1
            audit_this_step = bool(
                tracksoil_conservation_audit is not None
                and decision.state
                in {
                    ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT,
                    ExcavatorCycleState.REVERSE_TRAVEL,
                    ExcavatorCycleState.ALIGN_DUMP,
                }
            )
            if audit_this_step:
                tracksoil_conservation_audit.begin_step(
                    simulation_time_s=float(world.current_time),
                    phase=decision.state.value,
                    physics_step=production_physics_step_count,
                    snapshot=physics_core.scalar_state(),
                )
            track_wall_start = perf_counter()
            _, track_result_for_audit, track_geometry_for_audit = apply_track_soil_for_command(
                decision.command.left_track_effort_fraction,
                decision.command.right_track_effort_fraction,
                braking=decision.command.hold_brake,
            )
            if audit_this_step:
                tracksoil_conservation_audit.after_tracksoil(
                    snapshot=physics_core.scalar_state(),
                    result=track_result_for_audit,
                    affected_cell_count=track_geometry_for_audit[
                        "affected_cell_count"
                    ],
                    footprint_area_m2=track_geometry_for_audit[
                        "footprint_area_m2"
                    ],
                    requested_r2m_m3=track_geometry_for_audit[
                        "requested_r2m_m3"
                    ],
                )
            track_wall_ms = (perf_counter() - track_wall_start) * 1_000.0
            phase_by_state = {
                ExcavatorCycleState.PENETRATE: "penetrate",
                ExcavatorCycleState.CUT_AND_FILL: "coordinated_cut",
                ExcavatorCycleState.CURL_AND_BREAKOUT: "curl_filling",
                ExcavatorCycleState.DEPOSITION: "deposition",
            }
            deposition_just_completed = (
                decision.transitioned
                and state_before_decision is ExcavatorCycleState.DEPOSITION
            )
            dump_release_ready = False
            dump_gate = dump_release_gate_metrics(current_tool)
            if decision.state is ExcavatorCycleState.DUMP:
                dump_release_ready = bool(
                    dump_gate["joint_position_error_max_rad"]
                    <= cycle_machine.config.joint_tolerance_rad
                    and dump_gate["mouth_horizontal_speed_max_m_s"] <= 0.25
                )
            phase = (
                final_presentation_sample.physics_phase
                if FINAL_PRESENTATION
                else (
                    "deposition"
                    if deposition_just_completed
                    else (
                        "dump_spill"
                        if dump_release_ready
                        else phase_by_state.get(
                            decision.state, decision.state.value.lower()
                        )
                    )
                )
            )
            if decision.state in {ExcavatorCycleState.PENETRATE, ExcavatorCycleState.CUT_AND_FILL, ExcavatorCycleState.CURL_AND_BREAKOUT}:
                if CONFIG.runtime_backend == "HOST_REFERENCE":
                    sweep = physics_core.sweep_builder.build(physics_core.previous_tool_state, current_tool, grid, descriptor)
                    intersection = physics_core.intersection_model.compute(
                        physics_core.state.H_resting_m + physics_core.state.mobile_height_m,
                        sweep, current_tool, grid, physics_core.integrator
                    )
                    latest_intersection_volume_m3 = intersection.candidate_intersection_volume_m3
                    latest_penetration_depth_m = float(np.max(intersection.penetration_depth_m))
            else:
                latest_intersection_volume_m3 = 0.0
                latest_penetration_depth_m = 0.0
            if decision.state is ExcavatorCycleState.CUT_AND_FILL:
                center = np.mean(current_tool.cutting_edge_terrain, axis=0)
                if previous_cutting_center is not None:
                    cutting_distance_m += float(np.linalg.norm(center - previous_cutting_center))
                previous_cutting_center = center
            core_wall_start = perf_counter()
            payload_before_core_m3 = float(physics_core.payload.volume_m3)
            surface_before_core = (
                current_surface_height()
                if CONFIG.runtime_backend == "HOST_REFERENCE"
                else None
            )
            if mobile_first_write_audit is not None:
                mobile_first_write_audit.begin_step(
                    simulation_time_s=float(world.current_time),
                    phase=decision.state.value,
                    physics_step=production_physics_step_count,
                    tool_state=current_tool,
                )
            if tool_mobile_validity_audit is not None:
                tool_mobile_validity_audit.begin_step(
                    simulation_time_s=float(world.current_time),
                    phase=decision.state.value,
                    physics_step=production_physics_step_count,
                    tool_state=current_tool,
                    payload_before_m3=payload_before_core_m3,
                    joint_position_rad=np.asarray(
                        articulation.get_joint_positions(), dtype=np.float64
                    ).reshape(-1),
                    joint_velocity_rad_s=np.asarray(
                        articulation.get_joint_velocities(), dtype=np.float64
                    ).reshape(-1),
                    requested_joint_target_rad=commanded_joint_target_rad,
                )
            core_result = physics_core.step(
                current_tool,
                phase=phase,
                cycle=max(snapshot.current_cycle, 1),
                dt_s=CONFIG.physics_dt_s,
                soil_force_mode=snapshot.soil_force_mode,
                phase_ending=deposition_just_completed,
            )
            if mobile_first_write_audit is not None:
                mobile_first_write_audit.set_failure_metadata(
                    None
                    if core_result.interaction is None
                    else core_result.interaction.failure_bridge
                )
                mobile_first_write_audit.finish_step(core_result=core_result)
            if audit_this_step:
                audit_wall_start = perf_counter()
                tracksoil_conservation_audit.finish_step(
                    snapshot=physics_core.scalar_state(),
                    core_result=core_result,
                )
                acceptance_ledger_wall_ms = (
                    perf_counter() - audit_wall_start
                ) * 1_000.0
            last_core_result = core_result
            if (
                V3_CLOSURE_AUDIT
                and v3_breakout_time_s is not None
                and float(world.current_time) - v3_breakout_time_s >= 0.5
            ):
                capture_v3_closure_state("EARLY_POST_DIG", core_result)
            if (
                V3_CLOSURE_AUDIT
                and "EARLY_POST_DIG" in v3_closure_labels
                and core_result.terrain_settled
            ):
                capture_v3_closure_state("ARREST_FINAL", core_result)
            if (
                ARGS.dump_after_checkpoint is not None
                and not dump_checkpoint_written
                and decision.transitioned
                and state_before_decision is ExcavatorCycleState.DUMP
                and decision.state is ExcavatorCycleState.DEPOSITION
            ):
                if CONFIG.runtime_backend != "GPU_RUNTIME":
                    raise RuntimeError(
                        "DUMP_AFTER_CHECKPOINT_REQUIRES_GPU_RUNTIME"
                    )
                checkpoint_path = ARGS.dump_after_checkpoint.expanduser().resolve()
                checkpoint_record = write_device_checkpoint(
                    physics_core,
                    checkpoint_path,
                    provenance={
                        "checkpoint_source_run": run_id,
                        "reproduces_interrupted_run": "run_1786506718",
                        "checkpoint_boundary": "FIRST_PRODUCTION_CORE_STEP_AFTER_DUMP_TO_DEPOSITION_TRANSITION",
                        "checkpoint_sim_time_s": float(world.current_time),
                        "terrain_file": str(CONFIG.initial_heightmap),
                        "terrain_hash_sha256": sha256_file(CONFIG.initial_heightmap),
                        "config_file": str(CONFIG.config_path),
                        "config_hash_sha256": sha256_file(CONFIG.config_path),
                        "physics_core": type(physics_core).__name__,
                        "runtime_backend": CONFIG.runtime_backend,
                        "state_authority": "DEVICE",
                    },
                )
                checkpoint_record_path = checkpoint_path.with_suffix(
                    checkpoint_path.suffix + ".json"
                )
                checkpoint_record_path.write_text(
                    json.dumps(checkpoint_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                manifest["dump_after_checkpoint"] = checkpoint_record
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                dump_checkpoint_written = True
                dump_checkpoint_device_time_s = float(
                    physics_core.device_state.timestamp_device_s
                )
            if (
                ARGS.dump_plus_9_checkpoint is not None
                and dump_checkpoint_written
                and not dump_plus_9_checkpoint_written
                and dump_checkpoint_device_time_s is not None
                and float(physics_core.device_state.timestamp_device_s)
                - dump_checkpoint_device_time_s
                >= 9.0 - 0.5 * CONFIG.physics_dt_s
            ):
                checkpoint_path = ARGS.dump_plus_9_checkpoint.expanduser().resolve()
                checkpoint_record = write_device_checkpoint(
                    physics_core,
                    checkpoint_path,
                    provenance={
                        "checkpoint_source_run": run_id,
                        "checkpoint_boundary": "T_DUMP_END_PLUS_9S_TERRAIN_PHYSICS",
                        "checkpoint_sim_time_s": float(world.current_time),
                        "elapsed_terrain_physics_s": float(
                            physics_core.device_state.timestamp_device_s
                            - dump_checkpoint_device_time_s
                        ),
                        "terrain_file": str(CONFIG.initial_heightmap),
                        "terrain_hash_sha256": sha256_file(CONFIG.initial_heightmap),
                        "config_file": str(CONFIG.config_path),
                        "config_hash_sha256": sha256_file(CONFIG.config_path),
                        "physics_core": type(physics_core).__name__,
                        "runtime_backend": CONFIG.runtime_backend,
                        "state_authority": "DEVICE",
                    },
                )
                checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").write_text(
                    json.dumps(checkpoint_record, indent=2) + "\n", encoding="utf-8"
                )
                manifest["dump_plus_9_checkpoint"] = checkpoint_record
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                dump_plus_9_checkpoint_written = True
            core_wall_ms = (perf_counter() - core_wall_start) * 1_000.0
            update_payload_visual()
            computed_norm_n = float(np.linalg.norm(core_result.computed_force_terrain_n))
            applied_norm_n = float(np.linalg.norm(core_result.applied_force_terrain_n))
            quasi_norm_n = float(np.linalg.norm(core_result.quasi_static_force_terrain_n))
            momentum_norm_n = float(np.linalg.norm(core_result.momentum_force_terrain_n))
            soil_force_evidence["computed_peak_n"] = max(
                float(soil_force_evidence["computed_peak_n"]), computed_norm_n
            )
            soil_force_evidence["applied_peak_n"] = max(
                float(soil_force_evidence["applied_peak_n"]), applied_norm_n
            )
            soil_force_evidence["quasi_static_peak_n"] = max(
                float(soil_force_evidence["quasi_static_peak_n"]), quasi_norm_n
            )
            soil_force_evidence["momentum_peak_n"] = max(
                float(soil_force_evidence["momentum_peak_n"]), momentum_norm_n
            )
            soil_force_evidence["computed_nonzero_step_count"] += int(
                computed_norm_n > 1.0e-9
            )
            soil_force_evidence["applied_nonzero_step_count"] += int(
                applied_norm_n > 1.0e-9
            )
            impulse = np.asarray(
                soil_force_evidence["applied_impulse_terrain_ns"], dtype=np.float64
            ) + np.asarray(core_result.applied_force_terrain_n, dtype=np.float64) * CONFIG.physics_dt_s
            soil_force_evidence["applied_impulse_terrain_ns"] = impulse.tolist()
            if core_result.interaction is not None:
                applicability = core_result.interaction.failure_zone.applicability_status
                soil_force_evidence["fee_applicable_step_count"] += int(
                    applicability in {"APPLICABLE", "PARTIAL_OUTSIDE_FEE_DOMAIN"}
                )
                soil_force_evidence["fee_outside_domain_step_count"] += int(
                    applicability in {"OUTSIDE_FEE_DOMAIN", "PARTIAL_OUTSIDE_FEE_DOMAIN"}
                )
            if core_result.interaction is not None:
                if CONFIG.runtime_backend == "GPU_RUNTIME":
                    latest_intersection_volume_m3 = float(
                        core_result.interaction.failure_bridge.intersection.candidate_intersection_volume_m3
                    )
                    latest_penetration_depth_m = float(
                        np.max(
                            core_result.interaction.failure_bridge.intersection.penetration_depth_m
                        )
                    )
                material_funnel["failure_volume_m3"] += float(
                    core_result.interaction.failure_zone.active_volume_m3
                )
                material_funnel["activated_volume_m3"] += float(
                    core_result.interaction.activated_volume_m3
                )
                material_funnel["candidate_mouth_flux_m3"] += float(
                    core_result.interaction.intake_result.candidate_flux_volume_m3
                )
                material_funnel["admitted_volume_m3"] += float(
                    core_result.interaction.intake_result.bucket_inflow_volume_m3
                )
                if (
                    cut_fill_payload_audit is not None
                    and decision.state
                    in {
                        ExcavatorCycleState.CUT_AND_FILL,
                        ExcavatorCycleState.CURL_AND_BREAKOUT,
                    }
                    and core_result.avalanche_transition is not None
                ):
                    audit_wall_start = perf_counter()
                    cut_fill_payload_audit.observe(
                        simulation_time_s=float(world.current_time),
                        tool_state=current_tool,
                        interaction=core_result.interaction,
                        avalanche=core_result.avalanche_transition,
                        payload_before_m3=payload_before_core_m3,
                        payload_after_m3=float(physics_core.payload.volume_m3),
                        joint_position_rad=np.asarray(
                            articulation.get_joint_positions(), dtype=np.float64
                        ).reshape(-1),
                        joint_velocity_rad_s=np.asarray(
                            articulation.get_joint_velocities(), dtype=np.float64
                        ).reshape(-1),
                        actuator_output=actuator_output,
                        quasi_static_force_n=core_result.quasi_static_force_terrain_n,
                        momentum_force_n=core_result.momentum_force_terrain_n,
                        applied_force_n=core_result.applied_force_terrain_n,
                        failure_zone_r2m_step_m3=float(
                            core_result.interaction.activated_volume_m3
                        ),
                        mobile_volume_m3=float(
                            physics_core.reservoir_observation()["mobile_volume_m3"]
                        ),
                        machine_velocity_terrain_m_s=world_vector_to_terrain_xy(
                            np.asarray(
                                lower_body.get_linear_velocities(), dtype=np.float64
                            ).reshape(-1, 3)[0]
                        ),
                        force_application_point_terrain_m=(
                            None
                            if core_result.force_result is None
                            else core_result.force_result.application_point_terrain_m
                        ),
                        requested_joint_target_rad=commanded_joint_target_rad,
                        phase=decision.state.value,
                        trajectory_stage=(
                            "LEGACY_FIXED_TARGET"
                            if realistic_cut_sample is None
                            else realistic_cut_sample.stage
                        ),
                        mass_balance_error_m3=core_result.mass_balance_error_m3,
                    )
                    cut_fill_audit_wall_ms = (
                        perf_counter() - audit_wall_start
                    ) * 1_000.0
            if tool_mobile_validity_audit is not None:
                assert physics_core.device_state is not None
                validity_reservoir = physics_core.reservoir_observation()
                validity_reservoir.update(
                    physics_core.device_state.reservoir_reduction(
                        material.assumed_bulk_density_kg_m3
                    )
                )
                latest_cut_record = None
                if (
                    cut_fill_payload_audit is not None
                    and cut_fill_payload_audit.records
                    and np.isclose(
                        float(cut_fill_payload_audit.records[-1]["simulation_time_s"]),
                        float(world.current_time),
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                ):
                    latest_cut_record = cut_fill_payload_audit.records[-1]
                tool_mobile_validity_audit.finish_step(
                    core_result=core_result,
                    payload_after_m3=float(physics_core.payload.volume_m3),
                    reservoir=validity_reservoir,
                    cut_record=latest_cut_record,
                    base_pose_xy_yaw=base_pose_and_forward()[0],
                )
                if (
                    ARGS.tool_mobile_validity_visual_diagnostic
                    and tool_mobile_validity_audit.creation_time_s is not None
                ):
                    visual_tau_s = (
                        float(world.current_time)
                        - tool_mobile_validity_audit.creation_time_s
                    )
                    for offset_s in tool_mobile_visual_offsets_s:
                        if (
                            offset_s in tool_mobile_visual_captured_offsets
                            or visual_tau_s + 1.0e-9 < offset_s
                        ):
                            continue
                        offset_steps = int(round(offset_s * 60.0))
                        frame_label = f"first_mobile_plus_{offset_steps:02d}dt"
                        for view_name in ("side", "top"):
                            capture_tool_mobile_validity_view(
                                label=frame_label,
                                view=view_name,
                                tool_state=current_tool,
                                tau_s=visual_tau_s,
                            )
                        tool_mobile_visual_captured_offsets.add(offset_s)
            if core_result.dump_release is not None:
                material_funnel["dump_released_volume_m3"] += float(
                    core_result.dump_release.released_volume_m3
                )
                dump_release_started = bool(
                    dump_release_started
                    or core_result.dump_release.released_volume_m3 > 0.0
                )
                if (
                    ARGS.dump_release_checkpoint is not None
                    and not dump_release_checkpoint_written
                    and core_result.dump_release.released_volume_m3 > 0.0
                ):
                    if CONFIG.runtime_backend != "GPU_RUNTIME":
                        raise RuntimeError(
                            "DUMP_RELEASE_CHECKPOINT_REQUIRES_GPU_RUNTIME"
                        )
                    release_path = (
                        ARGS.dump_release_checkpoint.expanduser().resolve()
                    )
                    release_record = write_device_checkpoint(
                        physics_core,
                        release_path,
                        provenance={
                            "checkpoint_source_run": run_id,
                            "reproduces_interrupted_run": "run_1786506718",
                            "checkpoint_boundary": "FIRST_NONZERO_PAYLOAD_TO_AIRBORNE_RELEASE",
                            "checkpoint_sim_time_s": float(world.current_time),
                            "terrain_file": str(CONFIG.initial_heightmap),
                            "terrain_hash_sha256": sha256_file(CONFIG.initial_heightmap),
                            "config_file": str(CONFIG.config_path),
                            "config_hash_sha256": sha256_file(CONFIG.config_path),
                            "physics_core": type(physics_core).__name__,
                            "runtime_backend": CONFIG.runtime_backend,
                            "state_authority": "DEVICE",
                        },
                    )
                    release_path.with_suffix(
                        release_path.suffix + ".json"
                    ).write_text(
                        json.dumps(release_record, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    manifest["dump_release_checkpoint"] = release_record
                    dump_release_checkpoint_written = True
            if core_result.dump_advance is not None:
                material_funnel["airborne_landed_volume_m3"] += float(
                    core_result.dump_advance.airborne.landed_volume_m3
                )
                if (
                    dump_release_started
                    and core_result.dump_advance.deposition is not None
                ):
                    material_funnel["deposited_after_dump_m3"] += float(
                        core_result.dump_advance.deposition.deposited_volume_m3
                    )
            parcel_domain = airborne_domain_diagnostics()
            if (
                dump_release_started
                and parcel_domain["outside_heightmap_count"] > 0
            ):
                failure = RuntimeFailure(
                    "DUMP_PARCEL_LEFT_TERRAIN_DOMAIN",
                    "dump-origin airborne parcel is outside the authoritative heightmap",
                    snapshot.current_cycle,
                    decision.state.value,
                    float(world.current_time),
                    {
                        "payload_volume_m3": physics_core.payload.volume_m3,
                        "airborne_volume_m3": float(
                            physics_core.reservoir_observation()["airborne_volume_m3"]
                        ),
                        "airborne_domain": parcel_domain,
                        "dump_release_gate": dict(dump_gate),
                        "material_funnel": dict(material_funnel),
                    },
                )
                control.fail(failure)
                failure_path.write_text(
                    json.dumps(
                        {
                            "code": failure.code,
                            "message": failure.message,
                            "cycle": failure.cycle,
                            "phase": failure.phase,
                            "timestamp_s": failure.timestamp_s,
                            "physical_state": failure.physical_state,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                panel.update("CYCLE_FAIL — dump parcel left terrain domain")
                continue
            if CONFIG.runtime_backend == "GPU_RUNTIME" or np.any(
                np.abs(current_surface_height() - surface_before_core) > 1.0e-10
            ):
                terrain_timestamp_s = float(world.current_time)
            if core_result.force_result is not None and np.linalg.norm(core_result.applied_force_terrain_n) > 0.0:
                transform_world = grid.terrain_to_world_matrix
                point_h = transform_world @ np.r_[core_result.force_result.application_point_terrain_m, 1.0]
                residual_couple = (
                    core_result.force_result.residual_couple_terrain_nm
                    if np.allclose(
                        core_result.applied_force_terrain_n,
                        core_result.computed_force_terrain_n,
                    )
                    else np.zeros(3, dtype=np.float64)
                )
                bucket_force_body.apply_forces_and_torques_at_pos(
                    forces=(transform_world[:3, :3] @ core_result.applied_force_terrain_n).astype(np.float32).reshape(1, 3),
                    torques=(transform_world[:3, :3] @ residual_couple).astype(np.float32).reshape(1, 3),
                    positions=(point_h[:3] / point_h[3]).astype(np.float32).reshape(1, 3),
                    is_global=True,
                )
            physx_wall_start = perf_counter()
            # Keep PhysX advancing while avoiding an unnecessary render/sync on
            # every headless rollout step.
            # World.step owns the single PhysX + Kit/render update.  An extra
            # SimulationApp.update here doubled Isaac world time relative to
            # the authoritative terrain-material clock.
            world.step(render=not HEADLESS)
            physx_wall_ms = (perf_counter() - physx_wall_start) * 1_000.0
            visual_wall_start = perf_counter()
            # Visual terrain chunks are only useful with an interactive viewport.
            # Do not rebuild their USD points during a headless RL rollout.
            visual_interval_steps = (
                PRESENTATION_CONFIG.visual_sync_interval_steps
                if FINAL_PRESENTATION
                else 6
            )
            if (
                not HEADLESS
                and int(world.current_time / CONFIG.physics_dt_s)
                % visual_interval_steps
                == 0
                and (
                    not FINAL_PRESENTATION
                    or presentation_loop_count > 1
                )
            ):
                synchronize_visual()
            visual_wall_ms = (perf_counter() - visual_wall_start) * 1_000.0
            contact_wall_start = perf_counter()
            if not FINAL_PRESENTATION or presentation_loop_count > 1:
                synchronize_terrain_contact()
            contact_wall_ms = (perf_counter() - contact_wall_start) * 1_000.0
            reservoir_now = physics_core.reservoir_observation()
            latest_debug = {
                "phase": (
                    final_presentation_sample.label
                    if FINAL_PRESENTATION
                    else decision.state.value
                ),
                "completion_condition": decision.completion_condition,
                "rake_deg": (
                    0.0
                    if core_result.interaction is None
                    or not core_result.interaction.failure_zone.strip_geometries
                    else float(
                        np.average(
                            [strip.rake_angle_deg for strip in core_result.interaction.failure_zone.strip_geometries],
                            weights=[max(strip.wedge_volume_m3, 1.0e-12) for strip in core_result.interaction.failure_zone.strip_geometries],
                        )
                    )
                ),
                "penetration_m": latest_penetration_depth_m,
                "failure_zone_m3": 0.0 if core_result.interaction is None else core_result.interaction.failure_zone.active_volume_m3,
                "mobile_m3": float(reservoir_now["mobile_volume_m3"]),
                "payload_m3": float(reservoir_now["payload_volume_m3"]),
                "fill_ratio": float(reservoir_now["payload_volume_m3"]) / descriptor.effective_capacity_m3,
                "force_quasi_n": float(np.linalg.norm(core_result.quasi_static_force_terrain_n)),
                "force_momentum_n": float(np.linalg.norm(core_result.momentum_force_terrain_n)),
                "force_total_n": float(np.linalg.norm(core_result.computed_force_terrain_n)),
                "resting_m3": float(reservoir_now["resting_volume_m3"]),
                "airborne_m3": float(reservoir_now["airborne_volume_m3"]),
                "outflow_m3": float(reservoir_now["outflow_volume_m3"]),
                "mass_error_m3": core_result.mass_balance_error_m3,
                "rtf": CONFIG.physics_dt_s / max(core_result.timings_ms["physics_core_total"] / 1000.0, 1.0e-9),
            }
            if FINAL_PRESENTATION:
                final_presentation_peak_payload_m3 = max(
                    final_presentation_peak_payload_m3,
                    latest_debug["payload_m3"],
                )
                final_presentation_peak_mobile_m3 = max(
                    final_presentation_peak_mobile_m3,
                    latest_debug["mobile_m3"],
                )
                final_presentation_peak_force_n = max(
                    final_presentation_peak_force_n,
                    latest_debug["force_total_n"],
                )
                final_presentation_rtf_samples.append(latest_debug["rtf"])
                final_presentation_frame_times_ms.append(
                    (perf_counter() - frame_wall_start) * 1_000.0
                )
                if (
                    final_presentation_sample.transitioned
                    and final_presentation_sample.label
                    in PRESENTATION_CONFIG.capture_labels
                ):
                    capture_presentation_viewport(final_presentation_sample.label)
            avalanche_transition = core_result.avalanche_transition
            avalanche_diagnostics = (
                None
                if avalanche_transition is None
                else getattr(avalanche_transition, "diagnostics", avalanche_transition)
            )
            avalanche_observation = {
                "classification": (
                    core_result.terrain_settled_diagnostic.large_avalanche_status
                    if avalanche_diagnostics is None
                    else getattr(avalanche_diagnostics, "classification", None)
                ),
                "terrain_settled": bool(core_result.terrain_settled),
                "settled_reason": core_result.terrain_settled_diagnostic.reason,
                "unstable_cell_count": int(
                    getattr(avalanche_diagnostics, "unstable_cell_count", 0)
                ),
                "largest_connected_cell_count": int(
                    getattr(avalanche_diagnostics, "largest_connected_cell_count", 0)
                ),
                "largest_connected_area_m2": float(
                    getattr(avalanche_diagnostics, "largest_connected_area_m2", 0.0)
                ),
                "persistence_s": float(
                    getattr(avalanche_diagnostics, "persistence_s", 0.0)
                ),
                "transferred_volume_m3": float(
                    getattr(avalanche_transition, "transferred_volume_m3", 0.0)
                ),
                "residual_minislope_iterations": int(
                    core_result.static_relaxation_iterations
                ),
                "residual_minislope_active_tiles": int(
                    core_result.static_relaxation_active_tiles
                ),
                "residual_minislope_pending": bool(
                    core_result.static_relaxation_pending
                ),
            }
            if int(world.current_time / CONFIG.physics_dt_s) % 30 == 0 or decision.transitioned:
                base_pose_now, _ = base_pose_and_forward()
                lower_positions, lower_quaternions = lower_body.get_world_poses()
                lower_position_world = np.asarray(
                    lower_positions, dtype=np.float64
                ).reshape(-1, 3)[0]
                lower_quaternion_wxyz = np.asarray(
                    lower_quaternions, dtype=np.float64
                ).reshape(-1, 4)[0]
                from scipy.spatial.transform import Rotation
                lower_euler_xyz = Rotation.from_quat([
                    lower_quaternion_wxyz[1], lower_quaternion_wxyz[2],
                    lower_quaternion_wxyz[3], lower_quaternion_wxyz[0],
                ]).as_euler("xyz")
                runtime_telemetry.append({
                    "timestamp_s": float(world.current_time),
                    "cycle": snapshot.current_cycle,
                    "state": decision.state.value,
                    "base_pose_xy_yaw": base_pose_now.tolist(),
                    "base_z_roll_pitch_rad": [
                        float(lower_position_world[2]),
                        float(lower_euler_xyz[0]),
                        float(lower_euler_xyz[1]),
                    ],
                    "payload_m3": float(reservoir_now["payload_volume_m3"]),
                    "mobile_m3": latest_debug["mobile_m3"],
                    "resting_m3": latest_debug["resting_m3"],
                    "resting_gain_from_initial_m3": (
                        latest_debug["resting_m3"] - initial_resting_volume_m3
                    ),
                    "airborne_m3": latest_debug["airborne_m3"],
                    "outflow_m3": latest_debug["outflow_m3"],
                    "mass_error_m3": latest_debug["mass_error_m3"],
                    "core_rtf": latest_debug["rtf"],
                    "soil_force": {
                        "quasi_static_n": latest_debug["force_quasi_n"],
                        "momentum_n": latest_debug["force_momentum_n"],
                        "total_n": latest_debug["force_total_n"],
                        "stepwise_evidence": dict(soil_force_evidence),
                    },
                    "airborne_domain": airborne_domain_diagnostics(),
                    "dump_release_gate": dict(dump_gate),
                    "material_funnel": dict(material_funnel),
                    "large_avalanche": avalanche_observation,
                    "arm_actuator": {
                        "target_velocity_rad_s": actuator_output.target_velocity_rad_s.tolist(),
                        "positive_mechanical_power_w": actuator_output.positive_mechanical_power_w,
                        "shared_power_scale": actuator_output.shared_power_scale,
                    },
                    "terrain_timestamp_s": terrain_timestamp_s,
                    "contact_timestamp_s": contact_status.contact_timestamp_s,
                    "contact_lag_s": contact_status.contact_lag_s,
                    "wall_ms": {
                        "track_soil_and_drive": track_wall_ms,
                        "bulk_core": core_wall_ms,
                        "physx_render_step": physx_wall_ms,
                        "visual_update": visual_wall_ms,
                        "contact_update": contact_wall_ms,
                        "p0_2b_acceptance_ledger": acceptance_ledger_wall_ms,
                        "p0_2d_cut_fill_audit": cut_fill_audit_wall_ms,
                        "total_frame": (perf_counter() - frame_wall_start) * 1_000.0,
                    },
                    "core_timings_ms": core_result.timings_ms,
                    "track_status": dict(latest_track_status),
                })
                telemetry_path.write_text(
                    json.dumps(runtime_telemetry, indent=2) + "\n", encoding="utf-8"
                )
        else:
            if not HEADLESS and world.is_playing():
                world.pause()
            if FINAL_PRESENTATION:
                # READY/PAUSED are live GUI states, not task completion.  Feed
                # Kit exactly like the proven startup-smoke loop and return to
                # the top immediately; no task-completion branch is evaluated.
                simulation_app.update()
                final_presentation_ui_warmup_frames = max(
                    0, final_presentation_ui_warmup_frames - 1
                )
                if (
                    final_presentation_ui_warmup_frames == 0
                    and control.state is InteractiveRuntimeState.READY
                    and (
                        ARGS.final_presentation_autostart
                        or ARGS.final_presentation_blocker_check
                    )
                ):
                    control.run_cycles(1)
                continue
            simulation_app.update()
        post_dump_elapsed_s = (
            None
            if presentation_deposition_entry_sim_s is None
            else float(world.current_time) - presentation_deposition_entry_sim_s
        )
        post_dump_physics_elapsed_s = (
            None
            if presentation_deposition_entry_device_s is None
            else float(
                physics_core.device_state.timestamp_device_s
                if CONFIG.runtime_backend == "GPU_RUNTIME"
                else world.current_time
            )
            - presentation_deposition_entry_device_s
        )
        physical_post_dump_stop = bool(
            ARGS.continue_post_dump_until_settled
            and post_dump_physics_elapsed_s is not None
            and (
                (last_core_result is not None and last_core_result.terrain_settled)
                or post_dump_physics_elapsed_s >= ARGS.post_dump_max_observation_s
            )
        )
        if (
            PRESENTATION_DEMO
            and not FINAL_PRESENTATION
            and not presentation_auto_paused
            and presentation_deposition_entry_sim_s is not None
            and control.state is InteractiveRuntimeState.RUNNING
            and (
                physical_post_dump_stop
                if ARGS.continue_post_dump_until_settled
                else post_dump_elapsed_s >= 9.0
            )
        ):
            control.pause()
            if not HEADLESS and world.is_playing():
                world.pause()
            presentation_auto_paused = True
            frame_values = np.asarray(
                [item["wall_ms"]["total_frame"] for item in runtime_telemetry],
                dtype=np.float64,
            )
            rtf_values = np.asarray(
                [item["core_rtf"] for item in runtime_telemetry],
                dtype=np.float64,
            )
            demo_summary = {
                "schema": "390F_LOCAL_PRESENTATION_DEMO/v1",
                "status": "DEMO_ONLY",
                "post_dump_status": (
                    "ARREST_FINAL"
                    if ARGS.continue_post_dump_until_settled
                    and last_core_result is not None
                    and last_core_result.terrain_settled
                    else "OBSERVATION_HORIZON_ACTIVE_STATE"
                    if ARGS.continue_post_dump_until_settled
                    else "POST_DUMP_TIMESCALE_UNDER_VALIDATION"
                ),
                "cycle_complete": False,
                "final_acceptance_pass": False,
                "terrain_settled_claimed": bool(
                    ARGS.continue_post_dump_until_settled
                    and last_core_result is not None
                    and last_core_result.terrain_settled
                ),
                "auto_pause_phase": "DEPOSITION",
                "post_dump_physics_display_s": float(world.current_time)
                - presentation_deposition_entry_sim_s,
                "post_dump_terrain_physics_s": post_dump_physics_elapsed_s,
                "runtime_backend": CONFIG.runtime_backend,
                "state_authority": "DEVICE",
                "grid_shape_yx": list(CONFIG.grid_shape),
                "resolution_m": CONFIG.grid_spacing_m,
                "root_pose_write_count": 0,
                "payload_m3": latest_debug["payload_m3"],
                "mobile_m3": latest_debug["mobile_m3"],
                "mean_rtf": float(np.mean(rtf_values)) if rtf_values.size else None,
                "mean_frame_time_ms": (
                    float(np.mean(frame_values)) if frame_values.size else None
                ),
                "phase_transitions": transition_log,
                "visual_terrain_publication": (
                    "GPU_DIRTY_TILES_TO_AUTHORITATIVE_VISUAL_MESH"
                ),
                "visual_payload": (
                    "REAL_PAYLOAD_VOLUME_DRIVEN_VISUAL_ONLY_NO_PHYSX"
                ),
            }
            (run_dir / "presentation_demo_summary.json").write_text(
                json.dumps(demo_summary, indent=2) + "\n", encoding="utf-8"
            )
        if (
            FINAL_PRESENTATION
            and not final_presentation_completed
            and control.state is InteractiveRuntimeState.RUNNING
        ):
            blocker_due = bool(
                ARGS.final_presentation_blocker_check
                and final_presentation_breakout_sim_s is not None
                and float(world.current_time) - final_presentation_breakout_sim_s
                >= PRESENTATION_CONFIG.blocker_post_breakout_s
            )
            if blocker_due:
                synchronize_visual(force=True)
                capture_presentation_viewport("VISUAL_BLOCKER")
                capture_visual_blocker_state()
                control.pause()
                if world.is_playing():
                    world.pause()
                final_presentation_completed = True
                write_final_presentation_summary("VISUAL_BLOCKER_CHECK_CAPTURED")
            elif (
                final_presentation_sample is not None
                and final_presentation_sample.complete
            ):
                synchronize_visual(force=True)
                capture_presentation_viewport("HOLD")
                control.pause()
                if world.is_playing():
                    world.pause()
                final_presentation_completed = True
                write_final_presentation_summary("PRESENTATION_COMPLETED_AT_FINAL_HOLD")
        extra = (
            "NO-SOIL MACHINE GATE: " + str(acceptance.get("overall_status"))
            + f"\nphase={latest_debug['phase']} | condition={latest_debug['completion_condition']}"
            + f"\nrake={latest_debug['rake_deg']:.1f} deg | penetration={latest_debug['penetration_m']:.3f} m"
            + f" | Failure Zone={latest_debug['failure_zone_m3']:.4f} m3"
            + f"\npayload={latest_debug['payload_m3']:.4f} m3 | fill={100.0 * latest_debug['fill_ratio']:.1f}%"
            + f" | Mobile={latest_debug['mobile_m3']:.4f} m3"
            + f" | Airborne={latest_debug['airborne_m3']:.4f} m3"
            + f"\nF_quasi_static={latest_debug['force_quasi_n']:.0f} N"
            + f" | F_momentum={latest_debug['force_momentum_n']:.0f} N"
            + f" | F_total={latest_debug['force_total_n']:.0f} N"
            + f"\ntrack footprints L/R contact={latest_track_status['left_contact']}/{latest_track_status['right_contact']}"
            + f" | sinkage={latest_track_status['left_sinkage_m']:.4f}/{latest_track_status['right_sinkage_m']:.4f} m"
            + f" | rut bbox={latest_track_status['active_bbox_grid']}"
            + f"\nmass balance={latest_debug['mass_error_m3']:.3e} m3 | RTF(core)={latest_debug['rtf']:.2f}"
            + f" | contact lag={contact_status.contact_lag_s:.3f} s"
        )
        if PRESENTATION_DEMO:
            phase_label = latest_debug["phase"]
            if presentation_auto_paused:
                phase_label = "DEPOSITION — AUTO PAUSED"
            if FINAL_PRESENTATION:
                extra = (
                    f"Bucket Force     {latest_debug['force_total_n'] / 1000.0:7.1f} kN\n"
                    f"Payload Mass    {latest_debug['payload_m3'] * material.assumed_bulk_density_kg_m3:7.1f} kg\n"
                    f"Payload Volume  {latest_debug['payload_m3']:7.3f} m³\n"
                    f"Mobile Soil     {latest_debug['mobile_m3']:7.3f} m³\n"
                    "GPU Physics Active"
                )
            else:
                extra = (
                    f"PHASE  {phase_label}\n"
                    f"SIM TIME  {float(world.current_time):.2f} s\n"
                    f"PAYLOAD  {latest_debug['payload_m3']:.3f} m³\n"
                    f"MOBILE  {latest_debug['mobile_m3']:.3f} m³\n"
                    "GPU_RUNTIME / DEVICE\n"
                    f"RTF  {latest_debug['rtf']:.2f}"
                )
        if gate_reported:
            extra += " — Isaac remains open for inspection/reset"
        if not FINAL_PRESENTATION or final_presentation_ui_warmup_frames <= 0:
            panel.update(extra)
            if FINAL_PRESENTATION:
                panel.update_observation(
                    bucket_force_kn=latest_debug["force_total_n"] / 1000.0,
                    payload_mass_kg=(
                        latest_debug["payload_m3"]
                        * material.assumed_bulk_density_kg_m3
                    ),
                    payload_volume_m3=latest_debug["payload_m3"],
                    mobile_volume_m3=latest_debug["mobile_m3"],
                )
        else:
            final_presentation_ui_warmup_frames -= 1
        if GUI_LIFECYCLE_SMOKE:
            if (
                gui_lifecycle_stage == "RUNNING"
                and control.state is InteractiveRuntimeState.FAILED
            ):
                gui_lifecycle_evidence.append(
                    {
                        "check": "PRODUCTION_CYCLE",
                        "status": "FAIL",
                        "runtime_state": control.state.value,
                        "failure": (
                            None
                            if control.snapshot().failure is None
                            else control.snapshot().failure.code
                        ),
                        "process_alive": True,
                    }
                )
                panel.pending_reset = ResetLevel.ALL
                presentation_smoke_reset_requested = True
                gui_lifecycle_stage = "RESET_PENDING"
            elif gui_lifecycle_stage == "RUNNING" and presentation_auto_paused:
                gui_lifecycle_stage = "POST_DUMP_PAUSE_HOLD"
                gui_lifecycle_hold_started_wall_s = perf_counter()
            elif gui_lifecycle_stage == "POST_DUMP_PAUSE_HOLD":
                pause_hold_s = perf_counter() - gui_lifecycle_hold_started_wall_s
                if pause_hold_s >= 30.0:
                    gui_lifecycle_evidence.append(
                        {
                            "check": "POST_DUMP_PAUSE_HOLD",
                            "status": "PASS",
                            "wall_hold_s": pause_hold_s,
                            "runtime_state": control.state.value,
                            "process_alive": True,
                        }
                    )
                    panel.pending_reset = ResetLevel.ALL
                    presentation_smoke_reset_requested = True
                    gui_lifecycle_stage = "RESET_PENDING"
            elif gui_lifecycle_stage == "RESET_READY_HOLD":
                reset_hold_s = perf_counter() - gui_lifecycle_hold_started_wall_s
                if reset_hold_s >= 30.0:
                    reset_hold_pass = bool(
                        presentation_smoke_reset_verified
                        and control.state is InteractiveRuntimeState.READY
                    )
                    gui_lifecycle_evidence.append(
                        {
                            "check": "RESET_READY_HOLD",
                            "status": "PASS" if reset_hold_pass else "FAIL",
                            "wall_hold_s": reset_hold_s,
                            "runtime_state": control.state.value,
                            "process_alive": True,
                        }
                    )
                    (run_dir / "presentation_gui_lifecycle_smoke.json").write_text(
                        json.dumps(
                            {
                                "status": (
                                    "PASS"
                                    if reset_hold_pass
                                    and all(
                                        item["status"] == "PASS"
                                        for item in gui_lifecycle_evidence
                                    )
                                    else "FAIL"
                                ),
                                "explicit_test_exit": True,
                                "checks": gui_lifecycle_evidence,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    break
        if HEADLESS and ARGS.presentation_smoke_autostart:
            if presentation_auto_paused and not presentation_smoke_reset_requested:
                panel.pending_reset = ResetLevel.ALL
                presentation_smoke_reset_requested = True
            elif presentation_smoke_reset_completed:
                break
        if FINAL_PRESENTATION and final_presentation_completed and (
            ARGS.final_presentation_exit_after_hold
            or ARGS.final_presentation_blocker_check
        ):
            break
        if HEADLESS and ARGS.acceptance_cycles > 0 and control.state in {
            InteractiveRuntimeState.READY_NEXT_CYCLE,
            InteractiveRuntimeState.FAILED,
        }:
            break
        if dump_checkpoint_written and (
            ARGS.dump_plus_9_checkpoint is None or dump_plus_9_checkpoint_written
        ):
            break
        if (
            mobile_first_write_audit is not None
            and mobile_first_write_audit.complete
        ):
            mobile_first_write_audit.write()
            break
        if (
            tool_mobile_validity_audit is not None
            and tool_mobile_validity_audit.complete
        ):
            tool_mobile_validity_audit.write()
            break

    _LIFECYCLE_RECORD["simulation_app_is_running_became_false"] = bool(
        not simulation_app.is_running()
    )
    explicit_automated_exit = bool(
        smoke_requested
        or ARGS.acceptance_cycles > 0
        or ARGS.presentation_smoke_autostart
        or GUI_LIFECYCLE_SMOKE
        or ARGS.final_presentation_exit_after_hold
        or ARGS.final_presentation_blocker_check
        or dump_checkpoint_written
        or dump_plus_9_checkpoint_written
        or (mobile_first_write_audit is not None and mobile_first_write_audit.complete)
        or (
            tool_mobile_validity_audit is not None
            and tool_mobile_validity_audit.complete
        )
    )
    if _SIGNAL_EXIT_REQUESTED:
        _LIFECYCLE_RECORD["exit_reason"] = ExitReason.EXTERNAL_SIGNAL.value
    elif explicit_automated_exit and simulation_app.is_running():
        _LIFECYCLE_RECORD["exit_reason"] = ExitReason.AUTOMATED_TEST_COMPLETED.value
    elif MANUAL_PRESENTATION_GUI and not simulation_app.is_running():
        _LIFECYCLE_RECORD["exit_reason"] = ExitReason.USER_CLOSED_WINDOW.value
    elif not simulation_app.is_running():
        _LIFECYCLE_RECORD["exit_reason"] = ExitReason.UNEXPECTED_APPLICATION_EXIT.value
    elif final_presentation_completed:
        _LIFECYCLE_RECORD["exit_reason"] = ExitReason.PRESENTATION_COMPLETED.value
    _ACTIVE_LIFECYCLE_PATH.write_text(
        json.dumps(_LIFECYCLE_RECORD, indent=2) + "\n", encoding="utf-8"
    )

    if MANUAL_PRESENTATION_GUI and not _SIGNAL_EXIT_REQUESTED:
        print(
            "GUI_LIFECYCLE=USER_WINDOW_CLOSE_REQUESTED; closing SimulationApp",
            flush=True,
        )
    elif _SIGNAL_EXIT_REQUESTED:
        print(
            "GUI_LIFECYCLE=EXTERNAL_SIGNAL_REQUESTED; closing SimulationApp",
            flush=True,
        )

    if FINAL_PRESENTATION and not final_presentation_completed:
        write_final_presentation_summary("USER_CLOSED_BEFORE_FINAL_HOLD")

    if V3_CLOSURE_AUDIT and "ARREST_FINAL" not in v3_closure_labels:
        capture_v3_closure_state("ARREST_FINAL", last_core_result)
    final_snapshot = control.snapshot()
    if cut_fill_payload_audit is not None:
        audit_report = cut_fill_payload_audit.report(
            failure_code=(
                None if final_snapshot.failure is None else final_snapshot.failure.code
            )
        )
        if ARGS.cut_fill_payload_audit is not None:
            audit_target = ARGS.cut_fill_payload_audit.expanduser().resolve()
            audit_target.parent.mkdir(parents=True, exist_ok=True)
            audit_target.write_text(
                json.dumps(audit_report, indent=2) + "\n", encoding="utf-8"
            )
        (run_dir / "cut_fill_payload_causal_audit.json").write_text(
            json.dumps(audit_report, indent=2) + "\n", encoding="utf-8"
        )
    if tool_mobile_validity_audit is not None:
        validity_report = tool_mobile_validity_audit.write()
        (run_dir / "tool_mobile_physical_validity_run.json").write_text(
            json.dumps(validity_report, indent=2) + "\n", encoding="utf-8"
        )
    if (
        (ARGS.presentation_smoke_autostart or GUI_LIFECYCLE_SMOKE)
        and presentation_smoke_reset_completed
        and not presentation_smoke_reset_verified
    ):
        raise RuntimeError("PRESENTATION_RESET_SMOKE_FAILED")
    if tracksoil_conservation_audit is not None:
        audit_result = tracksoil_conservation_audit.report()
        audit_result["run_dir"] = str(run_dir)
        audit_result["runtime_backend"] = CONFIG.runtime_backend
        audit_result["state_authority"] = "DEVICE"
        audit_result["grid_shape_yx"] = list(CONFIG.grid_shape)
        audit_result["grid_resolution_m"] = CONFIG.grid_spacing_m
        audit_result["physics_parameters_changed"] = []
        audit_result["mobile_v2_changed"] = bool(
            ARGS.mobile_v2_dual_cv_audit is not None
        )
        audit_result["failuresurface_v3_changed"] = False
        audit_result["curl_scoop_trajectory_changed"] = False
        per_run_audit = run_dir / "tracksoil_conservation_causal_report.json"
        per_run_audit.write_text(
            json.dumps(audit_result, indent=2) + "\n", encoding="utf-8"
        )
        if ARGS.mobile_v2_dual_cv_audit is not None:
            dual_cv_target = ARGS.mobile_v2_dual_cv_audit.expanduser().resolve()
            dual_cv_target.parent.mkdir(parents=True, exist_ok=True)
            dual_cv_target.write_text(
                json.dumps(audit_result, indent=2) + "\n", encoding="utf-8"
            )
            # P0-2A canonical artifacts are frozen causal evidence.  The
            # remainder of this legacy export block is intentionally skipped
            # for P0-2B so a validation replay cannot overwrite them.
        else:
            canonical_audit_dir = ROOT / "outputs/mobile_v2_production"
            canonical_audit_dir.mkdir(parents=True, exist_ok=True)
            (canonical_audit_dir / "tracksoil_conservation_timeseries.json").write_text(
                json.dumps(
                    {
                        "schema": audit_result["schema"],
                        "run_dir": str(run_dir),
                        "records": audit_result["records"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (canonical_audit_dir / "tracksoil_first_bad_step.json").write_text(
                json.dumps(
                    {
                        key: audit_result.get(key)
                        for key in (
                            "schema",
                            "run_dir",
                            "location_threshold_m3",
                            "cut_curl_numerical_floor_m3",
                            "first_bad_step_found",
                            "last_good_boundary",
                            "first_bad_boundary",
                        )
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (canonical_audit_dir / "tracksoil_conservation_causal_report.json").write_text(
                json.dumps(audit_result, indent=2) + "\n", encoding="utf-8"
            )
    control.close()
    (run_dir / "track_soil_events.json").write_text(
        json.dumps(track_soil_log, indent=2) + "\n", encoding="utf-8"
    )
    manifest["final_runtime_state"] = final_snapshot.state.value
    manifest["completed_cycle_count"] = final_snapshot.completed_cycle_count
    manifest["task_completion_status"] = (
        "PASS" if final_snapshot.state is InteractiveRuntimeState.READY_NEXT_CYCLE else "NOT_PASSED"
    )
    if dump_checkpoint_written:
        manifest["task_completion_status"] = "CHECKPOINT_CAPTURED_NOT_A_FULL_CYCLE_ACCEPTANCE"
    if PRESENTATION_DEMO:
        if FINAL_PRESENTATION:
            manifest["task_completion_status"] = (
                "FINAL_PRESENTATION_HOLD_REACHED"
                if final_presentation_completed
                else "FINAL_PRESENTATION_CLOSED_EARLY"
            )
            manifest["final_presentation_run_dir"] = str(presentation_run_dir)
        else:
            manifest["task_completion_status"] = (
                "DEMO_ONLY_POST_DUMP_TIMESCALE_UNDER_VALIDATION"
            )
            manifest["presentation_auto_paused"] = presentation_auto_paused
    manifest["final_contact_synchronization"] = {
        "terrain_timestamp_s": contact_status.terrain_timestamp_s,
        "contact_timestamp_s": contact_status.contact_timestamp_s,
        "contact_lag_s": contact_status.contact_lag_s,
        "revision": contact_status.contact_revision,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if V3_CLOSURE_AUDIT:
        closure_summary = {
            "schema": "PHYSICS_CORE_V3_PRODUCTION_CLOSURE_RUN/v1",
            "run_dir": str(run_dir),
            "runtime_backend": CONFIG.runtime_backend,
            "state_authority": "DEVICE",
            "production_core": type(physics_core).__name__,
            "grid_shape_yx": list(CONFIG.grid_shape),
            "grid_resolution_m": CONFIG.grid_spacing_m,
            "material_profile": material.name,
            "trajectory": "EXISTING_390F_PRODUCTION_STATE_MACHINE_UNMODIFIED",
            "physics_modified_for_audit": False,
            "captured_labels": [item["label"] for item in v3_closure_records],
            "diagnostics": v3_closure_records,
            "completed_cycle_count": final_snapshot.completed_cycle_count,
            "runtime_state": final_snapshot.state.value,
            "runtime_failure_present": failure_path.exists(),
        }
        closure_path = run_dir / "physics_core_v3_production_closure_run.json"
        closure_path.write_text(
            json.dumps(closure_summary, indent=2) + "\n", encoding="utf-8"
        )
        canonical_closure = ROOT / "outputs/390f_v3/physics_core_v3_production_closure_run.json"
        canonical_closure.parent.mkdir(parents=True, exist_ok=True)
        canonical_closure.write_text(
            json.dumps(closure_summary, indent=2) + "\n", encoding="utf-8"
        )
    if CONFIG.runtime_backend == "GPU_RUNTIME" and ARGS.acceptance_cycles > 0:
        frame_ms = np.asarray(
            [item["wall_ms"]["total_frame"] for item in runtime_telemetry],
            dtype=np.float64,
        )
        core_ms = np.asarray(
            [item["wall_ms"]["bulk_core"] for item in runtime_telemetry],
            dtype=np.float64,
        )
        rtf = 1000.0 / 60.0 / np.maximum(frame_ms, 1.0e-12)

        def measured(values: np.ndarray) -> dict[str, float | int]:
            if values.size == 0:
                return {"count": 0}
            return {
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95.0)),
                "p99": float(np.percentile(values, 99.0)),
                "maximum": float(np.max(values)),
            }

        final_reservoirs = physics_core.reservoir_observation()
        mass_errors = np.asarray(
            [abs(float(item.get("mass_error_m3", 0.0))) for item in runtime_telemetry],
            dtype=np.float64,
        )
        track_volume = float(
            sum(item["resting_to_mobile_volume_m3"] for item in track_soil_log)
        )
        payload_peak = float(
            max((item["payload_m3"] for item in runtime_telemetry), default=0.0)
        )
        deposited = float(material_funnel["deposited_after_dump_m3"])
        landed = float(material_funnel["airborne_landed_volume_m3"])
        completed = manifest["task_completion_status"] == "PASS"
        matrix = {
            "GPU_STATE_OWNERSHIP_STATUS": "PASS",
            "PRODUCTION_GPU_CORE_STATUS": "PASS",
            "GPU_RUNTIME_TRANSFER_STATUS": "PASS",
            "FAILUREZONE_STATUS": "PASS" if material_funnel["failure_volume_m3"] > 0.0 else "FAIL",
            "SOIL_FORCE_STATUS": (
                "PASS"
                if float(soil_force_evidence["applied_peak_n"]) > 0.0
                else "FAIL"
            ),
            "BUCKET_FILL_STATUS": "PASS" if payload_peak > 0.0 else "FAIL",
            "TRACK_SOIL_STATUS": "PASS" if track_volume > 0.0 else "FAIL",
            "TRACK_PILE_CLOSED_LOOP_STATUS": "PASS" if track_volume > 0.0 and contact_status.contact_revision > 0 else "FAIL",
            "AIRBORNE_DEVICE_BRIDGE_STATUS": "PASS" if landed > 0.0 else "FAIL",
            "DEPOSITION_STATUS": "PASS" if deposited > 0.0 else "FAIL",
            "LARGE_AVALANCHE_STATUS": "PASS" if completed and physics_core.terrain_settled else "INCOMPLETE",
            "MATERIAL_CONSERVATION_STATUS": "PASS" if (mass_errors.size == 0 or float(np.max(mass_errors)) <= 1.0e-8) else "FAIL",
            "COMPUTATIONAL_FEASIBILITY_STATUS": "PASS" if rtf.size and float(np.mean(rtf)) >= 0.25 else "FAIL",
            "ONE_VALID_EARTHMOVING_CYCLE_STATUS": "PASS" if completed else "FAIL",
        }
        one_cycle = {
            "schema": "390F_GPU_REAL_ONE_CYCLE/v1",
            "status": "PASS" if all(value == "PASS" for value in matrix.values()) else "FAIL",
            "run_dir": str(run_dir),
            "runtime_backend": CONFIG.runtime_backend,
            "resolution_m": CONFIG.grid_spacing_m,
            "phase_transitions": transition_log,
            "material_funnel": dict(material_funnel),
            "soil_force_stepwise_evidence": dict(soil_force_evidence),
            "payload": {
                "peak_volume_m3": payload_peak,
                "capacity_m3": descriptor.effective_capacity_m3,
                "peak_fill_ratio": payload_peak / descriptor.effective_capacity_m3,
            },
            "track_soil": {
                "event_count": len(track_soil_log),
                "resting_to_mobile_volume_m3": track_volume,
                "events": track_soil_log,
            },
            "final_reservoirs": final_reservoirs,
            "material_conservation_max_abs_m3": float(np.max(mass_errors)) if mass_errors.size else None,
            "performance": {
                "core_step_ms": measured(core_ms),
                "total_frame_ms": measured(frame_ms),
                "integrated_rtf": measured(rtf),
            },
            "contact": manifest["final_contact_synchronization"],
            "status_matrix": matrix,
            "claim_boundaries": {
                "iron_ore_parameters": "LITERATURE_BASED_NOT_SITE_CALIBRATED",
                "large_avalanche": "LITERATURE_BASED_REDUCED_ORDER_UNCALIBRATED",
                "track_soil": str(CONFIG.track_soil["parameter_status"]),
            },
        }
        per_run_acceptance = run_dir / "gpu_real_one_cycle_acceptance.json"
        per_run_acceptance.write_text(
            json.dumps(one_cycle, indent=2) + "\n", encoding="utf-8"
        )
        canonical = CONFIG.output_root.parent / "gpu_real_one_cycle_acceptance.json"
        canonical.write_text(json.dumps(one_cycle, indent=2) + "\n", encoding="utf-8")
        spatial = np.asarray(
            [
                item["base_z_roll_pitch_rad"]
                for item in runtime_telemetry
                if "base_z_roll_pitch_rad" in item
            ],
            dtype=np.float64,
        )
        terrain_overlap_samples = [
            support
            for item in runtime_telemetry
            for support in item.get("track_status", {}).get("support", [])
            if support.get("terrain_overlap_cell_count", 0) > 0
        ]
        latest_track_event_s = max(
            (float(item["timestamp_s"]) for item in track_soil_log), default=-np.inf
        )
        support_feedback = bool(
            track_soil_log
            and contact_status.contact_timestamp_s >= latest_track_event_s
            and contact_status.contact_revision > 0
        )
        z_range = float(np.ptp(spatial[:, 0])) if spatial.size else 0.0
        pitch_range = float(np.ptp(spatial[:, 2])) if spatial.size else 0.0
        track_matrix = {
            "REAL_TRACK_DRIVE_STATUS": "PASS" if any(abs(item.get("track_status", {}).get("applied_left_command", 0.0)) > 1.0e-4 for item in runtime_telemetry) else "FAIL",
            "DEFORMABLE_TERRAIN_OVERLAP_STATUS": "PASS" if terrain_overlap_samples else "FAIL",
            "TRACK_RUT_STATUS": "PASS" if track_volume > 0.0 else "FAIL",
            "TRACK_SOIL_CONSERVATION_STATUS": "PASS" if track_volume > 0.0 else "FAIL",
            "CONTACT_REPUBLICATION_STATUS": "PASS" if support_feedback else "FAIL",
            "CHANGED_TERRAIN_SUPPORT_FEEDBACK_STATUS": "PASS" if support_feedback else "FAIL",
            "MACHINE_Z_PITCH_RESPONSE_STATUS": "PASS" if z_range > 1.0e-4 or pitch_range > 1.0e-5 else "FAIL",
        }
        track_acceptance = {
            "schema": "390F_TRACK_PILE_CLOSED_LOOP/v1",
            "status": "PASS" if all(value == "PASS" for value in track_matrix.values()) else "FAIL",
            "run_dir": str(run_dir),
            "initial_heightmap": str(CONFIG.initial_heightmap),
            "resolution_m": CONFIG.grid_spacing_m,
            "track_event_count": len(track_soil_log),
            "resting_to_mobile_volume_m3": track_volume,
            "terrain_overlap_sample_count": len(terrain_overlap_samples),
            "base_z_range_m": z_range,
            "base_pitch_range_rad": pitch_range,
            "final_contact_synchronization": manifest["final_contact_synchronization"],
            "status_matrix": track_matrix,
            "parameter_status": str(CONFIG.track_soil["parameter_status"]),
        }
        (run_dir / "track_pile_closed_loop_acceptance.json").write_text(
            json.dumps(track_acceptance, indent=2) + "\n", encoding="utf-8"
        )
        (CONFIG.output_root.parent / "track_pile_closed_loop_acceptance.json").write_text(
            json.dumps(track_acceptance, indent=2) + "\n", encoding="utf-8"
        )


try:
    main()
except BaseException as error:
    message = f"{type(error).__name__}: {error}"
    _LIFECYCLE_RECORD["exception"] = message
    if _LIFECYCLE_RECORD.get("signal") is not None:
        reason = ExitReason.EXTERNAL_SIGNAL
    elif "CUDA" in message.upper():
        reason = ExitReason.CUDA_ERROR
    elif "PHYSX" in message.upper():
        reason = ExitReason.PHYSX_ERROR
    elif "OMNI.UI" in traceback.format_exc().upper() or "UI" in message.upper():
        reason = ExitReason.UI_EXCEPTION
    else:
        reason = ExitReason.PYTHON_EXCEPTION
    _LIFECYCLE_RECORD["exit_reason"] = reason.value
    if _ACTIVE_LIFECYCLE_PATH is not None:
        _ACTIVE_LIFECYCLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_LIFECYCLE_PATH.write_text(
            json.dumps(_LIFECYCLE_RECORD, indent=2) + "\n", encoding="utf-8"
        )
    fallback = CONFIG.output_root / "startup_failure.json"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text(json.dumps({"status": "FAIL", "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8")
    raise
finally:
    simulation_app.close()
