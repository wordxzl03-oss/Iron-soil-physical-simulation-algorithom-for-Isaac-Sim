"""Phase-B flat-ground loader demo using sparse articulation commands.

Interactive controls: W/S drive, A/D steer, I/K lift, J/L curl/dump pose,
Space brake, R clear commands, Esc stop.  ``--headless --scripted`` provides a
reproducible acceptance run; neither path writes the chassis/root pose.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="physics steps; 0 means run interactively until Esc",
    )
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument(
        "--usd", type=Path, default=Path(__file__).with_name("wheel_loader.usd")
    )
    parser.add_argument(
        "--vehicle-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "phase_b_vehicle.yaml",
    )
    parser.add_argument("--lighting-preset", default="outdoor_day")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help="optional directory for headless RTX Phase-B visual evidence",
    )
    parser.add_argument(
        "--h0-path",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "isaac_interactive_run"
            / "episode_0002"
            / "H_initial.npy"
        ),
        help="authoritative 701x701 H0 loaded as a visual-only mesh for captures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "phase_b_manual_runtime.json",
    )
    args, kit_args = parser.parse_known_args()
    if args.steps < 0 or (args.scripted and args.steps < 1):
        parser.error("--steps must be >= 0 and scripted mode requires at least one step")
    if args.physics_dt <= 0.0:
        parser.error("--physics-dt must be positive")
    if args.headless and not args.scripted:
        parser.error("headless mode requires --scripted because no keyboard window exists")
    return args, kit_args


def _scripted_command(frame: int):
    from isaac_bulk_pipeline.vehicle import VehicleCommand

    # Settle, drive straight, exercise all three position actuators while
    # driving, then brake.  This is acceptance input, not an excavation cycle.
    if frame < 30:
        return VehicleCommand(brake=1.0)
    if frame < 180:
        return VehicleCommand(throttle=1.0)
    if frame < 300:
        return VehicleCommand(
            throttle=0.45,
            steering=0.45,
            lift=0.55,
            bucket_curl=0.65,
        )
    if frame < 450:
        return VehicleCommand(
            throttle=-0.15,
            steering=-0.35,
            lift=-0.50,
            bucket_curl=-0.80,
        )
    return VehicleCommand(brake=1.0)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * fraction)), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    args, kit_args = _parse_args()
    sys.argv = [sys.argv[0], *kit_args]
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "multi_gpu": False,
            "width": 960,
            "height": 540,
            # FXAA is deterministic for paused evidence frames and avoids the
            # temporal trails that DLSS/TAA can retain after loader motion.
            "anti_aliasing": 2,
        }
    )

    import numpy as np
    import yaml
    from isaacsim.core.api import World
    # Isaac Sim 4.5 exposes the controller on SingleArticulation; the plural
    # Articulation view intentionally has no get_articulation_controller().
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.async_engine import run_coroutine
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    from isaac_bulk_pipeline.bulk_state import SurfaceTopology, TerrainVolumeIntegrator
    from isaac_bulk_pipeline.config import MeshConfig
    from isaac_bulk_pipeline.terrain import TerrainGrid
    from isaac_bulk_pipeline.vehicle import (
        CommandSlewLimits,
        CommandSlewLimiter,
        LoaderControlConfig,
        LoaderLowLevelController,
        ManualLoaderController,
    )
    from isaac_bulk_pipeline.visualization import DynamicMeshAdapter
    from isaac_bulk_pipeline.visualization.lighting_manager import LightingManager
    from isaac_bulk_pipeline.visualization.terrain_material_adapter import TerrainMaterialAdapter

    keyboard_subscription = None
    input_interface = None
    keyboard = None
    summary: dict[str, Any] = {}
    try:
        document = yaml.safe_load(args.vehicle_config.read_text(encoding="utf-8"))
        config = LoaderControlConfig.from_mapping(document["vehicle"])
        slew_values = document["command_slew"]
        slew = CommandSlewLimiter(
            CommandSlewLimits(
                throttle_per_s=float(slew_values["throttle_per_s"]),
                steering_per_s=float(slew_values["steering_per_s"]),
                lift_per_s=float(slew_values["lift_per_s"]),
                bucket_curl_per_s=float(slew_values["bucket_curl_per_s"]),
            )
        )

        world = World(
            physics_dt=float(args.physics_dt),
            rendering_dt=float(args.physics_dt),
            stage_units_in_meters=1.0,
        )
        ground = world.scene.add_ground_plane(
            size=100.0,
            z_position=-0.03,
            static_friction=0.90,
            dynamic_friction=0.78,
            restitution=0.0,
        )
        add_reference_to_stage(str(args.usd.resolve()), "/World/WheelLoader")
        loader = world.scene.add(
            SingleArticulation(
                "/World/WheelLoader/rear_chassis", name="phase_b_wheel_loader"
            )
        )

        stage = omni.usd.get_context().get_stage()
        lighting = LightingManager(args.lighting_preset)
        lighting.author_usd(stage)

        # The default plane remains the only Phase-B physical support.  When
        # visual evidence is requested, show the authoritative episode H0 as a
        # separate derived mesh with no collision API.  It must not affect the
        # vehicle dynamics or become a second source of terrain truth.
        h0_visual_adapter = None
        h0_visual_metadata: dict[str, Any] | None = None
        wide_camera_metadata: dict[str, Any] | None = None
        if args.capture_dir is not None:
            h0_source = args.h0_path.expanduser().resolve()
            if not h0_source.is_file():
                raise FileNotFoundError(f"authoritative H0 not found: {h0_source}")
            h0 = np.load(h0_source, allow_pickle=False)
            if h0.shape != (701, 701):
                raise ValueError(
                    f"authoritative H0 must have shape (701, 701); got {h0.shape}"
                )
            if not np.issubdtype(h0.dtype, np.number) or not np.all(np.isfinite(h0)):
                raise ValueError("authoritative H0 must contain only finite numbers")
            if float(np.min(h0)) < 0.0:
                raise ValueError("authoritative H0 must contain non-negative metre heights")

            h0_grid = TerrainGrid(
                nx=701,
                ny=701,
                dx=0.05,
                dy=0.05,
                origin_x=0.0,
                origin_y=0.0,
                terrain_prim_path="/World/PhaseB/H0Visual",
            )
            h0_visual_adapter = DynamicMeshAdapter()
            h0_visual_adapter.initialize(
                stage,
                h0_grid,
                MeshConfig(
                    collision_enabled=False,
                    update_normals=True,
                    subdivision_scheme="none",
                    mesh_update_rate_hz=10.0,
                    normal_update_rate_hz=5.0,
                    double_sided=True,
                    display_color_rgb=(0.24, 0.075, 0.035),
                ),
                h0,
            )
            material_adapter = TerrainMaterialAdapter("iron_ore_fines")
            material = material_adapter.author_and_bind_usd(
                stage, h0_grid.terrain_prim_path
            )
            ground_visual_material = material_adapter.author_and_bind_usd(
                stage, ground.prim_path
            )
            h0_prim = stage.GetPrimAtPath(h0_grid.terrain_prim_path)
            visual_no_collision = not h0_prim.HasAPI(UsdPhysics.CollisionAPI)
            if not visual_no_collision:
                raise RuntimeError("authoritative H0 visual unexpectedly has CollisionAPI")
            ground_collision_prim_paths = sorted(
                str(prim.GetPath())
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(f"{ground.prim_path}/")
                and prim.HasAPI(UsdPhysics.CollisionAPI)
            )
            if not ground_collision_prim_paths:
                raise RuntimeError("default ground has no collision Prim")
            volume_integrator = TerrainVolumeIntegrator.from_grid(
                h0_grid,
                topology=SurfaceTopology.TRIANGLE_A_C,
            )
            strict_h0_volume_m3 = volume_integrator.integrate(h0)
            h0_visual_metadata = {
                "authoritative": True,
                "source": str(h0_source),
                "source_relative_to_repository": str(
                    h0_source.relative_to(REPOSITORY_ROOT)
                ),
                "heightmap_axis_order": "yx",
                "shape_yx": [701, 701],
                "dtype": str(h0.dtype),
                "grid_spacing_m_xy": [0.05, 0.05],
                "grid_origin_m_xy": [0.0, 0.0],
                "grid_span_m_xy": [35.0, 35.0],
                "height_range_m": [float(np.min(h0)), float(np.max(h0))],
                "volume_m3": strict_h0_volume_m3,
                "volume_integration": {
                    "authoritative": True,
                    "integrator": "TerrainVolumeIntegrator",
                    "topology": SurfaceTopology.TRIANGLE_A_C.value,
                    "matches_dynamic_mesh_diagonal": True,
                    "domain_area_m2": volume_integrator.domain_area_m2,
                },
                "prim_path": h0_grid.terrain_prim_path,
                "visual_no_collision": visual_no_collision,
                "physical_support_prim": ground.prim_path,
                "physical_support_role": "default_ground_only",
                "physical_support_z_m": -0.03,
                "physical_support_collision_prim_paths": (
                    ground_collision_prim_paths
                ),
                "terrain_support_collision_source_count": 1,
                "material": material,
                "ground_visual_material": ground_visual_material,
                "ground_and_h0_share_visual_material": True,
            }
            wide_camera_metadata = {
                "mode": "fixed_wide_overview",
                "eye_m": [17.5, -27.0, 32.0],
                "target_m": [17.5, 14.0, 2.5],
                "focal_length_mm": 17.0,
                "purpose": "show full 35 m irregular H0 footprint and wheel loader",
            }
        world.reset()

        dof_names = tuple(loader.dof_names)
        low_level = LoaderLowLevelController(dof_names, config)
        positions = np.asarray(loader.get_joint_positions(), dtype=np.float64).reshape(-1)
        low_level.reset(positions)
        articulation_controller = loader.get_articulation_controller()
        manual = ManualLoaderController(slew_limiter=slew)

        capture_records: list[dict[str, Any]] = []
        capture_schedule = {
            20: (
                "H0",
                "phase_b_H0.png",
                "authoritative_H_initial_visual_and_settled_vehicle",
                False,
            ),
            150: (
                "drive",
                "phase_b_drive.png",
                "real_wheel_drive_in_progress",
                False,
            ),
            285: (
                "lift_curl",
                "phase_b_lift_curl.png",
                "lift_and_bucket_curl_visual_pose",
                False,
            ),
            455: (
                "dump_pose",
                "phase_b_dump_pose.png",
                "bucket_dump_pose_only",
                False,
            ),
        }
        if args.capture_dir is not None:
            args.capture_dir.mkdir(parents=True, exist_ok=True)
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("active viewport is unavailable for requested screenshots")
            viewport.resolution = (960, 540)

        def capture_evidence(
            label: str,
            filename: str,
            scope: str,
            material_interaction: bool,
        ) -> None:
            if args.capture_dir is None:
                return
            from PIL import Image
            import omni.kit.renderer_capture

            destination = args.capture_dir / filename
            assert wide_camera_metadata is not None
            # Only the camera Prim is moved. The fixed wide view keeps both the
            # complete 35 m H0 domain and the PhysX-driven loader in frame.
            set_camera_view(
                eye=np.asarray(wide_camera_metadata["eye_m"], dtype=np.float64),
                target=np.asarray(wide_camera_metadata["target_m"], dtype=np.float64),
                camera_prim_path="/OmniverseKit_Persp",
            )
            camera = UsdGeom.Camera.Get(stage, "/OmniverseKit_Persp")
            if not camera or not camera.GetPrim().IsValid():
                raise RuntimeError("wide overview camera Prim is unavailable")
            camera.GetFocalLengthAttr().Set(wide_camera_metadata["focal_length_mm"])
            world.pause()
            # Let RTX temporal history converge on this paused physical state
            # before requesting the file, avoiding motion trails in evidence.
            for _ in range(20):
                simulation_app.update()
            capture = capture_viewport_to_file(
                get_active_viewport(), file_path=str(destination), is_hdr=False
            )
            pending = run_coroutine(capture.wait_for_result(completion_frames=60))
            for _ in range(120):
                if pending.done():
                    break
                simulation_app.update()
            if not pending.done() or not bool(pending.result()):
                raise RuntimeError(f"viewport capture timed out: {label}")
            omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
            world.play()
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError(f"empty screenshot: {destination}")
            rgb = np.asarray(Image.open(destination).convert("RGB"), dtype=np.float32) / 255.0
            luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            brightness = {
                "mean_luma": float(luma.mean()),
                "p05_luma": float(np.percentile(luma, 5.0)),
                "p95_luma": float(np.percentile(luma, 95.0)),
            }
            readable = brightness["mean_luma"] > 0.05 and (
                brightness["p95_luma"] - brightness["p05_luma"]
            ) > 0.08
            if not readable:
                raise RuntimeError(
                    f"screenshot failed brightness/readability check: {label} {brightness}"
                )
            capture_records.append(
                {
                    "label": label,
                    "filename": filename,
                    "scope": scope,
                    "material_interaction": material_interaction,
                    "visual_pose_only": label in {"lift_curl", "dump_pose"},
                    "authoritative_h0_visible": True,
                    "h0_visual_no_collision": True,
                    "brightness": brightness,
                    "readable": readable,
                    "size_bytes": destination.stat().st_size,
                }
            )

        if not args.headless:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            input_interface = carb.input.acquire_input_interface()
            keyboard = app_window.get_keyboard()

            def on_keyboard(event) -> bool:
                if event.type in (
                    carb.input.KeyboardEventType.KEY_PRESS,
                    carb.input.KeyboardEventType.KEY_REPEAT,
                ):
                    manual.on_key_event(event.input, True)
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    manual.on_key_event(event.input, False)
                return True

            keyboard_subscription = input_interface.subscribe_to_keyboard_events(
                keyboard, on_keyboard
            )

        initial_position, _ = loader.get_world_pose()
        initial_position = np.asarray(initial_position, dtype=np.float64).reshape(-1)
        frame_records: list[dict[str, Any]] = []
        controller_times_ms: list[float] = []
        observed_position_min = positions.copy()
        observed_position_max = positions.copy()
        root_z_min = float(initial_position[2])
        root_z_max = float(initial_position[2])
        previous_root_position = initial_position.copy()
        maximum_planar_step_m = 0.0
        nonzero_wheel_command_frames = 0
        installed_wheel_effort_limits: list[float] = []
        command_envelope_power_bounds: list[float] = []
        power_cap_worst_case_omegas: list[float] = []
        power_cap_denominators: list[float] = []
        post_step_measured_speed_power_bounds: list[float] = []
        post_step_measured_wheel_omegas: list[float] = []
        power_cap_basis_counts: dict[str, int] = {}
        strongest_measured_cap_sample: dict[str, Any] | None = None
        worst_post_step_power_sample: dict[str, Any] | None = None
        measured_joint_velocities_for_control = np.asarray(
            loader.get_joint_velocities(), dtype=np.float64
        ).reshape(-1)
        if (
            measured_joint_velocities_for_control.shape != positions.shape
            or not np.all(np.isfinite(measured_joint_velocities_for_control))
        ):
            raise RuntimeError(
                "initial measured joint velocities must match the finite articulation DOFs"
            )
        stopped_frame = None

        frame_iterator = itertools.count() if args.steps == 0 else range(args.steps)
        for frame in frame_iterator:
            if args.scripted:
                command = slew.step(_scripted_command(frame), args.physics_dt)
            else:
                command = manual.physics_step(args.physics_dt)
                if manual.stopped:
                    stopped_frame = frame
                    break

            started = time.perf_counter()
            # This is the measured state returned after the previous PhysX
            # step (or the post-reset state for frame zero), never a target.
            power_cap_input_velocities = measured_joint_velocities_for_control.copy()
            targets = low_level.step(
                command,
                args.physics_dt,
                measured_joint_velocities=power_cap_input_velocities,
            )
            wheel = targets.wheel_action
            position = targets.position_action
            if any(abs(value) > 1e-9 for value in wheel.joint_velocities):
                nonzero_wheel_command_frames += 1
            assert wheel.effort_limits is not None
            installed_wheel_effort_limits.extend(wheel.effort_limits)
            command_envelope_power_bounds.extend(
                targets.wheel_command_envelope_power_bound_w
            )
            power_cap_worst_case_omegas.extend(
                targets.wheel_power_cap_worst_case_omega_rad_s
            )
            power_cap_denominators.extend(
                targets.wheel_power_cap_denominator_rad_s
            )
            power_cap_basis_counts[targets.wheel_power_cap_basis] = (
                power_cap_basis_counts.get(targets.wheel_power_cap_basis, 0) + 1
            )
            cap_sample = {
                "frame": frame,
                "basis": targets.wheel_power_cap_basis,
                "previous_measured_omega_rad_s": list(
                    targets.wheel_power_cap_previous_measured_omega_rad_s or ()
                ),
                "target_omega_rad_s": list(
                    targets.wheel_power_cap_target_omega_rad_s
                ),
                "worst_case_omega_rad_s": list(
                    targets.wheel_power_cap_worst_case_omega_rad_s
                ),
                "denominator_rad_s": list(
                    targets.wheel_power_cap_denominator_rad_s
                ),
                "discrete_safety_factor": (
                    targets.wheel_power_cap_discrete_safety_factor
                ),
                "installed_effort_limit_nm": list(wheel.effort_limits),
                "command_envelope_power_bound_w": list(
                    targets.wheel_command_envelope_power_bound_w
                ),
            }

            # Limits are actively installed on the sparse DOFs, not merely
            # carried as metadata in the pure-Python target object.
            articulation_controller.set_max_efforts(
                np.asarray(wheel.effort_limits, dtype=np.float32),
                joint_indices=np.asarray(wheel.joint_indices, dtype=np.int32),
            )
            articulation_controller.set_max_efforts(
                np.asarray(position.effort_limits, dtype=np.float32),
                joint_indices=np.asarray(position.joint_indices, dtype=np.int32),
            )
            articulation_controller.apply_action(
                ArticulationAction(
                    joint_velocities=np.asarray(wheel.joint_velocities, dtype=np.float32),
                    joint_indices=np.asarray(wheel.joint_indices, dtype=np.int32),
                )
            )
            articulation_controller.apply_action(
                ArticulationAction(
                    joint_positions=np.asarray(position.joint_positions, dtype=np.float32),
                    joint_velocities=np.asarray(position.joint_velocities, dtype=np.float32),
                    joint_indices=np.asarray(position.joint_indices, dtype=np.int32),
                )
            )
            controller_times_ms.append((time.perf_counter() - started) * 1000.0)
            world.step(render=not args.headless)

            observed_positions = np.asarray(
                loader.get_joint_positions(), dtype=np.float64
            ).reshape(-1)
            observed_velocities = np.asarray(
                loader.get_joint_velocities(), dtype=np.float64
            ).reshape(-1)
            root_position, root_orientation = loader.get_world_pose()
            root_position = np.asarray(root_position, dtype=np.float64).reshape(-1)
            root_orientation = np.asarray(root_orientation, dtype=np.float64).reshape(-1)
            root_linear_velocity = np.asarray(
                loader.get_linear_velocity(), dtype=np.float64
            ).reshape(-1)
            maximum_planar_step_m = max(
                maximum_planar_step_m,
                float(np.linalg.norm(root_position[:2] - previous_root_position[:2])),
            )
            previous_root_position = root_position.copy()
            root_z_min = min(root_z_min, float(root_position[2]))
            root_z_max = max(root_z_max, float(root_position[2]))
            observed_position_min = np.minimum(observed_position_min, observed_positions)
            observed_position_max = np.maximum(observed_position_max, observed_positions)
            if not (
                np.all(np.isfinite(observed_positions))
                and np.all(np.isfinite(observed_velocities))
                and np.all(np.isfinite(root_position))
                and np.all(np.isfinite(root_orientation))
                and np.all(np.isfinite(root_linear_velocity))
            ):
                raise RuntimeError(f"non-finite physics state at frame={frame}")
            post_step_measured_omegas = tuple(
                float(observed_velocities[index])
                for index in low_level.mapping.wheel_indices
            )
            post_step_power_bounds = tuple(
                effort_limit * abs(omega)
                for effort_limit, omega in zip(
                    wheel.effort_limits, post_step_measured_omegas
                )
            )
            post_step_measured_wheel_omegas.extend(post_step_measured_omegas)
            post_step_measured_speed_power_bounds.extend(post_step_power_bounds)
            cap_sample["post_step_measured_omega_rad_s"] = list(
                post_step_measured_omegas
            )
            cap_sample["post_step_measured_speed_power_bound_w"] = list(
                post_step_power_bounds
            )
            if (
                strongest_measured_cap_sample is None
                or min(wheel.effort_limits)
                < min(strongest_measured_cap_sample["installed_effort_limit_nm"])
            ):
                strongest_measured_cap_sample = dict(cap_sample)
            if (
                worst_post_step_power_sample is None
                or max(post_step_power_bounds)
                > max(
                    worst_post_step_power_sample[
                        "post_step_measured_speed_power_bound_w"
                    ]
                )
            ):
                worst_post_step_power_sample = dict(cap_sample)
            measured_joint_velocities_for_control = observed_velocities.copy()

            if frame % 10 == 0 or (args.steps > 0 and frame == args.steps - 1):
                frame_records.append(
                    {
                        "frame": frame,
                        "command": command.as_dict(),
                        "targets": targets.as_dict(),
                        "power_cap_input_joint_velocities_rad_s": (
                            power_cap_input_velocities.tolist()
                        ),
                        "post_step_power_validation": {
                            "measured_wheel_omega_rad_s": list(
                                post_step_measured_omegas
                            ),
                            "installed_effort_measured_speed_power_bound_w": list(
                                post_step_power_bounds
                            ),
                        },
                        "joint_positions_rad": observed_positions.tolist(),
                        "joint_velocities_rad_s": observed_velocities.tolist(),
                        "root_position_m": np.asarray(root_position).tolist(),
                        "root_linear_velocity_m_s": root_linear_velocity.tolist(),
                    }
                )
            if args.capture_dir is not None and frame in capture_schedule:
                capture_evidence(*capture_schedule[frame])

        final_position, _ = loader.get_world_pose()
        final_position = np.asarray(final_position, dtype=np.float64).reshape(-1)
        displacement = float(np.linalg.norm(final_position[:2] - initial_position[:2]))
        spans = observed_position_max - observed_position_min
        position_checks = {
            "steering": bool(spans[low_level.mapping.steering_index] > 0.03),
            "lift": bool(spans[low_level.mapping.lift_index] > 0.03),
            "bucket_curl": bool(spans[low_level.mapping.bucket_index] > 0.03),
        }
        bounded_checks = {
            "steering": bool(
                observed_position_min[low_level.mapping.steering_index]
                >= config.steering.lower_rad - 0.02
                and observed_position_max[low_level.mapping.steering_index]
                <= config.steering.upper_rad + 0.02
            ),
            "lift": bool(
                observed_position_min[low_level.mapping.lift_index]
                >= config.lift.lower_rad - 0.02
                and observed_position_max[low_level.mapping.lift_index]
                <= config.lift.upper_rad + 0.02
            ),
            "bucket_curl": bool(
                observed_position_min[low_level.mapping.bucket_index]
                >= config.bucket.lower_rad - 0.02
                and observed_position_max[low_level.mapping.bucket_index]
                <= config.bucket.upper_rad + 0.02
            ),
        }
        measured_basis = "MEASURED_OR_TARGET_WORST_CASE"
        fallback_basis = "TARGET_JOINT_VELOCITY_FALLBACK"
        measured_basis_every_frame = (
            len(controller_times_ms) > 0
            and power_cap_basis_counts.get(measured_basis, 0)
            == len(controller_times_ms)
            and power_cap_basis_counts.get(fallback_basis, 0) == 0
        )
        maximum_command_envelope_power_bound_w = max(
            command_envelope_power_bounds, default=0.0
        )
        command_envelope_power_bound_within_limit = bool(
            command_envelope_power_bounds
        ) and maximum_command_envelope_power_bound_w <= (
            config.wheel_power_limit_w * (1.0 + 1e-12)
        )
        post_step_power_tolerance_w = max(
            config.wheel_power_limit_w * 1e-6, 1e-6
        )
        maximum_post_step_measured_speed_power_bound_w = max(
            post_step_measured_speed_power_bounds, default=0.0
        )
        post_step_measured_speed_power_bound_within_limit = bool(
            post_step_measured_speed_power_bounds
        ) and maximum_post_step_measured_speed_power_bound_w <= (
            config.wheel_power_limit_w + post_step_power_tolerance_w
        )
        measured_speed_cap_activated = bool(installed_wheel_effort_limits) and min(
            installed_wheel_effort_limits
        ) < config.wheel_effort_nm - 1e-6
        acceptance_checks = {
            "root_displacement_over_0_05_m": displacement >= 0.05,
            "nonzero_wheel_command_observed": nonzero_wheel_command_frames > 0,
            "all_four_wheels_rotated": all(
                spans[index] > 0.5 for index in low_level.mapping.wheel_indices
            ),
            "steering_moved": position_checks["steering"],
            "lift_moved": position_checks["lift"],
            "bucket_curl_moved": position_checks["bucket_curl"],
            "position_actuators_within_configured_limits": all(bounded_checks.values()),
            "root_height_stable": root_z_max - root_z_min < 0.5,
            "root_motion_continuous": maximum_planar_step_m < 1.0,
            "normal_loop_pose_setter_zero": True,
            "finite_sparse_effort_limits_installed": bool(
                installed_wheel_effort_limits
            )
            and all(np.isfinite(installed_wheel_effort_limits)),
            "measured_or_target_worst_case_cap_used_every_frame": (
                measured_basis_every_frame
            ),
            "command_envelope_power_bound_within_limit": (
                command_envelope_power_bound_within_limit
            ),
            "post_step_measured_speed_power_bound_within_limit": (
                post_step_measured_speed_power_bound_within_limit
            ),
            "measured_or_target_power_cap_activated": measured_speed_cap_activated,
        }
        summary = {
            "schema_version": "isaac-bulk-phase-b-runtime/v1",
            "status": "PASS",
            "mode": "scripted_headless" if args.scripted else "interactive_keyboard",
            "steps_requested": None if args.steps == 0 else args.steps,
            "steps_completed": len(controller_times_ms),
            "physics_dt_s": args.physics_dt,
            "configuration": {
                "vehicle_config": str(args.vehicle_config.resolve()),
                "vehicle_config_relative": _repository_relative(
                    args.vehicle_config
                ),
                "vehicle_config_sha256": _sha256_file(
                    args.vehicle_config.resolve()
                ),
            },
            "dof_names": list(dof_names),
            "initial_root_position_m": initial_position.tolist(),
            "final_root_position_m": final_position.tolist(),
            "horizontal_root_displacement_m": displacement,
            "maximum_planar_step_m": maximum_planar_step_m,
            "root_height_range_m": root_z_max - root_z_min,
            "joint_position_min_rad": observed_position_min.tolist(),
            "joint_position_max_rad": observed_position_max.tolist(),
            "joint_position_span_rad": {
                name: float(spans[index]) for index, name in enumerate(dof_names)
            },
            "position_actuator_limit_checks": bounded_checks,
            "nonzero_wheel_command_frames": nonzero_wheel_command_frames,
            "wheel_power_limit_evidence": {
                "semantics": (
                    "each installed per-wheel effort limit uses protected denominator "
                    "discrete_safety_factor * max(previous-step measured |omega|, "
                    "target |omega|, reference, minimum guard); "
                    "after PhysX, installed limit times current measured |omega| is "
                    "validated as a mechanical-power upper bound. This does not "
                    "claim measured drive torque or measured applied power"
                ),
                "configured_per_wheel_power_limit_w": config.wheel_power_limit_w,
                "configured_per_wheel_torque_limit_nm": config.wheel_effort_nm,
                "low_speed_reference_rad_s": config.wheel_power_reference_rad_s,
                "minimum_speed_guard_rad_s": (
                    config.wheel_power_min_guard_rad_s
                ),
                "discrete_speed_safety_factor": (
                    config.wheel_power_discrete_safety_factor
                ),
                "basis_counts": power_cap_basis_counts,
                "measured_or_target_worst_case_frames": power_cap_basis_counts.get(
                    measured_basis, 0
                ),
                "target_speed_fallback_frames": power_cap_basis_counts.get(
                    fallback_basis, 0
                ),
                "installed_effort_limit_nm": {
                    "min": min(installed_wheel_effort_limits, default=0.0),
                    "max": max(installed_wheel_effort_limits, default=0.0),
                },
                "max_control_denominator_rad_s": max(
                    power_cap_denominators, default=0.0
                ),
                "max_protected_speed_before_safety_factor_rad_s": max(
                    power_cap_worst_case_omegas, default=0.0
                ),
                "max_command_envelope_power_bound_w": (
                    maximum_command_envelope_power_bound_w
                ),
                "command_envelope_bound_within_configured_limit": (
                    command_envelope_power_bound_within_limit
                ),
                "max_abs_post_step_measured_omega_rad_s": max(
                    (abs(value) for value in post_step_measured_wheel_omegas),
                    default=0.0,
                ),
                "max_post_step_measured_speed_power_bound_w": (
                    maximum_post_step_measured_speed_power_bound_w
                ),
                "post_step_power_bound_tolerance_w": post_step_power_tolerance_w,
                "post_step_bound_within_configured_limit": (
                    post_step_measured_speed_power_bound_within_limit
                ),
                "strongest_cap_sample": strongest_measured_cap_sample,
                "worst_post_step_power_bound_sample": (
                    worst_post_step_power_sample
                ),
            },
            "pose_setter_count_normal_loop": 0,
            "sparse_max_effort_installed_each_step": True,
            "controller_timing_ms": {
                "mean": float(sum(controller_times_ms) / max(len(controller_times_ms), 1)),
                "p50": _percentile(controller_times_ms, 0.50),
                "p95": _percentile(controller_times_ms, 0.95),
                "max": max(controller_times_ms, default=0.0),
            },
            "lighting": lighting.inspect_usd(stage),
            "screenshots": {
                "capture_requested": args.capture_dir is not None,
                "authoritative_h0_visual": h0_visual_metadata,
                "camera": wide_camera_metadata,
                "records": capture_records,
                "post_dig": {
                    "status": "DEFERRED_TO_PHASE_F",
                    "reason": "Phase B has no material interaction or post-dig terrain state",
                },
                "material_interaction_claimed": False,
            },
            "acceptance_checks": acceptance_checks,
            "stopped_frame": stopped_frame,
            "frames": frame_records,
        }
        failed_checks = sorted(
            name for name, passed in acceptance_checks.items() if not passed
        )
        if args.scripted and failed_checks:
            summary["status"] = "FAIL"
            summary["failed_checks"] = failed_checks
            summary["failure"] = f"Phase-B acceptance checks failed: {failed_checks}"
            raise RuntimeError(summary["failure"])
    except Exception as error:
        if not summary:
            summary = {
                "schema_version": "isaac-bulk-phase-b-runtime/v1",
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        else:
            summary["status"] = "FAIL"
            summary.setdefault("error_type", type(error).__name__)
            summary.setdefault("error", str(error))
        raise
    finally:
        if (
            input_interface is not None
            and keyboard is not None
            and keyboard_subscription is not None
        ):
            input_interface.unsubscribe_to_keyboard_events(
                keyboard,
                keyboard_subscription,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        simulation_app.close()
    print(f"PHASE_B_RUNTIME_{summary['status']}: {args.output}")


if __name__ == "__main__":
    main()
