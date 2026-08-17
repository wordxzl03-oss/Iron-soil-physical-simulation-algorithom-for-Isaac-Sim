"""Presentation-only trajectory timing for the production 390F runtime.

This module owns no robot or soil state.  It emits smooth articulation targets
which are still consumed by the production force/velocity/acceleration-limited
390F actuator and PhysX articulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


_ALLOWED_ROOT_KEYS = {
    "schema",
    "production_config",
    "trajectory",
    "camera",
    "hud",
    "visualization",
    "output",
}
_FORBIDDEN_ROOT_KEYS = {"physics", "material", "terrain", "grid", "soil"}


@dataclass(frozen=True)
class PresentationWaypoint:
    label: str
    physics_phase: str
    duration_s: float
    target_rad: np.ndarray


@dataclass(frozen=True)
class PresentationDemoConfig:
    path: Path
    production_config: Path
    waypoints: tuple[PresentationWaypoint, ...]
    camera_eye_m: np.ndarray
    camera_target_m: np.ndarray
    camera_focal_length_mm: float
    visual_sync_interval_steps: int
    capture_labels: tuple[str, ...]
    blocker_post_breakout_s: float
    output_root: Path

    @property
    def duration_s(self) -> float:
        return float(sum(item.duration_s for item in self.waypoints))

    @classmethod
    def load(cls, path: str | Path) -> "PresentationDemoConfig":
        config_path = Path(path).expanduser().resolve()
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("[390FPresentation] config root must be a mapping")
        unknown = set(data) - _ALLOWED_ROOT_KEYS
        forbidden = set(data) & _FORBIDDEN_ROOT_KEYS
        if unknown or forbidden:
            raise ValueError(
                "[390FPresentation] presentation config may not override "
                f"production physics/material/terrain; invalid keys={sorted(unknown | forbidden)}"
            )
        root = config_path.parents[1] if config_path.parent.name == "configs" else config_path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

        trajectory = data.get("trajectory", {})
        raw_waypoints = trajectory.get("waypoints", ())
        if not isinstance(raw_waypoints, list) or not raw_waypoints:
            raise ValueError("[390FPresentation] trajectory.waypoints must be non-empty")
        waypoints: list[PresentationWaypoint] = []
        for raw in raw_waypoints:
            label = str(raw["label"]).upper()
            phase = str(raw["physics_phase"])
            duration = float(raw["duration_s"])
            target = np.deg2rad(np.asarray(raw["target_deg"], dtype=np.float64))
            if target.shape != (4,) or not np.all(np.isfinite(target)):
                raise ValueError(f"[390FPresentation] {label} target_deg must be finite [4]")
            if not np.isfinite(duration) or duration <= 0.0:
                raise ValueError(f"[390FPresentation] {label} duration must be positive")
            target = np.ascontiguousarray(target)
            target.setflags(write=False)
            waypoints.append(PresentationWaypoint(label, phase, duration, target))
        labels = tuple(item.label for item in waypoints)
        required = {"APPROACH", "PENETRATION", "ADVANCE_AND_CURL", "BUCKET_FILL", "BREAKOUT", "LIFT", "HOLD"}
        missing = sorted(required - set(labels))
        if missing:
            raise ValueError(f"[390FPresentation] missing required stages: {missing}")
        duration = sum(item.duration_s for item in waypoints)
        if not 50.0 <= duration <= 70.0:
            raise ValueError(
                f"[390FPresentation] trajectory duration must be 50-70 s, got {duration:.3f}"
            )
        camera = data.get("camera", {})
        eye = np.asarray(camera["eye_m"], dtype=np.float64)
        target = np.asarray(camera["target_m"], dtype=np.float64)
        if eye.shape != (3,) or target.shape != (3,) or not np.all(np.isfinite(np.r_[eye, target])):
            raise ValueError("[390FPresentation] camera eye/target must be finite [3]")
        visualization = data.get("visualization", {})
        interval = int(visualization.get("visual_sync_interval_steps", 3))
        if interval < 1:
            raise ValueError("[390FPresentation] visual sync interval must be >= 1")
        blocker_s = float(visualization.get("blocker_post_breakout_s", 8.0))
        if not 5.0 <= blocker_s <= 10.0:
            raise ValueError("[390FPresentation] blocker post-breakout interval must be 5-10 s")
        return cls(
            path=config_path,
            production_config=resolve(str(data["production_config"])),
            waypoints=tuple(waypoints),
            camera_eye_m=eye,
            camera_target_m=target,
            camera_focal_length_mm=float(camera.get("focal_length_mm", 32.0)),
            visual_sync_interval_steps=interval,
            capture_labels=tuple(str(item).upper() for item in visualization.get("capture_labels", ())),
            blocker_post_breakout_s=blocker_s,
            output_root=resolve(str(data.get("output", {}).get("root", "outputs/390f_presentation_demo"))),
        )


@dataclass(frozen=True)
class PresentationTrajectorySample:
    label: str
    physics_phase: str
    target_rad: np.ndarray
    elapsed_s: float
    stage_elapsed_s: float
    transitioned: bool
    complete: bool


class PresentationTrajectory:
    """C1-smooth, time-driven presentation targets with no pose writes."""

    def __init__(self, config: PresentationDemoConfig, initial_target_rad: np.ndarray) -> None:
        self.config = config
        initial = np.asarray(initial_target_rad, dtype=np.float64)
        if initial.shape != (4,) or not np.all(np.isfinite(initial)):
            raise ValueError("[390FPresentation] initial target must be finite [4]")
        self._initial = initial.copy()
        self.reset(initial)

    def reset(self, initial_target_rad: np.ndarray | None = None) -> None:
        if initial_target_rad is not None:
            self._initial = np.asarray(initial_target_rad, dtype=np.float64).copy()
        self.elapsed_s = 0.0
        self._previous_index = -1

    def sample(self, dt_s: float) -> PresentationTrajectorySample:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[390FPresentation] dt_s must be finite/positive")
        self.elapsed_s = min(self.elapsed_s + dt, self.config.duration_s)
        start_time = 0.0
        index = len(self.config.waypoints) - 1
        for candidate, waypoint in enumerate(self.config.waypoints):
            end = start_time + waypoint.duration_s
            if self.elapsed_s <= end or candidate == len(self.config.waypoints) - 1:
                index = candidate
                break
            start_time = end
        waypoint = self.config.waypoints[index]
        stage_elapsed = max(0.0, self.elapsed_s - start_time)
        fraction = min(1.0, stage_elapsed / waypoint.duration_s)
        # Cubic smoothstep avoids velocity discontinuities at waypoint joins.
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        start_target = self._initial if index == 0 else self.config.waypoints[index - 1].target_rad
        target = start_target + blend * (waypoint.target_rad - start_target)
        transitioned = index != self._previous_index
        self._previous_index = index
        result = np.ascontiguousarray(target)
        result.setflags(write=False)
        return PresentationTrajectorySample(
            label=waypoint.label,
            physics_phase=waypoint.physics_phase,
            target_rad=result,
            elapsed_s=float(self.elapsed_s),
            stage_elapsed_s=float(stage_elapsed),
            transitioned=transitioned,
            complete=bool(self.elapsed_s >= self.config.duration_s),
        )
