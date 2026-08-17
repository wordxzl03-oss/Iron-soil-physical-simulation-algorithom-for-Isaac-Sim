"""Thin array-to-schema bridge for Isaac Articulation runtime observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .schema import (
    ActuatorCategory,
    JointTelemetrySample,
    VehiclePoseSample,
    VehicleTelemetryFrame,
    WheelKinematicsSample,
    WheelTerrainContactSample,
)


class IsaacTelemetryInputAdapter:
    """Convert public Isaac joint arrays into validated telemetry records.

    The adapter does not import Isaac modules. Callers pass arrays returned by
    the Articulation API, keeping command/applied, measured and estimated effort
    channels distinct. This makes the same conversion unit-testable in CPython.
    """

    def __init__(self, joint_categories: Mapping[str, ActuatorCategory | str]) -> None:
        if not joint_categories:
            raise ValueError("[IsaacTelemetryInputAdapter] joint_categories is empty")
        self._categories = {
            str(name): ActuatorCategory(category)
            for name, category in joint_categories.items()
        }
        if any(not name.strip() for name in self._categories):
            raise ValueError("[IsaacTelemetryInputAdapter] joint names must be non-empty")

    def build_joint_samples(
        self,
        *,
        joint_names: Sequence[str],
        angular_velocities_rad_s: Sequence[float] | np.ndarray,
        applied_efforts_nm: Sequence[float] | np.ndarray | None = None,
        measured_efforts_nm: Sequence[float] | np.ndarray | None = None,
        estimated_efforts_nm: Sequence[float] | np.ndarray | None = None,
        target_velocities_rad_s: Sequence[float] | np.ndarray | None = None,
    ) -> tuple[JointTelemetrySample, ...]:
        names = tuple(str(name) for name in joint_names)
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError(
                "[IsaacTelemetryInputAdapter] joint_names must be unique/non-empty"
            )
        missing = sorted(set(names) - set(self._categories))
        if missing:
            raise KeyError(
                "[IsaacTelemetryInputAdapter] missing joint classifications: "
                + ", ".join(missing)
            )
        velocity = self._array(
            angular_velocities_rad_s,
            expected=len(names),
            name="angular_velocities_rad_s",
        )
        applied = self._optional_array(
            applied_efforts_nm,
            expected=len(names),
            name="applied_efforts_nm",
        )
        measured = self._optional_array(
            measured_efforts_nm,
            expected=len(names),
            name="measured_efforts_nm",
        )
        estimated = self._optional_array(
            estimated_efforts_nm,
            expected=len(names),
            name="estimated_efforts_nm",
        )
        targets = self._optional_array(
            target_velocities_rad_s,
            expected=len(names),
            name="target_velocities_rad_s",
        )
        if applied is None and measured is None and estimated is None:
            raise ValueError(
                "[IsaacTelemetryInputAdapter] at least one effort channel is required"
            )
        return tuple(
            JointTelemetrySample(
                joint_name=name,
                category=self._categories[name],
                angular_velocity_rad_s=velocity[index],
                applied_effort_nm=None if applied is None else applied[index],
                measured_effort_nm=None if measured is None else measured[index],
                estimated_effort_nm=None if estimated is None else estimated[index],
                target_velocity_rad_s=None if targets is None else targets[index],
            )
            for index, name in enumerate(names)
        )

    def build_frame(
        self,
        *,
        pose: VehiclePoseSample,
        joint_names: Sequence[str],
        angular_velocities_rad_s: Sequence[float] | np.ndarray,
        applied_efforts_nm: Sequence[float] | np.ndarray | None = None,
        measured_efforts_nm: Sequence[float] | np.ndarray | None = None,
        estimated_efforts_nm: Sequence[float] | np.ndarray | None = None,
        target_velocities_rad_s: Sequence[float] | np.ndarray | None = None,
        wheel_kinematics: Sequence[WheelKinematicsSample] = (),
        wheel_contacts: Sequence[WheelTerrainContactSample] = (),
        payload_volume_m3: float = 0.0,
        estimated_payload_mass_kg: float = 0.0,
    ) -> VehicleTelemetryFrame:
        joints = self.build_joint_samples(
            joint_names=joint_names,
            angular_velocities_rad_s=angular_velocities_rad_s,
            applied_efforts_nm=applied_efforts_nm,
            measured_efforts_nm=measured_efforts_nm,
            estimated_efforts_nm=estimated_efforts_nm,
            target_velocities_rad_s=target_velocities_rad_s,
        )
        return VehicleTelemetryFrame(
            pose=pose,
            joints=joints,
            wheel_kinematics=tuple(wheel_kinematics),
            wheel_contacts=tuple(wheel_contacts),
            payload_volume_m3=payload_volume_m3,
            estimated_payload_mass_kg=estimated_payload_mass_kg,
        )

    @staticmethod
    def _array(value: Sequence[float] | np.ndarray, *, expected: int, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (expected,) or not np.all(np.isfinite(result)):
            raise ValueError(
                f"[IsaacTelemetryInputAdapter] {name} must be finite shape "
                f"({expected},); received={result.shape}"
            )
        return result

    @classmethod
    def _optional_array(
        cls,
        value: Sequence[float] | np.ndarray | None,
        *,
        expected: int,
        name: str,
    ) -> np.ndarray | None:
        if value is None:
            return None
        return cls._array(value, expected=expected, name=name)
