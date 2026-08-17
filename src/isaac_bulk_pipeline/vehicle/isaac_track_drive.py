"""Isaac binding for the reduced-order differential track force actuator."""

from __future__ import annotations

import numpy as np

from .differential_track_drive import DifferentialTrackDriveModel, DifferentialTrackDriveOutput


def _rotation_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return a rigid rotation without requiring SciPy in CPU unit tests."""

    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("[TrackDrive] lower-body quaternion must be finite shape (4,)")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("[TrackDrive] lower-body quaternion has zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class IsaacDifferentialTrackDriveAdapter:
    """Apply the track wrench to the mobile articulation root in PhysX.

    The left/right application points remain the real CAD track centres.  The
    two forces are submitted to ``lower_body`` because it is the articulation
    root: applying them to fixed child links is not guaranteed to propagate a
    base wrench in PhysX reduced-coordinate articulations.  Force and moment
    about the root are unchanged; this is not a root-pose write or a kinematic
    drive.
    """

    def __init__(
        self,
        left_track_body,
        right_track_body,
        lower_body,
        model: DifferentialTrackDriveModel | None = None,
        *,
        left_application_offset_lower_local_m: np.ndarray | None = None,
        right_application_offset_lower_local_m: np.ndarray | None = None,
    ) -> None:
        self.left_track_body = left_track_body
        self.right_track_body = right_track_body
        self.lower_body = lower_body
        self.model = model or DifferentialTrackDriveModel()
        self._application_offsets_local = self._validated_offsets(
            left_application_offset_lower_local_m,
            right_application_offset_lower_local_m,
        )
        self._brake_reference_position_world: np.ndarray | None = None
        self._brake_reference_forward_world: np.ndarray | None = None

    def apply(
        self,
        left_command: float,
        right_command: float,
        dt_s: float,
        *,
        left_contact_active: bool,
        right_contact_active: bool,
        braking: bool = False,
    ) -> DifferentialTrackDriveOutput:
        position_world, quaternion_wxyz = self.lower_body.get_world_poses()
        position_world = np.asarray(position_world, dtype=np.float64).reshape(-1, 3)[0]
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1, 4)[0]
        rotation = _rotation_from_quaternion_wxyz(quaternion)
        forward = rotation @ np.asarray(
            self.model.config.forward_axis_lower_body_local, dtype=np.float64
        )
        forward[2] = 0.0
        forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
        if braking and self._brake_reference_position_world is None:
            self._brake_reference_position_world = position_world.copy()
            self._brake_reference_forward_world = forward.copy()
        elif not braking:
            self._brake_reference_position_world = None
            self._brake_reference_forward_world = None
        position_error = np.zeros(3, dtype=np.float64)
        yaw_error = 0.0
        if braking and self._brake_reference_position_world is not None:
            position_error = position_world - self._brake_reference_position_world
            reference_forward = self._brake_reference_forward_world
            yaw_error = float(
                np.arctan2(
                    reference_forward[0] * forward[1]
                    - reference_forward[1] * forward[0],
                    np.dot(reference_forward[:2], forward[:2]),
                )
            )
        if self._application_offsets_local is None:
            left_position = np.asarray(
                self.left_track_body.get_world_poses()[0], dtype=np.float64
            ).reshape(-1, 3)[0]
            right_position = np.asarray(
                self.right_track_body.get_world_poses()[0], dtype=np.float64
            ).reshape(-1, 3)[0]
        else:
            left_position = (
                position_world + rotation @ self._application_offsets_local[0]
            )
            right_position = (
                position_world + rotation @ self._application_offsets_local[1]
            )
        positive_pair_moment_z = float(
            np.cross(left_position - position_world, forward)[2]
            + np.cross(right_position - position_world, -forward)[2]
        )
        if abs(positive_pair_moment_z) <= 1.0e-9:
            if self._application_offsets_local is not None:
                raise RuntimeError("[TrackDrive] CAD track application offsets cannot create yaw")
            positive_pair_moment_z = -1.0
        result = self.model.step(
            left_command, right_command, rotation, dt_s,
            left_contact_active=left_contact_active,
            right_contact_active=right_contact_active,
            braking=braking,
            lower_body_linear_velocity_world=np.asarray(
                self.lower_body.get_linear_velocities(), dtype=np.float64
            ).reshape(-1, 3)[0],
            lower_body_angular_velocity_world=np.asarray(
                self.lower_body.get_angular_velocities(), dtype=np.float64
            ).reshape(-1, 3)[0],
            brake_position_error_world=position_error,
            brake_yaw_error_rad=yaw_error,
            positive_yaw_force_pair_torque_sign=float(
                np.sign(positive_pair_moment_z)
            ),
        )
        for force, position in (
            (result.left_force_world_n, left_position),
            (result.right_force_world_n, right_position),
        ):
            self.lower_body.apply_forces_and_torques_at_pos(
                forces=np.asarray(force, dtype=np.float32).reshape(1, 3),
                torques=np.zeros((1, 3), dtype=np.float32),
                positions=np.asarray(position, dtype=np.float32).reshape(1, 3),
                is_global=True,
            )
        return result

    def reset(self) -> None:
        self.model.reset()
        self._brake_reference_position_world = None
        self._brake_reference_forward_world = None

    @staticmethod
    def _validated_offsets(left, right):
        if left is None and right is None:
            return None
        if left is None or right is None:
            raise ValueError("[TrackDrive] both CAD application offsets are required")
        offsets = tuple(np.asarray(item, dtype=np.float64) for item in (left, right))
        if any(item.shape != (3,) or not np.all(np.isfinite(item)) for item in offsets):
            raise ValueError("[TrackDrive] application offsets must be finite shape (3,)")
        if np.linalg.norm(offsets[0] - offsets[1]) <= 1.0e-6:
            raise ValueError("[TrackDrive] left/right application offsets must differ")
        return tuple(item.copy() for item in offsets)
