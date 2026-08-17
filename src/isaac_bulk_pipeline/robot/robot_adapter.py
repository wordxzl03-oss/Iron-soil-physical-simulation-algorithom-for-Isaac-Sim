"""Isaac articulation and tool-link state behind a robot-independent API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..config.loader import RobotConfig


@dataclass(frozen=True)
class JointState:
    """One robot articulation observation in SI units."""

    timestamp: float
    names: tuple[str, ...]
    positions_rad: np.ndarray
    velocities_rad_s: np.ndarray


def usd_matrix_to_numpy_m(matrix: Any, meters_per_unit: float) -> np.ndarray:
    """Convert a Gf row-vector world matrix into NumPy column-vector metres."""

    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise ValueError(
            f"[RobotAdapter] meters_per_unit must be positive; value={meters_per_unit}"
        )
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"[RobotAdapter] USD world matrix must be finite (4,4); shape={values.shape}"
        )
    result = np.ascontiguousarray(values.T)
    result[:3, 3] *= meters_per_unit
    return result


class RobotAdapter:
    """Read joints and a configured tool-link pose from any Isaac articulation.

    The adapter deliberately contains no tool geometry, sweep or excavation
    logic. An already registered articulation can be injected by the runtime;
    otherwise the adapter creates and initializes a SingleArticulation wrapper.
    """

    def __init__(
        self,
        *,
        articulation: Any | None = None,
        time_source: Callable[[], float] | None = None,
        pose_reader: Callable[[Any, str], np.ndarray] | None = None,
    ) -> None:
        self._articulation = articulation
        self._time_source = time_source
        self._pose_reader = pose_reader
        self._stage: Any | None = None
        self._config: RobotConfig | None = None
        self._initialized = False

    @property
    def config(self) -> RobotConfig:
        """Return the validated active robot configuration."""

        if self._config is None:
            raise RuntimeError("[RobotAdapter] adapter is not initialized")
        return self._config

    @property
    def articulation(self) -> Any:
        """Return the wrapped Isaac articulation."""

        if self._articulation is None:
            raise RuntimeError("[RobotAdapter] adapter is not initialized")
        return self._articulation

    def initialize(
        self,
        stage: Any,
        config: RobotConfig,
        *,
        articulation: Any | None = None,
    ) -> None:
        """Validate Prim paths and initialize the articulation wrapper."""

        if self._initialized:
            raise RuntimeError("[RobotAdapter] initialize may only be called once")
        for label, prim_path in (
            ("robot_root_prim", config.robot_root_prim),
            ("articulation_root_prim", config.articulation_root_prim),
            ("tool_link_prim", config.tool_link_prim),
        ):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise ValueError(
                    f"[RobotAdapter] missing {label}; prim_path={prim_path}, "
                    f"robot_root={config.robot_root_prim}"
                )
        if articulation is not None:
            self._articulation = articulation
        if self._articulation is None:
            try:
                from isaacsim.core.prims import SingleArticulation
            except ImportError:
                try:
                    from omni.isaac.core.articulations import Articulation as SingleArticulation
                except ImportError as exc:
                    raise RuntimeError(
                        "[RobotAdapter] Isaac articulation bindings are unavailable"
                    ) from exc
            self._articulation = SingleArticulation(
                prim_path=config.articulation_root_prim,
                name="bulk_pipeline_robot",
            )
            self._articulation.initialize()
        names = self._joint_names(self._articulation)
        if not names:
            raise ValueError(
                "[RobotAdapter] articulation exposes no DOFs; "
                f"prim_path={config.articulation_root_prim}"
            )
        self._stage = stage
        self._config = config
        self._initialized = True

    def get_joint_state(self) -> JointState:
        """Return joint names, positions and velocities with independent arrays."""

        articulation = self.articulation
        names = self._joint_names(articulation)
        positions = np.asarray(articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
        velocities_value = articulation.get_joint_velocities()
        velocities = (
            np.zeros_like(positions)
            if velocities_value is None
            else np.asarray(velocities_value, dtype=np.float64).reshape(-1)
        )
        if positions.shape != (len(names),) or velocities.shape != (len(names),):
            raise ValueError(
                "[RobotAdapter] articulation state shape mismatch; "
                f"names={len(names)}, positions={positions.shape}, "
                f"velocities={velocities.shape}, "
                f"prim_path={self.config.articulation_root_prim}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError(
                "[RobotAdapter] articulation state contains NaN or Inf; "
                f"prim_path={self.config.articulation_root_prim}"
            )
        return JointState(
            timestamp=self._timestamp(),
            names=tuple(names),
            positions_rad=np.array(positions, copy=True),
            velocities_rad_s=np.array(velocities, copy=True),
        )

    def get_tool_link_pose_world(self) -> np.ndarray:
        """Return ``T_world_from_tool_link`` with translation in metres."""

        if self._stage is None:
            raise RuntimeError("[RobotAdapter] adapter is not initialized")
        if self._pose_reader is not None:
            matrix = np.asarray(
                self._pose_reader(self._stage, self.config.tool_link_prim),
                dtype=np.float64,
            )
        else:
            try:
                from pxr import UsdGeom
            except ImportError as exc:
                raise RuntimeError(
                    "[RobotAdapter] pxr bindings are unavailable for pose reading"
                ) from exc
            prim = self._stage.GetPrimAtPath(self.config.tool_link_prim)
            gf_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            matrix = usd_matrix_to_numpy_m(
                gf_matrix,
                float(UsdGeom.GetStageMetersPerUnit(self._stage)),
            )
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                "[RobotAdapter] tool-link pose reader returned invalid matrix; "
                f"shape={matrix.shape}, prim_path={self.config.tool_link_prim}"
            )
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10):
            raise ValueError(
                "[RobotAdapter] tool-link pose must be affine column-vector form; "
                f"prim_path={self.config.tool_link_prim}"
            )
        return np.array(matrix, dtype=np.float64, copy=True, order="C")

    def reset(self) -> None:
        """Reset the articulation wrapper when it exposes ``post_reset``."""

        if self._articulation is not None and hasattr(self._articulation, "post_reset"):
            self._articulation.post_reset()

    def _timestamp(self) -> float:
        if self._time_source is not None:
            value = float(self._time_source())
        else:
            try:
                import omni.timeline

                value = float(
                    omni.timeline.get_timeline_interface().get_current_time()
                )
            except ImportError:
                value = 0.0
        if not np.isfinite(value):
            raise ValueError(f"[RobotAdapter] timestamp is not finite; value={value}")
        return value

    @staticmethod
    def _joint_names(articulation: Any) -> Sequence[str]:
        names = getattr(articulation, "dof_names", None)
        if callable(names):
            names = names()
        return [] if names is None else [str(name) for name in names]
