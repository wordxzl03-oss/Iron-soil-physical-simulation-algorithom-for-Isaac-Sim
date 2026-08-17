"""Narrow public-API adapters for Phase-H force and payload feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..bulk_state import PayloadState
from ..tools import ToolDescriptor
from .model import SoilForceResult


@dataclass
class IsaacSoilForceAdapter:
    """Apply a computed force through a RigidPrim-like public interface."""

    bucket_rigid_prim: Any

    def apply(self, result: SoilForceResult, terrain_to_world_matrix: np.ndarray) -> None:
        transform = np.asarray(terrain_to_world_matrix, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("[IsaacSoilForce] terrain_to_world_matrix invalid")
        rotation = transform[:3, :3]
        force_world = rotation @ result.force_terrain_n
        residual_torque_world = rotation @ result.residual_couple_terrain_nm
        point_h = transform @ np.r_[result.application_point_terrain_m, 1.0]
        point_world = point_h[:3] / point_h[3]
        method = getattr(
            self.bucket_rigid_prim,
            "apply_forces_and_torques_at_pos",
            None,
        )
        if not callable(method):
            raise RuntimeError(
                "[IsaacSoilForce] public "
                "RigidPrim.apply_forces_and_torques_at_pos is unavailable"
            )
        method(
            forces=np.asarray(force_world, dtype=np.float32).reshape(1, 3),
            torques=np.asarray(residual_torque_world, dtype=np.float32).reshape(1, 3),
            positions=np.asarray(point_world, dtype=np.float32).reshape(1, 3),
            is_global=True,
        )


class IsaacPayloadMassAdapter:
    """Action-event update of bucket mass/COM using public USD MassAPI.

    The adapter records an assumed-density payload estimate. It never labels the
    result true mass and never changes mass at physics-frame frequency.
    """

    def __init__(
        self,
        stage: Any,
        bucket_prim_path: str,
        descriptor: ToolDescriptor,
        *,
        base_com_link_m: np.ndarray | None = None,
    ) -> None:
        from pxr import UsdPhysics  # type: ignore

        prim = stage.GetPrimAtPath(bucket_prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"[IsaacPayloadMass] invalid bucket prim: {bucket_prim_path}")
        api = UsdPhysics.MassAPI.Apply(prim)
        mass = api.GetMassAttr().Get()
        self._base_mass_kg = float(mass) if mass is not None else 0.0
        if not np.isfinite(self._base_mass_kg) or self._base_mass_kg <= 0.0:
            raise ValueError("[IsaacPayloadMass] bucket requires a positive authored base mass")
        center = api.GetCenterOfMassAttr().Get()
        authored_center = (
            None if center is None else np.asarray(center, dtype=np.float64)
        )
        if base_com_link_m is not None:
            resolved_center = np.asarray(base_com_link_m, dtype=np.float64)
        elif authored_center is not None and np.all(np.isfinite(authored_center)):
            resolved_center = authored_center
        else:
            raise ValueError(
                "[IsaacPayloadMass] authored base COM is absent/non-finite; "
                "pass a PhysX-runtime-derived base_com_link_m instead of guessing"
            )
        if resolved_center.shape != (3,) or not np.all(np.isfinite(resolved_center)):
            raise ValueError("[IsaacPayloadMass] base_com_link_m must be finite shape (3,)")
        self._base_com_link_m = np.array(resolved_center, copy=True)
        self._api = api
        self._descriptor = descriptor

    @property
    def base_mass_kg(self) -> float:
        return self._base_mass_kg

    def update_action_event(self, payload: PayloadState) -> dict[str, float | list[float] | str]:
        from pxr import Gf  # type: ignore

        payload_mass = payload.estimated_payload_mass_kg
        tool_point = np.r_[payload.center_of_mass_bucket_frame_m, 1.0]
        link_point = self._descriptor.tool_to_link_matrix @ tool_point
        payload_com_link = link_point[:3] / link_point[3]
        total_mass = self._base_mass_kg + payload_mass
        combined = (
            self._base_com_link_m * self._base_mass_kg
            + payload_com_link * payload_mass
        ) / total_mass
        self._api.CreateMassAttr().Set(float(total_mass))
        self._api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*combined.astype(float)))
        return {
            "mass_value_kind": "assumed-density payload estimate",
            "base_mass_kg": self._base_mass_kg,
            "estimated_payload_mass_kg": payload_mass,
            "total_authored_mass_kg": total_mass,
            "combined_center_of_mass_link_m": combined.tolist(),
        }
