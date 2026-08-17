"""Phase-E authoritative bulk-material state without transport algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np


BULK_STATE_SCHEMA_VERSION = "phase_e.bulk_state.v1"
UNCALIBRATED_LABEL = "UNCALIBRATED_SCENARIO_PARAMETER"


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"[BulkState] {name} must be finite; value={value!r}")
    return result


def _nonnegative(value: float, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"[BulkState] {name} must be non-negative; value={result}")
    return result


def _readonly_array(
    value: np.ndarray | Iterable[float],
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if shape is not None and result.shape != shape:
        raise ValueError(
            f"[BulkState] {name} must have shape={shape}; received={result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"[BulkState] {name} contains NaN or Inf")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"[BulkState] {name} contains negative values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MaterialScenario:
    """Explicitly uncalibrated material scenario; never measured truth."""

    name: str
    assumed_bulk_density_kg_m3: float
    internal_friction_angle_deg: float
    cohesion_proxy_pa: float
    tool_friction_coefficient: float
    start_angle_deg: float
    stop_angle_deg: float
    mobile_friction_coefficient: float
    calibration_status: str = UNCALIBRATED_LABEL

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("[MaterialScenario] name must be non-empty")
        if self.calibration_status != UNCALIBRATED_LABEL:
            raise ValueError(
                "[MaterialScenario] Phase E accepts only explicitly uncalibrated "
                f"parameters labelled {UNCALIBRATED_LABEL!r}"
            )
        object.__setattr__(
            self,
            "assumed_bulk_density_kg_m3",
            _nonnegative(
                self.assumed_bulk_density_kg_m3,
                name="assumed_bulk_density_kg_m3",
            ),
        )
        if self.assumed_bulk_density_kg_m3 <= 0.0:
            raise ValueError("[MaterialScenario] assumed density must be > 0")
        for name in (
            "internal_friction_angle_deg",
            "cohesion_proxy_pa",
            "tool_friction_coefficient",
            "start_angle_deg",
            "stop_angle_deg",
            "mobile_friction_coefficient",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name=name),
            )
        for name in (
            "internal_friction_angle_deg",
            "start_angle_deg",
            "stop_angle_deg",
        ):
            if not getattr(self, name) < 90.0:
                raise ValueError(f"[MaterialScenario] {name} must be < 90 degrees")
        if self.start_angle_deg <= self.stop_angle_deg:
            raise ValueError(
                "[MaterialScenario] start_angle_deg must exceed stop_angle_deg "
                "to define hysteresis"
            )

    def estimated_mass_kg(self, volume_m3: float) -> float:
        volume = _nonnegative(volume_m3, name="volume_m3")
        return float(volume * self.assumed_bulk_density_kg_m3)


@dataclass(frozen=True)
class PayloadFreeSurface:
    """Planar payload free-surface proxy in the bucket-local frame."""

    normal_bucket_frame: np.ndarray
    offset_m: float

    def __post_init__(self) -> None:
        normal = _readonly_array(
            self.normal_bucket_frame,
            name="normal_bucket_frame",
            shape=(3,),
        )
        norm = float(np.linalg.norm(normal))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                "[PayloadFreeSurface] normal_bucket_frame must have unit length"
            )
        object.__setattr__(self, "normal_bucket_frame", normal)
        object.__setattr__(
            self,
            "offset_m",
            _finite(self.offset_m, name="free_surface.offset_m"),
        )


class BucketFillPhase(str, Enum):
    """Reduced-order state of material occupying the bucket interior."""

    STATIC = "STATIC"
    RELAXING = "RELAXING"
    SPILLING = "SPILLING"


@dataclass(frozen=True)
class BucketInternalFillState:
    """Geometry-resolved payload state for an extruded bucket interior.

    The occupied polygon is the bucket-local ``x=0`` cross-section.  Its
    extrusion across ``interior_width_m`` defines the occupied volume.  The
    tensor is the payload mass moment of inertia about its current COM.
    """

    volume_m3: float
    mass_kg: float
    geometric_fill_ratio: float
    rated_fill_ratio: float
    free_surface: PayloadFreeSurface
    occupied_polygon_bucket_frame_m: np.ndarray
    center_of_mass_bucket_frame_m: np.ndarray
    second_moment_volume_m5: np.ndarray
    inertia_tensor_kg_m2: np.ndarray
    phase: BucketFillPhase
    secondary_separation_direction_bucket_frame: np.ndarray
    geometric_capacity_m3: float
    rated_capacity_m3: float

    def __post_init__(self) -> None:
        for name in (
            "volume_m3",
            "mass_kg",
            "geometric_fill_ratio",
            "rated_fill_ratio",
            "geometric_capacity_m3",
            "rated_capacity_m3",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name=name))
        if self.geometric_capacity_m3 <= 0.0 or self.rated_capacity_m3 <= 0.0:
            raise ValueError("[BucketInternalFillState] capacities must be positive")
        if self.geometric_fill_ratio > 1.0 + 1e-8:
            raise ValueError("[BucketInternalFillState] geometric fill ratio exceeds one")
        if self.rated_fill_ratio > self.geometric_capacity_m3 / self.rated_capacity_m3 + 1e-8:
            raise ValueError("[BucketInternalFillState] rated fill ratio is inconsistent")
        if not isinstance(self.free_surface, PayloadFreeSurface):
            raise TypeError("[BucketInternalFillState] free_surface has invalid type")
        polygon = np.asarray(self.occupied_polygon_bucket_frame_m, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 3 or not np.all(np.isfinite(polygon)):
            raise ValueError("[BucketInternalFillState] occupied polygon must have shape (N,3)")
        if len(polygon) in (1, 2):
            raise ValueError("[BucketInternalFillState] occupied polygon is invalid")
        object.__setattr__(
            self,
            "occupied_polygon_bucket_frame_m",
            _readonly_array(polygon, name="occupied_polygon_bucket_frame_m"),
        )
        object.__setattr__(
            self,
            "center_of_mass_bucket_frame_m",
            _readonly_array(
                self.center_of_mass_bucket_frame_m,
                name="internal_fill.center_of_mass_bucket_frame_m",
                shape=(3,),
            ),
        )
        second = _readonly_array(
            self.second_moment_volume_m5,
            name="second_moment_volume_m5",
            shape=(3, 3),
        )
        inertia = _readonly_array(
            self.inertia_tensor_kg_m2,
            name="inertia_tensor_kg_m2",
            shape=(3, 3),
        )
        if not np.allclose(second, second.T, atol=1e-10) or not np.allclose(
            inertia, inertia.T, atol=1e-8
        ):
            raise ValueError("[BucketInternalFillState] moment tensors must be symmetric")
        if np.min(np.linalg.eigvalsh(inertia)) < -1e-7:
            raise ValueError("[BucketInternalFillState] inertia tensor is not positive semidefinite")
        object.__setattr__(self, "second_moment_volume_m5", second)
        object.__setattr__(self, "inertia_tensor_kg_m2", inertia)
        phase = self.phase if isinstance(self.phase, BucketFillPhase) else BucketFillPhase(self.phase)
        object.__setattr__(self, "phase", phase)
        direction = _readonly_array(
            self.secondary_separation_direction_bucket_frame,
            name="secondary_separation_direction_bucket_frame",
            shape=(3,),
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("[BucketInternalFillState] secondary separation direction is zero")
        object.__setattr__(
            self,
            "secondary_separation_direction_bucket_frame",
            _readonly_array(
                direction / norm,
                name="secondary_separation_direction_bucket_frame",
                shape=(3,),
            ),
        )


@dataclass(frozen=True)
class PayloadState:
    """Capacity-bounded bucket payload with explicitly estimated mass."""

    volume_m3: float
    capacity_m3: float
    assumed_bulk_density_kg_m3: float
    center_of_mass_bucket_frame_m: np.ndarray
    free_surface: PayloadFreeSurface | None = None
    internal_fill: BucketInternalFillState | None = None

    def __post_init__(self) -> None:
        volume = _nonnegative(self.volume_m3, name="payload.volume_m3")
        capacity = _nonnegative(self.capacity_m3, name="payload.capacity_m3")
        density = _nonnegative(
            self.assumed_bulk_density_kg_m3,
            name="payload.assumed_bulk_density_kg_m3",
        )
        if capacity <= 0.0 or density <= 0.0:
            raise ValueError("[PayloadState] capacity and assumed density must be > 0")
        if volume > capacity:
            raise ValueError(
                f"[PayloadState] volume exceeds capacity: {volume} > {capacity} m3"
            )
        if self.free_surface is not None and not isinstance(
            self.free_surface,
            PayloadFreeSurface,
        ):
            raise TypeError(
                "[PayloadState] free_surface must be PayloadFreeSurface or None"
            )
        if self.internal_fill is not None:
            if not isinstance(self.internal_fill, BucketInternalFillState):
                raise TypeError("[PayloadState] internal_fill has invalid type")
            internal = self.internal_fill
            if not np.isclose(internal.volume_m3, volume, atol=1e-9, rtol=0.0):
                raise ValueError("[PayloadState] internal fill volume disagrees with payload")
            if not np.isclose(internal.mass_kg, volume * density, atol=1e-6, rtol=1e-10):
                raise ValueError("[PayloadState] internal fill mass disagrees with payload")
            if self.free_surface is None:
                object.__setattr__(self, "free_surface", internal.free_surface)
            elif not (
                np.allclose(
                    self.free_surface.normal_bucket_frame,
                    internal.free_surface.normal_bucket_frame,
                    atol=1e-9,
                )
                and np.isclose(
                    self.free_surface.offset_m,
                    internal.free_surface.offset_m,
                    atol=1e-9,
                )
            ):
                raise ValueError("[PayloadState] free surface disagrees with internal fill")
            object.__setattr__(
                self,
                "center_of_mass_bucket_frame_m",
                internal.center_of_mass_bucket_frame_m,
            )
        object.__setattr__(self, "volume_m3", volume)
        object.__setattr__(self, "capacity_m3", capacity)
        object.__setattr__(self, "assumed_bulk_density_kg_m3", density)
        object.__setattr__(
            self,
            "center_of_mass_bucket_frame_m",
            _readonly_array(
                self.center_of_mass_bucket_frame_m,
                name="payload.center_of_mass_bucket_frame_m",
                shape=(3,),
            ),
        )

    @property
    def fill_ratio(self) -> float:
        return float(self.volume_m3 / self.capacity_m3)

    @property
    def estimated_payload_mass_kg(self) -> float:
        return float(self.volume_m3 * self.assumed_bulk_density_kg_m3)

    @property
    def remaining_capacity_m3(self) -> float:
        return float(self.capacity_m3 - self.volume_m3)

    def with_volume(self, volume_m3: float) -> "PayloadState":
        return PayloadState(
            volume_m3=volume_m3,
            capacity_m3=self.capacity_m3,
            assumed_bulk_density_kg_m3=self.assumed_bulk_density_kg_m3,
            center_of_mass_bucket_frame_m=np.array(
                self.center_of_mass_bucket_frame_m,
                copy=True,
            ),
            free_surface=self.free_surface,
            internal_fill=None,
        )


@dataclass(frozen=True)
class MaterialParcel:
    """Coarse airborne volume carrier, not an individual DEM particle."""

    parcel_id: str
    volume_m3: float
    assumed_bulk_density_kg_m3: float
    position_world_m: np.ndarray
    velocity_world_m_s: np.ndarray
    timestamp_s: float
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.parcel_id, str) or not self.parcel_id.strip():
            raise ValueError("[MaterialParcel] parcel_id must be non-empty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("[MaterialParcel] source must be non-empty")
        volume = _nonnegative(self.volume_m3, name="parcel.volume_m3")
        density = _nonnegative(
            self.assumed_bulk_density_kg_m3,
            name="parcel.assumed_bulk_density_kg_m3",
        )
        if volume <= 0.0 or density <= 0.0:
            raise ValueError("[MaterialParcel] volume and assumed density must be > 0")
        object.__setattr__(self, "volume_m3", volume)
        object.__setattr__(self, "assumed_bulk_density_kg_m3", density)
        object.__setattr__(
            self,
            "position_world_m",
            _readonly_array(
                self.position_world_m,
                name="parcel.position_world_m",
                shape=(3,),
            ),
        )
        object.__setattr__(
            self,
            "velocity_world_m_s",
            _readonly_array(
                self.velocity_world_m_s,
                name="parcel.velocity_world_m_s",
                shape=(3,),
            ),
        )
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative(self.timestamp_s, name="parcel.timestamp_s"),
        )

    @property
    def estimated_mass_kg(self) -> float:
        return float(self.volume_m3 * self.assumed_bulk_density_kg_m3)


@dataclass(frozen=True)
class TerrainState:
    """One authoritative Phase-E resting/mobile/payload/airborne state snapshot."""

    H_resting_m: np.ndarray
    mobile_height_m: np.ndarray
    mobile_momentum_m2_s: np.ndarray
    payload: PayloadState
    airborne_parcels: tuple[MaterialParcel, ...]
    material: MaterialScenario
    outflow_volume_m3: float
    timestamp_s: float
    action_index: int
    schema_version: str = BULK_STATE_SCHEMA_VERSION
    _trusted_arrays: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != BULK_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"[TerrainState] schema_version must be {BULK_STATE_SCHEMA_VERSION!r}"
            )
        if self._trusted_arrays:
            resting = np.asarray(self.H_resting_m, dtype=np.float64)
            if not resting.flags.c_contiguous:
                resting = np.ascontiguousarray(resting)
            resting.setflags(write=False)
        else:
            resting = _readonly_array(
                self.H_resting_m,
                name="H_resting_m",
                nonnegative=True,
            )
        if resting.ndim != 2 or min(resting.shape) < 2:
            raise ValueError("[TerrainState] H_resting_m must be a 2-D vertex field")
        if self._trusted_arrays:
            mobile = np.asarray(self.mobile_height_m, dtype=np.float64)
            momentum = np.asarray(self.mobile_momentum_m2_s, dtype=np.float64)
            if mobile.shape != resting.shape or momentum.shape != resting.shape + (2,):
                raise ValueError("[TerrainState] trusted state array shape mismatch")
            if not mobile.flags.c_contiguous:
                mobile = np.ascontiguousarray(mobile)
            if not momentum.flags.c_contiguous:
                momentum = np.ascontiguousarray(momentum)
            mobile.setflags(write=False)
            momentum.setflags(write=False)
        else:
            mobile = _readonly_array(
                self.mobile_height_m,
                name="mobile_height_m",
                shape=resting.shape,
                nonnegative=True,
            )
            momentum = _readonly_array(
                self.mobile_momentum_m2_s,
                name="mobile_momentum_m2_s",
                shape=resting.shape + (2,),
            )
            dry = mobile <= 1e-12
            if np.any(np.linalg.norm(momentum[dry], axis=-1) > 1e-12):
                raise ValueError(
                    "[TerrainState] mobile momentum must be zero at dry vertices"
                )
        if not isinstance(self.payload, PayloadState):
            raise TypeError("[TerrainState] payload must be PayloadState")
        if not isinstance(self.material, MaterialScenario):
            raise TypeError("[TerrainState] material must be MaterialScenario")
        parcels = tuple(self.airborne_parcels)
        if any(not isinstance(parcel, MaterialParcel) for parcel in parcels):
            raise TypeError(
                "[TerrainState] airborne_parcels must contain MaterialParcel"
            )
        parcel_ids = [parcel.parcel_id for parcel in parcels]
        if len(parcel_ids) != len(set(parcel_ids)):
            raise ValueError("[TerrainState] airborne parcel IDs must be unique")
        density = self.material.assumed_bulk_density_kg_m3
        if not np.isclose(
            self.payload.assumed_bulk_density_kg_m3,
            density,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "[TerrainState] payload assumed density must match material scenario"
            )
        for parcel in parcels:
            if not np.isclose(
                parcel.assumed_bulk_density_kg_m3,
                density,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "[TerrainState] parcel assumed density must match material scenario"
                )
        if not isinstance(self.action_index, (int, np.integer)) or self.action_index < 0:
            raise ValueError("[TerrainState] action_index must be a non-negative integer")
        object.__setattr__(self, "H_resting_m", resting)
        object.__setattr__(self, "mobile_height_m", mobile)
        object.__setattr__(self, "mobile_momentum_m2_s", momentum)
        object.__setattr__(self, "airborne_parcels", parcels)
        object.__setattr__(
            self,
            "outflow_volume_m3",
            _nonnegative(self.outflow_volume_m3, name="outflow_volume_m3"),
        )
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative(self.timestamp_s, name="timestamp_s"),
        )
        object.__setattr__(self, "action_index", int(self.action_index))

    @property
    def airborne_volume_m3(self) -> float:
        return float(sum(parcel.volume_m3 for parcel in self.airborne_parcels))

    def snapshot(self) -> "TerrainState":
        """Deep copy all arrays and parcel vectors for reset/logging isolation."""

        payload_surface = self.payload.free_surface
        if payload_surface is not None:
            payload_surface = PayloadFreeSurface(
                normal_bucket_frame=np.array(
                    payload_surface.normal_bucket_frame,
                    copy=True,
                ),
                offset_m=payload_surface.offset_m,
            )
        internal = self.payload.internal_fill
        if internal is not None:
            internal = BucketInternalFillState(
                volume_m3=internal.volume_m3,
                mass_kg=internal.mass_kg,
                geometric_fill_ratio=internal.geometric_fill_ratio,
                rated_fill_ratio=internal.rated_fill_ratio,
                free_surface=PayloadFreeSurface(
                    normal_bucket_frame=np.array(
                        internal.free_surface.normal_bucket_frame, copy=True
                    ),
                    offset_m=internal.free_surface.offset_m,
                ),
                occupied_polygon_bucket_frame_m=np.array(
                    internal.occupied_polygon_bucket_frame_m, copy=True
                ),
                center_of_mass_bucket_frame_m=np.array(
                    internal.center_of_mass_bucket_frame_m, copy=True
                ),
                second_moment_volume_m5=np.array(
                    internal.second_moment_volume_m5, copy=True
                ),
                inertia_tensor_kg_m2=np.array(
                    internal.inertia_tensor_kg_m2, copy=True
                ),
                phase=internal.phase,
                secondary_separation_direction_bucket_frame=np.array(
                    internal.secondary_separation_direction_bucket_frame, copy=True
                ),
                geometric_capacity_m3=internal.geometric_capacity_m3,
                rated_capacity_m3=internal.rated_capacity_m3,
            )
        payload = PayloadState(
            volume_m3=self.payload.volume_m3,
            capacity_m3=self.payload.capacity_m3,
            assumed_bulk_density_kg_m3=self.payload.assumed_bulk_density_kg_m3,
            center_of_mass_bucket_frame_m=np.array(
                self.payload.center_of_mass_bucket_frame_m,
                copy=True,
            ),
            free_surface=payload_surface,
            internal_fill=internal,
        )
        parcels = tuple(
            MaterialParcel(
                parcel_id=parcel.parcel_id,
                volume_m3=parcel.volume_m3,
                assumed_bulk_density_kg_m3=parcel.assumed_bulk_density_kg_m3,
                position_world_m=np.array(parcel.position_world_m, copy=True),
                velocity_world_m_s=np.array(parcel.velocity_world_m_s, copy=True),
                timestamp_s=parcel.timestamp_s,
                source=parcel.source,
            )
            for parcel in self.airborne_parcels
        )
        return TerrainState(
            H_resting_m=np.array(self.H_resting_m, copy=True),
            mobile_height_m=np.array(self.mobile_height_m, copy=True),
            mobile_momentum_m2_s=np.array(self.mobile_momentum_m2_s, copy=True),
            payload=payload,
            airborne_parcels=parcels,
            material=self.material,
            outflow_volume_m3=self.outflow_volume_m3,
            timestamp_s=self.timestamp_s,
            action_index=self.action_index,
            schema_version=self.schema_version,
        )
