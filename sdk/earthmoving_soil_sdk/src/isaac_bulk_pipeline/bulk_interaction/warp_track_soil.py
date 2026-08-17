"""GPU-resident conservative TrackSoil reduced-order operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..performance import WarpRuntime
from .track_soil import TrackSoilConfig


_KERNELS: dict[int, Any] = {}


def _kernels(wp: Any) -> Any:
    cached = _KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def dilate4(
        source: wp.array(dtype=wp.int32),
        target: wp.array(dtype=wp.int32),
        rows: int,
        cols: int,
    ):
        index = wp.tid()
        row = index // cols
        col = index - row * cols
        value = source[index]
        if row > 0:
            value = wp.max(value, source[index - cols])
        if row + 1 < rows:
            value = wp.max(value, source[index + cols])
        if col > 0:
            value = wp.max(value, source[index - 1])
        if col + 1 < cols:
            value = wp.max(value, source[index + 1])
        target[index] = value

    @wp.kernel
    def make_shoulder(
        buffered: wp.array(dtype=wp.int32),
        footprint: wp.array(dtype=wp.int32),
        shoulder: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        if buffered[index] != 0 and footprint[index] == 0:
            shoulder[index] = 1
        else:
            shoulder[index] = 0

    @wp.kernel
    def mark_indices(
        target: wp.array(dtype=wp.int32),
        indices: wp.array(dtype=wp.int32),
    ):
        item = wp.tid()
        target[indices[item]] = 1

    @wp.kernel
    def sink_rut(
        resting: wp.array(dtype=wp.float64),
        initial_resting: wp.array(dtype=wp.float64),
        footprint: wp.array(dtype=wp.int32),
        weights: wp.array(dtype=wp.float64),
        track_rut: wp.array(dtype=wp.float64),
        requested: wp.float64,
        maximum_total_rut: wp.float64,
        removed_volume: wp.array(dtype=wp.float64),
        sinkage_sum: wp.array(dtype=wp.float64),
        sinkage_count: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        if footprint[index] != 0:
            existing = wp.max(
                initial_resting[index] - resting[index], wp.float64(0.0)
            )
            available = wp.max(maximum_total_rut - existing, wp.float64(0.0))
            sinkage = wp.min(requested, wp.min(resting[index], available))
            resting[index] = resting[index] - sinkage
            track_rut[index] = existing + sinkage
            wp.atomic_add(removed_volume, 0, sinkage * weights[index])
            wp.atomic_add(sinkage_sum, 0, sinkage)
            wp.atomic_add(sinkage_count, 0, 1)

    @wp.kernel
    def sum_shoulder_weight(
        shoulder: wp.array(dtype=wp.int32),
        weights: wp.array(dtype=wp.float64),
        total: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        if shoulder[index] != 0:
            wp.atomic_add(total, 0, weights[index])

    @wp.kernel
    def deposit_shoulder(
        mobile: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        shoulder: wp.array(dtype=wp.int32),
        added_height: wp.float64,
        track_velocity_x: wp.float64,
        track_velocity_y: wp.float64,
    ):
        index = wp.tid()
        if shoulder[index] != 0:
            mobile[index] = mobile[index] + added_height
            momentum_x[index] = momentum_x[index] + added_height * track_velocity_x
            momentum_y[index] = momentum_y[index] + added_height * track_velocity_y

    result = (
        dilate4,
        make_shoulder,
        mark_indices,
        sink_rut,
        sum_shoulder_weight,
        deposit_shoulder,
    )
    _KERNELS[id(wp)] = result
    return result


@dataclass(frozen=True)
class WarpTrackSoilStep:
    resting_to_mobile_volume_m3: float
    left_mean_sinkage_m: float
    right_mean_sinkage_m: float
    parameter_status: str


class WarpTrackSoilOperator:
    """Resident TrackSoil arrays; only two reduction scalars sync per track."""

    backend_name = "GPU_WARP_TRACK_SOIL"

    def __init__(
        self,
        shape: tuple[int, int],
        config: TrackSoilConfig | None = None,
        *,
        device: str = "cuda:0",
        runtime: WarpRuntime | None = None,
    ) -> None:
        self.shape = int(shape[0]), int(shape[1])
        self.size = self.shape[0] * self.shape[1]
        self.config = config or TrackSoilConfig()
        self.runtime = runtime or WarpRuntime(device)
        self._owns_runtime = runtime is None
        self._initialized = False

    def _initialize_workspace(self) -> None:
        wp = self.runtime.wp
        for name in ("track_footprint", "mask_a", "mask_b", "shoulder"):
            if name not in self.runtime.arrays:
                self.runtime.zeros(name, self.size, dtype=wp.int32)
        if "track_rut" not in self.runtime.arrays:
            self.runtime.zeros("track_rut", self.size, dtype=wp.float64)

    def bind_device_state(self, state: Any) -> None:
        """Use DeviceBulkState terrain arrays directly, without a shadow copy."""

        if getattr(state, "runtime", None) is not self.runtime:
            raise ValueError("[WarpTrackSoil] state/runtime ownership mismatch")
        if tuple(getattr(state, "shape", ())) != self.shape:
            raise ValueError("[WarpTrackSoil] state shape mismatch")
        required = {
            "initial_resting",
            "resting",
            "mobile",
            "momentum_x",
            "momentum_y",
            "weights",
        }
        missing = required - set(self.runtime.arrays)
        if missing:
            raise ValueError(
                f"[WarpTrackSoil] missing shared state arrays: {sorted(missing)}"
            )
        self._initialize_workspace()
        self._initialized = True

    def initialize(
        self,
        initial_resting_m: np.ndarray,
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        vertex_weights_m2: np.ndarray,
    ) -> None:
        fields = (
            np.asarray(initial_resting_m, dtype=np.float64),
            np.asarray(H_resting_m, dtype=np.float64),
            np.asarray(mobile_height_m, dtype=np.float64),
            np.asarray(vertex_weights_m2, dtype=np.float64),
        )
        if any(field.shape != self.shape for field in fields):
            raise ValueError("[WarpTrackSoil] resident scalar field shape mismatch")
        momentum = np.asarray(mobile_momentum_m2_s, dtype=np.float64)
        if momentum.shape != self.shape + (2,):
            raise ValueError("[WarpTrackSoil] resident momentum shape mismatch")
        wp = self.runtime.wp
        for name, value in (
            ("initial_resting", fields[0]),
            ("resting", fields[1]),
            ("mobile", fields[2]),
            ("momentum_x", momentum[..., 0]),
            ("momentum_y", momentum[..., 1]),
            ("weights", fields[3]),
        ):
            self.runtime.upload(name, value.ravel(), dtype=wp.float64)
        self._initialize_workspace()
        self._initialized = True

    def _small_download(self, array: Any) -> np.ndarray:
        self.runtime.synchronize()
        host = np.asarray(array.numpy())
        self.runtime.telemetry.record_d2h(host)
        return host

    def _one_track_indices(
        self,
        footprint_indices: np.ndarray,
        track_velocity: np.ndarray,
        base_velocity: np.ndarray,
        dt_s: float,
        label: str,
    ) -> tuple[float, float]:
        indices = np.asarray(footprint_indices, dtype=np.int32).ravel()
        if indices.size == 0:
            return 0.0, 0.0
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("[WarpTrackSoil] footprint index outside grid")
        indices = np.unique(indices)
        wp = self.runtime.wp
        # The footprint is a compact host command, not a 701x701 terrain-mask
        # upload.  The persistent device mask is rebuilt in place each step.
        footprint_gpu = self.runtime.arrays["track_footprint"]
        footprint_gpu.zero_()
        indices_gpu = self.runtime.upload(
            f"{label}_footprint_indices", indices, dtype=wp.int32
        )
        dilate, shoulder_kernel, mark, sink, sum_weight, deposit = _kernels(wp)
        self.runtime.launch(
            mark,
            dim=int(indices.size),
            inputs=[footprint_gpu, indices_gpu],
        )
        wp.copy(self.runtime.arrays["mask_a"], footprint_gpu)
        source = self.runtime.arrays["mask_a"]
        target = self.runtime.arrays["mask_b"]
        for _ in range(self.config.shoulder_halo_cells):
            self.runtime.launch(
                dilate,
                dim=self.size,
                inputs=[source, target, self.shape[0], self.shape[1]],
            )
            source, target = target, source
        self.runtime.launch(
            shoulder_kernel,
            dim=self.size,
            inputs=[source, footprint_gpu, self.runtime.arrays["shoulder"]],
        )
        slip = float(np.linalg.norm(track_velocity - base_velocity))
        slip_effect = max(slip - self.config.minimum_slip_speed_m_s, 0.0)
        requested = min(
            self.config.maximum_sinkage_per_step_m,
            self.config.sinkage_rate_m_s
            * float(dt_s)
            * (1.0 + self.config.slip_gain * slip_effect),
        )
        removed = wp.zeros(1, dtype=wp.float64, device=self.runtime.device)
        sink_sum = wp.zeros(1, dtype=wp.float64, device=self.runtime.device)
        sink_count = wp.zeros(1, dtype=wp.int32, device=self.runtime.device)
        shoulder_weight = wp.zeros(1, dtype=wp.float64, device=self.runtime.device)
        self.runtime.launch(
            sink,
            dim=self.size,
            inputs=[
                self.runtime.arrays["resting"],
                self.runtime.arrays["initial_resting"],
                footprint_gpu,
                self.runtime.arrays["weights"],
                self.runtime.arrays["track_rut"],
                requested,
                self.config.maximum_total_rut_depth_m,
                removed,
                sink_sum,
                sink_count,
            ],
        )
        self.runtime.launch(
            sum_weight,
            dim=self.size,
            inputs=[
                self.runtime.arrays["shoulder"],
                self.runtime.arrays["weights"],
                shoulder_weight,
            ],
        )
        scalars = self._small_download(removed)
        sink_host = self._small_download(sink_sum)
        count_host = self._small_download(sink_count)
        weight_host = self._small_download(shoulder_weight)
        removed_volume = float(scalars[0])
        total_weight = float(weight_host[0])
        if removed_volume > 0.0 and total_weight <= 0.0:
            raise RuntimeError("[WarpTrackSoil] no shoulder area for conservative transfer")
        if removed_volume > 0.0:
            added_height = removed_volume / total_weight
            self.runtime.launch(
                deposit,
                dim=self.size,
                inputs=[
                    self.runtime.arrays["mobile"],
                    self.runtime.arrays["momentum_x"],
                    self.runtime.arrays["momentum_y"],
                    self.runtime.arrays["shoulder"],
                    added_height,
                    float(track_velocity[0]),
                    float(track_velocity[1]),
                ],
            )
        count = int(count_host[0])
        mean = float(sink_host[0]) / count if count else 0.0
        return removed_volume, mean

    def _one_track(
        self,
        footprint: np.ndarray,
        track_velocity: np.ndarray,
        base_velocity: np.ndarray,
        dt_s: float,
        label: str,
    ) -> tuple[float, float]:
        mask = np.asarray(footprint, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError("[WarpTrackSoil] footprint shape mismatch")
        return self._one_track_indices(
            np.flatnonzero(mask), track_velocity, base_velocity, dt_s, label
        )

    def apply_resident(
        self,
        *,
        left_footprint_mask: np.ndarray,
        right_footprint_mask: np.ndarray,
        left_track_velocity_xy_m_s: np.ndarray,
        right_track_velocity_xy_m_s: np.ndarray,
        base_velocity_xy_m_s: np.ndarray,
        dt_s: float,
    ) -> WarpTrackSoilStep:
        if not self._initialized:
            raise RuntimeError("[WarpTrackSoil] initialize resident state first")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("[WarpTrackSoil] dt_s must be positive")
        left_velocity = np.asarray(left_track_velocity_xy_m_s, dtype=np.float64)
        right_velocity = np.asarray(right_track_velocity_xy_m_s, dtype=np.float64)
        base_velocity = np.asarray(base_velocity_xy_m_s, dtype=np.float64)
        if any(value.shape != (2,) for value in (left_velocity, right_velocity, base_velocity)):
            raise ValueError("[WarpTrackSoil] velocities must have shape (2,)")
        left_volume, left_mean = self._one_track(
            left_footprint_mask, left_velocity, base_velocity, dt_s, "left"
        )
        right_volume, right_mean = self._one_track(
            right_footprint_mask, right_velocity, base_velocity, dt_s, "right"
        )
        return WarpTrackSoilStep(
            resting_to_mobile_volume_m3=left_volume + right_volume,
            left_mean_sinkage_m=left_mean,
            right_mean_sinkage_m=right_mean,
            parameter_status=self.config.parameter_status,
        )

    def apply_compact_resident(
        self,
        *,
        left_footprint_indices: np.ndarray,
        right_footprint_indices: np.ndarray,
        left_track_velocity_xy_m_s: np.ndarray,
        right_track_velocity_xy_m_s: np.ndarray,
        base_velocity_xy_m_s: np.ndarray,
        dt_s: float,
    ) -> WarpTrackSoilStep:
        """Track update whose only host terrain command is compact indices."""

        if not self._initialized:
            raise RuntimeError("[WarpTrackSoil] initialize resident state first")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("[WarpTrackSoil] dt_s must be positive")
        left_velocity = np.asarray(left_track_velocity_xy_m_s, dtype=np.float64)
        right_velocity = np.asarray(right_track_velocity_xy_m_s, dtype=np.float64)
        base_velocity = np.asarray(base_velocity_xy_m_s, dtype=np.float64)
        if any(value.shape != (2,) for value in (left_velocity, right_velocity, base_velocity)):
            raise ValueError("[WarpTrackSoil] velocities must have shape (2,)")
        left_volume, left_mean = self._one_track_indices(
            left_footprint_indices, left_velocity, base_velocity, dt_s, "left"
        )
        right_volume, right_mean = self._one_track_indices(
            right_footprint_indices, right_velocity, base_velocity, dt_s, "right"
        )
        return WarpTrackSoilStep(
            resting_to_mobile_volume_m3=left_volume + right_volume,
            left_mean_sinkage_m=left_mean,
            right_mean_sinkage_m=right_mean,
            parameter_status=self.config.parameter_status,
        )

    def download_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resting = self.runtime.download("resting").reshape(self.shape)
        mobile = self.runtime.download("mobile").reshape(self.shape)
        momentum = np.stack(
            (
                self.runtime.download("momentum_x").reshape(self.shape),
                self.runtime.download("momentum_y").reshape(self.shape),
            ),
            axis=-1,
        )
        return resting, mobile, momentum

    def diagnostics(self) -> dict[str, object]:
        result = self.runtime.diagnostics(self.backend_name)
        result.update(
            {
                "parameter_status": self.config.parameter_status,
                "conservative_transfer": "resting_loss_equals_mobile_gain",
                "resident_state": ["resting", "mobile", "momentum_x", "momentum_y"],
            }
        )
        return result
