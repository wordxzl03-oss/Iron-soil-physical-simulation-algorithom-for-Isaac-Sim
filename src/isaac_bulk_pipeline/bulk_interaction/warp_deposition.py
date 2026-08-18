"""Device-resident Mobile-to-Resting deposition for GPU runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..bulk_state import MaterialScenario
from ..performance import WarpRuntime
from ..terrain import TerrainGrid


_KERNELS: dict[int, tuple[Any, Any, Any]] = {}


def _kernels(wp: Any) -> tuple[Any, Any, Any]:
    cached = _KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def compute_deposition(
        b_eff: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        forcing: wp.array(dtype=wp.int32),
        exclusion: wp.array(dtype=wp.int32),
        deposited: wp.array(dtype=wp.float64),
        removed_momentum_x: wp.array(dtype=wp.float64),
        removed_momentum_y: wp.array(dtype=wp.float64),
        mobile_to_resting_cumulative: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        rows: int,
        cols: int,
        dx: wp.float64,
        dy: wp.float64,
        dt: wp.float64,
        settling_rate: wp.float64,
        speed_threshold: wp.float64,
        density: wp.float64,
        gravity: wp.float64,
        cohesion_pa: wp.float64,
        stop_tangent: wp.float64,
    ):
        index = wp.tid()
        row = index // cols
        col = index - row * cols
        h = mobile[index]
        deposited[index] = wp.float64(0.0)
        removed_momentum_x[index] = wp.float64(0.0)
        removed_momentum_y[index] = wp.float64(0.0)
        if (
            h <= wp.float64(0.0)
            or forcing[index] != 0
            or exclusion[index] != 0
        ):
            return
        vx = momentum_x[index] / h
        vy = momentum_y[index] / h
        if wp.sqrt(vx * vx + vy * vy) > speed_threshold:
            return
        left = index
        right = index
        down = index
        up = index
        if col > 0:
            left = index - 1
        if col + 1 < cols:
            right = index + 1
        if row > 0:
            down = index - cols
        if row + 1 < rows:
            up = index + cols
        hx = wp.float64(2.0) * dx
        hy = wp.float64(2.0) * dy
        if col == 0 or col + 1 == cols:
            hx = dx
        if row == 0 or row + 1 == rows:
            hy = dy
        sx = (
            b_eff[right] + mobile[right] - b_eff[left] - mobile[left]
        ) / hx
        sy = (
            b_eff[up] + mobile[up] - b_eff[down] - mobile[down]
        ) / hy
        slope_tan = wp.sqrt(sx * sx + sy * sy)
        normalization = wp.sqrt(wp.float64(1.0) + slope_tan * slope_tan)
        drive = density * gravity * h * slope_tan / normalization
        resistance = cohesion_pa + density * gravity * h * stop_tangent / normalization
        if drive > resistance:
            return
        amount = wp.min(h, settling_rate * dt)
        fraction_removed = amount / h
        removed_momentum_x[index] = momentum_x[index] * fraction_removed
        removed_momentum_y[index] = momentum_y[index] * fraction_removed
        deposited[index] = amount

    @wp.kernel
    def apply_deposition(
        b_eff: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        deposited: wp.array(dtype=wp.float64),
        removed_momentum_x: wp.array(dtype=wp.float64),
        removed_momentum_y: wp.array(dtype=wp.float64),
        mobile_to_resting_cumulative: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        amount = deposited[index]
        if amount <= wp.float64(0.0):
            return
        # ``amount`` was computed from the immutable pre-deposition snapshot in
        # the preceding kernel.  No operator runs between these kernels, so the
        # same-cell state is still authoritative here.
        b_eff[index] = b_eff[index] + amount
        mobile[index] = mobile[index] - amount
        momentum_x[index] = momentum_x[index] - removed_momentum_x[index]
        momentum_y[index] = momentum_y[index] - removed_momentum_y[index]
        mobile_to_resting_cumulative[index] = (
            mobile_to_resting_cumulative[index] + amount * weights[index]
        )

    @wp.kernel
    def weighted_sum(
        deposited: wp.array(dtype=wp.float64),
        removed_momentum_x: wp.array(dtype=wp.float64),
        removed_momentum_y: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        density: wp.float64,
        total: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        wp.atomic_add(total, 0, deposited[item] * weights[item])
        wp.atomic_add(total, 1, density * removed_momentum_x[item] * weights[item])
        wp.atomic_add(total, 2, density * removed_momentum_y[item] * weights[item])

    result = (compute_deposition, apply_deposition, weighted_sum)
    _KERNELS[id(wp)] = result
    return result


@dataclass(frozen=True)
class WarpDepositionStep:
    deposited_volume_m3: float
    parameter_status: str
    removed_mobile_momentum_terrain_kg_m_s: np.ndarray


class WarpDepositionOperator:
    """Shared-state equivalent of the accepted incremental deposition operator."""

    backend_name = "GPU_WARP_INCREMENTAL_DEPOSITION"

    def __init__(
        self,
        *,
        settling_rate_m_s: float = 0.30,
        speed_threshold_m_s: float = 0.18,
        device: str = "cuda:0",
        runtime: WarpRuntime | None = None,
    ) -> None:
        if settling_rate_m_s <= 0.0 or speed_threshold_m_s < 0.0:
            raise ValueError("[WarpDeposition] invalid configuration")
        self.settling_rate_m_s = float(settling_rate_m_s)
        self.speed_threshold_m_s = float(speed_threshold_m_s)
        self.runtime = runtime or WarpRuntime(device)
        self._state: Any | None = None

    def bind_device_state(self, state: Any) -> None:
        if getattr(state, "runtime", None) is not self.runtime:
            raise ValueError("[WarpDeposition] state/runtime ownership mismatch")
        required = {
            "b_eff", "mobile", "momentum_x", "momentum_y", "material_mask",
            "deposition_exclusion_mask", "deposition_work", "weights",
            "avalanche_m2r_cumulative",
        }
        missing = required - set(self.runtime.arrays)
        if missing:
            raise ValueError(f"[WarpDeposition] missing shared fields: {sorted(missing)}")
        wp = self.runtime.wp
        if "deposition_removed_momentum_x" not in self.runtime.arrays:
            self.runtime.zeros("deposition_removed_momentum_x", state.size, dtype=wp.float64)
            self.runtime.zeros("deposition_removed_momentum_y", state.size, dtype=wp.float64)
        self._state = state

    def step_resident(
        self,
        material: MaterialScenario,
        dt_s: float,
    ) -> WarpDepositionStep:
        if self._state is None:
            raise RuntimeError("[WarpDeposition] bind DeviceBulkState first")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[WarpDeposition] dt_s must be finite/positive")
        state = self._state
        wp = self.runtime.wp
        weighted = wp.zeros(3, dtype=wp.float64, device=self.runtime.device)
        compute_deposition, apply_deposition, weighted_sum = _kernels(wp)

        # Phase 1 is read-only with respect to b_eff/mobile/momentum so every
        # thread evaluates slope/stop eligibility against the same state.
        self.runtime.launch(
            compute_deposition,
            dim=state.size,
            inputs=[
                self.runtime.arrays["b_eff"], self.runtime.arrays["mobile"],
                self.runtime.arrays["momentum_x"], self.runtime.arrays["momentum_y"],
                self.runtime.arrays["material_mask"],
                self.runtime.arrays["deposition_exclusion_mask"],
                self.runtime.arrays["deposition_work"],
                self.runtime.arrays["deposition_removed_momentum_x"],
                self.runtime.arrays["deposition_removed_momentum_y"],
                self.runtime.arrays["avalanche_m2r_cumulative"],
                self.runtime.arrays["weights"],
                state.shape[0], state.shape[1], state.grid.dx, state.grid.dy, dt,
                self.settling_rate_m_s, self.speed_threshold_m_s,
                material.assumed_bulk_density_kg_m3, 9.81,
                material.cohesion_proxy_pa,
                float(np.tan(np.deg2rad(material.stop_angle_deg))),
            ],
        )
        # Phase 2 commits the precomputed same-cell transfer.  Separating the
        # phases removes the former neighbor read/write race in the GPU kernel.
        self.runtime.launch(
            apply_deposition,
            dim=state.size,
            inputs=[
                self.runtime.arrays["b_eff"], self.runtime.arrays["mobile"],
                self.runtime.arrays["momentum_x"], self.runtime.arrays["momentum_y"],
                self.runtime.arrays["deposition_work"],
                self.runtime.arrays["deposition_removed_momentum_x"],
                self.runtime.arrays["deposition_removed_momentum_y"],
                self.runtime.arrays["avalanche_m2r_cumulative"],
                self.runtime.arrays["weights"],
            ],
        )
        self.runtime.launch(
            weighted_sum,
            dim=state.size,
            inputs=[
                self.runtime.arrays["deposition_work"],
                self.runtime.arrays["deposition_removed_momentum_x"],
                self.runtime.arrays["deposition_removed_momentum_y"],
                self.runtime.arrays["weights"],
                material.assumed_bulk_density_kg_m3,
                weighted,
            ],
        )
        self.runtime.synchronize()
        values = np.asarray(weighted.numpy(), dtype=np.float64)
        self.runtime.telemetry.record_d2h(values)
        return WarpDepositionStep(
            deposited_volume_m3=float(values[0]),
            parameter_status="LITERATURE_BASED_REDUCED_ORDER_UNCALIBRATED",
            removed_mobile_momentum_terrain_kg_m_s=np.asarray([values[1], values[2], 0.0]),
        )

    def diagnostics(self) -> dict[str, object]:
        result = self.runtime.diagnostics(self.backend_name)
        result.update(
            {
                "resident_state": ["b_eff", "mobile", "momentum_x", "momentum_y"],
                "semantics": (
                    "incremental_mobile_to_resting_below_cohesive_stop_yield"
                    "+tool_occupancy_exclusion"
                ),
                "momentum_removal": "EXPLICITLY_REPORTED",
            }
        )
        return result
