"""Device-resident Mobile-to-Resting deposition for GPU runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..bulk_state import MaterialScenario
from ..performance import WarpRuntime
from ..terrain import TerrainGrid


_KERNELS: dict[int, tuple[Any, Any]] = {}


def _kernels(wp: Any) -> tuple[Any, Any]:
    cached = _KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def deposit(
        resting: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        forcing: wp.array(dtype=wp.int32),
        deposited: wp.array(dtype=wp.float64),
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
        settle_subcell_tail: int,
    ):
        index = wp.tid()
        row = index // cols
        col = index - row * cols
        h = mobile[index]
        deposited[index] = wp.float64(0.0)
        if h <= wp.float64(0.0) or forcing[index] != 0:
            return
        if settle_subcell_tail != 0:
            resting[index] = resting[index] + h
            mobile[index] = wp.float64(0.0)
            momentum_x[index] = wp.float64(0.0)
            momentum_y[index] = wp.float64(0.0)
            deposited[index] = h
            mobile_to_resting_cumulative[index] = (
                mobile_to_resting_cumulative[index] + h * weights[index]
            )
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
            resting[right] + mobile[right] - resting[left] - mobile[left]
        ) / hx
        sy = (
            resting[up] + mobile[up] - resting[down] - mobile[down]
        ) / hy
        slope_tan = wp.sqrt(sx * sx + sy * sy)
        normalization = wp.sqrt(wp.float64(1.0) + slope_tan * slope_tan)
        drive = density * gravity * h * slope_tan / normalization
        resistance = cohesion_pa + density * gravity * h * stop_tangent / normalization
        if drive > resistance:
            return
        amount = wp.min(h, settling_rate * dt)
        resting[index] = resting[index] + amount
        mobile[index] = h - amount
        fraction = (h - amount) / h
        momentum_x[index] = momentum_x[index] * fraction
        momentum_y[index] = momentum_y[index] * fraction
        deposited[index] = amount
        mobile_to_resting_cumulative[index] = (
            mobile_to_resting_cumulative[index] + amount * weights[index]
        )

    @wp.kernel
    def weighted_sum(
        deposited: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        total: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        wp.atomic_add(total, 0, deposited[item] * weights[item])

    result = (deposit, weighted_sum)
    _KERNELS[id(wp)] = result
    return result


@dataclass(frozen=True)
class WarpDepositionStep:
    deposited_volume_m3: float
    parameter_status: str


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
            "resting", "mobile", "momentum_x", "momentum_y", "material_mask",
            "deposition_work", "weights",
            "avalanche_m2r_cumulative",
        }
        missing = required - set(self.runtime.arrays)
        if missing:
            raise ValueError(f"[WarpDeposition] missing shared fields: {sorted(missing)}")
        self._state = state

    def step_resident(
        self,
        material: MaterialScenario,
        dt_s: float,
        *,
        settle_subcell_tail: bool = False,
    ) -> WarpDepositionStep:
        if self._state is None:
            raise RuntimeError("[WarpDeposition] bind DeviceBulkState first")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[WarpDeposition] dt_s must be finite/positive")
        state = self._state
        wp = self.runtime.wp
        weighted = wp.zeros(1, dtype=wp.float64, device=self.runtime.device)
        deposit, weighted_sum = _kernels(wp)

        self.runtime.launch(
            deposit,
            dim=state.size,
            inputs=[
                self.runtime.arrays["resting"], self.runtime.arrays["mobile"],
                self.runtime.arrays["momentum_x"], self.runtime.arrays["momentum_y"],
                self.runtime.arrays["material_mask"], self.runtime.arrays["deposition_work"],
                self.runtime.arrays["avalanche_m2r_cumulative"],
                self.runtime.arrays["weights"],
                state.shape[0], state.shape[1], state.grid.dx, state.grid.dy, dt,
                self.settling_rate_m_s, self.speed_threshold_m_s,
                material.assumed_bulk_density_kg_m3, 9.81,
                material.cohesion_proxy_pa,
                float(np.tan(np.deg2rad(material.stop_angle_deg))),
                int(settle_subcell_tail),
            ],
        )
        self.runtime.launch(
            weighted_sum,
            dim=state.size,
            inputs=[self.runtime.arrays["deposition_work"], self.runtime.arrays["weights"], weighted],
        )
        self.runtime.synchronize()
        values = np.asarray(weighted.numpy(), dtype=np.float64)
        self.runtime.telemetry.record_d2h(values)
        return WarpDepositionStep(
            deposited_volume_m3=float(values[0]),
            parameter_status="LITERATURE_BASED_REDUCED_ORDER_UNCALIBRATED",
        )

    def diagnostics(self) -> dict[str, object]:
        result = self.runtime.diagnostics(self.backend_name)
        result.update(
            {
                "resident_state": ["resting", "mobile", "momentum_x", "momentum_y"],
                "semantics": "incremental_mobile_to_resting_below_cohesive_stop_yield",
            }
        )
        return result
