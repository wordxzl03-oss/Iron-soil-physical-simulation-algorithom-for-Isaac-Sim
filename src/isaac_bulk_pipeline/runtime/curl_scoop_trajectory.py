"""Feedback-driven production dig controller for the real 390F articulation.

This module intentionally does **not** prescribe one time-only A->E animation.
The excavator stays in a material-engaged cut/scoop controller while the task
state is ``CUT_AND_FILL`` and enters breakout only after the state machine has
observed the required cut/payload conditions.

The control structure follows two externally supported operating principles:

* professional backhoe digging uses stick motion and bucket curl together for
  breakout/filling rather than lifting the boom first;
* bucket filling should adapt between penetrate-and-drag and
  penetrate-and-rotate behaviours, with the cutting edge remaining engaged
  with the soil during the filling part of the motion.

The controller remains a reduced-order engineering controller.  It does not
claim to reproduce a particular operator or hydraulic valve law.  The existing
production actuator still owns joint speed/acceleration/force envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurlScoopSample:
    stage: str
    elapsed_s: float
    stage_elapsed_s: float
    target_rad: np.ndarray
    complete: bool


class RealisticCurlScoopTrajectory:
    """State-synchronised, terrain-feedback 390F backhoe dig controller.

    Joint polarity for the audited USD is:

    * negative boom motion raises the cutting edge;
    * positive stick motion retracts the cutting edge toward the machine;
    * positive bucket motion closes/curls the bucket.

    ``CUT_AND_FILL`` therefore applies stick retraction and bucket curl
    concurrently while a slow boom feedback loop holds cutting-edge depth.
    ``CURL_AND_BREAKOUT`` first closes the bucket, then raises/withdraws it.
    Breakout is never entered merely because an internal clock elapsed.
    """

    # Real USD revolute-joint limits, articulation order:
    # swing, boom, stick, bucket.
    LOWER_RAD = np.deg2rad(np.asarray([-np.inf, -25.0, -100.0, -120.0]))
    UPPER_RAD = np.deg2rad(np.asarray([np.inf, 45.0, 45.0, 60.0]))

    # Conservative command rates beneath the actuator's own hard velocity
    # limits.  These are engineering scheduling values, not material params.
    _STICK_CUT_RATE_RAD_S = 0.075      # 4.30 deg/s
    _BUCKET_EARLY_CURL_RATE_RAD_S = 0.18   # 10.3 deg/s
    _BUCKET_SCOOP_CURL_RATE_RAD_S = 0.30   # 17.2 deg/s
    _BUCKET_BREAKOUT_RATE_RAD_S = 0.32
    _STICK_BREAKOUT_RATE_RAD_S = 0.09
    _BOOM_MAX_RATE_RAD_S = 0.12

    # Depth-hold feedback.  A 0.10 m depth error requests ~3.4 deg/s before
    # rate limiting; this is intentionally much slower than bucket curl.
    _BOOM_DEPTH_GAIN_RAD_S_PER_M = 0.60

    def __init__(
        self,
        penetrate_rad: np.ndarray,
        coordinated_cut_rad: np.ndarray,
        curl_filling_rad: np.ndarray,
        breakout_rad: np.ndarray,
        *,
        target_cut_depth_m: float = 0.20,
        minimum_engaged_fraction: float = 0.70,
    ) -> None:
        self._penetrate = self._vector(penetrate_rad, "penetrate_rad")
        self._coordinated_cut = self._vector(
            coordinated_cut_rad, "coordinated_cut_rad"
        )
        self._curl_filling = self._vector(curl_filling_rad, "curl_filling_rad")
        self._breakout = self._vector(breakout_rad, "breakout_rad")
        for name, value in (
            ("penetrate", self._penetrate),
            ("coordinated_cut", self._coordinated_cut),
            ("curl_filling", self._curl_filling),
            ("breakout", self._breakout),
        ):
            self._validate_target(value, name)
        if not np.isfinite(target_cut_depth_m) or target_cut_depth_m <= 0.0:
            raise ValueError("[390FCurlScoop] target_cut_depth_m must be positive")
        if not np.isfinite(minimum_engaged_fraction) or not (
            0.0 < minimum_engaged_fraction <= 1.0
        ):
            raise ValueError(
                "[390FCurlScoop] minimum_engaged_fraction must lie in (0,1]"
            )
        self.target_cut_depth_m = float(target_cut_depth_m)
        self.minimum_engaged_fraction = float(minimum_engaged_fraction)
        self._command = self._penetrate.copy()
        self._entry_payload_m3 = 0.0
        self._mode = "CUT"
        self.elapsed_s = 0.0
        self.stage_elapsed_s = 0.0

        # During material engagement the boom is allowed to trim depth around
        # the penetrate/curl-fill range but not jump to the breakout lift pose.
        authored_min = min(self._penetrate[1], self._curl_filling[1])
        authored_max = max(self._penetrate[1], self._curl_filling[1])
        margin = np.deg2rad(3.0)
        self._cut_boom_lower = max(self.LOWER_RAD[1], authored_min - margin)
        self._cut_boom_upper = min(self.UPPER_RAD[1], authored_max + margin)

        # CUT caps are explicit filling targets.  They deliberately exclude
        # the breakout boom target so CUT cannot silently withdraw from soil.
        self._cut_stick_cap = float(
            np.clip(
                max(self._penetrate[2], self._curl_filling[2]),
                self.LOWER_RAD[2],
                self.UPPER_RAD[2],
            )
        )
        self._cut_bucket_cap = float(
            np.clip(
                max(self._penetrate[3], self._curl_filling[3]),
                self.LOWER_RAD[3],
                self.UPPER_RAD[3],
            )
        )

    def reset(self, entry_rad: np.ndarray, *, payload_m3: float = 0.0) -> None:
        self._command = self._vector(entry_rad, "entry_rad")
        self._validate_target(self._command, "entry")
        self._entry_payload_m3 = max(0.0, float(payload_m3))
        self._mode = "CUT"
        self.elapsed_s = 0.0
        self.stage_elapsed_s = 0.0

    def sample_cut(
        self,
        dt_s: float,
        *,
        mean_depth_m: float,
        engaged_fraction: float,
        cutting_distance_m: float,
        payload_volume_m3: float,
    ) -> CurlScoopSample:
        """Advance one physically engaged cut/scoop command.

        The cutting edge is kept near ``target_cut_depth_m`` by boom feedback.
        Stick retraction and bucket curl happen concurrently.  If edge
        engagement is lost, both stop while boom depth feedback attempts to
        restore contact; the controller never switches itself to the breakout
        pose.
        """

        dt = self._dt(dt_s)
        depth = float(mean_depth_m)
        engagement = float(np.clip(engaged_fraction, 0.0, 1.0))
        cut_distance = max(0.0, float(cutting_distance_m))
        payload_gain = max(0.0, float(payload_volume_m3) - self._entry_payload_m3)
        if not np.all(np.isfinite([depth, engagement, cut_distance, payload_gain])):
            raise ValueError("[390FCurlScoop] non-finite CUT feedback")

        if self._mode != "CUT":
            # A state-machine rollback must never continue a breakout schedule.
            self._mode = "CUT"
            self.stage_elapsed_s = 0.0

        self.elapsed_s += dt
        self.stage_elapsed_s += dt

        # Boom depth hold. Positive depth means the edge is below the terrain;
        # more-negative boom raises the audited 390F cutting edge.
        depth_error = depth - self.target_cut_depth_m
        boom_rate = float(
            np.clip(
                -self._BOOM_DEPTH_GAIN_RAD_S_PER_M * depth_error,
                -self._BOOM_MAX_RATE_RAD_S,
                self._BOOM_MAX_RATE_RAD_S,
            )
        )
        # Never clip a measured reset/entry pose instantaneously into the
        # authored CUT envelope.  If the observed joint starts outside that
        # envelope, approach the nearest bound at the same bounded boom rate.
        # This preserves command continuity at state entry/teleport recovery.
        current_boom = float(self._command[1])
        if current_boom < self._cut_boom_lower:
            self._command[1] = self._move_toward(
                current_boom, self._cut_boom_lower, self._BOOM_MAX_RATE_RAD_S * dt
            )
        elif current_boom > self._cut_boom_upper:
            self._command[1] = self._move_toward(
                current_boom, self._cut_boom_upper, self._BOOM_MAX_RATE_RAD_S * dt
            )
        else:
            self._command[1] = float(
                np.clip(
                    current_boom + boom_rate * dt,
                    self._cut_boom_lower,
                    self._cut_boom_upper,
                )
            )

        # Professional dig behaviour: drag and curl overlap.  When the whole
        # cutting edge is poorly engaged, do not "scoop in the air"; restore
        # contact first and admit no stick/curl filling motion until the edge is
        # materially engaged again.
        engagement_scale = float(
            np.clip(
                engagement / max(self.minimum_engaged_fraction, 1.0e-9),
                0.0,
                1.0,
            )
        )
        depth_scale = float(
            np.clip(depth / max(self.target_cut_depth_m, 1.0e-9), 0.0, 1.0)
        )
        contact_scale = min(engagement_scale, depth_scale)

        # Start curling immediately after engagement, then increase curl once
        # the edge has made measurable forward cut progress or material has
        # started entering the bucket.  This captures the continuum between
        # penetrate-and-drag and penetrate-and-rotate without a timed switch.
        developed_scoop = bool(cut_distance >= 0.15 or payload_gain > 1.0e-4)
        curl_rate = (
            self._BUCKET_SCOOP_CURL_RATE_RAD_S
            if developed_scoop
            else self._BUCKET_EARLY_CURL_RATE_RAD_S
        )
        stick_rate = self._STICK_CUT_RATE_RAD_S
        # The caps are filling targets, not instantaneous clamps.  A measured
        # entry pose already beyond a cap must not jump backwards on the first
        # controller sample.
        if self._command[2] < self._cut_stick_cap:
            self._command[2] = min(
                self._cut_stick_cap,
                self._command[2] + stick_rate * contact_scale * dt,
            )
        if self._command[3] < self._cut_bucket_cap:
            self._command[3] = min(
                self._cut_bucket_cap,
                self._command[3] + curl_rate * contact_scale * dt,
            )

        stage = (
            "C_COORDINATED_DRAG_AND_CURL"
            if developed_scoop
            else "B_ENGAGED_INITIAL_CUT"
        )
        return self._sample(stage, complete=False)

    def begin_breakout(self, entry_rad: np.ndarray) -> None:
        """Synchronise breakout to the *observed* task-state transition."""

        self._command = self._vector(entry_rad, "breakout_entry_rad")
        self._validate_target(self._command, "breakout_entry")
        self._mode = "BREAKOUT_CLOSE"
        self.stage_elapsed_s = 0.0

    def sample_breakout(self, dt_s: float) -> CurlScoopSample:
        """Close/retain material first, then withdraw and raise the bucket."""

        dt = self._dt(dt_s)
        self.elapsed_s += dt
        self.stage_elapsed_s += dt
        close_tolerance = np.deg2rad(5.0)

        # Continue bucket closure and converge stick to the audited breakout
        # value.  Boom is intentionally held until closure is established;
        # this prevents the previous "leave soil, then curl in the air" mode.
        self._command[3] = self._move_toward(
            self._command[3], self._breakout[3], self._BUCKET_BREAKOUT_RATE_RAD_S * dt
        )
        self._command[2] = self._move_toward(
            self._command[2], self._breakout[2], self._STICK_BREAKOUT_RATE_RAD_S * dt
        )
        bucket_closed = bool(abs(self._command[3] - self._breakout[3]) <= close_tolerance)

        if bucket_closed:
            if self._mode != "BREAKOUT_LIFT":
                self._mode = "BREAKOUT_LIFT"
                self.stage_elapsed_s = 0.0
            self._command[1] = self._move_toward(
                self._command[1], self._breakout[1], self._BOOM_MAX_RATE_RAD_S * dt
            )
            stage = "E_BREAKOUT_AFTER_BUCKET_CLOSURE"
        else:
            stage = "D_CLOSE_CAPTURE_BEFORE_LIFT"

        # Swing is not part of the excavation motion and remains fixed until
        # the subsequent transport/swing state owns it.
        self._command[0] = self._breakout[0]
        complete = bool(
            np.max(np.abs(self._command - self._breakout)) <= np.deg2rad(0.5)
        )
        return self._sample(stage, complete=complete)

    def _sample(self, stage: str, *, complete: bool) -> CurlScoopSample:
        target = np.ascontiguousarray(self._command.copy())
        self._validate_target(target, stage)
        target.setflags(write=False)
        return CurlScoopSample(
            stage=stage,
            elapsed_s=float(self.elapsed_s),
            stage_elapsed_s=float(self.stage_elapsed_s),
            target_rad=target,
            complete=bool(complete),
        )

    @classmethod
    def _validate_target(cls, value: np.ndarray, name: str) -> None:
        if np.any(value < cls.LOWER_RAD) or np.any(value > cls.UPPER_RAD):
            raise ValueError(f"[390FCurlScoop] {name} lies outside real USD joint limits")

    @staticmethod
    def _move_toward(value: float, target: float, maximum_delta: float) -> float:
        delta = float(target - value)
        return float(value + np.clip(delta, -maximum_delta, maximum_delta))

    @staticmethod
    def _dt(value: float) -> float:
        dt = float(value)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[390FCurlScoop] dt_s must be finite/positive")
        return dt

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (4,) or not np.all(np.isfinite(result)):
            raise ValueError(f"[390FCurlScoop] {name} must be finite shape (4,)")
        return np.ascontiguousarray(result.copy())
