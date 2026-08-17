"""Phase-D vehicle physics telemetry and profiling primitives."""

from .energy import (
    EnergyTotals,
    MechanicalEnergyAccumulator,
    MechanicalEnergyReport,
)
from .isaac_input import IsaacTelemetryInputAdapter
from .profiler import RuntimeProfiler, RuntimeStatistics
from .recorder import TelemetrySnapshot, VehicleTelemetryRecorder
from .slope_comparison import SlopeTrialMetrics, validate_comparable_slope_trials
from .schema import (
    TELEMETRY_SCHEMA_VERSION,
    ActuatorCategory,
    ActuatorPowerSample,
    EffortSource,
    JointTelemetrySample,
    VehiclePoseSample,
    VehicleTelemetryFrame,
    WheelKinematicsSample,
    WheelTerrainContactSample,
    compute_longitudinal_slip_ratio,
)

__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "ActuatorCategory",
    "ActuatorPowerSample",
    "EffortSource",
    "EnergyTotals",
    "IsaacTelemetryInputAdapter",
    "MechanicalEnergyAccumulator",
    "MechanicalEnergyReport",
    "JointTelemetrySample",
    "RuntimeProfiler",
    "RuntimeStatistics",
    "SlopeTrialMetrics",
    "TelemetrySnapshot",
    "VehiclePoseSample",
    "VehicleTelemetryFrame",
    "VehicleTelemetryRecorder",
    "WheelKinematicsSample",
    "WheelTerrainContactSample",
    "compute_longitudinal_slip_ratio",
    "validate_comparable_slope_trials",
]
