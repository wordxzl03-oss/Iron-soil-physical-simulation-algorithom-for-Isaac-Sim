"""RL-ready reduced-order earthmoving soil physics alpha."""

from ._version import PACKAGE_VERSION as __version__
from .core.config import MaterialConfig, SoilConfig
from .core.soil_physics import SoilPhysics
from .core.state import ReactionWrench, SoilStepResult
from .diagnostics.physics_diagnostics import PhysicsDiagnostics
from .interaction.tool_state import ToolGeometry, ToolState
from .rl.feedback import RLFeedbackAccumulator, RLSoilFeedback
from .running_gear.track_state import TrackGeometry, TrackState

__all__ = [
    "SoilPhysics", "SoilConfig", "MaterialConfig",
    "ToolGeometry", "ToolState", "TrackGeometry", "TrackState",
    "ReactionWrench", "SoilStepResult", "RLSoilFeedback",
    "RLFeedbackAccumulator", "PhysicsDiagnostics", "__version__",
]

