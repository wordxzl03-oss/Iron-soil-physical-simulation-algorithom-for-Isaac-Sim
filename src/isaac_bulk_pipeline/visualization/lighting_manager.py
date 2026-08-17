"""Validated, deterministic Phase-B daylight presets.

The schema is importable without Isaac.  USD imports are deliberately local to
``author_usd`` so pure unit tests never need to start Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"[LightingPreset] {name} must be finite")
    return result


def _color(value: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("[LightingPreset] color must contain exactly three channels")
    result = tuple(_finite("color", item) for item in value)
    if any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError("[LightingPreset] color channels must be in [0,1]")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class LightingPreset:
    name: str
    dome_intensity: float
    dome_exposure: float
    dome_color: tuple[float, float, float]
    sun_intensity: float
    sun_exposure: float
    sun_color: tuple[float, float, float]
    sun_angle_deg: float
    sun_rotation_xyz_deg: tuple[float, float, float]
    fill_enabled: bool
    fill_intensity: float
    fill_exposure: float
    fill_color: tuple[float, float, float]
    fill_radius_m: float
    fill_position_m: tuple[float, float, float]
    provenance: str = "VISUAL_ONLY"

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("[LightingPreset] name must be a non-empty identifier")
        if self.provenance != "VISUAL_ONLY":
            raise ValueError("[LightingPreset] provenance must be VISUAL_ONLY")
        for name in (
            "dome_intensity",
            "dome_exposure",
            "sun_intensity",
            "sun_exposure",
            "sun_angle_deg",
            "fill_intensity",
            "fill_exposure",
            "fill_radius_m",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.dome_intensity < 0.0 or self.sun_intensity < 0.0 or self.fill_intensity < 0.0:
            raise ValueError("[LightingPreset] intensities must be non-negative")
        if not 0.0 < self.sun_angle_deg <= 180.0:
            raise ValueError("[LightingPreset] sun_angle_deg must be in (0,180]")
        if self.fill_radius_m <= 0.0:
            raise ValueError("[LightingPreset] fill_radius_m must be positive")
        object.__setattr__(self, "dome_color", _color(self.dome_color))
        object.__setattr__(self, "sun_color", _color(self.sun_color))
        object.__setattr__(self, "fill_color", _color(self.fill_color))
        for name in ("sun_rotation_xyz_deg", "fill_position_m"):
            value = tuple(_finite(name, item) for item in getattr(self, name))
            if len(value) != 3:
                raise ValueError(f"[LightingPreset] {name} must have three components")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provenance": self.provenance,
            "dome": {
                "intensity": self.dome_intensity,
                "exposure": self.dome_exposure,
                "color": list(self.dome_color),
            },
            "sun": {
                "intensity": self.sun_intensity,
                "exposure": self.sun_exposure,
                "color": list(self.sun_color),
                "angle_deg": self.sun_angle_deg,
                "rotation_xyz_deg": list(self.sun_rotation_xyz_deg),
            },
            "fill": {
                "enabled": self.fill_enabled,
                "intensity": self.fill_intensity,
                "exposure": self.fill_exposure,
                "color": list(self.fill_color),
                "radius_m": self.fill_radius_m,
                "position_m": list(self.fill_position_m),
            },
        }


LIGHTING_PRESETS: Mapping[str, LightingPreset] = {
    "outdoor_day": LightingPreset(
        "outdoor_day",
        320.0,
        0.0,
        (0.82, 0.90, 1.0),
        1_250.0,
        0.0,
        (1.0, 0.91, 0.76),
        1.0,
        (-48.0, 24.0, -32.0),
        True,
        8_000.0,
        0.0,
        (0.88, 0.93, 1.0),
        6.0,
        (12.0, 8.0, 22.0),
    ),
    "overcast": LightingPreset(
        "overcast",
        650.0,
        0.0,
        (0.78, 0.84, 0.90),
        280.0,
        0.0,
        (0.92, 0.95, 1.0),
        8.0,
        (-65.0, 10.0, -15.0),
        True,
        5_000.0,
        0.0,
        (0.86, 0.91, 1.0),
        8.0,
        (8.0, -6.0, 20.0),
    ),
    "strong_sun": LightingPreset(
        "strong_sun",
        220.0,
        0.0,
        (0.72, 0.84, 1.0),
        2_400.0,
        0.0,
        (1.0, 0.86, 0.64),
        0.5,
        (-38.0, 32.0, -45.0),
        True,
        6_000.0,
        0.0,
        (0.80, 0.90, 1.0),
        5.0,
        (10.0, 10.0, 24.0),
    ),
}


def get_lighting_preset(name: str) -> LightingPreset:
    try:
        return LIGHTING_PRESETS[str(name)]
    except KeyError as error:
        raise ValueError(
            f"[LightingPreset] unknown preset={name!r}; available={sorted(LIGHTING_PRESETS)}"
        ) from error


class LightingManager:
    """Author and inspect one preset at stable USD paths."""

    def __init__(self, preset: str | LightingPreset = "outdoor_day") -> None:
        self.preset = get_lighting_preset(preset) if isinstance(preset, str) else preset

    def author_usd(self, stage: object, root_path: str = "/World/PhaseBLighting") -> dict[str, Any]:
        from pxr import Gf, UsdGeom, UsdLux  # type: ignore

        preset = self.preset
        UsdGeom.Xform.Define(stage, root_path)
        dome = UsdLux.DomeLight.Define(stage, f"{root_path}/Sky")
        dome.CreateIntensityAttr(preset.dome_intensity)
        dome.CreateExposureAttr(preset.dome_exposure)
        dome.CreateColorAttr(Gf.Vec3f(*preset.dome_color))

        sun = UsdLux.DistantLight.Define(stage, f"{root_path}/Sun")
        sun.CreateIntensityAttr(preset.sun_intensity)
        sun.CreateExposureAttr(preset.sun_exposure)
        sun.CreateColorAttr(Gf.Vec3f(*preset.sun_color))
        sun.CreateAngleAttr(preset.sun_angle_deg)
        sun_xform = UsdGeom.Xformable(sun.GetPrim())
        sun_xform.ClearXformOpOrder()
        sun_xform.AddRotateXYZOp().Set(Gf.Vec3f(*preset.sun_rotation_xyz_deg))

        fill_path = f"{root_path}/OverheadFill"
        if preset.fill_enabled:
            fill = UsdLux.SphereLight.Define(stage, fill_path)
            fill.CreateIntensityAttr(preset.fill_intensity)
            fill.CreateExposureAttr(preset.fill_exposure)
            fill.CreateColorAttr(Gf.Vec3f(*preset.fill_color))
            fill.CreateRadiusAttr(preset.fill_radius_m)
            fill_xform = UsdGeom.Xformable(fill.GetPrim())
            fill_xform.ClearXformOpOrder()
            fill_xform.AddTranslateOp().Set(Gf.Vec3d(*preset.fill_position_m))
        else:
            existing = stage.GetPrimAtPath(fill_path)
            if existing and existing.IsValid():
                stage.RemovePrim(fill_path)
        return preset.as_dict()

    def inspect_usd(self, stage: object, root_path: str = "/World/PhaseBLighting") -> dict[str, Any]:
        """Read back the core authored values for runtime acceptance evidence."""

        def value(path: str, attribute: str) -> Any:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return None
            attr = prim.GetAttribute(attribute)
            raw = attr.Get() if attr and attr.HasAuthoredValueOpinion() else None
            if raw is None or isinstance(raw, (str, int, float, bool)):
                return raw
            try:
                return [float(item) for item in raw]
            except TypeError:
                return str(raw)

        return {
            "preset": self.preset.name,
            "dome_intensity": value(f"{root_path}/Sky", "inputs:intensity"),
            "sun_intensity": value(f"{root_path}/Sun", "inputs:intensity"),
            "sun_angle_deg": value(f"{root_path}/Sun", "inputs:angle"),
            "fill_intensity": value(f"{root_path}/OverheadFill", "inputs:intensity"),
        }
