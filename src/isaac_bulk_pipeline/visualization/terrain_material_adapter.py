"""Visual-only iron-ore-fines material schema and USD adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class TerrainMaterialPreset:
    name: str
    base_color_rgb: tuple[float, float, float]
    roughness: float
    metallic: float
    opacity: float = 1.0
    provenance: str = "VISUAL_ONLY"
    displacement_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("[TerrainMaterial] name must be a non-empty identifier")
        if self.provenance != "VISUAL_ONLY" or self.displacement_enabled:
            raise ValueError("[TerrainMaterial] Phase-B material must remain visual-only without displacement")
        if len(self.base_color_rgb) != 3:
            raise ValueError("[TerrainMaterial] base_color_rgb must have three channels")
        color = tuple(float(value) for value in self.base_color_rgb)
        scalar_values = (float(self.roughness), float(self.metallic), float(self.opacity))
        if any(not math.isfinite(value) for value in (*color, *scalar_values)):
            raise ValueError("[TerrainMaterial] all values must be finite")
        if any(value < 0.0 or value > 1.0 for value in (*color, *scalar_values)):
            raise ValueError("[TerrainMaterial] color and PBR values must be in [0,1]")
        object.__setattr__(self, "base_color_rgb", color)
        object.__setattr__(self, "roughness", scalar_values[0])
        object.__setattr__(self, "metallic", scalar_values[1])
        object.__setattr__(self, "opacity", scalar_values[2])

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_color_rgb": list(self.base_color_rgb),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "opacity": self.opacity,
            "provenance": self.provenance,
            "displacement_enabled": self.displacement_enabled,
        }


TERRAIN_MATERIAL_PRESETS: Mapping[str, TerrainMaterialPreset] = {
    "iron_ore_fines": TerrainMaterialPreset(
        name="iron_ore_fines",
        base_color_rgb=(0.24, 0.075, 0.035),
        roughness=0.91,
        metallic=0.02,
    ),
    "damp_iron_ore": TerrainMaterialPreset(
        name="damp_iron_ore",
        base_color_rgb=(0.12, 0.035, 0.018),
        roughness=0.82,
        metallic=0.01,
    ),
}


def get_terrain_material_preset(name: str) -> TerrainMaterialPreset:
    try:
        return TERRAIN_MATERIAL_PRESETS[str(name)]
    except KeyError as error:
        raise ValueError(
            f"[TerrainMaterial] unknown preset={name!r}; available={sorted(TERRAIN_MATERIAL_PRESETS)}"
        ) from error


class TerrainMaterialAdapter:
    """Author a UsdPreviewSurface without touching terrain points or heights."""

    def __init__(self, preset: str | TerrainMaterialPreset = "iron_ore_fines") -> None:
        self.preset = (
            get_terrain_material_preset(preset) if isinstance(preset, str) else preset
        )

    def author_and_bind_usd(
        self,
        stage: object,
        terrain_prim_path: str,
        material_path: str = "/World/Looks/PhaseBIronOreFines",
    ) -> dict[str, Any]:
        from pxr import Gf, Sdf, UsdShade  # type: ignore

        terrain = stage.GetPrimAtPath(terrain_prim_path)
        if not terrain or not terrain.IsValid():
            raise ValueError(f"[TerrainMaterial] invalid terrain prim: {terrain_prim_path}")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*self.preset.base_color_rgb)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(self.preset.roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(self.preset.metallic)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(self.preset.opacity)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(terrain).Bind(material)
        return self.preset.as_dict()
