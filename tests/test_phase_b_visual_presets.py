import json
from pathlib import Path
import unittest

import yaml

from isaac_bulk_pipeline.visualization.lighting_manager import (
    LIGHTING_PRESETS,
    LightingManager,
    LightingPreset,
    get_lighting_preset,
)
from isaac_bulk_pipeline.visualization.terrain_material_adapter import (
    TERRAIN_MATERIAL_PRESETS,
    TerrainMaterialPreset,
    get_terrain_material_preset,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PhaseBVisualPresetTests(unittest.TestCase):
    def test_three_lighting_presets_are_valid_and_deterministic(self):
        self.assertEqual(set(LIGHTING_PRESETS), {"outdoor_day", "overcast", "strong_sun"})
        for name in sorted(LIGHTING_PRESETS):
            first = LightingManager(name).preset.as_dict()
            second = get_lighting_preset(name).as_dict()
            self.assertEqual(
                json.dumps(first, sort_keys=True, separators=(",", ":")),
                json.dumps(second, sort_keys=True, separators=(",", ":")),
            )
            self.assertGreater(first["dome"]["intensity"], 0.0)
            self.assertGreater(first["sun"]["intensity"], 0.0)

    def test_invalid_lighting_values_are_rejected(self):
        values = dict(LIGHTING_PRESETS["outdoor_day"].__dict__)
        values["dome_color"] = (1.2, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "color"):
            LightingPreset(**values)
        with self.assertRaisesRegex(ValueError, "unknown preset"):
            get_lighting_preset("moonbase")

    def test_terrain_material_is_dark_rough_nonmetallic_and_no_displacement(self):
        self.assertEqual(set(TERRAIN_MATERIAL_PRESETS), {"iron_ore_fines", "damp_iron_ore"})
        preset = get_terrain_material_preset("iron_ore_fines")
        self.assertLess(max(preset.base_color_rgb), 0.3)
        self.assertGreater(preset.roughness, 0.8)
        self.assertLess(preset.metallic, 0.05)
        self.assertFalse(preset.displacement_enabled)
        with self.assertRaisesRegex(ValueError, "visual-only"):
            TerrainMaterialPreset("bad", (0.1, 0.1, 0.1), 0.8, 0.0, displacement_enabled=True)

    def test_visual_yaml_names_only_known_presets_and_forbids_height_writeback(self):
        document = yaml.safe_load(
            (REPOSITORY_ROOT / "configs/phase_b_visual.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(document["provenance"], "VISUAL_ONLY")
        self.assertEqual(set(document["lighting"]["available_presets"]), set(LIGHTING_PRESETS))
        self.assertEqual(
            set(document["terrain_material"]["available_presets"]),
            set(TERRAIN_MATERIAL_PRESETS),
        )
        self.assertFalse(document["terrain_material"]["displacement_enabled"])
        self.assertFalse(document["terrain_material"]["heightmap_writeback_enabled"])


if __name__ == "__main__":
    unittest.main()
