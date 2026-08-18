# P0-2D Tool–Mobile Physical Validity Decision Audit

## Adjudication

```text
VISUAL_DIAGNOSTIC: PASS
FIRST_MOBILE_VISUAL_DIRECTION: NOT_VISUALLY_RESOLVABLE
TOOL_MOBILE_PAYLOAD_CAUSAL_EFFECT: YES
PAYLOAD_ON_M3: 0.016842841629717185
PAYLOAD_OFF_M3: 0.035727877312782687
DELTA_PAYLOAD_M3: 0.018885035683065501
MOUTH_FLUX_ON_M3: 0.016842841629717185
MOUTH_FLUX_OFF_M3: 0.035727877312782687
TOOL_MOBILE_EXPLAINS_OLD_TO_NEW_PAYLOAD_REGRESSION: YES
NORMAL_ORIENTATION_ERROR: NO
INNER_OUTER_SURFACE_CONFUSION: NO
CONTACT_SURFACE_SEMANTIC_ERROR: YES
INNER_FLOOR_3D_DIRECTION: INWARD_WITH_STRONG_DOWNWARD_AND_LATERAL_COMPONENT
INNER_BACK_3D_DIRECTION: NOT_OBSERVED_IN_THIS_TRAJECTORY
LOST_VERTICAL_COMPONENT_PHYSICALLY_IMPORTANT: YES
HORIZONTAL_PROJECTION_REVERSES_OR_DESTROYS_FILLING_DIRECTION: YES
ACTION_REACTION: PASS
MASS_CONSERVATION: PASS
CONTACT_ENERGY_ACCOUNTING: PASS
CURRENT_TOOL_MOBILE_MODEL_CLASS: CURRENT_2P5D_CLOSURE_IMPLEMENTATION_ERROR
PRIMARY_EVIDENCE: OFF recovers 89.15% of the payload gap; floor inward impulse is almost eliminated by 2-D projection; artificial mouth-cap triangles generate physical impulses.
NEXT_ARCHITECTURE_DECISION: separate open-mouth physical contact from watertight containment first; then test a conservative local bucket-frame 3-D retained-material state if floor-direction loss remains.
FIX_APPLIED: NO
```

## Controlled ON/OFF result

Both runs first create Mobile at exactly `5.6333336271345615 s`. The only deliberate intervention is the Tool–Mobile impulse write and its equal/opposite machine reaction. Divergence starts in the first fused Mobile substep at `tau=0`; this is the expected intervention boundary, not a pre-existing trajectory difference.

OFF raises payload from `0.016842841630` to `0.035727877313 m³`, recovering `89.15%` of the gap from ON to the old `0.038026252083 m³` baseline. This proves causality, not that OFF is physical. Capacity rejects remain zero and accepted flux equals candidate flux.

## 3-D to 2.5-D directional result

The coordinate order below is `[toward mouth/interior, lateral, world vertical]`.

| Surface | contact cell-substeps | implied 3-D impulse (N·s) | retained current 2-D impulse (N·s) |
|---|---:|---|---|
| SIDE_WALL | 1991 | [-7.185305126721033, -56.19789019892521, -26.19001436065704] | [-7.676889577602437, -54.000613950373804, 0.0] |
| INNER_FLOOR | 15417 | [139.7732701208495, -158.85452553643745, -267.6215443018478] | [0.13720823125262321, -19.473930671539122, 0.0] |
| TOOTH_CUTTING_EDGE | 16237 | [-141.04171387647855, -857.7830665349775, -1437.7905842492894] | [-342.55236310902836, -298.11233077301193, 0.0] |
| MOUTH_CAP | 4620 | [127.41895867041895, -27.76795475331107, 77.7089665300116] | [96.44675825843805, -63.627366084300014, 0.0] |

For `INNER_FLOOR`, the reconstructed 3-D tendency is `[139.7732701208495, -158.85452553643745, -267.6215443018478] N·s`; current 2-D retains `[0.13720823125262321, -19.473930671539122, 0.0] N·s`. Only `0.0982%` of its signed inward component remains. The lost vertical component is downward in this trajectory, not upward, but it is larger than the inward component and changes the contact interpretation. `INNER_BACK` produced no confirmed contact, so no direction is claimed.

The first 0.2 s aggregate is `{'horizon_s': 0.2, 'current_2d_impulse_terrain_ns': [2.631388029735696, -12.349751344454063, 0.0], 'implied_3d_impulse_terrain_ns': [12.125062951711232, -3.414018262901842, -40.05398072543176]}`. In the screenshots, most vectors/points are occluded by the real bucket at this scale, so the visual-only coherent direction is honestly `NOT_VISUALLY_RESOLVABLE`; the directional decision comes from exact contact records rather than appearance.

## Surface semantics

The descriptor is a validated 36-vertex/68-triangle reduced-order closed cavity. Normal orientation is generated as center-minus-closest-point (winding is used only at zero distance), and no geometric evidence supports flipping it. There is no inner/outer mix-up.

There is, however, a contact-surface semantic error: profile segment 16 closes the physically open mouth from top edge to lip so point-in-closed-mesh queries work, yet production also permits that artificial cap to deliver contact impulse. It was active for `2781` cell-substeps. That makes implementation semantics the primary model class even though floor projection also exposes a structural limitation.

The support's nominal 3-D normal field has a legacy publication semantic—x/y are normalized in-place while z is raw. Production only consumes the intended unit horizontal normal; this audit reconstructs a unit 3-D normal before comparison. No production field or law was changed.

## Conservation and energy

- Maximum mass error: ON `1.523e-11 m³`; OFF `1.728e-11 m³`.
- Maximum action/reaction residual: `0.000e+00 N·s`.
- Summed unexplained contact energy: `1.330e-15 J`; maximum per substep `2.665e-15 J`.
- Contact dissipation is nonnegative; sum `133.238366431 J`.
- Reconstructed/current Warp impulse maximum error: `3.553e-15 N·s`.

## Isaac visual evidence

Fourteen side/top frames were captured at 0/1/2/3/5/10/20 dt from first creation. Their first bucket pose, joints, penetration and R2M state match the frozen headless replay exactly. The GUI clock has a `-2.8 s` offset and capture/pause perturbs the later scheduling, so the GUI run is used only for state-aligned first-write visualization; all causal values above come from synchronized headless ON/OFF.

- `first_mobile_plus_00dt` side: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_00dt_side.png`
- `first_mobile_plus_00dt` top: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_00dt_top.png`
- `first_mobile_plus_01dt` side: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_01dt_side.png`
- `first_mobile_plus_01dt` top: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_01dt_top.png`
- `first_mobile_plus_02dt` side: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_02dt_side.png`
- `first_mobile_plus_02dt` top: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_02dt_top.png`
- `first_mobile_plus_03dt` side: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_03dt_side.png`
- `first_mobile_plus_03dt` top: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_03dt_top.png`
- `first_mobile_plus_05dt` side: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_05dt_side.png`
- `first_mobile_plus_05dt` top: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_05dt_top.png`
- `first_mobile_plus_10dt` side: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_10dt_side.png`
- `first_mobile_plus_10dt` top: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_10dt_top.png`
- `first_mobile_plus_20dt` side: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_20dt_side.png`
- `first_mobile_plus_20dt` top: `/home/eric/Desktop/mesh/outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual_frames/first_mobile_plus_20dt_top.png`

Arrow scale is visual-only: `endpoint = origin + 0.35 * q/h`.

## Reproduction

```bash
./run_390f_interactive.sh --headless --runtime-backend GPU_RUNTIME --realistic-cut-scoop --tool-mobile-validity-audit outputs/mobile_v2_production/p0_2d_tool_mobile_validity_on.json --tool-mobile-impulse-ablation ON
./run_390f_interactive.sh --headless --runtime-backend GPU_RUNTIME --realistic-cut-scoop --tool-mobile-validity-audit outputs/mobile_v2_production/p0_2d_tool_mobile_validity_off.json --tool-mobile-impulse-ablation OFF
./run_390f_interactive.sh --no-headless --runtime-backend GPU_RUNTIME --realistic-cut-scoop --tool-mobile-validity-audit outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual.json --tool-mobile-impulse-ablation ON --tool-mobile-validity-visual-diagnostic
```

Machine-readable outputs: `outputs/mobile_v2_production/p0_2d_tool_mobile_physical_validity_decision_audit.json` and `outputs/mobile_v2_production/p0_2d_tool_mobile_validity_on_off_timeseries.json`.
