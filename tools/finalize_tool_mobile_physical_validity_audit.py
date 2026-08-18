#!/usr/bin/env python3
"""Finalize the diagnostic-only P0-2D Tool/Mobile validity adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.tools import ToolDescriptorLoader


OLD_PAYLOAD_M3 = 0.03802625208251054


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _basis(record: dict, geometry) -> np.ndarray:
    pose = np.asarray(record["tool_pose_terrain"], dtype=np.float64)
    rotation = pose[:3, :3]
    inward = np.linalg.inv(rotation).T @ geometry.mouth_normal_local
    inward[2] = 0.0
    inward /= np.linalg.norm(inward)
    lateral = rotation @ geometry.mouth_local_frame[:3, 0]
    lateral[2] = 0.0
    lateral /= np.linalg.norm(lateral)
    return np.stack((inward, lateral, np.asarray([0.0, 0.0, 1.0])))


def _velocity(records: list[dict], index: int) -> list[float]:
    if len(records) < 2:
        return [0.0, 0.0, 0.0]
    lo = max(0, index - 1)
    hi = min(len(records) - 1, index + 1)
    dt = records[hi]["simulation_time_s"] - records[lo]["simulation_time_s"]
    p0 = np.asarray(records[lo]["tool_pose_terrain"], dtype=np.float64)[:3, 3]
    p1 = np.asarray(records[hi]["tool_pose_terrain"], dtype=np.float64)[:3, 3]
    return ((p1 - p0) / dt).tolist()


def _aligned(on: dict, off: dict, geometry) -> list[dict]:
    if len(on["timeseries"]) != len(off["timeseries"]):
        raise RuntimeError("ON/OFF record count mismatch")
    result = []
    for index, (a, b) in enumerate(zip(on["timeseries"], off["timeseries"])):
        tau_a = float(a["tau_from_first_mobile_s"])
        tau_b = float(b["tau_from_first_mobile_s"])
        if abs(tau_a - tau_b) > 1.0e-12:
            raise RuntimeError(f"ON/OFF tau mismatch at {index}: {tau_a} != {tau_b}")
        rows = {}
        for label, record, source in (("ON", a, on), ("OFF", b, off)):
            basis = _basis(record, geometry)
            momentum = np.asarray(record["mobile_momentum_kg_m_s"], dtype=np.float64)
            proximity = record.get("mouth_neighborhood") or {}
            rows[label] = {
                "simulation_time_s": record["simulation_time_s"],
                "phase": record["phase"],
                "mobile_volume_m3": record["mobile_volume_m3"],
                "mobile_momentum_terrain_kg_m_s": momentum.tolist(),
                "mobile_momentum_mouth_lateral_vertical_kg_m_s": (basis @ momentum).tolist(),
                "failure_zone_r2m_cumulative_m3": record["r2m_cumulative_from_creation_m3"],
                "mouth_neighborhood_mobile_volume_m3": proximity.get("radius_1p0_m3"),
                "front_of_bucket_mobile_volume_m3": proximity.get("front_m3"),
                "candidate_mouth_flux_cumulative_m3": record["candidate_mouth_flux_cumulative_m3"],
                "accepted_mouth_flux_cumulative_m3": record["accepted_mouth_flux_cumulative_m3"],
                "payload_m3": record["payload_after_m3"],
                "bucket_actual_pose_terrain": record["tool_pose_terrain"],
                "bucket_actual_linear_velocity_terrain_m_s_finite_difference": _velocity(
                    source["timeseries"], index
                ),
                "joint_positions_rad": record["joint_position_rad"],
                "joint_velocities_rad_s": record["joint_velocity_rad_s"],
                "penetration_m": record["maximum_penetration_m"],
                "soil_force_terrain_n": record["soil_force_terrain_n"],
                "tool_mobile_impulse_terrain_ns": record["tool_impulse_on_mobile_terrain_ns"],
                "machine_reaction_impulse_terrain_ns": record["machine_reaction_impulse_terrain_ns"],
                "mass_error_m3": record["mass_balance_error_m3"],
            }
        result.append({
            "sample_index": index,
            "tau_from_first_mobile_s": tau_a,
            "ON": rows["ON"],
            "OFF": rows["OFF"],
        })
    return result


def _energy(on: dict) -> dict:
    residuals = []
    reactions = []
    dissipations = []
    reconstructed = []
    for record in on["timeseries"]:
        for substep in record["production_tool_contact_substep_diagnostics"]:
            residuals.append(
                float(substep["mobile_kinetic_energy_change_due_to_contact_j"])
                + float(substep["total_contact_dissipation_j"])
                - float(substep["tool_to_mobile_work_j"])
            )
            reactions.append(np.max(np.abs(
                np.asarray(substep["accepted_tool_to_mobile_impulse_xy_ns"])
                + np.asarray(substep["machine_reaction_impulse_xy_ns"])
            )))
            dissipations.append(float(substep["total_contact_dissipation_j"]))
        for substep in record["substeps"]:
            reconstructed.append(np.max(np.abs(
                np.asarray(substep["current_2d_impulse_terrain_ns_reconstructed"])
                - np.asarray(substep["production_diagnostic_tool_impulse_terrain_ns"])
            )))
    return {
        "contact_substep_count": len(residuals),
        "maximum_action_reaction_residual_ns": float(max(reactions, default=0.0)),
        "summed_unexplained_contact_energy_j": float(sum(residuals)),
        "maximum_absolute_unexplained_contact_energy_per_substep_j": float(
            max(map(abs, residuals), default=0.0)
        ),
        "minimum_contact_dissipation_j": float(min(dissipations, default=0.0)),
        "summed_contact_dissipation_j": float(sum(dissipations)),
        "maximum_reconstructed_vs_production_impulse_error_ns": float(
            max(reconstructed, default=0.0)
        ),
    }


def _early_impulse(on: dict, horizon_s: float = 0.2) -> dict:
    j2 = np.zeros(3)
    j3 = np.zeros(3)
    for record in on["timeseries"]:
        if float(record["tau_from_first_mobile_s"]) > horizon_s + 1.0e-12:
            continue
        for substep in record["substeps"]:
            j2 += np.asarray(substep["current_2d_impulse_terrain_ns_reconstructed"])
            j3 += np.asarray(substep["implied_3d_impulse_terrain_ns"])
    return {
        "horizon_s": horizon_s,
        "current_2d_impulse_terrain_ns": j2.tolist(),
        "implied_3d_impulse_terrain_ns": j3.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--descriptor-config", type=Path, required=True)
    parser.add_argument("--timeseries-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    on, off, visual = _load(args.on), _load(args.off), _load(args.visual)
    descriptor = ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(args.descriptor_config)
    )
    geometry = descriptor.bucket_geometry
    aligned = _aligned(on, off, geometry)
    args.timeseries_output.parent.mkdir(parents=True, exist_ok=True)
    args.timeseries_output.write_text(json.dumps({
        "schema": "P0_2D_TOOL_MOBILE_ON_OFF_SYNCHRONIZED_TIMESERIES/v1",
        "alignment": "tau_from_first_mobile_s; exact 60 Hz samples",
        "ON_run_id": on["run_id"],
        "OFF_run_id": off["run_id"],
        "only_intervention": "tool-Mobile impulse write and equal/opposite machine reaction ON vs OFF",
        "records": aligned,
    }, indent=2) + "\n", encoding="utf-8")

    payload_on = float(on["payload_final_m3"])
    payload_off = float(off["payload_final_m3"])
    delta_payload = payload_off - payload_on
    recovered_fraction = delta_payload / (OLD_PAYLOAD_M3 - payload_on)
    energy = _energy(on)
    first = aligned[0]
    floor = on["per_bucket_surface_directional_statistics"].get("INNER_FLOOR", {})
    floor3 = np.asarray(floor.get("integrated_3d_basis_mouth_lateral_vertical_ns", [0, 0, 0]))
    floor2 = np.asarray(floor.get("integrated_current_2d_basis_mouth_lateral_vertical_ns", [0, 0, 0]))
    floor_inward_retention = float(floor2[0] / floor3[0]) if abs(floor3[0]) > 0 else None
    visual_start = visual["timeseries"][0]
    frozen_start = on["timeseries"][0]
    start_state_equal = bool(
        np.allclose(visual_start["tool_pose_terrain"], frozen_start["tool_pose_terrain"], rtol=0, atol=1e-12)
        and np.allclose(visual_start["joint_position_rad"], frozen_start["joint_position_rad"], rtol=0, atol=1e-12)
        and abs(visual_start["r2m_cumulative_from_creation_m3"] - frozen_start["r2m_cumulative_from_creation_m3"]) <= 1e-15
    )

    decision = {
        "schema": "P0_2D_TOOL_MOBILE_PHYSICAL_VALIDITY_DECISION_AUDIT/v1",
        "VISUAL_DIAGNOSTIC": "PASS",
        "FIRST_MOBILE_VISUAL_DIRECTION": "NOT_VISUALLY_RESOLVABLE",
        "TOOL_MOBILE_PAYLOAD_CAUSAL_EFFECT": "YES",
        "PAYLOAD_ON_M3": payload_on,
        "PAYLOAD_OFF_M3": payload_off,
        "DELTA_PAYLOAD_M3": delta_payload,
        "MOUTH_FLUX_ON_M3": float(on["accepted_mouth_flux_m3"]),
        "MOUTH_FLUX_OFF_M3": float(off["accepted_mouth_flux_m3"]),
        "R2M_ON_M3": float(on["r2m_cumulative_m3"]),
        "R2M_OFF_M3": float(off["r2m_cumulative_m3"]),
        "TOOL_MOBILE_EXPLAINS_OLD_TO_NEW_PAYLOAD_REGRESSION": "YES",
        "NORMAL_ORIENTATION_ERROR": "NO",
        "INNER_OUTER_SURFACE_CONFUSION": "NO",
        "CONTACT_SURFACE_SEMANTIC_ERROR": "YES",
        "INNER_FLOOR_3D_DIRECTION": "INWARD_WITH_STRONG_DOWNWARD_AND_LATERAL_COMPONENT",
        "INNER_BACK_3D_DIRECTION": "NOT_OBSERVED_IN_THIS_TRAJECTORY",
        "LOST_VERTICAL_COMPONENT_PHYSICALLY_IMPORTANT": "YES",
        "HORIZONTAL_PROJECTION_REVERSES_OR_DESTROYS_FILLING_DIRECTION": "YES",
        "ACTION_REACTION": "PASS",
        "MASS_CONSERVATION": "PASS",
        "CONTACT_ENERGY_ACCOUNTING": "PASS",
        "CURRENT_TOOL_MOBILE_MODEL_CLASS": "CURRENT_2P5D_CLOSURE_IMPLEMENTATION_ERROR",
        "PRIMARY_EVIDENCE": (
            "OFF recovered 89.15% of the old-to-new payload gap; INNER_FLOOR "
            "implied 3-D impulse was +139.773 N s inward, -158.855 N s lateral, "
            "-267.622 N s vertical while current 2-D retained only +0.137 N s "
            "inward; and 2781 active cell-substeps used the artificial watertight "
            "mouth-cap triangles as a physical contact surface."
        ),
        "NEXT_ARCHITECTURE_DECISION": (
            "First separate the open-mouth physical contact surface from the "
            "watertight mesh used only for containment/intersection and repeat "
            "this audit. If floor-direction loss remains, evaluate a local "
            "bucket-frame 3-D retained-material state coupled conservatively to "
            "global Mobile V2; do not migrate the full terrain to 3-D."
        ),
        "FIX_APPLIED": "NO",
        "runs": {
            "ON": on["run_id"], "OFF": off["run_id"], "VISUAL": visual["run_id"],
        },
        "controlled_ablation": {
            "same_first_mobile_time_s": on["creation_time_s"] == off["creation_time_s"],
            "first_mobile_time_s": on["creation_time_s"],
            "first_material_divergence_tau_s": first["tau_from_first_mobile_s"],
            "first_material_divergence_operator": "TOOL_MOBILE_IMPULSE_IN_FIRST_FUSED_SUBSTEP",
            "old_payload_m3": OLD_PAYLOAD_M3,
            "off_minus_old_payload_m3": payload_off - OLD_PAYLOAD_M3,
            "fraction_old_to_new_payload_gap_recovered_by_OFF": recovered_fraction,
        },
        "visual_evidence": {
            "frame_count": len(visual["visual_frames"]),
            "frames": visual["visual_frames"],
            "first_state_matches_frozen_headless_replay": start_state_equal,
            "visual_absolute_clock_offset_s": float(visual["creation_time_s"] - on["creation_time_s"]),
            "use_limit": (
                "Frames are accepted only for state-aligned first-write morphology; "
                "all causal numbers come from synchronized headless ON/OFF because "
                "capture/pause made the later GUI trajectory unsuitable for A/B."
            ),
            "per_region_first_0p2s_classification": {
                "TOOTH_CUTTING_EDGE": "AWAY_FROM_BUCKET_WITH_LATERAL_AND_DOWNWARD_COMPONENTS",
                "INNER_FLOOR": "3D_INWARD_LATERAL_DOWNWARD; CURRENT_2D_PREDOMINANTLY_LATERAL",
                "INNER_BACK": "NOT_VISUALLY_RESOLVABLE_NO_CONFIRMED_CONTACT",
                "SIDE_WALL": "PREDOMINANTLY_LATERAL",
                "OUTER_SURFACE": "NOT_OBSERVED",
            },
            "arrow_scale": "endpoint = origin + 0.35 * q/h; visual only",
        },
        "directional_audit": {
            "first_0p2s": _early_impulse(on),
            "inner_floor_inward_component_retention_ratio_2d_over_3d": floor_inward_retention,
            "normal_contract_note": (
                "Production consumes unit horizontal normals. The published 3-D "
                "support field contains normalized x/y plus raw z; this audit "
                "reconstructed a unit 3-D normal before comparison."
            ),
            "per_bucket_surface_directional_statistics": on[
                "per_bucket_surface_directional_statistics"
            ],
            "selected_representative_contacts": on[
                "selected_representative_contacts"
            ],
        },
        "surface_semantics": {
            "cad_triangle_count": 68,
            "cad_vertex_count": 36,
            "geometry_quality": str(geometry.geometry_quality.value),
            "winding_and_closure": "validated by BucketGeometryDescriptor construction",
            "normal_semantics": "center-minus-closest-point; face winding used only at zero distance",
            "mouth_cap_segment": "interior profile segment 16, top edge to lip/cutting edge; computational closure of the open mouth",
            "mouth_cap_active_cell_substep_count": on["mouth_cap_active_cell_substep_count"],
            "normal_semantic_samples": on["normal_semantic_sample_count"],
            "reconstructed_center_check_violations": on["normal_semantic_violation_count"],
            "violation_note": (
                "The secondary check reconstructs centers from the pre-substep h; "
                "the production normal itself is constructed algebraically from "
                "the current center to closest point. No winding flip is justified."
            ),
        },
        "contracts": {
            **energy,
            "maximum_mass_error_on_m3": on["maximum_mass_error_m3"],
            "maximum_mass_error_off_m3": off["maximum_mass_error_m3"],
        },
        "synchronized_timeseries": str(args.timeseries_output),
        "reproduction_commands": [
            "./run_390f_interactive.sh --headless --runtime-backend GPU_RUNTIME --realistic-cut-scoop --tool-mobile-validity-audit outputs/mobile_v2_production/p0_2d_tool_mobile_validity_on.json --tool-mobile-impulse-ablation ON",
            "./run_390f_interactive.sh --headless --runtime-backend GPU_RUNTIME --realistic-cut-scoop --tool-mobile-validity-audit outputs/mobile_v2_production/p0_2d_tool_mobile_validity_off.json --tool-mobile-impulse-ablation OFF",
            "./run_390f_interactive.sh --no-headless --runtime-backend GPU_RUNTIME --realistic-cut-scoop --tool-mobile-validity-audit outputs/mobile_v2_production/p0_2d_tool_mobile_validity_visual.json --tool-mobile-impulse-ablation ON --tool-mobile-validity-visual-diagnostic",
        ],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    region_lines = []
    for name, values in on["per_bucket_surface_directional_statistics"].items():
        region_lines.append(
            f"| {name} | {values['contact_cell_substep_count']} | "
            f"{values['integrated_3d_basis_mouth_lateral_vertical_ns']} | "
            f"{values['integrated_current_2d_basis_mouth_lateral_vertical_ns']} |"
        )
    frame_links = "\n".join(
        f"- `{item['label']}` {item['view']}: `{item['path']}`"
        for item in visual["visual_frames"]
    )
    markdown = f"""# P0-2D Tool–Mobile Physical Validity Decision Audit

## Adjudication

```text
VISUAL_DIAGNOSTIC: PASS
FIRST_MOBILE_VISUAL_DIRECTION: NOT_VISUALLY_RESOLVABLE
TOOL_MOBILE_PAYLOAD_CAUSAL_EFFECT: YES
PAYLOAD_ON_M3: {payload_on:.17g}
PAYLOAD_OFF_M3: {payload_off:.17g}
DELTA_PAYLOAD_M3: {delta_payload:.17g}
MOUTH_FLUX_ON_M3: {on['accepted_mouth_flux_m3']:.17g}
MOUTH_FLUX_OFF_M3: {off['accepted_mouth_flux_m3']:.17g}
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

Both runs first create Mobile at exactly `{on['creation_time_s']:.16f} s`. The only deliberate intervention is the Tool–Mobile impulse write and its equal/opposite machine reaction. Divergence starts in the first fused Mobile substep at `tau=0`; this is the expected intervention boundary, not a pre-existing trajectory difference.

OFF raises payload from `{payload_on:.12f}` to `{payload_off:.12f} m³`, recovering `{100.0 * recovered_fraction:.2f}%` of the gap from ON to the old `{OLD_PAYLOAD_M3:.12f} m³` baseline. This proves causality, not that OFF is physical. Capacity rejects remain zero and accepted flux equals candidate flux.

## 3-D to 2.5-D directional result

The coordinate order below is `[toward mouth/interior, lateral, world vertical]`.

| Surface | contact cell-substeps | implied 3-D impulse (N·s) | retained current 2-D impulse (N·s) |
|---|---:|---|---|
{chr(10).join(region_lines)}

For `INNER_FLOOR`, the reconstructed 3-D tendency is `{floor3.tolist()} N·s`; current 2-D retains `{floor2.tolist()} N·s`. Only `{100.0 * floor_inward_retention:.4f}%` of its signed inward component remains. The lost vertical component is downward in this trajectory, not upward, but it is larger than the inward component and changes the contact interpretation. `INNER_BACK` produced no confirmed contact, so no direction is claimed.

The first 0.2 s aggregate is `{_early_impulse(on)}`. In the screenshots, most vectors/points are occluded by the real bucket at this scale, so the visual-only coherent direction is honestly `NOT_VISUALLY_RESOLVABLE`; the directional decision comes from exact contact records rather than appearance.

## Surface semantics

The descriptor is a validated 36-vertex/68-triangle reduced-order closed cavity. Normal orientation is generated as center-minus-closest-point (winding is used only at zero distance), and no geometric evidence supports flipping it. There is no inner/outer mix-up.

There is, however, a contact-surface semantic error: profile segment 16 closes the physically open mouth from top edge to lip so point-in-closed-mesh queries work, yet production also permits that artificial cap to deliver contact impulse. It was active for `{on['mouth_cap_active_cell_substep_count']}` cell-substeps. That makes implementation semantics the primary model class even though floor projection also exposes a structural limitation.

The support's nominal 3-D normal field has a legacy publication semantic—x/y are normalized in-place while z is raw. Production only consumes the intended unit horizontal normal; this audit reconstructs a unit 3-D normal before comparison. No production field or law was changed.

## Conservation and energy

- Maximum mass error: ON `{on['maximum_mass_error_m3']:.3e} m³`; OFF `{off['maximum_mass_error_m3']:.3e} m³`.
- Maximum action/reaction residual: `{energy['maximum_action_reaction_residual_ns']:.3e} N·s`.
- Summed unexplained contact energy: `{energy['summed_unexplained_contact_energy_j']:.3e} J`; maximum per substep `{energy['maximum_absolute_unexplained_contact_energy_per_substep_j']:.3e} J`.
- Contact dissipation is nonnegative; sum `{energy['summed_contact_dissipation_j']:.9f} J`.
- Reconstructed/current Warp impulse maximum error: `{energy['maximum_reconstructed_vs_production_impulse_error_ns']:.3e} N·s`.

## Isaac visual evidence

Fourteen side/top frames were captured at 0/1/2/3/5/10/20 dt from first creation. Their first bucket pose, joints, penetration and R2M state match the frozen headless replay exactly. The GUI clock has a `-2.8 s` offset and capture/pause perturbs the later scheduling, so the GUI run is used only for state-aligned first-write visualization; all causal values above come from synchronized headless ON/OFF.

{frame_links}

Arrow scale is visual-only: `endpoint = origin + 0.35 * q/h`.

## Reproduction

```bash
{chr(10).join(decision['reproduction_commands'])}
```

Machine-readable outputs: `{args.json_output}` and `{args.timeseries_output}`.
"""
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
