#!/usr/bin/env python3
"""Validate saved Phase-B--E acceptance evidence without launching Isaac.

The validator is intentionally independent of Isaac Sim, NumPy and project
imports.  It only reads frozen evidence/configuration files and writes one
machine-readable validation report.  In particular, it verifies NPZ/NPY
headers itself and independently integrates the real Phase-E ``H_initial``
field with the authoritative ``triangle_a_c`` topology.
"""

from __future__ import annotations

import argparse
import array
import ast
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, BinaryIO
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "outputs" / "phase_bcde_acceptance_validation.json"
)
SLOPES = (0, 10, 20)

FULL_STEP_FIELDS = {
    "timestamps_s",
    "phase_code",
    "normalized_command",
    "root_position_world_m",
    "root_orientation_xyzw",
    "root_linear_velocity_world_m_s",
    "root_angular_velocity_world_rad_s",
    "root_roll_pitch_yaw_rad",
    "joint_velocity_rad_s",
    "applied_effort_command_nm",
    "measured_solver_joint_effort_nm",
    "joint_target_velocity_rad_s",
    "joint_target_position_rad",
    "wheel_slip_ratio",
    "wheel_normal_load_n",
    "wheel_tangential_load_n",
    "wheel_contact_point_world_m",
    "wheel_body_center_world_m",
    "wheel_surface_to_plane_clearance_m",
    "power_cap_basis_code",
    "wheel_power_min_guard_rad_s",
    "power_cap_input_full_joint_velocity_rad_s",
    "wheel_power_cap_previous_measured_omega_rad_s",
    "wheel_power_cap_target_omega_rad_s",
    "wheel_power_cap_worst_case_omega_rad_s",
    "wheel_power_cap_protected_denominator_rad_s",
    "wheel_installed_effort_limit_nm",
    "wheel_command_envelope_power_bound_w",
    "wheel_post_step_measured_omega_rad_s",
    "wheel_installed_effort_post_step_measured_speed_power_bound_w",
}

POWER_CAP_FIELDS = {
    "power_cap_basis_code",
    "wheel_power_min_guard_rad_s",
    "power_cap_input_full_joint_velocity_rad_s",
    "wheel_power_cap_previous_measured_omega_rad_s",
    "wheel_power_cap_target_omega_rad_s",
    "wheel_power_cap_worst_case_omega_rad_s",
    "wheel_power_cap_protected_denominator_rad_s",
    "wheel_installed_effort_limit_nm",
    "wheel_command_envelope_power_bound_w",
    "wheel_post_step_measured_omega_rad_s",
    "wheel_installed_effort_post_step_measured_speed_power_bound_w",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_python_tree(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise ValueError(f"no Python files below {root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    return _finite(left) and _finite(right) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=atol
    )


def _read_npy_header(stream: BinaryIO) -> dict[str, Any]:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("missing NPY magic")
    version_bytes = stream.read(2)
    if len(version_bytes) != 2:
        raise ValueError("truncated NPY version")
    version = tuple(version_bytes)
    if version == (1, 0):
        length_bytes = stream.read(2)
        if len(length_bytes) != 2:
            raise ValueError("truncated NPY v1 header length")
        header_length = struct.unpack("<H", length_bytes)[0]
        encoding = "latin1"
    elif version in {(2, 0), (3, 0)}:
        length_bytes = stream.read(4)
        if len(length_bytes) != 4:
            raise ValueError("truncated NPY header length")
        header_length = struct.unpack("<I", length_bytes)[0]
        encoding = "utf-8" if version == (3, 0) else "latin1"
    else:
        raise ValueError(f"unsupported NPY version={version}")
    header_bytes = stream.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("truncated NPY header")
    parsed = ast.literal_eval(header_bytes.decode(encoding).strip())
    if not isinstance(parsed, dict):
        raise ValueError("NPY header is not a mapping")
    shape = parsed.get("shape")
    descr = parsed.get("descr")
    fortran_order = parsed.get("fortran_order")
    if (
        not isinstance(shape, tuple)
        or any(not isinstance(item, int) or item < 0 for item in shape)
        or not isinstance(descr, str)
        or not isinstance(fortran_order, bool)
    ):
        raise ValueError("invalid NPY shape/dtype/order metadata")
    return {
        "version": list(version),
        "shape": list(shape),
        "descr": descr,
        "fortran_order": fortran_order,
    }


def _read_float64_npy(path: Path) -> tuple[dict[str, Any], array.array[float]]:
    with path.open("rb") as stream:
        header = _read_npy_header(stream)
        payload = stream.read()
    if header["fortran_order"]:
        raise ValueError("Fortran-order authoritative height maps are unsupported")
    descr = header["descr"]
    if descr not in {"<f8", ">f8", "=f8"}:
        raise ValueError(f"authoritative height map must be float64, got {descr}")
    count = math.prod(header["shape"])
    expected_bytes = count * 8
    if len(payload) != expected_bytes:
        raise ValueError(
            f"NPY payload size mismatch: expected={expected_bytes}, got={len(payload)}"
        )
    values = array.array("d")
    values.frombytes(payload)
    file_is_big_endian = descr.startswith(">")
    host_is_big_endian = sys.byteorder == "big"
    if file_is_big_endian != host_is_big_endian and not descr.startswith("="):
        values.byteswap()
    return header, values


def _triangle_a_c_volume(
    values: array.array[float],
    *,
    ny: int,
    nx: int,
    dx_m: float,
    dy_m: float,
) -> tuple[float, float, float]:
    if len(values) != ny * nx:
        raise ValueError("height-map element count does not match shape")
    minimum = math.inf
    maximum = -math.inf
    contributions: list[float] = []
    factor = dx_m * dy_m / 6.0
    for value in values:
        if not math.isfinite(value):
            raise ValueError("height map contains NaN or Inf")
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    for y_index in range(ny - 1):
        upper = y_index * nx
        lower = upper + nx
        for x_index in range(nx - 1):
            a = values[upper + x_index]
            b = values[upper + x_index + 1]
            c = values[lower + x_index + 1]
            d = values[lower + x_index]
            contributions.append(factor * (2.0 * a + b + 2.0 * c + d))
    return math.fsum(contributions), minimum, maximum


def _npz_headers(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("NPZ contains duplicate members")
        for member in names:
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe NPZ member path: {member}")
            if not member.endswith(".npy"):
                raise ValueError(f"unexpected non-NPY NPZ member: {member}")
            key = member[:-4]
            if "/" in key or "\\" in key:
                raise ValueError(f"nested NPZ member is unsupported: {member}")
            with archive.open(member, "r") as stream:
                result[key] = _read_npy_header(stream)
    return result


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError("invalid PNG signature or IHDR")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    return width, height


class Validator:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.checks: dict[str, bool] = {}
        self.failures: list[dict[str, str]] = []
        self.evidence: dict[str, dict[str, Any]] = {
            "json_files": {},
            "config_files": {},
            "runtime_files": {},
            "screenshots": {},
            "npz_files": {},
            "heightmaps": {},
            "source_trees": {},
        }

    def check(self, name: str, passed: Any, message: str) -> bool:
        if name in self.checks:
            raise RuntimeError(f"duplicate validation check name: {name}")
        result = bool(passed)
        self.checks[name] = result
        if not result:
            self.failures.append({"check": name, "message": message})
        return result

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def resolve_reference(self, raw: Any, *, context: str) -> Path | None:
        if not isinstance(raw, str) or not raw.strip():
            self.check(f"{context}.path_declared", False, "path is absent or not text")
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            self.check(
                f"{context}.path_within_repository",
                False,
                f"path escapes repository: {resolved}",
            )
            return None
        self.check(f"{context}.path_within_repository", True, "")
        return resolved

    def record_file(
        self,
        path: Path,
        *,
        context: str,
        bucket: str,
        declared_sha256: Any = None,
    ) -> str | None:
        exists = path.is_file()
        self.check(f"{context}.exists", exists, f"missing file: {path}")
        if not exists:
            return None
        size = path.stat().st_size
        self.check(f"{context}.nonempty", size > 0, f"empty file: {path}")
        digest = _sha256(path)
        relative = self.relative(path)
        entry = self.evidence[bucket].setdefault(
            relative,
            {"sha256": digest, "size_bytes": size, "validated_by": []},
        )
        entry["validated_by"].append(context)
        if declared_sha256 is not None:
            valid_declared = (
                isinstance(declared_sha256, str)
                and len(declared_sha256) == 64
                and declared_sha256 == digest
            )
            self.check(
                f"{context}.sha256_matches",
                valid_declared,
                f"declared SHA256 does not match {relative}",
            )
        return digest

    def declared_file(
        self,
        *,
        raw_path: Any,
        declared_sha256: Any,
        expected_relative: str,
        context: str,
        bucket: str,
    ) -> Path | None:
        path = self.resolve_reference(raw_path, context=context)
        if path is None:
            return None
        expected = (self.root / expected_relative).resolve()
        self.check(
            f"{context}.expected_path",
            path == expected,
            f"expected {expected_relative}, got {self.relative(path)}",
        )
        self.record_file(
            path,
            context=context,
            bucket=bucket,
            declared_sha256=declared_sha256,
        )
        return path

    def load_json(self, relative: str, *, context: str) -> Mapping[str, Any]:
        path = (self.root / relative).resolve()
        self.record_file(path, context=context, bucket="json_files")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            self.check(
                f"{context}.valid_json",
                False,
                f"cannot parse {relative}: {type(error).__name__}: {error}",
            )
            return {}
        valid = isinstance(document, Mapping)
        self.check(f"{context}.valid_json", valid, f"{relative} root is not a mapping")
        return _mapping(document)

    def all_true(self, values: Any, *, context: str) -> None:
        mapping = _mapping(values)
        self.check(f"{context}.nonempty", bool(mapping), "check mapping is empty")
        self.check(
            f"{context}.all_true",
            bool(mapping) and all(value is True for value in mapping.values()),
            "one or more acceptance checks are not the JSON boolean true",
        )

    def phase(self, name: str, callback: Any) -> None:
        try:
            callback()
        except Exception as error:
            self.check(
                f"{name}.validation_completed",
                False,
                f"unexpected validator error: {type(error).__name__}: {error}",
            )
        else:
            self.check(f"{name}.validation_completed", True, "")


def _validate_phase_b(validator: Validator) -> None:
    document = validator.load_json(
        "outputs/phase_b_manual_runtime.json", context="phase_b.json"
    )
    validator.check(
        "phase_b.status_pass",
        document.get("status") == "PASS",
        "Phase-B status is not PASS",
    )
    validator.all_true(
        document.get("acceptance_checks"), context="phase_b.acceptance_checks"
    )
    steps = document.get("steps_completed")
    requested = document.get("steps_requested")
    validator.check(
        "phase_b.full_480_steps",
        steps == 480 and requested == 480,
        f"expected 480/480 steps, got {steps}/{requested}",
    )

    configuration = _mapping(document.get("configuration"))
    vehicle_config = validator.declared_file(
        raw_path=configuration.get("vehicle_config_relative"),
        declared_sha256=configuration.get("vehicle_config_sha256"),
        expected_relative="configs/phase_b_vehicle.yaml",
        context="phase_b.vehicle_config",
        bucket="config_files",
    )
    absolute_config = validator.resolve_reference(
        configuration.get("vehicle_config"), context="phase_b.vehicle_config_absolute"
    )
    validator.check(
        "phase_b.vehicle_config_absolute.same_file",
        vehicle_config is not None and absolute_config == vehicle_config,
        "absolute and repository-relative Phase-B config paths differ",
    )

    power = _mapping(document.get("wheel_power_limit_evidence"))
    configured_limit = power.get("configured_per_wheel_power_limit_w")
    tolerance = power.get("post_step_power_bound_tolerance_w")
    post_peak = power.get("max_post_step_measured_speed_power_bound_w")
    command_peak = power.get("max_command_envelope_power_bound_w")
    validator.check(
        "phase_b.power.measured_basis_every_step",
        power.get("measured_or_target_worst_case_frames") == steps,
        "measured-or-target basis was not used for every Phase-B step",
    )
    validator.check(
        "phase_b.power.fallback_zero",
        power.get("target_speed_fallback_frames") == 0,
        "target-only fallback was used in Phase B",
    )
    validator.check(
        "phase_b.power.minimum_guard_10_rad_s",
        _close(power.get("minimum_speed_guard_rad_s"), 10.0),
        "Phase-B evidence does not declare the frozen 10 rad/s guard",
    )
    validator.check(
        "phase_b.power.command_envelope_within_limit",
        power.get("command_envelope_bound_within_configured_limit") is True
        and _finite(command_peak)
        and _finite(configured_limit)
        and float(command_peak) <= float(configured_limit) * (1.0 + 1e-12),
        "Phase-B command power envelope exceeds its configured limit",
    )
    validator.check(
        "phase_b.power.post_step_envelope_within_limit",
        power.get("post_step_bound_within_configured_limit") is True
        and _finite(post_peak)
        and _finite(configured_limit)
        and _finite(tolerance)
        and float(post_peak) <= float(configured_limit) + float(tolerance),
        "Phase-B post-step measured-speed power bound exceeds its limit",
    )
    validator.check(
        "phase_b.power.guard_semantics_declared",
        "minimum guard" in str(power.get("semantics", "")),
        "Phase-B power semantics omit the configured minimum guard",
    )

    screenshots = _mapping(document.get("screenshots"))
    records = _sequence(screenshots.get("records"))
    validator.check(
        "phase_b.screenshots.capture_requested",
        screenshots.get("capture_requested") is True,
        "Phase-B screenshot capture was not requested",
    )
    labels = [item.get("label") for item in records if isinstance(item, Mapping)]
    validator.check(
        "phase_b.screenshots.four_required_records",
        len(records) == 4 and set(labels) == {"H0", "drive", "lift_curl", "dump_pose"},
        f"unexpected screenshot labels: {labels}",
    )
    for item in records:
        record = _mapping(item)
        label = str(record.get("label", "unknown"))
        context = f"phase_b.screenshot.{label}"
        filename = record.get("filename")
        safe_name = (
            isinstance(filename, str)
            and bool(filename)
            and Path(filename).name == filename
        )
        validator.check(
            f"{context}.safe_filename", safe_name, f"unsafe screenshot name: {filename!r}"
        )
        validator.check(
            f"{context}.record_readable",
            record.get("readable") is True,
            "runtime marked screenshot unreadable",
        )
        validator.check(
            f"{context}.authoritative_h0_visible",
            record.get("authoritative_h0_visible") is True,
            "authoritative H0 is not reported visible",
        )
        validator.check(
            f"{context}.h0_visual_no_collision",
            record.get("h0_visual_no_collision") is True,
            "H0 visual unexpectedly owns collision",
        )
        if safe_name:
            path = validator.root / "outputs" / "phase_b_screenshots" / str(filename)
            digest = validator.record_file(
                path, context=context, bucket="screenshots"
            )
            validator.check(
                f"{context}.size_matches_record",
                path.is_file()
                and record.get("size_bytes") == path.stat().st_size,
                "screenshot byte count differs from the runtime record",
            )
            if digest is not None:
                try:
                    width, height = _png_dimensions(path)
                except Exception as error:
                    validator.check(
                        f"{context}.valid_png",
                        False,
                        f"invalid PNG: {type(error).__name__}: {error}",
                    )
                else:
                    validator.check(f"{context}.valid_png", True, "")
                    validator.evidence["screenshots"][
                        validator.relative(path)
                    ].update(
                        {"label": label, "width_px": width, "height_px": height}
                    )

    h0 = _mapping(screenshots.get("authoritative_h0_visual"))
    volume = _mapping(h0.get("volume_integration"))
    validator.check(
        "phase_b.h0.authoritative_visual",
        h0.get("authoritative") is True
        and h0.get("visual_no_collision") is True,
        "Phase-B H0 is not an authoritative collision-free visual",
    )
    validator.check(
        "phase_b.h0.single_default_ground_support",
        h0.get("terrain_support_collision_source_count") == 1
        and h0.get("physical_support_role") == "default_ground_only",
        "Phase-B H0 support must be the single default ground",
    )
    validator.check(
        "phase_b.h0.shared_visual_material",
        h0.get("ground_and_h0_share_visual_material") is True,
        "ground and H0 do not share the visual material",
    )
    validator.check(
        "phase_b.h0.triangle_a_c_strict_volume",
        volume.get("integrator") == "TerrainVolumeIntegrator"
        and volume.get("topology") == "triangle_a_c"
        and volume.get("matches_dynamic_mesh_diagonal") is True
        and _finite(h0.get("volume_m3")),
        "Phase-B H0 does not use strict triangle_a_c volume evidence",
    )
    h0_source = validator.resolve_reference(
        h0.get("source_relative_to_repository"), context="phase_b.h0.source"
    )
    if h0_source is not None:
        validator.record_file(
            h0_source, context="phase_b.h0.source", bucket="heightmaps"
        )


def _validate_phase_c(validator: Validator) -> None:
    contact_hashes: list[Any] = []
    terrain_hashes: list[Any] = []
    for slope in SLOPES:
        label = f"{slope:02d}deg"
        context = f"phase_c.{label}"
        document = validator.load_json(
            f"outputs/phase_c_contact_runtime_{label}.json",
            context=f"{context}.json",
        )
        validator.check(
            f"{context}.status_pass",
            document.get("status") == "PASS",
            f"Phase-C {slope} degree status is not PASS",
        )
        slope_fixture = _mapping(document.get("slope_fixture"))
        validator.check(
            f"{context}.requested_and_measured_slope",
            _close(slope_fixture.get("requested_deg"), slope)
            and _close(slope_fixture.get("measured_mean_deg"), slope, atol=1e-9)
            and _finite(slope_fixture.get("measured_max_abs_error_deg"))
            and float(slope_fixture["measured_max_abs_error_deg"]) <= 1e-9,
            "Phase-C requested/measured slope evidence is inconsistent",
        )

        configuration = _mapping(document.get("configuration"))
        contact_hashes.append(configuration.get("contact_config_sha256"))
        terrain_hashes.append(configuration.get("terrain_config_sha256"))
        contact_path = validator.declared_file(
            raw_path=configuration.get("contact_config_relative"),
            declared_sha256=configuration.get("contact_config_sha256"),
            expected_relative="configs/phase_c_contact.yaml",
            context=f"{context}.contact_config",
            bucket="config_files",
        )
        terrain_path = validator.declared_file(
            raw_path=configuration.get("terrain_config_relative"),
            declared_sha256=configuration.get("terrain_config_sha256"),
            expected_relative="configs/project_25m.yaml",
            context=f"{context}.terrain_config",
            bucket="config_files",
        )
        absolute_contact = validator.resolve_reference(
            configuration.get("contact_config"),
            context=f"{context}.contact_config_absolute",
        )
        absolute_terrain = validator.resolve_reference(
            configuration.get("terrain_config"),
            context=f"{context}.terrain_config_absolute",
        )
        validator.check(
            f"{context}.absolute_configs_match",
            contact_path is not None
            and terrain_path is not None
            and absolute_contact == contact_path
            and absolute_terrain == terrain_path,
            "Phase-C absolute and relative configuration evidence differs",
        )
        effective = _mapping(configuration.get("effective"))
        validator.check(
            f"{context}.three_configured_slopes",
            _sequence(effective.get("slope_degrees")) == [0.0, 10.0, 20.0],
            "Phase-C config does not contain the complete 0/10/20 slope set",
        )

        filtering = _mapping(document.get("collision_filtering"))
        membership = _mapping(filtering.get("membership_validation"))
        validator.check(
            f"{context}.collision_filtering_valid",
            filtering.get("bucket_terrain_filtered_bidirectional") is True
            and filtering.get("visual_prim_in_any_group") is False
            and membership.get("valid") is True
            and _sequence(membership.get("errors")) == [],
            "Phase-C collision filtering or membership is invalid",
        )
        mesh = _mapping(document.get("contact_mesh"))
        validator.check(
            f"{context}.hidden_separate_static_contact_mesh",
            mesh.get("collision_api") is True
            and mesh.get("mesh_collision_api") is True
            and mesh.get("rigid_body_api") is False
            and mesh.get("separate_from_visual") is True
            and mesh.get("visible") is False
            and mesh.get("visual_collision_api") is False,
            "Phase-C contact mesh is not hidden, separate, static and colliding",
        )
        capability = _mapping(document.get("heightfield_capability"))
        validator.check(
            f"{context}.triangle_mesh_backend_honest",
            capability.get("selected_backend") == "TriangleMeshContactBackend"
            and capability.get("public_authoring_verified") is False
            and capability.get("public_dynamic_update_verified") is False,
            "Phase-C heightfield/triangle-mesh capability claim is inconsistent",
        )
        handoff = _mapping(document.get("phase_d_handoff"))
        validator.check(
            f"{context}.no_pose_setter",
            handoff.get("pose_setter_count") == 0,
            "Phase-C handoff reports a pose setter",
        )

    validator.check(
        "phase_c.all_three_contact_config_hashes_same",
        len(contact_hashes) == 3
        and all(isinstance(value, str) and value == contact_hashes[0] for value in contact_hashes),
        "the three Phase-C records do not share contact-config evidence",
    )
    validator.check(
        "phase_c.all_three_terrain_config_hashes_same",
        len(terrain_hashes) == 3
        and all(isinstance(value, str) and value == terrain_hashes[0] for value in terrain_hashes),
        "the three Phase-C records do not share terrain-config evidence",
    )


def _validate_npz(
    validator: Validator,
    *,
    persistence: Mapping[str, Any],
    steps: Any,
    slope: int,
) -> None:
    context = f"phase_cd.{slope:02d}deg.full_npz"
    path = validator.resolve_reference(persistence.get("path"), context=context)
    validator.check(
        f"{context}.metadata",
        persistence.get("format") == "compressed_npz"
        and persistence.get("schema_version")
        == "isaac-bulk-phase-d-full-telemetry/v2"
        and persistence.get("all_required_fields_present") is True
        and persistence.get("frame_count") == steps,
        "full-step persistence metadata is incomplete or inconsistent",
    )
    expected = (
        validator.root
        / "outputs"
        / f"phase_cd_slope_{slope:02d}deg_full_telemetry.npz"
    ).resolve()
    validator.check(
        f"{context}.expected_path",
        path == expected,
        f"unexpected full telemetry path: {path}",
    )
    declared_power_fields = set(
        _sequence(persistence.get("power_cap_fields_recorded_every_step"))
    )
    validator.check(
        f"{context}.power_fields_declared",
        POWER_CAP_FIELDS.issubset(declared_power_fields),
        "persistence metadata omits required per-step power-cap fields",
    )
    if path is None:
        return
    digest = validator.record_file(
        path,
        context=context,
        bucket="npz_files",
        declared_sha256=persistence.get("sha256"),
    )
    if digest is None:
        return
    try:
        headers = _npz_headers(path)
    except Exception as error:
        validator.check(
            f"{context}.readable_safe_npz",
            False,
            f"cannot inspect NPZ: {type(error).__name__}: {error}",
        )
        return
    validator.check(f"{context}.readable_safe_npz", True, "")
    validator.check(
        f"{context}.all_full_step_fields_present",
        FULL_STEP_FIELDS.issubset(headers),
        f"missing full-step arrays: {sorted(FULL_STEP_FIELDS - set(headers))}",
    )
    malformed = {
        name: header.get("shape")
        for name, header in headers.items()
        if name in FULL_STEP_FIELDS
        and (
            not header.get("shape")
            or not isinstance(steps, int)
            or header["shape"][0] != steps
            or header.get("fortran_order") is not False
            or "O" in str(header.get("descr"))
        )
    }
    validator.check(
        f"{context}.every_required_array_has_full_frame_count",
        not malformed and FULL_STEP_FIELDS.issubset(headers),
        f"malformed or short full-step arrays: {malformed}",
    )
    relative = validator.relative(path)
    validator.evidence["npz_files"][relative].update(
        {
            "frame_count": steps,
            "array_count": len(headers),
            "required_full_step_field_count": len(FULL_STEP_FIELDS),
        }
    )


def _validate_phase_cd(validator: Validator) -> None:
    runs: dict[int, Mapping[str, Any]] = {}
    identities: list[Mapping[str, Any]] = []
    fingerprints: list[Any] = []
    run_triplets: list[tuple[Any, Any, Any]] = []
    for slope in SLOPES:
        label = f"{slope:02d}deg"
        context = f"phase_cd.{label}"
        document = validator.load_json(
            f"outputs/phase_cd_slope_{label}.json", context=f"{context}.json"
        )
        runs[slope] = document
        validator.check(
            f"{context}.status_pass",
            document.get("status") == "PASS",
            f"C/D {slope} degree status is not PASS",
        )
        validator.all_true(
            document.get("acceptance_checks"),
            context=f"{context}.acceptance_checks",
        )
        validator.check(
            f"{context}.failed_checks_empty",
            _sequence(document.get("failed_checks")) == [],
            "C/D failed_checks is not empty",
        )

        run_config = _mapping(document.get("run_config"))
        steps = run_config.get("steps")
        validator.check(
            f"{context}.run_shape",
            _close(run_config.get("slope_deg"), slope)
            and steps == 420
            and isinstance(run_config.get("settle_steps"), int)
            and isinstance(run_config.get("brake_steps"), int)
            and _finite(run_config.get("physics_dt_s")),
            "C/D run config is not the expected 420-step slope trial",
        )
        run_triplets.append(
            (run_config.get("throttle"), steps, run_config.get("physics_dt_s"))
        )
        identity = _mapping(run_config.get("scenario_identity"))
        identities.append(identity)
        fingerprint = run_config.get("scenario_fingerprint_sha256")
        fingerprints.append(fingerprint)
        validator.check(
            f"{context}.scenario_fingerprint_recomputed",
            isinstance(fingerprint, str)
            and len(fingerprint) == 64
            and bool(identity)
            and _sha256_json(identity) == fingerprint,
            "scenario fingerprint does not match canonical scenario identity",
        )

        config_files = (
            (
                "loader_usd",
                "loader_usd_sha256",
                "isaac_loader/wheel_loader.usd",
                "runtime_files",
            ),
            (
                "vehicle_config",
                "vehicle_config_sha256",
                "configs/phase_b_vehicle.yaml",
                "config_files",
            ),
            (
                "contact_config",
                "contact_config_sha256",
                "configs/phase_c_contact.yaml",
                "config_files",
            ),
            (
                "telemetry_config",
                "telemetry_config_sha256",
                "configs/phase_d_telemetry.yaml",
                "config_files",
            ),
        )
        for path_key, hash_key, expected, bucket in config_files:
            validator.declared_file(
                raw_path=run_config.get(path_key),
                declared_sha256=identity.get(hash_key),
                expected_relative=expected,
                context=f"{context}.{path_key}",
                bucket=bucket,
            )
        validator.declared_file(
            raw_path="isaac_loader/phase_cd_slope_drive_runtime.py",
            declared_sha256=identity.get("runtime_script_sha256"),
            expected_relative="isaac_loader/phase_cd_slope_drive_runtime.py",
            context=f"{context}.runtime_script",
            bucket="runtime_files",
        )
        validator.declared_file(
            raw_path="isaac_loader/phase_c_contact_runtime.py",
            declared_sha256=identity.get("phase_c_runtime_source_sha256"),
            expected_relative="isaac_loader/phase_c_contact_runtime.py",
            context=f"{context}.phase_c_runtime_script",
            bucket="runtime_files",
        )
        pipeline_root = validator.root / "src" / "isaac_bulk_pipeline"
        pipeline_digest = _sha256_python_tree(pipeline_root)
        validator.check(
            f"{context}.pipeline_python_source_tree.sha256_matches",
            identity.get("pipeline_python_source_tree_sha256")
            == pipeline_digest,
            "pipeline Python source-tree hash differs from scenario identity",
        )
        validator.evidence["source_trees"][
            "src/isaac_bulk_pipeline"
        ] = {
            "sha256": pipeline_digest,
            "file_count": len(tuple(pipeline_root.rglob("*.py"))),
        }
        validator.check(
            f"{context}.frozen_power_contract_in_identity",
            _close(identity.get("wheel_power_limit_w"), 160_000.0)
            and _close(identity.get("wheel_power_reference_rad_s"), 1.0)
            and _close(identity.get("wheel_power_min_guard_rad_s"), 10.0)
            and _close(
                identity.get("wheel_power_discrete_safety_factor"), 1.15
            ),
            "scenario identity omits or changes the frozen power contract",
        )

        power = _mapping(
            _mapping(document.get("phase_b_vehicle_control")).get(
                "wheel_power_limit_evidence"
            )
        )
        limit = power.get("configured_per_wheel_power_limit_w")
        post_peak = power.get(
            "max_installed_effort_post_step_measured_speed_power_bound_w"
        )
        tolerance = power.get("post_step_power_bound_tolerance_w")
        basis_counts = _mapping(power.get("basis_counts"))
        validator.check(
            f"{context}.power.measured_basis_every_step",
            power.get("measured_or_target_worst_case_frames") == steps
            and basis_counts.get("MEASURED_OR_TARGET_WORST_CASE") == steps,
            "C/D did not use measured-or-target power basis every step",
        )
        validator.check(
            f"{context}.power.target_fallback_zero",
            power.get("target_speed_fallback_frames") == 0
            and basis_counts.get("TARGET_JOINT_VELOCITY_FALLBACK", 0) == 0,
            "C/D used the target-only power fallback",
        )
        validator.check(
            f"{context}.power.minimum_guard_10_rad_s",
            _close(power.get("minimum_speed_guard_rad_s"), 10.0),
            "C/D evidence omits the frozen 10 rad/s guard",
        )
        validator.check(
            f"{context}.power.command_envelope_within_limit",
            power.get("command_envelope_bound_within_configured_limit") is True,
            "C/D command envelope failed",
        )
        validator.check(
            f"{context}.power.post_step_envelope_within_limit",
            power.get("post_step_bound_within_configured_limit") is True
            and _finite(limit)
            and _finite(post_peak)
            and _finite(tolerance)
            and float(post_peak) <= float(limit) + float(tolerance),
            "C/D post-step measured-speed power bound exceeds the configured limit",
        )

        phase_c = _mapping(document.get("phase_c_contact"))
        membership = _mapping(phase_c.get("membership_validation"))
        validator.check(
            f"{context}.phase_c_contact_contract",
            phase_c.get("backend") == "TriangleMeshContactBackend"
            and phase_c.get("visible") is False
            and phase_c.get("static") is True
            and phase_c.get("visual_has_collision_api") is False
            and membership.get("valid") is True
            and _sequence(membership.get("errors")) == [],
            "C/D embedded Phase-C contact evidence is invalid",
        )
        phase_d = _mapping(document.get("phase_d_telemetry"))
        validator.check(
            f"{context}.public_contact_load_and_points",
            _mapping(phase_d.get("contact_force")).get("available") is True
            and _mapping(phase_d.get("contact_points")).get("available") is True,
            "public wheel contact load or point evidence is unavailable",
        )
        _validate_npz(
            validator,
            persistence=_mapping(phase_d.get("full_step_persistence")),
            steps=steps,
            slope=slope,
        )

    validator.check(
        "phase_cd.same_scenario_identity_all_slopes",
        len(identities) == 3 and bool(identities[0]) and identities.count(identities[0]) == 3,
        "the three C/D runs do not have the same full scenario identity",
    )
    validator.check(
        "phase_cd.same_scenario_fingerprint_all_slopes",
        len(fingerprints) == 3
        and isinstance(fingerprints[0], str)
        and fingerprints.count(fingerprints[0]) == 3,
        "the three C/D runs do not share one scenario fingerprint",
    )
    validator.check(
        "phase_cd.same_throttle_steps_dt_all_slopes",
        len(run_triplets) == 3 and run_triplets.count(run_triplets[0]) == 3,
        "throttle, step count or dt differs across slope runs",
    )

    summary = validator.load_json(
        "outputs/phase_cd_slope_summary.json", context="phase_cd.summary.json"
    )
    validator.check(
        "phase_cd.summary.status_pass",
        summary.get("status") == "PASS",
        "C/D aggregate summary status is not PASS",
    )
    validator.all_true(
        summary.get("aggregate_checks"), context="phase_cd.summary.aggregate_checks"
    )
    validator.check(
        "phase_cd.summary.complete_slopes",
        _sequence(summary.get("required_slopes_deg")) == [0, 10, 20]
        and _sequence(summary.get("completed_slopes_deg")) == [0, 10, 20],
        "C/D summary does not contain all required slopes",
    )
    validator.check(
        "phase_cd.summary.same_fingerprint",
        summary.get("same_full_physics_scenario_fingerprint") is True
        and _sequence(summary.get("scenario_fingerprints_sha256")) == fingerprints,
        "C/D summary fingerprint evidence disagrees with run JSONs",
    )
    validator.check(
        "phase_cd.summary.same_run_shape",
        summary.get("same_throttle_steps_dt_across_runs") is True,
        "C/D summary reports different throttle/steps/dt",
    )
    slope_effect_checks = _mapping(summary.get("slope_effect_checks"))
    validator.check(
        "phase_cd.summary.measurable_slope_effect",
        summary.get("measurable_slope_effect") is True
        and sum(value is True for value in slope_effect_checks.values()) >= 2,
        "fewer than two telemetry metrics show the required slope effect",
    )
    run_files = _mapping(summary.get("run_files"))
    for slope in SLOPES:
        context = f"phase_cd.summary.run_file_{slope:02d}deg"
        path = validator.resolve_reference(run_files.get(str(slope)), context=context)
        expected = (
            validator.root / "outputs" / f"phase_cd_slope_{slope:02d}deg.json"
        ).resolve()
        validator.check(
            f"{context}.expected_path",
            path == expected,
            f"summary run file for slope {slope} is wrong",
        )


def _validate_phase_e(validator: Validator, phase_b: Mapping[str, Any]) -> None:
    document = validator.load_json(
        "outputs/phase_e_acceptance.json", context="phase_e.json"
    )
    validator.check(
        "phase_e.status_pass",
        document.get("status") == "PASS",
        "Phase-E status is not PASS",
    )
    validator.all_true(document.get("checks"), context="phase_e.checks")

    configuration = _mapping(document.get("configuration"))
    phase_e_config = _mapping(configuration.get("phase_e"))
    terrain_config = _mapping(configuration.get("terrain"))
    validator.declared_file(
        raw_path=phase_e_config.get("path"),
        declared_sha256=phase_e_config.get("sha256"),
        expected_relative="configs/phase_e_bulk_state.yaml",
        context="phase_e.phase_e_config",
        bucket="config_files",
    )
    validator.declared_file(
        raw_path=terrain_config.get("path"),
        declared_sha256=terrain_config.get("sha256"),
        expected_relative="configs/project_25m.yaml",
        context="phase_e.terrain_config",
        bucket="config_files",
    )
    effective = _mapping(configuration.get("effective_values"))
    abs_tolerance = effective.get("absolute_balance_tolerance_m3")
    rel_tolerance = effective.get("relative_balance_tolerance")
    validator.check(
        "phase_e.authoritative_grid_and_topology",
        effective.get("surface_topology") == "triangle_a_c"
        and effective.get("nx") == 701
        and effective.get("ny") == 701
        and _close(effective.get("dx_m"), 0.05)
        and _close(effective.get("dy_m"), 0.05)
        and _close(effective.get("span_x_m"), 35.0)
        and _close(effective.get("span_y_m"), 35.0),
        "Phase-E effective grid/topology is not frozen at 701x701, 0.05 m, triangle_a_c",
    )
    validator.check(
        "phase_e.finite_positive_balance_tolerances",
        _finite(abs_tolerance)
        and float(abs_tolerance) > 0.0
        and _finite(rel_tolerance)
        and float(rel_tolerance) > 0.0,
        "Phase-E conservation tolerances are not finite and positive",
    )

    reference = _mapping(document.get("reference"))
    validator.check(
        "phase_e.reference_metadata",
        reference.get("topology") == "triangle_a_c"
        and _sequence(reference.get("shape")) == [701, 701]
        and reference.get("dtype") == "float64"
        and _close(reference.get("dx_m"), 0.05)
        and _close(reference.get("dy_m"), 0.05),
        "Phase-E real H0 reference metadata is inconsistent",
    )
    strict = reference.get("strict_volume_m3")
    independent = reference.get("independent_formula_volume_m3")
    reported_difference = reference.get("absolute_error_m3")
    validator.check(
        "phase_e.strict_matches_reported_independent_formula",
        _finite(strict)
        and float(strict) >= 0.0
        and _finite(independent)
        and _finite(reported_difference)
        and math.isclose(
            abs(float(strict) - float(independent)),
            float(reported_difference),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and float(reported_difference) <= 1e-10,
        "strict and independently reported triangle volumes disagree",
    )

    heightmap = validator.resolve_reference(
        reference.get("path"), context="phase_e.reference_heightmap"
    )
    if heightmap is not None:
        digest = validator.record_file(
            heightmap,
            context="phase_e.reference_heightmap",
            bucket="heightmaps",
        )
        if digest is not None:
            try:
                header, values = _read_float64_npy(heightmap)
                recomputed, minimum, maximum = _triangle_a_c_volume(
                    values,
                    ny=701,
                    nx=701,
                    dx_m=0.05,
                    dy_m=0.05,
                )
            except Exception as error:
                validator.check(
                    "phase_e.reference_heightmap.independent_read_and_integrate",
                    False,
                    f"cannot independently integrate H0: {type(error).__name__}: {error}",
                )
            else:
                validator.check(
                    "phase_e.reference_heightmap.independent_read_and_integrate",
                    header.get("shape") == [701, 701]
                    and header.get("descr") == "<f8"
                    and header.get("fortran_order") is False
                    and minimum >= 0.0
                    and math.isclose(
                        recomputed,
                        float(strict),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ),
                    "independent stdlib triangle_a_c integration differs from Phase-E strict volume",
                )
                relative = validator.relative(heightmap)
                validator.evidence["heightmaps"][relative].update(
                    {
                        "shape_yx": header["shape"],
                        "dtype": header["descr"],
                        "minimum_height_m": minimum,
                        "maximum_height_m": maximum,
                        "independent_triangle_a_c_volume_m3": recomputed,
                    }
                )

    b_h0 = _mapping(_mapping(phase_b.get("screenshots")).get("authoritative_h0_visual"))
    validator.check(
        "phase_e.strict_volume_matches_phase_b_h0",
        _finite(strict)
        and _finite(b_h0.get("volume_m3"))
        and math.isclose(
            float(strict), float(b_h0["volume_m3"]), rel_tol=0.0, abs_tol=1e-9
        ),
        "Phase-E strict H0 volume differs from authoritative Phase-B H0 evidence",
    )

    closed = _mapping(document.get("closed_balance"))
    open_balance = _mapping(document.get("open_balance"))
    tolerances_valid = (
        _finite(abs_tolerance)
        and _finite(rel_tolerance)
        and float(abs_tolerance) > 0.0
        and float(rel_tolerance) > 0.0
    )
    validator.check(
        "phase_e.closed_conservation_strict",
        tolerances_valid
        and _finite(closed.get("absolute_error_m3"))
        and _finite(closed.get("relative_error"))
        and float(closed["absolute_error_m3"]) <= float(abs_tolerance)
        and float(closed["relative_error"]) <= float(rel_tolerance)
        and _close(_mapping(closed.get("reservoirs")).get("outflow_m3"), 0.0),
        "closed-boundary Phase-E conservation exceeds configured tolerances",
    )
    validator.check(
        "phase_e.open_conservation_strict",
        tolerances_valid
        and _finite(open_balance.get("absolute_error_m3"))
        and _finite(open_balance.get("relative_error"))
        and float(open_balance["absolute_error_m3"]) <= float(abs_tolerance)
        and float(open_balance["relative_error"]) <= float(rel_tolerance)
        and _finite(open_balance.get("outflow_m3"))
        and float(open_balance["outflow_m3"]) > 0.0,
        "open-boundary Phase-E balance/outflow exceeds configured tolerances",
    )
    reservoirs = _mapping(closed.get("reservoirs"))
    validator.check(
        "phase_e.closed_reservoirs_finite_nonnegative",
        bool(reservoirs)
        and all(_finite(value) and float(value) >= 0.0 for value in reservoirs.values()),
        "closed-boundary reservoirs contain a negative or non-finite value",
    )

    profile = _mapping(document.get("volume_profile"))
    validator.check(
        "phase_e.required_resolution_profile_present",
        {"128", "256", "512", "701"}.issubset(profile),
        "Phase-E volume profile omits a required resolution",
    )
    for resolution in (128, 256, 512, 701):
        item = _mapping(profile.get(str(resolution)))
        validator.check(
            f"phase_e.volume_profile.{resolution}.strict",
            _finite(item.get("volume_m3"))
            and _finite(item.get("expected_volume_m3"))
            and _finite(item.get("absolute_error_m3"))
            and math.isclose(
                abs(float(item["volume_m3"]) - float(item["expected_volume_m3"])),
                float(item["absolute_error_m3"]),
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            and float(item["absolute_error_m3"]) <= 1e-8,
            f"Phase-E {resolution} volume profile is inconsistent",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="validation JSON path (must remain inside this repository)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    output = output.resolve()
    try:
        output.relative_to(REPOSITORY_ROOT)
    except ValueError:
        raise SystemExit(f"output must remain inside repository: {output}")

    validator = Validator(REPOSITORY_ROOT)
    phase_b_document: Mapping[str, Any] = {}

    def phase_b() -> None:
        nonlocal phase_b_document
        _validate_phase_b(validator)
        phase_b_document = _mapping(
            json.loads(
                (REPOSITORY_ROOT / "outputs/phase_b_manual_runtime.json").read_text(
                    encoding="utf-8"
                )
            )
        )

    validator.phase("phase_b", phase_b)
    validator.phase("phase_c", lambda: _validate_phase_c(validator))
    validator.phase("phase_cd", lambda: _validate_phase_cd(validator))
    validator.phase(
        "phase_e", lambda: _validate_phase_e(validator, phase_b_document)
    )

    status = "PASS" if validator.checks and all(validator.checks.values()) else "FAIL"
    report = {
        "schema_version": "isaac-bulk-phase-bcde-acceptance-validation/v1",
        "status": status,
        "read_only_validation": True,
        "repository_root": str(REPOSITORY_ROOT),
        "check_count": len(validator.checks),
        "passed_check_count": sum(validator.checks.values()),
        "failed_check_count": len(validator.failures),
        "checks": validator.checks,
        "failures": validator.failures,
        "evidence": validator.evidence,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"PHASE_BCDE_ACCEPTANCE_VALIDATION_{status}: {output}")
    if validator.failures:
        for failure in validator.failures:
            print(f"- {failure['check']}: {failure['message']}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
