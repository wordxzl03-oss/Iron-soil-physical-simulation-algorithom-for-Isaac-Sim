#!/usr/bin/env python3
"""Collect a privacy-bounded Phase A runtime environment manifest.

The manifest intentionally records no environment variables, hostname, user name,
home directory, executable path, GPU UUID, or absolute repository/Isaac paths.
It is separate from the six-scoop baseline manifest, which owns artifact hashes and
episode invariants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


SCHEMA_VERSION = "phase-a-runtime-environment-v1"
CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_exact_text(path: Path, *, source: str) -> dict[str, object]:
    """Return exact UTF-8 content and its digest without exposing ``path``."""

    if not path.is_file():
        return {
            "status": "unavailable",
            "source": source,
            "reason": "file_not_found",
        }
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {
            "status": "unavailable",
            "source": source,
            "reason": "file_not_readable_as_utf8",
        }
    return {
        "status": "available",
        "source": source,
        "raw_text": raw_text,
        "sha256": _sha256_text(raw_text),
    }


def _discover_isaac_root(explicit_root: Path | None) -> tuple[Path | None, str]:
    if explicit_root is not None:
        return explicit_root.expanduser().resolve(), "cli_argument"

    candidates = (
        (Path.home() / "isaacsim", "home_isaacsim"),
        (Path("/opt/isaacsim"), "opt_isaacsim"),
    )
    for candidate, locator in candidates:
        if (candidate / "VERSION").is_file() or (
            candidate / "PACKAGE-INFO.yaml"
        ).is_file():
            return candidate.resolve(), locator
    return None, "not_found"


def collect_isaac_sim(explicit_root: Path | None = None) -> dict[str, object]:
    """Collect exact Isaac and Kit package files, if a supported root exists."""

    root, locator = _discover_isaac_root(explicit_root)
    if root is None:
        return {
            "status": "unavailable",
            "root_locator": locator,
            "reason": "isaac_root_not_found",
            "files": {},
        }

    files = {
        "VERSION": _read_exact_text(root / "VERSION", source="VERSION"),
        "PACKAGE-INFO.yaml": _read_exact_text(
            root / "PACKAGE-INFO.yaml", source="PACKAGE-INFO.yaml"
        ),
        "kit/PACKAGE-INFO.yaml": _read_exact_text(
            root / "kit" / "PACKAGE-INFO.yaml",
            source="kit/PACKAGE-INFO.yaml",
        ),
    }
    available = any(item["status"] == "available" for item in files.values())
    return {
        "status": "available" if available else "unavailable",
        "root_locator": locator,
        "reason": None if available else "package_files_not_found",
        "files": files,
    }


def _run_command(
    command: Sequence[str], timeout_s: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def collect_gpu(
    *,
    command_runner: CommandRunner = _run_command,
    command_available: bool | None = None,
) -> dict[str, object]:
    """Collect non-identifying NVIDIA GPU and driver facts via ``nvidia-smi``."""

    available = (
        shutil.which("nvidia-smi") is not None
        if command_available is None
        else command_available
    )
    if not available:
        return {
            "status": "unavailable",
            "collector": "nvidia-smi",
            "reason": "command_not_found",
            "gpus": [],
        }

    command = (
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = command_runner(command, 5.0)
    except subprocess.TimeoutExpired:
        return {
            "status": "unavailable",
            "collector": "nvidia-smi",
            "reason": "command_timeout",
            "gpus": [],
        }
    except OSError:
        return {
            "status": "unavailable",
            "collector": "nvidia-smi",
            "reason": "command_execution_failed",
            "gpus": [],
        }
    if completed.returncode != 0:
        return {
            "status": "unavailable",
            "collector": "nvidia-smi",
            "reason": "command_failed",
            "returncode": int(completed.returncode),
            "gpus": [],
        }

    gpus: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            return {
                "status": "unavailable",
                "collector": "nvidia-smi",
                "reason": "unexpected_output_schema",
                "gpus": [],
            }
        name, driver_version, memory_total_mib = fields
        try:
            memory_value: int | float = int(memory_total_mib)
        except ValueError:
            try:
                memory_value = float(memory_total_mib)
            except ValueError:
                return {
                    "status": "unavailable",
                    "collector": "nvidia-smi",
                    "reason": "invalid_memory_value",
                    "gpus": [],
                }
        gpus.append(
            {
                "name": name,
                "driver_version": driver_version,
                "memory_total_mib": memory_value,
            }
        )
    if not gpus:
        return {
            "status": "unavailable",
            "collector": "nvidia-smi",
            "reason": "no_gpu_rows",
            "gpus": [],
        }
    return {
        "status": "available",
        "collector": "nvidia-smi",
        "reason": None,
        "gpus": gpus,
    }


def _revision_from_git_metadata(repo_root: Path) -> tuple[str | None, str | None]:
    """Resolve a simple loose Git HEAD without requiring the git executable."""

    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None, None
    try:
        head = head_path.read_text(encoding="ascii").strip()
        if head.startswith("ref: "):
            ref_name = head[5:].strip()
            ref_path = git_dir.joinpath(*ref_name.split("/"))
            if not ref_path.is_file():
                return None, ref_name
            revision = ref_path.read_text(encoding="ascii").strip()
            return (revision if revision else None), ref_name
        return (head if head else None), None
    except (OSError, UnicodeError):
        return None, None


def collect_repo_revision(
    repo_root: Path,
    *,
    command_runner: CommandRunner = _run_command,
    git_available: bool | None = None,
) -> dict[str, object]:
    """Record an exact Git revision or an explicit unavailable state."""

    metadata_revision, ref_name = _revision_from_git_metadata(repo_root)
    available = shutil.which("git") is not None if git_available is None else git_available
    if available:
        try:
            completed = command_runner(
                ("git", "-C", str(repo_root), "rev-parse", "HEAD"), 5.0
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            revision = completed.stdout.strip()
            if revision:
                return {
                    "status": "available",
                    "vcs": "git",
                    "revision": revision,
                    "reference": ref_name,
                    "method": "git_command",
                }

    if metadata_revision is not None:
        return {
            "status": "available",
            "vcs": "git",
            "revision": metadata_revision,
            "reference": ref_name,
            "method": "git_metadata",
        }
    reason = "git_command_not_found_and_metadata_unavailable" if not available else "git_revision_unavailable"
    return {
        "status": "unavailable",
        "vcs": "git",
        "revision": None,
        "reference": ref_name,
        "reason": reason,
    }


def collect_manifest(
    *,
    repo_root: Path,
    isaac_root: Path | None = None,
    generated_at_utc: str | None = None,
    command_runner: CommandRunner = _run_command,
    nvidia_smi_available: bool | None = None,
    git_available: bool | None = None,
) -> dict[str, object]:
    """Build the complete manifest without reading process environment variables."""

    try:
        import numpy as np

        numpy_record: dict[str, object] = {
            "status": "available",
            "version": np.__version__,
        }
    except ImportError:
        numpy_record = {
            "status": "unavailable",
            "version": None,
            "reason": "import_failed",
        }

    libc_name, libc_version = platform.libc_ver()
    try:
        os_release = platform.freedesktop_os_release()
        distribution: dict[str, object] = {
            "status": "available",
            "id": os_release.get("ID"),
            "name": os_release.get("NAME"),
            "version_id": os_release.get("VERSION_ID"),
            "version": os_release.get("VERSION"),
            "pretty_name": os_release.get("PRETTY_NAME"),
        }
    except OSError:
        distribution = {
            "status": "unavailable",
            "reason": "os_release_not_available",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or _utc_timestamp(),
        "privacy": {
            "environment_variables_collected": False,
            "hostname_collected": False,
            "username_collected": False,
            "absolute_paths_collected": False,
            "gpu_uuid_or_serial_collected": False,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "distribution": distribution,
            "libc": {
                "name": libc_name or None,
                "version": libc_version or None,
            },
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
            "cache_tag": sys.implementation.cache_tag,
        },
        "numpy": numpy_record,
        "isaac_sim": collect_isaac_sim(isaac_root),
        "gpu": collect_gpu(
            command_runner=command_runner,
            command_available=nvidia_smi_available,
        ),
        "repository": collect_repo_revision(
            repo_root.resolve(),
            command_runner=command_runner,
            git_available=git_available,
        ),
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "outputs" / "phase_a_environment.json",
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root)
    parser.add_argument("--isaac-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = collect_manifest(
        repo_root=args.repo_root,
        isaac_root=args.isaac_root,
    )
    write_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "status": "PHASE_A_ENVIRONMENT_WRITTEN",
                "schema_version": manifest["schema_version"],
                "isaac_sim": manifest["isaac_sim"]["status"],
                "gpu": manifest["gpu"]["status"],
                "repository": manifest["repository"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
