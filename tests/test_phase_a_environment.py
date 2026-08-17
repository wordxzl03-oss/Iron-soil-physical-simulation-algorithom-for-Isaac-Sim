import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.collect_phase_a_environment import (
    SCHEMA_VERSION,
    collect_gpu,
    collect_isaac_sim,
    collect_manifest,
    collect_repo_revision,
    write_manifest,
)


class PhaseAEnvironmentTests(unittest.TestCase):
    def test_isaac_package_files_are_preserved_exactly_without_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kit").mkdir()
            version = "4.5.0-rc.36+release.test\n"
            package = "Package: isaac-sim-standalone\nVersion: exact-build\n"
            kit_package = "Package: OmniverseKit\nVersion: exact-kit\n"
            (root / "VERSION").write_text(version, encoding="utf-8")
            (root / "PACKAGE-INFO.yaml").write_text(package, encoding="utf-8")
            (root / "kit" / "PACKAGE-INFO.yaml").write_text(
                kit_package, encoding="utf-8"
            )

            record = collect_isaac_sim(root)

            self.assertEqual(record["status"], "available")
            self.assertEqual(record["root_locator"], "cli_argument")
            self.assertEqual(record["files"]["VERSION"]["raw_text"], version)
            self.assertEqual(
                record["files"]["PACKAGE-INFO.yaml"]["raw_text"], package
            )
            serialized = json.dumps(record)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("path", serialized.lower())

    def test_gpu_query_has_available_and_failure_contracts(self):
        def success(command, timeout):
            self.assertEqual(command[0], "nvidia-smi")
            self.assertEqual(timeout, 5.0)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "NVIDIA GeForce RTX 3070 Ti Laptop GPU, 580.173.02, 8192\n"
                ),
                stderr="",
            )

        available = collect_gpu(command_runner=success, command_available=True)
        self.assertEqual(available["status"], "available")
        self.assertEqual(
            available["gpus"],
            [
                {
                    "name": "NVIDIA GeForce RTX 3070 Ti Laptop GPU",
                    "driver_version": "580.173.02",
                    "memory_total_mib": 8192,
                }
            ],
        )

        def failure(command, timeout):
            return subprocess.CompletedProcess(command, 9, stdout="", stderr="secret")

        unavailable = collect_gpu(command_runner=failure, command_available=True)
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertEqual(unavailable["reason"], "command_failed")
        self.assertNotIn("secret", json.dumps(unavailable))

    def test_repo_revision_records_explicit_unavailable_or_loose_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unavailable = collect_repo_revision(root, git_available=False)
            self.assertEqual(unavailable["status"], "unavailable")
            self.assertIsNone(unavailable["revision"])
            self.assertIn("unavailable", unavailable["reason"])

            ref = root / ".git" / "refs" / "heads"
            ref.mkdir(parents=True)
            revision = "0123456789abcdef0123456789abcdef01234567"
            (root / ".git" / "HEAD").write_text(
                "ref: refs/heads/main\n", encoding="ascii"
            )
            (ref / "main").write_text(revision + "\n", encoding="ascii")
            available = collect_repo_revision(root, git_available=False)
            self.assertEqual(available["status"], "available")
            self.assertEqual(available["revision"], revision)
            self.assertEqual(available["reference"], "refs/heads/main")
            self.assertEqual(available["method"], "git_metadata")

    def test_manifest_is_privacy_bounded_and_writes_valid_json(self):
        def failed_command(command, timeout):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="ignored")

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {
                "USER": "private-user",
                "PASSWORD": "private-password",
                "TOP_SECRET_TOKEN": "private-token",
            },
            clear=False,
        ):
            root = Path(directory)
            manifest = collect_manifest(
                repo_root=root,
                isaac_root=root / "missing-isaac",
                generated_at_utc="2026-08-07T00:00:00Z",
                command_runner=failed_command,
                nvidia_smi_available=True,
                git_available=True,
            )
            destination = root / "manifest.json"
            write_manifest(destination, manifest)
            loaded = json.loads(destination.read_text(encoding="utf-8"))

            self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
            self.assertEqual(loaded["generated_at_utc"], "2026-08-07T00:00:00Z")
            self.assertEqual(loaded["gpu"]["status"], "unavailable")
            self.assertEqual(loaded["repository"]["status"], "unavailable")
            self.assertTrue(
                all(value is False for value in loaded["privacy"].values())
            )
            serialized = json.dumps(loaded)
            for forbidden in (
                "private-user",
                "private-password",
                "private-token",
                str(root),
            ):
                self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
