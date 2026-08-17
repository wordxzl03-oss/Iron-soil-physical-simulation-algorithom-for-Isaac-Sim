import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from phase_a_baseline import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    EXPECTED_CONTRACT,
    EXPECTED_EPISODE_FILES,
    FROZEN_KEY_HASHES,
    TRACKED_FILE_GROUPS,
    build_manifest,
    validate_manifest,
    validate_manifest_data,
)


class PhaseABaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = REPO_ROOT / DEFAULT_MANIFEST_PATH
        cls.checked_in_manifest = json.loads(
            cls.manifest_path.read_text(encoding="utf-8")
        )

    def test_generator_is_deterministic_and_covers_reference(self) -> None:
        generated = build_manifest(REPO_ROOT)
        self.assertEqual(generated, self.checked_in_manifest)
        self.assertEqual(len(generated["episode_files"]), 32)
        self.assertEqual(
            [item["path"].rsplit("/", 1)[-1] for item in generated["episode_files"]],
            list(EXPECTED_EPISODE_FILES),
        )
        self.assertEqual(generated["contract"], dict(EXPECTED_CONTRACT))
        self.assertEqual(generated["key_hashes"], dict(FROZEN_KEY_HASHES))
        self.assertEqual(
            sum(len(records) for records in generated["tracked_files"].values()),
            sum(len(paths) for paths in TRACKED_FILE_GROUPS.values()),
        )

    def test_validator_accepts_reference_and_rejects_manifest_tampering(self) -> None:
        report = validate_manifest(REPO_ROOT, self.manifest_path)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["verified_episode_file_count"], 32)
        self.assertEqual(report["contract"], dict(EXPECTED_CONTRACT))

        tampered = copy.deepcopy(self.checked_in_manifest)
        tampered["episode_files"][0]["sha256"] = "0" * 64
        tampered["contract"]["full_frame_count"] = 1439
        tampered_report = validate_manifest_data(REPO_ROOT, tampered)
        self.assertFalse(tampered_report["ok"])
        self.assertTrue(
            any("SHA-256 mismatch" in error for error in tampered_report["errors"])
        )
        self.assertTrue(
            any("full_frame_count mismatch" in error for error in tampered_report["errors"])
        )

    def test_cli_validator_reports_an_unreadable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-json.json"
            path.write_text("not JSON", encoding="utf-8")
            report = validate_manifest(REPO_ROOT, path)
        self.assertFalse(report["ok"])
        self.assertTrue(any("cannot read manifest" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
