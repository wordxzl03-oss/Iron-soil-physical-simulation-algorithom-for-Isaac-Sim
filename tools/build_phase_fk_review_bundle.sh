#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle_name="review_bundle_phase_f_k_20260807_v1"
bundle_root="${repository_root}/${bundle_name}"
archive_path="${repository_root}/${bundle_name}.zip"

if [[ -e "${bundle_root}" || -e "${archive_path}" ]]; then
  echo "Refusing to overwrite existing review bundle: ${bundle_root} or ${archive_path}" >&2
  exit 2
fi

mkdir -p "${bundle_root}/outputs"
cp -a "${repository_root}/src" "${bundle_root}/"
cp -a "${repository_root}/configs" "${bundle_root}/"
cp -a "${repository_root}/isaac_loader" "${bundle_root}/"
cp -a "${repository_root}/tests" "${bundle_root}/"
cp -a "${repository_root}/tools" "${bundle_root}/"
cp -a "${repository_root}/docs" "${bundle_root}/"
cp -a "${repository_root}/README.md" "${bundle_root}/"
cp -a "${repository_root}/architecture_audit.md" "${bundle_root}/"
cp -a "${repository_root}/pyproject.toml" "${bundle_root}/"
cp -a "${repository_root}/requirements.txt" "${bundle_root}/"
cp -a "${repository_root}/slope_model.py" "${bundle_root}/"
cp -a "${repository_root}/run_unit_tests.sh" "${bundle_root}/"
cp -a "${repository_root}/run_full_repository_tests.sh" "${bundle_root}/"
cp -a "${repository_root}/run_phase_b_manual.sh" "${bundle_root}/"
cp -a "${repository_root}/run_phase_cd_acceptance.sh" "${bundle_root}/"
cp -a "${repository_root}/run_isaac_demo.sh" "${bundle_root}/"
cp -a "${repository_root}/run_isaac_headless_acceptance.sh" "${bundle_root}/"
cp -a "${repository_root}/outputs/phase_a_baseline_manifest.json" "${bundle_root}/outputs/"
cp -a "${repository_root}/outputs/phase_a_usd_inventory.json" "${bundle_root}/outputs/"
cp -a "${repository_root}/outputs/phase_b_manual_runtime.json" "${bundle_root}/outputs/"
cp -a "${repository_root}/outputs/phase_bcde_acceptance_validation.json" "${bundle_root}/outputs/"
cp -a "${repository_root}/outputs/phase_cd_slope_summary.json" "${bundle_root}/outputs/"
cp -a "${repository_root}/outputs/phase_e_acceptance.json" "${bundle_root}/outputs/"
cp -a "${repository_root}/outputs/phase_f_to_k_acceptance.json" "${bundle_root}/outputs/"
cp -a "${repository_root}/outputs/phase_k_scalability_benchmark.json" "${bundle_root}/outputs/"

find "${bundle_root}" -type f -print0 | sort -z | xargs -0 sha256sum > "${bundle_root}/MANIFEST.sha256"

cd "${repository_root}"
zip -q -r "${archive_path}" "${bundle_name}"
sha256sum "${archive_path}" > "${archive_path}.sha256"

echo "${bundle_root}"
echo "${archive_path}"
