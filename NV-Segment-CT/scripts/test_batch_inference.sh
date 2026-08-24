#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_DIR=${WORK_DIR:-$(mktemp -d)}
KEEP_TEST_WORKDIR=${KEEP_TEST_WORKDIR:-0}

if [[ "${KEEP_TEST_WORKDIR}" != "1" ]]; then
    trap 'rm -rf "${WORK_DIR}"' EXIT
fi

INPUT_DIR="${WORK_DIR}/input"
OUTPUT_DIR="${WORK_DIR}/output"
mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"

cp "${BUNDLE_ROOT}/example/spleen_03.nii.gz" "${INPUT_DIR}/"

cd "${BUNDLE_ROOT}"

"${PYTHON_BIN}" -m monai.bundle run \
    --config_file="['configs/inference.json', 'configs/batch_inference.json']" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    2>&1 | tee "${WORK_DIR}/batch_inference.log"

EXPECTED_OUTPUT="${OUTPUT_DIR}/spleen_03/spleen_03_trans.nii.gz"
test -s "${EXPECTED_OUTPUT}"
grep -q "\[nvseg\] batch resume (skip existing outputs): 1 volume" "${WORK_DIR}/batch_inference.log"
! grep -q "registered Hugging Face download" "${WORK_DIR}/batch_inference.log"
"${PYTHON_BIN}" -m monai.bundle run \
    --config_file="['configs/inference.json', 'configs/batch_inference.json']" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    2>&1 | tee "${WORK_DIR}/batch_resume.log"

grep -q "\[nvseg\] batch: nothing to run (resume); ok" "${WORK_DIR}/batch_resume.log"
! grep -q "registered Hugging Face download" "${WORK_DIR}/batch_resume.log"

echo "[nvseg-test] CT batch inference smoke test passed: ${EXPECTED_OUTPUT}"
