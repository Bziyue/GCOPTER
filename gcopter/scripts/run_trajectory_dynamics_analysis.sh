#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COEFF_FILE="${1:-${SCRIPT_DIR}/latest_trajectory_coefficients.json}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/trajectory_dynamics_output}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export GCOPTER_ENABLE_MATPLOTLIB="${GCOPTER_ENABLE_MATPLOTLIB:-1}"

PYTHON_BIN="/home/zdp/anaconda3/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/trajectory_dynamics_analysis.py" \
  --coeff-file "${COEFF_FILE}" \
  --output-dir "${OUTPUT_DIR}"
