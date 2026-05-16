#!/usr/bin/env bash
# Uruchomienie konkretnego modułu `aaop.<pakiet>.run` z korzenia repozytorium,
# jak w README / run_all_labs — używane z dokumentów Quarto w `r_mirror/`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO}/src"
cd "$REPO"
if [[ -x "${REPO}/.venv/bin/python" ]]; then
  PY="${REPO}/.venv/bin/python"
else
  PY=python3
fi
MODULE="${1:?usage: run_aaop_lab.sh <pakiet_pod_aaop_labXX_YYY> [--args...]}"
shift || true
exec "${PY}" -m "aaop.${MODULE}.run" "$@"
