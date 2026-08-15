#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG.yaml" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$1"

for method in ours dc full_sh pbr_only; do
  output="${project_dir}/outputs/${method}"
  python -m raystyle train --config "${config}" --method "${method}" --output-dir "${output}"
  python -m raystyle evaluate \
    --config "${config}" \
    --method "${method}" \
    --output-dir "${output}" \
    --checkpoint "${output}/checkpoint_latest.pt"
done

python -m raystyle compare \
  "${project_dir}"/outputs/*/evaluation/summary.json \
  > "${project_dir}/outputs/comparison.csv"

