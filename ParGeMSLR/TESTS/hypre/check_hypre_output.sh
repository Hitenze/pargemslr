#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || "$4" != "--" ]]; then
  echo "usage: $0 name tolerance maxits -- command [args...]" >&2
  exit 2
fi

name=$1
tolerance=$2
maxits=$3
shift 4

output_file="$(mktemp "${TMPDIR:-/tmp}/pargemslr-hypre.XXXXXX")"
trap 'rm -f "${output_file}"' EXIT

set +e
"$@" 2>&1 | tee "${output_file}"
status=${PIPESTATUS[0]}
set -e
if [[ ${status} -ne 0 ]]; then
  echo "${name} failed with exit code ${status}" >&2
  exit "${status}"
fi

awk -v name="${name}" -v tol="${tolerance}" -v maxits="${maxits}" '
  /FlexGMRES Iterations =/ { iterations = $NF }
  /Final FlexGMRES Relative Residual Norm =/ { residual = $NF }
  END {
    number = "^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][-+]?[0-9]+)?$";
    integer = "^[0-9]+$";
    if (residual !~ number || iterations !~ integer ||
        tolower(residual) ~ /nan|inf/) {
      printf "%s produced missing or non-finite metrics\n", name > "/dev/stderr";
      exit 1;
    }
    if (residual < 0) {
      printf "%s produced a negative residual\n", name > "/dev/stderr";
      exit 1;
    }
    if (residual > tol || iterations > maxits) {
      printf "%s failed: iterations=%s residual=%s\n",
             name, iterations, residual > "/dev/stderr";
      exit 1;
    }
    printf "%s passed: iterations=%s residual=%s\n",
           name, iterations, residual;
  }' "${output_file}"
