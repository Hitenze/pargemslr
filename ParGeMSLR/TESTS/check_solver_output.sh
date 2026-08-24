#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 || "$7" != "--" ]]; then
  echo "usage: $0 name tolerance solution-tolerance residual-difference-tolerance maxits openmp-threads -- command [args...]" >&2
  exit 2
fi

name=$1
tolerance=$2
solution_tolerance=$3
residual_difference_tolerance=$4
maxits=$5
openmp_threads=$6
shift 7

output_file="$(mktemp "${TMPDIR:-/tmp}/pargemslr-solver.XXXXXX")"
trap 'rm -f "${output_file}"' EXIT

set +e
"$@" 2>&1 | tee "${output_file}"
status=${PIPESTATUS[0]}
set -e
if [[ ${status} -ne 0 ]]; then
  echo "${name} failed with exit code ${status}" >&2
  exit "${status}"
fi

awk -v name="${name}" -v tol="${tolerance}" -v soltol="${solution_tolerance}" \
    -v restol="${residual_difference_tolerance}" -v maxits="${maxits}" \
    -v expected_threads="${openmp_threads}" '
  /Final rel res:/ { final_res = $NF }
  /True rel res:/ { true_res = $NF }
  /Solution rel error:/ { solution_error = $NF }
  /Number of iterations:/ { iterations = $NF }
  /^OPENMP Info:/ { in_openmp = 1; next }
  /^[^[:space:]].*Info:/ { in_openmp = 0 }
  in_openmp && /OpenMP Threads Per Node:/ { actual_threads = $NF }
  END {
    number = "^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][-+]?[0-9]+)?$";
    integer = "^[0-9]+$";
    if (final_res !~ number || true_res !~ number || solution_error !~ number ||
        iterations !~ integer || tolower(final_res true_res solution_error) ~ /nan|inf/) {
      printf "%s produced missing or non-finite metrics\n", name > "/dev/stderr";
      exit 1;
    }
    if (final_res < 0 || true_res < 0 || solution_error < 0) {
      printf "%s produced negative residual or solution metrics\n", name > "/dev/stderr";
      exit 1;
    }
    difference = final_res - true_res;
    if (difference < 0) difference = -difference;
    if (final_res > tol || true_res > tol || solution_error > soltol ||
        difference > restol || iterations > maxits) {
      printf "%s failed: iterations=%s final=%s true=%s solution=%s residual-difference=%.17g\n",
             name, iterations, final_res, true_res, solution_error, difference > "/dev/stderr";
      exit 1;
    }
    if (expected_threads > 0 && actual_threads != expected_threads) {
      printf "%s reported %s OpenMP threads, expected %s\n",
             name, actual_threads, expected_threads > "/dev/stderr";
      exit 1;
    }
    printf "%s passed: iterations=%s final=%s true=%s solution=%s\n",
           name, iterations, final_res, true_res, solution_error;
  }' "${output_file}"
