#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../.." && pwd)"

read -r -a make_cmd <<< "${MAKE:-make}"
smoke_n="${PARGEMSLR_STANDARD_N:-16}"
smoke_tol="${PARGEMSLR_STANDARD_TOL:-1e-10}"
smoke_sol_tol="${PARGEMSLR_STANDARD_SOL_TOL:-1e-10}"
smoke_maxits="${PARGEMSLR_STANDARD_MAXITS:-100}"
smoke_kdim="${PARGEMSLR_STANDARD_KDIM:-40}"
res_compare_tol="${PARGEMSLR_STANDARD_RES_COMPARE_TOL:-1e-12}"
compare_tol="${PARGEMSLR_STANDARD_COMPARE_TOL:-1e-12}"
metric_compare_tol="${PARGEMSLR_STANDARD_METRIC_COMPARE_TOL:-5e-11}"
solution_compare_tol="${PARGEMSLR_STANDARD_SOLUTION_COMPARE_TOL:-1e-10}"
solution_max_abs_tol="${PARGEMSLR_STANDARD_SOLUTION_MAX_ABS_TOL:-5e-10}"
rerun_solution_compare_tol="${PARGEMSLR_STANDARD_RERUN_SOLUTION_COMPARE_TOL:-1e-12}"
rerun_solution_max_abs_tol="${PARGEMSLR_STANDARD_RERUN_SOLUTION_MAX_ABS_TOL:-1e-12}"
test_openmp="${PARGEMSLR_STANDARD_OPENMP:-1}"
test_rank4="${PARGEMSLR_STANDARD_RANK4:-1}"
test_cuda="${PARGEMSLR_STANDARD_CUDA:-${PARGEMSLR_TEST_CUDA:-0}}"

if [[ "${MPIRUN_RANK_FLAG+x}" == "x" && -z "${MPIRUN_RANK_FLAG}" ]]; then
  echo "standard-test requires a non-empty MPIRUN_RANK_FLAG so each case can set its MPI rank count." >&2
  exit 1
fi

created_workdir=0
if [[ -n "${PARGEMSLR_STANDARD_WORKDIR:-}" ]]; then
  workdir="${PARGEMSLR_STANDARD_WORKDIR}"
  mkdir -p "${workdir}"
else
  workdir="$(mktemp -d "${TMPDIR:-/tmp}/pargemslr-standard.XXXXXX")"
  created_workdir=1
fi

cleanup() {
  local code=$?
  if [[ ${code} -eq 0 && ${created_workdir} -eq 1 && "${PARGEMSLR_KEEP_TEST_WORKDIR:-0}" != "1" ]]; then
    rm -rf "${workdir}"
  else
    echo "Standard test workdir: ${workdir}" >&2
  fi
}
trap cleanup EXIT

log() {
  printf '\n==> %s\n' "$*"
}

run_smoke_case() {
  local name=$1
  local ranks=$2
  local threads=$3
  local openmp=$4
  local cuda=${5:-0}
  local case_dir="${workdir}/${name}"
  rm -rf "${case_dir}"
  mkdir -p "${case_dir}"

  log "${name}: ranks=${ranks} threads=${threads} openmp=${openmp} cuda=${cuda}"
  PARGEMSLR_SMOKE_WORKDIR="${case_dir}" \
  PARGEMSLR_SEQ_RANKS=1 \
  PARGEMSLR_PAR_RANKS="${ranks}" \
  PARGEMSLR_SMOKE_N="${smoke_n}" \
  PARGEMSLR_SMOKE_TOL="${smoke_tol}" \
  PARGEMSLR_SMOKE_SOL_TOL="${smoke_sol_tol}" \
  PARGEMSLR_SMOKE_MAXITS="${smoke_maxits}" \
  PARGEMSLR_SMOKE_KDIM="${smoke_kdim}" \
  PARGEMSLR_SMOKE_RES_COMPARE_TOL="${res_compare_tol}" \
  PARGEMSLR_SMOKE_THREADS="${threads}" \
  PARGEMSLR_SMOKE_WRITE_SOL=1 \
  PARGEMSLR_TEST_CUDA="${cuda}" \
  USING_OPENMP="${openmp}" \
  OMP_NUM_THREADS="${threads}" \
  OMP_DYNAMIC=FALSE \
  MKL_DYNAMIC=FALSE \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  OMP_PROC_BIND="${PARGEMSLR_STANDARD_OMP_PROC_BIND:-close}" \
  OMP_PLACES="${PARGEMSLR_STANDARD_OMP_PLACES:-cores}" \
  "${make_cmd[@]}" -C "${root_dir}" smoke-test
}

extract_metric() {
  local log_file=$1
  local pattern=$2
  awk -v pattern="${pattern}" '$0 ~ pattern {value=$NF} END {if (value == "") exit 1; print value}' "${log_file}"
}

assert_metric_close() {
  local label=$1
  local lhs=$2
  local rhs=$3
  local tol=$4
  awk -v label="${label}" -v lhs="${lhs}" -v rhs="${rhs}" -v tol="${tol}" '
    BEGIN {
      numeric = "^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][-+]?[0-9]+)?$";
      if (lhs !~ numeric || rhs !~ numeric || tolower(lhs rhs) ~ /nan|inf/) {
        printf "%s has non-finite metric: %s %s\n", label, lhs, rhs > "/dev/stderr";
        exit 1;
      }
      if ((lhs + 0.0) < 0.0 || (rhs + 0.0) < 0.0) {
        printf "%s has negative metric: %s %s\n", label, lhs, rhs > "/dev/stderr";
        exit 1;
      }
      diff = lhs - rhs;
      if (diff < 0) diff = -diff;
      if (diff > tol) {
        printf "%s differs by %.17g, tolerance %.17g (%s vs %s)\n", label, diff, tol, lhs, rhs > "/dev/stderr";
        exit 1;
      }
    }'
}

compare_case_logs_pair() {
  local lhs_name=$1
  local lhs_log_name=$2
  local rhs_name=$3
  local rhs_log_name=$4
  local tol=$5
  local lhs_log="${workdir}/${lhs_name}/${lhs_log_name}"
  local rhs_log="${workdir}/${rhs_name}/${rhs_log_name}"
  local lhs_final rhs_final lhs_true rhs_true lhs_sol rhs_sol lhs_its rhs_its

  lhs_final="$(extract_metric "${lhs_log}" "Final rel res:")"
  rhs_final="$(extract_metric "${rhs_log}" "Final rel res:")"
  lhs_true="$(extract_metric "${lhs_log}" "True rel res:")"
  rhs_true="$(extract_metric "${rhs_log}" "True rel res:")"
  lhs_sol="$(extract_metric "${lhs_log}" "Solution rel error:")"
  rhs_sol="$(extract_metric "${rhs_log}" "Solution rel error:")"
  lhs_its="$(extract_metric "${lhs_log}" "Number of iterations:")"
  rhs_its="$(extract_metric "${rhs_log}" "Number of iterations:")"

  assert_metric_close "${rhs_name}/${rhs_log_name} final residual vs ${lhs_name}/${lhs_log_name}" "${lhs_final}" "${rhs_final}" "${tol}"
  assert_metric_close "${rhs_name}/${rhs_log_name} true residual vs ${lhs_name}/${lhs_log_name}" "${lhs_true}" "${rhs_true}" "${tol}"
  assert_metric_close "${rhs_name}/${rhs_log_name} solution relative error vs ${lhs_name}/${lhs_log_name}" "${lhs_sol}" "${rhs_sol}" "${tol}"
  if [[ "${lhs_its}" != "${rhs_its}" ]]; then
    echo "${rhs_name}/${rhs_log_name} iteration count differs from ${lhs_name}/${lhs_log_name}: ${rhs_its} vs ${lhs_its}" >&2
    exit 1
  fi
}

compare_case_logs() {
  local lhs_name=$1
  local rhs_name=$2
  local log_name=$3
  compare_case_logs_pair "${lhs_name}" "${log_name}" "${rhs_name}" "${log_name}" "${compare_tol}"
}

compare_cross_case_logs() {
  local baseline_name=$1
  local baseline_log=$2
  local candidate_name=$3
  local candidate_log=$4
  compare_case_logs_pair "${baseline_name}" "${baseline_log}" "${candidate_name}" "${candidate_log}" "${metric_compare_tol}"
}

collect_parallel_solution() {
  local case_name=$1
  local prefix=$2
  local out_file=$3
  local files=("${workdir}/${case_name}/${prefix}00000.sol"[0-9][0-9][0-9][0-9][0-9])

  if [[ ${#files[@]} -eq 0 || ! -e "${files[0]}" ]]; then
    echo "Missing solution files for ${case_name}/${prefix}" >&2
    exit 1
  fi

  : > "${out_file}"
  local file
  for file in "${files[@]}"; do
    cat "${file}" >> "${out_file}"
  done
}

collect_sequential_solution() {
  local case_name=$1
  local prefix=$2
  local out_file=$3
  local files=("${workdir}/${case_name}/${prefix}"[0-9][0-9][0-9][0-9][0-9].sol)

  if [[ ${#files[@]} -eq 0 || ! -e "${files[0]}" ]]; then
    echo "Missing solution files for ${case_name}/${prefix}" >&2
    exit 1
  fi

  : > "${out_file}"
  local file
  for file in "${files[@]}"; do
    cat "${file}" >> "${out_file}"
  done
}

assert_solution_close() {
  local label=$1
  local baseline_file=$2
  local candidate_file=$3
  local tol=${4:-${solution_compare_tol}}
  local max_abs_tol=${5:-${solution_max_abs_tol}}

  awk -v label="${label}" -v tol="${tol}" -v maxtol="${max_abs_tol}" '
    function isnum(x) {
      return x ~ /^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][-+]?[0-9]+)?$/ && tolower(x) !~ /nan|inf/;
    }
    NR == FNR {
      if (!isnum($1)) {
        printf "%s baseline has non-finite value on line %d: %s\n", label, FNR, $1 > "/dev/stderr";
        exit 1;
      }
      base[++n] = $1 + 0.0;
      base_norm += base[n] * base[n];
      next;
    }
    {
      if (!isnum($1)) {
        printf "%s candidate has non-finite value on line %d: %s\n", label, FNR, $1 > "/dev/stderr";
        exit 1;
      }
      m++;
      if (m > n) {
        printf "%s candidate has more entries than baseline\n", label > "/dev/stderr";
        exit 1;
      }
      value = $1 + 0.0;
      diff = value - base[m];
      diff_norm += diff * diff;
      if (diff < 0.0) diff = -diff;
      if (diff > max_diff) max_diff = diff;
    }
    END {
      if (n == 0) {
        printf "%s baseline vector is empty\n", label > "/dev/stderr";
        exit 1;
      }
      if (m != n) {
        printf "%s vector lengths differ: %d vs %d\n", label, n, m > "/dev/stderr";
        exit 1;
      }
      denom = sqrt(base_norm);
      if (denom == 0.0) denom = 1.0;
      rel = sqrt(diff_norm) / denom;
      if (rel > (tol + 0.0)) {
        printf "%s solution relative difference %.17g exceeds tolerance %.17g (max abs %.17g)\n", label, rel, tol, max_diff > "/dev/stderr";
        exit 1;
      }
      if (max_diff > (maxtol + 0.0)) {
        printf "%s solution max-abs difference %.17g exceeds tolerance %.17g (rel %.17g)\n", label, max_diff, maxtol, rel > "/dev/stderr";
        exit 1;
      }
      printf "%s solution comparison passed: rel_diff=%.17g max_abs=%.17g\n", label, rel, max_diff;
    }' "${baseline_file}" "${candidate_file}"
}

compare_solution_vectors() {
  local baseline_case=$1
  local baseline_prefix=$2
  local candidate_case=$3
  local candidate_prefix=$4
  local tol=${5:-${solution_compare_tol}}
  local max_abs_tol=${6:-${solution_max_abs_tol}}
  local baseline_vec="${workdir}/${baseline_case}/${baseline_prefix}.vector"
  local candidate_vec="${workdir}/${candidate_case}/${candidate_prefix}.vector"

  collect_parallel_solution "${baseline_case}" "${baseline_prefix}" "${baseline_vec}"
  collect_parallel_solution "${candidate_case}" "${candidate_prefix}" "${candidate_vec}"
  assert_solution_close "${candidate_case} vs ${baseline_case}" "${baseline_vec}" "${candidate_vec}" "${tol}" "${max_abs_tol}"
}

compare_sequential_solution_vectors() {
  local baseline_case=$1
  local candidate_case=$2
  local tol=${3:-${solution_compare_tol}}
  local max_abs_tol=${4:-${solution_max_abs_tol}}
  local baseline_vec="${workdir}/${baseline_case}/sequential_cpu.vector"
  local candidate_vec="${workdir}/${candidate_case}/sequential_cpu.vector"

  collect_sequential_solution "${baseline_case}" sequential_cpu "${baseline_vec}"
  collect_sequential_solution "${candidate_case}" sequential_cpu "${candidate_vec}"
  assert_solution_close "${candidate_case} sequential vs ${baseline_case}" "${baseline_vec}" "${candidate_vec}" "${tol}" "${max_abs_tol}"
}

baseline_case=cpu_r1_t1_a
run_smoke_case "${baseline_case}" 1 1 0
run_smoke_case cpu_r1_t1_b 1 1 0
compare_case_logs "${baseline_case}" cpu_r1_t1_b sequential_cpu.log
compare_case_logs "${baseline_case}" cpu_r1_t1_b parallel_cpu.log
compare_sequential_solution_vectors "${baseline_case}" cpu_r1_t1_b "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
compare_solution_vectors "${baseline_case}" parallel_cpu cpu_r1_t1_b parallel_cpu "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"

run_smoke_case cpu_r2_t1_a 2 1 0
run_smoke_case cpu_r2_t1_b 2 1 0
compare_case_logs cpu_r2_t1_a cpu_r2_t1_b sequential_cpu.log
compare_case_logs cpu_r2_t1_a cpu_r2_t1_b parallel_cpu.log
compare_sequential_solution_vectors cpu_r2_t1_a cpu_r2_t1_b "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
compare_solution_vectors cpu_r2_t1_a parallel_cpu cpu_r2_t1_b parallel_cpu "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
compare_cross_case_logs "${baseline_case}" sequential_cpu.log cpu_r2_t1_a sequential_cpu.log
compare_cross_case_logs "${baseline_case}" parallel_cpu.log cpu_r2_t1_a parallel_cpu.log
compare_sequential_solution_vectors "${baseline_case}" cpu_r2_t1_a
compare_solution_vectors "${baseline_case}" parallel_cpu cpu_r2_t1_a parallel_cpu

if [[ "${test_rank4}" == "1" ]]; then
  run_smoke_case cpu_r4_t1_a 4 1 0
  run_smoke_case cpu_r4_t1_b 4 1 0
  compare_case_logs cpu_r4_t1_a cpu_r4_t1_b sequential_cpu.log
  compare_case_logs cpu_r4_t1_a cpu_r4_t1_b parallel_cpu.log
  compare_sequential_solution_vectors cpu_r4_t1_a cpu_r4_t1_b "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
  compare_solution_vectors cpu_r4_t1_a parallel_cpu cpu_r4_t1_b parallel_cpu "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
  compare_cross_case_logs "${baseline_case}" sequential_cpu.log cpu_r4_t1_a sequential_cpu.log
  compare_cross_case_logs "${baseline_case}" parallel_cpu.log cpu_r4_t1_a parallel_cpu.log
  compare_sequential_solution_vectors "${baseline_case}" cpu_r4_t1_a
  compare_solution_vectors "${baseline_case}" parallel_cpu cpu_r4_t1_a parallel_cpu
fi

if [[ "${test_openmp}" == "1" ]]; then
  run_smoke_case cpu_r2_t2_omp_a 2 2 1
  run_smoke_case cpu_r2_t2_omp_b 2 2 1
  compare_case_logs cpu_r2_t2_omp_a cpu_r2_t2_omp_b sequential_cpu.log
  compare_case_logs cpu_r2_t2_omp_a cpu_r2_t2_omp_b parallel_cpu.log
  compare_sequential_solution_vectors cpu_r2_t2_omp_a cpu_r2_t2_omp_b "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
  compare_solution_vectors cpu_r2_t2_omp_a parallel_cpu cpu_r2_t2_omp_b parallel_cpu "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
  compare_cross_case_logs "${baseline_case}" sequential_cpu.log cpu_r2_t2_omp_a sequential_cpu.log
  compare_cross_case_logs "${baseline_case}" parallel_cpu.log cpu_r2_t2_omp_a parallel_cpu.log
  compare_sequential_solution_vectors "${baseline_case}" cpu_r2_t2_omp_a
  compare_solution_vectors "${baseline_case}" parallel_cpu cpu_r2_t2_omp_a parallel_cpu
fi

if [[ "${test_cuda}" == "1" ]]; then
  run_smoke_case cuda_r2_t1_a 2 1 0 1
  run_smoke_case cuda_r2_t1_b 2 1 0 1
  compare_case_logs cuda_r2_t1_a cuda_r2_t1_b sequential_cpu.log
  compare_case_logs cuda_r2_t1_a cuda_r2_t1_b parallel_cpu.log
  compare_case_logs cuda_r2_t1_a cuda_r2_t1_b parallel_cuda.log
  compare_sequential_solution_vectors cuda_r2_t1_a cuda_r2_t1_b "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
  compare_solution_vectors cuda_r2_t1_a parallel_cpu cuda_r2_t1_b parallel_cpu "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
  compare_solution_vectors cuda_r2_t1_a parallel_cuda cuda_r2_t1_b parallel_cuda "${rerun_solution_compare_tol}" "${rerun_solution_max_abs_tol}"
  compare_cross_case_logs "${baseline_case}" sequential_cpu.log cuda_r2_t1_a sequential_cpu.log
  compare_cross_case_logs "${baseline_case}" parallel_cpu.log cuda_r2_t1_a parallel_cuda.log
  compare_sequential_solution_vectors "${baseline_case}" cuda_r2_t1_a
  compare_solution_vectors "${baseline_case}" parallel_cpu cuda_r2_t1_a parallel_cuda
fi

log "Standard tests passed"
