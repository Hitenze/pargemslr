#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
seq_dir="${project_dir}/TESTS/sequential"
par_dir="${project_dir}/TESTS/parallel"

read -r -a make_cmd <<< "${MAKE:-make}"
read -r -a mpirun_cmd <<< "${MPIRUN:-mpirun}"
mpirun_rank_flag="${MPIRUN_RANK_FLAG:--np}"
seq_ranks="${PARGEMSLR_SEQ_RANKS:-1}"
par_ranks="${PARGEMSLR_PAR_RANKS:-2}"
setup_check_ranks="${PARGEMSLR_SETUP_CHECK_RANKS:-2}"
smoke_n="${PARGEMSLR_SMOKE_N:-16}"
smoke_tol="${PARGEMSLR_SMOKE_TOL:-1e-10}"
smoke_sol_tol="${PARGEMSLR_SMOKE_SOL_TOL:-1e-10}"
smoke_maxits="${PARGEMSLR_SMOKE_MAXITS:-100}"
smoke_kdim="${PARGEMSLR_SMOKE_KDIM:-40}"
res_compare_tol="${PARGEMSLR_SMOKE_RES_COMPARE_TOL:-1e-12}"
using_mkl="${USING_MKL:-0}"
using_openmp="${USING_OPENMP:-0}"
smoke_threads="${PARGEMSLR_SMOKE_THREADS:-1}"
write_sol="${PARGEMSLR_SMOKE_WRITE_SOL:-0}"

if [[ ! "${smoke_threads}" =~ ^[0-9]+$ || "${smoke_threads}" -lt 1 ]]; then
  echo "PARGEMSLR_SMOKE_THREADS must be a positive integer." >&2
  exit 1
fi
if [[ ! "${setup_check_ranks}" =~ ^[0-9]+$ || "${setup_check_ranks}" -lt 2 ]]; then
  echo "PARGEMSLR_SETUP_CHECK_RANKS must be an integer greater than or equal to 2." >&2
  exit 1
fi

: "${OMP_DYNAMIC:=FALSE}"
: "${MKL_DYNAMIC:=FALSE}"
: "${MKL_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${BLIS_NUM_THREADS:=1}"
: "${VECLIB_MAXIMUM_THREADS:=1}"
: "${NUMEXPR_NUM_THREADS:=1}"
OMP_NUM_THREADS="${smoke_threads}"
export OMP_NUM_THREADS OMP_DYNAMIC MKL_DYNAMIC MKL_NUM_THREADS OPENBLAS_NUM_THREADS BLIS_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS

created_workdir=0
if [[ -n "${PARGEMSLR_SMOKE_WORKDIR:-}" ]]; then
  workdir="${PARGEMSLR_SMOKE_WORKDIR}"
  mkdir -p "${workdir}"
else
  workdir="$(mktemp -d "${TMPDIR:-/tmp}/pargemslr-smoke.XXXXXX")"
  created_workdir=1
fi

cleanup() {
  local code=$?
  if [[ ${code} -eq 0 && ${created_workdir} -eq 1 && "${PARGEMSLR_KEEP_TEST_WORKDIR:-0}" != "1" ]]; then
    rm -rf "${workdir}"
  else
    echo "Smoke test workdir: ${workdir}" >&2
  fi
}
trap cleanup EXIT

log() {
  printf '\n==> %s\n' "$*"
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

mpi_run() {
  local ranks=$1
  shift
  if [[ -n "${mpirun_rank_flag}" ]]; then
    "${mpirun_cmd[@]}" "${mpirun_rank_flag}" "${ranks}" "$@"
  else
    "${mpirun_cmd[@]}" "$@"
  fi
}

run_and_check() {
  local name=$1
  local cwd=$2
  local log_file=$3
  shift 3

  log "${name}"
  set +e
  (
    cd "${cwd}"
    "$@"
  ) 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "${name} failed with exit code ${status}" >&2
    exit "${status}"
  fi

  local final_res
  final_res="$(awk '/Final rel res:/ {value=$NF} END {if (value == "") exit 1; print value}' "${log_file}")"
  local true_res
  true_res="$(awk '/True rel res:/ {value=$NF} END {if (value == "") exit 1; print value}' "${log_file}")"
  local iterations
  iterations="$(awk '/Number of iterations:/ {value=$NF} END {if (value == "") exit 1; print value}' "${log_file}")"
  local solution_err
  solution_err="$(awk '/Solution rel error:/ {value=$NF} END {if (value == "") exit 1; print value}' "${log_file}")"

  awk -v finalres="${final_res}" -v trueres="${true_res}" -v tol="${smoke_tol}" \
      -v restol="${res_compare_tol}" -v sol="${solution_err}" -v soltol="${smoke_sol_tol}" \
      -v its="${iterations}" -v maxits="${smoke_maxits}" -v name="${name}" '
    BEGIN {
      numeric = "^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][-+]?[0-9]+)?$";
      integer = "^[-+]?[0-9]+$";
      if (tolower(finalres) ~ /nan|inf/ || finalres !~ numeric) {
        printf "%s final relative residual is not finite: %s\n", name, finalres > "/dev/stderr";
        exit 1;
      }
      if (tolower(trueres) ~ /nan|inf/ || trueres !~ numeric) {
        printf "%s true relative residual is not finite: %s\n", name, trueres > "/dev/stderr";
        exit 1;
      }
      if (tolower(its) ~ /nan|inf/ || its !~ integer) {
        printf "%s iteration count is not an integer: %s\n", name, its > "/dev/stderr";
        exit 1;
      }
      if (tolower(sol) ~ /nan|inf/ || sol !~ numeric) {
        printf "%s solution relative error is not finite: %s\n", name, sol > "/dev/stderr";
        exit 1;
      }
      if ((finalres + 0.0) < 0.0) {
        printf "%s final relative residual is negative: %s\n", name, finalres > "/dev/stderr";
        exit 1;
      }
      if ((trueres + 0.0) < 0.0) {
        printf "%s true relative residual is negative: %s\n", name, trueres > "/dev/stderr";
        exit 1;
      }
      if ((its + 0) < 0) {
        printf "%s iteration count is negative: %s\n", name, its > "/dev/stderr";
        exit 1;
      }
      if ((sol + 0.0) < 0.0) {
        printf "%s solution relative error is negative: %s\n", name, sol > "/dev/stderr";
        exit 1;
      }
      if ((finalres + 0.0) > (tol + 0.0)) {
        printf "%s final relative residual %g exceeds tolerance %g\n", name, finalres, tol > "/dev/stderr";
        exit 1;
      }
      if ((trueres + 0.0) > (tol + 0.0)) {
        printf "%s true relative residual %g exceeds tolerance %g\n", name, trueres, tol > "/dev/stderr";
        exit 1;
      }
      diff = finalres - trueres;
      if (diff < 0) diff = -diff;
      if (diff > (restol + 0.0)) {
        printf "%s final and true residuals differ by %.17g, tolerance %.17g (%s vs %s)\n", name, diff, restol, finalres, trueres > "/dev/stderr";
        exit 1;
      }
      if ((its + 0) > (maxits + 0)) {
        printf "%s iteration count %d exceeds maxits %d\n", name, its, maxits > "/dev/stderr";
        exit 1;
      }
      if ((sol + 0.0) > (soltol + 0.0)) {
        printf "%s solution relative error %g exceeds tolerance %g\n", name, sol, soltol > "/dev/stderr";
        exit 1;
      }
    }'
  if [[ "${using_openmp}" == "1" ]]; then
    local logged_threads
    logged_threads="$(awk '
      /^OPENMP Info:/ {in_openmp=1; next}
      /^[^[:space:]].*Info:/ {in_openmp=0}
      in_openmp && /OpenMP Threads Per Node:/ {value=$NF}
      END {if (value == "") exit 1; print value}
    ' "${log_file}")"
    if [[ "${logged_threads}" != "${smoke_threads}" ]]; then
      echo "${name} reported ${logged_threads} OpenMP threads, expected ${smoke_threads}" >&2
      exit 1
    fi
  fi
  printf '%s passed: iterations=%s final_rel_res=%s true_rel_res=%s solution_rel_error=%s\n' "${name}" "${iterations}" "${final_res}" "${true_res}" "${solution_err}"
}

run_plain_case() {
  local name=$1
  local cwd=$2
  local log_file=$3
  shift 3

  log "${name}"
  set +e
  (
    cd "${cwd}"
    "$@"
  ) 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "${name} failed with exit code ${status}" >&2
    exit "${status}"
  fi
  printf '%s passed\n' "${name}"
}

build_parallel_vector_setup_check() {
  run "${make_cmd[@]}" -C "${par_dir}" \
    USING_CUDA=0 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" \
    parallel_vector_setup_check.ex
}

build_parallel_matrix_ops_check() {
  run "${make_cmd[@]}" -C "${par_dir}" \
    USING_CUDA=0 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" \
    parallel_matrix_ops_check.ex
}

build_parallel_gemslr_setup_check() {
  run "${make_cmd[@]}" -C "${par_dir}" \
    USING_CUDA=0 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" \
    parallel_gemslr_setup_check.ex
}

build_parallel_error_checks_release() {
  run "${make_cmd[@]}" -C "${par_dir}" clean
  run "${make_cmd[@]}" -C "${project_dir}" clean
  run "${make_cmd[@]}" -C "${project_dir}" \
    USING_CUDA=0 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" DEBUG_MODE=0
  run "${make_cmd[@]}" -C "${par_dir}" \
    USING_CUDA=0 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" DEBUG_MODE=0 \
    parallel_vector_setup_check.ex parallel_matrix_ops_check.ex parallel_gemslr_setup_check.ex
}

build_ilu_solve_state_check() {
  run "${make_cmd[@]}" -C "${par_dir}" \
    USING_CUDA=0 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" \
    ilu_solve_state_check.ex
}

seq_workdir="${workdir}/sequential"
mkdir -p "${seq_workdir}"
printf '1\n%s %s %s 0.0 0.01 -0.01 0.0\n' "${smoke_n}" "${smoke_n}" "${smoke_n}" > "${seq_workdir}/lapfile_real"
par_lapfile="${workdir}/parallel_lapfile_real"
printf '1\n%s %s %s 0.0 0.01 -0.01 0.0\n' "${smoke_n}" "${smoke_n}" "${smoke_n}" > "${par_lapfile}"

log "Building CPU library and real Laplacian drivers"
run "${make_cmd[@]}" -C "${seq_dir}" clean
run "${make_cmd[@]}" -C "${par_dir}" clean
run "${make_cmd[@]}" -C "${project_dir}" clean
run "${make_cmd[@]}" -C "${project_dir}" USING_CUDA=0 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}"
run "${make_cmd[@]}" -C "${seq_dir}" USING_CUDA=0 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" driver_laplacian_gemslr_seq.ex
run "${make_cmd[@]}" -C "${par_dir}" USING_CUDA=0 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" driver_laplacian_gemslr_par.ex

build_parallel_vector_setup_check
run_plain_case "parallel vector Setup offsets" "${workdir}" "${workdir}/parallel_vector_setup.log" \
  mpi_run "${par_ranks}" "${par_dir}/parallel_vector_setup_check.ex"


build_parallel_matrix_ops_check
run_plain_case "parallel matrix structural operations" "${workdir}" "${workdir}/parallel_matrix_ops.log" \
  mpi_run "${par_ranks}" "${par_dir}/parallel_matrix_ops_check.ex"

build_parallel_gemslr_setup_check
run_plain_case "parallel GeMSLR repeated setup" "${workdir}" "${workdir}/parallel_gemslr_setup.log" \
  mpi_run "${setup_check_ranks}" "${par_dir}/parallel_gemslr_setup_check.ex"

build_ilu_solve_state_check
run_plain_case "ILU solve state" "${workdir}" "${workdir}/ilu_solve_state.log" \
  mpi_run 1 "${par_dir}/ilu_solve_state_check.ex"

sequential_cpu_cmd=(mpi_run "${seq_ranks}" "${seq_dir}/driver_laplacian_gemslr_seq.ex"
  -fromfile "${seq_dir}/inputs" -solone -maxits "${smoke_maxits}" -kdim "${smoke_kdim}" -tol "${smoke_tol}"
  -nthreads "${smoke_threads}")
if [[ "${write_sol}" == "1" ]]; then
  sequential_cpu_cmd+=(-writesol "${workdir}/sequential_cpu")
fi
run_and_check "sequential CPU Laplacian" "${seq_workdir}" "${workdir}/sequential_cpu.log" \
  "${sequential_cpu_cmd[@]}"

parallel_cpu_cmd=(mpi_run "${par_ranks}" "${par_dir}/driver_laplacian_gemslr_par.ex"
  -lapfile "${par_lapfile}" -fromfile "${par_dir}/inputs" -solone
  -maxits "${smoke_maxits}" -kdim "${smoke_kdim}" -tol "${smoke_tol}"
  -nthreads "${smoke_threads}")
if [[ "${write_sol}" == "1" ]]; then
  parallel_cpu_cmd+=(-writesol "${workdir}/parallel_cpu")
fi
run_and_check "parallel CPU Laplacian" "${workdir}" "${workdir}/parallel_cpu.log" \
  "${parallel_cpu_cmd[@]}"

build_parallel_error_checks_release
run_plain_case "parallel vector Setup offsets (release)" "${workdir}" "${workdir}/parallel_vector_setup_release.log" \
  mpi_run "${par_ranks}" "${par_dir}/parallel_vector_setup_check.ex"
run_plain_case "parallel matrix structural operations (release)" "${workdir}" "${workdir}/parallel_matrix_ops_release.log" \
  mpi_run "${par_ranks}" "${par_dir}/parallel_matrix_ops_check.ex"
run_plain_case "parallel GeMSLR setup error handling (release)" "${workdir}" "${workdir}/parallel_gemslr_setup_release.log" \
  mpi_run "${setup_check_ranks}" "${par_dir}/parallel_gemslr_setup_check.ex" --setup-error-only

if [[ "${PARGEMSLR_TEST_CUDA:-0}" == "1" ]]; then
  cuda_arch="${CUDA_ARCH:-86}"
  cuda_version="${CUDA_VERSION:-11}"
  log "Building CUDA library and parallel real Laplacian driver"
  run "${make_cmd[@]}" -C "${par_dir}" clean
  run "${make_cmd[@]}" -C "${project_dir}" clean
  run "${make_cmd[@]}" -C "${project_dir}" USING_CUDA=1 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" CUDA_ARCH="${cuda_arch}" CUDA_VERSION="${cuda_version}"
  run "${make_cmd[@]}" -C "${par_dir}" USING_CUDA=1 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" CUDA_ARCH="${cuda_arch}" CUDA_VERSION="${cuda_version}" driver_laplacian_gemslr_par.ex
  run "${make_cmd[@]}" -C "${par_dir}" USING_CUDA=1 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" CUDA_ARCH="${cuda_arch}" CUDA_VERSION="${cuda_version}" parallel_vector_setup_check.ex
  run_plain_case "parallel vector Setup offsets (CUDA build)" "${workdir}" "${workdir}/parallel_vector_setup_cuda.log" \
    mpi_run "${par_ranks}" "${par_dir}/parallel_vector_setup_check.ex"
  run "${make_cmd[@]}" -C "${par_dir}" USING_CUDA=1 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" CUDA_ARCH="${cuda_arch}" CUDA_VERSION="${cuda_version}" parallel_matrix_ops_check.ex
  run_plain_case "parallel matrix structural operations (CUDA build)" "${workdir}" "${workdir}/parallel_matrix_ops_cuda.log" \
    mpi_run "${par_ranks}" "${par_dir}/parallel_matrix_ops_check.ex"
  run "${make_cmd[@]}" -C "${par_dir}" USING_CUDA=1 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" CUDA_ARCH="${cuda_arch}" CUDA_VERSION="${cuda_version}" parallel_gemslr_setup_check.ex
  run_plain_case "parallel GeMSLR repeated setup (CUDA build)" "${workdir}" "${workdir}/parallel_gemslr_setup_cuda.log" \
    mpi_run "${setup_check_ranks}" "${par_dir}/parallel_gemslr_setup_check.ex" --device
  run "${make_cmd[@]}" -C "${par_dir}" USING_CUDA=1 USING_MKL="${using_mkl}" USING_OPENMP="${using_openmp}" CUDA_ARCH="${cuda_arch}" CUDA_VERSION="${cuda_version}" ilu_solve_state_check.ex
  run_plain_case "ILU solve state (CUDA build)" "${workdir}" "${workdir}/ilu_solve_state_cuda.log" \
    mpi_run 1 "${par_dir}/ilu_solve_state_check.ex"

  parallel_cuda_cmd=(mpi_run "${par_ranks}" "${par_dir}/driver_laplacian_gemslr_par.ex"
    -gpu -lapfile "${par_lapfile}" -fromfile "${par_dir}/inputs" -solone
    -maxits "${smoke_maxits}" -kdim "${smoke_kdim}" -tol "${smoke_tol}"
    -nthreads "${smoke_threads}")
  if [[ "${write_sol}" == "1" ]]; then
    parallel_cuda_cmd+=(-writesol "${workdir}/parallel_cuda")
  fi
  run_and_check "parallel CUDA Laplacian" "${workdir}" "${workdir}/parallel_cuda.log" \
    "${parallel_cuda_cmd[@]}"
fi

log "Smoke tests passed"
