#!/usr/bin/env bash
set -eu

mode="${1:-cpu}"
case "${mode}" in
   cpu|cuda) ;;
   *)
      echo "Usage: $0 [cpu|cuda]" >&2
      exit 2
      ;;
esac

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_dir="$(CDPATH= cd -- "${script_dir}/.." && pwd)"
hypre_dir="${script_dir}/external/hypre"
hypre_src="${hypre_dir}/src"
build_dir="${script_dir}/external/hypre-build/${mode}"
work_src="${build_dir}/src"

if [ ! -d "${hypre_src}" ]; then
   git -C "${repo_dir}" submodule update --init --recursive ParGeMSLR/external/hypre
fi

rm -rf "${build_dir}"
mkdir -p "${build_dir}"
cp -a "${hypre_src}" "${work_src}"
cd "${work_src}"

hypre_cc="${HYPRE_CC:-${CC:-mpicc}}"
hypre_cxx="${HYPRE_CXX:-${CXX:-mpicxx}}"

config_args=()
if [ "${mode}" = "cuda" ]; then
   cuda_arch="${CUDA_ARCH:-86}"
   config_args+=(--with-cuda --with-gpu-arch="${cuda_arch}")
   if [ -n "${CUDA_HOME:-}" ]; then
      config_args+=(--with-cuda-home="${CUDA_HOME}")
   fi
   if [ "${HYPRE_UNIFIED_MEMORY:-0}" != "0" ]; then
      config_args+=(--enable-unified-memory)
   fi
   if [ "${HYPRE_GPU_AWARE_MPI:-0}" != "0" ]; then
      config_args+=(--enable-gpu-aware-mpi)
   fi
   if [ "${HYPRE_WITH_UMPIRE:-0}" != "0" ]; then
      if [ -n "${HYPRE_UMPIRE_INCLUDE:-}" ]; then
         config_args+=(--with-umpire-include="${HYPRE_UMPIRE_INCLUDE}")
      fi
      if [ -n "${HYPRE_UMPIRE_LIB_DIRS:-}" ]; then
         config_args+=(--with-umpire-lib-dirs="${HYPRE_UMPIRE_LIB_DIRS}")
      fi
      if [ -n "${HYPRE_UMPIRE_LIBS:-}" ]; then
         config_args+=(--with-umpire-libs="${HYPRE_UMPIRE_LIBS}")
      fi
   else
      config_args+=(--without-umpire)
   fi
fi

env CC="${hypre_cc}" CXX="${hypre_cxx}" ./configure "${config_args[@]}"
make -j "${HYPRE_BUILD_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
