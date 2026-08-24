# Building with CMake

CMake is an alternative to the existing Makefile build; both paths remain
supported. Configure from the repository root and use an out-of-source build
directory.

## Requirements

- CMake 3.18 or newer; bundled hypre requires CMake 3.21 or newer.
- A C++11 compiler and MPI C++ implementation.
- BLAS, LAPACK, and a 64-bit-index ParMETIS build.
- CUDA Toolkit when `PARGEMSLR_ENABLE_CUDA=ON`.

If ParMETIS is not under `ParGeMSLR/parmetis`, pass its prefix with
`-DParMETIS_ROOT=/path/to/parmetis`.

## CPU Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DParMETIS_ROOT=/path/to/parmetis
cmake --build build --parallel
cd build && ctest --output-on-failure
```

The default provider is generic BLAS/LAPACK. Enable OpenMP with
`-DPARGEMSLR_ENABLE_OPENMP=ON`. For oneAPI MKL with BLAS kernels enabled:

```bash
cmake -S . -B build-mkl -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx \
  -DPARGEMSLR_ENABLE_MKL=ON \
  -DPARGEMSLR_ENABLE_BLAS=ON \
  -DPARGEMSLR_ENABLE_OPENMP=ON
```

Use `PARGEMSLR_ENABLE_OPENBLAS=ON` to select OpenBLAS instead. MKL and
OpenBLAS are mutually exclusive.

## CUDA Build

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DPARGEMSLR_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --parallel
cd build-cuda && ctest --output-on-failure
```

For a versioned toolkit installation, also set `CMAKE_CUDA_COMPILER` and
`CUDAToolkit_ROOT`, for example `/usr/local/cuda-12.8/bin/nvcc` and
`/usr/local/cuda-12.8`.

## hypre Interface

The default hypre provider builds the repository's pinned v3.1.0 submodule:

```bash
git submodule update --init ParGeMSLR/external/hypre
cmake -S . -B build-hypre -DCMAKE_BUILD_TYPE=Release \
  -DPARGEMSLR_ENABLE_HYPRE=ON
cmake --build build-hypre --parallel
cd build-hypre && ctest -R hypre --output-on-failure
```

Bundled CUDA hypre follows the existing `buildhypre.sh` baseline and disables
Umpire. Add `PARGEMSLR_ENABLE_CUDA=ON` and the CUDA architecture options above
to test host and device execution.

To consume an installed hypre v3 package instead:

```bash
cmake -S . -B build-hypre-system \
  -DPARGEMSLR_ENABLE_HYPRE=ON \
  -DPARGEMSLR_HYPRE_PROVIDER=system \
  -DHYPRE_DIR=/path/to/lib/cmake/HYPRE
```

The system package must provide `HYPRE::HYPRE` and enable MPI. A CUDA
ParGeMSLR build also requires a CUDA-enabled hypre package. BIGINT,
long-double, and complex-scalar hypre builds are not supported by the current
adapter.

## Common Options

| Option | Default | Purpose |
| --- | --- | --- |
| `PARGEMSLR_ENABLE_TIMING` | `ON` | Collect detailed runtime timing. |
| `PARGEMSLR_ENABLE_DEBUG_CHECKS` | `OFF` | Enable internal debug assertions, including in optimized builds. |
| `PARGEMSLR_ENABLE_DEBUG_MEMORY` | `OFF` | Track allocations; requires debug checks. |
| `PARGEMSLR_BUILD_TESTS` | `BUILD_TESTING` | Build drivers and regression checks. |

Set `BUILD_TESTING=OFF` for a library-only build. Build directories and CMake
generated files are ignored by Git.
