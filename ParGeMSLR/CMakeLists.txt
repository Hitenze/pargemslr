option(PARGEMSLR_ENABLE_OPENMP "Enable OpenMP parallel regions" OFF)
option(PARGEMSLR_ENABLE_BLAS "Use BLAS kernels where available" OFF)
option(PARGEMSLR_ENABLE_OPENBLAS "Use OpenBLAS as the BLAS/LAPACK provider" OFF)
option(PARGEMSLR_ENABLE_MKL "Use Intel oneAPI MKL as the BLAS/LAPACK provider" OFF)
option(PARGEMSLR_ENABLE_CUDA "Enable CUDA kernels" OFF)
option(PARGEMSLR_ENABLE_HYPRE "Build the hypre adapter and integration driver" OFF)
option(PARGEMSLR_ENABLE_TIMING "Collect detailed runtime timing" ON)
option(PARGEMSLR_ENABLE_DEBUG_CHECKS "Enable internal debug assertions" OFF)
option(PARGEMSLR_ENABLE_DEBUG_MEMORY "Track memory allocations and locations" OFF)
option(PARGEMSLR_BUILD_TESTS "Build ParGeMSLR test drivers" ${BUILD_TESTING})
set(PARGEMSLR_HYPRE_PROVIDER "bundled" CACHE STRING
  "hypre provider when PARGEMSLR_ENABLE_HYPRE is enabled: bundled or system")
set_property(CACHE PARGEMSLR_HYPRE_PROVIDER PROPERTY STRINGS bundled system)

if(PARGEMSLR_ENABLE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures for ParGeMSLR")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(PARGEMSLR_ENABLE_HYPRE AND NOT PARGEMSLR_BUILD_TESTS)
  message(FATAL_ERROR "PARGEMSLR_ENABLE_HYPRE requires PARGEMSLR_BUILD_TESTS")
endif()

if(PARGEMSLR_ENABLE_HYPRE)
  if(DEFINED HYPRE_ENABLE_BIGINT AND HYPRE_ENABLE_BIGINT)
    message(FATAL_ERROR
      "The ParGeMSLR hypre adapter does not support HYPRE_ENABLE_BIGINT")
  endif()
  if(DEFINED HYPRE_ENABLE_LONG_DOUBLE AND HYPRE_ENABLE_LONG_DOUBLE)
    message(FATAL_ERROR
      "The ParGeMSLR hypre adapter does not support HYPRE_ENABLE_LONG_DOUBLE")
  endif()

  if(PARGEMSLR_HYPRE_PROVIDER STREQUAL "bundled")
    if(CMAKE_VERSION VERSION_LESS 3.21)
      message(FATAL_ERROR "Bundled hypre v3.1.0 requires CMake 3.21 or newer")
    endif()
    set(_pargemslr_hypre_source "${CMAKE_CURRENT_SOURCE_DIR}/external/hypre/src")
    if(NOT EXISTS "${_pargemslr_hypre_source}/CMakeLists.txt")
      message(FATAL_ERROR
        "Bundled hypre is unavailable. Run 'git submodule update --init ParGeMSLR/external/hypre'.")
    endif()

    set(HYPRE_ENABLE_MPI ON CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_BIGINT OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_LONG_DOUBLE OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_OPENMP ${PARGEMSLR_ENABLE_OPENMP} CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_CUDA ${PARGEMSLR_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_TIMING ON CACHE BOOL "" FORCE)
    set(HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HYPRE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
    set(HYPRE_BUILD_UMPIRE OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_UMPIRE OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_UMPIRE_HOST OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_UMPIRE_PINNED OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_UMPIRE_DEVICE OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_UMPIRE_UM OFF CACHE BOOL "" FORCE)
    set(_pargemslr_install_prefix_was_default
      "${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT}")
    set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
    add_subdirectory("${_pargemslr_hypre_source}"
      "${CMAKE_CURRENT_BINARY_DIR}/external/hypre")
    set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
      "${_pargemslr_install_prefix_was_default}")
  elseif(PARGEMSLR_HYPRE_PROVIDER STREQUAL "system")
    find_package(HYPRE 3 CONFIG REQUIRED)
  else()
    message(FATAL_ERROR
      "PARGEMSLR_HYPRE_PROVIDER must be either 'bundled' or 'system'")
  endif()

  if(NOT TARGET HYPRE::HYPRE)
    message(FATAL_ERROR "The selected hypre provider did not define HYPRE::HYPRE")
  endif()
  if(DEFINED HYPRE_ENABLE_MPI AND NOT HYPRE_ENABLE_MPI)
    message(FATAL_ERROR "The ParGeMSLR hypre adapter requires an MPI-enabled hypre")
  endif()
  if(PARGEMSLR_ENABLE_CUDA AND DEFINED HYPRE_ENABLE_CUDA AND
     NOT HYPRE_ENABLE_CUDA)
    message(FATAL_ERROR
      "PARGEMSLR_ENABLE_CUDA requires a CUDA-enabled hypre provider")
  endif()
  if(DEFINED HYPRE_ENABLE_COMPLEX AND HYPRE_ENABLE_COMPLEX)
    message(FATAL_ERROR
      "The ParGeMSLR hypre adapter does not support complex-scalar hypre builds")
  endif()
  if(DEFINED HYPRE_ENABLE_BIGINT AND HYPRE_ENABLE_BIGINT)
    message(FATAL_ERROR
      "The ParGeMSLR hypre adapter does not support BIGINT hypre builds")
  endif()
  if(DEFINED HYPRE_ENABLE_LONG_DOUBLE AND HYPRE_ENABLE_LONG_DOUBLE)
    message(FATAL_ERROR
      "The ParGeMSLR hypre adapter does not support long-double hypre builds")
  endif()
endif()

if(PARGEMSLR_ENABLE_MKL AND PARGEMSLR_ENABLE_OPENBLAS)
  message(FATAL_ERROR "PARGEMSLR_ENABLE_MKL and PARGEMSLR_ENABLE_OPENBLAS are mutually exclusive")
endif()

if(PARGEMSLR_ENABLE_DEBUG_MEMORY AND NOT PARGEMSLR_ENABLE_DEBUG_CHECKS)
  message(FATAL_ERROR "PARGEMSLR_ENABLE_DEBUG_MEMORY requires PARGEMSLR_ENABLE_DEBUG_CHECKS")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)

if(NOT ParMETIS_ROOT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/parmetis")
  set(ParMETIS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/parmetis" CACHE PATH
    "Root directory of a ParMETIS installation or source build" FORCE)
endif()
find_package(ParMETIS REQUIRED)

if(PARGEMSLR_ENABLE_MKL)
  if(DEFINED ENV{MKLROOT})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{MKLROOT}")
  endif()
  set(MKL_LINK dynamic CACHE STRING "oneAPI MKL link type")
  set(MKL_INTERFACE lp64 CACHE STRING "oneAPI MKL integer interface")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_pargemslr_mkl_threading gnu_thread)
  else()
    set(_pargemslr_mkl_threading intel_thread)
  endif()
  set(MKL_THREADING "${_pargemslr_mkl_threading}" CACHE STRING "oneAPI MKL threading layer")
  find_package(MKL CONFIG REQUIRED)
else()
  if(PARGEMSLR_ENABLE_OPENBLAS)
    set(BLA_VENDOR OpenBLAS)
  else()
    set(BLA_VENDOR Generic)
  endif()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

set(PARGEMSLR_SOURCES
  SRC/utils/utils.cpp
  SRC/utils/parallel.cpp
  SRC/utils/memory.cpp
  SRC/utils/mmio.cpp
  SRC/vectors/vector.cpp
  SRC/vectors/sequential_vector.cpp
  SRC/vectors/int_vector.cpp
  SRC/vectors/parallel_vector.cpp
  SRC/vectors/vectorops.cpp
  SRC/matrices/matrix.cpp
  SRC/matrices/dense_matrix.cpp
  SRC/matrices/coo_matrix.cpp
  SRC/matrices/csr_matrix.cpp
  SRC/matrices/parallel_csr_matrix.cpp
  SRC/matrices/dense_matrix_ops.cpp
  SRC/matrices/csr_matrix_ops.cpp
  SRC/matrices/csr_matrix_partition.cpp
  SRC/matrices/csr_matrix_ordering.cpp
  SRC/matrices/matrix_arnoldi_ops.cpp
  SRC/matrices/matrix_io_plot_ops.cpp
  SRC/matrices/csr_matrix_permutation.cpp
  SRC/matrices/parallel_csr_matrix_permutation.cpp
  SRC/preconditioners/ilu.cpp
  SRC/preconditioners/gemslr.cpp
  SRC/preconditioners/block_jacobi.cpp
  SRC/preconditioners/parallel_gemslr.cpp)

if(PARGEMSLR_ENABLE_CUDA)
  list(APPEND PARGEMSLR_SOURCES
    SRC/utils/utils_cuda.cu
    SRC/vectors/int_vector_cuda.cu
    SRC/vectors/sequential_vector_cuda.cu
    SRC/vectors/vectorops_cuda.cu
    SRC/matrices/coo_matrix_cuda.cu
    SRC/matrices/csr_matrix_cuda.cu
    SRC/matrices/dense_matrix_cuda.cu
    SRC/matrices/matrixops_cuda.cu
    SRC/preconditioners/ilu_cuda.cu
    SRC/preconditioners/gemslr_cuda.cu)
  set_source_files_properties(${PARGEMSLR_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_library(pargemslr STATIC ${PARGEMSLR_SOURCES})
add_library(ParGeMSLR::pargemslr ALIAS pargemslr)

target_compile_features(pargemslr PUBLIC cxx_std_11)
target_compile_definitions(pargemslr
  PUBLIC
    UNIX
    $<$<BOOL:${PARGEMSLR_ENABLE_OPENMP}>:PARGEMSLR_OPENMP>
    $<$<BOOL:${PARGEMSLR_ENABLE_BLAS}>:PARGEMSLR_BLAS>
    $<$<BOOL:${PARGEMSLR_ENABLE_OPENBLAS}>:PARGEMSLR_OPENBLAS>
    $<$<BOOL:${PARGEMSLR_ENABLE_MKL}>:PARGEMSLR_MKL>
    $<$<BOOL:${PARGEMSLR_ENABLE_CUDA}>:PARGEMSLR_CUDA>
    $<$<BOOL:${PARGEMSLR_ENABLE_TIMING}>:PARGEMSLR_TIMING>
    $<$<BOOL:${PARGEMSLR_ENABLE_DEBUG_CHECKS}>:PARGEMSLR_DEBUG>
    $<$<BOOL:${PARGEMSLR_ENABLE_DEBUG_MEMORY}>:PARGEMSLR_DEBUG_MEMORY>)

if(PARGEMSLR_ENABLE_DEBUG_CHECKS)
  if(MSVC)
    target_compile_options(pargemslr PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:/UNDEBUG>
      $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
  else()
    target_compile_options(pargemslr PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>
      $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
  endif()
endif()

target_include_directories(pargemslr
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/INC>)

target_link_libraries(pargemslr
  PUBLIC
    MPI::MPI_CXX
    ParMETIS::ParMETIS)

if(PARGEMSLR_ENABLE_MKL)
  target_link_libraries(pargemslr PUBLIC MKL::MKL)
else()
  target_link_libraries(pargemslr PUBLIC BLAS::BLAS LAPACK::LAPACK)
endif()

if(PARGEMSLR_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(pargemslr PUBLIC OpenMP::OpenMP_CXX)
  if(PARGEMSLR_ENABLE_CUDA)
    separate_arguments(_pargemslr_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
    foreach(flag IN LISTS _pargemslr_openmp_flags)
      target_compile_options(pargemslr PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
    endforeach()
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  target_compile_options(pargemslr PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
endif()

set_target_properties(pargemslr PROPERTIES
  OUTPUT_NAME pargemslr
  POSITION_INDEPENDENT_CODE ON)

if(PARGEMSLR_ENABLE_CUDA)
  target_compile_features(pargemslr PUBLIC cuda_std_11)
  target_compile_definitions(pargemslr PUBLIC
    PARGEMSLR_CUDA_VERSION=${CUDAToolkit_VERSION_MAJOR})
  target_compile_options(pargemslr PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>)
  target_link_libraries(pargemslr PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::curand)
  set_target_properties(pargemslr PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON)
endif()

if(PARGEMSLR_BUILD_TESTS)
  add_subdirectory(TESTS)
endif()
