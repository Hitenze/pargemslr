set(pargemslr_hypre_interface_sources
  pargemslr_hypre_interface.cpp
  HYPRE_pargemslr.cpp)

if(PARGEMSLR_ENABLE_CUDA)
  set_source_files_properties(
    ${pargemslr_hypre_interface_sources}
    ij.cpp
    PROPERTIES LANGUAGE CUDA)
endif()

add_library(pargemslr_hypre_interface STATIC
  ${pargemslr_hypre_interface_sources})
add_library(ParGeMSLR::hypre_interface ALIAS pargemslr_hypre_interface)
target_compile_features(pargemslr_hypre_interface PUBLIC cxx_std_11)
target_compile_definitions(pargemslr_hypre_interface PRIVATE
  NOFEI
  HYPRE_TIMING
  MPICH_SKIP_MPICXX)
target_include_directories(pargemslr_hypre_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(pargemslr_hypre_interface PUBLIC
  ParGeMSLR::pargemslr
  HYPRE::HYPRE)

add_executable(pargemslr_hypre_ij ij.cpp)
target_compile_features(pargemslr_hypre_ij PRIVATE cxx_std_11)
target_compile_definitions(pargemslr_hypre_ij PRIVATE
  NOFEI
  HYPRE_TIMING
  MPICH_SKIP_MPICXX)
target_link_libraries(pargemslr_hypre_ij PRIVATE
  ParGeMSLR::hypre_interface)
set_target_properties(pargemslr_hypre_ij PROPERTIES
  OUTPUT_NAME ij
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

if(PARGEMSLR_ENABLE_CUDA)
  set_target_properties(
    pargemslr_hypre_interface
    pargemslr_hypre_ij
    PROPERTIES
      CUDA_STANDARD 11
      CUDA_STANDARD_REQUIRED ON)
endif()

if(BUILD_TESTING)
  set(pargemslr_hypre_runtime_dir "${CMAKE_CURRENT_BINARY_DIR}/runtime")
  file(MAKE_DIRECTORY "${pargemslr_hypre_runtime_dir}")
  configure_file(inputs "${pargemslr_hypre_runtime_dir}/inputs" COPYONLY)

  function(pargemslr_add_hypre_test name ranks execution_policy)
    add_test(NAME ${name}
      COMMAND
        "${CMAKE_CURRENT_SOURCE_DIR}/check_hypre_output.sh"
        "${name}" 1e-10 100 --
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} ${ranks}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:pargemslr_hypre_ij>
        ${MPIEXEC_POSTFLAGS}
        -solver 83
        -n 8 8 8
        -shift 0.5
        -k 50
        -tol 1e-10
        -max_iter 100
        ${execution_policy})
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT
        "OMP_NUM_THREADS=1;OMP_DYNAMIC=FALSE;MKL_DYNAMIC=FALSE;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
      PROCESSORS ${ranks}
      TIMEOUT 240
      WORKING_DIRECTORY "${pargemslr_hypre_runtime_dir}")
  endfunction()

  pargemslr_add_hypre_test(hypre_cpu_r1 1 -exec_host)
  pargemslr_add_hypre_test(hypre_cpu_r2 2 -exec_host)
  if(PARGEMSLR_ENABLE_CUDA)
    pargemslr_add_hypre_test(hypre_cuda_r1 1 -exec_device)
    pargemslr_add_hypre_test(hypre_cuda_r2 2 -exec_device)
  endif()
endif()
