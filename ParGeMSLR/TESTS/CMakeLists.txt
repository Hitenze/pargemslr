function(pargemslr_add_test_executable target output_dir)
  if(PARGEMSLR_ENABLE_CUDA)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
  endif()
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE ParGeMSLR::pargemslr)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${output_dir}"
    SUFFIX ".ex")
  if(PARGEMSLR_ENABLE_CUDA)
    set_target_properties(${target} PROPERTIES
      CUDA_STANDARD 11
      CUDA_STANDARD_REQUIRED ON)
  endif()
endfunction()

pargemslr_add_test_executable(driver_laplacian_gemslr_seq sequential
  sequential/driver_laplacian_gemslr_seq.cpp
  sequential/io.cpp)
pargemslr_add_test_executable(driver_laplacian_gemslrz_seq sequential
  sequential/driver_laplacian_gemslrz_seq.cpp
  sequential/io.cpp)
pargemslr_add_test_executable(driver_gen_gemslr_seq sequential
  sequential/driver_gen_gemslr_seq.cpp
  sequential/io.cpp)
pargemslr_add_test_executable(driver_gen_gemslrz_seq sequential
  sequential/driver_gen_gemslrz_seq.cpp
  sequential/io.cpp)

pargemslr_add_test_executable(driver_laplacian_gemslr_par parallel
  parallel/driver_laplacian_gemslr_par.cpp
  parallel/io_par.cpp)
pargemslr_add_test_executable(driver_laplacian_gemslrz_par parallel
  parallel/driver_laplacian_gemslrz_par.cpp
  parallel/io_par.cpp)
pargemslr_add_test_executable(driver_gen_gemslr_par parallel
  parallel/driver_gen_gemslr_par.cpp
  parallel/io_par.cpp)
pargemslr_add_test_executable(driver_gen_gemslrz_par parallel
  parallel/driver_gen_gemslrz_par.cpp
  parallel/io_par.cpp)

pargemslr_add_test_executable(parallel_vector_setup_check parallel
  parallel_vector_setup_check.cpp)
pargemslr_add_test_executable(parallel_matrix_ops_check parallel
  parallel_matrix_ops_check.cpp)
pargemslr_add_test_executable(parallel_gemslr_setup_check parallel
  parallel_gemslr_setup_check.cpp)
pargemslr_add_test_executable(ilu_solve_state_check parallel
  ilu_solve_state_check.cpp)

if(PARGEMSLR_ENABLE_HYPRE)
  add_subdirectory(hypre)
endif()

if(BUILD_TESTING)
  set(pargemslr_test_runtime_dir "${CMAKE_CURRENT_BINARY_DIR}/runtime")
  file(MAKE_DIRECTORY "${pargemslr_test_runtime_dir}/sequential")
  file(WRITE "${pargemslr_test_runtime_dir}/sequential/lapfile_real"
    "1\n16 16 16 0.0 0.01 -0.01 0.0\n")
  file(WRITE "${pargemslr_test_runtime_dir}/parallel_lapfile_real"
    "1\n16 16 16 0.0 0.01 -0.01 0.0\n")

  function(pargemslr_add_mpi_test name ranks target)
    add_test(NAME ${name}
      COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} ${ranks}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${target}>
        ${MPIEXEC_POSTFLAGS}
        ${ARGN})
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT
        "OMP_NUM_THREADS=1;OMP_DYNAMIC=FALSE;MKL_DYNAMIC=FALSE;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
      PROCESSORS ${ranks}
      TIMEOUT 120)
  endfunction()

  function(pargemslr_add_solver_test name ranks threads target working_dir)
    if(PARGEMSLR_ENABLE_OPENMP)
      set(expected_openmp_threads ${threads})
    else()
      set(expected_openmp_threads 0)
    endif()
    math(EXPR processors "${ranks} * ${threads}")
    add_test(NAME ${name}
      COMMAND
        "${CMAKE_CURRENT_SOURCE_DIR}/check_solver_output.sh"
        "${name}" 1e-10 1e-10 1e-12 100 ${expected_openmp_threads} --
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} ${ranks}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${target}>
        ${MPIEXEC_POSTFLAGS}
        ${ARGN})
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT
        "OMP_NUM_THREADS=${threads};OMP_DYNAMIC=FALSE;MKL_DYNAMIC=FALSE;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
      PROCESSORS ${processors}
      TIMEOUT 180
      WORKING_DIRECTORY "${working_dir}")
  endfunction()

  pargemslr_add_mpi_test(parallel_vector_setup 2 parallel_vector_setup_check)
  pargemslr_add_mpi_test(parallel_matrix_ops 2 parallel_matrix_ops_check)
  pargemslr_add_mpi_test(parallel_gemslr_setup 2 parallel_gemslr_setup_check)
  pargemslr_add_mpi_test(ilu_solve_state 1 ilu_solve_state_check)

  if(NOT PARGEMSLR_ENABLE_DEBUG_CHECKS)
    pargemslr_add_mpi_test(parallel_gemslr_release_errors 2
      parallel_gemslr_setup_check --setup-error-only)
  endif()

  pargemslr_add_solver_test(sequential_cpu_laplacian 1 1
    driver_laplacian_gemslr_seq
    "${pargemslr_test_runtime_dir}/sequential"
    -fromfile "${CMAKE_CURRENT_SOURCE_DIR}/sequential/inputs"
    -solone -maxits 100 -kdim 40 -tol 1e-10 -nthreads 1)

  foreach(ranks IN ITEMS 1 2 4)
    pargemslr_add_solver_test(parallel_cpu_laplacian_r${ranks} ${ranks} 1
      driver_laplacian_gemslr_par
      "${pargemslr_test_runtime_dir}"
      -lapfile "${pargemslr_test_runtime_dir}/parallel_lapfile_real"
      -fromfile "${CMAKE_CURRENT_SOURCE_DIR}/parallel/inputs"
      -solone -maxits 100 -kdim 40 -tol 1e-10 -nthreads 1)
  endforeach()

  if(PARGEMSLR_ENABLE_OPENMP)
    pargemslr_add_solver_test(sequential_cpu_laplacian_t2 1 2
      driver_laplacian_gemslr_seq
      "${pargemslr_test_runtime_dir}/sequential"
      -fromfile "${CMAKE_CURRENT_SOURCE_DIR}/sequential/inputs"
      -solone -maxits 100 -kdim 40 -tol 1e-10 -nthreads 2)

    foreach(threads IN ITEMS 2 4)
      pargemslr_add_solver_test(parallel_cpu_laplacian_r2_t${threads} 2 ${threads}
        driver_laplacian_gemslr_par
        "${pargemslr_test_runtime_dir}"
        -lapfile "${pargemslr_test_runtime_dir}/parallel_lapfile_real"
        -fromfile "${CMAKE_CURRENT_SOURCE_DIR}/parallel/inputs"
        -solone -maxits 100 -kdim 40 -tol 1e-10 -nthreads ${threads})
    endforeach()
  endif()

  if(PARGEMSLR_ENABLE_CUDA)
    pargemslr_add_mpi_test(parallel_gemslr_setup_cuda 2
      parallel_gemslr_setup_check --device)
    pargemslr_add_solver_test(parallel_cuda_laplacian_r2 2 1
      driver_laplacian_gemslr_par
      "${pargemslr_test_runtime_dir}"
      -gpu
      -lapfile "${pargemslr_test_runtime_dir}/parallel_lapfile_real"
      -fromfile "${CMAKE_CURRENT_SOURCE_DIR}/parallel/inputs"
      -solone -maxits 100 -kdim 40 -tol 1e-10 -nthreads 1)
  endif()
endif()
