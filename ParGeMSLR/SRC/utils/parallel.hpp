#ifndef PARGEMSLR_PARALLEL_H
#define PARGEMSLR_PARALLEL_H

/**
 * @file parallel.hpp
 * @brief Parallel related data structures and functions.
 */

#include <assert.h>
#include <mpi.h>
#include <vector>
#ifdef PARGEMSLR_OPENMP
#include "omp.h"
#endif
#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

#include "utils.hpp"

using namespace std;

/*- - - - - - - - - Timing information */

#define PARGEMSLR_TIMES_NUM             60

#define PARGEMSLR_BUILDTIME_PARTITION   0  // time for the entire partitioning
#define PARGEMSLR_BUILDTIME_IE          1  // time for split the interior and external nodes
#define PARGEMSLR_BUILDTIME_METIS       2  // time for calling the METIS
#define PARGEMSLR_BUILDTIME_STRUCTURE   3  // time for building the structure when having the domain number
#define PARGEMSLR_BUILDTIME_RCM         4  // time for applying the RCM ordering
#define PARGEMSLR_BUILDTIME_ILUT        5  // time for the ILUT factorization
#define PARGEMSLR_BUILDTIME_LRC         6  // time for building the low-rank correction
#define PARGEMSLR_BUILDTIME_ARNOLDI     7  // time for the standard arnoldi
#define PARGEMSLR_BUILDTIME_BUILD_RES   8  // time for phase the result of arnoldi
#define PARGEMSLR_BUILDTIME_SOLVELU     9  // time for the ILU solve in the setup phase
#define PARGEMSLR_BUILDTIME_SOLVELU_L   10 // time for the ILU solve in the setup phase on the last level
#define PARGEMSLR_BUILDTIME_SOLVELR     11 // time for applying the low-rank correction in the setup phase
#define PARGEMSLR_BUILDTIME_SOLVEEBFC   12 // time for solve with EBiFCi
#define PARGEMSLR_BUILDTIME_EXTRACTMAT  13 // time for extracting E, B, F, and C on the first level
#define PARGEMSLR_BUILDTIME_MOVEDATA    14 // time for moving data between levels
#define PARGEMSLR_BUILDTIME_LOCALPERM   15 // time for local permutation
#define PARGEMSLR_BUILDTIME_EMV         16 // time for matvec with E on all levels
#define PARGEMSLR_BUILDTIME_FMV         17 // time for matvec with F on all levels
#define PARGEMSLR_BUILDTIME_GEN_MAT     18 // time for generating matrix
#define PARGEMSLR_BUILDTIME_DECOMP      19 // time for decompositions. Hess, schur, eig, ordschur...
#define PARGEMSLR_BUILDTIME_MGS         20 // time for MGS in the Arnoldi
#define PARGEMSLR_BUILDTIME_EBFC        21 // time for EBFC in the Arnoldi
#define PARGEMSLR_PRECTIME_PRECOND      30 // time for applying the preconditioner
#define PARGEMSLR_PRECTIME_ILUT         31 // time for the ILU solve in the solve phase
#define PARGEMSLR_PRECTIME_ILUT_L       32 // time for the ILU solve in the solve phase on the last level
#define PARGEMSLR_PRECTIME_LRC          33 // time for applying the low-rank correction
#define PARGEMSLR_PRECTIME_EMV          34 // time for matvec with E on all levels
#define PARGEMSLR_PRECTIME_FMV          35 // time for matvec with F on all levels
#define PARGEMSLR_PRECTIME_INNER        36 // time for the inner iteration
#define PARGEMSLR_PRECTIME_MOVEDATA     37 // time for moving data between levels
#define PARGEMSLR_PRECTIME_LOCALPERM    38 // time for moving data between levels
#define PARGEMSLR_ITERTIME_AMV          40 // time for matvec with A
#define PARGEMSLR_ITERTIME_MGS          41 // time for MGS during solve
#define PARGEMSLR_TOTAL_GEN_MAT_TIME    50 // time for transfering data to device
#define PARGEMSLR_TOTAL_SETUP_TIME      51 // time for the setup phase
#define PARGEMSLR_TOTAL_SOLVE_TIME      52 // time for the solve phase
#define PARGEMSLR_BUILDTIME_BMV         53 // time for matvec with B on all levels
#define PARGEMSLR_BUILDTIME_CMV         54 // time for matvec with C on all levels
#define PARGEMSLR_BUILDTIME_SMV         55 // time for matvec with S on all levels
#define PARGEMSLR_PRECTIME_BMV          56 // time for matvec with B on all levels
#define PARGEMSLR_PRECTIME_CMV          57 // time for matvec with C on all levels
#define PARGEMSLR_PRECTIME_SMV          58 // time for matvec with S on all levels

#define PARGEMSLR_GLOBAL_FIRM_TIME_CALL(num, ...) {\
   pargemslr::PargemslrMpiTime( (*(pargemslr::ParallelLogClass::_gcomm)), pargemslr::ParallelLogClass::_times_buffer_start[num]);\
   (__VA_ARGS__);\
   pargemslr::PargemslrMpiTime( (*(pargemslr::ParallelLogClass::_gcomm)), pargemslr::ParallelLogClass::_times_buffer_end[num]);\
   pargemslr::ParallelLogClass::_times[num] += pargemslr::ParallelLogClass::_times_buffer_end[num] - pargemslr::ParallelLogClass::_times_buffer_start[num];\
}

#define PARGEMSLR_LOCAL_FIRM_TIME_CALL(num, ...) {\
   PARGEMSLR_CUDA_SYNCHRONIZE;\
   pargemslr::ParallelLogClass::_times_buffer_start[num] = MPI_Wtime();\
   (__VA_ARGS__);\
   PARGEMSLR_CUDA_SYNCHRONIZE;\
   pargemslr::ParallelLogClass::_times_buffer_end[num] = MPI_Wtime();\
   pargemslr::ParallelLogClass::_times[num] += pargemslr::ParallelLogClass::_times_buffer_end[num] - pargemslr::ParallelLogClass::_times_buffer_start[num];\
}

#define PARGEMSLR_FIRM_TIME_CALL(comm,num, ...) {\
   pargemslr::PargemslrMpiTime( (comm), pargemslr::ParallelLogClass::_times_buffer_start[num]);\
   (__VA_ARGS__);\
   pargemslr::PargemslrMpiTime( (comm), pargemslr::ParallelLogClass::_times_buffer_end[num]);\
   pargemslr::ParallelLogClass::_times[num] += pargemslr::ParallelLogClass::_times_buffer_end[num] - pargemslr::ParallelLogClass::_times_buffer_start[num];\
}

#ifdef PARGEMSLR_TIMING

#define PARGEMSLR_PRINT_TIMING_RESULT(print_level, ...) {\
   if(__VA_ARGS__)\
   {\
      PARGEMSLR_PRINT("\n");\
      PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);\
      PARGEMSLR_PRINT("Time info:\n");\
      PARGEMSLR_PRINT("\tLoad matrix time:   %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_GEN_MAT_TIME]);\
      PARGEMSLR_PRINT("\tPartition time:     %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_PARTITION]+pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_STRUCTURE]);\
      PARGEMSLR_PRINT("\tSetup time:         %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_SETUP_TIME]-pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_PARTITION]-pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_STRUCTURE]);\
      PARGEMSLR_PRINT("\tSolve time:         %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_SOLVE_TIME]);\
      PARGEMSLR_PRINT("\tTotal time:         %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_SETUP_TIME]+pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_SOLVE_TIME]);\
      PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);\
      PARGEMSLR_PRINT("\n");\
      if(print_level > 0)\
      {\
         PARGEMSLR_PRINT("\n");\
         PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);\
         PARGEMSLR_PRINT("Time detail:\n");\
         PARGEMSLR_PRINT("\tMatvec with A time:             %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_ITERTIME_AMV]);\
         PARGEMSLR_PRINT("\tPrecond setup time:             %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_SETUP_TIME]);\
         PARGEMSLR_PRINT("\t-GeMSLR reordering time:        %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_PARTITION]);\
         PARGEMSLR_PRINT("\t-GeMSLR Setup Structure time:   %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_STRUCTURE]);\
         PARGEMSLR_PRINT("\t-GeMSLR ILU setup time:         %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_ILUT]);\
         PARGEMSLR_PRINT("\t--GeMSLR ILU reordering time:   %fs - (note: this is the time on p0.)\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_RCM]);\
         PARGEMSLR_PRINT("\t-GeMSLR low-rank setup time:    %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_LRC]);\
         PARGEMSLR_PRINT("\t--GeMSLR arnoldi iter time:        %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_ARNOLDI]);\
         PARGEMSLR_PRINT("\t---GeMSLR MGS time:                   %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_MGS]);\
         PARGEMSLR_PRINT("\t---GeMSLR EB^{-1}FC^{-1} time:        %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_EBFC]);\
         PARGEMSLR_PRINT("\t---GeMSLR setup ILU solve time:       %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_SOLVELU]);\
         PARGEMSLR_PRINT("\t---GeMSLR setup ILU solve last lev:   %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_SOLVELU_L]);\
         PARGEMSLR_PRINT("\t---GeMSLR setup LRC apply time:       %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_SOLVELR]);\
         PARGEMSLR_PRINT("\t---GeMSLR setup sparse matvec time:   %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_EMV]+pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_FMV]);\
         PARGEMSLR_PRINT("\t--GeMSLR build result time:        %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_BUILD_RES]);\
         PARGEMSLR_PRINT("\t--GeMSLR Lapack Dcomp time:        %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_DECOMP]);\
         PARGEMSLR_PRINT("\tPrecond applying time:          %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_PRECTIME_PRECOND]);\
         PARGEMSLR_PRINT("\t-GeMSLR ILU solve time:         %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_PRECTIME_ILUT]);\
         PARGEMSLR_PRINT("\t-GeMSLR ILU solve last lev:     %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_PRECTIME_ILUT_L]);\
         PARGEMSLR_PRINT("\t-GeMSLR sparse matvec time:     %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_PRECTIME_EMV]+pargemslr::ParallelLogClass::_times[PARGEMSLR_PRECTIME_FMV]\
            +pargemslr::ParallelLogClass::_times[PARGEMSLR_PRECTIME_BMV]+pargemslr::ParallelLogClass::_times[PARGEMSLR_PRECTIME_SMV]+pargemslr::ParallelLogClass::_times[PARGEMSLR_PRECTIME_CMV]);\
         PARGEMSLR_PRINT("\t-GeMSLR LRC apply time:         %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_PRECTIME_LRC]);\
         PARGEMSLR_PRINT("\tIterative solve MGS time:       %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_ITERTIME_MGS]);\
         PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);\
         PARGEMSLR_PRINT("\n");\
      }\
   }\
}

#define PARGEMSLR_GLOBAL_TIME_CALL(num, ...) {\
   pargemslr::PargemslrMpiTime( (*(pargemslr::ParallelLogClass::_gcomm)), pargemslr::ParallelLogClass::_times_buffer_start[num]);\
   (__VA_ARGS__);\
   pargemslr::PargemslrMpiTime( (*(pargemslr::ParallelLogClass::_gcomm)), pargemslr::ParallelLogClass::_times_buffer_end[num]);\
   pargemslr::ParallelLogClass::_times[num] += pargemslr::ParallelLogClass::_times_buffer_end[num] - pargemslr::ParallelLogClass::_times_buffer_start[num];\
}

#define PARGEMSLR_LOCAL_TIME_CALL(num, ...) {\
   PARGEMSLR_CUDA_SYNCHRONIZE;\
   pargemslr::ParallelLogClass::_times_buffer_start[num] = MPI_Wtime();\
   (__VA_ARGS__);\
   PARGEMSLR_CUDA_SYNCHRONIZE;\
   pargemslr::ParallelLogClass::_times_buffer_end[num] = MPI_Wtime();\
   pargemslr::ParallelLogClass::_times[num] += pargemslr::ParallelLogClass::_times_buffer_end[num] - pargemslr::ParallelLogClass::_times_buffer_start[num];\
}

#define PARGEMSLR_TIME_CALL(comm,num, ...) {\
   pargemslr::PargemslrMpiTime( (comm), pargemslr::ParallelLogClass::_times_buffer_start[num]);\
   (__VA_ARGS__);\
   pargemslr::PargemslrMpiTime( (comm), pargemslr::ParallelLogClass::_times_buffer_end[num]);\
   pargemslr::ParallelLogClass::_times[num] += pargemslr::ParallelLogClass::_times_buffer_end[num] - pargemslr::ParallelLogClass::_times_buffer_start[num];\
}

#define PARGEMSLR_RESET_TIME std::fill(pargemslr::ParallelLogClass::_times.begin(), pargemslr::ParallelLogClass::_times.end(), 0.0);

#else

#define PARGEMSLR_GLOBAL_TIME_CALL(num, ...) {\
   (__VA_ARGS__);\
}

#define PARGEMSLR_LOCAL_TIME_CALL(num, ...) {\
   (__VA_ARGS__);\
}

#define PARGEMSLR_TIME_CALL(comm,num, ...) {\
   (__VA_ARGS__);\
}

#define PARGEMSLR_PRINT_TIMING_RESULT(print_level, ...) {\
   if(__VA_ARGS__)\
   {\
      PARGEMSLR_PRINT("\n");\
      PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);\
      PARGEMSLR_PRINT("Time info:\n");\
      PARGEMSLR_PRINT("\tLoad matrix time:   %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_GEN_MAT_TIME]);\
      PARGEMSLR_PRINT("\tPartition time:     %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_PARTITION]);\
      PARGEMSLR_PRINT("\tSetup time:         %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_SETUP_TIME]-pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_PARTITION]);\
      PARGEMSLR_PRINT("\tSolve time:         %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_SOLVE_TIME]);\
      PARGEMSLR_PRINT("\tTotal time:         %fs\n",pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_SETUP_TIME]+pargemslr::ParallelLogClass::_times[PARGEMSLR_TOTAL_SOLVE_TIME]);\
      PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);\
      PARGEMSLR_PRINT("\n");\
   }\
}

#define PARGEMSLR_RESET_TIME

#endif

namespace pargemslr
{
	/** 
    * @brief   The data structure for parallel computing, including data structures for MPI and CUDA.
    * @details The data structure for parallel computing, including data structures for MPI and CUDA. \n
    *          All CUDA information are shared, local MPI information can be different.
    */
   typedef class ParallelLogClass 
   {
      public:
      /* variables */
#ifdef PARGEMSLR_CUDA
      
      /**
       * @brief   The cuda random number generator.
       * @details The cuda random number generator.
       */
      static curandGenerator_t            _curand_gen;
      
      /**
       * @brief   The cuBLAS handle.
       * @details The cuBLAS handle.
       */
      static cublasHandle_t               _cublas_handle;
      
      /**
       * @brief   The cuSPARSE handle.
       * @details The cuSPARSE handle.
       */
      static cusparseHandle_t             _cusparse_handle;
      
      /**
       * @brief   The CUDA stream.
       * @details The CUDA stream.
       */
      static cudaStream_t                 _stream;
      
      /**
       * @brief   The cuSPARSE integer base, by default CUSPARSE_INDEX_BASE_ZERO.
       * @details The cuSPARSE integer base, by default CUSPARSE_INDEX_BASE_ZERO.
       */
      static cusparseIndexBase_t          _cusparse_idx_base;
      
      /**
       * @brief   The cuSPARSE general matrix descriptor.
       * @details The cuSPARSE general matrix descriptor.
       */
      static cusparseMatDescr_t           _mat_des;
      
      /**
       * @brief   The cuSPARSE unit diagonal lower triangular matrix descriptor.
       * @details The cuSPARSE unit diagonal lower triangular matrix descriptor.
       */
      static cusparseMatDescr_t           _matL_des;
      
      /**
       * @brief   The cuSPARSE arbitrary diagonal upper triangular matrix descriptor.
       * @details The cuSPARSE arbitrary diagonal upper triangular matrix descriptor.
       */
      static cusparseMatDescr_t           _matU_des;
      
      /**
       * @brief   The ilu solving policy for cuSPARSE.
       * @details The ilu solving policy for cuSPARSE. (Enable/disable level structure)
       */
      static cusparseSolvePolicy_t        _ilu_solve_policy;
      
      /**
       * @brief   Buffers for the cuSPARSE routines.
       * @details Buffers for the cuSPARSE routines.
       */
      static void                         *_cusparse_buffer;
      
      /**
       * @brief   Length of the buffers for the cuSPARSE routines.
       * @details Length of the buffers for the cuSPARSE routines.
       */
      static size_t                       _cusparse_buffer_length;

#if (PARGEMSLR_CUDA_VERSION == 11)
      /**
       * @brief   The cuSPARSE integer type, CUSPARSE_INDEX_32I or CUSPARSE_INDEX_64I.
       * @details The cuSPARSE integer type, CUSPARSE_INDEX_32I or CUSPARSE_INDEX_64I.
       */
      static cusparseIndexType_t          _cusparse_idx_type;
      
      /**
       * @brief   The algorithm of spMV for cuSPARSE.
       * @details The algorithm of spMV for cuSPARSE.
       */
      static cusparseSpMVAlg_t            _cusparse_spmv_algorithm;
#endif
      
#endif

      /**
       * @brief   The working location of the code (device/host).
       * @details The working location of the code (device/host).
       */
      static int                          _working_location;
      
      /**
       * @brief   The total number of global MPI ranks.
       * @details The total number of global MPI ranks.
       */
      static int                          _gsize;
      
      /**
       * @brief   The number of global MPI rank.
       * @details The number of global MPI rank.
       */
      static int                          _grank;
      
      /**
       * @brief   The global MPI comm.
       * @details The global MPI comm.
       */
      static MPI_Comm                     *_gcomm;
      
      /**
       * @brief   The local MPI comm (one np only, for consistancy).
       * @details The local MPI comm (one np only, for consistancy).
       */
      static MPI_Comm                     *_lcomm;
      
      /**
       * @brief   The total number of local MPI ranks.
       * @details The total number of local MPI ranks.
       */
      int                                 _size;
      
      /**
       * @brief   The number of local MPI rank.
       * @details The number of local MPI rank.
       */
      int                                 _rank;
      
      /**
       * @brief   The MPI comm that doesn't need to be freed.
       * @details The MPI comm that doesn't need to be freed.
       */
      MPI_Comm                            _commref;
      
      /**
       * @brief   The local MPI comm.
       * @details The local MPI comm.
       */
      MPI_Comm                            *_comm;
      
      /**
       * @brief   The std::vector stores the timing information.
       * @details The std::vector stores the timing information.
       */
      static vector<double>               _times;
      
      /**
       * @brief   The std::vector stores the start time of each section.
       * @details The std::vector stores the start time of each section.
       */
      static vector<double>               _times_buffer_start;
      
      /**
       * @brief   The std::vector stores the end time of each section.
       * @details The std::vector stores the end time of each section.
       */
      static vector<double>               _times_buffer_end;
      
      /**
       * @brief   Free the parallel_log.
       * @details Free the parallel_log.
       */
      int                                 Clear();
      
      /**
       * @brief   The default constructor of parallel_log.
       * @details The default constructor of parallel_log.
       */
      ParallelLogClass();
      
      /**
       * @brief   The copy constructor of parallel_log.
       * @details The copy constructor of parallel_log.
       */
      ParallelLogClass(const ParallelLogClass &parlog);
      
      /**
       * @brief   The = operator of parallel_log.
       * @details The = operator of parallel_log.
       * @param   [in]        parlog The ParallelLogClass.
       */
      ParallelLogClass( ParallelLogClass &&parlog);
      
      /**
       * @brief   The = operator of parallel_log.
       * @details The = operator of parallel_log.
       * @param   [in]        parlog The ParallelLogClass.
       * @return              Return the ParallelLogClass.
       */
      ParallelLogClass& operator= (const ParallelLogClass &parlog);
      
      /**
       * @brief   The = operator of parallel_log.
       * @details The = operator of parallel_log.
       * @param   [in]        parlog The ParallelLogClass.
       * @return              Return the ParallelLogClass.
       */
      ParallelLogClass& operator= ( ParallelLogClass &&parlog);
      
      /**
       * @brief   The constructor of parallel_log, setup a new local comm.
       * @details The constructor of parallel_log, setup a new local comm.
       * @param [in] comm_in The new comm.
       */
      ParallelLogClass(MPI_Comm comm_in);
      
      /**
       * @brief   The destructor of parallel_log.
       * @details The destructor of parallel_log.
       */
      ~ParallelLogClass();
      
      /**
       * @brief   Get comm, np, and myid. When _comm is NULL, get the global one, otherwise get the local one.
       * @details Get comm, np, and myid. When _comm is NULL, get the global one, otherwise get the local one.
       * @param   [in]        np The number of processors.
       * @param   [in]        myid The local MPI rank number.
       * @param   [in]        comm The MPI_Comm.
       * @return              Return error message.
       */
      int                                 GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
      
      /**
       * @brief      Get the MPI_comm. When _comm is NULL, get the global one, otherwise get the local one.
       * @details    Get the MPI_comm. When _comm is NULL, get the global one, otherwise get the local one.
       * @return     Return the MPI_comm.
       */
      MPI_Comm                            GetComm() const;
      
   }parallel_log, *parallel_logp;
   
   /**
    * @brief   Set the OpenMP thread number for each MPI process.
    * @details Set the OpenMP thread number for each MPI process.
    * @param   [in]     nthreads The number of threads per MPI process.
    * @return           Return error message.                                               
    */
   int PargemslrSetOpenmpNumThreads(int nthreads);
   
#ifdef PARGEMSLR_OPENMP

   /**
    * @brief   Get the local OpenMP thread number for each MPI process.
    * @details Get the local OpenMP thread number for each MPI process.
    * @return  Return the local thread number for each MPI process.
    */
   int PargemslrGetOpenmpThreadNum();
   

   /**
    * @brief   Get the current number of OpenMP threads.
    * @details Get the current number of OpenMP threads.
    * @return  Return the current thread number for each MPI process.
    */
   int PargemslrGetOpenmpNumThreads();
   
   /**
    * @brief   Get the max number of availiable OpenMP threads. Note that inside omp parallel region this function would return 1.
    * @details Get the max number of availiable OpenMP threads. Note that inside omp parallel region this function would return 1.
    * @return  Return the max availiable thread number for each MPI process.
    */
   int PargemslrGetOpenmpMaxNumThreads();
   
   /**
    * @brief   Get the max number of OpenMP threads.
    * @details Get the max number of OpenMP threads.
    * @return  Return the max thread number for each MPI process.
    */
   int PargemslrGetOpenmpGlobalMaxNumThreads();
   
#endif

   /**
    * @brief   Each MPI rank holds n_local, get the n_start and n_global.
    * @details Each MPI rank holds n_local, get the n_start and n_global.
    * @param [in]  	   n_local The local size.
    * @param [out]      n_start The start index.
    * @param [out]      n_global The global size.
    * @param [out]      comm The MPI_comm.
    * @return           Return error message.                                                         
    */
   int PargemslrNLocalToNGlobal( int n_local, pargemslr_long &n_start, pargemslr_long &n_global, MPI_Comm &comm);
   
   /**
    * @brief   Each MPI rank holds two n_locals, get the n_starts and n_globals.
    * @details Each MPI rank holds two n_locals, get the n_starts and n_globals.
    * @param [in]  	   nrow_local The first local size.
    * @param [in]  	   ncol_local The second local size.
    * @param [out]      nrow_start The first start index.
    * @param [out]      ncol_start The second start index.
    * @param [out]      nrow_global The first global size.
    * @param [out]      ncol_global The second global size.
    * @param [out]      comm The MPI_comm.
    * @return           Return error message.                                                         
    */
   int PargemslrNLocalToNGlobal( int nrow_local, int ncol_local, pargemslr_long &nrow_start, pargemslr_long &ncol_start, pargemslr_long &nrow_global, pargemslr_long &ncol_global, MPI_Comm &comm);
   
   /**
    * @brief   Initilize MPI, OpenMP, and CUDA. Note that if you have already called MPI_Init, call other init functions instead.
    * @details Initilize MPI, OpenMP, and CUDA. Note that if you have already called MPI_Init, call other init functions instead.
    * @param [in,out]  	argc Input of the main function.
    * @param [in,out]   argv Input of the main function.
    * @return           Return error message.                                                         
    */
   int PargemslrInit(int *argc, char ***argv);
   
   /**
    * @brief   Initilize MPI data struct with MPI_Comm.
    * @details Initilize MPI data struct with MPI_Comm. The ParGEMSLR package will duplicate this MPI_Comm.
    * @param [in]   comm The comm for ParGEMSLR, typically should be MPI_COMM_WORLD.
    * @return           Return error message.                                                         
    */
   int PargemslrInitMpi(MPI_Comm comm);
   
   /**
    * @brief   Initilize OpenMP and MKL.
    * @details Initilize OpenMP and MKL.
    * @param [in]   nthreads The max number of OpenMP threads.
    * @return           Return error message.                                                         
    */
   int PargemslrInitOpenMP(int nthreads);
   
   /**
    * @brief   Initilize CUDA.
    * @details Initilize CUDA.
    * @return  Return error message.                                                         
    */
   int PargemslrInitCUDA();
   
   /**
    * @brief   Print the parallel information to output.
    * @details Print the parallel information to output.
    * @return  Return error message.                                                         
    */
   int PargemslrPrintParallelInfo();
   
   /**
    * @brief   Finalize MPI, OpenMP, and CUDA. Note that if you don't want to call MPI_Finalize here, call other finalize functions.
    * @details Finalize MPI, OpenMP, and CUDA. Note that if you don't want to call MPI_Finalize here, call other finalize functions.
    * @return  Return error message.                                                         
    */
   int PargemslrFinalize();
   
   /**
    * @brief   Finalize MPI data structure. Note that MPI_Finalize won't be called here.
    * @details Finalize MPI data structure. Note that MPI_Finalize won't be called here.
    * @return  Return error message.                                                         
    */
   int PargemslrFinalizeMpi();
   
   /**
    * @brief   Finalize OpenMP data structure.
    * @details Finalize OpenMP data structure.
    * @return  Return error message.                                                         
    */
   int PargemslrFinalizeOpenMP();
   
   /**
    * @brief   Finalize CUDA data structure.
    * @details Finalize CUDA data structure.
    * @return  Return error message.                                                         
    */
   int PargemslrFinalizeCUDA();
   
   /**
    * @brief   Get current time using MPI_Wtime.
    * @details Get current time using MPI_Wtime.
    * @param [in]  comm The MPI_Comm. 
    * @param [out] t The time.  
    * @return  Return error message.                                                         
    */
   int PargemslrMpiTime(MPI_Comm comm, double &t);

#ifdef MPI_C_FLOAT_COMPLEX
   
   /**
    * @brief   MPI_Isend.
    * @details MPI_Isend.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   int PargemslrMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   int PargemslrMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Send.
    * @details MPI_Send.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm. 
    * @return      Return error message.                                                         
    */
   template <typename T>
   int PargemslrMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm);
   
   /**
    * @brief   MPI_Recv.
    * @details MPI_Recv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   int PargemslrMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
#else
   
   /**
    * @brief   MPI_Isend.
    * @details MPI_Isend.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Isend.
    * @details MPI_Isend.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Send.
    * @details MPI_Send.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm);
   
   /**
    * @brief   MPI_Send.
    * @details MPI_Send.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   typename std::enable_if<!PargemslrIsComplex<T1>::value&&!PargemslrIsComplex<T2>::value, int>::type
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   typename std::enable_if<!PargemslrIsComplex<T1>::value&&PargemslrIsComplex<T2>::value, int>::type
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   typename std::enable_if<PargemslrIsComplex<T1>::value&&!PargemslrIsComplex<T2>::value, int>::type
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   typename std::enable_if<PargemslrIsComplex<T1>::value&&PargemslrIsComplex<T2>::value, int>::type
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX
   
   /**
    * @brief   MPI_Bcast.
    * @details MPI_Bcast.
    * @param [in,out]   buf Pointer to the data. 
    * @param [in]       count Number of elements. 
    * @param [in]       root Root's MPI rank. 
    * @param [in]       comm MPI_Comm.    
    * @return           Return error message.                                                           
    */
   template <typename T>
   int PargemslrMpiBcast(T *buf, int count, int root, MPI_Comm comm);

#else

   /**
    * @brief   MPI_Bcast.
    * @details MPI_Bcast.
    * @param [in,out]   buf Pointer to the data. 
    * @param [in]       count Number of elements. 
    * @param [in]       root Root's MPI rank. 
    * @param [in]       comm MPI_Comm.    
    * @return           Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiBcast(T *buf, int count, int root, MPI_Comm comm);

   /**
    * @brief   MPI_Bcast.
    * @details MPI_Bcast.
    * @param [in,out]   buf Pointer to the data. 
    * @param [in]       count Number of elements. 
    * @param [in]       root Root's MPI rank. 
    * @param [in]       comm MPI_Comm.    
    * @return           Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiBcast(T *buf, int count, int root, MPI_Comm comm);

#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Scan.
    * @details MPI_Scan.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                         
    */
   template <typename T>
   int PargemslrMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);

#else

   /**
    * @brief   MPI_Scan.
    * @details MPI_Scan.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                             
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);

   /**
    * @brief   MPI_Scan.
    * @details MPI_Scan.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                              
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);

#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Reduce.
    * @details MPI_Reduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  root Root's MPI rank. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   int PargemslrMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

#else

   /**
    * @brief   MPI_Reduce.
    * @details MPI_Reduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  root Root's MPI rank. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                       
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

   /**
    * @brief   MPI_Reduce.
    * @details MPI_Reduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  root Root's MPI rank. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                        
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Allreduce.
    * @details MPI_Allreduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op.
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   int PargemslrMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
#else

   /**
    * @brief   MPI_Allreduce.
    * @details MPI_Allreduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op.
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
   /**
    * @brief   MPI_Allreduce.
    * @details MPI_Allreduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op.
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   In place MPI_Allreduce.
    * @details In place MPI_Allreduce.
    * @param [in,out] buf Pointer to the data. 
    * @param [in]     count Number of elements. 
    * @param [in]     op MPI_Op. 
    * @param [in]     comm MPI_Comm.    
    * @return         Return error message.                                                         
    */
   template <typename T>
   int PargemslrMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm);
   
#else

   /**
    * @brief   In place MPI_Allreduce.
    * @details In place MPI_Allreduce.
    * @param [in,out] buf Pointer to the data. 
    * @param [in]     count Number of elements. 
    * @param [in]     op MPI_Op. 
    * @param [in]     comm MPI_Comm.    
    * @return         Return error message.                                                        
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm);
   
   /**
    * @brief   In place MPI_Allreduce.
    * @details In place MPI_Allreduce.
    * @param [in,out] buf Pointer to the data. 
    * @param [in]     count Number of elements. 
    * @param [in]     op MPI_Op. 
    * @param [in]     comm MPI_Comm.    
    * @return         Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Gather.
    * @details MPI_Gather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data.
    * @param [in]  root The MPI rank of the reciver. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                         
    */
   template <typename T>
   int PargemslrMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm);
   
#else

   /**
    * @brief   MPI_Gather.
    * @details MPI_Gather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data.
    * @param [in]  root The MPI rank of the reciver. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                        
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm);
   
   /**
    * @brief   MPI_Gather.
    * @details MPI_Gather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data.
    * @param [in]  root The MPI rank of the reciver. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                        
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm);
     
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Allgather.
    * @details MPI_Allgather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   int PargemslrMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm);
   
#else

   /**
    * @brief   MPI_Allgather.
    * @details MPI_Allgather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm);
   
   /**
    * @brief   MPI_Allgather.
    * @details MPI_Allgather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm);
     
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Allgatherv.
    * @details MPI_Allgatherv.
    * @param [in]  sendbuf Pointer to the send data.
    * @param [in]  count Size of each single send/recv.  
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  recvcounts Number of elements on each processor. 
    * @param [in]  recvdisps Displacement of elements on each processor. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                         
    */
   template <typename T>
   int PargemslrMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   
#else

   /**
    * @brief   MPI_Allgatherv.
    * @details MPI_Allgatherv.
    * @param [in]  sendbuf Pointer to the send data.
    * @param [in]  count Size of each single send/recv.  
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  recvcounts Number of elements on each processor. 
    * @param [in]  recvdisps Displacement of elements on each processor. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   
   /**
    * @brief   MPI_Allgatherv.
    * @details MPI_Allgatherv.
    * @param [in]  sendbuf Pointer to the send data.
    * @param [in]  count Size of each single send/recv.  
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  recvcounts Number of elements on each processor. 
    * @param [in]  recvdisps Displacement of elements on each processor. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.   
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
     
#endif

#ifdef PARGEMSLR_CUDA

   /**
    * @brief   The cuda synchronize.
    * @details The cuda synchronize.
    * @return  Return error message.                                                         
    */
   int PargemslrCudaSynchronize();
#endif

   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template <typename T> 
   MPI_Datatype PargemslrMpiDataType();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype PargemslrMpiDataType<int>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype PargemslrMpiDataType<long int>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype PargemslrMpiDataType<float>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype PargemslrMpiDataType<double>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype PargemslrMpiDataType<complexs>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype PargemslrMpiDataType<complexd>();
   
}

/*- - - - - - - - - OPENMP default schedule */
#ifdef PARGEMSLR_OPENMP

/* some implementations requires same operation order, use PARGEMSLR_OPENMP_SCHEDULE_STATIC to make sure */
//#define PARGEMSLR_OPENMP_SCHEDULE_DEFAULT       schedule(dynamic)
#define PARGEMSLR_OPENMP_SCHEDULE_DEFAULT       schedule(static)
#define PARGEMSLR_OPENMP_SCHEDULE_STATIC        schedule(static)

#endif

/*- - - - - - - - - MPI calls */

#ifdef PARGEMSLR_DEBUG

#define PARGEMSLR_MPI_CALL(...) {\
   assert( (__VA_ARGS__) == MPI_SUCCESS);\
}

#else

#define PARGEMSLR_MPI_CALL(...) {\
   (__VA_ARGS__);\
}

#endif

/*- - - - - - - - - CUDA calls */

#ifdef PARGEMSLR_CUDA

#ifndef PARGEMSLR_CUDA_VERSION

/* the default CUDA version is 11, note that we only support CUDA 10 and CUDA 11 yet */
#define PARGEMSLR_CUDA_VERSION 11

#endif

#define PARGEMSLR_CUDA_SYNCHRONIZE PargemslrCudaSynchronize();

#ifdef PARGEMSLR_DEBUG

#define PARGEMSLR_CUDA_CALL(...) {\
   assert((__VA_ARGS__) == cudaSuccess);\
}

#define PARGEMSLR_CURAND_CALL(...) {\
   assert((__VA_ARGS__) == CURAND_STATUS_SUCCESS);\
}

#define PARGEMSLR_CUBLAS_CALL(...) {\
   assert( (__VA_ARGS__) == CUBLAS_STATUS_SUCCESS);\
}

#define PARGEMSLR_CUSPARSE_CALL(...) {\
   assert((__VA_ARGS__) == CUSPARSE_STATUS_SUCCESS);\
}

#else

#define PARGEMSLR_CUDA_CALL(...) {\
   (__VA_ARGS__);\
}

#define PARGEMSLR_CURAND_CALL(...) {\
   (__VA_ARGS__);\
}

#define PARGEMSLR_CUBLAS_CALL(...) {\
   (__VA_ARGS__);\
}

#define PARGEMSLR_CUSPARSE_CALL(...) {\
   (__VA_ARGS__);\
}

#endif

//#define PARGEMSLR_THRUST_CALL(thrust_function, ...) thrust::thrust_function(thrust::cuda::par.on(pargemslr::ParallelLogClass::_stream), __VA_ARGS__)

#define PARGEMSLR_THRUST_CALL(thrust_function, ...) thrust::thrust_function( __VA_ARGS__)

#else

#define PARGEMSLR_CUDA_SYNCHRONIZE

#endif

#define PARGEMSLR_GLOBAL_SEQUENTIAL_RUN(...) {\
   for(int pgsri = 0 ; pgsri < pargemslr::parallel_log::_gsize ; pgsri++)\
   {\
      if( pargemslr::parallel_log::_grank == pgsri)\
      {\
         (__VA_ARGS__);\
      }\
      MPI_Barrier(*(pargemslr::parallel_log::_gcomm));\
   }\
}

#endif
