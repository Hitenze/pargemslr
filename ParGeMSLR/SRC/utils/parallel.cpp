

#include <assert.h>
#include "parallel.hpp"
#include "utils.hpp"
#include "memory.hpp"
#include "protos.hpp"

namespace pargemslr
{
   /* variables */
#ifdef PARGEMSLR_CUDA
   
   curandGenerator_t parallel_log::_curand_gen = NULL;
   cublasHandle_t parallel_log::_cublas_handle = NULL;
   cusparseHandle_t parallel_log::_cusparse_handle = NULL;
   cudaStream_t parallel_log::_stream = 0;
   cusparseIndexBase_t parallel_log::_cusparse_idx_base = CUSPARSE_INDEX_BASE_ZERO;
   cusparseMatDescr_t parallel_log::_mat_des = NULL;
   cusparseMatDescr_t parallel_log::_matL_des = NULL;
   cusparseMatDescr_t parallel_log::_matU_des = NULL;
   //cusparseSolvePolicy_t parallel_log::_ilu_solve_policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
   cusparseSolvePolicy_t parallel_log::_ilu_solve_policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   void* parallel_log::_cusparse_buffer = NULL;
   size_t parallel_log::_cusparse_buffer_length = 0;

#if (PARGEMSLR_CUDA_VERSION == 11)

   cusparseIndexType_t parallel_log::_cusparse_idx_type = CUSPARSE_INDEX_32I;
   cusparseSpMVAlg_t parallel_log::_cusparse_spmv_algorithm = CUSPARSE_CSRMV_ALG1;

#endif
#endif

   int parallel_log::_working_location = kMemoryHost;
   int parallel_log::_gsize = 0;
   int parallel_log::_grank = 0;
   MPI_Comm* parallel_log::_gcomm = NULL;
   MPI_Comm* parallel_log::_lcomm = NULL;
   vector<double> parallel_log::_times = vector<double>(PARGEMSLR_TIMES_NUM, 0);
   vector<double> parallel_log::_times_buffer_start = vector<double>(PARGEMSLR_TIMES_NUM, 0);
   vector<double> parallel_log::_times_buffer_end = vector<double>(PARGEMSLR_TIMES_NUM, 0);
   
   int parallel_log::Clear()
   {
      if(this->_comm != NULL)
      {
#ifdef PARGEMSLR_DEBUG
         if(parallel_log::_gcomm == NULL)
         {
            PARGEMSLR_ERROR("Free MPI_Comm after MPI_Finalize().");
         }
#endif
         PARGEMSLR_MPI_CALL( MPI_Comm_free(this->_comm) );
         PARGEMSLR_FREE(this->_comm, kMemoryHost);
      }
      this->_commref = MPI_COMM_WORLD;
      return PARGEMSLR_SUCCESS;
   }
   
   parallel_log::ParallelLogClass()
   {
      this->_commref = MPI_COMM_WORLD;
      this->_comm = NULL;
   }
   
   parallel_log::ParallelLogClass(const ParallelLogClass &parlog)
   {
      this->_commref = parlog._commref;
      this->_size = parlog._size;
      this->_rank = parlog._rank;
      if(parlog._comm)
      {
         PARGEMSLR_MALLOC(this->_comm, 1, kMemoryHost, MPI_Comm);
         PARGEMSLR_MPI_CALL( (MPI_Comm_dup(*(parlog._comm), this->_comm)) );
      }
      else
      {
         this->_comm = NULL;
      }
   }
   
   parallel_log::ParallelLogClass( ParallelLogClass &&parlog)
   {
      this->_commref = parlog._commref;
      parlog._commref = MPI_COMM_WORLD;
      this->_comm = parlog._comm;
      parlog._comm = NULL;
      this->_rank = parlog._rank;
      parlog._rank = 0;
      this->_size = parlog._size;
      parlog._size = 0;
   }
   
   ParallelLogClass& parallel_log::operator= (const ParallelLogClass &parlog)
   {
      this->Clear();
      this->_size = parlog._size;
      this->_rank = parlog._rank;
      this->_commref = parlog._commref;
      if(parlog._comm)
      {
         PARGEMSLR_MALLOC(this->_comm, 1, kMemoryHost, MPI_Comm);
         PARGEMSLR_MPI_CALL( (MPI_Comm_dup(*(parlog._comm), this->_comm)) );
      }
      else
      {
         this->_comm = NULL;
      }
      return *this;
   }
   
   ParallelLogClass& parallel_log::operator= ( ParallelLogClass &&parlog)
   {
      this->Clear();
      this->_commref = parlog._commref;
      parlog._commref = MPI_COMM_WORLD;
      this->_comm = parlog._comm;
      parlog._comm = NULL;
      this->_rank = parlog._rank;
      parlog._rank = 0;
      this->_size = parlog._size;
      parlog._size = 0;
      return *this;
   }
   
   parallel_log::ParallelLogClass(MPI_Comm comm_in)
   {
      /* can only be called after we call the MPI_Init */
      PARGEMSLR_CHKERR(parallel_log::_gcomm == NULL);
      PARGEMSLR_MALLOC( this->_comm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_dup(comm_in, this->_comm) );
      PARGEMSLR_MPI_CALL( MPI_Comm_size(comm_in, &(this->_size)) );
      PARGEMSLR_MPI_CALL( MPI_Comm_rank(comm_in, &(this->_rank)) );
      this->_commref = MPI_COMM_WORLD;
   }
   
   parallel_log::~ParallelLogClass()
   {
      Clear();
   }
   
   int parallel_log::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const
   {
      
      if(this->_comm == NULL)
      {
         /* use the global one */
         if(this->_commref != MPI_COMM_WORLD)
         {
            comm = this->_commref;
            np    = this->_size;
            myid  = this->_rank;
         }
         else
         {
            comm  = *(parallel_log::_gcomm);
            np    = parallel_log::_gsize;
            myid  = parallel_log::_grank;
         }
      }
      else
      {
         /* use this one */
         comm  = *(this->_comm);
         np    = this->_size;
         myid  = this->_rank;
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   
   MPI_Comm parallel_log::GetComm() const
   {
      
      if(this->_comm == NULL)
      {
         /* use the global one */
         if(this->_commref != MPI_COMM_WORLD)
         {
            return this->_commref;
         }
         else
         {
            return  *(parallel_log::_gcomm);
         }
      }
      else
      {
         /* use this one */
         return  *(this->_comm);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   
   int PargemslrSetOpenmpNumThreads(int nthreads)
   {
#ifdef PARGEMSLR_OPENMP
      omp_set_num_threads(nthreads);
#endif
#ifdef PARGEMSLR_OPENBLAS
      openblas_set_num_threads(nthreads);
#endif
#ifdef PARGEMSLR_MKL
      mkl_set_num_threads(nthreads);
#endif
      return PARGEMSLR_SUCCESS;
   }
   
#ifdef PARGEMSLR_OPENMP

   int PargemslrGetOpenmpThreadNum()
   {
      return omp_get_thread_num();
   }
   
   int PargemslrGetOpenmpNumThreads()
   {
      return omp_get_num_threads();
   }
   
   int PargemslrGetOpenmpMaxNumThreads()
   {
      /* if already inside a parallel region, 
       * omp_get_max_threads() == omp_get_num_threads()
       * and thus this function returns 1, avoid nested OpenMP call.
       */
      return omp_get_max_threads()/omp_get_num_threads();
   }
   
   int PargemslrGetOpenmpGlobalMaxNumThreads()
   {
      return omp_get_max_threads();
   }
   
#endif

   int PargemslrNLocalToNGlobal( int n_local, pargemslr_long &n_start, pargemslr_long &n_global, MPI_Comm &comm)
   {
      
      int      np;
      PARGEMSLR_MPI_CALL(MPI_Comm_size(comm, &np));
      
      pargemslr_long n_local_long = (pargemslr_long) n_local;
      
      /* after scan, n_start is the exact n_start plus n_local */
      PargemslrMpiScan( &n_local_long, &n_start, 1, MPI_SUM, comm);
      
      /* now, the n_global on the last MPI rank is the exact one, bcast it */
      n_global = n_start;
      PargemslrMpiBcast( &n_global, 1, np-1, comm);
      
      /* shift back to get n_start */
      n_start -= n_local_long;
      
      return PARGEMSLR_SUCCESS;
   }
   
   int PargemslrNLocalToNGlobal( int nrow_local, int ncol_local, pargemslr_long &nrow_start, pargemslr_long &ncol_start, pargemslr_long &nrow_global, pargemslr_long &ncol_global, MPI_Comm &comm)
   {
      
      int      np;
      PARGEMSLR_MPI_CALL(MPI_Comm_size(comm, &np));
      
      pargemslr_long *pbuffer;
      PARGEMSLR_MALLOC( pbuffer, 6, kMemoryHost, pargemslr_long);
      
      pbuffer[0] = (pargemslr_long) nrow_local;
      pbuffer[1] = (pargemslr_long) ncol_local;
      
      /* after scan, n_start is the exact n_start plus n_local */
      PargemslrMpiScan( pbuffer, pbuffer+2, 2, MPI_SUM, comm);
      
      /* now, the n_global on the last MPI rank is the exact one, bcast it */
      pbuffer[4] = pbuffer[2];
      pbuffer[5] = pbuffer[3];
      PargemslrMpiBcast( pbuffer+4, 2, np-1, comm);
      
      /* shift back to get n_start */
      pbuffer[2] -= pbuffer[0];
      pbuffer[3] -= pbuffer[1];
      
      nrow_start = pbuffer[2];
      ncol_start = pbuffer[3];
      nrow_global = pbuffer[4];
      ncol_global = pbuffer[5];
      
      PARGEMSLR_FREE( pbuffer, kMemoryHost);
      
      return PARGEMSLR_SUCCESS;
   }
   
   int PargemslrMpiTime(MPI_Comm comm, double &t)
   {
      PARGEMSLR_CUDA_SYNCHRONIZE;
      PARGEMSLR_MPI_CALL( MPI_Barrier(comm) );
      t = MPI_Wtime();
      return PARGEMSLR_SUCCESS;
   }
   
   int PargemslrInit(int *argc, char ***argv)
   {
      int size, rank, nthreads = 1;
      /* if this is not null, we have called the init */
      if(parallel_log::_gcomm != NULL)
      {
         return PARGEMSLR_ERROR_DOUBLE_INIT_FREE;
      }
      
      /* init MPI */
      PARGEMSLR_MPI_CALL( MPI_Init(argc, argv) );
      
      /* We do not directly use MPI_COMM_WORLD */
      PARGEMSLR_MALLOC( parallel_log::_gcomm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_dup(MPI_COMM_WORLD, parallel_log::_gcomm) );
      
      PARGEMSLR_MPI_CALL( MPI_Comm_size( *(parallel_log::_gcomm), &size) );
      PARGEMSLR_MPI_CALL( MPI_Comm_rank( *(parallel_log::_gcomm), &rank) );
      
      parallel_log::_gsize = size;
      parallel_log::_grank = rank;
      
      PARGEMSLR_MALLOC( parallel_log::_lcomm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_split(MPI_COMM_WORLD, rank, rank, parallel_log::_lcomm) );
      
      /* prepare openmp and MKL */
      PargemslrReadInputArg( "nthreads", 1, &nthreads, *argc, *argv);

      PargemslrInitOpenMP(nthreads);

      PargemslrInitCUDA();
      
      return PARGEMSLR_SUCCESS;
   }
   
   int PargemslrInitMpi(MPI_Comm comm)
   {
      int size, rank;
      /* We do not directly use MPI_COMM_WORLD */
      PARGEMSLR_MALLOC( parallel_log::_gcomm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_dup(comm, parallel_log::_gcomm) );
      
      PARGEMSLR_MPI_CALL( MPI_Comm_size( *(parallel_log::_gcomm), &size) );
      PARGEMSLR_MPI_CALL( MPI_Comm_rank( *(parallel_log::_gcomm), &rank) );
      
      parallel_log::_gsize = size;
      parallel_log::_grank = rank;
      
      PARGEMSLR_MALLOC( parallel_log::_lcomm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_split(comm, rank, rank, parallel_log::_lcomm) );
      
      return PARGEMSLR_SUCCESS;
      
   }
   
   int PargemslrInitOpenMP(int nthreads)
   {
#ifdef PARGEMSLR_OPENMP     
      omp_set_num_threads(nthreads);
#endif
#ifdef PARGEMSLR_OPENBLAS
      openblas_set_num_threads(nthreads);
#endif
#ifdef PARGEMSLR_MKL
      mkl_set_num_threads(nthreads);
      /* set to 0 to fix the thread on each node to be nthreads, otherwise MKL might change this number during runtime */
      mkl_set_dynamic(1);
#endif
      return PARGEMSLR_SUCCESS;
   }

   int PargemslrInitCUDA()
   {

#ifdef PARGEMSLR_CUDA

      /* with CUDA enabled, the default working location is the device */
      parallel_log::_working_location = kMemoryDevice;
      
      /* split comm by shared memory region to know the number of GPU availiable in each range */
      
      int dc;
      PARGEMSLR_CUDA_CALL( (cudaGetDeviceCount(&dc)) );
      
      if(dc == 0 && parallel_log::_grank == 0)
      {
         printf("Error: no availiable device for MPI rank %d.\n", parallel_log::_grank == 0);
         exit(0);
      }
      
#ifdef PARGEMSLR_DEBUG
      printf("MPI rank %d have access to %d GPUs, set to %d/%d of them.\n", parallel_log::_grank, dc, parallel_log::_grank%dc, dc);
#endif   

      /* now set the device for each processor */
      PARGEMSLR_CUDA_CALL( (cudaSetDevice(parallel_log::_grank % dc)) );
      
      /* create new non-blocking stream 
       * stream does not synchronize with stream 0
       */
      //PARGEMSLR_CUDA_CALL( (cudaStreamCreateWithFlags(&(parallel_log::_stream), cudaStreamNonBlocking)) );
      parallel_log::_stream = 0;
      
      /* create new cublas handle */
      PARGEMSLR_CUBLAS_CALL( (cublasCreate(&(parallel_log::_cublas_handle))) );
      
      /* create new cusparse handle */
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreate(&(parallel_log::_cusparse_handle))) );
      
      /* bind stream */
      PARGEMSLR_CUBLAS_CALL( (cublasSetStream(parallel_log::_cublas_handle, parallel_log::_stream)) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetStream(parallel_log::_cusparse_handle, parallel_log::_stream)) );
      
      /* cublas setup */
      
      /* CUBLAS_POINTER_MODE_HOST, pass alpha/beta from the host, return value to the host would block the cuda */
      PARGEMSLR_CUBLAS_CALL( (cublasSetPointerMode(parallel_log::_cublas_handle, CUBLAS_POINTER_MODE_HOST)) );
      
      /* cusparse setup */
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetPointerMode(parallel_log::_cusparse_handle, CUSPARSE_POINTER_MODE_HOST)) );
      
      /* create cusparse matrix descriptor */
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateMatDescr(&(parallel_log::_mat_des))) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatIndexBase(parallel_log::_mat_des, CUSPARSE_INDEX_BASE_ZERO)) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatType(parallel_log::_mat_des, CUSPARSE_MATRIX_TYPE_GENERAL)) );
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateMatDescr(&(parallel_log::_matL_des)))  );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatIndexBase(parallel_log::_matL_des, CUSPARSE_INDEX_BASE_ZERO)) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatType(parallel_log::_matL_des, CUSPARSE_MATRIX_TYPE_GENERAL)) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatFillMode(parallel_log::_matL_des, CUSPARSE_FILL_MODE_LOWER)) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatDiagType(parallel_log::_matL_des, CUSPARSE_DIAG_TYPE_UNIT)) );
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateMatDescr(&(parallel_log::_matU_des))) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatIndexBase(parallel_log::_matU_des, CUSPARSE_INDEX_BASE_ZERO)) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatType(parallel_log::_matU_des, CUSPARSE_MATRIX_TYPE_GENERAL)) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatFillMode(parallel_log::_matU_des, CUSPARSE_FILL_MODE_UPPER)) );
      PARGEMSLR_CUSPARSE_CALL( (cusparseSetMatDiagType(parallel_log::_matU_des, CUSPARSE_DIAG_TYPE_NON_UNIT)) );
  
      /* ilu solving policy */
      parallel_log::_ilu_solve_policy        = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
      parallel_log::_cusparse_buffer_length  = 0;
      parallel_log::_cusparse_buffer         = NULL;
      
#if (PARGEMSLR_CUDA_VERSION == 11)

      /* cusparse general API */
      int size_of_int = sizeof(int);
      switch(size_of_int)
      {
         case 4:
         {
            /* int 32 */
            parallel_log::_cusparse_idx_type = CUSPARSE_INDEX_32I;
            break;
         }
         case 8:
         {
            /* int 64 */
            parallel_log::_cusparse_idx_type = CUSPARSE_INDEX_64I;
            break;
         }
         default:
         {
            return PARGEMSLR_ERROR_COMPILER;
            break;
         }
      }
      parallel_log::_cusparse_spmv_algorithm = CUSPARSE_CSRMV_ALG1;
      
#endif
    
      
      /* curand */
      PARGEMSLR_CURAND_CALL( curandCreateGenerator(&(parallel_log::_curand_gen), CURAND_RNG_PSEUDO_DEFAULT) );
      PARGEMSLR_CURAND_CALL( curandSetStream(parallel_log::_curand_gen, parallel_log::_stream) );
      
#endif

      return PARGEMSLR_SUCCESS;
   }

   int PargemslrPrintParallelInfo()
   {
      if(parallel_log::_grank == 0)
      {
         PargemslrPrintDashLine(pargemslr_global::_dash_line_width);
         printf("Printing parallel information\n");
         
         printf("MPI Info:\n");
         printf("\tNumber of MPI Ranks: %d\n", parallel_log::_gsize);
#ifdef PARGEMSLR_OPENMP
         printf("OPENMP Info:\n");
         printf("\tMaxinum Number of OpenMP Threads Per Node: %d\n",omp_get_max_threads());
#endif
#ifdef PARGEMSLR_CUDA
         printf("GPU Info:\n");
         printf("\tGPU Enabled\n");
#endif
#ifdef PARGEMSLR_MKL
         printf("MKL Info:\n");
         printf("\tMaxinum Number of OpenMP Threads Per Node: %d\n",mkl_get_max_threads());
         printf("\tMKL Dynamic Enabled\n");
#endif
         
         PargemslrPrintDashLine(pargemslr_global::_dash_line_width);
      }
      return PARGEMSLR_SUCCESS;
   }
   
   int PargemslrFinalize()
   {
      /* if this is null, we have free the value */
      if(parallel_log::_gcomm == NULL)
      {
         return PARGEMSLR_ERROR_DOUBLE_INIT_FREE;
      }
      PARGEMSLR_MPI_CALL( MPI_Comm_free(parallel_log::_gcomm) );
      PARGEMSLR_FREE(parallel_log::_gcomm, kMemoryHost);
      PARGEMSLR_MPI_CALL( MPI_Comm_free(parallel_log::_lcomm) );
      PARGEMSLR_FREE(parallel_log::_lcomm, kMemoryHost);
      
      PARGEMSLR_MPI_CALL( MPI_Finalize() );
      
      PargemslrFinalizeCUDA();
      
      /* close output */
      if( pargemslr_global::_out_file != stdout )
      {
         /* free the current */
         fclose(pargemslr_global::_out_file);
         pargemslr_global::_out_file = stdout;
      }
      
      
      return PARGEMSLR_SUCCESS;
   }
   
   int PargemslrFinalizeMpi()
   {
      if(parallel_log::_gcomm == NULL)
      {
         return PARGEMSLR_ERROR_DOUBLE_INIT_FREE;
      }
      PARGEMSLR_MPI_CALL( MPI_Comm_free(parallel_log::_gcomm) );
      PARGEMSLR_FREE(parallel_log::_gcomm, kMemoryHost);
      PARGEMSLR_MPI_CALL( MPI_Comm_free(parallel_log::_lcomm) );
      PARGEMSLR_FREE(parallel_log::_lcomm, kMemoryHost);
      
      return PARGEMSLR_SUCCESS;
   }
   
   int PargemslrFinalizeOpenMP()
   {
      return PARGEMSLR_SUCCESS;
   }
   
   int PargemslrFinalizeCUDA()
   {
      
#ifdef PARGEMSLR_CUDA
      /* free handles and data structures */
      
      PARGEMSLR_CURAND_CALL( curandDestroyGenerator( parallel_log::_curand_gen) );
      parallel_log::_curand_gen = NULL;
      
      PARGEMSLR_CUBLAS_CALL( (cublasDestroy(parallel_log::_cublas_handle)) );
      parallel_log::_cublas_handle = NULL;
         
      PARGEMSLR_CUSPARSE_CALL( (cusparseDestroyMatDescr(parallel_log::_mat_des)) );
      parallel_log::_mat_des = NULL;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseDestroyMatDescr(parallel_log::_matL_des)) );
      parallel_log::_matL_des = NULL;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseDestroyMatDescr(parallel_log::_matU_des)) );
      parallel_log::_matU_des = NULL;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseDestroy(parallel_log::_cusparse_handle)) );
      parallel_log::_cusparse_handle = NULL;
      
      parallel_log::_cusparse_buffer_length = 0;
      PARGEMSLR_FREE(parallel_log::_cusparse_buffer, kMemoryDevice);
      
#endif
      
      return PARGEMSLR_SUCCESS;
   }

#ifdef MPI_C_FLOAT_COMPLEX
   
   template <typename T>
   int PargemslrMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request)
   {
      PARGEMSLR_MPI_CALL( MPI_Isend( buf, count, PargemslrMpiDataType<T>(), dest, tag, comm, request) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiIsend(int *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIsend(long int *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIsend(float *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIsend(double *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIsend(complexs *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIsend(complexd *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   int PargemslrMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request)
   {
      PARGEMSLR_MPI_CALL( MPI_Irecv( buf, count, PargemslrMpiDataType<T>(), source, tag, comm, request) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiIrecv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIrecv(long int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIrecv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIrecv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIrecv(complexs *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIrecv(complexd *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   int PargemslrMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Send( buf, count, PargemslrMpiDataType<T>(), dest, tag, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiSend(int *buf, int count, int dest, int tag, MPI_Comm comm);
   template int PargemslrMpiSend(long int *buf, int count, int dest, int tag, MPI_Comm comm);
   template int PargemslrMpiSend(float *buf, int count, int dest, int tag, MPI_Comm comm);
   template int PargemslrMpiSend(double *buf, int count, int dest, int tag, MPI_Comm comm);
   template int PargemslrMpiSend(complexs *buf, int count, int dest, int tag, MPI_Comm comm);
   template int PargemslrMpiSend(complexd *buf, int count, int dest, int tag, MPI_Comm comm);
   
   template <typename T>
   int PargemslrMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status)
   {
      PARGEMSLR_MPI_CALL( MPI_Recv( buf, count, PargemslrMpiDataType<T>(), source, tag, comm, status) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiRecv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiRecv(long int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiRecv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiRecv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiRecv(complexs *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiRecv(complexd *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   
   template <typename T1, typename T2>
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      PARGEMSLR_MPI_CALL( MPI_Sendrecv( sendbuf, sendcount, PargemslrMpiDataType<T1>(), dest, sendtag, recvbuf, recvcount, PargemslrMpiDataType<T2>(), source, recvtag, comm, status) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiSendRecv(int *sendbuf, int sendcount, int dest, int sendtag, int *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiSendRecv(long int *sendbuf, int sendcount, int dest, int sendtag, long int *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiSendRecv(float *sendbuf, int sendcount, int dest, int sendtag, float *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiSendRecv(double *sendbuf, int sendcount, int dest, int sendtag, double *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiSendRecv(complexs *sendbuf, int sendcount, int dest, int sendtag, complexs *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiSendRecv(complexd *sendbuf, int sendcount, int dest, int sendtag, complexd *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
#else
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request)
   {
      PARGEMSLR_MPI_CALL( MPI_Isend( buf, count, PargemslrMpiDataType<T>(), dest, tag, comm, request) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiIsend(int *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIsend(long int *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIsend(float *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIsend(double *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request)
   {
      PARGEMSLR_MPI_CALL( MPI_Isend( buf, 2*count, PargemslrMpiDataType<T>(), dest, tag, comm, request) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiIsend(complexs *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIsend(complexd *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request)
   {
      PARGEMSLR_MPI_CALL( MPI_Irecv( buf, count, PargemslrMpiDataType<T>(), source, tag, comm, request) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiIrecv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIrecv(long int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIrecv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIrecv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request)
   {
      PARGEMSLR_MPI_CALL( MPI_Irecv( buf, 2*count, PargemslrMpiDataType<T>(), source, tag, comm, request) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiIrecv(complexs *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int PargemslrMpiIrecv(complexd *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Send( buf, count, PargemslrMpiDataType<T>(), dest, tag, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiSend(int *buf, int count, int dest, int tag, MPI_Comm comm);
   template int PargemslrMpiSend(long int *buf, int count, int dest, int tag, MPI_Comm comm);
   template int PargemslrMpiSend(float *buf, int count, int dest, int tag, MPI_Comm comm);
   template int PargemslrMpiSend(double *buf, int count, int dest, int tag, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Send( buf, 2*count, PargemslrMpiDataType<T>(), dest, tag, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiSend(complexs *buf, int count, int dest, int tag, MPI_Comm comm);
   template int PargemslrMpiSend(complexd *buf, int count, int dest, int tag, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status)
   {
      PARGEMSLR_MPI_CALL( MPI_Recv( buf, count, PargemslrMpiDataType<T>(), source, tag, comm, status) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiRecv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   template int PargemslrMpiRecv(long int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   template int PargemslrMpiRecv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   template int PargemslrMpiRecv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status)
   {
      PARGEMSLR_MPI_CALL( MPI_Recv( buf, 2*count, PargemslrMpiDataType<T>(), source, tag, comm, status) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiRecv(complexs *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   template int PargemslrMpiRecv(complexd *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   template <typename T1, typename T2>
   typename std::enable_if<!PargemslrIsComplex<T1>::value&&!PargemslrIsComplex<T2>::value, int>::type
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      PARGEMSLR_MPI_CALL( MPI_Sendrecv( sendbuf, sendcount, PargemslrMpiDataType<T1>(), dest, sendtag, recvbuf, recvcount, PargemslrMpiDataType<T2>(), source, recvtag, comm, status) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiSendRecv(int *sendbuf, int sendcount, int dest, int sendtag, int *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiSendRecv(long int *sendbuf, int sendcount, int dest, int sendtag, long int *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiSendRecv(float *sendbuf, int sendcount, int dest, int sendtag, float *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiSendRecv(double *sendbuf, int sendcount, int dest, int sendtag, double *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
   template <typename T1, typename T2>
   typename std::enable_if<!PargemslrIsComplex<T1>::value&&PargemslrIsComplex<T2>::value, int>::type
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      PARGEMSLR_MPI_CALL( MPI_Sendrecv( sendbuf, sendcount, PargemslrMpiDataType<T1>(), dest, sendtag, recvbuf, 2*recvcount, PargemslrMpiDataType<T2>(), source, recvtag, comm, status) );
      return PARGEMSLR_SUCCESS;
   }
   
   template <typename T1, typename T2>
   typename std::enable_if<PargemslrIsComplex<T1>::value&&!PargemslrIsComplex<T2>::value, int>::type
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      PARGEMSLR_MPI_CALL( MPI_Sendrecv( sendbuf, 2*sendcount, PargemslrMpiDataType<T1>(), dest, sendtag, recvbuf, recvcount, PargemslrMpiDataType<T2>(), source, recvtag, comm, status) );
      return PARGEMSLR_SUCCESS;
   }
   
   template <typename T1, typename T2>
   typename std::enable_if<PargemslrIsComplex<T1>::value&&PargemslrIsComplex<T2>::value, int>::type
   int PargemslrMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      PARGEMSLR_MPI_CALL( MPI_Sendrecv( sendbuf, 2*sendcount, PargemslrMpiDataType<T1>(), dest, sendtag, recvbuf, 2*recvcount, PargemslrMpiDataType<T2>(), source, recvtag, comm, status) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiSendRecv(complexs *sendbuf, int sendcount, int dest, int sendtag, complexs *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int PargemslrMpiSendRecv(complexd *sendbuf, int sendcount, int dest, int sendtag, complexd *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX
   
   template <typename T>
   int PargemslrMpiBcast(T *buf, int count, int root, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Bcast( buf, count, PargemslrMpiDataType<T>(), root, comm ) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiBcast(int *buf, int count, int root, MPI_Comm comm);
   template int PargemslrMpiBcast(long int *buf, int count, int root, MPI_Comm comm);
   template int PargemslrMpiBcast(float *buf, int count, int root, MPI_Comm comm);
   template int PargemslrMpiBcast(double *buf, int count, int root, MPI_Comm comm);
   template int PargemslrMpiBcast(complexs *buf, int count, int root, MPI_Comm comm);
   template int PargemslrMpiBcast(complexd *buf, int count, int root, MPI_Comm comm);
   
#else
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiBcast(T *buf, int count, int root, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Bcast( buf, count, PargemslrMpiDataType<T>(), root, comm ) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiBcast(int *buf, int count, int root, MPI_Comm comm);
   template int PargemslrMpiBcast(long int *buf, int count, int root, MPI_Comm comm);
   template int PargemslrMpiBcast(float *buf, int count, int root, MPI_Comm comm);
   template int PargemslrMpiBcast(double *buf, int count, int root, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiBcast(T *buf, int count, int root, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Bcast( buf, 2*count, PargemslrMpiDataType<T>(), root, comm ) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiBcast(complexs *buf, int count, int root, MPI_Comm comm);
   template int PargemslrMpiBcast(complexd *buf, int count, int root, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int PargemslrMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Scan( sendbuf, recvbuf, count, PargemslrMpiDataType<T>(), op, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiScan(int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiScan(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiScan(float *sendbuf, float *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiScan(double *sendbuf, double *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiScan(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiScan(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, MPI_Comm comm);

#else

   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Scan( sendbuf, recvbuf, count, PargemslrMpiDataType<T>(), op, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiScan(int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiScan(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiScan(float *sendbuf, float *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiScan(double *sendbuf, double *recvbuf, int count, MPI_Op op, MPI_Comm comm);


   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Scan( sendbuf, recvbuf, 2*count, PargemslrMpiDataType<T>(), op, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiScan(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiScan(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, MPI_Comm comm);


#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int PargemslrMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Reduce( sendbuf, recvbuf, count, PargemslrMpiDataType<T>(), op, root, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiReduce(int *sendbuf, int *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int PargemslrMpiReduce(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int PargemslrMpiReduce(float *sendbuf, float *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int PargemslrMpiReduce(double *sendbuf, double *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int PargemslrMpiReduce(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int PargemslrMpiReduce(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   
#else

   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Reduce( sendbuf, recvbuf, count, PargemslrMpiDataType<T>(), op, root, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiReduce(int *sendbuf, int *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int PargemslrMpiReduce(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int PargemslrMpiReduce(float *sendbuf, float *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int PargemslrMpiReduce(double *sendbuf, double *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Reduce( sendbuf, recvbuf, 2*count, PargemslrMpiDataType<T>(), op, root, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiReduce(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int PargemslrMpiReduce(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int PargemslrMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allreduce( sendbuf, recvbuf, count, PargemslrMpiDataType<T>(), op, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllreduce(int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduce(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduce(float *sendbuf, float *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduce(double *sendbuf, double *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduce(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduce(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
#else

   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allreduce( sendbuf, recvbuf, count, PargemslrMpiDataType<T>(), op, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllreduce(int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduce(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduce(float *sendbuf, float *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduce(double *sendbuf, double *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allreduce( sendbuf, recvbuf, 2*count, PargemslrMpiDataType<T>(), op, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllreduce(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduce(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int PargemslrMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allreduce( MPI_IN_PLACE, buf, count, PargemslrMpiDataType<T>(), op, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllreduceInplace(int *rbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduceInplace(long int *buf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduceInplace(float *buf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduceInplace(double *buf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduceInplace(complexs *buf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduceInplace(complexd *buf, int count, MPI_Op op, MPI_Comm comm);
   
#else

   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allreduce( MPI_IN_PLACE, buf, count, PargemslrMpiDataType<T>(), op, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllreduceInplace(int *rbuf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduceInplace(long int *buf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduceInplace(float *buf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduceInplace(double *buf, int count, MPI_Op op, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allreduce( MPI_IN_PLACE, buf, 2*count, PargemslrMpiDataType<T>(), op, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllreduceInplace(complexs *buf, int count, MPI_Op op, MPI_Comm comm);
   template int PargemslrMpiAllreduceInplace(complexd *buf, int count, MPI_Op op, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int PargemslrMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Gather( sendbuf, count, PargemslrMpiDataType<T>(), recvbuf, count, PargemslrMpiDataType<T>(), root, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiGather(int *sendbuf, int count, int *recvbuf, int root, MPI_Comm comm);
   template int PargemslrMpiGather(long int *sendbuf, int count, long int *recvbuf, int root, MPI_Comm comm);
   template int PargemslrMpiGather(float *sendbuf, int count, float *recvbuf, int root, MPI_Comm comm);
   template int PargemslrMpiGather(double *sendbuf, int count, double *recvbuf, int root, MPI_Comm comm);
   template int PargemslrMpiGather(complexs *sendbuf, int count, complexs *recvbuf, int root, MPI_Comm comm);
   template int PargemslrMpiGather(complexd *sendbuf, int count, complexd *recvbuf, int root, MPI_Comm comm);
   
#else
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Gather( sendbuf, count, PargemslrMpiDataType<T>(), recvbuf, count, PargemslrMpiDataType<T>(), root, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiGather(int *sendbuf, int count, int *recvbuf, int root, MPI_Comm comm);
   template int PargemslrMpiGather(long int *sendbuf, int count, long int *recvbuf, int root, MPI_Comm comm);
   template int PargemslrMpiGather(float *sendbuf, int count, float *recvbuf, int root, MPI_Comm comm);
   template int PargemslrMpiGather(double *sendbuf, int count, double *recvbuf, int root, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Gather( sendbuf, 2*count, PargemslrMpiDataType<T>(), recvbuf, 2*count, PargemslrMpiDataType<T>(), root, comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiGather(complexs *sendbuf, int count, complexs *recvbuf, int root, MPI_Comm comm);
   template int PargemslrMpiGather(complexd *sendbuf, int count, complexd *recvbuf, int root, MPI_Comm comm);
     
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int PargemslrMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allgather( sendbuf, count, PargemslrMpiDataType<T>(),recvbuf, count, PargemslrMpiDataType<T>(), comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllgather(int *sendbuf, int count, int *recvbuf, MPI_Comm comm);
   template int PargemslrMpiAllgather(long int *sendbuf, int count, long int *recvbuf, MPI_Comm comm);
   template int PargemslrMpiAllgather(float *sendbuf, int count, float *recvbuf, MPI_Comm comm);
   template int PargemslrMpiAllgather(double *sendbuf, int count, double *recvbuf, MPI_Comm comm);
   template int PargemslrMpiAllgather(complexs *sendbuf, int count, complexs *recvbuf, MPI_Comm comm);
   template int PargemslrMpiAllgather(complexd *sendbuf, int count, complexd *recvbuf, MPI_Comm comm);
   
#else
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allgather( sendbuf, count, PargemslrMpiDataType<T>(),recvbuf, count, PargemslrMpiDataType<T>(), comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllgather(int *sendbuf, int count, int *recvbuf, MPI_Comm comm);
   template int PargemslrMpiAllgather(long int *sendbuf, int count, long int *recvbuf, MPI_Comm comm);
   template int PargemslrMpiAllgather(float *sendbuf, int count, float *recvbuf, MPI_Comm comm);
   template int PargemslrMpiAllgather(double *sendbuf, int count, double *recvbuf, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allgather( sendbuf, 2*count, PargemslrMpiDataType<T>(),recvbuf, 2*count, PargemslrMpiDataType<T>(), comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllgather(complexs *sendbuf, int count, complexs *recvbuf, MPI_Comm comm);
   template int PargemslrMpiAllgather(complexd *sendbuf, int count, complexd *recvbuf, MPI_Comm comm);
     
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int PargemslrMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allgatherv( sendbuf, count, PargemslrMpiDataType<T>(), recvbuf, recvcounts, recvdisps, PargemslrMpiDataType<T>(), comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllgatherv(int *sendbuf, int count, int *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int PargemslrMpiAllgatherv(long int *sendbuf, int count, long int *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int PargemslrMpiAllgatherv(float *sendbuf, int count, float *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int PargemslrMpiAllgatherv(double *sendbuf, int count, double *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int PargemslrMpiAllgatherv(complexs *sendbuf, int count, complexs *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int PargemslrMpiAllgatherv(complexd *sendbuf, int count, complexd *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   
#else
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm)
   {
      PARGEMSLR_MPI_CALL( MPI_Allgatherv( sendbuf, count, PargemslrMpiDataType<T>(), recvbuf, recvcounts, recvdisps, PargemslrMpiDataType<T>(), comm) );
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllgatherv(int *sendbuf, int count, int *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int PargemslrMpiAllgatherv(long int *sendbuf, int count, long int *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int PargemslrMpiAllgatherv(float *sendbuf, int count, float *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int PargemslrMpiAllgatherv(double *sendbuf, int count, double *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm)
   {
      int *recvdisps2, *recvcounts2;
      int np, i;
      MPI_Comm_size(comm, &np);
      PARGEMSLR_MALLOC( recvdisps2, np, kMemoryHost, int);
      PARGEMSLR_MALLOC( recvcounts2, np, kMemoryHost, int);
      
      for(i = 0 ; i < np ; i ++)
      {
         recvdisps2[i] = recvdisps[i] * 2;
         recvcounts2[i] = recvcounts[i] * 2;
      }
      
      PARGEMSLR_MPI_CALL( MPI_Allgatherv( sendbuf, 2*count, PargemslrMpiDataType<T>(), recvbuf, recvcounts2, recvdisps2, PargemslrMpiDataType<T>(), comm) );
      PARGEMSLR_FREE( recvdisps2, kMemoryHost);
      PARGEMSLR_FREE( recvcounts2, kMemoryHost);
      
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrMpiAllgatherv(complexs *sendbuf, int count, complexs *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int PargemslrMpiAllgatherv(complexd *sendbuf, int count, complexd *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
     
#endif

#ifdef PARGEMSLR_CUDA
   int PargemslrCudaSynchronize()
   {
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
#endif
   
   template <typename T> 
   MPI_Datatype PargemslrMpiDataType()
   {
      PARGEMSLR_ERROR("Unimplemented MPI_Datatype.");
      return MPI_DATATYPE_NULL;
   }
   
   template<>
   MPI_Datatype PargemslrMpiDataType<int>()
   {
      return MPI_INT;
   }
   
   template<>
   MPI_Datatype PargemslrMpiDataType<long int>()
   {
      return MPI_LONG;
   }
   
   template<>
   MPI_Datatype PargemslrMpiDataType<float>()
   {
      return MPI_FLOAT;
   }
   
   template<>
   MPI_Datatype PargemslrMpiDataType<double>()
   {
      return MPI_DOUBLE;
   }
   
   template<>
   MPI_Datatype PargemslrMpiDataType<complexs>()
   {
#ifdef MPI_C_FLOAT_COMPLEX
      return MPI_C_FLOAT_COMPLEX;
#else
      return MPI_FLOAT;
#endif
   }
   
   template<>
   MPI_Datatype PargemslrMpiDataType<complexd>()
   {
#ifdef MPI_C_DOUBLE_COMPLEX
      return MPI_C_DOUBLE_COMPLEX;
#else
      return MPI_DOUBLE;
#endif
   }
}
