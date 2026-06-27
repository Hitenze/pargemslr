#include "HYPRE_pargemslr.hpp"
#include "pargemslr_hypre_interface.hpp"

static HYPRE_Int
hypre_PargemslrMemoryLocation(HYPRE_MemoryLocation hypre_location, int *location)
{
   hypre_MemoryLocation actual_location = hypre_GetActualMemLocation(hypre_location);

   if(actual_location == hypre_MEMORY_HOST)
   {
      *location = HYPRE_PARGEMSLR_MEMORY_HOST;
      return 0;
   }

#ifdef PARGEMSLR_CUDA
   if(actual_location == hypre_MEMORY_DEVICE)
   {
      *location = HYPRE_PARGEMSLR_MEMORY_DEVICE;
      return 0;
   }

   if(actual_location == hypre_MEMORY_UNIFIED)
   {
      *location = HYPRE_PARGEMSLR_MEMORY_UNIFIED;
      return 0;
   }
#endif

   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "ParGeMSLR hypre interface received unsupported memory location");
   return hypre_error_flag;
}

static HYPRE_Int
hypre_PargemslrCheckSameLocation(int first, int second)
{
   if(first != second)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "ParGeMSLR hypre interface requires matching memory locations");
      return hypre_error_flag;
   }

   return 0;
}

HYPRE_Int
HYPRE_InitGemslr()
{
   hypre_PargemslrInitMPI(hypre_MPI_COMM_WORLD);
#ifdef PARGEMSLR_CUDA
   hypre_PargemslrInitGPU();
#endif
#ifdef PARGEMSLR_OPENMP
   hypre_PargemslrInitOpenMP(hypre_NumThreads());
#else
   /* still need to set the OpenBLAS or MKL num threads when necessary */
   hypre_PargemslrInitOpenMP(1);
#endif
   return hypre_error_flag;
}

HYPRE_Int
HYPRE_FinalizeGemslr()
{
   hypre_PargemslrFinalizeMPI();
   hypre_PargemslrFinalizeOpenMP();
#ifdef PARGEMSLR_CUDA
   hypre_PargemslrFinalizeGPU();
#endif
   return hypre_error_flag;
}

/* create the gemslr solver */
HYPRE_Int
HYPRE_GEMSLRCreate( HYPRE_Solver *solver )
{
   return HYPRE_GEMSLRCreateFromFile( solver, NULL );
}

/* create the gemslr solver */
HYPRE_Int
HYPRE_GEMSLRCreateFromFile( HYPRE_Solver *solver, const char* filename )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   
   double *params;
   
   /* read the default input array */
   if(filename == NULL)
   {
      char tempfilename[1024];
      snprintf( tempfilename, 1024, "inputs");
      params = hypre_PargemslrCreateParameterArrayFromFile(tempfilename);
   }
   else
   {
      char tempfilename[1024];
      snprintf( tempfilename, 1024, "%s",filename);
      params = hypre_PargemslrCreateParameterArrayFromFile(tempfilename);
   }
   
   if( sizeof(HYPRE_Complex) == sizeof(hypre_double) )
   {
      /* double precision solver */
      *solver = ( (HYPRE_Solver) hypre_PargemslrParallelGEMSLRDCreate( ) );
      hypre_PargemslrParallelGEMSLRDSetParams(*solver, params);
   }
   else
   {
      /* single precision solver */
      *solver = ( (HYPRE_Solver) hypre_PargemslrParallelGEMSLRSCreate( ) );
      hypre_PargemslrParallelGEMSLRSSetParams(*solver, params);
   }
   
   /* free the parameter array */
   hypre_PargemslrDestroyParameterArray(params);
   
   return hypre_error_flag;
}

HYPRE_Int
HYPRE_GEMSLRDestroy( HYPRE_Solver solver )
{
   if( sizeof(HYPRE_Complex) == sizeof(hypre_double) )
   {
      return( hypre_PargemslrParallelGEMSLRDDestroy( (void *) solver ) );
   }
   else
   {
      return( hypre_PargemslrParallelGEMSLRSDestroy( (void *) solver ) );
   }
}

HYPRE_Int
HYPRE_GEMSLRSetup( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   hypre_Vector *b_local = hypre_ParVectorLocalVector(b);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   int matrix_location, offd_location, b_location, x_location;
   HYPRE_Int ierr;

   ierr = hypre_PargemslrMemoryLocation(hypre_CSRMatrixMemoryLocation(diag), &matrix_location);
   if(ierr) { return ierr; }

   ierr = hypre_PargemslrMemoryLocation(hypre_CSRMatrixMemoryLocation(offd), &offd_location);
   if(ierr) { return ierr; }

   ierr = hypre_PargemslrMemoryLocation(hypre_VectorMemoryLocation(b_local), &b_location);
   if(ierr) { return ierr; }

   ierr = hypre_PargemslrMemoryLocation(hypre_VectorMemoryLocation(x_local), &x_location);
   if(ierr) { return ierr; }

   ierr = hypre_PargemslrCheckSameLocation(matrix_location, offd_location);
   if(ierr) { return ierr; }

   ierr = hypre_PargemslrCheckSameLocation(matrix_location, b_location);
   if(ierr) { return ierr; }

   ierr = hypre_PargemslrCheckSameLocation(matrix_location, x_location);
   if(ierr) { return ierr; }

   if( sizeof(HYPRE_Complex) == sizeof(hypre_double) )
   {
      HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *parcsr_mat;
      int i, n_offd_map = hypre_CSRMatrixNumCols(offd);
      int diag_nnz = hypre_CSRMatrixNumNonzeros(diag);
      int offd_nnz = hypre_CSRMatrixNumNonzeros(offd);
      long int *offd_map;
      HYPRE_BigInt *offd_map_big = hypre_ParCSRMatrixColMapOffd(A);
      HYPRE_Complex *b_data = hypre_VectorData(b_local);
      HYPRE_Complex *x_data = hypre_VectorData(x_local);

      offd_map = hypre_TAlloc( long int, n_offd_map, HYPRE_MEMORY_HOST);
      for(i = 0 ; i < n_offd_map ; i ++)
      {
         offd_map[i] = (long int)offd_map_big[i];
      }

      parcsr_mat = hypre_PargemslrParallelCsrMatrixDCreate(
                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                    hypre_ParCSRMatrixGlobalNumCols(A),
                                    hypre_ParCSRMatrixRowStarts(A)[0],
                                    hypre_ParCSRMatrixColStarts(A)[0],
                                    hypre_ParCSRMatrixNumRows(A),
                                    hypre_ParCSRMatrixNumCols(A),
                                    diag_nnz,
                                    offd_nnz,
                                    hypre_CSRMatrixI(diag),
                                    hypre_CSRMatrixJ(diag),
                                    (double*)hypre_CSRMatrixData(diag),
                                    hypre_CSRMatrixI(offd),
                                    hypre_CSRMatrixJ(offd),
                                    (double*)hypre_CSRMatrixData(offd),
                                    n_offd_map,
                                    offd_map,
                                    hypre_ParCSRMatrixComm(A),
                                    matrix_location);

      hypre_TFree(offd_map, HYPRE_MEMORY_HOST);

      return hypre_PargemslrParallelGEMSLRDSetup( (void *) solver,
                                             parcsr_mat,
                                             (double*)x_data,
                                             (double*)b_data,
                                             matrix_location );
   }
   else
   {
      HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *parcsr_mat;
      int i, n_offd_map = hypre_CSRMatrixNumCols(offd);
      int diag_nnz = hypre_CSRMatrixNumNonzeros(diag);
      int offd_nnz = hypre_CSRMatrixNumNonzeros(offd);
      long int *offd_map;
      HYPRE_BigInt *offd_map_big = hypre_ParCSRMatrixColMapOffd(A);
      HYPRE_Complex *b_data = hypre_VectorData(b_local);
      HYPRE_Complex *x_data = hypre_VectorData(x_local);

      offd_map = hypre_TAlloc( long int, n_offd_map, HYPRE_MEMORY_HOST);
      for(i = 0 ; i < n_offd_map ; i ++)
      {
         offd_map[i] = (long int)offd_map_big[i];
      }

      parcsr_mat = hypre_PargemslrParallelCsrMatrixSCreate(
                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                    hypre_ParCSRMatrixGlobalNumCols(A),
                                    hypre_ParCSRMatrixRowStarts(A)[0],
                                    hypre_ParCSRMatrixColStarts(A)[0],
                                    hypre_ParCSRMatrixNumRows(A),
                                    hypre_ParCSRMatrixNumCols(A),
                                    diag_nnz,
                                    offd_nnz,
                                    hypre_CSRMatrixI(diag),
                                    hypre_CSRMatrixJ(diag),
                                    (float*)hypre_CSRMatrixData(diag),
                                    hypre_CSRMatrixI(offd),
                                    hypre_CSRMatrixJ(offd),
                                    (float*)hypre_CSRMatrixData(offd),
                                    n_offd_map,
                                    offd_map,
                                    hypre_ParCSRMatrixComm(A),
                                    matrix_location);

      hypre_TFree(offd_map, HYPRE_MEMORY_HOST);

      return hypre_PargemslrParallelGEMSLRSSetup( (void *) solver,
                                             parcsr_mat,
                                             (float*)x_data,
                                             (float*)b_data,
                                             matrix_location );
   }
}

HYPRE_Int
HYPRE_GEMSLRSolve( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   hypre_Vector *b_local = hypre_ParVectorLocalVector(b);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   int b_location, x_location;
   HYPRE_Int ierr;

   ierr = hypre_PargemslrMemoryLocation(hypre_VectorMemoryLocation(b_local), &b_location);
   if(ierr) { return ierr; }

   ierr = hypre_PargemslrMemoryLocation(hypre_VectorMemoryLocation(x_local), &x_location);
   if(ierr) { return ierr; }

   ierr = hypre_PargemslrCheckSameLocation(b_location, x_location);
   if(ierr) { return ierr; }

   if( sizeof(HYPRE_Complex) == sizeof(hypre_double) )
   {
      HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *parcsr_mat = NULL;
      HYPRE_Complex *b_data = hypre_VectorData(b_local);
      HYPRE_Complex *x_data = hypre_VectorData(x_local);

      return hypre_PargemslrParallelGEMSLRDSolve( (void *) solver,
                                                parcsr_mat,
                                                (double*)x_data,
                                                (double*)b_data,
                                                b_location );
   }
   else
   {
      HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *parcsr_mat = NULL;
      HYPRE_Complex *b_data = hypre_VectorData(b_local);
      HYPRE_Complex *x_data = hypre_VectorData(x_local);

      return hypre_PargemslrParallelGEMSLRSSolve( (void *) solver,
                                                parcsr_mat,
                                                (float*)x_data,
                                                (float*)b_data,
                                                b_location );
   }
}
