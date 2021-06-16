#include "_hypre_utilities.hpp"
#include "HYPRE_pargemslr.hpp"
#include "pargemslr_hypre_interface.hpp"

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
HYPRE_GEMSLRCreateFromFile( HYPRE_Solver *solver, char* filename )
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
      params = hypre_PargemslrCreateParameterArrayFromFile(filename);
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
   
   int location = 0;
   
#ifdef PARGEMSLR_CUDA
   /* device */
   location = 1;
#endif
   
   if( sizeof(HYPRE_Complex) == sizeof(hypre_double) )
   {
      HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *parcsr_mat;
      
      int i, n_offd_map = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
      long int *offd_map;
      HYPRE_BigInt *offd_map_big = hypre_ParCSRMatrixColMapOffd(A);
      
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
                                    hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A)),
                                    hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A)),
                                    (double*)hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A)),
                                    hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A)),
                                    hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A)),
                                    (double*)hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A)),
                                    n_offd_map,
                                    offd_map,
                                    hypre_ParCSRMatrixComm(A),
                                    location);
      
      hypre_TFree(offd_map, HYPRE_MEMORY_HOST);
      
      HYPRE_Complex *b_data = hypre_VectorData(hypre_ParVectorLocalVector(b));
      HYPRE_Complex *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
      
      hypre_PargemslrParallelGEMSLRDSetup( (void *) solver,
                                             parcsr_mat,
                                             (double*)x_data,
                                             (double*)b_data );
   }
   else
   {
      HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *parcsr_mat;
      
      int i, n_offd_map = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
      long int *offd_map;
      HYPRE_BigInt *offd_map_big = hypre_ParCSRMatrixColMapOffd(A);
      
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
                                    hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A)),
                                    hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A)),
                                    (float*)hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A)),
                                    hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A)),
                                    hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A)),
                                    (float*)hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A)),
                                    n_offd_map,
                                    offd_map,
                                    hypre_ParCSRMatrixComm(A),
                                    location);
      
      hypre_TFree(offd_map, HYPRE_MEMORY_HOST);
      
      HYPRE_Complex *b_data = hypre_VectorData(hypre_ParVectorLocalVector(b));
      HYPRE_Complex *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
      
      hypre_PargemslrParallelGEMSLRSSetup( (void *) solver,
                                             parcsr_mat,
                                             (float*)x_data,
                                             (float*)b_data );
   }
   return 0;
}

HYPRE_Int
HYPRE_GEMSLRSolve( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   if( sizeof(HYPRE_Complex) == sizeof(hypre_double) )
   {
      HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *parcsr_mat = NULL;
      
      HYPRE_Complex *b_data = hypre_VectorData(hypre_ParVectorLocalVector(b));
      HYPRE_Complex *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
      
      return( hypre_PargemslrParallelGEMSLRDSolve( (void *) solver,
                                                parcsr_mat,
                                                (double*)x_data,
                                                (double*)b_data ) );
   }
   else
   {
      HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *parcsr_mat = NULL;
      
      HYPRE_Complex *b_data = hypre_VectorData(hypre_ParVectorLocalVector(b));
      HYPRE_Complex *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
      
      return( hypre_PargemslrParallelGEMSLRSSolve( (void *) solver,
                                                parcsr_mat,
                                                (float*)x_data,
                                                (float*)b_data ) );
   }
}
