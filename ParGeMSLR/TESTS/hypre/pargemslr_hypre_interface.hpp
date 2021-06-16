#ifndef PARGEMSLR_HYPRE_INTERFACE_H
#define PARGEMSLR_HYPRE_INTERFACE_H

#include <mpi.h>
#include <string.h>

/**
 * @file pargemslr_hypre_interface.hpp
 * @brief The wrapper functions for hypre
 */

#ifndef C_PARGEMSLR_PARALLEL_GMRES_S
#define C_PARGEMSLR_PARALLEL_GMRES_S         void
#define C_PARGEMSLR_PARALLEL_GEMSLR_S        void
#define C_PARGEMSLR_PARALLEL_CSR_MATRIX_S    void
#define C_PARGEMSLR_PARALLEL_GMRES_D         void
#define C_PARGEMSLR_PARALLEL_GEMSLR_D        void
#define C_PARGEMSLR_PARALLEL_CSR_MATRIX_D    void
#endif

#ifndef HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S
#define HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S        void
#define HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S    void
#define HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D        void
#define HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D    void
#endif

/* utils */

/**
 * @brief   Setup MPI data struct with MPI_Comm, MPI_Init won't be called inside.
 * @details Setup MPI data struct with MPI_Comm, MPI_Init won't be called inside.
 * @param [in]  	comm The MPI_Comm, typically should be MPI_COMM_WORLD.
 * @return           Return error message.                                                                                                                                                
 */
int hypre_PargemslrInitMPI(MPI_Comm comm);

/**
 * @brief   Setup OpenMP and MKL data struct.
 * @details Setup OpenMP and MKL data struct.
 * @param [in]  	nthreads The number of threads.
 * @return           Return error message.                                                                                                                                                
 */
int hypre_PargemslrInitOpenMP(int nthreads);

/**
 * @brief   Setup CUDA data struct.
 * @details Setup CUDA data struct.
 * @return           Return error message.                                                                                                                                                
 */
int hypre_PargemslrInitGPU();

/**
 * @brief   Finalize MPI data struct. MPI_Finalize won't be called inside.
 * @details Finalize MPI data struct. MPI_Finalize won't be called inside.
 * @return           Return error message.                                                                                                                                                
 */
int hypre_PargemslrFinalizeMPI();

/**
 * @brief   Finalize OpenMP and MKL data struct.
 * @details Finalize OpenMP and MKL data struct.
 * @return           Return error message.                                                                                                                                                
 */
int hypre_PargemslrFinalizeOpenMP();

/**
 * @brief   Finalize CUDA data struct.
 * @details Finalize CUDA data struct.
 * @return           Return error message.                                                                                                                                                
 */
int hypre_PargemslrFinalizeGPU();

/* parallel_csr_matrix */

/**
 * @brief   Destroy Single Precision Parallel CSR matrix.
 * @details Destroy Single Precision Parallel CSR matrix.
 * @param [in] fgmres_data Pointer to the Parallel CSR matrix.
 * @return           Return error message.
 */
int
hypre_PargemslrParallelCsrMatrixSDestroy(HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *matrix);

/**
 * @brief   Destroy Double Precision Parallel CSR matrix.
 * @details Destroy Double Precision Parallel CSR matrix.
 * @param [in] fgmres_data Pointer to the Parallel CSR matrix.
 * @return           Return error message.
 */
int
hypre_PargemslrParallelCsrMatrixDDestroy(HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *matrix);

/**
 * @brief   Create Single Precision Parallel CSR Matrix.
 * @details Create Single Precision Parallel CSR Matrix.
 * @param [in]       nrow_global The number of global rows.
 * @param [in]       ncol_global The number of global cols.
 * @param [in]       nrow_start The number of first global row.
 * @param [in]       ncol_start The number of first global col.
 * @param [in]       diag_i The diagonal matrix row pointer of CSR format.
 * @param [in]       diag_j The diagonal matrix col array of CSR format.
 * @param [in]       diag_data The diagonal matrix data array of CSR format.
 * @param [in]       offd_i The off-diagonal matrix row pointer of CSR format.
 * @param [in]       offd_j The off-diagonal matrix col array of CSR format.
 * @param [in]       offd_data The off-diagonal matrix data array of CSR format.
 * @param [in]       comm The MPI_Comm.
 * @param [in]       data_location The data location. 0 for host, 1 for device, default is host.
 * @return           Return pointer to the parallel csr matrix.
 */
HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S* 
hypre_PargemslrParallelCsrMatrixSCreate( long int nrow_global,
                                          long int ncol_global,
                                          long int nrow_start,
                                          long int ncol_start,
                                          int nrow_local,
                                          int ncol_local,
                                          int *diag_i,
                                          int *diag_j,
                                          float *diag_data,
                                          int *offd_i,
                                          int *offd_j,
                                          float *offd_data,
                                          int n_offd_map,
                                          long int *offd_map,
                                          MPI_Comm comm,
                                          int data_location);

/**
 * @brief   Create Double Precision Parallel CSR Matrix.
 * @details Create Double Precision Parallel CSR Matrix.
 * @param [in]       nrow_global The number of global rows.
 * @param [in]       ncol_global The number of global cols.
 * @param [in]       nrow_start The number of first global row.
 * @param [in]       ncol_start The number of first global col.
 * @param [in]       diag_i The diagonal matrix row pointer of CSR format.
 * @param [in]       diag_j The diagonal matrix col array of CSR format.
 * @param [in]       diag_data The diagonal matrix data array of CSR format.
 * @param [in]       offd_i The off-diagonal matrix row pointer of CSR format.
 * @param [in]       offd_j The off-diagonal matrix col array of CSR format.
 * @param [in]       offd_data The off-diagonal matrix data array of CSR format.
 * @param [in]       comm The MPI_Comm.
 * @param [in]       data_location The data location. 0 for host, 1 for device, default is host.
 * @return           Return pointer to the parallel csr matrix.
 */
HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D* 
hypre_PargemslrParallelCsrMatrixDCreate( long int nrow_global,
                                          long int ncol_global,
                                          long int nrow_start,
                                          long int ncol_start,
                                          int nrow_local,
                                          int ncol_local,
                                          int *diag_i,
                                          int *diag_j,
                                          double *diag_data,
                                          int *offd_i,
                                          int *offd_j,
                                          double *offd_data,
                                          int n_offd_map,
                                          long int *offd_map,
                                          MPI_Comm comm,
                                          int data_location);

/**
 * @brief   Create Single Precision ParGEMSLR preconditioner.
 * @details Create Single Precision ParGEMSLR preconditioner.
 * @return           Return the ParGEMSLR preconditioner.
 */
HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S* 
hypre_PargemslrParallelGEMSLRSCreate();

/**
 * @brief   Create Double Precision ParGEMSLR preconditioner.
 * @details Create Double Precision ParGEMSLR preconditioner.
 * @return           Return the ParGEMSLR preconditioner.
 */
HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D* 
hypre_PargemslrParallelGEMSLRDCreate();

/**
 * @brief   Destroy Single Precision ParGEMSLR preconditioner.
 * @details Destroy Single Precision ParGEMSLR preconditioner.
 * @param [in] pargemslr_data Pointer to the ParGEMSLR preconditioner.
 * @return           Return error message.
 */
int   
hypre_PargemslrParallelGEMSLRSDestroy(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S *pargemslr_data);

/**
 * @brief   Destroy Double Precision ParGEMSLR preconditioner.
 * @details Destroy Double Precision ParGEMSLR preconditioner.
 * @param [in] pargemslr_data Pointer to the ParGEMSLR preconditioner.
 * @return           Return error message.
 */
int   
hypre_PargemslrParallelGEMSLRDDestroy(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D *pargemslr_data);

/**
 * @brief   Setup phase of the Single Precision ParGEMSLR preconditioner.
 * @details Setup phase of the Single Precision ParGEMSLR preconditioner. Note that is FlexGMRES in the ParGEMSLR package is used,
 *          there is no need to call this function.
 * @param [in] pargemslr_data Pointer to the ParGEMSLR preconditioner.
 * @param [in] matrix Pointer to the parallel csr matrix.
 * @param [in, out]  x The initial guess and the solution.
 * @param [in]       rhs The right-hand-side.
 * @return           Return error message.
 */
int   
hypre_PargemslrParallelGEMSLRSSetup(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S *pargemslr_data, 
                           HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *matrix, 
                           float *x,
                           float *rhs);

/**
 * @brief   Setup phase of the Double Precision ParGEMSLR preconditioner.
 * @details Setup phase of the Double Precision ParGEMSLR preconditioner. Note that is FlexGMRES in the ParGEMSLR package is used,
 *          there is no need to call this function.
 * @param [in] pargemslr_data Pointer to the ParGEMSLR preconditioner.
 * @param [in] matrix Pointer to the parallel csr matrix.
 * @param [in, out]  x The initial guess and the solution.
 * @param [in]       rhs The right-hand-side.
 * @return           Return error message.
 */
int   
hypre_PargemslrParallelGEMSLRDSetup(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D *pargemslr_data, 
                           HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *matrix, 
                           double *x,
                           double *rhs);

/**
 * @brief   Solve phase of the Single Precision ParGEMSLR preconditioner. Only one single solve M^{-1}rhs would be applied.
 * @details Solve phase of the Single Precision ParGEMSLR preconditioner. Only one single solve M^{-1}rhs would be applied.
 * @param [in] pargemslr_data Pointer to the ParGEMSLR preconditioner.
 * @param [in] matrix Pointer to the parallel csr matrix.
 * @param [in, out]  x The initial guess and the solution.
 * @param [in]       rhs The right-hand-side.
 * @return           Return error message.
 */
int   
hypre_PargemslrParallelGEMSLRSSolve(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S *pargemslr_data, 
                           HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *matrix, 
                           float *x,
                           float *rhs);

/**
 * @brief   Solve phase of the Double Precision ParGEMSLR preconditioner. Only one single solve M^{-1}rhs would be applied.
 * @details Solve phase of the Double Precision ParGEMSLR preconditioner. Only one single solve M^{-1}rhs would be applied.
 * @param [in] pargemslr_data Pointer to the ParGEMSLR preconditioner.
 * @param [in] matrix Pointer to the parallel csr matrix.
 * @param [in, out]  x The initial guess and the solution.
 * @param [in]       rhs The right-hand-side.
 * @return           Return error message.
 */
int   
hypre_PargemslrParallelGEMSLRDSolve(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D *pargemslr_data, 
                           HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *matrix, 
                           double *x,
                           double *rhs);

/**
 * @brief   Setup with parameter array.
 * @details Setup with parameter array.
 * @param [in] pargemslr_data Pointer to the ParGEMSLR preconditioner.
 * @param [in] params The parameter array.
 * @return     Return error message.
 */
int
hypre_PargemslrParallelGEMSLRSSetParams(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S *pargemslr_data, 
                                    double *params);

/**
 * @brief   Setup with parameter array.
 * @details Setup with parameter array.
 * @param [in] pargemslr_data Pointer to the ParGEMSLR preconditioner.
 * @param [in] params The parameter array.
 * @return     Return error message.
 */
int
hypre_PargemslrParallelGEMSLRDSetParams(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D *pargemslr_data, 
                                    double *params);

/**
 * @brief   Read setup array from file.
 * @details Read setup array from file.
 * @param [in]   filename The name of the file.
 * @return     Return the setup array.
 */
double*
hypre_PargemslrCreateParameterArrayFromFile(char *filename);

/**
 * @brief   Destroy setup array.
 * @details Destroy setup array.
 * @param [in] params The parameter array.
 * @return     Return error message.
 */
int
hypre_PargemslrDestroyParameterArray(double *params);

#endif

