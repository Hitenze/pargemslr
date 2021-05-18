#include "pargemslr_hypre_interface.hpp"
#include "../../INC/pargemslr.hpp"
#include <string.h>

using namespace pargemslr;

int hypre_PargemslrInitMPI(MPI_Comm comm)
{
   return PargemslrInitMpi(comm);
}

int hypre_PargemslrInitOpenMP(int nthreads)
{
   return PargemslrInitOpenMP(nthreads);
}

int hypre_PargemslrInitGPU()
{
   return PargemslrInitCUDA();
}

int hypre_PargemslrFinalizeMPI()
{
   return PargemslrFinalizeMpi();
}

int hypre_PargemslrFinalizeOpenMP()
{
   return PargemslrFinalizeOpenMP();
}

int hypre_PargemslrFinalizeGPU()
{
   return PargemslrFinalizeCUDA();
}

int
hypre_PargemslrParallelCsrMatrixSDestroy(HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *matrix)
{
   if(matrix)
   {
      matrix_csr_par_float *parcsr_mat = (matrix_csr_par_float*) matrix;
      parcsr_mat->Clear();
      delete parcsr_mat;
   }
   
   return 0;
}

int
hypre_PargemslrParallelCsrMatrixDDestroy(HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *matrix)
{
   if(matrix)
   {
      matrix_csr_par_double *parcsr_mat = (matrix_csr_par_double*) matrix;
      parcsr_mat->Clear();
      delete parcsr_mat;
   }
   
   return 0;
}


int
C_PargemslrParallelCsrMatrixDDestroyWrapper(C_PARGEMSLR_PARALLEL_CSR_MATRIX_D *matrix)
{
   if(matrix)
   {
      matrix_csr_par_double *parcsr_mat = (matrix_csr_par_double*) matrix;
      parcsr_mat->Clear();
      delete parcsr_mat;
   }
   
   return 0;
}

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
                                          int data_location)
{
   int location = kMemoryHost, diag_nnz, offd_nnz;
   matrix_csr_par_float *parcsr_mat = new matrix_csr_par_float();
   parallel_log parlog(comm);
   
   if(data_location == 1)
   {
      location = kMemoryDevice;
   }
   
   parcsr_mat->Setup(nrow_local, nrow_start, nrow_global, ncol_local, ncol_start, ncol_global, parlog);
   
   matrix_csr_float &diag_mat = parcsr_mat->GetDiagMat();
   matrix_csr_float &offd_mat = parcsr_mat->GetOffdMat();
   vector_long &offd_map_v = parcsr_mat->GetOffdMap();
   
   diag_nnz = diag_i[nrow_local];
   offd_nnz = offd_i[nrow_local];
   
   /* create data str */
   diag_mat.Setup(nrow_local, ncol_local, diag_nnz);
   offd_mat.Setup(nrow_local, n_offd_map, offd_nnz);
   offd_map_v.Setup(n_offd_map);
   
   /* copy diag data */
   PARGEMSLR_MEMCPY( diag_mat.GetI(), diag_i, nrow_local+1, kMemoryHost, location, int);
   PARGEMSLR_MEMCPY( diag_mat.GetJ(), diag_j, diag_nnz, kMemoryHost, location, int);
   PARGEMSLR_MEMCPY( diag_mat.GetData(), diag_data, diag_nnz, kMemoryHost, location, float);
   
   /* copy offd data */
   PARGEMSLR_MEMCPY( offd_mat.GetI(), offd_i, nrow_local+1, kMemoryHost, location, int);
   PARGEMSLR_MEMCPY( offd_mat.GetJ(), offd_j, offd_nnz, kMemoryHost, location, int);
   PARGEMSLR_MEMCPY( offd_mat.GetData(), offd_data, offd_nnz, kMemoryHost, location, float);
   
   /* copy offd map */
   PARGEMSLR_MEMCPY( offd_map_v.GetData(), offd_map, n_offd_map, kMemoryHost, location, long int);
   
   parcsr_mat->SetupMatvec();
   
   parcsr_mat->MoveData(location);
   
   parlog.Clear();
   
   return (void*)parcsr_mat;
}


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
                                          int data_location)
{
   int location = kMemoryHost, diag_nnz, offd_nnz;
   matrix_csr_par_double *parcsr_mat = new matrix_csr_par_double();
   parallel_log parlog(comm);
   
   if(data_location == 1)
   {
      location = kMemoryDevice;
   }
   
   parcsr_mat->Setup(nrow_local, nrow_start, nrow_global, ncol_local, ncol_start, ncol_global, parlog);
   
   matrix_csr_double &diag_mat = parcsr_mat->GetDiagMat();
   matrix_csr_double &offd_mat = parcsr_mat->GetOffdMat();
   vector_long &offd_map_v = parcsr_mat->GetOffdMap();
   
   diag_nnz = diag_i[nrow_local];
   offd_nnz = offd_i[nrow_local];
   
   /* create data str */
   diag_mat.Setup(nrow_local, ncol_local, diag_nnz);
   offd_mat.Setup(nrow_local, n_offd_map, offd_nnz);
   offd_map_v.Setup(n_offd_map);
   
   /* copy diag data */
   PARGEMSLR_MEMCPY( diag_mat.GetI(), diag_i, nrow_local+1, kMemoryHost, location, int);
   PARGEMSLR_MEMCPY( diag_mat.GetJ(), diag_j, diag_nnz, kMemoryHost, location, int);
   PARGEMSLR_MEMCPY( diag_mat.GetData(), diag_data, diag_nnz, kMemoryHost, location, double);
   
   /* copy offd data */
   PARGEMSLR_MEMCPY( offd_mat.GetI(), offd_i, nrow_local+1, kMemoryHost, location, int);
   PARGEMSLR_MEMCPY( offd_mat.GetJ(), offd_j, offd_nnz, kMemoryHost, location, int);
   PARGEMSLR_MEMCPY( offd_mat.GetData(), offd_data, offd_nnz, kMemoryHost, location, double);
   
   /* copy offd map */
   PARGEMSLR_MEMCPY( offd_map_v.GetData(), offd_map, n_offd_map, kMemoryHost, location, long int);
   
   parcsr_mat->SetupMatvec();
   
   parcsr_mat->MoveData(location);
   
   parlog.Clear();
   
   return (void*)parcsr_mat;
}

HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S* 
hypre_PargemslrParallelGEMSLRSCreate()
{
   precond_gemslr_csr_par_float *gemslr = new precond_gemslr_csr_par_float();
   
   return (void*)gemslr;
}

HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D* 
hypre_PargemslrParallelGEMSLRDCreate()
{
   precond_gemslr_csr_par_double *gemslr = new precond_gemslr_csr_par_double();
   
   return (void*)gemslr;
}

int   
hypre_PargemslrParallelGEMSLRSDestroy(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S *pargemslr_data)
{
   if(pargemslr_data)
   {
      precond_gemslr_csr_par_float *gemslr = (precond_gemslr_csr_par_float*) pargemslr_data;
      gemslr->Clear();
      delete gemslr;
   }
   
   return 0;
}

int   
hypre_PargemslrParallelGEMSLRDDestroy(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D *pargemslr_data)
{
   if(pargemslr_data)
   {
      precond_gemslr_csr_par_double *gemslr = (precond_gemslr_csr_par_double*) pargemslr_data;
      gemslr->Clear();
      delete gemslr;
   }
   
   return 0;
}

int   
hypre_PargemslrParallelGEMSLRSSetup(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S *pargemslr_data, 
                           HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *matrix, 
                           float *x,
                           float *rhs)
{
   
   precond_gemslr_csr_par_float *gemslr = (precond_gemslr_csr_par_float*) pargemslr_data;
   matrix_csr_par_float *parcsr_mat = (matrix_csr_par_float*) matrix;
   
   int location = parcsr_mat->GetDataLocation();
   
   vector_par_float par_x, par_rhs;
   
   par_x.SetupPtrStr(*parcsr_mat);
   par_rhs.SetupPtrStr(*parcsr_mat);
   
   par_x.UpdatePtr(x, location);
   par_rhs.UpdatePtr(rhs, location);
   
   gemslr->SetMatrixP(parcsr_mat);
   gemslr->SetOwnMatrix(true);
   gemslr->SetSolveLocation(location);
   
   gemslr->Setup(par_x, par_rhs);
   
   par_x.Clear();
   par_rhs.Clear();
   
   return 0;
}

int   
hypre_PargemslrParallelGEMSLRDSetup(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D *pargemslr_data, 
                           HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *matrix, 
                           double *x,
                           double *rhs)
{
   
   precond_gemslr_csr_par_double *gemslr = (precond_gemslr_csr_par_double*) pargemslr_data;
   matrix_csr_par_double *parcsr_mat = (matrix_csr_par_double*) matrix;
   
   int location = parcsr_mat->GetDataLocation();
   
   vector_par_double par_x, par_rhs;
   
   par_x.SetupPtrStr(*parcsr_mat);
   par_rhs.SetupPtrStr(*parcsr_mat);
   
   par_x.UpdatePtr(x, location);
   par_rhs.UpdatePtr(rhs, location);
   
   gemslr->SetMatrixP(parcsr_mat);
   gemslr->SetOwnMatrix(false);
   gemslr->SetSolveLocation(location);
   
   gemslr->Setup(par_x, par_rhs);
   
   par_x.Clear();
   par_rhs.Clear();
   
   par_x.Clear();
   par_rhs.Clear();
   
   return 0;
}

int   
hypre_PargemslrParallelGEMSLRSSolve(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S *pargemslr_data, 
                           HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_S *matrix, 
                           float *x,
                           float *rhs)
{
   
   precond_gemslr_csr_par_float *gemslr = (precond_gemslr_csr_par_float*) pargemslr_data;
   //matrix_csr_par_double *parcsr_mat = (matrix_csr_par_double*) matrix;
   matrix_csr_par_float *parcsr_mat = (matrix_csr_par_float*) gemslr->GetMatrix();
   
   int location = parcsr_mat->GetDataLocation();
   
   vector_par_float par_x, par_rhs;
   
   par_x.SetupPtrStr(*parcsr_mat);
   par_rhs.SetupPtrStr(*parcsr_mat);
   
   par_x.UpdatePtr(x, location);
   par_rhs.UpdatePtr(rhs, location);
   
   gemslr->Solve(par_x, par_rhs);
   
   par_x.Clear();
   par_rhs.Clear();
   
   return 0;
   
}
  
int
hypre_PargemslrParallelGEMSLRDSolve(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D *pargemslr_data, 
                           HYPRE_PARGEMSLR_PARALLEL_CSR_MATRIX_D *matrix, 
                           double *x,
                           double *rhs)
{
   
   precond_gemslr_csr_par_double *gemslr = (precond_gemslr_csr_par_double*) pargemslr_data;
   //matrix_csr_par_double *parcsr_mat = (matrix_csr_par_double*) matrix;
   matrix_csr_par_double *parcsr_mat = (matrix_csr_par_double*) gemslr->GetMatrix();
   
   int location = parcsr_mat->GetDataLocation();
   
   vector_par_double par_x, par_rhs;
   
   par_x.SetupPtrStr(*parcsr_mat);
   par_rhs.SetupPtrStr(*parcsr_mat);
   
   par_x.UpdatePtr(x, location);
   par_rhs.UpdatePtr(rhs, location);
   
   gemslr->Solve(par_x, par_rhs);
   
   par_x.Clear();
   par_rhs.Clear();
   
   return 0;
   
}

int
hypre_PargemslrParallelGEMSLRSSetParams(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_S *pargemslr_data, 
                                    double *params)
{
   precond_gemslr_csr_par_float *gemslr = (precond_gemslr_csr_par_float*) pargemslr_data;
   gemslr->SetWithParameterArray(params);
   return 0;
}

int
hypre_PargemslrParallelGEMSLRDSetParams(HYPRE_PARGEMSLR_PARALLEL_GEMSLR_D *pargemslr_data, 
                                    double *params)
{
   precond_gemslr_csr_par_double *gemslr = (precond_gemslr_csr_par_double*) pargemslr_data;
   gemslr->SetWithParameterArray(params);
   return 0;
}

int read_inputs_from_file(const char *filename, double *params)
{
   int linenum;
   FILE *f;
   char line[1024];
   char *word;
   
   if ((f = fopen(filename, "r")) == NULL)
   {
      PARGEMSLR_PRINT("Can't open file.\n");
      return PARGEMSLR_ERROR_IO_ERROR;
   }
   
   linenum = 0;
   while(fgets(line, 1024, f))
   {
      linenum++;
      switch(linenum)
      {
         case 1:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "BJ"))
            {
               params[PARGEMSLR_IO_PRECOND_GLOBAL_PRECOND] = kGemslrGlobalPrecondBJ;
            }
            else if(!strcmp(word, "ESCHUR"))
            {
               params[PARGEMSLR_IO_PRECOND_GLOBAL_PRECOND] = kGemslrGlobalPrecondESMSLR;
            }
            else if(!strcmp(word, "GEMSLR"))
            {
               params[PARGEMSLR_IO_PRECOND_GLOBAL_PRECOND] = kGemslrGlobalPrecondGeMSLR;
            }
            else if(!strcmp(word, "PSLR"))
            {
               params[PARGEMSLR_IO_PRECOND_GLOBAL_PRECOND] = kGemslrGlobalPrecondPSLR;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 2:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_PREPOSS_GLOBAL_PARTITION]);
            break;
         }
         case 3:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "ILUT"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1] = kGemslrBSolveILUT;
            }
            else if(!strcmp(word, "ILUK"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1] = kGemslrBSolveILUK;
            }
            else if(!strcmp(word, "GEMSLR"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1] = kGemslrBSolveGemslr;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 4:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1_LEVEL]);
            break;
         }
         case 5:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "ILUT"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND2] = kGemslrBSolveILUT;
            }
            else if(!strcmp(word, "ILUK"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND2] = kGemslrBSolveILUK;
            }
            else if(!strcmp(word, "GEMSLR"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND2] = kGemslrBSolveGemslr;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 6:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "ILUT"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3] = kGemslrCSolveILUT;
            }
            else if(!strcmp(word, "ILUK"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3] = kGemslrCSolveILUK;
            }
            else if(!strcmp(word, "BJILUT"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3] = kGemslrCSolveBJILUT;
            }
            else if(!strcmp(word, "BJILUK"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3] = kGemslrCSolveBJILUK;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 7:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_POLY_ORDER]);
            break;
         }
         case 8:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_PREPOSS_NLEV_GLOBAL], &params[PARGEMSLR_IO_PREPOSS_NLEV_LOCAL]);
            break;
         }
         case 9:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_PREPOSS_NCOMP_GLOBAL], &params[PARGEMSLR_IO_PREPOSS_NCOMP_LOCAL]);
            break;
         }
         case 10:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "ND"))
            {
               params[PARGEMSLR_IO_PREPOSS_PARTITION_GLOBAL] = kGemslrPartitionND;
            }
            else if(!strcmp(word, "RKWAY"))
            {
               params[PARGEMSLR_IO_PREPOSS_PARTITION_GLOBAL] = kGemslrPartitionRKway;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum+2);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 11:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "ND"))
            {
               params[PARGEMSLR_IO_PREPOSS_PARTITION_LOCAL] = kGemslrPartitionND;
            }
            else if(!strcmp(word, "RKWAY"))
            {
               params[PARGEMSLR_IO_PREPOSS_PARTITION_LOCAL] = kGemslrPartitionRKway;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum+2);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 12:
         {
            sscanf(line, "%lf %lf %lf %lf", &params[PARGEMSLR_IO_ILU_DROPTOL_B_GLOBAL], 
                                             &params[PARGEMSLR_IO_ILU_DROPTOL_EF_GLOBAL], 
                                             &params[PARGEMSLR_IO_ILU_DROPTOL_S_GLOBAL], 
                                             &params[PARGEMSLR_IO_ILU_DROPTOL_C_GLOBAL]);
            params[PARGEMSLR_IO_ILU_DROPTOL_B_LOCAL] = params[PARGEMSLR_IO_ILU_DROPTOL_B_GLOBAL];
            params[PARGEMSLR_IO_ILU_DROPTOL_EF_LOCAL] = params[PARGEMSLR_IO_ILU_DROPTOL_EF_GLOBAL];
            params[PARGEMSLR_IO_ILU_DROPTOL_S_LOCAL] = params[PARGEMSLR_IO_ILU_DROPTOL_S_GLOBAL];
            params[PARGEMSLR_IO_ILU_DROPTOL_C_LOCAL] = params[PARGEMSLR_IO_ILU_DROPTOL_C_GLOBAL];
            break;
         }
         case 13:
         {
            sscanf(line, "%lf %lf %lf ", &params[PARGEMSLR_IO_ILU_ROWNNZ_B_GLOBAL],
                                             &params[PARGEMSLR_IO_ILU_ROWNNZ_S_GLOBAL],
                                             &params[PARGEMSLR_IO_ILU_ROWNNZ_C_GLOBAL]);
            params[PARGEMSLR_IO_ILU_ROWNNZ_B_LOCAL] = params[PARGEMSLR_IO_ILU_ROWNNZ_B_GLOBAL];
            params[PARGEMSLR_IO_ILU_ROWNNZ_S_LOCAL] = params[PARGEMSLR_IO_ILU_ROWNNZ_S_GLOBAL];
            params[PARGEMSLR_IO_ILU_ROWNNZ_C_LOCAL] = params[PARGEMSLR_IO_ILU_ROWNNZ_C_GLOBAL];
            break;
         }
         case 14:
         {
            double temp;
            sscanf(line, "%lf %lf %lf ", &params[PARGEMSLR_IO_ILU_LFIL_B_GLOBAL],
                                          &temp, &params[PARGEMSLR_IO_ILU_LFIL_C_GLOBAL]);
            params[PARGEMSLR_IO_ILU_LFIL_B_LOCAL] = params[PARGEMSLR_IO_ILU_LFIL_B_GLOBAL];
            params[PARGEMSLR_IO_ILU_LFIL_C_LOCAL] = params[PARGEMSLR_IO_ILU_LFIL_C_GLOBAL];
            break;
         }
         case 15:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "RCM"))
            {
               params[PARGEMSLR_IO_ILU_PERM_OPTION_GLOBAL] = kIluReorderingRcm;
               params[PARGEMSLR_IO_ILU_PERM_OPTION_LOCAL] = kIluReorderingRcm;
            }
            else if(!strcmp(word, "AMD"))
            {
               params[PARGEMSLR_IO_ILU_PERM_OPTION_GLOBAL] = kIluReorderingAmd;
               params[PARGEMSLR_IO_ILU_PERM_OPTION_LOCAL] = kIluReorderingAmd;
            }
            else if(!strcmp(word, "ND"))
            {
               params[PARGEMSLR_IO_ILU_PERM_OPTION_GLOBAL] = kIluReorderingNd;
               params[PARGEMSLR_IO_ILU_PERM_OPTION_LOCAL] = kIluReorderingNd;
            }
            else if(!strcmp(word, "NO"))
            {
               params[PARGEMSLR_IO_ILU_PERM_OPTION_GLOBAL] = kIluReorderingNo;
               params[PARGEMSLR_IO_ILU_PERM_OPTION_LOCAL] = kIluReorderingNo;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 16:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "TR"))
            {
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_GLOBAL] = kGemslrLowrankThickRestart;
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_LOCAL] = kGemslrLowrankThickRestart;
            }
            else if(!strcmp(word, "STD"))
            {
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_GLOBAL] = kGemslrLowrankNoRestart;
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_LOCAL] = kGemslrLowrankNoRestart;
            }
            else if(!strcmp(word, "SUB"))
            {
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_GLOBAL] = kGemslrLowrankSubspaceIteration;
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_LOCAL] = kGemslrLowrankSubspaceIteration;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum+2);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 17:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "TR"))
            {
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_GLOBAL] = kGemslrLowrankThickRestart;
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_LOCAL] = kGemslrLowrankThickRestart;
            }
            else if(!strcmp(word, "STD"))
            {
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_GLOBAL] = kGemslrLowrankNoRestart;
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_LOCAL] = kGemslrLowrankNoRestart;
            }
            else if(!strcmp(word, "SUB"))
            {
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_GLOBAL] = kGemslrLowrankSubspaceIteration;
               params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_LOCAL] = kGemslrLowrankSubspaceIteration;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum+2);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 18:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_RANK1_GLOBAL],
                                         &params[PARGEMSLR_IO_LR_RANK2_GLOBAL]);
            params[PARGEMSLR_IO_LR_RANK1_LOCAL] = params[PARGEMSLR_IO_LR_RANK1_GLOBAL];
            params[PARGEMSLR_IO_LR_RANK2_LOCAL] = params[PARGEMSLR_IO_LR_RANK2_GLOBAL];
            break;
         }
         case 19:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_TOL_EIG1_GLOBAL], &params[PARGEMSLR_IO_LR_TOL_EIG2_GLOBAL]);
            params[PARGEMSLR_IO_LR_TOL_EIG1_LOCAL] = params[PARGEMSLR_IO_LR_TOL_EIG1_GLOBAL];
            params[PARGEMSLR_IO_LR_TOL_EIG2_LOCAL] = params[PARGEMSLR_IO_LR_TOL_EIG2_GLOBAL];
            break;
         }
         case 20:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_RANK_FACTOR1_GLOBAL], &params[PARGEMSLR_IO_LR_RANK_FACTOR2_GLOBAL]);
            params[PARGEMSLR_IO_LR_RANK_FACTOR1_LOCAL] = params[PARGEMSLR_IO_LR_RANK_FACTOR1_GLOBAL];
            params[PARGEMSLR_IO_LR_RANK_FACTOR2_LOCAL] = params[PARGEMSLR_IO_LR_RANK_FACTOR2_GLOBAL];
            break;
         }
         case 21:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_GLOBAL], &params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_GLOBAL]);
            params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_LOCAL] = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_GLOBAL];
            params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_LOCAL] = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_GLOBAL];
            break;
         }
         case 22:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_MAXITS1_GLOBAL], &params[PARGEMSLR_IO_LR_MAXITS2_GLOBAL]);
            params[PARGEMSLR_IO_LR_MAXITS1_LOCAL] = params[PARGEMSLR_IO_LR_MAXITS1_GLOBAL];
            params[PARGEMSLR_IO_LR_MAXITS2_LOCAL] = params[PARGEMSLR_IO_LR_MAXITS2_GLOBAL];
            break;
         }
         case 23:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_SCHUR_ENABLE]);
            break;
         }
         case 24:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_SCHUR_ITER_TOL]);
            break;
         }
         case 25:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_SCHUR_MAXITS]);
            break;
         }
         case 26:
         {
            //sscanf(line, "%lf", &params[PARGEMSLR_IO_ADVANCED_DIAG_SHIFT]);
            sscanf(line, "%lf", &params[PARGEMSLR_IO_ADVANCED_USE_COMPLEX_SHIFT]);
            break;
         }
         case 27:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "LU"))
            {
               params[PARGEMSLR_IO_ADVANCED_GLOBAL_SOLVE] = kGemslrLUSolve;
            }
            else if(!strcmp(word, "U"))
            {
               params[PARGEMSLR_IO_ADVANCED_GLOBAL_SOLVE] = kGemslrUSolve;
            }
            else if(!strcmp(word, "MUL"))
            {
               params[PARGEMSLR_IO_ADVANCED_GLOBAL_SOLVE] = kGemslrMulSolve;
            }
            else if(!strcmp(word, "MMUL"))
            {
               params[PARGEMSLR_IO_ADVANCED_GLOBAL_SOLVE] = kGemslrMmulSolve;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 28:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_ADVANCED_RESIDUAL_ITERS]);
            break;
         }
         case 29:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_ADVANCED_GRAM_SCHMIDT]);
            break;
         }
         case 30:
         {
            PargemslrReadFirstWord( line, &word);
            
            if(!strcmp(word, "MILU"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_SMOOTHER1] = kGemslrBSmoothBMILU;
            }
            else if(!strcmp(word, "ILU"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_SMOOTHER1] = kGemslrBSmoothBILU;
            }
            else
            {
               PARGEMSLR_PRINT("Error in inputs format at line %d.\n", linenum);
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            free(word);
            break;
         }
         case 31:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL]);
            break;
         }
         //case 32:
         //{
         //   sscanf(line, "%lf", &params[PARGEMSLR_IO_PREPOSS_VTXSEP_GLOBAL]);
         //   params[PARGEMSLR_IO_PREPOSS_VTXSEP_LOCAL] = params[PARGEMSLR_IO_PREPOSS_VTXSEP_GLOBAL];
         //   break;
         //}
      }
   }
   
   fclose(f);
   
   return PARGEMSLR_SUCCESS;
}

double*
hypre_PargemslrCreateParameterArrayFromFile(char *filename)
{
   double *params = NULL;
   PARGEMSLR_MALLOC( params, PARGEMSLR_IO_SIZE, kMemoryHost, double);
   PargemslrSetDefaultParameterArray(params);
   
   read_inputs_from_file( filename, params);
   
   return params;
}

int
hypre_PargemslrDestroyParameterArray(double *params)
{
   PARGEMSLR_FREE(params, kMemoryHost);
   return 0;
}
