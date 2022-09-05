/**
 * @file driver_laplacian_gemslr_par.cpp
 * @brief Example file, use parallel fgmres + gemslr.
 */


#include "pargemslr.hpp"
#include "io_par.hpp"
#include <iostream>

using namespace std;
using namespace pargemslr;

// this function converts right-hand side matrix into
// distributed right-hand side matrix
// input:
//   A_mat: stored in the Fortran style. Each column is continue in memory.
// return:
//   distributed matrix.
double* Rhs2DistRhs(ParallelCsrMatrixClass<double> *parcsr_mat, int nrhs, double *A_mat);

// this function solve multiple right-hand sides stored in matrix
// input:
//   solverp: solver
//   parcsr_mat: ParCSR matrix
//   A_rhs: distributed rhs
// input and output:
//   A_init: distributed initial guess and solution
int SolveMultipleRhs(FlexGmresClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double> *solverp, 
                     ParallelCsrMatrixClass<double> *parcsr_mat, int nrhs, double *A_rhs, double *A_init);

int main (int argc, char *argv[]) 
{
   long int n;
   int      i, nmats, *nx, *ny, *nz, n_local;
   double   *shift, *alphax, *alphay, *alphaz;
   bool     rand_perturb = false;
   long int nnzA, nnzM, nnzLR, nnzILU;
   double   *params = pargemslr_global::_params;
   
   //char outfile[1024], infile[1024], lapfile[1024], solfile[1024];
   char outfile[1024], infile[1024], lapfile[1024];
   int location;
   
   /*---------parallel CSR matrix */
   ParallelCsrMatrixClass<double> *parcsr_mat = new ParallelCsrMatrixClass<double>();
   CsrMatrixClass<double> *csr_mat = new CsrMatrixClass<double>();
   
   /*---------parallel vectors */
   ParallelVectorClass<double>   *x = new ParallelVectorClass<double>();
   ParallelVectorClass<double>   *b = new ParallelVectorClass<double>();
   
   /*---------create solver */
   FlexGmresClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double> *solverp = 
      new FlexGmresClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>();
   
   /*---------create precond */
   ParallelGemslrClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double> *precondp = 
      new ParallelGemslrClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>();
   
   /* - - - - - - - - - - - - - - - -
    * 1. Initialize ParGeMSLR package
    * - - - - - - - - - - - - - - - */
   
   PargemslrInit( &argc, &argv);
   
   /* Dummy communication, to fix MSI performance bugs */
   dummy_comm();
   
   /* print parallel information */
   PargemslrPrintParallelInfo();
   
   /* the gemslr parallel info data structure */
   parallel_log parlog;
   
   /* get MPI info */
   int      np, myid;
   MPI_Comm comm;
   parlog.GetMpiInfo(np, myid, comm);
   
   /* print help when necessary */
   if(PargemslrReadInputArg("help", argc, argv))
   {
      if(myid == 0)
      {
         print_parlap_test_usage();
      }
      PargemslrFinalize();
      return 0;
   }
   
   /* - - - - - - - - - - - - - - - -
    * 2. Read Laplacian Parameters
    * - - - - - - - - - - - - - - - */
   
   /* read from file "lapfile" when necessary */
   if(PargemslrReadInputArg("lapfile", lapfile, argc, argv))
   {
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Reading laplacian from file %s\n",lapfile);
      }
      if(read_double_laplacian_param(nmats, &nx, &ny, &nz, &shift, &alphax, &alphay, &alphaz, lapfile, true) != 0)
      {
         if(myid == 0)
         {
            PARGEMSLR_PRINT("Matrix file error\n");
         }
         PargemslrFinalize();
         return -1;
      }
      
   }
   else
   {
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Reading laplacian from file \"lapfile_real\"\n");
      }
      if(read_double_laplacian_param(nmats, &nx, &ny, &nz, &shift, &alphax, &alphay, &alphaz, lapfile, false) != 0)
      {
         if(myid == 0)
         {
            PARGEMSLR_PRINT("Matrix file error\n");
         }
         PargemslrFinalize();
         return -1;
      }
   }
   
   /* - - - - - - - - - - - - - - - -
    * 3. Read settings
    * - - - - - - - - - - - - - - - */
   
   /* read from file "input" when necessary */
   if(PargemslrReadInputArg("fromfile", infile, argc, argv))
   {
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Reading setup from file %s\n",infile);
      }
      read_inputs_from_file( infile, params);
   }
   else
   {
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Reading setup from file \"inputs\"\n");
      }
      read_inputs_from_file( "inputs", params);
   }
   
   if(PargemslrReadInputArg("outfile", outfile, argc, argv))
   {
      if(PargemslrSetOutputFile(outfile) != 0)
      {
         PARGEMSLR_PRINT("Reading output file error, write to stdout\n");
      }
      else
      {
         if(myid == 0)
         {
            PARGEMSLR_PRINT("Write to file %s\n",outfile);
         }
      }
   }
   
   /* read settings from terminal input 
    * note that those settings will overwrite the settings from file
    */
   read_inputs(params, argc, argv);
   
   /* set solver location */
   if ( (PargemslrReadInputArg("gpu", argc, argv)) )
	{
		location = kMemoryDevice;
	}
   else    
   { 
      location = kMemoryHost;
   }
   
   /* should we apply random pertubation to the diagonal? */
   if ( (PargemslrReadInputArg("rand_perturb", argc, argv)) )
	{
		rand_perturb = true;
	}
   
   /* main loop */
   if(myid == 0)
   {
      printf("Start running tests. Total %d tests.\n", nmats);
   }
   for(i = 0 ; i < nmats ; i ++)
   {
   
      /* - - - - - - - - - - - - - - - -
       * 4. Create problem
       * - - - - - - - - - - - - - - - */
      
      /* reset the seed */
      pargemslr_global::_mersenne_twister_engine.seed(0);
      
      //double  one = 1.0, zero = 0.0;
      
      /*---------------------------------
       * 4.1. Create test matrix
       *---------------------------------*/
      
      /* Create CSR Laplacian matrix */
      PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, 
            (csr_mat->Laplacian( nx[i], ny[i], nz[i], alphax[i], alphay[i], alphaz[i], shift[i], rand_perturb)));
      
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Converting CSR to ParCAR\n");
      }
      
      /* Those are the 3 vectors, A_j is 0-based. If 1-based set idxin = 1 */
      int idxin = 0;
      int *A_i = csr_mat->GetI();
      int *A_j = csr_mat->GetJ();
      double *A_data = csr_mat->GetData();
      
      /* Create ParCSR matrix */
      PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, 
            (parcsr_mat->ReadFromSingleCSR( nx[i]*ny[i]*nz[i], idxin, A_i, A_j, A_data, parlog)));
      
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Running test number %d\n",i+1);
         if(rand_perturb)
         {
            PARGEMSLR_PRINT("\tSolving Laplacian %d %d %d %lf %lf %lf %lf with random pertubation on the diagonal\n",nx[i],ny[i],nz[i],alphax[i],alphay[i],alphaz[i],shift[i]);
         }
         else
         {
            PARGEMSLR_PRINT("\tSolving Laplacian %d %d %d %lf %lf %lf %lf\n",nx[i],ny[i],nz[i],alphax[i],alphay[i],alphaz[i],shift[i]);
         }
      }
      
      /* get local and global length */
      n_local = parcsr_mat->GetNumRowsLocal();
      n = parcsr_mat->GetNumRowsGlobal();
      if(myid == 0)
      {
         PARGEMSLR_PRINT("\tProblem size on mpi rank 0 (local/global): %d/%ld\n",n_local, n);
      }
      
      /* move to device when device memory is availiable */
      parcsr_mat->SetupMatvec();
      parcsr_mat->MoveData(location);
      
      /*---------------------------------
       * 4.2. Create x and right-hand-side
       *---------------------------------*/
      
      /* create vector, on device when device memory is availiable */
      x->Setup(n_local, location, false, parlog);
      b->Setup(n_local, location, false, parlog);
      
      /* create solution, rhs, and initial guess */
      
      int nrhs = 2;
      double *A_rhs_g = NULL;
      double *A_init_g = NULL;
      double *A_rhs = NULL;
      double *A_init = NULL;
      
      // create global rhs and initial guess
      if(myid == 0)
      {
         PARGEMSLR_MALLOC( A_rhs_g, n*nrhs, kMemoryHost, double);
         
         // rhs with 1s and 2s
         for(int i = 0 ; i < n ; i ++)
         {
            A_rhs_g[i] = 1.0;
         }
         for(int i = n ; i < 2*n ; i ++)
         {
            A_rhs_g[i] = 2.0;
         }
         
         // zero init guess
         PARGEMSLR_CALLOC( A_init_g, n*nrhs, kMemoryHost, double);
      }
      
      // convert to distributed
      A_rhs = Rhs2DistRhs( parcsr_mat, nrhs, A_rhs_g);
      A_init = Rhs2DistRhs( parcsr_mat, nrhs, A_init_g);
      
      /* - - - - - - - - - - - - - - - -
       * 5. Create solver and precond
       * - - - - - - - - - - - - - - - */
      
      /* set matrix */
      solverp->SetMatrixP(parcsr_mat);
      /* set parameter with parameter array, can also use individual set functions */
      solverp->SetWithParameterArray(params);
      /* move to device when necessary */
      solverp->SetSolveLocation(location);
      
      /* set matrix */
      precondp->SetMatrixP(parcsr_mat);
      /* set parameter with parameter array, can also use individual set functions */
      precondp->SetWithParameterArray(params);
      /* move to device when necessary */
      precondp->SetSolveLocation(location);
      
      /*---------assign preconditioner to solver */
      solverp->SetPreconditionerP(precondp);
      
      /* - - - - - - - - - - - - - - - -
       * 6. Setup phase
       * - - - - - - - - - - - - - - - */
      
      // NOTE: x and b can be dummy
      PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_SETUP_TIME, (solverp->Setup(*x, *b)));
      
      /* - - - - - - - - - - - - - - - -
       * 6. Solve phase
       * - - - - - - - - - - - - - - - */
      
      //PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_SOLVE_TIME, (solverp->Solve(*x, *b)));
      
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Solve Multipli Right-hand sides\n");
      }
      
      SolveMultipleRhs(solverp, parcsr_mat, nrhs, A_rhs, A_init);
      
      PARGEMSLR_FREE(A_rhs, kMemoryHost);
      PARGEMSLR_FREE(A_init, kMemoryHost);
      
      /* - - - - - - - - - - - - - - - -
       * 7. Get Fill Level
       * - - - - - - - - - - - - - - - */
      
      nnzA = parcsr_mat->GetNumNonzeros();
      nnzM = precondp->GetNumNonzeros(nnzILU, nnzLR);
      
      /* - - - - - - - - - - - - - - - -
       * 8. Print
       * - - - - - - - - - - - - - - - */
      
      if(myid == 0)
      {
         PARGEMSLR_PRINT("\n");
         PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
         PARGEMSLR_PRINT("Solution info:\n");
         PARGEMSLR_PRINT("\tNumber of iterations: %d\n",solverp->GetNumberIterations());
         PARGEMSLR_PRINT("\tFinal rel res: %f\n",solverp->GetFinalRelativeResidual());
         PARGEMSLR_PRINT("\tPreconditioner fill level: ILU: %f; Low-rank: %f; Total: %f\n",(double)nnzILU/nnzA,(double)nnzLR/nnzA,(double)nnzM/nnzA);
         /*
         if(params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL] > 0)
         {
            x->MoveData(kMemoryHost);
            PARGEMSLR_PRINT("\tSample solution:\n");
            PargemslrPlotData(x->GetData(), n_local, 30, 5);
         }
         */
         PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
         PARGEMSLR_PRINT("\n");
      }
      
      PARGEMSLR_PRINT_TIMING_RESULT(params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL], (myid == 0));
      
      /* reset time */
      PARGEMSLR_RESET_TIME;
      
      x->Clear();
      b->Clear();
      csr_mat->Clear();
      parcsr_mat->Clear();
      solverp->Clear();
      precondp->Clear();
      
   }
   
   if(myid == 0)
   {
      printf("All tests done\n");
   }
   
   /* - - - - - - - - - - - - - - - -
    * 9. Deallocate before finalize
    * - - - - - - - - - - - - - - - */
   
   delete x;
   delete b;
   delete parcsr_mat;
   delete solverp;
   delete precondp;
   
   free(nx);
   free(ny);
   free(nz);
   free(shift);
   free(alphax);
   free(alphay);
   free(alphaz);
   
   /* - - - - - - - - - - - - - - - -
    * 10. Finalize
    * - - - - - - - - - - - - - - - */
   
   PargemslrFinalize();
   
   return 0;
}

double* Rhs2DistRhs(ParallelCsrMatrixClass<double> *parcsr_mat, int nrhs, double *A_mat)
{
   double *A_mat_global = NULL;
   double *dist_A_mat = NULL;
   
   MPI_Comm comm;
   int np, myid;
   parcsr_mat->GetMpiInfo(np, myid, comm);
   
   
   long int n = parcsr_mat->GetNumRowsGlobal();
   long int n_start = parcsr_mat->GetRowStartGlobal();
   int n_local = parcsr_mat->GetNumRowsLocal();
   
   if(myid != 0)
   {
      PARGEMSLR_MALLOC( A_mat_global, n*nrhs, kMemoryHost, double);
   }
   else
   {
      A_mat_global = A_mat;
   }
   PARGEMSLR_MALLOC( dist_A_mat, n_local*nrhs, kMemoryHost, double);
   
   PargemslrMpiBcast( A_mat_global, n*nrhs, 0, comm);
   
   double *ptr = dist_A_mat;
   for(int i = 0 ; i < nrhs ; i ++)
   {
      double *ptr_2 = A_mat_global + n_start + i * n;
      for(int j = 0; j < n_local ; j ++)
      {
         *(ptr++) = *(ptr_2++);
      }
   }
   
   if(myid != 0)
   {
      PARGEMSLR_FREE( A_mat_global, kMemoryHost);
   }
   
   return dist_A_mat;
}

int SolveMultipleRhs(FlexGmresClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double> *solverp, 
                     ParallelCsrMatrixClass<double> *parcsr_mat, int nrhs, double *A_rhs, double *A_init)
{
   vector_par_double v, w;
   parcsr_mat->SetupVectorPtrStr(v);
   parcsr_mat->SetupVectorPtrStr(w);
   
   int n_local = parcsr_mat->GetNumRowsLocal();
   
   for(int i = 0 ; i < nrhs ; i ++)
   {
      if(n_local > 0)
      {
         v.UpdatePtr( A_init+i*n_local, kMemoryHost);
         w.UpdatePtr( A_rhs+i*n_local, kMemoryHost);
      }
      solverp->Solve(v, w);
   }
   
   return 0;
}
