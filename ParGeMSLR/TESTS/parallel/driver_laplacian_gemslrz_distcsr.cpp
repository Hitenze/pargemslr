/**
 * @file driver_laplacian_gemslrz_par.cpp
 * @brief Example file, use parallel fgmres + gemslr solve complex problem.
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
complexd* Rhs2DistRhs(ParallelCsrMatrixClass<complexd> *parcsr_mat, int nrhs, complexd *A_mat);

// this function solve multiple right-hand sides stored in matrix
// input:
//   solverp: solver
//   parcsr_mat: ParCSR matrix
//   A_rhs: distributed rhs
// input and output:
//   A_init: distributed initial guess and solution
int SolveMultipleRhs(FlexGmresClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd> *solverp, 
                     ParallelCsrMatrixClass<complexd> *parcsr_mat, int nrhs, complexd *A_rhs, complexd *A_init);

int main (int argc, char *argv[]) 
{
   long int n;
   int      i, nmats, *nx, *ny, *nz, n_local;
   complexd *shift, *alphax, *alphay, *alphaz;
   long int nnzA, nnzM, nnzLR, nnzILU;
   double   *params = pargemslr_global::_params;
   
   char outfile[1024], infile[1024], lapfile[1024];
   int location;
   
   /*---------parallel CSR matrix */
   CsrMatrixClass<complexd> *csr_mat = new CsrMatrixClass<complexd>();
   ParallelCsrMatrixClass<complexd> *parcsr_mat = new ParallelCsrMatrixClass<complexd>();
   
   /*---------parallel vectors */
   ParallelVectorClass<complexd>   *x = new ParallelVectorClass<complexd>();
   ParallelVectorClass<complexd>   *b = new ParallelVectorClass<complexd>();
   
   /*---------create solver */
   FlexGmresClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd> *solverp = 
      new FlexGmresClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>();
   
   /*---------create precond */
   ParallelGemslrClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd> *precondp = 
      new ParallelGemslrClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>();
   
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
      if(read_double_complex_laplacian_param(nmats, &nx, &ny, &nz, &shift, &alphax, &alphay, &alphaz, lapfile, true) != 0)
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
      if(read_double_complex_laplacian_param(nmats, &nx, &ny, &nz, &shift, &alphax, &alphay, &alphaz, lapfile, false) != 0)
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
      
      //complexd  one = 1.0, zero = 0.0;
      
      /*---------------------------------
       * 4.1. Create test matrix
       *---------------------------------*/
      
      /* create Laplacian */
      if(0)
      {
         // directly create ParCSR
         if(np == params[PARGEMSLR_IO_PARALLEL_NPROCX]*params[PARGEMSLR_IO_PARALLEL_NPROCY]*params[PARGEMSLR_IO_PARALLEL_NPROCZ])
         {
            PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (parcsr_mat->Laplacian( nx[i], ny[i], nz[i], params[PARGEMSLR_IO_PARALLEL_NPROCX], params[PARGEMSLR_IO_PARALLEL_NPROCY], params[PARGEMSLR_IO_PARALLEL_NPROCZ], alphax[i], alphay[i], alphaz[i], shift[i], parlog)));
         }
         else
         {
            PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (parcsr_mat->Laplacian( nx[i], ny[i], nz[i], 1, 1, np, alphax[i], alphay[i], alphaz[i], shift[i], parlog)));
         }
      }
      if(1)
      {
         // create ParCSR from distributed CSR
         
         // first create distributed CSR, since we don't have this function,
         // we first csrate ParCSR and convert it into distributed CSR
         if(np == params[PARGEMSLR_IO_PARALLEL_NPROCX]*params[PARGEMSLR_IO_PARALLEL_NPROCY]*params[PARGEMSLR_IO_PARALLEL_NPROCZ])
         {
            PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (parcsr_mat->Laplacian( nx[i], ny[i], nz[i], params[PARGEMSLR_IO_PARALLEL_NPROCX], params[PARGEMSLR_IO_PARALLEL_NPROCY], params[PARGEMSLR_IO_PARALLEL_NPROCZ], alphax[i], alphay[i], alphaz[i], shift[i], parlog)));
         }
         else
         {
            PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (parcsr_mat->Laplacian( nx[i], ny[i], nz[i], 1, 1, np, alphax[i], alphay[i], alphaz[i], shift[i], parlog)));
         }
         
         // now we have ParCSR, convert into distributed CSR
         vector_long                      vtxdist, xadj, adjncy;
         vector_seq_complexd              value;
         parcsr_mat->GetDistCSRArrays(vtxdist, xadj, adjncy, value);
         
         long int * row_starts = vtxdist.GetData(); // first row on each MPI ranks, the last entry is n, length np + 1
         long int * dist_i = xadj.GetData(); // I for distributed CSR
         long int * dist_j = adjncy.GetData(); // J for distributed CSR
         complexd * dist_data = value.GetData(); // values for distributed CSR
         
         // Note: row_starts is the start of the first row on each processor, length np + 1. The last entry is the n+idxin
         //       dist_i is the I in distributed CSR
         //       dist_j is the J in distributed CSR
         //       dist_data is the A in distributed CSR
         
         parcsr_mat->Clear();
         
         // now convert into ParCSR format
         int idxin = 0; // the first row is row 0, set to 1 if the first row is 1
         PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, 
               (parcsr_mat->ReadFromDistributedCSR( row_starts, idxin, dist_i, dist_j, dist_data, parlog)));
         
         vtxdist.Clear();
         xadj.Clear();
         adjncy.Clear();
         value.Clear();
         
      }
      if(0)
      {
         // Create ParCSR matrix from CSR
         
         // First create local CSR
         PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, 
               (csr_mat->Laplacian( nx[i], ny[i], nz[i], alphax[i], alphay[i], alphaz[i], shift[i], false)));
         
         if(myid == 0)
         {
            PARGEMSLR_PRINT("Converting CSR to ParCSR\n");
         }
         
         // Those are the 3 vectors, A_j is 0-based. If 1-based set idxin = 1.
         int idxin = 0;
         int *A_i = csr_mat->GetI();
         int *A_j = csr_mat->GetJ();
         complexd *A_data = csr_mat->GetData();
         
         // Create ParCSR matrix
         PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, 
               (parcsr_mat->ReadFromSingleCSR( nx[i]*ny[i]*nz[i], idxin, A_i, A_j, A_data, parlog)));
      }
      
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Running test number %d\n",i+1);
         PARGEMSLR_PRINT("\tSolving Laplacian %d %d %d %lf+%lfi %lf+%lfi %lf+%lfi %lf+%lfi\n",nx[i],ny[i],nz[i],alphax[i].Real(),alphax[i].Imag(),alphay[i].Real(),alphay[i].Imag(),alphaz[i].Real(),alphaz[i].Imag(),shift[i].Real(),shift[i].Imag());
      }
      
      parcsr_mat->SetComplexShift(params[PARGEMSLR_IO_ADVANCED_DIAG_SHIFT]);
      
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
      complexd *A_rhs_g = NULL;
      complexd *A_init_g = NULL;
      complexd *A_rhs = NULL;
      complexd *A_init = NULL;
      
      // create global rhs and initial guess
      if(myid == 0)
      {
         PARGEMSLR_MALLOC( A_rhs_g, n*nrhs, kMemoryHost, complexd);
         
         // rhs with 1s and 2s
         for(int i = 0 ; i < n ; i ++)
         {
            A_rhs_g[i].Real() = 1.0;
            A_rhs_g[i].Imag() = 1.0;
         }
         for(int i = n ; i < 2*n ; i ++)
         {
            A_rhs_g[i].Real() = 2.0;
            A_rhs_g[i].Imag() = 2.0;
         }
         
         // zero init guess
         PARGEMSLR_CALLOC( A_init_g, n*nrhs, kMemoryHost, complexd);
      }
      
      // convert to distributed
      A_rhs = Rhs2DistRhs( parcsr_mat, nrhs, A_rhs_g);
      A_init = Rhs2DistRhs( parcsr_mat, nrhs, A_init_g);
      
      PARGEMSLR_FREE(A_rhs_g, kMemoryHost);
      PARGEMSLR_FREE(A_init_g, kMemoryHost);
      
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
      
      PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_SETUP_TIME, (solverp->Setup(*x, *b)));
      
      /* - - - - - - - - - - - - - - - -
       * 7. Solve phase
       * - - - - - - - - - - - - - - - */
      
      //PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_SOLVE_TIME, (solverp->Solve(*x, *b)));
      
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Solve Multiple Right-hand sides\n");
      }
      
      // mute output from GMRES
      solverp->SetPrintOption(0);
      // Solve
      SolveMultipleRhs(solverp, parcsr_mat, nrhs, A_rhs, A_init);
      
      // See the solution
      if(myid == 0)
      {
         cout<<A_init[0]<<" "<<A_init[1]<<endl;
         cout<<A_init[n_local]<<" "<<A_init[n_local+1]<<endl;
      }
      
      PARGEMSLR_FREE(A_rhs, kMemoryHost);
      PARGEMSLR_FREE(A_init, kMemoryHost);
      
      /* - - - - - - - - - - - - - - - -
       * 8. Get Fill Level
       * - - - - - - - - - - - - - - - */
      
      nnzA = parcsr_mat->GetNumNonzeros();
      nnzM = precondp->GetNumNonzeros(nnzILU, nnzLR);
      
      /* - - - - - - - - - - - - - - - -
       * 9. Print
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
   delete csr_mat;
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


complexd* Rhs2DistRhs(ParallelCsrMatrixClass<complexd> *parcsr_mat, int nrhs, complexd *A_mat)
{
   complexd *A_mat_global = NULL;
   complexd *dist_A_mat = NULL;
   
   MPI_Comm comm;
   int np, myid;
   parcsr_mat->GetMpiInfo(np, myid, comm);
   
   
   long int n = parcsr_mat->GetNumRowsGlobal();
   long int n_start = parcsr_mat->GetRowStartGlobal();
   int n_local = parcsr_mat->GetNumRowsLocal();
   
   if(myid != 0)
   {
      PARGEMSLR_MALLOC( A_mat_global, n*nrhs, kMemoryHost, complexd);
   }
   else
   {
      A_mat_global = A_mat;
   }
   PARGEMSLR_MALLOC( dist_A_mat, n_local*nrhs, kMemoryHost, complexd);
   
   PargemslrMpiBcast( A_mat_global, n*nrhs, 0, comm);
   
   complexd *ptr = dist_A_mat;
   for(int i = 0 ; i < nrhs ; i ++)
   {
      complexd *ptr_2 = A_mat_global + n_start + i * n;
      PARGEMSLR_MEMCPY( ptr, ptr_2, n_local, kMemoryHost, kMemoryHost, complexd);
      ptr += n_local;
   }
   
   if(myid != 0)
   {
      PARGEMSLR_FREE( A_mat_global, kMemoryHost);
   }
   
   return dist_A_mat;
}

int SolveMultipleRhs(FlexGmresClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd> *solverp, 
                     ParallelCsrMatrixClass<complexd> *parcsr_mat, int nrhs, complexd *A_rhs, complexd *A_init)
{
   vector_par_complexd v, w;
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
