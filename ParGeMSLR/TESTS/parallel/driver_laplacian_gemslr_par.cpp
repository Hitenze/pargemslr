/**
 * @file driver_laplacian_gemslr_par.cpp
 * @brief Example file, use parallel fgmres + gemslr.
 */


#include "pargemslr.hpp"
#include "io_par.hpp"
#include <iostream>

using namespace std;
using namespace pargemslr;

int main (int argc, char *argv[]) 
{
   long int n;
   int      i, nmats, *nx, *ny, *nz, dx, dy, dz, d2x, d2y, d2z, n_local;
   int      sol_opt = 0;//0: use 1 as solution; 1: use rand as solution; 2: use 1 as rhs; 3 : use rand as rhs.
   int      init_opt = 0;//0: use 0 as initial guess; 1: use one as initial guess; 2: use random initial guess;
   double   *shift, *alphax, *alphay, *alphaz;
   bool     rand_perturb = false;
   long int nnzA, nnzM, nnzLR, nnzILU;
   double   *params = pargemslr_global::_params;
   
   char outfile[1024], infile[1024], lapfile[1024], solfile[1024];
   bool writesol = false;
   int location;
   
   /*---------parallel CSR matrix */
   ParallelCsrMatrixClass<double> *parcsr_mat = new ParallelCsrMatrixClass<double>();
   
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
   
   if(PargemslrReadInputArg("writesol", solfile, argc, argv))
   {
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Writing solution to file %s\n",solfile);
      }
      writesol = true;
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
   
   if(PargemslrReadInputArg("solone", argc, argv))
   {
      sol_opt = 0;
   }
   
   if(PargemslrReadInputArg("solrand", argc, argv))
   {
      sol_opt = 1;
   }
   
   if(PargemslrReadInputArg("rhsone", argc, argv))
   {
      sol_opt = 2;
   }
   
   if(PargemslrReadInputArg("rhsrand", argc, argv))
   {
      sol_opt = 3;
   }
   
   if(PargemslrReadInputArg("initzero", argc, argv))
   {
      init_opt = 0;
   }
   
   if(PargemslrReadInputArg("initone", argc, argv))
   {
      init_opt = 1;
   }
   
   if(PargemslrReadInputArg("initrand", argc, argv))
   {
      init_opt = 2;
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
      
      double  one = 1.0, zero = 0.0;
      
      /*---------------------------------
       * 4.1. Create test matrix
       *---------------------------------*/
      
      /* create Laplacian */
      dx = params[PARGEMSLR_IO_PARALLEL_NPROCX];
      dy = params[PARGEMSLR_IO_PARALLEL_NPROCY];
      dz = params[PARGEMSLR_IO_PARALLEL_NPROCZ];
      
      d2x = params[PARGEMSLR_IO_PARALLEL_NDOMX];
      d2y = params[PARGEMSLR_IO_PARALLEL_NDOMY];
      d2z = params[PARGEMSLR_IO_PARALLEL_NDOMZ];
      
      if(np == dx*dy*dz)
      {
         //PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (parcsr_mat->Laplacian( nx[i], ny[i], nz[i], dx, dy, dz, alphax[i], alphay[i], alphaz[i], shift[i], parlog)));
         PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, 
            (parcsr_mat->LaplacianWithPartition( nx[i], ny[i], nz[i], dx, dy, dz, d2x, d2y, d2z, alphax[i], alphay[i], alphaz[i], shift[i], parlog, rand_perturb)));
      }
      else
      {
         //PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (parcsr_mat->Laplacian( nx[i], ny[i], nz[i], 1, np, 1, alphax[i], alphay[i], alphaz[i], shift[i], parlog)));
         PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, 
            (parcsr_mat->LaplacianWithPartition( nx[i], ny[i], nz[i], 1, np, 1, d2x, d2y, d2z, alphax[i], alphay[i], alphaz[i], shift[i], parlog, rand_perturb)));
      }
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
      
      switch(sol_opt)
      {
         case 1:
         {
            /* use random solution */
            x->Rand();
            b->Fill(zero);
            parcsr_mat->MatVec( 'N', one, *x, zero, *b);
            if(myid == 0)
            {
               PARGEMSLR_PRINT("\tUsing random solution\n");
            }
            break;
         }
         case 2:
         {
            /* use 1 rhs */
            b->Fill(one);
            if(myid == 0)
            {
               PARGEMSLR_PRINT("\tUsing one vector as right-hand-side\n");
            }
            break;
         }
         case 3:
         {
            /* use random rhs */
            b->Rand();
            if(myid == 0)
            {
               PARGEMSLR_PRINT("\tUsing random right-hand-side\n");
            }
            break;
         }
         case 0: default:
         {
            /* use 1 as solution */
            x->Fill(one);
            b->Fill(zero);
            parcsr_mat->MatVec( 'N', one, *x, zero, *b);
            if(myid == 0)
            {
               PARGEMSLR_PRINT("\tUsing one vector as solution\n");
            }
            break;
         }
      }
      
      switch(init_opt)
      {
         case 1:
         {
            /* one init guess */
            x->Fill(one);
            if(myid == 0)
            {
               PARGEMSLR_PRINT("\tUsing one vector as initial guess\n");
            }
            break;
         }
         case 2:
         {
            /* random initial guess */
            x->Rand();
            if(myid == 0)
            {
               PARGEMSLR_PRINT("\tUsing random initial guess\n");
            }
            break;
         }
         case 0: default:
         {
            /* zero init guess */
            x->Fill(zero);
            if(myid == 0)
            {
               PARGEMSLR_PRINT("\tUsing zero vector as initial guess\n");
            }
            break;
         }
      }
      
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
       * 6. Solve phase
       * - - - - - - - - - - - - - - - */
      
      PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_SOLVE_TIME, (solverp->Solve(*x, *b)));
      
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
      
      if(writesol)
      {
         char tempsolname[2048];
         snprintf( tempsolname, 2048, "./%s%05d%s", solfile, i, ".sol" );
         x->WriteToDisk(tempsolname);
      }
      
      x->Clear();
      b->Clear();
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
