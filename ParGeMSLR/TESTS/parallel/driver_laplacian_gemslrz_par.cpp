/**
 * @file driver_laplacian_gemslrz_par.cpp
 * @brief Example file, use parallel fgmres + gemslr solve complex problem.
 */


#include "pargemslr.hpp"
#include "io_par.hpp"
#include <iostream>

using namespace std;
using namespace pargemslr;

int main (int argc, char *argv[]) 
{
   long int n;
   int      i, nmats, *nx, *ny, *nz, n_local;
   int      sol_opt = 0;//0: use 1 as solution; 1: use rand as solution; 2: use 1 as rhs; 3 : use rand as rhs.
   int      init_opt = 0;//0: use 0 as initial guess; 1: use one as initial guess; 2: use random initial guess;
   complexd *shift, *alphax, *alphay, *alphaz;
   bool     rand_perturb = false;
   long int nnzA, nnzM, nnzLR, nnzILU;
   double   *params = pargemslr_global::_params;
   
   char outfile[1024], infile[1024], solfile[1024];
   bool writesol = false;
   int location;
   
   /*---------parallel CSR matrix */
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
   
   if(read_double_complex_laplacian_param(nmats, &nx, &ny, &nz, &shift, &alphax, &alphay, &alphaz) != 0)
   {
      if(myid == 0)
      {
         printf("Matrix file error\n");
      }
      return -1;
   }
   
   /* - - - - - - - - - - - - - - - -
    * 3. Read settings
    * - - - - - - - - - - - - - - - */
   
   /* read from file "input" when necessary */
   if(PargemslrReadInputArg("fromfile", infile, argc, argv))
   {
      if(myid == 0)
      {
         printf("Reading setup from file %s\n",infile);
      }
      read_inputs_from_file( infile, params);
   }
   else
   {
      if(myid == 0)
      {
         printf("Reading setup from file \"inputs\"\n");
      }
      read_inputs_from_file( "inputs", params);
   }
   
   if(PargemslrReadInputArg("outfile", outfile, argc, argv))
   {
      if(PargemslrSetOutputFile(outfile) != 0)
      {
         printf("Reading output file error, write to stdout\n");
      }
      else
      {
         if(myid == 0)
         {
            printf("Write to file %s\n",outfile);
         }
      }
   }
   
   if(PargemslrReadInputArg("writesol", solfile, argc, argv))
   {
      if(myid == 0)
      {
         printf("Writing solution to file %s\n",solfile);
      }
      writesol = true;
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
      
      complexd  one = 1.0, zero = 0.0;
      
      /*---------------------------------
       * 4.1. Create test matrix
       *---------------------------------*/
      
      /* create Laplacian */
      if(np == params[PARGEMSLR_IO_PARALLEL_NPROCX]*params[PARGEMSLR_IO_PARALLEL_NPROCY]*params[PARGEMSLR_IO_PARALLEL_NPROCZ])
      {
         PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (parcsr_mat->Laplacian( nx[i], ny[i], nz[i], params[PARGEMSLR_IO_PARALLEL_NPROCX], params[PARGEMSLR_IO_PARALLEL_NPROCY], params[PARGEMSLR_IO_PARALLEL_NPROCZ], alphax[i], alphay[i], alphaz[i], shift[i], parlog)));
      }
      else
      {
         PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (parcsr_mat->Laplacian( nx[i], ny[i], nz[i], 1, 1, np, alphax[i], alphay[i], alphaz[i], shift[i], parlog)));
      }
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Running test number %d\n",i+1);
         if(rand_perturb)
         {
            PARGEMSLR_PRINT("\tSolving Laplacian %d %d %d %lf+%lfi %lf+%lfi %lf+%lfi %lf+%lfi with random diagonal pertubation\n",nx[i],ny[i],nz[i],alphax[i].Real(),alphax[i].Imag(),alphay[i].Real(),alphay[i].Imag(),alphaz[i].Real(),alphaz[i].Imag(),shift[i].Real(),shift[i].Imag());
         }
         else
         {
            PARGEMSLR_PRINT("\tSolving Laplacian %d %d %d %lf+%lfi %lf+%lfi %lf+%lfi %lf+%lfi\n",nx[i],ny[i],nz[i],alphax[i].Real(),alphax[i].Imag(),alphay[i].Real(),alphay[i].Imag(),alphaz[i].Real(),alphaz[i].Imag(),shift[i].Real(),shift[i].Imag());
         }
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
               printf("\tUsing random solution\n");
            }
            break;
         }
         case 2:
         {
            /* use 1 rhs */
            b->Fill(one);
            if(myid == 0)
            {
               printf("\tUsing one vector as right-hand-side\n");
            }
            break;
         }
         case 3:
         {
            /* use random rhs */
            b->Rand();
            if(myid == 0)
            {
               printf("\tUsing random right-hand-side\n");
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
               printf("\tUsing one vector as solution\n");
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
               printf("\tUsing one vector as initial guess\n");
            }
            break;
         }
         case 2:
         {
            /* random initial guess */
            x->Rand();
            if(myid == 0)
            {
               printf("\tUsing random initial guess\n");
            }
            break;
         }
         case 0: default:
         {
            /* zero init guess */
            x->Fill(zero);
            if(myid == 0)
            {
               printf("\tUsing zero vector as initial guess\n");
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
