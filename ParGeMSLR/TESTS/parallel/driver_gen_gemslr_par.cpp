/**
 * @file driver_gen_gemslr_par.cpp
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
   int      i, nmats, n_local;
   long int nnzA, nnzM, nnzLR, nnzILU;
   int      sol_opt = 0;//0: use 1 as solution; 1: use rand as solution; 2: use 1 as rhs; 3 : use rand as rhs.
   int      init_opt = 0;//0: use 0 as initial guess; 1: use one as initial guess; 2: use random initial guess;
   double   *params = pargemslr_global::_params;
   
   int location;
   char **matfile, **vecfile, inmatfile[1024], outfile[1024], infile[1024], solfile[1024];
   bool writesol = false;
   int matfilebase;
   
   /*---------declare matrix */
   ParallelCsrMatrixClass<double> *parcsr_mat = new ParallelCsrMatrixClass<double>();
   
   /*---------declare x and rhs (b) */
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
         print_gen_test_usage();
      }
      PargemslrFinalize();
      return 0;
   }
   
   /* - - - - - - - - - - - - - - - -
    * 2. Read file settings
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
   
   if(PargemslrReadInputArg("writesol", solfile, argc, argv))
   {
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Writing solution to file %s\n",solfile);
      }
      writesol = true;
   }
   /* - - - - - - - - - - - - - - - -
    * 3. Read terminal settings
    * - - - - - - - - - - - - - - - */
   
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
   
   /* get index base */
   if ( (PargemslrReadInputArg("zerobase", argc, argv)) )
   {
      matfilebase = 0;
   }
   else    
   { 
      matfilebase = 1;
   }
   
   if(PargemslrReadInputArg("matfile", inmatfile, argc, argv))
   {
      if(read_matfile( inmatfile, nmats, &matfile, &vecfile) != 0)
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
      if(read_matfile( "matfile_real", nmats, &matfile, &vecfile) != 0)
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
    * 4. Create problem
    * - - - - - - - - - - - - - - - */
   
   /* main loop */
   if(myid == 0)
   {
      printf("Start running tests. Total %d tests.\n", nmats);
   }
   for(i = 0 ; i < nmats ; i ++)
   {
   
      /*---------------------------------
       * 4.1. Create test matrix
       *---------------------------------*/
      
      PARGEMSLR_GLOBAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (parcsr_mat->ReadFromSingleMMFile( matfile[i], matfilebase, parlog)));
      if(myid == 0)
      {
         PARGEMSLR_PRINT("\tSolving general matrix %s\n",matfile[i]);
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
      
      double  zero = 0.0;
   
      /* reset the seed */
      pargemslr_global::_mersenne_twister_engine.seed(0);
      
      if ( !(vecfile[i]) )
      {
         double  one = 1.0;
            
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
         
      }
      else    
      { 
         if(b->ReadFromSingleMMFile(vecfile[i], matfilebase) != 0)
         {
            if(myid == 0)
            {
               PARGEMSLR_PRINT("Error reading vector file %s\n",vecfile[i]);
            }
            PargemslrFinalize();
            return -1;
         }
         if(myid == 0)
         {
            PARGEMSLR_PRINT("\tSolving with right-hand-side from file %s\n",vecfile[i]);
         }
      }
   
      switch(init_opt)
      {
         case 1:
         {
            /* one init guess */
            double  one = 1.0;
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
   
   for(i = 0 ; i < nmats ; i ++)
   {
      if(matfile[i])
      {
         free(matfile[i]);
      }
      if(vecfile[i])
      {
         free(vecfile[i]);
      }
   }
   free(matfile);
   free(vecfile);
   
   /* - - - - - - - - - - - - - - - -
    * 10. Finalize
    * - - - - - - - - - - - - - - - */
   
   PargemslrFinalize();
   
   return 0;
}
