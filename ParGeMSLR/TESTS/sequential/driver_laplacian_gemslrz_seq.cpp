/**
 * @file driver_laplacian_gemslrz_seq.cpp
 * @brief Example file, use sequential fgmres + gemslr to solve complex laplacian.
 */


#include "pargemslr.hpp"
#include "io.hpp"
#include <iostream>

using namespace std;
using namespace pargemslr;

int main (int argc, char *argv[]) 
{
   
   int         n, i, nmats, *nx, *ny, *nz;
   int         sol_opt = 0;//0: use 1 as solution; 1: use rand as solution; 2: use 1 as rhs; 3 : use rand as rhs.
   int         init_opt = 0;//0: use 0 as initial guess; 1: use one as initial guess; 2: use random initial guess;
   complexd    *shift, *alphax, *alphay, *alphaz;
   bool        rand_perturb = false;
   long int    nnzA, nnzM, nnzLR, nnzILU;
   double      *params = pargemslr_global::_params;
   
   char outfile[1024], infile[1024], solfile[1024];
   bool writesol = false;
   int      location;
   
   /* print help when necessary */
   if(PargemslrReadInputArg("help", argc, argv))
   {
      print_lap_test_usage();
      return 0;
   }
   
   /* the sequential vector template */
   SequentialVectorClass<complexd> x, b;
   
   /* the sequential CSR matrix template */
   CsrMatrixClass<complexd> csr_mat;
   
   /* The FlexGMRES solver template, <Matrixtype, Vectortype, Datatype> */
   FlexGmresClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd> solver;
   
   /* The GemslrClass solver template, <Matrixtype, Vectortype, Datatype> */
   GemslrClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd> precond;
   
   /*--------------------------
    * 1. Read Laplacian Parameters
    *--------------------------*/
   
   if(read_double_complex_laplacian_param(nmats, &nx, &ny, &nz, &shift, &alphax, &alphay, &alphaz) != 0)
   {
      printf("Matrix file error\n");
      return -1;
   }
   
   /*-----------------------------------------
    * 2. Read Solver/Preconditioner settings
    *----------------------------------------*/
   
   /* should we read from input file? */
   if(!PargemslrReadInputArg("fromfile", infile, argc, argv))
   {
      printf("Reading setup from file \"inputs\"\n");
      PARGEMSLR_FIRM_CHKERR(read_inputs_from_file("inputs", params));
   }
   else
   {
      printf("Reading setup from file %s\n",infile);
      PARGEMSLR_FIRM_CHKERR(read_inputs_from_file(infile, params));
   }
   
   if(PargemslrReadInputArg("outfile", outfile, argc, argv))
   {
      if(PargemslrSetOutputFile(outfile) != 0)
      {
         printf("Reading output file error, write to stdout\n");
      }
      else
      {
         printf("Write to file %s\n",outfile);
      }
   }
   
   if(PargemslrReadInputArg("writesol", solfile, argc, argv))
   {
      printf("Write solution to file %s\n", solfile);
      writesol = true;
   }
   
   /* read terminal inputs, will overwrite file inputs */
   read_inputs(params, argc, argv);
   
   /* read solver location */
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
   
   /*-----------------------------------------
    * 3. Initialize ParGeMSLR package
    *----------------------------------------*/
   
   PargemslrInit( &argc, &argv);
   
   /* print parallel information */
   PargemslrPrintParallelInfo();
   
   if(parallel_log::_gsize != 1)
   {
      cout<<"Warning: this test file only works with np == 1"<<endl;
      PargemslrFinalize();
      return 0;
   }
   
   /* main loop */
   printf("Start running tests. Total %d tests.\n", nmats);
   for(i = 0 ; i < nmats ; i ++)
   {
      /*-----------------------------------------
       * 4. Create Test Matrix
       *----------------------------------------*/
      
      CsrMatrixClass<complexd> testcsr;
      
      PARGEMSLR_PRINT("Running test number %d\n",i+1);
      
      /* reset the seed */
      pargemslr_global::_mersenne_twister_engine.seed(0);
      
      /* generate Laplacian */
      PARGEMSLR_LOCAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_GEN_MAT_TIME, (csr_mat.Laplacian( nx[i], ny[i], nz[i], alphax[i], alphay[i], alphaz[i], shift[i], rand_perturb)));
      if(rand_perturb)
      {
         PARGEMSLR_PRINT("\tSolving Laplacian %d %d %d %lf+%lfi %lf+%lfi %lf+%lfi %lf+%lfi with random diagonal pertubation\n",nx[i],ny[i],nz[i],alphax[i].Real(),alphax[i].Imag(),alphay[i].Real(),alphay[i].Imag(),alphaz[i].Real(),alphaz[i].Imag(),shift[i].Real(),shift[i].Imag());
      }
      else
      {
         PARGEMSLR_PRINT("\tSolving Laplacian %d %d %d %lf+%lfi %lf+%lfi %lf+%lfi %lf+%lfi\n",nx[i],ny[i],nz[i],alphax[i].Real(),alphax[i].Imag(),alphay[i].Real(),alphay[i].Imag(),alphaz[i].Real(),alphaz[i].Imag(),shift[i].Real(),shift[i].Imag());
      }
      csr_mat.SetComplexShift(params[PARGEMSLR_IO_ADVANCED_DIAG_SHIFT]);
      
      n = csr_mat.GetNumRowsLocal();
      
      /* move to device when device memory is availiable */
      csr_mat.MoveData(location);
      
      /*------------------------------------------------
       * 5. Create initial guess and the right-hand-side
       *-----------------------------------------------*/
      
      complexd  one = 1.0, zero = 0.0;
      
      /* create vector, on device when device memory is availiable */
      x.Setup(n, location, false);
      b.Setup(n, location, false);
      
      switch(sol_opt)
      {
         case 1:
         {
            /* use random solution */
            x.Rand();
            b.Fill(zero);
            csr_mat.MatVec( 'N', one, x, zero, b);
            printf("\tUsing random solution\n");
            break;
         }
         case 2:
         {
            /* use 1 rhs */
            b.Fill(one);
            printf("\tUsing one vector as right-hand-side\n");
            break;
         }
         case 3:
         {
            /* use random rhs */
            b.Rand();
            printf("\tUsing random right-hand-side\n");
            break;
         }
         case 0: default:
         {
            /* use 1 as solution */
            x.Fill(one);
            b.Fill(zero);
            csr_mat.MatVec( 'N', one, x, zero, b);
            printf("\tUsing one vector as solution\n");
            break;
         }
      }
      
      switch(init_opt)
      {
         case 1:
         {
            /* one init guess */
            x.Fill(one);
            printf("\tUsing one vector as initial guess\n");
            break;
         }
         case 2:
         {
            /* random initial guess */
            x.Rand();
            printf("\tUsing random initial guess\n");
            break;
         }
         case 0: default:
         {
            /* zero init guess */
            x.Fill(zero);
            printf("\tUsing zero vector as initial guess\n");
            break;
         }
      }
      
      /*------------------------------------------------
       * 6. Create FlexGMRES solver
       *-----------------------------------------------*/
      
      /* set matrix */
      solver.SetMatrix(csr_mat);
      /* set parameter with parameter array, can also use individual set functions */
      solver.SetWithParameterArray(params);
      /* move to GPU when necessary */
      solver.SetSolveLocation(location);
      
      /*------------------------------------------------
       * 7. Create GeMSLR preconditioner
       *-----------------------------------------------*/
      
      /* set matrix */
      precond.SetMatrix(csr_mat);
      /* set parameter with parameter array, can also use individual set functions */
      precond.SetWithParameterArray(params);
      /* move to GPU when necessary */
      precond.SetSolveLocation(location);  
      
      /*------------------------------------------------
       * 8. Link the preconditioner with the solver
       *-----------------------------------------------*/
      
      /* assign preconditioner to solver */
      solver.SetPreconditioner(precond);
      
      /*------------------------------------------------
       * 9. Setup phase
       *-----------------------------------------------*/
      
      PARGEMSLR_LOCAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_SETUP_TIME, (solver.Setup(x, b)));
      
      /*------------------------------------------------
       * 10. Solve phase
       *-----------------------------------------------*/
      
      PARGEMSLR_LOCAL_FIRM_TIME_CALL(PARGEMSLR_TOTAL_SOLVE_TIME, (solver.Solve(x, b)));
      
      /*------------------------------------------------
       * 11. Compute fill-level
       *-----------------------------------------------*/
      
      nnzA = csr_mat.GetNumNonzeros();
      nnzM = precond.GetNumNonzeros(nnzILU, nnzLR);
      
      /*------------------------------------------------
       * 12. Print solution info
       *-----------------------------------------------*/
      
      PARGEMSLR_PRINT("\n");
      PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
      PARGEMSLR_PRINT("Solution info:\n");
      PARGEMSLR_PRINT("\tNumber of iterations: %d\n",solver.GetNumberIterations());
      PARGEMSLR_PRINT("\tFinal rel res: %f\n",solver.GetFinalRelativeResidual());
      PARGEMSLR_PRINT("\tPreconditioner fill level: ILU: %f; Low-rank: %f; Total: %f\n",(double)nnzILU/nnzA,(double)nnzLR/nnzA,(double)nnzM/nnzA);
      /*
      if(params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL] > 0)
      {
         x.MoveData(kMemoryHost);
         PARGEMSLR_PRINT("\tSample solution:\n");
         PargemslrPlotData(x.GetData(), n, 30, 5);
      }
      */
      PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
      PARGEMSLR_PRINT("\n");
      
      PARGEMSLR_PRINT_TIMING_RESULT(params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL], 1);
      
      /* reset time */
      PARGEMSLR_RESET_TIME;
      
      /*------------------------------------------------
       * 13. Deallocate
       *-----------------------------------------------*/
      
      if(writesol)
      {
         char tempsolname[2048];
         snprintf( tempsolname, 2048, "./%s%05d%s", solfile, i, ".sol" );
         x.WriteToDisk(tempsolname);
      }
      
      x.Clear();
      b.Clear();
      csr_mat.Clear();
      solver.Clear();
      precond.Clear();
   }
   
   printf("All tests done\n");
   
   free(nx);
   free(ny);
   free(nz);
   free(shift);
   free(alphax);
   free(alphay);
   free(alphaz);
   
   /*------------------------------------------------
    * 14. Finalize the ParGeMSLR package
    *-----------------------------------------------*/
   
   PargemslrFinalize();
   
   return 0;
}
