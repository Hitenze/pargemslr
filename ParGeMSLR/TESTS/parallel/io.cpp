#include "io.hpp"
#include "mpi.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io.hpp"
#include <iostream>

using namespace std;
using namespace pargemslr;

using namespace std;

int print_ilu_usage()
{
   printf("   -ILU options\n");
   printf("       -ilu_max_row_nnz [int]   Max number of nonzeros in L and U for ILUT.\n");
   printf("       -ilu_droptol     [real]  Drop tolerance of ILUT.\n");
   printf("       -ilu_perm_option [int]   Local reordering option for ILU.\n");
   printf("          0                     No special reordering.\n");
   printf("          1                     RCM ordering (defalut).\n");
   printf("       -ilu_omp_option  [int]   OpenMP option for ILU.\n");
   printf("          0                     Disable OpenMP.\n");
   printf("          1                     Level scheduling (not good for smaller problem) (defalut).\n");
   printf("          2                     Polynomial solve (won't generate same result as the sequential version) (defalut).\n");
   return 0;
}

int print_poly_usage()
{
   printf("   -Poly general options\n");
   printf("       -poly_order      [int]   The order of the polynomia (default 3).\n");
   return 0;
}

int print_gemslr_usage()
{
   printf("   -GeMSLR general options\n");
   //printf("       -A_reordering_option       [int]  Partition of the first level A.\n");
   //printf("          0                          Use same partition for the first level (default).\n");
   //printf("          1                          Use the nature ordering.\n");
   //printf("       -B_reordering_option       [int]  Set the local partition option.\n");
   //printf("          0                          No local partition.\n");
   //printf("          1                          Use RCM ordering(default).\n");
   //printf("       -use_global_setup          [bool] Should we use same setup for all levels.\n");
   printf("       -B_solve_option   [int]   Set the solve option for B.\n");
   printf("          0                          ILUT (default).\n");
   printf("          1                          Poly.\n");
   printf("       -global_perm_option [int] Set the global permutation option.\n");
   printf("          0                          No extra global permutation (default).\n");
   printf("          1                          Use the nature interior/exterior patition.\n");
   printf("       -schur_option     [int]   Set the schur complement option.\n");
   printf("          0                          Use the standard GeMSLR (default).\n");
   printf("          1                          Use partial ILU to form the first level schur complement.\n");
   printf("       -nlev             [int]   Set the target number of levels.\n");
   printf("       -minsep           [int]   Set the minimal size of the edge saperator.\n");
   printf("       -ncomp            [int]   Set the initial subdomain per level.\n");
   printf("       -kmin             [int]   Set the minimal subdomain per level.\n");
   printf("       -kfactor          [int]   Set the reduce factor of number of subdomains per level.\n");
   printf("   -GeMSLR B preconditioner options\n");
   printf("       -ilu_max_row_nnz  [int]   Max number of nonzeros in L and U for ILUT.\n");
   printf("       -ilu_droptol      [real]  Drop tolerance of ILUT.\n");
   printf("       -poly_order       [int]   The order of the polynomia (default 3).\n");
   printf("   -GeMSLR low-rank options\n");
   printf("       -lr_rand_init     [bool]  Should we use random init guess (false).\n");
   printf("       -lr_option        [int]   Set the arnoldi option.\n");
   printf("          0                          No restart (defalut).\n");
   printf("          1                          Thick-restart.\n");
   printf("       -lr_rank          [int]   Set the size of the low-rank correction (20).\n");
   printf("       -lr_rank_factor   [real]  Set the compute lr_rank*(1+lr_rank_factor) low-rank terms, and pick best of them.\n");
   printf("       -lr_arnoldi_factor[real]  Set the factor for the arnoldi steps each restart (3.0).\n");
   printf("       -lr_maxits        [int]   Set the max time of restarts (10).\n");
   printf("       -lr_chk_freq      [int]   Check convergence every lr_chk_freq restarts (1).\n");
   printf("       -lr_tr_factor     [real]  Set the thick-restart factor (0.25).\n");
   printf("       -lr_tol_eig       [real]  Set the tolerance of eigenvalues (1e-05).\n");
   printf("       -lr_tol_orth      [real]  Set the orthogonal threshold (1e-14).\n");
   printf("       -lr_tol_reorth    [real]  Set the re-orthogonal threshold (1/sqrt(2)).\n");
   return 0;
}

int print_fgmres_usage()
{
   printf("   -maxits         [int]    Set max # of iterations, default 1000.\n");
   printf("   -kdim           [int]    Set # of dimention for projection methods, default 50.\n");
   printf("   -tol            [float/double] Set tolorance of solving phase, default 1e-08.\n");
   return 0;
}

int print_laplacian_usage()
{
   printf("   Set the lapfile_real for real tests; lapfile_complex for complex tests\n");
   return 0;
}

int print_gen_usage()
{
   printf("   Set the matfile_real for real tests; matfile_complex for complex tests\n");
   printf("   -zerobase              The input is 0-based instead of 1-based.\n");
   return 0;
}

int print_lap_test_usage()
{
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Usage:               \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       IO Settings:         \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -fromfile              Read the \"inputs\" file in the current path as default setting. Any further setup will overwrite the setup.\n");
   printf("   -outfile    [str]      Write output to file str instead of stdout.\n");
   printf("   -print_option  [int]   Set the print option.\n");
   printf("       0                     Print only basic info (default).\n");
   printf("       1                     Print more info.\n");
   printf("       2                     Print even more info, use gnuplot to plot matrices.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Create Problem:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_laplacian_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Preconditioner:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_gemslr_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Solver:              \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_fgmres_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Other Options:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -gpu                     Enable CUDA.\n");
   
   return 0;
}

int print_parlap_test_usage()
{
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Usage:               \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("   Read settings from tile: \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -fromfile   [str]      Read input from new file instead of \"inputs\".\n");
   printf("   -outfile    [str]      Write output to file str instead of stdout.\n");
   printf("   -writesol   [str]      Write solution vector to file.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Create Problem:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_laplacian_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Other Options:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -gpu                   Enable CUDA.\n");
   printf("   -nthreads   [int]      Number of OpenMP threads.\n");
   
   return 0;
}

/*
int print_parlap_test_usage()
{
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Usage:               \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("   Read settings from tile: \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -fromfile              Read the \"inputs\" file in the current path as default setting. Any further setup will overwrite the setup.\n");
   printf("   -outfile    [str]      Write output to file str instead of stdout.\n");
   printf("   -print_option  [int]   Set the print option.\n");
   printf("       0                     Print only basic info (default).\n");
   printf("       1                     Print more info.\n");
   printf("       2                     Print even more info, use gnuplot to plot matrices.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Create Problem:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_laplacian_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Preconditioner:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_gemslr_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Solver:              \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_fgmres_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Other Options:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -gpu                     Enable CUDA.\n");
   
   return 0;
}
*/

int print_gen_test_usage()
{
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Usage:               \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("   Read settings from tile: \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -fromfile   [str]      Read input from new file instead of \"inputs\".\n");
   printf("   -outfile    [str]      Write output to file str instead of stdout.\n");
   printf("   -writesol   [str]      Write solution vector to file.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Create Problem:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_gen_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Other Options:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -gpu                   Enable CUDA.\n");
   printf("   -nthreads   [int]      Number of OpenMP threads.\n");
   
   return 0;
}

/*
int print_gen_test_usage()
{
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Usage:               \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("   Read settings from tile: \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -fromfile              Read the \"inputs\" file in the current path as default setting. Any further setup will overwrite the setup.\n");
   printf("   -outfile    [str]      Write output to file str instead of stdout.\n");
   printf("   -print_option  [int]   Set the print option.\n");
   printf("       0                     Print only basic info (default).\n");
   printf("       1                     Print more info.\n");
   printf("       2                     Print even more info, use gnuplot to plot matrices.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Create Problem:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_gen_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Preconditioner:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_gemslr_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Solver:              \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_fgmres_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Other Options:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -gpu                     Enable CUDA.\n");
   
   return 0;
}
*/

int print_sequential_test_usage()
{
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Usage:               \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("   Read settings from tile: \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -fromfile              Read the \"inputs\" file in the current path as default setting. Any further setup will overwrite the setup.\n");
   printf("   -outfile    [str]      Write output to file str instead of stdout.\n");
   printf("   -print_option  [int]   Set the print option.\n");
   printf("       0                     Print only basic info (default).\n");
   printf("       1                     Print more info.\n");
   printf("       2                     Print even more info, use gnuplot to plot matrices.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Precision:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -single                Use single precision.\n");
   printf("   -double                Use double precision (defualt).\n");
   printf("   -single_complex        Use single complex.\n");
   printf("   -double_complex        Use double complex.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Create Problem:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_gen_usage();
   print_laplacian_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Preconditioner:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_ilu_usage();
   print_gemslr_usage();
   print_poly_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Solver:              \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_fgmres_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Other Options:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -gpu                     Enable CUDA.\n");
   
   return 0;
}

int print_parallel_test_usage()
{
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Usage:               \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("   Read settings from tile: \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -fromfile              Read the \"inputs\" file in the current path as default setting. Any further setup will overwrite the setup.\n");
   printf("   -outfile    [str]      Write output to file str instead of stdout.\n");
   printf("   -print_option  [int]   Set the print option.\n");
   printf("       0                     Print only basic info (default).\n");
   printf("       1                     Print more info.\n");
   printf("       2                     Print even more info, use gnuplot to plot matrices.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Precision:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -single                Use single precision.\n");
   printf("   -double                Use double precision (defualt).\n");
   printf("   -single_complex        Use single complex.\n");
   printf("   -double_complex        Use double complex.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Create Matrix:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_gen_usage();
   print_laplacian_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("  Create Right-hand-side:   \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -vec         [str]     Read vector from file. Otherwise unit vector will be used as the solution.\n");
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Preconditioner:      \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_ilu_usage();
   print_gemslr_usage();
   print_poly_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Solver:              \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   print_fgmres_usage();
   printf(ANSI_COLOR_RED);
   printf(" - - - - - - - - - - - - - -\n");
   printf("       Other Options:       \n");
   printf(" - - - - - - - - - - - - - -\n");
   printf(ANSI_COLOR_RESET);
   printf("   -gpu                     Enable CUDA.\n");
   
   return 0;
}

int dummy_comm()
{
   int i, dummy_count, dummy_idx;
   dummy_count = 1;
   vector<MPI_Request> dummy_requests;
   
   int np, myid;
   
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   
   dummy_requests.resize(np*2);
   dummy_idx   = 0;
   vector<double> dummy_send(dummy_count*np,0), dummy_recv(dummy_count*np,0);
   for(i = 0 ; i < np ; i ++)
   {
      MPI_Isend(dummy_send.data()+dummy_count*i,dummy_count,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&(dummy_requests[dummy_idx++]));
   }
   for(i = 0 ; i < np ; i ++)
   {
      MPI_Irecv(dummy_recv.data()+dummy_count*i,dummy_count,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&(dummy_requests[dummy_idx++]));
   }
  
   MPI_Status dummy_status;
   for(i = 0 ; i < np*2 ; i ++)
   {
      MPI_Waitany(np*2,dummy_requests.data(),&dummy_idx,&dummy_status);
   }
   vector<double>().swap(dummy_send);
   vector<double>().swap(dummy_recv);
   vector<MPI_Request>().swap(dummy_requests);
   return 0;
}

int read_double_laplacian_param(int &nmats, int **nx, int **ny, int **nz, double **shift, double **alphax, double **alphay, double **alphaz)
{
   int i;
   
   FILE *f;
   
   if ((f = fopen("lapfile_real", "r")) == NULL)
   {
      printf("Can't open file.\n");
      return -1;
   }
   
   /* number of matrices */
   if(fscanf(f, "%d \n", &nmats) != 1)
   {
      printf("Error in lapfile number.\n");
      return -1;
   }
   
   *nx = (int*)malloc(sizeof(int)*nmats);
   *ny = (int*)malloc(sizeof(int)*nmats);
   *nz = (int*)malloc(sizeof(int)*nmats);
   *shift = (double*)malloc(sizeof(double)*nmats);
   *alphax = (double*)malloc(sizeof(double)*nmats);
   *alphay = (double*)malloc(sizeof(double)*nmats);
   *alphaz = (double*)malloc(sizeof(double)*nmats);
   
   for (i = 0; i < nmats; i++)
   {
      if(fscanf(f, "%d %d %d %lf %lf %lf %lf \n", 
            &((*nx)[i]), &((*ny)[i]), &((*nz)[i]),
            &((*shift)[i]),
            &((*alphax)[i]),
            &((*alphay)[i]),
            &((*alphaz)[i])) != 7)
      {
         printf("Error in inputs format at line %d.\n", i+1);
         return -1;
      }
   }
   fclose(f);
   
   return 0;
}

int read_double_complex_laplacian_param(int &nmats, int **nx, int **ny, int **nz, complexd **shift, complexd **alphax, complexd **alphay, complexd **alphaz)
{
   int i;
   
   FILE *f;
   
   if ((f = fopen("lapfile_complex", "r")) == NULL)
   {
      printf("Can't open file.\n");
      return -1;
   }
   
   /* number of matrices */
   if(fscanf(f, "%d \n", &nmats) != 1)
   {
      printf("Error in lapfile number.\n");
      return -1;
   }
   
   *nx = (int*)malloc(sizeof(int)*nmats);
   *ny = (int*)malloc(sizeof(int)*nmats);
   *nz = (int*)malloc(sizeof(int)*nmats);
   *shift = (complexd*)malloc(sizeof(complexd)*nmats);
   *alphax = (complexd*)malloc(sizeof(complexd)*nmats);
   *alphay = (complexd*)malloc(sizeof(complexd)*nmats);
   *alphaz = (complexd*)malloc(sizeof(complexd)*nmats);
   
   for (i = 0; i < nmats; i++)
   {
      if(fscanf(f, "%d %d %d %lf %lf %lf %lf %lf %lf %lf %lf \n", 
            &((*nx)[i]), &((*ny)[i]), &((*nz)[i]),
            &((*shift)[i].Real()), &((*shift)[i].Imag()),
            &((*alphax)[i].Real()), &((*alphax)[i].Imag()),
            &((*alphay)[i].Real()), &((*alphay)[i].Imag()),
            &((*alphaz)[i].Real()), &((*alphaz)[i].Imag())) != 11)
      {
         printf("Error in inputs format at line %d.\n", i+1);
         return -1;
      }
   }
   fclose(f);
   
   return 0;
}

#ifndef PI
#define PI 3.141592653589794
#endif

int read_double_complex_helmholtz_param(int &nmats, int **n, complexd **w)
{
   int i;
   double k;
   
   FILE *f;
   
   if ((f = fopen("helfile_complex", "r")) == NULL)
   {
      printf("Can't open file.\n");
      return -1;
   }
   
   /* number of matrices */
   if(fscanf(f, "%d \n", &nmats) != 1)
   {
      printf("Error in helfile number.\n");
      return -1;
   }
   
   *n = (int*)malloc(sizeof(int)*nmats);
   *w = (complexd*)malloc(sizeof(complexd)*nmats);
   
   for (i = 0; i < nmats; i++)
   {
      if(fscanf(f, "%d %lf \n", 
            &((*n)[i]),
            &(k)) != 2)
      {
         printf("Error in inputs format at line %d.\n", i+1);
         return -1;
      }
      (*w)[i]=complexd(k*2.0*PI,0.0);
   }
   fclose(f);
   
   return 0;
}

int read_matfile(const char *filename, int &nmats, char ***matfile, char ***vecfile)
{
   int i, j;
   
   FILE *f;
   
   if ((f = fopen(filename, "r")) == NULL)
   {
      printf("Can't open file.\n");
      return -1;
   }
   
   /* number of matrices */
   if(fscanf(f, "%d \n", &nmats) != 1)
   {
      printf("Error in lapfile number.\n");
      return -1;
   }
   
   *matfile = (char**)malloc(sizeof(char*)*nmats);
   *vecfile = (char**)malloc(sizeof(char*)*nmats);
   
   /* read mats and vecs */
   for (i = 0; i < nmats; i++)
   {
      (*matfile)[i] = (char*)malloc(sizeof(char)*1024);
      if(fscanf(f, "%s \n", (*matfile)[i]) != 1)
      {
         printf("Error in matrix inputs format at line %d.\n", i+1);
         return -1;
      }
      j = 0;
      if(fscanf(f, "%d ", &j) != 1)
      {
         printf("Error in vector inputs format at line %d.\n", i+1+nmats);
         return -1;
      }
      if(j != 0)
      {
         (*vecfile)[i] = (char*)malloc(sizeof(char)*1024);
         if(fscanf(f, "%s \n", (*vecfile)[i]) != 1)
         {
            printf("Error in vector inputs format at line %d.\n", i+1+nmats);
            return -1;
         }
      }
      else
      {
         (*vecfile)[i] = NULL;
         /* skip the current line */
         if(fscanf(f, "%*[^\n]\n"))
         {
            return PARGEMSLR_ERROR_IO_ERROR;
         }
      }
   }
   fclose(f);
   
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
            else if(!strcmp(word, "BJILUT2"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3] = kGemslrCSolveBJILUT2;
            }
            else if(!strcmp(word, "BJILUK2"))
            {
               params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3] = kGemslrCSolveBJILUK2;
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
            sscanf(line, "%lf", &params[PARGEMSLR_IO_SOLVER_KDIM]);
            break;
         }
         case 9:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_SOLVER_MAXITS]);
            break;
         }
         case 10:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_SOLVER_TOL]);
            break;
         }
         case 11:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_PREPOSS_NLEV_GLOBAL], &params[PARGEMSLR_IO_PREPOSS_NLEV_LOCAL]);
            break;
         }
         case 12:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_PREPOSS_NCOMP_GLOBAL], &params[PARGEMSLR_IO_PREPOSS_NCOMP_LOCAL]);
            break;
         }
         case 13:
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
         case 14:
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
         case 15:
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
         case 16:
         {
            sscanf(line, "%lf %lf %lf ", &params[PARGEMSLR_IO_ILU_ROWNNZ_B_GLOBAL],
                                             &params[PARGEMSLR_IO_ILU_ROWNNZ_S_GLOBAL],
                                             &params[PARGEMSLR_IO_ILU_ROWNNZ_C_GLOBAL]);
            params[PARGEMSLR_IO_ILU_ROWNNZ_B_LOCAL] = params[PARGEMSLR_IO_ILU_ROWNNZ_B_GLOBAL];
            params[PARGEMSLR_IO_ILU_ROWNNZ_S_LOCAL] = params[PARGEMSLR_IO_ILU_ROWNNZ_S_GLOBAL];
            params[PARGEMSLR_IO_ILU_ROWNNZ_C_LOCAL] = params[PARGEMSLR_IO_ILU_ROWNNZ_C_GLOBAL];
            break;
         }
         case 17:
         {
            double temp;
            sscanf(line, "%lf %lf %lf ", &params[PARGEMSLR_IO_ILU_LFIL_B_GLOBAL],
                                          &temp, &params[PARGEMSLR_IO_ILU_LFIL_C_GLOBAL]);
            params[PARGEMSLR_IO_ILU_LFIL_B_LOCAL] = params[PARGEMSLR_IO_ILU_LFIL_B_GLOBAL];
            params[PARGEMSLR_IO_ILU_LFIL_C_LOCAL] = params[PARGEMSLR_IO_ILU_LFIL_C_GLOBAL];
            break;
         }
         case 18:
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
         case 19:
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
         case 20:
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
         case 21:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_RANK1_GLOBAL],
                                         &params[PARGEMSLR_IO_LR_RANK2_GLOBAL]);
            params[PARGEMSLR_IO_LR_RANK1_LOCAL] = params[PARGEMSLR_IO_LR_RANK1_GLOBAL];
            params[PARGEMSLR_IO_LR_RANK2_LOCAL] = params[PARGEMSLR_IO_LR_RANK2_GLOBAL];
            break;
         }
         case 22:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_TOL_EIG1_GLOBAL], &params[PARGEMSLR_IO_LR_TOL_EIG2_GLOBAL]);
            params[PARGEMSLR_IO_LR_TOL_EIG1_LOCAL] = params[PARGEMSLR_IO_LR_TOL_EIG1_GLOBAL];
            params[PARGEMSLR_IO_LR_TOL_EIG2_LOCAL] = params[PARGEMSLR_IO_LR_TOL_EIG2_GLOBAL];
            break;
         }
         case 23:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_RANK_FACTOR1_GLOBAL], &params[PARGEMSLR_IO_LR_RANK_FACTOR2_GLOBAL]);
            params[PARGEMSLR_IO_LR_RANK_FACTOR1_LOCAL] = params[PARGEMSLR_IO_LR_RANK_FACTOR1_GLOBAL];
            params[PARGEMSLR_IO_LR_RANK_FACTOR2_LOCAL] = params[PARGEMSLR_IO_LR_RANK_FACTOR2_GLOBAL];
            break;
         }
         case 24:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_GLOBAL], &params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_GLOBAL]);
            params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_LOCAL] = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_GLOBAL];
            params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_LOCAL] = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_GLOBAL];
            break;
         }
         case 25:
         {
            sscanf(line, "%lf %lf", &params[PARGEMSLR_IO_LR_MAXITS1_GLOBAL], &params[PARGEMSLR_IO_LR_MAXITS2_GLOBAL]);
            params[PARGEMSLR_IO_LR_MAXITS1_LOCAL] = params[PARGEMSLR_IO_LR_MAXITS1_GLOBAL];
            params[PARGEMSLR_IO_LR_MAXITS2_LOCAL] = params[PARGEMSLR_IO_LR_MAXITS2_GLOBAL];
            break;
         }
         case 26:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_SCHUR_ENABLE]);
            break;
         }
         case 27:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_SCHUR_ITER_TOL]);
            break;
         }
         case 28:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_SCHUR_MAXITS]);
            break;
         }
         case 29:
         {
            //sscanf(line, "%lf", &params[PARGEMSLR_IO_ADVANCED_DIAG_SHIFT]);
            sscanf(line, "%lf", &params[PARGEMSLR_IO_ADVANCED_USE_COMPLEX_SHIFT]);
            break;
         }
         case 30:
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
         case 31:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_ADVANCED_RESIDUAL_ITERS]);
            break;
         }
         case 32:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_ADVANCED_GRAM_SCHMIDT]);
            break;
         }
         case 33:
         {
            sscanf(line, "%lf %lf %lf", &params[PARGEMSLR_IO_PARALLEL_NPROCX], &params[PARGEMSLR_IO_PARALLEL_NPROCY], &params[PARGEMSLR_IO_PARALLEL_NPROCZ]);
            break;
         }
         case 34:
         {
            sscanf(line, "%lf %lf %lf", &params[PARGEMSLR_IO_PARALLEL_NDOMX], &params[PARGEMSLR_IO_PARALLEL_NDOMY], &params[PARGEMSLR_IO_PARALLEL_NDOMZ]);
            break;
         }
         case 35:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL]);
            break;
         }
         case 36:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_PREPOSS_VTXSEP_GLOBAL]);
            params[PARGEMSLR_IO_PREPOSS_VTXSEP_LOCAL] = params[PARGEMSLR_IO_PREPOSS_VTXSEP_GLOBAL];
            break;
         }
         case 37:
         {
            sscanf(line, "%lf", &params[PARGEMSLR_IO_ADVANCED_DIAG_SHIFT_MODIFIED]);
            break;
         }
      }
   }
   
   fclose(f);
   
   return PARGEMSLR_SUCCESS;
}

/* read inputs from command line */
int read_inputs(double *input, int argc, char **argv)
{
   /* if not from file, set default values */
   int iint;
   double idouble;
   
   if( PargemslrReadInputArg("tol", 1, &idouble, argc, argv) )
   {
      input[PARGEMSLR_IO_SOLVER_TOL] = idouble;
   }
   
   if( PargemslrReadInputArg("kdim", 1, &iint, argc, argv) )
   {
      input[PARGEMSLR_IO_SOLVER_KDIM] = iint;
   }
   
   if( PargemslrReadInputArg("maxits", 1, &iint, argc, argv) )
   {
      input[PARGEMSLR_IO_SOLVER_MAXITS] = iint;
   }
   return 0;
}



