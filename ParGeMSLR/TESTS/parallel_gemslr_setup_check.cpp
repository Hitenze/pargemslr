#include "pargemslr.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>

static int AllRanksOk(int local_ok, MPI_Comm comm)
{
   int all_ok = 0;
   const int mpi_err = MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
   return mpi_err == MPI_SUCCESS && all_ok;
}

static int BuildIdentity(int local_n,
                         pargemslr::parallel_log &parlog,
                         pargemslr::ParallelCsrMatrixClass<double> &mat,
                         int rank,
                         int location)
{
   int err = mat.Setup(local_n, local_n, parlog);
   if(err == PARGEMSLR_SUCCESS)
   {
      err = mat.GetDiagMat().Setup(local_n, local_n, local_n);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = mat.GetDiagMat().Eye();
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = mat.GetOffdMat().Setup(local_n, 0, 0, true);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      int *offd_i = mat.GetOffdMat().GetI();
      for(int i = 0; i <= local_n; i++)
      {
         offd_i[i] = 0;
      }
      err = mat.SetOffdMatSorted(true);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = mat.SetupMatvec();
   }
   if(err == PARGEMSLR_SUCCESS && location != pargemslr::kMemoryHost)
   {
      err = mat.MoveData(location);
   }
   if(err != PARGEMSLR_SUCCESS)
   {
      std::fprintf(stderr, "rank %d failed to build identity matrix: %d\n", rank, err);
   }
   return err;
}

static int BuildNoInteriorCoupled(int local_n,
                                  pargemslr::parallel_log &parlog,
                                  pargemslr::ParallelCsrMatrixClass<double> &mat,
                                  int rank,
                                  int location)
{
   MPI_Comm comm = MPI_COMM_NULL;
   int np = 0;
   int myid = 0;
   parlog.GetMpiInfo(np, myid, comm);
   if(np < 2)
   {
      std::fprintf(stderr, "rank %d coupled fallback matrix requires at least two MPI ranks\n", rank);
      return PARGEMSLR_ERROR_INVALED_PARAM;
   }

   int err = mat.Setup(local_n, local_n, parlog);
   if(err == PARGEMSLR_SUCCESS)
   {
      err = mat.GetDiagMat().Setup(local_n, local_n, local_n);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = mat.GetDiagMat().Eye();
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      double *diag_data = mat.GetDiagMat().GetData();
      for(int i = 0; i < local_n; i++)
      {
         diag_data[i] = 2.0;
      }
      err = mat.GetOffdMat().Setup(local_n, 1, local_n);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      int *offd_i = mat.GetOffdMat().GetI();
      int *offd_j = mat.GetOffdMat().GetJ();
      double *offd_data = mat.GetOffdMat().GetData();
      for(int i = 0; i < local_n; i++)
      {
         offd_i[i] = i;
         offd_j[i] = 0;
         offd_data[i] = -0.25;
      }
      offd_i[local_n] = local_n;
      err = mat.GetOffdMap().Setup(1);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      pargemslr::vector_long starts;
      long int row_start = mat.GetRowStartGlobal();
      err = starts.Setup(np);
      if(err == PARGEMSLR_SUCCESS)
      {
         MPI_Allgather(&row_start, 1, MPI_LONG, starts.GetData(), 1, MPI_LONG, comm);
         mat.GetOffdMap()[0] = starts[(rank + 1) % np];
      }
      starts.Clear();
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = mat.SetOffdMatSorted(true);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = mat.SetupMatvec();
   }
   if(err == PARGEMSLR_SUCCESS && location != pargemslr::kMemoryHost)
   {
      err = mat.MoveData(location);
   }
   if(err != PARGEMSLR_SUCCESS)
   {
      std::fprintf(stderr, "rank %d failed to build coupled matrix: %d\n", rank, err);
   }
   return err;
}

static int ConfigurePreconditioner(pargemslr::precond_gemslr_csr_par_double &precond,
                                    int global_option,
                                    int num_levels)
{
   int err = precond.SetGlobalPrecondOption(global_option);
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetGlobalPartitionOption(false);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetNumLevels(num_levels);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetInnerIterationOption(false);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetLowRankRanksTopLevel(0);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetLowRankRanksOtherLevels(0);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetLowRankA(0);
   }
   return err;
}

static int ConfigurePreconditionerWithParameters(pargemslr::precond_gemslr_csr_par_double &precond,
                                                 int global_option,
                                                 int num_levels)
{
   double params[PARGEMSLR_IO_SIZE];
   pargemslr::PargemslrSetDefaultParameterArray(params);
   params[PARGEMSLR_IO_PRECOND_GLOBAL_PRECOND] = global_option;
   params[PARGEMSLR_IO_PREPOSS_GLOBAL_PARTITION] = 0;
   params[PARGEMSLR_IO_PREPOSS_NLEV_GLOBAL] = num_levels;
   params[PARGEMSLR_IO_ILU_PERM_OPTION_GLOBAL] = pargemslr::kIluReorderingRcm;
   params[PARGEMSLR_IO_ILU_PERM_OPTION_LOCAL] = pargemslr::kIluReorderingRcm;
   params[PARGEMSLR_IO_SCHUR_ENABLE] = 0;
   params[PARGEMSLR_IO_LR_RANK1_GLOBAL] = 0;
   params[PARGEMSLR_IO_LR_RANK2_GLOBAL] = 0;
   params[PARGEMSLR_IO_LR_RANK_A] = 0;
   return precond.SetWithParameterArray(params);
}

static int CheckEffectiveOption(const pargemslr::precond_gemslr_csr_par_double &precond,
                                int expected_option,
                                const char *stage,
                                int rank)
{
   if(precond.GetGlobalPrecondOption() != expected_option)
   {
      std::fprintf(stderr,
                   "rank %d %s effective global preconditioner is %d, expected %d\n",
                   rank, stage, precond.GetGlobalPrecondOption(), expected_option);
      return PARGEMSLR_ERROR_INVALED_OPTION;
   }
   return PARGEMSLR_SUCCESS;
}

static int CheckGlobalPartition(const pargemslr::precond_gemslr_csr_par_double &precond,
                                bool expected_option,
                                const char *stage,
                                int rank)
{
   if(precond.GetGlobalPartitionOption() != expected_option)
   {
      std::fprintf(stderr,
                   "rank %d %s global partition flag is %d, expected %d\n",
                   rank, stage,
                   precond.GetGlobalPartitionOption() ? 1 : 0,
                   expected_option ? 1 : 0);
      return PARGEMSLR_ERROR_INVALED_OPTION;
   }
   return PARGEMSLR_SUCCESS;
}

static int SetupOnce(pargemslr::precond_gemslr_csr_par_double &precond,
                     pargemslr::ParallelCsrMatrixClass<double> &mat,
                     pargemslr::parallel_log &parlog,
                     int rank,
                     int location,
                     bool check_solve = true)
{
   const int local_n = mat.GetNumRowsLocal();
   pargemslr::vector_par_double x;
   pargemslr::vector_par_double b;
   pargemslr::vector_par_double residual;
   int err = x.Setup(local_n, mat.GetRowStartGlobal(), mat.GetNumRowsGlobal(),
                     location, true, parlog);
   if(err == PARGEMSLR_SUCCESS)
   {
      err = b.Setup(local_n, mat.GetRowStartGlobal(), mat.GetNumRowsGlobal(),
                    location, true, parlog);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = residual.Setup(local_n, mat.GetRowStartGlobal(), mat.GetNumRowsGlobal(),
                           location, true, parlog);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = b.Fill(1.0);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetMatrixP(&mat);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetOwnMatrix(false);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetSolveLocation(location);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.Setup(x, b);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.Solve(x, b);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      const double one = 1.0;
      const double zero = 0.0;
      err = mat.MatVec('N', one, x, zero, residual);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      const double minus_one = -1.0;
      err = residual.Axpy(minus_one, b);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      double residual_norm = 0.0;
      double rhs_norm = 0.0;
      err = residual.Norm2(residual_norm);
      if(err == PARGEMSLR_SUCCESS)
      {
         err = b.Norm2(rhs_norm);
      }
      if(err == PARGEMSLR_SUCCESS)
      {
         if(!std::isfinite(residual_norm) || !std::isfinite(rhs_norm))
         {
            std::fprintf(stderr,
                         "rank %d repeated Parallel GeMSLR solve produced non-finite norm %.17e %.17e\n",
                         rank, residual_norm, rhs_norm);
            err = PARGEMSLR_ERROR_INVALED_PARAM;
         }
         else if(check_solve)
         {
            const double scaled_residual = rhs_norm > 0.0 ? residual_norm / rhs_norm : residual_norm;
            if(!std::isfinite(scaled_residual) || scaled_residual > 1.0e-12)
            {
               std::fprintf(stderr,
                            "rank %d repeated Parallel GeMSLR solve residual %.17e exceeds tolerance\n",
                            rank, scaled_residual);
               err = PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
      }
   }
   if(err != PARGEMSLR_SUCCESS)
   {
      std::fprintf(stderr, "rank %d repeated Parallel GeMSLR setup/solve returned %d\n", rank, err);
   }
   x.Clear();
   b.Clear();
   residual.Clear();
   return err;
}

static int CheckInvalidSolveLevelReturns(pargemslr::parallel_log &parlog,
                                         int rank,
                                         int location)
{
#ifdef PARGEMSLR_DEBUG
   (void)parlog;
   (void)rank;
   (void)location;
   return PARGEMSLR_SUCCESS;
#else
   MPI_Comm comm = MPI_COMM_NULL;
   int np = 0;
   int myid = 0;
   parlog.GetMpiInfo(np, myid, comm);

   pargemslr::precond_gemslr_csr_par_double precond;
   pargemslr::vector_par_double x;
   pargemslr::vector_par_double b;

   const int local_n = 1;
   const long int row_start = rank;
   const long int n_global = np;

   int err = x.Setup(local_n, row_start, n_global, location, true, parlog);
   if(err == PARGEMSLR_SUCCESS)
   {
      err = b.Setup(local_n, row_start, n_global, location, true, parlog);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      const int solve_err = precond.SolveLevelGemslr(x, b, -1, false);
      if(solve_err != PARGEMSLR_ERROR_INVALED_PARAM)
      {
         std::fprintf(stderr,
                      "rank %d invalid Parallel GeMSLR solve level returned %d, expected %d\n",
                      rank, solve_err, PARGEMSLR_ERROR_INVALED_PARAM);
         err = PARGEMSLR_ERROR_INVALED_PARAM;
      }
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      const int solve_err = precond.SolveB(x, b, 0, -1);
      if(solve_err != PARGEMSLR_ERROR_INVALED_PARAM)
      {
         std::fprintf(stderr,
                      "rank %d invalid Parallel GeMSLR B solve level returned %d, expected %d\n",
                      rank, solve_err, PARGEMSLR_ERROR_INVALED_PARAM);
         err = PARGEMSLR_ERROR_INVALED_PARAM;
      }
   }

   x.Clear();
   b.Clear();
   precond.Clear();
   return err;
#endif
}

static int CheckSetupErrorRestoresMatrix(pargemslr::ParallelCsrMatrixClass<double> &mat,
                                         pargemslr::parallel_log &parlog,
                                         int rank,
                                         int location)
{
#ifdef PARGEMSLR_DEBUG
   (void)mat;
   (void)parlog;
   (void)rank;
   (void)location;
   return PARGEMSLR_SUCCESS;
#else
   pargemslr::precond_gemslr_csr_par_double precond;
   pargemslr::vector_par_double x;
   pargemslr::vector_par_double b;
   double params[PARGEMSLR_IO_SIZE];

   pargemslr::PargemslrSetDefaultParameterArray(params);
   params[PARGEMSLR_IO_PRECOND_GLOBAL_PRECOND] = 4.0;
   params[PARGEMSLR_IO_PREPOSS_GLOBAL_PARTITION] = 0.0;

   const int local_n = mat.GetNumRowsLocal();
   int err = x.Setup(local_n, mat.GetRowStartGlobal(), mat.GetNumRowsGlobal(),
                     location, true, parlog);
   if(err == PARGEMSLR_SUCCESS)
   {
      err = b.Setup(local_n, mat.GetRowStartGlobal(), mat.GetNumRowsGlobal(),
                    location, true, parlog);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = b.Fill(1.0);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetWithParameterArray(params);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetMatrixP(&mat);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetOwnMatrix(false);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.SetSolveLocation(location);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      const int setup_err = precond.Setup(x, b);
      if(setup_err != PARGEMSLR_ERROR_INVALED_OPTION)
      {
         std::fprintf(stderr,
                      "rank %d invalid Parallel GeMSLR setup returned %d, expected %d\n",
                      rank, setup_err, PARGEMSLR_ERROR_INVALED_OPTION);
         err = PARGEMSLR_ERROR_INVALED_OPTION;
      }
   }
   if(err == PARGEMSLR_SUCCESS && mat.GetDataLocation() != location)
   {
      std::fprintf(stderr,
                   "rank %d failed setup left matrix at location %d, expected %d\n",
                   rank, mat.GetDataLocation(), location);
      err = PARGEMSLR_ERROR_MEMORY_LOCATION;
   }

   x.Clear();
   b.Clear();
   precond.Clear();
   return err;
#endif
}

int main(int argc, char **argv)
{
   int location = pargemslr::kMemoryHost;
   bool setup_error_only = false;
   for(int argi = 1; argi < argc; argi++)
   {
      if(std::strcmp(argv[argi], "--device") == 0)
      {
#ifdef PARGEMSLR_CUDA
         location = pargemslr::kMemoryDevice;
#else
         std::fprintf(stderr, "--device requires a CUDA build\n");
         return PARGEMSLR_ERROR_INVALED_PARAM;
#endif
      }
      else if(std::strcmp(argv[argi], "--setup-error-only") == 0)
      {
         setup_error_only = true;
      }
      else
      {
         std::fprintf(stderr, "unknown option: %s\n", argv[argi]);
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }
   }

   int err = pargemslr::PargemslrInit(&argc, &argv);
   if(err != PARGEMSLR_SUCCESS)
   {
      return err;
   }

   int exit_code = 0;
   {
      pargemslr::parallel_log parlog;
      MPI_Comm comm = MPI_COMM_NULL;
      int np = 0;
      int rank = 0;
      parlog.GetMpiInfo(np, rank, comm);

      pargemslr::ParallelCsrMatrixClass<double> mat_a;
      pargemslr::ParallelCsrMatrixClass<double> mat_b;
      pargemslr::ParallelCsrMatrixClass<double> mat_c;
      pargemslr::ParallelCsrMatrixClass<double> mat_d;
      pargemslr::ParallelCsrMatrixClass<double> mat_e;
      pargemslr::ParallelCsrMatrixClass<double> mat_f;
      pargemslr::ParallelCsrMatrixClass<double> mat_g;
      pargemslr::ParallelCsrMatrixClass<double> mat_h;
      pargemslr::ParallelCsrMatrixClass<double> mat_i;
      pargemslr::precond_gemslr_csr_par_double precond;
      pargemslr::precond_gemslr_csr_par_double fallback_precond;
      pargemslr::precond_gemslr_csr_par_double option_precond;
      pargemslr::precond_gemslr_csr_par_double params_precond;
      pargemslr::precond_gemslr_csr_par_double recovery_precond;

      int local_ok = true;
      if(setup_error_only)
      {
         err = BuildIdentity(2, parlog, mat_g, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
         if(AllRanksOk(local_ok, comm))
         {
            err = CheckSetupErrorRestoresMatrix(mat_g, parlog, rank, location);
            local_ok = err == PARGEMSLR_SUCCESS;
         }
         if(!AllRanksOk(local_ok, comm))
         {
            exit_code = 1;
         }
         else if(rank == 0)
         {
            std::printf("parallel GeMSLR setup error check passed: ranks=%d location=%s\n",
                        np, location == pargemslr::kMemoryDevice ? "device" : "host");
         }
         mat_g.Clear();
         parlog.Clear();
         err = pargemslr::PargemslrFinalize();
         if(exit_code == 0 && err != PARGEMSLR_SUCCESS)
         {
            exit_code = err;
         }
         return exit_code;
      }

      err = CheckInvalidSolveLevelReturns(parlog, rank, location);
      local_ok = err == PARGEMSLR_SUCCESS;
      if(AllRanksOk(local_ok, comm))
      {
         err = ConfigurePreconditioner(precond, pargemslr::kGemslrGlobalPrecondBJ, 1);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = BuildIdentity(2, parlog, mat_a, rank, location);
      }
      local_ok = err == PARGEMSLR_SUCCESS;
      if(AllRanksOk(local_ok, comm))
      {
         err = SetupOnce(precond, mat_a, parlog, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = BuildIdentity(3, parlog, mat_b, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = SetupOnce(precond, mat_b, parlog, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = BuildIdentity(2, parlog, mat_g, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = CheckSetupErrorRestoresMatrix(mat_g, parlog, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = ConfigurePreconditioner(recovery_precond, pargemslr::kGemslrGlobalPrecondBJ, 1);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = SetupOnce(recovery_precond, mat_g, parlog, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }

      if(AllRanksOk(local_ok, comm))
      {
         err = ConfigurePreconditioner(fallback_precond, pargemslr::kGemslrGlobalPrecondGeMSLR, 2);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(np >= 2 && AllRanksOk(local_ok, comm))
      {
         err = BuildNoInteriorCoupled(2, parlog, mat_c, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(np >= 2 && AllRanksOk(local_ok, comm))
      {
         err = SetupOnce(fallback_precond, mat_c, parlog, rank, location, false);
         if(err == PARGEMSLR_SUCCESS)
         {
            err = CheckGlobalPartition(fallback_precond, true, "fallback setup", rank);
         }
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = BuildIdentity(8, parlog, mat_d, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = SetupOnce(fallback_precond, mat_d, parlog, rank, location);
         if(err == PARGEMSLR_SUCCESS)
         {
            err = CheckEffectiveOption(fallback_precond, pargemslr::kGemslrGlobalPrecondGeMSLR,
                                       "restored setup", rank);
         }
         if(err == PARGEMSLR_SUCCESS && np >= 2)
         {
            err = CheckGlobalPartition(fallback_precond, false, "restored setup", rank);
         }
         local_ok = err == PARGEMSLR_SUCCESS;
      }

      if(np >= 2 && AllRanksOk(local_ok, comm))
      {
         err = ConfigurePreconditioner(option_precond, pargemslr::kGemslrGlobalPrecondGeMSLR, 2);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(np >= 2 && AllRanksOk(local_ok, comm))
      {
         err = BuildIdentity(8, parlog, mat_h, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(np >= 2 && AllRanksOk(local_ok, comm))
      {
         err = SetupOnce(option_precond, mat_h, parlog, rank, location);
         if(err == PARGEMSLR_SUCCESS)
         {
            err = CheckEffectiveOption(option_precond, pargemslr::kGemslrGlobalPrecondGeMSLR,
                                       "option initial setup", rank);
         }
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(np >= 2 && AllRanksOk(local_ok, comm))
      {
         option_precond._global_precond_option = pargemslr::kGemslrGlobalPrecondBJ;
         err = BuildIdentity(9, parlog, mat_i, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(np >= 2 && AllRanksOk(local_ok, comm))
      {
         err = SetupOnce(option_precond, mat_i, parlog, rank, location);
         if(err == PARGEMSLR_SUCCESS)
         {
            err = CheckEffectiveOption(option_precond, pargemslr::kGemslrGlobalPrecondBJ,
                                       "public option setup", rank);
         }
         if(err == PARGEMSLR_SUCCESS)
         {
            err = CheckGlobalPartition(option_precond, false, "public option setup", rank);
         }
         local_ok = err == PARGEMSLR_SUCCESS;
      }

      if(AllRanksOk(local_ok, comm))
      {
         err = ConfigurePreconditionerWithParameters(params_precond,
                                                     pargemslr::kGemslrGlobalPrecondBJ, 2);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = BuildIdentity(2, parlog, mat_e, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = SetupOnce(params_precond, mat_e, parlog, rank, location);
         if(err == PARGEMSLR_SUCCESS)
         {
            err = CheckEffectiveOption(params_precond, pargemslr::kGemslrGlobalPrecondBJ,
                                       "parameter-array setup", rank);
         }
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = BuildIdentity(3, parlog, mat_f, rank, location);
         local_ok = err == PARGEMSLR_SUCCESS;
      }
      if(AllRanksOk(local_ok, comm))
      {
         err = SetupOnce(params_precond, mat_f, parlog, rank, location);
         if(err == PARGEMSLR_SUCCESS)
         {
            err = CheckEffectiveOption(params_precond, pargemslr::kGemslrGlobalPrecondBJ,
                                       "parameter-array repeated setup", rank);
         }
         local_ok = err == PARGEMSLR_SUCCESS;
      }

      if(!AllRanksOk(local_ok, comm))
      {
         exit_code = 1;
      }
      else if(rank == 0)
      {
         std::printf("parallel GeMSLR repeated setup check passed: ranks=%d location=%s\n",
                     np, location == pargemslr::kMemoryDevice ? "device" : "host");
      }

      precond.Clear();
      fallback_precond.Clear();
      option_precond.Clear();
      params_precond.Clear();
      recovery_precond.Clear();
      mat_a.Clear();
      mat_b.Clear();
      mat_c.Clear();
      mat_d.Clear();
      mat_e.Clear();
      mat_f.Clear();
      mat_g.Clear();
      mat_h.Clear();
      mat_i.Clear();
      parlog.Clear();
   }

   err = pargemslr::PargemslrFinalize();
   if(exit_code == 0 && err != PARGEMSLR_SUCCESS)
   {
      exit_code = err;
   }
   return exit_code;
}
