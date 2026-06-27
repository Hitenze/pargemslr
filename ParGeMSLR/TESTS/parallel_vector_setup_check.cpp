#include "pargemslr.hpp"
#include <cstdio>

static int CheckVector(const char *label,
                       const pargemslr::ParallelVectorClass<double> &vec,
                       int local_n,
                       long int expected_start,
                       long int expected_global,
                       int rank)
{
   const int ok = vec.GetLengthLocal() == local_n &&
                  vec.GetStartGlobal() == expected_start &&
                  vec.GetLengthGlobal() == expected_global;
   if(!ok)
   {
      std::fprintf(stderr,
                   "rank %d %s mismatch: local=%d/%d start=%ld/%ld global=%ld/%ld\n",
                   rank,
                   label,
                   vec.GetLengthLocal(),
                   local_n,
                   vec.GetStartGlobal(),
                   expected_start,
                   vec.GetLengthGlobal(),
                   expected_global);
      return 1;
   }
   return 0;
}

static int AllRanksOk(int local_ok, MPI_Comm comm)
{
   int all_ok = 0;
   const int mpi_err = MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
   return mpi_err == MPI_SUCCESS && all_ok;
}

static int CheckZeroNorm(const char *label,
                         const pargemslr::ParallelVectorClass<double> &vec,
                         int rank)
{
   double norm = 0.0;
   const int err = vec.Norm2(norm);
   if(err != PARGEMSLR_SUCCESS)
   {
      std::fprintf(stderr, "rank %d %s Norm2 returned %d\n", rank, label, err);
      return 1;
   }
   if(norm != 0.0)
   {
      std::fprintf(stderr, "rank %d %s expected zero norm, actual=%.17e\n",
                   rank, label, norm);
      return 1;
   }
   return 0;
}

int main(int argc, char **argv)
{
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

      const int local_n = rank + 2;
      long int local_long = local_n;
      long int expected_start = 0;
      long int expected_global = 0;
      MPI_Exscan(&local_long, &expected_start, 1, MPI_LONG, MPI_SUM, comm);
      if(rank == 0)
      {
         expected_start = 0;
      }
      MPI_Allreduce(&local_long, &expected_global, 1, MPI_LONG, MPI_SUM, comm);

      {
         pargemslr::ParallelVectorClass<double> vec;
         err = vec.Setup(local_n, parlog);
         if(err != PARGEMSLR_SUCCESS)
         {
            std::fprintf(stderr, "rank %d Setup(n_local, parlog) returned %d\n", rank, err);
            exit_code = 1;
         }
         else
         {
            exit_code |= CheckVector("Setup(n_local, parlog)", vec, local_n, expected_start, expected_global, rank);
         }
      }

      {
         pargemslr::ParallelVectorClass<double> vec;
         int local_ok = 0;
         err = vec.Setup(local_n, true, parlog);
         if(err != PARGEMSLR_SUCCESS)
         {
            std::fprintf(stderr, "rank %d Setup(n_local, setzero, parlog) returned %d\n", rank, err);
            exit_code = 1;
         }
         else
         {
            const int check_err = CheckVector("Setup(n_local, setzero, parlog)", vec, local_n, expected_start, expected_global, rank);
            exit_code |= check_err;
            local_ok = check_err == 0;
         }
         if(AllRanksOk(local_ok, comm))
         {
            exit_code |= CheckZeroNorm("Setup(n_local, setzero, parlog)", vec, rank);
         }
      }

      {
         pargemslr::ParallelVectorClass<double> vec;
         int local_ok = 0;
         err = vec.Setup(local_n, pargemslr::kMemoryHost, true, parlog);
         if(err != PARGEMSLR_SUCCESS)
         {
            std::fprintf(stderr, "rank %d Setup(n_local, location, setzero, parlog) returned %d\n", rank, err);
            exit_code = 1;
         }
         else
         {
            const int check_err = CheckVector("Setup(n_local, location, setzero, parlog)", vec, local_n, expected_start, expected_global, rank);
            exit_code |= check_err;
            local_ok = check_err == 0;
         }
         if(AllRanksOk(local_ok, comm))
         {
            exit_code |= CheckZeroNorm("Setup(n_local, location, setzero, parlog)", vec, rank);
         }
      }

      {
         pargemslr::ParallelVectorClass<double> vec;
         err = vec.Setup(local_n, pargemslr::kMemoryHost, false, parlog);
         if(err != PARGEMSLR_SUCCESS)
         {
            std::fprintf(stderr, "rank %d Setup(n_local, location, setzero, parlog) returned %d\n", rank, err);
            exit_code = 1;
         }
         else
         {
            exit_code |= CheckVector("Setup(n_local, location, setzero, parlog)", vec, local_n, expected_start, expected_global, rank);
         }
      }

      int ok = exit_code == 0;
      int all_ok = 0;
      MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
      exit_code = all_ok ? 0 : 1;
      if(rank == 0 && exit_code == 0)
      {
         std::printf("parallel vector Setup offset check passed: ranks=%d global=%ld\n", np, expected_global);
      }
   }

   err = pargemslr::PargemslrFinalize();
   if(exit_code == 0 && err != PARGEMSLR_SUCCESS)
   {
      exit_code = err;
   }
   return exit_code;
}
