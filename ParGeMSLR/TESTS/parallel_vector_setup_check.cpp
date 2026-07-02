#include "pargemslr.hpp"
#include <cstdio>
#include <unistd.h>

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

static int WriteShortVectorFile(const char *filename, int rank)
{
   if(rank != 0)
   {
      return 0;
   }

   FILE *file = std::fopen(filename, "w");
   if(file == NULL)
   {
      return 1;
   }
   const int wrote = std::fprintf(file, "%%%%MatrixMarket matrix coordinate real general\n1 1 0\n");
   const int close_err = std::fclose(file);
   return (wrote < 0 || close_err != 0) ? 1 : 0;
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

      if(AllRanksOk(exit_code == 0, comm))
      {
         char missing_file[128];
         std::snprintf(missing_file, sizeof(missing_file),
                       "pargemslr_missing_parallel_vector_rank_%d.mtx", rank);
         std::remove(missing_file);

         pargemslr::ParallelVectorClass<double> missing_vec;
         err = missing_vec.Setup(local_n, parlog);
         if(err == PARGEMSLR_SUCCESS)
         {
            err = missing_vec.ReadFromSingleMMFile(missing_file, 0);
         }
         if(err != PARGEMSLR_ERROR_IO_ERROR)
         {
            std::fprintf(stderr,
                         "rank %d missing parallel vector Matrix Market read returned %d, expected %d\n",
                         rank, err, PARGEMSLR_ERROR_IO_ERROR);
            exit_code = 1;
         }
      }

      if(AllRanksOk(exit_code == 0, comm))
      {
         char mismatch_file[128];
         std::snprintf(mismatch_file, sizeof(mismatch_file),
                       "pargemslr_short_parallel_vector_rank_%d_pid_%ld.mtx",
                       rank, static_cast<long>(getpid()));
         const int file_ok = WriteShortVectorFile(mismatch_file, rank) == 0;
         int all_file_ok = 0;
         MPI_Allreduce(&file_ok, &all_file_ok, 1, MPI_INT, MPI_MIN, comm);
         MPI_Barrier(comm);

         if(!all_file_ok)
         {
            std::fprintf(stderr, "rank %d failed to create vector mismatch fixture\n", rank);
            exit_code = 1;
         }
         else
         {
            pargemslr::ParallelVectorClass<double> mismatch_vec;
            err = mismatch_vec.Setup(local_n, parlog);
            if(err == PARGEMSLR_SUCCESS)
            {
               err = mismatch_vec.ReadFromSingleMMFile(mismatch_file, 0);
            }
            if(err != PARGEMSLR_ERROR_INVALED_PARAM)
            {
               std::fprintf(stderr,
                            "rank %d mismatched parallel vector Matrix Market read returned %d, expected %d\n",
                            rank, err, PARGEMSLR_ERROR_INVALED_PARAM);
               exit_code = 1;
            }
         }

         MPI_Barrier(comm);
         if(rank == 0)
         {
            std::remove(mismatch_file);
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
