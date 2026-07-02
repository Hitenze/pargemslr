#include "pargemslr.hpp"
#include <cstdio>

static int CheckEqualLong(const char *label, long int actual, long int expected, int rank)
{
   if(actual != expected)
   {
      std::fprintf(stderr, "rank %d %s mismatch: actual=%ld expected=%ld\n",
                   rank, label, actual, expected);
      return 1;
   }
   return 0;
}

static int CheckEqualDouble(const char *label, double actual, double expected, int rank)
{
   if(actual != expected)
   {
      std::fprintf(stderr, "rank %d %s mismatch: actual=%.17e expected=%.17e\n",
                   rank, label, actual, expected);
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

static int CheckIdentityGraph(pargemslr::vector_long &vtxdist,
                              pargemslr::vector_long &xadj,
                              pargemslr::vector_long &adjncy,
                              int local_n,
                              long int expected_start,
                              long int expected_global,
                              int np,
                              int rank,
                              MPI_Comm comm)
{
   int exit_code = 0;
   pargemslr::vector_long starts;
   starts.Setup(np);
   MPI_Allgather(&expected_start, 1, MPI_LONG, starts.GetData(), 1, MPI_LONG, comm);

   exit_code |= CheckEqualLong("vtxdist length", vtxdist.GetLengthLocal(), np + 1, rank);
   for(int i = 0; i < np; i++)
   {
      exit_code |= CheckEqualLong("vtxdist entry", vtxdist[i], starts[i], rank);
   }
   exit_code |= CheckEqualLong("vtxdist final entry", vtxdist[np], expected_global, rank);

   exit_code |= CheckEqualLong("xadj length", xadj.GetLengthLocal(), local_n + 1, rank);
   for(int i = 0; i <= local_n; i++)
   {
      exit_code |= CheckEqualLong("xadj entry", xadj[i], 0, rank);
   }
   exit_code |= CheckEqualLong("adjncy length", adjncy.GetLengthLocal(), 0, rank);

   starts.Clear();
   return exit_code;
}

static int CheckCooEye(int rank)
{
   pargemslr::CooMatrixClass<double> coo;
   pargemslr::CsrMatrixClass<double> csr;
   int err = coo.Setup(4, 4);
   if(err == PARGEMSLR_SUCCESS)
   {
      err = coo.Eye();
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = coo.ToCsr(pargemslr::kMemoryHost, csr);
   }
   if(err != PARGEMSLR_SUCCESS)
   {
      std::fprintf(stderr, "rank %d COO identity construction returned %d\n", rank, err);
      return 1;
   }

   int exit_code = 0;
   exit_code |= CheckEqualLong("COO identity rows", csr.GetNumRowsLocal(), 4, rank);
   exit_code |= CheckEqualLong("COO identity cols", csr.GetNumColsLocal(), 4, rank);
   exit_code |= CheckEqualLong("COO identity nnz", csr.GetNumNonzeros(), 4, rank);

   int *row_ptr = csr.GetI();
   int *col_ind = csr.GetJ();
   double *data = csr.GetData();
   for(int row = 0; row < 4; row++)
   {
      const int row_start = row_ptr[row];
      const int row_nnz = row_ptr[row + 1] - row_start;
      exit_code |= CheckEqualLong("COO identity row nnz", row_nnz, 1, rank);
      if(row_nnz == 1)
      {
         exit_code |= CheckEqualLong("COO identity column", col_ind[row_start], row, rank);
         exit_code |= CheckEqualDouble("COO identity value", data[row_start], 1.0, rank);
      }
   }
   return exit_code;
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

      pargemslr::ParallelCsrMatrixClass<double> mat;
      err = mat.Setup(local_n, local_n, parlog);
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
         pargemslr::CsrMatrixClass<double> &diag = mat.GetDiagMat();
         int *diag_i = diag.GetI();
         int *diag_j = diag.GetJ();
         double *diag_data = diag.GetData();
         for(int row = 0; row < local_n; row++)
         {
            const int row_start = diag_i[row];
            if(diag_i[row + 1] - row_start != 1 || diag_j[row_start] != row)
            {
               std::fprintf(stderr, "rank %d diagonal row %d has unexpected CSR pattern\n", rank, row);
               err = PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
               break;
            }
            diag_data[row_start] = static_cast<double>(expected_start + row + 1);
         }
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
      int local_ok = err == PARGEMSLR_SUCCESS;
      if(err != PARGEMSLR_SUCCESS)
      {
         std::fprintf(stderr, "rank %d failed to construct identity ParallelCSR: %d\n", rank, err);
         exit_code = 1;
      }
      if(local_ok)
      {
         const int check_err =
            CheckEqualLong("local rows", mat.GetNumRowsLocal(), local_n, rank) |
            CheckEqualLong("local cols", mat.GetNumColsLocal(), local_n, rank) |
            CheckEqualLong("global rows", mat.GetNumRowsGlobal(), expected_global, rank) |
            CheckEqualLong("global cols", mat.GetNumColsGlobal(), expected_global, rank) |
            CheckEqualLong("row start", mat.GetRowStartGlobal(), expected_start, rank) |
            CheckEqualLong("col start", mat.GetColStartGlobal(), expected_start, rank) |
            CheckEqualLong("global nnz", mat.GetNumNonzeros(), expected_global, rank);
         exit_code |= check_err;
         local_ok = check_err == 0;
      }

      if(AllRanksOk(local_ok, comm))
      {
         pargemslr::vector_long vtxdist;
         pargemslr::vector_long xadj;
         pargemslr::vector_long adjncy;
         err = mat.GetGraphArrays(vtxdist, xadj, adjncy);
         local_ok = err == PARGEMSLR_SUCCESS;
         if(err != PARGEMSLR_SUCCESS)
         {
            std::fprintf(stderr, "rank %d GetGraphArrays returned %d\n", rank, err);
            exit_code = 1;
         }
         if(AllRanksOk(local_ok, comm))
         {
            exit_code |= CheckIdentityGraph(vtxdist, xadj, adjncy, local_n,
                                            expected_start, expected_global, np, rank, comm);
         }
      }

      local_ok = exit_code == 0;
      if(AllRanksOk(local_ok, comm))
      {
         pargemslr::vector_long rows;
         pargemslr::vector_long cols;
         int setup_err = rows.Setup(local_n);
         if(setup_err == PARGEMSLR_SUCCESS)
         {
            setup_err = cols.Setup(local_n);
         }
         local_ok = setup_err == PARGEMSLR_SUCCESS;
         if(setup_err != PARGEMSLR_SUCCESS)
         {
            std::fprintf(stderr, "rank %d failed to construct SubMatrix index vectors: %d\n", rank, setup_err);
            exit_code = 1;
         }
         if(local_ok)
         {
            for(int i = 0; i < local_n; i++)
            {
               const long int global_index = expected_start + (local_n - 1 - i);
               rows[i] = global_index;
               cols[i] = global_index;
            }
         }

         if(AllRanksOk(local_ok, comm))
         {
            pargemslr::ParallelCsrMatrixClass<double> submat;
            err = mat.SubMatrix(rows, cols, pargemslr::kMemoryHost, submat);
            local_ok = err == PARGEMSLR_SUCCESS;
            if(err != PARGEMSLR_SUCCESS)
            {
               std::fprintf(stderr, "rank %d SubMatrix returned %d\n", rank, err);
               exit_code = 1;
            }
            if(AllRanksOk(local_ok, comm))
            {
               exit_code |= CheckEqualLong("submatrix local rows", submat.GetNumRowsLocal(), local_n, rank);
               exit_code |= CheckEqualLong("submatrix local cols", submat.GetNumColsLocal(), local_n, rank);
               exit_code |= CheckEqualLong("submatrix global rows", submat.GetNumRowsGlobal(), expected_global, rank);
               exit_code |= CheckEqualLong("submatrix global cols", submat.GetNumColsGlobal(), expected_global, rank);
               exit_code |= CheckEqualLong("submatrix global nnz", submat.GetNumNonzeros(), expected_global, rank);
               exit_code |= CheckEqualLong("submatrix offd nnz", submat.GetOffdMat().GetNumNonzeros(), 0, rank);

               pargemslr::CsrMatrixClass<double> &subdiag = submat.GetDiagMat();
               int *sub_i = subdiag.GetI();
               int *sub_j = subdiag.GetJ();
               double *sub_data = subdiag.GetData();
               for(int row = 0; row < local_n; row++)
               {
                  const int row_start = sub_i[row];
                  const int row_nnz = sub_i[row + 1] - row_start;
                  const double expected_value = static_cast<double>(expected_start + local_n - row);
                  exit_code |= CheckEqualLong("submatrix row nnz", row_nnz, 1, rank);
                  if(row_nnz == 1)
                  {
                     exit_code |= CheckEqualLong("submatrix column order", sub_j[row_start], row, rank);
                     exit_code |= CheckEqualDouble("submatrix diagonal value", sub_data[row_start], expected_value, rank);
                  }
               }
            }
         }
      }

      if(AllRanksOk(exit_code == 0, comm))
      {
         exit_code |= CheckCooEye(rank);
      }

      if(AllRanksOk(exit_code == 0, comm))
      {
         char missing_file[128];
         std::snprintf(missing_file, sizeof(missing_file),
                       "pargemslr_missing_parallel_matrix_rank_%d.mtx", rank);
         std::remove(missing_file);

         pargemslr::ParallelCsrMatrixClass<double> missing_mat;
         const int read_err = missing_mat.ReadFromSingleMMFile(missing_file, 0, parlog);
         if(read_err != PARGEMSLR_ERROR_IO_ERROR)
         {
            std::fprintf(stderr,
                         "rank %d missing parallel Matrix Market read returned %d, expected %d\n",
                         rank, read_err, PARGEMSLR_ERROR_IO_ERROR);
            exit_code = 1;
         }
      }

      int ok = exit_code == 0;
      int all_ok = 0;
      MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
      exit_code = all_ok ? 0 : 1;
      if(rank == 0 && exit_code == 0)
      {
         std::printf("parallel matrix ops check passed: ranks=%d global=%ld\n", np, expected_global);
      }
   }

   err = pargemslr::PargemslrFinalize();
   if(exit_code == 0 && err != PARGEMSLR_SUCCESS)
   {
      exit_code = err;
   }
   return exit_code;
}
