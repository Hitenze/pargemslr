#include "pargemslr.hpp"
#include <cmath>
#include <cstdio>

static int SetupEmptyFactor(int n, pargemslr::matrix_csr_double &mat)
{
   int err = mat.Setup(n, n, 0, true);
   if(err == PARGEMSLR_SUCCESS)
   {
      int *i_ptr = mat.GetI();
      for(int i = 0; i <= n; i++)
      {
         i_ptr[i] = 0;
      }
   }
   return err;
}

static int BuildIdentityFactors(int n,
                                pargemslr::matrix_csr_double &a,
                                pargemslr::matrix_csr_double &l,
                                pargemslr::vector_seq_double &d,
                                pargemslr::matrix_csr_double &u)
{
   int err = a.Setup(n, n, n);
   if(err == PARGEMSLR_SUCCESS)
   {
      err = a.Eye();
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = SetupEmptyFactor(n, l);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = SetupEmptyFactor(n, u);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = d.Setup(n, true);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = d.Fill(1.0);
   }
   return err;
}

static int SolveAndCheck(pargemslr::precond_ilu_csr_seq_double &precond,
                         int n,
                         int location,
                         const char *stage)
{
   pargemslr::vector_seq_double x;
   pargemslr::vector_seq_double rhs;
   int err = x.Setup(n, location, true);
   if(err == PARGEMSLR_SUCCESS)
   {
      err = rhs.Setup(n, location, true);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = rhs.Fill(1.0);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = precond.Solve(x, rhs);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      err = x.MoveData(pargemslr::kMemoryHost);
   }
   if(err == PARGEMSLR_SUCCESS)
   {
      for(int i = 0; i < n; i++)
      {
         const double diff = std::fabs(x[i] - 1.0);
         if(!std::isfinite(diff) || diff > 1.0e-12)
         {
            std::fprintf(stderr, "%s ILU solve entry %d differs by %.17e\n", stage, i, diff);
            err = PARGEMSLR_ERROR_INVALED_PARAM;
            break;
         }
      }
   }
   if(err != PARGEMSLR_SUCCESS)
   {
      std::fprintf(stderr, "%s ILU solve check returned %d\n", stage, err);
   }
   x.Clear();
   rhs.Clear();
   return err;
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
      const int n = 5;
      pargemslr::matrix_csr_double a;
      pargemslr::matrix_csr_double l;
      pargemslr::matrix_csr_double u;
      pargemslr::vector_seq_double d;
      pargemslr::precond_ilu_csr_seq_double precond;

      err = BuildIdentityFactors(n, a, l, d, u);
      if(err == PARGEMSLR_SUCCESS)
      {
         err = precond.SetMatrix(a);
      }
      if(err == PARGEMSLR_SUCCESS)
      {
         err = precond.SetSolveLocation(pargemslr::kMemoryHost);
      }
      if(err == PARGEMSLR_SUCCESS)
      {
         err = precond.Setup(l, d, u, NULL, NULL);
      }
      if(err == PARGEMSLR_SUCCESS)
      {
         err = SolveAndCheck(precond, n, pargemslr::kMemoryHost, "host");
      }

#ifdef PARGEMSLR_CUDA
      if(err == PARGEMSLR_SUCCESS)
      {
         err = precond.SetSolveLocation(pargemslr::kMemoryDevice);
      }
      if(err == PARGEMSLR_SUCCESS)
      {
         err = SolveAndCheck(precond, n, pargemslr::kMemoryDevice, "device");
      }
      if(err == PARGEMSLR_SUCCESS)
      {
         err = precond.SetSolveLocation(pargemslr::kMemoryPinned);
      }
      if(err == PARGEMSLR_SUCCESS)
      {
         err = SolveAndCheck(precond, n, pargemslr::kMemoryPinned, "pinned");
      }
      if(err == PARGEMSLR_SUCCESS)
      {
         err = precond.SetSolveLocation(pargemslr::kMemoryDevice);
      }
      if(err == PARGEMSLR_SUCCESS)
      {
         err = SolveAndCheck(precond, n, pargemslr::kMemoryDevice, "device after pinned");
      }
#endif

      if(err == PARGEMSLR_SUCCESS)
      {
         std::printf("ILU solve state check passed\n");
      }
      else
      {
         exit_code = err;
      }

      precond.Clear();
      a.Clear();
      l.Clear();
      d.Clear();
      u.Clear();
   }

   err = pargemslr::PargemslrFinalize();
   if(exit_code == 0 && err != PARGEMSLR_SUCCESS)
   {
      exit_code = err;
   }
   return exit_code;
}
