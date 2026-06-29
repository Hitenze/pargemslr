#include <unordered_map>
#include <unordered_set>
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../utils/memory.hpp"
#include "../utils/protos.hpp"
#include "../vectors/vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "matrix.hpp"
#include "matrixops.hpp"
#include "dense_matrix.hpp"
#include "csr_matrix.hpp"
#include "parallel_csr_matrix.hpp"

#include <iostream>
#include <complex>
#include <limits>
#include <limits.h>

#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#endif

using namespace std;

namespace pargemslr
{
   template <typename T>
   int DenseMatrixPlotHost( DenseMatrixClass<T> &A, int conditiona, int conditionb, int width)
   {

      if(conditiona != conditionb)
      {
         return PARGEMSLR_SUCCESS;
      }

      std::cout<<"Ploting "<<A.GetNumRowsLocal()<<" by "<<A.GetNumColsLocal()<<" matrix."<<std::endl;

      int i, j, nrows, ncols;

      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();

      for(i = 0 ; i < nrows ; i ++)
      {
         for(j = 0 ; j < ncols ; j ++)
         {
            PargemslrPrintValueHost(A(i,j), width);
            std::cout<<", ";
         }
         std::cout<<std::endl;
      }
      return PARGEMSLR_SUCCESS;
   }
   template int DenseMatrixPlotHost( DenseMatrixClass<float> &A, int conditiona, int conditionb, int width);
   template int DenseMatrixPlotHost( DenseMatrixClass<double> &A, int conditiona, int conditionb, int width);
   template int DenseMatrixPlotHost( DenseMatrixClass<complexs> &A, int conditiona, int conditionb, int width);
   template int DenseMatrixPlotHost( DenseMatrixClass<complexd> &A, int conditiona, int conditionb, int width);

   template <typename T>
   int CsrMatrixPlotHost( CsrMatrixClass<T> &A, int *perm, int conditiona, int conditionb, int width)
   {
      if(!A.IsHoldingData())
      {
         std::cout<<"Plot a Matrix not holding value."<<std::endl;
         return PARGEMSLR_ERROR_INVALED_OPTION;
      }

      if(!(A.IsCsr()))
      {
         std::cout<<"Plot only for CSR matrix, convert to csr."<<std::endl;
         A.Convert(true);
      }

      int   i, ii, j, j1, j2;
      int   *A_i = A.GetI();
      int   *A_j = A.GetJ();
      T     *A_data = A.GetData();

      vector_int marker;
      SequentialVectorClass<T> value;
      if(conditiona == conditionb)
      {
         std::cout<<"Ploting "<<A.GetNumRowsLocal()<<" by "<<A.GetNumColsLocal()<<" matrix with "<<A.GetNumNonzeros()<<" nnzs."<<std::endl;
         if(A.GetNumRowsLocal() == 0 || A.GetNumColsLocal() == 0)
         {
            return PARGEMSLR_SUCCESS;
         }
         if(perm == NULL)
         {
            marker.Setup(A.GetNumColsLocal());
            value.Setup(A.GetNumColsLocal());
            for(i = 0 ; i < A.GetNumRowsLocal() ; i ++)
            {
               marker.Fill(-1);
               j1 = A_i[i];
               j2 = A_i[i+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  marker[A_j[j]] = 1;
                  value[A_j[j]] = A_data[j];
               }
               for( j = 0 ; j < A.GetNumColsLocal() ; j ++)
               {
                  if( marker[j] < 0)
                  {
                     PargemslrPrintValueHost(T(),width);
                  }
                  else
                  {
                     PargemslrPrintValueHost(value[j],width);
                  }
                  std::cout<<", ";
               }
               std::cout<<std::endl;
            }
            value.Clear();
            marker.Clear();
         }
         else
         {
            marker.Setup(A.GetNumColsLocal());
            value.Setup(A.GetNumColsLocal());
            for(ii = 0 ; ii < A.GetNumRowsLocal() ; ii ++)
            {
               marker.Fill(-1);
               i = perm[ii];
               j1 = A_i[i];
               j2 = A_i[i+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  marker[A_j[j]] = 1;
                  value[A_j[j]] = A_data[j];
               }
               for( j = 0 ; j < A.GetNumColsLocal() ; j ++)
               {
                  if( marker[perm[j]] < 0)
                  {
                     PargemslrPrintValueHost(T(),width);
                  }
                  else
                  {
                     PargemslrPrintValueHost(value[perm[j]],width);
                  }
                  std::cout<<", ";
               }
               std::cout<<std::endl;
            }
            value.Clear();
            marker.Clear();
         }
      }

      return PARGEMSLR_SUCCESS;

   }
   template int CsrMatrixPlotHost( CsrMatrixClass<float> &A, int *perm, int conditiona, int conditionb, int width);
   template int CsrMatrixPlotHost( CsrMatrixClass<double> &A, int *perm, int conditiona, int conditionb, int width);
   template int CsrMatrixPlotHost( CsrMatrixClass<complexs> &A, int *perm, int conditiona, int conditionb, int width);
   template int CsrMatrixPlotHost( CsrMatrixClass<complexd> &A, int *perm, int conditiona, int conditionb, int width);

   int CooMatrixReadFromFile(CooMatrixClass<float> &coo, const char *matfile, int idxin, int idxout)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;
      int i, I, J;
      int shift = idxin - idxout;
      float val;

      if ((f = fopen( matfile, "r")) == NULL)
      {
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if (mm_read_banner(f, &matcode) != 0)
      {
         printf("Could not process Matrix Market banner.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }


      /*  This is how one can screen matrix types if their application */
      /*  only supports a subset of the Matrix Market data types.      */

      if (  !(mm_is_real(matcode) || mm_is_integer(matcode)) ||
            !mm_is_matrix(matcode) ||
            !mm_is_coordinate(matcode))
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         PARGEMSLR_ERROR("Error reading MM file.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      {
         printf("Invalid Size.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( mm_is_general(matcode) )
      {
         coo.Setup( M, N, nz);
      }
      else
      {
         coo.Setup( M, N, 2*nz);
      }

      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

      for (i=0; i<nz; i++)
      {
         if( fscanf(f, "%d %d %f\n", &I, &J, &val) != 3 )
         {
            PARGEMSLR_ERROR("Error reading MM file.");
            return PARGEMSLR_ERROR_IO_ERROR;
         }
         I -= shift;  /* adjust from 1-based to 0-based */
         J -= shift;
         coo.PushBack( I, J, val);
         if(I != J)
         {
            if( mm_is_symmetric(matcode) )
            {
               coo.PushBack( J, I, val);
            }
         }
      }

      if (f !=stdin) fclose(f);

      return PARGEMSLR_SUCCESS;
   }

   int CooMatrixReadFromFile(CooMatrixClass<double> &coo, const char *matfile, int idxin, int idxout)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;
      int i, I, J;
      int shift = idxin - idxout;
      double val;

      if ((f = fopen( matfile, "r")) == NULL)
      {
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if (mm_read_banner(f, &matcode) != 0)
      {
         printf("Could not process Matrix Market banner.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }


      /*  This is how one can screen matrix types if their application */
      /*  only supports a subset of the Matrix Market data types.      */

      if (  !(mm_is_real(matcode) || mm_is_integer(matcode)) ||
            !mm_is_matrix(matcode) ||
            !mm_is_coordinate(matcode) )
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      {
         printf("Invalid Size.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( mm_is_general(matcode) )
      {
         coo.Setup( M, N, nz);
      }
      else
      {
         coo.Setup( M, N, 2*nz);
      }

      /* reseve memory for matrices */


      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

      for (i=0; i<nz; i++)
      {
         if( fscanf(f, "%d %d %lg\n", &I, &J, &val) != 3 )
         {
            PARGEMSLR_ERROR("Error reading MM file.");
            return PARGEMSLR_ERROR_IO_ERROR;
         }
         I -= shift;  /* adjust from 1-based to 0-based */
         J -= shift;
         coo.PushBack( I, J, val);
         if(I != J)
         {
            if( mm_is_symmetric(matcode) )
            {
               coo.PushBack( J, I, val);
            }
         }
      }

      if (f !=stdin) fclose(f);

      return PARGEMSLR_SUCCESS;
   }

   int CooMatrixReadFromFile(CooMatrixClass<complexs> &coo, const char *matfile, int idxin, int idxout)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;
      int i, I, J;
      int shift = idxin - idxout;
      float valr, vali;

      if ((f = fopen( matfile, "r")) == NULL)
      {
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if (mm_read_banner(f, &matcode) != 0)
      {
         printf("Could not process Matrix Market banner.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }


      /*  This is how one can screen matrix types if their application */
      /*  only supports a subset of the Matrix Market data types.      */

      if ( !(mm_is_complex(matcode) || mm_is_real(matcode) || mm_is_integer(matcode)) ||
            !mm_is_matrix(matcode) || !mm_is_coordinate(matcode) )
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      {
         printf("Invalid Size.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( mm_is_general(matcode) )
      {
         coo.Setup( M, N, nz);
      }
      else
      {
         coo.Setup( M, N, 2*nz);
      }

      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
      if(mm_is_complex(matcode))
      {
         /* read complex matrix */
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %f %f\n", &I, &J, &valr, &vali) != 4 )
            {
               PARGEMSLR_ERROR("Error reading MM file.");
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            I -= shift;  /* adjust from 1-based to 0-based */
            J -= shift;
            coo.PushBack( I, J, complexs(valr, vali));
            if(I != J)
            {
               if( mm_is_symmetric(matcode) )
               {
                  coo.PushBack( J, I, complexs(valr, vali));
               }
               else if( mm_is_hermitian(matcode) )
               {
                  coo.PushBack( J, I, complexs(valr, -vali));
               }
               else if( mm_is_skew(matcode) )
               {
                  coo.PushBack( J, I, complexs(-valr, vali));
               }
            }
         }
      }
      else
      {
         /* read real matrix */
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %f\n", &I, &J, &valr) != 3 )
            {
               PARGEMSLR_ERROR("Error reading MM file.");
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            I -= shift;  /* adjust from 1-based to 0-based */
            J -= shift;
            coo.PushBack( I, J, complexs(valr, 0.0));
            if(I != J)
            {
               if( mm_is_symmetric(matcode) )
               {
                  coo.PushBack( J, I, complexs(valr, 0.0));
               }
               else if( mm_is_hermitian(matcode) )
               {
                  coo.PushBack( J, I, complexs(valr, 0.0));
               }
               else if( mm_is_skew(matcode) )
               {
                  coo.PushBack( J, I, complexs(-valr, 0.0));
               }
            }
         }
      }

      if (f !=stdin) fclose(f);

      return PARGEMSLR_SUCCESS;
   }

   int CooMatrixReadFromFile(CooMatrixClass<complexd> &coo, const char *matfile, int idxin, int idxout)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;
      int i, I, J;
      int shift = idxin - idxout;
      double valr, vali;

      if ((f = fopen( matfile, "r")) == NULL)
      {
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if (mm_read_banner(f, &matcode) != 0)
      {
         printf("Could not process Matrix Market banner.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }


      /*  This is how one can screen matrix types if their application */
      /*  only supports a subset of the Matrix Market data types.      */

      if ( !(mm_is_complex(matcode) || mm_is_real(matcode) || mm_is_integer(matcode)) ||
            !mm_is_matrix(matcode) || !mm_is_coordinate(matcode) )
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      {
         printf("Invalid Size.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( mm_is_general(matcode) )
      {
         coo.Setup( M, N, nz);
      }
      else
      {
         coo.Setup( M, N, 2*nz);
      }

      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

      if(mm_is_complex(matcode))
      {
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %lg %lg\n", &I, &J, &valr, &vali) != 4 )
            {
               PARGEMSLR_ERROR("Error reading MM file.");
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            I -= shift;  /* adjust from 1-based to 0-based */
            J -= shift;
            coo.PushBack( I, J, complexd(valr, vali));
            if(I != J)
            {
               if( mm_is_symmetric(matcode) )
               {
                  coo.PushBack( J, I, complexd(valr, vali));
               }
               else if( mm_is_hermitian(matcode) )
               {
                  coo.PushBack( J, I, complexd(valr, -vali));
               }
               else if( mm_is_skew(matcode) )
               {
                  coo.PushBack( J, I, complexd(-valr, vali));
               }
            }
         }
      }
      else
      {
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %lg\n", &I, &J, &valr) != 3 )
            {
               PARGEMSLR_ERROR("Error reading MM file.");
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            I -= shift;  /* adjust from 1-based to 0-based */
            J -= shift;
            coo.PushBack( I, J, complexd(valr, 0.0));
            if(I != J)
            {
               if( mm_is_symmetric(matcode) )
               {
                  coo.PushBack( J, I, complexd(valr, 0.0));
               }
               else if( mm_is_hermitian(matcode) )
               {
                  coo.PushBack( J, I, complexd(valr, 0.0));
               }
               else if( mm_is_skew(matcode) )
               {
                  coo.PushBack( J, I, complexd(-valr, 0.0));
               }
            }
         }
      }

      if (f !=stdin) fclose(f);

      return PARGEMSLR_SUCCESS;
   }
}
