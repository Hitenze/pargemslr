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
   template<typename T>
   int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const T *aa, int nrows, int ncols, char trans, const T &alpha, const T *x, const T &beta, T *y)
   {

      int      i, j, j1, j2;
      T        r, xi, *x_temp = NULL;
      T        one = 1.0;
      T        zero = 0.0;
#ifdef PARGEMSLR_OPENMP
      int      num_threads, my_thread_id;
      T        *y_temp;
#endif

      /* 1. Compute y = beta*y
       * note that when x==y and alpha != 0.0, we need to copy x
       */

      /* copy x when x==y, otherwise scale y would modify x
       * TODO: memcpy or omp parallel?
       */
      if( (x == y) && (alpha != zero) )
      {
         if (trans == 'N')
         {
            PARGEMSLR_MALLOC(x_temp, nrows, kMemoryHost, T);
            PARGEMSLR_MEMCPY(x_temp, y, nrows, kMemoryHost, kMemoryHost, T);
            x = x_temp;
         }
         else if( (trans == 'T') || (trans == 'C') )
         {
            PARGEMSLR_MALLOC(x_temp, ncols, kMemoryHost, T);
            PARGEMSLR_MEMCPY(x_temp, y, ncols, kMemoryHost, kMemoryHost, T);
            x = x_temp;
         }
         else
         {
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
      }

      /* now scale y */
      if(beta != one)
      {
         /* when beta == 1.0, y = y, do nothing */
         if(beta != zero)
         {
            /* y = beta*y */
            if (trans == 'N')
            {
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < nrows; i++)
               {
                  y[i] *= beta;
               }
            }
            else if( (trans == 'T') || (trans == 'C') )
            {
               /* if x == y need to create new x */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < ncols; i++)
               {
                  y[i] *= beta;
               }
            }
            else
            {
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
         else
         {
            /* beta == 0.0, y = 0 */
            if (trans == 'N')
            {
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < nrows; i++)
               {
                  y[i] = zero;
               }
            }
            else if( (trans == 'T') || (trans == 'C') )
            {
               /* if x == y need to create new x */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < ncols; i++)
               {
                  y[i] = zero;
               }
            }
            else
            {
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
      }


      /* 2. the matvec y = alpha*A*x + y
       * when alpha == 0 we have y = y, do nothing
       */

      if(alpha != zero)
      {
         if(alpha != one)
         {
            if (trans == 'N')
            {
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, r, j1, j2) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
               for (i = 0; i < nrows; i++)
               {
                  r = 0.0;
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++)
                  {
                     r += aa[j] * x[ja[j]];
                  }
                  y[i] += alpha*r;
               }
            }
            else if(trans == 'T')
            {
#ifdef PARGEMSLR_OPENMP
               /* create buffer for OpenMP when needed */
               num_threads = PargemslrGetOpenmpMaxNumThreads();
               if(num_threads>1)
               {
                  PARGEMSLR_CALLOC(y_temp, num_threads * ncols, kMemoryHost, T);
#pragma omp parallel private(i, j, j1, j2, my_thread_id, xi)
                  {
                     my_thread_id = PargemslrGetOpenmpThreadNum();
                     T* y_local = y_temp + my_thread_id * ncols;
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
                     /* sum to local buffer */
                     for (i = 0; i < nrows; i++)
                     {
                        xi = alpha * x[i];
                        j1 = ia[i];
                        j2 = ia[i+1];
                        for (j = j1; j < j2; j++)
                        {
                           y_local[ja[j]] += aa[j] * xi;
                        }
                     }
                     /* sumup the local buffer to y */
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
                     for(i = 0 ; i < ncols ; i ++)
                     {
                        for(j = 0 ; j < num_threads ; j ++)
                        {
                           y[i] = y[i] + y_temp[i+j*ncols];
                        }
                     }
                  }
                  /* free temp buffer after openmp finished */
                  PARGEMSLR_FREE(y_temp, kMemoryHost);
               }
               else
               {
#endif
                  for (i = 0; i < nrows; i++)
                  {
                     xi = alpha * x[i];
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++)
                     {
                        y[ja[j]] += aa[j] * xi;
                     }
                  }
#ifdef PARGEMSLR_OPENMP
               }
#endif
            }
            else if(trans == 'C')
            {
#ifdef PARGEMSLR_OPENMP
               /* create buffer for OpenMP when needed */
               num_threads = PargemslrGetOpenmpMaxNumThreads();
               if(num_threads>1)
               {
                  PARGEMSLR_CALLOC(y_temp, num_threads * ncols, kMemoryHost, T);
#pragma omp parallel private(i, j, j1, j2, my_thread_id, xi)
                  {
                     my_thread_id = PargemslrGetOpenmpThreadNum();
                     T* y_local = y_temp + my_thread_id * ncols;
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
                     /* sum to local buffer */
                     for (i = 0; i < nrows; i++)
                     {
                        xi = alpha * x[i];
                        j1 = ia[i];
                        j2 = ia[i+1];
                        for (j = j1; j < j2; j++)
                        {
                           y_local[ja[j]] += PargemslrConj(aa[j]) * xi;
                        }
                     }
                     /* sumup the local buffer to y */
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
                     for(i = 0 ; i < ncols ; i ++)
                     {
                        for(j = 0 ; j < num_threads ; j ++)
                        {
                           y[i] = y[i] + y_temp[i+j*ncols];
                        }
                     }
                  }
                  /* free temp buffer after openmp finished */
                  PARGEMSLR_FREE(y_temp, kMemoryHost);
               }
               else
               {
#endif
                  for (i = 0; i < nrows; i++)
                  {
                     xi = alpha * x[i];
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++)
                     {
                        y[ja[j]] += PargemslrConj(aa[j]) * xi;
                     }
                  }
#ifdef PARGEMSLR_OPENMP
               }
#endif
            }
            else
            {
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
         else
         {
            if (trans == 'N')
            {
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, j1, j2) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < nrows; i++)
               {
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++)
                  {
                     y[i] += aa[j] * x[ja[j]];
                  }
               }
            }
            else if(trans == 'T')
            {
#ifdef PARGEMSLR_OPENMP
               /* create buffer for OpenMP when needed */
               num_threads = PargemslrGetOpenmpMaxNumThreads();
               if(num_threads>1)
               {
                  PARGEMSLR_CALLOC(y_temp, num_threads * ncols, kMemoryHost, T);
#pragma omp parallel private(i, j, j1, j2, my_thread_id)
                  {
                     my_thread_id = PargemslrGetOpenmpThreadNum();
                     T* y_local = y_temp + my_thread_id * ncols;
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
                     /* sum to local buffer */
                     for (i = 0; i < nrows; i++)
                     {
                        j1 = ia[i];
                        j2 = ia[i+1];
                        for (j = j1; j < j2; j++)
                        {
                           y_local[ja[j]] += aa[j] * x[i];
                        }
                     }
                     /* sumup the local buffer to y */
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
                     for(i = 0 ; i < ncols ; i ++)
                     {
                        for(j = 0 ; j < num_threads ; j ++)
                        {
                           y[i] = y[i] + y_temp[i+j*ncols];
                        }
                     }
                  }
                  /* free temp buffer after openmp finished */
                  PARGEMSLR_FREE(y_temp, kMemoryHost);
               }
               else
               {
#endif
                  for (i = 0; i < nrows; i++)
                  {
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++)
                     {
                        y[ja[j]] += aa[j] * x[i];
                     }
                  }
#ifdef PARGEMSLR_OPENMP
               }
#endif
            }
            else if(trans == 'C')
            {
#ifdef PARGEMSLR_OPENMP
               /* create buffer for OpenMP when needed */
               num_threads = PargemslrGetOpenmpMaxNumThreads();
               if(num_threads>1)
               {
                  PARGEMSLR_CALLOC(y_temp, num_threads * ncols, kMemoryHost, T);
#pragma omp parallel private(i, j, j1, j2, my_thread_id)
                  {
                     my_thread_id = PargemslrGetOpenmpThreadNum();
                     T* y_local = y_temp + my_thread_id * ncols;
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
                     /* sum to local buffer */
                     for (i = 0; i < nrows; i++)
                     {
                        j1 = ia[i];
                        j2 = ia[i+1];
                        for (j = j1; j < j2; j++)
                        {
                           y_local[ja[j]] += PargemslrConj(aa[j]) * x[i];
                        }
                     }
                     /* sumup the local buffer to y */
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
                     for(i = 0 ; i < ncols ; i ++)
                     {
                        for(j = 0 ; j < num_threads ; j ++)
                        {
                           y[i] = y[i] + y_temp[i+j*ncols];
                        }
                     }
                  }
                  /* free temp buffer after openmp finished */
                  PARGEMSLR_FREE(y_temp, kMemoryHost);
               }
               else
               {
#endif
                  for (i = 0; i < nrows; i++)
                  {
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++)
                     {
                        y[ja[j]] += PargemslrConj(aa[j]) * x[i];
                     }
                  }
#ifdef PARGEMSLR_OPENMP
               }
#endif
            }
            else
            {
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
      }

      if(x_temp)
      {
         PARGEMSLR_FREE( x_temp, kMemoryHost);
      }

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const float *aa, int nrows, int ncols, char trans, const float &alpha, const float *x, const float &beta, float *y);
   template int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const double *aa, int nrows, int ncols, char trans, const double &alpha, const double *x, const double &beta, double *y);
   template int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const complexs *aa, int nrows, int ncols, char trans, const complexs &alpha, const complexs *x, const complexs &beta, complexs *y);
   template int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const complexd *aa, int nrows, int ncols, char trans, const complexd &alpha, const complexd *x, const complexd &beta, complexd *y);

   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const float *aa, int nrows, int ncols, char trans, const float &alpha, const float *x, const float &beta, float *y)
   {
#ifdef PARGEMSLR_MKL
      /* mkl matdescra, general 0-based matrix
       * for MKL, I is two vectors of size nrow, I_start and I_end, thus, can be discontinues on J
       */
      mkl_scsrmv( &trans, &nrows, &ncols, &alpha, "GXXCXX", aa, ja, ia, ia+1, x, &beta , y);
      return PARGEMSLR_SUCCESS;
#endif
      return CsrMatrixPMatVecHostTemplate(ia, ja, aa, nrows, ncols, trans, alpha, x, beta, y);
   }

   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const double *aa, int nrows, int ncols, char trans, const double &alpha, const double *x, const double &beta, double *y)
   {
#ifdef PARGEMSLR_MKL
      /* mkl matdescra, general 0-based matrix
       * for MKL, I is two vectors of size nrow, I_start and I_end, thus, can be discontinues on J
       */
      mkl_dcsrmv( &trans, &nrows, &ncols, &alpha, "GXXCXX", aa, ja, ia, ia+1, x, &beta, y);
      return PARGEMSLR_SUCCESS;
#endif
      return CsrMatrixPMatVecHostTemplate(ia, ja, aa, nrows, ncols, trans, alpha, x, beta, y);
   }

   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const complexs *aa, int nrows, int ncols, char trans, const complexs &alpha, const complexs *x, const complexs &beta, complexs *y)
   {
#ifdef PARGEMSLR_MKL
      /* mkl matdescra, general 0-based matrix
       * for MKL, I is two vectors of size nrow, I_start and I_end, thus, can be discontinues on J
       */
      mkl_ccsrmv( &trans, &nrows, &ncols, (MKL_Complex8*)&alpha, "GXXCXX", (MKL_Complex8*)aa, ja, ia, ia+1, (MKL_Complex8*)x, (MKL_Complex8*)&beta , (MKL_Complex8*)y);
      return PARGEMSLR_SUCCESS;
#endif
      return CsrMatrixPMatVecHostTemplate(ia, ja, aa, nrows, ncols, trans, alpha, x, beta, y);
   }

   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const complexd *aa, int nrows, int ncols, char trans, const complexd &alpha, const complexd *x, const complexd &beta, complexd *y)
   {
#ifdef PARGEMSLR_MKL
      /* mkl matdescra, general 0-based matrix
       * for MKL, I is two vectors of size nrow, I_start and I_end, thus, can be discontinues on J
       */
      mkl_zcsrmv( &trans, &nrows, &ncols, (MKL_Complex16*)&alpha, "GXXCXX", (MKL_Complex16*)aa, ja, ia, ia+1, (MKL_Complex16*)x, (MKL_Complex16*)&beta , (MKL_Complex16*)y);
      return PARGEMSLR_SUCCESS;
#endif
      return CsrMatrixPMatVecHostTemplate(ia, ja, aa, nrows, ncols, trans, alpha, x, beta, y);
   }

   int CsrMatrixMatVec( const CsrMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y)
   {
      int   m, n;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();

#ifdef PARGEMSLR_DEBUG
      if (trans == 'N')
      {
         PARGEMSLR_CHKERR( m != y.GetLengthLocal() );
         PARGEMSLR_CHKERR( n != x.GetLengthLocal() );
      }
      else if(trans == 'C' || trans == 'T')
      {
         PARGEMSLR_CHKERR( n != y.GetLengthLocal() );
         PARGEMSLR_CHKERR( m != x.GetLengthLocal() );
         trans = 'T';
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               PARGEMSLR_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );

               return CsrMatrixSMatVecDevice( A, trans, alpha, x, beta, y);

               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );

                     return CsrMatrixSMatVecDevice( A, trans, alpha, x, beta, y);

                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return CsrMatrixSMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     PARGEMSLR_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               PARGEMSLR_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
         CsrMatrixPMatVecHost( A.GetI(), A.GetJ(), A.GetData(), A.GetNumRowsLocal(), A.GetNumColsLocal(), trans, alpha, x.GetData(), beta, y.GetData());
      }
      else if(( trans == 'T' && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return PARGEMSLR_SUCCESS;
   }

   int CsrMatrixMatVec( const CsrMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y)
   {
      int   m, n;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();

#ifdef PARGEMSLR_DEBUG
      if (trans == 'N')
      {
         PARGEMSLR_CHKERR( m != y.GetLengthLocal() );
         PARGEMSLR_CHKERR( n != x.GetLengthLocal() );
      }
      else if(trans == 'C' || trans == 'T')
      {
         PARGEMSLR_CHKERR( n != y.GetLengthLocal() );
         PARGEMSLR_CHKERR( m != x.GetLengthLocal() );
         trans = 'T';
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               PARGEMSLR_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );

               return CsrMatrixDMatVecDevice( A, trans, alpha, x, beta, y);

               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );

                     return CsrMatrixDMatVecDevice( A, trans, alpha, x, beta, y);

                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return CsrMatrixDMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     PARGEMSLR_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               PARGEMSLR_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
         CsrMatrixPMatVecHost( A.GetI(), A.GetJ(), A.GetData(), A.GetNumRowsLocal(), A.GetNumColsLocal(), trans, alpha, x.GetData(), beta, y.GetData());

      }
      else if(( trans == 'T' && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return PARGEMSLR_SUCCESS;
   }

   int CsrMatrixMatVec( const CsrMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y)
   {
      int   m, n;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();

#ifdef PARGEMSLR_DEBUG
      if (trans == 'N')
      {
         PARGEMSLR_CHKERR( m != y.GetLengthLocal() );
         PARGEMSLR_CHKERR( n != x.GetLengthLocal() );
      }
      else if(trans == 'C' || trans == 'T')
      {
         PARGEMSLR_CHKERR( n != y.GetLengthLocal() );
         PARGEMSLR_CHKERR( m != x.GetLengthLocal() );
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               PARGEMSLR_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );

               return CsrMatrixCMatVecDevice( A, trans, alpha, x, beta, y);

               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );

                     return CsrMatrixCMatVecDevice( A, trans, alpha, x, beta, y);

                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return CsrMatrixCMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     PARGEMSLR_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               PARGEMSLR_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
         CsrMatrixPMatVecHost( A.GetI(), A.GetJ(), A.GetData(), A.GetNumRowsLocal(), A.GetNumColsLocal(), trans, alpha, x.GetData(), beta, y.GetData());

      }
      else if(((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return PARGEMSLR_SUCCESS;
   }

   int CsrMatrixMatVec( const CsrMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y)
   {
      int   m, n;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();

#ifdef PARGEMSLR_DEBUG
      if (trans == 'N')
      {
         PARGEMSLR_CHKERR( m != y.GetLengthLocal() );
         PARGEMSLR_CHKERR( n != x.GetLengthLocal() );
      }
      else if(trans == 'C' || trans == 'T')
      {
         PARGEMSLR_CHKERR( n != y.GetLengthLocal() );
         PARGEMSLR_CHKERR( m != x.GetLengthLocal() );
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               PARGEMSLR_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );

               return CsrMatrixZMatVecDevice( A, trans, alpha, x, beta, y);

               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );

                     return CsrMatrixZMatVecDevice( A, trans, alpha, x, beta, y);

                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return CsrMatrixZMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     PARGEMSLR_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               PARGEMSLR_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
         CsrMatrixPMatVecHost( A.GetI(), A.GetJ(), A.GetData(), A.GetNumRowsLocal(), A.GetNumColsLocal(), trans, alpha, x.GetData(), beta, y.GetData());

      }
      else if(((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return PARGEMSLR_SUCCESS;
   }

   template <int INIDX, int OUTIDX, typename T>
   int CsrMatrixP2CscMatrixPHost( int nrows, int ncols, bool copy_data, T* ai, int *ji, int *ii, T *ao, int *jo, int *io)
   {
      int i, j, k;
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
      for ( i = 0; i < ncols + 1; i++)
      {
         io[i] = 0;
      }

#ifdef PARGEMSLR_OPENMP
      int num_threads = PargemslrGetOpenmpMaxNumThreads();
      if(num_threads > 1)
      {
         /* create temp buffer */
         vector_int io_temp_vec, size_temp_vec;
         int *io_temp, *size_temp;
         int ncolsp1 = ncols+1;
         io_temp_vec.Setup(ncolsp1*num_threads,true);
         size_temp_vec.Setup(ncolsp1*num_threads,true);
         io_temp = io_temp_vec.GetData();
         size_temp = size_temp_vec.GetData();
#pragma omp parallel private(i, j, k)
         {
            int idx1, idx2;
            int my_thread_id = PargemslrGetOpenmpThreadNum();
            int *io_local = io_temp + ncolsp1 * my_thread_id;
            int *size_local = size_temp + ncolsp1 * my_thread_id;

            /* get nnz of each columns of A, store in io
             * in this step, io is parallel, we have
             *     thread1        thread2      ...     threadn
             * | size_local_1 | size_local_2 | ... | size_local_n  |
             * size_local
             */
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
            for ( i = 0; i < nrows; i++)
            {
               for ( k = ii[i]; k < ii[i+1]; k++)
               {
                  size_local[ji[k]-INIDX] ++;
               }
            }

            /* copy to size_temp to io_temp and get
             * the accumulate size
             * memcpy is not thread safe
             */
            for(i = 0 ; i < ncols ; i ++)
            {
               io_local[i+1] = size_local[i] + io_local[i];
            }

            /* accumulate to result to the io vector
             * omp barrier is required, since we
             * need to touch io_temp from other threads
             * note that we don't need io[ncol]
             */
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
            for ( i = 1; i <= ncols; i++)
            {
               for(j = 0 ; j < num_threads ; j ++)
               {
                  io[i] += io_temp[i+j*ncolsp1];
               }
            }

            /* now start to copy data in
             * also will touch data from other refion, add #pragma omp barrier
             * we want to have io_local now store the shift value
             */
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
            for(i = 0 ; i < ncols ; i ++)
            {
               io_temp[i] = io[i];
               for(j = 1 ; j < num_threads ; j ++)
               {
                  idx2 = i+j*ncolsp1;
                  idx1 = idx2 - ncolsp1;
                  io_temp[idx2] = io_temp[idx1] + size_temp[idx1];
               }
            }

            /* now copy data
             */
            if(copy_data)
            {
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
               for ( i = 0; i < nrows; i++)
               {
                  for ( k = ii[i]; k < ii[i+1]; k++)
                  {
                     j = ji[k]-INIDX;
                     ao[io_local[j]] = ai[k];
                     jo[io_local[j]++] = i + OUTIDX;
                  }
               }
            }
            else
            {
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
               for ( i = 0; i < nrows; i++)
               {
                  for ( k = ii[i]; k < ii[i+1]; k++)
                  {
                     j = ji[k]-INIDX;
                     jo[io_local[j]++] = i + OUTIDX;
                  }
               }
            }
         }/* openmp ends */

         io_temp_vec.Clear();
         size_temp_vec.Clear();

         /*---- reshift iao and leave
          * no need of barrier here, io is not used
          */
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
         for(i = 1 ; i < ncols ; i ++)
         {
            io[i] += OUTIDX;
         }
         io[0] = OUTIDX;

         /* finally we need to sort local rows */
         /* csr matrix */
         if(copy_data)
         {
#pragma omp parallel private(i)
            {
               int length;
               vector_int cols, order;
               SequentialVectorClass<T> vals;
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
               for(i = 0 ; i < ncols ; i ++)
               {
                  length = io[i+1]-io[i];
                  cols.SetupPtr( jo + io[i], length, kMemoryHost);

                  /* sort in ascending order */
                  cols.Sort( order, true, false);

                  vals.SetupPtr( ao + io[i], length, kMemoryHost);

                  /* apply the permutation */
                  cols.Perm(order);
                  vals.Perm(order);

                  vals.Clear();
                  order.Clear();

                  cols.Clear();
               }
            }/* openmp ends */
         }
         else
         {
#pragma omp parallel private(i)
            {
               int length;
               vector_int cols;
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
               for(i = 0 ; i < ncols ; i ++)
               {
                  length = io[i+1]-io[i];

                  cols.SetupPtr( jo + io[i], length, kMemoryHost);

                  /* in this case just sort cols in ascending order */
                  cols.Sort(true);

                  cols.Clear();
               }
            }/* openmp ends */
         }
      }
      else
      {
#endif
         /* get nnz of each columns of A, store in io
          * note that the input might be 1-based, shift by INIDX
          * Note that for 1-based indexing, io should still start from
          * io[0], just the value should be 1
          */
         for ( i = 0; i < nrows; i++)
         {
            for ( k = ii[i]; k < ii[i+1]; k++)
            {
               /* ji[k] has shift INIDX */
               io[ji[k]+1-INIDX] ++;
            }
         }
         /* accumulate to get the I vector */
         for ( i = 0; i < ncols; i++)
         {
            io[i+1] += io[i];
         }
         /* copy J and A
          * j is the column in in-based, we store it in out-based
          */
         for ( i = 0; i < nrows; i++)
         {
            for ( k = ii[i]; k < ii[i+1]; k++)
            {
               /* get location in io */
               j = ji[k]-INIDX;
               if (copy_data)
               {
                  ao[io[j]] = ai[k];
               }
               jo[io[j]++] = i + OUTIDX;
            }
         }
         /* shift io to OUTIDX-based */
         for (i = ncols; i > 0; i--)
         {
            io[i] = io[i-1] + OUTIDX;
         }
         io[0] = OUTIDX;
#ifdef PARGEMSLR_OPENMP
      }
#endif

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixP2CscMatrixPHost<0,0>( int nrows, int ncols, bool copy_data, float* ai, int *ji, int *ii, float *ao, int *jo, int *io);
   template int CsrMatrixP2CscMatrixPHost<0,0>( int nrows, int ncols, bool copy_data, double* ai, int *ji, int *ii, double *ao, int *jo, int *io);
   template int CsrMatrixP2CscMatrixPHost<0,0>( int nrows, int ncols, bool copy_data, complexs* ai, int *ji, int *ii, complexs *ao, int *jo, int *io);
   template int CsrMatrixP2CscMatrixPHost<0,0>( int nrows, int ncols, bool copy_data, complexd* ai, int *ji, int *ii, complexd *ao, int *jo, int *io);

   template <int INIDX, int OUTIDX, typename T>
   int CooMatrixP2CsrMatrixPHost( int nrows, int ncols, int nnz, T* ai, int *ji, int *ii, T *ao, int *jo, int *io)
   {
      int   i, j;
      T     val;
      int   idx_shift = OUTIDX - INIDX;

#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
      for ( i = 0; i <= nrows; i++)
      {
         io[i] = 0;
      }

#ifdef PARGEMSLR_OPENMP
      int num_threads = PargemslrGetOpenmpMaxNumThreads();
      if(num_threads > 1)
      {
         vector_int io_temp_vec, size_temp_vec;
         int *io_temp, *size_temp;
         int nrowsp1 = nrows+1;
         io_temp_vec.Setup(nrowsp1*num_threads,true);
         size_temp_vec.Setup(nrowsp1*num_threads,true);
         io_temp = io_temp_vec.GetData();
         size_temp = size_temp_vec.GetData();
#pragma omp parallel private(i, j)
         {
            int my_thread_id = PargemslrGetOpenmpThreadNum();
            int *io_local = io_temp + nrowsp1 * my_thread_id;
            int *size_local = size_temp + nrowsp1 * my_thread_id;
            int idx1, idx2;

            /* this step is to prepare the io size */
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
            for ( i = 0; i < nnz; i++)
            {
               size_local[ii[i] - INIDX]++;
            }

            /* copy to size_temp to io_temp and get
             * the accumulate size
             * memcpy is not thread safe
             */
            for(i = 0 ; i < nrows ; i ++)
            {
               io_local[i+1] = size_local[i] + io_local[i];
            }

            /* accumulate to result to the io vector
             * omp barrier is required, since we
             * need to touch io_temp from other threads
             * note that we don't need io[ncol]
             */
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
            for ( i = 1; i <= nrows; i++)
            {
               for(j = 0 ; j < num_threads ; j ++)
               {
                  io[i] += io_temp[i+j*nrowsp1];
               }
            }

            /* now start to copy data in
             * also will touch data from other refion, add #pragma omp barrier
             * we want to have io_local now store the shift value
             */
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
            for(i = 0 ; i < nrows ; i ++)
            {
               io_temp[i] = io[i];
               for(j = 1 ; j < num_threads ; j ++)
               {
                  idx2 = i+j*nrowsp1;
                  idx1 = idx2 - nrowsp1;
                  io_temp[idx2] = io_temp[idx1] + size_temp[idx1];
               }
            }

            /* now copy data */
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_STATIC
            for ( i = 0; i < nnz; i++)
            {
               j = ii[i] - INIDX;
               val = ai[i];
               /* io[j] is now 0-based */
               ao[io_local[j]] = val;
               jo[io_local[j]++] = ji[i] + idx_shift;
            }

         }/* openmp ends */

         io_temp_vec.Clear();
         size_temp_vec.Clear();

#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
         for ( i = nrows; i > 0; i--)
         {
            io[i] += OUTIDX;
         }
         io[0] = OUTIDX;
      }
      else
      {
#endif
         /* this step is to prepare the io size */
         for ( i = 0; i < nnz; i++)
         {
            io[ii[i]-INIDX+1]++;
         }
         for ( i = 0; i < nrows; i++)
         {
            io[i+1] += io[i];
         }
         /* note that io is now 0-based */
         for ( i = 0; i < nnz; i++)
         {
            j = ii[i] - INIDX;
            val = ai[i];
            /* io[j] is now 0-based */
            ao[io[j]] = val;
            jo[io[j]++] = ji[i] + idx_shift;
         }
         for ( i = nrows; i > 0; i--)
         {
            io[i] = io[i-1] + OUTIDX;
         }
         io[0] = OUTIDX;
#ifdef PARGEMSLR_OPENMP
      }
#endif
      return PARGEMSLR_SUCCESS;
   }
   template int CooMatrixP2CsrMatrixPHost<0,0>( int nrows, int ncols, int nnz, float* ai, int *ji, int *ii, float *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<0,0>( int nrows, int ncols, int nnz, double* ai, int *ji, int *ii, double *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<0,0>( int nrows, int ncols, int nnz, complexs* ai, int *ji, int *ii, complexs *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<0,0>( int nrows, int ncols, int nnz, complexd* ai, int *ji, int *ii, complexd *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<1,0>( int nrows, int ncols, int nnz, float* ai, int *ji, int *ii, float *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<1,0>( int nrows, int ncols, int nnz, double* ai, int *ji, int *ii, double *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<1,0>( int nrows, int ncols, int nnz, complexs* ai, int *ji, int *ii, complexs *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<1,0>( int nrows, int ncols, int nnz, complexd* ai, int *ji, int *ii, complexd *ao, int *jo, int *io);

   template <typename T>
   int CsrMatrixTransposeHost( CsrMatrixClass<T> &A, CsrMatrixClass<T> &AT)
   {
      int nrows, ncols, nnz;

      if( A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Csr matrix transpose only works for the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();
      nnz = A.GetNumNonzeros();

      /* transpose switch rows and cols */
      AT.Setup( ncols, nrows, nnz, A.IsHoldingData(), false);

      CsrMatrixP2CscMatrixPHost<0,0>( nrows, ncols, A.IsHoldingData(), A.GetData(), A.GetJ(), A.GetI(), AT.GetData(), AT.GetJ(), AT.GetI());

      AT.IsCsr() = A.IsCsr();

      /* by the algorithm the col/row of the new csr/csc should be sorted */
      AT.IsRowSorted() = true;

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixTransposeHost( CsrMatrixClass<float> &A, CsrMatrixClass<float> &AT);
   template int CsrMatrixTransposeHost( CsrMatrixClass<double> &A, CsrMatrixClass<double> &AT);
   template int CsrMatrixTransposeHost( CsrMatrixClass<complexs> &A, CsrMatrixClass<complexs> &AT);
   template int CsrMatrixTransposeHost( CsrMatrixClass<complexd> &A, CsrMatrixClass<complexd> &AT);

   template <typename T>
   int CsrMatrixAddHost( CsrMatrixClass<T> &A, CsrMatrixClass<T> &B, CsrMatrixClass<T> &C)
   {
      /* TODO: add OpenMP support */
      PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal() || A.GetNumColsLocal() != B.GetNumColsLocal());
      PARGEMSLR_CHKERR(A.IsCsr() != B.IsCsr());

      if( A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("csr matrix add only works for the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      if( B.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("csr matrix add only works for the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      //int *iw;
      int   nnzC, i, j, col, pos, nrows, ncols, nnzA, nnzB;
      bool  csr;
      int   *A_i, *A_j, *B_i, *B_j, *C_i, *C_j;
      T     *A_data, *B_data, *C_data;

      csr = A.IsCsr();

      if( !csr )
      {
         PARGEMSLR_WARNING("in the current version, csradd only works for csr matrix, turn A and B into csr.");
         A.Transpose();
         B.Transpose();
      }

      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();
      nnzA = A.GetNumNonzeros();
      nnzB = B.GetNumNonzeros();

      /* reserve the max size */
      C.Setup( nrows, ncols, PargemslrMin( PargemslrMax(nrows * ncols, INT_MAX), nnzA + nnzB));

      A_i = A.GetI();
      A_j = A.GetJ();
      A_data = A.GetData();
      B_i = B.GetI();
      B_j = B.GetJ();
      B_data = B.GetData();
      C_i = C.GetI();
      C_j = C.GetJ();
      C_data = C.GetData();

      //Malloc(iw, A._ncol, int);
      IntVectorClass<int> iw;

      iw.Setup( ncols);
      iw.Fill(-1);

      nnzC = 0;

      C_i[0] = nnzC;
      for (i = 0; i < nrows; i++)
      {
         // A
         for (j = A_i[i]; j < A_i[i+1]; j++)
         {
            col = A_j[j];
            C_j[nnzC] = col;
            C_data[nnzC] = A_data[j];
            iw[col] = nnzC++;
         }
         // B
         for (j = B_i[i]; j < B_i[i+1]; j++)
         {
            col = B_j[j];
            pos = iw[col];
            if (-1 == pos)
            {
               C_j[nnzC] = col;
               C_data[nnzC] = B_data[j];
               iw[col] = nnzC++;
            }
            else
            {

               PARGEMSLR_CHKERR(C_j[pos] != col);

               C_data[pos] += B_data[j];
            }
         }
         C_i[i+1] = nnzC;
         // reset iw
         for (j = C_i[i]; j < C_i[i+1]; j++)
         {
            iw[C_j[j]] = -1;
         }
      }

      /* update the nnz */
      C.SetNumNonzeros();

      if( A.IsRowSorted() || B.IsRowSorted() )
      {
         C.SortRow();
      }

      iw.Clear();

      if( !csr )
      {
         A.Transpose();
         B.Transpose();
      }

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixAddHost( CsrMatrixClass<float> &A, CsrMatrixClass<float> &B, CsrMatrixClass<float> &C);
   template int CsrMatrixAddHost( CsrMatrixClass<double> &A, CsrMatrixClass<double> &B, CsrMatrixClass<double> &C);
   template int CsrMatrixAddHost( CsrMatrixClass<complexs> &A, CsrMatrixClass<complexs> &B, CsrMatrixClass<complexs> &C);
   template int CsrMatrixAddHost( CsrMatrixClass<complexd> &A, CsrMatrixClass<complexd> &B, CsrMatrixClass<complexd> &C);

   template <typename T>
   int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<T> &A, ParallelCsrMatrixClass<T> &AT)
   {

      if( A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Parallel Csr matrix transpose only works for the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      /* MPI info */
      int np, myid;
      MPI_Comm comm;

      A.GetMpiInfo(np, myid, comm);

      /* setup matvec to get communication helper */
      A.SetupMatvec();

      CsrMatrixClass<T> &A_diag = A.GetDiagMat();
      CsrMatrixClass<T> &A_offd = A.GetOffdMat();

      CsrMatrixClass<T> AT_diag;
      CsrMatrixClass<T> AT_offd;
      CooMatrixClass<T> AT_offd_coo;

      CsrMatrixTransposeHost(A_diag, AT_diag);
      CsrMatrixTransposeHost(A_offd, AT_offd);

      int   *AT_offd_i = AT_offd.GetI();
      int   *AT_offd_j = AT_offd.GetJ();
      T     *AT_offd_data = AT_offd.GetData();

      /* setup the structure of AT */
      AT.Setup(A.GetNumColsLocal(), A.GetColStartGlobal(), A.GetNumColsGlobal(), A.GetNumRowsLocal(), A.GetRowStartGlobal(), A.GetNumRowsGlobal(), A);

      /* setup the diagonal first */
      AT.GetDiagMat() = std::move(AT_diag);

      vector_long &offd_map_v = AT.GetOffdMap();

      /* now, we need to send data to targer processor
       * A._comm_helper._recv_from_v: list of nodes recv data from
       * A._comm_helper._recv_idx_v2: list of solc recv from each proc
       * To send the transpose, send the size of each row first
       */
      int nsends, nsendi, nsendi2, nrecvs, nrecvi, nrecvi2, i, j, j1, j2, k, idx, idx2, n_local, n_offd;
      vector_int marker;
      std::vector<vector_int> send_size_v2, recv_size_v2;
      vector_long send_se_v;
      std::vector<vector_long> recv_se_v2;
      std::vector<vector_int> send_j_v2, recv_j_v2;
      std::vector<SequentialVectorClass<T> > send_data_v2, recv_data_v2;
      vector<MPI_Request> requests_v;

      nsends = A._comm_helper._recv_from_v.GetLengthLocal();
      nrecvs = A._comm_helper._send_to_v.GetLengthLocal();
      send_size_v2.resize(nsends);
      send_j_v2.resize(nsends);
      send_data_v2.resize(nsends);

      recv_size_v2.resize(nrecvs);
      recv_j_v2.resize(nrecvs);
      recv_data_v2.resize(nrecvs);

      requests_v.resize(nsends + nrecvs);

      send_se_v.Setup(2);
      send_se_v[0] = A.GetRowStartGlobal();
      send_se_v[1] = send_se_v[0] + A.GetNumRowsLocal();
      recv_se_v2.resize(nrecvs);

      /* get the size of each row */
      for(i = 0 ; i < nsends ; i ++)
      {
         nsendi = A._comm_helper._recv_idx_v2[i].GetLengthLocal();
         /* we use the last index to store the total length */
         send_size_v2[i].Setup(nsendi+1);

         send_size_v2[i][nsendi] = 0;
         for(j = 0 ; j < nsendi ; j ++)
         {
            k = A._comm_helper._recv_idx_v2[i][j];
            send_size_v2[i][j] = AT_offd_i[k+1] - AT_offd_i[k];
            send_size_v2[i][nsendi] += send_size_v2[i][j];
         }

         nsendi2 = send_size_v2[i][nsendi];
         send_j_v2[i].Setup(nsendi2);
         send_data_v2[i].Setup(nsendi2);

         idx = 0;
         for(j = 0 ; j < nsendi ; j ++)
         {
            k = A._comm_helper._recv_idx_v2[i][j];
            j1 = AT_offd_i[k];
            j2 = AT_offd_i[k+1];
            for(k = j1 ; k < j2 ; k ++)
            {
               send_j_v2[i][idx] = AT_offd_j[k];
               send_data_v2[i][idx] = AT_offd_data[k];
               idx ++;
            }
         }
      }

      /* send target row size */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         nsendi = A._comm_helper._recv_idx_v2[i].GetLengthLocal();
         PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_size_v2[i].GetData(), nsendi+1,
               A._comm_helper._recv_from_v[i], 0, comm, &(requests_v[j++])) );
      }

      for(i = 0 ; i < nrecvs ; i ++)
      {
         nrecvi = A._comm_helper._send_idx_v2[i].GetLengthLocal();
         recv_size_v2[i].Setup(nrecvi+1);

         PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_size_v2[i].GetData(), nrecvi+1,
               A._comm_helper._send_to_v[i], 0, comm, &(requests_v[j++])) );
      }

      PARGEMSLR_MPI_CALL(MPI_Waitall( nsends+nrecvs, requests_v.data(), MPI_STATUSES_IGNORE));

      //recv_size_v2[0].Plot(0,0,6);

      /* The data str should be ready now
       * start to create buffer to recv income data
       */

      /* send idx first */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         nsendi = A._comm_helper._recv_idx_v2[i].GetLengthLocal();
         nsendi2 = send_size_v2[i][nsendi];
         PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_j_v2[i].GetData(), nsendi2,
               A._comm_helper._recv_from_v[i], 0, comm, &(requests_v[j++])) );
      }

      n_offd = 0;
      for(i = 0 ; i < nrecvs ; i ++)
      {
         nrecvi = A._comm_helper._send_idx_v2[i].GetLengthLocal();
         nrecvi2 = recv_size_v2[i][nrecvi];
         recv_j_v2[i].Setup(nrecvi2);
         n_offd += nrecvi2;

         PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_j_v2[i].GetData(), nrecvi2,
               A._comm_helper._send_to_v[i], 0, comm, &(requests_v[j++])) );
      }

      PARGEMSLR_MPI_CALL(MPI_Waitall( nsends+nrecvs, requests_v.data(), MPI_STATUSES_IGNORE));

      //recv_j_v2[0].Plot(0,0,6);

      /* now send data */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         nsendi = A._comm_helper._recv_idx_v2[i].GetLengthLocal();
         nsendi2 = send_size_v2[i][nsendi];
         PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_data_v2[i].GetData(), nsendi2,
               A._comm_helper._recv_from_v[i], 0, comm, &(requests_v[j++])) );
      }

      for(i = 0 ; i < nrecvs ; i ++)
      {
         nrecvi = A._comm_helper._send_idx_v2[i].GetLengthLocal();
         nrecvi2 = recv_size_v2[i][nrecvi];
         recv_data_v2[i].Setup(nrecvi2);

         PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_data_v2[i].GetData(), nrecvi2,
               A._comm_helper._send_to_v[i], 0, comm, &(requests_v[j++])) );
      }

      PARGEMSLR_MPI_CALL(MPI_Waitall( nsends+nrecvs, requests_v.data(), MPI_STATUSES_IGNORE));

      //recv_data_v2[0].Plot(0,0,6);

      /* now send start/end index of new cols */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_se_v.GetData(), 2,
               A._comm_helper._recv_from_v[i], 0, comm, &(requests_v[j++])) );
      }

      for(i = 0 ; i < nrecvs ; i ++)
      {
         recv_se_v2[i].Setup(2);

         PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_se_v2[i].GetData(), 2,
               A._comm_helper._send_to_v[i], 0, comm, &(requests_v[j++])) );
      }

      PARGEMSLR_MPI_CALL(MPI_Waitall( nsends+nrecvs, requests_v.data(), MPI_STATUSES_IGNORE));

      //recv_data_v2[0].Plot(0,0,6);

      /* Now, each proc has the external data
       * We can start putting data in
       *
       * |---------|---------|          |---------|
       * | x x o o | x o x o |          | x o o o |
       * | x x x o | o x o x |          | o x o o |
       * | o x x x | o o x o |          | x o x o |
       * | o o x x | o o o x |          | o x o x |
       * |---------|---------| => offd  |---------| => send to p1, need to build offd_map first.
       * | x o x o | x x o o |
       * | o x o x | x x x o |
       * | o o x o | o x x x |
       * | o o o x | o o x x |
       * |---------|---------|
       *
       * we do it section by section.
       *
       */

      /* the coo matrix of offd */
      AT_offd_coo.Setup( A.GetNumColsLocal(), INT_MAX, n_offd);

      /* loop through all recv parts */
      idx = 0;
      for(i = 0 ; i < nrecvs ; i ++)
      {
         /* nrecvi is the number of rows in this section */
         nrecvi = A._comm_helper._send_idx_v2[i].GetLengthLocal();

         /* n_local is the number of local columns in this proc */
         n_local = recv_se_v2[i][1] - recv_se_v2[i][0];

         /* create the helper array of size n_local */
         marker.Setup(n_local);
         marker.Fill(-1);

         /* now loop through all rows
          * nrecvi2 is the size of local offd matrix
          */
         nrecvi2 = recv_size_v2[i][nrecvi];

         for(j = 0 ; j < nrecvi2 ; j ++)
         {
            k = recv_j_v2[i][j];
            if(marker[k] < 0)
            {
               /* this is a new col, mark it */
               marker[k] = 1;
            }
         }

         /* now update offd_map_v */
         for(j = 0 ; j < n_local ; j ++)
         {
            if(marker[j] > 0)
            {
               /* got an offd map */
               marker[j] = idx;
               offd_map_v.PushBack(recv_se_v2[i][0]+j);
               idx ++;
            }
         }

         /* now insert value */
         idx2 = 0;
         for(j = 0 ; j < nrecvi ; j ++)
         {
            /* j1 is the new row idx */
            j1 = A._comm_helper._send_idx_v2[i][j];

            /* j2 is the number of data in this row */
            j2 = recv_size_v2[i][j];
            for(k = 0 ; k < j2 ; k ++)
            {
               AT_offd_coo.PushBack( j1, marker[recv_j_v2[i][idx2]], recv_data_v2[i][idx2]);
               idx2 ++;
            }
         }
      }

      /* done */
      AT_offd_coo.ToCsr(kMemoryHost, AT.GetOffdMat());

      /* deallocate */

      AT_diag.Clear();
      AT_offd.Clear();
      AT_offd_coo.Clear();

      marker.Clear();

      for(i = 0 ; i < nrecvs ; i ++)
      {
         recv_se_v2[i].Clear();
         recv_size_v2[i].Clear();
         recv_j_v2[i].Clear();
         recv_data_v2[i].Clear();
      }
      std::vector<vector_int>().swap(recv_size_v2);
      std::vector<vector_long>().swap(recv_se_v2);
      std::vector<vector_int>().swap(recv_j_v2);
      std::vector<SequentialVectorClass<T> >().swap(recv_data_v2);

      send_se_v.Clear();
      for(i = 0 ; i < nsends ; i ++)
      {
         send_size_v2[i].Clear();
         send_j_v2[i].Clear();
         send_data_v2[i].Clear();
      }
      std::vector<vector_int>().swap(send_size_v2);
      std::vector<vector_int>().swap(send_j_v2);
      std::vector<SequentialVectorClass<T> >().swap(send_data_v2);

      vector<MPI_Request>().swap(requests_v);

      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<float> &A, ParallelCsrMatrixClass<float> &AT);
   template int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<double> &A, ParallelCsrMatrixClass<double> &AT);
   template int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<complexs> &A, ParallelCsrMatrixClass<complexs> &AT);
   template int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<complexd> &A, ParallelCsrMatrixClass<complexd> &AT);

   template <typename T>
   int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<T> &A, ParallelCsrMatrixClass<T> &B, ParallelCsrMatrixClass<T> &C)
   {
      /* TODO: add OpenMP support */

      if( A.GetDataLocation() == kMemoryDevice || B.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Parallel Csr matrix add only works for the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      /* size should match */
      PARGEMSLR_CHKERR(A.GetNumRowsGlobal() != B.GetNumRowsGlobal());
      PARGEMSLR_CHKERR(A.GetNumColsGlobal() != B.GetNumColsGlobal());
      PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
      PARGEMSLR_CHKERR(A.GetNumColsLocal() != B.GetNumColsLocal());
      PARGEMSLR_CHKERR(A.GetRowStartGlobal() != B.GetRowStartGlobal());
      PARGEMSLR_CHKERR(A.GetColStartGlobal() != B.GetColStartGlobal());

      /* MPI info */
      int np, myid;
      MPI_Comm comm;

      A.GetMpiInfo(np, myid, comm);

      /* setup C */
      C.Setup(A.GetNumRowsLocal(), A.GetRowStartGlobal(), A.GetNumRowsGlobal(), A.GetNumColsLocal(), A.GetColStartGlobal(), A.GetNumColsGlobal(), A);

      /* add diagonal */
      CsrMatrixAddHost(A.GetDiagMat(), B.GetDiagMat(), C.GetDiagMat());

      int A_n_offd, B_n_offd, C_n_offd;

      /* now, add offdiagonal
       * need to build the offdiagonal map first
       */
      int i, j, j1, j2, idx, nnzA, nnzB, nnzC, n_local, col, pos;
      vector_int order, B_map, iw;
      vector_long A_offd_map_v_sorted;

      n_local = A.GetNumRowsLocal();

      CsrMatrixClass<T> &A_offd = A.GetOffdMat();
      CsrMatrixClass<T> &B_offd = B.GetOffdMat();
      CsrMatrixClass<T> &C_offd = C.GetOffdMat();

      vector_long &A_offd_map_v = A.GetOffdMap();
      vector_long &B_offd_map_v = B.GetOffdMap();
      vector_long &C_offd_map_v = C.GetOffdMap();

      A_n_offd = A_offd_map_v.GetLengthLocal();
      B_n_offd = B_offd_map_v.GetLengthLocal();

      /* merge them together */

      /* 1st step, sort A */
      A_offd_map_v.Sort(order, true, false);
      A_offd_map_v_sorted.Setup(A_n_offd);
      order.GatherPerm(A_offd_map_v, A_offd_map_v_sorted);

      /* 2nd step, get map from B to new C_offd_map */
      C_offd_map_v = A_offd_map_v;
      B_map.Setup(B_n_offd);
      for(i = 0 ; i < B_n_offd ; i ++)
      {
        if(A_offd_map_v_sorted.BinarySearch(B_offd_map_v[i], idx, true) < 0)
        {
           /* this is a new column */
           B_map[i]  = C_offd_map_v.GetLengthLocal();
           C_offd_map_v.PushBack(B_offd_map_v[i]);
        }
        else
        {
           /* this is a old column */
           B_map[i] = idx;
        }
      }

      nnzA = A_offd.GetNumNonzeros();
      nnzB = B_offd.GetNumNonzeros();

      C_n_offd = C_offd_map_v.GetLengthLocal();

      /* now, start adding those two offdiagonal matrices */
      C_offd.Setup( n_local, C_n_offd, PargemslrMin( PargemslrMax(n_local * C_n_offd, INT_MAX), nnzA + nnzB));
      iw.Setup(C_n_offd);
      iw.Fill(-1);

      int *A_i = A_offd.GetI();
      int *B_i = B_offd.GetI();
      int *C_i = C_offd.GetI();
      int *A_j = A_offd.GetJ();
      int *B_j = B_offd.GetJ();
      int *C_j = C_offd.GetJ();
      T *A_data = A_offd.GetData();
      T *B_data = B_offd.GetData();
      T *C_data = C_offd.GetData();

      nnzC = 0;
      C_i[0] = nnzC;
      for (i = 0; i < n_local; i++)
      {
         // A
         j1 = A_i[i];
         j2 = A_i[i+1];
         for (j = j1; j < j2; j++)
         {
            col = A_j[j];
            C_j[nnzC] = col;
            C_data[nnzC] = A_data[j];
            iw[col] = nnzC++;
         }
         // B
         j1 = B_i[i];
         j2 = B_i[i+1];
         for (j = j1; j < j2; j++)
         {
            col = B_map[B_j[j]];
            pos = iw[col];
            if (-1 == pos)
            {
               C_j[nnzC] = col;
               C_data[nnzC] = B_data[j];
               iw[col] = nnzC++;
            }
            else
            {
               PARGEMSLR_CHKERR(C_j[pos] != col);
               C_data[pos] += B_data[j];
            }
         }
         C_i[i+1] = nnzC;
         // reset iw
         for (j = C_i[i]; j < C_i[i+1]; j++)
         {
            iw[C_j[j]] = -1;
         }
      }

      /* update the nnz */
      C_offd.SetNumNonzeros();

      /* deallocate */
      order.Clear();
      B_map.Clear();
      iw.Clear();
      A_offd_map_v_sorted.Clear();

      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<float> &A, ParallelCsrMatrixClass<float> &B, ParallelCsrMatrixClass<float> &C);
   template int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<double> &A, ParallelCsrMatrixClass<double> &B, ParallelCsrMatrixClass<double> &C);
   template int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<complexs> &A, ParallelCsrMatrixClass<complexs> &B, ParallelCsrMatrixClass<complexs> &C);
   template int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<complexd> &A, ParallelCsrMatrixClass<complexd> &B, ParallelCsrMatrixClass<complexd> &C);

   template <typename T>
   int CsrMatrixSortRow( CsrMatrixClass<T> &A)
   {
      if(A.IsRowSorted() || A.GetNumNonzeros() == 0)
      {
         /* if already sorted, do nothing */
         return PARGEMSLR_SUCCESS;
      }

      int                        i, length;
      int                        ncols, nrows;
      int                        *a_i, *a_j;
      T                          *a_data = NULL;
      int                        location;
      bool                       hold_data;

      vector_int                 cols, ord;
      SequentialVectorClass<T>   vals;

#ifdef PARGEMSLR_CUDA
      if(A.GetDataLocation() == kMemoryDevice || A.GetDataLocation() == kMemoryUnified)
      {
         /* prefer to use the GPU for unified memory */
         CsrMatrixSortRowDevice(A);

         return PARGEMSLR_SUCCESS;
      }
#endif

      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();

      hold_data = A.IsHoldingData();
      location = A.GetDataLocation();

      a_i = A.GetI();
      a_j = A.GetJ();

      if(hold_data)
      {
         a_data = A.GetData();
      }

      /* sort row index in ascending order */
      if(A.IsCsr())
      {
         /* csr matrix */
#ifdef PARGEMSLR_OPENMP
/* use dynamic here since we don't know the size of each row in advance */
#pragma omp parallel for private(i, length, cols, vals, ord) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
         for(i = 0 ; i < nrows ; i ++)
         {
            length = a_i[i+1]-a_i[i];

            cols.SetupPtr( a_j + a_i[i], length, location);

            if(hold_data)
            {
               /* sort in ascending order */
               cols.Sort( ord, true, false);

               vals.SetupPtr( a_data + a_i[i], length, location);

               /* apply the permutation */
               cols.Perm(ord);
               vals.Perm(ord);

               vals.Clear();
               ord.Clear();

            }
            else
            {
               /* in this case just sort cols in ascending order */
               cols.Sort(true);
            }

            cols.Clear();
         }
      }
      else
      {
         /* csc matrix */
#ifdef PARGEMSLR_OPENMP
/* use dynamic here since we don't know the size of each col in advance */
#pragma omp parallel for private(i, length, cols, vals, ord) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
         for(i = 0 ; i < ncols ; i ++)
         {
            length = a_i[i+1]-a_i[i];

            cols.SetupPtr( a_j + a_i[i], length, location);

            if(hold_data)
            {
               /* sort in ascending order */
               cols.Sort( ord, true, false);

               vals.SetupPtr( a_data + a_i[i], length, location);

               /* apply the permutation */
               cols.Perm(ord);
               vals.Perm(ord);

               vals.Clear();
               ord.Clear();

            }
            else
            {
               /* in this case just sort cols in ascending order */
               cols.Sort(true);
            }

            cols.Clear();
         }
      }

      A.IsRowSorted() = true;

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixSortRow( CsrMatrixClass<float> &A);
   template int CsrMatrixSortRow( CsrMatrixClass<double> &A);
   template int CsrMatrixSortRow( CsrMatrixClass<complexs> &A);
   template int CsrMatrixSortRow( CsrMatrixClass<complexd> &A);
}
