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
   PrecisionEnum GetMatrixPrecision(const MatrixClass<int> &mat)
   {
      return kInt;
   }

   PrecisionEnum GetMatrixPrecision(const MatrixClass<long int> &mat)
   {
      return kLongInt;
   }

   PrecisionEnum GetMatrixPrecision(const MatrixClass<float> &mat)
   {
      return kSingleReal;
   }

   PrecisionEnum GetMatrixPrecision(const MatrixClass<double> &mat)
   {
      return kDoubleReal;
   }

   PrecisionEnum GetMatrixPrecision(const MatrixClass<complexs> &mat)
   {
      return kSingleComplex;
   }

   PrecisionEnum GetMatrixPrecision(const MatrixClass<complexd> &mat)
   {
      return kDoubleComplex;
   }

   PrecisionEnum GetMatrixPPrecision(const MatrixClass<int> *mat)
   {
      return kInt;
   }

   PrecisionEnum GetMatrixPPrecision(const MatrixClass<long int> *mat)
   {
      return kLongInt;
   }

   PrecisionEnum GetMatrixPPrecision(const MatrixClass<float> *mat)
   {
      return kSingleReal;
   }

   PrecisionEnum GetMatrixPPrecision(const MatrixClass<double> *mat)
   {
      return kDoubleReal;
   }

   PrecisionEnum GetMatrixPPrecision(const MatrixClass<complexs> *mat)
   {
      return kSingleComplex;
   }

   PrecisionEnum GetMatrixPPrecision(const MatrixClass<complexd> *mat)
   {
      return kDoubleComplex;
   }

   template <typename T>
   int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const T &alpha, const T *aa, int ldim, const T *x, const T &beta, T *y)
   {

      int      i, j;
      T        *x_temp = NULL;
      T        one = 1.0;
      T        zero = 0.0;
      const T  *a_temp;

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
            if(x == y)
            {
               PARGEMSLR_MALLOC(x_temp, nrows, kMemoryHost, T);
               PARGEMSLR_MEMCPY(x_temp, y, nrows, kMemoryHost, kMemoryHost, T);
               x = x_temp;
            }
         }
         else if( (trans == 'T') || (trans == 'C') )
         {
            if(x == y)
            {
               PARGEMSLR_MALLOC(x_temp, ncols, kMemoryHost, T);
               PARGEMSLR_MEMCPY(x_temp, y, ncols, kMemoryHost, kMemoryHost, T);
               x = x_temp;
            }
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
#pragma omp parallel private(i, j, a_temp)
#endif
               {
                  for(i = 0 ; i < ncols ; i ++)
                  {
                     a_temp = aa + i * ldim;
#ifdef PARGEMSLR_OPENMP
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
                     for(j = 0 ; j < nrows ; j ++)
                     {
                        y[j] += alpha * a_temp[j] * x[i];
                     }
                  }
               }/* end of OpenMP parallel */
            }
            else if(trans == 'T')
            {
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, a_temp) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
               for(i = 0 ; i < ncols ; i ++)
               {
                  a_temp = aa + i * ldim;
                  for(j = 0 ; j < nrows ; j ++)
                  {
                     y[i] += alpha * a_temp[j] * x[j];
                  }
               }
            }
            else if(trans == 'C')
            {
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, a_temp) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
               for(i = 0 ; i < ncols ; i ++)
               {
                  a_temp = aa + i * ldim;
                  for(j = 0 ; j < nrows ; j ++)
                  {
                     y[i] += alpha * PargemslrConj(a_temp[j]) * x[j];
                  }
               }
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
#pragma omp parallel private(i, j, a_temp)
#endif
               {
                  for(i = 0 ; i < ncols ; i ++)
                  {
                     a_temp = aa + i * ldim;
#ifdef PARGEMSLR_OPENMP
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
                     for(j = 0 ; j < nrows ; j ++)
                     {
                        y[j] += a_temp[j] * x[i];
                     }
                  }
               }/* end of OpenMP parallel */
            }
            else if(trans == 'T')
            {
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, a_temp) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
               for(i = 0 ; i < ncols ; i ++)
               {
                  a_temp = aa + i * ldim;
                  for(j = 0 ; j < nrows ; j ++)
                  {
                     y[i] += a_temp[j] * x[j];
                  }
               }
            }
            else if(trans == 'C')
            {
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, a_temp) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
               for(i = 0 ; i < ncols ; i ++)
               {
                  a_temp = aa + i * ldim;
                  for(j = 0 ; j < nrows ; j ++)
                  {
                     y[i] += PargemslrConj(a_temp[j]) * x[j];
                  }
               }
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
   template int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const float &alpha, const float *aa, int ldim, const float *x, const float &beta, float *y);
   template int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const double &alpha, const double *aa, int ldim, const double *x, const double &beta, double *y);
   template int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const complexs &alpha, const complexs *aa, int ldim, const complexs *x, const complexs &beta, complexs *y);
   template int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const complexd &alpha, const complexd *aa, int ldim, const complexd *x, const complexd &beta, complexd *y);

   int DenseMatrixMatVec( const DenseMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y)
   {
      int   m, n, ldim_a;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();
      ldim_a= A.GetLeadingDimension();

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

               return DenseMatrixSMatVecDevice( A, trans, alpha, x, beta, y);

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

                     return DenseMatrixSMatVecDevice( A, trans, alpha, x, beta, y);

                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return DenseMatrixSMatVecDevice( A, trans, alpha, x, beta, y);
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
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_SGEMV( &trans, &m, &n, &alpha, A.GetData(), &ldim_a, x.GetData(), &one, &beta, y.GetData(), &one);
#else
         DenseMatrixPMatVecTemplate( trans, m, n, alpha, A.GetData(), ldim_a, x.GetData(), beta, y.GetData());
#endif

      }
      else if( ((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0) )
      {
         y.Scale(beta);
      }
      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixMatVec( const DenseMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y)
   {
      int   m, n, ldim_a;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();
      ldim_a= A.GetLeadingDimension();

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

               return DenseMatrixDMatVecDevice( A, trans, alpha, x, beta, y);

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

                     return DenseMatrixDMatVecDevice( A, trans, alpha, x, beta, y);

                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return DenseMatrixDMatVecDevice( A, trans, alpha, x, beta, y);
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
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_DGEMV( &trans, &m, &n, &alpha, A.GetData(), &ldim_a, x.GetData(), &one, &beta, y.GetData(), &one);
#else
         DenseMatrixPMatVecTemplate( trans, m, n, alpha, A.GetData(), ldim_a, x.GetData(), beta, y.GetData());
#endif

      }
      else if( ((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0) )
      {
         y.Scale(beta);
      }
      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixMatVec( const DenseMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y)
   {
      int   m, n, ldim_a;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();
      ldim_a= A.GetLeadingDimension();

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

               return DenseMatrixCMatVecDevice( A, trans, alpha, x, beta, y);

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

                     return DenseMatrixCMatVecDevice( A, trans, alpha, x, beta, y);

                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return DenseMatrixCMatVecDevice( A, trans, alpha, x, beta, y);
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
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_CGEMV( &trans, &m, &n, PARGEMSLR_CAST( const ccomplexs*, &alpha),
                                    PARGEMSLR_CAST( const ccomplexs*, A.GetData()), &ldim_a,
                                    PARGEMSLR_CAST( const ccomplexs*, x.GetData()), &one,
                                    PARGEMSLR_CAST( const ccomplexs*, &beta), PARGEMSLR_CAST( ccomplexs*, y.GetData()), &one);
#else
         DenseMatrixPMatVecTemplate( trans, m, n, alpha, A.GetData(), ldim_a, x.GetData(), beta, y.GetData());
#endif

      }
      else if(((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixMatVec( const DenseMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y)
   {
      int   m, n, ldim_a;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();
      ldim_a= A.GetLeadingDimension();

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

               return DenseMatrixZMatVecDevice( A, trans, alpha, x, beta, y);

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

                     return DenseMatrixZMatVecDevice( A, trans, alpha, x, beta, y);

                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return DenseMatrixZMatVecDevice( A, trans, alpha, x, beta, y);
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
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_ZGEMV( &trans, &m, &n, PARGEMSLR_CAST( const ccomplexd*, &alpha),
                                    PARGEMSLR_CAST( const ccomplexd*, A.GetData()), &ldim_a,
                                    PARGEMSLR_CAST( const ccomplexd*, x.GetData()), &one,
                                    PARGEMSLR_CAST( const ccomplexd*, &beta), PARGEMSLR_CAST( ccomplexd*, y.GetData()), &one);
#else
         DenseMatrixPMatVecTemplate( trans, m, n, alpha, A.GetData(), ldim_a, x.GetData(), beta, y.GetData());
#endif

      }
      else if(((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixInvertHost( DenseMatrixClass<float> &A)
   {
      int m, ldim;

      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();

      PARGEMSLR_CHKERR(m != A.GetNumColsLocal());

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return PARGEMSLR_SUCCESS;
      }

      int                  info, lwork;
      vector_int           ipiv;
      vector_seq_float     work;

      ipiv.Setup(m);

      /* LU factorization */
      PARGEMSLR_BLASLAPACK_SGETRF( &m, &m, A.GetData(), &ldim, ipiv.GetData(), &info); PARGEMSLR_CHKERR(info);

      lwork = m;
      work.Setup(m);

      /* invert */
      PARGEMSLR_BLASLAPACK_SGETRI( &m, A.GetData(), &ldim, ipiv.GetData(), work.GetData(), &lwork, &info); PARGEMSLR_CHKERR(info);

      work.Clear();
      ipiv.Clear();

      return PARGEMSLR_SUCCESS;

   }

   int DenseMatrixInvertHost( DenseMatrixClass<double> &A)
   {
      int m, ldim;

      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();

      PARGEMSLR_CHKERR(m != A.GetNumColsLocal());

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return PARGEMSLR_SUCCESS;
      }

      int                  info, lwork;
      vector_int           ipiv;
      vector_seq_double    work;

      ipiv.Setup(m);

      /* LU factorization */
      PARGEMSLR_BLASLAPACK_DGETRF( &m, &m, A.GetData(), &ldim, ipiv.GetData(), &info); PARGEMSLR_CHKERR(info);

      lwork = m;
      work.Setup(m);

      /* invert */
      PARGEMSLR_BLASLAPACK_DGETRI( &m, A.GetData(), &ldim, ipiv.GetData(), work.GetData(), &lwork, &info); PARGEMSLR_CHKERR(info);

      work.Clear();
      ipiv.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixInvertHost( DenseMatrixClass<complexs> &A)
   {
      int m, ldim;

      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();

      PARGEMSLR_CHKERR(m != A.GetNumColsLocal());

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return PARGEMSLR_SUCCESS;
      }

      int                  info, lwork;
      vector_int           ipiv;
      vector_seq_complexs  work;

      ipiv.Setup(m);

      /* LU factorization */
      PARGEMSLR_BLASLAPACK_CGETRF( &m, &m, PARGEMSLR_CAST(ccomplexs* ,A.GetData()),
                                 &ldim, ipiv.GetData(), &info); PARGEMSLR_CHKERR(info);

      lwork = m;
      work.Setup(m);

      /* invert */
      PARGEMSLR_BLASLAPACK_CGETRI( &m, PARGEMSLR_CAST(ccomplexs* ,A.GetData()),
                                 &ldim, ipiv.GetData(), PARGEMSLR_CAST(ccomplexs* ,work.GetData()), &lwork, &info); PARGEMSLR_CHKERR(info);

      work.Clear();
      ipiv.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixInvertHost( DenseMatrixClass<complexd> &A)
   {
      int m, ldim;

      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();

      PARGEMSLR_CHKERR(m != A.GetNumColsLocal());

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return PARGEMSLR_SUCCESS;
      }

      int                  info, lwork;
      vector_int           ipiv;
      vector_seq_complexd  work;

      ipiv.Setup(m);

      /* LU factorization */
      PARGEMSLR_BLASLAPACK_ZGETRF( &m, &m, PARGEMSLR_CAST(ccomplexd* ,A.GetData()),
                                 &ldim, ipiv.GetData(), &info); PARGEMSLR_CHKERR(info);

      lwork = m;
      work.Setup(m);

      /* invert */
      PARGEMSLR_BLASLAPACK_ZGETRI( &m, PARGEMSLR_CAST(ccomplexd* ,A.GetData()),
                                 &ldim, ipiv.GetData(), PARGEMSLR_CAST(ccomplexd* ,work.GetData()), &lwork, &info); PARGEMSLR_CHKERR(info);

      work.Clear();
      ipiv.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<float> &A)
   {
      int m, ldim;

      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();

      PARGEMSLR_CHKERR(m != A.GetNumColsLocal());

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return PARGEMSLR_SUCCESS;
      }

      int         info;
      char        uplo = 'U';
      char        diag = 'N';

      PARGEMSLR_BLASLAPACK_STRTRI( &uplo, &diag, &m, A.GetData(), &ldim, &info); PARGEMSLR_CHKERR(info);

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<double> &A)
   {
      int m, ldim;

      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();

      PARGEMSLR_CHKERR(m != A.GetNumColsLocal());

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return PARGEMSLR_SUCCESS;
      }

      int         info;
      char        uplo = 'U';
      char        diag = 'N';

      PARGEMSLR_BLASLAPACK_DTRTRI( &uplo, &diag, &m, A.GetData(), &ldim, &info); PARGEMSLR_CHKERR(info);

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<complexs> &A)
   {
      int m, ldim;

      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();

      PARGEMSLR_CHKERR(m != A.GetNumColsLocal());

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return PARGEMSLR_SUCCESS;
      }

      int         info;
      char        uplo = 'U';
      char        diag = 'N';

      PARGEMSLR_BLASLAPACK_CTRTRI( &uplo, &diag, &m, PARGEMSLR_CAST(ccomplexs* ,A.GetData()), &ldim, &info); PARGEMSLR_CHKERR(info);

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<complexd> &A)
   {
      int m, ldim;

      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();

      PARGEMSLR_CHKERR(m != A.GetNumColsLocal());

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return PARGEMSLR_SUCCESS;
      }

      int         info;
      char        uplo = 'U';
      char        diag = 'N';

      PARGEMSLR_BLASLAPACK_ZTRTRI( &uplo, &diag, &m, PARGEMSLR_CAST(ccomplexd* ,A.GetData()), &ldim, &info); PARGEMSLR_CHKERR(info);

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixQRDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumRowsLocal();
      int                  n           = A.GetNumColsLocal();
      int                  minmn       = PargemslrMin(m, n);
      int                  lwork       = n*n;
      int                  ldim_A      = A.GetLeadingDimension();
      DenseMatrixClass<float>          R;
      vector_seq_float     work;             //working array
      vector_seq_float     tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m < n)
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
         {
            Q.Setup( m, m, kMemoryHost, false);
         }
      }
      else
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != n )
         {
            Q.Setup( m, n, kMemoryHost, false);
         }
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0 || n == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(minmn);
      work.Setup(lwork);

      /* QR factorization */
      PARGEMSLR_BLASLAPACK_SGEQRF( &m, &n, A.GetData(), &ldim_A, tau.GetData(), work.GetData(), &lwork, &info);

      /* Q might be allocated outside */
      Q.Fill(0.0);
      R.Setup( minmn, n, kMemoryHost, true);

      /* copy Q data, below the diagonal */
      for (i = 0; i < n; i++)
      {
         for (j = i+1; j < m; j++)
         {
            Q(j,i) = A(j,i);
         }
      }

      /* copy R data */
      for (i = 0; i < minmn; i++)
      {
         for (j = 0; j <= i; j++)
         {
            R(j,i) = A(j,i);
         }
      }
      /* m might be smaller than n */
      for (i = minmn; i < n; i++)
      {
         for (j = 0; j < minmn; j++)
         {
            R(j,i) = A(j,i);
         }
      }

      /* set A for return */
      A = R;

      /* generate matrix Q */
      if(m > n)
      {
         PARGEMSLR_BLASLAPACK_SORGQR(&m, &n, &minmn, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &lwork, &info); PARGEMSLR_CHKERR(info);
      }
      else
      {
         PARGEMSLR_BLASLAPACK_SORGQR(&m, &m, &minmn, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &lwork, &info); PARGEMSLR_CHKERR(info);
      }

      /* deallocate */
      tau.Clear();
      work.Clear();
      R.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixQRDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumRowsLocal();
      int                  n           = A.GetNumColsLocal();
      int                  minmn       = PargemslrMin(m, n);
      int                  lwork       = n*n;
      int                  ldim_A      = A.GetLeadingDimension();
      DenseMatrixClass<double>         R;
      vector_seq_double    work;             //working array
      vector_seq_double    tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m < n)
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
         {
            Q.Setup( m, m, kMemoryHost, false);
         }
      }
      else
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != n )
         {
            Q.Setup( m, n, kMemoryHost, false);
         }
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0 || n == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(minmn);
      work.Setup(lwork);

      /* QR factorization */

      PARGEMSLR_BLASLAPACK_DGEQRF( &m, &n, A.GetData(), &ldim_A, tau.GetData(), work.GetData(), &lwork, &info);

      /* Q might be allocated outside */
      Q.Fill(0.0);
      R.Setup( minmn, n, kMemoryHost, true);

      /* copy Q data, below the diagonal */
      for (i = 0; i < n; i++)
      {
         for (j = i+1; j < m; j++)
         {
            Q(j,i) = A(j,i);
         }
      }

      /* copy R data */
      for (i = 0; i < minmn; i++)
      {
         for (j = 0; j <= i; j++)
         {
            R(j,i) = A(j,i);
         }
      }
      /* m might be smaller than n */
      for (i = minmn; i < n; i++)
      {
         for (j = 0; j < minmn; j++)
         {
            R(j,i) = A(j,i);
         }
      }

      /* set A for return */
      A = R;

      /* generate matrix Q */
      if(m > n)
      {
         PARGEMSLR_BLASLAPACK_DORGQR(&m, &n, &minmn, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &lwork, &info); PARGEMSLR_CHKERR(info);
      }
      else
      {
         PARGEMSLR_BLASLAPACK_DORGQR(&m, &m, &minmn, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &lwork, &info); PARGEMSLR_CHKERR(info);
      }

      /* deallocate */
      tau.Clear();
      work.Clear();
      R.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixQRDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumRowsLocal();
      int                  n           = A.GetNumColsLocal();
      int                  minmn       = PargemslrMin(m, n);
      int                  lwork       = n*n;
      int                  ldim_A      = A.GetLeadingDimension();
      DenseMatrixClass<complexs>       R;
      vector_seq_complexs  work;             //working array
      vector_seq_complexs  tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m < n)
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
         {
            Q.Setup( m, m, kMemoryHost, false);
         }
      }
      else
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != n )
         {
            Q.Setup( m, n, kMemoryHost, false);
         }
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0 || n == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(minmn);
      work.Setup(lwork);

      /* QR factorization */
      PARGEMSLR_BLASLAPACK_CGEQRF( &m, &n, PARGEMSLR_CAST( ccomplexs*, A.GetData()), &ldim_A, PARGEMSLR_CAST( ccomplexs*, tau.GetData()),
                                    PARGEMSLR_CAST( ccomplexs*, work.GetData()), &lwork, &info);

      /* Q might be allocated outside */
      Q.Fill(0.0);
      R.Setup( minmn, n, kMemoryHost, true);

      /* copy Q data, below the diagonal */
      for (i = 0; i < n; i++)
      {
         for (j = i+1; j < m; j++)
         {
            Q(j,i) = A(j,i);
         }
      }

      /* copy R data */
      for (i = 0; i < minmn; i++)
      {
         for (j = 0; j <= i; j++)
         {
            R(j,i) = A(j,i);
         }
      }
      /* m might be smaller than n */
      for (i = minmn; i < n; i++)
      {
         for (j = 0; j < minmn; j++)
         {
            R(j,i) = A(j,i);
         }
      }

      /* set A for return */
      A = R;

      /* generate matrix Q */
      if(m > n)
      {
         PARGEMSLR_BLASLAPACK_CUNGQR(&m, &n, &minmn, PARGEMSLR_CAST( ccomplexs*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexs*, tau.GetData()),
                                    PARGEMSLR_CAST( ccomplexs*, work.GetData()), &lwork, &info); PARGEMSLR_CHKERR(info);
      }
      else
      {
         PARGEMSLR_BLASLAPACK_CUNGQR(&m, &m, &minmn, PARGEMSLR_CAST( ccomplexs*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexs*, tau.GetData()),
                                    PARGEMSLR_CAST( ccomplexs*, work.GetData()), &lwork, &info); PARGEMSLR_CHKERR(info);
      }

      /* deallocate */
      tau.Clear();
      work.Clear();
      R.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixQRDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumRowsLocal();
      int                  n           = A.GetNumColsLocal();
      int                  minmn       = PargemslrMin(m, n);
      int                  lwork       = n*n;
      int                  ldim_A      = A.GetLeadingDimension();
      DenseMatrixClass<complexd>       R;
      vector_seq_complexd  work;             //working array
      vector_seq_complexd  tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m < n)
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
         {
            Q.Setup( m, m, kMemoryHost, false);
         }
      }
      else
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != n )
         {
            Q.Setup( m, n, kMemoryHost, false);
         }
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0 || n == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(minmn);
      work.Setup(lwork);

      /* QR factorization */
      PARGEMSLR_BLASLAPACK_ZGEQRF( &m, &n, PARGEMSLR_CAST( ccomplexd*, A.GetData()), &ldim_A, PARGEMSLR_CAST( ccomplexd*, tau.GetData()),
                                    PARGEMSLR_CAST( ccomplexd*, work.GetData()), &lwork, &info);

      /* Q might be allocated outside */
      Q.Fill(0.0);
      R.Setup( minmn, n, kMemoryHost, true);

      /* copy Q data, below the diagonal */
      for (i = 0; i < n; i++)
      {
         for (j = i+1; j < m; j++)
         {
            Q(j,i) = A(j,i);
         }
      }

      /* copy R data */
      for (i = 0; i < minmn; i++)
      {
         for (j = 0; j <= i; j++)
         {
            R(j,i) = A(j,i);
         }
      }
      /* m might be smaller than n */
      for (i = minmn; i < n; i++)
      {
         for (j = 0; j < minmn; j++)
         {
            R(j,i) = A(j,i);
         }
      }

      /* set A for return */
      A = R;

      /* generate matrix Q */
      if(m > n)
      {
         PARGEMSLR_BLASLAPACK_ZUNGQR(&m, &n, &minmn, PARGEMSLR_CAST( ccomplexd*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexd*, tau.GetData()),
                                 PARGEMSLR_CAST( ccomplexd*, work.GetData()), &lwork, &info); PARGEMSLR_CHKERR(info);
      }
      else
      {
         PARGEMSLR_BLASLAPACK_ZUNGQR(&m, &m, &minmn, PARGEMSLR_CAST( ccomplexd*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexd*, tau.GetData()),
                                 PARGEMSLR_CAST( ccomplexd*, work.GetData()), &lwork, &info); PARGEMSLR_CHKERR(info);
      }

      /* deallocate */
      tau.Clear();
      work.Clear();
      R.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixHessDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q)
   {
      int               i, j;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A      = A.GetLeadingDimension();
      int               one         = 1;  //H is not already upper triangular in any part
      vector_seq_float  work;             //working array
      vector_seq_float  tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(m);
      work.Setup(m);

      /* reduce to upper Hessenberg form */
      PARGEMSLR_BLASLAPACK_SGEHRD( &m, &one, &m, A.GetData(), &ldim_A, tau.GetData()+1, work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++)
      {
         for (j = i+2; j < m; j++)
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0;
         }
      }

      /* generate matrix Q */
      tau[0] = 0.0;
      PARGEMSLR_BLASLAPACK_SORGQR(&m, &m, &m, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      tau.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixHessDecompositionHost( DenseMatrixClass<float> &A, int start, int end, DenseMatrixClass<float> &Q)
   {
      int               i, j;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A      = A.GetLeadingDimension();
      int               ilo, ihi;
      vector_seq_float  work;             //working array
      vector_seq_float  tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      if(m == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(m);
      work.Setup(m);

      /* reduce to upper Hessenberg form
       * A is already upper triangular in rows and columns 1:ILO-1 and IHI+1:N
       */
      ilo = start + 1;
      ihi = end;
      PARGEMSLR_BLASLAPACK_SGEHRD( &m, &ilo, &ihi, A.GetData(), &ldim_A, tau.GetData()+1, work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++)
      {
         for (j = i+2; j < m; j++)
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0;
         }
      }

      /* generate matrix Q */
      tau[0] = 0.0;
      PARGEMSLR_BLASLAPACK_SORGQR(&m, &m, &m, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      tau.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixHessDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q)
   {
      int               i, j;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A      = A.GetLeadingDimension();
      int               one         = 1;  //H is not already upper triangular in any part
      vector_seq_double work;             //working array
      vector_seq_double tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(m);
      work.Setup(m);

      /* reduce to upper Hessenberg form */
      PARGEMSLR_BLASLAPACK_DGEHRD( &m, &one, &m, A.GetData(), &ldim_A, tau.GetData()+1, work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++)
      {
         for (j = i+2; j < m; j++)
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0;
         }
      }

      /* generate matrix Q */
      tau[0] = 0.0;
      PARGEMSLR_BLASLAPACK_DORGQR(&m, &m, &m, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      tau.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixHessDecompositionHost( DenseMatrixClass<double> &A, int start, int end, DenseMatrixClass<double> &Q)
   {
      int               i, j;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A      = A.GetLeadingDimension();
      int               ilo, ihi;
      vector_seq_double work;             //working array
      vector_seq_double tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(m);
      work.Setup(m);

      /* reduce to upper Hessenberg form
       * A is already upper triangular in rows and columns 1:ILO-1 and IHI+1:N
       */
      ilo = start + 1;
      ihi = end;
      PARGEMSLR_BLASLAPACK_DGEHRD( &m, &ilo, &ihi, A.GetData(), &ldim_A, tau.GetData()+1, work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++)
      {
         for (j = i+2; j < m; j++)
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0;
         }
      }

      /* generate matrix Q */
      tau[0] = 0.0;
      PARGEMSLR_BLASLAPACK_DORGQR(&m, &m, &m, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      tau.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumColsLocal();
      int                  ldim_A      = A.GetLeadingDimension();
      int                  one         = 1;  //H is not already upper triangular in any part
      vector_seq_complexs  work;             //working array
      vector_seq_complexs  tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(m);
      work.Setup(m);

      /* reduce to upper Hessenberg form */
      PARGEMSLR_BLASLAPACK_CGEHRD( &m, &one, &m, PARGEMSLR_CAST( ccomplexs*, A.GetData()), &ldim_A, PARGEMSLR_CAST( ccomplexs*, tau.GetData()+1),
                                 PARGEMSLR_CAST( ccomplexs*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++)
      {
         for (j = i+2; j < m; j++)
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0;
         }
      }

      /* generate matrix Q */
      tau[0] = complexs(0.0,0.0);
      PARGEMSLR_BLASLAPACK_CUNGQR(&m, &m, &m, PARGEMSLR_CAST( ccomplexs*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexs*, tau.GetData()),
                                 PARGEMSLR_CAST( ccomplexs*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      tau.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexs> &A, int start, int end, DenseMatrixClass<complexs> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumColsLocal();
      int                  ldim_A      = A.GetLeadingDimension();
      int                  ilo, ihi;
      vector_seq_complexs  work;             //working array
      vector_seq_complexs  tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(m);
      work.Setup(m);

      /* reduce to upper Hessenberg form
       * A is already upper triangular in rows and columns 1:ILO-1 and IHI+1:N
       */
      ilo = start + 1;
      ihi = end;
      PARGEMSLR_BLASLAPACK_CGEHRD( &m, &ilo, &ihi, PARGEMSLR_CAST( ccomplexs*, A.GetData()), &ldim_A, PARGEMSLR_CAST( ccomplexs*, tau.GetData()+1),
                                 PARGEMSLR_CAST( ccomplexs*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++)
      {
         for (j = i+2; j < m; j++)
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0;
         }
      }

      /* generate matrix Q */
      tau[0] = complexs(0.0,0.0);
      PARGEMSLR_BLASLAPACK_CUNGQR(&m, &m, &m, PARGEMSLR_CAST( ccomplexs*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexs*, tau.GetData()),
                                 PARGEMSLR_CAST( ccomplexs*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      tau.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumColsLocal();
      int                  ldim_A      = A.GetLeadingDimension();
      int                  one         = 1;  //H is not already upper triangular in any part
      vector_seq_complexd  work;             //working array
      vector_seq_complexd  tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(m);
      work.Setup(m);

      /* reduce to upper Hessenberg form */
      PARGEMSLR_BLASLAPACK_ZGEHRD( &m, &one, &m, PARGEMSLR_CAST( ccomplexd*, A.GetData()), &ldim_A, PARGEMSLR_CAST( ccomplexd*, tau.GetData()+1),
                                 PARGEMSLR_CAST( ccomplexd*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++)
      {
         for (j = i+2; j < m; j++)
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0;
         }
      }

      /* generate matrix Q */
      tau[0] = complexd(0.0,0.0);
      PARGEMSLR_BLASLAPACK_ZUNGQR(&m, &m, &m, PARGEMSLR_CAST( ccomplexd*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexd*, tau.GetData()),
                                 PARGEMSLR_CAST( ccomplexd*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      tau.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexd> &A, int start, int end, DenseMatrixClass<complexd> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumColsLocal();
      int                  ldim_A      = A.GetLeadingDimension();
      int                  ilo, ihi;
      vector_seq_complexd  work;             //working array
      vector_seq_complexd  tau;

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q      = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(m == 0)
      {
         /* do nothing when A is empty */
         return PARGEMSLR_SUCCESS;
      }

      tau.Setup(m);
      work.Setup(m);

      /* reduce to upper Hessenberg form
       * A is already upper triangular in rows and columns 1:ILO-1 and IHI+1:N
       */
      ilo = start + 1;
      ihi = end;
      PARGEMSLR_BLASLAPACK_ZGEHRD( &m, &ilo, &ihi, PARGEMSLR_CAST( ccomplexd*, A.GetData()), &ldim_A, PARGEMSLR_CAST( ccomplexd*, tau.GetData()+1),
                                 PARGEMSLR_CAST( ccomplexd*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++)
      {
         for (j = i+2; j < m; j++)
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0;
         }
      }

      /* generate matrix Q */
      tau[0] = complexd(0.0,0.0);
      PARGEMSLR_BLASLAPACK_ZUNGQR(&m, &m, &m, PARGEMSLR_CAST( ccomplexd*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexd*, tau.GetData()),
                                 PARGEMSLR_CAST( ccomplexd*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      tau.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int              ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     one         = 1;  //H is not already upper triangular in any part
      vector_seq_float        work;           //work array

      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      work.Setup(m);

      PARGEMSLR_BLASLAPACK_SHSEQR(&job_schur, &compz, &m, &one, &m, A.GetData(), &ldim_A,
            wr.GetData(), wi.GetData(), Q.GetData(), &ldim_Q, work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<float> &A, int start, int end, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int              ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     ilo, ihi;
      vector_seq_float        work;           //work array

      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      work.Setup(m);

      /* note that if there are 2 by 2 blocks on 1:ilo-1 and ihi+1:m,
       * wr and wi would be inaccuarte on those entries
       */
      ilo = start + 1;
      ihi = end;
      PARGEMSLR_BLASLAPACK_SHSEQR(&job_schur, &compz, &m, &ilo, &ihi, A.GetData(), &ldim_A,
            wr.GetData(), wi.GetData(), Q.GetData(), &ldim_Q, work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int              ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     one         = 1;  //H is not already upper triangular in any part
      vector_seq_double       work;           //work array

      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      work.Setup(m);

      PARGEMSLR_BLASLAPACK_DHSEQR(&job_schur, &compz, &m, &one, &m, A.GetData(), &ldim_A,
            wr.GetData(), wi.GetData(), Q.GetData(), &ldim_Q, work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<double> &A, int start, int end, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int              ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     ilo, ihi;
      vector_seq_double       work;           //work array

      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      work.Setup(m);

      /* note that if there are 2 by 2 blocks on 1:ilo-1 and ihi+1:m,
       * wr and wi would be inaccuarte on those entries
       */
      ilo = start + 1;
      ihi = end;
      PARGEMSLR_BLASLAPACK_DHSEQR(&job_schur, &compz, &m, &ilo, &ihi, A.GetData(), &ldim_A,
            wr.GetData(), wi.GetData(), Q.GetData(), &ldim_Q, work.GetData(), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int              ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     one         = 1;  //H is not already upper triangular in any part
      vector_seq_complexs     work;           //work array

      /* allocate memory */
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      // work array, just use max(1,n)
      work.Setup(m);

      PARGEMSLR_BLASLAPACK_CHSEQR(&job_schur, &compz, &m, &one, &m, PARGEMSLR_CAST(ccomplexs*, A.GetData()),
            &ldim_A, PARGEMSLR_CAST(ccomplexs*, w.GetData()), PARGEMSLR_CAST(ccomplexs*, Q.GetData()),
            &ldim_Q, PARGEMSLR_CAST(ccomplexs*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexs> &A, int start, int end, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int              ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     ilo, ihi;
      vector_seq_complexs     work;           //work array

      /* allocate memory */
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      // work array, just use max(1,n)
      work.Setup(m);

      ilo = start + 1;
      ihi = end;
      PARGEMSLR_BLASLAPACK_CHSEQR(&job_schur, &compz, &m, &ilo, &ihi, PARGEMSLR_CAST(ccomplexs*, A.GetData()),
            &ldim_A, PARGEMSLR_CAST(ccomplexs*, w.GetData()), PARGEMSLR_CAST(ccomplexs*, Q.GetData()),
            &ldim_Q, PARGEMSLR_CAST(ccomplexs*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int              ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     one         = 1;  //H is not already upper triangular in any part
      vector_seq_complexd     work;           //work array

      /* allocate memory */
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      // work array, just use max(1,n)
      work.Setup(m);

      PARGEMSLR_BLASLAPACK_ZHSEQR(&job_schur, &compz, &m, &one, &m, PARGEMSLR_CAST(ccomplexd*, A.GetData()),
            &ldim_A, PARGEMSLR_CAST(ccomplexd*, w.GetData()), PARGEMSLR_CAST(ccomplexd*, Q.GetData()),
            &ldim_Q, PARGEMSLR_CAST(ccomplexd*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexd> &A, int start, int end, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int              ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     ilo, ihi;
      vector_seq_complexd     work;           //work array

      /* allocate memory */
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Matrix decomposition in host only.");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      // work array, just use max(1,n)
      work.Setup(m);

      ilo = start + 1;
      ihi = end;
      PARGEMSLR_BLASLAPACK_ZHSEQR(&job_schur, &compz, &m, &ilo, &ihi, PARGEMSLR_CAST(ccomplexd*, A.GetData()),
            &ldim_A, PARGEMSLR_CAST(ccomplexd*, w.GetData()), PARGEMSLR_CAST(ccomplexd*, Q.GetData()),
            &ldim_Q, PARGEMSLR_CAST(ccomplexd*, work.GetData()), &m, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealHessEigenDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi)
   {
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( wr.GetLengthLocal() != m);
      PARGEMSLR_CHKERR( wi.GetLengthLocal() != m);

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }

      /* Lapack paras */
      int               mmm         = m;
      char              side        = 'R';//only compute right eigenvalues
      char              eigsrc      = 'Q';//Eigenvalues were computed from DHSEQR
      char              initv       = 'N';//No initial vectors
      vector_int        select;              //Fortran LOGICAL array, select which eigenvalue to be computed.
      vector_int        ifailr;              //Check convergence of DHSEIN
      //GEMSLR_Int              ifailr_sum;          //Sum of the value
      int               mm;                  //# of cols in eigenvector matrix
      vector_seq_float  work;                //work array
      vector_int        iwork;
      int               one         = 1;  //H is not already upper triangular in any part

      /* select array, select all of them so set to 1 */
      select.Setup(m);
      select.Fill(1);

      /* work array */
      work.Setup((m+2)*m);
      /* fail array */
      ifailr.Setup(m);

      mm = m;

      PARGEMSLR_BLASLAPACK_SHSEIN(&side, &eigsrc, &initv, select.GetData(), &m,
                                 A.GetData(), &ldim_A, wr.GetData(), wi.GetData(), NULL, &one,
                                 Q.GetData(), &ldim_Q, &mm, &mmm, work.GetData(), NULL, ifailr.GetData(), &info); PARGEMSLR_CHKERR(info);

      //deallocate
      work.Clear();
      ifailr.Clear();
      select.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealHessEigenDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi)
   {
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( wr.GetLengthLocal() != m);
      PARGEMSLR_CHKERR( wi.GetLengthLocal() != m);

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }

      /* Lapack paras */
      int               mmm         = m;
      char              side        = 'R';//only compute right eigenvalues
      char              eigsrc      = 'Q';//Eigenvalues were computed from DHSEQR
      char              initv       = 'N';//No initial vectors
      vector_int        select;              //Fortran LOGICAL array, select which eigenvalue to be computed.
      vector_int        ifailr;              //Check convergence of DHSEIN
      //GEMSLR_Int              ifailr_sum;          //Sum of the value
      int               mm;                  //# of cols in eigenvector matrix
      vector_seq_double work;                //work array
      vector_int        iwork;
      int               one         = 1;  //H is not already upper triangular in any part

      /* select array, select all of them so set to 1 */
      select.Setup(m);
      select.Fill(1);

      /* work array */
      work.Setup((m+2)*m);
      /* fail array */
      ifailr.Setup(m);

      mm = m;

      PARGEMSLR_BLASLAPACK_DHSEIN(&side, &eigsrc, &initv, select.GetData(), &m,
                                 A.GetData(), &ldim_A, wr.GetData(), wi.GetData(), NULL, &one,
                                 Q.GetData(), &ldim_Q, &mm, &mmm, work.GetData(), NULL, ifailr.GetData(), &info); PARGEMSLR_CHKERR(info);

      //deallocate
      work.Clear();
      ifailr.Clear();
      select.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexHessEigenDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w)
   {
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( w.GetLengthLocal() != m);

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }

      /* Lapack paras */
      int                  mmm         = m;
      char                 side        = 'R';//only compute right eigenvalues
      char                 eigsrc      = 'Q';//Eigenvalues were computed from DHSEQR
      char                 initv       = 'N';//No initial vectors
      vector_int           select;              //Fortran LOGICAL array, select which eigenvalue to be computed.
      vector_int           ifailr;              //Check convergence of DHSEIN
      //GEMSLR_Int              ifailr_sum;          //Sum of the value
      int                  mm;                  //# of cols in eigenvector matrix
      vector_seq_float     rwork;                //work array
      vector_int           iwork;
      vector_seq_complexs  work;           //work array
      int                  one         = 1;  //H is not already upper triangular in any part

      // select array, select all of them so set to 1
      select.Setup(m);
      select.Fill(1);

      // work array
      work.Setup(m*m);
      rwork.Setup(m);
      // fail array
      ifailr.Setup(m);

      mm = m;

      PARGEMSLR_BLASLAPACK_CHSEIN(&side, &eigsrc, &initv, select.GetData(), &m,
                              PARGEMSLR_CAST(ccomplexs*, A.GetData()), &ldim_A, PARGEMSLR_CAST(ccomplexs*, w.GetData()),
                              NULL, &one, PARGEMSLR_CAST(ccomplexs*, Q.GetData()), &ldim_Q,
                              &mm, &mmm, PARGEMSLR_CAST(ccomplexs*, work.GetData()), rwork.GetData(),
                              NULL, ifailr.GetData(), &info); PARGEMSLR_CHKERR(info);

      //deallocate
      work.Clear();
      rwork.Clear();
      ifailr.Clear();
      select.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexHessEigenDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w)
   {
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A = A.GetLeadingDimension();

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( w.GetLengthLocal() != m);

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }

      int               ldim_Q = Q.GetLeadingDimension();

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }

      /* Lapack paras */
      int                  mmm         = m;
      char                 side        = 'R';//only compute right eigenvalues
      char                 eigsrc      = 'Q';//Eigenvalues were computed from DHSEQR
      char                 initv       = 'N';//No initial vectors
      vector_int           select;              //Fortran LOGICAL array, select which eigenvalue to be computed.
      vector_int           ifailr;              //Check convergence of DHSEIN
      //GEMSLR_Int              ifailr_sum;          //Sum of the value
      int                  mm;                  //# of cols in eigenvector matrix
      vector_seq_double    rwork;                //work array
      vector_int           iwork;
      vector_seq_complexd  work;           //work array
      int                  one         = 1;  //H is not already upper triangular in any part

      // select array, select all of them so set to 1
      select.Setup(m);
      select.Fill(1);

      // work array
      work.Setup(m*m);
      rwork.Setup(m);
      // fail array
      ifailr.Setup(m);

      mm = m;

      PARGEMSLR_BLASLAPACK_ZHSEIN(&side, &eigsrc, &initv, select.GetData(), &m,
                              PARGEMSLR_CAST(ccomplexd*, A.GetData()), &ldim_A, PARGEMSLR_CAST(ccomplexd*, w.GetData()),
                              NULL, &one, PARGEMSLR_CAST(ccomplexd*, Q.GetData()), &ldim_Q,
                              &mm, &mmm, PARGEMSLR_CAST(ccomplexd*, work.GetData()), rwork.GetData(),
                              NULL, ifailr.GetData(), &info); PARGEMSLR_CHKERR(info);

      //deallocate
      work.Clear();
      rwork.Clear();
      ifailr.Clear();
      select.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealOrderSchur( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi, vector_int &select)
   {
      /* pre processing the select array */
      int                  info         = 0;
      int                  m           = A.GetNumColsLocal();
      int                  mm;             // out put, dim of the invariant subspace
      vector_seq_float     work;           //work array
      vector_int           iwork;
      int                  lwork, liwork;          // length of working space

      if(m == 0)
      {
         return PARGEMSLR_SUCCESS;
      }

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumColsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( select.GetLengthLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();

      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }

      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }

      if(select.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         select.MoveData(kMemoryHost);
      }

      char                    job         = 'N';  // condition numbers not required
      char                    compq       = 'V';  // update the matrix

      lwork = m;
      work.Setup(lwork);
      liwork = 1;
      iwork.Setup(liwork);

      PARGEMSLR_BLASLAPACK_STRSEN( &job, &compq, select.GetData(), &m, A.GetData(), &ldim_A,
               Q.GetData(), &ldim_Q, wr.GetData(), wi.GetData(), &mm, NULL, NULL, work.GetData(),
               &lwork, iwork.GetData(), &liwork, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();
      iwork.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealOrderSchur( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi, vector_int &select)
   {
      /* pre processing the select array */
      int                  info         = 0;
      int                  m           = A.GetNumColsLocal();
      int                  mm;             // out put, dim of the invariant subspace
      vector_seq_double    work;           //work array
      vector_int           iwork;
      int                  lwork, liwork;          // length of working space

      if(m == 0)
      {
         return PARGEMSLR_SUCCESS;
      }

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumColsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( select.GetLengthLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();

      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }

      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }

      if(select.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         select.MoveData(kMemoryHost);
      }

      char                    job         = 'N';  // condition numbers not required
      char                    compq       = 'V';  // update the matrix

      lwork = m;
      work.Setup(lwork);
      liwork = 1;
      iwork.Setup(liwork);

      PARGEMSLR_BLASLAPACK_DTRSEN( &job, &compq, select.GetData(), &m, A.GetData(), &ldim_A,
               Q.GetData(), &ldim_Q, wr.GetData(), wi.GetData(), &mm, NULL, NULL, work.GetData(),
               &lwork, iwork.GetData(), &liwork, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();
      iwork.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexOrderSchur( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w, vector_int &select)
   {
      /* pre processing the select array */
      int                  info         = 0;
      int                  m           = A.GetNumColsLocal();
      int                  mm;             // out put, dim of the invariant subspace
      vector_seq_complexs  work;           //work array
      int                  lwork;          // length of working space

      if(m == 0)
      {
         return PARGEMSLR_SUCCESS;
      }

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumColsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( select.GetLengthLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();

      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }

      if(select.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         select.MoveData(kMemoryHost);
      }

      char                    job         = 'N';  // condition numbers not required
      char                    compq       = 'V';  // update the matrix

      lwork = m;
      work.Setup(lwork);

      PARGEMSLR_BLASLAPACK_CTRSEN(&job, &compq, select.GetData(), &m, PARGEMSLR_CAST( ccomplexs*, A.GetData()), &ldim_A,
                                 PARGEMSLR_CAST( ccomplexs*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexs*, w.GetData()),
                                 &mm, NULL, NULL, PARGEMSLR_CAST( ccomplexs*, work.GetData()), &lwork, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexOrderSchur( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w, vector_int &select)
   {
      /* pre processing the select array */
      int                  info         = 0;
      int                  m           = A.GetNumColsLocal();
      int                  mm;             // out put, dim of the invariant subspace
      vector_seq_complexd  work;           //work array
      int                  lwork;          // length of working space

      if(m == 0)
      {
         return PARGEMSLR_SUCCESS;
      }

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumColsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( select.GetLengthLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();

      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }

      if(select.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         select.MoveData(kMemoryHost);
      }

      char                    job         = 'N';  // condition numbers not required
      char                    compq       = 'V';  // update the matrix

      lwork = m;
      work.Setup(lwork);

      PARGEMSLR_BLASLAPACK_ZTRSEN(&job, &compq, select.GetData(), &m, PARGEMSLR_CAST( ccomplexd*, A.GetData()), &ldim_A,
                                 PARGEMSLR_CAST( ccomplexd*, Q.GetData()), &ldim_Q, PARGEMSLR_CAST( ccomplexd*, w.GetData()),
                                 &mm, NULL, NULL, PARGEMSLR_CAST( ccomplexd*, work.GetData()), &lwork, &info); PARGEMSLR_CHKERR(info);

      /* deallocate */
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealOrderSchurClusters( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi, vector_int &clusters)
   {
      /* pre processing the select array */
      int               i, j;
      int               ifst, ilst;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               idx;
      int               temp_idx;
      int               case_number = 0;
      vector_int        order;
      vector_int        iorder; // working array
      vector_seq_float  work;
      float             eps = std::numeric_limits<float>::epsilon();

      if(m == 0)
      {
         return PARGEMSLR_SUCCESS;
      }

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumColsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( clusters.GetLengthLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();

      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }

      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }

      if(clusters.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         clusters.MoveData(kMemoryHost);
      }

      // condition numbers not required
      // update the matrix
      char                    compq       = 'V';

      /* not working yet, the stable sort */
      clusters.Sort( order, false, true);

      iorder.Setup(m);

      for(i = 0 ; i < m ; i ++)
      {
         iorder[order[i]] = i;
      }

      work.Setup(m);

      i = 0;

      while(true)
      {
         idx = order[i];
         if(clusters[idx] <= 0)
         {
            break;
         }

         case_number = 0;
         /* we use the stable sort, so the order of the 2x2 block kept */
         if(idx != i)
         {

            /* need to insert */
            if(idx < m-1)
            {
               if(i < m-1)
               {
                  /* both not the last one, check if they are 2 by 2 blocks */
                  if( PargemslrAbs( A(idx+1, idx) ) > eps )
                  {
                     /* in this case, a 2 by 2 block is found at idx */
                     if( PargemslrAbs( A(i+1, i) ) > eps )
                     {
                        /* in this case, a 2 by 2 block is found at i
                         * replace a 2 by 2 block with another 2 by 2 block
                         * swap idx with i, and idx+1 with i+1
                         */

                        case_number = 22;

                     }
                     else
                     {
                        /* in this case, a 1 by 1 block is found at i
                         * replace a 1 by 1 block with a 2 by 2 block
                         */

                        case_number = 12;

                     }
                  }
                  else
                  {
                     /* in this case, a 1 by 1 block is found at idx */
                     if( PargemslrAbs( A(i+1, i) ) > eps )
                     {
                        /* in this case, a 2 by 2 block is found at i
                         * replace a 2 by 2 block with a 1 by 1 block
                         */

                        case_number = 21;

                     }
                     else
                     {
                        /* in this case, a 1 by 1 block is found at i
                         * both 1 by 1
                         */

                        case_number = 11;

                     }
                  }
               }
               else
               {
                  /* i is the last one, 1 by 1 block */
                  if( PargemslrAbs( A(idx+1, idx) ) > eps )
                  {
                     /* in this case, a 2 by 2 block is found at idx
                      * replace 1 by 1 with 2 by 2
                      */

                     case_number = 12;

                  }
                  else
                  {
                     /* in this case, a 1 by 1 block is found at idx
                      * both 1 by 1
                      */

                     case_number = 11;

                  }
               }
            }
            else
            {
               /* idx is the last one, 1 by 1 block */
               if( PargemslrAbs( A(i+1, i) ) > eps )
               {
                  /* in this case, a 2 by 2 block is found at i
                   * replace 2 by 2 with 1 by 1
                   */

                  case_number = 21;

               }
               else
               {
                  /* in this case, a 1 by 1 block is found at i
                   * both 1 by 1
                   */
                  case_number = 11;
               }
            }

            /* apply the insert */
            ifst = idx+1;
            ilst = i+1;
            PARGEMSLR_BLASLAPACK_STREXC(&compq, &m, A.GetData(), &ldim_A, Q.GetData(), &ldim_Q, &ifst, &ilst, work.GetData(), &info); PARGEMSLR_CHKERR(info);

            switch (case_number)
            {
               case 11:
               {
                  /* if we have i, i+1, ... , i+m, idx, and order[k0] = i, order[k1] = i+1, ..., order[km] = i+m,
                   * after swap we'll have order[k0] = i+1, order[k1] = i+2, ..., order[km] = idx
                   */

                  for(j = idx-1 ; j >= i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+1;
                     iorder[j+1] = temp_idx;
                  }
                  i++;

                  break;
               }
               case 12:
               {

                  /* if we have i, i+1, ... , i+m, idx, idx+1, and order[k0] = i, order[k1] = i+1, ..., order[km] = i+m,
                   * after swap we'll have order[k0] = i+2, order[k1] = i+3, ..., order[km] = idx+1
                   */

                  for(j = idx-1 ; j > i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+2;
                     iorder[j+2] = temp_idx;
                  }
                  i+=2;

                  break;
               }
               case 21:
               {

                  /* in this case same as 11, since we can guarentee that i is not the second of a 2 by 2 block */

                  for(j = idx-1 ; j >= i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+1;
                     iorder[j+1] = temp_idx;
                  }
                  i++;

                  break;
               }
               case 22:
               {

                  /* in this case same as 12, since we can guarentee that i is not the second of a 2 by 2 block */

                  for(j = idx-1 ; j > i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+2;
                     iorder[j+2] = temp_idx;
                  }
                  i+=2;

                  break;
               }
               default:
               {
                  return PARGEMSLR_ERROR_INVALED_PARAM;
               }

            }

         }
         else
         {
            /* no need to insert */
            if(idx < m-1)
            {
               /* both not the last one, check if they are 2 by 2 blocks */
               if( PargemslrAbs( A(idx+1, idx) ) > eps )
               {
                  i+=2;
               }
               else
               {
                  i++;
               }
            }
            else
            {
               i++;
            }
         }

         if(i >= m)
         {
            break;
         }

      }

      for(i = 0 ; i < m ; i ++)
      {
         if(i < m-1)
         {
            /* both not the last one, check if they are 2 by 2 blocks */
            if( PargemslrAbs( A(i+1, i) ) > eps )
            {
               wr[i] = A(i, i);
               wr[i+1] = A(i, i);
               wi[i] = sqrt(-A(i+1,i)*A(i,i+1));
               wi[i+1] = -wi[i];
               i++;
            }
            else
            {
               wr[i] = A(i, i);
               wi[i] = 0.0f;
            }
         }
         else
         {
            wr[i] = A(i, i);
            wi[i] = 0.0f;
         }
      }

      //deallocate
      order.Clear();
      iorder.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixRealOrderSchurClusters( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi, vector_int &clusters)
   {
      /* pre processing the select array */
      int               i, j;
      int               ifst, ilst;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               idx;
      int               temp_idx;
      int               case_number = 0;
      vector_int        order;
      vector_int        iorder; // working array
      vector_seq_double work;
      float             eps = std::numeric_limits<double>::epsilon();

      if(m == 0)
      {
         return PARGEMSLR_SUCCESS;
      }

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumColsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( clusters.GetLengthLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();

      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }

      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }

      if(clusters.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         clusters.MoveData(kMemoryHost);
      }

      // condition numbers not required
      // update the matrix
      char                    compq       = 'V';

      /* not working yet, the stable sort */
      clusters.Sort( order, false, true);

      iorder.Setup(m);

      for(i = 0 ; i < m ; i ++)
      {
         iorder[order[i]] = i;
      }

      work.Setup(m);

      i = 0;

      while(true)
      {
         idx = order[i];
         if(clusters[idx] <= 0)
         {
            break;
         }

         case_number = 0;
         /* we use the stable sort, so the order of the 2x2 block kept */
         if(idx != i)
         {

            /* need to insert */
            if(idx < m-1)
            {
               if(i < m-1)
               {
                  /* both not the last one, check if they are 2 by 2 blocks */
                  if( PargemslrAbs( A(idx+1, idx) ) > eps )
                  {
                     /* in this case, a 2 by 2 block is found at idx */
                     if( PargemslrAbs( A(i+1, i) ) > eps )
                     {
                        /* in this case, a 2 by 2 block is found at i
                         * replace a 2 by 2 block with another 2 by 2 block
                         * swap idx with i, and idx+1 with i+1
                         */

                        case_number = 22;

                     }
                     else
                     {
                        /* in this case, a 1 by 1 block is found at i
                         * replace a 1 by 1 block with a 2 by 2 block
                         */

                        case_number = 12;

                     }
                  }
                  else
                  {
                     /* in this case, a 1 by 1 block is found at idx */
                     if( PargemslrAbs( A(i+1, i) ) > eps )
                     {
                        /* in this case, a 2 by 2 block is found at i
                         * replace a 2 by 2 block with a 1 by 1 block
                         */

                        case_number = 21;

                     }
                     else
                     {
                        /* in this case, a 1 by 1 block is found at i
                         * both 1 by 1
                         */

                        case_number = 11;

                     }
                  }
               }
               else
               {
                  /* i is the last one, 1 by 1 block */
                  if( PargemslrAbs( A(idx+1, idx) ) > eps )
                  {
                     /* in this case, a 2 by 2 block is found at idx
                      * replace 1 by 1 with 2 by 2
                      */

                     case_number = 12;

                  }
                  else
                  {
                     /* in this case, a 1 by 1 block is found at idx
                      * both 1 by 1
                      */

                     case_number = 11;

                  }
               }
            }
            else
            {
               /* idx is the last one, 1 by 1 block */
               if( PargemslrAbs( A(i+1, i) ) > eps )
               {
                  /* in this case, a 2 by 2 block is found at i
                   * replace 2 by 2 with 1 by 1
                   */

                  case_number = 21;

               }
               else
               {
                  /* in this case, a 1 by 1 block is found at i
                   * both 1 by 1
                   */
                  case_number = 11;
               }
            }

            /* apply the insert */
            ifst = idx+1;
            ilst = i+1;
            PARGEMSLR_BLASLAPACK_DTREXC(&compq, &m, A.GetData(), &ldim_A, Q.GetData(), &ldim_Q, &ifst, &ilst, work.GetData(), &info); PARGEMSLR_CHKERR(info);

            switch (case_number)
            {
               case 11:
               {
                  /* if we have i, i+1, ... , i+m, idx, and order[k0] = i, order[k1] = i+1, ..., order[km] = i+m,
                   * after swap we'll have order[k0] = i+1, order[k1] = i+2, ..., order[km] = idx
                   */

                  for(j = idx-1 ; j >= i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+1;
                     iorder[j+1] = temp_idx;
                  }
                  i++;

                  break;
               }
               case 12:
               {

                  /* if we have i, i+1, ... , i+m, idx, idx+1, and order[k0] = i, order[k1] = i+1, ..., order[km] = i+m,
                   * after swap we'll have order[k0] = i+2, order[k1] = i+3, ..., order[km] = idx+1
                   */

                  for(j = idx-1 ; j > i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+2;
                     iorder[j+2] = temp_idx;
                  }
                  i+=2;

                  break;
               }
               case 21:
               {

                  /* in this case same as 11, since we can guarentee that i is not the second of a 2 by 2 block */

                  for(j = idx-1 ; j >= i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+1;
                     iorder[j+1] = temp_idx;
                  }
                  i++;

                  break;
               }
               case 22:
               {

                  /* in this case same as 12, since we can guarentee that i is not the second of a 2 by 2 block */

                  for(j = idx-1 ; j > i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+2;
                     iorder[j+2] = temp_idx;
                  }
                  i+=2;

                  break;
               }
               default:
               {
                  return PARGEMSLR_ERROR_INVALED_PARAM;
               }

            }

         }
         else
         {
            /* no need to insert */
            if(idx < m-1)
            {
               /* both not the last one, check if they are 2 by 2 blocks */
               if( PargemslrAbs( A(idx+1, idx) ) > eps )
               {
                  i+=2;
               }
               else
               {
                  i++;
               }
            }
            else
            {
               i++;
            }
         }

         if(i >= m)
         {
            break;
         }

      }

      for(i = 0 ; i < m ; i ++)
      {
         if(i < m-1)
         {
            /* both not the last one, check if they are 2 by 2 blocks */
            if( PargemslrAbs( A(i+1, i) ) > eps )
            {
               wr[i] = A(i, i);
               wr[i+1] = A(i, i);
               wi[i] = sqrt(-A(i+1,i)*A(i,i+1));
               wi[i+1] = -wi[i];
               i++;
            }
            else
            {
               wr[i] = A(i, i);
               wi[i] = 0.0f;
            }
         }
         else
         {
            wr[i] = A(i, i);
            wi[i] = 0.0f;
         }
      }

      //deallocate
      order.Clear();
      iorder.Clear();
      work.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexOrderSchurClusters( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w, vector_int &clusters)
   {
      /* pre processing the select array */
      int              i, j;
      int              ifst, ilst;
      int              info        = 0;
      int              m           = A.GetNumColsLocal();
      int              idx;
      int              temp_idx;
      vector_int       order;
      vector_int       iorder; // working array

      if(m == 0)
      {
         return PARGEMSLR_SUCCESS;
      }

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumColsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( clusters.GetLengthLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();

      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }

      if(clusters.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         clusters.MoveData(kMemoryHost);
      }

      // condition numbers not required
      // update the matrix
      char                    compq       = 'V';

      /* not working yet, the stable sort */
      clusters.Sort( order, false, true);

      iorder.Setup(m);

      for(i = 0 ; i < m ; i ++)
      {
         iorder[order[i]] = i;
      }

      i = 0;

      while(true)
      {
         idx = order[i];

         if(clusters[idx] <= 0)
         {
            break;
         }

         if(idx != i)
         {
            ifst = idx+1;
            ilst = i+1;
            PARGEMSLR_BLASLAPACK_CTREXC(&compq, &m, PARGEMSLR_CAST( ccomplexs*, A.GetData()), &ldim_A,
                                       PARGEMSLR_CAST( ccomplexs*, Q.GetData()), &ldim_Q, &ifst, &ilst, &info); PARGEMSLR_CHKERR(info);

            /* after the swap, if we have order[k] = i, we need to update order[k] = idx */

            for(j = idx-1 ; j >= i ; j --)
            {
               temp_idx = iorder[j];
               order[temp_idx] = j+1;
               iorder[j+1] = temp_idx;
            }

            /* we'll not touch index i anymore, no need to update order[i] and iorder[i] */

         }
         i++;

         if( i >= m)
         {
            break;
         }

      }

      for(i = 0 ; i < m ; i ++)
      {
         w[i] = A(i, i);
      }

      //deallocate
      order.Clear();
      iorder.Clear();

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixComplexOrderSchurClusters( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w, vector_int &clusters)
   {
      /* pre processing the select array */
      int              i, j;
      int              ifst, ilst;
      int              info        = 0;
      int              m           = A.GetNumColsLocal();
      int              idx;
      int              temp_idx;
      vector_int       order;
      vector_int       iorder; // working array

      if(m == 0)
      {
         return PARGEMSLR_SUCCESS;
      }

      PARGEMSLR_CHKERR( A.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumColsLocal() != m );
      PARGEMSLR_CHKERR( Q.GetNumRowsLocal() != m );
      PARGEMSLR_CHKERR( clusters.GetLengthLocal() != m );

      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }

      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }

      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();

      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }

      if(clusters.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         clusters.MoveData(kMemoryHost);
      }

      // condition numbers not required
      // update the matrix
      char                    compq       = 'V';

      /* not working yet, the stable sort */
      clusters.Sort( order, false, true);

      iorder.Setup(m);

      for(i = 0 ; i < m ; i ++)
      {
         iorder[order[i]] = i;
      }

      i = 0;

      while(true)
      {
         idx = order[i];

         if(clusters[idx] <= 0)
         {
            break;
         }

         if(idx != i)
         {
            ifst = idx+1;
            ilst = i+1;
            PARGEMSLR_BLASLAPACK_ZTREXC(&compq, &m, PARGEMSLR_CAST( ccomplexd*, A.GetData()), &ldim_A,
                                       PARGEMSLR_CAST( ccomplexd*, Q.GetData()), &ldim_Q, &ifst, &ilst, &info); PARGEMSLR_CHKERR(info);

            /* after the swap, if we have order[k] = i, we need to update order[k] = idx */

            for(j = idx-1 ; j >= i ; j --)
            {
               temp_idx = iorder[j];
               order[temp_idx] = j+1;
               iorder[j+1] = temp_idx;
            }

            /* we'll not touch index i anymore, no need to update order[i] and iorder[i] */

         }
         i++;

         if( i >= m)
         {
            break;
         }

      }

      for(i = 0 ; i < m ; i ++)
      {
         w[i] = A(i, i);
      }

      //deallocate
      order.Clear();
      iorder.Clear();

      return PARGEMSLR_SUCCESS;
   }

   template <typename T>
   int DenseMatrixMatMatTemplate( const T &alpha, const DenseMatrixClass<T> &A, char transa, const DenseMatrixClass<T> &B, char transb, const T &beta, DenseMatrixClass<T> &C)
   {

      /* C = alpha*op(A)*op(B) + beta*C */
      int      i, j, k;
      T        *A_data, *B_data, *C_data, *A_data_i, *B_data_i, *C_data_i;
      T        *A_data_temp = NULL, *B_data_temp = NULL;
      T        temp_val;
      T        zero = T();
      T        one = T(1.0);
      int      A_ldim, B_ldim, C_ldim;
      int      A_nrow, B_nrow, C_nrow;
      int      A_ncol, B_ncol, C_ncol;

      A_data = A.GetData();
      B_data = B.GetData();
      C_data = C.GetData();

      A_ldim = A.GetLeadingDimension();
      B_ldim = B.GetLeadingDimension();
      C_ldim = C.GetLeadingDimension();

      A_nrow = A.GetNumRowsLocal();
      B_nrow = B.GetNumRowsLocal();
      C_nrow = C.GetNumRowsLocal();

      A_ncol = A.GetNumColsLocal();
      B_ncol = B.GetNumColsLocal();
      C_ncol = C.GetNumColsLocal();

      /* 1. Compute C = beta*C
       * note that if alpha != 0, when A == C we need to copy A,
       * when B == C we need to copy B
       * TODO: memcpy or omp parallel?
       */

      if(alpha != zero)
      {
         if(A_data == C_data)
         {
            PARGEMSLR_MALLOC(A_data_temp, A_nrow * A_ncol, kMemoryHost, T);
            for(i = 0 ; i < A_ncol ; i ++)
            {
                j = i * A_ldim;
                k = i * A_nrow;
                PARGEMSLR_MEMCPY(A_data_temp+k, A_data+j, A_nrow, kMemoryHost, kMemoryHost, T);
            }
            A_data = A_data_temp;
            A_ldim = A_nrow;
         }
         if(B_data == C_data)
         {
            PARGEMSLR_MALLOC(B_data_temp, B_nrow * B_ncol, kMemoryHost, T);
            for(i = 0 ; i < B_ncol ; i ++)
            {
                j = i * B_ldim;
                k = i * B_nrow;
                PARGEMSLR_MEMCPY(B_data_temp+k, B_data+j, B_nrow, kMemoryHost, kMemoryHost, T);
            }
            B_data = B_data_temp;
            B_ldim = B_nrow;
         }
      }

      /* now scale C */
      if(beta != one)
      {
         /* when beta == 1.0, C = C, do nothing */
         if(beta != zero)
         {
            /* C = beta*C */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
            for(i = 0 ; i < C_ncol ; i ++)
            {
               for(j = 0, k = i*C_ldim ; j < C_nrow ; j++, k++)
               {
                  C_data[k] *= beta;
               }
            }
         }
         else
         {
            /* beta == 0.0, C = 0 */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
            for(i = 0 ; i < C_ncol ; i ++)
            {
               for(j = 0, k = i*C_ldim ; j < C_nrow ; j++, k++)
               {
                  C_data[k] = zero;
               }
            }
         }
      }

      /* 2. the matmat C = alpha*op(A)*op(B) + C
       * when alpha == 0 we have C = C, do nothing
       */

      if(alpha != zero)
      {
         if(alpha != one)
         {
            if(transa == 'N')
            {
               if(transb == 'N')
               {
                  /* alpha*A*B
                   * Each time we pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_nrow ; k ++)
                     {
                        temp_val = alpha * B_data[j*B_ldim+k];
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A*B^T
                   * Each time we again pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_ncol ; k ++)
                     {
                        temp_val = alpha * B_data[k*B_ldim+j];
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A*B^H
                   * Each time we again pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_ncol ; k ++)
                     {
                        temp_val = alpha * PargemslrConj(B_data[k*B_ldim+j]);
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else
               {
                  PARGEMSLR_ERROR("Unknown matrix operator.");
                  return PARGEMSLR_ERROR_INVALED_PARAM;
               }
            }
            else if(transa == 'T')
            {
               if(transb == 'N')
               {
                  /* alpha*A^T*B
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, B_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     B_data_i = B_data + j*B_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * B_data_i[k];
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A^T*B^T
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * B_data[k*B_ldim+j];
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A^T*B^T
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * PargemslrConj(B_data[k*B_ldim+j]);
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else
               {
                  PARGEMSLR_ERROR("Unknown matrix operator.");
                  return PARGEMSLR_ERROR_INVALED_PARAM;
               }
            }
            else if(transa == 'C')
            {
               if(transb == 'N')
               {
                  /* alpha*A^C*B
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, B_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     B_data_i = B_data + j*B_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += PargemslrConj(A_data_i[k]) * B_data_i[k];
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A^C*B^T
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += PargemslrConj(A_data_i[k]) * B_data[k*B_ldim+j];
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A^C*B^T
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += PargemslrConj(A_data_i[k]) * PargemslrConj(B_data[k*B_ldim+j]);
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else
               {
                  PARGEMSLR_ERROR("Unknown matrix operator.");
                  return PARGEMSLR_ERROR_INVALED_PARAM;
               }
            }
            else
            {
               PARGEMSLR_ERROR("Unknown matrix operator.");
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
         else
         {
            if(transa == 'N')
            {
               if(transb == 'N')
               {
                  /* alpha*A*B
                   * Each time we pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_nrow ; k ++)
                     {
                        temp_val = B_data[j*B_ldim+k];
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A*B^T
                   * Each time we again pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_ncol ; k ++)
                     {
                        temp_val = B_data[k*B_ldim+j];
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A*B^H
                   * Each time we again pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_ncol ; k ++)
                     {
                        temp_val = PargemslrConj(B_data[k*B_ldim+j]);
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else
               {
                  PARGEMSLR_ERROR("Unknown matrix operator.");
                  return PARGEMSLR_ERROR_INVALED_PARAM;
               }
            }
            else if(transa == 'T')
            {
               if(transb == 'N')
               {
                  /* alpha*A^T*B
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, B_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     B_data_i = B_data + j*B_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * B_data_i[k];
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A^T*B^T
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * B_data[k*B_ldim+j];
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A^T*B^T
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * PargemslrConj(B_data[k*B_ldim+j]);
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else
               {
                  PARGEMSLR_ERROR("Unknown matrix operator.");
                  return PARGEMSLR_ERROR_INVALED_PARAM;
               }
            }
            else if(transa == 'C')
            {
               if(transb == 'N')
               {
                  /* alpha*A^C*B
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, B_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     B_data_i = B_data + j*B_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += PargemslrConj(A_data_i[k]) * B_data_i[k];
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A^C*B^T
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += PargemslrConj(A_data_i[k]) * B_data[k*B_ldim+j];
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A^C*B^T
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) PARGEMSLR_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;

                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += PargemslrConj(A_data_i[k]) * PargemslrConj(B_data[k*B_ldim+j]);
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else
               {
                  PARGEMSLR_ERROR("Unknown matrix operator.");
                  return PARGEMSLR_ERROR_INVALED_PARAM;
               }
            }
            else
            {
               PARGEMSLR_ERROR("Unknown matrix operator.");
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
      }

      if(A_data_temp)
      {
         PARGEMSLR_FREE( A_data_temp, kMemoryHost);
      }

      if(B_data_temp)
      {
         PARGEMSLR_FREE( B_data_temp, kMemoryHost);
      }

      return PARGEMSLR_SUCCESS;
   }
   template int DenseMatrixMatMatTemplate( const float &alpha, const DenseMatrixClass<float> &A, char transa, const DenseMatrixClass<float> &B, char transb, const float &beta, DenseMatrixClass<float> &C);
   template int DenseMatrixMatMatTemplate( const double &alpha, const DenseMatrixClass<double> &A, char transa, const DenseMatrixClass<double> &B, char transb, const double &beta, DenseMatrixClass<double> &C);
   template int DenseMatrixMatMatTemplate( const complexs &alpha, const DenseMatrixClass<complexs> &A, char transa, const DenseMatrixClass<complexs> &B, char transb, const complexs &beta, DenseMatrixClass<complexs> &C);
   template int DenseMatrixMatMatTemplate( const complexd &alpha, const DenseMatrixClass<complexd> &A, char transa, const DenseMatrixClass<complexd> &B, char transb, const complexd &beta, DenseMatrixClass<complexd> &C);

   int DenseMatrixMatMat( const float &alpha, const DenseMatrixClass<float> &A, char transa, const DenseMatrixClass<float> &B, char transb, const float &beta, DenseMatrixClass<float> &C)
   {
      int m, n, k;

      if(transa == 'N')
      {
         if(transb == 'N')
         {
            PARGEMSLR_CHKERR( A.GetNumColsLocal() != B.GetNumRowsLocal() );
         }
         else
         {
            PARGEMSLR_CHKERR( A.GetNumColsLocal() != B.GetNumColsLocal() );
         }
      }
      else
      {
         if(transb == 'N')
         {
            PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
         }
         else
         {
            PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumColsLocal());
         }
      }


      m = transa == 'N' ? A.GetNumRowsLocal() : A.GetNumColsLocal();
      n = transb == 'N' ? B.GetNumColsLocal() : B.GetNumRowsLocal();
      k = transa == 'N' ? A.GetNumColsLocal() : A.GetNumRowsLocal();

      if(C.GetNumRowsLocal() != m || C.GetNumColsLocal() != n)
      {
         C.Setup(m, n, A.GetDataLocation(), true);
      }

      if( k==0 )
      {
         /* In this case, A and B empty, C is not empty. Scale C. */
         C.Scale(beta);
         return PARGEMSLR_SUCCESS;
      }

      if( m==0 || n==0 )
      {
         /* In this case, C is also empty. */
         return PARGEMSLR_SUCCESS;
      }

#ifdef PARGEMSLR_CUDA

      int loc_A = A.GetDataLocation();
      int loc_B = B.GetDataLocation();
      int loc_C = C.GetDataLocation();

      switch(loc_A)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matmat on device */
            PARGEMSLR_CHKERR( loc_B == kMemoryHost || loc_B == kMemoryPinned );
            PARGEMSLR_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );

            return DenseMatrixSMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);

            break;
         }
         case kMemoryUnified:
         {
            /* typically matrices should not be on the unified memory */
            switch(loc_B)
            {
               case kMemoryDevice:
               {
                  /* B is on device, need to apply matmat on device */
                  PARGEMSLR_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );

                  return DenseMatrixSMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);

                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_C == kMemoryDevice || loc_C == kMemoryUnified )
                  {
                     /* apply matmat on device */
                     return DenseMatrixSMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  }
                  /* otherwise on host */
                  break;
               }
               default:
               {
                  /* matmat on host */
                  PARGEMSLR_CHKERR( loc_C == kMemoryDevice );
                  break;
               }
            }
            break;
         }
         default:
         {
            /* matvec on host */
            PARGEMSLR_CHKERR( loc_B == kMemoryDevice || loc_C == kMemoryDevice );
            break;
         }
      }

#endif

#ifdef PARGEMSLR_BLAS

      int ldim_A, ldim_B, ldim_C;
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();

      PARGEMSLR_BLASLAPACK_SGEMM( &transa, &transb, &m, &n, &k, &alpha, A.GetData(), &ldim_A, B.GetData(), &ldim_B, &beta, C.GetData(), &ldim_C);
#else
      DenseMatrixMatMatTemplate(alpha, A, transa, B, transb, beta, C);
#endif
      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixMatMat( const double &alpha, const DenseMatrixClass<double> &A, char transa, const DenseMatrixClass<double> &B, char transb, const double &beta, DenseMatrixClass<double> &C)
   {
      int m, n, k;

      if(transa == 'N')
      {
         if(transb == 'N')
         {
            PARGEMSLR_CHKERR( A.GetNumColsLocal() != B.GetNumRowsLocal() );
         }
         else
         {
            PARGEMSLR_CHKERR( A.GetNumColsLocal() != B.GetNumColsLocal() );
         }
      }
      else
      {
         if(transb == 'N')
         {
            PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
         }
         else
         {
            PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumColsLocal());
         }
      }


      m = transa == 'N' ? A.GetNumRowsLocal() : A.GetNumColsLocal();
      n = transb == 'N' ? B.GetNumColsLocal() : B.GetNumRowsLocal();
      k = transa == 'N' ? A.GetNumColsLocal() : A.GetNumRowsLocal();

      if(C.GetNumRowsLocal() != m || C.GetNumColsLocal() != n)
      {
         C.Setup(m, n, A.GetDataLocation(), true);
      }

      if( k==0 )
      {
         /* In this case, A and B empty, C is not empty. Scale C. */
         C.Scale(beta);
         return PARGEMSLR_SUCCESS;
      }

      if( m==0 || n==0 )
      {
         /* In this case, C is also empty. */
         return PARGEMSLR_SUCCESS;
      }

#ifdef PARGEMSLR_CUDA

      int loc_A = A.GetDataLocation();
      int loc_B = B.GetDataLocation();
      int loc_C = C.GetDataLocation();

      switch(loc_A)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matmat on device */
            PARGEMSLR_CHKERR( loc_B == kMemoryHost || loc_B == kMemoryPinned );
            PARGEMSLR_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );

            return DenseMatrixDMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);

            break;
         }
         case kMemoryUnified:
         {
            /* typically matrices should not be on the unified memory */
            switch(loc_B)
            {
               case kMemoryDevice:
               {
                  /* B is on device, need to apply matmat on device */
                  PARGEMSLR_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );

                  return DenseMatrixDMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);

                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_C == kMemoryDevice || loc_C == kMemoryUnified )
                  {
                     /* apply matmat on device */
                     return DenseMatrixDMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  }
                  /* otherwise on host */
                  break;
               }
               default:
               {
                  /* matmat on host */
                  PARGEMSLR_CHKERR( loc_C == kMemoryDevice );
                  break;
               }
            }
            break;
         }
         default:
         {
            /* matvec on host */
            PARGEMSLR_CHKERR( loc_B == kMemoryDevice || loc_C == kMemoryDevice );
            break;
         }
      }

#endif

#ifdef PARGEMSLR_BLAS

      int ldim_A, ldim_B, ldim_C;
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();

      PARGEMSLR_BLASLAPACK_DGEMM( &transa, &transb, &m, &n, &k, &alpha, A.GetData(), &ldim_A, B.GetData(), &ldim_B, &beta, C.GetData(), &ldim_C);
#else
      DenseMatrixMatMatTemplate(alpha, A, transa, B, transb, beta, C);
#endif

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixMatMat( const complexs &alpha, const DenseMatrixClass<complexs> &A, char transa, const DenseMatrixClass<complexs> &B, char transb, const complexs &beta, DenseMatrixClass<complexs> &C)
   {
      int m, n, k;

      if(transa == 'N')
      {
         if(transb == 'N')
         {
            PARGEMSLR_CHKERR( A.GetNumColsLocal() != B.GetNumRowsLocal() );
         }
         else
         {
            PARGEMSLR_CHKERR( A.GetNumColsLocal() != B.GetNumColsLocal() );
         }
      }
      else
      {
         if(transb == 'N')
         {
            PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
         }
         else
         {
            PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumColsLocal());
         }
      }

      m = transa == 'N' ? A.GetNumRowsLocal() : A.GetNumColsLocal();
      n = transb == 'N' ? B.GetNumColsLocal() : B.GetNumRowsLocal();
      k = transa == 'N' ? A.GetNumColsLocal() : A.GetNumRowsLocal();

      if(C.GetNumRowsLocal() != m || C.GetNumColsLocal() != n)
      {
         C.Setup(m, n, A.GetDataLocation(), true);
      }

      if( k==0 )
      {
         /* In this case, A and B empty, C is not empty. Scale C. */
         C.Scale(beta);
         return PARGEMSLR_SUCCESS;
      }

      if( m==0 || n==0 )
      {
         /* In this case, C is also empty. */
         return PARGEMSLR_SUCCESS;
      }

#ifdef PARGEMSLR_CUDA

      int loc_A = A.GetDataLocation();
      int loc_B = B.GetDataLocation();
      int loc_C = C.GetDataLocation();

      switch(loc_A)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matmat on device */
            PARGEMSLR_CHKERR( loc_B == kMemoryHost || loc_B == kMemoryPinned );
            PARGEMSLR_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );

            return DenseMatrixCMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);

            break;
         }
         case kMemoryUnified:
         {
            /* typically matrices should not be on the unified memory */
            switch(loc_B)
            {
               case kMemoryDevice:
               {
                  /* B is on device, need to apply matmat on device */
                  PARGEMSLR_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );

                  return DenseMatrixCMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);

                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_C == kMemoryDevice || loc_C == kMemoryUnified )
                  {
                     /* apply matmat on device */
                     return DenseMatrixCMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  }
                  /* otherwise on host */
                  break;
               }
               default:
               {
                  /* matmat on host */
                  PARGEMSLR_CHKERR( loc_C == kMemoryDevice );
                  break;
               }
            }
            break;
         }
         default:
         {
            /* matvec on host */
            PARGEMSLR_CHKERR( loc_B == kMemoryDevice || loc_C == kMemoryDevice );
            break;
         }
      }

#endif

#ifdef PARGEMSLR_BLAS

      int ldim_A, ldim_B, ldim_C;
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();

      PARGEMSLR_BLASLAPACK_CGEMM( &transa, &transb, &m, &n, &k,
                                 PARGEMSLR_CAST( const ccomplexs*, &alpha), PARGEMSLR_CAST( const ccomplexs*, A.GetData()), &ldim_A,
                                 PARGEMSLR_CAST( const ccomplexs*, B.GetData()), &ldim_B,
                                 PARGEMSLR_CAST( const ccomplexs*, &beta), PARGEMSLR_CAST( ccomplexs*, C.GetData()), &ldim_C);
#else
      DenseMatrixMatMatTemplate(alpha, A, transa, B, transb, beta, C);
#endif

      return PARGEMSLR_SUCCESS;
   }

   int DenseMatrixMatMat( const complexd &alpha, const DenseMatrixClass<complexd> &A, char transa, const DenseMatrixClass<complexd> &B, char transb, const complexd &beta, DenseMatrixClass<complexd> &C)
   {
      int m, n, k;

      if(transa == 'N')
      {
         if(transb == 'N')
         {
            PARGEMSLR_CHKERR( A.GetNumColsLocal() != B.GetNumRowsLocal() );
         }
         else
         {
            PARGEMSLR_CHKERR( A.GetNumColsLocal() != B.GetNumColsLocal() );
         }
      }
      else
      {
         if(transb == 'N')
         {
            PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
         }
         else
         {
            PARGEMSLR_CHKERR(A.GetNumRowsLocal() != B.GetNumColsLocal());
         }
      }


      m = transa == 'N' ? A.GetNumRowsLocal() : A.GetNumColsLocal();
      n = transb == 'N' ? B.GetNumColsLocal() : B.GetNumRowsLocal();
      k = transa == 'N' ? A.GetNumColsLocal() : A.GetNumRowsLocal();

      if(C.GetNumRowsLocal() != m || C.GetNumColsLocal() != n)
      {
         C.Setup(m, n, A.GetDataLocation(), true);
      }

      if( k==0 )
      {
         /* In this case, A and B empty, C is not empty. Scale C. */
         C.Scale(beta);
         return PARGEMSLR_SUCCESS;
      }

      if( m==0 || n==0 )
      {
         /* In this case, C is also empty. */
         return PARGEMSLR_SUCCESS;
      }

#ifdef PARGEMSLR_CUDA

      int loc_A = A.GetDataLocation();
      int loc_B = B.GetDataLocation();
      int loc_C = C.GetDataLocation();

      switch(loc_A)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matmat on device */
            PARGEMSLR_CHKERR( loc_B == kMemoryHost || loc_B == kMemoryPinned );
            PARGEMSLR_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );

            return DenseMatrixZMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);

            break;
         }
         case kMemoryUnified:
         {
            /* typically matrices should not be on the unified memory */
            switch(loc_B)
            {
               case kMemoryDevice:
               {
                  /* B is on device, need to apply matmat on device */
                  PARGEMSLR_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );

                  return DenseMatrixZMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);

                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_C == kMemoryDevice || loc_C == kMemoryUnified )
                  {
                     /* apply matmat on device */
                     return DenseMatrixZMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  }
                  /* otherwise on host */
                  break;
               }
               default:
               {
                  /* matmat on host */
                  PARGEMSLR_CHKERR( loc_C == kMemoryDevice );
                  break;
               }
            }
            break;
         }
         default:
         {
            /* matvec on host */
            PARGEMSLR_CHKERR( loc_B == kMemoryDevice || loc_C == kMemoryDevice );
            break;
         }
      }

#endif

#ifdef PARGEMSLR_BLAS

      int ldim_A, ldim_B, ldim_C;
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();

      PARGEMSLR_BLASLAPACK_ZGEMM( &transa, &transb, &m, &n, &k,
                                 PARGEMSLR_CAST( const ccomplexd*, &alpha), PARGEMSLR_CAST( const ccomplexd*, A.GetData()), &ldim_A,
                                 PARGEMSLR_CAST( const ccomplexd*, B.GetData()), &ldim_B,
                                 PARGEMSLR_CAST( const ccomplexd*, &beta), PARGEMSLR_CAST( ccomplexd*, C.GetData()), &ldim_C);
#else
      DenseMatrixMatMatTemplate(alpha, A, transa, B, transb, beta, C);
#endif

      return PARGEMSLR_SUCCESS;
   }
}
