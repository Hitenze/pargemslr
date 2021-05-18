
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../utils/memory.hpp"
#include "../utils/protos.hpp"
#include "../utils/mmio.hpp"
#include "vector.hpp"
#include "sequential_vector.hpp"
#include "parallel_vector.hpp"
#include "vectorops.hpp"

#include <iostream>
#include <complex>

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
   
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<int> &vec)
   {
      return kInt;
   }
   
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<long int> &vec)
   {
      return kLongInt;
   }
   
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<float> &vec)
   {
      return kSingleReal;
   }
   
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<double> &vec)
   {
      return kDoubleReal;
   }
   
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<complexs> &vec)
   {
      return kSingleComplex;
   }
   
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<complexd> &vec)
   {
      return kDoubleComplex;
   }
   
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<int> *vec)
   {
      return kInt;
   }
   
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<long int> *vec)
   {
      return kLongInt;
   }
   
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<float> *vec)
   {
      return kSingleReal;
   }
   
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<double> *vec)
   {
      return kDoubleReal;
   }
   
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<complexs> *vec)
   {
      return kSingleComplex;
   }
   
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<complexd> *vec)
   {
      return kDoubleComplex;
   }
   
   template <typename T>
   int VectorPDotTemplate( int n, const T *x, const T *y, T &t )
   {
      int i;
      t = 0.0;
      if(PargemslrIsComplex<T>::value)
      {
         /* complex dot */
#ifdef PARGEMSLR_OPENMP
         int num_threads = PargemslrGetOpenmpMaxNumThreads();
         if(num_threads > 1)
         {
#pragma omp parallel for private(i) reduction(+:t) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
            for( i = 0 ; i < n ; i ++)
            {
               t += PargemslrConj(x[i]) * y[i];
            }
         }
         else
         {
#endif
            for( i = 0 ; i < n ; i ++)
            {
               t += PargemslrConj(x[i]) * y[i];
            }
#ifdef PARGEMSLR_OPENMP
         }
#endif
      }
      else
      {
         /* real dot */
#ifdef PARGEMSLR_OPENMP
         int num_threads = PargemslrGetOpenmpMaxNumThreads();
         if(num_threads > 1)
         {
#pragma omp parallel for private(i) reduction(+:t) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
            for( i = 0 ; i < n ; i ++)
            {
               t += x[i] * y[i];
            }
         }
         else
         {
#endif
            for( i = 0 ; i < n ; i ++)
            {
               t += x[i] * y[i];
            }
#ifdef PARGEMSLR_OPENMP
         }
#endif
      }
      return PARGEMSLR_SUCCESS;
   }
   template int VectorPDotTemplate( int n, const float *x, const float *y, float &t );
   template int VectorPDotTemplate( int n, const double *x, const double *y, double &t );
   template int VectorPDotTemplate( int n, const complexs *x, const complexs *y, complexs &t );
   template int VectorPDotTemplate( int n, const complexd *x, const complexd *y, complexd &t );
   
   int VectorDot( const vector_base_float &x, const vector_base_float &y, float &t)
   {

      PARGEMSLR_CHKERR(x.GetLengthLocal() != y.GetLengthLocal());
      PARGEMSLR_CHKERR(x.GetLengthGlobal() != y.GetLengthGlobal());
      
      int  n         = x.GetLengthLocal();
      float *x_data  = x.GetData();
      float *y_data  = y.GetData();
      
      t = 0.0;
      if(n > 0 )
      {
#ifdef PARGEMSLR_CUDA
         int   x_loc = x.GetDataLocation();
         int   y_loc = y.GetDataLocation();
         switch(x_loc)
         {
            case kMemoryDevice:
            {
               /* on device */
               PARGEMSLR_CHKERR(y_loc == kMemoryHost || y_loc == kMemoryPinned);
               VectorSdotDevice(x, y, t);
               
               if(x.IsParallel())
               {
                  /* this is a parallel vector, communication required */
                  float t_global;
                  PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
                  t = t_global;
               }
               
               return PARGEMSLR_SUCCESS;
               break;
            }
            case kMemoryUnified:
            {
               if(y_loc == kMemoryUnified || y_loc == kMemoryDevice)
               {
                  /* on device */
                  VectorSdotDevice(x, y, t);
                  
                  if(x.IsParallel())
                  {
                     /* this is a parallel vector, communication required */
                     float t_global;
                     PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
                     t = t_global;
                  }
                  
                  return PARGEMSLR_SUCCESS;
               }
               /* otherwise on host */
               break;
            }
            default:
            {
               /* on host */
               break;
            }
         }
#endif
#ifdef PARGEMSLR_BLAS
         int one = 1;
         t = PARGEMSLR_BLASLAPACK_SDOT( &n, x_data, &one, y_data, &one);
#else
         VectorPDotTemplate(n, x_data, y_data, t);
#endif
      }
      
      if(x.IsParallel())
      {
         /* this is a parallel vector, communication required */
         float t_global;
         PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
         t = t_global;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorDot( const vector_base_double &x, const vector_base_double &y, double &t)
   {

      PARGEMSLR_CHKERR(x.GetLengthLocal() != y.GetLengthLocal());

      int  n         = x.GetLengthLocal();
      double *x_data  = x.GetData();
      double *y_data  = y.GetData();
      
      t = 0.0;
      if(n > 0 )
      {
#ifdef PARGEMSLR_CUDA
         int   x_loc = x.GetDataLocation();
         int   y_loc = y.GetDataLocation();
         switch(x_loc)
         {
            case kMemoryDevice:
            {
               /* on device */
               PARGEMSLR_CHKERR(y_loc == kMemoryHost || y_loc == kMemoryPinned);
               VectorDdotDevice(x, y, t);
               
               if(x.IsParallel())
               {
                  /* this is a parallel vector, communication required */
                  double t_global;
                  PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
                  t = t_global;
               }
               
               return PARGEMSLR_SUCCESS;
               break;
            }
            case kMemoryUnified:
            {
               if(y_loc == kMemoryUnified || y_loc == kMemoryDevice)
               {
                  /* on device */
                  VectorDdotDevice(x, y, t);
                  
                  if(x.IsParallel())
                  {
                     /* this is a parallel vector, communication required */
                     double t_global;
                     PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
                     t = t_global;
                  }
                  
                  return PARGEMSLR_SUCCESS;
               }
               /* otherwise on host */
               break;
            }
            default:
            {
               /* on host */
               break;
            }
         }
#endif
#ifdef PARGEMSLR_BLAS
         int one = 1;
         t = PARGEMSLR_BLASLAPACK_DDOT( &n, x_data, &one, y_data, &one);
#else
         VectorPDotTemplate(n, x_data, y_data, t);
#endif
      }
      
      if(x.IsParallel())
      {
         /* this is a parallel vector, communication required */
         double t_global;
         PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
         t = t_global;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorDot( const vector_base_complexs &x, const vector_base_complexs &y, complexs &t)
   {

      PARGEMSLR_CHKERR(x.GetLengthLocal() != y.GetLengthLocal());

      int  n         = x.GetLengthLocal();
      complexs *x_data  = x.GetData();
      complexs *y_data  = y.GetData();
      
      t = 0.0;
      if(n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int   x_loc = x.GetDataLocation();
         int   y_loc = y.GetDataLocation();
         switch(x_loc)
         {
            case kMemoryDevice:
            {
               /* on device */
               PARGEMSLR_CHKERR(y_loc == kMemoryHost || y_loc == kMemoryPinned);
               VectorCdotDevice(x, y, t);
               
               if(x.IsParallel())
               {
                  /* this is a parallel vector, communication required */
                  complexs t_global;
                  PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
                  t = t_global;
               }
               
               return PARGEMSLR_SUCCESS;
               break;
            }
            case kMemoryUnified:
            {
               if(y_loc == kMemoryUnified || y_loc == kMemoryDevice)
               {
                  /* on device */
                  VectorCdotDevice(x, y, t);
                  
                  if(x.IsParallel())
                  {
                     /* this is a parallel vector, communication required */
                     complexs t_global;
                     PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
                     t = t_global;
                  }
                  
                  return PARGEMSLR_SUCCESS;
               }
               /* otherwise on host */
               break;
            }
            default:
            {
               /* on host */
               break;
            }
         }
#endif
#ifdef PARGEMSLR_BLAS
         int one = 1;
         ccomplexs t1;
         t1.real = 0.0;
         t1.imag = 0.0;
#ifdef __APPLE__
         PARGEMSLR_BLASLAPACK_CDOTC( &t1, &n, PARGEMSLR_CAST( ccomplexs*, x_data), &one, PARGEMSLR_CAST( ccomplexs*, y_data), &one);
#else
         t1 = PARGEMSLR_BLASLAPACK_CDOTC( &n, PARGEMSLR_CAST( ccomplexs*, x_data), &one, PARGEMSLR_CAST( ccomplexs*, y_data), &one);
#endif
         t = complexs(t1.real, t1.imag);
#else
         VectorPDotTemplate(n, x_data, y_data, t);
#endif
      }
      
      if(x.IsParallel())
      {
         /* this is a parallel vector, communication required */
         complexs t_global;
         PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
         t = t_global;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorDot( const vector_base_complexd &x, const vector_base_complexd &y, complexd &t) 
   {

      PARGEMSLR_CHKERR(x.GetLengthLocal() != y.GetLengthLocal());
      
      int      n        = x.GetLengthLocal();
      complexd *x_data  = x.GetData();
      complexd *y_data  = y.GetData();
      
      t = 0.0;
      if(n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int   x_loc = x.GetDataLocation();
         int   y_loc = y.GetDataLocation();
         switch(x_loc)
         {
            case kMemoryDevice:
            {
               /* on device */
               PARGEMSLR_CHKERR(y_loc == kMemoryHost || y_loc == kMemoryPinned);
               VectorZdotDevice(x, y, t);
               
               if(x.IsParallel())
               {
                  /* this is a parallel vector, communication required */
                  complexd t_global;
                  PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
                  t = t_global;
               }
               
               return PARGEMSLR_SUCCESS;
               break;
            }
            case kMemoryUnified:
            {
               if(y_loc == kMemoryUnified || y_loc == kMemoryDevice)
               {
                  /* on device */
                  VectorZdotDevice(x, y, t);
                  
                  if(x.IsParallel())
                  {
                     /* this is a parallel vector, communication required */
                     complexd t_global;
                     PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
                     t = t_global;
                  }
                  
                  return PARGEMSLR_SUCCESS;
               }
               /* otherwise on host */
               break;
            }
            default:
            {
               /* on host */
               break;
            }
         }
#endif
#ifdef PARGEMSLR_BLAS
         int one = 1;
         ccomplexd t1;
         t1.real = 0.0;
         t1.imag = 0.0;
#ifdef __APPLE__
         PARGEMSLR_BLASLAPACK_ZDOTC( &t1, &n, PARGEMSLR_CAST( ccomplexd*, x_data), &one, PARGEMSLR_CAST( ccomplexd*, y_data), &one);
#else
         t1 = PARGEMSLR_BLASLAPACK_ZDOTC( &n, PARGEMSLR_CAST( ccomplexd*, x_data), &one, PARGEMSLR_CAST( ccomplexd*, y_data), &one);
#endif
         t = complexd(t1.real, t1.imag);
#else
         VectorPDotTemplate(n, x_data, y_data, t);
#endif
      }
      
      if(x.IsParallel())
      {
         /* this is a parallel vector, communication required */
         complexd t_global;
         PargemslrMpiAllreduce( &t, &t_global, 1, MPI_SUM, x.GetComm());
         t = t_global;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   
   template <typename T>
   int VectorPScaleTemplate( int n, T *x, const T&a)
   {
      int i;
#ifdef PARGEMSLR_OPENMP
      int num_threads = PargemslrGetOpenmpMaxNumThreads();
      if(num_threads > 1)
      {
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
         for( i = 0 ; i < n ; i ++)
         {
            x[i] *= a;
         }
      }
      else
      {
#endif
         for( i = 0 ; i < n ; i ++)
         {
            x[i] *= a;
         }
#ifdef PARGEMSLR_OPENMP
      }
#endif
      return PARGEMSLR_SUCCESS;
   }
   template int VectorPScaleTemplate( int n, float *x, const float &a);
   template int VectorPScaleTemplate( int n, double *x, const double &a);
   template int VectorPScaleTemplate( int n, complexs *x, const complexs &a);
   template int VectorPScaleTemplate( int n, complexd *x, const complexd &a);
   
   int VectorScale(VectorClass<float> &x, const float &a) 
   {
      int n   = x.GetLengthLocal();
      
      if(n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int location = x.GetDataLocation();
         if( location == kMemoryDevice || location == kMemoryUnified)
         {
            VectorSscaleDevice(x, a);
            return PARGEMSLR_SUCCESS;
         }
#endif
         if(a == 0.0)
         {
            return x.Fill(0.0);
         }
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_SSCAL(&n, &a, x.GetData(), &one);
#else
         VectorPScaleTemplate( n, x.GetData(), a);
#endif
      }
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorScale(VectorClass<double> &x, const double &a) 
   {
      int n   = x.GetLengthLocal();
      
      if(n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int location = x.GetDataLocation();
         if( location == kMemoryDevice || location == kMemoryUnified)
         {
            VectorDscaleDevice(x, a);
            return PARGEMSLR_SUCCESS;
         }
#endif
         if(a == 0.0)
         {
            return x.Fill(0.0);
         }
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_DSCAL(&n, &a, x.GetData(), &one);
#else
         VectorPScaleTemplate( n, x.GetData(), a);
#endif
      }
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorScale(VectorClass<complexs> &x, const complexs &a) 
   {
      int n   = x.GetLengthLocal();
      
      if(n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int location = x.GetDataLocation();
         if( location == kMemoryDevice || location == kMemoryUnified)
         {
            VectorCscaleDevice(x, a);
            return PARGEMSLR_SUCCESS;
         }
#endif
         if(a == 0.0f)
         {
            return x.Fill(0.0f);
         }
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_CSCAL(&n, PARGEMSLR_CAST( const ccomplexs*, &a), PARGEMSLR_CAST( ccomplexs*, x.GetData()), &one);
#else
         VectorPScaleTemplate( n, x.GetData(), a);
#endif
      }
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorScale(VectorClass<complexd> &x, const complexd &a) 
   {
      int n   = x.GetLengthLocal();
      
      if(n > 0)
      {
#ifdef PARGEMSLR_CUDA
         int location = x.GetDataLocation();
         if( location == kMemoryDevice || location == kMemoryUnified)
         {
            VectorZscaleDevice(x, a);
            return PARGEMSLR_SUCCESS;
         }
#endif
         if(a == 0.0)
         {
            return x.Fill(0.0);
         }
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_ZSCAL(&n, PARGEMSLR_CAST( const ccomplexd*, &a), PARGEMSLR_CAST( ccomplexd*, x.GetData()), &one);
#else
         VectorPScaleTemplate( n, x.GetData(), a);
#endif
      }
      return PARGEMSLR_SUCCESS;
   }
   
   template <typename T>
   int VectorPAxpyTemplate( int n, const T &a, const T *x, T *y)
   {
      int i;
      
#ifdef PARGEMSLR_OPENMP
      int num_threads = PargemslrGetOpenmpMaxNumThreads();
      if(num_threads > 1)
      {
         if(a == T(1.0))
         {
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
            for( i = 0 ; i < n ; i ++)
            {
               y[i] += x[i];
            }
         }
         else
         {
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
            for( i = 0 ; i < n ; i ++)
            {
               y[i] += a * x[i];
            }
         }
      }
      else
      {
#endif
         if(a == T(1.0))
         {
            for( i = 0 ; i < n ; i ++)
            {
               y[i] += x[i];
            }
         }
         else
         {
            for( i = 0 ; i < n ; i ++)
            {
               y[i] += a * x[i];
            }
         }
         
#ifdef PARGEMSLR_OPENMP
      }
#endif
      return PARGEMSLR_SUCCESS;
   }
   template int VectorPAxpyTemplate( int n, const float &a, const float *x, float *y);
   template int VectorPAxpyTemplate( int n, const double &a, const double *x, double *y);
   template int VectorPAxpyTemplate( int n, const complexs &a, const complexs *x, complexs *y);
   template int VectorPAxpyTemplate( int n, const complexd &a, const complexd *x, complexd *y);
   
   int VectorAxpy( const float &a, const VectorClass<float> &x, VectorClass<float> &y)
   {

      PARGEMSLR_CHKERR(x.GetLengthLocal() != y.GetLengthLocal());
      
      int n   = x.GetLengthLocal();
      if(n > 0)
      {
         if(x.GetData() == y.GetData())
         {
            PARGEMSLR_ERROR("Currently we don't support in place Axpy.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
#ifdef PARGEMSLR_CUDA
         int x_loc = x.GetDataLocation();
         int y_loc = y.GetDataLocation();
         switch(x_loc)
         {
            case kMemoryDevice:
            {
               PARGEMSLR_CHKERR(y_loc == kMemoryHost || y_loc == kMemoryPinned);
               VectorSaxpyDevice(a, x, y);
               return PARGEMSLR_SUCCESS;
               break;
            }
            case kMemoryUnified:
            {
               if(y_loc == kMemoryUnified || y_loc == kMemoryDevice)
               {
                  /* in this case matvec on device */
                  VectorSaxpyDevice(a, x, y);
                  return PARGEMSLR_SUCCESS;
               }
               break;
            }
            default:
            {
               /* otherwise on host */
               PARGEMSLR_CHKERR(y_loc == kMemoryDevice);
               break;
            }
         }
#endif
         float *x_data = x.GetData();
         float *y_data = y.GetData();
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_SAXPY(&n, &a, x_data, &one, y_data, &one);
#else
         VectorPAxpyTemplate( n, a, x_data, y_data);
#endif
      }
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorAxpy( const double &a, const VectorClass<double> &x, VectorClass<double> &y) 
   {

      PARGEMSLR_CHKERR(x.GetLengthLocal() != y.GetLengthLocal());

      int n   = x.GetLengthLocal();
      if(n > 0)
      {
         if(x.GetData() == y.GetData())
         {
            PARGEMSLR_ERROR("Currently we don't support in place Axpy.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
#ifdef PARGEMSLR_CUDA
         int x_loc = x.GetDataLocation();
         int y_loc = y.GetDataLocation();
         switch(x_loc)
         {
            case kMemoryDevice:
            {
               PARGEMSLR_CHKERR(y_loc == kMemoryHost || y_loc == kMemoryPinned);
               VectorDaxpyDevice(a, x, y);
               return PARGEMSLR_SUCCESS;
               break;
            }
            case kMemoryUnified:
            {
               if(y_loc == kMemoryUnified || y_loc == kMemoryDevice)
               {
                  /* in this case matvec on device */
                  VectorDaxpyDevice(a, x, y);
                  return PARGEMSLR_SUCCESS;
               }
               break;
            }
            default:
            {
               /* otherwise on host */
               PARGEMSLR_CHKERR(y_loc == kMemoryDevice);
               break;
            }
         }
#endif
         double *x_data = x.GetData();
         double *y_data = y.GetData();
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_DAXPY(&n, &a, x_data, &one, y_data, &one);
#else
         VectorPAxpyTemplate( n, a, x_data, y_data);
#endif
      }
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorAxpy( const complexs &a, const VectorClass<complexs> &x, VectorClass<complexs> &y)
   {

      PARGEMSLR_CHKERR(x.GetLengthLocal() != y.GetLengthLocal());

      int n   = x.GetLengthLocal();
      if(n > 0)
      {
         if(x.GetData() == y.GetData())
         {
            PARGEMSLR_ERROR("Currently we don't support in place Axpy.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
#ifdef PARGEMSLR_CUDA
         int x_loc = x.GetDataLocation();
         int y_loc = y.GetDataLocation();
         switch(x_loc)
         {
            case kMemoryDevice:
            {
               PARGEMSLR_CHKERR(y_loc == kMemoryHost || y_loc == kMemoryPinned);
               VectorCaxpyDevice(a, x, y);
               return PARGEMSLR_SUCCESS;
               break;
            }
            case kMemoryUnified:
            {
               if(y_loc == kMemoryUnified || y_loc == kMemoryDevice)
               {
                  /* in this case matvec on device */
                  VectorCaxpyDevice(a, x, y);
                  return PARGEMSLR_SUCCESS;
               }
               break;
            }
            default:
            {
               /* otherwise on host */
               PARGEMSLR_CHKERR(y_loc == kMemoryDevice);
               break;
            }
         }
#endif
         complexs *x_data = x.GetData();
         complexs *y_data = y.GetData();
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_CAXPY(&n, PARGEMSLR_CAST( const ccomplexs*, &a), PARGEMSLR_CAST( const ccomplexs*, x_data), &one, PARGEMSLR_CAST( ccomplexs*, y_data), &one);
#else
         VectorPAxpyTemplate( n, a, x_data, y_data);
#endif
      }
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorAxpy( const complexd &a, const VectorClass<complexd> &x, VectorClass<complexd> &y)
   {

      PARGEMSLR_CHKERR(x.GetLengthLocal() != y.GetLengthLocal());

      int n   = x.GetLengthLocal();
      if(n > 0)
      {
         if(x.GetData() == y.GetData())
         {
            PARGEMSLR_ERROR("Currently we don't support in place Axpy.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
#ifdef PARGEMSLR_CUDA
         int x_loc = x.GetDataLocation();
         int y_loc = y.GetDataLocation();
         switch(x_loc)
         {
            case kMemoryDevice:
            {
               PARGEMSLR_CHKERR(y_loc == kMemoryHost || y_loc == kMemoryPinned);
               VectorZaxpyDevice(a, x, y);
               return PARGEMSLR_SUCCESS;
               break;
            }
            case kMemoryUnified:
            {
               if(y_loc == kMemoryUnified || y_loc == kMemoryDevice)
               {
                  /* in this case matvec on device */
                  VectorZaxpyDevice(a, x, y);
                  return PARGEMSLR_SUCCESS;
               }
               break;
            }
            default:
            {
               /* otherwise on host */
               PARGEMSLR_CHKERR(y_loc == kMemoryDevice);
               break;
            }
         }
#endif
         complexd *x_data = x.GetData();
         complexd *y_data = y.GetData();
#ifdef PARGEMSLR_BLAS
         int one = 1;
         PARGEMSLR_BLASLAPACK_ZAXPY(&n, PARGEMSLR_CAST( const ccomplexd*, &a), PARGEMSLR_CAST( const ccomplexd*, x_data), &one, PARGEMSLR_CAST( ccomplexd*, y_data), &one);
#else
         VectorPAxpyTemplate( n, a, x_data, y_data);
#endif
      }
      return PARGEMSLR_SUCCESS;
   }
   
   template <typename T>
   int VectorPBsearchHost( const T *v, const T &val, int s, int e, int &idx, bool ascending)
   {
      /* No OpenMP implementation */
      /* mid */
      int m;
      
      /* main loop */
      if(!ascending)
      {
         while (s <= e)
         {
            m = s + (e - s)/ 2;
            if(val < v[m])
            {
               s = m + 1;
            }
            else if(val > v[m])
            {
               e = m - 1;
            }
            else
            {
               /* get it */
               idx = m;
               return m;
            }
         }
         /* reach here is not found 
          * at the beginning of the last loop, s == e or s == e - 1
          * m == s
          * if val < v[m], val should be inserted after v[m], that is s
          * if val > v[m], val should be inserted at v[m], that is still s
          */
         idx = s;
      }
      else
      {
         while (s <= e)
         {
            m = s + (e - s)/ 2;
            if(val < v[m])
            {
               e = m - 1;
            }
            else if(val > v[m])
            {
               s = m + 1;
            }
            else
            {
               /* get it */
               idx = m;
               return m;
            }
         }
         /* reach here is not found 
          * at the beginning of the last loop, s == e or s == e - 1
          * m == s
          * if val < v[m], val should be inserted at v[m], that is s
          * if val > v[m], val should be inserted after v[m], that is still s
          */
         idx = s;
      }
      return -1;
   }
   template int VectorPBsearchHost(const int *v, const int &val, int s, int e, int &idx, bool ascending);
   template int VectorPBsearchHost(const long int *v, const long int &val, int s, int e, int &idx, bool ascending);
   template int VectorPBsearchHost(const float *v, const float &val, int s, int e, int &idx, bool ascending);
   template int VectorPBsearchHost(const double *v, const double &val, int s, int e, int &idx, bool ascending);
   
   template <typename T>
   int VectorPlotHost( const VectorVirtualClass<T> &x, int conditiona, int conditionb, int width)
   {
      
      if(conditiona==conditionb)
      {
         int      i, length;
         T        *data;
         
         length = x.GetLengthLocal();
         data = x.GetData();
         
         for(i = 0 ; i < length ; i ++)
         {
            PargemslrPrintValueHost(data[i], width);
            std::cout<<", ";
         }
         std::cout<<std::endl;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int VectorPlotHost( const VectorVirtualClass<int> &x, int conditiona, int conditionb, int width);
   template int VectorPlotHost( const VectorVirtualClass<long int> &x, int conditiona, int conditionb, int width);
   template int VectorPlotHost( const VectorVirtualClass<float> &x, int conditiona, int conditionb, int width);
   template int VectorPlotHost( const VectorVirtualClass<double> &x, int conditiona, int conditionb, int width);
   template int VectorPlotHost( const VectorVirtualClass<complexs> &x, int conditiona, int conditionb, int width);
   template int VectorPlotHost( const VectorVirtualClass<complexd> &x, int conditiona, int conditionb, int width);
   
   int SequentialVectorReadFromFile(SequentialVectorClass<float> &vec, const char *vecfile, int idxin)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;   
      int i, I, J;
      
      float val;
      
      if ((f = fopen( vecfile, "r")) == NULL)
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
      
      /* create vector, fill with 0 */
      vec.Setup(M, kMemoryHost, true);
      
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
         I -= idxin;  /* adjust from 1-based to 0-based */
         vec[I] = val;
      }

      if (f !=stdin) fclose(f);
       
      return PARGEMSLR_SUCCESS;
   }
   
   int SequentialVectorReadFromFile(SequentialVectorClass<double> &vec, const char *vecfile, int idxin)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;   
      int i, I, J;
      
      double val;
      
      if ((f = fopen( vecfile, "r")) == NULL)
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
      
      /* create vector, fill with 0 */
      vec.Setup(M, kMemoryHost, true);
      
      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
      
      for (i=0; i<nz; i++)
      {
         if( fscanf(f, "%d %d %lf\n", &I, &J, &val) != 3 ) 
         {
            PARGEMSLR_ERROR("Error reading MM file.");
            return PARGEMSLR_ERROR_IO_ERROR;
         }
         I -= idxin;  /* adjust from 1-based to 0-based */
         vec[I] = val;
      }

      if (f !=stdin) fclose(f);
       
      return PARGEMSLR_SUCCESS;
   }
   
   int SequentialVectorReadFromFile(SequentialVectorClass<complexs> &vec, const char *vecfile, int idxin)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;   
      int i, I, J;
      
      float valr, vali;
      
      if ((f = fopen( vecfile, "r")) == NULL)
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
      if ( !(mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_complex(matcode)) ||  
            !mm_is_matrix(matcode) || !mm_is_coordinate(matcode) )
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0 || N != 1)
      {
         printf("Invalid Size.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }
      
      /* create vector, fill with 0 */
      vec.Setup(M, kMemoryHost, true);
      
      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
      if(mm_is_complex(matcode))
      {
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %f %f\n", &I, &J, &valr, &vali) != 4 ) 
            {
               PARGEMSLR_ERROR("Error reading MM file.");
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            I -= idxin;  /* adjust from 1-based to 0-based */
            vec[I] = complexs(valr, vali);
         }
      }
      else
      {
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %f\n", &I, &J, &valr) != 3 ) 
            {
               PARGEMSLR_ERROR("Error reading MM file.");
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            I -= idxin;  /* adjust from 1-based to 0-based */
            vec[I] = complexs(valr, 0.0);
         }
      }

      if (f !=stdin) fclose(f);
       
      return PARGEMSLR_SUCCESS;
   }
   
   int SequentialVectorReadFromFile(SequentialVectorClass<complexd> &vec, const char *vecfile, int idxin)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;   
      int i, I, J;
      
      double valr, vali;
      
      if ((f = fopen( vecfile, "r")) == NULL)
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
      if ( !(mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_complex(matcode)) || 
            !mm_is_matrix(matcode) || !mm_is_coordinate(matcode) )
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0 || N != 1)
      {
         printf("Invalid Size.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }
      
      /* create vector, fill with 0 */
      vec.Setup(M, kMemoryHost, true);
      
      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
      if(mm_is_complex(matcode))
      {
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %lf %lf\n", &I, &J, &valr, &vali) != 4 ) 
            {
               PARGEMSLR_ERROR("Error reading MM file.");
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            I -= idxin;  /* adjust from 1-based to 0-based */
            vec[I] = complexd(valr, vali);
         }
      }
      else
      {
        for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %lf\n", &I, &J, &valr) != 3 ) 
            {
               PARGEMSLR_ERROR("Error reading MM file.");
               return PARGEMSLR_ERROR_IO_ERROR;
            }
            I -= idxin;  /* adjust from 1-based to 0-based */
            vec[I] = complexd(valr, 0.0);
         } 
      }

      if (f !=stdin) fclose(f);
       
      return PARGEMSLR_SUCCESS;
   }
   
}
