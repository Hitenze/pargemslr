
#ifdef PARGEMSLR_CUDA 

#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../utils/memory.hpp"
#include "../utils/protos.hpp"
#include "vector.hpp"
#include "sequential_vector.hpp"
#include "vectorops.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

namespace pargemslr
{

   int VectorSscaleDevice(VectorClass<float> &x, const float &a) 
   {
      int 	one = 1;
      int 	n = x.GetLengthLocal();
      float *data = x.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasSscal(parallel_log::_cublas_handle, n,
                        &a,
                        data, one) ));
                        
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorDscaleDevice(VectorClass<double> &x, const double &a) 
   {
      int 	one = 1;
      int 	n = x.GetLengthLocal();
      double *data = x.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasDscal(parallel_log::_cublas_handle, n,
                        &a,
                        data, one) ));
                        
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorCscaleDevice(VectorClass<complexs> &x, const complexs &a) 
   {
      int 	one = 1;
      int 	n = x.GetLengthLocal();
      complexs *data = x.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasCscal(parallel_log::_cublas_handle, n,
                        (cuComplex*) &a,
                        (cuComplex*) data, one) ) );
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorZscaleDevice(VectorClass<complexd> &x, const complexd &a) 
   {
      int 	one = 1;
      int 	n = x.GetLengthLocal();
      complexd *data = x.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasZscal(parallel_log::_cublas_handle, n,
                        (cuDoubleComplex*) &a,
                        (cuDoubleComplex*) data, one) ) );
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorSaxpyDevice( const float &a, const VectorClass<float> &x, VectorClass<float> &y)
   {
      int 	one = 1;
      int 	n = x.GetLengthLocal();
      float *x_data = x.GetData();
      float *y_data = y.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasSaxpy(parallel_log::_cublas_handle, n,
                        &a,
                        x_data, one,
                        y_data, one) ));
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorDaxpyDevice( const double &a, const VectorClass<double> &x, VectorClass<double> &y)
   {
      int 	one = 1;
      int 	n = x.GetLengthLocal();
      double *x_data = x.GetData();
      double *y_data = y.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasDaxpy(parallel_log::_cublas_handle, n,
                        &a,
                        x_data, one,
                        y_data, one) ));
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorCaxpyDevice( const complexs &a, const VectorClass<complexs> &x, VectorClass<complexs> &y)
   {
      int 	one = 1;
      int 	n = x.GetLengthLocal();
      complexs *x_data = x.GetData();
      complexs *y_data = y.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasCaxpy(parallel_log::_cublas_handle, n,
                        (cuComplex*) &a,
                        (cuComplex*) x_data, one,
                        (cuComplex*) y_data, one) ) );
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorZaxpyDevice( const complexd &a, const VectorClass<complexd> &x, VectorClass<complexd> &y)
   {
      int 	one = 1;
      int 	n = x.GetLengthLocal();
      complexd *x_data = x.GetData();
      complexd *y_data = y.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasZaxpy(parallel_log::_cublas_handle, n,
                        (cuDoubleComplex*) &a,
                        (cuDoubleComplex*) x_data, one,
                        (cuDoubleComplex*) y_data, one) ) );
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorRandDevice( VectorClass<float> &x)
   {
	   int n = x.GetLengthLocal();
	   float *data = x.GetData();
	   PARGEMSLR_CURAND_CALL(curandGenerateUniform(parallel_log::_curand_gen, data, n));
	   return PARGEMSLR_SUCCESS;
   }
   
   int VectorRandDevice( VectorClass<double> &x)
   {
	   int n = x.GetLengthLocal();
	   double *data = x.GetData();
	   PARGEMSLR_CURAND_CALL(curandGenerateUniformDouble(parallel_log::_curand_gen, data, n));
	   return PARGEMSLR_SUCCESS;
   }
   
   int VectorRandDevice( VectorClass<complexs> &x)
   {
	   int n = x.GetLengthLocal();
	   float *data = (float*) x.GetData();
	   PARGEMSLR_CURAND_CALL(curandGenerateUniform(parallel_log::_curand_gen, data, 2*n));
	   return PARGEMSLR_SUCCESS;
   }
   
   int VectorRandDevice( VectorClass<complexd> &x)
   {
	   int n = x.GetLengthLocal();
	   double *data = (double*) x.GetData();
	   PARGEMSLR_CURAND_CALL(curandGenerateUniformDouble(parallel_log::_curand_gen, data, 2*n));
	   return PARGEMSLR_SUCCESS;
   }
   
   int VectorSdotDevice( const vector_base_float &x, const vector_base_float &y, float &t)
   {
      int one = 1;
      int n = x.GetLengthLocal();
      float *x_data = x.GetData();
      float *y_data = y.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasSdot (parallel_log::_cublas_handle, n,
                        x_data, one,
                        y_data, one,
                        &t) ) );
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorDdotDevice( const vector_base_double &x, const vector_base_double &y, double &t)
   {
      int one = 1;
      int n = x.GetLengthLocal();
      double *x_data = x.GetData();
      double *y_data = y.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasDdot (parallel_log::_cublas_handle, n,
                        x_data, one,
                        y_data, one,
                        &t) ) );
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorCdotDevice( const vector_base_complexs &x, const vector_base_complexs &y, complexs &t)
   {
      int one = 1;
      int n = x.GetLengthLocal();
      complexs *x_data = x.GetData();
      complexs *y_data = y.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasCdotc(parallel_log::_cublas_handle, n,
                        (cuComplex*) x_data, one,
                        (cuComplex*) y_data, one,
                        (cuComplex*) &t) ) );
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }
   
   int VectorZdotDevice( const vector_base_complexd &x, const vector_base_complexd &y, complexd &t)
   {
      int one = 1;
      int n = x.GetLengthLocal();
      complexd *x_data = x.GetData();
      complexd *y_data = y.GetData();
      
      PARGEMSLR_CUBLAS_CALL( (cublasZdotc(parallel_log::_cublas_handle, n,
                        (cuDoubleComplex*) x_data, one,
                        (cuDoubleComplex*) y_data, one,
                        (cuDoubleComplex*) &t) ) );
      cudaDeviceSynchronize();
      return PARGEMSLR_SUCCESS;
   }

   int SequentialVectorCreateCusparseDnVec(vector_seq_float &v)
   {
#if (PARGEMSLR_CUDA_VERSION == 11)
      PARGEMSLR_CHKERR( v.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( v.GetDataLocation() == kMemoryPinned);
      
      int n = v.GetLengthLocal();
      
      cusparseDnVecDescr_t cusparse_vec;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateDnVec(&(cusparse_vec),
                              n,
                              (void*) v.GetData(),
                              CUDA_R_32F)) );
      
      v.SetCusparseVec(cusparse_vec);
#endif     
      return PARGEMSLR_SUCCESS;
   }
   
   int SequentialVectorCreateCusparseDnVec(vector_seq_double &v)
   {
#if (PARGEMSLR_CUDA_VERSION == 11)
      PARGEMSLR_CHKERR( v.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( v.GetDataLocation() == kMemoryPinned);
      
      int n = v.GetLengthLocal();
      
      cusparseDnVecDescr_t cusparse_vec;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateDnVec(&(cusparse_vec),
                              n,
                              (void*) v.GetData(),
                              CUDA_R_64F)) );
      
      v.SetCusparseVec(cusparse_vec);
#endif
      return PARGEMSLR_SUCCESS;
   }
   
   int SequentialVectorCreateCusparseDnVec(vector_seq_complexs &v)
   {
#if (PARGEMSLR_CUDA_VERSION == 11)
      PARGEMSLR_CHKERR( v.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( v.GetDataLocation() == kMemoryPinned);
      
      int n = v.GetLengthLocal();
      
      cusparseDnVecDescr_t cusparse_vec;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateDnVec(&(cusparse_vec),
                              n,
                              (void*) v.GetData(),
                              CUDA_C_32F)) );
      
      v.SetCusparseVec(cusparse_vec);
#endif
      return PARGEMSLR_SUCCESS;
   }
   
   int SequentialVectorCreateCusparseDnVec(vector_seq_complexd &v)
   {
#if (PARGEMSLR_CUDA_VERSION == 11)
      PARGEMSLR_CHKERR( v.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( v.GetDataLocation() == kMemoryPinned);
      
      int n = v.GetLengthLocal();
      
      cusparseDnVecDescr_t cusparse_vec;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateDnVec(&(cusparse_vec),
                              n,
                              (void*) v.GetData(),
                              CUDA_C_64F)) );
      
      v.SetCusparseVec(cusparse_vec);
#endif
      return PARGEMSLR_SUCCESS;
   }
   
}

#endif
