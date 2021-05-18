
#ifdef PARGEMSLR_CUDA 

#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../utils/memory.hpp"
#include "../utils/protos.hpp"
#include "../vectors/vector.hpp"
#include "matrix.hpp"
#include "matrixops.hpp"
#include "csr_matrix.hpp"
#include "dense_matrix.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace pargemslr
{
   
   int DenseMatrixSMatVecDevice( const DenseMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y)
   {
      int   one = 1;
      int   m, n;
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      
      cublasOperation_t cutrans;
      
      cutrans = (trans == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
      
      PARGEMSLR_CUBLAS_CALL( (cublasSgemv(parallel_log::_cublas_handle, cutrans,
                              m, n,
                              &alpha,
                              A.GetData(), A.GetLeadingDimension(),
                              x.GetData(), one,
                              &beta,
                              y.GetData(), one)) );
      
      return PARGEMSLR_SUCCESS;
   }
   
   int DenseMatrixDMatVecDevice( const DenseMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y)
   {
      int   one = 1;
      int   m, n;
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      
      cublasOperation_t cutrans;
      
      cutrans = (trans == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
      
      PARGEMSLR_CUBLAS_CALL( (cublasDgemv(parallel_log::_cublas_handle, cutrans,
                              m, n,
                              &alpha,
                              A.GetData(), A.GetLeadingDimension(),
                              x.GetData(), one,
                              &beta,
                              y.GetData(), one)) );
      
      return PARGEMSLR_SUCCESS;
   }
   
   int DenseMatrixCMatVecDevice( const DenseMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y)
   {
      int   one = 1;
      int   m, n;
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      
      cublasOperation_t cutrans;
      
      switch(trans)
      {
         case 'N':
         {
            cutrans = CUBLAS_OP_N;
            break;
         }
         case 'T':
         {
            cutrans = CUBLAS_OP_T;
            break;
         }
         default:
         {
            cutrans = CUBLAS_OP_C;
            break;
         }
      }
      
      PARGEMSLR_CUBLAS_CALL( (cublasCgemv(parallel_log::_cublas_handle, cutrans,
                              m, n,
                              (cuComplex*) &alpha,
                              (cuComplex*) A.GetData(), A.GetLeadingDimension(),
                              (cuComplex*) x.GetData(), one,
                              (cuComplex*) &beta,
                              (cuComplex*) y.GetData(), one)) );
      
      return PARGEMSLR_SUCCESS;
   }
   
   int DenseMatrixZMatVecDevice( const DenseMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y)
   {
      int   one = 1;
      int   m, n;
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      
      cublasOperation_t cutrans;
      
      switch(trans)
      {
         case 'N':
         {
            cutrans = CUBLAS_OP_N;
            break;
         }
         case 'T':
         {
            cutrans = CUBLAS_OP_T;
            break;
         }
         default:
         {
            cutrans = CUBLAS_OP_C;
            break;
         }
      }
      
      PARGEMSLR_CUBLAS_CALL( (cublasZgemv(parallel_log::_cublas_handle, cutrans,
                              m, n,
                              (cuDoubleComplex*) &alpha,
                              (cuDoubleComplex*) A.GetData(), A.GetLeadingDimension(),
                              (cuDoubleComplex*) x.GetData(), one,
                              (cuDoubleComplex*) &beta,
                              (cuDoubleComplex*) y.GetData(), one)) );
   
      return PARGEMSLR_SUCCESS;
   }
   
   int DenseMatrixSMatMatDevice( int m, int n, int k, const float &alpha, const DenseMatrixClass<float> &A, char transa, const DenseMatrixClass<float> &B, char transb, const float &beta, DenseMatrixClass<float> &C)
   {
      int ldim_A, ldim_B, ldim_C;
      
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();
      
      cublasOperation_t cutransa, cutransb;
      
      cutransa = (transa == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
      cutransb = (transb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
      
      PARGEMSLR_CUBLAS_CALL( (cublasSgemm(parallel_log::_cublas_handle,
                              cutransa, cutransb,
                              m, n, k,
                              &alpha,
                              A.GetData(), ldim_A,
                              B.GetData(), ldim_B,
                              &beta,
                              C.GetData(), ldim_C)) );
      
      return PARGEMSLR_SUCCESS;
      
   }
   
   int DenseMatrixDMatMatDevice( int m, int n, int k, const double &alpha, const DenseMatrixClass<double> &A, char transa, const DenseMatrixClass<double> &B, char transb, const double &beta, DenseMatrixClass<double> &C)
   {
      int ldim_A, ldim_B, ldim_C;
      
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();
      
      cublasOperation_t cutransa, cutransb;
      
      cutransa = (transa == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
      cutransb = (transb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
      
      PARGEMSLR_CUBLAS_CALL( (cublasDgemm(parallel_log::_cublas_handle,
                              cutransa, cutransb,
                              m, n, k,
                              &alpha,
                              A.GetData(), ldim_A,
                              B.GetData(), ldim_B,
                              &beta,
                              C.GetData(), ldim_C)) );
      
      return PARGEMSLR_SUCCESS;
   }
   
   int DenseMatrixCMatMatDevice( int m, int n, int k, const complexs &alpha, const DenseMatrixClass<complexs> &A, char transa, const DenseMatrixClass<complexs> &B, char transb, const complexs &beta, DenseMatrixClass<complexs> &C)
   {
      int ldim_A, ldim_B, ldim_C;
      
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();
      
      cublasOperation_t cutransa, cutransb;
      
      switch(transa)
      {
         case 'N':
         {
            cutransa = CUBLAS_OP_N;
            break;
         }
         case 'T':
         {
            cutransa = CUBLAS_OP_T;
            break;
         }
         default:
         {
            cutransa = CUBLAS_OP_C;
            break;
         }
      }
      
      switch(transb)
      {
         case 'N':
         {
            cutransb = CUBLAS_OP_N;
            break;
         }
         case 'T':
         {
            cutransb = CUBLAS_OP_T;
            break;
         }
         default:
         {
            cutransb = CUBLAS_OP_C;
            break;
         }
      }
      
      PARGEMSLR_CUBLAS_CALL( (cublasCgemm(parallel_log::_cublas_handle,
                              cutransa, cutransb,
                              m, n, k,
                              (cuComplex*) &alpha,
                              (cuComplex*) A.GetData(), ldim_A,
                              (cuComplex*) B.GetData(), ldim_B,
                              (cuComplex*) &beta,
                              (cuComplex*) C.GetData(), ldim_C)) );
      
      return PARGEMSLR_SUCCESS;
   }
   
   int DenseMatrixZMatMatDevice( int m, int n, int k, const complexd &alpha, const DenseMatrixClass<complexd> &A, char transa, const DenseMatrixClass<complexd> &B, char transb, const complexd &beta, DenseMatrixClass<complexd> &C)
   {
      int ldim_A, ldim_B, ldim_C;
      
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();
      
      cublasOperation_t cutransa, cutransb;
      
      switch(transa)
      {
         case 'N':
         {
            cutransa = CUBLAS_OP_N;
            break;
         }
         case 'T':
         {
            cutransa = CUBLAS_OP_T;
            break;
         }
         default:
         {
            cutransa = CUBLAS_OP_C;
            break;
         }
      }
      
      switch(transb)
      {
         case 'N':
         {
            cutransb = CUBLAS_OP_N;
            break;
         }
         case 'T':
         {
            cutransb = CUBLAS_OP_T;
            break;
         }
         default:
         {
            cutransb = CUBLAS_OP_C;
            break;
         }
      }
      
      PARGEMSLR_CUBLAS_CALL( (cublasZgemm(parallel_log::_cublas_handle,
                              cutransa, cutransb,
                              m, n, k,
                              (cuDoubleComplex*) &alpha,
                              (cuDoubleComplex*) A.GetData(), ldim_A,
                              (cuDoubleComplex*) B.GetData(), ldim_B,
                              (cuDoubleComplex*) &beta,
                              (cuDoubleComplex*) C.GetData(), ldim_C)) );
      
      return PARGEMSLR_SUCCESS;
   }
   
   int CsrMatrixSMatVecDevice( const CsrMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y)
   {
      int         nnz;
      nnz = A.GetNumNonzeros();
      
      if(nnz == 0)
      {
         y.Scale(beta);
         return PARGEMSLR_SUCCESS;
      }
      
      cusparseOperation_t cutrans;
      
      cutrans = trans == 'N' ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;

#if (PARGEMSLR_CUDA_VERSION == 11)

      size_t      bufferSize;
      
      /* check buffersize for this matvec */
      PARGEMSLR_CUSPARSE_CALL( ( cusparseSpMV_bufferSize( parallel_log::_cusparse_handle,
                              cutrans,
                              (void*) &alpha,
                              A.GetCusparseMat(),
                              x.GetCusparseVec(),
                              (void*) &beta,
                              y.GetCusparseVec(),
                              CUDA_R_32F,
                              parallel_log::_cusparse_spmv_algorithm,
                              &bufferSize)) );
      
      /* allocate memory if we don't have enough */
      if( bufferSize > parallel_log::_cusparse_buffer_length )
      {
         if(parallel_log::_cusparse_buffer_length == 0)
         {
            PARGEMSLR_REALLOC_VOID( parallel_log::_cusparse_buffer, parallel_log::_cusparse_buffer_length, bufferSize, kMemoryDevice)
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
         else
         {
            PARGEMSLR_MALLOC_VOID( parallel_log::_cusparse_buffer, bufferSize, kMemoryDevice);
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
      }
      
      PARGEMSLR_CUSPARSE_CALL( ( cusparseSpMV( parallel_log::_cusparse_handle,
                              cutrans,
                              (void*) &alpha,
                              A.GetCusparseMat(),
                              x.GetCusparseVec(),
                              (void*) &beta,
                              y.GetCusparseVec(),
                              CUDA_R_32F,
                              parallel_log::_cusparse_spmv_algorithm,
                              parallel_log::_cusparse_buffer)) );

#else

      int m, n;
      
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      
      PARGEMSLR_CUSPARSE_CALL( ( cusparseScsrmv( parallel_log::_cusparse_handle,
                              cutrans,
                              m,
                              n,
                              nnz,
                              &alpha,
                              parallel_log::_mat_des,
                              A.GetData(),
                              A.GetI(),
                              A.GetJ(),
                              x.GetData(),
                              &beta,
                              y.GetData() ) ) );
      
#endif

      return PARGEMSLR_SUCCESS;
   }
   
   int CsrMatrixDMatVecDevice( const CsrMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y)
   {
      int         nnz;
      nnz = A.GetNumNonzeros();
      
      if(nnz == 0)
      {
         y.Scale(beta);
         return PARGEMSLR_SUCCESS;
      }
      
      cusparseOperation_t cutrans;
      
      cutrans = trans == 'N' ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;

#if (PARGEMSLR_CUDA_VERSION == 11)

      size_t      bufferSize;
      
      /* check buffersize for this matvec */
      PARGEMSLR_CUSPARSE_CALL( ( cusparseSpMV_bufferSize( parallel_log::_cusparse_handle,
                              cutrans,
                              (void*) &alpha,
                              A.GetCusparseMat(),
                              x.GetCusparseVec(),
                              (void*) &beta,
                              y.GetCusparseVec(),
                              CUDA_R_64F,
                              parallel_log::_cusparse_spmv_algorithm,
                              &bufferSize)) );
      
      /* allocate memory if we don't have enough */
      if( bufferSize > parallel_log::_cusparse_buffer_length )
      {
         if(parallel_log::_cusparse_buffer_length == 0)
         {
            PARGEMSLR_REALLOC_VOID( parallel_log::_cusparse_buffer, parallel_log::_cusparse_buffer_length, bufferSize, kMemoryDevice)
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
         else
         {
            PARGEMSLR_MALLOC_VOID( parallel_log::_cusparse_buffer, bufferSize, kMemoryDevice);
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
      }
          
      PARGEMSLR_CUSPARSE_CALL( ( cusparseSpMV( parallel_log::_cusparse_handle,
                              cutrans,
                              (void*) &alpha,
                              A.GetCusparseMat(),
                              x.GetCusparseVec(),
                              (void*) &beta,
                              y.GetCusparseVec(),
                              CUDA_R_64F,
                              parallel_log::_cusparse_spmv_algorithm,
                              parallel_log::_cusparse_buffer)) );

#else

      int m, n;
      
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      
      PARGEMSLR_CUSPARSE_CALL( ( cusparseDcsrmv( parallel_log::_cusparse_handle,
                              cutrans,
                              m,
                              n,
                              nnz,
                              &alpha,
                              parallel_log::_mat_des,
                              A.GetData(),
                              A.GetI(),
                              A.GetJ(),
                              x.GetData(),
                              &beta,
                              y.GetData() ) ) );
      
#endif
   
      return PARGEMSLR_SUCCESS;
   }
   
   int CsrMatrixCMatVecDevice( const CsrMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y)
   {
      int         nnz;
      nnz = A.GetNumNonzeros();
      
      if(nnz == 0)
      {
         y.Scale(beta);
         return PARGEMSLR_SUCCESS;
      }
      
      cusparseOperation_t cutrans;
      
      switch (trans)
      {
         case 'N':
         {
            cutrans = CUSPARSE_OPERATION_NON_TRANSPOSE;
            break;
         }
         case 'T':
         {
            cutrans = CUSPARSE_OPERATION_TRANSPOSE;
            break;
         }
         default:
         {
            cutrans = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
            break;
         }
      }
 
#if (PARGEMSLR_CUDA_VERSION == 11) 
      
      size_t      bufferSize;
      
      /* check buffersize for this matvec */
      PARGEMSLR_CUSPARSE_CALL( ( cusparseSpMV_bufferSize( parallel_log::_cusparse_handle,
                              cutrans,
                              (void*) &alpha,
                              A.GetCusparseMat(),
                              x.GetCusparseVec(),
                              (void*) &beta,
                              y.GetCusparseVec(),
                              CUDA_C_32F,
                              parallel_log::_cusparse_spmv_algorithm,
                              &bufferSize)) );
      
      /* allocate memory if we don't have enough */
      if( bufferSize > parallel_log::_cusparse_buffer_length )
      {
         if(parallel_log::_cusparse_buffer_length == 0)
         {
            PARGEMSLR_REALLOC_VOID( parallel_log::_cusparse_buffer, parallel_log::_cusparse_buffer_length, bufferSize, kMemoryDevice)
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
         else
         {
            PARGEMSLR_MALLOC_VOID( parallel_log::_cusparse_buffer, bufferSize, kMemoryDevice);
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
      }
          
      PARGEMSLR_CUSPARSE_CALL( ( cusparseSpMV( parallel_log::_cusparse_handle,
                              cutrans,
                              (void*) &alpha,
                              A.GetCusparseMat(),
                              x.GetCusparseVec(),
                              (void*) &beta,
                              y.GetCusparseVec(),
                              CUDA_C_32F,
                              parallel_log::_cusparse_spmv_algorithm,
                              parallel_log::_cusparse_buffer)) );

#else

      int m, n;
      
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      
      PARGEMSLR_CUSPARSE_CALL( ( cusparseCcsrmv( parallel_log::_cusparse_handle,
                              cutrans,
                              m,
                              n,
                              nnz,
                              (cuComplex*)&alpha,
                              parallel_log::_mat_des,
                              (cuComplex*)A.GetData(),
                              A.GetI(),
                              A.GetJ(),
                              (cuComplex*)x.GetData(),
                              (cuComplex*)&beta,
                              (cuComplex*)y.GetData() ) ) );
      
#endif
      
      return PARGEMSLR_SUCCESS;
   }
   
   int CsrMatrixZMatVecDevice( const CsrMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y)
   {
      int         nnz;
      nnz = A.GetNumNonzeros();
      
      if(nnz == 0)
      {
         y.Scale(beta);
         return PARGEMSLR_SUCCESS;
      }
      
      cusparseOperation_t cutrans;
      
      switch (trans)
      {
         case 'N':
         {
            cutrans = CUSPARSE_OPERATION_NON_TRANSPOSE;
            break;
         }
         case 'T':
         {
            cutrans = CUSPARSE_OPERATION_TRANSPOSE;
            break;
         }
         default:
         {
            cutrans = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
            break;
         }
      }

#if (PARGEMSLR_CUDA_VERSION == 11)
      
      size_t      bufferSize;
      
      /* check buffersize for this matvec */
      PARGEMSLR_CUSPARSE_CALL( ( cusparseSpMV_bufferSize( parallel_log::_cusparse_handle,
                              cutrans,
                              (void*) &alpha,
                              A.GetCusparseMat(),
                              x.GetCusparseVec(),
                              (void*) &beta,
                              y.GetCusparseVec(),
                              CUDA_C_64F,
                              parallel_log::_cusparse_spmv_algorithm,
                              &bufferSize)) );
      
      /* allocate memory if we don't have enough */
      if( bufferSize > parallel_log::_cusparse_buffer_length )
      {
         if(parallel_log::_cusparse_buffer_length == 0)
         {
            PARGEMSLR_REALLOC_VOID( parallel_log::_cusparse_buffer, parallel_log::_cusparse_buffer_length, bufferSize, kMemoryDevice)
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
         else
         {
            PARGEMSLR_MALLOC_VOID( parallel_log::_cusparse_buffer, bufferSize, kMemoryDevice);
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
      }
          
      PARGEMSLR_CUSPARSE_CALL( ( cusparseSpMV( parallel_log::_cusparse_handle,
                              cutrans,
                              (void*) &alpha,
                              A.GetCusparseMat(),
                              x.GetCusparseVec(),
                              (void*) &beta,
                              y.GetCusparseVec(),
                              CUDA_C_64F,
                              parallel_log::_cusparse_spmv_algorithm,
                              parallel_log::_cusparse_buffer)) );

#else
      
      int m, n;
      
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      
      PARGEMSLR_CUSPARSE_CALL( ( cusparseZcsrmv( parallel_log::_cusparse_handle,
                              cutrans,
                              m,
                              n,
                              nnz,
                              (cuDoubleComplex*)&alpha,
                              parallel_log::_mat_des,
                              (cuDoubleComplex*)A.GetData(),
                              A.GetI(),
                              A.GetJ(),
                              (cuDoubleComplex*)x.GetData(),
                              (cuDoubleComplex*)&beta,
                              (cuDoubleComplex*)y.GetData() ) ) );
      
#endif
      
      return PARGEMSLR_SUCCESS;
   }
   
   int CsrMatrixCreateCusparseSpMat( CsrMatrixClass<float> &A)
   {
#if (PARGEMSLR_CUDA_VERSION == 11)
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryPinned);
      
      int m, n, nnz;
      
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      nnz = A.GetNumNonzeros();
      
      cusparseSpMatDescr_t cusparse_mat;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateCsr( &(cusparse_mat),
                                                m,
                                                n,
                                                nnz,
                                                (void*) A.GetI(),
                                                (void*) A.GetJ(),
                                                (void*) A.GetData(),
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_base,
                                                CUDA_R_32F)) );
      
      A.SetCusparseMat(cusparse_mat);
#endif      
      return PARGEMSLR_SUCCESS;
   }
   
   int CsrMatrixCreateCusparseSpMat( CsrMatrixClass<double> &A)
   {
#if (PARGEMSLR_CUDA_VERSION == 11)
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryPinned);
      
      int m, n, nnz;
      
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      nnz = A.GetNumNonzeros();
      
      cusparseSpMatDescr_t cusparse_mat;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateCsr( &(cusparse_mat),
                                                m,
                                                n,
                                                nnz,
                                                (void*) A.GetI(),
                                                (void*) A.GetJ(),
                                                (void*) A.GetData(),
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_base,
                                                CUDA_R_64F)) );
      
      A.SetCusparseMat(cusparse_mat);
#endif      
      return PARGEMSLR_SUCCESS;
   }
   
   int CsrMatrixCreateCusparseSpMat( CsrMatrixClass<complexs> &A)
   {
#if (PARGEMSLR_CUDA_VERSION == 11)
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryPinned);
      
      int m, n, nnz;
      
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      nnz = A.GetNumNonzeros();
      
      cusparseSpMatDescr_t cusparse_mat;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateCsr( &(cusparse_mat),
                                                m,
                                                n,
                                                nnz,
                                                (void*) A.GetI(),
                                                (void*) A.GetJ(),
                                                (void*) A.GetData(),
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_base,
                                                CUDA_C_32F)) );
      
      A.SetCusparseMat(cusparse_mat);
#endif      
      return PARGEMSLR_SUCCESS;
   }
   
   int CsrMatrixCreateCusparseSpMat( CsrMatrixClass<complexd> &A)
   {
#if (PARGEMSLR_CUDA_VERSION == 11)
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryPinned);
      
      int m, n, nnz;
      
      m = A.GetNumRowsLocal();
      n = A.GetNumColsLocal();
      nnz = A.GetNumNonzeros();
      
      cusparseSpMatDescr_t cusparse_mat;
      
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateCsr( &(cusparse_mat),
                                                m,
                                                n,
                                                nnz,
                                                (void*) A.GetI(),
                                                (void*) A.GetJ(),
                                                (void*) A.GetData(),
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_base,
                                                CUDA_C_64F)) );
      
      A.SetCusparseMat(cusparse_mat);
#endif      
      return PARGEMSLR_SUCCESS;
   }
   
   int CsrMatrixSortRowDevice( matrix_csr_float &A)
   {
      
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryPinned);
      
      int                  m, n, nnz;
      int                  *d_i, *d_j;
      float                *d_a = NULL;
      float                *d_a_temp = NULL;
      int                  location;
      size_t               colperm_buffersize;
      int                  *colperm = NULL;
      void                 *colperm_buffer = NULL;
      
      bool                 hold_data = A.IsHoldingData();
      bool                 is_csr = A.IsCsr();
      
      /* get problem size */
      if(is_csr)
      {
         m = A.GetNumRowsLocal();
         n = A.GetNumColsLocal();
      }
      else
      {
         /* for csc matrix, treat as a transposed csr */
         n = A.GetNumRowsLocal();
         m = A.GetNumColsLocal();
      }
      
      nnz = A.GetNumNonzeros();
      
      /* get i, j, and a */
      d_i   = A.GetI();
      d_j   = A.GetJ();
      
      if(hold_data)
      {
         /* only sort data when A is holding data */
         location = A.GetDataLocation();
         d_a = A.GetData();
         PARGEMSLR_MALLOC( d_a_temp, nnz, kMemoryDevice, float);
         PARGEMSLR_MEMCPY( d_a_temp, d_a, nnz, kMemoryDevice, location, float);
      }
      
      /* get buffer size */
      PARGEMSLR_CUSPARSE_CALL((cusparseXcsrsort_bufferSizeExt(parallel_log::_cusparse_handle, m, n, nnz, d_i, d_j, &colperm_buffersize)) );
      
      /* create buffer */
      PARGEMSLR_MALLOC_VOID( colperm_buffer, colperm_buffersize, kMemoryDevice);
      
      /* create unit permutation array */
      PARGEMSLR_MALLOC( colperm, nnz, kMemoryDevice, int);
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateIdentityPermutation(parallel_log::_cusparse_handle, nnz, colperm)) );
      
      /* sort column */
      PARGEMSLR_CUSPARSE_CALL( (cusparseXcsrsort(parallel_log::_cusparse_handle, m, n, nnz, parallel_log::_mat_des, d_i, d_j, colperm, colperm_buffer)) );
      
      /* free buffer */
      PARGEMSLR_FREE(colperm_buffer, kMemoryDevice);
      
      if(hold_data)
      {
         /* when holding data, apply permutation to data */
#if (PARGEMSLR_CUDA_VERSION == 11)
         
         /* cusparse vectors */
         cusparseSpVecDescr_t colperm_spVec;
         cusparseDnVecDescr_t temp_dnVec;
         
         /* create a SpVec */
         PARGEMSLR_CUSPARSE_CALL( (cusparseCreateSpVec( &colperm_spVec,
                                                     nnz,
                                                     nnz,
                                                     colperm,
                                                     (void*) d_a,
                                                     parallel_log::_cusparse_idx_type,
                                                     parallel_log::_cusparse_idx_base,
                                                     CUDA_R_32F)) );
         
         /* create a DnVec */
         PARGEMSLR_CUSPARSE_CALL( (cusparseCreateDnVec(  &temp_dnVec, nnz, (void*) d_a_temp, CUDA_R_32F)) );
         
         /* use Gather to apply the permutation colperm_spVec[i] = temp_dnVec[colperm_spVec[i]] */
         PARGEMSLR_CUSPARSE_CALL( (cusparseGather( parallel_log::_cusparse_handle, temp_dnVec, colperm_spVec)) );
         
         /* free cusparse vectors */
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroySpVec(colperm_spVec)) );
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroyDnVe(temp_dnVec)) );
         
#else
         
         /* apply permutation */
         PARGEMSLR_CUSPARSE_CALL( (cusparseSgthr(parallel_log::_cusparse_handle, nnz, d_a_temp, d_a, colperm, parallel_log::_cusparse_idx_base)) );
         
#endif
         /* free d_a_temp */
         PARGEMSLR_FREE(d_a_temp, kMemoryDevice);
      }
      
      /* free premutation vector */
      PARGEMSLR_FREE(colperm, kMemoryDevice);
      
      A.IsRowSorted() = true;
      return PARGEMSLR_SUCCESS;
      
   }
   
   int CsrMatrixSortRowDevice( matrix_csr_double &A)
   {
      
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryPinned);
      
      int                  m, n, nnz;
      int                  *d_i, *d_j;
      double               *d_a = NULL;
      double               *d_a_temp = NULL;
      int                  location;
      size_t               colperm_buffersize;
      int                  *colperm = NULL;
      void                 *colperm_buffer = NULL;
      
      bool                 hold_data = A.IsHoldingData();
      bool                 is_csr = A.IsCsr();
      
      /* get problem size */
      if(is_csr)
      {
         m = A.GetNumRowsLocal();
         n = A.GetNumColsLocal();
      }
      else
      {
         /* for csc matrix, treat as a transposed csr */
         n = A.GetNumRowsLocal();
         m = A.GetNumColsLocal();
      }
      
      nnz = A.GetNumNonzeros();
      
      /* get i, j, and a */
      d_i   = A.GetI();
      d_j   = A.GetJ();
      
      if(hold_data)
      {
         /* only sort data when A is holding data */
         location = A.GetDataLocation();
         d_a = A.GetData();
         PARGEMSLR_MALLOC( d_a_temp, nnz, kMemoryDevice, double);
         PARGEMSLR_MEMCPY( d_a_temp, d_a, nnz, kMemoryDevice, location, double);
      }
      
      /* get buffer size */
      PARGEMSLR_CUSPARSE_CALL((cusparseXcsrsort_bufferSizeExt(parallel_log::_cusparse_handle, m, n, nnz, d_i, d_j, &colperm_buffersize)) );
      
      /* create buffer */
      PARGEMSLR_MALLOC_VOID( colperm_buffer, colperm_buffersize, kMemoryDevice);
      
      /* create unit permutation array */
      PARGEMSLR_MALLOC( colperm, nnz, kMemoryDevice, int);
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateIdentityPermutation(parallel_log::_cusparse_handle, nnz, colperm)) );
      
      /* sort column */
      PARGEMSLR_CUSPARSE_CALL( (cusparseXcsrsort(parallel_log::_cusparse_handle, m, n, nnz, parallel_log::_mat_des, d_i, d_j, colperm, colperm_buffer)) );
      
      /* free buffer */
      PARGEMSLR_FREE(colperm_buffer, kMemoryDevice);
      
      if(hold_data)
      {
         /* when holding data, apply permutation to data */
#if (PARGEMSLR_CUDA_VERSION == 11)
         
         /* cusparse vectors */
         cusparseSpVecDescr_t colperm_spVec;
         cusparseDnVecDescr_t temp_dnVec;
         
         /* create a SpVec */
         PARGEMSLR_CUSPARSE_CALL( (cusparseCreateSpVec( &colperm_spVec,
                                                     nnz,
                                                     nnz,
                                                     colperm,
                                                     (void*) d_a,
                                                     parallel_log::_cusparse_idx_type,
                                                     parallel_log::_cusparse_idx_base,
                                                     CUDA_R_64F)) );
         
         /* create a DnVec */
         PARGEMSLR_CUSPARSE_CALL( (cusparseCreateDnVec(  &temp_dnVec, nnz, (void*) d_a_temp, CUDA_R_64F)) );
         
         /* use Gather to apply the permutation colperm_spVec[i] = temp_dnVec[colperm_spVec[i]] */
         PARGEMSLR_CUSPARSE_CALL( (cusparseGather( parallel_log::_cusparse_handle, temp_dnVec, colperm_spVec)) );
         
         /* free cusparse vectors */
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroySpVec(colperm_spVec)) );
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroyDnVe(temp_dnVec)) );
         
#else
         
         /* apply permutation */
         PARGEMSLR_CUSPARSE_CALL( (cusparseDgthr(parallel_log::_cusparse_handle, nnz, d_a_temp, d_a, colperm, parallel_log::_cusparse_idx_base)) );
         
#endif
         /* free d_a_temp */
         PARGEMSLR_FREE(d_a_temp, kMemoryDevice);
      }
      
      /* free premutation vector */
      PARGEMSLR_FREE(colperm, kMemoryDevice);
      
      A.IsRowSorted() = true;
      return PARGEMSLR_SUCCESS;
      
   }
   
   int CsrMatrixSortRowDevice( matrix_csr_complexs &A)
   {
      
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryPinned);
      
      int                  m, n, nnz;
      int                  *d_i, *d_j;
      complexs             *d_a = NULL;
      complexs             *d_a_temp = NULL;
      int                  location;
      size_t               colperm_buffersize;
      int                  *colperm = NULL;
      void                 *colperm_buffer = NULL;
      
      bool                 hold_data = A.IsHoldingData();
      bool                 is_csr = A.IsCsr();
      
      /* get problem size */
      if(is_csr)
      {
         m = A.GetNumRowsLocal();
         n = A.GetNumColsLocal();
      }
      else
      {
         /* for csc matrix, treat as a transposed csr */
         n = A.GetNumRowsLocal();
         m = A.GetNumColsLocal();
      }
      
      nnz = A.GetNumNonzeros();
      
      /* get i, j, and a */
      d_i   = A.GetI();
      d_j   = A.GetJ();
      
      if(hold_data)
      {
         /* only sort data when A is holding data */
         location = A.GetDataLocation();
         d_a = A.GetData();
         PARGEMSLR_MALLOC( d_a_temp, nnz, kMemoryDevice, complexs);
         PARGEMSLR_MEMCPY( d_a_temp, d_a, nnz, kMemoryDevice, location, complexs);
      }
      
      /* get buffer size */
      PARGEMSLR_CUSPARSE_CALL((cusparseXcsrsort_bufferSizeExt(parallel_log::_cusparse_handle, m, n, nnz, d_i, d_j, &colperm_buffersize)) );
      
      /* create buffer */
      PARGEMSLR_MALLOC_VOID( colperm_buffer, colperm_buffersize, kMemoryDevice);
      
      /* create unit permutation array */
      PARGEMSLR_MALLOC( colperm, nnz, kMemoryDevice, int);
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateIdentityPermutation(parallel_log::_cusparse_handle, nnz, colperm)) );
      
      /* sort column */
      PARGEMSLR_CUSPARSE_CALL( (cusparseXcsrsort(parallel_log::_cusparse_handle, m, n, nnz, parallel_log::_mat_des, d_i, d_j, colperm, colperm_buffer)) );
      
      /* free buffer */
      PARGEMSLR_FREE(colperm_buffer, kMemoryDevice);
      
      if(hold_data)
      {
         /* when holding data, apply permutation to data */
#if (PARGEMSLR_CUDA_VERSION == 11)
         
         /* cusparse vectors */
         cusparseSpVecDescr_t colperm_spVec;
         cusparseDnVecDescr_t temp_dnVec;
         
         /* create a SpVec */
         PARGEMSLR_CUSPARSE_CALL( (cusparseCreateSpVec( &colperm_spVec,
                                                     nnz,
                                                     nnz,
                                                     colperm,
                                                     (void*) d_a,
                                                     parallel_log::_cusparse_idx_type,
                                                     parallel_log::_cusparse_idx_base,
                                                     CUDA_C_32F)) );
         
         /* create a DnVec */
         PARGEMSLR_CUSPARSE_CALL( (cusparseCreateDnVec(  &temp_dnVec, nnz, (void*) d_a_temp, CUDA_C_32F)) );
         
         /* use Gather to apply the permutation colperm_spVec[i] = temp_dnVec[colperm_spVec[i]] */
         PARGEMSLR_CUSPARSE_CALL( (cusparseGather( parallel_log::_cusparse_handle, temp_dnVec, colperm_spVec)) );
         
         /* free cusparse vectors */
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroySpVec(colperm_spVec)) );
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroyDnVe(temp_dnVec)) );
         
#else
         
         /* apply permutation */
         PARGEMSLR_CUSPARSE_CALL( (cusparseCgthr(parallel_log::_cusparse_handle, nnz, (cuComplex*) d_a_temp, (cuComplex*) d_a, colperm, parallel_log::_cusparse_idx_base)) );
         
#endif
         /* free d_a_temp */
         PARGEMSLR_FREE(d_a_temp, kMemoryDevice);
      }
      
      /* free premutation vector */
      PARGEMSLR_FREE(colperm, kMemoryDevice);
      
      A.IsRowSorted() = true;
      return PARGEMSLR_SUCCESS;
      
   }
   
   int CsrMatrixSortRowDevice( matrix_csr_complexd &A)
   {
      
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR( A.GetDataLocation() == kMemoryPinned);
      
      int                  m, n, nnz;
      int                  *d_i, *d_j;
      complexd             *d_a = NULL;
      complexd             *d_a_temp = NULL;
      int                  location;
      size_t               colperm_buffersize;
      int                  *colperm = NULL;
      void                 *colperm_buffer = NULL;
      
      bool                 hold_data = A.IsHoldingData();
      bool                 is_csr = A.IsCsr();
      
      /* get problem size */
      if(is_csr)
      {
         m = A.GetNumRowsLocal();
         n = A.GetNumColsLocal();
      }
      else
      {
         /* for csc matrix, treat as a transposed csr */
         n = A.GetNumRowsLocal();
         m = A.GetNumColsLocal();
      }
      
      nnz = A.GetNumNonzeros();
      
      /* get i, j, and a */
      d_i   = A.GetI();
      d_j   = A.GetJ();
      
      if(hold_data)
      {
         /* only sort data when A is holding data */
         location = A.GetDataLocation();
         d_a = A.GetData();
         PARGEMSLR_MALLOC( d_a_temp, nnz, kMemoryDevice, complexd);
         PARGEMSLR_MEMCPY( d_a_temp, d_a, nnz, kMemoryDevice, location, complexd);
      }
      
      /* get buffer size */
      PARGEMSLR_CUSPARSE_CALL((cusparseXcsrsort_bufferSizeExt(parallel_log::_cusparse_handle, m, n, nnz, d_i, d_j, &colperm_buffersize)) );
      
      /* create buffer */
      PARGEMSLR_MALLOC_VOID( colperm_buffer, colperm_buffersize, kMemoryDevice);
      
      /* create unit permutation array */
      PARGEMSLR_MALLOC( colperm, nnz, kMemoryDevice, int);
      PARGEMSLR_CUSPARSE_CALL( (cusparseCreateIdentityPermutation(parallel_log::_cusparse_handle, nnz, colperm)) );
      
      /* sort column */
      PARGEMSLR_CUSPARSE_CALL( (cusparseXcsrsort(parallel_log::_cusparse_handle, m, n, nnz, parallel_log::_mat_des, d_i, d_j, colperm, colperm_buffer)) );
      
      /* free buffer */
      PARGEMSLR_FREE(colperm_buffer, kMemoryDevice);
      
      if(hold_data)
      {
         /* when holding data, apply permutation to data */
#if (PARGEMSLR_CUDA_VERSION == 11)
         
         /* cusparse vectors */
         cusparseSpVecDescr_t colperm_spVec;
         cusparseDnVecDescr_t temp_dnVec;
         
         /* create a SpVec */
         PARGEMSLR_CUSPARSE_CALL( (cusparseCreateSpVec( &colperm_spVec,
                                                     nnz,
                                                     nnz,
                                                     colperm,
                                                     (void*) d_a,
                                                     parallel_log::_cusparse_idx_type,
                                                     parallel_log::_cusparse_idx_base,
                                                     CUDA_C_64F)) );
         
         /* create a DnVec */
         PARGEMSLR_CUSPARSE_CALL( (cusparseCreateDnVec(  &temp_dnVec, nnz, (void*) d_a_temp, CUDA_C_64F)) );
         
         /* use Gather to apply the permutation colperm_spVec[i] = temp_dnVec[colperm_spVec[i]] */
         PARGEMSLR_CUSPARSE_CALL( (cusparseGather( parallel_log::_cusparse_handle, temp_dnVec, colperm_spVec)) );
         
         /* free cusparse vectors */
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroySpVec(colperm_spVec)) );
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroyDnVe(temp_dnVec)) );
         
#else
         
         /* apply permutation */
         PARGEMSLR_CUSPARSE_CALL( (cusparseZgthr(parallel_log::_cusparse_handle, nnz, (cuDoubleComplex*) d_a_temp, (cuDoubleComplex*) d_a, colperm, parallel_log::_cusparse_idx_base)) );
         
#endif
         /* free d_a_temp */
         PARGEMSLR_FREE(d_a_temp, kMemoryDevice);
      }
      
      /* free premutation vector */
      PARGEMSLR_FREE(colperm, kMemoryDevice);
      
      A.IsRowSorted() = true;
      return PARGEMSLR_SUCCESS;
      
   }
   
}

#endif
