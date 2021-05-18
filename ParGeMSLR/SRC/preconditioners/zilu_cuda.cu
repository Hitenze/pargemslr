/* this file is for the GPU version thrust call */

#ifdef PARSLR_CUDA

#include <iostream>
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../vectors/vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../matrices/matrix.hpp"
#include "../matrices/matrixops.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../matrices/dense_matrix.hpp"
#include "precond.hpp"
#include "zilu.hpp"

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"

namespace parslr
{  
   
   template <class MatrixType, class VectorType, typename RealDataType>
   int ZIluClass<MatrixType, VectorType, RealDataType>::Clear()
   {
      SolverClass<MatrixType, VectorType, RealDataType>::Clear();
      if(this->_LDU)
      {
         this->_LDU->Clear();
         PARSLR_FREE( this->_LDU, kMemoryHost);
      }
      if(this->_matL_info)
      {
         PARSLR_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(this->_matL_info) );
         this->_matL_info = NULL;
      }
      if(this->_matU_info)
      {
         PARSLR_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(this->_matU_info) );
         this->_matU_info = NULL;
      }
      this->_location = kMemoryHost;
      this->_n = 0;
      this->_nnz = 0;
      if(this->_L)
      {
         this->_L->Clear();
         PARSLR_FREE( this->_L, kMemoryHost);
      }
      if(this->_D)
      {
         this->_D->Clear();
         PARSLR_FREE( this->_D, kMemoryHost);
      }
      if(this->_U)
      {
         this->_U->Clear();
         PARSLR_FREE( this->_U, kMemoryHost);
      }
      this->_droptol = 1e-02;
      this->_max_row_nnz = 100;
      this->_option = 0;
      this->_perm_option = 0;
      this->_row_perm_vec.Clear();
      this->_col_perm_vec.Clear();
      this->_x_temp.Clear();
      
      return PARSLR_SUCCESS;
      
   }
   template int precond_zilu_csr_seq_complexs::Clear();
   template int precond_zilu_csr_seq_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename RealDataType>
   int ZIluClass<MatrixType, VectorType, RealDataType>::MoveData( const int &location)
   {
      
      /* we are not going to free the LDU matrix once constructed, and not going to free the L, D, U */
      if(this->_nnz == 0)
      {
         /* empty, do nothing */
         return PARSLR_SUCCESS;
      }
      
      if( location == kMemoryHost || (this->_LDU && location == this->_LDU->GetDataLocation() ) )
      {
         /* in this care, we already have the information, do nothing */
         return PARSLR_SUCCESS;
      }
      
      this->_location = location;
      
      int                           n, nnzLDU;
      
      int                                 *LDU_i, *LDU_j;
      ComplexValueClass<RealDataType>     *LDU_a;
      
      if(this->_LDU)
      {
         /* we already have LDU, just move the location */
         this->_LDU->MoveData(location);
         
         n = this->_n;
         LDU_i = this->_LDU->GetI();
         LDU_j = this->_LDU->GetJ();
         LDU_a = this->_LDU->GetData();
         
      }
      else
      {
         /* in this case, we don't have the data yet 
          * construct the LDU
          */
         
         int                                 nnzL, nnzU, i, j, k, i1, i2;
         int                                 *L_i, *L_j, *U_i, *U_j;
         ComplexValueClass<RealDataType>     *L_a, *U_a, *D;
         ComplexValueClass<RealDataType>     one;
         
         one = 1.0;
         
         n = this->_n;
         nnzL = this->_L->GetNumNonzeros();
         nnzU = this->_U->GetNumNonzeros();
         nnzLDU = nnzL+nnzU+n;
         
         PARSLR_PLACEMENT_NEW(this->_LDU, kMemoryHost, MatrixType);
         this->_LDU->Setup( n, n, nnzLDU);
         
         LDU_i = this->_LDU->GetI();
         LDU_j = this->_LDU->GetJ();
         LDU_a = this->_LDU->GetData();
         
         L_i = this->_L->GetI();
         L_j = this->_L->GetJ();
         L_a = this->_L->GetData();
         
         U_i = this->_U->GetI();
         U_j = this->_U->GetJ();
         U_a = this->_U->GetData();
         
         D = this->_D->GetData();
         
         LDU_i[0] = 0;
         k = 0;
         for(i = 0 ; i < n ; i ++)
         {
            i1 = L_i[i];
            i2 = L_i[i+1];
            for(j = i1 ; j < i2 ; j ++)
            {
               LDU_j[k]  = L_j[j];
               LDU_a[k++]  = L_a[j];
            }
            LDU_j[k]  = i;
            LDU_a[k++]  = one/D[i];
            i1 = U_i[i];
            i2 = U_i[i+1];
            for(j = i1 ; j < i2 ; j ++)
            {
               LDU_j[k]  = U_j[j];
               LDU_a[k++]  = U_a[j];
            }
            LDU_i[i+1] = k;
         }
         
         this->_LDU->MoveData(location);
         this->_LDU->SortRow();
         
         /* re obtain those values */
         LDU_i = this->_LDU->GetI();
         LDU_j = this->_LDU->GetJ();
         LDU_a = this->_LDU->GetData();
         
      }
      
      /* now starting setup the solver when necessary */
      if( location == kMemoryDevice || location == kMemoryUnified )
      {
         if(this->_x_temp.GetLengthLocal() != n)
         {
            this->_x_temp.Setup( n, kMemoryDevice, true);
         }
         
         bool           isdouble;
         int            matL_buffersize, matU_buffersize, bufferSize;
         
         if(sizeof(RealDataType) == sizeof(double))
         {
            isdouble = true;
         }
         else
         {
            isdouble = false;
         }
         
         /* 0. Destroy the current info */
         if(this->_matL_info)
         {
            PARSLR_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(this->_matL_info) );
            this->_matL_info = NULL;
         }
         if(this->_matU_info)
         {
            PARSLR_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(this->_matU_info) );
            this->_matU_info = NULL;
         }
         
         /* 1. Create info for ilu setup and solve */
         PARSLR_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&(this->_matL_info)));
         PARSLR_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&(this->_matU_info)));
         
         /* 2. Get working array size */
         if(isdouble)
         {

            PARSLR_CUSPARSE_CALL(cusparseZcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            n, 
                                                            nnzLDU,
                                                            parallel_log::_matL_des, 
                                                            (cuDoubleComplex *) LDU_a, 
                                                            LDU_i, 
                                                            LDU_j, 
                                                            this->_matL_info, 
                                                            &matL_buffersize));

            PARSLR_CUSPARSE_CALL(cusparseZcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            n, 
                                                            nnzLDU,
                                                            parallel_log::_matU_des, 
                                                            (cuDoubleComplex *) LDU_a, 
                                                            LDU_i, 
                                                            LDU_j, 
                                                            this->_matU_info, 
                                                            &matU_buffersize));
         }
         else
         {

            PARSLR_CUSPARSE_CALL(cusparseCcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            n, 
                                                            nnzLDU,
                                                            parallel_log::_matL_des, 
                                                            (cuComplex *) LDU_a, 
                                                            LDU_i, 
                                                            LDU_j, 
                                                            this->_matL_info, 
                                                            &matL_buffersize));

            PARSLR_CUSPARSE_CALL(cusparseCcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            n, 
                                                            nnzLDU,
                                                            parallel_log::_matU_des, 
                                                            (cuComplex *) LDU_a, 
                                                            LDU_i, 
                                                            LDU_j, 
                                                            this->_matU_info, 
                                                            &matU_buffersize));
         }
         bufferSize = ParslrMax( matL_buffersize, matU_buffersize );
         
         /* 3. Create working array, since they won't be visited by host, allocate on device */
         if( bufferSize > parallel_log::_cusparse_buffer_length )
         {
            PARSLR_REALLOC_VOID( parallel_log::_cusparse_buffer, parallel_log::_cusparse_buffer_length, bufferSize, kMemoryDevice)
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
         else
         {
            PARSLR_MALLOC_VOID( parallel_log::_cusparse_buffer, bufferSize, kMemoryDevice);
            parallel_log::_cusparse_buffer_length = bufferSize;
         }
         
         /* 4. Now perform the analysis */
         if(isdouble)
         {
            
            PARSLR_CUSPARSE_CALL(cusparseZcsrsv2_analysis(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            n,
                                                            nnzLDU,
                                                            parallel_log::_matL_des, 
                                                            (cuDoubleComplex *) LDU_a, 
                                                            LDU_i,
                                                            LDU_j,
                                                            this->_matL_info,
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
            
            PARSLR_CUSPARSE_CALL(cusparseZcsrsv2_analysis(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            n,
                                                            nnzLDU,
                                                            parallel_log::_matU_des, 
                                                            (cuDoubleComplex *) LDU_a, 
                                                            LDU_i,
                                                            LDU_j,
                                                            this->_matU_info,
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
            
         }
         else
         { 

            PARSLR_CUSPARSE_CALL(cusparseCcsrsv2_analysis(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            n,
                                                            nnzLDU,
                                                            parallel_log::_matL_des, 
                                                            (cuComplex *) LDU_a, 
                                                            LDU_i,
                                                            LDU_j,
                                                            this->_matL_info,
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
            
            PARSLR_CUSPARSE_CALL(cusparseCcsrsv2_analysis(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            n,
                                                            nnzLDU,
                                                            parallel_log::_matU_des, 
                                                            (cuComplex *) LDU_a, 
                                                            LDU_i,
                                                            LDU_j,
                                                            this->_matU_info,
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
            
         }                                                
      }
         
      return PARSLR_SUCCESS;
   }
   template int precond_zilu_csr_seq_complexs::MoveData( const int &location);
   template int precond_zilu_csr_seq_complexd::MoveData( const int &location);

   template <class MatrixType, class VectorType, typename RealDataType>
   int ZIluClass<MatrixType, VectorType, RealDataType>::Solve( VectorType &x, VectorType &rhs)
   {
      /* the solve phase of ilut */
      PARSLR_CHKERR(this->_n != x.GetLengthLocal() || this->_n != rhs.GetLengthLocal());
      
      if(this->_n == 0)
      {
         return PARSLR_SUCCESS;
      }
      
      int loc_x, loc_y;
      
      loc_x = x.GetDataLocation();
      loc_y = rhs.GetDataLocation();
      
      switch(this->_location)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matvec on device */
            PARSLR_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
            PARSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
            
            return this->SolveDevice(x, rhs);
            
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
                  PARSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                  
                  return this->SolveDevice(x, rhs);
                  
                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                  {
                     /* apply matvec on device */
                     return this->SolveDevice(x, rhs);
                  }
                  /* otherwise on host */
                  break;
               }
               default:
               {
                  /* matvec on host */
                  PARSLR_CHKERR( loc_y == kMemoryDevice );
                  break;
               }
            }
            break;
         }
         default:
         {
            /* matvec on host */
            PARSLR_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
            break;
         }
      }
      
      return this->SolveHost(x, rhs);
   }
   template int precond_zilu_csr_seq_complexs::Solve( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_zilu_csr_seq_complexd::Solve( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename RealDataType>
   int ZIluClass<MatrixType, VectorType, RealDataType>::SolveDevice( VectorType &x, VectorType &rhs)
   {
      /* the solve phase of ilut */
      RealDataType   one;
      bool           isdouble;
      
      one = 1.0;
      
      if(sizeof(RealDataType) == sizeof(double))
      {
         isdouble = true;
      }
      else
      {
         isdouble = false;
      }
      
      if(isdouble)
      {
         /* L solve - Forward solve */
         PARSLR_CUSPARSE_CALL(cusparseZcsrsv2_solve(parallel_log::_cusparse_handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                      this->_n,
                                                      this->_LDU->GetNumNonzeros(),
                                                      (cuDoubleComplex *) &one,
                                                      parallel_log::_matL_des,
                                                      (cuDoubleComplex *) this->_LDU->GetData(), 
                                                      this->_LDU->GetI(), 
                                                      this->_LDU->GetJ(), 
                                                      this->_matL_info, 
                                                      (cuDoubleComplex *) rhs.GetData(), 
                                                      (cuDoubleComplex *) this->_x_temp.GetData(), 
                                                      parallel_log::_ilu_solve_policy, 
                                                      parallel_log::_cusparse_buffer));
         
         /* U solve - Backward substitution */
         PARSLR_CUSPARSE_CALL(cusparseZcsrsv2_solve(parallel_log::_cusparse_handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                      this->_n,
                                                      this->_LDU->GetNumNonzeros(),
                                                      (cuDoubleComplex *) &one,
                                                      parallel_log::_matU_des,
                                                      (cuDoubleComplex *) this->_LDU->GetData(), 
                                                      this->_LDU->GetI(), 
                                                      this->_LDU->GetJ(), 
                                                      this->_matU_info, 
                                                      (cuDoubleComplex *) this->_x_temp.GetData(),
                                                      (cuDoubleComplex *) x.GetData(), 
                                                      parallel_log::_ilu_solve_policy, 
                                                      parallel_log::_cusparse_buffer));
      }
      else
      {
         /* L solve - Forward solve */
         PARSLR_CUSPARSE_CALL(cusparseCcsrsv2_solve(parallel_log::_cusparse_handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                      this->_n,
                                                      this->_LDU->GetNumNonzeros(),
                                                      (cuComplex *) &one,
                                                      parallel_log::_matL_des,
                                                      (cuComplex *) this->_LDU->GetData(), 
                                                      this->_LDU->GetI(), 
                                                      this->_LDU->GetJ(), 
                                                      this->_matL_info, 
                                                      (cuComplex *) rhs.GetData(), 
                                                      (cuComplex *) this->_x_temp.GetData(), 
                                                      parallel_log::_ilu_solve_policy, 
                                                      parallel_log::_cusparse_buffer));
         
         /* U solve - Backward substitution */
         PARSLR_CUSPARSE_CALL(cusparseCcsrsv2_solve(parallel_log::_cusparse_handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                      this->_n,
                                                      this->_LDU->GetNumNonzeros(),
                                                      (cuComplex *) &one,
                                                      parallel_log::_matU_des,
                                                      (cuComplex *) this->_LDU->GetData(), 
                                                      this->_LDU->GetI(), 
                                                      this->_LDU->GetJ(), 
                                                      this->_matU_info, 
                                                      (cuComplex *) this->_x_temp.GetData(),
                                                      (cuComplex *) x.GetData(), 
                                                      parallel_log::_ilu_solve_policy, 
                                                      parallel_log::_cusparse_buffer));
      }
      
      return PARSLR_SUCCESS;
   }
   template int precond_zilu_csr_seq_complexs::SolveDevice( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_zilu_csr_seq_complexd::SolveDevice( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
}
#endif
