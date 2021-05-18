/* this file is for the GPU version thrust call */

#ifdef PARGEMSLR_CUDA

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
#include "ilu.hpp"

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"

namespace pargemslr
{  
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::Clear()
   {
      SolverClass<MatrixType, VectorType, DataType>::Clear();
      this->_LDU.Clear();
      if(this->_matL_info)
      {
         PARGEMSLR_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(this->_matL_info) );
         this->_matL_info = NULL;
      }
      if(this->_matU_info)
      {
         PARGEMSLR_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(this->_matU_info) );
         this->_matU_info = NULL;
      }
      this->_cusparse_ready = false;
      this->_location = kMemoryHost;
      this->_n = 0;
      this->_nnz = 0;
      this->_L.Clear();
      this->_D.Clear();
      this->_U.Clear();
#ifdef PARGEMSLR_OPENMP
      this->_L_solver.Clear();
      this->_U_solver.Clear();
      this->_L_level.Clear();
      this->_D_level.Clear();
      this->_U_level.Clear();
      this->_level_ptr_l.Clear();
      this->_levels_l.Clear();
      this->_level_ptr_u.Clear();
      this->_levels_u.Clear();
      this->_levels_l_start = 0;
      this->_levels_l_end = 0;
      this->_levels_u_start = 0;
      this->_levels_u_end = 0;
#endif
      this->_droptol = 1e-02;
      this->_max_row_nnz = 100;
      this->_option = 0;
      this->_perm_option = 0;
      this->_omp_option = kIluOpenMPLevelScheduling;
      this->_poly_order = 3;
      this->_row_perm_vec.Clear();
      this->_col_perm_vec.Clear();
      this->_x_temp.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_ilu_csr_seq_float::Clear();
   template int precond_ilu_csr_seq_double::Clear();
   template int precond_ilu_csr_seq_complexs::Clear();
   template int precond_ilu_csr_seq_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::MoveData( const int &location)
   {
      
      /* we are not going to free the LDU matrix once constructed, and not going to free the L, D, U */
      if(this->_nnz == 0)
      {
         /* empty, do nothing */
         return PARGEMSLR_SUCCESS;
      }
      
      if( location == kMemoryHost || (this->_LDU.GetNumNonzeros() > 0 && location == this->_LDU.GetDataLocation() ) )
      {
         /* in this care, we already have the information, do nothing */
         return PARGEMSLR_SUCCESS;
      }
      
      /* move the solve buffer first */
      this->_x_temp.MoveData(location);
      this->_row_perm_vec.MoveData(location);
      this->_col_perm_vec.MoveData(location);
      
      this->_E.MoveData(location);
      this->_F.MoveData(location);
      this->_S.MoveData(location);
      
      /* define the data type */
      typedef DataType T;
      
      this->_location = location;
      
      int      n, nnzLDU;
      
      int      *LDU_i, *LDU_j;
      T        *LDU_a;
      
      if(this->_LDU.GetNumNonzeros()>0)
      {
         /* we already have LDU, just move the location */
         this->_LDU.MoveData(location);
         
         if(this->_option == kIluOptionPartialILUT)
         {
            n = this->_nB;
         }
         else
         {
            n = this->_n;
         }
         
         LDU_i = this->_LDU.GetI();
         LDU_j = this->_LDU.GetJ();
         LDU_a = this->_LDU.GetData();
         
      }
      else
      {
         /* in this case, we don't have the data yet 
          * construct the LDU
          */
         
         int            nnzL, nnzU, i, j, k, i1, i2;
         int            *L_i, *L_j, *U_i, *U_j;
         T              *L_a, *U_a, *D;
         
         if(this->_option == kIluOptionPartialILUT)
         {
            n = this->_nB;
         }
         else
         {
            n = this->_n;
         }
         
         nnzL = this->_L.GetNumNonzeros();
         nnzU = this->_U.GetNumNonzeros();
         nnzLDU = nnzL+nnzU+n;
         
         this->_LDU.Setup( n, n, nnzLDU);
         
         LDU_i = this->_LDU.GetI();
         LDU_j = this->_LDU.GetJ();
         LDU_a = this->_LDU.GetData();
         
         L_i = this->_L.GetI();
         L_j = this->_L.GetJ();
         L_a = this->_L.GetData();
         
         U_i = this->_U.GetI();
         U_j = this->_U.GetJ();
         U_a = this->_U.GetData();
         
         D = this->_D.GetData();
         
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
            LDU_a[k++]  = T(1.0)/D[i];
            i1 = U_i[i];
            i2 = U_i[i+1];
            for(j = i1 ; j < i2 ; j ++)
            {
               LDU_j[k]  = U_j[j];
               LDU_a[k++]  = U_a[j];
            }
            LDU_i[i+1] = k;
         }
         
         this->_LDU.MoveData(location);
         this->_LDU.SortRow();
         
         /* re obtain those values */
         LDU_i = this->_LDU.GetI();
         LDU_j = this->_LDU.GetJ();
         LDU_a = this->_LDU.GetData();
         
      }
      
      /* now starting setup the solver when necessary */
      if( location == kMemoryDevice || location == kMemoryUnified )
      {
         
         int            matL_buffersize, matU_buffersize, bufferSize;
         
         /* 0. Destroy the current info */
         if(this->_matL_info)
         {
            PARGEMSLR_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(this->_matL_info) );
            this->_matL_info = NULL;
         }
         if(this->_matU_info)
         {
            PARGEMSLR_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(this->_matU_info) );
            this->_matU_info = NULL;
         }
         
         /* 1. Create info for ilu setup and solve */
         PARGEMSLR_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&(this->_matL_info)));
         PARGEMSLR_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&(this->_matU_info)));
         
         /* 2. Get working array size */
         switch(this->_solver_precision)
         {
            case kSingleReal:
            {
               /* float */
               PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n, 
                                                               nnzLDU,
                                                               parallel_log::_matL_des, 
                                                               (float *) LDU_a, 
                                                               LDU_i, 
                                                               LDU_j, 
                                                               this->_matL_info, 
                                                               &matL_buffersize));

               PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n, 
                                                               nnzLDU,
                                                               parallel_log::_matU_des, 
                                                               (float *) LDU_a, 
                                                               LDU_i, 
                                                               LDU_j, 
                                                               this->_matU_info, 
                                                               &matU_buffersize));
               break;
            }
            case kDoubleReal:
            {
               /* double */
               PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n, 
                                                               nnzLDU,
                                                               parallel_log::_matL_des, 
                                                               (double *) LDU_a, 
                                                               LDU_i, 
                                                               LDU_j, 
                                                               this->_matL_info, 
                                                               &matL_buffersize));

               PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n, 
                                                               nnzLDU,
                                                               parallel_log::_matU_des, 
                                                               (double *) LDU_a, 
                                                               LDU_i, 
                                                               LDU_j, 
                                                               this->_matU_info, 
                                                               &matU_buffersize));
               break;
            }
            case kSingleComplex:
            {
               PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n, 
                                                               nnzLDU,
                                                               parallel_log::_matL_des, 
                                                               (cuComplex *) LDU_a, 
                                                               LDU_i, 
                                                               LDU_j, 
                                                               this->_matL_info, 
                                                               &matL_buffersize));

               PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n, 
                                                               nnzLDU,
                                                               parallel_log::_matU_des, 
                                                               (cuComplex *) LDU_a, 
                                                               LDU_i, 
                                                               LDU_j, 
                                                               this->_matU_info, 
                                                               &matU_buffersize));
               break;
            }
            case kDoubleComplex:
            {
               PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n, 
                                                               nnzLDU,
                                                               parallel_log::_matL_des, 
                                                               (cuDoubleComplex *) LDU_a, 
                                                               LDU_i, 
                                                               LDU_j, 
                                                               this->_matL_info, 
                                                               &matL_buffersize));

               PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_bufferSize(parallel_log::_cusparse_handle, 
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n, 
                                                               nnzLDU,
                                                               parallel_log::_matU_des, 
                                                               (cuDoubleComplex *) LDU_a, 
                                                               LDU_i, 
                                                               LDU_j, 
                                                               this->_matU_info, 
                                                               &matU_buffersize));
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unkown precision for ILU move data.");
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
         
         bufferSize = PargemslrMax( matL_buffersize, matU_buffersize );
         
         /* 3. Create working array, since they won't be visited by host, allocate on device */
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
         
         /* 4. Now perform the analysis */
         switch(this->_solver_precision)
         {
            case kSingleReal:
            {
               /* float */
               PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_analysis(parallel_log::_cusparse_handle,
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n,
                                                               nnzLDU,
                                                               parallel_log::_matL_des, 
                                                               (float *) LDU_a, 
                                                               LDU_i,
                                                               LDU_j,
                                                               this->_matL_info,
                                                               parallel_log::_ilu_solve_policy, 
                                                               parallel_log::_cusparse_buffer));
               
               PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_analysis(parallel_log::_cusparse_handle,
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n,
                                                               nnzLDU,
                                                               parallel_log::_matU_des, 
                                                               (float *) LDU_a, 
                                                               LDU_i,
                                                               LDU_j,
                                                               this->_matU_info,
                                                               parallel_log::_ilu_solve_policy, 
                                                               parallel_log::_cusparse_buffer));
               break;
            }
            case kDoubleReal:
            {
               /* double */
               PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_analysis(parallel_log::_cusparse_handle,
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n,
                                                               nnzLDU,
                                                               parallel_log::_matL_des, 
                                                               (double *) LDU_a, 
                                                               LDU_i,
                                                               LDU_j,
                                                               this->_matL_info,
                                                               parallel_log::_ilu_solve_policy, 
                                                               parallel_log::_cusparse_buffer));
               
               PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_analysis(parallel_log::_cusparse_handle,
                                                               CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                               n,
                                                               nnzLDU,
                                                               parallel_log::_matU_des, 
                                                               (double *) LDU_a, 
                                                               LDU_i,
                                                               LDU_j,
                                                               this->_matU_info,
                                                               parallel_log::_ilu_solve_policy, 
                                                               parallel_log::_cusparse_buffer));
               break;
            }
            case kSingleComplex:
            {
               /* single complex */
               PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_analysis(parallel_log::_cusparse_handle,
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
               
               PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_analysis(parallel_log::_cusparse_handle,
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
               break;
            }
            case kDoubleComplex:
            {
               /* double complex */
               PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_analysis(parallel_log::_cusparse_handle,
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
               
               PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_analysis(parallel_log::_cusparse_handle,
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
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unkown precision for ILU move data.");
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }                                             
      }
         
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::MoveData( const int &location);
   template int precond_ilu_csr_seq_double::MoveData( const int &location);
   template int precond_ilu_csr_seq_complexs::MoveData( const int &location);
   template int precond_ilu_csr_seq_complexd::MoveData( const int &location);

   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::Solve( VectorType &x, VectorType &rhs)
   {
      /* the solve phase of ilut */
      if(this->_option != kIluOptionILUT && this->_option != kIluOptionILUK)
      {
         /* The solve of Partial ILUT is not supported directly, call SolveL and SolveU instead */
         PARGEMSLR_ERROR("Solve phase of ILU only supports ILUT yet.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      PARGEMSLR_CHKERR(this->_n != x.GetLengthLocal() || this->_n != rhs.GetLengthLocal());
      
      if(this->_n == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      int loc_x, loc_y;
      
      loc_x = x.GetDataLocation();
      loc_y = rhs.GetDataLocation();
      
      switch(this->_location)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matvec on device */
            PARGEMSLR_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
            PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
            
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
                  PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                  
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
      
      return this->SolveHost(x, rhs);
   }
   template int precond_ilu_csr_seq_float::Solve( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::Solve( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::Solve( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::Solve( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveL( VectorType &x, VectorType &rhs)
   {
      /* the solve phase of ilut */
      if(this->_option != kIluOptionPartialILUT)
      {
         /* The solve of Partial ILUT is not supported directly, call SolveL and SolveU instead */
         PARGEMSLR_ERROR("Solve with L only supports Partial ILUT yet.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      PARGEMSLR_CHKERR(this->_nB != x.GetLengthLocal() || this->_nB != rhs.GetLengthLocal());
      
      if(this->_nB == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      int loc_x, loc_y;
      
      loc_x = x.GetDataLocation();
      loc_y = rhs.GetDataLocation();
      
      switch(this->_location)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matvec on device */
            PARGEMSLR_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
            PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
            
            return this->SolveLDevice(x, rhs);
            
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
                  
                  return this->SolveLDevice(x, rhs);
                  
                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                  {
                     /* apply matvec on device */
                     return this->SolveLDevice(x, rhs);
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
      
      return this->SolveLHost(x, rhs);
   }
   template int precond_ilu_csr_seq_float::SolveL( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveL( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveL( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveL( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveU( VectorType &x, VectorType &rhs)
   {
      /* the solve phase of ilut */
      if(this->_option != kIluOptionPartialILUT)
      {
         /* The solve of Partial ILUT is not supported directly, call SolveL and SolveU instead */
         PARGEMSLR_ERROR("Solve with U only supports Partial ILUT yet.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      PARGEMSLR_CHKERR(this->_nB != x.GetLengthLocal() || this->_nB != rhs.GetLengthLocal());
      
      if(this->_nB == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      int loc_x, loc_y;
      
      loc_x = x.GetDataLocation();
      loc_y = rhs.GetDataLocation();
      
      switch(this->_location)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matvec on device */
            PARGEMSLR_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
            PARGEMSLR_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
            
            return this->SolveUDevice(x, rhs);
            
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
                  
                  return this->SolveUDevice(x, rhs);
                  
                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                  {
                     /* apply matvec on device */
                     return this->SolveUDevice(x, rhs);
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
      
      return this->SolveUHost(x, rhs);
   }
   template int precond_ilu_csr_seq_float::SolveU( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveU( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveU( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveU( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveDevice( VectorType &x, VectorType &rhs)
   {
      PARGEMSLR_CHKERR(this->_x_temp.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR(this->_x_temp.GetDataLocation() == kMemoryPinned);
      
      /* define the data type */
      typedef DataType T;
      
      /* the solve phase of ilut */
      T     one;
      
      one = 1.0;
      
      /* call cusparse to solve */
      if(this->_row_perm_vec.GetLengthLocal() > 0)
      {
         this->_row_perm_vec.GatherPerm(rhs, this->_x_temp);
         switch(this->_solver_precision)
         {
            case kSingleReal:
            {
               /* float */
               /* L solve - Forward solve */
               PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (float *) &one,
                                                            parallel_log::_matL_des,
                                                            (float *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matL_info, 
                                                            (float *) this->_x_temp.GetData(), 
                                                            (float *) x.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               
               /* U solve - Backward substitution */
               PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (float *) &one,
                                                            parallel_log::_matU_des,
                                                            (float *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matU_info, 
                                                            (float *) x.GetData(),
                                                            (float *) this->_x_temp.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               break;
            }
            case kDoubleReal:
            {
               /* double */
               /* L solve - Forward solve */
               PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (double *) &one,
                                                            parallel_log::_matL_des,
                                                            (double *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matL_info, 
                                                            (double *) this->_x_temp.GetData(), 
                                                            (double *) x.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               
               /* U solve - Backward substitution */
               PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (double *) &one,
                                                            parallel_log::_matU_des,
                                                            (double *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matU_info, 
                                                            (double *) x.GetData(),
                                                            (double *) this->_x_temp.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               break;
            }
            case kSingleComplex:
            {
               /* L solve - Forward solve */
               PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (cuComplex *) &one,
                                                            parallel_log::_matL_des,
                                                            (cuComplex *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matL_info, 
                                                            (cuComplex *) this->_x_temp.GetData(), 
                                                            (cuComplex *) x.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               
               /* U solve - Backward substitution */
               PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (cuComplex *) &one,
                                                            parallel_log::_matU_des,
                                                            (cuComplex *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matU_info, 
                                                            (cuComplex *) x.GetData(),
                                                            (cuComplex *) this->_x_temp.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               break;
            }
            case kDoubleComplex:
            {
               /* L solve - Forward solve */
               PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (cuDoubleComplex *) &one,
                                                            parallel_log::_matL_des,
                                                            (cuDoubleComplex *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matL_info, 
                                                            (cuDoubleComplex *) this->_x_temp.GetData(), 
                                                            (cuDoubleComplex *) x.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               
               /* U solve - Backward substitution */
               PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (cuDoubleComplex *) &one,
                                                            parallel_log::_matU_des,
                                                            (cuDoubleComplex *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matU_info, 
                                                            (cuDoubleComplex *) x.GetData(),
                                                            (cuDoubleComplex *) this->_x_temp.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unkown precision for ILU move data.");
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
         if(this->_col_perm_vec.GetLengthLocal() > 0)
         {
            this->_col_perm_vec.ScatterRperm(this->_x_temp, x);
         }
         else
         {
            this->_row_perm_vec.ScatterRperm(this->_x_temp, x);
         }
      }
      else
      {
         switch(this->_solver_precision)
         {
            case kSingleReal:
            {
               /* float */
               /* L solve - Forward solve */
               PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (float *) &one,
                                                            parallel_log::_matL_des,
                                                            (float *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matL_info, 
                                                            (float *) rhs.GetData(), 
                                                            (float *) this->_x_temp.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               
               /* U solve - Backward substitution */
               PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (float *) &one,
                                                            parallel_log::_matU_des,
                                                            (float *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matU_info, 
                                                            (float *) this->_x_temp.GetData(),
                                                            (float *) x.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               break;
            }
            case kDoubleReal:
            {
               /* double */
               /* L solve - Forward solve */
               PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (double *) &one,
                                                            parallel_log::_matL_des,
                                                            (double *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matL_info, 
                                                            (double *) rhs.GetData(), 
                                                            (double *) this->_x_temp.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               
               /* U solve - Backward substitution */
               PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (double *) &one,
                                                            parallel_log::_matU_des,
                                                            (double *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matU_info, 
                                                            (double *) this->_x_temp.GetData(),
                                                            (double *) x.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               break;
            }
            case kSingleComplex:
            {
               /* L solve - Forward solve */
               PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (cuComplex *) &one,
                                                            parallel_log::_matL_des,
                                                            (cuComplex *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matL_info, 
                                                            (cuComplex *) rhs.GetData(), 
                                                            (cuComplex *) this->_x_temp.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               
               /* U solve - Backward substitution */
               PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (cuComplex *) &one,
                                                            parallel_log::_matU_des,
                                                            (cuComplex *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matU_info, 
                                                            (cuComplex *) this->_x_temp.GetData(),
                                                            (cuComplex *) x.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               break;
            }
            case kDoubleComplex:
            {
               /* L solve - Forward solve */
               PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (cuDoubleComplex *) &one,
                                                            parallel_log::_matL_des,
                                                            (cuDoubleComplex *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matL_info, 
                                                            (cuDoubleComplex *) rhs.GetData(), 
                                                            (cuDoubleComplex *) this->_x_temp.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               
               /* U solve - Backward substitution */
               PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_solve(parallel_log::_cusparse_handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                            this->_n,
                                                            this->_LDU.GetNumNonzeros(),
                                                            (cuDoubleComplex *) &one,
                                                            parallel_log::_matU_des,
                                                            (cuDoubleComplex *) this->_LDU.GetData(), 
                                                            this->_LDU.GetI(), 
                                                            this->_LDU.GetJ(), 
                                                            this->_matU_info, 
                                                            (cuDoubleComplex *) this->_x_temp.GetData(),
                                                            (cuDoubleComplex *) x.GetData(), 
                                                            parallel_log::_ilu_solve_policy, 
                                                            parallel_log::_cusparse_buffer));
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unkown precision for ILU move data.");
               return PARGEMSLR_ERROR_INVALED_PARAM;
            }
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SolveDevice( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveDevice( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveDevice( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveDevice( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveLDevice( VectorType &x, VectorType &rhs)
   {
      PARGEMSLR_CHKERR(this->_option != kIluOptionPartialILUT);
      PARGEMSLR_CHKERR(this->_x_temp.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR(this->_x_temp.GetDataLocation() == kMemoryPinned);
      
      /* define the data type */
      typedef DataType T;
      
      /* the solve phase of ilut */
      T     one;
      
      one = 1.0;
   
      switch(this->_solver_precision)
      {
         case kSingleReal:
         {
            /* float */
            /* L solve - Forward solve */
            PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_solve(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                         this->_nB,
                                                         this->_LDU.GetNumNonzeros(),
                                                         (float *) &one,
                                                         parallel_log::_matL_des,
                                                         (float *) this->_LDU.GetData(), 
                                                         this->_LDU.GetI(), 
                                                         this->_LDU.GetJ(), 
                                                         this->_matL_info, 
                                                         (float *) rhs.GetData(), 
                                                         (float *) x.GetData(), 
                                                         parallel_log::_ilu_solve_policy, 
                                                         parallel_log::_cusparse_buffer));
            
            break;
         }
         case kDoubleReal:
         {
            /* double */
            /* L solve - Forward solve */
            PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_solve(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                         this->_nB,
                                                         this->_LDU.GetNumNonzeros(),
                                                         (double *) &one,
                                                         parallel_log::_matL_des,
                                                         (double *) this->_LDU.GetData(), 
                                                         this->_LDU.GetI(), 
                                                         this->_LDU.GetJ(), 
                                                         this->_matL_info, 
                                                         (double *) rhs.GetData(), 
                                                         (double *) x.GetData(), 
                                                         parallel_log::_ilu_solve_policy, 
                                                         parallel_log::_cusparse_buffer));
            
            break;
         }
         case kSingleComplex:
         {
            /* L solve - Forward solve */
            PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_solve(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                         this->_nB,
                                                         this->_LDU.GetNumNonzeros(),
                                                         (cuComplex *) &one,
                                                         parallel_log::_matL_des,
                                                         (cuComplex *) this->_LDU.GetData(), 
                                                         this->_LDU.GetI(), 
                                                         this->_LDU.GetJ(), 
                                                         this->_matL_info, 
                                                         (cuComplex *) rhs.GetData(), 
                                                         (cuComplex *) x.GetData(), 
                                                         parallel_log::_ilu_solve_policy, 
                                                         parallel_log::_cusparse_buffer));
            
            break;
         }
         case kDoubleComplex:
         {
            /* L solve - Forward solve */
            PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_solve(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                         this->_nB,
                                                         this->_LDU.GetNumNonzeros(),
                                                         (cuDoubleComplex *) &one,
                                                         parallel_log::_matL_des,
                                                         (cuDoubleComplex *) this->_LDU.GetData(), 
                                                         this->_LDU.GetI(), 
                                                         this->_LDU.GetJ(), 
                                                         this->_matL_info, 
                                                         (cuDoubleComplex *) rhs.GetData(), 
                                                         (cuDoubleComplex *) x.GetData(), 
                                                         parallel_log::_ilu_solve_policy, 
                                                         parallel_log::_cusparse_buffer));
            
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unkown precision for ILU move data.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SolveLDevice( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveLDevice( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveLDevice( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveLDevice( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveUDevice( VectorType &x, VectorType &rhs)
   {
      PARGEMSLR_CHKERR(this->_option != kIluOptionPartialILUT);
      PARGEMSLR_CHKERR(this->_x_temp.GetDataLocation() == kMemoryHost);
      PARGEMSLR_CHKERR(this->_x_temp.GetDataLocation() == kMemoryPinned);
      
      /* define the data type */
      typedef DataType T;
      
      /* the solve phase of ilut */
      T     one;
      
      one = 1.0;
   
      switch(this->_solver_precision)
      {
         case kSingleReal:
         {
            /* float */
            
            /* U solve - Backward substitution */
            PARGEMSLR_CUSPARSE_CALL(cusparseScsrsv2_solve(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                         this->_nB,
                                                         this->_LDU.GetNumNonzeros(),
                                                         (float *) &one,
                                                         parallel_log::_matU_des,
                                                         (float *) this->_LDU.GetData(), 
                                                         this->_LDU.GetI(), 
                                                         this->_LDU.GetJ(), 
                                                         this->_matU_info, 
                                                         (float *) rhs.GetData(),
                                                         (float *) x.GetData(), 
                                                         parallel_log::_ilu_solve_policy, 
                                                         parallel_log::_cusparse_buffer));
            break;
         }
         case kDoubleReal:
         {
            /* double */
            
            /* U solve - Backward substitution */
            PARGEMSLR_CUSPARSE_CALL(cusparseDcsrsv2_solve(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                         this->_nB,
                                                         this->_LDU.GetNumNonzeros(),
                                                         (double *) &one,
                                                         parallel_log::_matU_des,
                                                         (double *) this->_LDU.GetData(), 
                                                         this->_LDU.GetI(), 
                                                         this->_LDU.GetJ(), 
                                                         this->_matU_info, 
                                                         (double *) rhs.GetData(),
                                                         (double *) x.GetData(), 
                                                         parallel_log::_ilu_solve_policy, 
                                                         parallel_log::_cusparse_buffer));
            break;
         }
         case kSingleComplex:
         {
            
            /* U solve - Backward substitution */
            PARGEMSLR_CUSPARSE_CALL(cusparseCcsrsv2_solve(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                         this->_nB,
                                                         this->_LDU.GetNumNonzeros(),
                                                         (cuComplex *) &one,
                                                         parallel_log::_matU_des,
                                                         (cuComplex *) this->_LDU.GetData(), 
                                                         this->_LDU.GetI(), 
                                                         this->_LDU.GetJ(), 
                                                         this->_matU_info, 
                                                         (cuComplex *) rhs.GetData(),
                                                         (cuComplex *) x.GetData(), 
                                                         parallel_log::_ilu_solve_policy, 
                                                         parallel_log::_cusparse_buffer));
            break;
         }
         case kDoubleComplex:
         {
            
            /* U solve - Backward substitution */
            PARGEMSLR_CUSPARSE_CALL(cusparseZcsrsv2_solve(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                         this->_nB,
                                                         this->_LDU.GetNumNonzeros(),
                                                         (cuDoubleComplex *) &one,
                                                         parallel_log::_matU_des,
                                                         (cuDoubleComplex *) this->_LDU.GetData(), 
                                                         this->_LDU.GetI(), 
                                                         this->_LDU.GetJ(), 
                                                         this->_matU_info, 
                                                         (cuDoubleComplex *) rhs.GetData(),
                                                         (cuDoubleComplex *) x.GetData(), 
                                                         parallel_log::_ilu_solve_policy, 
                                                         parallel_log::_cusparse_buffer));
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unkown precision for ILU move data.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SolveUDevice( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveUDevice( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveUDevice( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveUDevice( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
}
#endif
