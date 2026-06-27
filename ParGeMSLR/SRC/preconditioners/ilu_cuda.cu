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


#if PARGEMSLR_CUSPARSE_GENERIC_API
   template <typename T>
   cudaDataType PargemslrCudaDataType();

   template <>
   cudaDataType PargemslrCudaDataType<float>()
   {
      return CUDA_R_32F;
   }

   template <>
   cudaDataType PargemslrCudaDataType<double>()
   {
      return CUDA_R_64F;
   }

   template <>
   cudaDataType PargemslrCudaDataType<complexs>()
   {
      return CUDA_C_32F;
   }

   template <>
   cudaDataType PargemslrCudaDataType<complexd>()
   {
      return CUDA_C_64F;
   }

   template <typename T>
   int PargemslrCreateIluSpMat(cusparseSpMatDescr_t *mat, int n, int nnz, int *row, int *col, T *data,
                              cusparseFillMode_t fill_mode, cusparseDiagType_t diag_type)
   {
      PARGEMSLR_CUSPARSE_CALL(cusparseCreateCsr(mat,
                                                n,
                                                n,
                                                nnz,
                                                row,
                                                col,
                                                data,
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_type,
                                                parallel_log::_cusparse_idx_base,
                                                PargemslrCudaDataType<T>()));
      PARGEMSLR_CUSPARSE_CALL(cusparseSpMatSetAttribute(*mat,
                                                        CUSPARSE_SPMAT_FILL_MODE,
                                                        &fill_mode,
                                                        sizeof(fill_mode)));
      PARGEMSLR_CUSPARSE_CALL(cusparseSpMatSetAttribute(*mat,
                                                        CUSPARSE_SPMAT_DIAG_TYPE,
                                                        &diag_type,
                                                        sizeof(diag_type)));
      return PARGEMSLR_SUCCESS;
   }

   template <typename T>
   int PargemslrSpSVSolve(cusparseSpMatDescr_t mat, cusparseSpSVDescr_t info, const T &alpha,
                         cusparseDnVecDescr_t x, cusparseDnVecDescr_t y)
   {
      PARGEMSLR_CUSPARSE_CALL(cusparseSpSV_solve(parallel_log::_cusparse_handle,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha,
                                                mat,
                                                x,
                                                y,
                                                PargemslrCudaDataType<T>(),
                                                CUSPARSE_SPSV_ALG_DEFAULT,
                                                info));
      return PARGEMSLR_SUCCESS;
   }

#endif
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::Clear()
   {
      SolverClass<MatrixType, VectorType, DataType>::Clear();
      this->_LDU.Clear();
#if PARGEMSLR_CUSPARSE_GENERIC_API
      if(this->_matL_spsv_info)
      {
         PARGEMSLR_CUSPARSE_CALL( cusparseSpSV_destroyDescr(this->_matL_spsv_info) );
         this->_matL_spsv_info = NULL;
      }
      if(this->_matU_spsv_info)
      {
         PARGEMSLR_CUSPARSE_CALL( cusparseSpSV_destroyDescr(this->_matU_spsv_info) );
         this->_matU_spsv_info = NULL;
      }
      if(this->_matL_info)
      {
         PARGEMSLR_CUSPARSE_CALL( cusparseDestroySpMat(this->_matL_info) );
         this->_matL_info = NULL;
      }
      if(this->_matU_info)
      {
         PARGEMSLR_CUSPARSE_CALL( cusparseDestroySpMat(this->_matU_info) );
         this->_matU_info = NULL;
      }
      if(this->_matL_spsv_buffer)
      {
         PARGEMSLR_FREE(this->_matL_spsv_buffer, kMemoryDevice);
      }
      this->_matL_spsv_buffer_length = 0;
      if(this->_matU_spsv_buffer)
      {
         PARGEMSLR_FREE(this->_matU_spsv_buffer, kMemoryDevice);
      }
      this->_matU_spsv_buffer_length = 0;
#else
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
#endif
      this->_cusparse_ready = false;
      this->_location = kMemoryHost;
      this->_n = 0;
      this->_diag_shift_milu = 0;
      this->_nnz = 0;
      this->_L.Clear();
      this->_D.Clear();
      this->_U.Clear();
      this->_E.Clear();
      this->_F.Clear();
      this->_S.Clear();
      this->_nB = 0;
#ifdef PARGEMSLR_OPENMP
      this->_L_poly.Clear();
      this->_D_poly.Clear();
      this->_U_poly.Clear();
      this->_y_temp.Clear();
      this->_z_temp.Clear();
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
      this->_fill_level = 1;
      this->_max_row_nnz = 100;
      this->_max_row_nnz_s = 200;
      this->_option = kIluOptionILUT;
      this->_perm_option = kIluReorderingRcm;
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

      if( location == kMemoryHost )
      {
         return PARGEMSLR_SUCCESS;
      }
      if( this->_LDU.GetNumNonzeros() > 0 && location == this->_LDU.GetDataLocation() && this->_cusparse_ready )
      {
         return PARGEMSLR_SUCCESS;
      }

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

         nnzLDU = this->_LDU.GetNumNonzeros();

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

      if(this->_x_temp.GetLengthLocal() != n)
      {
         this->_x_temp.Setup( n, location, true);
      }
      else
      {
         this->_x_temp.MoveData(location);
      }

      /* now starting setup the solver when necessary */
      if( location == kMemoryDevice || location == kMemoryUnified )
      {

#if PARGEMSLR_CUSPARSE_GENERIC_API
         size_t         matL_buffersize, matU_buffersize;
         T              one;

         one = 1.0;

         this->_cusparse_ready = false;

         if(this->_matL_spsv_info)
         {
            PARGEMSLR_CUSPARSE_CALL( cusparseSpSV_destroyDescr(this->_matL_spsv_info) );
            this->_matL_spsv_info = NULL;
         }
         if(this->_matU_spsv_info)
         {
            PARGEMSLR_CUSPARSE_CALL( cusparseSpSV_destroyDescr(this->_matU_spsv_info) );
            this->_matU_spsv_info = NULL;
         }
         if(this->_matL_info)
         {
            PARGEMSLR_CUSPARSE_CALL( cusparseDestroySpMat(this->_matL_info) );
            this->_matL_info = NULL;
         }
         if(this->_matU_info)
         {
            PARGEMSLR_CUSPARSE_CALL( cusparseDestroySpMat(this->_matU_info) );
            this->_matU_info = NULL;
         }

         PargemslrCreateIluSpMat(&(this->_matL_info), n, nnzLDU, LDU_i, LDU_j, LDU_a,
                                 CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_UNIT);
         PargemslrCreateIluSpMat(&(this->_matU_info), n, nnzLDU, LDU_i, LDU_j, LDU_a,
                                 CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);
         PARGEMSLR_CUSPARSE_CALL(cusparseSpSV_createDescr(&(this->_matL_spsv_info)));
         PARGEMSLR_CUSPARSE_CALL(cusparseSpSV_createDescr(&(this->_matU_spsv_info)));

         PARGEMSLR_CUSPARSE_CALL(cusparseSpSV_bufferSize(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &one,
                                                         this->_matL_info,
                                                         this->_x_temp.GetCusparseVec(),
                                                         this->_x_temp.GetCusparseVec(),
                                                         PargemslrCudaDataType<T>(),
                                                         CUSPARSE_SPSV_ALG_DEFAULT,
                                                         this->_matL_spsv_info,
                                                         &matL_buffersize));
         PARGEMSLR_CUSPARSE_CALL(cusparseSpSV_bufferSize(parallel_log::_cusparse_handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &one,
                                                         this->_matU_info,
                                                         this->_x_temp.GetCusparseVec(),
                                                         this->_x_temp.GetCusparseVec(),
                                                         PargemslrCudaDataType<T>(),
                                                         CUSPARSE_SPSV_ALG_DEFAULT,
                                                         this->_matU_spsv_info,
                                                         &matU_buffersize));

         if( matL_buffersize > this->_matL_spsv_buffer_length )
         {
            PARGEMSLR_REALLOC_VOID( this->_matL_spsv_buffer, this->_matL_spsv_buffer_length, matL_buffersize, kMemoryDevice)
            this->_matL_spsv_buffer_length = matL_buffersize;
         }
         if( matU_buffersize > this->_matU_spsv_buffer_length )
         {
            PARGEMSLR_REALLOC_VOID( this->_matU_spsv_buffer, this->_matU_spsv_buffer_length, matU_buffersize, kMemoryDevice)
            this->_matU_spsv_buffer_length = matU_buffersize;
         }

         PARGEMSLR_CUSPARSE_CALL(cusparseSpSV_analysis(parallel_log::_cusparse_handle,
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       &one,
                                                       this->_matL_info,
                                                       this->_x_temp.GetCusparseVec(),
                                                       this->_x_temp.GetCusparseVec(),
                                                       PargemslrCudaDataType<T>(),
                                                       CUSPARSE_SPSV_ALG_DEFAULT,
                                                       this->_matL_spsv_info,
                                                       this->_matL_spsv_buffer));
         PARGEMSLR_CUSPARSE_CALL(cusparseSpSV_analysis(parallel_log::_cusparse_handle,
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       &one,
                                                       this->_matU_info,
                                                       this->_x_temp.GetCusparseVec(),
                                                       this->_x_temp.GetCusparseVec(),
                                                       PargemslrCudaDataType<T>(),
                                                       CUSPARSE_SPSV_ALG_DEFAULT,
                                                       this->_matU_spsv_info,
                                                       this->_matU_spsv_buffer));
         this->_cusparse_ready = true;
#else
         int            matL_buffersize, matU_buffersize, bufferSize;

         this->_cusparse_ready = false;

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
         this->_cusparse_ready = true;
#endif
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
      if(!this->_cusparse_ready)
      {
         int err = this->MoveData(this->_location);
         if(err != PARGEMSLR_SUCCESS)
         {
            return err;
         }
      }

      /* define the data type */
      typedef DataType T;

      /* the solve phase of ilut */
      T     one;

      one = 1.0;

#if PARGEMSLR_CUSPARSE_GENERIC_API
      if(this->_row_perm_vec.GetLengthLocal() > 0)
      {
         this->_row_perm_vec.GatherPerm(rhs, this->_x_temp);
         PargemslrSpSVSolve(this->_matL_info, this->_matL_spsv_info, one,
                            this->_x_temp.GetCusparseVec(), x.GetCusparseVec());
         PargemslrSpSVSolve(this->_matU_info, this->_matU_spsv_info, one,
                            x.GetCusparseVec(), this->_x_temp.GetCusparseVec());
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
         PargemslrSpSVSolve(this->_matL_info, this->_matL_spsv_info, one,
                            rhs.GetCusparseVec(), this->_x_temp.GetCusparseVec());
         PargemslrSpSVSolve(this->_matU_info, this->_matU_spsv_info, one,
                            this->_x_temp.GetCusparseVec(), x.GetCusparseVec());
      }

      return PARGEMSLR_SUCCESS;
#else
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

#endif
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
      if(!this->_cusparse_ready)
      {
         int err = this->MoveData(this->_location);
         if(err != PARGEMSLR_SUCCESS)
         {
            return err;
         }
      }

      /* define the data type */
      typedef DataType T;

      /* the solve phase of ilut */
      T     one;

      one = 1.0;

#if PARGEMSLR_CUSPARSE_GENERIC_API
      PargemslrSpSVSolve(this->_matL_info, this->_matL_spsv_info, one,
                         rhs.GetCusparseVec(), x.GetCusparseVec());
      return PARGEMSLR_SUCCESS;
#else
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

#endif
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
      if(!this->_cusparse_ready)
      {
         int err = this->MoveData(this->_location);
         if(err != PARGEMSLR_SUCCESS)
         {
            return err;
         }
      }

      /* define the data type */
      typedef DataType T;

      /* the solve phase of ilut */
      T     one;

      one = 1.0;

#if PARGEMSLR_CUSPARSE_GENERIC_API
      PargemslrSpSVSolve(this->_matU_info, this->_matU_spsv_info, one,
                         rhs.GetCusparseVec(), x.GetCusparseVec());
      return PARGEMSLR_SUCCESS;
#else
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

#endif
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SolveUDevice( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveUDevice( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveUDevice( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveUDevice( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);

}
#endif
