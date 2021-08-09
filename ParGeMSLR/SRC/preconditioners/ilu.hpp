#ifndef PARGEMSLR_ILU_H
#define PARGEMSLR_ILU_H

/**
 * @file ilu.hpp
 * @brief ILU preconditioner.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../solvers/solver.hpp"

/* IMPORTANT NOTE:
 * this file contains some functions from the package HYPRE 
 * parcsr_ls/par_ilu_setup.c
 * parcsr_ls/par_ilu_solve.c
 * 
 * Modifications are made to fit the ParGEMSLR data structure
 * the following is the license statement of HYPRE
 * 
 * Include the following functions:
 * 
 * ----------------------------------------------------------------------------------------------------------------
 * 
 * Intellectual Property Notice
 * ----------------------------
 * HYPRE is licensed under the Apache License, Version 2.0 (see LICENSE-APACHE or
 * http://www.apache.org/licenses/LICENSE-2.0) or the MIT license (see LICENSE-MIT
 * or http://opensource.org/licenses/MIT), at your option.
 * 
 * Copyrights and patents in the HYPRE project are retained by contributors.
 * No copyright assignment is required to contribute to HYPRE.
 * 
 * HYPRE's original open source license was LGPL-2.1.  Consent from contributors
 * was received and documented before relicensing to Apache-2.0/MIT.
 * 
 * SPDX usage
 * ----------
 * 
 * Individual files contain SPDX tags instead of the full license text.
 * This enables machine processing of license information based on the SPDX
 * License Identifiers that are available here: https://spdx.org/licenses/
 * 
 * Files that are dual-licensed as Apache-2.0 OR MIT contain the following
 * text in the license header:
 * 
 *     SPDX-License-Identifier: (Apache-2.0 OR MIT)
 * 
 * 
 * External Software
 * -----------------
 * 
 * External software in hypre is covered by various permissive licenses.  A summary
 * listing follows.  See the license included with the software for full details.
 * 
 * Directory: src/blas
 * License: University of Tennessee
 * 
 * Directory: src/lapack
 * License: University of Tennessee
 * 
 * ----------------------------------------------------------------------------------------------------------------
 * 
 */

#ifdef PARGEMSLR_CUDA
#include "cusparse.h"
#endif

namespace pargemslr
{
   
   /**
    * @brief   The ilu option.
    * @details The ilu option.
    */
   enum IluOptionEnum
   {
      kIluOptionILUT,
      kIluOptionPartialILUT,
      kIluOptionILUK
   };
   
   /**
    * @brief   The ilu reordering option.
    * @details The ilu reordering option.
    * @note    kIluReorderingInput: use user input option
    */
   enum IluReorderingOptionEnum
   {
      kIluReorderingNo,
      kIluReorderingRcm,
      kIluReorderingAmd,
      kIluReorderingNd,
      kIluReorderingDdpq,
      kIluReorderingInput
   };
   
   /**
    * @brief   The ilu OpenMP solve option.
    * @details The ilu OpenMP solve option.
    * @note    Note that ILUT + RCM might not suiteble for LevelScheduling solve.
    */
   enum IluOpenMPOptionEnum
   {
      kIluOpenMPNo,
      kIluOpenMPLevelScheduling,
      kIluOpenMPPoly
   };
   
   /**
    * @brief   The local ilu preconditioner, only work for sequential CSR matrix.
    * @details The local ilu preconditioner, only work for sequential CSR matrix.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class IluClass: public SolverClass<MatrixType, VectorType, DataType>
   {
   private:
      
      /** 
       * @brief   The diagonal shift for the modified ILU.
       * @details The diagonal shift for the modified ILU.
       */
      DataType                            _diag_shift_milu;
      
      /** 
       * @brief   The temp vector for permuted x.
       * @details The temp vector for permuted x.
       */
      VectorType                          _x_temp;
      
#ifdef PARGEMSLR_CUDA

      /**
       * @brief   The ILU factorization in one single sparse matrix for cusparse.
       * @details The ILU factorization in one single sparse matrix for cusparse.
       */
      CsrMatrixClass<DataType>            _LDU;
      
      /**
       * @brief   The ILU info for L matrix of cusparse.
       * @details The ILU info for L matrix of cusparse.
       */
      csrsv2Info_t                        _matL_info;
      
      /**
       * @brief   The ILU info for U matrix of cusparse.
       * @details The ILU info for U matrix of cusparse.
       */
      csrsv2Info_t                        _matU_info;
      
#endif
      
      /**
       * @brief   The size of the problem.
       * @details The size of the problem.
       */
      int                                 _n;
      
      /**
       * @brief   The number of nonzeros.
       * @details The number of nonzeros.
       */
      int                                 _nnz;
      
      /**
       * @brief   The complex shift. 0: disable. <0: dynamic. >0: fixed value.
       * @details The complex shift. 0: disable. <0: dynamic. >0: fixed value. \n
       *          0: No complex shift used.
       *          <0: Dynamic shift. Automatically decide the shift on each row. \n
       *          >0: Fixed shift. Shift by \sum_i |A_{ii}|/n * val. Val is this value.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _complex_shift;
      
      /**
       * @brief   The L matrix of the ILU factorizaiton without diagonal.
       * @details The L matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>            _L;
      
      /**
       * @brief   The inverse of diagonal of the U matrix.
       * @details The inverse of diagonal of the U matrix.
       */
      SequentialVectorClass<DataType>     _D;
      
      /**
       * @brief   The U matrix of the ILU factorizaiton without diagonal.
       * @details The U matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>            _U;
      
      /**
       * @brief   Number of nodes in the B block for Partial ILU.
       * @details Number of nodes in the B block for Partial ILU.
       */
      int                                 _nB;
      
      /**
       * @brief   The L^{-1}F matrix for the Partial ILU.
       * @details The L^{-1}F matrix for the Partial ILU.
       */
      CsrMatrixClass<DataType>            _E;
      
      /**
       * @brief   The EU^{-1} matrix for the Partial ILU.
       * @details The EU^{-1} matrix for the Partial ILU.
       */
      CsrMatrixClass<DataType>            _F;
      
      /**
       * @brief   The S matrix for the Partial ILU.
       * @details The S matrix for the Partial ILU.
       */
      CsrMatrixClass<DataType>            _S;
      
#ifdef PARGEMSLR_OPENMP
      
      /** 
       * @brief   The temp vector for permuted x.
       * @details The temp vector for permuted x.
       */
      VectorType                          _y_temp;
      
      /** 
       * @brief   The temp vector for permuted x.
       * @details The temp vector for permuted x.
       */
      VectorType                          _z_temp;
      
      /**
       * @brief   The L matrix of the ILU factorizaiton without diagonal.
       * @details The L matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>            _L_poly;
      
      /**
       * @brief   The inverse of diagonal of the U matrix.
       * @details The inverse of diagonal of the U matrix.
       */
      CsrMatrixClass<DataType>            _D_poly;
      
      /**
       * @brief   The U matrix of the ILU factorizaiton without diagonal.
       * @details The U matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>            _U_poly;
      
      /**
       * @brief   The L matrix of the ILU factorizaiton without diagonal.
       * @details The L matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>            _L_level;
      
      /**
       * @brief   The inverse of diagonal of the U matrix.
       * @details The inverse of diagonal of the U matrix.
       */
      SequentialVectorClass<DataType>     _D_level;
      
      /**
       * @brief   The U matrix of the ILU factorizaiton without diagonal.
       * @details The U matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>            _U_level;
      
      /**                                                                                                                                                                               
       * @brief   Pointer to the start of L levels.
       * @details Pointer to the start of L levels.
       */
      vector_int                          _level_ptr_l;
      
      /**                                                                                                                                                                               
       * @brief   Pointer to the start of L levels.
       * @details Pointer to the start of L levels.
       */
      vector_int                          _levels_l;
      
      /**                                                                                                                                                                               
       * @brief   Pointer to the start of L levels.
       * @details Pointer to the start of L levels.
       */
      vector_int                          _level_ptr_u;
      
      /**                                                                                                                                                                               
       * @brief   Pointer to the start of L levels.
       * @details Pointer to the start of L levels.
       */
      vector_int                          _levels_u;
      
      /**                                                                                                                                                                               
       * @brief   The start point of level l.
       * @details The start point of level l.
       */
      int                                 _levels_l_start;
      
      /**                                                                                                                                                                               
       * @brief   The end point of level l.
       * @details The end point of level l.
       */
      int                                 _levels_l_end;
      
      /**                                                                                                                                                                               
       * @brief   The start point of level u.
       * @details The start point of level u.
       */
      int                                 _levels_u_start;
      
      /**                                                                                                                                                                               
       * @brief   The end point of level u.
       * @details The end point of level u.
       */
      int                                 _levels_u_end;
      
#endif
      
      /**
       * @brief   Set the fill level for ILU(K).
       * @details Set the fill level for ILU(K).
       */
      int                                 _fill_level;
      
      /**
       * @brief   Set the drop tols for ILUT.
       * @details Set the drop tols for ILUT.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _droptol;
      
      /**
       * @brief   Set the drop tols for PartialILUT in the E and F part.
       * @details Set the drop tols for PartialILUT in the E and F part.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _droptol_ef;
      
      /**
       * @brief   Set the drop tols for PartialILUT in the S part.
       * @details Set the drop tols for PartialILUT in the S part.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _droptol_s;
      
      /**
       * @brief   Set the max fill allowed for each row/col of L and U.
       * @details Set the max fill allowed for each row/col of L and U.
       */
      int                                 _max_row_nnz;
      
      /**
       * @brief   Set the max fill allowed for each row/col of S for Partial ILUT.
       * @details Set the max fill allowed for each row/col of S for Partial ILUT.
       */
      int                                 _max_row_nnz_s;
      
      /**
       * @brief   The ILU option. See IluOptionEnum.
       * @details The ILU option. See IluOptionEnum.
       */
      int                                 _option;
      
      /**
       * @brief   The ILU openmp option. See IluOpenMPOptionEnum.
       * @details The ILU openmp option. See IluOpenMPOptionEnum.
       */
      int                                 _omp_option; 
      
      /**
       * @brief   The polynomia order for poly omp solve.
       * @details The polynomia order for poly omp solve.
       */
      int                                 _poly_order;
      
      /**
       * @brief   The ILU permutation option. See IluReorderingOptionEnum.
       * @details The ILU permutation option. See IluReorderingOptionEnum.
       */
      int                                 _perm_option;
      
      /**
       * @brief   Enbale modified ILU or not. Default is false.
       * @details Enbale modified ILU or not. Default is false.
       */
      bool                                _modified;
      
      /**
       * @brief   The row permutation vector.
       * @details The row permutation vector.
       */
      IntVectorClass<int>                 _row_perm_vec;
      
      /**
       * @brief   The column permutation vector.
       * @details The column permutation vector.
       */
      IntVectorClass<int>                 _col_perm_vec;
      
      /**
       * @brief   The location this preconditioner applied to.
       * @details The location this preconditioner applied to.
       */
      int                                 _location;
      
      /**
       * @brief   Temp value, we might need to move the matrix to the host to setup.
       * @details Temp value, we might need to move the matrix to the host to setup.
       */
      int                                 _matrix_location;
      
      /**
       * @brief   If the preconditioner is ready to be solved with cusparse.
       * @details If the preconditioner is ready to be solved with cusparse.
       */
      int                                 _cusparse_ready;
      
      /**                                                                                                                                                                               
       * @brief   Apply complex shift.
       * @details Apply complex shift.
       * @param [in/out]w the diagonal value.
       * @param [in]    nz the number of nonzeros in this row.
       * @param [in]    norm the norm.
       * @return     Return error message.
       */
      template <typename T>
      int ApplyShift( T &w, int nz, T norm);
      
      /**                                                                                                                                                                               
       * @brief   Apply complex shift.
       * @details Apply complex shift.
       * @note    From ZITSOL.
       * @param [in/out]w the diagonal value.
       * @param [in]    nz the number of nonzeros in this row.
       * @param [in]    norm the norm.
       * @return     Return error message.
       */
      template <typename T>
      int ApplyShift( ComplexValueClass<T> &w, int nz, T norm);
      
      /**                                                                                                                                                                               
       * @brief   Does a quick-sort split of a real array.
       * @details Does a quick-sort split of a real array.                                                                
       * @param [in/out]a the array to sort.
       * @param [in]    ind permuted indices.
       * @param [in]    n length of the array.
       * @param [in]    Ncut threshold.
       * @return     Return error message.
       */
      template <typename T>
      int         Qsplit(T *a, int *ind, int n, int Ncut); 

      /**
       * @brief   Symbolic factorization of ILU(K).
       * @details Symbolic factorization of ILU(K).
       * @param   [in]   n The size of the system.
       * @param   [in]   A_i The I of the CSR format of A.
       * @param   [in]   A_j The J of the CSR format of A.
       * @param   [in]   lfil The level of fill K.
       * @param   [in]   perm The row permutation.
       * @param   [in]   qperm The column permutation.
       * @param   [in]   rqperm The reverse column permutation.
       * @param   [in]   iw The working array of length 3*n.
       * @param   [in]   L_i The I of the CSR format of L, should be reserved.
       * @param   [in]   U_i The I of the CSR format of U, should be reserved.
       * @param   [out]  L_jp The pointer to the J of the CSR format of L.
       * @param   [out]  U_jp The pointer to the J of the CSR format of U.
       * @return     Return error message.
       */
      int SetupILUKSymbolic( int n, int *A_i, int *A_j, int lfil, int *perm, int *qperm, int *rqperm, int *iw, int *L_i, int *U_i, int **L_jp, int **U_jp);
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int SolveHost( VectorType &x, VectorType &rhs);

      /**
       * @brief   Solve phase, solve with L only. No permutation is applied, the size of the input x and rhs should be _nB.
       * @details Solve phase, solve with L only. No permutation is applied, the size of the input x and rhs should be _nB.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int SolveLHost( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Solve phase, solve with U only. No permutation is applied, the size of the input x and rhs should be _nB.
       * @details Solve phase, solve with U only. No permutation is applied, the size of the input x and rhs should be _nB.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int SolveUHost( VectorType &x, VectorType &rhs);
      
#ifdef PARGEMSLR_CUDA
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int SolveDevice( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Solve phase with L only. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase with L only. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      int SolveLDevice( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Solve phase with U only. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase with U only. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      int SolveUDevice( VectorType &x, VectorType &rhs);
      
#endif

#ifdef PARGEMSLR_OPENMP

      /**                                                                                                                                                                               
       * @brief   Setup level-scheduling.
       * @details Setup level-scheduling.
       * @return     Return error message.
       */
      int         BuildLevels();
      
      /**                                                                                                                                                                               
       * @brief   Setup the poly solver.
       * @details Setup the poly solver
       * @return     Return error message.
       */
      int         BuildPoly();
      
      /**
       * @brief   Solve phase with OpenMP.
       * @details Solve phase with OpenMP.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      int         SolveHostOmp( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Apply the inverse, option == 0: L, option == 1: D and U.
       * @details Apply the inverse. For example, L^{-1} = (I - L1)^{-1} \approx I + L1 + L1^2 + ... \n
       *          for example, (I + (I + (I + L1)*L1)*L1)b.
       * @note    _y_temp and _z_temp are used in this function.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      int         SolveHostOmpPoly( VectorType &x, VectorType &rhs, int option);
      
#endif

	public:

      /**
       * @brief   The constructor of precondioner class.
       * @details The constructor of precondioner class.
       */
      IluClass();
      
      /**
       * @brief   The copy constructor of precondioner class.
       * @details The copy constructor of precondioner class.
       */
      IluClass(const IluClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The move constructor of precondioner class.
       * @details The move constructor of precondioner class.
       */
      IluClass(IluClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   The operator = of precondioner class.
       * @details The operator = of precondioner class.
       */
      IluClass<MatrixType, VectorType, DataType>& operator=(const IluClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The operator = of precondioner class.
       * @details The operator = of precondioner class.
       */
      IluClass<MatrixType, VectorType, DataType>& operator=(IluClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   Free the current precondioner.
       * @details Free the current precondioner.
       * @return     Return error message.
       */
      virtual int Clear();
     
      /**
       * @brief   The destructor of precondioner class.
       * @details The destructor of precondioner class.
       */
      virtual ~IluClass();
      
      /**
       * @brief   Setup the precondioner phase. Will be called by the solver if not called directly.
       * @details Setup the precondioner phase. Will be called by the solver if not called directly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Setup ILU with exist factorization L, D, and U. The lower triangular part of L, inverse of diagonal of U, and the upper triangular part of U.
       * @details Setup ILU with exist factorization L, D, and U. The lower triangular part of L, inverse of diagonal of U, and the upper triangular part of U.
       * @param   [in]   L_mat the lower triangular part of L.
       * @param   [in]   D_vec the inverse of diagonal of U.
       * @param   [in]   U_mat the upper triangular part of U.
       * @param   [in]   pperm the row permutation vector, set to NULL to use the nature ordering.
       * @param   [in]   qperm the col permutation vector, set to NULL to use the nature ordering.
       * @return     Return error message.
       */
      int Setup(MatrixType &L_mat, VectorType &D_vec, MatrixType &U_mat, int *pperm, int *qperm);
      
      /**
       * @brief   Build the permutation vector.
       * @details Build the permutation vector.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      int SetupPermutation( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Setup ILUT, ingore the ilu option.
       * @details Setup ILUT, ingore the ilu option.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      int SetupILUT( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Setup ILU(K).
       * @details Setup ILU(K).
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      int SetupILUK( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Swap v[i] and v[j].
       * @details Swap v[i] and v[j].
       * @param   [in]   v The array.
       * @param   [in]   i The first index.
       * @param   [in]   j The second index.
       */
      template <typename T>
      static void Swap( T *v, int i, int j)
      {
         T temp;
         temp = v[i];
         v[i] = v[j];
         v[j] = temp;
      }
      
      /**
       * @brief   Swap v[i] and v[j], w[i] and w[j].
       * @details Swap v[i] and v[j], w[i] and w[j].
       * @param   [in]   v The first array.
       * @param   [in]   w The second array.
       * @param   [in]   i The first index.
       * @param   [in]   j The second index.
       */
      template <typename T1, typename T2>
      static void Swap(T1 *v, T2 *w, int i, int j)
      {
         T1 temp;
         T2 temp2;

         temp = v[i];
         v[i] = v[j];
         v[j] = temp;
         temp2 = w[i];
         w[i] = w[j];
         w[j] = temp2;
      }

      /**
       * @brief   Add value to a min heap based on the absolute value. RN stands for reverse (heap from right to left), normal (v1 based on normal order).
       * @details Add value to a min heap based on the absolute value. RN stands for reverse (heap from right to left), normal (v1 based on normal order).\n
       *          That is, the heap starts from heap[0], and goes to heap[-1], heap[-2] ... \n
       *          When swaping elements in the heap, elements in v1 swap in the same way (swap heap[i]<->heap[j] and v1[i]<->v1[j]).
       * @param   [in]   heap The heap.
       * @param   [in]   v1 The second array.
       * @param   [in]   v2 The third array.
       * @param   [in]   len The length.
       */
      template <typename T1, typename T2>
      static void AbsMinHeapAdd_RN(T1 *heap, T2 *v, int len)
      {
         /* parent, left, right */
         int p;
         len--;/* now len is the current index */
         while(len > 0)
         {
            /* get the parent index */
            p = (len-1)/2;
            if(PargemslrAbs(heap[-p]) < PargemslrAbs(heap[-len]))
            {
               /* this is smaller */
               IluClass<MatrixType, VectorType, DataType>::Swap<T2, T1>( v, heap, -p, -len);
               len = p;
            }
            else
            {
               break;
            }
         }
      }
      
      /**
       * @brief   Remove value from a min heap based on the absolute value. RN stands for reverse (heap from right to left), normal (v1 based on normal order).
       * @details Remove value from a min heap based on the absolute value. RN stands for reverse (heap from right to left), normal (v1 based on normal order).\n
       *          That is, the heap starts from heap[0], and goes to heap[-1], heap[-2] ... \n
       *          When swaping elements in the heap, elements in v1 swap in the same way (swap heap[i]<->heap[j] and v1[i]<->v1[j]).
       * @param   [in]   heap The heap.
       * @param   [in]   v1 The second array.
       * @param   [in]   v2 The third array.
       * @param   [in]   len The length.
       */
      template <typename T1, typename T2>
      static void AbsMinHeapRemove_RN(T1 *heap, T2 *v, int len)
      {
         /* parent, left, right */
         int p,l,r;
         len--;/* now len is the max index */
         /* swap the first element to last */
         IluClass<MatrixType, VectorType, DataType>::Swap<T2, T1>( v, heap, 0, -len);
         p = 0;
         l = 1;
         /* while I'm still in the heap */
         while(l < len)
         {
            r = 2*p+2;
            /* two childs, pick the smaller one */
            l = r >= len || PargemslrAbs(heap[-l])>PargemslrAbs(heap[-r]) ? l : r;
            if(PargemslrAbs(heap[-l])>PargemslrAbs(heap[-p]))
            {
               IluClass<MatrixType, VectorType, DataType>::Swap<T2, T1>( v, heap, -l, -p);
               p = l;
               l = 2*p+1;
            }
            else
            {
               break;
            }
         }
      }

      /**
       * @brief   Add value to a min heap. NNR stands for normal (heap from left to right), normal (v1 based on normal order), reverse (v2 based on index order).
       * @details Add value to a min heap. NNR stands for normal (heap from left to right), normal (v1 based on normal order), reverse (v2 based on index order). \n
       *          That is, the heap starts from heap[0], and goes to heap[1] ... \n
       *          When swaping elements in the heap, elements in v1 swap in the same way (swap heap[i]<->heap[j] and v1[i]<->v1[j]).
       *          When swaping elements in the heap, elements in v2 swap in based on the heap value (swap heap[i]<->heap[j] and v1[heal[i]]<->v1[heap[j]]).
       * @param   [in]   heap The heap.
       * @param   [in]   v1 The second array.
       * @param   [in]   v2 The third array.
       * @param   [in]   len The length.
       */
      template <typename T1, typename T2>
      static void MinHeapAdd_NNR(int *heap, T1 *v1, T2 *v2, int len)
      {
         /* parent, left, right */
         int p;
         len--;/* now len is the current index */
         while(len > 0)
         {
            /* get the parent index */
            p = (len-1)/2;
            if(heap[p] > heap[len])
            {
               /* this is smaller */
               IluClass<MatrixType, VectorType, DataType>::Swap<T2>( v2, heap[p], heap[len]);
               IluClass<MatrixType, VectorType, DataType>::Swap<int, T1>( heap, v1, p, len);
               len = p;
            }
            else
            {
               break;
            }
         }
      }
      
      /**
       * @brief   Remove value from a min heap. NNR stands for normal (heap from left to right), normal (v1 based on normal order), reverse (v2 based on index order).
       * @details Remove value from a min heap. NNR stands for normal (heap from left to right), normal (v1 based on normal order), reverse (v2 based on index order). \n
       *          That is, the heap starts from heap[0], and goes to heap[1] ... \n
       *          When swaping elements in the heap, elements in v1 swap in the same way (swap heap[i]<->heap[j] and v1[i]<->v1[j]).
       *          When swaping elements in the heap, elements in v2 swap in based on the heap value (swap heap[i]<->heap[j] and v1[heal[i]]<->v1[heap[j]]).
       * @param   [in]   heap The heap.
       * @param   [in]   v1 The second array.
       * @param   [in]   v2 The third array.
       * @param   [in]   len The length.
       */
      template <typename T1, typename T2>
      static void MinHeapRemovd_NNR(int *heap, T1 *v1, T2 *v2, int len)
      {
         /* parent, left, right */
         int p,l,r;
         len--;/* now len is the max index */
         /* swap the first element to last */
         IluClass<MatrixType, VectorType, DataType>::Swap<T2>(v2, heap[0], heap[len]);
         IluClass<MatrixType, VectorType, DataType>::Swap<int, T1>(heap, v1, 0, len);
         p = 0;
         l = 1;
         /* while I'm still in the heap */
         while(l < len)
         {
            r = 2*p+2;
            /* two childs, pick the smaller one */
            l = r >= len || heap[l]<heap[r] ? l : r;
            if(heap[l]<heap[p])
            {
               IluClass<MatrixType, VectorType, DataType>::Swap<T2>(v2,heap[p],heap[l]);
               IluClass<MatrixType, VectorType, DataType>::Swap<int, T1>( heap, v1, l, p);
               p = l;
               l = 2*p+1;
            }
            else
            {
               break;
            }
         }
      }
      
      /**                                                                                                                                                                               
       * @brief   Does a quick-sort split of a real array.
       * @details Does a quick-sort split of a real array.                                                                
       * @param [in/out]a the array to sort.
       * @param [in]    ind permuted indices.
       * @param [in]    n length of the array.
       * @param [in]    left the start location.
       * @param [in]    split from here.
       * @param [in]    right the end location.
       * @return     Return error message.
       */
      template <typename T>
      static void Qsplit(T *a, int *ind, int left, int bound, int right)
      {
         int i, last;
         if (left >= right)
         {
            return;
         }
         IluClass<MatrixType, VectorType, DataType>::Swap<int, T>( ind, a, left, (left+right)/2);
         last = left;
         for(i = left + 1 ; i <= right ; i ++)
         {
            if(PargemslrAbs(a[i]) > PargemslrAbs(a[left]))
            {
               IluClass<MatrixType, VectorType, DataType>::Swap<int, T>( ind, a, ++last, i);
            }
         }
         IluClass<MatrixType, VectorType, DataType>::Swap<int, T>( ind, a, left, last);
         IluClass<MatrixType, VectorType, DataType>::Qsplit<T>( a, ind, left, bound, last-1);
         if(bound > last)
         {
             IluClass<MatrixType, VectorType, DataType>::Qsplit<T>( a, ind, last+1, bound, right);
         }
      }
      
      /**
       * @brief   Setup Partial ILUT, form approximate Schur Complement.
       * @details Setup Partial ILUT, form approximate Schur Complement.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      int SetupPartialILUT(VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Solve with L only. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve with L only. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      int SolveL( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Solve with U only. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve with U only. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      int SolveU( VectorType &x, VectorType &rhs);
      
      /**
       * @brief      Set the data location that the preconditioner apply to.
       * @details    Set the data location that the preconditioner apply to.
       * @param [in] location The target solver location.
       * @return     Return error message.
       */
      virtual int SetSolveLocation( const int &location);
      
      /**
       * @brief      Move the preconditioner to another location. Only can be called after Setup.
       * @details    Move the preconditioner to another location. Only can be called after Setup.
       * @param [in] location The target solver location.
       * @return     Return error message.
       */
      virtual int MoveData( const int &location);
      
      /* --------- SETS and GETS ----------- */
      
      /**
       * @brief   Setup with parameter array.
       * @details Setup with parameter array.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetWithParameterArray(double *params)
      {
         if(this->_ready)
         {
            PARGEMSLR_ERROR("Change setup after preconditioner is built.");
            return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
         }
         
         this->_droptol = params[PARGEMSLR_IO_ILU_DROPTOL_B_LOCAL];
         this->_droptol_ef = params[PARGEMSLR_IO_ILU_DROPTOL_EF_LOCAL];
         this->_droptol_s = params[PARGEMSLR_IO_ILU_DROPTOL_S_LOCAL];
         this->_max_row_nnz = params[PARGEMSLR_IO_ILU_ROWNNZ_B_LOCAL];
         this->_max_row_nnz_s = params[PARGEMSLR_IO_ILU_ROWNNZ_S_LOCAL];
         this->_perm_option = params[PARGEMSLR_IO_ILU_PERM_OPTION_LOCAL];
         this->_omp_option = params[PARGEMSLR_IO_ILU_OMP_OPTION_LOCAL];
         this->_poly_order = params[PARGEMSLR_IO_POLY_ORDER];
         this->_print_option = params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL];
         this->_complex_shift = params[PARGEMSLR_IO_ADVANCED_USE_COMPLEX_SHIFT];
         this->_diag_shift_milu = params[PARGEMSLR_IO_ADVANCED_DIAG_SHIFT_MODIFIED];
         
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the level of fill for ILU(K).
       * @details Set the level of fill for ILU(K).
       * @param   [in]   lfil The new level of fill for ILU(K).
       * @return     Return error message.
       */
      int         SetLevelOfFill( int lfil)
      {
         if(this->_ready)
         {
            PARGEMSLR_WARNING("Change setting after Setup is not going to change the preconditioner.");
         }
         this->_fill_level = lfil;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the complex shift parameter (complex version only).
       * @details Set the complex shift parameter (complex version only).
       * @param   [in]   complex_shift The new complex shift value.
       * @return     Return error message.
       */
      template <typename T>
      int         SetIluComplexShift( T complex_shift)
      {
         if(this->_ready)
         {
            PARGEMSLR_WARNING("Change setting after Setup is not going to change the preconditioner.");
         }
         this->_complex_shift = complex_shift;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the drop tols for ILUT.
       * @details Set the drop tols for ILUT.
       * @param   [in]   droptol The new drop tol for ILUT.
       * @return     Return error message.
       */
      template <typename T>
      int         SetDropTolerance( T droptol)
      {
         if(this->_ready)
         {
            PARGEMSLR_WARNING("Change setting after Setup is not going to change the preconditioner.");
         }
         this->_droptol = droptol;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the drop tols in E, F part for Partial ILUT.
       * @details Set the drop tols in E, F part for Partial ILUT.
       * @param   [in]   droptol_ef The new drop tol in E, F part for Partial ILUT.
       * @return     Return error message.
       */
      template <typename T>
      int         SetDropToleranceEF( T droptol_ef)
      {
         if(this->_ready)
         {
            PARGEMSLR_WARNING("Change setting after Setup is not going to change the preconditioner.");
         }
         this->_droptol_ef = droptol_ef;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the drop tols in S part for Partial ILUT.
       * @details Set the drop tols in S part for Partial ILUT.
       * @param   [in]   droptol_ef The new drop tol in S part for Partial ILUT.
       * @return     Return error message.
       */
      template <typename T>
      int         SetDropToleranceS( T droptol_s)
      {
         if(this->_ready)
         {
            PARGEMSLR_WARNING("Change setting after Setup is not going to change the preconditioner.");
         }
         this->_droptol_s = droptol_s;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the max fill allowed for each row/col of L and U.
       * @details Set the max fill allowed for each row/col of L and U.
       * @param   [in]   max_row_nnz The new max fill allowed for each row/col of L and U.
       * @return     Return error message.
       */
      int         SetMaxNnzPerRow( int max_row_nnz);
      
      /**
       * @brief   Set the max fill allowed for each row/col of S in partial ILU.
       * @details Set the max fill allowed for each row/col of S in partial ILU.
       * @param   [in]   max_row_nnz_s The new max fill allowed for each row/col of S in partial ILU.
       * @return     Return error message.
       */
      int         SetMaxNnzPerRowSPart( int max_row_nnz_s);
      
      /**
       * @brief   Set the size of the B block.
       * @details Set the size of the B block.
       * @param   [in]   nB The number of nodes in the B block.
       * @return     Return error message.
       */
      int         SetNB(int nB);
      
      /**
       * @brief   Set the ILU option. See IluOptionEnum.
       * @details Set the ILU option. See IluOptionEnum.
       * @param   [in]   option The new ILUT option.
       * @return     Return error message.
       */
      int         SetOption( int option);
      
      /**
       * @brief   Set the OpenMP option. See IluOpenMPOptionEnum.
       * @details Set the OpenMP option. See IluOpenMPOptionEnum.
       * @param   [in]   omp_option The new OpenMP option.
       * @return     Return error message.
       */
      int         SetOpenMPOption( int omp_option);
      
      /**
       * @brief   Set the polynomia order in some OpenMP option. See IluOpenMPOptionEnum.
       * @details Set the polynomia order in some OpenMP option. See IluOpenMPOptionEnum.
       * @param   [in]   order The new order.
       * @return     Return error message.
       */
      int         SetPolyOrder( int order);
      
      /**
       * @brief   Set the ILU permutation option. See IluReorderingOptionEnum.
       * @details Set the ILU permutation option. See IluReorderingOptionEnum.
       * @param   [in]   perm_option The new ILU permutation option.
       * @return     Return error message.
       */
      int         SetPermutationOption( int perm_option);
      
      /**
       * @brief   Enable/Disable modified ILU.
       * @details Enable/Disable modified ILU.
       * @param   [in]   modified The new ILUT option.
       * @return     Return error message.
       */
      int         SetModified( bool modified);
      
      /**
       * @brief   Set the shift for the modified ILU.
       * @details Set the shift for the modified ILU.
       * @param   [in]   milu_shift The shift for the modified ILU.
       * @return     Return error message.
       */
      template <typename T>
      int         SetModifiedShift( T milu_shift)
      {
         if(this->_ready)
         {
            PARGEMSLR_WARNING("Change setting after Setup is not going to change the preconditioner.");
         }
         this->_diag_shift_milu = milu_shift;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Get the size of the problem.
       * @details Get the size of the problem.
       * @return     Return the problem size.
       */
      int         GetSize();
      
      /**
       * @brief   Get the total number of nonzeros the ILU.
       * @details Get the total number of nonzeros the ILU.
       * @return     Return the total number of nonzeros the ILU.
       */
      virtual long int  GetNumNonzeros();
      
      /**
       * @brief   Get the L matrix of the ILU factorizaiton without diagonal.
       * @details Get the L matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>& GetL();
      
      /**
       * @brief   Get the inverse of diagonal of the U matrix.
       * @details Get the inverse of diagonal of the U matrix.
       */
      SequentialVectorClass<DataType>& GetD();
      
      /**
       * @brief   Get the U matrix of the ILU factorizaiton without diagonal.
       * @details Get the U matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>& GetU();
      
      /**
       * @brief   Get the U matrix of the ILU factorizaiton without diagonal.
       * @details Get the U matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>& GetE();
      
      /**
       * @brief   Get the U matrix of the ILU factorizaiton without diagonal.
       * @details Get the U matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>& GetF();
      
      /**
       * @brief   Get the U matrix of the ILU factorizaiton without diagonal.
       * @details Get the U matrix of the ILU factorizaiton without diagonal.
       */
      CsrMatrixClass<DataType>& GetS();
      
      /**
       * @brief   Get the U matrix of the ILU factorizaiton without diagonal.
       * @details Get the U matrix of the ILU factorizaiton without diagonal.
       */
      int GetNB();
      
      /**
       * @brief   Get the row permutation vector.
       * @details Get the row permutation vector.
       */
      IntVectorClass<int>& GetRowPermutationVector();
      
      /**
       * @brief   Get the column permutation vector.
       * @details Get the column permutation vector.
       */
      IntVectorClass<int>& GetColPermutationVector();
      
	};
   
   typedef IluClass<CsrMatrixClass<float>, SequentialVectorClass<float>, float>        precond_ilu_csr_seq_float;
   typedef IluClass<CsrMatrixClass<double>, SequentialVectorClass<double>, double>     precond_ilu_csr_seq_double;
   typedef IluClass<CsrMatrixClass<complexs>, SequentialVectorClass<complexs>, complexs>  precond_ilu_csr_seq_complexs;
   typedef IluClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd> precond_ilu_csr_seq_complexd;
   
}

#endif
