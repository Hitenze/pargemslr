#ifndef PARGEMSLR_BLOCK_JACOBI_H
#define PARGEMSLR_BLOCK_JACOBI_H

/**
 * @file block_jacobi.hpp
 * @brief block jacobi preconditioner.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../matrices/parallel_csr_matrix.hpp"
#include "../solvers/solver.hpp"

#ifdef PARGEMSLR_CUDA
#include "cusparse.h"
#endif

namespace pargemslr
{
   /**
    * @brief   Block Jacobi preconditioner, only for parallel csr matrix.
    * @details Block Jacobi preconditioner, only for parallel csr matrix.
    */
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   class BlockJacobiClass: public SolverClass<ParallelMatrixType, ParallelVectorType, DataType>
   {
   private:
      
      /**
       * @brief   The local preconditioner.
       * @details The local preconditioner.
       */
      SolverClass<LocalMatrixType, LocalVectorType, DataType>                 *_local_precond;
      
      /**
       * @brief   Should we free the preconditioner when destroy.
       * @details Should we free the preconditioner when destroy.
       */
      int                                                                     _own_local_preconditioner;
      
      /**
       * @brief   The location this preconditioner applied to.
       * @details The location this preconditioner applied to.
       */
      int                                                                     _location;
      
	public:
      /**
       * @brief   The constructor of BlockJacobiClass.
       * @details The constructor of BlockJacobiClass.
       */
      BlockJacobiClass();
     
      /**
       * @brief   Free the current precondioner.
       * @details Free the current precondioner.
       * @return     Return error message.
       */
      virtual int Clear();
     
      /**
       * @brief   The destructor of BlockJacobiClass.
       * @details The destructor of BlockJacobiClass.
       */
      virtual ~BlockJacobiClass();
      
      /**
       * @brief   The copy constructor of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner.
       * @details The copy constructor of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner. \n
       *          The onwer is not this preconditioner.
       * @param   [in]   solve The target solver.
       */
      BlockJacobiClass(const BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &solver);
      
      /**
       * @brief   The move constructor of BlockJacobiClass.
       * @details The move constructor of BlockJacobiClass.
       * @param   [in]   solve The target solver.
       */
      BlockJacobiClass(BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &&solver);
      
      /**
       * @brief   The = operator of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner.
       * @details The = operator of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner. \n
       *          The onwer is not this preconditioner.
       * @param   [in]   solve The target solver.
       * @return     Return the solver.
       */
      BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>& operator= (const BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &solver);
      
      /**
       * @brief   The = operator of BlockJacobiClass.
       * @details The = operator of BlockJacobiClass.
       * @param   [in]   solve The target solver.
       * @return     Return the solver.
       */
      BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>& operator= (BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &&solver);
      
      /**
       * @brief   Setup the precondioner phase. Will be called by the solver if not called directly.
       * @details Setup the precondioner phase. Will be called by the solver if not called directly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( ParallelVectorType &x, ParallelVectorType &rhs);
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( ParallelVectorType &x, ParallelVectorType &rhs);
      
      /**
       * @brief   Get the total number of nonzeros.
       * @details Get the total number of nonzeros.
       * @return     Return the total number of nonzeros.
       */
      virtual long int  GetNumNonzeros();
      
      /**
       * @brief   Set the local preconditioenr.
       * @details Set the local preconditioenr.
       * @return     Return error message.
       */
      int         SetLocalPreconditioner(SolverClass<LocalMatrixType, LocalVectorType, DataType> &local_precond);
      
      /**
       * @brief   Set the local preconditioenr.
       * @details Set the local preconditioenr.
       * @return     Return error message.
       */
      int         SetLocalPreconditionerP(SolverClass<LocalMatrixType, LocalVectorType, DataType>* local_precond);
      
      /**
       * @brief   Get the local preconditioenr.
       * @details Get the local preconditioenr.
       * @return     Return the local preconditioner.
       */
      SolverClass<LocalMatrixType, LocalVectorType, DataType>* GetLocalPreconditionerP();
      
      /**
       * @brief   Set the own preconditioenr. If set to true, the preconditioner will be freed when destrooy
       * @details Set the own preconditioenr.
       * @return     Return the total number of nonzeros the ILU.
       */
      int         SetOwnLocalPreconditioner(bool own_local_preconditioner);
      
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
      
	};
   
   typedef BlockJacobiClass<matrix_csr_par_float, vector_par_float, matrix_csr_float, vector_seq_float, float>                precond_bj_csr_par_float;
   typedef BlockJacobiClass<matrix_csr_par_double, vector_par_double, matrix_csr_double, vector_seq_double, double>           precond_bj_csr_par_double;
   typedef BlockJacobiClass<matrix_csr_par_complexs, vector_par_complexs, matrix_csr_complexs, vector_seq_complexs, complexs> precond_bj_csr_par_complexs;
   typedef BlockJacobiClass<matrix_csr_par_complexd, vector_par_complexd, matrix_csr_complexd, vector_seq_complexd, complexd> precond_bj_csr_par_complexd;
   
}

#endif
