#ifndef PARGEMSLR_PARALLEL_MIX_PRECOND_H
#define PARGEMSLR_PARALLEL_MIX_PRECOND_H

/**
 * @file float_precond.hpp
 * @brief single precision real preconditioner.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../vectors/vectorops.hpp"
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
    * @brief   Single precision real preconditioner, only for parallel csr matrix.
    * @details Single precision real preconditioner, only for parallel csr matrix.
    */
   class ParallelMixPrecondClass: public SolverClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>
   {
   private:
      
      /**
       * @brief   The single precision preconditioner.
       * @details The single precision preconditioner.
       */
      SolverClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float> *_single_precond;
      
      /**
       * @brief   Should we free the preconditioner when destroy.
       * @details Should we free the preconditioner when destroy.
       */
      int                                                                           _own_single_preconditioner;
      
      /**
       * @brief   The location this preconditioner applied to.
       * @details The location this preconditioner applied to.
       */
      int                                                                           _location;
      
      /**
       * @brief   The single solution.
       * @details The single solution.
       */
      ParallelVectorClass<float>                                                    _x_float;
      
      /**
       * @brief   The single right-hand-side.
       * @details The single right-hand-side.
       */
      ParallelVectorClass<float>                                                    _rhs_float;
      
      /**
       * @brief   The single matrix.
       * @details The single matrix.
       */
      ParallelCsrMatrixClass<float>                                                 _matrix_float;
      
	public:
      /**
       * @brief   The constructor of BlockJacobiClass.
       * @details The constructor of BlockJacobiClass.
       */
      ParallelMixPrecondClass();
     
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
      virtual ~ParallelMixPrecondClass();
      
      /**
       * @brief   The copy constructor of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner.
       * @details The copy constructor of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner. \n
       *          The onwer is not this preconditioner.
       * @param   [in]   solve The target solver.
       */
      ParallelMixPrecondClass(const ParallelMixPrecondClass &solver);
      
      /**
       * @brief   The move constructor of BlockJacobiClass.
       * @details The move constructor of BlockJacobiClass.
       * @param   [in]   solve The target solver.
       */
      ParallelMixPrecondClass(ParallelMixPrecondClass &&solver);
      
      /**
       * @brief   The = operator of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner.
       * @details The = operator of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner. \n
       *          The onwer is not this preconditioner.
       * @param   [in]   solve The target solver.
       * @return     Return the solver.
       */
      ParallelMixPrecondClass& operator= (const ParallelMixPrecondClass &solver);
      
      /**
       * @brief   The = operator of BlockJacobiClass.
       * @details The = operator of BlockJacobiClass.
       * @param   [in]   solve The target solver.
       * @return     Return the solver.
       */
      ParallelMixPrecondClass& operator= (ParallelMixPrecondClass &&solver);
      
      /**
       * @brief   Setup the precondioner phase. Will be called by the solver if not called directly.
       * @details Setup the precondioner phase. Will be called by the solver if not called directly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
      
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
      int         SetSinglePreconditioner(SolverClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float> &single_precond);
      
      /**
       * @brief   Set the local preconditioenr.
       * @details Set the local preconditioenr.
       * @return     Return error message.
       */
      int         SetSinglePreconditionerP(SolverClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>* single_precond);
      
      /**
       * @brief   Get the local preconditioenr.
       * @details Get the local preconditioenr.
       * @return     Return the local preconditioner.
       */
      SolverClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>* GetSinglePreconditionerP();
      
      /**
       * @brief   Set the own preconditioenr. If set to true, the preconditioner will be freed when destrooy
       * @details Set the own preconditioenr.
       * @return     Return the total number of nonzeros the ILU.
       */
      int         SetOwnSinglePreconditioner(bool own_single_preconditioner);
      
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
   
   /**
    * @brief   Single precision complex preconditioner, only for parallel csr matrix.
    * @details Single precision complex preconditioner, only for parallel csr matrix.
    */
   class CParallelMixPrecondClass: public SolverClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>
   {
   private:
      
      /**
       * @brief   The single precision preconditioner.
       * @details The single precision preconditioner.
       */
      SolverClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>    *_single_precond;
      
      /**
       * @brief   Should we free the preconditioner when destroy.
       * @details Should we free the preconditioner when destroy.
       */
      int                                                                                       _own_single_preconditioner;
      
      /**
       * @brief   The location this preconditioner applied to.
       * @details The location this preconditioner applied to.
       */
      int                                                                                       _location;
      
      /**
       * @brief   The single solution.
       * @details The single solution.
       */
      ParallelVectorClass<complexs>                                                             _x_float;
      
      /**
       * @brief   The single right-hand-side.
       * @details The single right-hand-side.
       */
      ParallelVectorClass<complexs>                                                             _rhs_float;
      
      /**
       * @brief   The single matrix.
       * @details The single matrix.
       */
      ParallelCsrMatrixClass<complexs>                                                          _matrix_float;
      
	public:
      /**
       * @brief   The constructor of BlockJacobiClass.
       * @details The constructor of BlockJacobiClass.
       */
      CParallelMixPrecondClass();
     
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
      virtual ~CParallelMixPrecondClass();
      
      /**
       * @brief   The copy constructor of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner.
       * @details The copy constructor of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner. \n
       *          The onwer is not this preconditioner.
       * @param   [in]   solve The target solver.
       */
      CParallelMixPrecondClass(const CParallelMixPrecondClass &solver);
      
      /**
       * @brief   The move constructor of BlockJacobiClass.
       * @details The move constructor of BlockJacobiClass.
       * @param   [in]   solve The target solver.
       */
      CParallelMixPrecondClass(CParallelMixPrecondClass &&solver);
      
      /**
       * @brief   The = operator of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner.
       * @details The = operator of BlockJacobiClass. Note that this is not the true copy. We only copy the pointer to the local preconditioner. \n
       *          The onwer is not this preconditioner.
       * @param   [in]   solve The target solver.
       * @return     Return the solver.
       */
      CParallelMixPrecondClass& operator= (const CParallelMixPrecondClass &solver);
      
      /**
       * @brief   The = operator of BlockJacobiClass.
       * @details The = operator of BlockJacobiClass.
       * @param   [in]   solve The target solver.
       * @return     Return the solver.
       */
      CParallelMixPrecondClass& operator= (CParallelMixPrecondClass &&solver);
      
      /**
       * @brief   Setup the precondioner phase. Will be called by the solver if not called directly.
       * @details Setup the precondioner phase. Will be called by the solver if not called directly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
      
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
      int         SetSinglePreconditioner(SolverClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs> &single_precond);
      
      /**
       * @brief   Set the local preconditioenr.
       * @details Set the local preconditioenr.
       * @return     Return error message.
       */
      int         SetSinglePreconditionerP(SolverClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>* single_precond);
      
      /**
       * @brief   Get the local preconditioenr.
       * @details Get the local preconditioenr.
       * @return     Return the local preconditioner.
       */
      SolverClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>* GetSinglePreconditionerP();
      
      /**
       * @brief   Set the own preconditioenr. If set to true, the preconditioner will be freed when destrooy
       * @details Set the own preconditioenr.
       * @return     Return the total number of nonzeros the ILU.
       */
      int         SetOwnSinglePreconditioner(bool own_single_preconditioner);
      
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
   
}

#endif
