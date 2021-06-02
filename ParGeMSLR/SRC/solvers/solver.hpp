#ifndef PARGEMSLR_SOLVER_H
#define PARGEMSLR_SOLVER_H

/**
 * @file solver.hpp
 * @brief Virtual class of iterative solvers.
 */

#include "../utils/utils.hpp"
#include "../vectors/vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "../matrices/matrix.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../matrices/dense_matrix.hpp"
#include "../matrices/parallel_csr_matrix.hpp"

namespace pargemslr
{
   
   /**
    * @brief   The solver type.
    * @details The solver type.
    */
   enum SolverTypeEnum
   {
      kSolverUndefined = -1,
      kSolverFgmres,
      kSolverIlu,
      kSolverBJ,
      kSolverFloat,
      kSolverGemslr,
      kSolverParGemslr
   };
   
   /**
    * @brief   The base solver class.
    * @details The base solver class. Matrix type is the type of the matrix, PreconditionerType is the type of the preconditioner,
    *          VectorType is the type of the vector, DataType is the data type.\n
    *          To develope a new solver type, user typically needs to implement the following:\n
    *          _location, MoveData() or SetSolveLocation() function: to control the solver location (host/device).\n
    *          Setup() function: setup the solver.\n
    *          Solve() function: apply the solve.
    */
	template <class MatrixType, class VectorType, typename DataType>
   class SolverClass
	{
   protected:
      
      /**
       * @brief   The precision of the solver
       * @details The precision of the solver.
       */
      PrecisionEnum                                      _solver_precision;
      
      /**
       * @brief   The type of the solver.
       * @details The type of the solver.
       */
      SolverTypeEnum                                     _solver_type;
      
      /**
       * @brief   Is the preconditioner a mixed precision preconditioner?
       * @details Is the preconditioner a mixed precision preconditioner?
       */
      bool                                               _is_mixed;
      
      /**
       * @brief   The matrix.
       * @details The matrix.
       */
      MatrixType                                         *_matrix;
      
      /**
       * @brief   If the matrix is owned by this solver, default is false.
       * @details If the matrix is owned by this solver, default is false.
       */
      bool                                               _own_matrix;
      
      /**
       * @brief   The preconditioner.
       * @details The preconditioner.
       */
      SolverClass<MatrixType, VectorType, DataType>      *_preconditioner;
      
      /**
       * @brief   If the preconditioner is owned by this solver, default is false.
       * @details If the preconditioner is owned by this solver, default is false.
       */
      bool                                               _own_preconditioner;
      
      /**
       * @brief   Pointer to the solution, note that this vector is not going to be freed.
       * @details Pointer to the solution, note that this vector is not going to be freed.
       */
      VectorType                                         *_solution;
      
      /**
       * @brief   Pointer to the right-hand-size, note that this vector is not going to be freed.
       * @details Pointer to the right-hand-size, note that this vector is not going to be freed.
       */
      VectorType                                         *_right_hand_side;
      
      /**
       * @brief   If the solver is ready.
       * @details If the solver is ready.
       */
      bool                                               _ready;
      
      /**
       * @brief   The print option.
       * @details The print option.
       */
      int                                                _print_option;
      
      /**
       * @brief   Return error is the preconditioner is ready.
       * @details Return error is the preconditioner is ready.
       * @return     Return true if the solver is ready.
       */
      int CheckReadySetups(const char *str) const
      {
         if(this->_ready)
         {
            char str1[1024];
            snprintf( str1, 1024, "Change setup %s after preconditioner is built is invalid.", str );
            PARGEMSLR_ERROR(str1);
            return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
         }
         return PARGEMSLR_SUCCESS;
      }
      
	public:
      
      /**
       * @brief   The constructor of solver class.
       * @details The constructor of solver class.
       */
      SolverClass()
      {
         this->_solver_precision = kUnknownPrecision;
         this->_solver_type = kSolverUndefined;
         /* if vector type not equal to data type, this is a mixed precision preconditioner */
         this->_is_mixed = PargemslrIsDoublePrecision<VectorType>::value != PargemslrIsDoublePrecision<DataType>::value;
         this->_matrix = NULL;
         this->_own_matrix = false;
         this->_preconditioner = NULL;
         this->_own_preconditioner = false;
         this->_solution = NULL;
         this->_right_hand_side = NULL;
         this->_ready = false;
         this->_print_option = 0;
      }
      
      /**
       * @brief   The copy constructor of solver class.
       * @details The copy constructor of solver class.
       * @param   [in]   solver The solver.
       */
      SolverClass(const SolverClass<MatrixType, VectorType, DataType> &solver)
      {
         this->_solver_precision = solver._solver_precision;
         this->_solver_type = solver._solver_type;
         this->_is_mixed = solver._is_mixed;
         this->_matrix = solver._matrix;
         this->_own_matrix = false;
         this->_preconditioner = solver._preconditioner;
         this->_own_preconditioner = false;
         this->_right_hand_side = solver._right_hand_side;
         this->_ready = solver._ready;
         this->_print_option = solver._print_option;
      }
      
      /**
       * @brief   The move constructor of solver class.
       * @details The move constructor of solver class.
       * @param   [in]   solver The solver.
       */
      SolverClass(SolverClass<MatrixType, VectorType, DataType> &&solver)
      {
         this->_solver_precision = solver._solver_precision;
         solver._solver_precision = kUnknownPrecision;
         this->_solver_type = solver._solver_type;
         solver._solver_type = kSolverUndefined;
         this->_is_mixed = solver._is_mixed;
         this->_matrix = solver._matrix;
         solver._matrix = NULL;
         this->_own_matrix = solver._own_matrix;
         solver._own_matrix = false;
         this->_preconditioner = solver._preconditioner;
         solver._preconditioner = NULL;
         this->_own_preconditioner = solver._own_preconditioner;
         solver._own_preconditioner = false;
         this->_solution = solver._solution;
         solver._solution = NULL;
         this->_right_hand_side = solver._right_hand_side;
         solver._right_hand_side = NULL;
         this->_ready = solver._ready;
         solver._ready = false;
         this->_print_option = solver._print_option;
         solver._print_option = 0;
      }
      
      /**
       * @brief   The = operator of solver class.
       * @details The = operator of solver class.
       * @param   [in]   solver The solver.
       * @return     Return the solver
       */
      SolverClass<MatrixType, VectorType, DataType>& operator= (const SolverClass<MatrixType, VectorType, DataType> &solver)
      {
         this->Clear();
         this->_solver_precision = solver._solver_precision;
         this->_solver_type = solver._solver_type;
         this->_is_mixed = solver._is_mixed;
         this->_own_matrix = false;
         this->_preconditioner = solver._preconditioner;
         this->_own_preconditioner = false;
         this->_solution = solver._solution;
         this->_right_hand_side = solver._right_hand_side;
         this->_ready = solver._ready;
         this->_print_option = solver._print_option;
         return *this;
      }
      
      /**
       * @brief   The = operator of solver class.
       * @details The = operator of solver class.
       * @param   [in]   solver The solver.
       * @return     Return the solver.
       */
      SolverClass<MatrixType, VectorType, DataType>& operator= (SolverClass<MatrixType, VectorType, DataType> &&solver)
      {
         this->Clear();
         this->_solver_precision = solver._solver_precision;
         solver._solver_precision = kUnknownPrecision;
         this->_solver_type = solver._solver_type;
         solver._solver_type = kSolverUndefined;
         this->_is_mixed = solver._is_mixed;
         this->_matrix = solver._matrix;
         solver._matrix = NULL;
         this->_own_matrix = solver._own_matrix;
         solver._own_matrix = false;
         this->_preconditioner = solver._preconditioner;
         solver._preconditioner = NULL;
         this->_own_preconditioner = solver._own_preconditioner;
         solver._own_preconditioner = false;
         this->_solution = solver._solution;
         solver._solution = NULL;
         this->_right_hand_side = solver._right_hand_side;
         solver._right_hand_side = NULL;
         this->_ready = solver._ready;
         solver._ready = false;
         this->_print_option = solver._print_option;
         solver._print_option = 0;
         return *this;
      }
      
      /**
       * @brief   Free the current solver.
       * @details Free the current solver.
       * @return     Return error message.
       */
      virtual int Clear()
      {
         
         this->_solver_precision = kUnknownPrecision;
         this->_solver_type = kSolverUndefined;
         
         if(this->_matrix && this->_own_matrix)
         {
            this->_matrix->Clear();
            delete this->_matrix;
         }
         this->_matrix = NULL;
         
         if(this->_preconditioner && this->_own_preconditioner)
         {
            this->_preconditioner->Clear();
            delete this->_preconditioner;
         }
         this->_preconditioner = NULL;
         this->_solution = NULL;
         this->_right_hand_side = NULL;
         
         this->_own_matrix = false;
         this->_own_preconditioner = false;
         this->_ready = false;
         this->_print_option = 0;
         
         return PARGEMSLR_SUCCESS;
      }
     
      /**
       * @brief   The destructor of solver class.
       * @details The destructor of solver class.
       */
      virtual ~SolverClass()
      {
         /* call base clear function */
         SolverClass<MatrixType, VectorType, DataType>::Clear();
      }
      
      /**
       * @brief   Setup the solver phase, include building the preconditioner. Call this function before Solve, after SetPreconditioner.
       * @details Setup the solver phase, include building the preconditioner. Call this function before Solve, after SetPreconditioner.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( VectorType &x, VectorType &rhs) = 0;
      
      /**
       * @brief   Solve phase. Call this function after Setup.
       * @details Solve phase. Call this function after Setup.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( VectorType &x, VectorType &rhs) = 0;
      
      /**
       * @brief   Solve phase with multiple right-hand-sides. Call this function after Setup.
       * @details Solve phase with multiple right-hand-sides. Call this function after Setup.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( DenseMatrixClass<DataType> &x, DenseMatrixClass<DataType> &rhs)
      {
         PARGEMSLR_ERROR("Error: Solve with multiple right-hand-side unimplemented for this class");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      /**
       * @brief   Get the total number of nonzeros.
       * @details Get the total number of nonzeros.
       * @return     Return the total number of nonzeros.
       */
      virtual long int  GetNumNonzeros()
      {
         PARGEMSLR_ERROR("Error: GetNumNonzeros unimplemented for this class");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      /**
       * @brief   Get pointer to the matrix.
       * @details Get pointer to the matrix.
       * @return     Return pointer to the matrix.
       */
      MatrixType* GetMatrix()
      {
         return _matrix;
      }
      
      /**
       * @brief   Get pointer to the preconditioner.
       * @details Get pointer to the preconditioner.
       * @return     Return pointer to the preconditioner.
       */
      SolverClass<MatrixType, VectorType, DataType>* GetPreconditioner()
      {
         return _preconditioner;
      }
      
      /**
       * @brief   Setup with parameter array.
       * @details Setup with parameter array.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetWithParameterArray(double *params)
      {
         PARGEMSLR_ERROR("Error: SetWithParameterArray unimplemented for this class");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      /**
       * @brief      Set the data location that the solver apply to. Will also be applied to the preconditioner.
       * @details    Set the data location that the solver apply to. Will also be applied to the preconditioner.
       * @param [in] location The target solver location.
       * @return     Return error message.
       */
      virtual int SetSolveLocation( const int &location) = 0;
      
      /**
       * @brief   Set the target matrix for the current solver.
       * @details Set the target matrix for the current solver.
       * @param   [in]   matrix The target matrix.
       * @return     Return error message.
       */
      int SetMatrix( MatrixType &matrix)
      {
         return this->SetMatrixP(&matrix);
      }
      
      /**
       * @brief   Set the target matrix for the current solver.
       * @details Set the target matrix for the current solver.
       * @param   [in]   matrix The pointer to the target matrix.
       * @return     Return error message.
       */
      int SetMatrixP( MatrixType *matrix)
      {
         if(matrix == this->_matrix)
         {
            /* set to the same matrix, do nothing */
            return PARGEMSLR_SUCCESS;
         }
         
         if(this->_matrix && this->_own_matrix)
         {
            this->_matrix->Clear();
            delete this->_matrix;
         }
         
         /* set a new matrix */
         this->_matrix = matrix;
         
         if(this->_preconditioner && this->_own_preconditioner)
         {
            this->_preconditioner->Clear();
            delete this->_preconditioner;
         }
         
         /* also destroy the preconditioner */
         this->_preconditioner = NULL;
         this->_solution = NULL;
         this->_right_hand_side = NULL;
         
         this->_ready = false;
         
         this->_own_matrix = false;
         this->_own_preconditioner = false;
         
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the preconditioner for the current solver.
       * @details Set the preconditioner for the current solver.
       * @param   [in]   precond The preconditioner.
       * @return     Return error message.
       */
      int SetPreconditioner( SolverClass<MatrixType, VectorType, DataType> &precond)
      {
         return this->SetPreconditionerP(&precond);
      }
      
      /**
       * @brief   Set the preconditioner for the current solver.
       * @details Set the preconditioner for the current solver.
       * @param   [in]   precond The pointer to the preconditioner.
       * @return     Return error message.
       */
      int SetPreconditionerP( SolverClass<MatrixType, VectorType, DataType>* precond)
      {
         if(this->_preconditioner && this->_own_preconditioner)
         {
            this->_preconditioner->Clear();
            delete this->_preconditioner;
         }
         this->_preconditioner = precond;
         this->_own_preconditioner = false;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Check if the solver is ready to be used.
       * @details Check if the solver is ready to be used.
       * @return     Return true if the solver is ready.
       */
      bool IsReady() const
      {
         return this->_ready;
      }
      
      /**
       * @brief   Set if the matrix is owned by this solver, if so, matrix is freed when free the solver.
       * @details Set if the matrix is owned by this solver, if so, matrix is freed when free the solver.
       * @param   [in]     own_matrix The boolean value.
       * @return     Return error message.
       */
      int SetOwnMatrix(bool own_matrix)
      {
         this->_own_matrix = own_matrix;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set if the preconditioner is owned by this solver, if so, preconditioner is freed when free the solver.
       * @details Set if the preconditioner is owned by this solver, if so, preconditioner is freed when free the solver.
       * @param   [in]     own_preconditioner The boolean value.
       * @return     Return error message.
       */
      int SetOwnPreconditioner(bool own_preconditioner)
      {
         this->_own_preconditioner = own_preconditioner;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the print option.
       * @details Set the print option. 0: minimal output. 1: detailed output.
       * @param   [in]     print_option The print option.
       * @return     Return error message.
       */
      int SetPrintOption(int print_option)
      {
         this->_print_option = print_option;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Get the precision.
       * @details Get the precision.
       * @return     Return the precision.
       */
      PrecisionEnum GerPrecision() const
      {
         return this->_solver_precision;
      }
      
      /**
       * @brief   Get the solver type.
       * @details Get the solver type.
       * @return     Return the solver type.
       */
      SolverTypeEnum GetSolverType() const
      {
         return this->_solver_type;
      }
      
	};
   
   typedef SolverClass<CsrMatrixClass<float>, SequentialVectorClass<float>, float>           solver_csr_seq_float;
   typedef SolverClass<CsrMatrixClass<double>, SequentialVectorClass<double>, float>         solver_csr_seq_mix;
   typedef SolverClass<CsrMatrixClass<double>, SequentialVectorClass<double>, double>        solver_csr_seq_double;
   typedef SolverClass<CsrMatrixClass<complexs>, SequentialVectorClass<complexs>, complexs>  solver_csr_seq_complexs;
   typedef SolverClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexs>  solver_csr_seq_complexmix;
   typedef SolverClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd>  solver_csr_seq_complexd;
   
   typedef SolverClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>           solver_csr_par_float;
   typedef SolverClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>        solver_csr_par_double;
   typedef SolverClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>        solver_csr_par_double;
   typedef SolverClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>  solver_csr_par_complexs;
   typedef SolverClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexs>  solver_csr_par_complexmix;
   typedef SolverClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>  solver_csr_par_complexd;
}

#endif
