#ifndef PARGEMSLR_POLY_H
#define PARGEMSLR_POLY_H

/**
 * @file poly.hpp
 * @brief The polynomial preconditioners.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../matrices/parallel_csr_matrix.hpp"
#include "../solvers/solver.hpp"

namespace pargemslr
{
   
   /**
    * @brief   The simple preconditioner uses only matvec. Note that inv(I-A) \approx I+A+A^2+...
    * @details The simple preconditioner uses only matvec. Note that inv(I-A) \approx I+A+A^2+... \n
    *          Thus, inv(A) = inv(I-(I-A)) = inv(I-B) \approx I+B+B^2+..., which is (I+(I+B)B)... \n
    *          Since B = I-A, this is (I+(I+(I-A))(I-A))x
    *          Matrix type is the type of the matrix, 
    *          VectorType is the type of the vector. DataType is the data type.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class PolyClass: public SolverClass<MatrixType, VectorType, DataType>
   {
   private:
      
      /** 
       * @brief   The temp vector.
       * @details The temp vector.
       */
      VectorType                          _x_temp;
      
      /** 
       * @brief   The temp vector.
       * @details The temp vector.
       */
      VectorType                          _y_temp;
      
      /**
       * @brief   The size of the problem.
       * @details The size of the problem.
       */
      int                                 _n;
      
      /**
       * @brief   The order of the poly.
       * @details The order of the poly.
       */
      int                                 _order;
      
      /**
       * @brief   The location this preconditioner applied to.
       * @details The location this preconditioner applied to.
       */
      int                                 _location;
      
      /**
       * @brief   The scale.
       * @details The scale.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _scale;
      
	public:
      
      /**
       * @brief   The constructor of PolyClass.
       * @details The constructor of PolyClass.
       */
      PolyClass();
      
      /**
       * @brief   The copy constructor of PolyClass.
       * @details The copy constructor of PolyClass.
       * @param   [in]   solve The target solver.
       */
      PolyClass(const PolyClass<MatrixType, VectorType, DataType> &solver);
      
      /**
       * @brief   The move constructor of PolyClass.
       * @details The move constructor of PolyClass.
       * @param   [in]   solve The target solver.
       */
      PolyClass(PolyClass<MatrixType, VectorType, DataType> &&solver);
      
      /**
       * @brief   The = operator of PolyClass.
       * @details The = operator of PolyClass.
       * @param   [in]   solve The target solver.
       * @return     Return the solver.
       */
      PolyClass<MatrixType, VectorType, DataType>& operator= (const PolyClass<MatrixType, VectorType, DataType> &solver);
      
      /**
       * @brief   The = operator of PolyClass.
       * @details The = operator of PolyClass.
       * @param   [in]   solve The target solver.
       * @return     Return the solver.
       */
      PolyClass<MatrixType, VectorType, DataType>& operator= (PolyClass<MatrixType, VectorType, DataType> &&solver);
      
      /**
       * @brief   Free the current precondioner.
       * @details Free the current precondioner.
       * @return     Return error message.
       */
      virtual int Clear();
     
      /**
       * @brief   The destructor of PolyClass.
       * @details The destructor of PolyClass.
       */
      virtual ~PolyClass();
      
      /**
       * @brief   Setup the precondioner phase. Will be called by the solver if not called directly.
       * @details Setup the precondioner phase. Will be called by the solver if not called directly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Get the size of the problem.
       * @details Get the size of the problem.
       * @return     Return the problem size.
       */
      int         GetSize();
      
      /**
       * @brief   Get the total number of nonzeros the preconditioner.
       * @details Get the total number of nonzeros the preconditioner.
       * @return     Return the total number of nonzeros the preconditioner.
       */
      virtual long int  GetNumNonzeros();
      
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
      
      /* ------- SETS and GETS ------- */
      
      /**
       * @brief   Setup with parameter array.
       * @details Setup with parameter array.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetWithParameterArray(double *params)
      {
         this->_order = (int)params[PARGEMSLR_IO_POLY_ORDER];
         this->_print_option = params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL];
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the order of the polynomia.
       * @details Set the order of the polynomia.
       * @param   [in]   order The order.
       * @return     Return error message.
       */
      int         SetOrder( int order)
      {
         this->_order = order;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the diagonal scale of the problem.
       * @details Set the diagonal scale of the problem.
       * @param   [in]   scale The scale.
       * @return     Return error message.
       */
      template <typename T>
      int         SetDiagScale(T scale)
      {
         this->_scale = scale;
         return PARGEMSLR_SUCCESS;
      }
      
	};
   
   typedef PolyClass<CsrMatrixClass<float>, SequentialVectorClass<float>, float>                precond_poly_csr_seq_float;
   typedef PolyClass<CsrMatrixClass<double>, SequentialVectorClass<double>, double>             precond_poly_csr_seq_double;
   typedef PolyClass<CsrMatrixClass<complexs>, SequentialVectorClass<complexs>, complexs>          precond_poly_csr_seq_complexs;
   typedef PolyClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd>         precond_poly_csr_seq_complexd;
   typedef PolyClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>          precond_poly_csr_par_float;
   typedef PolyClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>       precond_poly_csr_par_double;
   typedef PolyClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>    precond_poly_csr_par_complexs;
   typedef PolyClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>   precond_poly_csr_par_complexd;
   
}

#endif
