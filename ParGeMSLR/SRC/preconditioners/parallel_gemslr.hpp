#ifndef PARGEMSLR_PARALLEL_GEMSLR_H
#define PARGEMSLR_PARALLEL_GEMSLR_H

/**
 * @file parallel_gemslr.hpp
 * @brief Parallel GeMSLR preconditioner.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../utils/structs.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../solvers/solver.hpp"
#include "gemslr.hpp"

#ifdef PARGEMSLR_CUDA
#include "cusparse.h"
#endif

namespace pargemslr
{
   
   template <class MatrixType, class VectorType, typename DataType> class ParallelGemslrClass;
   
	/**
    * @brief   Class of matvec for low-rank correction, not necessarily EB^{-1}FC^{-1}.
    * @details Class of matvec for low-rank correction, not necessarily EB^{-1}FC^{-1}.
    *          kGemslrGlobalPrecondESMSLR, kGemslrGlobalPrecondGeMSLR, kGemslrGlobalPrecondPSLR
    */
   template <class MatrixType, class VectorType, typename DataType>
   class ParallelGemslrEBFCMatrixClass
   {
   private:
      
      /* variables */
      
      /**
       * @brief   The level of the matvec.
       * @details The level of the matvec.
       */
      int                                                         _level;
      
      /**
       * @brief   The matvec option.
       * @details The matvec option.
       */
      int                                                         _option;
      
      /**
       * @brief   Vector holding the column index data.
       * @details Vector holding the column index data.
       */
      
      ParallelGemslrClass<MatrixType, VectorType, DataType>       *_gemslr;
      
   public:
      
      /**
       * @brief   Temp vector for the Arnoldi.
       * @details Temp vector for the Arnoldi.
       */
      VectorType                                                  _temp_v;
      
      /**
       * @brief   The constructor of ParallelGemslrEBFCMatrixClass.
       * @details The constructor of ParallelGemslrEBFCMatrixClass. The default memory location is the host memory.
       */
      ParallelGemslrEBFCMatrixClass();
      
      /**
       * @brief   The destructor of ParallelGemslrEBFCMatrixClass.
       * @details The destructor of ParallelGemslrEBFCMatrixClass. Simply a call to the free function.
       */
      virtual ~ParallelGemslrEBFCMatrixClass();
      
      /**
       * @brief   The copy constructor of ParallelGemslrEBFCMatrixClass.
       * @details The copy constructor of ParallelGemslrEBFCMatrixClass.
       */
      ParallelGemslrEBFCMatrixClass(const ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The move constructor of ParallelGemslrEBFCMatrixClass.
       * @details The move constructor of ParallelGemslrEBFCMatrixClass.
       */
      ParallelGemslrEBFCMatrixClass(ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   The operator = of ParallelGemslrEBFCMatrixClass.
       * @details The operator = of ParallelGemslrEBFCMatrixClass.
       */
      ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>& operator=(const ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The operator = of ParallelGemslrEBFCMatrixClass.
       * @details The operator = of ParallelGemslrEBFCMatrixClass.
       */
      ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>& operator=(ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   Set the current matrix to a certain GEMSLR level.
       * @details Set the current matrix to a certain GEMSLR level.
       * @param [in] level The level.
       * @param [in] option The matvec option.
       * @param [in] gemslr The GeMSLR structure.
       * @return     Return error message.
       */
      int      Setup(int level, int option, ParallelGemslrClass<MatrixType, VectorType, DataType> &gemslr);
      
      /**
       * @brief   Set the structure of a vector pointer that has same row partition as this matrix.
       * @details Set the structure of a vector pointer that has same row partition as this matrix.
       * @param [in,out] v The target vector.
       * @return     Return error message.
       */
      int      SetupVectorPtrStr(VectorType &v);
      
      /**
       * @brief   Free the current matrix.
       * @details Free the current matrix.
       * @return  Return error message.
       */
      int      Clear();
      
      /**
       * @brief   Get the local number of rows of the matrix.
       * @details Get the local number of rows of the matrix.
       * @return     Return the number of rows of the matrix.
       */
      int      GetNumRowsLocal();
      
      /**
       * @brief   Get the local number of columns of the matrix.
       * @details Get the local number of columns of the matrix.
       * @return     Return the number of columns of the matrix.
       */
      int      GetNumColsLocal();
      
      /**
       * @brief   In place csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @details In place csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int MatVec( char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   Get comm, np, and myid. Get the global one.
       * @details Get comm, np, and myid. Get the global one.
       * @param   [in]        np The number of processors.
       * @param   [in]        myid The local MPI rank number.
       * @param   [in]        comm The MPI_Comm.
       * @return     Return error message.
       */
      int GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const
      {
         this->_gemslr->GetMatrix()->GetMpiInfo(np, myid, comm);
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Get the MPI_comm.
       * @details Get the MPI_comm.
       * @return     Return the MPI_comm.
       */
      MPI_Comm       GetComm() const
      {
         return this->_gemslr->GetMatrix()->GetComm();
      }
      
      /**
       * @brief   Get the data location of the matrix.
       * @details Get the data location of the matrix.
       * @return     Return the length of the matrix.
       */
      int GetDataLocation() const
      {
         if(this->_gemslr)
         {
            return this->_gemslr->GetMatrix()->GetDataLocation();
         }
         return kMemoryHost;
      }
      
   };
   
   typedef ParallelGemslrEBFCMatrixClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>              precond_gemslrebfc_csr_par_float;
   typedef ParallelGemslrEBFCMatrixClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>           precond_gemslrebfc_csr_par_double;
   typedef ParallelGemslrEBFCMatrixClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>     precond_gemslrebfc_csr_par_complexs;
   typedef ParallelGemslrEBFCMatrixClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>     precond_gemslrebfc_csr_par_complexd;
   
	/**
    * @brief   Class of matvec with the Schur Complement.
    * @details Class of matvec with the Schur Complement.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class ParallelGemslrSchurMatrixClass
   {
   private:
      
      /* variables */
      
      /**
       * @brief   Vector holding the column pointer data.
       * @details Vector holding the column pointer data.
       */
      int                                                      _level;
      
      /**
       * @brief   The matvec option.
       * @details The matvec option.
       */
      int                                                      _option;
      
      /**
       * @brief   Vector holding the column index data.
       * @details Vector holding the column index data.
       */
      ParallelGemslrClass<MatrixType, VectorType, DataType>    *_gemslr;
      
   public:
      
      /**
       * @brief   Temp vector for the Arnoldi.
       * @details Temp vector for the Arnoldi.
       */
      VectorType                                               _temp_v;
      
      /**
       * @brief   The constructor of ParallelGemslrSchurMatrixClass.
       * @details The constructor of ParallelGemslrSchurMatrixClass. The default memory location is the host memory.
       */
      ParallelGemslrSchurMatrixClass();
      
      /**
       * @brief   The copy constructor of ParallelGemslrSchurMatrixClass.
       * @details The copy constructor of ParallelGemslrSchurMatrixClass.
       */
      ParallelGemslrSchurMatrixClass(const ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The move constructor of ParallelGemslrSchurMatrixClass.
       * @details The move constructor of ParallelGemslrSchurMatrixClass.
       */
      ParallelGemslrSchurMatrixClass(ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   The operator = of ParallelGemslrSchurMatrixClass.
       * @details The operator = of ParallelGemslrSchurMatrixClass.
       */
      ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>& operator=(const ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The operator = of ParallelGemslrSchurMatrixClass.
       * @details The operator = of ParallelGemslrSchurMatrixClass.
       */
      ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>& operator=(ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   The destructor of GemslrEBFCMatrixClass.
       * @details The destructor of GemslrEBFCMatrixClass. Simply a call to the free function.
       */
      virtual ~ParallelGemslrSchurMatrixClass();
      
      /**
       * @brief   Set the current matrix to a certain GEMSLR level.
       * @details Set the current matrix to a certain GEMSLR level.
       * @param [in] level The level.
       * @param [in] option The option.
       * @param [in] gemslr The GeMSLR structure.
       * @return     Return error message.
       */
      int      Setup(int level, int option, ParallelGemslrClass<MatrixType, VectorType, DataType> &gemslr);
      
      /**
       * @brief   Set the structure of a vector pointer that has same row partition as this matrix.
       * @details Set the structure of a vector pointer that has same row partition as this matrix.
       * @param [in,out] v The target vector.
       * @return     Return error message.
       */
      int      SetupVectorPtrStr(VectorType &v);
      
      /**
       * @brief   Free the current matrix.
       * @details Free the current matrix.
       * @return  Return error message.
       */
      int      Clear();
      
      /**
       * @brief   In place csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @details In place csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int MatVec( char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      int Solve( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Get comm, np, and myid. Get the global one.
       * @details Get comm, np, and myid. Get the global one.
       * @param   [in]        np The number of processors.
       * @param   [in]        myid The local MPI rank number.
       * @param   [in]        comm The MPI_Comm.
       * @return     Return error message.
       */
      int GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const
      {
         this->_gemslr->GetMatrix()->GetMpiInfo(np, myid, comm);
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Get the MPI_comm.
       * @details Get the MPI_comm.
       * @return     Return the MPI_comm.
       */
      MPI_Comm       GetComm() const
      {
         return this->_gemslr->GetMatrix()->GetComm();
      }
      
   };
   
   typedef ParallelGemslrSchurMatrixClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>             precond_gemslr_schur_par_float;
   typedef ParallelGemslrSchurMatrixClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>          precond_gemslr_schur_par_double;
   typedef ParallelGemslrSchurMatrixClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>    precond_gemslr_schur_par_complexs;
   typedef ParallelGemslrSchurMatrixClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>    precond_gemslr_schur_par_complexd;
   
	/**
    * @brief   Class of schur matvec.
    * @details Class of schur matvec.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class ParallelGemslrSchurSolveClass: public SolverClass<MatrixType, VectorType, DataType>
   {
	public:
      
      /**
       * @brief   The constructor of precondioner class.
       * @details The constructor of precondioner class.
       */
      ParallelGemslrSchurSolveClass() : SolverClass<MatrixType, VectorType, DataType>(){}
      
      /**
       * @brief   Free the current precondioner.
       * @details Free the current precondioner.
       * @return     Return error message.
       */
      virtual int Clear(){SolverClass<MatrixType, VectorType, DataType>::Clear();return PARGEMSLR_SUCCESS;}
      
      /**
       * @brief   The move constructor of ParallelGemslrSchurSolveClass class.
       * @details The move constructor of ParallelGemslrSchurSolveClass class.
       */
      ParallelGemslrSchurSolveClass(ParallelGemslrSchurSolveClass<MatrixType, VectorType, DataType> &precond) : SolverClass<MatrixType, VectorType, DataType>(precond)
      {
         this->Clear();
      }
      
      /**
       * @brief   The move constructor of ParallelGemslrSchurSolveClass class.
       * @details The move constructor of ParallelGemslrSchurSolveClass class.
       */
      ParallelGemslrSchurSolveClass(ParallelGemslrSchurSolveClass<MatrixType, VectorType, DataType> &&precond) : SolverClass<MatrixType, VectorType, DataType>(std::move(precond))
      {
         this->Clear();
      }
      
      /**
       * @brief   The operator = of ParallelGemslrSchurSolveClass class.
       * @details The operator = of ParallelGemslrSchurSolveClass class.
       */
      ParallelGemslrSchurSolveClass<MatrixType, VectorType, DataType>& operator=(const ParallelGemslrSchurSolveClass<MatrixType, VectorType, DataType> &precond)
      {
         this->Clear();
         SolverClass<MatrixType, VectorType, DataType>::operator=(precond);
         return *this;
      }
      
      /**
       * @brief   The operator = of ParallelGemslrSchurSolveClass class.
       * @details The operator = of ParallelGemslrSchurSolveClass class.
       */
      ParallelGemslrSchurSolveClass<MatrixType, VectorType, DataType>& operator=(ParallelGemslrSchurSolveClass<MatrixType, VectorType, DataType> &&precond)
      {
         this->Clear();
         SolverClass<MatrixType, VectorType, DataType>::operator=(std::move(precond));
         return *this;
      }
      
      /**
       * @brief   The destructor of precondioner class.
       * @details The destructor of precondioner class.
       */
      virtual ~ParallelGemslrSchurSolveClass(){this->Clear();}
      
      /**
       * @brief   Setup the precondioner phase. Will be called by the solver if not called directly.
       * @details Setup the precondioner phase. Will be called by the solver if not called directly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( VectorType &x, VectorType &rhs)
      {
         if(this->_ready)
         {
            return PARGEMSLR_SUCCESS;
         }
         
         if(!this->_matrix)
         {
            PARGEMSLR_ERROR("Setup without matrix.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         
         this->_ready = true;
         
         return PARGEMSLR_SUCCESS;
         
      }
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( VectorType &x, VectorType &rhs)
      {
         
         if(!(this->_ready))
         {
            PARGEMSLR_ERROR("Solve without setup.");
            return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
         }
         
         this->_matrix->Solve(x, rhs);
         
         return PARGEMSLR_SUCCESS;
         
      }
      
      /**
       * @brief   Get the total number of nonzeros the preconditioner.
       * @details Get the total number of nonzeros the preconditioner.
       * @return     Return the total number of nonzeros the preconditioner.
       */
      virtual long int  GetNumNonzeros(){return 0;};
      
      /**
       * @brief      Set the data location that the preconditioner apply to.
       * @details    Set the data location that the preconditioner apply to.
       * @param [in] location The target solver location.
       * @return     Return error message.
       */
      virtual int SetSolveLocation( const int &location){return PARGEMSLR_SUCCESS;};
      
      /**
       * @brief      Move the preconditioner to another location. Only can be called after Setup.
       * @details    Move the preconditioner to another location. Only can be called after Setup.
       * @param [in] location The target solver location.
       * @return     Return error message.
       */
      virtual int MoveData( const int &location){return PARGEMSLR_SUCCESS;};
      
      /* ------- SETS and GETS ------- */
      
      /**
       * @brief   Setup with parameter array.
       * @details Setup with parameter array.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetWithParameterArray(double *params)
      {
         this->_print_option = params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL];
         return PARGEMSLR_SUCCESS;
      }
      
   };
   
   typedef ParallelGemslrSchurMatrixClass<precond_gemslr_schur_par_float, ParallelVectorClass<float>, float>             precond_gemslr_schursolve_par_float;
   typedef ParallelGemslrSchurMatrixClass<precond_gemslr_schur_par_double, ParallelVectorClass<double>, double>          precond_gemslr_schursolve_par_double;
   typedef ParallelGemslrSchurMatrixClass<precond_gemslr_schur_par_complexs, ParallelVectorClass<complexs>, complexs>    precond_gemslr_schursolve_par_complexs;
   typedef ParallelGemslrSchurMatrixClass<precond_gemslr_schur_par_complexd, ParallelVectorClass<complexd>, complexd>    precond_gemslr_schursolve_par_complexd;
   
   /**
    * @brief   The GEMSLR information on each level, contains the solver for B and the low-rank information for S.
    * @details The GEMSLR information on each level, contains the solver for B and the low-rank information for S.
    *          VectorType is the type of the vector. DataType is the data type.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class ParallelGemslrLevelClass
   {
   public:
   
      /** 
       * @brief   The parallel log data structure.
       * @details The parallel log data structure.
       */
      parallel_log                                       _parlog;
      
      /* Variables */
      
      /** 
       * @brief   The size of low-rank correction on this level.
       * @details The size of low-rank correction on this level.
       */
      int                                                _lrc;
      
      /** 
       * @brief   Number of subdomains on this level.
       * @details Number of subdomains on this level.
       */
      int                                                _ncomps;
      
      /** 
       * @brief   The B matrix on this level. C matrix if this is the last level.
       * @details The B matrix on this level. C matrix if this is the last level.
       */
      std::vector<CsrMatrixClass<DataType> >             _B_mat_v;
      
      /** 
       * @brief   The E matrix on this level.
       * @details The E matrix on this level.
       */
      MatrixType                                         _E_mat;
      
      /** 
       * @brief   The F matrix on this level.
       * @details The F matrix on this level.
       */
      MatrixType                                         _F_mat;
      
      /** 
       * @brief   The A matrix on this level.
       * @details The A matrix on this level.
       */
      MatrixType                                         _A_mat;
      
      /** 
       * @brief   The C matrix on this level.
       * @details The C matrix on this level.
       */
      MatrixType                                         _C_mat;
      
      /** 
       * @brief   The S matrix on this level.
       * @details The S matrix on this level.
       */
      MatrixType                                         _S_mat;
      
      /** 
       * @brief   Number of interior nodes on the top level.
       * @details Number of interior nodes on the top level.
       */
      int                                                _nI;
      
      /** 
       * @brief   The EBFC matrix on this level.
       * @details The EBFC matrix on this level.
       */
      ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType> _EBFC;
      
      /** 
       * @brief   The preconditioners for B matrix.
       * @details The preconditioners for B matrix.
       */
      SolverClass<CsrMatrixClass<DataType>, SequentialVectorClass<DataType>, DataType> **_B_precond;
      
      /** 
       * @brief   The solvers for B matrix.
       * @details The solvers for B matrix.
       */
      SolverClass<CsrMatrixClass<DataType>, SequentialVectorClass<DataType>, DataType> **_B_solver;
      
      /** 
       * @brief   The H matrix for the low-rank correction W*H*W' on this level.
       * @details The H matrix for the low-rank correction W*H*W' on this level.
       */
      DenseMatrixClass<DataType>                         _Hk;
      
      /** 
       * @brief   The W matrix for the low-rank correction W*H*W' on this level.
       * @details The W matrix for the low-rank correction W*H*W' on this level.
       */
      DenseMatrixClass<DataType>                         _Wk;
      
      /** 
       * @brief   The WH matrix for the low-rank correction (W*H)*W' on this level.
       * @details The WH matrix for the low-rank correction (W*H)*W' on this level.
       */
      DenseMatrixClass<DataType>                         _WHk;
      
      /** 
       * @brief   The H matrix for the low-rank correction W*H*W' on the last level.
       * @details The H matrix for the low-rank correction W*H*W' on the last level.
       */
      DenseMatrixClass<DataType>                         _cHk;
      
      /** 
       * @brief   The W matrix for the low-rank correction W*H*W' on the last level.
       * @details The W matrix for the low-rank correction W*H*W' on the last level.
       */
      DenseMatrixClass<DataType>                         _cWk;
      
      /** 
       * @brief   The WH matrix for the low-rank correction (W*H)*W' on the last level.
       * @details The WH matrix for the low-rank correction (W*H)*W' on the last level.
       */
      DenseMatrixClass<DataType>                         _cWHk;
      
      /** 
       * @brief   The row permutation vector.
       * @details The row permutation vector.
       */
      IntVectorClass<int>                                _pperm;
      
      /** 
       * @brief   The column permutation vector.
       * @details The column permutation vector.
       * @note    For non-symmetric reordering.
       */
      IntVectorClass<int>                                _qperm;
      
      /** 
       * @brief   The communication helper.
       * @details The communication helper.
       */
      CommunicationHelperClass                           _comm_helper;
      
      /** 
       * @brief   Temp vector on this level.
       * @details Temp vector on this level.
       */
      SequentialVectorClass<DataType>                    _work_vector;
      
      /** 
       * @brief   The unit length of the work vector.
       * @details The unit length of the work vector.
       */
      int                                                _work_vector_unit_length;
      
#ifdef PARGEMSLR_DEBUG
      /** 
       * @brief   The status of the working vector, used in debug. 0: not used; 1: occupied.
       * @details The status of the working vector, used in debug. 0: not used; 1: occupied.
       */
      IntVectorClass<int>                                _work_vector_occupied;
      
#endif

      /** 
       * @brief   Temp vector, length equal to the size of this level, used to setup the Ptr for this level.
       * @details Temp vector, length equal to the size of this level, used to setup the Ptr for this level.
       */
      VectorType                                         _x_temp;
      
      /** 
       * @brief   Temp vector.
       * @details Temp vector.
       */
      VectorType                                         _xlr_temp;
      
      /** 
       * @brief   Temp vector.
       * @details Temp vector.
       */
      VectorType                                         _xlr1_temp;
      
      /** 
       * @brief   Temp vector.
       * @details Temp vector.
       */
      VectorType                                         _xlr2_temp;
      
      /** 
       * @brief   Temp vector.
       * @details Temp vector.
       */
      VectorType                                         _xlr1_temp_h;
      
      /** 
       * @brief   Temp vector.
       * @details Temp vector.
       */
      VectorType                                         _xlr2_temp_h;
      
      /** 
       * @brief   The temp vector for permuted x.
       * @details The temp vector for permuted x.
       */
      VectorType                                         _sol_temp;
      
      /** 
       * @brief   The temp vector for permuted rhs.
       * @details The temp vector for permuted rhs.
       */
      VectorType                                         _rhs_temp;
      
      /** 
       * @brief   The temp vector for permuted x.
       * @details The temp vector for permuted x.
       */
      VectorType                                         _sol2_temp;
      
      /** 
       * @brief   The temp vector for permuted rhs.
       * @details The temp vector for permuted rhs.
       */
      VectorType                                         _rhs2_temp;
      
      /** 
       * @brief   The temp vector for permuted x for the upper level.
       * @details The temp vector for permuted x for the upper level.
       */
      VectorType                                         _sol3_temp;
      
      /** 
       * @brief   The temp vector for permuted rhs for the upper level.
       * @details The temp vector for permuted rhs for the upper level.
       */
      VectorType                                         _rhs3_temp;
      
      /**
       * @brief   Free the current level structure, set everything to 0.
       * @details Free the current level structure, set everything to 0.
       * @return           return GEMSLR_SUCESS or error information.
       */
      int                                                Clear();
      
      /**
       * @brief   The constructor of ParallelGemslrLevelClass, set everything to 0.
       * @details The constructor of ParallelGemslrLevelClass, set everything to 0.
       */
      ParallelGemslrLevelClass();
      
      /**
       * @brief   The destructor of ParallelGemslrLevelClass.
       * @details The destructor of ParallelGemslrLevelClass, simply call the free function.
       */
      ~ParallelGemslrLevelClass();
      
      /**
       * @brief   The copy constructor of ParallelGemslrLevelClass.
       * @details The copy constructor of ParallelGemslrLevelClass.
       */
      ParallelGemslrLevelClass(const ParallelGemslrLevelClass<MatrixType, VectorType, DataType> &str);
      
      /**
       * @brief   The move constructor of ParallelGemslrLevelClass.
       * @details The move constructor of ParallelGemslrLevelClass.
       */
      ParallelGemslrLevelClass(ParallelGemslrLevelClass<MatrixType, VectorType, DataType> &&str);
      
      /**
       * @brief   The operator= of ParallelGemslrLevelClass.
       * @details The operator= of ParallelGemslrLevelClass.
       */
      ParallelGemslrLevelClass<MatrixType, VectorType, DataType>& operator=(const ParallelGemslrLevelClass<MatrixType, VectorType, DataType> &str);
      
      /**
       * @brief   The operator= of ParallelGemslrLevelClass.
       * @details The operator= of ParallelGemslrLevelClass.
       */
      ParallelGemslrLevelClass<MatrixType, VectorType, DataType>& operator=(ParallelGemslrLevelClass<MatrixType, VectorType, DataType> &&str);
      
      /**
       * @brief   Get the number of nonzeros in the low-rank correction and the ILU factorization on this level.
       * @details Get the number of nonzeros in the low-rank correction and the ILU factorization on this level. \n
       *          Note that this is the sequential version, int would be enough. No need to use the long int.
       * @param [out]      nnz_ilu The nnz for the ILU factorization.
       * @param [out]      nnz_lr The nnz for the low-rank correction.
       * @return           return nnz_ilu+nnz_lr.
       */
      int                                                GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
      
   };
   
   typedef ParallelGemslrLevelClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>             precond_gemslrlevel_csr_par_float;
   typedef ParallelGemslrLevelClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>          precond_gemslrlevel_csr_par_double;
   typedef ParallelGemslrLevelClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>    precond_gemslrlevel_csr_par_complexs;
   typedef ParallelGemslrLevelClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>    precond_gemslrlevel_csr_par_complexd;
   
   /**
    * @brief   The local real ilu preconditioner, only work for sequential CSR matrix.
    * @details The local real ilu preconditioner, only work for sequential CSR matrix. Matrix type is the type of the matrix, 
    *          VectorType is the type of the vector. DataType is the data type.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class ParallelGemslrClass: public SolverClass<MatrixType, VectorType, DataType>
   {
   private:
      
      /**
       * @brief   The size of the problem.
       * @details The size of the problem.
       */
      int                                 _n;
      
      /**
       * @brief   The GEMSLR setups.
       * @details The GEMSLR setups.
       */
      GemslrSetupStruct<DataType>         _gemslr_setups;
      
      /** 
       * @brief   Matrix for the inner iteration.
       * @details Matrix for the inner iteration.
       */
      ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType> _inner_iters_matrix;
      
      /** 
       * @brief   Precond for the inner iteration.
       * @details Precond for the inner iteration.
       */
      ParallelGemslrSchurSolveClass<ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>,
                              VectorType, DataType> _inner_iters_precond;
      
      /** 
       * @brief   Solver for the inner iteration.
       * @details Solver for the inner iteration.
       */
      FlexGmresClass<ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>, 
                              VectorType, DataType> _inner_iters_solver;
      
      /** 
       * @brief   The number of levels actually computed. In the solve phase, the last several levels might be combined together.
       * @details The number of levels actually computed. In the solve phase, the last several levels might be combined together.
       * @note    Not yet used.
       */
      int                                 _nlev_max;
      
      /** 
       * @brief   The number of levels in the C part. Can be smaller than _nlev_max. The total level number is _nlev_used + 1.
       * @details The number of levels in the C part. Can be smaller than _nlev_max. The total level number is _nlev_used + 1.
       */
      int                                 _nlev_used;
      
      /** 
       * @brief   The pointer to the start node of each level.
       * @details The pointer to the start node of each level.
       */
      IntVectorClass<int>                 _lev_ptr_v;
      
      /** 
       * @brief   The 2D vector pointer to the start node of each subdomain on each level.
       * @details The 2D vector pointer to the start node of each subdomain on each level.
       */
      std::vector<IntVectorClass<int> >   _dom_ptr_v2;      
      
      /** 
       * @brief   The local row permutation vector.
       * @details The local row permutation vector.
       */
      IntVectorClass<long int>            _pperm;
      
      /** 
       * @brief   The local column permutation vector.
       * @details The local column permutation vector.
       * @note    For non-symmetric reordering.
       */
      IntVectorClass<long int>            _qperm;
      
      /**
       * @brief   The location this preconditioner applied to.
       * @details The location this preconditioner applied to.
       */
      int                                 _location;
      
      /**
       * @brief   Free the current precondioner.
       * @details Free the current precondioner.
       * @return     Return error message.
       */
      template <typename RealDataType>
      int   OrdLowRank(int m, int &rank, RealDataType (*weight)(ComplexValueClass<RealDataType>), DenseMatrixClass<RealDataType> &R, DenseMatrixClass<RealDataType> &Q);
      
      /**
       * @brief   Free the current precondioner.
       * @details Free the current precondioner.
       * @return     Return error message.
       */
      template <typename RealDataType>
      int   OrdLowRank(int m, int &rank, RealDataType (*weight)(ComplexValueClass<RealDataType>), DenseMatrixClass<ComplexValueClass<RealDataType> > &R, DenseMatrixClass<ComplexValueClass<RealDataType> > &Q);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @details Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupLowRankThickRestartNoLock( ParallelVectorClass<DataType> &x, ParallelVectorClass<DataType> &rhs, int level, int option);
      
      /**
       * @brief   Compute the distance for the selection of eigenvalues.
       * @details Compute the distance for the selection of eigenvalues.
       * @param [in] val the value.
       * @return     Return the distance.
       */
      template <typename T1, typename T2>
      static T1     ComputeDistance(T2 val);
      
      /**
       * @brief   Compute the distance for the selection of eigenvalues for test.
       * @details Compute the distance for the selection of eigenvalues for test.
       * @param [in] val the value.
       * @return     Return the distance.
       */
      template <typename T1, typename T2>
      static T1     ComputeDistanceSC(T2 val);
      
      /**
       * @brief   Print the preconditioner info.
       * @details Print the preconditioner info.
       * @return     Return error message.
       */
      int PrintInfo();
      
	public:
      
      /* experimental algorithms */
      
      /**
       * @brief   The Schur complement option.
       * @details The Schur complement option.
       */
      int                                 _global_precond_option;
      
      /**
       * @brief   Setup the local partial ILUT.
       * @details Setup the local partial ILUT.
       */
      int SetupPartialILUT( VectorType &x, VectorType &rhs);
      
      /* end of experiment part */
      
      /** 
       * @brief   Vector holding the level struct for all levels starting from the second level.
       *          The size of this vector is equal to _nlev_used, and is the number of true levels.
       * @details Vector holding the level struct for all levels starting from the second level.
       *          The size of this vector is equal to _nlev_used, and is the number of true levels.
       */
      std::vector< ParallelGemslrLevelClass< MatrixType, VectorType, DataType> >  _levs_v;
      
      /** 
       * @brief   The structure for the A level low-rank correction.
       * @details The structure for the A level low-rank correction.
       */
      std::vector< ParallelGemslrLevelClass< MatrixType, VectorType, DataType> >  _lev_A;
      
      /**
       * @brief   The constructor of precondioner class.
       * @details The constructor of precondioner class.
       */
      ParallelGemslrClass();
     
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
      virtual ~ParallelGemslrClass();
      
      /**
       * @brief   The copy constructor of ParallelGemslrClass.
       * @details The copy constructor of ParallelGemslrClass.
       */
      ParallelGemslrClass(const ParallelGemslrClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The move constructor of ParallelGemslrClass.
       * @details The move constructor of ParallelGemslrClass.
       */
      ParallelGemslrClass(ParallelGemslrClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   The operator = of ParallelGemslrClass.
       * @details The operator = of ParallelGemslrClass.
       */
      ParallelGemslrClass<MatrixType, VectorType, DataType>& operator=(const ParallelGemslrClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The operator = of ParallelGemslrClass.
       * @details The operator = of ParallelGemslrClass.
       */
      ParallelGemslrClass<MatrixType, VectorType, DataType>& operator=(ParallelGemslrClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   Check the input parameters, avoid invalid settings.
       * @details Check the input parameters, avoid invalid settings.
       */
      int CheckParameters();
      
      /**
       * @brief   Setup the precondioner phase. Will be called by the solver if not called directly.
       * @details Setup the precondioner phase. Will be called by the solver if not called directly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Setup the permutation of the GeMSLR.
       * @details Setup the permutation of the GeMSLR.
       * @return     Return error message.
       */
      int SetupPermutation();
      
      /**
       * @brief   Setup the Recursive KWay partition of the GeMSLR.
       * @details Setup the Recursive KWay partition of the GeMSLR. This is a symmetric reordering algorithm work on A+A'. \n
       *          1) On each level, partition the current adjacency graph with kway partitioning. \n
       *          2) Find the adjacency graph of the interface matrix. \n
       *          3) Repeat until we reach the last level or the interface matrix is too small.
       * @return     Return error message.
       */
      int SetupPermutationRKway(MatrixType &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
      
      /**
       * @brief   Setup the Nested Dissection partition of the GeMSLR.
       * @details Setup the Nested Dissection partition of the GeMSLR. This is a symmetric reordering algorithm work on A+A'.
       * @return     Return error message.
       */
      int SetupPermutationND(MatrixType &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
      
      /**
       * @brief   Setup the level structure of the GeMSLR.
       * @details Setup the level structure of the GeMSLR.
       * @return     Return error message.
       */
      int SetupPermutationBuildLevelStructure( MatrixType &A, int level_start, vector_int &map_v, vector_int &mapptr_v);
      
      /**
       * @brief   Setup the solve of B matrices of the GeMSLR.
       * @details Setup the solve of B matrices of the GeMSLR.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      int SetupBSolve( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Setup the solve of B matrices of the GeMSLR with ILUT.
       * @details Setup the solve of B matrices of the GeMSLR with ILUT.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupBSolveILUT( VectorType &x, VectorType &rhs, int level);
      
      /**
       * @brief   Setup the solve of B matrices of the GeMSLR with ILUK.
       * @details Setup the solve of B matrices of the GeMSLR with ILUK.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupBSolveILUK( VectorType &x, VectorType &rhs, int level);
      
      /**
       * @brief   Setup the solve of B matrices of the GeMSLR with Poly.
       * @details Setup the solve of B matrices of the GeMSLR with Poly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupBSolvePoly( VectorType &x, VectorType &rhs, int level);
      
      /**
       * @brief   Setup the solve of B matrices of the GeMSLR with GeMSLR.
       * @details Setup the solve of B matrices of the GeMSLR with GeMSLR.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupBSolveGemslr( VectorType &x, VectorType &rhs, int level);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR.
       * @details Setup the low-rank part of the GeMSLR.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      int SetupLowRank( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR with subspace iteration.
       * @details Setup the low-rank part of the GeMSLR with subspace iteration
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @param [in]    option The setup option. GeMSLR, ESMSLR, or PSLR.
       * @return     Return error message.
       */
      int SetupLowRankSubspaceIteration( VectorType &x, VectorType &rhs, int level, int option);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR with arnodi no-restart.
       * @details Setup the low-rank part of the GeMSLR with arnodi no-restart.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @param [in]    option The setup option. GeMSLR, ESMSLR, or PSLR.
       * @return     Return error message.
       */
      int SetupLowRankNoRestart( VectorType &x, VectorType &rhs, int level, int option);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @details Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @param [in]    option The setup option. GeMSLR, ESMSLR, or PSLR.
       * @return     Return error message.
       */
      int SetupLowRankThickRestart( VectorType &x, VectorType &rhs, int level, int option);
      
      /**
       * @brief   Given the V and H from Arnoldi, compute the final low-rank correction of S^{-1} - C^{-1}.
       * @details Given the V and H from Arnoldi, compute the final low-rank correction of S^{-1} - C^{-1}. \n
       *          For GeMSLR, H -> (I-H)^{-1} - I \n
       *          For EsMSLR, X -> X^{-1} - I.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param [in]    V Matrix holding the orthogonal basis genrated by Arnoldi iteration.
       * @param [in]    H The Hessenberg matrix generated by Arnoldi iteration.
       * @param [in]    m The number of finished steps in Arnoldi iteration.
       * @param [in]    rank The number of eigenvectors to keep.
       * @param [in]    level The current level.
       * @param [in]    option The setup option. GeMSLR, ESMSLR, or PSLR.
       * @return        return # of Schur vectors, if return value < 0 an error occurs                                                                                                                                                    
       */
      int SetupLowRankBuildLowRank( VectorType &x, VectorType &rhs, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, int m, int rank, int level, int option);
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( VectorType &x, VectorType &rhs);
      
      /**
       * @brief   Solve starting from a certain level.
       * @details Solve starting from a certain level.
       * @param   [in,out] x_in The initial guess.
       * @param   [in]     rhs_out The right-hand-side.
       * @param   [in]     level The start level.
       * @param   [in]     doperm Do we apply permutation on this level? (Some time the system is already permuted).
       * @return     Return error message.
       */
      int         SolveLevelGemslr( VectorType &x_out, VectorType &rhs_in, int level, bool doperm);
      
      /**
       * @brief   Solve starting from a certain level.
       * @details Solve starting from a certain level.
       * @param   [in,out] x_in The initial guess.
       * @param   [in]     rhs_out The right-hand-side.
       * @param   [in]     level The start level.
       * @param   [in]     doperm Do we apply permutation on this level? (Some time the system is already permuted).
       * @return     Return error message.
       */
      int         SolveLevelGemslrU( VectorType &x_out, VectorType &rhs_in, int level, bool doperm);
      
      /**
       * @brief   Solve starting from a certain level.
       * @details Solve starting from a certain level.
       * @param   [in,out] x_in The initial guess.
       * @param   [in]     rhs_out The right-hand-side.
       * @param   [in]     level The start level.
       * @param   [in]     doperm Do we apply permutation on this level? (Some time the system is already permuted).
       * @return     Return error message.
       */
      int         SolveLevelGemslrMul( VectorType &x_out, VectorType &rhs_in, int level, bool doperm);
      
      /**
       * @brief   Solve starting from a certain level.
       * @details Solve starting from a certain level.
       * @param   [in,out] x_in The initial guess.
       * @param   [in]     rhs_out The right-hand-side.
       * @param   [in]     level The start level.
       * @param   [in]     doperm Do we apply permutation on this level? (Some time the system is already permuted).
       * @return     Return error message.
       */
      int         SolveLevelEsmslr( VectorType &x_out, VectorType &rhs_in, int level, bool doperm);
      
      /**
       * @brief   Solve starting from a certain level.
       * @details Solve starting from a certain level.
       * @param   [in,out] x_in The initial guess.
       * @param   [in]     rhs_out The right-hand-side.
       * @param   [in]     level The start level.
       * @param   [in]     doperm Do we apply permutation on this level? (Some time the system is already permuted).
       * @return     Return error message.
       */
      int         SolveLevelEsmslrU( VectorType &x_out, VectorType &rhs_in, int level, bool doperm);
      
      /**
       * @brief   Solve starting from a certain level.
       * @details Solve starting from a certain level.
       * @param   [in,out] x_in The initial guess.
       * @param   [in]     rhs_out The right-hand-side.
       * @param   [in]     level The start level.
       * @param   [in]     doperm Do we apply permutation on this level? (Some time the system is already permuted).
       * @return     Return error message.
       */
      int         SolveLevelEsmslrMul( VectorType &x_out, VectorType &rhs_in, int level, bool doperm);
      
      /**
       * @brief   Solve starting from a certain level.
       * @details Solve starting from a certain level.
       * @param   [in,out] x_in The initial guess.
       * @param   [in]     rhs_out The right-hand-side.
       * @param   [in]     level The start level.
       * @param   [in]     doperm Do we apply permutation on this level? (Some time the system is already permuted).
       * @return     Return error message.
       */
      int         SolveLevelPslr( VectorType &x_out, VectorType &rhs_in, int level, bool doperm);
      
      /**
       * @brief   Solve with B on a certain level.
       * @details Solve with B on a certain level.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @param   [in]     option 0: solve with B_solver; 1: solve with B_precond; 2: apply smoothing.
       * @param   [in]     level The start level.
       * @return     Return error message.
       */
      int         SolveB( VectorType &x, VectorType &rhs, int option, int level);
      
      /**
       * @brief   Apply the low-rank update on a certain level.
       * @details Apply the low-rank update on a certain level.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @param   [in]     level The start level.
       * @return     Return error message.
       */
      int         SolveApplyLowRankLevel( VectorType &x, VectorType &rhs, int level);
      
      /**
       * @brief   Get the local number of rows on certain level for the low-rank part.
       * @details Get the local number of rows on certain level for the low-rank part.
       * @return     Return the length of the matrix.
       */
      int         GetNumRows(int level);
      
      /**
       * @brief   The matvec function y = G*x = I - Ci*(Ci\x) + Ei*(UBi\(LBi\(Fi*(Ci\x). Note that alpha and beta are untouched.
       * @details The matvec function y = G*x = I - Ci*(Ci\x) + Ei*(UBi\(LBi\(Fi*(Ci\x). Note that alpha and beta are untouched.
       * @param [in]       level The matvec level.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int EBFCMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      
      /**
       * @brief   The matvec function y = G*x = (2EB^{-1}F + EB^{-1}BB^{-1}F)C^{-1}x. Note that alpha and beta are untouched.
       * @details The matvec function y = G*x = (2EB^{-1}F + EB^{-1}BB^{-1}F)C^{-1}x. Note that alpha and beta are untouched.
       * @param [in]       level The matvec level.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int RAPEBFCMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   The matvec function y = G*x = C\S*x-x. Note that alpha and beta are untouched.
       * @details The matvec function y = G*x = C\S*x-x. Note that alpha and beta are untouched.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int SCMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   The matvec function y = x-A*M^{-1}*x. Note that alpha and beta are untouched.
       * @details The matvec function y = x-A*M^{-1}*x. Note that alpha and beta are untouched. \n
       *          We are finding AM^{-1}(I-X)^{-1} = I -> X = I - A^M{-1}. After that, we can apply the preconditioner as M^{-1}(I-X)^{-1}x.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int ACMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   The matvec function y = (EsC^{-1})^m*x. Note that alpha and beta are untouched.
       * @details The matvec function y = (EsC^{-1})^m*x. Note that alpha and beta are untouched.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int PCLRMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   The matvec with the Schur Complement.
       * @details The matvec with the Schur Complement.
       * @param [in]       level The matvec level.
       * @param [in]       option The matvec option. 0: standard; 1: use -C_offd instad of C.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int SchurMatVec( int level, int option, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   The matvec with the Schur Complement via RAP.
       * @details The matvec with the Schur Complement via RAP.
       * @param [in]       level The matvec level.
       * @param [in]       option The matvec option. 0: use B_solve; 1: use B_precond.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int RAPMatVec( int level, int option, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   The matvec with the C on the current level.
       * @details The matvec with the C on the current level.
       * @param [in]       level The matvec level.
       * @param [in]       option The matvec option. 0: standard; 1: use -C_offd instad of C.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int CMatVec( int level, int option, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   The matvec with the B on the current level.
       * @details The matvec with the B on the current level.
       * @param [in]       level The matvec level.
       * @param [in]       option The matvec option. 0: standard; 1: use -C_offd instad of C.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int BMatVec( int level, int option, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
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
       * @brief   Get the total number of nonzeros the ILU.
       * @details Get the total number of nonzeros the ILU.
       * @return     Return the total number of nonzeros the ILU.
       */
      long int    GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
      
      /**
       * @brief   Plot the pattern of the parallel gemslr using gnuplot. Similar to spy in the MATLAB. Function for testing purpose.
       * @details Plot the pattern of the parallel gemslr using gnuplot. Similar to spy in the MATLAB. Function for testing purpose.
       * @param [in]   datafilename The filename of the temp file holding the data.
       * @return       Return error message.
       */
      int         PlotPatternGnuPlot( const char *datafilename);
      
      /* Sets and gets */
      
      /**
       * @brief   Setup with parameter array. This is the helper function to set the local gemslr for B solve.
       * @details Setup with parameter array. This is the helper function to set the local gemslr for B solve.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetLocalGemslr(GemslrClass<CsrMatrixClass<DataType>, SequentialVectorClass<DataType>, DataType> &gemslr)
      {
         
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslr_SetLocalGemslr."));
         
         /* general setup */
         
         /* mute the sub-gemslr print */
         gemslr.SetPrintOption(0);
         
         /* gemslr_setup */
         
         gemslr.SetPartitionOption(this->_gemslr_setups._partition_option_B_setup);
         gemslr.SetPermutationOption(this->_gemslr_setups._perm_option_B_setup);
         gemslr.SetNumLevels(this->_gemslr_setups._nlev_B_setup);
         gemslr.SetNumSubdomains(this->_gemslr_setups._ncomp_B_setup);
         gemslr.SetMinimalNumberSubdomains(this->_gemslr_setups._kmin_B_setup);
         gemslr.SetNumberSubdomainsReduceFactor(this->_gemslr_setups._kfactor_B_setup);
         gemslr.SetSeperatorOption(this->_gemslr_setups._vertexsep_B_setup);
         
         /* turn off local inner iteration for now. Arnoldi reauires fix operation */
         gemslr.SetInnerIterationOption(false);
         gemslr.SetSolveOption(kGemslrLUSolve);
         
         /* level_setup */
         
         gemslr.SetLowRankOptionTopLevel(this->_gemslr_setups._level_setups._lr_option1_B_setup);
         gemslr.SetLowRankOptionOtherLevels(this->_gemslr_setups._level_setups._lr_option2_B_setup);
         gemslr.SetLowRankRandomInitGuess(this->_gemslr_setups._level_setups._lr_rand_init_B_setup);
         gemslr.SetLowRankFactorTopLevel(this->_gemslr_setups._level_setups._lr_rank_factor1_B_setup);
         gemslr.SetLowRankFactorOtherLevels(this->_gemslr_setups._level_setups._lr_rank_factor2_B_setup);
         gemslr.SetLowRankRanksTopLevel(this->_gemslr_setups._level_setups._lr_rank1_B_setup);
         gemslr.SetLowRankRanksOtherLevels(this->_gemslr_setups._level_setups._lr_rank2_B_setup);
         gemslr.SetLowRankArnoldiFactorTopLevel(this->_gemslr_setups._level_setups._lr_arnoldi_factor1_B_setup);
         gemslr.SetLowRankArnoldiFactorOtherLevels(this->_gemslr_setups._level_setups._lr_arnoldi_factor2_B_setup);
         gemslr.SetLowRankMaxNumberIterationsTopLevel(this->_gemslr_setups._level_setups._lr_maxits1_B_setup);
         gemslr.SetLowRankMaxNumberIterationsOtherLevels(this->_gemslr_setups._level_setups._lr_maxits2_B_setup);
         gemslr.SetLowRankThresholdTopLevel(this->_gemslr_setups._level_setups._lr_tol_eig1_B_setup);
         gemslr.SetLowRankThresholdOtherLevels(this->_gemslr_setups._level_setups._lr_tol_eig2_B_setup);
         
         /* currently the inner solve is ILUT only */
         gemslr.SetPreconditionerOptionB(kGemslrBSolveILUT);
         gemslr.SetPreconditionerOptionC(kGemslrCSolveILUT);
         
         /* set ilu options */
         gemslr.SetIluResidualIters(this->_gemslr_setups._level_setups._ilu_residual_iters);
         gemslr.SetIluComplexShift(this->_gemslr_setups._level_setups._ilu_complex_shift);
         gemslr.SetIluDropTolB(this->_gemslr_setups._level_setups._B_ilu_tol_B_setup);
         gemslr.SetIluDropTolC(this->_gemslr_setups._level_setups._C_ilu_tol_B_setup);
         gemslr.SetIluMaxRowNnzB(this->_gemslr_setups._level_setups._B_ilu_max_row_nnz_B_setup);
         gemslr.SetIluMaxRowNnzC(this->_gemslr_setups._level_setups._C_ilu_max_row_nnz_B_setup);
         gemslr.SetIluFillLevelB(this->_gemslr_setups._level_setups._B_ilu_fill_level_B_setup);
         gemslr.SetIluFillLevelC(this->_gemslr_setups._level_setups._C_ilu_fill_level_B_setup);
         gemslr.SetPolyOrder(this->_gemslr_setups._level_setups._B_poly_order_B);
         
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Setup with parameter array. This is the helper function to set the lower level gemslr for explicite Schur solve.
       * @details Setup with parameter array. This is the helper function to set the lower level gemslr for explicite Schur solve.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetLowerGemslr(ParallelGemslrClass<MatrixType, VectorType, DataType> &gemslr)
      {
         
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslr_SetLowerGemslr."));
         
         /* general setup */
         
         /* mute the sub-gemslr print */
         gemslr.SetPrintOption(0);
         
         /* gemslr_setup */
         gemslr.SetPartitionOption(this->_gemslr_setups._partition_option_setup);
         gemslr.SetBPartitionOption(this->_gemslr_setups._partition_option_B_setup);
         gemslr.SetPermutationOption(this->_gemslr_setups._perm_option_setup);
         gemslr.SetBPermutationOption(this->_gemslr_setups._perm_option_B_setup);
         gemslr.SetNumLevels(this->_gemslr_setups._nlev_setup);
         gemslr.SetBNumLevels(this->_gemslr_setups._nlev_B_setup);
         gemslr.SetNumSubdomains(this->_gemslr_setups._ncomp_setup);
         gemslr.SetBNumSubdomains(this->_gemslr_setups._ncomp_B_setup);
         gemslr.SetMinimalNumberSubdomains(this->_gemslr_setups._kmin_setup);
         gemslr.SetBMinimalNumberSubdomains(this->_gemslr_setups._kmin_B_setup);
         gemslr.SetNumberSubdomainsReduceFactor(this->_gemslr_setups._kfactor_setup);
         gemslr.SetBNumberSubdomainsReduceFactor(this->_gemslr_setups._kfactor_B_setup);
         gemslr.SetSeperatorOption(this->_gemslr_setups._vertexsep_setup);
         gemslr.SetBSeperatorOption(this->_gemslr_setups._vertexsep_B_setup);
         
         /* turn off local inner iteration for now. Arnoldi reauires fix operation */
         gemslr.SetInnerIterationOption(false);
         gemslr.SetSolveOption(this->_gemslr_setups._solve_option_setup);
         
         /* turn on global partition */
         gemslr.SetGlobalPartitionOption(true);
         
         /* level_setup */
         /* the low-rank setups was changed, since the top level is the IO level */
         gemslr.SetLowRankOptionTopLevel(this->_gemslr_setups._level_setups._lr_option2_setup);
         gemslr.SetBLowRankOptionTopLevel(this->_gemslr_setups._level_setups._lr_option1_B_setup);
         gemslr.SetLowRankOptionOtherLevels(this->_gemslr_setups._level_setups._lr_option2_setup);
         gemslr.SetBLowRankOptionOtherLevels(this->_gemslr_setups._level_setups._lr_option2_B_setup);
         gemslr.SetLowRankRandomInitGuess(this->_gemslr_setups._level_setups._lr_rand_init_setup);
         gemslr.SetBLowRankRandomInitGuess(this->_gemslr_setups._level_setups._lr_rand_init_B_setup);
         gemslr.SetLowRankFactorTopLevel(this->_gemslr_setups._level_setups._lr_rank_factor2_setup);
         gemslr.SetBLowRankFactorTopLevel(this->_gemslr_setups._level_setups._lr_rank_factor1_B_setup);
         gemslr.SetLowRankFactorOtherLevels(this->_gemslr_setups._level_setups._lr_rank_factor2_setup);
         gemslr.SetBLowRankFactorOtherLevels(this->_gemslr_setups._level_setups._lr_rank_factor2_B_setup);
         gemslr.SetLowRankRanksTopLevel(this->_gemslr_setups._level_setups._lr_rank2_setup);
         gemslr.SetBLowRankRanksTopLevel(this->_gemslr_setups._level_setups._lr_rank1_B_setup);
         gemslr.SetLowRankRanksOtherLevels(this->_gemslr_setups._level_setups._lr_rank2_setup);
         gemslr.SetBLowRankRanksOtherLevels(this->_gemslr_setups._level_setups._lr_rank2_B_setup);
         gemslr.SetLowRankArnoldiFactorTopLevel(this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup);
         gemslr.SetBLowRankArnoldiFactorTopLevel(this->_gemslr_setups._level_setups._lr_arnoldi_factor1_B_setup);
         gemslr.SetLowRankArnoldiFactorOtherLevels(this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup);
         gemslr.SetBLowRankArnoldiFactorOtherLevels(this->_gemslr_setups._level_setups._lr_arnoldi_factor2_B_setup);
         gemslr.SetLowRankMaxNumberIterationsTopLevel(this->_gemslr_setups._level_setups._lr_maxits2_setup);
         gemslr.SetBLowRankMaxNumberIterationsTopLevel(this->_gemslr_setups._level_setups._lr_maxits1_B_setup);
         gemslr.SetLowRankMaxNumberIterationsOtherLevels(this->_gemslr_setups._level_setups._lr_maxits2_setup);
         gemslr.SetBLowRankMaxNumberIterationsOtherLevels(this->_gemslr_setups._level_setups._lr_maxits2_B_setup);
         gemslr.SetLowRankThresholdTopLevel(this->_gemslr_setups._level_setups._lr_tol_eig2_setup);
         gemslr.SetBLowRankThresholdTopLevel(this->_gemslr_setups._level_setups._lr_tol_eig1_B_setup);
         gemslr.SetLowRankThresholdOtherLevels(this->_gemslr_setups._level_setups._lr_tol_eig2_setup);
         gemslr.SetBLowRankThresholdOtherLevels(this->_gemslr_setups._level_setups._lr_tol_eig2_B_setup);
         
         /* set the solver into multilevel */
         gemslr.SetGlobalPrecondOption(kGemslrGlobalPrecondGeMSLR);
         gemslr.SetPreconditionerOption1(this->_gemslr_setups._level_setups._B_solve_option1);
         gemslr.SetPreconditionerOption1Levels(this->_gemslr_setups._level_setups._B_solve_option1_levels);
         gemslr.SetPreconditionerOption2(this->_gemslr_setups._level_setups._B_solve_option2);
         gemslr.SetPreconditionerOptionC(this->_gemslr_setups._level_setups._C_solve_option);
         gemslr.SetSmoothOptionB(this->_gemslr_setups._level_setups._B_smooth_option1);
         
         /* set ilu options */
         gemslr.SetIluResidualIters(this->_gemslr_setups._level_setups._ilu_residual_iters);
         gemslr.SetIluComplexShift(this->_gemslr_setups._level_setups._ilu_complex_shift);
         gemslr.SetIluDropTolB(this->_gemslr_setups._level_setups._B_ilu_tol_setup);
         gemslr.SetBIluDropTolB(this->_gemslr_setups._level_setups._B_ilu_tol_B_setup);
         gemslr.SetIluDropTolC(this->_gemslr_setups._level_setups._C_ilu_tol_setup);
         gemslr.SetBIluDropTolC(this->_gemslr_setups._level_setups._C_ilu_tol_B_setup);
         gemslr.SetIluMaxRowNnzB(this->_gemslr_setups._level_setups._B_ilu_max_row_nnz_setup);
         gemslr.SetBIluMaxRowNnzB(this->_gemslr_setups._level_setups._B_ilu_max_row_nnz_B_setup);
         gemslr.SetIluMaxRowNnzC(this->_gemslr_setups._level_setups._C_ilu_max_row_nnz_setup);
         gemslr.SetBIluMaxRowNnzC(this->_gemslr_setups._level_setups._C_ilu_max_row_nnz_B_setup);
         gemslr.SetIluFillLevelB(this->_gemslr_setups._level_setups._B_ilu_fill_level_setup);
         gemslr.SetBIluFillLevelB(this->_gemslr_setups._level_setups._B_ilu_fill_level_B_setup);
         gemslr.SetIluFillLevelC(this->_gemslr_setups._level_setups._C_ilu_fill_level_setup);
         gemslr.SetBIluFillLevelC(this->_gemslr_setups._level_setups._C_ilu_fill_level_B_setup);
         gemslr.SetPolyOrder(this->_gemslr_setups._level_setups._B_poly_order);
         gemslr.SetBPolyOrder(this->_gemslr_setups._level_setups._B_poly_order_B);
         
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Setup with parameter array.
       * @details Setup with parameter array.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetWithParameterArray(double *params)
      {
         
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslr_SetWithParameterArray."));
         
         /* general setup */
         
         //pargemslr_global::_metis_refine = params[PARGEMSLR_IO_PREPOSS_METIS_REFINE];
         this->_print_option = params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL];
         
         /* gemslr_setup */
         
         this->_gemslr_setups._partition_option_setup          = params[PARGEMSLR_IO_PREPOSS_PARTITION_GLOBAL];
         this->_gemslr_setups._partition_option_B_setup        = params[PARGEMSLR_IO_PREPOSS_PARTITION_LOCAL];
         this->_gemslr_setups._perm_option_setup               = params[PARGEMSLR_IO_ILU_PERM_OPTION_GLOBAL];
         this->_gemslr_setups._perm_option_B_setup             = params[PARGEMSLR_IO_ILU_PERM_OPTION_LOCAL];
         this->_gemslr_setups._nlev_setup                      = params[PARGEMSLR_IO_PREPOSS_NLEV_GLOBAL];
         this->_gemslr_setups._nlev_B_setup                    = params[PARGEMSLR_IO_PREPOSS_NLEV_LOCAL];
         this->_gemslr_setups._ncomp_setup                     = params[PARGEMSLR_IO_PREPOSS_NCOMP_GLOBAL];
         this->_gemslr_setups._ncomp_B_setup                   = params[PARGEMSLR_IO_PREPOSS_NCOMP_LOCAL];
         this->_gemslr_setups._kmin_setup                      = params[PARGEMSLR_IO_PREPOSS_KMIN_GLOBAL];
         this->_gemslr_setups._kmin_B_setup                    = params[PARGEMSLR_IO_PREPOSS_KMIN_LOCAL];
         this->_gemslr_setups._kfactor_setup                   = params[PARGEMSLR_IO_PREPOSS_KFACTOR_GLOBAL];
         this->_gemslr_setups._kfactor_B_setup                 = params[PARGEMSLR_IO_PREPOSS_KFACTOR_LOCAL];
         this->_gemslr_setups._vertexsep_setup                 = params[PARGEMSLR_IO_PREPOSS_VTXSEP_GLOBAL] != 0;
         this->_gemslr_setups._vertexsep_B_setup               = params[PARGEMSLR_IO_PREPOSS_VTXSEP_LOCAL] != 0;
         this->_gemslr_setups._global_partition_setup          = params[PARGEMSLR_IO_PREPOSS_GLOBAL_PARTITION] != 0;
         
         this->_gemslr_setups._enable_inner_iters_setup        = params[PARGEMSLR_IO_SCHUR_ENABLE] != 0;
         this->_gemslr_setups._inner_iters_tol_setup           = params[PARGEMSLR_IO_SCHUR_ITER_TOL];
         this->_gemslr_setups._inner_iters_maxits_setup        = params[PARGEMSLR_IO_SCHUR_MAXITS];
         this->_gemslr_setups._solve_option_setup              = params[PARGEMSLR_IO_ADVANCED_GLOBAL_SOLVE];
         
         this->_gemslr_setups._diag_shift_milu                 = params[PARGEMSLR_IO_ADVANCED_DIAG_SHIFT_MODIFIED];
         
         /* level_setup */
         
         this->_gemslr_setups._level_setups._lr_option1_setup                = params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_option2_setup                = params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_optionA_setup                = params[PARGEMSLR_IO_LR_ARNOLDI_OPTIONA];
         this->_gemslr_setups._level_setups._lr_rand_init_setup              = params[PARGEMSLR_IO_LR_RAND_INIT_GUESS] != 0.0;
         this->_gemslr_setups._level_setups._lr_rank_factor1_setup           = params[PARGEMSLR_IO_LR_RANK_FACTOR1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_rank_factor2_setup           = params[PARGEMSLR_IO_LR_RANK_FACTOR2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_rank_factorA_setup           = params[PARGEMSLR_IO_LR_RANK_FACTORA];
         this->_gemslr_setups._level_setups._lr_rank1_setup                  = params[PARGEMSLR_IO_LR_RANK1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_rank2_setup                  = params[PARGEMSLR_IO_LR_RANK2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_rankA_setup                  = params[PARGEMSLR_IO_LR_RANK_A];
         this->_gemslr_setups._level_setups._lr_arnoldi_factor1_setup        = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup        = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_arnoldi_factorA_setup        = params[PARGEMSLR_IO_LR_ARNOLDI_FACTORA];
         this->_gemslr_setups._level_setups._lr_maxits1_setup                = params[PARGEMSLR_IO_LR_MAXITS1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_maxits2_setup                = params[PARGEMSLR_IO_LR_MAXITS2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_maxitsA_setup                = params[PARGEMSLR_IO_LR_MAXITSA];
         this->_gemslr_setups._level_setups._lr_tol_eig1_setup               = params[PARGEMSLR_IO_LR_TOL_EIG1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_tol_eig2_setup               = params[PARGEMSLR_IO_LR_TOL_EIG2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_tol_eigA_setup               = params[PARGEMSLR_IO_LR_TOL_EIGA];
         
         this->_global_precond_option                                        = params[PARGEMSLR_IO_PRECOND_GLOBAL_PRECOND];
         this->_gemslr_setups._level_setups._B_solve_option1                 = params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1];
         this->_gemslr_setups._level_setups._B_solve_option1_levels          = params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1_LEVEL];
         this->_gemslr_setups._level_setups._B_solve_option2                 = params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND2];
         this->_gemslr_setups._level_setups._C_solve_option                  = params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3];
         this->_gemslr_setups._level_setups._B_smooth_option1                = params[PARGEMSLR_IO_PRECOND_LOCAL_SMOOTHER1];
         
         if(this->_gemslr_setups._level_setups._C_solve_option == kGemslrCSolveBJILUK2)
         {
            this->_gemslr_setups._level_setups._C_solve_option = kGemslrCSolveBJILUK;
            this->_gemslr_setups._level_setups._C_lr_pslr = true;
         }
         if(this->_gemslr_setups._level_setups._C_solve_option == kGemslrCSolveBJILUT2)
         {
            this->_gemslr_setups._level_setups._C_solve_option = kGemslrCSolveBJILUT;
            this->_gemslr_setups._level_setups._C_lr_pslr = true;
         }
         
         this->_gemslr_setups._level_setups._B_ilu_tol_setup                 = params[PARGEMSLR_IO_ILU_DROPTOL_B_GLOBAL];
         this->_gemslr_setups._level_setups._C_ilu_tol_setup                 = params[PARGEMSLR_IO_ILU_DROPTOL_C_GLOBAL];
         this->_gemslr_setups._level_setups._EF_ilu_tol_setup                = params[PARGEMSLR_IO_ILU_DROPTOL_EF_GLOBAL];
         this->_gemslr_setups._level_setups._S_ilu_tol_setup                 = params[PARGEMSLR_IO_ILU_DROPTOL_S_GLOBAL];
         this->_gemslr_setups._level_setups._B_ilu_max_row_nnz_setup         = params[PARGEMSLR_IO_ILU_ROWNNZ_B_GLOBAL];
         this->_gemslr_setups._level_setups._C_ilu_max_row_nnz_setup         = params[PARGEMSLR_IO_ILU_ROWNNZ_C_GLOBAL];
         this->_gemslr_setups._level_setups._S_ilu_max_row_nnz_setup         = params[PARGEMSLR_IO_ILU_ROWNNZ_S_GLOBAL];
         this->_gemslr_setups._level_setups._B_ilu_fill_level_setup          = params[PARGEMSLR_IO_ILU_LFIL_B_GLOBAL];
         this->_gemslr_setups._level_setups._C_ilu_fill_level_setup          = params[PARGEMSLR_IO_ILU_LFIL_C_GLOBAL];
         this->_gemslr_setups._level_setups._B_poly_order                    = params[PARGEMSLR_IO_POLY_ORDER];
         
         this->_gemslr_setups._level_setups._ilu_complex_shift               = params[PARGEMSLR_IO_ADVANCED_USE_COMPLEX_SHIFT] != 0.0;
         this->_gemslr_setups._level_setups._ilu_residual_iters              = params[PARGEMSLR_IO_ADVANCED_RESIDUAL_ITERS];
         
         pargemslr_global::_gram_schmidt                                     = params[PARGEMSLR_IO_ADVANCED_GRAM_SCHMIDT];
         
         this->_gemslr_setups._level_setups._lr_option1_B_setup              = params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_LOCAL];
         this->_gemslr_setups._level_setups._lr_option2_B_setup              = params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_LOCAL];
         this->_gemslr_setups._level_setups._lr_rand_init_B_setup            = params[PARGEMSLR_IO_LR_RAND_INIT_GUESS] != 0.0;
         this->_gemslr_setups._level_setups._lr_rank_factor1_B_setup         = params[PARGEMSLR_IO_LR_RANK_FACTOR1_LOCAL];
         this->_gemslr_setups._level_setups._lr_rank_factor2_B_setup         = params[PARGEMSLR_IO_LR_RANK_FACTOR2_LOCAL];
         this->_gemslr_setups._level_setups._lr_rank1_B_setup                = params[PARGEMSLR_IO_LR_RANK1_LOCAL];
         this->_gemslr_setups._level_setups._lr_rank2_B_setup                = params[PARGEMSLR_IO_LR_RANK2_LOCAL];
         this->_gemslr_setups._level_setups._lr_arnoldi_factor1_B_setup      = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_LOCAL];
         this->_gemslr_setups._level_setups._lr_arnoldi_factor2_B_setup      = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_LOCAL];
         this->_gemslr_setups._level_setups._lr_maxits1_B_setup              = params[PARGEMSLR_IO_LR_MAXITS1_LOCAL];
         this->_gemslr_setups._level_setups._lr_maxits2_B_setup              = params[PARGEMSLR_IO_LR_MAXITS2_LOCAL];
         this->_gemslr_setups._level_setups._lr_tol_eig1_B_setup             = params[PARGEMSLR_IO_LR_TOL_EIG1_LOCAL];
         this->_gemslr_setups._level_setups._lr_tol_eig2_B_setup             = params[PARGEMSLR_IO_LR_TOL_EIG2_LOCAL];
         
         this->_gemslr_setups._level_setups._B_ilu_tol_B_setup               = params[PARGEMSLR_IO_ILU_DROPTOL_B_LOCAL];
         this->_gemslr_setups._level_setups._C_ilu_tol_B_setup               = params[PARGEMSLR_IO_ILU_DROPTOL_C_LOCAL];
         this->_gemslr_setups._level_setups._B_ilu_max_row_nnz_B_setup       = params[PARGEMSLR_IO_ILU_ROWNNZ_B_LOCAL];
         this->_gemslr_setups._level_setups._C_ilu_max_row_nnz_B_setup       = params[PARGEMSLR_IO_ILU_ROWNNZ_C_LOCAL];
         this->_gemslr_setups._level_setups._B_ilu_fill_level_B_setup        = params[PARGEMSLR_IO_ILU_LFIL_B_LOCAL];
         this->_gemslr_setups._level_setups._C_ilu_fill_level_B_setup        = params[PARGEMSLR_IO_ILU_LFIL_C_LOCAL];
         this->_gemslr_setups._level_setups._B_poly_order_B                  = params[PARGEMSLR_IO_POLY_ORDER];
         
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the global precond option. 0: BJ; 1: ISCHUR; 2: ESCHUR; 3: MLEV.
       * @details Set the global precond option. 0: BJ; 1: ISCHUR; 2: ESCHUR; 3: MLEV.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetGlobalPrecondOption(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 3);
         this->_global_precond_option = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the global partition option. 0: ND; 1: RKway.
       * @details Set the global partition option. 0: ND; 1: RKway.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetPartitionOption(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 1);
         this->_gemslr_setups._partition_option_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part partition option. 0: ND; 1: RKway.
       * @details Set the B part partition option. 0: ND; 1: RKway.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBPartitionOption(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 1);
         this->_gemslr_setups._partition_option_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the global permutation option. 0: No; 1: RCM; 2: AMD.
       * @details Set the global permutation option. 0: No; 1: RCM; 2: AMD.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetPermutationOption(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._perm_option_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part permutation option. 0: No; 1: RCM; 2: AMD.
       * @details Set the B part permutation option. 0: No; 1: RCM; 2: AMD.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBPermutationOption(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._perm_option_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the global target number of levels of GeMSLR.
       * @details Set the global target number of levels of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetNumLevels(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._nlev_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part target number of levels of GeMSLR.
       * @details Set the B part target number of levels of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBNumLevels(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._nlev_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the global target number of subdomains on each level of GeMSLR.
       * @details Set the global target number of subdomains on each level of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetNumSubdomains(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._ncomp_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part target number of subdomains on each level of GeMSLR.
       * @details Set the B part target number of subdomains on each level of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBNumSubdomains(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._ncomp_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the global minimal number of subdomains on each level of GeMSLR.
       * @details Set the global minimal number of subdomains on each level of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetMinimalNumberSubdomains(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._kmin_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part minimal number of subdomains on each level of GeMSLR.
       * @details Set the B part minimal number of subdomains on each level of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBMinimalNumberSubdomains(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._kmin_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the global reduce factor of subdomains on each level of GeMSLR.
       * @details Set the global reduce factor of subdomains on each level of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetNumberSubdomainsReduceFactor(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._kfactor_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part reduce factor of subdomains on each level of GeMSLR.
       * @details Set the B part reduce factor of subdomains on each level of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBNumberSubdomainsReduceFactor(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._kfactor_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the global separator option of GeMSLR.
       * @details Set the global separator option of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetSeperatorOption(bool option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._vertexsep_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part separator option of GeMSLR.
       * @details Set the B part separator option of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBSeperatorOption(bool option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._vertexsep_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the global partition option of GeMSLR.
       * @details Set the global partition option of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetGlobalPartitionOption(bool option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._global_partition_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the inner iteration option of GeMSLR.
       * @details Set the inner iteration option of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetInnerIterationOption(bool option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._enable_inner_iters_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the stop threshold of inner iteration of GeMSLR.
       * @details Set the stop threshold of inner iteration of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetInnerIterationThreshold(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._inner_iters_tol_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the max number of iterations of inner iteration of GeMSLR.
       * @details Set the max number of iterations of inner iteration of GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetInnerIterationMaxNumberIterations(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._inner_iters_maxits_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the solve option. 0: additive LU solve; 1: additive L solve; 2: multi-solve.
       * @details Set the solve option. 0: additive LU solve; 1: additive L solve; 2: multi-solve.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetSolveOption(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._solve_option_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the low-rank option on the top level. 0: Standard. 1: Thick-restart.
       * @details Set the low-rank option on the top level. 0: Standard. 1: Thick-restart.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetLowRankOptionTopLevel(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._level_setups._lr_option1_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the low-rank option on other levels. 0: Standard. 1: Thick-restart.
       * @details Set the low-rank option on other levels. 0: Standard. 1: Thick-restart.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetLowRankOptionOtherLevels(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._level_setups._lr_option2_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the low-rank option on A. 0: Standard. 1: Thick-restart.
       * @details Set the low-rank option on A. 0: Standard. 1: Thick-restart.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetLowRankOptionA(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._level_setups._lr_optionA_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part low-rank option on the top level. 0: Standard. 1: Thick-restart.
       * @details Set the B part low-rank option on the top level. 0: Standard. 1: Thick-restart.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBLowRankOptionTopLevel(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._level_setups._lr_option1_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part low-rank option on other levels. 0: Standard. 1: Thick-restart.
       * @details Set the B part low-rank option on other levels. 0: Standard. 1: Thick-restart.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBLowRankOptionOtherLevels(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._level_setups._lr_option2_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set if we use random initial guess for Arnoldi. Otherwise we use 1 as initial guess.
       * @details Set if we use random initial guess for Arnoldi. Otherwise we use 1 as initial guess.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetLowRankRandomInitGuess(bool option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._lr_rand_init_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set if we use random initial guess for Arnoldi in the B part. Otherwise we use 1 as initial guess.
       * @details Set if we use random initial guess for Arnoldi in the B part. Otherwise we use 1 as initial guess.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBLowRankRandomInitGuess(bool option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._lr_rand_init_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the low-rank factor on the top level. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @details Set the low-rank factor on the top level. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankFactorTopLevel(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < T(1.0));
         this->_gemslr_setups._level_setups._lr_rank_factor1_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the low-rank factor on the other levels. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @details Set the low-rank factor on the other levels. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankFactorOtherLevels(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < T(1.0));
         this->_gemslr_setups._level_setups._lr_rank_factor2_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the low-rank factor on A. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @details Set the low-rank factor on A. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankFactorA(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < T(1.0));
         this->_gemslr_setups._level_setups._lr_rank_factorA_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part low-rank factor on the top level. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @details Set the B part low-rank factor on the top level. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBLowRankFactorTopLevel(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < T(1.0));
         this->_gemslr_setups._level_setups._lr_rank_factor1_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part low-rank factor on the other levels. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @details Set the B part low-rank factor on the other levels. The actuall computed number of low-rank terms is rank * factor >= rank.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBLowRankFactorOtherLevels(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < T(1.0));
         this->_gemslr_setups._level_setups._lr_rank_factor2_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the target number of low-rank terms on the top level.
       * @details Set the target number of low-rank terms on the top level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankRanksTopLevel(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_rank1_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the target number of low-rank terms on the other levels.
       * @details Set the target number of low-rank terms on the other levels.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankRanksOtherLevels(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_rank2_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the target number of low-rank terms for A.
       * @details Set the target number of low-rank terms for A.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankA(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_rankA_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part target number of low-rank terms on the top level.
       * @details Set the B part target number of low-rank terms on the top level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBLowRankRanksTopLevel(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_rank1_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the B part target number of low-rank terms on the other levels.
       * @details Set the B part target number of low-rank terms on the other levels.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBLowRankRanksOtherLevels(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_rank2_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the Arnoldi factor on the top level. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @details Set the Arnoldi factor on the top level. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankArnoldiFactorTopLevel(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < T(1.0));
         this->_gemslr_setups._level_setups._lr_arnoldi_factor1_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the Arnoldi factor on other levels. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @details Set the Arnoldi factor on other levels. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankArnoldiFactorOtherLevels(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the Arnoldi factor on A. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @details Set the Arnoldi factor on A. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankArnoldiFactorA(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_arnoldi_factorA_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the B part Arnoldi factor on the top level. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @details Set the B part Arnoldi factor on the top level. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBLowRankArnoldiFactorTopLevel(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < T(1.0));
         this->_gemslr_setups._level_setups._lr_arnoldi_factor1_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the B part Arnoldi factor on other levels. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @details Set the B part Arnoldi factor on other levels. m steps for arnoldi is rank * rank_factor * arnoldi_factor
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBLowRankArnoldiFactorOtherLevels(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < T(1.0));
         this->_gemslr_setups._level_setups._lr_arnoldi_factor2_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set max restarts of thick-restart Arnoldi on the top level.
       * @details Set max restarts of thick-restart Arnoldi on the top level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetLowRankMaxNumberIterationsTopLevel(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_maxits1_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set max restarts of thick-restart Arnoldi on other levels.
       * @details Set max restarts of thick-restart Arnoldi on other levels.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetLowRankMaxNumberIterationsOtherLevels(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_maxits2_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set max restarts of thick-restart Arnoldi on A.
       * @details Set max restarts of thick-restart Arnoldi on A.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetLowRankMaxNumberIterationsA(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_maxitsA_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set B part max restarts of thick-restart Arnoldi on the top level.
       * @details Set B part max restarts of thick-restart Arnoldi on the top level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBLowRankMaxNumberIterationsTopLevel(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_maxits1_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set B part max restarts of thick-restart Arnoldi on other levels.
       * @details Set B part max restarts of thick-restart Arnoldi on other levels.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBLowRankMaxNumberIterationsOtherLevels(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_maxits2_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set max restarts of thick-restart Arnoldi on the top level.
       * @details Set max restarts of thick-restart Arnoldi on the top level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankThresholdTopLevel(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._lr_tol_eig1_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set max restarts of thick-restart Arnoldi on other levels.
       * @details Set max restarts of thick-restart Arnoldi on other levels.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankThresholdOtherLevels(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._lr_tol_eig2_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set max restarts of thick-restart Arnoldi on A.
       * @details Set max restarts of thick-restart Arnoldi on A.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetLowRankThresholdA(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._lr_tol_eigA_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set B part max restarts of thick-restart Arnoldi on the top level.
       * @details Set B part max restarts of thick-restart Arnoldi on the top level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBLowRankThresholdTopLevel(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._lr_tol_eig1_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set B part max restarts of thick-restart Arnoldi on other levels.
       * @details Set B part max restarts of thick-restart Arnoldi on other levels.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBLowRankThresholdOtherLevels(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._lr_tol_eig2_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the threshold for ILUT of the B part.
       * @details Set the threshold for ILUT of the B part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetIluDropTolB(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._B_ilu_tol_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the threshold for ILUT of the last level.
       * @details Set the threshold for ILUT of the last level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetIluDropTolC(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._C_ilu_tol_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the threshold for ILUT of the S part.
       * @details Set the threshold for ILUT of the S part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetIluDropTolS(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._S_ilu_tol_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the threshold for ILUT of the EF part.
       * @details Set the threshold for ILUT of the EF part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetIluDropTolEF(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._EF_ilu_tol_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the recursive GeMSLR B part threshold for ILUT of the global B part.
       * @details Set the recursive GeMSLR B part threshold for ILUT of the global B part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBIluDropTolB(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._B_ilu_tol_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the recursive GeMSLR last level threshold for ILUT of the global B part.
       * @details Set the recursive GeMSLR last level threshold for ILUT of the global B part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      template <typename T>
      int         SetBIluDropTolC(T option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._C_ilu_tol_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the maxinum number of nonzeros for ILUT of the B part.
       * @details Set the maxinum number of nonzeros for ILUT of the B part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetIluMaxRowNnzB(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._B_ilu_max_row_nnz_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the maxinum number of nonzeros ILUT of the last level.
       * @details Set the maxinum number of nonzeros ILUT of the last level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetIluMaxRowNnzC(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._C_ilu_max_row_nnz_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the maxinum number of nonzeros ILUT of the S part.
       * @details Set the maxinum number of nonzeros ILUT of the S part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetIluMaxRowNnzS(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._S_ilu_max_row_nnz_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the recursive GeMSLR B part maxinum number of nonzeros ILUT.
       * @details Set the recursive GeMSLR B part maxinum number of nonzeros ILUT.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBIluMaxRowNnzB(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._B_ilu_max_row_nnz_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the recursive GeMSLR last level maxinum number of nonzeros ILUT.
       * @details Set the recursive GeMSLR last level maxinum number of nonzeros ILUT.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBIluMaxRowNnzC(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._C_ilu_max_row_nnz_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the fill level for ILUK of the B part.
       * @details Set the fill level for ILUK of the B part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetIluFillLevelB(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._B_ilu_fill_level_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the fill level for ILUK of the last level.
       * @details Set the fill level for ILUK of the last level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetIluFillLevelC(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._C_ilu_fill_level_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set the fill level for ILUK of the B part.
       * @details Set the fill level for ILUK of the B part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBIluFillLevelB(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._B_ilu_fill_level_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }

      /** 
       * @brief   Set the fill level for ILUK of the last level.
       * @details Set the fill level for ILUK of the last level.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBIluFillLevelC(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._C_ilu_fill_level_B_setup = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set poly order for Poly solve of the B part.
       * @details Set poly order for Poly solve of the B part.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetPolyOrder(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._level_setups._B_poly_order = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set poly order for Poly solve of the B part of the recursive GeMSLR.
       * @details Set poly order for Poly solve of the B part of the recursive GeMSLR.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBPolyOrder(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 1);
         this->_gemslr_setups._level_setups._B_poly_order_B = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set top level preconditioner.
       * @details Set top level preconditioner.
       * @note    ILU, GeMSLR, Poly.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetPreconditionerOption1(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._level_setups._B_solve_option1 = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set top level preconditioner apply levels.
       * @details Set top level preconditioner apply levels.
       * @note    ILU, GeMSLR, Poly.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetPreconditionerOption1Levels(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._B_solve_option1_levels = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set mid level preconditioner.
       * @details Set mid level preconditioner.
       * @note    ILU, GeMSLR, Poly.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetPreconditionerOption2(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._level_setups._B_solve_option2 = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set all top levels preconditioner.
       * @details Set all top levels preconditioner.
       * @note    ILU, BJ-ILU. Not used in the sequential version.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetPreconditionerOptionB(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._level_setups._B_solve_option1 = option;
         this->_gemslr_setups._level_setups._B_solve_option1_levels = 0;
         this->_gemslr_setups._level_setups._B_solve_option2 = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set all top levels smoother.
       * @details Set all top levels smoother.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetSmoothOptionB(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 1);
         this->_gemslr_setups._level_setups._B_smooth_option1 = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Set last level preconditioner.
       * @details Set last level preconditioner.
       * @note    ILU, BJ-ILU. Not used in the sequential version.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetPreconditionerOptionC(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 1);
         this->_gemslr_setups._level_setups._C_solve_option = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the number of residual iterations of B solves in the setup phase. Set to <= 1 to turn off.
       * @details Set the number of residual iterations of B solves in the setup phase. Set to <= 1 to turn off.
       * @param   [in]   residual_iters The number of residual iterations of B solves.
       * @return     Return error message.
       */
      int         SetIluResidualIters(int residual_iters)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._ilu_residual_iters = residual_iters;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set if we turn on the complex shift or not (complex version only).
       * @details Set if we turn on the complex shift or not (complex version only).
       * @param   [in]   complex_shift The new drop tol for ILUT.
       * @return     Return error message.
       */
      int         SetIluComplexShift(bool complex_shift)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("ParallelGemslrSets"));
         this->_gemslr_setups._level_setups._ilu_complex_shift = complex_shift;
         return PARGEMSLR_SUCCESS;
      }
      
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
      
      /**
       * @brief      Get the solve phase.
       * @details    Get the solve phase.
       * @return     Return the solve phase.
       */
      int         GetSolvePhase()
      {
         return this->_gemslr_setups._solve_phase_setup;
      }
      
	};
   
   typedef ParallelGemslrClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>               precond_gemslr_csr_par_float;
   typedef ParallelGemslrClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>            precond_gemslr_csr_par_double;
   typedef ParallelGemslrClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>      precond_gemslr_csr_par_complexs;
   typedef ParallelGemslrClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>      precond_gemslr_csr_par_complexd;
   
}

#endif
