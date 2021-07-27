#ifndef PARGEMSLR_GEMSLR_H
#define PARGEMSLR_GEMSLR_H

/**
 * @file gemslr.hpp
 * @brief Sequential GeMSLR preconditioner.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../utils/structs.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../solvers/solver.hpp"
#include "../solvers/fgmres.hpp"
#include "ilu.hpp"

#ifdef PARGEMSLR_CUDA
#include "cusparse.h"
#endif

namespace pargemslr
{
   
   /**
    * @brief   The low-rank option.
    * @details The low-rank option.
    */
   enum GemslrLowRankOptionEnum
   {
      kGemslrLowrankNoRestart,
      kGemslrLowrankThickRestart,
      kGemslrLowrankSubspaceIteration
   };
   
   /**
    * @brief   The partition option.
    * @details The partition option.
    */
   enum GemslrPartitionOptionEnum
   {
      kGemslrPartitionND,
      kGemslrPartitionRKway
   };
   
   /**
    * @brief   The reordering option.
    * @details The reordering option.
    */
   enum GemslrReorderingOptionEnum
   {
      kGemslrReorderingNoExtra,
      kGemslrReorderingIO,
      kGemslrReorderingDDPQ
   };
   
   /**
    * @brief   The B Solve option.
    * @details The B Solve option.
    */
   enum GemslrBSolveOptionEnum
   {
      kGemslrBSolveILUT,
      kGemslrBSolveILUK,
      kGemslrBSolveGemslr,
      kGemslrBSolvePoly
   };
   
   /**
    * @brief   The B Smoothing option.
    * @details The B Smoothing option.
    */
   enum GemslrBSmoothOptionEnum
   {
      kGemslrBSmoothBMILU,
      kGemslrBSmoothBILU,
      kGemslrBSmoothAMILU,
      kGemslrBSmoothAILU
   };
   
   /**
    * @brief   The Smoothing option for the multiplicative solve.
    * @details The Smoothing option for the multiplicative solve.
    */
   enum GemslrSmoothingOptionEnum
   {
      kGemslrSmoothingBilu,
      kGemslrSmoothingAbj,
      kGemslrSmoothingL1Jacobi
   };
   
   /**
    * @brief   The C Solve option.
    * @details The C Solve option. Sequential ILUT, sequential ILUK, BJILUT, BJILUK, BJILU with PSLR.
    */
   enum GemslrCSolveOptionEnum
   {
      kGemslrCSolveILUT,
      kGemslrCSolveILUK,
      kGemslrCSolveBJILUT,
      kGemslrCSolveBJILUK,
      kGemslrCSolveBJILUT2,
      kGemslrCSolveBJILUK2,
      kGemslrCSolveDirect
   };
   
   /**
    * @brief   The solve phase.
    * @details The solve phase.
    */
   enum GemslrSolvePhaseEnum
   {
      kGemslrPhaseSolve,
      kGemslrPhaseSetup
   };
   
   /**
    * @brief   The solve option.
    * @details The solve option. LU solve: both L solve and U solve; U solve: U solve only; \n
    *          Mul solve: the multiplicative solve; Mmul solve: the multiplicative solve with modified ILU.
    */
   enum GemslrSolveOptionEnum
   {
      kGemslrLUSolve,
      kGemslrUSolve,
      kGemslrMulSolve,
      kGemslrMmulSolve
   };
   
   /**
    * @brief   The schur complement option.
    * @details The schur complement option.
    */
   enum GemslrGlobalPrecondOptionEnum
   {
      kGemslrGlobalPrecondBJ,
      kGemslrGlobalPrecondESMSLR,
      kGemslrGlobalPrecondGeMSLR,
      kGemslrGlobalPrecondSchurILU, // Two-level Schur ILU via partial ILU
      kGemslrGlobalPrecondPSLR,
      kGemslrGlobalPrecondPCLR,
      kGemslrGlobalPrecondA // This is a special option, (I-X)^{-1}M^{-1}A = I => X = I - M^{-1}A
   };
   
	/**
    * @brief   Class of matvec EB^{-1}FC^{-1}.
    * @details Class of matvec EB^{-1}FC^{-1}.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class GemslrEBFCMatrixClass
   {
   private:
      
      /* variables */
      
      /**
       * @brief   Vector holding the column pointer data.
       * @details Vector holding the column pointer data.
       */
      int                                                      _level;
      
      /**
       * @brief   Vector holding the column index data.
       * @details Vector holding the column index data.
       */
      GemslrClass<MatrixType, VectorType, DataType>            *_gemslr;
      
   public:
      
      /**
       * @brief   Temp vector for the Arnoldi.
       * @details Temp vector for the Arnoldi.
       */
      VectorType                                               _temp_v;
      
      /**
       * @brief   The constructor of GemslrEBFCMatrixClass.
       * @details The constructor of GemslrEBFCMatrixClass. The default memory location is the host memory.
       */
      GemslrEBFCMatrixClass();
      
      /**
       * @brief   The destructor of GemslrEBFCMatrixClass.
       * @details The destructor of GemslrEBFCMatrixClass. Simply a call to the free function.
       */
      virtual ~GemslrEBFCMatrixClass();
      
      /**
       * @brief   The copy constructor of GemslrEBFCMatrixClass.
       * @details The copy constructor of GemslrEBFCMatrixClass.
       */
      GemslrEBFCMatrixClass(const GemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The move constructor of GemslrEBFCMatrixClass.
       * @details The move constructor of GemslrEBFCMatrixClass.
       */
      GemslrEBFCMatrixClass(GemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   The operator = of GemslrEBFCMatrixClass.
       * @details The operator = of GemslrEBFCMatrixClass.
       */
      GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>& operator=(const GemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The operator = of GemslrEBFCMatrixClass.
       * @details The operator = of GemslrEBFCMatrixClass.
       */
      GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>& operator=(GemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   Set the current matrix to a certain GEMSLR level.
       * @details Set the current matrix to a certain GEMSLR level.
       * @param [in] level The level.
       * @param [in] gemslr The GeMSLR structure.
       * @return     Return error message.
       */
      int      Setup(int level, GemslrClass<MatrixType, VectorType, DataType> &gemslr);
      
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
         comm = *parallel_log::_lcomm;
         np = 1;
         myid = 0;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Get the MPI_comm.
       * @details Get the MPI_comm.
       * @return     Return the MPI_comm.
       */
      MPI_Comm       GetComm() const
      {
         return *parallel_log::_lcomm;
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
   
   typedef GemslrEBFCMatrixClass<CsrMatrixClass<float>, SequentialVectorClass<float>, float>             precond_gemslrebfc_csr_seq_float;
   typedef GemslrEBFCMatrixClass<CsrMatrixClass<double>, SequentialVectorClass<double>, double>          precond_gemslrebfc_csr_seq_double;
   typedef GemslrEBFCMatrixClass<CsrMatrixClass<complexs>, SequentialVectorClass<complexs>, complexs>    precond_gemslrebfc_csr_seq_complexs;
   typedef GemslrEBFCMatrixClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd>    precond_gemslrebfc_csr_seq_complexd;
   
	/**
    * @brief   Class of matvec EB^{-1}FC^{-1}.
    * @details Class of matvec EB^{-1}FC^{-1}.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class GemslrSchurMatrixClass
   {
   private:
      
      /* variables */
      
      /**
       * @brief   Vector holding the column pointer data.
       * @details Vector holding the column pointer data.
       */
      int                                                      _level;
      
      /**
       * @brief   Vector holding the column index data.
       * @details Vector holding the column index data.
       */
      GemslrClass<MatrixType, VectorType, DataType>            *_gemslr;
      
   public:
      
      /**
       * @brief   Temp vector for the Arnoldi.
       * @details Temp vector for the Arnoldi.
       */
      VectorType                                               _temp_v;
      
      /**
       * @brief   The constructor of GemslrEBFCMatrixClass.
       * @details The constructor of GemslrEBFCMatrixClass. The default memory location is the host memory.
       */
      GemslrSchurMatrixClass();
      
      /**
       * @brief   The destructor of GemslrEBFCMatrixClass.
       * @details The destructor of GemslrEBFCMatrixClass. Simply a call to the free function.
       */
      virtual ~GemslrSchurMatrixClass();
      
      /**
       * @brief   The copy constructor of GemslrSchurMatrixClass.
       * @details The copy constructor of GemslrSchurMatrixClass.
       */
      GemslrSchurMatrixClass(const GemslrSchurMatrixClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The move constructor of GemslrSchurMatrixClass.
       * @details The move constructor of GemslrSchurMatrixClass.
       */
      GemslrSchurMatrixClass(GemslrSchurMatrixClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   The operator = of GemslrSchurMatrixClass.
       * @details The operator = of GemslrSchurMatrixClass.
       */
      GemslrSchurMatrixClass<MatrixType, VectorType, DataType>& operator=(const GemslrSchurMatrixClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The operator = of GemslrSchurMatrixClass.
       * @details The operator = of GemslrSchurMatrixClass.
       */
      GemslrSchurMatrixClass<MatrixType, VectorType, DataType>& operator=(GemslrSchurMatrixClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   Set the current matrix to a certain GEMSLR level.
       * @details Set the current matrix to a certain GEMSLR level.
       * @param [in] level The level.
       * @param [in] gemslr The GeMSLR structure.
       * @return     Return error message.
       */
      int      Setup(int level, GemslrClass<MatrixType, VectorType, DataType> &gemslr);
      
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
         comm = *parallel_log::_lcomm;
         np = 1;
         myid = 0;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Get the MPI_comm.
       * @details Get the MPI_comm.
       * @return     Return the MPI_comm.
       */
      MPI_Comm       GetComm() const
      {
         return *parallel_log::_lcomm;
      }
      
   };
   
   typedef GemslrSchurMatrixClass<CsrMatrixClass<float>, SequentialVectorClass<float>, float>             precond_gemslr_schur_seq_float;
   typedef GemslrSchurMatrixClass<CsrMatrixClass<double>, SequentialVectorClass<double>, double>          precond_gemslr_schur_seq_double;
   typedef GemslrSchurMatrixClass<CsrMatrixClass<complexs>, SequentialVectorClass<complexs>, complexs>    precond_gemslr_schur_seq_complexs;
   typedef GemslrSchurMatrixClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd>    precond_gemslr_schur_seq_complexd;
   
	/**
    * @brief   Class of matvec EB^{-1}FC^{-1}.
    * @details Class of matvec EB^{-1}FC^{-1}.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class GemslrSchurSolveClass: public SolverClass<MatrixType, VectorType, DataType>
   {
	public:
      
      /**
       * @brief   The constructor of precondioner class.
       * @details The constructor of precondioner class.
       */
      GemslrSchurSolveClass() : SolverClass<MatrixType, VectorType, DataType>(){}
      
      /**
       * @brief   The copy constructor of GemslrSchurSolveClass class.
       * @details The copy constructor of GemslrSchurSolveClass class.
       */
      GemslrSchurSolveClass(const GemslrSchurSolveClass<MatrixType, VectorType, DataType> &precond) : SolverClass<MatrixType, VectorType, DataType>(precond)
      {
         this->Clear();
      }
      
      /**
       * @brief   The move constructor of GemslrSchurSolveClass class.
       * @details The move constructor of GemslrSchurSolveClass class.
       */
      GemslrSchurSolveClass(GemslrSchurSolveClass<MatrixType, VectorType, DataType> &&precond) : SolverClass<MatrixType, VectorType, DataType>(std::move(precond))
      {
         this->Clear();
      }
      
      /**
       * @brief   The operator = of GemslrSchurSolveClass class.
       * @details The operator = of GemslrSchurSolveClass class.
       */
      GemslrSchurSolveClass<MatrixType, VectorType, DataType>& operator=(const GemslrSchurSolveClass<MatrixType, VectorType, DataType> &precond)
      {
         this->Clear();
         SolverClass<MatrixType, VectorType, DataType>::operator=(precond);
         return *this;
      }
      
      /**
       * @brief   The operator = of GemslrSchurSolveClass class.
       * @details The operator = of GemslrSchurSolveClass class.
       */
      GemslrSchurSolveClass<MatrixType, VectorType, DataType>& operator=(GemslrSchurSolveClass<MatrixType, VectorType, DataType> &&precond)
      {
         this->Clear();
         SolverClass<MatrixType, VectorType, DataType>::operator=(std::move(precond));
         return *this;
      }
      
      /**
       * @brief   Free the current precondioner.
       * @details Free the current precondioner.
       * @return     Return error message.
       */
      virtual int Clear(){SolverClass<MatrixType, VectorType, DataType>::Clear();return PARGEMSLR_SUCCESS;}
      
      /**
       * @brief   The destructor of precondioner class.
       * @details The destructor of precondioner class.
       */
      virtual ~GemslrSchurSolveClass(){this->Clear();}
      
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
   
   typedef GemslrSchurMatrixClass<precond_gemslr_schur_seq_float, SequentialVectorClass<float>, float>             precond_gemslr_schursolve_seq_float;
   typedef GemslrSchurMatrixClass<precond_gemslr_schur_seq_double, SequentialVectorClass<double>, double>          precond_gemslr_schursolve_seq_double;
   typedef GemslrSchurMatrixClass<precond_gemslr_schur_seq_complexs, SequentialVectorClass<complexs>, complexs>    precond_gemslr_schursolve_seq_complexs;
   typedef GemslrSchurMatrixClass<precond_gemslr_schur_seq_complexd, SequentialVectorClass<complexd>, complexd>    precond_gemslr_schursolve_seq_complexd;
   
   /**
    * @brief   The GEMSLR options on each level.
    * @details The GEMSLR options on each level.
    */
   template <typename DataType>
   struct GemslrLevelSetupStruct
   {
   public:
      /* Options */
      
      /* option for this solver */
      
      /** 
       * @brief   Set the Arnoldi iteration option for building the low-rank correction on the top level.
       * @details Set the Arnoldi iteration option for building the low-rank correction on the top level.
       */
      int                                    _lr_option1_setup;
      
      /** 
       * @brief   Set the Arnoldi iteration option for building the low-rank correction on other levels.
       * @details Set the Arnoldi iteration option for building the low-rank correction on other levels.
       */
      int                                    _lr_option2_setup;
      
      /** 
       * @brief   Set the Arnoldi iteration option for building the low-rank correction on A.
       * @details Set the Arnoldi iteration option for building the low-rank correction on A.
       */
      int                                    _lr_optionA_setup;
      
      /** 
       * @brief   Set to true to use random initial guess, otherwise use the unit vector.
       * @details Set to true to use random initial guess, otherwise use the unit vector.
       */
      bool                                   _lr_rand_init_setup;
      
      /** 
       * @brief   The factor for extra low-rank terms on the top level.
       * @details The factor for extra low-rank terms on the top level. Compute k*rank_factor eigenvalues, and pick k "best" out of them.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_rank_factor1_setup;
      
      /** 
       * @brief   The factor for extra low-rank terms on other levels.
       * @details The factor for extra low-rank terms on other levels. Compute k*rank_factor eigenvalues, and pick k "best" out of them.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_rank_factor2_setup;
      
      /** 
       * @brief   The factor for extra low-rank terms on A.
       * @details The factor for extra low-rank terms on A. Compute k*rank_factor eigenvalues, and pick k "best" out of them.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_rank_factorA_setup;
      
      /** 
       * @brief   Set the target number of low-rank terms we keep on the top level.
       * @details Set the target number of low-rank terms we keep on the top level.
       */
      int                                    _lr_rank1_setup;
      
      /** 
       * @brief   Set the target number of low-rank terms we keep on other levels.
       * @details Set the target number of low-rank terms we keep on other levels.
       */
      int                                    _lr_rank2_setup;
      
      /** 
       * @brief   Set the target number of low-rank terms we keep for the A.
       * @details Set the target number of low-rank terms we keep for the A.
       */
      int                                    _lr_rankA_setup;
      
      /** 
       * @brief   Set the maximum steps of the Arnoldi iteration on the top level.
       * @details Set the maximum steps of the Arnoldi iteration on the top level.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_arnoldi_factor1_setup;
      
      /** 
       * @brief   Set the maximum steps of the Arnoldi iteration on other levels.
       * @details Set the maximum steps of the Arnoldi iteration on other levels.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_arnoldi_factor2_setup;
      
      /** 
       * @brief   Set the maximum steps of the Arnoldi iteration on A.
       * @details Set the maximum steps of the Arnoldi iteration on A.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_arnoldi_factorA_setup;
      
      /** 
       * @brief   Set the maximum number of restart if thick-restart Arnoldi is used on the top level.
       * @details Set the maximum number of restart if thick-restart Arnoldi is used on the top level.
       */
      int                                    _lr_maxits1_setup;
      
      /** 
       * @brief   Set the maximum number of restart if thick-restart Arnoldi is used on other levels.
       * @details Set the maximum number of restart if thick-restart Arnoldi is used on other levels.
       */
      int                                    _lr_maxits2_setup;
      
      /** 
       * @brief   Set the maximum number of restart if thick-restart Arnoldi is used on A.
       * @details Set the maximum number of restart if thick-restart Arnoldi is used on A.
       */
      int                                    _lr_maxitsA_setup;
      
      /** 
       * @brief   The thick-restart factor. Portion of thick-resratr length comparing to the entire width.
       * @details The thick-restart factor. Portion of thick-resratr length comparing to the entire width.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_tr_factor_setup;
      
      /** 
       * @brief   The tolorance for eigenvalues on the top level. Eigenvalues with Schur vector smaller than it will be kept.
       * @details The tolorance for eigenvalues on the top level. Eigenvalues with Schur vector smaller than it will be kept. \n 
       *          More convergenced eigenvalues with more number of restarts is not guaranteed.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_tol_eig1_setup;
      
      /** 
       * @brief   The tolorance for eigenvalues on other levels. Eigenvalues with Schur vector smaller than it will be kept.
       * @details The tolorance for eigenvalues on other levels. Eigenvalues with Schur vector smaller than it will be kept. \n 
       *          More convergenced eigenvalues with more number of restarts is not guaranteed.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_tol_eig2_setup;
      
      /** 
       * @brief   The tolorance for eigenvalues on A. Eigenvalues with Schur vector smaller than it will be kept.
       * @details The tolorance for eigenvalues on A. Eigenvalues with Schur vector smaller than it will be kept. \n 
       *          More convergenced eigenvalues with more number of restarts is not guaranteed.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_tol_eigA_setup;
      
      /** 
       * @brief   Choose the B solve option on the first several levels.
       * @details Choose the B solve option on the first several levels.
       */
      int                                    _B_solve_option1;
      
      /** 
       * @brief   From level 0 to this level (exclude) we apply B_solve option 1.
       * @details From level 0 to this level (exclude) we apply B_solve option 1.
       */
      int                                    _B_solve_option1_levels;
      
      /** 
       * @brief   Choose the B solve option before the last level.
       * @details Choose the B solve option before the last level.
       */
      int                                    _B_solve_option2;
      
      /** 
       * @brief   Choose the B smooth option before the last level.
       * @details Choose the B smooth option before the last level.
       */
      int                                    _B_smooth_option1;
      
      /** 
       * @brief   Use power low-rank?
       * @details Use power low-rank?
       */
      bool                                   _C_lr_pslr;
      
      /** 
       * @brief   Choose the C solve option for the last level.
       * @details Choose the C solve option for the last level.
       */
      int                                    _C_solve_option;
      
      /** 
       * @brief   Should we apply residual iteration in the setup phase?
       * @details Should we apply residual iteration in the setup phase?
       */
      int                                    _ilu_residual_iters;
      
      /** 
       * @brief   Should we turn on complex shift?
       * @details Should we turn on complex shift?
       */
      bool                                   _ilu_complex_shift;
      
      /** 
       * @brief   The global droptol for ILUT.
       * @details The global droptol for ILUT.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _B_ilu_tol_setup;
      
      /** 
       * @brief   The droptol for ILUT on the last level.
       * @details The droptol for ILUT on the last level.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _C_ilu_tol_setup;
      
      /** 
       * @brief   The droptol for ILUT on the S part.
       * @details The droptol for ILUT on the S part.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _S_ilu_tol_setup;
      
      /** 
       * @brief   The droptol for ILUT on the EF part.
       * @details The droptol for ILUT on the EF part.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _EF_ilu_tol_setup;
      
      /** 
       * @brief   The global MaxFil for ILUT.
       * @details The global MaxFil for ILUT.
       */
      int                                    _B_ilu_max_row_nnz_setup;
      
      /** 
       * @brief   The global MaxFil for ILUT on the last level.
       * @details The global MaxFil for ILUT on the last level.
       */
      int                                    _C_ilu_max_row_nnz_setup;
      
      /** 
       * @brief   The global MaxFil for ILUT on the S part.
       * @details The global MaxFil for ILUT on the S part.
       */
      int                                    _S_ilu_max_row_nnz_setup;
      
      /** 
       * @brief   The global level of fill for ILUK.
       * @details The global level of fill for ILUK.
       */
      int                                    _B_ilu_fill_level_setup;
      
      /** 
       * @brief   The global level of fill for ILUK on the last level.
       * @details The global level of fill for ILUK on the last level.
       */
      int                                    _C_ilu_fill_level_setup;
      
      /** 
       * @brief   The order for poly for the B blocks.
       * @details The order for poly for the B blocks.
       */
      int                                    _B_poly_order;
      
      /* option for B solve */
      
      /** 
       * @brief   Set the Arnoldi iteration option for building the low-rank correction on the top level.
       * @details Set the Arnoldi iteration option for building the low-rank correction on the top level.
       */
      int                                    _lr_option1_B_setup;
      
      /** 
       * @brief   Set the Arnoldi iteration option for building the low-rank correction on other levels.
       * @details Set the Arnoldi iteration option for building the low-rank correction on other levels.
       */
      int                                    _lr_option2_B_setup;
      
      /** 
       * @brief   Set to true to use random initial guess, otherwise use the unit vector.
       * @details Set to true to use random initial guess, otherwise use the unit vector.
       */
      bool                                   _lr_rand_init_B_setup;
      
      /** 
       * @brief   The factor for extra low-rank terms on the top level.
       * @details The factor for extra low-rank terms on the top level. Compute k*rank_factor eigenvalues, and pick k "best" out of them.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_rank_factor1_B_setup;
      
      /** 
       * @brief   The factor for extra low-rank terms on other levels.
       * @details The factor for extra low-rank terms on other levels. Compute k*rank_factor eigenvalues, and pick k "best" out of them.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_rank_factor2_B_setup;
      
      /** 
       * @brief   Set the target number of low-rank terms we keep on the top level.
       * @details Set the target number of low-rank terms we keep on the top level.
       */
      int                                    _lr_rank1_B_setup;
      
      /** 
       * @brief   Set the target number of low-rank terms we keep on other levels.
       * @details Set the target number of low-rank terms we keep on other levels.
       */
      int                                    _lr_rank2_B_setup;
      
      /** 
       * @brief   Set the maximum steps of the Arnoldi iteration on the top level.
       * @details Set the maximum steps of the Arnoldi iteration on the top level.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_arnoldi_factor1_B_setup;
      
      /** 
       * @brief   Set the maximum steps of the Arnoldi iteration on other levels.
       * @details Set the maximum steps of the Arnoldi iteration on other levels.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_arnoldi_factor2_B_setup;
      
      /** 
       * @brief   Set the maximum number of restart if thick-restart Arnoldi is used on the top level.
       * @details Set the maximum number of restart if thick-restart Arnoldi is used on the top level.
       */
      int                                    _lr_maxits1_B_setup;
      
      /** 
       * @brief   Set the maximum number of restart if thick-restart Arnoldi is used on other levels.
       * @details Set the maximum number of restart if thick-restart Arnoldi is used on other levels.
       */
      int                                    _lr_maxits2_B_setup;
      
      /** 
       * @brief   The thick-restart factor. Portion of thick-resratr length comparing to the entire width.
       * @details The thick-restart factor. Portion of thick-resratr length comparing to the entire width.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_tr_factor_B_setup;
      
      /** 
       * @brief   The tolorance for eigenvalues on the top level. Eigenvalues with Schur vector smaller than it will be kept.
       * @details The tolorance for eigenvalues on the top level. Eigenvalues with Schur vector smaller than it will be kept. \n 
       *          Set to less than TR_TOL_EIG will enable the lock convergenced Schur vector function. Convergence vectors will be locked. \n
       *          If this value is larger than TR_TOL_EIG, more convergenced eigenvalues with more number of restarts is not guaranteed.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_tol_eig1_B_setup;
      
      /** 
       * @brief   The tolorance for eigenvalues on other levels. Eigenvalues with Schur vector smaller than it will be kept.
       * @details The tolorance for eigenvalues on other levels. Eigenvalues with Schur vector smaller than it will be kept. \n 
       *          Set to less than TR_TOL_EIG will enable the lock convergenced Schur vector function. Convergence vectors will be locked. \n
       *          If this value is larger than TR_TOL_EIG, more convergenced eigenvalues with more number of restarts is not guaranteed.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _lr_tol_eig2_B_setup;
      
      /** 
       * @brief   Choose the B solve option on the first several levels.
       * @details Choose the B solve option on the first several levels.
       */
      int                                    _B_solve_option1_B;
      
      /** 
       * @brief   From level 0 to this level (exclude) we apply B_solve option 1.
       * @details From level 0 to this level (exclude) we apply B_solve option 1.
       */
      int                                    _B_solve_option1_levels_B;
      
      /** 
       * @brief   Choose the B solve option before the last level.
       * @details Choose the B solve option before the last level.
       */
      int                                    _B_solve_option2_B;
      
      /** 
       * @brief   Choose the C solve option for the last level.
       * @details Choose the C solve option for the last level.
       */
      int                                    _C_solve_option_B;
      
      /** 
       * @brief   The global droptol for ILUT.
       * @details The global droptol for ILUT.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _B_ilu_tol_B_setup;
      
      /** 
       * @brief   The droptol for ILUT on the last level.
       * @details The droptol for ILUT on the last level.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _C_ilu_tol_B_setup;
      
      /** 
       * @brief   The global MaxFil for ILUT.
       * @details The global MaxFil for ILUT.
       */
      int                                    _B_ilu_max_row_nnz_B_setup;
      
      /** 
       * @brief   The global MaxFil for ILUT on the last level.
       * @details The global MaxFil for ILUT on the last level.
       */
      int                                    _C_ilu_max_row_nnz_B_setup;
      
      /** 
       * @brief   The global level of fill for ILUK.
       * @details The global level of fill for ILUK.
       */
      int                                    _B_ilu_fill_level_B_setup;
      
      /** 
       * @brief   The global level of fill for ILUK on the last level.
       * @details The global level of fill for ILUK on the last level.
       */
      int                                    _C_ilu_fill_level_B_setup;
      
      /** 
       * @brief   The order for poly for the B blocks.
       * @details The order for poly for the B blocks.
       */
      int                                    _B_poly_order_B;
      
      /** 
       * @brief   The constructor, set the default values.
       * @details The constructor, set the default values.
       */
      GemslrLevelSetupStruct();
      
      /** 
       * @brief   Set to default value.
       * @details Set to default value.
       */
      void                                   SetDefault();
      
      /** 
       * @brief   The copy constructor, set the default values.
       * @details The copy constructor, set the default values.
       */
      GemslrLevelSetupStruct(const GemslrLevelSetupStruct<DataType> &str);
      
      /** 
       * @brief   The move constructor, set the default values.
       * @details The move constructor, set the default values.
       */
      GemslrLevelSetupStruct(GemslrLevelSetupStruct<DataType> &&str);
      
      /** 
       * @brief   The operator =.
       * @details The operator =.
       */
      GemslrLevelSetupStruct<DataType>& operator=(const GemslrLevelSetupStruct<DataType> &str);
      
      /** 
       * @brief   The operator =.
       * @details The operator =.
       */
      GemslrLevelSetupStruct<DataType>& operator=(GemslrLevelSetupStruct<DataType> &&str);
      
   };
   
   /**
    * @brief   The GEMSLR information on each level, contains the solver for B and the low-rank information for S.
    * @details The GEMSLR information on each level, contains the solver for B and the low-rank information for S.
    *          VectorType is the type of the vector. DataType is the real data type.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class GemslrLevelClass
   {
   public:
      /* Variables */
      
      /** 
       * @brief   The size of low-rank correction on this level.
       * @details The size of low-rank correction on this level.
       */
      int                                                _lrc;
      
      /** 
       * @brief   The total number of EBFC matvec happens on this level.
       * @details The total number of EBFC matvec happens on this level.
       */
      int                                                _nmvs;
      
      /** 
       * @brief   Number of subdomains on this level.
       * @details Number of subdomains on this level.
       */
      int                                                _ncomps;
      
      /** 
       * @brief   The B matrix on this level. C matrix if this is the last level.
       * @details The B matrix on this level. C matrix if this is the last level.
       */
      std::vector<MatrixType >                           _B_mat_v;
      
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
       * @brief   The C matrix on this level.
       * @details The C matrix on this level.
       */
      MatrixType                                         _C_mat;
      
      /** 
       * @brief   The D matrix on this level.
       * @details The D matrix on this level.
       */
      MatrixType                                         _D_mat;
      
      /** 
       * @brief   The EBFC matrix on this level.
       * @details The EBFC matrix on this level.
       */
      GemslrEBFCMatrixClass<MatrixType, VectorType, DataType> _EBFC;
      
      /** 
       * @brief   The preconditioners for B matrix.
       * @details The preconditioners for B matrix.
       */
      SolverClass<MatrixType, VectorType, DataType>      **_B_precond;
      
      /** 
       * @brief   The solvers for B matrix.
       * @details The solvers for B matrix.
       */
      SolverClass<MatrixType, VectorType, DataType>      **_B_solver;
      
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
       * @brief   Temp vector.
       * @details Temp vector.
       */
      VectorType                                         _y_temp;
      
      /** 
       * @brief   Temp vector.
       * @details Temp vector.
       */
      VectorType                                         _z_temp;
      
      /** 
       * @brief   Temp vector.
       * @details Temp vector.
       */
      VectorType                                         _v_temp;
      
      /** 
       * @brief   Temp vector.
       * @details Temp vector.
       */
      VectorType                                         _w_temp;
      
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
       * @brief   Free the current level structure, set everything to 0.
       * @details Free the current level structure, set everything to 0.
       * @return           return GEMSLR_SUCESS or error information.
       */
      int                                                Clear();
      
      /**
       * @brief   The constructor of GemslrLevelClass, set everything to 0.
       * @details The constructor of GemslrLevelClass, set everything to 0.
       */
      GemslrLevelClass();
      
      /**
       * @brief   The destructor of GemslrLevelClass.
       * @details The destructor of GemslrLevelClass, simply call the free function.
       */
      ~GemslrLevelClass();
      
      /**
       * @brief   The copy constructor of GemslrLevelClass.
       * @details The copy constructor of GemslrLevelClass.
       */
      GemslrLevelClass(const GemslrLevelClass<MatrixType, VectorType, DataType> &str);
      
      /**
       * @brief   The move constructor of GemslrLevelClass.
       * @details The move constructor of GemslrLevelClass.
       */
      GemslrLevelClass(GemslrLevelClass<MatrixType, VectorType, DataType> &&str);
      
      /**
       * @brief   The operator= of GemslrLevelClass.
       * @details The operator= of GemslrLevelClass.
       */
      GemslrLevelClass<MatrixType, VectorType, DataType>& operator=(const GemslrLevelClass<MatrixType, VectorType, DataType> &str);
      
      /**
       * @brief   The operator= of GemslrLevelClass.
       * @details The operator= of GemslrLevelClass.
       */
      GemslrLevelClass<MatrixType, VectorType, DataType>& operator=(GemslrLevelClass<MatrixType, VectorType, DataType> &&str);
      
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
   
   
   typedef GemslrLevelClass<CsrMatrixClass<float>, SequentialVectorClass<float>, float>            precond_gemslrlevel_csr_seq_float;
   typedef GemslrLevelClass<CsrMatrixClass<double>, SequentialVectorClass<double>, double>         precond_gemslrlevel_csr_seq_double;
   typedef GemslrLevelClass<CsrMatrixClass<complexs>, SequentialVectorClass<complexs>, complexs>   precond_gemslrlevel_csr_seq_complexs;
   typedef GemslrLevelClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd>   precond_gemslrlevel_csr_seq_complexd;
   
   /**
    * @brief   The GEMSLR options.
    * @details The GEMSLR options.
    */
   template <typename DataType>
   struct GemslrSetupStruct
   {
   public:
      
      /* Options */
      
      /** 
       * @brief   The global setup of low-rank correction on this level.
       * @details The global setup of low-rank correction on this level.
       */
      GemslrLevelSetupStruct<DataType>   _level_setups;
      
      /** 
       * @brief   Set the solve location.
       * @details Set the solve location.
       */
      int                  _location;
      
      /* option for the main GeMSLR */
      
      /** 
       * @brief   Set the partition option.
       * @details Set the partition option. \n
       *          kGemslrPartitionRKway: Recursive Kway partition.
       */
      int                  _partition_option_setup;
      
      /** 
       * @brief   Set the local B reordering option.
       * @details Set the local B reordering option. \n
       *          kGemslrLocalReorderingNoExtra          Turn off the local reordering.  \n
       *          kGemslrLocalReorderingRCM              Use the local RCM. \n
       */
      int                  _perm_option_setup;
      
      /** 
       * @brief   The total number of levels user wants to have.
       * @details The total number of levels user wants to have.
       */
      int                  _nlev_setup;
      
      /** 
       * @brief   The target number of subdomians on each level.
       * @details The target number of subdomians on each level.
       */
      int                  _ncomp_setup;
      
      /** 
       * @brief   The minimal number of subdomains user wants on each level in the recursive Kway partition.
       * @details The minimal number of subdomains user wants on each level in the recursive Kway partition.
       */
      int                  _kmin_setup;
      
      /** 
       * @brief   In the recursive Kway partition, from the second level, each time the number of 
       *          terget subdomains is divided by _kfactor_setup, until reaching _kmin_setup.
       * @details In the recursive Kway partition, from the second level, each time the number of 
       *          terget subdomains is divided by _kfactor_setup, until reaching _kmin_setup.
       */
      int                  _kfactor_setup;
      
      /** 
       * @brief   Set to true to use vertex seperator. Note that k must be power of 2 for vertex seperator.
       * @details Set to true to use vertex seperator. Note that k must be power of 2 for vertex seperator.
       */
      bool                 _vertexsep_setup;
      
      /** 
       * @brief   Set to true to use vertex seperator. Note that k must be power of 2 for vertex seperator.
       * @details Set to true to use vertex seperator. Note that k must be power of 2 for vertex seperator.
       */
      bool                 _global_partition_setup;
      
      /* option for the B solve GeMSLR */
      
      /** 
       * @brief   Set the partition option. For B part is GeMSLR is used.
       * @details Set the partition option. For B part is GeMSLR is used.
       */
      int                  _partition_option_B_setup;
      
      /** 
       * @brief   Set the local B reordering option. For B part is GeMSLR is used.
       * @details Set the local B reordering option. For B part is GeMSLR is used.
       */
      int                  _perm_option_B_setup;
      
      /** 
       * @brief   The total number of levels user wants to have in the B part if GeMSLR is used.
       * @details The total number of levels user wants to have in the B part if GeMSLR is used.
       */
      int                  _nlev_B_setup;
      
      /** 
       * @brief   The target number of subdomians on each level in the B part if GeMSLR is used.
       * @details The target number of subdomians on each level in the B part if GeMSLR is used.
       */
      int                  _ncomp_B_setup;
      
      /** 
       * @brief   The minimal number of subdomains user wants on each level in the recursive Kway partition in the B part if GeMSLR is used.
       * @details The minimal number of subdomains user wants on each level in the recursive Kway partition in the B part if GeMSLR is used.
       */
      int                  _kmin_B_setup;
      
      /** 
       * @brief   In the recursive Kway partition, from the second level, each time the number of 
       *          terget subdomains is divided by _kfactor_setup, until reaching _kmin_setup. For B part is GeMSLR is used.
       * @details In the recursive Kway partition, from the second level, each time the number of 
       *          terget subdomains is divided by _kfactor_setup, until reaching _kmin_setup. For B part is GeMSLR is used.
       */
      int                  _kfactor_B_setup;
      
      /** 
       * @brief   Set to true to use vertex seperator. Note that k must be power of 2 for vertex seperator. For B part is GeMSLR is used.
       * @details Set to true to use vertex seperator. Note that k must be power of 2 for vertex seperator. For B part is GeMSLR is used.
       */
      bool                 _vertexsep_B_setup;
      
      /* other solve options */
      
      /** 
       * @brief   Set the solve phase of the preconditioner.
       * @details Set the solve phase of the preconditioner.
       */
      int                  _solve_phase_setup;
      
      /** 
       * @brief   Are we only going to put the low-rank part on the device?
       * @details Are we only going to put the low-rank part on the device?
       */
      bool                 _cuda_lowrank_only;
      
      /** 
       * @brief   The level of inner iteration. Solve Sx = b with preconditioned GMRES where GeMSLR is used as a preconditioner.
       * @details The level of inner iteration. Solve Sx = b with preconditioned GMRES where GeMSLR is used as a preconditioner. \n
       *          Set to false to turn off inneritaration. Set to true to turn on...
       */
      bool                 _enable_inner_iters_setup;
      
      /** 
       * @brief   The level of inner iteration. Solve Sx = b with preconditioned GMRES where GeMSLR is used as a preconditioner.
       * @details The level of inner iteration. Solve Sx = b with preconditioned GMRES where GeMSLR is used as a preconditioner. \n
       *          Set to less than 0 to turn off inneritaration. Set to 1 means start from the S of the root level...
       */
      int                  _inner_iters_maxits_setup;
      
      /**
       * @brief   The default convergence_tolorance to lock the eigenvalue.
       * @details The default convergence_tolorance to lock the eigenvalue.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _inner_iters_tol_setup;
      
      /**
       * @brief   The GEMSLR options.
       * @details The GEMSLR options.
       */
      int                  _solve_option_setup;
      
      /**
       * @brief   The GEMSLR B smoothing options.
       * @details The GEMSLR B smoothing options.
       */
      int                  _smoothing_option_setup;
      
      /** 
       * @brief   The diagonal shift for the modified ILU.
       * @details The diagonal shift for the modified ILU.
       */
      DataType             _diag_shift_milu;
      
      /** 
       * @brief   The constructor, set the default values.
       * @details The constructor, set the default values.
       */
      GemslrSetupStruct();
      
      /** 
       * @brief   Set the default values.
       * @details Set the default values.
       */
      void                 SetDefault();
      
      /** 
       * @brief   The copy constructor.
       * @details The copy constructor.
       */
      GemslrSetupStruct(const GemslrSetupStruct<DataType> &str);
      
      /** 
       * @brief   The move constructor.
       * @details The move constructor.
       */
      GemslrSetupStruct(GemslrSetupStruct<DataType> &&str);
      
      /** 
       * @brief   The operator=.
       * @details The operator=.
       */
      GemslrSetupStruct<DataType>& operator=(const GemslrSetupStruct<DataType> &str);
      
      /** 
       * @brief   The operator=.
       * @details The operator=.
       */
      GemslrSetupStruct<DataType>& operator=(GemslrSetupStruct<DataType> &&str);
      
   };
   
   /**
    * @brief   The local real ilu preconditioner, only work for sequential CSR matrix.
    * @details The local real ilu preconditioner, only work for sequential CSR matrix. Matrix type is the type of the matrix, 
    *          VectorType is the type of the vector. DataType is the data type.
    */
   template <class MatrixType, class VectorType, typename DataType>
   class GemslrClass: public SolverClass<MatrixType, VectorType, DataType>
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
      GemslrSchurMatrixClass<MatrixType, VectorType, DataType> _inner_iters_matrix;
      
      /** 
       * @brief   Precond for the inner iteration.
       * @details Precond for the inner iteration.
       */
      GemslrSchurSolveClass<GemslrSchurMatrixClass<MatrixType, VectorType, DataType>,
                              VectorType, DataType> _inner_iters_precond;
      
      /** 
       * @brief   Solver for the inner iteration.
       * @details Solver for the inner iteration.
       */
      FlexGmresClass<GemslrSchurMatrixClass<MatrixType, VectorType, DataType>, 
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
       * @brief   The row permutation vector.
       * @details The row permutation vector.
       */
      IntVectorClass<int>                 _pperm;
      
      /** 
       * @brief   The column permutation vector.
       * @details The column permutation vector.
       * @note    For non-symmetric reordering.
       */
      IntVectorClass<int>                 _qperm;
      
      /**
       * @brief   The location this preconditioner applied to.
       * @details The location this preconditioner applied to.
       */
      int                                 _location;
      
      /** 
       * @brief   The temp vector for permuted x.
       * @details The temp vector for permuted x.
       */
      VectorType                          _x_temp;
      
      /** 
       * @brief   The temp vector for permuted rhs.
       * @details The temp vector for permuted rhs.
       */
      VectorType                          _rhs_temp;
      
      /**
       * @brief   Compute the distance for the selection of eigenvalues.
       * @details Compute the distance for the selection of eigenvalues.
       * @param [in] val the value.
       * @return     Return the distance.
       */
      template <typename T1, typename T2>
      static T1     ComputeDistance(T2 val);
      
      /**
       * @brief   Reorder the result of Arnoldi, R is the first m*m part of H. Real version.
       * @details Reorder the result of Arnoldi, R is the first m*m part of H. Real version.
       * @param [in]       m the size of R and Q.
       * @param [in]       rank the rank we want to pick.
       * @param [in,out]   R the R matrix.
       * @param [in,out]   Q the Q matrix.
       * @return     Return error message.
       */
      template <typename RealDataType>
      int OrdLowRank(int m, int &rank, DenseMatrixClass<RealDataType> &R, DenseMatrixClass<RealDataType> &Q);
      
      /**
       * @brief   Reorder the result of Arnoldi, R is the first m*m part of H. Complex version.
       * @details Reorder the result of Arnoldi, R is the first m*m part of H. Complex version.
       * @param [in]       m the size of R and Q.
       * @param [in]       rank the rank we want to pick.
       * @param [in,out]   R the R matrix.
       * @param [in,out]   Q the Q matrix.
       * @return     Return error message.
       */
      template <typename RealDataType>
      int OrdLowRank(int m, int &rank, DenseMatrixClass<ComplexValueClass<RealDataType> > &R, DenseMatrixClass<ComplexValueClass<RealDataType> > &Q);
      
      /* ------ Setup permutation ------ */
      
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
      int SetupPermutationRKway( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
      
      /**
       * @brief   Setup the Nested Dissection partition of the GeMSLR.
       * @details Setup the Nested Dissection partition of the GeMSLR. This is a symmetric reordering algorithm work on A+A'.
       * @return     Return error message.
       */
      int SetupPermutationND( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
      
      /* ------ Build multilevel structure ------ */
      
      /**
       * @brief   Setup the level structure of the GeMSLR.
       * @details Setup the level structure of the GeMSLR.
       * @return     Return error message.
       */
      int SetupPermutationBuildLevelStructure( vector_int &map_v, vector_int &mapptr_v);
      
      /* ------ Setup B solve ------ */
      
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
       * @brief   Setup the solve of B matrices of the GeMSLR with GeMSLR.
       * @details Setup the solve of B matrices of the GeMSLR with GeMSLR.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupBSolveGemslr( VectorType &x, VectorType &rhs, int level);
      
      /**
       * @brief   Setup the solve of B matrices of the GeMSLR with Poly.
       * @details Setup the solve of B matrices of the GeMSLR with Poly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupBSolvePoly( VectorType &x, VectorType &rhs, int level);
      
      /* ------ Construct Low-rank correction ------ */
      
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
       * @param   [out]  nmvs The number of matrix-vector products.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupLowRankSubspaceIteration( VectorType &x, VectorType &rhs, int &nmvs, int level);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR with arnodi no-restart.
       * @details Setup the low-rank part of the GeMSLR with arnodi no-restart.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [out]  nmvs The number of matrix-vector products.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupLowRankNoRestart( VectorType &x, VectorType &rhs, int &nmvs, int level);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR with arnodi no-restart.
       * @details Setup the low-rank part of the GeMSLR with arnodi no-restart.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [out]  nmvs The number of matrix-vector products.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupLowRankThickRestart( VectorType &x, VectorType &rhs, int &nmvs, int level);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @details Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [out]  nmvs The number of matrix-vector products.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      int SetupLowRankThickRestartNoLock( SequentialVectorClass<DataType> &x, SequentialVectorClass<DataType> &rhs, int &nmvs, int level);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @details Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      template <typename RealDataType>
      int SetupLowRankThickRestartStandard( SequentialVectorClass<RealDataType> &x, SequentialVectorClass<RealDataType> &rhs, int level);
      
      /**
       * @brief   Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @details Setup the low-rank part of the GeMSLR with arnodi thick-restart.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param   [in]   level The target level.
       * @return     Return error message.
       */
      template <typename RealDataType>
      int SetupLowRankThickRestartStandard( SequentialVectorClass<ComplexValueClass<RealDataType> > &x, SequentialVectorClass<ComplexValueClass<RealDataType> > &rhs, int level);
      
      /**
       * @brief   Given the V and H from Arnoldi, compute the final low-rank correction of S^{-1} - C^{-1}.
       * @details Given the V and H from Arnoldi, compute the final low-rank correction of S^{-1} - C^{-1}.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param [in]    V Matrix holding the orthogonal basis genrated by Arnoldi iteration.
       * @param [in]    H The Hessenberg matrix generated by Arnoldi iteration.
       * @param [in]    m The number of finished steps in Arnoldi iteration.
       * @param [in]    rank The number of eigenvectors to keep.
       * @param [in]    level The current level.
       * @return        return # of Schur vectors, if return value < 0 an error occurs                                                                                                                                                    
       */
      int SetupLowRankBuildLowRank( VectorType &x, VectorType &rhs, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, int m, int rank, int level);
      
      /**
       * @brief   Given the V and H from Arnoldi, compute the final low-rank correction of S^{-1} - C^{-1}.
       * @details Given the V and H from Arnoldi, compute the final low-rank correction of S^{-1} - C^{-1}.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @param [in]    V Matrix holding the orthogonal basis genrated by Arnoldi iteration.
       * @param [in]    H 
       * @param [in]    Q H and Q is the Schur Decomposition of the Hessenberg matrix generated by Arnoldi iteration.
       * @param [in]    select Vector of length m. Set to nonzero to select one entry.
       * @param [in]    m The number of finished steps in Arnoldi iteration.
       * @param [in]    rank The number of eigenvectors to keep.
       * @param [in]    level The current level.
       * @return        return # of Schur vectors, if return value < 0 an error occurs                                                                                                                                                    
       */
      int SetupLowRankBuildLowRank( VectorType &x, VectorType &rhs, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, DenseMatrixClass<DataType> &Q, vector_int &select, int m, int rank, int level);
      
	public:
      
      /** 
       * @brief   Vector holding the level struct for all levels starting from the second level.
       *          The size of this vector is equal to _nlev_used, and is the number of true levels.
       * @details Vector holding the level struct for all levels starting from the second level.
       *          The size of this vector is equal to _nlev_used, and is the number of true levels.
       */
      std::vector< GemslrLevelClass< MatrixType, VectorType, DataType> >  _levs_v;
      
      /**
       * @brief   The constructor of precondioner class.
       * @details The constructor of precondioner class.
       */
      GemslrClass();
     
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
      virtual ~GemslrClass();
      
      /**
       * @brief   The copy constructor of GemslrClass.
       * @details The copy constructor of GemslrClass.
       */
      GemslrClass(const GemslrClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The move constructor of GemslrClass.
       * @details The move constructor of GemslrClass.
       */
      GemslrClass(GemslrClass<MatrixType, VectorType, DataType> &&precond);
      
      /**
       * @brief   The operator = of GemslrClass.
       * @details The operator = of GemslrClass.
       */
      GemslrClass<MatrixType, VectorType, DataType>& operator=(const GemslrClass<MatrixType, VectorType, DataType> &precond);
      
      /**
       * @brief   The operator = of GemslrClass.
       * @details The operator = of GemslrClass.
       */
      GemslrClass<MatrixType, VectorType, DataType>& operator=(GemslrClass<MatrixType, VectorType, DataType> &&precond);
      
      /* ------ Wrapper setup function ------ */
      
      /**
       * @brief   Setup the precondioner phase. Will be called by the solver if not called directly.
       * @details Setup the precondioner phase. Will be called by the solver if not called directly.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( VectorType &x, VectorType &rhs);
      
      /* ------ Wrapper of solve phase ------ */
      
      /**
       * @brief   Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @details Solve phase. Call this function after Setup. Solve with cusparse if unified memory/device memory is used.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( VectorType &x, VectorType &rhs);
      
      /* ------ Solve helper functions ------ */
      
      /**
       * @brief   Solve starting from a certain level.
       * @details Solve starting from a certain level.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @param   [in]     level The start level.
       * @return     Return error message.
       */
      int         SolveLevel( VectorType &x, VectorType &rhs, int level);
      
      /**
       * @brief   Solve with B on a certain level.
       * @details Solve with B on a certain level.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @param   [in]     level The start level.
       * @return     Return error message.
       */
      int         SolveB( VectorType &x, VectorType &rhs, int level);
      
      /**
       * @brief   Apply the low-rank update on a certain level.
       * @details Apply the low-rank update on a certain level.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @param   [in]     level The start level.
       * @return     Return error message.
       */
      int         SolveApplyLowRankLevel( VectorType &x, VectorType &rhs, int level);
      
      /* ------ Matvec for Arnoldi ------ */
      
      /**
       * @brief   Get the local number of rows on certain level for the low-rank part.
       * @details Get the local number of rows on certain level for the low-rank part.
       * @return     Return the length of the matrix.
       */
      int         GetNumRows(int level);
      
      /**
       * @brief   The matvec function y = G*x = Ei*(UBi\(LBi\(Fi*(Ci\x)). Note that alpha and beta are untouched.
       * @details The matvec function y = G*x = Ei*(UBi\(LBi\(Fi*(Ci\x)). Note that alpha and beta are untouched.
       * @note    TODO: change VectorType to the base class vector.
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
       * @brief   The matvec function y = alpha*S*x+beta*y where S = Ci - Ei*(UBi\(LBi\(Fi*x).
       * @details The matvec function y = alpha*S*x+beta*y where S = Ci - Ei*(UBi\(LBi\(Fi*x).
       * @note    TODO: change VectorType to the base class vector.
       * @param [in]       level The matvec level.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int SchurMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   The matvec function y = alpha*Ci*x+beta*y.
       * @details The matvec function y = alpha*Ci*x+beta*y.
       * @note    TODO: change VectorType to the base class vector.
       * @param [in]       level The matvec level.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int CMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /* ------ Get Solver Info ------ */
      
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
      
      /* ------ Sets and Gets ------ */
      
      /**
       * @brief   Setup with parameter array. This is the helper function to set the local gemslr for B solve.
       * @details Setup with parameter array. This is the helper function to set the local gemslr for B solve.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetLocalGemslr(GemslrClass<MatrixType, VectorType, DataType> &gemslr)
      {
         
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("Gemslr_SetLocalGemslr."));
         
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
       * @brief   Setup with parameter array.
       * @details Setup with parameter array.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetWithParameterArray(double *params)
      {
         
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("Gemslr_SetWithParameterArray."));
         
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
         
         this->_gemslr_setups._enable_inner_iters_setup        = params[PARGEMSLR_IO_SCHUR_ENABLE] != 0;
         this->_gemslr_setups._inner_iters_tol_setup           = params[PARGEMSLR_IO_SCHUR_ITER_TOL];
         this->_gemslr_setups._inner_iters_maxits_setup        = params[PARGEMSLR_IO_SCHUR_MAXITS];
         this->_gemslr_setups._solve_option_setup              = params[PARGEMSLR_IO_ADVANCED_GLOBAL_SOLVE];
         
         /* level_setup */
         
         this->_gemslr_setups._level_setups._lr_option1_setup                = params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_option2_setup                = params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_rand_init_setup              = params[PARGEMSLR_IO_LR_RAND_INIT_GUESS] != 0.0;
         this->_gemslr_setups._level_setups._lr_rank_factor1_setup           = params[PARGEMSLR_IO_LR_RANK_FACTOR1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_rank_factor2_setup           = params[PARGEMSLR_IO_LR_RANK_FACTOR2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_rank1_setup                  = params[PARGEMSLR_IO_LR_RANK1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_rank2_setup                  = params[PARGEMSLR_IO_LR_RANK2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_arnoldi_factor1_setup        = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup        = params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_maxits1_setup                = params[PARGEMSLR_IO_LR_MAXITS1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_maxits2_setup                = params[PARGEMSLR_IO_LR_MAXITS2_GLOBAL];
         this->_gemslr_setups._level_setups._lr_tol_eig1_setup               = params[PARGEMSLR_IO_LR_TOL_EIG1_GLOBAL];
         this->_gemslr_setups._level_setups._lr_tol_eig2_setup               = params[PARGEMSLR_IO_LR_TOL_EIG2_GLOBAL];
         
         this->_gemslr_setups._level_setups._B_solve_option1                 = params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1];
         this->_gemslr_setups._level_setups._B_solve_option1_levels          = params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1_LEVEL];
         this->_gemslr_setups._level_setups._B_solve_option2                 = params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND2];
         this->_gemslr_setups._level_setups._C_solve_option                  = params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3];
         this->_gemslr_setups._level_setups._B_smooth_option1                = params[PARGEMSLR_IO_PRECOND_LOCAL_SMOOTHER1];
         
         this->_gemslr_setups._level_setups._B_ilu_tol_setup                 = params[PARGEMSLR_IO_ILU_DROPTOL_B_GLOBAL];
         this->_gemslr_setups._level_setups._C_ilu_tol_setup                 = params[PARGEMSLR_IO_ILU_DROPTOL_C_GLOBAL];
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
       * @brief   Set the global partition option. 0: ND; 1: RKway.
       * @details Set the global partition option. 0: ND; 1: RKway.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetPartitionOption(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         this->_gemslr_setups._vertexsep_B_setup = option;
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         PARGEMSLR_FIRM_CHKERR(option > 2);
         this->_gemslr_setups._level_setups._lr_option2_setup = option;
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < T(1.0));
         this->_gemslr_setups._level_setups._lr_rank_factor2_setup = option;
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_rank2_setup = option;
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup = option;
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._lr_maxits2_setup = option;
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         this->_gemslr_setups._level_setups._lr_tol_eig2_setup = option;
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._S_ilu_max_row_nnz_setup = option;
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
       * @brief   Set the recursive GeMSLR B part maxinum number of nonzeros ILUT.
       * @details Set the recursive GeMSLR B part maxinum number of nonzeros ILUT.
       * @param   [in]   option The new option.
       * @return     Return error message.
       */
      int         SetBIluMaxRowNnzB(int option)
      {
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         PARGEMSLR_FIRM_CHKERR(option < 0);
         this->_gemslr_setups._level_setups._C_ilu_max_row_nnz_B_setup = option;
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
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
         PARGEMSLR_FIRM_CHKERR(this->CheckReadySetups("GemslrSets"));
         this->_gemslr_setups._level_setups._ilu_complex_shift = complex_shift;
         return PARGEMSLR_SUCCESS;
      }
      
      /* general preconditioner options */
      
      /**
       * @brief      Set the data location that the preconditioner apply to.
       * @details    Set the data location that the preconditioner apply to.
       * @param [in] location The target solver location.
       * @return     Return error message.
       */
      virtual int SetSolveLocation( const int &location);
      
      /** 
       * @brief   Set to true to only move the low-rank part to device (the triangular solve on GPU might be slow for small problems).
       * @details Set to true to only move the low-rank part to device (the triangular solve on GPU might be slow for small problems).
       * @param   [in]   cuda_lowrank_only Set to true to only move the low-rank part to device (the triangular solve on GPU might be slow for small problems).
       * @return     Return error message.
       */
      int         SetCUDAOption(bool cuda_lowrank_only);
      
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
   
   typedef GemslrClass<CsrMatrixClass<float>, SequentialVectorClass<float>, float>              precond_gemslr_csr_seq_float;
   typedef GemslrClass<CsrMatrixClass<double>, SequentialVectorClass<double>, double>           precond_gemslr_csr_seq_double;
   typedef GemslrClass<CsrMatrixClass<complexs>, SequentialVectorClass<complexs>, complexs>     precond_gemslr_csr_seq_complexs;
   typedef GemslrClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd>     precond_gemslr_csr_seq_complexd;
   
}

#endif
