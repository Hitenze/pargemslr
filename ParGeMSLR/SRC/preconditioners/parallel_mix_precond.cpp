
#include <iostream>
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../vectors/vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "../matrices/matrix.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../matrices/parallel_csr_matrix.hpp"
#include "parallel_mix_precond.hpp"

namespace pargemslr
{
	
   ParallelMixPrecondClass::ParallelMixPrecondClass() : SolverClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>()
   {
      this->_solver_type = kSolverFloat;
      this->_single_precond = NULL;
      this->_own_single_preconditioner = false;
      this->_location = kMemoryHost;
   }
   
   ParallelMixPrecondClass::~ParallelMixPrecondClass()
   {
      this->Clear();
   }
   
   ParallelMixPrecondClass::ParallelMixPrecondClass(const ParallelMixPrecondClass &solver) : SolverClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>(solver)
   {
      this->_single_precond = solver._single_precond;
      this->_own_single_preconditioner = false;
      this->_location = solver._location;
      this->_x_float = solver._x_float;
      this->_rhs_float = solver._rhs_float;
      this->_matrix_float = solver._matrix_float;
   }
   
   ParallelMixPrecondClass::ParallelMixPrecondClass( ParallelMixPrecondClass &&solver) : SolverClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>(std::move(solver))
   {
      this->_single_precond = solver._single_precond;
      solver._single_precond = NULL;
      this->_own_single_preconditioner = solver._own_single_preconditioner;
      solver._own_single_preconditioner = false;
      this->_location = solver._location;
      this->_x_float = std::move(solver._x_float);
      this->_rhs_float = std::move(solver._rhs_float);
      this->_matrix_float = std::move(solver._matrix_float);
      solver._location = kMemoryHost;
   }
   
   ParallelMixPrecondClass &ParallelMixPrecondClass::operator= (const ParallelMixPrecondClass &solver)
   {
      this->Clear();
      SolverClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>::operator=(solver);
      this->_single_precond = solver._single_precond;
      this->_own_single_preconditioner = false;
      this->_location = solver._location;
      this->_x_float = solver._x_float;
      this->_rhs_float = solver._rhs_float;
      this->_matrix_float = solver._matrix_float;
      return *this;
   }
   
   ParallelMixPrecondClass &ParallelMixPrecondClass::operator= ( ParallelMixPrecondClass &&solver)
   {
      this->Clear();
      SolverClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>::operator=(std::move(solver));
      this->_single_precond = solver._single_precond;
      solver._single_precond = NULL;
      this->_own_single_preconditioner = solver._own_single_preconditioner;
      solver._own_single_preconditioner = false;
      this->_location = solver._location;
      solver._location = kMemoryHost;
      this->_x_float = std::move(solver._x_float);
      this->_rhs_float = std::move(solver._rhs_float);
      this->_matrix_float = std::move(solver._matrix_float);
      return *this;
   }
   
   int ParallelMixPrecondClass::Clear()
   {
      SolverClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>::Clear();
      if(this->_own_single_preconditioner)
      {
         if(this->_single_precond)
         {
            this->_single_precond->Clear();
            this->_single_precond = NULL;
         }
      }
      this->_own_single_preconditioner = false;
      this->_location = kMemoryHost;
      
      this->_x_float.Clear();
      this->_rhs_float.Clear();
      this->_matrix_float.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   
   int ParallelMixPrecondClass::Setup( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs)
   {
      int err;
      if(this->_ready)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      PARGEMSLR_CHKERR(this->_single_precond == NULL);
      PARGEMSLR_CHKERR(this->_matrix == NULL);
      PARGEMSLR_CHKERR(this->_matrix->GetDataLocation() == kMemoryDevice);
      
      /* copy matrix to a single precision one */
      this->_matrix_float.Setup( this->_matrix->GetNumRowsLocal(), 
                                 this->_matrix->GetRowStartGlobal(), 
                                 this->_matrix->GetNumRowsGlobal(), 
                                 this->_matrix->GetNumColsLocal(), 
                                 this->_matrix->GetColStartGlobal(), 
                                 this->_matrix->GetNumColsGlobal(), 
                                 *this->_matrix);
      
      this->_matrix_float.GetDiagMat().Setup( this->_matrix->GetDiagMat().GetNumRowsLocal(), 
                                                this->_matrix->GetDiagMat().GetNumColsLocal(), 
                                                this->_matrix->GetDiagMat().GetNumNonzeros(), 
                                                false);
      
      this->_matrix_float.GetOffdMat().Setup( this->_matrix->GetOffdMat().GetNumRowsLocal(), 
                                                this->_matrix->GetOffdMat().GetNumColsLocal(), 
                                                this->_matrix->GetOffdMat().GetNumNonzeros(), 
                                                false);
      
      this->_matrix_float.GetOffdMap() = this->_matrix->GetOffdMap();
      
      this->_matrix_float.GetDiagMat().GetIVector() = this->_matrix->GetDiagMat().GetIVector();
      this->_matrix_float.GetDiagMat().GetJVector() = this->_matrix->GetDiagMat().GetJVector();
      this->_matrix_float.GetOffdMat().GetIVector() = this->_matrix->GetOffdMat().GetIVector();
      this->_matrix_float.GetOffdMat().GetJVector() = this->_matrix->GetOffdMat().GetJVector();
      
      VectorCopy(this->_matrix->GetDiagMat().GetDataVector(), this->_matrix_float.GetDiagMat().GetDataVector());
      VectorCopy(this->_matrix->GetOffdMat().GetDataVector(), this->_matrix_float.GetOffdMat().GetDataVector());
      
      /* set matrix to the diagonal mat */
      err = this->_single_precond->SetMatrix(this->_matrix_float); PARGEMSLR_CHKERR(err);
      err = this->_single_precond->SetSolveLocation(this->_location); PARGEMSLR_CHKERR(err);
      
      int n_local = this->_matrix->GetNumRowsLocal();
      
      this->_x_float.Setup( n_local, true, *this->_matrix);
      this->_rhs_float.Setup( n_local, true, *this->_matrix);
      
      /* setup the local preconditioner */
      err = this->_single_precond->Setup(this->_x_float, this->_rhs_float); PARGEMSLR_CHKERR(err);
      
      this->_solver_precision = x.GetPrecision();
      
      this->_ready = true;
      
      return err;
      
   }
   
   int ParallelMixPrecondClass::Solve( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs)
   {
      int err;
      if(!(this->_ready))
      {
         PARGEMSLR_ERROR("Solve without setup.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      VectorCopy( x, this->_x_float);
      VectorCopy( rhs, this->_rhs_float);
      
      err = this->_single_precond->Solve(_x_float, _rhs_float); PARGEMSLR_CHKERR(err);
      
      VectorCopy( this->_x_float, x);
      
      return err;
   }
   
   long int ParallelMixPrecondClass::GetNumNonzeros()
   {
      PARGEMSLR_CHKERR(this->_single_precond == NULL);
      
      long int nnz;
      
      // only half
      nnz = this->_single_precond->GetNumNonzeros()/2;
      
      return nnz;
   }
   
   int ParallelMixPrecondClass::SetSinglePreconditioner(SolverClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float> &single_precond)
   {
      return this->SetSinglePreconditionerP(&single_precond);
   }
   
   int ParallelMixPrecondClass::SetSinglePreconditionerP(SolverClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>* single_precond)
   {
      
      if(this->_single_precond && this->_own_single_preconditioner)
      {
         this->_single_precond->Clear();
         PARGEMSLR_FREE(this->_single_precond, kMemoryHost);
      }
      
      this->_single_precond = single_precond;
      this->_own_single_preconditioner = false;
      this->_ready = false;
      
      return PARGEMSLR_SUCCESS;
   }
   
   SolverClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>* ParallelMixPrecondClass::GetSinglePreconditionerP()
   {
      return this->_single_precond;
   }
   
   int ParallelMixPrecondClass::SetOwnSinglePreconditioner(bool own_single_preconditioner)
   {
      this->_own_single_preconditioner = own_single_preconditioner;
      return PARGEMSLR_SUCCESS;
   }
   
   int ParallelMixPrecondClass::SetSolveLocation( const int &location)
   {
      this->_location = location;
      
      if(this->_ready)
      {
         this->MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   
   int ParallelMixPrecondClass::MoveData( const int &location)
   {
      PARGEMSLR_CHKERR(this->_single_precond == NULL);
      PARGEMSLR_CHKERR(this->_ready == false);
      
      if(location == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Single Preconditioner Currently is Host Only.");
      }
      else
      {
         this->_location = location;
         this->_single_precond->SetSolveLocation(location);
         
         this->_x_float.MoveData(location);
         this->_rhs_float.MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   
   
   
   
   
   /* ------------------------------
    * Single Complex
    * ------------------------------
    */
   
   
   CParallelMixPrecondClass::CParallelMixPrecondClass() : SolverClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>()
   {
      this->_solver_type = kSolverFloat;
      this->_single_precond = NULL;
      this->_own_single_preconditioner = false;
      this->_location = kMemoryHost;
   }
   
   CParallelMixPrecondClass::~CParallelMixPrecondClass()
   {
      this->Clear();
   }
   
   CParallelMixPrecondClass::CParallelMixPrecondClass(const CParallelMixPrecondClass &solver) : SolverClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>(solver)
   {
      this->_single_precond = solver._single_precond;
      this->_own_single_preconditioner = false;
      this->_location = solver._location;
      this->_x_float = solver._x_float;
      this->_rhs_float = solver._rhs_float;
      this->_matrix_float = solver._matrix_float;
   }
   
   CParallelMixPrecondClass::CParallelMixPrecondClass( CParallelMixPrecondClass &&solver) : SolverClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>(std::move(solver))
   {
      this->_single_precond = solver._single_precond;
      solver._single_precond = NULL;
      this->_own_single_preconditioner = solver._own_single_preconditioner;
      solver._own_single_preconditioner = false;
      this->_location = solver._location;
      this->_x_float = std::move(solver._x_float);
      this->_rhs_float = std::move(solver._rhs_float);
      this->_matrix_float = std::move(solver._matrix_float);
      solver._location = kMemoryHost;
   }
   
   CParallelMixPrecondClass &CParallelMixPrecondClass::operator= (const CParallelMixPrecondClass &solver)
   {
      this->Clear();
      SolverClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>::operator=(solver);
      this->_single_precond = solver._single_precond;
      this->_own_single_preconditioner = false;
      this->_location = solver._location;
      this->_x_float = solver._x_float;
      this->_rhs_float = solver._rhs_float;
      this->_matrix_float = solver._matrix_float;
      return *this;
   }
   
   CParallelMixPrecondClass &CParallelMixPrecondClass::operator= ( CParallelMixPrecondClass &&solver)
   {
      this->Clear();
      SolverClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>::operator=(std::move(solver));
      this->_single_precond = solver._single_precond;
      solver._single_precond = NULL;
      this->_own_single_preconditioner = solver._own_single_preconditioner;
      solver._own_single_preconditioner = false;
      this->_location = solver._location;
      solver._location = kMemoryHost;
      this->_x_float = std::move(solver._x_float);
      this->_rhs_float = std::move(solver._rhs_float);
      this->_matrix_float = std::move(solver._matrix_float);
      return *this;
   }
   
   int CParallelMixPrecondClass::Clear()
   {
      SolverClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>::Clear();
      if(this->_own_single_preconditioner)
      {
         if(this->_single_precond)
         {
            this->_single_precond->Clear();
            this->_single_precond = NULL;
         }
      }
      this->_own_single_preconditioner = false;
      this->_location = kMemoryHost;
      
      this->_x_float.Clear();
      this->_rhs_float.Clear();
      this->_matrix_float.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   
   int CParallelMixPrecondClass::Setup( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs)
   {
      int err;
      if(this->_ready)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      PARGEMSLR_CHKERR(this->_single_precond == NULL);
      PARGEMSLR_CHKERR(this->_matrix == NULL);
      PARGEMSLR_CHKERR(this->_matrix->GetDataLocation() == kMemoryDevice);
      
      /* copy matrix to a single precision one */
      this->_matrix_float.Setup( this->_matrix->GetNumRowsLocal(), 
                                 this->_matrix->GetRowStartGlobal(), 
                                 this->_matrix->GetNumRowsGlobal(), 
                                 this->_matrix->GetNumColsLocal(), 
                                 this->_matrix->GetColStartGlobal(), 
                                 this->_matrix->GetNumColsGlobal(), 
                                 *this->_matrix);
      
      this->_matrix_float.GetDiagMat().Setup( this->_matrix->GetDiagMat().GetNumRowsLocal(), 
                                                this->_matrix->GetDiagMat().GetNumColsLocal(), 
                                                this->_matrix->GetDiagMat().GetNumNonzeros(), 
                                                false);
      
      this->_matrix_float.GetOffdMat().Setup( this->_matrix->GetOffdMat().GetNumRowsLocal(), 
                                                this->_matrix->GetOffdMat().GetNumColsLocal(), 
                                                this->_matrix->GetOffdMat().GetNumNonzeros(), 
                                                false);
      
      this->_matrix_float.GetOffdMap() = this->_matrix->GetOffdMap();
      
      this->_matrix_float.GetDiagMat().GetIVector() = this->_matrix->GetDiagMat().GetIVector();
      this->_matrix_float.GetDiagMat().GetJVector() = this->_matrix->GetDiagMat().GetJVector();
      this->_matrix_float.GetOffdMat().GetIVector() = this->_matrix->GetOffdMat().GetIVector();
      this->_matrix_float.GetOffdMat().GetJVector() = this->_matrix->GetOffdMat().GetJVector();
      
      VectorCopy(this->_matrix->GetDiagMat().GetDataVector(), this->_matrix_float.GetDiagMat().GetDataVector());
      VectorCopy(this->_matrix->GetOffdMat().GetDataVector(), this->_matrix_float.GetOffdMat().GetDataVector());
      
      /* set matrix to the diagonal mat */
      err = this->_single_precond->SetMatrix(this->_matrix_float); PARGEMSLR_CHKERR(err);
      err = this->_single_precond->SetSolveLocation(this->_location); PARGEMSLR_CHKERR(err);
      
      int n_local = this->_matrix->GetNumRowsLocal();
      
      this->_x_float.Setup( n_local, true, *this->_matrix);
      this->_rhs_float.Setup( n_local, true, *this->_matrix);
      
      /* setup the local preconditioner */
      err = this->_single_precond->Setup(this->_x_float, this->_rhs_float); PARGEMSLR_CHKERR(err);
      
      this->_solver_precision = x.GetPrecision();
      
      this->_ready = true;
      
      return err;
      
   }
   
   int CParallelMixPrecondClass::Solve( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs)
   {
      int err;
      if(!(this->_ready))
      {
         PARGEMSLR_ERROR("Solve without setup.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      VectorCopy(x, this->_x_float);
      VectorCopy(rhs, this->_rhs_float);
      
      err = this->_single_precond->Solve(_x_float, _rhs_float); PARGEMSLR_CHKERR(err);
      
      VectorCopy(this->_x_float, x);
      
      return err;
   }
   
   long int CParallelMixPrecondClass::GetNumNonzeros()
   {
      PARGEMSLR_CHKERR(this->_single_precond == NULL);
      
      long int nnz;
      
      // only half
      nnz = this->_single_precond->GetNumNonzeros()/2;
      
      return nnz;
   }
   
   int CParallelMixPrecondClass::SetSinglePreconditioner(SolverClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs> &single_precond)
   {
      return this->SetSinglePreconditionerP(&single_precond);
   }
   
   int CParallelMixPrecondClass::SetSinglePreconditionerP(SolverClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>* single_precond)
   {
      
      if(this->_single_precond && this->_own_single_preconditioner)
      {
         this->_single_precond->Clear();
         PARGEMSLR_FREE(this->_single_precond, kMemoryHost);
      }
      
      this->_single_precond = single_precond;
      this->_own_single_preconditioner = false;
      this->_ready = false;
      
      return PARGEMSLR_SUCCESS;
   }
   
   SolverClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>* CParallelMixPrecondClass::GetSinglePreconditionerP()
   {
      return this->_single_precond;
   }
   
   int CParallelMixPrecondClass::SetOwnSinglePreconditioner(bool own_single_preconditioner)
   {
      this->_own_single_preconditioner = own_single_preconditioner;
      return PARGEMSLR_SUCCESS;
   }
   
   int CParallelMixPrecondClass::SetSolveLocation( const int &location)
   {
      this->_location = location;
      
      if(this->_ready)
      {
         this->MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   
   int CParallelMixPrecondClass::MoveData( const int &location)
   {
      PARGEMSLR_CHKERR(this->_single_precond == NULL);
      PARGEMSLR_CHKERR(this->_ready == false);
      
      if(location == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Single Preconditioner Currently is Host Only.");
      }
      else
      {
         this->_location = location;
         this->_single_precond->SetSolveLocation(location);
         
         this->_x_float.MoveData(location);
         this->_rhs_float.MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   
}
