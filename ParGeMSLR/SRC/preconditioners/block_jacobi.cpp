
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
#include "block_jacobi.hpp"

namespace pargemslr
{
	
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::BlockJacobiClass() : SolverClass<ParallelMatrixType, ParallelVectorType, DataType>()
   {
      this->_solver_type = kSolverBJ;
      this->_local_precond = NULL;
      this->_own_local_preconditioner = false;
      this->_location = kMemoryHost;
   }
   template precond_bj_csr_par_float::BlockJacobiClass();
   template precond_bj_csr_par_double::BlockJacobiClass();
   template precond_bj_csr_par_complexs::BlockJacobiClass();
   template precond_bj_csr_par_complexd::BlockJacobiClass();
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::~BlockJacobiClass()
   {
      this->Clear();
   }
   template precond_bj_csr_par_float::~BlockJacobiClass();
   template precond_bj_csr_par_double::~BlockJacobiClass();
   template precond_bj_csr_par_complexs::~BlockJacobiClass();
   template precond_bj_csr_par_complexd::~BlockJacobiClass();
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::BlockJacobiClass(const BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &solver) : SolverClass<ParallelMatrixType, ParallelVectorType, DataType>(solver)
   {
      this->_local_precond = solver._local_precond;
      this->_own_local_preconditioner = false;
      this->_location = solver._location;
   }
   template precond_bj_csr_par_float::BlockJacobiClass(const precond_bj_csr_par_float &solver);
   template precond_bj_csr_par_double::BlockJacobiClass(const precond_bj_csr_par_double &solver);
   template precond_bj_csr_par_complexs::BlockJacobiClass(const precond_bj_csr_par_complexs &solver);
   template precond_bj_csr_par_complexd::BlockJacobiClass(const precond_bj_csr_par_complexd &solver);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::BlockJacobiClass( BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &&solver) : SolverClass<ParallelMatrixType, ParallelVectorType, DataType>(std::move(solver))
   {
      this->_local_precond = solver._local_precond;
      solver._local_precond = NULL;
      this->_own_local_preconditioner = solver._own_local_preconditioner;
      solver._own_local_preconditioner = false;
      this->_location = solver._location;
      solver._location = kMemoryHost;
   }
   template precond_bj_csr_par_float::BlockJacobiClass( precond_bj_csr_par_float &&solver);
   template precond_bj_csr_par_double::BlockJacobiClass( precond_bj_csr_par_double &&solver);
   template precond_bj_csr_par_complexs::BlockJacobiClass( precond_bj_csr_par_complexs &&solver);
   template precond_bj_csr_par_complexd::BlockJacobiClass( precond_bj_csr_par_complexd &&solver);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::operator= (const BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &solver)
   {
      this->Clear();
      SolverClass<ParallelMatrixType, ParallelVectorType, DataType>::operator=(solver);
      this->_local_precond = solver._local_precond;
      this->_own_local_preconditioner = false;
      this->_location = solver._location;
      return *this;
   }
   template precond_bj_csr_par_float& precond_bj_csr_par_float::operator= (const precond_bj_csr_par_float &solver);
   template precond_bj_csr_par_double& precond_bj_csr_par_double::operator= (const precond_bj_csr_par_double &solver);
   template precond_bj_csr_par_complexs& precond_bj_csr_par_complexs::operator= (const precond_bj_csr_par_complexs &solver);
   template precond_bj_csr_par_complexd& precond_bj_csr_par_complexd::operator= (const precond_bj_csr_par_complexd &solver);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::operator= ( BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType> &&solver)
   {
      this->Clear();
      SolverClass<ParallelMatrixType, ParallelVectorType, DataType>::operator=(std::move(solver));
      this->_local_precond = solver._local_precond;
      solver._local_precond = NULL;
      this->_own_local_preconditioner = solver._own_local_preconditioner;
      solver._own_local_preconditioner = false;
      this->_location = solver._location;
      solver._location = kMemoryHost;
      return *this;
   }
   template precond_bj_csr_par_float& precond_bj_csr_par_float::operator= ( precond_bj_csr_par_float &&solver);
   template precond_bj_csr_par_double& precond_bj_csr_par_double::operator= ( precond_bj_csr_par_double &&solver);
   template precond_bj_csr_par_complexs& precond_bj_csr_par_complexs::operator= ( precond_bj_csr_par_complexs &&solver);
   template precond_bj_csr_par_complexd& precond_bj_csr_par_complexd::operator= ( precond_bj_csr_par_complexd &&solver);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   int BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::Clear()
   {
      SolverClass<ParallelMatrixType, ParallelVectorType, DataType>::Clear();
      if(this->_own_local_preconditioner)
      {
         if(this->_local_precond)
         {
            this->_local_precond->Clear();
            this->_local_precond = NULL;
         }
      }
      this->_own_local_preconditioner = false;
      this->_location = kMemoryHost;
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_bj_csr_par_float::Clear();
   template int precond_bj_csr_par_double::Clear();
   template int precond_bj_csr_par_complexs::Clear();
   template int precond_bj_csr_par_complexd::Clear();
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   int BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::Setup( ParallelVectorType &x, ParallelVectorType &rhs)
   {
      int err;
      if(this->_ready)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      PARGEMSLR_CHKERR(this->_local_precond == NULL);
      PARGEMSLR_CHKERR(this->_matrix == NULL);
      
      /* set matrix to the diagonal mat */
      err = this->_local_precond->SetMatrix(this->_matrix->GetDiagMat()); PARGEMSLR_CHKERR(err);
      err = this->_local_precond->SetSolveLocation(this->_location); PARGEMSLR_CHKERR(err);
      
      /* setup the local preconditioner */
      err = this->_local_precond->Setup(x.GetDataVector(), rhs.GetDataVector()); PARGEMSLR_CHKERR(err);
      
      this->_solver_precision = x.GetPrecision();
      
      this->_ready = true;
      
      return err;
      
   }
   template int precond_bj_csr_par_float::Setup( vector_par_float &x, vector_par_float &rhs);
   template int precond_bj_csr_par_double::Setup( vector_par_double &x, vector_par_double &rhs);
   template int precond_bj_csr_par_complexs::Setup( vector_par_complexs &x, vector_par_complexs &rhs);
   template int precond_bj_csr_par_complexd::Setup( vector_par_complexd &x, vector_par_complexd &rhs);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   int BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::Solve( ParallelVectorType &x, ParallelVectorType &rhs)
   {
      int err;
      if(!(this->_ready))
      {
         PARGEMSLR_ERROR("Solve without setup.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      err = this->_local_precond->Solve(x.GetDataVector(), rhs.GetDataVector()); PARGEMSLR_CHKERR(err);
      
      return err;
   }
   template int precond_bj_csr_par_float::Solve( vector_par_float &x, vector_par_float &rhs);
   template int precond_bj_csr_par_double::Solve( vector_par_double &x, vector_par_double &rhs);
   template int precond_bj_csr_par_complexs::Solve( vector_par_complexs &x, vector_par_complexs &rhs);
   template int precond_bj_csr_par_complexd::Solve( vector_par_complexd &x, vector_par_complexd &rhs);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   long int BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::GetNumNonzeros()
   {
      PARGEMSLR_CHKERR(this->_local_precond == NULL);
      
      long int nnz_local, nnz_global;
      
      nnz_local = this->_local_precond->GetNumNonzeros();
      
      PARGEMSLR_MPI_CALL( PargemslrMpiAllreduce( &nnz_local, &nnz_global, 1, MPI_SUM, *(this->_matrix->_comm)) );
      
      return nnz_global;
   }
   template long int precond_bj_csr_par_float::GetNumNonzeros();
   template long int precond_bj_csr_par_double::GetNumNonzeros();
   template long int precond_bj_csr_par_complexs::GetNumNonzeros();
   template long int precond_bj_csr_par_complexd::GetNumNonzeros();
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   int BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::SetLocalPreconditioner(SolverClass<LocalMatrixType, LocalVectorType, DataType> &local_precond)
   {
      return this->SetLocalPreconditionerP(&local_precond);
   }
   template int precond_bj_csr_par_float::SetLocalPreconditioner(solver_csr_seq_float &local_precond);
   template int precond_bj_csr_par_double::SetLocalPreconditioner(solver_csr_seq_double &local_precond);
   template int precond_bj_csr_par_complexs::SetLocalPreconditioner(solver_csr_seq_complexs &local_precond);
   template int precond_bj_csr_par_complexd::SetLocalPreconditioner(solver_csr_seq_complexd &local_precond);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   int BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::SetLocalPreconditionerP(SolverClass<LocalMatrixType, LocalVectorType, DataType>* local_precond)
   {
      
      if(this->_local_precond && this->_own_local_preconditioner)
      {
         this->_local_precond->Clear();
         PARGEMSLR_FREE(this->_local_precond, kMemoryHost);
      }
      
      this->_local_precond = local_precond;
      this->_own_local_preconditioner = false;
      this->_ready = false;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_bj_csr_par_float::SetLocalPreconditionerP(solver_csr_seq_float* local_precond);
   template int precond_bj_csr_par_double::SetLocalPreconditionerP(solver_csr_seq_double* local_precond);
   template int precond_bj_csr_par_complexs::SetLocalPreconditionerP(solver_csr_seq_complexs* local_precond);
   template int precond_bj_csr_par_complexd::SetLocalPreconditionerP(solver_csr_seq_complexd* local_precond);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   SolverClass<LocalMatrixType, LocalVectorType, DataType>* BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::GetLocalPreconditionerP()
   {
      return this->_local_precond;
   }
   template solver_csr_seq_float* precond_bj_csr_par_float::GetLocalPreconditionerP();
   template solver_csr_seq_double* precond_bj_csr_par_double::GetLocalPreconditionerP();
   template solver_csr_seq_complexs* precond_bj_csr_par_complexs::GetLocalPreconditionerP();
   template solver_csr_seq_complexd* precond_bj_csr_par_complexd::GetLocalPreconditionerP();
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   int BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::SetOwnLocalPreconditioner(bool own_local_preconditioner)
   {
      this->_own_local_preconditioner = own_local_preconditioner;
      return PARGEMSLR_SUCCESS;
   }
   template int precond_bj_csr_par_float::SetOwnLocalPreconditioner(bool own_local_preconditioner);
   template int precond_bj_csr_par_double::SetOwnLocalPreconditioner(bool own_local_preconditioner);
   template int precond_bj_csr_par_complexs::SetOwnLocalPreconditioner(bool own_local_preconditioner);
   template int precond_bj_csr_par_complexd::SetOwnLocalPreconditioner(bool own_local_preconditioner);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   int BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::SetSolveLocation( const int &location)
   {
      this->_location = location;
      
      if(this->_ready)
      {
         this->MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_bj_csr_par_float::SetSolveLocation( const int &location);
   template int precond_bj_csr_par_double::SetSolveLocation( const int &location);
   template int precond_bj_csr_par_complexs::SetSolveLocation( const int &location);
   template int precond_bj_csr_par_complexd::SetSolveLocation( const int &location);
   
   template <class ParallelMatrixType, class ParallelVectorType, class LocalMatrixType, class LocalVectorType, typename DataType>
   int BlockJacobiClass<ParallelMatrixType, ParallelVectorType, LocalMatrixType, LocalVectorType, DataType>::MoveData( const int &location)
   {
      PARGEMSLR_CHKERR(this->_local_precond == NULL);
      PARGEMSLR_CHKERR(this->_ready == false);
      
      this->_location = location;
      this->_local_precond->SetSolveLocation(location);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_bj_csr_par_float::MoveData( const int &location);
   template int precond_bj_csr_par_double::MoveData( const int &location);
   template int precond_bj_csr_par_complexs::MoveData( const int &location);
   template int precond_bj_csr_par_complexd::MoveData( const int &location);
   
}
