
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
#include "gemslr.hpp"
#include "block_jacobi.hpp"
#include "parallel_gemslr.hpp"

namespace pargemslr
{
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::ParallelGemslrEBFCMatrixClass()
   {
      this->_level = 0;
      this->_option = kGemslrGlobalPrecondGeMSLR;
      this->_gemslr = NULL;
   }
   template precond_gemslrebfc_csr_par_float::ParallelGemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_par_double::ParallelGemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_par_complexs::ParallelGemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_par_complexd::ParallelGemslrEBFCMatrixClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::~ParallelGemslrEBFCMatrixClass()
   {
      this->Clear();
   }
   template precond_gemslrebfc_csr_par_float::~ParallelGemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_par_double::~ParallelGemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_par_complexs::~ParallelGemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_par_complexd::~ParallelGemslrEBFCMatrixClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::ParallelGemslrEBFCMatrixClass(const ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &precond)
   {
      this->_level = precond._level;
      this->_option = precond._option;
      this->_gemslr = precond._gemslr;
      this->_temp_v = precond._temp_v;
   }
   template precond_gemslrebfc_csr_par_float::ParallelGemslrEBFCMatrixClass(const precond_gemslrebfc_csr_par_float &precond);
   template precond_gemslrebfc_csr_par_double::ParallelGemslrEBFCMatrixClass(const precond_gemslrebfc_csr_par_double &precond);
   template precond_gemslrebfc_csr_par_complexs::ParallelGemslrEBFCMatrixClass(const precond_gemslrebfc_csr_par_complexs &precond);
   template precond_gemslrebfc_csr_par_complexd::ParallelGemslrEBFCMatrixClass(const precond_gemslrebfc_csr_par_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::ParallelGemslrEBFCMatrixClass(ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &&precond)
   {
      this->_level = precond._level;
      precond._level = 0;
      this->_option = precond._option;
      precond._option = kGemslrGlobalPrecondGeMSLR;
      this->_gemslr = precond._gemslr;
      precond._gemslr = NULL;
      this->_temp_v = std::move(precond._temp_v);
   }
   template precond_gemslrebfc_csr_par_float::ParallelGemslrEBFCMatrixClass(precond_gemslrebfc_csr_par_float &&precond);
   template precond_gemslrebfc_csr_par_double::ParallelGemslrEBFCMatrixClass(precond_gemslrebfc_csr_par_double &&precond);
   template precond_gemslrebfc_csr_par_complexs::ParallelGemslrEBFCMatrixClass(precond_gemslrebfc_csr_par_complexs &&precond);
   template precond_gemslrebfc_csr_par_complexd::ParallelGemslrEBFCMatrixClass(precond_gemslrebfc_csr_par_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>& ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::operator=(const ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &precond)
   {
      this->_level = precond._level;
      this->_option = precond._option;
      this->_gemslr = precond._gemslr;
      this->_temp_v = precond._temp_v;
      return *this;
   }
   template precond_gemslrebfc_csr_par_float& precond_gemslrebfc_csr_par_float::operator=(const precond_gemslrebfc_csr_par_float &precond);
   template precond_gemslrebfc_csr_par_double& precond_gemslrebfc_csr_par_double::operator=(const precond_gemslrebfc_csr_par_double &precond);
   template precond_gemslrebfc_csr_par_complexs& precond_gemslrebfc_csr_par_complexs::operator=(const precond_gemslrebfc_csr_par_complexs &precond);
   template precond_gemslrebfc_csr_par_complexd& precond_gemslrebfc_csr_par_complexd::operator=(const precond_gemslrebfc_csr_par_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>& ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::operator=(ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &&precond)
   {
      this->_level = precond._level;
      precond._level = 0;
      this->_option = precond._option;
      precond._option = kGemslrGlobalPrecondGeMSLR;
      this->_gemslr = precond._gemslr;
      precond._gemslr = NULL;
      this->_temp_v = std::move(precond._temp_v);
      return *this;
   }
   template precond_gemslrebfc_csr_par_float& precond_gemslrebfc_csr_par_float::operator=(precond_gemslrebfc_csr_par_float &&precond);
   template precond_gemslrebfc_csr_par_double& precond_gemslrebfc_csr_par_double::operator=(precond_gemslrebfc_csr_par_double &&precond);
   template precond_gemslrebfc_csr_par_complexs& precond_gemslrebfc_csr_par_complexs::operator=(precond_gemslrebfc_csr_par_complexs &&precond);
   template precond_gemslrebfc_csr_par_complexd& precond_gemslrebfc_csr_par_complexd::operator=(precond_gemslrebfc_csr_par_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::Clear()
   {
      this->_level = 0;
      this->_option = kGemslrGlobalPrecondGeMSLR;
      this->_gemslr = NULL;
      this->_temp_v.Clear();
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslrebfc_csr_par_float::Clear();
   template int precond_gemslrebfc_csr_par_double::Clear();
   template int precond_gemslrebfc_csr_par_complexs::Clear();
   template int precond_gemslrebfc_csr_par_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::Setup(int level, int option, ParallelGemslrClass<MatrixType, VectorType, DataType> &gemslr)
   {
      this->_level = level;
      this->_option = option;
      this->_gemslr = &gemslr;
      
      if(level >= 0)
      {
         this->_temp_v.SetupPtrStr(this->_gemslr->_levs_v[level]._E_mat);
      }
      else
      {
         this->_temp_v.SetupPtrStr(*this->_gemslr->GetMatrix());
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslrebfc_csr_par_float::Setup(int level, int option, precond_gemslr_csr_par_float &gemslr);
   template int precond_gemslrebfc_csr_par_double::Setup(int level, int option, precond_gemslr_csr_par_double &gemslr);
   template int precond_gemslrebfc_csr_par_complexs::Setup(int level, int option, precond_gemslr_csr_par_complexs &gemslr);
   template int precond_gemslrebfc_csr_par_complexd::Setup(int level, int option, precond_gemslr_csr_par_complexd &gemslr);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::SetupVectorPtrStr(VectorType &v)
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      v.SetupPtrStr(this->_temp_v);
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslrebfc_csr_par_float::SetupVectorPtrStr(vector_par_float &v);
   template int precond_gemslrebfc_csr_par_double::SetupVectorPtrStr(vector_par_double &v);
   template int precond_gemslrebfc_csr_par_complexs::SetupVectorPtrStr(vector_par_complexs &v);
   template int precond_gemslrebfc_csr_par_complexd::SetupVectorPtrStr(vector_par_complexd &v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::GetNumRowsLocal()
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      return this->_gemslr->GetNumRows(this->_level);
   }
   template int precond_gemslrebfc_csr_par_float::GetNumRowsLocal();
   template int precond_gemslrebfc_csr_par_double::GetNumRowsLocal();
   template int precond_gemslrebfc_csr_par_complexs::GetNumRowsLocal();
   template int precond_gemslrebfc_csr_par_complexd::GetNumRowsLocal();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::GetNumColsLocal()
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      return this->_gemslr->GetNumRows(this->_level);
   }
   template int precond_gemslrebfc_csr_par_float::GetNumColsLocal();
   template int precond_gemslrebfc_csr_par_double::GetNumColsLocal();
   template int precond_gemslrebfc_csr_par_complexs::GetNumColsLocal();
   template int precond_gemslrebfc_csr_par_complexd::GetNumColsLocal();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::MatVec( char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      switch(this->_option)
      {
         case kGemslrGlobalPrecondGeMSLR:
         {
            return this->_gemslr->EBFCMatVec(this->_level, trans, alpha, x, beta, y);
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            return this->_gemslr->SCMatVec(this->_level, trans, alpha, x, beta, y);
         }
         case kGemslrGlobalPrecondPCLR:
         {
            return this->_gemslr->PCLRMatVec(this->_level, trans, alpha, x, beta, y);
         }
         case kGemslrGlobalPrecondPSLR:
         {
            PARGEMSLR_ERROR("Unimplemented matvec option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
         case kGemslrGlobalPrecondA:
         {
            return this->_gemslr->ACMatVec(this->_level, trans, alpha, x, beta, y);
         }
         default:
         {
            PARGEMSLR_ERROR("Unkown low-rank matvec option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
   }
   template int precond_gemslrebfc_csr_par_float::MatVec( char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslrebfc_csr_par_double::MatVec( char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslrebfc_csr_par_complexs::MatVec( char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslrebfc_csr_par_complexd::MatVec( char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::ParallelGemslrSchurMatrixClass()
   {
      this->_level = 0;
      this->_option = kGemslrGlobalPrecondGeMSLR;
      this->_gemslr = NULL;
   }
   template precond_gemslr_schur_par_float::ParallelGemslrSchurMatrixClass();
   template precond_gemslr_schur_par_double::ParallelGemslrSchurMatrixClass();
   template precond_gemslr_schur_par_complexs::ParallelGemslrSchurMatrixClass();
   template precond_gemslr_schur_par_complexd::ParallelGemslrSchurMatrixClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::~ParallelGemslrSchurMatrixClass()
   {
      this->Clear();
   }
   template precond_gemslr_schur_par_float::~ParallelGemslrSchurMatrixClass();
   template precond_gemslr_schur_par_double::~ParallelGemslrSchurMatrixClass();
   template precond_gemslr_schur_par_complexs::~ParallelGemslrSchurMatrixClass();
   template precond_gemslr_schur_par_complexd::~ParallelGemslrSchurMatrixClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::ParallelGemslrSchurMatrixClass(const ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType> &precond)
   {
      this->_level = precond._level;
      this->_option = precond._option;
      this->_gemslr = precond._gemslr;
      this->_temp_v = precond._temp_v;
   }
   template precond_gemslr_schur_par_float::ParallelGemslrSchurMatrixClass(const precond_gemslr_schur_par_float &precond);
   template precond_gemslr_schur_par_double::ParallelGemslrSchurMatrixClass(const precond_gemslr_schur_par_double &precond);
   template precond_gemslr_schur_par_complexs::ParallelGemslrSchurMatrixClass(const precond_gemslr_schur_par_complexs &precond);
   template precond_gemslr_schur_par_complexd::ParallelGemslrSchurMatrixClass(const precond_gemslr_schur_par_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::ParallelGemslrSchurMatrixClass(ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType> &&precond)
   {
      this->_level = precond._level;
      precond._level = 0;
      this->_option = precond._option;
      precond._level = kGemslrGlobalPrecondGeMSLR;
      this->_gemslr = precond._gemslr;
      precond._gemslr = NULL;
      this->_temp_v = std::move(precond._temp_v);
   }
   template precond_gemslr_schur_par_float::ParallelGemslrSchurMatrixClass(precond_gemslr_schur_par_float &&precond);
   template precond_gemslr_schur_par_double::ParallelGemslrSchurMatrixClass(precond_gemslr_schur_par_double &&precond);
   template precond_gemslr_schur_par_complexs::ParallelGemslrSchurMatrixClass(precond_gemslr_schur_par_complexs &&precond);
   template precond_gemslr_schur_par_complexd::ParallelGemslrSchurMatrixClass(precond_gemslr_schur_par_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>& ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::operator=(const ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType> &precond)
   {
      this->_level = precond._level;
      this->_option = precond._option;
      this->_gemslr = precond._gemslr;
      this->_temp_v = precond._temp_v;
      return *this;
   }
   template precond_gemslr_schur_par_float& precond_gemslr_schur_par_float::operator=(const precond_gemslr_schur_par_float &precond);
   template precond_gemslr_schur_par_double& precond_gemslr_schur_par_double::operator=(const precond_gemslr_schur_par_double &precond);
   template precond_gemslr_schur_par_complexs& precond_gemslr_schur_par_complexs::operator=(const precond_gemslr_schur_par_complexs &precond);
   template precond_gemslr_schur_par_complexd& precond_gemslr_schur_par_complexd::operator=(const precond_gemslr_schur_par_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>& ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::operator=(ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType> &&precond)
   {
      this->_level = precond._level;
      precond._level = 0;
      this->_option = precond._option;
      precond._level = kGemslrGlobalPrecondGeMSLR;
      this->_gemslr = precond._gemslr;
      precond._gemslr = NULL;
      this->_temp_v = std::move(precond._temp_v);
      return *this;
   }
   template precond_gemslr_schur_par_float& precond_gemslr_schur_par_float::operator=(precond_gemslr_schur_par_float &&precond);
   template precond_gemslr_schur_par_double& precond_gemslr_schur_par_double::operator=(precond_gemslr_schur_par_double &&precond);
   template precond_gemslr_schur_par_complexs& precond_gemslr_schur_par_complexs::operator=(precond_gemslr_schur_par_complexs &&precond);
   template precond_gemslr_schur_par_complexd& precond_gemslr_schur_par_complexd::operator=(precond_gemslr_schur_par_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::Clear()
   {
      this->_level = 0;
      this->_gemslr = NULL;
      this->_option = kGemslrGlobalPrecondGeMSLR;
      this->_temp_v.Clear();
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_schur_par_float::Clear();
   template int precond_gemslr_schur_par_double::Clear();
   template int precond_gemslr_schur_par_complexs::Clear();
   template int precond_gemslr_schur_par_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::Setup(int level, int option, ParallelGemslrClass<MatrixType, VectorType, DataType> &gemslr)
   {
      this->_level = level;
      this->_option = option;
      this->_gemslr = &gemslr;
      
      /* TODO: update this */
      this->_temp_v.SetupPtrStr(this->_gemslr->_levs_v[level]._E_mat);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_schur_par_float::Setup(int level, int option, precond_gemslr_csr_par_float &gemslr);
   template int precond_gemslr_schur_par_double::Setup(int level, int option, precond_gemslr_csr_par_double &gemslr);
   template int precond_gemslr_schur_par_complexs::Setup(int level, int option, precond_gemslr_csr_par_complexs &gemslr);
   template int precond_gemslr_schur_par_complexd::Setup(int level, int option, precond_gemslr_csr_par_complexd &gemslr);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::SetupVectorPtrStr(VectorType &v)
   {
      v.SetupPtrStr(this->_temp_v);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_schur_par_float::SetupVectorPtrStr(vector_par_float &v);
   template int precond_gemslr_schur_par_double::SetupVectorPtrStr(vector_par_double &v);
   template int precond_gemslr_schur_par_complexs::SetupVectorPtrStr(vector_par_complexs &v);
   template int precond_gemslr_schur_par_complexd::SetupVectorPtrStr(vector_par_complexd &v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::MatVec( char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      switch(this->_option)
      {
         case kGemslrGlobalPrecondGeMSLR:
         {
            return this->_gemslr->SchurMatVec(this->_level, 0, trans, alpha, x, beta, y);
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            return this->_gemslr->SchurMatVec(this->_level, 0, trans, alpha, x, beta, y);
            //return this->_gemslr->_levs_v[_level]._S_mat.MatVec(trans, alpha, x, beta, y);
         }
         case kGemslrGlobalPrecondPSLR:
         {
            
         }
         default:
         {
            PARGEMSLR_ERROR("Unkown low-rank matvec option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
   }
   template int precond_gemslr_schur_par_float::MatVec( char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_schur_par_double::MatVec( char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_schur_par_complexs::MatVec( char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_schur_par_complexd::MatVec( char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrSchurMatrixClass<MatrixType, VectorType, DataType>::Solve( VectorType &x, VectorType &rhs)
   {
      /* solve with the Schur complement is the solve on the NEXT level */
      
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      switch(this->_option)
      {
         case kGemslrGlobalPrecondGeMSLR:
         {
            
            DataType one = DataType(1.0);
            
#ifdef PARGEMSLR_TIMING
            int np, myid;
            MPI_Comm comm;
            this->_gemslr->GetMatrix()->GetMpiInfo(np, myid, comm);
#endif
            
            ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_gemslr->_levs_v[this->_level];
            
            /* S-solve */
            if( level_str._lrc > 0)
            {
               /* apply low-rank */
#ifdef PARGEMSLR_TIMING
               if(this->_gemslr->GetSolvePhase() == kGemslrPhaseSetup)
               {
                  PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELR, (this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, rhs, this->_level)));
               }
               else
               {
                  PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_LRC, (this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, rhs, this->_level)));
               }
#else
               this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, rhs, this->_level);
#endif
               level_str._xlr_temp.Axpy( one, rhs);
               /* C solve */
               this->_gemslr->SolveLevelGemslr( x, level_str._xlr_temp, this->_level+1, true);
            }
            else
            {
               /* C solve */
               this->_gemslr->SolveLevelGemslr( x, rhs, this->_level+1, true);
            }
            
            break;
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            
            DataType one = DataType(1.0);
            
#ifdef PARGEMSLR_TIMING
            int np, myid;
            MPI_Comm comm;
            this->_gemslr->GetMatrix()->GetMpiInfo(np, myid, comm);
#endif
            
            ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_gemslr->_levs_v[this->_level];
            
            if(this->_level == 0)
            {
               /* S-solve */
               this->_gemslr->SolveLevelEsmslr( x, rhs, this->_level+1, true);
               if( level_str._lrc > 0)
               {
                  /* apply the low-rank update */
#ifdef PARGEMSLR_TIMING
                  if(this->_gemslr->GetSolvePhase() == kGemslrPhaseSetup)
                  {
                     PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELR, (this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, x, this->_level)));
                  }
                  else
                  {
                     PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_LRC, (this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, x, this->_level)));
                  }
#else
                  this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, x, this->_level);
#endif
                  x.Axpy( one, level_str._xlr_temp);
               }
            }
            else
            {
               if( level_str._lrc > 0)
               {
                  /* apply low-rank */
#ifdef PARGEMSLR_TIMING
                  if(this->_gemslr->GetSolvePhase() == kGemslrPhaseSetup)
                  {
                     PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELR, (this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, rhs, this->_level)));
                  }
                  else
                  {
                     PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_LRC, (this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, rhs, this->_level)));
                  }
#else
                  this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, rhs, this->_level);
#endif
                  level_str._xlr_temp.Axpy( one, rhs);
                  /* C solve */
                  this->_gemslr->SolveLevelEsmslr( x, level_str._xlr_temp, this->_level+1, true);
               }
               else
               {
                  /* C solve */
                  this->_gemslr->SolveLevelEsmslr( x, rhs, this->_level+1, true);
               }
            }
            
            break;
         }
         case kGemslrGlobalPrecondPSLR:
         {
            
         }
         default:
         {
            PARGEMSLR_ERROR("Unkown low-rank matvec option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_schur_par_float::Solve( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs);
   template int precond_gemslr_schur_par_double::Solve( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
   template int precond_gemslr_schur_par_complexs::Solve( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs);
   template int precond_gemslr_schur_par_complexd::Solve( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrLevelClass<MatrixType, VectorType, DataType>::ParallelGemslrLevelClass()
   {
      this->_lrc                             = 0;
      this->_ncomps                          = 0;
      this->_B_precond                       = NULL;
      this->_B_solver                        = NULL;
   }
   template precond_gemslrlevel_csr_par_float::ParallelGemslrLevelClass();
   template precond_gemslrlevel_csr_par_double::ParallelGemslrLevelClass();
   template precond_gemslrlevel_csr_par_complexs::ParallelGemslrLevelClass();
   template precond_gemslrlevel_csr_par_complexd::ParallelGemslrLevelClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrLevelClass<MatrixType, VectorType, DataType>::~ParallelGemslrLevelClass()
   {
      this->Clear();
   }
   template precond_gemslrlevel_csr_par_float::~ParallelGemslrLevelClass();
   template precond_gemslrlevel_csr_par_double::~ParallelGemslrLevelClass();
   template precond_gemslrlevel_csr_par_complexs::~ParallelGemslrLevelClass();
   template precond_gemslrlevel_csr_par_complexd::~ParallelGemslrLevelClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrLevelClass<MatrixType, VectorType, DataType>::ParallelGemslrLevelClass(const ParallelGemslrLevelClass<MatrixType, VectorType, DataType> &str)
   {
      typedef DataType T;
      
      int i;
      
      this->_lrc                             = str._lrc;
      this->_ncomps                          = str._ncomps;
      
      if(str._B_solver)
      {
         PARGEMSLR_MALLOC( this->_B_solver, this->_ncomps, kMemoryHost, SolverClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*);
         switch(str._B_solver[0]->GetSolverType())
         {
            case kSolverIlu:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
                  
                  IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &ilu1 = *(IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) str._B_solver[i];
                  IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &ilu2 = *(IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) this->_B_solver[i];
                  
                  ilu2 = ilu1;
                  
               }
               
               break;
            }
            case kSolverGemslr:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
                  
                  GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &gemslr1 = *(GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) str._B_solver[i];
                  GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &gemslr2 = *(GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) this->_B_solver[i];
                  
                  gemslr2 = gemslr1;
                  
               }
               
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown B solver type, copy failed.");
            }
         }
      }
      
      /* no B_preconditioner yet */
      this->_B_precond = NULL;
      
      this->_B_mat_v.resize(str._B_mat_v.size());
      for(i = 0 ; i < this->_ncomps ;  i++)
      {
         this->_B_mat_v[i] = str._B_mat_v[i];
      }
      
      this->_work_vector = str._work_vector;
#ifdef PARGEMSLR_DEBUG
      this->_work_vector_occupied = str._work_vector_occupied;
#endif
      this->_x_temp = str._x_temp;
      this->_xlr_temp = str._xlr_temp;
      this->_xlr1_temp = str._xlr1_temp;
      this->_xlr2_temp = str._xlr2_temp;
      this->_xlr1_temp_h = str._xlr1_temp_h;
      this->_xlr2_temp_h = str._xlr2_temp_h;
      this->_sol_temp = str._sol_temp;
      this->_rhs_temp = str._rhs_temp;
      this->_sol2_temp = str._sol2_temp;
      this->_rhs2_temp = str._rhs2_temp;
      this->_sol3_temp = str._sol3_temp;
      this->_rhs3_temp = str._rhs3_temp;
      
      this->_E_mat = str._E_mat;
      this->_F_mat = str._F_mat;
      this->_C_mat = str._C_mat;
      this->_S_mat = str._S_mat;
      this->_A_mat = str._A_mat;
      this->_EBFC = str._EBFC;
      this->_Hk = str._Hk;
      this->_Wk = str._Wk;
      this->_WHk = str._WHk;
      this->_cHk = str._cHk;
      this->_cWk = str._cWk;
      this->_cWHk = str._cWHk;
      
      this->_comm_helper = str._comm_helper;
      this->_pperm = str._pperm;
      this->_qperm = str._qperm;
      
   }
   template precond_gemslrlevel_csr_par_float::ParallelGemslrLevelClass(const precond_gemslrlevel_csr_par_float &str);
   template precond_gemslrlevel_csr_par_double::ParallelGemslrLevelClass(const precond_gemslrlevel_csr_par_double &str);
   template precond_gemslrlevel_csr_par_complexs::ParallelGemslrLevelClass(const precond_gemslrlevel_csr_par_complexs &str);
   template precond_gemslrlevel_csr_par_complexd::ParallelGemslrLevelClass(const precond_gemslrlevel_csr_par_complexd &str);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrLevelClass<MatrixType, VectorType, DataType>::ParallelGemslrLevelClass(ParallelGemslrLevelClass<MatrixType, VectorType, DataType> &&str)
   {
      typedef DataType T;
      
      int i;
      
      this->_lrc                             = str._lrc;str._lrc = 0;
      this->_ncomps                          = str._ncomps;str._ncomps = 0;
      
      if(str._B_solver)
      {
         PARGEMSLR_MALLOC( this->_B_solver, this->_ncomps, kMemoryHost, SolverClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*);
         switch(str._B_solver[0]->GetSolverType())
         {
            case kSolverIlu:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
                  
                  IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &ilu1 = *(IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) str._B_solver[i];
                  IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &ilu2 = *(IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) this->_B_solver[i];
                  
                  ilu2 = ilu1;
                  PARGEMSLR_FREE(str._B_solver[i], kMemoryHost);
               }
               
               break;
            }
            case kSolverGemslr:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
                  
                  GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &gemslr1 = *(GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) str._B_solver[i];
                  GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &gemslr2 = *(GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) this->_B_solver[i];
                  
                  gemslr2 = gemslr1;
                  PARGEMSLR_FREE(str._B_solver[i], kMemoryHost);
               }
               
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown B solver type, copy failed.");
            }
         }
         PARGEMSLR_FREE(str._B_solver, kMemoryHost);
      }
      
      /* no B_preconditioner yet */
      this->_B_precond = NULL;
      
      this->_B_mat_v.resize(str._B_mat_v.size());
      for(i = 0 ; i < this->_ncomps ;  i++)
      {
         this->_B_mat_v[i] = str._B_mat_v[i];
         str._B_mat_v[i].Clear();
      }
      std::vector<CsrMatrixClass<DataType> >().swap(str._B_mat_v);
      
      this->_work_vector = std::move(str._work_vector);
#ifdef PARGEMSLR_DEBUG
      this->_work_vector_occupied = std::move(str._work_vector_occupied);
#endif
      this->_x_temp = std::move(str._x_temp);
      this->_xlr_temp = std::move(str._xlr_temp);
      this->_xlr1_temp = std::move(str._xlr1_temp);
      this->_xlr2_temp = std::move(str._xlr2_temp);
      this->_xlr1_temp_h = std::move(str._xlr1_temp_h);
      this->_xlr2_temp_h = std::move(str._xlr2_temp_h);
      this->_sol_temp = std::move(str._sol_temp);
      this->_rhs_temp = std::move(str._rhs_temp);
      this->_sol2_temp = std::move(str._sol2_temp);
      this->_rhs2_temp = std::move(str._rhs2_temp);
      this->_sol3_temp = std::move(str._sol3_temp);
      this->_rhs3_temp = std::move(str._rhs3_temp);
      
      this->_E_mat = std::move(str._E_mat);
      this->_F_mat = std::move(str._F_mat);
      this->_C_mat = std::move(str._C_mat);
      this->_S_mat = std::move(str._S_mat);
      this->_A_mat = std::move(str._A_mat);
      this->_EBFC = std::move(str._EBFC);
      this->_Hk = std::move(str._Hk);
      this->_Wk = std::move(str._Wk);
      this->_WHk = std::move(str._WHk);
      this->_cHk = std::move(str._cHk);
      this->_cWk = std::move(str._cWk);
      this->_cWHk = std::move(str._cWHk);
      
      this->_comm_helper = std::move(str._comm_helper);
      this->_pperm = std::move(str._pperm);
      this->_qperm = std::move(str._qperm);
      
   }
   template precond_gemslrlevel_csr_par_float::ParallelGemslrLevelClass(precond_gemslrlevel_csr_par_float &&str);
   template precond_gemslrlevel_csr_par_double::ParallelGemslrLevelClass(precond_gemslrlevel_csr_par_double &&str);
   template precond_gemslrlevel_csr_par_complexs::ParallelGemslrLevelClass(precond_gemslrlevel_csr_par_complexs &&str);
   template precond_gemslrlevel_csr_par_complexd::ParallelGemslrLevelClass(precond_gemslrlevel_csr_par_complexd &&str);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrLevelClass<MatrixType, VectorType, DataType>& ParallelGemslrLevelClass<MatrixType, VectorType, DataType>::operator=(const ParallelGemslrLevelClass<MatrixType, VectorType, DataType> &str)
   {
      typedef DataType T;
      
      this->Clear();
      
      int i;
      
      this->_lrc                             = str._lrc;
      this->_ncomps                          = str._ncomps;
      
      if(str._B_solver)
      {
         PARGEMSLR_MALLOC( this->_B_solver, this->_ncomps, kMemoryHost, SolverClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*);
         switch(str._B_solver[0]->GetSolverType())
         {
            case kSolverIlu:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
                  
                  IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &ilu1 = *(IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) str._B_solver[i];
                  IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &ilu2 = *(IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) this->_B_solver[i];
                  
                  ilu2 = ilu1;
                  
               }
               
               break;
            }
            case kSolverGemslr:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
                  
                  GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &gemslr1 = *(GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) str._B_solver[i];
                  GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &gemslr2 = *(GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) this->_B_solver[i];
                  
                  gemslr2 = gemslr1;
                  
               }
               
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown B solver type, copy failed.");
            }
         }
      }
      
      /* no B_preconditioner yet */
      this->_B_precond = NULL;
      
      this->_B_mat_v.resize(str._B_mat_v.size());
      for(i = 0 ; i < this->_ncomps ;  i++)
      {
         this->_B_mat_v[i] = str._B_mat_v[i];
      }
      
      this->_work_vector = str._work_vector;
#ifdef PARGEMSLR_DEBUG
      this->_work_vector_occupied = str._work_vector_occupied;
#endif
      this->_x_temp = str._x_temp;
      this->_xlr_temp = str._xlr_temp;
      this->_xlr1_temp = str._xlr1_temp;
      this->_xlr2_temp = str._xlr2_temp;
      this->_xlr1_temp_h = str._xlr1_temp_h;
      this->_xlr2_temp_h = str._xlr2_temp_h;
      this->_sol_temp = str._sol_temp;
      this->_rhs_temp = str._rhs_temp;
      this->_sol2_temp = str._sol2_temp;
      this->_rhs2_temp = str._rhs2_temp;
      this->_sol3_temp = str._sol3_temp;
      this->_rhs3_temp = str._rhs3_temp;
      
      this->_E_mat = str._E_mat;
      this->_F_mat = str._F_mat;
      this->_C_mat = str._C_mat;
      this->_S_mat = str._S_mat;
      this->_A_mat = str._A_mat;
      this->_EBFC = str._EBFC;
      this->_Hk = str._Hk;
      this->_Wk = str._Wk;
      this->_cHk = str._cHk;
      this->_cWk = str._cWk;
      this->_WHk = str._WHk;
      this->_cWHk = str._cWHk;
      
      this->_comm_helper = str._comm_helper;
      this->_pperm = str._pperm;
      this->_qperm = str._qperm;
      
      return *this;
   }
   template precond_gemslrlevel_csr_par_float& precond_gemslrlevel_csr_par_float::operator=(const precond_gemslrlevel_csr_par_float &str);
   template precond_gemslrlevel_csr_par_double& precond_gemslrlevel_csr_par_double::operator=(const precond_gemslrlevel_csr_par_double &str);
   template precond_gemslrlevel_csr_par_complexs& precond_gemslrlevel_csr_par_complexs::operator=(const precond_gemslrlevel_csr_par_complexs &str);
   template precond_gemslrlevel_csr_par_complexd& precond_gemslrlevel_csr_par_complexd::operator=(const precond_gemslrlevel_csr_par_complexd &str);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrLevelClass<MatrixType, VectorType, DataType>& ParallelGemslrLevelClass<MatrixType, VectorType, DataType>::operator=(ParallelGemslrLevelClass<MatrixType, VectorType, DataType> &&str)
   {
      typedef DataType T;
      
      this->Clear();
      
      int i;
      
      this->_lrc                             = str._lrc;str._lrc = 0;
      this->_ncomps                          = str._ncomps;str._ncomps = 0;
      
      if(str._B_solver)
      {
         PARGEMSLR_MALLOC( this->_B_solver, this->_ncomps, kMemoryHost, SolverClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*);
         switch(str._B_solver[0]->GetSolverType())
         {
            case kSolverIlu:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
                  
                  IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &ilu1 = *(IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) str._B_solver[i];
                  IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &ilu2 = *(IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) this->_B_solver[i];
                  
                  ilu2 = ilu1;
                  PARGEMSLR_FREE(str._B_solver[i], kMemoryHost);
               }
               
               break;
            }
            case kSolverGemslr:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
                  
                  GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &gemslr1 = *(GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) str._B_solver[i];
                  GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> &gemslr2 = *(GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) this->_B_solver[i];
                  
                  gemslr2 = gemslr1;
                  PARGEMSLR_FREE(str._B_solver[i], kMemoryHost);
               }
               
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown B solver type, copy failed.");
            }
         }
         PARGEMSLR_FREE(str._B_solver, kMemoryHost);
      }
      
      /* no B_preconditioner yet */
      this->_B_precond = NULL;
      
      this->_B_mat_v.resize(str._B_mat_v.size());
      for(i = 0 ; i < this->_ncomps ;  i++)
      {
         this->_B_mat_v[i] = str._B_mat_v[i];
         str._B_mat_v[i].Clear();
      }
      std::vector<CsrMatrixClass<DataType> >().swap(str._B_mat_v);
      
      this->_work_vector = std::move(str._work_vector);
#ifdef PARGEMSLR_DEBUG
      this->_work_vector_occupied = std::move(str._work_vector_occupied);
#endif
      this->_x_temp = std::move(str._x_temp);
      this->_xlr_temp = std::move(str._xlr_temp);
      this->_xlr1_temp = std::move(str._xlr1_temp);
      this->_xlr2_temp = std::move(str._xlr2_temp);
      this->_xlr1_temp_h = std::move(str._xlr1_temp_h);
      this->_xlr2_temp_h = std::move(str._xlr2_temp_h);
      this->_sol_temp = std::move(str._sol_temp);
      this->_rhs_temp = std::move(str._rhs_temp);
      this->_sol2_temp = std::move(str._sol2_temp);
      this->_rhs2_temp = std::move(str._rhs2_temp);
      this->_sol3_temp = std::move(str._sol3_temp);
      this->_rhs3_temp = std::move(str._rhs3_temp);
      
      this->_E_mat = std::move(str._E_mat);
      this->_F_mat = std::move(str._F_mat);
      this->_C_mat = std::move(str._C_mat);
      this->_S_mat = std::move(str._S_mat);
      this->_A_mat = std::move(str._A_mat);
      this->_EBFC = std::move(str._EBFC);
      this->_Hk = std::move(str._Hk);
      this->_Wk = std::move(str._Wk);
      this->_cHk = std::move(str._cHk);
      this->_cWk = std::move(str._cWk);
      this->_WHk = std::move(str._WHk);
      this->_cWHk = std::move(str._cWHk);
      
      this->_comm_helper = std::move(str._comm_helper);
      this->_pperm = std::move(str._pperm);
      this->_qperm = std::move(str._qperm);
      
      return *this;
   }
   template precond_gemslrlevel_csr_par_float& precond_gemslrlevel_csr_par_float::operator=(precond_gemslrlevel_csr_par_float &&str);
   template precond_gemslrlevel_csr_par_double& precond_gemslrlevel_csr_par_double::operator=(precond_gemslrlevel_csr_par_double &&str);
   template precond_gemslrlevel_csr_par_complexs& precond_gemslrlevel_csr_par_complexs::operator=(precond_gemslrlevel_csr_par_complexs &&str);
   template precond_gemslrlevel_csr_par_complexd& precond_gemslrlevel_csr_par_complexd::operator=(precond_gemslrlevel_csr_par_complexd &&str);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrLevelClass<MatrixType, VectorType, DataType>::Clear()
   {
      int i, v_size;
      
      this->_lrc                             = 0;
      if(this->_B_solver)
      {
         for(i = 0 ; i < 1 ; i ++)
         {
            this->_B_solver[i]->Clear();
            PARGEMSLR_FREE(this->_B_solver[i], kMemoryHost);
         }
         PARGEMSLR_FREE(this->_B_solver, kMemoryHost);
      }
      if(this->_B_precond)
      {
         for(i = 0 ; i < 1 ; i ++)
         {
            this->_B_precond[i]->Clear();
            PARGEMSLR_FREE(this->_B_precond[i], kMemoryHost);
         }
         PARGEMSLR_FREE(this->_B_precond, kMemoryHost);
      }
      
      v_size = (int)(this->_B_mat_v.size());
      for(i = 0 ; i < v_size;  i++)
      {
         this->_B_mat_v[i].Clear();
      }
      std::vector<CsrMatrixClass<DataType> >().swap(this->_B_mat_v);
      
      this->_work_vector.Clear();
#ifdef PARGEMSLR_DEBUG
      this->_work_vector_occupied.Clear();
#endif
      this->_x_temp.Clear();
      this->_xlr_temp.Clear();
      this->_xlr1_temp.Clear();
      this->_xlr2_temp.Clear();
      this->_xlr1_temp_h.Clear();
      this->_xlr2_temp_h.Clear();
      this->_sol_temp.Clear();
      this->_rhs_temp.Clear();
      this->_sol2_temp.Clear();
      this->_rhs2_temp.Clear();
      this->_sol3_temp.Clear();
      this->_rhs3_temp.Clear();
      
      this->_ncomps                          = 0;
      this->_E_mat.Clear();
      this->_F_mat.Clear();
      this->_C_mat.Clear();
      this->_S_mat.Clear();
      this->_A_mat.Clear();
      this->_EBFC.Clear();
      this->_Hk.Clear();
      this->_Wk.Clear();
      this->_cHk.Clear();
      this->_cWk.Clear();
      this->_WHk.Clear();
      this->_cWHk.Clear();
      
      this->_parlog.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslrlevel_csr_par_float::Clear();
   template int precond_gemslrlevel_csr_par_double::Clear();
   template int precond_gemslrlevel_csr_par_complexs::Clear();
   template int precond_gemslrlevel_csr_par_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrLevelClass<MatrixType, VectorType, DataType>::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr)
   {
      int i;
      long int nnz_lr_local, nnz_bsolver_local;
      
      int np, myid;
      MPI_Comm comm;
      this->_parlog.GetMpiInfo(np, myid, comm);
      
      nnz_bsolver_local = 0;
      
      if(this->_B_precond)
      {
         for(i = 0 ; i < this->_ncomps ; i ++)
         {
            nnz_bsolver_local += this->_B_precond[i]->GetNumNonzeros();
         }
      }
      
      if(this->_B_solver)
      {
         for(i = 0 ; i < this->_ncomps ; i ++)
         {
            nnz_bsolver_local += this->_B_solver[i]->GetNumNonzeros();
         }
      }
      
      PARGEMSLR_MPI_CALL( PargemslrMpiAllreduce( &nnz_bsolver_local, &nnz_bsolver, 1, MPI_SUM, comm) );
      
      //nnz_lr_local = this->_Wk.GetNumNonzeros() + this->_Hk.GetNumNonzeros();
      nnz_lr_local = this->_Wk.GetNumNonzeros() + this->_WHk.GetNumNonzeros();
      
      PARGEMSLR_MPI_CALL( PargemslrMpiAllreduce( &nnz_lr_local, &nnz_lr, 1, MPI_SUM, comm) );
      
      return 0;
   }
   template int precond_gemslrlevel_csr_par_float::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template int precond_gemslrlevel_csr_par_double::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template int precond_gemslrlevel_csr_par_complexs::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template int precond_gemslrlevel_csr_par_complexd::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrClass<MatrixType, VectorType, DataType>::ParallelGemslrClass() : SolverClass<MatrixType, VectorType, DataType>()
   {
      this->_n                                  = 0;
      this->_solver_type                        = kSolverParGemslr;
      this->_nlev_max                           = 0;
      this->_nlev_used                          = 0;
      this->_location                           = kMemoryHost;
      this->_global_precond_option              = kGemslrGlobalPrecondGeMSLR;
   }
   template precond_gemslr_csr_par_float::ParallelGemslrClass();
   template precond_gemslr_csr_par_double::ParallelGemslrClass();
   template precond_gemslr_csr_par_complexs::ParallelGemslrClass();
   template precond_gemslr_csr_par_complexd::ParallelGemslrClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrClass<MatrixType, VectorType, DataType>::ParallelGemslrClass(const ParallelGemslrClass<MatrixType, VectorType, DataType> &precond) : SolverClass<MatrixType, VectorType, DataType>(precond)
   {
      
      int i;
      
      this->_n                                  = precond._n;
      this->_nlev_max                           = precond._nlev_max;
      this->_nlev_used                          = precond._nlev_used;
      this->_location                           = precond._location;
      
      this->_lev_ptr_v = precond._lev_ptr_v;
      
      this->_dom_ptr_v2.resize(this->_nlev_used);
      this->_levs_v.resize(this->_nlev_used);
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i] = precond._levs_v[i];
         this->_dom_ptr_v2[i] = precond._dom_ptr_v2[i];
      }
      
      if(precond._lev_A.size() > 0)
      {
         this->_lev_A.resize(1);
         this->_lev_A[0] = precond._lev_A[0];
      }
      
      this->_global_precond_option = precond._global_precond_option;
      
      this->_inner_iters_matrix = precond._inner_iters_matrix;
      this->_inner_iters_precond = precond._inner_iters_precond;
      this->_inner_iters_solver = precond._inner_iters_solver;
      
   }
   template precond_gemslr_csr_par_float::ParallelGemslrClass(const precond_gemslr_csr_par_float &precond);
   template precond_gemslr_csr_par_double::ParallelGemslrClass(const precond_gemslr_csr_par_double &precond);
   template precond_gemslr_csr_par_complexs::ParallelGemslrClass(const precond_gemslr_csr_par_complexs &precond);
   template precond_gemslr_csr_par_complexd::ParallelGemslrClass(const precond_gemslr_csr_par_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrClass<MatrixType, VectorType, DataType>::ParallelGemslrClass(ParallelGemslrClass<MatrixType, VectorType, DataType> &&precond) : SolverClass<MatrixType, VectorType, DataType>(std::move(precond))
   {
      
      int i;
      
      this->_n                                  = precond._n;precond._n = 0;
      this->_nlev_max                           = precond._nlev_max;precond._nlev_max = 0;
      this->_nlev_used                          = precond._nlev_used;precond._nlev_used = 0;
      this->_location                           = precond._location;precond._location = kMemoryHost;
      
      this->_lev_ptr_v = std::move(precond._lev_ptr_v);
      
      this->_dom_ptr_v2.resize(this->_nlev_used);
      this->_levs_v.resize(this->_nlev_used);
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i] = std::move(precond._levs_v[i]);
         this->_dom_ptr_v2[i] = std::move(precond._dom_ptr_v2[i]);
      }
      std::vector<IntVectorClass<int> >().swap(precond._dom_ptr_v2); 
      std::vector<ParallelGemslrLevelClass< MatrixType, VectorType, DataType> >().swap(precond._levs_v); 
      
      if(precond._lev_A.size() > 0)
      {
         this->_lev_A.resize(1);
         this->_lev_A[0] = std::move(precond._lev_A[0]);
         std::vector<ParallelGemslrLevelClass< MatrixType, VectorType, DataType> >().swap(precond._lev_A); 
      }
      
      this->_global_precond_option = precond._global_precond_option;precond._global_precond_option = kGemslrGlobalPrecondGeMSLR;
      
      this->_inner_iters_matrix = std::move(precond._inner_iters_matrix);
      this->_inner_iters_precond = std::move(precond._inner_iters_precond);
      this->_inner_iters_solver = std::move(precond._inner_iters_solver);
      
   }
   template precond_gemslr_csr_par_float::ParallelGemslrClass(precond_gemslr_csr_par_float &&precond);
   template precond_gemslr_csr_par_double::ParallelGemslrClass(precond_gemslr_csr_par_double &&precond);
   template precond_gemslr_csr_par_complexs::ParallelGemslrClass(precond_gemslr_csr_par_complexs &&precond);
   template precond_gemslr_csr_par_complexd::ParallelGemslrClass(precond_gemslr_csr_par_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrClass<MatrixType, VectorType, DataType>& ParallelGemslrClass<MatrixType, VectorType, DataType>::operator=(const ParallelGemslrClass<MatrixType, VectorType, DataType> &precond)
   {
      this->Clear();
      SolverClass<MatrixType, VectorType, DataType>::operator=(precond);
      
      int i;
      
      this->_n                                  = precond._n;
      this->_nlev_max                           = precond._nlev_max;
      this->_nlev_used                          = precond._nlev_used;
      this->_location                           = precond._location;
      
      this->_lev_ptr_v = precond._lev_ptr_v;
      
      this->_dom_ptr_v2.resize(this->_nlev_used);
      this->_levs_v.resize(this->_nlev_used);
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i] = precond._levs_v[i];
         this->_dom_ptr_v2[i] = precond._dom_ptr_v2[i];
      }
      
      if(precond._lev_A.size() > 0)
      {
         this->_lev_A.resize(1);
         this->_lev_A[0] = precond._lev_A[0];
      }
      
      this->_global_precond_option = precond._global_precond_option;
      
      this->_inner_iters_matrix = precond._inner_iters_matrix;
      this->_inner_iters_precond = precond._inner_iters_precond;
      this->_inner_iters_solver = precond._inner_iters_solver;
      
      return *this;
   }
   template precond_gemslr_csr_par_float& precond_gemslr_csr_par_float::operator=(const precond_gemslr_csr_par_float &precond);
   template precond_gemslr_csr_par_double& precond_gemslr_csr_par_double::operator=(const precond_gemslr_csr_par_double &precond);
   template precond_gemslr_csr_par_complexs& precond_gemslr_csr_par_complexs::operator=(const precond_gemslr_csr_par_complexs &precond);
   template precond_gemslr_csr_par_complexd& precond_gemslr_csr_par_complexd::operator=(const precond_gemslr_csr_par_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrClass<MatrixType, VectorType, DataType>& ParallelGemslrClass<MatrixType, VectorType, DataType>::operator=(ParallelGemslrClass<MatrixType, VectorType, DataType> &&precond)
   {
      this->Clear();
      SolverClass<MatrixType, VectorType, DataType>::operator=(std::move(precond));
      
      int i;
      
      this->_n                                  = precond._n;precond._n = 0;
      this->_nlev_max                           = precond._nlev_max;precond._nlev_max = 0;
      this->_nlev_used                          = precond._nlev_used;precond._nlev_used = 0;
      this->_location                           = precond._location;precond._location = kMemoryHost;
      
      this->_lev_ptr_v = std::move(precond._lev_ptr_v);
      
      this->_dom_ptr_v2.resize(this->_nlev_used);
      this->_levs_v.resize(this->_nlev_used);
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i] = std::move(precond._levs_v[i]);
         this->_dom_ptr_v2[i] = std::move(precond._dom_ptr_v2[i]);
      }
      std::vector<IntVectorClass<int> >().swap(precond._dom_ptr_v2); 
      std::vector<ParallelGemslrLevelClass< MatrixType, VectorType, DataType> >().swap(precond._levs_v); 
      
      if(precond._lev_A.size() > 0)
      {
         this->_lev_A.resize(1);
         this->_lev_A[0] = std::move(precond._lev_A[0]);
         std::vector<ParallelGemslrLevelClass< MatrixType, VectorType, DataType> >().swap(precond._lev_A); 
      }
      
      this->_global_precond_option = precond._global_precond_option;precond._global_precond_option = kGemslrGlobalPrecondGeMSLR;
      
      this->_inner_iters_matrix = std::move(precond._inner_iters_matrix);
      this->_inner_iters_precond = std::move(precond._inner_iters_precond);
      this->_inner_iters_solver = std::move(precond._inner_iters_solver);
      
      return *this;
   }
   template precond_gemslr_csr_par_float& precond_gemslr_csr_par_float::operator=(precond_gemslr_csr_par_float &&precond);
   template precond_gemslr_csr_par_double& precond_gemslr_csr_par_double::operator=(precond_gemslr_csr_par_double &&precond);
   template precond_gemslr_csr_par_complexs& precond_gemslr_csr_par_complexs::operator=(precond_gemslr_csr_par_complexs &&precond);
   template precond_gemslr_csr_par_complexd& precond_gemslr_csr_par_complexd::operator=(precond_gemslr_csr_par_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   ParallelGemslrClass<MatrixType, VectorType, DataType>::~ParallelGemslrClass()
   {
      this->Clear();
   }
   template precond_gemslr_csr_par_float::~ParallelGemslrClass();
   template precond_gemslr_csr_par_double::~ParallelGemslrClass();
   template precond_gemslr_csr_par_complexs::~ParallelGemslrClass();
   template precond_gemslr_csr_par_complexd::~ParallelGemslrClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::Clear()
   {
      SolverClass<MatrixType, VectorType, DataType>::Clear();
      
      int i;
      
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i].Clear();
         this->_dom_ptr_v2[i].Clear();
      }
      
      if(this->_lev_A.size() > 0)
      {
         this->_lev_A[0].Clear();
      }
      
      this->_lev_ptr_v.Clear();
      std::vector<IntVectorClass<int> >().swap(_dom_ptr_v2); 
      std::vector<ParallelGemslrLevelClass< MatrixType, VectorType, DataType> >().swap(this->_levs_v); 
      std::vector<ParallelGemslrLevelClass< MatrixType, VectorType, DataType> >().swap(this->_lev_A); 
      
      this->_n                                  = 0;
      this->_nlev_max                           = 0;
      this->_nlev_used                          = 0;
      this->_location                           = kMemoryHost;
      
      this->_global_precond_option = kGemslrGlobalPrecondGeMSLR;
      
      this->_inner_iters_matrix.Clear();
      this->_inner_iters_precond.Clear();
      this->_inner_iters_solver.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::Clear();
   template int precond_gemslr_csr_par_double::Clear();
   template int precond_gemslr_csr_par_complexs::Clear();
   template int precond_gemslr_csr_par_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::PrintInfo()
   {
      /* print the preconditioner info */
      
      int i, j, ncomp;
         
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
      
      if(myid == 0)
      {
         PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
         switch(this->_global_precond_option)
         {
            case kGemslrGlobalPrecondBJ:
            {
               PARGEMSLR_PRINT("Setup Parallel BJ\n");
               break;
            }
            case kGemslrGlobalPrecondGeMSLR:
            {
               PARGEMSLR_PRINT("Setup Parallel GeMSLR\n");
               if(this->_gemslr_setups._enable_inner_iters_setup)
               {
                  PARGEMSLR_PRINT("\tInner Iteration ON. tol: %g; maxits: %d\n",this->_gemslr_setups._inner_iters_tol_setup,this->_gemslr_setups._inner_iters_maxits_setup);
               }
               else
               {
                  PARGEMSLR_PRINT("\tInner Iteration OFF.\n");
               }
               switch(this->_gemslr_setups._solve_option_setup)
               {
                  case kGemslrLUSolve:
                  {
                     PARGEMSLR_PRINT("\tAdditive LU solve\n");
                     break;
                  }
                  case kGemslrUSolve:
                  {
                     PARGEMSLR_PRINT("\tAdditive U solve\n");
                     break;
                  }
                  case kGemslrMulSolve:
                  {
                     PARGEMSLR_PRINT("\tMultiplicative solve\n");
                     break;
                  }
                  default:
                  {
                     PARGEMSLR_PRINT("\tUnknown solve\n");
                     break;
                  }
               }
               break;
            }
            case kGemslrGlobalPrecondESMSLR:
            {
               PARGEMSLR_PRINT("Setup Parallel EsMSLR\n");
               if(this->_gemslr_setups._enable_inner_iters_setup)
               {
                  PARGEMSLR_PRINT("\tInner Iteration ON. tol: %g; maxits: %d\n",this->_gemslr_setups._inner_iters_tol_setup,this->_gemslr_setups._inner_iters_maxits_setup);
               }
               else
               {
                  PARGEMSLR_PRINT("\tInner Iteration OFF.\n");
               }
               switch(this->_gemslr_setups._solve_option_setup)
               {
                  case kGemslrLUSolve:
                  {
                     PARGEMSLR_PRINT("\tAdditive LU solve\n");
                     break;
                  }
                  case kGemslrUSolve:
                  {
                     PARGEMSLR_PRINT("\tAdditive U solve\n");
                     break;
                  }
                  case kGemslrMulSolve:
                  {
                     PARGEMSLR_PRINT("\tMultiplicative solve\n");
                     break;
                  }
                  default:
                  {
                     PARGEMSLR_PRINT("\tUnknown solve\n");
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* in general won't reach here */
               PARGEMSLR_PRINT("Setup Parallel GeMSLR with known option\n");
               break;
            }
         }
         PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
         PARGEMSLR_PRINT("Level   Ncomp   Size           Nnz            rk      nnzLU            nnzLR\n");
      }
      
      if(this->_lev_A[0]._lrc > 0)
      {
         long int nnz_bsolver, nnz_lr;
      
         this->_lev_A[0].GetNumNonzeros(nnz_bsolver, nnz_lr);
      
         /* Level Size Nnz rk nnzLU nnzLR */
         if(myid == 0)
         {
            PARGEMSLR_PRINT("OUTER     N/A            N/A            N/A   %5d   %10e   %10e\n", this->_lev_A[0]._lrc, (float)nnz_bsolver, (float)nnz_lr);
         }
      }
      
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[i];
         
         long int n_level_local = (long int)this->_n - (long int)this->_lev_ptr_v[i];
         long int n_level_global;
         
         PARGEMSLR_MPI_CALL(PargemslrMpiReduce( &n_level_local, &n_level_global, 1, MPI_SUM, 0, comm));
         
         //long int nnz_level_local = level_str._E_mat.GetNumNonzeros() + level_str._F_mat.GetNumNonzeros();
         long int nnz_level_local = 0;
         
         ncomp = level_str._B_mat_v.size();
         
         for(j = 0 ; j < ncomp ; j ++)
         {
            nnz_level_local += level_str._B_mat_v[j].GetNumNonzeros();
         }
         
         long int nnz_level_global;
         
         PARGEMSLR_MPI_CALL(PargemslrMpiReduce( &nnz_level_local, &nnz_level_global, 1, MPI_SUM, 0, comm));
         
         nnz_level_global += level_str._E_mat.GetNumNonzeros() + level_str._F_mat.GetNumNonzeros();
         
         int ncomp_local = level_str._ncomps;
         int ncomp_global;
         
         PARGEMSLR_MPI_CALL(PargemslrMpiReduce( &ncomp_local, &ncomp_global, 1, MPI_SUM, 0, comm));
         
         long int nnz_bsolver, nnz_lr;
         
         level_str.GetNumNonzeros(nnz_bsolver, nnz_lr);
         
         /* Level Size Nnz rk nnzLU nnzLR */
         if(myid == 0)
         {
            PARGEMSLR_PRINT("%5d   %5d   %12ld   %12ld   %5d   %10e   %10e\n", i, ncomp_global, n_level_global, nnz_level_global, level_str._lrc, (float)nnz_bsolver, (float)nnz_lr);
         }
      }
      long int nnzA, nnz, nnzLU, nnzLR;
      nnzA = this->_matrix->GetNumNonzeros();
      nnz = this->GetNumNonzeros(nnzLU, nnzLR);
      if(myid == 0)
      {
         PARGEMSLR_PRINT("Fill level: ILU: %f; Low-rank: %f; Total: %f\n",(double)nnzLU/nnzA,(double)nnzLR/nnzA,(double)nnz/nnzA);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::PrintInfo();
   template int precond_gemslr_csr_par_double::PrintInfo();
   template int precond_gemslr_csr_par_complexs::PrintInfo();
   template int precond_gemslr_csr_par_complexd::PrintInfo();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::CheckParameters()
   {
      
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
      
      /* handle special options 
       * kGemslrGlobalPrecondBJ:
       *    Do nothing.
       * 
       * kGemslrGlobalPrecondESMSLR:
       *    Currently we don't support global reordering.
       *    1. When the number of levels is 1, switch to BJ.
       *    2. When np is 1 and number of levels greater than 1, switch to GeMSLR with global reordering.
       * 
       * kGemslrGlobalPrecondGeMSLR:
       *    Currently we don't support global reordering.
       *    1. When the number of levels is 1, switch to BJ.
       *    2. When np is 1 and number of levels greater than 1, switch to GeMSLR with global reordering.
       * 
       */
      switch(this->_global_precond_option)
      {
         case kGemslrGlobalPrecondBJ:
         {
            /* BJ do nothing */
            break;
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            //if(this->_gemslr_setups._nlev_setup < 2 && this->_gemslr_setups._global_partition_setup)
            if(this->_gemslr_setups._nlev_setup < 2)
            {
               /* nlev less than 2, go to bj */
               this->_global_precond_option = kGemslrGlobalPrecondBJ;
               //this->_gemslr_setups._global_partition_setup = true;
               break;
            }
            
            //if( np == 1 && !this->_gemslr_setups._global_partition_setup)
            if( np == 1 )
            {
               /* single np, no input DD, goto GeMSLR with global reordering */
               this->_global_precond_option = kGemslrGlobalPrecondGeMSLR;
               this->_gemslr_setups._global_partition_setup = true;
               if(myid == 0)
               {
                  PARGEMSLR_WARNING("Single np, switch to GeMSLR with global reordering.");
               }
               break;
            }
            
            if(this->_gemslr_setups._solve_option_setup == kGemslrMulSolve)
            {
               /* no multiplicative solve, goto GeMSLR with global reordering */
               this->_global_precond_option = kGemslrGlobalPrecondGeMSLR;
               this->_gemslr_setups._global_partition_setup = true;
               if(myid == 0)
               {
                  PARGEMSLR_WARNING("EsMSLR doens't support multiplicative solve, switch to GeMSLR.");
               }
               break;
            }
            
            if(this->_gemslr_setups._nlev_setup == 2 && 
               (this->_gemslr_setups._level_setups._C_solve_option != kGemslrCSolveILUK && 
                this->_gemslr_setups._level_setups._C_solve_option != kGemslrCSolveILUT) )
            {
               /* 2 levels with PSMR */
               this->_global_precond_option = kGemslrGlobalPrecondGeMSLR;
               this->_gemslr_setups._global_partition_setup = true;
               if(myid == 0)
               {
                  PARGEMSLR_WARNING("EsMSLR doens't support PSLR with only two levels, switch to GeMSLR.");
               }
               break;
            }
            
            break;
         }
         case kGemslrGlobalPrecondGeMSLR:
         {
            
            //if(this->_gemslr_setups._nlev_setup < 2 && this->_gemslr_setups._global_partition_setup)
            if(this->_gemslr_setups._nlev_setup < 2)
            {
               /* nlev less than 2, go to bj */
               this->_global_precond_option = kGemslrGlobalPrecondBJ;
               break;
            }
            
            if( np == 1 && !this->_gemslr_setups._global_partition_setup)
            {
               /* single np, goto GeMSLR with global reordering */
               
               this->_gemslr_setups._global_partition_setup = true;
               if(myid == 0)
               {
                  PARGEMSLR_WARNING("Single np, switch to global reordering.");
               }
               break;
            }
            
            break;
         }
         case kGemslrGlobalPrecondPSLR:
         {
            
            if( np == 1 && !this->_gemslr_setups._global_partition_setup)
            {
               /* single np, go to global reordering */
               if(myid == 0)
               {
                  PARGEMSLR_WARNING("Single np, switch to global reordering.");
               }
               this->_gemslr_setups._global_partition_setup = true;
               break;
            }
            
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown Global partition option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::CheckParameters();
   template int precond_gemslr_csr_par_double::CheckParameters();
   template int precond_gemslr_csr_par_complexs::CheckParameters();
   template int precond_gemslr_csr_par_complexd::CheckParameters();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::Setup( VectorType &x, VectorType &rhs)
   {
      /* Wrapper of the setup phase of GeMSLR */
      
      if(this->_ready)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      if(!this->_matrix)
      {
         PARGEMSLR_ERROR("Call GeMSLR setup without matrix.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }
      
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
      
      this->CheckParameters();
      
      /* update the solver precision, leave this interface for half precision */
      this->_solver_precision = x.GetPrecision();
      
      /* save the location */
      int location = this->_matrix->GetDataLocation();
      if(location == kMemoryDevice)
      {
         PARGEMSLR_WARNING("The setup phase of Parallel GeMSLR is host only, moving matrix to the host.");
         this->_matrix->MoveData(kMemoryHost);
      }
      
      /* we might use differnt function calls during the setup phase and the solve phase */
      this->_gemslr_setups._solve_phase_setup = kGemslrPhaseSetup;
      
      /* create temp vector */
      int n = this->_matrix->GetNumRowsLocal();
      
      this->_n = n;
      
      /* create data str for global low-rank correction */
      this->_lev_A.resize(1);
      
      /* switch global solve options */
      if(this->_gemslr_setups._global_partition_setup)
      {
         /* In this case, we use the input DD of the original system */
         switch(this->_global_precond_option)
         {
            case kGemslrGlobalPrecondBJ:
            {
               /* block Jacobi ILU */
               
               /* set up the level structure.
                * For BJ, this is only a single level structure.
                */
               this->SetupPermutation();
               
               /* In the step, setup the B solver
                * also move B_mat and B_solver to the device when necessary
                */
#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_ILUT, this->SetupBSolve( x, rhs));
#else
               this->SetupBSolve( x, rhs);
#endif

#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_LRC, this->SetupLowRank( x, rhs));
#else
               this->SetupLowRank( x, rhs);
#endif
               
               break;
            }
            case kGemslrGlobalPrecondESMSLR:
            {
               /* Currently not supported */
               
               this->_gemslr_setups._global_partition_setup = false;
               
               goto label_esmslr_local;
               
               break;
            }
            case kGemslrGlobalPrecondPSLR:
            {
               /* Currently not supported */
               
               this->_global_precond_option = kGemslrGlobalPrecondGeMSLR;
               goto label_mlev_global;
               
               break;
            }
            case kGemslrGlobalPrecondGeMSLR:
            {
               /* GeMSLR with global reordering */
label_mlev_global:
               /* the setup phase should be on the host 
                * after this step, we can move the permutation onto the device when necessary
                */
               this->SetupPermutation();
            
               /* In the step, setup the B solver
                * also move B_mat and B_solver to the device when necessary
                */
#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_ILUT, this->SetupBSolve( x, rhs));
#else
               this->SetupBSolve( x, rhs);
#endif
               /* In the step, setup the low_rank
                * also move E_mat and F_mat to the device when necessary
                */
#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_LRC, this->SetupLowRank( x, rhs));
#else
               this->SetupLowRank( x, rhs);
#endif

               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown Global preconditioining option.");
               return PARGEMSLR_ERROR_INVALED_OPTION;
            }
         }
      }
      else
      {
         /* In this case, we use the input DD of the original system */
         switch(this->_global_precond_option)
         {
            case kGemslrGlobalPrecondBJ:
            {
               /* block Jacobi ILU */
                
               /* Currently the same BJ */
               this->SetupPermutation();
               
               /* In the step, setup the B solver
                * also move B_mat and B_solver to the device when necessary
                */
#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_ILUT, this->SetupBSolve( x, rhs));
#else
               this->SetupBSolve( x, rhs);
#endif

#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_LRC, this->SetupLowRank( x, rhs));
#else
               this->SetupLowRank( x, rhs);
#endif
               break;
            }
            case kGemslrGlobalPrecondESMSLR:
            {
label_esmslr_local:
               /* Partial ILU and explicit Schur complement */
               /* step 1: set permutation 
                * after this step, the level structure is ready except for the top level
                */
               if(this->SetupPermutation() == PARGEMSLR_RETURN_PARILU_NO_INTERIOR)
               {
                  /* input DD is not good, switch to GeMSLR */
                  this->_global_precond_option = kGemslrGlobalPrecondGeMSLR;
                  this->_gemslr_setups._global_partition_setup = true;
                  
                  PARGEMSLR_WARNING("The input DD is not good enough, switch to GeMSLR with global reordering.");
                  
                  goto label_mlev_global;
                  
               }
               
               /* setp 2: In the step, setup the B solver
                * including the partial ILU on the top level
                */
#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_ILUT, this->SetupBSolve( x, rhs));
#else
               this->SetupBSolve( x, rhs);
#endif
               
#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_LRC, this->SetupLowRank( x, rhs));
#else
               this->SetupLowRank( x, rhs);
#endif

               break;
            }
            case kGemslrGlobalPrecondPSLR:
            {
               /* the PSLR option */
               
               this->_global_precond_option = kGemslrGlobalPrecondGeMSLR;
               this->_gemslr_setups._global_partition_setup = true;
               
               goto label_mlev_global;
               
               break;
            }
            case kGemslrGlobalPrecondGeMSLR:
            {
               /* GeMSLR with local reordering */
               /* the setup phase should be on the host 
                * after this step, we can move the permutation onto the device when necessary
                */
               if(this->SetupPermutation() == PARGEMSLR_RETURN_PARILU_NO_INTERIOR)
               {
                  /* partition fail, switch to global option */
                  this->_global_precond_option = kGemslrGlobalPrecondGeMSLR;
                  this->_gemslr_setups._global_partition_setup = true;
                  
                  PARGEMSLR_WARNING("The input DD is not good enough, switch to GeMSLR with global reordering.");
                  
                  goto label_mlev_global;
                  
               }
            
               /* In the step, setup the B solver
                * also move B_mat and B_solver to the device when necessary
                */
#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_ILUT, this->SetupBSolve( x, rhs));
#else
               this->SetupBSolve( x, rhs);
#endif
               /* In the step, setup the low_rank
                * also move E_mat and F_mat to the device when necessary
                */
#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_LRC, this->SetupLowRank( x, rhs));
#else
               this->SetupLowRank( x, rhs);
#endif
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown Global preconditioining option.");
               return PARGEMSLR_ERROR_INVALED_OPTION;
            }
         }
      }
      
      this->_matrix->MoveData(location);
      
      /* set inner solve */
      switch(this->_global_precond_option)
      {
         case kGemslrGlobalPrecondBJ:
         {
            /* BJ do nothing, no inner solve */
            break;
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            /* inner solve with option kGemslrGlobalPrecondESMSLR */
            if(this->_gemslr_setups._enable_inner_iters_setup)
            {
               /* solve on the top level with GMRES */
               this->_inner_iters_matrix.Setup( 0, kGemslrGlobalPrecondESMSLR, *this);
               this->_inner_iters_precond.SetMatrix(this->_inner_iters_matrix);
               this->_inner_iters_solver.SetMatrix(this->_inner_iters_matrix);
               this->_inner_iters_solver.SetPreconditioner(this->_inner_iters_precond);
               
               this->_inner_iters_solver.SetKrylovSubspaceDimension(this->_gemslr_setups._inner_iters_maxits_setup);
               this->_inner_iters_solver.SetMaxNumberIterations(this->_gemslr_setups._inner_iters_maxits_setup);
               this->_inner_iters_solver.SetTolerance(this->_gemslr_setups._inner_iters_tol_setup);
               this->_inner_iters_solver.SetSolveLocation(this->_location);
               
               /* mute the print of fgmres */
               this->_inner_iters_solver.SetPrintOption(-1);
               
               this->_inner_iters_solver.Setup(x, rhs);
            }
            break;
         }
         case kGemslrGlobalPrecondGeMSLR:
         {
            
            /* inner solve with option kGemslrGlobalPrecondGeMSLR.
             * the MatVec option is RAP for multiplicative solve and C-EB^{-1}F for additive solve.
             */
            if(this->_gemslr_setups._enable_inner_iters_setup)
            {
               /* solve on the top level with RAP */
               this->_inner_iters_matrix.Setup( 0, kGemslrGlobalPrecondGeMSLR, *this);
               this->_inner_iters_precond.SetMatrix(this->_inner_iters_matrix);
               this->_inner_iters_solver.SetMatrix(this->_inner_iters_matrix);
               this->_inner_iters_solver.SetPreconditioner(this->_inner_iters_precond);
               
               this->_inner_iters_solver.SetKrylovSubspaceDimension(this->_gemslr_setups._inner_iters_maxits_setup);
               this->_inner_iters_solver.SetMaxNumberIterations(this->_gemslr_setups._inner_iters_maxits_setup);
               this->_inner_iters_solver.SetTolerance(this->_gemslr_setups._inner_iters_tol_setup);
               this->_inner_iters_solver.SetSolveLocation(this->_location);
               
               /* mute the print of fgmres */
               this->_inner_iters_solver.SetPrintOption(-1);
               
               this->_inner_iters_solver.Setup(x, rhs);
            }
            
            break;
         }
         case kGemslrGlobalPrecondPSLR:
         {
            /* TBA */
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown Global partition option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      /* print the solver info when necessary */
      if(this->_print_option > 0)
      {
         this->PrintInfo();
      }
      
      this->_ready = true;
      this->_gemslr_setups._solve_phase_setup = kGemslrPhaseSolve;
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::Setup( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs);
   template int precond_gemslr_csr_par_double::Setup( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
   template int precond_gemslr_csr_par_complexs::Setup( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs);
   template int precond_gemslr_csr_par_complexd::Setup( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupPermutation()
   {
      /* Wrapper of the permutation part of GeMSLR 
       * This function setup the multilevel structure on each level
       * |B F|
       * |E C|
       * After calling this function we'll have:
       *    1. The _comm_helper for the permutation on each level. On GPU when solve on GPU.
       *    2. The E, B, F, and C matrices on each level. On CPU even if solve on GPU. Will move to GPU later.
       *    3. The S for the ESchur is not yet ready.
       */
      vector_int  map_v, mapptr_v;
      
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
      
      /* First step is to get the map vector
       * map_v: map_v[i] = j, the i-th node is in the j-th domain.
       * mapptr_v: the [start,end) domain number on each level.
       * 
       * For BJ, we only need one level.
       * 
       */
      
      switch(this->_global_precond_option)
      {
         case kGemslrGlobalPrecondBJ:
         {
            /* in this case only one level, use the input DD
             * The C solve option should not be BJ.
             */
            
            if(this->_gemslr_setups._level_setups._C_solve_option == kGemslrCSolveBJILUT)
            {
               this->_gemslr_setups._level_setups._C_solve_option = kGemslrCSolveILUT;
            }
            else
            {
               this->_gemslr_setups._level_setups._C_solve_option = kGemslrCSolveILUK;
            }
            
            /* just follow the input DD */
            this->_nlev_max = 1;
            this->_nlev_used = 1;
            map_v.Setup(this->_matrix->GetDiagMat().GetNumRowsLocal());
            map_v.Fill(myid);
            mapptr_v.Setup(2);
            mapptr_v[0] = 0;
            mapptr_v[1] = np;
            break;
         }
         case kGemslrGlobalPrecondPSLR:
         {
            /* NOT YET SUPPORTED */
            PARGEMSLR_ERROR("TO BE IMPLEMENTED.");
            break;
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            /* uses edge separator on the top level */
         
            int n_local;
            vector_int map2_v, mapptr2_v;
            MatrixType A, AT, AAT;
            
            /* reserve space */
            this->_levs_v.resize(this->_gemslr_setups._nlev_setup);
            
            if(!this->_gemslr_setups._global_partition_setup)
            {
               /* no global partition */
               /* partition the first level based on the IO order, save the remaining part in C_mat */
               this->_levs_v[0]._B_mat_v.resize(1);
               
               CsrMatrixClass<DataType> E, F;
               ParallelCsrMatrixSetupIOOrder( *this->_matrix, this->_levs_v[0]._pperm, this->_levs_v[0]._nI, 
                                                this->_levs_v[0]._B_mat_v[0], E, F, 
                                                this->_levs_v[0]._C_mat, this->_gemslr_setups._perm_option_setup, true);
               
               //this->_matrix->PlotPatternGnuPlot("A", 1);
               //this->_levs_v[0]._C_mat.PlotPatternGnuPlot("C", 1);
               
               this->_levs_v[0]._ncomps = 1;
               
               n_local  = this->_matrix->GetNumRowsLocal();
                
               /* check for global size */ 
               int nE = n_local - this->_levs_v[0]._nI, min_nE, min_nI;
               
               PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &nE, &min_nE, 1, MPI_MIN, comm));
               PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &this->_levs_v[0]._nI, &min_nI, 1, MPI_MIN, comm));
               
               /* switch to global reordering when bad partition encountered */
               if(min_nI == 0)
               {
                  //if(this->_levs_v[0]._nI == 0)
                  if(myid == 0)
                  {
                     PARGEMSLR_WARNING("Some subdomain has no interior nodes, switch to global reordering GeMSLR.");
                  }
                  
                  this->_levs_v[0].Clear();
                  this->_levs_v[0]._pperm.Clear();
                  
                  E.Clear();
                  F.Clear();
                  
                  //this->_gemslr_setups._global_partition_setup = true;
                  
                  return PARGEMSLR_RETURN_PARILU_NO_INTERIOR;
               }
               else
               {
                  
                  /* setup E and F on the top level */
                     
                  this->_levs_v[0]._E_mat.Setup(nE, this->_levs_v[0]._nI, *this->_matrix);
                  
                  this->_levs_v[0]._F_mat.Setup( this->_levs_v[0]._E_mat.GetNumColsLocal(),
                                          this->_levs_v[0]._E_mat.GetColStartGlobal(),
                                          this->_levs_v[0]._E_mat.GetNumColsGlobal(),
                                          this->_levs_v[0]._E_mat.GetNumRowsLocal(),
                                          this->_levs_v[0]._E_mat.GetRowStartGlobal(),
                                          this->_levs_v[0]._E_mat.GetNumRowsGlobal(),
                                          *this->_matrix);
                  
                  this->_levs_v[0]._E_mat.GetDiagMat() = std::move(E);
                  this->_levs_v[0]._E_mat.GetOffdMat().Setup(nE, 0, 0);
                  this->_levs_v[0]._E_mat.GetOffdMat().GetIVector().Fill(0);
                  this->_levs_v[0]._F_mat.GetDiagMat() = std::move(F);
                  this->_levs_v[0]._F_mat.GetOffdMat().Setup(this->_levs_v[0]._nI, 0, 0);
                  this->_levs_v[0]._F_mat.GetOffdMat().GetIVector().Fill(0);
                  
                  this->_levs_v[0]._E_mat.SetupMatvecStart();
                  this->_levs_v[0]._F_mat.SetupMatvecStart();
                  
                  MatrixType  &A = this->_levs_v[0]._C_mat;
               
                  MatrixType  AT, AAT;
                  
                  ParallelCsrMatrixTransposeHost( A, AT);
                  ParallelCsrMatrixAddHost( A, AT, AAT);
                  
                  AT.Clear();
                  
                  switch(this->_gemslr_setups._partition_option_setup)
                  {
                     case kGemslrPartitionRKway:
                     {
                        /* Recursive KWay, use all levels */
                        PARGEMSLR_FIRM_TIME_CALL( comm, PARGEMSLR_BUILDTIME_PARTITION, 
                              this->SetupPermutationRKway( AAT, this->_gemslr_setups._nlev_setup-1, this->_nlev_max, this->_nlev_used, map_v, mapptr_v));
                        
                        break;
                     }
                     case kGemslrPartitionND:
                     {
                        /* ND ordering, use all levels, not yet supported */
                        PARGEMSLR_FIRM_TIME_CALL( comm, PARGEMSLR_BUILDTIME_PARTITION, 
                              this->SetupPermutationND( AAT, this->_gemslr_setups._nlev_setup-1, this->_nlev_max, this->_nlev_used, map_v, mapptr_v));
                        
                        break;
                     }
                     default:
                     {
                        PARGEMSLR_ERROR("Unknown GeMSLR partition option.");
                        return PARGEMSLR_ERROR_INVALED_OPTION;
                     }
                  }
                  AAT.Clear();
                  
                  this->_nlev_max++;
                  this->_nlev_used++;
                  
               }
            }
            else
            {
               PARGEMSLR_ERROR("TO BE IMPLEMENTED.");
            }
            break;
         }
         case kGemslrGlobalPrecondGeMSLR:
         {
            /* recursive k-way partition */
            if(!this->_gemslr_setups._global_partition_setup)
            {
               /* no global partition */
               /* partition the first level based on the IO order */
               int i, ii, n_local, nI;
               vector_int local_perm, map2_v, mapptr2_v;
               MatrixType A, AT, AAT;
               CsrMatrixClass<DataType> E, B, F;
               
               n_local  = this->_matrix->GetNumRowsLocal();
                
               /* 1. Get global coupling matrix */
               
               /* get the permutation, local_perm[i] is the new i-th node in the old index */
               ParallelCsrMatrixSetupIOOrder( *this->_matrix, local_perm, nI, B, E, F, A, kIluReorderingNo, false);
               B.Clear();
               E.Clear();
               F.Clear();
               
               /* check for global size */
               int nE = n_local - nI, min_nE, min_nI;
               
               PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &nE, &min_nE, 1, MPI_MIN, comm));
               PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &nI, &min_nI, 1, MPI_MIN, comm));
               
               if(min_nI == 0)
               {
                  /* in this case, we have some domians that has no interior nodes */
                  if(nI == 0)
                  {
                     PARGEMSLR_WARNING("Some subdomain has no interior nodes, switch to global reordering.");
                  }
                  
                  A.Clear();
                  local_perm.Clear();
                  
                  this->_gemslr_setups._global_partition_setup = true;
                  
                  goto perm_gemslr_global;
                  
               }
               else
               {
                  /* Every MPI rank has exterior nodes */
                  ParallelCsrMatrixTransposeHost( A, AT);
                  ParallelCsrMatrixAddHost( A, AT, AAT);
                  
                  A.Clear();
                  AT.Clear();
                  
                  switch(this->_gemslr_setups._partition_option_setup)
                  {
                     case kGemslrPartitionRKway:
                     {
                        /* Recursive KWay, use all levels */
                        PARGEMSLR_FIRM_TIME_CALL( comm, PARGEMSLR_BUILDTIME_PARTITION, 
                              this->SetupPermutationRKway( AAT, this->_gemslr_setups._nlev_setup-1, this->_nlev_max, this->_nlev_used, map2_v, mapptr2_v));
                        break;
                     }
                     case kGemslrPartitionND:
                     {
                        /* ND ordering, use all levels, not yet supported */
                        PARGEMSLR_FIRM_TIME_CALL( comm, PARGEMSLR_BUILDTIME_PARTITION, 
                              this->SetupPermutationND( AAT, this->_gemslr_setups._nlev_setup-1, this->_nlev_max, this->_nlev_used, map2_v, mapptr2_v));
                        break;
                     }
                     default:
                     {
                        PARGEMSLR_ERROR("Unknown GeMSLR partition option.");
                        return PARGEMSLR_ERROR_INVALED_OPTION;
                     }
                  }
                  
                  AAT.Clear();
                  
                  /* 4. return level number */
                  
                  map_v.Setup(n_local);
                  mapptr_v.Setup(mapptr2_v.GetLengthLocal()+1);
                  
                  for(ii = 0 ; ii < nI ; ii ++)
                  {
                     i = local_perm[ii];
                     map_v[i] = myid;
                  }
                  
                  for(ii = nI ; ii < n_local ; ii ++)
                  {
                     i = local_perm[ii];
                     map_v[i] = map2_v[ii-nI]+np;
                  }
                  
                  mapptr_v[0] = 0;
                  for(i = 0 ; i < mapptr2_v.GetLengthLocal(); i++)
                  {
                     mapptr_v[i+1] = mapptr2_v[i] + np;
                     
                  }
                  
                  this->_nlev_max++;
                  this->_nlev_used++;
                  
                  map2_v.Clear();
                  mapptr2_v.Clear();
                  local_perm.Clear();
               }
            }
            else
            {

perm_gemslr_global:

               /* apply global partition */
               MatrixType  &A = *(this->_matrix);
               
               MatrixType  AT, AAT;
               
               ParallelCsrMatrixTransposeHost( A, AT);
               ParallelCsrMatrixAddHost( A, AT, AAT);
               
               AT.Clear();
               
               /* copy the partition array */
               AAT.GetSeparatorNumSubdomains() = A.GetSeparatorNumSubdomains();
               AAT.GetSeparatorDomi() = A.GetSeparatorDomi();
               
               switch(this->_gemslr_setups._partition_option_setup)
               {
                  case kGemslrPartitionRKway:
                  {
                     /* Recursive KWay, use all levels */
                     PARGEMSLR_FIRM_TIME_CALL( comm, PARGEMSLR_BUILDTIME_PARTITION, 
                           this->SetupPermutationRKway( AAT, this->_gemslr_setups._nlev_setup, this->_nlev_max, this->_nlev_used, map_v, mapptr_v));
                     break;
                  }
                  case kGemslrPartitionND:
                  {
                     /* ND ordering, use all levels, not yet supported */
                     PARGEMSLR_FIRM_TIME_CALL( comm, PARGEMSLR_BUILDTIME_PARTITION, 
                           this->SetupPermutationND( AAT, this->_gemslr_setups._nlev_setup, this->_nlev_max, this->_nlev_used, map_v, mapptr_v));
                     break;
                  }
                  default:
                  {
                     PARGEMSLR_ERROR("Unknown GeMSLR partition option.");
                     return PARGEMSLR_ERROR_INVALED_OPTION;
                  }
               }
               
               AAT.Clear();
               
            }
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown GeMSLR global partition option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      /* build level structure */
      switch(this->_global_precond_option)
      {
         case kGemslrGlobalPrecondBJ:
         {
            /* Now building the level structure by extracting E, B, F, and C matrices */
#ifdef PARGEMSLR_TIMING
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_STRUCTURE, this->SetupPermutationBuildLevelStructure( *this->_matrix, 0, map_v, mapptr_v));
#else
            this->SetupPermutationBuildLevelStructure( *this->_matrix, 0, map_v, mapptr_v);
#endif
            break;
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            /* need extra setup */
            
            /* Now building the level structure by extracting E, B, F, and C matrices */
#ifdef PARGEMSLR_TIMING
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_STRUCTURE, this->SetupPermutationBuildLevelStructure( this->_levs_v[0]._C_mat, 1, map_v, mapptr_v));
#else
            this->SetupPermutationBuildLevelStructure( this->_levs_v[0]._C_mat, 1, map_v, mapptr_v);
#endif
      
            /* now update the level pointer and domain pointer */
            this->_lev_ptr_v[0] = 0;
            for(int i = 1 ; i < this->_nlev_used+1 ; i ++)
            {
               this->_lev_ptr_v[i] += this->_levs_v[0]._nI;
            }
            
            this->_dom_ptr_v2[0].Setup(2);
            this->_dom_ptr_v2[0][0] = 0;
            this->_dom_ptr_v2[0][1] = this->_levs_v[0]._nI;
            
            for(int i = 1 ; i < this->_nlev_used ; i ++)
            {
               for(int j = 0 ; j < this->_dom_ptr_v2[i].GetLengthLocal() ; j ++)
               {
                  this->_dom_ptr_v2[i][j] += this->_levs_v[0]._nI;
               }
            }
            
            /* create buffer for the top level */
            
            this->_levs_v[0]._work_vector.Setup( (this->_matrix->GetNumRowsLocal())*7, this->_location, true);
            this->_levs_v[0]._work_vector_unit_length = this->_matrix->GetNumRowsLocal();
#ifdef PARGEMSLR_DEBUG
            this->_levs_v[0]._work_vector_occupied.Setup( 7, true);
#endif
            this->_levs_v[0]._x_temp.Setup( this->_matrix->GetNumRowsLocal(), this->_location, true, *this->_matrix);
            
            if(this->_gemslr_setups._global_partition_setup)
            {
               /* with global reordering option, a global A and the permutation vector is required 
                * Not yet supported
                */
            }
            break;
         }
         case kGemslrGlobalPrecondPSLR:
         {
            /* need extra setup
             * in this example, a new top-level structure is required
             */
            break;
         }
         case kGemslrGlobalPrecondGeMSLR:
         {
            
            /* Now building the level structure by extracting E, B, F, and C matrices */
#ifdef PARGEMSLR_TIMING
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_STRUCTURE, this->SetupPermutationBuildLevelStructure( *this->_matrix, 0, map_v, mapptr_v));
#else
            this->SetupPermutationBuildLevelStructure( *this->_matrix, 0, map_v, mapptr_v);
#endif
      
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown GeMSLR global partition option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      /* De allocation */
      map_v.Clear();
      mapptr_v.Clear();
      
      this->_pperm.Clear();
      this->_qperm.Clear();
      
      if(this->_print_option > 1)
      {
         this->PlotPatternGnuPlot("Parallel_GeMSLR_partition.data");
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::SetupPermutation();
   template int precond_gemslr_csr_par_double::SetupPermutation();
   template int precond_gemslr_csr_par_complexs::SetupPermutation();
   template int precond_gemslr_csr_par_complexd::SetupPermutation();
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupPermutationRKway(MatrixType &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v)
   {
      /* Recursive KWay partition */
      int         err = 0, tlvl;
      long int    n, num_dom, minsep, kmin, kfactor;
      bool        bj_last = false;
      
      /* prepare the level structure */
      
      n        = A.GetNumRowsGlobal();
      //num_dom  = 2;
      num_dom  = (int)(PargemslrMin((long int)(this->_gemslr_setups._ncomp_setup), n));
      
      //minsep   = 2;
      minsep   = (int)(PargemslrMin((long int)(pargemslr_global::_minsep), n));
      kmin     = this->_gemslr_setups._kmin_setup;
      kfactor  = this->_gemslr_setups._kfactor_setup;
      tlvl     = nlev_setup;
      
      if(this->_gemslr_setups._level_setups._C_solve_option == kGemslrCSolveBJILUT ||
         this->_gemslr_setups._level_setups._C_solve_option == kGemslrCSolveBJILUK)
      {
         bj_last = true;
      }
      
      /* apply RKway, obtain map vector */
      
      err = ParallelCsrMatrixSetupPermutationParallelRKway( A, this->_gemslr_setups._vertexsep_setup, tlvl, num_dom, minsep, kmin, kfactor, map_v, mapptr_v, bj_last); PARGEMSLR_CHKERR(err);
      
      /* we can't have C in parallel in this option */
      switch(this->_gemslr_setups._level_setups._C_solve_option)
      {
         case kGemslrCSolveILUT: case kGemslrCSolveILUK:
         {
            /* do nothing */
            break;
         }
         case kGemslrCSolveBJILUT: case kGemslrCSolveBJILUK:
         {
            if( tlvl > 1 && mapptr_v[tlvl] == mapptr_v[tlvl-1]+1)
            {
               /* we don't have enough nodes in the last couplint matrix */
               if(this->_gemslr_setups._level_setups._C_solve_option == kGemslrCSolveBJILUT)
               {
                  this->_gemslr_setups._level_setups._C_solve_option = kGemslrCSolveILUT;
               }
               else
               {
                  this->_gemslr_setups._level_setups._C_solve_option = kGemslrCSolveILUK;
               }
            }
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown Parallel GeMSLR C solve option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      /* 4. return level number */
      nlev_max = tlvl;
      nlev_used = tlvl;
      
      return err;
      
   }
   template int precond_gemslr_csr_par_float::SetupPermutationRKway(ParallelCsrMatrixClass<float> &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_par_double::SetupPermutationRKway(ParallelCsrMatrixClass<double> &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_par_complexs::SetupPermutationRKway(ParallelCsrMatrixClass<complexs> &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_par_complexd::SetupPermutationRKway(ParallelCsrMatrixClass<complexd> &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupPermutationND(MatrixType &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v)
   {
      /* ND partition */
      int         err = 0, tlvl;
      long int    n, minsep;

      /* prepare the level structure */
      
      n        = A.GetNumRowsGlobal();
      minsep   = (int)(PargemslrMin((long int)(pargemslr_global::_minsep), n));
      tlvl     = nlev_setup;
      
      /* apply ND order, obtain map vector */
      err = ParallelCsrMatrixSetupPermutationParallelND(A, this->_gemslr_setups._vertexsep_setup, tlvl, minsep, map_v, mapptr_v);
      
      /* return level number */
      nlev_max = tlvl;
      nlev_used = tlvl;
      
      
      /* we can't have C in parallel in this option */
      switch(this->_gemslr_setups._level_setups._C_solve_option)
      {
         case kGemslrCSolveILUT: case kGemslrCSolveILUK:
         {
            /* do nothing */
            break;
         }
         case kGemslrCSolveBJILUT:
         {
            /* switch to ILUT */
            this->_gemslr_setups._level_setups._C_solve_option = kGemslrCSolveILUT;
            break;
         }
         case kGemslrCSolveBJILUK:
         {
            /* switch to ILUT */
            this->_gemslr_setups._level_setups._C_solve_option = kGemslrCSolveILUK;
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown Parallel GeMSLR C solve option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      
      return err;
      
   }
   template int precond_gemslr_csr_par_float::SetupPermutationND(ParallelCsrMatrixClass<float> &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_par_double::SetupPermutationND(ParallelCsrMatrixClass<double> &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_par_complexs::SetupPermutationND(ParallelCsrMatrixClass<complexs> &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_par_complexd::SetupPermutationND(ParallelCsrMatrixClass<complexd> &A, int nlev_setup, int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupPermutationBuildLevelStructure(MatrixType &A, int level_start, vector_int &map_v, vector_int &mapptr_v)
   {
      
      /* define the data type */
      typedef DataType T;
      
      int                                    levi, levi2, i, ii, i1, i2, j, j1, j2, k, k1, k2;
      int                                    nlev_used, nlev_max, nblocks, ncomp, dom, dom1, dom2, dom3, idx, sends, recvs;
      int                                    n_local, n_in, n_out, nI, nE, nmax;
      long int                               n_start, c_start, col;
      vector_int                             n_locals;
      vector_long                            n_disps, e_starts, c_starts, e_starts2, f_starts, f_starts2;
      vector_int                             marker, dom_marker, node_proc;
      vector_int                             E_marker, F_marker, C_marker;
      vector_long                            mpi_sendsizes, mpi_recvsizes;
      std::vector<vector_int>                dom_ptr;
      CsrMatrixClass<T>                      A_diag_new;
      vector_int                             rcm_order;
      vector_int                             order, order2, rcm_order_vec, lev_ptr_vec;
      vector_long                            perm_vec_sorted, perm_vec_sorted2;
      std::vector<vector_int>                perm_vec_idx;
      
      std::vector<vector_long>               i_v, j_v, perm_v, dom_v;
      std::vector<SequentialVectorClass<T> > a_v;
      
      std::vector<vector_long>               i2_v, j2_v, perm2_v, dom2_v;
      std::vector<SequentialVectorClass<T> > a2_v;
      
      CooMatrixClass<T>                      C_offd;
      std::vector<CooMatrixClass<T> >        B_diags, E_diags, E_offds, F_diags, F_offds;
      
      MPI_Comm                               comm;
      int                                    myid, np, pid, reqid, upid, downid, idshift;
      std::vector<MPI_Request>               requests;
      std::vector<MPI_Status>                status;
      
      //MatrixType &A = *(this->_matrix);
      A.GetMpiInfo(np, myid, comm);
      
      CsrMatrixClass<T> &A_diag   = A.GetDiagMat();
      CsrMatrixClass<T> &A_offd   = A.GetOffdMat();
      vector_long &A_offd_map    = A.GetOffdMap();
      
      int   *A_diag_i = A_diag.GetI();
      int   *A_diag_j = A_diag.GetJ();
      T     *A_diag_data = A_diag.GetData();
      int   *A_offd_i = A_offd.GetI();
      int   *A_offd_j = A_offd.GetJ();
      T     *A_offd_data = A_offd.GetData();
      
      n_start = A.GetRowStartGlobal();
      n_local = A.GetNumRowsLocal();
      /* shift this value by 0 or 1, we might using different partition on the top level */
      nlev_used = this->_nlev_used - level_start;
      nlev_max = this->_nlev_max - level_start;
      n_in = n_local;
      
      
      /* Step 0: check if we should change C into BJ */
      
      /* Step 1: Send out matrices to other procs.
       * This step is to build the diagonal matrices and apply RCM ordering
       */
      
      /* setup levels */
      this->_levs_v.resize(nlev_used+level_start);
      
      /* setup parallel log for each level */
      for(i = 0 ; i < nlev_used ; i ++)
      {
         this->_levs_v[i+level_start]._parlog = A;
      }
      
      node_proc.Setup(n_local);/* the proc that this node belongs to */
      node_proc.Fill(-1);
      dom_ptr.resize(nlev_used);
      for(i = 0 ; i < nlev_used ; i ++)
      {
         dom_ptr[i].Setup(np+1);
      }
      
      /* Step 1.1: Get the map information
       * The map vector now holds the domain information
       * For each map value, decide the processor that it belongs to
       * node_proc will hold the rank id of each local nodes
       */
      for(levi = 0 ; levi < nlev_used ; levi++)
      {
         
         /* for the last level we treat differently */
         nblocks = (levi != nlev_used - 1) ? mapptr_v[levi+1] - mapptr_v[levi] : mapptr_v[nlev_max] - mapptr_v[levi];
         
         dom1 = nblocks / np;
         dom2 = nblocks % np;
         dom3 = dom1+1;
         
         dom_ptr[levi][0] = mapptr_v[levi];
         
         for(i = 0 ; i < dom2 ; i ++)
         {
            dom_ptr[levi][i+1] = dom_ptr[levi][i] + dom3;
         
         }
         for(i = dom2 ; i < np ; i ++)
         {
            dom_ptr[levi][i+1] = dom_ptr[levi][i] + dom1;
         }
         
         if(myid < dom2)
         {
            this->_levs_v[levi+level_start]._ncomps = dom3;
         }
         else
         {
            this->_levs_v[levi+level_start]._ncomps = dom1;
         }
         
         for(i = 0 ; i < n_local ; i ++)
         {
            if(node_proc[i] < 0)
            {
               /* this node haven't yet been assigned */
               dom = map_v[i];
               if( dom_ptr[levi].BinarySearch( dom, pid, true) < 0 )
               {
                  pid--;
               }
               
               /* check if this is out-of-bound 
                * for example, when search in 1 2 3 4 4, we found 4 at 4, but we need to drop it.
                */
               if(dom_ptr[levi][pid] == dom_ptr[levi][np])
               {
                  continue;
               }
               
               /* also note that we might have multiple same value, always use the first one */
               while(pid > 0 && dom_ptr[levi][pid] == dom_ptr[levi][pid-1])
               {
                  pid--;
               }
               
               if(pid >=0 && pid < np)
               {
                  /* this row belongs to pid */
                  node_proc[i] = pid;
               }
               
            }
         }
      }
      
      
      /* Step 1.2: Set up the communication
       * Prepare to send all the data out
       */
       
      /* 2D arrays holding i, j, and a for each proc. */
      i_v.resize(np);
      j_v.resize(np);
      a_v.resize(np);
      
      /* perm_v is the global idx of each row */
      perm_v.resize(np);
      /* the map value of each row */
      dom_v.resize(np);
      for(i = 0 ; i < np ; i ++)
      {
         i_v[i].Setup(1,true);
      }
      
      MPI_Barrier(comm);
      for(i = 0 ; i < n_local ; i ++)
      {
         /* this row belongs to this processor */
         pid = node_proc[i];
         dom1 = i_v[pid].GetLengthLocal();
         i_v[pid].PushBack(i_v[pid].Back());
         perm_v[pid].PushBack(i+n_start);
         dom_v[pid].PushBack(map_v[i]);
         
         /* insert diagonal */
         j1 = A_diag_i[i];
         j2 = A_diag_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            j_v[pid].PushBack(A_diag_j[j] + n_start);
            a_v[pid].PushBack(A_diag_data[j]);
            i_v[pid][dom1]++;
         }
         
         /* now offd */
         j1 = A_offd_i[i];
         j2 = A_offd_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            j_v[pid].PushBack(A_offd_map[A_offd_j[j]]);
            a_v[pid].PushBack(A_offd_data[j]);
            i_v[pid][dom1]++;
         }
      }
      
      MPI_Barrier(comm);
      /* Step 1.3: Apply the communication
       * Almost all-to-all, sort of expensive
       */
      mpi_sendsizes.Setup(np*2);
      mpi_recvsizes.Setup(np*2);
      for(i = 0 ; i < np ; i ++)
      {
         /* send I first */
         mpi_sendsizes[2*i] = perm_v[i].GetLengthLocal();
         mpi_sendsizes[2*i+1] = j_v[i].GetLengthLocal();
      }
      
      MPI_Barrier(comm);
      /* know the size of each communication */
      PARGEMSLR_MPI_CALL( MPI_Alltoall( mpi_sendsizes.GetData(), 2, MPI_LONG, mpi_recvsizes.GetData(), 2, MPI_LONG, comm) );
      
      MPI_Barrier(comm);
      
      /* or MPI_Alltoallv? */
      reqid = 0;
      requests.resize(np*10);
      for(i = 0 ; i < np ; i ++)
      {
         if(mpi_sendsizes[2*i] > 0)
         {
            /* myid have data for processor i */
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( i_v[i].GetData(), mpi_sendsizes[2*i]+1, i, 0, comm, &(requests[reqid++])) );
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( j_v[i].GetData(), mpi_sendsizes[2*i+1], i, 1, comm, &(requests[reqid++])) );
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( a_v[i].GetData(), mpi_sendsizes[2*i+1], i, 2, comm, &(requests[reqid++])) );
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( perm_v[i].GetData(), mpi_sendsizes[2*i], i, 3, comm, &(requests[reqid++])) );
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( dom_v[i].GetData(), mpi_sendsizes[2*i], i, 4, comm, &(requests[reqid++])) );
         }
      }
      
      i2_v.resize(np);
      j2_v.resize(np);
      a2_v.resize(np);
      perm2_v.resize(np);
      dom2_v.resize(np);
      n_local = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(mpi_recvsizes[2*i] > 0)
         {
            n_local += mpi_recvsizes[2*i];
            i2_v[i].Setup(mpi_recvsizes[2*i]+1);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( i2_v[i].GetData(), mpi_recvsizes[2*i]+1, i, 0, comm, &(requests[reqid++])) );
            j2_v[i].Setup(mpi_recvsizes[2*i+1]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( j2_v[i].GetData(), mpi_recvsizes[2*i+1], i, 1, comm, &(requests[reqid++])) );
            a2_v[i].Setup(mpi_recvsizes[2*i+1]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( a2_v[i].GetData(), mpi_recvsizes[2*i+1], i, 2, comm, &(requests[reqid++])) );
            perm2_v[i].Setup(mpi_recvsizes[2*i]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( perm2_v[i].GetData(), mpi_recvsizes[2*i], i, 3, comm, &(requests[reqid++])) );
            dom2_v[i].Setup(mpi_recvsizes[2*i]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( dom2_v[i].GetData(), mpi_recvsizes[2*i], i, 4, comm, &(requests[reqid++])) );
         }
      }
      
      PARGEMSLR_MPI_CALL( MPI_Waitall( reqid, requests.data(), MPI_STATUSES_IGNORE) );
      
      /* Step 1.4: Build structs
       * _lev_ptr_v => ptr of local levels
       * _dom_ptr_v2
       */
      n_out = n_local;
       
      this->_lev_ptr_v.Setup(nlev_used+1+level_start);
      this->_dom_ptr_v2.resize(nlev_used+level_start);
      
      /* local pointer to the start of each level */
      this->_lev_ptr_v[0+level_start] = 0;
      /* the permutation array, the global index of each local row */
      perm_vec_sorted.Setup(n_local, false);
      /* points to useful information 
       * perm_vec_idx[i][0] is the 
       * perm_vec_idx[i][1] is the 
       */
      perm_vec_idx.resize(n_local);
      /* set information level by level */
      for(levi = 0 ; levi < nlev_used ; levi ++)
      {
         levi2 = levi + 1;
         this->_lev_ptr_v[levi2+level_start] = this->_lev_ptr_v[levi+level_start];
         
         /* dom_ptr is the pointer to map value, dom_ptr[levi] holds the ptr to domains on each procs on this level
          * the domain id (map value) values on this level are between i1 to i2
          * now we need to have the _dom_ptr_v2 points to the actual local node numbers of each domain
          */
         i1 = dom_ptr[levi][myid];
         i2 = dom_ptr[levi][myid+1];
         
         this->_dom_ptr_v2[levi+level_start].Setup(i2-i1+1);
         this->_dom_ptr_v2[levi+level_start][0] = this->_lev_ptr_v[levi2+level_start];
         
         for(i = i1 ; i < i2 ; i ++)
         {
            /* loop through all rows received from other procs. */
            for(pid = 0 ; pid < np ; pid ++)
            {
               j1 = mpi_recvsizes[2*pid];
               for(j = 0 ; j < j1 ; j ++)
               {
                  dom = dom2_v[pid][j];
                  if(i == dom)
                  {
                     /* this is a entry in the current domain */
                     perm_vec_sorted[this->_lev_ptr_v[levi2+level_start]] = perm2_v[pid][j];
                     perm_vec_idx[this->_lev_ptr_v[levi2+level_start]].Setup(2);
                     /* mark the id this row belongs to */
                     perm_vec_idx[this->_lev_ptr_v[levi2+level_start]][0] = pid;
                     /* and mark the idx in the recv buffer */
                     perm_vec_idx[this->_lev_ptr_v[levi2+level_start]][1] = j;
                     this->_lev_ptr_v[levi2+level_start]++;
                  }
               }
            }
            this->_dom_ptr_v2[levi+level_start][i-i1+1] = this->_lev_ptr_v[levi2+level_start];
         }
      }
      
      /* make sure that the level structure is correct */
      PARGEMSLR_CHKERR(this->_lev_ptr_v[nlev_used+level_start] != n_local);
      
      /* sort ascending, stable */
      perm_vec_sorted.Sort( order, true, false);
      
      /* also need to modify perm_vec_sorted */
      perm_vec_sorted.Perm( order);
      
      /* Step 1.5: Extract diagonal blocks for RCM
       */
      if(this->_gemslr_setups._perm_option_setup == kIluReorderingNo)
      {
         rcm_order.Setup(n_local);
         rcm_order.UnitPerm();
      }
      else
      {
         if(A_diag.GetNumRowsLocal() > 0)
         {
            A_diag_new.Setup(n_local , n_local, (int) (A_diag.GetNumNonzeros()*((double)n_local/A_diag.GetNumRowsLocal())), true);
         }
         else
         {
            A_diag_new.Setup(n_local , n_local, n_local*5, true);
         }
         
         int   *A_diag_new_i = A_diag_new.GetI();
         
         for( i = 0 ; i < n_local ; i ++)
         {
            /* get the pid and j */
            pid = perm_vec_idx[i][0];
            j = perm_vec_idx[i][1];
            dom = dom2_v[pid][j];
            
            k1 = i2_v[pid][j];
            k2 = i2_v[pid][j+1];
            /* loop through this row */
            for(k = k1 ; k < k2 ; k ++)
            {
               col = j2_v[pid][k];
               if( perm_vec_sorted.BinarySearch( col, idx, true) >= 0)
               {
                  /* this is a local row, need to find the dom of it */
                  idx = order[idx];
                  if(dom == dom2_v[perm_vec_idx[idx][0]][perm_vec_idx[idx][1]])
                  {
                     /* this is a entry in the diagonal blocks, use for RCM */
                     A_diag_new.PushBack(idx, a2_v[pid][k]);
                  }
               }
            }
            A_diag_new_i[i+1] = A_diag_new.GetJVector().GetLengthLocal();
         }
         A_diag_new.SetNumNonzeros();
         
         /* Step 1.6: Apply the RCM
          * Note that each connected component would be kept
          * in their original order.
          */
         switch(this->_gemslr_setups._perm_option_setup)
         {
            case kIluReorderingNo:
            {
               /* do nothing */
               break;
            }
            case kIluReorderingRcm:
            {
               CsrMatrixRcmHost( A_diag_new, rcm_order);
               break;
            }
            case kIluReorderingAmd:
            {
               CsrMatrixAmdHost( A_diag_new, rcm_order);
               break;
            }
            case kIluReorderingNd:
            {
               CsrMatrixNdHost( A_diag_new, rcm_order);
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown local reordering option.");
               return PARGEMSLR_ERROR_INVALED_OPTION;
            }
         }
         
         A_diag_new.Clear();
      }
      
      /* Step 1.7: Update schur perm in precond struct
       */
      this->_pperm.Setup(n_local);
      for(i = 0 ; i < n_local ; i ++)
      {
         idx = rcm_order[i];
         this->_pperm[i] = perm2_v[perm_vec_idx[idx][0]][perm_vec_idx[idx][1]];
         perm2_v[perm_vec_idx[idx][0]][perm_vec_idx[idx][1]] = i;
      }
      
      /* Step 2: Exchange new global permutation information
       * This step is to get necessary global permutation information
       */
      
      /* Step 2.1: Get new distribution of matrices and build communication
       */
      
      n_locals.Setup(np);
      n_disps.Setup(np+1);
      
      PARGEMSLR_MPI_CALL( MPI_Allgather( &n_local, 1, MPI_INT, n_locals.GetData(), 1, MPI_INT, comm) );
      
      n_disps[0] = 0;
      for( i = 0 ; i < np ; i++)
      {
         n_disps[i+1] = n_disps[i] + n_locals[i];
      }
      
      this->_levs_v[level_start]._comm_helper.Clear();
      this->_levs_v[level_start]._comm_helper._n_in = n_in;
      this->_levs_v[level_start]._comm_helper._n_out = n_out;
      
      j2 = 0;
      for(i = 0 ; i < np ; i ++)
      {
         j1 = (int) mpi_sendsizes[2*i];
         if( j1 > 0)
         {
            j2++;
         }
      }
      
      this->_levs_v[level_start]._comm_helper._send_to_v.Setup(j2);
      this->_levs_v[level_start]._comm_helper._send_idx_v2.resize(j2);
      
      j2 = 0;
      for(i = 0 ; i < np ; i ++)
      {
         j1 = (int) mpi_sendsizes[2*i];
         if( j1 > 0)
         {
            /* myid have data for processor i */
            this->_levs_v[level_start]._comm_helper._send_to_v[j2] = i;
            this->_levs_v[level_start]._comm_helper._send_idx_v2[j2].Setup(j1);
            for(j = 0 ; j < j1 ; j ++)
            {
               /* this is the index in the original matrix */
               this->_levs_v[level_start]._comm_helper._send_idx_v2[j2][j] = (int)(perm_v[i][j]-n_start);
            }
            j2 ++;
         }
      }
      
      j2 = 0;
      for(i = 0 ; i < np ; i ++)
      {
         j1 = (int) mpi_recvsizes[2*i];
         if( j1 > 0)
         {
            j2++;
         }
      }
      
      this->_levs_v[level_start]._comm_helper._recv_from_v.Setup(j2);
      this->_levs_v[level_start]._comm_helper._recv_idx_v2.resize(j2);
      
      j2 = 0;
      for(i = 0 ; i < np ; i ++)
      {
         j1 = (int) mpi_recvsizes[2*i];
         if( j1 > 0)
         {
            /* myid have data from processor i */
            this->_levs_v[level_start]._comm_helper._recv_from_v[j2] = i;
            this->_levs_v[level_start]._comm_helper._recv_idx_v2[j2].Setup(j1);
            for(j = 0 ; j < j1 ; j ++)
            {
               /* this is the index in the original matrix */
               this->_levs_v[level_start]._comm_helper._recv_idx_v2[j2][j] = (int)(perm2_v[i][j]);
            }
            j2 ++;
         }
      }
      
      /* create buffer */
      sends = this->_levs_v[level_start]._comm_helper._send_to_v.GetLengthLocal();
      recvs = this->_levs_v[level_start]._comm_helper._recv_from_v.GetLengthLocal();
      
      this->_levs_v[level_start]._comm_helper._requests_v.resize(sends+recvs);
      
      this->_levs_v[level_start]._comm_helper.CreateHostBuffer(sizeof(T));
      
      this->_levs_v[level_start]._comm_helper._is_ready = true;
      
      this->_levs_v[level_start]._comm_helper.MoveData(this->_location);
      
      this->_levs_v[level_start]._sol_temp.Setup( n_out, this->_location, true, A);
      this->_levs_v[level_start]._rhs_temp.Setup(this->_levs_v[level_start]._sol_temp.GetLengthLocal(), 
                                                   this->_levs_v[level_start]._sol_temp.GetStartGlobal(), 
                                                   this->_levs_v[level_start]._sol_temp.GetLengthGlobal(), 
                                                   this->_location,
                                                   true,
                                                   A);
      
      this->_levs_v[level_start]._sol2_temp.Setup(this->_levs_v[level_start]._sol_temp.GetLengthLocal(), 
                                                   this->_levs_v[level_start]._sol_temp.GetStartGlobal(), 
                                                   this->_levs_v[level_start]._sol_temp.GetLengthGlobal(), 
                                                   this->_location,
                                                   true,
                                                   A);
      
      this->_levs_v[level_start]._rhs2_temp.Setup(this->_levs_v[level_start]._sol_temp.GetLengthLocal(), 
                                                   this->_levs_v[level_start]._sol_temp.GetStartGlobal(), 
                                                   this->_levs_v[level_start]._sol_temp.GetLengthGlobal(), 
                                                   this->_location,
                                                   true,
                                                   A);
                                                   
      this->_levs_v[level_start]._sol3_temp.Setup( n_in, this->_location, true, A);
      this->_levs_v[level_start]._rhs3_temp.Setup(this->_levs_v[level_start]._sol3_temp.GetLengthLocal(), 
                                                   this->_levs_v[level_start]._sol3_temp.GetStartGlobal(), 
                                                   this->_levs_v[level_start]._sol3_temp.GetLengthGlobal(), 
                                                   this->_location,
                                                   true,
                                                   A);
      
      /* Step 2.2: Setup matrices structure
       */
      
      /* setup parallel matrices and coo matrices on each level */
      
      B_diags.resize(nlev_used);
      E_diags.resize(nlev_used-1);
      E_offds.resize(nlev_used-1);
      F_diags.resize(nlev_used-1);
      F_offds.resize(nlev_used-1);
      
      e_starts.Setup(nlev_used-1);
      e_starts2.Setup(nlev_used-1);
      f_starts.Setup(nlev_used-1);
      f_starts2.Setup(nlev_used-1);
      
      for(levi = 0 ; levi < nlev_used; levi ++)
      {
         
         ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[levi+level_start];
         
         if( levi < nlev_used - 1)
         {
            /* not the last level */
            i1 = this->_lev_ptr_v[levi+level_start];
            i2 = this->_lev_ptr_v[levi+1+level_start];
            
            nI = i2 - i1;
            nE = n_local - i2;
            nmax = PargemslrMax(nI, nE);
            
            ncomp = level_str._ncomps;
            
            level_str._B_mat_v.resize(ncomp);
            
            /* setup the parallel matrix structure */
            level_str._E_mat.Setup( nE, nI, A);
            
            /* reuse those information for F */
            
            level_str._F_mat.Setup( level_str._E_mat.GetNumColsLocal(),
                                    level_str._E_mat.GetColStartGlobal(),
                                    level_str._E_mat.GetNumColsGlobal(),
                                    level_str._E_mat.GetNumRowsLocal(),
                                    level_str._E_mat.GetRowStartGlobal(),
                                    level_str._E_mat.GetNumRowsGlobal(),
                                    A);
            
            /* now create the coo matrix */
            
            e_starts[levi] = level_str._E_mat.GetColStartGlobal();
            f_starts[levi] = level_str._F_mat.GetColStartGlobal();
            
            B_diags[levi].Setup( nI, nI, (int) (A_diag.GetNumNonzeros()*((double)nI/A_diag.GetNumRowsLocal())));
            
            E_diags[levi].Setup( nE, nI, nmax);
            F_diags[levi].Setup( nI, nE, nmax);
            
            nmax *= (np-1);
            E_offds[levi].Setup( nE, INT_MAX, nmax);
            F_offds[levi].Setup( nI, INT_MAX, nmax);
            
            level_str._work_vector.Setup((nI + nE)*7, this->_location, true);
            level_str._work_vector_unit_length = nI + nE;
#ifdef PARGEMSLR_DEBUG
            level_str._work_vector_occupied.Setup(7,true);
#endif
            
            level_str._x_temp.Setup(nI + nE, this->_location, true, A);
            
         }
         else
         {
            /* the last level */
            
            ncomp = level_str._ncomps;
            
            level_str._B_mat_v.resize(ncomp);
            
            i1 = this->_lev_ptr_v[levi+level_start];
            i2 = this->_lev_ptr_v[levi+1+level_start];
            
            nI = i2 - i1;
            
            B_diags[levi].Setup( nI, nI, (int) (A_diag.GetNumNonzeros()*((double)nI/A_diag.GetNumRowsLocal())));
            
            level_str._work_vector.Setup(nI*7, this->_location, true);
            level_str._work_vector_unit_length = nI;
#ifdef PARGEMSLR_DEBUG
            level_str._work_vector_occupied.Setup(7,true);
#endif
            
            level_str._x_temp.Setup(nI, true, this->_location, A);
            
            /* store the row/col information in E and F */
            level_str._F_mat.Setup( nI, nI, A);
            level_str._E_mat.Setup( nI, nI, A);
            
            /* setup the C matrix for the last level */
            level_str._C_mat.Setup( nI, nI, A);
            
            c_starts.Setup(np+1);
            c_starts[np] = level_str._C_mat.GetNumRowsGlobal();
            
            c_start = level_str._C_mat.GetRowStartGlobal();
            PARGEMSLR_MPI_CALL( PargemslrMpiAllgather( &c_start, 1, c_starts.GetData(), comm) );
            
            C_offd.Setup( nI, INT_MAX, nI*(np-1));
         }
         
      }
      
      /* Step 2.3: Insert values into matrices
       */
      
      /* local first, sort local information */
      perm_vec_sorted.Setup( n_local);
      PARGEMSLR_MEMCPY( perm_vec_sorted.GetData(), this->_pperm.GetData(), n_local, kMemoryHost, kMemoryHost, long int);
      
      perm_vec_sorted.Sort(order, true, false);
      
      perm_vec_sorted.Perm(order);
      
      lev_ptr_vec.Setup( nlev_used+1);
      PARGEMSLR_MEMCPY( lev_ptr_vec.GetData(), this->_lev_ptr_v.GetData()+level_start, nlev_used+1, kMemoryHost, kMemoryHost, int);
      
      for(levi = 0 ; levi < nlev_used; levi ++)
      {
         
         i1 = this->_lev_ptr_v[levi+level_start];
         i2 = this->_lev_ptr_v[levi+1+level_start];
         
         /* insert value */
         for(ii = i1 ; ii < i2 ; ii ++)
         {
            /* get the pid and j */
            i = rcm_order[ii];
            pid = perm_vec_idx[i][0];
            j = perm_vec_idx[i][1];
            
            k1 = i2_v[pid][j];
            k2 = i2_v[pid][j+1];
            
            /* loop through this row */
            for(k = k1 ; k < k2 ; k ++)
            {
               col = j2_v[pid][k]; /* get the global index */
               
               if(perm_vec_sorted.BinarySearch( col, idx, true) >= 0)
               {
                  /* this is a local row, need to find the lev of it */
                  idx = order[idx];
                  
                  if(lev_ptr_vec.BinarySearch( idx, dom, true) < 0)
                  {
                     dom--;
                  }
               
                  /* also note that we might have multiple same value, always use the first one */
                  while(dom > 0 && lev_ptr_vec[dom] == lev_ptr_vec[dom-1])
                  {
                     dom--;
                  }
                  
                  if(dom == levi)
                  {
                     /* on the same level, for sure to be in the diagonal block */
                     B_diags[levi].PushBack( ii-i1, idx-i1, a2_v[pid][k]);
                     
                  }
                  else if(dom > levi)
                  {
                     /* on lower levels, belongs to F */
                     F_diags[levi].PushBack( ii-i1, idx-i2, a2_v[pid][k]);
                     
                  }
                  else
                  {
                     /* on upper levels, belongs to E */
                     E_diags[dom].PushBack( ii-this->_lev_ptr_v[dom+1+level_start], idx-this->_lev_ptr_v[dom+level_start], a2_v[pid][k]);
                  }
               }
            }
         }
      }
      
      /* now working with other procs. */
      for( idshift = 1 ; idshift < np ; idshift ++)
      {
         upid = (myid + idshift) % np;
         downid = (myid - idshift + np) % np;
         
         perm_vec_sorted2.Setup(n_locals[downid]);
         order2.Setup(n_locals[downid]);
         
         E_marker.Resize(n_locals[downid], false, false);
         E_marker.Fill(-1);
         
         C_marker.Resize(n_locals[downid], false, false);
         C_marker.Fill(-1);
         
         MPI_Sendrecv( perm_vec_sorted.GetData(), n_locals[myid], MPI_LONG, upid, myid*upid,
                        perm_vec_sorted2.GetData(), n_locals[downid], MPI_LONG, downid, downid*myid,
                        comm, MPI_STATUS_IGNORE);
         
         MPI_Sendrecv( order.GetData(), n_locals[myid], MPI_INT, upid, myid*upid,
                        order2.GetData(), n_locals[downid], MPI_INT, downid, downid*myid,
                        comm, MPI_STATUS_IGNORE);
         
         MPI_Sendrecv( this->_lev_ptr_v.GetData()+level_start, nlev_used + 1, MPI_INT, upid, myid*upid,
                        lev_ptr_vec.GetData(), nlev_used + 1, MPI_INT, downid, downid*myid,
                        comm, MPI_STATUS_IGNORE);
                        
         MPI_Sendrecv( e_starts.GetData(), nlev_used - 1, MPI_LONG, upid, myid*upid,
                        e_starts2.GetData(), nlev_used - 1, MPI_LONG, downid, downid*myid,
                        comm, MPI_STATUS_IGNORE);
         
         MPI_Sendrecv( f_starts.GetData(), nlev_used - 1, MPI_LONG, upid, myid*upid,
                        f_starts2.GetData(), nlev_used - 1, MPI_LONG, downid, downid*myid,
                        comm, MPI_STATUS_IGNORE);
         
         for(levi = 0 ; levi < nlev_used; levi ++)
         {
            /* reset marker array for F */
            F_marker.Resize(n_locals[downid], false, false);
            F_marker.Fill(-1);
            
            i1 = this->_lev_ptr_v[levi+level_start];
            i2 = this->_lev_ptr_v[levi+1+level_start];
            
            /* insert value */
            for(ii = i1 ; ii < i2 ; ii ++)
            {
               
               /* get the pid and j */
               i = rcm_order[ii];
               
               pid = perm_vec_idx[i][0];
               j = perm_vec_idx[i][1];
               
               k1 = i2_v[pid][j];
               k2 = i2_v[pid][j+1];
               
               /* loop through this row */
               for(k = k1 ; k < k2 ; k ++)
               {
                  col = j2_v[pid][k]; /* get the global index */
                  
                  if(perm_vec_sorted2.BinarySearch( col, idx, true) >= 0)
                  {
                     /* this is a local row of proc downid, need to find the lev of it 
                      * idx is the local idx on proc downid
                      */
                     
                     idx = order2[idx];
                     
                     /* now, search on the proc downid's levptr */
                     if(lev_ptr_vec.BinarySearch( idx, dom, true) < 0)
                     {
                        dom--;
                     }
                     
                     /* also note that we might have multiple same value, always use the first one */
                     while(dom > 0 && lev_ptr_vec[dom] == lev_ptr_vec[dom-1])
                     {
                        dom--;
                     }
                     
                     if(dom > levi)
                     {
                        /* on lower levels, belongs to F offd 
                         * need to find the global col index in f
                         * first, find the level, then, remove the level shift
                         * finally, add the global column start
                         */
                        col = idx - lev_ptr_vec[levi+1] + f_starts2[levi];
                        
                        if(F_marker[idx] < 0)
                        {
                           /* a new node */
                           this->_levs_v[levi+level_start]._F_mat.GetOffdMap().PushBack(col);
                           F_marker[idx] = this->_levs_v[levi+level_start]._F_mat.GetOffdMap().GetLengthLocal()-1;
                        }
                        
                        F_offds[levi].PushBack( ii-i1, F_marker[idx], a2_v[pid][k]);
                        
                     }
                     else if(dom < levi)
                     {
                        
                        /* on upper levels, belongs to E offd 
                         * also need to fine the global col index in e
                         */
                        col = idx - lev_ptr_vec[dom] + e_starts2[dom];
                        if(E_marker[idx] < 0)
                        {
                           /* a new node */
                           this->_levs_v[dom+level_start]._E_mat.GetOffdMap().PushBack(col);
                           E_marker[idx] = this->_levs_v[dom+level_start]._E_mat.GetOffdMap().GetLengthLocal()-1;
                        }
                        
                        E_offds[dom].PushBack( ii-this->_lev_ptr_v[dom+1+level_start], E_marker[idx], a2_v[pid][k]);
                     }
                     else
                     {
                        /* this for sure belongs to the offdiagonal of C, and for sure the last level */
                        PARGEMSLR_CHKERR(levi != nlev_used - 1);
                        
                        col = idx - lev_ptr_vec[levi] + c_starts[downid];
                        
                        if(C_marker[idx] < 0)
                        {
                           /* a new node */
                           this->_levs_v[levi+level_start]._C_mat.GetOffdMap().PushBack(col);
                           C_marker[idx] = this->_levs_v[levi+level_start]._C_mat.GetOffdMap().GetLengthLocal()-1;
                        }
                        
                        C_offd.PushBack( ii-this->_lev_ptr_v[levi+level_start], C_marker[idx], a2_v[pid][k]);
                     }
                  }
               }
            }
         }
      }
      
      /* Step 2.4: Transfer into CSR
       */
      
      for(levi = nlev_used - 1 ; levi >= 0; levi --)
      {
         ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[levi+level_start];
         
         if( levi < nlev_used - 1)
         {
            /* transfer coo to csr, this makes the coding a lot easier
             * B first, this is a littie bit more complex
             */
            CsrMatrixClass<DataType> B_csr;
            B_diags[levi].ToCsr(kMemoryHost, B_csr);
            
            ncomp = level_str._ncomps;
            
            for(i = 0 ; i < ncomp ; i ++)
            {
               k1 = this->_dom_ptr_v2[levi+level_start][i] - this->_dom_ptr_v2[levi+level_start][0];
               k2 = this->_dom_ptr_v2[levi+level_start][i+1] - this->_dom_ptr_v2[levi+level_start][i];
               B_csr.SubMatrix( k1, k1, k2, k2, kMemoryHost, level_str._B_mat_v[i]);
               //level_str._B_mat_v[i].SetComplexShift(this->_matrix->_diagonal_shift);
            }
            
            B_csr.Clear();
            
            E_diags[levi].ToCsr(kMemoryHost, level_str._E_mat.GetDiagMat());
            F_diags[levi].ToCsr(kMemoryHost, level_str._F_mat.GetDiagMat());
            
            E_offds[levi].SetNumCols(level_str._E_mat.GetOffdMap().GetLengthLocal());
            E_offds[levi].ToCsr(kMemoryHost, level_str._E_mat.GetOffdMat());
            level_str._E_mat.SortOffdMap();
            
            F_offds[levi].SetNumCols(level_str._F_mat.GetOffdMap().GetLengthLocal());
            F_offds[levi].ToCsr(kMemoryHost, level_str._F_mat.GetOffdMat());
            level_str._F_mat.SortOffdMap();
            
            level_str._E_mat.SetupMatvecStart();
            level_str._F_mat.SetupMatvecStart();
            
         }
         else
         {
            /* the last level */
            
            ncomp = level_str._ncomps;
            
            if(ncomp > 0)
            {
               CsrMatrixClass<DataType> &B_csr = level_str._C_mat.GetDiagMat();
               
               B_diags[levi].ToCsr(kMemoryHost, B_csr);
               C_offd.SetNumCols(level_str._C_mat.GetOffdMap().GetLengthLocal());
               C_offd.ToCsr(kMemoryHost, level_str._C_mat.GetOffdMat());
               
               for(i = 0 ; i < ncomp ; i ++)
               {
                  k1 = this->_dom_ptr_v2[levi+level_start][i] - this->_dom_ptr_v2[levi+level_start][0];
                  k2 = this->_dom_ptr_v2[levi+level_start][i+1] - this->_dom_ptr_v2[levi+level_start][i];
                  B_csr.SubMatrix( k1, k1, k2, k2, kMemoryHost, level_str._B_mat_v[i]);
                  //level_str._B_mat_v[i].SetComplexShift(this->_matrix->_diagonal_shift);
               }
            }
            
            /* sort offdiagonal rows */
            level_str._C_mat.SortOffdMap();
         }
         
      }
      
      /* Step 3: ??? TBA
       */
      
      /* Step 4: Finishing up everything
       */
      
      /* deallocate */
      for(i = 0 ; i < nlev_used ; i ++)
      {
         dom_ptr[i].Clear();
         B_diags[i].Clear();
      }
      std::vector<vector_int>().swap(dom_ptr);
      std::vector<CooMatrixClass<T> >().swap(B_diags);
      
      C_offd.Clear();
      for(i = 0 ; i < nlev_used-1 ; i ++)
      {
         E_diags[i].Clear();
         E_offds[i].Clear();
         F_diags[i].Clear();
         F_offds[i].Clear();
      }
      std::vector<CooMatrixClass<T> >().swap(E_diags);
      std::vector<CooMatrixClass<T> >().swap(E_offds);
      std::vector<CooMatrixClass<T> >().swap(F_diags);
      std::vector<CooMatrixClass<T> >().swap(F_offds);
      
      n_locals.Clear();
      marker.Clear();
      dom_marker.Clear();
      node_proc.Clear();
      E_marker.Clear();
      F_marker.Clear();
      C_marker.Clear();
      rcm_order.Clear();
      n_disps.Clear();
      e_starts.Clear();
      e_starts2.Clear();
      f_starts.Clear();
      f_starts2.Clear();
      c_starts.Clear();
      mpi_sendsizes.Clear();
      mpi_recvsizes.Clear();
      order.Clear();
      order2.Clear();
      rcm_order_vec.Clear();
      lev_ptr_vec.Clear();
      perm_vec_sorted.Clear();
      perm_vec_sorted2.Clear();
      
      for(i = 0 ; i < n_local ; i ++)
      {
         perm_vec_idx[i].Clear();
      }
      std::vector<vector_int>().swap(perm_vec_idx);
      
      for(pid = 0 ; pid < np ; pid ++)
      {
         i_v[pid].Clear();
         j_v[pid].Clear();
         perm_v[pid].Clear();
         dom_v[pid].Clear();
         a_v[pid].Clear();
         
         i2_v[pid].Clear();
         j2_v[pid].Clear();
         perm2_v[pid].Clear();
         dom2_v[pid].Clear();
         a2_v[pid].Clear();
      }
      
      std::vector<vector_long>().swap(i_v);
      std::vector<vector_long>().swap(j_v);
      std::vector<vector_long>().swap(perm_v);
      std::vector<vector_long>().swap(dom_v);
      std::vector<SequentialVectorClass<T> >().swap(a_v);
      
      std::vector<vector_long>().swap(i2_v);
      std::vector<vector_long>().swap(j2_v);
      std::vector<vector_long>().swap(perm2_v);
      std::vector<vector_long>().swap(dom2_v);
      std::vector<SequentialVectorClass<T> >().swap(a2_v);
      
      std::vector<MPI_Request>().swap(requests);
      std::vector<MPI_Status>().swap(status);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::SetupPermutationBuildLevelStructure(ParallelCsrMatrixClass<float> &A, int level_start, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_par_double::SetupPermutationBuildLevelStructure(ParallelCsrMatrixClass<double> &A, int level_start, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_par_complexs::SetupPermutationBuildLevelStructure(ParallelCsrMatrixClass<complexs> &A, int level_start, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_par_complexd::SetupPermutationBuildLevelStructure(ParallelCsrMatrixClass<complexd> &A, int level_start, vector_int &map_v, vector_int &mapptr_v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupBSolve( VectorType &x, VectorType &rhs)
   {
      /* Wrapper of the B solver/preconditioner setup 
       * After calling this function, when enable GPU:
       * B_mat , B solver, and B preconditioner would be on device.
       * C_mat would be on device
       * 
       * remaining things: E, F, matrices and the low-rank correction.
       */
      int         level, option;
      int         level_shift;
      
      /* Last level options:
       * If we use the BJ, we consider it as B solve (apply the top level option).
       * For other options, we consider it as C solve (apply the last level option).
       */
      if(this->_global_precond_option != kGemslrGlobalPrecondBJ)
      {
         level = this->_nlev_used - 1;
         
         switch(this->_gemslr_setups._level_setups._C_solve_option)
         {
            case kGemslrCSolveILUT:
            {
               this->SetupBSolveILUT( x, rhs, level);
               break;
            }
            case kGemslrCSolveILUK:
            {
               this->SetupBSolveILUK( x, rhs, level);
               break;
            }
            case kGemslrCSolveBJILUT:
            {
               this->SetupBSolveILUT( x, rhs, level);
               break;
            }
            case kGemslrCSolveBJILUK:
            {
               this->SetupBSolveILUK( x, rhs, level);
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Invalid C solve option.");
               return PARGEMSLR_ERROR_INVALED_OPTION;
            }
         }
         
         /* move the C on the last level to device */
         this->_levs_v[level]._C_mat.MoveData(this->_location);
         
      }
      else
      {
         level = 0;
         option = this->_gemslr_setups._level_setups._B_solve_option1;
         
         switch(option)
         {
            case kGemslrBSolveILUT:
            {
               this->SetupBSolveILUT( x, rhs, level);
               break;
            }
            case kGemslrBSolveILUK:
            {
               this->SetupBSolveILUK( x, rhs, level);
               break;
            }
            case kGemslrBSolveGemslr:
            {
               this->SetupBSolveGemslr( x, rhs, level);
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown Parallel GeMSLR B solve option.");
               return PARGEMSLR_ERROR_INVALED_OPTION;
            }
         }
      }
      
      /* setup levels */
      switch(this->_global_precond_option)
      {
         case kGemslrGlobalPrecondBJ:
         {
            level_shift = 0;
            break;
         }
         case kGemslrGlobalPrecondGeMSLR:
         {
            level_shift = 0;
            break;
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            level_shift = 1;
            /* setup the top level with Partial ILU */
            
            /* form the S mat with partial ILU */ 
            this->SetupPartialILUT(x, rhs);
            break;
         }
         case kGemslrGlobalPrecondPSLR:
         {
            level_shift = 1;
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown Global solve option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
         
      }
      
      /* setup remaining levels */
      for(level = level_shift ; level < this->_nlev_used-1 ; level++)
      {
         if(level < this->_gemslr_setups._level_setups._B_solve_option1_levels)
         {
            option = this->_gemslr_setups._level_setups._B_solve_option1;
         }
         else
         {
            option = this->_gemslr_setups._level_setups._B_solve_option2;
         }
         
         switch(option)
         {
            case kGemslrBSolveILUT:
            {
               this->SetupBSolveILUT( x, rhs, level);
               break;
            }
            case kGemslrBSolveILUK:
            {
               this->SetupBSolveILUK( x, rhs, level);
               break;
            }
            case kGemslrBSolveGemslr:
            {
               this->SetupBSolveGemslr( x, rhs, level);
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown Parallel GeMSLR B solve option.");
               return PARGEMSLR_ERROR_INVALED_OPTION;
            }
         }
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::SetupBSolve( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs);
   template int precond_gemslr_csr_par_double::SetupBSolve( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
   template int precond_gemslr_csr_par_complexs::SetupBSolve( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs);
   template int precond_gemslr_csr_par_complexd::SetupBSolve( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupBSolveILUT( VectorType &x, VectorType &rhs, int level)
   {
      /* define the data type */
      typedef DataType T;
      
      /* B solver/preconditioner setup with ILU */
      int         i, ncomp;
      /* not going to need those in the setup */
      SequentialVectorClass<T>  dummyx, dummyrhs;
      
      /* setup levels */
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      ncomp = level_str._ncomps;
      
      /* for MMUL, except for the last level, we use MILU as smoother 
       * If this is the last level or MMUL is not the option, we simply setup the B solver
       * Otherwise we do something else
       */
      /* start building the ILU on this level */
      PARGEMSLR_MALLOC( level_str._B_solver, ncomp, kMemoryHost, SolverClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*);

#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for(i = 0 ; i < ncomp ; i ++)
      {
         //if(this->_gemslr_setups._level_setups._lr_rankA_setup > 0)
         /* build ILU solver */
         PARGEMSLR_PLACEMENT_NEW( level_str._B_solver[i], kMemoryHost, IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
         IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> *ilup 
                     = (IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) level_str._B_solver[i];
         
         /* assign matrix */
         ilup->SetMatrix(level_str._B_mat_v[i]);
         
         /* options */
         if(level == this->_nlev_used-1)
         {
            ilup->SetMaxNnzPerRow(this->_gemslr_setups._level_setups._C_ilu_max_row_nnz_setup);
            ilup->SetDropTolerance(this->_gemslr_setups._level_setups._C_ilu_tol_setup);
         }
         else
         {
            ilup->SetMaxNnzPerRow(this->_gemslr_setups._level_setups._B_ilu_max_row_nnz_setup);
            ilup->SetDropTolerance(this->_gemslr_setups._level_setups._B_ilu_tol_setup);
         }
         
         /* set option */
         ilup->SetOption(kIluOptionILUT);
         
         /* diagonal complex shift */
         ilup->SetIluComplexShift(this->_gemslr_setups._level_setups._ilu_complex_shift);
         
         /* turn off level-scheduling solve */
         ilup->SetOpenMPOption(kIluOpenMPNo);
         
         ilup->SetPermutationOption(kIluReorderingNo);
         
         /* setup the solver */
         ilup->Setup(dummyx, dummyrhs);
         
         /* TODO: currently the setup phase is host only, we first setup and move the preconditioner later */
         ilup->SetSolveLocation(this->_location);
         level_str._B_mat_v[i].MoveData(this->_location);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::SetupBSolveILUT( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_par_double::SetupBSolveILUT( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_par_complexs::SetupBSolveILUT( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_par_complexd::SetupBSolveILUT( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupBSolveILUK( VectorType &x, VectorType &rhs, int level)
   {
      
      /* define the data type */
      typedef  DataType T;
      
      /* B solver/preconditioner setup with ILU */
      int         i, ncomp;
      /* not going to need those in the setup */
      SequentialVectorClass<T>  dummyx, dummyrhs;
      
      /* setup levels */
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      ncomp = level_str._ncomps;
      
      /* start building the ILU on this level */
      PARGEMSLR_MALLOC( level_str._B_solver, ncomp, kMemoryHost, SolverClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*);

#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for(i = 0 ; i < ncomp ; i ++)
      {
         //if(this->_gemslr_setups._level_setups._lr_rankA_setup > 0)
         /* build ILU solver */
         PARGEMSLR_PLACEMENT_NEW( level_str._B_solver[i], kMemoryHost, IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
         IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> *ilup 
                     = (IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) level_str._B_solver[i];
         
         /* assign matrix */
         ilup->SetMatrix(level_str._B_mat_v[i]);
         
         /* options */
         if(level == this->_nlev_used-1)
         {
            ilup->SetLevelOfFill(this->_gemslr_setups._level_setups._C_ilu_fill_level_setup);
         }
         else
         {
            ilup->SetLevelOfFill(this->_gemslr_setups._level_setups._B_ilu_fill_level_setup);
         }

         /* set option */
         ilup->SetOption(kIluOptionILUK);
         
         /* diagonal complex shift */
         ilup->SetIluComplexShift(this->_gemslr_setups._level_setups._ilu_complex_shift);
         
         /* turn off level-scheduling solve */
         ilup->SetOpenMPOption(kIluOpenMPNo);
         
         ilup->SetPermutationOption(kIluReorderingNo);
         
         /* setup the solver */
         ilup->Setup(dummyx, dummyrhs);
         
         /* TODO: currently the setup phase is host only, we first setup and move the preconditioner later */
         ilup->SetSolveLocation(this->_location);
         level_str._B_mat_v[i].MoveData(this->_location);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::SetupBSolveILUK( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_par_double::SetupBSolveILUK( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_par_complexs::SetupBSolveILUK( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_par_complexd::SetupBSolveILUK( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupBSolveGemslr( VectorType &x, VectorType &rhs, int level)
   {
      
      /* define the data type */
      typedef DataType T;
      
      /* B solver/preconditioner setup with ILU */
      int         i, ncomp;
      /* not going to need those in the setup */
      SequentialVectorClass<T>  dummyx, dummyrhs;
      
      /* setup levels */
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      ncomp = level_str._ncomps;
      
      /* start building the ILU on this level */
      PARGEMSLR_MALLOC( level_str._B_solver, ncomp, kMemoryHost, SolverClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*);

#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for(i = 0 ; i < ncomp ; i ++)
      {
         /* build ILU solver */
         PARGEMSLR_PLACEMENT_NEW( level_str._B_solver[i], kMemoryHost, GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
         GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> *gemslrp 
                     = (GemslrClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) level_str._B_solver[i];
         
         /* assign matrix */
         gemslrp->SetMatrix(level_str._B_mat_v[i]);
         
         this->SetLocalGemslr(*gemslrp);
         
         /* setup the solver */
         gemslrp->Setup(dummyx, dummyrhs);
         
         /* TODO: currently the setup phase is host only, we first setup and move the preconditioner later */
         gemslrp->SetSolveLocation(this->_location);
         level_str._B_mat_v[i].MoveData(this->_location);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::SetupBSolveGemslr( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_par_double::SetupBSolveGemslr( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_par_complexs::SetupBSolveGemslr( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_par_complexd::SetupBSolveGemslr( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupLowRank( VectorType &x, VectorType &rhs)
   {
      /* Wrapper of the building of the low-rank correction part of GeMSLR 
       * BJ: No low-rank correction.
       * 
       * GeMSLR:
       *    Normal:  Approximate G = EB^{-1}FC^{-1}.
       *       We have S = (I-G)C => S^{-1} = C^{-1}(I-G)^{-1}.
       *    Last level: When BJ-ILU enabled, create the PSLR-like low-rank correction.
       *       We have S = C_diag - (C_diag-C+EB^{-1}F) = C_diag - (EB^{-1}F-C_offd) = C_diag - Err.
       *       Denote this as C0 - R. We have S = C0(I-C0^{-1}R).
       *       S^{-1} = (I-C0^{-1}R)^{-1}C0^{-1} = [\sum{(C0^{-1}R)^i}]C0^{-1}.
       * 
       *       To apply the solve, simply do C0^{-1} + C0^{-1}RC0^{-1} + C0^{-1}RC0^{-1}RC0^{-1} + ...
       * 
       *       To add the low-rank approximation, we assume I = S[\sum{(C0^{-1}R)^i}]C0^{-1} + SErr2, this is
       *       S[\sum{(C0^{-1}R)^i}]C0^{-1} = (I-SErr2) => S^{-1} = [\sum{(C0^{-1}R)^i}]C0^{-1}(I-SErr2)^{-1}
       * 
       *       SErr2 = I - S[\sum{(C0^{-1}R)^i}]C0^{-1} = I - (C0-R)[\sum{(C0^{-1}R)^i}]C0^{-1}
       *       = I - C0[\sum{(C0^{-1}R)^i}]C0^{-1} + R[\sum{(C0^{-1}R)^i}]C0^{-1}
       *       = I - \sum C0(C0^{-1}R)^i)C0^{-1} + \sum R(C0^{-1}R)^i)C0^{-1}
       *       = I - \sum_{0-k} (R C0^{-1})^i - \sum{1-k+1} (R C0^{-1})^{i+1}
       *       = (R C0^{-1})^{k+1} = LR
       *       
       *       R = (EB^{-1}F-C_offd)
       * 
       *       After that, S^{-1} = [\sum{(C0^{-1}R)^i}]C0^{-1}(I-LR)^{-1}.
       * 
       *    RAP: S = RAP =               | B F |  | -B^{-1}F |                             | -B^{-1}F |
       *                  [-EB^{-1} I ]  | E C |  |     I    | = [E-EB^{-1}B C-EB^{-1}F ]  |     I    |
       *                 = (EB^{-1}B-E)B^{-1}F + C-EB^{-1}F = C - 2EB^{-1}F + EB^{-1}BB^{-1}F
       * 
       *    RAP-BJ: S = C_diag - (2EB^{-1}F - EB^{-1}BB^{-1}F - C_offd)
       * 
       * EsMSLR:
       *    Top level: We have S availiable. S = (I-(I-SC^{-1}))C => G = (I-SC^{-1})
       *    Other levels: Use the GeMSLR.
       *    What if top level is the last level?
       *       S = C_diag - (C_diag - S) => R = C_diag - S.
       * 
       */
      int option;
      
      /* explicit Schur complement */
      int         rank, level, lr_option;
      int         nlev_start = this->_nlev_used - 2;
      
      /* not going to need those in the setup */
      VectorType  dummyx, dummyrhs;
      
      /* setup low-rank correctin of the last level if BJ ILUK or BJ ILUT is used for C */
      if((this->_gemslr_setups._level_setups._C_solve_option == kGemslrCSolveBJILUK ||
          this->_gemslr_setups._level_setups._C_solve_option == kGemslrCSolveBJILUT) &&
          this->_gemslr_setups._level_setups._C_lr_pslr)
      {
         /* In this case, setup low-rank correction for the last level */
         level = this->_nlev_used - 1;
         nlev_start = this->_nlev_used - 3;
         
         ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
         
         /* in this case, the C matrix is used on device */
         level_str._C_mat.MoveData(this->_location);
         
         /* upper level E and F required here */
         if(level != 0)
         {
            ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &levelu_str = this->_levs_v[level-1];
         
            levelu_str._E_mat.SetupMatvecOver();
            levelu_str._F_mat.SetupMatvecOver();
            levelu_str._E_mat.MoveData(this->_location);
            levelu_str._F_mat.MoveData(this->_location);
         }
         
         /* other levels same as GeMSLR */
         option = kGemslrGlobalPrecondGeMSLR;
         level_str._EBFC.Setup(level, kGemslrGlobalPrecondPCLR, *this);
         
         rank = this->_gemslr_setups._level_setups._lr_rank2_setup;
         lr_option = this->_gemslr_setups._level_setups._lr_option2_setup;
      
         if(rank > 0)
         {
            /* only form when we have low-rank term */
            switch(lr_option)
            {
               case kGemslrLowrankNoRestart:
               {
                  level_str._lrc = this->SetupLowRankNoRestart(dummyx, dummyrhs, level, option);
                  break;
               }
               case kGemslrLowrankThickRestart:
               {
                  level_str._lrc = this->SetupLowRankThickRestart(dummyx, dummyrhs, level, option);
                  break;
               }
               case kGemslrLowrankSubspaceIteration:
               {
                  level_str._lrc = this->SetupLowRankSubspaceIteration(dummyx, dummyrhs, level, option);
                  break;
               }
               default:
               {
                  PARGEMSLR_ERROR("Unknown low-rank building option.");
                  return PARGEMSLR_ERROR_INVALED_OPTION;
               }
            }
            
            /*
            SequentialVectorClass<DataType> test_v;
            test_v.Setup(level_str._lrc);
            for(int zz = 0 ; zz < level_str._lrc ; zz ++)
            {
               test_v[zz] = DataType(1.0)/(level_str._Hk(zz,zz)+DataType(1.0));
            }
            test_v.Plot(parallel_log::_grank,0,6);
            */
            
            if(level_str._lrc > 0)
            {
               level_str._xlr_temp.Setup( level_str._C_mat.GetNumRowsLocal(), 
                                             level_str._C_mat.GetRowStartGlobal(), 
                                             level_str._C_mat.GetNumRowsGlobal(), 
                                             this->_location,
                                             true, 
                                             *this->_matrix);
               level_str._xlr1_temp.Setup(level_str._lrc, this->_location, true, *this->_matrix);
               level_str._xlr2_temp.Setup( level_str._xlr1_temp.GetLengthLocal(), 
                                             level_str._xlr1_temp.GetStartGlobal(), 
                                             level_str._xlr1_temp.GetLengthGlobal(),
                                             this->_location,
                                             true, 
                                             *this->_matrix);
               level_str._xlr1_temp_h.Setup( level_str._xlr1_temp.GetLengthLocal(), 
                                             level_str._xlr1_temp.GetStartGlobal(), 
                                             level_str._xlr1_temp.GetLengthGlobal(),
                                             kMemoryHost,
                                             true, 
                                             *this->_matrix);
               level_str._xlr2_temp_h.Setup( level_str._xlr1_temp.GetLengthLocal(), 
                                             level_str._xlr1_temp.GetStartGlobal(), 
                                             level_str._xlr1_temp.GetLengthGlobal(),
                                             kMemoryHost,
                                             true, 
                                             *this->_matrix);
            }
         }
      }
      
      /* setup levels */
      for(level = nlev_start ; level >= 0 ; level--)
      {
         ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
         
         level_str._E_mat.SetupMatvecOver();
         level_str._F_mat.SetupMatvecOver();
         level_str._E_mat.MoveData(this->_location);
         level_str._F_mat.MoveData(this->_location);
         
         switch(this->_global_precond_option)
         {
            case kGemslrGlobalPrecondBJ:
            {
               /* no LR for BJ */
               break;
            }
            case kGemslrGlobalPrecondESMSLR:
            {
               if(level == 0)
               {
                  /* different low-rank option on the top level */
                  option = kGemslrGlobalPrecondESMSLR;
                  level_str._EBFC.Setup(level, kGemslrGlobalPrecondESMSLR, *this);
               }
               else
               {
                  /* other levels same as GeMSLR */
                  option = kGemslrGlobalPrecondGeMSLR;
                  level_str._EBFC.Setup(level, kGemslrGlobalPrecondGeMSLR, *this);
               }
               
               break;
            }
            case kGemslrGlobalPrecondPSLR:
            {
               PARGEMSLR_ERROR("To be implemented.");
               break;
            }
            case kGemslrGlobalPrecondGeMSLR:
            {
               option = kGemslrGlobalPrecondGeMSLR;
               level_str._EBFC.Setup(level, kGemslrGlobalPrecondGeMSLR, *this);
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown GeMSLR global preconditioner for low-rank correction.");
               break;
            }
         }
         
         if(level == 0)
         {
            /* different low-rank option on the top level */
            rank = this->_gemslr_setups._level_setups._lr_rank1_setup;
            lr_option = this->_gemslr_setups._level_setups._lr_option1_setup;
         }
         else
         {
            rank = this->_gemslr_setups._level_setups._lr_rank2_setup;
            lr_option = this->_gemslr_setups._level_setups._lr_option2_setup;
         }
         
         if( rank > 0)
         {
            /* only form when we have low-rank term */
            switch(lr_option)
            {
               case kGemslrLowrankNoRestart:
               {
                  level_str._lrc = this->SetupLowRankNoRestart(dummyx, dummyrhs, level, option);
                  break;
               }
               case kGemslrLowrankThickRestart:
               {
                  level_str._lrc = this->SetupLowRankThickRestart(dummyx, dummyrhs, level, option);
                  break;
               }
               case kGemslrLowrankSubspaceIteration:
               {
                  level_str._lrc = this->SetupLowRankSubspaceIteration(dummyx, dummyrhs, level, option);
                  break;
               }
               default:
               {
                  PARGEMSLR_ERROR("Unknown low-rank building option.");
                  return PARGEMSLR_ERROR_INVALED_OPTION;
               }
            }
            
            if(level_str._lrc > 0)
            {
               level_str._xlr_temp.Setup( level_str._E_mat.GetNumRowsLocal(), 
                                             level_str._E_mat.GetRowStartGlobal(), 
                                             level_str._E_mat.GetNumRowsGlobal(), 
                                             this->_location,
                                             true, 
                                             *this->_matrix);
               level_str._xlr1_temp.Setup(level_str._lrc, this->_location, true, *this->_matrix);
               level_str._xlr2_temp.Setup( level_str._xlr1_temp.GetLengthLocal(), 
                                             level_str._xlr1_temp.GetStartGlobal(), 
                                             level_str._xlr1_temp.GetLengthGlobal(),
                                             this->_location,
                                             true, 
                                             *this->_matrix);
               level_str._xlr1_temp_h.Setup( level_str._xlr1_temp.GetLengthLocal(), 
                                             level_str._xlr1_temp.GetStartGlobal(), 
                                             level_str._xlr1_temp.GetLengthGlobal(),
                                             kMemoryHost,
                                             true, 
                                             *this->_matrix);
               level_str._xlr2_temp_h.Setup( level_str._xlr1_temp.GetLengthLocal(), 
                                             level_str._xlr1_temp.GetStartGlobal(), 
                                             level_str._xlr1_temp.GetLengthGlobal(),
                                             kMemoryHost,
                                             true, 
                                             *this->_matrix);
            }
         }
      }
      
      /* setup low-rank correction for the top level A */
      if(this->_gemslr_setups._level_setups._lr_rankA_setup > 0)
      //if(0)
      {
         /* we use AM^{-1}(I-X)^{-1} = I, which is X = I - AM^{-1} */
         
         /* 1st step, move matrix to the location */
         this->_matrix->MoveData(this->_location);
      
         switch(this->_global_precond_option)
         {
            case kGemslrGlobalPrecondBJ: case kGemslrGlobalPrecondESMSLR: case kGemslrGlobalPrecondPSLR: case kGemslrGlobalPrecondGeMSLR:
            {
               option = kGemslrGlobalPrecondGeMSLR;
               this->_lev_A[0]._EBFC.Setup(-1, kGemslrGlobalPrecondA, *this);
               this->_lev_A[0]._work_vector.Setup(this->_matrix->GetNumRowsLocal(), this->_location, true);
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown GeMSLR global preconditioner for low-rank correction.");
               break;
            }
         }
         
         rank = this->_gemslr_setups._level_setups._lr_rankA_setup;
         lr_option = this->_gemslr_setups._level_setups._lr_optionA_setup;
         
         if(rank > 0)
         {
            /* only form when we have low-rank term */
            switch(lr_option)
            {
               case kGemslrLowrankNoRestart:
               {
                  this->_lev_A[0]._lrc = this->SetupLowRankNoRestart(dummyx, dummyrhs, -1, option);
                  break;
               }
               case kGemslrLowrankThickRestart:
               {
                  this->_lev_A[0]._lrc = this->SetupLowRankThickRestart(dummyx, dummyrhs, -1, option);
                  break;
               }
               case kGemslrLowrankSubspaceIteration:
               {
                  this->_lev_A[0]._lrc = this->SetupLowRankSubspaceIteration(dummyx, dummyrhs, -1, option);
                  break;
               }
               default:
               {
                  PARGEMSLR_ERROR("Unknown low-rank building option.");
                  return PARGEMSLR_ERROR_INVALED_OPTION;
               }
            }
            
            if(this->_lev_A[0]._lrc > 0)
            {
               this->_lev_A[0]._xlr_temp.Setup( this->_matrix->GetNumRowsLocal(), 
                                             this->_matrix->GetRowStartGlobal(), 
                                             this->_matrix->GetNumRowsGlobal(), 
                                             this->_location,
                                             true, 
                                             *this->_matrix);
               this->_lev_A[0]._xlr1_temp.Setup(this->_lev_A[0]._lrc, this->_location, true, *this->_matrix);
               this->_lev_A[0]._xlr2_temp.Setup( this->_lev_A[0]._xlr1_temp.GetLengthLocal(), 
                                             this->_lev_A[0]._xlr1_temp.GetStartGlobal(), 
                                             this->_lev_A[0]._xlr1_temp.GetLengthGlobal(),
                                             this->_location,
                                             true, 
                                             *this->_matrix);
               this->_lev_A[0]._xlr1_temp_h.Setup( this->_lev_A[0]._xlr1_temp.GetLengthLocal(), 
                                             this->_lev_A[0]._xlr1_temp.GetStartGlobal(), 
                                             this->_lev_A[0]._xlr1_temp.GetLengthGlobal(),
                                             kMemoryHost,
                                             true, 
                                             *this->_matrix);
               this->_lev_A[0]._xlr2_temp_h.Setup( this->_lev_A[0]._xlr1_temp.GetLengthLocal(), 
                                             this->_lev_A[0]._xlr1_temp.GetStartGlobal(), 
                                             this->_lev_A[0]._xlr1_temp.GetLengthGlobal(),
                                             kMemoryHost,
                                             true, 
                                             *this->_matrix);
            }
         }
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::SetupLowRank( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs);
   template int precond_gemslr_csr_par_double::SetupLowRank( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
   template int precond_gemslr_csr_par_complexs::SetupLowRank( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs);
   template int precond_gemslr_csr_par_complexd::SetupLowRank( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupLowRankSubspaceIteration( VectorType &x, VectorType &rhs, int level, int option)
   {
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      typedef DataType T;
      
      /* Set up parameters */
      int                                    neig_c, neig_k, maxits, err;
      long int                               n_global;
      RealDataType                           lr_fact;
      DenseMatrixClass<T>                    V, H;
   
#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif
      /*------------------------------------------------------------------------
       * 1: Init Phase
       * Declare all variables and setup parameters
       * 
       * B  F
       * E  C
       * 
       * the size of the low rank correction on this level is of size of C
       *------------------------------------------------------------------------*/
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = (level < 0) ? this->_lev_A[0] : this->_levs_v[level];
      
      if(level >= 0)
      {
         n_global = level_str._E_mat.GetNumRowsGlobal();
      }
      else
      {
         n_global = this->_matrix->GetNumRowsGlobal();
      }
      
      if(level == 0)
      {
         /* we don't want to do more steps than the size of the matrix */
         maxits      = this->_gemslr_setups._level_setups._lr_maxits1_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor1_setup;
         neig_k      = this->_gemslr_setups._level_setups._lr_rank1_setup;
         neig_c      = int(neig_k * lr_fact);
         
         neig_c      = (int)PargemslrMin((long int)neig_c, n_global);
         neig_k      = PargemslrMin(neig_c, neig_k);
      }
      else if(level == -1)
      {
         /* we don't want to do more steps than the size of the matrix */
         maxits      = this->_gemslr_setups._level_setups._lr_maxitsA_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factorA_setup;
         neig_k      = this->_gemslr_setups._level_setups._lr_rankA_setup;
         neig_c      = int(neig_k * lr_fact);
         
         neig_c      = (int)PargemslrMin((long int)neig_c, n_global);
         neig_k      = PargemslrMin(neig_c, neig_k);
      }
      else
      {
         /* we don't want to do more steps than the size of the matrix */
         maxits      = this->_gemslr_setups._level_setups._lr_maxits2_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor2_setup;
         neig_k      = this->_gemslr_setups._level_setups._lr_rank2_setup;
         neig_c      = int(neig_k * lr_fact);
         
         neig_c      = (int)PargemslrMin((long int)neig_c, n_global);
         neig_k      = PargemslrMin(neig_c, neig_k);
      }
      
      /*------------------------ 
       * 2: Arnoldi and get result
       *------------------------*/
      
#ifdef PARGEMSLR_TIMING
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_ARNOLDI, PargemslrSubSpaceIteration<VectorType>( level_str._EBFC, neig_c, maxits, V, H, RealDataType()));
#else
      PargemslrSubSpaceIteration<VectorType>( level_str._EBFC, neig_c, maxits, V, H, RealDataType());
#endif

      /* free of V and H are handled inside */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_BUILD_RES, err = this->SetupLowRankBuildLowRank(x, rhs, V, H, neig_c, neig_k, level, option));
      
      return err;
   }
   template int precond_gemslr_csr_par_float::SetupLowRankSubspaceIteration( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, int level, int option);
   template int precond_gemslr_csr_par_double::SetupLowRankSubspaceIteration( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, int level, int option);
   template int precond_gemslr_csr_par_complexs::SetupLowRankSubspaceIteration( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, int level, int option);
   template int precond_gemslr_csr_par_complexd::SetupLowRankSubspaceIteration( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, int level, int option);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupLowRankNoRestart( VectorType &x, VectorType &rhs, int level, int option)
   {
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      typedef DataType T;
      
      /* Set up parameters */
      int                                    n, neig_c, neig_k, m, err;
      long int                               n_global;
      bool                                   rand_init;
      RealDataType                           normv, lr_fact, ar_fact, tol_orth, tol_reorth;
      DenseMatrixClass<T>                    V, H;
      ParallelVectorClass<T>                 v;
   
#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif
      /*------------------------------------------------------------------------
       * 1: Init Phase
       * Declare all variables and setup parameters
       * 
       * B  F
       * E  C
       * 
       * the size of the low rank correction on this level is of size of C
       * 
       * Note that when level == -1 we build low-rank correction for A
       * 
       *------------------------------------------------------------------------*/
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = (level < 0) ? this->_lev_A[0] : this->_levs_v[level];
      
      if(level >= 0)
      {
         n = level_str._E_mat.GetNumRowsLocal();
         n_global = level_str._E_mat.GetNumRowsGlobal();
      }
      else
      {
         n = this->_matrix->GetNumRowsLocal();
         n_global = this->_matrix->GetNumRowsGlobal();
      }
      
      if(level == 0)
      {
         /* we don't want to do more steps than the size of the matrix */
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factor1_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor1_setup;
         neig_k      = this->_gemslr_setups._level_setups._lr_rank1_setup;
         neig_c      = int(neig_k * ar_fact * lr_fact);
         
         neig_c      = (int)PargemslrMin((long int)neig_c, n_global);
         neig_k      = PargemslrMin(neig_c, neig_k);
      }
      else if(level == -1)
      {
         /* we don't want to do more steps than the size of the matrix */
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factorA_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factorA_setup;
         neig_k      = this->_gemslr_setups._level_setups._lr_rankA_setup;
         neig_c      = int(neig_k * ar_fact * lr_fact);
         
         neig_c      = (int)PargemslrMin((long int)neig_c, n_global);
         neig_k      = PargemslrMin(neig_c, neig_k);
      }
      else
      {
         /* we don't want to do more steps than the size of the matrix */
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor2_setup;
         neig_k      = this->_gemslr_setups._level_setups._lr_rank2_setup;
         neig_c      = int(neig_k * ar_fact * lr_fact);
         
         neig_c      = (int)PargemslrMin((long int)neig_c, n_global);
         neig_k      = PargemslrMin(neig_c, neig_k);
      }
      
      rand_init   = this->_gemslr_setups._level_setups._lr_rand_init_setup;
      tol_orth    = pargemslr_global::_orth_tol;
      tol_reorth  = pargemslr_global::_reorth_tol;
      
      /* create matrix V and H used in Arnoldi */
      V.Setup( n, neig_c+1, this->_location, true);
      H.Setup( neig_c+1, neig_c, kMemoryHost, true);
   
      /* setup init guess */
      if(level >= 0)
      {
         v.SetupPtrStr(level_str._E_mat);
      }
      else
      {
         v.SetupPtrStr(*this->_matrix);
      }
      v.UpdatePtr(V.GetData(), V.GetDataLocation() );
   
      if(rand_init)
      {
         /* random init guess */
         
         /* reset the seed */
         pargemslr_global::_mersenne_twister_engine.seed(0);
         
         v.Rand();
      }
      else
      {
         /* set to unit vector */
         v.Fill(T(1.0));
      }
      
      /* normalize v */
      v.Norm2(normv);
      v.Scale(T(1.0)/normv);
      
      /*------------------------ 
       * 2: Arnoldi and get result
       *------------------------*/
      
#ifdef PARGEMSLR_TIMING
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_ARNOLDI, m = PargemslrArnoldiNoRestart<VectorType>( level_str._EBFC, 0, neig_c, V, H, tol_orth, tol_reorth));
#else
      m = PargemslrArnoldiNoRestart<VectorType>( level_str._EBFC, 0, neig_c, V, H, tol_orth, tol_reorth);
#endif
      
      /* free of V and H are handled inside */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_BUILD_RES, err = this->SetupLowRankBuildLowRank(x, rhs, V, H, m, neig_k, level, option));
      
      return err;
   }
   template int precond_gemslr_csr_par_float::SetupLowRankNoRestart( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, int level, int option);
   template int precond_gemslr_csr_par_double::SetupLowRankNoRestart( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, int level, int option);
   template int precond_gemslr_csr_par_complexs::SetupLowRankNoRestart( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, int level, int option);
   template int precond_gemslr_csr_par_complexd::SetupLowRankNoRestart( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, int level, int option);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupLowRankThickRestart( VectorType &x, VectorType &rhs, int level, int option)
   {
      //if(this->_gemslr_setups._level_setups._lr_tol_eig_setup > 0.0)
      //{
         /* the eigenvalues are not accurate enough, do not lock them */
         return this->SetupLowRankThickRestartNoLock(x, rhs, level, option);
      //}
      //else
      //{
         /* lock convergenced eigenvalues */
      //   return this->SetupLowRankThickRestartStandard(x, rhs, level);
      //}
   }
   template int precond_gemslr_csr_par_float::SetupLowRankThickRestart( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, int level, int option);
   template int precond_gemslr_csr_par_double::SetupLowRankThickRestart( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, int level, int option);
   template int precond_gemslr_csr_par_complexs::SetupLowRankThickRestart( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, int level, int option);
   template int precond_gemslr_csr_par_complexd::SetupLowRankThickRestart( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, int level, int option);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupLowRankThickRestartNoLock( ParallelVectorClass<DataType> &x, ParallelVectorClass<DataType> &rhs, int level, int option)
   {
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      
      int                                                      n_local, rank, rank2, lr_m, m, maxsteps, maxits, err;
      long int                                                 n_global;
      bool                                                     rand_init;
      RealDataType                                             normv;
      DataType                                                 one;
      RealDataType                                             tol_eig, lr_fact, ar_fact, tr_fact, tol_orth, tol_reorth;
      
      DenseMatrixClass<DataType>                               V, H;
      ParallelVectorClass<DataType>                            v;
      
      one                        = DataType(1.0);
      
      /*------------------------
       * 1: Init Phase
       * Declare all variables and setup parameters
       *------------------------*/
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = (level < 0) ? this->_lev_A[0] : this->_levs_v[level];
      
      if(level >= 0)
      {
         n_local = level_str._E_mat.GetNumRowsLocal();
         n_global = level_str._E_mat.GetNumRowsGlobal();
      }
      else
      {
         n_local = this->_matrix->GetNumRowsLocal();
         n_global = this->_matrix->GetNumRowsGlobal();
      }
      
      if(level == 0)
      {
         /* we don't want to do more steps than the size of the matrix */
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factor1_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor1_setup;
         rank        = this->_gemslr_setups._level_setups._lr_rank1_setup;
         maxits      = this->_gemslr_setups._level_setups._lr_maxits1_setup;
         tol_eig     = this->_gemslr_setups._level_setups._lr_tol_eig1_setup;
      }
      else if(level == -1)
      {
         /* we don't want to do more steps than the size of the matrix */
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factorA_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factorA_setup;
         rank        = this->_gemslr_setups._level_setups._lr_rankA_setup;
         maxits      = this->_gemslr_setups._level_setups._lr_maxitsA_setup;
         tol_eig     = this->_gemslr_setups._level_setups._lr_tol_eigA_setup;
      }
      else
      {
         /* we don't want to do more steps than the size of the matrix */
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor2_setup;
         rank        = this->_gemslr_setups._level_setups._lr_rank2_setup;
         maxits      = this->_gemslr_setups._level_setups._lr_maxits2_setup;
         tol_eig     = this->_gemslr_setups._level_setups._lr_tol_eig2_setup;
      }
      
      rank2       = (int)(rank * lr_fact);
      
      rand_init   = this->_gemslr_setups._level_setups._lr_rand_init_setup;
      tol_orth    = pargemslr_global::_orth_tol;
      tol_reorth  = pargemslr_global::_reorth_tol;
      tr_fact     = pargemslr_global::_tr_factor;
      
      /* compute the actual rank to compute */
      //rank2 = (int)(rank * (RealDataType(1.0)+lr_rank_factor));
      //rank2 = 100;
      rank2 = PargemslrMax(rank2, rank);
      
      /* compute initial number of Arnoldi steps */
      lr_m = (int)(rank2 * ar_fact);
      lr_m = PargemslrMax(lr_m, rank2);
      
      /* we don't want to do more steps than the size of the matrix 
       * maxsteps is the maximun size of steps we can have,
       */
      maxsteps    = PargemslrMin( (long int)(lr_m + (lr_m * tr_fact)), n_global);
      
      rank        = PargemslrMin(rank, maxsteps);
      rank2       = PargemslrMin(rank2, maxsteps);
      
      /* create matrix V and H used in Arnoldi 
       * Note that V can be on the device.
       * H can be on the host.
       */
      V.Setup( n_local, maxsteps+1, this->_location, true);
      H.Setup( maxsteps+1, maxsteps, kMemoryHost, true);
      
      /* setup init guess */
      if(level >= 0)
      {
         v.SetupPtrStr(level_str._E_mat);
      }
      else
      {
         v.SetupPtrStr(*this->_matrix);
      }
      v.UpdatePtr(V.GetData(), V.GetDataLocation() );
      
      if(rand_init)
      {
         /* random init guess */
         
         /* reset the seed */
         pargemslr_global::_mersenne_twister_engine.seed(0);
         
         v.Rand();
      }
      else
      {
         /* set to unit vector */
         v.Fill(one);
      }
      
      /* normalize v */
      v.Norm2(normv);
      v.Scale(one/normv);
      
      /* apply Arnoldi thick-restart */
      m = PargemslrArnoldiThickRestartNoLock<VectorType>( level_str._EBFC, lr_m, maxits, rank2, rank, RealDataType(0.0), tr_fact, tol_eig, RealDataType(1.0), RealDataType(0.0), &(ParallelGemslrClass<MatrixType, VectorType, DataType>::ComputeDistance), V, H, tol_orth, tol_reorth);
      
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_BUILD_RES, err = this->SetupLowRankBuildLowRank(x, rhs, V, H, m, m, level, option) );
      
      return err;
      
   }
   template int precond_gemslr_csr_par_float::SetupLowRankThickRestartNoLock( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, int level, int option);
   template int precond_gemslr_csr_par_double::SetupLowRankThickRestartNoLock( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, int level, int option);
   template int precond_gemslr_csr_par_complexs::SetupLowRankThickRestartNoLock( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, int level, int option);
   template int precond_gemslr_csr_par_complexd::SetupLowRankThickRestartNoLock( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, int level, int option);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename RealDataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::OrdLowRank(int m, int &rank, RealDataType (*weight)(ComplexValueClass<RealDataType>), DenseMatrixClass<RealDataType> &R, DenseMatrixClass<RealDataType> &Q)
   {
      int                                       i;
      ComplexValueClass<RealDataType>           eig_val;
      DataType                                  zero;
      IntVectorClass<int>                       order, select;
      SequentialVectorClass<RealDataType>       wr, wi, w;
      
      zero = 0.0;
      
      /* Schur factorizition of R
       * note that R is from the H of Arnoldi, we can call HessSchur
       */
      PARGEMSLR_LOCAL_TIME_CALL(PARGEMSLR_BUILDTIME_DECOMP, R.HessSchur(Q, wr, wi));
      
      /* sort based on lambda/(1-lambda) */
      w.Setup(m);
      for(i = 0 ; i < m ; i ++)
      {
         if( i < m-1 && wi[i] > 0 && wi[i] == -wi[i+1])
         {
            /* in this case we have a pair of eigenvalues */
            eig_val = ComplexValueClass<RealDataType>( wr[i], wi[i] );
            w[i] = (*weight)(eig_val);
            i++;
            w[i] = (*weight)(eig_val);
         }
         else
         {
            /* in this case, only real eigenvalue */
            eig_val = ComplexValueClass<RealDataType>( wr[i], zero );
            w[i] = (*weight)(eig_val);
         }
      }
      
      /* Apply stable sort to make sure two continues eigenvalues
       * are still close to each other
       */
      w.Sort( order, false, true);
      
      select.Setup(m, true);
      for(i = 0 ; i < rank ; i ++)
      {
         select[order[i]] = 1;
      }
      
      /* In this case, the last eigenvalue is in pair, keep both of them */
      if(rank < m)
      {
         if( wi[order[rank-1]] > 0 && wi[order[rank-1]] == -wi[order[rank]])
         {
            select[order[rank]] = 1;
            rank++;
         }
      }
      
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, R.OrdSchur(Q, wr, wi, select));
      
      w.Clear();
      wr.Clear();
      wi.Clear();
      order.Clear();
      select.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::OrdLowRank<float>(int m, int &rank, float (*weight)(complexs), matrix_dense_float &R, matrix_dense_float &Q);
   template int precond_gemslr_csr_par_double::OrdLowRank<double>(int m, int &rank, double (*weight)(complexd), matrix_dense_double &R, matrix_dense_double &Q);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename RealDataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::OrdLowRank(int m, int &rank, RealDataType (*weight)(ComplexValueClass<RealDataType>), DenseMatrixClass<ComplexValueClass<RealDataType> > &R, DenseMatrixClass<ComplexValueClass<RealDataType> > &Q)
   {
      typedef ComplexValueClass<RealDataType> T;
      
      int                                 i;
      T                                   eig_val;
      IntVectorClass<int>                 order, select;
      SequentialVectorClass<RealDataType> w;
      SequentialVectorClass<T>            wi;
      /* Schur factorizition of R
       * note that R is from the H of Arnoldi, we can call HessSchur
       */
      PARGEMSLR_LOCAL_TIME_CALL(PARGEMSLR_BUILDTIME_DECOMP, R.HessSchur(Q, wi));
      
      /* sort based on lambda/(1-lambda) */
      w.Setup(m);
      for(i = 0 ; i < m ; i ++)
      {
         /* in this case, only real eigenvalue */
         w[i] = (*weight)(wi[i]);
      }
      
      /* Apply stable sort to make sure two continues eigenvalues
       * are still close to each other
       */
      w.Sort( order, false, true);
      
      select.Setup(m, true);
      for(i = 0 ; i < rank ; i ++)
      {
         select[order[i]] = 1;
      }
      
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, R.OrdSchur(Q, wi, select));
      
      w.Clear();
      wi.Clear();
      order.Clear();
      select.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_complexs::OrdLowRank<float>(int m, int &rank, float (*weight)(complexs), matrix_dense_complexs &R, matrix_dense_complexs &Q);
   template int precond_gemslr_csr_par_complexd::OrdLowRank<double>(int m, int &rank, double (*weight)(complexd), matrix_dense_complexd &R, matrix_dense_complexd &Q);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename T1, typename T2>
   T1 ParallelGemslrClass<MatrixType, VectorType, DataType>::ComputeDistance(T2 val)
   {
      return PargemslrAbs(val);
      
      /*
      T2 temp_val = T2(1.0) - val;
      if(PargemslrAbs(temp_val) < 1e-06)
      {
         //return PargemslrAbs( val * T2(1e06) );
         //return PargemslrAbs(val);
         //return PargemslrAbs( val * T2(1e06) ) + PargemslrAbs(val)*PargemslrAbs(val);
         return PargemslrAbs( val * T2(1e06) ) + PargemslrAbs(val);
      }
      else
      {
         //return PargemslrAbs( val / temp_val );
         //return PargemslrAbs(val);
         //return PargemslrAbs( val / temp_val ) + PargemslrAbs(val)*PargemslrAbs(val);
         return PargemslrAbs( val / temp_val ) + PargemslrAbs(val);
      }
      */
   }
   template float precond_gemslr_csr_par_float::ComputeDistance(complexs val);
   template double precond_gemslr_csr_par_double::ComputeDistance(complexd val);
   template float precond_gemslr_csr_par_complexs::ComputeDistance(complexs val);
   template double precond_gemslr_csr_par_complexd::ComputeDistance(complexd val);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename T1, typename T2>
   T1 ParallelGemslrClass<MatrixType, VectorType, DataType>::ComputeDistanceSC(T2 val)
   {
      /* we hope 1/val-1 to be large, thus, just pick val close to 0 */
      T2 temp_val = val;
      if(PargemslrAbs(temp_val) < 1e-06)
      {
         return PargemslrAbs( val * T2(1e06) );
      }
      else
      {
         return PargemslrAbs( val / temp_val );
      }
   }
   template float precond_gemslr_csr_par_float::ComputeDistanceSC(complexs val);
   template double precond_gemslr_csr_par_double::ComputeDistanceSC(complexd val);
   template float precond_gemslr_csr_par_complexs::ComputeDistanceSC(complexs val);
   template double precond_gemslr_csr_par_complexd::ComputeDistanceSC(complexd val);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupLowRankBuildLowRank( VectorType &x, VectorType &rhs, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, int m, int rank, int level, int option)
   {
      
      /* define the data type */
      typedef DataType T;
      
      int                              n, i, j;
      bool                             isutri;
      T                                one, zero;
      DenseMatrixClass<T>              W_temp, W_temp1, Q, Q_temp, R_temp;
      
      if( m == 0 )
      {
         /* nothing computed, return 0 */
         V.Clear();
         H.Clear();
         return 0;
      }
      
      one = T(1.0);
      zero = T();
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = (level < 0) ? this->_lev_A[0] : this->_levs_v[level];
      
      DenseMatrixClass<T>              &W = level_str._Wk;
      DenseMatrixClass<T>              &WH = level_str._WHk;
      DenseMatrixClass<T>              &R = level_str._Hk;
      
      n = V.GetNumRowsLocal();
      
      if( m <= rank )
      {
         /* In this case, we keep all of them
          * no need to turn R into diagonal matrix 
          */
         W.Setup( n, m, this->_location, false);
         R.Setup( m, m, kMemoryHost, false);
         
         /* copy useful parts to W and R */
         
         /* W first */
         PARGEMSLR_MEMCPY( W.GetData(), V.GetData(), n*m, W.GetDataLocation(), V.GetDataLocation(), T);
         
         switch(option)
         {
            case kGemslrGlobalPrecondGeMSLR:
            {
               /* R = I-H */
               for (i = 0; i < m; i++) 
               {
                  for (j = 0; j < m; j++)
                  {
                     if(i == j)
                     {
                        R(j,i) = one - H(j,i); 
                     }
                     else
                     {
                        R(j,i) = -H(j,i); 
                     }
                  }
               }
               break;
            }
            case kGemslrGlobalPrecondESMSLR:
            {
               /* H = X - I when level == 0 
                * -> X = H + I
                */
               if(level == 0)
               {
                  for (i = 0; i < m; i++) 
                  {
                     for (j = 0; j < m; j++)
                     {
                        if(i == j)
                        {
                           R(j,i) = one + H(j,i); 
                        }
                        else
                        {
                           R(j,i) = H(j,i); 
                        }
                     }
                  }
               }
               else
               {
                  for (i = 0; i < m; i++) 
                  {
                     for (j = 0; j < m; j++)
                     {
                        if(i == j)
                        {
                           R(j,i) = one - H(j,i); 
                        }
                        else
                        {
                           R(j,i) = -H(j,i); 
                        }
                     }
                  }
               }
               break;
            }
            case kGemslrGlobalPrecondBJ: default:
            {
               /* should not reach here */
               PARGEMSLR_ERROR("No low-rank correction for this option.");
               break;
            }
         }
         
         /* now compute  R = (I-H)^{-1}*/
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, R.Invert());
         
         /* finally, R = (I-H)^{-1} - I */
         for(i = 0; i < m ; i ++)
         {
            R(i, i) -= one;
         }
         
         /* move R to device when necessary */
         R.MoveData(this->_location);
         
         DenseMatrixMatMat( one, W, 'N', R, 'N', zero, WH);
         
         V.Clear();
         H.Clear();
         
         return m;
         
      }
      
      /* In this case, we only keep part of the eigenvalues, need to sort
       */
      
      W_temp.Setup(n, m, this->_location, false);
      R_temp.Setup(m, m, kMemoryHost, false);
      
      for (i = 0; i < m; i++) 
      {
         for (j = 0; j < m; j++)
         {
            R_temp(j,i) = H(j,i); 
         }
      }
      
      PARGEMSLR_MEMCPY( W_temp.GetData(), V.GetData(), n*m, W_temp.GetDataLocation(), V.GetDataLocation(), T);
      
      /* Schur factorizition of R */
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      this->OrdLowRank<RealDataType>(m, rank, &(ParallelGemslrClass<MatrixType, VectorType, DataType>::ComputeDistance), R_temp, Q);
      
      Q.MoveData(this->_location);
      Q_temp.SetupPtr(Q, 0, 0, rank, rank);
      W_temp1.SetupPtr(W_temp, 0, 0, n, rank);
      
      W.MatMat( one, W_temp1, 'N', Q_temp, 'N', T());
      
      R.Setup(rank, rank, kMemoryHost, false);
      
      isutri = true;
      switch(option)
      {
         case kGemslrGlobalPrecondGeMSLR:
         {
            /* R = I-R_temp(1:rank,1:rank) */
            for (i = 0; i < rank; i++) 
            {
               for (j = 0; j < rank; j++)
               {
                  if(i == j)
                  {
                     R(j,i) = one - R_temp(j,i); 
                  }
                  else
                  {
                     R(j,i) = -R_temp(j,i); 
                  }
                  if(i+1 == j)
                  {
                     if(R(j,i) != T())
                     {
                        isutri = false;
                     }
                  }
               }
            }
            break;
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            /* R = H when level == 0 */
            if(level == 0)
            {
               for (i = 0; i < rank; i++) 
               {
                  for (j = 0; j < rank; j++)
                  {
                     if(i == j)
                     {
                        R(j,i) = one + R_temp(j,i); 
                     }
                     else
                     {
                        R(j,i) = R_temp(j,i); 
                     }
                     if(i+1 == j)
                     {
                        if(R(j,i) != T())
                        {
                           isutri = false;
                        }
                     }
                  }
               }
            }
            else
            {
               for (i = 0; i < rank; i++) 
               {
                  for (j = 0; j < rank; j++)
                  {
                     if(i == j)
                     {
                        R(j,i) = one - R_temp(j,i); 
                     }
                     else
                     {
                        R(j,i) = -R_temp(j,i); 
                     }
                     if(i+1 == j)
                     {
                        if(R(j,i) != T())
                        {
                           isutri = false;
                        }
                     }
                  }
               }
            }
            break;
         }
         case kGemslrGlobalPrecondBJ: default:
         {
            /* should not reach here */
            PARGEMSLR_ERROR("No low-rank correction for this option.");
            break;
         }
      }
      
      /* now compute  R = (I-H)^{-1}*/
      if(isutri)
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, R.InvertUpperTriangular() );
      }
      else
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, R.Invert() );
      }
      
      /* finally, R = (I-H)^{-1} - I */
      for(i = 0; i < rank ; i ++)
      {
         R(i, i) -= one;
      }
      
      /* move R to device when necessary */
      R.MoveData(this->_location);
      
      DenseMatrixMatMat( one, W, 'N', R, 'N', zero, WH);
      
      V.Clear();
      H.Clear();
      W_temp.Clear();
      R_temp.Clear();
      Q.Clear();
      W_temp1.Clear();
      Q_temp.Clear();
      
      return rank;
   }
   template int precond_gemslr_csr_par_float::SetupLowRankBuildLowRank( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, int m, int rank, int level, int option);
   template int precond_gemslr_csr_par_double::SetupLowRankBuildLowRank( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, int m, int rank, int level, int option);
   template int precond_gemslr_csr_par_complexs::SetupLowRankBuildLowRank( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, int m, int rank, int level, int option);
   template int precond_gemslr_csr_par_complexd::SetupLowRankBuildLowRank( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, int m, int rank, int level, int option);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::Solve( VectorType &x, VectorType &rhs)
   {
      /* Wrapper */
      if(this->_lev_A[0]._lrc > 0)
      {
         DataType one = 1.0;
#ifdef PARGEMSLR_TIMING
         int np, myid;
         MPI_Comm comm;
         this->_matrix->GetMpiInfo(np, myid, comm);
#endif
         /* In this case, apply the low-rank correction first, M^{-1}(I+LR)x */
         /* apply low-rank */
#ifdef PARGEMSLR_TIMING
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_LRC, (this->SolveApplyLowRankLevel( this->_lev_A[0]._xlr_temp, rhs, -1)));
#else
         this->SolveApplyLowRankLevel( this->_lev_A[0]._xlr_temp, rhs, -1);
#endif
         this->_lev_A[0]._xlr_temp.Axpy( one, rhs);
      }
      
      VectorType &temp_rhs = (this->_lev_A[0]._lrc > 0) ? this->_lev_A[0]._xlr_temp : rhs ;
      
      switch(this->_global_precond_option)
      {
         case kGemslrGlobalPrecondBJ: case kGemslrGlobalPrecondGeMSLR:
         {
            /* standard GeMSLR solve */
            if(this->_gemslr_setups._solve_option_setup == kGemslrMulSolve)
            {
               /* multiplicative solve */
               this->SolveLevelGemslrMul( x, temp_rhs, 0, true);
            }
            else
            {
               this->SolveLevelGemslr( x, temp_rhs, 0, true);
            }
            break;
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            /* explicite Schur complement */
            this->SolveLevelEsmslr( x, temp_rhs, 0, true);
            break;
         }
         case kGemslrGlobalPrecondPSLR:
         {
            /* PSLR solve */
            this->SolveLevelPslr( x, temp_rhs, 0, true);
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Invalid global solve option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
            break;
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::Solve( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs);
   template int precond_gemslr_csr_par_double::Solve( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
   template int precond_gemslr_csr_par_complexs::Solve( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs);
   template int precond_gemslr_csr_par_complexd::Solve( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SolveLevelGemslr( VectorType &x_out, VectorType &rhs_in, int level, bool doperm)
   {
      
      /* Buffer used: 
       * 1. _rhs_temp and _sol_temp if permutation enabled.
       * 2. 1, 2 used when solving last level.
       * 3. 2 used when solving other levels, 3 and 4 used in inner iteration.
       */
      
      /* define the data type */
      typedef DataType T;

#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif
      
      /* the solve phase of GeMSLR */
      
      PARGEMSLR_CHKERR(level < 0);
      PARGEMSLR_CHKERR(level >= this->_nlev_used);
      
      int n_local;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      /* might need to apply permutation 
       * This is always done on the top level
       * and on lower levels if edge separator is used
       */
      
      bool apply_perm = level_str._comm_helper._is_ready && doperm;
      
      if(apply_perm)
      {
         
         /* note that we are solving the equation with Apq, we might need to apply the permutation 
          * Ax = b => PAQQ'x = Pb, we are solving with Q'x
          * 
          * Note on the permutation:
          * 
          * swap row: left,  P(i, pperm[i]) = 1;
          * swap col: right, Q(qperm[i], i) = 1;
          * 
          * Q(qperm[i], i) = 1 => left apply Q is equal to Q(i, rqperm[i])
          *                    => Q' has Q'(i, qperm[i]) = 1;
          * 
          */
         level_str._comm_helper.DataTransfer(rhs_in, level_str._rhs_temp, this->_location, this->_location);
         level_str._sol_temp.Fill(0.0);
      }
      
      VectorType &x = apply_perm? level_str._sol_temp : x_out;
      VectorType &rhs = apply_perm? level_str._rhs_temp : rhs_in;
      
      if(level == this->_nlev_used - 1)
      {
         
         /* in this case we apply solve on the last level, a B solve */
#ifdef PARGEMSLR_TIMING
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU_L, this->SolveB(x, rhs, 0, level));
         }
         else
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_ILUT_L, this->SolveB(x, rhs, 0, level));
         }
#else
         this->SolveB(x, rhs, 0, level);
#endif
         
         /* might need to apply permutation 
          * This is always done on the top level
          * and on lower levels if edge separator is used
          */
         if(apply_perm)
         {
            level_str._comm_helper.DataTransferReverse(level_str._sol_temp, x_out, this->_location, this->_location);
         }
         
         return PARGEMSLR_SUCCESS;
         
      }
      
      n_local = level_str._F_mat.GetNumRowsLocal();
      
      T one, zero, mone;
      
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      VectorType z;
      VectorType rhsu, rhsl, xu, xl, zu, zl;
      
      rhsu.SetupPtrStr(x);
      z.UpdatePtr(level_str._work_vector.GetData() + level_str._work_vector_unit_length * 2, this->_location);
      
#ifdef PARGEMSLR_DEBUG
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[2]);
      level_str._work_vector_occupied[2] = 1;
#endif

      z.Fill(zero);
      rhsu.SetupPtrStr(level_str._F_mat);
      rhsu.UpdatePtr(rhs.GetData(), this->_location);
      rhsl.SetupPtrStr(level_str._E_mat);
      rhsl.UpdatePtr(rhs.GetData()+n_local, this->_location);
      xu.SetupPtrStr(level_str._F_mat);
      xu.UpdatePtr(x.GetData(), this->_location);
      xl.SetupPtrStr(level_str._E_mat);
      xl.UpdatePtr(x.GetData()+n_local, this->_location);
      zu.SetupPtrStr(level_str._F_mat);
      zu.UpdatePtr(z.GetData(), this->_location);
      zl.SetupPtrStr(level_str._E_mat);
      zl.UpdatePtr(z.GetData()+n_local, this->_location);
      
      if(this->_gemslr_setups._solve_option_setup == kGemslrUSolve && this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSolve)
      {
         /* skip the block L solve */
         /* Step 1: z = rhs */
         PARGEMSLR_MEMCPY(z.GetData(), rhs.GetData(), rhs.GetLengthLocal(), kMemoryHost, kMemoryHost, DataType);
      }
      else
      {
#ifdef PARGEMSLR_TIMING
         /* Step 1: z_u = U\L\rhs_u */
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU, (this->SolveB(zu, rhsu, 0, level)));
         }
         else
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_ILUT, (this->SolveB(zu, rhsu, 0, level)));
         }
#else
         this->SolveB(zu, rhsu, 0, level);
#endif
         
         /* Step 2: z_l = -E * z_u */
#ifdef PARGEMSLR_TIMING
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_EMV, (level_str._E_mat.MatVec( 'N', mone, zu, zero, zl)));
         }
         else
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_EMV, (level_str._E_mat.MatVec( 'N', mone, zu, zero, zl)));
         }
#else
         level_str._E_mat.MatVec( 'N', mone, zu, zero, zl);
#endif
         
         /* Step 3: z_l = rhs_l + z_l = rhs_l - E * z_u */
         zl.Axpy(one, rhsl);
      }
      
      /* Step 4: apply the low-rank correction 
       * x_l = C\z_l + W*H*W' * z_l
       */
      if( this->_gemslr_setups._enable_inner_iters_setup && this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSolve && level == 0)
      {
         /* solve with 0 as initial guess 
          * For the GeMSLR option, we use the SchurMatVec function.
          * buffer 3 and 4 used
          */
         xl.Fill(zero);
         this->_inner_iters_solver.Solve(xl, zl);
      }
      else
      {
         if( level_str._lrc > 0)
         {
            /* apply the low-rank update */
#ifdef PARGEMSLR_TIMING
            if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
            {
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELR, (this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level)));
            }
            else
            {
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_LRC, (this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level)));
            }
#else
            this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level);
#endif
            zl.Axpy( one, level_str._xlr_temp);
         }
         this->SolveLevelGemslr( xl, zl, level+1, true);
      }
      
      /* Step 5: z_u = - F * x_l */
      
#ifdef PARGEMSLR_TIMING
      if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_FMV, (level_str._F_mat.MatVec( 'N', mone, xl, zero, zu)));
      }
      else
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_FMV, (level_str._F_mat.MatVec( 'N', mone, xl, zero, zu)));
      }
#else
      level_str._F_mat.MatVec( 'N', mone, xl, zero, zu);
#endif
      /* Step 6: z_u = rhs_u - F * x_l */
      zu.Axpy(one, rhsu);
      
      /* Step 7: x_u = U\L\z_u */
#ifdef PARGEMSLR_TIMING
      if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU, (this->SolveB(xu, zu, 0, level)));
      }
      else
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_ILUT, (this->SolveB(xu, zu, 0, level)));
      }
#else
      this->SolveB(xu, zu, 0, level);
#endif

      /* might need to apply permutation 
       * This is always done on the top level
       * and on lower levels if edge separator is used
       */
      if(apply_perm)
      {
         level_str._comm_helper.DataTransferReverse(level_str._sol_temp, x_out, this->_location, this->_location);
      }
      
#ifdef PARGEMSLR_DEBUG
      level_str._work_vector_occupied[2] = 0;
#endif

      z.Clear();
      rhsu.Clear();
      rhsl.Clear();
      xu.Clear();
      xl.Clear();
      zu.Clear();
      zl.Clear();
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::SolveLevelGemslr( ParallelVectorClass<float> &x_out, ParallelVectorClass<float> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_double::SolveLevelGemslr( ParallelVectorClass<double> &x_out, ParallelVectorClass<double> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_complexs::SolveLevelGemslr( ParallelVectorClass<complexs> &x_out, ParallelVectorClass<complexs> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_complexd::SolveLevelGemslr( ParallelVectorClass<complexd> &x_out, ParallelVectorClass<complexd> &rhs_in, int level, bool doperm);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SolveLevelGemslrMul( VectorType &x_out, VectorType &rhs_in, int level, bool doperm)
   {
      
      /* Buffer used: 
       * 1. _rhs_temp and _sol_temp if permutation enabled.
       * 2. 1, 2 used when solving last level.
       * 3. 1, 2 used when solving other levels, 3 and 4 used in inner iteration.
       */
      
      /* define the data type */
      typedef DataType T;

#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif
      
      /* the solve phase of GeMSLR */
      
      PARGEMSLR_CHKERR(level < 0);
      PARGEMSLR_CHKERR(level >= this->_nlev_used);
      
      int n_local;
      int solve_option = 0;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      /* might need to apply permutation 
       * This is always done on the top level
       * and on lower levels if edge separator is used
       */
      bool apply_perm = level_str._comm_helper._is_ready && doperm;
      if(apply_perm)
      {
         
         /* note that we are solving the equation with Apq, we might need to apply the permutation 
          * Ax = b => PAQQ'x = Pb, we are solving with Q'x
          * 
          * Note on the permutation:
          * 
          * swap row: left,  P(i, pperm[i]) = 1;
          * swap col: right, Q(qperm[i], i) = 1;
          * 
          * Q(qperm[i], i) = 1 => left apply Q is equal to Q(i, rqperm[i])
          *                    => Q' has Q'(i, qperm[i]) = 1;
          * 
          */
         level_str._comm_helper.DataTransfer(rhs_in, level_str._rhs_temp, this->_location, this->_location);
         level_str._sol_temp.Fill(0.0);
      }
      
      VectorType &x = apply_perm? level_str._sol_temp : x_out;
      VectorType &rhs = apply_perm? level_str._rhs_temp : rhs_in;
      
      if(level == this->_nlev_used - 1)
      {
         /* in this case we apply solve on the last level, a B solve */
#ifdef PARGEMSLR_TIMING
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU_L, this->SolveB(x, rhs, 0, level));
         }
         else
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_ILUT_L, this->SolveB(x, rhs, 0, level));
         }
#else
         this->SolveB(x, rhs, 0, level);
#endif
         
         /* might need to apply permutation 
          * This is always done on the top level
          * and on lower levels if edge separator is used
          */
         if(apply_perm)
         {
            level_str._comm_helper.DataTransferReverse(level_str._sol_temp, x_out, this->_location, this->_location);
         }
         
         return PARGEMSLR_SUCCESS;
         
      }
      
      n_local = level_str._F_mat.GetNumRowsLocal();
      
      T one, zero, mone;
      
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      VectorType y, z, rhsu, rhsl, xu, xl, yu, yl, zu, zl;
      
      y.SetupPtrStr(x);
      y.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 1, this->_location);
      z.SetupPtrStr(x);
      z.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 2, this->_location);

#ifdef PARGEMSLR_DEBUG
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[1]);
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[2]);
      level_str._work_vector_occupied[1] = 1;
      level_str._work_vector_occupied[2] = 1;
#endif

      y.Fill(zero);
      z.Fill(zero);
      rhsu.SetupPtrStr(level_str._F_mat);
      rhsu.UpdatePtr(rhs.GetData(), this->_location);
      rhsl.SetupPtrStr(level_str._E_mat);
      rhsl.UpdatePtr(rhs.GetData()+n_local, this->_location);
      xu.SetupPtrStr(level_str._F_mat);
      xu.UpdatePtr(x.GetData(), this->_location);
      xl.SetupPtrStr(level_str._E_mat);
      xl.UpdatePtr(x.GetData()+n_local, this->_location);
      zu.SetupPtrStr(level_str._F_mat);
      zu.UpdatePtr(z.GetData(), this->_location);
      zl.SetupPtrStr(level_str._E_mat);
      zl.UpdatePtr(z.GetData()+n_local, this->_location);
      yu.SetupPtrStr(level_str._F_mat);
      yu.UpdatePtr(y.GetData(), this->_location);
      yl.SetupPtrStr(level_str._E_mat);
      yl.UpdatePtr(y.GetData()+n_local, this->_location);
      
      /* The multiplicative solve
       * 1. x1 = M^{-1}b, the smoothing
       * 2. r1 = b - Ax1
       * 3. r = [-EB^{-1},I]r1
       * 4. v = S^{-1}r
       * 5. x = x1 + [-B^{-1}F;I]v
       */
      
#ifdef PARGEMSLR_TIMING
      /* Step 1：
       * smoothing: x1 = M^{-1}b
       * 
       */
      if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU, (this->SolveB(xu, rhsu, 0, level)));
      }
      else
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_ILUT, (this->SolveB(xu, rhsu, 0, level)));
      }
#else
      this->SolveB(xu, rhsu, 0, level);
#endif
      
      /* Step 2: z = b - Ax 
       *   |b_u| | B F | |x_u|   |b_u| |Bx_u|
       * = |b_l|-| E C | | 0 | = |b_l|-|Ex_u|
       */
      zu.Fill(zero);
      zu.Axpy(one, rhsu);
      this->BMatVec( level, 0, 'N', mone, xu, one, zu);
      
      level_str._E_mat.MatVec( 'N', mone, xu, one, rhsl, zl); 
      
      /* Step 3: z_l = R*x = z_l - (EB^{-1})z_u 
       * y_u is used as the temp buffer
       */
      //yu.Fill(zero);
      this->SolveB(yu, zu, solve_option, level);
      
      level_str._E_mat.MatVec( 'N', mone, yu, one, zl);
      
      /* Step 4: apply the low-rank correction 
       * z_l = C\z_l + W*H*W' * z_l
       * or solve with inner FGMRES
       */
      if( this->_gemslr_setups._enable_inner_iters_setup && this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSolve && level == 0)
      {
         /* solve with 0 as initial guess
          * buffer 3 and 4 are used.
          */
         xl.Fill(zero);
         this->_inner_iters_solver.Solve(xl, zl);
      }
      else
      {
         if( level_str._lrc > 0)
         {
            /* apply the low-rank update */
#ifdef PARGEMSLR_TIMING
            if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
            {
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELR, (this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level)));
            }
            else
            {
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_LRC, (this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level)));
            }
#else
            this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level);
#endif
            zl.Axpy( one, level_str._xlr_temp);
         }
         this->SolveLevelGemslrMul( xl, zl, level+1, true);
      }
      
      /* Step 5: 5. x = x + [-B^{-1}F;I]xl
       * note that we know the original z has the lower part equals to 0, we can directly put zl into xl
       */
      level_str._F_mat.MatVec( 'N', mone, xl, zero, yu);
      this->SolveB(zu, yu, solve_option, level);
      
      xu.Axpy(one, zu);
      
      /* might need to apply permutation 
       * This is always done on the top level
       * and on lower levels if edge separator is used
       */
      if(apply_perm)
      {
         level_str._comm_helper.DataTransferReverse(level_str._sol_temp, x_out, this->_location, this->_location);
      }
      
      y.Clear();
      z.Clear();
      
#ifdef PARGEMSLR_DEBUG
      level_str._work_vector_occupied[1] = 0;
      level_str._work_vector_occupied[2] = 0;
#endif

      rhsu.Clear();
      rhsl.Clear();
      xu.Clear();
      xl.Clear();
      zu.Clear();
      zl.Clear();
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::SolveLevelGemslrMul( ParallelVectorClass<float> &x_out, ParallelVectorClass<float> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_double::SolveLevelGemslrMul( ParallelVectorClass<double> &x_out, ParallelVectorClass<double> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_complexs::SolveLevelGemslrMul( ParallelVectorClass<complexs> &x_out, ParallelVectorClass<complexs> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_complexd::SolveLevelGemslrMul( ParallelVectorClass<complexd> &x_out, ParallelVectorClass<complexd> &rhs_in, int level, bool doperm);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SolveLevelEsmslr( VectorType &x_out, VectorType &rhs_in, int level, bool doperm)
   {
      
      /* buffer used:
       * 1, 2 used on the last level
       * 1 used on other levels, 3 ane 4 might be used when inner iteration is activated
       */
      
      /* define the data type */
      typedef DataType T;
      
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
      
      /* the solve phase of GeMSLR */
      
      PARGEMSLR_CHKERR(level < 0);
      PARGEMSLR_CHKERR(level >= this->_nlev_used);
      
      int n_local;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      /* might need to apply permutation 
       * This is always done on the top level
       * and on lower levels if edge separator is used
       */
      bool apply_perm = level_str._comm_helper._is_ready && doperm;
      if(apply_perm)
      {
         
         /* note that we are solving the equation with Apq, we might need to apply the permutation 
          * Ax = b => PAQQ'x = Pb, we are solving with Q'x
          * 
          * Note on the permutation:
          * 
          * swap row: left,  P(i, pperm[i]) = 1;
          * swap col: right, Q(qperm[i], i) = 1;
          * 
          * Q(qperm[i], i) = 1 => left apply Q is equal to Q(i, rqperm[i])
          *                    => Q' has Q'(i, qperm[i]) = 1;
          * 
          */
         
         level_str._comm_helper.DataTransfer(rhs_in, level_str._rhs_temp, this->_location, this->_location);
         level_str._sol_temp.Fill(0.0);
      }
      
      VectorType &x = apply_perm? level_str._sol_temp : x_out;
      VectorType &rhs = apply_perm? level_str._rhs_temp : rhs_in;
      
      if(level == this->_nlev_used - 1)
      {
         /* in this case we apply solve on the last level, a B solve */
#ifdef PARGEMSLR_TIMING
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU_L, this->SolveB(x, rhs, 0, level));
         }
         else
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_ILUT_L, this->SolveB(x, rhs, 0, level));
         }
#else
         this->SolveB(x, rhs, 0, level);
#endif
         
         /* might need to apply permutation 
          * This is always done on the top level
          * and on lower levels if edge separator is used
          */
         if(apply_perm)
         {
            level_str._comm_helper.DataTransferReverse(level_str._sol_temp, x_out, this->_location, this->_location);
         }
         
         return PARGEMSLR_SUCCESS;
         
      }
      
      if(level == 0)
      {
         /* In this situation, we compute 
          * x_u = L \ rhs_u
          * z_l = rhs_l - EU^{-1}*x_u
          * x_l = S^{-1}z_l
          * z_u = x_u - L^{-1}F * x_l
          * x_u = U \ z_u
          */
         
         int nLU;
         
         IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> *ilup 
                     = (IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) level_str._B_solver[0];
         
         CsrMatrixClass<T> &EU = ilup->GetE();
         CsrMatrixClass<T> &LF = ilup->GetF();
         
         //int nE = EU.GetNumRowsLocal();
         nLU = LF.GetNumRowsLocal();
         //int n = nE + nLU;
         
         VectorType        rhsu, rhsl, xu, xl, zu, zl;
         
         VectorType z;
         z.SetupPtrStr(x);
         z.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 2, this->_location);

#ifdef PARGEMSLR_DEBUG
         PARGEMSLR_CHKERR(level_str._work_vector_occupied[2]);
         level_str._work_vector_occupied[2] = 1;
#endif

         rhsu.SetupPtrStr(level_str._F_mat);
         rhsu.UpdatePtr(rhs.GetData(), this->_location);
         rhsl.SetupPtrStr(level_str._E_mat);
         rhsl.UpdatePtr(rhs.GetData()+nLU, this->_location);
         xu.SetupPtrStr(level_str._F_mat);
         xu.UpdatePtr(x.GetData(), this->_location);
         xl.SetupPtrStr(level_str._E_mat);
         xl.UpdatePtr(x.GetData()+nLU, this->_location);
         zu.SetupPtrStr(level_str._F_mat);
         zu.UpdatePtr( z.GetData(), this->_location);
         zl.SetupPtrStr(level_str._E_mat);
         zl.UpdatePtr( z.GetData()+nLU, this->_location);
         
         T                 one, zero, mone;
         
         /* begin */
         one = 1.0;
         zero = 0.0;
         mone = -1.0;
         
         z.Fill(zero);
         
         /* 1st, apply L solve
          * x_u = L \ rhs_u
          */
         ilup->SolveL(xu.GetDataVector(), rhsu.GetDataVector());
         
         /* 2nd, compute the right-hand-side for the global schur system
          * z_l = rhs_l - EU^{-1}*x_u
          */
         
         zl.Fill(zero);
         zl.Axpy(one, rhsl);
         EU.MatVec( 'N', mone, xu, one, zl);
         
         /* 3rd need to solve global Schur Complement 
          * x_l = S^{-1}z_l
          */
         if(level_str._C_mat.GetNumRowsGlobal() > 0)
         {
            /*initialize solution to zero for residual equation */
            if( this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSolve && this->_gemslr_setups._enable_inner_iters_setup)
            {
               /* solve with 0 as initial guess 
                * 4 and 5 might be used
                */
               xl.Fill(zero);
               this->_inner_iters_solver.Solve( xl, zl);
            }
            else
            {
               /* solve, S^{-1} = (LR+I)*C^{-1} */
               this->SolveLevelEsmslr( xl, zl, level+1, true);
               if( level_str._lrc > 0)
               {
                  /* apply the low-rank update */
#ifdef PARGEMSLR_TIMING
                  if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
                  {
                     PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELR, (this->SolveApplyLowRankLevel( level_str._xlr_temp, xl, level)));
                  }
                  else
                  {
                     PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_LRC, (this->SolveApplyLowRankLevel( level_str._xlr_temp, xl, level)));
                  }
#else
                  this->SolveApplyLowRankLevel( level_str._xlr_temp, xl, level);
#endif
                  xl.Axpy( one, level_str._xlr_temp);
               }
            }
         }
         
         /* 4th compute z_u = x_u - L^{-1}F * x_l
          */
         zu.Fill(zero);
         zu.Axpy(one, xu);
         LF.MatVec( 'N', mone, xl, one, zu);
         
         /* 5th need to solve x_u = U \ z_u */
         ilup->SolveU(xu.GetDataVector(), zu.GetDataVector());
            
         /* might need to apply permutation 
          * This is always done on the top level
          * and on lower levels if edge separator is used
          */
         if(apply_perm)
         {
            level_str._comm_helper.DataTransferReverse(level_str._sol_temp, x_out, this->_location, this->_location);
         }
         
         z.Clear();

#ifdef PARGEMSLR_DEBUG
         level_str._work_vector_occupied[2] = 0;
#endif

         return PARGEMSLR_SUCCESS;
         
      }
      
      n_local = level_str._F_mat.GetNumRowsLocal();
      
      T one, zero, mone;
      
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      VectorType rhsu, rhsl, xu, xl, zu, zl;

      VectorType z;
      z.SetupPtrStr(x);
      z.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 2, this->_location);

#ifdef PARGEMSLR_DEBUG
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[2]);
      level_str._work_vector_occupied[2] = 1;
#endif

      z.Fill(zero);
      rhsu.SetupPtrStr(level_str._F_mat);
      rhsu.UpdatePtr(rhs.GetData(), this->_location);
      rhsl.SetupPtrStr(level_str._E_mat);
      rhsl.UpdatePtr(rhs.GetData()+n_local, this->_location);
      xu.SetupPtrStr(level_str._F_mat);
      xu.UpdatePtr(x.GetData(), this->_location);
      xl.SetupPtrStr(level_str._E_mat);
      xl.UpdatePtr(x.GetData()+n_local, this->_location);
      zu.SetupPtrStr(level_str._F_mat);
      zu.UpdatePtr( z.GetData(), this->_location);
      zl.SetupPtrStr(level_str._E_mat);
      zl.UpdatePtr( z.GetData()+n_local, this->_location);
      
      if(this->_gemslr_setups._solve_option_setup == kGemslrUSolve && this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSolve)
      {
         /* skip the block L solve */
         /* Step 1: z = rhs */
         PARGEMSLR_MEMCPY( z.GetData(), rhs.GetData(), rhs.GetLengthLocal(), kMemoryHost, kMemoryHost, DataType);
      }
      else
      {
#ifdef PARGEMSLR_TIMING
         /* Step 1: z_u = U\L\rhs_u */
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU, (this->SolveB(zu, rhsu, 0, level)));
         }
         else
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_ILUT, (this->SolveB(zu, rhsu, 0, level)));
         }
#else
         this->SolveB(zu, rhsu, 0, level);
#endif
         
         /* Step 2: z_l = -E * z_u */
#ifdef PARGEMSLR_TIMING
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_EMV, (level_str._E_mat.MatVec( 'N', mone, zu, zero, zl)));
         }
         else
         {
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_EMV, (level_str._E_mat.MatVec( 'N', mone, zu, zero, zl)));
         }
#else
         level_str._E_mat.MatVec( 'N', mone, zu, zero, zl);
#endif
         
         /* Step 3: z_l = rhs_l + z_l = rhs_l - E * z_u */
         zl.Axpy(one, rhsl);
      }
      
      /* Step 4: apply the low-rank correction 
       * x_l = C\z_l + W*H*W' * z_l
       */
      if( this->_gemslr_setups._enable_inner_iters_setup && this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSolve && level == 0)
      {
         /* solve with 0 as initial guess 
          * 3 and 4 used
          */
         xl.Fill(zero);
         this->_inner_iters_solver.Solve(xl, zl);
      }
      else
      {
         if( level_str._lrc > 0)
         {
            /* apply the low-rank update */
#ifdef PARGEMSLR_TIMING
            if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
            {
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELR, (this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level)));
            }
            else
            {
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_LRC, (this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level)));
            }
#else
            this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level);
#endif
            zl.Axpy( one, level_str._xlr_temp);
         }
         this->SolveLevelEsmslr( xl, zl, level+1, true);
      }
      
      /* Step 5: z_u = - F * x_l */
      
#ifdef PARGEMSLR_TIMING
      if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_FMV, (level_str._F_mat.MatVec( 'N', mone, xl, zero, zu)));
      }
      else
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_FMV, (level_str._F_mat.MatVec( 'N', mone, xl, zero, zu)));
      }
#else
      level_str._F_mat.MatVec( 'N', mone, xl, zero, zu);
#endif
      /* Step 6: z_u = rhs_u - F * x_l */
      zu.Axpy(one, rhsu);
      
      /* Step 7: x_u = U\L\z_u */
#ifdef PARGEMSLR_TIMING
      if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU, (this->SolveB(xu, zu, 0, level)));
      }
      else
      {
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_ILUT, (this->SolveB(xu, zu, 0, level)));
      }
#else
      this->SolveB(xu, zu, 0, level);
#endif

      /* might need to apply permutation 
       * This is always done on the top level
       * and on lower levels if edge separator is used
       */
      if(apply_perm)
      {
         level_str._comm_helper.DataTransferReverse(level_str._sol_temp, x_out, this->_location, this->_location);
      }
      
      z.Clear();

#ifdef PARGEMSLR_DEBUG
      level_str._work_vector_occupied[2] = 0;
#endif

      rhsu.Clear();
      rhsl.Clear();
      xu.Clear();
      xl.Clear();
      zu.Clear();
      zl.Clear();
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::SolveLevelEsmslr( ParallelVectorClass<float> &x_out, ParallelVectorClass<float> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_double::SolveLevelEsmslr( ParallelVectorClass<double> &x_out, ParallelVectorClass<double> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_complexs::SolveLevelEsmslr( ParallelVectorClass<complexs> &x_out, ParallelVectorClass<complexs> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_complexd::SolveLevelEsmslr( ParallelVectorClass<complexd> &x_out, ParallelVectorClass<complexd> &rhs_in, int level, bool doperm);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SolveLevelPslr( VectorType &x_out, VectorType &rhs_in, int level, bool doperm)
   {
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::SolveLevelPslr( ParallelVectorClass<float> &x_out, ParallelVectorClass<float> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_double::SolveLevelPslr( ParallelVectorClass<double> &x_out, ParallelVectorClass<double> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_complexs::SolveLevelPslr( ParallelVectorClass<complexs> &x_out, ParallelVectorClass<complexs> &rhs_in, int level, bool doperm);
   template int precond_gemslr_csr_par_complexd::SolveLevelPslr( ParallelVectorClass<complexd> &x_out, ParallelVectorClass<complexd> &rhs_in, int level, bool doperm);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SolveB( VectorType &x, VectorType &rhs, int option, int level)
   {
      /* the solve with B on a certain level 
       * Buffer used: 5 and 6 used in residual iteration. \n 
       * No extra buffer required if turn off residual iteration.
       */
      
      PARGEMSLR_CHKERR(level < 0);
      PARGEMSLR_CHKERR(level >= this->_nlev_used);
      
      int i, ncomp, n_start, n1, n2;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      ncomp = level_str._ncomps;
      
      n_start = this->_lev_ptr_v[level];
      
      /* apply the solve
       * B = diag[B1, B2, B3, B4]
       * solve them one by one
       */
      
      if( this->_global_precond_option != kGemslrGlobalPrecondBJ && 
            this->_global_precond_option != kGemslrGlobalPrecondGeMSLR && 
            this->_global_precond_option != kGemslrGlobalPrecondESMSLR)
      {
         PARGEMSLR_ERROR("Invalid Global Precond Option.");
         return PARGEMSLR_ERROR_INVALED_OPTION;
      }
      
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, n1, n2) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for(i = 0 ; i < ncomp ; i ++)
      {
         n1 = this->_dom_ptr_v2[level][i];
         n2 = this->_dom_ptr_v2[level][i+1];
         
         SequentialVectorClass<DataType> xi, rhsi;
         
         xi.SetupPtr(x.GetData() + n1 - n_start, n2 - n1, this->_location);
         rhsi.SetupPtr(rhs.GetData() + n1 - n_start, n2 - n1, this->_location);
         
         /* apply the residual iteration */
         if( this->_gemslr_setups._level_setups._ilu_residual_iters > 1 && this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            int j;
            
            DataType  one, mone;
            
            one = 1.0;
            mone = -1.0;
            
            SequentialVectorClass<DataType> yi, ri, ei;
            
            yi.SetupPtr(level_str._x_temp.GetData() + n1 - n_start, n2 - n1, this->_location);
            ri.SetupPtr(level_str._work_vector.GetData() + level_str._work_vector_unit_length * 5 + n1 - n_start, n2 - n1, this->_location);
            ei.SetupPtr(level_str._work_vector.GetData() + level_str._work_vector_unit_length * 6 + n1 - n_start, n2 - n1, this->_location);
            
#ifdef PARGEMSLR_DEBUG
            PARGEMSLR_CHKERR(level_str._work_vector_occupied[5]);
            PARGEMSLR_CHKERR(level_str._work_vector_occupied[6]);
            level_str._work_vector_occupied[5] = 1;
            level_str._work_vector_occupied[6] = 1;
#endif
            
            xi.Fill(DataType());
            
            for(j = 0 ; j < this->_gemslr_setups._level_setups._ilu_residual_iters ; j ++)
            {
               /* ri = rhsi - B * xi */
               ri.Fill(DataType());
               ri.Axpy(one, rhsi);
               level_str._B_mat_v[i].MatVec( 'N', mone, xi, one, ri);
               
               /* now solve B*ei = ri */
               ei.Fill(DataType());
               switch(this->_global_precond_option)
               {
                  case kGemslrGlobalPrecondESMSLR:
                  {
                     if(level == 0)
                     {
                        IluClass<CsrMatrixClass<DataType>, SequentialVectorClass<DataType>, DataType> *ilup 
                                    = (IluClass<CsrMatrixClass<DataType>, SequentialVectorClass<DataType>, DataType>*) level_str._B_solver[0];
                        
                        ilup->SolveL(ei, ri);
                        PARGEMSLR_MEMCPY( ri.GetData(), ei.GetData(), n2 - n1, this->_location, this->_location, DataType);
                        ilup->SolveU(ei, ri);
                        
                        break;
                     }
                     /* otherwise use GeMSLR solve */
                  }
                  case kGemslrGlobalPrecondBJ: case kGemslrGlobalPrecondGeMSLR:
                  {
                     if(option == 0 && level_str._B_solver != NULL)
                     {
                        level_str._B_solver[i]->Solve(ei, ri);
                     }
                     else
                     {
                        level_str._B_precond[i]->Solve(ei, ri);
                     }
                     break;
                  }
                  default:
                  {
                     //PARGEMSLR_ERROR("Invalid Global Precond Option.");
                     //return PARGEMSLR_ERROR_INVALED_OPTION;
                  }
               }
                  
               /* xi = xi + ri */
               xi.Axpy(one, ei);
               
#ifdef PARGEMSLR_DEBUG
            /* release the buffer */
            level_str._work_vector_occupied[5] = 0;
            level_str._work_vector_occupied[6] = 0;
#endif
            
            }
         }
         else
         {
            switch(this->_global_precond_option)
            {
               case kGemslrGlobalPrecondESMSLR:
               {
                  if(level == 0)
                  {
                     IluClass<CsrMatrixClass<DataType>, SequentialVectorClass<DataType>, DataType> *ilup 
                                 = (IluClass<CsrMatrixClass<DataType>, SequentialVectorClass<DataType>, DataType>*) level_str._B_solver[0];
                     
                     ilup->SolveL(xi, rhsi);
                     PARGEMSLR_MEMCPY( rhsi.GetData(), xi.GetData(), n2 - n1, this->_location, this->_location, DataType);
                     ilup->SolveU(xi, rhsi);
                     
                     break;
                  }
                  /* otherwise use GeMSLR solve */
               }
               case kGemslrGlobalPrecondBJ: case kGemslrGlobalPrecondGeMSLR:
               {
                  if(option == 0 && level_str._B_solver != NULL)
                  {
                     level_str._B_solver[i]->Solve(xi, rhsi);
                  }
                  else
                  {
                     level_str._B_precond[i]->Solve(xi, rhsi);
                  }
                  break;
               }
               default:
               {
                  //PARGEMSLR_ERROR("Invalid Global Precond Option.");
                  //return PARGEMSLR_ERROR_INVALED_OPTION;
               }
            }
         }
         
         xi.Clear();
         rhsi.Clear();
         
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::SolveB( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, int option, int level);
   template int precond_gemslr_csr_par_double::SolveB( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, int option, int level);
   template int precond_gemslr_csr_par_complexs::SolveB( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, int option, int level);
   template int precond_gemslr_csr_par_complexd::SolveB( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, int option, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SolveApplyLowRankLevel( VectorType &x, VectorType &rhs, int level)
   {
      
      /* define the data type */
      typedef DataType T;
      
      int            n_H;
      T              zero, one;
      
      MPI_Comm       comm;
      int            np, myid;
      
      this->_matrix->GetMpiInfo(np, myid, comm);
      
      zero  = 0.0;
      one   = 1.0;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = (level >= 0) ? this->_levs_v[level] : this->_lev_A[0];
      n_H   = level_str._Hk.GetNumColsLocal();
      
      if(n_H == 0)
      {
         /* no low-rank correction in this case, set output to 0.0 
          * typically should not reach here, we have another check outside
          */
         x.Fill(0.0);
         return PARGEMSLR_SUCCESS;
      }
   
      /* Step1: W'*x */
      level_str._Wk.MatVec('C', one, rhs, zero, level_str._xlr1_temp);
      
      /* Step2: H*W'*x 
       * Note that this communication need to go back to the host
       * TODO: CUDA aware MPI
       */
      if(this->_location == kMemoryDevice)
      {
         /* copy to host */
         PARGEMSLR_MEMCPY(level_str._xlr1_temp_h.GetData(), level_str._xlr1_temp.GetData(), level_str._xlr1_temp.GetLengthLocal(), kMemoryHost, kMemoryDevice, T);
         PARGEMSLR_MPI_CALL( PargemslrMpiAllreduce( level_str._xlr1_temp_h.GetData(), level_str._xlr2_temp_h.GetData(), level_str._xlr1_temp.GetLengthLocal(), MPI_SUM, comm) );
         /* copy back to device */
         PARGEMSLR_MEMCPY(level_str._xlr2_temp.GetData(), level_str._xlr2_temp_h.GetData(), level_str._xlr1_temp.GetLengthLocal(), kMemoryDevice, kMemoryHost, T);
      }
      else
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiAllreduce( level_str._xlr1_temp.GetData(), level_str._xlr2_temp.GetData(), level_str._xlr1_temp.GetLengthLocal(), MPI_SUM, comm) );
      }
      
      level_str._WHk.MatVec('N', one, level_str._xlr2_temp, zero, x);
      
      //level_str._Hk.MatVec('N', one, level_str._xlr2_temp, zero, level_str._xlr1_temp);
      
      /* Step3: W*H*W'*x */
      //level_str._Wk.MatVec('N', one, level_str._xlr1_temp, zero, x);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::SolveApplyLowRankLevel( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_par_double::SolveApplyLowRankLevel( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_par_complexs::SolveApplyLowRankLevel( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_par_complexd::SolveApplyLowRankLevel( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::GetNumRows(int level)
   {
      if(level == -1)
      {
         return this->_matrix->GetNumRowsLocal();
      }
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      return level_str._E_mat.GetNumRowsLocal();
   }
   template int precond_gemslr_csr_par_float::GetNumRows(int level);
   template int precond_gemslr_csr_par_double::GetNumRows(int level);
   template int precond_gemslr_csr_par_complexs::GetNumRows(int level);
   template int precond_gemslr_csr_par_complexd::GetNumRows(int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::EBFCMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      
      /* Buffer used:
       * 1 and 2
       */
      
      if(this->_gemslr_setups._solve_option_setup == kGemslrMulSolve)
      {
         return this->RAPEBFCMatVec(level, trans, alpha, x, beta, y);
      }
      
#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif
      
      /* define the data type */
      typedef DataType T;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      T one, zero, mone;
      
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      VectorType zu, zl, yu;
      
      VectorType y_temp, z_temp;
      y_temp.SetupPtrStr(x);
      y_temp.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 1, this->_location);
      z_temp.SetupPtrStr(x);
      z_temp.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 2, this->_location);

#ifdef PARGEMSLR_DEBUG
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[1]);
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[2]);
      level_str._work_vector_occupied[1] = 1;
      level_str._work_vector_occupied[2] = 1;
#endif

      y_temp.Fill(zero);
      z_temp.Fill(zero);
      
      yu.SetupPtrStr(level_str._F_mat);
      zu.SetupPtrStr(level_str._F_mat);
      zl.SetupPtrStr(level_str._E_mat);
      
      yu.UpdatePtr( y_temp.GetData(), this->_location);
      zu.UpdatePtr( z_temp.GetData(), this->_location);
      zl.UpdatePtr( z_temp.GetData()+level_str._F_mat.GetNumRowsLocal(), this->_location);
      
      /* Solve with C with GeMSLR */
      this->SolveLevelGemslr( zl, x, level+1, true);
      
      /* compute y = x - C_i * zl */
      this->CMatVec( level, 0, 'N', mone, zl, zero, y);
      y.Axpy(one, x);
      
      /* Matvec with E */
#ifdef PARGEMSLR_TIMING
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_FMV, (level_str._F_mat.MatVec( 'N', one, zl, zero, zu)));
      
      /* Solve with B */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU, (this->SolveB(yu, zu, 0, level)));
      
      /* Matvec with F */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_EMV, (level_str._E_mat.MatVec( 'N', one, yu, one, y)));
#else
      level_str._F_mat.MatVec( 'N', one, zl, zero, zu);
      
      /* Solve with B */
      this->SolveB(yu, zu, 0, level);
      
      /* Matvec with F */
      level_str._E_mat.MatVec( 'N', one, yu, one, y);
#endif

      y_temp.Clear();
      z_temp.Clear();

#ifdef PARGEMSLR_DEBUG
      level_str._work_vector_occupied[1] = 0;
      level_str._work_vector_occupied[2] = 0;
#endif

      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_par_float::EBFCMatVec( int level, char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_csr_par_double::EBFCMatVec( int level, char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_csr_par_complexs::EBFCMatVec( int level, char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_csr_par_complexd::EBFCMatVec( int level, char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::RAPEBFCMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      
      /* Buffer used:
       * 1 and 2
       * 
       * We are actually computing (2EB^{-1}F - EB^{-1}BB^{-1}F)C^{-1}x
       */
      
#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif
      
      /* define the data type */
      typedef DataType T;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      int solve_option = 0;
      
      T two, one, zero, mone;
      
      two = 2.0;
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      VectorType zu, zl, yu;
      
      VectorType y_temp, z_temp;
      y_temp.SetupPtrStr(x);
      y_temp.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 1, this->_location);
      z_temp.SetupPtrStr(x);
      z_temp.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 2, this->_location);

#ifdef PARGEMSLR_DEBUG
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[1]);
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[2]);
      level_str._work_vector_occupied[1] = 1;
      level_str._work_vector_occupied[2] = 1;
#endif

      y_temp.Fill(zero);
      z_temp.Fill(zero);
      
      yu.SetupPtrStr(level_str._F_mat);
      zu.SetupPtrStr(level_str._F_mat);
      zl.SetupPtrStr(level_str._E_mat);
      
      yu.UpdatePtr( y_temp.GetData(), this->_location);
      zu.UpdatePtr( z_temp.GetData(), this->_location);
      zl.UpdatePtr( z_temp.GetData()+level_str._F_mat.GetNumRowsLocal(), this->_location);
      
      /* Solve with C with GeMSLR */
      this->SolveLevelGemslrMul( zl, x, level+1, true);
      
      /* compute y = x - C_i * zl */
      this->CMatVec( level, 0, 'N', mone, zl, zero, y);
      y.Axpy(one, x);
      
      /* after that, we need 2EB^{-1}F - EB^{-1}BB^{-1}F */
      
#ifdef PARGEMSLR_TIMING
      /* Matvec with F */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_FMV, (level_str._F_mat.MatVec( 'N', one, zl, zero, zu)));
      
      /* Solve with B */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU, (this->SolveB(yu, zu, solve_option, level)));
      
      /* y = 2EB^{-1}F */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_EMV, (level_str._E_mat.MatVec( 'N', two, yu, one, y)));
      
      /* zu = BB^{-1}F */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_BMV, (this->BMatVec(level, 0, 'N', one, yu, zero, zu)));
      
      /* yu = B^{-1}BB^{-1}F */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU, (this->SolveB(yu, zu, solve_option, level)));
      
      /* y -= EB^{-1}BB^{-1}F */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_EMV, (level_str._E_mat.MatVec( 'N', mone, yu, one, y)));
      
#else
      /* Matvec with F */
      level_str._F_mat.MatVec( 'N', one, zl, zero, zu);
      
      /* Solve with B */
      this->SolveB(yu, zu, solve_option, level);
      
      /* after this, we need (2E + EB^{-1}B)B^{-1}F */
      
      /* y = 2EB^{-1}F */
      level_str._E_mat.MatVec( 'N', two, yu, one, y);
      
      /* zu = BB^{-1}F */
      this->BMatVec(level, 0, 'N', one, yu, zero, zu);
      
      /* yu = B^{-1}BB^{-1}F */
      this->SolveB(yu, zu, solve_option, level);
      
      /* y -= EB^{-1}BB^{-1}F */
      level_str._E_mat.MatVec( 'N', mone, yu, one, y);
      
#endif
      
      y_temp.Clear();
      z_temp.Clear();

#ifdef PARGEMSLR_DEBUG
      level_str._work_vector_occupied[1] = 0;
      level_str._work_vector_occupied[2] = 0;
#endif

      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_par_float::RAPEBFCMatVec( int level, char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_csr_par_double::RAPEBFCMatVec( int level, char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_csr_par_complexs::RAPEBFCMatVec( int level, char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_csr_par_complexd::RAPEBFCMatVec( int level, char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SCMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      
      /* buffer used:
       * 2 
       */
      
#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif
      
      /* define the data type */
      typedef DataType T;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      T one, zero, mone;
      
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      VectorType zl;
      
      VectorType z_temp;
      z_temp.SetupPtrStr(x);
      z_temp.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 2, this->_location);

#ifdef PARGEMSLR_DEBUG
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[2]);
      level_str._work_vector_occupied[2] = 1;
#endif

      z_temp.Fill(zero);
      
      zl.SetupPtrStr(level_str._E_mat);
      
      zl.UpdatePtr(z_temp.GetData()+level_str._F_mat.GetNumRowsLocal(), this->_location);
      
#ifdef PARGEMSLR_TIMING
      /* matvec with S */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SMV, (level_str._S_mat.MatVec( 'N', one, x, zero, zl)));
#else
      /* matvec with S */
      level_str._S_mat.MatVec( 'N', one, x, zero, zl);
#endif
      /* solve with C */
      this->SolveLevelEsmslr(y, zl, level+1, true);
      
      y.Axpy( mone, x);
      
      z_temp.Clear();

#ifdef PARGEMSLR_DEBUG
      level_str._work_vector_occupied[2] = 0;
#endif

      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_par_float::SCMatVec( int level, char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_csr_par_double::SCMatVec( int level, char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_csr_par_complexs::SCMatVec( int level, char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_csr_par_complexd::SCMatVec( int level, char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::ACMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      
      /* Compute the global low-rank correction as AM^{-1}(I-X)^{-1} = I 
       * Thus, we need Ix - AM^{-1}x
       */
      
      DataType one, zero, mone;
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      if(this->_lev_A[0]._work_vector.GetLengthLocal() < y.GetLengthLocal())
      {
         this->_lev_A[0]._work_vector.Setup(y.GetLengthLocal(), this->_location, true);
      }
      
      VectorType y_temp;
      y_temp.SetupPtrStr(y);
      y_temp.UpdatePtr( this->_lev_A[0]._work_vector.GetData(), this->_location);
      
      /* Compute M^{-1}x, put into y_temp */
      switch(this->_global_precond_option)
      {
         case kGemslrGlobalPrecondBJ: case kGemslrGlobalPrecondGeMSLR:
         {
            /* standard GeMSLR solve */
            if(this->_gemslr_setups._solve_option_setup == kGemslrMulSolve)
            {
               /* multiplicative solve */
               this->SolveLevelGemslrMul( y_temp, x, 0, true);
            }
            else
            {
               this->SolveLevelGemslr( y_temp, x, 0, true);
            }
            break;
         }
         case kGemslrGlobalPrecondESMSLR:
         {
            /* explicite Schur complement */
            this->SolveLevelEsmslr( y_temp, x, 0, true);
            break;
         }
         case kGemslrGlobalPrecondPSLR:
         {
            /* PSLR solve */
            this->SolveLevelPslr( y_temp, x, 0, true);
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Invalid global solve option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
            break;
         }
      }
      
      /* now, y_temp = M^{-1}x, next y = x - Ay_temp = (I-AM^{-1})x */
      
      this->_matrix->MatVec( 'N', mone, y_temp, zero, y);
      y.Axpy(one, x);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_par_float::ACMatVec( int level, char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_csr_par_double::ACMatVec( int level, char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_csr_par_complexs::ACMatVec( int level, char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_csr_par_complexd::ACMatVec( int level, char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::PCLRMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      
      /* buffer used: 1 and 2 */
      
      /* we only supports last level */
      PARGEMSLR_CHKERR(level != this->_nlev_used - 1);
      
#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif
      
      /* define the data type */
      typedef DataType T;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      bool apply_perm = level_str._comm_helper._is_ready;
      
      int i;
      T one, zero, mone;
      
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      VectorType yl, zl;
      yl.SetupPtrStr(x);
      yl.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 1, this->_location);
      zl.SetupPtrStr(x);
      zl.UpdatePtr( level_str._work_vector.GetData() + level_str._work_vector_unit_length * 2, this->_location);

#ifdef PARGEMSLR_DEBUG
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[1]);
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[2]);
      level_str._work_vector_occupied[1] = 1;
      level_str._work_vector_occupied[2] = 1;
#endif

      y.Fill(zero);
      yl.Fill(zero);
      zl.Fill(zero);
      
      /* simply (-CoffdC^{-1})^m */
      this->SolveB(zl, x, 0, level);
      
      if(level == 0)
      {
#ifdef PARGEMSLR_TIMING
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_CMV, (level_str._C_mat.MatVecOffd( 'N', mone, zl, zero, y)));
#else
         level_str._C_mat.MatVecOffd( 'N', mone, zl, zero, y);
#endif
      }
      else
      {
         if(apply_perm)
         {
            /* in this case, we need to apply the permutation */
            level_str._comm_helper.DataTransferReverse( zl, level_str._rhs3_temp, this->_location, this->_location);
            level_str._sol3_temp.Fill(0.0);
            this->SchurMatVec(level-1, 1, 'N', one, level_str._rhs3_temp, zero, level_str._sol3_temp);
            level_str._comm_helper.DataTransfer(level_str._sol3_temp, y, this->_location, this->_location);
         }
         else
         {
            this->SchurMatVec(level-1, 1, 'N', one, zl, zero, y);
         }
      }
      
      /* apply one extra time */
      for(i = 0 ; i < this->_gemslr_setups._level_setups._B_poly_order ; i ++)
      {
         this->SolveB(zl, y, 0, level);
         if(level == 0)
         {
#ifdef PARGEMSLR_TIMING
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_CMV, (level_str._C_mat.MatVecOffd( 'N', mone, zl, zero, y)));
#else
            level_str._C_mat.MatVecOffd( 'N', mone, zl, zero, y);
#endif
         }
         else
         {
            if(apply_perm)
            {
               /* in this case, we need to apply the permutation */
               level_str._comm_helper.DataTransferReverse( zl, level_str._rhs3_temp, this->_location, this->_location);
               level_str._sol3_temp.Fill(0.0);
               this->SchurMatVec(level-1, 1, 'N', one, level_str._rhs3_temp, zero, level_str._sol3_temp);
               level_str._comm_helper.DataTransfer(level_str._sol3_temp, y, this->_location, this->_location);
            }
            else
            {
               this->SchurMatVec(level-1, 1, 'N', one, zl, zero, y);
            }
         }
      }
      
      yl.Clear();
      zl.Clear();
      
#ifdef PARGEMSLR_DEBUG
      level_str._work_vector_occupied[1] = 0;
      level_str._work_vector_occupied[2] = 0;
#endif

      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_par_float::PCLRMatVec( int level, char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_csr_par_double::PCLRMatVec( int level, char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_csr_par_complexs::PCLRMatVec( int level, char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_csr_par_complexd::PCLRMatVec( int level, char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SchurMatVec( int level, int option, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      
      /* buffer used:
       * For the SchurMatVec: 3 and 4, sol2 and rhs2 on the next level.
       * For the RAP: 3 and 4, sol2 and rhs2 on the next level.
       */
      
      if(this->_gemslr_setups._solve_option_setup == kGemslrMulSolve)
      {
         return this->RAPMatVec(level, option, trans, alpha, x, beta, y);
      }
      
      /* define the data type */
      typedef DataType T;
      
      /* y = alpha*S*x + beta*y 
       * S = C - EB^{-1}F, we first compute -EB^{-1}F*x.
       */
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      int n_local;
      T one, zero, malpha;

#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif

      one = T(1.0);
      zero = T();
      malpha = -alpha;
      
      n_local = level_str._F_mat.GetNumRowsLocal();
      
      VectorType v, w, vu, wl, wu, xiu, xil, wiu, wil;
      
      v.SetupPtrStr(level_str._x_temp);
      w.SetupPtrStr(level_str._x_temp);
      
      v.UpdatePtr(level_str._work_vector.GetData() + level_str._work_vector_unit_length * 3, this->_location);
      w.UpdatePtr(level_str._work_vector.GetData() + level_str._work_vector_unit_length * 4, this->_location);

#ifdef PARGEMSLR_DEBUG
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[3]);
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[4]);
      level_str._work_vector_occupied[3] = 1;
      level_str._work_vector_occupied[4] = 1;
#endif

      vu.SetupPtrStr(level_str._F_mat);
      wu.SetupPtrStr(level_str._F_mat);
      wl.SetupPtrStr(level_str._E_mat);
      
      vu.UpdatePtr(v.GetData(), this->_location);
      wu.UpdatePtr(w.GetData(), this->_location);
      wl.UpdatePtr(w.GetData()+n_local, this->_location);
      
      /* we first compute y = -alpha*EB^{-1}F*x + beta y */
      
      /* Matvec with E */
#ifdef PARGEMSLR_TIMING
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_FMV, (level_str._F_mat.MatVec( 'N', one, x, zero, wu)));
      
      /* Solve with B */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_SOLVELU, (this->SolveB(vu, wu, 0, level)));
      
      /* Matvec with F */
      PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_EMV, (level_str._E_mat.MatVec( 'N', malpha, vu, beta, y)));
#else
      level_str._F_mat.MatVec( 'N', one, x, zero, wu);
      
      /* Solve with B */
      this->SolveB(vu, wu, 0, level);
      
      /* Matvec with F */
      level_str._E_mat.MatVec( 'N', malpha, vu, beta, y);
#endif
      
      /* now compute y = alpha*C*x + y 
       * note that C = | Bi  Fi | 
       *               | Ei  Ci |
       * We put everything to wl first, and apply alpha to wl
       * 
       * Note on the shift:
       * C = | Bi  Fi |         => temp_v/w start from here
       *     | Ei  Ci |  => x starts from here, requires extra shift
       * 
       * 
       */
      
      /* matvec with C, permutation might applied */
      this->CMatVec( level, option, 'N', alpha, x, one, y);
      
      v.Clear();
      w.Clear();
      
#ifdef PARGEMSLR_DEBUG
      level_str._work_vector_occupied[3] = 0;
      level_str._work_vector_occupied[4] = 0;
#endif

      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_par_float::SchurMatVec( int level, int option, char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_csr_par_double::SchurMatVec( int level, int option, char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_csr_par_complexs::SchurMatVec( int level, int option, char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_csr_par_complexd::SchurMatVec( int level, int option, char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::RAPMatVec( int level, int option, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      /* Buffer used:
       * 3 and 4
       */
      
      int solve_option = 0;
      
      /* define the data type */
      typedef DataType T;
      
      /* y = alpha*S*x + beta*y 
       * S = RAP = [-EB^{-1},I]|B F||-B^{-1}F|
       *                       |E C||   I    |
       */
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      int n_local;
      T one, zero, malpha, mone;
      
#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif

      one = T(1.0);
      mone = -one;
      zero = T();
      malpha = -alpha;
      
      n_local = level_str._F_mat.GetNumRowsLocal();
      
      VectorType v, w, vu, vl, wl, wu, xiu, xil, wiu, wil;
      
      v.SetupPtrStr(level_str._x_temp);
      w.SetupPtrStr(level_str._x_temp);
      
      v.UpdatePtr(level_str._work_vector.GetData() + level_str._work_vector_unit_length * 3, this->_location);
      w.UpdatePtr(level_str._work_vector.GetData() + level_str._work_vector_unit_length * 4, this->_location);

#ifdef PARGEMSLR_DEBUG
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[3]);
      PARGEMSLR_CHKERR(level_str._work_vector_occupied[4]);
      level_str._work_vector_occupied[3] = 1;
      level_str._work_vector_occupied[4] = 1;
#endif

      vu.SetupPtrStr(level_str._F_mat);
      wu.SetupPtrStr(level_str._F_mat);
      vl.SetupPtrStr(level_str._E_mat);
      wl.SetupPtrStr(level_str._E_mat);
      
      vu.UpdatePtr(v.GetData(), this->_location);
      wu.UpdatePtr(w.GetData(), this->_location);
      vl.UpdatePtr(v.GetData()+n_local, this->_location);
      wl.UpdatePtr(w.GetData()+n_local, this->_location);
      
      /* First compute v = Px = | -B^{-1}F| x
       *                        |    I    |
       */
      
      vl.Fill(zero);
      vl.Axpy(one, x);
      
      level_str._F_mat.MatVec( 'N', mone, x, zero, wu);
      this->SolveB(vu, wu, solve_option, level);
      
      /* APx = 
       * |B F| |v_u|
       * |E C| |v_l|
       */
      this->BMatVec(level, 0, 'N', one, vu, zero, wu);
      level_str._F_mat.MatVec( 'N', one, vl, one, wu);
      
      /* standard C matvec */
      level_str._E_mat.MatVec( 'N', one, vu, zero, wl);
      this->CMatVec( level, option, 'N', one, vl, one, wl);
      
      /* RAPx = [-EB^{-1},I] |w_u|
       *                     |w_l|
       * 
       * we need alpha*RAPx + beta * y
       */
      
      this->SolveB(vu, wu, solve_option, level);
      level_str._E_mat.MatVec( 'N', malpha, vu, beta, y);
      y.Axpy(alpha, wl);
      
      v.Clear();
      w.Clear();
      
#ifdef PARGEMSLR_DEBUG
      level_str._work_vector_occupied[3] = 0;
      level_str._work_vector_occupied[4] = 0;
#endif

      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_par_float::RAPMatVec( int level, int option, char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_csr_par_double::RAPMatVec( int level, int option, char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_csr_par_complexs::RAPMatVec( int level, int option, char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_csr_par_complexd::RAPMatVec( int level, int option, char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::CMatVec( int level, int option, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      /* C might have different permutation.
       * sol2 and rhs2 are used under this situation.
       * No extra buffer required.
       * 
       * When option is 1, only matvec with offdiagonal of C, this only works on the second last level.
       * 
       */
      
      /* define the data type */
      typedef DataType T;
      
      if(level == this->_nlev_used - 1)
      {
         /* we have no C in this level */
         return PARGEMSLR_SUCCESS;
      }
      
      //ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &leveld_str = this->_levs_v[level+1];
      
      bool apply_perm = leveld_str._comm_helper._is_ready;
      
      /* copy data to rhs_temp */
      if(apply_perm)
      {
         leveld_str._comm_helper.DataTransfer(x, leveld_str._rhs2_temp, this->_location, this->_location);
         leveld_str._comm_helper.DataTransfer(y, leveld_str._sol2_temp, this->_location, this->_location);
      }
      
      VectorType &x_compute = (apply_perm)? leveld_str._rhs2_temp : x;
      VectorType &y_compute = (apply_perm)? leveld_str._sol2_temp : y;
      
      int n_local;
      T one;

#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      this->_matrix->GetMpiInfo(np, myid, comm);
#endif
      if(option == 1)
      {
         T mone = -1.0;
         T malpha = mone*alpha;
         
         PARGEMSLR_CHKERR(level != this->_nlev_used - 2);
         
         /* this C_mat is stored on the last level, -C_offd + EB^{-1}F */
         this->_levs_v[level+1]._C_mat.MatVecOffd( 'N', malpha, x_compute, mone, y_compute);
         if(apply_perm)
         {
            leveld_str._comm_helper.DataTransferReverse(leveld_str._sol2_temp, y, this->_location, this->_location);
         }
         return PARGEMSLR_SUCCESS;
      }
      if(level == this->_nlev_used - 2)
      {
         /* last level, only need to matvec with C */
#ifdef PARGEMSLR_TIMING
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_CMV, (leveld_str._C_mat.MatVec( 'N', alpha, x_compute, beta, y_compute)));
#else
         leveld_str._C_mat.MatVec( 'N', alpha, x_compute, beta, y_compute);
#endif
         if(apply_perm)
         {
            leveld_str._comm_helper.DataTransferReverse(leveld_str._sol2_temp, y, this->_location, this->_location);
         }
         return PARGEMSLR_SUCCESS;
      }
      else
      {
         
         one = T(1.0);
         
         n_local = leveld_str._F_mat.GetNumRowsLocal();
         
         VectorType xu, xl, yu, yl;
         
         xu.SetupPtrStr(leveld_str._F_mat);
         yu.SetupPtrStr(leveld_str._F_mat);
         xl.SetupPtrStr(leveld_str._E_mat);
         yl.SetupPtrStr(leveld_str._E_mat);
         
         xu.UpdatePtr(x_compute.GetData(), this->_location);
         yu.UpdatePtr(y_compute.GetData(), this->_location);
         xl.UpdatePtr(x_compute.GetData()+n_local, this->_location);
         yl.UpdatePtr(y_compute.GetData()+n_local, this->_location);
         
#ifdef PARGEMSLR_TIMING
         /* Matvec with F */
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_FMV, (leveld_str._F_mat.MatVec( 'N', alpha, xl, beta, yu)));
         
         /* Matvec with E */
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_EMV, (leveld_str._E_mat.MatVec( 'N', alpha, xu, beta, yl)));
         
         /* Matvec with B */
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_BMV, (this->BMatVec(level+1, 0, 'N', alpha, xu, one, yu)));
         
         /* Matvec with C */
         this->CMatVec(level+1, 0, 'N', alpha, xl, one, yl);
#else
         /* Matvec with F */
         leveld_str._F_mat.MatVec( 'N', alpha, xl, beta, yu);
         
         /* Matvec with E */
         leveld_str._E_mat.MatVec( 'N', alpha, xu, beta, yl);
         
         /* Matvec with B */
         this->BMatVec(level+1, 0, 'N', alpha, xu, one, yu);
         
         /* Matvec with C */
         this->CMatVec(level+1, 0, 'N', alpha, xl, one, yl);
#endif
         
         if(apply_perm)
         {
            leveld_str._comm_helper.DataTransferReverse(leveld_str._sol2_temp, y, this->_location, this->_location);
         }
      }
      
      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_par_float::CMatVec( int level, int option, char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_csr_par_double::CMatVec( int level, int option, char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_csr_par_complexs::CMatVec( int level, int option, char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_csr_par_complexd::CMatVec( int level, int option, char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::BMatVec( int level, int option, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      /* the solve with B on a certain level 
       * no extra working buffer required
       */
      
      PARGEMSLR_CHKERR(level < 0);
      PARGEMSLR_CHKERR(level >= this->_nlev_used);
      
      int i, ncomp, n_start, n1, n2;
      
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      ncomp = level_str._ncomps;
      
      n_start = this->_lev_ptr_v[level];
      
      /* apply the matvec
       * B = diag[B1, B2, B3, B4]
       * solve them one by one
       */
      
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, n1, n2) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for(i = 0 ; i < ncomp ; i ++)
      {
         n1 = this->_dom_ptr_v2[level][i];
         n2 = this->_dom_ptr_v2[level][i+1];
         
         SequentialVectorClass<DataType> xi, yi;
         
         xi.SetupPtr(x.GetData() + n1 - n_start, n2 - n1, this->_location);
         yi.SetupPtr(y.GetData() + n1 - n_start, n2 - n1, this->_location);
         
         level_str._B_mat_v[i].MatVec( trans, alpha, xi, beta, yi);
         
         xi.Clear();
         yi.Clear();
      }
      
      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_par_float::BMatVec( int level, int option, char trans, const float &alpha, ParallelVectorClass<float> &x, const float &beta, ParallelVectorClass<float> &y);
   template int precond_gemslr_csr_par_double::BMatVec( int level, int option, char trans, const double &alpha, ParallelVectorClass<double> &x, const double &beta, ParallelVectorClass<double> &y);
   template int precond_gemslr_csr_par_complexs::BMatVec( int level, int option, char trans, const complexs &alpha, ParallelVectorClass<complexs> &x, const complexs &beta, ParallelVectorClass<complexs> &y);
   template int precond_gemslr_csr_par_complexd::BMatVec( int level, int option, char trans, const complexd &alpha, ParallelVectorClass<complexd> &x, const complexd &beta, ParallelVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::GetSize()
   {
      return this->_n;
   }
   template int precond_gemslr_csr_par_float::GetSize();
   template int precond_gemslr_csr_par_double::GetSize();
   template int precond_gemslr_csr_par_complexs::GetSize();
   template int precond_gemslr_csr_par_complexd::GetSize();
   
   template <class MatrixType, class VectorType, typename DataType>
   long int ParallelGemslrClass<MatrixType, VectorType, DataType>::GetNumNonzeros()
   {
      long int nnz_bsolver, nnz_lr;
      
      return this->GetNumNonzeros(nnz_bsolver, nnz_lr);
   }
   template long int precond_gemslr_csr_par_float::GetNumNonzeros();
   template long int precond_gemslr_csr_par_double::GetNumNonzeros();
   template long int precond_gemslr_csr_par_complexs::GetNumNonzeros();
   template long int precond_gemslr_csr_par_complexd::GetNumNonzeros();
   
   template <class MatrixType, class VectorType, typename DataType>
   long int ParallelGemslrClass<MatrixType, VectorType, DataType>::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr)
   {
      {
         int      i;
         long int nnz_bsolveri, nnz_lri;
         
         nnz_bsolver = 0;
         nnz_lr = 0;
         
         for(i = 0 ; i < this->_nlev_used ; i ++)
         {
            this->_levs_v[i].GetNumNonzeros(nnz_bsolveri, nnz_lri);
            nnz_bsolver += nnz_bsolveri;
            nnz_lr += nnz_lri;
         }
         
         this->_lev_A[0].GetNumNonzeros(nnz_bsolveri, nnz_lri);
         
         nnz_bsolver += nnz_bsolveri;
         nnz_lr += nnz_lri;
         
         return nnz_bsolver + nnz_lr;
      }
   }
   template long int precond_gemslr_csr_par_float::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template long int precond_gemslr_csr_par_double::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template long int precond_gemslr_csr_par_complexs::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template long int precond_gemslr_csr_par_complexd::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetSolveLocation( const int &location)
   {
      
      if(this->_ready && this->_location != location)
      {
         this->MoveData(location);
      }
      
      this->_location = location;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::SetSolveLocation( const int &location);
   template int precond_gemslr_csr_par_double::SetSolveLocation( const int &location);
   template int precond_gemslr_csr_par_complexs::SetSolveLocation( const int &location);
   template int precond_gemslr_csr_par_complexd::SetSolveLocation( const int &location);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::MoveData( const int &location)
   {
      if( this->_location == location)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      this->_location = location;
      
      if(this->_ready)
      {
         int level, ncomp, i;
         
         this->_inner_iters_solver.SetSolveLocation(this->_location);
         this->_inner_iters_precond.SetSolveLocation(this->_location);
         
         for(level = 0 ; level < this->_nlev_used ; level++)
         {
            ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
            
            ncomp = level_str._ncomps;
            
            for(i = 0 ; i < ncomp ; i ++)
            {
               level_str._B_mat_v[i].MoveData(this->_location);
            }
            
            if(level_str._B_solver)
            {
               for(i = 0 ; i < ncomp ; i ++)
               {
                  level_str._B_solver[i]->SetSolveLocation(this->_location);
               }
            }
            if(level_str._B_precond)
            {
               for(i = 0 ; i < ncomp ; i ++)
               {
                  level_str._B_precond[i]->SetSolveLocation(this->_location);
               }
            }
         }
         
         for(level = this->_nlev_used - 1 ; level >= 0 ; level--)
         {
            ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
            
            level_str._E_mat.MoveData(this->_location);
            level_str._F_mat.MoveData(this->_location);
            level_str._Wk.MoveData(this->_location);
            level_str._Hk.MoveData(this->_location);
            level_str._WHk.MoveData(this->_location);
            level_str._cWk.MoveData(this->_location);
            level_str._cHk.MoveData(this->_location);
            level_str._cWHk.MoveData(this->_location);
            level_str._C_mat.MoveData(this->_location);
            level_str._S_mat.MoveData(this->_location);
            level_str._A_mat.MoveData(this->_location);
            
            level_str._xlr_temp.MoveData(this->_location);
            level_str._xlr1_temp.MoveData(this->_location);
            level_str._xlr2_temp.MoveData(this->_location);
            
            level_str._work_vector.MoveData(this->_location);
            level_str._x_temp.MoveData(this->_location);
            
            level_str._rhs_temp.MoveData(this->_location);
            level_str._rhs2_temp.MoveData(this->_location);
            level_str._rhs3_temp.MoveData(this->_location);
            level_str._sol_temp.MoveData(this->_location);
            level_str._sol2_temp.MoveData(this->_location);
            level_str._sol3_temp.MoveData(this->_location);
            
            level_str._comm_helper.MoveData(this->_location);
            
         }
         
         this->_lev_A[0]._E_mat.MoveData(this->_location);
         this->_lev_A[0]._F_mat.MoveData(this->_location);
         this->_lev_A[0]._Wk.MoveData(this->_location);
         this->_lev_A[0]._Hk.MoveData(this->_location);
         this->_lev_A[0]._WHk.MoveData(this->_location);
         this->_lev_A[0]._cWk.MoveData(this->_location);
         this->_lev_A[0]._cHk.MoveData(this->_location);
         this->_lev_A[0]._cWHk.MoveData(this->_location);
         this->_lev_A[0]._C_mat.MoveData(this->_location);
         this->_lev_A[0]._S_mat.MoveData(this->_location);
         this->_lev_A[0]._A_mat.MoveData(this->_location);
         
         this->_lev_A[0]._xlr_temp.MoveData(this->_location);
         this->_lev_A[0]._xlr1_temp.MoveData(this->_location);
         this->_lev_A[0]._xlr2_temp.MoveData(this->_location);
         
         this->_lev_A[0]._work_vector.MoveData(this->_location);
         this->_lev_A[0]._x_temp.MoveData(this->_location);
         
         this->_lev_A[0]._rhs_temp.MoveData(this->_location);
         this->_lev_A[0]._rhs2_temp.MoveData(this->_location);
         this->_lev_A[0]._rhs3_temp.MoveData(this->_location);
         this->_lev_A[0]._sol_temp.MoveData(this->_location);
         this->_lev_A[0]._sol2_temp.MoveData(this->_location);
         this->_lev_A[0]._sol3_temp.MoveData(this->_location);
         
         this->_lev_A[0]._comm_helper.MoveData(this->_location);
         
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::MoveData( const int &location);
   template int precond_gemslr_csr_par_double::MoveData( const int &location);
   template int precond_gemslr_csr_par_complexs::MoveData( const int &location);
   template int precond_gemslr_csr_par_complexd::MoveData( const int &location);
   
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::PlotPatternGnuPlot( const char *datafilename)
   {
      
      if(this->_location == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Plot Pattern only works on host.");
      }
      
      MPI_Comm comm;
      int np, myid;
      this->_matrix->GetMpiInfo(np, myid, comm);
      
      int *diag_i;
      int *diag_j;
      int *offd_i;
      int *offd_j;
      long int *offd_map;
      long int n_level;
      
      int level, n_local, ncomp, i, j, k, j1, j2, layout;
      long int n_start, nshift, nE_start, nB_global;
      
      FILE *pgnuplot = NULL;
      
      layout = 1;
      while(layout*layout < this->_nlev_used)
      {
         layout++;
      }
      
      if(myid == 0)
      {
         
         if ((pgnuplot = popen("gnuplot -persistent", "w")) == NULL)
         {
            printf("Can't open gnuplot file.\n");
            return PARGEMSLR_ERROR_IO_ERROR;
         }
         
         fprintf(pgnuplot, "set multiplot layout %d, %d title \"Parallel GeMSLR partition\"\n", layout, layout);
         
      }
      
      for(level = 0 ; level < this->_nlev_used - 1 ; level++)
      {
         
         char filename[1024];
         snprintf( filename, 1024, "./TempData/%slev%05d%05d", datafilename, level, myid );
         
         FILE *fdata;
         
         if ((fdata = fopen(filename, "w")) == NULL)
         {
            printf("Can't open file.\n");
            return PARGEMSLR_ERROR_IO_ERROR;
         }
         
         ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
         
         ncomp = level_str._ncomps;
         
         nB_global = level_str._F_mat.GetNumRowsGlobal();
         n_start = level_str._F_mat.GetRowStartGlobal();
         n_level = nB_global + level_str._E_mat.GetNumRowsGlobal();
         
         for(k = 0 ; k < ncomp ; k ++)
         {
            diag_i = level_str._B_mat_v[k].GetI();
            diag_j = level_str._B_mat_v[k].GetJ();
            
            n_local = level_str._B_mat_v[k].GetNumRowsLocal();
            nshift = this->_dom_ptr_v2[level][k] - this->_dom_ptr_v2[level][0];
            
            for(i = 0 ; i < n_local ; i ++)
            {
               j1 = diag_i[i];
               j2 = diag_i[i+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  fprintf(fdata, "%ld %ld \n", n_start+diag_j[j]+1+nshift, n_level-n_start-i+1-nshift);
               }
            }
            
         }
         
         diag_i = level_str._E_mat.GetDiagMat().GetI();
         diag_j = level_str._E_mat.GetDiagMat().GetJ();
         offd_i = level_str._E_mat.GetOffdMat().GetI();
         offd_j = level_str._E_mat.GetOffdMat().GetJ();
         offd_map = level_str._E_mat.GetOffdMap().GetData();
         
         n_local = level_str._E_mat.GetNumRowsLocal();
         nE_start = level_str._E_mat.GetRowStartGlobal();
         
         for(i = 0 ; i < n_local ; i ++)
         {
            j1 = diag_i[i];
            j2 = diag_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               fprintf(fdata, "%ld %ld \n", n_start+diag_j[j]+1, n_level-nE_start-nB_global-i+1);
            }
            j1 = offd_i[i];
            j2 = offd_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               fprintf(fdata, "%ld %ld \n", offd_map[offd_j[j]]+1, n_level-nB_global-nE_start-i+1);
            }
         }
         
         diag_i = level_str._F_mat.GetDiagMat().GetI();
         diag_j = level_str._F_mat.GetDiagMat().GetJ();
         offd_i = level_str._F_mat.GetOffdMat().GetI();
         offd_j = level_str._F_mat.GetOffdMat().GetJ();
         offd_map = level_str._F_mat.GetOffdMap().GetData();
         
         n_local = level_str._F_mat.GetNumRowsLocal();
         
         for(i = 0 ; i < n_local ; i ++)
         {
            j1 = diag_i[i];
            j2 = diag_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               fprintf(fdata, "%ld %ld \n", nB_global+nE_start+diag_j[j]+1, n_level-n_start-i+1);
            }
            j1 = offd_i[i];
            j2 = offd_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               fprintf(fdata, "%ld %ld \n", nB_global+offd_map[offd_j[j]]+1, n_level-n_start-i+1);
            }
         }
         
         fclose(fdata);
         
         PARGEMSLR_MPI_CALL(MPI_Barrier(comm));
         
         if(myid == 0)
         {
            fprintf(pgnuplot, "set title \"level = %d\"\n", level);
            fprintf(pgnuplot, "set xrange [0:%ld]\n", n_level+1);
            fprintf(pgnuplot, "set yrange [0:%ld]\n", n_level+1);
            if(n_level < 200)
            {
               fprintf(pgnuplot, "plot '%s' pt 1 title \"p%d\"", filename, myid);
               for(i = 1 ; i < np ; i ++)
               {
                  char tempfilename[1024];
                  snprintf( tempfilename, 1024, "./TempData/%slev%05d%05d", datafilename, level, i );
                  fprintf(pgnuplot, ", '%s' pt 1 title \"p%d\"", tempfilename, i);
               }
               fprintf(pgnuplot, "\n");
            }
            else
            {
               fprintf(pgnuplot, "plot '%s' pt 0 title \"p%d\"", filename, myid);
               for(i = 1 ; i < np ; i ++)
               {
                  char tempfilename[1024];
                  snprintf( tempfilename, 1024, "./TempData/%slev%05d%05d", datafilename, level, i );
                  fprintf(pgnuplot, ", '%s' pt 0 title \"p%d\"", tempfilename, i);
               }
               fprintf(pgnuplot, "\n");
            }
         }
         
      }
      
      {
         
         level = this->_nlev_used - 1;
         
         char filename[1024];
         snprintf( filename, 1024, "./TempData/%slev%05d%05d", datafilename, level, myid );
         
         FILE *fdata;
         
         if ((fdata = fopen(filename, "w")) == NULL)
         {
            printf("Can't open file.\n");
            return PARGEMSLR_ERROR_IO_ERROR;
         }
         
         ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
         
         diag_i      = level_str._C_mat.GetDiagMat().GetI();
         diag_j      = level_str._C_mat.GetDiagMat().GetJ();
         offd_i      = level_str._C_mat.GetOffdMat().GetI();
         offd_j      = level_str._C_mat.GetOffdMat().GetJ();
         offd_map    = level_str._C_mat.GetOffdMap().GetData();
         
         n_local = level_str._C_mat.GetNumRowsLocal();
         n_start = level_str._C_mat.GetRowStartGlobal();
         nB_global = level_str._C_mat.GetNumRowsGlobal();
         
         for(i = 0 ; i < n_local ; i ++)
         {
            j1 = diag_i[i];
            j2 = diag_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               fprintf(fdata, "%ld %ld \n", n_start+diag_j[j]+1, nB_global-n_start-i);
            }
            j1 = offd_i[i];
            j2 = offd_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               fprintf(fdata, "%ld %ld \n", offd_map[offd_j[j]]+1, nB_global-n_start-i);
            }
         }
         
         fclose(fdata);
         
         PARGEMSLR_MPI_CALL(MPI_Barrier(comm));
         
         if(myid == 0)
         {
            fprintf(pgnuplot, "set title \"level = %d\"\n", level);
            fprintf(pgnuplot, "set xrange [0:%ld]\n", nB_global+1);
            fprintf(pgnuplot, "set yrange [0:%ld]\n", nB_global+1);
            if(n_local < 200)
            {
               fprintf(pgnuplot, "plot '%s' pt 1 title \"p%d\"", filename, myid);
               for(i = 1 ; i < np ; i ++)
               {
                  char tempfilename[1024];
                  snprintf( tempfilename, 1024, "./TempData/%slev%05d%05d", datafilename, level, i );
                  fprintf(pgnuplot, ", '%s' pt 1 title \"p%d\"", tempfilename, i);
               }
               fprintf(pgnuplot, "\n");
            }
            else
            {
               fprintf(pgnuplot, "plot '%s' pt 0 title \"p%d\"", filename, myid);
               for(i = 1 ; i < np ; i ++)
               {
                  char tempfilename[1024];
                  snprintf( tempfilename, 1024, "./TempData/%slev%05d%05d", datafilename, level, i );
                  fprintf(pgnuplot, ", '%s' pt 0 title \"p%d\"", tempfilename, i);
               }
               fprintf(pgnuplot, "\n");
            }
         }
         
      }
      
      if(pgnuplot)
      {
         pclose(pgnuplot);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::PlotPatternGnuPlot( const char *datafilename);
   template int precond_gemslr_csr_par_double::PlotPatternGnuPlot( const char *datafilename);
   template int precond_gemslr_csr_par_complexs::PlotPatternGnuPlot( const char *datafilename);
   template int precond_gemslr_csr_par_complexd::PlotPatternGnuPlot( const char *datafilename);
   
   void hypre_parilu_swap( int *v,
                    int  i,
                    int  j )
   {
      int temp;

      temp = v[i];
      v[i] = v[j];
      v[j] = temp;
   }

   template <typename T>
   void hypre_parilu_swap2(int     *v,
                     T  *w,
                     int      i,
                     int      j )
   {
      int temp;
      T temp2;

      temp = v[i];
      v[i] = v[j];
      v[j] = temp;
      temp2 = w[i];
      w[i] = w[j];
      w[j] = temp2;
   }
   
   /* see hypre_ILUMinHeapAddI for detail instructions */
   template <typename T>
   void hypre_parilu_MinHeap_Add_I_R_Ii(int *heap, T *I1, int *Ii1, int len)
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
            hypre_parilu_swap(Ii1,heap[p],heap[len]);
            hypre_parilu_swap2(heap,I1,p,len);
            len = p;
         }
         else
         {
            break;
         }
      }
   }
   
   /* see hypre_ILUMinHeapRemoveI for detail instructions */
   template <typename T>
   void hypre_parilu_MinHeap_Remove_I_R_Ii(int *heap, T *I1, int *Ii1, int len)
   {
      /* parent, left, right */
      int p,l,r;
      len--;/* now len is the max index */
      /* swap the first element to last */
      hypre_parilu_swap(Ii1,heap[0],heap[len]);
      hypre_parilu_swap2(heap,I1,0,len);
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
            hypre_parilu_swap(Ii1,heap[p],heap[l]);
            hypre_parilu_swap2(heap,I1,l,p);
            p = l;
            l = 2*p+1;
         }
         else
         {
            break;
         }
      }
   }

   /* see hypre_ILUMinHeapAddI for detail instructions */
   template <typename T>
   void hypre_parilu_MaxrHeap_Add_Rabs_I(T *heap, int *I1, int len)
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
            hypre_parilu_swap2(I1,heap,-p,-len);
            len = p;
         }
         else
         {
            break;
         }
      }
   }

   /* see hypre_ILUMinHeapRemoveI for detail instructions */
   template <typename T>
   void hypre_parilu_MaxrHeap_Remove_Rabs_I(T *heap, int *I1, int len)
   {
      /* parent, left, right */
      int p,l,r;
      len--;/* now len is the max index */
      /* swap the first element to last */
      hypre_parilu_swap2(I1,heap,0,-len);
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
            hypre_parilu_swap2(I1,heap,-l,-p);
            p = l;
            l = 2*p+1;
         }
         else
         {
            break;
         }
      }
   }

   /* Split based on quick sort algorithm (avoid sorting the entire array)
    * find the largest k elements out of original array
    * array: input array for compare
    * I: integer array bind with array
    * k: largest k elements
    * len: length of the array
    */
   template <typename T>
   void hypre_parilu_MaxQSplit_Rabs_I(T *array, int *I, int left, int bound, int right)
   {
      int i, last;
      if (left >= right)
      {
         return;
      }
      hypre_parilu_swap2(I,array,left,(left+right)/2);
      last = left;
      for(i = left + 1 ; i <= right ; i ++)
      {
         if(PargemslrAbs(array[i]) > PargemslrAbs(array[left]))
         {
            hypre_parilu_swap2(I,array,++last,i);
         }
      }
      hypre_parilu_swap2(I,array,left,last);
      hypre_parilu_MaxQSplit_Rabs_I(array,I,left,bound,last-1);
      if(bound > last)
      {
          hypre_parilu_MaxQSplit_Rabs_I(array,I,last+1,bound,right);
      }
   }
   
   /* Patial ILUT only on the top level.
    */
   template <class MatrixType, class VectorType, typename DataType>
   int ParallelGemslrClass<MatrixType, VectorType, DataType>::SetupPartialILUT( VectorType &x, VectorType &rhs)
   {
      
      /* after calling this function, the _S_mat and ilu will be on device */
      
      //typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      typedef DataType T;
      
      /* we target only the top level */
      ParallelGemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[0];
      MatrixType &A = (this->_gemslr_setups._global_partition_setup) ? level_str._A_mat : *this->_matrix;
      
      /* MPI info */
      int np, myid;
      MPI_Comm comm;
      
      A.GetMpiInfo(np, myid, comm);
      
      long int total_rows;
      int n, nLU, m;
      SequentialVectorClass<T> dummyx, dummyrhs;
      
      n = A.GetNumRowsLocal();
      nLU = level_str._nI;
      
      m = n - nLU;
      
      PARGEMSLR_MALLOC( level_str._B_solver, 1, kMemoryHost, SolverClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*);
      
      /* build partial ILU solver */
      PARGEMSLR_PLACEMENT_NEW( level_str._B_solver[0], kMemoryHost, IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>);
      IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType> *ilup 
                  = (IluClass<CsrMatrixClass<T>, SequentialVectorClass<T>, DataType>*) level_str._B_solver[0];
   
      /* apply factorization on diagonal mat */
      ilup->SetMatrix(A.GetDiagMat());
      
      /* nnz per level */
      ilup->SetMaxNnzPerRow(this->_gemslr_setups._level_setups._B_ilu_max_row_nnz_setup);
      ilup->SetMaxNnzPerRowSPart(this->_gemslr_setups._level_setups._S_ilu_max_row_nnz_setup);
      
      /* droptol */
      ilup->SetDropTolerance(this->_gemslr_setups._level_setups._B_ilu_tol_setup);
      ilup->SetDropToleranceEF(this->_gemslr_setups._level_setups._EF_ilu_tol_setup);
      ilup->SetDropToleranceS(this->_gemslr_setups._level_setups._S_ilu_tol_setup);
      
      /* disable openmp */
      ilup->SetOpenMPOption(kIluOpenMPNo);
      
      /* set factorize option to PartialILU */
      ilup->SetPermutationOption(kIluReorderingInput);
      ilup->GetRowPermutationVector().Setup(n);
      PARGEMSLR_MEMCPY( ilup->GetRowPermutationVector().GetData(), level_str._pperm.GetData(), n, kMemoryHost, level_str._pperm.GetDataLocation(), int);
      
      ilup->SetOption(kIluOptionPartialILUT);
      ilup->SetNB(nLU);
      
      /* now setup */
      ilup->Setup(dummyx, dummyrhs);
      
      /* now parepare the global S */
      
      /* First create Schur complement if necessary
       * Check if we need to create Schur complement
       */
      total_rows = level_str._C_mat.GetNumRowsGlobal();
      
      /* only form when total_rows > 0 */
      if( total_rows > 0 )
      {
         /* now create S 
          * Note that the offd part is exactly the same as C
          */
         level_str._S_mat.Setup( m, level_str._C_mat.GetRowStartGlobal(), total_rows, m, level_str._C_mat.GetRowStartGlobal(), total_rows, level_str._C_mat);
         
         /* the offdiagonal part should be the same */
         level_str._S_mat.GetOffdMat() = level_str._C_mat.GetOffdMat();
         level_str._S_mat.GetOffdMap() = level_str._C_mat.GetOffdMap();
         
         /* the diagonal part is from the partial ILU */
         level_str._S_mat.GetDiagMat() = ilup->GetS();
         
      }/* end of forming S */
      
      /* need to setup the communication helper */
      level_str._comm_helper._n_in = n;
      level_str._comm_helper._n_out = n;
      
      level_str._comm_helper._send_to_v.Setup(1);
      level_str._comm_helper._send_to_v[0] = myid;
      level_str._comm_helper._recv_from_v.Setup(1);
      level_str._comm_helper._recv_from_v[0] = myid;
      
      level_str._comm_helper._send_idx_v2.resize(1);
      level_str._comm_helper._recv_idx_v2.resize(1);
      
      level_str._comm_helper._send_idx_v2[0].Setup(n);
      PARGEMSLR_MEMCPY( level_str._comm_helper._send_idx_v2[0].GetData(), level_str._pperm.GetData(), n, kMemoryHost, level_str._pperm.GetDataLocation(), int);
      
      level_str._comm_helper._recv_idx_v2[0].Setup(n);
      level_str._comm_helper._recv_idx_v2[0].UnitPerm();
      
      level_str._comm_helper.CreateHostBuffer(sizeof(T));
      
      level_str._comm_helper._is_ready = true;
      
      /* move to desired memory location, and setup working vectors */
      
      level_str._comm_helper.MoveData(this->_location);
      
      level_str._sol_temp.Setup( n, this->_location, true, A);
      level_str._rhs_temp.Setup(level_str._sol_temp.GetLengthLocal(), 
                                 level_str._sol_temp.GetStartGlobal(), 
                                 level_str._sol_temp.GetLengthGlobal(), 
                                 this->_location,
                                 true,
                                 level_str._S_mat);
      
      
      level_str._sol2_temp.Setup(level_str._sol_temp.GetLengthLocal(), 
                                 level_str._sol_temp.GetStartGlobal(), 
                                 level_str._sol_temp.GetLengthGlobal(), 
                                 this->_location,
                                 true,
                                 level_str._S_mat);
      level_str._rhs2_temp.Setup(level_str._sol_temp.GetLengthLocal(), 
                                 level_str._sol_temp.GetStartGlobal(), 
                                 level_str._sol_temp.GetLengthGlobal(), 
                                 this->_location,
                                 true,
                                 level_str._S_mat);
                                 
      level_str._sol3_temp.Setup(level_str._sol_temp.GetLengthLocal(), 
                                 level_str._sol_temp.GetStartGlobal(), 
                                 level_str._sol_temp.GetLengthGlobal(), 
                                 this->_location,
                                 true,
                                 level_str._S_mat);
      level_str._rhs3_temp.Setup(level_str._sol_temp.GetLengthLocal(), 
                                 level_str._sol_temp.GetStartGlobal(), 
                                 level_str._sol_temp.GetLengthGlobal(), 
                                 this->_location,
                                 true,
                                 level_str._S_mat);
      
      level_str._S_mat.SortOffdMap();
      level_str._S_mat.MoveData(this->_location);
      
      ilup->SetSolveLocation(this->_location);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_par_float::SetupPartialILUT( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs);
   template int precond_gemslr_csr_par_double::SetupPartialILUT( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
   template int precond_gemslr_csr_par_complexs::SetupPartialILUT( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs);
   template int precond_gemslr_csr_par_complexd::SetupPartialILUT( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
   
}
