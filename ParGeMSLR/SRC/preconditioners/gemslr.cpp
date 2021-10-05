
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
#include "poly.hpp"
#include "gemslr.hpp"

namespace pargemslr
{
	
   template <class MatrixType, class VectorType, typename DataType>
   GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::GemslrEBFCMatrixClass() : ArnoldiMatrixClass<VectorType, DataType>()
   {
      this->_level = 0;
      this->_gemslr = NULL;
   }
   template precond_gemslrebfc_csr_seq_float::GemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_seq_double::GemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_seq_complexs::GemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_seq_complexd::GemslrEBFCMatrixClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::~GemslrEBFCMatrixClass()
   {
      this->Clear();
   }
   template precond_gemslrebfc_csr_seq_float::~GemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_seq_double::~GemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_seq_complexs::~GemslrEBFCMatrixClass();
   template precond_gemslrebfc_csr_seq_complexd::~GemslrEBFCMatrixClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::GemslrEBFCMatrixClass(const GemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &precond) : ArnoldiMatrixClass<VectorType, DataType>()
   {
      this->_level = precond._level;
      this->_gemslr = precond._gemslr;
      this->_temp_v = precond._temp_v;
   }
   template precond_gemslrebfc_csr_seq_float::GemslrEBFCMatrixClass(const precond_gemslrebfc_csr_seq_float &precond);
   template precond_gemslrebfc_csr_seq_double::GemslrEBFCMatrixClass(const precond_gemslrebfc_csr_seq_double &precond);
   template precond_gemslrebfc_csr_seq_complexs::GemslrEBFCMatrixClass(const precond_gemslrebfc_csr_seq_complexs &precond);
   template precond_gemslrebfc_csr_seq_complexd::GemslrEBFCMatrixClass(const precond_gemslrebfc_csr_seq_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::GemslrEBFCMatrixClass(GemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &&precond) : ArnoldiMatrixClass<VectorType, DataType>()
   {
      this->_level = precond._level;
      precond._level = 0;
      this->_gemslr = precond._gemslr;
      precond._gemslr = NULL;
      this->_temp_v = std::move(precond._temp_v);
   }
   template precond_gemslrebfc_csr_seq_float::GemslrEBFCMatrixClass(precond_gemslrebfc_csr_seq_float &&precond);
   template precond_gemslrebfc_csr_seq_double::GemslrEBFCMatrixClass(precond_gemslrebfc_csr_seq_double &&precond);
   template precond_gemslrebfc_csr_seq_complexs::GemslrEBFCMatrixClass(precond_gemslrebfc_csr_seq_complexs &&precond);
   template precond_gemslrebfc_csr_seq_complexd::GemslrEBFCMatrixClass(precond_gemslrebfc_csr_seq_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>& GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::operator=(const GemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &precond)
   {
      this->_level = precond._level;
      this->_gemslr = precond._gemslr;
      this->_temp_v = precond._temp_v;
      return *this;
   }
   template precond_gemslrebfc_csr_seq_float& precond_gemslrebfc_csr_seq_float::operator=(const precond_gemslrebfc_csr_seq_float &precond);
   template precond_gemslrebfc_csr_seq_double& precond_gemslrebfc_csr_seq_double::operator=(const precond_gemslrebfc_csr_seq_double &precond);
   template precond_gemslrebfc_csr_seq_complexs& precond_gemslrebfc_csr_seq_complexs::operator=(const precond_gemslrebfc_csr_seq_complexs &precond);
   template precond_gemslrebfc_csr_seq_complexd& precond_gemslrebfc_csr_seq_complexd::operator=(const precond_gemslrebfc_csr_seq_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>& GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::operator=(GemslrEBFCMatrixClass<MatrixType, VectorType, DataType> &&precond)
   {
      this->_level = precond._level;
      precond._level = 0;
      this->_gemslr = precond._gemslr;
      precond._gemslr = NULL;
      this->_temp_v = std::move(precond._temp_v);
      return *this;
   }
   template precond_gemslrebfc_csr_seq_float& precond_gemslrebfc_csr_seq_float::operator=(precond_gemslrebfc_csr_seq_float &&precond);
   template precond_gemslrebfc_csr_seq_double& precond_gemslrebfc_csr_seq_double::operator=(precond_gemslrebfc_csr_seq_double &&precond);
   template precond_gemslrebfc_csr_seq_complexs& precond_gemslrebfc_csr_seq_complexs::operator=(precond_gemslrebfc_csr_seq_complexs &&precond);
   template precond_gemslrebfc_csr_seq_complexd& precond_gemslrebfc_csr_seq_complexd::operator=(precond_gemslrebfc_csr_seq_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::Clear()
   {
      this->_level = 0;
      this->_gemslr = NULL;
      this->_temp_v.Clear();
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslrebfc_csr_seq_float::Clear();
   template int precond_gemslrebfc_csr_seq_double::Clear();
   template int precond_gemslrebfc_csr_seq_complexs::Clear();
   template int precond_gemslrebfc_csr_seq_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::Setup(int level, GemslrClass<MatrixType, VectorType, DataType> &gemslr)
   {
      this->_level = level;
      this->_gemslr = &gemslr;
      
      this->_temp_v.SetupPtrStr(this->_gemslr->_levs_v[level]._E_mat);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslrebfc_csr_seq_float::Setup(int level, precond_gemslr_csr_seq_float &gemslr);
   template int precond_gemslrebfc_csr_seq_double::Setup(int level, precond_gemslr_csr_seq_double &gemslr);
   template int precond_gemslrebfc_csr_seq_complexs::Setup(int level, precond_gemslr_csr_seq_complexs &gemslr);
   template int precond_gemslrebfc_csr_seq_complexd::Setup(int level, precond_gemslr_csr_seq_complexd &gemslr);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::SetupVectorPtrStr(VectorType &v)
   {
      v.SetupPtrStr(this->_temp_v);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslrebfc_csr_seq_float::SetupVectorPtrStr(vector_seq_float &v);
   template int precond_gemslrebfc_csr_seq_double::SetupVectorPtrStr(vector_seq_double &v);
   template int precond_gemslrebfc_csr_seq_complexs::SetupVectorPtrStr(vector_seq_complexs &v);
   template int precond_gemslrebfc_csr_seq_complexd::SetupVectorPtrStr(vector_seq_complexd &v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::GetNumRowsLocal() const
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      return this->_gemslr->GetNumRows(this->_level);
   }
   template int precond_gemslrebfc_csr_seq_float::GetNumRowsLocal() const;
   template int precond_gemslrebfc_csr_seq_double::GetNumRowsLocal() const;
   template int precond_gemslrebfc_csr_seq_complexs::GetNumRowsLocal() const;
   template int precond_gemslrebfc_csr_seq_complexd::GetNumRowsLocal() const;
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::GetNumColsLocal() const
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      return this->_gemslr->GetNumRows(this->_level);
   }
   template int precond_gemslrebfc_csr_seq_float::GetNumColsLocal() const;
   template int precond_gemslrebfc_csr_seq_double::GetNumColsLocal() const;
   template int precond_gemslrebfc_csr_seq_complexs::GetNumColsLocal() const;
   template int precond_gemslrebfc_csr_seq_complexd::GetNumColsLocal() const;
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrEBFCMatrixClass<MatrixType, VectorType, DataType>::MatVec( char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      return this->_gemslr->EBFCMatVec(this->_level, trans, alpha, x, beta, y);
   }
   template int precond_gemslrebfc_csr_seq_float::MatVec( char trans, const float &alpha, SequentialVectorClass<float> &x, const float &beta, SequentialVectorClass<float> &y);
   template int precond_gemslrebfc_csr_seq_double::MatVec( char trans, const double &alpha, SequentialVectorClass<double> &x, const double &beta, SequentialVectorClass<double> &y);
   template int precond_gemslrebfc_csr_seq_complexs::MatVec( char trans, const complexs &alpha, SequentialVectorClass<complexs> &x, const complexs &beta, SequentialVectorClass<complexs> &y);
   template int precond_gemslrebfc_csr_seq_complexd::MatVec( char trans, const complexd &alpha, SequentialVectorClass<complexd> &x, const complexd &beta, SequentialVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::GemslrSchurMatrixClass()
   {
      this->_level = 0;
      this->_gemslr = NULL;
   }
   template precond_gemslr_schur_seq_float::GemslrSchurMatrixClass();
   template precond_gemslr_schur_seq_double::GemslrSchurMatrixClass();
   template precond_gemslr_schur_seq_complexs::GemslrSchurMatrixClass();
   template precond_gemslr_schur_seq_complexd::GemslrSchurMatrixClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::~GemslrSchurMatrixClass()
   {
      this->Clear();
   }
   template precond_gemslr_schur_seq_float::~GemslrSchurMatrixClass();
   template precond_gemslr_schur_seq_double::~GemslrSchurMatrixClass();
   template precond_gemslr_schur_seq_complexs::~GemslrSchurMatrixClass();
   template precond_gemslr_schur_seq_complexd::~GemslrSchurMatrixClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::GemslrSchurMatrixClass(const GemslrSchurMatrixClass<MatrixType, VectorType, DataType> &precond)
   {
      this->_level = precond._level;
      this->_gemslr = precond._gemslr;
      this->_temp_v = precond._temp_v;
   }
   template precond_gemslr_schur_seq_float::GemslrSchurMatrixClass(const precond_gemslr_schur_seq_float &precond);
   template precond_gemslr_schur_seq_double::GemslrSchurMatrixClass(const precond_gemslr_schur_seq_double &precond);
   template precond_gemslr_schur_seq_complexs::GemslrSchurMatrixClass(const precond_gemslr_schur_seq_complexs &precond);
   template precond_gemslr_schur_seq_complexd::GemslrSchurMatrixClass(const precond_gemslr_schur_seq_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::GemslrSchurMatrixClass(GemslrSchurMatrixClass<MatrixType, VectorType, DataType> &&precond)
   {
      this->_level = precond._level;
      precond._level = 0;
      this->_gemslr = precond._gemslr;
      precond._gemslr = NULL;
      this->_temp_v = std::move(precond._temp_v);
   }
   template precond_gemslr_schur_seq_float::GemslrSchurMatrixClass(precond_gemslr_schur_seq_float &&precond);
   template precond_gemslr_schur_seq_double::GemslrSchurMatrixClass(precond_gemslr_schur_seq_double &&precond);
   template precond_gemslr_schur_seq_complexs::GemslrSchurMatrixClass(precond_gemslr_schur_seq_complexs &&precond);
   template precond_gemslr_schur_seq_complexd::GemslrSchurMatrixClass(precond_gemslr_schur_seq_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrSchurMatrixClass<MatrixType, VectorType, DataType>& GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::operator=(const GemslrSchurMatrixClass<MatrixType, VectorType, DataType> &precond)
   {
      this->_level = precond._level;
      this->_gemslr = precond._gemslr;
      this->_temp_v = precond._temp_v;
      return *this;
   }
   template precond_gemslr_schur_seq_float& precond_gemslr_schur_seq_float::operator=(const precond_gemslr_schur_seq_float &precond);
   template precond_gemslr_schur_seq_double& precond_gemslr_schur_seq_double::operator=(const precond_gemslr_schur_seq_double &precond);
   template precond_gemslr_schur_seq_complexs& precond_gemslr_schur_seq_complexs::operator=(const precond_gemslr_schur_seq_complexs &precond);
   template precond_gemslr_schur_seq_complexd& precond_gemslr_schur_seq_complexd::operator=(const precond_gemslr_schur_seq_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrSchurMatrixClass<MatrixType, VectorType, DataType>& GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::operator=(GemslrSchurMatrixClass<MatrixType, VectorType, DataType> &&precond)
   {
      this->_level = precond._level;
      precond._level = 0;
      this->_gemslr = precond._gemslr;
      precond._gemslr = NULL;
      this->_temp_v = std::move(precond._temp_v);
      return *this;
   }
   template precond_gemslr_schur_seq_float& precond_gemslr_schur_seq_float::operator=(precond_gemslr_schur_seq_float &&precond);
   template precond_gemslr_schur_seq_double& precond_gemslr_schur_seq_double::operator=(precond_gemslr_schur_seq_double &&precond);
   template precond_gemslr_schur_seq_complexs& precond_gemslr_schur_seq_complexs::operator=(precond_gemslr_schur_seq_complexs &&precond);
   template precond_gemslr_schur_seq_complexd& precond_gemslr_schur_seq_complexd::operator=(precond_gemslr_schur_seq_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::Clear()
   {
      this->_level = 0;
      this->_gemslr = NULL;
      this->_temp_v.Clear();
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_schur_seq_float::Clear();
   template int precond_gemslr_schur_seq_double::Clear();
   template int precond_gemslr_schur_seq_complexs::Clear();
   template int precond_gemslr_schur_seq_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::Setup(int level, GemslrClass<MatrixType, VectorType, DataType> &gemslr)
   {
      this->_level = level;
      this->_gemslr = &gemslr;
      
      this->_temp_v.SetupPtrStr(this->_gemslr->_levs_v[level]._E_mat);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_schur_seq_float::Setup(int level, precond_gemslr_csr_seq_float &gemslr);
   template int precond_gemslr_schur_seq_double::Setup(int level, precond_gemslr_csr_seq_double &gemslr);
   template int precond_gemslr_schur_seq_complexs::Setup(int level, precond_gemslr_csr_seq_complexs &gemslr);
   template int precond_gemslr_schur_seq_complexd::Setup(int level, precond_gemslr_csr_seq_complexd &gemslr);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::SetupVectorPtrStr(VectorType &v)
   {
      v.SetupPtrStr(this->_temp_v);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_schur_seq_float::SetupVectorPtrStr(vector_seq_float &v);
   template int precond_gemslr_schur_seq_double::SetupVectorPtrStr(vector_seq_double &v);
   template int precond_gemslr_schur_seq_complexs::SetupVectorPtrStr(vector_seq_complexs &v);
   template int precond_gemslr_schur_seq_complexd::SetupVectorPtrStr(vector_seq_complexd &v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::MatVec( char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      /* matvec of the Schur complement on this level */
      return this->_gemslr->SchurMatVec(this->_level, trans, alpha, x, beta, y);
   }
   template int precond_gemslr_schur_seq_float::MatVec( char trans, const float &alpha, SequentialVectorClass<float> &x, const float &beta, SequentialVectorClass<float> &y);
   template int precond_gemslr_schur_seq_double::MatVec( char trans, const double &alpha, SequentialVectorClass<double> &x, const double &beta, SequentialVectorClass<double> &y);
   template int precond_gemslr_schur_seq_complexs::MatVec( char trans, const complexs &alpha, SequentialVectorClass<complexs> &x, const complexs &beta, SequentialVectorClass<complexs> &y);
   template int precond_gemslr_schur_seq_complexd::MatVec( char trans, const complexd &alpha, SequentialVectorClass<complexd> &x, const complexd &beta, SequentialVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrSchurMatrixClass<MatrixType, VectorType, DataType>::Solve( VectorType &x, VectorType &rhs)
   {
      PARGEMSLR_CHKERR(this->_gemslr == NULL);
      /* solve with the Schur complement is the solve on the NEXT level */
      
      DataType one = DataType(1.0);
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_gemslr->_levs_v[this->_level];
      
      /* S-solve */
      if( level_str._lrc > 0)
      {
         /* apply low-rank */
         if(this->_gemslr->GetSolvePhase() == kGemslrPhaseSetup)
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_SOLVELR, (this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, rhs, this->_level)));
         }
         else
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_LRC, (this->_gemslr->SolveApplyLowRankLevel( level_str._xlr_temp, rhs, this->_level)));
         }
         level_str._xlr_temp.Axpy( one, rhs);
         /* C solve */
         this->_gemslr->SolveLevel( x, level_str._xlr_temp, this->_level+1);
      }
      else
      {
         /* C solve */
         this->_gemslr->SolveLevel( x, rhs, this->_level+1);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_schur_seq_float::Solve( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_gemslr_schur_seq_double::Solve( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_gemslr_schur_seq_complexs::Solve( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_gemslr_schur_seq_complexd::Solve( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <typename DataType>
   void GemslrLevelSetupStruct<DataType>::SetDefault()
   {
      this->_lr_option1_setup                = kGemslrLowrankThickRestart;
      this->_lr_option2_setup                = kGemslrLowrankThickRestart;
      this->_lr_rand_init_setup              = true;
      this->_lr_rank_factor1_setup           = 1.0;
      this->_lr_rank_factor2_setup           = 1.0;
      this->_lr_rank1_setup                  = 50;
      this->_lr_rank2_setup                  = 20;
      this->_lr_arnoldi_factor1_setup        = 3.0;
      this->_lr_arnoldi_factor2_setup        = 3.0;
      this->_lr_maxits1_setup                = 3;
      this->_lr_maxits2_setup                = 3;
      this->_lr_tol_eig1_setup               = 1e-05;
      this->_lr_tol_eig2_setup               = 1e-09;
      
      this->_B_solve_option1                 = kGemslrBSolveILUT;
      this->_B_solve_option1_levels          = 1;
      this->_B_solve_option2                 = kGemslrBSolveILUT;
      this->_C_solve_option                  = kGemslrCSolveILUT;
      this->_B_smooth_option1                = kGemslrBSmoothBMILU;
      this->_C_lr_pslr                       = false;
      this->_B_ilu_tol_setup                 = 1e-02;
      this->_C_ilu_tol_setup                 = 1e-03;
      this->_S_ilu_tol_setup                 = 1e-03;
      this->_B_ilu_max_row_nnz_setup         = 100;
      this->_S_ilu_max_row_nnz_setup         = 200;
      this->_B_poly_order                    = 2;
      
      this->_lr_option1_B_setup              = kGemslrLowrankThickRestart;
      this->_lr_option2_B_setup              = kGemslrLowrankThickRestart;
      this->_lr_rand_init_B_setup            = true;
      this->_lr_rank_factor1_B_setup         = 1.0;
      this->_lr_rank_factor2_B_setup         = 1.0;
      this->_lr_rank1_B_setup                = 50;
      this->_lr_rank2_B_setup                = 20;
      this->_lr_arnoldi_factor1_B_setup      = 3.0;
      this->_lr_arnoldi_factor2_B_setup      = 3.0;
      this->_lr_maxits1_B_setup              = 3;
      this->_lr_maxits2_B_setup              = 3;
      this->_lr_tol_eig1_B_setup             = 1e-05;
      this->_lr_tol_eig2_B_setup             = 1e-09;
      
      this->_B_solve_option1_B               = kGemslrBSolveILUT;
      this->_B_solve_option1_levels_B        = 1;
      this->_B_solve_option2_B               = kGemslrBSolveILUT;
      this->_C_solve_option_B                = kGemslrCSolveILUT;
      this->_B_ilu_tol_B_setup               = 1e-02;
      this->_C_ilu_tol_B_setup               = 1e-03;
      this->_B_ilu_max_row_nnz_B_setup       = 100;
      this->_C_ilu_max_row_nnz_B_setup       = 200;
      this->_B_poly_order_B                  = 2;
   }
   template void GemslrLevelSetupStruct<float>::SetDefault();
   template void GemslrLevelSetupStruct<double>::SetDefault();
   template void GemslrLevelSetupStruct<complexs>::SetDefault();
   template void GemslrLevelSetupStruct<complexd>::SetDefault();
   
   template <typename DataType>
   GemslrLevelSetupStruct<DataType>::GemslrLevelSetupStruct()
   {
      this->_lr_option1_setup                = kGemslrLowrankThickRestart;
      this->_lr_option2_setup                = kGemslrLowrankThickRestart;
      this->_lr_rand_init_setup              = true;
      this->_lr_rank_factor1_setup           = 1.0;
      this->_lr_rank_factor2_setup           = 1.0;
      this->_lr_rank1_setup                  = 50;
      this->_lr_rank2_setup                  = 20;
      this->_lr_arnoldi_factor1_setup        = 3.0;
      this->_lr_arnoldi_factor2_setup        = 3.0;
      this->_lr_maxits1_setup                = 3;
      this->_lr_maxits2_setup                = 3;
      this->_lr_tol_eig1_setup               = 1e-05;
      this->_lr_tol_eig2_setup               = 1e-09;
      
      this->_B_solve_option1                 = kGemslrBSolveILUT;
      this->_B_solve_option1_levels          = 1;
      this->_B_solve_option2                 = kGemslrBSolveILUT;
      this->_C_solve_option                  = kGemslrCSolveILUT;
      this->_B_smooth_option1                = kGemslrBSmoothBMILU;
      this->_C_lr_pslr                       = false;
      this->_B_ilu_tol_setup                 = 1e-02;
      this->_C_ilu_tol_setup                 = 1e-03;
      this->_S_ilu_tol_setup                 = 1e-03;
      this->_B_ilu_max_row_nnz_setup         = 100;
      this->_S_ilu_max_row_nnz_setup         = 200;
      this->_B_ilu_fill_level_setup          = 1;
      this->_C_ilu_fill_level_setup          = 1;
      this->_B_poly_order                    = 2;
      
      this->_lr_option1_B_setup              = kGemslrLowrankThickRestart;
      this->_lr_option2_B_setup              = kGemslrLowrankThickRestart;
      this->_lr_rand_init_B_setup            = true;
      this->_lr_rank_factor1_B_setup         = 1.0;
      this->_lr_rank_factor2_B_setup         = 1.0;
      this->_lr_rank1_B_setup                = 50;
      this->_lr_rank2_B_setup                = 20;
      this->_lr_arnoldi_factor1_B_setup      = 3.0;
      this->_lr_arnoldi_factor2_B_setup      = 3.0;
      this->_lr_maxits1_B_setup              = 3;
      this->_lr_maxits2_B_setup              = 3;
      this->_lr_tol_eig1_B_setup             = 1e-05;
      this->_lr_tol_eig2_B_setup             = 1e-09;
      
      this->_B_solve_option1_B               = kGemslrBSolveILUT;
      this->_B_solve_option1_levels_B        = 1;
      this->_B_solve_option2_B               = kGemslrBSolveILUT;
      this->_C_solve_option_B                = kGemslrCSolveILUT;
      this->_B_ilu_tol_B_setup               = 1e-02;
      this->_C_ilu_tol_B_setup               = 1e-03;
      this->_B_ilu_max_row_nnz_B_setup       = 100;
      this->_C_ilu_max_row_nnz_B_setup       = 200;
      this->_B_ilu_fill_level_B_setup        = 1;
      this->_C_ilu_fill_level_B_setup        = 1;
      this->_B_poly_order_B                  = 2;
      
      this->_ilu_complex_shift               = false;
      this->_ilu_residual_iters              = 1;
      
   }
   template GemslrLevelSetupStruct<float>::GemslrLevelSetupStruct();
   template GemslrLevelSetupStruct<double>::GemslrLevelSetupStruct();
   template GemslrLevelSetupStruct<complexs>::GemslrLevelSetupStruct();
   template GemslrLevelSetupStruct<complexd>::GemslrLevelSetupStruct();
   
   template <typename DataType>
   GemslrLevelSetupStruct<DataType>::GemslrLevelSetupStruct(const GemslrLevelSetupStruct<DataType> &str)
   {
      this->_lr_option1_setup                = str._lr_option1_setup;
      this->_lr_option2_setup                = str._lr_option2_setup;
      this->_lr_rand_init_setup              = str._lr_rand_init_setup;
      this->_lr_rank_factor1_setup           = str._lr_rank_factor1_setup;
      this->_lr_rank_factor2_setup           = str._lr_rank_factor2_setup;
      this->_lr_rank1_setup                  = str._lr_rank1_setup;
      this->_lr_rank2_setup                  = str._lr_rank2_setup;
      this->_lr_arnoldi_factor1_setup        = str._lr_arnoldi_factor1_setup;
      this->_lr_arnoldi_factor2_setup        = str._lr_arnoldi_factor2_setup;
      this->_lr_maxits1_setup                = str._lr_maxits1_setup;
      this->_lr_maxits2_setup                = str._lr_maxits2_setup;
      this->_lr_tol_eig1_setup               = str._lr_tol_eig1_setup;
      this->_lr_tol_eig2_setup               = str._lr_tol_eig2_setup;
      
      this->_B_solve_option1                 = str._B_solve_option1;
      this->_B_solve_option1_levels          = str._B_solve_option1_levels;
      this->_B_solve_option2                 = str._B_solve_option2;
      this->_C_solve_option                  = str._C_solve_option;
      this->_B_smooth_option1                = str._B_smooth_option1;
      this->_C_lr_pslr                       = str._C_lr_pslr;
      this->_B_ilu_tol_setup                 = str._B_ilu_tol_setup;
      this->_C_ilu_tol_setup                 = str._C_ilu_tol_setup;
      this->_S_ilu_tol_setup                 = str._S_ilu_tol_setup;
      this->_B_ilu_max_row_nnz_setup         = str._B_ilu_max_row_nnz_setup;
      this->_S_ilu_max_row_nnz_setup         = str._S_ilu_max_row_nnz_setup;
      this->_B_ilu_fill_level_setup          = str._B_ilu_fill_level_setup;
      this->_C_ilu_fill_level_setup          = str._C_ilu_fill_level_setup;
      this->_B_poly_order                    = str._B_poly_order;
      
      this->_lr_option1_B_setup              = str._lr_option1_B_setup;
      this->_lr_option2_B_setup              = str._lr_option2_B_setup;
      this->_lr_rand_init_B_setup            = str._lr_rand_init_B_setup;
      this->_lr_rank_factor1_B_setup         = str._lr_rank_factor1_B_setup;
      this->_lr_rank_factor2_B_setup         = str._lr_rank_factor2_B_setup;
      this->_lr_rank1_B_setup                = str._lr_rank1_B_setup;
      this->_lr_rank2_B_setup                = str._lr_rank2_B_setup;
      this->_lr_arnoldi_factor1_B_setup      = str._lr_arnoldi_factor1_B_setup;
      this->_lr_arnoldi_factor2_B_setup      = str._lr_arnoldi_factor2_B_setup;
      this->_lr_maxits1_B_setup              = str._lr_maxits1_B_setup;
      this->_lr_maxits2_B_setup              = str._lr_maxits2_B_setup;
      this->_lr_tol_eig1_B_setup             = str._lr_tol_eig1_B_setup;
      this->_lr_tol_eig2_B_setup             = str._lr_tol_eig2_B_setup;
      
      this->_B_solve_option1_B               = str._B_solve_option1_B;
      this->_B_solve_option1_levels_B        = str._B_solve_option1_levels_B;
      this->_B_solve_option2_B               = str._B_solve_option2_B;
      this->_C_solve_option_B                = str._C_solve_option_B;
      this->_B_ilu_tol_B_setup               = str._B_ilu_tol_B_setup;
      this->_C_ilu_tol_B_setup               = str._C_ilu_tol_B_setup;
      this->_B_ilu_max_row_nnz_B_setup       = str._B_ilu_max_row_nnz_B_setup;
      this->_C_ilu_max_row_nnz_B_setup       = str._C_ilu_max_row_nnz_B_setup;
      this->_B_ilu_fill_level_B_setup        = str._B_ilu_fill_level_B_setup;
      this->_C_ilu_fill_level_B_setup        = str._C_ilu_fill_level_B_setup;
      this->_B_poly_order_B                  = str._B_poly_order_B;
      
      this->_ilu_complex_shift               = str._ilu_complex_shift;
      this->_ilu_residual_iters              = str._ilu_residual_iters;
      
   }
   template GemslrLevelSetupStruct<float>::GemslrLevelSetupStruct(const GemslrLevelSetupStruct<float> &str);
   template GemslrLevelSetupStruct<double>::GemslrLevelSetupStruct(const GemslrLevelSetupStruct<double> &str);
   template GemslrLevelSetupStruct<complexs>::GemslrLevelSetupStruct(const GemslrLevelSetupStruct<complexs> &str);
   template GemslrLevelSetupStruct<complexd>::GemslrLevelSetupStruct(const GemslrLevelSetupStruct<complexd> &str);
   
   template <typename DataType>
   GemslrLevelSetupStruct<DataType>::GemslrLevelSetupStruct(GemslrLevelSetupStruct<DataType> &&str)
   {
      this->_lr_option1_setup                = str._lr_option1_setup;
      this->_lr_option2_setup                = str._lr_option2_setup;
      this->_lr_rand_init_setup              = str._lr_rand_init_setup;
      this->_lr_rank_factor1_setup           = str._lr_rank_factor1_setup;
      this->_lr_rank_factor2_setup           = str._lr_rank_factor2_setup;
      this->_lr_rank1_setup                  = str._lr_rank1_setup;
      this->_lr_rank2_setup                  = str._lr_rank2_setup;
      this->_lr_arnoldi_factor1_setup        = str._lr_arnoldi_factor1_setup;
      this->_lr_arnoldi_factor2_setup        = str._lr_arnoldi_factor2_setup;
      this->_lr_maxits1_setup                = str._lr_maxits1_setup;
      this->_lr_maxits2_setup                = str._lr_maxits2_setup;
      this->_lr_tol_eig1_setup               = str._lr_tol_eig1_setup;
      this->_lr_tol_eig2_setup               = str._lr_tol_eig2_setup;
      
      this->_B_solve_option1                 = str._B_solve_option1;
      this->_B_solve_option1_levels          = str._B_solve_option1_levels;
      this->_B_solve_option2                 = str._B_solve_option2;
      this->_C_solve_option                  = str._C_solve_option;
      this->_B_smooth_option1                = str._B_smooth_option1;
      this->_C_lr_pslr                       = str._C_lr_pslr;
      this->_B_ilu_tol_setup                 = str._B_ilu_tol_setup;
      this->_C_ilu_tol_setup                 = str._C_ilu_tol_setup;
      this->_S_ilu_tol_setup                 = str._S_ilu_tol_setup;
      this->_B_ilu_max_row_nnz_setup         = str._B_ilu_max_row_nnz_setup;
      this->_S_ilu_max_row_nnz_setup         = str._S_ilu_max_row_nnz_setup;
      this->_B_ilu_fill_level_setup          = str._B_ilu_fill_level_setup;
      this->_C_ilu_fill_level_setup          = str._C_ilu_fill_level_setup;
      this->_B_poly_order                    = str._B_poly_order;
      
      this->_lr_option1_B_setup              = str._lr_option1_B_setup;
      this->_lr_option2_B_setup              = str._lr_option2_B_setup;
      this->_lr_rand_init_B_setup            = str._lr_rand_init_B_setup;
      this->_lr_rank_factor1_B_setup         = str._lr_rank_factor1_B_setup;
      this->_lr_rank_factor2_B_setup         = str._lr_rank_factor2_B_setup;
      this->_lr_rank1_B_setup                = str._lr_rank1_B_setup;
      this->_lr_rank2_B_setup                = str._lr_rank2_B_setup;
      this->_lr_arnoldi_factor1_B_setup      = str._lr_arnoldi_factor1_B_setup;
      this->_lr_arnoldi_factor2_B_setup      = str._lr_arnoldi_factor2_B_setup;
      this->_lr_maxits1_B_setup              = str._lr_maxits1_B_setup;
      this->_lr_maxits2_B_setup              = str._lr_maxits2_B_setup;
      this->_lr_tol_eig1_B_setup             = str._lr_tol_eig1_B_setup;
      this->_lr_tol_eig2_B_setup             = str._lr_tol_eig2_B_setup;
      
      this->_B_solve_option1_B               = str._B_solve_option1_B;
      this->_B_solve_option1_levels_B        = str._B_solve_option1_levels_B;
      this->_B_solve_option2_B               = str._B_solve_option2_B;
      this->_C_solve_option_B                = str._C_solve_option_B;
      this->_B_ilu_tol_B_setup               = str._B_ilu_tol_B_setup;
      this->_C_ilu_tol_B_setup               = str._C_ilu_tol_B_setup;
      this->_B_ilu_max_row_nnz_B_setup       = str._B_ilu_max_row_nnz_B_setup;
      this->_C_ilu_max_row_nnz_B_setup       = str._C_ilu_max_row_nnz_B_setup;
      this->_B_ilu_fill_level_B_setup        = str._B_ilu_fill_level_B_setup;
      this->_C_ilu_fill_level_B_setup        = str._C_ilu_fill_level_B_setup;
      this->_B_poly_order_B                  = str._B_poly_order_B;
      
      this->_ilu_complex_shift               = str._ilu_complex_shift;
      this->_ilu_residual_iters              = str._ilu_residual_iters;
      
      str.SetDefault();
   }
   template GemslrLevelSetupStruct<float>::GemslrLevelSetupStruct(GemslrLevelSetupStruct<float> &&str);
   template GemslrLevelSetupStruct<double>::GemslrLevelSetupStruct(GemslrLevelSetupStruct<double> &&str);
   template GemslrLevelSetupStruct<complexs>::GemslrLevelSetupStruct(GemslrLevelSetupStruct<complexs> &&str);
   template GemslrLevelSetupStruct<complexd>::GemslrLevelSetupStruct(GemslrLevelSetupStruct<complexd> &&str);
   
   template <typename DataType>
   GemslrLevelSetupStruct<DataType>& GemslrLevelSetupStruct<DataType>::operator=(const GemslrLevelSetupStruct<DataType> &str)
   {
      this->_lr_option1_setup                = str._lr_option1_setup;
      this->_lr_option2_setup                = str._lr_option2_setup;
      this->_lr_rand_init_setup              = str._lr_rand_init_setup;
      this->_lr_rank_factor1_setup           = str._lr_rank_factor1_setup;
      this->_lr_rank_factor2_setup           = str._lr_rank_factor2_setup;
      this->_lr_rank1_setup                  = str._lr_rank1_setup;
      this->_lr_rank2_setup                  = str._lr_rank2_setup;
      this->_lr_arnoldi_factor1_setup        = str._lr_arnoldi_factor1_setup;
      this->_lr_arnoldi_factor2_setup        = str._lr_arnoldi_factor2_setup;
      this->_lr_maxits1_setup                = str._lr_maxits1_setup;
      this->_lr_maxits2_setup                = str._lr_maxits2_setup;
      this->_lr_tol_eig1_setup               = str._lr_tol_eig1_setup;
      this->_lr_tol_eig2_setup               = str._lr_tol_eig2_setup;
      
      this->_B_solve_option1                 = str._B_solve_option1;
      this->_B_solve_option1_levels          = str._B_solve_option1_levels;
      this->_B_solve_option2                 = str._B_solve_option2;
      this->_C_solve_option                  = str._C_solve_option;
      this->_B_smooth_option1                = str._B_smooth_option1;
      this->_C_lr_pslr                       = str._C_lr_pslr;
      this->_B_ilu_tol_setup                 = str._B_ilu_tol_setup;
      this->_C_ilu_tol_setup                 = str._C_ilu_tol_setup;
      this->_S_ilu_tol_setup                 = str._S_ilu_tol_setup;
      this->_B_ilu_max_row_nnz_setup         = str._B_ilu_max_row_nnz_setup;
      this->_S_ilu_max_row_nnz_setup         = str._S_ilu_max_row_nnz_setup;
      this->_B_ilu_fill_level_setup          = str._B_ilu_fill_level_setup;
      this->_C_ilu_fill_level_setup          = str._C_ilu_fill_level_setup;
      this->_B_poly_order                    = str._B_poly_order;
      
      this->_lr_option1_B_setup              = str._lr_option1_B_setup;
      this->_lr_option2_B_setup              = str._lr_option2_B_setup;
      this->_lr_rand_init_B_setup            = str._lr_rand_init_B_setup;
      this->_lr_rank_factor1_B_setup         = str._lr_rank_factor1_B_setup;
      this->_lr_rank_factor2_B_setup         = str._lr_rank_factor2_B_setup;
      this->_lr_rank1_B_setup                = str._lr_rank1_B_setup;
      this->_lr_rank2_B_setup                = str._lr_rank2_B_setup;
      this->_lr_arnoldi_factor1_B_setup      = str._lr_arnoldi_factor1_B_setup;
      this->_lr_arnoldi_factor2_B_setup      = str._lr_arnoldi_factor2_B_setup;
      this->_lr_maxits1_B_setup              = str._lr_maxits1_B_setup;
      this->_lr_maxits2_B_setup              = str._lr_maxits2_B_setup;
      this->_lr_tol_eig1_B_setup             = str._lr_tol_eig1_B_setup;
      this->_lr_tol_eig2_B_setup             = str._lr_tol_eig2_B_setup;
      
      this->_B_solve_option1_B               = str._B_solve_option1_B;
      this->_B_solve_option1_levels_B        = str._B_solve_option1_levels_B;
      this->_B_solve_option2_B               = str._B_solve_option2_B;
      this->_C_solve_option_B                = str._C_solve_option_B;
      this->_B_ilu_tol_B_setup               = str._B_ilu_tol_B_setup;
      this->_C_ilu_tol_B_setup               = str._C_ilu_tol_B_setup;
      this->_B_ilu_max_row_nnz_B_setup       = str._B_ilu_max_row_nnz_B_setup;
      this->_C_ilu_max_row_nnz_B_setup       = str._C_ilu_max_row_nnz_B_setup;
      this->_B_ilu_fill_level_B_setup        = str._B_ilu_fill_level_B_setup;
      this->_C_ilu_fill_level_B_setup        = str._C_ilu_fill_level_B_setup;
      this->_B_poly_order_B                  = str._B_poly_order_B;
      
      this->_ilu_complex_shift               = str._ilu_complex_shift;
      this->_ilu_residual_iters              = str._ilu_residual_iters;
      
      return *this;
   }
   template GemslrLevelSetupStruct<float>& GemslrLevelSetupStruct<float>::operator=(const GemslrLevelSetupStruct<float> &str);
   template GemslrLevelSetupStruct<double>& GemslrLevelSetupStruct<double>::operator=(const GemslrLevelSetupStruct<double> &str);
   template GemslrLevelSetupStruct<complexs>& GemslrLevelSetupStruct<complexs>::operator=(const GemslrLevelSetupStruct<complexs> &str);
   template GemslrLevelSetupStruct<complexd>& GemslrLevelSetupStruct<complexd>::operator=(const GemslrLevelSetupStruct<complexd> &str);
   
   template <typename DataType>
   GemslrLevelSetupStruct<DataType>& GemslrLevelSetupStruct<DataType>::operator=(GemslrLevelSetupStruct<DataType> &&str)
   {
      this->_lr_option1_setup                = str._lr_option1_setup;
      this->_lr_option2_setup                = str._lr_option2_setup;
      this->_lr_rand_init_setup              = str._lr_rand_init_setup;
      this->_lr_rank_factor1_setup           = str._lr_rank_factor1_setup;
      this->_lr_rank_factor2_setup           = str._lr_rank_factor2_setup;
      this->_lr_rank1_setup                  = str._lr_rank1_setup;
      this->_lr_rank2_setup                  = str._lr_rank2_setup;
      this->_lr_arnoldi_factor1_setup        = str._lr_arnoldi_factor1_setup;
      this->_lr_arnoldi_factor2_setup        = str._lr_arnoldi_factor2_setup;
      this->_lr_maxits1_setup                = str._lr_maxits1_setup;
      this->_lr_maxits2_setup                = str._lr_maxits2_setup;
      this->_lr_tol_eig1_setup               = str._lr_tol_eig1_setup;
      this->_lr_tol_eig2_setup               = str._lr_tol_eig2_setup;
      
      this->_B_solve_option1                 = str._B_solve_option1;
      this->_B_solve_option1_levels          = str._B_solve_option1_levels;
      this->_B_solve_option2                 = str._B_solve_option2;
      this->_C_solve_option                  = str._C_solve_option;
      this->_B_smooth_option1                = str._B_smooth_option1;
      this->_C_lr_pslr                       = str._C_lr_pslr;
      this->_B_ilu_tol_setup                 = str._B_ilu_tol_setup;
      this->_C_ilu_tol_setup                 = str._C_ilu_tol_setup;
      this->_S_ilu_tol_setup                 = str._S_ilu_tol_setup;
      this->_B_ilu_max_row_nnz_setup         = str._B_ilu_max_row_nnz_setup;
      this->_S_ilu_max_row_nnz_setup         = str._S_ilu_max_row_nnz_setup;
      this->_B_ilu_fill_level_setup          = str._B_ilu_fill_level_setup;
      this->_C_ilu_fill_level_setup          = str._C_ilu_fill_level_setup;
      this->_B_poly_order                    = str._B_poly_order;
      
      this->_lr_option1_B_setup              = str._lr_option1_B_setup;
      this->_lr_option2_B_setup              = str._lr_option2_B_setup;
      this->_lr_rand_init_B_setup            = str._lr_rand_init_B_setup;
      this->_lr_rank_factor1_B_setup         = str._lr_rank_factor1_B_setup;
      this->_lr_rank_factor2_B_setup         = str._lr_rank_factor2_B_setup;
      this->_lr_rank1_B_setup                = str._lr_rank1_B_setup;
      this->_lr_rank2_B_setup                = str._lr_rank2_B_setup;
      this->_lr_arnoldi_factor1_B_setup      = str._lr_arnoldi_factor1_B_setup;
      this->_lr_arnoldi_factor2_B_setup      = str._lr_arnoldi_factor2_B_setup;
      this->_lr_maxits1_B_setup              = str._lr_maxits1_B_setup;
      this->_lr_maxits2_B_setup              = str._lr_maxits2_B_setup;
      this->_lr_tol_eig1_B_setup             = str._lr_tol_eig1_B_setup;
      this->_lr_tol_eig2_B_setup             = str._lr_tol_eig2_B_setup;
      
      this->_B_solve_option1_B               = str._B_solve_option1_B;
      this->_B_solve_option1_levels_B        = str._B_solve_option1_levels_B;
      this->_B_solve_option2_B               = str._B_solve_option2_B;
      this->_C_solve_option_B                = str._C_solve_option_B;
      this->_B_ilu_tol_B_setup               = str._B_ilu_tol_B_setup;
      this->_C_ilu_tol_B_setup               = str._C_ilu_tol_B_setup;
      this->_B_ilu_max_row_nnz_B_setup       = str._B_ilu_max_row_nnz_B_setup;
      this->_C_ilu_max_row_nnz_B_setup       = str._C_ilu_max_row_nnz_B_setup;
      this->_B_ilu_fill_level_B_setup        = str._B_ilu_fill_level_B_setup;
      this->_C_ilu_fill_level_B_setup        = str._C_ilu_fill_level_B_setup;
      this->_B_poly_order_B                  = str._B_poly_order_B;
      
      this->_ilu_complex_shift               = str._ilu_complex_shift;
      this->_ilu_residual_iters              = str._ilu_residual_iters;
      
      str.SetDefault();
      
      return *this;
   }
   template GemslrLevelSetupStruct<float>& GemslrLevelSetupStruct<float>::operator=(GemslrLevelSetupStruct<float> &&str);
   template GemslrLevelSetupStruct<double>& GemslrLevelSetupStruct<double>::operator=(GemslrLevelSetupStruct<double> &&str);
   template GemslrLevelSetupStruct<complexs>& GemslrLevelSetupStruct<complexs>::operator=(GemslrLevelSetupStruct<complexs> &&str);
   template GemslrLevelSetupStruct<complexd>& GemslrLevelSetupStruct<complexd>::operator=(GemslrLevelSetupStruct<complexd> &&str);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrLevelClass<MatrixType, VectorType, DataType>::GemslrLevelClass()
   {
      this->_lrc                             = 0;
      this->_nmvs                            = 0;
      this->_ncomps                          = 0;
      this->_B_precond                       = NULL;
      this->_B_solver                        = NULL;
   }
   template precond_gemslrlevel_csr_seq_float::GemslrLevelClass();
   template precond_gemslrlevel_csr_seq_double::GemslrLevelClass();
   template precond_gemslrlevel_csr_seq_complexs::GemslrLevelClass();
   template precond_gemslrlevel_csr_seq_complexd::GemslrLevelClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrLevelClass<MatrixType, VectorType, DataType>::~GemslrLevelClass()
   {
      this->Clear();
   }
   template precond_gemslrlevel_csr_seq_float::~GemslrLevelClass();
   template precond_gemslrlevel_csr_seq_double::~GemslrLevelClass();
   template precond_gemslrlevel_csr_seq_complexs::~GemslrLevelClass();
   template precond_gemslrlevel_csr_seq_complexd::~GemslrLevelClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrLevelClass<MatrixType, VectorType, DataType>::GemslrLevelClass(const GemslrLevelClass<MatrixType, VectorType, DataType> &str)
   {
      int i;
      
      this->_lrc                             = str._lrc;
      this->_nmvs                            = str._nmvs;
      this->_ncomps                          = str._ncomps;
      
      if(str._B_solver)
      {
         PARGEMSLR_MALLOC( this->_B_solver, this->_ncomps, kMemoryHost, SolverClass<MatrixType, VectorType, DataType>*);
         switch(str._B_solver[0]->GetSolverType())
         {
            case kSolverIlu:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, IluClass<MatrixType, VectorType, DataType>);
                  
                  IluClass<MatrixType, VectorType, DataType> &ilu1 = *(IluClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  IluClass<MatrixType, VectorType, DataType> &ilu2 = *(IluClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
                  ilu2 = ilu1;
                  
               }
               
               break;
            }
            case kSolverPoly:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, PolyClass<MatrixType, VectorType, DataType>);
                  
                  PolyClass<MatrixType, VectorType, DataType> &poly1 = *(PolyClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  PolyClass<MatrixType, VectorType, DataType> &poly2 = *(PolyClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
                  poly2 = poly1;
                  
               }
               
               break;
            }
            case kSolverGemslr:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, GemslrClass<MatrixType, VectorType, DataType>);
                  
                  GemslrClass<MatrixType, VectorType, DataType> &gemslr1 = *(GemslrClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  GemslrClass<MatrixType, VectorType, DataType> &gemslr2 = *(GemslrClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
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
      
      this->_y_temp = str._y_temp;
      this->_z_temp = str._z_temp;
      this->_v_temp = str._v_temp;
      this->_w_temp = str._w_temp;
      this->_xlr_temp = str._xlr_temp;
      this->_xlr1_temp = str._xlr1_temp;
      this->_xlr2_temp = str._xlr2_temp;
      
      this->_E_mat = str._E_mat;
      this->_F_mat = str._F_mat;
      this->_C_mat = str._C_mat;
      this->_D_mat = str._D_mat;
      this->_EBFC = str._EBFC;
      this->_Hk = str._Hk;
      this->_Wk = str._Wk;
      
   }
   template precond_gemslrlevel_csr_seq_float::GemslrLevelClass(const precond_gemslrlevel_csr_seq_float &str);
   template precond_gemslrlevel_csr_seq_double::GemslrLevelClass(const precond_gemslrlevel_csr_seq_double &str);
   template precond_gemslrlevel_csr_seq_complexs::GemslrLevelClass(const precond_gemslrlevel_csr_seq_complexs &str);
   template precond_gemslrlevel_csr_seq_complexd::GemslrLevelClass(const precond_gemslrlevel_csr_seq_complexd &str);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrLevelClass<MatrixType, VectorType, DataType>::GemslrLevelClass(GemslrLevelClass<MatrixType, VectorType, DataType> &&str)
   {
      int i;
      
      this->_lrc                             = str._lrc;str._lrc = 0;
      this->_nmvs                            = str._nmvs;str._nmvs = 0;
      this->_ncomps                          = str._ncomps;str._ncomps = 0;
      
      if(str._B_solver)
      {
         PARGEMSLR_MALLOC( this->_B_solver, this->_ncomps, kMemoryHost, SolverClass<MatrixType, VectorType, DataType>*);
         switch(str._B_solver[0]->GetSolverType())
         {
            case kSolverIlu:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, IluClass<MatrixType, VectorType, DataType>);
                  
                  IluClass<MatrixType, VectorType, DataType> &ilu1 = *(IluClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  IluClass<MatrixType, VectorType, DataType> &ilu2 = *(IluClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
                  ilu2 = ilu1;
                  PARGEMSLR_FREE(str._B_solver[i], kMemoryHost);
               }
               
               break;
            }
            case kSolverPoly:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, PolyClass<MatrixType, VectorType, DataType>);
                  
                  PolyClass<MatrixType, VectorType, DataType> &poly1 = *(PolyClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  PolyClass<MatrixType, VectorType, DataType> &poly2 = *(PolyClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
                  poly2 = poly1;
                  PARGEMSLR_FREE(str._B_solver[i], kMemoryHost);
               }
               
               break;
            }
            case kSolverGemslr:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, GemslrClass<MatrixType, VectorType, DataType>);
                  
                  GemslrClass<MatrixType, VectorType, DataType> &gemslr1 = *(GemslrClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  GemslrClass<MatrixType, VectorType, DataType> &gemslr2 = *(GemslrClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
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
      std::vector<MatrixType >().swap(str._B_mat_v);
      
      this->_y_temp = std::move(str._y_temp);
      this->_z_temp = std::move(str._z_temp);
      this->_v_temp = std::move(str._v_temp);
      this->_w_temp = std::move(str._w_temp);
      this->_xlr_temp = std::move(str._xlr_temp);
      this->_xlr1_temp = std::move(str._xlr1_temp);
      this->_xlr2_temp = std::move(str._xlr2_temp);
      
      this->_E_mat = std::move(str._E_mat);
      this->_F_mat = std::move(str._F_mat);
      this->_C_mat = std::move(str._C_mat);
      this->_D_mat = std::move(str._D_mat);
      this->_EBFC = std::move(str._EBFC);
      this->_Hk = std::move(str._Hk);
      this->_Wk = std::move(str._Wk);
   }
   template precond_gemslrlevel_csr_seq_float::GemslrLevelClass(precond_gemslrlevel_csr_seq_float &&str);
   template precond_gemslrlevel_csr_seq_double::GemslrLevelClass(precond_gemslrlevel_csr_seq_double &&str);
   template precond_gemslrlevel_csr_seq_complexs::GemslrLevelClass(precond_gemslrlevel_csr_seq_complexs &&str);
   template precond_gemslrlevel_csr_seq_complexd::GemslrLevelClass(precond_gemslrlevel_csr_seq_complexd &&str);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrLevelClass<MatrixType, VectorType, DataType>& GemslrLevelClass<MatrixType, VectorType, DataType>::operator=(const GemslrLevelClass<MatrixType, VectorType, DataType> &str)
   {
      this->Clear();
      
      int i;
      
      this->_lrc                             = str._lrc;
      this->_nmvs                            = str._nmvs;
      this->_ncomps                          = str._ncomps;
      
      if(str._B_solver)
      {
         PARGEMSLR_MALLOC( this->_B_solver, this->_ncomps, kMemoryHost, SolverClass<MatrixType, VectorType, DataType>*);
         switch(str._B_solver[0]->GetSolverType())
         {
            case kSolverIlu:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, IluClass<MatrixType, VectorType, DataType>);
                  
                  IluClass<MatrixType, VectorType, DataType> &ilu1 = *(IluClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  IluClass<MatrixType, VectorType, DataType> &ilu2 = *(IluClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
                  ilu2 = ilu1;
                  
               }
               
               break;
            }
            case kSolverPoly:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, PolyClass<MatrixType, VectorType, DataType>);
                  
                  PolyClass<MatrixType, VectorType, DataType> &poly1 = *(PolyClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  PolyClass<MatrixType, VectorType, DataType> &poly2 = *(PolyClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
                  poly2 = poly1;
                  
               }
               
               break;
            }
            case kSolverGemslr:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, GemslrClass<MatrixType, VectorType, DataType>);
                  
                  GemslrClass<MatrixType, VectorType, DataType> &gemslr1 = *(GemslrClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  GemslrClass<MatrixType, VectorType, DataType> &gemslr2 = *(GemslrClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
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
      
      this->_y_temp = str._y_temp;
      this->_z_temp = str._z_temp;
      this->_v_temp = str._v_temp;
      this->_w_temp = str._w_temp;
      this->_xlr_temp = str._xlr_temp;
      this->_xlr1_temp = str._xlr1_temp;
      this->_xlr2_temp = str._xlr2_temp;
      
      this->_E_mat = str._E_mat;
      this->_F_mat = str._F_mat;
      this->_C_mat = str._C_mat;
      this->_D_mat = str._D_mat;
      this->_EBFC = str._EBFC;
      this->_Hk = str._Hk;
      this->_Wk = str._Wk;
      
      return *this;
   }
   template precond_gemslrlevel_csr_seq_float& precond_gemslrlevel_csr_seq_float::operator=(const precond_gemslrlevel_csr_seq_float &str);
   template precond_gemslrlevel_csr_seq_double& precond_gemslrlevel_csr_seq_double::operator=(const precond_gemslrlevel_csr_seq_double &str);
   template precond_gemslrlevel_csr_seq_complexs& precond_gemslrlevel_csr_seq_complexs::operator=(const precond_gemslrlevel_csr_seq_complexs &str);
   template precond_gemslrlevel_csr_seq_complexd& precond_gemslrlevel_csr_seq_complexd::operator=(const precond_gemslrlevel_csr_seq_complexd &str);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrLevelClass<MatrixType, VectorType, DataType>& GemslrLevelClass<MatrixType, VectorType, DataType>::operator=(GemslrLevelClass<MatrixType, VectorType, DataType> &&str)
   {
      this->Clear();
      
      int i;
      
      this->_lrc                             = str._lrc;str._lrc = 0;
      this->_nmvs                            = str._nmvs;str._nmvs = 0;
      this->_ncomps                          = str._ncomps;str._ncomps = 0;
      
      if(str._B_solver)
      {
         PARGEMSLR_MALLOC( this->_B_solver, this->_ncomps, kMemoryHost, SolverClass<MatrixType, VectorType, DataType>*);
         switch(str._B_solver[0]->GetSolverType())
         {
            case kSolverIlu:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, IluClass<MatrixType, VectorType, DataType>);
                  
                  IluClass<MatrixType, VectorType, DataType> &ilu1 = *(IluClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  IluClass<MatrixType, VectorType, DataType> &ilu2 = *(IluClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
                  ilu2 = ilu1;
                  PARGEMSLR_FREE(str._B_solver[i], kMemoryHost);
               }
               
               break;
            }
            case kSolverPoly:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, PolyClass<MatrixType, VectorType, DataType>);
                  
                  PolyClass<MatrixType, VectorType, DataType> &poly1 = *(PolyClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  PolyClass<MatrixType, VectorType, DataType> &poly2 = *(PolyClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
                  poly2 = poly1;
                  PARGEMSLR_FREE(str._B_solver[i], kMemoryHost);
               }
               
               break;
            }
            case kSolverGemslr:
            {
               /* ILU solve on the next level */
               for(i = 0 ; i < this->_ncomps ; i ++)
               {
                  PARGEMSLR_PLACEMENT_NEW( this->_B_solver[i], kMemoryHost, GemslrClass<MatrixType, VectorType, DataType>);
                  
                  GemslrClass<MatrixType, VectorType, DataType> &gemslr1 = *(GemslrClass<MatrixType, VectorType, DataType>*) str._B_solver[i];
                  GemslrClass<MatrixType, VectorType, DataType> &gemslr2 = *(GemslrClass<MatrixType, VectorType, DataType>*) this->_B_solver[i];
                  
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
      std::vector<MatrixType >().swap(str._B_mat_v);
      
      this->_y_temp = std::move(str._y_temp);
      this->_z_temp = std::move(str._z_temp);
      this->_v_temp = std::move(str._v_temp);
      this->_w_temp = std::move(str._w_temp);
      this->_xlr_temp = std::move(str._xlr_temp);
      this->_xlr1_temp = std::move(str._xlr1_temp);
      this->_xlr2_temp = std::move(str._xlr2_temp);
      
      this->_E_mat = std::move(str._E_mat);
      this->_E_mat = std::move(str._E_mat);
      this->_C_mat = std::move(str._C_mat);
      this->_D_mat = std::move(str._D_mat);
      this->_EBFC = std::move(str._EBFC);
      this->_Hk = std::move(str._Hk);
      this->_Wk = std::move(str._Wk);
      
      return *this;
   }
   template precond_gemslrlevel_csr_seq_float& precond_gemslrlevel_csr_seq_float::operator=(precond_gemslrlevel_csr_seq_float &&str);
   template precond_gemslrlevel_csr_seq_double& precond_gemslrlevel_csr_seq_double::operator=(precond_gemslrlevel_csr_seq_double &&str);
   template precond_gemslrlevel_csr_seq_complexs& precond_gemslrlevel_csr_seq_complexs::operator=(precond_gemslrlevel_csr_seq_complexs &&str);
   template precond_gemslrlevel_csr_seq_complexd& precond_gemslrlevel_csr_seq_complexd::operator=(precond_gemslrlevel_csr_seq_complexd &&str);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrLevelClass<MatrixType, VectorType, DataType>::Clear()
   {
      int i;
      
      this->_lrc                             = 0;
      this->_nmvs                            = 0;
      if(this->_B_solver)
      {
         for(i = 0 ; i < this->_ncomps ; i ++)
         {
            this->_B_solver[i]->Clear();
            PARGEMSLR_FREE(this->_B_solver[i], kMemoryHost);
         }
         PARGEMSLR_FREE(this->_B_solver, kMemoryHost);
      }
      if(this->_B_precond)
      {
         for(i = 0 ; i < this->_ncomps ; i ++)
         {
            this->_B_precond[i]->Clear();
            PARGEMSLR_FREE(this->_B_precond[i], kMemoryHost);
         }
         PARGEMSLR_FREE(this->_B_precond, kMemoryHost);
      }
      
      for(i = 0 ; i < this->_ncomps ;  i++)
      {
         this->_B_mat_v[i].Clear();
      }
      std::vector<MatrixType >().swap(this->_B_mat_v);
      this->_y_temp.Clear();
      this->_z_temp.Clear();
      this->_v_temp.Clear();
      this->_w_temp.Clear();
      this->_xlr_temp.Clear();
      this->_xlr1_temp.Clear();
      this->_xlr2_temp.Clear();
      
      this->_ncomps                          = 0;
      this->_E_mat.Clear();
      this->_F_mat.Clear();
      this->_C_mat.Clear();
      this->_D_mat.Clear();
      this->_EBFC.Clear();
      this->_Hk.Clear();
      this->_Wk.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslrlevel_csr_seq_float::Clear();
   template int precond_gemslrlevel_csr_seq_double::Clear();
   template int precond_gemslrlevel_csr_seq_complexs::Clear();
   template int precond_gemslrlevel_csr_seq_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrLevelClass<MatrixType, VectorType, DataType>::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr)
   {
      int i;
      
      nnz_bsolver = 0;
      
      if(this->_B_precond)
      {
         for(i = 0 ; i < _ncomps ; i ++)
         {
            nnz_bsolver += this->_B_precond[i]->GetNumNonzeros();
         }
      }
      
      if(this->_B_solver)
      {
         for(i = 0 ; i < _ncomps ; i ++)
         {
            nnz_bsolver += this->_B_solver[i]->GetNumNonzeros();
         }
      }
      
      nnz_lr = this->_Wk.GetNumNonzeros() + this->_Hk.GetNumNonzeros();
      
      return 0;
   }
   template int precond_gemslrlevel_csr_seq_float::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template int precond_gemslrlevel_csr_seq_double::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template int precond_gemslrlevel_csr_seq_complexs::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template int precond_gemslrlevel_csr_seq_complexd::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   
   template <typename DataType>
   GemslrSetupStruct<DataType>::GemslrSetupStruct()
   {
      this->_location                                    = kMemoryHost;
      this->_partition_option_setup                      = kGemslrPartitionRKway;
      this->_partition_option_B_setup                    = kGemslrPartitionND;
      this->_perm_option_setup                           = kIluReorderingRcm;
      this->_perm_option_B_setup                         = kIluReorderingRcm;
      this->_nlev_setup                                  = 5;
      this->_nlev_B_setup                                = 5;
      this->_ncomp_setup                                 = 4;
      this->_ncomp_B_setup                               = 4;
      this->_kmin_setup                                  = 4;
      this->_kmin_B_setup                                = 4;
      this->_kfactor_setup                               = 1;
      this->_kfactor_B_setup                             = 1;
      this->_vertexsep_setup                             = true;
      this->_vertexsep_B_setup                           = true;
      this->_solve_phase_setup                           = kGemslrPhaseSetup;
      this->_cuda_lowrank_only                           = false;
      this->_enable_inner_iters_setup                    = true;
      this->_inner_iters_tol_setup                       = 1e-02;
      this->_inner_iters_maxits_setup                    = 5;
      this->_solve_option_setup                          = kGemslrCSolveILUT;
      this->_diag_shift_milu                             = DataType();
   }
   template GemslrSetupStruct<float>::GemslrSetupStruct();
   template GemslrSetupStruct<double>::GemslrSetupStruct();
   template GemslrSetupStruct<complexs>::GemslrSetupStruct();
   template GemslrSetupStruct<complexd>::GemslrSetupStruct();

   template <typename DataType>
   void GemslrSetupStruct<DataType>::SetDefault()
   {
      this->_location                                    = kMemoryHost;
      this->_partition_option_setup                      = kGemslrPartitionRKway;
      this->_partition_option_B_setup                    = kGemslrPartitionND;
      this->_perm_option_setup                           = kIluReorderingRcm;
      this->_perm_option_B_setup                         = kIluReorderingRcm;
      this->_nlev_setup                                  = 5;
      this->_nlev_B_setup                                = 5;
      this->_ncomp_setup                                 = 4;
      this->_ncomp_B_setup                               = 4;
      this->_kmin_setup                                  = 4;
      this->_kmin_B_setup                                = 4;
      this->_kfactor_setup                               = 1;
      this->_kfactor_B_setup                             = 1;
      this->_vertexsep_setup                             = true;
      this->_vertexsep_B_setup                           = true;
      this->_solve_phase_setup                           = kGemslrPhaseSetup;
      this->_cuda_lowrank_only                           = false;
      this->_enable_inner_iters_setup                    = true;
      this->_inner_iters_tol_setup                       = 1e-02;
      this->_inner_iters_maxits_setup                    = 5;
      this->_solve_option_setup                          = kGemslrCSolveILUT;
      this->_diag_shift_milu                             = DataType();
   }
   template void GemslrSetupStruct<float>::SetDefault();
   template void GemslrSetupStruct<double>::SetDefault();
   template void GemslrSetupStruct<complexs>::SetDefault();
   template void GemslrSetupStruct<complexd>::SetDefault();

   template <typename DataType>
   GemslrSetupStruct<DataType>::GemslrSetupStruct(const GemslrSetupStruct<DataType> &str)
   {
      this->SetDefault();
      
      this->_location                                    = str._location;
      this->_partition_option_setup                      = str._partition_option_setup;
      this->_partition_option_B_setup                    = str._partition_option_B_setup;
      this->_perm_option_setup                           = str._perm_option_setup;
      this->_perm_option_B_setup                         = str._perm_option_B_setup;
      this->_nlev_setup                                  = str._nlev_setup;
      this->_nlev_B_setup                                = str._nlev_B_setup;
      this->_ncomp_setup                                 = str._ncomp_setup;
      this->_ncomp_B_setup                               = str._ncomp_B_setup;
      this->_kmin_setup                                  = str._kmin_setup;
      this->_kmin_B_setup                                = str._kmin_B_setup;
      this->_kfactor_setup                               = str._kfactor_setup;
      this->_kfactor_B_setup                             = str._kfactor_B_setup;
      this->_vertexsep_setup                             = str._vertexsep_setup;
      this->_vertexsep_B_setup                           = str._vertexsep_B_setup;
      this->_solve_phase_setup                           = str._solve_phase_setup;
      this->_cuda_lowrank_only                           = str._cuda_lowrank_only;
      this->_enable_inner_iters_setup                    = str._enable_inner_iters_setup;
      this->_inner_iters_tol_setup                       = str._inner_iters_tol_setup;
      this->_inner_iters_maxits_setup                    = str._inner_iters_maxits_setup;
      this->_solve_option_setup                          = str._solve_option_setup;
      this->_diag_shift_milu                             = str._diag_shift_milu;
   }
   template GemslrSetupStruct<float>::GemslrSetupStruct(const GemslrSetupStruct<float> &str);
   template GemslrSetupStruct<double>::GemslrSetupStruct(const GemslrSetupStruct<double> &str);
   template GemslrSetupStruct<complexs>::GemslrSetupStruct(const GemslrSetupStruct<complexs> &str);
   template GemslrSetupStruct<complexd>::GemslrSetupStruct(const GemslrSetupStruct<complexd> &str);

   template <typename DataType>
   GemslrSetupStruct<DataType>::GemslrSetupStruct(GemslrSetupStruct<DataType> &&str)
   {
      this->SetDefault();
      
      this->_location                                    = str._location;
      this->_partition_option_setup                      = str._partition_option_setup;
      this->_partition_option_B_setup                    = str._partition_option_B_setup;
      this->_perm_option_setup                           = str._perm_option_setup;
      this->_perm_option_B_setup                         = str._perm_option_B_setup;
      this->_nlev_setup                                  = str._nlev_setup;
      this->_nlev_B_setup                                = str._nlev_B_setup;
      this->_ncomp_setup                                 = str._ncomp_setup;
      this->_ncomp_B_setup                               = str._ncomp_B_setup;
      this->_kmin_setup                                  = str._kmin_setup;
      this->_kmin_B_setup                                = str._kmin_B_setup;
      this->_kfactor_setup                               = str._kfactor_setup;
      this->_kfactor_B_setup                             = str._kfactor_B_setup;
      this->_vertexsep_setup                             = str._vertexsep_setup;
      this->_vertexsep_B_setup                           = str._vertexsep_B_setup;
      this->_solve_phase_setup                           = str._solve_phase_setup;
      this->_cuda_lowrank_only                           = str._cuda_lowrank_only;
      this->_enable_inner_iters_setup                    = str._enable_inner_iters_setup;
      this->_inner_iters_tol_setup                       = str._inner_iters_tol_setup;
      this->_inner_iters_maxits_setup                    = str._inner_iters_maxits_setup;
      this->_solve_option_setup                          = str._solve_option_setup;
      this->_diag_shift_milu                             = str._diag_shift_milu;
      
      str.SetDefault();
      
   }
   template GemslrSetupStruct<float>::GemslrSetupStruct(GemslrSetupStruct<float> &&str);
   template GemslrSetupStruct<double>::GemslrSetupStruct(GemslrSetupStruct<double> &&str);
   template GemslrSetupStruct<complexs>::GemslrSetupStruct(GemslrSetupStruct<complexs> &&str);
   template GemslrSetupStruct<complexd>::GemslrSetupStruct(GemslrSetupStruct<complexd> &&str);

   template <typename DataType>
   GemslrSetupStruct<DataType>& GemslrSetupStruct<DataType>::operator=(const GemslrSetupStruct<DataType> &str)
   {
      this->SetDefault();
      
      this->_location                                    = str._location;
      this->_partition_option_setup                      = str._partition_option_setup;
      this->_partition_option_B_setup                    = str._partition_option_B_setup;
      this->_perm_option_setup                           = str._perm_option_setup;
      this->_perm_option_B_setup                         = str._perm_option_B_setup;
      this->_nlev_setup                                  = str._nlev_setup;
      this->_nlev_B_setup                                = str._nlev_B_setup;
      this->_ncomp_setup                                 = str._ncomp_setup;
      this->_ncomp_B_setup                               = str._ncomp_B_setup;
      this->_kmin_setup                                  = str._kmin_setup;
      this->_kmin_B_setup                                = str._kmin_B_setup;
      this->_kfactor_setup                               = str._kfactor_setup;
      this->_kfactor_B_setup                             = str._kfactor_B_setup;
      this->_vertexsep_setup                             = str._vertexsep_setup;
      this->_vertexsep_B_setup                           = str._vertexsep_B_setup;
      this->_solve_phase_setup                           = str._solve_phase_setup;
      this->_cuda_lowrank_only                           = str._cuda_lowrank_only;
      this->_enable_inner_iters_setup                    = str._enable_inner_iters_setup;
      this->_inner_iters_tol_setup                       = str._inner_iters_tol_setup;
      this->_inner_iters_maxits_setup                    = str._inner_iters_maxits_setup;
      this->_solve_option_setup                          = str._solve_option_setup;
      this->_diag_shift_milu                             = str._diag_shift_milu;
      
      return *this;
   }
   template GemslrSetupStruct<float>& GemslrSetupStruct<float>::operator=(const GemslrSetupStruct<float> &str);
   template GemslrSetupStruct<double>& GemslrSetupStruct<double>::operator=(const GemslrSetupStruct<double> &str);
   template GemslrSetupStruct<complexs>& GemslrSetupStruct<complexs>::operator=(const GemslrSetupStruct<complexs> &str);
   template GemslrSetupStruct<complexd>& GemslrSetupStruct<complexd>::operator=(const GemslrSetupStruct<complexd> &str);

   template <typename DataType>
   GemslrSetupStruct<DataType>& GemslrSetupStruct<DataType>::operator=(GemslrSetupStruct<DataType> &&str)
   {
      this->SetDefault();
      
      this->_location                                    = str._location;
      this->_partition_option_setup                      = str._partition_option_setup;
      this->_partition_option_B_setup                    = str._partition_option_B_setup;
      this->_perm_option_setup                           = str._perm_option_setup;
      this->_perm_option_B_setup                         = str._perm_option_B_setup;
      this->_nlev_setup                                  = str._nlev_setup;
      this->_nlev_B_setup                                = str._nlev_B_setup;
      this->_ncomp_setup                                 = str._ncomp_setup;
      this->_ncomp_B_setup                               = str._ncomp_B_setup;
      this->_kmin_setup                                  = str._kmin_setup;
      this->_kmin_B_setup                                = str._kmin_B_setup;
      this->_kfactor_setup                               = str._kfactor_setup;
      this->_kfactor_B_setup                             = str._kfactor_B_setup;
      this->_vertexsep_setup                             = str._vertexsep_setup;
      this->_vertexsep_B_setup                           = str._vertexsep_B_setup;
      this->_solve_phase_setup                           = str._solve_phase_setup;
      this->_cuda_lowrank_only                           = str._cuda_lowrank_only;
      this->_enable_inner_iters_setup                    = str._enable_inner_iters_setup;
      this->_inner_iters_tol_setup                       = str._inner_iters_tol_setup;
      this->_inner_iters_maxits_setup                    = str._inner_iters_maxits_setup;
      this->_solve_option_setup                          = str._solve_option_setup;
      this->_diag_shift_milu                             = str._diag_shift_milu;
      
      str.SetDefault();
      
      return *this;
   }
   template GemslrSetupStruct<float>& GemslrSetupStruct<float>::operator=(GemslrSetupStruct<float> &&str);
   template GemslrSetupStruct<double>& GemslrSetupStruct<double>::operator=(GemslrSetupStruct<double> &&str);
   template GemslrSetupStruct<complexs>& GemslrSetupStruct<complexs>::operator=(GemslrSetupStruct<complexs> &&str);
   template GemslrSetupStruct<complexd>& GemslrSetupStruct<complexd>::operator=(GemslrSetupStruct<complexd> &&str);

   template <class MatrixType, class VectorType, typename DataType>
   GemslrClass<MatrixType, VectorType, DataType>::GemslrClass() : SolverClass<MatrixType, VectorType, DataType>()
   {
      
      this->_n                                  = 0;
      this->_solver_type                        = kSolverGemslr;
      this->_nlev_max                           = 0;
      this->_nlev_used                          = 0;
      this->_location                           = kMemoryHost;
      
   }
   template precond_gemslr_csr_seq_float::GemslrClass();
   template precond_gemslr_csr_seq_double::GemslrClass();
   template precond_gemslr_csr_seq_complexs::GemslrClass();
   template precond_gemslr_csr_seq_complexd::GemslrClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrClass<MatrixType, VectorType, DataType>::GemslrClass(const GemslrClass<MatrixType, VectorType, DataType> &precond) : SolverClass<MatrixType, VectorType, DataType>(precond)
   {
      int i;
      this->_n = precond._n;
      this->_gemslr_setups = precond._gemslr_setups;
      this->_inner_iters_matrix = precond._inner_iters_matrix;
      this->_inner_iters_precond = precond._inner_iters_precond;
      this->_inner_iters_solver = precond._inner_iters_solver;
      this->_nlev_max = precond._nlev_max;
      this->_nlev_used = precond._nlev_used;
      this->_location = precond._location;
      
      this->_lev_ptr_v = precond._lev_ptr_v;
      this->_dom_ptr_v2.resize(this->_nlev_used);
      this->_levs_v.resize(this->_nlev_used);
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i] = precond._levs_v[i];
         this->_dom_ptr_v2[i] = precond._dom_ptr_v2[i];
      }
      
      this->_pperm = precond._pperm;
      this->_qperm = precond._qperm;
      this->_x_temp = precond._x_temp;
      this->_rhs_temp = precond._rhs_temp;
      
   }
   template precond_gemslr_csr_seq_float::GemslrClass(const precond_gemslr_csr_seq_float &precond);
   template precond_gemslr_csr_seq_double::GemslrClass(const precond_gemslr_csr_seq_double &precond);
   template precond_gemslr_csr_seq_complexs::GemslrClass(const precond_gemslr_csr_seq_complexs &precond);
   template precond_gemslr_csr_seq_complexd::GemslrClass(const precond_gemslr_csr_seq_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrClass<MatrixType, VectorType, DataType>::GemslrClass(GemslrClass<MatrixType, VectorType, DataType> &&precond) : SolverClass<MatrixType, VectorType, DataType>(std::move(precond))
   {
      int i;
      this->_n = precond._n;precond._n = 0;
      this->_gemslr_setups = precond._gemslr_setups;precond._gemslr_setups.SetDefault();
      this->_inner_iters_matrix = std::move(precond._inner_iters_matrix);
      this->_inner_iters_precond = std::move(precond._inner_iters_precond);
      this->_inner_iters_solver = std::move(precond._inner_iters_solver);
      this->_nlev_max = precond._nlev_max;precond._nlev_max = 0;
      this->_nlev_used = precond._nlev_used;precond._nlev_used = 0;
      this->_location = precond._location;precond._location = 0;
      
      this->_lev_ptr_v = std::move(precond._lev_ptr_v);
      this->_dom_ptr_v2.resize(this->_nlev_used);
      this->_levs_v.resize(this->_nlev_used);
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i] = std::move(precond._levs_v[i]);
         this->_dom_ptr_v2[i] = std::move(precond._dom_ptr_v2[i]);
      }
      std::vector<IntVectorClass<int> >().swap(precond._dom_ptr_v2);      
      std::vector<GemslrLevelClass< MatrixType, VectorType, DataType> >().swap(precond._levs_v);      
      
      this->_pperm = std::move(precond._pperm);
      this->_qperm = std::move(precond._qperm);
      this->_x_temp = std::move(precond._x_temp);
      this->_rhs_temp = std::move(precond._rhs_temp);
      
   }
   template precond_gemslr_csr_seq_float::GemslrClass(precond_gemslr_csr_seq_float &&precond);
   template precond_gemslr_csr_seq_double::GemslrClass(precond_gemslr_csr_seq_double &&precond);
   template precond_gemslr_csr_seq_complexs::GemslrClass(precond_gemslr_csr_seq_complexs &&precond);
   template precond_gemslr_csr_seq_complexd::GemslrClass(precond_gemslr_csr_seq_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrClass<MatrixType, VectorType, DataType>& GemslrClass<MatrixType, VectorType, DataType>::operator=(const GemslrClass<MatrixType, VectorType, DataType> &precond)
   {
      int i;
      this->Clear();
      SolverClass<MatrixType, VectorType, DataType>::operator=(precond);
      
      this->_n = precond._n;
      this->_gemslr_setups = precond._gemslr_setups;
      this->_inner_iters_matrix = precond._inner_iters_matrix;
      this->_inner_iters_precond = precond._inner_iters_precond;
      this->_inner_iters_solver = precond._inner_iters_solver;
      this->_nlev_max = precond._nlev_max;
      this->_nlev_used = precond._nlev_used;
      this->_location = precond._location;
      
      this->_lev_ptr_v = precond._lev_ptr_v;
      this->_dom_ptr_v2.resize(this->_nlev_used);
      this->_levs_v.resize(this->_nlev_used);
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i] = precond._levs_v[i];
         this->_dom_ptr_v2[i] = precond._dom_ptr_v2[i];
      }
      
      this->_pperm = precond._pperm;
      this->_qperm = precond._qperm;
      this->_x_temp = precond._x_temp;
      this->_rhs_temp = precond._rhs_temp;
      
      return *this;
   }
   template precond_gemslr_csr_seq_float& precond_gemslr_csr_seq_float::operator=(const precond_gemslr_csr_seq_float &precond);
   template precond_gemslr_csr_seq_double& precond_gemslr_csr_seq_double::operator=(const precond_gemslr_csr_seq_double &precond);
   template precond_gemslr_csr_seq_complexs& precond_gemslr_csr_seq_complexs::operator=(const precond_gemslr_csr_seq_complexs &precond);
   template precond_gemslr_csr_seq_complexd& precond_gemslr_csr_seq_complexd::operator=(const precond_gemslr_csr_seq_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrClass<MatrixType, VectorType, DataType>& GemslrClass<MatrixType, VectorType, DataType>::operator=(GemslrClass<MatrixType, VectorType, DataType> &&precond)
   {
      int i;
      this->Clear();
      SolverClass<MatrixType, VectorType, DataType>::operator=(std::move(precond));
      
      this->_n = precond._n;precond._n = 0;
      this->_gemslr_setups = precond._gemslr_setups;precond._gemslr_setups.SetDefault();
      this->_inner_iters_matrix = std::move(precond._inner_iters_matrix);
      this->_inner_iters_precond = std::move(precond._inner_iters_precond);
      this->_inner_iters_solver = std::move(precond._inner_iters_solver);
      this->_nlev_max = precond._nlev_max;precond._nlev_max = 0;
      this->_nlev_used = precond._nlev_used;precond._nlev_used = 0;
      this->_location = precond._location;precond._location = 0;
      
      this->_lev_ptr_v = std::move(precond._lev_ptr_v);
      this->_dom_ptr_v2.resize(this->_nlev_used);
      this->_levs_v.resize(this->_nlev_used);
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i] = std::move(precond._levs_v[i]);
         this->_dom_ptr_v2[i] = std::move(precond._dom_ptr_v2[i]);
      }
      std::vector<IntVectorClass<int> >().swap(precond._dom_ptr_v2);      
      std::vector<GemslrLevelClass< MatrixType, VectorType, DataType> >().swap(precond._levs_v);      
      
      this->_pperm = std::move(precond._pperm);
      this->_qperm = std::move(precond._qperm);
      this->_x_temp = std::move(precond._x_temp);
      this->_rhs_temp = std::move(precond._rhs_temp);
      
      return *this;
   }
   template precond_gemslr_csr_seq_float& precond_gemslr_csr_seq_float::operator=(precond_gemslr_csr_seq_float &&precond);
   template precond_gemslr_csr_seq_double& precond_gemslr_csr_seq_double::operator=(precond_gemslr_csr_seq_double &&precond);
   template precond_gemslr_csr_seq_complexs& precond_gemslr_csr_seq_complexs::operator=(precond_gemslr_csr_seq_complexs &&precond);
   template precond_gemslr_csr_seq_complexd& precond_gemslr_csr_seq_complexd::operator=(precond_gemslr_csr_seq_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   GemslrClass<MatrixType, VectorType, DataType>::~GemslrClass()
   {
      this->Clear();
   }
   template precond_gemslr_csr_seq_float::~GemslrClass();
   template precond_gemslr_csr_seq_double::~GemslrClass();
   template precond_gemslr_csr_seq_complexs::~GemslrClass();
   template precond_gemslr_csr_seq_complexd::~GemslrClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::Clear()
   {
      SolverClass<MatrixType, VectorType, DataType>::Clear();
      
      int i;
      
      for(i = 0 ; i < this->_nlev_used ; i ++)
      {
         this->_levs_v[i].Clear();
         this->_dom_ptr_v2[i].Clear();
      }
      
      this->_lev_ptr_v.Clear();
      std::vector<IntVectorClass<int> >().swap(_dom_ptr_v2);      
      std::vector<GemslrLevelClass< MatrixType, VectorType, DataType> >().swap(_levs_v);      
      
      this->_x_temp.Clear();
      this->_rhs_temp.Clear();
      this->_pperm.Clear();
      this->_qperm.Clear();
      
      this->_inner_iters_matrix.Clear();
      this->_inner_iters_precond.Clear();
      this->_inner_iters_solver.Clear();
      
      this->_n                                  = 0;
      this->_nlev_max                           = 0;
      this->_nlev_used                          = 0;
      this->_location                           = kMemoryHost;
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::Clear();
   template int precond_gemslr_csr_seq_double::Clear();
   template int precond_gemslr_csr_seq_complexs::Clear();
   template int precond_gemslr_csr_seq_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::Setup( VectorType &x, VectorType &rhs)
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
      
      int location = this->_matrix->GetDataLocation();
      if(location == kMemoryDevice)
      {
         PARGEMSLR_WARNING("The setup phase of GeMSLR is host only, moving matrix to the host.");
         this->_matrix->MoveData(kMemoryHost);
      }
      
      this->_gemslr_setups._solve_phase_setup = kGemslrPhaseSetup;
      
      /* update the solver precision, leave this interface for half precision */
      this->_solver_precision = x.GetPrecision();
      
      /* create temp vector */
      int n = this->_matrix->GetNumRowsLocal();
      
      this->_n = n;
      this->_x_temp.Setup(n, this->_location, true);
      this->_rhs_temp.Setup(n, this->_location, true);
      
      /* first setup the permutation, this is host only 
       * after this function, permutation array could be on the device when necessary
       */
      this->SetupPermutation();
      
      /* now setup the B solver
       * note that for the device code, B, E, F,
       * and B solver should be on the device
       */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_ILUT, this->SetupBSolve( x, rhs));
      
      /* finally setup the low-rank term
       * we can have this on the device
       */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_LRC, this->SetupLowRank( x, rhs));
      
      /* move the matrix back */
      this->_matrix->MoveData(location);
      
      /* setup inner iteration */
      if(this->_gemslr_setups._enable_inner_iters_setup)
      {
         /* inner iteration is enabled */
         
         /* apply inner iteration on the top level */
         this->_inner_iters_matrix.Setup(0, *this);
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
      
      if(this->_print_option > 0)
      {
         int i, j;
      
         PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
         PARGEMSLR_PRINT("Setup GeMSLR\n");
         PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
         PARGEMSLR_PRINT("|Level|  Ncomp|  Size      |  Nnz         |  rk   |  nmvs |  nnzLU       |  nnzLR       |\n");
         
         for(i = 0 ; i < this->_nlev_used ; i ++)
         {
            GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[i];
            
            int ncomp = level_str._ncomps;
            int n_level = this->_n - this->_lev_ptr_v[i];
            long int nnz_level = level_str._E_mat.GetNumNonzeros() + level_str._F_mat.GetNumNonzeros();
            long int nnz_bsolver, nnz_lr;
            
            level_str.GetNumNonzeros(nnz_bsolver, nnz_lr);
            
            for(j = 0 ; j < ncomp ; j ++)
            {
               nnz_level += level_str._B_mat_v[j].GetNumNonzeros();
            }
            
            /* Level Size Nnz rk nnzLU nnzLR */
            PARGEMSLR_PRINT("|%5d|  %5d|  %10d|  %12ld|  %5d|  %5d|  %10e|  %10e|\n", i, ncomp, n_level, nnz_level, level_str._lrc, level_str._nmvs, (float)nnz_bsolver, (float)nnz_lr);
         }
      }
      
      this->_ready = true;
      this->_gemslr_setups._solve_phase_setup = kGemslrPhaseSolve;
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::Setup( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_gemslr_csr_seq_double::Setup( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_gemslr_csr_seq_complexs::Setup( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_gemslr_csr_seq_complexd::Setup( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupPermutation()
   {
      /* Wrapper of the permutation part of GeMSLR */
      vector_int  map_v, mapptr_v;
      
      /* First step is to get the map vector
       * map_v: map_v[i] = j, the i-th node is in the j-th domain.
       * mapptr_v: the [start,end) domain number on each level.
       */
      switch(this->_gemslr_setups._partition_option_setup)
      {
         case kGemslrPartitionRKway:
         {
            /* Recursive KWay, use all levels */
            PARGEMSLR_LOCAL_FIRM_TIME_CALL( PARGEMSLR_BUILDTIME_PARTITION, this->SetupPermutationRKway( this->_nlev_max, this->_nlev_used, map_v, mapptr_v));
            break;
         }
         case kGemslrPartitionND:
         {
            /* Recursive KWay, use all levels */
            PARGEMSLR_LOCAL_FIRM_TIME_CALL( PARGEMSLR_BUILDTIME_PARTITION, this->SetupPermutationND( this->_nlev_max, this->_nlev_used, map_v, mapptr_v));
            this->_nlev_used = this->_nlev_max;
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown GeMSLR partition option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      /* Now building the level structure by extracting E, B, F, and C matrices */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_STRUCTURE, this->SetupPermutationBuildLevelStructure( map_v, mapptr_v));
      
      /* De allocation */
      map_v.Clear();
      mapptr_v.Clear();
      
      if(this->_print_option > 1)
      {
         if(this->_qperm.GetLengthLocal() > 0)
         {
            this->_matrix->PlotPatternGnuPlot("GeMSLR_partition.data", this->_pperm.GetData(), this->_qperm.GetData(), 0, 0);
         }
         else if(this->_pperm.GetLengthLocal() > 0)
         {
            this->_matrix->PlotPatternGnuPlot("GeMSLR_partition.data", this->_pperm.GetData(), this->_pperm.GetData(), 0, 0);
         }
         else
         {
            this->_matrix->PlotPatternGnuPlot("GeMSLR_partition.data", NULL, NULL, 0, 0);
         }
      }
      
      this->_pperm.MoveData(this->_location);
      this->_qperm.MoveData(this->_location);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::SetupPermutation();
   template int precond_gemslr_csr_seq_double::SetupPermutation();
   template int precond_gemslr_csr_seq_complexs::SetupPermutation();
   template int precond_gemslr_csr_seq_complexd::SetupPermutation();
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupPermutationRKway( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v)
   {
      /* Recursive KWay partition */
      int         n, num_dom, minsep, kmin, kfactor, err, clvl, tlvl;
      
      /* 1. build the matrix A+A' */
      MatrixType  &A = *(this->_matrix);
      MatrixType  AT, AAT;
      
      CsrMatrixTransposeHost( A, AT);
      CsrMatrixAddHost( A, AT, AAT);
      
      AT.Clear();
      
      /* 2. prepare the level structure */
      
      n        = AAT.GetNumRowsLocal();
      
      //num_dom  = 2;
      num_dom  = PargemslrMin(this->_gemslr_setups._ncomp_setup, n);
      //minsep   = 2;
      minsep   = PargemslrMin(pargemslr_global::_minsep, n);
      kmin     = this->_gemslr_setups._kmin_setup;
      kfactor  = this->_gemslr_setups._kfactor_setup;
      tlvl     = this->_gemslr_setups._nlev_setup;
      clvl     = 0;
      
      map_v.Setup(n);
      mapptr_v.Setup(tlvl+1);
      
      /* 3. apply RKway, obtain map vector */
      mapptr_v[0] = 0;
      err = SetupPermutationRKwayRecursive( AAT, this->_gemslr_setups._vertexsep_setup, clvl, tlvl, num_dom, minsep, kmin, kfactor, map_v, mapptr_v); PARGEMSLR_CHKERR(err);
      mapptr_v.Resize( tlvl+1, true, false);
      AAT.Clear();
      
      /* 4. return level number */
      nlev_max = tlvl;
      nlev_used = tlvl;
      
      return err;
      
   }
   template int precond_gemslr_csr_seq_float::SetupPermutationRKway( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_seq_double::SetupPermutationRKway( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_seq_complexs::SetupPermutationRKway( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_seq_complexd::SetupPermutationRKway( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupPermutationND( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v)
   {
      /* ND partition */
      int         i, j, k, n, size1, size2, domi, minsep, err, clvl, tlvl;
      std::vector<std::vector<vector_int> > level_str;
      
      /* 1. build the matrix A+A' */
      MatrixType  &A = *(this->_matrix);
      MatrixType  AT, AAT;
      
      CsrMatrixTransposeHost( A, AT);
      CsrMatrixAddHost( A, AT, AAT);
      
      AT.Clear();
      
      /* 2. prepare the level structure */
      
      n        = AAT.GetNumRowsLocal();
      //minsep   = 2;
      minsep   = PargemslrMin(pargemslr_global::_minsep, n);
      tlvl     = this->_gemslr_setups._nlev_setup;
      clvl     = 0;
      
      /* 3. apply RKway, obtain map vector */
      err = SetupPermutationNDRecursive( AAT, this->_gemslr_setups._vertexsep_setup, clvl, tlvl, minsep, level_str); PARGEMSLR_CHKERR(err);
      
      AAT.Clear();
      
      /* 4. prepare return value */
      map_v.Setup(n);
      mapptr_v.Setup(tlvl+1);
      
      nlev_max = tlvl;
      nlev_used = tlvl;
      
      domi = 0;
      mapptr_v[0] = 0;
      for(i = 0 ; i < nlev_used ; i ++)
      {
         size1 = level_str[i].size();
         for(j = 0 ; j < size1 ; j ++)
         {
            size2 = level_str[i][j].GetLengthLocal();
            for(k = 0 ; k < size2 ; k ++)
            {
               map_v[level_str[i][j][k]] = domi;
            }
            domi++;
         }
         mapptr_v[i+1] = domi;
      }
      
      /* free */
      for(i = 0 ; i < nlev_used ; i ++)
      {
         size1 = level_str[i].size();
         for(j = 0 ; j < size1 ; j ++)
         {
            level_str[i][j].Clear();
         }
         std::vector<vector_int>().swap(level_str[i]);
      }
      std::vector<std::vector<vector_int> >().swap(level_str);
      
      return err;
   
   }
   template int precond_gemslr_csr_seq_float::SetupPermutationND( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_seq_double::SetupPermutationND( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_seq_complexs::SetupPermutationND( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_seq_complexd::SetupPermutationND( int &nlev_max, int &nlev_used, vector_int &map_v, vector_int &mapptr_v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupPermutationBuildLevelStructure( vector_int &map_v, vector_int &mapptr_v)
   {
      /* Recursive KWay partition */
      int         n, i, j, nlev_used, nlev_max, ndom, level, ncomp, maps, mape, n_start, n_end, n_local, n_remain, ni;
      vector_int  domsize_v, domptr_v, temp_perm, local_perm;
      
      MatrixType  &A = *(this->_matrix);
      MatrixType  Apq;
      
      n        = A.GetNumRowsLocal();
      nlev_used = this->_nlev_used;
      nlev_max = this->_nlev_max;
      
      this->_pperm.Setup(n);
      this->_qperm.Clear();
      
      ndom = mapptr_v[nlev_max];
      domptr_v.Setup(ndom+1, true);
      domsize_v.Setup(ndom, true);
      
      /* create the permutation, symmetric so pperm = qperm
       * thus, we set qperm into NULL vector
       */
      for(i = 0 ; i < n ; i ++)
      {
         domsize_v[map_v[i]]++;
      }
      
      domptr_v[0] = 0;
      domptr_v[1] = 0;
      for(i = 2 ; i <= ndom ; i ++)
      {
         domptr_v[i] = domptr_v[i-1] + domsize_v[i-2];
      }
      
      for(i = 0 ; i < n ; i ++)
      {
         this->_pperm[domptr_v[map_v[i]+1]++] = i;
      }
      
      /* create level structure, the E, B, F, and C */
      this->_levs_v.resize(nlev_used);
      
      this->_lev_ptr_v.Setup(nlev_used+1);
      this->_dom_ptr_v2.resize(nlev_used);
      
      this->_lev_ptr_v[nlev_used] = n;
      
      /* Start building the level structure
       * | B F |
       * | E C |
       * on each level, note that the last level only has B
       * 
       * B: block diagonal, stored in vector of matrices
       * E, F: matrices
       */
      for(level = 0 ; level < nlev_used ; level++)
      {
         
         GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
         
         if(level == nlev_used - 1)
         {
            /* last level, only B */
            mape = mapptr_v[nlev_max];
            maps = mapptr_v[level];
            ncomp = mape - maps;
            level_str._ncomps = ncomp;
            
            this->_lev_ptr_v[level] = domptr_v[maps];
            this->_dom_ptr_v2[level].Setup(ncomp+1);
            
            for(i = 0, j = maps ; i <= ncomp ; i ++, j++)
            {
               this->_dom_ptr_v2[level][i] = domptr_v[j];
            }
            
            /* apply local permutation when necessary */
            switch(this->_gemslr_setups._perm_option_setup)
            {
               case kIluReorderingNo:
               {
                  /* do nothing */
                  break;
               }
               case kIluReorderingRcm:
               {
                  /* RCM */
                  for(i = 0 ; i < ncomp ; i ++)
                  {
                     ni = this->_dom_ptr_v2[level][i+1] - this->_dom_ptr_v2[level][i];
                     temp_perm.SetupPtr(this->_pperm, ni, this->_dom_ptr_v2[level][i]);
                     
                     CsrSubMatrixRcmHost(A, temp_perm, local_perm);
                     
                     temp_perm.Perm(local_perm);
                  }
                  break;
               }
               case kIluReorderingAmd:
               {
                  /* AMD */
                  for(i = 0 ; i < ncomp ; i ++)
                  {
                     ni = this->_dom_ptr_v2[level][i+1] - this->_dom_ptr_v2[level][i];
                     temp_perm.SetupPtr(this->_pperm, ni, this->_dom_ptr_v2[level][i]);
                     
                     CsrSubMatrixAmdHost(A, temp_perm, local_perm);
                     
                     temp_perm.Perm(local_perm);
                  }
                  break;
               }
               case kIluReorderingNd:
               {
                  /* AMD */
                  for(i = 0 ; i < ncomp ; i ++)
                  {
                     ni = this->_dom_ptr_v2[level][i+1] - this->_dom_ptr_v2[level][i];
                     temp_perm.SetupPtr(this->_pperm, ni, this->_dom_ptr_v2[level][i]);
                     CsrSubMatrixNdHost(A, temp_perm, local_perm);
                     
                     temp_perm.Perm(local_perm);
                  }
                  break;
               }
               default:
               {
                  PARGEMSLR_ERROR("Unknowning Gemslr Local Reordering Option.");
                  return PARGEMSLR_ERROR_INVALED_OPTION;
                  break;
               }
            }
         }
         else
         {
            /* other levels */
            
            /* number of subdomains on this level */
            mape = mapptr_v[level+1];
            maps = mapptr_v[level];
            ncomp = mape - maps;
            level_str._ncomps = ncomp;
            
            this->_lev_ptr_v[level] = domptr_v[maps];
            this->_dom_ptr_v2[level].Setup(ncomp+1);
            
            for(i = 0, j = maps ; i <= ncomp ; i ++, j++)
            {
               this->_dom_ptr_v2[level][i] = domptr_v[j];
            }
            
            /* apply local permutation when necessary */
            switch(this->_gemslr_setups._perm_option_setup)
            {
               case kIluReorderingNo:
               {
                  /* do nothing */
                  break;
               }
               case kIluReorderingRcm:
               {
                  /* RCM */
                  for(i = 0 ; i < ncomp ; i ++)
                  {
                     ni = this->_dom_ptr_v2[level][i+1] - this->_dom_ptr_v2[level][i];
                     temp_perm.SetupPtr(this->_pperm, ni, this->_dom_ptr_v2[level][i]);
                     
                     CsrSubMatrixRcmHost(A, temp_perm, local_perm);
                     
                     temp_perm.Perm(local_perm);
                  }
                  break;
               }
               case kIluReorderingAmd:
               {
                  /* AMD */
                  for(i = 0 ; i < ncomp ; i ++)
                  {
                     ni = this->_dom_ptr_v2[level][i+1] - this->_dom_ptr_v2[level][i];
                     temp_perm.SetupPtr(this->_pperm, ni, this->_dom_ptr_v2[level][i]);
                     
                     CsrSubMatrixAmdHost(A, temp_perm, local_perm);
                     
                     temp_perm.Perm(local_perm);
                  }
                  break;
               }
               case kIluReorderingNd:
               {
                  /* AMD */
                  for(i = 0 ; i < ncomp ; i ++)
                  {
                     ni = this->_dom_ptr_v2[level][i+1] - this->_dom_ptr_v2[level][i];
                     temp_perm.SetupPtr(this->_pperm, ni, this->_dom_ptr_v2[level][i]);
                     
                     CsrSubMatrixNdHost(A, temp_perm, local_perm);
                     
                     temp_perm.Perm(local_perm);
                  }
                  break;
               }
               default:
               {
                  PARGEMSLR_ERROR("Unknowning Gemslr Local Reordering Option.");
                  return PARGEMSLR_ERROR_INVALED_OPTION;
                  break;
               }
            }
         }
      }
      
      A.SubMatrix(this->_pperm, this->_pperm, kMemoryHost, Apq);
      
      /* Create buffer and Extract submatrices */
      for(level = 0 ; level < nlev_used ; level++)
      {
         
         GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
         
         if(level == nlev_used - 1)
         {
            /* last level, only B */
            mape = mapptr_v[nlev_max];
            maps = mapptr_v[level];
            ncomp = mape - maps;
            level_str._ncomps = ncomp;
            
            n_start = domptr_v[maps];
            n_end = domptr_v[mape];
            n_local = n_end-n_start;
            n_remain = n - n_end;
            
            level_str._z_temp.Setup(n-n_start, this->_location, true);
            level_str._y_temp.Setup(n-n_start, this->_location, true);
            level_str._v_temp.Setup(n-n_start, this->_location, true);
            level_str._w_temp.Setup(n-n_start, this->_location, true);
            
            level_str._B_mat_v.resize(ncomp);
            for(i = 0 ; i < ncomp ; i ++)
            {
               ni = this->_dom_ptr_v2[level][i+1] - this->_dom_ptr_v2[level][i];
               
               Apq.SubMatrix( this->_dom_ptr_v2[level][i], this->_dom_ptr_v2[level][i], 
                              ni, ni, kMemoryHost, level_str._B_mat_v[i]);
               level_str._B_mat_v[i].SetComplexShift(this->_matrix->_diagonal_shift);
            }
            
         }
         else
         {
            /* other levels */
            
            /* number of subdomains on this level */
            mape = mapptr_v[level+1];
            maps = mapptr_v[level];
            ncomp = mape - maps;
            level_str._ncomps = ncomp;
            
            n_start = domptr_v[maps];
            n_end = domptr_v[mape];
            n_local = n_end-n_start;
            n_remain = n - n_end;
            
            level_str._y_temp.Setup(n-n_start, this->_location, true);
            level_str._z_temp.Setup(n-n_start, this->_location, true);
            level_str._v_temp.Setup(n-n_start, this->_location, true);
            level_str._w_temp.Setup(n-n_start, this->_location, true);
            
            level_str._B_mat_v.resize(ncomp);
            for(i = 0 ; i < ncomp ; i ++)
            {
               ni = this->_dom_ptr_v2[level][i+1] - this->_dom_ptr_v2[level][i];
               Apq.SubMatrix( this->_dom_ptr_v2[level][i], this->_dom_ptr_v2[level][i], 
                              ni, ni, kMemoryHost, level_str._B_mat_v[i]);
               level_str._B_mat_v[i].SetComplexShift(this->_matrix->_diagonal_shift);
            }
            Apq.SubMatrix( n_end, n_start, n_remain, n_local, kMemoryHost, level_str._E_mat);
            Apq.SubMatrix( n_start, n_end, n_local, n_remain, kMemoryHost, level_str._F_mat);
            
            if(level == 0)
            {
               Apq.SubMatrix( n_end, n_end, n_remain, n_remain, kMemoryHost, level_str._C_mat);
               SequentialVectorClass<DataType> one_vec, sum_vec;
               
               DataType one = 1.0; 
               
               one_vec.Setup(n_remain);
               sum_vec.Setup(n_remain);
               one_vec.Fill(one);
            }
            
         }
      }
      
      /* 5. Deallocate */
      Apq.Clear();
      temp_perm.Clear();
      local_perm.Clear();
      domptr_v.Clear();
      domsize_v.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::SetupPermutationBuildLevelStructure( vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_seq_double::SetupPermutationBuildLevelStructure( vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_seq_complexs::SetupPermutationBuildLevelStructure( vector_int &map_v, vector_int &mapptr_v);
   template int precond_gemslr_csr_seq_complexd::SetupPermutationBuildLevelStructure( vector_int &map_v, vector_int &mapptr_v);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupBSolve( VectorType &x, VectorType &rhs)
   {
      /* Wrapper of the B solver/preconditioner setup 
       * Note that we have 3 different parts
       * [0, a)      => precond 1
       * [a, nlev-1) => precond 2
       * nlev-1      => C solve
       */
      
      int         level, option;
      
      /* C solve */
      {
         /* this is the last level */
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
               PARGEMSLR_ERROR("Unknown GeMSLR C solve option.");
               return PARGEMSLR_ERROR_INVALED_OPTION;
            }
         }
         
      }
      
      /* B solve */
      
      for(level = 0 ; level < this->_nlev_used - 1; level++)
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
            case kGemslrBSolvePoly:
            {
               this->SetupBSolvePoly( x, rhs, level);
               break;
            }
            case kGemslrBSolveGemslr:
            {
               this->SetupBSolveGemslr( x, rhs, level);
               break;
            }
            default:
            {
               PARGEMSLR_ERROR("Unknown GeMSLR B solve option.");
               return PARGEMSLR_ERROR_INVALED_OPTION;
            }
         }
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::SetupBSolve( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_gemslr_csr_seq_double::SetupBSolve( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_gemslr_csr_seq_complexs::SetupBSolve( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_gemslr_csr_seq_complexd::SetupBSolve( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupBSolveILUT( VectorType &x, VectorType &rhs, int level)
   {
      /* B solver/preconditioner setup with ILU */
      int         i, ncomp;
      /* not going to need those in the setup */
      VectorType  dummyx, dummyrhs;
      
      /* setup levels */
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      ncomp = level_str._ncomps;
      
      /* start building the ILU on this level */
      PARGEMSLR_MALLOC( level_str._B_solver, ncomp, kMemoryHost, SolverClass<MatrixType, VectorType, DataType>*);

#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for(i = 0 ; i < ncomp ; i ++)
      {
         /* build ILU solver */
         PARGEMSLR_PLACEMENT_NEW( level_str._B_solver[i], kMemoryHost, IluClass<MatrixType, VectorType, DataType>);
         IluClass<MatrixType, VectorType, DataType> *ilup = (IluClass<MatrixType, VectorType, DataType>*) level_str._B_solver[i];
         
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
         
         /* set solve option */
         ilup->SetOption(kIluOptionILUT);
         
         /* diagonal complex shift */
         ilup->SetIluComplexShift(this->_gemslr_setups._level_setups._ilu_complex_shift);
         
         /* turn off level-scheduling solve since we have one level of Openmp already */
         ilup->SetOpenMPOption(kIluOpenMPNo);
         
         /* turn off permutation, since we already have it */
         ilup->SetPermutationOption(kIluReorderingNo);
         
         /* setup the solver */
         ilup->Setup(dummyx, dummyrhs);
         
         /* TODO: currently the setup phase is host only, we first setup and move the preconditioner later */
         ilup->SetSolveLocation(this->_location);
         level_str._B_mat_v[i].MoveData(this->_location);
      }
      
      /* move E and F to device when necessary */
      level_str._E_mat.MoveData(this->_location);
      level_str._F_mat.MoveData(this->_location);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::SetupBSolveILUT( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_seq_double::SetupBSolveILUT( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_seq_complexs::SetupBSolveILUT( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_seq_complexd::SetupBSolveILUT( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupBSolveILUK( VectorType &x, VectorType &rhs, int level)
   {
      /* B solver/preconditioner setup with ILU */
      int         i, ncomp;
      /* not going to need those in the setup */
      VectorType  dummyx, dummyrhs;
      
      /* setup levels */
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      ncomp = level_str._ncomps;
      
      /* start building the ILU on this level */
      PARGEMSLR_MALLOC( level_str._B_solver, ncomp, kMemoryHost, SolverClass<MatrixType, VectorType, DataType>*);

#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for(i = 0 ; i < ncomp ; i ++)
      {
         /* build ILU solver */
         PARGEMSLR_PLACEMENT_NEW( level_str._B_solver[i], kMemoryHost, IluClass<MatrixType, VectorType, DataType>);
         IluClass<MatrixType, VectorType, DataType> *ilup = (IluClass<MatrixType, VectorType, DataType>*) level_str._B_solver[i];
         
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
         
         /* set solve option */
         ilup->SetOption(kIluOptionILUK);
         
         /* diagonal complex shift */
         ilup->SetIluComplexShift(this->_gemslr_setups._level_setups._ilu_complex_shift);
         
         /* turn off level-scheduling solve since we have one level of Openmp already */
         ilup->SetOpenMPOption(kIluOpenMPNo);
         
         /* turn off permutation, since we already have it */
         ilup->SetPermutationOption(kIluReorderingNo);
         
         /* setup the solver */
         ilup->Setup(dummyx, dummyrhs);
         
         /* TODO: currently the setup phase is host only, we first setup and move the preconditioner later */
         ilup->SetSolveLocation(this->_location);
         level_str._B_mat_v[i].MoveData(this->_location);
      }
      
      
      /* move E and F to device when necessary */
      level_str._E_mat.MoveData(this->_location);
      level_str._F_mat.MoveData(this->_location);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::SetupBSolveILUK( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_seq_double::SetupBSolveILUK( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_seq_complexs::SetupBSolveILUK( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_seq_complexd::SetupBSolveILUK( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupBSolveGemslr( VectorType &x, VectorType &rhs, int level)
   {
      /* B solver/preconditioner setup with ILU */
      int         i, ncomp;
      
      /* not going to need those in the setup */
      VectorType  dummyx, dummyrhs;
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
         
      ncomp = level_str._ncomps;
      
      /* start building the ILU on this level */
      PARGEMSLR_MALLOC( level_str._B_solver, ncomp, kMemoryHost, SolverClass<MatrixType, VectorType, DataType>*);
   
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for(i = 0 ; i < ncomp ; i ++)
      {
         /* build ILU solver */
         PARGEMSLR_PLACEMENT_NEW( level_str._B_solver[i], kMemoryHost, GemslrClass<MatrixType, VectorType, DataType>);
         GemslrClass<MatrixType, VectorType, DataType> *gemslrp = (GemslrClass<MatrixType, VectorType, DataType>*) level_str._B_solver[i];
         
         /* assign matrix */
         gemslrp->SetMatrix(level_str._B_mat_v[i]);
         
         /* options */
         this->SetLocalGemslr(*gemslrp);
         
         /* setup the solver */
         gemslrp->Setup(dummyx, dummyrhs);
         
         /* TODO: currently the setup phase is host only, we first setup and move the preconditioner later */
         gemslrp->SetSolveLocation(this->_location);
         level_str._B_mat_v[i].MoveData(this->_location);
      }
      
      /* move to device for GPU */
      level_str._E_mat.MoveData(this->_location);
      level_str._F_mat.MoveData(this->_location);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::SetupBSolveGemslr( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_seq_double::SetupBSolveGemslr( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_seq_complexs::SetupBSolveGemslr( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_seq_complexd::SetupBSolveGemslr( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupBSolvePoly( VectorType &x, VectorType &rhs, int level)
   {
      /* B solver/preconditioner setup with ILU */
      int         i, ncomp;
      /* not going to need those in the setup */
      VectorType  dummyx, dummyrhs;
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      ncomp = level_str._ncomps;
      
      /* start building the ILU on this level */
      PARGEMSLR_MALLOC( level_str._B_solver, ncomp, kMemoryHost, SolverClass<MatrixType, VectorType, DataType>*);

#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for(i = 0 ; i < ncomp ; i ++)
      {
         /* build ILU solver */
         PARGEMSLR_PLACEMENT_NEW( level_str._B_solver[i], kMemoryHost, PolyClass<MatrixType, VectorType, DataType>);
         PolyClass<MatrixType, VectorType, DataType> *polyp = (PolyClass<MatrixType, VectorType, DataType>*) level_str._B_solver[i];
         
         /* assign matrix */
         polyp->SetMatrix(level_str._B_mat_v[i]);
         
         /* options */
         polyp->SetOrder(this->_gemslr_setups._level_setups._B_poly_order);
         
         /* setup the solver */
         polyp->Setup(dummyx, dummyrhs);
         
         /* TODO: currently the setup phase is host only, we first setup and move the preconditioner later */
         polyp->SetSolveLocation(this->_location);
         level_str._B_mat_v[i].MoveData(this->_location);
      }
      
      level_str._E_mat.MoveData(this->_location);
      level_str._F_mat.MoveData(this->_location);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::SetupBSolvePoly( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_seq_double::SetupBSolvePoly( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_seq_complexs::SetupBSolvePoly( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_seq_complexd::SetupBSolvePoly( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupLowRank( VectorType &x, VectorType &rhs)
   {
      /* Wrapper of the building of the low-rank correction part of GeMSLR */
      
      int         level, rank, lr_option;
      int         i, maxits;
      
      /* not going to need those in the setup */
      VectorType  dummyx, dummyrhs;
      
      maxits = 1;
      
      /* setup levels */
      for(level = this->_nlev_used - 2 ; level >= 0 ; level--)
      {
         GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
         
         /* we can move the E, F on this level to device */
         level_str._EBFC.Setup(level, *this);
         
         if(level == 0)
         {
            lr_option = this->_gemslr_setups._level_setups._lr_option1_setup;
            rank = this->_gemslr_setups._level_setups._lr_rank1_setup;
         }
         else
         {
            lr_option = this->_gemslr_setups._level_setups._lr_option2_setup;
            rank = this->_gemslr_setups._level_setups._lr_rank2_setup;
         }
         
         /* prepare the test */
         
         for(i = 0 ; i < maxits ; i ++)
         {
            if(rank > 0)
            {
               switch(lr_option)
               {
                  case kGemslrLowrankNoRestart:
                  {
                     level_str._lrc = this->SetupLowRankNoRestart(dummyx, dummyrhs, level_str._nmvs, level);
                     break;
                  }
                  case kGemslrLowrankThickRestart:
                  {
                     level_str._lrc = this->SetupLowRankThickRestart(dummyx, dummyrhs, level_str._nmvs, level);
                     break;
                  }
                  case kGemslrLowrankSubspaceIteration:
                  {
                     level_str._lrc = this->SetupLowRankSubspaceIteration(dummyx, dummyrhs, level_str._nmvs, level);
                     break;
                  }
                  default:
                  {
                     PARGEMSLR_ERROR("Unknown low-rank building option.");
                     return PARGEMSLR_ERROR_INVALED_OPTION;
                  }
               }
            }
            else
            {
               level_str._lrc = 0;
            }
         
            if(level_str._lrc > 0)
            {
               level_str._xlr_temp.Setup(this->_lev_ptr_v[this->_nlev_used]-this->_lev_ptr_v[level+1], this->_location, true);
               level_str._xlr1_temp.Setup(level_str._lrc, this->_location, true);
               level_str._xlr2_temp.Setup(level_str._lrc, this->_location, true);
            }
            
         }
         
      }
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::SetupLowRank( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_gemslr_csr_seq_double::SetupLowRank( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_gemslr_csr_seq_complexs::SetupLowRank( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_gemslr_csr_seq_complexd::SetupLowRank( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupLowRankSubspaceIteration( VectorType &x, VectorType &rhs, int &nmvs, int level)
   {
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      typedef DataType T;
      
      /* Set up parameters */
      int                                    n, neig_c, neig_k, maxits, err;
      RealDataType                           lr_fact;
      DenseMatrixClass<T>                    V, H;
   
      /*------------------------------------------------------------------------
       * 1: Init Phase
       * Declare all variables and setup parameters
       * 
       * B  F
       * E  C
       * 
       * the size of the low rank correction on this level is of size of C
       *------------------------------------------------------------------------*/
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      n = level_str._E_mat.GetNumRowsLocal();
      
      /* we don't want to do more steps than the size of the matrix */
      if(level == 0)
      {
         neig_k      = this->_gemslr_setups._level_setups._lr_rank1_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor1_setup;
         maxits      = this->_gemslr_setups._level_setups._lr_maxits1_setup;
      }
      else
      {
         neig_k      = this->_gemslr_setups._level_setups._lr_rank2_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor2_setup;
         maxits      = this->_gemslr_setups._level_setups._lr_maxits1_setup;
      }
      
      neig_c      = int(neig_k * lr_fact);
      neig_c      = PargemslrMin(neig_c, n);
      neig_k      = PargemslrMin(neig_c, neig_k);
      
      if(neig_k == 0)
      {
         /* don't compute */
         return 0;
      }
      
      /*------------------------ 
       * 2: Arnoldi and get result
       *------------------------*/
      
      ArnoldiMatrixClass<VectorType, DataType> &temp_EBFC = level_str._EBFC;
      PARGEMSLR_LOCAL_TIME_CALL(PARGEMSLR_BUILDTIME_ARNOLDI, PargemslrSubSpaceIteration<VectorType>( temp_EBFC, neig_c, maxits, V, H, RealDataType(), nmvs));
      
      /* free of V and H are handled inside */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_BUILD_RES, err = this->SetupLowRankBuildLowRank(x, rhs, V, H, neig_c, neig_k, level));
      
      return err;
   }
   template int precond_gemslr_csr_seq_float::SetupLowRankSubspaceIteration( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_double::SetupLowRankSubspaceIteration( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_complexs::SetupLowRankSubspaceIteration( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_complexd::SetupLowRankSubspaceIteration( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int &nmvs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupLowRankNoRestart( VectorType &x, VectorType &rhs, int &nmvs, int level)
   {
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      typedef DataType T;
      
      /* Set up parameters */
      int                                    n, neig_c, neig_k, m, err;
      bool                                   rand_init;
      RealDataType                           normv, ar_fact, lr_fact, tol_orth, tol_reorth;
      DenseMatrixClass<T>                    V, H;
      SequentialVectorClass<T>               v;
   
      /*------------------------------------------------------------------------
       * 1: Init Phase
       * Declare all variables and setup parameters
       * 
       * B  F
       * E  C
       * 
       * the size of the low rank correction on this level is of size of C
       *------------------------------------------------------------------------*/
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      n = level_str._E_mat.GetNumRowsLocal();
      
      /* we don't want to do more steps than the size of the matrix */
      if(level == 0)
      {
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factor1_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor1_setup;
         neig_k      = this->_gemslr_setups._level_setups._lr_rank1_setup;
      }
      else
      {
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor2_setup;
         neig_k      = this->_gemslr_setups._level_setups._lr_rank2_setup;
      }
      neig_c      = int(neig_k * ar_fact * lr_fact);
      
      neig_c      = PargemslrMin(neig_c, n);
      neig_k      = PargemslrMin(neig_c, neig_k);
      
      if(neig_k == 0)
      {
         /* don't compute */
         return 0;
      }
      
      rand_init   = this->_gemslr_setups._level_setups._lr_rand_init_setup;
      tol_orth    = pargemslr_global::_orth_tol;
      tol_reorth  = pargemslr_global::_reorth_tol;
      
      /* create matrix V and H used in Arnoldi 
       * Note that V can be on the device.
       * H can be on the host.
       */
      V.Setup( n, neig_c+1, this->_location, true);
      H.Setup( neig_c+1, neig_c, kMemoryHost, true);
   
      /* setup init guess */
      v.SetupPtr(V.GetData(), n, V.GetDataLocation() );
      
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
      
      ArnoldiMatrixClass<VectorType, DataType> &temp_EBFC = level_str._EBFC;
      PARGEMSLR_LOCAL_TIME_CALL(PARGEMSLR_BUILDTIME_ARNOLDI, m = PargemslrArnoldiNoRestart<VectorType>( temp_EBFC, 0, neig_c, V, H, tol_orth, tol_reorth, nmvs));
      
      /* free of V and H are handled inside */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_BUILD_RES, err = this->SetupLowRankBuildLowRank(x, rhs, V, H, m, neig_k, level));
      
      return err;
   }
   template int precond_gemslr_csr_seq_float::SetupLowRankNoRestart( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_double::SetupLowRankNoRestart( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_complexs::SetupLowRankNoRestart( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_complexd::SetupLowRankNoRestart( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int &nmvs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupLowRankThickRestart( VectorType &x, VectorType &rhs, int &nmvs, int level)
   {
      //if(this->_gemslr_setups._level_setups._lr_tol_eig_setup > GemslrClass<MatrixType, VectorType, DataType>::_convergence_tolorance)
      //{
         /* the eigenvalues are not accurate enough, do not lock them */
         return this->SetupLowRankThickRestartNoLock(x, rhs, nmvs, level);
      //}
      //else
      //{
         /* lock convergenced eigenvalues */
      //   return this->SetupLowRankThickRestartStandard(x, rhs, level);
      //}
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_seq_float::SetupLowRankThickRestart( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_double::SetupLowRankThickRestart( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_complexs::SetupLowRankThickRestart( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_complexd::SetupLowRankThickRestart( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int &nmvs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupLowRankThickRestartNoLock( SequentialVectorClass<DataType> &x, SequentialVectorClass<DataType> &rhs, int &nmvs, int level)
   {
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      
      int                                                      n, rank, rank2, lr_m, m, maxsteps, maxits, err;
      bool                                                     rand_init;
      RealDataType                                             normv;
      DataType                                                 one;
      RealDataType                                             tol_eig, ar_fact, lr_fact, tr_fact, tol_orth, tol_reorth;
      
      DenseMatrixClass<DataType>                               V, H;
      SequentialVectorClass<DataType>                          v;
      
      one                        = DataType(1.0);
      
      /*------------------------
       * 1: Init Phase
       * Declare all variables and setup parameters
       *------------------------*/
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      n = level_str._E_mat.GetNumRowsLocal();
      
      if(level == 0)
      {
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factor1_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor1_setup;
         rank        = this->_gemslr_setups._level_setups._lr_rank1_setup;
         tol_eig     = this->_gemslr_setups._level_setups._lr_tol_eig1_setup;
         maxits      = this->_gemslr_setups._level_setups._lr_maxits1_setup;
      }
      else
      {
         ar_fact     = this->_gemslr_setups._level_setups._lr_arnoldi_factor2_setup;
         lr_fact     = this->_gemslr_setups._level_setups._lr_rank_factor2_setup;
         rank        = this->_gemslr_setups._level_setups._lr_rank2_setup;
         tol_eig     = this->_gemslr_setups._level_setups._lr_tol_eig2_setup;
         maxits      = this->_gemslr_setups._level_setups._lr_maxits1_setup;
      }
      
      tr_fact        = pargemslr_global::_tr_factor;
      
      rand_init      = this->_gemslr_setups._level_setups._lr_rand_init_setup;
      tol_orth       = pargemslr_global::_orth_tol;
      tol_reorth     = pargemslr_global::_reorth_tol;
      
      /* compute actual rank to compute */
      rank2 = (int)(rank * lr_fact);
      rank2 = PargemslrMax(rank2, rank);
      
      /* compute initial number of Arnoldi steps */
      lr_m = (int)(rank2 * ar_fact);
      lr_m = PargemslrMax(lr_m, rank2);
      
      /* we don't want to do more steps than the size of the matrix 
       * maxsteps is the maximun size of steps we can have,
       */
      maxsteps    = PargemslrMin( (int)(lr_m + (lr_m * tr_fact)), n);
      
      rank        = PargemslrMin(rank, maxsteps);
      rank2       = PargemslrMin(rank2, maxsteps);
      
      /* create matrix V and H used in Arnoldi 
       * Note that V can be on the device.
       * H can be on the host.
       */
      V.Setup( n, maxsteps+1, this->_location, true);
      H.Setup( maxsteps+1, maxsteps, kMemoryHost, true);
      
      /* setup init guess */
      v.SetupPtr(V.GetData(), n, V.GetDataLocation() );
      
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
      
      ArnoldiMatrixClass<VectorType, DataType> &temp_EBFC = level_str._EBFC;
      m = PargemslrArnoldiThickRestartNoLock<VectorType>( temp_EBFC, lr_m, maxits, rank2, rank, RealDataType(0.0), tr_fact, tol_eig, RealDataType(1.0), RealDataType(0.0), &(GemslrClass<MatrixType, VectorType, DataType>::ComputeDistance), V, H, tol_orth, tol_reorth, nmvs);
      
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_BUILD_RES, err = this->SetupLowRankBuildLowRank(x, rhs, V, H, m, m, level));
      
      return err;
   }
   template int precond_gemslr_csr_seq_float::SetupLowRankThickRestartNoLock( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_double::SetupLowRankThickRestartNoLock( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_complexs::SetupLowRankThickRestartNoLock( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int &nmvs, int level);
   template int precond_gemslr_csr_seq_complexd::SetupLowRankThickRestartNoLock( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int &nmvs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename RealDataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupLowRankThickRestartStandard( SequentialVectorClass<RealDataType> &x, SequentialVectorClass<RealDataType> &rhs, int level)
   {
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_seq_float::SetupLowRankThickRestartStandard( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_seq_double::SetupLowRankThickRestartStandard( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename RealDataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupLowRankThickRestartStandard( SequentialVectorClass<ComplexValueClass<RealDataType> > &x, SequentialVectorClass<ComplexValueClass<RealDataType> > &rhs, int level)
   {
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_seq_complexs::SetupLowRankThickRestartStandard( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_seq_complexd::SetupLowRankThickRestartStandard( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename T1, typename T2>
   T1 GemslrClass<MatrixType, VectorType, DataType>::ComputeDistance(T2 val)
   {
      return PargemslrAbs(val);
      
      /*
      T2 temp_val = T2(1.0) - val;
      
      if(PargemslrAbs(temp_val) < 1e-02)
      {
         return PargemslrAbs( val * T2(1e02) ) + PargemslrAbs(val);
      }
      else
      {
         return PargemslrAbs( val / temp_val ) + PargemslrAbs(val);
      }
      */
   }
   template float precond_gemslr_csr_seq_float::ComputeDistance(complexs val);
   template double precond_gemslr_csr_seq_double::ComputeDistance(complexd val);
   template float precond_gemslr_csr_seq_complexs::ComputeDistance(complexs val);
   template double precond_gemslr_csr_seq_complexd::ComputeDistance(complexd val);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename RealDataType>
   int GemslrClass<MatrixType, VectorType, DataType>::OrdLowRank(int m, int &rank, DenseMatrixClass<RealDataType> &R, DenseMatrixClass<RealDataType> &Q)
   {
      
      typedef ComplexValueClass<DataType> T;
      
      int                                       i;
      T                                         eig_val, cone;
      DataType                                  eig_val_dist;
      vector_int                                order, select;
      SequentialVectorClass<DataType>           wr, wi, w;
      
      cone                                      = T(1.0,0.0);
      
      /* Schur factorizition of R
       * note that R is from the H of Arnoldi, we can call HessSchur
       */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, R.HessSchur(Q, wr, wi) );
      
      /* sort based on lambda/(1-lambda) */
      w.Setup(m);
      for(i = 0 ; i < m ; i ++)
      {
         if( i < m-1 && wi[i] > 0 && wi[i] == -wi[i+1])
         {
            /* in this case we have a pair of eigenvalues */
            eig_val = T( wr[i], wi[i] );
            eig_val_dist = this->ComputeDistance<RealDataType>(eig_val);
            w[i] = eig_val_dist;
            i++;
            w[i] = eig_val_dist;
         }
         else
         {
            /* in this case, only real eigenvalue */
            eig_val = T( wr[i], 0.0 );
            eig_val_dist = this->ComputeDistance<RealDataType>(eig_val);
            w[i] = eig_val_dist;
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
      
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, R.OrdSchur(Q, wr, wi, select) );
      
      w.Clear();
      wr.Clear();
      wi.Clear();
      order.Clear();
      select.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::OrdLowRank(int m, int &rank, matrix_dense_float &R, matrix_dense_float &Q);
   template int precond_gemslr_csr_seq_double::OrdLowRank(int m, int &rank, matrix_dense_double &R, matrix_dense_double &Q);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename RealDataType>
   int GemslrClass<MatrixType, VectorType, DataType>::OrdLowRank(int m, int &rank, DenseMatrixClass<ComplexValueClass<RealDataType> > &R, DenseMatrixClass<ComplexValueClass<RealDataType> > &Q)
   {
      typedef DataType T;
      
      int                                 i;
      T                                   cone;
      IntVectorClass<int>                 order, select;
      SequentialVectorClass<RealDataType> w;
      SequentialVectorClass<T>            wi;
      
      cone                                = T(1.0,0.0);
      
      /* Schur factorizition of R
       * note that R is from the H of Arnoldi, we can call HessSchur
       */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, R.HessSchur(Q, wi) );
      
      /* sort based on lambda/(1-lambda) */
      w.Setup(m);
      for(i = 0 ; i < m ; i ++)
      {
         /* in this case, only real eigenvalue */
         w[i] = this->ComputeDistance<RealDataType>(wi[i]);
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
      
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, R.OrdSchur(Q, wi, select) );
      
      w.Clear();
      wi.Clear();
      order.Clear();
      select.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_complexs::OrdLowRank(int m, int &rank, matrix_dense_complexs &R, matrix_dense_complexs &Q);
   template int precond_gemslr_csr_seq_complexd::OrdLowRank(int m, int &rank, matrix_dense_complexd &R, matrix_dense_complexd &Q);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupLowRankBuildLowRank( VectorType &x, VectorType &rhs, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, int m, int rank, int level)
   {
      
      /* define the data type */
      typedef DataType T;
      
      int                              n, i, j;
      bool                             isutri;
      DenseMatrixClass<T>              W_temp, W_temp1, Q, Q_temp, R_temp;
      
      if( m == 0 )
      {
         /* nothing computed, return 0 */
         V.Clear();
         H.Clear();
         return 0;
      }
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      DenseMatrixClass<T>              &W = level_str._Wk;
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
         
         /* R = I-H */
         for (i = 0; i < m; i++) 
         {
            for (j = 0; j < m; j++)
            {
               if(i == j)
               {
                  R(j,i) = T(1.0) - H(j,i); 
               }
               else
               {
                  R(j,i) = -H(j,i); 
               }
            }
         }
         
         /* now compute  R = (I-H)^{-1}*/
         R.Invert();
         
         /* finally, R = (I-H)^{-1} - I */
         for(i = 0; i < m ; i ++)
         {
            R(i, i) -= T(1.0);
         }
         
         if(this->_print_option > 2)
         {
            vector_seq_double plot_vec;
            plot_vec.Setup(m);
            for(i = 0; i < m ; i ++)
            {
               plot_vec[i] = PargemslrAbs(R(i, i));
            }
            
            plot_vec.Sort(false);
            
            char tempfilename[1024];
            snprintf( tempfilename, 1024, "%s%d", "Low_rank_R_diag_", level );
            
            plot_vec.PlotAbsGnuPlot(tempfilename, 0, 0, false, true, 1);
            plot_vec.Clear();
         }
         
         /* now move R to device when necessary */
         R.MoveData(this->_location);
         
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
      this->OrdLowRank(m, rank, R_temp, Q);
      
      /* move Q to device when necessary */
      Q.MoveData(this->_location);
      Q_temp.SetupPtr(Q, 0, 0, m, rank);
      W_temp1.SetupPtr(W_temp, 0, 0, n, m);
      
      W.MatMat( T(1.0), W_temp1, 'N', Q_temp, 'N', T());
      
      R.Setup(rank, rank, kMemoryHost, false);
      
      /* R = I-R_temp(1:rank,1:rank) */
      isutri = true;
      for (i = 0; i < rank; i++) 
      {
         for (j = 0; j < rank; j++)
         {
            if(i == j)
            {
               R(j,i) = T(1.0) - R_temp(j,i); 
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
      
      /* now compute  R = (I-H)^{-1}*/
      if(isutri)
      {
         R.InvertUpperTriangular();
      }
      else
      {
         R.Invert();
      }
      
      /* finally, R = (I-H)^{-1} - I */
      for(i = 0; i < rank ; i ++)
      {
         R(i, i) -= T(1.0);
      }
      
      if(this->_print_option > 2)
      {
         vector_seq_double plot_vec;
         plot_vec.Setup(rank);
         for(i = 0; i < rank ; i ++)
         {
            plot_vec[i] = PargemslrAbs(R(i, i));
         }
         
         plot_vec.Sort(false);
         
         char tempfilename[1024];
         snprintf( tempfilename, 1024, "%s%d", "Low_rank_R_diag_", level );
         
         plot_vec.PlotAbsGnuPlot(tempfilename, 0, 0, false, true, 1);
         plot_vec.Clear();
      }
      
      /* now move R to device when necessary */
      R.MoveData(this->_location);
      
      V.Clear();
      H.Clear();
      W_temp.Clear();
      R_temp.Clear();
      Q.Clear();
      W_temp1.Clear();
      Q_temp.Clear();
      
      return rank;
   }
   template int precond_gemslr_csr_seq_float::SetupLowRankBuildLowRank( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, int m, int rank, int level);
   template int precond_gemslr_csr_seq_double::SetupLowRankBuildLowRank( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, int m, int rank, int level);
   template int precond_gemslr_csr_seq_complexs::SetupLowRankBuildLowRank( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, int m, int rank, int level);
   template int precond_gemslr_csr_seq_complexd::SetupLowRankBuildLowRank( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, int m, int rank, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetupLowRankBuildLowRank( VectorType &x, VectorType &rhs, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, DenseMatrixClass<DataType> &Q, vector_int &select, int m, int rank, int level)
   {
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      typedef DataType T;
      
      int                                                      n, i, j;
      //bool                                                     isutri;
      T                                                        one, zero;
      DenseMatrixClass<T>                                      W_temp, H_temp, Q_temp;
      DenseMatrixClass<RealDataType>                           H_temp_real, Q_temp_real;
      DenseMatrixClass<ComplexValueClass<RealDataType> >       H_temp_complex, Q_temp_complex;
      SequentialVectorClass<RealDataType>                      wi, wr;
      SequentialVectorClass<ComplexValueClass<RealDataType> >  w;
      
      if( m == 0 )
      {
         /* nothing computed, return 0 */
         V.Clear();
         H.Clear();
         return 0;
      }
      
      one = T(1.0);
      zero = T();
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      DenseMatrixClass<T>              &W = level_str._Wk;
      DenseMatrixClass<T>              &R = level_str._Hk;
      
      n = V.GetNumRowsLocal();
      
      if( m <= rank )
      {
         /* In this case, we keep all of them
          * no need to turn R into diagonal matrix 
          */
         
         W.Setup( n, m, this->_location, true);
         R.Setup( m, m, kMemoryHost, false);
         
         /* copy useful parts to W and R */
         
         /* W first */
         W_temp.SetupPtr(V, 0, 0, n, m);
         W.MatMat( one, W_temp, 'N', Q, 'N', zero);
         
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
         
         /* now compute  R = (I-H)^{-1}*/
         R.Invert();
         
         /* finally, R = (I-H)^{-1} - I */
         for(i = 0; i < m ; i ++)
         {
            R(i, i) -= one;
         }
         
         /* now move R to device when necessary */
         R.MoveData(this->_location);
         
         V.Clear();
         H.Clear();
         Q.Clear();
         select.Clear();
         
         return m;
         
      }
      
      /* In this case, we only keep part of the eigenvalues, use OrdSchur */
      
      if(PargemslrIsComplex<DataType>::value)
      {
         /* complex */
         H_temp_complex.SetupPtr( (ComplexValueClass<RealDataType>*)(H.GetData()), H.GetNumRowsLocal(), H.GetNumColsLocal(), H.GetLeadingDimension(), kMemoryHost);
         Q_temp_complex.SetupPtr( (ComplexValueClass<RealDataType>*)(Q.GetData()), Q.GetNumRowsLocal(), Q.GetNumColsLocal(), Q.GetLeadingDimension(), kMemoryHost);
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, H_temp_complex.OrdSchur(Q_temp_complex, w, select) );
      }
      else
      {
         /* real */
         H_temp_real.SetupPtr( (RealDataType*)(H.GetData()), H.GetNumRowsLocal(), H.GetNumColsLocal(), H.GetLeadingDimension(), kMemoryHost);
         Q_temp_real.SetupPtr( (RealDataType*)(Q.GetData()), Q.GetNumRowsLocal(), Q.GetNumColsLocal(), Q.GetLeadingDimension(), kMemoryHost);
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, H_temp_real.OrdSchur(Q_temp_real, wr, wi, select) );
      }
      
      W.Setup( n, rank, this->_location, true);
      R.Setup( rank, rank, kMemoryHost, false);
      
      /* W first */
      W_temp.SetupPtr(V, 0, 0, n, m);
      Q.MoveData(this->_location);
      Q_temp.SetupPtr(Q, 0, 0, m, rank);
      W.MatMat( one, W_temp, 'N', Q_temp, 'N', zero);
      
      /* R = I-H */
      for (i = 0; i < rank; i++) 
      {
         for (j = 0; j < rank; j++)
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
      
      /* now compute  R = (I-H)^{-1}*/
      R.Invert();
      
      /* finally, R = (I-H)^{-1} - I */
      for(i = 0; i < rank ; i ++)
      {
         R(i, i) -= one;
      }
      
      /* now move R to device when necessary */
      R.MoveData(this->_location);
      
      V.Clear();
      H.Clear();
      Q.Clear();
      H_temp_complex.Clear();
      H_temp_real.Clear();
      Q_temp_complex.Clear();
      Q_temp_real.Clear();
      W_temp.Clear();
      Q_temp.Clear();
      H_temp.Clear();
      w.Clear();
      wi.Clear();
      wr.Clear();
      
      return rank;
   }
   template int precond_gemslr_csr_seq_float::SetupLowRankBuildLowRank( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, DenseMatrixClass<float> &Q, vector_int &select, int m, int rank, int level);
   template int precond_gemslr_csr_seq_double::SetupLowRankBuildLowRank( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, DenseMatrixClass<double> &Q, vector_int &select, int m, int rank, int level);
   template int precond_gemslr_csr_seq_complexs::SetupLowRankBuildLowRank( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, DenseMatrixClass<complexs> &Q, vector_int &select, int m, int rank, int level);
   template int precond_gemslr_csr_seq_complexd::SetupLowRankBuildLowRank( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, DenseMatrixClass<complexd> &Q, vector_int &select, int m, int rank, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::Solve( VectorType &x, VectorType &rhs)
   {
      /* Wrapper */
      
      /* note that we are solving the equation with Apq, we need to apply the permutation 
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
      
      /* 1. get Pb */ 
      this->_pperm.GatherPerm( rhs, this->_rhs_temp);
      
      /* 2. solve PAQ(Q'x) = Pb */
      this->SolveLevel(this->_x_temp, this->_rhs_temp, 0);
      
      /* 3. now we have xout = Q'x, we need to have Qxout = x */
      if(this->_qperm.GetLengthLocal() > 0)
      {
         /* nonsym permutation */
         this->_qperm.ScatterRperm( this->_x_temp, x);
      }
      else
      {
         /* sym permutation */
         this->_pperm.ScatterRperm( this->_x_temp, x);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_seq_float::Solve( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_gemslr_csr_seq_double::Solve( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_gemslr_csr_seq_complexs::Solve( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_gemslr_csr_seq_complexd::Solve( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SolveLevel( VectorType &x, VectorType &rhs, int level)
   {
      
      /* define the data type */
      typedef DataType T;
      
      /* the solve phase of GeMSLR */
      
      PARGEMSLR_CHKERR(level < 0);
      PARGEMSLR_CHKERR(level >= this->_nlev_used);
      
      int n_start, n_local, n_end, n_remain;
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      if(level == this->_nlev_used - 1)
      {
         /* in this case we apply solve on the last level */
         
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_SOLVELU_L, this->SolveB(x, rhs, level));
         }
         else
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_ILUT_L, this->SolveB(x, rhs, level));
         }
         
         return PARGEMSLR_SUCCESS;
         
      }
      
      T one, zero, mone;
      
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      VectorType rhsu, rhsl, xu, xl, zu, zl;
      
      n_start = this->_lev_ptr_v[level];
      n_end = this->_lev_ptr_v[level+1];
      n_local = n_end - n_start;
      n_remain = this->_n - n_end;
      
      rhsu.SetupPtr(rhs, n_local, 0, false);
      rhsl.SetupPtr(rhs, n_remain, n_local, false);
      xu.SetupPtr(x, n_local, 0, false);
      xl.SetupPtr(x, n_remain, n_local, false);
      zu.SetupPtr(level_str._z_temp, n_local, 0, false);
      zl.SetupPtr(level_str._z_temp, n_remain, n_local, false);
      
      /* in the solve phase, one can switch to the U solve */
      if(this->_gemslr_setups._solve_option_setup == kGemslrUSolve && this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSolve)
      {
         /* skip the block L solve */
         /* Step 1: z = rhs */
         PARGEMSLR_MEMCPY(level_str._z_temp.GetData(), rhs.GetData(), rhs.GetLengthLocal(), kMemoryHost, kMemoryHost, DataType);
      }
      else
      {
      
         /* Step 1: z_u = U\L\rhs_u */
         
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_SOLVELU, this->SolveB(zu, rhsu, level));
         }
         else
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_ILUT, this->SolveB(zu, rhsu, level));
         }
         
         /* Step 2: z_l = -E * z_u */
         
         if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_EMV, (level_str._E_mat.MatVec( 'N', mone, zu, zero, zl)));
         }
         else
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_EMV, (level_str._E_mat.MatVec( 'N', mone, zu, zero, zl)));
         }
         
         /* Step 3: z_l = rhs_l + z_l = rhs_l - E * z_u */
         zl.Axpy(one, rhsl);
      }
   
      /* Step 4: apply the low-rank correction 
       * x_l = C\z_l + W*H*W' * z_l
       */
      if( this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSolve 
            && this->_gemslr_setups._enable_inner_iters_setup 
            && level == 0)
      {
         /* solve with 0 as initial guess */
         xl.Fill(zero);
         this->_inner_iters_solver.Solve(xl, zl);
      }
      else
      {
         if( level_str._lrc > 0)
         {
            /* apply the low-rank update */
            if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
            {
               PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_SOLVELR, (this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level)));
            }
            else
            {
               PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_LRC, (this->SolveApplyLowRankLevel( level_str._xlr_temp, zl, level)));
            }
            zl.Axpy( one, level_str._xlr_temp);
         }
         this->SolveLevel( xl, zl, level+1);
      }
      
      /* Step 5: z_u = - F * x_l */
      
      if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_FMV, (level_str._F_mat.MatVec( 'N', mone, xl, zero, zu)));
      }
      else
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_FMV, (level_str._F_mat.MatVec( 'N', mone, xl, zero, zu)));
      }
      
      /* Step 6: z_u = rhs_u - F * x_l */
      zu.Axpy(one, rhsu);
      
   
      /* Step 7: x_u = U\L\z_u */
      
      if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_SOLVELU, this->SolveB(xu, zu, level));
      }
      else
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_ILUT, this->SolveB(xu, zu, level));
      }
      
      rhsu.Clear();
      rhsl.Clear();
      xu.Clear();
      xl.Clear();
      zu.Clear();
      zl.Clear();
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_seq_float::SolveLevel( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_seq_double::SolveLevel( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_seq_complexs::SolveLevel( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_seq_complexd::SolveLevel( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SolveB( VectorType &x, VectorType &rhs, int level)
   {
      
      /* the solve with B on a certain level */
      
      PARGEMSLR_CHKERR(level < 0);
      PARGEMSLR_CHKERR(level >= this->_nlev_used);
      
      int i, ncomp, n_start, n1, n2;
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      ncomp = level_str._ncomps;
      
      n_start = this->_lev_ptr_v[level];
      
      /* apply the solve
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
         
         VectorType xi, rhsi;
         
         xi.SetupPtr(x, n2 - n1, n1 - n_start, false);
         rhsi.SetupPtr(rhs, n2 - n1, n1 - n_start, false);

         level_str._B_solver[i]->Solve(xi, rhsi);
         
         xi.Clear();
         rhsi.Clear();
         
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_seq_float::SolveB( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_seq_double::SolveB( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_seq_complexs::SolveB( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_seq_complexd::SolveB( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SolveApplyLowRankLevel( VectorType &x, VectorType &rhs, int level)
   {
      
      /* define the data type */
      typedef DataType T;
      
      int            n_H;
      T              zero, one;
      
      zero  = 0.0;
      one   = 1.0;
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
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
      
      /* Step2: H*W'*x */
      level_str._Hk.MatVec('N', one, level_str._xlr1_temp, zero, level_str._xlr2_temp);
      
      /* Step3: W*H*W'*x */
      level_str._Wk.MatVec('N', one, level_str._xlr2_temp, zero, x);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_gemslr_csr_seq_float::SolveApplyLowRankLevel( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int level);
   template int precond_gemslr_csr_seq_double::SolveApplyLowRankLevel( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int level);
   template int precond_gemslr_csr_seq_complexs::SolveApplyLowRankLevel( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int level);
   template int precond_gemslr_csr_seq_complexd::SolveApplyLowRankLevel( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::GetNumRows(int level)
   {
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      return level_str._E_mat.GetNumRowsLocal();
   }
   template int precond_gemslr_csr_seq_float::GetNumRows(int level);
   template int precond_gemslr_csr_seq_double::GetNumRows(int level);
   template int precond_gemslr_csr_seq_complexs::GetNumRows(int level);
   template int precond_gemslr_csr_seq_complexd::GetNumRows(int level);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::EBFCMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      PARGEMSLR_CHKERR(level >= this->_nlev_used - 1);
      
      /* define the data type */
      typedef DataType T;
      
      int n_start, n_local, n_end, n_remain;
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      T one, zero, mone;
      
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      n_start = this->_lev_ptr_v[level];
      n_end = this->_lev_ptr_v[level+1];
      n_local = n_end - n_start;
      n_remain = this->_n - n_end;
      
      VectorType zu, zl, worku;
      
      worku.SetupPtr(level_str._y_temp, n_local, 0, false);
      zu.SetupPtr(level_str._z_temp, n_local, 0, false);
      zl.SetupPtr(level_str._z_temp, n_remain, n_local, false);
      
      /* Now compute I - SC^{-1} */
 
      /* Solve with C */
      this->SolveLevel( zl, x, level+1);
      
      /* y = (I-CC^{-1})x */
      this->CMatVec( level, 'N', mone, zl, zero, y);
      y.Axpy(one, x);
      
      /* next we compute EB^{-1}F */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_FMV, (level_str._F_mat.MatVec( 'N', one, zl, zero, zu)));
      
      /* Solve with B */
      
      if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_SOLVELU, this->SolveB(worku, zu, level));
      }
      else
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_ILUT, this->SolveB(worku, zu, level));
      }
      
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_EMV, (level_str._E_mat.MatVec( 'N', one, worku, one, y)));
      
      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_seq_float::EBFCMatVec( int level, char trans, const float &alpha, SequentialVectorClass<float> &x, const float &beta, SequentialVectorClass<float> &y);
   template int precond_gemslr_csr_seq_double::EBFCMatVec( int level, char trans, const double &alpha, SequentialVectorClass<double> &x, const double &beta, SequentialVectorClass<double> &y);
   template int precond_gemslr_csr_seq_complexs::EBFCMatVec( int level, char trans, const complexs &alpha, SequentialVectorClass<complexs> &x, const complexs &beta, SequentialVectorClass<complexs> &y);
   template int precond_gemslr_csr_seq_complexd::EBFCMatVec( int level, char trans, const complexd &alpha, SequentialVectorClass<complexd> &x, const complexd &beta, SequentialVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SchurMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      /* no inner iteration on the last level */
      PARGEMSLR_CHKERR(level >= this->_nlev_used - 1);
      
      /* define the data type */
      typedef DataType T;
      
      int i, n_start, n_local, n_end, n_remain, n1, n2;
      int j, ncompi, n_starti, n_locali, n_endi, n_remaini;
      
      /* y = alpha*S*x + beta*y 
       * S = C - EB^{-1}F, we first compute -EB^{-1}F*x.
       */
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      T one, zero, malpha;
      
      one = 1.0;
      zero = 0.0;
      malpha = -alpha;
      
      n_start = this->_lev_ptr_v[level];
      n_end = this->_lev_ptr_v[level+1];
      n_local = n_end - n_start;
      n_remain = this->_n - n_end;
      
      VectorType xiu, xil, ziu, zil;
      VectorType zu, zl, worku, workl;
      
      worku.SetupPtr(level_str._v_temp, n_local, 0, false);
      workl.SetupPtr(level_str._v_temp, n_remain, n_local, false);
      zu.SetupPtr(level_str._w_temp, n_local, 0, false);
      zl.SetupPtr(level_str._w_temp, n_remain, n_local, false);
      
      /* we first compute y = -alpha*EB^{-1}F*x + beta y */
      
      /* Matvec zu = Fx */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_FMV, (level_str._F_mat.MatVec( 'N', one, x, zero, zu)));
      
      /* Solve worku = B\zu */
      
      if(this->_gemslr_setups._solve_phase_setup == kGemslrPhaseSetup)
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_SOLVELU, this->SolveB(worku, zu, level));
      }
      else
      {
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_ILUT, this->SolveB(worku, zu, level));
      }
      
      /* Matvec y = -alpha*E*worku + beta*y */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_EMV, (level_str._E_mat.MatVec( 'N', malpha, worku, beta, y)));
      
      /* now compute y = alpha*C*x + y 
       * note that C = | Bi  Fi | 
       *               | Ei  Ci |
       * We put everything to zl first, and apply alpha to zl
       */
      
      zl.Fill(zero);
      for(j = level + 1 ; j < this->_nlev_used ; j ++)
      {
         GemslrLevelClass< MatrixType, VectorType, DataType> &level_strj = this->_levs_v[j];
         
         ncompi = level_strj._ncomps;
         
         n_starti = this->_lev_ptr_v[j];
         n_endi = this->_lev_ptr_v[j+1];
         n_locali = n_endi - n_starti;
         n_remaini = this->_n - n_endi;
         
         if(j == this->_nlev_used-1)
         {
            /* the last level, matvec with B */
            
            /* z = B*x */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, n1, n2) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
            for(i = 0 ; i < ncompi ; i ++)
            {
               
               VectorType zi, xi;
               
               n1 = this->_dom_ptr_v2[j][i];
               n2 = this->_dom_ptr_v2[j][i+1];
               /* note that x only has the lower part 
                * x and z has different shift
                */
               xi.SetupPtr(x, n2 - n1, n1 - n_start - n_local, false);
               zi.SetupPtr(level_str._w_temp, n2 - n1, n1 - n_start, false);
               
               level_strj._B_mat_v[i].MatVec( 'N', one, xi, one, zi);
               
               xi.Clear();
               zi.Clear();
            }
            
            break;
         }
         /* not the last level 
          * we need Fi, Ei, and Bi
          */
         
         /* apply B matvec */
         
         /* z = B*x */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, n1, n2) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
         for(i = 0 ; i < ncompi ; i ++)
         {
            n1 = this->_dom_ptr_v2[j][i];
            n2 = this->_dom_ptr_v2[j][i+1];
            
            VectorType zi, xi;
            
            /* note that x only has the lower part 
             * x and z has different shift
             */
            xi.SetupPtr(x, n2 - n1, n1 - n_start - n_local, false);
            zi.SetupPtr(level_str._w_temp, n2 - n1, n1 - n_start, false);
            
            level_strj._B_mat_v[i].MatVec( 'N', one, xi, one, zi);
            
            xi.Clear();
            zi.Clear();
         }
         
         /* with F and E */
         xil.SetupPtr(x, n_remaini, n_starti + n_locali - n_start - n_local, false);
         ziu.SetupPtr(level_str._w_temp, n_locali, n_starti - n_start, false);
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_FMV, (level_strj._F_mat.MatVec( 'N', one, xil, one, ziu)));
         
         xiu.SetupPtr(x, n_locali, n_starti - n_start - n_local, false);
         zil.SetupPtr(level_str._w_temp, n_remaini, n_starti + n_locali - n_start, false);
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_EMV, (level_strj._E_mat.MatVec( 'N', one, xiu, one, zil)));
         
      }
      
      /* now add the result to y */
      y.Axpy(alpha, zl);
      
      xiu.Clear();
      xil.Clear();
      ziu.Clear();
      zil.Clear();
      zu.Clear();
      zl.Clear();
      worku.Clear();
      workl.Clear();
      
      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_seq_float::SchurMatVec( int level, char trans, const float &alpha, SequentialVectorClass<float> &x, const float &beta, SequentialVectorClass<float> &y);
   template int precond_gemslr_csr_seq_double::SchurMatVec( int level, char trans, const double &alpha, SequentialVectorClass<double> &x, const double &beta, SequentialVectorClass<double> &y);
   template int precond_gemslr_csr_seq_complexs::SchurMatVec( int level, char trans, const complexs &alpha, SequentialVectorClass<complexs> &x, const complexs &beta, SequentialVectorClass<complexs> &y);
   template int precond_gemslr_csr_seq_complexd::SchurMatVec( int level, char trans, const complexd &alpha, SequentialVectorClass<complexd> &x, const complexd &beta, SequentialVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::CMatVec( int level, char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      /* no inner iteration on the last level */
      PARGEMSLR_CHKERR(level >= this->_nlev_used - 1);
      
      /* define the data type */
      typedef DataType T;
      
      int i, n_start, n_local, n_end, n_remain, n1, n2;
      int j, ncompi, n_starti, n_locali, n_endi, n_remaini;
      
      GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
      
      T one, zero, malpha;
      
      one = 1.0;
      zero = 0.0;
      malpha = -alpha;
      
      n_start = this->_lev_ptr_v[level];
      n_end = this->_lev_ptr_v[level+1];
      n_local = n_end - n_start;
      n_remain = this->_n - n_end;
      
      VectorType xiu, xil, ziu, zil;
      VectorType zu, zl, worku, workl;
      
      worku.SetupPtr(level_str._v_temp, n_local, 0, false);
      workl.SetupPtr(level_str._v_temp, n_remain, n_local, false);
      zu.SetupPtr(level_str._w_temp, n_local, 0, false);
      zl.SetupPtr(level_str._w_temp, n_remain, n_local, false);
      
      /* scale y = beta * y */
      
      y.Scale(beta);

      /* now compute y = alpha*C*x + y 
       * note that C = | Bi  Fi | 
       *               | Ei  Ci |
       * We put everything to zl first, and apply alpha to zl
       */
      
      zl.Fill(zero);
      for(j = level + 1 ; j < this->_nlev_used ; j ++)
      {
         GemslrLevelClass< MatrixType, VectorType, DataType> &level_strj = this->_levs_v[j];
         
         ncompi = level_strj._ncomps;
         
         n_starti = this->_lev_ptr_v[j];
         n_endi = this->_lev_ptr_v[j+1];
         n_locali = n_endi - n_starti;
         n_remaini = this->_n - n_endi;
         
         if(j == this->_nlev_used-1)
         {
            /* the last level, matvec with B */
            
            /* z = B*x */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, n1, n2) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
            for(i = 0 ; i < ncompi ; i ++)
            {
               
               VectorType zi, xi;
               
               n1 = this->_dom_ptr_v2[j][i];
               n2 = this->_dom_ptr_v2[j][i+1];
               /* note that x only has the lower part 
                * x and z has different shift
                */
               xi.SetupPtr(x, n2 - n1, n1 - n_start - n_local, false);
               zi.SetupPtr(level_str._w_temp, n2 - n1, n1 - n_start, false);
               
               level_strj._B_mat_v[i].MatVec( 'N', one, xi, one, zi);
               
               xi.Clear();
               zi.Clear();
            }
            
            break;
         }
         /* not the last level 
          * we need Fi, Ei, and Bi
          */
         
         /* apply B matvec */
         
         /* z = B*x */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i, n1, n2) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
         for(i = 0 ; i < ncompi ; i ++)
         {
            n1 = this->_dom_ptr_v2[j][i];
            n2 = this->_dom_ptr_v2[j][i+1];
            
            VectorType zi, xi;
            
            /* note that x only has the lower part 
             * x and z has different shift
             */
            xi.SetupPtr(x, n2 - n1, n1 - n_start - n_local, false);
            zi.SetupPtr(level_str._w_temp, n2 - n1, n1 - n_start, false);
            
            level_strj._B_mat_v[i].MatVec( 'N', one, xi, one, zi);
            
            xi.Clear();
            zi.Clear();
         }
         
         /* with F and E */
         xil.SetupPtr(x, n_remaini, n_starti + n_locali - n_start - n_local, false);
         ziu.SetupPtr(level_str._w_temp, n_locali, n_starti - n_start, false);
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_FMV, (level_strj._F_mat.MatVec( 'N', one, xil, one, ziu)));
         
         xiu.SetupPtr(x, n_locali, n_starti - n_start - n_local, false);
         zil.SetupPtr(level_str._w_temp, n_remaini, n_starti + n_locali - n_start, false);
         PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_PRECTIME_EMV, (level_strj._E_mat.MatVec( 'N', one, xiu, one, zil)));
         
      }
      
      /* now add the result to y */
      y.Axpy(alpha, zl);
      
      xiu.Clear();
      xil.Clear();
      ziu.Clear();
      zil.Clear();
      zu.Clear();
      zl.Clear();
      worku.Clear();
      workl.Clear();
      
      return PARGEMSLR_SUCCESS;
   
   }
   template int precond_gemslr_csr_seq_float::CMatVec( int level, char trans, const float &alpha, SequentialVectorClass<float> &x, const float &beta, SequentialVectorClass<float> &y);
   template int precond_gemslr_csr_seq_double::CMatVec( int level, char trans, const double &alpha, SequentialVectorClass<double> &x, const double &beta, SequentialVectorClass<double> &y);
   template int precond_gemslr_csr_seq_complexs::CMatVec( int level, char trans, const complexs &alpha, SequentialVectorClass<complexs> &x, const complexs &beta, SequentialVectorClass<complexs> &y);
   template int precond_gemslr_csr_seq_complexd::CMatVec( int level, char trans, const complexd &alpha, SequentialVectorClass<complexd> &x, const complexd &beta, SequentialVectorClass<complexd> &y);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::GetSize()
   {
      return this->_n;
   }
   template int precond_gemslr_csr_seq_float::GetSize();
   template int precond_gemslr_csr_seq_double::GetSize();
   template int precond_gemslr_csr_seq_complexs::GetSize();
   template int precond_gemslr_csr_seq_complexd::GetSize();
   
   template <class MatrixType, class VectorType, typename DataType>
   long int GemslrClass<MatrixType, VectorType, DataType>::GetNumNonzeros()
   {
      long int nnz_bsolver, nnz_lr;
      
      return this->GetNumNonzeros(nnz_bsolver, nnz_lr);
   }
   template long int precond_gemslr_csr_seq_float::GetNumNonzeros();
   template long int precond_gemslr_csr_seq_double::GetNumNonzeros();
   template long int precond_gemslr_csr_seq_complexs::GetNumNonzeros();
   template long int precond_gemslr_csr_seq_complexd::GetNumNonzeros();
   
   template <class MatrixType, class VectorType, typename DataType>
   long int GemslrClass<MatrixType, VectorType, DataType>::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr)
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
      
      return nnz_bsolver + nnz_lr;
   }
   template long int precond_gemslr_csr_seq_float::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template long int precond_gemslr_csr_seq_double::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template long int precond_gemslr_csr_seq_complexs::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   template long int precond_gemslr_csr_seq_complexd::GetNumNonzeros(long int &nnz_bsolver, long int &nnz_lr);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::SetSolveLocation( const int &location)
   {
      
      if( this->_location != location && this->_ready)
      {
         this->MoveData(location);
      }
      
      this->_location = location;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_seq_float::SetSolveLocation( const int &location);
   template int precond_gemslr_csr_seq_double::SetSolveLocation( const int &location);
   template int precond_gemslr_csr_seq_complexs::SetSolveLocation( const int &location);
   template int precond_gemslr_csr_seq_complexd::SetSolveLocation( const int &location);
   
   template <class MatrixType, class VectorType, typename DataType>
   int GemslrClass<MatrixType, VectorType, DataType>::MoveData( const int &location)
   {
      if( this->_location == location)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      this->_location = location;
      
      if(this->_ready)
      {
         
         int         i, level, ncomp;
         
         /* temp buffer */
         this->_x_temp.MoveData(this->_location);
         this->_rhs_temp.MoveData(this->_location);
         
         this->_inner_iters_solver.SetSolveLocation(this->_location);
         this->_inner_iters_precond.SetSolveLocation(this->_location);
         
         /* permutation array */
         this->_pperm.MoveData(this->_location);
         this->_qperm.MoveData(this->_location);
         
         /* move B and B solver */
         for(level = 0 ; level < this->_nlev_used ; level++)
         {
            GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
            
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
         
         /* move E, F, and the low-rank term */
         for(level = 0 ; level < this->_nlev_used-1 ; level++)
         {
            GemslrLevelClass< MatrixType, VectorType, DataType> &level_str = this->_levs_v[level];
            
            level_str._E_mat.MoveData(this->_location);
            level_str._F_mat.MoveData(this->_location);
            level_str._C_mat.MoveData(this->_location);
            level_str._D_mat.MoveData(this->_location);
            
            level_str._xlr_temp.MoveData(this->_location);
            level_str._xlr1_temp.MoveData(this->_location);
            level_str._xlr2_temp.MoveData(this->_location);
            
            level_str._y_temp.MoveData(this->_location);
            level_str._z_temp.MoveData(this->_location);
            level_str._v_temp.MoveData(this->_location);
            level_str._w_temp.MoveData(this->_location);
            
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_gemslr_csr_seq_float::MoveData( const int &location);
   template int precond_gemslr_csr_seq_double::MoveData( const int &location);
   template int precond_gemslr_csr_seq_complexs::MoveData( const int &location);
   template int precond_gemslr_csr_seq_complexd::MoveData( const int &location);
   
}
