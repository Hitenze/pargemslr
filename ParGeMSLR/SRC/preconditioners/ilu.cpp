
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

namespace pargemslr
{
	
   template <class MatrixType, class VectorType, typename DataType>
   IluClass<MatrixType, VectorType, DataType>::IluClass() : SolverClass<MatrixType, VectorType, DataType>()
   {
      this->_solver_type = kSolverIlu;
#ifdef PARGEMSLR_CUDA
      this->_matL_info = NULL;
      this->_matU_info = NULL;
#endif
      this->_location = kMemoryHost;
      this->_n = 0;
      this->_modified = false;
      this->_complex_shift = false;
      this->_nB = 0;
      this->_nnz = 0;
      this->_fill_level = 1;
      this->_droptol = 1e-02;
      this->_max_row_nnz = 100;
      this->_option = kIluOptionILUT;
      this->_perm_option = kIluReorderingRcm;
      this->_omp_option = kIluOpenMPLevelScheduling;
      this->_poly_order = 3;
   }
   template precond_ilu_csr_seq_float::IluClass();
   template precond_ilu_csr_seq_double::IluClass();
   template precond_ilu_csr_seq_complexs::IluClass();
   template precond_ilu_csr_seq_complexd::IluClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   IluClass<MatrixType, VectorType, DataType>::IluClass(const IluClass<MatrixType, VectorType, DataType> &precond) : SolverClass<MatrixType, VectorType, DataType>(precond)
   {
      this->Clear();
#ifdef PARGEMSLR_CUDA
      this->_LDU = precond._LDU;
      this->_matL_info = precond._matL_info;
      this->_matU_info = precond._matU_info;
      this->_cusparse_ready = precond._cusparse_ready;
#endif
      this->_location = precond._location;
      this->_n = precond._n;
      this->_complex_shift = precond._complex_shift;
      this->_nnz = precond._nnz;
      this->_L = precond._L;
      this->_D = precond._D;
      this->_U = precond._U;
      this->_nB = precond._nB;
      this->_E = precond._E;
      this->_F = precond._F;
      this->_S = precond._S;
#ifdef PARGEMSLR_OPENMP
      this->_L_poly = precond._L_poly;
      this->_D_poly = precond._D_poly;
      this->_U_poly = precond._U_poly;
      this->_L_level = precond._L_level;
      this->_D_level = precond._D_level;
      this->_U_level = precond._U_level;
      this->_level_ptr_l = precond._level_ptr_l;
      this->_levels_l = precond._levels_l;
      this->_level_ptr_u = precond._level_ptr_u;
      this->_levels_u = precond._levels_u;
      this->_levels_l_start = precond._levels_l_start;
      this->_levels_l_end = precond._levels_l_end;
      this->_levels_u_start = precond._levels_u_start;
      this->_levels_u_end = precond._levels_u_end;
      this->_y_temp = precond._y_temp;
      this->_z_temp = precond._z_temp;
#endif
      this->_droptol = precond._droptol;
      this->_fill_level = precond._fill_level;
      this->_max_row_nnz = precond._max_row_nnz;
      this->_option = precond._option;
      this->_perm_option = precond._perm_option;
      this->_omp_option = precond._omp_option;
      this->_poly_order = precond._poly_order;
      this->_row_perm_vec = precond._row_perm_vec;
      this->_col_perm_vec = precond._col_perm_vec;
      this->_x_temp = precond._x_temp;
      this->_modified = precond._modified;
      
   }
   template precond_ilu_csr_seq_float::IluClass(const precond_ilu_csr_seq_float &precond);
   template precond_ilu_csr_seq_double::IluClass(const precond_ilu_csr_seq_double &precond);
   template precond_ilu_csr_seq_complexs::IluClass(const precond_ilu_csr_seq_complexs &precond);
   template precond_ilu_csr_seq_complexd::IluClass(const precond_ilu_csr_seq_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   IluClass<MatrixType, VectorType, DataType>::IluClass( IluClass<MatrixType, VectorType, DataType> &&precond) : SolverClass<MatrixType, VectorType, DataType>(std::move(precond))
   {
      this->Clear();
#ifdef PARGEMSLR_CUDA
      this->_LDU = std::move(precond._LDU);
      this->_matL_info = precond._matL_info;
      precond._matL_info = NULL;
      this->_matU_info = precond._matU_info;
      precond._matU_info = NULL;
      this->_cusparse_ready = precond._cusparse_ready;
      precond._cusparse_ready = false;
#endif
      this->_location = precond._location;
      precond._location = kMemoryHost;
      this->_n = precond._n;
      precond._n = 0;
      this->_complex_shift = precond._complex_shift;
      precond._complex_shift = false;
      this->_nnz = precond._nnz;
      precond._nnz = 0;
      this->_L = std::move(precond._L);
      this->_D = std::move(precond._D);
      this->_U = std::move(precond._U);
      this->_nB = precond._nB;
      precond._nB = 0;
      this->_E = std::move(precond._E);
      this->_F = std::move(precond._F);
      this->_S = std::move(precond._S);
#ifdef PARGEMSLR_OPENMP
      this->_L_poly = std::move(precond._L_poly);
      this->_D_poly = std::move(precond._D_poly);
      this->_U_poly = std::move(precond._U_poly);
      this->_L_level = std::move(precond._L_level);
      this->_D_level = std::move(precond._D_level);
      this->_U_level = std::move(precond._U_level);
      this->_level_ptr_l = std::move(precond._level_ptr_l);
      this->_levels_l = std::move(precond._levels_l);
      this->_level_ptr_u = std::move(precond._level_ptr_u);
      this->_levels_u = std::move(precond._levels_u);
      this->_levels_l_start = precond._levels_l_start;
      precond._levels_l_start = 0;
      this->_levels_l_end = precond._levels_l_end;
      precond._levels_l_end = 0;
      this->_levels_u_start = precond._levels_u_start;
      precond._levels_u_start = 0;
      this->_levels_u_end = precond._levels_u_end;
      precond._levels_u_end = 0;
      this->_y_temp = std::move(precond._y_temp);
      this->_z_temp = std::move(precond._z_temp);
      this->_modified = precond._modified;
      precond._modified = false;
#endif
      this->_droptol = precond._droptol;
      precond._droptol = 1e-02;
      this->_fill_level = precond._fill_level;
      precond._fill_level = 1;
      this->_max_row_nnz = precond._max_row_nnz;
      precond._max_row_nnz = 100;
      this->_option = precond._option;
      precond._option = kIluOptionILUT;
      this->_perm_option = precond._perm_option;
      precond._perm_option = kIluReorderingRcm;
      this->_omp_option = precond._omp_option;
      precond._omp_option = kIluOpenMPLevelScheduling;
      this->_poly_order = precond._poly_order;
      precond._poly_order = 3;
      this->_row_perm_vec = std::move(precond._row_perm_vec);
      this->_col_perm_vec = std::move(precond._col_perm_vec);
      this->_x_temp = std::move(precond._x_temp);
      
   }
   template precond_ilu_csr_seq_float::IluClass( precond_ilu_csr_seq_float &&precond);
   template precond_ilu_csr_seq_double::IluClass( precond_ilu_csr_seq_double &&precond);
   template precond_ilu_csr_seq_complexs::IluClass( precond_ilu_csr_seq_complexs &&precond);
   template precond_ilu_csr_seq_complexd::IluClass( precond_ilu_csr_seq_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   IluClass<MatrixType, VectorType, DataType>& IluClass<MatrixType, VectorType, DataType>::operator= (const IluClass<MatrixType, VectorType, DataType> &precond)
   {
      this->Clear();
      SolverClass<MatrixType, VectorType, DataType>::operator=(precond);
#ifdef PARGEMSLR_CUDA
      this->_LDU = precond._LDU;
      this->_matL_info = precond._matL_info;
      this->_matU_info = precond._matU_info;
      this->_cusparse_ready = precond._cusparse_ready;
#endif
      this->_location = precond._location;
      this->_n = precond._n;
      this->_complex_shift = precond._complex_shift;
      this->_nnz = precond._nnz;
      this->_L = precond._L;
      this->_D = precond._D;
      this->_U = precond._U;
      this->_nB = precond._nB;
      this->_E = precond._E;
      this->_F = precond._F;
      this->_S = precond._S;
#ifdef PARGEMSLR_OPENMP
      this->_L_poly = precond._L_poly;
      this->_D_poly = precond._D_poly;
      this->_U_poly = precond._U_poly;
      this->_L_level = precond._L_level;
      this->_D_level = precond._D_level;
      this->_U_level = precond._U_level;
      this->_level_ptr_l = precond._level_ptr_l;
      this->_levels_l = precond._levels_l;
      this->_level_ptr_u = precond._level_ptr_u;
      this->_levels_u = precond._levels_u;
      this->_levels_l_start = precond._levels_l_start;
      this->_levels_l_end = precond._levels_l_end;
      this->_levels_u_start = precond._levels_u_start;
      this->_levels_u_end = precond._levels_u_end;
      this->_y_temp = precond._y_temp;
      this->_z_temp = precond._z_temp;
#endif
      this->_droptol = precond._droptol;
      this->_fill_level = precond._fill_level;
      this->_max_row_nnz = precond._max_row_nnz;
      this->_option = precond._option;
      this->_perm_option = precond._perm_option;
      this->_omp_option = precond._omp_option;
      this->_poly_order = precond._poly_order;
      this->_row_perm_vec = precond._row_perm_vec;
      this->_col_perm_vec = precond._col_perm_vec;
      this->_x_temp = precond._x_temp;
      this->_modified = precond._modified;
      return *this;
      
   }
   template precond_ilu_csr_seq_float& precond_ilu_csr_seq_float::operator= (const precond_ilu_csr_seq_float &precond);
   template precond_ilu_csr_seq_double& precond_ilu_csr_seq_double::operator= (const precond_ilu_csr_seq_double &precond);
   template precond_ilu_csr_seq_complexs& precond_ilu_csr_seq_complexs::operator= (const precond_ilu_csr_seq_complexs &precond);
   template precond_ilu_csr_seq_complexd& precond_ilu_csr_seq_complexd::operator= (const precond_ilu_csr_seq_complexd &precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   IluClass<MatrixType, VectorType, DataType>& IluClass<MatrixType, VectorType, DataType>::operator= ( IluClass<MatrixType, VectorType, DataType> &&precond)
   {
      this->Clear();
      SolverClass<MatrixType, VectorType, DataType>::operator=(std::move(precond));
#ifdef PARGEMSLR_CUDA
      this->_LDU = std::move(precond._LDU);
      this->_matL_info = precond._matL_info;
      precond._matL_info = NULL;
      this->_matU_info = precond._matU_info;
      precond._matU_info = NULL;
      this->_cusparse_ready = precond._cusparse_ready;
      precond._cusparse_ready = false;
#endif
      this->_location = precond._location;
      precond._location = kMemoryHost;
      this->_n = precond._n;
      precond._n = 0;
      this->_complex_shift = precond._complex_shift;
      precond._complex_shift = false;
      this->_nnz = precond._nnz;
      precond._nnz = 0;
      this->_L = std::move(precond._L);
      this->_D = std::move(precond._D);
      this->_U = std::move(precond._U);
      this->_nB = precond._nB;
      precond._nB = 0;
      this->_E = std::move(precond._E);
      this->_F = std::move(precond._F);
      this->_S = std::move(precond._S);
#ifdef PARGEMSLR_OPENMP
      this->_L_poly = std::move(precond._L_poly);
      this->_D_poly = std::move(precond._D_poly);
      this->_U_poly = std::move(precond._U_poly);
      this->_L_level = std::move(precond._L_level);
      this->_D_level = std::move(precond._D_level);
      this->_U_level = std::move(precond._U_level);
      this->_level_ptr_l = std::move(precond._level_ptr_l);
      this->_levels_l = std::move(precond._levels_l);
      this->_level_ptr_u = std::move(precond._level_ptr_u);
      this->_levels_u = std::move(precond._levels_u);
      this->_levels_l_start = precond._levels_l_start;
      precond._levels_l_start = 0;
      this->_levels_l_end = precond._levels_l_end;
      precond._levels_l_end = 0;
      this->_levels_u_start = precond._levels_u_start;
      precond._levels_u_start = 0;
      this->_levels_u_end = precond._levels_u_end;
      precond._levels_u_end = 0;
      this->_y_temp = std::move(precond._y_temp);
      this->_z_temp = std::move(precond._z_temp);
#endif
      this->_droptol = precond._droptol;
      precond._droptol = 1e-02;
      this->_fill_level = precond._fill_level;
      precond._fill_level = 1;
      this->_max_row_nnz = precond._max_row_nnz;
      precond._max_row_nnz = 100;
      this->_option = precond._option;
      precond._option = kIluOptionILUT;
      this->_perm_option = precond._perm_option;
      precond._perm_option = kIluReorderingRcm;
      this->_omp_option = precond._omp_option;
      precond._omp_option = kIluOpenMPLevelScheduling;
      this->_poly_order = precond._poly_order;
      precond._poly_order = 3;
      this->_row_perm_vec = std::move(precond._row_perm_vec);
      this->_col_perm_vec = std::move(precond._col_perm_vec);
      this->_x_temp = std::move(precond._x_temp);
      this->_modified = precond._modified;
      precond._modified = false;
      return *this;
      
   }
   template precond_ilu_csr_seq_float& precond_ilu_csr_seq_float::operator= ( precond_ilu_csr_seq_float &&precond);
   template precond_ilu_csr_seq_double& precond_ilu_csr_seq_double::operator= ( precond_ilu_csr_seq_double &&precond);
   template precond_ilu_csr_seq_complexs& precond_ilu_csr_seq_complexs::operator= ( precond_ilu_csr_seq_complexs &&precond);
   template precond_ilu_csr_seq_complexd& precond_ilu_csr_seq_complexd::operator= ( precond_ilu_csr_seq_complexd &&precond);
   
   template <class MatrixType, class VectorType, typename DataType>
   IluClass<MatrixType, VectorType, DataType>::~IluClass()
   {
      this->Clear();
   }
   template precond_ilu_csr_seq_float::~IluClass();
   template precond_ilu_csr_seq_double::~IluClass();
   template precond_ilu_csr_seq_complexs::~IluClass();
   template precond_ilu_csr_seq_complexd::~IluClass();

#ifndef PARGEMSLR_CUDA

   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::Clear()
   {
      SolverClass<MatrixType, VectorType, DataType>::Clear();
      this->_location = kMemoryHost;
      this->_n = 0;
      this->_nnz = 0;
      this->_L.Clear();
      this->_D.Clear();
      this->_U.Clear();
      this->_E.Clear();
      this->_F.Clear();
      this->_S.Clear();
      this->_nB = 0;
#ifdef PARGEMSLR_OPENMP
      this->_L_poly.Clear();
      this->_D_poly.Clear();
      this->_U_poly.Clear();
      this->_y_temp.Clear();
      this->_z_temp.Clear();
      this->_L_level.Clear();
      this->_D_level.Clear();
      this->_U_level.Clear();
      this->_level_ptr_l.Clear();
      this->_levels_l.Clear();
      this->_level_ptr_u.Clear();
      this->_levels_u.Clear();
      this->_levels_l_start = 0;
      this->_levels_l_end = 0;
      this->_levels_u_start = 0;
      this->_levels_u_end = 0;
#endif
      this->_droptol = 1e-02;
      this->_fill_level = 1;
      this->_max_row_nnz = 100;
      this->_option = kIluOptionILUT;
      this->_perm_option = kIluReorderingRcm;
      this->_omp_option = kIluOpenMPLevelScheduling;
      this->_poly_order = 3;
      this->_row_perm_vec.Clear();
      this->_col_perm_vec.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_ilu_csr_seq_float::Clear();
   template int precond_ilu_csr_seq_double::Clear();
   template int precond_ilu_csr_seq_complexs::Clear();
   template int precond_ilu_csr_seq_complexd::Clear();
   
#endif

   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::Setup( VectorType &x, VectorType &rhs)
   {
      /* Wrapper of the setup phase of ilu */
      if(this->_ready)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      if(!this->_matrix)
      {
         PARGEMSLR_ERROR("Call ILU without matrix.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }
      
      /* update the solver precision, leave this interface for half precision */
      this->_solver_precision = x.GetPrecision();
      
      /* set the backup location of the matrix, might need to move during the setup phase */
      this->_matrix_location = this->_matrix->GetDataLocation();
      
      /* build the local permutation (for example RCM) */
      this->SetupPermutation( x, rhs);
      
      /* start the real factorization */
      switch(this->_option)
      {
         case kIluOptionILUT:
         {
            /* ILUT */
            if(this->_matrix->GetDataLocation() == kMemoryDevice)
            {
               PARGEMSLR_WARNING("ILUT only works on the host memory. Moving data to the host.");
               this->_matrix->MoveData(kMemoryHost);
            }
            this->SetupILUT( x, rhs);
            break;
         }
         case kIluOptionILUK:
         {
            /* ILUT */
            if(this->_matrix->GetDataLocation() == kMemoryDevice)
            {
               PARGEMSLR_WARNING("ILUK only works on the host memory. Moving data to the host.");
               this->_matrix->MoveData(kMemoryHost);
            }
            this->SetupILUK( x, rhs);
            break;
         }
         case kIluOptionPartialILUT:
         {
            /* partial ILUT */
            if(this->_matrix->GetDataLocation() == kMemoryDevice)
            {
               PARGEMSLR_WARNING("ILUT only works on the host memory. Moving data to the host.");
               this->_matrix->MoveData(kMemoryHost);
            }
            this->SetupPartialILUT( x, rhs);
            break;
         }
         default:
         {
            /* Invalid option */
            PARGEMSLR_ERROR("Invalid ILU option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      /* setup the solve buffer */
      if(this->_x_temp.GetLengthLocal() != this->_n)
      {
         this->_x_temp.Setup( this->_n, this->_location, true);
      }
      
      /* the the preconditioner is ready, move to location */
      this->_ready = true;
      this->MoveData(this->_location);
      
      /* move the matrix back to its original location */
      this->_matrix->MoveData(this->_matrix_location);

#ifdef PARGEMSLR_OPENMP
      if( (this->_option == kIluOptionILUT || this->_option == kIluOptionILUK)
         && this->_omp_option == kIluOpenMPLevelScheduling && PargemslrGetOpenmpGlobalMaxNumThreads() > 1)
      {
         switch(this->_omp_option)
         {
            case kIluOpenMPLevelScheduling:
            {
               this->BuildLevels();
               break;
            }
            case kIluOpenMPPoly:
            {
               this->BuildPoly();
               break;
            }
            case kIluOpenMPNo: default:
            {
               /* do nothing */
               break;
            }
         }
      }
#endif
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_ilu_csr_seq_float::Setup( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::Setup( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::Setup( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::Setup( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::Setup(MatrixType &L_mat, VectorType &D_vec, MatrixType &U_mat, int *pperm, int *qperm)
   {
      /* Wrapper of the setup phase of ilu */
      if(this->_ready)
      {
         PARGEMSLR_WARNING("Set user LDU failed.");
         return PARGEMSLR_SUCCESS;
      }
      
      if(!this->_matrix)
      {
         PARGEMSLR_ERROR("Call ILU without matrix.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }
      
      if(L_mat.GetDataLocation() == kMemoryDevice || D_vec.GetDataLocation() == kMemoryDevice || U_mat.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Setup ILU with L, D, and U, must all be on the host memory.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }
      
      /* update the solver precision, leave this interface for half precision */
      this->_solver_precision = this->_matrix->GetPrecision();
      
      /* set the backup location of the matrix, might need to move during the setup phase */
      this->_matrix_location = this->_matrix->GetDataLocation();
      
      /* set the local permutation */
      this->_n = D_vec.GetLengthLocal();
      
      /* by using this setup, we are actually doing ILUT */
      this->_option = kIluOptionILUT;
      
      if(pperm)
      {
         
         /* copy the permutation */
         
         if(qperm)
         {
            this->_row_perm_vec.Setup(this->_n);
            this->_col_perm_vec.Setup(this->_n);
            PARGEMSLR_MEMCPY(this->_row_perm_vec.GetData(), pperm, this->_n, kMemoryHost, kMemoryHost, int );
            PARGEMSLR_MEMCPY(this->_col_perm_vec.GetData(), qperm, this->_n, kMemoryHost, kMemoryHost, int );
         }
         else
         {
            this->_row_perm_vec.Setup(this->_n);
            this->_col_perm_vec.Setup(0);
            PARGEMSLR_MEMCPY(this->_row_perm_vec.GetData(), pperm, this->_n, kMemoryHost, kMemoryHost, int );
         }
         
      }
      else
      {
         /* NULL, use nature ordering */
         this->_row_perm_vec.Setup(0);
         this->_col_perm_vec.Setup(0);
      }
      
      /* setup the factorization */
      this->_L = L_mat;
      this->_D = D_vec;
      this->_U = U_mat;
      
      this->_nnz = this->_n + L_mat.GetNumNonzeros() + U_mat.GetNumNonzeros();
      
      /* setup the solve buffer */
      if(this->_x_temp.GetLengthLocal() != this->_n)
      {
         this->_x_temp.Setup( this->_n, this->_location, true);
      }
      
      /* the the preconditioner is ready, move to location */
      this->_ready = true;
      this->MoveData(this->_location);
      
      /* move the matrix back to its original location */
      this->_matrix->MoveData(this->_matrix_location);

#ifdef PARGEMSLR_OPENMP
      if(this->_omp_option == kIluOpenMPLevelScheduling && PargemslrGetOpenmpGlobalMaxNumThreads() > 1)
      {
         switch(this->_omp_option)
         {
            case kIluOpenMPLevelScheduling:
            {
               this->BuildLevels();
               break;
            }
            case kIluOpenMPPoly:
            {
               this->BuildPoly();
               break;
            }
            case kIluOpenMPNo: default:
            {
               /* do nothing */
               break;
            }
         }
      }
#endif
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::Setup( CsrMatrixClass<float> &L_mat, SequentialVectorClass<float> &D_vec,  CsrMatrixClass<float> &U_mat, int *pperm, int *qperm);
   template int precond_ilu_csr_seq_double::Setup( CsrMatrixClass<double> &L_mat, SequentialVectorClass<double> &D_vec,  CsrMatrixClass<double> &U_mat, int *pperm, int *qperm);
   template int precond_ilu_csr_seq_complexs::Setup( CsrMatrixClass<complexs> &L_mat, SequentialVectorClass<complexs> &D_vec,  CsrMatrixClass<complexs> &U_mat, int *pperm, int *qperm);
   template int precond_ilu_csr_seq_complexd::Setup( CsrMatrixClass<complexd> &L_mat, SequentialVectorClass<complexd> &D_vec,  CsrMatrixClass<complexd> &U_mat, int *pperm, int *qperm);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetupPermutation( VectorType &x, VectorType &rhs)
   {
      //int n_local = this->_matrix->GetNumRowsLocal();
      
      PARGEMSLR_CHKERR(this->_matrix->GetNumRowsLocal() != this->_matrix->GetNumColsLocal())
      
      switch(this->_perm_option)
      {
         case kIluReorderingNo:
         {
            /* nature ordering, the default reordering location is on the host */
            this->_row_perm_vec.Setup(0);
            this->_col_perm_vec.Setup(0);
            
            break;
         }
         case kIluReorderingRcm:
         {
            if(this->_matrix->GetDataLocation() == kMemoryDevice)
            {
               PARGEMSLR_WARNING("ILU reordering only works on the host memory. Moving data to the host.");
               this->_matrix->MoveData(kMemoryHost);
            }
            
            /* note that we can't sychronize this part with other MPI ranks */
            pargemslr::ParallelLogClass::_times_buffer_start[PARGEMSLR_BUILDTIME_RCM] = MPI_Wtime();
            CsrMatrixRcmHost(*(this->_matrix), this->_row_perm_vec);
            pargemslr::ParallelLogClass::_times_buffer_end[PARGEMSLR_BUILDTIME_RCM] = MPI_Wtime();
            pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_RCM] += pargemslr::ParallelLogClass::_times_buffer_end[PARGEMSLR_BUILDTIME_RCM] - pargemslr::ParallelLogClass::_times_buffer_start[PARGEMSLR_BUILDTIME_RCM];
            
            if(this->_print_option > 1)
            {
               this->_matrix->PlotPatternGnuPlot("A_rcm.data",this->_row_perm_vec.GetData(),this->_row_perm_vec.GetData(), 0, 0);
            }
            break;
         }
         case kIluReorderingAmd:
         {
            if(this->_matrix->GetDataLocation() == kMemoryDevice)
            {
               PARGEMSLR_WARNING("ILU reordering only works on the host memory. Moving data to the host.");
               this->_matrix->MoveData(kMemoryHost);
            }
            
            /* note that we can't sychronize this part with other MPI ranks */
            pargemslr::ParallelLogClass::_times_buffer_start[PARGEMSLR_BUILDTIME_RCM] = MPI_Wtime();
            CsrMatrixAmdHost(*(this->_matrix), this->_row_perm_vec);
            pargemslr::ParallelLogClass::_times_buffer_end[PARGEMSLR_BUILDTIME_RCM] = MPI_Wtime();
            pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_RCM] += pargemslr::ParallelLogClass::_times_buffer_end[PARGEMSLR_BUILDTIME_RCM] - pargemslr::ParallelLogClass::_times_buffer_start[PARGEMSLR_BUILDTIME_RCM];
            
            if(this->_print_option > 1)
            {
               this->_matrix->PlotPatternGnuPlot("A_amd.data",this->_row_perm_vec.GetData(),this->_row_perm_vec.GetData(), 0, 0);
            }
            break;
         }
         case kIluReorderingNd:
         {
            if(this->_matrix->GetDataLocation() == kMemoryDevice)
            {
               PARGEMSLR_WARNING("ILU reordering only works on the host memory. Moving data to the host.");
               this->_matrix->MoveData(kMemoryHost);
            }
            
            /* note that we can't sychronize this part with other MPI ranks */
            pargemslr::ParallelLogClass::_times_buffer_start[PARGEMSLR_BUILDTIME_RCM] = MPI_Wtime();
            CsrMatrixNdHost(*(this->_matrix), this->_row_perm_vec);
            pargemslr::ParallelLogClass::_times_buffer_end[PARGEMSLR_BUILDTIME_RCM] = MPI_Wtime();
            pargemslr::ParallelLogClass::_times[PARGEMSLR_BUILDTIME_RCM] += pargemslr::ParallelLogClass::_times_buffer_end[PARGEMSLR_BUILDTIME_RCM] - pargemslr::ParallelLogClass::_times_buffer_start[PARGEMSLR_BUILDTIME_RCM];
            
            if(this->_print_option > 1)
            {
               this->_matrix->PlotPatternGnuPlot("A_amd.data",this->_row_perm_vec.GetData(),this->_row_perm_vec.GetData(), 0, 0);
            }
            break;
         }
         case kIluReorderingDdpq:
         {
            PARGEMSLR_ERROR("Ddpq ordering not supported yet.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
            break;
         }
         case kIluReorderingInput:
         {
            /* use the user input nB and permutation array */
            break;
         }
         default:
         {
            /* Invalid option */
            PARGEMSLR_ERROR("Invalid ILU permutation option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_ilu_csr_seq_float::SetupPermutation( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SetupPermutation( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SetupPermutation( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SetupPermutation( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename T>
   int IluClass<MatrixType, VectorType, DataType>::ApplyShift( T &w, int nz, T norm)
   {
      /* for real problem */
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::ApplyShift( float &w, int nz, float norm);
   template int precond_ilu_csr_seq_double::ApplyShift( double &w, int nz, double norm);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename T>
   int IluClass<MatrixType, VectorType, DataType>::ApplyShift( ComplexValueClass<T> &w, int nz, T norm)
   {
      if(this->_complex_shift)
      {
         T shf;
         /* |rho +  I *(del + eta) | > average |off-diagonal entry|  */
         /*-------------------- 
            del^2 + 2 del * eta - theta^2 > 0 when del > eta and signs OK */
         shf = (norm - T(2.0)*PargemslrAbs(w))/(T)nz;
         /*    shf = tnorm/(double)nzcount; */
         shf = w.Imag() <=0 ? -shf : shf;
         w += ComplexValueClass<T>( T(), shf);
      }
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_complexs::ApplyShift( ComplexValueClass<float> &w, int nz, float norm);
   template int precond_ilu_csr_seq_complexd::ApplyShift( ComplexValueClass<double> &w, int nz, double norm);
   
   template <class MatrixType, class VectorType, typename DataType>
   template <typename T>
   int IluClass<MatrixType, VectorType, DataType>::Qsplit(T *a, int *ind, int n, int Ncut)
   {
      /*----------------------------------------------------------------------
      |     does a quick-sort split of a real array.
      |     on input a[0 : (n-1)] is a real array
      |     on output is permuted such that its elements satisfy:
      |
      |     PargemslrAbs(a[i]) >= PargemslrAbs(a[Ncut-1]) for i < Ncut-1 and
      |     PargemslrAbs(a[i]) <= PargemslrAbs(a[Ncut-1]) for i > Ncut-1
      |
      |     ind[0 : (n-1)] is an integer array permuted in the same way as a.
      |---------------------------------------------------------------------*/
   
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
   
      RealDataType   tmp, PargemslrAbskey;
      int            j, itmp, first, mid, last, ncut;
      ncut  = Ncut - 1;
      
      first = 0;
      last  = n-1;
      if (ncut<first || ncut>last) 
      {
         return PARGEMSLR_SUCCESS;
      }
      /* outer loop -- while mid != ncut */
      do
      {
         mid = first;
         PargemslrAbskey = PargemslrAbs(a[mid]);
         for (j=first+1; j<=last; j++) 
         {
            if (PargemslrAbs(a[j]) > PargemslrAbskey) 
            {
               mid = mid+1;
               tmp = a[mid];
               itmp = ind[mid];
               a[mid] = a[j];
               ind[mid] = ind[j];
               a[j]  = tmp;
               ind[j] = itmp;
            }
         }
   /*-------------------- interchange */
         tmp = a[mid];
         a[mid] = a[first];
         a[first]  = tmp;
         itmp = ind[mid];
         ind[mid] = ind[first];
         ind[first] = itmp;
   /*-------------------- test for while loop */
         if (mid == ncut) 
         {  
            break;
         }
         if (mid > ncut) 
         {
            last = mid-1;
         }
         else
         {
            first = mid+1;
         }
      } 
      while(mid != ncut);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::Qsplit(float *a, int *ind, int n, int Ncut);
   template int precond_ilu_csr_seq_double::Qsplit(double *a, int *ind, int n, int Ncut);
   template int precond_ilu_csr_seq_complexs::Qsplit(float *a, int *ind, int n, int Ncut);
   template int precond_ilu_csr_seq_complexd::Qsplit(double *a, int *ind, int n, int Ncut);

#ifdef PARGEMSLR_OPENMP
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::BuildLevels()
   {
      /* TODO: OpenMP support for the setup phase of level scheduling */
      if(!(this->_ready))
      {
         PARGEMSLR_ERROR("ILU level scheduling without setup.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      if(this->_level_ptr_l.GetLengthLocal() > 0)
      {
         /* alreay built, return */
         return PARGEMSLR_SUCCESS;
      }
      
      if(this->_print_option > 0)
      {
         PARGEMSLR_PRINT("Using level scheduling ILU\n");
      }
      
      int i, j, k1, k2, k, level, max_level;
      vector_int levels;
      vector_int perm, unit_perm;
      levels.Setup(this->_n);
      perm.Setup(this->_n);
      unit_perm.Setup(this->_n);
      unit_perm.UnitPerm();
      
      int      *L_i  = this->_L.GetI();
      int      *L_j  = this->_L.GetJ();
      int      *U_i  = this->_U.GetI();
      int      *U_j  = this->_U.GetJ();
      
      /* L part */
      levels.Fill(-1);
      max_level = 0;
      for( i = 0; i < this->_n; i++ )
      {
         //y[i] = x[i];
         k1 = L_i[i]; 
         k2 = L_i[i+1];
         
         level = -1;
         /* TODO: No OpenMP for this loop, typically k2-k1 should be small, not worth the extra parallel cost */
         for(j = k1 ; j < k2; j++) 
         {
            /* update of i-th entry of x requirex L_j[j]-th entry of x 
             * note that in the solve phase, perm is apply to x, no need to consider in the level scheduling
             */
            level = PargemslrMax(level, levels[L_j[j]]);
         }
         /* now, level is the highest level that this node depends on */
         levels[i] = level + 1;
         max_level = PargemslrMax(max_level, levels[i]);
      }
      
      /* start building the level structure of L */
      max_level++;
      this->_level_ptr_l.Setup(max_level+1, true);
      if(this->_print_option > 0)
      {
         PARGEMSLR_PRINT("L part %d/%d levels\n",max_level,this->_n);
      }
      /* first sum the size on each level */
      for( i = 0; i < this->_n; i++ )
      {
         this->_level_ptr_l[levels[i]+1]++;
      }
      
      /* get the displacement */
      for( i = 0; i < max_level; i++ )
      {
         this->_level_ptr_l[i+1] += this->_level_ptr_l[i];
      }
      
      /* fill levels in */
      this->_levels_l.Setup(this->_n);
      for( i = 0; i < this->_n; i++ )
      {
         j = levels[i];
         this->_levels_l[this->_level_ptr_l[j]++] = i;
      }
      
      /* shift back */
      for( i = max_level; i > 0; i-- )
      {
         this->_level_ptr_l[i] = this->_level_ptr_l[i-1];
      }
      this->_level_ptr_l[0] = 0;
      
      /* create permuted L */
      k = 0;
      for( i = 0; i < max_level; i++ )
      {
         k1 = this->_level_ptr_l[i];
         k2 = this->_level_ptr_l[i+1];
         for(j = k1 ; j < k2 ; j ++)
         {
            perm[k++] = this->_levels_l[j];
         }
      }
      
      this->_L.SubMatrix(perm, unit_perm, kMemoryHost, this->_L_level);
      
      /* get the start/end location */
      this->_levels_l_start = max_level;
      for( i = 0; i < max_level; i++ )
      {
         if(this->_level_ptr_l[i+1] - this->_level_ptr_l[i] > pargemslr_global::_openmp_min_loopsize)
         {
            this->_levels_l_start = i;
            break;
         }
      }
      
      this->_levels_l_end = max_level;
      for( i = max_level-1; i >= 0; i-- )
      {
         if(this->_level_ptr_l[i+1] - this->_level_ptr_l[i] > pargemslr_global::_openmp_min_loopsize)
         {
            this->_levels_l_end = i;
            break;
         }
      }
      
      /* U part */
      levels.Fill(-1);
      max_level = 0;
      for( i = this->_n-1; i >= 0; i-- ) 
      {
         k1 = U_i[i]; 
         k2 = U_i[i+1];
         
         level = -1;
         /* TODO: No OpenMP for this loop, typically k2-k1 should be small, not worth the extra parallel cost */
         for( j = k1 ; j < k2; j++) 
         {
            /* update of i-th entry of x requirex U_j[j]-th entry of x
             * note that in the solve phase, perm is apply to x, no need to consider in the level scheduling
             */
            level = PargemslrMax(level, levels[U_j[j]]);
         }
         /* now, level is the highest level that this node depends on */
         levels[i] = level + 1; 
         max_level = PargemslrMax(max_level, levels[i]);   
      }
      
      /* start building the level structure of U */
      max_level++;
      this->_level_ptr_u.Setup(max_level+1, true);
      if(this->_print_option > 0)
      {
         PARGEMSLR_PRINT("U part %d/%d levels\n",max_level,this->_n);
      }
      /* first sum the size on each level */
      for( i = 0; i < this->_n; i++ )
      {
         this->_level_ptr_u[levels[i]+1]++;
      }
      
      /* get the displacement */
      for( i = 0; i < max_level; i++ )
      {
         this->_level_ptr_u[i+1] += this->_level_ptr_u[i];
      }
      
      /* fill levels in */
      this->_levels_u.Setup(this->_n);
      for( i = 0; i < this->_n; i++ )
      {
         j = levels[i];
         this->_levels_u[this->_level_ptr_u[j]++] = i;
      }
      
      /* shift back */
      for( i = max_level; i > 0; i-- )
      {
         this->_level_ptr_u[i] = this->_level_ptr_u[i-1];
      }
      this->_level_ptr_u[0] = 0;
      
      /* create permuted U */
      k = 0;
      for( i = 0; i < max_level; i++ )
      {
         k1 = this->_level_ptr_u[i];
         k2 = this->_level_ptr_u[i+1];
         for(j = k1 ; j < k2 ; j ++)
         {
            perm[k++] = this->_levels_u[j];
         }
      }
      
      this->_U.SubMatrix(perm, unit_perm, kMemoryHost, this->_U_level);
      
      this->_D_level.Setup(this->_n);
      perm.GatherPerm(this->_D, this->_D_level);
      
      /* get the start/end location */
      this->_levels_u_start = max_level;
      for( i = 0; i < max_level; i++ )
      {
         if(this->_level_ptr_u[i+1] - this->_level_ptr_u[i] > pargemslr_global::_openmp_min_loopsize)
         {
            this->_levels_u_start = i;
            break;
         }
      }
      
      this->_levels_u_end = max_level;
      for( i = max_level-1; i >= 0; i-- )
      {
         if(this->_level_ptr_u[i+1] - this->_level_ptr_u[i] > pargemslr_global::_openmp_min_loopsize)
         {
            this->_levels_u_end = i;
            break;
         }
      }
      
      /* deallocate */
      levels.Clear();
      perm.Clear();
      unit_perm.Clear();
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::BuildLevels();
   template int precond_ilu_csr_seq_double::BuildLevels();
   template int precond_ilu_csr_seq_complexs::BuildLevels();
   template int precond_ilu_csr_seq_complexd::BuildLevels();
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::BuildPoly()
   {
      /* TODO: OpenMP support for the setup phase of level scheduling */
      if(!(this->_ready))
      {
         PARGEMSLR_ERROR("ILU level scheduling without setup.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      if(this->_D_poly.GetNumNonzeros() > 0)
      {
         /* alreay built, return */
         return PARGEMSLR_SUCCESS;
      }
      
      /* A \approx L*D*U where L and U has 1s on their diagonal
       * As a result, for example, L = I - L1 where L1 only has pure-lower triangular part
       * L^{-1} = (I - L1)^{-1} => I + L1 + L1^2 + L1^3 + ...
       * we need L1 and U1
       */
      
      /* define the data type */
      typedef DataType T;
      if(this->_print_option > 0)
      {
         PARGEMSLR_PRINT("Using polynomia ILU solve\n");
      }
      
      int   i, j1, j2, j, Lptr, Uptr, nnzL, nnzU;
      
      int      *L_i  = this->_L.GetI();
      int      *L_j  = this->_L.GetJ();
      T        *L_a  = this->_L.GetData();
      int      *U_i  = this->_U.GetI();
      int      *U_j  = this->_U.GetJ();
      T        *U_a  = this->_U.GetData();
      T        *D_a  = this->_D.GetData();
      
      nnzL = this->_L.GetNumNonzeros();
      nnzU = this->_U.GetNumNonzeros();
      this->_L_poly.Setup(this->_n, this->_n, nnzL, true);
      this->_D_poly.Setup(this->_n, this->_n, this->_n, true);
      this->_U_poly.Setup(this->_n, this->_n, nnzU, true);
      
      int      *L_poly_i  = this->_L_poly.GetI();
      int      *D_poly_i  = this->_D_poly.GetI();
      int      *U_poly_i  = this->_U_poly.GetI();
      
      Lptr = 0;
      Uptr = 0;
      for(i = 0 ; i < this->_n ; i ++)
      {
         L_poly_i[i] = Lptr;
         j1 = L_i[i];
         j2 = L_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            this->_L_poly.PushBack(L_j[j], -L_a[j]);
         }
         Lptr += (j2-j1);
         
         D_poly_i[i+1] = i+1;
         this->_D_poly.PushBack(i, D_a[i]);
         
         U_poly_i[i] = Uptr;
         j1 = U_i[i];
         j2 = U_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            this->_U_poly.PushBack(U_j[j], - (D_a[i] * U_a[j]));
         }
         Uptr += (j2-j1);
      }
      L_poly_i[this->_n] = Lptr;
      U_poly_i[this->_n] = Uptr;
      
      /* setup the solve buffer */
      if(this->_y_temp.GetLengthLocal() != this->_n)
      {
         this->_y_temp.Setup( this->_n, kMemoryHost, true);
         this->_z_temp.Setup( this->_n, kMemoryHost, true);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::BuildPoly();
   template int precond_ilu_csr_seq_double::BuildPoly();
   template int precond_ilu_csr_seq_complexs::BuildPoly();
   template int precond_ilu_csr_seq_complexd::BuildPoly();
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveHostOmpPoly( VectorType &x, VectorType &rhs, int option)
   {
      
      /* define the data type */
      typedef DataType T;
      
      int n, i;
      T one, zero;
      T *rhs_a, *x_a, *z_temp;
      
      n = this->_n;
      rhs_a = rhs.GetData();
      x_a = x.GetData();
      //y_temp = this->_y_temp.GetData();
      z_temp = this->_z_temp.GetData();
      
      one = T(1.0);
      zero = T(0.0);
      
      
      /* now apply the solve, (I - M)^{-1} \approx (I + M + M^2 + M^3 + ...)
       * for example, I + M + M^2 + M^3 = rhs + M*rhs + M^2*rhs + M^3*rhs
       */
      
      switch(option)
      {
         case 0:
         {
            CsrMatrixClass<T> &matrix = this->_L_poly;
            /* step 1: copy rhs to z */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
            for(i = 0 ; i < n ; i ++)
            {
               z_temp[i] = rhs_a[i];
            }
            
            if(this->_poly_order > 0)
            {
               /* y_temp = M * rhs */
               matrix.MatVec( 'N', one, rhs, zero, this->_y_temp);
               /* sum to z */
               this->_z_temp.Axpy( one, this->_y_temp);
               for(i = 1 ; i < this->_poly_order ; i ++)
               {
                  if(i%2)
                  {
                     /* 1 3 5 7 ... */
                     /* y_temp = M * y_temp */
                     matrix.MatVec( 'N', one, this->_y_temp, zero, x);
                     /* sum to x */
                     this->_z_temp.Axpy( one, x);
                  }
                  else
                  {
                     /* 2 4 6 8 ... */
                     /* y_temp = M * y_temp */
                     matrix.MatVec( 'N', one, x, zero, this->_y_temp);
                     /* sum to x */
                     this->_z_temp.Axpy( one, this->_y_temp);
                  }
               }
            }
            
            /* apply D */
            this->_D_poly.MatVec( 'N', one, this->_z_temp, zero, x);
            
            break;
         }
         case 1:
         {
            CsrMatrixClass<T> &matrix = this->_U_poly;
            /* step 1: copy rhs to x */

#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
            for(i = 0 ; i < n ; i ++)
            {
               x_a[i] = rhs_a[i];
            }
      
            if(this->_poly_order > 0)
            {
               /* y_temp = M * rhs */
               matrix.MatVec( 'N', one, rhs, zero, this->_y_temp);
               /* sum to x */
               x.Axpy( one, this->_y_temp);
               for(i = 1 ; i < this->_poly_order ; i ++)
               {
                  if(i%2)
                  {
                     /* 1 3 5 7 ... */
                     /* y_temp = M * y_temp */
                     matrix.MatVec( 'N', one, this->_y_temp, zero, this->_z_temp);
                     /* sum to x */
                     x.Axpy( one, this->_z_temp);
                  }
                  else
                  {
                     /* 2 4 6 8 ... */
                     /* y_temp = M * y_temp */
                     matrix.MatVec( 'N', one, this->_z_temp, zero, this->_y_temp);
                     /* sum to x */
                     x.Axpy( one, this->_y_temp);
                  }
               }
            }
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Invalid ILU polynomial solve option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SolveHostOmpPoly( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs, int option);
   template int precond_ilu_csr_seq_double::SolveHostOmpPoly( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs, int option);
   template int precond_ilu_csr_seq_complexs::SolveHostOmpPoly( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs, int option);
   template int precond_ilu_csr_seq_complexd::SolveHostOmpPoly( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs, int option);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveHostOmp( VectorType &x, VectorType &rhs)
   {
      
      /* the solve phase of ilut */
      int   i, j, k1, k2;
      
      /* define the data type */
      typedef DataType T;
      
      switch(this->_omp_option)
      {
         case kIluOpenMPLevelScheduling:
         {
            /* the level scheduling solve */
            
            /* build level when necessary */
            if(this->_level_ptr_l.GetLengthLocal() == 0)
            {
               this->BuildLevels();
            }
            
            int ii, level, l1, l2, nlevels;
            
            T        *x_a  = x.GetData();
            T        *rhs_a  = rhs.GetData();
            T        *D    = this->_D_level.GetData();
            int      *L_i  = this->_L_level.GetI();
            int      *L_j  = this->_L_level.GetJ();
            T        *L_a  = this->_L_level.GetData();
            int      *U_i  = this->_U_level.GetI();
            int      *U_j  = this->_U_level.GetJ();
            T        *U_a  = this->_U_level.GetData();
            
            if(this->_row_perm_vec.GetLengthLocal() > 0)
            {
               this->_row_perm_vec.GatherPerm(rhs, this->_x_temp);
               x_a = this->_x_temp.GetData();
               /* L solve */
               nlevels = this->_level_ptr_l.GetLengthLocal()-1;
               for(level = 0 ; level < this->_levels_l_start ; level++)
               {
                  l1 = this->_level_ptr_l[level];
                  l2 = this->_level_ptr_l[level+1];
                  for(ii = l1 ; ii < l2 ; ii ++)
                  {
                     i = this->_levels_l[ii];
                     k1 = L_i[ii]; 
                     k2 = L_i[ii+1];
                     for(j = k1 ; j < k2; j++) 
                     {
                        x_a[i] -= L_a[j] * x_a[L_j[j]];
                     }
                  }
               }
               if(this->_levels_l_end > this->_levels_l_start)
               {
#pragma omp parallel private(i, j, k1, k2, ii, level, l1, l2)
                  {
                     for(level = _levels_l_start ; level < _levels_l_end ; level++)
                     {
                        l1 = this->_level_ptr_l[level];
                        l2 = this->_level_ptr_l[level+1];
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
                        for(ii = l1 ; ii < l2 ; ii ++)
                        {
                           i = this->_levels_l[ii];
                           k1 = L_i[ii]; 
                           k2 = L_i[ii+1];
                           for(j = k1 ; j < k2; j++) 
                           {
                              //x_a[qperm[i]] -= L_a[j] * x_a[qperm[L_j[j]]];
                              x_a[i] -= L_a[j] * x_a[L_j[j]];
                           }
                        }
                     }
                  }/* end of omp */
               }
               for(level = _levels_l_end ; level < nlevels ; level++)
               {
                  l1 = this->_level_ptr_l[level];
                  l2 = this->_level_ptr_l[level+1];
                  for(ii = l1 ; ii < l2 ; ii ++)
                  {
                     i = this->_levels_l[ii];
                     k1 = L_i[ii]; 
                     k2 = L_i[ii+1];
                     for(j = k1 ; j < k2; j++) 
                     {
                        x_a[i] -= L_a[j] * x_a[L_j[j]];
                     }
                  }
               }
                 
               /* U solve */ 
               nlevels = this->_level_ptr_u.GetLengthLocal()-1;
               for(level = 0 ; level < this->_levels_u_start ; level++)
               {
                  l1 = this->_level_ptr_u[level];
                  l2 = this->_level_ptr_u[level+1];
                  for(ii = l1 ; ii < l2 ; ii ++)
                  {
                     i = this->_levels_u[ii];
                     k1 = U_i[ii]; 
                     k2 = U_i[ii+1];
                     for( j = k1 ; j < k2; j++) 
                     {
                        //x_a[qperm[i]] -= U_a[j] * x_a[qperm[U_j[j]]];
                        x_a[i] -= U_a[j] * x_a[U_j[j]];
                     }
                     /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
                     //x_a[qperm[i]] *= D[i];
                     x_a[i] *= D[ii];
                  }
               }
               if(this->_levels_u_end > this->_levels_u_start)
               {
#pragma omp parallel private(i, j, k1, k2, ii, level, l1, l2)
                  {
                     for(level = _levels_u_start ; level < _levels_u_end ; level++)
                     {
                        l1 = this->_level_ptr_u[level];
                        l2 = this->_level_ptr_u[level+1];
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
                        for(ii = l1 ; ii < l2 ; ii ++)
                        {
                           i = this->_levels_u[ii];
                           k1 = U_i[ii]; 
                           k2 = U_i[ii+1];
                           for( j = k1 ; j < k2; j++) 
                           {
                              //x_a[qperm[i]] -= U_a[j] * x_a[qperm[U_j[j]]];
                              x_a[i] -= U_a[j] * x_a[U_j[j]];
                           }
                           /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
                           //x_a[qperm[i]] *= D[i];
                           x_a[i] *= D[ii];
                        }
                     }
                  }/* end of omp */
               }
               for(level = _levels_u_end ; level < nlevels ; level++)
               {
                  l1 = this->_level_ptr_u[level];
                  l2 = this->_level_ptr_u[level+1];
                  for(ii = l1 ; ii < l2 ; ii ++)
                  {
                     i = this->_levels_u[ii];
                     k1 = U_i[ii]; 
                     k2 = U_i[ii+1];
                     for( j = k1 ; j < k2; j++) 
                     {
                        //x_a[qperm[i]] -= U_a[j] * x_a[qperm[U_j[j]]];
                        x_a[i] -= U_a[j] * x_a[U_j[j]];
                     }
                     /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
                     //x_a[qperm[i]] *= D[i];
                     x_a[i] *= D[ii];
                  }
               }
               
               if(this->_col_perm_vec.GetLengthLocal() > 0)
               {
                  this->_col_perm_vec.ScatterRperm(this->_x_temp, x);
               }
               else
               {
                  this->_row_perm_vec.ScatterRperm(this->_x_temp, x);
               }
            }
            else
            {
               /* L solve */
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
               for( i = 0; i < this->_n; i++ )
               {
                  x_a[i] = rhs_a[i];
               }
               
               nlevels = this->_level_ptr_l.GetLengthLocal()-1;
               for(level = 0 ; level < this->_levels_l_start ; level++)
               {
                  l1 = this->_level_ptr_l[level];
                  l2 = this->_level_ptr_l[level+1];
                  for(ii = l1 ; ii < l2 ; ii ++)
                  {
                     i = this->_levels_l[ii];
                     k1 = L_i[ii]; 
                     k2 = L_i[ii+1];
                     for(j = k1 ; j < k2; j++) 
                     {
                        x_a[i] -= L_a[j] * x_a[L_j[j]];
                     }
                  }
               }
               if(this->_levels_l_end > this->_levels_l_start)
               {
#pragma omp parallel private(i, j, k1, k2, ii, level, l1, l2)
                  {
                     for(level = _levels_l_start ; level < _levels_l_end ; level++)
                     {
                        l1 = this->_level_ptr_l[level];
                        l2 = this->_level_ptr_l[level+1];
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
                        for(ii = l1 ; ii < l2 ; ii ++)
                        {
                           i = this->_levels_l[ii];
                           k1 = L_i[ii]; 
                           k2 = L_i[ii+1];
                           for(j = k1 ; j < k2; j++) 
                           {
                              //x_a[qperm[i]] -= L_a[j] * x_a[qperm[L_j[j]]];
                              x_a[i] -= L_a[j] * x_a[L_j[j]];
                           }
                        }
                     }
                  }/* end of omp */
               }
               for(level = _levels_l_end ; level < nlevels ; level++)
               {
                  l1 = this->_level_ptr_l[level];
                  l2 = this->_level_ptr_l[level+1];
                  for(ii = l1 ; ii < l2 ; ii ++)
                  {
                     i = this->_levels_l[ii];
                     k1 = L_i[ii]; 
                     k2 = L_i[ii+1];
                     for(j = k1 ; j < k2; j++) 
                     {
                        x_a[i] -= L_a[j] * x_a[L_j[j]];
                     }
                  }
               }
                 
               /* U solve */ 
               nlevels = this->_level_ptr_u.GetLengthLocal()-1;
               for(level = 0 ; level < this->_levels_u_start ; level++)
               {
                  l1 = this->_level_ptr_u[level];
                  l2 = this->_level_ptr_u[level+1];
                  for(ii = l1 ; ii < l2 ; ii ++)
                  {
                     i = this->_levels_u[ii];
                     k1 = U_i[ii]; 
                     k2 = U_i[ii+1];
                     for( j = k1 ; j < k2; j++) 
                     {
                        //x_a[qperm[i]] -= U_a[j] * x_a[qperm[U_j[j]]];
                        x_a[i] -= U_a[j] * x_a[U_j[j]];
                     }
                     /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
                     //x_a[qperm[i]] *= D[i];
                     x_a[i] *= D[ii];
                  }
               }
               if(this->_levels_u_end > this->_levels_u_start)
               {
#pragma omp parallel private(i, j, k1, k2, ii, level, l1, l2)
                  {
                     for(level = _levels_u_start ; level < _levels_u_end ; level++)
                     {
                        l1 = this->_level_ptr_u[level];
                        l2 = this->_level_ptr_u[level+1];
#pragma omp barrier
#pragma omp for PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
                        for(ii = l1 ; ii < l2 ; ii ++)
                        {
                           i = this->_levels_u[ii];
                           k1 = U_i[ii]; 
                           k2 = U_i[ii+1];
                           for( j = k1 ; j < k2; j++) 
                           {
                              //x_a[qperm[i]] -= U_a[j] * x_a[qperm[U_j[j]]];
                              x_a[i] -= U_a[j] * x_a[U_j[j]];
                           }
                           /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
                           //x_a[qperm[i]] *= D[i];
                           x_a[i] *= D[ii];
                        }
                     }
                  }/* end of omp */
               }
               for(level = _levels_u_end ; level < nlevels ; level++)
               {
                  l1 = this->_level_ptr_u[level];
                  l2 = this->_level_ptr_u[level+1];
                  for(ii = l1 ; ii < l2 ; ii ++)
                  {
                     i = this->_levels_u[ii];
                     k1 = U_i[ii]; 
                     k2 = U_i[ii+1];
                     for( j = k1 ; j < k2; j++) 
                     {
                        //x_a[qperm[i]] -= U_a[j] * x_a[qperm[U_j[j]]];
                        x_a[i] -= U_a[j] * x_a[U_j[j]];
                     }
                     /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
                     //x_a[qperm[i]] *= D[i];
                     x_a[i] *= D[ii];
                  }
               }
            }
            break;
         }
         case kIluOpenMPPoly:
         {
            /* the poly solve 
             * A \approx L*D*U where L and U has 1s on their diagonal
             * As a result, for example, L = I - L1 where L1 only has pure-lower triangular part
             * L^{-1} = (I - L1)^{-1} => I + L1 + L1^2 + L1^3 + ...
             */
            
            /* build poly when necessary */
            if(this->_D_poly.GetNumNonzeros() == 0)
            {
               this->BuildPoly();
            }
            
            if(this->_row_perm_vec.GetLengthLocal() > 0)
            {
               this->_row_perm_vec.GatherPerm(rhs, this->_x_temp);
               /* L solve and D solve */
               this->SolveHostOmpPoly(x, this->_x_temp, 0);
               
               /* U solve */
               this->SolveHostOmpPoly(this->_x_temp, x, 1);
               
               if(this->_col_perm_vec.GetLengthLocal() > 0)
               {
                  this->_col_perm_vec.ScatterRperm(this->_x_temp, x);
               }
               else
               {
                  this->_row_perm_vec.ScatterRperm(this->_x_temp, x);
               }
            }
            else
            {
               /* L solve and D solve */
               this->SolveHostOmpPoly(this->_x_temp, rhs, 0);
               
               /* U solve */
               this->SolveHostOmpPoly(x, this->_x_temp, 1);
            }
            
            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown ILU openmp option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SolveHostOmp( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveHostOmp( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveHostOmp( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveHostOmp( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
#endif

   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetupILUT( VectorType &x, VectorType &rhs)
   {
      /* TODO: OpenMP setup */
      /* the setup phase of ilut 
       * for modified ILU, we compute LU factorization such that Err*x = rhs if x and rhs are given.
       * Otherwise we do the standard ILU factorization Err*ones = zeros.
       * shift = (sum(H_ijx_j)-rhs_i)/x_i
       */
      
      if(this->_ready)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      if(!this->_matrix)
      {
         PARGEMSLR_ERROR("Call ILU without matrix.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      typedef DataType T;
      
      if(this->_print_option>0)
      {
         PARGEMSLR_PRINT("Start ILUT\n");
         PARGEMSLR_PRINT("Droptol: %e\nMax number of nonzeros each row: %d\n", this->_droptol, this->_max_row_nnz);
      }
      
      /* variables */
      
      int                              n, len, lenu, lenl, init_nnz, nnzL_max, nnzU_max, nnzL_max2, nnzU_max2, nnzL, nnzU;
      int                              *jbuf, *iw, *ja, i, ii, j, j1, j2, k, k1, k2;
      int                              col, jpos, jrow, upos;
      RealDataType                     tnorm, *wn, tolnorm;
      RealDataType                     lfil, tol;
      T                                t, fact, lxu, *D, *w, *ma; 
      T                                diag_shift;
      bool                             standard_milu = true;
      bool                             small_warning = false;
      SequentialVectorClass<DataType>  milu_v;
      
      int                              *pperm, *qperm, *rqperm;
      vector_int                       unitperm;
      int                              *A_i, *A_j, *L_i, *L_j, *U_i, *U_j;
      T                                *A_a, *L_a, *U_a;
      T                                drop_val = this->_diag_shift_milu;
      
      n = this->_matrix->GetNumRowsLocal();
      lfil = this->_max_row_nnz;
      tol = this->_droptol;
      
      PARGEMSLR_CHKERR(n != this->_matrix->GetNumColsLocal())
      
      if(_modified && x.GetLengthLocal() == n && rhs.GetLengthLocal() == n)
      {
         /* use special ILU factorization 
          * we hope to have L*U*x = rhs
          * A*x = (L*U + Err) * x = L*U*x + Err*x
          * Err * x = A*x - rhs
          */
         standard_milu = false;
         milu_v.Setup(n);
         T one, mone;
         one = 1.0;
         mone = -1.0;
         this->_matrix->MatVec( 'N', one, x, mone, rhs, milu_v);
      }
      
      /* get the diagonal shift */
      this->_matrix->GetComplexShift(diag_shift);
      
      /* setup permutation array first */
      if(this->_row_perm_vec.GetLengthLocal() > 0)
      {
         pperm = this->_row_perm_vec.GetData();
         if(this->_col_perm_vec.GetLengthLocal() > 0)
         {
            qperm = this->_col_perm_vec.GetData();
         }
         else
         {
            qperm = pperm;
         }
         PARGEMSLR_MALLOC(rqperm, n, kMemoryHost, int);
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
         for(i = 0 ; i < n ; i++)
         {
            rqperm[qperm[i]] = i;
         }
      }
      else
      {
         unitperm.Setup(n);
         unitperm.UnitPerm();
         pperm = unitperm.GetData();
         qperm = pperm;
         rqperm = pperm;
      }
      
      A_i = this->_matrix->GetI();
      A_j = this->_matrix->GetJ();
      A_a = this->_matrix->GetData();
      
      /* guess initial memory */
      init_nnz = PargemslrMin((int)(n + this->_matrix->GetNumNonzeros() / 2.0), (int)(n * lfil));
      nnzL_max = init_nnz;
      nnzU_max = init_nnz;
      
      /* Create on the host */
      this->_L.Setup(n, n, 0);
      this->_D.Setup(n);
      this->_U.Setup(n, n, 0);
      
      L_i = this->_L.GetI();
      L_i[0] = 0;
      nnzL = 0;
      
      D = this->_D.GetData();
      
      U_i = this->_U.GetI();
      U_i[0] = 0;
      nnzU = 0;
      
      PARGEMSLR_MALLOC(L_j, init_nnz, kMemoryHost, int);
      PARGEMSLR_MALLOC(U_j, init_nnz, kMemoryHost, int);
      PARGEMSLR_MALLOC(L_a, init_nnz, kMemoryHost, T);
      PARGEMSLR_MALLOC(U_a, init_nnz, kMemoryHost, T);
      
      /* working buffers */
      PARGEMSLR_MALLOC(iw, n, kMemoryHost, int);
      PARGEMSLR_MALLOC(jbuf, n, kMemoryHost, int);
      PARGEMSLR_MALLOC(wn, n, kMemoryHost, RealDataType);
      PARGEMSLR_MALLOC(w, n, kMemoryHost, T);

      /* set indicator array jw to -1 */
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
      for( i = 0; i < n; i++ ) 
      {
         iw[i] = -1;
      }

      /* beginning of main loop */
      for( ii = 0; ii < n; ii++ ) 
      {
         i = pperm[ii];
         
         j1 = A_i[i];
         j2 = A_i[i+1];
/*---------- unpack L-part and U-part of column of A in arrays w */
         tnorm = 0.0;
         lenu = 0;
         lenl = 0;
         jbuf[ii] = ii;
         w[ii] = 0.0;
         if(_modified)
         {
            if(standard_milu)
            {
               drop_val = this->_diag_shift_milu;
            }
            else
            {
               drop_val = 0.0;
               //drop_val = this->_diag_shift_milu;
            }
         }
         iw[ii] = ii;
         for( j = j1; j < j2; j++ ) 
         {
            col = rqperm[A_j[j]];
            t = A_a[j];
            tnorm += PargemslrAbs(t);
            if( col < ii ) 
            {
               iw[col] = lenl;
               jbuf[lenl] = col;
               w[lenl] = t;
               lenl++;
            } 
            else if( col == ii ) 
            {
               w[ii] = t+diag_shift;
            } 
            else 
            {
               lenu++;
               jpos = ii + lenu;
               iw[col] = jpos;
               jbuf[jpos] = col;
               w[jpos] = t;
            }
         }
         j = -1;
         len = 0;

         if( tnorm == 0.0 ) 
         {
            PARGEMSLR_WARNING("ILUT zero row encountered.");
            return PARGEMSLR_ERROR_ILU_EMPTY_ROW;
         }
         
         /* apply the complex diagonal shift */
         if(!_modified)
         {
            /* only apply the shift when modified ILU is not used */
            this->ApplyShift(w[ii], j2-j1, tnorm);
         }
         
         tnorm /= (RealDataType) (j2-j1);
         tolnorm = tol * tnorm;
/*---------- eliminate previous rows */
         while( ++j < lenl ) 
         {
/*----------------------------------------------------------------------------
 *  in order to do the elimination in the correct order we must select the
 *  smallest column index among jbuf[k], k = j+1, ..., lenl
 *--------------------------------------------------------------------------*/
            jrow = jbuf[j];
            jpos = j;
/*---------- determine smallest column index */
            for( k = j + 1; k < lenl; k++ ) 
            {
               if( jbuf[k] < jrow ) 
               {
                  jrow = jbuf[k];
                  jpos = k;
               }
            }
            if( jpos != j ) 
            {
   /*---------- swaps */
               col = jbuf[j];
               jbuf[j] = jbuf[jpos];
               jbuf[jpos] = col;
               iw[jrow] = j;
               iw[col]  = jpos;
               t = w[j];
               w[j] = w[jpos];
               w[jpos] = t;
            }
   /*---------- get the multiplier */
            fact = w[j] * D[jrow];
            w[j] = fact;
            /* zero out element in row by resetting iw(n+jrow) to -1 */
            iw[jrow] = -1;
   /*---------- combine current row and row jrow */
            k1 = U_i[jrow];
            k2 = U_i[jrow+1];
            for( k = k1; k < k2; k++ ) 
            {
               col = U_j[k];
               jpos = iw[col];
               lxu = -fact * U_a[k];
   /*---------- if fill-in element is small then disregard */
               if( PargemslrAbs( lxu ) < tolnorm && jpos == -1 ) 
               {  
                  if(_modified)
                  {
                     if(standard_milu)
                     {
                        drop_val += lxu;
                     }
                     else
                     {
                        /* H_ij * x_j */
                        drop_val += lxu * x[col];
                     }
                  }
                  continue;
               }
               if( col < ii ) 
               {
   /*---------- dealing with lower part */
                  if( jpos == -1 ) 
                  {
   /*---------- this is a fill-in element */
                     jbuf[lenl] = col;
                     iw[col] = lenl;
                     w[lenl] = lxu;
                     lenl++;
                  } 
                  else 
                  {
                     w[jpos] += lxu;
                  }
               } 
               else 
               {
   /*---------- dealing with upper part */
                  if( jpos == -1 && PargemslrAbs(lxu) > tolnorm) 
                  {
   /*---------- this is a fill-in element */
                     lenu++;
                     upos = ii + lenu;
                     jbuf[upos] = col;
                     w[upos] = lxu;
                     iw[col] = upos;
                  } 
                  else 
                  {
                     w[jpos] += lxu;
                  }
               }
            }
         }

   /*---------- restore iw */
         iw[ii] = -1;
         for( j = 0; j < lenu; j++ ) 
         {
            iw[jbuf[ii+j+1]] = -1;
         }
   /*---------- case when diagonal is zero */
         //if( PargemslrAbs(w[ii]) < 1e-06) 
         //{
         //   w[ii] = 1e-06;
         //   PARGEMSLR_WARNING("ILUT small diagonal encountered.");
         //}
         //D[ii] = T(1.0) / w[ii]; 
   /*-------------------- update/store row of L-matrix */
   //--------------------    len = min( lenl, lfil );
         if(lenl < lfil) 
         {  
            len = lenl;
         }
         else 
         {  
            len = lfil;
         }
         /*--------------------*/
         for( j = 0; j < lenl; j++ ) 
         {
            wn[j] = PargemslrAbs( w[j] );
            iw[j] = j;
         }
         this->Qsplit( wn, iw, lenl, len );
         nnzL = L_i[ii] + len;
         L_i[ii+1] = nnzL;
         if( nnzL > nnzL_max ) 
         {
            nnzL_max2 = nnzL_max;
            while( nnzL > nnzL_max2 )
            {
               nnzL_max2 = PargemslrMax(nnzL_max+1, (int)(nnzL_max*pargemslr::pargemslr_global::_expand_fact));
            }
            PARGEMSLR_REALLOC(L_j, nnzL_max, nnzL_max2, kMemoryHost, int);
            PARGEMSLR_REALLOC(L_a, nnzL_max, nnzL_max2, kMemoryHost, T);
            nnzL_max = nnzL_max2;
         }
         ja = L_j+L_i[ii];
         ma = L_a+L_i[ii];
         for( j = 0; j < len; j++ ) 
         {
            jpos = iw[j];
            ja[j] = jbuf[jpos];
            ma[j] = w[jpos];
         }
         if(_modified)
         {
            for(j = len ; j < lenl ; j ++)
            {
               jpos = iw[j];
               if(standard_milu)
               {
                  drop_val += w[jpos];
               }
               else
               {
                  drop_val += w[jpos]*x[jpos];
               }
            }
         }
         for( j = 0; j < lenl; j++ ) 
         {
            iw[j] = -1;
         }

/*-------------------- update/store U-matrix */
/*---------- len = min(lfil, lenu) */
         len = (lenu < lfil) ? lenu : lfil;
         for( j = 0; j < lenu; j++ ) 
         {
            wn[j] = PargemslrAbs( w[ii+j+1] );
            iw[j] = ii+j+1;
         }
         this->Qsplit( wn, iw, lenu, len );
         nnzU = U_i[ii] + len;
         U_i[ii+1] = nnzU;
         if( nnzU > nnzU_max ) 
         {
            nnzU_max2 = nnzU_max;
            while( nnzU > nnzU_max2 )
            {
               nnzU_max2 = PargemslrMax(nnzU_max+1, (int)(nnzU_max*pargemslr::pargemslr_global::_expand_fact));
            }
            PARGEMSLR_REALLOC(U_j, nnzU_max, nnzU_max2, kMemoryHost, int);
            PARGEMSLR_REALLOC(U_a, nnzU_max, nnzU_max2, kMemoryHost, T);
            nnzU_max = nnzU_max2;
         }
         ja = U_j+U_i[ii];
         ma = U_a+U_i[ii];
         for( j = 0; j < len; j++ ) 
         {
            jpos = iw[j];
            ja[j] = jbuf[jpos];
            ma[j] = w[jpos];
         }
         if(_modified)
         {
            for(j = len ; j < lenu ; j ++)
            {
               jpos = iw[j];
               if(standard_milu)
               {
                  drop_val += w[jpos];
               }
               else
               {
                  drop_val += w[jpos] * x[jpos];
               }
            }
         }
         for( j = 0; j < lenu; j++ ) 
         {
            iw[j] = -1;
         }
         
         if(_modified)
         {
            if(standard_milu)
            {
               w[ii] += drop_val;
            }
            else
            {
               if( PargemslrAbs(x[ii]) < 1e-06) 
               {
                  /* if x[ii] == 0, might be a good idea not to apply any shift */
                  //w[ii] += (drop_val - rhs[ii] ) / 1e-06; 
               }
               else
               {
                  T milu_vi = milu_v[ii];
                  if(PargemslrReal(milu_vi) < 0.0)
                  {
                     /* do not apply negative shift */
                     w[ii] += (drop_val - milu_v[ii] ) / x[ii];
                  }
                  else
                  {
                     w[ii] += drop_val / x[ii];
                  }
                  //w[ii] += drop_val / x[ii];
               }
            }
         }
         
         if( PargemslrAbs(w[ii]) < 1e-06) 
         {
            if(!small_warning)
            {
               PARGEMSLR_WARNING("ILUT small diagonal encountered.");
               small_warning = true;
            }
            w[ii] = T(1e-06);
         }
         
         D[ii] = T(1.0) / w[ii]; 
         
      }
      
      this->_n = n;
      this->_nnz = n + nnzL + nnzU;
      
      /* update pointers and nnz for L and u */
      this->_L.GetJVector().SetupPtr( L_j, nnzL, kMemoryHost, true);
      this->_L.GetDataVector().SetupPtr( L_a, nnzL, kMemoryHost, true);
      
      this->_L.SetNumNonzeros();
      
      this->_U.GetJVector().SetupPtr( U_j, nnzU, kMemoryHost, true);
      this->_U.GetDataVector().SetupPtr( U_a, nnzU, kMemoryHost, true);
      
      this->_U.SetNumNonzeros();
      
      PARGEMSLR_FREE( w, kMemoryHost);
      PARGEMSLR_FREE( iw, kMemoryHost);
      PARGEMSLR_FREE( jbuf, kMemoryHost);
      PARGEMSLR_FREE( wn, kMemoryHost);
      
      if(this->_row_perm_vec.GetLengthLocal() > 0)
      {
         PARGEMSLR_FREE(rqperm, kMemoryHost);
      }
      unitperm.Clear();
      
      if(_modified && standard_milu)
      {
         milu_v.Clear();
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_ilu_csr_seq_float::SetupILUT( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SetupILUT( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SetupILUT( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SetupILUT( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetupILUK( VectorType &x, VectorType &rhs)
   {
      
      if(this->_ready)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      if(!this->_matrix)
      {
         PARGEMSLR_ERROR("Call ILU without matrix.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }
      
      /* define the data type */
      //typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      typedef DataType T;
      
      if(this->_print_option>0)
      {
         PARGEMSLR_PRINT("Start ILUK\n");
         PARGEMSLR_PRINT("Level of fill: %d\n", this->_fill_level);
      }
      
      /*
       * 1: Setup and create buffers
       * matL/U: the ParCSR matrix for L and U
       * L/U_diag: the diagonal csr matrix of matL/U
       * A_diag_*: tempory p9ointer for the diagonal matrix of A and its '*' slot
       * ii = outer loop from 0 to nLU - 1
       * i = the real col number in diag inside the outer loop
       * iw =  working array store the reverse of active col number
       * iL = working array store the active col number
       */
      int                              i, lfil, n, ii, j, k, k1, k2, kl, ku, jpiv, col, icol;
      int                              *iw;
      MPI_Comm                         comm;
      int                              np,  myid;
      T                                diag_shift, one;
      bool                             standard_milu = true;
      bool                             small_warning = false;
      SequentialVectorClass<DataType>  milu_v;

      /* data objects for A */
      
      int                              *pperm, *qperm, *rqperm;
      vector_int                       unitperm;
      int                              *A_i, *A_j, *L_i, *L_j, *U_i, *U_j;
      T                                *A_a, *L_a = NULL, *U_a = NULL, *D;
      
      T                                drop_val = this->_diag_shift_milu;
      
      n = this->_matrix->GetNumRowsLocal();
      lfil = this->_fill_level;
      one = T(1.0);
      
      if(_modified && x.GetLengthLocal() == n && rhs.GetLengthLocal() == n)
      {
         /* use special ILU factorization 
          * we hope to have L*U*x = rhs
          * A*x = (L*U + Err) * x = L*U*x + Err*x
          * Err * x = A*x - rhs
          */
         standard_milu = false;
         milu_v.Setup(n);
         T one, mone;
         one = 1.0;
         mone = -1.0;
         this->_matrix->MatVec( 'N', one, x, mone, rhs, milu_v);
      }
      
      PARGEMSLR_CHKERR(n != this->_matrix->GetNumColsLocal())
      
      /* get the diagonal shift */
      this->_matrix->GetComplexShift(diag_shift);
      
      /* setup permutation array first */
      if(this->_row_perm_vec.GetLengthLocal() > 0)
      {
         pperm = this->_row_perm_vec.GetData();
         if(this->_col_perm_vec.GetLengthLocal() > 0)
         {
            qperm = this->_col_perm_vec.GetData();
         }
         else
         {
            qperm = pperm;
         }
         PARGEMSLR_MALLOC(rqperm, n, kMemoryHost, int);
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
         for(i = 0 ; i < n ; i++)
         {
            rqperm[qperm[i]] = i;
         }
      }
      else
      {
         unitperm.Setup(n);
         unitperm.UnitPerm();
         pperm = unitperm.GetData();
         qperm = pperm;
         rqperm = pperm;
      }
      
      A_i = this->_matrix->GetI();
      A_j = this->_matrix->GetJ();
      A_a = this->_matrix->GetData();
      
      /* Create on the host */
      this->_L.Setup(n, n, 0);
      this->_D.Setup(n);
      this->_U.Setup(n, n, 0);
      
      L_i = this->_L.GetI();
      L_i[0] = 0;
      
      D = this->_D.GetData();
      
      U_i = this->_U.GetI();
      U_i[0] = 0;
      
      this->_matrix->GetMpiInfo(np, myid, comm);
      
      /*
       * 2: Symbolic factorization
       * setup iw and rperm first
       */
      /* allocate work arrays */
      PARGEMSLR_CALLOC( iw, 3*n, kMemoryHost, int);
      L_i[0] = U_i[0] = 0;
      /* get reverse permutation (rperm).
       * rperm holds the reordered indexes.
       */
      
      /* symbolic factorization */
      this->SetupILUKSymbolic(n, A_i, A_j, lfil, pperm, qperm, rqperm, iw, L_i, U_i, &L_j, &U_j);
      
      /*
       * after this, we have our I,J for L, U and S ready, and L sorted
       * iw are still -1 after symbolic factorization
       * now setup helper array here
       */
      if(L_i[n])
      {
         PARGEMSLR_CALLOC( L_a, L_i[n], kMemoryHost, T);
      }
      if(U_i[n])
      {
         PARGEMSLR_CALLOC( U_a, U_i[n], kMemoryHost, T);
      }
      
      /*
       * 3: Begin real factorization
       * we already have L and U structure ready, so no extra working array needed
       */
      /* first loop for upper part */
      for( ii = 0; ii < n; ii++ )
      {
         // get row i
         i = pperm[ii];
         kl = L_i[ii+1];
         ku = U_i[ii+1];
         k1 = A_i[i];
         k2 = A_i[i+1];
         /* set up working arrays */
         for(j = L_i[ii] ; j < kl ; j ++)
         {
            col = L_j[j];
            iw[col] = j;
         }
         if(_modified)
         {
            if(standard_milu)
            {
               drop_val = this->_diag_shift_milu;
            }
            else
            {
               drop_val = 0.0;
               //drop_val = this->_diag_shift_milu;
            }
         }
         D[ii] = 0.0;
         iw[ii] = ii;
         for(j = U_i[ii] ; j < ku ; j ++)
         {
            col = U_j[j];
            iw[col] = j;
         }
         /* copy data from A into L, D and U */
         for(j = k1 ; j < k2 ; j ++)
         {
            /* compute everything in new index */
            col = rqperm[A_j[j]];
            icol = iw[col];
            /* A for sure to be inside the pattern */
            if(col < ii)
            {
               L_a[icol] = A_a[j];
            }
            else if(col == ii)
            {
               D[ii] = A_a[j];
            }
            else
            {
               U_a[icol] = A_a[j];
            }
         }
         
         /* elimination */
         for(j = L_i[ii] ; j < kl ; j ++)
         {
            jpiv = L_j[j];
            L_a[j] *= D[jpiv];
            ku = U_i[jpiv+1];

            for(k = U_i[jpiv] ; k < ku ; k ++)
            {
               col = U_j[k];
               icol = iw[col];
               if(icol < 0)
               {
                  /* not in partern */
                  if(_modified)
                  {
                     if(standard_milu)
                     {
                        drop_val -= L_a[j]*U_a[k];
                     }
                     else
                     {
                        drop_val -= L_a[j]*U_a[k] * x[col];
                     }
                  }
                  continue;
               }
               if(col < ii)
               {
                  /* L part */
                  L_a[icol] -= L_a[j]*U_a[k];
               }
               else if(col == ii)
               {
                  /* diag part */
                  D[icol] -= L_a[j]*U_a[k];
               }
               else
               {
                  /* U part */
                  U_a[icol] -= L_a[j]*U_a[k];
               }
            }
         }
         
         /* reset working array */
         ku = U_i[ii+1];
         for(j = L_i[ii] ; j < kl ; j ++)
         {
            col = L_j[j];
            iw[col] = -1;
         }
         iw[ii] = -1;
         for(j = U_i[ii] ; j < ku ; j ++)
         {
            col = U_j[j];
            iw[col] = -1;
         }
         
         if(_modified)
         {
            if(standard_milu)
            {
               D[ii] += drop_val;
            }
            else
            {
               if( PargemslrAbs(x[ii]) < 1e-06) 
               {
                  /* if x[ii] == 0, might be a good idea not to apply any shift */
                  //D[ii] += (drop_val - rhs[ii] ) / 1e-06; 
               }
               else
               {
                  T milu_vi = milu_v[ii];
                  if(PargemslrReal(milu_vi) < 0.0)
                  {
                     /* do not apply negative shift */
                     D[ii] += (drop_val - milu_v[ii] ) / x[ii];
                  }
                  else
                  {
                     D[ii] += drop_val / x[ii];
                  }
               }
            }
         }
         
         /* diagonal part (we store the inverse) */
         if(PargemslrAbs(D[ii]) < 1e-06)
         {
            if(!small_warning)
            {
               PARGEMSLR_WARNING("ILUK small diagonal encountered.");
               small_warning = true;
            }
            D[ii] = T(1e-06);
         }
         
         D[ii] = one / D[ii];
         
      }
      
      this->_n = n;
      this->_nnz = n + L_i[n] + U_i[n];
      
      /* update pointers and nnz for L and u */
      this->_L.GetJVector().SetupPtr( L_j, L_i[n], kMemoryHost, true);
      this->_L.GetDataVector().SetupPtr( L_a, L_i[n], kMemoryHost, true);
      
      this->_L.SetNumNonzeros();
      
      this->_U.GetJVector().SetupPtr( U_j, U_i[n], kMemoryHost, true);
      this->_U.GetDataVector().SetupPtr( U_a, U_i[n], kMemoryHost, true);
      
      this->_U.SetNumNonzeros();
      
      PARGEMSLR_FREE( iw, kMemoryHost);
      
      if(this->_row_perm_vec.GetLengthLocal() > 0)
      {
         PARGEMSLR_FREE(rqperm, kMemoryHost);
      }
      unitperm.Clear();
      
      if(_modified && standard_milu)
      {
         milu_v.Clear();
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetupILUK( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SetupILUK( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SetupILUK( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SetupILUK( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetupILUKSymbolic( int n, int *A_i, int *A_j, int lfil, int *perm, int *qperm, int *rqperm, int *iw, int *L_i, int *U_i, int **L_jp, int **U_jp)
   {
      /* define the data type */
      //typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      //typedef DataType T;
      
      int         *temp_L_j, *temp_U_j, *u_levels;
      int         *iL, *iLev;
      int         ii, i, j, k, ku, lena, lenl, lenu, lenh, ilev, lev, col, icol;

      /* memory management */
      int         ctrL, ctrU, capacity_L, capacity_U;
      int         initial_alloc = 0;
      int         nnz_A;

      /* set iL and iLev to right place in iw array */
      iL          = iw + n;
      iLev        = iw + 2*n;

      /* setup initial memory used */
      nnz_A       = A_i[n];
      if(n > 0)
      {
         initial_alloc     = n + ceil((nnz_A / 2.0));
      }
      capacity_L        = initial_alloc;
      capacity_U        = initial_alloc;

      /* allocate other memory for L and U struct */
      PARGEMSLR_CALLOC( temp_L_j, capacity_L, kMemoryHost, int);
      PARGEMSLR_CALLOC( temp_U_j, capacity_U, kMemoryHost, int);
      PARGEMSLR_CALLOC( u_levels, capacity_U, kMemoryHost, int);
      
      ctrL = ctrU = 0;

      /* set initial value for working array */
      for(ii = 0 ; ii < n ; ii ++)
      {
         iw[ii] = -1;
      }

      /*
       * 2: Start of main loop
       * those in iL are NEW col index (after permutation)
       */
      for(ii = 0 ; ii < n ; ii ++)
      {
         i = perm[ii];
         lenl = 0;
         lenh = 0;/* this is the current length of heap */
         lenu = ii;
         lena = A_i[i+1];
         /* put those already inside original pattern, and set their level to 0 */
         for(j = A_i[i] ; j < lena ; j ++)
         {
            /* get the neworder of that col */
            col = rqperm[A_j[j]];
            if(col < ii)
            {
               /*
                * this is an entry in L
                * we maintain a heap structure for L part
                */
               iL[lenh] = col;
               iLev[lenh] = 0;
               iw[col] = lenh++;
               /*now miantian a heap structure*/
               IluClass<MatrixType, VectorType, DataType>::MinHeapAdd_NNR<int, int>(iL,iLev,iw,lenh);
            }
            else if(col > ii)
            {
               /* this is an entry in U */
               iL[lenu] = col;
               iLev[lenu] = 0;
               iw[col] = lenu++;
            }
         }/* end of j loop for adding pattern in original matrix */

         /*
          * search lower part of current row and update pattern based on level
          */
         while(lenh > 0)
         {
            /*
             * k is now the new col index after permutation
             * the first element of the heap is the smallest
             */
            k = iL[0];
            ilev = iLev[0];
            /*
             * we now need to maintain the heap structure
             */
            IluClass<MatrixType, VectorType, DataType>::MinHeapRemovd_NNR<int, int>(iL,iLev,iw,lenh);
            lenh--;
            /* copy to the end of array */
            lenl++;
            /* reset iw for that, not using anymore */
            iw[k]=-1;
            IluClass<MatrixType, VectorType, DataType>::Swap<int, int>(iL,iLev,ii-lenl,lenh);
            
            /*
             * now the elimination on current row could start.
             * eliminate row k (new index) from current row
             */
            ku = U_i[k+1];
            for(j = U_i[k] ; j < ku ; j ++)
            {
               col = temp_U_j[j];
               lev = u_levels[j] + ilev + 1;
               /* ignore large level */
               icol = iw[col];
               /* skill large level */
               if(lev > lfil)
               {
                  continue;
               }
               if(icol < 0)
               {
                  /* not yet in */
                  if(col < ii)
                  {
                     /*
                      * if we add to the left L, we need to maintian the
                      *    heap structure
                      */
                     iL[lenh] = col;
                     iLev[lenh] = lev;
                     iw[col] = lenh++;
                     /*swap it with the element right after the heap*/

                     /* maintain the heap */
                     IluClass<MatrixType, VectorType, DataType>::MinHeapAdd_NNR<int, int>(iL,iLev,iw,lenh);
                  }
                  else if(col > ii)
                  {
                     iL[lenu] = col;
                     iLev[lenu] = lev;
                     iw[col] = lenu++;
                  }
               }
               else
               {
                  iLev[icol] = PargemslrMin(lev, iLev[icol]);
               }
            }/* end of loop j for level update */
         }/* end of while loop for iith row */

         /* now update everything, indices, levels and so */
         L_i[ii+1] = L_i[ii] + lenl;
         if(lenl > 0)
         {
            /* check if memory is enough */
            if(ctrL + lenl > capacity_L)
            {
               int tmp = capacity_L;
               while(ctrL + lenl > capacity_L)
               {
                  capacity_L = capacity_L * pargemslr_global::_expand_fact + 1;
               }
               PARGEMSLR_REALLOC( temp_L_j, tmp, capacity_L, kMemoryHost, int);
            }
            
            /* now copy L data, reverse order */
            for(j = 0 ; j < lenl ; j ++)
            {
               temp_L_j[ctrL+j] = iL[ii-j-1];
            }
            ctrL += lenl;
         }
         k = lenu - ii;
         U_i[ii+1] = U_i[ii] + k;
         if(k > 0)
         {
            /* check if memory is enough */
            if(ctrU + k > capacity_U)
            {
               int tmp = capacity_U;
               while(ctrU + k > capacity_U)
               {
                  capacity_U = capacity_U * pargemslr_global::_expand_fact + 1;
               }
               PARGEMSLR_REALLOC( temp_U_j, tmp, capacity_U, kMemoryHost, int);
               PARGEMSLR_REALLOC( u_levels, tmp, capacity_U, kMemoryHost, int);
            }
            
            PARGEMSLR_MEMCPY( temp_U_j+ctrU, iL+ii, k, kMemoryHost, kMemoryHost, int);
            PARGEMSLR_MEMCPY( u_levels+ctrU, iLev+ii, k, kMemoryHost, kMemoryHost, int);
            
            ctrU += k;
            
         }
         
         /* reset iw */
         for(j = ii ; j < lenu ; j ++)
         {
            iw[iL[j]] = -1;
         }

      }/* end of main loop ii from 0 to nLU-1 */
      
      PARGEMSLR_FREE( u_levels, kMemoryHost);
      
      *L_jp = temp_L_j;
      *U_jp = temp_U_j;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetupILUKSymbolic( int n, int *A_i, int *A_j, int lfil, int *perm, int *qperm, int *rqperm, int *iw, int *L_i, int *U_i, int **L_jp, int **U_jp);
   template int precond_ilu_csr_seq_double::SetupILUKSymbolic( int n, int *A_i, int *A_j, int lfil, int *perm, int *qperm, int *rqperm, int *iw, int *L_i, int *U_i, int **L_jp, int **U_jp);
   template int precond_ilu_csr_seq_complexs::SetupILUKSymbolic( int n, int *A_i, int *A_j, int lfil, int *perm, int *qperm, int *rqperm, int *iw, int *L_i, int *U_i, int **L_jp, int **U_jp);
   template int precond_ilu_csr_seq_complexd::SetupILUKSymbolic( int n, int *A_i, int *A_j, int lfil, int *perm, int *qperm, int *rqperm, int *iw, int *L_i, int *U_i, int **L_jp, int **U_jp);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetupPartialILUT(VectorType &x, VectorType &rhs)
   {
      
      typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
      typedef DataType T;
      
      /*
       * 1: Setup and create buffers
       * matL/U: the ParCSR matrix for L and U
       * L/U_diag: the diagonal csr matrix of matL/U
       * A_*: tempory pointer for the diagonal matrix of A and its '*' slot
       * ii = outer loop from 0 to nLU - 1
       * i = the real col number in diag inside the outer loop
       * iw =  working array store the reverse of active col number
       * iL = working array store the active col number
       */
      int                        n, m, i, ii, j, k1, k2, kl, ku, col, icol, lenl, lenu, lenhu, lenhlr, lenhll, jpos, jrow;
      int                        nLU;
      RealDataType               inorm, itolb, itolef, itols;
      T                          dpiv, lxu;
      vector_int                 iw_vec;
      int                        *iw, *iL;
      SequentialVectorClass<T>   w_vec;
      vector_int                 u_end;
      T                          *w;
      bool                       small_warning = false;

      /* memory management */
      int                        ctrL;
      int                        ctrU, ctrUB;
      int                        initial_alloc = 0;
      int                        capacity_L;
      int                        capacity_U;
      int                        ctrS;
      int                        capacity_S = 0;
      int                        nnz_A;
      
      /* communication stuffs for S */
      MPI_Comm                   comm;
      int                        np, myid;
      
      this->_matrix->GetMpiInfo(np, myid, comm);             
      
      /* data objects for A */
      CsrMatrixClass<T>          &A        = *this->_matrix;
      int                        *A_i      = A.GetI();
      int                        *A_j      = A.GetJ();
      T                          *A_data   = A.GetData();
      
      /* data objects for LU */
      int                        *L_i = NULL;
      int                        *L_j = NULL;
      T                          *L_data = NULL;
      int                        *U_i = NULL;
      int                        *U_j = NULL;
      T                          *U_data = NULL;
      
      /* data objects for LU */
      int                        *S_i = NULL;
      int                        *S_j = NULL;
      T                          *S_data = NULL;
      
      /* reverse permutation */
      vector_int                 temp_perm_vec;
      int                        *rperm;
      int                        *perm;// *qperm;

      /* problem size
       * m is n - nLU, num of rows of local Schur system
       */
       
      n = A.GetNumRowsLocal();
      
      /* setup the local C matrix */
      this->_n = n;
      nLU = this->_nB;
      
      m = n - nLU;
      
      u_end.Setup(nLU);
      
      /* setup initial memory, in ILUT, just guess with max nnz per row */
      nnz_A = A_i[nLU];
      if(n > 0)
      {
         initial_alloc = PargemslrMin(nLU + ceil((nnz_A / 2.0) * nLU / n), (double)nLU * this->_max_row_nnz);
      }
      capacity_L = initial_alloc;
      capacity_U = initial_alloc;
      
      this->_D.Setup(nLU);
      PARGEMSLR_MALLOC( L_i, n+1, kMemoryHost, int);
      PARGEMSLR_MALLOC( U_i, nLU+1, kMemoryHost, int);
      PARGEMSLR_MALLOC( L_j, capacity_L, kMemoryHost, int);
      PARGEMSLR_MALLOC( U_j, capacity_U, kMemoryHost, int);
      PARGEMSLR_MALLOC( L_data, capacity_L, kMemoryHost, T);
      PARGEMSLR_MALLOC( U_data, capacity_U, kMemoryHost, T);
      
      ctrL = ctrU = ctrUB = 0;

      ctrS = 0;
      PARGEMSLR_MALLOC( S_i, m+1, kMemoryHost, int);
      S_i[0] = 0;
      /* only setup S part when n > nLU */
      if(m > 0)
      {
         capacity_S = PargemslrMin(m + ceil((nnz_A / 2.0) * m / n), (double)m * this->_max_row_nnz_s);
         PARGEMSLR_MALLOC( S_j, capacity_S, kMemoryHost, int);
         PARGEMSLR_MALLOC( S_data, capacity_S, kMemoryHost, T);
      }
      
      /* setting up working array */
      iw_vec.Setup(3*n);
      
      iw = iw_vec.GetData();
      
      iL = iw + n;
      
      w_vec.Setup(n);
      w = w_vec.GetData();
      
      /* fill iw */
      for(i = 0 ; i < n ; i ++)
      {
         iw[i] = -1;
      }
      L_i[0] = U_i[0] = 0;
      
      /* get reverse permutation (rperm).
       * rperm holds the reordered indexes.
       * rperm[old] -> new
       * perm[new]  -> old
       */
      rperm = iw + 2*n;
      
      if(this->_row_perm_vec.GetLengthLocal() == n)
      {
         /* in this case, we have the permutation array */
         perm = this->_row_perm_vec.GetData();
      }
      else
      {
         temp_perm_vec.Setup(n);
         temp_perm_vec.UnitPerm();
         perm = temp_perm_vec.GetData();
      }
      /* non symmetric permutation not supported yet */
      //qperm = this->_col_perm_vec.GetData();
      
      for(i = 0 ; i < n ; i ++)
      {
         rperm[perm[i]] = i;
      }
      
      /*
       * 2: Main loop of elimination
       * maintain two heaps
       * |----->*********<-----|-----*********|
       * |col heap***value heap|value in U****|
       */
      /* main outer loop for upper part */
      for(ii = 0 ; ii < nLU ; ii ++)
      {
         /* get real row with perm */
         i = perm[ii];
         k1 = A_i[i];
         k2 = A_i[i+1];
         kl = ii-1;
         /* reset row norm of ith row */
         inorm = 0.0;
         for(j = k1 ; j < k2 ; j ++)
         {
            inorm += PargemslrAbs(A_data[j]);
         }
         if(inorm == 0.0)
         {
            PARGEMSLR_WARNING("Empty row in partial ILUT.");
         }
         inorm /= (RealDataType)(k2-k1);
         /* set the scaled tol for that row */
         itolb = this->_droptol * inorm;
         itolef = this->_droptol_ef * inorm;

         /* reset displacement */
         lenhll = lenhlr = lenu = 0;
         w[ii] = 0.0;
         iw[ii] = ii;
         /* copy in data from A */
         for(j = k1 ; j < k2 ; j ++)
         {
            /* get now col number */
            col = rperm[A_j[j]];
            if(col < ii)
            {
               /* L part of it */
               iL[lenhll] = col;
               w[lenhll] = A_data[j];
               iw[col] = lenhll++;
               /* add to heap, by col number */
               IluClass<MatrixType, VectorType, DataType>::MinHeapAdd_NNR<T, int>(iL,w,iw,lenhll);
            }
            else if(col == ii)
            {
               w[ii] = A_data[j];
            }
            else
            {
               lenu++;
               jpos = lenu + ii;
               iL[jpos] = col;
               w[jpos] = A_data[j];
               iw[col] = jpos;
            }
         }

         /*
          * main elimination
          * need to maintain 2 heaps for L, one heap for col and one heaps for value
          * maintian an array for U, and do qsplit with quick sort after that
          * while the heap of col is greater than zero
          */
         while(lenhll > 0)
         {

            /* get the next row from top of the heap */
            jrow = iL[0];
            dpiv = w[0] * this->_D[jrow];
            w[0] = dpiv;
            /* now remove it from the top of the heap */
            IluClass<MatrixType, VectorType, DataType>::MinHeapRemovd_NNR<T, int>(iL,w,iw,lenhll);
            lenhll--;
            /*
             * reset the drop part to -1
             * we don't need this iw anymore
             */
            iw[jrow] = -1;
            /* need to keep this one, move to the end of the heap */
            /* no longer need to maintain iw */
            IluClass<MatrixType, VectorType, DataType>::Swap<int, T>(iL, w, lenhll, kl-lenhlr);
            lenhlr++;
            IluClass<MatrixType, VectorType, DataType>::AbsMinHeapAdd_RN<T, int>(w+kl,iL+kl,lenhlr);
            /* loop for elimination */
            ku = U_i[jrow+1];
            for(j = U_i[jrow] ; j < ku ; j ++)
            {
               col = U_j[j];
               icol = iw[col];
               lxu = - dpiv*U_data[j];
               /* we don't want to fill small number to empty place */
               if( icol == -1 && ( (col < nLU && PargemslrAbs(lxu) < itolb) || (col >= nLU && PargemslrAbs(lxu) < itolef) ) )
               {
                  continue;
               }
               if(icol == -1)
               {
                  if(col < ii)
                  {
                     /* L part
                      * not already in L part
                      * put it to the end of heap
                      * might overwrite some small entries, no issue
                      */
                     iL[lenhll] = col;
                     w[lenhll] = lxu;
                     iw[col] = lenhll++;
                     /* add to heap, by col number */
                     IluClass<MatrixType, VectorType, DataType>::MinHeapAdd_NNR<T, int>(iL,w,iw,lenhll);
                  }
                  else if(col == ii)
                  {
                     w[ii] += lxu;
                  }
                  else
                  {
                     /*
                      * not already in U part
                      * put is to the end of heap
                      */
                     lenu++;
                     jpos = lenu + ii;
                     iL[jpos] = col;
                     w[jpos] = lxu;
                     iw[col] = jpos;
                  }
               }
               else
               {
                  w[icol] += lxu;
               }
            }
         }/* while loop for the elimination of current row */

         if(PargemslrAbs(w[ii]) < 1e-06)
         {
            if(!small_warning)
            {
               PARGEMSLR_WARNING("ILUT small diagonal encountered.");
               small_warning = true;
            }
            w[ii]=1e-06;
         }
         this->_D[ii] = T(1.0)/w[ii];
         iw[ii] = -1;

         /*
          * now pick up the largest lfil from L
          * L part is guarantee to be larger than itol
          */

         lenl = lenhlr < this->_max_row_nnz ? lenhlr : this->_max_row_nnz;
         L_i[ii+1] = L_i[ii] + lenl;
         if(lenl > 0)
         {
            /* test if memory is enough */
            if(ctrL + lenl > capacity_L)
            {
               int tmp = capacity_L;
               while(ctrL + lenl > capacity_L)
               {
                  capacity_L = capacity_L * pargemslr_global::_expand_fact + 1;
               }
               PARGEMSLR_REALLOC(L_j, tmp, capacity_L, kMemoryHost, int);
               PARGEMSLR_REALLOC(L_data, tmp, capacity_L, kMemoryHost, T);
            }
            ctrL += lenl;
            /* copy large data in */
            for(j = L_i[ii] ; j < ctrL ; j ++)
            {
               L_j[j] = iL[kl];
               L_data[j] = w[kl];
               IluClass<MatrixType, VectorType, DataType>::AbsMinHeapRemove_RN<T, int>(w+kl,iL+kl,lenhlr);
               lenhlr--;
            }
         }
         /*
          * now reset working array
          * L part already reset when move out of heap, only U part
          */
         ku = lenu+ii;
         for(j = ii + 1 ; j <= ku ; j ++)
         {
            iw[iL[j]] = -1;
         }

         if(lenu < this->_max_row_nnz)
         {
            /* we simply keep all of the data, no need to sort */
            lenhu = lenu;
         }
         else
         {
            /* need to sort the first small(hopefully) part of it */
            lenhu = this->_max_row_nnz;
            /* quick split, only sort the first small part of the array */
            IluClass<MatrixType, VectorType, DataType>::Qsplit<T>(w,iL,ii+1,ii+lenhu,ii+lenu);
         }

         U_i[ii+1] = U_i[ii] + lenhu;
         if(lenhu > 0)
         {
            /* test if memory is enough */
            if(ctrU + lenhu > capacity_U)
            {
               int tmp = capacity_U;
               while(ctrU + lenhu > capacity_U)
               {
                  capacity_U = capacity_U * pargemslr_global::_expand_fact + 1;
               }
               PARGEMSLR_REALLOC(U_j, tmp, capacity_U, kMemoryHost, int);
               PARGEMSLR_REALLOC(U_data, tmp, capacity_U, kMemoryHost, T);
            }
            ctrU += lenhu;
            /* copy large data in */
            for(j = U_i[ii] ; j < ctrU ; j ++)
            {
               jpos = ii+1+j-U_i[ii];
               U_j[j] = iL[jpos];
               U_data[j] = w[jpos];
            }
         }
         /* check and build u_end array */
         if(m > 0)
         {
            vector_int temp_u_vec, temp_order;
            SequentialVectorClass<T> temp_val_vec;
            
            temp_u_vec.SetupPtr( U_j+U_i[ii], U_i[ii+1]-U_i[ii], kMemoryHost);
            temp_val_vec.SetupPtr( U_data+U_i[ii], U_i[ii+1]-U_i[ii], kMemoryHost);
            
            temp_u_vec.Sort(temp_order, true, false);
            temp_u_vec.Perm(temp_order);
            temp_val_vec.Perm(temp_order);
            
            /* serach for nLU, this is the first index in the F part
             * if we can't find it, still return the first index
             */
            temp_u_vec.BinarySearch( nLU, u_end[ii], true);
            
            /* this is the number of elements in the real U part */
            ctrUB += u_end[ii];
            
            u_end[ii] += U_i[ii];
            
            temp_u_vec.Clear();
            temp_val_vec.Clear();
            temp_order.Clear();
            
         }
         else
         {
            /* Everything is in U */
            u_end[ii] = ctrU;
         }
      }/* end of ii loop from 0 to nLU-1 */


      /* now main loop for Schur comlement part */
      for(ii = nLU ; ii < n ; ii ++)
      {
         /* get real row with perm */
         i = perm[ii];
         k1 = A_i[i];
         k2 = A_i[i+1];
         kl = nLU-1;
         /* reset row norm of ith row */
         inorm = 0.0;
         for(j = k1 ; j < k2 ; j ++)
         {
            inorm += PargemslrAbs(A_data[j]);
         }
         if(inorm == 0.0)
         {
            PARGEMSLR_WARNING("Empty row in partial ILUT.");
         }
         inorm /= (RealDataType)(k2-k1);
         /* set the scaled tol for that row */
         itols = this->_droptol_s * inorm;
         itolef = this->_droptol_ef * inorm;

         /* reset displacement */
         lenhll = lenhlr = lenu = 0;
         /* copy in data from A */
         for(j = k1 ; j < k2 ; j ++)
         {
            /* get now col number */
            col = rperm[A_j[j]];
            if(col < nLU)
            {
               /* L part of it */
               iL[lenhll] = col;
               w[lenhll] = A_data[j];
               iw[col] = lenhll++;
               /* add to heap, by col number */
               IluClass<MatrixType, VectorType, DataType>::MinHeapAdd_NNR<T, int>(iL,w,iw,lenhll);
            }
            else if(col == ii)
            {
               /* the diagonla entry of S */
               iL[nLU] = col;
               w[nLU] = A_data[j];
               iw[col] = nLU;
            }
            else
            {
               /* S part of it */
               lenu++;
               jpos = lenu + nLU;
               iL[jpos] = col;
               w[jpos] = A_data[j];
               iw[col] = jpos;
            }
         }

         /*
          * main elimination
          * need to maintain 2 heaps for L, one heap for col and one heaps for value
          * maintian an array for S, and do qsplit with quick sort after that
          * while the heap of col is greater than zero
          */
         while(lenhll > 0)
         {
            /* get the next row from top of the heap */
            jrow = iL[0];
            dpiv = w[0] * this->_D[jrow];
            w[0] = dpiv;
            /* now remove it from the top of the heap */
            IluClass<MatrixType, VectorType, DataType>::MinHeapRemovd_NNR<T, int>(iL,w,iw,lenhll);
            lenhll--;
            /*
             * reset the drop part to -1
             * we don't need this iw anymore
             */
            iw[jrow] = -1;
            /* need to keep this one, move to the end of the heap */
            /* no longer need to maintain iw */
            IluClass<MatrixType, VectorType, DataType>::Swap<int, T>(iL,w,lenhll,kl-lenhlr);
            lenhlr++;
            IluClass<MatrixType, VectorType, DataType>::AbsMinHeapAdd_RN<T, int>(w+kl,iL+kl,lenhlr);
            /* loop for elimination */
            ku = U_i[jrow+1];
            for(j = U_i[jrow] ; j < ku ; j ++)
            {
               col = U_j[j];
               icol = iw[col];
               lxu = - dpiv*U_data[j];
               /* we don't want to fill small number to empty place */
               if(icol == -1 && ( (col < nLU && PargemslrAbs(lxu) < itolef) || ( col >= nLU && PargemslrAbs(lxu) < itols ) ) )
               {
                  continue;
               }
               if(icol == -1)
               {
                  if(col < nLU)
                  {
                     /* L part
                      * not already in L part
                      * put it to the end of heap
                      * might overwrite some small entries, no issue
                      */
                     iL[lenhll] = col;
                     w[lenhll] = lxu;
                     iw[col] = lenhll++;
                     /* add to heap, by col number */
                     IluClass<MatrixType, VectorType, DataType>::MinHeapAdd_NNR<T, int>(iL,w,iw,lenhll);
                  }
                  else if(col == ii)
                  {
                     /* the diagonla entry of S */
                     iL[nLU] = col;
                     w[nLU] = A_data[j];
                     iw[col] = nLU;
                  }
                  else
                  {
                     /*
                      * not already in S part
                      * put is to the end of heap
                      */
                     lenu++;
                     jpos = lenu + nLU;
                     iL[jpos] = col;
                     w[jpos] = lxu;
                     iw[col] = jpos;
                  }
               }
               else
               {
                  w[icol] += lxu;
               }
            }
         }/* while loop for the elimination of current row */

         /*
          * now pick up the largest lfil from L
          * L part is guarantee to be larger than itol
          */
         lenl = lenhlr < this->_max_row_nnz ? lenhlr : this->_max_row_nnz;
         L_i[ii+1] = L_i[ii] + lenl;
         if(lenl > 0)
         {
            /* test if memory is enough */
            if(ctrL + lenl > capacity_L)
            {
               int tmp = capacity_L;
               while(ctrL + lenl > capacity_L)
               {
                  capacity_L = capacity_L * pargemslr_global::_expand_fact + 1;
               }
               PARGEMSLR_REALLOC(L_j, tmp, capacity_L, kMemoryHost, int);
               PARGEMSLR_REALLOC(L_data, tmp, capacity_L, kMemoryHost, T);
            }
            ctrL += lenl;
            /* copy large data in */
            for(j = L_i[ii] ; j < ctrL ; j ++)
            {
               L_j[j] = iL[kl];
               L_data[j] = w[kl];
               IluClass<MatrixType, VectorType, DataType>::AbsMinHeapRemove_RN<T, int>(w+kl,iL+kl,lenhlr);
               lenhlr--;
            }
         }
         /*
          * now reset working array
          * L part already reset when move out of heap, only S part
          */
         ku = lenu+nLU;
         for(j = nLU ; j <= ku ; j ++)
         {
            iw[iL[j]] = -1;
         }

         /* the dropping for S */
         lenhu = lenu < this->_max_row_nnz_s ? lenu : this->_max_row_nnz_s;
         //lenhu = lenu;
         /* quick split, only sort the first small part of the array */
         IluClass<MatrixType, VectorType, DataType>::Qsplit<T>(w,iL,nLU+1,nLU+lenhu,nLU+lenu);
         /* we have diagonal in S anyway */
         /* test if memory is enough */
         if(ctrS + lenhu + 1 > capacity_S)
         {
            int tmp = capacity_S;
            while(ctrS + lenhu + 1 > capacity_S)
            {
               capacity_S = capacity_S * pargemslr_global::_expand_fact + 1;
            }
            PARGEMSLR_REALLOC(S_j, tmp, capacity_S, kMemoryHost, int);
            PARGEMSLR_REALLOC(S_data, tmp, capacity_S, kMemoryHost, T);
         }

         ctrS += (lenhu+1);
         S_i[ii-nLU+1] = ctrS;

         /* copy large data in, diagonal first */
         S_j[S_i[ii-nLU]] = iL[nLU]-nLU;
         S_data[S_i[ii-nLU]] = w[nLU];
         for(j = S_i[ii-nLU] + 1 ; j < ctrS ; j ++)
         {
            jpos = nLU+j-S_i[ii-nLU];
            S_j[j] = iL[jpos]-nLU;
            S_data[j] = w[jpos];
         }
      }/* end of ii loop from nLU to n-1 */

      /*
       * 3: Finishing up and free
       */
      
      /* set the L part first, L is nLU * nLU */
      this->_L.Setup( nLU, nLU, L_i[nLU]);
      
      /* L is continious in memory, copy data inside */
      PARGEMSLR_MEMCPY( this->_L.GetI(), L_i, nLU+1, kMemoryHost, kMemoryHost, int);
      PARGEMSLR_MEMCPY( this->_L.GetJ(), L_j, L_i[nLU], kMemoryHost, kMemoryHost, int);
      PARGEMSLR_MEMCPY( this->_L.GetData(), L_data, L_i[nLU], kMemoryHost, kMemoryHost, T);
      
      /* now the remaining part of L, in E */
      this->_E.Setup( m, nLU, L_i[n]-L_i[nLU]);
      
      PARGEMSLR_MEMCPY( this->_E.GetI(), L_i+nLU, m+1, kMemoryHost, kMemoryHost, int);
      PARGEMSLR_MEMCPY( this->_E.GetJ(), L_j+L_i[nLU], L_i[n]-L_i[nLU], kMemoryHost, kMemoryHost, int);
      PARGEMSLR_MEMCPY( this->_E.GetData(), L_data+L_i[nLU], L_i[n]-L_i[nLU], kMemoryHost, kMemoryHost, T);
      
      for(i = 0 ; i < m + 1 ; i ++)
      {
         this->_E.GetI()[i] -= L_i[nLU];
      }
      
      /* set the U part, note that U is not continious in memory, copy them line by line */
      this->_U.Setup( nLU, nLU, ctrUB);
      this->_F.Setup( nLU, m, ctrU-ctrUB);
      
      this->_U.GetI()[0] = 0;
      this->_F.GetI()[0] = 0;
      
      for(i = 0 ; i < nLU ; i ++)
      {
         this->_U.GetI()[i+1] = this->_U.GetI()[i] + u_end[i] - U_i[i];
         this->_F.GetI()[i+1] = this->_F.GetI()[i] + U_i[i+1] - u_end[i];
         
         PARGEMSLR_MEMCPY( this->_U.GetJ() + this->_U.GetI()[i], U_j + U_i[i], u_end[i] - U_i[i], kMemoryHost, kMemoryHost, int);
         PARGEMSLR_MEMCPY( this->_U.GetData() + this->_U.GetI()[i], U_data + U_i[i], u_end[i] - U_i[i], kMemoryHost, kMemoryHost, T);
         
         PARGEMSLR_MEMCPY( this->_F.GetJ() + this->_F.GetI()[i], U_j + u_end[i], U_i[i+1] - u_end[i], kMemoryHost, kMemoryHost, int);
         PARGEMSLR_MEMCPY( this->_F.GetData() + this->_F.GetI()[i], U_data + u_end[i], U_i[i+1] - u_end[i], kMemoryHost, kMemoryHost, T);
         
         for(j = this->_F.GetI()[i] ; j < this->_F.GetI()[i+1] ; j ++)
         {
            this->_F.GetJ()[j] -= nLU;
         }
      }
      
      /* now S */
      this->_S.Setup(m, m, 0);
      this->_S.GetIVector().SetupPtr( S_i, m+1, kMemoryHost, true);
      this->_S.GetJVector().SetupPtr( S_j, ctrS, kMemoryHost, true);
      this->_S.GetDataVector().SetupPtr( S_data, ctrS, kMemoryHost, true);
      
      this->_S.SetNumNonzeros();
      
      this->_nnz = ctrL + ctrU + ctrS + nLU;
      
      /* ready, free */
      PARGEMSLR_FREE( L_i, kMemoryHost);
      PARGEMSLR_FREE( L_j, kMemoryHost);
      PARGEMSLR_FREE( L_data, kMemoryHost);
      PARGEMSLR_FREE( U_i, kMemoryHost);
      PARGEMSLR_FREE( U_j, kMemoryHost);
      PARGEMSLR_FREE( U_data, kMemoryHost);
      
      /* free working array */
      iw_vec.Clear();
      w_vec.Clear();
      u_end.Clear();
      temp_perm_vec.Clear();
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetupPartialILUT( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SetupPartialILUT( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SetupPartialILUT( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SetupPartialILUT( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   
#ifndef PARGEMSLR_CUDA
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::Solve( VectorType &x, VectorType &rhs)
   {
      /* the solve phase of ilut */
      if(this->_option != kIluOptionILUT && this->_option != kIluOptionILUK)
      {
         /* The solve of Partial ILUT is not supported directly, call SolveL and SolveU instead */
         PARGEMSLR_ERROR("Solve phase of ILU only supports ILUT yet.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      PARGEMSLR_CHKERR(this->_n != x.GetLengthLocal() || this->_n != rhs.GetLengthLocal());
      
      if(!(this->_ready))
      {
         PARGEMSLR_ERROR("Solve without setup.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      if(this->_n == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      return this->SolveHost(x, rhs);
   }
   template int precond_ilu_csr_seq_float::Solve( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::Solve( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::Solve( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::Solve( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveL( VectorType &x, VectorType &rhs)
   {
      /* the solve phase of ilut */
      if(this->_option != kIluOptionPartialILUT)
      {
         /* The solve of Partial ILUT is not supported directly, call SolveL and SolveU instead */
         PARGEMSLR_ERROR("Solve with L only supports Partial ILUT yet.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      PARGEMSLR_CHKERR(this->_nB != x.GetLengthLocal() || this->_nB != rhs.GetLengthLocal());
      
      if(!(this->_ready))
      {
         PARGEMSLR_ERROR("Solve without setup.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      if(this->_nB == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      return this->SolveLHost(x, rhs);
   }
   template int precond_ilu_csr_seq_float::SolveL( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveL( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveL( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveL( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveU( VectorType &x, VectorType &rhs)
   {
      /* the solve phase of ilut */
      if(this->_option != kIluOptionPartialILUT)
      {
         /* The solve of Partial ILUT is not supported directly, call SolveL and SolveU instead */
         PARGEMSLR_ERROR("Solve with U only supports Partial ILUT yet.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      PARGEMSLR_CHKERR(this->_nB != x.GetLengthLocal() || this->_nB != rhs.GetLengthLocal());
      
      if(!(this->_ready))
      {
         PARGEMSLR_ERROR("Solve without setup.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      if(this->_nB == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      return this->SolveUHost(x, rhs);
   }
   template int precond_ilu_csr_seq_float::SolveU( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveU( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveU( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveU( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   
#endif
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveLHost( VectorType &x, VectorType &rhs)
   {
      PARGEMSLR_CHKERR(this->_option != kIluOptionPartialILUT);
      PARGEMSLR_CHKERR(this->_x_temp.GetDataLocation() == kMemoryDevice);
      
      /* the solve phase of ilut */
      int   i, j, k1, k2;
      
      /* define the data type */
      typedef DataType T;
      
      T        *x_a  = x.GetData();
      T        *rhs_a  = rhs.GetData();
      int      *L_i  = this->_L.GetI();
      int      *L_j  = this->_L.GetJ();
      T        *L_a  = this->_L.GetData();
      
      /* no permutation */
      /* L solve */
      
      for( i = 0; i < this->_nB; i++ )
      {
         x_a[i] = rhs_a[i];
      }
      
      for( i = 0; i < this->_nB; i++ )
      {
         //y[i] = x[i];
         k1 = L_i[i]; 
         k2 = L_i[i+1];
         for(j = k1 ; j < k2; j++) 
         {
            x_a[i] -= L_a[j] * x_a[L_j[j]];
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SolveLHost( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveLHost( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveLHost( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveLHost( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);

   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveUHost( VectorType &x, VectorType &rhs)
   {
      PARGEMSLR_CHKERR(this->_option != kIluOptionPartialILUT);
      PARGEMSLR_CHKERR(this->_x_temp.GetDataLocation() == kMemoryDevice);
      
      /* the solve phase of ilut */
      int   i, j, k1, k2;
      
      /* define the data type */
      typedef DataType T;
      
      T        *x_a  = x.GetData();
      T        *rhs_a  = rhs.GetData();
      T        *D    = this->_D.GetData();
      int      *U_i  = this->_U.GetI();
      int      *U_j  = this->_U.GetJ();
      T        *U_a  = this->_U.GetData();
      
      /* no permutation */
      
      for( i = 0; i < this->_nB; i++ )
      {
         x_a[i] = rhs_a[i];
      }
      
      /* U solve */    
      for( i = this->_nB-1; i >= 0; i-- ) 
      {
         k1 = U_i[i]; 
         k2 = U_i[i+1];
         for( j = k1 ; j < k2; j++) 
         {
            x_a[i] -= U_a[j] * x_a[U_j[j]];
         }
         /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
         x_a[i] *= D[i];
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SolveUHost( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveUHost( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveUHost( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveUHost( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);

   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SolveHost( VectorType &x, VectorType &rhs)
   {
      PARGEMSLR_CHKERR(this->_x_temp.GetDataLocation() == kMemoryDevice);
      
#ifdef PARGEMSLR_OPENMP
      int num_threads = PargemslrGetOpenmpMaxNumThreads();
      if(num_threads > 1 && this->_omp_option != kIluOpenMPNo)
      {
         return this->SolveHostOmp(x, rhs);
      }
#endif
      
      /* the solve phase of ilut */
      int   i, j, k1, k2;
      
      /* define the data type */
      typedef DataType T;
      
      T        *x_a  = x.GetData();
      T        *rhs_a  = rhs.GetData();
      T        *D    = this->_D.GetData();
      int      *L_i  = this->_L.GetI();
      int      *L_j  = this->_L.GetJ();
      T        *L_a  = this->_L.GetData();
      int      *U_i  = this->_U.GetI();
      int      *U_j  = this->_U.GetJ();
      T        *U_a  = this->_U.GetData();
      
      if(this->_row_perm_vec.GetLengthLocal() > 0)
      {
         this->_row_perm_vec.GatherPerm(rhs, this->_x_temp);
         x_a = this->_x_temp.GetData();
         /* has permutation */
         /* L solve */
         //for( i = 0; i < this->_n; i++ )
         //{
            //x_a[qperm[i]] = rhs_a[pperm[i]];
            //x_a[i] = rhs_a[pperm[i]];
         //}
         
         for( i = 0; i < this->_n; i++ )
         {
            //y[i] = x[i];
            k1 = L_i[i]; 
            k2 = L_i[i+1];
            for(j = k1 ; j < k2; j++) 
            {
               //x_a[qperm[i]] -= L_a[j] * x_a[qperm[L_j[j]]];
               x_a[i] -= L_a[j] * x_a[L_j[j]];
            }
         }
         /* U solve */    
         for( i = this->_n-1; i >= 0; i-- ) 
         {
            k1 = U_i[i]; 
            k2 = U_i[i+1];
            for( j = k1 ; j < k2; j++) 
            {
               //x_a[qperm[i]] -= U_a[j] * x_a[qperm[U_j[j]]];
               x_a[i] -= U_a[j] * x_a[U_j[j]];
            }
            /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
            //x_a[qperm[i]] *= D[i];
            x_a[i] *= D[i]; 
         }
         
         if(this->_col_perm_vec.GetLengthLocal() > 0)
         {
            this->_col_perm_vec.ScatterRperm(this->_x_temp, x);
         }
         else
         {
            this->_row_perm_vec.ScatterRperm(this->_x_temp, x);
         }
      }
      else
      {
         /* no permutation */
         /* L solve */
         for( i = 0; i < this->_n; i++ )
         {
            x_a[i] = rhs_a[i];
         }
         
         for( i = 0; i < this->_n; i++ )
         {
            //y[i] = x[i];
            k1 = L_i[i]; 
            k2 = L_i[i+1];
            for(j = k1 ; j < k2; j++) 
            {
               x_a[i] -= L_a[j] * x_a[L_j[j]];
            }
         }
         /* U solve */    
         for( i = this->_n-1; i >= 0; i-- ) 
         {
            k1 = U_i[i]; 
            k2 = U_i[i+1];
            for( j = k1 ; j < k2; j++) 
            {
               x_a[i] -= U_a[j] * x_a[U_j[j]];
            }
            /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
            x_a[i] *= D[i];
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SolveHost( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_ilu_csr_seq_double::SolveHost( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_ilu_csr_seq_complexs::SolveHost( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_ilu_csr_seq_complexd::SolveHost( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);

   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetMaxNnzPerRow( int max_row_nnz)
   {
      this->_max_row_nnz = max_row_nnz;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetMaxNnzPerRow( int max_row_nnz);
   template int precond_ilu_csr_seq_double::SetMaxNnzPerRow( int max_row_nnz);
   template int precond_ilu_csr_seq_complexs::SetMaxNnzPerRow( int max_row_nnz);
   template int precond_ilu_csr_seq_complexd::SetMaxNnzPerRow( int max_row_nnz);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetMaxNnzPerRowSPart( int max_row_nnz_s)
   {
      this->_max_row_nnz_s = max_row_nnz_s;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetMaxNnzPerRowSPart( int max_row_nnz_s);
   template int precond_ilu_csr_seq_double::SetMaxNnzPerRowSPart( int max_row_nnz_s);
   template int precond_ilu_csr_seq_complexs::SetMaxNnzPerRowSPart( int max_row_nnz_s);
   template int precond_ilu_csr_seq_complexd::SetMaxNnzPerRowSPart( int max_row_nnz_s);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetNB( int nB)
   {
      if(this->_ready)
      {
         PARGEMSLR_ERROR("Change size of B block after Setup is not allowed.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      this->_nB = nB;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetNB( int nB);
   template int precond_ilu_csr_seq_double::SetNB( int nB);
   template int precond_ilu_csr_seq_complexs::SetNB( int nB);
   template int precond_ilu_csr_seq_complexd::SetNB( int nB);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetOption( int option)
   {
      if(this->_ready)
      {
         PARGEMSLR_ERROR("Change ILU option after Setup is not allowed.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      this->_option = option;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetOption( int option);
   template int precond_ilu_csr_seq_double::SetOption( int option);
   template int precond_ilu_csr_seq_complexs::SetOption( int option);
   template int precond_ilu_csr_seq_complexd::SetOption( int option);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetOpenMPOption( int omp_option)
   {

#ifdef PARGEMSLR_OPENMP
      if(this->_ready)
      {
         if(omp_option != this->_omp_option && PargemslrGetOpenmpGlobalMaxNumThreads() > 1)
         {
            switch(omp_option)
            {
               case kIluOpenMPLevelScheduling:
               {
                  this->BuildLevels();
                  break;
               }
               case kIluOpenMPPoly:
               {
                  this->BuildPoly();
                  break;
               }
               case kIluOpenMPNo: default:
               {
                  /* do nothing */
                  break;
               }
            }
         }
      }
#endif

      this->_omp_option = omp_option;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetOpenMPOption( int omp_option);
   template int precond_ilu_csr_seq_double::SetOpenMPOption( int omp_option);
   template int precond_ilu_csr_seq_complexs::SetOpenMPOption( int omp_option);
   template int precond_ilu_csr_seq_complexd::SetOpenMPOption( int omp_option);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetPolyOrder( int order)
   {
      
      this->_poly_order = order;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetPolyOrder( int order);
   template int precond_ilu_csr_seq_double::SetPolyOrder( int order);
   template int precond_ilu_csr_seq_complexs::SetPolyOrder( int order);
   template int precond_ilu_csr_seq_complexd::SetPolyOrder( int order);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetPermutationOption( int perm_option)
   {
      this->_perm_option = perm_option;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetPermutationOption( int perm_option);
   template int precond_ilu_csr_seq_double::SetPermutationOption( int perm_option);
   template int precond_ilu_csr_seq_complexs::SetPermutationOption( int perm_option);
   template int precond_ilu_csr_seq_complexd::SetPermutationOption( int perm_option);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetModified( bool modified)
   {
      this->_modified = modified;
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetModified( bool modified);
   template int precond_ilu_csr_seq_double::SetModified( bool modified);
   template int precond_ilu_csr_seq_complexs::SetModified( bool modified);
   template int precond_ilu_csr_seq_complexd::SetModified( bool modified);
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::GetSize()
   {
      return this->_n;
   }
   template int precond_ilu_csr_seq_float::GetSize();
   template int precond_ilu_csr_seq_double::GetSize();
   template int precond_ilu_csr_seq_complexs::GetSize();
   template int precond_ilu_csr_seq_complexd::GetSize();
   
   template <class MatrixType, class VectorType, typename DataType>
   long int IluClass<MatrixType, VectorType, DataType>::GetNumNonzeros()
   {
      return (long int)(this->_nnz);
   }
   template long int precond_ilu_csr_seq_float::GetNumNonzeros();
   template long int precond_ilu_csr_seq_double::GetNumNonzeros();
   template long int precond_ilu_csr_seq_complexs::GetNumNonzeros();
   template long int precond_ilu_csr_seq_complexd::GetNumNonzeros();
   
   template <class MatrixType, class VectorType, typename DataType>
   CsrMatrixClass<DataType>& IluClass<MatrixType, VectorType, DataType>::GetL()
   {
      return this->_L;
   }
   template CsrMatrixClass<float>& precond_ilu_csr_seq_float::GetL();
   template CsrMatrixClass<double>& precond_ilu_csr_seq_double::GetL();
   template CsrMatrixClass<complexs>& precond_ilu_csr_seq_complexs::GetL();
   template CsrMatrixClass<complexd>& precond_ilu_csr_seq_complexd::GetL();
   
   template <class MatrixType, class VectorType, typename DataType>
   SequentialVectorClass<DataType>& IluClass<MatrixType, VectorType, DataType>::GetD()
   {
      return this->_D;
   }
   template SequentialVectorClass<float>& precond_ilu_csr_seq_float::GetD();
   template SequentialVectorClass<double>& precond_ilu_csr_seq_double::GetD();
   template SequentialVectorClass<complexs>& precond_ilu_csr_seq_complexs::GetD();
   template SequentialVectorClass<complexd>& precond_ilu_csr_seq_complexd::GetD();
   
   template <class MatrixType, class VectorType, typename DataType>
   CsrMatrixClass<DataType>& IluClass<MatrixType, VectorType, DataType>::GetU()
   {
      return this->_U;
   }
   template CsrMatrixClass<float>& precond_ilu_csr_seq_float::GetU();
   template CsrMatrixClass<double>& precond_ilu_csr_seq_double::GetU();
   template CsrMatrixClass<complexs>& precond_ilu_csr_seq_complexs::GetU();
   template CsrMatrixClass<complexd>& precond_ilu_csr_seq_complexd::GetU();
   
   template <class MatrixType, class VectorType, typename DataType>
   CsrMatrixClass<DataType>& IluClass<MatrixType, VectorType, DataType>::GetF()
   {
      return this->_F;
   }
   template CsrMatrixClass<float>& precond_ilu_csr_seq_float::GetF();
   template CsrMatrixClass<double>& precond_ilu_csr_seq_double::GetF();
   template CsrMatrixClass<complexs>& precond_ilu_csr_seq_complexs::GetF();
   template CsrMatrixClass<complexd>& precond_ilu_csr_seq_complexd::GetF();
   
   template <class MatrixType, class VectorType, typename DataType>
   CsrMatrixClass<DataType>& IluClass<MatrixType, VectorType, DataType>::GetE()
   {
      return this->_E;
   }
   template CsrMatrixClass<float>& precond_ilu_csr_seq_float::GetE();
   template CsrMatrixClass<double>& precond_ilu_csr_seq_double::GetE();
   template CsrMatrixClass<complexs>& precond_ilu_csr_seq_complexs::GetE();
   template CsrMatrixClass<complexd>& precond_ilu_csr_seq_complexd::GetE();
   
   template <class MatrixType, class VectorType, typename DataType>
   CsrMatrixClass<DataType>& IluClass<MatrixType, VectorType, DataType>::GetS()
   {
      return this->_S;
   }
   template CsrMatrixClass<float>& precond_ilu_csr_seq_float::GetS();
   template CsrMatrixClass<double>& precond_ilu_csr_seq_double::GetS();
   template CsrMatrixClass<complexs>& precond_ilu_csr_seq_complexs::GetS();
   template CsrMatrixClass<complexd>& precond_ilu_csr_seq_complexd::GetS();
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::GetNB()
   {
      return this->_nB;
   }
   template int precond_ilu_csr_seq_float::GetNB();
   template int precond_ilu_csr_seq_double::GetNB();
   template int precond_ilu_csr_seq_complexs::GetNB();
   template int precond_ilu_csr_seq_complexd::GetNB();
   
   template <class MatrixType, class VectorType, typename DataType>
   IntVectorClass<int>& IluClass<MatrixType, VectorType, DataType>::GetRowPermutationVector()
   {
      return this->_row_perm_vec;
   }
   template IntVectorClass<int>& precond_ilu_csr_seq_float::GetRowPermutationVector();
   template IntVectorClass<int>& precond_ilu_csr_seq_double::GetRowPermutationVector();
   template IntVectorClass<int>& precond_ilu_csr_seq_complexs::GetRowPermutationVector();
   template IntVectorClass<int>& precond_ilu_csr_seq_complexd::GetRowPermutationVector();
   
   template <class MatrixType, class VectorType, typename DataType>
   IntVectorClass<int>& IluClass<MatrixType, VectorType, DataType>::GetColPermutationVector()
   {
      return this->_col_perm_vec;
   }
   template IntVectorClass<int>& precond_ilu_csr_seq_float::GetColPermutationVector();
   template IntVectorClass<int>& precond_ilu_csr_seq_double::GetColPermutationVector();
   template IntVectorClass<int>& precond_ilu_csr_seq_complexs::GetColPermutationVector();
   template IntVectorClass<int>& precond_ilu_csr_seq_complexd::GetColPermutationVector();
   
   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::SetSolveLocation( const int &location)
   {
      this->_location = location;
      
      if(this->_ready)
      {
         this->MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::SetSolveLocation( const int &location);
   template int precond_ilu_csr_seq_double::SetSolveLocation( const int &location);
   template int precond_ilu_csr_seq_complexs::SetSolveLocation( const int &location);
   template int precond_ilu_csr_seq_complexd::SetSolveLocation( const int &location);

#ifndef PARGEMSLR_CUDA

   template <class MatrixType, class VectorType, typename DataType>
   int IluClass<MatrixType, VectorType, DataType>::MoveData( const int &location)
   {
      return PARGEMSLR_SUCCESS;
   }
   template int precond_ilu_csr_seq_float::MoveData( const int &location);
   template int precond_ilu_csr_seq_double::MoveData( const int &location);
   template int precond_ilu_csr_seq_complexs::MoveData( const int &location);
   template int precond_ilu_csr_seq_complexd::MoveData( const int &location);
   
#endif

}
