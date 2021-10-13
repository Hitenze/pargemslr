
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "matrix.hpp"
#include "csr_matrix.hpp"
#include "coo_matrix.hpp"
#include "matrixops.hpp"

#include <cstring>
#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

namespace pargemslr
{
   
   template <typename T>
   CsrMatrixClass<T>::CsrMatrixClass()
   {
      this->_i_vec.Clear();
      this->_j_vec.Clear();
      this->_a_vec.Clear();
      this->_nrows = 0;
      this->_ncols = 0;
      this->_nnz = 0;
      this->_isrowsorted = false;
      this->_isholdingdata = true;
      this->_iscsr = true;
      this->_diagonal_shift = 0.0;
#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
      _cusparse_mat = NULL;
#endif
#endif
   }
   template CsrMatrixClass<float>::CsrMatrixClass();
   template CsrMatrixClass<double>::CsrMatrixClass();
   template CsrMatrixClass<complexs>::CsrMatrixClass();
   template CsrMatrixClass<complexd>::CsrMatrixClass();
   
   template <typename T>
   CsrMatrixClass<T>::CsrMatrixClass(const CsrMatrixClass<T> &mat) : MatrixClass<T>(mat)
   {
      this->_nrows = mat._nrows;
      this->_ncols = mat._ncols;
      this->_nnz = mat._nnz;
      this->_isrowsorted = mat._isrowsorted;
      this->_isholdingdata = mat._isholdingdata;
      this->_iscsr = mat._iscsr;
      this->_diagonal_shift = mat._diagonal_shift;
      
      this->_i_vec = mat._i_vec;
      this->_j_vec = mat._j_vec;
      this->_a_vec = mat._a_vec;
      
#ifdef PARGEMSLR_CUDA
      int location = this->GetDataLocation();
      if(location == kMemoryDevice || location == kMemoryUnified)
      {
         CsrMatrixCreateCusparseSpMat(*this);
      }
#endif

   }
   template CsrMatrixClass<float>::CsrMatrixClass(const CsrMatrixClass<float> &mat);
   template CsrMatrixClass<double>::CsrMatrixClass(const CsrMatrixClass<double> &mat);
   template CsrMatrixClass<complexs>::CsrMatrixClass(const CsrMatrixClass<complexs> &mat);
   template CsrMatrixClass<complexd>::CsrMatrixClass(const CsrMatrixClass<complexd> &mat);
   
   template <typename T>
   CsrMatrixClass<T>::CsrMatrixClass( CsrMatrixClass<T> &&mat) : MatrixClass<T>(std::move(mat))
   {
      this->_nrows = mat._nrows;
      mat._nrows = 0;
      this->_ncols = mat._ncols;
      mat._ncols = 0;
      this->_nnz = mat._nnz;
      mat._nnz = 0;
      this->_isrowsorted = mat._isrowsorted;
      mat._isrowsorted = false;
      this->_isholdingdata = mat._isholdingdata;
      mat._isholdingdata = false;
      this->_iscsr = mat._iscsr;
      mat._iscsr = true;
      this->_diagonal_shift = mat._diagonal_shift;
      mat._diagonal_shift = 0.0;
      
      this->_i_vec = std::move(mat._i_vec);
      this->_j_vec = std::move(mat._j_vec);
      this->_a_vec = std::move(mat._a_vec);
      
#ifdef PARGEMSLR_CUDA
#if (PARGEMSLR_CUDA_VERSION == 11)
      this->_cusparse_mat = mat._cusparse_mat;
      mat._cusparse_mat = NULL;
#endif
#endif

   }
   template CsrMatrixClass<float>::CsrMatrixClass( CsrMatrixClass<float> &&mat);
   template CsrMatrixClass<double>::CsrMatrixClass( CsrMatrixClass<double> &&mat);
   template CsrMatrixClass<complexs>::CsrMatrixClass( CsrMatrixClass<complexs> &&mat);
   template CsrMatrixClass<complexd>::CsrMatrixClass( CsrMatrixClass<complexd> &&mat);
   
   template <typename T>
   CsrMatrixClass<T>& CsrMatrixClass<T>::operator= (const CsrMatrixClass<T> &mat)
   {
      this->Clear();
      ParallelLogClass::operator=(mat);
      this->_nrows = mat._nrows;
      this->_ncols = mat._ncols;
      this->_nnz = mat._nnz;
      this->_isrowsorted = mat._isrowsorted;
      this->_isholdingdata = mat._isholdingdata;
      this->_iscsr = mat._iscsr;
      this->_diagonal_shift = mat._diagonal_shift;
      
      this->_i_vec = mat._i_vec;
      this->_j_vec = mat._j_vec;
      this->_a_vec = mat._a_vec;
      
#ifdef PARGEMSLR_CUDA
      int location = this->GetDataLocation();
      if(location == kMemoryDevice || location == kMemoryUnified)
      {
         CsrMatrixCreateCusparseSpMat(*this);
      }
#endif
      return *this;

   }
   template CsrMatrixClass<float>& CsrMatrixClass<float>::operator= (const CsrMatrixClass<float> &mat);
   template CsrMatrixClass<double>& CsrMatrixClass<double>::operator= (const CsrMatrixClass<double> &mat);
   template CsrMatrixClass<complexs>& CsrMatrixClass<complexs>::operator= (const CsrMatrixClass<complexs> &mat);
   template CsrMatrixClass<complexd>& CsrMatrixClass<complexd>::operator= (const CsrMatrixClass<complexd> &mat);
   
   template <typename T>
   CsrMatrixClass<T>& CsrMatrixClass<T>::operator= ( CsrMatrixClass<T> &&mat)
   {
      this->Clear();
      ParallelLogClass::operator=(std::move(mat));
      this->_nrows = mat._nrows;
      mat._nrows = 0;
      this->_ncols = mat._ncols;
      mat._ncols = 0;
      this->_nnz = mat._nnz;
      mat._nnz = 0;
      this->_isrowsorted = mat._isrowsorted;
      mat._isrowsorted = false;
      this->_isholdingdata = mat._isholdingdata;
      mat._isholdingdata = false;
      this->_iscsr = mat._iscsr;
      mat._iscsr = true;
      this->_diagonal_shift = mat._diagonal_shift;
      mat._diagonal_shift = 0.0;
      
      this->_i_vec = std::move(mat._i_vec);
      this->_j_vec = std::move(mat._j_vec);
      this->_a_vec = std::move(mat._a_vec);
      
#ifdef PARGEMSLR_CUDA
#if (PARGEMSLR_CUDA_VERSION == 11)
      this->_cusparse_mat = mat._cusparse_mat;
      mat._cusparse_mat = NULL;
#endif
#endif
      return *this;

   }
   template CsrMatrixClass<float>& CsrMatrixClass<float>::operator= ( CsrMatrixClass<float> &&mat);
   template CsrMatrixClass<double>& CsrMatrixClass<double>::operator= ( CsrMatrixClass<double> &&mat);
   template CsrMatrixClass<complexs>& CsrMatrixClass<complexs>::operator= ( CsrMatrixClass<complexs> &&mat);
   template CsrMatrixClass<complexd>& CsrMatrixClass<complexd>::operator= ( CsrMatrixClass<complexd> &&mat);
   
   template <typename T>
   CsrMatrixClass<T>::~CsrMatrixClass()
   {
      this->Clear();
   }
   template CsrMatrixClass<float>::~CsrMatrixClass();
   template CsrMatrixClass<double>::~CsrMatrixClass();
   template CsrMatrixClass<complexs>::~CsrMatrixClass();
   template CsrMatrixClass<complexd>::~CsrMatrixClass();
   
   template <typename T>
   int CsrMatrixClass<T>::Clear()
   {
      /* base class clear */
      MatrixClass<T>::Clear();
      
      this->_i_vec.Clear();
      this->_j_vec.Clear();
      this->_a_vec.Clear();
      this->_nrows = 0;
      this->_ncols = 0;
      this->_nnz = 0;
      this->_isrowsorted = false;
      this->_isholdingdata = true;
      this->_iscsr = true;
      this->_diagonal_shift = 0.0;
#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
      if(this->_cusparse_mat)
      {
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroySpMat(this->_cusparse_mat)) );
         this->_cusparse_mat = NULL;
      }
#endif
#endif
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::Clear();
   template int CsrMatrixClass<double>::Clear();
   template int CsrMatrixClass<complexs>::Clear();
   template int CsrMatrixClass<complexd>::Clear();
   
   template <typename T>
   int CsrMatrixClass<T>::Setup(int nrows, int ncols, int nnz)
   {
      return this->Setup( nrows, ncols, nnz, kMemoryHost, true, false, true);
   }
   template int CsrMatrixClass<float>::Setup(int nrows, int ncols, int nnz);
   template int CsrMatrixClass<double>::Setup(int nrows, int ncols, int nnz);
   template int CsrMatrixClass<complexs>::Setup(int nrows, int ncols, int nnz);
   template int CsrMatrixClass<complexd>::Setup(int nrows, int ncols, int nnz);
   
   template <typename T>
   int CsrMatrixClass<T>::Setup(int nrows, int ncols, int nnz, int location)
   {
      return this->Setup( nrows, ncols, nnz, location, true, false, true);
   }
   template int CsrMatrixClass<float>::Setup(int nrows, int ncols, int nnz, int location);
   template int CsrMatrixClass<double>::Setup(int nrows, int ncols, int nnz, int location);
   template int CsrMatrixClass<complexs>::Setup(int nrows, int ncols, int nnz, int location);
   template int CsrMatrixClass<complexd>::Setup(int nrows, int ncols, int nnz, int location);
   
   template <typename T>
   int CsrMatrixClass<T>::Setup(int nrows, int ncols, int nnz, bool setzero)
   {
      return this->Setup( nrows, ncols, nnz, kMemoryHost, true, setzero, true);
   }
   template int CsrMatrixClass<float>::Setup(int nrows, int ncols, int nnz, bool setzero);
   template int CsrMatrixClass<double>::Setup(int nrows, int ncols, int nnz, bool setzero);
   template int CsrMatrixClass<complexs>::Setup(int nrows, int ncols, int nnz, bool setzero);
   template int CsrMatrixClass<complexd>::Setup(int nrows, int ncols, int nnz, bool setzero);
   
   template <typename T>
   int CsrMatrixClass<T>::Setup(int nrows, int ncols, int nnz, bool holdvalue, bool setzero)
   {
      return this->Setup( nrows, ncols, nnz, kMemoryHost, holdvalue, setzero, true);
   }
   template int CsrMatrixClass<float>::Setup(int nrows, int ncols, int nnz, bool holdvalue, bool setzero);
   template int CsrMatrixClass<double>::Setup(int nrows, int ncols, int nnz, bool holdvalue, bool setzero);
   template int CsrMatrixClass<complexs>::Setup(int nrows, int ncols, int nnz, bool holdvalue, bool setzero);
   template int CsrMatrixClass<complexd>::Setup(int nrows, int ncols, int nnz, bool holdvalue, bool setzero);
   
   template <typename T>
   int CsrMatrixClass<T>::Setup(int nrows, int ncols, int nnz, int location, bool setzero)
   {
      return this->Setup( nrows, ncols, nnz, location, true, setzero, true);
   }
   template int CsrMatrixClass<float>::Setup(int nrows, int ncols, int nnz, int location, bool setzero);
   template int CsrMatrixClass<double>::Setup(int nrows, int ncols, int nnz, int location, bool setzero);
   template int CsrMatrixClass<complexs>::Setup(int nrows, int ncols, int nnz, int location, bool setzero);
   template int CsrMatrixClass<complexd>::Setup(int nrows, int ncols, int nnz, int location, bool setzero);
   
   template <typename T>
   int CsrMatrixClass<T>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero)
   {
      return this->Setup( nrows, ncols, nnz, location, true, setzero, true);
   }
   template int CsrMatrixClass<float>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero);
   template int CsrMatrixClass<double>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero);
   template int CsrMatrixClass<complexs>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero);
   template int CsrMatrixClass<complexd>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero);
   
   template <typename T>
   int CsrMatrixClass<T>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero, bool iscsr)
   {
      PARGEMSLR_CHKERR( nrows < 0 );
      PARGEMSLR_CHKERR( ncols < 0 );
      PARGEMSLR_CHKERR( nnz < 0 );
      
#ifdef PARGEMSLR_CUDA
      if(setzero)
      {
         if(location == kMemoryDevice)
         {
            PARGEMSLR_ERROR("Csr matrix setup setzero option is not availiabe for the device memory.");
            return PARGEMSLR_ERROR_MEMORY_LOCATION;
         }
      }
#endif
      
      this->Clear();
      
      this->_i_vec.Clear();
      this->_j_vec.Clear();
      this->_a_vec.Clear();
      this->_nrows = nrows;
      this->_ncols = ncols;
      this->_nnz = nnz;
      this->_isrowsorted = false;
      this->_isholdingdata = holdvalue;
      this->_iscsr = iscsr;
      
      if(this->_iscsr)
      {
         if(setzero)
         {
            this->_i_vec.Setup(nrows+1, nrows+1, location, false);
            /* host version */
            this->_i_vec[0] = 0;
            this->_j_vec.Setup(0, nnz, location, false);
            if(this->_isholdingdata)
            {
               this->_a_vec.Setup(0, nnz, location, false);
            }
         }
         else
         {
            this->_i_vec.Setup(nrows+1, nrows+1, location, false);
            this->_j_vec.Setup(nnz, nnz, location, false);
            if(this->_isholdingdata)
            {
               this->_a_vec.Setup(nnz, nnz, location, false);
            }
         }
      }
      else
      {
         if(setzero)
         {
            this->_i_vec.Setup(ncols+1, ncols+1, location, false);
            this->_i_vec[0] = 0;
            this->_j_vec.Setup(0, nnz, location, false);
            if(this->_isholdingdata)
            {
               this->_a_vec.Setup(0, nnz, location, false);
            }
         }
         else
         {
            this->_i_vec.Setup(ncols+1, ncols+1, location, false);
            this->_j_vec.Setup(nnz, nnz, location, false);
            if(this->_isholdingdata)
            {
               this->_a_vec.Setup(nnz, nnz, location, false);
            }
         }
      }

#ifdef PARGEMSLR_CUDA
      if(location == kMemoryDevice || location == kMemoryUnified)
      {
         CsrMatrixCreateCusparseSpMat(*this);
      }
#endif

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero, bool iscsr);
   template int CsrMatrixClass<double>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero, bool iscsr);
   template int CsrMatrixClass<complexs>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero, bool iscsr);
   template int CsrMatrixClass<complexd>::Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero, bool iscsr);
   
   template <typename T>
   int CsrMatrixClass<T>::PushBack( int col, T v)
   {
      
      PARGEMSLR_CHKERR( (this->_iscsr && (col < 0 || col >= this->_ncols) ) );
      PARGEMSLR_CHKERR( (!(this->_iscsr) && (col < 0 || col >= this->_nrows) ) );

#ifdef PARGEMSLR_CUDA
      if( this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Csr matrix PushBack function is not availiabe for the device memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
#endif      
      this->_j_vec.PushBack(col);
      this->_a_vec.PushBack(v);
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int CsrMatrixClass<float>::PushBack( int col, float v);
   template int CsrMatrixClass<double>::PushBack( int col, double v);
   template int CsrMatrixClass<complexs>::PushBack( int col, complexs v);
   template int CsrMatrixClass<complexd>::PushBack( int col, complexd v);
   
   template <typename T>
   int CsrMatrixClass<T>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, CsrMatrixClass<T> &csrmat_out)
   {
      
      int err = 0;
      
      vector_int row_starts, col_starts, row_ends, col_ends;
      
      row_starts.Setup(1);
      col_starts.Setup(1);
      row_ends.Setup(1);
      col_ends.Setup(1);
      
      row_starts[0] = row_start;
      col_starts[0] = col_start;
      
      row_ends[0] = row_start+num_rows;
      col_ends[0] = col_start+num_cols;
      
      err = this->SubMatrix( row_starts, col_starts, row_ends, col_ends, location, csrmat_out);
      
      row_starts.Clear();
      col_starts.Clear();
      row_ends.Clear();
      col_ends.Clear();
      
      return err;
   }
   template int CsrMatrixClass<float>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, CsrMatrixClass<float> &csrmat_out);
   template int CsrMatrixClass<double>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, CsrMatrixClass<double> &csrmat_out);
   template int CsrMatrixClass<complexs>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, CsrMatrixClass<complexs> &csrmat_out);
   template int CsrMatrixClass<complexd>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, CsrMatrixClass<complexd> &csrmat_out);
   
   template <typename T>
   int CsrMatrixClass<T>::SubMatrix( vector_int &row_starts, vector_int &col_starts, vector_int &row_ends, vector_int &col_ends, int location, CsrMatrixClass<T> &csrmat_out)
   {
      int           nrA, ncA, nrB, ncB, nnzA, i, j, k, nblocks, rowi, rr, jj, cc, nnz_guess;
      vector_int    marker;
      
      bool iscsr = this->_iscsr;
      
      if(!iscsr)
      {
         /* if is csc, convert to csr */
         this->Convert(true);
      }

#ifdef PARGEMSLR_CUDA
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Extracting csr sub-matrix only availiable on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      if(row_starts.GetDataLocation() == kMemoryDevice || col_starts.GetDataLocation() == kMemoryDevice || row_ends.GetDataLocation() == kMemoryDevice || col_ends.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Working vectors for extracting csr sub-matrix can only be on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
#endif
      
      nblocks  = row_starts.GetLengthLocal();
      
      PARGEMSLR_CHKERR(nblocks != col_starts.GetLengthLocal());
      
      nrB      = 0;
      ncB      = 0;
      
      /* We can use sum here, but doing this could avoid overflow */
      for(i = 0 ; i < nblocks ; i ++)
      {
         nrB += row_ends[i] - row_starts[i];
      }
      for(i = 0 ; i < nblocks ; i ++)
      {
         ncB += col_ends[i] - col_starts[i];
      }
      nrA      = this->GetNumRowsLocal();
      ncA      = this->GetNumColsLocal();
      nnzA     = this->GetNumNonzeros();
      
      PARGEMSLR_CHKERR( nrB > nrA);
      PARGEMSLR_CHKERR( ncB > ncA);
      
      if(nrB == 0 || ncB == 0)
      {
         csrmat_out.Setup(nrB, ncB, 0, location, false);
         return PARGEMSLR_SUCCESS;
      }
      
      /* reserve some space for B */
      nnz_guess   = ( ((nrB+0.0)/nrA) * ((ncB+0.0)/ncA) * nnzA );
      
      if(this->_isholdingdata)
      {
         /* in this case csr matrix */
         csrmat_out.Setup( nrB, ncB, nnz_guess, kMemoryHost, true);
      }
      else
      {
         /* in this case only a csr pattern matrix */
         csrmat_out.Setup( nrB, ncB, nnz_guess, kMemoryHost, false, true);
      }
      
      /* mark cols to extract by B indices */
      marker.Setup(ncA);
      marker.Fill(-1);
      k = 0;
      for( i = 0 ; i < nblocks ; i ++)
      {
         for(j = col_starts[i] ; j < col_ends[i] ; j ++)
         {
            
            PARGEMSLR_CHKERR( j < 0);
            PARGEMSLR_CHKERR( j >= ncA);
            PARGEMSLR_CHKERR( marker[j] != -1);
            
            marker[j] = k++;
         }
      }
      
      /* extract B = A(rows, cols) */
      rowi = 1;
      for(i = 0 ; i < nblocks ; i ++)
      {
         for(rr = row_starts[i] ; rr < row_ends[i] ; rr ++)
         {
            for(j = this->GetI()[rr]; j < this->GetI()[rr+1]; j ++)
            {
               cc = this->GetJ()[j];
               jj = marker[cc];
               if (jj!=-1)
               {
                  csrmat_out.GetJVector().PushBack(jj);
                  if(this->_isholdingdata)
                  {
                     csrmat_out.GetDataVector().PushBack(this->GetData()[j]);
                  }
               }
            }
            csrmat_out.GetI()[rowi++] = csrmat_out.GetJVector().GetLengthLocal();
         }
      }
      
      csrmat_out.SetNumNonzeros();
      csrmat_out.MoveData(location);
      
      marker.Clear();
      
      if(!iscsr)
      {
         this->Convert(false);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::SubMatrix( vector_int &row_starts, vector_int &col_starts, vector_int &row_ends, vector_int &col_ends, int location, CsrMatrixClass<float> &csrmat_out);
   template int CsrMatrixClass<double>::SubMatrix( vector_int &row_starts, vector_int &col_starts, vector_int &row_ends, vector_int &col_ends, int location, CsrMatrixClass<double> &csrmat_out);
   template int CsrMatrixClass<complexs>::SubMatrix( vector_int &row_starts, vector_int &col_starts, vector_int &row_ends, vector_int &col_ends, int location, CsrMatrixClass<complexs> &csrmat_out);
   template int CsrMatrixClass<complexd>::SubMatrix( vector_int &row_starts, vector_int &col_starts, vector_int &row_ends, vector_int &col_ends, int location, CsrMatrixClass<complexd> &csrmat_out);
   
   template <typename T>
   int CsrMatrixClass<T>::SubMatrix( vector_int &rows, vector_int &cols, int location, CsrMatrixClass<T> &csrmat_out)
   {
      
      int           nrB, ncB, nrA, ncA, nnzA, k, j, i, rr, cc, jj, nnz_guess;
      vector_int    marker;
      
      bool iscsr = this->_iscsr;
      
      if(!iscsr)
      {
         /* if is csc, convert to csr */
         this->Convert(true);
      }
      
#ifdef PARGEMSLR_CUDA
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Extracting csr sub-matrix only availiable on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      if(rows.GetDataLocation() == kMemoryDevice || cols.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Working vectors for extracting csr sub-matrix can be on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
#endif
      
      nrB         = rows.GetLengthLocal();
      ncB         = cols.GetLengthLocal();
      nrA         = this->GetNumRowsLocal();
      ncA         = this->GetNumColsLocal();
      nnzA        = this->GetNumNonzeros();
      
      PARGEMSLR_CHKERR( nrB > nrA);
      PARGEMSLR_CHKERR( ncB > ncA);
      
      if(nrB == 0 || ncB == 0)
      {
         csrmat_out.Setup(nrB, ncB, 0, kMemoryDevice, false);
         return PARGEMSLR_SUCCESS;
      }
      
      /* reserve some space for B */
      nnz_guess   = ( ((nrB+0.0)/nrA) * ((ncB+0.0)/ncA) * nnzA );
      
      if(this->_isholdingdata)
      {
         /* in this case csr matrix */
         csrmat_out.Setup( nrB, ncB, nnz_guess, kMemoryHost, true);
      }
      else
      {
         /* in this case only a csr pattern matrix */
         csrmat_out.Setup( nrB, ncB, nnz_guess, kMemoryHost, false, true);
      }
      
      /* helper array marking cols to extract by B indices */
      marker.Setup(ncA);
      marker.Fill(-1);
      k = 0;
      for (i = 0; i < ncB; i++) 
      {
         j = cols[i];
         
         PARGEMSLR_CHKERR( j < 0);
         PARGEMSLR_CHKERR( j >= ncA);
         PARGEMSLR_CHKERR( marker[j] != -1);
         
         marker[j] = k++;
      }
      
      /* extract B = A(rows, cols) */
      
      for ( i = 0; i < nrB; i++) 
      {
         rr = rows[i];
         for (j = this->GetI()[rr]; j < this->GetI()[rr+1]; j++) 
         {
            cc = this->GetJ()[j];
            jj = marker[cc];
            if (jj != -1) 
            {
               csrmat_out.GetJVector().PushBack(jj);
               if(this->_isholdingdata)
               {
                  csrmat_out.GetDataVector().PushBack(this->GetData()[j]);
               }
            }
         }
         csrmat_out.GetI()[i+1] = csrmat_out.GetJVector().GetLengthLocal();
      }
      
      csrmat_out.SetNumNonzeros();
      csrmat_out.MoveData(location);
      
      marker.Clear();
      
      if(!iscsr)
      {
         this->Convert(false);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::SubMatrix( vector_int &rows, vector_int &cols, int location, CsrMatrixClass<float> &csrmat_out);
   template int CsrMatrixClass<double>::SubMatrix( vector_int &rows, vector_int &cols, int location, CsrMatrixClass<double> &csrmat_out);
   template int CsrMatrixClass<complexs>::SubMatrix( vector_int &rows, vector_int &cols, int location, CsrMatrixClass<complexs> &csrmat_out);
   template int CsrMatrixClass<complexd>::SubMatrix( vector_int &rows, vector_int &cols, int location, CsrMatrixClass<complexd> &csrmat_out);
   
   template <typename T>
   int CsrMatrixClass<T>::SubMatrixNoPerm( vector_int &rows, vector_int &cols, vector_int &row_perm, vector_int &col_perm, bool complement, int location, CsrMatrixClass<T> &csrmat_out)
   {
      int i, nr, nc, err = 0;
      
      row_perm.Setup(this->_nrows, kMemoryHost, false);
      col_perm.Setup(this->_nrows, kMemoryHost, false);
      
      nr = 0;
      nc = 0;
      if(complement)
      {
         for(i = 0 ; i < this->_nrows ; i ++)
         {
            if(rows[i] == 0)
            {
               row_perm[nr] = i;
               nr++;
            }
         }
         for(i = 0 ; i < this->_ncols ; i ++)
         {
            if(cols[i] == 0)
            {
               col_perm[nc] = i;
               nc++;
            }
         }
      }
      else
      {
         for(i = 0 ; i < this->_nrows ; i ++)
         {
            if(rows[i] != 0)
            {
               row_perm[nr] = i;
               nr++;
            }
         }
         for(i = 0 ; i < this->_ncols ; i ++)
         {
            if(cols[i] != 0)
            {
               col_perm[nc] = i;
               nc++;
            }
         }
      }
      
      row_perm.Resize(nr, true, false);
      col_perm.Resize(nc, true, false);
      
      err = this->SubMatrix(row_perm, col_perm, location, csrmat_out); PARGEMSLR_CHKERR(err);
      
      return err;
   }
   template int CsrMatrixClass<float>::SubMatrixNoPerm( vector_int &rows, vector_int &cols, vector_int &row_perm, vector_int &col_perm, bool complement, int location, CsrMatrixClass<float> &csrmat_out);
   template int CsrMatrixClass<double>::SubMatrixNoPerm( vector_int &rows, vector_int &cols, vector_int &row_perm, vector_int &col_perm, bool complement, int location, CsrMatrixClass<double> &csrmat_out);
   template int CsrMatrixClass<complexs>::SubMatrixNoPerm( vector_int &rows, vector_int &cols, vector_int &row_perm, vector_int &col_perm, bool complement, int location, CsrMatrixClass<complexs> &csrmat_out);
   template int CsrMatrixClass<complexd>::SubMatrixNoPerm( vector_int &rows, vector_int &cols, vector_int &row_perm, vector_int &col_perm, bool complement, int location, CsrMatrixClass<complexd> &csrmat_out);
   
   template <typename T>
   int CsrMatrixClass<T>::GetConnectedComponents(std::vector<vector_int> &comp_indices, int &ncomps)
   {
      
      if(this->_nrows != this->_ncols)
      {
         PARGEMSLR_ERROR("Connected components only for square matrices.");
      }
      
      bool iscsr = this->_iscsr;
      
      if(!iscsr)
      {
         /* if is csc, convert to csr */
         this->Convert(true);
      }
      
#ifdef PARGEMSLR_CUDA
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Extracting connected components only availiable on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
#endif
      
      int i, n, qs, qe, ccomp, accumcomp, idx, size, idx2, j1, j2, j;
      vector_int marker, queue, comps_size, order, comps_size_adj, comps_map;
      
      CsrMatrixClass<T> AT, AAT;
      CsrMatrixTransposeHost(*this, AT);
      CsrMatrixAddHost(*this, AT, AAT);
      AT.Clear();
      
      int *A_i = AAT.GetI();
      int *A_j = AAT.GetJ();
      
      n = this->_nrows;
      
      /* fill marker with -1 */
      marker.Setup(n);
      marker.Fill(-1);
      
      /* create the helper vectors */
      queue.Setup(n);
      comps_size.Setup(0,1,kMemoryHost,false);
      
      /* current component, start from 0 */
      ccomp = 0;
      
      /* main loop */
      for(i = 0 ; i < n ; i ++)
      {
         if(marker[i] < 0)
         {
            /* haven't visited this node */
            queue[0] = i;
            qs = 0;
            qe = 1;
            size = 0;
            /* mark as visited, and add size */
            marker[i] = ccomp;
            size++;
            
            /* BFS loop */
            while(qe > qs)
            {
               idx = queue[qs];
               
               /* visit nbhs */
               j1 = A_i[idx];
               j2 = A_i[idx+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  idx2 = A_j[j];
                  if(marker[idx2]<0)
                  {
                     /* haven't visited, add to queue */
                     queue[qe++] = idx2;
                     /* mark as visited, and add size */
                     marker[idx2] = ccomp;
                     size++;
                  }
               }
               /* go to next */
               qs++;
            }
            
            /* done of this component */
            ccomp++;
            comps_size.PushBack(size);
         }
      }/* end of the main loop */
      
      if(ncomps > 1 && ncomps < ccomp)
      {
         /* in this case, merge multiple components together 
          * TODO: better algorithm to have more even partition
          * we start from the smallest one, and accumulate
          */
         accumcomp = 0;
         j = 0;
         
         comps_size_adj.Setup(ncomps);
         comps_map.Setup(ccomp);
         
         /* sort ascending */
         comps_size.Sort(order, true, false);
         
         /* main loop */
         for(i = 0 ; i < ccomp ; i ++)
         {
            accumcomp += comps_size[order[i]];
            comps_map[order[i]] = j;
            
            /* use n/ncomps+1, since n/ncomps is an integer, might have (n/ncomps)*ncomps < n */
            if(accumcomp > (n/ncomps+1))
            {
               /* we have accumulated enough for this one */
               comps_size_adj[j] = accumcomp;
               accumcomp = 0;
               j++;
            }
            
            if((ccomp-i)==(ncomps-j))
            {
               /* in this case, we don't have enough remaining parts
                * need to stop anyway
                */
               comps_size_adj[j] = accumcomp;
               i++;
               j++;
               for( ; i < ccomp ; i++)
               {
                  comps_size_adj[j] = comps_size[order[i]];
                  comps_map[order[i]] = j;
                  j++;
               }
            }
         }
         /* now build the comp_indices */
         comp_indices.resize(ncomps);
         for(i = 0 ; i < ncomps ; i ++)
         {
            /* reserve size for pushback */
            comp_indices[i].Setup(0, comps_size_adj[i], kMemoryHost, false);
         }
         for(i = 0 ; i < n ; i ++)
         {
            idx = comps_map[marker[i]];
            comp_indices[idx].PushBack(i);
         }
         for(i = 0 ; i < ncomps ; i ++)
         {
            /* sort in ascending order */
            comp_indices[i].Sort(true);
         }
      }
      else
      {
         /* now build the comp_indices */
         comp_indices.resize(ccomp);
         for(i = 0 ; i < ccomp ; i ++)
         {
            /* reserve size for pushback */
            comp_indices[i].Setup(0, comps_size[i], kMemoryHost, false);
         }
         for(i = 0 ; i < n ; i ++)
         {
            idx = marker[i];
            comp_indices[idx].PushBack(i);
         }
         for(i = 0 ; i < ccomp ; i ++)
         {
            /* sort in ascending order */
            comp_indices[i].Sort(true);
         }
         ncomps = ccomp;
      }
      
      /* deallocate */
      AAT.Clear();
      comps_size.Clear();
      marker.Clear();
      queue.Clear();
      order.Clear();
      
      if(!iscsr)
      {
         this->Convert(false);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int CsrMatrixClass<float>::GetConnectedComponents(std::vector<vector_int> &comp_indices, int &ncomps);
   template int CsrMatrixClass<double>::GetConnectedComponents(std::vector<vector_int> &comp_indices, int &ncomps);
   template int CsrMatrixClass<complexs>::GetConnectedComponents(std::vector<vector_int> &comp_indices, int &ncomps);
   template int CsrMatrixClass<complexd>::GetConnectedComponents(std::vector<vector_int> &comp_indices, int &ncomps);
   
   template <typename T>
   int CsrMatrixClass<T>::SetupVectorPtrStr(SequentialVectorClass<T> &vec)
   {
      return vec.SetupPtrStr(*this);
   }
   template int CsrMatrixClass<float>::SetupVectorPtrStr(SequentialVectorClass<float> &vec);
   template int CsrMatrixClass<double>::SetupVectorPtrStr(SequentialVectorClass<double> &vec);
   template int CsrMatrixClass<complexs>::SetupVectorPtrStr(SequentialVectorClass<complexs> &vec);
   template int CsrMatrixClass<complexd>::SetupVectorPtrStr(SequentialVectorClass<complexd> &vec);
   
   template <typename T>
   int CsrMatrixClass<T>::GetDataLocation() const
   {
      PARGEMSLR_CHKERR( this->_i_vec.GetDataLocation() != this->_j_vec.GetDataLocation() );
#ifdef PARGEMSLR_DEBUG
      if(this->_isholdingdata)
      {
         PARGEMSLR_CHKERR( this->_i_vec.GetDataLocation() != this->_a_vec.GetDataLocation() );
      }
#endif
      return this->_i_vec.GetDataLocation();
   }
   template int CsrMatrixClass<float>::GetDataLocation() const;
   template int CsrMatrixClass<double>::GetDataLocation() const;
   template int CsrMatrixClass<complexs>::GetDataLocation() const;
   template int CsrMatrixClass<complexd>::GetDataLocation() const;
   
   template <typename T>
   int CsrMatrixClass<T>::GetNumRowsLocal() const
   {
      return this->_nrows;
   }
   template int CsrMatrixClass<float>::GetNumRowsLocal() const;
   template int CsrMatrixClass<double>::GetNumRowsLocal() const;
   template int CsrMatrixClass<complexs>::GetNumRowsLocal() const;
   template int CsrMatrixClass<complexd>::GetNumRowsLocal() const;
   
   template <typename T>
   int CsrMatrixClass<T>::GetNumColsLocal() const
   {
      return this->_ncols;
   }
   template int CsrMatrixClass<float>::GetNumColsLocal() const;
   template int CsrMatrixClass<double>::GetNumColsLocal() const;
   template int CsrMatrixClass<complexs>::GetNumColsLocal() const;
   template int CsrMatrixClass<complexd>::GetNumColsLocal() const;
   
   template <typename T>
   long int CsrMatrixClass<T>::GetNumNonzeros() const
   {
      return (long int)(this->_nnz);
   }
   template long int CsrMatrixClass<float>::GetNumNonzeros() const;
   template long int CsrMatrixClass<double>::GetNumNonzeros() const;
   template long int CsrMatrixClass<complexs>::GetNumNonzeros() const;
   template long int CsrMatrixClass<complexd>::GetNumNonzeros() const;
   
   template <typename T>
   int CsrMatrixClass<T>::SetNumNonzeros()
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Set csr matrix nnz only work on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      
      if(this->_iscsr)
      {
         this->_nnz = this->_i_vec[this->_nrows];
      }
      else
      {
         this->_nnz = this->_i_vec[this->_ncols];
      }
      
      this->_j_vec.Resize( this->_nnz, true, false);
      this->_a_vec.Resize( this->_nnz, true, false);
      
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::SetNumNonzeros();
   template int CsrMatrixClass<double>::SetNumNonzeros();
   template int CsrMatrixClass<complexs>::SetNumNonzeros();
   template int CsrMatrixClass<complexd>::SetNumNonzeros();
   
   template <typename T>
   int* CsrMatrixClass<T>::GetI() const
   {
      return this->_i_vec.GetData();
   }
   template int* CsrMatrixClass<float>::GetI() const;
   template int* CsrMatrixClass<double>::GetI() const;
   template int* CsrMatrixClass<complexs>::GetI() const;
   template int* CsrMatrixClass<complexd>::GetI() const;
   
   template <typename T>
   int* CsrMatrixClass<T>::GetJ() const
   {
      return this->_j_vec.GetData();
   }
   template int* CsrMatrixClass<float>::GetJ() const;
   template int* CsrMatrixClass<double>::GetJ() const;
   template int* CsrMatrixClass<complexs>::GetJ() const;
   template int* CsrMatrixClass<complexd>::GetJ() const;
   
   template <typename T>
   T* CsrMatrixClass<T>::GetData() const
   {
      return this->_a_vec.GetData();
   }
   template float* CsrMatrixClass<float>::GetData() const;
   template double* CsrMatrixClass<double>::GetData() const;
   template complexs* CsrMatrixClass<complexs>::GetData() const;
   template complexd* CsrMatrixClass<complexd>::GetData() const;
   
   template <typename T>
   IntVectorClass<int>& CsrMatrixClass<T>::GetIVector()
   {
      return this->_i_vec;
   }
   template IntVectorClass<int>& CsrMatrixClass<float>::GetIVector();
   template IntVectorClass<int>& CsrMatrixClass<double>::GetIVector();
   template IntVectorClass<int>& CsrMatrixClass<complexs>::GetIVector();
   template IntVectorClass<int>& CsrMatrixClass<complexd>::GetIVector();
   
   
   template <typename T>
   const IntVectorClass<int>& CsrMatrixClass<T>::GetIVector() const
   {
      return this->_i_vec;
   }
   template const IntVectorClass<int>& CsrMatrixClass<float>::GetIVector() const;
   template const IntVectorClass<int>& CsrMatrixClass<double>::GetIVector() const;
   template const IntVectorClass<int>& CsrMatrixClass<complexs>::GetIVector() const;
   template const IntVectorClass<int>& CsrMatrixClass<complexd>::GetIVector() const;
   
   template <typename T>
   IntVectorClass<int>& CsrMatrixClass<T>::GetJVector()
   {
      return this->_j_vec;
   }
   template IntVectorClass<int>& CsrMatrixClass<float>::GetJVector();
   template IntVectorClass<int>& CsrMatrixClass<double>::GetJVector();
   template IntVectorClass<int>& CsrMatrixClass<complexs>::GetJVector();
   template IntVectorClass<int>& CsrMatrixClass<complexd>::GetJVector();
   
   
   template <typename T>
   const IntVectorClass<int>& CsrMatrixClass<T>::GetJVector() const
   {
      return this->_j_vec;
   }
   template const IntVectorClass<int>& CsrMatrixClass<float>::GetJVector() const;
   template const IntVectorClass<int>& CsrMatrixClass<double>::GetJVector() const;
   template const IntVectorClass<int>& CsrMatrixClass<complexs>::GetJVector() const;
   template const IntVectorClass<int>& CsrMatrixClass<complexd>::GetJVector() const;
   
   template <typename T>
   SequentialVectorClass<T>& CsrMatrixClass<T>::GetDataVector()
   {
      return this->_a_vec;
   }
   template SequentialVectorClass<float>& CsrMatrixClass<float>::GetDataVector();
   template SequentialVectorClass<double>& CsrMatrixClass<double>::GetDataVector();
   template SequentialVectorClass<complexs>& CsrMatrixClass<complexs>::GetDataVector();
   template SequentialVectorClass<complexd>& CsrMatrixClass<complexd>::GetDataVector();
   
   
   template <typename T>
   const SequentialVectorClass<T>& CsrMatrixClass<T>::GetDataVector() const
   {
      return this->_a_vec;
   }
   template const SequentialVectorClass<float>& CsrMatrixClass<float>::GetDataVector() const;
   template const SequentialVectorClass<double>& CsrMatrixClass<double>::GetDataVector() const;
   template const SequentialVectorClass<complexs>& CsrMatrixClass<complexs>::GetDataVector() const;
   template const SequentialVectorClass<complexd>& CsrMatrixClass<complexd>::GetDataVector() const;
   
#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
   template <typename T>
   cusparseSpMatDescr_t CsrMatrixClass<T>::GetCusparseMat() const
   {
      return this->_cusparse_mat;
   }
   template cusparseSpMatDescr_t CsrMatrixClass<float>::GetCusparseMat() const;
   template cusparseSpMatDescr_t CsrMatrixClass<double>::GetCusparseMat() const;
   template cusparseSpMatDescr_t CsrMatrixClass<complexs>::GetCusparseMat() const;
   template cusparseSpMatDescr_t CsrMatrixClass<complexd>::GetCusparseMat() const;
   
   template <typename T>
   int CsrMatrixClass<T>::SetCusparseMat(cusparseSpMatDescr_t cusparse_mat)
   {
      if(this->_cusparse_mat)
      {
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroySpMat(this->_cusparse_mat)) );
      }
      this->_cusparse_mat = cusparse_mat;
   }
   template int CsrMatrixClass<float>::SetCusparseMat(cusparseSpMatDescr_t cusparse_mat);
   template int CsrMatrixClass<double>::SetCusparseMat(cusparseSpMatDescr_t cusparse_mat);
   template int CsrMatrixClass<complexs>::SetCusparseMat(cusparseSpMatDescr_t cusparse_mat);
   template int CsrMatrixClass<complexd>::SetCusparseMat(cusparseSpMatDescr_t cusparse_mat);
#endif
#endif
   
   template <typename T>
   bool CsrMatrixClass<T>::IsHoldingData() const
   {
      return this->_isholdingdata;
   }
   template bool CsrMatrixClass<float>::IsHoldingData() const;
   template bool CsrMatrixClass<double>::IsHoldingData() const;
   template bool CsrMatrixClass<complexs>::IsHoldingData() const;
   template bool CsrMatrixClass<complexd>::IsHoldingData() const;
   
   template <typename T>
   bool& CsrMatrixClass<T>::IsCsr()
   {
      return this->_iscsr;
   }
   template bool& CsrMatrixClass<float>::IsCsr();
   template bool& CsrMatrixClass<double>::IsCsr();
   template bool& CsrMatrixClass<complexs>::IsCsr();
   template bool& CsrMatrixClass<complexd>::IsCsr();
   
   template <typename T>
   const bool& CsrMatrixClass<T>::IsCsr() const
   {
      return this->_iscsr;
   }
   template const bool& CsrMatrixClass<float>::IsCsr() const;
   template const bool& CsrMatrixClass<double>::IsCsr() const;
   template const bool& CsrMatrixClass<complexs>::IsCsr() const;
   template const bool& CsrMatrixClass<complexd>::IsCsr() const;
   
   template <typename T>
   bool& CsrMatrixClass<T>::IsRowSorted()
   {
      return this->_isrowsorted;
   }
   template bool& CsrMatrixClass<float>::IsRowSorted();
   template bool& CsrMatrixClass<double>::IsRowSorted();
   template bool& CsrMatrixClass<complexs>::IsRowSorted();
   template bool& CsrMatrixClass<complexd>::IsRowSorted();
   
   template <typename T>
   const bool& CsrMatrixClass<T>::IsRowSorted() const
   {
      return this->_isrowsorted;
   }
   template const bool& CsrMatrixClass<float>::IsRowSorted() const;
   template const bool& CsrMatrixClass<double>::IsRowSorted() const;
   template const bool& CsrMatrixClass<complexs>::IsRowSorted() const;
   template const bool& CsrMatrixClass<complexd>::IsRowSorted() const;
   
   /**
    * @brief   Sort the column/row indices in each row/column.
    * @details Sort the column/row indices in each row/column.
    * @return     Return error message.
    */
   template <typename T>
   int CsrMatrixClass<T>::SortRow()
   {
      return CsrMatrixSortRow(*this);
   }
   template int CsrMatrixClass<float>::SortRow();
   template int CsrMatrixClass<double>::SortRow();
   template int CsrMatrixClass<complexs>::SortRow();
   template int CsrMatrixClass<complexd>::SortRow();
   
#ifndef PARGEMSLR_CUDA
   
   template <typename T>
   int CsrMatrixClass<T>::Eye()
   {
      /* don't need to worry for the host version */
      int   n, err = 0;
      
      PARGEMSLR_CHKERR( this->_ncols != this->_nrows);
      
      /* change to hold value */
      this->_isholdingdata = true;
      
      if(this->_ncols == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      n = this->_ncols;
      
      this->_nnz = n;
      
      /* create I, J, and A */
      err = this->_i_vec.UnitPerm(); PARGEMSLR_CHKERR(err);
      err = this->_j_vec.Setup(n); PARGEMSLR_CHKERR(err);
      err = this->_j_vec.UnitPerm(); PARGEMSLR_CHKERR(err);
      err = this->_a_vec.Setup(n); PARGEMSLR_CHKERR(err);
      err = this->_a_vec.Fill(1.0); PARGEMSLR_CHKERR(err);
      
      return err;
      
   }
   template int CsrMatrixClass<float>::Eye();
   template int CsrMatrixClass<double>::Eye();
   template int CsrMatrixClass<complexs>::Eye();
   template int CsrMatrixClass<complexd>::Eye();
   
   template <typename T>
   int CsrMatrixClass<T>::MoveData( const int &location)
   {
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::MoveData( const int &location);
   template int CsrMatrixClass<double>::MoveData( const int &location);
   template int CsrMatrixClass<complexs>::MoveData( const int &location);
   template int CsrMatrixClass<complexd>::MoveData( const int &location);
   
#endif
   
   template <typename T>
   int CsrMatrixClass<T>::Fill(const T &v)
   {
      int err = 0;
      
      err = this->_a_vec.Fill(v); PARGEMSLR_CHKERR(err);
      
      return err;
   }
   template int CsrMatrixClass<float>::Fill(const float &v);
   template int CsrMatrixClass<double>::Fill(const double &v);
   template int CsrMatrixClass<complexs>::Fill(const complexs &v);
   template int CsrMatrixClass<complexd>::Fill(const complexd &v);
   
   template <typename T>
   int CsrMatrixClass<T>::Scale(const T &alpha)
   {
      PARGEMSLR_CHKERR(!this->IsHoldingData());
      this->_a_vec.Scale(alpha);
      
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::Scale(const float &alpha);
   template int CsrMatrixClass<double>::Scale(const double &alpha);
   template int CsrMatrixClass<complexs>::Scale(const complexs &alpha);
   template int CsrMatrixClass<complexd>::Scale(const complexd &alpha);
   
   template <typename T>
   int CsrMatrixClass<T>::GetDiagScale(T &scale)
   {
      int      i, j, j1, j2;
      double   val, temp_val;
      
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("GetDiagScale only work for the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      
      scale = T();
      val = 0.0;
      
      if(this->_iscsr)
      {
         for(i = 0 ; i < this->_nrows ; i ++)
         {
            j1 = this->_i_vec[i];
            j2 = this->_i_vec[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               if(this->_j_vec[j] == i)
               {
                  /* diagonal! */
                  temp_val = PargemslrAbs(this->_a_vec[j]);
                  if(temp_val > val )
                  {
                     scale = this->_a_vec[j];
                     val = temp_val;
                  }
               }
            }
         }
      }
      else
      {
         for(i = 0 ; i < this->_ncols ; i ++)
         {
            j1 = this->_i_vec[i];
            j2 = this->_i_vec[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               if(this->_j_vec[j] == i)
               {
                  /* diagonal! */
                  temp_val = PargemslrAbs(this->_a_vec[j]);
                  if(temp_val > val )
                  {
                     scale = this->_a_vec[j];
                     val = temp_val;
                  }
               }
            }
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::GetDiagScale(float &scale);
   template int CsrMatrixClass<double>::GetDiagScale(double &scale);
   template int CsrMatrixClass<complexs>::GetDiagScale(complexs &scale);
   template int CsrMatrixClass<complexd>::GetDiagScale(complexd &scale);
   
   template <typename T>
   int CsrMatrixClass<T>::MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y)
   {
      return CsrMatrixMatVec( *this, trans, alpha, x, beta, y);
   }
   template int CsrMatrixClass<float>::MatVec( char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   template int CsrMatrixClass<double>::MatVec( char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   template int CsrMatrixClass<complexs>::MatVec( char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   template int CsrMatrixClass<complexd>::MatVec( char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
   template <typename T>
   int CsrMatrixClass<T>::MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, const VectorClass<T> &y, VectorClass<T> &z)
   {
      if(y.GetData() == z.GetData())
      {
         /* in this case, no need to touch y */
         return CsrMatrixMatVec( *this, trans, alpha, x, beta, z);
      }
      
      T zero = T(0.0);
      T one = T(1.0);
      
      z.Fill(zero);
      z.Axpy(one, y);
      
      return CsrMatrixMatVec( *this, trans, alpha, x, beta, z);
   }
   template int CsrMatrixClass<float>::MatVec( char trans, const float &alpha, const VectorClass<float> &x, const float &beta, const VectorClass<float> &y, VectorClass<float> &z);
   template int CsrMatrixClass<double>::MatVec( char trans, const double &alpha, const VectorClass<double> &x, const double &beta, const VectorClass<double> &y, VectorClass<double> &z);
   template int CsrMatrixClass<complexs>::MatVec( char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, const VectorClass<complexs> &y, VectorClass<complexs> &z);
   template int CsrMatrixClass<complexd>::MatVec( char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, const VectorClass<complexd> &y, VectorClass<complexd> &z);
   
   template <typename T>
   int CsrMatrixClass<T>::MatMat( const T &alpha, const CsrMatrixClass<T> &A, char transa, const CsrMatrixClass<T> &B, char transb, const T &beta)
   {
      PARGEMSLR_ERROR("CSR matrix matmat not supported yet.");
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::MatMat( const float &alpha, const CsrMatrixClass<float> &A, char transa, const CsrMatrixClass<float> &B, char transb, const float &beta);
   template int CsrMatrixClass<double>::MatMat( const double &alpha, const CsrMatrixClass<double> &A, char transa, const CsrMatrixClass<double> &B, char transb, const double &beta);
   template int CsrMatrixClass<complexs>::MatMat( const complexs &alpha, const CsrMatrixClass<complexs> &A, char transa, const CsrMatrixClass<complexs> &B, char transb, const complexs &beta);
   template int CsrMatrixClass<complexd>::MatMat( const complexd &alpha, const CsrMatrixClass<complexd> &A, char transa, const CsrMatrixClass<complexd> &B, char transb, const complexd &beta);
   
   template <typename T>
   int CsrMatrixClass<T>::Convert(bool csr)
   {
      if(this->_iscsr == csr)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      int   ncols, nrows, nnz;
      int   *i_temp, *j_temp;
      T     *a_temp = NULL;
      int   location;
      bool  hold_data;
      
      location = this->GetDataLocation();
      hold_data = this->_isholdingdata;
      
      if(location == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Convert between csr and csc only works on the host.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      
      if(_iscsr)
      {
         ncols = this->_ncols;
         nrows = this->_nrows;
      }
      else
      {
         nrows = this->_ncols;
         ncols = this->_nrows;
      }
      
      nnz = this->_nnz;
      
      PARGEMSLR_MALLOC( i_temp, nrows+1, kMemoryHost, int);
      PARGEMSLR_MEMCPY( i_temp, this->GetI(), nrows+1, kMemoryHost, location, int);
      
      PARGEMSLR_MALLOC( j_temp, nnz, kMemoryHost, int);
      PARGEMSLR_MEMCPY( j_temp, this->GetJ(), nnz, kMemoryHost, location, int);
      
      if(hold_data)
      {
         PARGEMSLR_MALLOC( a_temp, nnz, kMemoryHost, T);
         PARGEMSLR_MEMCPY( a_temp, this->GetData(), nnz, kMemoryHost, location, T);
         
         this->Setup( ncols, nrows, nnz, location, true, false, csr);
         CsrMatrixP2CscMatrixPHost<0,0>(nrows, ncols, true, a_temp, j_temp, i_temp, this->GetData(), this->GetJ(), this->GetI());
      }
      else
      {
         this->Setup( ncols, nrows, nnz, location, false, false, csr);
         CsrMatrixP2CscMatrixPHost<0,0>(nrows, ncols, false, a_temp, j_temp, i_temp, a_temp, this->GetJ(), this->GetI());
      }
      
      PARGEMSLR_FREE(i_temp, kMemoryHost);
      PARGEMSLR_FREE(j_temp, kMemoryHost);
      
      if(hold_data)
      {
         PARGEMSLR_FREE(a_temp, kMemoryHost);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int CsrMatrixClass<float>::Convert(bool csr);
   template int CsrMatrixClass<double>::Convert(bool csr);
   template int CsrMatrixClass<complexs>::Convert(bool csr);
   template int CsrMatrixClass<complexd>::Convert(bool csr);
   
   template <typename T>
   int CsrMatrixClass<T>::Transpose()
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Transpose only works on host.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      
      CsrMatrixClass<T> AT = std::move(*this);
      
      CsrMatrixTransposeHost( AT, *this);
      
      AT.Clear();
      
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::Transpose();
   template int CsrMatrixClass<double>::Transpose();
   template int CsrMatrixClass<complexs>::Transpose();
   template int CsrMatrixClass<complexd>::Transpose();
   
   template <typename T>
   int CsrMatrixClass<T>::Transpose(CsrMatrixClass<T> &AT)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Transpose only works on host.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      
      return CsrMatrixTransposeHost( *this, AT);
   }
   template int CsrMatrixClass<float>::Transpose(CsrMatrixClass<float> &AT);
   template int CsrMatrixClass<double>::Transpose(CsrMatrixClass<double> &AT);
   template int CsrMatrixClass<complexs>::Transpose(CsrMatrixClass<complexs> &AT);
   template int CsrMatrixClass<complexd>::Transpose(CsrMatrixClass<complexd> &AT);
   
   template <typename T>
   int CsrMatrixClass<T>::PlotPattern( int *rperm, int *cperm, int conditiona, int conditionb)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Plot Pattern only works on host.");
      }
      
      if(!(this->_iscsr))
      {
         PARGEMSLR_ERROR("Plot Pattern only works for CSR matrix.");
      }
      
      int i, ii, j, j1, j2;
      vector_int marker;
      if(conditiona == conditionb)
      {
         std::cout<<this->_nrows<<" by "<<this->_ncols<<" matrix with "<<this->_nnz<<" nnzs."<<std::endl;
         if(rperm == NULL || cperm == NULL)
         {
            marker.Setup(_ncols);
            for(i = 0 ; i < _nrows ; i ++)
            {
               marker.Fill(-1);
               j1 = this->_i_vec[i];
               j2 = this->_i_vec[i+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  marker[this->_j_vec[j]] = 1;
               }
               for( j = 0 ; j < _ncols ; j ++)
               {
                  if( marker[j] < 0)
                  {
                     std::cout<<"  ";
                  }
                  else
                  {
                     std::cout<<"* ";
                  }
               }
               std::cout<<std::endl;
            }
            marker.Clear();
         }
         else
         {
            marker.Setup(_ncols);
            for(ii = 0 ; ii < _nrows ; ii ++)
            {
               marker.Fill(-1);
               i = rperm[ii];
               j1 = this->_i_vec[i];
               j2 = this->_i_vec[i+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  marker[this->_j_vec[j]] = 1;
               }
               for( j = 0 ; j < _ncols ; j ++)
               {
                  if( marker[cperm[j]] < 0)
                  {
                     std::cout<<"  ";
                  }
                  else
                  {
                     std::cout<<"* ";
                  }
               }
               std::cout<<std::endl;
            }
            marker.Clear();
         }
      }
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::PlotPattern( int *rperm, int *cperm, int conditiona, int conditionb);
   template int CsrMatrixClass<double>::PlotPattern( int *rperm, int *cperm, int conditiona, int conditionb);
   template int CsrMatrixClass<complexs>::PlotPattern( int *rperm, int *cperm, int conditiona, int conditionb);
   template int CsrMatrixClass<complexd>::PlotPattern( int *rperm, int *cperm, int conditiona, int conditionb);
   
   template <typename T>
   int CsrMatrixClass<T>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Plot Pattern only works on host.");
      }
      
      if(!(this->_iscsr))
      {
         PARGEMSLR_ERROR("Plot Pattern only works for CSR matrix.");
      }
      
      int i, ii, j, j1, j2;
      
      if(conditiona == conditionb)
      {
         
         FILE *fdata, *pgnuplot;
         
         char tempfilename[1024];
         snprintf( tempfilename, 1024, "./TempData/%s", datafilename );
         
         if ((fdata = fopen(tempfilename, "w")) == NULL)
         {
            printf("Can't open file.\n");
            return PARGEMSLR_ERROR_IO_ERROR;
         }
         
         if ((pgnuplot = popen("gnuplot -persistent", "w")) == NULL)
         {
            printf("Can't open gnuplot file.\n");
            return PARGEMSLR_ERROR_IO_ERROR;
         }
         
         if(rperm == NULL || cperm == NULL)
         {
            for(i = 0 ; i < _nrows ; i ++)
            {
               j1 = this->_i_vec[i];
               j2 = this->_i_vec[i+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  fprintf(fdata, "%d %d \n", this->_j_vec[j]+1, _nrows-i+1);
               }
            }
         }
         else
         {
            vector_int rcperm;
            rcperm.Setup(_ncols);
            for(i = 0 ; i < _ncols ; i ++)
            {
               rcperm[cperm[i]] = i;
            }
            for(ii = 0 ; ii < _nrows ; ii ++)
            {
               i = rperm[ii];
               j1 = this->_i_vec[i];
               j2 = this->_i_vec[i+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  fprintf(fdata, "%d %d \n", rcperm[this->_j_vec[j]]+1, _nrows-ii+1);
               }
            }
            rcperm.Clear();
         }
         
         fclose(fdata);
         
         fprintf(pgnuplot, "set title \"nnz = %ld\"\n", this->GetNumNonzeros());
         fprintf(pgnuplot, "set xrange [1:%d]\n", _ncols);
         fprintf(pgnuplot, "set yrange [1:%d]\n", _nrows);
         fprintf(pgnuplot, "plot '%s' pt 0\n", tempfilename);
         
         pclose(pgnuplot);
      }
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb);
   template int CsrMatrixClass<double>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb);
   template int CsrMatrixClass<complexs>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb);
   template int CsrMatrixClass<complexd>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb);
   
   template <typename T>
   int CsrMatrixClass<T>::Plot( int *perm, int conditiona, int conditionb, int width)
   {
      
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Plot matrix only availiable on host.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      
      return CsrMatrixPlotHost( *this, perm, conditiona, conditionb, width);
      
   }
   template int CsrMatrixClass<float>::Plot( int *perm, int conditiona, int conditionb, int width);
   template int CsrMatrixClass<double>::Plot( int *perm, int conditiona, int conditionb, int width);
   template int CsrMatrixClass<complexs>::Plot( int *perm, int conditiona, int conditionb, int width);
   template int CsrMatrixClass<complexd>::Plot( int *perm, int conditiona, int conditionb, int width);
   
   template <typename T>
   int CsrMatrixClass<T>::Laplacian(int nx, int ny, int nz, T alphax, T alphay, T alphaz, T shift, bool rand_perturb)
   {
      int err;
      CooMatrixClass<T> coo_mat;
      err = coo_mat.Laplacian( nx, ny, nz, alphax, alphay, alphaz, shift, rand_perturb); PARGEMSLR_CHKERR(err);
      err = coo_mat.ToCsr( kMemoryHost, *this); PARGEMSLR_CHKERR(err);
      err = coo_mat.Clear(); PARGEMSLR_CHKERR(err);

      return err;
   }
   template int CsrMatrixClass<float>::Laplacian(int nx, int ny, int nz, float alphax, float alphay, float alphaz, float shift, bool rand_perturb);
   template int CsrMatrixClass<double>::Laplacian(int nx, int ny, int nz, double alphax, double alphay, double alphaz, double shift, bool rand_perturb);
   template int CsrMatrixClass<complexs>::Laplacian(int nx, int ny, int nz, complexs alphax, complexs alphay, complexs alphaz, complexs shift, bool rand_perturb);
   template int CsrMatrixClass<complexd>::Laplacian(int nx, int ny, int nz, complexd alphax, complexd alphay, complexd alphaz, complexd shift, bool rand_perturb);
   
   template <typename T>
   int CsrMatrixClass<T>::Helmholtz(int n, T w)
   {
      int err;
      CooMatrixClass<T> coo_mat;
      err = coo_mat.Helmholtz( n, w); PARGEMSLR_CHKERR(err);
      err = coo_mat.ToCsr( kMemoryHost, *this); PARGEMSLR_CHKERR(err);
      err = coo_mat.Clear(); PARGEMSLR_CHKERR(err);

      return err;
   }
   template int CsrMatrixClass<complexs>::Helmholtz(int n, complexs w);
   template int CsrMatrixClass<complexd>::Helmholtz(int n, complexd w);
   
   template <typename T>
   int CsrMatrixClass<T>::ReadFromMMFile(const char *matfile, int idxin)
   {
      int err;
      CooMatrixClass<T> coo_mat;
      err = coo_mat.ReadFromMMFile( matfile, idxin); PARGEMSLR_CHKERR(err);
      err = coo_mat.ToCsr( kMemoryHost, *this); PARGEMSLR_CHKERR(err);
      err = coo_mat.Clear(); PARGEMSLR_CHKERR(err);

      return err;
   }
   template int CsrMatrixClass<float>::ReadFromMMFile(const char *matfile, int idxin);
   template int CsrMatrixClass<double>::ReadFromMMFile(const char *matfile, int idxin);
   template int CsrMatrixClass<complexs>::ReadFromMMFile(const char *matfile, int idxin);
   template int CsrMatrixClass<complexd>::ReadFromMMFile(const char *matfile, int idxin);
   
   template <typename T>
   int CsrMatrixClass<T>::SetNumCols(int cols)
   {
      PARGEMSLR_CHKERR(cols < 0);
      
      this->_ncols = cols;
      
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::SetNumCols(int cols);
   template int CsrMatrixClass<double>::SetNumCols(int cols);
   template int CsrMatrixClass<complexs>::SetNumCols(int cols);
   template int CsrMatrixClass<complexd>::SetNumCols(int cols);
   
   template <typename T>
   template <typename T1>
   int CsrMatrixClass<T>::GetComplexShift(T1 &diagonal_shift)
   {
      /* real case, return 0 */
      diagonal_shift = T1();
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::GetComplexShift(float &diagonal_shift);
   template int CsrMatrixClass<double>::GetComplexShift(double &diagonal_shift);
   
   template <typename T>
   template <typename T1>
   int CsrMatrixClass<T>::GetComplexShift(ComplexValueClass<T1> &diagonal_shift)
   {
      /* complex case, return the shift */
      diagonal_shift = ComplexValueClass<T1>(0.0,this->_diagonal_shift);
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<complexs>::GetComplexShift(complexs &diagonal_shift);
   template int CsrMatrixClass<complexd>::GetComplexShift(complexd &diagonal_shift);
   
}
