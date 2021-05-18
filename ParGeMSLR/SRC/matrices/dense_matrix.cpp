
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "matrix.hpp"
#include "dense_matrix.hpp"
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
   DenseMatrixClass<T>::DenseMatrixClass()
   {
      this->_data_vec.Clear();
      this->_nrows = 0;
      this->_ncols = 0;
      this->_ldim = 0;
   }
   template DenseMatrixClass<float>::DenseMatrixClass();
   template DenseMatrixClass<double>::DenseMatrixClass();
   template DenseMatrixClass<complexs>::DenseMatrixClass();
   template DenseMatrixClass<complexd>::DenseMatrixClass();
   
   template <typename T>
   DenseMatrixClass<T>::DenseMatrixClass(const DenseMatrixClass<T> &mat) : MatrixClass<T>(mat)
   {
      this->_nrows = mat._nrows;
      this->_ldim = mat._ldim;
      this->_ncols = mat._ncols;
      this->_data_vec = mat._data_vec;
   }
   template DenseMatrixClass<float>::DenseMatrixClass(const DenseMatrixClass<float> &mat);
   template DenseMatrixClass<double>::DenseMatrixClass(const DenseMatrixClass<double> &mat);
   template DenseMatrixClass<complexs>::DenseMatrixClass(const DenseMatrixClass<complexs> &mat);
   template DenseMatrixClass<complexd>::DenseMatrixClass(const DenseMatrixClass<complexd> &mat);
   
   template <typename T>
   DenseMatrixClass<T>::DenseMatrixClass( DenseMatrixClass<T> &&mat) : MatrixClass<T>(std::move(mat))
   {
      this->_nrows = mat._nrows;
      mat._nrows = 0;
      this->_ldim = mat._ldim;
      mat._ldim = 0;
      this->_ncols = mat._ncols;
      mat._ncols = 0;
      this->_data_vec = std::move(mat._data_vec);
   }
   template DenseMatrixClass<float>::DenseMatrixClass( DenseMatrixClass<float> &&mat);
   template DenseMatrixClass<double>::DenseMatrixClass( DenseMatrixClass<double> &&mat);
   template DenseMatrixClass<complexs>::DenseMatrixClass( DenseMatrixClass<complexs> &&mat);
   template DenseMatrixClass<complexd>::DenseMatrixClass( DenseMatrixClass<complexd> &&mat);
   
   template <typename T>
   DenseMatrixClass<T>& DenseMatrixClass<T>::operator= (const DenseMatrixClass<T> &mat)
   {
      this->Clear();
      ParallelLogClass::operator=(mat);
      this->_nrows = mat._nrows;
      this->_ldim = mat._ldim;
      this->_ncols = mat._ncols;
      this->_data_vec = mat._data_vec;
      return *this;
   }
   template DenseMatrixClass<float>& DenseMatrixClass<float>::operator= (const DenseMatrixClass<float> &mat);
   template DenseMatrixClass<double>& DenseMatrixClass<double>::operator= (const DenseMatrixClass<double> &mat);
   template DenseMatrixClass<complexs>& DenseMatrixClass<complexs>::operator= (const DenseMatrixClass<complexs> &mat);
   template DenseMatrixClass<complexd>& DenseMatrixClass<complexd>::operator= (const DenseMatrixClass<complexd> &mat);
   
   template <typename T>
   DenseMatrixClass<T>& DenseMatrixClass<T>::operator= ( DenseMatrixClass<T> &&mat)
   {
      this->Clear();
      ParallelLogClass::operator=(std::move(mat));
      this->_nrows = mat._nrows;
      mat._nrows = 0;
      this->_ldim = mat._ldim;
      mat._ldim = 0;
      this->_ncols = mat._ncols;
      mat._ncols = 0;
      this->_data_vec = std::move(mat._data_vec);
      return *this;
   }
   template DenseMatrixClass<float>& DenseMatrixClass<float>::operator= ( DenseMatrixClass<float> &&mat);
   template DenseMatrixClass<double>& DenseMatrixClass<double>::operator= ( DenseMatrixClass<double> &&mat);
   template DenseMatrixClass<complexs>& DenseMatrixClass<complexs>::operator= ( DenseMatrixClass<complexs> &&mat);
   template DenseMatrixClass<complexd>& DenseMatrixClass<complexd>::operator= ( DenseMatrixClass<complexd> &&mat);
   
   template <typename T>
   DenseMatrixClass<T>::~DenseMatrixClass()
   {
      this->Clear();
   }
   template DenseMatrixClass<float>::~DenseMatrixClass();
   template DenseMatrixClass<double>::~DenseMatrixClass();
   template DenseMatrixClass<complexs>::~DenseMatrixClass();
   template DenseMatrixClass<complexd>::~DenseMatrixClass();
   
   template <typename T>
   T& DenseMatrixClass<T>::operator()( int row, int col)
   {
      PARGEMSLR_CHKERR(row < 0);
      PARGEMSLR_CHKERR(col < 0);
      PARGEMSLR_CHKERR(row >= this->_nrows);
      PARGEMSLR_CHKERR(col >= this->_ncols);
      
      return this->_data_vec[row+col*this->_ldim];
   }
   template float& DenseMatrixClass<float>::operator()( int row, int col);
   template double& DenseMatrixClass<double>::operator()( int row, int col);
   template complexs& DenseMatrixClass<complexs>::operator()( int row, int col);
   template complexd& DenseMatrixClass<complexd>::operator()( int row, int col);
   
   template <typename T>
   int DenseMatrixClass<T>::Clear()
   {
      /* base class clear */
      MatrixClass<T>::Clear();
      
      this->_data_vec.Clear();
      this->_nrows = 0;
      this->_ncols = 0;
      this->_ldim = 0;
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int DenseMatrixClass<float>::Clear();
   template int DenseMatrixClass<double>::Clear();
   template int DenseMatrixClass<complexs>::Clear();
   template int DenseMatrixClass<complexd>::Clear();
   
   template <typename T>
   T* DenseMatrixClass<T>::GetData() const
   {
      return this->_data_vec.GetData();
   }
   template float* DenseMatrixClass<float>::GetData() const;
   template double* DenseMatrixClass<double>::GetData() const;
   template complexs* DenseMatrixClass<complexs>::GetData() const;
   template complexd* DenseMatrixClass<complexd>::GetData() const;
   
   template <typename T>
   SequentialVectorClass<T>&  DenseMatrixClass<T>::GetDataVector()
   {
      return this->_data_vec;
   }
   template SequentialVectorClass<float>&  DenseMatrixClass<float>::GetDataVector();
   template SequentialVectorClass<double>&  DenseMatrixClass<double>::GetDataVector();
   template SequentialVectorClass<complexs>&  DenseMatrixClass<complexs>::GetDataVector();
   template SequentialVectorClass<complexd>&  DenseMatrixClass<complexd>::GetDataVector();
   
   template <typename T>
   const SequentialVectorClass<T>&  DenseMatrixClass<T>::GetDataVector() const
   {
      return this->_data_vec;
   }
   template const SequentialVectorClass<float>&  DenseMatrixClass<float>::GetDataVector() const;
   template const SequentialVectorClass<double>&  DenseMatrixClass<double>::GetDataVector() const;
   template const SequentialVectorClass<complexs>&  DenseMatrixClass<complexs>::GetDataVector() const;
   template const SequentialVectorClass<complexd>&  DenseMatrixClass<complexd>::GetDataVector() const;
   
   template <typename T>
   int DenseMatrixClass<T>::GetDataLocation() const
   {
      return this->_data_vec.GetDataLocation();
   }
   template int DenseMatrixClass<float>::GetDataLocation() const;
   template int DenseMatrixClass<double>::GetDataLocation() const;
   template int DenseMatrixClass<complexs>::GetDataLocation() const;
   template int DenseMatrixClass<complexd>::GetDataLocation() const;
   
   template <typename T>
   int DenseMatrixClass<T>::GetNumRowsLocal() const
   {
      return this->_nrows;
   }
   template int DenseMatrixClass<float>::GetNumRowsLocal() const;
   template int DenseMatrixClass<double>::GetNumRowsLocal() const;
   template int DenseMatrixClass<complexs>::GetNumRowsLocal() const;
   template int DenseMatrixClass<complexd>::GetNumRowsLocal() const;
   
   template <typename T>
   int DenseMatrixClass<T>::GetNumColsLocal() const
   {
      return this->_ncols;
   }
   template int DenseMatrixClass<float>::GetNumColsLocal() const;
   template int DenseMatrixClass<double>::GetNumColsLocal() const;
   template int DenseMatrixClass<complexs>::GetNumColsLocal() const;
   template int DenseMatrixClass<complexd>::GetNumColsLocal() const;
   
   template <typename T>
   long int DenseMatrixClass<T>::GetNumNonzeros() const
   {
      return (long int)(this->_nrows) * this->_ncols;
   }
   template long int DenseMatrixClass<float>::GetNumNonzeros() const;
   template long int DenseMatrixClass<double>::GetNumNonzeros() const;
   template long int DenseMatrixClass<complexs>::GetNumNonzeros() const;
   template long int DenseMatrixClass<complexd>::GetNumNonzeros() const;
   
   template <typename T>
   int DenseMatrixClass<T>::GetLeadingDimension() const
   {
      return this->_ldim;
   }
   template int DenseMatrixClass<float>::GetLeadingDimension() const;
   template int DenseMatrixClass<double>::GetLeadingDimension() const;
   template int DenseMatrixClass<complexs>::GetLeadingDimension() const;
   template int DenseMatrixClass<complexd>::GetLeadingDimension() const;
   
   template <typename T>
   bool DenseMatrixClass<T>::IsHoldingData() const
   {
      return this->_data_vec.IsHoldingData();
   }
   template bool DenseMatrixClass<float>::IsHoldingData() const;
   template bool DenseMatrixClass<double>::IsHoldingData() const;
   template bool DenseMatrixClass<complexs>::IsHoldingData() const;
   template bool DenseMatrixClass<complexd>::IsHoldingData() const;
   
   template <typename T>
   int DenseMatrixClass<T>::Setup(int nrows, int ncols)
   {
      return this->Setup( nrows, ncols, this->GetDataLocation(), false);
   }
   template int DenseMatrixClass<float>::Setup(int nrows, int ncols);
   template int DenseMatrixClass<double>::Setup(int nrows, int ncols);
   template int DenseMatrixClass<complexs>::Setup(int nrows, int ncols);
   template int DenseMatrixClass<complexd>::Setup(int nrows, int ncols);
   
   template <typename T>
   int DenseMatrixClass<T>::Setup(int nrows, int ncols, bool setzero)
   {
      return this->Setup( nrows, ncols, kMemoryHost, setzero);
   }
   template int DenseMatrixClass<float>::Setup(int nrows, int ncols, bool setzero);
   template int DenseMatrixClass<double>::Setup(int nrows, int ncols, bool setzero);
   template int DenseMatrixClass<complexs>::Setup(int nrows, int ncols, bool setzero);
   template int DenseMatrixClass<complexd>::Setup(int nrows, int ncols, bool setzero);
   
   template <typename T>
   int DenseMatrixClass<T>::Setup(int nrows, int ncols, int location, bool setzero)
   {
      PARGEMSLR_CHKERR(nrows < 0);
      PARGEMSLR_CHKERR(ncols < 0);

      /* first check if we can use current memory */
      if( location == this->GetDataLocation() && this->IsHoldingData() && this->_ldim * this->_ncols >= nrows * ncols)
      {
         /* we have enough space here, keep it */
         this->_nrows = nrows;
         this->_ldim = nrows;
         this->_ncols = ncols;
         
         if(setzero)
         {
            /* fill the vector with 0 when necessary */
            this->Fill(0);
         }
      }
      else
      {
         this->Clear();
         if(nrows * ncols == 0)
         {
            this->_data_vec.Clear();
         }
         else
         {
            this->_data_vec.Setup( nrows*ncols, nrows*ncols, location, setzero);
         }
         this->_nrows = nrows;
         this->_ldim = nrows;
         this->_ncols = ncols;
      }
      return PARGEMSLR_SUCCESS;
   }
   template int DenseMatrixClass<float>::Setup(int nrows, int ncols, int location, bool setzero);
   template int DenseMatrixClass<double>::Setup(int nrows, int ncols, int location, bool setzero);
   template int DenseMatrixClass<complexs>::Setup(int nrows, int ncols, int location, bool setzero);
   template int DenseMatrixClass<complexd>::Setup(int nrows, int ncols, int location, bool setzero);
   
   template <typename T>
   int DenseMatrixClass<T>::SetupPtr( T* data, int nrows, int ncols, int ldim, int location)
   {
      PARGEMSLR_CHKERR( nrows < 0);
      PARGEMSLR_CHKERR( ncols < 0);
      PARGEMSLR_CHKERR( ldim < nrows);

      this->Clear();
      this->_nrows = nrows;
      this->_ncols = ncols;
      this->_ldim = ldim;
      
      this->_data_vec.SetupPtr(data, ldim*ncols, location);
      
      return PARGEMSLR_SUCCESS;
   }
   template int DenseMatrixClass<float>::SetupPtr( float* data, int nrows, int ncols, int ldim, int location);
   template int DenseMatrixClass<double>::SetupPtr( double* data, int nrows, int ncols, int ldim, int location);
   template int DenseMatrixClass<complexs>::SetupPtr( complexs* data, int nrows, int ncols, int ldim, int location);
   template int DenseMatrixClass<complexd>::SetupPtr( complexd* data, int nrows, int ncols, int ldim, int location);
   
   template <typename T>
   int DenseMatrixClass<T>::SetupPtr(const DenseMatrixClass<T> &mat_in, int row_start, int col_start, int num_rows, int num_cols)
   {
      PARGEMSLR_CHKERR( num_rows < 0);
      PARGEMSLR_CHKERR( num_cols < 0);
      PARGEMSLR_CHKERR( row_start < 0);
      PARGEMSLR_CHKERR( col_start < 0);
      PARGEMSLR_CHKERR( row_start + num_rows > mat_in.GetNumRowsLocal());
      PARGEMSLR_CHKERR( col_start + num_cols > mat_in.GetNumColsLocal());
      
      return this->SetupPtr( mat_in.GetData() + col_start*mat_in.GetLeadingDimension()+row_start, 
                              num_rows, num_cols, mat_in.GetLeadingDimension(), mat_in.GetDataLocation());
   }
   template int DenseMatrixClass<float>::SetupPtr(const DenseMatrixClass<float> &mat_in, int row_start, int col_start, int num_rows, int num_cols);
   template int DenseMatrixClass<double>::SetupPtr(const DenseMatrixClass<double> &mat_in, int row_start, int col_start, int num_rows, int num_cols);
   template int DenseMatrixClass<complexs>::SetupPtr(const DenseMatrixClass<complexs> &mat_in, int row_start, int col_start, int num_rows, int num_cols);
   template int DenseMatrixClass<complexd>::SetupPtr(const DenseMatrixClass<complexd> &mat_in, int row_start, int col_start, int num_rows, int num_cols);
   
   template <typename T>
   int DenseMatrixClass<T>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, DenseMatrixClass<T> &mat_out)
   {
      PARGEMSLR_CHKERR( num_rows < 0);
      PARGEMSLR_CHKERR( num_cols < 0);
      PARGEMSLR_CHKERR( row_start < 0);
      PARGEMSLR_CHKERR( col_start < 0);
      PARGEMSLR_CHKERR( row_start + num_rows > this->_nrows);
      PARGEMSLR_CHKERR( col_start + num_cols > this->_ncols);

      int   i;
      int   loc_from = this->GetDataLocation();
      T*    data_from = this->GetData() + row_start + this->_ldim * col_start;
      
      mat_out.Setup(num_rows, num_cols, location, false);
      T*    data_to = mat_out.GetData();
      
      if( num_rows == this->_ldim)
      {
         /* data is contineous, copy all at once */
         PARGEMSLR_MEMCPY( data_to, data_from, num_rows*num_cols, location, loc_from, T);
      }
      else
      {
         for (i = 0 ; i < num_cols ; i ++) 
         {
            PARGEMSLR_MEMCPY( data_to, data_from, num_rows, location, loc_from, T);
            data_to += num_rows;
            data_from += this->_ldim;
         }
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int DenseMatrixClass<float>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, DenseMatrixClass<float> &mat_out);
   template int DenseMatrixClass<double>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, DenseMatrixClass<double> &mat_out);
   template int DenseMatrixClass<complexs>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, DenseMatrixClass<complexs> &mat_out);
   template int DenseMatrixClass<complexd>::SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, DenseMatrixClass<complexd> &mat_out);
   
   template <typename T>
   int DenseMatrixClass<T>::Fill(const T &v)
   {
      if( this->_ldim == this->_nrows)
      {
         /* in this case, the data is continous in memory */
         SequentialVectorClass<T> temp_vec;
         temp_vec.SetupPtr( this->GetData(), (this->_nrows) * (this->_ncols), this->GetDataLocation());
         temp_vec.Fill(v);
         temp_vec.Clear();
      }
      else
      {
         /* in this case, the data is discontinous in memory */
         int i;
         T* data = this->GetData();
         SequentialVectorClass<T> temp_vec;
         for(i = 0 ; i < this->_ncols ; i ++)
         {
            temp_vec.SetupPtr( data, this->_nrows, this->GetDataLocation());
            temp_vec.Fill(v);
            temp_vec.Clear();
            data += this->_ldim;
         }
      }

      return PARGEMSLR_SUCCESS;
   }
   template int DenseMatrixClass<float>::Fill(const float &v);
   template int DenseMatrixClass<double>::Fill(const double &v);
   template int DenseMatrixClass<complexs>::Fill(const complexs &v);
   template int DenseMatrixClass<complexd>::Fill(const complexd &v);
   
   template <typename T>
   int DenseMatrixClass<T>::Scale(const T &alpha)
   {
      if( this->_ldim == this->_nrows)
      {
         /* in this case, the data is continous in memory */
         SequentialVectorClass<T> temp_vec;
         temp_vec.SetupPtr( this->GetData(), (this->_nrows) * (this->_ncols), this->GetDataLocation());
         temp_vec.Scale(alpha);
         temp_vec.Clear();
      }
      else
      {
         /* in this case, the data is discontinous in memory */
         int i;
         T* data = this->GetData();
         SequentialVectorClass<T> temp_vec;
         for(i = 0 ; i < this->_ncols ; i ++)
         {
            temp_vec.SetupPtr( data, this->_nrows, this->GetDataLocation());
            temp_vec.Scale(alpha);
            temp_vec.Clear();
            data += this->_ldim;
         }
      }

      return PARGEMSLR_SUCCESS;
   }
   template int DenseMatrixClass<float>::Scale(const float &alpha);
   template int DenseMatrixClass<double>::Scale(const double &alpha);
   template int DenseMatrixClass<complexs>::Scale(const complexs &alpha);
   template int DenseMatrixClass<complexd>::Scale(const complexd &alpha);
   
   template <typename T>
   int DenseMatrixClass<T>::Rand()
   {
      if( this->_ldim == this->_nrows)
      {
         /* in this case, the data is continous in memory */
         SequentialVectorClass<T> temp_vec;
         temp_vec.SetupPtr( this->GetData(), (this->_nrows) * (this->_ncols), this->GetDataLocation());
         temp_vec.Rand();
         temp_vec.Clear();
      }
      else
      {
         /* in this case, the data is discontinous in memory */
         int i;
         T* data = this->GetData();
         SequentialVectorClass<T> temp_vec;
         for(i = 0 ; i < this->_ncols ; i ++)
         {
            temp_vec.SetupPtr( data, this->_nrows, this->GetDataLocation());
            temp_vec.Rand();
            temp_vec.Clear();
            data += this->_ldim;
         }
      }

      return PARGEMSLR_SUCCESS;
   }
   template int DenseMatrixClass<float>::Rand();
   template int DenseMatrixClass<double>::Rand();
   template int DenseMatrixClass<complexs>::Rand();
   template int DenseMatrixClass<complexd>::Rand();
   
#ifndef PARGEMSLR_CUDA 
   template <typename T>
   int DenseMatrixClass<T>::Eye()
   {
      PARGEMSLR_CHKERR( this->_nrows != this->_ncols);
      
      int i;
      
      this->Fill(0.0);
      
      for(i = 0 ; i < this->_nrows ; i ++)
      {
         this->operator()(i, i) = 1.0;
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int DenseMatrixClass<float>::Eye();
   template int DenseMatrixClass<double>::Eye();
   template int DenseMatrixClass<complexs>::Eye();
   template int DenseMatrixClass<complexd>::Eye();
#endif
   
   template <typename T>
   int DenseMatrixClass<T>::MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y)
   {
      return DenseMatrixMatVec( *this, trans, alpha, x, beta, y);
   }
   template int DenseMatrixClass<float>::MatVec( char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   template int DenseMatrixClass<double>::MatVec( char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   template int DenseMatrixClass<complexs>::MatVec( char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   template int DenseMatrixClass<complexd>::MatVec( char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
   template <typename T>
   int DenseMatrixClass<T>::MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, const VectorClass<T> &y, VectorClass<T> &z)
   {
      if(y.GetData() == z.GetData())
      {
         /* in this case, no need to touch y */
         return DenseMatrixMatVec( *this, trans, alpha, x, beta, z);
      }
      
      T zero = T(0.0);
      T one = T(1.0);
      
      z.Fill(zero);
      z.Axpy(one, y);
      
      return DenseMatrixMatVec( *this, trans, alpha, x, beta, z);
   }
   template int DenseMatrixClass<float>::MatVec( char trans, const float &alpha, const VectorClass<float> &x, const float &beta, const VectorClass<float> &y, VectorClass<float> &z);
   template int DenseMatrixClass<double>::MatVec( char trans, const double &alpha, const VectorClass<double> &x, const double &beta, const VectorClass<double> &y, VectorClass<double> &z);
   template int DenseMatrixClass<complexs>::MatVec( char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, const VectorClass<complexs> &y, VectorClass<complexs> &z);
   template int DenseMatrixClass<complexd>::MatVec( char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, const VectorClass<complexd> &y, VectorClass<complexd> &z);
   
   template <typename T>
   int DenseMatrixClass<T>::MatMat( const T &alpha, const DenseMatrixClass<T> &A, char transa, const DenseMatrixClass<T> &B, char transb, const T &beta)
   {
      return DenseMatrixMatMat( alpha, A, transa, B, transb, beta, *this);
   }
   template int DenseMatrixClass<float>::MatMat( const float &alpha, const DenseMatrixClass<float> &A, char transa, const DenseMatrixClass<float> &B, char transb, const float &beta);
   template int DenseMatrixClass<double>::MatMat( const double &alpha, const DenseMatrixClass<double> &A, char transa, const DenseMatrixClass<double> &B, char transb, const double &beta);
   template int DenseMatrixClass<complexs>::MatMat( const complexs &alpha, const DenseMatrixClass<complexs> &A, char transa, const DenseMatrixClass<complexs> &B, char transb, const complexs &beta);
   template int DenseMatrixClass<complexd>::MatMat( const complexd &alpha, const DenseMatrixClass<complexd> &A, char transa, const DenseMatrixClass<complexd> &B, char transb, const complexd &beta);
   
   template <typename T>
   int DenseMatrixClass<T>::MoveData( const int &location)
   {
      this->_data_vec.MoveData(location);
      return PARGEMSLR_SUCCESS;
   }
   template int DenseMatrixClass<float>::MoveData( const int &location);
   template int DenseMatrixClass<double>::MoveData( const int &location);
   template int DenseMatrixClass<complexs>::MoveData( const int &location);
   template int DenseMatrixClass<complexd>::MoveData( const int &location);
   
   template <typename T>
   int DenseMatrixClass<T>::Invert()
   {
      /* the memory location will be checked in this function */
      return DenseMatrixInvertHost(*this);
   }
   template int DenseMatrixClass<float>::Invert();
   template int DenseMatrixClass<double>::Invert();
   template int DenseMatrixClass<complexs>::Invert();
   template int DenseMatrixClass<complexd>::Invert();
   
   template <typename T>
   int DenseMatrixClass<T>::InvertUpperTriangular()
   {
      /* the memory location will be checked in this function */
      return DenseMatrixInvertUpperTriangularHost(*this);
   }
   template int DenseMatrixClass<float>::InvertUpperTriangular();
   template int DenseMatrixClass<double>::InvertUpperTriangular();
   template int DenseMatrixClass<complexs>::InvertUpperTriangular();
   template int DenseMatrixClass<complexd>::InvertUpperTriangular();
   
   template <typename T>
   int DenseMatrixClass<T>::Hess( DenseMatrixClass<T> &Q)
   {
      /* the memory location will be checked in this function */
      return DenseMatrixHessDecompositionHost(*this, Q);
   }
   template int DenseMatrixClass<float>::Hess( DenseMatrixClass<float> &Q);
   template int DenseMatrixClass<double>::Hess( DenseMatrixClass<double> &Q);
   template int DenseMatrixClass<complexs>::Hess( DenseMatrixClass<complexs> &Q);
   template int DenseMatrixClass<complexd>::Hess( DenseMatrixClass<complexd> &Q);
   
   template <typename T>
   int DenseMatrixClass<T>::Hess( DenseMatrixClass<T> &Q, int start, int end)
   {
      /* the memory location will be checked in this function */
      return DenseMatrixHessDecompositionHost(*this, start, end, Q);
   }
   template int DenseMatrixClass<float>::Hess( DenseMatrixClass<float> &Q, int start, int end);
   template int DenseMatrixClass<double>::Hess( DenseMatrixClass<double> &Q, int start, int end);
   template int DenseMatrixClass<complexs>::Hess( DenseMatrixClass<complexs> &Q, int start, int end);
   template int DenseMatrixClass<complexd>::Hess( DenseMatrixClass<complexd> &Q, int start, int end);
   
   template <typename T>
   int DenseMatrixClass<T>::HessSchur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi)
   {
      /* the memory location will be checked in this function */
      return DenseMatrixRealHessSchurDecompositionHost( *this, Q, wr, wi);
   }
   template int DenseMatrixClass<float>::HessSchur( DenseMatrixClass<float> &Q, SequentialVectorClass<float> &wr, SequentialVectorClass<float> &wi);
   template int DenseMatrixClass<double>::HessSchur( DenseMatrixClass<double> &Q, SequentialVectorClass<double> &wr, SequentialVectorClass<double> &wi);
   
   template <typename T>
   int DenseMatrixClass<T>::HessSchur( DenseMatrixClass<T> &Q, int start, int end, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi)
   {
      /* the memory location will be checked in this function */
      return DenseMatrixRealHessSchurDecompositionHost( *this, start, end, Q, wr, wi);
   }
   template int DenseMatrixClass<float>::HessSchur( DenseMatrixClass<float> &Q, int start, int end, SequentialVectorClass<float> &wr, SequentialVectorClass<float> &wi);
   template int DenseMatrixClass<double>::HessSchur( DenseMatrixClass<double> &Q, int start, int end, SequentialVectorClass<double> &wr, SequentialVectorClass<double> &wi);
   
   template <typename T>
   int DenseMatrixClass<T>::HessSchur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w)
   {
      /* the memory location will be checked in this function */
      return DenseMatrixComplexHessSchurDecompositionHost( *this, Q, w);
   }
   template int DenseMatrixClass<complexs>::HessSchur( DenseMatrixClass<complexs> &Q, SequentialVectorClass<complexs> &w);
   template int DenseMatrixClass<complexd>::HessSchur( DenseMatrixClass<complexd> &Q, SequentialVectorClass<complexd> &w);
   
   template <typename T>
   int DenseMatrixClass<T>::HessSchur( DenseMatrixClass<T> &Q, int start, int end, SequentialVectorClass<T> &w)
   {
      /* the memory location will be checked in this function */
      return DenseMatrixComplexHessSchurDecompositionHost( *this, start, end, Q, w);
   }
   template int DenseMatrixClass<complexs>::HessSchur( DenseMatrixClass<complexs> &Q, int start, int end, SequentialVectorClass<complexs> &w);
   template int DenseMatrixClass<complexd>::HessSchur( DenseMatrixClass<complexd> &Q, int start, int end, SequentialVectorClass<complexd> &w);
   
   template <typename T>
   int DenseMatrixClass<T>::Schur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi)
   {
      int err = 0;
      T   alpha = 1.0;
      T   beta = 0.0;
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host corrently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      DenseMatrixClass<T> Q1, Q2;
      err = this->Hess(Q1); PARGEMSLR_CHKERR(err);
      err = this->HessSchur(Q2, wr, wi); PARGEMSLR_CHKERR(err);
      
      Q.MatMat( alpha, Q1, 'N', Q2, 'N', beta);
      
      /* might need to synchronize if Q, Q1, and Q2 are all on the shared memory */
      PARGEMSLR_CUDA_SYNCHRONIZE;
      
      Q1.Clear();
      Q2.Clear();
      
      return err;
      
   }
   template int DenseMatrixClass<float>::Schur( DenseMatrixClass<float> &Q, SequentialVectorClass<float> &wr, SequentialVectorClass<float> &wi);
   template int DenseMatrixClass<double>::Schur( DenseMatrixClass<double> &Q, SequentialVectorClass<double> &wr, SequentialVectorClass<double> &wi);
   
   template <typename T>
   int DenseMatrixClass<T>::Schur( DenseMatrixClass<T> &Q, int start, int end, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi)
   {
      int err = 0;
      T   alpha = 1.0;
      T   beta = 0.0;
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host corrently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      DenseMatrixClass<T> Q1, Q2;
      err = this->Hess(Q1, start, end); PARGEMSLR_CHKERR(err);
      err = this->HessSchur(Q2, start, end, wr, wi); PARGEMSLR_CHKERR(err);
      
      Q.MatMat( alpha, Q1, 'N', Q2, 'N', beta);
      
      /* might need to synchronize if Q, Q1, and Q2 are all on the shared memory */
      PARGEMSLR_CUDA_SYNCHRONIZE;
      
      Q1.Clear();
      Q2.Clear();
      
      return err;
      
   }
   template int DenseMatrixClass<float>::Schur( DenseMatrixClass<float> &Q, int start, int end, SequentialVectorClass<float> &wr, SequentialVectorClass<float> &wi);
   template int DenseMatrixClass<double>::Schur( DenseMatrixClass<double> &Q, int start, int end, SequentialVectorClass<double> &wr, SequentialVectorClass<double> &wi);
   
   template <typename T>
   int DenseMatrixClass<T>::Schur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w)
   {
      int err = 0;
      T   alpha = 1.0;
      T   beta = 0.0;
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host corrently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      DenseMatrixClass<T> Q1, Q2;
      err = this->Hess(Q1); PARGEMSLR_CHKERR(err);
      err = this->HessSchur(Q2, w); PARGEMSLR_CHKERR(err);
      
      Q.MatMat( alpha, Q1, 'N', Q2, 'N', beta);
      
      /* might need to synchronize if Q, Q1, and Q2 are all on the shared memory */
      PARGEMSLR_CUDA_SYNCHRONIZE;
      
      Q1.Clear();
      Q2.Clear();
      
      return err;
      
   }
   template int DenseMatrixClass<complexs>::Schur( DenseMatrixClass<complexs> &Q, SequentialVectorClass<complexs> &w);
   template int DenseMatrixClass<complexd>::Schur( DenseMatrixClass<complexd> &Q, SequentialVectorClass<complexd> &w);
   
   template <typename T>
   int DenseMatrixClass<T>::Schur( DenseMatrixClass<T> &Q, int start, int end, SequentialVectorClass<T> &w)
   {
      int err = 0;
      T   alpha = 1.0;
      T   beta = 0.0;
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host corrently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      DenseMatrixClass<T> Q1, Q2;
      err = this->Hess(Q1, start, end); PARGEMSLR_CHKERR(err);
      err = this->HessSchur(Q2, start, end, w); PARGEMSLR_CHKERR(err);
      
      Q.MatMat( alpha, Q1, 'N', Q2, 'N', beta);
      
      /* might need to synchronize if Q, Q1, and Q2 are all on the shared memory */
      PARGEMSLR_CUDA_SYNCHRONIZE;
      
      Q1.Clear();
      Q2.Clear();
      
      return err;
      
   }
   template int DenseMatrixClass<complexs>::Schur( DenseMatrixClass<complexs> &Q, int start, int end, SequentialVectorClass<complexs> &w);
   template int DenseMatrixClass<complexd>::Schur( DenseMatrixClass<complexd> &Q, int start, int end, SequentialVectorClass<complexd> &w);
   
   template <typename T>
   int DenseMatrixClass<T>::HessEig( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi)
   {
      return DenseMatrixRealHessEigenDecompositionHost( *this, Q, wr, wi);
   }
   template int DenseMatrixClass<float>::HessEig( DenseMatrixClass<float> &Q, SequentialVectorClass<float> &wr, SequentialVectorClass<float> &wi);
   template int DenseMatrixClass<double>::HessEig( DenseMatrixClass<double> &Q, SequentialVectorClass<double> &wr, SequentialVectorClass<double> &wi);
   
   template <typename T>
   int DenseMatrixClass<T>::HessEig( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w)
   {
      return DenseMatrixComplexHessEigenDecompositionHost( *this, Q, w);
   }
   template int DenseMatrixClass<complexs>::HessEig( DenseMatrixClass<complexs> &Q, SequentialVectorClass<complexs> &w);
   template int DenseMatrixClass<complexd>::HessEig( DenseMatrixClass<complexd> &Q, SequentialVectorClass<complexd> &w);
   
   template <typename T>
   int DenseMatrixClass<T>::Eig( DenseMatrixClass<T> &QS, DenseMatrixClass<T> &QE, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi)
   {
      int err;
      err = this->Schur( QS, wr, wi); PARGEMSLR_CHKERR(err);
      err = this->HessEig( QE, wr, wi); PARGEMSLR_CHKERR(err);
      return err;
   }
   template int DenseMatrixClass<float>::Eig( DenseMatrixClass<float> &QS, DenseMatrixClass<float> &QE, SequentialVectorClass<float> &wr, SequentialVectorClass<float> &wi);
   template int DenseMatrixClass<double>::Eig( DenseMatrixClass<double> &QS, DenseMatrixClass<double> &QE, SequentialVectorClass<double> &wr, SequentialVectorClass<double> &wi);
   
   template <typename T>
   int DenseMatrixClass<T>::Eig( DenseMatrixClass<T> &QS, DenseMatrixClass<T> &QE, SequentialVectorClass<T> &w)
   {
      int err;
      err = this->Schur( QS, w); PARGEMSLR_CHKERR(err);
      err = this->HessEig( QE, w); PARGEMSLR_CHKERR(err);
      return err;
   }
   template int DenseMatrixClass<complexs>::Eig( DenseMatrixClass<complexs> &QS, DenseMatrixClass<complexs> &QE, SequentialVectorClass<complexs> &w);
   template int DenseMatrixClass<complexd>::Eig( DenseMatrixClass<complexd> &QS, DenseMatrixClass<complexd> &QE, SequentialVectorClass<complexd> &w);
   
   template <typename T>
   int DenseMatrixClass<T>::OrdSchur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, vector_int &select)
   {
      return DenseMatrixRealOrderSchur( *this, Q, wr, wi, select);
   }
   template int DenseMatrixClass<float>::OrdSchur( DenseMatrixClass<float> &Q, SequentialVectorClass<float> &wr, SequentialVectorClass<float> &wi, vector_int &select);
   template int DenseMatrixClass<double>::OrdSchur( DenseMatrixClass<double> &Q, SequentialVectorClass<double> &wr, SequentialVectorClass<double> &wi, vector_int &select);
   
   template <typename T>
   int DenseMatrixClass<T>::OrdSchur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w, vector_int &select)
   {
      return DenseMatrixComplexOrderSchur( *this, Q, w, select);
   }
   template int DenseMatrixClass<complexs>::OrdSchur( DenseMatrixClass<complexs> &Q, SequentialVectorClass<complexs> &w, vector_int &select);
   template int DenseMatrixClass<complexd>::OrdSchur( DenseMatrixClass<complexd> &Q, SequentialVectorClass<complexd> &w, vector_int &select);
   
   template <typename T>
   int DenseMatrixClass<T>::OrdSchurClusters( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, vector_int &clusters)
   {
      return DenseMatrixRealOrderSchurClusters( *this, Q, wr, wi, clusters);
   }
   template int DenseMatrixClass<float>::OrdSchurClusters( DenseMatrixClass<float> &Q, SequentialVectorClass<float> &wr, SequentialVectorClass<float> &wi, vector_int &clusters);
   template int DenseMatrixClass<double>::OrdSchurClusters( DenseMatrixClass<double> &Q, SequentialVectorClass<double> &wr, SequentialVectorClass<double> &wi, vector_int &clusters);
   
   template <typename T>
   int DenseMatrixClass<T>::OrdSchurClusters( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w, vector_int &clusters)
   {
      return DenseMatrixComplexOrderSchurClusters( *this, Q, w, clusters);
   }
   template int DenseMatrixClass<complexs>::OrdSchurClusters( DenseMatrixClass<complexs> &Q, SequentialVectorClass<complexs> &w, vector_int &clusters);
   template int DenseMatrixClass<complexd>::OrdSchurClusters( DenseMatrixClass<complexd> &Q, SequentialVectorClass<complexd> &w, vector_int &clusters);
   
   template <typename T>
   int DenseMatrixClass<T>::Plot( int conditiona, int conditionb, int width)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Plot matrix only availiable on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      
      return DenseMatrixPlotHost( *this, conditiona, conditionb, width);
      
   }
   template int DenseMatrixClass<float>::Plot( int conditiona, int conditionb, int width);
   template int DenseMatrixClass<double>::Plot( int conditiona, int conditionb, int width);
   template int DenseMatrixClass<complexs>::Plot( int conditiona, int conditionb, int width);
   template int DenseMatrixClass<complexd>::Plot( int conditiona, int conditionb, int width);
   
}
