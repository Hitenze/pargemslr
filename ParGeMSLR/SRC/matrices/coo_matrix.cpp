
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "matrix.hpp"
#include "coo_matrix.hpp"
#include "csr_matrix.hpp"
#include "matrixops.hpp"
#include <iomanip>

#include <cstring>
#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

namespace pargemslr
{
   
   template <typename T>
   CooMatrixClass<T>::CooMatrixClass()
   {
      this->_i_vec.Clear();
      this->_j_vec.Clear();
      this->_a_vec.Clear();
      this->_nrows = 0;
      this->_ncols = 0;
      this->_nnz = 0;
   }
   template CooMatrixClass<float>::CooMatrixClass();
   template CooMatrixClass<double>::CooMatrixClass();
   template CooMatrixClass<complexs>::CooMatrixClass();
   template CooMatrixClass<complexd>::CooMatrixClass();
   
   template <typename T>
   CooMatrixClass<T>::CooMatrixClass(const CooMatrixClass<T> &mat) : MatrixClass<T>(mat)
   {
      this->_nrows = mat._nrows;
      this->_ncols = mat._ncols;
      this->_nnz = mat._nnz;
      this->_i_vec = mat._i_vec;
      this->_j_vec = mat._j_vec;
      this->_a_vec = mat._a_vec;
   }
   template CooMatrixClass<float>::CooMatrixClass(const CooMatrixClass<float> &mat);
   template CooMatrixClass<double>::CooMatrixClass(const CooMatrixClass<double> &mat);
   template CooMatrixClass<complexs>::CooMatrixClass(const CooMatrixClass<complexs> &mat);
   template CooMatrixClass<complexd>::CooMatrixClass(const CooMatrixClass<complexd> &mat);
   
   template <typename T>
   CooMatrixClass<T>::CooMatrixClass( CooMatrixClass<T> &&mat) : MatrixClass<T>(std::move(mat))
   {
      this->_nrows = mat._nrows;
      mat._nrows = 0;
      this->_ncols = mat._ncols;
      mat._ncols = 0;
      this->_nnz = mat._nnz;
      mat._nnz = 0;
      this->_i_vec = std::move(mat._i_vec);
      this->_j_vec = std::move(mat._j_vec);
      this->_a_vec = std::move(mat._a_vec);
   }
   template CooMatrixClass<float>::CooMatrixClass( CooMatrixClass<float> &&mat);
   template CooMatrixClass<double>::CooMatrixClass( CooMatrixClass<double> &&mat);
   template CooMatrixClass<complexs>::CooMatrixClass( CooMatrixClass<complexs> &&mat);
   template CooMatrixClass<complexd>::CooMatrixClass( CooMatrixClass<complexd> &&mat);
   
   template <typename T>
   CooMatrixClass<T>& CooMatrixClass<T>::operator= (const CooMatrixClass<T> &mat)
   {
      this->Clear();
      ParallelLogClass::operator=(mat);
      this->_nrows = mat._nrows;
      this->_ncols = mat._ncols;
      this->_nnz = mat._nnz;
      this->_i_vec = mat._i_vec;
      this->_j_vec = mat._j_vec;
      this->_a_vec = mat._a_vec;
      return *this;
   }
   template CooMatrixClass<float>& CooMatrixClass<float>::operator= (const CooMatrixClass<float> &mat);
   template CooMatrixClass<double>& CooMatrixClass<double>::operator= (const CooMatrixClass<double> &mat);
   template CooMatrixClass<complexs>& CooMatrixClass<complexs>::operator= (const CooMatrixClass<complexs> &mat);
   template CooMatrixClass<complexd>& CooMatrixClass<complexd>::operator= (const CooMatrixClass<complexd> &mat);
   
   template <typename T>
   CooMatrixClass<T>& CooMatrixClass<T>::operator= ( CooMatrixClass<T> &&mat)
   {
      this->Clear();
      ParallelLogClass::operator=(std::move(mat));
      this->_nrows = mat._nrows;
      mat._nrows = 0;
      this->_ncols = mat._ncols;
      mat._ncols = 0;
      this->_nnz = mat._nnz;
      mat._nnz = 0;
      this->_i_vec = std::move(mat._i_vec);
      this->_j_vec = std::move(mat._j_vec);
      this->_a_vec = std::move(mat._a_vec);
      return *this;
   }
   template CooMatrixClass<float>& CooMatrixClass<float>::operator= ( CooMatrixClass<float> &&mat);
   template CooMatrixClass<double>& CooMatrixClass<double>::operator= ( CooMatrixClass<double> &&mat);
   template CooMatrixClass<complexs>& CooMatrixClass<complexs>::operator= ( CooMatrixClass<complexs> &&mat);
   template CooMatrixClass<complexd>& CooMatrixClass<complexd>::operator= ( CooMatrixClass<complexd> &&mat);
   
   template <typename T>
   CooMatrixClass<T>::~CooMatrixClass()
   {
      this->Clear();
   }
   template CooMatrixClass<float>::~CooMatrixClass();
   template CooMatrixClass<double>::~CooMatrixClass();
   template CooMatrixClass<complexs>::~CooMatrixClass();
   template CooMatrixClass<complexd>::~CooMatrixClass();
   
   template <typename T>
   int CooMatrixClass<T>::Clear()
   {
      /* base class clear */
      MatrixClass<T>::Clear();
      
      this->_i_vec.Clear();
      this->_j_vec.Clear();
      this->_a_vec.Clear();
      this->_nrows = 0;
      this->_ncols = 0;
      this->_nnz = 0;
      return PARGEMSLR_SUCCESS;
   }
   template int CooMatrixClass<float>::Clear();
   template int CooMatrixClass<double>::Clear();
   template int CooMatrixClass<complexs>::Clear();
   template int CooMatrixClass<complexd>::Clear();
   
   template <typename T>
   int CooMatrixClass<T>::Setup(int nrows, int ncols)
   {
      return this->Setup( nrows, ncols, PargemslrMax( nrows, ncols)*pargemslr::pargemslr_global::_coo_reserve_fact, kMemoryHost);
   }
   template int CooMatrixClass<float>::Setup(int nrows, int ncols);
   template int CooMatrixClass<double>::Setup(int nrows, int ncols);
   template int CooMatrixClass<complexs>::Setup(int nrows, int ncols);
   template int CooMatrixClass<complexd>::Setup(int nrows, int ncols);
   
   template <typename T>
   int CooMatrixClass<T>::Setup(int nrows, int ncols, int reserve)
   {
      return this->Setup( nrows, ncols, reserve, kMemoryHost);
   }
   template int CooMatrixClass<float>::Setup(int nrows, int ncols, int reserve);
   template int CooMatrixClass<double>::Setup(int nrows, int ncols, int reserve);
   template int CooMatrixClass<complexs>::Setup(int nrows, int ncols, int reserve);
   template int CooMatrixClass<complexd>::Setup(int nrows, int ncols, int reserve);
   
   template <typename T>
   int CooMatrixClass<T>::Setup(int nrows, int ncols, int reserve, int location)
   {
      PARGEMSLR_CHKERR(nrows<0);
      PARGEMSLR_CHKERR(ncols<0);
      
      if(nrows == 0 || ncols == 0)
      {
         this->_nrows = nrows;
         this->_ncols = ncols;
         this->_nnz = 0;
         return PARGEMSLR_SUCCESS;
      }
      
      this->_i_vec.Setup(0, reserve, location, false);
      this->_j_vec.Setup(0, reserve, location, false);
      this->_a_vec.Setup(0, reserve, location, false);
      
      this->_nrows = nrows;
      this->_ncols = ncols;
      this->_nnz = 0;
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int CooMatrixClass<float>::Setup(int nrows, int ncols, int reserve, int location);
   template int CooMatrixClass<double>::Setup(int nrows, int ncols, int reserve, int location);
   template int CooMatrixClass<complexs>::Setup(int nrows, int ncols, int reserve, int location);
   template int CooMatrixClass<complexd>::Setup(int nrows, int ncols, int reserve, int location);
   
   template <typename T>
   int CooMatrixClass<T>::PushBack( int row, int col, T v)
   {
      
      PARGEMSLR_CHKERR( row < 0 || row >= this->_nrows);
      //PARGEMSLR_CHKERR( col < 0 || col >= this->_ncols);
      if(col < 0 || col >= this->_ncols)
      {
         printf("COO pushback error at: %d %d\n",col, this->_ncols);
         PARGEMSLR_CHKERR( col < 0 || col >= this->_ncols);
      }

#ifdef PARGEMSLR_CUDA
      if( this->GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Warning: PushBack function is not availiabe for the device memory, moving data to the host."<<std::endl;
         this->MoveData(kMemoryHost);
      }
#endif      
      this->_i_vec.PushBack(row);
      this->_j_vec.PushBack(col);
      this->_a_vec.PushBack(v);
      
      this->_nnz ++;
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int CooMatrixClass<float>::PushBack( int row, int col, float v);
   template int CooMatrixClass<double>::PushBack( int row, int col, double v);
   template int CooMatrixClass<complexs>::PushBack( int row, int col, complexs v);
   template int CooMatrixClass<complexd>::PushBack( int row, int col, complexd v);
   
   /**
    * @brief   Get the data location of the matrix.
    * @details Get the data location of the matrix.
    * @return     Return the length of the vector.
    */
   template <typename T>
   int CooMatrixClass<T>::GetDataLocation() const
   {
      PARGEMSLR_CHKERR( this->_i_vec.GetDataLocation() != this->_j_vec.GetDataLocation() );
      PARGEMSLR_CHKERR( this->_i_vec.GetDataLocation() != this->_a_vec.GetDataLocation() );
      
      return this->_i_vec.GetDataLocation();
   }
   template int CooMatrixClass<float>::GetDataLocation() const;
   template int CooMatrixClass<double>::GetDataLocation() const;
   template int CooMatrixClass<complexs>::GetDataLocation() const;
   template int CooMatrixClass<complexd>::GetDataLocation() const;
   
   /**
    * @brief   Get the local number of rows of the matrix.
    * @details Get the local number of rows of the matrix.
    * @return     Return the length of the matrix.
    */
   template <typename T>
   int CooMatrixClass<T>::GetNumRowsLocal() const
   {
      return this->_nrows;
   }
   template int CooMatrixClass<float>::GetNumRowsLocal() const;
   template int CooMatrixClass<double>::GetNumRowsLocal() const;
   template int CooMatrixClass<complexs>::GetNumRowsLocal() const;
   template int CooMatrixClass<complexd>::GetNumRowsLocal() const;
   
   /**
    * @brief   Get the local number of columns of the matrix.
    * @details Get the local number of columns of the matrix.
    * @return     Return the length of the matrix.
    */
   template <typename T>
   int CooMatrixClass<T>::GetNumColsLocal() const
   {
      return this->_ncols;
   }
   template int CooMatrixClass<float>::GetNumColsLocal() const;
   template int CooMatrixClass<double>::GetNumColsLocal() const;
   template int CooMatrixClass<complexs>::GetNumColsLocal() const;
   template int CooMatrixClass<complexd>::GetNumColsLocal() const;
   
   template <typename T>
   long int CooMatrixClass<T>::GetNumNonzeros() const
   {
      return (long int)this->_nnz;
   }
   template long int CooMatrixClass<float>::GetNumNonzeros() const;
   template long int CooMatrixClass<double>::GetNumNonzeros() const;
   template long int CooMatrixClass<complexs>::GetNumNonzeros() const;
   template long int CooMatrixClass<complexd>::GetNumNonzeros() const;
   
   template <typename T>
   int CooMatrixClass<T>::SetNumNonzeros()
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Warning: set nnz only work on the host, moving matrix to the host."<<std::endl;
         this->MoveData(kMemoryHost);
      }
      
      this->_nnz = this->_i_vec.GetLengthLocal();
      
      return PARGEMSLR_SUCCESS;
   }
   template int CooMatrixClass<float>::SetNumNonzeros();
   template int CooMatrixClass<double>::SetNumNonzeros();
   template int CooMatrixClass<complexs>::SetNumNonzeros();
   template int CooMatrixClass<complexd>::SetNumNonzeros();
   
   template <typename T>
   int CooMatrixClass<T>::SetNumCols(int cols)
   {
      PARGEMSLR_CHKERR(cols < 0);
      
      this->_ncols = cols;
      
      return PARGEMSLR_SUCCESS;
   }
   template int CooMatrixClass<float>::SetNumCols(int cols);
   template int CooMatrixClass<double>::SetNumCols(int cols);
   template int CooMatrixClass<complexs>::SetNumCols(int cols);
   template int CooMatrixClass<complexd>::SetNumCols(int cols);
   
   template <typename T>
   int* CooMatrixClass<T>::GetI() const
   {
      return this->_i_vec.GetData();
   }
   template int* CooMatrixClass<float>::GetI() const;
   template int* CooMatrixClass<double>::GetI() const;
   template int* CooMatrixClass<complexs>::GetI() const;
   template int* CooMatrixClass<complexd>::GetI() const;
   
   template <typename T>
   int* CooMatrixClass<T>::GetJ() const
   {
      return this->_j_vec.GetData();
   }
   template int* CooMatrixClass<float>::GetJ() const;
   template int* CooMatrixClass<double>::GetJ() const;
   template int* CooMatrixClass<complexs>::GetJ() const;
   template int* CooMatrixClass<complexd>::GetJ() const;
   
   template <typename T>
   T* CooMatrixClass<T>::GetData() const
   {
      return this->_a_vec.GetData();
   }
   template float* CooMatrixClass<float>::GetData() const;
   template double* CooMatrixClass<double>::GetData() const;
   template complexs* CooMatrixClass<complexs>::GetData() const;
   template complexd* CooMatrixClass<complexd>::GetData() const;
   
   template <typename T>
   IntVectorClass<int>& CooMatrixClass<T>::GetIVector()
   {
      return this->_i_vec;
   }
   template IntVectorClass<int>& CooMatrixClass<float>::GetIVector();
   template IntVectorClass<int>& CooMatrixClass<double>::GetIVector();
   template IntVectorClass<int>& CooMatrixClass<complexs>::GetIVector();
   template IntVectorClass<int>& CooMatrixClass<complexd>::GetIVector();
   
   template <typename T>
   const IntVectorClass<int>& CooMatrixClass<T>::GetIVector() const
   {
      return this->_i_vec;
   }
   template const IntVectorClass<int>& CooMatrixClass<float>::GetIVector() const;
   template const IntVectorClass<int>& CooMatrixClass<double>::GetIVector() const;
   template const IntVectorClass<int>& CooMatrixClass<complexs>::GetIVector() const;
   template const IntVectorClass<int>& CooMatrixClass<complexd>::GetIVector() const;
   
   template <typename T>
   IntVectorClass<int>& CooMatrixClass<T>::GetJVector()
   {
      return this->_j_vec;
   }
   template IntVectorClass<int>& CooMatrixClass<float>::GetJVector();
   template IntVectorClass<int>& CooMatrixClass<double>::GetJVector();
   template IntVectorClass<int>& CooMatrixClass<complexs>::GetJVector();
   template IntVectorClass<int>& CooMatrixClass<complexd>::GetJVector();
   
   
   template <typename T>
   const IntVectorClass<int>& CooMatrixClass<T>::GetJVector() const
   {
      return this->_j_vec;
   }
   template const IntVectorClass<int>& CooMatrixClass<float>::GetJVector() const;
   template const IntVectorClass<int>& CooMatrixClass<double>::GetJVector() const;
   template const IntVectorClass<int>& CooMatrixClass<complexs>::GetJVector() const;
   template const IntVectorClass<int>& CooMatrixClass<complexd>::GetJVector() const;
   
   template <typename T>
   SequentialVectorClass<T>& CooMatrixClass<T>::GetDataVector()
   {
      return this->_a_vec;
   }
   template SequentialVectorClass<float>& CooMatrixClass<float>::GetDataVector();
   template SequentialVectorClass<double>& CooMatrixClass<double>::GetDataVector();
   template SequentialVectorClass<complexs>& CooMatrixClass<complexs>::GetDataVector();
   template SequentialVectorClass<complexd>& CooMatrixClass<complexd>::GetDataVector();
   
   
   template <typename T>
   const SequentialVectorClass<T>& CooMatrixClass<T>::GetDataVector() const
   {
      return this->_a_vec;
   }
   template const SequentialVectorClass<float>& CooMatrixClass<float>::GetDataVector() const;
   template const SequentialVectorClass<double>& CooMatrixClass<double>::GetDataVector() const;
   template const SequentialVectorClass<complexs>& CooMatrixClass<complexs>::GetDataVector() const;
   template const SequentialVectorClass<complexd>& CooMatrixClass<complexd>::GetDataVector() const;
   
#ifndef PARGEMSLR_CUDA
   
   template <typename T>
   int CooMatrixClass<T>::Eye()
   {
      /* don't need to worry for the host version */
      int   n, err = 0;
      
      PARGEMSLR_CHKERR( this->_ncols != this->_nrows);
      
      if(this->_ncols == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      n = this->_ncols;
      
      this->_nnz = n;
      
      /* create I, J, and A */
      err = this->_j_vec.Setup(n); PARGEMSLR_CHKERR(err);
      err = this->_i_vec.UnitPerm(); PARGEMSLR_CHKERR(err);
      err = this->_j_vec.Setup(n); PARGEMSLR_CHKERR(err);
      err = this->_j_vec.UnitPerm(); PARGEMSLR_CHKERR(err);
      err = this->_a_vec.Setup(n); PARGEMSLR_CHKERR(err);
      err = this->_a_vec.Fill(1.0); PARGEMSLR_CHKERR(err);
      
      return err;
      
   }
   template int CooMatrixClass<float>::Eye();
   template int CooMatrixClass<double>::Eye();
   template int CooMatrixClass<complexs>::Eye();
   template int CooMatrixClass<complexd>::Eye();
   
   template <typename T>
   int CooMatrixClass<T>::MoveData( const int &location)
   {
      return PARGEMSLR_SUCCESS;
   }
   template int CooMatrixClass<float>::MoveData( const int &location);
   template int CooMatrixClass<double>::MoveData( const int &location);
   template int CooMatrixClass<complexs>::MoveData( const int &location);
   template int CooMatrixClass<complexd>::MoveData( const int &location);
   
#endif
   
   template <typename T>
   int CooMatrixClass<T>::Fill(const T &v)
   {
      int err = 0;
      
      err = this->_a_vec.Fill(v); PARGEMSLR_CHKERR(err);
      
      return err;
   }
   template int CooMatrixClass<float>::Fill(const float &v);
   template int CooMatrixClass<double>::Fill(const double &v);
   template int CooMatrixClass<complexs>::Fill(const complexs &v);
   template int CooMatrixClass<complexd>::Fill(const complexd &v);
   
   template <typename T>
   int CooMatrixClass<T>::Scale(const T &alpha)
   {
      
      this->_a_vec.Scale(alpha);
      
      return PARGEMSLR_SUCCESS;
   }
   template int CooMatrixClass<float>::Scale(const float &alpha);
   template int CooMatrixClass<double>::Scale(const double &alpha);
   template int CooMatrixClass<complexs>::Scale(const complexs &alpha);
   template int CooMatrixClass<complexd>::Scale(const complexd &alpha);
   
   template <typename T>
   int CooMatrixClass<T>::ToCsr(int location, CsrMatrixClass<T> &csrmat_out)
   {
      
      if(location == kMemoryDevice)
      {
         std::cout<<"Error: to csr function is currently not availiable for the device memroy as output."<<std::endl;
         return PARGEMSLR_ERROR_INVALED_OPTION;
      }
      
      if(this->GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Warning: to csr function is currently not availiable for the device memroy as input, moving data to the host."<<std::endl;
         this->MoveData(kMemoryHost);
         return PARGEMSLR_ERROR_INVALED_OPTION;
      }
      
      int nrows, ncols, nnz;
      nrows = this->_nrows;
      ncols = this->_ncols;
      nnz = this->_nnz;
      
      csrmat_out.Setup( nrows, ncols, nnz, location, false);
      
      if(nrows == 0 || ncols == 0 || nnz == 0)
      {
         csrmat_out.GetIVector().Fill(0);
         return PARGEMSLR_SUCCESS;
      }
      
      CooMatrixP2CsrMatrixPHost<0,0>( nrows, ncols, nnz, this->GetData(), this->GetJ(), this->GetI(), csrmat_out.GetData(), csrmat_out.GetJ(), csrmat_out.GetI());
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int CooMatrixClass<float>::ToCsr(int location, CsrMatrixClass<float> &csrmat_out);
   template int CooMatrixClass<double>::ToCsr(int location, CsrMatrixClass<double> &csrmat_out);
   template int CooMatrixClass<complexs>::ToCsr(int location, CsrMatrixClass<complexs> &csrmat_out);
   template int CooMatrixClass<complexd>::ToCsr(int location, CsrMatrixClass<complexd> &csrmat_out);
   
   template <typename T>
   int CooMatrixClass<T>::Laplacian(int nx, int ny, int nz, T alphax, T alphay, T alphaz, T shift, bool rand_perturb)
   {
      
      typedef typename std::conditional<PargemslrIsDoublePrecision<T>::value, double, float>::type RealDataType;
      
      int            n = nx * ny * nz;
      int            i, j, k, ii, jj;
      T              v, vxp, vxn, vyp, vyn, vzp, vzn, vd;
      RealDataType   ptrb, shift_abs;
      
      /* zero matrix if empty */
      if(n == 0)
      {
         this->Setup( 0,0);
         return PARGEMSLR_SUCCESS;
      }
      
      /* check if 2d */
      int numones = 0;
      
      if(nx == 1)
      {
         numones++;
      }
      
      if(ny == 1)
      {
         numones++;
      }
      
      if(nz == 1)
      {
         numones++;
      }
      
      bool is2d;
      
      is2d = numones > 0 ? true : false;
      
      if(numones == 2)
      {
         std::cout<<"Warning: generating a 1D Laplacian problem"<<std::endl;
      }
      
      v = -1.0;
      vxp = v-alphax;
      vxn = v+alphax;
      vyp = v-alphay;
      vyn = v+alphay;
      vzp = v-alphaz;
      vzn = v+alphaz;
      
      vd = is2d ? 4.0 : 6.0;
      vd -= shift;
      
      shift_abs = PargemslrAbs(shift);
      
      if(is2d)
      {
         this->Setup( n, n, 5*n);
      }
      else
      {
         this->Setup( n, n, 7*n);
      }
      
      for ( ii = 0; ii < n; ii++) 
      {
         v = -1.0;
         k = ii / (nx*ny);
         i = (ii - k*nx*ny) / nx;
         j = ii - k*nx*ny - i*nx;

         if (k > 0) 
         {
            jj = ii - nx * ny;
            this->PushBack(ii, jj, vzn);
         }
         if (k < nz-1) 
         {
            jj = ii + nx * ny;
            this->PushBack(ii, jj, vzp);
         }

         if (i > 0) 
         {
            jj = ii - nx; 
            this->PushBack(ii, jj, vyn);
         }
         
         if (i < ny-1) 
         {
            jj = ii + nx;
            this->PushBack(ii, jj, vyp);
         }

         if (j > 0) 
         { 
            jj = ii - 1;
            this->PushBack(ii, jj, vxn);
         }
         
         if (j < nx-1) 
         {
            jj = ii + 1;
            this->PushBack(ii, jj, vxp);
         }
         
         if(rand_perturb)
         {
            PargemslrValueRandHost(ptrb);
            this->PushBack(ii, ii, vd - shift_abs*ptrb);
         }
         else
         {
            this->PushBack(ii, ii, vd);
         }
         
      }

      return PARGEMSLR_SUCCESS;
   }
   template int CooMatrixClass<float>::Laplacian(int nx, int ny, int nz, float alphax, float alphay, float alphaz, float shift, bool rand_perturb);
   template int CooMatrixClass<double>::Laplacian(int nx, int ny, int nz, double alphax, double alphay, double alphaz, double shift, bool rand_perturb);
   template int CooMatrixClass<complexs>::Laplacian(int nx, int ny, int nz, complexs alphax, complexs alphay, complexs alphaz, complexs shift, bool rand_perturb);
   template int CooMatrixClass<complexd>::Laplacian(int nx, int ny, int nz, complexd alphax, complexd alphay, complexd alphaz, complexd shift, bool rand_perturb);
   
   template <typename T>
   int CooMatrixClass<T>::Helmholtz(int n, T w)
   {
      
      int   n3 = n * n * n;
      int   i, j, k, ii, jj;
      T     v, vd, h, vzp, vzm, vxp, vxm, vyp, vym, c;
      
      /* zero matrix if empty */
      if(n < 1)
      {
         this->Setup( 0, 0);
         return PARGEMSLR_SUCCESS;
      }
      
      /* this is the grid size */
      h = T(1.0)/T(n-1);
      
      vd = T(6.0) - w*w*h*h;
      c = T(0.0,2.0)*h*w;
      
      this->Setup( n3, n3, 7*n3);
      
      /* The boundary condition is u' = iwu.
       * x=0 -> n = [-1 0 0], -ux = iwu -> u[-1]=u[1]+2ihwu[0]
       * x=1 -> n = [ 1 0 0],  ux = iwu -> u[1]=u[-1]+2ihwu[0]
       * y=0 -> n = [0 -1 0], -uy = iwu
       * y=1 -> n = [0  1 0],  uy = iwu
       * z=0 -> n = [0 0 -1], -uz = iwu
       * z=1 -> n = [0 0  1],  uz = iwu
       */
      
      for ( ii = 0; ii < n3; ii++) 
      {
         k = ii / (n*n);
         i = (ii - k*n*n) / n;
         j = ii - k*n*n - i*n;
         
         /* init values */
         v = vd;
         vzp = T(-1.0);
         vyp = T(-1.0);
         vxp = T(-1.0);
         vzm = T(-1.0);
         vym = T(-1.0);
         vxm = T(-1.0);
         
         if(k == 0)
         {
            /* z == 0 */
            vzp = T(-2.0);
            v = v - c;
         }
         if(k == n-1)
         {
            /* z == 1 */
            vzm = T(-2.0);
            v = v - c;
         }
         if(j == 0)
         {
            /* y == 0 */
            vyp = T(-2.0);
            v = v - c;
         }
         if(j == n-1)
         {
            /* y == 1 */
            vym = T(-2.0);
            v = v - c;
         }
         if(i == 0)
         {
            /* x == 0 */
            vxp = T(-2.0);
            v = v - c;
         }
         if(i == n-1)
         {
            /* x == 1 */
            vxm = T(-2.0);
            v = v - c;
         }
         
         if (k > 0) 
         {
            jj = ii - n * n;
            this->PushBack(ii, jj, vzm);
         }
         if (k < n-1) 
         {
            jj = ii + n * n;
            this->PushBack(ii, jj, vzp);
         }

         if (i > 0) 
         {
            jj = ii - n; 
            this->PushBack(ii, jj, vxm);
         }
         
         if (i < n-1) 
         {
            jj = ii + n;
            this->PushBack(ii, jj, vxp);
         }

         if (j > 0) 
         { 
            jj = ii - 1;
            this->PushBack(ii, jj, vym);
         }
         
         if (j < n-1) 
         {
            jj = ii + 1;
            this->PushBack(ii, jj, vyp);
         }
         
         this->PushBack(ii, ii, v);
         
      }

      return PARGEMSLR_SUCCESS;
   }
   template int CooMatrixClass<complexs>::Helmholtz(int n, complexs w);
   template int CooMatrixClass<complexd>::Helmholtz(int n, complexd w);
   
   template <typename T>
   int CooMatrixClass<T>::ReadFromMMFile(const char *matfile, int idxin)
   {
      return CooMatrixReadFromFile( *this, matfile, idxin, 0);
   }
   template int CooMatrixClass<float>::ReadFromMMFile(const char *matfile, int idxin);
   template int CooMatrixClass<double>::ReadFromMMFile(const char *matfile, int idxin);
   template int CooMatrixClass<complexs>::ReadFromMMFile(const char *matfile, int idxin);
   template int CooMatrixClass<complexd>::ReadFromMMFile(const char *matfile, int idxin);
   
   template <typename T>
   int CooMatrixClass<T>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Plot Pattern only works on host.");
      }
      
      int i;
      
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
            for(i = 0 ; i < _nnz ; i ++)
            {
               fprintf(fdata, "%d %d \n", this->_j_vec[i]+1, _nrows-this->_i_vec[i]+1);
            }
         }
         else
         {
            vector_int rrperm, rcperm;
            rrperm.Setup(_nrows);
            for(i = 0 ; i < _nrows ; i ++)
            {
               rrperm[rperm[i]] = i;
            }
            rcperm.Setup(_ncols);
            for(i = 0 ; i < _ncols ; i ++)
            {
               rcperm[cperm[i]] = i;
            }
            for(i = 0 ; i < _nnz ; i ++)
            {
               fprintf(fdata, "%d %d \n", rcperm[this->_j_vec[i]]+1, _nrows-rrperm[this->_i_vec[i]]+1);
            }
            rrperm.Clear();
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
   template int CooMatrixClass<float>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb);
   template int CooMatrixClass<double>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb);
   template int CooMatrixClass<complexs>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb);
   template int CooMatrixClass<complexd>::PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb);
   
}
