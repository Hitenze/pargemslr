#ifndef PARGEMSLR_COO_MATRIX_H
#define PARGEMSLR_COO_MATRIX_H

/**
 * @file coo_matrix.hpp
 * @brief COO matrix data structure.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../vectors/sequential_vector.hpp"
#include "matrix.hpp"
#include "csr_matrix.hpp"

namespace pargemslr
{
   
	/**
    * @brief   Class of COO matrices. Note that most coo matrix functions are not yet on the device.
    * @details Class of COO matrices. Note that most coo matrix functions are not yet on the device.
    */
   template <typename T>
   class CooMatrixClass: public MatrixClass<T>
   {
   private:
      
      /* variables */
      
      /**
       * @brief   Vector holding the column pointer data.
       * @details Vector holding the column pointer data.
       */
      IntVectorClass<int>        _i_vec;
      
      /**
       * @brief   Vector holding the column index data.
       * @details Vector holding the column index data.
       */
      IntVectorClass<int>        _j_vec;
      
      /**
       * @brief   Vector holding the matrix data.
       * @details Vector holding the matrix data.
       */
      SequentialVectorClass<T>   _a_vec;
      
      /**
       * @brief   The number of rows of the matrix.
       * @details The number of rows of the matrix.
       */
      int                        _nrows;
      
      /**
       * @brief   The number of columns of the matrix.
       * @details The number of columns of the matrix.
       */
      int                        _ncols;
      
      /**
       * @brief   The number of nonzeros of the matrix.
       * @details The number of nonzeros of the matrix.
       */
      int                        _nnz;
      
   public:
      
      /**
       * @brief   The constructor of CooMatrixClass.
       * @details The constructor of CooMatrixClass.
       */
      CooMatrixClass();
      
      /**
       * @brief   The copy constructor of CooMatrixClass.
       * @details The copy constructor of CooMatrixClass.
       * @param   [in]   mat The other matrix.
       */
      CooMatrixClass(const CooMatrixClass<T> &mat);
      
      /**
       * @brief   The move constructor of CooMatrixClass.
       * @details The move constructor of CooMatrixClass.
       * @param   [in]   mat The other matrix.
       */
      CooMatrixClass(CooMatrixClass<T> &&mat);
      
      /**
       * @brief   The = operator of CooMatrixClass.
       * @details The = operator of CooMatrixClass.
       * @param   [in]   mat The other matrix.
       * @return  Return this matrix.
       */
      CooMatrixClass<T>& operator= (const CooMatrixClass<T> &mat);
      
      /**
       * @brief   The = operator of CooMatrixClass.
       * @details The = operator of CooMatrixClass.
       * @param   [in]   mat The other matrix.
       * @return  Return this matrix.
       */
      CooMatrixClass<T>& operator= (CooMatrixClass<T> &&mat);
      
      /**
       * @brief   Free the current matrix.
       * @details Free the current matrix.
       * @return     Return error message.
       */
      virtual int       Clear();
      
      /**
       * @brief   The destructor of CooMatrixClass.
       * @details The destructor of CooMatrixClass.
       */
      virtual ~CooMatrixClass();
      
      /**
       * @brief   Free the current matrix, and create the new matrix using the default reserved memory size on the host.
       * @details Free the current matrix, and create the new matrix. The default reserved memory size on the host is MAX(nrow, ncol)*_coo_reserve_fact.
       * @param [in] nrows The number of rows of the new matrix.
       * @param [in] ncols The number of cols of the new matrix.
       * @return     Return the length of the matrix.
       */
      int               Setup(int nrows, int ncols);
      
      /**
       * @brief   Free the current matrix, and create the new matrix and reserve some space in the host memory.
       * @details Free the current matrix, and create the new matrix and reserve some space in the host memory.
       * @param [in] nrows The number of rows of the new matrix.
       * @param [in] ncols The number of cols of the new matrix.
       * @param [in] reserve The reserved vector length. A good estimation can avoid to many re-allocation when inserting values.
       * @return     Return the length of the matrix.
       */
      int               Setup(int nrows, int ncols, int reserve);
      
      /**
       * @brief   Free the current matrix, and create the new matrix and reserve some space in the memory.
       * @details Free the current matrix, and create the new matrix and reserve some space in the memory.
       * @param [in] nrows The number of rows of the new matrix.
       * @param [in] ncols The number of cols of the new matrix.
       * @param [in] reserve The reserved vector length. A good estimation can avoid to many re-allocation when inserting values.
       * @param [in] location The location of the memory.
       * @return     Return the length of the matrix.
       */
      int               Setup(int nrows, int ncols, int reserve, int location);
      
      /**
       * @brief   Get the data location of the matrix.
       * @details Get the data location of the matrix.
       * @return     Return the data location of the matrix.
       */
      virtual int       GetDataLocation() const;
      
      /**
       * @brief   Get the local number of rows of the matrix.
       * @details Get the local number of rows of the matrix.
       * @return     Return the local number of rows of the matrix.
       */
      virtual int       GetNumRowsLocal() const;
      
      /**
       * @brief   Get the local number of columns of the matrix.
       * @details Get the local number of columns of the matrix.
       * @return     Return the local number of columns of the matrix.
       */
      virtual int       GetNumColsLocal() const;
      
      /**
       * @brief   Get the number of nonzeros in this matrix.
       * @details Get the number of nonzeros in this matrix.
       * @return     Return the number of nonzeros in this matrix.
       */
      virtual long int  GetNumNonzeros() const;
      
      /**
       * @brief   Set the number of cols in this matrix.
       * @details Set the number of cols in this matrix.
       * @return     Return error message.
       */
      int               SetNumCols(int cols);
      
      /**
       * @brief   Set the number of nonzeros in this matrix.
       * @details Set the number of nonzeros in this matrix.
       * @return     Return error message.
       */
      int               SetNumNonzeros();
      
      /**
       * @brief   Create an indentity matrix.
       * @details Create an indentity matrix.
       * @return     Return error message.
       */
      virtual int       Eye();
      
      /**
       * @brief   Fill the matrix pattern with constant value.
       * @details Fill the matrix pattern with constant value.
       * @param   [in]   v The value to be filled.
       * @return     Return error message.
       */
      virtual int       Fill(const T &v);
      
      /**
       * @brief   Scale the matrix.
       * @details Scale the matrix.
       * @param   [in]   alpha The scale.
       * @return     Return error message.
       */
      virtual int       Scale(const T &alpha);
      
      /**
       * @brief      Move the data to another memory location.
       * @details    Move the data to another memory location.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      virtual int       MoveData( const int &location);
      
      /**
       * @brief   Insert value to the COO matrix.
       * @details Insert value to the COO matrix.
       * @param   [in]   row.
       * @param   [in]   col The col/row to be inserted into the csr/csc.
       * @param   [in]   v The value to be inserted.
       * @return     Return error message.
       */
      int               PushBack( int row, int col, T v);
      
      /**
       * @brief   Get the I pointer of the matrix.
       * @details Get the I pointer of the matrix.
       * @return  Return the data pointer.
       */
      int*              GetI() const;
      
      /**
       * @brief   Get the J pointer of the matrix.
       * @details Get the J pointer of the matrix.
       * @return  Return the data pointer.
       */
      int*              GetJ() const;
      
      /**
       * @brief   Get the data pointer of the matrix.
       * @details Get the data pointer of the matrix.
       * @return  Return the data pointer.
       */
      T*                GetData() const;
      
      /**
       * @brief   Get the reference to the I vector.
       * @details Get the reference to the I vector.
       * @return  Return the reference to the I vector.
       */
      IntVectorClass<int>&          GetIVector();
      
      /**
       * @brief   Get the reference to the I vector.
       * @details Get the reference to the I vector.
       * @return  Return the reference to the I vector.
       */
      const IntVectorClass<int>&          GetIVector() const;
      
      /**
       * @brief   Get the reference to the J vector.
       * @details Get the reference to the J vector.
       * @return  Return the reference to the J vector.
       */
      IntVectorClass<int>&          GetJVector();
      
      /**
       * @brief   Get the reference to the J vector.
       * @details Get the reference to the J vector.
       * @return  Return the reference to the J vector.
       */
      const IntVectorClass<int>&          GetJVector() const;
      
      /**
       * @brief   Get the reference to the data vector.
       * @details Get the reference to the data vector.
       * @return  Return the reference to the data vector.
       */
      SequentialVectorClass<T>&     GetDataVector();
      
      /**
       * @brief   Get the reference to the data vector.
       * @details Get the reference to the data vector.
       * @return  Return the reference to the data vector.
       */
      const SequentialVectorClass<T>&     GetDataVector() const;
      
      /**
       * @brief   Convert the matrix to csr.
       * @details Convert the matrix to csr.
       * @param   [in]    location The location of the output csr matrix.
       * @param   [out]   csrmat_out The output csr matrix.
       * @return     Return error message.
       */
      int               ToCsr(int location, CsrMatrixClass<T> &csrmat_out);
      
      /**
       * @brief   Generate Laplacian matrix, 5-pt for 2D problem and 7-pt for 3D problem.
       * @details Generate Laplacian matrix, 5-pt for 2D problem and 7-pt for 3D problem.
       * @param   [in]    nx
       * @param   [in]    ny
       * @param   [in]    nz Problem size on each dimension
       * @param   [in]    alphax
       * @param   [in]    alphay
       * @param   [in]    alphaz Alpha value on each direction
       * @param   [in]    shift The shift value on diagonal.
       * @param   [in]    rand_perturb Set to true to add extra random real pertubation to diagonal entries as abs(shift)*Rand().
       *                  We only add shift that might makes the problem harder.
       * @return     Return error message.
       */
      int               Laplacian(int nx, int ny, int nz, T alphax, T alphay, T alphaz, T shift, bool rand_perturb = false);
      
      /**
       * @brief   Generate 3D Helmholtz matrix with 7-pt on [0,1]^3. -\Delta u - w^2 u = f.
       * @details Generate 3D Helmholtz matrix with 7-pt on [0,1]^3. -\Delta u - w^2 u = f. Robin boundary condition u'-iwu = 0 is used.
       * @param   [in]    n Problem size.
       * @param   [in]    w The w value.
       * @return     Return error message.
       */
      int               Helmholtz(int n, T w);
      
      /**
       * @brief   Read matrix on the host memory, matrix market format.
       * @details Read matrix on the host memory, matrix market format.
       * @param   [in]    matfile The file.
       * @param   [in]    idxin The index base of the input file. 0-based or 1-based.
       * @return     Return error message.
       */
      int               ReadFromMMFile(const char *matfile, int idxin);
      
      /**
       * @brief   Plot the pattern of the csr matrix using gnuplot. Similar to spy in the MATLAB. Function for testing purpose.
       * @details Plot the pattern of the csr matrix using gnuplot. Similar to spy in the MATLAB. Function for testing purpose.
       * @param [in]   datafilename The filename of the temp file holding the data.
       * @param [in]   rperm The row permutation vector.
       * @param [in]   cperm The column permutation vector. Set either rperm or cperm to NULL to spy(A), otherwise spy(A(rperm,cperm)).
       * @param [in]   conditiona First condition.
       * @param [in]   conditionb Secend condition, only plot when conditiona == conditionb.
       * @return       Return error message.
       */
      int            PlotPatternGnuPlot( const char *datafilename, int *rperm, int *cperm, int conditiona, int conditionb);
      
   };
   
   typedef CooMatrixClass<float>     matrix_coo_float;
   typedef CooMatrixClass<double>    matrix_coo_double;
   typedef CooMatrixClass<complexs>  matrix_coo_complexs;
   typedef CooMatrixClass<complexd>  matrix_coo_complexd;
   template<> struct PargemslrIsComplex<CooMatrixClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<CooMatrixClass<complexd> > : public std::true_type {};
   
}

#endif
