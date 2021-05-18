#ifndef PARGEMSLR_CSR_MATRIX_H
#define PARGEMSLR_CSR_MATRIX_H

/**
 * @file csr_matrix.hpp
 * @brief CSR matrix data structure.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "matrix.hpp"

#ifdef PARGEMSLR_CUDA
#include "cusparse.h"
#endif

namespace pargemslr
{
   
	/**
    * @brief   Class of CSR/CSC matrices.
    * @details Class of CSR/CSC matrices.
    */
   template <typename T>
   class CsrMatrixClass: public MatrixClass<T>
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
      
      /**
       * @brief   Set to true if the _j_vec between _i_vec[k] to _i_vec[k+1] is sorted for all valid k value.
       * @details Set to true if the _j_vec between _i_vec[k] to _i_vec[k+1] is sorted for all valid k value.
       */
      bool                       _isrowsorted;
      
      /**
       * @brief   Set to false to only store the pattern of the matrix.
       * @details Set to false to only store the pattern of the matrix.
       */
      bool                       _isholdingdata;
      
      /**
       * @brief   Set to true for CSR matrix; set to false for CSC matrix.
       * @details Set to true for CSR matrix; set to false for CSC matrix.
       */
      bool                       _iscsr;
      
#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
      /**
       * @brief   For cusparse general spmv.
       * @details For cusparse general spmv.
       */
      cusparseSpMatDescr_t       _cusparse_mat;
#endif
#endif
      
   public:
      
      /**
       * @brief   The diagonal complex shift in some preconditioners. The preconditioner would be built on A+_diagonal_shift*I.
       * @details The diagonal complex shift in some preconditioners. The preconditioner would be built on A+_diagonal_shift*I.
       * @note    TODO: if useful, change to private.
       */
      typename std::conditional<PargemslrIsDoublePrecision<T>::value, 
                                 double, 
                                 float>::type _diagonal_shift;
      
      /**
       * @brief   The constructor of CsrMatrixClass.
       * @details The constructor of CsrMatrixClass. The default memory location is the host memory.
       */
      CsrMatrixClass();
      
      /**
       * @brief   The copy constructor of CsrMatrixClass.
       * @details The copy constructor of CsrMatrixClass.
       * @param [in] mat The target matrix.
       */
      CsrMatrixClass(const CsrMatrixClass<T> &mat);
      
      /**
       * @brief   The move constructor of CsrMatrixClass.
       * @details The move constructor of CsrMatrixClass.
       * @param [in] mat The target matrix.
       */
      CsrMatrixClass(CsrMatrixClass<T> &&mat);
      
      /**
       * @brief   The = operator of CsrMatrixClass.
       * @details The = operator of CsrMatrixClass.
       * @param [in] mat The target matrix.
       * @return     Return the matrix.
       */
      CsrMatrixClass<T>& operator= (const CsrMatrixClass<T> &mat);
      
      /**
       * @brief   The = operator of CsrMatrixClass.
       * @details The = operator of CsrMatrixClass.
       * @param [in] mat The target matrix.
       * @return     Return the matrix.
       */
      CsrMatrixClass<T>& operator= (CsrMatrixClass<T> &&mat);
      
      /**
       * @brief   The destructor of CsrMatrixClass.
       * @details The destructor of CsrMatrixClass. Simply a call to the free function.
       */
      virtual ~CsrMatrixClass();
      
      /**
       * @brief   Free the current matrix, and allocate memory to create a new matrix.
       * @details Free the current matrix, and allocate memory to create a new matrix. The default location of data is the host memory. \n
       *          If you want to use the PushBack funciton for J and A, please use other Setup function with setzero option.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number of cols.
       * @param [in] nnz The size reserved in J and A array.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols, int nnz);
      
      /**
       * @brief   Free the current matrix, and allocate memory to create a new matrix.
       * @details Free the current matrix, and allocate memory to create a new matrix.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number of cols.
       * @param [in] nnz The size reserved in J and A array.
       * @param [in] location The location of the data.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols, int nnz, int location);
      
      /**
       * @brief   Free the current matrix, and allocate memory to create a new matrix.
       * @details Free the current matrix, and allocate memory to create a new matrix. The default location of data is the host memory. \n
       *          If setzero is set to true, _i_vec[0] and the length of J and A will be set to 0, so that user can use the PushBack function for vectors.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number of cols.
       * @param [in] nnz The size reserved in J and A array.
       * @param [in] setzero If set to true, I[0] and the length of J and A are set to 0 so you can use the PushBack function.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols, int nnz, bool setzero);
      
      /**
       * @brief   Free the current matrix, and allocate memory to create a new matrix.
       * @details Free the current matrix, and allocate memory to create a new matrix. The default location of data is the host memory. \n
       *          If setzero is set to true, _i_vec[0] and the length of J and A will be set to 0, so that user can use the PushBack function for vectors.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number of cols.
       * @param [in] nnz The size reserved in J and A array.
       * @param [in] holdvalue Set to false to disable the A vector. Only save the pattern of the matrix.
       * @param [in] setzero If set to true, _i_vec[0] and the length of J and A is set to 0 so you can use the push_back function.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols, int nnz, bool holdvalue, bool setzero);
      
      /**
       * @brief   Free the current matrix, and allocate memory to create a new matrix.
       * @details Free the current matrix, and allocate memory to create a new matrix. The default location of data is the host memory. \n
       *          If setzero is set to true, _i_vec[0] and the length of J and A will be set to 0, so that user can use the PushBack function for vectors.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number of cols.
       * @param [in] nnz The size reserved in J and A array.
       * @param [in] location The location of the data.
       * @param [in] setzero If set to true, _i_vec[0] and the length of J and A is set to 0 so you can use the push_back function.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols, int nnz, int location, bool setzero);
      
      /**
       * @brief   Free the current matrix, and allocate memory to create a new matrix.
       * @details Free the current matrix, and allocate memory to create a new matrix. The default location of data is the host memory. \n
       *          If setzero is set to true, _i_vec[0] and the length of J and A will be set to 0, so that user can use the PushBack function for vectors.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number of cols.
       * @param [in] nnz The size reserved in J and A array.
       * @param [in] location The location of the data.
       * @param [in] holdvalue Set to false to disable the A vector. Only save the pattern of the matrix.
       * @param [in] setzero If set to true, _i_vec[0] and the length of J and A is set to 0 so you can use the push_back function.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero);
      
      /**
       * @brief   Free the current matrix, and allocate memory to create a new matrix.
       * @details Free the current matrix, and allocate memory to create a new matrix. The default location of data is the host memory. \n
       *          If setzero is set to true, _i_vec[0] and the length of J and A will be set to 0, so that user can use the PushBack function for vectors.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number of cols.
       * @param [in] nnz The size reserved in J and A array.
       * @param [in] location The location of the data.
       * @param [in] holdvalue Set to false to disable the A vector. Only save the pattern of the matrix.
       * @param [in] setzero If set to true, _i_vec[0] and the length of J and A is set to 0 so you can use the push_back function.
       * @param [in] iscsr Set to true for csr matrix, false for csc matrix.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols, int nnz, int location, bool holdvalue, bool setzero, bool iscsr);
      
      /**
       * @brief   Insert value at the end of the J and A vector, user need to update the I vector.
       * @details Insert value at the end of the J and A vector, user need to update the I vector. \n
       *          Currently only for host memory.
       * @param   [in]   col The col/row to be inserted into the csr/csc.
       * @param   [in]   v The value to be inserted.
       * @return     Return error message.
       */
      int            PushBack( int col, T v);
      
      /**
       * @brief   Copy data to extract a submatrix of the current matrix.
       * @details Copy data to extract a submatrix of the current matrix.
       * @param [in]  row_start The starting row of that submatrix.
       * @param [in]  col_start The starting column of that submatrix.
       * @param [in]  num_rows The number of rows.
       * @param [in]  num_cols The number fo cols.
       * @param [in]  location The location of the submatrix.
       * @param [out] csrmat_out The submatrix.
       * @return      Return error message.
       */
      int            SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, CsrMatrixClass<T> &csrmat_out);
      
      /**
       * @brief   Copy data to extract a submatrix of the current matrix.
       * @details Copy data to extract a submatrix of the current matrix. \n
       *          Each interval [ rows_start[i], rows_end[i] ) gives an interval of rows to keep; 
       *          interval [ cols_start[i], cols_end[i] ) gives an interval of columns to keep.
       * @param [in]  row_starts The starting rows of the submatrix.
       * @param [in]  col_starts The starting columns of the submatrix.
       * @param [in]  rows_ends The ending rows(excluse) of each row interval.
       * @param [in]  cols_ends The ending columns(excluse) of each column interval.
       * @param [in]  location The location of the submatrix.
       * @param [out] csrmat_out The submatrix.
       * @return      Return error message.
       */
      int            SubMatrix( vector_int &row_starts, vector_int &col_starts, vector_int &row_ends, vector_int &col_ends, int location, CsrMatrixClass<T> &csrmat_out);
      
      /**
       * @brief   Copy data to extract a submatrix of the current matrix.
       * @details Copy data to extract a submatrix of the current matrix. Rows in the array rows and columns in the array cols will be kept in the order they
       *          appear in rows and cols. Can be used to apply permutation.
       * @param [in]       rows Rows to keep.
       * @param [in]       cols Cols to keep.
       * @param [in]       location The location of the submatrix.
       * @param [out]      csrmat_out The submatrix.
       * @return     Return error message.
       */
      int            SubMatrix( vector_int &rows, vector_int &cols, int location, CsrMatrixClass<T> &csrmat_out);
      
      /**
       * @brief   Copy data to extract a submatrix of the current matrix.
       * @details Copy data to extract a submatrix of the current matrix. \n 
       *          if complement == false, Nonzero rows and columns in the array rows and cols are kept in their original order. \n
       *          if complement == true, Zero rows and columns in the array rows and cols are kept in their original order. \n
       * @param [in]       rows Rows to keep. Vector of length _nrows.
       * @param [in]       cols Cols to keep. Vector of length _nrows.
       * @param [out]      row_perm The row permutation vector from the original matrix to the new matrix.
       * @param [out]      col_perm The col permutation vector from the original matrix to the new matrix.
       * @param [in]       complement Keep zero (true) or nonzero (false) in vector rows and cols.
       * @param [in]       location The location of the submatrix.
       * @param [out]      csrmat_out The submatrix.
       * @return     Return error message.
       */
      int            SubMatrixNoPerm( vector_int &rows, vector_int &cols, vector_int &row_perm, vector_int &col_perm, bool complement, int location, CsrMatrixClass<T> &csrmat_out);
      
      /**
       * @brief   Get the connected components of A + AT, where A is the current matrix.
       * @details Get the connected components of A + AT, where A is the current matrix.
       * @param [out]      comp_indices 2D vector, comp_indices[i] is the index of the elements in component i.
       * @param [in,out]   ncomps On entry, If max_ncomps<=1, this parameter will be ignored. If max_ncomps>1, when the actual number
       *                   of connected component is greater than max_ncomps, we will merge several connected components together. \n
       *                   On exit, the number of final components (after merge).
       * @return     Return error message.
       */
      int            GetConnectedComponents(std::vector<vector_int> &comp_indices, int &ncomps);
      
      /**
       * @brief   Update the structure of a vector to have same row permutation.
       * @details Update the structure of a vector to have same row permutation.
       * @param [out] vec The target vector.
       * @return      Return error message.
       */
      int            SetupVectorPtrStr(SequentialVectorClass<T> &vec);
      
      /**
       * @brief   Free the current matrix.
       * @details Free the current matrix.
       * @return  Return error message.
       */
      virtual int    Clear();
      
      /**
       * @brief   Get the data location of the matrix.
       * @details Get the data location of the matrix.
       * @return     Return the data location of the matrix.
       */
      virtual int    GetDataLocation() const;
      
      /**
       * @brief   Get the local number of rows of the matrix.
       * @details Get the local number of rows of the matrix.
       * @return     Return the local number of rows of the matrix.
       */
      virtual int    GetNumRowsLocal() const;
      
      /**
       * @brief   Get the local number of columns of the matrix.
       * @details Get the local number of columns of the matrix.
       * @return     Return the local number of columns of the matrix.
       */
      virtual int    GetNumColsLocal() const;
      
      /**
       * @brief   Get the number of nonzeros in this matrix.
       * @details Get the number of nonzeros in this matrix.
       * @return     Return the number of nonzeros of the matrix.
       */
      virtual long int  GetNumNonzeros() const;
      
      /**
       * @brief   Set the number of nonzeros in this matrix.
       * @details Set the number of nonzeros in this matrix.
       * @return     Return the number of nonzeros in this matrix.
       */
      int            SetNumNonzeros();
      
      /**
       * @brief   Get the I pointer of the matrix.
       * @details Get the I pointer of the matrix.
       * @return  Return the I pointer.
       */
      int*           GetI() const;
      
      /**
       * @brief   Get the J pointer of the matrix.
       * @details Get the J pointer of the matrix.
       * @return  Return the J pointer.
       */
      int*           GetJ() const;
      
      /**
       * @brief   Get the data pointer of the matrix.
       * @details Get the data pointer of the matrix.
       * @return  Return the data pointer.
       */
      T*             GetData() const;
      
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
      
#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
      /**
       * @brief   For cusparse general spmv.
       * @details For cusparse general spmv.
       * @return  Return the cusparse cusparseSpMatDescr_t.
       */
      cusparseSpMatDescr_t          GetCusparseMat() const;
      
      /**
       * @brief   For cusparse general spmv.
       * @details For cusparse general spmv.
       * @param [in] cusparse_mat The cusparseSpMatDescr_t.
       * @return     Return error message.
       */
      int                           SetCusparseMat(cusparseSpMatDescr_t cusparse_mat);
#endif
#endif

      /**
       * @brief   Check if the matrix holding its own data.
       * @details Check if the matrix holding its own data.
       * @return     Return true if matrix holding data.
       */
      bool           IsHoldingData() const;
      
      /**
       * @brief   Csr or csc.
       * @details Csr or csc.
       * @return  Return true if matrix is csr.
       */
      bool&          IsCsr();
      
      /**
       * @brief   Csr or csc.
       * @details Csr or csc.
       * @return  Return true if matrix is csr.
       */
      const bool&          IsCsr() const;
      
      /**
       * @brief   Tell if the column/row indices in each row/column is sorted.
       * @details Tell if the column/row indices in each row/column is sorted.
       * @return  Return true if col/row indices are sorted.
       */
      bool&          IsRowSorted();
      
      /**
       * @brief   Tell if the column/row indices in each row/column is sorted.
       * @details Tell if the column/row indices in each row/column is sorted.
       * @return  Return reference to a bollean indicates if the col/row indices are sorted.
       */
      const bool&          IsRowSorted() const;
      
      /**
       * @brief   Sort the column/row indices in each row/column.
       * @details Sort the column/row indices in each row/column.
       * @return     Return error message.
       */
      int            SortRow();
      
      /**
       * @brief   Create an indentity matrix.
       * @details Create an indentity matrix.
       * @return     Return error message.
       */
      virtual int    Eye();
      
      /**
       * @brief   Fill the matrix with constant value.
       * @details Fill the matrix with constant value.
       * @param   [in]   v The value to be filled.
       * @return     Return error message.
       */
      virtual int    Fill(const T &v);
      
      /**
       * @brief      Move the data to another memory location.
       * @details    Move the data to another memory location.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      virtual int    MoveData( const int &location);
      
      /**
       * @brief      Scale the current csr matrix.
       * @details    Scale the current csr matrix.
       * @param [in] alpha The scale.
       * @return     Return error message.
       */
      virtual int    Scale(const T &alpha);
      
      /**
       * @brief         Get the value leads to max diagonal no.
       * @details       Scale the current csr matrix.
       * @param [in]    scale The scale.
       * @return        Return error message.
       */
      int            GetDiagScale(T &scale);
      
      /**
       * @brief      Convert between csr and csc.
       * @details    Convert between csr and csc.
       * @param [in] csr Set to true to use csr, otherwise csc.
       * @return     Return error message.
       */
      int            Convert(bool csr);
      
      /**
       * @brief      Transpost the current matrix.
       * @details    Transpost the current matrix.
       * @return     Return error message.
       */
      int            Transpose();
      
      /**
       * @brief      Transpost the current matrix.
       * @details    Transpost the current matrix.
       * @param [out]   AT The transpose matrix.
       * @return     Return error message.
       */
      int            Transpose(CsrMatrixClass<T> &AT);
      
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
      virtual int    MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y);
      
      /**
       * @brief   In place Matrix-Vector product ==>  z := alpha*A*x + beta*y, or z := alpha*A'*x + beta*y.
       * @details In place Matrix-Vector product ==>  z := alpha*A*x + beta*y, or z := alpha*A'*x + beta*y.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The first vector.
       * @param [in]       beta The beta value.
       * @param [in]       y The second vector.
       * @param [out]      z The output vector.
       * @return           Return error message.
       */
      virtual int    MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, const VectorClass<T> &y, VectorClass<T> &z);
      
      /**
       * @brief   Dense Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C where C is this matrix.
       * @details Dense Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C where C is this matrix. \n
       *          If C is not pre-allocated, C will be allocated such that it has the same location as A.
       * @param [in]       alpha The alpha value.
       * @param [in]       A The first csr matrix.
       * @param [in]       transa Whether we transpose A or not.
       * @param [in]       beta The beta value.
       * @param [in]       B The second csr matrix.
       * @param [in]       transb Whether we transpose B or not.
       * @return           Return error message.
       */
      int            MatMat( const T &alpha, const CsrMatrixClass<T> &A, char transa, const CsrMatrixClass<T> &B, char transb, const T &beta);
      
      /**
       * @brief   Plot the pattern of the csr matrix to the terminal output. Similar to spy in the MATLAB. Function for testing purpose.
       * @details Plot the pattern of the csr matrix to the terminal output. Similar to spy in the MATLAB. Function for testing purpose.
       * @param [in]   rperm The row permutation vector.
       * @param [in]   cperm The column permutation vector. Set either rperm or cperm to NULL to spy(A), otherwise spy(A(rperm,cperm)).
       * @param [in]   conditiona First condition.
       * @param [in]   conditionb Secend condition, only spy when conditiona == conditionb.
       * @return       Return error message.
       */
      int            PlotPattern( int *rperm, int *cperm, int conditiona, int conditionb);
      
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
      
      /**
       * @brief   Plot the csr matrix to the terminal output. Function for testing purpose.
       * @details Plot the csr matrix to the terminal output. Function for testing purpose.
       * @param [in]   perm The permutation vector. Set to NULL to plot(A), otherwise plot(A(perm,perm)).
       * @param [in]   conditiona First condition.
       * @param [in]   conditionb Secend condition, only plot when conditiona == conditionb.
       * @param [in]   width The plot width.
       * @return       Return error message.
       */
      int            Plot( int *perm, int conditiona, int conditionb, int width);
      
      /**
       * @brief   Generate Laplacian matrix on the host memory, 5-pt for 2D problem and 7-pt for 3D problem.
       * @details Generate Laplacian matrix on the host memory, 5-pt for 2D problem and 7-pt for 3D problem.
       * @param   [in]    nx
       * @param   [in]    ny
       * @param   [in]    nz Problem size on each dimension
       * @param   [in]    alphax
       * @param   [in]    alphay
       * @param   [in]    alphaz Alpha value on each direction
       * @param   [in]    shift The shift value on diagonal
       * @param   [in]    rand_perturb Set to true to add extra random real pertubation to diagonal entries as abs(shift)*Rand().
       *                  We only add shift that might makes the problem harder
       * @return     Return error message.
       */
      int            Laplacian(int nx, int ny, int nz, T alphax, T alphay, T alphaz, T shift, bool rand_perturb = false);
      
      /**
       * @brief   Generate 3D Helmholtz matrix with 7-pt on [0,1]^3. -\Delta u - w^2 u = 0.
       * @details Generate 3D Helmholtz matrix with 7-pt on [0,1]^3. -\Delta u - w^2 u = 0. Robin boundary condition u'-iwu = 0 is used.
       * @param   [in]    n Problem size.
       * @param   [in]    w The w value.
       * @return     Return error message.
       */
      int            Helmholtz(int n, T w);
      
      /**
       * @brief   Read matrix on the host memory, matrix market format.
       * @details Read matrix on the host memory, matrix market format.
       * @param   [in]    matfile The file.
       * @param   [in]    idxin The index base of the input file. 0-based or 1-based.
       * @return     Return error message.
       */
      int            ReadFromMMFile(const char *matfile, int idxin);
      
      /**
       * @brief   Set the diagonal complex shift for some preconditioner options.
       * @details Set the diagonal complex shift for some preconditioner options.
       * @param   [in]    diagonal_shift The complex shift will be 0+shifti.
       * @return     Return error message.
       */
      template <typename T1>
      int            SetComplexShift(T1 diagonal_shift)
      {
         this->_diagonal_shift = diagonal_shift;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Get the diagonal complex shift for some preconditioner options.
       * @details Get the diagonal complex shift for some preconditioner options.
       * @param   [out]    diagonal_shift The complex shift.
       * @return     Return error message.
       */
      template <typename T1>
      int            GetComplexShift(T1 &diagonal_shift);
      
      /**
       * @brief   Get the diagonal complex shift for some preconditioner options.
       * @details Get the diagonal complex shift for some preconditioner options.
       * @param   [out]    diagonal_shift The complex shift.
       * @return     Return error message.
       */
      template <typename T1>
      int            GetComplexShift(ComplexValueClass<T1> &diagonal_shift);
      
      /**
       * @brief   Get comm, np, and myid. Get the sequential comm.
       * @details Get comm, np, and myid. Get the sequential comm.
       * @param   [in]        np The number of processors.
       * @param   [in]        myid The local MPI rank number.
       * @param   [in]        comm The MPI_Comm.
       * @return     Return error message.
       */
      int            GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const
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
   
   typedef CsrMatrixClass<float>     matrix_csr_float;
   typedef CsrMatrixClass<double>    matrix_csr_double;
   typedef CsrMatrixClass<complexs>  matrix_csr_complexs;
   typedef CsrMatrixClass<complexd>  matrix_csr_complexd;
   template<> struct PargemslrIsComplex<CsrMatrixClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<CsrMatrixClass<complexd> > : public std::true_type {};
   
}

#endif
