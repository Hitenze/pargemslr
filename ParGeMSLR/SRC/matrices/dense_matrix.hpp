#ifndef PARGEMSLR_DENSE_MATRIX_H
#define PARGEMSLR_DENSE_MATRIX_H

/**
 * @file dense_matrix.hpp
 * @brief Dense matrix data structure.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../vectors/sequential_vector.hpp"
#include "matrix.hpp"

namespace pargemslr
{
   
   /**
    * @brief   Class of dense matrix.
    * @details Class of dense matrix.
    */
   template <typename T>
   class DenseMatrixClass: public MatrixClass<T>
   {
   private:
      
      /* variables */
      
      /**
       * @brief   Vector holding the data, in the BLAS order.
       * @details Vector holding the data, in the BLAS order.
       */
      SequentialVectorClass<T>   _data_vec;
      
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
       * @brief   The leading dimension of the matrix.
       * @details The leading dimension of the matrix.
       */
      int                        _ldim;
      
      
   public:
      
      /**
       * @brief   The constructor of DenseMatrixClass.
       * @details The constructor of DenseMatrixClass. The default memory location is the host memory.
       */
      DenseMatrixClass();
      
      /**
       * @brief   The copy constructor of DenseMatrixClass.
       * @details The copy constructor of DenseMatrixClass.
       * @param [in] mat The target matrix.
       */
      DenseMatrixClass(const DenseMatrixClass<T> &mat);
      
      /**
       * @brief   The move constructor of DenseMatrixClass.
       * @details The move constructor of DenseMatrixClass.
       * @param [in] mat The target matrix.
       */
      DenseMatrixClass(DenseMatrixClass<T> &&mat);
      
      /**
       * @brief   The = operator of DenseMatrixClass.
       * @details The = operator of DenseMatrixClass.
       * @param [in] mat The target matrix.
       * @return     Return the matrix.
       */
      DenseMatrixClass<T>& operator= (const DenseMatrixClass<T> &mat);
      
      /**
       * @brief   The = operator of DenseMatrixClass.
       * @details The = operator of DenseMatrixClass.
       * @param [in] mat The target matrix.
       * @return     Return the matrix.
       */
      DenseMatrixClass<T>& operator= (DenseMatrixClass<T> &&mat);
      
      /**
       * @brief   The destructor of DenseMatrixClass.
       * @details The destructor of DenseMatrixClass. Simply a call to the free function.
       */
      virtual ~DenseMatrixClass();
      
      /**
       * @brief   Free the current matrix, and malloc memory to create a new matrix on the host.
       * @details Free the current matrix, and malloc memory to create a new matrix on the host. The memory location is controlled by _location. \n
       *          The _ldim is set to be equal to nrow.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number fo columns.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols);
      
      /**
       * @brief   Free the current matrix, and malloc memory to create a new matrix.
       * @details Free the current matrix, and malloc memory to create a new matrix. The memory location is controlled by _location. \n
       *          The _ldim is set to be equal to nrow.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number fo columns.
       * @param [in] setzero Set to true to use calloc instead of malloc.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols, bool setzero);
      
      /**
       * @brief   Free the current matrix, and malloc memory to create a new matrix.
       * @details Free the current matrix, and malloc memory to create a new matrix. The memory location is controlled by _location. \n
       *          The _ldim is set to be equal to nrow.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number fo columns.
       * @param [in] location The location of the data.
       * @param [in] setzero Set to true to use calloc instead of malloc.
       * @return     Return error message.
       */
      int            Setup(int nrows, int ncols, int location, bool setzero);
      
      /**
       * @brief   Free the current matrix, and point the matrix to data.
       * @details Free the current matrix, and point the matrix to data.
       * @param [in] data Pointer to the data.
       * @param [in] nrows The number of rows.
       * @param [in] ncols The number fo columns.
       * @param [in] ldim The leading dimension of the matrix.
       * @param [in] location The location of the data.
       * @return     Return error message.
       */
      int            SetupPtr( T* data, int nrows, int ncols, int ldim, int location);
      
      /**
       * @brief   Free the current matrix, and point the matrix to a submatrix of another matrix.
       * @details Free the current matrix, and point the matrix to a submatrix of another matrix.
       * @param [in] mat_in The matrix holding data.
       * @param [in] row_start The starting row of that submatrix.
       * @param [in] col_start The starting column of that submatrix.
       * @param [in] num_rows The number of rows.
       * @param [in] num_cols The number fo cols.
       * @return     Return error message.
       */
      int            SetupPtr(const DenseMatrixClass<T> &mat_in, int row_start, int col_start, int num_rows, int num_cols);
      
      /**
       * @brief   Copy data to extract a submatrix of the current matrix.
       * @details Copy data to extract a submatrix of the current matrix.
       * @param [in] row_start The starting row of that submatrix.
       * @param [in] col_start The starting column of that submatrix.
       * @param [in] num_rows The number of rows.
       * @param [in] num_cols The number fo cols.
       * @param [in] location The location of the submatrix.
       * @param [in] mat_out The submatrix.
       * @return     Return error message.
       */
      int            SubMatrix( int row_start, int col_start, int num_rows, int num_cols, int location, DenseMatrixClass<T> &mat_out);
      
      /**
       * @brief   Get the reference of a value in the matrix based on the index.
       * @details Get the reference of a value in the matrix based on the index.
       * @param [in] row The row index.
       * @param [in] col The column index.
       * @return     return the reference to the target value.
       */
      T&             operator()( int row, int col);
      
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
       * @brief   Get the data pointer of the matrix.
       * @details Get the data pointer of the matrix.
       * @return  Return the data pointer.
       */
      T*             GetData() const;
      
      /**
       * @brief   Get the reference to the data vector.
       * @details Get the reference to the data vector.
       * @return  Return the reference to the data vector.
       */
      SequentialVectorClass<T>& GetDataVector();
      
      /**
       * @brief   Get the reference to the data vector.
       * @details Get the reference to the data vector.
       * @return  Return the reference to the data vector.
       */
      const SequentialVectorClass<T>& GetDataVector() const;
      
      /**
       * @brief   Get the leading dimension of the matrix.
       * @details Get the leading dimension of the matrix.
       * @return     Return the leading dimension of the matrix.
       */
      int            GetLeadingDimension() const;
      
      /**
       * @brief   Check if the matrix holding its own data.
       * @details Check if the matrix holding its own data.
       * @return     Return true if matrix holding data.
       */
      bool           IsHoldingData() const;
      
      /**
       * @brief   Create an indentity matrix.
       * @details Create an indentity matrix.
       * @return     Return error message.
       */
      virtual int    Eye();
      
      /**
       * @brief   Create an random matrix, real value in between (0, 1), complex value real and imag part in between (0, 1).
       * @details Create an random matrix, real value in between (0, 1), complex value real and imag part in between (0, 1).
       * @return     Return error message.
       */
      int            Rand();
      
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
       * @brief      Scale the current .
       * @details    Scale the current .
       * @param [in] alpha The scale.
       * @return     Return error message.
       */
      virtual int    Scale(const T &alpha);
      
      /**
       * @brief   In place dense Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @details In place dense Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           return GEMSLR_SUCESS or error information.
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
       * @param [in]       A The first dense matrix.
       * @param [in]       transa Whether we transpose A or not.
       * @param [in]       beta The beta value.
       * @param [in]       B The second dense matrix.
       * @param [in]       transb Whether we transpose B or not.
       * @return           Return error message.
       */
      int            MatMat( const T &alpha, const DenseMatrixClass<T> &A, char transa, const DenseMatrixClass<T> &B, char transb, const T &beta);
      
      /**
       * @brief   Invert the matrix.
       * @details Invert the matrix. Only work on the host.
       * @return  Return error message.
       */
      int            Invert();
      
      /**
       * @brief   Invert the matrix. Only works when current matrix is upper triangular.
       * @details Invert the matrix. Only works when current matrix is upper triangular. Only work on the host.
       * @return  Return error message.
       */
      int            InvertUpperTriangular();
      
      /**
       * @brief   Transform this matrix A into hessenberg matrix Q^HAQ = Hess.
       * @details Transform this matrix A into hessenberg matrix Q^HAQ = Hess.
       * @param [out]      Q the Q matrix.
       * @return           Return error message.
       */
      int            Hess( DenseMatrixClass<T> &Q);
      
      /**
       * @brief   Transform this matrix A into hessenberg matrix Q^HAQ = Hess.
       * @details Transform this matrix A into hessenberg matrix Q^HAQ = Hess.
       * @param [out]      Q the Q matrix.
       * @param [in]       start
       * @param [in]       end Outside [start, end) the matrix is already hessenberg.
       * @return           Return error message.
       */
      int            Hess( DenseMatrixClass<T> &Q, int start, int end);
      
      /**
       * @brief   Transform this Hess matrix A into the schur form matrix U = Q^HAQ.
       * @details Transform this Hess matrix A into the schur form matrix U = Q^HAQ. \n
       *          Only works if the current matrix is a hessenberg matrix, otherwise please call the Schur function.
       * @param [out]      Q The Q matrix.
       * @param [out]      wr The vector of the real part of the eignelvalues.
       * @param [out]      wi The vector of the imag part of the eignelvalues.
       * @return           Return error message.
       */
      int            HessSchur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi);
      
      /**
       * @brief   Transform this Hess matrix A into the schur form matrix U = Q^HAQ.
       * @details Transform this Hess matrix A into the schur form matrix U = Q^HAQ. \n
       *          Only works if the current matrix is a hessenberg matrix, otherwise please call the Schur function.
       * @param [out]      Q The Q matrix.
       * @param [in]       start
       * @param [in]       end Outside [start, end) the matrix is already upper triangular.
       * @param [out]      wr The vector of the real part of the eignelvalues.
       * @param [out]      wi The vector of the imag part of the eignelvalues.
       * @return           Return error message.
       */
      int            HessSchur( DenseMatrixClass<T> &Q, int start, int end, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi);
      
      /**
       * @brief   Transform this matrix A into the schur form matrix U = Q^HAQ.
       * @details Transform this matrix A into the schur form matrix U = Q^HAQ. \n
       *          Only works if the current matrix is a hessenberg matrix, otherwise please call the Schur function.
       * @param [out]      Q The Q matrix.
       * @param [out]      w The vector of eignelvalues.
       * @return           Return error message.
       */
      int            HessSchur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w);
      
      /**
       * @brief   Transform this matrix A into the schur form matrix U = Q^HAQ.
       * @details Transform this matrix A into the schur form matrix U = Q^HAQ. \n
       *          Only works if the current matrix is a hessenberg matrix, otherwise please call the Schur function.
       * @param [out]      Q The Q matrix.
       * @param [in]       start
       * @param [in]       end Outside [start, end) the matrix is already upper triangular.
       * @param [out]      w The vector of eignelvalues.
       * @return           Return error message.
       */
      int            HessSchur( DenseMatrixClass<T> &Q, int start, int end, SequentialVectorClass<T> &w);
      
      /**
       * @brief   Transform this matrix A (real) into the schur form matrix U = Q^HAQ.
       * @details Transform this matrix A (real) into the schur form matrix U = Q^HAQ. \n
       *          Only works if the current matrix is a hessenberg matrix, otherwise please call the Schur function.
       * @param [out]      Q The Q matrix.
       * @param [out]      wr The vector of the real part of the eignelvalues.
       * @param [out]      wi The vector of the imag part of the eignelvalues.
       * @return           Return error message.
       */
      int            Schur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi);
      
      /**
       * @brief   Transform this matrix A (real) into the schur form matrix U = Q^HAQ.
       * @details Transform this matrix A (real) into the schur form matrix U = Q^HAQ. \n
       *          Only works if the current matrix is a hessenberg matrix, otherwise please call the Schur function.
       * @param [out]      Q The Q matrix.
       * @param [in]       start
       * @param [in]       end Outside [start, end) the matrix is already hessenberg.
       * @param [out]      wr The vector of the real part of the eignelvalues.
       * @param [out]      wi The vector of the imag part of the eignelvalues.
       * @return           Return error message.
       */
      int            Schur( DenseMatrixClass<T> &Q, int start, int end, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi);
      
      /**
       * @brief   Transform this matrix A (complex) into the schur form matrix U = Q^HAQ.
       * @details Transform this matrix A (complex) into the schur form matrix U = Q^HAQ.
       * @param [out]      Q The Q matrix.
       * @param [out]      w The vector of eignelvalues.
       * @return           Return error message.
       */
      int            Schur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w);
      
      /**
       * @brief   Transform this matrix A (complex) into the schur form matrix U = Q^HAQ.
       * @details Transform this matrix A (complex) into the schur form matrix U = Q^HAQ.
       * @param [out]      Q The Q matrix.
       * @param [in]       start
       * @param [in]       end Outside [start, end) the matrix is already upper triangular.
       * @param [out]      w The vector of eignelvalues.
       * @return           Return error message.
       */
      int            Schur( DenseMatrixClass<T> &Q, int start, int end, SequentialVectorClass<T> &w);
      
      /**
       * @brief   Compute the Eigen decomposition of real dense hessenberg matrix A, AQ = QD. A is this matrix and is not modified.
       * @details Compute the Eigen decomposition of real dense hessenberg matrix A, AQ = QD. A is this matrix and is not modified.
       * @param [out]      Q the Q matrix AQ = QD.
       * @param [in,out]   wr Array of real part of eigenvalues.
       * @param [in]       wi Array of imag part of eigenvalues.
       * @return           Return error message.
       */
      int            HessEig( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi);
      
      /**
       * @brief   Compute the Eigen decomposition of complex dense hessenberg matrix A, AQ = QD. A is this matrix and is not modified.
       * @details Compute the Eigen decomposition of complex dense hessenberg matrix A, AQ = QD. A is this matrix and is not modified.
       * @param [out]      Q the Q matrix AQ = QD.
       * @param [in,out]   w Array of eigenvalues.
       * @return           Return error message.
       */
      int            HessEig( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w);
      
      /**
       * @brief   Compute the Schur decomposition of real dense matrix A, U = QS^HAQS, and the Eigen decomposition UQE = QED. A is this matrix and on return be set to U.
       * @details Compute the Schur decomposition of real dense matrix A, U = QS^HAQS, and the Eigen decomposition UQE = QED. A is this matrix and on return be set to U.
       * @param [out]      QS the SQ matrix U = QS^HAQS.
       * @param [out]      QE the QE matrix UQE = QED.
       * @param [in,out]   wr Array of real part of eigenvalues.
       * @param [in]       wi Array of imag part of eigenvalues.
       * @return           Return error message.
       */
      int            Eig( DenseMatrixClass<T> &QS, DenseMatrixClass<T> &QE, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi);
      
      /**
       * @brief   Compute the Schur decomposition of complex dense matrix A, U = QS^HAQS, and the Eigen decomposition UQE = QED. A is this matrix and on return be set to U.
       * @details Compute the Schur decomposition of complex dense matrix A, U = QS^HAQS, and the Eigen decomposition UQE = QED. A is this matrix and on return be set to U.
       * @param [out]      QS the SQ matrix U = QS^HAQS.
       * @param [out]      QE the QE matrix UQE = QED.
       * @param [in,out]   w Array of eigenvalues.
       * @return           Return error message.
       */
      int            Eig( DenseMatrixClass<T> &QS, DenseMatrixClass<T> &QE, SequentialVectorClass<T> &w);
      
      /**
       * @brief   Reorder the this real schur decomposition, puch selected eigenvalues to the leading part.
       * @details Reorder the this real schur decomposition, puch selected eigenvalues to the leading part. A 2x2 block on the diagonal of Q need to be
    *          in the same cluster.
       * @param [in,out]   Q This matrix and Q are the original schur decomposition, on return, the new decomposition.
       * @param [out]      wr Array of real part of eigenvalues.
       * @param [out]      wi Array of imag part of eigenvalues.
       * @param [in]       select Eigenvalues corresponting to positive entries in select vector will be put in the leading part.
       * @return           Return error message.
       */
      int            OrdSchur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, vector_int &select);
      
      /**
       * @brief   Reorder the this complex schur decomposition, puch selected eigenvalues to the leading part.
       * @details Reorder the this complex schur decomposition, puch selected eigenvalues to the leading part.
       * @param [in,out]   Q This matrix and Q are the original schur decomposition, on return, the new decomposition.
       * @param [out]      w Array of eigenvalues.
       * @param [in]       select Eigenvalues corresponting to positive entries in select vector will be put in the leading part.
       * @return           Return error message.
       */
      int            OrdSchur( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w, vector_int &select);
      
      /**
       * @brief   Reorder the this real schur decomposition, puch selected eigenvalues to the leading part.
       * @details Reorder the this real schur decomposition, puch selected eigenvalues to the leading part. A 2x2 block on the diagonal of Q need to be
    *          in the same cluster.
       * @param [in,out]   Q This matrix and Q are the original schur decomposition, on return, the new decomposition.
       * @param [out]      wr Array of real part of eigenvalues.
       * @param [out]      wi Array of imag part of eigenvalues.
       * @param [in]       clusters Eigenvalues corresponding to positive entries in the clusters vector will be put in the leading part. \n
       *                   The order of those eigenvalues are based on the values in the cluster vector in descending order.
       * @return           Return error message.
       */
      int            OrdSchurClusters( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, vector_int &clusters);
      
      /**
       * @brief   Reorder the this complex schur decomposition, puch selected eigenvalues to the leading part.
       * @details Reorder the this complex schur decomposition, puch selected eigenvalues to the leading part.
       * @param [in,out]   Q This matrix and Q are the original schur decomposition, on return, the new decomposition.
       * @param [out]      w Array of eigenvalues.
       * @param [in]       clusters Eigenvalues corresponding to positive entries in the clusters vector will be put in the leading part. \n
       *                   The order of those eigenvalues are based on the values in the cluster vector in descending order.
       * @return           Return error message.
       */
      int            OrdSchurClusters( DenseMatrixClass<T> &Q, SequentialVectorClass<T> &w, vector_int &clusters);
      
      /**
       * @brief   Plot the dense matrix to the terminal output. Function for testing purpose.
       * @details Plot the dense matrix to the terminal output. Function for testing purpose.
       * @param [in]   conditiona First condition.
       * @param [in]   conditionb Secend condition, only spy when conditiona == conditionb.
       * @param [in]   width The plot width.
       * @return       Return error message.
       */
      int            Plot( int conditiona, int conditionb, int width);
      
   };
   
   typedef DenseMatrixClass<float>     matrix_dense_float;
   typedef DenseMatrixClass<double>    matrix_dense_double;
   typedef DenseMatrixClass<complexs>  matrix_dense_complexs;
   typedef DenseMatrixClass<complexd>  matrix_dense_complexd;
   template<> struct PargemslrIsComplex<DenseMatrixClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<DenseMatrixClass<complexd> > : public std::true_type {};
   
}

#endif
