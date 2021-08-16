#ifndef PARGEMSLR_MATRIXOPS_H
#define PARGEMSLR_MATRIXOPS_H

/**
 * @file       matrixops.hpp
 * @brief      Matrix operations.
 * @details    Matrix operations. \n
 *             DenseMatrixXxx: functions for dense matrices. \n
 *             CsrMatrixXxx: functions for csr/csc matrices.
 */

#include <vector>

#include "../vectors/vector.hpp"
#include "../utils/mmio.hpp"
#include "matrix.hpp"
#include "csr_matrix.hpp"
#include "parallel_csr_matrix.hpp"
#include "coo_matrix.hpp"
#include "dense_matrix.hpp"

using namespace std;

namespace pargemslr
{
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPrecision(const MatrixClass<int> &mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPrecision(const MatrixClass<long int> &mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPrecision(const MatrixClass<float> &mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPrecision(const MatrixClass<double> &mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPrecision(const MatrixClass<complexs> &mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPrecision(const MatrixClass<complexd> &mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<int> *mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<long int> *mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<float> *mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<double> *mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<complexs> *mat);
   
   /**
    * @brief   Get the precision of a matrix.
    * @details Get the precision of a matrix.
    * @param [in]  mat the matrix.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<complexd> *mat);
   
   /**
    * @brief   In place dense complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place dense complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       nrows Number of rows.
    * @param [in]       ncols Number of columns.
    * @param [in]       alpha The alpha value.
    * @param [in]       aa Pointer to the data.
    * @param [in]       ldim The leading dimension of A.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The result vector.
    * @return           Return error message.
    */
   template <typename T>
   int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const T &alpha, const T *aa, int ldim, const T *x, const T &beta, T *y);
   
   /**
    * @brief   In place dense float Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place dense float Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The dense matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The result vector.
    * @return           Return error message.
    */
   int DenseMatrixMatVec( const DenseMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   
   /**
    * @brief   In place dense double Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place dense double Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The dense matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The result vector.
    * @return           Return error message.
    */
   int DenseMatrixMatVec( const DenseMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   
   /**
    * @brief   In place dense single complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place dense single complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The dense matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The result vector.
    * @return           Return error message.
    */
   int DenseMatrixMatVec( const DenseMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   
   /**
    * @brief   In place dense double complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place dense double complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The dense matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The result vector.
    * @return           Return error message.
    */
   int DenseMatrixMatVec( const DenseMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
#ifdef PARGEMSLR_CUDA 
   
   /**
    * @brief   In place dense float Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place dense float Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y. \n
    *          Note that you can call this matvec with parallel vector, as long as the local part of the vector has correct size.
    * @param [in]       A The dense matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int DenseMatrixSMatVecDevice( const DenseMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   
   /**
    * @brief   In place dense double Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place dense double Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y. \n
    *          Note that you can call this matvec with parallel vector, as long as the local part of the vector is correct.
    * @param [in]       A The dense matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int DenseMatrixDMatVecDevice( const DenseMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   
   /**
    * @brief   In place dense single complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place dense single complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y. \n
    *          Note that you can call this matvec with parallel vector, as long as the local part of the vector is correct.
    * @param [in]       A The dense matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int DenseMatrixCMatVecDevice( const DenseMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   
   /**
    * @brief   In place dense double complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place dense double complex Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y. \n
    *          Note that you can call this matvec with parallel vector, as long as the local part of the vector is correct.
    * @param [in]       A The dense matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int DenseMatrixZMatVecDevice( const DenseMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
#endif
   
   /**
    * @brief   Compute the inverese of a general float matrix.
    * @details Compute the inverese of a general float matrix.
    * @param [in,out]   A The dense matrix, on exit the inverse of it.
    * @return           Return error message.
    */
   int DenseMatrixInvertHost( DenseMatrixClass<float> &A);
   
   /**
    * @brief   Compute the inverese of a general double matrix.
    * @details Compute the inverese of a general double matrix.
    * @param [in,out]   A The dense matrix, on exit the inverse of it.
    * @return           Return error message.
    */
   int DenseMatrixInvertHost( DenseMatrixClass<double> &A);
   
   /**
    * @brief   Compute the inverese of a general single complex matrix.
    * @details Compute the inverese of a general single complex matrix.
    * @param [in,out]   A The dense matrix, on exit the inverse of it.
    * @return           Return error message.
    */
   int DenseMatrixInvertHost( DenseMatrixClass<complexs> &A);
   
   /**
    * @brief   Compute the inverese of a general double complex matrix.
    * @details Compute the inverese of a general double complex matrix.
    * @param [in,out]   A The dense matrix, on exit the inverse of it.
    * @return           Return error message.
    */
   int DenseMatrixInvertHost( DenseMatrixClass<complexd> &A);
   
   /**
    * @brief   Compute the inverese of a upper triangular float matrix.
    * @details Compute the inverese of a upper triangular float matrix.
    * @param [in,out]   A The dense matrix, on exit the inverse of it.
    * @return           Return error message.
    */
   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<float> &A);
   
   /**
    * @brief   Compute the inverese of a upper triangular double matrix.
    * @details Compute the inverese of a upper triangular double matrix.
    * @param [in,out]   A The dense matrix, on exit the inverse of it.
    * @return           Return error message.
    */
   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<double> &A);
   
   /**
    * @brief   Compute the inverese of a upper triangular single complex matrix.
    * @details Compute the inverese of a upper triangular single complex matrix.
    * @param [in,out]   A The dense matrix, on exit the inverse of it.
    * @return           Return error message.
    */
   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<complexs> &A);
   
   /**
    * @brief   Compute the inverese of a upper triangular double complex matrix.
    * @details Compute the inverese of a upper triangular double complex matrix.
    * @param [in,out]   A The dense matrix, on exit the inverse of it.
    * @return           Return error message.
    */
   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<complexd> &A);
   
   /**
    * @brief   Compute the thin QR decomposition of mxn matrix A = QR. When m >= n, Q is mxn and R is nxn. When n > m, Q is mxm and R is mxn.
    * @details Compute the thin QR decomposition of mxn matrix A = QR. When m >= n, Q is mxn and R is nxn. When n > m, Q is mxm and R is mxn.
    * @param [in,out]   A The input matrix A. On return the R matrix.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixQRDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q);
   
   /**
    * @brief   Compute the thin QR decomposition of mxn matrix A = QR. When m >= n, Q is mxn and R is nxn. When n > m, Q is mxm and R is mxn.
    * @details Compute the thin QR decomposition of mxn matrix A = QR. When m >= n, Q is mxn and R is nxn. When n > m, Q is mxm and R is mxn.
    * @param [in,out]   A The input matrix A. On return the R matrix.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixQRDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q);
   
   /**
    * @brief   Compute the thin QR decomposition of mxn matrix A = QR. When m >= n, Q is mxn and R is nxn. When n > m, Q is mxm and R is mxn.
    * @details Compute the thin QR decomposition of mxn matrix A = QR. When m >= n, Q is mxn and R is nxn. When n > m, Q is mxm and R is mxn.
    * @param [in,out]   A The input matrix A. On return the R matrix.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixQRDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q);
   
   /**
    * @brief   Compute the thin QR decomposition of mxn matrix A = QR. When m >= n, Q is mxn and R is nxn. When n > m, Q is mxm and R is mxn.
    * @details Compute the thin QR decomposition of mxn matrix A = QR. When m >= n, Q is mxn and R is nxn. When n > m, Q is mxm and R is mxn.
    * @param [in,out]   A The input matrix A. On return the R matrix.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixQRDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q);
   
   /**
    * @brief   Transform a dense float matrix A into hessenberg matrix Q^HAQ = Hess.
    * @details Transform a dense float matrix A into hessenberg matrix Q^HAQ = Hess.
    * @param [in,out]   A The input matrix A, on return, the hessenberg matrix.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q);
   
   /**
    * @brief   Transform a dense float matrix A into hessenberg matrix Q^HAQ = Hess.
    * @details Transform a dense float matrix A into hessenberg matrix Q^HAQ = Hess.
    * @param [in,out]   A The input matrix A, on return, the hessenberg matrix.
    * @param [in]       start
    * @param [in]       end Outside [start, end) the matrix is already hessenberg.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<float> &A, int start, int end, DenseMatrixClass<float> &Q);
   
   /**
    * @brief   Transform a dense double matrix A into hessenberg matrix Q^HAQ = Hess.
    * @details Transform a dense double matrix A into hessenberg matrix Q^HAQ = Hess.
    * @param [in,out]   A The input matrix A, on return, the hessenberg matrix.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q);
   
   /**
    * @brief   Transform a dense double matrix A into hessenberg matrix Q^HAQ = Hess.
    * @details Transform a dense double matrix A into hessenberg matrix Q^HAQ = Hess.
    * @param [in,out]   A The input matrix A, on return, the hessenberg matrix.
    * @param [in]       start
    * @param [in]       end Outside [start, end) the matrix is already hessenberg.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<double> &A, int start, int end, DenseMatrixClass<double> &Q);
   
   /**
    * @brief   Transform a dense single complex matrix A into hessenberg matrix Q^HAQ = Hess.
    * @details Transform a dense single complex matrix A into hessenberg matrix Q^HAQ = Hess.
    * @param [in,out]   A The input matrix A, on return, the hessenberg matrix.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q);
   
   /**
    * @brief   Transform a dense single complex matrix A into hessenberg matrix Q^HAQ = Hess.
    * @details Transform a dense single complex matrix A into hessenberg matrix Q^HAQ = Hess.
    * @param [in,out]   A The input matrix A, on return, the hessenberg matrix.
    * @param [in]       start
    * @param [in]       end Outside [start, end) the matrix is already hessenberg.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexs> &A, int start, int end, DenseMatrixClass<complexs> &Q);
   
   /**
    * @brief   Transform a dense double complex matrix A into hessenberg matrix Q^HAQ = Hess.
    * @details Transform a dense double complex matrix A into hessenberg matrix Q^HAQ = Hess.
    * @param [in,out]   A The input matrix A, on return, the hessenberg matrix.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q);
   
   /**
    * @brief   Transform a dense double complex matrix A into hessenberg matrix Q^HAQ = Hess.
    * @details Transform a dense double complex matrix A into hessenberg matrix Q^HAQ = Hess.
    * @param [in,out]   A The input matrix A, on return, the hessenberg matrix.
    * @param [in]       start
    * @param [in]       end Outside [start, end) the matrix is already hessenberg.
    * @param [out]      Q the Q matrix.
    * @return           Return error message.
    */
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexd> &A, int start, int end, DenseMatrixClass<complexd> &Q);
   
   /**
    * @brief   Compute the Schur decomposition of a float dense hessenberg matrix A = QUQ^H.
    * @details Compute the Schur decomposition of a float dense hessenberg matrix A = QUQ^H.
    * @param [in,out]   A The input matrix A, on return, the U matrix.
    * @param [out]      Q the Q matrix A = QUQ^H.
    * @param [out]      wr Array of real part of the eigenvalues.
    * @param [out]      wi Array of imag part of the eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi);
   
   /**
    * @brief   Compute the Schur decomposition of a float dense hessenberg matrix A = QUQ^H.
    * @details Compute the Schur decomposition of a float dense hessenberg matrix A = QUQ^H.
    * @param [in,out]   A The input matrix A, on return, the U matrix.
    * @param [in]       start
    * @param [in]       end Outside [start, end) the matrix is already upper triangular.
    * @param [out]      Q the Q matrix A = QUQ^H.
    * @param [out]      wr Array of real part of the eigenvalues.
    * @param [out]      wi Array of imag part of the eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<float> &A, int start, int end, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi);
   
   /**
    * @brief   Compute the Schur decomposition of a double dense hessenberg matrix A = QUQ^H.
    * @details Compute the Schur decomposition of a double dense hessenberg matrix A = QUQ^H.
    * @param [in,out]   A The input matrix A, on return, the U matrix.
    * @param [out]      Q the Q matrix A = QUQ^H.
    * @param [out]      wr Array of real part of the eigenvalues.
    * @param [out]      wi Array of imag part of the eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi);
   
   /**
    * @brief   Compute the Schur decomposition of a double dense hessenberg matrix A = QUQ^H.
    * @details Compute the Schur decomposition of a double dense hessenberg matrix A = QUQ^H.
    * @param [in,out]   A The input matrix A, on return, the U matrix.
    * @param [in]       start
    * @param [in]       end Outside [start, end) the matrix is already upper triangular.
    * @param [out]      Q the Q matrix A = QUQ^H.
    * @param [out]      wr Array of real part of the eigenvalues.
    * @param [out]      wi Array of imag part of the eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<double> &A, int start, int end, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi);
   
   /**
    * @brief   Compute the Schur decomposition of a single complex dense hessenberg matrix A = QUQ^H.
    * @details Compute the Schur decomposition of a single complex dense hessenberg matrix A = QUQ^H.
    * @param [in,out]   A The input matrix A, on return, the U matrix.
    * @param [out]      Q the Q matrix A = QUQ^H.
    * @param [out]      w Array of the eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w);
   
   /**
    * @brief   Compute the Schur decomposition of a single complex dense hessenberg matrix A = QUQ^H.
    * @details Compute the Schur decomposition of a single complex dense hessenberg matrix A = QUQ^H.
    * @param [in,out]   A The input matrix A, on return, the U matrix.
    * @param [in]       start
    * @param [in]       end Outside [start, end) the matrix is already upper triangular.
    * @param [out]      Q the Q matrix A = QUQ^H.
    * @param [out]      w Array of the eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexs> &A, int start, int end, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w);
   
   /**
    * @brief   Compute the Schur decomposition of a double complex dense hessenberg matrix A = QUQ^H.
    * @details Compute the Schur decomposition of a double complex dense hessenberg matrix A = QUQ^H.
    * @param [in,out]   A The input matrix A, on return, the U matrix.
    * @param [out]      Q the Q matrix A = QUQ^H.
    * @param [out]      w Array of the eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w);
   
   /**
    * @brief   Compute the Schur decomposition of a double complex dense hessenberg matrix A = QUQ^H.
    * @details Compute the Schur decomposition of a double complex dense hessenberg matrix A = QUQ^H.
    * @param [in,out]   A The input matrix A, on return, the U matrix.
    * @param [in]       start
    * @param [in]       end Outside [start, end) the matrix is already upper triangular.
    * @param [out]      Q the Q matrix A = QUQ^H.
    * @param [out]      w Array of the eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexd> &A, int start, int end, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w);
   
   /**
    * @brief   Reorder the float schur decomposition, puch selected eigenvalues to the leading part.
    * @details Reorder the float schur decomposition, puch selected eigenvalues to the leading part.
    * @param [in,out]   A A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [in,out]   Q A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [out]      wr Array of real part of eigenvalues.
    * @param [out]      wi Array of imag part of eigenvalues.
    * @param [in]       select Eigenvalues corresponting to positive entries in select vector will be put in the leading part.
    * @return           Return error message.
    */
   int DenseMatrixRealOrderSchur( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi, vector_int &select);
   
   /**
    * @brief   Reorder the double schur decomposition, puch selected eigenvalues to the leading part.
    * @details Reorder the double schur decomposition, puch selected eigenvalues to the leading part.
    * @param [in,out]   A A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [in,out]   Q A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [out]      wr Array of real part of eigenvalues.
    * @param [out]      wi Array of imag part of eigenvalues.
    * @param [in]       select Eigenvalues corresponting to positive entries in select vector will be put in the leading part.
    * @return           Return error message.
    */
   int DenseMatrixRealOrderSchur( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi, vector_int &select);
   
   /**
    * @brief   Reorder the single complex schur decomposition, puch selected eigenvalues to the leading part.
    * @details Reorder the single complex schur decomposition, puch selected eigenvalues to the leading part. A 2x2 block on the diagonal of Q need to be
    *          in the same cluster.
    * @param [in,out]   A A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [in,out]   Q A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [out]      w Array of eigenvalues.
    * @param [in]       select Eigenvalues corresponting to positive entries in select vector will be put in the leading part.
    * @return           Return error message.
    */
   int DenseMatrixComplexOrderSchur( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w, vector_int &select);
   
   /**
    * @brief   Reorder the double complex schur decomposition, puch selected eigenvalues to the leading part.
    * @details Reorder the double complex schur decomposition, puch selected eigenvalues to the leading part.
    * @param [in,out]   A A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [in,out]   Q A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [out]      w Array of eigenvalues.
    * @param [in]       select Eigenvalues corresponting to positive entries in select vector will be put in the leading part.
    * @return           Return error message.
    */
   int DenseMatrixComplexOrderSchur( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w, vector_int &select);
   
   /**
    * @brief   Reorder the float schur decomposition, puch selected eigenvalues to the leading part.
    * @details Reorder the float schur decomposition, puch selected eigenvalues to the leading part. A 2x2 block on the diagonal of Q need to be
    *          in the same cluster.
    * @param [in,out]   A A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [in,out]   Q A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [out]      wr Array of real part of eigenvalues.
    * @param [out]      wi Array of imag part of eigenvalues.
    * @param [in]       clusters Eigenvalues corresponding to positive entries in the clusters vector will be put in the leading part. \n
    *                   The order of those eigenvalues are based on the values in the cluster vector in descending order.
    * @return           Return error message.
    */
   int DenseMatrixRealOrderSchurClusters( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi, vector_int &clusters);
   
   /**
    * @brief   Reorder the double schur decomposition, puch selected eigenvalues to the leading part.
    * @details Reorder the double schur decomposition, puch selected eigenvalues to the leading part. A 2x2 block on the diagonal of Q need to be
    *          in the same cluster.
    * @param [in,out]   A A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [in,out]   Q A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [out]      wr Array of real part of eigenvalues.
    * @param [out]      wi Array of imag part of eigenvalues.
    * @param [in]       clusters Eigenvalues corresponding to positive entries in the clusters vector will be put in the leading part. \n
    *                   The order of those eigenvalues are based on the values in the cluster vector in descending order.
    * @return           Return error message.
    */
   int DenseMatrixRealOrderSchurClusters( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi, vector_int &clusters);
   
   /**
    * @brief   Reorder the single complex schur decomposition, puch selected eigenvalues to the leading part.
    * @details Reorder the single complex schur decomposition, puch selected eigenvalues to the leading part.
    * @param [in,out]   A A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [in,out]   Q A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [out]      w Array of eigenvalues.
    * @param [in]       clusters Eigenvalues corresponding to positive entries in the clusters vector will be put in the leading part. \n
    *                   The order of those eigenvalues are based on the values in the cluster vector in descending order.
    * @return           Return error message.
    */
   int DenseMatrixComplexOrderSchurClusters( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w, vector_int &clusters);
   
   /**
    * @brief   Reorder the double complex schur decomposition, puch selected eigenvalues to the leading part.
    * @details Reorder the double complex schur decomposition, puch selected eigenvalues to the leading part.
    * @param [in,out]   A A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [in,out]   Q A and Q are the original schur decomposition, on return, the new decomposition.
    * @param [out]      w Array of eigenvalues.
    * @param [in]       clusters Eigenvalues corresponding to positive entries in the clusters vector will be put in the leading part. \n
    *                   The order of those eigenvalues are based on the values in the cluster vector in descending order.
    */
   int DenseMatrixComplexOrderSchurClusters( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w, vector_int &clusters);
   
   /**
    * @brief   Compute the Eigen decomposition of float dense hessenberg matrix A, AQ = QD. A is not modified.
    * @details Compute the Eigen decomposition of float dense hessenberg matrix A, AQ = QD. A is not modified.
    * @param [in]       A The input matrix A.
    * @param [out]      Q the Q matrix AQ = QD.
    * @param [in,out]   wr Array of real part of eigenvalues.
    * @param [in]       wi Array of imag part of eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixRealHessEigenDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi);
   
   /**
    * @brief   Compute the Eigen decomposition of double dense hessenberg matrix A, AQ = QD. A is not modified.
    * @details Compute the Eigen decomposition of double dense hessenberg matrix A, AQ = QD. A is not modified.
    * @param [in]       A The input matrix A.
    * @param [out]      Q the Q matrix AQ = QD.
    * @param [in,out]   wr Array of real part of eigenvalues.
    * @param [in]       wi Array of imag part of eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixRealHessEigenDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi);
   
   /**
    * @brief   Compute the Eigen decomposition of single complex dense hessenberg matrix A, AQ = QD. A is not modified.
    * @details Compute the Eigen decomposition of single complex dense hessenberg matrix A, AQ = QD. A is not modified.
    * @param [in]       A The input matrix A.
    * @param [out]      Q the Q matrix AQ = QD.
    * @param [in,out]   w Array of real part of eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixComplexHessEigenDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w);
   
   /**
    * @brief   Compute the Eigen decomposition of double complex dense hessenberg matrix A, AQ = QD. A is not modified.
    * @details Compute the Eigen decomposition of double complex dense hessenberg matrix A, AQ = QD. A is not modified. 
    * @param [in]       A The input matrix A.
    * @param [out]      Q the Q matrix AQ = QD.
    * @param [in,out]   w Array of real part of eigenvalues.
    * @return           Return error message.
    */
   int DenseMatrixComplexHessEigenDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w);
   
   /**
    * @brief   Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C.
    * @details Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C. \n
    *          C is assumed to be pre-allocated.
    * @param [in]       alpha The alpha value.
    * @param [in]       A The first csr matrix.
    * @param [in]       transa Whether we transpose A or not.
    * @param [in]       B The second csr matrix.
    * @param [in]       transb Whether we transpose B or not.
    * @param [in]       beta The beta value.
    * @param [out]      C The output C=A*B. Need not to allocate memory in advance.
    * @return           Return error message.
    */
   template <typename T>
   int DenseMatrixMatMatTemplate( const T &alpha, const DenseMatrixClass<T> &A, char transa, const DenseMatrixClass<T> &B, char transb, const T &beta, DenseMatrixClass<T> &C);
   
   /**
    * @brief   Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C.
    * @details Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C. \n
    *          If C is not pre-allocated, C will be allocated such that it has the same location as A.
    * @param [in]       alpha The alpha value.
    * @param [in]       A The first csr matrix.
    * @param [in]       transa Whether we transpose A or not.
    * @param [in]       B The second csr matrix.
    * @param [in]       transb Whether we transpose B or not.
    * @param [in]       beta The beta value.
    * @param [out]      C The output C=A*B. Need not to allocate memory in advance.
    * @return           Return error message.
    */
   int DenseMatrixMatMat( const float &alpha, const DenseMatrixClass<float> &A, char transa, const DenseMatrixClass<float> &B, char transb, const float &beta, DenseMatrixClass<float> &C);
   
   /**
    * @brief   Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C.
    * @details Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C. \n
    *          If C is not pre-allocated, C will be allocated such that it has the same location as A.
    * @param [in]       alpha The alpha value.
    * @param [in]       A The first csr matrix.
    * @param [in]       transa Whether we transpose A or not.
    * @param [in]       B The second csr matrix.
    * @param [in]       transb Whether we transpose B or not.
    * @param [in]       beta The beta value.
    * @param [out]      C The output C=A*B. Need not to allocate memory in advance.
    * @return           Return error message.
    */
   int DenseMatrixMatMat( const double &alpha, const DenseMatrixClass<double> &A, char transa, const DenseMatrixClass<double> &B, char transb, const double &beta, DenseMatrixClass<double> &C);
   
   /**
    * @brief   Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C.
    * @details Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C. \n
    *          If C is not pre-allocated, C will be allocated such that it has the same location as A.
    * @param [in]       alpha The alpha value.
    * @param [in]       A The first csr matrix.
    * @param [in]       transa Whether we transpose A or not.
    * @param [in]       B The second csr matrix.
    * @param [in]       transb Whether we transpose B or not.
    * @param [in]       beta The beta value.
    * @param [out]      C The output C=A*B. Need not to allocate memory in advance.
    * @return           Return error message.
    */
   int DenseMatrixMatMat( const complexs &alpha, const DenseMatrixClass<complexs> &A, char transa, const DenseMatrixClass<complexs> &B, char transb, const complexs &beta, DenseMatrixClass<complexs> &C);
   
   /**
    * @brief   Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C.
    * @details Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C. \n
    *          If C is not pre-allocated, C will be allocated such that it has the same location as A.
    * @param [in]       alpha The alpha value.
    * @param [in]       A The first csr matrix.
    * @param [in]       transa Whether we transpose A or not.
    * @param [in]       B The second csr matrix.
    * @param [in]       transb Whether we transpose B or not.
    * @param [in]       beta The beta value.
    * @param [out]      C The output C=A*B. Need not to allocate memory in advance.
    * @return           Return error message.
    */
   int DenseMatrixMatMat( const complexd &alpha, const DenseMatrixClass<complexd> &A, char transa, const DenseMatrixClass<complexd> &B, char transb, const complexd &beta, DenseMatrixClass<complexd> &C);
   
#ifdef PARGEMSLR_CUDA 
   
   /**
    * @brief   Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C.
    * @details Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C. C should be pre-allocated.
    * @param [in]       alpha The alpha value.
    * @param [in]       A The first csr matrix.
    * @param [in]       transa Whether we transpose A or not.
    * @param [in]       B The second csr matrix.
    * @param [in]       transb Whether we transpose B or not.
    * @param [in]       beta The beta value.
    * @param [out]      C The output C=A*B. Need not to allocate memory in advance.
    * @return           Return error message.
    */
   int DenseMatrixSMatMatDevice( int m, int n, int k, const float &alpha, const DenseMatrixClass<float> &A, char transa, const DenseMatrixClass<float> &B, char transb, const float &beta, DenseMatrixClass<float> &C);
   
   /**
    * @brief   Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C.
    * @details Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C. C should be pre-allocated.
    * @param [in]       alpha The alpha value.
    * @param [in]       A The first csr matrix.
    * @param [in]       transa Whether we transpose A or not.
    * @param [in]       B The second csr matrix.
    * @param [in]       transb Whether we transpose B or not.
    * @param [in]       beta The beta value.
    * @param [out]      C The output C=A*B. Need not to allocate memory in advance.
    * @return           Return error message.
    */
   int DenseMatrixDMatMatDevice( int m, int n, int k, const double &alpha, const DenseMatrixClass<double> &A, char transa, const DenseMatrixClass<double> &B, char transb, const double &beta, DenseMatrixClass<double> &C);
   
   /**
    * @brief   Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C.
    * @details Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C. C should be pre-allocated.
    * @param [in]       alpha The alpha value.
    * @param [in]       A The first csr matrix.
    * @param [in]       transa Whether we transpose A or not.
    * @param [in]       B The second csr matrix.
    * @param [in]       transb Whether we transpose B or not.
    * @param [in]       beta The beta value.
    * @param [out]      C The output C=A*B. Need not to allocate memory in advance.
    * @return           Return error message.
    */
   int DenseMatrixCMatMatDevice( int m, int n, int k, const complexs &alpha, const DenseMatrixClass<complexs> &A, char transa, const DenseMatrixClass<complexs> &B, char transb, const complexs &beta, DenseMatrixClass<complexs> &C);
   
   /**
    * @brief   Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C.
    * @details Dense float Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C. C should be pre-allocated.
    * @param [in]       alpha The alpha value.
    * @param [in]       A The first csr matrix.
    * @param [in]       transa Whether we transpose A or not.
    * @param [in]       B The second csr matrix.
    * @param [in]       transb Whether we transpose B or not.
    * @param [in]       beta The beta value.
    * @param [out]      C The output C=A*B. Need not to allocate memory in advance.
    * @return           Return error message.
    */
   int DenseMatrixZMatMatDevice( int m, int n, int k, const complexd &alpha, const DenseMatrixClass<complexd> &A, char transa, const DenseMatrixClass<complexd> &B, char transb, const complexd &beta, DenseMatrixClass<complexd> &C);
   
#endif
   
   /**
    * @brief   In place float csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place float csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       ia The column pointer vector.
    * @param [in]       ja The column vector.
    * @param [in]       aa The data pointer vector.
    * @param [in]       nrows The number of rows in the vector.
    * @param [in]       ncolss The number of cols in the vector.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   template<typename T>
   int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const T *aa, int nrows, int ncols, char trans, const T &alpha, const T *x, const T &beta, T *y);
   
   /**
    * @brief   In place csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       ia The column pointer vector.
    * @param [in]       ja The column vector.
    * @param [in]       aa The data pointer vector.
    * @param [in]       nrows The number of rows in the vector.
    * @param [in]       ncolss The number of cols in the vector.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const float *aa, int nrows, int ncols, char trans, const float &alpha, const float *x, const float &beta, float *y);
   
   /**
    * @brief   In place double csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place double csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       ia The column pointer vector.
    * @param [in]       ja The column vector.
    * @param [in]       aa The data pointer vector.
    * @param [in]       nrows The number of rows in the vector.
    * @param [in]       ncolss The number of cols in the vector.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const double *aa, int nrows, int ncols, char trans, const double &alpha, const double *x, const double &beta, double *y);
   
   /**
    * @brief   In place single complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place single complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       ia The column pointer vector.
    * @param [in]       ja The column vector.
    * @param [in]       aa The data pointer vector.
    * @param [in]       nrows The number of rows in the vector.
    * @param [in]       ncolss The number of cols in the vector.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const complexs *aa, int nrows, int ncols, char trans, const complexs &alpha, const complexs *x, const complexs &beta, complexs *y);
   
   /**
    * @brief   In place double complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place double complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       ia The column pointer vector.
    * @param [in]       ja The column vector.
    * @param [in]       aa The data pointer vector.
    * @param [in]       nrows The number of rows in the vector.
    * @param [in]       ncolss The number of cols in the vector.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const complexd *aa, int nrows, int ncols, char trans, const complexd &alpha, const complexd *x, const complexd &beta, complexd *y);
   
   /**
    * @brief   In place float csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place float csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixMatVec( const CsrMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   
   /**
    * @brief   In place double csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place double csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixMatVec( const CsrMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   
   /**
    * @brief   In place single complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place single complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixMatVec( const CsrMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   
   /**
    * @brief   In place double complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place double complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixMatVec( const CsrMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
#ifdef PARGEMSLR_CUDA
   
   /**
    * @brief   In place float csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place float csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixSMatVecDevice( const CsrMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   
   /**
    * @brief   In place double csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place double csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixDMatVecDevice( const CsrMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   
   /**
    * @brief   In place single complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place single complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixCMatVecDevice( const CsrMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   
   /**
    * @brief   In place double complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @details In place double complex csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
    * @param [in]       A The matrix.
    * @param [in]       trans Whether or not transpose matrix A.
    * @param [in]       alpha The alpha value.
    * @param [in]       x The left vector.
    * @param [in]       beta The beta value.
    * @param [in,out]   y The product vector.
    * @return           Return error message.
    */
   int CsrMatrixZMatVecDevice( const CsrMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
   /**
    * @brief   Setup cusparse csr matrix.
    * @details Setup cusparse csr matrix.
    * @param [in,out] A The matrix.
    * @return     Return error message.      
    */
   int CsrMatrixCreateCusparseSpMat( CsrMatrixClass<float> &A);
   
   /**
    * @brief   Setup cusparse csr matrix.
    * @details Setup cusparse csr matrix.
    * @param [in,out] A The matrix.
    * @return     Return error message.      
    */
   int CsrMatrixCreateCusparseSpMat( CsrMatrixClass<double> &A);
   
   /**
    * @brief   Setup cusparse csr matrix.
    * @details Setup cusparse csr matrix.
    * @param [in,out] A The matrix.
    * @return     Return error message.      
    */
   int CsrMatrixCreateCusparseSpMat( CsrMatrixClass<complexs> &A);
   
   /**
    * @brief   Setup cusparse csr matrix.
    * @details Setup cusparse csr matrix.
    * @param [in,out] A The matrix.
    * @return     Return error message.      
    */
   int CsrMatrixCreateCusparseSpMat( CsrMatrixClass<complexd> &A);
   
#endif   
   
   /**
    * @brief   Converts a csr matrix to csc matrix.
    * @details Converts a csr matrix to csc matrix.
    * @note    The OpenMP version requires fixed schedule for correct result.
    * @param [in]       INIDX specifies if COO should be 0/1 index.
    * @param [in]       OUTIDX specifies if CSR should be 0/1 index.
    * @param [in]       nrow Number of rows.
    * @param [in]       ncol Number of columns.
    * @param [in]       copy_data Whether we copy the data vector.
    * @param [in]       ai Data vector of the input csr matrix.
    * @param [in]       ji J vector of the input csr matrix.
    * @param [in]       ii I vector of the input csr matrix.
    * @param [out]      ao Data vector for the output csc matrix.
    * @param [out]      jo J vector for the output csc matrix.
    * @param [out]      io I vector for the output csc matrix.
    * @return           Return error message.  
    */
   template <int INIDX, int OUTIDX, typename T>
   int CsrMatrixP2CscMatrixPHost( int nrows, int ncols, bool copy_data, T* ai, int *ji, int *ii, T *ao, int *jo, int *io);
   
   /**
    * @brief   Converts a coo matrix to csr matrix.
    * @details Converts a coo matrix to csr matrix.
    * @param [in]       INIDX specifies if COO should be 0/1 index.
    * @param [in]       OUTIDX specifies if CSR should be 0/1 index.
    * @param [in]       nrows Number of rows.
    * @param [in]       ncols Number of columns.
    * @param [in]       nnz Number of nonzero entries.
    * @param [in]       ai Data vector of the input coo matrix.
    * @param [in]       ji Col indices vector of the input coo matrix.
    * @param [in]       ii Row indices vector of the input coo matrix.
    * @param [out]      ao Data vector for the output csr matrix.
    * @param [out]      jo J vector for the output csr matrix.
    * @param [out]      io I vector for the output csr matrix.
    * @return           Return error message.  
    */
   template <int INIDX, int OUTIDX, typename T>
   int CooMatrixP2CsrMatrixPHost( int nrows, int ncols, int nnz, T* ai, int *ji, int *ii, T *ao, int *jo, int *io);
   
   /**
    * @brief   Compute the transpose of a csr/csc matrix.
    * @details Compute the transpose of a csr/csc matrix.
    * @param [in]       A The input matrix.
    * @param [out]      AT The output transpose matrix.
    * @return           Return error message.  
    */
   template <typename T>
   int CsrMatrixTransposeHost( CsrMatrixClass<T> &A, CsrMatrixClass<T> &AT);
   
   /**
    * @brief   Add sparse matrix in CSR format, C= A + B.
    * @details Add sparse matrix in CSR format, C= A + B.
    * @param [in]    A Matrix A.
    * @param [in]    B Matrix B.
    * @param [out]   C Matrix C = A + B.
    * @return        Return error message.
    */
   template <typename T>
   int CsrMatrixAddHost( CsrMatrixClass<T> &A, CsrMatrixClass<T> &B, CsrMatrixClass<T> &C);
   
   /**
    * @brief   Compute the transpose of a parallel csr matrix.
    * @details Compute the transpose of a parallel csr matrix.
    * @param [in]       A The input matrix.
    * @param [out]      AT The output transpose matrix.
    * @return           Return error message.  
    */
   template <typename T>
   int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<T> &A, ParallelCsrMatrixClass<T> &AT);
   
   /**
    * @brief   Add sparse matrix in Parallel CSR format, C= A + B.
    * @details Add sparse matrix in Parallel CSR format, C= A + B.
    * @param [in]    A Matrix A.
    * @param [in]    B Matrix B.
    * @param [out]   C Matrix C = A + B.
    * @return        Return error message.
    */
   template <typename T>
   int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<T> &A, ParallelCsrMatrixClass<T> &B, ParallelCsrMatrixClass<T> &C);
   
   /**
    * @brief   Kway metis partition. For the vertex seperator, we only supports 2-way vertex seperator.
    * @details Kway metis partition. For the vertex seperator, we only supports 2-way vertex seperator.
    * @note    Only for symmetrix matirx, please compute A = A+A' before calling this function
    * @param [in]       A Matrix A.
    * @param [in,out]   num_dom Number of domains.
    * @param [out]      map The map vector, map[i] is the domain number of the ith node.
    * @param [in]       vertexsep Set to true to compute vertex seperator. Only work when num_dom is 2.
    * @param [out]      sep The seperator vector, sep[i] != 0 means the ith node is in the seperator.
    * @param [out]      edgecut The size of the nodes inside vertexsep.
    * @param [out]      perm The permutation array.
    * @param [out]      dom_ptr Pointer to the start node of each domain in the new order of length num_dom+1, the last entry is the size of A.
    * @return           Return METIS error message.
    */
   template <typename T>
   int CsrMatrixMetisKwayHost( CsrMatrixClass<T> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr);
   
   /**
    * @brief   Use Hopcroft–Karp algorithm to find a maximal matching.
    * @details Use Hopcroft–Karp algorithm to find a maximal matching.
    * @note    See https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm for more detail.
    * @param [in]       A Matrix A.
    * @param [out]      nmatch Number of matched pairs.
    * @param [out]      match_row The column map vector, match_row[i] = j if i-th row is mapping to j-th col, match_row[i] = -1 if not matched.
    * @param [out]      match_col The column map vector, match_col[i] = j if i-th col is mapping to j-th row, match_col[i] = -1 if not matched.
    * @return           Return error message.
    */
   template <typename T>
   int CsrMatrixMaxMatchingHost( CsrMatrixClass<T> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);
   
   /**
    * @brief   The BFS for maximal matching algorithm.
    * @details The BFS for maximal matching algorithm.
    * @note    See https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm for more detail.
    * @param [in]       nrows The number of rows.
    * @param [in]       ncols The number of cols.
    * @param [in]       A_i The I in CSR format.
    * @param [in]       A_j The J in CSR format.
    * @param [in]       dist The distance array.
    * @param [in]       match_row The match array for rows.
    * @param [in]       match_col The match array for cols.
    * @return           Return true if Bfs find new path.
    */
   bool CsrMatrixMaxMatchingBfsHost(int nrows, int ncols, int *A_i, int *A_j, IntVectorClass<int> &dist, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);
   
   /**
    * @brief   The BFS for maximal matching algorithm.
    * @details The BFS for maximal matching algorithm.
    * @note    See https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm for more detail.
    * @param [in]       nrows The number of rows.
    * @param [in]       ncols The number of cols.
    * @param [in]       A_i The I in CSR format.
    * @param [in]       A_j The J in CSR format.
    * @param [in]       dist The distance array.
    * @param [in]       match_row The match array for rows.
    * @param [in]       match_col The match array for cols.
    * @param [in]       i The start node.
    * @return           Return true if Dfs find new path.
    */
   bool CsrMatrixMaxMatchingDfsHost(int nrows, int ncols, int *A_i, int *A_j, IntVectorClass<int> &dist, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col, int i);
   
   /**
    * @brief   Sort columns inside each row of a real csr matrix by ascending order.
    * @details Sort columns inside each row of a real csr matrix by ascending order.
    * @param [in,out]   A The target matrix.
    * @return           Return error message.  
    */
   template <typename T>
   int CsrMatrixSortRow( CsrMatrixClass<T> &A);
   
#ifdef PARGEMSLR_CUDA
   /**
    * @brief   Sort columns inside each row of a real csr matrix by ascending order.
    * @details Sort columns inside each row of a real csr matrix by ascending order.
    * @param [in,out]   A The target matrix.
    * @return           Return error message.  
    */
   int CsrMatrixSortRowDevice( matrix_csr_float &A);
   
   /**
    * @brief   Sort columns inside each row of a real csr matrix by ascending order.
    * @details Sort columns inside each row of a real csr matrix by ascending order.
    * @param [in,out]   A The target matrix.
    * @return           Return error message.  
    */
   int CsrMatrixSortRowDevice( matrix_csr_double &A);
   
   /**
    * @brief   Sort columns inside each row of a real csr matrix by ascending order.
    * @details Sort columns inside each row of a real csr matrix by ascending order.
    * @param [in,out]   A The target matrix.
    * @return           Return error message.  
    */
   int CsrMatrixSortRowDevice( matrix_csr_complexs &A);
   
   /**
    * @brief   Sort columns inside each row of a real csr matrix by ascending order.
    * @details Sort columns inside each row of a real csr matrix by ascending order.
    * @param [in,out]   A The target matrix.
    * @return           Return error message.  
    */
   int CsrMatrixSortRowDevice( matrix_csr_complexd &A);
#endif
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function computes the AMD ordering of a submatrix A[rowscols,rowscols] + trans(A[rowscols,rowscols]).
    * @details This function computes the AMD ordering of a submatrix A[rowscols,rowscols] + trans(A[rowscols,rowscols]).
    * 
    * @param [in]       A        The target CSR matrix.                                                                                                           
    * @param [in]       rowscols The target rows/cols.    
    * @param [out]      perm     The rcm order.    
    * @return           Return error message.
    */
   template <typename T>
   int CsrSubMatrixAmdHost(CsrMatrixClass<T> &A, vector_int &rowscols, vector_int &perm);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function computes the Amd ordering of a CSR matrix A+trans(A).
    * @details This function computes the Amd ordering of A+trans(A).
    * 
    * @param [in]       A        The target CSR matrix.  
    * @param [out]      perm     The RCM permutation.  
    * @return           Return error message.
    */
   template <typename T>
   int CsrMatrixAmdHost(CsrMatrixClass<T> &A, vector_int &perm);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function computes the ND ordering of a submatrix A[rowscols,rowscols].
    * @details This function computes the ND ordering of a submatrix A[rowscols,rowscols].
    * 
    * @param [in]       A        The target CSR matrix.                                                                                                           
    * @param [in]       rowscols The target rows/cols.    
    * @param [out]      perm     The rcm order.    
    * @return           Return error message.
    */
   template <typename T>
   int CsrSubMatrixNdHost(CsrMatrixClass<T> &A, vector_int &rowscols, vector_int &perm);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function computes the ND ordering of a CSR matrix A.
    * @details This function computes the ND ordering of a CSR matrix A.
    * 
    * @param [in]       A        The target CSR matrix.  
    * @param [out]      perm     The RCM permutation.  
    * @return           Return error message.
    */
   template <typename T>
   int CsrMatrixNdHost(CsrMatrixClass<T> &A, vector_int &perm);

   /**                                                                                                                                                                                                                                                                   
    * @brief   This function computes the RCM ordering of a submatrix A[rowscols,rowscols] + trans(A[rowscols,rowscols]).
    * @details This function computes the RCM ordering of a submatrix A[rowscols,rowscols] + trans(A[rowscols,rowscols]).
    * 
    * @param [in]       A        The target CSR matrix.                                                                                                           
    * @param [in]       rowscols The target rows/cols.    
    * @param [out]      perm     The rcm order.    
    * @return           Return error message.
    */
   template <typename T>
   int CsrSubMatrixRcmHost(CsrMatrixClass<T> &A, vector_int &rowscols, vector_int &perm);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function computes the RCM ordering of a CSR matrix A+trans(A).
    * @details This function computes the RCM ordering of A+trans(A). \n
    *          Connected component will be ordered by the node number of the first node in the original A in ascending order.
    *          So, if you have a block diagonal matrix, using this RCM function won't influence the block diagonal structure.
    * 
    * @param [in]       A        The target CSR matrix.  
    * @param [out]      perm     The RCM permutation.  
    * @return           Return error message.
    */
   template <typename T>
   int CsrMatrixRcmHost(CsrMatrixClass<T> &A, vector_int &perm);

   /**                                                                                                                                                                                                                                                                   
    * @brief   This function find the next root for the RCM ordering of graph G
    * @details This function find the next root for the RCM ordering of graph G. \n
    *          It serachs for the lowest degree node in the next connect component.
    * 
    * @param [in]       G        The target graph in CSR format.  
    * @param [in]       marker   Helper array of length equals to the size of G.  
    * @param [out]      root     The node number of the root node.      
    * @return           Return error message.
    */
   template <typename T>
   int CsrMatrixRcmRootHost(CsrMatrixClass<T> &G, vector_int &marker, int &root);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function computes the RCM numbering for a connect component start from root.
    * @details This function computes the RCM numbering for a connect component start from root.
    * 
    * @param [in]       G        The target graph in CSR format.  
    * @param [in]       root     The node number of the root node.      
    * @param [in]       marker   Helper array of length equals to the size of G. 
    *                   On return, value inside will be change so that we know this connect component is visited.
    * @param [in,out]   perm     The permutation of this connect component will be in here. 
    * @param [in]       current_num   Start number of the current connect component. 
    * @return           Return error message.
    */
   template <typename T>
   int CsrMatrixRcmNumberingHost(CsrMatrixClass<T> &G, int root, vector_int &marker, vector_int &perm, int &current_num);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function find an end of the pseudo-peripheral starting from the root.
    * @details This function find an end of the pseudo-peripheral starting from the root.
    * 
    * @param [in]       G        The target graph in CSR format.  
    * @param [in]       root     The node number of the root node.      
    * @param [in]       marker   Helper array of length equals to the size of G.
    * @return           Return error message.
    * @note    Reference: Gibbs, Norman E., William G. Poole, Jr, and Paul K. Stockmeyer. 
    *          "An algorithm for reducing the bandwidth and profile of a sparse matrix." 
    *          SIAM Journal on Numerical Analysis 13.2 (1976): 236-250.                                                                                           
    */
   template <typename T>
   int CsrMatrixRcmPerphnHost(CsrMatrixClass<T> &G, int &root, vector_int &marker);

   /**                                                                                                                                                                                                                                                                   
    * @brief   This function builds the level structure starting from node root.
    * @details This function builds the level structure starting from node root with BFS.
    * 
    * @param [in]       G        The target graph in CSR format.  
    * @param [in]       root     The node number of the root node.      
    * @param [in]       marker   Helper array of length equals to the size of G.
    * @param [out]      level    The level structure
    * @return           Return error message.
    */
   template <typename T>
   int CsrMatrixRcmBfsHost(CsrMatrixClass<T> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function frees the level structure.
    * @details This function frees the level structure.
    * 
    * @param [in,out]   level    The level structure
    * @return           Return error message.
    */
   int CsrMatrixRcmClearLevelHost(std::vector<std::vector<int> > &level);
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function swap two element in the perm array.
    * @details This function swap two element in the perm array.
    * 
    * @param [in,out]   perm  The permutation vector
    * @param [in]       a     One of the index.
    * @param [in]       b     The other index.
    * @return           Return error message.
    */
   int CsrMatrixRcmSwapHost(vector_int &perm, int a, int b);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function reverse an array between [start,end].
    * @details This function reverse an array between [start,end].
    * 
    * @param [in,out]   perm  The permutation vector
    * @param [in]       start The start index.
    * @param [in]       end   The end index(includd).
    * @return           Return error message.
    */
   int CsrMatrixRcmReverseHost(vector_int &perm, int start, int end);
   
   /**
    * @brief 	Call ParMETIS to partition A based on the adjancency graph of A.
    * @details Call ParMETIS to partition A based on the adjancency graph of A.
    * @note	   See ParMETIS documentation for more details.
    * @param [in]       vtxdist Vector of the number of the first vertex on each processor.
    * @param [in] 	   xadj Local I vector in the CSR format.
    * @param [in]       adjncy Local J vector in the CSR format.
    * @param [in,out]   num_dom Number of requested partitions. On exit the number of partitions we get.
    * @param [out]      map The domain number of each vertex.
    * @param [in]       parallel_log Parallel info struct.
    * @return           return GEMSLR_SUCESS or error information. 
    */
   int ParmetisKwayHost(vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, long int &num_dom, vector_long &map, parallel_log &parlog);
   
   /**
    * @brief 	Partition with log(p) levels nd, p is the number of processors.
    * @details Partition with log(p) levels nd, p is the number of processors.
    * @note	   See ParMETIS documentation for more details.
    * @param [in]       vtxdist Vector of the number of the first vertex on each processor.
    * @param [in] 	   xadj Local I vector in the CSR format.
    * @param [in]       adjncy Local J vector in the CSR format.
    * @param [in,out]   num_dom Number of requested partitions. On exit the number of partitions we get.
    * @param [out]      map The domain number of each vertex.
    * @param [in]       parallel_log Parallel info struct.
    * @return           return GEMSLR_SUCESS or error information. 
    */
   int ParmetisNodeND(vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, long int &num_dom, vector_long &map, parallel_log &parlog);
   
   /**
    * @brief   The subspace iteration procedure on a matrix. Note that we do not check for convergence. Apply a fixed number of iterations.
    * @details The subspace iteration procedure on a matrix. Note that we do not check for convergence. Apply a fixed number of iterations.
    * @note    Template function, matrix requires SetupVectorPtrStr, GetNumRowsLocal, MatVec, and GetDataLocation functions.
    * @param [in]       A The target matrix with Matvec and GetNumRows function.
    * @param [in]       k the dimension of the subspace.
    * @param [in]       its (Maximum) Number of iterations.
    * @param [in,out]   V Matrix holding the final Q.
    * @param [in,out]   H Matrix holding the final R.
    * @param [in]       tol Not yet in used.
    * @param [out]      nmvs Number of matrix vector products applied.
    * @return           Return error message.                                                                                                                                                       
    */
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int PargemslrSubSpaceIteration( MatrixType &A, int k, int its, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, RealDataType tol, int &nmvs);
   
   /**
    * @brief   The standard Arnoldi procedure on a matrix.
    * @details The standard Arnoldi procedure on a matrix.\n
    *          Can start from a certan vector instead of the first vector in V.
    * @note    Template function, matrix requires SetupVectorPtrStr, GetNumRowsLocal, MatVec, and GetDataLocation functions.
    * @param [in]       A The target matrix with Matvec and GetNumRows function.
    * @param [in]       mstart the index of the starting vector (start from V_mstart).
    * @param [in]       msteps (Maximum) Number of Arnoldi steps is msteps - mstart.
    * @param [in,out]   V Matrix holding the Krylov basis, space should be reserved, V(:,0:mstart) should be set already.
    * @param [in,out]   H The upper Hessenberg matrix generated by Arnoldi, space should be reserved.
    * @param [in]       tol_orth the tolorance for the Arnoldi.
    * @param [in]       tol_reorth the tolorance for the reorthgonation, set to -1.0 to turn off reorthgonation.
    * @param [out]      nmvs Number of matrix vector products applied.
    * @return           # of steps we actually do, might break before reaching msteps                                                                                                                                                       
    */
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int PargemslrArnoldiNoRestart( MatrixType &A, int mstart, int msteps, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, RealDataType tol_orth, RealDataType tol_reorth, int &nmvs);
   
   /**
    * @brief   The thick-restart Arnoldi without locking. This version should be used when convergence tolerance is high.
    * @details The thick-restart Arnoldi without locking. This version should be used when convergence tolerance is high.
    * @note    Template function, matrix requires SetupVectorPtrStr, GetNumRowsLocal, MatVec, and GetDataLocation functions.
    * @param [in]       A The target matrix with Matvec and GetNumRows function.
    * @param [in]       msteps Initial number of Arnoldi steps.
    * @param [in]       maxits Maximum number of restarts.
    * @param [in]       rank The number of convergenced eigenvalues to compute.
    * @param [in]       rank2 The number of convergenced eigenvalues to keep.
    * @param [in]       tr_fact Real value between (0,1), thick-restart factor. Control the size of thick-restart. \n
    *                   Each restart would restart with number of convergenced plus number of unconvergenced eigenvalue * tr_fact.
    * @param [in]       tol_eig The tolorance of eigenvalue convergence.
    * @param [in]       weight Compute eigenvalue has larger weight(val) first.
    * @param [in,out]   V Matrix holding the Krylov basis, space should be reserved, V(:,0:mstart) should be set already.
    * @param [in,out]   H The upper Hessenberg matrix generated by Arnoldi, space should be reserved.
    * @param [in]       tol_orth the tolorance for the Arnoldi.
    * @param [in]       tol_reorth the tolorance for the reorthgonation, set to -1.0 to turn off reorthgonation.
    * @param [out]      nmvs Number of matrix vector products applied.
    * @return           # of convergenced eigenvalues computed.                                                                                                                                                       
    */
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int PargemslrArnoldiThickRestartNoLock( MatrixType &A, int msteps, int maxits, int rank, int rank2, RealDataType truncate, RealDataType tr_fact, RealDataType tol_eig, RealDataType eig_target_mag, RealDataType eig_truncate, RealDataType (*weight)(ComplexValueClass<RealDataType>), DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, RealDataType tol_orth, RealDataType tol_reorth, int &nmvs);
   
   /**
    * @brief   Pick convergenced eigenvalues and unconvergenced eigenvalues.
    * @details Pick convergenced eigenvalues and unconvergenced eigenvalues.
    * @param [in]    H The H matrix after Schur decomposition.
    * @param [in]    Q The Q matrix after Schur decomposition.
    * @param [in]    h_last The H(m+1.m) value.
    * @param [in]    weight The weight function.
    * @param [out]   ncov The number of convergenced eigenvalues.
    * @param [out]   nicov The number of unconvergenced eigenvalues.
    * @param [in]    wr The real part of eigenvalues.
    * @param [in]    wi The imag part of eigenvalues.
    * @param [out]   icov The index of convergenced eigenvalues.
    * @param [out]   iicov The index of unconvergenced eigenvalues.
    * @param [out]   dcov The distance of convergenced eigenvalues.
    * @param [out]   dicov The distance of unconvergenced eigenvalues.
    * @return     Return error message.
    */
   template <typename T>
   int PargemslrArnoldiThickRestartChooseEigenValuesReal( DenseMatrixClass<T> &H, DenseMatrixClass<T> &Q, T h_last, T (*weight)(ComplexValueClass<T>), T truncate, int &ncov, int &nicov, int &nsatis, T tol_eig, T eig_target_mag, T eig_truncate, bool &cut, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, vector_int &icov, vector_int &iicov, vector_int &isatis, SequentialVectorClass<T> &dcov, SequentialVectorClass<T> &dicov, SequentialVectorClass<T> &dsatis);
   
   /**
    * @brief   Pick convergenced eigenvalues and unconvergenced eigenvalues.
    * @details Pick convergenced eigenvalues and unconvergenced eigenvalues.
    * @param [in]    H The H matrix after Schur decomposition.
    * @param [in]    Q The Q matrix after Schur decomposition.
    * @param [in]    h_last The H(m+1.m) value.
    * @param [in]    weight The weight function.
    * @param [out]   ncov The number of convergenced eigenvalues.
    * @param [out]   nicov The number of unconvergenced eigenvalues.
    * @param [in]    w The eigenvalues.
    * @param [out]   icov The index of convergenced eigenvalues.
    * @param [out]   iicov The index of unconvergenced eigenvalues.
    * @param [out]   dcov The distance of convergenced eigenvalues.
    * @param [out]   dicov The distance of unconvergenced eigenvalues.
    * @return     Return error message.
    */
   template <typename DataType, typename RealDataType>
   int PargemslrArnoldiThickRestartChooseEigenValuesComplex( DenseMatrixClass<DataType> &H, DenseMatrixClass<DataType> &Q, DataType h_last, RealDataType (*weight)(DataType), RealDataType truncate, int &ncov, int &nicov, int &nsatis, RealDataType tol_eig, RealDataType eig_target_mag, RealDataType eig_truncate, bool &cut, SequentialVectorClass<DataType> &w, vector_int &icov, vector_int &iicov, vector_int &isatis, SequentialVectorClass<RealDataType> &dcov, SequentialVectorClass<RealDataType> &dicov, SequentialVectorClass<RealDataType> &dsatis);
   
   /**
    * @brief   The current V is a invarient subspace, pick a new vector to restart.
    * @details The current V is a invarient subspace, pick a new vector to restart.
    * @param [in]       V The V at the current step.
    * @param [in]       H The H at the current step.
    * @param [in]       m The size of Hm.
    * @param [in]       tol_orth The threshold for orthgonal.
    * @param [in]       tol_reorth The threshold for reorthgonal.
    * @param [in]       v Working vector pointer, set by SetupVectorPtrStr().
    * @return     Return error message.
    */
   template <class VectorType, typename DataType, typename RealDataType>
   int PargemslrArnoldiThickRestartBuildThickRestartNewVector( DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, int m, RealDataType tol_orth, RealDataType tol_reorth, VectorType &v);
   
   /**
    * @brief   The classical Gram-Schmidt with re-orthgonal (always one step of re-orth).
    * @details The classical Gram-Schmidt with re-orthgonal (always one step of re-orth).
    * @param [in]       w Parallel vector to project to I-VV'.
    * @param [in,out]   V Matrix holding the Krylov basis.
    * @param [in,out]   H The upper Hessenberg matrix generated by Arnoldi.
    * @param [out]      t Scalar to hold ||w||.
    * @param [in]       k Number of columns used in V, exclude w.
    * @param [in]       tol_orth tol used to check break.
    * @return           return GEMSLR_SUCESS or error information.                                                                                                                              
    */
   template <class VectorType, typename DataType, typename RealDataType>
   int PargemslrCgs2( VectorType &w, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, RealDataType &t, int k, RealDataType tol_orth);
   
   /**
    * @brief   The modified Gram-Schmidt with re-orthgonal.
    * @details The modified Gram-Schmidt with re-orthgonal.
    * @param [in]       w Parallel vector to project to I-VV'.
    * @param [in,out]   V Matrix holding the Krylov basis.
    * @param [in,out]   H The upper Hessenberg matrix generated by Arnoldi.
    * @param [out]      t Scalar to hold ||w||.
    * @param [in]       k Number of columns used in V, exclude w.
    * @param [in]       tol_orth tol used to check break.
    * @param [in]       tol_reorth tol used in reorth for higher accuracy.
    * @return           return GEMSLR_SUCESS or error information.                                                                                                                              
    */
   template <class VectorType, typename DataType, typename RealDataType>
   int PargemslrMgs( VectorType &w, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, RealDataType &t, int k, RealDataType tol_orth, RealDataType tol_reorth);
   
   /**
    * @brief   The modified Gram-Schmidt with re-orthgonal without H matrix.
    * @details The modified Gram-Schmidt with re-orthgonal without H matrix.
    * @param [in]       w Parallel vector to project to I-VV'.
    * @param [in,out]   V Matrix holding the Krylov basis.
    * @param [out]      t Scalar to hold ||w||.
    * @param [in]       k Number of columns used in V, exclude w.
    * @param [in]       tol_orth tol used to check break.
    * @param [in]       tol_reorth tol used in reorth for higher accuracy.
    * @return           return GEMSLR_SUCESS or error information.                                                                                                                              
    */
   template <class VectorType, typename DataType, typename RealDataType>
   int PargemslrOrthogonal( VectorType &w, DenseMatrixClass<DataType> &V, RealDataType &t, int k, RealDataType tol_orth, RealDataType tol_reorth);
   
   /**
    * @brief   Plot the dense matrix to the terminal output. Function for testing purpose.
    * @details Plot the dense matrix to the terminal output. Function for testing purpose.
    * @param [in]   A The target matrix.
    * @param [in]   conditiona First condition.
    * @param [in]   conditionb Secend condition, only spy when conditiona == conditionb.
    * @param [in]   width The plot width.
    * @return       Return error message.
    */
   template <typename T>
   int DenseMatrixPlotHost( DenseMatrixClass<T> &A, int conditiona, int conditionb, int width);
   
   /**
    * @brief   Plot the csr matrix to the terminal output. Function for testing purpose.
    * @details Plot the csr matrix to the terminal output. Function for testing purpose.
    * @param [in]   A The target matrix.
    * @param [in]   perm The permutation vector. Set to NULL to plot(A), otherwise plot(A(perm,perm)).
    * @param [in]   conditiona First condition.
    * @param [in]   conditionb Secend condition, only spy when conditiona == conditionb.
    * @param [in]   width The plot width.
    * @return       Return error message.
    */
   template <typename T>
   int CsrMatrixPlotHost( CsrMatrixClass<T> &A, int *perm, int conditiona, int conditionb, int width);
   
   /**
    * @brief   Read a coo matrix from matrix marker format file.
    * @details Read a coo matrix from matrix marker format file.
    * @param [out]  coo The return matrix.
    * @param [in]   matfile The file name.
    * @param [in]   idxin The index base of the input matrix, 0-based or 1-based.
    * @param [in]   idxin The index base of the output matrix, 0-based or 1-based.
    * @return       Return error message.
    */
   int CooMatrixReadFromFile(CooMatrixClass<float> &coo, const char *matfile, int idxin, int idxout);
   
   /**
    * @brief   Read a coo matrix from matrix marker format file.
    * @details Read a coo matrix from matrix marker format file.
    * @param [out]  coo The return matrix.
    * @param [in]   matfile The file name.
    * @param [in]   idxin The index base of the input matrix, 0-based or 1-based.
    * @param [in]   idxin The index base of the output matrix, 0-based or 1-based.
    * @return       Return error message.
    */
   int CooMatrixReadFromFile(CooMatrixClass<double> &coo, const char *matfile, int idxin, int idxout);
   
   /**
    * @brief   Read a coo matrix from matrix marker format file.
    * @details Read a coo matrix from matrix marker format file.
    * @param [out]  coo The return matrix.
    * @param [in]   matfile The file name.
    * @param [in]   idxin The index base of the input matrix, 0-based or 1-based.
    * @param [in]   idxin The index base of the output matrix, 0-based or 1-based.
    * @return       Return error message.
    */
   int CooMatrixReadFromFile(CooMatrixClass<complexs> &coo, const char *matfile, int idxin, int idxout);
   
   /**
    * @brief   Read a coo matrix from matrix marker format file.
    * @details Read a coo matrix from matrix marker format file.
    * @param [out]  coo The return matrix.
    * @param [in]   matfile The file name.
    * @param [in]   idxin The index base of the input matrix, 0-based or 1-based.
    * @param [in]   idxin The index base of the output matrix, 0-based or 1-based.
    * @return       Return error message.
    */
   int CooMatrixReadFromFile(CooMatrixClass<complexd> &coo, const char *matfile, int idxin, int idxout);
   
   /**
    * @brief   Recursive KWay partition call.
    * @details Recursive KWay partition call.
    * @param   [in]     A The target matrix.
    * @param   [in]     vertexset Use edge seperator or vertex seperator.
    * @param   [in]     clvl The current level.
    * @param   [in,out] tlvl The total number of levels.
    * @param   [in]     num_dom The target number of domains on this level, must be no less than 2.
    * @param   [in]     minsep The minimal size of the edge seperator.
    * @param   [in]     kmin The minimal number of domains on this level.
    * @param   [in]     kfactor The reduce factor of the number of domains.
    * @param   [in]     map_v The map from node number to domain number.
    * @param   [in]     mapptr_v The vector holds the start domian number on each level.
    * @return     Return error message.
    */
   template <typename T>
   int SetupPermutationRKwayRecursive( CsrMatrixClass<T> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v);
   
   /**
    * @brief   Recursive ND partition call.
    * @details Recursive ND partition call.
    * @param   [in]     A The target matrix.
    * @param   [in]     vertexset Use edge seperator or vertex seperator.
    * @param   [in]     clvl The current level.
    * @param   [in,out] tlvl The total number of levels.
    * @param   [in]     num_dom The target number of domains on this level.
    * @param   [in]     minsep The minimal size of the edge seperator.
    * @param   [in]     level_str The level structure. level_str[i][j] is the nodes in the i-th component, of the j-th level in this component.
    * @return     Return error message.
    */
   template <typename T>
   int SetupPermutationNDRecursive( CsrMatrixClass<T> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str);
   
   /**
    * @brief   Compress a certain level of level_str from SetupPermutationNDRecursive into a give number of domains.
    * @details Compress a certain level of level_str from SetupPermutationNDRecursive into a give number of domains.
    * @param   [in]     level_stri The level of a level_str.
    * @param   [in,out] ndom The target number of domains, on return the ndom we get.
    * @return     Return error message.
    */
   int SetupPermutationNDCombineLevels(std::vector<vector_int> &level_stri, int &ndom);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   The main recursive Kway reordering function.
    * @details The first part of the function conatins the following steps: \n 
    *          1) On each level, partition the current adjacency graph with kway partitioning. \n
    *          2) Find the adjacency graph of the interface matrix. \n
    *          3) Repeat until we reach the last level or the interface matrix is too small. \n
    *          After that, we re-assign all nodes to their new processor, and apply communication to transfer data.
    * 
    * @param [in]       A        The target parallel CSR matrix.  
    * @param   [in]     vertexset Use edge seperator or vertex seperator.                                                      
    * @param [in,out]   nlev     The target number of levels.
    * @param [in]       ncomp    Number of components.
    * @param [in]       minsep   Minimal size of the seperator.
    * @param [in]       kmin     Minimal size of the number of subdomains on a level.
    * @param [in]       kfactor  At each level, ncomp is deviced by kfactor, for example, when ncomp = 16 and kfactor = 2, size would be 16, 8, 4, ....
    * @param [out]      map_v    The map from node number to domain number.
    * @param [out]      mapptr_v The vector holds the start domian number on each level.
    * @param [in]       bj_last  Should we treat last level with block Jacobi?
    * @note             We assume that A_mat has symmetric pattern.
    */
   template <typename T>
   int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<T> &A, bool vertexsep, int &nlev, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This is the recursive function to build the mapping information.
    * @details The function conatins the following steps: \n 
    *          1) On each level, partition the current adjacency graph with kway partitioning. \n
    *          2) Find the adjacency graph of the interface matrix. \n
    *          3) Repeat until we reach the last level or the interface matrix is too small. \n
    * 
    * @param [in]       vtxdist  The distribution of rows for parMETIS.                                         
    * @param [in]       xadj     The column ptr for parMETIS.    
    * @param [in]       adjncy   The column numbers for parMETIS.   
    * @param [in]       vertexsep Do we find the vertex separator?
    * @param [in]       clvl     Current level number. 
    * @param [in,out]   tlvl     The max number of levels. If the algorithm stops at a level lower than desired level, reset tlvl to current number of levels.
    * @param [in]       ncomp    The k for the kway partition on the current level.
    * @param [in]       minsep   The minimal size of the edge separator. The algorithm stops if the size of the current interface matrix is too small.
    * @param [in]       kmin     The minimal value of k for the kway partition.
    * @param [in]       kfactor  On the next level, ncomp = ncomp/kfactor. We might want more subdomains on higher levels.
    * @param [out]      map_v    The subdomain number of each node.
    * @param [out]      mapptr_v The vector holds the start domian number on each level.
    * @param [in]       bj_last  Should we treat last level with block Jacobi?
    * @param [in]       parallel_log Parallel info struct.
    * @return           return PARGEMSLR_SUCCESS or error information.
    * @note             We assume that A_mat has symmetric pattern.
    */
   int SetupPermutationParallelRKwayRecursive(vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, bool vertexsep, int clvl, int &tlvl, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function finds the separator based on the map information.
    * @details This function finds the separator based on the map information.
    * @param [in]       vtxdist  The distribution of rows for parMETIS.                                         
    * @param [in]       xadj     The column ptr for parMETIS.    
    * @param [in]       adjncy   The column numbers for parMETIS.    
    * @param [in]       vertexsep Do we find the vertex separator? 
    * @param [in]       vtxdist_s   The distribution of rows for the edge separator.                                         
    * @param [in]       xadj_s      The column ptr for the edge separator.    
    * @param [in]       adjncy_s    The column numbers for the edge separator.
    * @param [in]       map      The subdomain number of each node.
    * @param [in]       num_dom      The number ofsubdomain number of each node.
    * @param [out]      vtxsep   vtxsep[i] > 0 if node i is inside the edge separator.
    * @param [in]       parallel_log Parallel info struct.
    * @return           return PARGEMSLR_SUCCESS or error information.
    * @note             We assume that A_mat has symmetric pattern.
    */
   int ParallelRKwayGetSeparator( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, bool vertexsep, vector_long &vtxdist_s,  vector_long &xadj_s,  vector_long &adjncy_s, vector_long &map, int num_dom, vector_int &vtxsep, parallel_log &parlog);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function finds the separator based on the map information given by GetSeparatorNumSubdomains function.
    * @details This function finds the separator based on the map information given by GetSeparatorNumSubdomains function.                                       
    * @param [in]       vtxdist  The distribution of rows for parMETIS.                                         
    * @param [in]       xadj     The column ptr for parMETIS.    
    * @param [in]       adjncy   The column numbers for parMETIS.    
    * @param [in]       vertexsep Do we find the vertex separator? 
    * @param [in]       vtxdist_s   The distribution of rows for the edge separator.                                         
    * @param [in]       xadj_s      The column ptr for the edge separator.    
    * @param [in]       adjncy_s    The column numbers for the edge separator.
    * @param [in]       map      The subdomain number of each node, <0 means the separator.
    * @param [in]       num_dom      The number ofsubdomain number of each node.
    * @param [out]      vtxsep   vtxsep[i] > 0 if node i is inside the edge separator.
    * @param [in]       parallel_log Parallel info struct.
    * @return           return PARGEMSLR_SUCCESS or error information.
    * @note             We assume that A_mat has symmetric pattern.
    */
   int ParallelRKwayGetSeparator2( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, bool vertexsep, vector_long &vtxdist_s,  vector_long &xadj_s,  vector_long &adjncy_s, vector_int &map, int num_dom, vector_int &vtxsep, parallel_log &parlog);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This is the recursive function to build the mapping information, using an approximate vetrex separator (not the minimal).
    * @details This is the recursive function to build the mapping information, using an approximate vetrex separator (not the minimal).
    * @param [in]       A        The target matrix.
    * @param [in]       clvl     Current level number. 
    * @param [in,out]   tlvl     The max number of levels. If the algorithm stops at a level lower than desired level, reset tlvl to current number of levels.
    * @param [in]       ncomp    The k for the kway partition on the current level.
    * @param [in]       minsep   The minimal size of the edge separator. The algorithm stops if the size of the current interface matrix is too small.
    * @param [in]       kmin     The minimal value of k for the kway partition.
    * @param [in]       kfactor  On the next level, ncomp = ncomp/kfactor. We might want more subdomains on higher levels.
    * @param [out]      map_v    The subdomain number of each node.
    * @param [out]      mapptr_v The vector holds the start domian number on each level.
    * @param [in]       bj_last  Should we treat last level with block Jacobi?
    * @param [in]       parallel_log Parallel info struct.
    * @return           return PARGEMSLR_SUCCESS or error information.
    * @note             We assume that A_mat has symmetric pattern.
    */
   template <typename T>
   int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<T> &A, int clvl, int &tlvl, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function build the k-way partition where k is the power of 2, and provide an approximate vertex saperator.
    * @details This function build the k-way partition where k is the power of 2, and provide an approximate vertex saperator.
    * @param [in]       A        The target matrix.
    * @param [in, out]  ncomp    The k for the kway partition, on return, the actual number of subdomains.
    * @param [out]      map_v    The subdomain number of each node, starting from 0 to ncomp-1. if map_v[i] == ncomp this is the separator.
    * @param [out]      perm_sep The permutation, A(perm_sep, perm_sep) is the global coupling matrix.
    * @param [in]       parallel_log Parallel info struct.
    * @return           return PARGEMSLR_SUCCESS or error information.
    * @note             We assume that A has symmetric pattern.
    */
   template <typename T>
   int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<T> &A, long int &ncomp, vector_int &map_v, vector_long &perm_sep, parallel_log &parlog);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This recursie call function for building the k-way partition where k is the power of 2, and provide an approximate vertex saperator.
    * @details This recursie call function for building the k-way partition where k is the power of 2, and provide an approximate vertex saperator.
    * @param [in]       A        The target matrix.
    * @param [in]       clvl     The current level starting from 0.
    * @param [in]       tlvl     The total number of levels, map from level to domain: 2->2, 3->4, 4->8 ....
    * @param [in, out]  succeed  Tell if the partition get desired result.
    * @param [out]      map_v    The subdomain number of each node, starting from 0 to ncomp-1. if map_v[i] == ncomp this is the separator.
    * @param [in]       parallel_log Parallel info struct.
    * @return           return PARGEMSLR_SUCCESS or error information.
    * @note             We assume that A has symmetric pattern.
    */
   template <typename T>
   int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<T> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   This function finds the vertex separator based on the map information. A very simple appraoch, find the edge separator and only keep one side.
    * @details This function finds the vertex separator based on the map information. A very simple appraoch, find the edge separator and only keep one side.
    * @param [in]       vtxdist  The distribution of rows for parMETIS.                                         
    * @param [in]       xadj     The column ptr for parMETIS.    
    * @param [in]       adjncy   The column numbers for parMETIS. 
    * @param [in, out]  map      The subdomain number of each node, on return, edge separator would be marked as 2.
    * @param [out]      ndom1    Global number of interior nodes in dom1.
    * @param [out]      ndom2    Global number of interior nodes in dom2.
    * @param [out]      edge_cut Global number of edge_cut.
    * @param [in]       parallel_log Parallel info struct.
    * @return           return PARGEMSLR_SUCCESS or error information.
    * @note             We assume that A_mat has symmetric pattern.
    */
   int ParallelNDGetSeparator( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, vector_long &map, long int &ndom1, long int &ndom2, long int &edge_cut, parallel_log &parlog);
   
   /**                                                                                                                                                                                                                                                                   
    * @brief   The main ND reordering function using ND in the ParMETIS.
    * @details The main ND reordering function using ND in the ParMETIS. The number of level is fixed once the np is given.
    * @param [in]       A           The target parallel CSR matrix. 
    * @param [in]       vertexsep   Use edge seperator or vertex seperator (currently not in use).                                                       
    * @param [in,out]   nlev        The target number of levels (currently not in use).
    * @param [in]       minsep      Minimal size of the seperator (currently not in use).
    * @return           map_v       The map from node number to domain number.
    * @return           mapptr_v    The vector holds the start domian number on each level.
    */
   template <typename T>
   int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<T> &A, bool vertexsep, int &nlev, long int minsep, vector_int &map_v, vector_int &mapptr_v);

   /**
    * @brief   Setup the local order splits the interior and exterior nodes.
    * @details Setup the local order splits the interior and exterior nodes.
    * @param   [in]     parcsr_in The input matrix.
    * @param   [out]    local_perm The local permutation.
    * @param   [out]    nI The number of local interior nodes.
    * @param   [out]    B_mat The local B matrix.
    * @param   [out]    E_mat The local E matrix.
    * @param   [out]    F_mat The local F matrix.
    * @param   [out]    C_mat The global coupling matrix.
    * @param   [in]     perm_option The permutation option. kIluReorderingNo, kIluReorderingRCM, kIluReorderingAMD.
    * @param   [in]     bool perm_c Permute B only or B and C?
    * @return     Return error message.
    */
   template <typename T>
   int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<T> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<T> &B_mat, CsrMatrixClass<T> &E_mat, CsrMatrixClass<T> &F_mat, ParallelCsrMatrixClass<T> &C_mat, int perm_option, bool perm_c);
   
}

#endif
