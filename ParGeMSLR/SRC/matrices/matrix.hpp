#ifndef PARGEMSLR_MATRIX_H
#define PARGEMSLR_MATRIX_H

/**
 * @file  matrix.hpp
 * @brief The virtual matrix classes.
 */

#include "../utils/utils.hpp"
#include "../utils/parallel.hpp"
#include "../utils/memory.hpp"
#include "../vectors/vector.hpp"

namespace pargemslr
{
   
   /**
    * @brief   The virtual class of all matrix classes.
    * @details The virtual class of all matrix classes.
    */
   template <typename T>
   class MatrixClass: public parallel_log
   {
   public:
      
      /**
       * @brief   The constructor of MatrixClass.
       * @details The constructor of MatrixClass.
       */
      MatrixClass();
      
      /**
       * @brief   The copy constructor of MatrixClass.
       * @details The copy constructor of MatrixClass.
       * @param   [in]   mat The other matrix.
       */
      MatrixClass(const MatrixClass<T> &mat);
      
      /**
       * @brief   The move constructor of MatrixClass.
       * @details The move constructor of MatrixClass.
       * @param   [in]   mat The other matrix.
       */
      MatrixClass(MatrixClass<T> &&mat);
      
      /**
       * @brief   Free the current matrix.
       * @details Free the current matrix.
       * @return     Return error message.
       */
      virtual int Clear();
      
      /**
       * @brief   The destructor of MatrixClass.
       * @details The destructor of MatrixClass.
       */
      virtual ~MatrixClass();
      
      /**
       * @brief   Get the data location of the matrix.
       * @details Get the data location of the matrix.
       * @return     Return the data location of the matrix.
       */
      virtual int GetDataLocation() const;
      
      /**
       * @brief   Get the local number of rows of the matrix.
       * @details Get the local number of rows of the matrix.
       * @return     Return the local number of rows of the matrix.
       */
      virtual int GetNumRowsLocal() const = 0;
      
      /**
       * @brief   Get the local number of columns of the matrix.
       * @details Get the local number of columns of the matrix.
       * @return     Return the local number of columns of the matrix.
       */
      virtual int GetNumColsLocal() const = 0;
      
      /**
       * @brief   Get the number of nonzeros in this matrix.
       * @details Get the number of nonzeros in this matrix.
       * @return     Return the number of nonzeros in this matrix.
       */
      virtual long int  GetNumNonzeros() const;
      
      /**
       * @brief   Create an indentity matrix.
       * @details Create an indentity matrix.
       * @return     Return error message.
       */
      virtual int Eye();
      
      /**
       * @brief   Fill the matrix pattern with constant value.
       * @details Fill the matrix pattern with constant value.
       * @param   [in]   v The value to be filled.
       * @return     Return error message.
       */
      virtual int Fill(const T &v);
      
      /**
       * @brief   Scale the matrix.
       * @details Scale the matrix.
       * @param   [in]   alpha The scale.
       * @return     Return error message.
       */
      virtual int Scale(const T &alpha);
      
      /**
       * @brief   In place Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @details In place Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The first vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The second vector.
       * @return           Return error message.
       */
      virtual int MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y);
      
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
      virtual int MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, const VectorClass<T> &y, VectorClass<T> &z);
      
      /**
       * @brief      Move the data to another memory location.
       * @details    Move the data to another memory location.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      virtual int MoveData( const int &location);
      
      /**
       * @brief      Return the precision of the matrix.
       * @details    Return the precision of the matrix.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      PrecisionEnum GetPrecision() const;
      
   };
   
   typedef MatrixClass<float>     matrix_base_float;
   typedef MatrixClass<double>    matrix_base_double;
   typedef MatrixClass<complexs>  matrix_base_complexs;
   typedef MatrixClass<complexd>  matrix_base_complexd;
   template<> struct PargemslrIsComplex<MatrixClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<MatrixClass<complexd> > : public std::true_type {};
   
}

#endif
