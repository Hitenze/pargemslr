
#include <iostream>
#include "../utils/utils.hpp"
#include "matrix.hpp"
#include "matrixops.hpp"

namespace pargemslr
{
   
	template <typename T>
   MatrixClass<T>::MatrixClass()
   {
   }
   template MatrixClass<float>::MatrixClass();
   template MatrixClass<double>::MatrixClass();
   template MatrixClass<complexs>::MatrixClass();
   template MatrixClass<complexd>::MatrixClass();
   
	template <typename T>
   MatrixClass<T>::MatrixClass(const MatrixClass<T> &mat) : ParallelLogClass(mat)
   {
   }
   template MatrixClass<float>::MatrixClass(const MatrixClass<float> &mat);
   template MatrixClass<double>::MatrixClass(const MatrixClass<double> &mat);
   template MatrixClass<complexs>::MatrixClass(const MatrixClass<complexs> &mat);
   template MatrixClass<complexd>::MatrixClass(const MatrixClass<complexd> &mat);
   
	template <typename T>
   MatrixClass<T>::MatrixClass( MatrixClass<T> &&mat) : ParallelLogClass(std::move(mat))
   {
   }
   template MatrixClass<float>::MatrixClass( MatrixClass<float> &&mat);
   template MatrixClass<double>::MatrixClass( MatrixClass<double> &&mat);
   template MatrixClass<complexs>::MatrixClass( MatrixClass<complexs> &&mat);
   template MatrixClass<complexd>::MatrixClass( MatrixClass<complexd> &&mat);
   
	template <typename T>
   int MatrixClass<T>::Clear()
   {
      /* base class clear */
      parallel_log::Clear();
      return PARGEMSLR_SUCCESS;
   }
   template int MatrixClass<float>::Clear();
   template int MatrixClass<double>::Clear();
   template int MatrixClass<complexs>::Clear();
   template int MatrixClass<complexd>::Clear();
   
   template <typename T>
   MatrixClass<T>::~MatrixClass()
   {
   }
   template MatrixClass<float>::~MatrixClass();
   template MatrixClass<double>::~MatrixClass();
   template MatrixClass<complexs>::~MatrixClass();
   template MatrixClass<complexd>::~MatrixClass();
   
   template <typename T>
   int MatrixClass<T>::GetDataLocation() const
   {
      PARGEMSLR_WARNING("Use default location host since virtual function GetDataLocation() not inplemented for this class.");
      return kMemoryHost;
   }
   template int MatrixClass<float>::GetDataLocation() const;
   template int MatrixClass<double>::GetDataLocation() const;
   template int MatrixClass<complexs>::GetDataLocation() const;
   template int MatrixClass<complexd>::GetDataLocation() const;
   
   template <typename T>
   long int MatrixClass<T>::GetNumNonzeros() const
   {
      PARGEMSLR_ERROR("virtual function GetNumNonzeros() not inplemented for this class.");
      return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
   }
   template long int MatrixClass<float>::GetNumNonzeros() const;
   template long int MatrixClass<double>::GetNumNonzeros() const;
   template long int MatrixClass<complexs>::GetNumNonzeros() const;
   template long int MatrixClass<complexd>::GetNumNonzeros() const;
   
   template <typename T>
   int MatrixClass<T>::Eye()
   {
      PARGEMSLR_ERROR("virtual function Eye() not inplemented for this class.");
      return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
   }
   template int MatrixClass<float>::Eye();
   template int MatrixClass<double>::Eye();
   template int MatrixClass<complexs>::Eye();
   template int MatrixClass<complexd>::Eye();
   
   template <typename T>
   int MatrixClass<T>::Fill(const T &v)
   {
      PARGEMSLR_ERROR("virtual function Fill(const T &v) not inplemented for this class.");
      return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
   }
   template int MatrixClass<float>::Fill(const float &v);
   template int MatrixClass<double>::Fill(const double &v);
   template int MatrixClass<complexs>::Fill(const complexs &v);
   template int MatrixClass<complexd>::Fill(const complexd &v);
   
   
   template <typename T>
   int MatrixClass<T>::Scale(const T &alpha)
   {
      PARGEMSLR_ERROR("virtual function Scale(const T &alpha) not inplemented for this class.");
      return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
   }
   template int MatrixClass<float>::Scale(const float &alpha);
   template int MatrixClass<double>::Scale(const double &alpha);
   template int MatrixClass<complexs>::Scale(const complexs &alpha);
   template int MatrixClass<complexd>::Scale(const complexd &alpha);
   
   template <typename T>
   int MatrixClass<T>::MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y)
   {
      PARGEMSLR_ERROR("virtual function MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y) not inplemented for this class.");
      return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
   }
   template int MatrixClass<float>::MatVec( char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   template int MatrixClass<double>::MatVec( char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   template int MatrixClass<complexs>::MatVec( char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   template int MatrixClass<complexd>::MatVec( char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
   template <typename T>
   int MatrixClass<T>::MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, const VectorClass<T> &y, VectorClass<T> &z)
   {
      PARGEMSLR_ERROR("virtual function MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, const VectorClass<T> &y, VectorClass<T> &y) not inplemented for this class.");
      return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
   }
   template int MatrixClass<float>::MatVec( char trans, const float &alpha, const VectorClass<float> &x, const float &beta, const VectorClass<float> &y, VectorClass<float> &z);
   template int MatrixClass<double>::MatVec( char trans, const double &alpha, const VectorClass<double> &x, const double &beta, const VectorClass<double> &y, VectorClass<double> &z);
   template int MatrixClass<complexs>::MatVec( char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, const VectorClass<complexs> &y, VectorClass<complexs> &z);
   template int MatrixClass<complexd>::MatVec( char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, const VectorClass<complexd> &y, VectorClass<complexd> &z);
   
   template <typename T>
   int MatrixClass<T>::MoveData( const int &location)
   {
      PARGEMSLR_WARNING("NO GPU support since virtual function MoveData( const int &location) not inplemented for this class.");
      return PARGEMSLR_SUCCESS;
   }
   template int MatrixClass<float>::MoveData( const int &location);
   template int MatrixClass<double>::MoveData( const int &location);
   template int MatrixClass<complexs>::MoveData( const int &location);
   template int MatrixClass<complexd>::MoveData( const int &location);
   
   template <typename T>
   PrecisionEnum MatrixClass<T>::GetPrecision() const
   {
      return GetMatrixPPrecision(this);
   }
   template PrecisionEnum MatrixClass<float>::GetPrecision() const;
   template PrecisionEnum MatrixClass<double>::GetPrecision() const;
   template PrecisionEnum MatrixClass<complexs>::GetPrecision() const;
   template PrecisionEnum MatrixClass<complexd>::GetPrecision() const;
   
}
