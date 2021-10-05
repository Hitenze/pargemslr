
#include <iostream>
#include "../utils/utils.hpp"
#include "matrix.hpp"
#include "matrixops.hpp"

namespace pargemslr
{
   /* ArnoldiMatrixClass */
   
	template <class VectorType, typename DataType>
   ArnoldiMatrixClass<VectorType, DataType>::ArnoldiMatrixClass()
   {
   }
   template ArnoldiMatrixClass<vector_seq_float, float>::ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_seq_double, double>::ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_seq_complexs, complexs>::ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_seq_complexd, complexd>::ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_par_float, float>::ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_par_double, double>::ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_par_complexs, complexs>::ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_par_complexd, complexd>::ArnoldiMatrixClass();
   
	template <class VectorType, typename DataType>
   ArnoldiMatrixClass<VectorType, DataType>::ArnoldiMatrixClass(const ArnoldiMatrixClass<VectorType, DataType> &mat)
   {
   }
   template ArnoldiMatrixClass<vector_seq_float, float>::ArnoldiMatrixClass(const ArnoldiMatrixClass<vector_seq_float, float> &mat);
   template ArnoldiMatrixClass<vector_seq_double, double>::ArnoldiMatrixClass(const ArnoldiMatrixClass<vector_seq_double, double> &mat);
   template ArnoldiMatrixClass<vector_seq_complexs, complexs>::ArnoldiMatrixClass(const ArnoldiMatrixClass<vector_seq_complexs, complexs> &mat);
   template ArnoldiMatrixClass<vector_seq_complexd, complexd>::ArnoldiMatrixClass(const ArnoldiMatrixClass<vector_seq_complexd, complexd> &mat);
   template ArnoldiMatrixClass<vector_par_float, float>::ArnoldiMatrixClass(const ArnoldiMatrixClass<vector_par_float, float> &mat);
   template ArnoldiMatrixClass<vector_par_double, double>::ArnoldiMatrixClass(const ArnoldiMatrixClass<vector_par_double, double> &mat);
   template ArnoldiMatrixClass<vector_par_complexs, complexs>::ArnoldiMatrixClass(const ArnoldiMatrixClass<vector_par_complexs, complexs> &mat);
   template ArnoldiMatrixClass<vector_par_complexd, complexd>::ArnoldiMatrixClass(const ArnoldiMatrixClass<vector_par_complexd, complexd> &mat);
   
	template <class VectorType, typename DataType>
   ArnoldiMatrixClass<VectorType, DataType>::ArnoldiMatrixClass( ArnoldiMatrixClass<VectorType, DataType> &&mat)
   {
   }
   template ArnoldiMatrixClass<vector_seq_float, float>::ArnoldiMatrixClass( ArnoldiMatrixClass<vector_seq_float, float> &&mat);
   template ArnoldiMatrixClass<vector_seq_double, double>::ArnoldiMatrixClass( ArnoldiMatrixClass<vector_seq_double, double> &&mat);
   template ArnoldiMatrixClass<vector_seq_complexs, complexs>::ArnoldiMatrixClass( ArnoldiMatrixClass<vector_seq_complexs, complexs> &&mat);
   template ArnoldiMatrixClass<vector_seq_complexd, complexd>::ArnoldiMatrixClass( ArnoldiMatrixClass<vector_seq_complexd, complexd> &&mat);
   template ArnoldiMatrixClass<vector_par_float, float>::ArnoldiMatrixClass( ArnoldiMatrixClass<vector_par_float, float> &&mat);
   template ArnoldiMatrixClass<vector_par_double, double>::ArnoldiMatrixClass( ArnoldiMatrixClass<vector_par_double, double> &&mat);
   template ArnoldiMatrixClass<vector_par_complexs, complexs>::ArnoldiMatrixClass( ArnoldiMatrixClass<vector_par_complexs, complexs> &&mat);
   template ArnoldiMatrixClass<vector_par_complexd, complexd>::ArnoldiMatrixClass( ArnoldiMatrixClass<vector_par_complexd, complexd> &&mat);
   
	template <class VectorType, typename DataType>
   int ArnoldiMatrixClass<VectorType, DataType>::Clear()
   {
      /* base class clear */
      return PARGEMSLR_SUCCESS;
   }
   template int ArnoldiMatrixClass<vector_seq_float, float>::Clear();
   template int ArnoldiMatrixClass<vector_seq_double, double>::Clear();
   template int ArnoldiMatrixClass<vector_seq_complexs, complexs>::Clear();
   template int ArnoldiMatrixClass<vector_seq_complexd, complexd>::Clear();
   template int ArnoldiMatrixClass<vector_par_float, float>::Clear();
   template int ArnoldiMatrixClass<vector_par_double, double>::Clear();
   template int ArnoldiMatrixClass<vector_par_complexs, complexs>::Clear();
   template int ArnoldiMatrixClass<vector_par_complexd, complexd>::Clear();
   
   template <class VectorType, typename DataType>
   ArnoldiMatrixClass<VectorType, DataType>::~ArnoldiMatrixClass()
   {
   }
   template ArnoldiMatrixClass<vector_seq_float, float>::~ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_seq_double, double>::~ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_seq_complexs, complexs>::~ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_seq_complexd, complexd>::~ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_par_float, float>::~ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_par_double, double>::~ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_par_complexs, complexs>::~ArnoldiMatrixClass();
   template ArnoldiMatrixClass<vector_par_complexd, complexd>::~ArnoldiMatrixClass();
   
   template <class VectorType, typename DataType>
   int ArnoldiMatrixClass<VectorType, DataType>::GetDataLocation() const
   {
      PARGEMSLR_WARNING("Use default location host since virtual function GetDataLocation() not inplemented for this class.");
      return kMemoryHost;
   }
   template int ArnoldiMatrixClass<vector_seq_float, float>::GetDataLocation() const;
   template int ArnoldiMatrixClass<vector_seq_double, double>::GetDataLocation() const;
   template int ArnoldiMatrixClass<vector_seq_complexs, complexs>::GetDataLocation() const;
   template int ArnoldiMatrixClass<vector_seq_complexd, complexd>::GetDataLocation() const;
   template int ArnoldiMatrixClass<vector_par_float, float>::GetDataLocation() const;
   template int ArnoldiMatrixClass<vector_par_double, double>::GetDataLocation() const;
   template int ArnoldiMatrixClass<vector_par_complexs, complexs>::GetDataLocation() const;
   template int ArnoldiMatrixClass<vector_par_complexd, complexd>::GetDataLocation() const;
   
   template <class VectorType, typename DataType>
   int ArnoldiMatrixClass<VectorType, DataType>::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const
   {
      PARGEMSLR_WARNING("Use global comm since virtual function GetMpiInfo(int &np, int &myid, MPI_Comm &comm) not inplemented for this class.");
      comm  = *(parallel_log::_gcomm);
      np    = parallel_log::_gsize;
      myid  = parallel_log::_grank;
      return PARGEMSLR_SUCCESS;
   }
   template int ArnoldiMatrixClass<vector_seq_float, float>::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
   template int ArnoldiMatrixClass<vector_seq_double, double>::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
   template int ArnoldiMatrixClass<vector_seq_complexs, complexs>::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
   template int ArnoldiMatrixClass<vector_seq_complexd, complexd>::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
   template int ArnoldiMatrixClass<vector_par_float, float>::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
   template int ArnoldiMatrixClass<vector_par_double, double>::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
   template int ArnoldiMatrixClass<vector_par_complexs, complexs>::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
   template int ArnoldiMatrixClass<vector_par_complexd, complexd>::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
   
   template <class VectorType, typename DataType>
   MPI_Comm ArnoldiMatrixClass<VectorType, DataType>::GetComm() const
   {
      PARGEMSLR_WARNING("Use global comm since virtual function GetComm() not inplemented for this class.");
      return *(parallel_log::_gcomm);
   }
   template MPI_Comm ArnoldiMatrixClass<vector_seq_float, float>::GetComm() const;
   template MPI_Comm ArnoldiMatrixClass<vector_seq_double, double>::GetComm() const;
   template MPI_Comm ArnoldiMatrixClass<vector_seq_complexs, complexs>::GetComm() const;
   template MPI_Comm ArnoldiMatrixClass<vector_seq_complexd, complexd>::GetComm() const;
   template MPI_Comm ArnoldiMatrixClass<vector_par_float, float>::GetComm() const;
   template MPI_Comm ArnoldiMatrixClass<vector_par_double, double>::GetComm() const;
   template MPI_Comm ArnoldiMatrixClass<vector_par_complexs, complexs>::GetComm() const;
   template MPI_Comm ArnoldiMatrixClass<vector_par_complexd, complexd>::GetComm() const;
   
   template <class VectorType, typename DataType>
   int ArnoldiMatrixClass<VectorType, DataType>::MatVec( char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y)
   {
      PARGEMSLR_ERROR("virtual function MatVec( char trans, const T &alpha, const VectorType &x, const T &beta, VectorType &y) not inplemented for this class.");
      return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
   }
   template int ArnoldiMatrixClass<vector_seq_float, float>::MatVec( char trans, const float &alpha, vector_seq_float &x, const float &beta, vector_seq_float &y);
   template int ArnoldiMatrixClass<vector_seq_double, double>::MatVec( char trans, const double &alpha, vector_seq_double &x, const double &beta, vector_seq_double &y);
   template int ArnoldiMatrixClass<vector_seq_complexs, complexs>::MatVec( char trans, const complexs &alpha, vector_seq_complexs &x, const complexs &beta, vector_seq_complexs &y);
   template int ArnoldiMatrixClass<vector_seq_complexd, complexd>::MatVec( char trans, const complexd &alpha, vector_seq_complexd &x, const complexd &beta, vector_seq_complexd &y);
   template int ArnoldiMatrixClass<vector_par_float, float>::MatVec( char trans, const float &alpha, vector_par_float &x, const float &beta, vector_par_float &y);
   template int ArnoldiMatrixClass<vector_par_double, double>::MatVec( char trans, const double &alpha, vector_par_double &x, const double &beta, vector_par_double &y);
   template int ArnoldiMatrixClass<vector_par_complexs, complexs>::MatVec( char trans, const complexs &alpha, vector_par_complexs &x, const complexs &beta, vector_par_complexs &y);
   template int ArnoldiMatrixClass<vector_par_complexd, complexd>::MatVec( char trans, const complexd &alpha, vector_par_complexd &x, const complexd &beta, vector_par_complexd &y);

   template <class VectorType, typename DataType>
   int ArnoldiMatrixClass<VectorType, DataType>::SetupVectorPtrStr(VectorType &v)
   {
      PARGEMSLR_ERROR("virtual function SetupVectorPtrStr(VectorType &v) not inplemented for this class.");
      return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
   }
   template int ArnoldiMatrixClass<vector_seq_float, float>::SetupVectorPtrStr( vector_seq_float &v);
   template int ArnoldiMatrixClass<vector_seq_double, double>::SetupVectorPtrStr( vector_seq_double &v);
   template int ArnoldiMatrixClass<vector_seq_complexs, complexs>::SetupVectorPtrStr( vector_seq_complexs &v);
   template int ArnoldiMatrixClass<vector_seq_complexd, complexd>::SetupVectorPtrStr( vector_seq_complexd &v);
   template int ArnoldiMatrixClass<vector_par_float, float>::SetupVectorPtrStr( vector_par_float &v);
   template int ArnoldiMatrixClass<vector_par_double, double>::SetupVectorPtrStr( vector_par_double &v);
   template int ArnoldiMatrixClass<vector_par_complexs, complexs>::SetupVectorPtrStr( vector_par_complexs &v);
   template int ArnoldiMatrixClass<vector_par_complexd, complexd>::SetupVectorPtrStr( vector_par_complexd &v);

}
