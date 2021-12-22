
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/protos.hpp"
#include "vector.hpp"
#include "sequential_vector.hpp"
#include "parallel_vector.hpp"
#include "vectorops.hpp"
#include "../utils/utils.hpp"

#include <cstring>
#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

namespace pargemslr
{

   template <typename T>
   ParallelVectorClass<T>::ParallelVectorClass()
   {
      this->_n_start = 0;
      this->_n_global = 0;
   }
   template ParallelVectorClass<float>::ParallelVectorClass();
   template ParallelVectorClass<double>::ParallelVectorClass();
   template ParallelVectorClass<complexs>::ParallelVectorClass();
   template ParallelVectorClass<complexd>::ParallelVectorClass();

   template <typename T>
   ParallelVectorClass<T>::ParallelVectorClass(const ParallelVectorClass<T> &vec) : VectorClass<T>(vec)
   {
      this->_data_vec = vec._data_vec;
      this->_n_start = vec._n_start;
      this->_n_global = vec._n_global;
   }
   template ParallelVectorClass<float>::ParallelVectorClass(const ParallelVectorClass<float> &vec);
   template ParallelVectorClass<double>::ParallelVectorClass(const ParallelVectorClass<double> &vec);
   template ParallelVectorClass<complexs>::ParallelVectorClass(const ParallelVectorClass<complexs> &vec);
   template ParallelVectorClass<complexd>::ParallelVectorClass(const ParallelVectorClass<complexd> &vec);

   template <typename T>
   ParallelVectorClass<T>::ParallelVectorClass( ParallelVectorClass<T> &&vec) : VectorClass<T>(std::move(vec))
   {
      this->_data_vec = std::move(vec._data_vec);
      this->_n_start = vec._n_start;
      vec._n_start = 0;
      this->_n_global = vec._n_global;
      vec._n_global = 0;
   }
   template ParallelVectorClass<float>::ParallelVectorClass( ParallelVectorClass<float> &&vec);
   template ParallelVectorClass<double>::ParallelVectorClass( ParallelVectorClass<double> &&vec);
   template ParallelVectorClass<complexs>::ParallelVectorClass( ParallelVectorClass<complexs> &&vec);
   template ParallelVectorClass<complexd>::ParallelVectorClass( ParallelVectorClass<complexd> &&vec);

   template <typename T>
   ParallelVectorClass<T>& ParallelVectorClass<T>::operator= (const ParallelVectorClass<T> &vec)
   {
      this->Clear();
      ParallelLogClass::operator=(vec);
      this->_data_vec = vec._data_vec;
      this->_n_start = vec._n_start;
      this->_n_global = vec._n_global;
      return *this;
   }
   template ParallelVectorClass<float>&      ParallelVectorClass<float>::operator= (const ParallelVectorClass<float> &vec);
   template ParallelVectorClass<double>&     ParallelVectorClass<double>::operator= (const ParallelVectorClass<double> &vec);
   template ParallelVectorClass<complexs>&   ParallelVectorClass<complexs>::operator= (const ParallelVectorClass<complexs> &vec);
   template ParallelVectorClass<complexd>&   ParallelVectorClass<complexd>::operator= (const ParallelVectorClass<complexd> &vec);
   
   template <typename T>
   ParallelVectorClass<T>& ParallelVectorClass<T>::operator= ( ParallelVectorClass<T> &&vec)
   {
      this->Clear();
      ParallelLogClass::operator=(std::move(vec));
      this->_data_vec = std::move(vec._data_vec);
      this->_n_start = vec._n_start;
      vec._n_start = 0;
      this->_n_global = vec._n_global;
      vec._n_global = 0;
      return *this;
   }
   template ParallelVectorClass<float>&      ParallelVectorClass<float>::operator= ( ParallelVectorClass<float> &&vec);
   template ParallelVectorClass<double>&     ParallelVectorClass<double>::operator= ( ParallelVectorClass<double> &&vec);
   template ParallelVectorClass<complexs>&   ParallelVectorClass<complexs>::operator= ( ParallelVectorClass<complexs> &&vec);
   template ParallelVectorClass<complexd>&   ParallelVectorClass<complexd>::operator= ( ParallelVectorClass<complexd> &&vec);

   template <typename T>
   T& ParallelVectorClass<T>::operator[] (int i)
   {
      return (this->_data_vec[i]);
   }
   template float&      ParallelVectorClass<float>::operator[] (int i);
   template double&     ParallelVectorClass<double>::operator[] (int i);
   template complexs&   ParallelVectorClass<complexs>::operator[] (int i);
   template complexd&   ParallelVectorClass<complexd>::operator[] (int i);

   template <typename T>
   int ParallelVectorClass<T>::Setup(int n_local, parallel_log &parlog)
   {
      return Setup( n_local, n_local, this->GetDataLocation(), false, parlog);
   }
   template int ParallelVectorClass<float>::Setup(int n_local, parallel_log &parlog);
   template int ParallelVectorClass<double>::Setup(int n_local, parallel_log &parlog);
   template int ParallelVectorClass<complexs>::Setup(int n_local, parallel_log &parlog);
   template int ParallelVectorClass<complexd>::Setup(int n_local, parallel_log &parlog);

   template <typename T>
   int ParallelVectorClass<T>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, parallel_log &parlog)
   {
      return Setup( n_local, n_start, n_global, n_local, this->GetDataLocation(), false, parlog);
   }
   template int ParallelVectorClass<float>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, parallel_log &parlog);
   template int ParallelVectorClass<double>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, parallel_log &parlog);
   template int ParallelVectorClass<complexs>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, parallel_log &parlog);
   template int ParallelVectorClass<complexd>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, parallel_log &parlog);

   template <typename T>
   int ParallelVectorClass<T>::Setup(int n_local, bool setzero, parallel_log &parlog)
   {
      return Setup( n_local, n_local, this->GetDataLocation(), setzero, parlog);
   }
   template int ParallelVectorClass<float>::Setup(int n_local, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<double>::Setup(int n_local, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexs>::Setup(int n_local, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexd>::Setup(int n_local, bool setzero, parallel_log &parlog);

   template <typename T>
   int ParallelVectorClass<T>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, bool setzero, parallel_log &parlog)
   {
      return Setup( n_local, n_start, n_global, n_local, this->GetDataLocation(), setzero, parlog);
   }
   template int ParallelVectorClass<float>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<double>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexs>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexd>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, bool setzero, parallel_log &parlog);

   template <typename T>
   int ParallelVectorClass<T>::Setup(int n_local, int location, bool setzero, parallel_log &parlog)
   {
      return Setup( n_local, n_local, location, setzero, parlog);
   }
   template int ParallelVectorClass<float>::Setup(int n_local, int location, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<double>::Setup(int n_local, int location, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexs>::Setup(int n_local, int location, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexd>::Setup(int n_local, int location, bool setzero, parallel_log &parlog);

   template <typename T>
   int ParallelVectorClass<T>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int location, bool setzero, parallel_log &parlog)
   {
      return Setup( n_local, n_start, n_global, n_local, location, setzero, parlog);
   }
   template int ParallelVectorClass<float>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int location, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<double>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int location, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexs>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int location, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexd>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int location, bool setzero, parallel_log &parlog);

   template <typename T>
   int ParallelVectorClass<T>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int reserve, int location, bool setzero, parallel_log &parlog)
   {
      PARGEMSLR_CHKERR(n_local < 0);
      PARGEMSLR_CHKERR(n_local > reserve);
      
      this->_n_start = n_start;
      this->_n_global = n_global;
      
      this->_data_vec.Setup(n_local, reserve, location, setzero);
      
      MPI_Comm comm;
      int      np, myid;
      
      parlog.GetMpiInfo(np, myid, comm);
      
      PARGEMSLR_MALLOC(this->_comm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( (MPI_Comm_dup(comm, this->_comm)) );
      this->_size = np;
      this->_rank = myid;
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int reserve, int location, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<double>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int reserve, int location, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexs>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int reserve, int location, bool setzero, parallel_log &parlog);
   template int ParallelVectorClass<complexd>::Setup(int n_local, pargemslr_long n_start, pargemslr_long n_global, int reserve, int location, bool setzero, parallel_log &parlog);

   template <typename T>
   int ParallelVectorClass<T>::SetupPtrStr( ParallelVectorClass<T> &x)
   {
      Clear();
      
      this->_n_start = x.GetStartGlobal();
      this->_n_global = x.GetLengthGlobal();
      
      this->_data_vec.SetupPtrStr(x.GetDataVector());
      
      MPI_Comm comm;
      int      np, myid;
      
      x.GetMpiInfo(np, myid, comm);
      
      this->_commref = comm;
      this->_size = np;
      this->_rank = myid;
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::SetupPtrStr( ParallelVectorClass<float> &x);
   template int ParallelVectorClass<double>::SetupPtrStr( ParallelVectorClass<double> &x);
   template int ParallelVectorClass<complexs>::SetupPtrStr( ParallelVectorClass<complexs> &x);
   template int ParallelVectorClass<complexd>::SetupPtrStr( ParallelVectorClass<complexd> &x);
   
   template <typename T>
   int ParallelVectorClass<T>::SetupPtrStr( ParallelCsrMatrixClass<T> &A)
   {
      Clear();
      
      this->_n_start = A.GetRowStartGlobal();
      this->_n_global = A.GetNumRowsGlobal();
      
      this->_data_vec.SetupPtrStr(A.GetNumRowsLocal());
      
      MPI_Comm comm;
      int      np, myid;
      
      A.GetMpiInfo(np, myid, comm);
      
      this->_commref = comm;
      this->_size = np;
      this->_rank = myid;
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::SetupPtrStr( ParallelCsrMatrixClass<float> &A);
   template int ParallelVectorClass<double>::SetupPtrStr( ParallelCsrMatrixClass<double> &A);
   template int ParallelVectorClass<complexs>::SetupPtrStr( ParallelCsrMatrixClass<complexs> &A);
   template int ParallelVectorClass<complexd>::SetupPtrStr( ParallelCsrMatrixClass<complexd> &A);
   
   template <typename T>
   int ParallelVectorClass<T>::UpdatePtr( void* data, int location)
   {
      this->_data_vec.UpdatePtr(data, location);
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::UpdatePtr( void* data, int location);
   template int ParallelVectorClass<double>::UpdatePtr( void* data, int location);
   template int ParallelVectorClass<complexs>::UpdatePtr( void* data, int location);
   template int ParallelVectorClass<complexd>::UpdatePtr( void* data, int location);
   
   template <typename T>
   int ParallelVectorClass<T>::Clear()
   {
      /* base class clear */
      VectorClass<T>::Clear();
      
      this->_data_vec.Clear();
      this->_n_start = 0;
      this->_n_global = 0;
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::Clear();
   template int ParallelVectorClass<double>::Clear();
   template int ParallelVectorClass<complexs>::Clear();
   template int ParallelVectorClass<complexd>::Clear();

   template <typename T>
   ParallelVectorClass<T>::~ParallelVectorClass()
   {
      Clear();
   }
   template ParallelVectorClass<float>::~ParallelVectorClass();
   template ParallelVectorClass<double>::~ParallelVectorClass();
   template ParallelVectorClass<complexs>::~ParallelVectorClass();
   template ParallelVectorClass<complexd>::~ParallelVectorClass();
   
   template <typename T>
   T* ParallelVectorClass<T>::GetData() const
   {
      return this->_data_vec.GetData();
   }
   template float* ParallelVectorClass<float>::GetData() const;
   template double* ParallelVectorClass<double>::GetData() const;
   template complexs* ParallelVectorClass<complexs>::GetData() const;
   template complexd* ParallelVectorClass<complexd>::GetData() const;
   
   template <typename T>
   SequentialVectorClass<T>& ParallelVectorClass<T>::GetDataVector()
   {
      return this->_data_vec;
   }
   template SequentialVectorClass<float>& ParallelVectorClass<float>::GetDataVector();
   template SequentialVectorClass<double>& ParallelVectorClass<double>::GetDataVector();
   template SequentialVectorClass<complexs>& ParallelVectorClass<complexs>::GetDataVector();
   template SequentialVectorClass<complexd>& ParallelVectorClass<complexd>::GetDataVector();
   
   template <typename T>
   int ParallelVectorClass<T>::GetDataLocation() const
   {
      return this->_data_vec.GetDataLocation();
   }
   template int ParallelVectorClass<float>::GetDataLocation() const;
   template int ParallelVectorClass<double>::GetDataLocation() const;
   template int ParallelVectorClass<complexs>::GetDataLocation() const;
   template int ParallelVectorClass<complexd>::GetDataLocation() const;
   
   template <typename T>
   int ParallelVectorClass<T>::GetLengthLocal() const
   {
      return this->_data_vec.GetLengthLocal();
   }
   template int ParallelVectorClass<float>::GetLengthLocal() const;
   template int ParallelVectorClass<double>::GetLengthLocal() const;
   template int ParallelVectorClass<complexs>::GetLengthLocal() const;
   template int ParallelVectorClass<complexd>::GetLengthLocal() const;
   
   template <typename T>
   pargemslr_long ParallelVectorClass<T>::GetLengthGlobal() const
   {
      return this->_n_global;
   }
   template pargemslr_long ParallelVectorClass<float>::GetLengthGlobal() const;
   template pargemslr_long ParallelVectorClass<double>::GetLengthGlobal() const;
   template pargemslr_long ParallelVectorClass<complexs>::GetLengthGlobal() const;
   template pargemslr_long ParallelVectorClass<complexd>::GetLengthGlobal() const;
   
   template <typename T>
   pargemslr_long ParallelVectorClass<T>::GetStartGlobal() const
   {
      return this->_n_start;
   }
   template pargemslr_long ParallelVectorClass<float>::GetStartGlobal() const;
   template pargemslr_long ParallelVectorClass<double>::GetStartGlobal() const;
   template pargemslr_long ParallelVectorClass<complexs>::GetStartGlobal() const;
   template pargemslr_long ParallelVectorClass<complexd>::GetStartGlobal() const;
   
   template <typename T>
   bool ParallelVectorClass<T>::IsHoldingData() const
   {
      return this->_data_vec.IsHoldingData();
   }
   template bool ParallelVectorClass<float>::IsHoldingData() const;
   template bool ParallelVectorClass<double>::IsHoldingData() const;
   template bool ParallelVectorClass<complexs>::IsHoldingData() const;
   template bool ParallelVectorClass<complexd>::IsHoldingData() const;
   
   template <typename T>
   int ParallelVectorClass<T>::SetHoldingData(bool hold_data)
   {
      return this->_data_vec.SetHoldingData(hold_data);
   }
   template int ParallelVectorClass<float>::SetHoldingData(bool hold_data);
   template int ParallelVectorClass<double>::SetHoldingData(bool hold_data);
   template int ParallelVectorClass<complexs>::SetHoldingData(bool hold_data);
   template int ParallelVectorClass<complexd>::SetHoldingData(bool hold_data);
   
   template <typename T>
   int ParallelVectorClass<T>::WriteToDisk( const char *datafilename)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Can only write host data.");
      }
      
      int      np,myid;
      MPI_Comm comm;
      
      this->GetMpiInfo(np, myid, comm);
      
      char tempfilename[1024];
      snprintf( tempfilename, 1024, "./%s%05d", datafilename, myid );
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<T>::value, double, float>::type RealDataType;
      typedef ComplexValueClass<RealDataType> ComplexDataType;
      
      int i, n_local;
      
      RealDataType      *rval;
      ComplexDataType   *cval;
   
      FILE *fdata;
      
      if ((fdata = fopen(tempfilename, "w")) == NULL)
      {
         printf("Can't open file.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }
      
      n_local = this->GetLengthLocal();
      if(PargemslrIsComplex<T>::value)
      {
         cval = (ComplexDataType*)this->_data_vec.GetData();
         
         for(i = 0 ; i < n_local ; i ++)
         {
            fprintf(fdata, "%16.12f %16.12f\n", cval[i].Real(), cval[i].Imag());
         }
      }
      else
      {
         rval = (RealDataType*)this->_data_vec.GetData();
         
         for(i = 0 ; i < n_local ; i ++)
         {
            fprintf(fdata, "%16.12f \n", rval[i]);
         }
      }
      
      fclose(fdata);
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::WriteToDisk( const char *datafilename);
   template int ParallelVectorClass<double>::WriteToDisk( const char *datafilename);
   template int ParallelVectorClass<complexs>::WriteToDisk( const char *datafilename);
   template int ParallelVectorClass<complexd>::WriteToDisk( const char *datafilename);
   
   template <typename T>
   int ParallelVectorClass<T>::Plot( int conditiona, int conditionb, int width)
   {
      /* TODO: No OpenMP implemntation yet, we don't have that large array to search from */
      if(this->GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Plot vector only can be used on the host, moving data to the host."<<std::endl;
         this->MoveData( kMemoryHost);
      }
      return VectorPlotHost( *this, conditiona, conditionb, width);
   }
   template int ParallelVectorClass<float>::Plot( int conditiona, int conditionb, int width);
   template int ParallelVectorClass<double>::Plot( int conditiona, int conditionb, int width);
   template int ParallelVectorClass<complexs>::Plot( int conditiona, int conditionb, int width);
   template int ParallelVectorClass<complexd>::Plot( int conditiona, int conditionb, int width);
   
   template <typename T>
   int ParallelVectorClass<T>::Fill(const T &v)
   {
      return this->_data_vec.Fill(v);;
   }
   template int ParallelVectorClass<float>::Fill(const float &v);
   template int ParallelVectorClass<double>::Fill(const double &v);
   template int ParallelVectorClass<complexs>::Fill(const complexs &v);
   template int ParallelVectorClass<complexd>::Fill(const complexd &v);

   template <typename T>
   int ParallelVectorClass<T>::Rand()
   {
      return this->_data_vec.Rand();
   }
   template int ParallelVectorClass<float>::Rand();
   template int ParallelVectorClass<double>::Rand();
   template int ParallelVectorClass<complexs>::Rand();
   template int ParallelVectorClass<complexd>::Rand();
   
   template <typename T>
   int ParallelVectorClass<T>::Scale( const T &alpha)
   {
      return this->_data_vec.Scale(alpha);
   }
   template int ParallelVectorClass<float>::Scale( const float &alpha);
   template int ParallelVectorClass<double>::Scale( const double &alpha);
   template int ParallelVectorClass<complexs>::Scale( const complexs &alpha);
   template int ParallelVectorClass<complexd>::Scale( const complexd &alpha);
   
   template <typename T>
   int ParallelVectorClass<T>::Axpy( const T &alpha, const VectorClass<T> &x)
   {
      return VectorAxpy( alpha, x, *this);
   }
   template int ParallelVectorClass<float>::Axpy( const float &alpha, const VectorClass<float> &x);
   template int ParallelVectorClass<double>::Axpy( const double &alpha, const VectorClass<double> &x);
   template int ParallelVectorClass<complexs>::Axpy( const complexs &alpha, const VectorClass<complexs> &x);
   template int ParallelVectorClass<complexd>::Axpy( const complexd &alpha, const VectorClass<complexd> &x);
   
   template <typename T>
   int ParallelVectorClass<T>::Axpy( const T &alpha, const VectorClass<T> &x, VectorClass<T> &y)
   {
      T one = T(1.0);
      T zero = T(0.0);
      this->Fill(zero);
      VectorAxpy( alpha, x, *this);
      return VectorAxpy( one, y, *this);
   }
   template int ParallelVectorClass<float>::Axpy( const float &alpha, const VectorClass<float> &x, VectorClass<float> &y);
   template int ParallelVectorClass<double>::Axpy( const double &alpha, const VectorClass<double> &x, VectorClass<double> &y);
   template int ParallelVectorClass<complexs>::Axpy( const complexs &alpha, const VectorClass<complexs> &x, VectorClass<complexs> &y);
   template int ParallelVectorClass<complexd>::Axpy( const complexd &alpha, const VectorClass<complexd> &x, VectorClass<complexd> &y);
   
   template <typename T>
   int ParallelVectorClass<T>::Axpy( const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y)
   {
      T zero = T(0.0);
      this->Fill(zero);
      VectorAxpy( alpha, x, *this);
      return VectorAxpy( beta, y, *this);
   }
   template int ParallelVectorClass<float>::Axpy( const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   template int ParallelVectorClass<double>::Axpy( const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   template int ParallelVectorClass<complexs>::Axpy( const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   template int ParallelVectorClass<complexd>::Axpy( const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
   template <typename T>
   int ParallelVectorClass<T>::Norm2( float &norm) const
   {
      T t;
      VectorDot( *this, *this, t);
      norm = sqrt(PargemslrAbs(t));
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::Norm2( float &norm) const;
   template int ParallelVectorClass<complexs>::Norm2( float &norm) const;
   template int ParallelVectorClass<double>::Norm2( float &norm) const;
   template int ParallelVectorClass<complexd>::Norm2( float &norm) const;
   
   template <typename T>
   int ParallelVectorClass<T>::Norm2( double &norm) const
   {
      T t;
      VectorDot( *this, *this, t);
      norm = sqrt(PargemslrAbs(t));
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::Norm2( double &norm) const;
   template int ParallelVectorClass<complexs>::Norm2( double &norm) const;
   template int ParallelVectorClass<double>::Norm2( double &norm) const;
   template int ParallelVectorClass<complexd>::Norm2( double &norm) const;
   
   template <typename T>
   int ParallelVectorClass<T>::NormInf( float &norm)
   {
      
      float local_norm;
      this->_data_vec.NormInf(local_norm);
      
      PargemslrMpiAllreduce( &local_norm, &norm, 1, MPI_MAX, *(this->_comm));
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::NormInf( float &norm);
   template int ParallelVectorClass<complexs>::NormInf( float &norm);
   template int ParallelVectorClass<double>::NormInf( float &norm);
   template int ParallelVectorClass<complexd>::NormInf( float &norm);
   
   template <typename T>
   int ParallelVectorClass<T>::NormInf( double &norm)
   {
      
      double local_norm;
      this->_data_vec.NormInf(local_norm);
      
      PargemslrMpiAllreduce( &local_norm, &norm, 1, MPI_MAX, *(this->_comm));
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<float>::NormInf( double &norm);
   template int ParallelVectorClass<complexs>::NormInf( double &norm);
   template int ParallelVectorClass<double>::NormInf( double &norm);
   template int ParallelVectorClass<complexd>::NormInf( double &norm);
   
   template <typename T>
   int ParallelVectorClass<T>::Dot( const VectorClass<T> &y, T &t) const
   {
      PARGEMSLR_CHKERR( this->GetLengthLocal() != y.GetLengthLocal() || this->GetLengthGlobal() != y.GetLengthGlobal() );
      
      VectorDot( *this, y, t);
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelVectorClass<double>::Dot( const VectorClass<double> &y, double &t) const;
   template int ParallelVectorClass<complexd>::Dot( const VectorClass<complexd> &y, complexd &t) const;
   template int ParallelVectorClass<float>::Dot( const VectorClass<float> &y, float &t) const;
   template int ParallelVectorClass<complexs>::Dot( const VectorClass<complexs> &y, complexs &t) const;
   
   template <typename T>
   bool ParallelVectorClass<T>::IsParallel() const
   {
      return true;
   }
   template bool ParallelVectorClass<float>::IsParallel() const;
   template bool ParallelVectorClass<double>::IsParallel() const;
   template bool ParallelVectorClass<complexs>::IsParallel() const;
   template bool ParallelVectorClass<complexd>::IsParallel() const;
   
   template <typename T>
   int ParallelVectorClass<T>::MoveData( const int &location)
   {
      return this->_data_vec.MoveData(location);
   }
   template int ParallelVectorClass<float>::MoveData( const int &location);
   template int ParallelVectorClass<double>::MoveData( const int &location);
   template int ParallelVectorClass<complexs>::MoveData( const int &location);
   template int ParallelVectorClass<complexd>::MoveData( const int &location);

#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
   
   template <typename T>
   cusparseDnVecDescr_t ParallelVectorClass<T>::GetCusparseVec() const
   {
      return this->_data_vec.GetCusparseVec();
   }
   template cusparseDnVecDescr_t ParallelVectorClass<float>::GetCusparseVec() const;
   template cusparseDnVecDescr_t ParallelVectorClass<double>::GetCusparseVec() const;
   template cusparseDnVecDescr_t ParallelVectorClass<complexs>::GetCusparseVec() const;
   template cusparseDnVecDescr_t ParallelVectorClass<complexd>::GetCusparseVec() const;
   
   template <typename T>
   int ParallelVectorClass<T>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec)
   {
      return this->_data_vec.SetCusparseVec(cusparse_vec);
   }
   template int ParallelVectorClass<float>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
   template int ParallelVectorClass<double>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
   template int ParallelVectorClass<complexs>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
   template int ParallelVectorClass<complexd>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
#endif
#endif
   
   template <typename T>
   int ParallelVectorClass<T>::ReadFromSingleMMFile(const char *vecfile, int idxin)
   {
      /* TODO: this is a sequential read and not support long int */
      
      int                     err = 0, i;
      
      /* overwrite any current data */
      this->_data_vec.Setup(this->GetLengthLocal(), kMemoryHost, false);
      
      MPI_Comm comm;
      int np, myid;
      this->GetMpiInfo(np, myid, comm);
      
      SequentialVectorClass<T>         global_vec;
      IntVectorClass<pargemslr_long>   nstarts;
      
      /* only one MPI rank, do as sequential */
      if(np == 1)
      {
         err = global_vec.ReadFromMMFile( vecfile, idxin); PARGEMSLR_CHKERR(err);
         
         PARGEMSLR_CHKERR( this->_n_global != (pargemslr_long)(global_vec.GetLengthLocal()));
         
         this->_data_vec = std::move(global_vec);
         
         return PARGEMSLR_SUCCESS;
      }
      
      if(myid == 0)
      {
         /* read the global vector */
         err = global_vec.ReadFromMMFile( vecfile, idxin); PARGEMSLR_CHKERR(err);
         PARGEMSLR_CHKERR( this->_n_global != (pargemslr_long)(global_vec.GetLengthLocal()));
         nstarts.Setup(np+1);
      }
      
      PARGEMSLR_MPI_CALL( PargemslrMpiGather( &(this->_n_start), 1, nstarts.GetData(), 0, comm) );
      
      /* now start sending data */
      if(myid == 0)
      {
         nstarts[np] = this->_n_global;
         for(i = 1 ; i < np ; i ++)
         {
            PARGEMSLR_MPI_CALL(PargemslrMpiSend( global_vec.GetData()+nstarts[i], nstarts[i+1]-nstarts[i], i, 0, comm));
         }
         PARGEMSLR_MEMCPY( this->_data_vec.GetData(), global_vec.GetData(), this->GetLengthLocal(), kMemoryHost, kMemoryHost, T);
      }
      else
      {
         PARGEMSLR_MPI_CALL(PargemslrMpiRecv( this->_data_vec.GetData(), this->GetLengthLocal(), 0, 0, comm, MPI_STATUS_IGNORE));
      }
      
      return err;
   }
   template int ParallelVectorClass<float>::ReadFromSingleMMFile(const char *vecfile, int idxin);
   template int ParallelVectorClass<double>::ReadFromSingleMMFile(const char *vecfile, int idxin);
   template int ParallelVectorClass<complexs>::ReadFromSingleMMFile(const char *vecfile, int idxin);
   template int ParallelVectorClass<complexd>::ReadFromSingleMMFile(const char *vecfile, int idxin);

   
}
