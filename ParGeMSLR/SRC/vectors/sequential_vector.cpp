
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "vector.hpp"
#include "sequential_vector.hpp"
#include "vectorops.hpp"
#include "../utils/utils.hpp"
#include "../matrices/csr_matrix.hpp"

#include <cstring>
#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

namespace pargemslr
{

   template <typename T>
   SequentialVectorClass<T>::SequentialVectorClass()
   {
#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
      this->_cusparse_vec = NULL;
#endif
#endif
      this->_data = NULL;
      this->_length = 0;
      this->_maxlength = 0;
      this->_hold_data = true;
      this->_location = kMemoryHost;
   }
   template SequentialVectorClass<float>::SequentialVectorClass();
   template SequentialVectorClass<double>::SequentialVectorClass();
   template SequentialVectorClass<complexs>::SequentialVectorClass();
   template SequentialVectorClass<complexd>::SequentialVectorClass();
   
   template <typename T>
   SequentialVectorClass<T>::SequentialVectorClass(const SequentialVectorClass<T> &vec) : VectorClass<T>(vec)
   {
      if(vec._hold_data)
      {
         PARGEMSLR_MALLOC( this->_data, vec._maxlength, vec._location, T);
         PARGEMSLR_MEMCPY( this->_data, vec._data, vec._length, vec._location, vec._location, T);
         this->_hold_data = true;
      }
      else
      {
         this->_data = vec._data;
         this->_hold_data = false;
      }
      
      this->_location = vec._location;
      this->_length = vec._length;
      this->_maxlength = vec._maxlength;
#ifdef PARGEMSLR_CUDA
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         SequentialVectorCreateCusparseDnVec(*this);
      }
#endif
   }
   template SequentialVectorClass<float>::SequentialVectorClass(const SequentialVectorClass<float> &vec);
   template SequentialVectorClass<double>::SequentialVectorClass(const SequentialVectorClass<double> &vec);
   template SequentialVectorClass<complexs>::SequentialVectorClass(const SequentialVectorClass<complexs> &vec);
   template SequentialVectorClass<complexd>::SequentialVectorClass(const SequentialVectorClass<complexd> &vec);
   
   template <typename T>
   SequentialVectorClass<T>::SequentialVectorClass(SequentialVectorClass<T>&& vec) : VectorClass<T>(std::move(vec))
   {
      this->_data = vec._data;
      vec._data = NULL;
      
      this->_hold_data = vec._hold_data;
      vec._hold_data = true;
      
      this->_location = vec._location;
      vec._location = kMemoryHost;
      
      this->_length = vec._length;
      vec._length = 0;
      
      this->_maxlength = vec._maxlength;
      vec._maxlength = 0;
      
#ifdef PARGEMSLR_CUDA
#if (PARGEMSLR_CUDA_VERSION == 11)
      this->_cusparse_vec = vec._cusparse_vec;
      vec._cusparse_vec = NULL;
#endif
#endif
   }
   template SequentialVectorClass<float>::SequentialVectorClass(SequentialVectorClass<float>&& vec);
   template SequentialVectorClass<double>::SequentialVectorClass(SequentialVectorClass<double>&& vec);
   template SequentialVectorClass<complexs>::SequentialVectorClass(SequentialVectorClass<complexs>&& vec);
   template SequentialVectorClass<complexd>::SequentialVectorClass(SequentialVectorClass<complexd>&& vec);
   
   template <typename T>
   SequentialVectorClass<T>& SequentialVectorClass<T>::operator= (const SequentialVectorClass<T> &vec)
   {
      this->Clear();
      ParallelLogClass::operator=(vec);
      if(vec._hold_data)
      {
         PARGEMSLR_MALLOC( this->_data, vec._maxlength, vec._location, T);
         PARGEMSLR_MEMCPY( this->_data, vec._data, vec._length, vec._location, vec._location, T);
         this->_hold_data = true;
      }
      else
      {
         this->_data = vec._data;
         this->_hold_data = false;
      }
      
      this->_location = vec._location;
      this->_length = vec._length;
      this->_maxlength = vec._maxlength;
#ifdef PARGEMSLR_CUDA
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         SequentialVectorCreateCusparseDnVec(*this);
      }
#endif
      return *this;
   }
   template SequentialVectorClass<float>& SequentialVectorClass<float>::operator= (const SequentialVectorClass<float> &vec);
   template SequentialVectorClass<double>& SequentialVectorClass<double>::operator= (const SequentialVectorClass<double> &vec);
   template SequentialVectorClass<complexs>& SequentialVectorClass<complexs>::operator= (const SequentialVectorClass<complexs> &vec);
   template SequentialVectorClass<complexd>& SequentialVectorClass<complexd>::operator= (const SequentialVectorClass<complexd> &vec);
   
   template <typename T>
   SequentialVectorClass<T>& SequentialVectorClass<T>::operator= ( SequentialVectorClass<T> &&vec)
   {
      this->Clear();
      ParallelLogClass::operator=(std::move(vec));
      this->_data = vec._data;
      vec._data = NULL;
      
      this->_hold_data = vec._hold_data;
      vec._hold_data = true;
      
      this->_location = vec._location;
      vec._location = kMemoryHost;
      
      this->_length = vec._length;
      vec._length = 0;
      
      this->_maxlength = vec._maxlength;
      vec._maxlength = 0;
      
#ifdef PARGEMSLR_CUDA
#if (PARGEMSLR_CUDA_VERSION == 11)
      this->_cusparse_vec = vec._cusparse_vec;
      vec._cusparse_vec = NULL;
#endif
#endif
      return *this;
   }
   template SequentialVectorClass<float>& SequentialVectorClass<float>::operator= ( SequentialVectorClass<float> &&vec);
   template SequentialVectorClass<double>& SequentialVectorClass<double>::operator= ( SequentialVectorClass<double> &&vec);
   template SequentialVectorClass<complexs>& SequentialVectorClass<complexs>::operator= ( SequentialVectorClass<complexs> &&vec);
   template SequentialVectorClass<complexd>& SequentialVectorClass<complexd>::operator= ( SequentialVectorClass<complexd> &&vec);
   
   template <typename T>
   T& SequentialVectorClass<T>::operator[] (int i)
   {
      PARGEMSLR_CHKERR(i < 0);
      PARGEMSLR_CHKERR(i >= this->_length);
      return (this->_data[i]);
   }
   template float&      SequentialVectorClass<float>::operator[] (int i);
   template double&     SequentialVectorClass<double>::operator[] (int i);
   template complexs&   SequentialVectorClass<complexs>::operator[] (int i);
   template complexd&   SequentialVectorClass<complexd>::operator[] (int i);

   template <typename T>
   int SequentialVectorClass<T>::Setup(int length)
   {
      return Setup( length, length, this->_location, false);
   }
   template int SequentialVectorClass<float>::Setup(int length);
   template int SequentialVectorClass<double>::Setup(int length);
   template int SequentialVectorClass<complexs>::Setup(int length);
   template int SequentialVectorClass<complexd>::Setup(int length);

   template <typename T>
   int SequentialVectorClass<T>::Setup(int length, bool setzero)
   {
      return Setup( length, length, this->_location, setzero);
   }
   template int SequentialVectorClass<float>::Setup(int length, bool setzero);
   template int SequentialVectorClass<double>::Setup(int length, bool setzero);
   template int SequentialVectorClass<complexs>::Setup(int length, bool setzero);
   template int SequentialVectorClass<complexd>::Setup(int length, bool setzero);

   template <typename T>
   int SequentialVectorClass<T>::Setup(int length, int location, bool setzero)
   {
      return Setup( length, length, location, setzero);
   }
   template int SequentialVectorClass<float>::Setup(int length, int location, bool setzero);
   template int SequentialVectorClass<double>::Setup(int length, int location, bool setzero);
   template int SequentialVectorClass<complexs>::Setup(int length, int location, bool setzero);
   template int SequentialVectorClass<complexd>::Setup(int length, int location, bool setzero);

   template <typename T>
   int SequentialVectorClass<T>::Setup(int length, int reserve, int location, bool setzero)
   {
      PARGEMSLR_CHKERR(length < 0);
      PARGEMSLR_CHKERR(length > reserve);

      /* first check if we can use current memory */
      if(location == this->_location && this->_hold_data && this->_maxlength >= reserve)
      {
         /* we have enough space, don't need to reset */
         if(setzero)
         {
            /* Fill the vector with 0 when necessary */
            this->_length = this->_maxlength;
            Fill(0);
         }
         this->_length = length;
      }
      else
      {
         /* in this case, we need new memory location */
         Clear();
         if(reserve == 0)
         {
            this->_data = NULL;
         }
         else
         {
            if(setzero)
            {
               PARGEMSLR_CALLOC( this->_data, reserve, location, T);
            }
            else
            {
               PARGEMSLR_MALLOC( this->_data, reserve, location, T);
            }
        }
        this->_hold_data = true;
        this->_location = location;
        this->_length = length;
        this->_maxlength = reserve;
      }
#ifdef PARGEMSLR_CUDA
      if( location == kMemoryDevice || location == kMemoryUnified)
      {
         SequentialVectorCreateCusparseDnVec(*this);
      }
#endif
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Setup(int length, int reserve, int location, bool setzero);
   template int SequentialVectorClass<double>::Setup(int length, int reserve, int location, bool setzero);
   template int SequentialVectorClass<complexs>::Setup(int length, int reserve, int location, bool setzero);
   template int SequentialVectorClass<complexd>::Setup(int length, int reserve, int location, bool setzero);
   
   template <typename T>
   int SequentialVectorClass<T>::SetupPtrStr( SequentialVectorClass<T> &x)
   {
      /* first of all free current vector */
      Clear();
      this->_hold_data = false;
      this->_length = x.GetLengthLocal();
      this->_maxlength = this->_length;// this value is not referenced
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::SetupPtrStr( SequentialVectorClass<float> &x);
   template int SequentialVectorClass<double>::SetupPtrStr( SequentialVectorClass<double> &x);
   template int SequentialVectorClass<complexs>::SetupPtrStr( SequentialVectorClass<complexs> &x);
   template int SequentialVectorClass<complexd>::SetupPtrStr( SequentialVectorClass<complexd> &x);
   
   template <typename T>
   int SequentialVectorClass<T>::SetupPtrStr( CsrMatrixClass<T> &A)
   {
      /* first of all free current vector */
      Clear();
      this->_hold_data = false;
      this->_length = A.GetNumRowsLocal();
      this->_maxlength = this->_length;// this value is not referenced
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::SetupPtrStr( CsrMatrixClass<float> &A);
   template int SequentialVectorClass<double>::SetupPtrStr( CsrMatrixClass<double> &A);
   template int SequentialVectorClass<complexs>::SetupPtrStr( CsrMatrixClass<complexs> &A);
   template int SequentialVectorClass<complexd>::SetupPtrStr( CsrMatrixClass<complexd> &A);
   
   template <typename T>
   int SequentialVectorClass<T>::SetupPtrStr( int length)
   {
      /* first of all free current vector */
      Clear();
      this->_hold_data = false;
      this->_length = length;
      this->_maxlength = this->_length;// this value is not referenced
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::SetupPtrStr( int length);
   template int SequentialVectorClass<double>::SetupPtrStr( int length);
   template int SequentialVectorClass<complexs>::SetupPtrStr( int length);
   template int SequentialVectorClass<complexd>::SetupPtrStr( int length);
   
   template <typename T>
   int SequentialVectorClass<T>::UpdatePtr( void* data, int location)
   {
      this->_data = (T*)data;
      this->_location = location;
      
#ifdef PARGEMSLR_CUDA  
      if( location == kMemoryDevice || location == kMemoryUnified)
      {
         SequentialVectorCreateCusparseDnVec(*this);
      }
#endif
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::UpdatePtr( void* data, int location);
   template int SequentialVectorClass<double>::UpdatePtr( void* data, int location);
   template int SequentialVectorClass<complexs>::UpdatePtr( void* data, int location);
   template int SequentialVectorClass<complexd>::UpdatePtr( void* data, int location);
   
   template <typename T>
   int SequentialVectorClass<T>::SetupPtr( void* data, int length, int location)
   {
      return this->SetupPtr(data, length, location, false);
   }
   template int SequentialVectorClass<float>::SetupPtr( void* data, int length, int location);
   template int SequentialVectorClass<double>::SetupPtr( void* data, int length, int location);
   template int SequentialVectorClass<complexs>::SetupPtr( void* data, int length, int location);
   template int SequentialVectorClass<complexd>::SetupPtr( void* data, int length, int location);
   
   template <typename T>
   int SequentialVectorClass<T>::SetupPtr( void* data, int length, int location, bool hold_data)
   {
      
      PARGEMSLR_CHKERR(length < 0 );
      
      /* first of all free current vector */
      Clear();
      this->_data = (T*)data;
      this->_hold_data = hold_data;
      this->_location = location;
      this->_length = length;
      this->_maxlength = length;// this value is not referenced
#ifdef PARGEMSLR_CUDA  
      if( location == kMemoryDevice || location == kMemoryUnified)
      {
         SequentialVectorCreateCusparseDnVec(*this);
      }
#endif
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::SetupPtr( void *data, int length, int location, bool hold_data);
   template int SequentialVectorClass<double>::SetupPtr( void *data, int length, int location, bool hold_data);
   template int SequentialVectorClass<complexs>::SetupPtr( void *data, int length, int location, bool hold_data);
   template int SequentialVectorClass<complexd>::SetupPtr( void *data, int length, int location, bool hold_data);
   
   template <typename T>
   int SequentialVectorClass<T>::SetupPtr(const VectorClass<T> &vec, int length, int shift)
   {
      PARGEMSLR_CHKERR( shift < 0);
      PARGEMSLR_CHKERR( vec.GetLengthLocal() < length + shift );
      
      return SetupPtr( vec.GetData() + shift, length, vec.GetDataLocation(), false);
   }
   template int SequentialVectorClass<float>::SetupPtr(const vector_base_float &vec, int length, int shift);
   template int SequentialVectorClass<double>::SetupPtr(const vector_base_double &vec, int length, int shift);
   template int SequentialVectorClass<complexs>::SetupPtr(const vector_base_complexs &vec, int length, int shift);
   template int SequentialVectorClass<complexd>::SetupPtr(const vector_base_complexd &vec, int length, int shift);

   
   template <typename T>
   int SequentialVectorClass<T>::SetupPtr(const VectorClass<T> &vec, int length, int shift, bool hold_data)
   {
      PARGEMSLR_CHKERR( shift < 0);
      PARGEMSLR_CHKERR( vec.GetLengthLocal() < length + shift );
      
      return SetupPtr( vec.GetData() + shift, length, vec.GetDataLocation(), hold_data);
   }
   template int SequentialVectorClass<float>::SetupPtr(const vector_base_float &vec, int length, int shift, bool hold_data);
   template int SequentialVectorClass<double>::SetupPtr(const vector_base_double &vec, int length, int shift, bool hold_data);
   template int SequentialVectorClass<complexs>::SetupPtr(const vector_base_complexs &vec, int length, int shift, bool hold_data);
   template int SequentialVectorClass<complexd>::SetupPtr(const vector_base_complexd &vec, int length, int shift, bool hold_data);

   /**
    * @brief   Free the current vector, allocate memory, and copy data to initilize the vector.
    * @details Free the current vector, allocate memory, and copy data to initilize the vector.
    * @param   [in]        data The target memory address.
    * @param   [in]        length The length of the vector.
    * @param   [in]        loc_from The location of the input data.
    * @param   [in]        loc_to The location of the new vector.
    * @return     Return error messgae.
    */
   template <typename T>
   int SequentialVectorClass<T>::Copy(const T *data, int length, int loc_from, int loc_to)
   {

      PARGEMSLR_CHKERR(length < 0 );

      /* first of all free current vector */
      Clear();
      PARGEMSLR_MALLOC( this->_data, length, loc_to, T);
      PARGEMSLR_MEMCPY( this->_data, data, length, loc_to, loc_from, T);
      this->_hold_data = true;
      this->_location = loc_to;
      this->_length = length;
      this->_maxlength = length;// this value is not referenced
#ifdef PARGEMSLR_CUDA
      if( loc_to == kMemoryDevice || loc_to == kMemoryUnified)
      {
         SequentialVectorCreateCusparseDnVec(*this);
      }
#endif
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Copy(const float *data, int length, int loc_from, int loc_to);
   template int SequentialVectorClass<double>::Copy(const double *data, int length, int loc_from, int loc_to);
   template int SequentialVectorClass<complexs>::Copy(const complexs *data, int length, int loc_from, int loc_to);
   template int SequentialVectorClass<complexd>::Copy(const complexd *data, int length, int loc_from, int loc_to);

   /**
    * @brief   Free the current vector.
    * @details Free the current vector.
    * @return     Return error messgae.
    */
   template <typename T>
   int SequentialVectorClass<T>::Clear()
   {
      /* base class clear */
      VectorClass<T>::Clear();
      
      if( this->_hold_data && this->_maxlength > 0)
      {
         PARGEMSLR_FREE( this->_data, this->_location);
      }
      else
      {
         this->_data = NULL;
      }
      this->_length = 0;
      this->_maxlength = 0;
      this->_location = kMemoryHost;
#ifdef PARGEMSLR_CUDA
#if (PARGEMSLR_CUDA_VERSION == 11)
      if( this->_cusparse_vec )
      {
         /* free the current one */
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroyDnVec( this->_cusparse_vec )) );
		  
         this->_cusparse_vec = NULL;
      }
#endif
#endif
      
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Clear();
   template int SequentialVectorClass<double>::Clear();
   template int SequentialVectorClass<complexs>::Clear();
   template int SequentialVectorClass<complexd>::Clear();

   /**
    * @brief   Insert value at the end of the vector, expand the vector when necessary. Work only when the vector is holding data.
    * @details Insert value at the end of the vector, expand the vector when necessary. Work only when the vector is holding data.
    * @param   [in]   v The value to be inserted.
    * @return     Return error messgae.
    */
   template <typename T>
   int SequentialVectorClass<T>::PushBack(T v)
   {
      
      PARGEMSLR_CHKERR(!_hold_data);

      while( this->_length >= this->_maxlength)
      {
         Resize( _length, PargemslrMax( this->_length+1, (int)( this->_length * pargemslr::pargemslr_global::_expand_fact)), true, false);
      }
      this->_data[ (this->_length)++] = v;
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::PushBack(float v);
   template int SequentialVectorClass<double>::PushBack(double v);
   template int SequentialVectorClass<complexs>::PushBack(complexs v);
   template int SequentialVectorClass<complexd>::PushBack(complexd v);

   /**
    * @brief   Resize the vector. Re-allocate memory when necessary.
    * @details Resize the vector. Re-allocate memory when necessary.
    * @param   [in]   length The length of the vector.
    * @param   [in]   keepdata Do we keep the data in the current vector.
    * @param   [in]   setzero Do we Fill the new extra memory with 0 if we expand the vector.
    * @return     Return error messgae.
    */
   template <typename T>
   int SequentialVectorClass<T>::Resize(int length, bool keepdata, bool setzero)
   {
      return Resize(length, length, keepdata, setzero);
   }
   template int SequentialVectorClass<float>::Resize(int length, bool keepdata, bool setzero);
   template int SequentialVectorClass<double>::Resize(int length, bool keepdata, bool setzero);
   template int SequentialVectorClass<complexs>::Resize(int length, bool keepdata, bool setzero);
   template int SequentialVectorClass<complexd>::Resize(int length, bool keepdata, bool setzero);

   /**
    * @brief   Resize the vector. Re-allocate memory when necessary.
    * @details Resize the vector. Re-allocate memory when necessary.
    * @param   [in]   length The length of the vector.
    * @param   [in]   reserve The length allocated in the memory, should be no less than length.
    * @param   [in]   keepdata Do we keep the data in the current vector.
    * @param   [in]   setzero Do we Fill the new extra memory with 0 if we expand the vector.
    * @return     Return error messgae.
    */
   template <typename T>
   int SequentialVectorClass<T>::Resize(int length, int reserve, bool keepdata, bool setzero)
   {

      PARGEMSLR_CHKERR(length < 0)
      PARGEMSLR_CHKERR(length > reserve);

      SequentialVectorClass<T>  temp;
      T*                temp_data;
      if(_hold_data)
      {
         /* if we don't have enough memory, need to allocate new memory */
         if(reserve > _maxlength)
         {
            /* we are epxanding the memory */
            if(setzero)
            {
               PARGEMSLR_CALLOC(temp_data, reserve, this->_location, T);
            }
            else
            {
               PARGEMSLR_MALLOC(temp_data, reserve, this->_location, T);
            }
            if(keepdata)
            {
               PARGEMSLR_MEMCPY( temp_data, this->_data, this->_length, this->_location, this->_location, T);
            }
            PARGEMSLR_FREE( this->_data, this->_location);
            this->_data = temp_data;
            this->_length = length;
            this->_maxlength = reserve;
         }
         else
         {
            /* we have enouth memory */
            if(setzero)
            {
               if(length > _length)
               {
                  /* expanding */
                  temp.SetupPtr( *this, this->_maxlength - this->_length, this->_length);
                  temp.Fill(0);
               }
               else
               {
                  temp.SetupPtr( *this, this->_maxlength - length, length);
                  temp.Fill(0);
               }
            }
            _length = length;
         }
      }
      else
      {
         /* in this case just change the length */
         _length = length;
      }
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Resize(int length, int reserve, bool keepdata, bool setzero);
   template int SequentialVectorClass<double>::Resize(int length, int reserve, bool keepdata, bool setzero);
   template int SequentialVectorClass<complexs>::Resize(int length, int reserve, bool keepdata, bool setzero);
   template int SequentialVectorClass<complexd>::Resize(int length, int reserve, bool keepdata, bool setzero);

   /**
    * @brief   The desstructor of seq_vector.
    * @details The desnstructor of seq_vector. Just call the free function.
    */
   template <typename T>
   SequentialVectorClass<T>::~SequentialVectorClass()
   {
      Clear();
   }
   template SequentialVectorClass<float>::~SequentialVectorClass();
   template SequentialVectorClass<double>::~SequentialVectorClass();
   template SequentialVectorClass<complexs>::~SequentialVectorClass();
   template SequentialVectorClass<complexd>::~SequentialVectorClass();
   
   template <typename T>
   T* SequentialVectorClass<T>::GetData() const
   {
      return this->_data;
   }
   template float* SequentialVectorClass<float>::GetData() const;
   template double* SequentialVectorClass<double>::GetData() const;
   template complexs* SequentialVectorClass<complexs>::GetData() const;
   template complexd* SequentialVectorClass<complexd>::GetData() const;
   
#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
   /**
    * @brief   For cusparse general spmv.
    * @details For cusparse general spmv.
    */
   template <typename T>
   cusparseDnVecDescr_t SequentialVectorClass<T>::GetCusparseVec() const
   {
      return this->_cusparse_vec;
   }
   template cusparseDnVecDescr_t SequentialVectorClass<float>::GetCusparseVec() const;
   template cusparseDnVecDescr_t SequentialVectorClass<double>::GetCusparseVec() const;
   template cusparseDnVecDescr_t SequentialVectorClass<complexs>::GetCusparseVec() const;
   template cusparseDnVecDescr_t SequentialVectorClass<complexd>::GetCusparseVec() const;
   
   template <typename T>
   int SequentialVectorClass<T>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec)
   {
      if( this->_cusparse_vec )
      {
         /* free the current one */
         PARGEMSLR_CUSPARSE_CALL( (cusparseDestroyDnVec( this->_cusparse_vec )) );
      }
      this->_cusparse_vec = cusparse_vec;
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
   template int SequentialVectorClass<double>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
   template int SequentialVectorClass<complexs>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
   template int SequentialVectorClass<complexd>::SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
#endif
#endif
   
   template <typename T>
   int SequentialVectorClass<T>::GetDataLocation() const
   {
#ifndef PARGEMSLR_CUDA
      return kMemoryHost;
#else
      return this->_location;
#endif
   }
   template int SequentialVectorClass<float>::GetDataLocation() const;
   template int SequentialVectorClass<double>::GetDataLocation() const;
   template int SequentialVectorClass<complexs>::GetDataLocation() const;
   template int SequentialVectorClass<complexd>::GetDataLocation() const;
   
   template <typename T>
   int SequentialVectorClass<T>::GetLengthLocal() const
   {
      return this->_length;
   }
   template int SequentialVectorClass<float>::GetLengthLocal() const;
   template int SequentialVectorClass<double>::GetLengthLocal() const;
   template int SequentialVectorClass<complexs>::GetLengthLocal() const;
   template int SequentialVectorClass<complexd>::GetLengthLocal() const;
   
   template <typename T>
   long int SequentialVectorClass<T>::GetLengthGlobal() const
   {
      return (int long)(this->_length);
   }
   template long int SequentialVectorClass<float>::GetLengthGlobal() const;
   template long int SequentialVectorClass<double>::GetLengthGlobal() const;
   template long int SequentialVectorClass<complexs>::GetLengthGlobal() const;
   template long int SequentialVectorClass<complexd>::GetLengthGlobal() const;
   
   template <typename T>
   long int SequentialVectorClass<T>::GetStartGlobal() const
   {
      return 0;
   }
   template long int SequentialVectorClass<float>::GetStartGlobal() const;
   template long int SequentialVectorClass<double>::GetStartGlobal() const;
   template long int SequentialVectorClass<complexs>::GetStartGlobal() const;
   template long int SequentialVectorClass<complexd>::GetStartGlobal() const;
   
   template <typename T>
   bool SequentialVectorClass<T>::IsHoldingData() const
   {
      return this->_hold_data;
   }
   template bool SequentialVectorClass<float>::IsHoldingData() const;
   template bool SequentialVectorClass<double>::IsHoldingData() const;
   template bool SequentialVectorClass<complexs>::IsHoldingData() const;
   template bool SequentialVectorClass<complexd>::IsHoldingData() const;
   
   template <typename T>
   int SequentialVectorClass<T>::SetHoldingData(bool hold_data)
   {
      this->_hold_data = hold_data;
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::SetHoldingData(bool hold_data);
   template int SequentialVectorClass<double>::SetHoldingData(bool hold_data);
   template int SequentialVectorClass<complexs>::SetHoldingData(bool hold_data);
   template int SequentialVectorClass<complexd>::SetHoldingData(bool hold_data);
   
#ifndef PARGEMSLR_CUDA 
   template <typename T>
   int SequentialVectorClass<T>::Fill(const T &v)
   {
      int i;
#ifdef PARGEMSLR_OPENMP
      /* avoid nested OpenMP call */
      int num_threads = PargemslrGetOpenmpMaxNumThreads();
      if(num_threads > 1)
      {
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
         for(i = 0 ; i < this->_length ; i ++)
         {
            this->_data[i] = v;
         }
      }
      else
      {
#endif
         for(i = 0 ; i < this->_length ; i ++)
         {
            this->_data[i] = v;
         }
#ifdef PARGEMSLR_OPENMP
      }
#endif

      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Fill(const float &v);
   template int SequentialVectorClass<double>::Fill(const double &v);
   template int SequentialVectorClass<complexs>::Fill(const complexs &v);
   template int SequentialVectorClass<complexd>::Fill(const complexd &v);

   template <typename T>
   int SequentialVectorClass<T>::Rand()
   {
      int i;
      for(i = 0 ; i < this->_length ; i ++)
      {
         pargemslr::PargemslrValueRandHost(this->_data[i]);
      }

      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Rand();
   template int SequentialVectorClass<double>::Rand();
   template int SequentialVectorClass<complexs>::Rand();
   template int SequentialVectorClass<complexd>::Rand();
   
   template <typename T>
   int SequentialVectorClass<T>::Sort(bool ascending)
   {
      if(ascending)
      {
         std::sort( this->_data, this->_data + this->_length, std::less<T>());
      }
      else
      {
         std::sort( this->_data, this->_data + this->_length, std::greater<T>());
      }

      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Sort(bool ascending);
   template int SequentialVectorClass<double>::Sort(bool ascending);
   
   template <typename T>
   int SequentialVectorClass<T>::Sort(IntVectorClass<int> &order, bool ascending, bool stable)
   {
      std::vector<CompareStruct<T> > compare;
      
      int i;
      int n = order.GetLengthLocal();
      
      if(n != this->_length)
      {
         order.Setup( this->_length);
      }
      
      compare.resize( this->_length);
      
      for( i = 0 ; i < this->_length ; i ++)
      {
         compare[i].val = this->_data[i];
         compare[i].ord = i;
      }
      
      if(stable)
      {
         if(ascending)
         {
            std::stable_sort( compare.begin(), compare.end(), CompareStructLess<CompareStruct<T> >());
         }
         else
         {
            std::stable_sort( compare.begin(), compare.end(), CompareStructGreater<CompareStruct<T> >());
         }
      }
      else
      {
         if(ascending)
         {
            std::sort( compare.begin(), compare.end(), CompareStructLess<CompareStruct<T> >());
         }
         else
         {
            std::sort( compare.begin(), compare.end(), CompareStructGreater<CompareStruct<T> >());
         }
      }
      
      for( i = 0 ; i < this->_length ; i ++)
      {
         order[i] = compare[i].ord;
      }
      
      std::vector<CompareStruct<T> >().swap(compare);
      
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Sort(IntVectorClass<int> &order, bool ascending, bool stable);
   template int SequentialVectorClass<double>::Sort(IntVectorClass<int> &order, bool ascending, bool stable);
   
   template <typename T>
   int SequentialVectorClass<T>::Perm(IntVectorClass<int> &perm)
   {
      int   i;
      T     *temp_data;
      
      PARGEMSLR_CHKERR( perm.GetLengthLocal() != this->_length);
      
      PARGEMSLR_MALLOC( temp_data, this->_length, this->_location, T);
      PARGEMSLR_MEMCPY( temp_data, this->_data, this->_length, this->_location, this->_location, T);
      
      for(i = 0 ; i < this->_length ; i ++)
      {
         this->_data[i] = temp_data[perm[i]];
      }
      
      PARGEMSLR_FREE( temp_data, this->_location);
      
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Perm(IntVectorClass<int> &perm);
   template int SequentialVectorClass<double>::Perm(IntVectorClass<int> &perm);
   template int SequentialVectorClass<complexs>::Perm(IntVectorClass<int> &perm);
   template int SequentialVectorClass<complexd>::Perm(IntVectorClass<int> &perm);
   
   template <typename T>
   T SequentialVectorClass<T>::Max() const
   {
      T* val;
      val = std::max_element( this->_data, this->_data+this->_length);
      return *val;
   }
   template float SequentialVectorClass<float>::Max() const;
   template double SequentialVectorClass<double>::Max() const;
   
   template <typename T>
   int SequentialVectorClass<T>::MaxIndex() const
   {
      return std::distance( this->_data, std::max_element( this->_data, this->_data+this->_length));
   }
   template int SequentialVectorClass<float>::MaxIndex() const;
   template int SequentialVectorClass<double>::MaxIndex() const;
   
   template <typename T>
   T SequentialVectorClass<T>::Min() const
   {
      T* val;
      val = std::min_element( this->_data, this->_data+this->_length);
      return *val;
   }
   template float SequentialVectorClass<float>::Min() const;
   template double SequentialVectorClass<double>::Min() const;
   
   template <typename T>
   int SequentialVectorClass<T>::MinIndex() const
   {
      return std::distance( this->_data, std::min_element( this->_data, this->_data+this->_length));
   }
   template int SequentialVectorClass<float>::MinIndex() const;
   template int SequentialVectorClass<double>::MinIndex() const;
   
#endif
   
   template <typename T>
   int SequentialVectorClass<T>::BinarySearch(const T &val, int &idx, bool ascending)
   {
      if(this->_location == kMemoryDevice)
      {
         std::cout<<"Current version only support BinarySearch on the host, moving data to the host."<<std::endl;
         this->MoveData( kMemoryHost);
      }
      return VectorPBsearchHost( this->_data, val, 0, this->_length-1, idx, ascending, 2);
   }
   template int SequentialVectorClass<float>::BinarySearch(const float &val, int &idx, bool ascending);
   template int SequentialVectorClass<double>::BinarySearch(const double &val, int &idx, bool ascending);
   
   template <typename T>
   int SequentialVectorClass<T>::Plot( int conditiona, int conditionb, int width)
   {
      /* TODO: No OpenMP implemntation yet, we don't have that large array to search from */
      if(this->_location == kMemoryDevice)
      {
         std::cout<<"Plot vector only can be used on the host, moving data to the host."<<std::endl;
         this->MoveData( kMemoryHost);
      }
      return VectorPlotHost( *this, conditiona, conditionb, width);
   }
   template int SequentialVectorClass<float>::Plot( int conditiona, int conditionb, int width);
   template int SequentialVectorClass<double>::Plot( int conditiona, int conditionb, int width);
   template int SequentialVectorClass<complexs>::Plot( int conditiona, int conditionb, int width);
   template int SequentialVectorClass<complexd>::Plot( int conditiona, int conditionb, int width);
   
   template <typename T>
   int SequentialVectorClass<T>::PlotAbsGnuPlot( const char *datafilename, int conditiona, int conditionb, bool logx, bool logy, int pttype)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Plot Abs only works on host.");
      }
      
      int i;
      
      if(conditiona == conditionb)
      {
         double val, maxval = 0.0;
         
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
         
         for(i = 0 ; i < this->_length ; i ++)
         {
            val = PargemslrAbs(this->_data[i]);
            fprintf(fdata, "%d %f \n", i, val);
            
            maxval = val > maxval ? val : maxval;
            
         }
         
         fclose(fdata);
         
         fprintf(pgnuplot, "set title \"length = %d\"\n", this->_length);
         fprintf(pgnuplot, "set xrange [-1:%d]\n", this->_length);
         fprintf(pgnuplot, "set yrange [-1:%f]\n", maxval*1.1);
         if(logx)
         {
            fprintf(pgnuplot, "set logscale y\n");
         }
         if(logy)
         {
            fprintf(pgnuplot, "set logscale y\n");
         }
         fprintf(pgnuplot, "plot '%s' pt %d\n", tempfilename, pttype);
         
         pclose(pgnuplot);
      }
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::PlotAbsGnuPlot( const char *datafilename, int conditiona, int conditionb, bool logx, bool logy, int pttype);
   template int SequentialVectorClass<double>::PlotAbsGnuPlot( const char *datafilename, int conditiona, int conditionb, bool logx, bool logy, int pttype);
   template int SequentialVectorClass<complexs>::PlotAbsGnuPlot( const char *datafilename, int conditiona, int conditionb, bool logx, bool logy, int pttype);
   template int SequentialVectorClass<complexd>::PlotAbsGnuPlot( const char *datafilename, int conditiona, int conditionb, bool logx, bool logy, int pttype);
   
   template <typename T>
   int SequentialVectorClass<T>::WriteToDisk( const char *datafilename)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Can only write host data.");
      }
      
      /* define the data type */
      typedef typename std::conditional<PargemslrIsDoublePrecision<T>::value, double, float>::type RealDataType;
      typedef ComplexValueClass<RealDataType> ComplexDataType;
      
      int i;
      
      RealDataType      *rval;
      ComplexDataType   *cval;
   
      FILE *fdata;
      
      if ((fdata = fopen(datafilename, "w")) == NULL)
      {
         printf("Can't open file.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }
      
      if(PargemslrIsComplex<T>::value)
      {
         cval = (ComplexDataType*)this->_data;
         
         for(i = 0 ; i < this->_length ; i ++)
         {
            fprintf(fdata, "%16.12f %16.12f\n", cval[i].Real(), cval[i].Imag());
         }
      }
      else
      {
         rval = (RealDataType*)this->_data;
         
         for(i = 0 ; i < this->_length ; i ++)
         {
            fprintf(fdata, "%16.12f \n", rval[i]);
         }
      }
      
      fclose(fdata);
      
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::WriteToDisk( const char *datafilename);
   template int SequentialVectorClass<double>::WriteToDisk( const char *datafilename);
   template int SequentialVectorClass<complexs>::WriteToDisk( const char *datafilename);
   template int SequentialVectorClass<complexd>::WriteToDisk( const char *datafilename);
   
   template <typename T>
   int SequentialVectorClass<T>::Scale( const T &alpha)
   {
      return VectorScale( *this, alpha);
   }
   template int SequentialVectorClass<float>::Scale( const float &alpha);
   template int SequentialVectorClass<double>::Scale( const double &alpha);
   template int SequentialVectorClass<complexs>::Scale( const complexs &alpha);
   template int SequentialVectorClass<complexd>::Scale( const complexd &alpha);
   
   template <typename T>
   int SequentialVectorClass<T>::Axpy( const T &alpha, const VectorClass<T> &x)
   {
      return VectorAxpy( alpha, x, *this);
   }
   template int SequentialVectorClass<float>::Axpy( const float &alpha, const VectorClass<float> &x);
   template int SequentialVectorClass<double>::Axpy( const double &alpha, const VectorClass<double> &x);
   template int SequentialVectorClass<complexs>::Axpy( const complexs &alpha, const VectorClass<complexs> &x);
   template int SequentialVectorClass<complexd>::Axpy( const complexd &alpha, const VectorClass<complexd> &x);
   
   template <typename T>
   int SequentialVectorClass<T>::Axpy( const T &alpha, const VectorClass<T> &x, VectorClass<T> &y)
   {
      T one = T(1.0);
      T zero = T(0.0);
      this->Fill(zero);
      VectorAxpy( alpha, x, *this);
      return VectorAxpy( one, y, *this);
   }
   template int SequentialVectorClass<float>::Axpy( const float &alpha, const VectorClass<float> &x, VectorClass<float> &y);
   template int SequentialVectorClass<double>::Axpy( const double &alpha, const VectorClass<double> &x, VectorClass<double> &y);
   template int SequentialVectorClass<complexs>::Axpy( const complexs &alpha, const VectorClass<complexs> &x, VectorClass<complexs> &y);
   template int SequentialVectorClass<complexd>::Axpy( const complexd &alpha, const VectorClass<complexd> &x, VectorClass<complexd> &y);
   
   template <typename T>
   int SequentialVectorClass<T>::Axpy( const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y)
   {
      T zero = T(0.0);
      this->Fill(zero);
      VectorAxpy( alpha, x, *this);
      return VectorAxpy( beta, y, *this);
   }
   template int SequentialVectorClass<float>::Axpy( const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   template int SequentialVectorClass<double>::Axpy( const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   template int SequentialVectorClass<complexs>::Axpy( const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   template int SequentialVectorClass<complexd>::Axpy( const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
   template <typename T>
   int SequentialVectorClass<T>::Norm2( float &norm) const
   {
      T t;
      VectorDot( *this, *this, t);
      norm = sqrt(PargemslrAbs(t));
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Norm2( float &norm) const;
   template int SequentialVectorClass<complexs>::Norm2( float &norm) const;
   template int SequentialVectorClass<double>::Norm2( float &norm) const;
   template int SequentialVectorClass<complexd>::Norm2( float &norm) const;
   
   template <typename T>
   int SequentialVectorClass<T>::Norm2( double &norm) const
   {
      T t;
      VectorDot( *this, *this, t);
      norm = sqrt(PargemslrAbs(t));
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Norm2( double &norm) const;
   template int SequentialVectorClass<complexs>::Norm2( double &norm) const;
   template int SequentialVectorClass<double>::Norm2( double &norm) const;
   template int SequentialVectorClass<complexd>::Norm2( double &norm) const;
   
   template <typename T>
   int SequentialVectorClass<T>::NormInf( float &norm)
   {
      int i, length;
      float norm_temp;
      
      if(this->_location == kMemoryDevice)
      {
         std::cout<<"Current version only support inf norm on the host, moving data to the host."<<std::endl;
         this->MoveData( kMemoryHost);
      }
      
      norm = 0.0;
      length = this->GetLengthLocal();
      
      for(i = 0 ; i < length ; i ++)
      {
         norm_temp = PargemslrAbs(this->_data[i]);
         norm = norm_temp > norm ? norm_temp : norm;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::NormInf( float &norm);
   template int SequentialVectorClass<complexs>::NormInf( float &norm);
   template int SequentialVectorClass<double>::NormInf( float &norm);
   template int SequentialVectorClass<complexd>::NormInf( float &norm);
   
   template <typename T>
   int SequentialVectorClass<T>::NormInf( double &norm)
   {
      int i, length;
      double norm_temp;
      
      if(this->_location == kMemoryDevice)
      {
         std::cout<<"Current version only support inf norm on the host, moving data to the host."<<std::endl;
         this->MoveData( kMemoryHost);
      }
      
      norm = 0.0;
      length = this->GetLengthLocal();
      
      for(i = 0 ; i < length ; i ++)
      {
         norm_temp = PargemslrAbs(this->_data[i]);
         norm = norm_temp > norm ? norm_temp : norm;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::NormInf( double &norm);
   template int SequentialVectorClass<complexs>::NormInf( double &norm);
   template int SequentialVectorClass<double>::NormInf( double &norm);
   template int SequentialVectorClass<complexd>::NormInf( double &norm);
   
   template <typename T>
   int SequentialVectorClass<T>::Dot( const VectorClass<T> &y, T &t) const
   {
      
      PARGEMSLR_CHKERR( this->_length != y.GetLengthLocal() || this->_length != y.GetLengthGlobal() );
      
      return VectorDot( *this, y, t);
   }
   template int SequentialVectorClass<double>::Dot( const VectorClass<double> &y, double &t) const;
   template int SequentialVectorClass<complexd>::Dot( const VectorClass<complexd> &y, complexd &t) const;
   template int SequentialVectorClass<float>::Dot( const VectorClass<float> &y, float &t) const;
   template int SequentialVectorClass<complexs>::Dot( const VectorClass<complexs> &y, complexs &t) const;
   
#ifndef PARGEMSLR_CUDA
   template <typename T>
   int SequentialVectorClass<T>::MoveData( const int &location)
   {
      //PARGEMSLR_CHKERR( location != kMemoryHost);
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::MoveData( const int &location);
   template int SequentialVectorClass<double>::MoveData( const int &location);
   template int SequentialVectorClass<complexs>::MoveData( const int &location);
   template int SequentialVectorClass<complexd>::MoveData( const int &location);
#endif
   
   template <typename T>
   int SequentialVectorClass<T>::ReadFromMMFile(const char *vecfile, int idxin)
   {
      return SequentialVectorReadFromFile( *this, vecfile, idxin);
   }
   template int SequentialVectorClass<float>::ReadFromMMFile(const char *vecfile, int idxin);
   template int SequentialVectorClass<double>::ReadFromMMFile(const char *vecfile, int idxin);
   template int SequentialVectorClass<complexs>::ReadFromMMFile(const char *vecfile, int idxin);
   template int SequentialVectorClass<complexd>::ReadFromMMFile(const char *vecfile, int idxin);
   
}
