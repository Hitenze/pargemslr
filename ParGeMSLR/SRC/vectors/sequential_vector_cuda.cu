/* this file is for the GPU version thrust call */

#ifdef PARGEMSLR_CUDA

#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "vector.hpp"
#include "sequential_vector.hpp"
#include "vectorops.hpp"
#include "../utils/utils.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/distance.h>

namespace pargemslr
{  
   template <typename T>
   int SequentialVectorClass<T>::Fill(const T &v)
   {
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<T> dev_ptr(_data);
         PARGEMSLR_THRUST_CALL( fill, dev_ptr, dev_ptr+this->_length, v);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         return PARGEMSLR_SUCCESS;
      }
      
      int i;
      if(PargemslrAbs(v) == 0)
      {
         /* in this case Fill memory */
         memset( this->_data, 0, sizeof(T) * this->_length);
      }
      else
      {
         for(i = 0 ; i < this->_length ; i ++)
         {
            this->_data[i] = v;
         }
      }

      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Fill(const float &v);
   template int SequentialVectorClass<double>::Fill(const double &v);
   template int SequentialVectorClass<complexs>::Fill(const complexs &v);
   template int SequentialVectorClass<complexd>::Fill(const complexd &v);
   
   template <typename T>
   int SequentialVectorClass<T>::Rand()
   {
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         VectorRandDevice(*this);
         return PARGEMSLR_SUCCESS;
      }
	   
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
         if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
         {
            //PARGEMSLR_CUDA_SYNCHRONIZE;
            thrust::device_ptr<T> dev_ptr(_data);
            PARGEMSLR_THRUST_CALL( sort, dev_ptr, dev_ptr+this->_length);
            //PARGEMSLR_CUDA_SYNCHRONIZE;
            return PARGEMSLR_SUCCESS;
         }
         /* host version */
         PARGEMSLR_THRUST_CALL( sort, this->_data, this->_data + this->_length);
         //std::sort( this->_data, this->_data + this->_length, std::less<T>());
      }
      else
      {
         if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
         {
            //PARGEMSLR_CUDA_SYNCHRONIZE;
            thrust::device_ptr<T> dev_ptr(_data);
            PARGEMSLR_THRUST_CALL( sort, dev_ptr, dev_ptr+this->_length, thrust::greater<T>());
            //PARGEMSLR_CUDA_SYNCHRONIZE;
            return PARGEMSLR_SUCCESS;
         }
         /* host version */
         PARGEMSLR_THRUST_CALL( sort, this->_data, this->_data + this->_length, thrust::greater<T>());
         //std::sort( this->_data, this->_data + this->_length, std::greater<T>());
      }

      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Sort(bool ascending);
   template int SequentialVectorClass<double>::Sort(bool ascending);
   
   template <typename T>
   int SequentialVectorClass<T>::Sort(IntVectorClass<int> &order, bool ascending, bool stable)
   {
      
      int n = order.GetLengthLocal();
      int location = order.GetDataLocation();
      
      if(n != this->_length)
      {
         order.Setup( this->_length);
      }
      
      order.UnitPerm();
      
      T* temp_data;
      
      PARGEMSLR_MALLOC( temp_data, this->_length, location, T);
      PARGEMSLR_MEMCPY( temp_data, this->_data, this->_length, this->_location, this->_location, T);
      
      if(stable)
      {
         if(ascending)
         {
            if( location == kMemoryDevice || location == kMemoryUnified)
            {
               //PARGEMSLR_CUDA_SYNCHRONIZE;
               thrust::device_ptr<T> dev_ptr_key(temp_data);
               thrust::device_ptr<int> dev_ptr_value(order.GetData());
               PARGEMSLR_THRUST_CALL( stable_sort_by_key, dev_ptr_key, dev_ptr_key+this->_length, dev_ptr_value);
               //PARGEMSLR_CUDA_SYNCHRONIZE;
               return PARGEMSLR_SUCCESS;
            }
            /* host version */
            PARGEMSLR_THRUST_CALL( stable_sort_by_key, temp_data, temp_data+this->_length, order.GetData());
         }
         else
         {
            if( location == kMemoryDevice || location == kMemoryUnified)
            {
               //PARGEMSLR_CUDA_SYNCHRONIZE;
               thrust::device_ptr<T> dev_ptr_key(temp_data);
               thrust::device_ptr<int> dev_ptr_value(order.GetData());
               PARGEMSLR_THRUST_CALL( stable_sort_by_key, dev_ptr_key, dev_ptr_key+this->_length, dev_ptr_value, thrust::greater<T>());
               //PARGEMSLR_CUDA_SYNCHRONIZE;
               return PARGEMSLR_SUCCESS;
            }
            /* host version */
            PARGEMSLR_THRUST_CALL( stable_sort_by_key, temp_data, temp_data+this->_length, order.GetData(), thrust::greater<T>());
         }
      }
      else
      {
         if(ascending)
         {
            if( location == kMemoryDevice || location == kMemoryUnified)
            {
               //PARGEMSLR_CUDA_SYNCHRONIZE;
               thrust::device_ptr<T> dev_ptr_key(temp_data);
               thrust::device_ptr<int> dev_ptr_value(order.GetData());
               PARGEMSLR_THRUST_CALL( sort_by_key, dev_ptr_key, dev_ptr_key+this->_length, dev_ptr_value);
               //PARGEMSLR_CUDA_SYNCHRONIZE;
               return PARGEMSLR_SUCCESS;
            }
            /* host version */
            PARGEMSLR_THRUST_CALL( sort_by_key, temp_data, temp_data+this->_length, order.GetData());
         }
         else
         {
            if( location == kMemoryDevice || location == kMemoryUnified)
            {
               //PARGEMSLR_CUDA_SYNCHRONIZE;
               thrust::device_ptr<T> dev_ptr_key(temp_data);
               thrust::device_ptr<int> dev_ptr_value(order.GetData());
               PARGEMSLR_THRUST_CALL( sort_by_key, dev_ptr_key, dev_ptr_key+this->_length, dev_ptr_value, thrust::greater<T>());
               //PARGEMSLR_CUDA_SYNCHRONIZE;
               return PARGEMSLR_SUCCESS;
            }
            /* host version */
            PARGEMSLR_THRUST_CALL( sort_by_key, temp_data, temp_data+this->_length, order.GetData(), thrust::greater<T>());
         }
      }
      
      PARGEMSLR_FREE( temp_data, location);
      
      return PARGEMSLR_SUCCESS;
   }
   template int SequentialVectorClass<float>::Sort(IntVectorClass<int> &order, bool ascending, bool stable);
   template int SequentialVectorClass<double>::Sort(IntVectorClass<int> &order, bool ascending, bool stable);
   
   template <typename T>
   int SequentialVectorClass<T>::Perm(IntVectorClass<int> &perm)
   {
      
      int location = perm.GetDataLocation();
      
      T* temp_data;
      
      PARGEMSLR_MALLOC( temp_data, this->_length, this->_location, T);
      PARGEMSLR_MEMCPY( temp_data, this->_data, this->_length, this->_location, this->_location, T);
      
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         if( location == kMemoryHost || location == kMemoryPinned)
         {
            PARGEMSLR_ERROR("Permute device vector with host permutation array.");
            return PARGEMSLR_ERROR_MEMORY_LOCATION;
         }
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<int> dev_ptr_map(perm.GetData());
         thrust::device_ptr<T> dev_ptr_in(temp_data);
         thrust::device_ptr<T> dev_ptr_out(this->_data);
         PARGEMSLR_THRUST_CALL( gather, dev_ptr_map, dev_ptr_map+this->_length, dev_ptr_in, dev_ptr_out);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
      }
      else
      {
         if( location == kMemoryDevice || location == kMemoryUnified)
         {
            PARGEMSLR_ERROR("Permute host vector with device permutation array.");
            return PARGEMSLR_ERROR_MEMORY_LOCATION;
         }
         /* host version */
         PARGEMSLR_THRUST_CALL( gather, perm.GetData(), perm.GetData()+this->_length, temp_data, this->_data);
         
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
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<T> dev_ptr(this->_data);
         thrust::device_ptr<T> dev_ptr_val = PARGEMSLR_THRUST_CALL( max_element, dev_ptr, dev_ptr+this->_length);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         return dev_ptr_val[0];
      }
      else
      {
         /* host version */
         T* val_loc;
         val_loc = PARGEMSLR_THRUST_CALL( max_element, this->_data, this->_data+this->_length);
         return *val_loc;
      }
   }
   template float SequentialVectorClass<float>::Max() const;
   template double SequentialVectorClass<double>::Max() const;
   
   template <typename T>
   int SequentialVectorClass<T>::MaxIndex() const
   {
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<T> dev_ptr(this->_data);
         thrust::device_ptr<T> dev_ptr_val = PARGEMSLR_THRUST_CALL( max_element, dev_ptr, dev_ptr+this->_length);
         PARGEMSLR_THRUST_CALL( distance, dev_ptr, dev_ptr_val);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         return PARGEMSLR_SUCCESS;
      }
      else
      {
         /* host version */
         T* val_loc;
         val_loc = PARGEMSLR_THRUST_CALL( max_element, this->_data, this->_data+this->_length);
         return PARGEMSLR_THRUST_CALL( distance, this->_data, val_loc);
      }
   }
   template int SequentialVectorClass<float>::MaxIndex() const;
   template int SequentialVectorClass<double>::MaxIndex() const;
   
   template <typename T>
   T SequentialVectorClass<T>::Min() const
   {
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<T> dev_ptr(this->_data);
         thrust::device_ptr<T> dev_ptr_val = PARGEMSLR_THRUST_CALL( min_element, dev_ptr, dev_ptr+this->_length);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         return dev_ptr_val[0];
      }
      else
      {
         /* host version */
         T* val_loc;
         val_loc = PARGEMSLR_THRUST_CALL( min_element, this->_data, this->_data+this->_length);
         return *val_loc;
      }
   }
   template float SequentialVectorClass<float>::Min() const;
   template double SequentialVectorClass<double>::Min() const;
   
   template <typename T>
   int SequentialVectorClass<T>::MinIndex() const
   {
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<T> dev_ptr(this->_data);
         thrust::device_ptr<T> dev_ptr_val = PARGEMSLR_THRUST_CALL( min_element, dev_ptr, dev_ptr+this->_length);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         PARGEMSLR_THRUST_CALL( distance, dev_ptr, dev_ptr_val);
         return PARGEMSLR_SUCCESS;
      }
      else
      {
         /* host version */
         T* val_loc;
         val_loc = PARGEMSLR_THRUST_CALL( min_element, this->_data, this->_data+this->_length);
         return PARGEMSLR_THRUST_CALL( distance, this->_data, val_loc);
      }
   }
   template int SequentialVectorClass<float>::MinIndex() const;
   template int SequentialVectorClass<double>::MinIndex() const;
   
   template <typename T>
   int SequentialVectorClass<T>::MoveData( const int &location)
   {
      if( _location == location || _maxlength == 0)
      {
         /* in this case no need to move */
         this->_location = location;
         return PARGEMSLR_SUCCESS;
      }
      
      /* move data only when necessary */
      T *new_data;
    
      /* allocate new memory and copy current data in */
      PARGEMSLR_MALLOC( new_data, _maxlength, location, T);
      PARGEMSLR_MEMCPY( new_data, _data, _maxlength, location, _location, T);
      if(this->_hold_data)
      {
         /* when owning data, we can free the original one */
         PARGEMSLR_FREE( _data, _location);
      }
      else
      {
         /* if this is a pointer, we can't free the original data, but now we own the data */
         this->_hold_data = true;
      }
      this->_data = new_data;
      this->_location = location;
      
      if(location == kMemoryDevice || location == kMemoryUnified)
      {
         SequentialVectorCreateCusparseDnVec( *this);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int SequentialVectorClass<float>::MoveData( const int &location);
   template int SequentialVectorClass<double>::MoveData( const int &location);
   template int SequentialVectorClass<complexs>::MoveData( const int &location);
   template int SequentialVectorClass<complexd>::MoveData( const int &location);
   
}
#endif
