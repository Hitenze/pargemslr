/* this file is for the GPU version thrust call */

#ifdef PARGEMSLR_CUDA

#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "vector.hpp"
#include "int_vector.hpp"
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
   int IntVectorClass<T>::Fill(const T &v)
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
      if(v == 0)
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
   template int IntVectorClass<int>::Fill(const int &v);
   template int IntVectorClass<long int>::Fill(const long int &v);

   template <typename T>
   int IntVectorClass<T>::UnitPerm()
   {
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         PARGEMSLR_CUSPARSE_CALL( (cusparseCreateIdentityPermutation( parallel_log::_cusparse_handle, this->_length, this->_data)) );
         return PARGEMSLR_SUCCESS;
      }
      
      int i;
      for(i = 0 ; i < this->_length ; i ++)
      {
         this->_data[i] = i;
      }

      return PARGEMSLR_SUCCESS;
   }
   template int IntVectorClass<int>::UnitPerm();
   
   template <typename T>
   T IntVectorClass<T>::Max() const
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
   template int IntVectorClass<int>::Max() const;
   template long int IntVectorClass<long int>::Max() const;
   
   template <typename T>
   int IntVectorClass<T>::MaxIndex() const
   {
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<T> dev_ptr(this->_data);
         thrust::device_ptr<T> dev_ptr_val = PARGEMSLR_THRUST_CALL( max_element, dev_ptr, dev_ptr+this->_length);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         return PARGEMSLR_THRUST_CALL( distance, dev_ptr, dev_ptr_val);
      }
      else
      {
         /* host version */
         T* val_loc;
         val_loc = PARGEMSLR_THRUST_CALL( max_element, this->_data, this->_data+this->_length);
         return PARGEMSLR_THRUST_CALL( distance, this->_data, val_loc);
      }
   }
   template int IntVectorClass<int>::MaxIndex() const;
   template int IntVectorClass<long int>::MaxIndex() const;
   
   template <typename T>
   T IntVectorClass<T>::Min() const
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
         /* host version  */
         T* val_loc;
         val_loc = PARGEMSLR_THRUST_CALL( min_element, this->_data, this->_data+this->_length);
         return *val_loc;
      }
   }
   template int IntVectorClass<int>::Min() const;
   template long int IntVectorClass<long int>::Min() const;
   
   template <typename T>
   int IntVectorClass<T>::MinIndex() const
   {
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<T> dev_ptr(this->_data);
         thrust::device_ptr<T> dev_ptr_val = PARGEMSLR_THRUST_CALL( min_element, dev_ptr, dev_ptr+this->_length);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         return PARGEMSLR_THRUST_CALL( distance, dev_ptr, dev_ptr_val);
      }
      else
      {
         /* host version */
         T* val_loc;
         val_loc = PARGEMSLR_THRUST_CALL( min_element, this->_data, this->_data+this->_length);
         return PARGEMSLR_THRUST_CALL( distance, this->_data, val_loc);
      }
   }
   template int IntVectorClass<int>::MinIndex() const;
   template int IntVectorClass<long int>::MinIndex() const;
   
   template <typename T>
   int IntVectorClass<T>::Sort(bool ascending)
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
   template int IntVectorClass<int>::Sort(bool ascending);
   template int IntVectorClass<long int>::Sort(bool ascending);
   
   template <typename T>
   int IntVectorClass<T>::Sort(IntVectorClass<int> &order, bool ascending, bool stable)
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
   template int IntVectorClass<int>::Sort(IntVectorClass<int> &order, bool ascending, bool stable);
   template int IntVectorClass<long int>::Sort(IntVectorClass<int> &order, bool ascending, bool stable);
   
   template <typename T>
   int IntVectorClass<T>::Perm(IntVectorClass<int> &perm)
   {
      
      int location = perm.GetDataLocation();
      
      T* temp_data;
      
      PARGEMSLR_MALLOC( temp_data, this->_length, this->_location, T);
      PARGEMSLR_MEMCPY( temp_data, this->_data, this->_length, this->_location, this->_location, T);
      
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         if( location == kMemoryHost || location == kMemoryPinned)
         {
            //perm.MoveData( kMemoryDevice);
            //std::cout<<"Permute device vector with host permutation array, moving perm vector to the device."<<std::endl;
            PARGEMSLR_ERROR("Permute device vector with host permutation array, moving perm vector to the device.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
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
            //perm.MoveData( kMemoryHost);
            //std::cout<<"Permute host vector with device permutation array, moving perm vector to the host."<<std::endl;
            PARGEMSLR_ERROR("Permute host vector with device permutation array, moving perm vector to the host.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         /* host version */
         PARGEMSLR_THRUST_CALL( gather, perm.GetData(), perm.GetData()+this->_length, temp_data, this->_data);
         
      }
      
      PARGEMSLR_FREE( temp_data, this->_location);
      
      return PARGEMSLR_SUCCESS;
   }
   template int IntVectorClass<int>::Perm(IntVectorClass<int> &perm);
   template int IntVectorClass<long int>::Perm(IntVectorClass<int> &perm);
   
   template <typename T>
   template <typename T1>
   int IntVectorClass<T>::ScatterRperm(const VectorVirtualClass<T1> &v_in, VectorVirtualClass<T1> &v_out)
   {
      PARGEMSLR_CHKERR(v_in.GetLengthLocal() < this->_length);
      
      int loc_in = v_in.GetDataLocation();
      int loc_out = v_out.GetDataLocation();
      
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         if( loc_in == kMemoryHost || loc_in == kMemoryPinned)
         {
            //v_in.MoveData( kMemoryDevice);
            //std::cout<<"Scatter host input vector with divice permutation array, moving input vector to the device."<<std::endl;
            PARGEMSLR_ERROR("Scatter host input vector with divice permutation array.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         if( loc_out == kMemoryHost || loc_out == kMemoryPinned)
         {
            //v_out.MoveData( kMemoryDevice);
            //std::cout<<"Scatter host output vector with divice permutation array, moving output vector to the device."<<std::endl;
            PARGEMSLR_ERROR("Scatter host output vector with divice permutation array.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<T1> dev_ptr_in(v_in.GetData());
         thrust::device_ptr<T1> dev_ptr_out(v_out.GetData());
         thrust::device_ptr<T> dev_ptr_map(this->_data);
         PARGEMSLR_THRUST_CALL( scatter, dev_ptr_in, dev_ptr_in+this->_length, dev_ptr_map, dev_ptr_out);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
      }
      else
      {
         if( loc_in == kMemoryDevice || loc_in == kMemoryUnified)
         {
            //v_in.MoveData( kMemoryHost);
            //std::cout<<"Scatter device input vector with host permutation array, moving input vector to the host."<<std::endl;
            PARGEMSLR_ERROR("Scatter device input vector with host permutation array.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         if( loc_out == kMemoryDevice || loc_out == kMemoryUnified)
         {
            //v_out.MoveData( kMemoryHost);
            //std::cout<<"Scatter device output vector with host permutation array, moving output vector to the host."<<std::endl;
            PARGEMSLR_ERROR("Scatter device output vector with host permutation array.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         /* host version */
         PARGEMSLR_THRUST_CALL( scatter, v_in.GetData(), v_in.GetData()+this->_length, this->_data, v_out.GetData());
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<int> &v_in, VectorVirtualClass<int> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<long int> &v_in, VectorVirtualClass<long int> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<float> &v_in, VectorVirtualClass<float> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<double> &v_in, VectorVirtualClass<double> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<complexs> &v_in, VectorVirtualClass<complexs> &v_out);
   template int IntVectorClass<int>::ScatterRperm(const VectorVirtualClass<complexd> &v_in, VectorVirtualClass<complexd> &v_out);
   
   template <typename T>
   template <typename T1>
   int IntVectorClass<T>::GatherPerm(const VectorVirtualClass<T1> &v_in, VectorVirtualClass<T1> &v_out)
   {
      PARGEMSLR_CHKERR(v_out.GetLengthLocal() < this->_length);
      
      int loc_in = v_in.GetDataLocation();
      int loc_out = v_out.GetDataLocation();
      
      if( this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         if( loc_in == kMemoryHost || loc_in == kMemoryPinned)
         {
            //v_in.MoveData( kMemoryDevice);
            //std::cout<<"Gather host input vector with divice permutation array, moving input vector to the device."<<std::endl;
            PARGEMSLR_ERROR("Gather host input vector with divice permutation array.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         if( loc_out == kMemoryHost || loc_out == kMemoryPinned)
         {
            //v_out.MoveData( kMemoryDevice);
            //std::cout<<"Gather host output vector with divice permutation array, moving output vector to the device."<<std::endl;
            PARGEMSLR_ERROR("Gather host output vector with divice permutation array.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         //PARGEMSLR_CUDA_SYNCHRONIZE;
         thrust::device_ptr<T1> dev_ptr_in(v_in.GetData());
         thrust::device_ptr<T1> dev_ptr_out(v_out.GetData());
         thrust::device_ptr<T> dev_ptr_map(this->_data);
         PARGEMSLR_THRUST_CALL( gather, dev_ptr_map, dev_ptr_map+this->_length, dev_ptr_in, dev_ptr_out);
         //PARGEMSLR_CUDA_SYNCHRONIZE;
      }
      else
      {
         if( loc_in == kMemoryDevice || loc_in == kMemoryUnified)
         {
            //v_in.MoveData( kMemoryHost);
            //std::cout<<"Gather device input vector with host permutation array, moving input vector to the host."<<std::endl;
            PARGEMSLR_ERROR("Gather device input vector with host permutation array.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         if( loc_out == kMemoryDevice || loc_out == kMemoryUnified)
         {
            //v_out.MoveData( kMemoryHost);
            //std::cout<<"Gather device output vector with host permutation array, moving output vector to the host."<<std::endl;
            PARGEMSLR_ERROR("Gather device input vector with host permutation array.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
         /* the host version */
         PARGEMSLR_THRUST_CALL( gather, this->_data, this->_data+this->_length, v_in.GetData(), v_out.GetData());
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<int> &v_in, VectorVirtualClass<int> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<long int> &v_in, VectorVirtualClass<long int> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<float> &v_in, VectorVirtualClass<float> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<double> &v_in, VectorVirtualClass<double> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<complexs> &v_in, VectorVirtualClass<complexs> &v_out);
   template int IntVectorClass<int>::GatherPerm(const VectorVirtualClass<complexd> &v_in, VectorVirtualClass<complexd> &v_out);
   
   template <typename T>
   int IntVectorClass<T>::MoveData( const int &location)
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
      _data = new_data;
      _location = location;
      
      return PARGEMSLR_SUCCESS;
   }
   template int IntVectorClass<int>::MoveData( const int &location);
   template int IntVectorClass<long int>::MoveData( const int &location);
   
}
#endif
