
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>

#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#endif

#include "utils.hpp"
#include "parallel.hpp"
#include "memory.hpp"

using namespace std;

namespace pargemslr
{

   /**
    * @brief   Set memory to CPU if compile without GPU.
    * @details Set memory to CPU if compile without GPU.
    * @param [in] location Location of the memory
    */
   static inline int PargemslrCheckMemLocation(int location)
   {
      /* Set memory to CPU if compile without GPU */
#ifndef PARGEMSLR_CUDA
      return kMemoryHost;
#else
      return location;
#endif
   }

   /**
    * @brief   Malloc host memory.
    * @details Malloc host memory.
    * @param [in] size size of the memory.
    */
   static inline void* PargemslrMallocHost(size_t size)
   {
      void *ptr = NULL;
      ptr = malloc(size);
      return ptr;
   }

   /**
    * @brief   Calloc host memory.
    * @details Calloc host memory.
    * @note    Inline function.
    * @param [in] length length of the array
    * @param [in] untisize size of each element in the array.
    */
   static inline void* PargemslrCallocHost(size_t length, int unitsize)
   {
      void *ptr = NULL;
      ptr = calloc(length, unitsize);
      return ptr;
   }

   /**
    * @brief   Realloc host memory.
    * @details Realloc host memory.
    * @param [in] prt pointer to the previous.
    * @param [in] size size of the memory.
    */
   static inline void* PargemslrReallocHost(void *ptr, size_t size)
   {
      return realloc( ptr, size );
   }

   /**
    * @brief   Free host memory.
    * @details Free host memory.
    * @note    Inline function.
    * @param [in] ptr the pointer to be freed
    */
   static inline void PargemslrFreeHost(void *ptr)
   {
      free(ptr);
   }

   /**
    * @brief   Malloc device memory.
    * @details Malloc device memory.
    * @param [in] size size of the memory.
    * @return  Return the pointer to the memory.
    */
   static inline void* PargemslrMallocDevice(int size)
   {
      void        *ptr  = NULL;
#ifdef PARGEMSLR_CUDA
      PARGEMSLR_CUDA_CALL( cudaMalloc( &ptr, size) );
#else
      cout<<"Error: attempt to allocate device memory with the host-only version."<<endl;
      exit(0);
#endif
      return ptr;
   }

   /**
    * @brief   Calloc device memory.
    * @details Calloc device memory.
    * @note    Inline function.
    * @param [in] length length of the array
    * @param [in] untisize size of each element in the array.
    * @return  Return the pointer to the memory.
    */
   static inline void* PargemslrCallocDevice(size_t length, int unitsize)
   {
      void        *ptr = NULL;
#ifdef PARGEMSLR_CUDA
      /* allocate memory, and use cudaMemset to set to 0 */
      size_t size = length * unitsize;
      PARGEMSLR_CUDA_CALL( cudaMalloc( &ptr, size) );
      PARGEMSLR_CUDA_CALL( cudaMemset( ptr, 0, size) );
#else
      cout<<"Error: attempt to allocate device memory with the host-only version."<<endl;
      exit(0);
#endif
      return ptr;
   }

   /**
    * @brief   Free device memory.
    * @details Free device memory.
    * @note    Inline function.
    * @param [in] ptr the pointer to be freed
    */
   static inline void PargemslrFreeDevice(void *ptr)
   {
#ifdef PARGEMSLR_CUDA
      PARGEMSLR_CUDA_CALL( cudaFree( ptr) );
#else
      cout<<"Error: attempt to free device memory with the host-only version."<<endl;
      exit(0);
#endif
   }

   /**
    * @brief   Malloc unified memory.
    * @details Malloc unified memory.
    * @param [in] size size of the memory.
    * @return  Return the pointer to the memory.
    */
   static inline void* PargemslrMallocUnified(size_t size)
   {
      void        *ptr  = NULL;
#ifdef PARGEMSLR_CUDA
      PARGEMSLR_CUDA_CALL( cudaMallocManaged( &ptr, size) );
#else
      cout<<"Error: attempt to allocate unified memory with the host-only version."<<endl;
      exit(0);
#endif
      return ptr;
   }

   /**
    * @brief   Calloc unified memory.
    * @details Calloc unified memory.
    * @note    Inline function.
    * @param [in] length length of the array
    * @param [in] untisize size of each element in the array.
    * @return  Return the pointer to the memory.
    */
   static inline void* PargemslrCallocUnified(size_t length, int unitsize)
   {
      void        *ptr = NULL;
#ifdef PARGEMSLR_CUDA
      size_t size = length * unitsize;
      PARGEMSLR_CUDA_CALL( cudaMallocManaged( &ptr, size) );
      PARGEMSLR_CUDA_CALL( cudaMemset( ptr, 0, size) );
#else
      cout<<"Error: attempt to allocate unified memory with the host-only version."<<endl;
      exit(0);
#endif
      return ptr;
   }

   /**
    * @brief   Free device memory.
    * @details Free device memory.
    * @note    Inline function.
    * @param [in] ptr the pointer to be freed
    */
   static inline void PargemslrFreeUnified(void *ptr)
   {
#ifdef PARGEMSLR_CUDA
      PARGEMSLR_CUDA_CALL( cudaFree( ptr) );
#else
      cout<<"Error: attempt to free unified memory with the host-only version."<<endl;
      exit(0);
#endif
   }

   /**
    * @brief   Malloc page-locked memory.
    * @details Malloc page-locked memory.
    * @param [in] size size of the memory.
    * @return  Return the pointer to the memory.
    */
   static inline void* PargemslrMallocPinned(size_t size)
   {
      void        *ptr  = NULL;
#ifdef PARGEMSLR_CUDA
      PARGEMSLR_CUDA_CALL( cudaMallocHost( &ptr, size) );
#else
      cout<<"Error: attempt to allocate page-locked memory with the host-only version."<<endl;
      exit(0);
#endif
      return ptr;
   }

   /**
    * @brief   Calloc page-locked memory.
    * @details Calloc page-locked memory.
    * @note    Inline function.
    * @param [in] length length of the array
    * @param [in] untisize size of each element in the array.
    * @return  Return the pointer to the memory.
    */
   static inline void* PargemslrCallocPinned(size_t length, int unitsize)
   {
      void        *ptr = NULL;
#ifdef PARGEMSLR_CUDA
      size_t size = length * unitsize;
      PARGEMSLR_CUDA_CALL( cudaMallocHost( &ptr, size) );
      PARGEMSLR_CUDA_CALL( cudaMemset( ptr, 0, size) );
#else
      cout<<"Error: attempt to allocate page-locked memory with the host-only version."<<endl;
      exit(0);
#endif
      return ptr;
   }

   /**
    * @brief   Free page-locked memory.
    * @details Free page-locked memory.
    * @note    Inline function.
    * @param [in] ptr the pointer to be freed
    */
   static inline void PargemslrFreePinned(void *ptr)
   {
#ifdef PARGEMSLR_CUDA
      PARGEMSLR_CUDA_CALL( cudaFreeHost( ptr) );
#else
      cout<<"Error: attempt to free page-locked memory with the host-only version."<<endl;
      exit(0);
#endif
   }

   /**
    * @brief   Wrapper function for malloc.
    * @details Wrapper function for malloc.
    * @note    Inline function.
    * @param [in] size size of the memory in byte.
    * @param [in] location location of the memory.
    * @return  Return the pointer to the memory.
    */
   static inline void* PargemslrMallocWrapper(size_t size, int location)
   {
      if (size == 0)
      {
         return NULL;
      }
      void     *ptr  = NULL;
      /* check location, for the host-only version move everything to host */
      location       = PargemslrCheckMemLocation(location);

      switch (location)
      {
         case kMemoryHost:
         {
            /* memroy on the host */
            ptr = PargemslrMallocHost(size);
            break;
         }
         case kMemoryDevice:
         {
            /* memory on the device */
            ptr = PargemslrMallocDevice(size);
            break;
         }
         case kMemoryUnified:
         {
            /* shared memory */
            ptr = PargemslrMallocUnified(size);
            break;
         }
         case kMemoryPinned:
         {
            /* page-locked memory */
            ptr = PargemslrMallocPinned(size);
            break;
         }
         default:
         {
            cout<<"Error: invalied memory location."<<endl;
            exit(0);
         }
      }

      if (!ptr)
      {
         cout<<"Error: memory allocation failed."<<endl;
         exit(0);
      }

      return ptr;
   }

   /**
    * @brief   Wrapper function for calloc.
    * @details Wrapper function for calloc.
    * @note    Inline function.
    * @param [in] length length of the array
    * @param [in] untisize size of each element in the array.
    * @param [in] location location of the memory.
    * @return  Return the pointer to the memory.
    */
   static inline void* PargemslrCallocWrapper(size_t length, int unitsize, int location)
   {

      PARGEMSLR_CHKERR(length < 0);
      PARGEMSLR_CHKERR(unitsize < 0);

      if (length == 0 || unitsize == 0)
      {
         return NULL;
      }
      void     *ptr  = NULL;
      //check location
      location       = PargemslrCheckMemLocation(location);

      switch (location)
      {
         case kMemoryHost:
         {
            /* memroy on the host */
            ptr = PargemslrCallocHost(length, unitsize);
            break;
         }
         case kMemoryDevice:
         {
            /* device memory */
            ptr = PargemslrCallocDevice(length, unitsize);
            break;
         }
         case kMemoryUnified:
         {
            /* shared memory */
            ptr = PargemslrCallocUnified(length, unitsize);
            break;
         }
         case kMemoryPinned:
         {
            /* page-locked memory */
            ptr = PargemslrCallocPinned(length, unitsize);
            break;
         }
         default:
         {
            cout<<"Error: invalied memory location."<<endl;
            exit(0);
         }
      }

      if (!ptr)
      {
         cout<<"Error: memory allocation failed."<<endl;
         exit(0);
      }

      return ptr;
   }

   /**
    * @brief   Memory copy function.
    * @details Memory copy function.
    * @note    Inline function.
    * @param [in] ptr_to Copy to this address.
    * @param [in] ptr_from Copy from this address.
    * @param [in] size size of the memory in bytes.
    * @param [in] loc_to Location of ptr_to.
    * @param [in] loc_from Location of ptr_from.
    */
   static inline void PargemslrMemcpyWrapper(void *ptr_to, const void *ptr_from, size_t size, int loc_to, int loc_from)
   {
      if (size == 0)
      {
         return;
      }
      /* check location, avoid using GPU when no device availiable */
      loc_to         = PargemslrCheckMemLocation(loc_to);
      loc_from       = PargemslrCheckMemLocation(loc_from);

      switch (loc_to)
      {
         case kMemoryHost:
            switch (loc_from)
            {
               case kMemoryDevice:
               {
                  /* memcpy from device to host */
#ifdef PARGEMSLR_CUDA
                  cudaMemcpy( ptr_to, ptr_from, size, cudaMemcpyDeviceToHost);
#else
                  cout<<"Error: accessing device memory with the host-only version."<<endl;
                  exit(0);
#endif
                  break;
               }
               case kMemoryHost:
               case kMemoryUnified:
               case kMemoryPinned:
               {
                  /* memcpy from host to host */
                  memcpy( ptr_to, ptr_from, size);
                  break;
               }
               default:
               {
                  cout<<"Error: invalied memory location."<<endl;
                  exit(0);
               }
            }
            break;
         case kMemoryDevice:
         {
            switch (loc_from)
            {
               case kMemoryDevice:
               case kMemoryUnified:
               {
                  /* memcpy from device to device */
#ifdef PARGEMSLR_CUDA
                  cudaMemcpy( ptr_to, ptr_from, size, cudaMemcpyDeviceToDevice);
#else
                  cout<<"Error: accessing device memory with the host-only version."<<endl;
                  exit(0);
#endif
                  break;
               }
               case kMemoryHost:
               case kMemoryPinned:
               {
                  /* memcpy from host to device */
#ifdef PARGEMSLR_CUDA
                  cudaMemcpy( ptr_to, ptr_from, size, cudaMemcpyHostToDevice);
#else
                  cout<<"Error: accessing device memory with the host-only version."<<endl;
                  exit(0);
#endif
                  break;
               }
               default:
               {
                  cout<<"Error: invalied memory location."<<endl;
                  exit(0);
               }
            }
            break;
         }
         case kMemoryUnified:
         {
            switch (loc_from)
            {
               case kMemoryUnified:
               case kMemoryDevice:
               {
                  /* memcpy from device to device */
#ifdef PARGEMSLR_CUDA
                  cudaMemcpy( ptr_to, ptr_from, size, cudaMemcpyDeviceToDevice);
#else
                  cout<<"Error: accessing device memory with the host-only version."<<endl;
                  exit(0);
#endif
                  break;
               }
               case kMemoryHost:
               case kMemoryPinned:
               {
                  /* memcpy from host to host */
                  memcpy( ptr_to, ptr_from, size);
                  break;
               }
               default:
               {
                  cout<<"Error: invalied memory location."<<endl;
                  exit(0);
               }
            }
            break;
         }
         case kMemoryPinned:
         {
            switch (loc_from)
            {
               case kMemoryDevice:
               {
                  //memcpy from device to host
#ifdef PARGEMSLR_CUDA
                  cudaMemcpy( ptr_to, ptr_from, size, cudaMemcpyDeviceToHost);
#else
                  cout<<"Error: accessing device memory with the host-only version."<<endl;
                  exit(0);
#endif
                  break;
               }
               case kMemoryHost:
               case kMemoryUnified:
               case kMemoryPinned:
               {
                  //memcpy from host to host
                  memcpy( ptr_to, ptr_from, size);
                  break;
               }
               default :
               {
                  cout<<"Error: invalied memory location."<<endl;
                  exit(0);
               }
            }
            break;
         }
         default:
         {
            cout<<"Error: invalied memory location."<<endl;
            exit(0);
         }
      }
   }

   /**
    * @brief   Free the memory.
    * @details Free the memory.
    * @note    Inline function.
    * @param [in] ptr Pointer to the memory.
    * @param [in] location Location of the memory.
    */
   static inline void PargemslrFreeWrapper(void *ptr, int location)
   {
      if (!ptr)
      {
         return;
      }
      
      //check location
      location       = PargemslrCheckMemLocation(location);

      switch (location)
      {
         case kMemoryHost:
         {
            //memroy on CPU
            PargemslrFreeHost(ptr);
            break;
         }
         case kMemoryDevice:
         {
            //memory on GPU
            PargemslrFreeDevice(ptr);
            break;
         }
         case kMemoryUnified:
         {
            //memory that is shared by CPU and GPU
            PargemslrFreeUnified(ptr);
            break;
         }
         case kMemoryPinned:
         {
            //memory on CPU that is page-locked
            PargemslrFreePinned(ptr);
            break;
         }
         default :
         {
            cout<<"Error: invalied memory location."<<endl;
            exit(0);
         }
      }
   }

   /**
    * @brief   Wrapper function for realloc.
    * @details Wrapper function for realloc.
    * @note    Inline function.
    * @param [in] ptr pointer to the previous memory.
    * @param [in] size size of the memory in byte.
    * @param [in] location location of the memory.
    * @return  Return the pointer to the memory.
    */
   static inline void* PargemslrReallocWrapper(void* ptr, size_t current_size, size_t new_size, int location)
   {
      PARGEMSLR_CHKERR(new_size < 0);
      PARGEMSLR_CHKERR(current_size < 0);
    
      if (new_size == 0)
      {
         PargemslrFreeWrapper(ptr, location);
         return NULL;
      }
      //check location
      location       = PargemslrCheckMemLocation(location);

      switch (location)
      {
         case kMemoryHost:
         {
            /* memroy on the host */
            ptr = PargemslrReallocHost(ptr, new_size);
            break;
         }
         case kMemoryDevice:
         case kMemoryUnified:
         case kMemoryPinned:
         {
            void *ptr2 = PargemslrMallocWrapper(new_size, location);
            if( new_size >= current_size)
            {
               PargemslrMemcpyWrapper( ptr2, ptr, current_size, location, location);
            }
            else
            {
               PargemslrMemcpyWrapper( ptr2, ptr, new_size, location, location);
            }
            PargemslrFree(ptr, location);
            ptr = ptr2;
            break;
         }
         default:
         {
            cout<<"Error: invalied memory location."<<endl;
            exit(0);
         }
      }

      if (!ptr)
      {
         cout<<"Error: memory allocation failed."<<endl;
         exit(0);
      }

      return ptr;
   }

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 * User functions
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   
   void* PargemslrMalloc( size_t length, int location)
   {
      return PargemslrMallocWrapper( length, location);
   }
   
   void* PargemslrCalloc( size_t length, int location)
   {
      return PargemslrCallocWrapper( length, 1, location);
   }
   
   void* PargemslrRealloc( void* ptr, size_t current_length, size_t new_length, int location)
   {
      return PargemslrReallocWrapper( (void*) ptr, current_length, new_length, location);
   }
   
   void PargemslrMemcpy( void* ptr_to, const void* ptr_from, size_t length, int loc_to, int loc_from)
   {
      PargemslrMemcpyWrapper( (void*) ptr_to, (void*) ptr_from, length, loc_to, loc_from);
   }
   
   void PargemslrFree( void* ptr, int location)
   {
      if(ptr)
      {
         PargemslrFreeWrapper( ptr, location);
      }
   }

#ifdef PARGEMSLR_DEBUG_MEMORY
   
   std::unordered_map< void*, memory_debugger> memory_debugger::_memory_tracker = std::unordered_map< void*, memory_debugger>();
   
   /* TBA, currently not availiable */
   void memory_debugger::CheckMemoryLocation( const void *ptr, int location, const char *filename, const char *function, const int line)
   {
      return;
   }
   
   void memory_debugger::InsertMemoryAction( char action, size_t length, const void *address, const char *filename, const char *function, const int line)
   {
      return;
   }
   
#endif

}
