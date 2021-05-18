#ifndef PARGEMSLR_MEMORY_H
#define PARGEMSLR_MEMORY_H

#include <vector>
#include <unordered_map>

/**
 * @file memory.hpp
 * @brief Memory menagement functions for CPU and GPU.
 */

namespace pargemslr
{
   
   /**
    * @brief   The memory location. Host, device, shared, page-locked.
    * @details The memory location. Host, device, shared, page-locked.
    */
   enum MemoryLocationEnum
   {
      kMemoryHost,
      kMemoryDevice,
      kMemoryUnified,
      kMemoryPinned
   };
   
   /**
    * @brief   Malloc memory at given location.
    * @details Malloc memory at given location.
    * @param [in]  length The length of the array.
    * @param [in]  location The location of the array.
    * @return      Return pointer to the memory location.
    */
   void* PargemslrMalloc( size_t length, int location);
   /**
    * @brief   Calloc memory at given location.
    * @details Calloc memory at given location.
    * @param [in]  length The length of array.
    * @param [in]  location The location of the array.
    * @return      Return pointer to the memory location.
    */
   void* PargemslrCalloc( size_t length, int location);
   
   /**
    * @brief   Realloc memory at given location.
    * @details Realloc memory at given location.
    * @param [in]  ptr The pointer to the memory location.
    * @param [in]  current_length Current length of the array.
    * @param [in]  new_length Wanted length of the array.
    * @param [in]  location The location of the array.
    * @return      Return pointer to the memory location.
    */
   void* PargemslrRealloc( void* ptr, size_t current_length, size_t new_length, int location);

   /**
    * @brief   Memory copy function.
    * @details Memory copy function.
    * @param [in]  ptr_to Copy to this address.
    * @param [in]  ptr_from Copy from this address.
    * @param [in]  length The length of the array.
    * @param [in]  loc_to Location of ptr_to.
    * @param [in]  loc_from Location of ptr_from.
    */
   void PargemslrMemcpy( void* ptr_to, const void* ptr_from, size_t length, int loc_to, int loc_from);

   /**
    * @brief   Free the memory.
    * @details Free the memory.
    * @param [in,out] ptr Pointer to the memory, set to NULL on return.
    * @param [in]     location Location of the memory.
    */
   void PargemslrFree( void* ptr, int location); 

#ifdef PARGEMSLR_DEBUG_MEMORY
   /**
    * @brief   Data structure for memory debuging. Not in use in the release version.
    * @details Data structure for memory debuging. Not in use in the release version.
    */
   typedef struct MemoryDebuggerStruct
   {
      /**
       * @brief   Memory actions, 'm' for malloc, 'c' for calloc, 'r' for reallocate, 'f' for free.
       * @details Memory actions, 'm' for malloc, 'c' for calloc, 'r' for reallocate, 'f' for free.
       */
      std::vector<char>                                                 _actions;
      
      /**
       * @brief   Length of the allocated memory.
       * @details Length of the allocated memory.
       */
      std::vector<size_t>                                               _lengths;
      
      /**
       * @brief   Memory address of the memory.
       * @details Memory address of the memory.
       */
      std::vector<void*>                                                _addresses;
      
      /**
       * @brief   The file that operates the memory.
       * @details The file that operates the memory.
       */
      std::vector<std::string>                                          _filenames;
      
      /**
       * @brief   The function that operates the memory.
       * @details The function that operates the memory.
       */
      std::vector<std::string>                                          _functions;
      
      /**
       * @brief   The line that operates the memory.
       * @details The line that operates the memory.
       */
      std::vector<int>                                                  _line;
      
      /**
       * @brief   The hash table that tracking the memory location.
       * @details The hash table that tracking the memory location.
       */
      static std::unordered_map< void*, MemoryDebuggerStruct>           _memory_tracker;
      
      /**
       * @brief   Insert an memory allocation.
       * @details Insert an memory allocation.
       */
      static void InsertMemoryAction( char action, size_t length, const void *address, const char *filename, const char *function, const int line);
      
      /**
       * @brief   Check the memory location.
       * @details Check the memory location.
       */
      static void CheckMemoryLocation( const void *ptr, int location, const char *filename, const char *function, const int line);
      
   }memory_debugger;
   
#endif
   
}

#ifdef PARGEMSLR_DEBUG_MEMORY

#define PARGEMSLR_MALLOC_VOID(ptr, length, location) {\
   (ptr) = pargemslr::PargemslrMalloc( (size_t)(length), location);\
   pargemslr::MemoryDebuggerStruct::InsertMemoryAction('m', (size_t)(length), (const void*)ptr, __FILE__, __func__, __LINE__);\
}

#define PARGEMSLR_MALLOC(ptr, length, location, ...) {\
   (ptr) = (__VA_ARGS__*) pargemslr::PargemslrMalloc( (size_t)(length)*sizeof(__VA_ARGS__), location);\
   pargemslr::MemoryDebuggerStruct::InsertMemoryAction('m', (size_t)(length)*sizeof(__VA_ARGS__), (const void*)ptr, __FILE__, __func__, __LINE__);\
}

#define PARGEMSLR_CALLOC_VOID(ptr, length, location) {\
   (ptr) = pargemslr::PargemslrCalloc( (size_t)(length), location);\
   pargemslr::MemoryDebuggerStruct::InsertMemoryAction('c', (size_t)(length), (const void*)ptr, __FILE__, __func__, __LINE__);\
}

#define PARGEMSLR_CALLOC(ptr, length, location, ...) {\
   (ptr) = (__VA_ARGS__*) pargemslr::PargemslrCalloc( (size_t)length*sizeof(__VA_ARGS__), location);\
   pargemslr::MemoryDebuggerStruct::InsertMemoryAction('c', (size_t)(length)*sizeof(__VA_ARGS__), (const void*)ptr, __FILE__, __func__, __LINE__);\
}

#define PARGEMSLR_REALLOC_VOID(ptr, current_length, new_length, location) {\
   (ptr) = pargemslr::PargemslrRealloc( (void*)(ptr), (size_t)(current_length), (size_t)(new_length), location);\
   pargemslr::MemoryDebuggerStruct::InsertMemoryAction('r', (size_t)(new_length), (const void*)ptr, __FILE__, __func__, __LINE__);\
}

#define PARGEMSLR_REALLOC(ptr, current_length, new_length, location, ...) {\
   (ptr) = (__VA_ARGS__*) pargemslr::PargemslrRealloc( (void*)(ptr), (size_t)(current_length)*sizeof(__VA_ARGS__), (size_t)(new_length)*sizeof(__VA_ARGS__), location);\
   pargemslr::MemoryDebuggerStruct::InsertMemoryAction('r', (size_t)(new_length)*sizeof(__VA_ARGS__), (const void*)ptr, __FILE__, __func__, __LINE__);\
}

#define PARGEMSLR_MEMCPY(ptr_to, ptr_from, length, loc_to, loc_from, ...) {\
   pargemslr::MemoryDebuggerStruct::CheckMemoryLocation( (const void*)(ptr_to), loc_to, __FILE__, __func__, __LINE__);\
   pargemslr::MemoryDebuggerStruct::CheckMemoryLocation( (const void*)(ptr_from), loc_from, __FILE__, __func__, __LINE__);\
   pargemslr::PargemslrMemcpy( (void*)(ptr_to), (void*)(ptr_from), (size_t)(length)*sizeof(__VA_ARGS__), loc_to, loc_from);\
}

#define PARGEMSLR_FREE( ptr, location) {\
   pargemslr::MemoryDebuggerStruct::CheckMemoryLocation( (const void*)ptr, location, __FILE__, __func__, __LINE__);\
   pargemslr::PargemslrFree( (void*)(ptr), location);\
   (ptr) = NULL;\
}

#else

#define PARGEMSLR_MALLOC_VOID(ptr, length, location) {\
   (ptr) = pargemslr::PargemslrMalloc( (size_t)(length), location);\
}

#define PARGEMSLR_MALLOC(ptr, length, location, ...) {\
   (ptr) = (__VA_ARGS__*) pargemslr::PargemslrMalloc( (size_t)(length)*sizeof(__VA_ARGS__), location);\
}

#define PARGEMSLR_CALLOC_VOID(ptr, length, location) {\
   (ptr) = pargemslr::PargemslrCalloc( (size_t)(length), location);\
}

#define PARGEMSLR_CALLOC(ptr, length, location, ...) {\
   (ptr) = (__VA_ARGS__*) pargemslr::PargemslrCalloc( (size_t)(length)*sizeof(__VA_ARGS__), location);\
}

#define PARGEMSLR_REALLOC_VOID(ptr, current_length, new_length, location) {\
   (ptr) = (void*) pargemslr::PargemslrRealloc( (void*)(ptr), (size_t)(current_length), (size_t)(new_length), location);\
}

#define PARGEMSLR_REALLOC(ptr, current_length, new_length, location, ...) {\
   (ptr) = (__VA_ARGS__*) pargemslr::PargemslrRealloc( (void*)(ptr), (size_t)(current_length)*sizeof(__VA_ARGS__), (size_t)(new_length)*sizeof(__VA_ARGS__), location);\
}

#define PARGEMSLR_MEMCPY(ptr_to, ptr_from, length, loc_to, loc_from, ...) {\
   pargemslr::PargemslrMemcpy( (void*)(ptr_to), (void*)(ptr_from), (size_t)(length)*sizeof(__VA_ARGS__), loc_to, loc_from);\
}

#define PARGEMSLR_FREE( ptr, location) {\
   pargemslr::PargemslrFree( (void*)(ptr), location);\
   (ptr) = NULL;\
}

#endif

#define PARGEMSLR_PLACEMENT_NEW(ptr, location, ...) {\
   (ptr) = (__VA_ARGS__*) pargemslr::PargemslrMalloc( (size_t)sizeof(__VA_ARGS__), location);\
   (ptr) = new (ptr) __VA_ARGS__();\
}

#endif
