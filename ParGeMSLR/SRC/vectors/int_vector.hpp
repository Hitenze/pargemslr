#ifndef PARGEMSLR_INT_VECTOR_H
#define PARGEMSLR_INT_VECTOR_H

/**
 * @file int_vector.hpp
 * @brief Integer vector data structure, not a vector_base class.
 */

#include "../utils/utils.hpp"
#include "vector.hpp"
#include <algorithm>
#include <iostream>

#ifdef PARGEMSLR_CUDA
#include "cusparse.h"
#endif

namespace pargemslr
{
   
   /**
    * @brief   The class of sequential integer vector.
    * @details The class of sequential integer vector.
    */
   template <typename T>
   class IntVectorClass: public VectorVirtualClass<T>
   {
   private:
      
      /**
      * @brief   Pointer to the data.
      * @details Pointer to the data.
      */
      T*             _data;
      
      /**
      * @brief   The length of the vector.
      * @details The length of the vector, can be smller than the actual memory size.
      */
      int            _length;
      
      /**
      * @brief   The reserved length in the memory.
      * @details The reserved length in the memory, can be greater than _length.
      */
      int            _maxlength;
      
      /**
      * @brief   Tell if this vector allocates memory.
      * @details Tell if this vector allocates memory. If set to GEMSLR_FALSE, the memory will not be freed when deallocate.
      */
      bool           _hold_data;
      
      /**
      * @brief   The location of data.
      * @details The location of data. Can be GEMSLR_MEMORY_HOST, GEMSLR_MEMORY_DEVICE, GEMSLR_MEMORY_PINNED, or GEMSLR_MEMORY_UNIFIED.
      */
      int            _location;
      
      
   public:
      
      /**
       * @brief   The constructor of IntVectorClass
       * @details The constructor of IntVectorClass. The default memory location is the host memory.
       */
      IntVectorClass();
      
      /**
       * @brief   The copy constructor of IntVectorClass.
       * @details The copy constructor of IntVectorClass.
       */
      IntVectorClass(const IntVectorClass<T> &vec);
      
      /**
       * @brief   The move constructor of IntVectorClass.
       * @details The move constructor of IntVectorClass.
       * @param   [in]        vec The vector.
       */
      IntVectorClass(IntVectorClass<T> &&vec);
      
      /**
       * @brief   The = operator of IntVectorClass.
       * @details The = operator of IntVectorClass.
       * @param   [in]        vec The vector.
       * @return     Return the vector.
       */
      IntVectorClass<T>& operator= (const IntVectorClass<T> &vec);
      
      /**
       * @brief   The = operator of IntVectorClass.
       * @details The = operator of IntVectorClass.
       * @param   [in]        vec The vector.
       * @return     Return the vector.
       */
      IntVectorClass<T>& operator= (IntVectorClass<T> &&vec);
      
      /**
       * @brief   The destructor of IntVectorClass.
       * @details The destructor of IntVectorClass. Simply a call to the free function.
       */
      virtual ~IntVectorClass();
      
      /**
       * @brief   Free the current vector, and malloc memory to initilize the vector.
       * @details Free the current vector, and malloc memory to initilize the vector. The memory location is controlled by _location.
       * @param   [in]        length The length of the vector.
       * @return     Return error message.
       */
      int            Setup(int length);
      
      /**
       * @brief   Free the current vector, and allocate memory to initilize the vector.
       * @details Free the current vector, and allocate memory to initilize the vector. The memory location is controlled by _location.
       * @param   [in]        length The length of the vector.
       * @param   [in]        setzero Call calloc if we need to fill the memory with 0.
       * @return     Return error message.
       */
      int            Setup(int length, bool setzero);
      
      /**
       * @brief   Free the current vector, and allocate memory at give location to initilize the vector.
       * @details Free the current vector, and allocate memory at give location to initilize the vector.
       * @param   [in]        length The length of the vector.
       * @param   [in]        location The location of the vector.
       * @param   [in]        setzero Call calloc if we need to fill the memory with 0.
       * @return     Return error message.
       */
      int            Setup(int length, int location, bool setzero);
      
      /**
      * @brief   Free the current vector, and allocate memory at give location to initilize the vector.
      * @details Free the current vector, and allocate memory at give location to initilize the vector. \n
      *          The actual memory size is set by the value reserve, and the vector length is given by the value length.
      * @param   [in]        length The length of the vector.
      * @param   [in]        reserve The length allocated in the memory, should be no less than length.
      * @param   [in]        location The location of the vector.
      * @param   [in]        setzero Call calloc if we need to fill the memory with 0.
      * @return     Return error message.
      */
      int            Setup(int length, int reserve, int location, bool setzero);
      
      /**
       * @brief   Free the current vector, and points the vector to an address in the memory.
       * @details Free the current vector, and points the vector to an address in the memory. \n
       *          Data is not freed by this vector.
       * @param   [in]        data The address of the data.
       * @param   [in]        length The new length of this vector.
       * @param   [in]        location The location of the data.
       * @return     Return error message.
       */
      int            SetupPtr( T* data, int length, int location);
      
      /**
       * @brief   Free the current vector, and points the vector to an address in the memory.
       * @details Free the current vector, and points the vector to an address in the memory.
       * @param   [in]        data The address of the data.
       * @param   [in]        length The new length of this vector.
       * @param   [in]        location The location of the data.
       * @param   [in]        hold_data Set to true if we need to free data.
       * @return     Return error message.
       */
      int            SetupPtr( T* data, int length, int location, bool hold_data);
      
      /**
       * @brief   Free the current vector, and points the vector to a given IntVectorClass.
       * @details Free the current vector, and points the vector to a given IntVectorClass.
       * @param   [in]        vec The target vector.
       * @param   [in]        length The new length of this vector.
       * @param   [in]        shift The shift in the target vector. this->_data = vec.GetData() + shift.
       * @return     Return error message.
       */
      int            SetupPtr(const IntVectorClass<T> &vec, int length, int shift);
      
      /**
       * @brief   Free the current vector, and points the vector to a given IntVectorClass.
       * @details Free the current vector, and points the vector to a given IntVectorClass.
       * @param   [in]        vec The target vector.
       * @param   [in]        length The new length of this vector.
       * @param   [in]        shift The shift in the target vector. this->_data = vec.GetData() + shift.
       * @param   [in]        hold_data Set to true if this vector should free the data. Typically this function should not be used.
       * @return     Return error message.
       */
      int            SetupPtr(const IntVectorClass<T> &vec, int length, int shift, bool hold_data);
      
      /**
       * @brief   Free the current vector, allocate memory, and copy data to initilize the vector.
       * @details Free the current vector, allocate memory, and copy data to initilize the vector.
       * @param   [in]        data The target memory address.
       * @param   [in]        length The new length of this vector.
       * @param   [in]        loc_from The location of the input data.
       * @param   [in]        loc_to The location of the new vector.
       * @return     Return error message.
       */
      int            Copy(const T *data, int length, int loc_from, int loc_to);
      
      /**
       * @brief   Insert value at the end of the vector, expand the vector when necessary.
       * @details Insert value at the end of the vector, expand the vector when necessary. \n 
       *          Work only when the vector is holding data.
       * @param   [in]   v The value to be inserted.
       * @return     Return error message.
       */
      int            PushBack(T v);
      
      /**
       * @brief   Resize the vector. Re-allocate memory when necessary.
       * @details Resize the vector. Re-allocate memory when necessary.
       * @param   [in]   length The new length of this vector.
       * @param   [in]   keepdata Set to true to keep the data in the current vector.
       * @param   [in]   setzero Set to true to fill the new extra memory with 0 if the vector is expanded.
       * @return     Return error message.
       */
      int            Resize(int length, bool keepdata, bool setzero);
      
      /**
       * @brief   Resize the vector. Re-allocate memory when necessary.
       * @details Resize the vector. Re-allocate memory when necessary.
       * @param   [in]   length The new length of this vector.
       * @param   [in]   reserve The new reserve length in the memory, should be no less than length.
       * @param   [in]   keepdata Set to true to keep the data in the current vector.
       * @param   [in]   setzero Set to true to fill the new extra memory with 0 if the vector is expanded.
       * @return     Return error message.
       */
      int            Resize(int length, int reserve, bool keepdata, bool setzero);
      
      /**
       * @brief   Get the reference of an index in the vector.
       * @brief   Get the reference of an index in the vector.
       * @param   [in]     i The index.
       * @return  The reference of the value on that index.                                                                                                                                                       
       */
      virtual T&     operator[] (int i);
      
      /**
       * @brief   Get the reference of the last index in the vector.
       * @brief   Get the reference of the last index in the vector.
       * @return  The reference of the value on the last index.                                                                                                                                                       
       */
      T&             Back();
      
      /**
       * @brief   Free the current vector.
       * @details Free the current vector.
       * @return     Return error message.
       */
      virtual int    Clear();
      
      /**
       * @brief   Get the data pointer of the vector.
       * @details Get the data pointer of the vector.
       * @return     Return the data pointer.
       */
      virtual T*     GetData() const;

      /**
       * @brief   Get the data location of the vector.
       * @details Get the data location of the vector.
       * @return     Return the data location of the vector.
       */
      virtual int    GetDataLocation() const;
      
      /**
       * @brief   Get the length of the vector.
       * @details Get the length of the vector.
       * @return     Return the length of the vector.
       */
      virtual int    GetLengthLocal() const;
      
      /**
       * @brief   Fill the vector with constant value.
       * @details Fill the vector with constant value.
       * @param   [in]   v The value to be filled.
       * @return     Return error message.
       */
      virtual int    Fill(const T &v);
      
      /**
       * @brief      Generate a unit permutation vector.
       * @details    Generate a unit permutation vector. \n
       *             Should call Setup first to set the length and data of this vector.
       * @return     Return error message.
       */
      int 			   UnitPerm();
      
      /**
       * @brief      Get the max value in the vector.
       * @details    Get the max value in the vector.
       * @return     Return the max value.
       */
      T              Max() const;
      
      /**
       * @brief      Get the index of the max value in the vector.
       * @details    Get the index of the max value in the vector.
       * @return     Return the index of the max value.
       */
      int            MaxIndex() const;
      
       /**
       * @brief      Get the min value in the vector.
       * @details    Get the min value in the vector.
       * @return     Return the min value.
       */
      T              Min() const;
      
      /**
       * @brief      Get the index of the min value in the vector.
       * @details    Get the index of the min value in the vector.
       * @return     Return the index of the min value.
       */
      int            MinIndex() const;
      
      /**
       * @brief      Binary search between [s, e] inside an array. Report the last if there are duplicates.
       * @details    Binary search between [s, e] inside an array. Report the last if there are duplicates. \n
       *             The default option is to get the LAST occurance of a certain value
       *             need to handle duplicates always search to the right ->
       *             example: \n
       *             [0 0 1 1 1 3 3 5 5 6 7 8] if we get an 1, search to the right until we get (1, 3)
       *             we won't get 8 so this would be fine.
       * @param [in]    val The target value.
       * @param [out]   idx If found the value, set to the index of the value. Otherwise the position to insert, or -1 if s > e.
       * @param [in]    descend The array is descend or ascend.
       * @return return -1 if the value isnot found. Otherwise the index of it.
       */
      int            BinarySearch(const T &val, int &idx, bool ascending);
      
      /**
       * @brief      Sort the current vector.
       * @details    Sort the current vector.
       * @param [in] ascending Sort in ascending or descending.
       * @return     Return error message.
       */
      int            Sort(bool ascending);
      
      /**
       * @brief       Get the order such that v[order] is in ascending or descending order.
       * @details     Get the order such that v[order] is in ascending or descending order. \n
       *              Without modifying the current vector.
       * @param [out] order The order vector.
       * @param [in]  ascending Sort in ascending or descending.
       * @param [in]  stable Use stable sort or not.
       * @return      Return error message.
       */
      int            Sort(IntVectorClass<int> &order, bool ascending, bool stable);
      
      /**
       * @brief      Apply permutation to the current vector v := v(perm).
       * @details    Sort the current vector.
       * @param [in] perm The permutation vector.
       * @return     Return error message.
       */
      int            Perm(IntVectorClass<int> &perm);
      
      /**
       * @brief        Print the vector.
       * @details      Print the vector.
       * @param [in]   conditiona First condition.
       * @param [in]   conditionb Secend condition, only print when conditiona == conditionb.
       * @param [in]   width The plot width.
       * @return           Return error message.
       */
      int            Plot( int conditiona, int conditionb, int width);
      
      /**
       * @brief Copy element from a vector to another vector use vector as map. (v_out(map) := v_in)
       * @brief Copy element from a vector to another vector use vector as map. (v_out(map) := v_in) \n
       *        Scatter the local value at i to the map[i] location of another vector, the size of v_in is equal to the size of map, 
       *        but the size of v_out can be larger.
       * @param [in]  v_out The output vector.
       * @param [out] v_in The input vector.
       * @return           Return error message.
       */
      template <typename T1>
      int ScatterRperm(const VectorVirtualClass<T1> &v_in, VectorVirtualClass<T1> &v_out);
      
      /**
       * @brief Copy element from a vector to another vector use this vector as map. (v_out := v_in(map))
       * @brief Copy element from a vector to another vector use vector as map. (v_out := v_in(map)) \n
       *        Gather the value from map[i] locatioin of another vector to i location of this vector, the
       *        size of v_out is equal to the size of map, but the size of v_in can be larger.
       * @param [in]  v_out The output vector.
       * @param [out] v_in The input vector.
       * @return           Return error message.
       */
      template <typename T1>
      int GatherPerm(const VectorVirtualClass<T1> &v_in, VectorVirtualClass<T1> &v_out);
      
      /**
       * @brief      Move the data to another memory location.
       * @details    Move the data to another memory location.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      virtual int MoveData( const int &location);
      
   };
   
   typedef IntVectorClass<int>         vector_int;
   typedef IntVectorClass<long int>    vector_long;
   template<> struct PargemslrIsComplex<IntVectorClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<IntVectorClass<complexd> > : public std::true_type {};
   
}

#endif
