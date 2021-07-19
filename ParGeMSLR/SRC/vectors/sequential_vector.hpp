#ifndef PARGEMSLR_SEQUENTIAL_VECTOR_H
#define PARGEMSLR_SEQUENTIAL_VECTOR_H

/**
 * @file sequential_vector.hpp
 * @brief Sequential vector data structure.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../utils/structs.hpp"
#include "vector.hpp"
#include "int_vector.hpp"

#ifdef PARGEMSLR_CUDA
#include "cusparse.h"
#endif

namespace pargemslr
{
   template <typename T> class CsrMatrixClass;
   
   /**
    * @brief   The class of sequential real/complex vector.
    * @details The class of sequential real/complex vector.
    */
   template <typename T>
   class SequentialVectorClass: public VectorClass<T>
   {
   private:
      
      /* variables */
#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
      /**
      * @brief   For cusparse general spmv.
      * @details For cusparse general spmv.
      */
      cusparseDnVecDescr_t          _cusparse_vec;
#endif
#endif
      
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
       * @details Tell if this vector allocates memory. If set to fasle, the memory will not be freed when deallocate.
       */
      bool           _hold_data;
      
      /**
       * @brief   The location of data.
       * @details The location of data. See MemoryLocationEnum.
       */
      int            _location;
      
   public:
      
      /**
       * @brief   The constructor of SequentialVectorClass.
       * @details The constructor of SequentialVectorClass. The default memory location is the host memory.
       */
      SequentialVectorClass();
      
      /**
       * @brief   The copy constructor of SequentialVectorClass.
       * @details The copy constructor of SequentialVectorClass.
       * @param   [in]        vec The other vector.
       */
      SequentialVectorClass(const SequentialVectorClass<T> &vec);
      
      /**
       * @brief   The move constructor of SequentialVectorClass.
       * @details The move constructor of SequentialVectorClass.
       * @param   [in]        vec The other vector.
       */
      SequentialVectorClass(SequentialVectorClass<T>&& vec);
      
      /**
       * @brief   The = operator of SequentialVectorClass.
       * @details The = operator of SequentialVectorClass.
       * @param   [in]        vec The other vector.
       * @return     Return the vector.
       */
      SequentialVectorClass<T>& operator= (const SequentialVectorClass<T> &vec);
      
      /**
       * @brief   The = operator of SequentialVectorClass.
       * @details The = operator of SequentialVectorClass.
       * @param   [in]        vec The other vector.
       * @return     Return the vector.
       */
      SequentialVectorClass<T>& operator= (SequentialVectorClass<T>&& vec);
      
      /**
       * @brief   The destructor of SequentialVectorClass.
       * @details The destructor of SequentialVectorClass. Simply a call to the free function.
       */
      virtual ~SequentialVectorClass();
      
      /**
       * @brief   Free the current vector, and malloc memory to initilize the vector.
       * @details Free the current vector, and malloc memory to initilize the vector.
       * @param   [in]  length The length of the vector.
       * @return     Return error message.
       */
      int            Setup(int length);
      
      /**
       * @brief   Free the current vector, and allocate memory to initilize the vector.
       * @details Free the current vector, and allocate memory to initilize the vector.
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
       * @brief   Setup the length information of a vector pointer to be same as another vector.
       * @details Setup the length information of a vector pointer to be same as another vector. \n
       *          This will only set the length, need to call UpdatePtr function to set data.
       * @param   [in]        x The SequentialVectorClass.
       * @return     Return error message.
       */
      int            SetupPtrStr( SequentialVectorClass<T> &x);
      
      /**
       * @brief   Setup the length information of a vector pointer to be same as the number of rows in a CSR matrix.
       * @details Setup the length information of a vector pointer to be same as the number of rows in a CSR matrix. \n
       *          This will only set the length, need to call UpdatePtr function to set data.
       * @param   [in]        A The CsrMatrixClass.
       * @return     Return error message.
       */
      int            SetupPtrStr( CsrMatrixClass<T> &A);
      
      /**
       * @brief   Setup the length information of a vector pointer.
       * @details Setup the length information of a vector pointer. \n
       *          This will only set the length, need to call UpdatePtr function to set data.
       * @param   [in]        length The new length.
       * @return     Return error message.
       */
      int            SetupPtrStr( int length);
      
      /**
       * @brief   Update the Ptr with new memory address, keep the current length information. Need to call SetupPtrStr first.
       * @details Update the Ptr with new memory address, keep the current length information. Need to call SetupPtrStr first.
       * @param   [in]        data The address of the data.
       * @param   [in]        location The location of the data.
       * @return     Return error message.
       */
      int            UpdatePtr( void* data, int location);
      
      /**
       * @brief   Free the current vector, and points the vector to an address in the memory.
       * @details Free the current vector, and points the vector to an address in the memory. The memory will not be freed by this vector.
       * @param   [in]        data The address of the data.
       * @param   [in]        length The length of the vector.
       * @param   [in]        location The location of the data.
       * @return     Return error message.
       */
      int            SetupPtr( void* data, int length, int location);
      
      /**
       * @brief   Free the current vector, and points the vector to an address in the memory.
       * @details Free the current vector, and points the vector to an address in the memory.
       * @param   [in]        data The address of the data.
       * @param   [in]        length The length of the vector.
       * @param   [in]        location The location of the data.
       * @param   [in]        hold_data Set to true if this vector should free the data.
       * @return              Return error message.
       */
      int            SetupPtr( void* data, int length, int location, bool hold_data);
      
      /**
       * @brief   Free the current vector, and points the vector to a given VectorClass.
       * @details Free the current vector, and points the vector to a given VectorClass. The memory will not be freed by this vector.
       * @param   [in]        vec The target vector.
       * @param   [in]        length The new length of this vector.
       * @param   [in]        shift The shift in the target vector. this->_data = vec.GetData() + shift.
       * @return     Return error message.
       */
      int            SetupPtr(const VectorClass<T> &vec, int length, int shift);
      
      /**
       * @brief   Free the current vector, and points the vector to a given VectorClass.
       * @details Free the current vector, and points the vector to a given VectorClass.
       * @param   [in]        vec The target vector.
       * @param   [in]        length The new length of this vector.
       * @param   [in]        shift The shift in the target vector. this->_data = vec.GetData() + shift.
       * @param   [in]        hold_data Set to true if this vector should free the data. Typically this function should not be used.
       * @return     Return error message.
       */
      int            SetupPtr(const VectorClass<T> &vec, int length, int shift, bool hold_data);
      
      /**
       * @brief   Free the current vector, allocate memory, and copy data to initilize this vector.
       * @details Free the current vector, allocate memory, and copy data to initilize this vector.
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

#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
      /**
       * @brief   For cusparse general spmv.
       * @details For cusparse general spmv.
       */
      virtual cusparseDnVecDescr_t  GetCusparseVec() const;
      
      /**
       * @brief   Set the cusparse vector for cusparse general spmv. Will free the current one if already exists.
       * @details Set the cusparse vector for cusparse general spmv. Will free the current one if already exists.
       * @param [in]  cusparse_vec The cusparseDnVecDescr_t.
       */
      int     SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
#endif
#endif

      /**
       * @brief   Get the data location of the vector.
       * @details Get the data location of the vector.
       * @return     Return the data location of the vector.
       */
      virtual int    GetDataLocation() const;
      
      /**
       * @brief   Get the local length of the vector.
       * @details Get the local length of the vector.
       * @return     Return the local length of the vector.
       */
      virtual int    GetLengthLocal() const;
      
      /**
       * @brief   Get the global length of the vector.
       * @details Get the global length of the vector.
       * @return     Return the global length of the vector.
       */
      virtual long int GetLengthGlobal() const;
      
      /**
       * @brief   Get the global start index of the vector.
       * @details Get the global start index of the vector.
       * @return     Return the global start index of the vector.
       */
      virtual long int GetStartGlobal() const;
      
      /**
       * @brief   Check if the vector holding its own data.
       * @details Check if the vector holding its own data.
       * @return     Return true if vector holding data.
       */
      bool           IsHoldingData() const;
      
      /**
       * @brief   Set if the vector holding its own data.
       * @details Set if the vector holding its own data.
       * @return     Return error message.
       */
      int            SetHoldingData(bool hold_data);
      
      /**
       * @brief   Fill the vector with constant value.
       * @details Fill the vector with constant value.
       * @param   [in]   v The value to be filled.
       * @return     Return error message.
       */
      virtual int    Fill(const T &v);
      
      /**
       * @brief      Fill the vector with random value.
       * @details    Fill the vector with random value.
       * @return     Return error message.
       */
      virtual int    Rand();
      
      /**
       * @brief      Get the max value in the vector.
       * @details    Get the max value in the vector. \n
       *             For real vectors only.
       * @return     Return the max value.
       */
      T              Max() const;
      
      /**
       * @brief      Get the index of the max value in the vector.
       * @details    Get the index of the max value in the vector. \n
       *             For real vectors only.
       * @return     Return the index of the max value.
       */
      int            MaxIndex() const;
      
       /**
       * @brief      Get the min value in the vector.
       * @details    Get the min value in the vector. \n
       *             For real vectors only.
       * @return     Return the min value.
       */
      T              Min() const;
      
      /**
       * @brief      Get the index of the min value in the vector.
       * @details    Get the index of the min value in the vector. \n
       *             For real vectors only.
       * @return     Return the index of the min value.
       */
      int            MinIndex() const;
      
      /**
       * @brief         Binary search between [s, e] inside an array. Report the last if there are duplicates.
       * @details       Binary search between [s, e] inside an array. Report the last if there are duplicates. \n
       *                The default option is to get the LAST occurance of a certain value
       *                need to handle duplicates always search to the right ->
       *                example: \n
       *                [0 0 1 1 1 3 3 5 5 6 7 8] if we get an 1, search to the right until we get (1, 3)
       *                we won't get 8 so this would be fine.
       * @param [in]    val The target value.
       * @param [out]   idx If found the value, set to the index of the value. Otherwise set to the position to insert, or -1 if s > e.
       * @param [in]    descend The array is descend or ascend.
       * @return        Return -1 if the value isnot found. Otherwise the index of it.
       */
      int            BinarySearch(const T &val, int &idx, bool ascending);
      
      /**
       * @brief       Sort the current vector.
       * @details     Sort the current vector.\n
       *             For real vectors only.
       * @param [in]  ascending Sort in ascending or descending.
       * @return      Return error message.
       */
      int            Sort(bool ascending);
      
      /**
       * @brief      Get the order such that v[order] is in ascending or descending order.
       * @details    Get the order such that v[order] is in ascending or descending order. \n
       *             Without modifying the current vector. \n
       *             For real vectors only.
       * @param [out] order The order vector.
       * @param [in]  ascending Sort in ascending or descending.
       * @param [in]  stable Use stable sort or not.
       * @return      Return error message.
       */
      int            Sort(IntVectorClass<int> &order, bool ascending, bool stable);
      
      /**
       * @brief      Apply permutation to the current vector v := v(perm).
       * @details    Apply permutation to the current vector v := v(perm).
       * @param [in] perm The permutation vector.
       * @return     Return error message.
       */
      int            Perm(IntVectorClass<int> &perm);
      
      /**
       * @brief       Print the vector.
       * @details     Print the vector.
       * @param [in]  conditiona First condition.
       * @param [in]  conditionb Secend condition, only plot when conditiona == conditionb.
       * @param [in]  width The plot width.
       * @return      Return error message.
       */
      int            Plot( int conditiona, int conditionb, int width);
      
      /**
       * @brief      Print the absolute value of this vector using GnuPlot.
       * @details    Print the absolute value of this vector using GnuPlot.
       * @param [in]  datafilename The filename of the temp file holding the data.
       * @param [in]  conditiona First condition.
       * @param [in]  conditionb Secend condition, only plot when conditiona == conditionb.
       * @param [in]  logx Set to true to plot in log in x.
       * @param [in]  logy Set to true to plot in log in y.
       * @param [in]  pttype The dot type in GnuPlot.
       * @return      Return error message.
       */
      int            PlotAbsGnuPlot( const char *datafilename, int conditiona, int conditionb, bool logx, bool logy, int pttype = 0);
      
      /**
       * @brief      Write file to disk.
       * @details    Write file to disk.
       * @param [in]  datafilename The filename of the file holding the data.
       * @return      Return error message.
       */
      int            WriteToDisk( const char *datafilename);
      
      /**
       * @brief      Scale the vector by x = alpha*x, where x is this vector.
       * @details    Scale the vector by x = alpha*x, where x is this vector.
       * @param [in] alpha The scale.
       * @return     Return error message.
       */
      virtual int    Scale( const T &alpha);
      
      /**
       * @brief      Compute y = alpha * x + y, where y is this vector. Currenlty we don't support x == y.
       * @details    Compute y = alpha * x + y, where y is this vector. Currenlty we don't support x == y.
       * @param [in] alpha The alpha value.
       * @param [in] x The vector.
       * @return     Return error message.
       */
      virtual int    Axpy( const T &alpha, const VectorClass<T> &x);
      
      /**
       * @brief       Compute z = alpha * x + y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @details     Compute z = alpha * x + y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @param [in]  alpha The alpha value.
       * @param [in]  x The first vector.
       * @param [in]  y The second vector.
       * @return      Return error message.
       */
      virtual int    Axpy( const T &alpha, const VectorClass<T> &x, VectorClass<T> &z);
      
      /**
       * @brief       Compute z = alpha * x + beta * y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @details     Compute z = alpha * x + beta * y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @param [in]  alpha The alpha value.
       * @param [in]  x The first vector.
       * @param [in]  beta The beta value.
       * @param [in]  y The second vector.
       * @return      Return error message.
       */
      virtual int    Axpy( const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &z);
      
      /**
       * @brief      Compute the 2-norm of a vector, result is type float.
       * @details    Compute the 2-norm of a vector, result is type float.
       * @param [out] norm The 2-norm.
       * @return     Return error message.
       */
      virtual int    Norm2( float &norm) const;
      
      /**
       * @brief      Compute the 2-norm of a vector, result is type double.
       * @details    Compute the 2-norm of a vector, result is type double.
       * @param [out] norm The 2-norm.
       * @return     Return error message.
       */
      virtual int    Norm2( double &norm) const;
            
      /**
       * @brief      Compute the inf-norm of a vector, result is type float.
       * @details    Compute the inf-norm of a vector, result is type float.
       * @param [out] norm The inf-norm.
       * @return     Return error message.
       */
      virtual int    NormInf( float &norm);
      
      /**
       * @brief      Compute the inf-norm of a vector, result is type double.
       * @details    Compute the inf-norm of a vector, result is type double.
       * @param [out] norm The inf-norm.
       * @return     Return error message.
       */
      virtual int    NormInf( double &norm);
      
      /**
       * @brief      Compute the dot product.
       * @details    Compute the dot product.
       * @param [in]  y The vector.
       * @param [out] t The result.
       * @return      Return error message.
       */
      virtual int Dot( const VectorClass<T> &y, T &t) const;
      
      /**
       * @brief      Move the data to another memory location.
       * @details    Move the data to another memory location.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      virtual int MoveData( const int &location);
      
      /**
       * @brief   Read vector on the host memory, matrix market format.
       * @details Read vector on the host memory, matrix market format.
       * @param   [in]    vecfile The file.
       * @param   [in]    idxin The index base of the input file. 0-based or 1-based.
       * @return     Return error message.
       */
      int            ReadFromMMFile(const char *vecfile, int idxin);
      
   };
   
   typedef SequentialVectorClass<float>     vector_seq_float;
   typedef SequentialVectorClass<double>    vector_seq_double;
   typedef SequentialVectorClass<complexs>  vector_seq_complexs;
   typedef SequentialVectorClass<complexd>  vector_seq_complexd;
   template<> struct PargemslrIsComplex<SequentialVectorClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<SequentialVectorClass<complexd> > : public std::true_type {};
   
}

#endif
