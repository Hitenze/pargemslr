#ifndef PARGEMSLR_VECTOR_H
#define PARGEMSLR_VECTOR_H

/**
 * @file  vector.hpp
 * @brief The virtual vector classes.
 */

#include "../utils/utils.hpp"
#include "../utils/parallel.hpp"
#include "../utils/memory.hpp"

namespace pargemslr
{
   
   /**
    * @brief   The virtual class of all vector classes.
    * @details The virtual class of all vector classes.
    */
   template <typename T>
   class VectorVirtualClass: public parallel_log
   {
   public:
      
      /**
       * @brief   The constructor of VectorVirtualClass.
       * @details The constructor of VectorVirtualClass.
       */
      VectorVirtualClass();
      
      /**
       * @brief   The copy constructor of VectorVirtualClass.
       * @details The copy constructor of VectorVirtualClass.
       */
      VectorVirtualClass(const VectorVirtualClass<T> &vec);
      
      /**
       * @brief   The move constructor of VectorVirtualClass.
       * @details The move constructor of VectorVirtualClass.
       */
      VectorVirtualClass(VectorVirtualClass<T> &&vec);
      
      /**
       * @brief   Free the current vector.
       * @details Free the current vector.
       * @return  Return error message.
       */
      virtual int Clear();
      
      /**
       * @brief   The destructor of VectorVirtualClass.
       * @details The destructor of VectorVirtualClass.
       */
      virtual ~VectorVirtualClass();
      
      /**
       * @brief   Get the reference of an index in the vector.
       * @brief   Get the reference of an index in the vector.
       * @param   [in] i The index.
       * @return  The  reference of the value on that index.                                                                                                                                                       
       */
      virtual T&     operator[] (int i) = 0;
      
      /**
       * @brief      Get the data pointer of the vector.
       * @details    Get the data pointer of the vector.
       * @return     Return the data pointer.
       */
      virtual T* GetData() const = 0;
      
      /**
       * @brief      Get the data location of the vector.
       * @details    Get the data location of the vector.
       * @return     Return the data location of the vector.
       */
      virtual int GetDataLocation() const = 0;
      
      /**
       * @brief      Get the local length of the vector.
       * @details    Get the local length of the vector.
       * @return     Return the length of the vector.
       */
      virtual int GetLengthLocal() const = 0;
      
      /**
       * @brief   Fill the vector with constant value.
       * @details Fill the vector with constant value.
       * @param   [in]   v The value to be filled.
       * @return         Return error message.
       */
      virtual int Fill(const T &v) = 0;
      
      /**
       * @brief      Move the data to another memory location.
       * @details    Move the data to another memory location.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      virtual int MoveData( const int &location) = 0;
      
      /**
       * @brief      Tell if this is a prallel vector.
       * @details    Tell if this is a prallel vector.
       * @return     Return true is this is a parallel vector.
       */
      virtual bool IsParallel() const;
      
      /**
       * @brief      Get the data precision of the vector.
       * @details    Get the data precision of the vector.
       * @return     Return the precision in PrecisionEnum.
       */
      PrecisionEnum GetPrecision() const;
      
   };
   
   template<> struct PargemslrIsComplex<VectorVirtualClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<VectorVirtualClass<complexd> > : public std::true_type {};
   
   /**
    * @brief   The virtual class of real/complex vector class.
    * @details The virtual class of real/complex vector class.
    */
   template <typename T>
   class VectorClass: public VectorVirtualClass<T>
   {
   public:
   
      /**
       * @brief   The constructor of VectorClass.
       * @details The constructor of VectorClass.
       */
      VectorClass();
      
      /**
       * @brief   The copy constructor of VectorClass.
       * @details The copy constructor of VectorClass.
       */
      VectorClass(const VectorClass<T> &vec);
      
      /**
       * @brief   The move constructor of VectorClass.
       * @details The move constructor of VectorClass.
       */
      VectorClass(VectorClass<T> &&vec);
      
      /**
       * @brief   Free the current vector.
       * @details Free the current vector.
       * @return     Return error message.
       */
      virtual int Clear();
      
      /**
       * @brief   The destructor of VectorClass.
       * @details The destructor of VectorClass.
       */
      virtual ~VectorClass();
      
      
      /* Operations - - - - - - - */
      
      /**
       * @brief   Get the reference of an index in the vector.
       * @brief   Get the reference of an index in the vector.
       * @param   [in]     i The index.
       * @return  The reference of the value on that index.                                                                                                                                                       
       */
      virtual T&     operator[] (int i) = 0;
      
      /**
       * @brief   Get the data pointer of the vector.
       * @details Get the data pointer of the vector.
       * @return     Return the data pointer.
       */
      virtual T* GetData() const = 0;
      
      /**
       * @brief   Get the data location of the vector.
       * @details Get the data location of the vector.
       * @return     Return the data location of the vector.
       */
      virtual int GetDataLocation() const = 0;
      
      /**
       * @brief   Get the local length of the vector.
       * @details Get the local length of the vector.
       * @return     Return the length of the vector.
       */
      virtual int GetLengthLocal() const = 0;
      
      /**
       * @brief   Get the global length of the vector.
       * @details Get the global length of the vector.
       * @return     Return the length of the vector.
       */
      virtual pargemslr_long GetLengthGlobal() const = 0;
      
      /**
       * @brief   Get the global start index of the vector.
       * @details Get the global start index of the vector.
       * @return     Return the global start index of the vector.
       */
      virtual pargemslr_long GetStartGlobal() const = 0;
      
      /**
       * @brief   Fill the vector with constant value.
       * @details Fill the vector with constant value.
       * @param   [in]   v The value to be filled.
       * @return     Return error message.
       */
      virtual int Fill(const T &v) = 0;
      
      /**
       * @brief      Fill the vector with random value.
       * @details    Fill the vector with random value.
       * @return     Return error message.
       */
      virtual int Rand() = 0;
      
      /**
       * @brief      Scale the vector by x = alpha*x, where x is this vector.
       * @details    Scale the vector by x = alpha*x, where x is this vector.
       * @param [in] alpha The scale.
       * @return     Return error message.
       */
      virtual int Scale( const T &alpha) = 0;
      
      /**
       * @brief       Compute y = alpha * x + y, where y is this vector. Currenlty we don't support x == y.
       * @details     Compute y = alpha * x + y, where y is this vector. Currenlty we don't support x == y.
       * @param [in]  alpha The alpha value.
       * @param [out] x The vector.
       * @return      Return error message.
       */
      virtual int Axpy( const T &alpha, const VectorClass<T> &x) = 0;
      
      /**
       * @brief       Compute z = alpha * x + y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @details     Compute z = alpha * x + y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @param [in]  alpha The alpha value.
       * @param [in]  x The first vector.
       * @param [in]  y The second vector.
       * @return      Return error message.
       */
      virtual int Axpy( const T &alpha, const VectorClass<T> &x, VectorClass<T> &y) = 0;
      
      /**
       * @brief       Compute z = alpha * x + beta * y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @details     Compute z = alpha * x + beta * y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @param [in]  alpha The alpha value.
       * @param [in]  x The first vector.
       * @param [in]  beta The beta value.
       * @param [in]  y The second vector.
       * @return      Return error message.
       */
      virtual int Axpy( const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y) = 0;
      
      /**
       * @brief       Compute the 2-norm of a vector, result is type float.
       * @details     Compute the 2-norm of a vector, result is type float.
       * @param [out] norm The 2-norm.
       * @return      Return error message.
       */
      virtual int Norm2( float &norm) const = 0;
      
      /**
       * @brief       Compute the 2-norm of a vector, result is type double.
       * @details     Compute the 2-norm of a vector, result is type double.
       * @param [out] norm The 2-norm.
       * @return      Return error message.
       */
      virtual int Norm2( double &norm) const = 0;
      
      /**
       * @brief       Compute the inf-norm of a vector, result is type float.
       * @details     Compute the inf-norm of a vector, result is type float.
       * @param [out] norm The inf-norm.
       * @return      Return error message.
       */
      virtual int NormInf( float &norm) = 0;
      
      /**
       * @brief       Compute the inf-norm of a vector, result is type double.
       * @details     Compute the inf-norm of a vector, result is type double.
       * @param [out] norm The inf-norm.
       * @return      Return error message.
       */
      virtual int NormInf( double &norm) = 0;
      
      /**
       * @brief       Compute the dot product.
       * @details     Compute the dot product.
       * @param [in]  y The vector.
       * @param [out] t The result.
       * @return      Return error message.
       */
      virtual int Dot( const VectorClass<T> &y, T &t) const = 0;

#ifdef PARGEMSLR_CUDA 
#if (PARGEMSLR_CUDA_VERSION == 11)
      /**
       * @brief   For cusparse general spmv.
       * @details For cusparse general spmv.
       */
      virtual cusparseDnVecDescr_t GetCusparseVec() const = 0;
#endif
#endif

      /**
       * @brief      Move the data to another memory location.
       * @details    Move the data to another memory location.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      virtual int MoveData( const int &location) = 0;
      
   };
   
   typedef VectorClass<float>     vector_base_float;
   typedef VectorClass<double>    vector_base_double;
   typedef VectorClass<complexs>  vector_base_complexs;
   typedef VectorClass<complexd>  vector_base_complexd;
   template<> struct PargemslrIsComplex<VectorClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<VectorClass<complexd> > : public std::true_type {};
   
}

#endif
