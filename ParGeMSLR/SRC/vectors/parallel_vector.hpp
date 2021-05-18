#ifndef PARGEMSLR_PARALLEL_VECTOR_H
#define PARGEMSLR_PARALLEL_VECTOR_H

/**
 * @file parallel_vector.hpp
 * @brief Parallel vector data structure.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "vector.hpp"
#include "int_vector.hpp"
#include "sequential_vector.hpp"
#include "../matrices/parallel_csr_matrix.hpp"

namespace pargemslr
{
   
   /**
    * @brief   The class of parallel real/complex vector.
    * @details The class of parallel real/complex vector.
    */
   template <typename T>
   class ParallelVectorClass: public VectorClass<T>
   {
   private:
      
      /**
       * @brief   Data vector.
       * @details Data vector.
       */
      SequentialVectorClass<T>   _data_vec;
      
      /**
       * @brief   The start index of the vector.
       * @details The start index of the vector.
       */
      long int                   _n_start;
      
      /**
       * @brief   The global length of the vector.
       * @details The global length of the vector.
       */
      long int                   _n_global;
      
   public:
      
      /**
       * @brief   The constructor of ParallelVectorClass.
       * @details The constructor of ParallelVectorClass. The default memory location is the host memory.
       */
      ParallelVectorClass();
      
      /**
       * @brief   The copy constructor of ParallelVectorClass.
       * @details The copy constructor of ParallelVectorClass.
       * @param   [in]        vec The target vector.
       */
      ParallelVectorClass(const ParallelVectorClass<T> &vec);
      
      /**
       * @brief   The copy constructor of ParallelVectorClass.
       * @details The copy constructor of ParallelVectorClass.
       * @param   [in]        vec The target vector.
       */
      ParallelVectorClass( ParallelVectorClass<T> &&vec);
      
      /**
       * @brief   The operator = of ParallelVectorClass.
       * @details The operator = of ParallelVectorClass.
       * @param   [in]        vec The target vector.
       * @return     Return the vector.
       */
      ParallelVectorClass<T>& operator= (const ParallelVectorClass<T> &vec);
      
      /**
       * @brief   The operator = of ParallelVectorClass.
       * @details The operator = of ParallelVectorClass.
       * @param   [in]        vec The target vector.
       * @return     Return the vector.
       */
      ParallelVectorClass<T>& operator= ( ParallelVectorClass<T> &&vec);
      
      /**
       * @brief   The destructor of ParallelVectorClass.
       * @details The destructor of ParallelVectorClass. Simply a call to the free function.
       */
      virtual ~ParallelVectorClass();
      
      /**
       * @brief   Free the current vector, and malloc memory to initilize the vector on comm.
       * @details Free the current vector, and malloc memory to initilize the vector on comm.
       * @param   [in]        n_local The local length of the vector.
       * @param   [in]        parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int n_local, parallel_log &parlog);
      
      /**
       * @brief   Free the current vector, and allocate memory to initilize the vector.
       * @details Free the current vector, and allocate memory to initilize the vector.
       * @param   [in]        n_local The local length of the vector.
       * @param   [in]        setzero Call calloc if we need to fill the memory with 0.
       * @param   [in]        parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int n_local, bool setzero, parallel_log &parlog);
      
      /**
       * @brief   Free the current vector, and allocate memory at give location to initilize the vector.
       * @details Free the current vector, and allocate memory at give location to initilize the vector.
       * @param   [in]        n_local The local length of the vector.
       * @param   [in]        location The location of the vector.
       * @param   [in]        setzero Call calloc if we need to fill the memory with 0.
       * @param   [in]        parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int n_local, int location, bool setzero, parallel_log &parlog);
      
      /**
       * @brief   Free the current vector, and allocate memory at give location to initilize the vector.
       * @details Free the current vector, and allocate memory at give location to initilize the vector. \n
       *          The actual memory size is set by the value reserve, and the vector length is given by the value length.
       * @param   [in]        n_local The local length of the vector.
       * @param   [in]        reserve The length allocated in the memory, should be no less than length.
       * @param   [in]        location The location of the vector.
       * @param   [in]        setzero Call calloc if we need to fill the memory with 0.
       * @param   [in]        parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int n_local, int reserve, int location, bool setzero, parallel_log &parlog);
      
      /**
       * @brief   Free the current vector, and malloc memory to initilize the vector on comm.
       * @details Free the current vector, and malloc memory to initilize the vector on comm.
       * @param   [in]        n_local The local length of the vector.
       * @param   [in]        n_start The start index of the vector.
       * @param   [in]        n_global The global length of the vector.
       * @param   [in]        parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int n_local, long int n_start, long int n_global, parallel_log &parlog);
      
      /**
       * @brief   Free the current vector, and allocate memory to initilize the vector.
       * @details Free the current vector, and allocate memory to initilize the vector.
       * @param   [in]        n_local The local length of the vector.
       * @param   [in]        n_start The start index of the vector.
       * @param   [in]        n_global The global length of the vector.
       * @param   [in]        setzero Call calloc if we need to fill the memory with 0.
       * @param   [in]        parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int n_local, long int n_start, long int n_global, bool setzero, parallel_log &parlog);
      
      /**
       * @brief   Free the current vector, and allocate memory at give location to initilize the vector.
       * @details Free the current vector, and allocate memory at give location to initilize the vector.
       * @param   [in]        n_local The local length of the vector.
       * @param   [in]        n_start The start index of the vector.
       * @param   [in]        n_global The global length of the vector.
       * @param   [in]        location The location of the vector.
       * @param   [in]        setzero Call calloc if we need to fill the memory with 0.
       * @param   [in]        parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int n_local, long int n_start, long int n_global, int location, bool setzero, parallel_log &parlog);
      
      /**
       * @brief   Free the current vector, and allocate memory at give location to initilize the vector.
       * @details Free the current vector, and allocate memory at give location to initilize the vector. \n
       *          The actual memory size is set by the value reserve, and the vector length is given by the value length.
       * @param   [in]        n_local The local length of the vector.
       * @param   [in]        n_start The start index of the vector.
       * @param   [in]        n_global The global length of the vector.
       * @param   [in]        reserve The length allocated in the memory, should be no less than length.
       * @param   [in]        location The location of the vector.
       * @param   [in]        setzero Call calloc if we need to fill the memory with 0.
       * @param   [in]        parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int n_local, long int n_start, long int n_global, int reserve, int location, bool setzero, parallel_log &parlog);
      
      /**
       * @brief   Setup the length information of a vector pointer with the information of a ParallelVectorClass.
       * @details Setup the length information of a vector pointer with the information of a ParallelVectorClass. \n
       *          Need to call UpdatePtr function to set the data before use.
       * @param   [in]        x The ParallelVectorClass.
       * @return     Return error message.
       */
      int            SetupPtrStr( ParallelVectorClass<T> &x);
      
      /**
       * @brief   Setup the length information of a vector pointer with the row information of a ParallelCsrMatrixClass.
       * @details Setup the length information of a vector pointer with the row information of a ParallelCsrMatrixClass. \n
       *          Need to call UpdatePtr function to set the data before use.
       * @param   [in]        A The ParallelCsrMatrixClass.
       * @return     Return error message.
       */
      int            SetupPtrStr( ParallelCsrMatrixClass<T> &A);
      
      /**
       * @brief   Update the data with new memory address. Need to call SetupPtrStr first.
       * @details Update the data with new memory address. Need to call SetupPtrStr first.
       * @param   [in]        data The address of the data.
       * @param   [in]        location The location of the data.
       * @return     Return error message.
       */
      int            UpdatePtr( void* data, int location);
      
      /**
       * @brief   Get the reference of an index in the local vector.
       * @brief   Get the reference of an index in the local vector. vec[0] is the first local value.
       * @param   [in]     i The local index.
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

      /**
       * @brief   Get the reference of the data vector.
       * @details Get the reference of the data vector.
       * @return     Return the data vector.
       */
      SequentialVectorClass<T>&  GetDataVector();

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
       * @brief   Print the vector.
       * @details Print the vector.
       * @param [in]   conditiona First condition.
       * @param [in]   conditionb Secend condition, only print when conditiona == conditionb.
       * @param [in]   width The plot width.
       * @return           Return error message.
       */
      int            Plot( int conditiona, int conditionb, int width);
      
      /**
       * @brief      Write file to disk.
       * @details    Write file to disk.
       * @param [in]  datafilename The filename of the file holding the data.
       * @return      Return error message.
       */
      int            WriteToDisk( const char *datafilename);
      
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
       * @param [in] beta The beta value.
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
      virtual int    Axpy( const T &alpha, const VectorClass<T> &x, VectorClass<T> &y);
      
      /**
       * @brief       Compute z = alpha * x + beta * y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @details     Compute z = alpha * x + beta * y, where z is this vector. Currenlty we don't support x == y, x == z or y == z.
       * @param [in]  alpha The alpha value.
       * @param [in]  x The first vector.
       * @param [in]  beta The beta value.
       * @param [in]  y The second vector.
       * @return      Return error message.
       */
      virtual int    Axpy( const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y);
      
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
       * @brief       Compute the dot product.
       * @details     Compute the dot product.
       * @param [in]  y The vector.
       * @param [out] t The result.
       * @return      Return error message.
       */
      virtual int    Dot( const VectorClass<T> &y, T &t) const;
      
      /**
       * @brief      Tell if this is a prallel vector.
       * @details    Tell if this is a prallel vector.
       * @return     Return true is this is a parallel vector.
       */
      virtual bool   IsParallel() const;

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
      int            SetCusparseVec(cusparseDnVecDescr_t cusparse_vec);
#endif
#endif

      /**
       * @brief      Move the data to another memory location.
       * @details    Move the data to another memory location.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      virtual int    MoveData( const int &location);
      
      /**
       * @brief   Read vector on the host memory, matrix market format. Note that the vector should be setup already (have the global and local infomation).
       * @details Read vector on the host memory, matrix market format. Note that the vector should be setup already (have the global and local infomation).
       * @param   [in]    vecfile The file.
       * @param   [in]    idxin The index base of the input file. 0-based or 1-based.
       * @return     Return error message.
       */
      int            ReadFromSingleMMFile(const char *vecfile, int idxin);
      
   };
   
   typedef ParallelVectorClass<float>     vector_par_float;
   typedef ParallelVectorClass<double>    vector_par_double;
   typedef ParallelVectorClass<complexs>  vector_par_complexs;
   typedef ParallelVectorClass<complexd>  vector_par_complexd;
   template<> struct PargemslrIsComplex<ParallelVectorClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<ParallelVectorClass<complexd> > : public std::true_type {};
   template<> struct PargemslrIsParallel<ParallelVectorClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsParallel<ParallelVectorClass<complexd> > : public std::true_type {};
   
}

#endif
