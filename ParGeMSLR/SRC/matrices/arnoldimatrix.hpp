#ifndef PARGEMSLR_ARNOLDIMATRIX_H
#define PARGEMSLR_ARNOLDIMATRIX_H

/**
 * @file  matrix.hpp
 * @brief The virtual matrix classes.
 */

#include "../utils/utils.hpp"
#include "../utils/parallel.hpp"
#include "../utils/memory.hpp"
#include "../vectors/vector.hpp"
#include "../vectors/parallel_vector.hpp"

namespace pargemslr
{
   
   /**
    * @brief   The virtual class of all matrix classes that could be used by the Arnoldi functions.
    * @details The virtual class of all matrix classes that could be used by the Arnoldi functions.
    */
   template <class VectorType, typename DataType>
   class ArnoldiMatrixClass
   {
   public:
      
      /**
       * @brief   The constructor of ArnoldiMatrixClass.
       * @details The constructor of ArnoldiMatrixClass.
       */
      ArnoldiMatrixClass();
      
      /**
       * @brief   The copy constructor of ArnoldiMatrixClass.
       * @details The copy constructor of ArnoldiMatrixClass.
       * @param   [in]   mat The other matrix.
       */
      ArnoldiMatrixClass(const ArnoldiMatrixClass<VectorType, DataType> &mat);
      
      /**
       * @brief   The move constructor of ArnoldiMatrixClass.
       * @details The move constructor of ArnoldiMatrixClass.
       * @param   [in]   mat The other matrix.
       */
      ArnoldiMatrixClass(ArnoldiMatrixClass<VectorType, DataType> &&mat);
      
      /**
       * @brief   Free the current matrix.
       * @details Free the current matrix.
       * @return     Return error message.
       */
      virtual int Clear();
      
      /**
       * @brief   The destructor of ArnoldiMatrixClass.
       * @details The destructor of ArnoldiMatrixClass.
       */
      virtual ~ArnoldiMatrixClass();
      
      /**
       * @brief   Get the data location of the matrix.
       * @details Get the data location of the matrix.
       * @return     Return the data location of the matrix.
       */
      virtual int GetDataLocation() const;
      
      /**
       * @brief   Get the local number of rows of the matrix.
       * @details Get the local number of rows of the matrix.
       * @return     Return the local number of rows of the matrix.
       */
      virtual int GetNumRowsLocal() const = 0;
      
      /**
       * @brief   Get the local number of columns of the matrix.
       * @details Get the local number of columns of the matrix.
       * @return     Return the local number of columns of the matrix.
       */
      virtual int GetNumColsLocal() const = 0;
      
      /**
       * @brief   Set the structure of a vector pointer that has same row partition as this matrix.
       * @details Set the structure of a vector pointer that has same row partition as this matrix.
       * @param [in,out] v The target vector.
       * @return     Return error message.
       */
      virtual int SetupVectorPtrStr(VectorType &v);
      
      /**
       * @brief   In place Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @details In place Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The first vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The second vector.
       * @return           Return error message.
       */
      virtual int MatVec( char trans, const DataType &alpha, VectorType &x, const DataType &beta, VectorType &y);
      
      /**
       * @brief   Get comm, np, and myid. Get the global one.
       * @details Get comm, np, and myid. Get the global one.
       * @param   [in]        np The number of processors.
       * @param   [in]        myid The local MPI rank number.
       * @param   [in]        comm The MPI_Comm.
       * @return     Return error message.
       */
      virtual int GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
      
      /**
       * @brief   Get the MPI_comm.
       * @details Get the MPI_comm.
       * @return     Return the MPI_comm.
       */
      virtual MPI_Comm GetComm() const;
      
   };
   
   typedef ArnoldiMatrixClass<vector_seq_float, float>         arnoldimatrix_seq_float;
   typedef ArnoldiMatrixClass<vector_seq_double, double>       arnoldimatrix_seq_double;
   typedef ArnoldiMatrixClass<vector_seq_complexs, complexs>   arnoldimatrix_seq_complexs;
   typedef ArnoldiMatrixClass<vector_seq_complexd, complexd>   arnoldimatrix_seq_complexd;
   typedef ArnoldiMatrixClass<vector_par_float, float>         arnoldimatrix_par_float;
   typedef ArnoldiMatrixClass<vector_par_double, double>       arnoldimatrix_par_double;
   typedef ArnoldiMatrixClass<vector_par_complexs, complexs>   arnoldimatrix_par_complexs;
   typedef ArnoldiMatrixClass<vector_par_complexd, complexd>   arnoldimatrix_par_complexd;
   
   template<> struct PargemslrIsComplex<ArnoldiMatrixClass<vector_seq_complexs, complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<ArnoldiMatrixClass<vector_seq_complexd, complexd> > : public std::true_type {};
   template<> struct PargemslrIsComplex<ArnoldiMatrixClass<vector_par_complexs, complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<ArnoldiMatrixClass<vector_par_complexd, complexd> > : public std::true_type {};
   
   
}

#endif
