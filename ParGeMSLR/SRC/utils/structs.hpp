#ifndef STRUCTS_H
#define STRUCTS_H

/**
 * @file structs.hpp
 * @brief Declare (almost) all data structures.
 */
 
namespace pargemslr
{

   template <typename T> class SequentialVectorClass;

   template <typename T> class ParallelVectorClass;

   template <typename T> class CsrMatrixClass;

   template <typename T> class ParallelCsrMatrixClass;

   template <class MatrixType, class VectorType, typename DataType> class GemslrClass;
   
}

#endif
