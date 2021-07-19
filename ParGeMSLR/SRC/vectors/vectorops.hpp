#ifndef PARGEMSLR_VECTOROPS_H
#define PARGEMSLR_VECTOROPS_H

/**
 * @file       vectorops.hpp
 * @brief      Vector operations.
 * @details    Vector operations. \n
 *             VectorXxx: functions for base vector. \n
 *             SequentialVectorXxx: functions for sequential vector. \n
 */

#include <vector>

#include "../utils/utils.hpp"
#include "vector.hpp"
#include "int_vector.hpp"
#include "sequential_vector.hpp"
#include "parallel_vector.hpp"

using namespace std;

namespace pargemslr
{
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<int> &vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<long int> &vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<float> &vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<double> &vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<complexs> &vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPrecision(const VectorVirtualClass<complexd> &vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<int> *vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<long int> *vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<float> *vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<double> *vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<complexs> *vec);
   
   /**
    * @brief   Get the precision of a vector.
    * @details Get the precision of a vector.
    * @param [in]  vec the vector type.
    * @return      Return the precision in PrecisionEnum form.
    */
   PrecisionEnum GetVectorPPrecision(const VectorVirtualClass<complexd> *vec);
   
#ifdef PARGEMSLR_CUDA
   /**
    * @brief   Generate float random number between 0 and 1.
    * @details Generate float random number between 0 and 1.
    * @param [out] x the random vector.
    * @return      Return error message.
    */
   int VectorRandDevice( VectorClass<float> &x);
   
   /**
    * @brief   Generate double random number between 0 and 1.
    * @details Generate double random number between 0 and 1.
    * @param [out] x the random vector.
    * @return      Return error message.
    */
   int VectorRandDevice( VectorClass<double> &x);
   
   /**
    * @brief   Generate single complex random number a + bi. a and b are between 0 and 1.
    * @details Generate single complex random number a + bi. a and b are between 0 and 1.
    * @param [out] x the random vector.
    * @return      Return error message.
    */
   int VectorRandDevice( VectorClass<complexs> &x);
   
   /**
    * @brief   Generate double complex random number a + bi. a and b are between 0 and 1.
    * @details Generate double complex random number a + bi. a and b are between 0 and 1.
    * @param [out] x the random vector.
    * @return      Return error message.
    */
   int VectorRandDevice( VectorClass<complexd> &x);

#endif
   
   
   /**
    * @brief   DOT product of two float vectors.
    * @details DOT product of two float vectors.
    * @param [in]   n The length of the vectors.
    * @param [in]   x The left vector.
    * @param [in]   y The right vector.
    * @param [out]  t The dot product.
    * @return           Return error message.
    */
   template <typename T>
   int VectorPDotTemplate( int n, const T *x, const T *y, T &t );
   
   /**
    * @brief   DOT product of two float vectors.
    * @details DOT product of two float vectors.
    * @param [in]   x The left vector.
    * @param [in]   y The right vector.
    * @param [out]  t The dot product.
    * @return           Return error message.
    */
   int VectorDot( const vector_base_float &x, const vector_base_float &y, float &t);
   
   /**
    * @brief   DOT product of two double vectors.
    * @details DOT product of two double vectors.
    * @param [in]   x The left vector.
    * @param [in]   y The right vector.
    * @param [out]  t The dot product.
    * @return           Return error message.
    */
   int VectorDot( const vector_base_double &x, const vector_base_double &y, double &t);
   
   /**
    * @brief   DOTC product of two single complex vectors.
    * @details DOTC product of two single complex vectors.
    * @param [in]   x The left vector.
    * @param [in]   y The right vector.
    * @param [out]  t The dot product.
    * @return           Return error message.
    */
   int VectorDot( const vector_base_complexs &x, const vector_base_complexs &y, complexs &t);
   
   /**
    * @brief   DOTC product of two double complex vectors.
    * @details DOTC product of two double complex vectors.
    * @param [in]   x The left vector.
    * @param [in]   y The right vector.
    * @param [out]  t The dot product.
    * @return           Return error message.
    */
   int VectorDot( const vector_base_complexd &x, const vector_base_complexd &y, complexd &t);

#ifdef PARGEMSLR_CUDA 
   
   /**
    * @brief   DOT product of two float vectors.
    * @details DOT product of two float vectors.
    * @param [in]   x The left vector.
    * @param [in]   y The right vector.
    * @param [out]  t The dot product.
    * @return           Return error message.
    */
   int VectorSdotDevice( const vector_base_float &x, const vector_base_float &y, float &t);
   
   /**
    * @brief   DOT product of two double vectors.
    * @details DOT product of two double vectors.
    * @param [in]   x The left vector.
    * @param [in]   y The right vector.
    * @param [out]  t The dot product.
    * @return           Return error message.
    */
   int VectorDdotDevice( const vector_base_double &x, const vector_base_double &y, double &t);
   
   /**
    * @brief   DOTC product of two single complex vectors.
    * @details DOTC product of two single complex vectors.
    * @param [in]   x The left vector.
    * @param [in]   y The right vector.
    * @param [out]  t The dot product.
    * @return           Return error message.
    */
   int VectorCdotDevice( const vector_base_complexs &x, const vector_base_complexs &y, complexs &t);
   
   /**
    * @brief   DOTC product of two double complex vectors.
    * @details DOTC product of two double complex vectors.
    * @param [in]   x The left vector.
    * @param [in]   y The right vector.
    * @param [out]  t The dot product.
    * @return           Return error message.
    */
   int VectorZdotDevice( const vector_base_complexd &x, const vector_base_complexd &y, complexd &t);
   
#endif
   
   /**
    * @brief   Scale a float vector.
    * @details Scale a float vector.
    * @param [in]   n The length of the vectors.
    * @param [in,out]   x The target vector.
    * @param [in]       a The scaling scalar.
    * @return           Return error message.
    */
   template <typename T>
   int VectorPScaleTemplate( int n, T *x, const T&a);
   
   /**
    * @brief   Scale a float vector.
    * @details Scale a float vector.
    * @param [in,out]   x The target vector.
    * @param [in]       a The scaling scalar.
    * @return           Return error message.
    */
   int VectorScale(VectorClass<float> &x, const float &a);
   
   /**
    * @brief   Scale a double vector.
    * @details Scale a double vector.
    * @param [in,out]   x The target vector.
    * @param [in]       a The scaling scalar.
    * @return           Return error message.
    */
   int VectorScale(VectorClass<double> &x, const double &a);
   
   /**
    * @brief   Scale a single complex vector.
    * @details Scale a single complex vector.
    * @param [in,out]   x The target vector.
    * @param [in]       a The scaling scalar.
    * @return           Return error message.
    */
   int VectorScale(VectorClass<complexs> &x, const complexs &a);
   
   /**
    * @brief   Scale a double complex vector.
    * @details Scale a double complex vector.
    * @param [in,out]   x The target vector.
    * @param [in]       a The scaling scalar.
    * @return           Return error message.
    */
   int VectorScale(VectorClass<complexd> &x, const complexd &a);
   
#ifdef PARGEMSLR_CUDA
   
   /**
    * @brief   Scale a float vector.
    * @details Scale a float vector.
    * @param [in,out]   x The target vector.
    * @param [in]       a The scaling scalar.
    * @return           Return error message.
    */
   int VectorSscaleDevice(VectorClass<float> &x, const float &a);
   
   /**
    * @brief   Scale a double vector.
    * @details Scale a double vector.
    * @param [in,out]   x The target vector.
    * @param [in]       a The scaling scalar.
    * @return           Return error message.
    */
   int VectorDscaleDevice(VectorClass<double> &x, const double &a);
   
   /**
    * @brief   Scale a single complex vector.
    * @details Scale a single complex vector.
    * @param [in,out]   x The target vector.
    * @param [in]       a The scaling scalar.
    * @return           Return error message.
    */
   int VectorCscaleDevice(VectorClass<complexs> &x, const complexs &a);
   
   /**
    * @brief   Scale a double complex vector.
    * @details Scale a double complex vector.
    * @param [in,out]   x The target vector.
    * @param [in]       a The scaling scalar.
    * @return           Return error message.
    */
   int VectorZscaleDevice(VectorClass<complexd> &x, const complexd &a);
   
#endif
   
   /**
    * @brief   AXPY of two float vectors, in place version. y = a*x + y.
    * @details AXPY of two float vectors, in place version. y = a*x + y.
    * @param [in]       n The length of the vectors.
    * @param [in]       a The scaling scalar.
    * @param [in]       x The left vector.
    * @param [in,out]   y The right vector.
    * @return           Return error message.
    */
   template <typename T>
   int VectorPAxpyTemplate( int n, const T &a, const T *x, T *y);
   
   /**
    * @brief   AXPY of two float vectors, in place version. y = a*x + y.
    * @details AXPY of two float vectors, in place version. y = a*x + y.
    * @param [in]       a The scaling scalar.
    * @param [in]       x The left vector.
    * @param [in,out]   y The right vector.
    * @return           Return error message.
    */
   int VectorAxpy( const float &a, const VectorClass<float> &x, VectorClass<float> &y);
   
   /**
    * @brief   AXPY of two double vectors, in place version. y = a*x + y.
    * @details AXPY of two double vectors, in place version. y = a*x + y.
    * @param [in]       a The scaling scalar.
    * @param [in]       x The left vector.
    * @param [in,out]   y The right vector.
    * @return           Return error message.
    */
   int VectorAxpy( const double &a, const VectorClass<double> &x, VectorClass<double> &y);
   
   /**
    * @brief   AXPY of two single complex vectors, in place version. y = a*x + y.
    * @details AXPY of two single complex vectors, in place version. y = a*x + y.
    * @param [in]       a The scaling scalar.
    * @param [in]       x The left vector.
    * @param [in,out]   y The right vector.
    * @return           Return error message.
    */
   int VectorAxpy( const complexs &a, const VectorClass<complexs> &x, VectorClass<complexs> &y);
   
   /**
    * @brief   AXPY of two double complex vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @details AXPY of two double complex vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @param [in]       a The scaling scalar.
    * @param [in]       x The left vector.
    * @param [in,out]   y The right vector.
    * @return           Return error message.
    */
   int VectorAxpy( const complexd &a, const VectorClass<complexd> &x, VectorClass<complexd> &y);
   
#ifdef PARGEMSLR_CUDA 
   
   /**
    * @brief   AXPY of two float vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @details AXPY of two float vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @param [in]       a The scaling scalar.
    * @param [in]       x The left vector.
    * @param [in,out]   y The right vector.
    * @return           Return error message.
    */
   int VectorSaxpyDevice( const float &a, const VectorClass<float> &x, VectorClass<float> &y);
   
   /**
    * @brief   AXPY of two double vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @details AXPY of two double vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @param [in]       a The scaling scalar.
    * @param [in]       x The left vector.
    * @param [in,out]   y The right vector.
    * @return           Return error message.
    */
   int VectorDaxpyDevice( const double &a, const VectorClass<double> &x, VectorClass<double> &y);
   
   /**
    * @brief   AXPY of two single complex vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @details AXPY of two single complex vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @param [in]       a The scaling scalar.
    * @param [in]       x The left vector.
    * @param [in,out]   y The right vector.
    * @return           Return error message.
    */
   int VectorCaxpyDevice( const complexs &a, const VectorClass<complexs> &x, VectorClass<complexs> &y);
   
   /**
    * @brief   AXPY of two double complex vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @details AXPY of two double complex vectors, in place version. y = a*x + y. Currenlty we don't support x == y.
    * @param [in]       a The scaling scalar.
    * @param [in]       x The left vector.
    * @param [in,out]   y The right vector.
    * @return           Return error message.
    */
   int VectorZaxpyDevice( const complexd &a, const VectorClass<complexd> &x, VectorClass<complexd> &y);

   /**
    * @brief   Setup cusparse dense vector.
    * @details Setup cusparse dense vector.
    * @param [in,out] v The vector.
    * @return     Return error message.      
    */
   int SequentialVectorCreateCusparseDnVec(vector_seq_float &v);
   
   /**
    * @brief   Setup cusparse dense vector.
    * @details Setup cusparse dense vector.
    * @param [in,out] v The vector.
    * @return     Return error message.      
    */
   int SequentialVectorCreateCusparseDnVec(vector_seq_double &v);
   
   /**
    * @brief   Setup cusparse dense vector.
    * @details Setup cusparse dense vector.
    * @param [in,out] v The vector.
    * @return     Return error message.      
    */
   int SequentialVectorCreateCusparseDnVec(vector_seq_complexs &v);
   
   /**
    * @brief   Setup cusparse dense vector.
    * @details Setup cusparse dense vector.
    * @param [in,out] v The vector.
    * @return     Return error message.      
    */
   int SequentialVectorCreateCusparseDnVec(vector_seq_complexd &v);

#endif
   
   /**
    * @brief      Binary search between [s, e] inside an array.
    * @details    Binary search between [s, e] inside an array.
    * @param [in]    v The target vector.
    * @param [in]    val The target value.
    * @param [in]    s The start location in the array
    * @param [in]    e The end(include) location in the array
    * @param [out]    idx If found the value, set to the index of the value. Otherwise the position to insert, or -1 if s > e.
    * @param [in]    ascending The array is descend or ascend.
    * @param [in]    option Other search options. 0: no extra option. 1: report the first if there are duplicates. 2: report the last.
    * @return return -1 if the value isnot found. Otherwise the index of it.
    */
   template <typename T>
   int VectorPBsearchHost(const T *v, const T &val, int s, int e, int &idx, bool ascending, int option);
   
   /**
    * @brief   Print the vector.
    * @details Print the vector.
    * @param [in]   x The target vector.
    * @param [in]   conditiona First condition.
    * @param [in]   conditionb Secend condition, only print when conditiona == conditionb.
    * @param [in]   width The plot width.
    * @return           Return error message.
    */
   template <typename T>
   int VectorPlotHost( const VectorVirtualClass<T> &x, int conditiona, int conditionb, int width);
   
   /**
    * @brief   Read a vector from matrix marker format file.
    * @details Read a vector from matrix marker format file.
    * @param [out]  vec The return vector.
    * @param [in]   vecfile The file name.
    * @param [in]   idxin The index base of the input vector, 0-based or 1-based.
    * @return       Return error message.
    */
   int SequentialVectorReadFromFile(SequentialVectorClass<float> &vec, const char *vecfile, int idxin);
   
   /**
    * @brief   Read a vector from matrix marker format file.
    * @details Read a vector from matrix marker format file.
    * @param [out]  vec The return vector.
    * @param [in]   vecfile The file name.
    * @param [in]   idxin The index base of the input vector, 0-based or 1-based.
    * @return       Return error message.
    */
   int SequentialVectorReadFromFile(SequentialVectorClass<double> &vec, const char *vecfile, int idxin);
   
   /**
    * @brief   Read a vector from matrix marker format file.
    * @details Read a vector from matrix marker format file.
    * @param [out]  vec The return vector.
    * @param [in]   vecfile The file name.
    * @param [in]   idxin The index base of the input vector, 0-based or 1-based.
    * @return       Return error message.
    */
   int SequentialVectorReadFromFile(SequentialVectorClass<complexs> &vec, const char *vecfile, int idxin);
   
   /**
    * @brief   Read a vector from matrix marker format file.
    * @details Read a vector from matrix marker format file.
    * @param [out]  vec The return vector.
    * @param [in]   vecfile The file name.
    * @param [in]   idxin The index base of the input vector, 0-based or 1-based.
    * @return       Return error message.
    */
   int SequentialVectorReadFromFile(SequentialVectorClass<complexd> &vec, const char *vecfile, int idxin);
   
   /**
    * @brief   Copy data from vector of type T1 to vector of type T2.
    * @details Copy data from vector of type T1 to vector of type T2. \n
    *          Currently only supports float <-> double and complexs <-> conplexd
    * @param [in]   vec_in The input vector.
    * @param [out]  vec_out The output vector.
    * @return       Return error message.
    */
   template <typename T1, typename T2>
   int VectorCopy(VectorClass<T1> &vec_in, VectorClass<T2> &vec2_out);
   
}

#endif
