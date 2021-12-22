#ifndef PARGEMSLR_COMPLEX_H
#define PARGEMSLR_COMPLEX_H

/**
 * @file complex.hpp
 * @brief The complex value data structure.
 */

#include <complex>
#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#endif
//#include <iostream>

namespace pargemslr
{
   
   /**
    * @brief   The template class complex.
    * @details The template class complex.
    */
   template <typename T>
   class ComplexValueClass
   {
   private:
      
      /**
       * @brief   The real part.
       * @details The real part.
       */
      T  _real;
      
      /**
       * @brief   The imag part.
       * @details The imag part.
       */
      T  _imag;
      
   public:
      /* constructor */
      /**
       * @brief   The constructor of complex value class.
       * @details The constructor of complex value class.
       */
      ComplexValueClass()
      {
         _real = T(); 
         _imag = T();
      }
      
      /**
       * @brief   The constructor of complex value class.
       * @details The constructor of complex value class.
       * @param [in] real The real part, imag part is 0.0.
       */
      ComplexValueClass(T real)
      {
         _real = real;
         _imag = T();
      }
      
      /**
       * @brief   The constructor of complex value class.
       * @details The constructor of complex value class.
       * @param [in] real The real part.
       * @param [in] imag The imag part.
       */
      ComplexValueClass(T real, T imag)
      { 
         _real = real;
         _imag = imag;
      }
      
      /**
       * @brief   Get the real part.
       * @details Get the real part.
       * @return  Return the real part.
       */
#ifdef PARGEMSLR_CUDA
__host__ __device__
#endif
      T& Real()
      {
         return _real;
      }
      
      /**
       * @brief   Get the real part.
       * @details Get the real part.
       * @return  Return the real part.
       */
#ifdef PARGEMSLR_CUDA
__host__ __device__
#endif
      const T& Real() const
      {
         return _real;
      }
      
      /**
       * @brief   Get the imag part.
       * @details Get the imag part.
       * @return  Return the imag part.
       */
#ifdef PARGEMSLR_CUDA
__host__ __device__
#endif
      T& Imag()
      {
         return _imag;
      }
      
      /**
       * @brief   Get the imag part.
       * @details Get the imag part.
       * @return  Return the imag part.
       */
#ifdef PARGEMSLR_CUDA
__host__ __device__
#endif
      const T& Imag() const
      {
         return _imag;
      }
      
      /**
       * @brief   Get the square of 2-norm. Note that std::complex use the this value as norm.
       * @details Get the square of 2-norm. Note that std::complex use the this value as norm.
       * @return  Return the norm.
       */
      T Norm() const
      {
         return _real * _real + _imag * _imag;
      }
      
      /**
       * @brief   Get the 2-norm. Note that std::complex use the square of this value as norm.
       * @details Get the 2-norm. Note that std::complex use the square of this value as norm.
       * @return  Return the norm.
       */
      T Norm2() const
      {
         return std::sqrt(_real * _real + _imag * _imag);
      }
      
      /**
       * @brief   Get the 2-norm as absolute value.
       * @details Get the 2-norm as absolute value.
       * @return  Return the sqrt of norm as absolute value.
       */
      T Abs() const
      {
         return std::sqrt(_real * _real + _imag * _imag);
      }
      
      /**
       * @brief   Get the conjugate.
       * @details Get the conjugate.
       * @return  Return the conjugate.
       */
      ComplexValueClass<T> Conj() const
      {
         return ComplexValueClass<T>(_real, -_imag);
      }
      
      /**
       * @brief   Conver to real value, only keep the real part.
       * @details Conver to real value, only keep the real part.
       * @note    This function is used to overcome some compiling issues, should not be called directly.\n
       *          Some operations, for example, complexs(1.0,-1.0) == 1.0, will call this function.
       * @return  Return the real part.
       */
      /*
      operator T() const
      {
         if(_imag != T())
         {
            std::cout<<"Error: converting complex type to real type with nonzero imag part, imag part lost."<<std::endl;            
            std::cout<<"This message should never be printed from a correct program."<<std::endl;            
            std::cout<<"Possible cause: operating real value with complex value."<<std::endl;            
         }
         return _real;
      }
      */
      
      /**
       * @brief   = operator with real value.
       * @details = operator with real value.
       * @param [in] real The real value.
       * @return  Return the result.
       */
#ifdef PARGEMSLR_CUDA
__host__ __device__
#endif
      ComplexValueClass<T>& operator=(const T& real)
      {
         _real = real; 
         _imag = T(); 
         return *this;
      }
      
      /**
       * @brief   += operator with real value.
       * @details += operator with real value.
       * @param [in] real The real value.
       * @return  Return the result.
       */
      ComplexValueClass<T>& operator+=(const T& real)
      { 
         _real += real; 
         return *this;
      }
      
      /**
       * @brief   -= operator with real value.
       * @details -= operator with real value.
       * @param [in] real The real value.
       * @return  Return the result.
       */
      ComplexValueClass<T>& operator-=(const T& real)
      { 
         _real -= real; 
         return *this;
      }
      
      /**
       * @brief   *= operator with real value.
       * @details *= operator with real value.
       * @param [in] real The real value.
       * @return  Return the result.
       */
      ComplexValueClass<T>& operator*=(const T& real)
      { 
         _real *= real; 
         _imag *= real; 
         return *this;
      }
      
      /**
       * @brief   /= operator with real value.
       * @details /= operator with real value.
       * @param [in] real The real value.
       * @return  Return the result.
       */
      ComplexValueClass<T>& operator/=(const T& real)
      { 
         _real /= real; 
         _imag /= real; 
         return *this;
      }
      
      /**
       * @brief   = operator with real value.
       * @details = operator with real value.
       * @param [in] val The other value.
       * @return  Return the result.
       */
#ifdef PARGEMSLR_CUDA
__host__ __device__
#endif
      ComplexValueClass<T>& operator=(const ComplexValueClass<T>& val)
      { 
         _real = val.Real(); 
         _imag = val.Imag(); 
         return *this;
      }
      
      /**
       * @brief   += operator with real value.
       * @details += operator with real value.
       * @param [in] val The other value.
       * @return  Return the result.
       */
      ComplexValueClass<T>& operator+=(const ComplexValueClass<T>& val)
      { 
         _real += val.Real(); 
         _imag += val.Imag(); 
         return *this;
      }
      
      /**
       * @brief   -= operator with real value.
       * @details -= operator with real value.
       * @param [in] val The other value.
       * @return  Return the result.
       */
      ComplexValueClass<T>& operator-=(const ComplexValueClass<T>& val)
      { 
         _real -= val.Real(); 
         _imag -= val.Imag(); 
         return *this;
      }
      
      /**
       * @brief   *= operator with real value.
       * @details *= operator with real value.
       * @param [in] val The other value.
       * @return  Return the result.
       */
      ComplexValueClass<T>& operator*=(const ComplexValueClass<T>& val)
      { 
         /* (a+bi)*(c+di) = ac-bd + (ad+bc)i */
         const T new_real = _real * val.Real() - _imag * val.Imag();
         _imag = _real * val.Imag() + _imag * val.Real();
         _real = new_real; 
         return *this;
      }
      
      /**
       * @brief   /= operator with real value.
       * @details /= operator with real value.
       * @param [in] val The other value.
       * @return  Return the result.
       */
      ComplexValueClass<T>& operator/=(const ComplexValueClass<T>& val)
      { 
         /* (a+bi)/(c+di) is slightly more complex
          * [(a+bi)*(c-di)]/[(c+di)*(c-di)] =
          * [ ac+bd + (bc-ad)i ]/(c*c+d*d)
          */
         const T new_lower = val.Norm();
         const T new_real = _real * val.Real() + _imag * val.Imag();
         _imag = (val.Real() * _imag - val.Imag() * _real) / new_lower;
         _real = new_real / new_lower; 
         return *this;
      }
      
   };
   
   /**
    * @brief   + operator.
    * @details + operator.
    * @param [in] val The complex value.
    * @param [in] real The real value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator+(const ComplexValueClass<T> &val, const T& real)
   {
      ComplexValueClass<T> val1 = val;
      val1 += real; 
      return val1;
   }
   
   /**
    * @brief   + operator.
    * @details + operator.
    * @param [in] real The real value.
    * @param [in] val The complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator+( const T& real, const ComplexValueClass<T> &val)
   {
      ComplexValueClass<T> val1 = val;
      val1 += real; 
      return val1;
   }
   
   /**
    * @brief   + operator.
    * @details + operator.
    * @param [in] val1 The first complex value.
    * @param [in] val2 The second complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator+( const ComplexValueClass<T> &val1, const ComplexValueClass<T> &val2)
   {
      ComplexValueClass<T> val = val1;
      val += val2; 
      return val;
   }
   
   /**
    * @brief   + operator.
    * @details + operator.
    * @param [in] val The complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator+( const ComplexValueClass<T> &val)
   {
      return ComplexValueClass<T>( val.Real(), val.Imag());
   }
   
   /**
    * @brief   - operator.
    * @details - operator.
    * @param [in] val The complex value.
    * @param [in] real The real value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator-(const ComplexValueClass<T> &val, const T& real)
   {
      ComplexValueClass<T> val1 = val;
      val1 -= real; 
      return val1;
   }
   
   /**
    * @brief   - operator.
    * @details - operator.
    * @param [in] real The real value.
    * @param [in] val The complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator-( const T& real, const ComplexValueClass<T> &val)
   {
      ComplexValueClass<T> val1 = ComplexValueClass<T>(real, T());
      val1 -= val; 
      return val1;
   }
   
   /**
    * @brief   - operator.
    * @details - operator.
    * @param [in] val1 The first complex value.
    * @param [in] val2 The second complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator-( const ComplexValueClass<T> &val1, const ComplexValueClass<T> &val2)
   {
      ComplexValueClass<T> val = val1;
      val -= val2; 
      return val;
   }
   
   /**
    * @brief   - operator.
    * @details - operator.
    * @param [in] val The complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator-( const ComplexValueClass<T> &val)
   {
      return ComplexValueClass<T>( -val.Real(), -val.Imag());
   }
   
   /**
    * @brief   * operator.
    * @details * operator.
    * @param [in] val The complex value.
    * @param [in] real The real value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator*(const ComplexValueClass<T> &val, const T& real)
   {
      ComplexValueClass<T> val1 = val;
      val1 *= real; 
      return val1;
   }
   
   /**
    * @brief   * operator.
    * @details * operator.
    * @param [in] real The real value.
    * @param [in] val The complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator*( const T& real, const ComplexValueClass<T> &val)
   {
      ComplexValueClass<T> val1 = val;
      val1 *= real; 
      return val1;
   }
   
   /**
    * @brief   * operator.
    * @details * operator.
    * @param [in] val1 The first complex value.
    * @param [in] val2 The second complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator*( const ComplexValueClass<T> &val1, const ComplexValueClass<T> &val2)
   {
      ComplexValueClass<T> val = val1;
      val *= val2; 
      return val;
   }
   
   /**
    * @brief   / operator.
    * @details / operator.
    * @param [in] val The complex value.
    * @param [in] real The real value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator/(const ComplexValueClass<T> &val, const T& real)
   {
      ComplexValueClass<T> val1 = val;
      val1 /= real; 
      return val1;
   }
   
   /**
    * @brief   / operator.
    * @details / operator.
    * @param [in] real The real value.
    * @param [in] val The complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator/( const T& real, const ComplexValueClass<T> &val)
   {
      ComplexValueClass<T> val1 = ComplexValueClass<T>(real, T());
      val1 /= val; 
      return val1;
   }
   
   /**
    * @brief   / operator.
    * @details / operator.
    * @param [in] val1 The first complex value.
    * @param [in] val2 The second complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline ComplexValueClass<T> operator/( const ComplexValueClass<T> &val1, const ComplexValueClass<T> &val2)
   {
      ComplexValueClass<T> val = val1;
      val /= val2; 
      return val;
   }
   
   /**
    * @brief   == operator.
    * @details == operator.
    * @param [in] val1 The first complex value.
    * @param [in] val2 The second value.
    * @return  Return the result.
    */
   template <typename T>
   inline bool operator==( const ComplexValueClass<T> &val1, const T &val2)
   {
      return (val1.Real() == val2) && (val1.Imag() == T());
   }
   
   /**
    * @brief   == operator.
    * @details == operator.
    * @param [in] val1 The first value.
    * @param [in] val2 The second complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline bool operator==( const T &val1, const ComplexValueClass<T> &val2)
   {
      return (val1 == val2.Real()) && (T() == val2.Imag());
   }
   
   /**
    * @brief   == operator.
    * @details == operator.
    * @param [in] val1 The first complex value.
    * @param [in] val2 The second complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline bool operator==( const ComplexValueClass<T> &val1, const ComplexValueClass<T> &val2)
   {
      return (val1.Real() == val2.Real()) && (val1.Imag() == val2.Imag());
   }
   
   /**
    * @brief   != operator.
    * @details != operator.
    * @param [in] val1 The first complex value.
    * @param [in] val2 The second value.
    * @return  Return the result.
    */
   template <typename T>
   inline bool operator!=( const ComplexValueClass<T> &val1, const T &val2)
   {
      return (val1.Real() != val2) || (val1.Imag() != T());
   }
   
   /**
    * @brief   != operator.
    * @details != operator.
    * @param [in] val1 The first value.
    * @param [in] val2 The second complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline bool operator!=( const T &val1, const ComplexValueClass<T> &val2)
   {
      return (val1 != val2.Real()) || (T() != val2.Imag());
   }
   
   /**
    * @brief   != operator.
    * @details != operator.
    * @param [in] val1 The first complex value.
    * @param [in] val2 The second complex value.
    * @return  Return the result.
    */
   template <typename T>
   inline bool operator!=( const ComplexValueClass<T> &val1, const ComplexValueClass<T> &val2)
   {
      return (val1.Real() != val2.Real()) || (val1.Imag() != val2.Imag());
   }
   
   /**
    * @brief   ostream.
    * @details ostream.
    * @param [in] os The ostream.
    * @param [in] val2 The second complex value.
    * @return  Return the result.
    */
   template<typename T, typename CharT, class Traits>
   std::basic_ostream<CharT, Traits>&
   operator<<(std::basic_ostream<CharT, Traits>& basic_os, const ComplexValueClass<T>& val)
   {
      std::basic_ostringstream<CharT, Traits> basic_ostr;
      basic_ostr.flags(basic_os.flags());
      basic_ostr.imbue(basic_os.getloc());
      basic_ostr.precision(basic_os.precision());
      
      T val_r = val.Real();
      T val_i = val.Imag();
      
      if(val_r < 0)
      {
         basic_ostr<<"-";
      }
      else
      {
         basic_ostr<<"+";
      }
      basic_ostr<<std::abs(val_r);
      if(val_i < 0)
      {
         basic_ostr<<"-";
      }
      else
      {
         basic_ostr<<"+";
      }
      basic_ostr<<std::abs(val_i)<<"i";
      return basic_os << basic_ostr.str();
   }
   
   typedef ComplexValueClass<float>    complexs;
   typedef ComplexValueClass<double>   complexd;

#ifdef PARGEMSLR_OPENMP

#pragma omp declare reduction(+:complexs:omp_out += omp_in) initializer (omp_priv=complexs())
#pragma omp declare reduction(+:complexd:omp_out += omp_in) initializer (omp_priv=complexd())

#endif

#ifdef PARGEMSLR_MKL
   #define ccomplexs MKL_Complex8
   #define ccomplexd MKL_Complex16
#else
   /**
    * @brief   The c style struct of single complex.
    * @details The c style struct of single complex.
    */
   typedef struct CComplexSingleStruct
   {
      float real, imag;
   }ccomplexs;
   
   /**
    * @brief   The c style struct of double complex.
    * @details The c style struct of double complex.
    */
   typedef struct CComplexDoubleStruct
   {
      double real, imag;
   }ccomplexd;
#endif   

   /**
    * @brief   Tell if a value is a complex value.
    * @details Tell if a value is a complex value.
    */
   template <class T> struct PargemslrIsComplex : public std::false_type {};
   template <class T> struct PargemslrIsComplex<const T> : public PargemslrIsComplex<T> {};
   template <class T> struct PargemslrIsComplex<volatile const T> : public PargemslrIsComplex<T>{};
   template <class T> struct PargemslrIsComplex<volatile T> : public PargemslrIsComplex<T>{};
   template<> struct PargemslrIsComplex<complexs> : public std::true_type {};
   template<> struct PargemslrIsComplex<complexd> : public std::true_type {};
   
   /**
    * @brief   Tell if a value is a real value.
    * @details Tell if a value is a real value.
    */
   template <class T> struct PargemslrIsReal : public std::false_type {};
   template <class T> struct PargemslrIsReal<const T> : public PargemslrIsReal<T> {};
   template <class T> struct PargemslrIsReal<volatile const T> : public PargemslrIsReal<T>{};
   template <class T> struct PargemslrIsReal<volatile T> : public PargemslrIsReal<T>{};
   template<> struct PargemslrIsReal<float> : public std::true_type {};
   template<> struct PargemslrIsReal<double> : public std::true_type {};
   
}

#endif
