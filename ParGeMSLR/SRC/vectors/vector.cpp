
#include "vector.hpp"
#include "vectorops.hpp"

namespace pargemslr
{
   template <typename T>
   VectorVirtualClass<T>::VectorVirtualClass() : ParallelLogClass()
   {
   }
   template VectorVirtualClass<int>::VectorVirtualClass();
   template VectorVirtualClass<long int>::VectorVirtualClass();
   template VectorVirtualClass<float>::VectorVirtualClass();
   template VectorVirtualClass<double>::VectorVirtualClass();
   template VectorVirtualClass<complexs>::VectorVirtualClass();
   template VectorVirtualClass<complexd>::VectorVirtualClass();
   
   template <typename T>
   VectorVirtualClass<T>::VectorVirtualClass(const VectorVirtualClass<T> &vec) : ParallelLogClass(vec)
   {
   }
   template VectorVirtualClass<int>::VectorVirtualClass(const VectorVirtualClass<int> &vec);
   template VectorVirtualClass<long int>::VectorVirtualClass(const VectorVirtualClass<long int> &vec);
   template VectorVirtualClass<float>::VectorVirtualClass(const VectorVirtualClass<float> &vec);
   template VectorVirtualClass<double>::VectorVirtualClass(const VectorVirtualClass<double> &vec);
   template VectorVirtualClass<complexs>::VectorVirtualClass(const VectorVirtualClass<complexs> &vec);
   template VectorVirtualClass<complexd>::VectorVirtualClass(const VectorVirtualClass<complexd> &vec);
   
   template <typename T>
   VectorVirtualClass<T>::VectorVirtualClass( VectorVirtualClass<T> &&vec) : ParallelLogClass(std::move(vec))
   {
   }
   template VectorVirtualClass<int>::VectorVirtualClass( VectorVirtualClass<int> &&vec);
   template VectorVirtualClass<long int>::VectorVirtualClass( VectorVirtualClass<long int> &&vec);
   template VectorVirtualClass<float>::VectorVirtualClass( VectorVirtualClass<float> &&vec);
   template VectorVirtualClass<double>::VectorVirtualClass( VectorVirtualClass<double> &&vec);
   template VectorVirtualClass<complexs>::VectorVirtualClass( VectorVirtualClass<complexs> &&vec);
   template VectorVirtualClass<complexd>::VectorVirtualClass( VectorVirtualClass<complexd> &&vec);
   
   template <typename T>
   VectorVirtualClass<T>::~VectorVirtualClass()
   {
   }
   template VectorVirtualClass<int>::~VectorVirtualClass();
   template VectorVirtualClass<long int>::~VectorVirtualClass();
   template VectorVirtualClass<float>::~VectorVirtualClass();
   template VectorVirtualClass<double>::~VectorVirtualClass();
   template VectorVirtualClass<complexs>::~VectorVirtualClass();
   template VectorVirtualClass<complexd>::~VectorVirtualClass();
   
   template <typename T>
   int VectorVirtualClass<T>::Clear()
   {
      /* call the base class clear function */
      parallel_log::Clear();
      return PARGEMSLR_SUCCESS;
   }
   template int VectorVirtualClass<int>::Clear();
   template int VectorVirtualClass<long int>::Clear();
   template int VectorVirtualClass<float>::Clear();
   template int VectorVirtualClass<double>::Clear();
   template int VectorVirtualClass<complexs>::Clear();
   template int VectorVirtualClass<complexd>::Clear();
   
   template <typename T>
   PrecisionEnum VectorVirtualClass<T>::GetPrecision() const
   {
      return GetVectorPPrecision(this);
   }
   template PrecisionEnum VectorVirtualClass<int>::GetPrecision() const;
   template PrecisionEnum VectorVirtualClass<long int>::GetPrecision() const;
   template PrecisionEnum VectorVirtualClass<float>::GetPrecision() const;
   template PrecisionEnum VectorVirtualClass<double>::GetPrecision() const;
   template PrecisionEnum VectorVirtualClass<complexs>::GetPrecision() const;
   template PrecisionEnum VectorVirtualClass<complexd>::GetPrecision() const;
   
   template <typename T>
   bool VectorVirtualClass<T>::IsParallel() const
   {
      return false;
   }
   template bool VectorVirtualClass<int>::IsParallel() const;
   template bool VectorVirtualClass<long int>::IsParallel() const;
   template bool VectorVirtualClass<float>::IsParallel() const;
   template bool VectorVirtualClass<double>::IsParallel() const;
   template bool VectorVirtualClass<complexs>::IsParallel() const;
   template bool VectorVirtualClass<complexd>::IsParallel() const;
   
	template <typename T>
   VectorClass<T>::VectorClass()
   {
   }
   template VectorClass<float>::VectorClass();
   template VectorClass<double>::VectorClass();
   template VectorClass<complexs>::VectorClass();
   template VectorClass<complexd>::VectorClass();
   
	template <typename T>
   VectorClass<T>::VectorClass(const VectorClass<T> &vec) : VectorVirtualClass<T>(vec)
   {
   }
   template VectorClass<float>::VectorClass(const VectorClass<float> &vec);
   template VectorClass<double>::VectorClass(const VectorClass<double> &vec);
   template VectorClass<complexs>::VectorClass(const VectorClass<complexs> &vec);
   template VectorClass<complexd>::VectorClass(const VectorClass<complexd> &vec);
   
	template <typename T>
   VectorClass<T>::VectorClass( VectorClass<T> &&vec) : VectorVirtualClass<T>(std::move(vec))
   {
   }
   template VectorClass<float>::VectorClass( VectorClass<float> &&vec);
   template VectorClass<double>::VectorClass( VectorClass<double> &&vec);
   template VectorClass<complexs>::VectorClass( VectorClass<complexs> &&vec);
   template VectorClass<complexd>::VectorClass( VectorClass<complexd> &&vec);
   
	template <typename T>
   int VectorClass<T>::Clear()
   {
      /* call the base class clear */
      VectorVirtualClass<T>::Clear();
      return PARGEMSLR_SUCCESS;
   }
   template int VectorClass<float>::Clear();
   template int VectorClass<double>::Clear();
   template int VectorClass<complexs>::Clear();
   template int VectorClass<complexd>::Clear();
   
   template <typename T>
   VectorClass<T>::~VectorClass()
   {
   }
   template VectorClass<float>::~VectorClass();
   template VectorClass<double>::~VectorClass();
   template VectorClass<complexs>::~VectorClass();
   template VectorClass<complexd>::~VectorClass();
   
}
