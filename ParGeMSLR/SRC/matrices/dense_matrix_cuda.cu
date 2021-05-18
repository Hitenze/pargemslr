/* this file is for the GPU version thrust call */

#ifdef PARGEMSLR_CUDA

#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "matrix.hpp"
#include "../vectors/sequential_vector.hpp"
#include "dense_matrix.hpp"
#include "matrixops.hpp"
#include "../utils/utils.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace pargemslr
{  
	template <typename T>
   int DenseMatrixClass<T>::Eye()
   {
      PARGEMSLR_CHKERR( this->_nrows != this->_ncols);
      
      if(this->GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Dense matrix EYE function currently is only on the host."<<std::endl;
         this->MoveData(kMemoryHost);
      }
      
      int i;
      
      this->Fill(0.0);
      
      for(i = 0 ; i < this->_nrows ; i ++)
      {
         this->operator()(i, i) = 1.0;
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int DenseMatrixClass<float>::Eye();
   template int DenseMatrixClass<double>::Eye();
   template int DenseMatrixClass<complexs>::Eye();
   template int DenseMatrixClass<complexd>::Eye();
}
#endif
