/* this file is for the GPU version thrust call */

#ifdef PARGEMSLR_CUDA

#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "matrix.hpp"
#include "csr_matrix.hpp"
#include "matrixops.hpp"

#include <cstring>
#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

namespace pargemslr
{  
   template <typename T>
   int CsrMatrixClass<T>::Eye()
   {
      /* don't need to worry for the host version */
      int   n, err = 0;
      int   location = this->GetDataLocation();
      
      PARGEMSLR_CHKERR( this->_ncols != this->_nrows);
      
      /* change to hold value */
      this->_isholdingdata = true;
      
      if(this->_ncols == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      n = this->_ncols;
      
      this->_nnz = n;
      
      /* create I, J, and A */
      err = this->_i_vec.UnitPerm(); PARGEMSLR_CHKERR(err);
      err = this->_j_vec.Setup(n, location, false); PARGEMSLR_CHKERR(err);
      err = this->_j_vec.UnitPerm(); PARGEMSLR_CHKERR(err);
      err = this->_a_vec.Setup(n, location, false); PARGEMSLR_CHKERR(err);
      err = this->_a_vec.Fill(1.0); PARGEMSLR_CHKERR(err);
      
      if( location == kMemoryDevice || location == kMemoryUnified)
      {
         CsrMatrixCreateCusparseSpMat(*this);
      }
      
      return err;
      
   }
   template int CsrMatrixClass<float>::Eye();
   template int CsrMatrixClass<double>::Eye();
   template int CsrMatrixClass<complexs>::Eye();
   template int CsrMatrixClass<complexd>::Eye();
   
   template <typename T>
   int CsrMatrixClass<T>::MoveData( const int &location)
   {
      int loc_from = this->GetDataLocation();
      
      if(location == loc_from)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      if(location == kMemoryDevice || location == kMemoryUnified)
      {
         this->SortRow();
      }
      
      this->_i_vec.MoveData(location);
      this->_j_vec.MoveData(location);
      if(this->_isholdingdata)
      {
         this->_a_vec.MoveData(location);
      }
      
      if( location == kMemoryDevice || location == kMemoryUnified)
      {
         CsrMatrixCreateCusparseSpMat(*this);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixClass<float>::MoveData( const int &location);
   template int CsrMatrixClass<double>::MoveData( const int &location);
   template int CsrMatrixClass<complexs>::MoveData( const int &location);
   template int CsrMatrixClass<complexd>::MoveData( const int &location);
}
#endif
