
#include <iostream>
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../vectors/vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "../matrices/matrix.hpp"
#include "../matrices/matrixops.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../matrices/parallel_csr_matrix.hpp"
#include "../matrices/dense_matrix.hpp"
#include "poly.hpp"

namespace pargemslr
{
	
   template <class MatrixType, class VectorType, typename DataType>
   PolyClass<MatrixType, VectorType, DataType>::PolyClass() : SolverClass<MatrixType, VectorType, DataType>()
   {
      this->_solver_type = kSolverPoly;
      this->_order = 2;
      this->_scale = 0.15;
      this->_n = 0;
      this->_location = kMemoryHost;
   }
   template precond_poly_csr_seq_float::PolyClass();
   template precond_poly_csr_seq_double::PolyClass();
   template precond_poly_csr_seq_complexs::PolyClass();
   template precond_poly_csr_seq_complexd::PolyClass();
   template precond_poly_csr_par_float::PolyClass();
   template precond_poly_csr_par_double::PolyClass();
   template precond_poly_csr_par_complexs::PolyClass();
   template precond_poly_csr_par_complexd::PolyClass();
   
   template <class MatrixType, class VectorType, typename DataType>
   PolyClass<MatrixType, VectorType, DataType>::PolyClass(const PolyClass<MatrixType, VectorType, DataType> &solver) : SolverClass<MatrixType, VectorType, DataType>(solver)
   {
      this->_order = solver._order;
      this->_scale = solver._scale;
      this->_n = solver._n;
      this->_location = solver._location;
      this->_x_temp = solver._x_temp;
      this->_y_temp = solver._y_temp;
   }
   template precond_poly_csr_seq_float::PolyClass(const precond_poly_csr_seq_float &solver);
   template precond_poly_csr_seq_double::PolyClass(const precond_poly_csr_seq_double &solver);
   template precond_poly_csr_seq_complexs::PolyClass(const precond_poly_csr_seq_complexs &solver);
   template precond_poly_csr_seq_complexd::PolyClass(const precond_poly_csr_seq_complexd &solver);
   template precond_poly_csr_par_float::PolyClass(const precond_poly_csr_par_float &solver);
   template precond_poly_csr_par_double::PolyClass(const precond_poly_csr_par_double &solver);
   template precond_poly_csr_par_complexs::PolyClass(const precond_poly_csr_par_complexs &solver);
   template precond_poly_csr_par_complexd::PolyClass(const precond_poly_csr_par_complexd &solver);
   
   template <class MatrixType, class VectorType, typename DataType>
   PolyClass<MatrixType, VectorType, DataType>::PolyClass( PolyClass<MatrixType, VectorType, DataType> &&solver) : SolverClass<MatrixType, VectorType, DataType>(std::move(solver))
   {
      this->_order = solver._order;
      solver._order = 0;
      this->_scale = solver._scale;
      solver._scale = 0.15;
      this->_n = solver._n;
      solver._n = 0;
      this->_location = solver._location;
      solver._location = kMemoryHost;
      this->_x_temp = std::move(solver._x_temp);
      this->_y_temp = std::move(solver._y_temp);
   }
   template precond_poly_csr_seq_float::PolyClass( precond_poly_csr_seq_float &&solver);
   template precond_poly_csr_seq_double::PolyClass( precond_poly_csr_seq_double &&solver);
   template precond_poly_csr_seq_complexs::PolyClass( precond_poly_csr_seq_complexs &&solver);
   template precond_poly_csr_seq_complexd::PolyClass( precond_poly_csr_seq_complexd &&solver);
   template precond_poly_csr_par_float::PolyClass( precond_poly_csr_par_float &&solver);
   template precond_poly_csr_par_double::PolyClass( precond_poly_csr_par_double &&solver);
   template precond_poly_csr_par_complexs::PolyClass( precond_poly_csr_par_complexs &&solver);
   template precond_poly_csr_par_complexd::PolyClass( precond_poly_csr_par_complexd &&solver);
   
   template <class MatrixType, class VectorType, typename DataType>
   PolyClass<MatrixType, VectorType, DataType> &PolyClass<MatrixType, VectorType, DataType>::operator= (const PolyClass<MatrixType, VectorType, DataType> &solver)
   {
      this->Clear();
      SolverClass<MatrixType, VectorType, DataType>::operator=(solver);
      this->_order = solver._order;
      this->_scale = solver._scale;
      this->_n = solver._n;
      this->_location = solver._location;
      this->_x_temp = solver._x_temp;
      this->_y_temp = solver._y_temp;
      return *this;
   }
   template precond_poly_csr_seq_float& precond_poly_csr_seq_float::operator= (const precond_poly_csr_seq_float &solver);
   template precond_poly_csr_seq_double& precond_poly_csr_seq_double::operator= (const precond_poly_csr_seq_double &solver);
   template precond_poly_csr_seq_complexs& precond_poly_csr_seq_complexs::operator= (const precond_poly_csr_seq_complexs &solver);
   template precond_poly_csr_seq_complexd& precond_poly_csr_seq_complexd::operator= (const precond_poly_csr_seq_complexd &solver);
   template precond_poly_csr_par_float& precond_poly_csr_par_float::operator= (const precond_poly_csr_par_float &solver);
   template precond_poly_csr_par_double& precond_poly_csr_par_double::operator= (const precond_poly_csr_par_double &solver);
   template precond_poly_csr_par_complexs& precond_poly_csr_par_complexs::operator= (const precond_poly_csr_par_complexs &solver);
   template precond_poly_csr_par_complexd& precond_poly_csr_par_complexd::operator= (const precond_poly_csr_par_complexd &solver);
   
   template <class MatrixType, class VectorType, typename DataType>
   PolyClass<MatrixType, VectorType, DataType> &PolyClass<MatrixType, VectorType, DataType>::operator= ( PolyClass<MatrixType, VectorType, DataType> &&solver)
   {
      this->Clear();
      SolverClass<MatrixType, VectorType, DataType>::operator=(std::move(solver));
      this->_order = solver._order;
      solver._order = 0;
      this->_scale = solver._scale;
      solver._scale = 0.15;
      this->_n = solver._n;
      solver._n = 0;
      this->_location = solver._location;
      solver._location = kMemoryHost;
      this->_x_temp = std::move(solver._x_temp);
      this->_y_temp = std::move(solver._y_temp);
      return *this;
   }
   template precond_poly_csr_seq_float& precond_poly_csr_seq_float::operator= ( precond_poly_csr_seq_float &&solver);
   template precond_poly_csr_seq_double& precond_poly_csr_seq_double::operator= ( precond_poly_csr_seq_double &&solver);
   template precond_poly_csr_seq_complexs& precond_poly_csr_seq_complexs::operator= ( precond_poly_csr_seq_complexs &&solver);
   template precond_poly_csr_seq_complexd& precond_poly_csr_seq_complexd::operator= ( precond_poly_csr_seq_complexd &&solver);
   template precond_poly_csr_par_float& precond_poly_csr_par_float::operator= ( precond_poly_csr_par_float &&solver);
   template precond_poly_csr_par_double& precond_poly_csr_par_double::operator= ( precond_poly_csr_par_double &&solver);
   template precond_poly_csr_par_complexs& precond_poly_csr_par_complexs::operator= ( precond_poly_csr_par_complexs &&solver);
   template precond_poly_csr_par_complexd& precond_poly_csr_par_complexd::operator= ( precond_poly_csr_par_complexd &&solver);
   
   template <class MatrixType, class VectorType, typename DataType>
   PolyClass<MatrixType, VectorType, DataType>::~PolyClass()
   {
      this->Clear();
   }
   template precond_poly_csr_seq_float::~PolyClass();
   template precond_poly_csr_seq_double::~PolyClass();
   template precond_poly_csr_seq_complexs::~PolyClass();
   template precond_poly_csr_seq_complexd::~PolyClass();
   template precond_poly_csr_par_float::~PolyClass();
   template precond_poly_csr_par_double::~PolyClass();
   template precond_poly_csr_par_complexs::~PolyClass();
   template precond_poly_csr_par_complexd::~PolyClass();

   template <class MatrixType, class VectorType, typename DataType>
   int PolyClass<MatrixType, VectorType, DataType>::Clear()
   {
      SolverClass<MatrixType, VectorType, DataType>::Clear();
      this->_order = 2;
      this->_scale = 0.0;
      this->_n = 0;
      this->_location = kMemoryHost;
      this->_x_temp.Clear();
      this->_y_temp.Clear();
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_poly_csr_seq_float::Clear();
   template int precond_poly_csr_seq_double::Clear();
   template int precond_poly_csr_seq_complexs::Clear();
   template int precond_poly_csr_seq_complexd::Clear();
   template int precond_poly_csr_par_float::Clear();
   template int precond_poly_csr_par_double::Clear();
   template int precond_poly_csr_par_complexs::Clear();
   template int precond_poly_csr_par_complexd::Clear();
   
   template <class MatrixType, class VectorType, typename DataType>
   int PolyClass<MatrixType, VectorType, DataType>::Setup( VectorType &x, VectorType &rhs)
   {
      /* Wrapper of the setup phase of ilu */
      if(this->_ready)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      if(!this->_matrix)
      {
         PARGEMSLR_ERROR("Setup without matrix.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }

      if(this->_print_option > 0)
      {
         PARGEMSLR_PRINT("Using Poly order %d\n",this->_order);
      }

      /* define the data type */
      typedef DataType T;
      T* temp_data;
      
      /* necessary info */
      this->_n = this->_matrix->GetNumRowsLocal();
      
      /* the the preconditioner is ready, move to location */
      this->_ready = true;
      this->_x_temp.SetupPtrStr(*(this->_matrix));
      PARGEMSLR_CALLOC(temp_data, this->_n, this->_location, T);
      this->_x_temp.UpdatePtr(temp_data, this->_location);
      this->_x_temp.SetHoldingData(true);
      
      this->_y_temp.SetupPtrStr(*(this->_matrix));
      PARGEMSLR_CALLOC(temp_data, this->_n, this->_location, T);
      this->_y_temp.UpdatePtr(temp_data, this->_location);
      this->_y_temp.SetHoldingData(true);
      
      this->_solver_precision = x.GetPrecision();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int precond_poly_csr_seq_float::Setup( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_poly_csr_seq_double::Setup( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_poly_csr_seq_complexs::Setup( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_poly_csr_seq_complexd::Setup( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   template int precond_poly_csr_par_float::Setup( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs);
   template int precond_poly_csr_par_double::Setup( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
   template int precond_poly_csr_par_complexs::Setup( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs);
   template int precond_poly_csr_par_complexd::Setup( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int PolyClass<MatrixType, VectorType, DataType>::Solve( VectorType &x, VectorType &rhs)
   {
      /* the solve phase of ilut */
      PARGEMSLR_CHKERR(this->_n != x.GetLengthLocal() || this->_n != rhs.GetLengthLocal());
      
      if(!(this->_ready))
      {
         PARGEMSLR_ERROR("Solve without setup.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      if(this->_n == 0)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      /* The solve phase, note that we are computing x = (I + (I + (I + B...)B)B)rhs = I + B + B^2 + B^3 + ... where B = I-A
       * Let's use x = (I + (I + (I + B)B)B)rhs as an example.
       * 1. x = rhs + (I + (I + B)B)Brhs => x_out = rhs, x_temp = B*rhs
       *    now, x = x_out + (I + (I + B)B)x_temp
       * 2. x = x_out + x_temp + (I + B)Bx_temp => x_out += x_temp, x_temp = Bx_temp
       *    now, x = x_out + (I + B)x_temp
       * 3. x = x_out + x_temp + Bx_temp => x_out += x_temp, x_temp = Bx_temp
       * 4. finally, x_out += x_temp
       * Finally, note that we are compting inv(scale * A), the final result need to be scaled back
       */
      
      /* define the data type */
      typedef DataType T;
      
      int   i, j, n;
      T     one, zero, mscale;
      T     *rhs_a;
      T     *x_temp, *y_temp;
      
      mscale = -(this->_scale);
      one = 1.0;
      zero = 0.0;
      
      n = this->_n;
      rhs_a = rhs.GetData();
      x_temp = this->_x_temp.GetData();
      y_temp = this->_y_temp.GetData();
      
      /* copy rhs to x_temp first */
#ifdef PARGEMSLR_CUDA
      if(this->_location == kMemoryDevice || this->_location == kMemoryUnified)
      {
         /* for unified memory, use it on device */
         PARGEMSLR_MEMCPY(x_temp, rhs_a, n, kMemoryDevice, rhs.GetDataLocation(), T);
      }
      else
      {
#endif
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
         for(i = 0 ; i < n ; i ++)
         {
            x_temp[i] = rhs_a[i];
         }
#ifdef PARGEMSLR_CUDA
      }
#endif
      x.Fill(zero);
      for(j = 0 ; j < this->_order ; j ++)
      {
         /* first, x += x_temp */
         x.Axpy( one, this->_x_temp);
         
         /* copy x_temp to y_temp */
#ifdef PARGEMSLR_CUDA
         if(this->_location == kMemoryDevice || this->_location == kMemoryUnified)
         {
            /* for unified memory, use it on device */
            PARGEMSLR_MEMCPY(y_temp, x_temp, n, kMemoryDevice, kMemoryDevice, T);
         }
         else
         {
#endif
#ifdef PARGEMSLR_OPENMP
#pragma omp parallel for private(i) PARGEMSLR_OPENMP_SCHEDULE_DEFAULT
#endif
               for(i = 0 ; i < n ; i ++)
               {
                  y_temp[i] = x_temp[i];
               }
#ifdef PARGEMSLR_CUDA
         }
#endif
         
         /* x_temp = B*y_temp = (I-A)*y_temp = -A*y_temp + x_temp */
         this->_matrix->MatVec( 'N', mscale, this->_y_temp, one, this->_x_temp);
      }
      
      /* finally, the last x += x_temp */
      x.Axpy( one, this->_x_temp);
      
      /* scale back, we are computing inv(scale A) */
      x.Scale(this->_scale);
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_poly_csr_seq_float::Solve( SequentialVectorClass<float> &x, SequentialVectorClass<float> &rhs);
   template int precond_poly_csr_seq_double::Solve( SequentialVectorClass<double> &x, SequentialVectorClass<double> &rhs);
   template int precond_poly_csr_seq_complexs::Solve( SequentialVectorClass<complexs> &x, SequentialVectorClass<complexs> &rhs);
   template int precond_poly_csr_seq_complexd::Solve( SequentialVectorClass<complexd> &x, SequentialVectorClass<complexd> &rhs);
   template int precond_poly_csr_par_float::Solve( ParallelVectorClass<float> &x, ParallelVectorClass<float> &rhs);
   template int precond_poly_csr_par_double::Solve( ParallelVectorClass<double> &x, ParallelVectorClass<double> &rhs);
   template int precond_poly_csr_par_complexs::Solve( ParallelVectorClass<complexs> &x, ParallelVectorClass<complexs> &rhs);
   template int precond_poly_csr_par_complexd::Solve( ParallelVectorClass<complexd> &x, ParallelVectorClass<complexd> &rhs);
   
   template <class MatrixType, class VectorType, typename DataType>
   int PolyClass<MatrixType, VectorType, DataType>::GetSize()
   {
      return this->_n;
   }
   template int precond_poly_csr_seq_float::GetSize();
   template int precond_poly_csr_seq_double::GetSize();
   template int precond_poly_csr_seq_complexs::GetSize();
   template int precond_poly_csr_seq_complexd::GetSize();
   template int precond_poly_csr_par_float::GetSize();
   template int precond_poly_csr_par_double::GetSize();
   template int precond_poly_csr_par_complexs::GetSize();
   template int precond_poly_csr_par_complexd::GetSize();
   
   template <class MatrixType, class VectorType, typename DataType>
   long int PolyClass<MatrixType, VectorType, DataType>::GetNumNonzeros()
   {
      return 0;
   }
   template long int precond_poly_csr_seq_float::GetNumNonzeros();
   template long int precond_poly_csr_seq_double::GetNumNonzeros();
   template long int precond_poly_csr_seq_complexs::GetNumNonzeros();
   template long int precond_poly_csr_seq_complexd::GetNumNonzeros();
   template long int precond_poly_csr_par_float::GetNumNonzeros();
   template long int precond_poly_csr_par_double::GetNumNonzeros();
   template long int precond_poly_csr_par_complexs::GetNumNonzeros();
   template long int precond_poly_csr_par_complexd::GetNumNonzeros();
   
   template <class MatrixType, class VectorType, typename DataType>
   int PolyClass<MatrixType, VectorType, DataType>::SetSolveLocation( const int &location)
   {
      this->_location = location;
      
      if(this->_ready)
      {
         this->MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_poly_csr_seq_float::SetSolveLocation( const int &location);
   template int precond_poly_csr_seq_double::SetSolveLocation( const int &location);
   template int precond_poly_csr_seq_complexs::SetSolveLocation( const int &location);
   template int precond_poly_csr_seq_complexd::SetSolveLocation( const int &location);
   template int precond_poly_csr_par_float::SetSolveLocation( const int &location);
   template int precond_poly_csr_par_double::SetSolveLocation( const int &location);
   template int precond_poly_csr_par_complexs::SetSolveLocation( const int &location);
   template int precond_poly_csr_par_complexd::SetSolveLocation( const int &location);

   template <class MatrixType, class VectorType, typename DataType>
   int PolyClass<MatrixType, VectorType, DataType>::MoveData( const int &location)
   {
      this->_location = location;
      
      if(this->_ready)
      {
         this->_x_temp.MoveData(location);
         this->_y_temp.MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int precond_poly_csr_seq_float::MoveData( const int &location);
   template int precond_poly_csr_seq_double::MoveData( const int &location);
   template int precond_poly_csr_seq_complexs::MoveData( const int &location);
   template int precond_poly_csr_seq_complexd::MoveData( const int &location);
   template int precond_poly_csr_par_float::MoveData( const int &location);
   template int precond_poly_csr_par_double::MoveData( const int &location);
   template int precond_poly_csr_par_complexs::MoveData( const int &location);
   template int precond_poly_csr_par_complexd::MoveData( const int &location);
   
}
