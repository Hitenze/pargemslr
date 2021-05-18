#ifndef PARGEMSLR_FGMRES_H
#define PARGEMSLR_FGMRES_H

/**
 * @file fgmres.hpp
 * @brief The header of the Flex GMRES solve.
 */

#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../vectors/vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../matrices/matrix.hpp"
#include "../matrices/matrixops.hpp"
#include "../matrices/dense_matrix.hpp"
#include "solver.hpp"

namespace pargemslr
{
   
   /**
    * @brief   The real flexgmres solver class.
    * @details The real flexgmres solver class. MatrixType is the type of the matrix, VectorType is the type of the vector, DataType is the data type. \n
    *          MatrixType required function: SetupVectorPtrStr, Matvec, GetMpiInfo (for PARGEMSLR_TIMING).
    */
   template <class MatrixType, class VectorType, typename DataType>
   class FlexGmresClass: public SolverClass<MatrixType, VectorType, DataType>
   {
   private:

      /**
       * @brief   Tolorance for stopping the iteration. 
       * @details Tolorance for stopping the iteration. 
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _tol;
      
      /**
       * @brief   The maximum number of iterations.
       * @details The maximum number of iterations.
       */
      int                              _maxits;
      
      /**
       * @brief   Maximum number of flexible GMRES inner iterations per cycle.
       * @details Maximum number of flexible GMRES inner iterations per cycle.
       */
      int                              _kdim;
      
      /**
       * @brief   Do we use absolute tolorance? If so, stop when ||r|| < tol. Otherwise stop when ||r||/||r0|| < tol.
       * @details Do we use absolute tolorance? If so, stop when ||r|| < tol. Otherwise stop when ||r||/||r0|| < tol.
       */
      bool                             _absolute_tol;
      
      /**
       * @brief   The final relative residual.
       * @details The final relative residual.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type _rel_res;
      
      /**
       * @brief   The number of iterations.
       * @details The number of iterations.
       */
      int                              _iter;
      
      /**
       * @brief   Relative residual at each step.
       * @details Relative residual at each step.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 vector_seq_double, 
                                 vector_seq_float>::type  _rel_res_vector;
      
      /**
       * @brief   The location this solver applied to.
       * @details The location this solver applied to.
       */
      int                              _location;
      
   public:
      
      /**
       * @brief   The constructor.
       * @details The constructor.
       */
      FlexGmresClass() : SolverClass<MatrixType, VectorType, DataType>()
      {
         this->_solver_type = kSolverFgmres;
         this->_location = kMemoryHost;
         this->_kdim = 50;
         this->_maxits = 200;
         this->_tol = 1e-08;
         this->_absolute_tol = false;
         this->_rel_res = 0.0;
         this->_iter = 0;
      }
      
      /**
       * @brief   The copy constructor.
       * @details The copy constructor.
       * @param   [in]   solver The solver.
       */
      FlexGmresClass(const FlexGmresClass<MatrixType, VectorType, DataType> &solver) : SolverClass<MatrixType, VectorType, DataType>(solver)
      {
         this->_location = solver._location;
         this->_kdim = solver._kdim;
         this->_maxits = solver._maxits;
         this->_tol = solver._tol;
         this->_absolute_tol = solver._absolute_tol;
         this->_rel_res = solver._rel_res;
         this->_iter = solver._iter;
         this->_rel_res_vector = solver._rel_res_vector;
      }
      
      /**
       * @brief   The move constructor.
       * @details The move constructor.
       * @param   [in]   solver The solver.
       */
      FlexGmresClass(FlexGmresClass<MatrixType, VectorType, DataType> &&solver) : SolverClass<MatrixType, VectorType, DataType>(std::move(solver))
      {
         this->_location = solver._location;
         solver._location = kMemoryHost;
         this->_kdim = solver._kdim;
         solver._kdim = 0;
         this->_maxits = solver._maxits;
         solver._maxits = 0;
         this->_tol = solver._tol;
         solver._tol = 0;
         this->_absolute_tol = solver._absolute_tol;
         solver._absolute_tol = false;
         this->_rel_res = solver._rel_res;
         solver._rel_res = 0;
         this->_iter = solver._iter;
         solver._iter = 0;
         this->_rel_res_vector = std::move(solver._rel_res_vector);
      }
      
      /**
       * @brief   The = operator.
       * @details The = operator.
       * @param   [in]   solver The solver.
       * @return     Return the solver.
       */
      FlexGmresClass<MatrixType, VectorType, DataType>& operator=(const FlexGmresClass<MatrixType, VectorType, DataType> &solver)
      {
         this->Clear();
         SolverClass<MatrixType, VectorType, DataType>::operator=(solver);
         this->_location = solver._location;
         this->_kdim = solver._kdim;
         this->_maxits = solver._maxits;
         this->_tol = solver._tol;
         this->_absolute_tol = solver._absolute_tol;
         this->_rel_res = solver._rel_res;
         this->_iter = solver._iter;
         this->_rel_res_vector = solver._rel_res_vector;
         return *this;
      }
      
      /**
       * @brief   The = operator.
       * @details The = operator.
       * @param   [in]   solver The solver.
       * @return     Return the solver.
       */
      FlexGmresClass<MatrixType, VectorType, DataType>& operator=(FlexGmresClass<MatrixType, VectorType, DataType> &&solver)
      {
         this->Clear();
         SolverClass<MatrixType, VectorType, DataType>::operator=(std::move(solver));
         this->_location = solver._location;
         solver._location = kMemoryHost;
         this->_kdim = solver._kdim;
         solver._kdim = 0;
         this->_maxits = solver._maxits;
         solver._maxits = 0;
         this->_tol = solver._tol;
         solver._tol = 0;
         this->_absolute_tol = solver._absolute_tol;
         solver._absolute_tol = false;
         this->_rel_res = solver._rel_res;
         solver._rel_res = 0;
         this->_iter = solver._iter;
         solver._iter = 0;
         this->_rel_res_vector = std::move(solver._rel_res_vector);
         return *this;
      }
      
      /**
       * @brief   Free function.
       * @details Free function.
       * @return     Return error message.
       */
      int Clear()
      {
         /* call base clear function */
         SolverClass<MatrixType, VectorType, DataType>::Clear();
         
         this->_location = kMemoryHost;
         
         _rel_res = 0.0;
         _iter = 0;
         _rel_res_vector.Clear();
         
         return PARGEMSLR_SUCCESS;
         
      }
      
      /**
       * @brief   The destructor.
       * @details The destructor.
       */
      ~FlexGmresClass()
      {
         this->Clear();
      }
      
      /**
       * @brief   Setup the solver phase, include building the preconditioner, call this function before Solve, after SetPreconditioner.
       * @details Setup the solver phase, include building the preconditioner, call this function before Solve, after SetPreconditioner.
       * @param   [in]   x The initial guess.
       * @param   [in]   rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Setup( VectorType &x, VectorType &rhs)
      {
         
         if(this->_ready)
         {
            /* don't need to setup if already done so */
            return PARGEMSLR_SUCCESS;
         }
         
         /* for gmres we only need to setup the preconditioner */
         if(this->_preconditioner)
         {
            this->_preconditioner->SetSolveLocation(this->_location);
            this->_preconditioner->Setup( x, rhs);
         }
         
         this->_solution = &x;
         this->_right_hand_side = &rhs;
         
         this->_solver_precision = x.GetPrecision();
         
         this->_ready = true;
         
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Solve phase. Call this function after Setup.
       * @details Solve phase. Call this function after Setup.
       * @param   [in,out] x The initial guess.
       * @param   [in]     rhs The right-hand-side.
       * @return     Return error message.
       */
      virtual int Solve( VectorType &x, VectorType &rhs)
      {
         if(!(this->_ready))
         {
            PARGEMSLR_ERROR("Solve without setup.");
            return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
         }
         
         /* define the data type */
         typedef typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, double, float>::type RealDataType;
         typedef DataType T;
         
         /* Declare variables */
         int            n_local, i, j, k;
         T              one, zero, mone;
         RealDataType   normb, EPSILON, normr, tolr, t, gam;
         T              inormr, hii, hii1;

#ifdef PARGEMSLR_TIMING
         int np, myid;
         MPI_Comm comm;
         
         this->_matrix->GetMpiInfo(np, myid, comm);
#endif

         SequentialVectorClass<T>      c, s, rs;
         VectorType                    v, z, w;
         DenseMatrixClass<T>           V, Z, H;
         
         EPSILON              = std::numeric_limits<RealDataType>::epsilon();
         
         one                  = 1.0;
         zero                 = 0.0;
         mone                 = -1.0;
         
         /* get the local length */
         n_local              = x.GetLengthLocal();
         
         /* Working matrices to hold the Krylov basis and upper-Hessenberg matrices */
         V.Setup(n_local, this->_kdim+1, this->_location, true); 
         Z.Setup(n_local, this->_kdim+1, this->_location, true); 
         H.Setup(this->_kdim+1, this->_kdim, kMemoryHost, true);

         /* Temp vector pointers */
         this->_matrix->SetupVectorPtrStr(v);
         this->_matrix->SetupVectorPtrStr(z);
         this->_matrix->SetupVectorPtrStr(w);
         
         /* Working vectors for Givens rotations */
         c.Setup(this->_kdim, true);
         s.Setup(this->_kdim, true);
         rs.Setup(this->_kdim+1, true);
         
         /* Exit if RHS is zero */
         //rhs.NormInf(normb);
         rhs.Norm2(normb);
         
         /* the solution is direct */
         if (normb < EPSILON) 
         {
            x.Fill(0.0);
            this->_rel_res = 0.0;
            this->_iter = 0;
            this->_rel_res_vector.Setup(1, true);
            return PARGEMSLR_SUCCESS;
         }
         
         /* Make b the first vector of the basis (will normalize later) */
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, 0), V.GetDataLocation());
         }
         
         /* Compute the residual norm v = rhs - A*x */
         PARGEMSLR_MEMCPY(v.GetData(), rhs.GetData(), n_local, v.GetDataLocation(), rhs.GetDataLocation(), T);
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_ITERTIME_AMV, (this->_matrix->MatVec( 'N', mone, x, one, v)));
         
         /* Compute the 2-norm of residual */
         v.Norm2(normr);
         
         /* the solution is direct */
         if (normr < EPSILON) 
         {
            this->_rel_res = 0.0;
            this->_iter = 0;
            this->_rel_res_vector.Setup(1, true);
            return PARGEMSLR_SUCCESS;
         }
         
         if(this->_absolute_tol)
         {
            tolr = this->_tol;
         }
         else
         {
            tolr = this->_tol*normb;
         }
         
         /* A few variables used to keep track of the loop's state */
         this->_rel_res_vector.Setup(this->_maxits+1, true);
         //this->_rel_res_vector[0] = normr;
         this->_rel_res_vector[0] = normr/normb;
         this->_iter = 0;
         
         if(this->_print_option>0 && parallel_log::_grank == 0)
         {
            PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
            PARGEMSLR_PRINT("Start FlexGMRES(%d)\n",this->_kdim);
            PARGEMSLR_PRINT("Residual Tol: %e\nMax number of inner iterations: %d\n", tolr, this->_maxits);
            PargemslrPrintDashLine(pargemslr::pargemslr_global::_dash_line_width);
            PARGEMSLR_PRINT("Step    Residual norm  Relative res.  Convergence Rate\n");
            //PARGEMSLR_PRINT("%5d   %8e   %8e   N/A\n", 0, normr, 1.0);
            PARGEMSLR_PRINT("%5d   %8e   %8e   N/A\n", 0, normr, this->_rel_res_vector[0]);
         }
         
         /* Outer loop */
         while (this->_iter < this->_maxits) 
         {
            /* Scale the starting vector */
            rs[0] = normr;
            inormr = 1.0/normr;
            
            v.Scale(inormr);
            
            // Inner loop
            i = 0;
            
            while (i < this->_kdim && this->_iter < this->_maxits) 
            {
               i++;
               this->_iter++;
               if( n_local > 0)
               {
                  v.UpdatePtr( &V(0, i-1), V.GetDataLocation() );
                  z.UpdatePtr( &Z(0, i-1), Z.GetDataLocation() );
                  w.UpdatePtr( &V(0, i), V.GetDataLocation() );
               }
               
               /* zi = M^{-1} * vi -- apply the preconditioner */
               if(this->_preconditioner)
               {
#ifdef PARGEMSLR_TIMING
                  PARGEMSLR_TIME_CALL( comm, PARGEMSLR_PRECTIME_PRECOND, (this->_preconditioner->Solve(z, v)));
                  PARGEMSLR_TIME_CALL( comm, PARGEMSLR_ITERTIME_AMV, (this->_matrix->MatVec( 'N', one, z, zero, w)));
#else
                  this->_preconditioner->Solve(z, v);
                  this->_matrix->MatVec( 'N', one, z, zero, w);
#endif
               }
               else
               {
                  PARGEMSLR_MEMCPY(z.GetData(), v.GetData(), n_local, z.GetDataLocation(), v.GetDataLocation(), T);
#ifdef PARGEMSLR_TIMING
                  PARGEMSLR_TIME_CALL( comm, PARGEMSLR_ITERTIME_AMV, (this->_matrix->MatVec( 'N', one, v, zero, w)));
#else
                  this->_matrix->MatVec( 'N', one, v, zero, w);
#endif
               }
               
               /* Modified Gram-schmidt without re-orth */
#ifdef PARGEMSLR_TIMING
               PARGEMSLR_TIME_CALL( comm, PARGEMSLR_ITERTIME_MGS, (PargemslrMgs( w, V, H, t, i-1, RealDataType(1e-12), RealDataType(-1.0))));
#else
               PargemslrMgs( w, V, H, t, i-1, RealDataType(1e-12), RealDataType(-1.0));
#endif
               
               if (PargemslrAbs(t) < EPSILON) 
               {
                  if(this->_print_option>0 && parallel_log::_grank == 0)
                  {
                     PARGEMSLR_PRINT("Break down in the current cycle\n");
                  }
                  goto label;
               }
               /* Scale w (t=||w|| is already computed in the mgs function) */
               //scale(w, 1.0 / t);

               /* Least squares problem of H */
               for (j = 1; j < i; j++) 
               {
                  hii = H(j-1,i-1);
                  H(j-1,i-1) = PargemslrConj(c[j-1])*hii + s[j-1]*H(j,i-1);
                  H(j,i-1) = -s[j-1]*hii + c[j-1]*H(j,i-1);
               }
               
               hii = H(i-1,i-1);
               hii1 = H(i,i-1);
               gam = sqrt(PargemslrAbs(hii)*PargemslrAbs(hii) + PargemslrAbs(hii1)*PargemslrAbs(hii1));
               if (PargemslrAbs(gam) < EPSILON) 
               {
                  goto label;
               }
               c[i-1] = hii / gam;
               s[i-1] = hii1 / gam;
               rs[i] = -s[i-1] * rs[i-1];
               rs[i-1] = PargemslrConj(c[i-1]) * rs[i-1];
               /* Check residual norm */
               H(i-1,i-1) = PargemslrConj(c[i-1])*hii + s[i-1]*hii1;
               normr = PargemslrAbs(rs[i]);
               //this->_rel_res_vector[this->_iter] = normr;
               this->_rel_res_vector[this->_iter] = normr/normb;
               if(this->_print_option>0 && parallel_log::_grank == 0)
               {
                  //PARGEMSLR_PRINT("%5d   %8e   %8e   %8.6f\n", this->_iter, normr, normr / this->_rel_res_vector[0], normr / this->_rel_res_vector[this->_iter-1]);
                  PARGEMSLR_PRINT("%5d   %8e   %8e   %8.6f\n", this->_iter, normr, this->_rel_res_vector[this->_iter], this->_rel_res_vector[this->_iter] / this->_rel_res_vector[this->_iter-1]);
               }
               if (normr <= tolr) 
               {
                  break;
               }
            } // End of inner loop
            
            /* Print residual norm at the end of each cycle */
            if (this->_print_option==0 && parallel_log::_grank == 0)
            {
               //PARGEMSLR_PRINT("Rel. residual at the end of current cycle (# of steps per cycle: %d): %e \n", this->_kdim, normr / this->_rel_res_vector[0]);
               PARGEMSLR_PRINT("Rel. residual at the end of current cycle (# of steps per cycle: %d): %e \n", this->_kdim, this->_rel_res_vector[0]);
            }
            
            /* Solve the upper triangular system */
            rs[i-1] /= H(i-1,i-1);
            for ( k = i-2; k >= 0; k--) 
            {
               for ( j = k+1; j < i; j++) 
               {
                  rs[k] -= H(k,j)*rs[j];
               }
               rs[k] /= H(k,k);
            }

            /* Get current approximate solution */
            for ( j = 0; j < i; j++) 
            {
               if(n_local > 0)
               {
                  z.UpdatePtr( &Z(0, j), Z.GetDataLocation() );
               }
               x.Axpy( rs[j], z);
            }
            
            /* Test convergence */
            if (normr <= tolr) 
            {
               this->_rel_res = normr;
               break;
            }

            PARGEMSLR_CHKERR(i==0 && this->_iter != this->_maxits);

            /* Restart */
            
            if(n_local)
            {
               v.UpdatePtr( &V(0, 0), V.GetDataLocation() );
            }
            PARGEMSLR_MEMCPY(v.GetData(), rhs.GetData(), n_local, v.GetDataLocation(), rhs.GetDataLocation(), T);
#ifdef PARGEMSLR_TIMING
            PARGEMSLR_TIME_CALL( comm, PARGEMSLR_ITERTIME_AMV, (this->_matrix->MatVec( 'N', one, x, zero, w)) );
#else
            this->_matrix->MatVec( 'N', one, x, zero, w);
#endif            

            v.Axpy(mone, w);
            
            /*
            for ( j = i; j > 0; j--) 
            {
               rs[j-1] = -s[j-1] * rs[j];
               rs[j] = c[j-1] * rs[j];
            }

            if(i) 
            {
               if(n_local > 0)
               {
                  v.UpdatePtr( &V(0, 0), V.GetDataLocation() );
                  v.Scale(rs[0]);
                  for ( j = 1 ; j <= i; j++) 
                  {
                     w.UpdatePtr( &V(0, j), V.GetDataLocation() );
                     v.Axpy(rs[j], w);
                  }
               }
            }
            */
            
         } // End of outer loop

   label:
         //this->_rel_res = normr / this->_rel_res_vector[0];
         this->_rel_res = normr / normb;
         
         this->_rel_res_vector.Resize( this->_iter+1, true, false);

         /* De-allocate */
         c.Clear();
         s.Clear();
         rs.Clear();
         V.Clear();
         Z.Clear();
         H.Clear();
         v.Clear();
         z.Clear();
         w.Clear();
         
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief      Set the data location that the solver apply to.
       * @details    Set the data location that the solver apply to.
       * @param [in] location The target solver location.
       * @return     Return error message.
       */
      virtual int SetSolveLocation( const int &location)
      {
         if(this->_preconditioner)
         {
            this->_preconditioner->SetSolveLocation(location);
            this->_location = location;
            return PARGEMSLR_SUCCESS;
         }
         
         if(this->_location == location)
         {
            return PARGEMSLR_SUCCESS;
         }
         
         this->_location = location;
         
         return PARGEMSLR_SUCCESS;
         
      }
      
      /* ------ SETS and GETS ------ */
      
      /**
       * @brief   Setup with parameter array.
       * @details Setup with parameter array. \n
       *          Entry used: PARGEMSLR_IO_SOLVER_TOL, PARGEMSLR_IO_SOLVER_MAXITS, PARGEMSLR_IO_SOLVER_KDIM, and PARGEMSLR_IO_GENERAL_PRINT_LEVEL.
       * @param [in] params The parameter array.
       * @return     Return error message.
       */
      virtual int SetWithParameterArray(double *params)
      {
         this->_tol = params[PARGEMSLR_IO_SOLVER_TOL];
         this->_maxits = params[PARGEMSLR_IO_SOLVER_MAXITS];
         this->_kdim = params[PARGEMSLR_IO_SOLVER_KDIM];
         this->_absolute_tol = params[PARGEMSLR_IO_SOLVER_ATOL] != 0.0;
         this->_print_option = params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL];
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the tolorance for stop the iteration.
       * @details Set the tolorance for stop the iteration.
       * @param   [in]     tol The tolorance for stopping the iteration.
       * @return     Return error message.
       */
      template <typename T>
      int         SetTolerance(T tol)
      {
         this->_tol = tol;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the maximum number of iterations.
       * @details Set the maximum number of iterations.
       * @param   [in]     maxits The maximum number of iterations.
       * @return     Return error message.
       */
      int         SetMaxNumberIterations(int maxits)
      {
         this->_maxits = maxits;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set the maximum number of flexible GMRES inner iterations per cycle.
       * @details Set the maximum number of flexible GMRES inner iterations per cycle.
       * @param   [in]     maxits The maximum number of flexible GMRES inner iterations per cycle.
       * @return     Return error message.
       */
      int         SetKrylovSubspaceDimension(int kdim)
      {
         this->_kdim = kdim;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Set if we use absolute threshold ||r|| or ||r||/||b||.
       * @details Set if we use absolute threshold ||r|| or ||r||/||b||.
       * @param   [in]     option The bool of new option.
       * @return     Return error message.
       */
      int         SetAbsoluteTol(bool option)
      {
         this->_absolute_tol = option;
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Get the final relative residual.
       * @details Get the final relative residual.
       * @return     Return the final relative residual.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 double, 
                                 float>::type GetFinalRelativeResidual() const
      {
         return this->_rel_res;
      }
      
      /**
       * @brief   Get the number of iterations.
       * @details Get the number of iterations.
       * @return     Return the number of iterations.
       */
      int         GetNumberIterations() const
      {
         return this->_iter;
      }
      
      /**
       * @brief   Get the relative residual at each step as a vector.
       * @details Get the relative residual at each step as a vector.
       * @return     Return the relative residual at each step as a vector.
       */
      typename std::conditional<PargemslrIsDoublePrecision<DataType>::value, 
                                 vector_seq_double, 
                                 vector_seq_float>::type& GetRelativeResidual()
      {
         return this->_rel_res_vector;
      }
      
   };
   
   typedef FlexGmresClass<CsrMatrixClass<float>, SequentialVectorClass<float>, float>                                   fgmres_csr_seq_float;
   typedef FlexGmresClass<CsrMatrixClass<double>, SequentialVectorClass<double>, double>                                fgmres_csr_seq_double;
   typedef FlexGmresClass<CsrMatrixClass<complexs>, SequentialVectorClass<complexs>, complexs>                          fgmres_csr_seq_complexs;
   typedef FlexGmresClass<CsrMatrixClass<complexd>, SequentialVectorClass<complexd>, complexd>                          fgmres_csr_seq_complexd;
   typedef FlexGmresClass<ParallelCsrMatrixClass<float>, ParallelVectorClass<float>, float>                             fgmres_csr_par_float;
   typedef FlexGmresClass<ParallelCsrMatrixClass<double>, ParallelVectorClass<double>, double>                          fgmres_csr_par_double;
   typedef FlexGmresClass<ParallelCsrMatrixClass<complexs>, ParallelVectorClass<complexs>, complexs>                    fgmres_csr_par_complexs;
   typedef FlexGmresClass<ParallelCsrMatrixClass<complexd>, ParallelVectorClass<complexd>, complexd>                    fgmres_csr_par_complexd;
   
}

#endif
