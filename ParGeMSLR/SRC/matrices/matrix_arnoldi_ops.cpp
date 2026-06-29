#include <unordered_map>
#include <unordered_set>
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../utils/memory.hpp"
#include "../utils/protos.hpp"
#include "../vectors/vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "matrix.hpp"
#include "matrixops.hpp"
#include "dense_matrix.hpp"
#include "csr_matrix.hpp"
#include "parallel_csr_matrix.hpp"
#include "../preconditioners/gemslr.hpp"
#include "../preconditioners/parallel_gemslr.hpp"

#include <iostream>
#include <complex>
#include <limits>
#include <limits.h>

#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#endif

using namespace std;

namespace pargemslr
{
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int PargemslrSubSpaceIteration( MatrixType &A, int k, int its, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, RealDataType tol)
   {

      int                  i, j, ii, n;
      DataType             one, zero;

      int                  location = A.GetDataLocation();

      RealDataType         orth_tol = pargemslr_global::_orth_tol;
      RealDataType         reorth_tol = pargemslr_global::_reorth_tol;
      RealDataType         t, normv;

      if(orth_tol < std::numeric_limits<RealDataType>::epsilon())
      {
         /* the tolerance should not be too small */
         orth_tol = std::numeric_limits<RealDataType>::epsilon();
      }

      VectorType           v, w;

      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);

      n = A.GetNumRowsLocal();
      one = 1.0;
      zero = 0.0;

      int np, myid;
      MPI_Comm comm;
      A.GetMpiInfo(np, myid, comm);

      DenseMatrixClass<DataType>                         B, Q1, Q, R, R1;
      void                                               *q_data;
      DenseMatrixClass<RealDataType>                     Qr;
      DenseMatrixClass<ComplexValueClass<RealDataType> > Qc;
      B.Setup(n, k, location, false);
      Q.Setup(n, k, location, false);

      /* R is on host, store the R of QR, not used */
      R.Setup(k, k, kMemoryHost, false);
      R1.SetupPtr(R, 0, 1, k, k-1);

      /* reset the seed */
      pargemslr_global::_mersenne_twister_engine.seed(0);

      B.Rand();

      for(i = 0 ; i < its ; i ++)
      {
         //DenseMatrixQRDecompositionHost(B, Q);
         /* QR decomposition of B */

         /* first normoralize first column in B */
         if( n > 0)
         {
            v.UpdatePtr( &B(0, 0), location );
         }

         v.Norm2(normv);
         v.Scale(one/normv);

         R.Fill(zero);
         R(0,0) = normv;
         for(ii = 0 ; ii < k-1 ; ii ++)
         {
            if( n > 0)
            {
               v.UpdatePtr( &B(0, ii+1), location );
            }
            if(pargemslr_global::_gram_schmidt == 1)
            {
               PARGEMSLR_TIME_CALL(comm, PARGEMSLR_BUILDTIME_MGS, PargemslrMgs( v, B, R1, t, ii, orth_tol, reorth_tol));
            }
            else
            {
               PARGEMSLR_TIME_CALL(comm, PARGEMSLR_BUILDTIME_MGS, PargemslrCgs2( v, B, R1, t, ii, orth_tol));
            }
         }

         PARGEMSLR_MEMCPY( Q.GetData(), B.GetData(), n*k, location, location, DataType);

         /* B = A * Q */
         for(j = 0 ; j < k ; j ++)
         {
            if( n > 0)
            {
               v.UpdatePtr( &Q(0, j), location );
               w.UpdatePtr( &B(0, j), location );
            }
            PARGEMSLR_TIME_CALL(comm, PARGEMSLR_BUILDTIME_EBFC, A.MatVec('N', one, v, zero, w));
         }
      }

      /* H = Q^H * B */
      H.Setup(k, k, location, true);
      H.MatMat( one, Q, 'C', B, 'N', zero);

      if(PargemslrIsComplex<DataType>::value)
      {
         DenseMatrixClass<ComplexValueClass<RealDataType> > H1;
         SequentialVectorClass<ComplexValueClass<RealDataType> > w;

         if(location != kMemoryDevice)
         {
            if(np > 1)
            {
               PARGEMSLR_MPI_CALL( PargemslrMpiAllreduceInplace( H.GetData(), k*k, MPI_SUM, comm) );
            }
            H1.SetupPtr( (ComplexValueClass<RealDataType>*)(H.GetData()), H.GetNumRowsLocal(), H.GetNumColsLocal(), H.GetLeadingDimension(), location);

            /* Schur of H => H1(H) and Q1 */

            H1.Schur(Qc, w);
            w.Clear();

            q_data = (void*)Qc.GetData();

         }
         else
         {
            DenseMatrixClass<DataType> H_host;
            /* copy to host */
            H_host.Setup(k, k);
            PARGEMSLR_MEMCPY( H_host.GetData(), H.GetData(), k*k, kMemoryHost, location, DataType);
            if(np > 1)
            {
               PARGEMSLR_MPI_CALL( PargemslrMpiAllreduceInplace( H_host.GetData(), k*k, MPI_SUM, comm) );
            }
            H1.SetupPtr( (ComplexValueClass<RealDataType>*)(H_host.GetData()), H_host.GetNumRowsLocal(), H_host.GetNumColsLocal(), H_host.GetLeadingDimension(), kMemoryHost);

            /* Schur of H => H1(H) and Q1 */
            H1.Schur(Qc, w);
            w.Clear();

            /* move back to device */
            PARGEMSLR_MEMCPY( H.GetData(), H_host.GetData(), k*k, location, kMemoryHost, DataType);
            H_host.Clear();

            Qc.MoveData(location);

            q_data = (void*)Qc.GetData();

         }

      }
      else
      {
         DenseMatrixClass<RealDataType> H1;
         SequentialVectorClass<RealDataType> wr, wi;

         if(location != kMemoryDevice)
         {
            if(np > 1)
            {
               PARGEMSLR_MPI_CALL( PargemslrMpiAllreduceInplace( H.GetData(), k*k, MPI_SUM, comm) );
            }
            H1.SetupPtr( (RealDataType*)(H.GetData()), H.GetNumRowsLocal(), H.GetNumColsLocal(), H.GetLeadingDimension(), location);

            /* Schur of H => H1(H) and Q1 */

            H1.Schur(Qr, wr, wi);
            wr.Clear();
            wi.Clear();

            q_data = (void*)Qr.GetData();

         }
         else
         {
            DenseMatrixClass<DataType> H_host;
            /* copy to host */
            H_host.Setup(k, k);
            PARGEMSLR_MEMCPY( H_host.GetData(), H.GetData(), k*k, kMemoryHost, location, DataType);
            if(np > 1)
            {
               PARGEMSLR_MPI_CALL( PargemslrMpiAllreduceInplace( H_host.GetData(), k*k, MPI_SUM, comm) );
            }
            H1.SetupPtr( (RealDataType*)(H_host.GetData()), H_host.GetNumRowsLocal(), H_host.GetNumColsLocal(), H_host.GetLeadingDimension(), kMemoryHost);

            /* Schur of H => H1(H) and Q1 */
            H1.Schur(Qr, wr, wi);
            wr.Clear();
            wi.Clear();

            /* move back to device */
            PARGEMSLR_MEMCPY( H.GetData(), H_host.GetData(), k*k, location, kMemoryHost, DataType);
            H_host.Clear();

            Qr.MoveData(location);

            q_data = (void*)Qr.GetData();

         }

      }

      Q1.SetupPtr( (DataType*)(q_data), k, k, k, location);

      /* V = Q*Q1; */
      V.MatMat( one, Q, 'N', Q1, 'N', zero);

      Q1.Clear();
      Qr.Clear();
      Qc.Clear();

      v.Clear();
      w.Clear();
      Q.Clear();
      B.Clear();

      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrSubSpaceIteration<SequentialVectorClass<float> >( precond_gemslrebfc_csr_seq_float &A, int k, int its, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol);
   template int PargemslrSubSpaceIteration<SequentialVectorClass<double> >( precond_gemslrebfc_csr_seq_double &A, int k, int its, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol);
   template int PargemslrSubSpaceIteration<SequentialVectorClass<complexs> >( precond_gemslrebfc_csr_seq_complexs &A, int k, int its, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol);
   template int PargemslrSubSpaceIteration<SequentialVectorClass<complexd> >( precond_gemslrebfc_csr_seq_complexd &A, int k, int its, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol);
   template int PargemslrSubSpaceIteration<ParallelVectorClass<float> >( precond_gemslrebfc_csr_par_float &A, int k, int its, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol);
   template int PargemslrSubSpaceIteration<ParallelVectorClass<double> >( precond_gemslrebfc_csr_par_double &A, int k, int its, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol);
   template int PargemslrSubSpaceIteration<ParallelVectorClass<complexs> >( precond_gemslrebfc_csr_par_complexs &A, int k, int its, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol);
   template int PargemslrSubSpaceIteration<ParallelVectorClass<complexd> >( precond_gemslrebfc_csr_par_complexd &A, int k, int its, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol);
   template int PargemslrSubSpaceIteration<SequentialVectorClass<float> >( matrix_csr_float &A, int k, int its, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol);
   template int PargemslrSubSpaceIteration<SequentialVectorClass<double> >( matrix_csr_double &A, int k, int its, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol);
   template int PargemslrSubSpaceIteration<SequentialVectorClass<complexs> >( matrix_csr_complexs &A, int k, int its, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol);
   template int PargemslrSubSpaceIteration<SequentialVectorClass<complexd> >( matrix_csr_complexd &A, int k, int its, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol);
   template int PargemslrSubSpaceIteration<ParallelVectorClass<float> >( matrix_csr_par_float &A, int k, int its, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol);
   template int PargemslrSubSpaceIteration<ParallelVectorClass<double> >( matrix_csr_par_double &A, int k, int its, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol);
   template int PargemslrSubSpaceIteration<ParallelVectorClass<complexs> >( matrix_csr_par_complexs &A, int k, int its, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol);
   template int PargemslrSubSpaceIteration<ParallelVectorClass<complexd> >( matrix_csr_par_complexd &A, int k, int its, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol);

   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int PargemslrArnoldiNoRestart( MatrixType &A, int mstart, int msteps, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H,
                                 RealDataType tol_orth, RealDataType tol_reorth)
   {

      int                  k, n;
      RealDataType         t;
      DataType             one, zero;

      VectorType           v, w;

      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);

      n = A.GetNumRowsLocal();
      one = 1.0;
      zero = 0.0;

#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      A.GetMpiInfo(np, myid, comm);
#endif

      if(tol_orth < std::numeric_limits<RealDataType>::epsilon())
      {
         /* the tolerance should not be too small */
         tol_orth = std::numeric_limits<RealDataType>::epsilon();
      }

      /*------------------------
       * Start arnoldi loop
       * Compute matvec u = A*v
       * Apply Modified Gram - Schmidt
       *------------------------*/

      for (k = mstart; k < msteps; k++)
      {
         /* set to each column. v is the current row, w is the next row
          * only set is local size is not 0
          */
         if( n > 0)
         {
            v.UpdatePtr( &V(0, k), V.GetDataLocation() );
            w.UpdatePtr( &V(0, k+1), V.GetDataLocation() );
         }

#ifdef PARGEMSLR_TIMING
         PARGEMSLR_TIME_CALL(comm, PARGEMSLR_BUILDTIME_EBFC, A.MatVec('N', one, v, zero, w));
         if(pargemslr_global::_gram_schmidt == 1)
         {
            PARGEMSLR_TIME_CALL(comm, PARGEMSLR_BUILDTIME_MGS, PargemslrMgs( w, V, H, t, k, tol_orth, tol_reorth));
         }
         else
         {
            PARGEMSLR_TIME_CALL(comm, PARGEMSLR_BUILDTIME_MGS, PargemslrCgs2( w, V, H, t, k, tol_orth));
         }
#else
         A.MatVec('N', one, v, zero, w);
         if(pargemslr_global::_gram_schmidt == 1)
         {
            PargemslrMgs( w, V, H, t, k, tol_orth, tol_reorth);
         }
         else
         {
            PargemslrCgs2( w, V, H, t, k, tol_orth);
         }
#endif

         /* check "0.0" norm -- breakdown */
         if (PargemslrAbs(t) < tol_orth)
         {
            k++;
            break;
         }

      }

      v.Clear();
      w.Clear();

      return k;
   }
   template int PargemslrArnoldiNoRestart<SequentialVectorClass<float> >( precond_gemslrebfc_csr_seq_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiNoRestart<SequentialVectorClass<double> >( precond_gemslrebfc_csr_seq_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiNoRestart<SequentialVectorClass<complexs> >( precond_gemslrebfc_csr_seq_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiNoRestart<SequentialVectorClass<complexd> >( precond_gemslrebfc_csr_seq_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiNoRestart<ParallelVectorClass<float> >( precond_gemslrebfc_csr_par_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiNoRestart<ParallelVectorClass<double> >( precond_gemslrebfc_csr_par_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiNoRestart<ParallelVectorClass<complexs> >( precond_gemslrebfc_csr_par_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiNoRestart<ParallelVectorClass<complexd> >( precond_gemslrebfc_csr_par_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiNoRestart<SequentialVectorClass<float> >( matrix_csr_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiNoRestart<SequentialVectorClass<double> >( matrix_csr_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiNoRestart<SequentialVectorClass<complexs> >( matrix_csr_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiNoRestart<SequentialVectorClass<complexd> >( matrix_csr_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiNoRestart<ParallelVectorClass<float> >( matrix_csr_par_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiNoRestart<ParallelVectorClass<double> >( matrix_csr_par_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiNoRestart<ParallelVectorClass<complexs> >( matrix_csr_par_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiNoRestart<ParallelVectorClass<complexd> >( matrix_csr_par_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth);

   template <class VectorType, typename DataType, typename RealDataType>
   int PargemslrArnoldiThickRestartBuildThickRestartNewVector( DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, int m, RealDataType tol_orth, RealDataType tol_reorth, VectorType &v)
   {
      int                                 n_local, location;
      DataType                            one;
      RealDataType                        normv, t = 0.0;

      location = V.GetDataLocation();
      n_local = V.GetNumRowsLocal();
      one = 1.0;

      /* Pick a new vector */
      if(n_local > 0)
      {
         v.UpdatePtr( &(V(0, m)), location );
      }
      v.Rand();
      v.Norm2(normv);
      v.Scale(one/normv);

      /* now orth againist all previous cols, note that k is number of column "used" in V, exclude w */
      if(m > 0)
      {
         PargemslrOrthogonal( v, V, t, m-1, tol_orth, tol_reorth);
      }
      H(m, m-1) = 0;

      /* now we have a new v that is orthogal to all previous vectors, check if it's norm is too small */
      if(t < tol_orth)
      {
         /* in this case the new guess is not good enough, give up */
         //PARGEMSLR_PRINT("Thick restart can't add more eigenvectors.\n");
         return -1;
      }

      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_float &V, matrix_dense_float &H, int m, float tol_orth, float tol_reorth, vector_seq_float &v);
   template int PargemslrArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_double &V, matrix_dense_double &H, int m, double tol_orth, double tol_reorth, vector_seq_double &v);
   template int PargemslrArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_complexs &V, matrix_dense_complexs &H, int m, float tol_orth, float tol_reorth, vector_seq_complexs &v);
   template int PargemslrArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_complexd &V, matrix_dense_complexd &H, int m, double tol_orth, double tol_reorth, vector_seq_complexd &v);
   template int PargemslrArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_float &V, matrix_dense_float &H, int m, float tol_orth, float tol_reorth, vector_par_float &v);
   template int PargemslrArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_double &V, matrix_dense_double &H, int m, double tol_orth, double tol_reorth, vector_par_double &v);
   template int PargemslrArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_complexs &V, matrix_dense_complexs &H, int m, float tol_orth, float tol_reorth, vector_par_complexs &v);
   template int PargemslrArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_complexd &V, matrix_dense_complexd &H, int m, double tol_orth, double tol_reorth, vector_par_complexd &v);

   template <typename T>
   int TestPlotGnuPlotEigReal( const char *datafilename, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, IntVectorClass<int> &select)
   {
      int n, i;

      FILE *fdata, *pgnuplot;

      char tempfilename[1024];
      snprintf( tempfilename, 1024, "./TempData/%s", datafilename );

      if ((fdata = fopen(tempfilename, "w")) == NULL)
      {
         printf("Can't open file.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if ((pgnuplot = popen("gnuplot -persistent", "w")) == NULL)
      {
         printf("Can't open gnuplot file.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      n = wr.GetLengthLocal();

      PARGEMSLR_CHKERR(n != wi.GetLengthLocal());

      for(i = 0 ; i < n ; i ++)
      {
         if(select[i] > 0)
         {
            fprintf(fdata, "%f %f \n", wr[i], wi[i]);
         }
      }

      fclose(fdata);
      /*
      fprintf(pgnuplot, "set title \"Final Eigenvalues\"\n");
      fprintf(pgnuplot, "set logscale x\n");
      //fprintf(pgnuplot, "set xrange [1:%d]\n", _ncols);
      //fprintf(pgnuplot, "set yrange [1:%d]\n", _nrows);
      fprintf(pgnuplot, "plot '%s' pt 1\n", tempfilename);
      */
      pclose(pgnuplot);

      return PARGEMSLR_SUCCESS;
   }

   template <typename T>
   int PargemslrArnoldiThickRestartBuildResultReal(DenseMatrixClass<T> &Vm, DenseMatrixClass<T> &Hm, DenseMatrixClass<T> &Q,
                                                int &ncov, int neig_keep, vector_int &icov, SequentialVectorClass<T> &dcov,
                                                vector_int &work_int, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi,
                                                DenseMatrixClass<T> &V, DenseMatrixClass<T> &H)
   {
      if(ncov == 0)
      {
         /* in this case return immediatly */
         return PARGEMSLR_SUCCESS;
      }

      int                        i, j, n_local, m, location;
      vector_int                 select, order;
      T                          one, zero;
      DenseMatrixClass<T>        Vm_temp, Q_temp, V_temp;

      one = T(1.0);
      zero = T();

      /* get size first */
      n_local = Vm.GetNumRowsLocal();
      m = Vm.GetNumColsLocal();

      location = V.GetDataLocation();
      /* avoid allocating memory, use the working vector */
      select.SetupPtr(work_int, m, 0);
      select.Fill(0);
      order.SetupPtr(work_int, m, m);

      if(ncov > neig_keep)
      {
         /* select best ones
          * sort based on weight on descending order */
         dcov.Sort(order, false, true);
         for(i = 0 ; i < neig_keep ; i ++)
         {
            select[icov[order[i]]] = 1;
         }
         /* update ncov */
         ncov = neig_keep;

         /* check if the last two are in pairs */
         i = icov[order[neig_keep-1]];
         /* note that i might be m-1, check it to avoid index out of bound */
         if( i < m-1 && wi[i] > 0 && wi[i] == -wi[i+1])
         {
            /* In this case, we keep an extra one
             * note that we are using the stable sort, should be here
             */
            select[icov[order[neig_keep]]] = 1;
            ncov++;
         }

         order.Clear();
      }
      else
      {
         /* select all */
         for(i = 0 ; i < ncov ; i ++)
         {
            select[icov[i]] = 1;
         }
      }
      /*
      if(parallel_log::_grank == 0 && n_local > 400)
      {
         TestPlotGnuPlotEigReal( "TR-Final", wr, wi, select);
      }
      */
      /* ordschur to put those into leading part */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, Hm.OrdSchur(Q, wr, wi, select));

      /* update V and H for return
       * V first
       */
      Vm_temp.SetupPtr(Vm, 0, 0, n_local, m);
      V_temp.SetupPtr(V, 0, 0, n_local, ncov);
      if(location == kMemoryDevice)
      {
         /* extract matrix on device to apply matmat */
         Q.SubMatrix( 0, 0, m, ncov, kMemoryDevice, Q_temp);
      }
      else
      {
         Q_temp.SetupPtr(Q, 0, 0, m, ncov);
      }

      V_temp.MatMat( one, Vm_temp, 'N', Q_temp, 'N', zero);

      /* now H */
      H.Fill(zero);
      for (i = 0; i < ncov; i++)
      {
         for (j = 0; j < ncov; j++)
         {
            H(j,i) = Hm(j,i);
         }
      }

      Vm_temp.Clear();
      Q_temp.Clear();
      V_temp.Clear();
      select.Clear();
      order.Clear();

      return PARGEMSLR_SUCCESS;

   }

   template <typename DataType, typename RealDataType>
   int PargemslrArnoldiThickRestartBuildResultComplex(DenseMatrixClass<DataType> &Vm, DenseMatrixClass<DataType> &Hm, DenseMatrixClass<DataType> &Q,
                                                int &ncov, int neig_keep, vector_int &icov, SequentialVectorClass<RealDataType> &dcov,
                                                vector_int &work_int, SequentialVectorClass<DataType> &w,
                                                DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H)
   {

      if(ncov == 0)
      {
         /* in this case return immediatly */
         return PARGEMSLR_SUCCESS;
      }

      int                        i, j, n_local, m, location;
      vector_int                 select, order;
      DataType                   one, zero;
      DenseMatrixClass<DataType> Vm_temp, Q_temp, V_temp;

      one = DataType(1.0);
      zero = DataType();

      /* get size first */
      n_local = Vm.GetNumRowsLocal();
      m = Vm.GetNumColsLocal();

      location = V.GetDataLocation();

      /* avoid allocating memory, use the working vector */
      select.SetupPtr(work_int, m, 0);
      select.Fill(0);
      order.SetupPtr(work_int, m, m);

      if(ncov > neig_keep)
      {
         /* select best ones
          * sort based on weight on descending order */
         dcov.Sort(order, false, true);
         for(i = 0 ; i < neig_keep ; i ++)
         {
            select[icov[order[i]]] = 1;
         }
         /* update ncov */
         ncov = neig_keep;

         order.Clear();
      }
      else
      {
         /* select all */
         for(i = 0 ; i < ncov ; i ++)
         {
            select[icov[i]] = 1;
         }
      }

      /* ordschur to put those into leading part */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, Hm.OrdSchur(Q, w, select));

      /* update V and H for return
       * V first
       */
      Vm_temp.SetupPtr(Vm, 0, 0, n_local, m);
      V_temp.SetupPtr(V, 0, 0, n_local, ncov);
      if(location == kMemoryDevice)
      {
         /* extract matrix on device to apply matmat */
         Q.SubMatrix( 0, 0, m, ncov, kMemoryDevice, Q_temp);
      }
      else
      {
         Q_temp.SetupPtr(Q, 0, 0, m, ncov);
      }

      V_temp.MatMat( one, Vm_temp, 'N', Q_temp, 'N', zero);

      /* now H */
      H.Fill(zero);
      for (i = 0; i < ncov; i++)
      {
         for (j = 0; j < ncov; j++)
         {
            H(j,i) = Hm(j,i);
         }
      }

      Vm_temp.Clear();
      Q_temp.Clear();
      V_temp.Clear();
      select.Clear();
      order.Clear();

      return PARGEMSLR_SUCCESS;

   }

   template <typename T>
   int PargemslrArnoldiThickRestartBuildThickRestartNoLockReal(DenseMatrixClass<T> &Vm, DenseMatrixClass<T> &Hm, DenseMatrixClass<T> &Q,
                                                T h_last, int ncov, int nicov, int &npick,
                                                vector_int &icov, vector_int &iicov, SequentialVectorClass<T> &dicov,
                                                vector_int &work_int, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi,
                                                DenseMatrixClass<T> &V, DenseMatrixClass<T> &H)
   {
      PARGEMSLR_CHKERR(npick == 0);
      if(npick == nicov)
      {
         /* restart with all vectors
          * just apply more Arnoldi. Typically should not reach here.
          */
         return PARGEMSLR_SUCCESS;
      }

      int                  i, j, n_local, m, location;
      int                  trlen = npick + ncov;
      T                    one, zero;
      vector_int           select, order;
      DenseMatrixClass<T>  Vm_temp, Q_temp, V_temp;

      one = T(1.0);
      zero =T();

      /* get size first */
      n_local = Vm.GetNumRowsLocal();
      m = Vm.GetNumColsLocal();

      location = V.GetDataLocation();

      /* avoid allocating memory, use the working vector */
      select.SetupPtr(work_int, m, 0);
      select.Fill(0);
      order.SetupPtr(work_int, m, m);

      /* select all the convergenced eigenvalues first */
      for(i = 0 ; i < ncov ; i ++)
      {
         select[icov[i]] = 1;
      }

      /* now pick those "best" unconvergenced eigenvalues */
      dicov.Sort(order, false, true);

      for(i = 0 ; i < npick ; i ++)
      {
         select[iicov[order[i]]] = 1;
      }

      /* check if the last two are in pair */
      i = iicov[order[npick-1]];
      /* note that i might be m-1, check it to avoid index out of bound */
      if( i < m-1 && wi[i] > 0 && wi[i] == -wi[i+1] )
      {
         /* In this case, we keep an extra one */
         select[iicov[order[npick]]] = 1;
         npick++;
         trlen++;
      }

      /* now build the thick-restart */

      /* ordschur to put those into leading part */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, Hm.OrdSchur(Q, wr, wi, select));

      /* copy the first trlen * trlen */
      H.Fill(zero);
      for(i = 0 ; i < trlen ; i ++)
      {
         for(j = 0 ; j < trlen ; j ++)
         {
            H(j, i) = Hm(j, i);
         }
      }

      /* update the row trlen + 1
       *   * * * *
       *     * * *
       *     * * * <= might have 2x2 blocks
       *         *
       *   * * * * <= now this row
       */
      for(i = 0 ; i < trlen ; i ++)
      {
         H(trlen, i) = h_last * Q(m-1, i);
      }

      /* now prepare V, V := (Vm * Qhs)(:, 1:trlen) */
      Vm_temp.SetupPtr(Vm, 0, 0, n_local, m);
      V_temp.SetupPtr(V, 0, 0, n_local, trlen);
      if(location == kMemoryDevice)
      {
         Q.SubMatrix( 0, 0, m, trlen, kMemoryDevice, Q_temp);
      }
      else
      {
         Q_temp.SetupPtr(Q, 0, 0, m, trlen);
      }

      V_temp.MatMat( one, Vm_temp, 'N', Q_temp, 'N', zero);

      /* now copy the restart vector, the original last vector */
      PARGEMSLR_MEMCPY(V.GetData()+trlen*n_local, V.GetData()+m*n_local, n_local, location, location, T);

      /* all set, ready to restart */

      Vm_temp.Clear();
      Q_temp.Clear();
      V_temp.Clear();
      select.Clear();
      order.Clear();

      return PARGEMSLR_SUCCESS;

   }

   template <typename DataType, typename RealDataType>
   int PargemslrArnoldiThickRestartBuildThickRestartNoLockComplex(DenseMatrixClass<DataType> &Vm, DenseMatrixClass<DataType> &Hm, DenseMatrixClass<DataType> &Q,
                                                DataType h_last, int ncov, int nicov, int &npick,
                                                vector_int &icov, vector_int &iicov, SequentialVectorClass<RealDataType> &dicov,
                                                vector_int &work_int, SequentialVectorClass<DataType> &w,
                                                DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H)
   {
      PARGEMSLR_CHKERR(npick == 0);
      if(npick == nicov)
      {
         /* restart with all vectors
          * just apply more Arnoldi. Typically should not reach here.
          */
         return PARGEMSLR_SUCCESS;
      }

      int                           i, j, n_local, m, location;
      int                           trlen = npick + ncov;
      DataType                      one, zero;
      vector_int                    select, order;
      DenseMatrixClass<DataType>    Vm_temp, Q_temp, V_temp;

      one = DataType(1.0);
      zero = DataType();

      /* get size first */
      n_local = Vm.GetNumRowsLocal();
      m = Vm.GetNumColsLocal();

      location = V.GetDataLocation();

      /* avoid allocating memory, use the working vector */
      select.SetupPtr(work_int, m, 0);
      select.Fill(0);
      order.SetupPtr(work_int, m, m);

      /* select all the convergenced eigenvalues first */
      for(i = 0 ; i < ncov ; i ++)
      {
         select[icov[i]] = 1;
      }

      /* now pick those "best" unconvergenced eigenvalues */
      dicov.Sort(order, false, true);

      for(i = 0 ; i < npick ; i ++)
      {
         select[iicov[order[i]]] = 1;
      }

      /* now build the thick-restart */

      /* ordschur to put those into leading part */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, Hm.OrdSchur(Q, w, select));

      /* copy the first trlen * trlen */
      H.Fill(zero);
      for(i = 0 ; i < trlen ; i ++)
      {
         for(j = 0 ; j < trlen ; j ++)
         {
            H(j, i) = Hm(j, i);
         }
      }

      /* update the row trlen + 1
       *   * * * *
       *     * * *
       *     * * * <= might have 2x2 blocks
       *         *
       *   * * * * <= now this row
       */
      for(i = 0 ; i < trlen ; i ++)
      {
         H(trlen, i) = h_last * Q(m-1, i);
      }

      /* now prepare V, V := (Vm * Qhs)(:, 1:trlen) */
      Vm_temp.SetupPtr(Vm, 0, 0, n_local, m);
      V_temp.SetupPtr(V, 0, 0, n_local, trlen);
      if(location == kMemoryDevice)
      {
         Q.SubMatrix( 0, 0, m, trlen, kMemoryDevice, Q_temp);
      }
      else
      {
         Q_temp.SetupPtr(Q, 0, 0, m, trlen);
      }

      V_temp.MatMat( one, Vm_temp, 'N', Q_temp, 'N', zero);

      /* now copy the restart vector, the original last vector */
      PARGEMSLR_MEMCPY(V.GetData()+trlen*n_local, V.GetData()+m*n_local, n_local, location, location, DataType);

      /* all set, ready to restart */

      Vm_temp.Clear();
      Q_temp.Clear();
      V_temp.Clear();
      select.Clear();
      order.Clear();

      return PARGEMSLR_SUCCESS;

   }

   template <typename T>
   int TestPlotGnuPlotEigReal( const char *datafilename, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, IntVectorClass<int> &icov, IntVectorClass<int> &iicov)
   {
      int n, i;

      FILE *fdata, *pgnuplot;

      char tempfilename[1024];
      snprintf( tempfilename, 1024, "./TempData/%s", datafilename );

      if ((fdata = fopen(tempfilename, "w")) == NULL)
      {
         printf("Can't open file.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      if ((pgnuplot = popen("gnuplot -persistent", "w")) == NULL)
      {
         printf("Can't open gnuplot file.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }

      n = icov.GetLengthLocal();

      for(i = 0 ; i < n ; i ++)
      {
         fprintf(fdata, "%f %f \n", wr[icov[i]], wi[icov[i]]);
      }

      fclose(fdata);

      /*
      fprintf(pgnuplot, "set title \"Eigenvalues\"\n");
      fprintf(pgnuplot, "set logscale x\n");
      //fprintf(pgnuplot, "set xrange [1:%d]\n", _ncols);
      //fprintf(pgnuplot, "set yrange [1:%d]\n", _nrows);
      fprintf(pgnuplot, "plot '%s' pt 1\n", tempfilename);
      */

      pclose(pgnuplot);

      return PARGEMSLR_SUCCESS;
   }

   template <typename T>
   int PargemslrArnoldiThickRestartChooseEigenValuesReal( DenseMatrixClass<T> &H, DenseMatrixClass<T> &Q, T h_last,
                                                         T (*weight)(ComplexValueClass<T>), T truncate,
                                                         int &ncov, int &nicov, int &nsatis, T tol_eig, T eig_target_mag, T eig_truncate, bool &cut,
                                                         SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi,
                                                         vector_int &icov, vector_int &iicov, vector_int &isatis,
                                                         SequentialVectorClass<T> &dcov, SequentialVectorClass<T> &dicov, SequentialVectorClass<T> &dsatis)
   {
      /* typically we should not have those on the device memory */
      PARGEMSLR_CHKERR(icov.GetDataLocation() == kMemoryDevice);
      PARGEMSLR_CHKERR(iicov.GetDataLocation() == kMemoryDevice);
      PARGEMSLR_CHKERR(dcov.GetDataLocation() == kMemoryDevice);
      PARGEMSLR_CHKERR(dicov.GetDataLocation() == kMemoryDevice);


      typedef ComplexValueClass<T> TC;

      int                              i, m;
      T                                res, zero;
      TC                               temp_val;
      T                                temp_val_real, maxval, maxweight;
      DenseMatrixClass<T>              Qe, Q_last, Qe_last;

      /* the index */
      icov.Resize(0, false, false);
      iicov.Resize(0, false, false);
      isatis.Resize(0, false, false);
      /* the distance */
      dcov.Resize(0, false, false);
      dicov.Resize(0, false, false);
      dsatis.Resize(0, false, false);

      m = wr.GetLengthLocal();
      PARGEMSLR_CHKERR(wi.GetLengthLocal() != m);

      cut = false;
      zero = 0.0;
      maxweight = 0.0;
      maxval = 0.0;

      /* Eigen-decomposition */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, H.HessEig(Qe, wr, wi) );

      /* check convergence based on the lase row of the eigen decomposition Qh*Qs*Qe */
      Q_last.SetupPtr( Q, m-1, 0, 1, m);
      Qe_last.MatMat( h_last, Q_last, 'N', Qe, 'N', zero);

      ncov = 0;
      nicov = 0;

      /* loop through w */
      for(i = 0 ; i < m ; i ++)
      {
         if( i < m-1 && wi[i] > 0 && wi[i] == -wi[i+1])
         {
            /* In this case we have a pair of eigenvalues */
            res = PargemslrAbs(Qe_last(0, i)+Qe_last(0, i+1));
            if(res < tol_eig)
            {
               /* those are two convergenced eigenvalues */
               icov.PushBack(i);
               icov.PushBack(i+1);
               temp_val = TC(wr[i], wi[i]);
               //temp_val = TC(wr[i], -wi[i]);
               temp_val_real = (*weight)(temp_val);
               /*
               if(res > 1e-12)
               {
                  temp_val_real = 1.0/res;
               }
               else
               {
                  temp_val_real = 1e12;
               }
               */
               maxweight = PargemslrMax(temp_val_real, maxweight);
               temp_val_real = PargemslrAbs(temp_val);
               maxval = PargemslrMax(temp_val_real, maxval);
               dcov.PushBack(temp_val_real);
               dcov.PushBack(temp_val_real);
               i++;
               ncov+=2;
            }
            else
            {
               /* those are two inconvergenced eigenvalue */
               iicov.PushBack(i);
               iicov.PushBack(i+1);
               temp_val = TC(wr[i], wi[i]);
               //temp_val = TC(wr[i], -wi[i]);
               temp_val_real = (*weight)(temp_val);
               /*
               if(res > 1e-12)
               {
                  temp_val_real = 1.0/res;
               }
               else
               {
                  temp_val_real = 1e12;
               }
               */
               maxweight = PargemslrMax(temp_val_real, maxweight);
               dicov.PushBack(temp_val_real);
               dicov.PushBack(temp_val_real);
               i++;
               nicov+=2;
            }
         }
         else
         {
            res = PargemslrAbs(Qe_last(0, i));
            if(res < tol_eig)
            {
               /* this is a convergenced eigenvalue */
               icov.PushBack(i);
               temp_val = TC(wr[i], T());
               temp_val_real = (*weight)(temp_val);
               /*
               if(res > 1e-12)
               {
                  temp_val_real = 1.0/res;
               }
               else
               {
                  temp_val_real = 1e12;
               }
               */
               maxweight = PargemslrMax(temp_val_real, maxweight);
               dcov.PushBack(temp_val_real);
               temp_val_real = PargemslrAbs(temp_val);
               maxval = PargemslrMax(temp_val_real, maxval);
               ncov++;
            }
            else
            {
               /* this is not a convergenced eigenvalue */
               iicov.PushBack(i);
               temp_val = TC(wr[i], T());
               temp_val_real = (*weight)(temp_val);
               /*
               if(res > 1e-12)
               {
                  temp_val_real = 1.0/res;
               }
               else
               {
                  temp_val_real = 1e12;
               }
               */
               maxweight = PargemslrMax(temp_val_real, maxweight);
               dicov.PushBack(temp_val_real);
               nicov++;
            }
         }
      }

      /* now select satisfied values */

      nsatis = 0;

      //PARGEMSLR_PRINT_DEBUG(parallel_log::_grank, 0, "Current step max function value %f, maxval %f\n", maxweight, maxval);

      /* only check when we are likely to get the largest eigenvalues */
      maxweight *= truncate;

      maxval = PargemslrMin(eig_target_mag, maxval);
      maxval *= eig_truncate;

      for(i = 0 ; i < ncov ; i ++)
      {

         if(dcov[i] >= maxweight)
         {
            isatis.PushBack(icov[i]);
            dsatis.PushBack(dcov[i]);
            nsatis++;
         }
         else if(!cut)
         {
            /* we reject this one, check if it was too small */
            temp_val = TC(wr[icov[i]], wi[icov[i]]);
            if(PargemslrAbs(temp_val) < maxval)
            {
               /* this value is too small, we don't need any further computation */
               PARGEMSLR_PRINT_DEBUG(parallel_log::_grank, 0, "Cut val (%f,%f)\n", wr[icov[i]], wi[icov[i]]);
               cut = true;
            }
         }
      }

      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrArnoldiThickRestartChooseEigenValuesReal( matrix_dense_float &H, matrix_dense_float &Q, float h_last, float (*weight)(complexs), float truncate, int &ncov, int &nicov, int &nsatis, float tol_eig, float eig_target_mag, float eig_truncate, bool &cut, vector_seq_float &wr, vector_seq_float &wi, vector_int &icov, vector_int &iicov, vector_int &isatis, vector_seq_float &dcov, vector_seq_float &dicov, vector_seq_float &dsatis);
   template int PargemslrArnoldiThickRestartChooseEigenValuesReal( matrix_dense_double &H, matrix_dense_double &Q, double h_last, double (*weight)(complexd), double truncate, int &ncov, int &nicov, int &nsatis, double tol_eig, double eig_target_mag, double eig_truncate, bool &cut, vector_seq_double &wr, vector_seq_double &wi, vector_int &icov, vector_int &iicov, vector_int &isatis, vector_seq_double &dcov, vector_seq_double &dicov, vector_seq_double &dsatis);

   template <typename DataType, typename RealDataType>
   int PargemslrArnoldiThickRestartChooseEigenValuesComplex( DenseMatrixClass<DataType> &H, DenseMatrixClass<DataType> &Q, DataType h_last,
                                                         RealDataType (*weight)(DataType), RealDataType truncate,
                                                         int &ncov, int &nicov, int &nsatis, RealDataType tol_eig, RealDataType eig_target_mag, RealDataType eig_truncate, bool &cut,
                                                         SequentialVectorClass<DataType> &w,
                                                         vector_int &icov, vector_int &iicov, vector_int &isatis,
                                                         SequentialVectorClass<RealDataType> &dcov, SequentialVectorClass<RealDataType> &dicov, SequentialVectorClass<RealDataType> &dsatis)
   {
      /* typically we should not have those on the device memory */
      PARGEMSLR_CHKERR(icov.GetDataLocation() == kMemoryDevice);
      PARGEMSLR_CHKERR(iicov.GetDataLocation() == kMemoryDevice);
      PARGEMSLR_CHKERR(dcov.GetDataLocation() == kMemoryDevice);
      PARGEMSLR_CHKERR(dicov.GetDataLocation() == kMemoryDevice);

      int                              i, m;
      RealDataType                     res, temp_val, maxval, maxweight;
      DataType                         zero;
      DenseMatrixClass<DataType>       Qe, Q_last, Qe_last;

      /* the index */
      icov.Resize(0, false, false);
      iicov.Resize(0, false, false);
      isatis.Resize(0, false, false);
      /* the distance */
      dcov.Resize(0, false, false);
      dicov.Resize(0, false, false);
      dsatis.Resize(0, false, false);

      m = w.GetLengthLocal();

      cut = false;
      zero = 0.0;
      maxval = 0.0;
      maxweight = 0.0;

      /* Eigen-decomposition */
      PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, H.HessEig(Qe, w) );

      /* check convergence based on the lase row of the eigen decomposition Qh*Qs*Qe */
      Q_last.SetupPtr( Q, m-1, 0, 1, m);
      Qe_last.MatMat( h_last, Q_last, 'N', Qe, 'N', zero);

      ncov = 0;
      nicov = 0;

      /* loop through w */
      for(i = 0 ; i < m ; i ++)
      {
         res = PargemslrAbs(Qe_last(0, i));
         if(res < tol_eig)
         {
            /* this is a convergenced eigenvalue */
            icov.PushBack(i);
            temp_val = (*weight)(w[i]);
            dcov.PushBack(temp_val);
            maxval = PargemslrMax(PargemslrAbs(w[i]), maxval);
            maxweight = PargemslrMax(temp_val, maxweight);
            ncov++;
         }
         else
         {
            /* this is not a convergenced eigenvalue */
            iicov.PushBack(i);
            temp_val = (*weight)(w[i]);
            maxweight = PargemslrMax(temp_val, maxweight);
            dicov.PushBack(temp_val);
            nicov++;
         }
      }

      /* now select satisfied values */

      nsatis = 0;

      //PARGEMSLR_PRINT_DEBUG(parallel_log::_grank, 0, "Current step max function value %f, maxval %f \n", maxweight, maxval);

      maxweight *= truncate;

      maxval = PargemslrMin(eig_target_mag, maxval);
      maxval *= eig_truncate;

      for(i = 0 ; i < ncov ; i ++)
      {
         if(dcov[i] >= maxweight)
         {
            isatis.PushBack(icov[i]);
            dsatis.PushBack(dcov[i]);
            nsatis++;
         }
         else if(!cut)
         {
            if(PargemslrAbs(w[icov[i]]) < maxval)
            {
               /* this value is too small, we don't need any further computation */
               PARGEMSLR_PRINT_DEBUG(parallel_log::_grank, 0, "Cut val (%f,%f)\n", w[icov[i]].Real(), w[icov[i]].Imag());
               cut = true;
            }
         }
      }

      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrArnoldiThickRestartChooseEigenValuesComplex( matrix_dense_complexs &H, matrix_dense_complexs &Q, complexs h_last, float (*weight)(complexs), float truncate, int &ncov, int &nicov, int &nsatis, float tol_eig, float eig_target_mag, float eig_truncate, bool &cut, vector_seq_complexs &w, vector_int &icov, vector_int &iicov, vector_int &isatis, vector_seq_float &dcov, vector_seq_float &dicov, vector_seq_float &dsatis);
   template int PargemslrArnoldiThickRestartChooseEigenValuesComplex( matrix_dense_complexd &H, matrix_dense_complexd &Q, complexd h_last, double (*weight)(complexd), double truncate, int &ncov, int &nicov, int &nsatis, double tol_eig, double eig_target_mag, double eig_truncate, bool &cut, vector_seq_complexd &w, vector_int &icov, vector_int &iicov, vector_int &isatis, vector_seq_double &dcov, vector_seq_double &dicov, vector_seq_double &dsatis);

   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int PargemslrArnoldiThickRestartNoLock( MatrixType &A, int msteps, int maxits, int rank, int rank2, RealDataType truncate,
                                 RealDataType tr_fact, RealDataType tol_eig, RealDataType eig_target_mag, RealDataType eig_truncate,
                                 RealDataType (*weight)(ComplexValueClass<RealDataType>),
                                 DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H,
                                 RealDataType tol_orth, RealDataType tol_reorth)
   {
      PARGEMSLR_CHKERR(rank < 0);
      PARGEMSLR_CHKERR(maxits < 0);
      PARGEMSLR_CHKERR(msteps < 0);

      if(rank == 0 || msteps == 0 || maxits == 0)
      {
         return 0;
      }

      if(tol_orth < std::numeric_limits<RealDataType>::epsilon())
      {
         /* the tolerance should not be too small */
         tol_orth = std::numeric_limits<RealDataType>::epsilon();
      }

      if(tol_eig < std::numeric_limits<RealDataType>::epsilon())
      {
         /* the tolerance should not be too small */
         tol_eig = std::numeric_limits<RealDataType>::epsilon();
      }

      /* - - - - - - - - - - - - - - - - - -
       * Thick-restart version Arnoldi.
       *
       * Version 1: When the dropping tolorance is large.
       *
       * In this setup, no locked eigenvalues.
       *
       * Step 1: Generate the intial guess, set trlen := 0.
       *
       * Step 2: Apply Arnoldi iteration starting from trlen, to
       *         msteps = m + trlen.
       *
       * Step 3: H is now having the structure:
       *         * * * * * * *
       *           * * * * * *
       *             * * * * *
       *         * * * * * * *
       *               * * * *
       *                 * * *
       *                   * *
       *                     *
       *         Extract Hm := H(1:m, :), and check convergence.
       *
       * Step 4: Assume we have ncov convergenced eigenvalues, nicov inconvergenced
       *         eigenvalues, restart with trlen = ncov + nicov * trfact.
       *         When there are no incovergenced eigenvalues, it's likely that we've
       *         found an invarient subspace. In this case, keep the current result,
       *         start from a new initial guess.
       *         Go back to step 2, or stop.
       */
      /* define the data type */
      typedef ComplexValueClass<RealDataType> ComplexDataType;

      int                                                      n_local, i, j, m, mstepsi, maxsteps, trlen, npick, its;
      int                                                      location;
      int                                                      ncov = 0, nicov, nsatis;
      bool                                                     cut;
      vector_int                                               icov, iicov, isatis;
      SequentialVectorClass<RealDataType>                      dcov, dicov, dsatis;
      vector_int                                               work_int;
      SequentialVectorClass<RealDataType>                      wr, wi;
      SequentialVectorClass<ComplexDataType>                   w;

      DataType                                                 h_last;
      RealDataType                                             zero_r, h_last_r;
      ComplexDataType                                          zero_c, h_last_c;

      DenseMatrixClass<DataType>                               V_data, H_data, Qhs;
      DenseMatrixClass<DataType>                               Vm, Hm;
      DenseMatrixClass<RealDataType>                           *V_r, *H_r, *Vm_r, *Hm_r, *Qhs_r;
      DenseMatrixClass<ComplexDataType>                        *V_c, *H_c, *Vm_c, *Hm_c, *Qhs_c;
      DenseMatrixClass<DataType>                               V_temp, Vm_temp, Q_temp;

#ifdef PARGEMSLR_TIMING
      int np, myid;
      MPI_Comm comm;
      A.GetMpiInfo(np, myid, comm);
#endif

      /* update pointers
       * TODO: find a better way to combine those templates
       */
      V_r = (DenseMatrixClass<RealDataType>*)&V;
      V_c = (DenseMatrixClass<ComplexDataType>*)&V;
      H_r = (DenseMatrixClass<RealDataType>*)&H;
      H_c = (DenseMatrixClass<ComplexDataType>*)&H;
      Vm_r = (DenseMatrixClass<RealDataType>*)&Vm;
      Vm_c = (DenseMatrixClass<ComplexDataType>*)&Vm;
      Hm_r = (DenseMatrixClass<RealDataType>*)&Hm;
      Hm_c = (DenseMatrixClass<ComplexDataType>*)&Hm;
      Qhs_r = (DenseMatrixClass<RealDataType>*)&Qhs;
      Qhs_c = (DenseMatrixClass<ComplexDataType>*)&Qhs;

      VectorType                                               v;

      /*------------------------
       * 1: Init Phase
       * Declare all variables and setup parameters
       *------------------------*/

      /* this is the maximum number of columns we can have, use this to generate the working buffer */
      maxsteps = H.GetNumColsLocal();

      zero_r                     = RealDataType();
      zero_c                     = ComplexDataType();

      A.SetupVectorPtrStr(v);

      n_local = V.GetNumRowsLocal();
      location = V.GetDataLocation();

      /* create working buffer.
       * Note that V can be on the device.
       * H can be on the host.
       */
      V_data.Setup( n_local, maxsteps+1, location, true);
      H_data.Setup( maxsteps+1, maxsteps, kMemoryHost, true);

      work_int.Setup(2*maxsteps);

      icov.Setup(maxsteps);
      iicov.Setup(maxsteps);
      isatis.Setup(maxsteps);
      /* the distance */
      dcov.Setup(maxsteps);
      dicov.Setup(maxsteps);
      dsatis.Setup(maxsteps);

      /*------------------------
      * 2: Arnoldi and get result
      *------------------------*/

      its   = 0;
      trlen = 0;

      /* main loop */

      /* The tolorance for the residual of the dropping tolorance is too large.
       * Lock of eigenvalues disabled, more restarts doesn't guarantee more eigenvalues.
       */
      while(its < maxits)
      {
         its ++;

         /*------------------------
         * 2.1: Apply Arnoldi
         *------------------------*/

         /* we restart with length m + trlen */
         mstepsi   = PargemslrMin( msteps + trlen, maxsteps);

         /* compute arnoldi starting from the trlen */
#ifdef PARGEMSLR_TIMING
         PARGEMSLR_TIME_CALL( comm, PARGEMSLR_BUILDTIME_ARNOLDI, m = PargemslrArnoldiNoRestart<VectorType>( A, trlen, mstepsi, V, H, tol_orth, tol_reorth));
#else
         m = PargemslrArnoldiNoRestart<VectorType>( A, trlen, mstepsi, V, H, tol_orth, tol_reorth);
#endif
         /* record the last entry */
         h_last = H(m, m-1);
         h_last_c = h_last;
         h_last_r = PargemslrAbs(h_last);

         /* check if H(m, m-1) is zero */
         if(PargemslrAbs(h_last) < tol_orth)
         {
            /* reset to exact 0 */
            h_last_c = ComplexDataType(0.0);
            h_last_r = RealDataType(0.0);
            /* if H(m, m-1) is zero, and this is not the last loop, we find another vector to restart with */
            if(m < rank && its < maxits && PargemslrArnoldiThickRestartBuildThickRestartNewVector(V, H, m, tol_orth, tol_reorth, v) >= 0)
            {
               /* in this case, we can restart
                * and we've built the new vector
                * restart with length m
                */
               trlen = m;
               continue;
            }

            /* if we reach here, we can't find the new vector, stop here
             * we still need to apply a Schur Decomposition on H, copy to Vm and Hm
             */

            Vm.SetupPtr(V_data, 0, 0, n_local, m);
            Hm.SetupPtr(H_data, 0, 0, m, m);

            /* copy data from V to Vm */
            PARGEMSLR_MEMCPY(Vm.GetData(), V.GetData(), n_local*m, location, location, DataType);

            /* copy data from H to Hm */
            for (i = 0; i < m; i++)
            {
               for (j = 0; j < m; j++)
               {
                  Hm(j,i) = H(j,i);
               }
            }

            /* Compute the schur decomposition Hm = QhsUQhs^H
             * note that in the real case, there might be diagonal 2*2 blocks
             */
            if(PargemslrIsComplex<DataType>::value)
            {
               PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, Hm_c->Schur(*Qhs_c, w));
            }
            else
            {
               PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, Hm_r->Schur(*Qhs_r, wr, wi));
            }

            if(truncate <= 0.0 && m <= rank && rank <= rank2)
            {
               /* in this case, we keep all, no need to compute the eigen decomposition
                * otherwise we need to compute the distance array
                */
               nsatis = m;
               ncov = m;
               nicov = 0;
               icov.Resize(m, false, false);
               icov.UnitPerm();
            }
            else
            {
               /* Now build the eigen decomposition of Hm */
               if(PargemslrIsComplex<DataType>::value)
               {
                  PargemslrArnoldiThickRestartChooseEigenValuesComplex(*Hm_c, *Qhs_c, zero_c, weight, truncate, ncov, nicov, nsatis, tol_eig, eig_target_mag, eig_truncate, cut, w, icov, iicov, isatis, dcov, dicov, dsatis);
               }
               else
               {
                  PargemslrArnoldiThickRestartChooseEigenValuesReal(*Hm_r, *Qhs_r, zero_r, weight, truncate, ncov, nicov, nsatis, tol_eig, eig_target_mag, eig_truncate, cut, wr, wi, icov, iicov, isatis, dcov, dicov, dsatis);
               }
            }

            /* go to build V and H */
            break;

         }

         /*--------------------------
         * 2.2: Compute decomposition
         *--------------------------*/

         /* if we reach here, H(m, m-1) is not zero */

         /* get Vm and Hm */
         Vm.SetupPtr(V_data, 0, 0, n_local, m);
         Hm.SetupPtr(H_data, 0, 0, m, m);

         /* copy data from V to Vm */
         PARGEMSLR_MEMCPY(Vm.GetData(), V.GetData(), n_local*m, location, location, DataType);

         for (i = 0; i < m; i++)
         {
            for (j = 0; j < m; j++)
            {
               Hm(j,i) = H(j,i);
            }
         }

         /* Compute the schur decomposition Hm = QhsUQhs^H
          * note that in the real case, there might be diagonal 2*2 blocks
          */
         if(PargemslrIsComplex<DataType>::value)
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, Hm_c->Schur(*Qhs_c, w));
         }
         else
         {
            PARGEMSLR_LOCAL_TIME_CALL( PARGEMSLR_BUILDTIME_DECOMP, Hm_r->Schur(*Qhs_r, wr, wi));
         }

         /*------------------------
          * 2.3: Check convergence
          *------------------------*/

         /* Now check convergence based on the eigen decomposition */
         if(PargemslrIsComplex<DataType>::value)
         {
            PargemslrArnoldiThickRestartChooseEigenValuesComplex(*Hm_c, *Qhs_c, h_last_c, weight, truncate, ncov, nicov, nsatis, tol_eig, eig_target_mag, eig_truncate, cut, w, icov, iicov, isatis, dcov, dicov, dsatis);
         }
         else
         {
            PargemslrArnoldiThickRestartChooseEigenValuesReal(*Hm_r, *Qhs_r, h_last_r, weight, truncate, ncov, nicov, nsatis, tol_eig, eig_target_mag, eig_truncate, cut, wr, wi, icov, iicov, isatis, dcov, dicov, dsatis);

            /*
            if(parallel_log::_grank == 0 && n_local > 400)
            {
               char filename[1024];
               snprintf( filename, 1024, "TR-loop-%05d", its );
               TestPlotGnuPlotEigReal( filename, wr, wi, icov, iicov);
            }
            */
         }

         if(nsatis >= rank || its == maxits || cut)
         {
            /* we've got enough, or this is the last loop, stop */
            //PARGEMSLR_PRINT_DEBUG(parallel_log::_grank, 0, "Thick restart arnoldi breat with cut: %s\n", (cut) ? "true" : "false");
            break;
         }

         /* if we reach here, we still have the next loop, prepare restart */

         /*------------------------
          * 2.4: Prepare restart
          *------------------------*/

         if(nicov > 0)
         {
            /* Case 1: Not enough, have unconvergenced eigenvalues.
             * Prepare for the thick restart with cov + icov*tr_factor */

            /* first pick number of eigevalues, we want to restart with at least one vector
             * we first need to check if we have extra size
             */
            npick = PargemslrMax((int)(nicov * tr_fact), 1);
            npick = PargemslrMin(npick, nicov);

            /* buld thick restart */
            if(PargemslrIsComplex<DataType>::value)
            {
               PargemslrArnoldiThickRestartBuildThickRestartNoLockComplex(*Vm_c, *Hm_c, *Qhs_c, h_last_c, ncov, nicov, npick, icov, iicov, dicov, work_int, w, *V_c, *H_c);
            }
            else
            {
               PargemslrArnoldiThickRestartBuildThickRestartNoLockReal(*Vm_r, *Hm_r, *Qhs_r, h_last_r, ncov, nicov, npick, icov, iicov, dicov, work_int, wr, wi, *V_r, *H_r);
            }

            /* npick might be updated in the loop */
            trlen = ncov + npick;

         }
         else
         {
            /* In this case, no inconvergenced eigs, but we haven't got enough, just keep doing Arnoldi */
            trlen = m;
            /* if h_last is too small, we need to restart with new vector */
            if( PargemslrAbs(h_last) < tol_orth && PargemslrArnoldiThickRestartBuildThickRestartNewVector(V, H, m, tol_orth, tol_reorth, v) < 0)
            {
               //PARGEMSLR_PRINT("Thick restart can't add more eigenvectors.\n");
               /* ncov already computed */
               break;
            }
         }

      }

      /* done here, return all the convergenced result */
      //PARGEMSLR_PRINT_DEBUG(parallel_log::_grank, 0, "Thick restart arnoldi got %d eig convergenced, %d kept, with %d numits\n", ncov, nsatis, its);
      if(PargemslrIsComplex<DataType>::value)
      {
         PargemslrArnoldiThickRestartBuildResultComplex(*Vm_c, *Hm_c, *Qhs_c, ncov, rank2, icov, dcov, work_int, w, *V_c, *H_c);
      }
      else
      {
         PargemslrArnoldiThickRestartBuildResultReal(*Vm_r, *Hm_r, *Qhs_r, ncov, rank2, icov, dcov, work_int, wr, wi, *V_r, *H_r);
      }

      /* Deallocate */
      V_data.Clear();
      H_data.Clear();
      V_temp.Clear();
      Vm_temp.Clear();
      Q_temp.Clear();
      work_int.Clear();
      icov.Clear();
      iicov.Clear();
      isatis.Clear();
      dcov.Clear();
      dicov.Clear();
      dsatis.Clear();
      wr.Clear();
      wi.Clear();

      return ncov;
   }
   template int PargemslrArnoldiThickRestartNoLock<vector_seq_float>( precond_gemslrebfc_csr_seq_float &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float eig_target_mag, float eig_truncate, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiThickRestartNoLock<vector_seq_double>( precond_gemslrebfc_csr_seq_double &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double eig_target_mag, double eig_truncate, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiThickRestartNoLock<vector_par_float>( precond_gemslrebfc_csr_par_float &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float eig_target_mag, float eig_truncate, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiThickRestartNoLock<vector_par_double>( precond_gemslrebfc_csr_par_double &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double eig_target_mag, double eig_truncate, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiThickRestartNoLock<vector_seq_complexs>( precond_gemslrebfc_csr_seq_complexs &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float eig_target_mag, float eig_truncate, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiThickRestartNoLock<vector_seq_complexd>( precond_gemslrebfc_csr_seq_complexd &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double eig_target_mag, double eig_truncate, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth);
   template int PargemslrArnoldiThickRestartNoLock<vector_par_complexs>( precond_gemslrebfc_csr_par_complexs &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float eig_target_mag, float eig_truncate, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth);
   template int PargemslrArnoldiThickRestartNoLock<vector_par_complexd>( precond_gemslrebfc_csr_par_complexd &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double eig_target_mag, double eig_truncate, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth);

   template <class VectorType, typename DataType, typename RealDataType>
   int PargemslrCgs2( VectorType &w, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H,
                  RealDataType &t, int k, RealDataType tol_orth)
   {

      if(k < 0)
      {
         /* in this case, we don't have any previous vectors, return immediatly */
         return PARGEMSLR_SUCCESS;
      }

      int np, myid;
      MPI_Comm comm;
      w.GetMpiInfo(np, myid, comm);

      /*------------------------
       * 1: Modified Gram-Schmidt
       *------------------------*/

      int               i, n_local, err = 0;
      DataType          t1, one, zero;

      one = 1.0;
      zero = 0.0;

      VectorType        v;
      v.SetupPtrStr(w);

      n_local = w.GetLengthLocal();

      SequentialVectorClass<DataType>  v_local, w_local;
      SequentialVectorClass<DataType>  s;

      s.Setup(k+1);

      if(n_local > 0)
      {
         w_local.SetupPtr( w.GetData(), n_local, w.GetDataLocation() );
      }

      /* H(1:i,i) = V(:,1:i)'*V(:,i1)
       * V(:,i1) = V(:,i1) - V(:,1:i)*H(1:i,i)
       */

      for( i = 0 ; i <= k ; i ++)
      {
         /* inner produce and update H, w */
         if(n_local > 0)
         {
            v_local.SetupPtr( &V(0, i), n_local, V.GetDataLocation() );
            err = v_local.Dot(w_local, t1); PARGEMSLR_CHKERR(err);
            H(i, k) = t1;
         }
         else
         {
            H(i, k) = zero;
         }
      }

      if(w.IsParallel())
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiAllreduceInplace( &H(0, k), k+1, MPI_SUM, comm) );
      }

      for( i = 0 ; i <= k ; i ++)
      {
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, i), V.GetDataLocation() );
         }
         err = w.Axpy(-H(i, k), v); PARGEMSLR_CHKERR(err);
      }

      /* re-orth
       * s = V(:,1:i)'*V(:,i1)
       * V(:,i1) = V(:,i1) - V(:,1:i)*s
       * H(1:i,i) = H(1:i,i) + s;
       */

      for( i = 0 ; i <= k ; i ++)
      {
         /* inner produce and update H, w */
         if(n_local > 0)
         {
            v_local.SetupPtr( &V(0, i), n_local, V.GetDataLocation() );
            err = v_local.Dot(w_local, t1); PARGEMSLR_CHKERR(err);
            s[i] = t1;
         }
         else
         {
            s[i] = zero;
         }
      }

      if(w.IsParallel())
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiAllreduceInplace( s.GetData(), k+1, MPI_SUM, comm) );
      }

      for( i = 0 ; i <= k ; i ++)
      {
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, i), V.GetDataLocation() );
         }
         err = w.Axpy(-s[i], v); PARGEMSLR_CHKERR(err);
         H(i, k) = H(i, k) + s[i];
      }

      /* Compute ||w|| */
      err = w.Norm2(t); PARGEMSLR_CHKERR(err);

      H(k+1,k) = t;

      /* scale w in this function */
      err = w.Scale(one/t); PARGEMSLR_CHKERR(err);

      v.Clear();
      s.Clear();

      return err;

   }
   template int PargemslrCgs2( SequentialVectorClass<float> &w, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H,
                  float &t, int k, float tol_orth);
   template int PargemslrCgs2( SequentialVectorClass<double> &w, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H,
                  double &t, int k, double tol_orth);
   template int PargemslrCgs2( SequentialVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H,
                  float &t, int k, float tol_orth);
   template int PargemslrCgs2( SequentialVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H,
                  double &t, int k, double tol_orth);
   template int PargemslrCgs2( ParallelVectorClass<float> &w, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H,
                  float &t, int k, float tol_orth);
   template int PargemslrCgs2( ParallelVectorClass<double> &w, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H,
                  double &t, int k, double tol_orth);
   template int PargemslrCgs2( ParallelVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H,
                  float &t, int k, float tol_orth);
   template int PargemslrCgs2( ParallelVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H,
                  double &t, int k, double tol_orth);

   template <class VectorType, typename DataType, typename RealDataType>
   int PargemslrMgs( VectorType &w, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H,
                  RealDataType &t, int k, RealDataType tol_orth, RealDataType tol_reorth)
   {

      if(k < 0)
      {
         /* in this case, we don't have any previous vectors, return immediatly */
         return PARGEMSLR_SUCCESS;
      }

      /*------------------------
       * 1: Modified Gram-Schmidt
       *------------------------*/

      int               i, n_local, err = 0;
      DataType          t1;
      RealDataType      normw;

      VectorType        v;
      v.SetupPtrStr(w);

      n_local = w.GetLengthLocal();

      /* compute initial ||w|| if we need to reorth */
      if(tol_reorth > 0.0)
      {
         err = w.Norm2(normw); PARGEMSLR_CHKERR(err);
      }
      else
      {
         normw = 0.0;
      }

      for( i = 0 ; i <= k ; i ++)
      {
         /* inner produce and update H, w */
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, i), V.GetDataLocation() );
         }
         err = v.Dot(w, t1); PARGEMSLR_CHKERR(err);
         H(i, k) = t1;
         err = w.Axpy(-t1, v); PARGEMSLR_CHKERR(err);
      }
      /* Compute ||w|| */
      err = w.Norm2(t); PARGEMSLR_CHKERR(err);

      /*------------------------
       * 2: Re-orth step
       *------------------------*/

      /* t < tol_orth is considered be lucky breakdown */
      while(t < normw * tol_reorth && t >= tol_orth)
      {
         normw = t;
         /* Re-orth */
         for (i = 0; i <= k; i++)
         {
            if(n_local > 0)
            {
               v.UpdatePtr( &V(0, i), V.GetDataLocation() );
            }
            err = v.Dot(w, t1); PARGEMSLR_CHKERR(err);
            H(i, k) += t1;
            err = w.Axpy(-t1,v); PARGEMSLR_CHKERR(err);
         }
         /* Compute ||w|| */
         err = w.Norm2(t); PARGEMSLR_CHKERR(err);

      }
      H(k+1,k) = t;

      /* scale w in this function */
      err = w.Scale(DataType(1.0)/t); PARGEMSLR_CHKERR(err);

      v.Clear();

      return err;

   }
   template int PargemslrMgs( SequentialVectorClass<float> &w, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H,
                  float &t, int k, float tol_orth, float tol_reorth);
   template int PargemslrMgs( SequentialVectorClass<double> &w, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H,
                  double &t, int k, double tol_orth, double tol_reorth);
   template int PargemslrMgs( SequentialVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H,
                  float &t, int k, float tol_orth, float tol_reorth);
   template int PargemslrMgs( SequentialVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H,
                  double &t, int k, double tol_orth, double tol_reorth);
   template int PargemslrMgs( ParallelVectorClass<float> &w, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H,
                  float &t, int k, float tol_orth, float tol_reorth);
   template int PargemslrMgs( ParallelVectorClass<double> &w, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H,
                  double &t, int k, double tol_orth, double tol_reorth);
   template int PargemslrMgs( ParallelVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H,
                  float &t, int k, float tol_orth, float tol_reorth);
   template int PargemslrMgs( ParallelVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H,
                  double &t, int k, double tol_orth, double tol_reorth);

   template <class VectorType, typename DataType, typename RealDataType>
   int PargemslrOrthogonal( VectorType &w, DenseMatrixClass<DataType> &V,
                  RealDataType &t, int k, RealDataType tol_orth, RealDataType tol_reorth)
   {

      if(k < 0)
      {
         /* in this case, we don't have any previous vectors, return immediatly */
      }

      /*------------------------
       * 1: Modified Gram-Schmidt
       *------------------------*/

      int               i, n_local, err;
      DataType          t1;
      RealDataType      normw;

      VectorType        v;
      v.SetupPtrStr(w);

      n_local = w.GetLengthLocal();

      /* compute initial ||w|| if we need to reorth */
      if(tol_reorth > 0.0)
      {
         err = w.Norm2(normw); PARGEMSLR_CHKERR(err);
      }
      else
      {
         normw = 0.0;
      }

      for( i = 0 ; i <= k ; i ++)
      {
         /* inner produce and update w */
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, i), V.GetDataLocation() );
         }
         err = v.Dot(w, t1); PARGEMSLR_CHKERR(err);
         err = w.Axpy(-t1, v); PARGEMSLR_CHKERR(err);

      }
      /* Compute ||w|| */
      err = w.Norm2(t); PARGEMSLR_CHKERR(err);

      /*------------------------
       * 2: Re-orth step
       *------------------------*/

      /* t < tol_orth is considered be lucky breakdown */
      while(t < normw * tol_reorth && t >= tol_orth)
      {
         normw = t;
         /* Re-orth */
         for (i = 0; i <= k; i++)
         {
            if(n_local > 0)
            {
               v.UpdatePtr( &V(0, i), V.GetDataLocation() );
            }
            err = v.Dot(w, t1); PARGEMSLR_CHKERR(err);
            err = w.Axpy(-t1,v); PARGEMSLR_CHKERR(err);
         }
         /* Compute ||w|| */
         err = w.Norm2(t); PARGEMSLR_CHKERR(err);

      }

      /* scale w in this function */
      err = w.Scale(DataType(1.0)/t); PARGEMSLR_CHKERR(err);

      v.Clear();

      return err;

   }
   template int PargemslrOrthogonal( SequentialVectorClass<float> &w, DenseMatrixClass<float> &V,
                  float &t, int k, float tol_orth, float tol_reorth);
   template int PargemslrOrthogonal( SequentialVectorClass<double> &w, DenseMatrixClass<double> &V,
                  double &t, int k, double tol_orth, double tol_reorth);
   template int PargemslrOrthogonal( SequentialVectorClass<complexs> &w, DenseMatrixClass<complexs> &V,
                  float &t, int k, float tol_orth, float tol_reorth);
   template int PargemslrOrthogonal( SequentialVectorClass<complexd> &w, DenseMatrixClass<complexd> &V,
                  double &t, int k, double tol_orth, double tol_reorth);
   template int PargemslrOrthogonal( ParallelVectorClass<float> &w, DenseMatrixClass<float> &V,
                  float &t, int k, float tol_orth, float tol_reorth);
   template int PargemslrOrthogonal( ParallelVectorClass<double> &w, DenseMatrixClass<double> &V,
                  double &t, int k, double tol_orth, double tol_reorth);
   template int PargemslrOrthogonal( ParallelVectorClass<complexs> &w, DenseMatrixClass<complexs> &V,
                  float &t, int k, float tol_orth, float tol_reorth);
   template int PargemslrOrthogonal( ParallelVectorClass<complexd> &w, DenseMatrixClass<complexd> &V,
                  double &t, int k, double tol_orth, double tol_reorth);

}
