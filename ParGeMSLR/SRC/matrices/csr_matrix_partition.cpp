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
   template <typename T>
   int CsrMatrixMetisKwayHost( CsrMatrixClass<T> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr)
   {
      /* TODO: OpenMP implementation */
      PARGEMSLR_CHKERR(num_dom <= 0);

      int      nrows, ncols, nnz, col, p, i, i1, i2, j, jj, err = 0;
      int      *A_i, *A_j;
      //T        *A_data;

      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();

      if (nrows != ncols)
      {
         PARGEMSLR_ERROR("METIS partition only works for square matrix.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }

      if( A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("METIS partition only works on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      if(num_dom == 1)
      {
         /* in this case no need to do the partition */
      }

      /* # of nnz */
      nnz = A.GetNumNonzeros();

      /* sort rows of A */
      A.SortRow();

      A_i = A.GetI();
      A_j = A.GetJ();
      //A_data = A.GetData();

      // Prepare data structures used by METIS
      //long int lone = 1;
      IntVectorClass<long int> xadj;
      IntVectorClass<long int> adjncy;
      IntVectorClass<long int> vwgt;
      IntVectorClass<long int> adjwgt;
      IntVectorClass<long int> lmap;

      lmap.Setup(nrows);
      xadj.Setup(nrows+1);
      adjncy.Setup(nnz);
      vwgt.Setup(nrows, true);
      adjwgt.Setup(nnz);

      /* Fill the vectors with the appropriate values */

      map.Setup(nrows);
      perm.Setup(nrows);
      dom_ptr.Setup(num_dom+1, true);

      // Costruct a CSR-like representation of A as required by METIS. Extract and keep the diagonal entries separately.
      xadj[0] = 0;
      jj = 0;
      for ( i = 0 ; i < nrows; i++)
      {
         i1 = A_i[i];
         i2 = A_i[i+1];
         for ( j = i1; j < i2; j++)
         {
            col = A_j[j];
            if (col != i)
            {
               adjncy[jj] = col;
               /* at least should be 1 */
               //adjwgt[jj] = PargemslrMax(adjwgt[jj], lone);
               //adjwgt[jj] = (long int) PargemslrAbs(A_data[j]);
               adjwgt[jj] = 1;
               jj++;
            }
            else
            {
               //vwgt[i] = (long int) PargemslrAbs(A_data[j]);
               //vwgt[i] = PargemslrMax(vwgt[i], lone);
               vwgt[i] = 6;
            }
         }
         if(vwgt[i] == 0)
         {
            /* in this case, we don't have a diagonal entry */
            vwgt[i] = 6;
         }
         xadj[i+1] = jj;
      }

      /* METIS parameters, note that long int is used */

      long int lnrows = (long int)nrows;
      long int lnum_dom = (long int)num_dom;
      long int ledgecut;
      long int ncon = 1;        //no weight in use

      /* call METIS */

      if(lnum_dom > lnrows)
      {
         lnum_dom = lnrows;
      }

      if(lnum_dom >= 8)
      {
         METIS_PartGraphKway(&lnrows, &ncon, xadj.GetData(), adjncy.GetData(), vwgt.GetData(), NULL, adjwgt.GetData(), &lnum_dom, NULL, NULL, NULL, &ledgecut, lmap.GetData());
      }
      else
      {
         METIS_PartGraphRecursive(&lnrows, &ncon, xadj.GetData(), adjncy.GetData(), vwgt.GetData(), NULL, adjwgt.GetData(), &lnum_dom, NULL, NULL, NULL, &ledgecut, lmap.GetData());
      }

      /* transfer from long int to int */
      num_dom = (int) lnum_dom;
      for ( i = 0; i < nrows; i++)
      {
         map[i] = (int)lmap[i];
      }

      /* Determine the number of nodes associated with each partition (subdomain)
       * e.g., two subdomains: dom_ptr = [0,x,y]
       */
      for ( i = 0; i < nrows; i++)
      {
         dom_ptr[map[i]+1] ++;
      }
      /* Accumulate nodes, e.g., two subdomains: dom_ptr = [0,x,x+y] */
      int num_dom2 = 0;
      for ( i = 0; i < num_dom; i++)
      {
         if(dom_ptr[i+1] > 0)
         {
            num_dom2 ++;
         }
         dom_ptr[i+1] += dom_ptr[i];
      }

      /* in this case, we have some empty domains
       * remove them
       */
      if(num_dom2 < num_dom)
      {
         vector_int dom_ptr2, map2;

         dom_ptr2.Setup(num_dom+1);
         map2.Setup(num_dom);
         map2.Fill(-1);

         PARGEMSLR_MEMCPY( dom_ptr2.GetData(), dom_ptr.GetData(), num_dom+1, kMemoryHost, kMemoryHost, int);

         /* locate empty domains */
         dom_ptr.Setup(num_dom2+1);
         dom_ptr.Setup(num_dom2+1);
         num_dom2 = 0;
         dom_ptr[0] = 0;
         for(i = 0 ; i < num_dom ; i ++)
         {
            if(dom_ptr2[i+1] > dom_ptr2[i])
            {
               map2[i] = num_dom2;
               dom_ptr[++num_dom2] = dom_ptr2[i+1];
            }
         }

         /* update the map */
         for ( i = 0; i < nrows; i++)
         {
            map[i] = map2[map[i]];
         }

         dom_ptr2.Clear();
         map2.Clear();
         num_dom = num_dom2;
      }

      /* dom_ptr[num_dom] should be equal to nrow */
      PARGEMSLR_CHKERR(dom_ptr[num_dom] != nrows);

      /* build perm array */
      for ( i = 0; i < nrows; i++)
      {
         /* determine the domain of the ith node */
         p = map[i]; // maps i to its subdomain
         perm[dom_ptr[p]++] = i;
      }

      // Re-adjust: dom_ptr = [0,x,x+y]
      for ( i = num_dom; i > 0; i--)
      {
         dom_ptr[i] = dom_ptr[i-1];
      }
      dom_ptr[0] = 0;

      /* now start building the seperator */
      if(vertexsep && num_dom == 2)
      {
         /* vertex seperator, only supports num_dom == 2 */
         sep.Setup(nrows, true);
         edgecut = 0;

         vector_int sep_size;
         sep_size.Setup(num_dom, true);

         int nmatch, nbound_a, nbound_b;
         vector_int bound_a, bound_b;
         vector_int match_a, match_b;
         CsrMatrixClass<T> B;

         bound_a.Setup(0, (int)ledgecut, kMemoryHost, false);
         bound_b.Setup(0, (int)ledgecut, kMemoryHost, false);

         for(i = 0 ; i < nrows ; i ++)
         {
            i1 = A_i[i];
            i2 = A_i[i+1];
            p = map[i];
            if(p == 0)
            {
               for(j = i1 ; j < i2 ; j ++)
               {
                  col = A_j[j];
                  if(map[col] == 1)
                  {
                     /* have nbhs in different domain */
                     bound_a.PushBack(i);
                     break;
                  }
               }
            }
            else
            {
               for(j = i1 ; j < i2 ; j ++)
               {
                  col = A_j[j];
                  if(map[col] == 0)
                  {
                     /* have nbhs in different domain */
                     bound_b.PushBack(i);
                     break;
                  }
               }
            }
         }

         A.SubMatrix(bound_a, bound_b, kMemoryHost, B);

         int *B_i = B.GetI();
         int *B_j = B.GetJ();

         CsrMatrixMaxMatchingHost( B, nmatch, match_a, match_b);

         nbound_a = match_a.GetLengthLocal();
         nbound_b = match_b.GetLengthLocal();

         /* for CSR matrix, just pick with match_a, otherwise we would need to compute the transpose
          * now we have A, B, and D (unmatched)
          * we pick those from A \intersection D from A, and thse not its neiborhood in B, as not neiborhood
          */
         match_b.Fill(-1);
         for(i = 0 ; i < nbound_a ; i ++)
         {
            if(match_a[i] < 0)
            {
               /* this is an unmatched node */
               i1 = B_i[i];
               i2 = B_i[i+1];
               for(j = i1 ; j < i2 ; j ++)
               {
                  col = B_j[j];
                  /* this is a neiborhood, pick it */
                  match_b[col] = 1;
               }
            }
            else
            {
               /* this is an matched node, put into the seperator */
               sep[bound_a[i]] = 1;
               sep_size[0]++;
               edgecut++;
            }
         }

         for(i = 0 ; i < nbound_b ; i ++)
         {
            if(match_b[i] > 0)
            {
               /* this is an matched node, put into the seperator */
               sep[bound_b[i]] = 1;
               sep_size[1]++;
               edgecut++;
            }
         }

         /* now we need to check if any subdomain
          * has no interior nodes
          */

         for(i = 0 ; i < num_dom ; i ++)
         {
            if(sep_size[i] == dom_ptr[i+1] - dom_ptr[i])
            {
               err = PARGEMSLR_RETURN_METIS_NO_INTERIOR;
               break;
            }
         }
         sep_size.Clear();
      }
      else
      {
         /* edge seperator */
         sep.Setup(nrows, true);
         edgecut = 0;

         vector_int sep_size;
         sep_size.Setup(num_dom, true);

         for(i = 0 ; i < nrows ; i ++)
         {
            i1 = A_i[i];
            i2 = A_i[i+1];
            p = map[i];
            for(j = i1 ; j < i2 ; j ++)
            {
               col = A_j[j];
               if(p != map[col])
               {
                  /* have nbhs in different domain */
                  sep[i] = 1;
                  sep_size[p]++;
                  edgecut++;
                  break;
               }
            }
         }

         /* now we need to check if any subdomain
          * has no interior nodes
          */
         for(i = 0 ; i < num_dom ; i ++)
         {
            if(sep_size[i] == dom_ptr[i+1] - dom_ptr[i])
            {
               err = PARGEMSLR_RETURN_METIS_NO_INTERIOR;
               break;
            }
         }
         sep_size.Clear();
      }

      lmap.Clear();
      xadj.Clear();
      adjncy.Clear();
      vwgt.Clear();
      adjwgt.Clear();

      return err;

   }
   template int CsrMatrixMetisKwayHost( CsrMatrixClass<float> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr);
   template int CsrMatrixMetisKwayHost( CsrMatrixClass<double> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr);
   template int CsrMatrixMetisKwayHost( CsrMatrixClass<complexs> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr);
   template int CsrMatrixMetisKwayHost( CsrMatrixClass<complexd> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr);

   template <typename T>
   int CsrMatrixMaxMatchingHost( CsrMatrixClass<T> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col)
   {
      int      nrows, ncols;

      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();

      if(match_row.GetLengthLocal() != nrows)
      {
         match_row.Setup(nrows);
      }

      if(match_col.GetLengthLocal() != ncols)
      {
         match_col.Setup(ncols);
      }

      if(A.GetDataLocation() == kMemoryDevice || match_row.GetDataLocation() == kMemoryDevice || match_col.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("MaxMatching only works for the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      if(nrows == 0 || ncols == 0)
      {
         nmatch = 0;
         return PARGEMSLR_SUCCESS;
      }

      int         i;
      vector_int  dist;
      int         *A_i = A.GetI();
      int         *A_j = A.GetJ();

      /* set all to unmached */
      match_row.Fill(ncols);
      match_col.Fill(nrows);
      nmatch = 0;
      dist.Setup(nrows+1);

      while( CsrMatrixMaxMatchingBfsHost(nrows, ncols, A_i, A_j, dist, match_row, match_col) )
      {
         for(i = 0 ; i < nrows ; i ++)
         {
            if(match_row[i] == ncols)
            {
               if( CsrMatrixMaxMatchingDfsHost(nrows, ncols, A_i, A_j, dist, match_row, match_col, i) )
               {
                  /* this is an augmenting path */
                  nmatch++;
               }
            }
         }
      }

      for(i = 0 ; i < nrows ; i ++)
      {
         if(match_row[i] == ncols)
         {
            match_row[i] = -1;
         }
      }

      for(i = 0 ; i < ncols ; i ++)
      {
         if(match_col[i] == nrows)
         {
            match_col[i] = -1;
         }
      }

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixMaxMatchingHost( CsrMatrixClass<float> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);
   template int CsrMatrixMaxMatchingHost( CsrMatrixClass<double> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);
   template int CsrMatrixMaxMatchingHost( CsrMatrixClass<complexs> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);
   template int CsrMatrixMaxMatchingHost( CsrMatrixClass<complexd> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);

   bool CsrMatrixMaxMatchingBfsHost(int nrows, int ncols, int *A_i, int *A_j, IntVectorClass<int> &dist, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col)
   {
      int i, j, j1, j2, col, row, qs, qe, maxlength;
      vector_int queue;

      /* we can't visit more than nrows times */
      maxlength = nrows+1;

      queue.Setup(maxlength);
      qe = 0;
      qs = 0;

      for(i = 0 ; i < nrows ; i ++)
      {
         if(match_row[i] == ncols)
         {
            /* free vertex, enqueue */
            dist[i] = 0;
            queue[qe++] = i;
         }
         else
         {
            /* matched vertex */
            dist[i] = maxlength;
         }
      }
      dist[nrows] = maxlength;

      while(qe > qs)
      {
         /* dequeue */
         i = queue[qs++];
         if( dist[i] < dist[nrows])
         {
            /* we only add the shortest ones
             * If i == nrows, this is the end node.
             * Otherwise:
             *
             * if dist[i] == 0, this is a free vertex
             * the connection from i to any of its nbhd is through unmatched edge.
             *
             * if dist[i] > 0, this is a matched vertex connected with another matched vertex
             * through matched edge
             */
            j1 = A_i[i];
            j2 = A_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               /* for each row find all the col nbhd
                * note that we are going through an unmatched edge
                * also we won't go back to the same u node
                */
               col = A_j[j];

               /* find the match row of this col
                * if row == nrows this is unmatched
                */
               row = match_col[col];

               /* go through the matched edge
                * note that we won't have row == i
                */
               if(dist[row] == maxlength)
               {
                  /* if the matched edge has not yet been inside the queue */
                  dist[row] = dist[i] + 1;
                  queue[qe++] = row;
               }
            }
         }
      }

      queue.Clear();

      return dist[nrows] < maxlength;
   }

   bool CsrMatrixMaxMatchingDfsHost(int nrows, int ncols, int *A_i, int *A_j, IntVectorClass<int> &dist, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col, int i)
   {
      int j, j1, j2, col, row, maxlength, distip1;

      maxlength = nrows + 1;
      distip1 = dist[i]+1;

      if(i < nrows)
      {
         j1 = A_i[i];
         j2 = A_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = A_j[j];
            row = match_col[col];
            if(dist[row] == distip1)
            {
               /* possible next, search */
               if( CsrMatrixMaxMatchingDfsHost(nrows, ncols, A_i, A_j, dist, match_row, match_col, row) )
               {
                  /* reset match */
                  match_row[i] = col;
                  match_col[col] = i;
                  return true;
               }
            }
         }
         /* deadend, avoid visit here again */
         dist[i] = maxlength;
         return false;
      }
      return true;
   }

   int ParmetisKwayHost(vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, long int &num_dom, vector_long &map, parallel_log &parlog)
   {
      /* Declare variables */
      long int          i, j, j1, j2, k, nrow, refs;
      long int          wtflag, numflag, edgecut, num_dom2, ncon, idx, idx2, idx3;

      int               nI, nC;
      long int          nI_l, nC_l, nI_global, nC_global;
      vector_int        isolate;

      /* first we mark those isolated nodes, that is, those nodes that has no connection with other nodes */
      MPI_Comm          comm;
      int               np, myid;
      parlog.GetMpiInfo(np, myid, comm);

      /* first check the size of the problem */
      if(vtxdist[np] < np || vtxdist[np] < num_dom)
      {
         /* In this case, we have empty processors, or ndom is larger than the problem size
          * there is no points for keep doing the partition
          */
         //PARGEMSLR_WARNING("Problem too small, try to reduce number of suddomains.");
         num_dom = 0;
         return PARGEMSLR_RETURN_METIS_PROBLEM_TOO_SMALL;
      }

      nrow = vtxdist[myid + 1] - vtxdist[myid];

      isolate.Setup(nrow, true);
      nI = 0;

      j1 = xadj[0];
      for(i = 0 ; i < nrow ; i ++)
      {
         j2 = xadj[i+1];
         if(j1 == j2)
         {
            isolate[i] = 1;
            nI ++;
         }
         j1 = j2;
      }

      nC = nrow - nI;

      /* now we have local isolated nI and connected nC  */
      nI_l = nI;
      PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &nI_l, &nI_global, 1, MPI_SUM,comm));
      nC_global = vtxdist[np] - nI_global;

      if(nC_global < np || nC_global < num_dom)
      {
         /* in this case, we don't have enough "connected" nodes,
          * we apply an naive partition without calling ParMETIS.
          * We assign node to i%num_dom
          */
         map.Setup(nrow);
         for(i = 0, j = vtxdist[myid] ; i < nrow ; i ++, j++)
         {
            /* assign global node number ig = ig % num_dom */
            map[i] = j % num_dom;
         }
         return PARGEMSLR_SUCCESS;
      }

      /* if we reach here, we have enough connected nodes (more than np), now we need to check
       * if we need redistribute
       * Exaplme:
       * np  node
       *  0  1 3 4
       *  1
       *  2  9 10 12
       *  => redistribute:
       * np  node
       *  0  1 3
       *  1  4 9
       *  2  10 12
       */
      double      lb_factor, lb_factor_global;

      lb_factor = nC / (double)(nC_global/np);
      PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &lb_factor, &lb_factor_global, 1, MPI_MIN,comm));

      if(lb_factor_global > pargemslr_global::_metis_loading_balance_tol)
      {
         /* fits the loading balance, no need to re-assign */

         /* partition */
         vector_long       marker, marker2;
         vector_seq_double tpwgts, ubvec;

         /* setup helper arrays and parameters */
         map.Setup(nrow);
         marker.Setup(num_dom);
         marker.Fill(-1);
         tpwgts.Setup(num_dom);
         tpwgts.Fill(1.0/num_dom);
         ubvec.Setup(1);
         ubvec.Fill(1.05);

         long int option[40]     = {0};
         wtflag                  = 0;      //  2: Weights on the vertices only (adjwgt is NULL).
         numflag                 = 0;      //  C-style
         ncon                    = 1;      //  no weight in use

         ParMETIS_V3_PartKway(vtxdist.GetData(), xadj.GetData(), adjncy.GetData(), NULL, NULL, &wtflag, &numflag,
                                    &ncon, &num_dom, tpwgts.GetData(), ubvec.GetData(), &option[0], &edgecut, map.GetData(), &comm);

         for(refs = 0 ; refs < pargemslr_global::_metis_refine ; refs++)
         {
            ParMETIS_V3_RefineKway(vtxdist.GetData(), xadj.GetData(), adjncy.GetData(), NULL, NULL, &wtflag, &numflag,
                                    &ncon, &num_dom, tpwgts.GetData(), ubvec.GetData(), &option[0], &edgecut, map.GetData(), &comm);
         }

         /* mark local domains */
         for (i = 0; i < nrow; i++)
         {
            marker[map[i]] = 1;
         }


         /* check for empty domain */
         PARGEMSLR_MPI_CALL( PargemslrMpiAllreduceInplace( marker.GetData(), num_dom, MPI_MAX, comm) );

         num_dom2 = 0;
         for (i = 0; i < num_dom; i++)
         {
            if(marker[i] > 0)
            {
               marker[i] = num_dom2++;
            }
         }

         /* now swap the marker number for balence size
          * 0 1 2 3 4 5 ... k*np-1 into
          * 0 np 2np ... 1 np+1 2np+1
          */
         idx = 0;
         idx2 = 0;
         idx3 = 0;
         marker2.Setup(num_dom2);
         while(idx < num_dom2)
         {
            marker2[idx2] = idx;
            idx ++;
            idx2 += np;
            if(idx2 >= num_dom2)
            {
               idx3++;
               idx2 = idx3;
            }
         }

         /* remove empty domains */
         for (i = 0; i < nrow; i++)
         {
            map[i] = marker2[marker[map[i]]];
         }

         num_dom = num_dom2;

         tpwgts.Clear();
         ubvec.Clear();
         //vwgt.Clear();
         marker.Clear();
         marker2.Clear();

         return PARGEMSLR_SUCCESS;

      }
      else
      {
         /* number of local nodes on some processors is too small, redistribute */
         //PARGEMSLR_WARNING("Redistribute.");

         int               pid;
         vector_long       vtxdistc, vtxdisteven;
         vector_long       vtxdist2, xadj2, adjncy2;

         /* we redistribute those connected nodes to each processor */
         long int          nC1, nC2;

         nC1      = nC_global/np;
         nC2      = nC_global%np;

         vtxdistc.Setup(np+1);
         vtxdist2.Setup(np+1);
         vtxdisteven.Setup(np+1);

         vtxdistc[0] = 0;
         vtxdisteven[0] = 0;
         vtxdisteven[np] = nC_global;

         nC_l = nC;
         PARGEMSLR_MPI_CALL( PargemslrMpiAllgather( &nC_l, 1, vtxdistc.GetData()+1, comm) );

         for(i = 1 ; i < np ; i ++)
         {
            vtxdistc[i] += vtxdistc[i-1];
            if(i <= nC2)
            {
               vtxdisteven[i] = vtxdisteven[i-1] + nC1 + 1;
            }
            else
            {
               vtxdisteven[i] = vtxdisteven[i-1] + nC1;
            }
         }

         vtxdistc[np] = nC_global;

         /* now setup vtxdist2 */
         vtxdist2[0] = 0;
         vtxdist2[np] = vtxdist[np];

         j = 0;
         j1 = vtxdistc[myid];
         for(i = 0 ; i < np-1 ; i ++)
         {
            /* Example:
             * [0 3 6 9]
             * [0 6 6 9]
             * search for 3, should be on p0
             * search for 6, both p0, p1, and p2 works
             */

            if( vtxdistc.BinarySearch(vtxdisteven[i+1], pid, true) < 0)
            {
               /* In this case, we haven't found it, belongs to the previous MPI rank */
               pid--;
            }

            /* note that vtxdisteven[i+1] is nonzero */

            /* the last element of vtxdistc[i+1] belongs to processor ps */
            if(myid == pid)
            {
               /* search for it */
               while(j1 < vtxdisteven[i+1])
               {
                  j1++;
                  while(isolate[j])
                  {
                     j++;
                  }
                  j++;
               }
               vtxdist2[i+1] = vtxdist[myid] + j;
            }
            PARGEMSLR_MPI_CALL( PargemslrMpiBcast( vtxdist2.GetData()+i+1, 1, pid, comm) );
         }

         /* now get the amount we need to send to other processor
          * search in the array, if not found,
          */
         int                        ps, pe, nsend, nrecv, nsendrecv, nadj2, toid;
         vector_int                 sends, recvs, send_to_v, recv_from_v, send_size_v, recv_size_v, send_size2_v, recv_size2_v;
         std::vector<vector_int >   send_count_v2,recv_count_v2;
         std::vector<MPI_Request >  request_v;

         if( vtxdist2.BinarySearch(vtxdist[myid], ps, true) < 0)
         {
            /* In this case, we haven't found it, belongs to the previous MPI rank */
            ps--;
         }

         if(ps == np)
         {
            ps--;
         }

         if( vtxdist2.BinarySearch(vtxdist[myid+1], pe, true) < 0)
         {
            /* In this case, we haven't found it, belongs to the previous MPI rank */
            pe--;
         }

         if(pe == np)
         {
            pe--;
         }

         sends.Setup(np, true);
         recvs.Setup(np, true);

         if(nrow > 0)
         {
            for(i = ps ; i <= pe ; i ++)
            {
               sends[i] = 1;
            }
         }

         PARGEMSLR_MPI_CALL( MPI_Alltoall( sends.GetData(), 1, MPI_INT, recvs.GetData(), 1, MPI_INT, comm) );

         nsend = 0;
         nrecv = 0;
         for(i = 0 ; i < np ; i ++)
         {
            if(sends[i] > 0)
            {
               send_to_v.PushBack(i);
               nsend++;
            }
            if(recvs[i] > 0)
            {
               recv_from_v.PushBack(i);
               nrecv++;
            }
         }

         nsendrecv = nsend + nrecv;
         send_size_v.Setup(nsend);
         recv_size_v.Setup(nrecv);
         request_v.resize(nsendrecv);

         send_count_v2.resize(nsend);

         j = 0;
         j1 = vtxdist[myid];
         for(i = 0 ; i < nsend-1 ; i ++)
         {
            /* get the amound of data we need to send */
            toid = send_to_v[i];
            j2 = vtxdist2[toid+1];
            send_size_v[i] = j2 - j1;
            send_count_v2[i].Setup(send_size_v[i]);
            for(k = 0 ; k < send_size_v[i] ; k ++)
            {
               send_count_v2[i][k] = xadj[j+1] - xadj[j];
               j++;
            }
            j1 = j2;
         }
         if(nsend > 0)
         {
            i = nsend-1;
            j2 = vtxdist[myid+1];
            send_size_v[i] = j2 - j1;
            send_count_v2[i].Setup(send_size_v[i]);
            for(k = 0 ; k < send_size_v[i] ; k ++)
            {
               send_count_v2[i][k] = xadj[j+1] - xadj[j];
               j++;
            }
         }

         j = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_size_v.GetData()+i, 1, send_to_v[i], send_to_v[i], comm, &(request_v[j++]) ) );
         }

         for(i = 0 ; i < nrecv ; i ++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_size_v.GetData()+i, 1, recv_from_v[i], myid, comm, &(request_v[j++]) ) );
         }

         PARGEMSLR_MPI_CALL( MPI_Waitall( nsendrecv, request_v.data(), MPI_STATUSES_IGNORE) );

         /* prepare the send count */

         recv_count_v2.resize(nrecv);

         j = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_count_v2[i].GetData(), send_size_v[i], send_to_v[i], send_to_v[i], comm, &(request_v[j++]) ) );
         }

         for(i = 0 ; i < nrecv ; i ++)
         {
            recv_count_v2[i].Setup(recv_size_v[i]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_count_v2[i].GetData(), recv_size_v[i], recv_from_v[i], myid, comm, &(request_v[j++]) ) );
         }

         PARGEMSLR_MPI_CALL( MPI_Waitall( nsendrecv, request_v.data(), MPI_STATUSES_IGNORE) );

         /* prepare to send the adjncy */
         send_size2_v.Setup(nsend, true);
         recv_size2_v.Setup(nrecv, true);

         xadj2.Setup(vtxdist2[myid+1]-vtxdist2[myid]+1);
         xadj2[0] = 0;

         nadj2 = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            for( j = 0; j < send_size_v[i] ; j ++)
            {
               send_size2_v[i] += send_count_v2[i][j];
            }
            //nadj2 += send_size2_v[i];
         }

         k = 0;
         for(i = 0 ; i < nrecv ; i ++)
         {
            for( j = 0; j < recv_size_v[i] ; j ++)
            {
               recv_size2_v[i] += recv_count_v2[i][j];
               xadj2[k+1] = xadj2[k] + recv_count_v2[i][j];
               k++;
            }
            nadj2 += recv_size2_v[i];
         }

         adjncy2.Setup(nadj2);

         j = 0;
         j1 = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( adjncy.GetData()+j1, send_size2_v[i], send_to_v[i], send_to_v[i], comm, &(request_v[j++]) ) );
            j1 += send_size2_v[i];
         }

         j1 = 0;
         for(i = 0 ; i < nrecv ; i ++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( adjncy2.GetData()+j1, recv_size2_v[i], recv_from_v[i], myid, comm, &(request_v[j++]) ) );
            j1 += recv_size2_v[i];
         }

         //PARGEMSLR_MPI_CALL( MPI_Waitany( sendrecv, this->_comm_helper._requests_v.data(), &reqidx, MPI_STATUSES_IGNORE) );
         PARGEMSLR_MPI_CALL( MPI_Waitall( nsendrecv, request_v.data(), MPI_STATUSES_IGNORE) );

         /* partition of the redistributed graph */
         vector_long       marker, marker2, map2;
         vector_seq_double tpwgts, ubvec;

         /* setup helper arrays and parameters */
         map.Setup(nrow);
         map2.Setup(vtxdist2[myid+1]-vtxdist2[myid]);
         marker.Setup(num_dom);
         marker.Fill(-1);
         tpwgts.Setup(num_dom);
         tpwgts.Fill(1.0/num_dom);
         ubvec.Setup(1);
         ubvec.Fill(1.05);

         long int option[40]     = {0};
         wtflag                  = 0;      //  2: Weights on the vertices only (adjwgt is NULL).
         numflag                 = 0;      //  C-style
         ncon                    = 1;      //  no weight in use

         ParMETIS_V3_PartKway(vtxdist2.GetData(), xadj2.GetData(), adjncy2.GetData(), NULL, NULL, &wtflag, &numflag,
                                    &ncon, &num_dom, tpwgts.GetData(), ubvec.GetData(), &option[0], &edgecut, map2.GetData(), &comm);

         for(refs = 0 ; refs < pargemslr_global::_metis_refine ; refs++)
         {
            ParMETIS_V3_RefineKway(vtxdist2.GetData(), xadj2.GetData(), adjncy2.GetData(), NULL, NULL, &wtflag, &numflag,
                                    &ncon, &num_dom, tpwgts.GetData(), ubvec.GetData(), &option[0], &edgecut, map2.GetData(), &comm);
         }

         /* send the map data back */

         j = 0;
         j1 = 0;
         for(i = 0 ; i < nrecv ; i ++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( map2.GetData()+j1, recv_size_v[i], recv_from_v[i], myid, comm, &(request_v[j++]) ) );
            j1 += recv_size_v[i];
         }

         j1 = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( map.GetData()+j1, send_size_v[i], send_to_v[i], send_to_v[i], comm, &(request_v[j++]) ) );
            j1 += send_size_v[i];
         }

         PARGEMSLR_MPI_CALL( MPI_Waitall( nsendrecv, request_v.data(), MPI_STATUSES_IGNORE) );

         /* mark local domains */
         for (i = 0; i < nrow; i++)
         {
            marker[map[i]] = 1;
         }

         /* check for empty domain */
         PARGEMSLR_MPI_CALL( MPI_Allreduce(MPI_IN_PLACE, marker.GetData(), num_dom, MPI_LONG, MPI_MAX, comm) );

         num_dom2 = 0;
         for (i = 0; i < num_dom; i++)
         {
            if(marker[i] > 0)
            {
               marker[i] = num_dom2++;
            }
         }

         /* now swap the marker number for balence size
          * 0 1 2 3 4 5 ... k*np-1 into
          * 0 np 2np ... 1 np+1 2np+1
          */
         idx = 0;
         idx2 = 0;
         idx3 = 0;
         marker2.Setup(num_dom2);
         while(idx < num_dom2)
         {
            marker2[idx2] = idx;
            idx ++;
            idx2 += np;
            if(idx2 >= num_dom2)
            {
               idx3++;
               idx2 = idx3;
            }
         }

         /* remove empty domains */
         for (i = 0; i < nrow; i++)
         {
            map[i] = marker2[marker[map[i]]];
         }

         num_dom = num_dom2;

         sends.Clear();
         recvs.Clear();
         send_to_v.Clear();
         recv_from_v.Clear();
         send_size_v.Clear();
         recv_size_v.Clear();
         send_size2_v.Clear();
         recv_size2_v.Clear();

         for(i = 0 ; i < nsend ; i ++)
         {
            send_count_v2[i].Clear();
         }

         for(i = 0 ; i < nrecv ; i ++)
         {
            recv_count_v2[i].Clear();
         }

         std::vector<vector_int >().swap(send_count_v2);
         std::vector<vector_int >().swap(recv_count_v2);

         std::vector<MPI_Request >().swap(request_v);

         tpwgts.Clear();
         ubvec.Clear();
         //vwgt.Clear();
         marker.Clear();
         marker2.Clear();

         return PARGEMSLR_SUCCESS;

      }
   }

   int ParmetisNodeND(vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, long int &num_dom, vector_long &map, parallel_log &parlog)
   {
      /* Declare variables */
      vector_long       order, sizes;

      long int          i, nrow;
      long int          numflag;
      int               idx;

      /* MPI */
      MPI_Comm    comm;
      int         np, myid;

      parlog.GetMpiInfo(np, myid, comm);

      nrow = vtxdist[myid + 1] - vtxdist[myid];

      /* setup helper arrays and parameters */
      map.Setup(nrow);
      order.Setup(nrow);
      sizes.Setup(2*np+1, true);

      long int option[40]     = {0};
      numflag                 = 0;      //  C-style

      /* parMetis ND partition into log(p) comp
       * order[i]: now global number of i-th local vertex.
       * sizez[i]: each of the sizes arrays are identical.
       *
       */
      ParMETIS_V3_NodeND(vtxdist.GetData(), xadj.GetData(), adjncy.GetData(), &numflag, &option[0], order.GetData(), sizes.GetData()+1, &comm);

      //PARGEMSLR_GLOBAL_SEQUENTIAL_RUN({order.Plot(0,0,6);});

      /* now setup the map array */
      sizes[0] = 0;

      num_dom = 2*np;

      for(i = 0; i < num_dom; i++)
      {
         sizes[i+1] += sizes[i];
      }

      /* ignore empty ones at the end */
      while(num_dom > 0 && (sizes[num_dom] == sizes[num_dom-1]))
      {
         num_dom--;
      }

      /* resize the size array */
      sizes.Resize( num_dom+1, true, false);

      /* set the map array */
      for(i = 0 ; i < nrow ; i ++)
      {
         /* if we didn't find it, the real domain number is the smaller one
          * Example: [0, 2, 4], if we search for 1, the result is 1, the domain number is 0
          */
         if(sizes.BinarySearch(order[i], idx, true) < 0 )
         {
            idx--;
         }
         map[i] = idx;
      }

      sizes.Clear();
      order.Clear();

      return PARGEMSLR_SUCCESS;
   }
}
