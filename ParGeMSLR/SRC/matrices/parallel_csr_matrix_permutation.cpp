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
#include "../preconditioners/ilu.hpp"

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
   int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<T> &A, bool vertexsep, int &nlev, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last)
   {
      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Recursive kway partition only works for host matrices.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      int                              err = 0;
      int                              clvl;
      vector_long                      vtxdist, xadj, adjncy;

      /* The ND version currently removed, limitid weak scalability
      if(vertexsep && ( (ncomp)&(ncomp-1) )==0 )
      {
         clvl = 1;
         mapptr_v.Setup(1, true);
         err = SetupPermutationParallelRKwayRecursive2( A, clvl, nlev, ncomp, minsep, kmin, kfactor, map_v, mapptr_v, bj_last, A); PARGEMSLR_CHKERR(err);
      }
      */

      A.GetGraphArrays(vtxdist, xadj, adjncy);

      /* call rKway function */
      clvl = 1;
      mapptr_v.Setup(1, true);

      err = SetupPermutationParallelRKwayRecursive( vtxdist, xadj, adjncy, vertexsep, clvl, nlev, ncomp, minsep, kmin, kfactor, map_v, mapptr_v, bj_last, A); PARGEMSLR_CHKERR(err);

      return err;
   }
   template int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<float> &A, bool vertexsep, int &nlev, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last);
   template int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<double> &A, bool vertexsep, int &nlev, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last);
   template int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<complexs> &A, bool vertexsep, int &nlev, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last);
   template int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<complexd> &A, bool vertexsep, int &nlev, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last);

   int SetupPermutationParallelRKwayRecursive(vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, bool vertexsep, int clvl, int &tlvl, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog)
   {
      long int          i, j, n_local, nA, col, ncomp2;
      int               err = 0;
      vector_long       map;
      vector_int        map2_v;
      vector_int        vtxsep;
      vector_long       vtxdist_s, xadj_s, adjncy_s;

      MPI_Comm          comm;
      int               myid, np;
      parlog.GetMpiInfo(np, myid, comm);

      nA = vtxdist[np];
      n_local = vtxdist[myid+1] - vtxdist[myid];

      /* main loop */
      if (minsep < nA && ncomp <= nA && clvl < tlvl)
      {

         /* call parMetis for partition */
         ncomp2 = ncomp;

         err = ParmetisKwayHost( vtxdist, xadj, adjncy, ncomp2, map, parlog); //PARGEMSLR_CHKERR(err);

         if( err || ncomp2 < ncomp)
         {
            /* in this case, we don't have enough domains, stop here */
            if(bj_last && nA >= np)
            {
               /* treat last level with block-Jacobi like partition
                * we assign np subdomains to the last level
                */

               long int       bjdom1, bjdom2, k;
               vector_long    bjdisps, bjns;

               bjdom1 = nA / np;
               bjdom2 = nA % np;

               bjdisps.Setup(np+1);
               bjns.Setup(np);

               for(i = 0 ; i < np ; i ++)
               {
                  if(i < bjdom2)
                  {
                     bjns[i] = bjdom1 + 1;
                  }
                  else
                  {
                     bjns[i] = bjdom1;
                  }
               }

               bjdisps[0] = 0;
               for(i = 0 ; i < np ; i ++)
               {
                  bjdisps[i+1] = bjdisps[i] + bjns[i];
               }

               col = mapptr_v.Back();
               mapptr_v.PushBack(col+np);
               map_v.Setup(n_local);

               j = 0;
               for(i = 0, k = vtxdist[myid]; i < n_local ; i ++, k++)
               {
                  while(bjdisps[j+1] <= k)
                  {
                     /* in this case, current node blongs to the next one
                      * we want k in [bjdisps[j], bjdisps[j+1])
                      * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                      */
                     j++;
                  }
                  map_v[i] = j+col;
               }
            }
            else
            {
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+1);
               map_v.Setup(n_local);
               map_v.Fill(col);
            }

            tlvl = clvl;
         }
         else
         {
            /* get the separator. In array vtxsep, mark as 1 to be in the seprator */
            err = ParallelRKwayGetSeparator( vtxdist, xadj, adjncy, vertexsep, vtxdist_s, xadj_s, adjncy_s, map, ncomp, vtxsep, parlog);

            if(err == -1)
            {
               /* No next level availiable */
               if(bj_last && nA >= np)
               {
                  /* treat last level with block-Jacobi like partition
                   * we assign np subdomains to the last level
                   */

                  long int       bjdom1, bjdom2, k;
                  vector_long    bjdisps, bjns;

                  bjdom1 = nA / np;
                  bjdom2 = nA % np;

                  bjdisps.Setup(np+1);
                  bjns.Setup(np);

                  for(i = 0 ; i < np ; i ++)
                  {
                     if(i < bjdom2)
                     {
                        bjns[i] = bjdom1 + 1;
                     }
                     else
                     {
                        bjns[i] = bjdom1;
                     }
                  }

                  bjdisps[0] = 0;
                  for(i = 0 ; i < np ; i ++)
                  {
                     bjdisps[i+1] = bjdisps[i] + bjns[i];
                  }

                  col = mapptr_v.Back();
                  mapptr_v.PushBack(col+np);
                  map_v.Setup(n_local);

                  j = 0;
                  for(i = 0, k = vtxdist[myid]; i < n_local ; i ++, k++)
                  {
                     while(bjdisps[j+1] <= k)
                     {
                        /* in this case, current node blongs to the next one
                         * we want k in [bjdisps[j], bjdisps[j+1])
                         * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                         */
                        j++;
                     }
                     map_v[i] = j+col;
                  }
               }
               else
               {
                  col = mapptr_v.Back();
                  mapptr_v.PushBack(col+1);
                  map_v.Setup(n_local);
                  map_v.Fill(col);
               }

               tlvl = clvl;
               return PARGEMSLR_SUCCESS;
            }
            PARGEMSLR_CHKERR(err);

            mapptr_v.PushBack(mapptr_v.Back()+ncomp2);

            if(ncomp > kmin)
            {
               ncomp = ncomp / kfactor;
               if(ncomp < kmin)
               {
                  ncomp = kmin;
               }
            }

            // (Increase clvl by 1)
            SetupPermutationParallelRKwayRecursive( vtxdist_s, xadj_s, adjncy_s, vertexsep, clvl+1, tlvl, ncomp, minsep, kmin, kfactor, map2_v, mapptr_v, bj_last, parlog);

            /* udpate map information */
            map_v.Setup(n_local);
            j = 0;
            for(i = 0 ; i < n_local ; i ++)
            {
               if(vtxsep[i] <= 0)
               {
                  /* interior nodes */
                  map_v[i] = map[i] + mapptr_v[clvl-1];
               }
               else
               {
                  /* exterior nodes */
                  map_v[i] = map2_v[j++];
               }
            }
         }
      }
      else
      {
         /* in this case, we don't have enough domains, stop here
          * a special case is when nA == 0
          */
         if(nA > 0)
         {
            if(bj_last && nA >= np)
            {
               /* treat last level with block-Jacobi like partition
                * we assign np subdomains to the last level
                */

               long int       bjdom1, bjdom2, k;
               vector_long    bjdisps, bjns;

               bjdom1 = nA / np;
               bjdom2 = nA % np;

               bjdisps.Setup(np+1);
               bjns.Setup(np);

               for(i = 0 ; i < np ; i ++)
               {
                  if(i < bjdom2)
                  {
                     bjns[i] = bjdom1 + 1;
                  }
                  else
                  {
                     bjns[i] = bjdom1;
                  }
               }

               bjdisps[0] = 0;
               for(i = 0 ; i < np ; i ++)
               {
                  bjdisps[i+1] = bjdisps[i] + bjns[i];
               }

               col = mapptr_v.Back();
               mapptr_v.PushBack(col+np);
               map_v.Setup(n_local);

               j = 0;
               for(i = 0, k = vtxdist[myid]; i < n_local ; i ++, k++)
               {
                  while(bjdisps[j+1] <= k)
                  {
                     /* in this case, current node blongs to the next one
                      * we want k in [bjdisps[j], bjdisps[j+1])
                      * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                      */
                     j++;
                  }
                  map_v[i] = j+col;
               }
            }
            else
            {
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+1);
               map_v.Setup(n_local);
               map_v.Fill(col);
            }

            tlvl = clvl;
         }
         else
         {
            tlvl = clvl-1;
         }
      }

      return PARGEMSLR_SUCCESS;
   }

   int ParallelRKwayGetSeparator( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, bool vertexsep, vector_long &vtxdist_s,  vector_long &xadj_s,  vector_long &adjncy_s, vector_long &map, int num_dom, vector_int &vtxsep, parallel_log &parlog)
   {
      long int                   i, ii, j, j1, j2;
      long int                   n_local, n_start, n_end, dom, col, nnz, n_local_s, local_diff, global_diff;
      int                        id, idx;
      vector_int                 ids, marker2;
      vector_long                n_local_ss;

      std::unordered_map<long int, int> col_map_hash;
      int                        ncols;
      vector_long                cols;
      vector_int                 col_ids;

      std::unordered_map<long int, int> col_map_uncertain_hash;
      vector_int                 sendsize, recvsize;
      std::vector<vector_long>   send_v2, recv_v2;
      std::vector<vector_int>    send2_v2, recv2_v2;

      MPI_Comm                comm;
      int                     myid, np, numwaits;
      vector<MPI_Request>     requests;

      parlog.GetMpiInfo(np, myid, comm);

      n_start = vtxdist[myid];
      n_end = vtxdist[myid+1];
      n_local = n_end - n_start;

      if(n_local == 0)
      {
         /* empty, no need to setup */
         nnz = 0;
         vtxsep.Clear();
      }
      else
      {
         nnz = xadj[n_local];
         vtxsep.Setup(n_local, true);
         ids.Setup(nnz, true);
      }

      /* -------------------------
       * Step 1: put all local columns into the hash table
       * -------------------------
       */
      ncols = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index
             * note that there might be duplicate entries
             */
            auto find_col =  col_map_hash.find(col);
            if(find_col == col_map_hash.end())
            {
               /* a new one */
               if(vtxdist.BinarySearch( col, id, true) < 0)
               {
                  /* in this case, col fall in between */
                  id--;
               }

               cols.PushBack(col);
               col_ids.PushBack(id);
               col_map_hash[col] = ncols;
               ncols++;
            }
         }
      }

      /* ids[j] is the MPI process adjncy[j] belongs to */
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index
             * note that there might be duplicate entries
             */
            auto find_col =  col_map_hash.find(col);
            ids[j] = col_ids[find_col->second];
         }
      }

      /* -------------------------
       * Step 2: check local first
       * some rows/cols requires
       * accessing offdiagonal entries
       * -------------------------
       */

      /* This loop find local vtxsep
       * 1. local cols already has multiple maps:
       *    => set vtxsep to 1, separator.
       * 2. all cols are local, and all same map values:
       *    => set vtxsep to 0, interior.
       * 3. all local cols same map, however, have exterior cols
       *    => set vtxsep to -1, TBD
       */
      for(i = 0 ; i < n_local ; i ++)
      {
         /* dom is the domain of the row */
         dom = map[i];
         j1 = xadj[i];
         j2 = xadj[i+1];

         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            id = ids[j]; /* this is the id this col belongs to */

            /* we can only check the map value of local parts */
            if(id == myid)
            {
               col -= n_start;
               if(map[col] != dom)
               {
                  /* this is diagonal, exterior node */
                  vtxsep[i] = 1;
                  break;
               }
            }
            else
            {
               /* if has offd, mark to -1 instead */
               vtxsep[i] = -1;
            }
         }
      }

      /* -------------------------
       * Step 3: check remaining
       * -------------------------
       */

      sendsize.Setup(np, true);
      recvsize.Setup(np, true);
      send_v2.resize(np);
      recv_v2.resize(np);
      send2_v2.resize(np);
      recv2_v2.resize(np);

      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] == -1 || (vtxsep[i] == 1 && vertexsep))
         {
            /* this is a target column */
            j1 = xadj[i];
            j2 = xadj[i+1];

            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */

               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col == col_map_uncertain_hash.end())
                  {
                     /* breaking news! a NEW column! */
                     send_v2[id].PushBack(col);
                     col_map_uncertain_hash[col] = sendsize[id];
                     sendsize[id]++;
                  }
               }
            }
         }
      }

      /* communicate send and recv size */
      PARGEMSLR_MPI_CALL( MPI_Alltoall( sendsize.GetData(), 1, MPI_INT, recvsize.GetData(), 1, MPI_INT, comm) );

      /* then apply communication */

      requests.resize(2*np);

      /* first send cols */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            /* myid have data for processor i */
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv_v2[i].Setup(recvsize[i]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      PARGEMSLR_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );

      /* get the map value */
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv2_v2[i].Setup(recvsize[i]);
            for(j = 0 ; j < recvsize[i]; j ++)
            {
               recv2_v2[i][j] = map[recv_v2[i][j] - n_start];
            }
         }
      }

      /* then apply communication again */

      /* or MPI_Alltoallv? */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            /* myid have data for processor i */
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( recv2_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            send2_v2[i].Setup(sendsize[i]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( send2_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      PARGEMSLR_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );

      /* finally check the received columns */

      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] == -1)
         {
            /* this is a target column */
            dom = map[i];
            j1 = xadj[i];
            j2 = xadj[i+1];

            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */

               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(send2_v2[id][find_col->second] != dom)
                  {
                     vtxsep[i] = 1;
                     break;
                  }
               }
            }

            /* if still -1, this is interior */
            if(vtxsep[i] == -1)
            {
               vtxsep[i] = 0;
            }
         }
      }

      /* -------------------------
       * Step 3.5: update the edge
       * separator into a rough
       * vertex separator
       * -------------------------
       */

      /* We use a simply recursive algorithm
       * Each time split node set V into V1 and V2
       * Find all edges across V1 and V2, mark one of
       * the end as vertex separator.
       * Recursively apply this strategy to V1 and V2
       */

      if(vertexsep)
      {
         /* search vector on each level:
          * 0          m          ndom  2+1
          * 0    n1    m    n2    ndom  4+1
          * 0 o1 n1 o2 m o3 n2 o4 ndom  8+1
          * 2^{n + 1) + n - 2
          */
         vector_long tree;

         int k, ts, te, tlen, idxi, leveli, tree_size, tree_level;
         long int n1, n2, n1_g, n2_g, domi;

         bool marker, even;

         tree_size = 2;
         tree_level = 1;

         while( tree_size < num_dom )
         {
            tree_size = tree_size << 1;
            tree_level++;
         }

         tree_size = 2*pow(2,tree_level)+tree_level-2;

         tree.Setup(tree_size, true);

         tree[0] = 0;
         tree[1] = num_dom/2;
         tree[2] = num_dom;

         ts = 3;

         for(i = 1 ; i < tree_level ; i ++)
         {
            /* [0-2] [3-7] [8-17] [7-14] */
            te = ts + pow(2,i+1) + 1;

            for(j = ts, k = ts-pow(2,i)-1 ; j < te ; j +=2, k+=1)
            {
               tree[j] = tree[k];
            }
            for(j = ts + 1, k = ts-pow(2,i)-1 ; j < te - 1 ; j +=2, k+=1)
            {
               tree[j] = (tree[k+1]+tree[k])/2;
            }
            ts = te;
         }

         ts = 0;
         for(leveli = 0 ; leveli < tree_level ; leveli ++)
         {
            /* search within each level */
            tlen = pow(2,leveli+1) + 1;
            te = ts + tlen;

            n1 = 0;
            n2 = 0;

            vector_long treei;
            std::unordered_map<long int, int> tree_idx_hash;

            treei.SetupPtr( tree, tlen, ts);

            /* before we start, we first update the index of each off-diagonal columns to avoid duplicate search */
            for(i = 0 ; i < n_local ; i ++)
            {
               if(vtxsep[i] == 1)
               {
                  dom = map[i];

                  /* check if we know the index of the local domain */
                  auto find_dom =  tree_idx_hash.find(dom);
                  if(find_dom == tree_idx_hash.end())
                  {
                     /* a new one */
                     if( treei.BinarySearch( dom, idx, true) < 0)
                     {
                        /* In this case, we haven't found it, belongs to the previous inteval */
                        idx--;
                     }

                     tree_idx_hash[dom] = idx;
                  }

                  /* new check nbhds */
                  j1 = xadj[i];
                  j2 = xadj[i+1];
                  for(j = j1 ; j < j2 ; j ++)
                  {

                     col = adjncy[j];
                     id = ids[j]; /* this is the id this col belongs to */

                     if(id != myid)
                     {
                        auto find_col = col_map_uncertain_hash.find(col);
                        domi = send2_v2[id][find_col->second];
                     }
                     else
                     {
                        domi = map[col-n_start];
                     }

                     auto find_dom =  tree_idx_hash.find(domi);
                     if(find_dom == tree_idx_hash.end())
                     {
                        /* a new one */
                        if( treei.BinarySearch( domi, idx, true) < 0)
                        {
                           /* In this case, we haven't found it, belongs to the previous inteval */
                           idx--;
                        }

                        tree_idx_hash[domi] = idx;
                     }
                  }
               }
            }


            /* search all nodes */
            for(i = 0 ; i < n_local ; i ++)
            {
               if(vtxsep[i] == 1)
               {
                  /* this is in the edge separator, and haven't been marked */
                  dom = map[i];

                  auto find_idx = tree_idx_hash.find(dom);
                  idx = find_idx->second;

                  marker = false;

                  if( idx % 2 == 0)
                  {
                     even = true;
                  }
                  else
                  {
                     even = false;
                  }

                  /* even value, this is the LOWER half of a pair */
                  j1 = xadj[i];
                  j2 = xadj[i+1];

                  for(j = j1 ; j < j2 ; j ++)
                  {
                     col = adjncy[j];
                     id = ids[j]; /* this is the id this col belongs to */

                     if(id != myid)
                     {
                        /* only check off-diagonal entries */
                        auto find_col = col_map_uncertain_hash.find(col);
                        domi = send2_v2[id][find_col->second];
                     }
                     else
                     {
                        domi = map[col-n_start];
                     }

                     auto find_idx = tree_idx_hash.find(domi);
                     idxi = find_idx->second;

                     if( (even && idx +1 == idxi) || (!even && idx -1 == idxi) )
                     {
                        /* target col, this node is in the separator */
                        marker = true;
                        break;
                     }
                  }

                  if(marker)
                  {
                     if(even)
                     {
                        vtxsep[i] = 3;
                        n1++;
                     }
                     else
                     {
                        vtxsep[i] = 4;
                        n2++;
                     }
                  }
               }
            }

            /* now check which to add to separator */
            PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &n1, &n1_g, 1, MPI_SUM,comm));
            PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &n2, &n2_g, 1, MPI_SUM,comm));

            /* main loop done */
            if(n1_g <= n2_g)
            {
               /* in this case putting those marked 3 into 2 */
               for(i = 0 ; i < n_local ; i ++)
               {
                  if(vtxsep[i] == 3)
                  {
                     vtxsep[i] = 2;
                  }
                  else if(vtxsep[i] == 4)
                  {
                     vtxsep[i] = 1;
                  }
               }
            }
            else
            {
               /* in this case putting those marked 3 into 2 */
               for(i = 0 ; i < n_local ; i ++)
               {
                  if(vtxsep[i] == 4)
                  {
                     vtxsep[i] = 2;
                  }
                  else if(vtxsep[i] == 3)
                  {
                     vtxsep[i] = 1;
                  }
               }
            }

            tree_idx_hash.clear();
            treei.Clear();

            ts = te;
         }

         /* adjust marker value */
         for(i = 0 ; i < n_local ; i ++)
         {
            if(vtxsep[i] == 1)
            {
               vtxsep[i] = 0;
            }
            else if(vtxsep[i] == 2)
            {
               vtxsep[i] = 1;
            }
         }
      }

      /* -------------------------
       * Step 4: check the separator
       * If some subdomain has no
       * interior nodes, we would
       * have to reject this
       * -------------------------
       */

      /* check if some color has no interior nodes */

      marker2.Setup(num_dom);
      marker2.Fill(-1);

      /* mark local domains */
      for (i = 0; i < n_local; i++)
      {
         /* if found interior, mark to 1 */
         if(vtxsep[i] == 0)
         {
            marker2[map[i]] = 1;
         }
      }

      /* check for empty domain */
      PARGEMSLR_MPI_CALL( PargemslrMpiAllreduceInplace( marker2.GetData(), num_dom, MPI_MAX, comm) );

      for (i = 0; i < num_dom; i++)
      {
         if(marker2[i] == -1)
         {
            return -1;
         }
      }

      /* -------------------------
       * Step 5: now form the
       * reduced system
       * first get the local vertices
       * -------------------------
       */

      n_local_s = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] > 0)
         {
            vtxsep[i] = n_local_s;
            n_local_s++;
         }
         else
         {
            vtxsep[i] = -1;
         }
      }

      local_diff = n_local - n_local_s;
      PARGEMSLR_MPI_CALL( PargemslrMpiAllreduce( &local_diff, &global_diff, 1, MPI_MIN, comm) );

      if(global_diff == 0)
      {
         /* we have no exterior nodes, stop here */
         return -1;
      }

      /* global displacement */
      n_local_ss.Setup(np);

      PARGEMSLR_MPI_CALL( MPI_Allgather( &n_local_s, 1, MPI_LONG, n_local_ss.GetData(), 1, MPI_LONG, comm) );

      vtxdist_s.Setup(np+1);
      vtxdist_s[0] = 0;
      for(i = 0 ; i < np ; i ++)
      {
         vtxdist_s[i+1] = vtxdist_s[i] + n_local_ss[i];
      }

      /* --------------------------------
       * Step 6: get vtxsep info of offds
       * --------------------------------
       */

      col_map_uncertain_hash.clear();

      sendsize.Fill(0);
      recvsize.Fill(0);
      for(i = 0 ; i < np ; i ++)
      {
         send_v2[i].Clear();
         send_v2[i].Clear();
         send2_v2[i].Clear();
         recv2_v2[i].Clear();
      }

      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] >= 0)
         {
            /* this is a vtxsep */
            j1 = xadj[i];
            j2 = xadj[i+1];

            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */

               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col == col_map_uncertain_hash.end())
                  {
                     /* breaking news! a NEW column! */
                     send_v2[id].PushBack(col);
                     col_map_uncertain_hash[col] = sendsize[id];
                     sendsize[id]++;
                  }
               }
            }
         }
      }

      /* communicate send and recv size */
      PARGEMSLR_MPI_CALL( MPI_Alltoall( sendsize.GetData(), 1, MPI_INT, recvsize.GetData(), 1, MPI_INT, comm) );

      /* then apply communication */

      /* first send cols */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            /* myid have data for processor i */
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv_v2[i].Setup(recvsize[i]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      PARGEMSLR_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );

      /* get the map value */
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv2_v2[i].Setup(recvsize[i]);
            for(j = 0 ; j < recvsize[i]; j ++)
            {
               recv2_v2[i][j] = vtxsep[recv_v2[i][j] - n_start];
            }
         }
      }

      /* then apply communication again */

      /* or MPI_Alltoallv? */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            /* myid have data for processor i */
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( recv2_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            send2_v2[i].Setup(sendsize[i]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( send2_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      PARGEMSLR_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );

      /* now adding elements */

      xadj_s.Setup(n_local_s+1);
      adjncy_s.Resize(0, false, false);

      xadj_s[0] = 0;

      ii = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] >= 0)
         {
            xadj_s[ii+1] = xadj_s[ii];
            j1 = xadj[i];
            j2 = xadj[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j];

               if(id != myid)
               {
                  /* off-diagonal blocks */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col != col_map_uncertain_hash.end())
                  {
                     idx = send2_v2[id][find_col->second];
                     if(idx >= 0)
                     {
                        adjncy_s.PushBack(idx+vtxdist_s[id]);
                        xadj_s[ii+1]++;
                     }
                  }
               }
               else
               {
                  /* diagonal block */
                  idx = vtxsep[col-n_start];
                  if(idx >= 0)
                  {
                     adjncy_s.PushBack(idx+vtxdist_s[myid]);
                     xadj_s[ii+1]++;
                  }
               }
            }
            ii++;
         }
      }

      /* reset vtxsep */
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] >= 0)
         {
            vtxsep[i] = 1;
         }
         else
         {
            vtxsep[i] = -1;
         }
      }

      /* deallocate */
      ids.Clear();
      marker2.Clear();
      n_local_ss.Clear();
      sendsize.Clear();
      recvsize.Clear();

      for(i = 0 ; i < np ; i ++)
      {
         send_v2[i].Clear();
         recv_v2[i].Clear();
         send2_v2[i].Clear();
         recv2_v2[i].Clear();
      }

      std::vector<vector_long>().swap(send_v2);
      std::vector<vector_long>().swap(recv_v2);
      std::vector<vector_int>().swap(send2_v2);
      std::vector<vector_int>().swap(recv2_v2);

      col_map_uncertain_hash.clear();
      vector<MPI_Request>().swap(requests);

      return PARGEMSLR_SUCCESS;
   }

   template <typename T>
   int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<T> &A, int clvl, int &tlvl, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog)
   {
      long int          i, j, n_local, n_start, nA, col, ncomp2;
      int               err = 0;
      vector_int        map;
      vector_long       perm_sep;
      vector_int        map2_v;
      vector_int        vtxsep;
      ParallelCsrMatrixClass<T> S;

      MPI_Comm          comm;
      int               myid, np;
      parlog.GetMpiInfo(np, myid, comm);

      nA = A.GetNumRowsGlobal();
      n_local = A.GetNumRowsLocal();
      n_start = A.GetRowStartGlobal();

      /* main loop */
      if (minsep < nA && ncomp <= nA && clvl < tlvl)
      {

         /* build a k-way partition with vertex separator
          * perm_sep is the global partition perm_sep = p such that A(p, p) is the global separator
          * map2_v[i] = j if i-th node is in the j-th domain. Domain start from 0, domain k is the edge separator
          */
         ncomp2 = ncomp;

         if( A.GetSeparatorNumSubdomains() == ncomp2)
         {
            /* the partition is provided */
            map = A.GetSeparatorDomi();
            for(i = 0 ; i < n_local ; i ++)
            {
               if(map[i] < 0)
               {
                  map[i] = ncomp;
                  perm_sep.PushBack((long int)(n_start + i));
               }
            }
         }
         else
         {
            err = SetupPermutationParallelKwayVertexSep( A, ncomp2, map, perm_sep, parlog); PARGEMSLR_CHKERR(err);
         }

         if(ncomp2 < ncomp)
         {
            /* in this case, partition failes, stop here */
            if(bj_last && nA >= np)
            {
               /* treat last level with block-Jacobi like partition
                * we assign np subdomains to the last level
                */

               long int       bjdom1, bjdom2, k;
               vector_long    bjdisps, bjns;

               bjdom1 = nA / np;
               bjdom2 = nA % np;

               bjdisps.Setup(np+1);
               bjns.Setup(np);

               for(i = 0 ; i < np ; i ++)
               {
                  if(i < bjdom2)
                  {
                     bjns[i] = bjdom1 + 1;
                  }
                  else
                  {
                     bjns[i] = bjdom1;
                  }
               }

               bjdisps[0] = 0;
               for(i = 0 ; i < np ; i ++)
               {
                  bjdisps[i+1] = bjdisps[i] + bjns[i];
               }

               col = mapptr_v.Back();
               mapptr_v.PushBack(col+np);
               map_v.Setup(n_local);

               j = 0;
               for(i = 0, k = n_start; i < n_local ; i ++, k++)
               {
                  while(bjdisps[j+1] <= k)
                  {
                     /* in this case, current node blongs to the next one
                      * we want k in [bjdisps[j], bjdisps[j+1])
                      * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                      */
                     j++;
                  }
                  map_v[i] = j+col;
               }
            }
            else
            {
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+1);
               map_v.Setup(n_local);
               map_v.Fill(col);
            }

            tlvl = clvl;
         }
         else
         {
            /* prepare the recursive call */
            A.SubMatrix( perm_sep, perm_sep, kMemoryHost, S);

            mapptr_v.PushBack(mapptr_v.Back()+ncomp2);

            if(ncomp > kmin)
            {
               ncomp = ncomp / kfactor;
               if(ncomp < kmin)
               {
                  ncomp = kmin;
               }
            }

            // (Increase clvl by 1)
            SetupPermutationParallelRKwayRecursive2( S, clvl+1, tlvl, ncomp, minsep, kmin, kfactor, map2_v, mapptr_v, bj_last, parlog);

            S.Clear();

            /* udpate map information */
            map_v.Setup(n_local);
            j = 0;
            for(i = 0 ; i < n_local ; i ++)
            {
               if(map[i] < ncomp2)
               {
                  /* interior nodes */
                  map_v[i] = map[i] + mapptr_v[clvl-1];
               }
               else
               {
                  /* exterior nodes */
                  map_v[i] = map2_v[j++];
               }
            }

         }
      }
      else
      {
         /* in this case, we don't have enough domains, stop here
          * a special case is when nA == 0
          */
         if(nA > 0)
         {
            if(bj_last && nA >= np)
            {
               /* treat last level with block-Jacobi like partition
                * we assign np subdomains to the last level
                */

               long int       bjdom1, bjdom2, k;
               vector_long    bjdisps, bjns;

               bjdom1 = nA / np;
               bjdom2 = nA % np;

               bjdisps.Setup(np+1);
               bjns.Setup(np);

               for(i = 0 ; i < np ; i ++)
               {
                  if(i < bjdom2)
                  {
                     bjns[i] = bjdom1 + 1;
                  }
                  else
                  {
                     bjns[i] = bjdom1;
                  }
               }

               bjdisps[0] = 0;
               for(i = 0 ; i < np ; i ++)
               {
                  bjdisps[i+1] = bjdisps[i] + bjns[i];
               }

               col = mapptr_v.Back();
               mapptr_v.PushBack(col+np);
               map_v.Setup(n_local);

               j = 0;
               for(i = 0, k = n_start; i < n_local ; i ++, k++)
               {
                  while(bjdisps[j+1] <= k)
                  {
                     /* in this case, current node blongs to the next one
                      * we want k in [bjdisps[j], bjdisps[j+1])
                      * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                      */
                     j++;
                  }
                  map_v[i] = j+col;
               }
            }
            else
            {
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+1);
               map_v.Setup(n_local);
               map_v.Fill(col);
            }

            tlvl = clvl;
         }
         else
         {
            tlvl = clvl-1;
         }
      }

      return err;
   }
   template int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<float> &A, int clvl, int &tlvl, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);
   template int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<double> &A, int clvl, int &tlvl, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);
   template int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<complexs> &A, int clvl, int &tlvl, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);
   template int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<complexd> &A, int clvl, int &tlvl, long int ncomp, long int minsep, long int kmin, long int kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);

   template <typename T>
   int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<T> &A, long int &ncomp, vector_int &map_v, vector_long &perm_sep, parallel_log &parlog)
   {
      int         n_local, i, nd_tlvl, nd_clvl, ncomp_temp;
      long int    n_start;
      bool        succeed;

      /* now find 2^k = num_dom */
      ncomp_temp = ncomp;

      PARGEMSLR_CHKERR( ( ( ncomp_temp ) & ( ncomp_temp - 1 ) ) != 0 );

      nd_tlvl = 0;
      while (ncomp_temp > 0)
      {
         ncomp_temp = ncomp_temp >> 1;
         nd_tlvl++;
      }

      /* start the ND ordering */
      nd_clvl = 0;
      SetupPermutationParallelKwayVertexSepRecursive(A, nd_clvl, nd_tlvl, succeed, map_v, parlog);

      if(!succeed)
      {
         /* partition fails */
         ncomp = 0;
      }
      else
      {
         perm_sep.Resize(0, false, false);
         n_local = A.GetNumRowsLocal();
         n_start = A.GetRowStartGlobal();
         for(i = 0 ; i < n_local ; i ++)
         {
            PARGEMSLR_CHKERR( map_v[i] > ncomp);
            if(map_v[i] == ncomp)
            {
               /* this is an edge cut */
               perm_sep.PushBack((long int)(n_start + i));
            }
         }
      }

      return PARGEMSLR_SUCCESS;
   }
   template int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<float> &A, long int &ncomp, vector_int &map_v, vector_long &perm_sep, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<double> &A, long int &ncomp, vector_int &map_v, vector_long &perm_sep, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<complexs> &A, long int &ncomp, vector_int &map_v, vector_long &perm_sep, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<complexd> &A, long int &ncomp, vector_int &map_v, vector_long &perm_sep, parallel_log &parlog);

   template <typename T>
   int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<T> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog)
   {
      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Recursive kway partition only works for host matrices.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      /* at least should have two levels */
      PARGEMSLR_CHKERR(tlvl < 2);

      if(clvl == 0)
      {
         /* on the root level, set succeed to true */
         succeed = true;
      }

      int                              i, j, j1, j2, err = 0, idx;
      int                              n_local, nnz_local, ncomp, lev_shift;
      long int                         n_start, two, ndom1, ndom2, edge_cut;
      vector_long                      vtxdist, xadj, adjncy, dom1, dom2, map;
      vector_int                       vtxsep, map1_v, map2_v, idx_v;
      ParallelCsrMatrixClass<T>        A1, A2;

      MPI_Comm                         comm;
      int                              myid, np;

      A.GetMpiInfo(np, myid, comm);

      CsrMatrixClass<T> &A_diag        = A.GetDiagMat();
      CsrMatrixClass<T> &A_offd        = A.GetOffdMat();
      vector_long &offd_map_v          = A.GetOffdMap();

      n_start                          = A.GetRowStartGlobal();
      n_local                          = A.GetNumRowsLocal();
      nnz_local                        = A_diag.GetNumNonzeros() + A_offd.GetNumNonzeros();

      /* set vtxdist */
      vtxdist.Setup(np+1);
      vtxdist[np] = A.GetNumRowsGlobal();

      PARGEMSLR_MPI_CALL( MPI_Allgather(&n_start, 1, MPI_LONG, vtxdist.GetData(), 1, MPI_LONG, comm) );

      /* xadj and adjncy */
      xadj.Setup(n_local+1);
      adjncy.Setup(nnz_local-n_local);

      /* Costruct a CSR-like representation of A as required by METIS. Extract and keep the diagonal entries separately */

      int *A_diag_i = A_diag.GetI();
      int *A_diag_j = A_diag.GetJ();
      int *A_offd_i = A_offd.GetI();
      int *A_offd_j = A_offd.GetJ();
      long int *offd_map = offd_map_v.GetData();

      xadj[0] = 0;
      for (i = 0; i < n_local; i++)
      {
         xadj[i+1] = xadj[i];
         j1 = A_diag_i[i];
         j2 = A_diag_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            if(A_diag_j[j] != i)
            {
               adjncy[xadj[i+1]] = A_diag_j[j] + n_start;
               xadj[i+1]++;
            }
         }

         j1 = A_offd_i[i];
         j2 = A_offd_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            adjncy[xadj[i+1]] = offd_map[A_offd_j[j]];
            xadj[i+1]++;
         }
      }

      /* now, call a two-way partition of the current matrix */
      two = 2;

      err = ParmetisKwayHost( vtxdist, xadj, adjncy, two, map, parlog);

      if(two < 2 || err)
      {
         /* in this case, we can't have two subdomains, stop here
          * reset error message, this is not a critical error
          */
         succeed = false;
         err = PARGEMSLR_SUCCESS;
      }
      else
      {
         /* in this case, we build the separator */
         ParallelNDGetSeparator(vtxdist, xadj, adjncy, map, ndom1, ndom2, edge_cut, parlog);

         if(ndom1 == 0 || ndom2 == 0)
         {
            succeed = false;
         }
         else
         {
            if(clvl == tlvl-2)
            {
               /* no more levels, mark vtxsep and done */
               ncomp = (int)pow(2.0, (double)(tlvl-1));

               map_v.Setup(n_local);
               for(i = 0 ; i < n_local ; i ++)
               {
                  switch(map[i])
                  {
                     case 0:
                     {
                        /* the first subdomain */
                        map_v[i] = 0;
                        break;
                     }
                     case 1:
                     {
                        /* the second subdomain */
                        map_v[i] = 1;
                        break;
                     }
                     default:
                     {
                        /* the separator */
                        map_v[i] = ncomp;
                        break;
                     }
                  }
               }
            }
            else
            {

               /* build the dom1, dom2, and idx_v */

               idx_v.Setup(n_local);
               dom1.Setup(0, n_local/2, kMemoryHost, false);
               dom2.Setup(0, n_local/2, kMemoryHost, false);
               j1 = 0;
               j2 = 0;
               for(i = 0 ; i < n_local ; i ++)
               {
                  switch(map[i])
                  {
                     case 0:
                     {
                        /* this node belongs to the first subdomain */
                        dom1.PushBack(i+n_start);
                        idx_v[i] = j1++;
                        break;
                     }
                     case 1:
                     {
                        /* this node belongs to the second subdomain */
                        dom2.PushBack(i+n_start);
                        idx_v[i] = j2++;
                        break;
                     }
                     default:
                     {
                        /* the separator, do nothing */
                        break;
                     }
                  }
               }

               /* get the first and the second  */
               A.SubMatrix(dom1, dom1, kMemoryHost, A1);
               A.SubMatrix(dom2, dom2, kMemoryHost, A2);

               SetupPermutationParallelKwayVertexSepRecursive(A1, clvl+1, tlvl, succeed, map1_v, parlog);

               if(succeed)
               {
                  SetupPermutationParallelKwayVertexSepRecursive(A2, clvl+1, tlvl, succeed, map2_v, parlog);
               }

               if(succeed)
               {
                  /* the above two recursive partition succeed, proceed to update the local map
                   *
                   *       - - -
                   *      /     \
                   *     -       -
                   *    / \     / \
                   *   -   -   -   -
                   *  / \ / \ / \ / \
                   *  1 2 3 4 5 6 7 8
                   *
                   * the returned mapping infomation
                   *
                   */

                  /* nlev has 2^{nlev-1} comps */
                  ncomp = (int)pow(2.0, (double)(tlvl-1));

                  /* apply the shift
                   * for example, on the third lst level when tlvl - clvl = 2
                   * 0 1 0 1 -> 0 1 2 3, the shift is 2.
                   */
                  lev_shift = (int)pow(2.0, (double)(tlvl-clvl-2));

                  map_v.Setup(n_local);
                  for(i = 0 ; i < n_local ; i ++)
                  {
                     switch(map[i])
                     {
                        case 0:
                        {
                           /* the first subdomain */
                           idx = map1_v[idx_v[i]];
                           if(idx == ncomp)
                           {
                              /* in the saperator */
                              map_v[i] =ncomp;
                           }
                           else
                           {
                              /* keep the map */
                              map_v[i] = idx;
                           }
                           break;
                        }
                        case 1:
                        {
                           /* the second subdomain */
                           idx = map2_v[idx_v[i]];
                           if(idx == ncomp)
                           {
                              /* in the saperator */
                              map_v[i] = ncomp;
                           }
                           else
                           {
                              /* keep the map plus a shift */
                              map_v[i] = idx + lev_shift;
                           }
                           break;
                        }
                        default:
                        {
                           /* the separator */
                           map_v[i] = ncomp;
                           break;
                        }
                     }
                  }

               }
            }
         }
      }/* end of else for ncomp < 2 */

      /* deallocate */
      vtxdist.Clear();
      xadj.Clear();
      adjncy.Clear();
      dom1.Clear();
      dom2.Clear();
      map.Clear();
      map1_v.Clear();
      map2_v.Clear();
      idx_v.Clear();
      A1.Clear();
      A2.Clear();

      return err;
   }
   template int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<float> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<double> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<complexs> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<complexd> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog);

   int ParallelNDGetSeparator( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy, vector_long &map, long int &ndom1, long int &ndom2, long int &edge_cut, parallel_log &parlog)
   {
      long int                   i, j, j1, j2;
      long int                   n_local, n_start, n_end, dom, col, nnz;
      int                        id;
      vector_int                 ids, vtxsep;

      std::unordered_map<long int, int> col_map_hash;
      int                        ncols;
      vector_long                cols;
      vector_int                 col_ids;

      std::unordered_map<long int, int> col_map_uncertain_hash;
      vector_int                 sendsize, recvsize;
      std::vector<vector_long>   send_v2, recv_v2;
      std::vector<vector_int>    send2_v2, recv2_v2;

      MPI_Comm                   comm;
      int                        myid, np, numwaits;
      vector<MPI_Request>        requests;

      parlog.GetMpiInfo(np, myid, comm);

      n_start = vtxdist[myid];
      n_end = vtxdist[myid+1];
      n_local = n_end - n_start;

      if(n_local == 0)
      {
         /* empty, no need to setup */
         nnz = 0;
         vtxsep.Clear();
      }
      else
      {
         nnz = xadj[n_local];
         vtxsep.Setup(n_local, true);
         ids.Setup(nnz, true);
      }

      /* -------------------------
       * Step 1: put all local columns into the hash table
       * -------------------------
       */
      ncols = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index
             * note that there might be duplicate entries
             */
            auto find_col =  col_map_hash.find(col);
            if(find_col == col_map_hash.end())
            {
               /* a new one */
               if(vtxdist.BinarySearch( col, id, true) < 0)
               {
                  /* in this case, col fall in between */
                  id--;
               }

               cols.PushBack(col);
               col_ids.PushBack(id);
               col_map_hash[col] = ncols;
               ncols++;
            }
         }
      }

      /* ids[j] is the MPI process adjncy[j] belongs to */
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index
             * note that there might be duplicate entries
             */
            auto find_col =  col_map_hash.find(col);
            ids[j] = col_ids[find_col->second];
         }
      }

      /* -------------------------
       * Step 2: check local first
       * some rows/cols requires
       * accessing offdiagonal entries
       * -------------------------
       */

      /* This loop find local vtxsep
       * 1. local cols already has multiple maps:
       *    => set vtxsep to 1, separator.
       * 2. all cols are local, and all same map values:
       *    => set vtxsep to 0, interior.
       * 3. all local cols same map, however, have exterior cols
       *    => set vtxsep to -1, TBD
       */
      for(i = 0 ; i < n_local ; i ++)
      {
         /* dom is the domain of the row */
         dom = map[i];
         j1 = xadj[i];
         j2 = xadj[i+1];

         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            id = ids[j]; /* this is the id this col belongs to */

            /* we can only check the map value of local parts */
            if(id == myid)
            {
               col -= n_start;
               if(map[col] != dom)
               {
                  /* this is diagonal, exterior node */
                  vtxsep[i] = 1;
                  break;
               }
            }
            else
            {
               /* if has offd, mark to -1 instead */
               vtxsep[i] = -1;
            }
         }
      }

      /* -------------------------
       * Step 3: check remaining
       * -------------------------
       */

      sendsize.Setup(np, true);
      recvsize.Setup(np, true);
      send_v2.resize(np);
      recv_v2.resize(np);
      send2_v2.resize(np);
      recv2_v2.resize(np);

      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] == -1)
         {
            /* this is a target column */
            j1 = xadj[i];
            j2 = xadj[i+1];

            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */

               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col == col_map_uncertain_hash.end())
                  {
                     /* breaking news! a NEW column! */
                     send_v2[id].PushBack(col);
                     col_map_uncertain_hash[col] = sendsize[id];
                     sendsize[id]++;
                  }
               }
            }
         }
      }

      /* communicate send and recv size */
      PARGEMSLR_MPI_CALL( MPI_Alltoall( sendsize.GetData(), 1, MPI_INT, recvsize.GetData(), 1, MPI_INT, comm) );

      /* then apply communication */

      requests.resize(2*np);

      /* first send cols */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            /* myid have data for processor i */
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv_v2[i].Setup(recvsize[i]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      PARGEMSLR_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );

      /* get the map value */
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv2_v2[i].Setup(recvsize[i]);
            for(j = 0 ; j < recvsize[i]; j ++)
            {
               recv2_v2[i][j] = map[recv_v2[i][j] - n_start];
            }
         }
      }

      /* then apply communication again */

      /* or MPI_Alltoallv? */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            /* myid have data for processor i */
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( recv2_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            send2_v2[i].Setup(sendsize[i]);
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( send2_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }

      PARGEMSLR_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );

      /* finally check the received columns */

      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] == -1)
         {
            /* this is a target column */
            dom = map[i];
            j1 = xadj[i];
            j2 = xadj[i+1];

            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */

               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(send2_v2[id][find_col->second] != dom)
                  {
                     vtxsep[i] = 1;
                     break;
                  }
               }
            }

            /* if still -1, this is interior */
            if(vtxsep[i] == -1)
            {
               vtxsep[i] = 0;
            }
         }
      }

      /* now form the reduced system */
      long int ndom1_local = 0, ndom2_local = 0, ec1_local = 0, ec2_local = 0, ec1, ec2;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] > 0)
         {
            /* this is in the seperator */
            if(map[i] == 0)
            {
               ec1_local++;
            }
            else
            {
               ec2_local++;
               map[i] = 1;
            }
         }
         else
         {
            if(map[i] == 0)
            {
               ndom1_local++;
            }
            else
            {
               ndom2_local++;
               map[i] = 1;
            }
         }
      }

      PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &ndom1_local, &ndom1, 1, MPI_SUM,comm));
      PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &ndom2_local, &ndom2, 1, MPI_SUM,comm));
      PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &ec1_local, &ec1, 1, MPI_SUM,comm));
      PARGEMSLR_MPI_CALL(PargemslrMpiAllreduce( &ec2_local, &ec2, 1, MPI_SUM,comm));

      if(ec2 > ec1)
      {
         edge_cut = ec2;
         for(i = 0 ; i < n_local ; i ++)
         {
            if(vtxsep[i] > 0)
            {
               /* this is in the seperator */
               if(map[i] == 1)
               {
                  map[i] = 2;
               }
            }
         }
      }
      else
      {
         edge_cut = ec1;
         for(i = 0 ; i < n_local ; i ++)
         {
            if(vtxsep[i] > 0)
            {
               /* this is in the seperator */
               if(map[i] == 0)
               {
                  map[i] = 2;
               }
            }
         }
      }

      /* deallocate */
      ids.Clear();
      sendsize.Clear();
      recvsize.Clear();

      for(i = 0 ; i < np ; i ++)
      {
         send_v2[i].Clear();
         recv_v2[i].Clear();
         send2_v2[i].Clear();
         recv2_v2[i].Clear();
      }

      std::vector<vector_long>().swap(send_v2);
      std::vector<vector_long>().swap(recv_v2);
      std::vector<vector_int>().swap(send2_v2);
      std::vector<vector_int>().swap(recv2_v2);

      col_map_uncertain_hash.clear();
      vector<MPI_Request>().swap(requests);

      return PARGEMSLR_SUCCESS;
   }

   template <typename T>
   int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<T> &A, bool vertexsep, int &nlev, long int minsep, vector_int &map_v, vector_int &mapptr_v)
   {
      if(A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("ND partition only works for host matrices.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      int                              i, j, k, j1, j2, err = 0;
      int                              n_local, nnz_local, clvl, tlvl, domi, size1, size2, ndom_temp;
      long int                         n_start, ndom;
      vector_long                      vtxdist, xadj, adjncy, map_long;

      MPI_Comm                         comm;
      int                              myid, np;

      A.GetMpiInfo(np, myid, comm);

      CsrMatrixClass<T> &A_diag        = A.GetDiagMat();
      CsrMatrixClass<T> &A_offd        = A.GetOffdMat();
      vector_long &offd_map_v          = A.GetOffdMap();

      n_start                          = A.GetRowStartGlobal();
      n_local                          = A.GetNumRowsLocal();
      nnz_local                        = A_diag.GetNumNonzeros() + A_offd.GetNumNonzeros();

      if(np > 1)
      {

         /* set vtxdist */
         vtxdist.Setup(np+1);
         vtxdist[np] = A.GetNumRowsGlobal();

         PARGEMSLR_MPI_CALL( MPI_Allgather(&n_start, 1, MPI_LONG, vtxdist.GetData(), 1, MPI_LONG, comm) );

         /* xadj and adjncy */
         xadj.Setup(n_local+1);
         adjncy.Setup(nnz_local-n_local);

         /* Costruct a CSR-like representation of A as required by METIS. Extract and keep the diagonal entries separately */

         int *A_diag_i = A_diag.GetI();
         int *A_diag_j = A_diag.GetJ();
         int *A_offd_i = A_offd.GetI();
         int *A_offd_j = A_offd.GetJ();
         long int *offd_map = offd_map_v.GetData();

         xadj[0] = 0;
         for (i = 0; i < n_local; i++)
         {
            xadj[i+1] = xadj[i];
            j1 = A_diag_i[i];
            j2 = A_diag_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               if(A_diag_j[j] != i)
               {
                  adjncy[xadj[i+1]] = A_diag_j[j] + n_start;
                  xadj[i+1]++;
               }
            }

            j1 = A_offd_i[i];
            j2 = A_offd_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               adjncy[xadj[i+1]] = offd_map[A_offd_j[j]];
               xadj[i+1]++;
            }
         }

         /* call metis ND */
         ParmetisNodeND(vtxdist, xadj, adjncy, ndom, map_long, A);

         nlev = 0;
         ndom_temp = ndom + 1;
         while (ndom_temp > 1)
         {
            ndom_temp = ndom_temp >> 1;
            nlev++;
         }

         map_v.Setup(n_local);
         mapptr_v.Setup(nlev+1, true);

         for(i = 0 ; i < n_local ; i ++)
         {
            map_v[i] = map_long[i];
         }

         mapptr_v[0] = 0;
         j = (ndom + 1)/2;
         for(i = 1 ; i < nlev ; i ++)
         {
            mapptr_v[i] = mapptr_v[i-1] + j;
            j/=2;
         }
         mapptr_v[nlev] = ndom;

      }
      else
      {
         std::vector<std::vector<vector_int> > level_str;

         /* in this case, only a single processor, apply the sequential one */
         clvl = 0;
         tlvl = nlev;
         err = SetupPermutationNDRecursive( A_diag, vertexsep, clvl, tlvl, minsep, level_str); PARGEMSLR_CHKERR(err);

         /* 4. prepare return value */
         map_v.Setup(n_local);
         mapptr_v.Setup(tlvl+1);

         /* update the final level */
         nlev = tlvl;

         domi = 0;
         mapptr_v[0] = 0;
         for(i = 0 ; i < nlev ; i ++)
         {
            size1 = level_str[i].size();
            for(j = 0 ; j < size1 ; j ++)
            {
               size2 = level_str[i][j].GetLengthLocal();
               for(k = 0 ; k < size2 ; k ++)
               {
                  map_v[level_str[i][j][k]] = domi;
               }
               domi++;
            }
            mapptr_v[i+1] = domi;
         }

         /* free */
         for(i = 0 ; i < nlev ; i ++)
         {
            size1 = level_str[i].size();
            for(j = 0 ; j < size1 ; j ++)
            {
               level_str[i][j].Clear();
            }
            std::vector<vector_int>().swap(level_str[i]);
         }
         std::vector<std::vector<vector_int> >().swap(level_str);

      }

      return err;
   }
   template int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<float> &A, bool vertexsep, int &nlev, long int minsep, vector_int &map_v, vector_int &mapptr_v);
   template int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<double> &A, bool vertexsep, int &nlev, long int minsep, vector_int &map_v, vector_int &mapptr_v);
   template int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<complexs> &A, bool vertexsep, int &nlev, long int minsep, vector_int &map_v, vector_int &mapptr_v);
   template int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<complexd> &A, bool vertexsep, int &nlev, long int minsep, vector_int &map_v, vector_int &mapptr_v);

   template <typename T>
   int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<T> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<T> &B_mat, CsrMatrixClass<T> &E_mat, CsrMatrixClass<T> &F_mat, ParallelCsrMatrixClass<T> &C_mat, int perm_option, bool perm_c)
   {

      int i, j, n_local, nE, nsends, nsendsi, s, e;
      long int n_start;
      vector_int marker;
      vector_long exterior_row;

      MPI_Comm comm;
      int np, myid;

      parcsr_in.GetMpiInfo(np, myid, comm);

      parcsr_in.SetupMatvecStart();
      n_local = parcsr_in.GetNumRowsLocal();
      n_start = parcsr_in.GetRowStartGlobal();

      ParallelCsrMatrixClass<T> &A = parcsr_in;
      CsrMatrixClass<T> &A_diag = A.GetDiagMat();
      //CsrMatrixClass<DataType> &A_offd = A.GetOffdMat();

      int *A_offd_i = parcsr_in.GetOffdMat().GetI();

      /* start to update marker while waiting for the result */
      marker.Setup(n_local);
      marker.Fill(-1);

      /* mark local  */
      for(i = 0 ; i < n_local ; i ++)
      {
         /* check if this is an external node */
         if(A_offd_i[i] < A_offd_i[i+1])
         {
            marker[i] = 0;
         }
      }

      parcsr_in.SetupMatvecOver();

      /* now check col */
      nsends = (int) parcsr_in._comm_helper._send_idx_v2.size();
      for(i = 0 ; i < nsends ; i++)
      {
         if(parcsr_in._comm_helper._send_to_v[i] != myid)
         {
            nsendsi = parcsr_in._comm_helper._send_idx_v2[i].GetLengthLocal();
            for(j = 0 ; j < nsendsi ; j ++)
            {
               marker[parcsr_in._comm_helper._send_idx_v2[i][j]] = 0;
            }
         }
      }

      /* on host */
      local_perm.Setup(n_local);
      s = 0;
      e = n_local - 1;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(marker[i] < 0)
         {
            /* interior node */
            local_perm[s++] = i;
         }
         else
         {
            /* exterior node */
            local_perm[e--] = i;
         }
      }

      nI = s;
      nE = n_local - s;

      /* apply RCM/AMD */
      switch(perm_option)
      {
         case kIluReorderingNo:
         {
            vector_int temp_perm1, temp_perm2;

            temp_perm1.SetupPtr(local_perm, nI, 0);
            temp_perm2.SetupPtr(local_perm, nE, nI);

            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, B_mat);
            A_diag.SubMatrix(temp_perm2, temp_perm1, kMemoryHost, E_mat);
            A_diag.SubMatrix(temp_perm1, temp_perm2, kMemoryHost, F_mat);

            break;
         }
         case kIluReorderingRcm:
         {
            vector_int temp_perm1, temp_perm2, rcm_order;
            CsrMatrixClass<T> Temp_diag;

            temp_perm1.SetupPtr(local_perm, nI, 0);
            temp_perm2.SetupPtr(local_perm, nE, nI);

            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, Temp_diag);

            CsrMatrixRcmHost( Temp_diag, rcm_order);

            temp_perm1.Perm(rcm_order);
            Temp_diag.Clear();
            rcm_order.Clear();

            if(perm_c)
            {
               A_diag.SubMatrix(temp_perm2, temp_perm2, kMemoryHost, Temp_diag);

               CsrMatrixRcmHost( Temp_diag, rcm_order);

               temp_perm2.Perm(rcm_order);
               Temp_diag.Clear();
               rcm_order.Clear();
            }

            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, B_mat);
            A_diag.SubMatrix(temp_perm2, temp_perm1, kMemoryHost, E_mat);
            A_diag.SubMatrix(temp_perm1, temp_perm2, kMemoryHost, F_mat);

            break;
         }
         case kIluReorderingAmd:
         {
            vector_int temp_perm1, temp_perm2, rcm_order;
            CsrMatrixClass<T> Temp_diag;

            temp_perm1.SetupPtr(local_perm, nI, 0);
            temp_perm2.SetupPtr(local_perm, nE, nI);

            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, Temp_diag);

            CsrMatrixAmdHost( Temp_diag, rcm_order);

            temp_perm1.Perm(rcm_order);
            Temp_diag.Clear();
            rcm_order.Clear();

            if(perm_c)
            {
               A_diag.SubMatrix(temp_perm2, temp_perm2, kMemoryHost, Temp_diag);

               CsrMatrixAmdHost( Temp_diag, rcm_order);

               temp_perm2.Perm(rcm_order);
               Temp_diag.Clear();
               rcm_order.Clear();
            }

            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, B_mat);
            A_diag.SubMatrix(temp_perm2, temp_perm1, kMemoryHost, E_mat);
            A_diag.SubMatrix(temp_perm1, temp_perm2, kMemoryHost, F_mat);

            break;
         }
         case kIluReorderingNd:
         {
            vector_int temp_perm1, temp_perm2, rcm_order;
            CsrMatrixClass<T> Temp_diag;

            temp_perm1.SetupPtr(local_perm, nI, 0);
            temp_perm2.SetupPtr(local_perm, nE, nI);

            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, Temp_diag);

            CsrMatrixNdHost( Temp_diag, rcm_order);

            temp_perm1.Perm(rcm_order);
            Temp_diag.Clear();
            rcm_order.Clear();

            if(perm_c)
            {
               A_diag.SubMatrix(temp_perm2, temp_perm2, kMemoryHost, Temp_diag);

               CsrMatrixNdHost( Temp_diag, rcm_order);

               temp_perm2.Perm(rcm_order);
               Temp_diag.Clear();
               rcm_order.Clear();
            }

            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, B_mat);
            A_diag.SubMatrix(temp_perm2, temp_perm1, kMemoryHost, E_mat);
            A_diag.SubMatrix(temp_perm1, temp_perm2, kMemoryHost, F_mat);

            break;
         }
         default:
         {
            PARGEMSLR_ERROR("Unknown local reordering option.");
            return PARGEMSLR_ERROR_INVALED_OPTION;
         }
      }

      exterior_row.Setup(nE);

      for(i = 0 ; i < nE ; i ++)
      {
         exterior_row[i] = (long int)local_perm[++e] + n_start;
      }

      parcsr_in.SubMatrix( exterior_row, exterior_row, kMemoryHost, C_mat);

      C_mat.SortOffdMap();

      exterior_row.Clear();

      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<float> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<float> &B_mat, CsrMatrixClass<float> &E_mat, CsrMatrixClass<float> &F_mat, ParallelCsrMatrixClass<float> &C_mat, int perm_option, bool perm_c);
   template int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<double> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<double> &B_mat, CsrMatrixClass<double> &E_mat, CsrMatrixClass<double> &F_mat, ParallelCsrMatrixClass<double> &C_mat, int perm_option, bool perm_c);
   template int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<complexs> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<complexs> &B_mat, CsrMatrixClass<complexs> &E_mat, CsrMatrixClass<complexs> &F_mat, ParallelCsrMatrixClass<complexs> &C_mat, int perm_option, bool perm_c);
   template int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<complexd> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<complexd> &B_mat, CsrMatrixClass<complexd> &E_mat, CsrMatrixClass<complexd> &F_mat, ParallelCsrMatrixClass<complexd> &C_mat, int perm_option, bool perm_c);
}
