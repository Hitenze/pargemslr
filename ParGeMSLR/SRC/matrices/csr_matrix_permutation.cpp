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
   int SetupPermutationRKwayRecursive( CsrMatrixClass<T> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v)
   {

      /* immediatly return if we don't have enoguh levels */
      if(tlvl < 2 || A.GetNumRowsLocal() < 2)
      {
         /* only one level */
         mapptr_v[1]=1;
         map_v.Fill(0);
         tlvl = 1;
         return PARGEMSLR_SUCCESS;
      }


      /* should not call this function with num_dom < 2 */
      PARGEMSLR_CHKERR(num_dom < 2);

      /* TODO: disconnected components */
      /* now start calling the recursive KWay main function */
      int               i, j, k, nA, domi, edgecut, num_dom2;
      int               num_dom_temp, nd_clvl, nd_tlvl, nd_minsep;
      std::vector<std::vector<vector_int> > nd_level_str;
      vector_int        map, vtxsep, perm, dom_ptr, perm_c, mapc;
      CsrMatrixClass<T> C;

      if( A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("RKway partition only works for host.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      if( A.GetNumRowsLocal() != A.GetNumColsLocal())
      {
         PARGEMSLR_ERROR("RKway partition only works for square matrix.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      if(clvl == tlvl-1)
      {
         /* treat the last level as a single block */
         domi = mapptr_v[clvl];
         mapptr_v[clvl+1]=domi+1;
         map_v.Fill(domi);
         tlvl = clvl+1;
         return PARGEMSLR_SUCCESS;
      }

      nA = A.GetNumRowsLocal();

      /* stop if we don't have enough number of nodes on the next level */
      if(minsep >= nA || num_dom >= nA )
      {
         /* in this case, we can't apply the partition any more,
          * put the reaminning part into one single level */
         domi = mapptr_v[clvl];
         mapptr_v[clvl+1]=domi+1;
         map_v.Fill(domi);
         tlvl = clvl+1;

         return PARGEMSLR_SUCCESS;
      }

      /* otherwise we have enough levels */
      if(vertexsep)
      {
         /* Use vertex seperator, we use multiple 2-way partition
          * we can take advantage of the ND code
          */

         /* first make sure num_dom is power of 2 */
         PARGEMSLR_CHKERR( ( ( num_dom ) & ( num_dom - 1 ) ) != 0 );

         /* now find 2^k = num_dom */
         num_dom_temp = num_dom;
         nd_tlvl = 0;
         while (num_dom_temp > 0)
         {
            num_dom_temp = num_dom_temp >> 1;
            nd_tlvl++;
         }
         /* now 2^nd_tlvl = num_dom, we apply a k level ND
          * with vertex seperator, keep all other level as edge seperator
          */
         nd_clvl = 0;
         /* in this partition we partition till very small blocks */
         nd_minsep = 2;

         /* call ND ordering */
         SetupPermutationNDRecursive( A, true, nd_clvl, nd_tlvl, nd_minsep, nd_level_str);

         /* combine different domains together */
         SetupPermutationNDCombineLevels(nd_level_str[0], num_dom);

         /* now, nd_tlvl is the total number of levels
          * we only going to keep the level 0, and put all other levels into a big vertex seperator.
          * Note that perm is not used in this function, we are not going to build it.
          */
         map.Setup(nA);
         vtxsep.Setup(nA);

         /* the size on level 0 is the final k */
         num_dom2 = nd_level_str[0].size();
         dom_ptr.Setup(num_dom2+1);

         /* we assign all other levels to the last level */
         map.Fill(num_dom2-1);
         vtxsep.Fill(1);

         /* first setup the interior nodes */
         domi = 0;
         dom_ptr[0] = 0;

         for(i = 0 ; i < num_dom2 ; i ++)
         {
            k = nd_level_str[0][i].GetLengthLocal();
            for(j = 0 ; j < k ; j ++)
            {
               map[nd_level_str[0][i][j]] = domi;
               vtxsep[nd_level_str[0][i][j]] = 0;
            }
            domi++;
            dom_ptr[domi] = dom_ptr[domi-1] + k;
         }

         edgecut = nA - dom_ptr[num_dom2];
         dom_ptr[num_dom2] = nA;

         /* now put the remaining le */

         for(i = 0 ; i < nd_tlvl ; i ++)
         {
            k = nd_level_str[i].size();
            for(j = 0 ; j < k ; j ++)
            {
               nd_level_str[i][j].Clear();
            }
            std::vector<vector_int>().swap(nd_level_str[i]);
         }
         std::vector<std::vector<vector_int> >().swap(nd_level_str);

      }
      else
      {
         /* use the standart edge seperator */
         num_dom2 = num_dom;
         std::vector<vector_int> test;
         int testcomp;
         A.GetConnectedComponents( test, testcomp);

         if(CsrMatrixMetisKwayHost( A, num_dom2, map, false, vtxsep, edgecut, perm, dom_ptr) == PARGEMSLR_RETURN_METIS_NO_INTERIOR )
         {
            /* in this case, at least one subdomain has no interior nodes, we should stop on this level
             * Set edgecut to nA so that we'll go to the terminate phase
             */
            edgecut = nA;
         }
      }

      if(edgecut == 0)
      {
         /* we have no next level, and the partition of this level is perfect */
         domi = mapptr_v[clvl];

         mapptr_v[clvl+1]=domi + num_dom2;

         /* add a shift to the map vector */
         for(i = 0 ; i < nA ; i ++)
         {
            map_v[i] = map[i] + domi;
         }

         tlvl = clvl+1;
         return PARGEMSLR_SUCCESS;
      }

      if( num_dom2 < num_dom || edgecut == nA )
      {
         /* treat the last level as a single block */
         domi = mapptr_v[clvl];
         mapptr_v[clvl+1]=domi+1;
         map_v.Fill(domi);
         tlvl = clvl+1;
         return PARGEMSLR_SUCCESS;
      }

      /* start forming the C matrix */
      perm_c.Setup(edgecut);

      j = 0;
      for(i = 0 ; i < nA ; i ++)
      {
         if(vtxsep[i] > 0)
         {
            perm_c[j++] = i;
         }
      }

      /* extract the C matrix */
      A.SubMatrix( perm_c, perm_c, kMemoryHost, C);
      perm_c.Clear();

      /* update the mapptr */
      mapptr_v[clvl+1] = mapptr_v[clvl] + num_dom2;

      /* recursive partition */
      if(num_dom > kmin)
      {
         num_dom = num_dom / kfactor;
         if(num_dom < kmin)
         {
            num_dom = kmin;
         }
      }

      mapc.Setup(edgecut);
      /* keep using nd */
      //SetupPermutationRKwayRecursive( C, vertexsep, clvl+1, tlvl, num_dom, minsep, kmin, kfactor, mapc, mapptr_v);
      /* only use vertex sep on the top level */
      SetupPermutationRKwayRecursive( C, false, clvl+1, tlvl, num_dom, minsep, kmin, kfactor, mapc, mapptr_v);

      /* go back to this level, update map_v */
      j = 0;
      domi = mapptr_v[clvl];
      for(i = 0 ; i < nA ; i ++)
      {
         if(vtxsep[i] > 0)
         {
            map_v[i] = mapc[j++];
         }
         else
         {
            map_v[i] = map[i] + domi;
         }
      }

      mapc.Clear();

      return PARGEMSLR_SUCCESS;
   }
   template int SetupPermutationRKwayRecursive( CsrMatrixClass<float> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v);
   template int SetupPermutationRKwayRecursive( CsrMatrixClass<double> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v);
   template int SetupPermutationRKwayRecursive( CsrMatrixClass<complexs> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v);
   template int SetupPermutationRKwayRecursive( CsrMatrixClass<complexd> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v);

   /**
    * @brief   Compress a certain level of level_str from SetupPermutationNDRecursive into a give number of domains.
    * @details Compress a certain level of level_str from SetupPermutationNDRecursive into a give number of domains.
    * @param   [in]     level_stri The level of a level_str.
    * @param   [in,out] ndom The target number of domains, on return the ndom we get.
    * @return     Return error message.
    */
   int SetupPermutationNDCombineLevels(std::vector<vector_int> &level_stri, int &ndom)
   {
      int i, n, ndom_in;
      vector_int size, marker;

      ndom_in = level_stri.size();

      if(ndom_in <= ndom)
      {
         /* in this case, do nothing */
         return PARGEMSLR_SUCCESS;
      }

      size.Setup(ndom_in);
      marker.Setup(ndom_in, true);

      n = 0;
      for(i = 0 ; i < ndom_in ; i ++)
      {
         size[i] = level_stri[i].GetLengthLocal();
         n += size[i];
      }

      /* TODO: some dynamic programming algorithms */

      size.Clear();

      return PARGEMSLR_SUCCESS;
   }

   template <typename T>
   int SetupPermutationNDRecursive( CsrMatrixClass<T> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str)
   {
      /* ND ordering
       * start with each connected components of A. For example, A has 2 connected compononts, 0 and 1.
       * The size of 0 might be very small, and can only be partitioned into two parts, 2 and 3, we put it into a higher level.
       * After that, the partition of 1 again leads to a small component 4.
       *
       * Partition:            A
       *                       |
       *              0------------------1
       *                                 |
       *                           2-----3-----4
       *                                 |     |
       *                                5-6   7-8
       * where 0 and 2 are very small connected components that can't be further partitioned. We form the following level structure:
       *
       * level 2:   1
       * level 1:   3, 4
       * level 0:   0, 2, 5, 6, 7, 8
       *
       * Algorith: we use a recursive algorithm, starting from the first level.
       *
       * When raeching the leaf component, we push it to level_str[0] immediatly
       */

      /* now start calling the recursive ND main function */
      int                     i, j, k, idx, nS, ndom, edgecut, ncomps, size, k1, k2, err = 0;
      vector_int              map, vtxsep, perm, dom_ptr, row_perm, col_perm, perm_c, mapc, tlvls;
      std::vector<vector_int> comp_indices;

      CsrMatrixClass<T> B, C;

      if( A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("ND partition only works for host.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      if( A.GetNumRowsLocal() != A.GetNumColsLocal())
      {
         PARGEMSLR_ERROR("ND partition only works for square matrix.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      /* setup level_str] */
      for(i = 0 ; i < (int)level_str.size() ; i ++)
      {
         for(j = 0 ; j < (int)level_str[i].size() ; j ++ )
         {
            level_str[i][j].Clear();
         }
         std::vector<vector_int>().swap(level_str[i]);
      }
      std::vector<std::vector<vector_int> >().swap(level_str);

      if(tlvl < 2 || A.GetNumRowsLocal() < 2)
      {
         /* only one level */
         tlvl = 1;
         level_str.resize(1);
         level_str[0].resize(1);
         level_str[0][0].Setup(A.GetNumRowsLocal());
         level_str[0][0].UnitPerm();
         return PARGEMSLR_SUCCESS;
      }

      if(clvl >= tlvl - 1)
      {
         /* first step, find the connected components */
         ncomps = 0;
         err = A.GetConnectedComponents( comp_indices, ncomps);
         tlvl = 1;
         level_str.resize(1);
         level_str[0].resize(ncomps);
         for(i = 0 ; i < ncomps ; i ++)
         {
            level_str[0][i] = std::move(comp_indices[i]);
         }
         return PARGEMSLR_SUCCESS;
      }

      level_str.resize(tlvl-clvl);

      /* first step, find the connected components */
      ncomps = 0;
      err = A.GetConnectedComponents( comp_indices, ncomps);

      /* apply ND for each component */
      tlvls.Setup(ncomps);
      for(i = 0 ; i < ncomps ; i ++)
      {
         /* First, check the size of the current component
          * if the size is too small, stop here
          */
         nS = comp_indices[i].GetLengthLocal();

         if( nS <= minsep || nS <= 2)
         {
            /* we've reach the end of this components, this is a node of level 0
             * mark the level of this components as clvl + 1
             */
            tlvls[i] = 1;
            level_str[0].push_back(comp_indices[i]);
            continue;
         }
         else
         {
            /* get this component */
            A.SubMatrix( comp_indices[i], comp_indices[i], kMemoryHost, C);

            /* apply 2-way partition */
            ndom = 2;
            if( CsrMatrixMetisKwayHost( C, ndom, map, vertexsep, vtxsep, edgecut, perm, dom_ptr) == PARGEMSLR_RETURN_METIS_NO_INTERIOR )
            {
               /* in this case, we don't have 2 subdomains, or at least one subdomain has no interior nodes
                * we should stop here. Go the the terminate phase by setting edgecut to nS
                */
               edgecut = nS;
            }

            if( ndom < 2 || edgecut == nS)
            {
               tlvls[i] = 1;
               level_str[0].push_back(comp_indices[i]);
               continue;
            }

            /* remove the seperator */
            C.SubMatrixNoPerm(vtxsep, vtxsep, row_perm, col_perm, true, kMemoryHost, B);

            /* go to next level
             * on exit, tlvls[i] is the max level of this component
             */
            tlvls[i] = tlvl;
            std::vector<std::vector<vector_int> > sub_level_str;
            err = SetupPermutationNDRecursive( B, vertexsep, clvl+1, tlvls[i], minsep, sub_level_str); PARGEMSLR_RETURN_ON_ERROR(err);

            /* now back, set indices */
            for( j = 0 ; j < tlvls[i] ; j ++)
            {
               idx = j;
               size = sub_level_str[idx].size();
               for(k = 0 ; k < size ; k ++)
               {
                  /* push this to the same level of level_str */
                  level_str[idx].push_back(sub_level_str[idx][k]);

                  /* update index */
                  vector_int &nodes = level_str[idx].back();
                  k2 = nodes.GetLengthLocal();
                  for(k1 = 0 ; k1 < k2 ; k1 ++)
                  {
                     nodes[k1] = comp_indices[i][row_perm[nodes[k1]]];
                  }
               }
            }
            /* also add the edge seperator */
            idx = tlvls[i];
            level_str[idx].push_back(vector_int());
            vector_int &nodes = level_str[idx].back();
            nodes.Setup(0, edgecut, kMemoryHost, false);
            for(j = 0 ; j < nS ; j ++)
            {
               if(vtxsep[j] != 0)
               {
                  nodes.PushBack(comp_indices[i][j]);
               }
            }
            tlvls[i] += 1;
         }
      }

      tlvl = tlvls.Max();

      return err;
   }
   template int SetupPermutationNDRecursive( CsrMatrixClass<float> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str);
   template int SetupPermutationNDRecursive( CsrMatrixClass<double> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str);
   template int SetupPermutationNDRecursive( CsrMatrixClass<complexs> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str);
   template int SetupPermutationNDRecursive( CsrMatrixClass<complexd> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str);
}
