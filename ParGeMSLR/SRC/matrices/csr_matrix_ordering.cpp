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
   int CsrSubMatrixAmdHost(CsrMatrixClass<T> &A, vector_int &rowscols, vector_int &perm)
   {
      int err;

      CsrMatrixClass<T> B;

      /* get that sub matrix */
      err = A.SubMatrix(rowscols, rowscols, kMemoryHost, B); PARGEMSLR_RETURN_ON_ERROR(err);

      /* apply RCM */
      err = CsrMatrixAmdHost(B, perm);

      return err;

   }
   template int CsrSubMatrixAmdHost(CsrMatrixClass<float> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixAmdHost(CsrMatrixClass<double> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixAmdHost(CsrMatrixClass<complexs> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixAmdHost(CsrMatrixClass<complexd> &A, vector_int &rowscols, vector_int &perm);

   template <typename T>
   int CsrMatrixAmdHost(CsrMatrixClass<T> &A, vector_int &perm)
   {
      /*---------------------------------------------------------------------------
      * AMD ordering of a sparse symmetric matrix A.
      *
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * A         = CSR Matrix object.
      *
      * on return:
      * ==========
      * err       = return value.
      *             err  == 0   --> successful return.
      *             err  != 0   --> Error occurs.
      * perm      = Integer vector. Permutation vector.
      *----------------------------------------------------------------------------
      * Note:
      * C-style 0-based index.
      *
      *--------------------------------------------------------------------------*/
      PARGEMSLR_ERROR("Csr matrix AMD ordering currenlty unsupported.");
      return PARGEMSLR_ERROR_INVALED_OPTION;
      /*
      if( A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Csr matrix RCM ordering only works on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      int                  err = 0;
      int                  nA;
      CsrMatrixClass<T>    A2;
      CsrMatrixClass<T>    AT;

      nA = A.GetNumRowsLocal();

      perm.Clear();
      perm.Setup(nA);

      PARGEMSLR_CHKERR(nA != A.GetNumColsLocal() || nA < 0);

      // directly return if A is empty
      if(nA==0)
      {
         return 0;
      }

      //-----------------------------------
      //-----Build Graph Data Structure----
      //-----------------------------------

      CsrMatrixTransposeHost(A, AT);
      CsrMatrixAddHost(A, AT, A2);

      AT.Clear();
      if(A2.GetNumNonzeros() < nA)
      {
         PARGEMSLR_ERROR("Zero diagonal in RCM ordering.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }

      A2.SortRow();

      err = amd_order( nA, A2.GetI(), A2.GetJ(), perm.GetData(), NULL, NULL); PARGEMSLR_RETURN_ON_ERROR(err);

      A2.Clear();

      return err;
      */
   }
   template int CsrMatrixAmdHost(CsrMatrixClass<float> &A, vector_int &perm);
   template int CsrMatrixAmdHost(CsrMatrixClass<double> &A, vector_int &perm);
   template int CsrMatrixAmdHost(CsrMatrixClass<complexs> &A, vector_int &perm);
   template int CsrMatrixAmdHost(CsrMatrixClass<complexd> &A, vector_int &perm);

   template <typename T>
   int CsrSubMatrixNdHost(CsrMatrixClass<T> &A, vector_int &rowscols, vector_int &perm)
   {
      int err;

      CsrMatrixClass<T> B;

      /* get that sub matrix */
      err = A.SubMatrix(rowscols, rowscols, kMemoryHost, B); PARGEMSLR_RETURN_ON_ERROR(err);

      /* apply RCM */
      err = CsrMatrixNdHost(B, perm);

      return err;

   }
   template int CsrSubMatrixNdHost(CsrMatrixClass<float> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixNdHost(CsrMatrixClass<double> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixNdHost(CsrMatrixClass<complexs> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixNdHost(CsrMatrixClass<complexd> &A, vector_int &rowscols, vector_int &perm);

   template <typename T>
   int CsrMatrixNdHost(CsrMatrixClass<T> &A, vector_int &perm)
   {
      /*---------------------------------------------------------------------------
      * Nd ordering of a sparse symmetric matrix A.
      *
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * A         = CSR Matrix object.
      *
      * on return:
      * ==========
      * err       = return value.
      *             err  == 0   --> successful return.
      *             err  != 0   --> Error occurs.
      * perm      = Integer vector. Permutation vector.
      *----------------------------------------------------------------------------
      * Note:
      * C-style 0-based index.
      *
      *--------------------------------------------------------------------------*/

      /*
      int      nrows, ncols, nnz, col, i, i1, i2, j, jj, err = 0;
      int      *A_i, *A_j;
      T        *A_data;

      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();

      if (nrows != ncols)
      {
         PARGEMSLR_ERROR("Csr matrix ND ordering only works for square matrix.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }

      if( A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Csr matrix ND ordering only works on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      // # of nnz
      nnz = A.GetNumNonzeros();

      // sort rows of A
      A.SortRow();

      A_i = A.GetI();
      A_j = A.GetJ();
      A_data = A.GetData();

      // Prepare data structures used by METIS
      IntVectorClass<long int> xadj;
      IntVectorClass<long int> adjncy;
      IntVectorClass<long int> vwgt;
      IntVectorClass<long int> perml;
      IntVectorClass<long int> iperml;

      adjncy.Setup(nnz);
      xadj.Setup(nrows+1);
      vwgt.Setup(nrows, true);

      // Fill the vectors with the appropriate values

      perm.Setup(nrows);
      perml.Setup(nrows);
      iperml.Setup(nrows);

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
            }
            else
            {
               vwgt[i] = (long int) PargemslrAbs(A_data[j]);
            }
         }
         if(vwgt[i] == 0)
         {
            // in this case, we don't have a diagonal entry, still give it a default weight
            vwgt[i] = 1.0;
         }
         xadj[i+1] = jj;
      }

      // METIS parameters, note that long int is used

      long int lnrows = (long int)nrows;

      // call METIS

      METIS_NodeND( &lnrows, xadj.GetData(), adjncy.GetData(), vwgt.GetData(), NULL, perml.GetData(), iperml.GetData());

      for ( i = 0; i < nrows; i++)
      {
         perm[i] = (int)perml[i];
      }

      xadj.Clear();
      adjncy.Clear();
      vwgt.Clear();
      perml.Clear();
      iperml.Clear();

      return err;
      */


      PARGEMSLR_ERROR("Csr matrix ND ordering currenlty unsupported.");

      return PARGEMSLR_ERROR_INVALED_OPTION;

   }
   template int CsrMatrixNdHost(CsrMatrixClass<float> &A, vector_int &perm);
   template int CsrMatrixNdHost(CsrMatrixClass<double> &A, vector_int &perm);
   template int CsrMatrixNdHost(CsrMatrixClass<complexs> &A, vector_int &perm);
   template int CsrMatrixNdHost(CsrMatrixClass<complexd> &A, vector_int &perm);

   template <typename T>
   int CsrSubMatrixRcmHost(CsrMatrixClass<T> &A, vector_int &rowscols, vector_int &perm)
   {
      int err;

      CsrMatrixClass<T> B;

      /* get that sub matrix */
      err = A.SubMatrix(rowscols, rowscols, kMemoryHost, B); PARGEMSLR_RETURN_ON_ERROR(err);

      /* apply RCM */
      err = CsrMatrixRcmHost(B, perm);

      return err;

   }
   template int CsrSubMatrixRcmHost(CsrMatrixClass<float> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixRcmHost(CsrMatrixClass<double> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixRcmHost(CsrMatrixClass<complexs> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixRcmHost(CsrMatrixClass<complexd> &A, vector_int &rowscols, vector_int &perm);

   template <typename T>
   int CsrMatrixRcmHost(CsrMatrixClass<T> &A, vector_int &perm)
   {
      /*---------------------------------------------------------------------------
      * RCM ordering of a sparse symmetric matrix A.
      *
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * A         = CSR Matrix object.
      *
      * on return:
      * ==========
      * err       = return value.
      *             err  == 0   --> successful return.
      *             err  != 0   --> Error occurs.
      * perm      = Integer vector. Permutation vector.
      *----------------------------------------------------------------------------
      * Note:
      * C-style 0-based index.
      *
      *--------------------------------------------------------------------------*/

      if( A.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Csr matrix RCM ordering only works on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }

      int                  i, j, k1, k2;
      int                  nA, nodei, current_num;
      vector_int           marker;
      CsrMatrixClass<T>    G;
      CsrMatrixClass<T>    A2;
      CsrMatrixClass<T>    AT;

      nA = A.GetNumRowsLocal();

      perm.Clear();
      perm.Setup(nA);

      PARGEMSLR_CHKERR(nA != A.GetNumColsLocal() || nA < 0);

      /* skip reorder if A is empty */
      if(nA==0)
      {
         return 0;
      }

      //-----------------------------------
      //-----Build Graph Data Structure----
      //-----------------------------------

      CsrMatrixTransposeHost(A, AT);
      CsrMatrixAddHost(A, AT, A2);

      AT.Clear();
      if(A2.GetNumNonzeros() < nA)
      {
         PARGEMSLR_ERROR("Zero diagonal in RCM ordering.");
         return PARGEMSLR_ERROR_INVALED_PARAM;
      }
      G.Setup(nA, nA, A2.GetNumNonzeros()-nA, false, false);

      int   *G_i = G.GetI();
      int   *G_j = G.GetJ();
      int   *A2_i = A2.GetI();
      int   *A2_j = A2.GetJ();

      G_i[0] = 0;
      for(i = 0 ; i < nA ; i ++)
      {
         G_i[i+1] = G_i[i];
         k1 = A2_i[i], k2 = A2_i[i+1];
         for(j = k1 ; j < k2 ; j ++)
         {
            if(A2_j[j] != i)
            {
               G_j[G_i[i+1]++] = A2_j[j];
            }
         }
      }

      A2.Clear();

      //------------------------
      //-----Find RCM ORDER-----
      //------------------------

      //create working array
      marker.Setup(nA);
      marker.Fill(-1);
      current_num = 0;
      while( current_num < nA )
      {
         //find unvised node with minimum degree
         CsrMatrixRcmRootHost(G, marker, nodei);
         //find pseudo-peripheral node
         CsrMatrixRcmPerphnHost(G, nodei, marker);
         //number this connect component
         CsrMatrixRcmNumberingHost(G, nodei, marker, perm, current_num);
      }

      //De-allocate
      marker.Clear();

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixRcmHost(CsrMatrixClass<float> &A, vector_int &perm);
   template int CsrMatrixRcmHost(CsrMatrixClass<double> &A, vector_int &perm);
   template int CsrMatrixRcmHost(CsrMatrixClass<complexs> &A, vector_int &perm);
   template int CsrMatrixRcmHost(CsrMatrixClass<complexd> &A, vector_int &perm);

   template <typename T>
   int CsrMatrixRcmRootHost(CsrMatrixClass<T> &G, vector_int &marker, int &root)
   {
      int                              i, j, k, n;
      int                              nlev, degree, min_degree, lev_degree;
      std::vector<std::vector<int> >   level;

      n           = G.GetNumRowsLocal();
      min_degree  = n + 1;
      root        = 0;

      for(i = 0 ; i < n ; i ++)
      {
         if(marker[i] < 0)
         {
            /* find the connect component starting from here */
            CsrMatrixRcmBfsHost(G, i, marker, level);
            break;
         }
      }

      int *G_i = G.GetI();

      nlev = level.size();
      for(i = 0 ; i < nlev ; i ++)
      {
         lev_degree = level[i].size();
         for(j = 0 ; j < lev_degree ; j ++)
         {
            k = level[i][j];
            degree = G_i[k+1]-G_i[k];
            if( degree < min_degree)
            {
               root = k;
               min_degree = degree;
            }
         }
      }

      CsrMatrixRcmClearLevelHost(level);

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixRcmRootHost(CsrMatrixClass<float> &G, vector_int &marker, int &root);
   template int CsrMatrixRcmRootHost(CsrMatrixClass<double> &G, vector_int &marker, int &root);
   template int CsrMatrixRcmRootHost(CsrMatrixClass<complexs> &G, vector_int &marker, int &root);
   template int CsrMatrixRcmRootHost(CsrMatrixClass<complexd> &G, vector_int &marker, int &root);

   template <typename T>
   int CsrMatrixRcmNumberingHost(CsrMatrixClass<T> &G, int root, vector_int &marker, vector_int &perm, int &current_num)
   {
      /*---------------------------------------------------------------------------
      * RCM numbering for a connect component start form root with BFS.
      *
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * G            = CSR Pattern object. Adjencency graph.
      * root         = Integer. The node we start seartch from.
      * marker       = Integer vector. Helper array, length equal to # of nodes.
      *                Should be allocated already, value will be used.
      *                Untouched nodes should be marked with negative value.
      * perm         = Integer vector. Permutation array, length equal to # of nodes.
      *                Should be allocated already. Value not used and will be overwritten.
      * current_num  = Integer. # of root in the permutation array. Since we might have built
      *                the RCM for some connect components already.
      *
      * on return:
      * ==========
      * err          = return value.
      *                err  == 0   --> successful return.
      *                err  != 0   --> Error occurs.
      * marker       = Integer vector. If node i belongs to the current connect-component,
      *                marker[i] will be set into some positive value.
      * perm         = Integer vector. Permutation vector.
      * current_num  = Integer. # of next root in the permutation array.
      *
      *--------------------------------------------------------------------------*/

      int      i;
      int      j, j1, j2;
      int      nodei;
      int      nodej;
      int      node_start;
      int      node_end;

      int      comp_start        = current_num;
      int      lev_start         = current_num;
      marker[root]               = 0;
      perm[current_num++]        = root;
      int      lev_end           = current_num;
      int      *G_i              = G.GetI();
      int      *G_j              = G.GetJ();

      //explore nbhds of all nodes in current level
      while(lev_end > lev_start)
      {
         //loop through all nodes in current level
         for(i = lev_start ;  i < lev_end ; i ++)
         {
            //node to be explored
            nodei = perm[i];
            //explore nbhds of this node
            node_start = current_num;
            j1 = G_i[nodei];
            j2 = G_i[nodei+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               nodej = G_j[j];
               if(marker[nodej]<0)
               {
                  //this is an unmarked node, save its degree in marker
                  marker[nodej] = G_i[nodej+1] - G_i[nodej];
                  perm[current_num++] = nodej;
               }
            }
            node_end = current_num;
            if(node_end-1 > node_start)
            {
               /* sort based on degree when we have at least 2 nodes */
               std::sort(perm.GetData()+node_end, perm.GetData()+node_end);
            }
         }
         lev_start = lev_end;
         lev_end = current_num;
      }

      //reverse
      CsrMatrixRcmReverseHost(perm, comp_start, lev_end-1);
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixRcmNumberingHost(CsrMatrixClass<float> &G, int root, vector_int &marker, vector_int &perm, int &current_num);
   template int CsrMatrixRcmNumberingHost(CsrMatrixClass<double> &G, int root, vector_int &marker, vector_int &perm, int &current_num);
   template int CsrMatrixRcmNumberingHost(CsrMatrixClass<complexs> &G, int root, vector_int &marker, vector_int &perm, int &current_num);
   template int CsrMatrixRcmNumberingHost(CsrMatrixClass<complexd> &G, int root, vector_int &marker, vector_int &perm, int &current_num);

   template <typename T>
   int CsrMatrixRcmPerphnHost(CsrMatrixClass<T> &G, int &root, vector_int &marker)
   {
      /*---------------------------------------------------------------------------
      * RCM numbering for a connect component start form root with BFS.
      *
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * G            = CSR Pattern object. Adjencency graph.
      * root         = Integer. The node we start seartch from.
      * marker       = Integer vector. Helper array, length equal to # of nodes.
      *                Should be allocated already, value will be used.
      *                Untouched nodes should be marked with negative value.
      *
      * on return:
      * ==========
      * err          = return value.
      *                err  == 0   --> successful return.
      *                err  != 0   --> Error occurs.
      * root         = Integer vector. An end of the pseudo-peripheral.
      *
      *--------------------------------------------------------------------------*/

      int                              i;
      int                              last_level_size;
      int                              min_degree;
      int                              lev_degree;
      std::vector<std::vector<int> >   level;
      //build level structure for root
      int                              nG       = G.GetNumRowsLocal();
      CsrMatrixRcmBfsHost(G, root, marker, level);
      int                              nlev     = level.size();
      int                              newnlev  = nlev + 1;
      int                              *G_i = G.GetI();

      while(nlev < newnlev)
      {
         nlev = level.size();
         std::vector<int> &last_level = level[nlev-1];
         last_level_size = last_level.size();
         min_degree = nG;
         for(i = 0 ; i < last_level_size ; i ++)
         {
            //we select the last level, pick min-degree node
            lev_degree = G_i[last_level[i]+1] - G_i[last_level[i]];
            if(min_degree > lev_degree)
            {
               min_degree = lev_degree;
               root = last_level[i];
            }
         }
         CsrMatrixRcmClearLevelHost(level);
         CsrMatrixRcmBfsHost(G, root, marker, level);
         newnlev = level.size();
      }
      CsrMatrixRcmClearLevelHost(level);
      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixRcmPerphnHost(CsrMatrixClass<float> &G, int &root, vector_int &marker);
   template int CsrMatrixRcmPerphnHost(CsrMatrixClass<double> &G, int &root, vector_int &marker);
   template int CsrMatrixRcmPerphnHost(CsrMatrixClass<complexs> &G, int &root, vector_int &marker);
   template int CsrMatrixRcmPerphnHost(CsrMatrixClass<complexd> &G, int &root, vector_int &marker);

   template <typename T>
   int CsrMatrixRcmBfsHost(CsrMatrixClass<T> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level)
   {
      /*---------------------------------------------------------------------------
      * Apply the BFS start from root.
      *
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * G            = CSR Pattern object. Adjencency graph.
      * root         = Integer. The node we start seartch from.
      * marker       = Integer vector. Helper array, length equal to # of nodes.
      *                Untouched nodes should have negative value. Only search them.
      * level        = Vector of integer vector. Level structure, nodes at level[i]
      *                are on the ith level in the BFS.
      *
      * on return:
      * ==========
      * err          = return value.
      *                err  == 0   --> successful return.
      *                err  != 0   --> Error occurs.
      * root         = Integer vector. An end of the pseudo-peripheral.
      *
      *--------------------------------------------------------------------------*/

      int     i;
      int     j, j1, j2;
      int     nodei;
      int     nodej;
      int     lev_degree;

      //current version use BFS to build this level structure
      //we assume to be given a empty vector level
      level.push_back(std::vector<int>());
      level[0].push_back(root);
      marker[root] = 0;
      int nlev = 0;
      int *G_i = G.GetI();
      int *G_j = G.GetJ();

      //explore nbhds of all nodes in current level
      while(level[nlev].size() > 0)
      {
         //create next level
         level.push_back(std::vector<int>());
         std::vector<int> &last_lev = level[nlev];
         nlev++;
         std::vector<int> &next_lev = level[nlev];
         int last_lev_size = last_lev.size();
         for(i = 0 ;  i < last_lev_size ; i ++)
         {
            //node to be explored
            nodei = last_lev[i];
            //explore nbhds of nodei
            j1 = G_i[nodei];
            j2 = G_i[nodei+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               nodej = G_j[j];
               if(marker[nodej]<0)
               {
                  //an unmarked node
                  marker[nodej] = 0;
                  next_lev.push_back(nodej);
               }
            }
         }
      }
      //the last level is empty, just pop it
      level.pop_back();

      //new we have set the level structure, reset marker array
      for(i = 0 ; i < nlev ; i ++)
      {
         lev_degree = level[i].size();
         for(j = 0 ; j < lev_degree ; j ++)
         {
            marker[level[i][j]] = -1;
         }
      }

      return PARGEMSLR_SUCCESS;
   }
   template int CsrMatrixRcmBfsHost(CsrMatrixClass<float> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level);
   template int CsrMatrixRcmBfsHost(CsrMatrixClass<double> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level);
   template int CsrMatrixRcmBfsHost(CsrMatrixClass<complexs> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level);
   template int CsrMatrixRcmBfsHost(CsrMatrixClass<complexd> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level);

   int CsrMatrixRcmClearLevelHost(std::vector<std::vector<int> > &level)
   {
      /*---------------------------------------------------------------------------
      * Clear level struct (vector of integer vector)
      *
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * level        = Vector of integer vector. Level structure.
      *
      * on return:
      * ==========
      * err          = return value.
      *                err  == 0   --> successful return.
      *                err  != 0   --> Error occurs.
      *
      *--------------------------------------------------------------------------*/

      int     i;
      int     nlev = level.size();
      for(i = 0 ; i < nlev ; i ++)
      {
         std::vector<int>().swap(level[i]);
      }
      std::vector<std::vector<int> >().swap(level);

      return PARGEMSLR_SUCCESS;
   }

   int CsrMatrixRcmSwapHost(vector_int &perm, int a, int b)
   {
      //helper function in sort and reverse. Swap two elements in an array

      int      temp;
      temp     = perm[a];
      perm[a]  = perm[b];
      perm[b]  = temp;
      return PARGEMSLR_SUCCESS;
   }

   int CsrMatrixRcmReverseHost(vector_int &perm, int start, int end)
   {
      //helper function. Reverse permutation to change from CM to RCM.

      int     i;
      int     j;
      int     mid = (start + end + 1) / 2;

      for(i = start, j = end ; i < mid ; i ++, j--)
      {
         CsrMatrixRcmSwapHost(perm, i, j);
      }
      return PARGEMSLR_SUCCESS;
   }

}
