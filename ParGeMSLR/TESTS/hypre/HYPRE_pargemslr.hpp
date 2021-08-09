#ifndef PARGEMSLR_HYPRE_H
#define PARGEMSLR_HYPRE_H

/**
 * @file HYPRE_pargemslr.hpp
 * @brief The interface for hypre
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"

#ifdef __cplusplus
extern "C" 
{
#endif
   /* initialize gemslr */
   HYPRE_Int
   HYPRE_InitGemslr();

   /* finalize gemslr */
   HYPRE_Int
   HYPRE_FinalizeGemslr();

   /* create the gemslr solver */
   HYPRE_Int
   HYPRE_GEMSLRCreate( HYPRE_Solver *solver );

   /* create the gemslr solver */
   HYPRE_Int
   HYPRE_GEMSLRCreateFromFile( HYPRE_Solver *solver, const char *filename );

   HYPRE_Int
   HYPRE_GEMSLRDestroy( HYPRE_Solver solver );

   HYPRE_Int
   HYPRE_GEMSLRSetup( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x      );
   HYPRE_Int
   HYPRE_GEMSLRSolve( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x      );
#ifdef __cplusplus
}
#endif

#endif
