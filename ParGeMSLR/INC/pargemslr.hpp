#ifndef PARGEMSLR_H
#define PARGEMSLR_H

/**
 * @file pargemslr.hpp
 * @brief Global header of the ParSLR
 */

#ifndef PARGEMSLR_SUCCESS
#define PARGEMSLR_SUCCESS        	                  0
#define PARGEMSLR_RETURN_METIS_INSUFFICIENT_NDOM      1
#define PARGEMSLR_RETURN_METIS_NO_INTERIOR            2
#define PARGEMSLR_RETURN_METIS_ISOLATE_NODE           3
#define PARGEMSLR_ERROR_INVALED_OPTION                100
#define PARGEMSLR_ERROR_INVALED_PARAM   	            101
#define PARGEMSLR_ERROR_IO_ERROR        	            102
#define PARGEMSLR_ERROR_ILU_EMPTY_ROW   	            103
#define PARGEMSLR_ERROR_DOUBLE_INIT_FREE              104 // call init function for multiple times
#define PARGEMSLR_ERROR_COMPILER                      105
#define PARGEMSLR_ERROR_FUNCTION_CALL_ERR             106
#define PARGEMSLR_ERROR_MEMORY_LOCATION               107
#endif

#include "../SRC/utils/structs.hpp"
#include "../SRC/utils/utils.hpp"
#include "../SRC/utils/parallel.hpp"
#include "../SRC/utils/memory.hpp"
#include "../SRC/utils/protos.hpp"
#include "../SRC/utils/mmio.hpp"

#include "../SRC/vectors/vector.hpp"
#include "../SRC/vectors/sequential_vector.hpp"
#include "../SRC/vectors/parallel_vector.hpp"
#include "../SRC/vectors/int_vector.hpp"
#include "../SRC/vectors/vectorops.hpp"

#include "../SRC/matrices/matrix.hpp"
#include "../SRC/matrices/matrixops.hpp"
#include "../SRC/matrices/coo_matrix.hpp"
#include "../SRC/matrices/csr_matrix.hpp"
#include "../SRC/matrices/dense_matrix.hpp"
#include "../SRC/matrices/parallel_csr_matrix.hpp"

#include "../SRC/solvers/solver.hpp"
#include "../SRC/solvers/fgmres.hpp"

#include "../SRC/preconditioners/ilu.hpp"
#include "../SRC/preconditioners/poly.hpp"
#include "../SRC/preconditioners/block_jacobi.hpp"
#include "../SRC/preconditioners/parallel_mix_precond.hpp"
#include "../SRC/preconditioners/gemslr.hpp"
#include "../SRC/preconditioners/parallel_gemslr.hpp"

#endif
