#ifndef PARGEMSLR_UTILS_H
#define PARGEMSLR_UTILS_H

/**
 * @file utils.hpp
 * @brief Basic ultility functions.
 */

#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <type_traits>
#include <ctype.h>
#include "complex.hpp"
#include <random>
#include <climits>
#include <cassert>

#define PARGEMSLR_FIRM_CHKERR(ierr) {{if(ierr){printf("Err value: %d on MPI rank %d\n",ierr, parallel_log::_grank);assert(!(ierr));};}}

#ifdef PARGEMSLR_DEBUG
   #define PARGEMSLR_ERROR(message) {printf("Error: %s on MPI rank %d\n", message, parallel_log::_grank);assert(0);}
   #define PARGEMSLR_CHKERR(ierr) {{if(ierr){printf("Err value: %d\n",ierr);assert(!(ierr));};}}
   #define PARGEMSLR_PRINT_DEBUG(conda, condb, ...) {if(conda==condb) {printf("DEBUG: ");printf(__VA_ARGS__);}}
#else
   #define PARGEMSLR_ERROR(message) {printf("Error: %s on MPI rank %d\n", message, parallel_log::_grank);}
   #define PARGEMSLR_CHKERR(ierr) {;}
   #define PARGEMSLR_PRINT_DEBUG(conda, condb, ...) {;}
#endif
#ifdef PARGEMSLR_NO_WARNING
   #define PARGEMSLR_WARNING(message) {;}
#else
   #define PARGEMSLR_WARNING(message) {printf("Warning: %s on MPI rank %d\n", message, parallel_log::_grank);}
#endif

#define PARGEMSLR_PRINT(...) fprintf(pargemslr::pargemslr_global::_out_file, __VA_ARGS__);

#ifndef PARGEMSLR_SUCCESS
#define PARGEMSLR_SUCCESS        	                  0
#define PARGEMSLR_RETURN_METIS_INSUFFICIENT_NDOM      1
#define PARGEMSLR_RETURN_METIS_NO_INTERIOR            2
#define PARGEMSLR_RETURN_METIS_PROBLEM_TOO_SMALL      3
#define PARGEMSLR_RETURN_PARILU_NO_INTERIOR           10
#define PARGEMSLR_ERROR_INVALED_OPTION                100
#define PARGEMSLR_ERROR_INVALED_PARAM   	            101
#define PARGEMSLR_ERROR_IO_ERROR        	            102
#define PARGEMSLR_ERROR_ILU_EMPTY_ROW   	            103
#define PARGEMSLR_ERROR_DOUBLE_INIT_FREE              104 // call init function for multiple times
#define PARGEMSLR_ERROR_COMPILER                      105
#define PARGEMSLR_ERROR_FUNCTION_CALL_ERR             106
#define PARGEMSLR_ERROR_MEMORY_LOCATION               107
#endif

#define PARGEMSLR_CAST( type, val) reinterpret_cast<type>((val))

/* The following is the detail of all the options 
 * The GEMSLR supports 4 different options yet.
 * 
 * BJ -> Block Jacobi.
 *       This option uses the input partition of matrix A, apply local solve on the diagonal blocks.
 * 
 * ESCHUR-> PartialILUT to form Approximate Schur on the top level.
 *       This option uses the PartialILUT to form an approximate Schur complement.
 *       The input partition of matrix A is used.
 *       Local solve can only be the PartialILUT.
 * 
 * ISCHUR-> GEMSLR with IO ordering on the top level.
 *       This option uses the GeMSLR to approximate the Schur complement.
 *       The input partition of marix A is used.
 * 
 * MLEV-> GEMSLR with true multilevel.
 *       This option uses the GeMSLR to approximate the Schur complement.
 *       ParMETIS is used on the top level to comstruct the partition.
 * 
 */

#define PARGEMSLR_IO_SIZE                             180

/*----------- Choose preconditioner */

/* the global preconditioenr. 0: BJ; 1: ESCHUR; 2: ISCHUR; 3: MLEV */
#define PARGEMSLR_IO_PRECOND_GLOBAL_PRECOND           0
/* the first local preconditioner option on the B part. 0: ILUT; 1: GEMSLR. */
#define PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1           1
/* the levels we apply the first preconditioner (start from level 0 till which level we apply local preconditioner 1) */
#define PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1_LEVEL     2
/* the second local preconditioner option on the B part. 0: ILUT; 1: GEMSLR. Before the last level we apply this second preconditioner */
#define PARGEMSLR_IO_PRECOND_LOCAL_REPCOND2           3
/* the local preconditioner option for the last C block. 0: ILUT; 1: ILUK; 2: BJ-ILUT; 3: BJ-ILUK; 4: BJ-ILUT2; 5: BJ-ILUK2. 
 * On the global last level we apply this last preconditioner */
#define PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3           4
/* the smoother for the MILU in the B part. 0: B-ILU; 1: B-MILU. */
#define PARGEMSLR_IO_PRECOND_LOCAL_SMOOTHER1          5

/*----------- Choose solver */

/* the solver. 0: FGMRES. */
#define PARGEMSLR_IO_SOLVER_OPTION                    10
#define PARGEMSLR_IO_SOLVER_KDIM                      11
#define PARGEMSLR_IO_SOLVER_MAXITS                    12
#define PARGEMSLR_IO_SOLVER_TOL                       13
#define PARGEMSLR_IO_SOLVER_ATOL                      14

/*----------- partition parameters */

/* the global number of levels for parallel solver, the IO order is considered as one level */
#define PARGEMSLR_IO_PREPOSS_NLEV_GLOBAL              20
/* the local number of levels for the B part local solve */
#define PARGEMSLR_IO_PREPOSS_NLEV_LOCAL               21
/* the number of domains on each level, except the top level if IO order is used */
#define PARGEMSLR_IO_PREPOSS_NCOMP_GLOBAL             22
/* the number of domains on each level */
#define PARGEMSLR_IO_PREPOSS_NCOMP_LOCAL              23
/* the global partition option. 0: parallel ND; 1: parallel RKWay */
#define PARGEMSLR_IO_PREPOSS_PARTITION_GLOBAL         24
/* the local partition option. 0: ND; 1: RKWay */
#define PARGEMSLR_IO_PREPOSS_PARTITION_LOCAL          25
/* the minimal size of edge separator, use same value for parallel and local */
#define PARGEMSLR_IO_PREPOSS_MINSEP                   26
/* the minimal size of number of domains for the global RKWay partition */
#define PARGEMSLR_IO_PREPOSS_KMIN_GLOBAL              27
/* the minimal size of number of domains for the local RKWay partition */
#define PARGEMSLR_IO_PREPOSS_KMIN_LOCAL               28
/* the reduce factor of number of domains for the global RKWay partition */
#define PARGEMSLR_IO_PREPOSS_KFACTOR_GLOBAL           29
/* the reduce factor of number of domains for the local RKWay partition */
#define PARGEMSLR_IO_PREPOSS_KFACTOR_LOCAL            30
/* use vertex seperator in global? 0: no; 1: yes. */
#define PARGEMSLR_IO_PREPOSS_VTXSEP_GLOBAL            31
/* use vertex seperator in local? 0: no; 1: yes. */
#define PARGEMSLR_IO_PREPOSS_VTXSEP_LOCAL             32
/* numbers of metis refine on parallel partition */
#define PARGEMSLR_IO_PREPOSS_METIS_REFINE             33
/* use global partition? 0: no; 1: yes. */
#define PARGEMSLR_IO_PREPOSS_GLOBAL_PARTITION         34

/*----------- ILU parameters */

/* droptol for the ILUT or the B block of global GeMSLR / PartialILUT */
#define PARGEMSLR_IO_ILU_DROPTOL_B_GLOBAL             40
/* droptol for the ILUT or the B block of local GeMSLR */
#define PARGEMSLR_IO_ILU_DROPTOL_B_LOCAL              41
/* droptol for the E/F part of the global PartialILUT */
#define PARGEMSLR_IO_ILU_DROPTOL_EF_GLOBAL            42
/* droptol for the E/F part of the global PartialILUT */
#define PARGEMSLR_IO_ILU_DROPTOL_EF_LOCAL             43
/* droptol for the last level of global GeMSLR */
#define PARGEMSLR_IO_ILU_DROPTOL_C_GLOBAL             44
/* droptol for the last level of lobal GeMSLR */
#define PARGEMSLR_IO_ILU_DROPTOL_C_LOCAL              45
/* droptol for the S part of the global PartialILUT */
#define PARGEMSLR_IO_ILU_DROPTOL_S_GLOBAL             46
/* droptol for the S part of the local PartialILUT */
#define PARGEMSLR_IO_ILU_DROPTOL_S_LOCAL              47
/* maximal row nonzeros for the ILUT or the B, E, F block of global GeMSLR / PartialILUT */
#define PARGEMSLR_IO_ILU_ROWNNZ_B_GLOBAL              48
/* maximal row nonzeros for the ILUT or the B, E, F block of local GeMSLR / PartialILUT */
#define PARGEMSLR_IO_ILU_ROWNNZ_B_LOCAL               49
/* maximal row nonzeros for the last level of global GeMSLR */
#define PARGEMSLR_IO_ILU_ROWNNZ_C_GLOBAL              50
/* maximal row nonzeros for the last level of local GeMSLR */
#define PARGEMSLR_IO_ILU_ROWNNZ_C_LOCAL               51
/* lfil for the ILUK of the B, E, F block of global GeMSLR / PartialILUT */
#define PARGEMSLR_IO_ILU_LFIL_B_GLOBAL                52
/* lfil for the ILUK of the B, E, F block of local GeMSLR / PartialILUT */
#define PARGEMSLR_IO_ILU_LFIL_B_LOCAL                 53
/* lfil for the ILUK for the last level of global GeMSLR */
#define PARGEMSLR_IO_ILU_LFIL_C_GLOBAL                54
/* lfil for the ILUK for the last level of local GeMSLR */
#define PARGEMSLR_IO_ILU_LFIL_C_LOCAL                 55
/* maximal row nonzeros for the last level of global PartialILUT */
#define PARGEMSLR_IO_ILU_ROWNNZ_S_GLOBAL              56
/* maximal row nonzeros for the last level of local PartialILUT */
#define PARGEMSLR_IO_ILU_ROWNNZ_S_LOCAL               57
/* reordering option for the global GeMSLR */
#define PARGEMSLR_IO_ILU_PERM_OPTION_GLOBAL           58
/* reordering option for the local GeMSLR */
#define PARGEMSLR_IO_ILU_PERM_OPTION_LOCAL            59
/* openmp option for the global GeMSLR */
#define PARGEMSLR_IO_ILU_OMP_OPTION_GLOBAL            60
/* openmp option for the local GeMSLR */
#define PARGEMSLR_IO_ILU_OMP_OPTION_LOCAL             61

/*----------- low-rank parameters */

/* Arnoldi option for the top level global GeMSLR */
#define PARGEMSLR_IO_LR_ARNOLDI_OPTION1_GLOBAL        70
/* Arnoldi option for other levels global GeMSLR */
#define PARGEMSLR_IO_LR_ARNOLDI_OPTION2_GLOBAL        71
/* Arnoldi option for the top level local GeMSLR */
#define PARGEMSLR_IO_LR_ARNOLDI_OPTION1_LOCAL         72
/* Arnoldi option for other levels local GeMSLR */
#define PARGEMSLR_IO_LR_ARNOLDI_OPTION2_LOCAL         73
/* rank for the top level global GeMSLR */
#define PARGEMSLR_IO_LR_RANK1_GLOBAL                  74
/* rank for other levels global GeMSLR */
#define PARGEMSLR_IO_LR_RANK2_GLOBAL                  75
/* rank for the top level local GeMSLR */
#define PARGEMSLR_IO_LR_RANK1_LOCAL                   76
/* rank for other levels local GeMSLR */
#define PARGEMSLR_IO_LR_RANK2_LOCAL                   77
/* rank factor for the top level global GeMSLR */
#define PARGEMSLR_IO_LR_RANK_FACTOR1_GLOBAL           78
/* rank factor for other levels global GeMSLR */
#define PARGEMSLR_IO_LR_RANK_FACTOR2_GLOBAL           79
/* rank factor for the top level local GeMSLR */
#define PARGEMSLR_IO_LR_RANK_FACTOR1_LOCAL            80
/* rank factor for other levels local GeMSLR */
#define PARGEMSLR_IO_LR_RANK_FACTOR2_LOCAL            81
/* arnoldi factor for the top level global GeMSLR */
#define PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_GLOBAL        82
/* arnoldi factor for other levels global GeMSLR */
#define PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_GLOBAL        83
/* arnoldi factor for the top level local GeMSLR */
#define PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_LOCAL         84
/* arnoldi factor for other levels local GeMSLR */
#define PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_LOCAL         85
/* eigenvalue accuracy for the top level global GeMSLR */
#define PARGEMSLR_IO_LR_TOL_EIG1_GLOBAL               86
/* eigenvalue accuracy for other levels global GeMSLR */
#define PARGEMSLR_IO_LR_TOL_EIG2_GLOBAL               87
/* eigenvalue accuracy for the top level local GeMSLR */
#define PARGEMSLR_IO_LR_TOL_EIG1_LOCAL                88
/* eigenvalue accuracy for other levels local GeMSLR */
#define PARGEMSLR_IO_LR_TOL_EIG2_LOCAL                89
/* max number of restarts for the top level global GeMSLR */
#define PARGEMSLR_IO_LR_MAXITS1_GLOBAL                90
/* max number of restarts for other levels global GeMSLR */
#define PARGEMSLR_IO_LR_MAXITS2_GLOBAL                91
/* max number of restarts for the top level local GeMSLR */
#define PARGEMSLR_IO_LR_MAXITS1_LOCAL                 92
/* max number of restarts for other levels local GeMSLR */
#define PARGEMSLR_IO_LR_MAXITS2_LOCAL                 93
/* the thick-restart factor */
#define PARGEMSLR_IO_LR_TR_FACTOR                     94
/* the orthogonal threshold */
#define PARGEMSLR_IO_LR_TOL_ORTH                      95
/* the re-orthogonal threshold */
#define PARGEMSLR_IO_LR_TOL_REORTH                    96
/* should we use random initla guess? 0: no; 1: yes. (for parallel random vector, different np might leads to different result) */
#define PARGEMSLR_IO_LR_RAND_INIT_GUESS               97
/* rank for special global GeMSLR options */
#define PARGEMSLR_IO_LR_RANK_C_NITER                  98
/* Arnoldi option for the A */
#define PARGEMSLR_IO_LR_ARNOLDI_OPTIONA               99
/* rank for the A correction */
#define PARGEMSLR_IO_LR_RANK_A                        100
/* rank factor for the A */
#define PARGEMSLR_IO_LR_RANK_FACTORA                  101
/* arnoldi factor for the A */
#define PARGEMSLR_IO_LR_ARNOLDI_FACTORA               102
/* eigenvalue accuracy for the A */
#define PARGEMSLR_IO_LR_TOL_EIGA                      103
/* max number of restarts for the A */
#define PARGEMSLR_IO_LR_MAXITSA                       104

/*----------- schur option */

/* turn on schur iteration? 0: no; 1: yes */
#define PARGEMSLR_IO_SCHUR_ENABLE                     130
/* max number of schur iterations */
#define PARGEMSLR_IO_SCHUR_ITER_TOL                   131
/* stop tolerance for the schur FGMRES */
#define PARGEMSLR_IO_SCHUR_MAXITS                     132

/*----------- other options */

/* precondition with A+sI (complex version only) */
#define PARGEMSLR_IO_ADVANCED_DIAG_SHIFT              140
/* global solve option. 0: Add solve; 1: Add solve L only; 2: Mult solve */
#define PARGEMSLR_IO_ADVANCED_GLOBAL_SOLVE            141
/* should we turn on complex shift in ZILUT? */
#define PARGEMSLR_IO_ADVANCED_USE_COMPLEX_SHIFT       142
/* number of residual iterations for the B solve in the setup phase */
#define PARGEMSLR_IO_ADVANCED_RESIDUAL_ITERS          143
/* the gram schmidt option, see pargemslr_global */
#define PARGEMSLR_IO_ADVANCED_GRAM_SCHMIDT            144
/* the diagonal shift for the modified ILU, not in use now */
#define PARGEMSLR_IO_ADVANCED_DIAG_SHIFT_MODIFIED     145

/*----------- general options */

/* the print option */
#define PARGEMSLR_IO_GENERAL_PRINT_LEVEL              150

/*----------- poly preconditioner options */

#define PARGEMSLR_IO_POLY_ORDER                       160

/*----------- parallel options */

#define PARGEMSLR_IO_PARALLEL_NPROCX                  170
#define PARGEMSLR_IO_PARALLEL_NPROCY                  171
#define PARGEMSLR_IO_PARALLEL_NPROCZ                  172
#define PARGEMSLR_IO_PARALLEL_NDOMX                   173
#define PARGEMSLR_IO_PARALLEL_NDOMY                   174
#define PARGEMSLR_IO_PARALLEL_NDOMZ                   175

namespace pargemslr
{
   /** 
    * @brief   The data structure for parallel computing, including data structures for MPI and CUDA.
    * @details The data structure for parallel computing, including data structures for MPI and CUDA. \n
    *          All CUDA information are shared, local MPI information can be different.
    */
   typedef class PargemslrGlobalClass 
   {
      public:
      
      /**
       * @brief   Expand factor, used when expand vectors with PushBack.
       * @details Expand factor, used when expand vectors with PushBack.
       */
      static double                                _expand_fact;
      
      /**
       * @brief   Reserved size of COO matrix, default is nrow * _coo_reserve_fact.
       * @details Reserved size of COO matrix, default is nrow * _coo_reserve_fact.
       */
      static int                                   _coo_reserve_fact;
      
      /**
       * @brief   Disable OpenMP in some cases when a loop size is too small.
       * @details Disable OpenMP in some cases when a loop size is too small.
       */
      static int                                   _openmp_min_loopsize;
      
      /**
       * @brief   Numter of Metis refine (for ParMetis).
       * @details Numter of Metis refine (for ParMetis).
       */
      static int                                   _metis_refine;
      
      /**
       * @brief   The tolorance for loading balance when apply the parallel kway partition.
       * @details The tolorance for loading balance when apply the parallel kway partition.
       */
      static double                                _metis_loading_balance_tol;
      
      /**
       * @brief   Min size of the edge separator.
       * @details Min size of the edge separator.
       */
      static int                                   _minsep;
      
      /**
       * @brief   Thick-restart factor for thick-restart Arnoldi.
       * @details Thick-restart factor for thick-restart Arnoldi.
       */
      static double                                _tr_factor;
      
      /**
       * @brief   The tolorance for orthogonalization for Arnoldi.
       * @details The tolorance for orthogonalization for Arnoldi.
       */
      static double                                _orth_tol;
      
      /**
       * @brief   The tolorance for re-orthogonalization for Arnoldi.
       * @details The tolorance for re-orthogonalization for Arnoldi.
       */
      static double                                _reorth_tol;
      
      /**
       * @brief   Default width of the dashline in the output.
       * @details Default width of the dashline in the output.
       */
      static int                                   _dash_line_width;
      
      /**
       * @brief   Used to obtain a seed for the random number engine.
       * @details Used to obtain a seed for the random number engine.
       */
      static std::random_device                    _random_device;
      
      /**
       * @brief   Mersenne_twister_engine.
       * @details Mersenne_twister_engine.
       */
      static std::mt19937                          _mersenne_twister_engine;
      
      /**
       * @brief   Uniform_int_distribution.
       * @details Uniform_int_distribution.
       */
      static std::uniform_int_distribution<int>    _uniform_int_distribution;
      
      /**
       * @brief   The output file, default is stdout.
       * @details The output file, default is stdout.
       */
      static FILE                                  *_out_file;
      
      /**
       * @brief   The global parameters' array.
       * @details The global parameters' array.
       */
      static double                                *_params;
      
      /**
       * @brief   Gram schmidt option for the eigenvalue solver.
       * @details Gram schmidt option for the eigenvalue solver. \n
       *          0: CGS-2.
       *          1: MGS.
       */
      static int                                   _gram_schmidt;
      
   }pargemslr_global;
   
   /**
    * @brief   Set the default parameter array.
    * @details Set the default parameter array.
    * @param [in] params The parameter array, memroy should be allocated.
    * @return  Return this array.
    */
   double* PargemslrSetDefaultParameterArray(double *params);
   
   /**
    * @brief   Tell if a value is integer.
    * @details Tell if a value is integer.
    */
   template <class T> struct PargemslrIsInteger : public std::false_type {};
   template <class T> struct PargemslrIsInteger<const T> : public PargemslrIsInteger<T> {};
   template <class T> struct PargemslrIsInteger<volatile const T> : public PargemslrIsInteger<T>{};
   template <class T> struct PargemslrIsInteger<volatile T> : public PargemslrIsInteger<T>{};
   template<> struct PargemslrIsInteger<int> : public std::true_type {};
   template<> struct PargemslrIsInteger<long int> : public std::true_type {};
   
   /**
    * @brief   Tell if a value is in double precision.
    * @details Tell if a value is in double precision.
    */
   template <class T> struct PargemslrIsDoublePrecision : public std::false_type {};
   template <class T> struct PargemslrIsDoublePrecision<const T> : public PargemslrIsDoublePrecision<T> {};
   template <class T> struct PargemslrIsDoublePrecision<volatile const T> : public PargemslrIsDoublePrecision<T>{};
   template <class T> struct PargemslrIsDoublePrecision<volatile T> : public PargemslrIsDoublePrecision<T>{};
   template<> struct PargemslrIsDoublePrecision<double> : public std::true_type {};
   template<> struct PargemslrIsDoublePrecision<complexd> : public std::true_type {};
   
   /**
    * @brief   Tell if a value is a parallel data structure.
    * @details Tell if a value is a parallel data structure.
    */
   template <class T> struct PargemslrIsParallel : public std::false_type {};
   template <class T> struct PargemslrIsParallel<const T> : public PargemslrIsParallel<T> {};
   template <class T> struct PargemslrIsParallel<volatile const T> : public PargemslrIsParallel<T>{};
   
   /**
    * @brief   The precision enum.
    * @details The precision enum.
    */
   enum PrecisionEnum
   {
      kUnknownPrecision = -1,
      kInt,
      kLongInt,
      kHalfReal,
      kHalfComplex,
      kSingleReal,
      kSingleComplex,
      kDoubleReal,
      kDoubleComplex
   };
   
   /**
    * @brief   The struct of for sorting.
    * @details The struct of for sorting.
    */
   template <typename T>
   struct CompareStruct
   {
      T     val;
      int   ord;
   };
   
   typedef CompareStruct<int>       compareord_int;
   typedef CompareStruct<long int>  compareord_long;
   typedef CompareStruct<float>     compareord_float;
   typedef CompareStruct<double>    compareord_double;
   
   /**
    * @brief   The operator > for CompareStruct.
    * @details The operator > for CompareStruct.
    * @param [in]   a First value.
    * @param [in]   b Second value.
    * @return       Return true or false.
    */
   template <class T>
   struct CompareStructGreater
   {
       bool operator()(T const &a, T const &b) const { return a.val > b.val; }
   };
   
   /**
    * @brief   The operator < for CompareStruct.
    * @details The operator < for CompareStruct.
    * @param [in]   a First value.
    * @param [in]   b Second value.
    * @return       Return true or false.
    */
   template <class T>
   struct CompareStructLess
   {
       bool operator()(T const &a, T const &b) const { return a.val < b.val; }
   };
   
   /**
    * @brief   Get the larger one out of two numbers.
    * @details Get the larger one out of two numbers.
    * @param [in]   a First value.
    * @param [in]   b Second value.
    * @return       Return the larger value.
    */
   template <typename T>
   T PargemslrMax(T a, T b);
   
   /**
    * @brief   Get the smaller one out of two numbers.
    * @details Get the smaller one out of two numbers.
    * @param [in]   a First value.
    * @param [in]   b Second value.
    * @return       Return the smaller value.
    */
   template <typename T>
   T PargemslrMin(T a, T b);
   
   /**
    * @brief   Get the absolute value of a numbers.
    * @details Get the absolute value of a numbers.
    * @param [in]   a The value.
    * @return       Return the absolute value.
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, T>::type
   PargemslrAbs( const T &a);
   
   /**
    * @brief   Get the absolute value of a numbers.
    * @details Get the absolute value of a numbers.
    * @param [in]   a The value.
    * @return       Return the absolute value.
    */
   float PargemslrAbs( const complexs &a);
   
   /**
    * @brief   Get the absolute value of a numbers.
    * @details Get the absolute value of a numbers.
    * @param [in]   a The value.
    * @return       Return the absolute value.
    */
   double PargemslrAbs( const complexd &a);
   
   /**
    * @brief   Get the real part of a numbers.
    * @details Get the real part value of a numbers.
    * @param [in]   a The value.
    * @return       Return the real part value.
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, T>::type
   PargemslrReal( const T &a);
   
   /**
    * @brief   Get the real part of a numbers.
    * @details Get the real part of a numbers.
    * @param [in]   a The value.
    * @return       Return the real part value.
    */
   float PargemslrReal( const complexs &a);
   
   /**
    * @brief   Get the real part of a numbers.
    * @details Get the real part of a numbers.
    * @param [in]   a The value.
    * @return       Return the real part value.
    */
   double PargemslrReal( const complexd &a);
   
   /**
    * @brief   Get the conjugate of a numbers, for real value we do nothing.
    * @details Get the conjugate of a numbers, for real value we do nothing.
    * @param [in]   a The value.
    * @return       Return the conjugate.
    */
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, T>::type
   PargemslrConj( const T &a);
   
   /**
    * @brief   Get the conjugate of a numbers, for real value we do nothing.
    * @details Get the conjugate of a numbers, for real value we do nothing.
    * @param [in]   a The value.
    * @return       Return the conjugate.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, T>::type
   PargemslrConj( const T &a);
   
   /**
    * @brief   Generate random integer number at host memory.
    * @details Generate random integer number at host memory.
    * @param [out]   a The random value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsInteger<T>::value, int>::type
   PargemslrValueRandHost(T &a);
   
   /**
    * @brief   Generate random float number between [0, 1] at host memory.
    * @details Generate random float number between [0, 1] at host memory.
    * @param [out]   a The random value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsReal<T>::value, int>::type
   PargemslrValueRandHost(T &a);
   
   /**
    * @brief   Generate random single complex number, real and imag part both between [0, 1] at host memory.
    * @details Generate random single complex number, real and imag part both between [0, 1] at host memory.
    * @param [out]   a The random value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrValueRandHost(T &a);
   
   /**
    * @brief   Print spaces using cout.
    * @details Print spaces using cout.
    * @param [in]   width The width of the space.
    * @return       Return error message.
    */
   int PargemslrPrintSpace(int width);
   
   /**
    * @brief   Print a dash line using cout.
    * @details Print a dash line using cout.
    * @param [in]   width The width of the line.
    * @return       Return error message.
    */
   int PargemslrPrintDashLine(int width);
   
   /**
    * @brief   Print a integer value, fixed width.
    * @details Print a integer value, fixed width.
    * @param [in]   val The value.
    * @param [in]   width The width of the value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsInteger<T>::value, int>::type 
   PargemslrPrintValueHost(T val, int width);
   
   /**
    * @brief   Print a real value, fixed width.
    * @details Print a real value, fixed width.
    * @param [in]   val The value.
    * @param [in]   width The width of the value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsReal<T>::value, int>::type 
   PargemslrPrintValueHost(T val, int width);
   
   /**
    * @brief   Print a complex value, fixed width.
    * @details Print a complex value, fixed width.
    * @param [in]   val The value.
    * @param [in]   width The width of the value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type 
   PargemslrPrintValueHost(T val, int width);
   
   /**
    * @brief   Read the first word from a input string, convert to upper case.
    * @details Read the first word from a input string, convert to upper case.
    * @param [in]   pin The input char*.
    * @param [out]  pout The output char**.
    */
   void PargemslrReadFirstWord(char *pin, char **pout);

   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int PargemslrReadInputArg(const char *argname, int amount, float *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int PargemslrReadInputArg(const char *argname, int amount, double *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int PargemslrReadInputArg(const char *argname, int amount, complexs *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int PargemslrReadInputArg(const char *argname, int amount, complexd *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int PargemslrReadInputArg(const char *argname, int amount, int *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int PargemslrReadInputArg(const char *argname, char *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int PargemslrReadInputArg(const char *argname, bool *val, int argc, char **argv);
   
   /**
    * @brief   Check if we have an argument. If we want to find "xxx", the user input should be "-xxx".
    * @details Check if we have an argument. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int PargemslrReadInputArg(const char *argname, int argc, char **argv);
   
   /**
    * @brief   Plot the data to the terminal output.
    * @details Plot the data to the terminal output.
    * @param [in]   data The data.
    * @param [in]   length length of the data.
    * @param [in]   numx the number of grids on x.
    * @param [in]   numx the number of grids on y.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<!(PargemslrIsComplex<T>::value), int>::type
   PargemslrPlotData(T* ydata, int length, int numx, int numy);
   
   /**
    * @brief   Plot the data to the terminal output.
    * @details Plot the data to the terminal output.
    * @param [in]   argname The target argument.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrPlotData(T* ydata, int length, int numx, int numy);
   
   /**
    * @brief   Set output file of PARGEMSLR_PTINT.
    * @details Set output file of PARGEMSLR_PTINT.
    * @param [in]   filename The pointer of the file.
    * @return       Return error message.
    */
   int PargemslrSetOutputFile(const char *filename);
   
}

#endif
