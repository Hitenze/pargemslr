#ifndef PARGEMSLR_PROTOS_H
#define PARGEMSLR_PROTOS_H

/**
 * @file protos.hpp
 * @brief Protos for external functions used in PARGEMSLR, and the interface call
 */

#include "utils.hpp"

#ifdef PARGEMSLR_HYPRE

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"

#endif

/* add PARGEMSLR_ to make those name unique, in case they might be used in other librarys */

#ifdef PARGEMSLR_MKL
//#include "mkl_types.h"
#include "mkl.h"

#define PARGEMSLR_BLASLAPACK_SAXPY        saxpy
#define PARGEMSLR_BLASLAPACK_DAXPY        daxpy
#define PARGEMSLR_BLASLAPACK_CAXPY        caxpy
#define PARGEMSLR_BLASLAPACK_ZAXPY        zaxpy

#define PARGEMSLR_BLASLAPACK_SSCAL        sscal
#define PARGEMSLR_BLASLAPACK_DSCAL        dscal
#define PARGEMSLR_BLASLAPACK_CSCAL        cscal
#define PARGEMSLR_BLASLAPACK_ZSCAL        zscal

#define PARGEMSLR_BLASLAPACK_SDOT         sdot
#define PARGEMSLR_BLASLAPACK_DDOT         ddot
#define PARGEMSLR_BLASLAPACK_CDOTC        cdotc
#define PARGEMSLR_BLASLAPACK_ZDOTC        zdotc

#define PARGEMSLR_BLASLAPACK_SGEMV        sgemv
#define PARGEMSLR_BLASLAPACK_DGEMV        dgemv
#define PARGEMSLR_BLASLAPACK_CGEMV        cgemv
#define PARGEMSLR_BLASLAPACK_ZGEMV        zgemv

#define PARGEMSLR_BLASLAPACK_SGEMM        sgemm
#define PARGEMSLR_BLASLAPACK_DGEMM        dgemm
#define PARGEMSLR_BLASLAPACK_CGEMM        cgemm
#define PARGEMSLR_BLASLAPACK_ZGEMM        zgemm

#define PARGEMSLR_BLASLAPACK_STRTRI       strtri
#define PARGEMSLR_BLASLAPACK_DTRTRI       dtrtri
#define PARGEMSLR_BLASLAPACK_CTRTRI       ctrtri
#define PARGEMSLR_BLASLAPACK_ZTRTRI       ztrtri

#define PARGEMSLR_BLASLAPACK_SGETRF       sgetrf
#define PARGEMSLR_BLASLAPACK_DGETRF       dgetrf
#define PARGEMSLR_BLASLAPACK_CGETRF       cgetrf
#define PARGEMSLR_BLASLAPACK_ZGETRF       zgetrf

#define PARGEMSLR_BLASLAPACK_SGETRI       sgetri
#define PARGEMSLR_BLASLAPACK_DGETRI       dgetri
#define PARGEMSLR_BLASLAPACK_CGETRI       cgetri
#define PARGEMSLR_BLASLAPACK_ZGETRI       zgetri

#define PARGEMSLR_BLASLAPACK_SGEHRD       sgehrd
#define PARGEMSLR_BLASLAPACK_DGEHRD       dgehrd
#define PARGEMSLR_BLASLAPACK_CGEHRD       cgehrd
#define PARGEMSLR_BLASLAPACK_ZGEHRD       zgehrd

#define PARGEMSLR_BLASLAPACK_SORGQR       sorgqr
#define PARGEMSLR_BLASLAPACK_DORGQR       dorgqr
#define PARGEMSLR_BLASLAPACK_CUNGQR       cungqr
#define PARGEMSLR_BLASLAPACK_ZUNGQR       zungqr

#define PARGEMSLR_BLASLAPACK_SHSEQR       shseqr
#define PARGEMSLR_BLASLAPACK_DHSEQR       dhseqr
#define PARGEMSLR_BLASLAPACK_CHSEQR       chseqr
#define PARGEMSLR_BLASLAPACK_ZHSEQR       zhseqr

#define PARGEMSLR_BLASLAPACK_SGEQRF       sgeqrf
#define PARGEMSLR_BLASLAPACK_DGEQRF       dgeqrf
#define PARGEMSLR_BLASLAPACK_CGEQRF       cgeqrf
#define PARGEMSLR_BLASLAPACK_ZGEQRF       zgeqrf

#define PARGEMSLR_BLASLAPACK_SHSEIN       shsein
#define PARGEMSLR_BLASLAPACK_DHSEIN       dhsein
#define PARGEMSLR_BLASLAPACK_CHSEIN       chsein
#define PARGEMSLR_BLASLAPACK_ZHSEIN       zhsein

#define PARGEMSLR_BLASLAPACK_STRSEN       strsen
#define PARGEMSLR_BLASLAPACK_DTRSEN       dtrsen
#define PARGEMSLR_BLASLAPACK_CTRSEN       ctrsen
#define PARGEMSLR_BLASLAPACK_ZTRSEN       ztrsen

#define PARGEMSLR_BLASLAPACK_STREXC       strexc
#define PARGEMSLR_BLASLAPACK_DTREXC       dtrexc
#define PARGEMSLR_BLASLAPACK_CTREXC       ctrexc
#define PARGEMSLR_BLASLAPACK_ZTREXC       ztrexc

#else

#ifdef PARGEMSLR_CBLAS

#include "cblas.h"

/* not yet supported */

#else

#define PARGEMSLR_BLASLAPACK_SAXPY        saxpy_
#define PARGEMSLR_BLASLAPACK_DAXPY        daxpy_
#define PARGEMSLR_BLASLAPACK_CAXPY        caxpy_
#define PARGEMSLR_BLASLAPACK_ZAXPY        zaxpy_

#define PARGEMSLR_BLASLAPACK_SSCAL        sscal_
#define PARGEMSLR_BLASLAPACK_DSCAL        dscal_
#define PARGEMSLR_BLASLAPACK_CSCAL        cscal_
#define PARGEMSLR_BLASLAPACK_ZSCAL        zscal_

#define PARGEMSLR_BLASLAPACK_SDOT         sdot_
#define PARGEMSLR_BLASLAPACK_DDOT         ddot_
#define PARGEMSLR_BLASLAPACK_CDOTC        cdotc_
#define PARGEMSLR_BLASLAPACK_ZDOTC        zdotc_

#define PARGEMSLR_BLASLAPACK_SGEMV        sgemv_
#define PARGEMSLR_BLASLAPACK_DGEMV        dgemv_
#define PARGEMSLR_BLASLAPACK_CGEMV        cgemv_
#define PARGEMSLR_BLASLAPACK_ZGEMV        zgemv_

#define PARGEMSLR_BLASLAPACK_SGEMM        sgemm_
#define PARGEMSLR_BLASLAPACK_DGEMM        dgemm_
#define PARGEMSLR_BLASLAPACK_CGEMM        cgemm_
#define PARGEMSLR_BLASLAPACK_ZGEMM        zgemm_

#define PARGEMSLR_BLASLAPACK_STRTRI       strtri_
#define PARGEMSLR_BLASLAPACK_DTRTRI       dtrtri_
#define PARGEMSLR_BLASLAPACK_CTRTRI       ctrtri_
#define PARGEMSLR_BLASLAPACK_ZTRTRI       ztrtri_

#define PARGEMSLR_BLASLAPACK_SGETRF       sgetrf_
#define PARGEMSLR_BLASLAPACK_DGETRF       dgetrf_
#define PARGEMSLR_BLASLAPACK_CGETRF       cgetrf_
#define PARGEMSLR_BLASLAPACK_ZGETRF       zgetrf_

#define PARGEMSLR_BLASLAPACK_SGETRI       sgetri_
#define PARGEMSLR_BLASLAPACK_DGETRI       dgetri_
#define PARGEMSLR_BLASLAPACK_CGETRI       cgetri_
#define PARGEMSLR_BLASLAPACK_ZGETRI       zgetri_

#define PARGEMSLR_BLASLAPACK_SGEHRD       sgehrd_
#define PARGEMSLR_BLASLAPACK_DGEHRD       dgehrd_
#define PARGEMSLR_BLASLAPACK_CGEHRD       cgehrd_
#define PARGEMSLR_BLASLAPACK_ZGEHRD       zgehrd_

#define PARGEMSLR_BLASLAPACK_SORGQR       sorgqr_
#define PARGEMSLR_BLASLAPACK_DORGQR       dorgqr_
#define PARGEMSLR_BLASLAPACK_CUNGQR       cungqr_
#define PARGEMSLR_BLASLAPACK_ZUNGQR       zungqr_

#define PARGEMSLR_BLASLAPACK_SHSEQR       shseqr_
#define PARGEMSLR_BLASLAPACK_DHSEQR       dhseqr_
#define PARGEMSLR_BLASLAPACK_CHSEQR       chseqr_
#define PARGEMSLR_BLASLAPACK_ZHSEQR       zhseqr_

#define PARGEMSLR_BLASLAPACK_SGEQRF       sgeqrf_
#define PARGEMSLR_BLASLAPACK_DGEQRF       dgeqrf_
#define PARGEMSLR_BLASLAPACK_CGEQRF       cgeqrf_
#define PARGEMSLR_BLASLAPACK_ZGEQRF       zgeqrf_

#define PARGEMSLR_BLASLAPACK_SHSEIN       shsein_
#define PARGEMSLR_BLASLAPACK_DHSEIN       dhsein_
#define PARGEMSLR_BLASLAPACK_CHSEIN       chsein_
#define PARGEMSLR_BLASLAPACK_ZHSEIN       zhsein_

#define PARGEMSLR_BLASLAPACK_STRSEN       strsen_
#define PARGEMSLR_BLASLAPACK_DTRSEN       dtrsen_
#define PARGEMSLR_BLASLAPACK_CTRSEN       ctrsen_
#define PARGEMSLR_BLASLAPACK_ZTRSEN       ztrsen_

#define PARGEMSLR_BLASLAPACK_STREXC       strexc_
#define PARGEMSLR_BLASLAPACK_DTREXC       dtrexc_
#define PARGEMSLR_BLASLAPACK_CTREXC       ctrexc_
#define PARGEMSLR_BLASLAPACK_ZTREXC       ztrexc_

#endif

#endif

#ifdef PARGEMSLR_PARPACK
#define PARGEMSLR_PARPACK_PSNAUPD         psnaupd_
#define PARGEMSLR_PARPACK_PDNAUPD         pdnaupd_
#define PARGEMSLR_PARPACK_PCNAUPD         pcnaupd_
#define PARGEMSLR_PARPACK_PZNAUPD         pznaupd_
#define PARGEMSLR_PARPACK_PSNEUPD         psneupd_
#define PARGEMSLR_PARPACK_PDNEUPD         pdneupd_
#define PARGEMSLR_PARPACK_PCNEUPD         pcneupd_
#define PARGEMSLR_PARPACK_PZNEUPD         pzneupd_
#endif

namespace pargemslr
{
   /* note that those functions can be call directly without namespace */
   extern "C" 
   {
      
#ifndef PARGEMSLR_MKL

      void PARGEMSLR_BLASLAPACK_SAXPY(int *n, const float *alpha, const float *x, int *incx, float *y, int *incy);
      void PARGEMSLR_BLASLAPACK_DAXPY(int *n, const double *alpha, const double *x, int *incx, double *y, int *incy);
      void PARGEMSLR_BLASLAPACK_CAXPY(int *n, const ccomplexs *alpha, const ccomplexs *x, int *incx, ccomplexs *y, int *incy);
      void PARGEMSLR_BLASLAPACK_ZAXPY(int *n, const ccomplexd *alpha, const ccomplexd *x, int *incx, ccomplexd *y, int *incy);
   
      void PARGEMSLR_BLASLAPACK_SSCAL(int *n, const float *alpha, float *x, int *incx);
      void PARGEMSLR_BLASLAPACK_DSCAL(int *n, const double *alpha, double *x, int *incx);
      void PARGEMSLR_BLASLAPACK_CSCAL(int *n, const ccomplexs *alpha, ccomplexs *x, int *incx);
      void PARGEMSLR_BLASLAPACK_ZSCAL(int *n, const ccomplexd *alpha, ccomplexd *x, int *incx);
      
      float PARGEMSLR_BLASLAPACK_SDOT(int *n, const float *x, int *incx, const float *y, int *incy);
      double PARGEMSLR_BLASLAPACK_DDOT(int *n, const double *x, int *incx, const double *y, int *incy);
      
#ifdef __APPLE__
      void PARGEMSLR_BLASLAPACK_CDOTC(ccomplexs *t, int *n, const ccomplexs *x, int *incx, const ccomplexs *y, int *incy);//x^H*y
      void PARGEMSLR_BLASLAPACK_ZDOTC(ccomplexd *t, int *n, const ccomplexd *x, int *incx, const ccomplexd *y, int *incy);//x^H*y
#else
      ccomplexs PARGEMSLR_BLASLAPACK_CDOTC( int *n, const ccomplexs *x, int *incx, const ccomplexs *y, int *incy);//x^H*y
      ccomplexd PARGEMSLR_BLASLAPACK_ZDOTC( int *n, const ccomplexd *x, int *incx, const ccomplexd *y, int *incy);//x^H*y
#endif
      
      void PARGEMSLR_BLASLAPACK_SGEMV(char *trans, int *m, int *n, const float *alpha, const float *a, int *lda, const float *x, int *incx, const float *beta, float *y, int *incy);
      void PARGEMSLR_BLASLAPACK_DGEMV(char *trans, int *m, int *n, const double *alpha, const double *a, int *lda, const double *x, int *incx, const double *beta, double *y, int *incy);
      void PARGEMSLR_BLASLAPACK_CGEMV(char *trans, int *m, int *n, const ccomplexs *alpha, const ccomplexs *a, int *lda, const ccomplexs *x, int *incx, const ccomplexs *beta, ccomplexs *y, int *incy);
      void PARGEMSLR_BLASLAPACK_ZGEMV(char *trans, int *m, int *n, const ccomplexd *alpha, const ccomplexd *a, int *lda, const ccomplexd *x, int *incx, const ccomplexd *beta, ccomplexd *y, int *incy);
      
      void PARGEMSLR_BLASLAPACK_SGEMM(char *transa, char *transb, int *m, int *n, int *k, const float *alpha, const float *a, int *lda, const float *b, int *ldb, const float *beta, float *c, int *ldc);
      void PARGEMSLR_BLASLAPACK_DGEMM(char *transa, char *transb, int *m, int *n, int *k, const double *alpha, const double *a, int *lda, const double *b, int *ldb, const double *beta, double *c, int *ldc);
      void PARGEMSLR_BLASLAPACK_CGEMM(char *transa, char *transb, int *m, int *n, int *k, const ccomplexs *alpha, const ccomplexs *a, int *lda, const ccomplexs *b, int *ldb, const ccomplexs *beta, const ccomplexs *c, int *ldc);
      void PARGEMSLR_BLASLAPACK_ZGEMM(char *transa, char *transb, int *m, int *n, int *k, const ccomplexd *alpha, const ccomplexd *a, int *lda, const ccomplexd *b, int *ldb, const ccomplexd *beta, const ccomplexd *c, int *ldc);
      
      void PARGEMSLR_BLASLAPACK_SGETRF(int *m, int *n, float *a, int *lda, int *ipiv, int *info);	
      void PARGEMSLR_BLASLAPACK_DGETRF(int *m, int *n, double *a, int *lda, int *ipiv, int *info);	
      void PARGEMSLR_BLASLAPACK_CGETRF(int *m, int *n, ccomplexs *a, int *lda, int *ipiv, int *info);
      void PARGEMSLR_BLASLAPACK_ZGETRF(int *m, int *n, ccomplexd *a, int *lda, int *ipiv, int *info);
      
      void PARGEMSLR_BLASLAPACK_SGETRI(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info);	
      void PARGEMSLR_BLASLAPACK_DGETRI(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);	
      void PARGEMSLR_BLASLAPACK_CGETRI(int *n, ccomplexs *a, int *lda, int *ipiv, ccomplexs *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_ZGETRI(int *n, ccomplexd *a, int *lda, int *ipiv, ccomplexd *work, int *lwork, int *info);
      
      void PARGEMSLR_BLASLAPACK_STRTRI(char *uplo, char *diag, int *n, float *a, int *lda, int *info);
      void PARGEMSLR_BLASLAPACK_DTRTRI(char *uplo, char *diag, int *n, double *a, int *lda, int *info);
      void PARGEMSLR_BLASLAPACK_CTRTRI(char *uplo, char *diag, int *N, ccomplexs *a, int *lda, int *info);	
      void PARGEMSLR_BLASLAPACK_ZTRTRI(char *uplo, char *diag, int *N, ccomplexd *a, int *lda, int *info);	
      
      void PARGEMSLR_BLASLAPACK_SGEHRD(int *n, int *ilo, int *ihi, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_DGEHRD(int *n, int *ilo, int *ihi, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_CGEHRD(int *n, int *ilo, int *ihi, ccomplexs *a, int *lda, ccomplexs *tau, ccomplexs *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_ZGEHRD(int *n, int *ilo, int *ihi, ccomplexd *a, int *lda, ccomplexd *tau, ccomplexd *work, int *lwork, int *info);
      
      void PARGEMSLR_BLASLAPACK_SORGQR(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_DORGQR(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_CUNGQR(int *m, int *n, int *k, ccomplexs *a, int *lda, ccomplexs *tau, ccomplexs *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_ZUNGQR(int *m, int *n, int *k, ccomplexd *a, int *lda, ccomplexd *tau, ccomplexd *work, int *lwork, int *info);
   
      void PARGEMSLR_BLASLAPACK_SHSEQR(char *job, char *compz, int *n, int *ilo, int *ihi, float *h, int *ldh, float *wr, float *wi, float *z, int *ldz, float *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_DHSEQR(char *job, char *compz, int *n, int *ilo, int *ihi, double *h, int *ldh, double *wr, double *wi, double *z, int *ldz, double *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_CHSEQR(char *job, char *compz, int *n, int *ilo, int *ihi, ccomplexs *h, int *ldh, ccomplexs *w, ccomplexs *z, int *ldz, ccomplexs *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_ZHSEQR(char *job, char *compz, int *n, int *ilo, int *ihi, ccomplexd *h, int *ldh, ccomplexd *w, ccomplexd *z, int *ldz, ccomplexd *work, int *lwork, int *info);
      
      void PARGEMSLR_BLASLAPACK_SGEQRF(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_DGEQRF(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_CGEQRF(int *m, int *n, ccomplexs *a, int *lda, ccomplexs *tau, ccomplexs *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_ZGEQRF(int *m, int *n, ccomplexd *a, int *lda, ccomplexd *tau, ccomplexd *work, int *lwork, int *info);
      
      void PARGEMSLR_BLASLAPACK_SHSEIN(char *side, char *eigsrc, char *initv, int *select, int *n, float *h, int *ldh, float *wr, float *wi, float *vl, int* ldvl, float *vr, int *ldvr, int *mm, int *m, float *work, int* ifaill, int* ifailr, int* info );
      void PARGEMSLR_BLASLAPACK_DHSEIN(char *side, char *eigsrc, char *initv, int *select, int *n, double *h, int *ldh, double *wr, double *wi, double *vl, int* ldvl, double *vr, int *ldvr, int *mm, int *m, double *work, int* ifaill, int* ifailr, int* info );
      void PARGEMSLR_BLASLAPACK_CHSEIN(char *side, char *eigsrc, char *initv, int *select, int *n, ccomplexs *h, int *ldh, ccomplexs *w, ccomplexs *vl, int* ldvl, ccomplexs *vr, int *ldvr, int *mm, int *m, ccomplexs *work, float *rwork, int* ifaill, int* ifailr, int* info);
      void PARGEMSLR_BLASLAPACK_ZHSEIN(char *side, char *eigsrc, char *initv, int *select, int *n, ccomplexd *h, int *ldh, ccomplexd *w, ccomplexd *vl, int* ldvl, ccomplexd *vr, int *ldvr, int *mm, int *m, ccomplexd *work, double *rwork, int* ifaill, int* ifailr, int* info);
      
      void PARGEMSLR_BLASLAPACK_STRSEN(char *job, char *compq, int *select, int *n, float *t, int *ldt, float *q, int *ldq, float *wr, float *wi, int *m, float *s, float *sep, float *work, int *lwork, int *iwork, int *liwork, int *info);
      void PARGEMSLR_BLASLAPACK_DTRSEN(char *job, char *compq, int *select, int *n, double *t, int *ldt, double *q, int *ldq, double *wr, double *wi, int *m, double *s, double *sep, double *work, int *lwork, int *iwork, int *liwork, int *info);
      void PARGEMSLR_BLASLAPACK_CTRSEN(char *job, char *compq, int *select, int *n, ccomplexs *t, int *ldt, ccomplexs *q, int *ldq, ccomplexs *w, int *m, float *s, float *sep, ccomplexs *work, int *lwork, int *info);
      void PARGEMSLR_BLASLAPACK_ZTRSEN(char *job, char *compq, int *select, int *n, ccomplexd *t, int *ldt, ccomplexd *q, int *ldq, ccomplexd *w, int *m, double *s, double *sep, ccomplexd *work, int *lwork, int *info);
      
      void PARGEMSLR_BLASLAPACK_STREXC(char *compq, int *n, float *t, int *ldt, float *q, int *ldq, int *ifst, int *ilst, float *work, int *info);
      void PARGEMSLR_BLASLAPACK_DTREXC(char *compq, int *n, double *t, int *ldt, double *q, int *ldq, int *ifst, int *ilst, double *work, int *info);
      void PARGEMSLR_BLASLAPACK_CTREXC(char *compq, int *n, ccomplexs *t, int *ldt, ccomplexs *q, int *ldq, int *ifst, int *ilst, int *info);
      void PARGEMSLR_BLASLAPACK_ZTREXC(char *compq, int *n, ccomplexd *t, int *ldt, ccomplexd *q, int *ldq, int *ifst, int *ilst, int *info);

#endif

      int METIS_PartGraphRecursive(pargemslr_long *nvtxs, pargemslr_long *ncon, pargemslr_long *xadj, pargemslr_long *adjncy, pargemslr_long *vwgt, pargemslr_long *vsize, pargemslr_long *adjwgt, pargemslr_long *nparts, float *tpwgts, float *ubvec, pargemslr_long *options, pargemslr_long *edgecut, pargemslr_long *part);
      int METIS_PartGraphKway(pargemslr_long *nvtxs, pargemslr_long *ncon, pargemslr_long *xadj, pargemslr_long *adjncy, pargemslr_long *vwgt, pargemslr_long *vsize, pargemslr_long *adjwgt, pargemslr_long *nparts, float *tpwgts, float *ubvec, pargemslr_long *options, pargemslr_long *edgecut, pargemslr_long *part);
      int METIS_NodeND(pargemslr_long *nvtxs, pargemslr_long *xadj, pargemslr_long *adjncy, pargemslr_long *vwgt, pargemslr_long *options, pargemslr_long *perm, pargemslr_long *iperm);
      int ParMETIS_V3_PartKway(pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, float*, float*, pargemslr_long*, pargemslr_long*, pargemslr_long*, MPI_Comm* );
		int ParMETIS_V3_RefineKway(pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, float*, float*, pargemslr_long*, pargemslr_long*, pargemslr_long*, MPI_Comm*);
      int ParMETIS_V3_NodeND(pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*, pargemslr_long*,pargemslr_long*, MPI_Comm*);
      
      /* AMD order for experiments only */
      //int amd_order(int, const int*, const int*, int*, double*, double*);
      
#ifdef PARGEMSLR_OPENBLAS
      void openblas_set_num_threads(int);
#endif      
      
#ifdef PARGEMSLR_PARPACK
      void PARGEMSLR_PARPACK_PSNAUPD( int *comm, int* ido, char* bmat, int* n, char* which, int* nev, float* tol, float* resid, int* ncv, float* v, int* ldv, int* iparam, int* ipntr, float* workd, float* workl, int* lworkl, int* info);
      void PARGEMSLR_PARPACK_PSNEUPD( int *comm, int* rvec, char* howmny, int* select, float* dr, float* di, float* z, int* ldz, float* sigmar, float* sigmai, float* workev, char* bmat, int* n, char* which, int* nev, float* tol, float* resid, int* ncv, float* v, int* ldv, int* iparam, int* ipntr, float* workd, float* workl, int* lworkl, int* info);
      void PARGEMSLR_PARPACK_PDNAUPD( int *comm, int* ido, char* bmat, int* n, char* which, int* nev, double* tol, double* resid, int* ncv, double* v, int* ldv, int* iparam, int* ipntr, double* workd, double* workl, int* lworkl, int* info);
      void PARGEMSLR_PARPACK_PDNEUPD( int *comm, int* rvec, char* howmny, int* select, double* dr, double* di, double* z, int* ldz, double* sigmar, double* sigmai, double* workev, char* bmat, int* n, char* which, int* nev, double* tol, double* resid, int* ncv, double* v, int* ldv, int* iparam, int* ipntr, double* workd, double* workl, int* lworkl, int* info);
      void PARGEMSLR_PARPACK_PCNAUPD( int *comm, int* ido, char* bmat, int* n, char* which, int* nev, float* tol, ccomplexs* resid, int* ncv, ccomplexs* v, int* ldv, int* iparam, int* ipntr, ccomplexs* workd, ccomplexs* workl, int* lworkl, float* rwork, int* info);
      void PARGEMSLR_PARPACK_PCNEUPD( int *comm, int* rvec, char* howmny, int* select, ccomplexs* d, ccomplexs* z, int* ldz, ccomplexs* sigma, ccomplexs* workev, char* bmat, int* n, char* which, int* nev, float* tol, ccomplexs* resid, int* ncv, ccomplexs* v, int* ldv, int* iparam, int* ipntr, ccomplexs* workd, ccomplexs* workl, int* lworkl, float* rwork, int* info);
      void PARGEMSLR_PARPACK_PZNAUPD( int *comm, int* ido, char* bmat, int* n, char* which, int* nev, double* tol, ccomplexd* resid, int* ncv, ccomplexd* v, int* ldv, int* iparam, int* ipntr, ccomplexd* workd, ccomplexd* workl, int* lworkl, double* rwork, int* info);
      void PARGEMSLR_PARPACK_PZNEUPD( int *comm, int* rvec, char* howmny, int* select, ccomplexd* d, ccomplexd* z, int* ldz, ccomplexd* sigma, ccomplexd* workev, char* bmat, int* n, char* which, int* nev, double* tol, ccomplexd* resid, int* ncv, ccomplexd* v, int* ldv, int* iparam, int* ipntr, ccomplexd* workd, ccomplexd* workl, int* lworkl, double* rwork, int* info);
#endif 
   }
   
}

#endif
