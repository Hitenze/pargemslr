# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

default:all

include ../../makefile.in
include $(HYPRE_PATH)/config/Makefile.config

LINK_CXX +=  -std=c++11
CFLAGS += -I../../INC
LIBPARGEMSLR = -L../../ -lpargemslr -L../../$(LAPACK_PATH) $(LIBLAPACK) -L../../$(METIS_PATH)/libparmetis -lparmetis -L../../$(METIS_PATH)/libmetis -lmetis

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

CDEFS = -DHYPRE_TIMING -DHYPRE_FORTRAN
CXXDEFS = -DNOFEI -DHYPRE_TIMING -DMPICH_SKIP_MPICXX

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_BUILD_DIR}/include\
 $(SUPERLU_INCLUDE)\
 $(DSUPERLU_INCLUDE)\
 ${CINCLUDES}\
 ${CDEFS}

CXX_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../FEI_mv/fei-base\
 -I${HYPRE_BUILD_DIR}/include\
 $(SUPERLU_INCLUDE)\
 $(DSUPERLU_INCLUDE)\
 ${CINCLUDES}\
 ${CXXDEFS}
 
ifneq ($(USING_CUDA),0)
CXX_COMPILE_FLAGS += $(CUFLAGS) -DPARGEMSLR_CUDA -DPARGEMSLR_CUDA_VERSION=10
endif

F77_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}

MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS}
LAPACKLIBFLAGS = ${LAPACKLIBDIRS} ${LAPACKLIBS}
BLASLIBFLAGS = ${BLASLIBDIRS} ${BLASLIBS}
LIBFLAGS = ${LDFLAGS} ${LIBS}

ifeq (${LINK_CC}, nvcc)
   XLINK = -Xlinker=-rpath,${HYPRE_BUILD_DIR}/lib
else
   XLINK = -Wl,-rpath,${HYPRE_BUILD_DIR}/lib
endif

LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib -lHYPRE\
 ${XLINK}\
 ${DSUPERLU_LIBS}\
 ${SUPERLU_LIBS}\
 ${MPILIBFLAGS}\
 ${LAPACKLIBFLAGS}\
 ${BLASLIBFLAGS}\
 ${LIBFLAGS}\
 ${LIBPARGEMSLR}

ifneq ($(USING_CUDA),0)
LFLAGS += -lcublas -lcusparse -lcurand
endif

EXTRA = pargemslr_hypre_interface.o HYPRE_pargemslr.o


##################################################################
# Targets
##################################################################

HYPRE_DRIVERS =\
 ij.cpp

HYPRE_DRIVER_EXECS=${HYPRE_DRIVERS:.cpp=}

all: ${HYPRE_DRIVER_EXECS}

%.o : %.cpp
	$(LINK_CXX) $(CXX_COMPILE_FLAGS) -o $@ -c $<


install:

clean:
	rm -f *.o*
	rm -rf pchdir tca.map *inslog*
	rm -f ij
	rm -f TempData/*

distclean: clean
	rm -f ${HYPRE_DRIVER_EXECS}

##################################################################
# Rules
##################################################################

# C

ij: ij.o $(EXTRA)
	@echo  "Building" $@ "... "
	${LINK_CXX} -o $@ $< $(EXTRA) ${LFLAGS}
