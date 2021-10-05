include makefile.in

.SUFFIXES : .c .cpp .o

INCLUDES = -I./INC 

ifneq ($(USING_MKL),0)
INCLUDES += -I$(MKLROOT)/include
endif

# Object files
EXTRASRC = ./SRC/utils/utils.o\
	./SRC/utils/parallel.o\
	./SRC/utils/memory.o\
	./SRC/utils/mmio.o\
	./SRC/vectors/vector.o\
	./SRC/vectors/sequential_vector.o\
	./SRC/vectors/int_vector.o\
	./SRC/vectors/parallel_vector.o\
	./SRC/vectors/vectorops.o\
	./SRC/matrices/matrix.o\
	./SRC/matrices/arnoldimatrix.o\
	./SRC/matrices/dense_matrix.o\
	./SRC/matrices/coo_matrix.o\
	./SRC/matrices/csr_matrix.o\
	./SRC/matrices/parallel_csr_matrix.o\
	./SRC/matrices/matrixops.o\
	./SRC/preconditioners/ilu.o\
	./SRC/preconditioners/gemslr.o\
	./SRC/preconditioners/poly.o\
	./SRC/preconditioners/block_jacobi.o\
	./SRC/preconditioners/parallel_mix_precond.o\
	./SRC/preconditioners/parallel_gemslr.o

ifneq ($(USING_CUDA),0)
EXTRASRC += ./SRC/utils/utils_cuda.o\
	./SRC/vectors/int_vector_cuda.o\
	./SRC/vectors/sequential_vector_cuda.o\
	./SRC/vectors/vectorops_cuda.o\
	./SRC/matrices/coo_matrix_cuda.o\
	./SRC/matrices/csr_matrix_cuda.o\
	./SRC/matrices/dense_matrix_cuda.o\
	./SRC/matrices/matrixops_cuda.o\
	./SRC/preconditioners/ilu_cuda.o\
	./SRC/preconditioners/gemslr_cuda.o
endif

LIBPARGEMSLR = libpargemslr.a

LIB := -L$(METIS_PATH)/libmetis -lmetis -L$(METIS_PATH)/libparmetis -lparmetis -L$(LAPACK_PATH) $(LIBLAPACK) $(LIBOPENMP) $(LIBMKL) -lgfortran $(LIBCUDA)

#LIB += -L$(CSPARSE_PATH)/Lib -lcsparse
LIB += -L$(AMD_PATH)/Lib -lamd

# Rules
default: $(LIBPARGEMSLR)
all: $(LIBPARGEMSLR)

%.o : %.f
	$(FORT) $(FLAGS) $(INCLUDES) -o $@ -c $<
%.o : %.c
	$(CC) $(FLAGS) $(INCLUDES) -o $@ -c $<
%.o : %.cpp
	$(CXX) $(FLAGS) $(INCLUDES) -o $@ -c $<
ifneq ($(USING_CUDA),0)
%.o : %.cu
	$(NVCC) $(FLAGS) $(INCLUDES) -o $@ -c $<
endif

# Exe

$(LIBPARGEMSLR): $(EXTRASRC)
	$(AR) $@ $(EXTRASRC)
	$(RANLIB) $@

clean:
	rm -f $(LIBPARGEMSLR) $(EXTRASRC) $(CSRC)
