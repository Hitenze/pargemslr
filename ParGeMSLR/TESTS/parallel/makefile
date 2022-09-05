
include ../../makefile.in

.SUFFIXES : .c .cpp .o

INCLUDES = -I../../INC 

ifneq ($(USING_MKL),0)
INCLUDES += -I$(MKLROOT)/include
endif

EXTRA = io_par.o

#OBJ_TEST  = test.o
#OBJ_PAR = parallel_test.o
OBJ_PAR_LAP = driver_laplacian_gemslr_par.o
OBJ_PAR_LAPZ = driver_laplacian_gemslrz_par.o
OBJ_PAR_GEN = driver_gen_gemslr_par.o
OBJ_PAR_GENZ = driver_gen_gemslrz_par.o
OBJ_PAR_CSR = driver_laplacian_gemslr_csr.o
OBJ_ALLTESTS = driver_laplacian_gemslr_par.o\
	driver_laplacian_gemslrz_par.o\
	driver_gen_gemslr_par.o\
	driver_gen_gemslrz_par.o\
	driver_laplacian_gemslr_csr.o

LIB := -L../../ -lpargemslr -L../../$(METIS_PATH)/libparmetis -lparmetis -L../../$(METIS_PATH)/libmetis -lmetis -L../../$(LAPACK_PATH) $(LIBLAPACK) $(LIBOPENMP) $(LIBMKL) -lgfortran $(LIBCUDA)

ALLEXE = driver_laplacian_gemslr_par.ex\
	driver_laplacian_gemslrz_par.ex\
	driver_gen_gemslr_par.ex\
	driver_gen_gemslrz_par.ex\
	driver_laplacian_gemslr_csr.ex

# Rules
default: driver_laplacian_gemslr_par.ex\
	driver_laplacian_gemslrz_par.ex\
	driver_gen_gemslr_par.ex\
	driver_gen_gemslrz_par.ex\
	driver_laplacian_gemslr_csr.ex

all: $(ALLEXE)
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

driver_laplacian_gemslr_par.ex: $(OBJ_PAR_LAP) $(EXTRA)
	$(LINK) $(FLAGS) -o driver_laplacian_gemslr_par.ex $(INCLUDES) $(OBJ_PAR_LAP) $(EXTRA) $(LIB)

driver_laplacian_gemslrz_par.ex: $(OBJ_PAR_LAPZ) $(EXTRA)
	$(LINK) $(FLAGS) -o driver_laplacian_gemslrz_par.ex $(INCLUDES) $(OBJ_PAR_LAPZ) $(EXTRA) $(LIB)

driver_gen_gemslr_par.ex: $(OBJ_PAR_GEN) $(EXTRA)
	$(LINK) $(FLAGS) -o driver_gen_gemslr_par.ex $(INCLUDES) $(OBJ_PAR_GEN) $(EXTRA) $(LIB)

driver_gen_gemslrz_par.ex: $(OBJ_PAR_GENZ) $(EXTRA)
	$(LINK) $(FLAGS) -o driver_gen_gemslrz_par.ex $(INCLUDES) $(OBJ_PAR_GENZ) $(EXTRA) $(LIB)

driver_laplacian_gemslr_csr.ex: $(OBJ_PAR_CSR) $(EXTRA)
	$(LINK) $(FLAGS) -o driver_laplacian_gemslr_csr.ex $(INCLUDES) $(OBJ_PAR_CSR) $(EXTRA) $(LIB)

clean:
	rm -f $(OBJ_ALLTESTS) *.ex *.o
	rm -f TempData/*
