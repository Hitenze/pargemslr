
include ../../makefile.in

.SUFFIXES : .c .cpp .o

INCLUDES = -I../../INC 

ifneq ($(USING_MKL),0)
INCLUDES += -I$(MKLROOT)/include
endif

EXTRA = io.o

#OBJ_TEST  = test.o
#OBJ_SEQ = sequential_test.o
OBJ_SEQ_LAP = driver_laplacian_gemslr_seq.o
OBJ_SEQ_LAPZ = driver_laplacian_gemslrz_seq.o
OBJ_SEQ_GEN = driver_gen_gemslr_seq.o
OBJ_SEQ_GENZ = driver_gen_gemslrz_seq.o
OBJ_ALLTESTS = driver_laplacian_gemslr_seq.o\
	driver_laplacian_gemslrz_seq.o\
	driver_gen_gemslr_seq.o\
	driver_gen_gemslrz_seq.o

LIB := -L../../ -lpargemslr -L../../$(METIS_PATH)/libparmetis -lparmetis -L../../$(METIS_PATH)/libmetis -lmetis -L../../$(LAPACK_PATH) $(LIBLAPACK) $(LIBOPENMP) $(LIBMKL) -lgfortran $(LIBCUDA)

ALLEXE = driver_laplacian_gemslr_seq.ex\
	driver_laplacian_gemslrz_seq.ex\
	driver_gen_gemslr_seq.ex\
	driver_gen_gemslrz_seq.ex

# Rules
default: driver_laplacian_gemslr_seq.ex\
	driver_laplacian_gemslrz_seq.ex\
	driver_gen_gemslr_seq.ex\
	driver_gen_gemslrz_seq.ex

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

driver_laplacian_gemslr_seq.ex: $(OBJ_SEQ_LAP) $(EXTRA)
	$(LINK) $(FLAGS) -o driver_laplacian_gemslr_seq.ex $(INCLUDES) $(OBJ_SEQ_LAP) $(EXTRA) $(LIB)
	
driver_laplacian_gemslrz_seq.ex: $(OBJ_SEQ_LAPZ) $(EXTRA)
	$(LINK) $(FLAGS) -o driver_laplacian_gemslrz_seq.ex $(INCLUDES) $(OBJ_SEQ_LAPZ) $(EXTRA) $(LIB)

driver_gen_gemslr_seq.ex: $(OBJ_SEQ_GEN) $(EXTRA)
	$(LINK) $(FLAGS) -o driver_gen_gemslr_seq.ex $(INCLUDES) $(OBJ_SEQ_GEN) $(EXTRA) $(LIB)

driver_gen_gemslrz_seq.ex: $(OBJ_SEQ_GENZ) $(EXTRA)
	$(LINK) $(FLAGS) -o driver_gen_gemslrz_seq.ex $(INCLUDES) $(OBJ_SEQ_GENZ) $(EXTRA) $(LIB)

clean:
	rm -f $(OBJ_ALLTESTS) *.ex *.o
	rm -f TempData/*
