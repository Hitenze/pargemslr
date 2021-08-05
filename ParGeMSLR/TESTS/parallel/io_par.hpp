#include "pargemslr.hpp"
#include <iostream>

using namespace pargemslr;

#define EPSILON 1e-12 

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

/* print the commands to generate Laplacian */
int print_laplacian_usage();

/* print the commands to read general matrix */
int print_gen_usage();

/* print the commands for the laplacian tests */
int print_lap_test_usage();

/* print the commands for the parallellaplacian tests */
int print_parlap_test_usage();

/* print the commands for the general tests */
int print_gen_test_usage();

/* apply some MPI communication first */
int dummy_comm();

/* read laplacian params */
int read_double_laplacian_param(int &nmats, int **nx, int **ny, int **nz, double **shift, double **alphax, double **alphay, double **alphaz, const char *filename, bool fromfile);

/* read laplacian params */
int read_double_complex_laplacian_param(int &nmats, int **nx, int **ny, int **nz, complexd **shift, complexd **alphax, complexd **alphay, complexd **alphaz, const char *filename, bool fromfile);

/* read helmholtz params */
int read_double_complex_helmholtz_param(int &nmats, int **n, complexd **w);

/* read mat params */
int read_matfile(const char *filename, int &nmats, char ***matfile, char ***vecfile);

/* read inputs from command line */
int read_inputs(double *input, int argc, char **argv);

/* read inputs from file */
int read_inputs_from_file(const char *filename, double *params);
