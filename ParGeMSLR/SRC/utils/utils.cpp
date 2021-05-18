
#include "utils.hpp"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <limits.h>

namespace pargemslr
{
   
   double pargemslr_global::_expand_fact = 1.3;
   int pargemslr_global::_coo_reserve_fact = 7;
   int pargemslr_global::_dash_line_width = 32;
   int pargemslr_global::_openmp_min_loopsize = 12;
   int pargemslr_global::_metis_refine = 0;
   double pargemslr_global::_metis_loading_balance_tol = 0.2;
   int pargemslr_global::_minsep = 10;
   double pargemslr_global::_tr_factor = 0.25;
   double pargemslr_global::_orth_tol = 1e-14;
   double pargemslr_global::_reorth_tol = 1.0/sqrt(2.0);
   std::random_device pargemslr_global::_random_device;
   std::mt19937 pargemslr_global::_mersenne_twister_engine(pargemslr_global::_random_device());
   std::uniform_int_distribution<int> pargemslr_global::_uniform_int_distribution(0, INT_MAX);
   FILE* pargemslr_global::_out_file = stdout;
   
   static double params_buff[PARGEMSLR_IO_SIZE];
   double* pargemslr_global::_params = PargemslrSetDefaultParameterArray(params_buff);
   
   int pargemslr_global::_gram_schmidt = 0;
   
   double* PargemslrSetDefaultParameterArray(double *params)
   {
      for(int i = 0 ; i < PARGEMSLR_IO_SIZE ; i ++)
      {
         params[i] = 0;
      }
      params[PARGEMSLR_IO_PRECOND_GLOBAL_PRECOND] = 0;
      params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1] = 0;
      params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND1_LEVEL] = 1;
      params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND2] = 0;
      params[PARGEMSLR_IO_PRECOND_LOCAL_REPCOND3] = 0;
      
      params[PARGEMSLR_IO_SOLVER_OPTION] = 0;
      params[PARGEMSLR_IO_SOLVER_KDIM] = 50;
      params[PARGEMSLR_IO_SOLVER_MAXITS] = 100;
      params[PARGEMSLR_IO_SOLVER_TOL] = 1e-06;
      
      params[PARGEMSLR_IO_PREPOSS_NLEV_GLOBAL] = 5;
      params[PARGEMSLR_IO_PREPOSS_NLEV_LOCAL] = 5;
      params[PARGEMSLR_IO_PREPOSS_NCOMP_GLOBAL] = 4;
      params[PARGEMSLR_IO_PREPOSS_NCOMP_LOCAL] = 4;
      params[PARGEMSLR_IO_PREPOSS_PARTITION_GLOBAL] = 1;
      params[PARGEMSLR_IO_PREPOSS_PARTITION_LOCAL] = 0;
      params[PARGEMSLR_IO_PREPOSS_MINSEP] = 10;
      params[PARGEMSLR_IO_PREPOSS_KMIN_GLOBAL] = 1;
      params[PARGEMSLR_IO_PREPOSS_KMIN_LOCAL] = 1;
      params[PARGEMSLR_IO_PREPOSS_KFACTOR_GLOBAL] = 1;
      params[PARGEMSLR_IO_PREPOSS_KFACTOR_LOCAL] = 1;
      params[PARGEMSLR_IO_PREPOSS_VTXSEP_GLOBAL] = 1;
      params[PARGEMSLR_IO_PREPOSS_VTXSEP_LOCAL] = 1;
      params[PARGEMSLR_IO_PREPOSS_METIS_REFINE] = 0;
      
      params[PARGEMSLR_IO_ILU_DROPTOL_B_GLOBAL] = 1e-02;
      params[PARGEMSLR_IO_ILU_DROPTOL_B_LOCAL] = 1e-02;
      params[PARGEMSLR_IO_ILU_DROPTOL_EF_GLOBAL] = 1e-02;
      params[PARGEMSLR_IO_ILU_DROPTOL_EF_LOCAL] = 1e-02;
      params[PARGEMSLR_IO_ILU_DROPTOL_C_GLOBAL] = 1e-03;
      params[PARGEMSLR_IO_ILU_DROPTOL_C_LOCAL] = 1e-03;
      params[PARGEMSLR_IO_ILU_DROPTOL_S_GLOBAL] = 1e-03;
      params[PARGEMSLR_IO_ILU_DROPTOL_S_LOCAL] = 1e-03;
      params[PARGEMSLR_IO_ILU_ROWNNZ_B_GLOBAL] = 100;
      params[PARGEMSLR_IO_ILU_ROWNNZ_B_LOCAL] = 100;
      params[PARGEMSLR_IO_ILU_ROWNNZ_C_GLOBAL] = 200;
      params[PARGEMSLR_IO_ILU_ROWNNZ_C_LOCAL] = 200;
      params[PARGEMSLR_IO_ILU_ROWNNZ_S_GLOBAL] = 200;
      params[PARGEMSLR_IO_ILU_ROWNNZ_S_LOCAL] = 200;
      params[PARGEMSLR_IO_ILU_PERM_OPTION_GLOBAL] = 2;
      params[PARGEMSLR_IO_ILU_PERM_OPTION_LOCAL] = 2;
      params[PARGEMSLR_IO_ILU_OMP_OPTION_GLOBAL] = 0;
      params[PARGEMSLR_IO_ILU_OMP_OPTION_LOCAL] = 0;
      
      params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_GLOBAL] = 1;
      params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_GLOBAL] = 1;
      params[PARGEMSLR_IO_LR_ARNOLDI_OPTION1_LOCAL] = 1;
      params[PARGEMSLR_IO_LR_ARNOLDI_OPTION2_LOCAL] = 1;
      params[PARGEMSLR_IO_LR_RANK1_GLOBAL] = 100;
      params[PARGEMSLR_IO_LR_RANK1_LOCAL] = 100;
      params[PARGEMSLR_IO_LR_RANK2_GLOBAL] = 50;
      params[PARGEMSLR_IO_LR_RANK2_LOCAL] = 50;
      params[PARGEMSLR_IO_LR_RANK_FACTOR1_GLOBAL] = 1.0;
      params[PARGEMSLR_IO_LR_RANK_FACTOR1_LOCAL] = 1.0;
      params[PARGEMSLR_IO_LR_RANK_FACTOR2_GLOBAL] = 1.0;
      params[PARGEMSLR_IO_LR_RANK_FACTOR2_LOCAL] = 1.0;
      params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_GLOBAL] = 2.0;
      params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR1_LOCAL] = 2.0;
      params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_GLOBAL] = 2.0;
      params[PARGEMSLR_IO_LR_ARNOLDI_FACTOR2_LOCAL] = 2.0;
      params[PARGEMSLR_IO_LR_TOL_EIG1_GLOBAL] = 1e-09;
      params[PARGEMSLR_IO_LR_TOL_EIG1_LOCAL] = 1e-09;
      params[PARGEMSLR_IO_LR_TOL_EIG2_GLOBAL] = 1e-09;
      params[PARGEMSLR_IO_LR_TOL_EIG2_LOCAL] = 1e-09;
      params[PARGEMSLR_IO_LR_MAXITS1_GLOBAL] = 3;
      params[PARGEMSLR_IO_LR_MAXITS1_LOCAL] = 3;
      params[PARGEMSLR_IO_LR_MAXITS2_GLOBAL] = 3;
      params[PARGEMSLR_IO_LR_MAXITS2_LOCAL] = 3;
      params[PARGEMSLR_IO_LR_TR_FACTOR] = 0.25;
      params[PARGEMSLR_IO_LR_TOL_ORTH] = 1e-14;
      params[PARGEMSLR_IO_LR_TOL_REORTH] = 1/sqrt(2.0);
      params[PARGEMSLR_IO_LR_RAND_INIT_GUESS] = 1;
      
      params[PARGEMSLR_IO_SCHUR_ENABLE] = 1;
      params[PARGEMSLR_IO_SCHUR_ITER_TOL] = 1e-02;
      params[PARGEMSLR_IO_SCHUR_MAXITS] = 5;
      
      params[PARGEMSLR_IO_ADVANCED_DIAG_SHIFT] = 0.0;
      params[PARGEMSLR_IO_ADVANCED_GLOBAL_SOLVE] = 0;
      params[PARGEMSLR_IO_ADVANCED_USE_COMPLEX_SHIFT] = 0;
      params[PARGEMSLR_IO_ADVANCED_RESIDUAL_ITERS] = 1;
      params[PARGEMSLR_IO_ADVANCED_GRAM_SCHMIDT] = 0;
      
      params[PARGEMSLR_IO_GENERAL_PRINT_LEVEL] = 1;
      
      params[PARGEMSLR_IO_POLY_ORDER] = 3;
      
      return params;
      
   }
   
   template <typename T>
   bool operator<(const CompareStruct<T> &a, const CompareStruct<T> &b)
   {
      return a.val < b.val;
   }
   template bool operator<(const CompareStruct<int> &a, const CompareStruct<int> &b);
   template bool operator<(const CompareStruct<long int> &a, const CompareStruct<long int> &b);
   
   template <typename T>
   T PargemslrMax(T a, T b)
   {
      return a >= b ? a : b;
   }
   template int PargemslrMax(int a, int b);
   template long int PargemslrMax(long int a, long int b);
   template float PargemslrMax(float a, float b);
   template double PargemslrMax(double a, double b);
   
   template <typename T>
   T PargemslrMin(T a, T b)
   {
      return a <= b ? a : b;
   }
   template int PargemslrMin(int a, int b);
   template long int PargemslrMin(long int a, long int b);
   template float PargemslrMin(float a, float b);
   template double PargemslrMin(double a, double b);
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, T>::type
   PargemslrAbs( const T &a)
   {
      return std::abs(a);
   }
   template int PargemslrAbs( const int &a);
   template long int PargemslrAbs( const long int &a);
   template float PargemslrAbs( const float &a);
   template double PargemslrAbs( const double &a);
   
   float PargemslrAbs( const complexs &a)
   {
      return a.Abs();
   }
   
   double PargemslrAbs( const complexd &a)
   {
      return a.Abs();
   }
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, T>::type
   PargemslrReal( const T &a)
   {
      return a;
   }
   template int PargemslrReal( const int &a);
   template long int PargemslrReal( const long int &a);
   template float PargemslrReal( const float &a);
   template double PargemslrReal( const double &a);
   
   float PargemslrReal( const complexs &a)
   {
      return a.Real();
   }
   
   double PargemslrReal( const complexd &a)
   {
      return a.Real();
   }
   
   template <typename T>
   typename std::enable_if<!PargemslrIsComplex<T>::value, T>::type
   PargemslrConj( const T &a)
   {
      return a;
   }
   template int PargemslrConj( const int &a);
   template long int PargemslrConj( const long int &a);
   template float PargemslrConj( const float &a);
   template double PargemslrConj( const double &a);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, T>::type
   PargemslrConj( const T &a)
   {
      return a.Conj();
   }
   template complexs PargemslrConj( const complexs &a);
   template complexd PargemslrConj( const complexd &a);
   
   template <typename T>
   typename std::enable_if<PargemslrIsInteger<T>::value, int>::type
   PargemslrValueRandHost(T &a)
   {
      a = (T)(pargemslr_global::_uniform_int_distribution(pargemslr_global::_mersenne_twister_engine));
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrValueRandHost(int &a);
   template int PargemslrValueRandHost(long int &a);
   
   template <typename T>
   typename std::enable_if<PargemslrIsReal<T>::value, int>::type
   PargemslrValueRandHost(T &a)
   {
      a = (pargemslr_global::_uniform_int_distribution(pargemslr_global::_mersenne_twister_engine))/((T)pargemslr_global::_uniform_int_distribution.max());
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrValueRandHost(float &a);
   template int PargemslrValueRandHost(double &a);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrValueRandHost(T &a)
   {
      a = T((pargemslr_global::_uniform_int_distribution(pargemslr_global::_mersenne_twister_engine))/((double)pargemslr_global::_uniform_int_distribution.max()),(pargemslr_global::_uniform_int_distribution(pargemslr_global::_mersenne_twister_engine))/((double)pargemslr_global::_uniform_int_distribution.max()));
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrValueRandHost(complexs &a);
   template int PargemslrValueRandHost(complexd &a);
   
   int PargemslrPrintSpace(int width)
   {
      int i;
      for(i = 0 ; i < width ; i ++)
      {
         PARGEMSLR_PRINT(" ");
      }
      return PARGEMSLR_SUCCESS;
   }
   
   int PargemslrPrintDashLine(int width)
   {
      int i;
      for(i = 0 ; i < width ; i ++)
      {
         PARGEMSLR_PRINT("-");
      }
      PARGEMSLR_PRINT("\n");
      return PARGEMSLR_SUCCESS;
   }
   
   template <typename T>
   typename std::enable_if<PargemslrIsInteger<T>::value, int>::type 
   PargemslrPrintValueHost(T val, int width)
   {
      long int vall = (long int) val;
      if(vall < 0)
      {
         PARGEMSLR_PRINT("-%*ld ",width,-vall);
      }
      else
      {
         PARGEMSLR_PRINT("+%*ld ",width,vall);
      }
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrPrintValueHost(int val, int width);
   template int PargemslrPrintValueHost(long int val, int width);
   
   template <typename T>
   typename std::enable_if<PargemslrIsReal<T>::value, int>::type 
   PargemslrPrintValueHost(T val, int width)
   {
      if(val < 0)
      {
         PARGEMSLR_PRINT("-%*.*f",width,width-2,PargemslrAbs(val));
      }
      else
      {
         PARGEMSLR_PRINT("+%*.*f",width,width-2,PargemslrAbs(val));
      }
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrPrintValueHost(float val, int width);
   template int PargemslrPrintValueHost(double val, int width);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type 
   PargemslrPrintValueHost(T val, int width)
   {
      double val_r, val_i;
      val_r = val.Real();
      val_i = val.Imag();
      
      if(val_r < 0)
      {
         PARGEMSLR_PRINT("-%*.*f",width,width-2,PargemslrAbs(val_r));
      }
      else
      {
         PARGEMSLR_PRINT("+%*.*f",width,width-2,PargemslrAbs(val_r));
      }
      if(val_i < 0)
      {
         PARGEMSLR_PRINT("-%*.*f",width,width-2,PargemslrAbs(val_i));
      }
      else
      {
         PARGEMSLR_PRINT("+%*.*f",width,width-2,PargemslrAbs(val_i));
      }
      PARGEMSLR_PRINT("i");
      
      return PARGEMSLR_SUCCESS;
   }
   template int PargemslrPrintValueHost(ComplexValueClass<float> val, int width);
   template int PargemslrPrintValueHost(ComplexValueClass<double> val, int width);
   
   void PargemslrReadFirstWord(char *pin, char **pout)
   {
      char *p_read, *p_write;
      p_read = pin;
      
      /* locate the start of the first word */                                
      while (' ' == *p_read) 
      {                        
         p_read++;                                     
      }                                    
      /* allocate the return buffer */
      *pout = (char*)malloc(sizeof(char)*(strlen(p_read)+1));
      p_write = *pout;                                   
      while (' ' != *p_read) 
      {
         /* convert to upper */           
         *p_write = toupper(*p_read);                       
         p_write++;                                     
         p_read++;                                    
      }
      /* only keep the first word */                     
      *p_write = '\0';
   }
   
   int PargemslrReadInputArg(const char *argname, int amount, float *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = atof(argv[i+j+1]);
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int PargemslrReadInputArg(const char *argname, int amount, double *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = atof(argv[i+j+1]);
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int PargemslrReadInputArg(const char *argname, int amount, complexs *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = complexs(atof(argv[i+2*j+1]),atof(argv[i+2*j+2]));
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int PargemslrReadInputArg(const char *argname, int amount, complexd *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = complexd(atof(argv[i+2*j+1]),atof(argv[i+2*j+2]));
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int PargemslrReadInputArg(const char *argname, int amount, int *val, int argc, char **argv)
   {
      int i, j;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            for(j = 0 ; j < amount ; j ++)
            {
               val[j] = atoi(argv[i+j+1]);
            }
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int PargemslrReadInputArg(const char *argname, char *val, int argc, char **argv)
   {
      int i;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            sprintf(val, "%s", argv[i+1]);
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int PargemslrReadInputArg(const char *argname, bool *val, int argc, char **argv)
   {
      int i, temp_bool;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            if (i+1 >= argc || argv[i+1][0] == '-') 
            {
               /* in this case, nothing follows the param, skip this one */
               continue;
            }
            temp_bool = atoi(argv[i+1]);
            val[0] = temp_bool ? true : false;
            return 1;
            break;
         }
      }
      return 0;
   }
   
   int PargemslrReadInputArg(const char *argname, int argc, char **argv)
   {
      int i;
      
      for ( i=0; i<argc; i++) 
      {
         if (argv[i][0] != '-') 
         {
            continue;
         }
         if (!strcmp(argname, argv[i]+1)) 
         {
            /* we find it */
            return 1;
         }
      }
      return 0;
   }
   
   template <typename T>
   typename std::enable_if<!(PargemslrIsComplex<T>::value), int>::type
   PargemslrPlotData(T* ydata, int length, int numx, int numy)
   {
      PARGEMSLR_CHKERR(length <= 0 || numx <= 0 || numy <= 0);
      /* is the total plot number greater than length? */
      numx = numx > length ? length : numx;
      int unitx = length/numx;
      
      /* now, collect data every unitx points */
      double maxval = 0.0, minval = 0.0;
      std::vector<double> vals;
      vals.resize(numx);
      
      for(int i = 0 ; i < numx-1 ; i ++)
      {
         int i1 = i*unitx;
         int i2 = i1+unitx;
         vals[i] = 0;
         for(int j = i1 ; j < i2 ; j ++)
         {
            vals[i] += (double)ydata[j]/(double)unitx;
         }
         maxval = maxval > vals[i] ? maxval : vals[i];
         minval = minval < vals[i] ? minval : vals[i];
      }
      
      int i1 = (numx-1)*unitx;
      int i2 = length;
      unitx = i2 - i1;
      vals[numx-1] = 0;
      for(int j = i1 ; j < i2 ; j ++)
      {
         vals[numx-1] += (double)ydata[j]/(double)unitx;
      }
      maxval = maxval > vals[numx-1] ? maxval : vals[numx-1];
      minval = minval < vals[numx-1] ? minval : vals[numx-1];
      
      /* all data collected, start plot */
      double maxabsval = PargemslrMax(maxval, -minval);
      int imaxabsval = (int)maxabsval;
      numy = numy > imaxabsval ? imaxabsval : numy;
      if(numy == 0)
      {
         numy = 1;
      }
      int unity = maxabsval/numy;
      if(unity*numy<maxabsval || unity == 0)
      {
         unity++;
      }
      int width = 0;
      while(imaxabsval>0)
      {
         imaxabsval/=10;
         width++;
      }
      width = PargemslrMax(6, width);
      
      std::vector<std::vector<int> > val2d_positive;
      std::vector<std::vector<int> > val2d_negative;
      val2d_positive.resize(numy+1);
      val2d_negative.resize(numy);
      
      for(int i = 0 ; i <= numy ; i ++)
      {
         val2d_positive[i].resize(numx, 0);
      }
      
      for(int i = 0 ; i < numy ; i ++)
      {
         val2d_negative[i].resize(numx, 0);
      }
      
      for(int i = 0 ; i < numx ; i ++)
      {
         int ival = (int)vals[i] / unity;
         if(ival >= 0)
         {
            val2d_positive[ival][i] = 2;
            for(int j = 0 ; j < ival ; j ++)
            {
               val2d_positive[j][i] = 1;
            }
         }
         else
         {
            val2d_negative[-1-ival][i] = 2;
            for(int j = 0 ; j < -1-ival ; j ++)
            {
               val2d_negative[j][i] = 1;
            }
         }
      }
      
      PARGEMSLR_PRINT("yunit: %d ymax: %f\n",unity,maxabsval);
      for(int i = numy ; i >= 0 ; i --)
      {
         PARGEMSLR_PRINT("%*d| ",width,i*unity);
         for(int j = 0 ; j < numx ; j ++)
         {
            if(val2d_positive[i][j] == 2)
            {
               PARGEMSLR_PRINT("* ");
            }
            else if(val2d_positive[i][j] == 1)
            {
               PARGEMSLR_PRINT("| ");
            }
            else
            {
               PARGEMSLR_PRINT("  ");
            }
         }
         PARGEMSLR_PRINT("\n");
      }
      PARGEMSLR_PRINT("xvals: ");
      unitx = length/numx;
      for(int i = 0 ; i < numx ; i ++)
      {
          PARGEMSLR_PRINT("-|");
      }
      PARGEMSLR_PRINT("- xunit: %d xmax: %d\n",unitx,length);
      if(minval < 0.0)
      {
         for(int i = 0 ; i < numy ; i ++)
         {
            PARGEMSLR_PRINT("%*d| ",width,(-i-1)*unity);
            for(int j = 0 ; j < numx ; j ++)
            {
               if(val2d_negative[i][j] == 2)
               {
                  PARGEMSLR_PRINT("* ");
               }
               else if(val2d_negative[i][j] == 1)
               {
                  PARGEMSLR_PRINT("| ");
               }
               else
               {
                  PARGEMSLR_PRINT("  ");
               }
            }
            PARGEMSLR_PRINT("\n");
         }
      }
      
      return 0;
   }
   template int PargemslrPlotData(int* ydata, int length, int numx, int numy);
   template int PargemslrPlotData(long int* ydata, int length, int numx, int numy);
   template int PargemslrPlotData(float* ydata, int length, int numx, int numy);
   template int PargemslrPlotData(double* ydata, int length, int numx, int numy);
   
   template <typename T>
   typename std::enable_if<PargemslrIsComplex<T>::value, int>::type
   PargemslrPlotData(T* ydata, int length, int numx, int numy)
   {
      
      int i;
      std::vector<double> rvals, ivals;
      
      rvals.resize(length);
      ivals.resize(length);
      
      for(i = 0 ; i < length ; i ++)
      {
         rvals[i] = ydata[i].Real();
         ivals[i] = ydata[i].Imag();
      }
      
      PargemslrPlotData( (double*)rvals.data(), length, numx, numy);
      PargemslrPlotData( (double*)ivals.data(), length, numx, numy);
      
      std::vector<double>().swap(rvals);
      std::vector<double>().swap(ivals);
      
      return 0;
   }
   template int PargemslrPlotData(complexs* ydata, int length, int numx, int numy);
   template int PargemslrPlotData(complexd* ydata, int length, int numx, int numy);
   
   int PargemslrSetOutputFile(const char *filename)
   {
      
      if( pargemslr_global::_out_file != stdout )
      {
         /* free the current */
         fclose(pargemslr_global::_out_file);
      }
      
      if ((pargemslr_global::_out_file = fopen(filename, "w")) == NULL)
      {
         printf("Can't open output file.\n");
         pargemslr_global::_out_file = stdout;
         return PARGEMSLR_ERROR_IO_ERROR;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   
}
