////////////////////////////////////////////////////////////////////////////////
//
// ompdgtest.c
//
// Floating point performance benchmark
//
// F.Gygi 2021-07-25
//
// Performs multiple matrix products in separate OpenMP threads
//
// Compile with Intel MKL library:
// (this may require "module load intel mkl")
// $ icc -qopenmp -mkl ompdgtest.c
// $ export OMP_NUM_THREADS=<numthreads>
// $ ./a.out 2000 2000 2000
//
// Compile with blas library:
// $ gcc -fopenmp -lblas ompdgtest.c
// $ export OMP_NUM_THREADS=<numthreads>
// $ ./a.out 2000 2000 2000
//
////////////////////////////////////////////////////////////////////////////////

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<float.h> // DBL_MAX
#include<limits.h> // LONG_MAX
#include<omp.h>

long long readTSC(void)
{
  union { long long complete; unsigned int part[2]; } ticks;
  __asm__ ("rdtsc; mov %%eax,%0;mov %%edx,%1"
            : "=mr" (ticks.part[0]),
              "=mr" (ticks.part[1])
            : /* no inputs */
            : "eax", "edx");
  return ticks.complete;
}

void dgemm_(char *ta, char *tb, int *m, int *n, int *k,
 double *alpha, double *a, int *lda, double *b, int *ldb,
 double *beta, double *c, int *ldc);

int main(int argc, char** argv)
{
  int nthreads = omp_get_max_threads();
  int m=atoi(argv[1]), n=atoi(argv[2]), k=atoi(argv[3]);
  double nops=2.0*m*n*k;
  assert(sizeof(long) == 8);
  assert(sizeof(long long) == 8);
  double minwtime=DBL_MAX,maxwtime=0.0;
  long long minclk=LONG_MAX,maxclk=0;
#pragma omp parallel reduction(min:minwtime) reduction(max:maxwtime) \
reduction(min:minclk) reduction(max:maxclk)
  {
  int tid = omp_get_thread_num();
  int i,j;
  long int clk = 0;
  double wtime;
  double *a=(double*)malloc(m*k*sizeof(double));
  double *b=(double*)malloc(k*n*sizeof(double));
  double *c=(double*)malloc(m*n*sizeof(double));

  double zero=0.0,one=1.0;
  char cn='N';

  for ( i = 0; i < m*k; i++ )
    a[i] = 1.5;
  for ( i = 0; i < k*n; i++ )
    b[i] = 1.5;
  for ( i = 0; i < m*n; i++ )
    c[i] = 1.5;

  wtime = omp_get_wtime();
  clk = readTSC();
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  clk = readTSC() - clk;
  clk /= 5;
  wtime = omp_get_wtime() - wtime;
  wtime /= 5;

  minwtime = wtime;
  maxwtime = wtime;
  minclk = clk;
  maxclk = clk;
  } // end parallel

  printf("\n ompdgtest nthreads=%d m,n,k= %d %d %d\n\n", nthreads, m, n, k);
  printf(" flop count:           %e\n", nops );
  printf(" wtime min/max:        %f / %f\n", minwtime,maxwtime);
  printf(" clk   min/max:        %ld / %ld\n", minclk,maxclk);
  printf(" flops/clock min/max:  %f / %f\n", nops/maxclk,nops/minclk);
  printf(" aggregate flop count: %e\n", nops*nthreads);
  printf(" aggregate GFlops:     %f\n", nops*nthreads/(1.0e9*maxwtime));

  return 0;
}
