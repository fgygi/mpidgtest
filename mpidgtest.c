////////////////////////////////////////////////////////////////////////////////
//
// mpidgtest.c
//
// Floating point performance benchmark
//
// F.Gygi 2021-06-23
//
// Performs multiple matrix products in separate MPI tasks using dgemm
//
// Use on a single node with as many MPI tasks as available cores
// Compile with Intel MKL library:
// (this may require "module load intelmpi intel mkl")
// $ mpicc -mkl mpidgtest.c
// $ mpirun -np 8 ./a.out 2000 2000 2000
//
// Compile with blas library:
// $ mpicc -lblas mpidgtest.c
// $ mpirun -np 8 ./a.out 2000 2000 2000
//
////////////////////////////////////////////////////////////////////////////////

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include "mpi.h"

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
  int myid, ntasks;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  int i,j,m=atoi(argv[1]), n=atoi(argv[2]), k=atoi(argv[3]);
  assert(sizeof(long) == 8);
  assert(sizeof(long long) == 8);

  long int clk = 0;
  double wtime;
  double *a=(double*)malloc(m*k*sizeof(double));
  double *b=(double*)malloc(k*n*sizeof(double));
  double *c=(double*)malloc(m*n*sizeof(double));

  double zero=0.0,one=1.0;
  char cn='N';
  double nops=2.0*m*n*k;

  for ( i = 0; i < m*k; i++ )
    a[i] = 1.5;
  for ( i = 0; i < k*n; i++ )
    b[i] = 1.5;
  for ( i = 0; i < m*n; i++ )
    c[i] = 1.5;

  wtime = MPI_Wtime();
  clk = readTSC();
  MPI_Barrier(MPI_COMM_WORLD);
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  dgemm_(&cn,&cn,&n,&m,&k,&one,a,&m,b,&k,&zero,c,&m);
  MPI_Barrier(MPI_COMM_WORLD);
  clk = readTSC() - clk;
  clk /= 5;
  wtime = MPI_Wtime() - wtime;
  wtime /= 5;

  double minwtime,maxwtime;
  MPI_Reduce(&wtime,&minwtime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&wtime,&maxwtime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  long long minclk,maxclk;
  MPI_Reduce(&clk,&minclk,1,MPI_LONG,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&clk,&maxclk,1,MPI_LONG,MPI_MAX,0,MPI_COMM_WORLD);

  if ( myid == 0 )
  {
    printf("\n mpidgtest ntasks=%d m,n,k= %d %d %d\n\n", ntasks, m, n, k);
    printf(" flop count:           %e\n", nops );
    printf(" wtime min/max:        %f / %f\n", minwtime,maxwtime);
    printf(" clk   min/max:        %ld / %ld\n", minclk,maxclk);
    printf(" flops/clock min/max:  %f / %f\n", nops/maxclk,nops/minclk);
    printf(" aggregate flop count: %e\n", nops*ntasks);
    printf(" aggregate GFlops:     %f\n", nops*ntasks/(1.0e9*maxwtime));
  }

  MPI_Finalize();
  return 0;
}
