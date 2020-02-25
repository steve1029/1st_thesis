#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

#define aaa 1

void vecadd(double* a, double *b, double *c, int n)
{
	int idx, tid, max_th;
	//#pragma omp parallel for shared(a,b,c,n) private(idx)
	#pragma omp parallel for shared(a,b,c,n) private(idx,tid)
	for(idx = 0; idx < n; idx++)
	{
		tid    = omp_get_thread_num();
		max_th = omp_get_max_threads();
		printf("idx: %05d, tid: %d, max_th: %d\n", idx, tid, max_th);
		c[idx] = a[idx] + b[idx];
	}
}

int main(void)
{
	int n = 1;
	int idx;
	double *a, *b, *c;

	a = (double*)calloc(n, sizeof(double));
	b = (double*)calloc(n, sizeof(double));
	c = (double*)calloc(n, sizeof(double));

	for(idx=0; idx<n; idx++)
	{
		a[idx] = 1;
		b[idx] = 2;
	}

	vecadd(a,b,c,n);	

	return 0;
}
