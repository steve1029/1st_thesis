#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

int main(void){

	double complex a = 1 + I * 1;

	printf("%lf, %lf\n", creal(a), cimag(a));
	printf("size of double complex type: %lu bytes\n", sizeof(double complex));

	double complex *aa;
	int i,N=5;

	aa = (double complex*) calloc(N, sizeof(double complex));

	for(i=0;i<N;i++){

		aa[i] = i + I * i;
		printf("real: %lf, imag: %lf\n",creal(aa[i]), cimag(aa[i]));
	}
	return 0;
}
