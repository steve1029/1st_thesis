#include<stdio.h>
#include<stdlib.h>

void npclx_printf( double *target, int size){

	int i;

	for(i=0; i< (2*size); i++){
		printf("%lf\n",target[i]);
	}

	return;
}
