#include <stdio.h>

void index_test( double *array, int Nx, int Ny, int Nz){

	int i,j,k;
	int idx;

	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx = k + j * Nz + i * Nz * Ny;
				array[idx] = (double) idx;

				printf("%lf", array[idx]);

			}
			printf("\n");
		}
	printf("\n\n");
	}

	return;
}
