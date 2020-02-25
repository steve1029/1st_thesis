#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

int main(void){

	int Nx, Ny, Nz;
	int i,j,k;
	int ijk;

	fftw_complex *my3Darr = fftw_alloc_complex(Nx * Ny * Nz);

	fftw_free(my3Darr);

	return 0;
}
