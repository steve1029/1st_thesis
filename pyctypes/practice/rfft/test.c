#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <fftw3.h>

void rfft_1d_of_3d(int Nx, int Ny, int Nz, double* input, double* output);

int main(){

	double* input;
	double* output;

	int Nx=2, Ny=4, Nz=8;
	int i,j,k,idx;

	input  = (double*) malloc(Nx*Ny*Nz*sizeof(double));
	output = (double*) malloc(Nx*Ny*Nz*sizeof(double));
	
	for(i=0; i<Nx; i++){
		for(j=0; j<Ny; j++){
			for(k=0; k<Nz; k++){

				idx = k + j*Nz + i*Nz*Ny;
				input[idx] = idx;
				output[idx] = 0.;
			}
		}
	}

	rfft_1d_of_3d(Nx, Ny, Nz, input, output);

	for(i=0; i<Nx; i++){
		for(j=0; j<Ny; j++){
			for(k=0; k<Nz; k++){

				idx = k + j*Nz + i*Nz*Ny;
				printf("%f, %f\n", input[idx], output[idx]);
			}
		}
	}

	return 0;
}


void rfft_1d_of_3d(int Nx, int Ny, int Nz, double* input, double* output)
{
    int i,j,k,idx;

    // Allocate memory space.
    fftw_complex *FFTz = fftw_alloc_complex(Nx*Ny*(Nz/2+1));

    // Set forward FFTz parameters
    int rankz = 1;
    int nz[] = {Nz};
    int howmanyz = (Nx*Ny);
    const int *inembedz = NULL, *onembedz = NULL;
    int istridez = 1, ostridez = 1;
    int idistz = Nz, odistz= (Nz/2+1);

	//printf("%d\n", howmanyz);
	//printf("%d\n", *nz);
	//printf("%d\n", odistz);

    // Setup Forward plans.
    fftw_plan FFTz_for_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, input, inembedz, istridez, idistz, \
														FFTz, onembedz, ostridez, odistz, FFTW_ESTIMATE);

    // Set backward FFTz parameters
    int rankbz = 1;
    int nbz[] = {Nz}; // This is not (Nz/2+1)!! It took 24 hours to discover this error. Urhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
    const int *inembedbz = NULL, *onembedbz = NULL;
    int istridebz = 1, ostridebz = 1;
    int idistbz = (Nz/2+1), odistbz = Nz;
    int howmanybz = (Nx*Ny);

    // Setup Backward plans.
    fftw_plan FFTz_bak_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTz, inembedbz, istridebz, idistbz, \
														output, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

    fftw_execute(FFTz_for_plan);
    fftw_execute(FFTz_bak_plan);

	// Normalize the result.
	for(i=0; i<Nx; i++){
		for(j=0; j<Ny; j++){
			for(k=0; k<Nz; k++){
				
				idx = k + j*Nz + i*Nz*Ny;
				output[idx] = output[idx] / Nz;
			}
		}
	}

	fftw_destroy_plan(FFTz_for_plan);
	fftw_destroy_plan(FFTz_bak_plan);
    fftw_free(FFTz);

    return;
}
