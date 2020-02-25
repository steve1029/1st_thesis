#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
//#include <complex.h>

void fftw3d_yz_Z2Z (double* input_Re , double* input_Im,	\
					double* output_Re, double* output_Im,	\
					int Nx, int Ny, int Nz);

void ifftw3d_yz_Z2Z(double* input_Re , double* input_Im,	\
					double* output_Re, double* output_Im,	\
					int Nx, int Ny, int Nz);

void ifftw_ik_fftw3d_yz_Z2Z(					\
		double *Ex_Re, double *Ex_Im,			\
		double *diffyEx_Re, double *diffyEx_Im, \
		double *diffzEx_Re, double *diffzEx_Im, \
		double *ky, double *kz, int Nx, int Ny, int Nz);

void fftw3d_yz_Z2Z(								\
		double* input_Re , double* input_Im ,	\
		double* output_Re, double* output_Im,	\
		int Nx, int Ny, int Nz){

	/*	Put 3d numpy array with data type 'np.float64'
		3D numpy array will be treated by C as 1D array
		(that's why the data type of arguments are 1-level pointer). */

	int i,j,k;
	int dims = Nx * Ny * Nz;
	int idx;
	
	fftw_complex *fftw_input  = fftw_alloc_complex(dims);
	fftw_complex *fftw_output = fftw_alloc_complex(dims);

	/* Initialize input array */
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx = k + j * Nz + i * Ny * Nz;
				fftw_input[idx][0] = input_Re[idx];
				fftw_input[idx][1] = input_Im[idx];

			}
		}
	}

	/* Setup a FORWARD plan */
	int rank = 2;
	int n[] = {Ny,Nz};
	const int *inembed = NULL, *onembed = NULL;
	int istride = 1, ostride = 1;
	int idist = Ny*Nz, odist = Ny*Nz;
	int howmany = Nx;

	fftw_plan FFTW_2D_TF_OF_3D_FORWARD = fftw_plan_many_dft(rank, n, howmany,		\
											fftw_input , inembed, istride, idist,	\
											fftw_output, onembed, ostride, odist,	\
											FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(FFTW_2D_TF_OF_3D_FORWARD);

	/* Return output array */
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx = k + j * Nz + i * Ny * Nz;
				output_Re[idx] = fftw_output[idx][0] / (Ny*Nz);
				output_Im[idx] = fftw_output[idx][1] / (Ny*Nz);

			}
		}
	}

	fftw_destroy_plan(FFTW_2D_TF_OF_3D_FORWARD);
	fftw_free(fftw_input);
	fftw_free(fftw_output);

	return;
}

void ifftw3d_yz_Z2Z(								\
			double* input_Re , double* input_Im ,	\
			double* output_Re, double* output_Im,	\
			int Nx, int Ny, int Nz){

	/*	Put 3d numpy array with data type 'np.float64'
		3D numpy array will be treated by C as 1D array
		(that's why the data type of arguments are 1-level pointer). */

	int i,j,k;
	int dims = Nx * Ny * Nz;
	int idx;
	
	fftw_complex *fftw_input  = fftw_alloc_complex(dims);
	fftw_complex *fftw_output = fftw_alloc_complex(dims);

	/* Initialize input array */
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx = k + j * Nz + i * Ny * Nz;
				fftw_input[idx][0] = input_Re[idx];
				fftw_input[idx][1] = input_Im[idx];

			}
		}
	}

	/* Setup a BACKWARD plan */
	int rank = 2;
	int n[] = {Ny,Nz};
	const int *inembed = NULL, *onembed = NULL;
	int istride = 1, ostride = 1;
	int idist = Ny*Nz, odist = Ny*Nz;
	int howmany = Nx;

	fftw_plan FFTW_2D_TF_OF_3D_BACKWARD = fftw_plan_many_dft(rank, n, howmany, fftw_input, inembed, istride, idist,\
						fftw_output, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_execute(FFTW_2D_TF_OF_3D_BACKWARD);

	/* Return output array with normalization */
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx = k + j * Nz + i * Ny * Nz;
				output_Re[idx] = fftw_output[idx][0] / (Ny * Nz);
				output_Im[idx] = fftw_output[idx][1] / (Ny * Nz);

			}
		}
	}

	fftw_destroy_plan(FFTW_2D_TF_OF_3D_BACKWARD);
	fftw_free(fftw_input);
	fftw_free(fftw_output);

	return;
}

void ifftw_ik_fftw3d_yz_Z2Z(					\
		double *Ex_Re, double *Ex_Im,			\
		double *diffyEx_Re, double *diffyEx_Im, \
		double *diffzEx_Re, double *diffzEx_Im, \
		double *ky, double *kz,					\
		int Nx, int Ny, int Nz){

	int i,j,k;
	int dims = Nx * Ny * Nz;
	int idx;
	
	fftw_complex *diffyEx  = fftw_alloc_complex(dims);
	fftw_complex *diffzEx  = fftw_alloc_complex(dims);

	/* Fill diffxEy and diffzEy with input(Ex) array */
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx = k + j * Nz + i * Ny * Nz;

				diffyEx[idx][0] = Ex_Re[idx];
				diffyEx[idx][1] = Ex_Im[idx];

			}
		}
	}

	/* Setup FFT and IFFT plan */
	int rank = 2;
	int n[] = {Ny,Nz};
	const int *inembed = NULL, *onembed = NULL;
	int istride = 1, ostride = 1;
	int idist = Ny*Nz, odist = Ny*Nz;
	int howmany = Nx;

	fftw_plan diffyEx_FORWARD  = fftw_plan_many_dft(rank, n, howmany, diffyEx, inembed, istride, idist,\
						diffyEx, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan diffyEx_BACKWARD = fftw_plan_many_dft(rank, n, howmany, diffyEx, inembed, istride, idist,\
						diffyEx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan diffzEx_BACKWARD = fftw_plan_many_dft(rank, n, howmany, diffzEx, inembed, istride, idist,\
						diffzEx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Forward Transform
	fftw_execute(diffyEx_FORWARD);

	// Copy FT of Ex at diffyEx to diffzEx.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				idx = k + j * Nz + i * Nz * Ny;
				diffzEx[idx][0] = diffyEx[idx][0];
				diffzEx[idx][1] = diffyEx[idx][1];

			}
		}
	}

	// Multiply iky and ikz to each of them
	double real, imag;
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx = k + j * Nz + i * Ny * Nz;

				real = diffyEx[idx][0];
				imag = diffyEx[idx][1];

				diffyEx[idx][0] = -ky[j] * imag;
				diffyEx[idx][1] =  ky[j] * real;

				real = diffzEx[idx][0];
				imag = diffzEx[idx][1];

				diffzEx[idx][0] = -kz[k] * imag;
				diffzEx[idx][1] =  kz[k] * real;
			}
		}
	}

	// BACKWARD Transform
	fftw_execute(diffyEx_BACKWARD);
	fftw_execute(diffzEx_BACKWARD);

	/* Return the result diffyEx, diffzEx */
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx = k + j * Nz + i * Ny * Nz;
				diffyEx_Re[idx] = diffyEx[idx][0] / (Ny * Nz);
				diffyEx_Im[idx] = diffyEx[idx][1] / (Ny * Nz);

				diffzEx_Re[idx] = diffzEx[idx][0] / (Ny * Nz);
				diffzEx_Im[idx] = diffzEx[idx][1] / (Ny * Nz);

			}
		}
	}

	fftw_destroy_plan(diffyEx_FORWARD);
	fftw_destroy_plan(diffyEx_BACKWARD);
	fftw_destroy_plan(diffzEx_BACKWARD);
	fftw_free(diffyEx);
	fftw_free(diffzEx);

	return;
}
