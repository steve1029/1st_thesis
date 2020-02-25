#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <fftw3.h>

void rfft_1d(
	int Nx, int Ny, int Nz, 
	double* kz, double* data, double* diffz_data
){

	int Nzh = Nz/2+1;

	int i,j,k,idx,idx_T;
	int nthreads = omp_get_max_threads();
	double real, imag;

	// initialize multi-threaded fftw3.
	fftw_init_threads();
	fftw_plan_with_nthreads(nthreads);

	double* diffz = fftw_alloc_real(Nx*Ny*Nz);
	fftw_complex *FFTz = fftw_alloc_complex(Nx*Ny*Nzh);

	// Set forward FFTz parameters
	int rankz = 1;
	int nz[1] = {Nz};
	int howmanyz = (Nx*Ny);
	const int *inembedz = NULL, *onembedz = NULL;
	int istridez = 1, ostridez = 1;
	int idistz = Nz, odistz= Nzh;

	// Setup Forward plans.
	fftw_plan FFTz_for_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, diffz, inembedz, istridez, idistz, \
														FFTz, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	// Set backward FFTz parameters
	int rankbz = 1;
	int nbz[1] = {Nz};
	int howmanybz = (Nx*Ny);
	const int *inembedbz = NULL, *onembedbz = NULL;
	int istridebz = 1, ostridebz = 1;
	int idistbz = Nzh, odistbz = Nz;

	// Setup Backward plans.
	fftw_plan FFTz_bak_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTz, inembedbz, istridebz, idistbz, \
														diffz, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	// Initialize diffz.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				idx   = k + j*Nz + i*Nz*Ny;

				diffz[idx] = data[idx];
			}
		}
	}

	// Perform 1D FFT along z axis.
	fftw_execute(FFTz_for_plan);

	// Multiply ikz.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nzh; k++){
				
				idx   = k + j*Nzh + i*Nzh*Ny;

				real = FFTz[idx][0];
				imag = FFTz[idx][1];

				FFTz[idx][0] = -kz[k] * imag;
				FFTz[idx][1] =  kz[k] * real;

			}
		}
	}

	// Perform 1D IFFT along y and z axis.
	fftw_execute(FFTz_bak_plan);

	// Normalize reconstructed signal.
	// Transpose to restore original field array.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				idx   = k + j*Nz + i*Nz*Ny;

				diffz_data[idx] = diffz[idx] / Nz;

			}
		}
	}

	// Destroy the plan
	fftw_destroy_plan(FFTz_for_plan);
	fftw_destroy_plan(FFTz_bak_plan);

	// Free the memory.
	fftw_free(FFTz);
	fftw_free(diffz);

	return;
}

void rfft_1d_of_3d( \
	int Nx, int Ny, int Nz, \
	double* ky, double* kz, \
	double* data, double* diffy_data, double* diffz_data \
){

	int Nzh = Nz/2 + 1;
	int Nyh = Ny/2 + 1;

	int i,j,k,idx,idx_T;
	int nthreads = omp_get_max_threads();
	double real, imag;

	// initialize multi-threaded fftw3.
	fftw_init_threads();
	fftw_plan_with_nthreads(nthreads);

	// Allocate memory space.
	double* diffz = fftw_alloc_real(Nx*Ny*Nz);
	double* diffy = fftw_alloc_real(Nx*Ny*Nz);
	fftw_complex *FFTz = fftw_alloc_complex(Nx*Ny*Nzh);
	fftw_complex *FFTy = fftw_alloc_complex(Nx*Nyh*Nz);

	// Set forward FFTz parameters
	int rankz = 1;
	int nz[] = {Nz};
	int howmanyz = (Nx*Ny);
	const int *inembedz = NULL, *onembedz = NULL;
	int istridez = 1, ostridez = 1;
	int idistz = Nz, odistz= Nzh;

	// Setup Forward plans.
	fftw_plan FFTz_for_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, diffz, inembedz, istridez, idistz, \
														FFTz, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	// Set backward FFTz parameters
	int rankbz = 1;
	int nbz[] = {Nz};
	int howmanybz = (Nx*Ny);
	const int *inembedbz = NULL, *onembedbz = NULL;
	int istridebz = 1, ostridebz = 1;
	int idistbz = Nzh, odistbz = Nz;

	// Setup Backward plans.
	fftw_plan FFTz_bak_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTz, inembedbz, istridebz, idistbz, \
														diffz, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	// Set forward FFTy parameters
	int ranky = 1;
	int ny[] = {Ny};
	int howmanyy = Nx*Nz;
	int istridey = 1, ostridey = 1;
	int idisty = Ny, odisty = Nyh;
	const int *inembedy = NULL, *onembedy = NULL;

	// Setup Forward plans.
	fftw_plan FFTy_for_plan = fftw_plan_many_dft_r2c(ranky, ny, howmanyy, diffy, inembedy, istridey, idisty, \
														FFTy, onembedy, ostridey, odisty, FFTW_ESTIMATE);

	// Set backward FFTy parameters
	int rankby = 1;
	int nby[] = {Ny};
	int howmanyby = Nx*Nz;
	int istrideby = 1, ostrideby = 1;
	int idistby = Nyh, odistby = Ny;
	const int *inembedby = NULL, *onembedby = NULL;

	// Setup Backward plans.
	fftw_plan FFTy_bak_plan = fftw_plan_many_dft_c2r(rankby, nby, howmanyby, FFTy, inembedby, istrideby, idistby, \
														diffy, onembedby, ostrideby, odistby, FFTW_ESTIMATE);

	// Transpose data to get y-derivatives.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx   = k + j*Nz + i*Nz*Ny;
				idx_T = j + k*Ny + i*Nz*Ny;

				diffz[idx] = data[idx];
				diffy[idx_T] = data[idx];

			}
		}
	}

	// Perform 1D FFT along y and z axis.
	fftw_execute(FFTz_for_plan);
	fftw_execute(FFTy_for_plan);

	// Multiply ikz.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nzh; k++){
				
				idx   = k + j*Nzh + i*Nzh*Ny;

				real = FFTz[idx][0];
				imag = FFTz[idx][1];

				FFTz[idx][0] = -kz[k] * imag;
				FFTz[idx][1] =  kz[k] * real;

			}
		}
	}

	// Multiply iky.
	for(i=0; i < Nx; i++){
		for(k=0; k < Nz; k++){
			for(j=0; j < Nyh; j++){
				
				idx = j + k*Nyh + i*Nz*Nyh;

				real = FFTy[idx][0];
				imag = FFTy[idx][1];

				FFTy[idx][0] = -ky[j] * imag;
				FFTy[idx][1] =  ky[j] * real;
			}
		}
	}

	// Perform 1D IFFT along y and z axis.
	fftw_execute(FFTz_bak_plan);
	fftw_execute(FFTy_bak_plan);

	// Normalize reconstructed signal.
	// Transpose to restore original field array.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				idx   = k + j*Nz + i*Nz*Ny;
				idx_T = j + k*Ny + i*Nz*Ny;

				diffz_data[idx] = diffz[idx] / Nz;
				diffy_data[idx] = diffy[idx_T] / Ny;

			}
		}
	}

	// Destroy the plan
	fftw_destroy_plan(FFTz_for_plan);
	fftw_destroy_plan(FFTz_bak_plan);
	fftw_destroy_plan(FFTy_for_plan);
	fftw_destroy_plan(FFTy_bak_plan);

	// Free the memory.
	fftw_free(FFTz);
	fftw_free(FFTy);
	fftw_free(diffy);
	fftw_free(diffz);
	return;
}


void rfft_2d_of_3d( \
	int Nx, int Ny, int Nz, \
	double* data \
){

	// initialize multi-threaded fftw3.
	//fftw_init_threads();
	//int nthreads = omp_get_max_threads();
	//fftw_plan_with_nthreads(nthreads);

	int i,j,k, idx;

	fftw_complex *fftyz = fftw_alloc_complex(Nx*Ny*Nz/2+1);

	// Set forward FFT parameters
	int rank = 2;
	int n[] = {Ny,Nz};
	const int *inembed = NULL, *onembed = NULL;
	int istride = 1, ostride = 1;
	int idist = Ny*Nz, odist= Ny*Nz;
	int howmany = Nx;

	printf("Here?\n");
	// Setup Forward plans.
	fftw_plan FFT2D_for_plan = fftw_plan_many_dft_r2c(rank, n, howmany, data, inembed, istride, idist, \
														fftyz, onembed, ostride, odist, FFTW_ESTIMATE);

	// Set backward FFT parameters
	int rank_b = 2;
	int n_b[] = {Ny,Nz/2+1};
	const int *inembed_b = NULL, *onembed_b = NULL;
	int istride_b = 1, ostride_b = 1;
	int idist_b = Ny*(Nz/2+1), odist_b= Ny*(Nz/2+1);
	int howmany_b = Nx;

	// Setup Backward plans.
	fftw_plan FFT2D_bak_plan = fftw_plan_many_dft_c2r(rank_b, n_b, howmany_b, fftyz, inembed_b, istride_b, idist_b, \
														data, onembed_b, ostride_b, odist_b, FFTW_ESTIMATE);

	// Perform 2D FFT along x and y axis.
	fftw_execute(FFT2D_for_plan);

	// Perform 2D IFFT along x and y axis.
	fftw_execute(FFT2D_bak_plan);

	// Normalize the results.
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx = k + j*Nz + i*Nz*Ny;

				data[idx] = data[idx] / (Ny*Nz);

			}
		}
	}

	// Destroy the plan
	fftw_destroy_plan(FFT2D_for_plan);
	fftw_destroy_plan(FFT2D_bak_plan);

	// Free the memory.
	fftw_free(fftyz);
	
	return;
}
