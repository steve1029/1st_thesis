#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <fftw3.h>
//#include <complex.h>

/*	
	Author: Donggun Lee	
	Date  : 18.02.14

	This script only contains update equations of the hybrid PSTD-FDTD method.
	Update equations for UPML or CPML, ADE-FDTD is not developed here.

	Core update equations are useful when testing the speed of an algorithm
	or the performance of the hardware such as CPU, GPU or memory.

	Update Equations for E and H field
	Update Equations for MPI Boundary

	Discription of variables
	------------------------
	i: x index
	j: y index
	k: z index
	
	idx  : 1 dimensional index of elements where its index in 3D is (i  , j  , k  ).
	i_idx: 1 dimensional index of elements where its index in 3D is (i+1, j  , k  ).
	idx_i: 1 dimensional index of elements where its index in 3D is (i-1, j  , k  ).
	idx_0: 1 dimensional index of elements where its index in 3D is (0  , j  , k  ).

	ex)
		idx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
		i_idx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
		idx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
		idx_0 = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;

*/

/***********************************************************************************/
/******************************** FUNCTION DECLARATION *****************************/
/***********************************************************************************/

//Get derivatives of E field.
void get_deriv_z_E(
	int Nx, int Ny, int Nz,
	double* Ex_re,
	double* Ey_re,
	double* kz,
	double* diffzEx_re,
	double* diffzEy_re
);

void get_deriv_y_E(
	int Nx, int Ny, int Nz,
	double* Ex_re,
	double* Ez_re,
	double* ky,
	double* diffyEx_re,
	double* diffyEz_re
);

void get_deriv_x_E(
	int	Nx, int Ny, int Nz,
	double  dx,
	double* Ey_re,
	double* Ez_re,
	double* diffxEy_re,
	double* diffxEz_re
);

// Update H field.
void updateH(										\
	int		Nx,		int		Ny,		int		Nz,	\
	double  dt,										\
	double *Hx_re,		double *Hx_im,				\
	double *Hy_re,		double *Hy_im,				\
	double *Hz_re,		double *Hz_im,				\
	double *mu_HEE,		double *mu_EHH,				\
	double *mcon_HEE,	double *mcon_EHH,			\
	double *diffxEy_re, double *diffxEy_im,			\
	double *diffxEz_re, double *diffxEz_im,			\
	double *diffyEx_re, double *diffyEx_im,			\
	double *diffyEz_re, double *diffyEz_im,			\
	double *diffzEx_re, double *diffzEx_im,			\
	double *diffzEy_re, double *diffzEy_im			\
);

// Get derivatives of H field.
void get_deriv_z_H(
	int Nx, int Ny, int Nz,
	double* Hx_re,
	double* Hy_re,
	double* kz,
	double* diffzHx_re,
	double* diffzHy_re
);

void get_deriv_y_H(
	int Nx, int Ny, int Nz,
	double* Hx_re,
	double* Hz_re,
	double* ky,
	double* diffyHx_re,
	double* diffyHz_re
);

void get_deriv_x_H(
	int Nx, int Ny, int Nz,
	double dx,
	double* Hy_re,
	double* Hz_re,
	double* diffxHy_re,
	double* diffxHz_re
);

// Update E field
void updateE(
	int		Nx,		int		Ny,		int		Nz,	\
	double dt,										\
	double *Ex_re,		double *Ex_im,				\
	double *Ey_re,		double *Ey_im,				\
	double *Ez_re,		double *Ez_im,				\
	double *eps_HEE,	double *eps_EHH,			\
	double *econ_HEE,	double *econ_EHH,			\
	double *diffxHy_re, double *diffxHy_im,			\
	double *diffxHz_re, double *diffxHz_im,			\
	double *diffyHx_re, double *diffyHx_im,			\
	double *diffyHz_re, double *diffyHz_im,			\
	double *diffzHx_re, double *diffzHx_im,			\
	double *diffzHy_re, double *diffzHy_im			\
);

/***********************************************************************************/
/******************************** FUNCTION DESCRIPTION *****************************/
/***********************************************************************************/

// Get derivatives of E field in the first and middle rank.

void get_deriv_z_E(
	int Nx, int Ny, int Nz,
	double* Ex_re,
	double* Ey_re,
	double* kz,
	double* diffzEx_re,
	double* diffzEy_re
){
	// int for index
	int i, j, k, idx;
	int nthreads = omp_get_max_threads();
	double real, imag;

	// initialize multi-threaded fftw3.
	fftw_init_threads();

	// initialize nthreaded fftw3 plan.
	fftw_plan_with_nthreads(nthreads);

	// Memory allocation for transpose of the field data.
	fftw_complex *FFTzEx   = fftw_alloc_complex(Nx*Ny*(Nz/2+1));
	fftw_complex *FFTzEy   = fftw_alloc_complex(Nx*Ny*(Nz/2+1));

	// Set Plans for real FFT along z axis.
	int rankz = 1;
	int nz[]  = {Nz};
	int howmanyz = Nx*Ny;
	int istridez = 1, ostridez = 1;
	int idistz = Nz, odistz = Nz/2+1;
	const int *inembedz = NULL, *onembedz = NULL;

	fftw_plan FFTz_Ex_FOR_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, Ex_re, inembedz, istridez, idistz, \
														FFTzEx, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	fftw_plan FFTz_Ey_FOR_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, Ey_re, inembedz, istridez, idistz, \
														FFTzEy, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	// Set Plans for inverse real FFT along z axis.
	int rankbz = 1;
	int nbz[]  = {Nz};
	int howmanybz = Nx*Ny;
	int istridebz = 1, ostridebz = 1;
	int idistbz = Nz/2+1, odistbz = Nz;
	const int *inembedbz = NULL, *onembedbz = NULL;

	// Setup BACKWARD plans.
	fftw_plan FFTz_Ex_BAK_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTzEx, inembedbz, istridebz, idistbz, \
														diffzEx_re, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	fftw_plan FFTz_Ey_BAK_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTzEy, inembedbz, istridebz, idistbz, \
														diffzEy_re, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	fftw_execute(FFTz_Ex_FOR_plan);
	fftw_execute(FFTz_Ey_FOR_plan);

	// Multiply ikz.
	#pragma omp parallel for	\
		shared(Nx, Ny, Nz, FFTzEx, FFTzEy, kz)	\
		private(i, j, k, idx, real, imag)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz/2+1); k++){

				idx = k + j*(Nz/2+1) + i*(Nz/2+1)*Ny;

				real = FFTzEx[idx][0];
				imag = FFTzEx[idx][1];

				FFTzEx[idx][0] = kz[k] * imag;
				FFTzEx[idx][1] = -kz[k] * real;

				//FFTzEx[idx][0] = -kz[k] * imag;
				//FFTzEx[idx][1] =  kz[k] * real;

				real = FFTzEy[idx][0];
				imag = FFTzEy[idx][1];

				FFTzEy[idx][0] = kz[k] * imag;
				FFTzEy[idx][1] = -kz[k] * real;

				//FFTzEy[idx][0] = -kz[k] * imag;
				//FFTzEy[idx][1] =  kz[k] * real;

			}
		}
	}

	// Backward FFT.
	fftw_execute(FFTz_Ex_BAK_plan);
	fftw_execute(FFTz_Ey_BAK_plan);

	// Normalize the results of pseudo-spectral method.
	#pragma omp parallel for \
		shared(	Nx, Ny, Nz, \
		diffzEx_re, \
		diffzEy_re) \
		private(i, j, k, idx)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;

				diffzEx_re[idx] = diffzEx_re[idx] / Nz;
				diffzEy_re[idx] = diffzEy_re[idx] / Nz;

			}
		}
	}

	// Destroy the plan and free the memory.
	fftw_destroy_plan(FFTz_Ex_FOR_plan);
	fftw_destroy_plan(FFTz_Ey_FOR_plan);
	fftw_destroy_plan(FFTz_Ex_BAK_plan);
	fftw_destroy_plan(FFTz_Ey_BAK_plan);
	fftw_free(FFTzEx);
	fftw_free(FFTzEy);

	return;
}

void get_deriv_y_E(
	int Nx, int Ny, int Nz,
	double* Ex_re,
	double* Ez_re,
	double* ky,
	double* diffyEx_re,
	double* diffyEz_re
){

	// int for index
	int i,j,k;
	int idx, i_idx, idx_T, idx_0;
	int nthreads = omp_get_max_threads();
	double real, imag;

	// initialize multi-threaded fftw3.
	fftw_init_threads();

	// initialize nthreaded fftw3 plan.
	fftw_plan_with_nthreads(nthreads);

	// Memory allocation for transpose of the field data.
	double* diffyEx_T = fftw_alloc_real(Nx*Ny*Nz);
	double* diffyEz_T = fftw_alloc_real(Nx*Ny*Nz);
	fftw_complex* FFTyEx_T = fftw_alloc_complex(Nx*(Ny/2+1)*Nz);
	fftw_complex* FFTyEz_T = fftw_alloc_complex(Nx*(Ny/2+1)*Nz);

	// Set Plans for real FFT along y axis.
	int ranky = 1;
	int ny[]  = {Ny};
	int howmanyy = Nx*Nz;
	int istridey = 1, ostridey = 1;
	int idisty = Ny, odisty = Ny/2+1;
	const int *inembedy = NULL, *onembedy = NULL;

	fftw_plan FFTy_Ex_FOR_plan = fftw_plan_many_dft_r2c(ranky, ny, howmanyy, diffyEx_T, inembedy, istridey, idisty, \
														FFTyEx_T, onembedy, ostridey, odisty, FFTW_ESTIMATE);

	fftw_plan FFTy_Ez_FOR_plan = fftw_plan_many_dft_r2c(ranky, ny, howmanyy, diffyEz_T, inembedy, istridey, idisty, \
														FFTyEz_T, onembedy, ostridey, odisty, FFTW_ESTIMATE);
	
	// Set Plans for inverse real FFT along y axis.
	int rankby = 1;
	int nby[]  = {Ny};
	int howmanyby = Nx*Nz;
	int istrideby = 1, ostrideby = 1;
	int idistby = Ny/2+1, odistby = Ny;
	const int *inembedby = NULL, *onembedby = NULL;

	fftw_plan FFTy_Ex_BAK_plan = fftw_plan_many_dft_c2r(rankby, nby, howmanyby, FFTyEx_T, inembedby, istrideby, idistby, \
														diffyEx_T, onembedby, ostrideby, odistby, FFTW_ESTIMATE);

	fftw_plan FFTy_Ez_BAK_plan = fftw_plan_many_dft_c2r(rankby, nby, howmanyby, FFTyEz_T, inembedby, istrideby, idistby, \
														diffyEz_T, onembedby, ostrideby, odistby, FFTW_ESTIMATE);

	// Transpose y and z axis of the Ex and Ez to get y-derivatives of them.
	#pragma omp parallel for	\
		shared(Nx, Ny, Nz, diffyEx_T, diffyEz_T, Ex_re, Ez_re)	\
		private(i, j, k, idx, idx_T)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx   = k + j*Nz + i*Nz*Ny;
				idx_T = j + k*Ny + i*Nz*Ny;

				diffyEx_T[idx] = Ex_re[idx_T];
				diffyEz_T[idx] = Ez_re[idx_T];

			}
		}
	}

	// Perform 2D FFT along y and z-axis.
	fftw_execute(FFTy_Ex_FOR_plan);
	fftw_execute(FFTy_Ez_FOR_plan);

	// Multiply iky.
	#pragma omp parallel for	\
		shared(Nx, Ny, Nz, FFTyEx_T, FFTyEz_T, ky)	\
		private(i, j, k, idx, real, imag)
	for(i=0; i < Nx; i++){
		for(k=0; k < Nz; k++){
			for(j=0; j < (Ny/2+1); j++){

				idx_T = j + k*(Ny/2+1) + i*(Ny/2+1)*Nz;

				real = FFTyEx_T[idx_T][0];
				imag = FFTyEx_T[idx_T][1];

				FFTyEx_T[idx_T][0] = ky[j] * imag;
				FFTyEx_T[idx_T][1] = -ky[j] * real;

				//FFTyEx_T[idx_T][0] = -ky[j] * imag;
				//FFTyEx_T[idx_T][1] =  ky[j] * real;

				real = FFTyEz_T[idx_T][0];
				imag = FFTyEz_T[idx_T][1];

				FFTyEz_T[idx_T][0] = ky[j] * imag;
				FFTyEz_T[idx_T][1] = -ky[j] * real;

				//FFTyEz_T[idx_T][0] = -ky[j] * imag;
				//FFTyEz_T[idx_T][1] =  ky[j] * real;

			}
		}
	}

	// Perform Inverse FFT.
	fftw_execute(FFTy_Ex_BAK_plan);
	fftw_execute(FFTy_Ez_BAK_plan);

	// Normalize the results of pseudo-spectral method.
	// Get diffx, diffy and diffz of H fields.
	#pragma omp parallel for \
		shared(	Nx, Ny, Nz, \
				diffyEx_re, \
				diffyEz_re, \
				diffyEx_T, \
				diffyEz_T) \
		private(i, j, k, idx, idx_T)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;
				idx_T = (j  ) + (k  )*Ny + (i  )*Nz*Ny;

				diffyEx_re[idx] = diffyEx_T[idx_T] / Ny;
				diffyEz_re[idx] = diffyEz_T[idx_T] / Ny;

			}
		}
	}

	fftw_destroy_plan(FFTy_Ex_FOR_plan);
	fftw_destroy_plan(FFTy_Ez_FOR_plan);
	fftw_destroy_plan(FFTy_Ex_BAK_plan);
	fftw_destroy_plan(FFTy_Ez_BAK_plan);

	fftw_free(FFTyEx_T);
	fftw_free(FFTyEz_T);
	free(diffyEx_T);
	free(diffyEz_T);

	return;
}

void get_deriv_x_E(
	int	Nx, int Ny, int Nz,
	double  dx,
	double* Ey_re,
	double* Ez_re,
	double* diffxEy_re,
	double* diffxEz_re
){

	// int for index
	int i,j,k;
	int idx, i_idx, idx_T, idx_0;
	int nthreads = omp_get_max_threads();
	double real, imag;

	// initialize multi-threaded fftw3.
	fftw_init_threads();

	// initialize nthreaded fftw3 plan.
	fftw_plan_with_nthreads(nthreads);

	// Memory allocation for transpose of the field data.
	double* diffxEy_T = fftw_alloc_real(Nx*Ny*Nz);
	double* diffxEz_T = fftw_alloc_real(Nx*Ny*Nz);
	fftw_complex* FFTxEy_T = fftw_alloc_complex((Nx/2+1)*Ny*Nz);
	fftw_complex* FFTxEz_T = fftw_alloc_complex((Nx/2+1)*Ny*Nz);

	// Set Plans for real FFT along y axis.
	int rankx = 1;
	int nx[]  = {Nx};
	int howmanyx = Ny*Nz;
	int istridex = 1, ostridex = 1;
	int idistx = Nx, odistx = Nx/2+1;
	const int *inembedx = NULL, *onembedx = NULL;

	fftw_plan FFTx_Ey_FOR_plan = fftw_plan_many_dft_r2c(rankx, nx, howmanyx, diffxEy_T, inembedx, istridex, idistx, \
														FFTxEy_T, onembedx, ostridex, odistx, FFTW_ESTIMATE);

	fftw_plan FFTx_Ez_FOR_plan = fftw_plan_many_dft_r2c(rankx, nx, howmanyx, diffxEz_T, inembedx, istridex, idistx, \
														FFTxEz_T, onembedx, ostridex, odistx, FFTW_ESTIMATE);
	
	// Set Plans for inverse real FFT along y axis.
	int rankbx = 1;
	int nbx[]  = {Nx};
	int howmanybx = Ny*Nz;
	int istridebx = 1, ostridebx = 1;
	int idistbx = Nx/2+1, odistbx = Nx;
	const int *inembedbx = NULL, *onembedbx = NULL;

	fftw_plan FFTx_Ey_BAK_plan = fftw_plan_many_dft_c2r(rankbx, nbx, howmanybx, FFTxEy_T, inembedbx, istridebx, idistbx, \
														diffxEy_T, onembedbx, ostridebx, odistbx, FFTW_ESTIMATE);

	fftw_plan FFTx_Ez_BAK_plan = fftw_plan_many_dft_c2r(rankbx, nbx, howmanybx, FFTxEz_T, inembedbx, istridebx, idistbx, \
														diffxEz_T, onembedbx, ostridebx, odistbx, FFTW_ESTIMATE);

	// Transpose y and z axis of the Ex and Ez to get y-derivatives of them.
	#pragma omp parallel for	\
		shared(Nx, Ny, Nz, diffyEx_T, diffyEz_T, Ex_re, Ez_re)	\
		private(i, j, k, idx, idx_T)
	for(k=0; k < Nz; k++){
		for(j=0; j < Ny; j++){
			for(i=0; i < Nx; i++){

				idx   = i + j*Nx + k*Ny*Nx;
				idx_T = k + j*Nz + i*Ny*Nz;

				diffxEy_T[idx] = Ey_re[idx_T];
				diffxEz_T[idx] = Ez_re[idx_T];

			}
		}
	}

	// Perform 2D FFT along y and z-axis.
	fftw_execute(FFTx_Ey_FOR_plan);
	fftw_execute(FFTx_Ez_FOR_plan);

	// Multiply ikx.
	#pragma omp parallel for	\
		shared(Nx, Ny, Nz, FFTxEy_T, FFTxEz_T, kx)	\
		private(i, j, k, idx, real, imag)
	for(k=0; k < Nz; k++){
		for(j=0; j < Ny; j++){
			for(i=0; i < (Nx/2+1); i++){

				idx = i + j*(Nx/2+1) + k*(Nx/2+1)*Ny;

				real = FFTxEy_T[idx_T][0];
				imag = FFTxEy_T[idx_T][1];

				FFTxEy_T[idx_T][0] =  kx[j] * imag;
				FFTxEy_T[idx_T][1] = -kx[j] * real;

				real = FFTxEz_T[idx_T][0];
				imag = FFTxEz_T[idx_T][1];

				FFTxEz_T[idx_T][0] =  kx[j] * imag;
				FFTxEz_T[idx_T][1] = -kx[j] * real;

			}
		}
	}

	// Perform Inverse FFT.
	fftw_execute(FFTx_Ey_BAK_plan);
	fftw_execute(FFTx_Ez_BAK_plan);

	// Normalize the results of pseudo-spectral method.
	// Get diffx, diffy and diffz of H fields.
	#pragma omp parallel for \
		shared(	Nx, Ny, Nz, \
				diffxEy_re, \
				diffxEz_re, \
				diffxEy_T, \
				diffxEz_T) \
		private(i, j, k, idx, idx_T)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx_T = i + j*Nx + k*Ny*Nx;
				idx   = k + j*Nz + i*Ny*Nz;

				diffxEy_re[idx] = diffxEy_T[idx_T] / Nx;
				diffxEz_re[idx] = diffxEz_T[idx_T] / Nx;

			}
		}
	}

	fftw_destroy_plan(FFTx_Ey_FOR_plan);
	fftw_destroy_plan(FFTx_Ez_FOR_plan);
	fftw_destroy_plan(FFTx_Ey_BAK_plan);
	fftw_destroy_plan(FFTx_Ez_BAK_plan);

	fftw_free(FFTxEy_T);
	fftw_free(FFTxEz_T);
	free(diffxEy_T);
	free(diffxEz_T);

	return;
}

// Update H field.
void updateH(										\
	int		Nx,		int		Ny,		int		Nz,	\
	double  dt,										\
	double *Hx_re,		double *Hx_im,				\
	double *Hy_re,		double *Hy_im,				\
	double *Hz_re,		double *Hz_im,				\
	double *mu_HEE,		double *mu_EHH,				\
	double *mcon_HEE,	double *mcon_EHH,			\
	double *diffxEy_re, double *diffxEy_im,			\
	double *diffxEz_re, double *diffxEz_im,			\
	double *diffyEx_re, double *diffyEx_im,			\
	double *diffyEz_re, double *diffyEz_im,			\
	double *diffzEx_re, double *diffzEx_im,			\
	double *diffzEy_re, double *diffzEy_im			\
){
	/* MAIN UPDATE EQUATIONS */

	int i,j,k;
	int idx;

	double CHx1, CHx2;
	double CHy1, CHy2;
	double CHz1, CHz2;

	// Update H field.
	#pragma omp parallel for			\
		shared(	Nx, Ny, Nz, dt,		\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hx_re, Hx_im,			\
				Hy_re, Hy_im,			\
				Hz_re, Hz_im,			\
				diffxEy_re, diffxEy_im, \
				diffxEz_re, diffxEz_im,	\
				diffyEx_re, diffyEx_im, \
				diffyEz_re, diffyEz_im,	\
				diffzEx_re, diffzEx_im, \
				diffzEy_re, diffzEy_im)	\
		private(i, j, k, idx, CHx1, CHx2, CHy1, CHy2, CHz1, CHz2)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				idx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHx1 =	(2.*mu_HEE[idx] - mcon_HEE[idx]*dt) / (2.*mu_HEE[idx] + mcon_HEE[idx]*dt);
				CHy1 =	(2.*mu_EHH[idx] - mcon_EHH[idx]*dt) / (2.*mu_EHH[idx] + mcon_EHH[idx]*dt);
				CHz1 =	(2.*mu_EHH[idx] - mcon_EHH[idx]*dt) / (2.*mu_EHH[idx] + mcon_EHH[idx]*dt);

				CHx2 =	(-2*dt) / (2.*mu_HEE[idx] + mcon_HEE[idx]*dt);
				CHy2 =	(-2*dt) / (2.*mu_EHH[idx] + mcon_EHH[idx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[idx] + mcon_EHH[idx]*dt);

				// Update Hx
				Hx_re[idx] = CHx1 * Hx_re[idx] + CHx2 * (diffyEz_re[idx] - diffzEy_re[idx]);
				Hx_im[idx] = CHx1 * Hx_im[idx] + CHx2 * (diffyEz_im[idx] - diffzEy_im[idx]);

				// Update Hy
				Hy_re[idx] = CHy1 * Hy_re[idx] + CHy2 * (diffzEx_re[idx] - diffxEz_re[idx]);
				Hy_im[idx] = CHy1 * Hy_im[idx] + CHy2 * (diffzEx_im[idx] - diffxEz_im[idx]);

				// Update Hz
				Hz_re[idx] = CHz1 * Hz_re[idx] + CHz2 * (diffxEy_re[idx] - diffyEx_re[idx]);
				Hz_im[idx] = CHz1 * Hz_im[idx] + CHz2 * (diffxEy_im[idx] - diffyEx_im[idx]);
			}
		}
	}

	return;
}

void get_deriv_z_H(
	int Nx, int Ny, int Nz,
	double* Hx_re,
	double* Hy_re,
	double* kz,
	double* diffzHx_re,
	double* diffzHy_re
){

	// int for index
	int i,j,k;
	int idx, idx_i, idx_T;
	int nthreads = omp_get_max_threads();
	double real, imag;

	// initialize multi-threaded fftw3.
	fftw_init_threads();

	// initialize nthreaded fftw3 plan.
	fftw_plan_with_nthreads(nthreads);

	fftw_complex *FFTzHx   = fftw_alloc_complex(Nx*Ny*(Nz/2+1));
	fftw_complex *FFTzHy   = fftw_alloc_complex(Nx*Ny*(Nz/2+1));

	// Set Plans for real FFT along z axis.
	int rankz = 1;
	int nz[]  = {Nz};
	int howmanyz = Nx*Ny;
	int istridez = 1, ostridez = 1;
	int idistz = Nz, odistz = Nz/2+1;
	const int *inembedz = NULL, *onembedz = NULL;

	fftw_plan FFTz_Hx_FOR_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, Hx_re, inembedz, istridez, idistz, \
														FFTzHx, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	fftw_plan FFTz_Hy_FOR_plan = fftw_plan_many_dft_r2c(rankz, nz, howmanyz, Hy_re, inembedz, istridez, idistz, \
														FFTzHy, onembedz, ostridez, odistz, FFTW_ESTIMATE);

	// Set Plans for inverse real FFT along z axis.
	int rankbz = 1;
	int nbz[]  = {Nz};
	int howmanybz = Nx*Ny;
	int istridebz = 1, ostridebz = 1;
	int idistbz = Nz/2+1, odistbz = Nz;
	const int *inembedbz = NULL, *onembedbz = NULL;

	fftw_plan FFTz_Hx_BAK_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTzHx, inembedbz, istridebz, idistbz, \
														diffzHx_re, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	fftw_plan FFTz_Hy_BAK_plan = fftw_plan_many_dft_c2r(rankbz, nbz, howmanybz, FFTzHy, inembedbz, istridebz, idistbz, \
														diffzHy_re, onembedbz, ostridebz, odistbz, FFTW_ESTIMATE);

	// Perform 1D FFT along z-axis.
	fftw_execute(FFTz_Hx_FOR_plan);
	fftw_execute(FFTz_Hy_FOR_plan);

	// Multiply ikz.
	#pragma omp parallel for \
		shared(Nx, Ny, Nz, FFTzHx, FFTzHy, kz) \
		private(i, j, k, idx, real, imag)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz/2+1); k++){

				idx = k + j*(Nz/2+1) + i*(Nz/2+1)*Ny;

				real = FFTzHx[idx][0];
				imag = FFTzHx[idx][1];

				FFTzHx[idx][0] = kz[k] * imag;
				FFTzHx[idx][1] = -kz[k] * real;

				//FFTzHx[idx][0] = -kz[k] * imag;
				//FFTzHx[idx][1] =  kz[k] * real;

				real = FFTzHy[idx][0];
				imag = FFTzHy[idx][1];

				FFTzHy[idx][0] = kz[k] * imag;
				FFTzHy[idx][1] = -kz[k] * real;
				
				//FFTzHy[idx][0] = -kz[k] * imag;
				//FFTzHy[idx][1] =  kz[k] * real;
				
			}
		}
	}

	fftw_execute(FFTz_Hx_BAK_plan);
	fftw_execute(FFTz_Hy_BAK_plan);

	// Normalize the results of pseudo-spectral method
	#pragma omp parallel for \
		shared(	Nx, Ny, Nz, \
				diffzHx_re, \
				diffzHy_re) \
		private(i, j, k, idx)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;

				diffzHx_re[idx] = diffzHx_re[idx] / Nz;
				diffzHy_re[idx] = diffzHy_re[idx] / Nz;

			}
		}
	}

	// Destroy the plan and free the memory.
	fftw_destroy_plan(FFTz_Hx_FOR_plan);
	fftw_destroy_plan(FFTz_Hy_FOR_plan);
	fftw_destroy_plan(FFTz_Hx_BAK_plan);
	fftw_destroy_plan(FFTz_Hy_BAK_plan);
	fftw_free(FFTzHx);
	fftw_free(FFTzHy);

	return;
}

void get_deriv_y_H(
	int Nx, int Ny, int Nz,
	double* Hx_re,
	double* Hz_re,
	double* ky,
	double* diffyHx_re,
	double* diffyHz_re
){

	// int for index
	int i,j,k;
	int idx, idx_i, idx_T;
	int nthreads = omp_get_max_threads();
	double real, imag;

	// initialize multi-threaded fftw3.
	fftw_init_threads();

	// initialize nthreaded fftw3 plan.
	fftw_plan_with_nthreads(nthreads);

	// Memory allocation for transpose of the field data.
	double* diffyHx_T = (double*) malloc(Nx*Ny*Nz*sizeof(double));
	double* diffyHz_T = (double*) malloc(Nx*Ny*Nz*sizeof(double));
	fftw_complex *FFTyHx_T = fftw_alloc_complex(Nx*(Ny/2+1)*Nz);
	fftw_complex *FFTyHz_T = fftw_alloc_complex(Nx*(Ny/2+1)*Nz);

	// Set Plans for real FFT along y axis.
	int ranky = 1;
	int ny[]  = {Ny};
	int howmanyy = Nx*Nz;
	int istridey = 1, ostridey = 1;
	int idisty = Ny, odisty = Ny/2+1;
	const int *inembedy = NULL, *onembedy = NULL;

	fftw_plan FFTy_Hx_FOR_plan = fftw_plan_many_dft_r2c(ranky, ny, howmanyy, diffyHx_T, inembedy, istridey, idisty, \
														FFTyHx_T, onembedy, ostridey, odisty, FFTW_ESTIMATE);

	fftw_plan FFTy_Hz_FOR_plan = fftw_plan_many_dft_r2c(ranky, ny, howmanyy, diffyHz_T, inembedy, istridey, idisty, \
														FFTyHz_T, onembedy, ostridey, odisty, FFTW_ESTIMATE);

	// Set Plans for inverse real FFT along y axis.
	int rankby = 1;
	int nby[]  = {Ny};
	int howmanyby = Nx*Nz;
	int istrideby = 1, ostrideby = 1;
	int idistby = Ny/2+1, odistby = Ny;
	const int *inembedby = NULL, *onembedby = NULL;

	fftw_plan FFTy_Hx_BAK_plan = fftw_plan_many_dft_c2r(rankby, nby, howmanyby, FFTyHx_T, inembedby, istrideby, idistby, \
																	diffyHx_T, onembedby, ostrideby, odistby, FFTW_ESTIMATE);

	fftw_plan FFTy_Hz_BAK_plan = fftw_plan_many_dft_c2r(rankby, nby, howmanyby, FFTyHz_T, inembedby, istrideby, idistby, \
																	diffyHz_T, onembedby, ostrideby, odistby, FFTW_ESTIMATE);

	// Transpose y and z axis of the Hx and Hz to get y-derivatives of them.
	#pragma omp parallel for \
		shared(Nx, Ny, Nz, diffyHx_T, diffyHz_T, Hx_re, Hz_re) \
		private(i, j, k, idx, idx_T)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx   = k + j*Nz + i*Nz*Ny; // (x,y,z)
				idx_T = j + k*Ny + i*Nz*Ny; // (x,z,y)

				diffyHx_T[idx] = Hx_re[idx_T];
				diffyHz_T[idx] = Hz_re[idx_T];

			}
		}
	}

	// Perform 1D FFT along z-axis.
	fftw_execute(FFTy_Hx_FOR_plan);
	fftw_execute(FFTy_Hz_FOR_plan);

	// Multiply iky.
	#pragma omp parallel for \
		shared(Nx, Ny, Nz, FFTyHx_T, FFTyHz_T, ky) \
		private(i, j, k, idx_T, real, imag)
	for(i=0; i < Nx; i++){
		for(k=0; k < Nz; k++){
			for(j=0; j < (Ny/2+1); j++){

				idx_T = j + k*(Ny/2+1) + i*Nz*(Ny/2+1);

				real = FFTyHx_T[idx_T][0];
				imag = FFTyHx_T[idx_T][1];

				FFTyHx_T[idx_T][0] = ky[j] * imag;
				FFTyHx_T[idx_T][1] = -ky[j] * real;

				//FFTyHx_T[idx_T][0] = -ky[j] * imag;
				//FFTyHx_T[idx_T][1] =  ky[j] * real;

				real = FFTyHz_T[idx_T][0];
				imag = FFTyHz_T[idx_T][1];

				FFTyHz_T[idx_T][0] = ky[j] * imag;
				FFTyHz_T[idx_T][1] = -ky[j] * real;

				//FFTyHz_T[idx_T][0] = -ky[j] * imag;
				//FFTyHz_T[idx_T][1] =  ky[j] * real;

			}
		}
	}

	// Perform Inverse FFT.
	fftw_execute(FFTy_Hx_BAK_plan);
	fftw_execute(FFTy_Hz_BAK_plan);

	// Normalize the results of pseudo-spectral method
	// Get diffx, diffy, diffz of H field
	#pragma omp parallel for \
		shared(	Nx, Ny, Nz, \
				diffyHx_re, \
				diffyHz_re,	\
				diffyHx_T, \
				diffyHz_T) \
		private(i, j, k, idx, idx_i, idx_T)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;
				idx_T = (j  ) + (k  )*Ny + (i  )*Nz*Ny;

				diffyHx_re[idx] = diffyHx_T[idx_T] / Ny;
				diffyHz_re[idx] = diffyHz_T[idx_T] / Ny;

			}
		}
	}

	fftw_destroy_plan(FFTy_Hx_FOR_plan);
	fftw_destroy_plan(FFTy_Hz_FOR_plan);
	fftw_destroy_plan(FFTy_Hx_BAK_plan);
	fftw_destroy_plan(FFTy_Hz_BAK_plan);
	fftw_free(FFTyHx_T);
	fftw_free(FFTyHz_T);
	free(diffyHx_T);
	free(diffyHz_T);

	return;
}

void get_deriv_x_H(
	int Nx, int Ny, int Nz,
	double dx,
	double* Hy_re,
	double* Hz_re,
	double* diffxHy_re,
	double* diffxHz_re
){

	int i, j, k, idx, idx_i;

	#pragma omp parallel for \
		shared(	Nx, Ny, Nz, dx, \
				Hy_re, \
				Hz_re, \
				diffxHy_re, \
				diffxHz_re)	\
		private(i, j, k, idx, idx_i)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx   = (k  ) + (j  )*Nz + (i  )*Nz*Ny;
				idx_i = (k  ) + (j  )*Nz + (i-1)*Nz*Ny;

				if (i > 0){
					diffxHy_re[idx] = (Hy_re[idx] - Hy_re[idx_i]) / dx;
					diffxHz_re[idx] = (Hz_re[idx] - Hz_re[idx_i]) / dx;
				}		
			}
		}
	}

	return;
}

void updateE(
	int		Nx,		int		Ny,		int		Nz,	\
	double dt,										\
	double *Ex_re,		double *Ex_im,				\
	double *Ey_re,		double *Ey_im,				\
	double *Ez_re,		double *Ez_im,				\
	double *eps_HEE,	double *eps_EHH,			\
	double *econ_HEE,	double *econ_EHH,			\
	double *diffxHy_re, double *diffxHy_im,			\
	double *diffxHz_re, double *diffxHz_im,			\
	double *diffyHx_re, double *diffyHx_im,			\
	double *diffyHz_re, double *diffyHz_im,			\
	double *diffzHx_re, double *diffzHx_im,			\
	double *diffzHy_re, double *diffzHy_im			\
){
	/* MAIN UPDATE EQUATIONS */

	int i,j,k;
	int idx;

	double CEx1, CEx2;
	double CEy1, CEy2;
	double CEz1, CEz2;

	// Update Ex.
	#pragma omp parallel for			\
		shared(	Nx, Ny, Nz, dt,		\
				eps_EHH,	eps_HEE,	\
				econ_EHH,	econ_HEE,	\
				Ex_re,		Ex_im,		\
				Ey_re,		Ey_im,		\
				Ez_re,		Ez_im,		\
				diffxHy_re, diffxHy_im,	\
				diffxHz_re, diffxHz_im,	\
				diffyHx_re, diffyHx_im,	\
				diffyHz_re, diffyHz_im,	\
				diffzHx_re, diffzHx_im,	\
				diffzHy_re, diffzHy_im)	\
		private(i, j, k, idx, CEx1, CEx2, CEy1, CEy2, CEz1, CEz2)
	for(i=0; i < Nx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				idx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEx1 = (2.*eps_EHH[idx] - econ_EHH[idx]*dt) / (2.*eps_EHH[idx] + econ_EHH[idx]*dt);
				CEy1 = (2.*eps_HEE[idx] - econ_HEE[idx]*dt) / (2.*eps_HEE[idx] + econ_HEE[idx]*dt);
				CEz1 = (2.*eps_HEE[idx] - econ_HEE[idx]*dt) / (2.*eps_HEE[idx] + econ_HEE[idx]*dt);

				CEx2 =	(2.*dt) / (2.*eps_EHH[idx] + econ_EHH[idx]*dt);
				CEy2 =	(2.*dt) / (2.*eps_HEE[idx] + econ_HEE[idx]*dt);
				CEz2 =	(2.*dt) / (2.*eps_HEE[idx] + econ_HEE[idx]*dt);

				// PEC condition.
				if(eps_EHH[idx] > 1e3){
					CEx1 = 0.;
					CEx2 = 0.;
				}

				if(eps_HEE[idx] > 1e3){
					CEy1 = 0.;
					CEy2 = 0.;
					CEz1 = 0.;
					CEz2 = 0.;
				}

				// Update Ex.
				Ex_re[idx] = CEx1 * Ex_re[idx] + CEx2 * (diffyHz_re[idx] - diffzHy_re[idx]);
				Ex_im[idx] = CEx1 * Ex_im[idx] + CEx2 * (diffyHz_im[idx] - diffzHy_im[idx]);
		
				// Update Ey.
				Ey_re[idx] = CEy1 * Ey_re[idx] + CEy2 * (diffzHx_re[idx] - diffxHz_re[idx]);
				Ey_im[idx] = CEy1 * Ey_im[idx] + CEy2 * (diffzHx_im[idx] - diffxHz_im[idx]);

				// Update Ez.
				Ez_re[idx] = CEz1 * Ez_re[idx] + CEz2 * (diffxHy_re[idx] - diffyHx_re[idx]);
				Ez_im[idx] = CEz1 * Ez_im[idx] + CEz2 * (diffxHy_im[idx] - diffyHx_im[idx]);

			}		
		}
	}

	return;
}
