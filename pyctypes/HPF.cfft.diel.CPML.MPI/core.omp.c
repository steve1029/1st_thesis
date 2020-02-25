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
	
	myidx  : 1 dimensional index of elements where its index in 3D is (i  , j  , k  ).
	i_myidx: 1 dimensional index of elements where its index in 3D is (i+1, j  , k  ).
	myidx_i: 1 dimensional index of elements where its index in 3D is (i-1, j  , k  ).
	myidx_0: 1 dimensional index of elements where its index in 3D is (0  , j  , k  ).

	ex)
		myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
		i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
		myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
		myidx_0 = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;

*/

/***********************************************************************************/
/******************************** FUNCTION DECLARATION *****************************/
/***********************************************************************************/

// Get derivatives of H field in the first rank.
void get_diff_of_H_rank_F(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double	dt,			double  dx,		double  dy,		double dz,	\
	double *ky,			double *kz,									\
	double *Hx_re,		double *Hx_im,								\
	double *Hy_re,		double *Hy_im,								\
	double *Hz_re,		double *Hz_im,								\
	double *diffxHy_re, double *diffxHy_im,							\
	double *diffxHz_re, double *diffxHz_im,							\
	double *diffyHx_re, double *diffyHx_im,							\
	double *diffyHz_re, double *diffyHz_im,							\
	double *diffzHx_re, double *diffzHx_im,							\
	double *diffzHy_re, double *diffzHy_im							\
);

// Get derivatives of H field in the middle and last rank.
void get_diff_of_H_rankML(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double	dt,			double  dx,		double  dy,		double dz,	\
	double *ky,			double *kz,									\
	double *recvHyfirst_re,	double *recvHyfirst_im,					\
	double *recvHzfirst_re,	double *recvHzfirst_im,					\
	double *Hx_re,		double *Hx_im,								\
	double *Hy_re,		double *Hy_im,								\
	double *Hz_re,		double *Hz_im,								\
	double *diffxHy_re, double *diffxHy_im,							\
	double *diffxHz_re, double *diffxHz_im,							\
	double *diffyHx_re, double *diffyHx_im,							\
	double *diffyHz_re, double *diffyHz_im,							\
	double *diffzHx_re, double *diffzHx_im,							\
	double *diffzHy_re, double *diffzHy_im							\
);

// Update E field
void updateE(
	int		myNx,		int		Ny,		int		Nz,	\
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

//Get derivatives of E field in the first and middle rank.
void get_diff_of_E_rankFM(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,			double  dx,		double	dy,		double dz,	\
	double *ky,			double *kz,									\
	double *recvEylast_re,	double *recvEylast_im,					\
	double *recvEzlast_re,	double *recvEzlast_im,					\
	double *Ex_re,		double *Ex_im,								\
	double *Ey_re,		double *Ey_im,								\
	double *Ez_re,		double *Ez_im,								\
	double *diffxEy_re, double *diffxEy_im,							\
	double *diffxEz_re, double *diffxEz_im,							\
	double *diffyEx_re, double *diffyEx_im,							\
	double *diffyEz_re, double *diffyEz_im,							\
	double *diffzEx_re, double *diffzEx_im,							\
	double *diffzEy_re, double *diffzEy_im							\
);

//Get derivatives of E field in the last rank.
void get_diff_of_E_rank_L(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,			double  dx,		double	dy,		double dz,	\
	double *ky,			double *kz,									\
	double *Ex_re,		double *Ex_im,								\
	double *Ey_re,		double *Ey_im,								\
	double *Ez_re,		double *Ez_im,								\
	double *diffxEy_re, double *diffxEy_im,							\
	double *diffxEz_re, double *diffxEz_im,							\
	double *diffyEx_re, double *diffyEx_im,							\
	double *diffyEz_re, double *diffyEz_im,							\
	double *diffzEx_re, double *diffzEx_im,							\
	double *diffzEy_re, double *diffzEy_im							\
);

// Update H field.
void updateH(										\
	int		myNx,		int		Ny,		int		Nz,	\
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


/***********************************************************************************/
/******************************** FUNCTION DESCRIPTION *****************************/
/***********************************************************************************/

// Get derivatives of E field in the first rank.
void get_diff_of_H_rank_F(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double	dt,			double  dx,		double  dy,		double dz,	\
	double *ky,			double *kz,									\
	double *Hx_re,		double *Hx_im,								\
	double *Hy_re,		double *Hy_im,								\
	double *Hz_re,		double *Hz_im,								\
	double *diffxHy_re, double *diffxHy_im,							\
	double *diffxHz_re, double *diffxHz_im,							\
	double *diffyHx_re, double *diffyHx_im,							\
	double *diffyHz_re, double *diffyHz_im,							\
	double *diffzHx_re, double *diffzHx_im,							\
	double *diffzHy_re, double *diffzHy_im							\
){

	// initialize multi-threaded fftw3.
	fftw_init_threads();

	// int for index
	int i,j,k;
	int myidx, myidx_i;

	fftw_complex *diffyHx = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffzHx = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffzHy = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffyHz = fftw_alloc_complex(myNx * Ny * Nz);

	// Initialize diff* arrays by copying Hx, Hy and Hz to them.
	#pragma omp parallel for					\
		shared(myNx, Ny, Nz, diffyHx, diffzHy, diffyHz, Hx_re, Hx_im, Hy_re, Hy_im, Hz_re, Hz_im)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				//diffyHx[myidx] = Hx_re[myidx] + I * Hx_im[myidx];
				//diffzHy[myidx] = Hy_re[myidx] + I * Hy_im[myidx];
				//diffyHz[myidx] = Hz_re[myidx] + I * Hz_im[myidx];

				diffyHx[myidx][0] = Hx_re[myidx];
				diffyHx[myidx][1] = Hx_im[myidx];

				diffzHy[myidx][0] = Hy_re[myidx];
				diffzHy[myidx][1] = Hy_im[myidx];

				diffyHz[myidx][0] = Hz_re[myidx];
				diffyHz[myidx][1] = Hz_im[myidx];

			}
		}
	}

	// Set FFT parameters.
	int rank = 2;
	int n[]  = {Ny,Nz};
	const int *inembed = NULL, *onembed = NULL;
	int istride = 1, ostride = 1;
	int idist = Ny*Nz, odist = Ny*Nz;
	int howmany = myNx;

	// initialize nthreaded fftw3 plan.
	int nthreads = omp_get_max_threads();
	fftw_plan_with_nthreads(nthreads);

	// Setup FORWARD plans.
	fftw_plan FFT2D_diffyHx_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyHx, inembed, istride, idist, \
											diffyHx, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzHy_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzHy, inembed, istride, idist, \
											diffzHy, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffyHz_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyHz, inembed, istride, idist, \
											diffyHz, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	// Setup BACKWARD plans.
	fftw_plan FFT2D_diffyHx_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyHx, inembed, istride, idist, \
											diffyHx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzHx_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzHx, inembed, istride, idist, \
											diffzHx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzHy_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzHy, inembed, istride, idist, \
											diffzHy, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffyHz_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyHz, inembed, istride, idist, \
											diffyHz, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Perform 2D FFT along y and z-axis.
	fftw_execute(FFT2D_diffyHx_FORWARD_plan);
	fftw_execute(FFT2D_diffzHy_FORWARD_plan);
	fftw_execute(FFT2D_diffyHz_FORWARD_plan);

	// Copy FFT2D_diffyHx to FFT2D_diffzHx.
	#pragma omp parallel for					\
		shared(myNx, Ny, Nz, diffzHx, diffyHx)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				diffzHx[myidx][0] = diffyHx[myidx][0];
				diffzHx[myidx][1] = diffyHx[myidx][1];

			}
		}
	}

	// Multiply iky and ikz.
	double real, imag;

	#pragma omp parallel for												\
		shared(myNx, Ny, Nz, diffyHx, diffzHx, diffzHy, diffyHz, ky, kz)	\
		private(i,j,k, myidx, real, imag)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				real = diffyHx[myidx][0];
				imag = diffyHx[myidx][1];

				diffyHx[myidx][0] = -ky[j] * imag;
				diffyHx[myidx][1] =  ky[j] * real;

				real = diffzHx[myidx][0];
				imag = diffzHx[myidx][1];

				diffzHx[myidx][0] = -kz[k] * imag;
				diffzHx[myidx][1] =  kz[k] * real;

				real = diffzHy[myidx][0];
				imag = diffzHy[myidx][1];

				diffzHy[myidx][0] = -kz[k] * imag;
				diffzHy[myidx][1] =  kz[k] * real;

				real = diffyHz[myidx][0];
				imag = diffyHz[myidx][1];

				diffyHz[myidx][0] = -ky[j] * imag;
				diffyHz[myidx][1] =  ky[j] * real;

			}
		}
	}

	// Perform Inverse FFT.
	fftw_execute(FFT2D_diffyHx_BACKWARD_plan);
	fftw_execute(FFT2D_diffzHx_BACKWARD_plan);
	fftw_execute(FFT2D_diffzHy_BACKWARD_plan);
	fftw_execute(FFT2D_diffyHz_BACKWARD_plan);

	// Normalize the results of pseudo-spectral method
	// Get diffx, diffy, diffz of H field
	#pragma omp parallel for			\
		shared(	myNx, Ny, Nz, dx,		\
				Hy_re,		Hy_im,		\
				Hz_re,		Hz_im,		\
				diffxHy_re, diffxHy_im, \
				diffxHz_re, diffxHz_im, \
				diffyHx_re, diffyHx_im, \
				diffyHz_re, diffyHz_im, \
				diffzHx_re, diffzHx_im, \
				diffzHy_re, diffzHy_im, \
				diffyHx,	diffyHz,	\
				diffzHx,	diffzHy)	\
		private(i, j, k, myidx, myidx_i)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;

				if (i > 0){
					diffxHy_re[myidx] = (Hy_re[myidx] - Hy_re[myidx_i]) / dx;
					diffxHy_im[myidx] = (Hy_im[myidx] - Hy_im[myidx_i]) / dx;

					diffxHz_re[myidx] = (Hz_re[myidx] - Hz_re[myidx_i]) / dx;
					diffxHz_im[myidx] = (Hz_im[myidx] - Hz_im[myidx_i]) / dx;
				}		

				diffyHx_re[myidx] = diffyHx[myidx][0] / (Ny*Nz);
				diffyHx_im[myidx] = diffyHx[myidx][1] / (Ny*Nz);

				diffyHz_re[myidx] = diffyHz[myidx][0] / (Ny*Nz);
				diffyHz_im[myidx] = diffyHz[myidx][1] / (Ny*Nz);

				diffzHx_re[myidx] = diffzHx[myidx][0] / (Ny*Nz);
				diffzHx_im[myidx] = diffzHx[myidx][1] / (Ny*Nz);

				diffzHy_re[myidx] = diffzHy[myidx][0] / (Ny*Nz);
				diffzHy_im[myidx] = diffzHy[myidx][1] / (Ny*Nz);

				/*
				diffyHx_im[myidx] = 0.;
				diffyHz_im[myidx] = 0.;
				diffzHx_im[myidx] = 0.;
				diffzHy_im[myidx] = 0.;
				*/
			}
		}
	}

	// Destroy the plan and free the memory.
	fftw_destroy_plan(FFT2D_diffyHx_FORWARD_plan);
	fftw_destroy_plan(FFT2D_diffzHy_FORWARD_plan);
	fftw_destroy_plan(FFT2D_diffyHz_FORWARD_plan);

	fftw_destroy_plan(FFT2D_diffyHx_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffzHx_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffzHy_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffyHz_BACKWARD_plan);

	fftw_free(diffyHx);
	fftw_free(diffzHx);
	fftw_free(diffzHy);
	fftw_free(diffyHz);

	return;
}

// Get derivatives of H field in the middle and last rank.
void get_diff_of_H_rankML(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double	dt,			double  dx,		double  dy,		double dz,	\
	double *ky,			double *kz,									\
	double *recvHyfirst_re,	double *recvHyfirst_im,					\
	double *recvHzfirst_re,	double *recvHzfirst_im,					\
	double *Hx_re,		double *Hx_im,								\
	double *Hy_re,		double *Hy_im,								\
	double *Hz_re,		double *Hz_im,								\
	double *diffxHy_re, double *diffxHy_im,							\
	double *diffxHz_re, double *diffxHz_im,							\
	double *diffyHx_re, double *diffyHx_im,							\
	double *diffyHz_re, double *diffyHz_im,							\
	double *diffzHx_re, double *diffzHx_im,							\
	double *diffzHy_re, double *diffzHy_im							\
){

	// initialize multi-threaded fftw3.
	fftw_init_threads();

	// int for index.
	int i,j,k;
	int myidx, myidx_i, myidx_0;

	fftw_complex *diffyHx = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffzHx = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffzHy = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffyHz = fftw_alloc_complex(myNx * Ny * Nz);

	// Initialize diff* arrays by copying Hx, Hy and Hz to them.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				//diffyHx[myidx] = Hx_re[myidx] + I * Hx_im[myidx];
				//diffzHy[myidx] = Hy_re[myidx] + I * Hy_im[myidx];
				//diffyHz[myidx] = Hz_re[myidx] + I * Hz_im[myidx];

				diffyHx[myidx][0] = Hx_re[myidx];
				diffyHx[myidx][1] = Hx_im[myidx];

				diffzHy[myidx][0] = Hy_re[myidx];
				diffzHy[myidx][1] = Hy_im[myidx];

				diffyHz[myidx][0] = Hz_re[myidx];
				diffyHz[myidx][1] = Hz_im[myidx];

			}
		}
	}

	// Set FFT parameters.
	int rank = 2;
	int n[]  = {Ny,Nz};
	const int *inembed = NULL, *onembed = NULL;
	int istride = 1, ostride = 1;
	int idist = Ny*Nz, odist = Ny*Nz;
	int howmany = myNx;

	int nthreads = omp_get_max_threads();
	fftw_plan_with_nthreads(nthreads);

	// Setup FORWARD plans.
	fftw_plan FFT2D_diffyHx_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyHx, inembed, istride, idist, \
											diffyHx, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzHy_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzHy, inembed, istride, idist, \
											diffzHy, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffyHz_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyHz, inembed, istride, idist, \
											diffyHz, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	// Setup BACKWARD plans.
	fftw_plan FFT2D_diffyHx_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyHx, inembed, istride, idist, \
											diffyHx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzHx_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzHx, inembed, istride, idist, \
											diffzHx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzHy_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzHy, inembed, istride, idist, \
											diffzHy, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffyHz_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyHz, inembed, istride, idist, \
											diffyHz, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Perform 2D FFT along y and z-axis.
	fftw_execute(FFT2D_diffyHx_FORWARD_plan);
	fftw_execute(FFT2D_diffzHy_FORWARD_plan);
	fftw_execute(FFT2D_diffyHz_FORWARD_plan);

	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffzHx, diffyHx)	\
		private(i, j, k, myidx)
	// Copy FFT2D_diffyHx to FFT2D_diffzHx.
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				diffzHx[myidx][0] = diffyHx[myidx][0];
				diffzHx[myidx][1] = diffyHx[myidx][1];

			}
		}
	}

	// Multiply iky and ikz.
	double real, imag;

	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffyHx, diffzHx, diffzHy, diffyHz, ky, kz)	\
		private(i, j, k, myidx, real, imag)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				real = diffyHx[myidx][0];
				imag = diffyHx[myidx][1];

				diffyHx[myidx][0] = -ky[j] * imag;
				diffyHx[myidx][1] =  ky[j] * real;

				real = diffzHx[myidx][0];
				imag = diffzHx[myidx][1];

				diffzHx[myidx][0] = -kz[k] * imag;
				diffzHx[myidx][1] =  kz[k] * real;

				real = diffzHy[myidx][0];
				imag = diffzHy[myidx][1];

				diffzHy[myidx][0] = -kz[k] * imag;
				diffzHy[myidx][1] =  kz[k] * real;

				real = diffyHz[myidx][0];
				imag = diffyHz[myidx][1];

				diffyHz[myidx][0] = -ky[j] * imag;
				diffyHz[myidx][1] =  ky[j] * real;

			}
		}
	}

	// Perform Inverse FFT.
	fftw_execute(FFT2D_diffyHx_BACKWARD_plan);
	fftw_execute(FFT2D_diffzHx_BACKWARD_plan);
	fftw_execute(FFT2D_diffzHy_BACKWARD_plan);
	fftw_execute(FFT2D_diffyHz_BACKWARD_plan);

	// Normalize the results.
	#pragma omp parallel for						\
		shared(	myNx, Ny, Nz, dx,					\
				Hy_re,			Hy_im,				\
				Hz_re,			Hz_im,				\
				recvHyfirst_re, recvHyfirst_im,		\
				recvHzfirst_re, recvHzfirst_im,		\
				diffxHy_re,		diffxHy_im,			\
				diffxHz_re,		diffxHz_im,			\
				diffyHx_re,		diffyHx_im,			\
				diffyHz_re,		diffyHz_im,			\
				diffzHx_re,		diffzHx_im,			\
				diffzHy_re,		diffzHy_im,			\
				diffyHx,		diffyHz,			\
				diffzHx,		diffzHy)			\
		private(i, j, k, myidx, myidx_i, myidx_0)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k) + (j) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k) + (j) * Nz + (i-1) * Nz * Ny;
				myidx_0 = (k) + (j) * Nz + (0  ) * Nz * Ny;

				if (i > 0){
					diffxHy_re[myidx] = (Hy_re[myidx] - Hy_re[myidx_i]) / dx;
					diffxHy_im[myidx] = (Hy_im[myidx] - Hy_im[myidx_i]) / dx;

					diffxHz_re[myidx] = (Hz_re[myidx] - Hz_re[myidx_i]) / dx;
					diffxHz_im[myidx] = (Hz_im[myidx] - Hz_im[myidx_i]) / dx;

				} else{
					diffxHy_re[myidx] = (Hy_re[myidx] - recvHyfirst_re[myidx]) / dx;
					diffxHy_im[myidx] = (Hy_im[myidx] - recvHyfirst_im[myidx]) / dx;

					diffxHz_re[myidx] = (Hz_re[myidx] - recvHzfirst_re[myidx]) / dx;
					diffxHz_im[myidx] = (Hz_im[myidx] - recvHzfirst_im[myidx]) / dx;
				}
		
				diffyHx_re[myidx] = diffyHx[myidx][0] / (Ny*Nz);
				diffyHx_im[myidx] = diffyHx[myidx][1] / (Ny*Nz);

				diffyHz_re[myidx] = diffyHz[myidx][0] / (Ny*Nz);
				diffyHz_im[myidx] = diffyHz[myidx][1] / (Ny*Nz);

				diffzHx_re[myidx] = diffzHx[myidx][0] / (Ny*Nz);
				diffzHx_im[myidx] = diffzHx[myidx][1] / (Ny*Nz);

				diffzHy_re[myidx] = diffzHy[myidx][0] / (Ny*Nz);
				diffzHy_im[myidx] = diffzHy[myidx][1] / (Ny*Nz);

				/*
				diffyHx_im[myidx] = 0.;
				diffyHz_im[myidx] = 0.;
				diffzHx_im[myidx] = 0.;
				diffzHy_im[myidx] = 0.;
				*/
			}
		}
	}

	fftw_destroy_plan(FFT2D_diffyHx_FORWARD_plan);
	fftw_destroy_plan(FFT2D_diffzHy_FORWARD_plan);
	fftw_destroy_plan(FFT2D_diffyHz_FORWARD_plan);

	fftw_destroy_plan(FFT2D_diffyHx_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffzHx_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffzHy_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffyHz_BACKWARD_plan);

	fftw_free(diffyHx);
	fftw_free(diffzHx);
	fftw_free(diffzHy);
	fftw_free(diffyHz);

	return;
}

void updateE(
	int		myNx,		int		Ny,		int		Nz,	\
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
	int myidx;

	double CEx1, CEx2;
	double CEy1, CEy2;
	double CEz1, CEz2;

	// Update Ex.
	#pragma omp parallel for			\
		shared(	myNx, Ny, Nz, dt,		\
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
		private(i, j, k, myidx, CEx1, CEx2, CEy1, CEy2, CEz1, CEz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEx1 = (2.*eps_EHH[myidx] - econ_EHH[myidx]*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEy1 = (2.*eps_HEE[myidx] - econ_HEE[myidx]*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz1 = (2.*eps_HEE[myidx] - econ_HEE[myidx]*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				CEx2 =	(2.*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEy2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// PEC condition.
				if(eps_EHH[myidx] > 1e3){
					CEx1 = 0.;
					CEx2 = 0.;
				}

				if(eps_HEE[myidx] > 1e3){
					CEy1 = 0.;
					CEy2 = 0.;
					CEz1 = 0.;
					CEz2 = 0.;
				}

				// Update Ex.
				Ex_re[myidx] = CEx1 * Ex_re[myidx] + CEx2 * (diffyHz_re[myidx] - diffzHy_re[myidx]);
				Ex_im[myidx] = CEx1 * Ex_im[myidx] + CEx2 * (diffyHz_im[myidx] - diffzHy_im[myidx]);
		
				// Update Ey.
				Ey_re[myidx] = CEy1 * Ey_re[myidx] + CEy2 * (diffzHx_re[myidx] - diffxHz_re[myidx]);
				Ey_im[myidx] = CEy1 * Ey_im[myidx] + CEy2 * (diffzHx_im[myidx] - diffxHz_im[myidx]);

				// Update Ez.
				Ez_re[myidx] = CEz1 * Ez_re[myidx] + CEz2 * (diffxHy_re[myidx] - diffyHx_re[myidx]);
				Ez_im[myidx] = CEz1 * Ez_im[myidx] + CEz2 * (diffxHy_im[myidx] - diffyHx_im[myidx]);

			}		
		}
	}

	return;
}

// Get derivatives of E field in the first and middle rank.
void get_diff_of_E_rankFM(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,			double  dx,		double	dy,		double dz,	\
	double *ky,			double *kz,									\
	double *recvEylast_re,	double *recvEylast_im,					\
	double *recvEzlast_re,	double *recvEzlast_im,					\
	double *Ex_re,		double *Ex_im,								\
	double *Ey_re,		double *Ey_im,								\
	double *Ez_re,		double *Ez_im,								\
	double *diffxEy_re, double *diffxEy_im,							\
	double *diffxEz_re, double *diffxEz_im,							\
	double *diffyEx_re, double *diffyEx_im,							\
	double *diffyEz_re, double *diffyEz_im,							\
	double *diffzEx_re, double *diffzEx_im,							\
	double *diffzEy_re, double *diffzEy_im							\
){

	// initialize multi-threaded fftw3.
	fftw_init_threads();

	fftw_complex *diffyEx = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffzEx = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffzEy = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffyEz = fftw_alloc_complex(myNx * Ny * Nz);

	// int for index
	int i,j,k;
	int myidx, i_myidx, myidx_0;

	// Initialize diff* arrays by copying Ex, Ey and Ez to them.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffyEx, diffzEy, diffyEz, Ex_re, Ex_im, Ey_re, Ey_im, Ez_re, Ez_im)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				//diffyHx[myidx] = Hx_re[myidx] + I * Hx_im[myidx];
				//diffzHy[myidx] = Hy_re[myidx] + I * Hy_im[myidx];
				//diffyHz[myidx] = Hz_re[myidx] + I * Hz_im[myidx];

				diffyEx[myidx][0] = Ex_re[myidx];
				diffyEx[myidx][1] = Ex_im[myidx];

				diffzEy[myidx][0] = Ey_re[myidx];
				diffzEy[myidx][1] = Ey_im[myidx];

				diffyEz[myidx][0] = Ez_re[myidx];
				diffyEz[myidx][1] = Ez_im[myidx];

			}
		}
	}

	// Set FFT parameters.
	int rank = 2;
	int n[]  = {Ny,Nz};
	const int *inembed = NULL, *onembed = NULL;
	int istride = 1, ostride = 1;
	int idist = Ny*Nz, odist = Ny*Nz;
	int howmany = myNx;

	int nthreads = omp_get_max_threads();
	fftw_plan_with_nthreads(nthreads);

	// Setup FORWARD plans.
	fftw_plan FFT2D_diffyEx_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyEx, inembed, istride, idist, \
											diffyEx, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzEy_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzEy, inembed, istride, idist, \
											diffzEy, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffyEz_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyEz, inembed, istride, idist, \
											diffyEz, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	// Setup BACKWARD plans.
	fftw_plan FFT2D_diffyEx_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyEx, inembed, istride, idist, \
											diffyEx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzEx_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzEx, inembed, istride, idist, \
											diffzEx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzEy_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzEy, inembed, istride, idist, \
											diffzEy, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffyEz_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyEz, inembed, istride, idist, \
											diffyEz, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Perform 2D FFT along y and z-axis.
	fftw_execute(FFT2D_diffyEx_FORWARD_plan);
	fftw_execute(FFT2D_diffzEy_FORWARD_plan);
	fftw_execute(FFT2D_diffyEz_FORWARD_plan);

	// Copy FFT2D_diffyEx to FFT2D_diffzEx.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffzEx, diffyEx)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				//diffzEx[myidx] = diffyEx[myidx];
				diffzEx[myidx][0] = diffyEx[myidx][0];
				diffzEx[myidx][1] = diffyEx[myidx][1];

			}
		}
	}

	// Multiply iky and ikz.
	double real, imag;
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffyEx, diffzEx, diffzEy, diffyEz, ky, kz)	\
		private(i, j, k, real, imag)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				real = diffyEx[myidx][0];
				imag = diffyEx[myidx][1];

				diffyEx[myidx][0] = -ky[j] * imag;
				diffyEx[myidx][1] =  ky[j] * real;

				real = diffzEx[myidx][0];
				imag = diffzEx[myidx][1];

				diffzEx[myidx][0] = -kz[k] * imag;
				diffzEx[myidx][1] =  kz[k] * real;

				real = diffzEy[myidx][0];
				imag = diffzEy[myidx][1];

				diffzEy[myidx][0] = -kz[k] * imag;
				diffzEy[myidx][1] =  kz[k] * real;

				real = diffyEz[myidx][0];
				imag = diffyEz[myidx][1];

				diffyEz[myidx][0] = -ky[j] * imag;
				diffyEz[myidx][1] =  ky[j] * real;

			}
		}
	}

	// Perform Inverse FFT.
	fftw_execute(FFT2D_diffyEx_BACKWARD_plan);
	fftw_execute(FFT2D_diffzEx_BACKWARD_plan);
	fftw_execute(FFT2D_diffzEy_BACKWARD_plan);
	fftw_execute(FFT2D_diffyEz_BACKWARD_plan);

	// Normalize the results of pseudo-spectral method.
	// Get diffx, diffy and diffz of H fields.
	#pragma omp parallel for					\
		shared(	myNx, Ny, Nz, dx,				\
				Ey_re,		Ey_im,				\
				Ez_re,		Ez_im,				\
				recvEylast_re, recvEylast_im,	\
				recvEzlast_re, recvEzlast_im,	\
				diffxEy_re, diffxEy_im,			\
				diffxEz_re, diffxEz_im,			\
				diffyEx_re, diffyEx_im,			\
				diffyEz_re, diffyEz_im,			\
				diffzEx_re, diffzEx_im,			\
				diffzEy_re, diffzEy_im,			\
				diffyEx,	diffyEz,			\
				diffzEx,	diffzEy)			\
		private(i, j, k, myidx, i_myidx, myidx_0)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
				myidx_0 = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;

				if(i < (myNx-1)){

					diffxEy_re[myidx] = (Ey_re[i_myidx] - Ey_re[myidx]) / dx;
					diffxEy_im[myidx] = (Ey_im[i_myidx] - Ey_im[myidx]) / dx;

					diffxEz_re[myidx] = (Ez_re[i_myidx] - Ez_re[myidx]) / dx;
					diffxEz_im[myidx] = (Ez_im[i_myidx] - Ez_im[myidx]) / dx;

				} else{		

					diffxEy_re[myidx] = (recvEylast_re[myidx_0] - Ey_re[myidx]) / dx;
					diffxEy_im[myidx] = (recvEylast_im[myidx_0] - Ey_im[myidx]) / dx;

					diffxEz_re[myidx] = (recvEzlast_re[myidx_0] - Ez_re[myidx]) / dx;
					diffxEz_im[myidx] = (recvEzlast_im[myidx_0] - Ez_im[myidx]) / dx;

				}

				diffyEx_re[myidx] = diffyEx[myidx][0] / (Ny*Nz);
				diffyEx_im[myidx] = diffyEx[myidx][1] / (Ny*Nz);

				diffyEz_re[myidx] = diffyEz[myidx][0] / (Ny*Nz);
				diffyEz_im[myidx] = diffyEz[myidx][1] / (Ny*Nz);

				diffzEx_re[myidx] = diffzEx[myidx][0] / (Ny*Nz);
				diffzEx_im[myidx] = diffzEx[myidx][1] / (Ny*Nz);

				diffzEy_re[myidx] = diffzEy[myidx][0] / (Ny*Nz);
				diffzEy_im[myidx] = diffzEy[myidx][1] / (Ny*Nz);

				/*
				diffyEx_im[myidx] = 0.;
				diffyEz_im[myidx] = 0.;
				diffzEx_im[myidx] = 0.;
				diffzEy_im[myidx] = 0.;
				*/
			}
		}
	}

	fftw_destroy_plan(FFT2D_diffyEx_FORWARD_plan);
	fftw_destroy_plan(FFT2D_diffzEy_FORWARD_plan);
	fftw_destroy_plan(FFT2D_diffyEz_FORWARD_plan);

	fftw_destroy_plan(FFT2D_diffyEx_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffzEx_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffzEy_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffyEz_BACKWARD_plan);

	fftw_free(diffyEx);
	fftw_free(diffzEx);
	fftw_free(diffzEy);
	fftw_free(diffyEz);

	return;
}

//Get derivatives of E field in the last rank.
void get_diff_of_E_rank_L(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,			double  dx,		double	dy,		double dz,	\
	double *ky,			double *kz,									\
	double *Ex_re,		double *Ex_im,								\
	double *Ey_re,		double *Ey_im,								\
	double *Ez_re,		double *Ez_im,								\
	double *diffxEy_re, double *diffxEy_im,							\
	double *diffxEz_re, double *diffxEz_im,							\
	double *diffyEx_re, double *diffyEx_im,							\
	double *diffyEz_re, double *diffyEz_im,							\
	double *diffzEx_re, double *diffzEx_im,							\
	double *diffzEy_re, double *diffzEy_im							\
){

	// Initialize multi-threaded fftw3.
	fftw_init_threads();

	fftw_complex *diffyEx = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffzEx = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffzEy = fftw_alloc_complex(myNx * Ny * Nz);
	fftw_complex *diffyEz = fftw_alloc_complex(myNx * Ny * Nz);

	// int for index
	int i,j,k;
	int myidx, i_myidx, myidx_0;

	// Initialize diff* arrays by copying Ex, Ey and Ez to them.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffyEx, diffzEy, diffyEz, Ex_re, Ex_im, Ey_re, Ey_im, Ez_re, Ez_im)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				//diffyHx[myidx] = Hx_re[myidx] + I * Hx_im[myidx];
				//diffzHy[myidx] = Hy_re[myidx] + I * Hy_im[myidx];
				//diffyHz[myidx] = Hz_re[myidx] + I * Hz_im[myidx];

				diffyEx[myidx][0] = Ex_re[myidx];
				diffyEx[myidx][1] = Ex_im[myidx];

				diffzEy[myidx][0] = Ey_re[myidx];
				diffzEy[myidx][1] = Ey_im[myidx];

				diffyEz[myidx][0] = Ez_re[myidx];
				diffyEz[myidx][1] = Ez_im[myidx];

			}
		}
	}

	// Set FFT parameters.
	int rank = 2;
	int n[]  = {Ny,Nz};
	const int *inembed = NULL, *onembed = NULL;
	int istride = 1, ostride = 1;
	int idist = Ny*Nz, odist = Ny*Nz;
	int howmany = myNx;

	int nthreads = omp_get_max_threads();
	fftw_plan_with_nthreads(nthreads);

	// Setup FORWARD plans.
	fftw_plan FFT2D_diffyEx_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyEx, inembed, istride, idist, \
											diffyEx, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzEy_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzEy, inembed, istride, idist, \
											diffzEy, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffyEz_FORWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyEz, inembed, istride, idist, \
											diffyEz, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

	// Setup BACKWARD plans.
	fftw_plan FFT2D_diffyEx_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyEx, inembed, istride, idist, \
											diffyEx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzEx_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzEx, inembed, istride, idist, \
											diffzEx, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffzEy_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffzEy, inembed, istride, idist, \
											diffzEy, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_plan FFT2D_diffyEz_BACKWARD_plan = fftw_plan_many_dft(rank, n, howmany, diffyEz, inembed, istride, idist, \
											diffyEz, onembed, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Perform 2D FFT along y and z-axis.
	fftw_execute(FFT2D_diffyEx_FORWARD_plan);
	fftw_execute(FFT2D_diffzEy_FORWARD_plan);
	fftw_execute(FFT2D_diffyEz_FORWARD_plan);

	// Copy FFT2D_diffyEx to FFT2D_diffzEx.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffzEx, diffyEx)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				//diffzEx[myidx] = diffyEx[myidx];
				diffzEx[myidx][0] = diffyEx[myidx][0];
				diffzEx[myidx][1] = diffyEx[myidx][1];

			}
		}
	}

	// Multiply iky and ikz.
	double real, imag;

	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffyEx, ky, diffzEx, kz, diffzEy, diffyEz)	\
		private(i, j, k, myidx, real, imag)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;

				real = diffyEx[myidx][0];
				imag = diffyEx[myidx][1];

				diffyEx[myidx][0] = -ky[j] * imag;
				diffyEx[myidx][1] =  ky[j] * real;

				real = diffzEx[myidx][0];
				imag = diffzEx[myidx][1];

				diffzEx[myidx][0] = -kz[k] * imag;
				diffzEx[myidx][1] =  kz[k] * real;

				real = diffzEy[myidx][0];
				imag = diffzEy[myidx][1];

				diffzEy[myidx][0] = -kz[k] * imag;
				diffzEy[myidx][1] =  kz[k] * real;

				real = diffyEz[myidx][0];
				imag = diffyEz[myidx][1];

				diffyEz[myidx][0] = -ky[j] * imag;
				diffyEz[myidx][1] =  ky[j] * real;

			}
		}
	}

	// Perform Inverse FFT.
	fftw_execute(FFT2D_diffyEx_BACKWARD_plan);
	fftw_execute(FFT2D_diffzEx_BACKWARD_plan);
	fftw_execute(FFT2D_diffzEy_BACKWARD_plan);
	fftw_execute(FFT2D_diffyEz_BACKWARD_plan);

	// Normalize the results.
	#pragma omp parallel for			\
		shared(	myNx, Ny, Nz, dx,		\
				Ey_re,		Ey_im,		\
				Ez_re,		Ez_im,		\
				diffxEy_re, diffxEy_im,	\
				diffxEz_re, diffxEz_im,	\
				diffyEx_re, diffyEx_im,	\
				diffyEz_re, diffyEz_im,	\
				diffzEx_re, diffzEx_im,	\
				diffzEy_re,	diffzEy_im,	\
				diffyEx,	diffyEz,	\
				diffzEx,	diffzEy)	\
		private(i, j, k, myidx, i_myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;

				if(i < (myNx-1)){

					diffxEy_re[myidx] = (Ey_re[i_myidx] - Ey_re[myidx]) / dx;
					diffxEy_im[myidx] = (Ey_im[i_myidx] - Ey_im[myidx]) / dx;

					diffxEz_re[myidx] = (Ez_re[i_myidx] - Ez_re[myidx]) / dx;
					diffxEz_im[myidx] = (Ez_im[i_myidx] - Ez_im[myidx]) / dx;

				}

				diffyEx_re[myidx] = diffyEx[myidx][0] / (Ny*Nz);
				diffyEx_im[myidx] = diffyEx[myidx][1] / (Ny*Nz);

				diffyEz_re[myidx] = diffyEz[myidx][0] / (Ny*Nz);
				diffyEz_im[myidx] = diffyEz[myidx][1] / (Ny*Nz);

				diffzEx_re[myidx] = diffzEx[myidx][0] / (Ny*Nz);
				diffzEx_im[myidx] = diffzEx[myidx][1] / (Ny*Nz);

				diffzEy_re[myidx] = diffzEy[myidx][0] / (Ny*Nz);
				diffzEy_im[myidx] = diffzEy[myidx][1] / (Ny*Nz);

				/*
				diffyEx_im[myidx] = 0.;
				diffyEz_im[myidx] = 0.;
				diffzEx_im[myidx] = 0.;
				diffzEy_im[myidx] = 0.;
				*/
			}
		}
	}

	fftw_destroy_plan(FFT2D_diffyEx_FORWARD_plan);
	fftw_destroy_plan(FFT2D_diffzEy_FORWARD_plan);
	fftw_destroy_plan(FFT2D_diffyEz_FORWARD_plan);

	fftw_destroy_plan(FFT2D_diffyEx_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffzEx_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffzEy_BACKWARD_plan);
	fftw_destroy_plan(FFT2D_diffyEz_BACKWARD_plan);

	fftw_free(diffyEx);
	fftw_free(diffzEx);
	fftw_free(diffzEy);
	fftw_free(diffyEz);

	return;
}

// Update H field.
void updateH(										\
	int		myNx,		int		Ny,		int		Nz,	\
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
	int myidx;

	double CHx1, CHx2;
	double CHy1, CHy2;
	double CHz1, CHz2;

	// Update Hx
	#pragma omp parallel for			\
		shared(	myNx, Ny, Nz, dt,		\
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
		private(i, j, k, myidx, CHx1, CHx2, CHy1, CHy2, CHz1, CHz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHx1 =	(2.*mu_HEE[myidx] - mcon_HEE[myidx]*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHy1 =	(2.*mu_EHH[myidx] - mcon_EHH[myidx]*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz1 =	(2.*mu_EHH[myidx] - mcon_EHH[myidx]*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				CHx2 =	(-2*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHy2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hx
				Hx_re[myidx] = CHx1 * Hx_re[myidx] + CHx2 * (diffyEz_re[myidx] - diffzEy_re[myidx]);
				Hx_im[myidx] = CHx1 * Hx_im[myidx] + CHx2 * (diffyEz_im[myidx] - diffzEy_im[myidx]);

				// Update Hy
				Hy_re[myidx] = CHy1 * Hy_re[myidx] + CHy2 * (diffzEx_re[myidx] - diffxEz_re[myidx]);
				Hy_im[myidx] = CHy1 * Hy_im[myidx] + CHy2 * (diffzEx_im[myidx] - diffxEz_im[myidx]);

				// Update Hz
				Hz_re[myidx] = CHz1 * Hz_re[myidx] + CHz2 * (diffxEy_re[myidx] - diffyEx_re[myidx]);
				Hz_im[myidx] = CHz1 * Hz_im[myidx] + CHz2 * (diffxEy_im[myidx] - diffyEx_im[myidx]);
			}
		}
	}

	return;
}
