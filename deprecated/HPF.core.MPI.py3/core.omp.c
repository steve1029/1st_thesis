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
	
	myidx  : 1 dimensional index of elements where its index in 3D is (i  ,j  ,k  ).
	i_myidx: 1 dimensional index of elements where its index in 3D is (i+1,j  ,k  ).
	myidx_i: 1 dimensional index of elements where its index in 3D is (i-1,j  ,k  ).

	ex)
		myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
		i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
		myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;

*/

// update E in the first rank.
void updateE_rank_F(												\
	int		myNx,	 int	 Ny,	int		Nz,						\
	double	dt,		 double  dx,	double  dy,		double dz,		\
	double *Ex_re,	 double *Ex_im,	double *Hx_re,	double *Hx_im,	\
	double *Ey_re,	 double *Ey_im,	double *Hy_re,	double *Hy_im,	\
	double *Ez_re,	 double *Ez_im,	double *Hz_re,	double *Hz_im,	\
	double *eps_HEE, double *eps_EHH,								\
	double *ky,		 double *kz										\
);

// update E in the middle and last rank.
void updateE_rankML(												\
	int	    myNx,	 int	 Ny,	int	    Nz,						\
	double  dt,		 double  dx,	double  dy,		double dz,		\
	double *Ex_re,	 double *Ex_im,	double *Hx_re,	double *Hx_im,	\
	double *Ey_re,	 double *Ey_im,	double *Hy_re,	double *Hy_im,	\
	double *Ez_re,	 double *Ez_im,	double *Hz_re,	double *Hz_im,	\
	double *eps_HEE, double *eps_EHH,								\
	double *ky,		 double *kz,									\
	double *recvHyfirst_re,	double *recvHyfirst_im,					\
	double *recvHzfirst_re,	double *recvHzfirst_im					\
);

// update H in the first and middle rank.
void updateH_rankFM(												\
	int		myNx,	 int	Ny,		int		Nz,						\
	double  dt,		 double  dx,	double	dy,		double dz,		\
	double *Ex_re,	 double *Ex_im,	double *Hx_re,	double *Hx_im,	\
	double *Ey_re,	 double *Ey_im,	double *Hy_re,	double *Hy_im,	\
	double *Ez_re,	 double *Ez_im,	double *Hz_re,	double *Hz_im,	\
	double *mu_HEE,	 double *mu_EHH,								\
	double *ky,		 double *kz,									\
	double *recvEylast_re,	double *recvEylast_im,					\
	double *recvEzlast_re,	double *recvEzlast_im					\
);

// update H in the last rank.
void updateH_rank_L(												\
	int		myNx,	 int	 Ny,	int		Nz,						\
	double  dt,		 double  dx,	double  dy,		double dz,		\
	double *Ex_re,	 double *Ex_im,	double *Hx_re,	double *Hx_im,	\
	double *Ey_re,	 double *Ey_im,	double *Hy_re,	double *Hy_im,	\
	double *Ez_re,	 double *Ez_im,	double *Hz_re,	double *Hz_im,	\
	double *mu_HEE,	 double *mu_EHH,								\
	double *ky,		 double *kz										\
);

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void updateE_rank_F(												\
	int		myNx,	 int	 Ny,	int		Nz,						\
	double	dt,		 double  dx,	double  dy,		double dz,		\
	double *Ex_re,	 double *Ex_im,	double *Hx_re,	double *Hx_im,	\
	double *Ey_re,	 double *Ey_im,	double *Hy_re,	double *Hy_im,	\
	double *Ez_re,	 double *Ez_im,	double *Hz_re,	double *Hz_im,	\
	double *eps_HEE, double *eps_EHH,								\
	double *ky,		 double *kz										\
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

	// Normalize the results.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffyHx, diffzHx, diffzHy, diffyHz)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;
		
				diffyHx[myidx][0] = diffyHx[myidx][0] / (Ny*Nz);
				diffyHx[myidx][1] = diffyHx[myidx][1] / (Ny*Nz);

				diffzHx[myidx][0] = diffzHx[myidx][0] / (Ny*Nz);
				diffzHx[myidx][1] = diffzHx[myidx][1] / (Ny*Nz);

				diffzHy[myidx][0] = diffzHy[myidx][0] / (Ny*Nz);
				diffzHy[myidx][1] = diffzHy[myidx][1] / (Ny*Nz);

				diffyHz[myidx][0] = diffyHz[myidx][0] / (Ny*Nz);
				diffyHz[myidx][1] = diffyHz[myidx][1] / (Ny*Nz);

			}
		}
	}

	double diffxHz_re, diffxHz_im;
	double diffxHy_re, diffxHy_im;

	// Update Ex.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, dt, Ex_re, Ex_im, eps_EHH, diffyHz, diffzHy)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				Ex_re[myidx] = Ex_re[myidx] + (dt/eps_EHH[myidx]) * (diffyHz[myidx][0] - diffzHy[myidx][0]);
				Ex_im[myidx] = Ex_im[myidx] + (dt/eps_EHH[myidx]) * (diffyHz[myidx][1] - diffzHy[myidx][1]);
		
			}		
		}
	}

	// Update Ey.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, dx, dt, Hz_re, Hz_im, Ey_re, Ey_im, eps_HEE, diffzHx)	\
		private(i, j, k, myidx, myidx_i, diffxHz_re, diffxHz_im)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;

				diffxHz_re = (Hz_re[myidx] - Hz_re[myidx_i]) / dx;
				diffxHz_im = (Hz_im[myidx] - Hz_im[myidx_i]) / dx;
					
				Ey_re[myidx] = Ey_re[myidx] + (-dt/eps_HEE[myidx]) * (diffxHz_re - diffzHx[myidx][0]);
				Ey_im[myidx] = Ey_im[myidx] + (-dt/eps_HEE[myidx]) * (diffxHz_im - diffzHx[myidx][1]);

			}
		}
	}

	// Update Ez.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, dx, dt, Hy_re, Hy_im, Ez_re, Ez_im, eps_HEE, diffyHx)	\
		private(i, j, k, myidx, myidx_i, diffxHy_re, diffxHy_im)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;

				diffxHy_re = (Hy_re[myidx] - Hy_re[myidx_i]) / dx;
				diffxHy_im = (Hy_im[myidx] - Hy_im[myidx_i]) / dx;
					
				Ez_re[myidx] = Ez_re[myidx] + (dt/eps_HEE[myidx]) * (diffxHy_re - diffyHx[myidx][0]);
				Ez_im[myidx] = Ez_im[myidx] + (dt/eps_HEE[myidx]) * (diffxHy_im - diffyHx[myidx][1]);

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

void updateE_rankML(												\
	int	    myNx,	 int	 Ny,	int	    Nz,						\
	double  dt,		 double  dx,	double  dy,		double dz,		\
	double *Ex_re,	 double *Ex_im,	double *Hx_re,	double *Hx_im,	\
	double *Ey_re,	 double *Ey_im,	double *Hy_re,	double *Hy_im,	\
	double *Ez_re,	 double *Ez_im,	double *Hz_re,	double *Hz_im,	\
	double *eps_HEE, double *eps_EHH,								\
	double *ky,		 double *kz,									\
	double *recvHyfirst_re,	double *recvHyfirst_im,					\
	double *recvHzfirst_re,	double *recvHzfirst_im					\
){

	fftw_init_threads();

	// int for index.
	int i,j,k;
	int myidx, myidx_i;

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
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffyHx, diffzHx, diffzHy, diffyHz)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;
		
				diffyHx[myidx][0] = diffyHx[myidx][0] / (Ny*Nz);
				diffyHx[myidx][1] = diffyHx[myidx][1] / (Ny*Nz);

				diffzHx[myidx][0] = diffzHx[myidx][0] / (Ny*Nz);
				diffzHx[myidx][1] = diffzHx[myidx][1] / (Ny*Nz);

				diffzHy[myidx][0] = diffzHy[myidx][0] / (Ny*Nz);
				diffzHy[myidx][1] = diffzHy[myidx][1] / (Ny*Nz);

				diffyHz[myidx][0] = diffyHz[myidx][0] / (Ny*Nz);
				diffyHz[myidx][1] = diffyHz[myidx][1] / (Ny*Nz);

			}
		}
	}

	// Update Ex.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, dt, eps_EHH, diffyHz, diffzHy)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				Ex_re[myidx] = Ex_re[myidx] + (dt/eps_EHH[myidx]) * (diffyHz[myidx][0] - diffzHy[myidx][0]);
				Ex_im[myidx] = Ex_im[myidx] + (dt/eps_EHH[myidx]) * (diffyHz[myidx][1] - diffzHy[myidx][1]);
		
			}		
		}
	}

	double diffxHz_re, diffxHz_im;

	// Update Ey.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, dt, dx, eps_HEE, Hz_re, Hz_im, Ey_re, Ey_im, diffzHx)	\
		private(i, j, k, myidx, myidx_i, diffxHz_re, diffxHz_im)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;

				diffxHz_re = (Hz_re[myidx] - Hz_re[myidx_i]) / dx;
				diffxHz_im = (Hz_im[myidx] - Hz_im[myidx_i]) / dx;
					
				Ey_re[myidx] = Ey_re[myidx] + (-dt/eps_HEE[myidx]) * (diffxHz_re - diffzHx[myidx][0]);
				Ey_im[myidx] = Ey_im[myidx] + (-dt/eps_HEE[myidx]) * (diffxHz_im - diffzHx[myidx][1]);

			}
		}
	}

	// update Ey at i=0.
	#pragma omp parallel for	\
		shared(Ny, Nz, dx, dt, eps_HEE, Hz_re, Hz_im, recvHzfirst_re, recvHzfirst_im, diffzHx)	\
		private(j, k, myidx, diffxHz_re, diffxHz_im)
	for(j=0; j < Ny; j++){
		for(k=0; k < Nz; k++){

			myidx = k + j * Nz + (0) * Nz * Ny;

			diffxHz_re = (Hz_re[myidx] - recvHzfirst_re[myidx]) / dx;
			diffxHz_im = (Hz_im[myidx] - recvHzfirst_im[myidx]) / dx;
				
			Ey_re[myidx] = Ey_re[myidx] + (-dt/eps_HEE[myidx]) * (diffxHz_re - diffzHx[myidx][0]);
			Ey_im[myidx] = Ey_im[myidx] + (-dt/eps_HEE[myidx]) * (diffxHz_im - diffzHx[myidx][1]);

		}
	}

	double diffxHy_re, diffxHy_im;

	// Update Ez.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, dt, dx, eps_HEE, Hy_re, Hy_im, diffyHx)	\
		private(i, j, k, myidx, myidx_i, diffxHy_re, diffxHy_im)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;

				diffxHy_re = (Hy_re[myidx] - Hy_re[myidx_i]) / dx;
				diffxHy_im = (Hy_im[myidx] - Hy_im[myidx_i]) / dx;
					
				Ez_re[myidx] = Ez_re[myidx] + (dt/eps_HEE[myidx]) * (diffxHy_re - diffyHx[myidx][0]);
				Ez_im[myidx] = Ez_im[myidx] + (dt/eps_HEE[myidx]) * (diffxHy_im - diffyHx[myidx][1]);

			}
		}
	}

	// Update Ez at i=0.
	#pragma omp parallel for	\
		shared(Ny, Nz, Hy_re, Hy_im, recvHyfirst_re, recvHyfirst_im, dx, dt, eps_HEE, diffyHx)	\
		private(j, k, myidx, diffxHy_re, diffxHy_im)
	for(j=0; j < Ny; j++){
		for(k=0; k < Nz; k++){

			myidx = k + j * Nz + (0) * Nz * Ny;

			diffxHy_re = (Hy_re[myidx] - recvHyfirst_re[myidx]) / dx;
			diffxHy_im = (Hy_im[myidx] - recvHyfirst_im[myidx]) / dx;
				
			Ez_re[myidx] = Ez_re[myidx] + (dt/eps_HEE[myidx]) * (diffxHy_re - diffyHx[myidx][0]);
			Ez_im[myidx] = Ez_im[myidx] + (dt/eps_HEE[myidx]) * (diffxHy_im - diffyHx[myidx][1]);

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

// update H in the first and middle rank.
void updateH_rankFM(												\
	int		myNx,	 int	Ny,		int		Nz,						\
	double  dt,		 double  dx,	double	dy,		double dz,		\
	double *Ex_re,	 double *Ex_im,	double *Hx_re,	double *Hx_im,	\
	double *Ey_re,	 double *Ey_im,	double *Hy_re,	double *Hy_im,	\
	double *Ez_re,	 double *Ez_im,	double *Hz_re,	double *Hz_im,	\
	double *mu_HEE,	 double *mu_EHH,								\
	double *ky,		 double *kz,									\
	double *recvEylast_re,	double *recvEylast_im,					\
	double *recvEzlast_re,	double *recvEzlast_im					\
){

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

	// Normalize the results.
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffyEx, diffzEx, diffzEy, diffyEz)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;
		
				diffyEx[myidx][0] = diffyEx[myidx][0] / (Ny*Nz);
				diffyEx[myidx][1] = diffyEx[myidx][1] / (Ny*Nz);

				diffzEx[myidx][0] = diffzEx[myidx][0] / (Ny*Nz);
				diffzEx[myidx][1] = diffzEx[myidx][1] / (Ny*Nz);

				diffzEy[myidx][0] = diffzEy[myidx][0] / (Ny*Nz);
				diffzEy[myidx][1] = diffzEy[myidx][1] / (Ny*Nz);

				diffyEz[myidx][0] = diffyEz[myidx][0] / (Ny*Nz);
				diffyEz[myidx][1] = diffyEz[myidx][1] / (Ny*Nz);

			}
		}
	}

	// Update Hx
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, Hx_re, Hx_im, dt, mu_HEE, diffyEz, diffzEy)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				Hx_re[myidx] = Hx_re[myidx] + (-dt/mu_HEE[myidx]) * (diffyEz[myidx][0] - diffzEy[myidx][0]);
				Hx_im[myidx] = Hx_im[myidx] + (-dt/mu_HEE[myidx]) * (diffyEz[myidx][1] - diffzEy[myidx][1]);
			}
		}
	}

	double diffxEz_re, diffxEz_im;
	double diffxEy_re, diffxEy_im;

	// Update Hy
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, Ez_re, Ez_im, dx, Hy_re, Hy_im, dt, mu_EHH, diffzEx)	\
		private(i, j, k, myidx, i_myidx, diffxEz_re, diffxEz_im)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;

				diffxEz_re = (Ez_re[i_myidx] - Ez_re[myidx]) / dx;
				diffxEz_im = (Ez_im[i_myidx] - Ez_im[myidx]) / dx;

				Hy_re[myidx] = Hy_re[myidx] + (dt/mu_EHH[myidx]) * (diffxEz_re - diffzEx[myidx][0]);
				Hy_im[myidx] = Hy_im[myidx] + (dt/mu_EHH[myidx]) * (diffxEz_im - diffzEx[myidx][1]);
			}
		}
	}

	// Update Hy at i=myNx-1.
	#pragma omp parallel for	\
		shared(Ny, Nz, recvEzlast_re, recvEzlast_im, Ez_re, Ez_im, dx, Hy_re, Hy_im, dt, mu_EHH, diffzEx)	\
		private(j, k, myidx, myidx_0, diffxEz_re, diffxEz_im)
	for(j=0; j < Ny; j++){
		for(k=0; k < Nz; k++){

			myidx   = k + j * Nz + (myNx-1) * Nz * Ny;
			myidx_0 = k + j * Nz + (0     ) * Nz * Ny;

			diffxEz_re = (recvEzlast_re[myidx_0] - Ez_re[myidx]) / dx;
			diffxEz_im = (recvEzlast_im[myidx_0] - Ez_im[myidx]) / dx;

			Hy_re[myidx] = Hy_re[myidx] + (dt/mu_EHH[myidx]) * (diffxEz_re - diffzEx[myidx][0]);
			Hy_im[myidx] = Hy_im[myidx] + (dt/mu_EHH[myidx]) * (diffxEz_im - diffzEx[myidx][1]);

		}
	}

	// Update Hz
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, Ey_re, Ey_im, dx, Hz_re, Hz_im, dt, mu_EHH, diffyEx)	\
		private(i, j, k, myidx, i_myidx, diffxEy_re, diffxEy_im)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;

				diffxEy_re = (Ey_re[i_myidx] - Ey_re[myidx]) / dx;
				diffxEy_im = (Ey_im[i_myidx] - Ey_im[myidx]) / dx;

				Hz_re[myidx] = Hz_re[myidx] + (-dt/mu_EHH[myidx]) * (diffxEy_re - diffyEx[myidx][0]);
				Hz_im[myidx] = Hz_im[myidx] + (-dt/mu_EHH[myidx]) * (diffxEy_im - diffyEx[myidx][1]);

			}
		}
	}

	// Update Hz at i=myNx-1.
	#pragma omp parallel for	\
		shared(Ny, Nz, recvEylast_re, recvEylast_im, Ey_re, Ey_im, dx, Hz_re, Hz_im, dt, mu_EHH, diffyEx)	\
		private(j, k, myidx, myidx_0, diffxEy_re, diffxEy_im)
	for(j=0; j < Ny; j++){
		for(k=0; k < Nz; k++){

			myidx   = k + j * Nz + (myNx-1) * Nz * Ny;
			myidx_0 = k + j * Nz + (0     ) * Nz * Ny;

			diffxEy_re = (recvEylast_re[myidx_0] - Ey_re[myidx]) / dx;
			diffxEy_im = (recvEylast_im[myidx_0] - Ey_im[myidx]) / dx;

			Hz_re[myidx] = Hz_re[myidx] + (-dt/mu_EHH[myidx]) * (diffxEy_re - diffyEx[myidx][0]);
			Hz_im[myidx] = Hz_im[myidx] + (-dt/mu_EHH[myidx]) * (diffxEy_im - diffyEx[myidx][1]);

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

// update H in the last rank.
void updateH_rank_L(												\
	int		myNx,	 int	 Ny,	int		Nz,						\
	double  dt,		 double  dx,	double  dy,		double dz,		\
	double *Ex_re,	 double *Ex_im,	double *Hx_re,	double *Hx_im,	\
	double *Ey_re,	 double *Ey_im,	double *Hy_re,	double *Hy_im,	\
	double *Ez_re,	 double *Ez_im,	double *Hz_re,	double *Hz_im,	\
	double *mu_HEE,	 double *mu_EHH,								\
	double *ky,		 double *kz										\
){

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
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, diffyEx, diffzEx, diffzEy, diffyEz)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx = k + j * Nz + i * Nz * Ny;
		
				diffyEx[myidx][0] = diffyEx[myidx][0] / (Ny*Nz);
				diffyEx[myidx][1] = diffyEx[myidx][1] / (Ny*Nz);

				diffzEx[myidx][0] = diffzEx[myidx][0] / (Ny*Nz);
				diffzEx[myidx][1] = diffzEx[myidx][1] / (Ny*Nz);

				diffzEy[myidx][0] = diffzEy[myidx][0] / (Ny*Nz);
				diffzEy[myidx][1] = diffzEy[myidx][1] / (Ny*Nz);

				diffyEz[myidx][0] = diffyEz[myidx][0] / (Ny*Nz);
				diffyEz[myidx][1] = diffyEz[myidx][1] / (Ny*Nz);

			}
		}
	}

	// Update Hx
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, Hx_re, Hx_im, dt, mu_HEE, diffyEz, diffzEy)	\
		private(i, j, k, myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				Hx_re[myidx] = Hx_re[myidx] + (-dt/mu_HEE[myidx]) * (diffyEz[myidx][0] - diffzEy[myidx][0]);
				Hx_im[myidx] = Hx_im[myidx] + (-dt/mu_HEE[myidx]) * (diffyEz[myidx][1] - diffzEy[myidx][1]);
			}
		}
	}

	double diffxEz_re, diffxEz_im;
	double diffxEy_re, diffxEy_im;

	// Update Hy
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, Ez_re, Ez_im, dx, Hy_re, Hy_im, dt, mu_EHH, diffzEx)	\
		private(i, j, k, myidx, i_myidx, diffxEz_re, diffxEz_im)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;

				diffxEz_re = (Ez_re[i_myidx] - Ez_re[myidx]) / dx;
				diffxEz_im = (Ez_im[i_myidx] - Ez_im[myidx]) / dx;

				Hy_re[myidx] = Hy_re[myidx] + (dt/mu_EHH[myidx]) * (diffxEz_re - diffzEx[myidx][0]);
				Hy_im[myidx] = Hy_im[myidx] + (dt/mu_EHH[myidx]) * (diffxEz_im - diffzEx[myidx][1]);
			}
		}
	}

	// Update Hz
	#pragma omp parallel for	\
		shared(myNx, Ny, Nz, Ey_re, Ey_im, dx, Hz_re, Hz_im, dt, mu_EHH, diffyEx)	\
		private(i, j, k, myidx, i_myidx, diffxEy_re, diffxEy_im)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;

				diffxEy_re = (Ey_re[i_myidx] - Ey_re[myidx]) / dx;
				diffxEy_im = (Ey_im[i_myidx] - Ey_im[myidx]) / dx;

				Hz_re[myidx] = Hz_re[myidx] + (-dt/mu_EHH[myidx]) * (diffxEy_re - diffyEx[myidx][0]);
				Hz_im[myidx] = Hz_im[myidx] + (-dt/mu_EHH[myidx]) * (diffxEy_im - diffyEx[myidx][1]);

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
