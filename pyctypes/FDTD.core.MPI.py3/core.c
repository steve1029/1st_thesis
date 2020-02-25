#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//#include <complex.h>
//#include <fftw3.h>

/*	
	Author: Donggun Lee	
	Date  : 18.01.17

	This script only contains update equations of basic FDTD.
	Update equations for UPML or CPML, ADE-FDTD is not developed here.

	Core update equations are useful when testing the speed of an algorithm
	or the performance of the hardware such as CPU, GPU or memory.

	Update Equations for E,H field
	Update Equations for MPI Boundary

	Discription of variables
	------------------------
	i: x index
	j: y index
	k: z index
	
	myidx  : 1 dimensional index of elements where its index in 3D is (i  ,j  ,k  ).

	i_myidx: 1 dimensional index of elements where its index in 3D is (i+1,j  ,k  ).
	j_myidx: 1 dimensional index of elements where its index in 3D is (i  ,j+1,k  ).
	k_myidx: 1 dimensional index of elements where its index in 3D is (i  ,j  ,k+1).

	myidx_i: 1 dimensional index of elements where its index in 3D is (i-1,j  ,k  ).
	myidx_j: 1 dimensional index of elements where its index in 3D is (i  ,j-1,k  ).
	myidx_k: 1 dimensional index of elements where its index in 3D is (i  ,j  ,k-1).

	ex)
		myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

		i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
		j_myidx = (k  ) + (j+1) * Nz + (i  ) * Nz * Ny;
		k_myidx = (k+1) + (j  ) * Nz + (i  ) * Nz * Ny;

		myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
		myidx_j = (k  ) + (j-1) * Nz + (i  ) * Nz * Ny;
		myidx_k = (k-1) + (j  ) * Nz + (i  ) * Nz * Ny;

*/

void update_inner_E(double *Ex_re , double *Ex_im , double *Ey_re , double *Ey_im, double *Ez_re, double *Ez_im, \
					double *Hx_re , double *Hx_im , double *Hy_re , double *Hy_im, double *Hz_re, double *Hz_im, \
					double *eps_Ex, double *eps_Ey, double *eps_Ez, \
					double dt, double dx, double dy, double dz, int myNx, int Ny, int Nz){
	
	// int for index
	int i,j,k;
	int myidx, myidx_i, myidx_j, myidx_k;

	// To update Ex
	double diffyHz_re, diffyHz_im;
	double diffzHy_re, diffzHy_im;

	// To update Ey
	double diffxHz_re, diffxHz_im;
	double diffzHx_re, diffzHx_im;

	// To update Ez
	double diffxHy_re, diffxHy_im;
	double diffyHx_re, diffyHx_im;

	// Update Ex
	for(i=0; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_j = (k  ) + (j-1) * Nz + (i  ) * Nz * Ny;
				myidx_k = (k-1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffyHz_re = (Hz_re[myidx] - Hz_re[myidx_j]) / dy;
				diffyHz_im = (Hz_im[myidx] - Hz_im[myidx_j]) / dy;
					
				diffzHy_re = (Hy_re[myidx] - Hy_re[myidx_k]) / dz;
				diffzHy_im = (Hy_im[myidx] - Hy_im[myidx_k]) / dz;

				Ex_re[myidx] = Ex_re[myidx] + (dt/eps_Ex[myidx]) * (diffyHz_re - diffzHy_re);
				Ex_im[myidx] = Ex_im[myidx] + (dt/eps_Ex[myidx]) * (diffyHz_im - diffzHy_im);
				
			}
		}
	}

	// Update Ey
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
				myidx_k = (k-1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffxHz_re = (Hz_re[myidx] - Hz_re[myidx_i]) / dx;
				diffxHz_im = (Hz_im[myidx] - Hz_im[myidx_i]) / dx;
					
				diffzHx_re = (Hx_re[myidx] - Hx_re[myidx_k]) / dz;
				diffzHx_im = (Hx_im[myidx] - Hx_im[myidx_k]) / dz;

				Ey_re[myidx] = Ey_re[myidx] + (-dt/eps_Ey[myidx]) * (diffxHz_re - diffzHx_re);
				Ey_im[myidx] = Ey_im[myidx] + (-dt/eps_Ey[myidx]) * (diffxHz_im - diffzHx_im);

			}
		}
	}

	// Update Ez
	for(i=1; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
				myidx_j = (k  ) + (j-1) * Nz + (i  ) * Nz * Ny;

				diffxHy_re = (Hy_re[myidx] - Hy_re[myidx_i]) / dx;
				diffxHy_im = (Hy_im[myidx] - Hy_im[myidx_i]) / dx;
					
				diffyHx_re = (Hx_re[myidx] - Hx_re[myidx_j]) / dy;
				diffyHx_im = (Hx_im[myidx] - Hx_im[myidx_j]) / dy;

				Ez_re[myidx] = Ez_re[myidx] + (dt/eps_Ez[myidx]) * (diffxHy_re - diffyHx_re);
				Ez_im[myidx] = Ez_im[myidx] + (dt/eps_Ez[myidx]) * (diffxHy_im - diffyHx_im);

			}
		}
	}

	return;
}

void update_inner_H(double *Hx_re, double *Hx_im, double *Hy_re, double *Hy_im, double *Hz_re, double *Hz_im, \
					double *Ex_re, double *Ex_im, double *Ey_re, double *Ey_im, double *Ez_re, double *Ez_im, \
					double *mu_Hx, double *mu_Hy, double *mu_Hz, \
					double dt, double dx, double dy, double dz, int myNx, int Ny, int Nz){

	// int for index
	int i,j,k;
	int myidx, i_myidx, j_myidx, k_myidx;

	// To update Hx
	double diffyEz_re, diffyEz_im;
	double diffzEy_re, diffzEy_im;

	// To update Hy
	double diffxEz_re, diffxEz_im;
	double diffzEx_re, diffzEx_im;

	// To update Hz
	double diffxEy_re, diffxEy_im;
	double diffyEx_re, diffyEx_im;

	// Update Hx
	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				j_myidx = (k  ) + (j+1) * Nz + (i  ) * Nz * Ny;
				k_myidx = (k+1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffyEz_re = (Ez_re[j_myidx] - Ez_re[myidx]) / dy;
				diffyEz_im = (Ez_im[j_myidx] - Ez_im[myidx]) / dy;

				diffzEy_re = (Ey_re[k_myidx] - Ey_re[myidx]) / dz;
				diffzEy_im = (Ey_im[k_myidx] - Ey_im[myidx]) / dz;

				Hx_re[myidx] = Hx_re[myidx] + (-dt/mu_Hx[myidx]) * (diffyEz_re - diffzEy_re);
				Hx_im[myidx] = Hx_im[myidx] + (-dt/mu_Hx[myidx]) * (diffyEz_im - diffzEy_im);
			}
		}
	}

	// Update Hy
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
				k_myidx = (k+1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffxEz_re = (Ez_re[i_myidx] - Ez_re[myidx]) / dx;
				diffxEz_im = (Ez_im[i_myidx] - Ez_im[myidx]) / dx;

				diffzEx_re = (Ex_re[k_myidx] - Ex_re[myidx]) / dz;
				diffzEx_im = (Ex_im[k_myidx] - Ex_im[myidx]) / dz;

				Hy_re[myidx] = Hy_re[myidx] + (dt/mu_Hy[myidx]) * (diffxEz_re - diffzEx_re);
				Hy_im[myidx] = Hy_im[myidx] + (dt/mu_Hy[myidx]) * (diffxEz_im - diffzEx_im);
			}
		}
	}

	// Update Hz
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
				j_myidx = (k  ) + (j+1) * Nz + (i  ) * Nz * Ny;

				diffxEy_re = (Ey_re[i_myidx] - Ey_re[myidx]) / dx;
				diffxEy_im = (Ey_im[i_myidx] - Ey_im[myidx]) / dx;

				diffyEx_re = (Ex_re[j_myidx] - Ex_re[myidx]) / dy;
				diffyEx_im = (Ex_im[j_myidx] - Ex_im[myidx]) / dy;

				Hz_re[myidx] = Hz_re[myidx] + (-dt/mu_Hz[myidx]) * (diffxEy_re - diffyEx_re);
				Hz_im[myidx] = Hz_im[myidx] + (-dt/mu_Hz[myidx]) * (diffxEy_im - diffyEx_im);
			}
		}
	}

	return;
}

void update_first_EyEz(	double *Ey_re, double *Ey_im, double *Ez_re, double *Ez_im, \
						double *Hx_re, double *Hx_im, double *Hy_re, double *Hy_im, double *Hz_re, double *Hz_im, \
						double *recvHyfirst_re, double *recvHyfirst_im, double *recvHzfirst_re, double *recvHzfirst_im, \
						double *eps_Ey, double *eps_Ez, \
						double dt, double dx, double dy, double dz, int myNx, int Ny, int Nz){
	
	int j,k;
	int myidx, myidx_j, myidx_k;

	// To update Ey
	double diffxHz_re, diffxHz_im;
	double diffzHx_re, diffzHx_im;

	// To update Ez
	double diffxHy_re, diffxHy_im;
	double diffyHx_re, diffyHx_im;

	// Update Ey at x=0
	for(j=0; j < Ny; j++){
		for(k=1; k < Nz; k++){

			myidx   = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;
			myidx_k = (k-1) + (j  ) * Nz + (0  ) * Nz * Ny;

			diffxHz_re = (Hz_re[myidx] - recvHzfirst_re[myidx]) / dx;
			diffxHz_im = (Hz_im[myidx] - recvHzfirst_im[myidx]) / dx;
				
			diffzHx_re = (Hx_re[myidx] - Hx_re[myidx_k]) / dz;
			diffzHx_im = (Hx_im[myidx] - Hx_im[myidx_k]) / dz;

			Ey_re[myidx] = Ey_re[myidx] + (-dt/eps_Ey[myidx]) * (diffxHz_re - diffzHx_re);
			Ey_im[myidx] = Ey_im[myidx] + (-dt/eps_Ey[myidx]) * (diffxHz_im - diffzHx_im);

		}
	}

	// Update Ez at x=0
	for(j=1; j < Ny; j++){
		for(k=0; k < Nz; k++){

			myidx   = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;
			myidx_j = (k  ) + (j-1) * Nz + (0  ) * Nz * Ny;

			diffxHy_re = (Hy_re[myidx] - recvHyfirst_re[myidx]) / dx;
			diffxHy_im = (Hy_im[myidx] - recvHyfirst_im[myidx]) / dx;
				
			diffyHx_re = (Hx_re[myidx] - Hx_re[myidx_j]) / dy;
			diffyHx_im = (Hx_im[myidx] - Hx_im[myidx_j]) / dy;

			Ez_re[myidx] = Ez_re[myidx] + (dt/eps_Ez[myidx]) * (diffxHy_re - diffyHx_re);
			Ez_im[myidx] = Ez_im[myidx] + (dt/eps_Ez[myidx]) * (diffxHy_im - diffyHx_im);
		}
	}

	return;
}

void update_last_HyHz(	double *Hy_re, double *Hy_im, double *Hz_re, double *Hz_im, \
						double *Ex_re, double *Ex_im, double *Ey_re, double *Ey_im, double *Ez_re, double *Ez_im, \
						double *recvEylast_re, double *recvEylast_im, double *recvEzlast_re, double *recvEzlast_im, \
						double *mu_Hy, double *mu_Hz, \
						double dt, double dx, double dy, double dz, int myNx, int Ny, int Nz){

	int j,k;
	int myidx, j_myidx, k_myidx;
	int yzidx;

	/* To update Hy */
	double diffxEz_re, diffxEz_im;
	double diffzEx_re, diffzEx_im;

	/* To update Hz */
	double diffxEy_re, diffxEy_im;
	double diffyEx_re, diffyEx_im;

	/* Update Hy at x=myNx-1 */
	for(j=0; j < Ny; j++){
		for(k=0; k < (Nz-1); k++){
				
			myidx   = (k  ) + (j  ) * Nz + (myNx-1) * Nz * Ny;
			k_myidx = (k+1) + (j  ) * Nz + (myNx-1) * Nz * Ny;
			yzidx   = (k  ) + (j  ) * Nz + (0     ) * Nz * Ny;

			diffxEz_re = (recvEzlast_re[yzidx] - Ez_re[myidx]) / dx;
			diffxEz_im = (recvEzlast_im[yzidx] - Ez_im[myidx]) / dx;

			diffzEx_re = (Ex_re[k_myidx] - Ex_re[myidx]) / dz;
			diffzEx_im = (Ex_im[k_myidx] - Ex_im[myidx]) / dz;

			Hy_re[myidx] = Hy_re[myidx] + (dt/mu_Hy[myidx]) * (diffxEz_re - diffzEx_re);
			Hy_im[myidx] = Hy_im[myidx] + (dt/mu_Hy[myidx]) * (diffxEz_im - diffzEx_im);
		}
	}

	/* Update Hz at x=myNx-1 */
	for(j=0; j < (Ny-1); j++){
		for(k=0; k < Nz; k++){
				
			myidx   = (k  ) + (j  ) * Nz + (myNx-1) * Nz * Ny;
			j_myidx = (k  ) + (j+1) * Nz + (myNx-1) * Nz * Ny;
			yzidx   = (k  ) + (j  ) * Nz + (0     ) * Nz * Ny;

			diffxEy_re = (recvEylast_re[yzidx] - Ey_re[myidx]) / dx;
			diffxEy_im = (recvEylast_im[yzidx] - Ey_im[myidx]) / dx;

			diffyEx_re = (Ex_re[j_myidx] - Ex_re[myidx]) / dy;
			diffyEx_im = (Ex_im[j_myidx] - Ex_im[myidx]) / dy;

			Hz_re[myidx] = Hz_re[myidx] + (-dt/mu_Hz[myidx]) * (diffxEy_re - diffyEx_re);
			Hz_im[myidx] = Hz_im[myidx] + (-dt/mu_Hz[myidx]) * (diffxEy_im - diffyEx_im);
		}
	}

	return;
}
