#include <stdio.h>
#include <stdlib.h>
//#include <complex.h>
//#include <fftw3.h>

/*	
	This script only contains update equations of basic FDTD.
	Update equations for UPML or CPML, ADE-FDTD is not developed here.

	Core update equations are useful when testing the speed of algorithm or
	performance of the hardware such as CPU, GPU or memory.

	Author: Donggun Lee	
	Data  : 18.01.17

	Update Equations for E,H field
	Update Equations for MPI Boundary

*/

void update_inner_Hx(	double *Hx_re, double *Hx_im, \
						double *Ey_re, double *Ey_im, double *Ez_re, double *Ez_im, \
						double *mu, double dt, double dy, double dz, int myNx, int Ny, int Nz){

	int i,j,k,idx;
	double diffyEz_re, diffzEy_re;
	double diffyEz_im, diffzEy_im;

	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < (Nz-1); k++){
				
				idx = k + j * Nz + i * Nz * Ny;

				diffyEz_re = (Ez_re[idx + Nz] - Ez_re[idx]) / dy;
				diffzEy_re = (Ey_re[idx + 1 ] - Ey_re[idx]) / dz;
				diffyEz_im = (Ez_im[idx + Nz] - Ez_im[idx]) / dy;
				diffzEy_im = (Ey_im[idx + 1 ] - Ey_im[idx]) / dz;
				Hx_re[idx] = Hx_re[idx] + (-dt/mu[idx])*(diffyEz_re - diffzEy_re)
				Hx_im[idx] = Hx_im[idx] + (-dt/mu[idx])*(diffyEz_im - diffzEy_im)
			}
		}
	}

	return;
}

void update_inner_Hy(	double *Hx_re, double *Hx_im, \
						double *Ey_re, double *Ey_im, double *Ez_re, double *Ez_im, \
						double *mu, double dt, double dy, double dz, int myNx, int Ny, int Nz){

	int i,j,k,idx;
	double diffyEz_re, diffzEy_re;
	double diffyEz_im, diffzEy_im;

	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < (Nz-1); k++){
				
				idx = k + j * Nz + i * Nz * Ny;

				diffyEz_re = (Ez_re[idx + Nz] - Ez_re[idx]) / dy;
				diffzEy_re = (Ey_re[idx + 1 ] - Ey_re[idx]) / dz;
				diffyEz_im = (Ez_im[idx + Nz] - Ez_im[idx]) / dy;
				diffzEy_im = (Ey_im[idx + 1 ] - Ey_im[idx]) / dz;
				Hx_re[idx] = Hx_re[idx] + (-dt/mu[idx])*(diffyEz_re - diffzEy_re)
				Hx_im[idx] = Hx_im[idx] + (-dt/mu[idx])*(diffyEz_im - diffzEy_im)
			}
		}
	}

	return;
}

