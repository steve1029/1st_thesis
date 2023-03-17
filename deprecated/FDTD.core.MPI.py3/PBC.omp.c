#include <stdio.h>
#include <stdlib.h>

/*
	Author: Donggun Lee
	Date : 18.01.22

	This script contains update equations to apply PBC.

	Total grid is disassembled along x-axis, which is the most significant index.
	Decomposed grid is distributed to each node and each nodes are connected through
	MPI communication. Thus, PBC along x-axis needs MPI communication between
	the first node and the last node. PBC along y and z-axis does not need MPI communication.

	LOG
	---
	18.01.19: apply_PBC_yplus has been finished.
	18.01.22: update equations for PBC in core.c is detached to PBC.c

*/

/*
void apply_PBC_xminus(	double *Ey_re, double *Ey_im, double *Ez_re, double *Ez_im, \
						double *recvHzlast_re, double *recvHzlast_im, double *recvHylast_re, double *recvHylast_im, \
						double *Hx_re, double *Hx_im, double *Hy_re, double *Hy_im, double *Hz_re, double *Hz_im, \
						double *eps_Ey, double *eps_Ez, double dt, double dx, double dy, double dz, int myNx, int Ny, int Nz){

	   By the Yee grid, a plane of Ey or Ez at i=0 cannot be updated.
	   We will update this plane with the plane of Hz or Hy at i=(Nx-1).
	   However, a plane at i=0 is in the memory of rank 0.
	   To update Ey and Ez at i=0 in rank 0, we need to get Hy and Hz at i=(Nx-1) in rank (MPIsize-1).

	update_first_Ey(Ey_re, Ey_im, recvHzlast_re, recvHzlast_im, Hx_re, Hx_im, Hz_re, Hz_im, \
					eps_Ey, dt, dx, dy, dz, myNx, Ny, Nz);
	update_first_Ez(Ez_re, Ez_im, recvHylast_re, recvHylast_im, Hx_re, Hx_im, Hy_re, Hy_im, \
					eps_Ez, dt, dx, dy, dz, myNx, Ny, Nz);

	return;
}

void apply_PBC_xplus(	double *Hy_re, double *Hy_im, double *Hz_re, double *Hz_im, \
						double *recvEzfirst_re, double *recvEzfirst_im, double *recvEyfirst_re, double *recvEyfirst_im, \
						double *Ex_re, double *Ex_im, double *Ey_re, double *Ey_im, double *Ez_re, double *Ez_im, \
						double *mu_Hy, double *mu_Hz, double dt, double dx, double dy, double dz, int myNx, int Ny, int Nz){

	update_last_Hy(Hy_re, Hy_im, recvEzfirst_re, recvEzfirst_im, Ex_re, Ex_im, Ez_re, Ez_im, \
					mu_Hy, dt, dx, dy, dz, myNx, Ny, Nz);
	update_last_Hz(Hz_re, Hz_im, recvEyfirst_re, recvEyfirst_im, Ex_re, Ex_im, Ey_re, Ey_im, \
					mu_Hz, dt, dx, dy, dz, myNx, Ny, Nz);

	return;
}
*/

void apply_PBC_inner_yminus(\
	double *Ex_re, double *Ex_im, double *Ez_re, double *Ez_im, \
	double *Hx_re, double *Hx_im, double *Hy_re, double *Hy_im, double *Hz_re, double *Hz_im, \
	double *eps_Ex, double *eps_Ez, \
	double dt, double dx, double dy, double dz, \
	int myNx, int Ny, int Nz){

	// By the staggered nature of the Yee grid, a plane of Ex or Ez at j=0 cannot be updated.
	// We will update this plane with the plane of Hz or Hx at j=(Ny-1), respectively.
	// This procedure effectively makes the simulation space periodic along y-axis.

	int i,k;
	int first_idx, last_idx, first_idx_k, first_idx_i;
	double diffyHz_re, diffzHy_re, diffxHy_re, diffyHx_re;
	double diffyHz_im, diffzHy_im, diffxHy_im, diffyHx_im;

	// Update Ex at j=0 by using Hy at j=Ny-1
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Hy_re, Hy_im, Hz_re, Hz_im, Ex_re, Ex_im, eps_Ex)	\
		private(i, k, first_idx, first_idx_k, last_idx, diffyHz_re, diffyHz_im, diffzHy_re, diffzHy_im)
	for(i=0; i < myNx; i++){
		for(k=1; k < Nz; k++){
	
			first_idx   = (k  ) + (0   ) * Nz + i * Nz * Ny;
			first_idx_k = (k-1) + (0   ) * Nz + i * Nz * Ny;
			last_idx    = (k  ) + (Ny-1) * Nz + i * Nz * Ny;

			diffyHz_re = (Hz_re[first_idx] - Hz_re[last_idx]) / dy;
			diffyHz_im = (Hz_im[first_idx] - Hz_im[last_idx]) / dy;

			diffzHy_re = (Hy_re[first_idx] - Hy_re[first_idx_k]) / dz;
			diffzHy_im = (Hy_im[first_idx] - Hy_im[first_idx_k]) / dz;

			Ex_re[first_idx] = Ex_re[first_idx] + (dt/eps_Ex[first_idx])*(diffyHz_re - diffzHy_re);
			Ex_im[first_idx] = Ex_im[first_idx] + (dt/eps_Ex[first_idx])*(diffyHz_im - diffzHy_im);

		}
	}

	// Update Ez at j=0 by using Hx at j=Ny-1
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Hx_re, Hx_im, Hy_re, Hy_im, Ez_re, Ez_im, eps_Ez)	\
		private(i, k, first_idx, first_idx_i, last_idx, diffxHy_re, diffxHy_im, diffyHx_re, diffyHx_im)
	for(i=1; i < myNx; i++){
		for(k=0; k < Nz; k++){
	
			first_idx   = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			first_idx_i = (k  ) + (0   ) * Nz + (i-1) * Nz * Ny;
			last_idx    = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;

			diffxHy_re = (Hy_re[first_idx] - Hy_re[first_idx_i]) / dx;
			diffxHy_im = (Hy_im[first_idx] - Hy_im[first_idx_i]) / dx;

			diffyHx_re = (Hx_re[first_idx] - Hx_re[last_idx]) / dy;
			diffyHx_im = (Hx_im[first_idx] - Hx_im[last_idx]) / dy;
		
			Ez_re[first_idx] = Ez_re[first_idx] + (dt/eps_Ez[first_idx])*(diffxHy_re - diffyHx_re);
			Ez_im[first_idx] = Ez_im[first_idx] + (dt/eps_Ez[first_idx])*(diffxHy_im - diffyHx_im);

		}
	}
			
	return;
}

void apply_PBC_outer_yminus(double *Ez_re, double *Ez_im, \
							double *Hx_re, double *Hx_im, double *Hy_re, double *Hy_im, \
							double *recvHyfirst_re, double *recvHyfirst_im, \
							double *eps_Ez, \
							double dt, double dx, double dy, \
							int myNx, int Ny, int Nz){

	// By the staggered nature of the Yee grid, a plane of Ex or Ez at j=0 cannot be updated.
	// We will update this plane with the plane of Hz or Hx at j=(Ny-1), respectively.
	// This procedure effectively makes the simulation space periodic along y-axis.

	int k;
	int first_ij_idx, last_j_idx;
	double diffxHy_re, diffxHy_im;
	double diffyHx_re, diffyHx_im;

	// Update Ez at i=0 and j=0 by using Hx at j=(Ny-1) and Hy from previous rank.
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Hx_re, Hx_im, Hy_re, Hy_im, recvHyfirst_re, recvHyfirst_im, Ez_re, Ez_im, eps_Ez)	\
		private(k, first_ij_idx, last_j_idx, diffxHy_re, diffxHy_im, diffyHx_re, diffyHx_im)
	for(k=0; k < Nz; k++){
		
		first_ij_idx = k + (0   ) * Nz + (0) * Nz * Ny;
		last_j_idx   = k + (Ny-1) * Nz + (0) * Nz * Ny;

		diffxHy_re = (Hy_re[first_ij_idx] - recvHyfirst_re[first_ij_idx]) / dx;
		diffxHy_im = (Hy_im[first_ij_idx] - recvHyfirst_im[first_ij_idx]) / dx;

		diffyHx_re = (Hx_re[first_ij_idx] - Hx_re[last_j_idx]) / dy;
		diffyHx_im = (Hx_im[first_ij_idx] - Hx_im[last_j_idx]) / dy;

		Ez_re[first_ij_idx] = Ez_re[first_ij_idx] + (dt/eps_Ez[first_ij_idx]) * (diffxHy_re - diffyHx_re);
		Ez_im[first_ij_idx] = Ez_im[first_ij_idx] + (dt/eps_Ez[first_ij_idx]) * (diffxHy_im - diffyHx_im);
	}

	return;
}

void apply_PBC_inner_yplus( \
	double *Hx_re, double *Hx_im, double *Hz_re, double *Hz_im, \
	double *Ex_re, double *Ex_im, double *Ey_re, double *Ey_im, double *Ez_re, double *Ez_im, \
	double *mu_Hx, double *mu_Hz, \
	double dt, double dx, double dy, double dz, \
	int myNx, int Ny, int Nz){

   //By the Yee grid, a plane of Hx and Hz at j=(Ny-1) cannot be updated.
   //We will update this plane with the plane of Ez or Ex at j=0, respectively.

	int i,k;
	int first_idx, k_last_idx, i_last_idx, last_idx;
	double diffyEz_re, diffzEy_re, diffxEy_re, diffyEx_re;
	double diffyEz_im, diffzEy_im, diffxEy_im, diffyEx_im;

	// Update Hx at j=(Ny-1) by using Ez at j=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Ey_re, Ey_im, Ez_re, Ez_im, Hx_re, Hx_im, mu_Hx)	\
		private(i, k, first_idx, last_idx, k_last_idx, diffyEz_re, diffyEz_im, diffzEy_re, diffzEy_im)
	for(i=0; i < myNx; i++){
		for(k=0; k < (Nz-1); k++){
	
			first_idx   = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			last_idx    = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;
			k_last_idx  = (k+1) + (Ny-1) * Nz + (i  ) * Nz * Ny;

			diffyEz_re = (Ez_re[first_idx] - Ez_re[last_idx]) / dy;
			diffyEz_im = (Ez_im[first_idx] - Ez_im[last_idx]) / dy;

			diffzEy_re = (Ey_re[k_last_idx] - Ey_re[last_idx]) / dz;
			diffzEy_im = (Ey_im[k_last_idx] - Ey_im[last_idx]) / dz;

			Hx_re[last_idx] = Hx_re[last_idx] + (-dt/mu_Hx[last_idx])*(diffyEz_re - diffzEy_re);
			Hx_im[last_idx] = Hx_im[last_idx] + (-dt/mu_Hx[last_idx])*(diffyEz_im - diffzEy_im);

		}
	}

	// Update Hz at j=(Ny-1) by using Ex at j=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Ex_re, Ex_im, Ey_re, Ey_im, Hz_re, Hz_im, mu_Hz)	\
		private(i, k, first_idx, last_idx, i_last_idx, diffxEy_re, diffxEy_im, diffyEx_re, diffyEx_im)
	for(i=0; i < (myNx-1); i++){
		for(k=0; k < Nz; k++){
	
			first_idx   = (k  ) + (0   ) * Nz + (i  ) * Nz * Ny;
			last_idx    = (k  ) + (Ny-1) * Nz + (i  ) * Nz * Ny;
			i_last_idx  = (k  ) + (Ny-1) * Nz + (i+1) * Nz * Ny;


			diffxEy_re = (Ey_re[i_last_idx] - Ey_re[last_idx]) / dx;
			diffxEy_im = (Ey_im[i_last_idx] - Ey_im[last_idx]) / dx;

			diffyEx_re = (Ex_re[first_idx] - Ex_re[last_idx]) / dy;
			diffyEx_im = (Ex_im[first_idx] - Ex_im[last_idx]) / dy;
		
			Hz_re[last_idx] = Hz_re[last_idx] + (-dt/mu_Hz[last_idx])*(diffxEy_re - diffyEx_re);
			Hz_im[last_idx] = Hz_im[last_idx] + (-dt/mu_Hz[last_idx])*(diffxEy_im - diffyEx_im);

		}
	}
			
	return;
}

void apply_PBC_outer_yplus(	double *Hz_re, double *Hz_im, \
							double *Ey_re, double *Ey_im, double *Ex_re, double *Ex_im, \
							double *recvEylast_re, double *recvEylast_im, \
							double *mu_Hz, \
							double dt, double dx, double dy, \
							int myNx, int Ny, int Nz){

	int k;
	int last_ij_idx, first_i_idx, first_j_idx;
	double diffxEy_re, diffxEy_im;
	double diffyEx_re, diffyEx_im;

	// Update Hz at j=(Ny-1) and i=(myNx-1) by using Ex at j=0 and Ey from next rank.
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dy, Ex_re, Ex_im, Ey_re, Ey_im, recvEylast_re, recvEylast_im, Hz_re, Hz_im, mu_Hz)	\
		private(k, last_ij_idx, first_i_idx, first_j_idx, diffxEy_re, diffxEy_im, diffyEx_re, diffyEx_im)
	for(k=0; k < Nz; k++){

		last_ij_idx = k + (Ny-1) * Nz + (myNx-1) * Nz * Ny;
		first_i_idx = k + (Ny-1) * Nz + (0     ) * Nz * Ny;
		first_j_idx = k + (0   ) * Nz + (myNx-1) * Nz * Ny;

		diffxEy_re = (recvEylast_re[first_i_idx] - Ey_re[last_ij_idx]) / dx;
		diffxEy_im = (recvEylast_im[first_i_idx] - Ey_im[last_ij_idx]) / dx;

		diffyEx_re = (Ex_re[first_j_idx] - Ex_re[last_ij_idx]) / dy;
		diffyEx_im = (Ex_im[first_j_idx] - Ex_im[last_ij_idx]) / dy;

		Hz_re[last_ij_idx] = Hz_re[last_ij_idx] + (-dt/mu_Hz[last_ij_idx])*(diffxEy_re - diffyEx_re);
		Hz_im[last_ij_idx] = Hz_im[last_ij_idx] + (-dt/mu_Hz[last_ij_idx])*(diffxEy_im - diffyEx_im);

	}

	return;
}

void apply_PBC_inner_zminus( \
	double *Ex_re, double *Ex_im, double *Ey_re, double *Ey_im, \
	double *Hx_re, double *Hx_im, double *Hy_re, double *Hy_im, double *Hz_re, double *Hz_im, \
	double *eps_Ex, double *eps_Ey, \
	double dt, double dx, double dy, double dz, \
	int myNx, int Ny, int Nz){

		//By the staggered nature of the Yee grid, a plane of Ex or Ey at j=0 cannot be updated.
		//We will update this plane with the plane of Hz or Hx at j=(Ny-1), respectively.
		//This procedure effectively makes the simulation space periodic along y-axis.

	int i,j,k;
	int first_idx, first_idx_i, first_idx_j, last_idx;
	double diffyHz_re, diffzHy_re, diffxHz_re, diffzHx_re;
	double diffyHz_im, diffzHy_im, diffxHz_im, diffzHx_im;

	// Update Ex at k=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Hy_re, Hy_im, Hz_re, Hz_im, Ex_re, Ex_im, eps_Ex)	\
		private(i, j, first_idx, first_idx_j, last_idx, diffyHz_re, diffyHz_im, diffzHy_re, diffzHy_im)
	for(i=0; i < myNx; i++){
		for(j=1; j < Ny; j++){

			first_idx   = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			first_idx_j = (0   ) + (j-1) * Nz + (i  ) * Nz * Ny;
			last_idx    = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			diffyHz_re = (Hz_re[first_idx] - Hz_re[first_idx_j]) / dy;	
			diffyHz_im = (Hz_im[first_idx] - Hz_im[first_idx_j]) / dy;	

			diffzHy_re = (Hy_re[first_idx] - Hy_re[last_idx]) / dz;
			diffzHy_im = (Hy_im[first_idx] - Hy_im[last_idx]) / dz;

			Ex_re[first_idx] = Ex_re[first_idx] + (dt/eps_Ex[first_idx])*(diffyHz_re - diffzHy_re);
			Ex_im[first_idx] = Ex_im[first_idx] + (dt/eps_Ex[first_idx])*(diffyHz_im - diffzHy_im);

		}
	}

	// Update Ey at k=0
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Hx_re, Hx_im, Hz_re, Hz_im, Ey_re, Ey_im, eps_Ey)	\
		private(i, j, first_idx, first_idx_i, last_idx, diffxHz_re, diffxHz_im, diffzHx_re, diffzHx_im)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){

			first_idx   = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			first_idx_i = (0   ) + (j  ) * Nz + (i-1) * Nz * Ny;
			last_idx    = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			diffxHz_re = (Hz_re[first_idx] - Hz_re[first_idx_i]) / dx;	
			diffxHz_im = (Hz_im[first_idx] - Hz_im[first_idx_i]) / dx;	

			diffzHx_re = (Hx_re[first_idx] - Hx_re[last_idx]) / dz;
			diffzHx_im = (Hx_im[first_idx] - Hx_im[last_idx]) / dz;

			Ey_re[first_idx] = Ey_re[first_idx] + (-dt/eps_Ey[first_idx])*(diffxHz_re - diffzHx_re);
			Ey_im[first_idx] = Ey_im[first_idx] + (-dt/eps_Ey[first_idx])*(diffxHz_im - diffzHx_im);

		}
	}

	return;
}

void apply_PBC_outer_zminus(double *Ey_re, double *Ey_im, \
							double *Hx_re, double *Hx_im, double *Hz_re, double *Hz_im, \
							double *recvHzfirst_re, double *recvHzfirst_im, \
							double *eps_Ey, \
							double dt, double dx, double dz, \
							int myNx, int Ny, int Nz){

	int j;
	int first_ik_idx, last_k_idx;
	double diffxHz_re, diffxHz_im;
	double diffzHx_re, diffzHx_im;

	// Update Ey at i=0 and k=0 by using Hx at k=(Nz-1) and Hz from previous rank.
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Hx_re, Hx_im, Hz_re, Hz_im, recvHzfirst_re, recvHzfirst_im, Ey_re, Ey_im, eps_Ey)	\
		private(j, first_ik_idx, last_k_idx, diffxHz_re, diffxHz_im, diffzHx_re, diffzHx_im)
	for(j=0; j < Ny; j++){
		
		first_ik_idx = (0   ) + j * Nz + 0 * Nz * Ny;
		last_k_idx   = (Nz-1) + j * Nz + 0 * Nz * Ny;

		diffxHz_re = (Hz_re[first_ik_idx] - recvHzfirst_re[first_ik_idx]) / dx;
		diffxHz_im = (Hz_im[first_ik_idx] - recvHzfirst_im[first_ik_idx]) / dx;

		diffzHx_re = (Hx_re[first_ik_idx] - Hx_re[last_k_idx]) / dz;
		diffzHx_im = (Hx_im[first_ik_idx] - Hx_im[last_k_idx]) / dz;

		Ey_re[first_ik_idx] = Ey_re[first_ik_idx] + (-dt/eps_Ey[first_ik_idx])*(diffxHz_re - diffzHx_re);
		Ey_im[first_ik_idx] = Ey_im[first_ik_idx] + (-dt/eps_Ey[first_ik_idx])*(diffxHz_im - diffzHx_im);

	}

	return;
}

void apply_PBC_inner_zplus(	\
	double *Hx_re, double *Hx_im, double *Hy_re, double *Hy_im, \
	double *Ex_re, double *Ex_im, double *Ey_re, double *Ey_im, double *Ez_re, double *Ez_im, \
	double *mu_Hx, double *mu_Hy, \
	double dt, double dx, double dy, double dz, \
	int myNx, int Ny, int Nz){

	// By the Yee grid, a plane of Hx or Hy at k=(Nz-1) cannot be updated.
	// We will update this plane with the plane of Ey or Ex at k=0, respectively.

	int i,j,k;
	int last_idx, first_idx, i_last_idx, j_last_idx;
	double diffyEz_re, diffzEy_re, diffxEz_re, diffzEx_re;
	double diffyEz_im, diffzEy_im, diffxEz_im, diffzEx_im;

	// Update Hx at k=(Nz-1)
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dy, dz, Ey_re, Ey_im, Ez_re, Ez_im, Hx_re, Hx_im, mu_Hx)	\
		private(i, j, last_idx, j_last_idx, first_idx, diffyEz_re, diffyEz_im, diffzEy_re, diffzEy_im)
	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){

			last_idx   = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			j_last_idx = (Nz-1) + (j+1) * Nz + (i  ) * Nz * Ny;
			first_idx  = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			diffyEz_re = (Ez_re[j_last_idx] - Ez_re[last_idx]) / dy;	
			diffyEz_im = (Ez_im[j_last_idx] - Ez_im[last_idx]) / dy;	

			diffzEy_re = (Ey_re[first_idx] - Ey_re[last_idx]) / dz;
			diffzEy_im = (Ey_im[first_idx] - Ey_im[last_idx]) / dz;

			Hx_re[last_idx] = Hx_re[last_idx] + (-dt/mu_Hx[last_idx])*(diffyEz_re - diffzEy_re);
			Hx_im[last_idx] = Hx_im[last_idx] + (-dt/mu_Hx[last_idx])*(diffyEz_im - diffzEy_im);

		}
	}

	// Update Hy at k=(Nz-1)
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Ex_re, Ex_im, Ez_re, Ez_im, Hy_re, Hy_im, mu_Hy)	\
		private(i, j, last_idx, i_last_idx, first_idx, diffxEz_re, diffxEz_im, diffzEx_re, diffzEx_im)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){

			last_idx   = (Nz-1) + (j  ) * Nz + (i  ) * Nz * Ny;
			i_last_idx = (Nz-1) + (j  ) * Nz + (i+1) * Nz * Ny;
			first_idx  = (0   ) + (j  ) * Nz + (i  ) * Nz * Ny;
			
			diffxEz_re = (Ez_re[i_last_idx] - Ez_re[last_idx]) / dx;	
			diffxEz_im = (Ez_im[i_last_idx] - Ez_im[last_idx]) / dx;	

			diffzEx_re = (Ex_re[first_idx] - Ex_re[last_idx]) / dz;
			diffzEx_im = (Ex_im[first_idx] - Ex_im[last_idx]) / dz;

			Hy_re[last_idx] = Hy_re[last_idx] + (dt/mu_Hy[last_idx])*(diffxEz_re - diffzEx_re);
			Hy_im[last_idx] = Hy_im[last_idx] + (dt/mu_Hy[last_idx])*(diffxEz_im - diffzEx_im);

		}
	}

	return;
}

void apply_PBC_outer_zplus(	double *Hy_re, double *Hy_im, \
							double *Ex_re, double *Ex_im, double *Ez_re, double *Ez_im, \
							double *recvEzlast_re, double *recvEzlast_im, \
							double *mu_Hy, \
							double dt, double dx, double dz, \
							int myNx, int Ny, int Nz){

	int j;
	int last_ik_idx, first_i_idx, first_k_idx;
	double diffxEz_re, diffxEz_im;
	double diffzEx_re, diffzEx_im;

	// Update Hy at k=(Nz-1) by using Ex at k=0 and Ez from next rank.
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dt, dx, dz, Ex_re, Ex_im, Ez_re, Ez_im, recvEzlast_re, recvEzlast_im, Hy_re, Hy_im, mu_Hy)	\
		private(j, last_ik_idx, first_i_idx, first_k_idx, diffxEz_re, diffxEz_im, diffzEx_re, diffzEx_im)
	for(j=0; j < Ny; j++){
		
		last_ik_idx = (Nz-1) + j * Nz + (myNx-1) * Nz * Ny;
		first_i_idx = (Nz-1) + j * Nz + (0     ) * Nz * Ny;
		first_k_idx = (0   ) + j * Nz + (myNx-1) * Nz * Ny;

		diffxEz_re = (recvEzlast_re[first_i_idx] - Ez_re[last_ik_idx]) / dx;
		diffxEz_im = (recvEzlast_im[first_i_idx] - Ez_im[last_ik_idx]) / dx;

		diffzEx_re = (Ex_re[first_k_idx] - Ex_re[last_ik_idx]) / dz;
		diffzEx_im = (Ex_im[first_k_idx] - Ex_im[last_ik_idx]) / dz;

		Hy_re[last_ik_idx] = Hy_re[last_ik_idx] + (dt/mu_Hy[last_ik_idx])*(diffxEz_re - diffzEx_re);
		Hy_im[last_ik_idx] = Hy_im[last_ik_idx] + (dt/mu_Hy[last_ik_idx])*(diffxEz_im - diffzEx_im);

	}

	return;
}
