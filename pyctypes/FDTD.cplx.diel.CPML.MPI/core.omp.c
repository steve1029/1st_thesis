#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//#include <complex.h>
//#include <fftw3.h>

/*	
	Author: Donggun Lee	
	Date  : 18.01.17

	This script only contains update equations of basic FDTD.
	Update equations for UPML or CPML, ADE-FDTD are not developed here.

	Core update equations are useful when testing the speed of the algorithm
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

/***********************************************************************************/
/******************************** FUNCTION DECLARATION *****************************/
/***********************************************************************************/

// Get derivatives of H field in the first rank.
void get_diff_of_H_rank_F(											\
	int	   myNx,		int	   Ny,		int	   Nz,					\
	double dt,			double dx,		double dy,		double dz,	\
	double *Hx_re,		\
	double *Hy_re,		\
	double *Hz_re,		\
	double *diffxHy_re, \
	double *diffxHz_re, \
	double *diffyHx_re, \
	double *diffyHz_re, \
	double *diffzHx_re, \
	double *diffzHy_re \
);


// Get derivatives of H field in the middle and last rank.
void get_diff_of_H_rankML(											\
	int	   myNx,		int		Ny,		int		Nz,					\
	double dt,			double  dx,		double  dy,		double dz,	\
	double *recvHyfirst_re,	\
	double *recvHzfirst_re,	\
	double *Hx_re,		\
	double *Hy_re,		\
	double *Hz_re,		\
	double *diffxHy_re, \
	double *diffxHz_re, \
	double *diffyHx_re, \
	double *diffyHz_re, \
	double *diffzHx_re, \
	double *diffzHy_re \
);


// Update E field
void updateE_rank_F(
	int	   myNx,		int		Ny,		int		Nz,				\
	double dt,													\
	double *eps_Ex,		double *eps_Ey,		double *eps_Ez,		\
	double *econ_Ex,	double *econ_Ey,	double *econ_Ez,	\
	double *Ex_re,		\
	double *Ey_re,		\
	double *Ez_re,		\
	double *diffxHy_re, \
	double *diffxHz_re, \
	double *diffyHx_re, \
	double *diffyHz_re, \
	double *diffzHx_re, \
	double *diffzHy_re \
);


void updateE_rankML(
	int	   myNx,		int		Ny,		int		Nz,				\
	double dt,													\
	double *eps_Ex,		double *eps_Ey,		double *eps_Ez,		\
	double *econ_Ex,	double *econ_Ey,	double *econ_Ez,	\
	double *Ex_re,		\
	double *Ey_re,		\
	double *Ez_re,		\
	double *diffxHy_re, \
	double *diffxHz_re, \
	double *diffyHx_re, \
	double *diffyHz_re, \
	double *diffzHx_re, \
	double *diffzHy_re \
);


//Get derivatives of E field in the first and middle rank.
void get_diff_of_E_rankFM(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,			double  dx,		double	dy,		double dz,	\
	double *recvEylast_re,	\
	double *recvEzlast_re,	\
	double *Ex_re,		\
	double *Ey_re,		\
	double *Ez_re,		\
	double *diffxEy_re, \
	double *diffxEz_re, \
	double *diffyEx_re, \
	double *diffyEz_re, \
	double *diffzEx_re, \
	double *diffzEy_re \
);


//Get derivatives of E field in the last rank.
void get_diff_of_E_rank_L(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,			double  dx,		double	dy,		double dz,	\
	double *Ex_re,		\
	double *Ey_re,		\
	double *Ez_re,		\
	double *diffxEy_re, \
	double *diffxEz_re, \
	double *diffyEx_re, \
	double *diffyEz_re, \
	double *diffzEx_re, \
	double *diffzEy_re \
);


// Update H field.
void updateH_rankFM(														\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,														\
	double *mu_Hx,		double *mu_Hy,		double *mu_Hz,			\
	double *mcon_Hx,	double *mcon_Hy,	double *mcon_Hz,		\
	double *Hx_re,		\
	double *Hy_re,		\
	double *Hz_re,		\
	double *diffxEy_re, \
	double *diffxEz_re, \
	double *diffyEx_re, \
	double *diffyEz_re, \
	double *diffzEx_re, \
	double *diffzEy_re \
);


void updateH_rank_L(														\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,														\
	double *mu_Hx,		double *mu_Hy,		double *mu_Hz,			\
	double *mcon_Hx,	double *mcon_Hy,	double *mcon_Hz,		\
	double *Hx_re,		\
	double *Hy_re,		\
	double *Hz_re,		\
	double *diffxEy_re, \
	double *diffxEz_re, \
	double *diffyEx_re, \
	double *diffyEz_re, \
	double *diffzEx_re, \
	double *diffzEy_re \
);


/***********************************************************************************/
/******************************** FUNCTION DESCRIPTION *****************************/
/***********************************************************************************/

// Get derivatives of H field in the first rank.
void get_diff_of_H_rank_F(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double	dt,			double  dx,		double  dy,		double dz,	\
	double *Hx_re,		\
	double *Hy_re,		\
	double *Hz_re,		\
	double *diffxHy_re, \
	double *diffxHz_re, \
	double *diffyHx_re, \
	double *diffyHz_re, \
	double *diffzHx_re, \
	double *diffzHy_re \
){	
	// int for index
	int i,j,k;
	int myidx, myidx_i, myidx_j, myidx_k;

	// Get derivatives of Hy and Hz to update Ex
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dy, dz, Hy_re, Hz_re, diffyHz_re, diffzHy_re)	\
		private(i,j,k, myidx, myidx_j, myidx_k)
	for(i=0; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_j = (k  ) + (j-1) * Nz + (i  ) * Nz * Ny;
				myidx_k = (k-1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffyHz_re[myidx] = (Hz_re[myidx] - Hz_re[myidx_j]) / dy;
				//diffyHz_im[myidx] = (Hz_im[myidx] - Hz_im[myidx_j]) / dy;
					
				diffzHy_re[myidx] = (Hy_re[myidx] - Hy_re[myidx_k]) / dz;
				//diffzHy_im[myidx] = (Hy_im[myidx] - Hy_im[myidx_k]) / dz;
				
			}
		}
	}

	// Get derivatives of Hx and Hz to update Ey
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dx, dz, Hx_re, Hz_re, diffxHz_re, diffzHx_re)	\
		private(i,j,k, myidx, myidx_i, myidx_k)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
				myidx_k = (k-1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffzHx_re[myidx] = (Hx_re[myidx] - Hx_re[myidx_k]) / dz;
				//diffzHx_im[myidx] = (Hx_im[myidx] - Hx_im[myidx_k]) / dz;

				diffxHz_re[myidx] = (Hz_re[myidx] - Hz_re[myidx_i]) / dx;
				//diffxHz_im[myidx] = (Hz_im[myidx] - Hz_im[myidx_i]) / dx;
					
			}
		}
	}

	// Get derivatives of Hx and Hy to update Ez
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dy, dz, Hy_re, Hz_re,diffxHy_re, diffyHx_re)	\
		private(i,j,k, myidx, myidx_i, myidx_j)
	for(i=1; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
				myidx_j = (k  ) + (j-1) * Nz + (i  ) * Nz * Ny;

				diffxHy_re[myidx] = (Hy_re[myidx] - Hy_re[myidx_i]) / dx;
				//diffxHy_im[myidx] = (Hy_im[myidx] - Hy_im[myidx_i]) / dx;
					
				diffyHx_re[myidx] = (Hx_re[myidx] - Hx_re[myidx_j]) / dy;
				//diffyHx_im[myidx] = (Hx_im[myidx] - Hx_im[myidx_j]) / dy;

			}
		}
	}

	return;
}

// Get derivatives of H field in the middle and last rank.
void get_diff_of_H_rankML(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double	dt,			double  dx,		double  dy,		double dz,	\
	double *recvHyfirst_re,	
	double *recvHzfirst_re,	
	double *Hx_re,		
	double *Hy_re,		
	double *Hz_re,		
	double *diffxHy_re, 
	double *diffxHz_re, 
	double *diffyHx_re, 
	double *diffyHz_re, 
	double *diffzHx_re, 
	double *diffzHy_re 
){

	// int for index
	int i,j,k;
	int myidx, myidx_i, myidx_j, myidx_k;

	// Get derivatives of Hy and Hz to update Ex
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dy, dz, Hy_re, Hz_re, diffyHz_re, diffzHy_re)	\
		private(i,j,k, myidx, myidx_j, myidx_k)
	for(i=0; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_j = (k  ) + (j-1) * Nz + (i  ) * Nz * Ny;
				myidx_k = (k-1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffyHz_re[myidx] = (Hz_re[myidx] - Hz_re[myidx_j]) / dy;
				//diffyHz_im[myidx] = (Hz_im[myidx] - Hz_im[myidx_j]) / dy;
					
				diffzHy_re[myidx] = (Hy_re[myidx] - Hy_re[myidx_k]) / dz;
				//diffzHy_im[myidx] = (Hy_im[myidx] - Hy_im[myidx_k]) / dz;
				
			}
		}
	}

	// Get derivatives of Hx and Hz to update Ey
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dy, dz, Hx_re, Hz_re, diffxHz_re, diffzHx_re)	\
		private(i,j,k, myidx, myidx_i, myidx_k)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
				myidx_k = (k-1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffzHx_re[myidx] = (Hx_re[myidx] - Hx_re[myidx_k]) / dz;
				//diffzHx_im[myidx] = (Hx_im[myidx] - Hx_im[myidx_k]) / dz;

				diffxHz_re[myidx] = (Hz_re[myidx] - Hz_re[myidx_i]) / dx;
				//diffxHz_im[myidx] = (Hz_im[myidx] - Hz_im[myidx_i]) / dx;
					
			}
		}
	}

	// Get derivatives of Hx and Hy to update Ez
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dy, dz, Hy_re, Hz_re, diffxHy_re, diffyHx_re)	\
		private(i, j, k, myidx, myidx_i, myidx_j)
	for(i=1; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				myidx_i = (k  ) + (j  ) * Nz + (i-1) * Nz * Ny;
				myidx_j = (k  ) + (j-1) * Nz + (i  ) * Nz * Ny;

				diffxHy_re[myidx] = (Hy_re[myidx] - Hy_re[myidx_i]) / dx;
				//diffxHy_im[myidx] = (Hy_im[myidx] - Hy_im[myidx_i]) / dx;
					
				diffyHx_re[myidx] = (Hx_re[myidx] - Hx_re[myidx_j]) / dy;
				//diffyHx_im[myidx] = (Hx_im[myidx] - Hx_im[myidx_j]) / dy;

			}
		}
	}

	// Get derivatives of Hx and Hz to update Ey at x=0
	#pragma omp parallel for \
		shared(Ny, Nz, dx, dz, Hx_re, Hz_re, recvHzfirst_re, diffxHz_re, diffzHx_re)	\
		private(j, k, myidx, myidx_k)
	for(j=0; j < Ny; j++){
		for(k=1; k < Nz; k++){

			myidx   = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;
			myidx_k = (k-1) + (j  ) * Nz + (0  ) * Nz * Ny;

			diffxHz_re[myidx] = (Hz_re[myidx] - recvHzfirst_re[myidx]) / dx;
			//diffxHz_im[myidx] = (Hz_im[myidx] - recvHzfirst_im[myidx]) / dx;
				
			diffzHx_re[myidx] = (Hx_re[myidx] - Hx_re[myidx_k]) / dz;
			//diffzHx_im[myidx] = (Hx_im[myidx] - Hx_im[myidx_k]) / dz;

		}
	}

	// Get derivatives of Hx and Hy to update Ez at x=0
	#pragma omp parallel for \
		shared(Ny, Nz, dx, dy, Hx_re, Hy_re, recvHyfirst_re, diffxHy_re, diffyHx_re)	\
		private(j, k, myidx, myidx_j)
	for(j=1; j < Ny; j++){
		for(k=0; k < Nz; k++){

			myidx   = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;
			myidx_j = (k  ) + (j-1) * Nz + (0  ) * Nz * Ny;

			diffxHy_re[myidx] = (Hy_re[myidx] - recvHyfirst_re[myidx]) / dx;
			//diffxHy_im[myidx] = (Hy_im[myidx] - recvHyfirst_im[myidx]) / dx;
				
			diffyHx_re[myidx] = (Hx_re[myidx] - Hx_re[myidx_j]) / dy;
			//diffyHx_im[myidx] = (Hx_im[myidx] - Hx_im[myidx_j]) / dy;

		}
	}

	return;
}

// Update E field
void updateE_rank_F(
	int    myNx,		int	   Ny,			int	   Nz,				\
	double dt,														\
	double *eps_Ex,		double *eps_Ey,		double *eps_Ez,			\
	double *econ_Ex,	double *econ_Ey,	double *econ_Ez,		\
	double *Ex_re,		
	double *Ey_re,		
	double *Ez_re,		
	double *diffxHy_re, 
	double *diffxHz_re, 
	double *diffyHx_re, 
	double *diffyHz_re, 
	double *diffzHx_re, 
	double *diffzHy_re 
){

	// int for index
	int i,j,k;
	int myidx;

	double CEx1, CEx2;
	double CEy1, CEy2;
	double CEz1, CEz2;

	// Update Ex
	#pragma omp parallel for \
		shared(	myNx, Ny, Nz, dt, eps_Ex, econ_Ex, Ex_re, Ex_im, \
				diffyHz_re, diffyHz_im, diffzHy_re, diffzHy_im)	\
		private(i, j, k, myidx, CEx1, CEx2)
	for(i=0; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEx1 = (2.*eps_Ex[myidx] - econ_Ex[myidx]*dt) / (2.*eps_Ex[myidx] + econ_Ex[myidx]*dt);
				CEx2 = (2.*dt) / (2.*eps_Ex[myidx] + econ_Ex[myidx]*dt);

				// PEC condition.
				if(eps_Ex[myidx] > 1e3){
					CEx1 = 0.;
					CEx2 = 0.;
				}

				Ex_re[myidx] = CEx1 * Ex_re[myidx] + CEx2 * (diffyHz_re[myidx] - diffzHy_re[myidx]);
				//Ex_im[myidx] = CEx1 * Ex_im[myidx] + CEx2 * (diffyHz_im[myidx] - diffzHy_im[myidx]);

			}
		}
	}
				
	// Update Ey
	#pragma omp parallel for \
		shared(	myNx, Ny, Nz, dt, eps_Ey, econ_Ey, Ey_re, Ey_im, \
				diffxHz_re, diffxHz_im,	diffzHx_re, diffzHx_im)	\
		private(i, j, k, myidx, CEy1, CEy2)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEy1 = (2.*eps_Ey[myidx] - econ_Ey[myidx]*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);
				CEy2 = (2.*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);

				// PEC condition.
				if(eps_Ey[myidx] > 1e3){
					CEy1 = 0.;
					CEy2 = 0.;
				}
				Ey_re[myidx] = CEy1 * Ey_re[myidx] + CEy2 * (diffzHx_re[myidx] - diffxHz_re[myidx]);
				//Ey_im[myidx] = CEy1 * Ey_im[myidx] + CEy2 * (diffzHx_im[myidx] - diffxHz_im[myidx]);

			}
		}
	}

	// Update Ez
	#pragma omp parallel for \
		shared(	myNx, Ny, Nz, dt, eps_Ez, econ_Ez, Ez_re, Ez_im,\
				diffxHy_re, diffxHy_im, diffyHx_re, diffyHx_im)	\
		private(i, j, k, myidx, CEz1, CEz2)
	for(i=1; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEz1 = (2.*eps_Ez[myidx] - econ_Ez[myidx]*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);
				CEz2 = (2.*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);

				// PEC condition.
				if(eps_Ez[myidx] > 1e3){
					CEz1 = 0.;
					CEz2 = 0.;
				}

				Ez_re[myidx] = CEz1 * Ez_re[myidx] + CEz2 * (diffxHy_re[myidx] - diffyHx_re[myidx]);
				//Ez_im[myidx] = CEz1 * Ez_im[myidx] + CEz2 * (diffxHy_im[myidx] - diffyHx_im[myidx]);

			}
		}
	}

	return;
}

void updateE_rankML(
	int    myNx,		int	   Ny,			int	   Nz,				\
	double dt,														\
	double *eps_Ex,		double *eps_Ey,		double *eps_Ez,			\
	double *econ_Ex,	double *econ_Ey,	double *econ_Ez,		\
	double *Ex_re,		double *Ex_im,								\
	double *Ey_re,		double *Ey_im,								\
	double *Ez_re,		double *Ez_im,								\
	double *diffxHy_re, double *diffxHy_im,							\
	double *diffxHz_re, double *diffxHz_im,							\
	double *diffyHx_re, double *diffyHx_im,							\
	double *diffyHz_re, double *diffyHz_im,							\
	double *diffzHx_re, double *diffzHx_im,							\
	double *diffzHy_re, double *diffzHy_im							\
){

	// int for index
	int i,j,k;
	int myidx;

	double CEx1, CEx2;
	double CEy1, CEy2;
	double CEz1, CEz2;

	// Update Ex
	#pragma omp parallel for \
		shared(	myNx, Ny, Nz, dt, eps_Ex, econ_Ex, Ex_re, Ex_im, \
				diffyHz_re, diffyHz_im, diffzHy_re, diffzHy_im)	\
		private(i, j, k, myidx, CEx1, CEx2)
	for(i=0; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEx1 = (2.*eps_Ex[myidx] - econ_Ex[myidx]*dt) / (2.*eps_Ex[myidx] + econ_Ex[myidx]*dt);
				CEx2 = (2.*dt) / (2.*eps_Ex[myidx] + econ_Ex[myidx]*dt);

				// PEC condition.
				if(eps_Ex[myidx] > 1e3){
					CEx1 = 0.;
					CEx2 = 0.;
				}

				Ex_re[myidx] = CEx1 * Ex_re[myidx] + CEx2 * (diffyHz_re[myidx] - diffzHy_re[myidx]);
				//Ex_im[myidx] = CEx1 * Ex_im[myidx] + CEx2 * (diffyHz_im[myidx] - diffzHy_im[myidx]);

			}
		}
	}
				
	// Update Ey
	#pragma omp parallel for \
		shared(	myNx, Ny, Nz, dt, eps_Ey, econ_Ey, Ey_re, Ey_im, \
				diffxHz_re, diffxHz_im,	diffzHx_re, diffzHx_im)	\
		private(i, j, k, myidx, CEy1, CEy2)
	for(i=1; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=1; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEy1 = (2.*eps_Ey[myidx] - econ_Ey[myidx]*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);
				CEy2 = (2.*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);

				// PEC condition.
				if(eps_Ey[myidx] > 1e3){
					CEy1 = 0.;
					CEy2 = 0.;
				}

				Ey_re[myidx] = CEy1 * Ey_re[myidx] + CEy2 * (diffzHx_re[myidx] - diffxHz_re[myidx]);
				//Ey_im[myidx] = CEy1 * Ey_im[myidx] + CEy2 * (diffzHx_im[myidx] - diffxHz_im[myidx]);

			}
		}
	}

	// Update Ez
	#pragma omp parallel for \
		shared(	myNx, Ny, Nz, dt, eps_Ez, econ_Ez, Ez_re, Ez_im,\
				diffxHy_re, diffxHy_im, diffyHx_re, diffyHx_im)	\
		private(i, j, k, myidx, CEz1, CEz2)
	for(i=1; i < myNx; i++){
		for(j=1; j < Ny; j++){
			for(k=0; k < Nz; k++){

				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEz1 = (2.*eps_Ez[myidx] - econ_Ez[myidx]*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);
				CEz2 = (2.*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);

				// PEC condition.
				if(eps_Ez[myidx] > 1e3){
					CEz1 = 0.;
					CEz2 = 0.;
				}

				Ez_re[myidx] = CEz1 * Ez_re[myidx] + CEz2 * (diffxHy_re[myidx] - diffyHx_re[myidx]);
				//Ez_im[myidx] = CEz1 * Ez_im[myidx] + CEz2 * (diffxHy_im[myidx] - diffyHx_im[myidx]);

			}
		}
	}

	// Update Ey at x=0
	#pragma omp parallel for \
		shared(	myNx, Ny, Nz, dt, eps_Ey, econ_Ey, Ey_re, Ey_im, \
				diffxHz_re, diffxHz_im,	diffzHx_re, diffzHx_im)	\
		private(i, j, k, myidx, CEy1, CEy2)
	for(j=0; j < Ny; j++){
		for(k=1; k < Nz; k++){

			myidx   = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;

			CEy1 = (2.*eps_Ey[myidx] - econ_Ey[myidx]*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);
			CEy2 = (2.*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);

			// PEC condition.
			if(eps_Ey[myidx] > 1e3){
				CEy1 = 0.;
				CEy2 = 0.;
			}

			Ey_re[myidx] = CEy1 * Ey_re[myidx] + CEy2 * (diffzHx_re[myidx] - diffxHz_re[myidx]);
			//Ey_im[myidx] = CEy1 * Ey_im[myidx] + CEy2 * (diffzHx_im[myidx] - diffxHz_im[myidx]);

		}
	}

	// Update Ez at x=0
	#pragma omp parallel for \
		shared(	myNx, Ny, Nz, dt, eps_Ez, econ_Ez, Ez_re, Ez_im,\
				diffxHy_re, diffxHy_im, diffyHx_re, diffyHx_im)	\
		private(i, j, k, myidx, CEz1, CEz2)
	for(j=1; j < Ny; j++){
		for(k=0; k < Nz; k++){

			myidx   = (k  ) + (j  ) * Nz + (0  ) * Nz * Ny;

			CEz1 = (2.*eps_Ez[myidx] - econ_Ez[myidx]*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);
			CEz2 = (2.*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);

			// PEC condition.
			if(eps_Ez[myidx] > 1e3){
				CEz1 = 0.;
				CEz2 = 0.;
			}

			Ez_re[myidx] = CEz1 * Ez_re[myidx] + CEz2 * (diffxHy_re[myidx] - diffyHx_re[myidx]);
			//Ez_im[myidx] = CEz1 * Ez_im[myidx] + CEz2 * (diffxHy_im[myidx] - diffyHx_im[myidx]);

		}
	}

	return;
}

//Get derivatives of E field in the first and middle rank.
void get_diff_of_E_rankFM(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,			double  dx,		double	dy,		double dz,	\
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

	// int for index
	int i,j,k;
	int myidx, i_myidx, j_myidx, k_myidx, yzidx;

	// Update Hx
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dy, dz, Ey_re, Ey_im, Ez_re, Ez_im, diffyEz_re, diffyEz_im, diffzEy_re, diffzEy_im)	\
		private(i,j,k, myidx, j_myidx, k_myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				j_myidx = (k  ) + (j+1) * Nz + (i  ) * Nz * Ny;
				k_myidx = (k+1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffyEz_re[myidx] = (Ez_re[j_myidx] - Ez_re[myidx]) / dy;
				//diffyEz_im[myidx] = (Ez_im[j_myidx] - Ez_im[myidx]) / dy;

				diffzEy_re[myidx] = (Ey_re[k_myidx] - Ey_re[myidx]) / dz;
				//diffzEy_im[myidx] = (Ey_im[k_myidx] - Ey_im[myidx]) / dz;

			}
		}
	}

	// Update Hy
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dx, dz, Ex_re, Ex_im, Ez_re, Ez_im, diffxEz_re, diffxEz_im, diffzEx_re, diffzEx_im)	\
		private(i,j,k, myidx, i_myidx, k_myidx)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
				k_myidx = (k+1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffzEx_re[myidx] = (Ex_re[k_myidx] - Ex_re[myidx]) / dz;
				//diffzEx_im[myidx] = (Ex_im[k_myidx] - Ex_im[myidx]) / dz;

				diffxEz_re[myidx] = (Ez_re[i_myidx] - Ez_re[myidx]) / dx;
				//diffxEz_im[myidx] = (Ez_im[i_myidx] - Ez_im[myidx]) / dx;

			}
		}
	}

	// Update Hz
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dx, dy, Ex_re, Ex_im, Ey_re, Ey_im, diffxEy_re, diffxEy_im, diffyEx_re, diffyEx_im)	\
		private(i,j,k, myidx, i_myidx, j_myidx)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
				j_myidx = (k  ) + (j+1) * Nz + (i  ) * Nz * Ny;

				diffxEy_re[myidx] = (Ey_re[i_myidx] - Ey_re[myidx]) / dx;
				//diffxEy_im[myidx] = (Ey_im[i_myidx] - Ey_im[myidx]) / dx;

				diffyEx_re[myidx] = (Ex_re[j_myidx] - Ex_re[myidx]) / dy;
				//diffyEx_im[myidx] = (Ex_im[j_myidx] - Ex_im[myidx]) / dy;

			}
		}
	}

	/* Update Hy at x=myNx-1 */
	#pragma omp parallel for \
		shared(Ny, Nz, dx, dz, Ex_re, Ex_im, Ez_re, Ez_im, recvEzlast_re, recvEzlast_im, diffxEz_re, diffxEz_im, diffzEx_re, diffzEx_im)	\
		private(j,k, myidx, k_myidx, yzidx)
	for(j=0; j < Ny; j++){
		for(k=0; k < (Nz-1); k++){
				
			myidx   = (k  ) + (j  ) * Nz + (myNx-1) * Nz * Ny;
			k_myidx = (k+1) + (j  ) * Nz + (myNx-1) * Nz * Ny;
			yzidx   = (k  ) + (j  ) * Nz + (0     ) * Nz * Ny;

			diffzEx_re[myidx] = (Ex_re[k_myidx] - Ex_re[myidx]) / dz;
			//diffzEx_im[myidx] = (Ex_im[k_myidx] - Ex_im[myidx]) / dz;

			diffxEz_re[myidx] = (recvEzlast_re[yzidx] - Ez_re[myidx]) / dx;
			//diffxEz_im[myidx] = (recvEzlast_im[yzidx] - Ez_im[myidx]) / dx;

		}
	}

	/* Update Hz at x=myNx-1 */
	#pragma omp parallel for \
		shared(Ny, Nz, dx, dy, Ex_re, Ex_im, Ey_re, Ey_im, recvEylast_re, recvEylast_im, diffxEy_re, diffxEy_im, diffyEx_re, diffyEx_im)	\
		private(j,k, myidx, j_myidx, yzidx)
	for(j=0; j < (Ny-1); j++){
		for(k=0; k < Nz; k++){
				
			myidx   = (k  ) + (j  ) * Nz + (myNx-1) * Nz * Ny;
			j_myidx = (k  ) + (j+1) * Nz + (myNx-1) * Nz * Ny;
			yzidx   = (k  ) + (j  ) * Nz + (0     ) * Nz * Ny;

			diffxEy_re[myidx] = (recvEylast_re[yzidx] - Ey_re[myidx]) / dx;
			//diffxEy_im[myidx] = (recvEylast_im[yzidx] - Ey_im[myidx]) / dx;

			diffyEx_re[myidx] = (Ex_re[j_myidx] - Ex_re[myidx]) / dy;
			//diffyEx_im[myidx] = (Ex_im[j_myidx] - Ex_im[myidx]) / dy;

		}
	}

	return;
};


//Get derivatives of E field in the last rank.
void get_diff_of_E_rank_L(											\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,			double  dx,		double	dy,		double dz,	\
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

	// int for index
	int i,j,k;
	int myidx, i_myidx, j_myidx, k_myidx;

	// Update Hx
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dy, dz, Ey_re, Ey_im, Ez_re, Ez_im, diffyEz_re, diffyEz_im, diffzEy_re, diffzEy_im)	\
		private(i,j,k, myidx, j_myidx, k_myidx)
	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				j_myidx = (k  ) + (j+1) * Nz + (i  ) * Nz * Ny;
				k_myidx = (k+1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffyEz_re[myidx] = (Ez_re[j_myidx] - Ez_re[myidx]) / dy;
				//diffyEz_im[myidx] = (Ez_im[j_myidx] - Ez_im[myidx]) / dy;

				diffzEy_re[myidx] = (Ey_re[k_myidx] - Ey_re[myidx]) / dz;
				//diffzEy_im[myidx] = (Ey_im[k_myidx] - Ey_im[myidx]) / dz;

			}
		}
	}

	// Update Hy
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dx, dz, Ex_re, Ex_im, Ez_re, Ez_im, diffxEz_re, diffxEz_im, diffzEx_re, diffzEx_im)	\
		private(i,j,k, myidx, i_myidx, k_myidx)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
				k_myidx = (k+1) + (j  ) * Nz + (i  ) * Nz * Ny;

				diffzEx_re[myidx] = (Ex_re[k_myidx] - Ex_re[myidx]) / dz;
				//diffzEx_im[myidx] = (Ex_im[k_myidx] - Ex_im[myidx]) / dz;

				diffxEz_re[myidx] = (Ez_re[i_myidx] - Ez_re[myidx]) / dx;
				//diffxEz_im[myidx] = (Ez_im[i_myidx] - Ez_im[myidx]) / dx;

			}
		}
	}

	// Update Hz
	#pragma omp parallel for \
		shared(myNx, Ny, Nz, dx, dy, Ex_re, Ex_im, Ey_re, Ey_im, diffxEy_re, diffxEy_im, diffyEx_re, diffyEx_im)	\
		private(i,j,k, myidx, i_myidx, j_myidx)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				i_myidx = (k  ) + (j  ) * Nz + (i+1) * Nz * Ny;
				j_myidx = (k  ) + (j+1) * Nz + (i  ) * Nz * Ny;

				diffxEy_re[myidx] = (Ey_re[i_myidx] - Ey_re[myidx]) / dx;
				//diffxEy_im[myidx] = (Ey_im[i_myidx] - Ey_im[myidx]) / dx;

				diffyEx_re[myidx] = (Ex_re[j_myidx] - Ex_re[myidx]) / dy;
				//diffyEx_im[myidx] = (Ex_im[j_myidx] - Ex_im[myidx]) / dy;

			}
		}
	}

	return;
};


// Update H field.
void updateH_rankFM(														\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,														\
	double *mu_Hx,		double *mu_Hy,		double *mu_Hz,			\
	double *mcon_Hx,	double *mcon_Hy,	double *mcon_Hz,		\
	double *Hx_re,		double *Hx_im,								\
	double *Hy_re,		double *Hy_im,								\
	double *Hz_re,		double *Hz_im,								\
	double *diffxEy_re, double *diffxEy_im,							\
	double *diffxEz_re, double *diffxEz_im,							\
	double *diffyEx_re, double *diffyEx_im,							\
	double *diffyEz_re, double *diffyEz_im,							\
	double *diffzEx_re, double *diffzEx_im,							\
	double *diffzEy_re, double *diffzEy_im							\
){
	/* MAIN UPDATE EQUATIONS */

	int i,j,k;
	int myidx;

	double CHx1, CHx2;
	double CHy1, CHy2;
	double CHz1, CHz2;

	// Update Hx
	#pragma omp parallel for				\
		shared(	myNx, Ny, Nz, dt, mu_Hx, mcon_Hx, Hx_re, Hx_im,	\
				diffyEz_re, diffyEz_im, diffzEy_re, diffzEy_im)	\
		private(i, j, k, myidx, CHx1, CHx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHx1 =	(2.*mu_Hx[myidx] - mcon_Hx[myidx]*dt) / (2.*mu_Hx[myidx] + mcon_Hx[myidx]*dt);
				CHx2 =	(-2*dt) / (2.*mu_Hx[myidx] + mcon_Hx[myidx]*dt);

				Hx_re[myidx] = CHx1 * Hx_re[myidx] + CHx2 * (diffyEz_re[myidx] - diffzEy_re[myidx]);
				//Hx_im[myidx] = CHx1 * Hx_im[myidx] + CHx2 * (diffyEz_im[myidx] - diffzEy_im[myidx]);

			}
		}
	}
	
	// Update Hy
	#pragma omp parallel for				\
		shared(	myNx, Ny, Nz, dt, mu_Hy, mcon_Hy, Hy_re, Hy_im,	\
				diffxEz_re, diffxEz_im,	diffzEx_re, diffzEx_im)	\
		private(i, j, k, myidx, CHy1, CHy2)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHy1 =	(2.*mu_Hy[myidx] - mcon_Hy[myidx]*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);
				CHy2 =	(-2*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);

				Hy_re[myidx] = CHy1 * Hy_re[myidx] + CHy2 * (diffzEx_re[myidx] - diffxEz_re[myidx]);
				//Hy_im[myidx] = CHy1 * Hy_im[myidx] + CHy2 * (diffzEx_im[myidx] - diffxEz_im[myidx]);

			}
		}
	}
	
	// Update Hz
	#pragma omp parallel for				\
		shared(	myNx, Ny, Nz, dt, mu_Hz, mcon_Hz, Hz_re, Hz_im,	\
				diffxEy_re, diffxEy_im,	diffyEx_re, diffyEx_im)	\
		private(i, j, k, myidx, CHz1, CHz2)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHz1 =	(2.*mu_Hz[myidx] - mcon_Hz[myidx]*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);

				Hz_re[myidx] = CHz1 * Hz_re[myidx] + CHz2 * (diffxEy_re[myidx] - diffyEx_re[myidx]);
				//Hz_im[myidx] = CHz1 * Hz_im[myidx] + CHz2 * (diffxEy_im[myidx] - diffyEx_im[myidx]);
			}
		}
	}
	
	// Update Hy at x=myNx-1
	#pragma omp parallel for				\
		shared(	myNx, Ny, Nz, dt, mu_Hy, mcon_Hy, Hy_re, Hy_im,	\
				diffxEz_re, diffxEz_im,	diffzEx_re, diffzEx_im)	\
		private(i, j, k, myidx, CHy1, CHy2)
	for(j=0; j < Ny; j++){
		for(k=0; k < (Nz-1); k++){
			
			myidx   = (k  ) + (j  ) * Nz + (myNx-1) * Nz * Ny;

			CHy1 = (2.*mu_Hy[myidx] - mcon_Hy[myidx]*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);
			CHy2 = (-2*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);

			Hy_re[myidx] = CHy1 * Hy_re[myidx] + CHy2 * (diffzEx_re[myidx] - diffxEz_re[myidx]);
			//Hy_im[myidx] = CHy1 * Hy_im[myidx] + CHy2 * (diffzEx_im[myidx] - diffxEz_im[myidx]);

		}
	}

	// Update Hz at x=myNx-1
	#pragma omp parallel for				\
		shared(	myNx, Ny, Nz, dt, mu_Hz, mcon_Hz, Hz_re, Hz_im,	\
				diffxEy_re, diffxEy_im,	diffyEx_re, diffyEx_im)	\
		private(i, j, k, myidx, CHz1, CHz2)
	for(j=0; j < (Ny-1); j++){
		for(k=0; k < Nz; k++){
			
			myidx   = (k  ) + (j  ) * Nz + (myNx-1) * Nz * Ny;

			CHz1 = (2.*mu_Hz[myidx] - mcon_Hz[myidx]*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);
			CHz2 = (-2*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);

			Hz_re[myidx] = CHz1 * Hz_re[myidx] + CHz2 * (diffxEy_re[myidx] - diffyEx_re[myidx]);
			//Hz_im[myidx] = CHz1 * Hz_im[myidx] + CHz2 * (diffxEy_im[myidx] - diffyEx_im[myidx]);
		}
	}

	return;
}

void updateH_rank_L(												\
	int		myNx,		int		Ny,		int		Nz,					\
	double  dt,														\
	double *mu_Hx,		double *mu_Hy,		double *mu_Hz,			\
	double *mcon_Hx,	double *mcon_Hy,	double *mcon_Hz,		\
	double *Hx_re,		double *Hx_im,								\
	double *Hy_re,		double *Hy_im,								\
	double *Hz_re,		double *Hz_im,								\
	double *diffxEy_re, double *diffxEy_im,							\
	double *diffxEz_re, double *diffxEz_im,							\
	double *diffyEx_re, double *diffyEx_im,							\
	double *diffyEz_re, double *diffyEz_im,							\
	double *diffzEx_re, double *diffzEx_im,							\
	double *diffzEy_re, double *diffzEy_im							\
){
	/* MAIN UPDATE EQUATIONS */

	int i,j,k;
	int myidx;

	double CHx1, CHx2;
	double CHy1, CHy2;
	double CHz1, CHz2;

	// Update Hx
	#pragma omp parallel for				\
		shared(	myNx, Ny, Nz, dt, mu_Hx, mcon_Hx, Hx_re, Hx_im,	\
				diffyEz_re, diffyEz_im, diffzEy_re, diffzEy_im)	\
		private(i, j, k, myidx, CHx1, CHx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHx1 =	(2.*mu_Hx[myidx] - mcon_Hx[myidx]*dt) / (2.*mu_Hx[myidx] + mcon_Hx[myidx]*dt);
				CHx2 =	(-2*dt) / (2.*mu_Hx[myidx] + mcon_Hx[myidx]*dt);

				Hx_re[myidx] = CHx1 * Hx_re[myidx] + CHx2 * (diffyEz_re[myidx] - diffzEy_re[myidx]);
				//Hx_im[myidx] = CHx1 * Hx_im[myidx] + CHx2 * (diffyEz_im[myidx] - diffzEy_im[myidx]);

			}
		}
	}
	
	// Update Hy
	#pragma omp parallel for				\
		shared(	myNx, Ny, Nz, dt, mu_Hy, mcon_Hy, Hy_re, Hy_im,	\
				diffxEz_re, diffxEz_im,	diffzEx_re, diffzEx_im)	\
		private(i, j, k, myidx, CHy1, CHy2)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz-1); k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHy1 =	(2.*mu_Hy[myidx] - mcon_Hy[myidx]*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);
				CHy2 =	(-2*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);

				Hy_re[myidx] = CHy1 * Hy_re[myidx] + CHy2 * (diffzEx_re[myidx] - diffxEz_re[myidx]);
				//Hy_im[myidx] = CHy1 * Hy_im[myidx] + CHy2 * (diffzEx_im[myidx] - diffxEz_im[myidx]);

			}
		}
	}
	
	// Update Hz
	#pragma omp parallel for				\
		shared(	myNx, Ny, Nz, dt, mu_Hz, mcon_Hz, Hz_re, Hz_im,	\
				diffxEy_re, diffxEy_im,	diffyEx_re, diffyEx_im)	\
		private(i, j, k, myidx, CHz1, CHz2)
	for(i=0; i < (myNx-1); i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < Nz; k++){
				
				myidx   = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHz1 =	(2.*mu_Hz[myidx] - mcon_Hz[myidx]*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);

				Hz_re[myidx] = CHz1 * Hz_re[myidx] + CHz2 * (diffxEy_re[myidx] - diffyEx_re[myidx]);
				//Hz_im[myidx] = CHz1 * Hz_im[myidx] + CHz2 * (diffxEy_im[myidx] - diffyEx_im[myidx]);
			}
		}
	}
	
	return;
}
