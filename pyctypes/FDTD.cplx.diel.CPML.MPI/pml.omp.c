#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
	Auther: Donggun Lee
	Date  : 18.07.24

	Update equations of PML region is written in here.
	Simulation method is the FDTD method.
	Applied PML theory is Convolution PML(CPML) introduced by Roden and Gedney.
	For more details, see 'Convolution PML (CPML): An efficient FDTD implementation
	of the CFS-PML for arbitrary media', 22 June 2000, Microwave and Optical technology letters.

	Function declaration
	------------------------------

	void PML_updateH_px();
	void PML_updateH_mx();
	void PML_updateH_py();
	void PML_updateH_my();
	void PML_updateH_pz();
	void PML_updateH_mz();

	void PML_updateE_px();
	void PML_updateE_mx();
	void PML_updateE_py();
	void PML_updateE_my();
	void PML_updateE_pz();
	void PML_updateE_mz();
*/

/***********************************************************************************/
/******************************** FUNCTION DECLARATION *****************************/
/***********************************************************************************/

// PML at x+.
void PML_updateH_px(\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_Hy,			double *mu_Hz,							\
	double *mcon_Hy,		double *mcon_Hz,						\
	double *Hy_re,			double *Hy_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffxEy_re,		double *diffxEy_im,						\
	double *diffxEz_re,		double *diffxEz_im,						\
	double *psi_hyx_p_re,	double *psi_hyx_p_im,					\
	double *psi_hzx_p_re,	double *psi_hzx_p_im);


void PML_updateE_px(\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_Ey,			double *eps_Ez,							\
	double *econ_Ey,		double *econ_Ez,						\
	double *Ey_re,			double *Ey_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffxHy_re,		double *diffxHy_im,						\
	double *diffxHz_re,		double *diffxHz_im,						\
	double *psi_eyx_p_re,	double *psi_eyx_p_im,					\
	double *psi_ezx_p_re,	double *psi_ezx_p_im);


// PML at x-.
void PML_updateH_mx(\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_Hy,			double *mu_Hz,							\
	double *mcon_Hy,		double *mcon_Hz,						\
	double *Hy_re,			double *Hy_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffxEy_re,		double *diffxEy_im,						\
	double *diffxEz_re,		double *diffxEz_im,						\
	double *psi_hyx_m_re,	double *psi_hyx_m_im,					\
	double *psi_hzx_m_re,	double *psi_hzx_m_im);


void PML_updateE_mx(\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_Ey,			double *eps_Ez,							\
	double *econ_Ey,		double *econ_Ez,						\
	double *Ey_re,			double *Ey_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffxHy_re,		double *diffxHy_im,						\
	double *diffxHz_re,		double *diffxHz_im,						\
	double *psi_eyx_m_re,	double *psi_eyx_m_im,					\
	double *psi_ezx_m_re,	double *psi_ezx_m_im);


// PML at y+.
void PML_updateH_py(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_Hx,		   double *mu_Hz,							\
	double *mcon_Hx,	   double *mcon_Hz,							\
	double *Hx_re,			double *Hx_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffyEx_re,		double *diffyEx_im,						\
	double *diffyEz_re,		double *diffyEz_im,						\
	double *psi_hxy_p_re,	double *psi_hxy_p_im,					\
	double *psi_hzy_p_re,	double *psi_hzy_p_im					\
);


void PML_updateE_py(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_Ex,			double *eps_Ez,							\
	double *econ_Ex,		double *econ_Ez,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffyHx_re,		double *diffyHx_im,						\
	double *diffyHz_re,		double *diffyHz_im,						\
	double *psi_exy_p_re,	double *psi_exy_p_im,					\
	double *psi_ezy_p_re,	double *psi_ezy_p_im					\
);


// PML at y-.
void PML_updateH_my(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_Hx,			double *mu_Hz,							\
	double *mcon_Hx,		double *mcon_Hz,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffyEx_re,		double *diffyEx_im,						\
	double *diffyEz_re,		double *diffyEz_im,						\
	double *psi_hxy_m_re,	double *psi_hxy_m_im,					\
	double *psi_hzy_m_re,	double *psi_hzy_m_im					\
);
void PML_updateE_my(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_Ex,			double *eps_Ez,							\
	double *econ_Ex,		double *econ_Ez,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffyHx_re,		double *diffyHx_im,						\
	double *diffyHz_re,		double *diffyHz_im,						\
	double *psi_exy_m_re,	double *psi_exy_m_im,					\
	double *psi_ezy_m_re,	double *psi_ezy_m_im					\
);

// PML at z+.
void PML_updateH_pz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_Hx,			double *mu_Hy,							\
	double *mcon_Hx,		double *mcon_Hy,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hy_re,			double *Hy_im,							\
	double *diffzEx_re,		double *diffzEx_im,						\
	double *diffzEy_re,		double *diffzEy_im,						\
	double *psi_hxz_p_re,	double *psi_hxz_p_im,					\
	double *psi_hyz_p_re,	double *psi_hyz_p_im					\
);

void PML_updateE_pz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_Ex,			double *eps_Ey,							\
	double *econ_Ex,		double *econ_Ey,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ey_re,			double *Ey_im,							\
	double *diffzHx_re,		double *diffzHx_im,						\
	double *diffzHy_re,		double *diffzHy_im,						\
	double *psi_exz_p_re,	double *psi_exz_p_im,					\
	double *psi_eyz_p_re,	double *psi_eyz_p_im					\
);

// PML at z-.
void PML_updateH_mz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_Hx,			double *mu_Hy,							\
	double *mcon_Hx,		double *mcon_Hy,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hy_re,			double *Hy_im,							\
	double *diffzEx_re,		double *diffzEx_im,						\
	double *diffzEy_re,		double *diffzEy_im,						\
	double *psi_hxz_m_re,	double *psi_hxz_m_im,					\
	double *psi_hyz_m_re,	double *psi_hyz_m_im					\
);

void PML_updateE_mz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_Ex,			double *eps_Ey,							\
	double *econ_Ex,		double *econ_Ey,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ey_re,			double *Ey_im,							\
	double *diffzHx_re,		double *diffzHx_im,						\
	double *diffzHy_re,		double *diffzHy_im,						\
	double *psi_exz_m_re,	double *psi_exz_m_im,					\
	double *psi_eyz_m_re,	double *psi_eyz_m_im					\
);

/***********************************************************************************/
/******************************** FUNCTION DESCRIPTION *****************************/
/***********************************************************************************/

/*----------------------------------- PML at x+ -----------------------------------*/
void PML_updateH_px(\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_Hy,			double *mu_Hz,							\
	double *mcon_Hy,		double *mcon_Hz,						\
	double *Hy_re,			double *Hy_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffxEy_re,		double *diffxEy_im,						\
	double *diffxEz_re,		double *diffxEz_im,						\
	double *psi_hyx_p_re,	double *psi_hyx_p_im,					\
	double *psi_hzx_p_re,	double *psi_hzx_p_im){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHy2, CHz2;

	// Update Hy
	#pragma omp parallel for \
		shared(	npml, myNx, Ny, Nz,	dt,	\
				mu_Hy, mcon_Hy, Hy_re, Hy_im, diffxEz_re, diffxEz_im, \
				PMLkappax, PMLbx, PMLax, psi_hyx_p_re, psi_hyx_p_im) \
		private(i, j, k,odd, psiidx, myidx, CHy2)
	for(i=0; i < (npml-1); i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz-1); k++){
				
				odd    = 2*i + 1;
				psiidx = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

				CHy2 =	(-2*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);

				psi_hyx_p_re[psiidx] = (PMLbx[odd] * psi_hyx_p_re[psiidx]) + (PMLax[odd] * diffxEz_re[myidx]);
				//psi_hyx_p_im[psiidx] = (PMLbx[odd] * psi_hyx_p_im[psiidx]) + (PMLax[odd] * diffxEz_im[myidx]);

				Hy_re[myidx] += CHy2 * (-((1./PMLkappax[odd] - 1.) * diffxEz_re[myidx]) - psi_hyx_p_re[psiidx]);
				//Hy_im[myidx] += CHy2 * (-((1./PMLkappax[odd] - 1.) * diffxEz_im[myidx]) - psi_hyx_p_im[psiidx]);
			}
		}
	}

	// Update Hz
	#pragma omp parallel for \
		shared(	npml, myNx, Ny, Nz,	dt,	\
				mu_Hz, mcon_Hz,	Hz_re, Hz_im, diffxEy_re, diffxEy_im, \
				PMLkappax, PMLbx, PMLax, psi_hzx_p_re, psi_hzx_p_im) \
		private(i, j, k,odd, psiidx, myidx, CHz2)
	for(i=0; i < (npml-1); i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < Nz; k++){
				
				odd    = 2*i + 1;
				psiidx = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

				CHz2 =	(-2*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);

				psi_hzx_p_re[psiidx] = (PMLbx[odd] * psi_hzx_p_re[psiidx]) + (PMLax[odd] * diffxEy_re[myidx]);
				//psi_hzx_p_im[psiidx] = (PMLbx[odd] * psi_hzx_p_im[psiidx]) + (PMLax[odd] * diffxEy_im[myidx]);

				Hz_re[myidx] += CHz2 * (+((1./PMLkappax[odd] - 1.) * diffxEy_re[myidx]) + psi_hzx_p_re[psiidx]);
				//Hz_im[myidx] += CHz2 * (+((1./PMLkappax[odd] - 1.) * diffxEy_im[myidx]) + psi_hzx_p_im[psiidx]);
			}
		}
	}

	return;
}


void PML_updateE_px(\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_Ey,			double *eps_Ez,							\
	double *econ_Ey,		double *econ_Ez,						\
	double *Ey_re,			double *Ey_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffxHy_re,		double *diffxHy_im,						\
	double *diffxHz_re,		double *diffxHz_im,						\
	double *psi_eyx_p_re,	double *psi_eyx_p_im,					\
	double *psi_ezx_p_re,	double *psi_ezx_p_im){

	int i,j,k;
	int even;
	int psiidx, myidx;
	
	double CEy2, CEz2;

	// Update Ey. Note that i starts from 0, not 1.
	#pragma omp parallel for \
		shared(	npml, myNx, Ny, Nz,	dt,	\
				eps_Ey,	econ_Ey, Ey_re, Ey_im, diffxHz_re, diffxHz_im, \
				PMLkappax, PMLbx, PMLax, psi_eyx_p_re, psi_eyx_p_im) \
		private(i, j, k, even, psiidx, myidx, CEy2, CEz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=1; k < Nz; k++){

				even   = 2*i;
				psiidx = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

				CEy2 =	(2.*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);

				psi_eyx_p_re[psiidx] = (PMLbx[even] * psi_eyx_p_re[psiidx]) + (PMLax[even] * diffxHz_re[myidx]);
				//psi_eyx_p_im[psiidx] = (PMLbx[even] * psi_eyx_p_im[psiidx]) + (PMLax[even] * diffxHz_im[myidx]);

				Ey_re[myidx] += CEy2 * (-(1./PMLkappax[even] - 1.) * diffxHz_re[myidx] - psi_eyx_p_re[psiidx]);
				//Ey_im[myidx] += CEy2 * (-(1./PMLkappax[even] - 1.) * diffxHz_im[myidx] - psi_eyx_p_im[psiidx]);

			}		
		}
	}

	// Update Ez. Note that i starts from 0, not 1.
	#pragma omp parallel for \
		shared(	npml, myNx, Ny, Nz,	dt,	\
				eps_Ez, econ_Ez, Ez_re, Ez_im,	diffxHy_re, diffxHy_im,	\
				PMLkappax,	PMLbx,	PMLax, psi_ezx_p_re, psi_ezx_p_im)	\
		private(i, j, k, even, psiidx, myidx, CEy2, CEz2)
	for(i=0; i < npml; i++){
		for(j=1; j < Ny; j++){
			for(k=0; k < Nz; k++){

				even   = 2*i;
				psiidx = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

				CEz2 =	(2.*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);

				psi_ezx_p_re[psiidx] = (PMLbx[even] * psi_ezx_p_re[psiidx]) + (PMLax[even] * diffxHy_re[myidx]);
				//psi_ezx_p_im[psiidx] = (PMLbx[even] * psi_ezx_p_im[psiidx]) + (PMLax[even] * diffxHy_im[myidx]);

				Ez_re[myidx] += CEz2 * (+(1./PMLkappax[even] - 1.) * diffxHy_re[myidx] + psi_ezx_p_re[psiidx]);
				//Ez_im[myidx] += CEz2 * (+(1./PMLkappax[even] - 1.) * diffxHy_im[myidx] + psi_ezx_p_im[psiidx]);

			}		
		}
	}

	return;
}

/*----------------------------------- PML at x- -----------------------------------*/
void PML_updateH_mx(\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_Hy,			double *mu_Hz,							\
	double *mcon_Hy,		double *mcon_Hz,						\
	double *Hy_re,			double *Hy_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffxEy_re,		double *diffxEy_im,						\
	double *diffxEz_re,		double *diffxEz_im,						\
	double *psi_hyx_m_re,	double *psi_hyx_m_im,					\
	double *psi_hzx_m_re,	double *psi_hzx_m_im){

	int i,j,k;
	int even;
	int psiidx, myidx;
	
	double CHy2, CHz2;

	// Update Hy
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,	dt,	\
				mu_Hy, mcon_Hy,	Hy_re, Hy_im, diffxEz_re, diffxEz_im, \
				PMLkappax, PMLbx, PMLax,psi_hyx_m_re, psi_hyx_m_im) \
		private(i, j, k, even, psiidx, myidx, CHy2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < (Nz-1); k++){
				
				even   = (2*npml) - (2*i + 2);
				psiidx = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;

				CHy2 =	(-2*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);

				psi_hyx_m_re[psiidx] = (PMLbx[even] * psi_hyx_m_re[psiidx]) + (PMLax[even] * diffxEz_re[myidx]);
				//psi_hyx_m_im[psiidx] = (PMLbx[even] * psi_hyx_m_im[psiidx]) + (PMLax[even] * diffxEz_im[myidx]);

				Hy_re[myidx] += CHy2 * (-((1./PMLkappax[even] - 1.) * diffxEz_re[myidx]) - psi_hyx_m_re[psiidx]);
				//Hy_im[myidx] += CHy2 * (-((1./PMLkappax[even] - 1.) * diffxEz_im[myidx]) - psi_hyx_m_im[psiidx]);
			}
		}
	}

	// Update Hz
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,	dt,	\
				mu_Hz, mcon_Hz,	Hz_re, Hz_im, diffxEy_re, diffxEy_im, \
				PMLkappax, PMLbx, PMLax, psi_hzx_m_re, psi_hzx_m_im) \
		private(i, j, k, even, psiidx, myidx, CHz2)
	for(i=0; i < npml; i++){
		for(j=0; j < (Ny-1); j++){
			for(k=0; k < Nz; k++){
				
				even   = (2*npml) - (2*i + 2);
				psiidx = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;

				CHz2 =	(-2*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);

				psi_hzx_m_re[psiidx] = (PMLbx[even] * psi_hzx_m_re[psiidx]) + (PMLax[even] * diffxEy_re[myidx]);
				//psi_hzx_m_im[psiidx] = (PMLbx[even] * psi_hzx_m_im[psiidx]) + (PMLax[even] * diffxEy_im[myidx]);

				Hz_re[myidx] += CHz2 * (+((1./PMLkappax[even] - 1.) * diffxEy_re[myidx]) + psi_hzx_m_re[psiidx]);
				//Hz_im[myidx] += CHz2 * (+((1./PMLkappax[even] - 1.) * diffxEy_im[myidx]) + psi_hzx_m_im[psiidx]);
			}
		}
	}

	return;
}

void PML_updateE_mx(\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_Ey,			double *eps_Ez,							\
	double *econ_Ey,		double *econ_Ez,						\
	double *Ey_re,			double *Ey_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffxHy_re,		double *diffxHy_im,						\
	double *diffxHz_re,		double *diffxHz_im,						\
	double *psi_eyx_m_re,	double *psi_eyx_m_im,					\
	double *psi_ezx_m_re,	double *psi_ezx_m_im){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEy2, CEz2;

	// Update Ey.
	#pragma omp parallel for		\
		shared(	npml, myNx, Ny, Nz,	dt,	\
				eps_Ey,	econ_Ey, Ey_re, Ey_im, diffxHz_re, diffxHz_im, \
				PMLkappax, PMLbx, PMLax, psi_eyx_m_re, psi_eyx_m_im) \
		private(i, j, k, odd, psiidx, myidx, CEy2)
	for(i=1; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=1; k < Nz; k++){

				odd    = (2*npml) - (2*i+1);
				psiidx = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;

				CEy2 =	(2.*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);

				psi_eyx_m_re[psiidx] = (PMLbx[odd] * psi_eyx_m_re[psiidx]) + (PMLax[odd] * diffxHz_re[myidx]);
				//psi_eyx_m_im[psiidx] = (PMLbx[odd] * psi_eyx_m_im[psiidx]) + (PMLax[odd] * diffxHz_im[myidx]);

				Ey_re[myidx] += CEy2 * (-(1./PMLkappax[odd] - 1.) * diffxHz_re[myidx] - psi_eyx_m_re[psiidx]);
				//Ey_im[myidx] += CEy2 * (-(1./PMLkappax[odd] - 1.) * diffxHz_im[myidx] - psi_eyx_m_im[psiidx]);

			}		
		}
	}

	// Update Ez.
	#pragma omp parallel for \
		shared(	npml, myNx, Ny, Nz,	dt,	\
				eps_Ez, econ_Ez, Ez_re, Ez_im, diffxHy_re, diffxHy_im, \
				PMLkappax, PMLbx, PMLax, psi_ezx_m_re, psi_ezx_m_im)\
		private(i, j, k, odd, psiidx, myidx, CEy2, CEz2)
	for(i=1; i < npml; i++){
		for(j=1; j < Ny; j++){
			for(k=0; k < Nz; k++){

				odd    = (2*npml) - (2*i+1);
				psiidx = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;

				CEz2 =	(2.*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);

				psi_ezx_m_re[psiidx] = (PMLbx[odd] * psi_ezx_m_re[psiidx]) + (PMLax[odd] * diffxHy_re[myidx]);
				//psi_ezx_m_im[psiidx] = (PMLbx[odd] * psi_ezx_m_im[psiidx]) + (PMLax[odd] * diffxHy_im[myidx]);

				Ez_re[myidx] += CEz2 * (+(1./PMLkappax[odd] - 1.) * diffxHy_re[myidx] + psi_ezx_m_re[psiidx]);
				//Ez_im[myidx] += CEz2 * (+(1./PMLkappax[odd] - 1.) * diffxHy_im[myidx] + psi_ezx_m_im[psiidx]);

			}
		}
	}

	return;
}


/***************************** PML at y+ *********************************/
void PML_updateH_py(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_Hx,			double *mu_Hz,							\
	double *mcon_Hx,		double *mcon_Hz,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffyEx_re,		double *diffyEx_im,						\
	double *diffyEz_re,		double *diffyEz_im,						\
	double *psi_hxy_p_re,	double *psi_hxy_p_im,					\
	double *psi_hzy_p_re,	double *psi_hzy_p_im					\
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHx2, CHz2;

	/* Since we already calculated derivatives for all points,
		We don't need to match the indexes. Just run the 3 level loops for all (Ny,Ny,Nz). */

	// Update Hx
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				mu_Hx, mcon_Hx, Hx_re, Hx_im, diffyEz_re, diffyEz_im, \
				PMLkappay, PMLby, PMLay, psi_hxy_p_re, psi_hxy_p_im) \
		private(i, j, k, odd, psiidx, myidx, CHx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				odd    = 2*j+1;
				psiidx = (k  ) + (j		   )*Nz + (i	)*Nz*npml;
				myidx  = (k  ) + (j+Ny-npml)*Nz + (i	)*Nz*Ny;
				
				CHx2 =	(-2*dt) / (2.*mu_Hx[myidx] + mcon_Hx[myidx]*dt);
				
				psi_hxy_p_re[psiidx] = (PMLby[odd] * psi_hxy_p_re[psiidx]) + (PMLay[odd] * diffyEz_re[myidx]);
				//psi_hxy_p_im[psiidx] = (PMLby[odd] * psi_hxy_p_im[psiidx]) + (PMLay[odd] * diffyEz_im[myidx]);
				
				Hx_re[myidx] += CHx2 * (+((1./PMLkappay[odd] - 1.) * diffyEz_re[myidx]) + psi_hxy_p_re[psiidx]);
				//Hx_im[myidx] += CHx2 * (+((1./PMLkappay[odd] - 1.) * diffyEz_im[myidx]) + psi_hxy_p_im[psiidx]);
				
			}
		}
	}

	// Update Hz
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				mu_Hz, mcon_Hz, Hz_re, Hz_im, diffyEx_re, diffyEx_im, \
				PMLkappay, PMLby, PMLay, psi_hzy_p_re, psi_hzy_p_im) \
		private(i, j, k, odd, psiidx, myidx, CHz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				
				odd    = 2*j+1;
				psiidx = (k  ) + (j		   ) * Nz + (i	) * Nz * npml;
				myidx  = (k  ) + (j+Ny-npml) * Nz + (i	) * Nz * Ny;
				
				CHz2 =	(-2*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);
				
				psi_hzy_p_re[psiidx] = (PMLby[odd] * psi_hzy_p_re[psiidx]) + (PMLay[odd] * diffyEx_re[myidx]);
				//psi_hzy_p_im[psiidx] = (PMLby[odd] * psi_hzy_p_im[psiidx]) + (PMLay[odd] * diffyEx_im[myidx]);
				
				Hz_re[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_re[myidx]) - psi_hzy_p_re[psiidx]);
				//Hz_im[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_im[myidx]) - psi_hzy_p_im[psiidx]);

			}
		}
	}

	return;
};


void PML_updateE_py(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_Ex,			double *eps_Ez,							\
	double *econ_Ex,		double *econ_Ez,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffyHx_re,		double *diffyHx_im,						\
	double *diffyHz_re,		double *diffyHz_im,						\
	double *psi_exy_p_re,	double *psi_exy_p_im,					\
	double *psi_ezy_p_re,	double *psi_ezy_p_im					\
){
	int i,j,k;
	int even;
	int psiidx, myidx;

	double CEx2, CEz2;

	/* Since we already calculated derivatives for all points,
		We don't need to match the indexes. Just run the 3 level loops for all (Ny,Ny,Nz). */

	// Update Ex.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				eps_Ex, econ_Ex, Ex_re, Ex_im, diffyHz_re, diffyHz_im, \
				PMLkappay, PMLby, PMLay, psi_exy_p_re, psi_exy_p_im) \
		private(i, j, k, even, psiidx, myidx, CEx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){

				even   = 2*j;
				psiidx = (k  ) + (j		   ) * Nz + (i	) * Nz * npml;
				myidx  = (k  ) + (j+Ny-npml) * Nz + (i	) * Nz * Ny;

				CEx2 =	(2*dt) / (2.*eps_Ex[myidx] + econ_Ex[myidx]*dt);

				psi_exy_p_re[psiidx] = (PMLby[even] * psi_exy_p_re[psiidx]) + (PMLay[even] * diffyHz_re[myidx]);
				//psi_exy_p_im[psiidx] = (PMLby[even] * psi_exy_p_im[psiidx]) + (PMLay[even] * diffyHz_im[myidx]);

				Ex_re[myidx] += CEx2 * (+((1./PMLkappay[even] - 1.) * diffyHz_re[myidx]) + psi_exy_p_re[psiidx]);
				//Ex_im[myidx] += CEx2 * (+((1./PMLkappay[even] - 1.) * diffyHz_im[myidx]) + psi_exy_p_im[psiidx]);

			}
		}
	}

	// Update Ez.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				eps_Ez, econ_Ez, Ez_re, Ez_im, diffyHx_re, diffyHx_im, \
				PMLkappay, PMLby, PMLay, psi_ezy_p_re, psi_ezy_p_im) \
		private(i, j, k, even, psiidx, myidx, CEz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){

				even   = 2*j;
				psiidx = (k  ) + (j		   ) * Nz + (i	) * Nz * npml;
				myidx  = (k  ) + (j+Ny-npml) * Nz + (i	) * Nz * Ny;

				CEz2 =	(2*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);

				psi_ezy_p_re[psiidx] = (PMLby[even] * psi_ezy_p_re[psiidx]) + (PMLay[even] * diffyHx_re[myidx]);
				//psi_ezy_p_im[psiidx] = (PMLby[even] * psi_ezy_p_im[psiidx]) + (PMLay[even] * diffyHx_im[myidx]);

				Ez_re[myidx] += CEz2 * (-((1./PMLkappay[even] - 1.) * diffyHx_re[myidx]) - psi_ezy_p_re[psiidx]);
				//Ez_im[myidx] += CEz2 * (-((1./PMLkappay[even] - 1.) * diffyHx_im[myidx]) - psi_ezy_p_im[psiidx]);
			}
		}
	}

	return;

};


/***************************** PML at y- *********************************/
void PML_updateH_my(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_Hx,			double *mu_Hz,							\
	double *mcon_Hx,		double *mcon_Hz,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffyEx_re,		double *diffyEx_im,						\
	double *diffyEz_re,		double *diffyEz_im,						\
	double *psi_hxy_m_re,	double *psi_hxy_m_im,					\
	double *psi_hzy_m_re,	double *psi_hzy_m_im					\
){
    int i,j,k;
    int even;
    int psiidx, myidx;

    double CHx2, CHz2;

	/* Since we already calculated derivatives for all points,
		We don't need to match the indexes. Just run the 3 level loops for all (Ny,Ny,Nz). */

	// Update Hx.
    #pragma omp parallel for \
        shared( npml, myNx, Ny, Nz, dt, \
                mu_Hx, mcon_Hx, Hx_re, Hx_im, diffyEz_re, diffyEz_im, \
                PMLkappay, PMLby, PMLay, psi_hxy_m_re, psi_hxy_m_im) \
        private(i, j, k, even, psiidx, myidx, CHx2)
    for(i=0; i < myNx; i++){
        for(j=0; j < npml; j++){
            for(k=0; k < Nz; k++){

                even   = (2*npml) - (2*j+2);
                psiidx = (k  ) + (j  ) * Nz + (i  ) * Nz * npml;
                myidx  = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

                CHx2 =  (-2*dt) / (2.*mu_Hx[myidx] + mcon_Hx[myidx]*dt);

                psi_hxy_m_re[psiidx] = (PMLby[even] * psi_hxy_m_re[psiidx]) + (PMLay[even] * diffyEz_re[myidx]);
                //psi_hxy_m_im[psiidx] = (PMLby[even] * psi_hxy_m_im[psiidx]) + (PMLay[even] * diffyEz_im[myidx]);

                Hx_re[myidx] += CHx2 * (+((1./PMLkappay[even] - 1.) * diffyEz_re[myidx]) + psi_hxy_m_re[psiidx]);
                //Hx_im[myidx] += CHx2 * (+((1./PMLkappay[even] - 1.) * diffyEz_im[myidx]) + psi_hxy_m_im[psiidx]);

            }
        }
    }

	// Update Hz.
	#pragma omp parallel for            \
		shared( npml, myNx, Ny, Nz, dt, \
				mu_Hz, mcon_Hz,	Hz_re, Hz_im, diffyEx_re, diffyEx_im, \
				PMLkappay, PMLby, PMLay, psi_hzy_m_re, psi_hzy_m_im) \
		private(i, j, k, even, psiidx, myidx, CHx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){

				even   = (2*npml) - (2*j+2);
				psiidx = (k  ) + (j  ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CHz2 =  (-2*dt) / (2.*mu_Hz[myidx] + mcon_Hz[myidx]*dt);

				psi_hzy_m_re[psiidx] = (PMLby[even] * psi_hzy_m_re[psiidx]) + (PMLay[even] * diffyEx_re[myidx]);
				//psi_hzy_m_im[psiidx] = (PMLby[even] * psi_hzy_m_im[psiidx]) + (PMLay[even] * diffyEx_im[myidx]);

				Hz_re[myidx] += CHz2 * (-((1./PMLkappay[even] - 1.) * diffyEx_re[myidx]) - psi_hzy_m_re[psiidx]);
				//Hz_im[myidx] += CHz2 * (-((1./PMLkappay[even] - 1.) * diffyEx_im[myidx]) - psi_hzy_m_im[psiidx]);
			}
		}
	}
	return;
};


void PML_updateE_my(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_Ex,			double *eps_Ez,							\
	double *econ_Ex,		double *econ_Ez,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffyHx_re,		double *diffyHx_im,						\
	double *diffyHz_re,		double *diffyHz_im,						\
	double *psi_exy_m_re,	double *psi_exy_m_im,					\
	double *psi_ezy_m_re,	double *psi_ezy_m_im					\
){
    int i,j,k;
    int odd;
    int psiidx;
	int myidx;

    double CEx2, CEz2;

	/* Since we already calculated derivatives for all points,
		We don't need to match the indexes. Just run the 3 level loops for all (Ny,Ny,Nz). */

	// Update Ex
    #pragma omp parallel for            \
        shared( npml, myNx, Ny, Nz, dt, \
                eps_Ex, econ_Ex, Ex_re, Ex_im, diffyHz_re, diffyHz_im, \
                PMLkappay, PMLby, PMLay, psi_exy_m_re, psi_exy_m_im) \
        private(i, j, k, odd, psiidx, myidx, CEx2)
    for(i=0; i < myNx; i++){
        for(j=0; j < npml; j++){
            for(k=0; k < Nz; k++){

                odd    = (2*npml) - (2*j+1);
                psiidx = (k  ) + (j  ) * Nz + (i  ) * Nz * npml;
                myidx  = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

                CEx2 =  (2*dt) / (2.*eps_Ex[myidx] + econ_Ex[myidx]*dt);

                psi_exy_m_re[psiidx] = (PMLby[odd] * psi_exy_m_re[psiidx]) + (PMLay[odd] * diffyHz_re[myidx]);
                //psi_exy_m_im[psiidx] = (PMLby[odd] * psi_exy_m_im[psiidx]) + (PMLay[odd] * diffyHz_im[myidx]);

                Ex_re[myidx] += CEx2 * (+((1./PMLkappay[odd] - 1.) * diffyHz_re[myidx]) + psi_exy_m_re[psiidx]);
                //Ex_im[myidx] += CEx2 * (+((1./PMLkappay[odd] - 1.) * diffyHz_im[myidx]) + psi_exy_m_im[psiidx]);

            }
        }
    }

	// Update Ez
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				eps_Ez, econ_Ez, Ez_re, Ez_im, diffyHx_re, diffyHx_im, \
				PMLkappay, PMLby, PMLay, psi_ezy_m_re, psi_ezy_m_im) \
		private(i, j, k, odd, psiidx, myidx, CEz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){

				odd    = (2*npml) - (2*j+1);
				psiidx = (k  ) + (j  ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEz2 =  (2*dt) / (2.*eps_Ez[myidx] + econ_Ez[myidx]*dt);

				psi_ezy_m_re[psiidx] = (PMLby[odd] * psi_ezy_m_re[psiidx]) + (PMLay[odd] * diffyHx_re[myidx]);
				//psi_ezy_m_im[psiidx] = (PMLby[odd] * psi_ezy_m_im[psiidx]) + (PMLay[odd] * diffyHx_im[myidx]);

				Ez_re[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_re[myidx]) - psi_ezy_m_re[psiidx]);
				//Ez_im[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_im[myidx]) - psi_ezy_m_im[psiidx]);
			}
		}
	}

    return;
};


/***************************** PML at z+ *********************************/
void PML_updateH_pz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_Hx,			double *mu_Hy,							\
	double *mcon_Hx,		double *mcon_Hy,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hy_re,			double *Hy_im,							\
	double *diffzEx_re,		double *diffzEx_im,						\
	double *diffzEy_re,		double *diffzEy_im,						\
	double *psi_hxz_p_re,	double *psi_hxz_p_im,					\
	double *psi_hyz_p_re,	double *psi_hyz_p_im					\
){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHx2, CHy2;

	/* Since we already calculated derivatives for all points,
		We don't need to match the indexes. Just run the 3 level loops for all (Ny,Ny,Nz). */

	// Update Hx.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				mu_Hx, mcon_Hx, Hx_re, Hx_im, diffzEy_re, diffzEy_im, \
				PMLkappaz, PMLbz, PMLaz, psi_hxz_p_re, psi_hxz_p_im) \
		private(i, j, k, odd, psiidx, myidx, CHx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = 2*k+1;
				psiidx = (k        ) + (j  )*npml + (i	)*Ny*npml;
				myidx  = (k+Nz-npml) + (j  )*Nz   + (i	)*Ny*Nz;
				
				CHx2 =	(-2*dt) / (2.*mu_Hx[myidx] + mcon_Hx[myidx]*dt);
				
				psi_hxz_p_re[psiidx] = (PMLbz[odd] * psi_hxz_p_re[psiidx]) + (PMLaz[odd] * diffzEy_re[myidx]);
				//psi_hxz_p_im[psiidx] = (PMLbz[odd] * psi_hxz_p_im[psiidx]) + (PMLaz[odd] * diffzEy_im[myidx]);
				
				Hx_re[myidx] += CHx2 * (-((1./PMLkappaz[odd] - 1.) * diffzEy_re[myidx]) - psi_hxz_p_re[psiidx]);
				//Hx_im[myidx] += CHx2 * (-((1./PMLkappaz[odd] - 1.) * diffzEy_im[myidx]) - psi_hxz_p_im[psiidx]);
				
			}
		}
	}

	// Update Hy.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				mu_Hy, mcon_Hy, Hy_re, Hy_im, diffzEx_re, diffzEx_im, \
				PMLkappaz, PMLbz, PMLaz, psi_hyz_p_re, psi_hyz_p_im) \
		private(i, j, k, odd, psiidx, myidx, CHy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = 2*k+1;
				psiidx = (k        ) + (j  )*npml + (i	)*Ny*npml;
				myidx  = (k+Nz-npml) + (j  )*Nz   + (i	)*Ny*Nz;
				
				CHy2 =	(-2*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);
				
				psi_hyz_p_re[psiidx] = (PMLbz[odd] * psi_hyz_p_re[psiidx]) + (PMLaz[odd] * diffzEx_re[myidx]);
				//psi_hyz_p_im[psiidx] = (PMLbz[odd] * psi_hyz_p_im[psiidx]) + (PMLaz[odd] * diffzEx_im[myidx]);
				
				Hy_re[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_re[myidx]) + psi_hyz_p_re[psiidx]);
				//Hy_im[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_im[myidx]) + psi_hyz_p_im[psiidx]);
			}
		}
	}

	return;

};

void PML_updateE_pz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_Ex,			double *eps_Ey,							\
	double *econ_Ex,		double *econ_Ey,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ey_re,			double *Ey_im,							\
	double *diffzHx_re,		double *diffzHx_im,						\
	double *diffzHy_re,		double *diffzHy_im,						\
	double *psi_exz_p_re,	double *psi_exz_p_im,					\
	double *psi_eyz_p_re,	double *psi_eyz_p_im					\
){
	int i,j,k;
	int even;
	int psiidx, myidx;

	double CEx2, CEy2;

	/* Since we already calculated derivatives for all points,
		We don't need to match the indexes. Just run the 3 level loops for all (Ny,Ny,Nz). */

	// Update Ex.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				eps_Ex, econ_Ex, Ex_re, Ex_im, diffzHy_re, diffzHy_im, \
				PMLkappaz, PMLbz, PMLaz, psi_exz_p_re, psi_exz_p_im) \
		private(i, j, k, even, psiidx, myidx, CEx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){

				even   = 2*k;
				psiidx = (k        ) + (j  )*npml + (i  )*Ny*npml;
				myidx  = (k+Nz-npml) + (j  )*Nz   + (i  )*Nz*Ny;

				CEx2 =	(2*dt) / (2.*eps_Ex[myidx] + econ_Ex[myidx]*dt);

				psi_exz_p_re[psiidx] = (PMLbz[even] * psi_exz_p_re[psiidx]) + (PMLaz[even] * diffzHy_re[myidx]);
				//psi_exz_p_im[psiidx] = (PMLbz[even] * psi_exz_p_im[psiidx]) + (PMLaz[even] * diffzHy_im[myidx]);

				Ex_re[myidx] += CEx2 * (-((1./PMLkappaz[even] - 1.) * diffzHy_re[myidx]) - psi_exz_p_re[psiidx]);
				//Ex_im[myidx] += CEx2 * (-((1./PMLkappaz[even] - 1.) * diffzHy_im[myidx]) - psi_exz_p_im[psiidx]);

			}
		}
	}

	// Update Ey.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				eps_Ey, econ_Ey, Ey_re, Ey_im, diffzHx_re, diffzHx_im, \
				PMLkappaz, PMLbz, PMLaz, psi_eyz_p_re, psi_eyz_p_im) \
		private(i, j, k, even, psiidx, myidx, CEy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){

				even   = 2*k;
				psiidx = (k        ) + (j  )*npml + (i  )*Ny*npml;
				myidx  = (k+Nz-npml) + (j  )*Nz   + (i  )*Nz*Ny;

				CEy2 =	(2*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);

				psi_eyz_p_re[psiidx] = (PMLbz[even] * psi_eyz_p_re[psiidx]) + (PMLaz[even] * diffzHx_re[myidx]);
				//psi_eyz_p_im[psiidx] = (PMLbz[even] * psi_eyz_p_im[psiidx]) + (PMLaz[even] * diffzHx_im[myidx]);

				Ey_re[myidx] += CEy2 * (+((1./PMLkappaz[even] - 1.) * diffzHx_re[myidx]) + psi_eyz_p_re[psiidx]);
				//Ey_im[myidx] += CEy2 * (+((1./PMLkappaz[even] - 1.) * diffzHx_im[myidx]) + psi_eyz_p_im[psiidx]);
			}
		}
	}

	return;
};

/***************************** PML at z- *********************************/
void PML_updateH_mz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_Hx,			double *mu_Hy,							\
	double *mcon_Hx,		double *mcon_Hy,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hy_re,			double *Hy_im,							\
	double *diffzEx_re,		double *diffzEx_im,						\
	double *diffzEy_re,		double *diffzEy_im,						\
	double *psi_hxz_m_re,	double *psi_hxz_m_im,					\
	double *psi_hyz_m_re,	double *psi_hyz_m_im					\
){

	/* Since we already calculated derivatives for all points,
		We don't need to match the indexes. Just run the 3 level loops for all (Ny,Ny,Nz). */

	int i,j,k;
	int even;
	int psiidx, myidx;
	
	double CHx2, CHy2;

	// Update Hx. Note that j and k end at Ny and npml, not Ny-1 and npml-1.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				mu_Hx, mcon_Hx, Hx_re, Hx_im, diffzEy_re, diffzEy_im, \
				PMLkappaz, PMLbz, PMLaz, psi_hxz_m_re, psi_hxz_m_im) \
		private(i, j, k, even, psiidx, myidx, CHx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				even   = (2*npml) - (2*k+2);
				psiidx = (k  ) + (j  )*npml + (i  )*Ny*npml;
				myidx  = (k  ) + (j  )*Nz   + (i  )*Ny*Nz;
				
				CHx2 =	(-2*dt) / (2.*mu_Hx[myidx] + mcon_Hx[myidx]*dt);
				
				psi_hxz_m_re[psiidx] = (PMLbz[even] * psi_hxz_m_re[psiidx]) + (PMLaz[even] * diffzEy_re[myidx]);
				//psi_hxz_m_im[psiidx] = (PMLbz[even] * psi_hxz_m_im[psiidx]) + (PMLaz[even] * diffzEy_im[myidx]);
				
				Hx_re[myidx] += CHx2 * (-((1./PMLkappaz[even] - 1.) * diffzEy_re[myidx]) - psi_hxz_m_re[psiidx]);
				//Hx_im[myidx] += CHx2 * (-((1./PMLkappaz[even] - 1.) * diffzEy_im[myidx]) - psi_hxz_m_im[psiidx]);
				
			}
		}
	}

	// Update Hy.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				mu_Hy, mcon_Hy, Hy_re, Hy_im, diffzEx_re, diffzEx_im, \
				PMLkappaz, PMLbz, PMLaz, psi_hyz_m_re, psi_hyz_m_im) \
		private(i, j, k, even, psiidx, myidx, CHy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				even   = (2*npml) - (2*k+2);
				psiidx = (k  ) + (j  )*npml + (i  )*Ny*npml;
				myidx  = (k  ) + (j  )*Nz   + (i  )*Ny*Nz;
				
				CHy2 =	(-2*dt) / (2.*mu_Hy[myidx] + mcon_Hy[myidx]*dt);
				
				psi_hyz_m_re[psiidx] = (PMLbz[even] * psi_hyz_m_re[psiidx]) + (PMLaz[even] * diffzEx_re[myidx]);
				//psi_hyz_m_im[psiidx] = (PMLbz[even] * psi_hyz_m_im[psiidx]) + (PMLaz[even] * diffzEx_im[myidx]);
				
				Hy_re[myidx] += CHy2 * (+((1./PMLkappaz[even] - 1.) * diffzEx_re[myidx]) + psi_hyz_m_re[psiidx]);
				//Hy_im[myidx] += CHy2 * (+((1./PMLkappaz[even] - 1.) * diffzEx_im[myidx]) + psi_hyz_m_im[psiidx]);
			}
		}
	}

	return;

};

void PML_updateE_mz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double	dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_Ex,			double *eps_Ey,							\
	double *econ_Ex,		double *econ_Ey,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ey_re,			double *Ey_im,							\
	double *diffzHx_re,		double *diffzHx_im,						\
	double *diffzHy_re,		double *diffzHy_im,						\
	double *psi_exz_m_re,	double *psi_exz_m_im,					\
	double *psi_eyz_m_re,	double *psi_eyz_m_im					\
){

	/* Since we already calculated derivatives for all points,
		We don't need to match the indexes. Just run the 3 level loops for all (Ny,Ny,Nz). */

	int i,j,k;
	int odd;
	int psiidx, myidx;

	double CEx2, CEy2;

	// Update Ex.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				eps_Ex, econ_Ex, Ex_re, Ex_im, diffzHy_re, diffzHy_im, \
				PMLkappaz, PMLbz, PMLaz, psi_exz_m_re, psi_exz_m_im) \
		private(i, j, k, odd, psiidx, myidx, CEx2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){

				odd    = (2*npml) - (2*k+1);
				psiidx = (k  ) + (j  )*npml + (i  )*Ny*npml;
				myidx  = (k  ) + (j  )*Nz   + (i  )*Nz*Ny;

				CEx2 =	(2*dt) / (2.*eps_Ex[myidx] + econ_Ex[myidx]*dt);

				psi_exz_m_re[psiidx] = (PMLbz[odd] * psi_exz_m_re[psiidx]) + (PMLaz[odd] * diffzHy_re[myidx]);
				//psi_exz_m_im[psiidx] = (PMLbz[odd] * psi_exz_m_im[psiidx]) + (PMLaz[odd] * diffzHy_im[myidx]);

				Ex_re[myidx] += CEx2 * (-((1./PMLkappaz[odd] - 1.) * diffzHy_re[myidx]) - psi_exz_m_re[psiidx]);
				//Ex_im[myidx] += CEx2 * (-((1./PMLkappaz[odd] - 1.) * diffzHy_im[myidx]) - psi_exz_m_im[psiidx]);

			}
		}
	}

	// Update Ey.
	#pragma omp parallel for \
		shared( npml, myNx, Ny, Nz, dt, \
				eps_Ey, econ_Ey, Ey_re, Ey_im, diffzHx_re, diffzHx_im, \
				PMLkappaz, PMLbz, PMLaz, psi_eyz_m_re, psi_eyz_m_im) \
		private(i, j, k, odd, psiidx, myidx, CEy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){

				odd    = (2*npml) - (2*k+1);
				psiidx = (k  ) + (j  )*npml + (i  )*Ny*npml;
				myidx  = (k  ) + (j  )*Nz   + (i  )*Nz*Ny;

				CEy2 =	(2*dt) / (2.*eps_Ey[myidx] + econ_Ey[myidx]*dt);

				psi_eyz_m_re[psiidx] = (PMLbz[odd] * psi_eyz_m_re[psiidx]) + (PMLaz[odd] * diffzHx_re[myidx]);
				//psi_eyz_m_im[psiidx] = (PMLbz[odd] * psi_eyz_m_im[psiidx]) + (PMLaz[odd] * diffzHx_im[myidx]);

				Ey_re[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_re[myidx]) + psi_eyz_m_re[psiidx]);
				//Ey_im[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_im[myidx]) + psi_eyz_m_im[psiidx]);
			}
		}
	}

	return;

};
