#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

/*
	Auther: Donggun Lee
	Date  : 18.07.24

	Update equations of PML region is written in here.
	Simulation method is the hybrid PSTD-FDTD method.
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
void PML_updateH_px(												\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hy_re,			double *Hy_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffxEy_re,		double *diffxEy_im,						\
	double *diffxEz_re,		double *diffxEz_im,						\
	double *psi_hyx_p_re,	double *psi_hyx_p_im,					\
	double *psi_hzx_p_re,	double *psi_hzx_p_im					\
);
void PML_updateE_px(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ey_re,			double *Ey_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffxHy_re,		double *diffxHy_im,						\
	double *diffxHz_re,		double *diffxHz_im,						\
	double *psi_eyx_p_re,	double *psi_eyx_p_im,					\
	double *psi_ezx_p_re,	double *psi_ezx_p_im					\
);

// PML at x-.
void PML_updateH_mx(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hy_re,			double *Hy_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffxEy_re,		double *diffxEy_im,						\
	double *diffxEz_re,		double *diffxEz_im,						\
	double *psi_hyx_m_re,	double *psi_hyx_m_im,					\
	double *psi_hzx_m_re,	double *psi_hzx_m_im					\
);
void PML_updateE_mx(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ey_re,			double *Ey_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffxHy_re,		double *diffxHy_im,						\
	double *diffxHz_re,		double *diffxHz_im,						\
	double *psi_eyx_m_re,	double *psi_eyx_m_im,					\
	double *psi_ezx_m_re,	double *psi_ezx_m_im					\
);

// PML at y+.
void PML_updateH_py(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffyEx_re,		double *diffyEx_im,						\
	double *diffyEz_re,		double *diffyEz_im,						\
	double *psi_hxy_p_re,	double *psi_hxy_p_im,					\
	double *psi_hzy_p_re,	double *psi_hzy_p_im					\
);
void PML_updateE_py(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
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
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffyEx_re,		double *diffyEx_im,						\
	double *diffyEz_re,		double *diffyEz_im,						\
	double *psi_hxy_p_re,	double *psi_hxy_p_im,					\
	double *psi_hzy_p_re,	double *psi_hzy_p_im					\
);
void PML_updateE_my(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffyHx_re,		double *diffyHx_im,						\
	double *diffyHz_re,		double *diffyHz_im,						\
	double *psi_exy_p_re,	double *psi_exy_p_im,					\
	double *psi_ezy_p_re,	double *psi_ezy_p_im					\
);

// PML at z+.
void PML_updateH_pz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hy_re,			double *Hy_im,							\
	double *diffzEx_re,		double *diffzEx_im,						\
	double *diffzEy_re,		double *diffzEy_im,						\
	double *psi_hxz_p_re,	double *psi_hxz_p_im,					\
	double *psi_hyz_p_re,	double *psi_hyz_p_im					\
);
void PML_updateE_pz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
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
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hy_re,			double *Hy_im,							\
	double *diffzEx_re,		double *diffzEx_im,						\
	double *diffzEy_re,		double *diffzEy_im,						\
	double *psi_hxz_m_re,	double *psi_hxz_m_im,					\
	double *psi_hyz_m_re,	double *psi_hyz_m_im					\
);
void PML_updateE_mz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
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
void PML_updateH_px(											\
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hy_re,			double *Hy_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffxEy_re,		double *diffxEy_im,						\
	double *diffxEz_re,		double *diffxEz_im,						\
	double *psi_hyx_p_re,	double *psi_hyx_p_im,					\
	double *psi_hzx_p_re,	double *psi_hzx_p_im					\
){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	//printf("Here?\n");
	double CHy2, CHz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappax, PMLbx, PMLax,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hy_re, Hy_im,			\
				Hz_re, Hz_im,			\
				diffxEy_re, diffxEy_im, \
				diffxEz_re, diffxEz_im,	\
				psi_hyx_p_re, psi_hyx_p_im, \
				psi_hzx_p_re, psi_hzx_p_im)	\
		private(i, j, k,odd, psiidx, myidx, CHy2, CHz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				odd    = 2*i + 1;
				psiidx = (k  ) + (j  ) * Nz + (i          ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

				CHy2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hy
				psi_hyx_p_re[psiidx] = (PMLbx[odd] * psi_hyx_p_re[psiidx]) + (PMLax[odd] * diffxEz_re[myidx]);
				psi_hyx_p_im[psiidx] = (PMLbx[odd] * psi_hyx_p_im[psiidx]) + (PMLax[odd] * diffxEz_im[myidx]);

				Hy_re[myidx] += CHy2 * (-((1./PMLkappax[odd] - 1.) * diffxEz_re[myidx]) - psi_hyx_p_re[psiidx]);
				Hy_im[myidx] += CHy2 * (-((1./PMLkappax[odd] - 1.) * diffxEz_im[myidx]) - psi_hyx_p_im[psiidx]);
				// Update Hz
				psi_hzx_p_re[psiidx] = (PMLbx[odd] * psi_hzx_p_re[psiidx]) + (PMLax[odd] * diffxEy_re[myidx]);
				psi_hzx_p_im[psiidx] = (PMLbx[odd] * psi_hzx_p_im[psiidx]) + (PMLax[odd] * diffxEy_im[myidx]);

				Hz_re[myidx] += CHz2 * (+((1./PMLkappax[odd] - 1.) * diffxEy_re[myidx]) + psi_hzx_p_re[psiidx]);
				Hz_im[myidx] += CHz2 * (+((1./PMLkappax[odd] - 1.) * diffxEy_im[myidx]) + psi_hzx_p_im[psiidx]);
			}
		}
	}
	return;
};


void PML_updateE_px(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ey_re,			double *Ey_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffxHy_re,		double *diffxHy_im,						\
	double *diffxHz_re,		double *diffxHz_im,						\
	double *psi_eyx_p_re,	double *psi_eyx_p_im,					\
	double *psi_ezx_p_re,	double *psi_ezx_p_im					\
){

	int i,j,k;
	int even;
	int psiidx, myidx;
	
	double CEy2, CEz2;

	#pragma omp parallel for				\
		shared(	npml, myNx, Ny, Nz,			\
				dt,							\
				PMLkappax,	PMLbx,	PMLax,	\
				eps_EHH,	eps_HEE,		\
				econ_EHH,	econ_HEE,		\
				Ey_re,		Ey_im,			\
				Ez_re,		Ez_im,			\
				diffxHy_re, diffxHy_im,		\
				diffxHz_re, diffxHz_im,		\
				psi_eyx_p_re, psi_eyx_p_im,		\
				psi_ezx_p_re, psi_ezx_p_im)		\
		private(i, j, k, even, psiidx, myidx, CEy2, CEz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				even   = 2*i;
				psiidx = (k  ) + (j  ) * Nz + (i          ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i+myNx-npml) * Nz * Ny;

				CEy2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ey.
				psi_eyx_p_re[psiidx] = (PMLbx[even] * psi_eyx_p_re[psiidx]) + (PMLax[even] * diffxHz_re[myidx]);
				psi_eyx_p_im[psiidx] = (PMLbx[even] * psi_eyx_p_im[psiidx]) + (PMLax[even] * diffxHz_im[myidx]);

				Ey_re[myidx] += CEy2 * (-(1./PMLkappax[even] - 1.) * diffxHz_re[myidx] - psi_eyx_p_re[psiidx]);
				Ey_im[myidx] += CEy2 * (-(1./PMLkappax[even] - 1.) * diffxHz_im[myidx] - psi_eyx_p_im[psiidx]);

				// Update Ez.
				psi_ezx_p_re[psiidx] = (PMLbx[even] * psi_ezx_p_re[psiidx]) + (PMLax[even] * diffxHy_re[myidx]);
				psi_ezx_p_im[psiidx] = (PMLbx[even] * psi_ezx_p_im[psiidx]) + (PMLax[even] * diffxHy_im[myidx]);

				Ez_re[myidx] += CEz2 * (+(1./PMLkappax[even] - 1.) * diffxHy_re[myidx] + psi_ezx_p_re[psiidx]);
				Ez_im[myidx] += CEz2 * (+(1./PMLkappax[even] - 1.) * diffxHy_im[myidx] + psi_ezx_p_im[psiidx]);

			}		
		}
	}

	return;
}

/*----------------------------------- PML at x- -----------------------------------*/
void PML_updateH_mx(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hy_re,			double *Hy_im,							\
	double *Hz_re,			double *Hz_im,							\
	double *diffxEy_re,		double *diffxEy_im,						\
	double *diffxEz_re,		double *diffxEz_im,						\
	double *psi_hyx_m_re,	double *psi_hyx_m_im,					\
	double *psi_hzx_m_re,	double *psi_hzx_m_im					\
){

	int i,j,k;
	int even;
	int psiidx, myidx;
	
	double CHy2, CHz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappax, PMLbx, PMLax,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hy_re, Hy_im,			\
				Hz_re, Hz_im,			\
				diffxEy_re, diffxEy_im, \
				diffxEz_re, diffxEz_im,	\
				psi_hyx_m_re, psi_hyx_m_im, \
				psi_hzx_m_re, psi_hzx_m_im)	\
		private(i, j, k, even, psiidx, myidx, CHy2, CHz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){
				
				even   = (2*npml) - (2*i + 2);
				psiidx = (k  ) + (j  ) * Nz + (i          ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;

				CHy2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hy
				psi_hyx_m_re[psiidx] = (PMLbx[even] * psi_hyx_m_re[psiidx]) + (PMLax[even] * diffxEz_re[myidx]);
				psi_hyx_m_im[psiidx] = (PMLbx[even] * psi_hyx_m_im[psiidx]) + (PMLax[even] * diffxEz_im[myidx]);

				Hy_re[myidx] += CHy2 * (-((1./PMLkappax[even] - 1.) * diffxEz_re[myidx]) - psi_hyx_m_re[psiidx]);
				Hy_im[myidx] += CHy2 * (-((1./PMLkappax[even] - 1.) * diffxEz_im[myidx]) - psi_hyx_m_im[psiidx]);
				// Update Hz
				psi_hzx_m_re[psiidx] = (PMLbx[even] * psi_hzx_m_re[psiidx]) + (PMLax[even] * diffxEy_re[myidx]);
				psi_hzx_m_im[psiidx] = (PMLbx[even] * psi_hzx_m_im[psiidx]) + (PMLax[even] * diffxEy_im[myidx]);

				Hz_re[myidx] += CHz2 * (+((1./PMLkappax[even] - 1.) * diffxEy_re[myidx]) + psi_hzx_m_re[psiidx]);
				Hz_im[myidx] += CHz2 * (+((1./PMLkappax[even] - 1.) * diffxEy_im[myidx]) + psi_hzx_m_im[psiidx]);
			}
		}
	}
	return;
};

void PML_updateE_mx(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappax,		double *PMLbx,	double *PMLax,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ey_re,			double *Ey_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffxHy_re,		double *diffxHy_im,						\
	double *diffxHz_re,		double *diffxHz_im,						\
	double *psi_eyx_m_re,	double *psi_eyx_m_im,					\
	double *psi_ezx_m_re,	double *psi_ezx_m_im					\
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEy2, CEz2;

	#pragma omp parallel for				\
		shared(	npml, myNx, Ny, Nz,			\
				dt,							\
				PMLkappax,	PMLbx,	PMLax,	\
				eps_EHH,	eps_HEE,		\
				econ_EHH,	econ_HEE,		\
				Ey_re,		Ey_im,			\
				Ez_re,		Ez_im,			\
				diffxHy_re, diffxHy_im,		\
				diffxHz_re, diffxHz_im,		\
				psi_eyx_m_re, psi_eyx_m_im,		\
				psi_ezx_m_re, psi_ezx_m_im)		\
		private(i, j, k, odd, psiidx, myidx, CEy2, CEz2)
	for(i=0; i < npml; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < Nz; k++){

				odd    = (2*npml) - (2*i+1);
				psiidx = (k  ) + (j  ) * Nz + (i          ) * Nz * Ny;
				myidx  = (k  ) + (j  ) * Nz + (i		  ) * Nz * Ny;

				CEy2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);
				CEz2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ey.
				psi_eyx_m_re[psiidx] = (PMLbx[odd] * psi_eyx_m_re[psiidx]) + (PMLax[odd] * diffxHz_re[myidx]);
				psi_eyx_m_im[psiidx] = (PMLbx[odd] * psi_eyx_m_im[psiidx]) + (PMLax[odd] * diffxHz_im[myidx]);

				Ey_re[myidx] += CEy2 * (-(1./PMLkappax[odd] - 1.) * diffxHz_re[myidx] - psi_eyx_m_re[psiidx]);
				Ey_im[myidx] += CEy2 * (-(1./PMLkappax[odd] - 1.) * diffxHz_im[myidx] - psi_eyx_m_im[psiidx]);

				// Update Ez.
				psi_ezx_m_re[psiidx] = (PMLbx[odd] * psi_ezx_m_re[psiidx]) + (PMLax[odd] * diffxHy_re[myidx]);
				psi_ezx_m_im[psiidx] = (PMLbx[odd] * psi_ezx_m_im[psiidx]) + (PMLax[odd] * diffxHy_im[myidx]);

				Ez_re[myidx] += CEz2 * (+(1./PMLkappax[odd] - 1.) * diffxHy_re[myidx] + psi_ezx_m_re[psiidx]);
				Ez_im[myidx] += CEz2 * (+(1./PMLkappax[odd] - 1.) * diffxHy_im[myidx] + psi_ezx_m_im[psiidx]);

			}		
		}
	}
	return;
};

/*----------------------------------- PML at y+ -----------------------------------*/
void PML_updateH_py(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
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
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappay, PMLby, PMLay,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hx_re, Hx_im,			\
				Hz_re, Hz_im,			\
				diffyEx_re, diffyEx_im, \
				diffyEz_re, diffyEz_im,	\
				psi_hxy_p_re, psi_hxy_p_im, \
				psi_hzy_p_re, psi_hzy_p_im)	\
		private(i, j, k, odd, psiidx, myidx, CHx2, CHz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				
				odd    = 2*j+1;
				psiidx = (k  ) + (j		   ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j+Ny-npml) * Nz + (i  ) * Nz * Ny;
				//if(i==0){printf("%d has %d\n", omp_get_thread_num(), psiidx);};

				CHx2 =	(-2*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hx
				psi_hxy_p_re[psiidx] = (PMLby[odd] * psi_hxy_p_re[psiidx]) + (PMLay[odd] * diffyEz_re[myidx]);
				psi_hxy_p_im[psiidx] = (PMLby[odd] * psi_hxy_p_im[psiidx]) + (PMLay[odd] * diffyEz_im[myidx]);

				Hx_re[myidx] += CHx2 * (+((1./PMLkappay[odd] - 1.) * diffyEz_re[myidx]) + psi_hxy_p_re[psiidx]);
				Hx_im[myidx] += CHx2 * (+((1./PMLkappay[odd] - 1.) * diffyEz_im[myidx]) + psi_hxy_p_im[psiidx]);

				// Update Hz
				psi_hzy_p_re[psiidx] = (PMLby[odd] * psi_hzy_p_re[psiidx]) + (PMLay[odd] * diffyEx_re[myidx]);
				psi_hzy_p_im[psiidx] = (PMLby[odd] * psi_hzy_p_im[psiidx]) + (PMLay[odd] * diffyEx_im[myidx]);

				Hz_re[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_re[myidx]) - psi_hzy_p_re[psiidx]);
				Hz_im[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_im[myidx]) - psi_hzy_p_im[psiidx]);
			}
		}
	}

	return;
};

void PML_updateE_py(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ez_re,			double *Ez_im,							\
	double *diffyHx_re,		double *diffyHx_im,						\
	double *diffyHz_re,		double *diffyHz_im,						\
	double *psi_exy_p_re,	double *psi_exy_p_im,					\
	double *psi_ezy_p_re,	double *psi_ezy_p_im					\
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEx2, CEz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappay, PMLby, PMLay,\
				eps_HEE,	eps_EHH,	\
				econ_HEE,	econ_EHH,	\
				Ex_re, Ex_im,			\
				Ez_re, Ez_im,			\
				diffyHx_re, diffyHx_im, \
				diffyHz_re, diffyHz_im,	\
				psi_exy_p_re, psi_exy_p_im, \
				psi_ezy_p_re, psi_ezy_p_im)	\
		private(i, j, k, odd, psiidx, myidx, CEx2, CEz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				
				odd    = 2*j+1;
				psiidx = (k  ) + (j		   ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j+Ny-npml) * Nz + (i  ) * Nz * Ny;

				CEx2 =	(2*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEz2 =	(2*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ex
				psi_exy_p_re[psiidx] = (PMLby[odd] * psi_exy_p_re[psiidx]) + (PMLay[odd] * diffyHz_re[myidx]);
				psi_exy_p_im[psiidx] = (PMLby[odd] * psi_exy_p_im[psiidx]) + (PMLay[odd] * diffyHz_im[myidx]);

				Ex_re[myidx] += CEx2 * (+((1./PMLkappay[odd] - 1.) * diffyHz_re[myidx]) + psi_exy_p_re[psiidx]);
				Ex_im[myidx] += CEx2 * (+((1./PMLkappay[odd] - 1.) * diffyHz_im[myidx]) + psi_exy_p_im[psiidx]);

				// Update Ez
				psi_ezy_p_re[psiidx] = (PMLby[odd] * psi_ezy_p_re[psiidx]) + (PMLay[odd] * diffyHx_re[myidx]);
				psi_ezy_p_im[psiidx] = (PMLby[odd] * psi_ezy_p_im[psiidx]) + (PMLay[odd] * diffyHx_im[myidx]);

				Ez_re[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_re[myidx]) - psi_ezy_p_re[psiidx]);
				Ez_im[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_im[myidx]) - psi_ezy_p_im[psiidx]);
			}
		}
	}
	return;
};

/*----------------------------------- PML at y- -----------------------------------*/
void PML_updateH_my(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
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
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappay, PMLby, PMLay,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hx_re, Hx_im,			\
				Hz_re, Hz_im,			\
				diffyEx_re, diffyEx_im, \
				diffyEz_re, diffyEz_im,	\
				psi_hxy_p_re, psi_hxy_p_im, \
				psi_hzy_p_re, psi_hzy_p_im)	\
		private(i, j, k, odd, psiidx, myidx, CHx2, CHz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				
				odd    = (2*npml) - (2*j+1);
				psiidx = (k  ) + (j  ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;
				//if(i==0){printf("%d has %d\n", omp_get_thread_num(), psiidx);};

				CHx2 =	(-2*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHz2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hx
				psi_hxy_p_re[psiidx] = (PMLby[odd] * psi_hxy_p_re[psiidx]) + (PMLay[odd] * diffyEz_re[myidx]);
				psi_hxy_p_im[psiidx] = (PMLby[odd] * psi_hxy_p_im[psiidx]) + (PMLay[odd] * diffyEz_im[myidx]);

				Hx_re[myidx] += CHx2 * (+((1./PMLkappay[odd] - 1.) * diffyEz_re[myidx]) + psi_hxy_p_re[psiidx]);
				Hx_im[myidx] += CHx2 * (+((1./PMLkappay[odd] - 1.) * diffyEz_im[myidx]) + psi_hxy_p_im[psiidx]);

				// Update Hz
				psi_hzy_p_re[psiidx] = (PMLby[odd] * psi_hzy_p_re[psiidx]) + (PMLay[odd] * diffyEx_re[myidx]);
				psi_hzy_p_im[psiidx] = (PMLby[odd] * psi_hzy_p_im[psiidx]) + (PMLay[odd] * diffyEx_im[myidx]);

				Hz_re[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_re[myidx]) - psi_hzy_p_re[psiidx]);
				Hz_im[myidx] += CHz2 * (-((1./PMLkappay[odd] - 1.) * diffyEx_im[myidx]) - psi_hzy_p_im[psiidx]);
			}
		}
	}
	return;
};

void PML_updateE_my(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappay,		double *PMLby,	double *PMLay,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			double *Ex_im,							\
	doble *Ez_re,			double *Ez_im,							\
	double *diffyHx_re,		double *diffyHx_im,						\
	double *diffyHz_re,		double *diffyHz_im,						\
	double *psi_exy_p_re,	double *psi_exy_p_im,					\
	double *psi_ezy_p_re,	double *psi_ezy_p_im					\
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEx2, CEz2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappay, PMLby, PMLay,\
				eps_HEE,	eps_EHH,	\
				econ_HEE,	econ_EHH,	\
				Ex_re, Ex_im,			\
				Ez_re, Ez_im,			\
				diffyHx_re, diffyHx_im, \
				diffyHz_re, diffyHz_im,	\
				psi_exy_p_re, psi_exy_p_im, \
				psi_ezy_p_re, psi_ezy_p_im)	\
		private(i, j, k, odd, psiidx, myidx, CEx2, CEz2)
	for(i=0; i < myNx; i++){
		for(j=0; j < npml; j++){
			for(k=0; k < Nz; k++){
				
				odd    = (2*npml) - (2*j+1);
				psiidx = (k  ) + (j  ) * Nz + (i  ) * Nz * npml;
				myidx  = (k  ) + (j  ) * Nz + (i  ) * Nz * Ny;

				CEx2 =	(2*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEz2 =	(2*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ex
				psi_exy_p_re[psiidx] = (PMLby[odd] * psi_exy_p_re[psiidx]) + (PMLay[odd] * diffyHz_re[myidx]);
				psi_exy_p_im[psiidx] = (PMLby[odd] * psi_exy_p_im[psiidx]) + (PMLay[odd] * diffyHz_im[myidx]);

				Ex_re[myidx] += CEx2 * (+((1./PMLkappay[odd] - 1.) * diffyHz_re[myidx]) + psi_exy_p_re[psiidx]);
				Ex_im[myidx] += CEx2 * (+((1./PMLkappay[odd] - 1.) * diffyHz_im[myidx]) + psi_exy_p_im[psiidx]);

				// Update Ez
				psi_ezy_p_re[psiidx] = (PMLby[odd] * psi_ezy_p_re[psiidx]) + (PMLay[odd] * diffyHx_re[myidx]);
				psi_ezy_p_im[psiidx] = (PMLby[odd] * psi_ezy_p_im[psiidx]) + (PMLay[odd] * diffyHx_im[myidx]);

				Ez_re[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_re[myidx]) - psi_ezy_p_re[psiidx]);
				Ez_im[myidx] += CEz2 * (-((1./PMLkappay[odd] - 1.) * diffyHx_im[myidx]) - psi_ezy_p_im[psiidx]);
			}
		}
	}
	return;
};

/*----------------------------------- PML at z+ -----------------------------------*/
void PML_updateH_pz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
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
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappaz, PMLbz, PMLaz,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hx_re, Hx_im,			\
				Hy_re, Hy_im,			\
				diffzEx_re, diffzEx_im, \
				diffzEy_re, diffzEy_im,	\
				psi_hxz_p_re, psi_hxz_p_im, \
				psi_hyz_p_re, psi_hyz_p_im)	\
		private(i, j, k, odd, psiidx, myidx, CHx2, CHy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = 2*k+1;
				psiidx = (k        ) + (j  ) * npml + (i  ) * npml * Ny;
				myidx  = (k+Nz-npml) + (j  ) * Nz   + (i  ) * Nz   * Ny;
				//if(i==0){printf("%d has %d\n", omp_get_thread_num(), myidx);};

				CHx2 =	(-2*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHy2 =	(-2*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);
				// Update Hx
				psi_hxz_p_re[psiidx] = (PMLbz[odd] * psi_hxz_p_re[psiidx]) + (PMLaz[odd] * diffzEy_re[myidx]);
				psi_hxz_p_im[psiidx] = (PMLbz[odd] * psi_hxz_p_im[psiidx]) + (PMLaz[odd] * diffzEy_im[myidx]);

				Hx_re[myidx] += CHx2 * (-((1./PMLkappaz[odd] - 1.) * diffzEy_re[myidx]) - psi_hxz_p_re[psiidx]);
				Hx_im[myidx] += CHx2 * (-((1./PMLkappaz[odd] - 1.) * diffzEy_im[myidx]) - psi_hxz_p_im[psiidx]);

				// Update Hy
				psi_hyz_p_re[psiidx] = (PMLbz[odd] * psi_hyz_p_re[psiidx]) + (PMLaz[odd] * diffzEx_re[myidx]);
				psi_hyz_p_im[psiidx] = (PMLbz[odd] * psi_hyz_p_im[psiidx]) + (PMLaz[odd] * diffzEx_im[myidx]);

				Hy_re[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_re[myidx]) + psi_hyz_p_re[psiidx]);
				Hy_im[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_im[myidx]) + psi_hyz_p_im[psiidx]);
			}
		}
	}
	return;
};

void PML_updateE_pz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ey_re,			double *Ey_im,							\
	double *diffzHx_re,		double *diffzHx_im,						\
	double *diffzHy_re,		double *diffzHy_im,						\
	double *psi_exz_p_re,	double *psi_exz_p_im,					\
	double *psi_eyz_p_re,	double *psi_eyz_p_im					\
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEx2, CEy2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappaz, PMLbz, PMLaz,\
				eps_HEE,	eps_EHH,	\
				econ_HEE,	econ_EHH,	\
				Ex_re, Ex_im,			\
				Ey_re, Ey_im,			\
				diffzHx_re, diffzHx_im, \
				diffzHy_re, diffzHy_im,	\
				psi_exz_p_re, psi_exz_p_im, \
				psi_eyz_p_re, psi_eyz_p_im)	\
		private(i, j, k, odd, psiidx, myidx, CEx2, CEy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = 2*k+1;
				psiidx = (k        ) + (j  ) * npml + (i  ) * npml * Ny;
				myidx  = (k+Nz-npml) + (j  ) * Nz   + (i  ) * Nz   * Ny;

				CEx2 =	(2*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEy2 =	(2*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ex
				psi_exz_p_re[psiidx] = (PMLbz[odd] * psi_exz_p_re[psiidx]) + (PMLaz[odd] * diffzHy_re[myidx]);
				psi_exz_p_im[psiidx] = (PMLbz[odd] * psi_exz_p_im[psiidx]) + (PMLaz[odd] * diffzHy_im[myidx]);

				Ex_re[myidx] += CEx2 * (-((1./PMLkappaz[odd] - 1.) * diffzHy_re[myidx]) - psi_exz_p_re[psiidx]);
				Ex_im[myidx] += CEx2 * (-((1./PMLkappaz[odd] - 1.) * diffzHy_im[myidx]) - psi_exz_p_im[psiidx]);

				// Update Ey
				psi_eyz_p_re[psiidx] = (PMLbz[odd] * psi_eyz_p_re[psiidx]) + (PMLaz[odd] * diffzHx_re[myidx]);
				psi_eyz_p_im[psiidx] = (PMLbz[odd] * psi_eyz_p_im[psiidx]) + (PMLaz[odd] * diffzHx_im[myidx]);

				Ey_re[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_re[myidx]) + psi_eyz_p_re[psiidx]);
				Ey_im[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_im[myidx]) + psi_eyz_p_im[psiidx]);
			}
		}
	}
	return;
};

/*----------------------------------- PML at z- -----------------------------------*/
void PML_updateH_mz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *mu_HEE,			double *mu_EHH,							\
	double *mcon_HEE,		double *mcon_EHH,						\
	double *Hx_re,			double *Hx_im,							\
	double *Hy_re,			double *Hy_im,							\
	double *diffzEx_re,		double *diffzEx_im,						\
	double *diffzEy_re,		double *diffzEy_im,						\
	double *psi_hxz_m_re,	double *psi_hxz_m_im,					\
	double *psi_hyz_m_re,	double *psi_hyz_m_im					\
){

	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CHx2, CHy2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappaz, PMLbz, PMLaz,\
				mu_HEE,		mu_EHH,		\
				mcon_HEE,	mcon_EHH,	\
				Hx_re, Hx_im,			\
				Hy_re, Hy_im,			\
				diffzEx_re, diffzEx_im, \
				diffzEy_re, diffzEy_im,	\
				psi_hxz_m_re, psi_hxz_m_im, \
				psi_hyz_m_re, psi_hyz_m_im)	\
		private(i, j, k, odd, psiidx, myidx, CHx2, CHy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = (2*npml) - (2*k+1);
				psiidx = (k        ) + (j  ) * npml + (i  ) * npml * Ny;
				myidx  = (k		   ) + (j  ) * Nz   + (i  ) * Nz   * Ny;

				CHx2 =	(-2.*dt) / (2.*mu_HEE[myidx] + mcon_HEE[myidx]*dt);
				CHy2 =	(-2.*dt) / (2.*mu_EHH[myidx] + mcon_EHH[myidx]*dt);

				// Update Hx
				psi_hxz_m_re[psiidx] = (PMLbz[odd] * psi_hxz_m_re[psiidx]) + (PMLaz[odd] * diffzEy_re[myidx]);
				psi_hxz_m_im[psiidx] = (PMLbz[odd] * psi_hxz_m_im[psiidx]) + (PMLaz[odd] * diffzEy_im[myidx]);

				Hx_re[myidx] += CHx2 * (-((1./PMLkappaz[odd] - 1.) * diffzEy_re[myidx]) - psi_hxz_m_re[psiidx]);
				Hx_im[myidx] += CHx2 * (-((1./PMLkappaz[odd] - 1.) * diffzEy_im[myidx]) - psi_hxz_m_im[psiidx]);

				// Update Hy
				psi_hyz_m_re[psiidx] = (PMLbz[odd] * psi_hyz_m_re[psiidx]) + (PMLaz[odd] * diffzEx_re[myidx]);
				psi_hyz_m_im[psiidx] = (PMLbz[odd] * psi_hyz_m_im[psiidx]) + (PMLaz[odd] * diffzEx_im[myidx]);

				Hy_re[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_re[myidx]) + psi_hyz_m_re[psiidx]);
				Hy_im[myidx] += CHy2 * (+((1./PMLkappaz[odd] - 1.) * diffzEx_im[myidx]) + psi_hyz_m_im[psiidx]);
			}
		}
	}
	return;
};

void PML_updateE_mz(
	int myNx,				int Ny,			int Nz,		int npml,	\
	double  dt,														\
	double *PMLkappaz,		double *PMLbz,	double *PMLaz,			\
	double *eps_HEE,		double *eps_EHH,						\
	double *econ_HEE,		double *econ_EHH,						\
	double *Ex_re,			double *Ex_im,							\
	double *Ey_re,			double *Ey_im,							\
	double *diffzHx_re,		double *diffzHx_im,						\
	double *diffzHy_re,		double *diffzHy_im,						\
	double *psi_exz_m_re,	double *psi_exz_m_im,					\
	double *psi_eyz_m_re,	double *psi_eyz_m_im					\
){
	int i,j,k;
	int odd;
	int psiidx, myidx;
	
	double CEx2, CEy2;
	#pragma omp parallel for			\
		shared(	npml, myNx, Ny, Nz,		\
				dt,			 			\
				PMLkappaz, PMLbz, PMLaz,\
				eps_HEE,	eps_EHH,	\
				econ_HEE,	econ_EHH,	\
				Ex_re, Ex_im,			\
				Ey_re, Ey_im,			\
				diffzHx_re, diffzHx_im, \
				diffzHy_re, diffzHy_im,	\
				psi_exz_m_re, psi_exz_m_im, \
				psi_eyz_m_re, psi_eyz_m_im)	\
		private(i, j, k, odd, psiidx, myidx, CEx2, CEy2)
	for(i=0; i < myNx; i++){
		for(j=0; j < Ny; j++){
			for(k=0; k < npml; k++){
				
				odd    = (2*npml) - (2*k+1);
				psiidx = (k        ) + (j  ) * npml + (i  ) * npml * Ny;
				myidx  = (k		   ) + (j  ) * Nz   + (i  ) * Nz   * Ny;
				//if(i==0){printf("%d has %d\n", omp_get_thread_num(), psiidx);};

				CEx2 =	(2.*dt) / (2.*eps_EHH[myidx] + econ_EHH[myidx]*dt);
				CEy2 =	(2.*dt) / (2.*eps_HEE[myidx] + econ_HEE[myidx]*dt);

				// Update Ex
				psi_exz_m_re[psiidx] = (PMLbz[odd] * psi_exz_m_re[psiidx]) + (PMLaz[odd] * diffzHy_re[myidx]);
				psi_exz_m_im[psiidx] = (PMLbz[odd] * psi_exz_m_im[psiidx]) + (PMLaz[odd] * diffzHy_im[myidx]);

				Ex_re[myidx] += CEx2 * (-((1./PMLkappaz[odd] - 1.) * diffzHy_re[myidx]) - psi_exz_m_re[psiidx]);
				Ex_im[myidx] += CEx2 * (-((1./PMLkappaz[odd] - 1.) * diffzHy_im[myidx]) - psi_exz_m_im[psiidx]);

				// Update Ey
				psi_eyz_m_re[psiidx] = (PMLbz[odd] * psi_eyz_m_re[psiidx]) + (PMLaz[odd] * diffzHx_re[myidx]);
				psi_eyz_m_im[psiidx] = (PMLbz[odd] * psi_eyz_m_im[psiidx]) + (PMLaz[odd] * diffzHx_im[myidx]);

				Ey_re[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_re[myidx]) + psi_eyz_m_re[psiidx]);
				Ey_im[myidx] += CEy2 * (+((1./PMLkappaz[odd] - 1.) * diffzHx_im[myidx]) + psi_eyz_m_im[psiidx]);
			}
		}
	}

	return;
};
