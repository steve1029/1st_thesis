/*
Author  : Myung-su Seok
Purpose : openCL kernels to update the PML region of FDTD
Target  : GPU
Created : 2012.03.29
*/

#include <omp.h>

__ACTIVATE_FLOAT64__

__KERNEL__ void update_e_2d( int ax, int dr, int sg, int nx, int ny, int np \
			   , __GLOBAL__ __FLOAT__* ef, __GLOBAL__ __FLOAT__* hf \
		    	   , __GLOBAL__ __FLOAT__* ce \
			   , __GLOBAL__ __FLOAT__* ds \
			   , __GLOBAL__ __FLOAT__* pcb, __GLOBAL__ __FLOAT__* pca \
			   , __GLOBAL__ __FLOAT__* psi \
			   )
{
    int idx;
    int i, j, im, jm, idx0, idx1, idx2, ids, idpc;

    if(ax==0) { im = np, jm = ny; }
    if(ax==1) { im = nx, jm = np; }

    #pragma omp parallel for \
    shared( ax, dr, sg, nx, ny, np, im, jm \
          , psi, pca, pcb, ef, hf \
	  , ce, ds \
	  ) \
    private( i, j \
           , idx, idx0, idx1, idx2 \
           , idpc, ids \
	   ) \
    schedule(guided)
    for(idx=0;idx<im*jm;idx++)
    {
	if(ax==0)
	{
	    i    = idx    /jm;
	    j    = idx - i*jm;
	    ids  =  i+1+dr*(nx-np-1 );
	    idpc =  i                ;
	    idx0 = (i+1+dr*(nx-np-1))*ny + j;
	    idx1 = (i  +dr*(nx-np-1))*ny + j;
	    idx2 = idx0;
	    }
	if(ax==1)
	{
	    i    = idx    /jm;
	    j    = idx - i*jm;
	    ids  =         j+1+dr*(ny-np-1 );
	    idpc =         j                ;
	    idx0 = i*ny + (j+1+dr*(ny-np-1));
	    idx1 = i*ny + (j  +dr*(ny-np-1));
	    idx2 = idx0;
	}
        psi[idx]  = pcb[idpc]*psi[idx] + pca[idpc]*(hf[idx2]-hf[idx1]);
        ef[idx0] += sg*ce[idx0]*ds[ids]*psi[idx];
    }
}

__KERNEL__ void update_h_2d( int ax, int dr, int sg, int nx, int ny, int np \
                           , __GLOBAL__ __FLOAT__* ef, __GLOBAL__ __FLOAT__* hf \
			   , __GLOBAL__ __FLOAT__* ch \
			   , __GLOBAL__ __FLOAT__* ds \
                           , __GLOBAL__ __FLOAT__* pcb, __GLOBAL__ __FLOAT__* pca \
			   , __GLOBAL__ __FLOAT__* psi \
			   )
{
    int idx;
    int i, j, im, jm, idx0, idx1, idx2, ids, idpc;

    if(ax==0) { im = np, jm = ny; }
    if(ax==1) { im = nx, jm = np; }

    #pragma omp parallel for \
    shared( ax, dr, sg, nx, ny, np, im, jm \
          , psi, pca, pcb, ef, hf \
          , ch, ds \
	  ) \
    private( i, j \
           , idx, idx0, idx1, idx2 \
           , idpc, ids \
	   ) \
    schedule(guided)
    for(idx=0;idx<im*jm;idx++)
    {
	if(ax==0)
	{
	    i    = idx    /jm;
	    j    = idx - i*jm;
	    ids  =  i  +dr*(nx-np-1);
	    idpc =  i               ;
	    idx0 = (i  +dr*(nx-np-1))*ny + j;
	    idx1 = idx0;
	    idx2 = (i+1+dr*(nx-np-1))*ny + j;
	}
	if(ax==1)
	{
	    i    = idx    /jm;
	    j    = idx - i*jm;
	    ids  =         j  +dr*(ny-np-1 );
	    idpc =         j                ;
	    idx0 = i*ny + (j  +dr*(ny-np-1));
	    idx1 = idx0;
	    idx2 = i*ny + (j+1+dr*(ny-np-1));
	}
        psi[idx]  = pcb[idpc]*psi[idx] + pca[idpc]*(ef[idx2]-ef[idx1]);
        hf[idx0] += sg*ch[idx0]*ds[ids]*psi[idx];
    }
}

__KERNEL__ void update_e_3d( int ax, int dr, int nx, int ny, int nz, int np \
			 , __GLOBAL__ __FLOAT__* ef1, __GLOBAL__ __FLOAT__* ef2 \
			 , __GLOBAL__ __FLOAT__* hf1, __GLOBAL__ __FLOAT__* hf2 \
			 , __GLOBAL__ __FLOAT__* ce1, __GLOBAL__ __FLOAT__* ce2 \
			 , __GLOBAL__ __FLOAT__* ds \
			 , __GLOBAL__ __FLOAT__* pcb, __GLOBAL__ __FLOAT__* pca \
			 , __GLOBAL__ __FLOAT__* psi1, __GLOBAL__ __FLOAT__* psi2 \
			 )
{
    int idx;
    int i, j, k, im, jm, km, idx0, idx1, idx2, ids, idpc;

    if(ax==0) { im = np, jm = ny, km = nz; }
    if(ax==1) { im = nx, jm = np, km = nz; }
    if(ax==2) { im = nx, jm = ny, km = np; }

    #pragma omp parallel for \
    shared( ax, dr, nx, ny, nz, np, im, jm, km \
          , psi1, psi2, pca, pcb \
	  , ef1, ef2, hf1, hf2 \
	  , ce1, ce2, ds \
	  ) \
    private( i, j, k \
           , idx, idx0, idx1, idx2 \
           , idpc, ids \
	   ) \
    schedule(guided)
    for(idx=0;idx<im*jm*km;idx++)
    {
	if(ax==0)
	{
	    i    =  idx   /(jm*km)      ;
	    j    = (idx - i*jm*km)   /km;
	    k    =  idx - i*jm*km - j*km;
	    ids  =  i+1+dr*(nx-np-1);
	    idpc =  i               ;
	    idx0 = (i+1+dr*(nx-np-1))*ny*nz + j*nz + k;
	    idx1 = (i  +dr*(nx-np-1))*ny*nz + j*nz + k;
	    idx2 = idx0;
	}
	if(ax==1)
	{
	    i    =  idx   /(jm*km)      ;
	    j    = (idx - i*jm*km)   /km;
	    k    =  idx - i*jm*km - j*km;
	    ids  =            j+1+dr*(ny-np-1);
	    idpc =            j               ;
	    idx0 = i*ny*nz + (j+1+dr*(ny-np-1))*nz + k;
	    idx1 = i*ny*nz + (j  +dr*(ny-np-1))*nz + k;
	    idx2 = idx0;
	}
	if(ax==2)
	{
	    i    =  idx   /(jm*km)      ;
	    j    = (idx - i*jm*km)   /km;
	    k    =  idx - i*jm*km - j*km;
	    ids  =                   k+1+dr*(nz-np-1);
	    idpc =                   k               ;
	    idx0 = i*ny*nz + j*nz + (k+1+dr*(nz-np-1));
	    idx1 = i*ny*nz + j*nz + (k  +dr*(nz-np-1));
	    idx2 = idx0;
	}
        psi1[idx] =  pcb[idpc]*psi1[idx] + pca[idpc]*(hf1[idx2]-hf1[idx1]);
        ef1[idx0] += ce1[idx0]*ds[ids]*psi1[idx];
        psi2[idx] =  pcb[idpc]*psi2[idx] + pca[idpc]*(hf2[idx2]-hf2[idx1]);
        ef2[idx0] -= ce2[idx0]*ds[ids]*psi2[idx];
    }
}

__KERNEL__ void update_h_3d( int ax, int dr, int nx, int ny, int nz, int np \
			   , __GLOBAL__ __FLOAT__* ef1, __GLOBAL__ __FLOAT__* ef2 \
			   , __GLOBAL__ __FLOAT__* hf1, __GLOBAL__ __FLOAT__* hf2 \
			   , __GLOBAL__ __FLOAT__* ch1, __GLOBAL__ __FLOAT__* ch2 \
			   , __GLOBAL__ __FLOAT__* ds \
			   , __GLOBAL__ __FLOAT__* pcb, __GLOBAL__ __FLOAT__* pca \
			   , __GLOBAL__ __FLOAT__* psi1, __GLOBAL__ __FLOAT__* psi2 \
			   )
{
    int idx;
    int i, j, k, im, jm, km, idx0, idx1, idx2, ids, idpc;

    if(ax==0) { im = np, jm = ny, km = nz; }
    if(ax==1) { im = nx, jm = np, km = nz; }
    if(ax==2) { im = nx, jm = ny, km = np; }

    #pragma omp parallel for \
    shared( ax, dr, nx, ny, nz, np, im, jm, km \
          , psi1, psi2, pca, pcb \
	  , ef1, ef2, hf1, hf2 \
	  , ch1, ch2, ds \
	  ) \
    private( i, j, k \
           , idx, idx0, idx1, idx2 \
           , idpc, ids \
	   ) \
    schedule(guided)
    for(idx=0;idx<im*jm*km;idx++)
    {
	if(ax==0)
	{
	    i    =  idx   /(jm*km)      ;
	    j    = (idx - i*jm*km)   /km;
	    k    =  idx - i*jm*km - j*km;
	    ids  =  i  +dr*(nx-np-1);
	    idpc =  i               ;
	    idx0 = (i  +dr*(nx-np-1))*ny*nz + j*nz + k;
	    idx1 = idx0;
	    idx2 = (i+1+dr*(nx-np-1))*ny*nz + j*nz + k;
	}
	if(ax==1)
	{
	    i    =  idx   /(jm*km)      ;
	    j    = (idx - i*jm*km)   /km;
	    k    =  idx - i*jm*km - j*km;
	    ids  =            j  +dr*(ny-np-1);
	    idpc =            j               ;
	    idx0 = i*ny*nz + (j  +dr*(ny-np-1))*nz + k;
	    idx1 = idx0;
	    idx2 = i*ny*nz + (j+1+dr*(ny-np-1))*nz + k;
	}
	if(ax==2)
	{
	    i    =  idx   /(jm*km)      ;
	    j    = (idx - i*jm*km)   /km;
	    k    =  idx - i*jm*km - j*km;
	    ids  =                   k  +dr*(nz-np-1);
	    idpc =                   k               ;
	    idx0 = i*ny*nz + j*nz + (k  +dr*(nz-np-1));
	    idx1 = idx0;
	    idx2 = i*ny*nz + j*nz + (k+1+dr*(nz-np-1));
	}
        psi1[idx] =  pcb[idpc]*psi1[idx] + pca[idpc]*(ef1[idx2]-ef1[idx1]);
        hf1[idx0] += ch1[idx0]*ds[ids]*psi1[idx];
        psi2[idx] =  pcb[idpc]*psi2[idx] + pca[idpc]*(ef2[idx2]-ef2[idx1]);
        hf2[idx0] -= ch2[idx0]*ds[ids]*psi2[idx];
    }
}
