/*
Author  : Myung-su Seok
Purpose : Nvidia CUDA kernels to update the core of FDTD
Target  : GPU
Created : 2012.05.02
Modified: 2013.06.07
*/

#include <omp.h>

__ACTIVATE_FLOAT64__

__KERNEL__ void update_2d( int fd_type, int ax, int nx, int ny \
                         , __FLOAT__ sinkl, __FLOAT__ coskl \
                         , __GLOBAL__ __FLOAT__* fd_real, __GLOBAL__ __FLOAT__* fd_imag)
{
    int idx, idx_tgt, idx_src, idx_swp, idx_max;

    if(ax==0) { idx_max = ny; }
    if(ax==1) { idx_max = nx; }

    #pragma omp parallel for \
    shared( fd_type, ax, nx, ny \
          , sinkl, coskl \
          , fd_real, fd_imag \
          , idx_max ) \
    private( idx, idx_tgt, idx_src, idx_swp ) \
    schedule(guided)
    for(idx=0; idx<idx_max; idx++)
    {
        if(ax==0)
        {
            idx_tgt = (nx-1)*ny + idx;
            idx_src = (   0)*ny + idx;
        }
        if(ax==1)
        {
            idx_tgt = idx*ny + ny-1;
            idx_src = idx*ny +    0;
        }
        if(fd_type==0)
        {
            idx_swp = idx_src;
            idx_src = idx_tgt;
            idx_tgt = idx_swp;
        }
        fd_real[idx_tgt] = coskl*fd_real[idx_src] - sinkl*fd_imag[idx_src];
        fd_imag[idx_tgt] = sinkl*fd_real[idx_src] + coskl*fd_imag[idx_src];
    }
}

__KERNEL__ void update_3d( int fd_type, int ax, int nx, int ny, int nz \
                         , __FLOAT__ sinkl, __FLOAT__ coskl \
                         , __GLOBAL__ __FLOAT__* fd_real, __GLOBAL__ __FLOAT__* fd_imag)
{
    int idx;
    int idx_tgt, idx_src, idx_swp, idx_max;
    int i, k;

    if(ax==0) { idx_max = ny*nz; }
    if(ax==1) { idx_max = nz*nx; }
    if(ax==2) { idx_max = nx*ny; }

    #pragma omp parallel for \
    shared( fd_type, ax, nx, ny, nz \
          , sinkl, coskl \
          , fd_real, fd_imag \
          , idx_max ) \
    private( idx, idx_tgt, idx_src, idx_swp, i, k ) \
    schedule(guided)
    for(idx=0; idx<idx_max; idx++)
    {
        if(ax==0)
        {
            idx_tgt = (nx-1)*ny*nz + idx;
            idx_src = (   0)*ny*nz + idx;
        }
        if(ax==1)
        {
            i = idx  /nz;
            k = idx-i*nz;
            idx_tgt = i*ny*nz + (ny-1)*nz + k;
            idx_src = i*ny*nz + (   0)*nz + k;
        }
        if(ax==2)
        {
            idx_tgt = idx*nz + (nz-1);
            idx_src = idx*nz + (   0);
        }
        if(fd_type==0)
        {
            idx_swp = idx_src;
            idx_src = idx_tgt;
            idx_tgt = idx_swp;
        }
        fd_real[idx_tgt] = coskl*fd_real[idx_src] - sinkl*fd_imag[idx_src];
        fd_imag[idx_tgt] = sinkl*fd_real[idx_src] + coskl*fd_imag[idx_src];
    }
}
