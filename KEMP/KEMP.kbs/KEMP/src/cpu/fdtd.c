/*
Author  : Myung-su Seok
Purpose : Intel CPU kernels to update the core of FDTD
Target  : Intel CPU
Created : 2013.07.09
Modified: 2013.07.09
*/
#include <omp.h>

//__ACTIVATE_FLOAT64__

__KERNEL__ void update_e_2dte( int nx, int ny \
                              , __GLOBAL__ __FLOAT__* ex, __GLOBAL__ __FLOAT__* ey \
                              , __GLOBAL__ __FLOAT__* hz \
                              , __GLOBAL__ __FLOAT__* ce1x, __GLOBAL__ __FLOAT__* ce1y \
                              , __GLOBAL__ __FLOAT__* ce2x, __GLOBAL__ __FLOAT__* ce2y \
                              , __GLOBAL__ __FLOAT__* rdx_e, __GLOBAL__ __FLOAT__* rdy_e \
                              )
{
    int idx, i, j;
    #pragma omp parallel for \
    shared( nx, ny \
          , ex, ey, hz \
          , ce1x, ce1y \
          , ce2x, ce2y \
          , rdx_e, rdy_e \
          ) \
    private(idx, i, j) \
    schedule(guided)
    for(idx=0; idx<nx*ny; idx++)
    {
        i   = idx /   ny      ;
        j   = idx - i*ny      ;
        if(i>0) { ey[idx] = ce1y[idx]*ey[idx] - ce2y[idx]*rdx_e[i]*(hz[idx] - hz[idx-ny]); }
        if(j>0) { ex[idx] = ce1x[idx]*ex[idx] + ce2x[idx]*rdy_e[j]*(hz[idx] - hz[idx- 1]); }
    }
}

__KERNEL__ void update_h_2dte( int nx, int ny \
                             , __GLOBAL__ __FLOAT__* ex, __GLOBAL__ __FLOAT__* ey \
                             , __GLOBAL__ __FLOAT__* hz \
                             , __GLOBAL__ __FLOAT__* ch1z, __GLOBAL__ __FLOAT__* ch2z \
                             , __GLOBAL__ __FLOAT__* rdx_h, __GLOBAL__ __FLOAT__* rdy_h \
                             )
{
    int idx, i, j;
    #pragma omp parallel for \
    shared( nx, ny \
          , ex, ey, hz \
          , ch1z \
          , ch2z \
          , rdx_h, rdy_h \
          ) \
    private(idx, i, j) \
    schedule(guided)
    for(idx=0; idx<nx*ny; idx++)
    {
        i   = idx /   ny      ;
        j   = idx - i*ny      ;
        if(i<nx-1 && j<ny-1) { hz[idx] = ch1z[idx]*hz[idx] - ch2z[idx]*(rdx_h[i]*(ey[idx+ny] - ey[idx]) - rdy_h[j]*(ex[idx+ 1] - ex[idx])); }
    }
}

__KERNEL__ void update_e_2dtm( int nx, int ny \
                             , __GLOBAL__ __FLOAT__* ez \
                             , __GLOBAL__ __FLOAT__* hx, __GLOBAL__ __FLOAT__* hy \
                             , __GLOBAL__ __FLOAT__* ce1z \
                             , __GLOBAL__ __FLOAT__* ce2z \
                             , __GLOBAL__ __FLOAT__* rdx_e, __GLOBAL__ __FLOAT__* rdy_e \
                             )
{
    int idx, i, j;

    #pragma omp parallel for \
    shared( nx, ny \
          , ez, hx, hy \
          , ce1z \
          , ce2z \
          , rdx_e, rdy_e \
          ) \
    private(idx, i, j) \
    schedule(guided)
    for(idx=0; idx<nx*ny; idx++)
    {
        i   = idx /   ny      ;
        j   = idx - i*ny      ;
        if(i>0 && j>0) { ez[idx] = ce1z[idx]*ez[idx] + ce2z[idx]*(rdx_e[i]*(hy[idx] - hy[idx-ny]) - rdy_e[j]*(hx[idx] - hx[idx- 1])); }
    }
}

__KERNEL__ void update_h_2dtm( int nx, int ny \
                             , __GLOBAL__ __FLOAT__* ez \
                             , __GLOBAL__ __FLOAT__* hx, __GLOBAL__ __FLOAT__* hy \
                             , __GLOBAL__ __FLOAT__* ch1x, __GLOBAL__ __FLOAT__* ch1y \
                             , __GLOBAL__ __FLOAT__* ch2x, __GLOBAL__ __FLOAT__* ch2y \
                             , __GLOBAL__ __FLOAT__* rdx_h, __GLOBAL__ __FLOAT__* rdy_h \
                             )
{
    int idx, i, j;

    #pragma omp parallel for \
    shared( nx, ny \
          , ez, hx, hy \
          , ch1x, ch1y \
          , ch2x, ch2y \
          , rdx_h, rdy_h \
          ) \
    private(idx, i, j) \
    schedule(guided)
    for(idx=0;idx<nx*ny;idx++)
    {
        i   = idx /   ny      ;
        j   = idx - i*ny      ;
        if(i<nx-1) { hy[idx] = ch1y[idx]*hy[idx] + ch2y[idx]*rdx_h[i]*(ez[idx+ny] - ez[idx]); }
        if(j<ny-1) { hx[idx] = ch1x[idx]*hx[idx] - ch2x[idx]*rdy_h[j]*(ez[idx+ 1] - ez[idx]); }
    }
}

__KERNEL__ void update_e_3d( int nx, int ny, int nz \
                           , __GLOBAL__ __FLOAT__* ex, __GLOBAL__ __FLOAT__* ey, __GLOBAL__ __FLOAT__* ez \
                           , __GLOBAL__ __FLOAT__* hx, __GLOBAL__ __FLOAT__* hy, __GLOBAL__ __FLOAT__* hz \
                           , __GLOBAL__ __FLOAT__* ce1x, __GLOBAL__ __FLOAT__* ce1y, __GLOBAL__ __FLOAT__* ce1z \
                           , __GLOBAL__ __FLOAT__* ce2x, __GLOBAL__ __FLOAT__* ce2y, __GLOBAL__ __FLOAT__* ce2z \
                           , __GLOBAL__ __FLOAT__* rdx_e, __GLOBAL__ __FLOAT__* rdy_e, __GLOBAL__ __FLOAT__* rdz_e
                           )
{
    int idx, i, j, k;
    #pragma omp parallel for \
    shared( nx, ny, nz \
          , ex, ey, ez \
          , hx, hy, hz \
          , ce1x, ce1y, ce1z \
          , ce2x, ce2y, ce2z \
          , rdx_e, rdy_e, rdz_e \
          ) \
    private(idx, i, j, k) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i   =  idx /  (ny*nz)      ;
        j   = (idx - i*ny*nz)/   nz;
        k   =  idx - i*ny*nz - j*nz;
        if(i>0 && j>0) { ez[idx] = ce1z[idx]*ez[idx] + ce2z[idx]*(rdx_e[i]*(hy[idx] - hy[idx-ny*nz]) - rdy_e[j]*(hx[idx] - hx[idx-   nz])); }
        if(j>0 && k>0) { ex[idx] = ce1x[idx]*ex[idx] + ce2x[idx]*(rdy_e[j]*(hz[idx] - hz[idx-   nz]) - rdz_e[k]*(hy[idx] - hy[idx-    1])); }
        if(k>0 && i>0) { ey[idx] = ce1y[idx]*ey[idx] + ce2y[idx]*(rdz_e[k]*(hx[idx] - hx[idx-    1]) - rdx_e[i]*(hz[idx] - hz[idx-ny*nz])); }
    }
}

__KERNEL__ void update_h_3d( int nx, int ny, int nz \
                           , __GLOBAL__ __FLOAT__* ex, __GLOBAL__ __FLOAT__* ey, __GLOBAL__ __FLOAT__* ez \
                           , __GLOBAL__ __FLOAT__* hx, __GLOBAL__ __FLOAT__* hy, __GLOBAL__ __FLOAT__* hz \
                           , __GLOBAL__ __FLOAT__* ch1x, __GLOBAL__ __FLOAT__* ch1y, __GLOBAL__ __FLOAT__* ch1z \
                           , __GLOBAL__ __FLOAT__* ch2x, __GLOBAL__ __FLOAT__* ch2y, __GLOBAL__ __FLOAT__* ch2z \
                           , __GLOBAL__ __FLOAT__* rdx_h, __GLOBAL__ __FLOAT__* rdy_h, __GLOBAL__ __FLOAT__* rdz_h \
                           )
{
    int idx, i, j, k;

    #pragma omp parallel for \
    shared( nx, ny, nz \
          , ex, ey, ez \
          , hx, hy, hz \
          , ch1x, ch1y, ch1z \
          , ch2x, ch2y, ch2z \
          , rdx_h, rdy_h, rdz_h \
          ) \
    private(idx, i, j, k) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i   =  idx /  (ny*nz)      ;
        j   = (idx - i*ny*nz)/   nz;
        k   =  idx - i*ny*nz - j*nz;
        if(i<nx-1 && j<ny-1) { hz[idx] = ch1z[idx]*hz[idx] - ch2z[idx]*(rdx_h[i]*(ey[idx+ny*nz] - ey[idx]) - rdy_h[j]*(ex[idx+   nz] - ex[idx])); }
        if(j<ny-1 && k<nz-1) { hx[idx] = ch1x[idx]*hx[idx] - ch2x[idx]*(rdy_h[j]*(ez[idx+   nz] - ez[idx]) - rdz_h[k]*(ey[idx+    1] - ey[idx])); }
        if(k<nz-1 && i<nx-1) { hy[idx] = ch1y[idx]*hy[idx] - ch2y[idx]*(rdz_h[k]*(ex[idx+    1] - ex[idx]) - rdx_h[i]*(ez[idx+ny*nz] - ez[idx])); }
    }
}
