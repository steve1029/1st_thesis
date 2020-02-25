/*
Author  : Myung-su Seok
Purpose : openCL kernels to update the core of FDTD
Target  : GPU
Created : 2012.03.29
*/

#include <omp.h>

__ACTIVATE_FLOAT64__

__KERNEL__ void update_dpole_2dte( int snx, int sny \
                                 , int  nx, int  ny \
                                 , int  px, int  py \
                                 , __GLOBAL__ int* mx, __GLOBAL__ int* my \
                                 , __GLOBAL__ __FLOAT__* ex  , __GLOBAL__ __FLOAT__* ey   \
                                 , __GLOBAL__ __FLOAT__* fx_r, __GLOBAL__ __FLOAT__* fy_r \
                                 , __GLOBAL__ const __FLOAT__* cf_r \
                                 , __GLOBAL__ const __FLOAT__* ce_r )
{
    int idx, i, j, idxe, mkx, mky;
    __FLOAT__ bx_r, by_r;

    #pragma omp parallel for \
    shared( snx, sny, nx, ny, px, py \
          , mx, my, ex, ey \
          , fx_r, fy_r, cf_r, ce_r) \
    private( idx, i, j, idxe \
           , mkx, mky, bx_r, by_r) \
    schedule(guided)
    for(idx=0; idx<snx*sny; idx++)
    {
        i    = idx /   sny     ;
        j    = idx - i*sny     ;
        idxe = (i+px)*ny+(j+py);
        mkx  = mx[idx];
        mky  = my[idx];
        bx_r = cf_r[mkx]*fx_r[idx] + ce_r[mkx]*ex[idxe];
        by_r = cf_r[mky]*fy_r[idx] + ce_r[mky]*ey[idxe];
        fx_r[idx] = bx_r;
        fy_r[idx] = by_r;
    }
}

__KERNEL__ void update_dpole_2dtm( int snx, int sny \
                                 , int  nx, int  ny \
                                 , int  px, int  py \
                                 , __GLOBAL__ int* mz \
                                 , __GLOBAL__ __FLOAT__* ez   \
                                 , __GLOBAL__ __FLOAT__* fz_r \
                                 , __GLOBAL__ const __FLOAT__* cf_r \
                                 , __GLOBAL__ const __FLOAT__* ce_r )
{
    int idx, i, j, idxe, mkz;
    __FLOAT__ bz_r;

    #pragma omp parallel for \
    shared( snx, sny, nx, ny, px, py \
          , mz, ez \
          , fz_r, cf_r, ce_r) \
    private( idx, i, j, idxe \
           , mkz, bz_r) \
    schedule(guided)
    for(idx=0; idx<snx*sny; idx++)
    {
        i    = idx /   sny     ;
        j    = idx - i*sny     ;
        idxe = (i+px)*ny+(j+py);
        mkz  = mz[idx];
        bz_r = cf_r[mkz]*fz_r[idx] + ce_r[mkz]*ez[idxe];
        fz_r[idx] = bz_r;
    }
}

__KERNEL__ void update_dpole_3d( int snx, int sny, int snz \
                               , int  nx, int  ny, int  nz \
                               , int  px, int  py, int  pz \
                               , __GLOBAL__ int* mx, __GLOBAL__ int* my, __GLOBAL__ int* mz \
                               , __GLOBAL__ __FLOAT__* ex  , __GLOBAL__ __FLOAT__* ey  , __GLOBAL__ __FLOAT__* ez   \
                               , __GLOBAL__ __FLOAT__* fx_r, __GLOBAL__ __FLOAT__* fy_r, __GLOBAL__ __FLOAT__* fz_r \
                               , __GLOBAL__ const __FLOAT__* cf_r \
                               , __GLOBAL__ const __FLOAT__* ce_r )
{
    int idx, i, j, k, idxe, mkx, mky, mkz;
    __FLOAT__ bx_r, by_r, bz_r;

    #pragma omp parallel for \
    shared( snx, sny, snz, nx, ny, nz, px, py, pz \
          , mx, my, mz, ex, ey, ez \
          , fx_r, fy_r, fz_r, cf_r, ce_r) \
    private( idx, i, j, k, idxe \
           , mkx, mky, mkz, bx_r, by_r, bz_r) \
    schedule(guided)
    for(idx=0; idx<snx*sny*snz; idx++)
    {
        i    =  idx /  (sny*snz)       ;
        j    = (idx - i*sny*snz)/   snz;
        k    =  idx - i*sny*snz - j*snz;
        idxe = (i+px)*ny*nz+(j+py)*nz+(k+pz);
        mkx  = mx[idx];
        mky  = my[idx];
        mkz  = mz[idx];
        bx_r = cf_r[mkx]*fx_r[idx] + ce_r[mkx]*ex[idxe];
        by_r = cf_r[mky]*fy_r[idx] + ce_r[mky]*ey[idxe];
        bz_r = cf_r[mkz]*fz_r[idx] + ce_r[mkz]*ez[idxe];
        fx_r[idx] = bx_r;
        fy_r[idx] = by_r;
        fz_r[idx] = bz_r;
    }
}

__KERNEL__ void update_cpole_2dte( int snx, int sny \
                                 , int  nx, int  ny \
                                 , int  px, int  py \
                                 , __GLOBAL__ int* mx, __GLOBAL__ int* my \
                                 , __GLOBAL__ __FLOAT__* ex  , __GLOBAL__ __FLOAT__* ey   \
                                 , __GLOBAL__ __FLOAT__* fx_r, __GLOBAL__ __FLOAT__* fy_r \
                                 , __GLOBAL__ __FLOAT__* fx_i, __GLOBAL__ __FLOAT__* fy_i \
                                 , __GLOBAL__ const __FLOAT__* cf_r, __GLOBAL__ const __FLOAT__* cf_i \
                                 , __GLOBAL__ const __FLOAT__* ce_r, __GLOBAL__ const __FLOAT__* ce_i )
{
    int idx, i, j, idxe, mkx, mky;
    __FLOAT__ bx_r, bx_i, by_r, by_i;

    #pragma omp parallel for \
    shared( snx, sny, nx, ny, px, py \
          , mx, my, ex, ey \
          , fx_r, fy_r, fx_i, fy_i \
          , cf_r, cf_i, ce_r, ce_i \
          ) \
    private( idx, i, j, idxe \
           , mkx, mky, bx_r, bx_i, by_r, by_i \
           ) \
    schedule(guided)
    for(idx=0; idx<snx*sny; idx++)
    {
        i    = idx /   sny     ;
        j    = idx - i*sny     ;
        idxe = (i+px)*ny+(j+py);
        mkx = mx[idx];
        mky = my[idx];
        bx_r = cf_r[mkx]*fx_r[idx] - cf_i[mkx]*fx_i[idx] + ce_r[mkx]*ex[idxe];
        bx_i = cf_r[mkx]*fx_i[idx] + cf_i[mkx]*fx_r[idx] + ce_i[mkx]*ex[idxe];
        by_r = cf_r[mky]*fy_r[idx] - cf_i[mky]*fy_i[idx] + ce_r[mky]*ey[idxe];
        by_i = cf_r[mky]*fy_i[idx] + cf_i[mky]*fy_r[idx] + ce_i[mky]*ey[idxe];
        fx_r[idx] = bx_r;
        fx_i[idx] = bx_i;
        fy_r[idx] = by_r;
        fy_i[idx] = by_i;
    }
}

__KERNEL__ void update_cpole_2dtm( int snx, int sny \
                                 , int  nx, int  ny \
                                 , int  px, int  py \
                                 , __GLOBAL__ int* mz \
                                 , __GLOBAL__ __FLOAT__* ez   \
                                 , __GLOBAL__ __FLOAT__* fz_r, __GLOBAL__ __FLOAT__* fz_i \
                                 , __GLOBAL__ const __FLOAT__* cf_r, __GLOBAL__ const __FLOAT__* cf_i \
                                 , __GLOBAL__ const __FLOAT__* ce_r, __GLOBAL__ const __FLOAT__* ce_i )
{
    int idx, i, j, idxe, mkz;
    __FLOAT__ bz_r, bz_i;

    #pragma omp parallel for \
    shared( snx, sny, nx, ny, px, py \
          , mz, ez \
          , fz_r, fz_i \
          , cf_r, cf_i, ce_r, ce_i \
          ) \
    private( idx, i, j, idxe \
           , mkz, bz_r, bz_i \
           ) \
    schedule(guided)
    for(idx=0; idx<snx*sny; idx++)
    {
        i    = idx /   sny     ;
        j    = idx - i*sny     ;
        idxe = (i+px)*ny+(j+py);
        mkz  = mz[idx];
        bz_r = cf_r[mkz]*fz_r[idx] - cf_i[mkz]*fz_i[idx] + ce_r[mkz]*ez[idxe];
        bz_i = cf_r[mkz]*fz_i[idx] + cf_i[mkz]*fz_r[idx] + ce_i[mkz]*ez[idxe];
        fz_r[idx] = bz_r;
        fz_i[idx] = bz_i;
    }
}

__KERNEL__ void update_cpole_3d( int snx, int sny, int snz \
                               , int  nx, int  ny, int  nz \
                               , int  px, int  py, int  pz \
                               , __GLOBAL__ int* mx, __GLOBAL__ int* my, __GLOBAL__ int* mz \
                               , __GLOBAL__ __FLOAT__* ex  , __GLOBAL__ __FLOAT__* ey  , __GLOBAL__ __FLOAT__* ez   \
                               , __GLOBAL__ __FLOAT__* fx_r, __GLOBAL__ __FLOAT__* fy_r, __GLOBAL__ __FLOAT__* fz_r \
                               , __GLOBAL__ __FLOAT__* fx_i, __GLOBAL__ __FLOAT__* fy_i, __GLOBAL__ __FLOAT__* fz_i \
                               , __GLOBAL__ const __FLOAT__* cf_r, __GLOBAL__ const __FLOAT__* cf_i \
                               , __GLOBAL__ const __FLOAT__* ce_r, __GLOBAL__ const __FLOAT__* ce_i )
{
    int idx, i, j, k, idxe, mkx, mky, mkz;
    __FLOAT__ bx_r, bx_i, by_r, by_i, bz_r, bz_i;

    #pragma omp parallel for \
    shared( snx, sny, snz, nx, ny, nz, px, py, pz \
          , mx, my, mz, ex, ey, ez \
          , fx_r, fy_r, fz_r, fx_i, fy_i, fz_i \
          , cf_r, cf_i, ce_r, ce_i \
          ) \
    private( idx, i, j, k, idxe \
           , mkx, mky, mkz, bx_r, bx_i, by_r, by_i, bz_r, bz_i \
           ) \
    schedule(guided)
    for(idx=0; idx<snx*sny*snz; idx++)
    {
        i    =  idx /  (sny*snz)       ;
        j    = (idx - i*sny*snz)/   snz;
        k    =  idx - i*sny*snz - j*snz;
        idxe = (i+px)*ny*nz+(j+py)*nz+(k+pz);
        mkx  = mx[idx];
        mky  = my[idx];
        mkz  = mz[idx];
        bx_r = cf_r[mkx]*fx_r[idx] - cf_i[mkx]*fx_i[idx] + ce_r[mkx]*ex[idxe];
        bx_i = cf_r[mkx]*fx_i[idx] + cf_i[mkx]*fx_r[idx] + ce_i[mkx]*ex[idxe];
        by_r = cf_r[mky]*fy_r[idx] - cf_i[mky]*fy_i[idx] + ce_r[mky]*ey[idxe];
        by_i = cf_r[mky]*fy_i[idx] + cf_i[mky]*fy_r[idx] + ce_i[mky]*ey[idxe];
        bz_r = cf_r[mkz]*fz_r[idx] - cf_i[mkz]*fz_i[idx] + ce_r[mkz]*ez[idxe];
        bz_i = cf_r[mkz]*fz_i[idx] + cf_i[mkz]*fz_r[idx] + ce_i[mkz]*ez[idxe];
        fx_r[idx] = bx_r;
        fx_i[idx] = bx_i;
        fy_r[idx] = by_r;
        fy_i[idx] = by_i;
        fz_r[idx] = bz_r;
        fz_i[idx] = bz_i;
    }
}

__KERNEL__ void update_e_2dte( int snx, int sny \
                             , int  nx, int  ny \
                             , int  px, int  py \
                             , __GLOBAL__ __FLOAT__* ex  , __GLOBAL__ __FLOAT__* ey   \
                             , __GLOBAL__ __FLOAT__* fx_r, __GLOBAL__ __FLOAT__* fy_r )
{
    int idx, i, j, idxe;

    #pragma omp parallel for \
    shared( snx, sny, nx, ny, px, py \
          , ex, ey, fx_r, fy_r \
          ) \
    private( idx, i, j, idxe ) \
    schedule(guided)
    for(idx=0; idx<snx*sny; idx++)
    {
        i    = idx /   sny     ;
        j    = idx - i*sny     ;
        idxe = (i+px)*ny+(j+py);
        ex[idxe] += fx_r[idx];
        ey[idxe] += fy_r[idx];
    }
}

__KERNEL__ void update_e_2dtm( int snx, int sny \
                             , int  nx, int  ny \
                             , int  px, int  py \
                             , __GLOBAL__ __FLOAT__* ez   \
                             , __GLOBAL__ __FLOAT__* fz_r )
{
    int idx, i, j, idxe;

    #pragma omp parallel for \
    shared( snx, sny, nx, ny, px, py \
          , ez, fz_r \
          ) \
    private( idx, i, j, idxe ) \
    schedule(guided)
    for(idx=0; idx<snx*sny; idx++)
    {
        i    = idx /   sny     ;
        j    = idx - i*sny     ;
        idxe = (i+px)*ny+(j+py);
        ez[idxe] += fz_r[idx];
    }
}

__KERNEL__ void update_e_3d( int snx, int sny, int snz \
                           , int  nx, int  ny, int  nz \
                           , int  px, int  py, int  pz \
                           , __GLOBAL__ __FLOAT__* ex  , __GLOBAL__ __FLOAT__* ey  , __GLOBAL__ __FLOAT__* ez   \
                           , __GLOBAL__ __FLOAT__* fx_r, __GLOBAL__ __FLOAT__* fy_r, __GLOBAL__ __FLOAT__* fz_r )
{
    int idx, i, j, k, idxe;

    #pragma omp parallel for \
    shared( snx, sny, snz, nx, ny, nz, px, py, pz \
          , ex, ey, ez, fx_r, fy_r, fz_r \
          ) \
    private( idx, i, j, k, idxe ) \
    schedule(guided)
    for(idx=0; idx<snx*sny*snz; idx++)
    {
        i    =  idx /  (sny*snz)       ;
        j    = (idx - i*sny*snz)/   snz;
        k    =  idx - i*sny*snz - j*snz;
        idxe = (i+px)*ny*nz+(j+py)*nz+(k+pz);
        ex[idxe] += fx_r[idx];
        ey[idxe] += fy_r[idx];
        ez[idxe] += fz_r[idx];
    }
}
