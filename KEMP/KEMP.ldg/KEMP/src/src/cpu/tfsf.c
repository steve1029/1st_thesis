/*
Author        : Myung-Su Seok
Purpose       : OpenCL kernels to incident wave
Target        : GPU
Last Modified : 2012.05.08
*/

#include <omp.h>

__ACTIVATE_FLOAT64__

__KERNEL__ void update_tfsf_2d( int ax, int dr, int sg \
                              , int id_tfsf, int id_incf \
                              , int i_strt, int j_strt \
                              , int nx_tfsf, int ny_tfsf \
                              , int nx, int ny \
                              , __GLOBAL__ __FLOAT__* tfsf, __GLOBAL__ __FLOAT__* incf \
                              , __GLOBAL__ __FLOAT__* cf \
                              , __FLOAT__ rds )
{
    int idx;
    int idx_tfsf, idx_incf, idx_max, i, j;

    if(ax==0){ idx_max = ny_tfsf; }
    if(ax==1){ idx_max = nx_tfsf; }

    #pragma omp parallel for \
    shared( ax, dr, sg \
          , id_tfsf, id_incf \
          , i_strt, j_strt \
          , nx_tfsf, ny_tfsf \
          , nx, ny \
          , tfsf, incf \
          , cf \
          , rds \
          , idx_max ) \
    private( idx, idx_tfsf, idx_incf, i, j ) \
    schedule(guided)
    for(idx=0; idx<idx_max; idx++)
    {
        if(ax==0)
        {
            i = i_strt + dr*(nx_tfsf-1);
            j = j_strt + idx;
            idx_tfsf = (i+id_tfsf)*ny + j;
            idx_incf = (i+id_incf)*ny + j;
        }
        if(ax==1)
        {
            i = i_strt + idx;
            j = j_strt + dr*(ny_tfsf-1);
            idx_tfsf = i*ny + (j+id_tfsf);
            idx_incf = i*ny + (j+id_incf);
        }
        tfsf[idx_tfsf] += sg*cf[idx_tfsf]*rds*incf[idx_incf];
    }
}

__KERNEL__ void update_tfsf_3d( int ax, int dr \
                              , int id_tfsf, int id_incf \
                              , int i_strt, int j_strt, int k_strt \
                              , int nx_tfsf, int ny_tfsf, int nz_tfsf \
                              , int nx, int ny, int nz \
                              , __GLOBAL__ __FLOAT__* tfsf0, __GLOBAL__ __FLOAT__* incf0 \
                              , __GLOBAL__ __FLOAT__* tfsf1, __GLOBAL__ __FLOAT__* incf1 \
                              , __GLOBAL__ __FLOAT__* cf0, __GLOBAL__ __FLOAT__* cf1 \
                              , __FLOAT__ rds )
{
    int idx;
    int idx_tfsf, idx_incf, idx_max, i, j, k;

    if(ax==0){ idx_max = ny_tfsf*nz_tfsf; }
    if(ax==1){ idx_max = nz_tfsf*nx_tfsf; }
    if(ax==2){ idx_max = nx_tfsf*ny_tfsf; }

    #pragma omp parallel for \
    shared( ax, dr \
          , id_tfsf, id_incf \
          , i_strt, j_strt, k_strt \
          , nx_tfsf, ny_tfsf, nz_tfsf \
          , nx, ny, nz \
          , tfsf0, incf0 \
          , tfsf1, incf1 \
          , cf0, cf1 \
          , rds \
          , idx_max ) \
    private( idx, idx_tfsf, idx_incf, i, j, k ) \
    schedule(guided)
    for(idx=0; idx<idx_max; idx++)
    {
        if(ax==0)
        {
            i = dr*(nx_tfsf-1);
            j = idx /   nz_tfsf;
            k = idx - j*nz_tfsf;
            i += i_strt;
            j += j_strt;
            k += k_strt;
            idx_tfsf = (i+id_tfsf)*ny*nz + j*nz + k;
            idx_incf = (i+id_incf)*ny*nz + j*nz + k;
        }
        if(ax==1)
        {
            i = idx /   nz_tfsf;
            j = dr*(ny_tfsf-1);
            k = idx - i*nz_tfsf;
            i += i_strt;
            j += j_strt;
            k += k_strt;
            idx_tfsf = i*ny*nz + (j+id_tfsf)*nz + k;
            idx_incf = i*ny*nz + (j+id_incf)*nz + k;
        }
        if(ax==2)
        {
            i = idx /   ny_tfsf;
            j = idx - i*ny_tfsf;
            k = dr*(nz_tfsf-1);
            i += i_strt;
            j += j_strt;
            k += k_strt;
            idx_tfsf = i*ny*nz + j*nz + (k+id_tfsf);
            idx_incf = i*ny*nz + j*nz + (k+id_incf);
        }
        tfsf0[idx_tfsf] += cf0[idx_tfsf]*rds*incf0[idx_incf];
        tfsf1[idx_tfsf] -= cf1[idx_tfsf]*rds*incf1[idx_incf];
    }
}
