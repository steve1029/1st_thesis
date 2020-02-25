/*
Author        : Myung-Su Seok
Purpose       : OpenCL kernels for RFT
Target        : GPU
Last Modified : 2013.04.05
*/

#include <omp.h>

__ACTIVATE_FLOAT64__

__KERNEL__ void update( int i_start, int j_start, int k_start \
                      , int nx_d   , int ny_d   , int nz_d    \
                      , int nx_h   , int ny_h   , int nz_h    \
                      , int nw     , int tstep  \
                      , __FLOAT__ dt \
                      , __FLOAT__ pha_re \
                      , __FLOAT__ pha_im \
                      , __GLOBAL__ __FLOAT__* wfreq \
                      , __GLOBAL__ __FLOAT__* field \
                      , __GLOBAL__ __FLOAT__* rft_re \
                      , __GLOBAL__ __FLOAT__* rft_im )
{
    int idx_h, i_h, j_h, k_h, idx_d, i_d, j_d, k_d, w;
    __FLOAT__ fd, wt;

    #pragma omp parallel for \
    shared( i_start, j_start, k_start \
          , nx_h, ny_h, nz_h \
          , nx_d, ny_d, nz_d \
          , nw, tstep, dt \
          , pha_re, pha_im \
          , wfreq, field \
          , rft_re, rft_im \
          ) \
    private( idx_h, idx_d, w \
           , i_h, j_h, k_h \
           , i_d, j_d, k_d \
           , w, fd, wt \
           )
    for(idx_h=0; idx_h<nx_h*ny_h*nz_h*nw; idx++)
    {
        i_h   =  idx_h /     (ny_h*nz_h*nw)                         ;
        j_h   = (idx_h - i_h*(ny_h*nz_h*nw))     /(nz_h*nw)         ;
        k_h   = (idx_h - i_h*(ny_h*nz_h*nw) - j_h*(nz_h*nw))     /nw;
        w     =  idx_h - i_h*(ny_h*nz_h*nw) - j_h*(nz_h*nw) - k_h*nw;
        i_d   = i_h + i_start;
        j_d   = j_h + j_start;
        k_d   = k_h + k_start;
        idx_d = i_d*ny_d*nz_d + j_d*nz_d + k_d;
        fd = field[idx_d];
        wt = wfreq[w]*tstep*dt;
        rft_re[idx_h] += fd*cos(wt + pha_re);
        rft_im[idx_h] += fd*cos(wt + pha_im);
    }
}
/*
__kernel void update(int i_start, int j_start, int k_start, \
                     int i_stop , int j_stop , int k_stop , \
                     int nx_d   , int ny_d   , int nz_d   , \
                     int nx_h   , int ny_h   , int nz_h   , \
                     int nt     , int nw     , int t      , \
                     FLOAT dt   , __global FLOAT* wdomain , \
                     __global FLOAT* field, __global FLOAT* rft_re, __global FLOAT* rft_im)
{
    int idx_h = get_global_id(0);
    int i_h   =  idx_h /     (ny_h*nz_h*nw)                         ;
    int j_h   = (idx_h - i_h*(ny_h*nz_h*nw))     /(nz_h*nw)         ;
    int k_h   = (idx_h - i_h*(ny_h*nz_h*nw) - j_h*(nz_h*nw))     /nw;
    int w     =  idx_h - i_h*(ny_h*nz_h*nw) - j_h*(nz_h*nw) - k_h*nw;
    int i_d   = i_h + i_start;
    int j_d   = j_h + j_start;
    int k_d   = k_h + k_start;
    int idx_d = i_d*ny_d*nz_d + j_d*nz_d + k_d;

    if(idx_h<nx_h*ny_h*nz_h*nw)
    {
        rft_re[idx_h] += field[idx_d]*cos(wdomain[w]*t*dt)/nt;
        rft_im[idx_h] += field[idx_d]*sin(wdomain[w]*t*dt)/nt;
    }
}

__kernel void update_meanvalue(int i_start, int j_start, int k_start, \
                               int i_stop , int j_stop , int k_stop , \
                               int nx_d   , int ny_d   , int nz_d   , \
                               int nx_h   , int ny_h   , int nz_h   , \
                               int nt     , int nw     , int t      , \
                               FLOAT dt   , __global FLOAT* wdomain , \
                               __global FLOAT* field, __global FLOAT* rft_re, __global FLOAT* rft_im)
{
    int idx_h = get_global_id(0);
    int i_h   =  idx_h /     (ny_h*nz_h*nw)                         ;
    int j_h   = (idx_h - i_h*(ny_h*nz_h*nw))     /(nz_h*nw)         ;
    int k_h   = (idx_h - i_h*(ny_h*nz_h*nw) - j_h*(nz_h*nw))     /nw;
    int w     =  idx_h - i_h*(ny_h*nz_h*nw) - j_h*(nz_h*nw) - k_h*nw;
    int i_d   = i_h + i_start;
    int j_d   = j_h + j_start;
    int k_d   = k_h + k_start;
    int idx_d = i_d*ny_d*nz_d + j_d*nz_d + k_d;

    if(idx_h<nx_h*ny_h*nz_h*nw)
    {
        rft_re[w] += field[idx_d]*cos(wdomain[w]*t*dt)/(nt*nx_h*ny_h*nz_h);
        rft_im[w] += field[idx_d]*sin(wdomain[w]*t*dt)/(nt*nx_h*ny_h*nz_h);
    }
}

*/