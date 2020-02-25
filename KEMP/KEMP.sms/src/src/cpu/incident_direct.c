/*
Author        : Myung-Su Seok
Purpose       : OpenCL kernels to incident wave
Target        : GPU
Last Modified : 2012.05.08
*/

#include <omp.h>

ACTIVATE_FLOAT64

__kernel void sine(int i_start, int j_start, int k_start, \
                   int i_stop, int j_stop, int k_stop, \
                   int nx_d, int ny_d, int nz_d, int nx_h, int ny_h, int nz_h, \
                   __global FLOAT* field, FLOAT func_input, __global FLOAT* random_phase, \
                   __global FLOAT* sin_tht, __global FLOAT* cos_tht, \
                   __global FLOAT* sin_phi, __global FLOAT* cos_phi, FLOAT coeff, int ii)
{
    int idx_h = get_global_id(0);
    int i_h   =  idx_h /     (ny_h*nz_h)           ;
    int j_h   = (idx_h - i_h*(ny_h*nz_h))     /nz_h;
    int k_h   =  idx_h - i_h*(ny_h*nz_h) - j_h*nz_h;
    int i_d   = i_h + i_start;
    int j_d   = j_h + j_start;
    int k_d   = k_h + k_start;
    int idx_d = i_d*ny_d*nz_d + j_d*nz_d + k_d;

    if(idx_h<nx_h*ny_h*nz_h)
    {
        if(ii == 0)
        {
            field[idx_d] += sin_tht[idx_h]*cos_phi[idx_h]*sin(func_input + random_phase[idx_h])*coeff;
        }
        if(ii == 1)
        {
            field[idx_d] += sin_tht[idx_h]*sin_phi[idx_h]*sin(func_input + random_phase[idx_h])*coeff;
        }
        if(ii == 2)
        {
            field[idx_d] += cos_tht[idx_h]               *sin(func_input + random_phase[idx_h])*coeff;
        }
    }
}

__kernel void cosn(int i_start, int j_start, int k_start, \
                   int i_stop, int j_stop, int k_stop, \
                   int nx_d, int ny_d, int nz_d, int nx_h, int ny_h, int nz_h, \
                   __global FLOAT* field, FLOAT func_input, __global FLOAT* random_phase, \
                   __global FLOAT* sin_tht, __global FLOAT* cos_tht, \
                   __global FLOAT* sin_phi, __global FLOAT* cos_phi, FLOAT coeff, int ii)
{
    int idx_h = get_global_id(0);
    int i_h   =  idx_h /     (ny_h*nz_h)           ;
    int j_h   = (idx_h - i_h*(ny_h*nz_h))     /nz_h;
    int k_h   =  idx_h - i_h*(ny_h*nz_h) - j_h*nz_h;
    int i_d   = i_h + i_start;
    int j_d   = j_h + j_start;
    int k_d   = k_h + k_start;
    int idx_d = i_d*ny_d*nz_d + j_d*nz_d + k_d;

    if(idx_h<nx_h*ny_h*nz_h)
    {
        if(ii == 0)
        {
            field[idx_d] += sin_tht[idx_h]*cos_phi[idx_h]*cos(func_input + random_phase[idx_h])*coeff;
        }
        if(ii == 1)
        {
            field[idx_d] += sin_tht[idx_h]*sin_phi[idx_h]*cos(func_input + random_phase[idx_h])*coeff;
        }
        if(ii == 2)
        {
            field[idx_d] += cos_tht[idx_h]               *cos(func_input + random_phase[idx_h])*coeff;
        }
    }
}

__kernel void expn(int i_start, int j_start, int k_start, \
                   int i_stop, int j_stop, int k_stop, \
                   int nx_d, int ny_d, int nz_d, int nx_h, int ny_h, int nz_h, \
                   __global FLOAT* field, FLOAT func_input, __global FLOAT* random_phase, \
                   __global FLOAT* sin_tht, __global FLOAT* cos_tht, \
                   __global FLOAT* sin_phi, __global FLOAT* cos_phi, FLOAT coeff, int ii)
{
    int idx_h = get_global_id(0);
    int i_h   =  idx_h /     (ny_h*nz_h)           ;
    int j_h   = (idx_h - i_h*(ny_h*nz_h))     /nz_h;
    int k_h   =  idx_h - i_h*(ny_h*nz_h) - j_h*nz_h;
    int i_d   = i_h + i_start;
    int j_d   = j_h + j_start;
    int k_d   = k_h + k_start;
    int idx_d = i_d*ny_d*nz_d + j_d*nz_d + k_d;

    if(idx_h<nx_h*ny_h*nz_h)
    {
        if(ii == 0)
        {
            field[idx_d] += sin_tht[idx_h]*cos_phi[idx_h]*exp(func_input + random_phase[idx_h])*coeff;
        }
        if(ii == 1)
        {
            field[idx_d] += sin_tht[idx_h]*sin_phi[idx_h]*exp(func_input + random_phase[idx_h])*coeff;
        }
        if(ii == 2)
        {
            field[idx_d] += cos_tht[idx_h]               *exp(func_input + random_phase[idx_h])*coeff;
        }
    }
}
