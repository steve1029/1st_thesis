/*
Author        : Myung-Su Seok
Purpose       : OpenCL kernels to setting geometric objects
Target        : GPU
Last Modified : 2012.12.04
*/

#include <omp.h>

__ACTIVATE_FLOAT64__

__FLOAT__ return_max(__FLOAT__ val1, __FLOAT__ val2)
{
    if  (val1>val2){ return val1; }
    else           { return val2; }
}

__FLOAT__ return_min(__FLOAT__ val1, __FLOAT__ val2)
{
    if  (val1<val2){ return val1; }
    else           { return val2; }
}

void set_ces( int idx \
            , __GLOBAL__ __FLOAT__* material_params \
            , __GLOBAL__ __FLOAT__* ce1s \
            , __GLOBAL__ __FLOAT__* ce2s \
            , __MARK__ \
	    )
{
    __FLOAT__ ce1 =      material_params[0];
    __FLOAT__ ce2 =      material_params[1];
    int   mrk     = (int)material_params[2];
    ce1s[idx]     = ce1;
    ce2s[idx]     = ce2;
    __SET_MARK__;
}

void set_chs( int idx \
            , __GLOBAL__ __FLOAT__* material_params \
            , __GLOBAL__ __FLOAT__* ch1s \
            , __GLOBAL__ __FLOAT__* ch2s \
            , __MARK__ \
	    )
{
    __FLOAT__ ch1 =      material_params[0];
    __FLOAT__ ch2 =      material_params[1];
    int   mrk     = (int)material_params[2];
    ch1s[idx]     = ch1;
    ch2s[idx]     = ch2;
    __SET_MARK__;
}

void set_cehs( int idx \
             , __GLOBAL__ __FLOAT__* material_params \
             , __GLOBAL__ __FLOAT__* ce1s \
             , __GLOBAL__ __FLOAT__* ce2s \
             , __GLOBAL__ __FLOAT__* ch1s \
             , __GLOBAL__ __FLOAT__* ch2s \
             , __MARK__ \
	     )
{
    __FLOAT__ ce1 =      material_params[0];
    __FLOAT__ ce2 =      material_params[1];
    __FLOAT__ ch1 =      material_params[2];
    __FLOAT__ ch2 =      material_params[3];
    int   mrk     = (int)material_params[4];
    ce1s[idx]     = ce1;
    ce2s[idx]     = ce2;
    ch1s[idx]     = ch1;
    ch2s[idx]     = ch2;
    __SET_MARK__;
}

__KERNEL__ void mark_to_mk_2d( int  nx, int  ny \
                             , int snx, int sny \
                             , int  px, int  py \
                             , __GLOBAL__ int* mrkx, __GLOBAL__ int* mrky, __GLOBAL__ int* mrkz \
                             , __GLOBAL__ int* mx  , __GLOBAL__ int* my  , __GLOBAL__ int* mz   \
			     )
{
    int idx, i, j, idxm;

    #pragma omp parallel for \
    shared( nx, ny, snx, sny, px, py \
	  , mrkx, mrky, mrkz, mx, my, mz \
	  ) \
    private( idx, i, j, idxm ) \
    schedule(guided)
    for(idx=0; idx<snx*sny; idx++)
    {
        i    = idx    /sny;
        j    = idx - i*sny;
        idxm = (i+px)*ny + (j+py);
        mx[idx] = mrkx[idxm];
        my[idx] = mrky[idxm];
        mz[idx] = mrkz[idxm];
    }
}

__KERNEL__ void mark_to_mk_3d( int  nx, int  ny, int  nz \
                             , int snx, int sny, int snz \
                             , int  px, int  py, int  pz \
                             , __GLOBAL__ int* mrkx, __GLOBAL__ int* mrky, __GLOBAL__ int* mrkz \
                             , __GLOBAL__ int* mx  , __GLOBAL__ int* my  , __GLOBAL__ int* mz   \
			     )
{
    int idx, i, j, k, idxm;

    #pragma omp parallel for \
    shared( nx, ny, nz, snx, sny, snz, px, py, pz \
	  , mrkx, mrky, mrkz, mx, my, mz \
	  ) \
    private( idx, i, j, k, idxm ) \
    schedule(guided)
    for(idx=0; idx<snx*sny*snz; idx++)
    {
        i   =  idx /  (sny*snz)       ;
        j   = (idx - i*sny*snz)   /snz;
        k   =  idx - i*sny*snz - j*snz;
        idxm = (i+px)*ny*nz + (j+py)*nz + (k+pz);
        mx[idx] = mrkx[idxm];
        my[idx] = mrky[idxm];
        mz[idx] = mrkz[idxm];
    }
}

__KERNEL__ void set_struc_poly( __GLOBAL__ __FLOAT__* poly \
                              , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y \
                              , int nx, int ny, int np \
                              , __GLOBAL__ __FLOAT__* org \
                              , __FLOAT__ sin, __FLOAT__ cos \
                              , __GLOBAL__ __FLOAT__* material_params \
                              , __COEFFICIENT_FIELDS__ \
			      )
{
    int idx, i, j;
    int p, idp1, idp2, sg;
    __FLOAT__ px, py, px_rot, py_rot, sign, p1x, p1y, p2x, p2y, org_x, org_y;
    org_x = org[0];
    org_y = org[1];

    #pragma omp parallel for \
    shared( poly, x, y \
          , nx, ny, np \
	  , org, sin, cos, material_params \
	  , __COEFFICIENTS__ \
          ) \
    private( idx, i, j, p, idp1, idp2, sg \
           , px, py, px_rot, py_rot \
	   , p1x, p1y, p2x, p2y \
	   , org_x, org_y \
           ) \
    schedule(guided)
    for(idx=0; idx<nx*ny; idx++)
    {
        i  = idx    /ny;
        j  = idx - i*ny;
        px = x[i]-org_x;
        py = y[j]-org_y;
        px_rot =  px*cos + py*sin;
        py_rot = -px*sin + py*cos;
        sg = 0;
        for(p=0; p<np; p++)
        {
            idp1 = ((p  )   )*2;
            idp2 = ((p+1)%np)*2;
            p1x  = poly[idp1  ] - org_x;
            p1y  = poly[idp1+1] - org_y;
            p2x  = poly[idp2  ] - org_x;
            p2y  = poly[idp2+1] - org_y;
            if(py_rot>=return_min(p1y,p2y) && py_rot<=return_max(p1y,p2y) && px_rot<=return_max(p1x,p2x))
            {
                sign = (p2y-p1y)*((py_rot-p1y)*(p2x-p1x) - (px_rot-p1x)*(p2y-p1y));
                if(p1x == p2x || p1y == p2y || sign >= 0) { sg += 1; }
            }
        }
        if(sg%2 == 1)
        {
            __SET_MATERIAL__;
        }
    }
}

__KERNEL__ void set_struc_rect( __GLOBAL__ __FLOAT__* points \
                              , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y \
                              , int nx, int ny \
                              , __GLOBAL__ __FLOAT__* material_params \
                              , __COEFFICIENT_FIELDS__ \
			      )
{
    int idx, i, j;
    __FLOAT__ px, py;
    #pragma omp parallel for \
    shared( points, x, y \
          , nx, ny \
	  , material_params \
	  , __COEFFICIENTS__ \
	  ) \
    private( idx, i, j, px, py ) \
    schedule(guided)
    for(idx=0; idx<nx*ny; idx++)
    {
        i  = idx    /ny;
        j  = idx - i*ny;
        px = x[i];
        py = y[j];
        if(px>=points[0] && px<=points[2] && py>=points[1] && py<=points[3])
        {
            __SET_MATERIAL__;
        }
    }
}

__KERNEL__ void set_struc_ellp2d( __GLOBAL__ __FLOAT__* com, __GLOBAL__ __FLOAT__* radius \
                                , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y \
                                , int nx, int ny \
                                , __GLOBAL__ __FLOAT__* org \
                                , __FLOAT__ sin, __FLOAT__ cos \
                                , __GLOBAL__ __FLOAT__* material_params \
                                , __COEFFICIENT_FIELDS__ \
				)
{
    int idx, i, j;
    __FLOAT__ com_x, com_y, px, py, px_rot, py_rot, r_x, r_y, org_x, org_y;
    r_x = radius[0];
    r_y = radius[1];
    org_x = org[0];
    org_y = org[1];
    com_x = com[0] - org_x;
    com_y = com[1] - org_y;

    #pragma omp parallel for \
    shared( com, radius, x, y \
          , nx, ny \
	  , org, sin, cos, material_params \
	  , __COEFFICIENTS__ \
	  , com_x, com_y, r_x, r_y, org_x, org_y \
	  ) \
    private( idx, i, j \
           , px, py, px_rot, py_rot \
           ) \
    schedule(guided)
    for(idx=0; idx<nx*ny; idx++)
    {
        i   = idx    /ny;
        j   = idx - i*ny;
        px = x[i] - org_x;
        py = y[j] - org_y;
        px_rot =  px*cos + py*sin;
        py_rot = -px*sin + py*cos;
        if((px_rot-com_x)*(px_rot-com_x)/(r_x*r_x)+(py_rot-com_y)*(py_rot-com_y)/(r_y*r_y) <= 1)
        {
            __SET_MATERIAL__;
        }
    }
}

__KERNEL__ void set_struc_ellp3d( __GLOBAL__ __FLOAT__* com, __GLOBAL__ __FLOAT__* radius \
                                , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y, __GLOBAL__ __FLOAT__* z \
                                , int nx, int ny, int nz \
                                , __GLOBAL__ __FLOAT__* org, __GLOBAL__ __FLOAT__* mat \
                                , __GLOBAL__ __FLOAT__* material_params \
                                , __COEFFICIENT_FIELDS__ \
				)
{
    int idx, i, j, k;
    __FLOAT__ com_x, com_y, com_z, px, py, pz, px_rot, py_rot, pz_rot, \
              r_x, r_y, r_z, org_x, org_y, org_z;
    r_x = radius[0];
    r_y = radius[1];
    r_z = radius[2];
    org_x = org[0];
    org_y = org[1];
    org_z = org[2];
    com_x = com[0] - org_x;
    com_y = com[1] - org_y;
    com_z = com[2] - org_z;

    #pragma omp parallel for \
    shared( com, radius, x, y, z \
          , nx, ny, nz \
          , org, mat, material_params \
	  , __COEFFICIENTS__ \
	  , r_x, r_y, r_z, org_x, org_y, org_z, com_x, com_y, com_z \
          ) \
    private( idx, i, j, k \
           , px, py, pz, px_rot, py_rot, pz_rot \
           ) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i  =  idx /  (ny*nz)      ;
        j  = (idx - i*ny*nz)   /nz;
        k  =  idx - i*ny*nz - j*nz;
        px = x[i]-org_x;
        py = y[j]-org_y;
        pz = z[k]-org_z;
        px_rot = px*mat[0] + py*mat[1] + pz*mat[2];
        py_rot = px*mat[3] + py*mat[4] + pz*mat[5];
        pz_rot = px*mat[6] + py*mat[7] + pz*mat[8];
        if((px_rot-com_x)*(px_rot-com_x)/(r_x*r_x)+(py_rot-com_y)*(py_rot-com_y)/(r_y*r_y)+(pz_rot-com_z)*(pz_rot-com_z)/(r_z*r_z) <= 1)
        {
            __SET_MATERIAL__;
        }
    }
}

__KERNEL__ void set_struc_recd( __GLOBAL__ __FLOAT__* com, __GLOBAL__ __FLOAT__* length_width, __FLOAT__ height, int ax \
                              , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y, __GLOBAL__ __FLOAT__* z \
                              , int nx, int ny, int nz \
                              , __GLOBAL__ __FLOAT__* org, __GLOBAL__ __FLOAT__* mat \
                              , __GLOBAL__ __FLOAT__* material_params \
                              , __COEFFICIENT_FIELDS__ \
			      )
{
    int idx, i, j, k;
    __FLOAT__ com_x, com_y, com_z, lw0, lw1, \
              org_x, org_y, org_z, base, top, \
              px, py, pz, px_rot, py_rot, pz_rot;
    lw0   = length_width[0];
    lw1   = length_width[1];
    org_x = org[0];
    org_y = org[1];
    org_z = org[2];
    com_x = com[0] - org_x;
    com_y = com[1] - org_y;
    com_z = com[2] - org_z;
    if     (ax==0)
    {
        base = com_x - height*.5;
        top  = com_x + height*.5;
    }
    else if(ax==1)
    {
        base = com_y - height*.5;
        top  = com_y + height*.5;
    }
    else if(ax==2)
    {
        base = com_z - height*.5;
        top  = com_z + height*.5;
    }
    #pragma omp parallel for \
    shared( com, length_width, height, ax \
	  , x, y, z \
	  , nx, ny, nz \
	  , org, mat, material_params \
	  , __COEFFICIENTS__ \
	  , com_x, com_y, com_z \
	  , org_x, org_y, org_z \
	  , lw0, lw1, base, top \
          ) \
    private( idx, i, j, k \
           , px, py, pz \
	   , px_rot, py_rot, pz_rot \
	   ) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i   =  idx /  (ny*nz)      ;
        j   = (idx - i*ny*nz)   /nz;
        k   =  idx - i*ny*nz - j*nz;
        px = x[i]-org_x;
        py = y[j]-org_y;
        pz = z[k]-org_z;
        px_rot = px*mat[0] + py*mat[1] + pz*mat[2];
        py_rot = px*mat[3] + py*mat[4] + pz*mat[5];
        pz_rot = px*mat[6] + py*mat[7] + pz*mat[8];
        if     (ax==0 && px_rot>=base && px_rot<=top)
        {
            if(py_rot>=com_y-lw0*.5 && py_rot<=com_y+lw0*.5 && pz_rot>=com_z-lw1*.5 && pz_rot<=com_z+lw1*.5)
            {
                __SET_MATERIAL__;
            }
        }
        else if(ax==1 && py_rot>=base && py_rot<=top)
        {
            if(px_rot>=com_x-lw0*.5 && px_rot<=com_x+lw0*.5 && pz_rot>=com_z-lw1*.5 && pz_rot<=com_z+lw1*.5)
            {
                __SET_MATERIAL__;
            }
        }
        else if(ax==2 && pz_rot>=base && pz_rot<=top)
        {
            if(px_rot>=com_x-lw0*.5 && px_rot<=com_x+lw0*.5 && py_rot>=com_y-lw1*.5 && py_rot<=com_y+lw1*.5)
            {
                __SET_MATERIAL__;
            }
        }
    }
}

__KERNEL__ void set_struc_elcd( __GLOBAL__ __FLOAT__* com, __GLOBAL__ __FLOAT__* radius, __FLOAT__ height, int ax \
                              , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y, __GLOBAL__ __FLOAT__* z \
                              , int nx, int ny, int nz \
                              , __GLOBAL__ __FLOAT__* org, __GLOBAL__ __FLOAT__* mat \
                              , __GLOBAL__ __FLOAT__* material_params \
                              , __COEFFICIENT_FIELDS__ \
			      )
{
    int idx, i, j, k;
    __FLOAT__ com_x, com_y, com_z, r0, r1, \
              org_x, org_y, org_z, base, top, \
              px, py, pz, px_rot, py_rot, pz_rot;
    r0 = radius[0];
    r1 = radius[1];
    org_x = org[0];
    org_y = org[1];
    org_z = org[2];
    com_x = com[0] - org_x;
    com_y = com[1] - org_y;
    com_z = com[2] - org_z;
    if     (ax==0)
    {
        base = com_x - height*.5;
        top  = com_x + height*.5;
    }
    else if(ax==1)
    {
        base = com_y - height*.5;
        top  = com_y + height*.5;
    }
    else if(ax==2)
    {
        base = com_z - height*.5;
        top  = com_z + height*.5;
    }
    #pragma omp parallel for \
    shared( com, radius, height, ax \
	  , x, y, z \
	  , nx, ny, nz \
	  , org, mat, material_params \
	  , __COEFFICIENTS__ \
	  , com_x, com_y, com_z \
	  , org_x, org_y, org_z \
	  , r0, r1, base, top \
          ) \
    private( idx, i, j, k \
           , px, py, pz \
	   , px_rot, py_rot, pz_rot \
	   ) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i   =  idx /  (ny*nz)      ;
        j   = (idx - i*ny*nz)   /nz;
        k   =  idx - i*ny*nz - j*nz;
        px = x[i]-org_x;
        py = y[j]-org_y;
        pz = z[k]-org_z;
        px_rot = px*mat[0] + py*mat[1] + pz*mat[2];
        py_rot = px*mat[3] + py*mat[4] + pz*mat[5];
        pz_rot = px*mat[6] + py*mat[7] + pz*mat[8];
        if     (ax==0 && px_rot>=base && px_rot<=top)
        {
            if((py_rot-com_y)*(py_rot-com_y)/(r0*r0)+(pz_rot-com_z)*(pz_rot-com_z)/(r1*r1) <= 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(ax==1 && py_rot>=base && py_rot<=top)
        {
            if((pz_rot-com_z)*(pz_rot-com_z)/(r0*r0)+(px_rot-com_x)*(px_rot-com_x)/(r1*r1) <= 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(ax==2 && pz_rot>=base && pz_rot<=top)
        {
            if((px_rot-com_x)*(px_rot-com_x)/(r0*r0)+(py_rot-com_y)*(py_rot-com_y)/(r1*r1) <= 1)
            {
                __SET_MATERIAL__;
            }
        }
    }
}

__KERNEL__ void set_struc_plpm( __GLOBAL__ __FLOAT__* base_poly, __FLOAT__ height, int np, int axis \
                              , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y, __GLOBAL__ __FLOAT__* z \
                              , int nx, int ny, int nz \
                              , __GLOBAL__ __FLOAT__* org, __GLOBAL__ __FLOAT__* mat \
                              , __GLOBAL__ __FLOAT__* material_params \
                              , __COEFFICIENT_FIELDS__ \
			      )
{
    int idx, i, j, k;
    int p, idp0, idp1, sg;
    __FLOAT__ p00, p01, p10, p11, sign, \
              org_x, org_y, org_z, com_x, com_y, com_z, base, top, \
              px, py, pz, px_rot, py_rot, pz_rot;
    org_x = org[0];
    org_y = org[1];
    org_z = org[2];
    com_x = org[0];
    com_y = org[1];
    com_z = org[2];
    if     (axis==0)
    {
        base = base_poly[0]          - org_x;
        top  = base_poly[0] + height - org_x;
    }
    else if(axis==1)
    {
        base = base_poly[1]          - org_y;
        top  = base_poly[1] + height - org_y;
    }
    else if(axis==2)
    {
        base = base_poly[2]          - org_z;
        top  = base_poly[2] + height - org_z;
    }
    #pragma omp parallel for \
    shared( base_poly, height, np, axis \
	  , x, y, z \
	  , nx, ny, nz \
	  , org, mat, material_params \
	  , __COEFFICIENTS__ \
	  , org_x, org_y, org_z \
	  , com_x, com_y, com_z \
	  , base, top \
	  ) \
    private( idx, i, j, k \
           , p, idp0, idp1, sg \
	   , p00, p01, p10, p11, sign \
	   , px, py, pz \
	   , px_rot, py_rot, pz_rot \
	   ) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i   =  idx /  (ny*nz)      ;
        j   = (idx - i*ny*nz)   /nz;
        k   =  idx - i*ny*nz - j*nz;
        px = x[i]-org_x;
        py = y[j]-org_y;
        pz = z[k]-org_z;
        px_rot = px*mat[0] + py*mat[1] + pz*mat[2];
        py_rot = px*mat[3] + py*mat[4] + pz*mat[5];
        pz_rot = px*mat[6] + py*mat[7] + pz*mat[8];
        sg     = 0;
        if     (axis==0 && px_rot>=base && px_rot<=top)
        {
            for(p=0; p<np; p++){
                idp0 = ((p  )   )*3;
                idp1 = ((p+1)%np)*3;
                p00  = base_poly[idp0+1] - org_y;
                p01  = base_poly[idp0+2] - org_z;
                p10  = base_poly[idp1+1] - org_y;
                p11  = base_poly[idp1+2] - org_z;
                if(pz_rot >= return_min(p01,p11) && pz_rot <= return_max(p01,p11) && py_rot <= return_max(p00,p10))
                {
                    sign = (p11-p01)*((pz_rot-p01)*(p10-p00) - (py_rot-p00)*(p11-p01));
                    if(p00 == p10 || sign >= 0) { sg += 1; }
                }
            }
            if(sg%2 == 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(axis==1 && py_rot>=base && py_rot<=top)
        {
            for(p=0; p<np; p++){
                idp0 = ((p  )   )*3;
                idp1 = ((p+1)%np)*3;
                p00  = base_poly[idp0+2] - org_z;
                p01  = base_poly[idp0  ] - org_x;
                p10  = base_poly[idp1+2] - org_z;
                p11  = base_poly[idp1  ] - org_x;
                if(px_rot >= return_min(p01,p11) && px_rot <= return_max(p01,p11) && pz_rot <= return_max(p00,p10))
                {
                    sign = (p11-p01)*((px_rot-p01)*(p10-p00) - (pz_rot-p00)*(p11-p01));
                    if(p00 == p10 || sign >= 0) { sg += 1; }
                }
            }
            if(sg%2 == 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(axis==2 && pz_rot>=base && pz_rot<=top)
        {
            for(p=0; p<np; p++){
                idp0 = ((p  )   )*3;
                idp1 = ((p+1)%np)*3;
                p00  = base_poly[idp0  ] - org_x;
                p01  = base_poly[idp0+1] - org_y;
                p10  = base_poly[idp1  ] - org_x;
                p11  = base_poly[idp1+1] - org_y;
                if(py_rot >= return_min(p01,p11) && py_rot <= return_max(p01,p11) && px_rot <= return_max(p00,p10))
                {
                    sign = (p11-p01)*((py_rot-p01)*(p10-p00) - (px_rot-p00)*(p11-p01));
                    if(p00 == p10 || sign >= 0) { sg += 1; }
                }
            }
            if(sg%2 == 1)
            {
                __SET_MATERIAL__;
            }
        }
    }
}

__KERNEL__ void set_struc_boxs( __GLOBAL__ __FLOAT__* points \
                              , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y, __GLOBAL__ __FLOAT__* z \
                              , int nx, int ny, int nz \
                              , __GLOBAL__ __FLOAT__* org, __GLOBAL__ __FLOAT__* mat \							  
                              , __GLOBAL__ __FLOAT__* material_params \
                              , __COEFFICIENT_FIELDS__ \
			      )
{
    int idx, i, j, k;
	__FLOAT__ org_x, org_y, org_z, poo0, poo1, poo2, poo3, poo4, poo5, \
              px, py, pz, px_rot, py_rot, pz_rot;
    org_x = org[0];
    org_y = org[1];
    org_z = org[2];
    #pragma omp parallel for \
    shared( points \
          , x, y, z \
          , nx, ny, nz \
	  , org, mat\
	  , material_params \
	  , __COEFFICIENTS__ \
	  , org_x, org_y, org_z \
	  ) \
    private( idx, i, j, k \
           , px, py, pz \
	   , px_rot, py_rot, pz_rot \
	   ,poo0, poo1, poo2, poo3, poo4, poo5 \
	   ) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i   =  idx /  (ny*nz)      ;
        j   = (idx - i*ny*nz)   /nz;
        k   =  idx - i*ny*nz - j*nz;
        px = x[i]-org_x;
        py = y[j]-org_y;
        pz = z[k]-org_z;
        px_rot = px*mat[0] + py*mat[1] + pz*mat[2];
        py_rot = px*mat[3] + py*mat[4] + pz*mat[5];
        pz_rot = px*mat[6] + py*mat[7] + pz*mat[8];

		if (points[0]-org_x <= points[3]-org_x)         {poo0 = points[0]-org_x; poo3 = points[3]-org_x;}
		else if (points[0]-org_x >= points[3]-org_x)    {poo3 = points[0]-org_x; poo0 = points[3]-org_x;}
		if (points[1]-org_y <= points[4]-org_y)         {poo1 = points[1]-org_y; poo4 = points[4]-org_y;}
		else if (points[1]-org_y >= points[4]-org_y)    {poo4 = points[1]-org_y; poo1 = points[4]-org_y;}
		if (points[2]-org_z <= points[5]-org_z)         {poo2 = points[2]-org_z; poo5 = points[5]-org_z;}
		else if (points[2]-org_z >= points[5]-org_z)    {poo5 = points[2]-org_z; poo2 = points[5]-org_z;}
		if(px_rot>=poo0 && px_rot<=poo3 && py_rot>=poo1 && py_rot<=poo4 && pz_rot>=poo2 && pz_rot<=poo5)
        {
            __SET_MATERIAL__;
        }
    }
}



__KERNEL__ void set_struc_elcn( __GLOBAL__ __FLOAT__* com, __GLOBAL__ __FLOAT__* radius, __FLOAT__ height, int ax \
                              , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y, __GLOBAL__ __FLOAT__* z \
                              , int nx, int ny, int nz \
                              , __GLOBAL__ __FLOAT__* org, __GLOBAL__ __FLOAT__* mat \
                              , __GLOBAL__ __FLOAT__* material_params \
                              , __COEFFICIENT_FIELDS__ \
			      )

{
    int idx, i, j, k;
    __FLOAT__ com_x, com_y, com_z, r0, r1, r0_, r1_, \
          org_x, org_y, org_z, base, top, \
          px, py, pz, px_rot, py_rot, pz_rot;
    r0 = radius[0];
    r1 = radius[1];
    org_x = org[0];
    org_y = org[1];
    org_z = org[2];
    com_x = com[0] - org_x;
    com_y = com[1] - org_y;
    com_z = com[2] - org_z;
    if(ax==0)
    {
        base = com_x - height*1./3.;
        top  = com_x + height*2./3.;
    }
    else if(ax==1)
    {
        base = com_y - height*1./3.;
        top  = com_y + height*2./3.;
    }
    else if(ax==2)
    {
        base = com_z - height*1./3.;
        top  = com_z + height*2./3.;
    }
    #pragma omp parallel for \
    shared( com, radius, height, ax \
	  , x, y, z \
	  , nx, ny, nz \
	  , org, mat \
	  , material_params \
	  , __COEFFICIENTS__ \
	  , org_x, org_y, org_z \
	  , com_x, com_y, com_z \
	  , r0, r1 \
	  , base, top \
          ) \
    private( idx, i, j, k \
           , px, py, pz \
	   , px_rot, py_rot, pz_rot \
	   , r0_, r1_ \
           ) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i   =  idx /  (ny*nz)      ;
        j   = (idx - i*ny*nz)   /nz;
        k   =  idx - i*ny*nz - j*nz;
        px = x[i]-org_x;
        py = y[j]-org_y;
        pz = z[k]-org_z;
        px_rot = px*mat[0] + py*mat[1] + pz*mat[2];
        py_rot = px*mat[3] + py*mat[4] + pz*mat[5];
        pz_rot = px*mat[6] + py*mat[7] + pz*mat[8];
        if     (ax==0 && px_rot>=base && px_rot<=top)
        {
            r0_ = r0*(top-px_rot)/height;
            r1_ = r1*(top-px_rot)/height;
            if((py_rot-com_y)*(py_rot-com_y)/(r0_*r0_)+(pz_rot-com_z)*(pz_rot-com_z)/(r1_*r1_) <= 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(ax==1 && py_rot>=base && py_rot<=top)
        {
            r0_ = r0*(top-py_rot)/height;
            r1_ = r1*(top-py_rot)/height;
            if((pz_rot-com_z)*(pz_rot-com_z)/(r0_*r0_)+(px_rot-com_x)*(px_rot-com_x)/(r1_*r1_) <= 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(ax==2 && pz_rot>=base && pz_rot<=top)
        {
            r0_ = r0*(top-pz_rot)/height;
            r1_ = r1*(top-pz_rot)/height;
            if((px_rot-com_x)*(px_rot-com_x)/(r0_*r0_)+(py_rot-com_y)*(py_rot-com_y)/(r1_*r1_) <= 1)
            {
                __SET_MATERIAL__;
            }
        }
    }
}

__KERNEL__ void set_struc_eltrcn( __GLOBAL__ __FLOAT__* com, __GLOBAL__ __FLOAT__* radius, __FLOAT__ height, __FLOAT__ t_height, int ax \
                                , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y, __GLOBAL__ __FLOAT__* z \
                                , int nx, int ny, int nz \
                                , __GLOBAL__ __FLOAT__* org, __GLOBAL__ __FLOAT__* mat \
                                , __GLOBAL__ __FLOAT__* material_params \
                                , __COEFFICIENT_FIELDS__ \
				)
{
    int idx, i, j, k;
    __FLOAT__ com_x, com_y, com_z, r0, r1, r0_, r1_, \
          org_x, org_y, org_z, base, top, top_t, \
          px, py, pz, px_rot, py_rot, pz_rot;
    r0 = radius[0];
    r1 = radius[1];
    org_x = org[0];
    org_y = org[1];
    org_z = org[2];
    com_x = com[0] - org_x;
    com_y = com[1] - org_y;
    com_z = com[2] - org_z;
    if(ax==0)
    {
        base = com_x - height*1./3.;
        top  = com_x + height*2./3.;
        top_t= com_x + height*2./3.-t_height;
    }
    else if(ax==1)
    {
        base = com_y - height*1./3.;
        top  = com_y + height*2./3.;
        top_t= com_y + height*2./3.-t_height;
    }
    else if(ax==2)
    {
        base = com_z - height*1./3.;
        top  = com_z + height*2./3.;
        top_t= com_z + height*2./3.-t_height;
    }

    #pragma omp parallel for \
    shared( com, radius, height, t_height, ax \
	  , x, y, z \
	  , nx, ny, nz \
	  , org, mat \
	  , material_params \
	  , __COEFFICIENTS__ \
	  , org_x, org_y, org_z \
	  , com_x, com_y, com_z \
	  , r0, r1, base, top, top_t \
          ) \
    private( idx, i, j, k \
	   , px, py, pz \
	   , px_rot, py_rot, pz_rot \
	   , r0_, r1_ \
           ) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i   =  idx /  (ny*nz)      ;
        j   = (idx - i*ny*nz)   /nz;
        k   =  idx - i*ny*nz - j*nz;
        px = x[i]-org_x;
        py = y[j]-org_y;
        pz = z[k]-org_z;
        px_rot = px*mat[0] + py*mat[1] + pz*mat[2];
        py_rot = px*mat[3] + py*mat[4] + pz*mat[5];
        pz_rot = px*mat[6] + py*mat[7] + pz*mat[8];
        if     (ax==0 && px_rot>=base && px_rot<=top_t)
        {
            r0_ = r0*(top-px_rot)/height;
            r1_ = r1*(top-px_rot)/height;
            if((py_rot-com_y)*(py_rot-com_y)/(r0_*r0_)+(pz_rot-com_z)*(pz_rot-com_z)/(r1_*r1_) <= 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(ax==1 && py_rot>=base && py_rot<=top_t)
        {
            r0_ = r0*(top-py_rot)/height;
            r1_ = r1*(top-py_rot)/height;
            if((pz_rot-com_z)*(pz_rot-com_z)/(r0_*r0_)+(px_rot-com_x)*(px_rot-com_x)/(r1_*r1_) <= 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(ax==2 && pz_rot>=base && pz_rot<=top_t)
        {
            r0_ = r0*(top-pz_rot)/height;
            r1_ = r1*(top-pz_rot)/height;
            if((px_rot-com_x)*(px_rot-com_x)/(r0_*r0_)+(py_rot-com_y)*(py_rot-com_y)/(r1_*r1_) <= 1)
            {
                __SET_MATERIAL__;
            }
        }
    }
}

__KERNEL__ void set_struc_plpy( __GLOBAL__ __FLOAT__* base_poly, __GLOBAL__ __FLOAT__* com, __FLOAT__ height, int np, int axis \
                              , __GLOBAL__ __FLOAT__* x, __GLOBAL__ __FLOAT__* y, __GLOBAL__ __FLOAT__* z \
                              , int nx, int ny, int nz \
                              , __GLOBAL__ __FLOAT__* org, __GLOBAL__ __FLOAT__* mat \
                              , __GLOBAL__ __FLOAT__* material_params \
                              , __COEFFICIENT_FIELDS__ \
			      )
{
    int idx, i, j, k;
    int p, idp0, idp1, sg;
    __FLOAT__ p00, p01, p10, p11, sign, h0, h1, \
              com_x, com_y, com_z, org_x, org_y, org_z, base, top, \
              px, py, pz, px_rot, py_rot, pz_rot;
    org_x = org[0];
    org_y = org[1];
    org_z = org[2];
    com_x = com[0] - org_x;
    com_y = com[1] - org_y;
    com_z = com[2] - org_z;
    if(axis==0)
    {
        base = com_x - height*.5;
        top  = com_x + height*.5;
    }
    else if(axis==1)
    {
        base = com_y - height*.5;
        top  = com_y + height*.5;
    }
    else if(axis==2)
    {
        base = com_z - height*.5;
        top  = com_z + height*.5;
    }
    #pragma omp parallel for \
    shared( base_poly, com, height, axis \
          , x, y, z \
	  , nx, ny, nz \
	  , org, mat \
	  , material_params \
	  , __COEFFICIENTS__ \
	  , org_x, org_y, org_z \
	  , com_x, com_y, com_z \
	  , base, top \
	  ) \
    private( idx, i, j, k \
           , p, idp0, idp1, sg \
	   , p00, p01, p10, p11, sign, h0, h1 \
	   , px, py, pz, px_rot, py_rot, pz_rot \
           ) \
    schedule(guided)
    for(idx=0; idx<nx*ny*nz; idx++)
    {
        i   =  idx /  (ny*nz)      ;
        j   = (idx - i*ny*nz)   /nz;
        k   =  idx - i*ny*nz - j*nz;
        px = x[i]-org_x;
        py = y[j]-org_y;
        pz = z[k]-org_z;
        px_rot = px*mat[0] + py*mat[1] + pz*mat[2];
        py_rot = px*mat[3] + py*mat[4] + pz*mat[5];
        pz_rot = px*mat[6] + py*mat[7] + pz*mat[8];
        sg     = 0;
        if     (axis==0 && px_rot>=base && px_rot<=top)
        {
            h0 = top -   px;
            h1 = px  - base;
            for(p=0; p<np; p++)
            {
                idp0 = ((p  )   )*3;
                idp1 = ((p+1)%np)*3;
                p00  = base_poly[idp0+1] - org_y;
                p01  = base_poly[idp0+2] - org_z;
                p10  = base_poly[idp1+1] - org_y;
                p11  = base_poly[idp1+2] - org_z;
                p00  = (p00*h0+com_y*h1)/(h0+h1);
                p01  = (p01*h0+com_z*h1)/(h0+h1);
                p10  = (p10*h0+com_y*h1)/(h0+h1);
                p11  = (p11*h0+com_z*h1)/(h0+h1);
                if(pz_rot >= return_min(p01,p11) && pz_rot <= return_max(p01,p11) && py_rot <= return_max(p00,p10))
                {
                    sign = (p11-p01)*((pz_rot-p01)*(p10-p00) - (py_rot-p00)*(p11-p01));
                    if(p00 == p10 || sign >= 0) { sg += 1; }
                }
            }
            if(sg%2 == 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(axis==1 && py_rot>=base && py_rot<=top)
        {
            h0 = top -   py;
            h1 = py  - base;
            for(p=0; p<np; p++)
            {
                idp0 = ((p  )   )*3;
                idp1 = ((p+1)%np)*3;
                p00  = base_poly[idp0+2] - org_z;
                p01  = base_poly[idp0  ] - org_x;
                p10  = base_poly[idp1+2] - org_z;
                p11  = base_poly[idp1  ] - org_x;
                p00  = (p00*h0+com_z*h1)/(h0+h1);
                p01  = (p01*h0+com_x*h1)/(h0+h1);
                p10  = (p10*h0+com_z*h1)/(h0+h1);
                p11  = (p11*h0+com_x*h1)/(h0+h1);
                if(px_rot >= return_min(p01,p11) && px_rot <= return_max(p01,p11) && pz_rot <= return_max(p00,p10))
                {
                    sign = (p11-p01)*((px_rot-p01)*(p10-p00) - (pz_rot-p00)*(p11-p01));
                    if(p00 == p10 || sign >= 0) { sg += 1; }
                }
            }
            if(sg%2 == 1)
            {
                __SET_MATERIAL__;
            }
        }
        else if(axis==2 && pz_rot>=base && pz_rot<=top)
        {
            h0 = top -   pz;
            h1 = pz  - base;
            for(p=0; p<np; p++)
            {
                idp0 = ((p  )   )*3;
                idp1 = ((p+1)%np)*3;
                p00  = base_poly[idp0  ] - org_x;
                p01  = base_poly[idp0+1] - org_y;
                p10  = base_poly[idp1  ] - org_x;
                p11  = base_poly[idp1+1] - org_y;
                p00  = (p00*h0+com_x*h1)/(h0+h1);
                p01  = (p01*h0+com_y*h1)/(h0+h1);
                p10  = (p10*h0+com_x*h1)/(h0+h1);
                p11  = (p11*h0+com_y*h1)/(h0+h1);
                if(py_rot >= return_min(p01,p11) && py_rot <= return_max(p01,p11) && px_rot <= return_max(p00,p10))
                {
                    sign = (p11-p01)*((py_rot-p01)*(p10-p00) - (px_rot-p00)*(p11-p01));
                    if(p00 == p10 || sign >= 0) { sg += 1; }
                }
            }
            if(sg%2 == 1)
            {
                __SET_MATERIAL__;
            }
        }
    }
}
