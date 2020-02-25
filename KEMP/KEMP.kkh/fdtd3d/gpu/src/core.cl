/* 
Author  : Ki-Hwan Kim
Purpose : OpenCL kernels to update the core of FDTD
Target  : GPU
Created : 2012-01-30
Modified: 
*/

PRAGMA_fp64

__kernel void update_e(int nx, int ny, int nz, 
		__global DTYPE *ex, __global DTYPE *ey, __global DTYPE *ez,
		__global DTYPE *hx, __global DTYPE *hy, __global DTYPE *hz ARGS_CE ARGS_RD) {
	int tx = get_local_id(0);
	int idx = get_global_id(0);
	int i, j, k;
	
	__local DTYPE sx[DX+1], sy[DX+1], sz[DX];

	if( idx < (nx*ny-1)*nz ) {
		sx[tx] = hx[idx];
		sy[tx] = hy[idx];
		sz[tx] = hz[idx];
		if( tx == DX-1 ) {
			sx[tx+1] = hx[idx+1];
			sy[tx+1] = hy[idx+1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		i = idx/(ny*nz);
		j = (idx - i*ny*nz)/nz;
		k = idx%nz;

		if( j<ny-1 && k<nz-1 ) ex[idx] += CEX * (RDY (hz[idx+nz] - sz[tx]) - RDZ (sy[tx+1] - sy[tx]));
		if( i<nx-1 && k<nz-1 ) ey[idx] += CEY * (RDZ (sx[tx+1] - sx[tx]) - RDX (hz[idx+ny*nz] - sz[tx]));
		if( i<nx-1 && j<ny-1 ) ez[idx] += CEZ * (RDX (hy[idx+ny*nz] - sy[tx]) - RDY (hx[idx+nz] - sx[tx]));
	}
}



__kernel void update_h(int nx, int ny, int nz, 
		__global DTYPE *ex, __global DTYPE *ey, __global DTYPE *ez, 
		__global DTYPE *hx, __global DTYPE *hy, __global DTYPE *hz ARGS_CH ARGS_RD) {
	int tx = get_local_id(0);
	int idx = get_global_id(0) + nz;
	int i, j, k;

	__local DTYPE s[3*DX+2];
	__local DTYPE *sx, *sy, *sz;
	sz = s;
	sy = &sz[DX+1];
	sx = &sy[DX+1];

	if( idx < nx*ny*nz ) {
		sx[tx] = ex[idx];
		sy[tx] = ey[idx];
		sz[tx] = ez[idx];
		if( tx == 0 ) {
			sx[tx-1] = ex[idx-1];
			sy[tx-1] = ey[idx-1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		i = idx/(ny*nz);
		j = (idx - i*ny*nz)/nz;
		k = idx%nz;

		if( j>0 && k>0 ) hx[idx] -= CHX * (RDY (sz[tx] - ez[idx-nz]) - RDZ (sy[tx] - sy[tx-1]));
		if( i>0 && k>0 ) hy[idx] -= CHY * (RDZ (sx[tx] - sx[tx-1]) - RDX (sz[tx] - ez[idx-ny*nz]));
		if( i>0 && j>0 ) hz[idx] -= CHZ * (RDX (sy[tx] - ey[idx-ny*nz]) - RDY (sx[tx] - ex[idx-nz]));
	}
}
