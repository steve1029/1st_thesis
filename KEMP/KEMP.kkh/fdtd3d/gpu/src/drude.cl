/* 
Author  : Ki-Hwan Kim
Purpose : OpenCL kernel to update a Drude-type metal (single pole)
Target  : GPU
Created : 2012-01-30
Modified: 
*/

PRAGMA_fp64

__kernel void update_e(
		int nx, int ny, int nz,
		DTYPE cb, DTYPE cc, DTYPE pca, DTYPE pcb,
		__global DTYPE *ex, __global DTYPE *ey, __global DTYPE *ez,
		__global DTYPE *psix, __global DTYPE *psiy, __global DTYPE *psiz, 
		__global DTYPE *maskx, __global DTYPE *masky, __global DTYPE *maskz) {
	int tid = get_local_id(0);
	int gid = get_global_id(0);
			
	if( gid < NMAX ) {
		int sub_idx = XID*ny*nz + YID*nz + ZID;
        
		__local DTYPE sx[DX], sy[DX], sz[DX];
		__local DTYPE spx[DX], spy[DX], spz[DX];
		
		sx[tid] = ex[sub_idx];
		sy[tid] = ey[sub_idx];
		sz[tid] = ez[sub_idx];
		spx[tid] = psix[gid];
		spy[tid] = psiy[gid];
		spz[tid] = psiz[gid];
			
        ez[sub_idx] = sx[tid] + maskx[gid] * (cb * sx[tid] + cc * spx[tid]);
        ey[sub_idx] = sy[tid] + masky[gid] * (cb * sy[tid] + cc * spy[tid]);
        ez[sub_idx] = sz[tid] + maskz[gid] * (cb * sz[tid] + cc * spz[tid]);
        
		psix[gid] = pca * spx[tid] + pcb * sx[tid];
		psiy[gid] = pca * spy[tid] + pcb * sy[tid];
		psiz[gid] = pca * spz[tid] + pcb * sz[tid];	
	}	
}