/* 
Author  : Ki-Hwan Kim
Purpose : OpenCL kernel to update the CPML
Target  : GPU
Created : 2012-01-30
Modified: 
*/

PRAGMA_fp64

__kernel void update(
		int nx, int ny, int nz, int npml, 
		__global DTYPE *f1, __global DTYPE *f2,
		__global DTYPE *f3, __global DTYPE *f4,
		__global DTYPE *psi1, __global DTYPE *psi2,
		__constant DTYPE *pcb, __constant DTYPE *pca ARGS_CF) {
	int idx = get_global_id(0);
	int nmax = NMAX;
	int ic, if1, if2, if3;
	
	if( idx < nmax ) {
		ic = IDX_PC;
		if1 = IDX_F1;
		if2 = IDX_F2;
		if3 = IDX_F3;
				
        psi1[idx] = pcb[ic] * psi1[idx] + pca[ic] * (f3[if2] - f3[if3]);
        psi2[idx] = pcb[ic] * psi2[idx] + pca[ic] * (f4[if2] - f4[if3]);
        f1[if1] -= CF1 * psi1[idx];
        f2[if1] += CF2 * psi2[idx];
	}
}
