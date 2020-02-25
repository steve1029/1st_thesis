/* 
Author  : Ki-Hwan Kim
Purpose : C function to update the CPML
Target  : CPU
Created : 2012-02-13
Modified: 
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>


static PyObject *update(PyObject *self, PyObject *args) {
	int nx, ny, nz, npml;
	DTYPE *f1, *f2, *f3, *f4, *psi1, *psi2, *pcb, *pca;
	PyArrayObject *F1, *F2, *F3, *F4, *PSI1, *PSI2, *PCB, *PCA;
	COEFF DTYPE *c1, *c2;
	COEFF PyArrayObject *C1, *C2;
	
	if(!PyArg_ParseTuple(args, "PARSE_ARGS_C", &npml, &F1, &F2, &F3, &F4, &PSI1, &PSI2, &PCB, &PCA PYARRAYOBJECT_C)) return NULL;
	nx = (int)(F1->dimensions)[0];
	ny = (int)(F1->dimensions)[1];
	nz = (int)(F1->dimensions)[2];
	f1 = (DTYPE*)(F1->data);
	f2 = (DTYPE*)(F2->data);
	f3 = (DTYPE*)(F3->data);
	f4 = (DTYPE*)(F4->data);
	psi1 = (DTYPE*)(PSI1->data);
	psi2 = (DTYPE*)(PSI2->data);
	pcb = (DTYPE*)(PCB->data);
	pca = (DTYPE*)(PCA->data);
	COEFF c1 = (DTYPE*)(C1->data);
	COEFF c2 = (DTYPE*)(C2->data);
	
	int idx, ic, if1, if2, if3;
	SET_NUM_THREADS
    OMP Py_BEGIN_ALLOW_THREADS
    OMP #pragma omp parallel for private(idx, ic, if1, if2, if3)
    for(idx=0; idx<NMAX; idx++) {
		ic = IDX_PC;
		if1 = IDX_F1;
		if2 = IDX_F2;
		if3 = IDX_F3;
				
		psi1[idx] = pcb[ic] * psi1[idx] + pca[ic] * (f3[if2] - f3[if3]);
		psi2[idx] = pcb[ic] * psi2[idx] + pca[ic] * (f4[if2] - f4[if3]);
		f1[if1] -= CF1 * psi1[idx];
		f2[if1] += CF2 * psi2[idx];
		//printf("pca[ic] = %g\n", pca[ic]);
	}
	OMP Py_END_ALLOW_THREADS
	
	Py_INCREF(Py_None);
	return Py_None;
}



static PyMethodDef ufunc_methods[] = {
	{"update", update, METH_VARARGS, ""},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcpml() {
	Py_InitModule("cpml", ufunc_methods);
	import_array();
}