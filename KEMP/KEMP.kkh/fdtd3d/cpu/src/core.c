/* 
Author  : Ki-Hwan Kim
Purpose : C functions to update the core of FDTD
Target  : CPU
Created : 2012-02-13
Modified: 
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>


static PyObject *update_e(PyObject *self, PyObject *args) {
	int nx, ny, nz;
	DTYPE *ex, *ey, *ez, *hx, *hy, *hz;
	PyArrayObject *Ex, *Ey, *Ez, *Hx, *Hy, *Hz;
	COEFF_E DTYPE *cx, *cy, *cz;
	COEFF_E PyArrayObject *Cx, *Cy, *Cz;
	
	if(!PyArg_ParseTuple(args, "PARSE_ARGS_CE", &Ex, &Ey, &Ez, &Hx, &Hy, &Hz PYARRAYOBJECT_CE)) return NULL;
	nx = (int)(Ex->dimensions)[0];
	ny = (int)(Ex->dimensions)[1];
	nz = (int)(Ex->dimensions)[2];
	ex = (DTYPE*)(Ex->data);
	ey = (DTYPE*)(Ey->data);
	ez = (DTYPE*)(Ez->data);
	hx = (DTYPE*)(Hx->data);
	hy = (DTYPE*)(Hy->data);
	hz = (DTYPE*)(Hz->data);
	COEFF_E cx = (DTYPE*)(Cx->data);
	COEFF_E cy = (DTYPE*)(Cy->data);
	COEFF_E cz = (DTYPE*)(Cz->data);
	
	int idx, i, j, k;
	int nyz = ny * nz;
	SET_NUM_THREADS
    OMP Py_BEGIN_ALLOW_THREADS
    OMP #pragma omp parallel for private(idx, i, j, k)
    for(i=0; i<nx; i++) {
	    for(j=0; j<ny-1; j++) {
		    for(k=0; k<nz-1; k++) {
				idx = i*nyz + j*nz + k;
				ex[idx] += CEX * ((hz[idx+nz] - hz[idx]) - (hy[idx+1] - hy[idx]));
			}
		}
	}
	
    OMP #pragma omp parallel for private(idx, i, j, k)
    for(i=0; i<nx-1; i++) {
	    for(j=0; j<ny; j++) {
		    for(k=0; k<nz-1; k++) {
				idx = i*nyz + j*nz + k;
				ey[idx] += CEY * ((hx[idx+1] - hx[idx]) - (hz[idx+nyz] - hz[idx]));
			}
		}
	}
	
    OMP #pragma omp parallel for private(idx, i, j, k)
    for(i=0; i<nx-1; i++) {
	    for(j=0; j<ny-1; j++) {
		    for(k=0; k<nz; k++) {
				idx = i*nyz + j*nz + k;
				ez[idx] += CEZ * ((hy[idx+nyz] - hy[idx]) - (hx[idx+nz] - hx[idx]));
			}
		}
	}
	OMP Py_END_ALLOW_THREADS
	
	Py_INCREF(Py_None);
	return Py_None;
}



static PyObject *update_h(PyObject *self, PyObject *args) {
	int nx, ny, nz;
	DTYPE *ex, *ey, *ez, *hx, *hy, *hz;
	PyArrayObject *Ex, *Ey, *Ez, *Hx, *Hy, *Hz;
	COEFF_H DTYPE *cx, *cy, *cz;
	COEFF_H PyArrayObject *Cx, *Hy, *Cz;
	
	if(!PyArg_ParseTuple(args, "PARSE_ARGS_CH", &Ex, &Ey, &Ez, &Hx, &Hy, &Hz PYARRAYOBJECT_CH)) return NULL;
	nx = (int)(Ex->dimensions)[0];
	ny = (int)(Ex->dimensions)[1];
	nz = (int)(Ex->dimensions)[2];
	ex = (DTYPE*)(Ex->data);
	ey = (DTYPE*)(Ey->data);
	ez = (DTYPE*)(Ez->data);
	hx = (DTYPE*)(Hx->data);
	hy = (DTYPE*)(Hy->data);
	hz = (DTYPE*)(Hz->data);
	COEFF_H cx = (DTYPE*)(Cx->data);
	COEFF_H cy = (DTYPE*)(Cy->data);
	COEFF_H cz = (DTYPE*)(Cz->data);
	
	int idx, i, j, k;
	int nyz = ny * nz;

	SET_NUM_THREADS
    OMP Py_BEGIN_ALLOW_THREADS
    OMP #pragma omp parallel for private(idx, i, j, k)
    for(i=0; i<nx; i++) {
	    for(j=1; j<ny; j++) {
		    for(k=1; k<nz; k++) {
				idx = i*nyz + j*nz + k;
				hx[idx] -= CHX * ((ez[idx] - ez[idx-nz]) - (ey[idx] - ey[idx-1]));
			}
		}
	}
	
    OMP #pragma omp parallel for private(idx, i, j, k)
    for(i=1; i<nx; i++) {
	    for(j=0; j<ny; j++) {
		    for(k=1; k<nz; k++) {
				idx = i*nyz + j*nz + k;
				hy[idx] -= CHY * ((ex[idx] - ex[idx-1]) - (ez[idx] - ez[idx-nyz]));
			}
		}
	}
	
    OMP #pragma omp parallel for private(idx, i, j, k) 
    for(i=1; i<nx; i++) {
	    for(j=1; j<ny; j++) {
		    for(k=0; k<nz; k++) {
				idx = i*nyz + j*nz + k;
				hz[idx] -= CHZ * ((ey[idx] - ey[idx-nyz]) - (ex[idx] - ex[idx-nz]));
			}
		}
	}
	OMP Py_END_ALLOW_THREADS

	Py_INCREF(Py_None);
	return Py_None;
}



static PyMethodDef ufunc_methods[] = {
	{"update_e", update_e, METH_VARARGS, ""},
	{"update_h", update_h, METH_VARARGS, ""},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcore() {
	Py_InitModule("core", ufunc_methods);
	import_array();
}