import numpy as np
import ctypes

Nx = 3
Ny = 4
Nz = 5

npclx128_py = np.zeros((Nx,Ny,Nz), dtype=np.complex128)
npclx128_C  = np.zeros((Nx,Ny,Nz), dtype=np.complex128)

for i in range(Nx):
	for j in range(Ny):
		for k in range(Nz):

			myidx = k + j * Nz + i * Nz * Ny

			npclx128_py.flat[myidx] = myidx + myidx * 1j

print(npclx128_py)

clib = ctypes.cdll.LoadLibrary("./npclx.fftwclx.test.so")

ptr1d = np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS')
ptr2d = np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS')
ptr3d = np.ctypeslib.ndpointer(dtype=np.complex128, ndim=3, flags='C_CONTIGUOUS')

clib.npclx_printf.restype = None
clib.npclx_printf.argtypes = [ptr3d, ctypes.c_int]

clib.npclx_printf( npclx128_py, Nx*Ny*Nz)
