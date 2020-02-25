import numpy as np
import ctypes

Nx = 4
Ny = 5
Nz = 6

test_p = np.zeros((Nx,Ny,Nz), dtype=np.double)
test_c = np.zeros((Nx,Ny,Nz), dtype=np.double)

for i in range(Nx):
	for j in range(Ny):
		for k in range(Nz):
		
			idx = k + j * Nz + i * Nz * Ny
			test_p.flat[idx] = idx

ptr3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='C_CONTIGUOUS')
clib = ctypes.cdll.LoadLibrary("./index_test.so")

clib.index_test.restype = None
clib.index_test.argtypes = [ptr3d, ctypes.c_int, ctypes.c_int, ctypes.c_int]

print(test_c)
clib.index_test(test_c, Nx, Ny,Nz)

print(test_c)
