import numpy as np
import ctypes

"""
Author: Donggun Lee
Data: Thursday, 18-01-11

Description
-------------------------------------
Practice of how to calculate 3d arrays
by using ctypes modules.
-------------------------------------

Learnings 1.

	Question:
		We send 3D numpy array to C function with data type: '1-level pointer'.
		This is obviously wrong. But why does it works?

	Answer:
		N-dimensional numpy array is actually treated by interpreter as 1D array.
		Therefore, C-functions in shared object file must be written with 1-level pointer.

Learnings 2.
	
	Question:
		Then why we define ndim? 

	Answer:
		As stated before, numpy 3D array object is treated as '1D' by numpy and C. 
		But that does not mean numpy cannot distinguish 3D numpy array from 1D numpy array.
		Numpy treats them differently.
		So we have to tell to ctypes that np.ctypeslib.ndpointer(which is 1-level pointer)
		will be used by 3D numpy array. That is a argument 'ndim' does in np.ctypes.ndpointer.

Learnings 3.
	
	Question:
		What flags does?

	Answer:
		flags is an 'Additional explanation' of numpy array object who will use that ndpointer.
		For example, let's say that

			a = np.arange(100)
			b = a[::5]

		'a' occupies real memory space. However, 'b' does not. 'b' refers to 'a'.
		Moreover, memory space of 'a' is continuous, but 'b' refers to 'a' by skipping 5 spaces.
		In this case, we say that 'b' is not contiguous. Then we must tell to ctypes that
		'b' is not contiguous. That is what 'flags' do.

"""

xgrid = 3
ygrid = 4
zgrid = 5

dtype = np.double

ptr_of1d = np.ctypeslib.ndpointer(dtype=dtype, ndim=1, flags='C_CONTIGUOUS')
ptr_of2d = np.ctypeslib.ndpointer(dtype=dtype, ndim=2, flags='C_CONTIGUOUS')
ptr_of3d = np.ctypeslib.ndpointer(dtype=dtype, ndim=3, flags='C_CONTIGUOUS')

clib = ctypes.cdll.LoadLibrary("./math3d_D2D.so")

clib.add3d_double.restype  = None # set return type
clib.mul3d_double.restype  = None # set return type
clib.sub3d_double.restype  = None # set return type
clib.div3d_double.restype  = None # set return type

clib.add3d_double.argtypes = [ptr_of3d, ptr_of3d, ptr_of3d, ctypes.c_int, ctypes.c_int, ctypes.c_int]
clib.mul3d_double.argtypes = [ptr_of3d, ptr_of3d, ptr_of3d, ctypes.c_int, ctypes.c_int, ctypes.c_int]
clib.sub3d_double.argtypes = [ptr_of3d, ptr_of3d, ptr_of3d, ctypes.c_int, ctypes.c_int, ctypes.c_int]
clib.div3d_double.argtypes = [ptr_of3d, ptr_of3d, ptr_of3d, ctypes.c_int, ctypes.c_int, ctypes.c_int]

a = np.ones ((xgrid,ygrid,zgrid), dtype=dtype) * 2
b = np.ones ((xgrid,ygrid,zgrid), dtype=dtype) * 3

add_result = np.zeros((xgrid,ygrid,zgrid), dtype=dtype)
sub_result = np.zeros((xgrid,ygrid,zgrid), dtype=dtype)
mul_result = np.zeros((xgrid,ygrid,zgrid), dtype=dtype)
div_result = np.zeros((xgrid,ygrid,zgrid), dtype=dtype)

clib.add3d_double(a, b, add_result, xgrid, ygrid, zgrid)
clib.mul3d_double(a, b, mul_result, xgrid, ygrid, zgrid)
clib.sub3d_double(a, b, sub_result, xgrid, ygrid, zgrid)
clib.div3d_double(a, b, div_result, xgrid, ygrid, zgrid)

print(add_result)
print(mul_result)
print(sub_result)
print(div_result)
