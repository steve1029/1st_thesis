import numpy as np
import ctypes

xgrid = 3
ygrid = 4
zgrid = 5

nm = 1e-9
dx = 5 * nm
dy = 5 * nm
dz = 5 * nm

ptr_of1d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
ptr_of2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
ptr_of3d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS')

clib = ctypes.cdll.LoadLibrary("./fftw3d_yz_Z2Z.so")

#------------ set types ------------#
clib.fftw3d_yz_Z2Z.restype = None
clib.fftw3d_yz_Z2Z.argtypes = [ptr_of3d, ptr_of3d, ptr_of3d, ptr_of3d,\
							ctypes.c_int, ctypes.c_int, ctypes.c_int]

clib.ifftw_ik_fftw3d_yz_Z2Z.restype  = None
clib.ifftw_ik_fftw3d_yz_Z2Z.argtypes = [ptr_of3d, ptr_of3d,\
										ptr_of3d, ptr_of3d, ptr_of3d, ptr_of3d, \
										ptr_of1d, ptr_of1d, \
										ctypes.c_int, ctypes.c_int, ctypes.c_int]

#-------- Declare test arrays ---------#

# test arrays for FFT or IFFT test
a       = np.zeros((xgrid,ygrid,zgrid), dtype=np.complex128)
a_npfft = np.zeros((xgrid,ygrid,zgrid), dtype=np.complex128)

a_Re = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)
a_Im = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)

a_Re_cfftw = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)
a_Im_cfftw = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)


# test arrays for ifft_ik_fft test
Ex    = np.zeros((xgrid,ygrid,zgrid), dtype=np.complex128)
Ex_Re = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)
Ex_Im = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)

diffyEx_Re = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)
diffyEx_Im = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)

diffzEx_Re = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)
diffzEx_Im = np.zeros((xgrid,ygrid,zgrid), dtype=np.float64)

diffyEx_C = np.zeros((xgrid,ygrid,zgrid), dtype=np.complex128)
diffzEx_C = np.zeros((xgrid,ygrid,zgrid), dtype=np.complex128)

diffyEx_np = np.zeros((xgrid,ygrid,zgrid), dtype=np.complex128)
diffzEx_np = np.zeros((xgrid,ygrid,zgrid), dtype=np.complex128)

Ex_fft_np = np.zeros((xgrid,ygrid,zgrid), dtype=np.complex128)

ky = np.fft.fftfreq(ygrid, dy) * 2. * np.pi
kz = np.fft.fftfreq(zgrid, dz) * 2. * np.pi

#---------- Initialize test array --------#
for i in range(xgrid):
	for j in range(ygrid):
		for k in range(zgrid):

			#a    [i][j][k] = k + j * zgrid
			#a_Re [i][j][k] = k + j * zgrid
			#Ex   [i][j][k] = k + j * zgrid
			#Ex_Re[i][j][k] = k + j * zgrid

			a    [i][j][k] = np.random.random_sample()
			a_Re [i][j][k] = np.random.random_sample()
			Ex   [i][j][k] = np.random.random_sample()
			Ex_Re[i][j][k] = np.random.random_sample()

#---------- Perform FFT in numpy and FFTW in C ---------#
clib.fftw3d_yz_Z2Z(a_Re, a_Im, a_Re_cfftw, a_Im_cfftw, xgrid,ygrid,zgrid)
a_npfft = np.fft.fftn(a, axes=(1,2))

clib.ifftw_ik_fftw3d_yz_Z2Z(Ex_Re, Ex_Im, \
							diffyEx_Re, diffyEx_Im, diffzEx_Re, diffzEx_Im, \
							ky, kz, xgrid, ygrid, zgrid)

Ex_fft_np = np.fft.fftn(Ex, axes=(1,2))

for i in range(xgrid):
	for j in range(ygrid):
		for k in range(zgrid):
			diffyEx_np[i][j][k] = Ex_fft_np[i][j][k] * 1j *  ky[j]
			diffzEx_np[i][j][k] = Ex_fft_np[i][j][k] * 1j *  kz[k]

diffyEx_np = np.fft.ifftn( diffyEx_np, axes=(1,2))
diffzEx_np = np.fft.ifftn( diffzEx_np, axes=(1,2))

#---------- Get the result ----------#

a_cfftw = np.zeros((xgrid, ygrid, zgrid), dtype=np.complex128)

for i in range(xgrid):
	for j in range(ygrid):
		for k in range(zgrid):

			a_cfftw.real[i,j,k] = a_Re_cfftw[i,j,k]
			a_cfftw.imag[i,j,k] = a_Im_cfftw[i,j,k]
			#a_cfftw.real[i,j,k] = a_npfft.real[i,j,k]
			#a_cfftw.imag[i,j,k] = a_npfft.imag[i,j,k]
			diffyEx_C[i][j][k] = diffyEx_Re[i][j][k] + 1j * diffyEx_Im[i][j][k]
			diffzEx_C[i][j][k] = diffzEx_Re[i][j][k] + 1j * diffzEx_Im[i][j][k]

np.set_printoptions(precision=4, suppress=True)

# Show FFT result
print("FFT result:")
print(a_cfftw)
print(a_npfft)

# Show IFFT_IK_FFT result
print("diffyEx_C result:")
print(diffyEx_C)
print("")
print("diffyEx_np result:")
print(diffyEx_np)
print("")
print("diffzEx_C result:")
print(diffzEx_C)
print("")
print("diffzEx_np result:")
print(diffzEx_np)
print("")

np.set_printoptions(precision=15)
print(diffzEx_C - diffzEx_np)
print(diffyEx_C - diffyEx_np)
