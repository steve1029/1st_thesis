import ctypes, datetime, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

Nx, Ny, Nz = 2, 8, 8

dy = 0.1
dz = 0.1

assert Ny % 2 == 0
assert Nz % 2 == 0

E = np.random.rand(Nx,Ny,Nz)

# Copy.
diffzE_npft = np.copy(E)
diffzET_npft = np.copy(E).transpose((0,2,1))
diffyE_npft = np.copy(E)
diffzE_fftw = np.zeros_like(E, dtype=np.double)
diffyE_fftw = np.zeros_like(E, dtype=np.double)

# Get derivatives using rFFT in numpy.
ky = np.fft.rfftfreq(Ny, dy) * 2 * np.pi
kz = np.fft.rfftfreq(Nz, dz) * 2 * np.pi

diffzE_npft  = np.fft.rfftn(diffzE_npft, axes=(2,))
diffyE_npft  = np.fft.rfftn(diffyE_npft, axes=(1,))
diffzET_npft = np.fft.rfftn(diffzET_npft, axes=(2,))

for i in range(Nx):
	for j in range(Ny):
		for k in range(int(Nz/2+1)):

			idx = int(k + j*(Nz/2+1) + i*(Nz/2+1)*Ny)
			diffzE_npft.flat[idx] = diffzE_npft.flat[idx] * 1j * kz[k]

for i in range(Nx):
	for k in range(Nz):
		for j in range(int(Ny/2+1)):

			idx = int(j + k*(Ny/2+1) + i*(Ny/2+1)*Nz)

			diffzET_npft.flat[idx] = diffzET_npft.flat[idx] * 1j * ky[j]

nax = np.newaxis
diffyE_npft = diffyE_npft * ky[nax,:,nax] * 1j

diffzE_npft = np.fft.irfftn(diffzE_npft, axes=(2,))
diffzET_npft = np.fft.irfftn(diffzET_npft, axes=(2,))
diffyE_npft = np.fft.irfftn(diffyE_npft, axes=(1,))

diffzET_npft = diffzET_npft.transpose((0,2,1))

# Get derivatives using FFTW3 library.
ptr1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')
ptr2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS')
ptr3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='C_CONTIGUOUS')

clib = ctypes.cdll.LoadLibrary("./rfftw3.so")

clib.rfft_1d_of_3d.restype = None
clib.rfft_1d_of_3d.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr1d, ptr1d, ptr3d, ptr3d, ptr3d]
clib.rfft_1d_of_3d(Nx, Ny, Nz, ky, kz, E, diffyE_fftw, diffzE_fftw)

#clib.rfft_1d.restype = None
#clib.rfft_1d.argtypes= [ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr1d, ptr3d, ptr3d]
#clib.rfft_1d(Nx, Ny, Nz, kz, E, diffzE_fftw)

# Comparison.
#np.set_printoptions(precision=8)
print(diffyE_npft - diffzET_npft)
print(diffyE_npft - diffyE_fftw)
print(diffzE_npft - diffzE_fftw)
#print(E - diffyE_fftw)
#print(E - diffzE_fftw)
#print(E)
#print(diffzE_fftw)
sys.exit()

# Plot.
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)

ax4 = fig.add_subplot(2,3,4)
ax5 = fig.add_subplot(2,3,5)
ax6 = fig.add_subplot(2,3,6)

ax1.plot(x, E[:,0,0], label='along x')
ax2.plot(y, E[0,:,0], label='along y')
ax3.plot(z, E[0,0,:], label='along z')

#ax4.plot(x, E[:,0,0], label='along x')
ax5.plot(y, diffyEC[0,:,0], label='fftw3 df/dy')
ax5.plot(y, diffyE [0,:,0], label='numpy df/dy')
ax6.plot(z, diffzEC[0,0,:], label='fftw3 df/dz')
ax6.plot(z, diffzE [0,0,:], label='numpy df/dz')

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
ax5.grid(True)
ax6.grid(True)

ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')
ax4.legend(loc='best')
ax5.legend(loc='best')
ax6.legend(loc='best')

fig.tight_layout()
fig.savefig("./plot.png")
