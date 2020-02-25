import numpy as np
import matplotlib.pyplot as plt
import ctypes

dt = 0.005
t = np.arange(0,1,dt)

freq1 = 10.	# Hz.
freq2 = 20.
freq3 = 30.

wave1 = np.sin( 2 * np.pi * freq1 * t)
wave2 = np.sin( 2 * np.pi * freq2 * t)
wave3 = np.sin( 2 * np.pi * freq3 * t)
packet = wave1 + wave2 + wave3
imag = np.zeros(len(t), dtype=float)

np_vs_fftw = plt.figure(figsize=(21,9))

# axis 1 depicts sine wave in time domain.
# axis 2 shows the result of numpy.fft of axis1.
# axis 3 shows the derivatives of axis1 deduced by the numpy.fft
# axis 4 depicts no plots.
# axis 5 shows the result of fftw of axis1.
ax1 = np_vs_fftw.add_subplot(2,3,1)
ax2 = np_vs_fftw.add_subplot(2,3,2)
ax3 = np_vs_fftw.add_subplot(2,3,3)
ax4 = np_vs_fftw.add_subplot(2,3,4)
ax5 = np_vs_fftw.add_subplot(2,3,5)
ax6 = np_vs_fftw.add_subplot(2,3,6)

#-----------------------------------------------------#
#--------------------- Plot axis 1 -------------------#
#-----------------------------------------------------#
#ax1.plot(t, wave1, label="%.1f Hz" %(freq1))
#ax1.plot(t, wave2, label="%.1f Hz" %(freq2))
#ax1.plot(t, wave3, label="%.1f Hz" %(freq3))
ax1.plot(t, packet, label="Packet")
ax1.set_title("sine wave in time domain")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("Amp")
ax1.grid(True)
ax1.legend(loc='best')

FT_wave1 = np.fft.fftn(wave1, axes=(0,))
FT_wave2 = np.fft.fftn(wave2, axes=(0,))
FT_wave3 = np.fft.fftn(wave3, axes=(0,))
FT_packet = np.fft.fftn(packet, axes=(0,))
FT_freq = np.fft.fftfreq(len(t), dt)

#--------------------------------------------------------#
#--------------------- Plot axis 2 ----------------------#
#--------------------------------------------------------#
#ax2.plot(FT_freq, abs(FT_wave1), label="fft of wave1")
#ax2.plot(FT_freq, FT_wave2, label="fft of wave2")
#ax2.plot(FT_freq, FT_wave3, label="fft of wave3")
ax2.plot(FT_freq, abs(FT_packet), label="fft of packet")
ax2.grid(True)
ax2.legend(loc='best')

#--------------------------------------------------------#
#--------------------- Plot axis 3 ----------------------#
#--------------------------------------------------------#
omega = 2 * np.pi * FT_freq
difft1 = np.fft.ifftn( 1j * omega * FT_wave1, axes=(0,))
#difft11 = np.fft.ifftn( -omega * FT_wave1.imag + 1j * FT_wave1.real * omega, axes=(0,))
difftp = np.fft.ifftn( 1j * omega * FT_packet, axes=(0,))

#ax3.plot(t, difft1.real, label="difft of wave1, Re")
#ax3.plot(t, difft11.real, label="difft of wave11, Re")
#ax3.plot(t, difft1.imag, label="difft of wave1, Im")
#ax3.plot(t, difft11.imag, label="difft of wave11, Im")
ax3.plot(t, difftp.real, label="difft of packet, Re")
ax3.plot(t, difftp.imag, label="difft of packet, Im")
ax3.grid(True)
ax3.legend(loc='best')

#--------------------------------------------------------#
#--------------------- Plot axis 5 ----------------------#
#--------------------------------------------------------#
fftw_re  = np.zeros(len(t), dtype=float)
fftw_im  = np.zeros(len(t), dtype=float)
deriv_re = np.zeros(len(t), dtype=float)
deriv_im = np.zeros(len(t), dtype=float)

ptr1d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
clib = ctypes.cdll.LoadLibrary("./fftw_1d.so")
clib.DERIV_1D.restype  = None
clib.DERIV_1D.argtypes = [ptr1d, ptr1d, ptr1d, ptr1d, ptr1d, ptr1d, ptr1d, ctypes.c_int]
clib.DERIV_1D(packet, imag, fftw_re, fftw_im, deriv_re, deriv_im, omega, len(t))

ax5.plot(FT_freq, np.sqrt(fftw_re**2 + fftw_im**2), label='fftw3')
ax5.grid(True)
ax5.legend(loc='best')
#--------------------------------------------------------#
#--------------------- Plot axis 6 ----------------------#
#--------------------------------------------------------#

ax6.plot(t, deriv_re/len(t), label='deriv by fftw3, Re')
ax6.plot(t, deriv_im/len(t), label='deriv by fftw3, Im')
ax6.grid(True)
ax6.legend(loc='best')

#--------------------------------------------------------#
np_vs_fftw.savefig("./np_vs_fftw.png")
