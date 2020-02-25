import numpy as np
import matplotlib.pyplot as plt
import os, datetime, time
from scipy.constants import c, mu_0, epsilon_0
import reikna.cluda as cld
from reikna.fft import FFT, FFTShift

nm = 1e-9
S = 1./4
dx = 1*nm
dy = 1*nm
dt = S*min(dx,dy)/c
nsteps = 2**14 
tsteps = np.arange(nsteps, dtype=np.complex64)
t = tsteps * dt

wavelength = np.arange(400,800,.5) * nm
wlc   = (wavelength[0] + wavelength[-1])/2.
freq  = (c/wavelength)
freqc = (freq[0] + freq[-1])/2

w0 = 2 * np.pi * freqc
ws = 0.3 * w0

ts = 1./ws
tc = 6000.*dt

pulse = (np.exp((-.5)*(((tsteps*dt-tc)*ws)**2)) * np.exp(-1.j*w0*(tsteps*dt-tc))).real
pulse = pulse.astype(np.complex128)

diffpulse = np.exp((-.5)*(((tsteps*dt-tc)*ws)**2)) \
		* (((tsteps*dt-tc)*(ws**2)*np.cos(w0*(tsteps*dt-tc)))-(w0*np.sin(w0*(tsteps*dt-tc))))

fft_cpu = np.fft.fftn(pulse,axes=(0,))
fftfreq = np.fft.fftfreq(nsteps, dt)
#print(fftfreq.dtype.name)
np.savetxt('/root/3D_PSTD/fftfreq.txt', fftfreq, fmt='%3.3e',newline='\r\n', header='fftfreq : \n')

maxfreq = (1./2/dt)
print("max freq : %3.3e " %(maxfreq))
print(maxfreq in fftfreq)
#fft_cpu = np.fft.fftshift(fft_cpu)
#fftfreq = np.fft.fftshift(fftfreq)

iwfft_cpu = fft_cpu * 1j * fftfreq * 2 * np.pi

ifftiwfft_cpu = np.fft.ifftn(iwfft_cpu, axes=(0,))

#####################################################################################
######################################## GPU ########################################
#####################################################################################

api = cld.get_api('cuda')
platforms = api.get_platforms()

dev1 = platforms[0].get_devices()[0]
thr1 = api.Thread(dev1)

dtype = np.complex128

program = thr1.compile("""
KERNEL void MUL(
	GLOBAL_MEM ${ctype} *dest,
	GLOBAL_MEM ${ctype} *a,
	GLOBAL_MEM ${ctype} *b)
{
	SIZE_T i = get_global_id(0);

	dest[i] = ${mul}(a[i],b[i]);
}

KERNEL void ADD(
	GLOBAL_MEM ${ctype} *dest,
	GLOBAL_MEM ${ctype} *a,
	GLOBAL_MEM ${ctype} *b)
{
	SIZE_T i = get_global_id(0);

	dest[i] = ${add}(a[i],b[i]);
}
""",render_kwds=dict( ctype=cld.dtypes.ctype(dtype),
						mul=cld.functions.mul(dtype,dtype,out_dtype=dtype),
						add=cld.functions.add(dtype,dtype,out_dtype=dtype)))

MUL = program.MUL
ADD = program.ADD

iw = 1j * fftfreq * 2. * np.pi

iw_dev = thr1.to_device(iw)
pulse_dev = thr1.to_device(pulse)

pulse_fft_dev = thr1.empty_like(pulse)
pulse_iwfft_dev = thr1.empty_like(pulse)
pulse_ifftiwfft_dev = thr1.empty_like(pulse)

fftusinggpu  = FFT(pulse_dev,axes=(0,))
fftusinggpuc = fftusinggpu.compile(thr1, fast_math=True)

fftusinggpuc(pulse_fft_dev, pulse_dev)

MUL(pulse_iwfft_dev, iw_dev, pulse_fft_dev, local_size=512, global_size=nsteps)

fftusinggpuc(pulse_ifftiwfft_dev, pulse_iwfft_dev, inverse=True)

fft_gpu = pulse_fft_dev.get()
iwfft_gpu = pulse_iwfft_dev.get()
ifftiwfft_gpu = pulse_ifftiwfft_dev.get()

np.savetxt('/root/3D_PSTD/fft_gpu.txt', fft_gpu, fmt='%3.3e',newline='\r\n', header='fftfreq : \n')
np.savetxt('/root/3D_PSTD/fft_cpu.txt', fft_cpu, fmt='%3.3e',newline='\r\n', header='fftfreq : \n')

####################################################################################
####################################### PLOT #######################################
####################################################################################

fig = plt.figure(figsize=(30,9))

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(tsteps.real/1000, pulse.real, label='Pulse')
ax1.grid(True)
ax1.set_xlabel('time, 1000 steps')
ax1.legend()

ax2.plot(tsteps.real/1000, ifftiwfft_cpu.real, label='cpu',color='r',alpha=0.5,linewidth=5.)
ax2.plot(tsteps.real/1000, diffpulse.real, label='diffpulse',color='g')
ax2.plot(tsteps.real/1000, ifftiwfft_gpu.real, label='gpu',color='b')
ax2.set_title("$\\frac{\partial E_x}{\partial t}$")
ax2.set_xlabel("time, 1000 steps")
ax2.grid(True)
ax2.legend()

ax3.plot(tsteps.real/1000, ifftiwfft_gpu.real, label='gpu')
ax3.grid(True)
ax3.legend()

fig.savefig('/root/3D_PSTD/diffEx.png')
plt.close()
print(os.path.basename(__file__))
os.system('/root/dropbox_uploader.sh upload /root/3D_PSTD/%s Python/MyPSTD/3D_PSTD/' %os.path.basename(__file__))
os.system('/root/dropbox_uploader.sh upload /root/3D_PSTD/diffEx.png Python/MyPSTD/3D_PSTD/')
os.system('/root/dropbox_uploader.sh upload /root/3D_PSTD/fftfreq.txt Python/MyPSTD/3D_PSTD/')
os.system('/root/dropbox_uploader.sh upload /root/3D_PSTD/fft_cpu.txt Python/MyPSTD/3D_PSTD/')
os.system('/root/dropbox_uploader.sh upload /root/3D_PSTD/fft_gpu.txt Python/MyPSTD/3D_PSTD/')
