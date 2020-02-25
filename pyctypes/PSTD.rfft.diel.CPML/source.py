import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0

class Gaussian(object):
	
	def __init__(self,dt,dtype):
		
		self.dt    = dt
		self.dtype = dtype

		self.set_wave = False
		self.set_freq = False

	def freq(self,freq_property):

		assert self.set_wave == False, "wavelength is already set"
		assert self.set_freq == False, "frequency is already set"

		start,end,interval,spread = freq_property

		self.freq   = np.arange(start, end, interval, dtype=self.dtype) 
		self.omega  = self.freq * 2. * np.pi
		self.wvlen  = c/self.freq[::-1]
		self.freqc  = (self.freq [0] + self.freq [-1])/2
		self.wvlenc = (self.wvlen[0] + self.wvlen[-1])/2
		self.spread = spread
	
		self.set_wave = True
		self.set_freq = True

	def wvlen(self,wave_property):

		assert self.set_wave == False, "wavelength is already set"
		assert self.set_freq == False, "frequency is already set"

		start,end,interval,spread = wave_property

		self.wvlen  = np.arange(start,end,interval,dtype=self.dtype)
		self.freq   = c/self.wvlen[::-1] 
		self.omega  = self.freq * 2. * np.pi
		self.freqc  = (self.freq [0] + self.freq [-1])/2
		self.wvlenc = (self.wvlen[0] + self.wvlen[-1])/2
		self.spread = spread

		self.set_wave = True
		self.set_freq = True

	def omega(self) : 

		assert self.set_wave == True, "You should define Gaussian.wvlen or Gaussian.freq first."
		assert self.set_freq == True, "You should define Gaussian.wvlen or Gaussian.freq first."
		
		return self._omega

	def pulse_re(self,step,pick_pos):
		
		assert self.set_wave == True, "You should define Gaussian.wvlen or Gaussian.freq first."
		assert self.set_freq == True, "You should define Gaussian.wvlen or Gaussian.freq first."
		
		self.pick_pos = pick_pos
		w0 = 2 * np.pi * self.freqc
		ws = self.spread * w0
		ts = 1./ws
		tc = self.pick_pos * self.dt	

		pulse = np.exp((-.5) * (((step*self.dt-tc)*ws)**2)) * np.cos(w0*(step*self.dt-tc))

		return pulse

	def pulse_im(self,step,pick_pos):
		
		assert self.set_wave == True, "You should define Gaussian.wvlen or Gaussian.freq first."
		assert self.set_freq == True, "You should define Gaussian.wvlen or Gaussian.freq first."
		
		self.pick_pos = pick_pos
		w0 = 2 * np.pi * self.freqc
		ws = self.spread * w0
		ts = 1./ws
		tc = self.pick_pos * self.dt	

		pulse = np.exp((-.5) * (((step*self.dt-tc)*ws)**2)) * np.sin(w0*(step*self.dt-tc))

		return pulse

	def plot_pulse(self, tsteps, pick_pos,savedir):
		
		self.pick_pos = pick_pos
		w0 = 2 * np.pi * self.freqc
		ws = self.spread * w0
		ts = 1./ws
		tc = self.pick_pos * self.dt	
		nax = np.newaxis

		time_domain = np.arange(tsteps, dtype=self.dtype)
		t = time_domain * self.dt

		pulse_re = np.exp((-.5) * (((t-tc)*ws)**2)) * np.cos(w0*(t-tc))
		pulse_im = np.exp((-.5) * (((t-tc)*ws)**2)) * np.sin(w0*(t-tc))

		pulse_re_ft = (self.dt * pulse_re[nax,:]* np.exp(1j*2*np.pi*self.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2*np.pi)
		pulse_im_ft = (self.dt * pulse_im[nax,:]* np.exp(1j*2*np.pi*self.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2*np.pi)

		pulse_re_ft_amp = abs(pulse_re_ft)**2
		pulse_im_ft_amp = abs(pulse_im_ft)**2

		fig = plt.figure(figsize=(15,7))

		ax1 = fig.add_subplot(1,2,1)
		ax2 = fig.add_subplot(1,2,2)

		ax1.plot(time_domain, pulse_re, color='b', label='real')
		ax1.plot(time_domain, pulse_im, color='r', label='imag', linewidth='1.5', alpha=0.5)

		ax2.plot(self.freq/10**12, pulse_re_ft_amp, color='b', label='real')
		ax2.plot(self.freq/10**12, pulse_im_ft_amp, color='r', label='imag', linewidth='1.5', alpha=0.5)

		ax1.set_xlabel('time step')
		ax1.set_ylabel('Amp')
		ax1.legend(loc='best')
		ax1.grid(True)

		ax2.set_xlabel('freq(THz)')
		ax2.set_ylabel('Amp')
		ax2.legend(loc='best')
		ax2.grid(True)

		fig.savefig(savedir+"src_input.png")

		return None


class Sine(object):

	def __init__(self, dt, dtype):

		self.dt = dt
		self.dtype = dtype

	def set_freq(self, freq):
		
		self.freq = freq
		self.wvlen = c / self.freq
		self.omega = 2 * np.pi * self.freq
		self.wvector = 2 * np.pi / self.wvlen

	def set_wvlen(self, wvlen):

		self.wvlen = wvlen
		self.freq = c / self.wvlen
		self.omega = 2 * np.pi * self.freq
		self.wvector = 2 * np.pi / self.wvlen

	def pulse_re(self, tstep, pick_pos):

		pulse_re = np.sin(self.omega * tstep * self.dt)

		return pulse_re

	def pulse_im(self, tstep, pick_pos):

		pulse_im = np.cos(self.omega * tstep * self.dt)

		return pulse_im
