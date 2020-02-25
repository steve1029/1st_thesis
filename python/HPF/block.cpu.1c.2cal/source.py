import numpy as np
from scipy.constants import c, mu_0, epsilon_0

class Gaussian(object):
	
	def __init__(self,Space):
		
		IEp   = Space.gridx
		JEp   = Space.gridy
		KEp   = Space.gridz
		self.dt    = Space.dt
		self.dtype = Space.dtype

		self.set_wave = False
		self.set_freq = False

	@property
	def freq(self): return self._freq

	@freq.setter
	def freq(self,freq_property):

		assert self.set_wave == False, "wavelength is already set"
		assert self.set_freq == False, "frequency is already set"

		start,end,interval,spread = freq_property

		self._freq  = np.arange(start,end,interval,dtype=self.dtype) 
		self._omega = self._freq * 2. * np.pi
		self._wvlen = c/self.freq
		self.freqc  = (self._freq[0]  + self._freq[-1] )/2
		self.wvlenc = (self._wvlen[0] + self._wvlen[-1])/2
		self.spread = spread
	
		self.set_wave = True
		self.set_freq = True

	@property
	def wvlen(self) : return self._wvlen

	@wvlen.setter
	def wvlen(self,wave_property):

		assert self.set_wave == False, "wavelength is already set"
		assert self.set_freq == False, "frequency is already set"

		start,end,interval,spread = wave_property

		self._wvlen  = np.arange(start,end,interval,dtype=self.dtype)
		self._freq   = c/self._wvlen 
		self._omega = self._freq * 2. * np.pi
		self.freqc  = (self._freq[0]  + self._freq[-1] )/2
		self.wvlenc = (self._wvlen[0] + self._wvlen[-1])/2
		self.spread = spread

		self.set_wave = True
		self.set_freq = True

	def omega(self) : return self._omega

	def pulse(self,step,pick_pos):
		
		assert self.set_wave == True, "You should define Gaussian.wvlen or Gaussian.freq."
		assert self.set_freq == True, "You should define Gaussian.wvlen or Gaussian.freq."
		
		w0 = 2 * np.pi * self.freqc
		ws = self.spread * w0
		ts = 1./ws
		tc = pick_pos * self.dt	

		pulse = np.exp((-.5) * (((step*self.dt-tc)*ws)**2)) * np.cos(w0*(step*self.dt-tc))

		return pulse
