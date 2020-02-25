import numpy as np
import scipy as sc
import scipy.constants as sct

c0_SI = sct.physical_constants['speed of light in vacuum'][0]
c0    = 1.

ep0_SI = sct.physical_constants['electric constant'][0]
mu0_SI = sct.physical_constants['magn. constant'][0]
Z0_SI  = sct.physical_constants['characteristic impedance of vacuum'][0]

def to_SI(fdtd, name, value):
    coeffs = {'wavelength'       : fdtd.min_ds, \
              'wave number'      : 1./fdtd.min_ds, \
              'frequency'        : c0_SI/fdtd.min_ds, \
              'angular frequency': c0_SI/fdtd.min_ds, \
              'sigma'            : c0_SI/fdtd.min_ds, \
              'E-field'          : 1., \
              'H-field'          : 1./Z0_SI, \
              'distance'         : fdtd.min_ds, \
              'time'             : fdtd.min_ds/c0_SI}
    return value*coeffs[name]

def to_NU(fdtd, name, value):
    coeffs = {'wavelength'       : fdtd.min_ds, \
              'wave number'      : 1./fdtd.min_ds, \
              'frequency'        : c0_SI/fdtd.min_ds, \
              'angular frequency': c0_SI/fdtd.min_ds, \
              'sigma'            : c0_SI/fdtd.min_ds, \
              'E-field'          : 1., \
              'H-field'          : 1./Z0_SI, \
              'distance'         : fdtd.min_ds, \
              'time'             : fdtd.min_ds/c0_SI}
    return value/coeffs[name]

def to_epr(fdtd, n=1., kappa=None, alpha=None, wavelength=None):
    if alpha is None:
        eps = n**2
        #sgm = 0.
    if kappa is not None:
#        if wavelength is None:
#            raise ValueError, 'Please put the value of wavelength'
#        wfreq_SI = 2.*np.pi*c0_SI/wavelength
        mu  = 1.
        eps = (n**2) + (kappa**2) + 2.j*n*kappa
        #sgm_SI = 2*n*kappa*wfreq_SI
        #sgm = to_NU(fdtd, 'sigma', sgm_SI)
    if alpha is not None:
        if wavelength is None:
            raise ValueError, 'Please put the value of wavelength'
        wfreq_SI = 2.*np.pi*c0_SI/wavelength
        kappa = c0_SI*alpha/(2.*wfreq_SI) # dimensionless quantity

        mu  = 1.
        eps = (n**2) + (kappa**2) + 2.j*n*kappa
        #k0_SI = wfreq_SI/c0_SI
        #sgm_SI = 2*n*kappa*wfreq_SI
        #sgm_SI = k0_SI*sc.sqrt((2*((alpha/2./k0_SI)**2) + n**2)**2 - n**4)
        #sgm = to_NU(fdtd, 'sigma', sgm_SI)
    return eps
#    if sgm is 0.: return eps
#    else: return eps, sgm_SI
