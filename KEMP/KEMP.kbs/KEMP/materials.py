import numpy as np
import units as unit
import scipy.constants as sct

c0_SI = sct.physical_constants['speed of light in vacuum'][0]

def nk_to_eps(n, k):
    comp_n   = n + 1.j*k
    comp_eps = comp_n**2
    return comp_eps

def set_conductivity():
    pass

class Dielectric:
    def __init__(self, epr, wavelength=None):
        self.classification = 'dielectric'
        epr_re = epr.real
        epr_im = epr.imag
        if isinstance(epr, np.ndarray):
            if wavelength is not None:
                wfreq_SI = 2.*np.pi*c0_SI/wavelength
            else:
                if abs(epr_im).sum() != 0.:
                    print 'Warning: Complex permittivity needs wavelength parameter'
                wfreq_SI = 0.
            if   epr.size == 2:
                self.epr_x = epr_re[0]
                self.epr_y = epr_re[1]
                self.epr_z = epr_re[1]
                self.sgm_x = epr_im[0]*wfreq_SI
                self.sgm_y = epr_im[1]*wfreq_SI
                self.sgm_z = epr_im[1]*wfreq_SI
            elif epr.size == 3:
                self.epr_x = epr_re[0]
                self.epr_y = epr_re[1]
                self.epr_z = epr_re[2]
                self.sgm_x = epr_im[0]*wfreq_SI
                self.sgm_y = epr_im[1]*wfreq_SI
                self.sgm_z = epr_im[2]*wfreq_SI
            else:
                raise ValueError, 'tensor epsilon must be size of 2 or 3'
        else:
            if wavelength is not None:
                wfreq_SI = 2.*np.pi*c0_SI/wavelength
            else:
                if epr_im != 0.:
                    print 'Warning: Complex permittivity needs wavelength parameter'
                wfreq_SI = 0.
            self.epr_x = epr_re
            self.epr_y = epr_re
            self.epr_z = epr_re
            self.sgm_x = epr_im*wfreq_SI
            self.sgm_y = epr_im*wfreq_SI
            self.sgm_z = epr_im*wfreq_SI

        '''
        if n is not None:
            if epr is not None:
                raise ValueError, 'Please DO NOT SET both epsilon and refractive index!'
            else:
                if k != 0.:
                    if wavelength is None:
                        raise ValueError, 'Please SET wavelength'
                    wfreq_SI = 2.*np.pi*c0_SI/wavelength
                    self.epr = n**2 - k**2
                    self.sigma = 2*n*k*wfreq_SI
                else:
                    self.epr = n**2
                    self.sigma = 0.
        else:
            if epr is None:
                raise ValueError, 'Please SET epsilon or refractive index!'
            else:
                if isinstance(epr, np.ndarray):
                    if epr.size == 2:
                        self.epr_x = epr[0]
                        self.epr_y = epr[1]
                        self.epr_z = 1.
                    elif epr.size == 3:
                        self.epr_x = epr[0]
                        self.epr_y = epr[1]
                        self.epr_z = epr[2]
                    else:
                        raise ValueError, 'tensor epsilon must be size of 2 or 3'
                else:
                    self.epr_x = epr
                    self.epr_y = epr
                    self.epr_z = epr
        '''

    def set_coeff(self, fdtd):
        dt  = fdtd.dt
        epr_x = self.epr_x
        epr_y = self.epr_y
        epr_z = self.epr_z
        mur_x = 1.
        mur_y = 1.
        mur_z = 1.
        sgm_epr_x = unit.to_NU(fdtd, 'sigma', self.sgm_x)
        sgm_epr_y = unit.to_NU(fdtd, 'sigma', self.sgm_y)
        sgm_epr_z = unit.to_NU(fdtd, 'sigma', self.sgm_z)
        sgm_mur_x = 0.
        sgm_mur_y = 0.
        sgm_mur_z = 0.

        self.ce1x = (2. - sgm_epr_x*dt/epr_x)/(2. + sgm_epr_x*dt/epr_x)
        self.ce2x = (2. *           dt/epr_x)/(2. + sgm_epr_x*dt/epr_x)
        self.ch1x = (2. - sgm_mur_x*dt/mur_x)/(2. + sgm_mur_x*dt/mur_x)
        self.ch2x = (2. *           dt/mur_x)/(2. + sgm_mur_x*dt/mur_x)

        self.ce1y = (2. - sgm_epr_y*dt/epr_y)/(2. + sgm_epr_y*dt/epr_y)
        self.ce2y = (2. *           dt/epr_y)/(2. + sgm_epr_y*dt/epr_y)
        self.ch1y = (2. - sgm_mur_y*dt/mur_y)/(2. + sgm_mur_y*dt/mur_y)
        self.ch2y = (2. *           dt/mur_y)/(2. + sgm_mur_y*dt/mur_y)

        self.ce1z = (2. - sgm_epr_z*dt/epr_z)/(2. + sgm_epr_z*dt/epr_z)#
        self.ce2z = (2. *           dt/epr_z)/(2. + sgm_epr_z*dt/epr_z)
        self.ch1z = (2. - sgm_mur_z*dt/mur_z)/(2. + sgm_mur_z*dt/mur_z)
        self.ch2z = (2. *           dt/mur_z)/(2. + sgm_mur_z*dt/mur_z)

        if   fdtd.materials_classification in ['dielectric', 'electric dispersive']:
            self.params = [[self.ce1x, self.ce2x, 0], \
                           [self.ce1y, self.ce2y, 0], \
                           [self.ce1z, self.ce2z, 0]]
        else:
            self.params = [[self.ce1x, self.ce2x, self.ch1x, self.ch2x, 0], \
                           [self.ce1y, self.ce2y, self.ch1y, self.ch2y, 0], \
                           [self.ce1z, self.ce2z, self.ch1z, self.ch2z, 0]]

class Dimagnetic:
    def __init__(self, mur=1., wavelength=None):
        self.classification = 'dimagnetic'
        mur_re = mur.real
        mur_im = mur.imag
        if isinstance(mur, np.ndarray):
            if wavelength is not None:
                wfreq_SI = 2.*np.pi*c0_SI/wavelength
            else:
                if abs(mur_im).sum() != 0.:
                    print 'Warning: Complex permittivity needs wavelength parameter'
                wfreq_SI = 0.
            if   mur.size == 2:
                self.mur_x = mur_re[0]
                self.mur_y = mur_re[1]
                self.mur_z = mur_re[1]
                self.sgm_x = mur_im[0]*wfreq_SI
                self.sgm_y = mur_im[1]*wfreq_SI
                self.sgm_z = mur_im[1]*wfreq_SI
            elif mur.size == 3:
                self.mur_x = mur_re[0]
                self.mur_y = mur_re[1]
                self.mur_z = mur_re[2]
                self.sgm_x = mur_im[0]*wfreq_SI
                self.sgm_y = mur_im[1]*wfreq_SI
                self.sgm_z = mur_im[2]*wfreq_SI
            else:
                raise ValueError, 'tensor epsilon must be size of 2 or 3'
        else:
            if wavelength is not None:
                wfreq_SI = 2.*np.pi*c0_SI/wavelength
            else:
                if mur_im != 0.:
                    print 'Warning: Complex permittivity needs wavelength parameter'
                wfreq_SI = 0.
            self.mur_x = mur_re
            self.mur_y = mur_re
            self.mur_z = mur_re
            self.sgm_x = mur_im*wfreq_SI
            self.sgm_y = mur_im*wfreq_SI
            self.sgm_z = mur_im*wfreq_SI

    def set_coeff(self, fdtd):
        dt  = fdtd.dt
        epr_x = 1.
        epr_y = 1.
        epr_z = 1.
        mur_x = self.mur_x
        mur_y = self.mur_y
        mur_z = self.mur_z
        sgm_epr_x = 0.
        sgm_epr_y = 0.
        sgm_epr_z = 0.
        sgm_mur_x = unit.to_NU(fdtd, 'sigma', self.sgm_x)
        sgm_mur_y = unit.to_NU(fdtd, 'sigma', self.sgm_y)
        sgm_mur_z = unit.to_NU(fdtd, 'sigma', self.sgm_z)

        self.ce1x = (2. - sgm_epr_x*dt/epr_x)/(2. + sgm_epr_x*dt/epr_x)
        self.ce2x = (2. *           dt/epr_x)/(2. + sgm_epr_x*dt/epr_x)
        self.ch1x = (2. - sgm_mur_x*dt/mur_x)/(2. + sgm_mur_x*dt/mur_x)
        self.ch2x = (2. *           dt/mur_x)/(2. + sgm_mur_x*dt/mur_x)

        self.ce1y = (2. - sgm_epr_y*dt/epr_y)/(2. + sgm_epr_y*dt/epr_y)
        self.ce2y = (2. *           dt/epr_y)/(2. + sgm_epr_y*dt/epr_y)
        self.ch1y = (2. - sgm_mur_y*dt/mur_y)/(2. + sgm_mur_y*dt/mur_y)
        self.ch2y = (2. *           dt/mur_y)/(2. + sgm_mur_y*dt/mur_y)

        self.ce1z = (2. - sgm_epr_z*dt/epr_z)/(2. + sgm_epr_z*dt/epr_z)
        self.ce2z = (2. *           dt/epr_z)/(2. + sgm_epr_z*dt/epr_z)
        self.ch1z = (2. - sgm_mur_z*dt/mur_z)/(2. + sgm_mur_z*dt/mur_z)
        self.ch2z = (2. *           dt/mur_z)/(2. + sgm_mur_z*dt/mur_z)

        if   fdtd.materials_classification in ['dimagnetic', 'magnetic dispersive']:
            self.params = [[self.ch1x, self.ch2x, 0], \
                           [self.ch1y, self.ch2y, 0], \
                           [self.ch1z, self.ch2z, 0]]
        else:
            self.params = [[self.ce1x, self.ce2x, self.ch1x, self.ch2x, 0], \
                           [self.ce1y, self.ce2y, self.ch1y, self.ch2y, 0], \
                           [self.ce1z, self.ce2z, self.ch1z, self.ch2z, 0]]

class Dielectromagnetic:
    def __init__(self, epr=1., mur=1., wavelength=None):
        self.classification  = 'dielectromagnetic'
        epr_re = epr.real
        epr_im = epr.imag
        mur_re = mur.real
        mur_im = mur.imag
        if   wavelength is not None:
            wfreq_SI = 2.*np.pi*c0_SI/wavelength
        else:
            if isinstance(epr, np.ndarray) or isinstance(mur, np.ndarray):
                if abs(epr_im).sum() != 0. and abs(mur_im).sum() != 0:
                    print 'Warning: Complex permittivity needs wavelength parameter'
            else:
                if epr_im != 0. and mur_im != 0:
                    print 'Warning: Complex permittivity needs wavelength parameter'
            wfreq_SI = 0.
        if isinstance(epr, np.ndarray):
            if   epr.size == 2:
                self.epr_x = epr_re[0]
                self.epr_y = epr_re[1]
                self.epr_z = epr_re[1]
                self.sgm_epr_x = epr_im[0]*wfreq_SI
                self.sgm_epr_y = epr_im[1]*wfreq_SI
                self.sgm_epr_z = epr_im[1]*wfreq_SI
            elif epr.size == 3:
                self.epr_x = epr_re[0]
                self.epr_y = epr_re[1]
                self.epr_z = epr_re[2]
                self.sgm_epr_x = epr_im[0]*wfreq_SI
                self.sgm_epr_y = epr_im[1]*wfreq_SI
                self.sgm_epr_z = epr_im[2]*wfreq_SI
            else:
                raise ValueError, 'tensor epsilon must be size of 2 or 3'
        else:
            self.epr_x = epr_re
            self.epr_y = epr_re
            self.epr_z = epr_re
            self.sgm_epr_x = epr_im*wfreq_SI
            self.sgm_epr_y = epr_im*wfreq_SI
            self.sgm_epr_z = epr_im*wfreq_SI

        if isinstance(mur, np.ndarray):
            if   mur.size == 2:
                self.mur_x = mur_re[0]
                self.mur_y = mur_re[1]
                self.mur_z = mur_re[1]
                self.sgm_mur_x = mur_im[0]*wfreq_SI
                self.sgm_mur_y = mur_im[1]*wfreq_SI
                self.sgm_mur_z = mur_im[1]*wfreq_SI
            elif mur.size == 3:
                self.mur_x = mur_re[0]
                self.mur_y = mur_re[1]
                self.mur_z = mur_re[2]
                self.sgm_mur_x = mur_im[0]*wfreq_SI
                self.sgm_mur_y = mur_im[1]*wfreq_SI
                self.sgm_mur_z = mur_im[2]*wfreq_SI
            else:
                raise ValueError, 'tensor epsilon must be size of 2 or 3'
        else:
            if wavelength is not None:
                wfreq_SI = 2.*np.pi*c0_SI/wavelength
            else:
                if mur_im != 0.:
                    print 'Warning: Complex permittivity needs wavelength parameter'
                wfreq_SI = 0.
            self.mur_x = mur_re
            self.mur_y = mur_re
            self.mur_z = mur_re
            self.sgm_mur_x = mur_im*wfreq_SI
            self.sgm_mur_y = mur_im*wfreq_SI
            self.sgm_mur_z = mur_im*wfreq_SI

    def set_coeff(self, fdtd):
        dt  = fdtd.dt
        epr_x = self.epr_x
        epr_y = self.epr_y
        epr_z = self.epr_z
        mur_x = self.mur_x
        mur_y = self.mur_y
        mur_z = self.mur_z
        sgm_epr_x = unit.to_NU(fdtd, 'sigma', self.sgm_epr_x)
        sgm_epr_y = unit.to_NU(fdtd, 'sigma', self.sgm_epr_y)
        sgm_epr_z = unit.to_NU(fdtd, 'sigma', self.sgm_epr_z)
        sgm_mur_x = unit.to_NU(fdtd, 'sigma', self.sgm_mur_x)
        sgm_mur_y = unit.to_NU(fdtd, 'sigma', self.sgm_mur_y)
        sgm_mur_z = unit.to_NU(fdtd, 'sigma', self.sgm_mur_z)

        self.ce1x = (2. - sgm_epr_x*dt/epr_x)/(2. + sgm_epr_x*dt/epr_x)
        self.ce2x = (2. *           dt/epr_x)/(2. + sgm_epr_x*dt/epr_x)
        self.ch1x = (2. - sgm_mur_x*dt/mur_x)/(2. + sgm_mur_x*dt/mur_x)
        self.ch2x = (2. *           dt/mur_x)/(2. + sgm_mur_x*dt/mur_x)

        self.ce1y = (2. - sgm_epr_y*dt/epr_y)/(2. + sgm_epr_y*dt/epr_y)
        self.ce2y = (2. *           dt/epr_y)/(2. + sgm_epr_y*dt/epr_y)
        self.ch1y = (2. - sgm_mur_y*dt/mur_y)/(2. + sgm_mur_y*dt/mur_y)
        self.ch2y = (2. *           dt/mur_y)/(2. + sgm_mur_y*dt/mur_y)

        self.ce1z = (2. - sgm_epr_z*dt/epr_z)/(2. + sgm_epr_z*dt/epr_z)
        self.ce2z = (2. *           dt/epr_z)/(2. + sgm_epr_z*dt/epr_z)
        self.ch1z = (2. - sgm_mur_z*dt/mur_z)/(2. + sgm_mur_z*dt/mur_z)
        self.ch2z = (2. *           dt/mur_z)/(2. + sgm_mur_z*dt/mur_z)

        self.params = [[self.ce1x, self.ce2x, self.ch1x, self.ch2x, 0], \
                       [self.ce1y, self.ce2y, self.ch1y, self.ch2y, 0], \
                       [self.ce1z, self.ce2z, self.ch1z, self.ch2z, 0]]

import dispersive as dp
gold = dp.gold
Au   = gold

silver = dp.silver
Ag     = silver

aluminium = dp.aluminum
Al        = aluminium

silicon = dp.silicon
Si      = silicon
