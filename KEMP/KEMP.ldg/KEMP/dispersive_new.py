# Author  : Myung-Su Seok
# Purpose : Python wrappers of the dispersive materials of FDTD
# Target  : CPU, GPU

# Python modules
import numpy as np
import h5py as h5
import os
from scipy.constants import c as c0, hbar

from ndarray import Fields
from mainfdtd import Basic_FDTD
from units import to_SI, to_NU
from util  import *

# CLASSES for Frequency-Dependent-Materials
class Dispersive_2d:
    def __init__(self, fdtd):
        self.fdtd = fdtd
        self.fdtd.cores['dispersive'] = self
        self.fdtd.cores['pole'] = self
        self.fdtd.engine.updates['pole_e'] = self.update_pole
        self.fdtd.engine.updates['disp_e'] = self.update_e
        self.fdtd.engine.updates['pole_h'] = self.update_polh
        self.fdtd.engine.updates['disp_h'] = self.update_h
        self.materials = []

class Dispersive_3d:
    def __init__(self, fdtd):
        self.fdtd = fdtd
        self.fdtd.cores['dispersive'] = self
        self.fdtd.cores['pole'] = self
        self.fdtd.engine.updates['pole_e'] = self.update_pole
        self.fdtd.engine.updates['disp_e'] = self.update_e
        self.fdtd.engine.updates['pole_h'] = self.update_polh
        self.fdtd.engine.updates['disp_h'] = self.update_h
        self.materials = []

# CLASSES for CP-model materials
# Including Drude, Debye, Lorentz materials

class CP_pole:
    def __init__(self, wfreq, gamma, amp, phase):
		"""
		PARAMETERS
		----------
		wfreq	: float
		
		gamma	: float

		amp		: float

		phase	: float

		RETURNS
		-------
		None
		"""
        self.classification  = 'cp'
        self.wfreq = wfreq
        self.gamma = gamma
        self.amp   = amp
        self.phase = phase

    def set_coeff(self, fdtd):
        wfreq = to_NU(fdtd, 'angular frequency', self.wfreq)
        gamma = to_NU(fdtd, 'angular frequency', self.gamma)
        amp   = self.amp
        phase = self.phase
        dt    = fdtd.dt
        alpha = gamma + 1.j*wfreq
        beta  = 2.j*amp*wfreq*np.exp(1.j*phase)
        self.alpha = alpha
        self.beta  = beta

        self.c1 = 2./(2.+alpha*dt)
        self.c2 = np.real(beta)+np.real(alpha*beta*dt/(2.-alpha*dt))
        self.c3 = np.real(beta)-np.real(alpha*beta*dt/(2.+alpha*dt))

        self.cf_r = comp_to_real(fdtd.dtype)(np.real((2.-alpha*dt)/(2.+alpha*dt)))
        self.cf_i = comp_to_real(fdtd.dtype)(np.imag((2.-alpha*dt)/(2.+alpha*dt)))
        self.ce_r = comp_to_real(fdtd.dtype)(np.real(2.*alpha*beta*dt/(2.+alpha*dt)* \
                                      ((1./(2.+alpha*dt))+(1./(2.-alpha*dt)))))
        self.ce_i = comp_to_real(fdtd.dtype)(np.imag(2.*alpha*beta*dt/(2.+alpha*dt)* \
                                      ((1./(2.+alpha*dt))+(1./(2.-alpha*dt)))))

class Drude_pole:
    def __init__(self, drude_wfreqp, drude_gamma):
		"""
		PARAMETERS
		----------
		
		drude_wfreqp : float
			Plasma frequency of the material.

		drude_gamma : float
			Damping factor of the material.

		RETURNS
		-------
		None

		"""
        self.classification  = 'drude'
        self.wfreq = drude_wfreqp
        self.gamma = drude_gamma

    def set_coeff(self, fdtd):
        wfreq = to_NU(fdtd, 'angular frequency', self.wfreq)
        gamma = to_NU(fdtd, 'angular frequency', self.gamma)
        dt    = fdtd.dt

        self.c1 = 2./(2.+gamma*dt)
        self.c2 = -(wfreq**2)*dt/(2.-gamma*dt)
        self.c3 = +(wfreq**2)*dt/(2.+gamma*dt)

        self.cf_r = comp_to_real(fdtd.dtype)( (2.- gamma    *dt)/(2.+gamma*dt))
        self.ce_r = comp_to_real(fdtd.dtype)(-(2.*(wfreq**2)*dt)/(2.+gamma*dt)* \
                                            (((1./(2.+gamma*dt))+(1./(2.-gamma*dt)))))

class Debye_pole(Drude_pole):
    def __init__(self, tau, d_epsilon):
        self.classification  = 'drude'
        self.tau = tau
        self.d_epsilon = d_epsilon

    def set_coeff(self, fdtd):
        tau       = self.tau
        d_epsilon = self.d_epsilon
        dt    = fdtd.dt

        self.c1 = 2.*tau/(2.*tau+dt)
        self.c2 = 2.*d_epsilon/(2.*tau+dt)
        self.c3 = 2.*d_epsilon/(2.*tau-dt)

        self.cf_r = comp_to_real(fdtd.dtype)((2.*tau-dt)/(2.*tau+dt))
        self.ce_r = comp_to_real(fdtd.dtype)((4.*d_epsilon*tau)/(2.*tau+dt)* \
                                            (((1./(2.*tau-dt))-(1./(2.*tau+dt)))))

class Lorentz_under_damp_pole(CP_pole):
    def __init__(self, wfreq, delta, d_epsilon):
        self.classification  = 'cp'
        self.wfreq = wfreq
        self.delta = delta
        self.d_epsilon = d_epsilon

    def set_coeff(self, fdtd):
        wfreq = to_NU(fdtd, 'angular frequency', self.wfreq)
        delta = to_NU(fdtd, 'angular frequency', self.delta)
        d_epsilon = self.d_epsilon
        dt    = fdtd.dt
        alpha = delta + 1.j*np.sqrt(wfreq**2-delta**2)
        beta  = +1.j*d_epsilon*(wfreq**2)/np.sqrt(wfreq**2-delta**2)
        self.alpha = alpha
        self.beta  = beta

        self.c1 = 2./(2.+alpha*dt)
        self.c2 = np.real(beta)+np.real(alpha*beta*dt/(2.-alpha*dt))
        self.c3 = np.real(beta)-np.real(alpha*beta*dt/(2.+alpha*dt))

        self.cf_r = comp_to_real(fdtd.dtype)(np.real((2.-alpha*dt)/(2.+alpha*dt)))
        self.cf_i = comp_to_real(fdtd.dtype)(np.imag((2.-alpha*dt)/(2.+alpha*dt)))
        self.ce_r = comp_to_real(fdtd.dtype)(np.real(2.*alpha*beta*dt/(2.+alpha*dt)* \
                                      ((1./(2.+alpha*dt))+(1./(2.-alpha*dt)))))
        self.ce_i = comp_to_real(fdtd.dtype)(np.imag(2.*alpha*beta*dt/(2.+alpha*dt)* \
                                      ((1./(2.+alpha*dt))+(1./(2.-alpha*dt)))))

class Lorentz_over_damp_pole1(Debye_pole):
    def __init__(self, wfreq, delta, d_epsilon):
        sqroot = np.sqrt(delta**2-wfreq**2)
        tau   = 1./(delta-sqroot)
        d_epr = +(d_epsilon*wfreq**2)/(2.*sqroot)/(delta-sqroot)
        Debye_pole.__init__(self, tau, d_epr)

class Lorentz_over_damp_pole2(Debye_pole):
    def __init__(self, wfreq, delta, d_epsilon):
        sqroot = np.sqrt(delta**2-wfreq**2)
        tau   = 1./(delta+sqroot)
        d_epr = -(d_epsilon*wfreq**2)/(2.*sqroot)/(delta+sqroot)
        Debye_pole.__init__(self, tau, d_epr)

def Lorentz_pole(wfreq, delta, d_epsilon):
    if wfreq == delta:
        raise NotImplementedError, 'Case of wfreq==delta in Lorentz Pole is NOT IMPLEMENTED in KEMP'
    elif wfreq > delta:
        return [Lorentz_under_damp_pole(wfreq, delta, d_epsilon)]
    else:
        return [Lorentz_over_damp_pole1(wfreq, delta, d_epsilon), Lorentz_over_damp_pole2(wfreq, delta, d_epsilon)]

class epr_profile:
    def __init__(self, epr_inf, epr_sigma, poles):
        self.epr_inf   = epr_inf
        self.epr_sigma = epr_sigma
        self.epr_poles = poles

class mur_profile:
    def __init__(self, mur_inf, mur_sigma, poles):
        self.mur_inf   = epr_inf
        self.mur_sigma = epr_sigma
        self.mur_poles = poles

class dispersive_material:
    def __init__(self, mark_number, epr_prof=None, mur_prof=None):
        self.mark_number = mark_number
        if   epr_prof != None and mur_prof != None:
            self.classification  = 'electromagnetic dispersive'
            self.epr_inf, self.epr_sigma = epr_prof.epr_inf, epr_prof.epr_sigma
            self.mur_inf, self.mur_sigma = mur_prof.mur_inf, mur_prof.mur_sigma
        elif epr_prof != None:
            self.classification  = 'electric dispersive'
            self.epr_inf, self.epr_sigma = epr_prof.epr_inf, epr_prof.epr_sigma
            self.mur_inf, self.mur_sigma =              1.0,                0.0
        elif mur_prof != None:
            self.classification  = 'magnetic dispersive'
            self.epr_inf, self.epr_sigma =              1.0,                0.0
            self.mur_inf, self.mur_sigma = mur_prof.mur_inf, mur_prof.mur_sigma
        else:
            from exception import FDTDError
            raise FDTDError, ''

        # cp_poles is list of elements(class CP_pole)
        self.dpoles     = drude_poles
        self.cpoles     = cp_poles
        self.poles      = self.dpoles + self.cpoles
        self.dpole_num  = len(self.dpoles)
        self.cpole_num  = len(self.cpoles)
        self.pole_num   = self.dpole_num + self.cpole_num

    def set_coeff(self, fdtd):
        dt = fdtd.dt
        self.c2_sum = comp_to_real(fdtd.dtype)(0.)
        self.c3_sum = comp_to_real(fdtd.dtype)(0.)
        for pole in self.poles:
            self.c2_sum += pole.c2
            self.c3_sum += pole.c3
        self.c2_sum += self.sigma
        self.c3_sum += self.sigma

        self.ce1 = comp_to_real(fdtd.dtype)((2.-fdtd.dt*self.c2_sum/self.eps_r)/(2.+fdtd.dt*self.c3_sum/self.eps_r))
        self.ce2 = comp_to_real(fdtd.dtype)((2.*fdtd.dt            /self.eps_r)/(2.+fdtd.dt*self.c3_sum/self.eps_r))
        self.ch1 = comp_to_real(fdtd.dtype)(1.)
        self.ch2 = comp_to_real(fdtd.dtype)(fdtd.dt)

        for pole in self.poles:
            pole.ce_r *= 2.*fdtd.dt/self.eps_r/(2.+fdtd.dt*self.c3_sum/self.eps_r)
            if pole.classification == 'cp':
                pole.ce_i *= 2.*fdtd.dt/self.eps_r/(2.+fdtd.dt*self.c3_sum/self.eps_r)

        if   fdtd.materials_classification in ['dielectric', 'electric dispersive']:
            self.params = [[self.ce1, self.ce2, self.mark_number], \
                           [self.ce1, self.ce2, self.mark_number], \
                           [self.ce1, self.ce2, self.mark_number]]
        elif fdtd.materials_classification in ['dimagnetic', 'magnetic dispersive']:
            self.params = [[self.ch1, self.ch2, self.mark_number], \
                           [self.ch1, self.ch2, self.mark_number], \
                           [self.ch1, self.ch2, self.mark_number]]
        else:
            self.params = [[self.ce1, self.ce2, self.ch1, self.ch2, self.mark_number], \
                           [self.ce1, self.ce2, self.ch1, self.ch2, self.mark_number], \
                           [self.ce1, self.ce2, self.ch1, self.ch2, self.mark_number]]

class Drude_CP_material:
    def __init__(self, mark_number, eps_r, sigma, drude_poles, cp_poles):
		"""Apply drude poles and critical point poles(cp poles) 
		PARAMETERS
		----------
		"""
        self.classification  = 'electric dispersive'
        self.eps_r = eps_r
        self.sigma = sigma
        self.mark_number = mark_number
        # cp_pole object is an element of the list
        self.dpoles     = drude_poles
        self.cpoles     = cp_poles
        self.poles      = self.dpoles + self.cpoles
        self.dpole_num  = len(self.dpoles)
        self.cpole_num  = len(self.cpoles)
        self.pole_num   = self.dpole_num + self.cpole_num

        for dpole in self.dpoles:
            if dpole.classification != 'drude':
                raise ValueError, 'must be drude poles'

        for cpole in self.cpoles:
            if cpole.classification != 'cp':
                raise ValueError, 'must be cp poles'

    def set_coeff(self, fdtd):
        dt = fdtd.dt
        self.c2_sum = comp_to_real(fdtd.dtype)(0.)
        self.c3_sum = comp_to_real(fdtd.dtype)(0.)
        for pole in self.poles:
            self.c2_sum += pole.c2
            self.c3_sum += pole.c3
        self.c2_sum += self.sigma
        self.c3_sum += self.sigma

        self.ce1 = comp_to_real(fdtd.dtype)((2.-fdtd.dt*self.c2_sum/self.eps_r)/(2.+fdtd.dt*self.c3_sum/self.eps_r))
        self.ce2 = comp_to_real(fdtd.dtype)((2.*fdtd.dt            /self.eps_r)/(2.+fdtd.dt*self.c3_sum/self.eps_r))
        self.ch1 = comp_to_real(fdtd.dtype)(1.)
        self.ch2 = comp_to_real(fdtd.dtype)(fdtd.dt)

        for pole in self.poles:
            pole.ce_r *= 2.*fdtd.dt/self.eps_r/(2.+fdtd.dt*self.c3_sum/self.eps_r)
            if pole.classification == 'cp':
                pole.ce_i *= 2.*fdtd.dt/self.eps_r/(2.+fdtd.dt*self.c3_sum/self.eps_r)

        if   fdtd.materials_classification in ['dielectric', 'electric dispersive']:
            self.params = [[self.ce1, self.ce2, self.mark_number], \
                           [self.ce1, self.ce2, self.mark_number], \
                           [self.ce1, self.ce2, self.mark_number]]
        else:
            self.params = [[self.ce1, self.ce2, self.ch1, self.ch2, self.mark_number], \
                           [self.ce1, self.ce2, self.ch1, self.ch2, self.mark_number], \
                           [self.ce1, self.ce2, self.ch1, self.ch2, self.mark_number]]

class Drude_CP_world_2d(Dispersive_2d):
    def __init__(self, fdtd, region, materials):
        Dispersive_2d.__init__(self, fdtd)
        self.materials = materials
        self.region = region
        self.pt0 = region[0]
        self.pt1 = region[1]
        self.nx  = fdtd.x[self.pt0[0]:self.pt1[0]+1].size
        self.ny  = fdtd.y[self.pt0[1]:self.pt1[1]+1].size
        self.mx, self.my, self.mz = \
            [Fields(fdtd, (self.nx, self.ny), dtype=np.int32, \
             init_value=np.zeros((self.nx, self.ny), dtype=np.int32) , name='mark') for i in xrange(3)]
        self.px, self.py = [self.pt0[i] for i in xrange(2)]

        if 'opencl' in fdtd.engine.name:
            self.gs  = cl_global_size(self.nx*self.ny, fdtd.engine.ls)
        else:
            self.gs  = self.nx*self.ny

        self.material_nums_list = []
        self.dpole_nums_list = []
        self.cpole_nums_list = []
        self.pole_nums_list  = []
        for material in self.materials:
            self.material_nums_list.append(material.mark_number)
            self.dpole_nums_list.append(material.dpole_num)
            self.cpole_nums_list.append(material.cpole_num)
            self.pole_nums_list.append(material.pole_num)
            for pole in material.poles:
                pole.set_coeff(fdtd)
            material.set_coeff(fdtd)

        self.materials_num = int(np.array(self.material_nums_list).max())
        self.dpolenum_max  = int(np.array(self.dpole_nums_list).max())
        self.cpolenum_max  = int(np.array(self.cpole_nums_list).max())
        self.polenum_max   = int(np.array(self.pole_nums_list).max())

    def setup(self):
        fdtd = self.fdtd
        self.dr_cf_r, self.dr_ce_r = \
            [[] for i in xrange(2)]
        self.dr_fx_r, self.dr_fy_r, self.dr_fz_r = [[] for i in xrange(3)]
        self.cp_cf_r, self.cp_cf_i, self.cp_ce_r, self.cp_ce_i, \
            self.cp_fx_r, self.cp_fx_i, self.cp_fy_r, self.cp_fy_i, self.cp_fz_r, self.cp_fz_i = \
            [[] for i in xrange(10)]
        dr_cf_r_buf, dr_ce_r_buf, = \
            [[np.zeros(self.materials_num+1, dtype=comp_to_real(fdtd.dtype)) for p in xrange(self.dpolenum_max)] for i in xrange(2)]
        cp_cf_r_buf, cp_cf_i_buf, cp_ce_r_buf, cp_ce_i_buf = \
            [[np.zeros(self.materials_num+1, dtype=comp_to_real(fdtd.dtype)) for p in xrange(self.cpolenum_max)] for i in xrange(4)]
        f_buf = np.zeros((self.nx, self.ny), dtype=comp_to_real(fdtd.dtype))
        for material in self.materials:
            m = material.mark_number
            for p, dpole in enumerate(material.dpoles):
                dr_cf_r_buf[p][m] = dpole.cf_r
                dr_ce_r_buf[p][m] = dpole.ce_r
            for p, cpole in enumerate(material.cpoles):
                cp_cf_r_buf[p][m] = cpole.cf_r
                cp_cf_i_buf[p][m] = cpole.cf_i
                cp_ce_r_buf[p][m] = cpole.ce_r
                cp_ce_i_buf[p][m] = cpole.ce_i
        for p in xrange(self.dpolenum_max):
            self.dr_cf_r.append(Fields(fdtd, (self.materials_num+1,), dtype=comp_to_real(fdtd.dtype), mem_flag='r' , init_value=dr_cf_r_buf[p]))
            self.dr_ce_r.append(Fields(fdtd, (self.materials_num+1,), dtype=comp_to_real(fdtd.dtype), mem_flag='r' , init_value=dr_ce_r_buf[p]))
            if   'TE' in fdtd.mode:
                self.dr_fx_r.append(Fields(fdtd, (self.nx, self.ny), dtype=fdtd.dtype, mem_flag='rw', init_value=f_buf))
                self.dr_fy_r.append(Fields(fdtd, (self.nx, self.ny), dtype=fdtd.dtype, mem_flag='rw', init_value=f_buf))
            elif 'TM' in fdtd.mode:
                self.dr_fz_r.append(Fields(fdtd, (self.nx, self.ny), dtype=fdtd.dtype, mem_flag='rw', init_value=f_buf))
        for p in xrange(self.cpolenum_max):
            self.cp_cf_r.append(Fields(fdtd, (self.materials_num+1,), dtype=comp_to_real(fdtd.dtype), mem_flag='r' , init_value=cp_cf_r_buf[p]))
            self.cp_cf_i.append(Fields(fdtd, (self.materials_num+1,), dtype=comp_to_real(fdtd.dtype), mem_flag='r' , init_value=cp_cf_i_buf[p]))
            self.cp_ce_r.append(Fields(fdtd, (self.materials_num+1,), dtype=comp_to_real(fdtd.dtype), mem_flag='r' , init_value=cp_ce_r_buf[p]))
            self.cp_ce_i.append(Fields(fdtd, (self.materials_num+1,), dtype=comp_to_real(fdtd.dtype), mem_flag='r' , init_value=cp_ce_i_buf[p]))
            if   'TE' in fdtd.mode:
                self.cp_fx_r.append(Fields(fdtd, (self.nx, self.ny), dtype=fdtd.dtype, mem_flag='rw', init_value=f_buf))
                self.cp_fx_i.append(Fields(fdtd, (self.nx, self.ny), dtype=fdtd.dtype, mem_flag='rw', init_value=f_buf))
                self.cp_fy_r.append(Fields(fdtd, (self.nx, self.ny), dtype=fdtd.dtype, mem_flag='rw', init_value=f_buf))
                self.cp_fy_i.append(Fields(fdtd, (self.nx, self.ny), dtype=fdtd.dtype, mem_flag='rw', init_value=f_buf))
            elif 'TM' in fdtd.mode:
                self.cp_fz_r.append(Fields(fdtd, (self.nx, self.ny), dtype=fdtd.dtype, mem_flag='rw', init_value=f_buf))
                self.cp_fz_i.append(Fields(fdtd, (self.nx, self.ny), dtype=fdtd.dtype, mem_flag='rw', init_value=f_buf))
        '''
        if   'TE' in fdtd.mode:
            self.mz.release_data()
        elif 'TM' in fdtd.mode:
            self.mx.release_data()
            self.my.release_data()
        '''

        if 'cuda' in fdtd.engine.name:
            code_prev = ['__DECLARE_CONSTANT_ARRAYS__', \
                         'cf_r[mkx]', 'cf_r[mky]', 'cf_r[mkz]', \
                         'cf_i[mkx]', 'cf_i[mky]', 'cf_i[mkz]', \
                         'ce_r[mkx]', 'ce_r[mky]', 'ce_r[mkz]', \
                         'ce_i[mkx]', 'ce_i[mky]', 'ce_i[mkz]', ]
            code_post = ['''__DEVICE__ __CONSTANT__ __FLOAT__ cst_cf_r[%s];
                            __DEVICE__ __CONSTANT__ __FLOAT__ cst_cf_i[%s];
                            __DEVICE__ __CONSTANT__ __FLOAT__ cst_ce_r[%s];
                            __DEVICE__ __CONSTANT__ __FLOAT__ cst_ce_i[%s];''' % tuple(self.materials_num + 1 for i in xrange(4)), \
                           'cst_cf_r[mkx]', 'cst_cf_r[mky]', 'cst_cf_r[mkz]', \
                           'cst_cf_i[mkx]', 'cst_cf_i[mky]', 'cst_cf_i[mkz]', \
                           'cst_ce_r[mkx]', 'cst_ce_r[mky]', 'cst_ce_r[mkz]', \
                           'cst_ce_i[mkx]', 'cst_ce_i[mky]', 'cst_ce_i[mkz]', ]
        else:
            code_prev = ['__DECLARE_CONSTANT_ARRAYS__']
            code_post = ['']

        code = template_to_code(fdtd.engine.templates['dispersive'], code_prev+fdtd.engine.code_prev, code_post+fdtd.engine.code_post)
        fdtd.engine.programs['dispersive'] = fdtd.engine.build(code)
        fdtd.engine.kernel_args['dr_pole'] = {}
        fdtd.engine.kernel_args['cp_pole'] = {}
        fdtd.engine.kernel_args['dispersive'] = {'dr_pole':{}, 'cp_pole':{}}

        if 'cuda' in fdtd.engine.name:
            self.cst_cf_r = fdtd.engine.get_global(fdtd.engine.programs['dispersive'], 'cst_cf_r')
            self.cst_cf_i = fdtd.engine.get_global(fdtd.engine.programs['dispersive'], 'cst_cf_i')
            self.cst_ce_r = fdtd.engine.get_global(fdtd.engine.programs['dispersive'], 'cst_ce_r')
            self.cst_ce_i = fdtd.engine.get_global(fdtd.engine.programs['dispersive'], 'cst_ce_i')

        if fdtd.mode == '2DTE':
            for p in xrange(self.dpolenum_max):
                fdtd.engine.kernel_args['dr_pole'][p] = {'real':None, 'imag':None}
                fdtd.engine.kernel_args['dispersive']['dr_pole'][p] = {'real':None, 'imag':None}
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['dr_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), \
                                                                   np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                                   np.int32(self.px), np.int32(self.py), \
                                                                   self.mx.data, self.my.data, \
                                                                   fdtd.ex.__dict__[part].data, fdtd.ey.__dict__[part].data, \
                                                                   self.dr_fx_r[p].__dict__[part].data, self.dr_fy_r[p].__dict__[part].data, \
                                                                   self.dr_cf_r[p].data, self.dr_ce_r[p].data  ]
                    fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), \
                                                                                 np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                                                 np.int32(self.px), np.int32(self.py), \
                                                                                 fdtd.ex.__dict__[part].data, fdtd.ey.__dict__[part].data, \
                                                                                 self.dr_fx_r[p].__dict__[part].data, self.dr_fy_r[p].__dict__[part].data]
            for p in xrange(self.cpolenum_max):
                fdtd.engine.kernel_args['cp_pole'][p] = {'real':None, 'imag':None}
                fdtd.engine.kernel_args['dispersive']['cp_pole'][p] = {'real':None, 'imag':None}
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['cp_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), \
                                                                   np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                                   np.int32(self.px), np.int32(self.py), \
                                                                   self.mx.data, self.my.data, \
                                                                   fdtd.ex.__dict__[part].data, fdtd.ey.__dict__[part].data, \
                                                                   self.cp_fx_r[p].__dict__[part].data, self.cp_fy_r[p].__dict__[part].data, \
                                                                   self.cp_fx_i[p].__dict__[part].data, self.cp_fy_i[p].__dict__[part].data, \
                                                                   self.cp_cf_r[p].data, self.cp_cf_i[p].data, \
                                                                   self.cp_ce_r[p].data, self.cp_ce_i[p].data  ]
                    fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), \
                                                                                 np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                                                 np.int32(self.px), np.int32(self.py), \
                                                                                 fdtd.ex.__dict__[part].data, fdtd.ey.__dict__[part].data, \
                                                                                 self.cp_fx_r[p].__dict__[part].data, self.cp_fy_r[p].__dict__[part].data]

        if fdtd.mode == '2DTM':
            for p in xrange(self.dpolenum_max):
                fdtd.engine.kernel_args['dr_pole'][p] = {'real':None, 'imag':None}
                fdtd.engine.kernel_args['dispersive']['dr_pole'][p] = {'real':None, 'imag':None}
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['dr_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), \
                                                                   np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                                   np.int32(self.px), np.int32(self.py), \
                                                                   self.mz.data, \
                                                                   fdtd.ez.__dict__[part].data, \
                                                                   self.dr_fz_r[p].__dict__[part].data, \
                                                                   self.dr_cf_r[p].data, self.dr_ce_r[p].data]
                    fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), \
                                                                                 np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                                                 np.int32(self.px), np.int32(self.py), \
                                                                                 fdtd.ez.__dict__[part].data, \
                                                                                 self.dr_fz_r[p].__dict__[part].data]
            for p in xrange(self.cpolenum_max):
                fdtd.engine.kernel_args['cp_pole'][p] = {'real':None, 'imag':None}
                fdtd.engine.kernel_args['dispersive']['cp_pole'][p] = {'real':None, 'imag':None}
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['cp_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), \
                                                                   np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                                   np.int32(self.px), np.int32(self.py), \
                                                                   self.mz.data, \
                                                                   fdtd.ez.__dict__[part].data, \
                                                                   self.cp_fz_r[p].__dict__[part].data, self.cp_fz_i[p].__dict__[part].data, \
                                                                   self.cp_cf_r[p].data, self.cp_cf_i[p].data, \
                                                                   self.cp_ce_r[p].data, self.cp_ce_i[p].data  ]
                    fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), \
                                                                                 np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                                                 np.int32(self.px), np.int32(self.py), \
                                                                                 fdtd.ez.__dict__[part].data, \
                                                                                 self.cp_fz_r[p].__dict__[part].data]

        if 'opencl' in fdtd.engine.name:
            gs = cl_global_size(self.nx*self.ny, fdtd.engine.ls)
            for p in xrange(self.dpolenum_max):
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['dr_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dr_pole'][p][part]
                    fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part]
            for p in xrange(self.cpolenum_max):
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['cp_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['cp_pole'][p][part]
                    fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part]

        else:
            gs = self.nx*self.ny
            for p in xrange(self.dpolenum_max):
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['dr_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dr_pole'][p][part]
                    fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part]
            for p in xrange(self.cpolenum_max):
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['cp_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['cp_pole'][p][part]
                    fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part]

        if   fdtd.mode == '2DTE':
            if 'opencl' in fdtd.engine.name:
                fdtd.engine.kernels['dr_pole']    = fdtd.engine.programs['dispersive'].update_dpole_2dte
                fdtd.engine.kernels['cp_pole']    = fdtd.engine.programs['dispersive'].update_cpole_2dte
                fdtd.engine.kernels['dispersive'] = fdtd.engine.programs['dispersive'].update_e_2dte
            elif 'cuda' in fdtd.engine.name:
                fdtd.engine.kernels['dr_pole']    = fdtd.engine.get_function(fdtd.engine.programs['dispersive'], 'update_dpole_2dte')
                fdtd.engine.kernels['cp_pole']    = fdtd.engine.get_function(fdtd.engine.programs['dispersive'], 'update_cpole_2dte')
                fdtd.engine.kernels['dispersive'] = fdtd.engine.get_function(fdtd.engine.programs['dispersive'], 'update_e_2dte')
                if self.dpolenum_max > 0:
                    fdtd.engine.prepare(fdtd.engine.kernels['dr_pole']   , fdtd.engine.kernel_args['dr_pole'][0]['real'])
                    fdtd.engine.prepare(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['dr_pole'][0]['real'])
                if self.cpolenum_max > 0:
                    fdtd.engine.prepare(fdtd.engine.kernels['cp_pole']   , fdtd.engine.kernel_args['cp_pole'][0]['real'])
                    fdtd.engine.prepare(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['cp_pole'][0]['real'])
            elif  'cpu' in fdtd.engine.name:
                fdtd.engine.kernels['dr_pole']    = fdtd.engine.set_kernel(fdtd.engine.programs['dispersive'].update_dpole_2dte, fdtd.engine.kernel_args['dr_pole'][0]['real'])
                fdtd.engine.kernels['cp_pole']    = fdtd.engine.set_kernel(fdtd.engine.programs['dispersive'].update_cpole_2dte, fdtd.engine.kernel_args['cp_pole'][0]['real'])
                fdtd.engine.kernels['dispersive'] = fdtd.engine.set_kernel(fdtd.engine.programs['dispersive'].update_e_2dte    , fdtd.engine.kernel_args['dispersive']['dr_pole'][0]['real'])

        elif fdtd.mode == '2DTM':
            if 'opencl' in fdtd.engine.name:
                fdtd.engine.kernels['dr_pole']    = fdtd.engine.programs['dispersive'].update_dpole_2dtm
                fdtd.engine.kernels['cp_pole']    = fdtd.engine.programs['dispersive'].update_cpole_2dtm
                fdtd.engine.kernels['dispersive'] = fdtd.engine.programs['dispersive'].update_e_2dtm
            elif 'cuda' in fdtd.engine.name:
                fdtd.engine.kernels['dr_pole']    = fdtd.engine.get_function(fdtd.engine.programs['dispersive'], 'update_dpole_2dtm')
                fdtd.engine.kernels['cp_pole']    = fdtd.engine.get_function(fdtd.engine.programs['dispersive'], 'update_cpole_2dtm')
                fdtd.engine.kernels['dispersive'] = fdtd.engine.get_function(fdtd.engine.programs['dispersive'], 'update_e_2dtm')
                if self.dpolenum_max > 0:
                    fdtd.engine.prepare(fdtd.engine.kernels['dr_pole']   , fdtd.engine.kernel_args['dr_pole'][0]['real'])
                    fdtd.engine.prepare(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['dr_pole'][0]['real'])
                if self.cpolenum_max > 0:
                    fdtd.engine.prepare(fdtd.engine.kernels['cp_pole']   , fdtd.engine.kernel_args['cp_pole'][0]['real'])
                    fdtd.engine.prepare(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['cp_pole'][0]['real'])
            elif  'cpu' in fdtd.engine.name:
                fdtd.engine.kernels['dr_pole']    = fdtd.engine.set_kernel(fdtd.engine.programs['dispersive'].update_dpole_2dtm, fdtd.engine.kernel_args['dr_pole'][0]['real'])
                fdtd.engine.kernels['cp_pole']    = fdtd.engine.set_kernel(fdtd.engine.programs['dispersive'].update_cpole_2dtm, fdtd.engine.kernel_args['cp_pole'][0]['real'])
                fdtd.engine.kernels['dispersive'] = fdtd.engine.set_kernel(fdtd.engine.programs['dispersive'].update_e_2dtm    , fdtd.engine.kernel_args['dispersive']['dr_pole'][0]['real'])

    def update_pole(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for p in xrange(self.dpolenum_max):
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dr_pole'](*(fdtd.engine.kernel_args['dr_pole'][p][part]))
                elif 'cuda' in fdtd.engine.name:
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_cf_r[0], self.dr_cf_r[p].data, self.dr_cf_r[p].nbytes])
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_ce_r[0], self.dr_ce_r[p].data, self.dr_ce_r[p].nbytes])
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['dr_pole'], fdtd.engine.kernel_args['dr_pole'][p][part])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dr_pole'](*(fdtd.engine.kernel_args['dr_pole'][p][part][3:]))
                evts.append(evt)
        for p in xrange(self.cpolenum_max):
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['cp_pole'](*(fdtd.engine.kernel_args['cp_pole'][p][part]))
                elif 'cuda' in fdtd.engine.name:
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_cf_r[0], self.cp_cf_r[p].data, self.cp_cf_r[p].nbytes])
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_cf_i[0], self.cp_cf_i[p].data, self.cp_cf_i[p].nbytes])
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_ce_r[0], self.cp_ce_r[p].data, self.cp_ce_r[p].nbytes])
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_ce_i[0], self.cp_ce_i[p].data, self.cp_ce_i[p].nbytes])
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['cp_pole'], fdtd.engine.kernel_args['cp_pole'][p][part])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['cp_pole'](*(fdtd.engine.kernel_args['cp_pole'][p][part][3:]))
                evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for p in xrange(self.dpolenum_max):
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dispersive'](*(fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part]))
                elif 'cuda' in fdtd.engine.name:
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dispersive'](*(fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part][3:]))
                evts.append(evt)
        for p in xrange(self.cpolenum_max):
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dispersive'](*(fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part]))
                elif 'cuda' in fdtd.engine.name:
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dispersive'](*(fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part][3:]))
                evts.append(evt)
        wait_for_events(fdtd, evts)
        return evts

class Drude_CP_world_3d(Dispersive_3d):
    def __init__(self, fdtd, region, materials):
        Dispersive_3d.__init__(self, fdtd)
        self.materials = materials
        self.region = region
        self.pt0 = region[0]
        self.pt1 = region[1]
        self.nx  = fdtd.x[self.pt0[0]:self.pt1[0]+1].size
        self.ny  = fdtd.y[self.pt0[1]:self.pt1[1]+1].size
        self.nz  = fdtd.z[self.pt0[2]:self.pt1[2]+1].size
        if 'opencl' in fdtd.engine.name:
            self.gs  = cl_global_size(self.nx*self.ny*self.nz, fdtd.engine.ls)
        else:
            self.gs  = self.nx*self.ny*self.nz
        self.mx, self.my, self.mz = \
            [Fields(fdtd, (self.nx, self.ny, self.nz), dtype=np.int32, \
             init_value=np.zeros((self.nx, self.ny, self.nz), dtype=np.int32) , name='mark') for i in xrange(3)]
        self.px, self.py, self.pz = [self.pt0[i] for i in xrange(3)]

        self.material_nums_list = []
        self.dpole_nums_list = []
        self.cpole_nums_list = []
        self.pole_nums_list  = []
        for material in self.materials:
            self.material_nums_list.append(material.mark_number)
            self.dpole_nums_list.append(material.dpole_num)
            self.cpole_nums_list.append(material.cpole_num)
            self.pole_nums_list.append(material.pole_num)
            for pole in material.poles:
                pole.set_coeff(fdtd)
            material.set_coeff(fdtd)

        self.materials_num = int(np.array(self.material_nums_list).max())
        self.dpolenum_max  = int(np.array(self.dpole_nums_list).max())
        self.cpolenum_max  = int(np.array(self.cpole_nums_list).max())
        self.polenum_max   = int(np.array(self.pole_nums_list).max())

    def setup(self):
        fdtd = self.fdtd
        self.dr_cf_r, self.dr_ce_r = [[] for i in xrange(2)]
        self.dr_fx_r, self.dr_fy_r, self.dr_fz_r = [[] for i in xrange(3)]
        self.cp_cf_r, self.cp_cf_i, self.cp_ce_r, self.cp_ce_i, \
            self.cp_fx_r, self.cp_fx_i, self.cp_fy_r, self.cp_fy_i, self.cp_fz_r, self.cp_fz_i = \
            [[] for i in xrange(10)]
        dr_cf_r_buf, dr_ce_r_buf, = \
            [[np.zeros(self.materials_num+1, dtype=comp_to_real(fdtd.dtype)) for p in xrange(self.dpolenum_max)] for i in xrange(2)]
        cp_cf_r_buf, cp_cf_i_buf, cp_ce_r_buf, cp_ce_i_buf =\
            [[np.zeros(self.materials_num+1, dtype=comp_to_real(fdtd.dtype)) for p in xrange(self.cpolenum_max)] for i in xrange(4)]
        f_buf = np.zeros((self.nx, self.ny, self.nz), dtype=comp_to_real(fdtd.dtype))
        for material in self.materials:
            m = material.mark_number
            for p, dpole in enumerate(material.dpoles):
                dr_cf_r_buf[p][m] = dpole.cf_r
                dr_ce_r_buf[p][m] = dpole.ce_r
            for p, cpole in enumerate(material.cpoles):
                cp_cf_r_buf[p][m] = cpole.cf_r
                cp_cf_i_buf[p][m] = cpole.cf_i
                cp_ce_r_buf[p][m] = cpole.ce_r
                cp_ce_i_buf[p][m] = cpole.ce_i
        for p in xrange(self.dpolenum_max):
            self.dr_cf_r.append(Fields(fdtd, (    self.materials_num+1,), comp_to_real(fdtd.dtype), mem_flag='r' , init_value=dr_cf_r_buf[p]))
            self.dr_ce_r.append(Fields(fdtd, (    self.materials_num+1,), comp_to_real(fdtd.dtype), mem_flag='r' , init_value=dr_ce_r_buf[p]))
            self.dr_fx_r.append(Fields(fdtd, (self.nx, self.ny, self.nz),              fdtd.dtype , mem_flag='rw', init_value=f_buf))
            self.dr_fy_r.append(Fields(fdtd, (self.nx, self.ny, self.nz),              fdtd.dtype , mem_flag='rw', init_value=f_buf))
            self.dr_fz_r.append(Fields(fdtd, (self.nx, self.ny, self.nz),              fdtd.dtype , mem_flag='rw', init_value=f_buf))
        for p in xrange(self.cpolenum_max):
            self.cp_cf_r.append(Fields(fdtd, (    self.materials_num+1,), comp_to_real(fdtd.dtype), mem_flag='r' , init_value=cp_cf_r_buf[p]))
            self.cp_cf_i.append(Fields(fdtd, (    self.materials_num+1,), comp_to_real(fdtd.dtype), mem_flag='r' , init_value=cp_cf_i_buf[p]))
            self.cp_ce_r.append(Fields(fdtd, (    self.materials_num+1,), comp_to_real(fdtd.dtype), mem_flag='r' , init_value=cp_ce_r_buf[p]))
            self.cp_ce_i.append(Fields(fdtd, (    self.materials_num+1,), comp_to_real(fdtd.dtype), mem_flag='r' , init_value=cp_ce_i_buf[p]))
            self.cp_fx_r.append(Fields(fdtd, (self.nx, self.ny, self.nz),              fdtd.dtype , mem_flag='rw', init_value=f_buf))
            self.cp_fx_i.append(Fields(fdtd, (self.nx, self.ny, self.nz),              fdtd.dtype , mem_flag='rw', init_value=f_buf))
            self.cp_fy_r.append(Fields(fdtd, (self.nx, self.ny, self.nz),              fdtd.dtype , mem_flag='rw', init_value=f_buf))
            self.cp_fy_i.append(Fields(fdtd, (self.nx, self.ny, self.nz),              fdtd.dtype , mem_flag='rw', init_value=f_buf))
            self.cp_fz_r.append(Fields(fdtd, (self.nx, self.ny, self.nz),              fdtd.dtype , mem_flag='rw', init_value=f_buf))
            self.cp_fz_i.append(Fields(fdtd, (self.nx, self.ny, self.nz),              fdtd.dtype , mem_flag='rw', init_value=f_buf))

        if 'cuda' in fdtd.engine.name:
            code_prev = ['__DECLARE_CONSTANT_ARRAYS__', \
                         'cf_r[mkx]', 'cf_r[mky]', 'cf_r[mkz]', \
                         'cf_i[mkx]', 'cf_i[mky]', 'cf_i[mkz]', \
                         'ce_r[mkx]', 'ce_r[mky]', 'ce_r[mkz]', \
                         'ce_i[mkx]', 'ce_i[mky]', 'ce_i[mkz]', ]
            code_post = ['''__DEVICE__ __CONSTANT__ __FLOAT__ cst_cf_r[%s];
                            __DEVICE__ __CONSTANT__ __FLOAT__ cst_cf_i[%s];
                            __DEVICE__ __CONSTANT__ __FLOAT__ cst_ce_r[%s];
                            __DEVICE__ __CONSTANT__ __FLOAT__ cst_ce_i[%s];''' % tuple(self.materials_num + 1 for i in xrange(4)), \
                           'cst_cf_r[mkx]', 'cst_cf_r[mky]', 'cst_cf_r[mkz]', \
                           'cst_cf_i[mkx]', 'cst_cf_i[mky]', 'cst_cf_i[mkz]', \
                           'cst_ce_r[mkx]', 'cst_ce_r[mky]', 'cst_ce_r[mkz]', \
                           'cst_ce_i[mkx]', 'cst_ce_i[mky]', 'cst_ce_i[mkz]', ]
        else:
            code_prev = ['__DECLARE_CONSTANT_ARRAYS__']
            code_post = ['']

        code = template_to_code(fdtd.engine.templates['dispersive'], code_prev+fdtd.engine.code_prev, code_post+fdtd.engine.code_post)

        fdtd.engine.programs['dispersive'] = fdtd.engine.build(code)
        fdtd.engine.kernel_args['dr_pole'] = {}
        fdtd.engine.kernel_args['cp_pole'] = {}
        fdtd.engine.kernel_args['dispersive'] = {'dr_pole':{}, 'cp_pole':{}}

        if 'cuda' in fdtd.engine.name:
            self.cst_cf_r = fdtd.engine.get_global(fdtd.engine.programs['dispersive'], 'cst_cf_r')
            self.cst_cf_i = fdtd.engine.get_global(fdtd.engine.programs['dispersive'], 'cst_cf_i')
            self.cst_ce_r = fdtd.engine.get_global(fdtd.engine.programs['dispersive'], 'cst_ce_r')
            self.cst_ce_i = fdtd.engine.get_global(fdtd.engine.programs['dispersive'], 'cst_ce_i')

        for p in xrange(self.dpolenum_max):
            fdtd.engine.kernel_args['dr_pole'][p] = {'real':None, 'imag':None}
            fdtd.engine.kernel_args['dispersive']['dr_pole'][p] = {'real':None, 'imag':None}
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['dr_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                               np.int32(self.px), np.int32(self.py), np.int32(self.pz), \
                                                               self.mx.data, self.my.data, self.mz.data, \
                                                               fdtd.ex.__dict__[part].data, fdtd.ey.__dict__[part].data, fdtd.ez.__dict__[part].data, \
                                                               self.dr_fx_r[p].__dict__[part].data, self.dr_fy_r[p].__dict__[part].data, self.dr_fz_r[p].__dict__[part].data, \
                                                               self.dr_cf_r[p].data, self.dr_ce_r[p].data]
                fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                                             np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                                             np.int32(self.px), np.int32(self.py), np.int32(self.pz), \
                                                                             fdtd.ex.__dict__[part].data, fdtd.ey.__dict__[part].data, fdtd.ez.__dict__[part].data, \
                                                                             self.dr_fx_r[p].__dict__[part].data, self.dr_fy_r[p].__dict__[part].data, self.dr_fz_r[p].__dict__[part].data]
        for p in xrange(self.cpolenum_max):
            fdtd.engine.kernel_args['cp_pole'][p] = {'real':None, 'imag':None}
            fdtd.engine.kernel_args['dispersive']['cp_pole'][p] = {'real':None, 'imag':None}
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['cp_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                               np.int32(self.px), np.int32(self.py), np.int32(self.pz), \
                                                               self.mx.data, self.my.data, self.mz.data, \
                                                               fdtd.ex.__dict__[part].data, fdtd.ey.__dict__[part].data, fdtd.ez.__dict__[part].data, \
                                                               self.cp_fx_r[p].__dict__[part].data, self.cp_fy_r[p].__dict__[part].data, self.cp_fz_r[p].__dict__[part].data, \
                                                               self.cp_fx_i[p].__dict__[part].data, self.cp_fy_i[p].__dict__[part].data, self.cp_fz_i[p].__dict__[part].data, \
                                                               self.cp_cf_r[p].data, self.cp_cf_i[p].data, \
                                                               self.cp_ce_r[p].data, self.cp_ce_i[p].data  ]
                fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part] = [np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                                             np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                                             np.int32(self.px), np.int32(self.py), np.int32(self.pz), \
                                                                             fdtd.ex.__dict__[part].data, fdtd.ey.__dict__[part].data, fdtd.ez.__dict__[part].data, \
                                                                             self.cp_fx_r[p].__dict__[part].data, self.cp_fy_r[p].__dict__[part].data, self.cp_fz_r[p].__dict__[part].data]

        if 'opencl' in fdtd.engine.name:
            gs = cl_global_size(self.nx*self.ny*self.nz, fdtd.engine.ls)
            for p in xrange(self.dpolenum_max):
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['dr_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dr_pole'][p][part]
                    fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part]
            for p in xrange(self.cpolenum_max):
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['cp_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['cp_pole'][p][part]
                    fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part]

        else:
            gs = self.nx*self.ny*self.nz
            for p in xrange(self.dpolenum_max):
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['dr_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dr_pole'][p][part]
                    fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part]
            for p in xrange(self.cpolenum_max):
                for part in fdtd.complex_parts:
                    fdtd.engine.kernel_args['cp_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['cp_pole'][p][part]
                    fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part] = [fdtd.engine.queue, (gs,), (fdtd.engine.ls,)] + fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part]

        if 'opencl' in fdtd.engine.name:
            fdtd.engine.kernels['dr_pole']    = fdtd.engine.programs['dispersive'].update_dpole_3d
            fdtd.engine.kernels['cp_pole']    = fdtd.engine.programs['dispersive'].update_cpole_3d
            fdtd.engine.kernels['dispersive'] = fdtd.engine.programs['dispersive'].update_e_3d
        elif 'cuda' in fdtd.engine.name:
            fdtd.engine.kernels['dr_pole']    = fdtd.engine.get_function(fdtd.engine.programs['dispersive'], 'update_dpole_3d')
            fdtd.engine.kernels['cp_pole']    = fdtd.engine.get_function(fdtd.engine.programs['dispersive'], 'update_cpole_3d')
            fdtd.engine.kernels['dispersive'] = fdtd.engine.get_function(fdtd.engine.programs['dispersive'], 'update_e_3d')
            if self.dpolenum_max > 0:
                fdtd.engine.prepare(fdtd.engine.kernels['dr_pole']   , fdtd.engine.kernel_args['dr_pole'][0]['real'])
                fdtd.engine.prepare(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['dr_pole'][0]['real'])
            if self.cpolenum_max > 0:
                fdtd.engine.prepare(fdtd.engine.kernels['cp_pole']   , fdtd.engine.kernel_args['cp_pole'][0]['real'])
                fdtd.engine.prepare(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['cp_pole'][0]['real'])
        elif  'cpu' in fdtd.engine.name:
            fdtd.engine.kernels['dr_pole']    = fdtd.engine.set_kernel(fdtd.engine.programs['dispersive'].update_dpole_3d, fdtd.engine.kernel_args['dr_pole'][0]['real'])
            fdtd.engine.kernels['cp_pole']    = fdtd.engine.set_kernel(fdtd.engine.programs['dispersive'].update_cpole_3d, fdtd.engine.kernel_args['cp_pole'][0]['real'])
            fdtd.engine.kernels['dispersive'] = fdtd.engine.set_kernel(fdtd.engine.programs['dispersive'].update_e_3d    , fdtd.engine.kernel_args['dispersive']['dr_pole'][0]['real'])

    def update_pole(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for p in xrange(self.dpolenum_max):
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dr_pole'](*(fdtd.engine.kernel_args['dr_pole'][p][part]))
                elif 'cuda' in fdtd.engine.name:
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_cf_r[0], self.dr_cf_r[p].data, self.dr_cf_r[p].nbytes]).wait()
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_ce_r[0], self.dr_ce_r[p].data, self.dr_ce_r[p].nbytes]).wait()
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['dr_pole'], fdtd.engine.kernel_args['dr_pole'][p][part])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dr_pole'](*(fdtd.engine.kernel_args['dr_pole'][p][part][3:]))
                evts.append(evt)
        for p in xrange(self.cpolenum_max):
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['cp_pole'](*(fdtd.engine.kernel_args['cp_pole'][p][part]))
                elif 'cuda' in fdtd.engine.name:
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_cf_r[0], self.cp_cf_r[p].data, self.cp_cf_r[p].nbytes]).wait()
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_cf_i[0], self.cp_cf_i[p].data, self.cp_cf_i[p].nbytes]).wait()
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_ce_r[0], self.cp_ce_r[p].data, self.cp_ce_r[p].nbytes]).wait()
                    fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtod, [self.cst_ce_i[0], self.cp_ce_i[p].data, self.cp_ce_i[p].nbytes]).wait()
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['cp_pole'], fdtd.engine.kernel_args['cp_pole'][p][part])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['cp_pole'](*(fdtd.engine.kernel_args['cp_pole'][p][part][3:]))
                evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for p in xrange(self.dpolenum_max):
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dispersive'](*(fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part]))
                elif 'cuda' in fdtd.engine.name:
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dispersive'](*(fdtd.engine.kernel_args['dispersive']['dr_pole'][p][part][3:]))
                evts.append(evt)
        for p in xrange(self.cpolenum_max):
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dispersive'](*(fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part]))
                elif 'cuda' in fdtd.engine.name:
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['dispersive'], fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['dispersive'](*(fdtd.engine.kernel_args['dispersive']['cp_pole'][p][part][3:]))
                evts.append(evt)
        if True: wait_for_events(fdtd, evts)
        return evts

def Drude_CP_world(fdtd, (pt0, pt1), cp_materials):
    if '2D' in fdtd.mode:
        return Drude_CP_world_2d(fdtd, (pt0, pt1), cp_materials)
    if '3D' in fdtd.mode:
        return Drude_CP_world_3d(fdtd, (pt0, pt1), cp_materials)


# Metal (Drude_CP_materials)
# input parameters of Drude_pole: wfreq_p, gamma
# input parameters of    CP_pole: wfreq_p, gamma, amplitude, phase

# FORMAT by Ph.D Jong-Ho Choe:
# Case of Drude pole  -> Drude_pole(D1CP2.wp[1]   , D1CP2.gamma[1])
# Case of 1st CP_pole ->    CP_pole(D1CP2.OMEGA[1], D1CP2.GAMMA[1], D1CP2.A[1], D1CP2.PHI[1])
# Case of 2nd CP_pole ->    CP_pole(D1CP2.OMEGA[2], D1CP2.GAMMA[2], D1CP2.A[2], D1CP2.PHI[2])
# Material(D1CP2)     -> Drude_CP_material(D1CP2.EPS, 0., [Drude_pole], [1st_CP_pole, 2nd_CP_Pole])

#----------------------------------------- Gold ----------------------------------------#

# 1: Gold (Au), 250 nm ~ 1000 nm, ref. P. G. Etchegoin
Au_eps_r = 1.53
Au_sigma = 0.
Au_pole0 = Drude_pole(1.29907004642e16, 1.10803033371e14)
Au_pole1 =    CP_pole(4.02489651134e15, 8.18978942308e14, 0.94, -0.785398163397)
Au_pole2 =    CP_pole(5.69079023356e15, 2.00388464607e15, 1.36, -0.785398163397)
Au_D1CP2_250nm1000nm_Etchegoin = Drude_CP_material(1, Au_eps_r, Au_sigma, [Au_pole0], [Au_pole1, Au_pole2])

# 2: Gold (Au), 400 nm ~ 1000 nm, ref. Johnson and Christy, STerr: 0.0294043
Au_eps_r = 1.
Au_sigma = 0.
Au_pole0 = Drude_pole(1.3116170613829832e16, 1.150170034487477e14)
Au_pole1 =    CP_pole(4.706954941760446e15, 2.0986871407450945e15, 2.559635958760465, -0.9231394055396535)
Au_pole2 =    CP_pole(3.9510687439963015e15, 4.759121356224116e14, -0.29813628368518946, -3.942078027009713)
Au_D1CP2_400nm1000nm_Johnson = Drude_CP_material(2, Au_eps_r, Au_sigma, [Au_pole0], [Au_pole1, Au_pole2])

# 3: Gold (Au), 300 nm ~ 900 nm, ref. Johnson and Christy, STerr: 0.0313929
Au_eps_r = 2.5
Au_sigma = 0.
Au_pole0 = Drude_pole(1.3148109152556018e16, 9.505359374012766e13)
Au_pole1 =    CP_pole(6.315056318675713e15, 2.2162289463678742e15, 1.5340437478846873, 0.034309981641033445)
Au_pole2 =    CP_pole(3.9459276514390085e15, 7.135781715940518e14, -0.7120539753544073, -4.169089425678387)
Au_D1CP2_300nm900nm_Johnson = Drude_CP_material(3, Au_eps_r, Au_sigma, [Au_pole0], [Au_pole1, Au_pole2])

# 4: Gold (Au), 1000 nm ~ 3000 nm, ref. Palik, STerr: 180.603
Au_eps_r = 1.
Au_sigma = 0.
Au_pole0 = Drude_pole(1.2058181639891528e16, 1.1999034535903264e14)
Au_pole1 =    CP_pole(1.e14, 1.e16, 0., 0.)
Au_pole2 =    CP_pole(1.e15, 1.e16, 0., 0.)
Au_D1CP2_1000nm3000nm_Palik = Drude_CP_material(4, Au_eps_r, Au_sigma, [Au_pole0], [Au_pole1, Au_pole2])

# 5: Gold (Au), 300 nm ~ 1700 nm, ref. McPeak, STerr: 0.0475981
Au_eps_r = 4.065591832582097
Au_sigma = 0.
Au_pole0 = Drude_pole(1.4087158159423078e16, 4.649527376536535e13)
Au_pole1 =    CP_pole(4.0157110418091255e15, 6.218088670598591e14, 0.6369181728688444, -1.0091644728455471)
Au_pole2 =    CP_pole(6.435237140411385e15, 1.8055075391571838e15, 1.6994546225231348, 6.51888116560112)
Au_D1CP2_300nm1700nm_Palik = Drude_CP_material(5, Au_eps_r, Au_sigma, [Au_pole0], [Au_pole1, Au_pole2])

# 6: Gold (Au), 206.6 nm ~ 12400 nm, ref. Babar and Weaver, STerr: 2.49078
Au_eps_r = 2.0223236158791513
Au_sigma = 0.
Au_pole0 = Drude_pole(1.2995914102472828e16, 3.2659275191010527e13)
Au_pole1 =    CP_pole(4.005464310109812e15, 1.4884255103289445e15, -2.580163232352201, 1.8690249756854396)
Au_pole2 =    CP_pole(6.890124349847865e14, 1.1871787062445868e15, 4.438035804798203, 5.622962858101441)
Au_D1CP2_206nm12um_Palik = Drude_CP_material(6, Au_eps_r, Au_sigma, [Au_pole0], [Au_pole1, Au_pole2])

# 7: Gold (Au), 300 nm ~ 24 um, ref. Olmon, STerr: 676.552
Au_eps_r = -4.133785382659285
Au_sigma = 0.
Au_pole0 = Drude_pole(1.4307700540284808e16,6.088899374622031e13)
Au_pole1 =    CP_pole( 4.629257793746139e13, 2.55563564740321e13, 8377.88033429251, 3.1455746041267716)
Au_D1CP1_300nm24um_Olmon = Drude_CP_material(7, Au_eps_r, Au_sigma, [Au_pole0], [Au_pole1])

#----------------------------------------- Silver ----------------------------------------#

# 8: Silver (Ag), 400 nm ~ 800 nm, ref. Dominique Barchiesi
Ag_eps_r = -19.4464
Ag_sigma = 0.
Ag_pole0 = Drude_pole(1.32472e16, 9.52533e13)
Ag_pole1 =    CP_pole(3.47550e16, 5.18324e16, 1727.14, 1.04286)
Ag_pole2 =    CP_pole(4.76055e16, 3.23542e16, -742.061, 1.75438)
Ag_D1CP2_400nm800nm_Dominique = Drude_CP_material(8, Ag_eps_r, Ag_sigma, [Ag_pole0], [Ag_pole1, Ag_pole2])

# 9: Silver (Ag), 300 nm ~ 900 nm, ref. Johnson, STerr: 0.0271776
Ag_eps_r = 1.
Ag_sigma = 0.
Ag_pole0 = Drude_pole(1.394932972644896e16, 1.7781630288180668e13)
Ag_pole1 =    CP_pole(6.222080372689097e15, 5.960612191043039e14, 0.284373188260343, 5.106945466121798)
Ag_pole2 =    CP_pole(1.217425709766018e16, 1.2152751718223278e16, 2.0963112499088674, -0.23587497382082553)
Ag_D1CP2_300nm900nm_Johnson = Drude_CP_material(9, Ag_eps_r, Ag_sigma, [Ag_pole0], [Ag_pole1, Ag_pole2])

# 10: Silver (Ag), 300 nm ~ 900 nm, ref. Johnson, loss, STerr: 0.0271776
Ag_eps_r = 1.
Ag_sigma = 0.
Ag_pole0 = Drude_pole(1.394932972644896e16, 10*1.7781630288180668e13)
Ag_pole1 =    CP_pole(6.222080372689097e15, 5.960612191043039e14, 0.284373188260343, 5.106945466121798)
Ag_pole2 =    CP_pole(1.217425709766018e16, 1.2152751718223278e16, 2.0963112499088674, -0.23587497382082553)
Ag_D1CP2_300nm900nm_JohnsonLoss = Drude_CP_material(10, Ag_eps_r, Ag_sigma, [Ag_pole0], [Ag_pole1, Ag_pole2])

# 11: Silver (Ag), 370 nm ~ 900 nm, ref. Winsemius, STerr: 0.0147798
Ag_eps_r = 4.125
Ag_sigma = 0.
Ag_pole0 = Drude_pole(1.3462259458308412e16, 5.191123309227644e13)
Ag_pole1 =    CP_pole(3.397350990276859e14, 1.4256692514480055e15, 3.988543261091087, 4.693855988590337)
Ag_pole2 =    CP_pole(1.011e15, 1.e15, 0, 0)
Ag_D1CP2_370nm900nm_Winsemius = Drude_CP_material(10, Ag_eps_r, Ag_sigma, [Ag_pole0], [Ag_pole1, Ag_pole2])

# 12: Silver (Ag), 370 nm ~ 900 nm, ref. Winsemius, STerr: 0.0147798
Ag_eps_r = 1.
Ag_sigma = 0.
Ag_pole0 = Drude_pole(1.37e16, 8.5e13)
Ag_pole1 =    CP_pole(1.011e15, 1.e15, 0, 0)
Ag_pole2 =    CP_pole(1.011e15, 1.e15, 0, 0)
Ag_D1CP2_370nm900nm_Winsemius = Drude_CP_material(11, Ag_eps_r, Ag_sigma, [Ag_pole0], [Ag_pole1, Ag_pole2])

# 13: Silver (Ag), 400 nm, ref. Palik, monopole
Ag_eps_r = 1.
Ag_sigma = 0.
Ag_pole0 = Drude_pole(1.03905e16, 6.65967e14)
Ag_pole1 =    CP_pole(0, 0, 0, 0)
Ag_pole2 =    CP_pole(0, 0, 0, 0)
Ag_D1CP2_400nmm_Palik = Drude_CP_material(12, Ag_eps_r, Ag_sigma, [Ag_pole0], [Ag_pole1, Ag_pole2])

#----------------------------------------- Aluminium ----------------------------------------#

# 14: Aluminum (Al), 300 nm ~ 900 nm, ref. Rakic, STerr: 0.137391
Al_eps_r = 1.
Al_sigma = 0.
Al_pole0 = Drude_pole(2.0356022851971e16, 2.1558899154101e14)
Al_pole1 =    CP_pole(2.2590644979400e15, 3.5735790649025e14, 5.6588777555160, -0.4850996787410)
Al_pole2 =    CP_pole(2.9387088835961e15, 1.4710570697224e15, 3.1116433236877, 0.5083054567608)
Al_D1CP2_300nm900nm_Rakic = Drude_CP_material(13, Al_eps_r, Al_sigma, [Al_pole0], [Al_pole1, Al_pole2])

# 15: Aluminum (Al), 300 nm ~ 900 nm, ref. Rakic, STerr: 0.232058
Al_eps_r = 1.
Al_sigma = 0.
Al_pole0 = Drude_pole(2.366522367001322e16, 9.655523715011404e14)
Al_pole1 =    CP_pole(2.218e15, 3.781e14, 0, 0)
Al_pole2 =    CP_pole(9.454e15, 6.421e15, 0, 0)
Al_D1CP2_300nm900nm_Rakic_1 = Drude_CP_material(14, Al_eps_r, Al_sigma, [Al_pole0], [Al_pole1, Al_pole2])

# 16: Aluminum (Al), 300 nm ~ 900 nm, ref. Palik
Al_eps_r = 2.2
Al_sigma = 0.
Al_pole0 = Drude_pole(2.075493476122525e16, 2.692830652318520e14)
Al_pole1 =    CP_pole(2.218812242846060e15, 3.781756919663996e14, 6.986369587800884, -0.690883344268221)
Al_pole2 =    CP_pole(9.454315381674146e15, 6.421007308803584e15, 2.971741559189552, 3.167232355361811)
Al_D1CP2_300nm900nm_Palik = Drude_CP_material(15, Al_eps_r, Al_sigma, [Al_pole0], [Al_pole1, Al_pole2])

# 17: Aluminum (Al), 200 nm ~ 1000 nm, ref. Vial
Al_eps_r = 1.0
Al_sigma = 0.
Al_pole0 = Drude_pole(2.0598e16, 2.2876e14)
Al_pole1 =    CP_pole(2.2694e15, 3.2867e14, 5.2306, -0.51202)
Al_pole2 =    CP_pole(2.4668e15, 1.7731e15, 5.2704, 0.42503)
Al_D1CP2_200nm1000nm_Vial = Drude_CP_material(16, Al_eps_r, Al_sigma, [Al_pole0], [Al_pole1, Al_pole2])

# 18: Aluminum (Al), 2500 nm ~ 20 um, ref. Rakic, STerr: 4339.99, not proved yet 
Al_eps_r = 1.
Al_sigma = 0.
Al_pole0 = Drude_pole(6.98085692248606e14, 4.983166394818760e15)
Al_pole1 =    CP_pole(8.063815654482060e15, 1.3570087005378300e16, -9.23112651799599, 0.373580469857281)
Al_pole2 =    CP_pole(4.023522533464460e15, 7.770734838045830e15, -3.9247032358421, 1.9889191876293132)
Al_D1CP2_2500nm20um_Rakic = Drude_CP_material(17, Al_eps_r, Al_sigma, [Al_pole0], [Al_pole1, Al_pole2])

# 19: Aluminum 203 (Al203), 200 nm ~ 2000 nm
Al203_eps_r = 10.
Al203_sigma = 0.
Al203_pole0 = Drude_pole(9.477738681090427e13, 1.8110413855758624e16)
Al203_pole1 =    CP_pole(1.2913060603756078e17, 87098.34804175566, 138.81050697190886, 2.136541683106582)
Al203_pole2 =    CP_pole(8.930811378051955e13, 3.16432230299051e13, 1282.8390489461387, -2.6149060674484863)
Al203_D1CP2_200nm2um = Drude_CP_material(18, Al203_eps_r, Al203_sigma, [Al203_pole0], [Al203_pole1, Al203_pole2])

# 20: Aluminum (Al), 300 nm ~ 900 nm
Al203_eps_r = 2.
Al203_sigma = 0.
Al203_pole0 = Drude_pole(2.0383455518558504e16, 2.5497979690494766e14)
Al203_pole1 =    CP_pole(2.286120027634404e15, 3.7882335531953e14, 6.443269968772944, 5.862266167897745)
Al203_pole2 =    CP_pole(2.124386766566152e22, 7.633967261901861e15, 4.373314452470692, 1.8832236656967871)
Al203_D1CP2_300nm900nm = Drude_CP_material(19, Al203_eps_r, Al203_sigma, [Al203_pole0], [Al203_pole1, Al203_pole2])

#----------------------------------------- Aluminium Nitride ----------------------------------------#

# 21: Aluminium Nitride (AlN), 1.54 um ~ 14.28 um, ref. Kischkat
AlN_epsr_inf = 4.
AlN_simga = 0.
AlN_DP  = Drude_pole(4.8186750800584776e16, 1.4506957288892336e19)
AlN_CP1 = CP_pole(1.3457428585331106e14, 1.8351940983214508e13,-1.282652694844099, -2.719717524061664)
AlN_D1CP1_1540nm14280nm_Kischkat = Drude_CP_material(20, AlN_epsr_inf, AlN_sigma, [AlN_DP], [AlN_CP1])

#----------------------------------------- Silicon ----------------------------------------#

# 22: Silicon (Si), 300 nm ~ 800 nm, ref. Palik, STerr: 0.944317
Si_eps_r = 2.
Si_sigma = 0.
Si_pole0 = Drude_pole(0., 0.)
Si_pole1 =    CP_pole(5.115832370098545e15, 2.418956338146272e14, -1.3281240719318108, 2.3078590980047147)
Si_pole2 =    CP_pole(6.492623507279991e15, 7.795088221370624e14, -4.410075824564194 , 3.290102642882403)
Si_D1CP2_300nm800nm_Palik = Drude_CP_material(21, Si_eps_r, Si_sigma, [Si_pole0], [Si_pole1, Si_pole2])

# 23: Silicon (Si), 300 nm ~ 800 nm, ref. Palik, STerr: 0.944317, with new fitting parameters
Si_eps_r = 4.
Si_sigma = 0.
Si_pole0 = Drude_pole(1.1231275999647966e10, 8.335330727043009e15)
Si_pole1 =    CP_pole(5.108836820561175e15, 2.563237475701456e14, -1.4327321393565065, 2.2545987900496454)
Si_pole2 =    CP_pole(6.579562414662795e15, 6.484860512520668e14, -3.93520817586869, -2.7762596154236685)
Si_D1CP2_300nm800nm_Palik_2 = Drude_CP_material(22, Si_eps_r, Si_sigma, [Si_pole0], [Si_pole1, Si_pole2])

# 24: Crystalline silicon (Si), 200 nm ~ 2000 nm
Si_eps_r = 1.
Si_sigma = 0.
Si_pole0 = Drude_pole(1.692278822642980e15, 1.89381876845954e3)
Si_pole1 =    CP_pole(5.102676157839700e15, 2.18366306151624e14, -1.27932362167047, 2.31431929891286)
Si_pole2 =    CP_pole(6.398823419340790e15, 8.48481229311499e14, -4.73795481342157, 3.15695601979578)
Si_D1CP2_200nm2000nm = Drude_CP_material(23, Si_eps_r, Si_sigma, [Si_pole0], [Si_pole1, Si_pole2])

# 25: Amorphous Silicon (a-Si), 300 nm ~ 800 nm, ref. Sopra
aSi_eps_r = 1.
aSi_sigma = 0.
aSi_pole0 = Drude_pole(1.850435611831103e13, 9.583406339327194e15)
aSi_pole1 =    CP_pole(5.324466171382987e15, 1.733426422846888e15, -10.870449387447888, 2.944731850871322)
aSi_pole2 =    CP_pole(2.323289363616541e15, 1.8167392076732152e15, -5.18018639132158, -0.9939548517326946)
aSi_D1CP2_300nm800nm_Sopra = Drude_CP_material(24, aSi_eps_r, aSi_sigma, [aSi_pole0], [aSi_pole1, aSi_pole2])

# 26: Amorphous Silicon (a-Si), 300 nm ~ 800 nm, ref. Pierce
aSi_eps_r = 1.
aSi_sigma = 0.
aSi_pole0 = Drude_pole(0, 8.061205596111416e15)
aSi_pole1 =    CP_pole(4.94171527990468e15, 1.8595339509450368e15, -9.10222319585717, 2.8453371370897904)
aSi_pole2 =    CP_pole(1.0072165244475052e15, 1.575944678802349e15, -11.625322053622188, -2.1262302665474273)
aSi_D1CP2_300nm800nm_Pierce = Drude_CP_material(25, aSi_eps_r, aSi_sigma, [aSi_pole0], [aSi_pole1, aSi_pole2])

# 27: Amorphous Silicon (a-Si), 300 nm ~ 800 nm
aSi_eps_r = 2.738587518697226
aSi_sigma = 0.
aSi_pole0 = Drude_pole(1.6139197453237578e15, 4.111985744321898e15)
aSi_pole1 =    CP_pole(4.773078399299514e15, 5.02709596296994e14, -3.1530715132508322, 3.077898703926377)
aSi_pole2 =    CP_pole(2.5409853403772105e15, 1.9569351317100125e15, -0.7918925733319867, -0.14546067567560064)
aSi_D1CP2_300nm800nm = Drude_CP_material(26, aSi_eps_r, aSi_sigma, [aSi_pole0], [aSi_pole1, aSi_pole2])

#----------------------------------------- Nickel ----------------------------------------#

# 28: Nickel (Ni), 300 nm ~ 900 nm, Palik, STerr: 0.0141569
Ni_eps_r = 1.
Ni_sigma = 0.
Ni_pole0 = Drude_pole(1.325874380259268e16, 1.442377560503351e15)
Ni_pole1 =    CP_pole(9.690661318230162e15, 4.816375073060784e15, 3.214814494683990, 0.428870654403958)
Ni_pole2 =    CP_pole(1.977760585715344e15, 6.261403123618544e14, 2.663011434478377, -1.277875400574552)
Ni_D1CP2_300nm900nm_Palik = Drude_CP_material(27, Ni_eps_r, Ni_sigma, [Ni_pole0], [Ni_pole1, Ni_pole2])

# 29: Nickel (Ni), 300 nm ~ 900 nm, Palik, STerr: 0.0139261
Ni_eps_r = 2.1
Ni_sigma = 0.
Ni_pole0 = Drude_pole(1.33501820171681e16, 1.49754687584861e15)
Ni_pole1 =    CP_pole(1.10548790949880e16, 5.31430242142796e15, 4.00077466183896, 0.71981478102242)
Ni_pole2 =    CP_pole(2.01517491930328e15, 6.12440241489153e14, 2.31128985637125, -1.24412187290537)
Ni_D1CP2_300nm900nm_Palik_2 = Drude_CP_material(28, Ni_eps_r, Ni_sigma, [Ni_pole0], [Ni_pole1, Ni_pole2])

# 30: Nickel (Ni), 300 nm ~ 900 nm, Palik, STerr: 0.0139261
Ni_eps_r = 5.
Ni_sigma = 0.
Ni_pole0 = Drude_pole(1.7419885228910362e16, 3.8420271476535495e15)
Ni_pole1 =    CP_pole(1.1e16, 5.31e15, 0., 0.)
Ni_pole2 =    CP_pole(2.01e15, 6.12e14, 0., 0.)
Ni_D1CP2_300nm900nm_Palik = Drude_CP_material(29, Ni_eps_r, Ni_sigma, [Ni_pole0], [Ni_pole1, Ni_pole2])

#----------------------------------------- Platinum ----------------------------------------#

# 31: Platinum (Pt), 300nm ~ 1000 nm, ref. Palik, STerr: 0.0103642
Pt_eps_r = 1.2528
Pt_sigma = 0.
Pt_pole0 = Drude_pole(1.468666843162885e16, 1.723657769636042e15)
Pt_pole1 =    CP_pole(9.859574888333026e14, 2.123559785624624e15, 21.24739472151703, -0.8927848047859648)
Pt_pole2 =    CP_pole(2.e15, 1.e15, 0., 0.)
Pt_D1CP2_300nm1000nm_Palik = Drude_CP_material(30, Pt_eps_r, Pt_sigma, [Pt_pole0], [Pt_pole1, Pt_pole2])

#----------------------------------------- Chromium ----------------------------------------#

# 32: Chromium (Cr), 400 nm ~ 1000 nm, ref. Vial, STerr: 0.25875
Cr_eps_r = 1.1297
Cr_sigma = 0.
Cr_pole0 = Drude_pole(8.8128e15, 3.8828e14)
Cr_pole1 =    CP_pole(1.7398e15, 1.6329e15, 33.086, -0.25722)
Cr_pole2 =    CP_pole(3.7925e15, 7.3567e14, 1.6592, 0.83533)
Cr_D1CP2_400nm1000nm_Vial = Drude_CP_material(31, Cr_eps_r, Cr_sigma, [Cr_pole0], [Cr_pole1, Cr_pole2])

# 33: Chromium (Cr), 300 nm ~ 800 nm, ref. Palik
# NOTE: DCP2 model cannot make perfect fitting on Cr, but with this parameters, simulation does not blow up.
Cr_eps_r = 1.
Cr_sigma = 0.
Cr_pole0 = Drude_pole(1.4189265790760204e16, 2.1135321666341358e15)
Cr_pole1 =    CP_pole(2.1115486212815998e15, 1.0809212270681449e15, 7.149177217361324, 10.862707457965554)
Cr_pole2 =    CP_pole(3.37682808441949e15, 5.1898621522682756e14, -0.7242878504699284, 2.397841968339894)
Cr_D1CP2_300nm800nm_Palik = Drude_CP_material(32, Cr_eps_r, Cr_sigma, [Cr_pole0], [Cr_pole1, Cr_pole2])

#----------------------------------------- Copper ----------------------------------------#

# 34: Copper (Co), 400nm ~ 1000 nm, ref. Rakic and Ordal
# NOTE: Not tested yet
Cu_eps_r = 1.1297
Cu_sigma = 0.
Cu_pole0 = Drude_pole(1.3053428765243808e16, 1.7209298246055475e14)
Cu_pole1 =    CP_pole(3.5522820868727255e15, 8.54988679632223e14, -1.183257295364979, -4.4491395169549754)
Cu_pole2 =    CP_pole(1.2414779093214122e17, 0., -27.889005704684465, -1.6081249799192)
Cu_D1CP2_400nm1000nm_Rakic = Drude_CP_material(33, Cu_eps_r, Cu_sigma, [Cu_pole0], [Cu_pole1, Cu_pole2])

# 35: Copper (Co), 300nm ~ 1700 nm, ref. McPeak, STerr: 0.0464957
Cu_eps_r = 13.37811731156239
Cu_sigma = 0.
Cu_pole0 = Drude_pole(1.3639905771844794e16, 3.7709039157828836e13)
Cu_pole1 =    CP_pole(1.2955469681381042e16, 3.5841206815875215e15, 6.694932194613791, 1.8582049652019874)
Cu_pole2 =    CP_pole(3.397274585407835, 3.8106975921621424e14, 0.49019890323739823, 4.912625612795645)
Cu_D1CP2_300nm1700nm_McPeak = Drude_CP_material(34, Cu_eps_r, Cu_sigma, [Cu_pole0], [Cu_pole1, Cu_pole2])

#----------------------------------------- PbSe ----------------------------------------#

# 36: PbSe (by triumph)
PbSe_eps_r = -5.
PbSe_sigma =  0.
PbSe_pole0 = Drude_pole(1.7414385557381866e14, 3.052157269596333e16)
PbSe_pole1 =    CP_pole( 8.807042221568294e14, 8.807042221568294e14, 5.333166601595723, -0.5521547355558631)
PbSe       = Drude_CP_material(35, PbSe_eps_r, PbSe_sigma, [PbSe_pole0], [PbSe_pole1])

#----------------------------------------- Perfect Absorber ----------------------------------------#

# 37: Perfect Absorber (by triumph)
PA_eps_r = 1.0
PA_sigma = 0.0
PA_pole0 = Lorentz_pole(4520469903415956.0, 343873311189568.5, 0.88297503638502084)
PA_pole1 = Lorentz_pole(4826330019965662.0, 202736289826248.88, 0.40806050170920971)
PA_pole2 = Lorentz_pole(4982534240577065.0, 64945076416321.305, 0.12162449163805744)
PA_pole3 = Lorentz_pole(3611713050637068.0, 452597112806037.94, 2.2962686900550193)
PA_pole4 = Lorentz_pole(4096009523077053.0, 451026879333732.75, 1.5979295608383155)
PA_pole5 = Lorentz_pole(3218269729949211.0, 283317005775503.75, 1.6634307663587071)
PA_pole6 = Lorentz_pole(3063891792899861.5, 64034556040084.125, 0.44100321223323763)
PA = Drude_CP_material(36, PA_eps_r, PA_sigma, [], PA_pole0 + PA_pole1 + PA_pole2 + PA_pole3 + PA_pole4 + PA_pole5 + PA_pole6)

#bl = h5.File(os.path.dirname(os.path.abspath(__file__)) + '/material_data/black_metal.h5')
#bl_params = np.array(bl['lorentz'])
#bl_eps_r = float(np.array(bl['epr_inf']))
#bl_sigma = 0.
#bl_poles = []
#for p in xrange(np.shape(bl_params)[0]):
#    bl_poles += Lorentz_pole(*(bl_params[p]))
#PA = Drude_CP_material(10, bl_eps_r, bl_sigma, [], bl_poles)

#----------------------------------------- MoS2 ----------------------------------------#

# 38: MoS2 (MoS2), 600nm ~ 700 nm, ref. Shen, C.-C.,et. al. Applied Physics Express, 6, 125801. (2013).
MoS2_eps_r = 1.7877878776532141
MoS2_sigma =  0.
MoS2_pole0 = Drude_pole(5.042429410702844e16, 1.8700838553056144e17)
MoS2_pole1 =    CP_pole(4.3074125920431435e15, 3.8927228925060125e14, 4.305335953507419, -6.5847621217867065)
MoS2_pole2 =    CP_pole(2.811287665167908e15, 1.990435384450116e14, 1.3712296287344166, 4.951711455269794)
MoS2_D1CP2_600nm700nm_Shen = Drude_CP_material(37, MoS2_eps_r, MoS2_sigma, [], [MoS2_pole1,MoS2_pole2])

# 39: MoS2 (MoS2)
MoS2_eps_r = 1.
MoS2_sigma =  0.
MoS2_pole0 = Drude_pole(0.   , 0.)
MoS2_pole1 =    CP_pole(2.83493788897e+15, 6.50424227469e+13, 0.360919616535, 0.605278307226)
MoS2_pole2 =    CP_pole(3.06007813397e+15, 1.14319710908e+14, 0.529569141526, 0.430471812178)
MoS2_pole3 =    CP_pole(3.44290314872e+15, 2.11501781005e+14, 0.295530592753, 0.972296610952)
MoS2_pole4 =    CP_pole(4.3173327284e+15 , 3.64019522117e+14, 3.81014523992, 0.266919830913)
MoS2_pole5 =    CP_pole(5.74240560652e+15, 1.14251040018e+14, 2.63093353094, -0.440206526296)
MoS2_D1CP5 = Drude_CP_material(38, MoS2_eps_r, MoS2_sigma, [MoS2_pole0], [MoS2_pole1,MoS2_pole2,MoS2_pole3,MoS2_pole4,MoS2_pole5])

#----------------------------------------- ITO ----------------------------------------#

# 40: Indium Tin Oxide (ITO)
ITO_epsr_inf = 4.255350996093319
ITO_sigma = 0.
ITO_pole0 = Drude_pole(3.0626827828368245e15, 2.6936428300031965e13) 
ITO_pole1 = CP_pole(7.23165662351179e15, 9.314219696115896e14, 0., 2.2404322429544323) 
ITO_pole2 = CP_pole(1.9785002624362748e15, 1.0395814917565928e16, 0.,3.5752330243400845) 
ITO_D1CP2 = Drude_CP_material(39, ITO_epsr_inf, ITO_sigma, [ITO_pole0], [ITO_pole1, ITO_pole2])

#----------------------------------------- Gallium Phophide ----------------------------------------#

# 41: Gallium Phosphide (GaP)
GaP_epsr_inf = 6.5
GaP_sigma = 0.0
GaP_pole0 = Drude_pole(    2327669176.17035, 7.992844208248990e15)
GaP_pole1 =    CP_pole(5.594796671965160e15, 2.905883763421080e14,-1.08334463810630, 2.663199256740390)
GaP_pole2 =    CP_pole(6.884406554806240e15, 1.267549774767240e13, 1.20593067044096, 0.765913138789777)
GaP = Drude_CP_material(40, GaP_epsr_inf, GaP_sigma, [GaP_pole0], [GaP_pole1, GaP_pole2])

#----------------------------------------- InGaSb ----------------------------------------#

# 42: InGaSb, 255 nm ~ 825 nm, ref. Sopra, STerr: 0.290245
InGaSb_epsr_inf = 0.4245947499534844
InGaSb_sigma = 0.
InGaSb_pole0 = Drude_pole(6.228806414751226e16, 6.304905793461495e17)
InGaSb_pole1 =    CP_pole(5.800276353106057e15, 7.318921625021468e14, 1.995660040154981, 0.040558401799376485)
InGaSb_pole2 =    CP_pole(3.415905557537589e15, 4.049826284325365e14, -1.6685242626065668, 1.407052204849289)
InGaSb_pole3 =    CP_pole(-2.957510745135073, 4.214099138920396, 3.2897751520872247, 0.3037862333026691)
InGaSb_255nm825nm_Sopra = Drude_CP_material(41, InGaSb_epsr_inf, InGaSb_sigma_, [InGaSb_pole0], [InGaSb_pole1, InGaSb_pole2, InGaSb_pole3])

#----------------------------------------- GaAs ----------------------------------------#

# 43: Gallium Arsenide(GaAs), 206.6 nm ~ 826.6 nm, ref. Aspnes, STerr: 1.65775
GaAs_epsr_inf = 2.0634872506626794
GaAs_sigma = 0.
GaAs_pole0 = Drude_pole( 2.0634872506626794, 2.2945067733253125e18 )
GaAs_pole1 =    CP_pole( 7.156831516453248e15,6.999396526873551e14 , 2.086208045906452, 0.07399528889038613)
GaAs_pole2 =    CP_pole(4.501884786419084e15 ,6.128985482214921e14 ,2.325011537083982, -0.6311288346112683)
GaAs_206nm826nm_Aspnes = Drude_CP_material(42, GaAs_epsr_inf, GaAs_sigma, [GaAs_pole0], [GaAs_pole1, GaAs_pole2] )

# 44: Gallium Arsenide(GaAs), 234 nm ~ 840 nm, ref. Jellison, STerr: 1.01481
GaAs_epsr_inf = 3.3554089523436685
GaAs_sigma = 0.
GaAs_pole0 = Drude_pole(7.296506519326266e16 , 2.036238322536015e18)
GaAs_pole1 =    CP_pole( 7.339898215649058e15, 7.218749833976028e14, 2.4670894733989615, 0.2969393709825485)
GaAs_pole2 =    CP_pole( 4.521068160233081e15, 5.230986208323917e14, 1.8938811614297584, 5.6781640604217065)
GaAs_234nm840nm_Jellison = Drude_CP_material(43, GaAs_epsr_inf, GaAs_sigma, [GaAs_pole0], [GaAs_pole1, GaAs_pole2] )

#----------------------------------------- SiO2 ----------------------------------------#

# 45: SiO2, 1.54 um ~ 14.29 um, ref. Kischkat, STerr: 0.0418849
SiO2_epsr_inf = 2.780430025713438 
SiO2_sigma = 0.
SiO2_pole0 = Drude_pole(2.7959713843290903e14 ,7.136150373297847e15 )
SiO2_pole1 =    CP_pole(1.9835572406402744e14 ,6.128734103254499e12 , 0.322027387712222, -6.259413858758254)
SiO2_pole2 =    CP_pole(1.990875133340747e16 ,2.53363960345988e16 ,0.5991593734550135 , 2.0181788303423245 )
SiO2_1540nm14um_Aspnes = Drude_CP_material(44, SiO2_epsr_inf, SiO2_sigma, [SiO2_pole0], [SiO2_pole1, SiO2_pole2] )

# 46: SiO2, 252 nm ~ 1250 nm, ref. Gao, STerr: 
SiO2_epsr_inf = 2.223867208309895
SiO2_sigma = 0.
SiO2_pole0 = Drude_pole(4.54662564443799e14 ,2.2034472743021878e11 )
SiO2_pole2 =    CP_pole(0 ,0 ,0 ,0 )
SiO2_D1CP1_252nm1250nm_Aspnes = Drude_CP_material(45, SiO2_epsr_inf, SiO2_sigma, [SiO2_pole0], [SiO2_pole1] )

# 47: SiO2, 250 nm ~ 2500 nm, ref. Lemarchand, STerr: 4.33354e-7
SiO2_epsr_inf = -0.3155816233900237
SiO2_sigma = 0.
SiO2_pole0 = Drude_pole(5.5173661201265625e13 , 3.7893521318777465e15)
SiO2_pole1 =    CP_pole( 6.093782748415532e15, 1.1191939194201606e16, 3.6386279673194375, -0.8959781331070341)
SiO2_pole2 =    CP_pole( 1.2711861270290458e16, 1.226075567496467e16, 3.840548646916963, -2.5137697949848317)
SiO2_D1CP2_250nm2500nm_Aspnes = Drude_CP_material(46, SiO2_epsr_inf, SiO2_sigma, [SiO2_pole0], [SiO2_pole1, SiO2_pole2] )

#----------------------------------------- SiO2 ----------------------------------------#

# 48: SiO2, 250 nm ~ 2500 nm, ref. Lemarchand, STerr: 4.33354e-7
SiO2_epsr_inf = -0.3155816233900237
SiO2_sigma = 0.
SiO2_pole0 = Drude_pole(5.5173661201265625e13 , 3.7893521318777465e15)
SiO2_pole1 =    CP_pole( 6.093782748415532e15, 1.1191939194201606e16, 3.6386279673194375, -0.8959781331070341)
SiO2_pole2 =    CP_pole( 1.2711861270290458e16, 1.226075567496467e16, 3.840548646916963, -2.5137697949848317)
SiO2_D1CP2_250nm2500nm_Aspnes = Drude_CP_material(47, SiO2_epsr_inf, SiO2_sigma, [SiO2_pole0], [SiO2_pole1, SiO2_pole2] )

#----------------------------------------- SiC ----------------------------------------#

# 49: SiO2, 100 nm ~ 300 nm, ref. Larruquert, STerr: 0.109679
SiC_epsr_inf = 0.6043976609288952 
SiC_sigma = 0.
SiC_pole0 = Drude_pole( 1.7812535359896454e10, 8.1392469834521e15)
SiC_pole1 =    CP_pole(8.210676714469327e15, 4.2698438090753775e15, -6.519889806601736, 2.994998480906719)
SiC_D1CP1_100nm300nm_Larruquert = Drude_CP_material(48, SiC_epsr_inf, SiC_sigma, [SiC_pole0], [SiC_pole1] )

# 50: SiO2, 300 nm ~ 800 nm, ref. Larruquert, STerr: 0.0588523 
SiC_epsr_inf = 10.800365086473315 
SiC_sigma = 0.
SiC_pole0 = Drude_pole(3.897078764332207e16, 8.806292440240474e17)
SiC_pole1 =    CP_pole(0., 0., 0., 0.)
SiC_D1_300nm800nm_Larruquert = Drude_CP_material(49, SiC_epsr_inf, SiC_sigma, [SiC_pole0], [SiC_pole1] )

# 51: SiO2, 1.2 um ~ 60 um, ref. Larruquert, STerr: 
SiC_epsr_inf = 10.111451002371808
SiC_sigma = 0.
SiC_pole0 = Drude_pole( 3.6640681357061175e14, 6.084062046034084e13 )
SiC_pole1 =    CP_pole(0., 0., 0., 0.)
SiC_D1_1200nm60um_Larruquert = Drude_CP_material(50, SiC_epsr_inf, SiC_sigma, [SiC_pole0], [SiC_pole1] )

# 52: C(Graphene), 210nm ~ 1000nm, ref: Weber
C_graphene_inf = 3.626775501541622
C_graphene_sigma = 0.
C_graphene_pole0 = Drude_pole(1.058957872899501e17, 5.7398482416045037e17)
C_graphene_pole1 = CP_pole(5.6825270150584e15, 7.209872451587759e14, 0.5027889733537447,3.2744241060156174)
C_graphene_pole2 = CP_pole(6.716429181412444e15, 7.766711026836548e14, 1.7560853571632558, 0.45309050353453645)
C_graphene_210nm1000nm_Weber = Drude_CP_material(51, C_graphene_inf, C_graphene_sigma, [C_graphene_pole0, C_graphene_pole1, C_graphene_pole2])

# 53: InP(Indium Phosphide, 210nm~830nm, ref: Aspnes
InP_inf = 3.626775501541622
InP_sigma = 0.
InP_pole0 = Drude_pole(1.058957872899501e17, 5.7398482416045037e17)
InP_pole1 = CP_pole(5.6825270150584e15, 7.209872451587759e14, 0.5027889733537447, 3.2744241060156174)
InP_pole2 = CP_pole(6.716429181412444e15, 7.766711026836548e14, 1.7560853571632558, 0.45309050353453645)
InP_210nm830nm_Aspnes = Drude_CP_material(52, InP_inf, InP_sigma, [InP_pole0, InP_pole1, InP_pole2])

# 54: Pb(Lead), 667nm~667000nm, ref: Aspnes
Pb_inf = 63.78475797585781
Pb_sigma = 0
Pb_pole0 = Drude_pole(5.961965202354131e15, 7.0333834394938664e13)
Pb_pole1 = CP_pole(2.911259237145358e16, 1.5191160932544688e12, -112.45264894690537, 4.005404297145654)
Pb_pole2 = CP_pole(1.1115119590603518e16, 5.798392865760157e15, 118.94546947844819, -0.21651681495803615)
Pb_667nm667um_Aspnes = Drude_CP_material(53, Pb_inf, Pb_sigma, [Pb_pole0, Pb_pole1, Pb_pole2])

# 55: Pb(Lead), 17.6nm~2480nm, ref: Werner
Pb_Werner_inf = 0.922013869186799
Pb_Werner_sigma = 0
Pb_Werner_pole0 = Drude_pole(1.9319813021951652e16, 5.595950189758007e14)
Pb_Werner_pole1 = CP_pole(7.068243560857568e15, 5.728311829540656e15, 1.3277109392312452, 4.608246371572353)
Pb_Werner_pole2 = CP_pole(2.885906496329074e15, 4.0634912549077506e14, 0.9494841129188148, 6.231514654899704)
Pb_17nm2480m_Werner = Drude_CP_material(54, Pb_Werner_inf, Pb_Werner_sigma, [Pb_Werner_pole0, Pb_Werner_pole1, Pb_Werner_pole2])

# 56: MoS2, 81nm ~ 1240nm, ref: Beal and Huges, 1979
MoS2_Beal_inf = 0.6666629636519996
MoS2_Beal_sigma = 0
MoS2_Beal_pole0 = Drude_pole(5.787933891316715e16, 6.022834035970166e17)
MoS2_Beal_pole1 = CP_pole(3.6800797789883615e15, 5.1516442503079006e14, -5.286687305304518, 1.3328867314338284)
MoS2_Beal_pole2 = CP_pole(3.089841259963743e15, 6.20391451645071e14, 6.328850020354818, -0.282986224460243)
MoS2_81nm1240nm_Beal = Drude_CP_material(55, MoS2_Beal_inf, MoS2_Beal_sigma, [MoS2_Beal_pole0, MoS2_Beal_pole1, MoS2_Beal_pole2])

# 57: Zn, 17nm ~ 2480 nm, ref: Werner
Zn_Werner_inf   = 0.9353326112666989
Zn_Werner_sigma = 0.
Zn_Werner_pole0 = Drude_pole(1.8650030896382956e16, 1.156763515474315e15)
Zn_Werner_pole1 = CP_pole(6.281154942432975e15, 3.3447763102316656e14, -0.07876104352186862, 2.642498975074368)
Zn_Werner_pole2 = CP_pole(2.928644004947927e15, 5.669334886633945e15, 3.2983875171188544, 4.114616544131449)
Zn_17nm2480nm_Werner = Drude_CP_material(56, Zn_Werner_inf, Zn_Werner_sigma, [Zn_Werner_pole0, Zn_Werner_pole1, Zn_Werner_pole2])

# 58: Zn, 360nm ~ 55600 nm, ref: Querry
Zn_Querry_inf   = -36.14809540520504
Zn_Querry_sigma = 0.
Zn_Querry_pole0 = Drude_pole(1.0858963094772044e16, 1.2458659337925967e13)
Zn_Querry_pole1 = CP_pole(5.903328254401487e13, 9.849856336616131e13, 33832.16944734923, -6.384095639922304)
Zn_360nm55um_Querry_ = Drude_CP_material(57, Zn_Querry_inf, Zn_Querry_sigma, [Zn_Querry_pole0, Zn_Querry_pole1])

# 59: Be(Beryllium), 248nm ~ 62000nm ref: Rakic, 1998
Be_rakic_inf   = 14.50503897777084
Be_rakic_sigma = 0.
Be_rakic_pole0 = Drude_pole(8.081047578394016e15, 5.437403816225399e13)
Be_rakic_pole1 = CP_pole(1.029662164136151e16, 5.638239444026941e15, -21.120946910365273, 3.731926618257504)
Be_rakic_pole2 = CP_pole(2.7903643299816916e16, 1.5191160368267437e12, 7.462381844297737, 7.008346324404507)
Be_248nm62um_rakic = Drude_CP_material(58, Be_rakic_inf, Be_rakic_sigma, [Be_rakic_pole0, Be_rakic_pole1, Be_rakic_pole2])

# 60: Bi(Bismuth), 1.55nm ~ 6199nm, ref: Hagemann
Bi_Hage_inf = 1.0391469824583945
Bi_Hage_sigma = 0.
Bi_Hage_pole0 = Drude_pole(1.5819154537970676e16, 1.3158328081662697e14)
Bi_Hage_pole1 = CP_pole(7.712148762374499e15, 3.011224336566192e15, -1.0172002138147653, 2.1238378054244977)
Bi_Hage_pole2 = CP_pole(5.935928372817811e15, 1.5206980572344568e15, 1.0677970096796718, 12.879245698725278)
Bi_2nm62um_Hage = Drude_CP_material(59, Bi_Hage_inf, Bi_Hage_sigma, [Bi_Hage_pole0, Bi_Hage_pole1, Bi_Hage_pole2])

# 61: Co(cobalt), 188nm ~ 1937nm, ref: Johnson
Co_Johnson_inf = -1.4184914770801542
Co_Johnson_sigma = 0.
Co_Johnson_pole0 = Drude_pole(1.092801370610973e16, 1.5191160368267437e12)
Co_Johnson_pole1 = CP_pole(4.3764872094965145e15, 4.632233701077526e15, 5.515772363317161, 3.1282129720827414)
Co_Johnson_pole2 = CP_pole(5.553858499232483e14, 1.39135863662823e14, 94.1448076673615, -2.4944365385232725)
Co_188nm2um_Johnson = Drude_CP_material(60, Co_Johnson_inf, Co_Johnson_sigma, [Co_Johnson_pole0, Co_Johnson_pole1, Co_Johnson_pole2])
