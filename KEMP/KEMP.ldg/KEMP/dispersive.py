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
        self.materials = []

class Dispersive_3d:
    def __init__(self, fdtd):
        self.fdtd = fdtd
        self.fdtd.cores['dispersive'] = self
        self.fdtd.cores['pole'] = self
        self.fdtd.engine.updates['pole_e'] = self.update_pole
        self.fdtd.engine.updates['disp_e'] = self.update_e
        self.materials = []

# CLASSES for CP-model materials
# Including Drude, Debye, Lorentz materials

class CP_pole:
    def __init__(self, wfreq, gamma, amp, phase):
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

class Drude_CP_material:
    def __init__(self, mark_number, eps_r, sigma, drude_poles, cp_poles):
        self.classification  = 'electric dispersive'
        self.eps_r = eps_r
        self.sigma = sigma
        self.mark_number = mark_number
        # cp_poles is list of elements(class CP_pole)
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

    def epr(self, wavelength):
        c0 = 299792458.
        wfreq = 2.*np.pi*c0/wavelength
        epr_temp = 0.
        epr_temp += self.eps_r
        for pole in self.dpoles:
            epr_temp += -(pole.wfreq**2)/((wfreq**2)+1.j*pole.gamma*wfreq)
        for pole in self.cpoles:
            alpha = pole.gamma+1.j*pole.wfreq
            beta  = 2.j*pole.amp*pole.wfreq*np.exp(1.j*pole.phase)
            epr_temp += .5*(beta/(alpha-1.j*wfreq) + beta.conjugate()/(alpha.conjugate()-1.j*wfreq))
        return epr_temp

    def mur(self, wavelength):
        return 1.

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
# input parameters of CP_pole   : wfreq_p, gamma, amplitude, phase

# For Ph.D Jong-Ho Choe's FORMAT:
# Case of Drude pole  -> Drude_pole(D1CP2.wp[1]   , D1CP2.gamma[1])
# Case of 1st CP_pole -> CP_pole(D1CP2.OMEGA[1], D1CP2.GAMMA[1], D1CP2.A[1], D1CP2.PHI[1])
# Case of 2nd CP_pole -> CP_pole(D1CP2.OMEGA[2], D1CP2.GAMMA[2], D1CP2.A[2], D1CP2.PHI[2])
# Material(D1CP2)     -> Drude_CP_material(D1CP2.EPS, 0., [Drude_pole], [1st_CP_pole, 2nd_CP_Pole])

# 0: None (No material)

# 1: Gold (Au, Johnson & Christy)
Au_eps_r = 1.
Au_sigma = 0.
Au_pole0 = Drude_pole(1.309985248506338e16, 1.11042047627014e14)
Au_pole1 =    CP_pole(3.9824789679063745e15, 6.429695525898232e14, 0.5585964221628438, -7.132211522998809)
Au_pole2 =    CP_pole(5.095186997089663e15, 2.42740347167135e15, 2.230551423881718, -7.1271367386950395)
Au       = Drude_CP_material(1, Au_eps_r, Au_sigma, [Au_pole0], [Au_pole1, Au_pole2])
gold     = Au
Gold     = Au

# 2: Silver (Ag)
Ag_eps_r = 2.326297450762337
Ag_sigma = 0.
Ag_pole0 = Drude_pole(1.391109719295765e16, 1.895188885873623e13)
Ag_pole1 =    CP_pole(6.307186498028408e15, 1.218011670965938e15, -1.774907705226043, 1.321095730578253)
Ag_pole2 =    CP_pole(6.450702973027045e15, 1.807921894755012e15, 2.098652912575634, 0.930399714614142)
Ag       = Drude_CP_material(2, Ag_eps_r, Ag_sigma, [Ag_pole0], [Ag_pole1, Ag_pole2])
Silver = Ag
silver = Ag

# 3: Aluminum (Al)
Al_eps_r = 1.
Al_sigma = 0.
Al_pole0 = Drude_pole(2.0356022851971e16, 2.155889915410091e14)
Al_pole1 =    CP_pole(2.259064497940003e15, 2.573579064902523e14, 5.658877755515978, -0.4850996787410118)
Al_pole2 =    CP_pole(2.9387088835860865e15, 1.4710570697224062e15, 3.1116433236877286, 0.5083054567607643)
Al       = Drude_CP_material(3, Al_eps_r, Al_sigma, [Al_pole0], [Al_pole1, Al_pole2])
Aluminum = Al
aluminum = Al

# 4: Silicon (Si)
Si_eps_r = 2.
Si_sigma = 0.
Si_pole0 = Drude_pole(0., 0.)
Si_pole1 =    CP_pole(5.115832370098545e15, 2.418956338146272e14, -1.3281240719318108, 2.3078590980047147)
Si_pole2 =    CP_pole(6.492623507279991e15, 7.795088221370624e14, -4.410075824564194 , 3.290102642882403)
Si       = Drude_CP_material(4, Si_eps_r, Si_sigma, [Si_pole0], [Si_pole1, Si_pole2])
Silicon  = Si
silicon  = Si

# 5: Nickel (Ni)

# 6: Platinum (Pt)

# 7: Chromium (Cr)

# 8: Alummina (Al2O3)

# 9: PbSe (by triumph)
PbSe_eps_r = -5.
PbSe_sigma =  0.
PbSe_pole0 = Drude_pole(1.7414385557381866e14, 3.052157269596333e16)
PbSe_pole1 =    CP_pole( 8.807042221568294e14, 8.807042221568294e14, 5.333166601595723, -0.5521547355558631)
#PbSe_pole2 =    CP_pole( , , , )
PbSe       = Drude_CP_material(9, PbSe_eps_r, PbSe_sigma, [PbSe_pole0], [PbSe_pole1])

# 10: Perfect Absorber (by triumph)
PA_eps_r = 1.0
PA_sigma = 0.0
#PA_pole0 = Lorentz_pole(4520469903415956.0, 343873311189568.5, 0.88297503638502084)
#PA_pole1 = Lorentz_pole(4826330019965662.0, 202736289826248.88, 0.40806050170920971)
#PA_pole2 = Lorentz_pole(4982534240577065.0, 64945076416321.305, 0.12162449163805744)
#PA_pole3 = Lorentz_pole(3611713050637068.0, 452597112806037.94, 2.2962686900550193)
#PA_pole4 = Lorentz_pole(4096009523077053.0, 451026879333732.75, 1.5979295608383155)
#PA_pole5 = Lorentz_pole(3218269729949211.0, 283317005775503.75, 1.6634307663587071)
#PA_pole6 = Lorentz_pole(3063891792899861.5, 64034556040084.125, 0.44100321223323763)
#PA = Drude_CP_material(10, PA_eps_r, PA_sigma, [], PA_pole0 + PA_pole1 + PA_pole2 + PA_pole3 + PA_pole4 + PA_pole5 + PA_pole6)

# 11: GaP (Gallium Phosphide)
#GaP_eps_r = 1.0
#GaP_sigma = 0.0
#GaP_pole0 = Drude_pole(1.5191160368267435e15, 3.81086172452832e15)
#GaP_pole1 =    CP_pole(4.225854687726881e15, 5.605378709626744e15, 0.12181890479555313, 0.3109878551374564)
#GaP_pole2 =    CP_pole(1.044836939130798e15, 6.298249883131429e15, 1.0931626166214075, 0.5688245821144617)
#GaP = Drude_CP_material(11, GaP_eps_r, GaP_sigma, [GaP_pole0], [GaP_pole1, GaP_pole2])

# 11: GaP (Gallium Phosphide: eps_inf = 3)
#GaP_eps_r = 3.0
#GaP_sigma = 0.0
#GaP_pole0 = Drude_pole(1.382483812352820e15, 4.183900511226060e14)
#GaP_pole1 =    CP_pole(5.622297114927660e15, 1.984055442582460e14, -0.640882178698002, 2.92485614955248)
#GaP_pole2 =    CP_pole(6.469716689353180e15, 1.072388189856030e15, -2.829843126250670, 2.73993832151344)
#GaP = Drude_CP_material(11, GaP_eps_r, GaP_sigma, [GaP_pole0], [GaP_pole1, GaP_pole2])

# 11: GaP (Gallium Phosphide: eps_inf = 5.5)
#GaP_eps_r = 5.5
#GaP_sigma = 0.0
#GaP_pole0 = Drude_pole(1.950357601055320e15, 4.26763099248827e6)
#GaP_pole1 =    CP_pole(5.622164645847630e15, 2.30947383768417e14, -0.814390693918477, 2.89202859162397)
#GaP_pole2 =    CP_pole(6.615211792110890e15, 6.33741552292504e14, -1.763482360547650, 3.08826904425043)
#GaP = Drude_CP_material(11, GaP_eps_r, GaP_sigma, [GaP_pole0], [GaP_pole1, GaP_pole2])

# 11: GaP (Gallium Phosphide: eps_inf = 6.5)
GaP_eps_r = 6.5
GaP_sigma = 0.0
GaP_pole0 = Drude_pole(    2327669176.17035, 7.992844208248990e15)
GaP_pole1 =    CP_pole(5.594796671965160e15, 2.905883763421080e14,-1.08334463810630, 2.663199256740390)
GaP_pole2 =    CP_pole(6.884406554806240e15, 1.267549774767240e13, 1.20593067044096, 0.765913138789777)
GaP = Drude_CP_material(11, GaP_eps_r, GaP_sigma, [GaP_pole0], [GaP_pole1, GaP_pole2])

bl = h5.File(os.path.dirname(os.path.abspath(__file__)) + '/material_data/black_metal.h5')
bl_params = np.array(bl['lorentz'])
bl_eps_r = float(np.array(bl['epr_inf']))
bl_sigma = 0.
bl_poles = []

# 12: Silver (Ag)
Agp_eps_r = 4.125
Agp_sigma = 0.
Agp_pole0 = Drude_pole(1.3462259458308412e16, 5.1911233092276440e13)
Agp_pole1 =    CP_pole(3.3973509902768590e14, 1.4256692514480055e15, 3.9885432610910870, 4.693855988590337)
Agpalik   = Drude_CP_material(12, Agp_eps_r, Agp_sigma, [Agp_pole0], [Agp_pole1])
Silver = Ag
silver = Ag

for p in xrange(np.shape(bl_params)[0]):
    bl_poles += Lorentz_pole(*(bl_params[p]))

PA = Drude_CP_material(10, bl_eps_r, bl_sigma, [], bl_poles)
