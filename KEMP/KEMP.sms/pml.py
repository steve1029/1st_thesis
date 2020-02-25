from __future__ import division
import scipy as sc
import numpy as np
import sys

from util   import *
from units  import to_SI, to_NU
from ndarray import Fields

class CPML_1d:
    def __init__(self, fdtd, pml_apply, pml_thick={'z':(10,10)}, alpha0=0., kappa0=1., malpha=4, msigma=4, lnR0=-16.):
        self.fdtd   = fdtd
        self.fdtd.cores['pml'] = self
        self.fdtd.engine.updates['pml_e'] = self.update_e
        self.fdtd.engine.updates['pml_h'] = self.update_h
        self.nz = self.fdtd.nz
        self.pml_apply = pml_apply
        self.pml_thick = pml_thick
        if '-' in self.pml_apply['z']:
            self.pcb_ezm, self.pcb_hzm, self.pca_ezm, self.pca_hzm, self.psi_exz_m, self.psi_hyz_m = \
                self.set_pml_params('z', '-', alpha0, kappa0, malpha, msigma, lnR0)
        if '+' in self.pml_apply['z']:
            self.pcb_ezp, self.pcb_hzp, self.pca_ezp, self.pca_hzp, self.psi_exz_p, self.psi_hyz_p = \
                self.set_pml_params('z', '+', alpha0, kappa0, malpha, msigma, lnR0)

    def set_pml_params(self, pml_axis, pml_direction, alpha_max, kappa_max, malpha, msigma, lnR0):
        fdtd = self.fdtd
        real_dtype = comp_to_real(fdtd.dtype)

        if pml_direction == '-': direction = 0
        if pml_direction == '+': direction = 1

        npml = self.pml_thick[pml_axis][direction]

        sigma_max = (msigma+1)*(-lnR0)/(2.*npml)

        if pml_direction == '-':
            plnml_i = np.arange(.5, npml  , dtype=real_dtype)[::-1]/npml
            plnml_o = np.arange(1., npml+1, dtype=real_dtype)[::-1]/npml

            plnla_i = np.arange(1., npml+1, dtype=real_dtype)      /npml
            plnla_o = np.arange(.5, npml  , dtype=real_dtype)      /npml

            alpha_e = alpha_max*(plnla_i)**malpha
            sigma_e = sigma_max*(plnml_i)**msigma
            kappa_e = 1. + (kappa_max - 1)*(plnml_i)**msigma

            alpha_h = alpha_max*(plnla_o)**malpha
            sigma_h = sigma_max*(plnml_o)**msigma
            kappa_h = 1. + (kappa_max - 1)*(plnml_o)**msigma

            eps_r_e  = fdtd.eps_r[:npml]
            eps_r_h  = fdtd.eps_r[1:npml+1]

        if pml_direction == '+':
            plnml_i = np.arange(.5, npml  , dtype=real_dtype)      /npml
            plnml_o = np.arange(1., npml+1, dtype=real_dtype)      /npml

            plnla_i = np.arange(1., npml+1, dtype=real_dtype)[::-1]/npml
            plnla_o = np.arange(.5, npml  , dtype=real_dtype)[::-1]/npml

            alpha_e = alpha_max*(plnla_o)**malpha
            sigma_e = sigma_max*(plnml_o)**msigma
            kappa_e = 1. + (kappa_max - 1)*(plnml_o)**msigma

            alpha_h = alpha_max*(plnla_i)**malpha
            sigma_h = sigma_max*(plnml_i)**msigma
            kappa_h = 1. + (kappa_max - 1)*(plnml_i)**msigma

            eps_r_e  = fdtd.eps_r[-npml-1:-1]
            eps_r_h  = fdtd.eps_r[-npml:]

        pcb_e = np.exp(-(sigma_e/kappa_e + alpha_e)*fdtd.dt)
        pcb_h = np.exp(-(sigma_h/kappa_h + alpha_h)*fdtd.dt)

        pca_e = sigma_e*(pcb_e - 1.)/(sigma_e + kappa_e*alpha_e)
        pca_h = sigma_h*(pcb_h - 1.)/(sigma_h + kappa_h*alpha_h)

        if pml_direction == '-':
            fdtd.rdz_e[:npml]    /= kappa_e[:]
            fdtd.rdz_h[1:npml+1] /= kappa_h[:]

        if pml_direction == '+':
            fdtd.rdz_e[-npml-1:-1] /= kappa_e[:]
            fdtd.rdz_h[-npml:]     /= kappa_h[:]

        psi_e = np.zeros(npml, dtype=real_dtype)
        psi_h = np.zeros(npml, dtype=real_dtype)

        return pcb_e, pcb_h, pca_e, pca_h, psi_e, psi_h

    def sub_update_e(self, efield, hfield, cfield, ds_e, pcb, pca, psi, sle, sl0, sl1, sld, sln):
        fdtd = self.fdtd
        psi[sln] = pcb[sln]*psi[sln] + pca[sln]*(hfield[sl1]-hfield[sl0])
        efield[sle] -= fdtd.ce2x[sle]*psi[sln]*ds_e[sld]

    def sub_update_h(self, efield, hfield,         ds_h, pcb, pca, psi, slh, sl0, sl1, sld, sln):
        psi[sln] = pcb[sln]*psi[sln] + pca[sln]*(efield[sl1]-efield[sl0])
        hfield[slh] -= .5*psi[sln]*ds_h[sld]

    def update_e(self):
        fdtd = self.fdtd
        if '+' in self.pml_apply['z']:
            npml = self.pml_thick['z'][1]
            sl0 = slice(-npml-1, -1)
            sl1 = slice(-npml, None)
            sln = slice(None,  None)
            sld = sl1
            sle = sl1
            self.sub_update_e(fdtd.ex, fdtd.hy, fdtd.ce2x, fdtd.rdz_e, self.pcb_ezp, self.pca_ezp, self.psi_exz_p, sle, sl0, sl1, sld, sln)

        if '-' in self.pml_apply['z']:
            npml = self.pml_thick['z'][0]
            sl0 = slice(None, npml)
            sl1 = slice(1,  npml+1)
            sln = slice(None, None)
            sld = sl1
            sle = sl1
            self.sub_update_e(fdtd.ex, fdtd.hy, fdtd.ce2x, fdtd.rdz_e, self.pcb_ezm, self.pca_ezm, self.psi_exz_m, sle, sl0, sl1, sld, sln)

    def update_h(self):
        fdtd = self.fdtd
        if '+' in self.pml_apply['z']:
            npml = self.pml_thick['z'][1]
            sl0 = slice(-npml-1, -1)
            sl1 = slice(-npml, None)
            sln = slice(None,  None)
            sld = sl0
            slh = sl0
            self.sub_update_h(fdtd.ex, fdtd.hy, fdtd.rdz_h, self.pcb_hzp, self.pca_hzp, self.psi_hyz_p, slh, sl0, sl1, sld, sln)

        if '-' in self.pml_apply['z']:
            npml = self.pml_thick['z'][0]
            sl0 = slice(None, npml)
            sl1 = slice(1,  npml+1)
            sln = slice(None, None)
            sld = sl0
            slh = sl0
            self.sub_update_h(fdtd.ex, fdtd.hy, fdtd.rdz_h, self.pcb_hzm, self.pca_hzm, self.psi_hyz_m, slh, sl0, sl1, sld, sln)

class CPML_2d:
    def __init__(self, fdtd, pml_apply, pml_thick={'x':(10,10),'y':(10,10)}, alpha0=0., kappa0=1., malpha=4, msigma=4, lnR0=-16.):
        self.fdtd = fdtd
        self.fdtd.cores['pml'] = self
        self.fdtd.engine.updates['pml_e'] = self.update_e
        self.fdtd.engine.updates['pml_h'] = self.update_h
        self.pml_apply = pml_apply
        self.pml_thick = pml_thick
        self.alpha = alpha0
        self.kappa = kappa0
        self.alpha_exponent = malpha
        self.sigma_exponent = msigma

        self.npxm = self.pml_thick['x'][0]
        self.npxp = self.pml_thick['x'][1]
        self.npym = self.pml_thick['y'][0]
        self.npyp = self.pml_thick['y'][1]

    def setup(self):
        fdtd = self.fdtd
        code_ces_prev = [', __GLOBAL__ __FLOAT__* ce1', ', __GLOBAL__ __FLOAT__* ce2', ', __GLOBAL__ __FLOAT__* ce', \
                         'ce1[idx0]', 'ce2[idx0]', 'ce[idx0]']
        code_chs_prev = [', __GLOBAL__ __FLOAT__* ch1', ', __GLOBAL__ __FLOAT__* ch2', ', __GLOBAL__ __FLOAT__* ch', \
                         'ch1[idx0]', 'ch2[idx0]', 'ch[idx0]']
        code_rds_prev = [', __GLOBAL__ __FLOAT__* ds', 'ds[ids]*']
        ces = '%s' % self.fdtd.dt
        chs = '%s' % self.fdtd.dt
        if    self.fdtd.is_electric    : code_ces_post = code_ces_prev
        else                           : code_ces_post = ['', '', '', ces, ces, ces]
        if    self.fdtd.is_magnetic    : code_chs_post = code_chs_prev
        else                           : code_chs_post = ['', '', '', chs, chs, chs]
        if    self.fdtd.is_uniform_grid: code_rds_post = ['', '']
        else                           : code_rds_post = code_rds_prev

        if 'cpu' in fdtd.engine.name:
            omp_ces_prev = [', ce1', ', ce2', ', ce']
            code_ces_prev += omp_ces_prev
            if self.fdtd.is_electric    : code_ces_post += omp_ces_prev
            else                        : code_ces_post += ['', '', '']
            omp_chs_prev = [', ch1', ', ch2', ', ch']
            code_chs_prev += omp_chs_prev
            if self.fdtd.is_magnetic    : code_chs_post += omp_chs_prev
            else                        : code_chs_post += ['', '', '']
            omp_rds_prev = [', ds']
            code_rds_prev += omp_rds_prev
            if self.fdtd.is_uniform_grid: code_rds_post += ['']
            else                        : code_rds_post += omp_rds_prev

        code = template_to_code(fdtd.engine.templates['cpml'], \
                                code_ces_prev + code_chs_prev + code_rds_prev + fdtd.engine.code_prev, \
                                code_ces_post + code_chs_post + code_rds_post + fdtd.engine.code_post)

        fdtd.engine.programs['cpml'] = fdtd.engine.build(code)
        fdtd.engine.kernel_args['pml_e'] = {'x':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}, \
                                            'y':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}}
        fdtd.engine.kernel_args['pml_h'] = {'x':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}, \
                                            'y':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}}

        for part in fdtd.complex_parts:
            if   fdtd.mode == '2DTE':
                update_e_args_xp = [np.int32(0), np.int32(1), np.int32(-1), \
                                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                    np.int32(self.pml_thick['x'][1]), \
                                    fdtd.ey.__dict__[part].data, fdtd.hz.__dict__[part].data]
                update_h_args_xp = [np.int32(0), np.int32(1), np.int32(-1), \
                                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                    np.int32(self.pml_thick['x'][1]), \
                                    fdtd.ey.__dict__[part].data, fdtd.hz.__dict__[part].data]
                update_e_args_xm = [np.int32(0), np.int32(0), np.int32(-1), \
                                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                    np.int32(self.pml_thick['x'][0]), \
                                    fdtd.ey.__dict__[part].data, fdtd.hz.__dict__[part].data]
                update_h_args_xm = [np.int32(0), np.int32(0), np.int32(-1), \
                                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                    np.int32(self.pml_thick['x'][0]), \
                                    fdtd.ey.__dict__[part].data, fdtd.hz.__dict__[part].data]
                update_e_args_yp = [np.int32(1), np.int32(1), np.int32( 1), \
                                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                    np.int32(self.pml_thick['y'][1]), \
                                    fdtd.ex.__dict__[part].data, fdtd.hz.__dict__[part].data]
                update_h_args_yp = [np.int32(1), np.int32(1), np.int32( 1), \
                                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                    np.int32(self.pml_thick['y'][1]), \
                                    fdtd.ex.__dict__[part].data, fdtd.hz.__dict__[part].data]
                update_e_args_ym = [np.int32(1), np.int32(0), np.int32( 1), \
                                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                    np.int32(self.pml_thick['y'][0]), \
                                    fdtd.ex.__dict__[part].data, fdtd.hz.__dict__[part].data]
                update_h_args_ym = [np.int32(1), np.int32(0), np.int32( 1), \
                                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                    np.int32(self.pml_thick['y'][0]), \
                                    fdtd.ex.__dict__[part].data, fdtd.hz.__dict__[part].data]
                if fdtd.is_electric:
                    update_e_args_xp += [fdtd.ce2y.data]
                    update_e_args_xm += [fdtd.ce2y.data]
                    update_e_args_yp += [fdtd.ce2x.data]
                    update_e_args_ym += [fdtd.ce2x.data]
                if fdtd.is_magnetic:
                    update_h_args_xp += [fdtd.ch2z.data]
                    update_h_args_xm += [fdtd.ch2z.data]
                    update_h_args_yp += [fdtd.ch2z.data]
                    update_h_args_yp += [fdtd.ch2z.data]
                if not fdtd.is_uniform_grid:
                    update_e_args_xp += [fdtd.rdx_e.data]
                    update_h_args_xp += [fdtd.rdx_h.data]
                    update_e_args_xm += [fdtd.rdx_e.data]
                    update_h_args_xm += [fdtd.rdx_h.data]
                    update_e_args_yp += [fdtd.rdy_e.data]
                    update_h_args_yp += [fdtd.rdy_h.data]
                    update_e_args_ym += [fdtd.rdy_e.data]
                    update_h_args_ym += [fdtd.rdy_h.data]

                if '+' in self.pml_apply['x']:
                    update_e_args_xp += [self.pcb_exp.data, self.pca_exp.data, self.psi_eyx_p.__dict__[part].data]
                    update_h_args_xp += [self.pcb_hxp.data, self.pca_hxp.data, self.psi_hzx_p.__dict__[part].data]
                if '-' in self.pml_apply['x']:
                    update_e_args_xm += [self.pcb_exm.data, self.pca_exm.data, self.psi_eyx_m.__dict__[part].data]
                    update_h_args_xm += [self.pcb_hxm.data, self.pca_hxm.data, self.psi_hzx_m.__dict__[part].data]
                if '+' in self.pml_apply['y']:
                    update_e_args_yp += [self.pcb_eyp.data, self.pca_eyp.data, self.psi_exy_p.__dict__[part].data]
                    update_h_args_yp += [self.pcb_hyp.data, self.pca_hyp.data, self.psi_hzy_p.__dict__[part].data]
                if '-' in self.pml_apply['y']:
                    update_e_args_ym += [self.pcb_eym.data, self.pca_eym.data, self.psi_exy_m.__dict__[part].data]
                    update_h_args_ym += [self.pcb_hym.data, self.pca_hym.data, self.psi_hzy_m.__dict__[part].data]

            elif fdtd.mode == '2DTM':
                update_e_args_xp  = [np.int32(0), np.int32(1), np.int32( 1), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                     np.int32(self.pml_thick['x'][1]), \
                                     fdtd.ez.__dict__[part].data, fdtd.hy.__dict__[part].data]
                update_h_args_xp  = [np.int32(0), np.int32(1), np.int32( 1), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                     np.int32(self.pml_thick['x'][1]), \
                                     fdtd.ez.__dict__[part].data, fdtd.hy.__dict__[part].data]
                update_e_args_xm  = [np.int32(0), np.int32(0), np.int32( 1), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                     np.int32(self.pml_thick['x'][0]), \
                                     fdtd.ez.__dict__[part].data, fdtd.hy.__dict__[part].data]
                update_h_args_xm  = [np.int32(0), np.int32(0), np.int32( 1), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                     np.int32(self.pml_thick['x'][0]), \
                                     fdtd.ez.__dict__[part].data, fdtd.hy.__dict__[part].data]
                update_e_args_yp  = [np.int32(1), np.int32(1), np.int32(-1), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                     np.int32(self.pml_thick['y'][1]), \
                                     fdtd.ez.__dict__[part].data, fdtd.hx.__dict__[part].data]
                update_h_args_yp  = [np.int32(1), np.int32(1), np.int32(-1), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                     np.int32(self.pml_thick['y'][1]), \
                                     fdtd.ez.__dict__[part].data, fdtd.hx.__dict__[part].data]
                update_e_args_ym  = [np.int32(1), np.int32(0), np.int32(-1), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                     np.int32(self.pml_thick['y'][0]), \
                                     fdtd.ez.__dict__[part].data, fdtd.hx.__dict__[part].data]
                update_h_args_ym  = [np.int32(1), np.int32(0), np.int32(-1), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                     np.int32(self.pml_thick['y'][0]), \
                                     fdtd.ez.__dict__[part].data, fdtd.hx.__dict__[part].data]
                if fdtd.is_electric:
                    update_e_args_xp += [fdtd.ce2z.data]
                    update_e_args_xm += [fdtd.ce2z.data]
                    update_e_args_yp += [fdtd.ce2z.data]
                    update_e_args_ym += [fdtd.ce2z.data]
                if fdtd.is_magnetic:
                    update_h_args_xp += [fdtd.ch2y.data]
                    update_h_args_xm += [fdtd.ch2y.data]
                    update_h_args_yp += [fdtd.ch2x.data]
                if not fdtd.is_uniform_grid:
                    update_e_args_xp += [fdtd.rdx_e.data]
                    update_h_args_xp += [fdtd.rdx_h.data]
                    update_e_args_xm += [fdtd.rdx_e.data]
                    update_h_args_xm += [fdtd.rdx_h.data]
                    update_e_args_yp += [fdtd.rdy_e.data]
                    update_h_args_yp += [fdtd.rdy_h.data]
                    update_e_args_ym += [fdtd.rdy_e.data]
                    update_h_args_ym += [fdtd.rdy_h.data]

                if '+' in self.pml_apply['x']:
                    update_e_args_xp += [self.pcb_exp.data, self.pca_exp.data, self.psi_ezx_p.__dict__[part].data]
                    update_h_args_xp += [self.pcb_hxp.data, self.pca_hxp.data, self.psi_hyx_p.__dict__[part].data]
                if '-' in self.pml_apply['x']:
                    update_e_args_xm += [self.pcb_exm.data, self.pca_exm.data, self.psi_ezx_m.__dict__[part].data]
                    update_h_args_xm += [self.pcb_hxm.data, self.pca_hxm.data, self.psi_hyx_m.__dict__[part].data]
                if '+' in self.pml_apply['y']:
                    update_e_args_yp += [self.pcb_eyp.data, self.pca_eyp.data, self.psi_ezy_p.__dict__[part].data]
                    update_h_args_yp += [self.pcb_hyp.data, self.pca_hyp.data, self.psi_hxy_p.__dict__[part].data]
                if '-' in self.pml_apply['y']:
                    update_e_args_ym += [self.pcb_eym.data, self.pca_eym.data, self.psi_ezy_m.__dict__[part].data]
                    update_h_args_ym += [self.pcb_hym.data, self.pca_hym.data, self.psi_hxy_m.__dict__[part].data]

            if 'opencl' in fdtd.engine.name:
                gs_xp = cl_global_size(self.pml_thick['x'][1]*fdtd.ny, fdtd.engine.ls)
                gs_xm = cl_global_size(self.pml_thick['x'][0]*fdtd.ny, fdtd.engine.ls)
                gs_yp = cl_global_size(self.pml_thick['y'][1]*fdtd.nx, fdtd.engine.ls)
                gs_ym = cl_global_size(self.pml_thick['y'][0]*fdtd.nx, fdtd.engine.ls)
            else:
                gs_xp = self.pml_thick['x'][1]*fdtd.ny
                gs_xm = self.pml_thick['x'][0]*fdtd.ny
                gs_yp = self.pml_thick['y'][1]*fdtd.nx
                gs_ym = self.pml_thick['y'][0]*fdtd.nx

            if '+' in self.pml_apply['x']:
                fdtd.engine.kernel_args['pml_e']['x']['+'][part] = [fdtd.engine.queue, (gs_xp,), (fdtd.engine.ls,)] + update_e_args_xp
                fdtd.engine.kernel_args['pml_h']['x']['+'][part] = [fdtd.engine.queue, (gs_xp,), (fdtd.engine.ls,)] + update_h_args_xp
            if '-' in self.pml_apply['x']:
                fdtd.engine.kernel_args['pml_e']['x']['-'][part] = [fdtd.engine.queue, (gs_xm,), (fdtd.engine.ls,)] + update_e_args_xm
                fdtd.engine.kernel_args['pml_h']['x']['-'][part] = [fdtd.engine.queue, (gs_xm,), (fdtd.engine.ls,)] + update_h_args_xm
            if '+' in self.pml_apply['y']:
                fdtd.engine.kernel_args['pml_e']['y']['+'][part] = [fdtd.engine.queue, (gs_yp,), (fdtd.engine.ls,)] + update_e_args_yp
                fdtd.engine.kernel_args['pml_h']['y']['+'][part] = [fdtd.engine.queue, (gs_yp,), (fdtd.engine.ls,)] + update_h_args_yp
            if '-' in self.pml_apply['y']:
                fdtd.engine.kernel_args['pml_e']['y']['-'][part] = [fdtd.engine.queue, (gs_ym,), (fdtd.engine.ls,)] + update_e_args_ym
                fdtd.engine.kernel_args['pml_h']['y']['-'][part] = [fdtd.engine.queue, (gs_ym,), (fdtd.engine.ls,)] + update_h_args_ym

        if 'opencl' in fdtd.engine.name:
            fdtd.engine.kernels['pml_e'] = fdtd.engine.programs['cpml'].update_e_2d
            fdtd.engine.kernels['pml_h'] = fdtd.engine.programs['cpml'].update_h_2d
        elif 'cuda' in fdtd.engine.name:
            fdtd.engine.kernels['pml_e'] = fdtd.engine.get_function(fdtd.engine.programs['cpml'], 'update_e_2d')
            fdtd.engine.kernels['pml_h'] = fdtd.engine.get_function(fdtd.engine.programs['cpml'], 'update_h_2d')
            for ax in ['x', 'y']:
                if   '+' in self.pml_apply[ax]:
                    axis = ax
                    dirc = '+'
                elif '-' in self.pml_apply[ax]:
                    axis = ax
                    dirc = '-'
            fdtd.engine.prepare(fdtd.engine.kernels['pml_e'], fdtd.engine.kernel_args['pml_e'][axis][dirc]['real'])
            fdtd.engine.prepare(fdtd.engine.kernels['pml_h'], fdtd.engine.kernel_args['pml_h'][axis][dirc]['real'])
        elif 'cpu' in fdtd.engine.name:
            for ax in ['x', 'y']:
                if   '+' in self.pml_apply[ax]:
                    axis = ax
                    dirc = '+'
                elif '-' in self.pml_apply[ax]:
                    axis = ax
                    dirc = '-'
            fdtd.engine.kernels['pml_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['cpml'].update_e_2d, fdtd.engine.kernel_args['pml_e'][axis][dirc]['real'])
            fdtd.engine.kernels['pml_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['cpml'].update_h_2d, fdtd.engine.kernel_args['pml_h'][axis][dirc]['real'])

    def set_pml_params(self, pml_axis, pml_direction, alpha_max, kappa_max, malpha, msigma, lnR0):
        fdtd = self.fdtd
        real_dtype = comp_to_real(fdtd.dtype)

        if pml_direction == '-': direction = 0
        if pml_direction == '+': direction = 1

        npml = self.pml_thick[pml_axis][direction]

        if pml_direction == '-':
            if pml_axis == 'x':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdx_e[1:npml+1]).sum())
            if pml_axis == 'y':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdy_e[1:npml+1]).sum())
        if pml_direction == '+':
            if pml_axis == 'x':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdx_h[-npml:-1]).sum())
            if pml_axis == 'y':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdy_h[-npml:-1]).sum())

        if pml_direction == '-':
            plnml_i = np.arange(.5, npml  , dtype=real_dtype)[::-1]/npml
            plnml_o = np.arange(1., npml+1, dtype=real_dtype)[::-1]/npml

            plnla_i = np.arange(1., npml+1, dtype=real_dtype)      /npml
            plnla_o = np.arange(.5, npml  , dtype=real_dtype)      /npml

            alpha_e = alpha_max*(plnla_i)**malpha
            sigma_e = sigma_max*(plnml_i)**msigma
            kappa_e = 1. + (kappa_max - 1)*(plnml_i)**msigma

            alpha_h = alpha_max*(plnla_o)**malpha
            sigma_h = sigma_max*(plnml_o)**msigma
            kappa_h = 1. + (kappa_max - 1)*(plnml_o)**msigma

        if pml_direction == '+':
            plnml_i = np.arange(.5, npml  , dtype=real_dtype)      /npml
            plnml_o = np.arange(1., npml+1, dtype=real_dtype)      /npml

            plnla_i = np.arange(1., npml+1, dtype=real_dtype)[::-1]/npml
            plnla_o = np.arange(.5, npml  , dtype=real_dtype)[::-1]/npml

            alpha_e = alpha_max*(plnla_o)**malpha
            sigma_e = sigma_max*(plnml_o)**msigma
            kappa_e = 1. + (kappa_max - 1)*(plnml_o)**msigma

            alpha_h = alpha_max*(plnla_i)**malpha
            sigma_h = sigma_max*(plnml_i)**msigma
            kappa_h = 1. + (kappa_max - 1)*(plnml_i)**msigma

        if pml_axis == 'x':
            pcb_e = np.exp(-(sigma_e/kappa_e + alpha_e)*fdtd.dt)
            pcb_h = np.exp(-(sigma_h/kappa_h + alpha_h)*fdtd.dt)
            pca_e = (sigma_e/(sigma_e + kappa_e*alpha_e))*(pcb_e - 1.)
            pca_h = (sigma_h/(sigma_h + kappa_h*alpha_h))*(pcb_h - 1.)

        if pml_axis == 'y':
            pcb_e = np.exp(-(sigma_e/kappa_e + alpha_e)*fdtd.dt)
            pcb_h = np.exp(-(sigma_h/kappa_h + alpha_h)*fdtd.dt)
            pca_e = (sigma_e/(sigma_e + kappa_e*alpha_e))*(pcb_e - 1.)
            pca_h = (sigma_h/(sigma_h + kappa_h*alpha_h))*(pcb_h - 1.)

        if pml_direction == '-':
            sle = slice(1,  npml+1)
            slh = slice(None, npml)
            if pml_axis == 'x':
                fdtd.rdx_e[sle] /= kappa_e
                fdtd.rdx_h[slh] /= kappa_h
            if pml_axis == 'y':
                fdtd.rdy_e[sle] /= kappa_e
                fdtd.rdy_h[slh] /= kappa_h

        if pml_direction == '+':
            sle = slice(-npml, None)
            slh = slice(-npml-1, -1)
            if pml_axis == 'x':
                fdtd.rdx_e[sle] /= kappa_e
                fdtd.rdx_h[slh] /= kappa_h
            if pml_axis == 'y':
                fdtd.rdy_e[sle] /= kappa_e
                fdtd.rdy_h[slh] /= kappa_h

        if pml_axis == 'x':	psi_e, psi_h = [np.zeros((npml, fdtd.ny), dtype=real_dtype) for i in xrange(2)]
        if pml_axis == 'y':	psi_e, psi_h = [np.zeros((fdtd.nx, npml), dtype=real_dtype) for i in xrange(2)]

        pcb_e_dev = Fields(fdtd, np.shape(pcb_e), real_dtype)
        pcb_h_dev = Fields(fdtd, np.shape(pcb_h), real_dtype)
        pca_e_dev = Fields(fdtd, np.shape(pca_e), real_dtype)
        pca_h_dev = Fields(fdtd, np.shape(pca_h), real_dtype)
        psi_e_dev = Fields(fdtd, np.shape(psi_e), fdtd.dtype)
        psi_h_dev = Fields(fdtd, np.shape(psi_h), fdtd.dtype)

        if   'opencl' in fdtd.engine.name:
            fdtd.engine.cl.enqueue_write_buffer(fdtd.engine.queue, pcb_e_dev.data, pcb_e).wait()
            fdtd.engine.cl.enqueue_write_buffer(fdtd.engine.queue, pcb_h_dev.data, pcb_h).wait()
            fdtd.engine.cl.enqueue_write_buffer(fdtd.engine.queue, pca_e_dev.data, pca_e).wait()
            fdtd.engine.cl.enqueue_write_buffer(fdtd.engine.queue, pca_h_dev.data, pca_h).wait()
        elif   'cuda' in fdtd.engine.name:
            fdtd.engine.enqueue(fdtd.engine.drv.memcpy_htod, [pcb_e_dev.data, pcb_e]).wait()
            fdtd.engine.enqueue(fdtd.engine.drv.memcpy_htod, [pcb_h_dev.data, pcb_h]).wait()
            fdtd.engine.enqueue(fdtd.engine.drv.memcpy_htod, [pca_e_dev.data, pca_e]).wait()
            fdtd.engine.enqueue(fdtd.engine.drv.memcpy_htod, [pca_h_dev.data, pca_h]).wait()
        else:
            pcb_e_dev[:] = pcb_e
            pcb_h_dev[:] = pcb_h
            pca_e_dev[:] = pca_e
            pca_h_dev[:] = pca_h

        psi_e_dev[:,:] = 0.
        psi_h_dev[:,:] = 0.

        return pcb_e_dev, pcb_h_dev, pca_e_dev, pca_h_dev, psi_e_dev, psi_h_dev

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        axis      = ['x', 'y']
        direction = ['+', '-']
        for part in fdtd.complex_parts:
            for ax in axis:
                for dr in direction:
                    if dr in self.pml_apply[ax]:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pml_e'](*(fdtd.engine.kernel_args['pml_e'][ax][dr][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pml_e'], fdtd.engine.kernel_args['pml_e'][ax][dr][part])
                        elif  'cpu' in fdtd.engine.name:
                            fdtd.engine.set_kernel(fdtd.engine.programs['cpml'].update_e_2d, fdtd.engine.kernel_args['pml_e'][ax][dr][part])
                            evt = fdtd.engine.kernels['pml_e'](*(fdtd.engine.kernel_args['pml_e'][ax][dr][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        axis      = ['x', 'y']
        direction = ['+', '-']
        for part in fdtd.complex_parts:
            for ax in axis:
                for dr in direction:
                    if dr in self.pml_apply[ax]:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pml_h'](*(fdtd.engine.kernel_args['pml_h'][ax][dr][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pml_h'], fdtd.engine.kernel_args['pml_h'][ax][dr][part])
                        elif  'cpu' in fdtd.engine.name:
                            fdtd.engine.set_kernel(fdtd.engine.programs['cpml'].update_h_2d, fdtd.engine.kernel_args['pml_h'][ax][dr][part])
                            evt = fdtd.engine.kernels['pml_h'](*(fdtd.engine.kernel_args['pml_h'][ax][dr][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

class CPML_2dte(CPML_2d):
    def __init__(self, fdtd, pml_apply, pml_thick={'x':(10,10),'y':(10,10)}, alpha0=0., kappa0=1., malpha=4, msigma=4, lnR0=-16.):
        CPML_2d.__init__(self, fdtd, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma)
        if '-' in self.pml_apply['x']:
            self.pcb_exm, self.pcb_hxm, self.pca_exm, self.pca_hxm, self.psi_eyx_m, self.psi_hzx_m = \
                self.set_pml_params('x', '-', alpha0, kappa0, malpha, msigma, lnR0)
        if '+' in self.pml_apply['x']:
            self.pcb_exp, self.pcb_hxp, self.pca_exp, self.pca_hxp, self.psi_eyx_p, self.psi_hzx_p = \
                self.set_pml_params('x', '+', alpha0, kappa0, malpha, msigma, lnR0)
        if '-' in self.pml_apply['y']:
            self.pcb_eym, self.pcb_hym, self.pca_eym, self.pca_hym, self.psi_exy_m, self.psi_hzy_m = \
                self.set_pml_params('y', '-', alpha0, kappa0, malpha, msigma, lnR0)
        if '+' in self.pml_apply['y']:
            self.pcb_eyp, self.pcb_hyp, self.pca_eyp, self.pca_hyp, self.psi_exy_p, self.psi_hzy_p = \
                self.set_pml_params('y', '+', alpha0, kappa0, malpha, msigma, lnR0)

        self.setup()
        if kappa0 != 1. and self.fdtd.is_uniform_grid == True:
            self.fdtd.is_uniform_grid = False
            self.fdtd.setup()

class CPML_2dtm(CPML_2d):
    def __init__(self, fdtd, pml_apply, pml_thick={'x':(10,10),'y':(10,10)}, alpha0=0., kappa0=1., malpha=4, msigma=4, lnR0=-16.):
        CPML_2d.__init__(self, fdtd, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma)
        if '-' in self.pml_apply['x']:
            self.pcb_exm, self.pcb_hxm, self.pca_exm, self.pca_hxm, self.psi_ezx_m, self.psi_hyx_m = \
                self.set_pml_params('x', '-', alpha0, kappa0, malpha, msigma, lnR0)
        if '+' in self.pml_apply['x']:
            self.pcb_exp, self.pcb_hxp, self.pca_exp, self.pca_hxp, self.psi_ezx_p, self.psi_hyx_p = \
                self.set_pml_params('x', '+', alpha0, kappa0, malpha, msigma, lnR0)
        if '-' in self.pml_apply['y']:
            self.pcb_eym, self.pcb_hym, self.pca_eym, self.pca_hym, self.psi_ezy_m, self.psi_hxy_m = \
                self.set_pml_params('y', '-', alpha0, kappa0, malpha, msigma, lnR0)
        if '+' in self.pml_apply['y']:
            self.pcb_eyp, self.pcb_hyp, self.pca_eyp, self.pca_hyp, self.psi_ezy_p, self.psi_hxy_p = \
                self.set_pml_params('y', '+', alpha0, kappa0, malpha, msigma, lnR0)

        self.setup()
        if kappa0 != 1. and self.fdtd.is_uniform_grid == True:
            self.fdtd.is_uniform_grid = False
            self.fdtd.setup()

class CPML_3d:
    def __init__(self, fdtd, pml_apply, pml_thick={'x':(10,10),'y':(10,10),'z':(10,10)}, alpha0=0., kappa0=1., malpha=4, msigma=4, lnR0=-16.):
        self.fdtd = fdtd
        self.fdtd.cores['pml'] = self
        self.fdtd.engine.updates['pml_e'] = self.update_e
        self.fdtd.engine.updates['pml_h'] = self.update_h
        self.pml_apply = pml_apply
        self.pml_thick = pml_thick
        self.alpha = alpha0
        self.kappa = kappa0
        self.alpha_exponent = malpha
        self.sigma_exponent = msigma

        if '-' in self.pml_apply['x']:
            self.pcb_exm, self.pcb_hxm, self.pca_exm, self.pca_hxm, self.psi_ezx_m, self.psi_eyx_m, self.psi_hyx_m, self.psi_hzx_m = \
                self.set_pml_params('x', '-', alpha0, kappa0, malpha, msigma, lnR0)
        if '+' in self.pml_apply['x']:
            self.pcb_exp, self.pcb_hxp, self.pca_exp, self.pca_hxp, self.psi_ezx_p, self.psi_eyx_p, self.psi_hyx_p, self.psi_hzx_p = \
                self.set_pml_params('x', '+', alpha0, kappa0, malpha, msigma, lnR0)
        if '-' in self.pml_apply['y']:
            self.pcb_eym, self.pcb_hym, self.pca_eym, self.pca_hym, self.psi_exy_m, self.psi_ezy_m, self.psi_hzy_m, self.psi_hxy_m = \
                self.set_pml_params('y', '-', alpha0, kappa0, malpha, msigma, lnR0)
        if '+' in self.pml_apply['y']:
            self.pcb_eyp, self.pcb_hyp, self.pca_eyp, self.pca_hyp, self.psi_exy_p, self.psi_ezy_p, self.psi_hzy_p, self.psi_hxy_p = \
                self.set_pml_params('y', '+', alpha0, kappa0, malpha, msigma, lnR0)
        if '-' in self.pml_apply['z']:
            self.pcb_ezm, self.pcb_hzm, self.pca_ezm, self.pca_hzm, self.psi_eyz_m, self.psi_exz_m, self.psi_hxz_m, self.psi_hyz_m = \
                self.set_pml_params('z', '-', alpha0, kappa0, malpha, msigma, lnR0)
        if '+' in self.pml_apply['z']:
            self.pcb_ezp, self.pcb_hzp, self.pca_ezp, self.pca_hzp, self.psi_eyz_p, self.psi_exz_p, self.psi_hxz_p, self.psi_hyz_p = \
                self.set_pml_params('z', '+', alpha0, kappa0, malpha, msigma, lnR0)

        self.setup()
        if kappa0 != 1. and self.fdtd.is_uniform_grid:
            self.fdtd.is_uniform_grid = False
            self.fdtd.setup()

    def setup(self):
        fdtd = self.fdtd
        code_ces_prev = [', __GLOBAL__ __FLOAT__* ce1', ', __GLOBAL__ __FLOAT__* ce2', ', __GLOBAL__ __FLOAT__* ce', \
                         'ce1[idx0]', 'ce2[idx0]', 'ce[idx0]']
        code_chs_prev = [', __GLOBAL__ __FLOAT__* ch1', ', __GLOBAL__ __FLOAT__* ch2', ', __GLOBAL__ __FLOAT__* ch', \
                         'ch1[idx0]', 'ch2[idx0]', 'ch[idx0]']
        code_rds_prev = [', __GLOBAL__ __FLOAT__* ds', 'ds[ids]*']
        ces = '%s' % self.fdtd.dt
        chs = '%s' % self.fdtd.dt
        if    self.fdtd.is_electric    : code_ces_post = code_ces_prev
        else                           : code_ces_post = ['', '', '', ces, ces, ces]
        if    self.fdtd.is_magnetic    : code_chs_post = code_chs_prev
        else                           : code_chs_post = ['', '', '', chs, chs, chs]
        if    self.fdtd.is_uniform_grid: code_rds_post = ['', '']
        else                           : code_rds_post = code_rds_prev

        if 'cpu' in fdtd.engine.name:
            omp_ces_prev = [', ce1', ', ce2', ', ce']
            code_ces_prev += omp_ces_prev
            if self.fdtd.is_electric    : code_ces_post += omp_ces_prev
            else                        : code_ces_post += ['', '', '']
            omp_chs_prev = [', ch1', ', ch2', ', ch']
            code_chs_prev += omp_chs_prev
            if self.fdtd.is_magnetic    : code_chs_post += omp_chs_prev
            else                        : code_chs_post += ['', '', '']
            omp_rds_prev = [', ds']
            code_rds_prev += omp_rds_prev
            if self.fdtd.is_uniform_grid: code_rds_post += ['']
            else                        : code_rds_post += omp_rds_prev

        code = template_to_code(fdtd.engine.templates['cpml'], \
                                code_ces_prev + code_chs_prev + code_rds_prev + fdtd.engine.code_prev, \
                                code_ces_post + code_chs_post + code_rds_post + fdtd.engine.code_post)

        fdtd.engine.programs['cpml'] = fdtd.engine.build(code)
        fdtd.engine.kernel_args['pml_e'] = {'x':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}, \
                                            'y':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}, \
                                            'z':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}}
        fdtd.engine.kernel_args['pml_h'] = {'x':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}, \
                                            'y':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}, \
                                            'z':{'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}}

        for part in self.fdtd.complex_parts:
            update_e_args_xp = [np.int32(0), np.int32(1), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['x'][1]), \
                                fdtd.ez.__dict__[part].data, fdtd.ey.__dict__[part].data, \
                                fdtd.hy.__dict__[part].data, fdtd.hz.__dict__[part].data]
            update_h_args_xp = [np.int32(0), np.int32(1), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['x'][1]), \
                                fdtd.ez.__dict__[part].data, fdtd.ey.__dict__[part].data, \
                                fdtd.hy.__dict__[part].data, fdtd.hz.__dict__[part].data]
            update_e_args_xm = [np.int32(0), np.int32(0), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['x'][0]), \
                                fdtd.ez.__dict__[part].data, fdtd.ey.__dict__[part].data, \
                                fdtd.hy.__dict__[part].data, fdtd.hz.__dict__[part].data]
            update_h_args_xm = [np.int32(0), np.int32(0), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['x'][0]), \
                                fdtd.ez.__dict__[part].data, fdtd.ey.__dict__[part].data, \
                                fdtd.hy.__dict__[part].data, fdtd.hz.__dict__[part].data]
            update_e_args_yp = [np.int32(1), np.int32(1), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['y'][1]), \
                                fdtd.ex.__dict__[part].data, fdtd.ez.__dict__[part].data, \
                                fdtd.hz.__dict__[part].data, fdtd.hx.__dict__[part].data]
            update_h_args_yp = [np.int32(1), np.int32(1), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['y'][1]), \
                                fdtd.ex.__dict__[part].data, fdtd.ez.__dict__[part].data, \
                                fdtd.hz.__dict__[part].data, fdtd.hx.__dict__[part].data]
            update_e_args_ym = [np.int32(1), np.int32(0), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['y'][0]), \
                                fdtd.ex.__dict__[part].data, fdtd.ez.__dict__[part].data, \
                                fdtd.hz.__dict__[part].data, fdtd.hx.__dict__[part].data]
            update_h_args_ym = [np.int32(1), np.int32(0), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['y'][0]), \
                                fdtd.ex.__dict__[part].data, fdtd.ez.__dict__[part].data, \
                                fdtd.hz.__dict__[part].data, fdtd.hx.__dict__[part].data]
            update_e_args_zp = [np.int32(2), np.int32(1), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['z'][1]), \
                                fdtd.ey.__dict__[part].data, fdtd.ex.__dict__[part].data, \
                                fdtd.hx.__dict__[part].data, fdtd.hy.__dict__[part].data]
            update_h_args_zp = [np.int32(2), np.int32(1), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['z'][1]), \
                                fdtd.ey.__dict__[part].data, fdtd.ex.__dict__[part].data, \
                                fdtd.hx.__dict__[part].data, fdtd.hy.__dict__[part].data]
            update_e_args_zm = [np.int32(2), np.int32(0), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['z'][0]), \
                                fdtd.ey.__dict__[part].data, fdtd.ex.__dict__[part].data, \
                                fdtd.hx.__dict__[part].data, fdtd.hy.__dict__[part].data]
            update_h_args_zm = [np.int32(2), np.int32(0), \
                                np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                np.int32(self.pml_thick['z'][0]), \
                                fdtd.ey.__dict__[part].data, fdtd.ex.__dict__[part].data, \
                                fdtd.hx.__dict__[part].data, fdtd.hy.__dict__[part].data]
            if fdtd.is_electric:
                update_e_args_xp += [fdtd.ce2z.data, fdtd.ce2y.data]
                update_e_args_xm += [fdtd.ce2z.data, fdtd.ce2y.data]
                update_e_args_yp += [fdtd.ce2x.data, fdtd.ce2z.data]
                update_e_args_ym += [fdtd.ce2x.data, fdtd.ce2z.data]
                update_e_args_zp += [fdtd.ce2y.data, fdtd.ce2x.data]
                update_e_args_zm += [fdtd.ce2y.data, fdtd.ce2x.data]
            if fdtd.is_magnetic:
                update_h_args_xp += [fdtd.ch2y.data, fdtd.ch2z.data]
                update_h_args_xm += [fdtd.ch2y.data, fdtd.ch2z.data]
                update_h_args_yp += [fdtd.ch2z.data, fdtd.ch2x.data]
                update_h_args_ym += [fdtd.ch2z.data, fdtd.ch2x.data]
                update_h_args_zp += [fdtd.ch2x.data, fdtd.ch2y.data]
                update_h_args_zm += [fdtd.ch2x.data, fdtd.ch2y.data]
            if not fdtd.is_uniform_grid:
                update_e_args_xp += [fdtd.rdx_e.data]
                update_h_args_xp += [fdtd.rdx_h.data]
                update_e_args_xm += [fdtd.rdx_e.data]
                update_h_args_xm += [fdtd.rdx_h.data]
                update_e_args_yp += [fdtd.rdy_e.data]
                update_h_args_yp += [fdtd.rdy_h.data]
                update_e_args_ym += [fdtd.rdy_e.data]
                update_h_args_ym += [fdtd.rdy_h.data]
                update_e_args_zp += [fdtd.rdz_e.data]
                update_h_args_zp += [fdtd.rdz_h.data]
                update_e_args_zm += [fdtd.rdz_e.data]
                update_h_args_zm += [fdtd.rdz_h.data]
            if '+' in self.pml_apply['x']:
                update_e_args_xp += [self.pcb_exp.data, self.pca_exp.data, \
                                     self.psi_ezx_p.__dict__[part].data, \
                                     self.psi_eyx_p.__dict__[part].data]
                update_h_args_xp += [self.pcb_hxp.data, self.pca_hxp.data, \
                                     self.psi_hyx_p.__dict__[part].data, \
                                     self.psi_hzx_p.__dict__[part].data]
            if '-' in self.pml_apply['x']:
                update_e_args_xm += [self.pcb_exm.data, self.pca_exm.data, \
                                     self.psi_ezx_m.__dict__[part].data, \
                                     self.psi_eyx_m.__dict__[part].data]
                update_h_args_xm += [self.pcb_hxm.data, self.pca_hxm.data, \
                                     self.psi_hyx_m.__dict__[part].data,
                                     self.psi_hzx_m.__dict__[part].data]
            if '+' in self.pml_apply['y']:
                update_e_args_yp += [self.pcb_eyp.data, self.pca_eyp.data, \
                                     self.psi_exy_p.__dict__[part].data,\
                                     self.psi_ezy_p.__dict__[part].data]
                update_h_args_yp += [self.pcb_hyp.data, self.pca_hyp.data, \
                                     self.psi_hzy_p.__dict__[part].data, \
                                     self.psi_hxy_p.__dict__[part].data]
            if '-' in self.pml_apply['y']:
                update_e_args_ym += [self.pcb_eym.data, self.pca_eym.data, \
                                     self.psi_exy_m.__dict__[part].data, \
                                     self.psi_ezy_m.__dict__[part].data]
                update_h_args_ym += [self.pcb_hym.data, self.pca_hym.data, \
                                     self.psi_hzy_m.__dict__[part].data, \
                                     self.psi_hxy_m.__dict__[part].data]
            if '+' in self.pml_apply['z']:
                update_e_args_zp += [self.pcb_ezp.data, self.pca_ezp.data, \
                                     self.psi_eyz_p.__dict__[part].data,\
                                     self.psi_exz_p.__dict__[part].data]
                update_h_args_zp += [self.pcb_hzp.data, self.pca_hzp.data, \
                                     self.psi_hxz_p.__dict__[part].data, \
                                     self.psi_hyz_p.__dict__[part].data]
            if '-' in self.pml_apply['z']:
                update_e_args_zm += [self.pcb_ezm.data, self.pca_ezm.data, \
                                     self.psi_eyz_m.__dict__[part].data, \
                                     self.psi_exz_m.__dict__[part].data]
                update_h_args_zm += [self.pcb_hzm.data, self.pca_hzm.data, \
                                     self.psi_hxz_m.__dict__[part].data, \
                                     self.psi_hyz_m.__dict__[part].data]

            if 'opencl' in fdtd.engine.name:
                gs_xp = cl_global_size(self.pml_thick['x'][1]*fdtd.ny*fdtd.nz, fdtd.engine.ls)
                gs_xm = cl_global_size(self.pml_thick['x'][0]*fdtd.ny*fdtd.nz, fdtd.engine.ls)
                gs_yp = cl_global_size(self.pml_thick['y'][1]*fdtd.nz*fdtd.nx, fdtd.engine.ls)
                gs_ym = cl_global_size(self.pml_thick['y'][0]*fdtd.nz*fdtd.nx, fdtd.engine.ls)
                gs_zp = cl_global_size(self.pml_thick['z'][1]*fdtd.nx*fdtd.ny, fdtd.engine.ls)
                gs_zm = cl_global_size(self.pml_thick['z'][0]*fdtd.nx*fdtd.ny, fdtd.engine.ls)
            else:
                gs_xp = self.pml_thick['x'][1]*fdtd.ny*fdtd.nz
                gs_xm = self.pml_thick['x'][0]*fdtd.ny*fdtd.nz
                gs_yp = self.pml_thick['y'][1]*fdtd.nz*fdtd.nx
                gs_ym = self.pml_thick['y'][0]*fdtd.nz*fdtd.nx
                gs_zp = self.pml_thick['z'][1]*fdtd.nx*fdtd.ny
                gs_zm = self.pml_thick['z'][0]*fdtd.nx*fdtd.ny

            if '+' in self.pml_apply['x']:
                fdtd.engine.kernel_args['pml_e']['x']['+'][part] = [fdtd.engine.queue, (gs_xp,), (fdtd.engine.ls,)] + update_e_args_xp
                fdtd.engine.kernel_args['pml_h']['x']['+'][part] = [fdtd.engine.queue, (gs_xp,), (fdtd.engine.ls,)] + update_h_args_xp
            if '-' in self.pml_apply['x']:
                fdtd.engine.kernel_args['pml_e']['x']['-'][part] = [fdtd.engine.queue, (gs_xm,), (fdtd.engine.ls,)] + update_e_args_xm
                fdtd.engine.kernel_args['pml_h']['x']['-'][part] = [fdtd.engine.queue, (gs_xm,), (fdtd.engine.ls,)] + update_h_args_xm
            if '+' in self.pml_apply['y']:
                fdtd.engine.kernel_args['pml_e']['y']['+'][part] = [fdtd.engine.queue, (gs_yp,), (fdtd.engine.ls,)] + update_e_args_yp
                fdtd.engine.kernel_args['pml_h']['y']['+'][part] = [fdtd.engine.queue, (gs_yp,), (fdtd.engine.ls,)] + update_h_args_yp
            if '-' in self.pml_apply['y']:
                fdtd.engine.kernel_args['pml_e']['y']['-'][part] = [fdtd.engine.queue, (gs_ym,), (fdtd.engine.ls,)] + update_e_args_ym
                fdtd.engine.kernel_args['pml_h']['y']['-'][part] = [fdtd.engine.queue, (gs_ym,), (fdtd.engine.ls,)] + update_h_args_ym
            if '+' in self.pml_apply['z']:
                fdtd.engine.kernel_args['pml_e']['z']['+'][part] = [fdtd.engine.queue, (gs_zp,), (fdtd.engine.ls,)] + update_e_args_zp
                fdtd.engine.kernel_args['pml_h']['z']['+'][part] = [fdtd.engine.queue, (gs_zp,), (fdtd.engine.ls,)] + update_h_args_zp
            if '-' in self.pml_apply['z']:
                fdtd.engine.kernel_args['pml_e']['z']['-'][part] = [fdtd.engine.queue, (gs_zm,), (fdtd.engine.ls,)] + update_e_args_zm
                fdtd.engine.kernel_args['pml_h']['z']['-'][part] = [fdtd.engine.queue, (gs_zm,), (fdtd.engine.ls,)] + update_h_args_zm

        if 'opencl' in fdtd.engine.name:
            fdtd.engine.kernels['pml_e'] = fdtd.engine.programs['cpml'].update_e_3d
            fdtd.engine.kernels['pml_h'] = fdtd.engine.programs['cpml'].update_h_3d
        elif 'cuda' in fdtd.engine.name:
            fdtd.engine.kernels['pml_e'] = fdtd.engine.get_function(fdtd.engine.programs['cpml'], 'update_e_3d')
            fdtd.engine.kernels['pml_h'] = fdtd.engine.get_function(fdtd.engine.programs['cpml'], 'update_h_3d')
            for ax in ['x', 'y', 'z']:
                if   '+' in self.pml_apply[ax]:
                    axis = ax
                    dirc = '+'
                elif '-' in self.pml_apply[ax]:
                    axis = ax
                    dirc = '-'
            fdtd.engine.prepare(fdtd.engine.kernels['pml_e'], fdtd.engine.kernel_args['pml_e'][axis][dirc]['real'])
            fdtd.engine.prepare(fdtd.engine.kernels['pml_h'], fdtd.engine.kernel_args['pml_h'][axis][dirc]['real'])
        elif  'cpu' in fdtd.engine.name:
            for ax in ['x', 'y', 'z']:
                if   '+' in self.pml_apply[ax]:
                    axis = ax
                    dirc = '+'
                elif '-' in self.pml_apply[ax]:
                    axis = ax
                    dirc = '-'
            fdtd.engine.kernels['pml_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['cpml'].update_e_3d, fdtd.engine.kernel_args['pml_e'][axis][dirc]['real'])
            fdtd.engine.kernels['pml_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['cpml'].update_h_3d, fdtd.engine.kernel_args['pml_h'][axis][dirc]['real'])

    def set_pml_params(self, pml_axis, pml_direction, alpha_max, kappa_max, malpha, msigma, lnR0):
        fdtd = self.fdtd
        real_dtype = comp_to_real(fdtd.dtype)

        if pml_direction == '-': direction = 0
        if pml_direction == '+': direction = 1

        npml = self.pml_thick[pml_axis][direction]

        if pml_direction == '-':
            if pml_axis == 'x':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdx_e[1:npml+1]).sum())
            if pml_axis == 'y':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdy_e[1:npml+1]).sum())
            if pml_axis == 'z':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdz_e[1:npml+1]).sum())
        if pml_direction == '+':
            if pml_axis == 'x':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdx_h[-npml:]).sum())
            if pml_axis == 'y':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdy_h[-npml:]).sum())
            if pml_axis == 'z':
                sigma_max = (msigma+1)*(-lnR0)/(2.*(1/fdtd.rdz_h[-npml:]).sum())

        if pml_direction == '-':
            plnml_i = np.arange(.5, npml  , dtype=real_dtype)[::-1]/npml
            plnml_o = np.arange(1., npml+1, dtype=real_dtype)[::-1]/npml

            plnla_i = np.arange(1., npml+1, dtype=real_dtype)      /npml
            plnla_o = np.arange(.5, npml  , dtype=real_dtype)      /npml

            alpha_e = alpha_max*(plnla_i)**malpha
            sigma_e = sigma_max*(plnml_i)**msigma
            kappa_e = 1. + (kappa_max - 1)*(plnml_i)**msigma

            alpha_h = alpha_max*(plnla_o)**malpha
            sigma_h = sigma_max*(plnml_o)**msigma
            kappa_h = 1. + (kappa_max - 1)*(plnml_o)**msigma

        if pml_direction == '+':
            plnml_i = np.arange(.5, npml  , dtype=real_dtype)      /npml
            plnml_o = np.arange(1., npml+1, dtype=real_dtype)      /npml

            plnla_i = np.arange(1., npml+1, dtype=real_dtype)[::-1]/npml
            plnla_o = np.arange(.5, npml  , dtype=real_dtype)[::-1]/npml

            alpha_e = alpha_max*(plnla_o)**malpha
            sigma_e = sigma_max*(plnml_o)**msigma
            kappa_e = 1. + (kappa_max - 1)*(plnml_o)**msigma

            alpha_h = alpha_max*(plnla_i)**malpha
            sigma_h = sigma_max*(plnml_i)**msigma
            kappa_h = 1. + (kappa_max - 1)*(plnml_i)**msigma

        pcb_e = np.exp(-(sigma_e/kappa_e + alpha_e)*fdtd.dt)
        pcb_h = np.exp(-(sigma_h/kappa_h + alpha_h)*fdtd.dt)
        pca_e = (sigma_e/(sigma_e + kappa_e*alpha_e))*(pcb_e - 1.)
        pca_h = (sigma_h/(sigma_h + kappa_h*alpha_h))*(pcb_h - 1.)

        if pml_direction == '-':
            sle = slice(1,  npml+1)
            slh = slice(None, npml)
            if pml_axis == 'x':
                fdtd.rdx_e[sle] /= kappa_e
                fdtd.rdx_h[slh] /= kappa_h
            if pml_axis == 'y':
                fdtd.rdy_e[sle] /= kappa_e
                fdtd.rdy_h[slh] /= kappa_h
            if pml_axis == 'z':
                fdtd.rdz_e[sle] /= kappa_e
                fdtd.rdz_h[slh] /= kappa_h

        if pml_direction == '+':
            sle = slice(-npml, None)
            slh = slice(-npml-1, -1)
            if pml_axis == 'x':
                fdtd.rdx_e[sle] /= kappa_e
                fdtd.rdx_h[slh] /= kappa_h
            if pml_axis == 'y':
                fdtd.rdy_e[sle] /= kappa_e
                fdtd.rdy_h[slh] /= kappa_h
            if pml_axis == 'z':
                fdtd.rdz_e[sle] /= kappa_e
                fdtd.rdz_h[slh] /= kappa_h

        if pml_axis == 'x':	psi_e1, psi_e2, psi_h1, psi_h2 = [Fields(fdtd, (npml, fdtd.ny, fdtd.nz), fdtd.dtype) for i in xrange(4)]
        if pml_axis == 'y':	psi_e1, psi_e2, psi_h1, psi_h2 = [Fields(fdtd, (fdtd.nx, npml, fdtd.nz), fdtd.dtype) for i in xrange(4)]
        if pml_axis == 'z':	psi_e1, psi_e2, psi_h1, psi_h2 = [Fields(fdtd, (fdtd.nx, fdtd.ny, npml), fdtd.dtype) for i in xrange(4)]

        pcb_e_dev = Fields(fdtd, np.shape(pcb_e), real_dtype, mem_flag='r')
        pcb_h_dev = Fields(fdtd, np.shape(pcb_h), real_dtype, mem_flag='r')
        pca_e_dev = Fields(fdtd, np.shape(pca_e), real_dtype, mem_flag='r')
        pca_h_dev = Fields(fdtd, np.shape(pca_h), real_dtype, mem_flag='r')

        if 'opencl' in fdtd.engine.name:
            fdtd.engine.cl.enqueue_write_buffer(fdtd.engine.queue, pcb_e_dev.data, pcb_e).wait()
            fdtd.engine.cl.enqueue_write_buffer(fdtd.engine.queue, pcb_h_dev.data, pcb_h).wait()
            fdtd.engine.cl.enqueue_write_buffer(fdtd.engine.queue, pca_e_dev.data, pca_e).wait()
            fdtd.engine.cl.enqueue_write_buffer(fdtd.engine.queue, pca_h_dev.data, pca_h).wait()
        elif 'cuda' in fdtd.engine.name:
            fdtd.engine.enqueue(fdtd.engine.drv.memcpy_htod, [pcb_e_dev.data, pcb_e]).wait()
            fdtd.engine.enqueue(fdtd.engine.drv.memcpy_htod, [pcb_h_dev.data, pcb_h]).wait()
            fdtd.engine.enqueue(fdtd.engine.drv.memcpy_htod, [pca_e_dev.data, pca_e]).wait()
            fdtd.engine.enqueue(fdtd.engine.drv.memcpy_htod, [pca_h_dev.data, pca_h]).wait()
        else:
            pcb_e_dev[:] = pcb_e
            pcb_h_dev[:] = pcb_h
            pca_e_dev[:] = pca_e
            pca_h_dev[:] = pca_h

        psi_e1[:,:,:] = 0.
        psi_e2[:,:,:] = 0.
        psi_h1[:,:,:] = 0.
        psi_h2[:,:,:] = 0.

        return pcb_e_dev, pcb_h_dev, pca_e_dev, pca_h_dev, psi_e1, psi_e2, psi_h1, psi_h2

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        axis = ['x', 'y', 'z']
        direction = ['+', '-']
        for part in fdtd.complex_parts:
            for ax in axis:
                for dr in direction:
                    if dr in self.pml_apply[ax]:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pml_e'](*(fdtd.engine.kernel_args['pml_e'][ax][dr][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pml_e'], fdtd.engine.kernel_args['pml_e'][ax][dr][part])
                        elif  'cpu' in fdtd.engine.name:
                            fdtd.engine.set_kernel(fdtd.engine.programs['cpml'].update_e_3d, fdtd.engine.kernel_args['pml_e'][ax][dr][part])
                            evt = fdtd.engine.kernels['pml_e'](*(fdtd.engine.kernel_args['pml_e'][ax][dr][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        axis = ['x', 'y', 'z']
        direction = ['+', '-']
        for part in fdtd.complex_parts:
            for ax in axis:
                for dr in direction:
                    if dr in self.pml_apply[ax]:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pml_h'](*(fdtd.engine.kernel_args['pml_h'][ax][dr][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pml_h'], fdtd.engine.kernel_args['pml_h'][ax][dr][part])
                        elif  'cpu' in fdtd.engine.name:
                            fdtd.engine.set_kernel(fdtd.engine.programs['cpml'].update_h_3d, fdtd.engine.kernel_args['pml_h'][ax][dr][part])
                            evt = fdtd.engine.kernels['pml_h'](*(fdtd.engine.kernel_args['pml_h'][ax][dr][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts


def PML(fdtd, pml_apply, pml_thick={'x':(10,10), 'y':(10,10), 'z':(10,10)}, alpha0=0., kappa0=1., malpha=4, msigma=4, lnR0=-16.):
    if   fdtd.mode == '1D':
        return CPML_1d(fdtd, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)
    elif fdtd.mode == '2DTE':
        return CPML_2dte(fdtd, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)
    elif fdtd.mode == '2DTM':
        return CPML_2dtm(fdtd, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)
    elif fdtd.mode == '3D':
        return CPML_3d(fdtd, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)
