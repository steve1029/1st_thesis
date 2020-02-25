# Author  : Ki-Hwan Kim
# Purpose : Update function for the CPML (Convolutional Perfectly Matched Layer)
# Target  : CPU using Numpy
# Created : 2012-01-19
# Modified: 

from __future__ import division
import numpy as np

from kemp.fdtd3d.util import common
from fields import Fields


class Pml:
    def __init__(self, fields, directions, npml=10, sigma_max=4, kappa_max=1, alpha_max=0, m_sigma=3, m_alpha=1):
        common.check_type('fields', fields, Fields)
        common.check_type('directions', directions, (list, tuple), str)
        common.check_type('npml', npml, int)
        common.check_type('sigma_max', sigma_max, (int, float))
        common.check_type('kappa_max', kappa_max, (int, float))
        common.check_type('alpha_max', alpha_max, (int, float))
        common.check_type('m_sigma', m_sigma, (int, float))
        common.check_type('m_alpha', m_alpha, (int, float))

        assert len(directions) == 3
        for axis in directions:
            assert axis in ['+', '-', '+-', '']

        # local variables
        dt = fields.dt
        nx, ny, nz = fields.ns
        dtype = fields.dtype

        # allocations
        psi_xs = [np.zeros((2*npml, ny, nz), dtype) for i in range(4)]
        psi_ys = [np.zeros((nx, 2*npml, nz), dtype) for i in range(4)]
        psi_zs = [np.zeros((nx, ny, 2*npml), dtype) for i in range(4)]

        i_half = np.arange(0.5, npml)
        i_one = np.arange(1, npml+1)
        sigma_half = sigma_max * (i_half / npml) ** m_sigma
        sigma_one = sigma_max * (i_one / npml) ** m_sigma
        kappa_half = 1 + (kappa_max - 1) * (i_half / npml) ** m_sigma
        kappa_one = 1 + (kappa_max - 1) * (i_one / npml) ** m_sigma
        alpha_half = alpha_max * ((npml - i_half) / npml) ** m_alpha
        alpha_one = alpha_max * ((npml - i_one) / npml) ** m_alpha

        pcb_half = np.exp(-(sigma_half / kappa_half + alpha_half) * dt)
        pcb_one = np.exp(-(sigma_one / kappa_one + alpha_one) * dt)
        pca_half = sigma_half / (sigma_half + alpha_half * kappa_half) * (pcb_half - 1)
        pca_one = sigma_one / (sigma_one + alpha_one * kappa_one) * (pcb_one - 1)

        # modify reciprocal ds
        if kappa_max != 1:
            erds, hrds = fields.get_rds()
            for erd, hrd, pms in zip(erds, hrds, directions):
                if '+' in pms:
                    erd[-npml:] /= kappa_half
                    hrd[-npml:] /= kappa_one
                if '-' in pms:
                    erd[:npml] /= kappa_one
                    hrd[:npml] /= kappa_half
        
        # global variables
        self.mainf = fields
        self.directions = directions
        self.npml = npml
        self.psi_xs = psi_xs
        self.psi_ys = psi_ys
        self.psi_zs = psi_zs
        self.pcs_half = [pcb_half, pca_half]
        self.pcs_one = [pcb_one, pca_one]

        # append to the update list
        self.priority_type = 'pml'
        fields.append_instance(self)


    def update(self, sl, slc, sl1, sl2, sl3, f1, f2, f3, f4, psi1, psi2, pcb, pca, c1, c2):
        psi1[sl] = pcb[slc] * psi1[sl] + pca[slc] * (f3[sl2] - f3[sl3])
        psi2[sl] = pcb[slc] * psi2[sl] + pca[slc] * (f4[sl2] - f4[sl3])
        c1 = c1[sl1] if isinstance(c1, np.ndarray) else c1
        c2 = c2[sl1] if isinstance(c2, np.ndarray) else c2
        f1[sl1] -= c1 * psi1[sl]
        f2[sl1] += c2 * psi2[sl]


    def update_e(self):
        npml = self.npml
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        cex, cey, cez = self.mainf.ces
        psi_eyx, psi_ezx, psi_hyx, psi_hzx = self.psi_xs
        psi_ezy, psi_exy, psi_hzy, psi_hxy = self.psi_ys 
        psi_exz, psi_eyz, psi_hxz, psi_hyz = self.psi_zs
        pcb_half, pca_half = self.pcs_half
        pcb_one, pca_one = self.pcs_one
        directions = self.directions
        sln = slice(None, None)
        nax = np.newaxis

        if '+' in directions[0]:
            sl = (slice(-npml, None), sln, sln)
            sls = (slice(-npml-1, -1), sln, sln)
            slc = (sln, nax, nax)
            self.update(sl, slc, sls, sl, sls, ey, ez, hz, hy, psi_eyx, psi_ezx, pcb_half, pca_half, cey, cez)

        if '-' in directions[0]:
            sl = (slice(None, npml), sln, sln)
            sls = (slice(1, npml+1), sln, sln)
            slc = (slice(None, None, -1), nax, nax)
            self.update(sl, slc, sl, sls, sl, ey, ez, hz, hy, psi_eyx, psi_ezx, pcb_one, pca_one, cey, cez)
            
        if '+' in directions[1]:
            sl = (sln, slice(-npml, None), sln)
            sls = (sln, slice(-npml-1, -1), sln)
            slc = (sln, nax)
            self.update(sl, slc, sls, sl, sls, ez, ex, hx, hz, psi_ezy, psi_exy, pcb_half, pca_half, cez, cex)
            
        if '-' in directions[1]:
            sl = (sln, slice(None, npml), sln)
            sls = (sln, slice(1, npml+1), sln)
            slc = (slice(None, None, -1), nax)
            self.update(sl, slc, sl, sls, sl, ez, ex, hx, hz, psi_ezy, psi_exy, pcb_one, pca_one, cez, cex)
            
        if '+' in directions[2]:
            sl = (sln, sln, slice(-npml, None))
            sls = (sln, sln, slice(-npml-1, -1))
            slc = sln
            self.update(sl, slc, sls, sl, sls, ex, ey, hy, hx, psi_exz, psi_eyz, pcb_half, pca_half, cex, cey)
            
        if '-' in directions[2]:
            sl = (sln, sln, slice(None, npml))
            sls = (sln, sln, slice(1, npml+1))
            slc = slice(None, None, -1)
            self.update(sl, slc, sl, sls, sl, ex, ey, hy, hx, psi_exz, psi_eyz, pcb_one, pca_one, cex, cey)
            

    def update_h(self):
        npml = self.npml
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        chx, chy, chz = self.mainf.chs
        psi_eyx, psi_ezx, psi_hyx, psi_hzx = self.psi_xs
        psi_ezy, psi_exy, psi_hzy, psi_hxy = self.psi_ys 
        psi_exz, psi_eyz, psi_hxz, psi_hyz = self.psi_zs
        pcb_half, pca_half = self.pcs_half
        pcb_one, pca_one = self.pcs_one
        directions = self.directions
        sln = slice(None, None)
        nax = np.newaxis

        if '+' in directions[0]:
            sl = (slice(-npml, None), sln, sln)
            sls = (slice(-npml-1, -1), sln, sln)
            slc = (sln, nax, nax)
            self.update(sl, slc, sl, sl, sls, hz, hy, ey, ez, psi_hzx, psi_hyx, pcb_one, pca_one, chz, chy)

        if '-' in directions[0]:
            sl = (slice(None, npml), sln, sln)
            sls = (slice(1, npml+1), sln, sln)
            slc = (slice(None, None, -1), nax, nax)
            self.update(sl, slc, sls, sls, sl, hz, hy, ey, ez, psi_hzx, psi_hyx, pcb_half, pca_half, chz, chy)
            
        if '+' in directions[1]:
            sl = (sln, slice(-npml, None), sln)
            sls = (sln, slice(-npml-1, -1), sln)
            slc = (sln, nax)
            self.update(sl, slc, sl, sl, sls, hx, hz, ez, ex, psi_hxy, psi_hzy, pcb_one, pca_one, chx, chz)
            
        if '-' in directions[1]:
            sl = (sln, slice(None, npml), sln)
            sls = (sln, slice(1, npml+1), sln)
            slc = (slice(None, None, -1), nax)
            self.update(sl, slc, sls, sls, sl, hx, hz, ez, ex, psi_hxy, psi_hzy, pcb_half, pca_half, chx, chz)
            
        if '+' in directions[2]:
            sl = (sln, sln, slice(-npml, None))
            sls = (sln, sln, slice(-npml-1, -1))
            slc = sln
            self.update(sl, slc, sl, sl, sls, hy, hx, ex, ey, psi_hyz, psi_hxz, pcb_one, pca_one, chy, chx)
            
        if '-' in directions[2]:
            sl = (sln, sln, slice(None, npml))
            sls = (sln, sln, slice(1, npml+1))
            slc = slice(None, None, -1)
            self.update(sl, slc, sls, sls, sl, hy, hx, ex, ey, psi_hyz, psi_hxz, pcb_half, pca_half, chy, chx)