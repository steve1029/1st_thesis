from __future__ import division
import numpy as np

from kemp.fdtd3d.util import common
from fields import Fields


class Pml:
    def __init__(self, fields, directions, npml=10, sigma_max=4, kappa_max=1, alpha_max=0, m_sigma=3, m_alpha=1):
        common.check_type('fields', fields, Fields)
        common.check_type('directions', directions, (list, tuple), str)

        assert len(directions) == 3
        for axis in directions:
            assert axis in ['+', '-', '+-', '']

        # local variables
        dt = fields.dt
        nx, ny, nz = fields.ns
        dtype = fields.dtype

        # allocations
        psi_xs = [np.zeros((2*npml + 2, ny, nz), dtype) for i in range(4)]
        psi_ys = [np.zeros((nx, 2*npml + 2, nz), dtype) for i in range(4)]
        psi_zs = [np.zeros((nx, 2*npml + 2, nz), dtype) for i in range(4)]

        i_half = np.arange(0.5, npml)
        i_one = np.arange(1, npml+1)
        sigma_half = sigma_max * (i_half / npml) ** m_sigma
        sigma_one = sigma_max * (i_one / npml) ** m_sigma
        kappa_half = 1 + (kappa_max - 1) * (i_half / npml) ** m_sigma
        kappa_one = 1 + (kappa_max - 1) * (i_one / npml) ** m_sigma
        alpha_half = alpha_max * ((npml - i_half) / npml) ** m_alpha
        alpha_one = alpha_max * ((npml - i_one) / npml) ** m_alpha

        g_half = (kappa_half * alpha_half + sigma_half) * dt + 2 * kappa_half
        g_one = (kappa_one * alpha_one + sigma_one) * dt + 2 * kappa_one
        pca_half = 4 * kappa_half / g_half - 1
        pca_one = 4 * kappa_one / g_one - 1
        pcb_half = (alpha_half * dt - 2 + 4 * kappa_half) / g_half - 1
        pcb_one = (alpha_one * dt - 2 + 4 * kappa_one) / g_one - 1
        pcc_half = (alpha_half * dt + 2) / g_half - 1
        pcc_one = (alpha_one * dt + 2) / g_one - 1

        # global variables
        self.mainf = fields
        self.directions = directions
        self.npml = npml
        self.psi_xs = psi_xs
        self.psi_ys = psi_ys
        self.psi_zs = psi_zs
        self.pcs_half = [pca_half, pcb_half, pcc_half]
        self.pcs_one = [pca_one, pcb_one, pcc_one]

        # append to the update list
        self.priority_type = 'pml'
        fields.append_instance(self)


    def update(self, sl, sls, slc, f1, f2, f3, f4, psi1, psi2, psi3, psi4, pca, pcb, pcc, c1, c2):

        '''
        print 'sl', sl
        print 'sls', sls
        print 'f1', f1[sl].shape
        print 'c1', c1[sl].shape
        print 'psi4', psi4[sls].shape
        print 'psi4', psi4[sl].shape
        '''
        f1[sl] -= c1[sl] * (psi4[sls] - psi4[sl])
        f2[sl] += c2[sl] * (psi3[sls] - psi3[sl])
        psi1[sl] += pcc[slc] * f1[sl]
        psi2[sl] += pcc[slc] * f2[sl]
        psi3[sls] = pca[slc] * psi3[sls] + pcb[slc] * f3[sls]
        psi4[sls] = pca[slc] * psi4[sls] + pcb[slc] * f4[sls]


    def update_e(self):
        npml = self.npml
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        cex, cey, cez = self.mainf.ces
        psi_eyx, psi_ezx, psi_hyx, psi_hzx = self.psi_xs
        psi_ezy, psi_exy, psi_hzy, psi_hxy = self.psi_ys 
        psi_exz, psi_eyz, psi_hxz, psi_hyz = self.psi_zs
        pca_half, pcb_half, pcc_half = self.pcs_half
        pca_one, pcb_one, pcc_one = self.pcs_one
        directions = self.directions
        sln = slice(None, None)
        nax = np.newaxis

        if '+' in directions[0]:
            sl = (slice(-npml-1, -1), sln, sln)
            sls = (slice(-npml, None), sln, sln)
            slc = (sln, nax, nax)
            self.update(sl, sls, slc, ey, ez, hy, hz, psi_eyx, psi_ezx, psi_hyx, psi_hzx, pca_one, pcb_one, pcc_half, cey, cez)
            '''
            ey[sl] -= cey[sl] * (psi_hzx[sls] - psi_hzx[sl])
            ez[sl] += cez[sl] * (psi_hyx[sls] - psi_hyx[sl])
            psi_eyx[sl] += pcc_half[slc] * ey[sl]
            psi_ezx[sl] += pcc_half[slc] * ez[sl]
            psi_hyx[sls] = pca_one[slc] * psi_hyx[sls] + pcb_one[slc] * hy[sls]
            psi_hzx[sls] = pca_one[slc] * psi_hzx[sls] + pcb_one[slc] * hz[sls]
            '''

        if '-' in directions[0]:
            sl = (slice(None, npml), sln, sln)
            sls = (slice(1, npml+1), sln, sln)
            slc = (slice(None, None, -1), nax, nax)
            self.update(sl, sls, slc, ey, ez, hy, hz, psi_eyx, psi_ezx, psi_hyx, psi_hzx, pca_half, pcb_half, pcc_one, cey, cez)

        '''
        if '+' in directions[1]:
            sl = (sln, slice(-npml-1, -1), sln)
            sls = (sln, slice(-npml, None), sln)
            slc = (sln, nax)
            self.update(sl, sls, slc, pcs, ez, ex, hz, hx, psi_ezy, psi_exy, psi_hzy, psi_hxy, cez, cex)

        if '-' in directions[1]:
            sl = (sln, slice(None, npml), sln)
            sls = (sln, slice(1, npml+1), sln)
            slc = (slice(None, None, -1), nax)
            self.update(sl, sls, slc, pcs, ez, ex, hz, hx, psi_ezy, psi_exy, psi_hzy, psi_hxy, cez, cex)

        if '+' in directions[2]:
            sl = (sln, sln, slice(-npml-1, -1))
            sls = (sln, sln, slice(-npml, None))
            slc = sln
            self.update(sl, sls, slc, pcs, ex, ey, hx, hy, psi_exz, psi_eyz, psi_hxz, psi_hyz, cex, cey)

        if '-' in directions[2]:
            sl = (sln, sln, slice(None, npml))
            sls = (sln, sln, slice(1, npml+1))
            slc = slice(None, None, -1)
            self.update(sl, sls, slc, pcs, ex, ey, hx, hy, psi_exz, psi_eyz, psi_hxz, psi_hyz, cex, cey)
        '''


    def update_h(self):
        npml = self.npml
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        chx, chy, chz = self.mainf.chs
        psi_eyx, psi_ezx, psi_hyx, psi_hzx = self.psi_xs
        psi_ezy, psi_exy, psi_hzy, psi_hxy = self.psi_ys 
        psi_exz, psi_eyz, psi_hxz, psi_hyz = self.psi_zs
        pca_half, pcb_half, pcc_half = self.pcs_half
        pca_one, pcb_one, pcc_one = self.pcs_one
        directions = self.directions
        sln = slice(None, None)
        nax = np.newaxis

        if '+' in directions[0]:
            sl = (slice(-npml, None), sln, sln)
            sls = (slice(-npml-1, -1), sln, sln)
            slc = (sln, nax, nax)
            self.update(sl, sls, slc, hy, hz, ey, ez, psi_hyx, psi_hzx, psi_eyx, psi_ezx, pca_one, pcb_one, pcc_half, chy, chz)
            '''
            hy[sl] -= chy[sl] * (psi_ezx[sls] - psi_ezx[sl])
            hz[sl] += chz[sl] * (psi_eyx[sls] - psi_eyx[sl])
            psi_hyx[sl] += pcc_one[slc] * hy[sl]
            psi_hzx[sl] += pcc_one[slc] * hz[sl]
            psi_eyx[sls] = pca_half[slc] * psi_eyx[sls] + pcb_half[slc] * ey[sls]
            psi_ezx[sls] = pca_half[slc] * psi_ezx[sls] + pcb_half[slc] * ez[sls]
            '''

        if '-' in directions[0]:
            sl = (slice(1, npml+1), sln, sln)
            sls = (slice(None, npml), sln, sln)
            slc = (slice(None, None, -1), nax, nax)
            self.update(sl, sls, slc, hy, hz, ey, ez, psi_hyx, psi_hzx, psi_eyx, psi_ezx, pca_half, pcb_half, pcc_one, chy, chz)

        '''
        if '+' in directions[1]:
            sl = (sln, slice(-npml, None), sln)
            sls = (sln, slice(-npml-1, -1), sln)
            slc = (sln, nax)
            self.update(sl, sls, slc, pcs, hx, hz, ex, ez, psi_hxy, psi_hzy, psi_exy, psi_ezy, chx, chz)

        if '-' in directions[1]:
            sl = (sln, slice(1, npml+1), sln)
            sls = (sln, slice(None, npml), sln)
            slc = (slice(None, None, -1), nax)
            self.update(sl, sls, slc, pcs, hx, hz, ex, ez, psi_hxy, psi_hzy, psi_exy, psi_ezy, chx, chz)

        if '+' in directions[2]:
            sl = (sln, sln, slice(-npml, None))
            sls = (sln, sln, slice(-npml-1, -1))
            slc = sln
            self.update(sl, sls, slc, pcs, hy, hx, ey, ex, psi_hyz, psi_hxz, psi_eyz, psi_exz, chy, chx)

        if '-' in directions[2]:
            sl = (sln, sln, slice(1, npml+1))
            sls = (sln, sln, slice(None, npml))
            slc = slice(None, None, -1)
            self.update(sl, sls, slc, pcs, hy, hx, ey, ex, psi_hyz, psi_hxz, psi_eyz, psi_exz, chy, chx)
        '''
