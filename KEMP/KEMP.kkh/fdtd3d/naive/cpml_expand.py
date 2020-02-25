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
        pcb_half, pca_half = self.pcs_half
        pcb_one, pca_one = self.pcs_one
        directions = self.directions
        sln = slice(None, None)
        nax = np.newaxis

        if '+' in directions[0]:
            sl = (slice(-npml, None), sln, sln)
            sls = (slice(-npml-1, -1), sln, sln)
            slc = (sln, nax, nax)
            #self.update(sl, sls, slc, ey, ez, hy, hz, psi_eyx, psi_ezx, psi_hyx, psi_hzx, pca_one, pcb_one, pcc_half, cey, cez)
            
            psi_eyx[sl] = pcb_half[slc] * psi_eyx[sl] + pca_half[slc] * (hz[sl] - hz[sls])
            psi_ezx[sl] = pcb_half[slc] * psi_ezx[sl] + pca_half[slc] * (hy[sl] - hy[sls])
            ey[sls] -= cey[sls] * psi_eyx[sl]
            ez[sls] += cez[sls] * psi_ezx[sl]

        if '-' in directions[0]:
            sl = (slice(None, npml), sln, sln)
            sls = (slice(1, npml+1), sln, sln)
            slc = (slice(None, None, -1), nax, nax)
            #self.update(sl, sls, slc, ey, ez, hy, hz, psi_eyx, psi_ezx, psi_hyx, psi_hzx, pca_half, pcb_half, pcc_one, cey, cez)

            psi_eyx[sl] = pcb_one[slc] * psi_eyx[sl] + pca_one[slc] * (hz[sls] - hz[sl])
            psi_ezx[sl] = pcb_one[slc] * psi_ezx[sl] + pca_one[slc] * (hy[sls] - hy[sl])
            ey[sl] -= cey[sl] * psi_eyx[sl]
            ez[sl] += cez[sl] * psi_ezx[sl]
            
        if '+' in directions[1]:
            sl = (sln, slice(-npml, None), sln)
            sls = (sln, slice(-npml-1, -1), sln)
            slc = (sln, nax)

            psi_ezy[sl] = pcb_half[slc] * psi_ezy[sl] + pca_half[slc] * (hx[sl] - hx[sls])
            psi_exy[sl] = pcb_half[slc] * psi_exy[sl] + pca_half[slc] * (hz[sl] - hz[sls])
            ez[sls] -= cez[sls] * psi_ezy[sl]
            ex[sls] += cex[sls] * psi_exy[sl]
            
        if '-' in directions[1]:
            sl = (sln, slice(None, npml), sln)
            sls = (sln, slice(1, npml+1), sln)
            slc = (slice(None, None, -1), nax)

            psi_ezy[sl] = pcb_one[slc] * psi_ezy[sl] + pca_one[slc] * (hx[sls] - hx[sl])
            psi_exy[sl] = pcb_one[slc] * psi_exy[sl] + pca_one[slc] * (hz[sls] - hz[sl])
            ez[sl] -= cez[sl] * psi_ezy[sl]
            ex[sl] += cex[sl] * psi_exy[sl]
            
        if '+' in directions[2]:
            sl = (sln, sln, slice(-npml, None))
            sls = (sln, sln, slice(-npml-1, -1))
            slc = sln

            psi_exz[sl] = pcb_half[slc] * psi_exz[sl] + pca_half[slc] * (hy[sl] - hy[sls])
            psi_eyz[sl] = pcb_half[slc] * psi_eyz[sl] + pca_half[slc] * (hx[sl] - hx[sls])
            ex[sls] -= cex[sls] * psi_exz[sl]
            ey[sls] += cey[sls] * psi_eyz[sl]
            
        if '-' in directions[2]:
            sl = (sln, sln, slice(None, npml))
            sls = (sln, sln, slice(1, npml+1))
            slc = slice(None, None, -1)

            psi_exz[sl] = pcb_one[slc] * psi_exz[sl] + pca_one[slc] * (hy[sls] - hy[sl])
            psi_eyz[sl] = pcb_one[slc] * psi_eyz[sl] + pca_one[slc] * (hx[sls] - hx[sl])
            ex[sl] -= cex[sl] * psi_exz[sl]
            ey[sl] += cey[sl] * psi_eyz[sl]
            

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
            #self.update(sl, sls, slc, hy, hz, ey, ez, psi_hyx, psi_hzx, psi_eyx, psi_ezx, pca_one, pcb_one, pcc_half, chy, chz)
            
            psi_hyx[sl] = pcb_one[slc] * psi_hyx[sl] + pca_one[slc] * (ez[sl] - ez[sls])
            psi_hzx[sl] = pcb_one[slc] * psi_hzx[sl] + pca_one[slc] * (ey[sl] - ey[sls])
            hy[sl] += chy[sl] * psi_hyx[sl]
            hz[sl] -= chz[sl] * psi_hzx[sl]

        if '-' in directions[0]:
            sl = (slice(None, npml), sln, sln)
            sls = (slice(1, npml+1), sln, sln)
            slc = (slice(None, None, -1), nax, nax)
            #self.update(sl, sls, slc, hy, hz, ey, ez, psi_hyx, psi_hzx, psi_eyx, psi_ezx, pca_half, pcb_half, pcc_one, chy, chz)

            psi_hyx[sl] = pcb_half[slc] * psi_hyx[sl] + pca_half[slc] * (ez[sls] - ez[sl])
            psi_hzx[sl] = pcb_half[slc] * psi_hzx[sl] + pca_half[slc] * (ey[sls] - ey[sl])
            hy[sls] += chy[sls] * psi_hyx[sl]
            hz[sls] -= chz[sls] * psi_hzx[sl]
            
        if '+' in directions[1]:
            sl = (sln, slice(-npml, None), sln)
            sls = (sln, slice(-npml-1, -1), sln)
            slc = (sln, nax)

            psi_hzy[sl] = pcb_one[slc] * psi_hzy[sl] + pca_one[slc] * (ex[sl] - ex[sls])
            psi_hxy[sl] = pcb_one[slc] * psi_hxy[sl] + pca_one[slc] * (ez[sl] - ez[sls])
            hz[sl] += chz[sl] * psi_hzy[sl]
            hx[sl] -= chx[sl] * psi_hxy[sl]
            
        if '-' in directions[1]:
            sl = (sln, slice(None, npml), sln)
            sls = (sln, slice(1, npml+1), sln)
            slc = (slice(None, None, -1), nax)

            psi_hzy[sl] = pcb_half[slc] * psi_hzy[sl] + pca_half[slc] * (ex[sls] - ex[sl])
            psi_hxy[sl] = pcb_half[slc] * psi_hxy[sl] + pca_half[slc] * (ez[sls] - ez[sl])
            hz[sls] += chz[sls] * psi_hzy[sl]
            hx[sls] -= chx[sls] * psi_hxy[sl]
            
        if '+' in directions[2]:
            sl = (sln, sln, slice(-npml, None))
            sls = (sln, sln, slice(-npml-1, -1))
            slc = sln

            psi_hxz[sl] = pcb_one[slc] * psi_hxz[sl] + pca_one[slc] * (ey[sl] - ey[sls])
            psi_hyz[sl] = pcb_one[slc] * psi_hyz[sl] + pca_one[slc] * (ex[sl] - ex[sls])
            hx[sl] += chx[sl] * psi_hxz[sl]
            hy[sl] -= chy[sl] * psi_hyz[sl]
            
        if '-' in directions[2]:
            sl = (sln, sln, slice(None, npml))
            sls = (sln, sln, slice(1, npml+1))
            slc = slice(None, None, -1)
            
            psi_hxz[sl] = pcb_half[slc] * psi_hxz[sl] + pca_half[slc] * (ey[sls] - ey[sl])
            psi_hyz[sl] = pcb_half[slc] * psi_hyz[sl] + pca_half[slc] * (ex[sls] - ex[sl])
            hx[sls] += chx[sls] * psi_hxz[sl]
            hy[sls] -= chy[sls] * psi_hyz[sl]