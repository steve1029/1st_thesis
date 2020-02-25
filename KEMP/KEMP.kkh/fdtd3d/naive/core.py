import numpy as np

from kemp.fdtd3d.util import common
from fields import Fields


class Core:
    def __init__(self, fields):
        common.check_type('fields', fields, Fields)

        # global variables
        self.mainf = fields
        
        if fields.ce_on:
            cex, cey, cez = fields.ces
            self.ces = cex[:,:-1,:-1], cey[:-1,:,:-1], cez[:-1,:-1,:]
        else:
            self.ces = fields.ces
            
        if fields.ch_on:
            chx, chy, chz = fields.chs
            self.chs = chx[:,1:,1:], cey[1:,:,1:], cez[1:,1:,:]
        else:
            self.chs = fields.chs

        if fields.rd_on:
            erdx, erdy, erdz = fields.erds
            hrdx, hrdy, hrdz = fields.hrds
            sln = slice(None, None)
            nax = np.newaxis
            self.erds = erdx[sln, nax, nax], erdy[sln, nax], erdz[sln]
            self.hrds = hrdx[sln, nax, nax], hrdy[sln, nax], hrdz[sln]
        else:
            self.erds = fields.erds
            self.hrds = fields.hrds

        # append to the update list
        self.priority_type = 'core'
        fields.append_instance(self)


    def update_e(self):
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        cex, cey, cez = self.ces
        erdx, erdy, erdz = self.erds

        ex[:,:-1,:-1] += cex * (erdy * (hz[:,1:,:-1] - hz[:,:-1,:-1]) \
                                - erdz * (hy[:,:-1,1:] - hy[:,:-1,:-1]))
        ey[:-1,:,:-1] += cey * (erdz * (hx[:-1,:,1:] - hx[:-1,:,:-1]) \
                                - erdx * (hz[1:,:,:-1] - hz[:-1,:,:-1]))
        ez[:-1,:-1,:] += cez * (erdx * (hy[1:,:-1,:] - hy[:-1,:-1,:]) \
                                - erdy * (hx[:-1,1:,:] - hx[:-1,:-1,:]))


    def update_h(self):
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        chx, chy, chz = self.chs
        hrdx, hrdy, hrdz = self.hrds

        hx[:,1:,1:] -= chx * (hrdy * (ez[:,1:,1:] - ez[:,:-1,1:]) \
                              - hrdz * (ey[:,1:,1:] - ey[:,1:,:-1]))
        hy[1:,:,1:] -= chy * (hrdz * (ex[1:,:,1:] - ex[1:,:,:-1]) \
                              - hrdx * (ez[1:,:,1:] - ez[:-1,:,1:]))
        hz[1:,1:,:] -= chz * (hrdx * (ey[1:,1:,:] - ey[:-1,1:,:]) \
                              - hrdy * (ex[1:,1:,:] - ex[1:,:-1,:]))