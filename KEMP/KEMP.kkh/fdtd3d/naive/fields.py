import numpy as np

from kemp.fdtd3d.util import common


class Fields(object):
    def __init__(self, nx, ny, nz, precision_float='single'):
        common.check_type('nx', nx, int)
        common.check_type('ny', ny, int)
        common.check_type('nz', nz, int)
        common.check_value('precision_float', precision_float, ('single', 'double'))

        # local variables
        dtype = {'single':np.float32, 'double':np.float64}[precision_float]
        ns = [nx, ny, nz]

        # allocations
        ehs = [np.zeros(ns, dtype) for i in range(6)]

        # global variables
        self.dx = 1.
        self.dt = 0.5
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ns = ns

        self.precision_float = precision_float
        self.dtype = dtype

        self.ehs = ehs
        self.ex, self.ey, self.ez = ehs[:3]
        self.hx, self.hy, self.hz = ehs[3:]
        
        self.ce_on, self.ch_on, self.rd_on = False, False, False
        self.ces = self.cex, self.cey, self.cez = 0.5, 0.5, 0.5
        self.chs = self.chx, self.chy, self.chz = 0.5, 0.5, 0.5
        self.erds = self.erdx, self.erdy, self.erdz = 1., 1., 1.
        self.hrds = self.hrdx, self.hrdy, self.hrdz = 1., 1., 1.

        # update list
        self.instance_list = []
        self.append_instance = lambda instance: \
                common.append_instance(self.instance_list, instance)


    def get(self, str_f):
        value_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
        common.check_value('str_f', str_f, value_list)

        return self.__dict__[str_f]

    
    def get_ces(self):
        if not self.ce_on:
            self.ces = [np.ones(self.ns, self.dtype)*0.5 for i in range(3)]
            self.cex, self.cey, self.cez = self.ces
            self.ce_on = True
        
        return self.ces
        

    def get_chs(self):
        if not self.ch_on:
            self.chs = [np.ones(self.ns, self.dtype)*0.5 for i in range(3)]
            self.chx, self.chy, self.chz = self.chs
            self.ch_on = True
        
        return self.chs
        

    def get_rds(self):      # reciprocal ds
        if not self.rd_on:
            self.erds = [np.ones(n-1, self.dtype) for n in self.ns]
            self.hrds = [np.ones(n-1, self.dtype) for n in self.ns]
            self.erdx, self.erdy, self.erdz = self.erds
            self.hrdx, self.hrdy, self.hrdz = self.hrds
            self.rd_on = True
        
        return self.erds, self.hrds
        

    def set_ehs(self, ex, ey, ez, hx, hy, hz):  # for unittest
        eh_list = [ex, ey, ez, hx, hy, hz]
        for eh in eh_list:
            common.check_type('eh', eh, np.ndarray)

        for eh, f in zip(self.ehs, eh_list):
            eh[:] = f[:]


    def set_ces(self, cex, cey, cez):
        for ce in [cex, cey, cez]:
            common.check_type('ce', ce, np.ndarray)

        if self.ce_on:
            self.cex[:] = cex[:]
            self.cey[:] = cey[:]
            self.cez[:] = cez[:]
        else:
            raise AttributeError("The Fields instance has no ce arrays.")
        


    def set_chs(self, chx, chy, chz):
        for ch in [chx, chy, chz]:
            common.check_type('ch', ch, np.ndarray)

        if self.ch_on:
            self.chx[:] = chx[:]
            self.chy[:] = chy[:]
            self.chz[:] = chz[:]
        else:
            raise AttributeError("The Fields instance has no ch arrays.")


    def update_e(self):
        for instance in self.instance_list:
            instance.update_e()


    def update_h(self):
        for instance in self.instance_list:
            instance.update_h()