import numpy as np
import scipy as sc
from util  import *
from units import *
from scipy.constants import c as c0

class RFT:
    def __init__(self, fdtd, field, region, freq_domain):
        self.tstep = 0
        self.update_e_flag = False
        self.update_h_flag = False
        self.fieldname = field
        self.fdtd = fdtd
        self.fdtd.rfts.append(self)
        self.set_field(self.fieldname)
        self.freq_domain  = to_NU(fdtd, 'frequency', freq_domain.astype(comp_to_real(fdtd.dtype)))
        self.wfreq_domain = self.freq_domain*2.*np.pi
        self.nw = self.wfreq_domain.size
        self.rft_complex_parts = ['real', 'imag']
        self.rft_data = {'real':FakeBuffer(), 'imag':FakeBuffer()}
        if region is not None:
            self.set_region(region)
        else:
            self.region = None
        if   fdtd.extension == 'single':
            self.init_single()
        elif fdtd.extension == 'multi':
            self.init_multi()
        elif fdtd.extension == 'mpi':
            self.init_mpi()

    def init_single(self):
        self.setup()

    def init_multi(self):
        fdtd = self.fdtd
        regions = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
        self.sub_rfts = []
        for i, sfdtd in enumerate(self.fdtd.fdtd_group):
            rft = sfdtd.apply_RFT(self.fieldname, regions[i])
            self.sub_rfts.append(rft)

    def init_mpi(self):
        fdtd = self.fdtd
        regions = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
        self.sub_rfts = [fdtd.sub_fdtd.running_fourier_transform(self.fieldname, regions[fdtd.rank])]
        '''
        self.buf_rfts = []
        if fdtd.comm_mode == 'overlap':
            if regions[fdtd.rank] is not None:
                x0 = regions[fdtd.rank][0][0]
                x1 = regions[fdtd.rank][1][0]
                y0 = regions[fdtd.rank][0][1]
                y1 = regions[fdtd.rank][1][1]
                z0 = regions[fdtd.rank][0][2]
                z1 = regions[fdtd.rank][1][2]
                self.buf_slices = []
                if   x0 == 0:
                    if   x1 >= 1:
                        buf_fdtd_region0 = ((0, y0, z0), (1, y1, z1))
                        self.buf_slices.append(slice(0, 2, None))
                    else:
                        buf_fdtd_region0 = ((0, y0, z0), (0, y1, z1))
                        self.buf_slices.append(slice(0, 1, None))
                elif x0 == 1:
                    buf_fdtd_region0 = ((1, y0, z0), (1, y1, z1))
                    self.buf_slices.append(slice(1, 2, None))
                else:
                    buf_fdtd_region0 = None
                    self.buf_slices.append(None)

                if   x1 == fdtd.nx_group[fdtd.rank] - 1:
                    if   x0 < fdtd.nx_group[fdtd.rank] - 1:
                        buf_fdtd_region1 = ((0, y0, z0), (1, y1, z1))
                        self.buf_slices.append(slice(-2, None, None))
                    else:
                        buf_fdtd_region1 = ((1, y0, z0), (1, y1, z1))
                        self.buf_slices.append(slice(-1, None, None))
                elif x1 == fdtd.nx_group[fdtd.rank] - 2:
                    buf_fdtd_region1 = ((0, y0, z0), (0, y1, z1))
                    self.buf_slices.append(slice(-2, -1, None))
                else:
                    buf_fdtd_region1 = None
                    self.buf_slices.append(None)

                self.buf_rfts.append(fdtd.buf_fdtd_group[ 0].apply_RFT(self.fieldname, buf_fdtd_region0))
                self.buf_rfts.append(fdtd.buf_fdtd_group[-1].apply_RFT(self.fieldname, buf_fdtd_region1))
            else:
                self.buf_slices = [None, None]
                self.buf_rfts.append(fdtd.buf_fdtd_group[ 0].apply_RFT(self.fieldname, None))
                self.buf_rfts.append(fdtd.buf_fdtd_group[-1].apply_RFT(self.fieldname, None))
        '''

    def set_field(self, field):
        fdtd = self.fdtd
        check_type('field', field, str)
        if   self.fdtd.mode == '2DTE':
            check_value('fields', field, ('Ex', 'Ey',       'ex', 'ey',       \
                                                      'Hz',             'hz'))
        elif self.fdtd.mode == '2DTM':
            check_value('fields', field, (            'Ez',             'ez', \
                                          'Hx', 'Hy',       'hx', 'hy'      ))
        elif self.fdtd.mode == '3D':
            check_value('fields', field, ('Ex', 'Ey', 'Ez', 'ex', 'ey', 'ez', \
                                          'Hx', 'Hy', 'Hz', 'hx', 'hy', 'hz'))
        if   'E' in field or 'e' in field:
            self.update_e_flag = True
            if   field in ['Ex', 'ex']:
                self.field = fdtd.ex
            elif field in ['Ey', 'ey']:
                self.field = fdtd.ey
            elif field in ['Ez', 'ez']:
                self.field = fdtd.ez
        elif 'H' in field or 'h' in field:
            self.update_h_flag = True
            if   field in ['Hx', 'hx']:
                self.field = fdtd.hx
            elif field in ['Hy', 'hy']:
                self.field = fdtd.hy
            elif field in ['Hz', 'hz']:
                self.field = fdtd.hz

    def set_region(self, region):
        fdtd = self.fdtd
        check_type('region', region, tuple, tuple)
        check_type('region[0]', region[0], tuple, tuple(  int_types))
        check_type('region[1]', region[1], tuple, tuple(  int_types))

        if   '2D' in fdtd.mode:
            self.x_strt = region[0][0]
            self.y_strt = region[0][1]
            self.x_stop = region[1][0]
            self.y_stop = region[1][1]

            if self.x_strt < 0:
                self.x_strt += fdtd.nx
            if self.x_stop < 0:
                self.x_stop += fdtd.nx
            if self.y_strt < 0:
                self.y_strt += fdtd.ny
            if self.y_stop < 0:
                self.y_stop += fdtd.ny

            self.z_strt = 0
            self.z_stop = 0
            self.nx = self.x_stop - self.x_strt + 1
            self.ny = self.y_stop - self.y_strt + 1
            self.nz =                             1
            self.gs = self.nx*self.ny*self.nw
            shape = (self.nx, self.ny)
            self.shape = []
            self.ndim = 2
            for i, n in enumerate(shape):
                if n == 1:
                    self.ndim -= 1
                else:
                    self.shape.append(n)
            self.shape = tuple(self.shape)
            self.region = ((self.x_strt, self.y_strt), \
                           (self.x_stop, self.y_stop))
            self.slcs = [slice(self.x_strt, self.x_stop+1, None), \
                         slice(self.y_strt, self.y_stop+1, None)]
            if self.nx == 1:
                self.slcs[0] = self.x_strt
            if self.ny == 1:
                self.slcs[1] = self.y_strt
            self.rft_shape = self.shape + (self.nw,)

        elif '3D' in fdtd.mode:
            self.x_strt = region[0][0]
            self.y_strt = region[0][1]
            self.z_strt = region[0][2]
            self.x_stop = region[1][0]
            self.y_stop = region[1][1]
            self.z_stop = region[1][2]

            if self.x_strt < 0:
                self.x_strt += fdtd.nx
            if self.x_stop < 0:
                self.x_stop += fdtd.nx
            if self.y_strt < 0:
                self.y_strt += fdtd.ny
            if self.y_stop < 0:
                self.y_stop += fdtd.ny
            if self.z_strt < 0:
                self.z_strt += fdtd.nz
            if self.z_stop < 0:
                self.z_stop += fdtd.nz

            self.nx = self.x_stop - self.x_strt + 1
            self.ny = self.y_stop - self.y_strt + 1
            self.nz = self.z_stop - self.z_strt + 1
            self.gs = self.nx*self.ny*self.nz*self.nw
            shape = (self.nx, self.ny, self.nz)
            self.shape = []
            self.ndim = 3
            self.divided = True
            for i, n in enumerate(shape):
                if n == 1:
                    if i == 0:
                        self.divided = False
                    self.ndim -= 1
                else:
                    self.shape.append(n)
            self.shape = tuple(self.shape)
            self.region = ((self.x_strt, self.y_strt, self.z_strt),\
                           (self.x_stop, self.y_stop, self.z_stop))
            self.slcs = [slice(self.x_strt, self.x_stop+1, None), \
                         slice(self.y_strt, self.y_stop+1, None), \
                         slice(self.z_strt, self.z_stop+1, None)]
            if self.nx == 1:
                self.slcs[0] = self.x_strt
            if self.ny == 1:
                self.slcs[1] = self.y_strt
            if self.nz == 1:
                self.slcs[2] = self.z_strt
            self.rft_shape = self.shape + (self.nw,)

    def setup(self):
        fdtd = self.fdtd
        if 'opencl' in fdtd.engine.name:
            mf = fdtd.engine.cl.mem_flags
            self.wfreq_data = fdtd.engine.cl.Buffer(fdtd.engine.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                                                    hostbuf=self.wfreq_domain)
        elif 'cuda' in fdtd.engine.name:
            self.wfreq_data = fdtd.engine.to_device(self.wfreq_domain)
        elif  'cpu' in fdtd.engine.name:
            self.wfreq_data = self.wfreq_domain
        for part in self.rft_complex_parts:
            if 'opencl' in fdtd.engine.name:
                mf = fdtd.engine.cl.mem_flags
                self.rft_data[part] = fdtd.engine.cl.Buffer(fdtd.engine.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, \
                                                            hostbuf=np.zeros(self.rft_shape, dtype=comp_to_real(fdtd.dtype)))
            elif 'cuda' in fdtd.engine.name:
                self.rft_data[part] = fdtd.engine.to_device(np.zeros(self.rft_shape, dtype=comp_to_real(fdtd.dtype)))
            elif  'cpu' in fdtd.engine.name:
                self.rft_data[part] = np.zeros(self.rft_shape, comp_to_real(fdtd.dtype))

        self.kernel_args = {'rft_e':{'real':None, 'imag':None}, 'rft_h':{'real':None, 'imag':None}}
        if self.region is not None:
            if 'opencl' in fdtd.engine.name:
                self.gs = cl_global_size(self.nx*self.ny*self.nz*self.nw, fdtd.engine.ls)
            if ('opencl' in fdtd.engine.name) or ('cuda' in fdtd.engine.name):
                rdtype = comp_to_real(fdtd.dtype)
                if   '2D' in fdtd.mode:
                    for part in fdtd.complex_parts:
                        if part == 'real':
                            pha_re = +0.0*np.pi
                            pha_im = +0.5*np.pi
                        if part == 'imag':
                            pha_re = -0.5*np.pi
                            pha_im = +0.0*np.pi
                        if   self.update_e_flag:
                            self.kernel_args['rft_e'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(      1), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               np.int32(self.nw), np.int32(self.tstep), \
                                                               rdtype(fdtd.dt), rdtype(pha_re), rdtype(pha_im), \
                                                               self.wfreq_data, self.field.__dict__[part].data, \
                                                               self.rft_data['real'], self.rft_data['imag']]
                        elif self.update_h_flag:
                            self.kernel_args['rft_h'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(      1), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               np.int32(self.nw), np.int32(self.tstep), \
                                                               rdtype(fdtd.dt), rdtype(pha_re), rdtype(pha_im), \
                                                               self.wfreq_data, self.field.__dict__[part].data, \
                                                               self.rft_data['real'], self.rft_data['imag']]
                elif '3D' in fdtd.mode:
                    for part in fdtd.complex_parts:
                        if part == 'real':
                            pha_re = +0.0*np.pi
                            pha_im = +0.5*np.pi
                        if part == 'imag':
                            pha_re = -0.5*np.pi
                            pha_im = +0.0*np.pi
                        if   self.update_e_flag:
                            self.kernel_args['rft_e'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               np.int32(self.nw), np.int32(self.tstep), \
                                                               rdtype(fdtd.dt), rdtype(pha_re), rdtype(pha_im), \
                                                               self.wfreq_data, self.field.__dict__[part].data, \
                                                               self.rft_data['real'], self.rft_data['imag']]
                        elif self.update_h_flag:
                            self.kernel_args['rft_h'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               np.int32(self.nw), np.int32(self.tstep), \
                                                               rdtype(fdtd.dt), rdtype(pha_re), rdtype(pha_im), \
                                                               self.wfreq_data, self.field.__dict__[part].data, \
                                                               self.rft_data['real'], self.rft_data['imag']]

    def export(self):
        fdtd = self.fdtd
        if   fdtd.extension in ['single', 'multi']:
            rft_out_real = np.zeros(self.rft_shape, dtype=comp_to_real(fdtd.dtype))
            rft_out_imag = np.zeros(self.rft_shape, dtype=comp_to_real(fdtd.dtype))
        elif fdtd.extension == 'mpi':
            rft_out = np.zeros(self.rft_shape, dtype=real_to_comp(fdtd.dtype))
            pass # Not implemented
        evts = []
        if 'opencl' in fdtd.engine.name:
            evt = fdtd.engine.cl.enqueue_read_buffer(fdtd.engine.queue, self.rft_data['real'], rft_out_real)
            evts.append(evt)
            evt = fdtd.engine.cl.enqueue_read_buffer(fdtd.engine.queue, self.rft_data['imag'], rft_out_imag)
            evts.append(evt)
        elif 'cuda' in fdtd.engine.name:
            evt = fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtoh, [rft_out_real, self.rft_data['real']])
            evts.append(evt)
            evt = fdtd.engine.enqueue(fdtd.engine.drv.memcpy_dtoh, [rft_out_imag, self.rft_data['imag']])
            evts.append(evt)
        elif  'cpu' in fdtd.engine.name:
            pass
        wait_for_events(fdtd, evts)
        rft_out = (1./(2*np.pi))*to_SI(fdtd, 'time', fdtd.dt)*(rft_out_real + 1.j*rft_out_imag)
        return rft_out

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if self.update_e_flag:
            self.tstep += 1
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    self.kernel_args['rft_e'][part][13] = np.int32(self.tstep)
                    evt = fdtd.engine.kernels['rft'](*(self.kernel_args['rft_e'][part]))
                elif 'cuda' in fdtd.engine.name:
                    self.kernel_args['rft_e'][part][13] = np.int32(self.tstep)
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['rft'], (self.kernel_args['rft_e'][part]))
                elif  'cpu' in fdtd.engine.name:
                    evt = FakeEvent()
                evts.append(evt)
            if wait:
                wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if self.update_h_flag:
            self.tstep += 1
            for part in fdtd.complex_parts:
                if 'opencl' in fdtd.engine.name:
                    self.kernel_args['rft_h'][part][13] = np.int32(self.tstep)
                    evt = fdtd.engine.kernels['rft'](*(self.kernel_args['rft_h'][part]))
                elif 'cuda' in fdtd.engine.name:
                    self.kernel_args['rft_h'][part][13] = np.int32(self.tstep)
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['rft'], (self.kernel_args['rft_h'][part]))
                elif  'cpu' in fdtd.engine.name:
                    evt = FakeEvent()
                evts.append(evt)
            if wait:
                wait_for_events(fdtd, evts)
        return evts
