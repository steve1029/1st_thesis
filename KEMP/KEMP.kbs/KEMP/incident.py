import numpy as np
import scipy as sc
import exception
import types
import sys
from mainfdtd import Basic_FDTD, Basic_FDTD_2d, Basic_FDTD_3d
from  ndarray import Fields
from    units import to_SI, to_NU
from     util import *

class DirectSource:
    def __init__(self, fdtd, field, region, func=None):
        self.tstep = 0
        self.update_e_flag = False
        self.update_h_flag = False
        self.fieldname = field
        self.value_buff = {'real':FakeBuffer(0.), 'imag':FakeBuffer(0.)}
        self.fdtd = fdtd
        self.fdtd.sources.append(self)
        self.set_field(self.fieldname)
        if region is not None:
            self.set_region(region)
        else:
            self.region = None
        if   fdtd.extension == 'single':
            self.init_single()
        elif fdtd.extension == 'multi':
            self.init_multi()
        elif fdtd.extension == 'mpi':
            if fdtd.worker:
                self.init_mpi()

    def init_single(self):
        self.setup()

    def init_multi(self):
        fdtd = self.fdtd
        regions = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
        self.sub_srcs = []
        for i, sfdtd in enumerate(self.fdtd.fdtd_group):
            src = sfdtd.apply_direct_source(self.fieldname, regions[i])
            self.sub_srcs.append(src)

    def init_mpi(self):
        fdtd = self.fdtd
        regions = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
        self.sub_srcs = [fdtd.sub_fdtd.apply_direct_source(self.fieldname, regions[fdtd.rank])]
        self.buf_srcs = []
        '''
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

                self.buf_srcs.append(fdtd.buf_fdtd_group[ 0].apply_direct_source(self.fieldname, buf_fdtd_region0))
                self.buf_srcs.append(fdtd.buf_fdtd_group[-1].apply_direct_source(self.fieldname, buf_fdtd_region1))
            else:
                self.buf_slices = [None, None]
                self.buf_srcs.append(fdtd.buf_fdtd_group[ 0].apply_direct_source(self.fieldname, None))
                self.buf_srcs.append(fdtd.buf_fdtd_group[-1].apply_direct_source(self.fieldname, None))
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
            self.gs = self.nx*self.ny
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
            self.slcs = [slice(self.x_strt, self.x_stop+1, None), slice(self.y_strt, self.y_stop+1, None)]
            if self.nx == 1:
                self.slcs[0] = self.x_strt
            if self.ny == 1:
                self.slcs[1] = self.y_strt

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
            self.gs = self.nx*self.ny*self.nz
            shape = (self.nx, self.ny, self.nz)
            self.shape3 = shape
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
            self.slcs = [slice(self.x_strt, self.x_stop+1, None), slice(self.y_strt, self.y_stop+1, None), slice(self.z_strt, self.z_stop+1, None)]
            if self.nx == 1:
                self.slcs[0] = self.x_strt
            if self.ny == 1:
                self.slcs[1] = self.y_strt
            if self.nz == 1:
                self.slcs[2] = self.z_strt

    def set_source(self, value):
        fdtd = self.fdtd
        value = fdtd.dtype(value)
        if fdtd.extension == 'single':
            self.value = value
            if self.value is not None:
                check_type('value', value, (int, np.int32, np.int64, \
                                            float, np.float32, np.float64, \
                                            complex, np.complex64, np.complex128, \
                                            np.ndarray))
                if isinstance(self.value, np.ndarray):
                    if np.shape(self.value) != self.shape:
                        raise ValueError, 'could not broadcast input array from shape %s, into shape %s' % (np.shape(self.value), self.shape)
                    comp_value = {'real':np.real(self.value), 'imag':np.imag(self.value)}
                    for part in fdtd.complex_parts:
                        if 'opencl' in fdtd.engine.name:
                            self.value_buff[part].release()
                        elif 'cuda' in fdtd.engine.name:
                            fdtd.engine.enqueue(self.value_buff[part].free, [])
                        if 'opencl' in fdtd.engine.name:
                            mf = fdtd.engine.cl.mem_flags
                            self.value_buff[part] = fdtd.engine.cl.Buffer(fdtd.engine.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                                                                          hostbuf=np.ascontiguousarray(comp_value[part]))
                        elif 'cuda' in fdtd.engine.name:
                            self.value_buff[part] = fdtd.engine.to_device(np.ascontiguousarray(comp_value[part]))
                        elif  'cpu' in fdtd.engine.name:
                            self.value_buff[part] = np.ascontiguousarray(comp_value[part])
        elif fdtd.extension == 'multi':
            for i, sfdtd in enumerate(fdtd.fdtd_group):
                if isinstance(value, np.ndarray):
                    self.value = divide_array(value, self.region, i, self.divided, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
                else:
                    self.value = value
                self.sub_srcs[i].set_source(self.value)
        elif fdtd.extension == 'mpi':
            if fdtd.worker:
                if isinstance(value, np.ndarray):
                    self.value = divide_array(value, self.region, fdtd.rank, self.divided, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
                else:
                    self.value = value
                self.sub_srcs[0].set_source(self.value)
                '''
                if fdtd.comm_mode == 'overlap':
                    for i in xrange(2):
                        if self.buf_slices[i] is not None:
                            if isinstance(self.value, np.ndarray):
                                if self.divided:
                                    slc = [self.buf_slices[i]]
                                else:
                                    slc = [slice(None, None, None)]
                                for n in xrange(self.value.ndim-1):
                                    slc.append(slice(None, None, None))
                                slc = tuple(slc)
                                self.buf_srcs[i].set_source(self.value.__getitem__(slc))
                            else:
                                self.buf_srcs[i].set_source(self.value)
                '''


    def setup(self):
        fdtd = self.fdtd
        self.kernel_args = {'src_e':{'real':None, 'imag':None}, 'src_h':{'real':None, 'imag':None}}
        if self.region is not None:
            if 'opencl' in fdtd.engine.name:
                self.gs = cl_global_size(self.nx*self.ny*self.nz, fdtd.engine.ls)
            if ('opencl' in fdtd.engine.name) or ('cuda' in fdtd.engine.name):
                if   '2D' in fdtd.mode:
                    for part in fdtd.complex_parts:
                        if   self.update_e_flag:
                            self.kernel_args['src_e'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(      1), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               self.field.__dict__[part].data]
                        elif self.update_h_flag:
                            self.kernel_args['src_h'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(      1), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               self.field.__dict__[part].data]
                elif '3D' in fdtd.mode:
                    for part in fdtd.complex_parts:
                        if   self.update_e_flag:
                            self.kernel_args['src_e'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               self.field.__dict__[part].data]
                        elif self.update_h_flag:
                            self.kernel_args['src_h'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               self.field.__dict__[part].data]

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if (self.region is not None) and (self.value is not None):
            for part in fdtd.complex_parts:
                if self.update_e_flag:
                    if isinstance(self.value, np.ndarray):
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['src_a'](*(self.kernel_args['src_e'][part] + [self.value_buff[part]]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['src_a'], (self.kernel_args['src_e'][part] + [self.value_buff[part]]))
                        elif  'cpu' in fdtd.engine.name:
                            self.field.__dict__[part].data[self.slcs] += self.value_buff[part]
                            evt = FakeEvent()
                    else:
                        if   part == 'real':
                            val = np.real(self.value).astype(comp_to_real(fdtd.dtype))
                        elif part == 'imag':
                            val = np.imag(self.value).astype(comp_to_real(fdtd.dtype))
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['src_v'](*(self.kernel_args['src_e'][part] + [val]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['src_v'], (self.kernel_args['src_e'][part] + [val]))
                        elif  'cpu' in fdtd.engine.name:
                            self.field.__dict__[part].data[self.slcs] += val
                            evt = FakeEvent()
                    evts.append(evt)
            if wait:
                wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if self.region is not None:
            for part in fdtd.complex_parts:
                if self.update_h_flag:
                    if isinstance(self.value, np.ndarray):
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['src_a'](*(self.kernel_args['src_h'][part] + [self.value_buff[part]]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['src_a'], (self.kernel_args['src_h'][part] + [self.value_buff[part]]))
                        elif  'cpu' in fdtd.engine.name:
                            self.field.__dict__[part][self.slcs] += self.value_buff[part]
                            evt = FakeEvent()
                    else:
                        if   part == 'real':
                            val = np.real(self.value).astype(comp_to_real(fdtd.dtype))
                        elif part == 'imag':
                            val = np.imag(self.value).astype(comp_to_real(fdtd.dtype))
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['src_v'](*(self.kernel_args['src_h'][part] + [val]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['src_v'], (self.kernel_args['src_h'][part] + [val]))
                        elif  'cpu' in fdtd.engine.name:
                            self.field.__dict__[part][self.slcs] += val
                            evt = FakeEvent()
                    evts.append(evt)
                if wait:
                    wait_for_events(fdtd, evts)
        self.tstep += 1
        return evts

class MonochromaticSource(DirectSource):
    def __init__(self, fdtd, field, region, freq):
        self.fdtd = fdtd
        self.tstep = 0
        self.switch = 1.
        self.freq = freq
        self.set_freq(self.freq)
        self.value_buffs = {'coeff':{'real':FakeBuffer(1.), 'imag':FakeBuffer(1.)}, \
                            'phase':{'real':FakeBuffer(0.), 'imag':FakeBuffer(0.)}}
        self.values = {'coeff':None, 'phase':None}
        DirectSource.__init__(self, fdtd, field, region)
        self.set_coeff(np.ones(self.shape, dtype=comp_to_real(fdtd.dtype)))
        self.set_phase(np.zeros(self.shape, dtype=comp_to_real(fdtd.dtype)))
        self.set_switch(comp_to_real(fdtd.dtype)(1.))

    def init_single(self):
        self.setup()

    def init_multi(self):
        fdtd = self.fdtd
        regions = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
        self.sub_srcs = []
        for i, sfdtd in enumerate(self.fdtd.fdtd_group):
            src = sfdtd.apply_monochromatic_source(self.fieldname, regions[i], self.freq)
            self.sub_srcs.append(src)

    def init_mpi(self):
        fdtd = self.fdtd
        regions = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
        self.sub_srcs = [fdtd.sub_fdtd.apply_monochromatic_source(self.fieldname, regions[fdtd.rank])]
        self.buf_srcs = []
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

                self.buf_srcs.append(fdtd.buf_fdtd_group[ 0].apply_monochromatic_source(self.fieldname, buf_fdtd_region0))
                self.buf_srcs.append(fdtd.buf_fdtd_group[-1].apply_monochromatic_source(self.fieldname, buf_fdtd_region1))
            else:
                self.buf_slices = [None, None]
                self.buf_srcs.append(fdtd.buf_fdtd_group[ 0].apply_monochromatic_source(self.fieldname, None))
                self.buf_srcs.append(fdtd.buf_fdtd_group[-1].apply_monochromatic_source(self.fieldname, None))

    def set_values(self, value, value_type):
        fdtd = self.fdtd
        if isinstance(value, np.ndarray):
            value = value.astype(comp_to_real(fdtd.dtype))
        else:
            value = comp_to_real(fdtd.dtype)(value)
        check_value('value_type', value_type, ('coeff', 'phase'))
        if fdtd.extension == 'single':
            self.value = value
            if self.value is not None:
                check_type('value', value, (int, np.int32, np.int64, \
                                            float, np.float32, np.float64, \
                                            complex, np.complex64, np.complex128, \
                                            np.ndarray))
                if isinstance(self.value, np.ndarray):
                    if np.shape(self.value) != self.shape:
                        raise ValueError, 'could not broadcast input array from shape %s, into shape %s' % (np.shape(self.value), self.shape)
                    comp_value = {'real':np.real(self.value), 'imag':np.imag(self.value)}
                    for part in fdtd.complex_parts:
                        if 'opencl' in fdtd.engine.name:
                            self.value_buffs[value_type][part].release()
                        elif 'cuda' in fdtd.engine.name:
                            fdtd.engine.enqueue(self.value_buffs[value_type][part].free, [])

                        if 'opencl' in fdtd.engine.name:
                            mf = fdtd.engine.cl.mem_flags
                            self.value_buffs[value_type][part] = fdtd.engine.cl.Buffer(fdtd.engine.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, \
                                                                                       hostbuf=np.ascontiguousarray(comp_value[part]))
                        elif 'cuda' in fdtd.engine.name:
                            self.value_buffs[value_type][part] = fdtd.engine.to_device(np.ascontiguousarray(comp_value[part]))
                        elif  'cpu' in fdtd.engine.name:
                            self.value_buffs[value_type][part] = np.ascontiguousarray(comp_value[part])
        elif fdtd.extension == 'multi':
            for i, sfdtd in enumerate(fdtd.fdtd_group):
                if isinstance(value, np.ndarray):
                    temp_value = divide_array(value, self.region, i, self.divided, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
#                    self.values[value_type] = divide_array(value, self.region, i, self.divided, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
                else:
                    temp_value = value
#                    self.values[value_type] = value
                self.sub_srcs[i].set_values(temp_value, value_type)
#                self.sub_srcs[i].set_values(self.values[value_type], value_type)
        elif fdtd.extension == 'mpi':
            if isinstance(value, np.ndarray):
                self.values[value_type] = divide_array(value, self.region, fdtd.rank, self.divided, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
            else:
                self.values[value_type] = value
            self.sub_srcs[0].set_values(self.values[value_type], value_type)
            if fdtd.comm_mode == 'overlap':
                for i in xrange(2):
                    if self.buf_slices[i] is not None:
                        if isinstance(self.values[value_type], np.ndarray):
                            if self.divided:
                                slc = [self.buf_slices[i]]
                            else:
                                slc = [slice(None, None, None)]
                            for n in xrange(self.values[value_type].ndim-1):
                                slc.append(slice(None, None, None))
                            slc = tuple(slc)
                            self.buf_srcs[i].set_values(self.values[value_type].__getitem__(slc))
                        else:
                            self.buf_srcs[i].set_values(self.values[value_type])

    def set_freq(self, freq):
        check_type('freq', freq, float_types)
        self.wfreq = to_NU(self.fdtd, 'angular frequency', 2.*np.pi*freq)
#        if self.fdtd.extension in ['multi', 'mpi']:
#            for sub_src in self.sub_srcs:
#                sub_src.set_freq(freq)

    def set_coeff(self, coeff):
        check_type('coeff', coeff, np.ndarray, float_types)
        self.set_values(coeff, 'coeff')

    def set_phase(self, phase):
        check_type('phase', phase, np.ndarray, float_types)
        self.set_values(phase, 'phase')

    def set_switch(self, switch):
        check_type('switch', switch, float_types)
        self.switch = switch

    def setup(self):
        fdtd = self.fdtd
        self.kernel_args = {'src_e':{'real':None, 'imag':None}, 'src_h':{'real':None, 'imag':None}}
        if self.region is not None:
            if 'opencl' in fdtd.engine.name:
                self.gs = cl_global_size(self.nx*self.ny*self.nz, fdtd.engine.ls)
            if ('opencl' in fdtd.engine.name) or ('cuda' in fdtd.engine.name):
                if   '2D' in fdtd.mode:
                    for part in fdtd.complex_parts:
                        if part == 'real': phase_imag = 0.
                        else             : phase_imag = 1.
                        if   self.update_e_flag:
                            self.kernel_args['src_e'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(      1), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               comp_to_real(fdtd.dtype)(self.wfreq*self.tstep*fdtd.dt), \
                                                               comp_to_real(fdtd.dtype)(-.5*np.pi*phase_imag), \
                                                               comp_to_real(fdtd.dtype)(1.), \
                                                               self.value_buffs['coeff'][part], self.value_buffs['phase'][part], \
                                                               self.field.__dict__[part].data]
                        elif self.update_h_flag:
                            self.kernel_args['src_h'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(      1), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               comp_to_real(fdtd.dtype)(self.wfreq*self.tstep*fdtd.dt), \
                                                               comp_to_real(fdtd.dtype)(-.5*np.pi*phase_imag), \
                                                               comp_to_real(fdtd.dtype)(1.), \
                                                               self.value_buffs['coeff'][part], self.value_buffs['phase'][part], \
                                                               self.field.__dict__[part].data]
                elif '3D' in fdtd.mode:
                    for part in fdtd.complex_parts:
                        if part == 'real': phase_imag = 0.
                        else             : phase_imag = 1.
                        if   self.update_e_flag:
                            self.kernel_args['src_e'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               comp_to_real(fdtd.dtype)(self.wfreq*self.tstep*fdtd.dt), \
                                                               comp_to_real(fdtd.dtype)(-.5*np.pi*phase_imag), \
                                                               comp_to_real(fdtd.dtype)(1.), \
                                                               self.value_buffs['coeff'][part], self.value_buffs['phase'][part], \
                                                               self.field.__dict__[part].data]
                        elif self.update_h_flag:
                            self.kernel_args['src_h'][part] = [fdtd.engine.queue, (self.gs,), (fdtd.engine.ls,), \
                                                               np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                               np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                               comp_to_real(fdtd.dtype)(self.wfreq*self.tstep*fdtd.dt), \
                                                               comp_to_real(fdtd.dtype)(-.5*np.pi*phase_imag), \
                                                               comp_to_real(fdtd.dtype)(1.), \
                                                               self.value_buffs['coeff'][part], self.value_buffs['phase'][part], \
                                                               self.field.__dict__[part].data]

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if (self.region is not None) and (self.value is not None):
            for part in fdtd.complex_parts:
                if self.update_e_flag:
                    if ('opencl' in fdtd.engine.name) or ('cuda' in fdtd.engine.name):
                        self.kernel_args['src_e'][part][12] = comp_to_real(fdtd.dtype)(self.wfreq*self.tstep*fdtd.dt)
                        self.kernel_args['src_e'][part][14] = comp_to_real(fdtd.dtype)(self.switch)
                        self.kernel_args['src_e'][part][15] = self.value_buffs['coeff'][part]
                        self.kernel_args['src_e'][part][16] = self.value_buffs['phase'][part]
                    if 'opencl' in fdtd.engine.name:
                        evt = fdtd.engine.kernels['src_m'](*(self.kernel_args['src_e'][part]))
                    elif 'cuda' in fdtd.engine.name:
                        evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['src_m'], (self.kernel_args['src_e'][part]))
                    elif  'cpu' in fdtd.engine.name:
                        pha = self.wfreq*self.tstep*fdtd.dt+self.value_buffs['phase'][part]
                        self.field.__dict__[part].data[self.slcs] += self.switch*self.value_buffs['coeff'][part]*np.sin(pha)
                        evt = FakeEvent()
                    evts.append(evt)
            if wait:
                wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if self.region is not None:
            for part in fdtd.complex_parts:
                if self.update_h_flag:
                    if ('opencl' in fdtd.engine.name) or ('cuda' in fdtd.engine.name):
                        self.kernel_args['src_h'][part][12] = comp_to_real(fdtd.dtype)(self.wfreq*self.tstep*fdtd.dt)
                        self.kernel_args['src_h'][part][14] = comp_to_real(fdtd.dtype)(self.switch)
                        self.kernel_args['src_h'][part][15] = self.value_buffs['coeff'][part]
                        self.kernel_args['src_h'][part][16] = self.value_buffs['phase'][part]
                    if 'opencl' in fdtd.engine.name:
                        evt = fdtd.engine.kernels['src_m'](*(self.kernel_args['src_h'][part]))
                    elif 'cuda' in fdtd.engine.name:
                        evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['src_m'], (self.kernel_args['src_h'][part]))
                    elif  'cpu' in fdtd.engine.name:
                        pha = self.wfreq*self.tstep*fdtd.dt+self.value_buffs['phase'][part]
                        self.field.__dict__[part].data[self.slcs] += self.switch*self.value_buffs['coeff'][part]*np.sin(pha)
                        evt = FakeEvent()
                    evts.append(evt)
                if wait:
                    wait_for_events(fdtd, evts)
        self.tstep += 1
        return evts

class TFSF_Boundary:
    def __init__(self, fdtd, TFSF_region, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, is_oblique=True, incfdtd=None):
        self.fdtd = fdtd
        self.set_region(TFSF_region)
        self.is_oblique = is_oblique
        self.init(self.region, boundary, incfdtd)
        if fdtd.extension == 'single':
            fdtd.cores['tfsf'] = self
            self.setup()

    def init(self, region, boundary, incfdtd):
        fdtd = self.fdtd
        fdtd.is_tfsf = True
        self.boundary = boundary
        if self.is_oblique:
            dtype = real_to_comp(fdtd.dtype)
        else:
            dtype = fdtd.dtype
        if   '2D' in fdtd.mode:
            self.tfsf_fdtd = Basic_FDTD_2d(mode=fdtd.mode, space_grid=fdtd.space_grid, dtype=dtype, \
                                           engine=fdtd.engines, device_id=fdtd.device_id)
        elif '3D' in fdtd.mode:
            if incfdtd is None:
                self.tfsf_fdtd = Basic_FDTD_3d(mode=fdtd.mode, space_grid=fdtd.space_grid, dtype=dtype, \
                                               engine=fdtd.engines, device_id=fdtd.device_id, MPI_extension=fdtd.comm_mode)
            else:
                self.tfsf_fdtd = incfdtd
        if   fdtd.extension == 'multi':
            self.regions  = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
            x0 = self.region[0][0]
            x1 = self.region[1][0]

            for i in xrange(len(fdtd.nx_group)):
                x_strt_node = fdtd.x_strt_group[i]
                x_stop_node = fdtd.x_stop_group[i]
                if   x0 == 0:
                    i_strt = 0
                elif x0 >  x_strt_node and x0 <= x_stop_node:
                    i_strt = i
                if   x1 >= x_strt_node and x1 <= x_stop_node:
                    i_stop = i

            self.sub_tfsfs = []
            for i, sfdtd in enumerate(fdtd.fdtd_group):
                temp_boundary = {'x':'+-', 'y':'+-', 'z':'+-'}
                for ax in ['x', 'y', 'z']:
                    temp_boundary[ax] = self.boundary[ax]
                if i >= i_strt and i <= i_stop:
                    if i != i_strt:
                        temp_boundary['x'] = temp_boundary['x'].replace('-', '')
                    if i != i_stop:
                        temp_boundary['x'] = temp_boundary['x'].replace('+', '')
                    sub_tfsf = TFSF_Boundary(sfdtd, self.regions[i], temp_boundary, self.is_oblique, self.tfsf_fdtd.fdtd_group[i])
                    sfdtd.is_tfsf = True
                    sfdtd.tfsf = sub_tfsf
                    self.sub_tfsfs.append(sub_tfsf)

        elif fdtd.extension ==   'mpi':
            self.regions   = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
            x0 = self.region[0][0]
            x1 = self.region[1][0]

            for i in xrange(len(fdtd.nx_group)):
                x_strt_node = fdtd.x_strt_group[i]
                x_stop_node = fdtd.x_stop_group[i]
                if   x0 == 0:
                    i_strt = 0
                elif x0 >  x_strt_node and x0 <= x_stop_node:
                    i_strt = i
                if   x1 >= x_strt_node and x1 <= x_stop_node:
                    i_stop = i

            temp_boundary = {'x':'+-', 'y':'+-', 'z':'+-'}
            for ax in ['x', 'y', 'z']:
                temp_boundary[ax] = self.boundary[ax]
            if fdtd.rank >= i_strt and fdtd.rank <= i_stop:
                if fdtd.rank != i_strt:
                    temp_boundary['x'] = temp_boundary['x'].replace('-', '')
                if fdtd.rank != i_stop:
                    temp_boundary['x'] = temp_boundary['x'].replace('+', '')
                sub_tfsf = TFSF_Boundary(fdtd.sub_fdtd, self.regions[fdtd.rank], temp_boundary, self.is_oblique, self.tfsf_fdtd.sub_fdtd)
                fdtd.sub_fdtd.is_tfsf = True
                fdtd.sub_fdtd.tfsf = sub_tfsf
                self.sub_tfsfs = [sub_tfsf]
                if fdtd.comm_mode == 'overlap':
                    x0 = self.regions[fdtd.rank][0][0]
                    x1 = self.regions[fdtd.rank][1][0]
                    y0 = self.regions[fdtd.rank][0][1]
                    y1 = self.regions[fdtd.rank][1][1]
                    z0 = self.regions[fdtd.rank][0][2]
                    z1 = self.regions[fdtd.rank][1][2]

                    self.buf_tfsfs = []
                    if   x0 == 0:
                        if x1 == 0:
                            region0 = ((0, y0, z0), (0, y1, z1))
                            if '+' in temp_boundary['x']:
                                buf_boundary0 = {'x':'+', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            else:
                                buf_boundary0 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                        else:
                            region0 = ((0, y0, z0), (1, y1, z1))
                            buf_boundary0 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                        buf_tfsf0 = TFSF_Boundary(fdtd.buf_fdtd_group[ 0], region0, buf_boundary0, self.is_oblique, self.tfsf_fdtd.buf_fdtd_group[ 0])
                        self.buf_tfsfs.append(buf_tfsf0)
                    elif x0 == 1:
                        region0 = ((1, y0, z0), (1, y1, z1))
                        if '-' in temp_boundary['x']:
                            buf_boundary0 = {'x':'-', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                        else:
                            buf_boundary0 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                        buf_tfsf0 = TFSF_Boundary(fdtd.buf_fdtd_group[ 0], region0, buf_boundary0, self.is_oblique, self.tfsf_fdtd.buf_fdtd_group[ 0])
                        self.buf_tfsfs.append(buf_tfsf0)

                    if   x0 <= fdtd.nx_group[fdtd.rank]-2:
                        if x1 == fdtd.nx_group[fdtd.rank]-2:
                            region1 = ((0, y0, z0), (0, y1, z1))
                            if '+' in temp_boundary['x']:
                                buf_boundary1 = {'x':'+', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            else:
                                buf_boundary1 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                        else:
                            region1 = ((0, y0, z0), (1, y1, z1))
                            buf_boundary1 = {'x':'', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                        buf_tfsf1 = TFSF_Boundary(fdtd.buf_fdtd_group[-1], region1, buf_boundary1, self.is_oblique, self.tfsf_fdtd.buf_fdtd_group[-1])
                        self.buf_tfsfs.append(buf_tfsf1)
                    elif x0 == fdtd.nx_group[fdtd.rank]-1:
                        region1 = ((1, y0, z0), (1, y1, z1))
                        if '-' in temp_boundary['x']:
                            buf_boundary1 = {'x':'-', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                        else:
                            buf_boundary1 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                        buf_tfsf1 = TFSF_Boundary(fdtd.buf_fdtd_group[-1], region1, buf_boundary1, self.is_oblique, self.tfsf_fdtd.buf_fdtd_group[-1])
                        self.buf_tfsfs.append(buf_tfsf1)
            else:
                self.sub_tfsfs = []
                self.buf_tfsfs = []

    def setup(self):
        fdtd  = self.fdtd
        ifdtd = self.tfsf_fdtd
        fdtd.engine.kernel_args['tfsf'] = {'e':{'x':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}} , \
                                                'y':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}} , \
                                                'z':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}}}, \
                                           'h':{'x':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}} , \
                                                'y':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}} , \
                                                'z':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}}}}


        if   fdtd.mode == '2DTE':
            if 'opencl' in fdtd.engine.name:
                gss = {'x': cl_global_size(self.ny, fdtd.engine.ls), \
                       'y': cl_global_size(self.nx, fdtd.engine.ls)  }
            else:
                gss = {'x': self.ny, \
                       'y': self.nx  }
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32(+1), \
                                                                        np.int32( 0), np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32(-1), \
                                                                        np.int32(+1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32(-1), \
                                                                        np.int32( 0), np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32(+1), \
                                                                        np.int32(+1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32(+1), \
                                                                        np.int32(-1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ey.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32(-1), \
                                                                        np.int32( 0), np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ey.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32(-1), \
                                                                        np.int32(-1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32(+1), \
                                                                        np.int32( 0), np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data]
            for part in fdtd.complex_parts:
                if ifdtd.is_electric:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [ifdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [ifdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [ifdtd.ce2x.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [ifdtd.ce2x.__dict__[part].data]
                if ifdtd.is_magnetic:
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [ifdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [ifdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [ifdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [ifdtd.ch2z.__dict__[part].data]
            for part in fdtd.complex_parts:
                if not ifdtd.is_uniform_grid:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [ifdtd.rdx_e[self.i_strt]]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [ifdtd.rdx_e[self.i_stop]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [ifdtd.rdy_e[self.j_strt]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [ifdtd.rdy_e[self.j_stop]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [ifdtd.rdx_h[self.i_strt]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [ifdtd.rdx_h[self.i_stop]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [ifdtd.rdy_h[self.j_strt]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [ifdtd.rdy_h[self.j_stop]]

        elif fdtd.mode == '2DTM':
            if 'opencl' in fdtd.engine.name:
                gss = {'x': cl_global_size(self.ny, fdtd.engine.ls), \
                       'y': cl_global_size(self.nx, fdtd.engine.ls)  }
            else:
                gss = {'x': self.ny, \
                       'y': self.nx  }
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32(-1), \
                                                                        np.int32( 0), np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32(+1), \
                                                                        np.int32(+1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32(+1), \
                                                                        np.int32( 0), np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32(-1), \
                                                                        np.int32(+1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32(-1), \
                                                                        np.int32(-1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32(+1), \
                                                                        np.int32( 0), np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32(+1), \
                                                                        np.int32(-1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32(-1), \
                                                                        np.int32( 0), np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ez.__dict__[part].data]
            for part in fdtd.complex_parts:
                if ifdtd.is_electric:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [ifdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [ifdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [ifdtd.ce2x.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [ifdtd.ce2x.__dict__[part].data]
                if ifdtd.is_magnetic:
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [ifdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [ifdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [ifdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [ifdtd.ch2z.__dict__[part].data]
            for part in fdtd.complex_parts:
                if not ifdtd.is_uniform_grid:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [ifdtd.rdx_e[self.i_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [ifdtd.rdx_e[self.i_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [ifdtd.rdy_e[self.j_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [ifdtd.rdy_e[self.j_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [ifdtd.rdx_h[self.i_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [ifdtd.rdx_h[self.i_stop  ]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [ifdtd.rdy_h[self.j_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [ifdtd.rdy_h[self.j_stop  ]]

        elif '3D' in fdtd.mode:
            if 'opencl' in fdtd.engine.name:
                gss = {'x': cl_global_size(self.ny*self.nz, fdtd.engine.ls), \
                       'y': cl_global_size(self.nz*self.nx, fdtd.engine.ls), \
                       'z': cl_global_size(self.nx*self.ny, fdtd.engine.ls)  }
            else:
                gss = {'x': self.ny*self.nz, \
                       'y': self.nz*self.nx, \
                       'z': self.nx*self.ny  }
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), \
                                                                        np.int32( 0), np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), \
                                                                        np.int32(+1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data, \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), \
                                                                        np.int32( 0), np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data, \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), \
                                                                        np.int32(+1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['z']['-'][part] = [fdtd.engine.queue, (gss['z'],), (fdtd.engine.ls,), \
                                                                        np.int32( 2), np.int32( 0), \
                                                                        np.int32( 0), np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hy.__dict__[part].data, \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['z']['+'][part] = [fdtd.engine.queue, (gss['z'],), (fdtd.engine.ls,), \
                                                                        np.int32( 2), np.int32( 1), \
                                                                        np.int32(+1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hx.__dict__[part].data, \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), \
                                                                        np.int32(-1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ey.__dict__[part].data, \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), \
                                                                        np.int32( 0), np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ez.__dict__[part].data, \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ey.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), \
                                                                        np.int32(-1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ez.__dict__[part].data, \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), \
                                                                        np.int32( 0), np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data, \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['z']['-'][part] = [fdtd.engine.queue, (gss['z'],), (fdtd.engine.ls,), \
                                                                        np.int32( 2), np.int32( 0), \
                                                                        np.int32(-1), np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ex.__dict__[part].data, \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ey.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['z']['+'][part] = [fdtd.engine.queue, (gss['z'],), (fdtd.engine.ls,), \
                                                                        np.int32( 2), np.int32( 1), \
                                                                        np.int32( 0), np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz), \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ey.__dict__[part].data, \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ex.__dict__[part].data]
            for part in fdtd.complex_parts:
                if ifdtd.is_electric:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [ifdtd.ce2y.__dict__[part].data, ifdtd.ce2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [ifdtd.ce2z.__dict__[part].data, ifdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [ifdtd.ce2z.__dict__[part].data, ifdtd.ce2x.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [ifdtd.ce2x.__dict__[part].data, ifdtd.ce2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['z']['-'][part] += [ifdtd.ce2x.__dict__[part].data, ifdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['z']['+'][part] += [ifdtd.ce2y.__dict__[part].data, ifdtd.ce2x.__dict__[part].data]
                if ifdtd.is_magnetic:
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [ifdtd.ch2z.__dict__[part].data, ifdtd.ch2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [ifdtd.ch2y.__dict__[part].data, ifdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [ifdtd.ch2x.__dict__[part].data, ifdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [ifdtd.ch2z.__dict__[part].data, ifdtd.ch2x.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['z']['-'][part] += [ifdtd.ch2y.__dict__[part].data, ifdtd.ch2x.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['z']['+'][part] += [ifdtd.ch2x.__dict__[part].data, ifdtd.ch2y.__dict__[part].data]
            for part in fdtd.complex_parts:
                if not ifdtd.is_uniform_grid:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [ifdtd.rdx_e[self.i_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [ifdtd.rdx_e[self.i_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [ifdtd.rdy_e[self.j_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [ifdtd.rdy_e[self.j_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['e']['z']['-'][part] += [ifdtd.rdy_e[self.k_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['z']['+'][part] += [ifdtd.rdy_e[self.k_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [ifdtd.rdx_h[self.i_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [ifdtd.rdx_h[self.i_stop  ]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [ifdtd.rdy_h[self.j_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [ifdtd.rdy_h[self.j_stop  ]]
                    fdtd.engine.kernel_args['tfsf']['h']['z']['-'][part] += [ifdtd.rdy_h[self.k_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['z']['+'][part] += [ifdtd.rdy_h[self.k_stop  ]]

        if 'cpu' in fdtd.engine.name:
            code_prev_elec = [', __FLOAT__ rds', 'rds*', ', rds', \
                              ', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', ', cf0', \
                              ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', ', cf1', \
                              ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*' , ', cf'   ]
            code_prev_magn = [', __FLOAT__ rds', 'rds*', ', rds', \
                              ', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', ', cf0', \
                              ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', ', cf1', \
                              ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*' , ', cf'   ]
        else:
            code_prev_elec = [', __FLOAT__ rds', 'rds*', \
                              ', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', \
                              ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', \
                              ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*'   ]
            code_prev_magn = [', __FLOAT__ rds', 'rds*', \
                              ', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', \
                              ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', \
                              ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*'   ]

        code_post_elec = []
        code_post_magn = []
        if ifdtd.is_uniform_grid:
            if 'cpu' in fdtd.engine.name:
                code_post_elec += ['', '', '']
                code_post_magn += ['', '', '']
            else:
                code_post_elec += ['', '']
                code_post_magn += ['', '']
        else:
            if 'cpu' in fdtd.engine.name:
                code_post_elec += [', __FLOAT__ rds', 'rds*', ', rds']
                code_post_magn += [', __FLOAT__ rds', 'rds*', ', rds']
            else:
                code_post_elec += [', __FLOAT__ rds', 'rds*']
                code_post_magn += [', __FLOAT__ rds', 'rds*']
        if ifdtd.is_electric:
            if 'cpu' in fdtd.engine.name:
                code_post_elec += [', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', ', cf0', \
                                   ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', ', cf1', \
                                   ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*' , ', cf'   ]
            else:
                code_post_elec += [', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', \
                                   ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', \
                                   ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*'   ]
        else:
            if 'cpu' in fdtd.engine.name:
                code_post_elec += ['', '%s*' % ifdtd.dt, '', \
                                   '', '%s*' % ifdtd.dt, '', \
                                   '', '%s*' % ifdtd.dt, ''  ]
            else:
                code_post_elec += ['', '%s*' % ifdtd.dt, \
                                   '', '%s*' % ifdtd.dt, \
                                   '', '%s*' % ifdtd.dt  ]

        if ifdtd.is_magnetic:
            if 'cpu' in fdtd.engine.name:
                code_post_magn += [', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', ', cf0', \
                                   ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', ', cf1', \
                                   ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*' , ', cf'   ]
            else:
                code_post_magn += [', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', \
                                   ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', \
                                   ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*'   ]
        else:
            if 'cpu' in fdtd.engine.name:
                code_post_magn += ['', '%s*' % ifdtd.dt, '', \
                                   '', '%s*' % ifdtd.dt, '', \
                                   '', '%s*' % ifdtd.dt, ''  ]
            else:
                code_post_magn += ['', '%s*' % ifdtd.dt, \
                                   '', '%s*' % ifdtd.dt, \
                                   '', '%s*' % ifdtd.dt  ]

        code_elec = template_to_code(fdtd.engine.templates['tfsf'], code_prev_elec+fdtd.engine.code_prev, code_post_elec+fdtd.engine.code_post)
        code_magn = template_to_code(fdtd.engine.templates['tfsf'], code_prev_magn+fdtd.engine.code_prev, code_post_magn+fdtd.engine.code_post)
        fdtd.engine.programs['tfsf_e'] = fdtd.engine.build(code_elec)
        fdtd.engine.programs['tfsf_h'] = fdtd.engine.build(code_magn)
        fdtd.engine.updates['tfsf_e'] = self.update_e
        fdtd.engine.updates['tfsf_h'] = self.update_h

        if   '2D' in fdtd.mode:
            if 'opencl' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.programs['tfsf_e'].update_tfsf_2d
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.programs['tfsf_h'].update_tfsf_2d
            elif 'cuda' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.get_function(fdtd.engine.programs['tfsf_e'], 'update_tfsf_2d')
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.get_function(fdtd.engine.programs['tfsf_h'], 'update_tfsf_2d')
                fdtd.engine.prepare(fdtd.engine.kernels['tfsf_e'], fdtd.engine.kernel_args['tfsf']['e']['x']['+']['real'])
                fdtd.engine.prepare(fdtd.engine.kernels['tfsf_h'], fdtd.engine.kernel_args['tfsf']['h']['x']['+']['real'])
            elif  'cpu' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['tfsf_e'].update_tfsf_2d, fdtd.engine.kernel_args['tfsf']['e']['x']['+']['real'])
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['tfsf_h'].update_tfsf_2d, fdtd.engine.kernel_args['tfsf']['h']['x']['+']['real'])
        elif '3D' in fdtd.mode:
            if 'opencl' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.programs['tfsf_e'].update_tfsf_3d
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.programs['tfsf_h'].update_tfsf_3d
            elif 'cuda' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.get_function(fdtd.engine.programs['tfsf_e'], 'update_tfsf_3d')
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.get_function(fdtd.engine.programs['tfsf_h'], 'update_tfsf_3d')
                fdtd.engine.prepare(fdtd.engine.kernels['tfsf_e'], fdtd.engine.kernel_args['tfsf']['e']['x']['+']['real'])
                fdtd.engine.prepare(fdtd.engine.kernels['tfsf_h'], fdtd.engine.kernel_args['tfsf']['h']['x']['+']['real'])
            elif  'cpu' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['tfsf_e'].update_tfsf_3d, fdtd.engine.kernel_args['tfsf']['e']['x']['+']['real'])
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['tfsf_h'].update_tfsf_3d, fdtd.engine.kernel_args['tfsf']['h']['x']['+']['real'])

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
            self.gs = self.nx*self.ny
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
            self.gs = self.nx*self.ny*self.nz
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
            self.region = ((self.x_strt, self.y_strt, self.z_strt), \
                           (self.x_stop, self.y_stop, self.z_stop))

    def update_e(self, wait=True):
        fdtd  = self.fdtd
        tfdtd = self.tfsf_fdtd
        evts = []
        if   '2D' in fdtd.mode:
            axes = ['x','y']
        elif '3D' in fdtd.mode:
            axes = ['x','y','z']
        for axis in axes:
            for direction in ['-','+']:
                for part in fdtd.complex_parts:
                    if direction in self.boundary[axis]:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['tfsf_e'](*(fdtd.engine.kernel_args['tfsf']['e'][axis][direction][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['tfsf_e'], fdtd.engine.kernel_args['tfsf']['e'][axis][direction][part])
                        elif  'cpu' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['tfsf_e'](*(fdtd.engine.kernel_args['tfsf']['e'][axis][direction][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd  = self.fdtd
        tfdtd = self.tfsf_fdtd
        evts = []
        if   '2D' in fdtd.mode:
            axes = ['x','y']
        elif '3D' in fdtd.mode:
            axes = ['x','y','z']
        for axis in axes:
            for direction in ['-','+']:
                for part in fdtd.complex_parts:
                    if direction in self.boundary[axis]:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['tfsf_h'](*(fdtd.engine.kernel_args['tfsf']['h'][axis][direction][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['tfsf_h'], fdtd.engine.kernel_args['tfsf']['h'][axis][direction][part])
                        elif  'cpu' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['tfsf_h'](*(fdtd.engine.kernel_args['tfsf']['h'][axis][direction][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

class TFSF_1D_Boundary:
    def __init__(self, fdtd, field, TFSF_region, rot_vec, pol_vec, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, material=None, is_oblique=False, incfdtd=None):
        self.fdtd = fdtd
        self.set_region(TFSF_region)
        self.is_oblique = is_oblique
        self.rot_vec = np.array(rot_vec)
        self.pol_vec = np.array(pol_vec)
        self._rvphase = 1.
        self.material = material
        if   '2D' in self.fdtd.mode:
            if not np.shape(self.rot_vec) == (2,):
                from exception import FDTDError
                raise FDTDError, 'TFSF_1D_Boundary instance must have 2-dimension direction vector'
            if not np.shape(self.pol_vec) == (2,):
                from exception import FDTDError
                raise FDTDError, 'TFSF_1D_Boundary instance must have 2-dimension polarization vector'
            if np.cross(self.rot_vec, self.pol_vec) >= 0:
                self.pol_vec *= -1
        elif '3D' in self.fdtd.mode:
            if not np.shape(self.rot_vec) == (3,):
                from exception import FDTDError
                raise FDTDError, 'TFSF_1D_Boundary instance must have 3-dimension direction vector'
            if not np.shape(self.pol_vec) == (3,):
                from exception import FDTDError
                raise FDTDError, 'TFSF_1D_Boundary instance must have 2-dimension polarization vector'
        else:
            import exception
            raise exception.FDTDError, 'Invalid fdtd mode'

        inner_product = np.dot(self.rot_vec, self.pol_vec)
        if  inner_product != 0.:
            print 'Warning:', 'TFSF-FDTD space vector is probably not perpendicular to polarization vector', inner_product
        self.rcp_vec = np.cross(self.rot_vec, self.pol_vec)
        #print 'rot', self.rot_vec
        #print 'pol', self.pol_vec
        #print 'rcp', self.rcp_vec

        check_type('field', field, str)
        check_value('fields', field, ('E', 'e', 'H', 'h'))
        self._field = field
        if   '2DTE' in self.fdtd.mode:
            if   'E' in field or 'e' in field:
                self.field = fdtd.ex
            elif 'H' in field or 'h' in field:
                self.field = fdtd.hz
        elif '2DTM' in self.fdtd.mode:
            if   'E' in field or 'e' in field:
                self.field = fdtd.ez
            elif 'H' in field or 'h' in field:
                self.field = fdtd.hx
        elif '3D' in self.fdtd.mode:
            if   'E' in field or 'e' in field:
                self.field = fdtd.ex
            elif 'H' in field or 'h' in field:
                self.field = fdtd.hy

        self.init(self.region, boundary, incfdtd)
        if fdtd.extension == 'single':
            fdtd.cores['tfsf'] = self
            self.setup()

    def init(self, region, boundary, incfdtd):
        fdtd = self.fdtd
        fdtd.is_tfsf = True
        self.boundary = boundary
        self.bn = 100
        if self.is_oblique:
            dtype = real_to_comp(fdtd.dtype)
        else:
            dtype = fdtd.dtype
        if   '2D' in fdtd.mode:
            self.nn = np.int32(np.sqrt((fdtd.nx-1)**2 + (fdtd.ny-1)**2)) + self.bn*2
            self.x_buff = 0
            temp_space_grid = (np.ones(      1, dtype=np.float64)*fdtd.min_ds, \
                               np.ones(self.nn, dtype=np.float64)*fdtd.min_ds)
            self.tfsf_fdtd = Basic_FDTD_2d(mode=fdtd.mode, space_grid=temp_space_grid, dtype=dtype, \
                                           engine=fdtd.engines, device_id=fdtd.device_id)
            pbc_apply = {'x':True, 'y':False}
            pml_apply = {'x':  '', 'y': '+-'}
            self.tfsf_fdtd.apply_PBC(pbc_apply)
            self.tfsf_fdtd.apply_PML(pml_apply)
            self.tfsf_inc = self.tfsf_fdtd.apply_direct_source(self.field.name, ((0,50),(-1,50)))
            self.tfsf_inc.correct = self.correct
            if self.material is not None:
                import structures as stc
                lx, ly = self.tfsf_fdtd.min_ds*self.tfsf_fdtd.nx, self.tfsf_fdtd.min_ds*self.tfsf_fdtd.ny
                structures_list = [stc.Rectangle(self.material, ((0.,0.),(lx,ly)))]
                self.tfsf_fdtd.set_structures(structures_list)

        elif '3D' in fdtd.mode:
#            if incfdtd is None:
#            if fdtd.extension == 'single':
#                self.nn = np.int32(np.sqrt((fdtd.nx-1)**2 + (fdtd.ny-1)**2 + (fdtd.nz-1)**2)) + self.bn*2
#                self.x_buff = 0
#                temp_space_grid = (np.ones(      1, dtype=np.float64)*fdtd.min_ds, \
#                                   np.ones(      1, dtype=np.float64)*fdtd.min_ds, \
#                                   np.ones(self.nn, dtype=np.float64)*fdtd.min_ds)
#                self.tfsf_fdtd = Basic_FDTD_3d(mode=fdtd.mode, space_grid=temp_space_grid, dtype=dtype, \
#                                               engine=fdtd.engines, device_id=fdtd.device_id)
#                pbc_apply = {'x':True, 'y':True, 'z':False}
#                pml_apply = {'x':  '', 'y':  '', 'z': '+-'}
#                self.tfsf_fdtd.apply_PBC(pbc_apply)
#                self.tfsf_fdtd.apply_PML(pml_apply)
#                self.tfsf_inc = self.tfsf_fdtd.apply_direct_source(self.field.name, ((0,0,50),(-1,-1,50)))
#            else:
#                self.tfsf_fdtd = incfdtd
#                self.tfsf_inc  = self.tfsf_fdtd.apply_direct_source(self.field.name, ((0,0,50),(-1,-1,50)))
            if fdtd.extension == 'single':
                self.nn = np.int32(np.sqrt((fdtd.nx-1)**2 + (fdtd.ny-1)**2 + (fdtd.nz-1)**2)) + self.bn*2
                self.x_buff = 0
                temp_space_grid = (np.ones(      1, dtype=np.float64)*fdtd.min_ds, \
                                   np.ones(      1, dtype=np.float64)*fdtd.min_ds, \
                                   np.ones(self.nn, dtype=np.float64)*fdtd.min_ds)
                self.tfsf_fdtd = Basic_FDTD_3d(mode=fdtd.mode, space_grid=temp_space_grid, dtype=dtype, \
                                               engine=fdtd.engines, device_id=fdtd.device_id)
                pbc_apply = {'x':True, 'y':True, 'z':False}
                pml_apply = {'x':  '', 'y':  '', 'z': '+-'}
                self.tfsf_fdtd.apply_PBC(pbc_apply)
                self.tfsf_fdtd.apply_PML(pml_apply)
                self.tfsf_inc = self.tfsf_fdtd.apply_direct_source(self.field.name, ((0,0,50),(-1,-1,50)))
                self.tfsf_inc.correct = self.correct
                if self.material is not None:
                    import structures as stc
                    lx, ly, lz = self.tfsf_fdtd.min_ds*self.tfsf_fdtd.nx, self.tfsf_fdtd.min_ds*self.tfsf_fdtd.ny, self.tfsf_fdtd.min_ds*self.tfsf_fdtd.nz
                    structures_list = [stc.Box(self.material, ((0.,0.,0.),(lx,ly,lz)))]
                    self.tfsf_fdtd.set_structures(structures_list)
            if fdtd.extension == 'multi':
                self.regions  = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
                x0 = self.region[0][0]
                x1 = self.region[1][0]

                for i in xrange(len(fdtd.nx_group)):
                    x_strt_node = fdtd.x_strt_group[i]
                    x_stop_node = fdtd.x_stop_group[i]
                    if   x0 == 0:
                        i_strt = 0
                    elif x0 >  x_strt_node and x0 <= x_stop_node:
                        i_strt = i
                    if   x1 >= x_strt_node and x1 <= x_stop_node:
                        i_stop = i

                self.sub_tfsfs = []
                for i, sfdtd in enumerate(fdtd.fdtd_group):
                    temp_boundary = {'x':'+-', 'y':'+-', 'z':'+-'}
                    for ax in ['x', 'y', 'z']:
                        temp_boundary[ax] = self.boundary[ax]
                    if i >= i_strt and i <= i_stop:
                        if i != i_strt:
                            temp_boundary['x'] = temp_boundary['x'].replace('-', '')
                        if i != i_stop:
                            temp_boundary['x'] = temp_boundary['x'].replace('+', '')
                        sub_tfsf = TFSF_1D_Boundary(sfdtd, self._field, self.regions[i], self.rot_vec, self.pol_vec, temp_boundary, self.material)
                        sfdtd.is_tfsf = True
                        sfdtd.tfsf = sub_tfsf
                        self.sub_tfsfs.append(sub_tfsf)

            elif fdtd.extension ==   'mpi':
                self.regions   = divide_region(self.region, fdtd.x_strt_group, fdtd.x_stop_group, fdtd.nx_group)
                x0 = self.region[0][0]
                x1 = self.region[1][0]

                for i in xrange(len(fdtd.nx_group)):
                    x_strt_node = fdtd.x_strt_group[i]
                    x_stop_node = fdtd.x_stop_group[i]
                    if   x0 == 0:
                        i_strt = 0
                    elif x0 >  x_strt_node and x0 <= x_stop_node:
                        i_strt = i
                    if   x1 >= x_strt_node and x1 <= x_stop_node:
                        i_stop = i

                temp_boundary = {'x':'+-', 'y':'+-', 'z':'+-'}
                for ax in ['x', 'y', 'z']:
                    temp_boundary[ax] = self.boundary[ax]
                if fdtd.rank >= i_strt and fdtd.rank <= i_stop:
                    if fdtd.rank != i_strt:
                        temp_boundary['x'] = temp_boundary['x'].replace('-', '')
                    if fdtd.rank != i_stop:
                        temp_boundary['x'] = temp_boundary['x'].replace('+', '')
                    sub_tfsf = TFSF_Boundary(fdtd.sub_fdtd, self.regions[fdtd.rank], temp_boundary, self.is_oblique)
                    fdtd.sub_fdtd.is_tfsf = True
                    fdtd.sub_fdtd.tfsf = sub_tfsf
                    self.sub_tfsfs = [sub_tfsf]
                    if fdtd.comm_mode == 'overlap':
                        x0 = self.regions[fdtd.rank][0][0]
                        x1 = self.regions[fdtd.rank][1][0]
                        y0 = self.regions[fdtd.rank][0][1]
                        y1 = self.regions[fdtd.rank][1][1]
                        z0 = self.regions[fdtd.rank][0][2]
                        z1 = self.regions[fdtd.rank][1][2]

                        self.buf_tfsfs = []
                        if   x0 == 0:
                            if x1 == 0:
                                region0 = ((0, y0, z0), (0, y1, z1))
                                if '+' in temp_boundary['x']:
                                    buf_boundary0 = {'x':'+', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                                else:
                                    buf_boundary0 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            else:
                                region0 = ((0, y0, z0), (1, y1, z1))
                                buf_boundary0 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            buf_tfsf0 = TFSF_Boundary(fdtd.buf_fdtd_group[ 0], region0, buf_boundary0, self.is_oblique, self.tfsf_fdtd.buf_fdtd_group[ 0])
                            self.buf_tfsfs.append(buf_tfsf0)
                        elif x0 == 1:
                            region0 = ((1, y0, z0), (1, y1, z1))
                            if '-' in temp_boundary['x']:
                                buf_boundary0 = {'x':'-', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            else:
                                buf_boundary0 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            buf_tfsf0 = TFSF_Boundary(fdtd.buf_fdtd_group[ 0], region0, buf_boundary0, self.is_oblique, self.tfsf_fdtd.buf_fdtd_group[ 0])
                            self.buf_tfsfs.append(buf_tfsf0)

                        if   x0 <= fdtd.nx_group[fdtd.rank]-2:
                            if x1 == fdtd.nx_group[fdtd.rank]-2:
                                region1 = ((0, y0, z0), (0, y1, z1))
                                if '+' in temp_boundary['x']:
                                    buf_boundary1 = {'x':'+', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                                else:
                                    buf_boundary1 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            else:
                                region1 = ((0, y0, z0), (1, y1, z1))
                                buf_boundary1 = {'x':'', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            buf_tfsf1 = TFSF_Boundary(fdtd.buf_fdtd_group[-1], region1, buf_boundary1, self.is_oblique, self.tfsf_fdtd.buf_fdtd_group[-1])
                            self.buf_tfsfs.append(buf_tfsf1)
                        elif x0 == fdtd.nx_group[fdtd.rank]-1:
                            region1 = ((1, y0, z0), (1, y1, z1))
                            if '-' in temp_boundary['x']:
                                buf_boundary1 = {'x':'-', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            else:
                                buf_boundary1 = {'x': '', 'y':temp_boundary['y'], 'z':temp_boundary['z']}
                            buf_tfsf1 = TFSF_Boundary(fdtd.buf_fdtd_group[-1], region1, buf_boundary1, self.is_oblique, self.tfsf_fdtd.buf_fdtd_group[-1])
                            self.buf_tfsfs.append(buf_tfsf1)
                else:
                    self.sub_tfsfs = []
                    self.buf_tfsfs = []

    def setup(self):
        fdtd  = self.fdtd
        ifdtd = self.tfsf_fdtd
        fdtd.engine.kernel_args['tfsf'] = {'e':{'x':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}} , \
                                                'y':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}} , \
                                                'z':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}}}, \
                                           'h':{'x':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}} , \
                                                'y':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}} , \
                                                'z':{'+':{'real':None, 'imag':None}  , \
                                                     '-':{'real':None, 'imag':None}}}}


        if   fdtd.mode == '2DTE':
            if 'opencl' in fdtd.engine.name:
                gss = {'x': cl_global_size(self.ny, fdtd.engine.ls), \
                       'y': cl_global_size(self.nx, fdtd.engine.ls)  }
            else:
                gss = {'x': self.ny, \
                       'y': self.nx  }
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32(+1), \
                                                                        np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(             +1), \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32(-1), \
                                                                        np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(             +1), \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32(-1), \
                                                                        np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(             +1), \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32(+1), \
                                                                        np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(             +1), \
                                                                        fdtd.ex.__dict__[part].data , ifdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32(+1), \
                                                                        np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[1]), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32(-1), \
                                                                        np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0+1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[1]), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32(-1), \
                                                                        np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[0]), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32(+1), \
                                                                        np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+1.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[0]), \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data]
            for part in fdtd.complex_parts:
                if fdtd.is_electric:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [fdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [fdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [fdtd.ce2x.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [fdtd.ce2x.__dict__[part].data]
                if fdtd.is_magnetic:
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [fdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [fdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [fdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [fdtd.ch2z.__dict__[part].data]
            for part in fdtd.complex_parts:
                if not fdtd.is_uniform_grid:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [fdtd.rdx_e[self.i_strt]]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [fdtd.rdx_e[self.i_stop]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [fdtd.rdy_e[self.j_strt]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [fdtd.rdy_e[self.j_stop]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [fdtd.rdx_h[self.i_strt]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [fdtd.rdx_h[self.i_stop]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [fdtd.rdy_h[self.j_strt]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [fdtd.rdy_h[self.j_stop]]

        elif fdtd.mode == '2DTM':
            if 'opencl' in fdtd.engine.name:
                gss = {'x': cl_global_size(self.ny, fdtd.engine.ls), \
                       'y': cl_global_size(self.nx, fdtd.engine.ls)  }
            else:
                gss = {'x': self.ny, \
                       'y': self.nx  }
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32(-1), \
                                                                        np.int32( 0), \
                                                                        #np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[1]), \
                                                                        #fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32(+1), \
                                                                        np.int32(+1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[1]), \
                                                                        #fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32(+1), \
                                                                        np.int32( 0), \
                                                                        #np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[0]), \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32(-1), \
                                                                        np.int32(+1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[0]), \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32(-1), \
                                                                        np.int32(-1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(              +1), \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32(+1), \
                                                                        np.int32( 0), \
                                                                        #np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(              +1), \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32(+1), \
                                                                        np.int32(-1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(              +1), \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32(-1), \
                                                                        np.int32( 0), \
                                                                        #np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), \
                                                                        np.int32(self.nn    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+1.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(              +1), \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ez.__dict__[part].data]
            for part in fdtd.complex_parts:
                if fdtd.is_electric:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [fdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [fdtd.ce2y.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [fdtd.ce2x.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [fdtd.ce2x.__dict__[part].data]
                if fdtd.is_magnetic:
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [fdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [fdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [fdtd.ch2z.__dict__[part].data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [fdtd.ch2z.__dict__[part].data]
            for part in fdtd.complex_parts:
                if not fdtd.is_uniform_grid:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [fdtd.rdx_e[self.i_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [fdtd.rdx_e[self.i_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [fdtd.rdy_e[self.j_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [fdtd.rdy_e[self.j_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [fdtd.rdx_h[self.i_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [fdtd.rdx_h[self.i_stop  ]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [fdtd.rdy_h[self.j_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [fdtd.rdy_h[self.j_stop  ]]

        elif '3D' in fdtd.mode:
            if 'opencl' in fdtd.engine.name:
                gss = {'x': cl_global_size(self.ny*self.nz, fdtd.engine.ls), \
                       'y': cl_global_size(self.nz*self.nx, fdtd.engine.ls), \
                       'z': cl_global_size(self.nx*self.ny, fdtd.engine.ls)  }
            else:
                gss = {'x': self.ny*self.nz, \
                       'y': self.nz*self.nx, \
                       'z': self.nx*self.ny  }
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32( 0), \
                                                                        #np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hy.__dict__[part].data, \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32(+1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data, \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32( 0), \
                                                                        #np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        #fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data, \
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data, \
                                                                        #fdtd.ex.__dict__[part].data, ifdtd.hz.__dict__[part].data]
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32(+1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        #fdtd.ex.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hy.__dict__[part].data, \
                                                                        #fdtd.ez.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                                                                        fdtd.ez.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['z']['-'][part] = [fdtd.engine.queue, (gss['z'],), (fdtd.engine.ls,), \
                                                                        np.int32( 2), np.int32( 0), np.int32( 0), \
                                                                        #np.int32(-1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5-1.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hy.__dict__[part].data, \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hx.__dict__[part].data]
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['e']['z']['+'][part] = [fdtd.engine.queue, (gss['z'],), (fdtd.engine.ls,), \
                                                                        np.int32( 2), np.int32( 1), np.int32(+1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.5), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rcp_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hx.__dict__[part].data, \
                                                                        fdtd.ey.__dict__[part].data, ifdtd.hy.__dict__[part].data, \
                                                                        fdtd.ex.__dict__[part].data, ifdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 0), np.int32(-1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        #fdtd.hz.__dict__[part].data, ifdtd.ey.__dict__[part].data, \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data, \
                                                                        #fdtd.hy.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] = [fdtd.engine.queue, (gss['x'],), (fdtd.engine.ls,), \
                                                                        np.int32( 0), np.int32( 1), np.int32( 0), \
                                                                        #np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.0+1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0+1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        #fdtd.hy.__dict__[part].data, ifdtd.ez.__dict__[part].data, \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ex.__dict__[part].data, \
                                                                        #fdtd.hz.__dict__[part].data, ifdtd.ey.__dict__[part].data]
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 0), np.int32(-1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        #fdtd.hx.__dict__[part].data, ifdtd.ez.__dict__[part].data, \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ex.__dict__[part].data, \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] = [fdtd.engine.queue, (gss['y'],), (fdtd.engine.ls,), \
                                                                        np.int32( 1), np.int32( 1), np.int32( 0), \
                                                                        #np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0+1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0+1.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(-0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        fdtd.hz.__dict__[part].data, ifdtd.ex.__dict__[part].data, \
                                                                        #fdtd.hx.__dict__[part].data, ifdtd.ez.__dict__[part].data]
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['z']['-'][part] = [fdtd.engine.queue, (gss['z'],), (fdtd.engine.ls,), \
                                                                        np.int32( 2), np.int32( 0), np.int32(-1), \
                                                                        #np.int32( 0), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(+0.5), \
                                                                        comp_to_real(fdtd.dtype)(+0.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ex.__dict__[part].data, \
                                                                        #fdtd.hx.__dict__[part].data, ifdtd.ey.__dict__[part].data]
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['tfsf']['h']['z']['+'][part] = [fdtd.engine.queue, (gss['z'],), (fdtd.engine.ls,), \
                                                                        np.int32( 2), np.int32( 1), np.int32( 0), \
                                                                        #np.int32(+1), \
                                                                        np.int32(self.x_strt), np.int32(self.y_strt), np.int32(self.z_strt), \
                                                                        np.int32(self.nx    ), np.int32(self.ny    ), np.int32(self.nz    ), \
                                                                        np.int32(fdtd.nx    ), np.int32(fdtd.ny    ), np.int32(fdtd.nz    ), \
                                                                        np.int32(self.nn    ), np.int32(self.x_buff), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0+1.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.5    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+0.0+1.0), \
                                                                        comp_to_real(fdtd.dtype)(-0.0    ), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.rot_vec[2]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[1]), \
                                                                        comp_to_real(fdtd.dtype)(+self.pol_vec[0]), \
                                                                        comp_to_real(fdtd.dtype)( self._rvphase  ), \
                                                                        #fdtd.ey.__dict__[part].data, ifdtd.hz.__dict__[part].data, \
                                                                        #fdtd.hx.__dict__[part].data, ifdtd.ey.__dict__[part].data, \
                                                                        fdtd.hx.__dict__[part].data, ifdtd.ex.__dict__[part].data, \
                                                                        fdtd.hy.__dict__[part].data, ifdtd.ex.__dict__[part].data]
            for part in fdtd.complex_parts:
                if fdtd.is_electric:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [fdtd.ce2y.data, fdtd.ce2z.data]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [fdtd.ce2z.data, fdtd.ce2y.data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [fdtd.ce2z.data, fdtd.ce2x.data]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [fdtd.ce2x.data, fdtd.ce2z.data]
                    fdtd.engine.kernel_args['tfsf']['e']['z']['-'][part] += [fdtd.ce2x.data, fdtd.ce2y.data]
                    fdtd.engine.kernel_args['tfsf']['e']['z']['+'][part] += [fdtd.ce2y.data, fdtd.ce2x.data]
                if fdtd.is_magnetic:
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [fdtd.ch2z.data, fdtd.ch2y.data]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [fdtd.ch2y.data, fdtd.ch2z.data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [fdtd.ch2x.data, fdtd.ch2z.data]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [fdtd.ch2z.data, fdtd.ch2x.data]
                    fdtd.engine.kernel_args['tfsf']['h']['z']['-'][part] += [fdtd.ch2y.data, fdtd.ch2x.data]
                    fdtd.engine.kernel_args['tfsf']['h']['z']['+'][part] += [fdtd.ch2x.data, fdtd.ch2y.data]
            for part in fdtd.complex_parts:
                if not ifdtd.is_uniform_grid:
                    fdtd.engine.kernel_args['tfsf']['e']['x']['-'][part] += [fdtd.rdx_e[self.i_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['x']['+'][part] += [fdtd.rdx_e[self.i_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['-'][part] += [fdtd.rdy_e[self.j_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['y']['+'][part] += [fdtd.rdy_e[self.j_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['e']['z']['-'][part] += [fdtd.rdy_e[self.k_strt  ]]
                    fdtd.engine.kernel_args['tfsf']['e']['z']['+'][part] += [fdtd.rdy_e[self.k_stop+1]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['-'][part] += [fdtd.rdx_h[self.i_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['x']['+'][part] += [fdtd.rdx_h[self.i_stop  ]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['-'][part] += [fdtd.rdy_h[self.j_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['y']['+'][part] += [fdtd.rdy_h[self.j_stop  ]]
                    fdtd.engine.kernel_args['tfsf']['h']['z']['-'][part] += [fdtd.rdy_h[self.k_strt-1]]
                    fdtd.engine.kernel_args['tfsf']['h']['z']['+'][part] += [fdtd.rdy_h[self.k_stop  ]]

        if 'cpu' in fdtd.engine.name:
            code_prev_elec = [', __FLOAT__ rds', 'rds*', ', rds', \
                              ', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', ', cf0', \
                              ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', ', cf1', \
                              ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*' , ', cf'   ]
            code_prev_magn = [', __FLOAT__ rds', 'rds*', ', rds', \
                              ', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', ', cf0', \
                              ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', ', cf1', \
                              ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*' , ', cf'   ]
        else:
            code_prev_elec = [', __FLOAT__ rds', 'rds*', \
                              ', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', \
                              ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', \
                              ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*'   ]
            code_prev_magn = [', __FLOAT__ rds', 'rds*', \
                              ', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', \
                              ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', \
                              ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*'   ]

        code_post_elec = []
        code_post_magn = []
        if ifdtd.is_uniform_grid:
            if 'cpu' in fdtd.engine.name:
                code_post_elec += ['', '', '']
                code_post_magn += ['', '', '']
            else:
                code_post_elec += ['', '']
                code_post_magn += ['', '']
        else:
            if 'cpu' in fdtd.engine.name:
                code_post_elec += [', __FLOAT__ rds', 'rds*', ', rds']
                code_post_magn += [', __FLOAT__ rds', 'rds*', ', rds']
            else:
                code_post_elec += [', __FLOAT__ rds', 'rds*']
                code_post_magn += [', __FLOAT__ rds', 'rds*']
        if ifdtd.is_electric:
            if 'cpu' in fdtd.engine.name:
                code_post_elec += [', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', ', cf0', \
                                   ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', ', cf1', \
                                   ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*' , ', cf'   ]
            else:
                code_post_elec += [', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', \
                                   ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', \
                                   ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*'   ]
        else:
            if 'cpu' in fdtd.engine.name:
                code_post_elec += ['', '%s*' % fdtd.dt, '', \
                                   '', '%s*' % fdtd.dt, '', \
                                   '', '%s*' % fdtd.dt, ''  ]
            else:
                code_post_elec += ['', '%s*' % fdtd.dt, \
                                   '', '%s*' % fdtd.dt, \
                                   '', '%s*' % fdtd.dt  ]

        if ifdtd.is_magnetic:
            if 'cpu' in fdtd.engine.name:
                code_post_magn += [', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', ', cf0', \
                                   ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', ', cf1', \
                                   ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*' , ', cf'   ]
            else:
                code_post_magn += [', __GLOBAL__ __FLOAT__* cf0', 'cf0[idx_tfsf]*', \
                                   ', __GLOBAL__ __FLOAT__* cf1', 'cf1[idx_tfsf]*', \
                                   ', __GLOBAL__ __FLOAT__* cf' , 'cf[idx_tfsf]*'   ]
        else:
            if 'cpu' in fdtd.engine.name:
                code_post_magn += ['', '%s*' % fdtd.dt, '', \
                                   '', '%s*' % fdtd.dt, '', \
                                   '', '%s*' % fdtd.dt, ''  ]
            else:
                code_post_magn += ['', '%s*' % fdtd.dt, \
                                   '', '%s*' % fdtd.dt, \
                                   '', '%s*' % fdtd.dt  ]

        code_elec = template_to_code(fdtd.engine.templates['tfsf'], code_prev_elec+fdtd.engine.code_prev, code_post_elec+fdtd.engine.code_post)
        code_magn = template_to_code(fdtd.engine.templates['tfsf'], code_prev_magn+fdtd.engine.code_prev, code_post_magn+fdtd.engine.code_post)
        fdtd.engine.programs['tfsf_e'] = fdtd.engine.build(code_elec)
        fdtd.engine.programs['tfsf_h'] = fdtd.engine.build(code_magn)
        fdtd.engine.updates['tfsf_e'] = self.update_e
        fdtd.engine.updates['tfsf_h'] = self.update_h

        if   '2D' in fdtd.mode:
            if 'opencl' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.programs['tfsf_e'].update_tfsf_1d_to_2d
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.programs['tfsf_h'].update_tfsf_1d_to_2d
            elif 'cuda' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.get_function(fdtd.engine.programs['tfsf_e'], 'update_tfsf_1d_to_2d')
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.get_function(fdtd.engine.programs['tfsf_h'], 'update_tfsf_1d_to_2d')
                fdtd.engine.prepare(fdtd.engine.kernels['tfsf_e'], fdtd.engine.kernel_args['tfsf']['e']['x']['+']['real'])
                fdtd.engine.prepare(fdtd.engine.kernels['tfsf_h'], fdtd.engine.kernel_args['tfsf']['h']['x']['+']['real'])
            elif  'cpu' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['tfsf_e'].update_tfsf_1d_to_2d, fdtd.engine.kernel_args['tfsf']['e']['x']['+']['real'])
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['tfsf_h'].update_tfsf_1d_to_2d, fdtd.engine.kernel_args['tfsf']['h']['x']['+']['real'])
        elif '3D' in fdtd.mode:
            if 'opencl' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.programs['tfsf_e'].update_tfsf_1d_to_3d
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.programs['tfsf_h'].update_tfsf_1d_to_3d
            elif 'cuda' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.get_function(fdtd.engine.programs['tfsf_e'], 'update_tfsf_1d_to_3d')
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.get_function(fdtd.engine.programs['tfsf_h'], 'update_tfsf_1d_to_3d')
                fdtd.engine.prepare(fdtd.engine.kernels['tfsf_e'], fdtd.engine.kernel_args['tfsf']['e']['x']['+']['real'])
                fdtd.engine.prepare(fdtd.engine.kernels['tfsf_h'], fdtd.engine.kernel_args['tfsf']['h']['x']['+']['real'])
            elif  'cpu' in fdtd.engine.name:
                fdtd.engine.kernels['tfsf_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['tfsf_e'].update_tfsf_1d_to_3d, fdtd.engine.kernel_args['tfsf']['e']['x']['+']['real'])
                fdtd.engine.kernels['tfsf_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['tfsf_h'].update_tfsf_1d_to_3d, fdtd.engine.kernel_args['tfsf']['h']['x']['+']['real'])

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
            self.gs = self.nx*self.ny
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
            self.gs = self.nx*self.ny*self.nz
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
            self.region = ((self.x_strt, self.y_strt, self.z_strt), \
                           (self.x_stop, self.y_stop, self.z_stop))

    def correct(self, wavelength):
        fdtd = self.fdtd
        tfdtd = self.tfsf_fdtd
        ds = tfdtd.min_ds
        nl = int(wavelength/ds)
        s  = tfdtd.dt
        k0 = (2./ds)*np.arcsin((1./s)*np.sin(np.pi*s/nl))
        kvec = self.rot_vec
        self._rvphase = correct_phase_velocity_tfsf1d(k0, s, ds, nl, kvec, 100)
        self.setup()

    def update_e(self, wait=True):
        fdtd  = self.fdtd
        tfdtd = self.tfsf_fdtd
        evts = []
        if   '2D' in fdtd.mode:
            axes = ['x','y']
        elif '3D' in fdtd.mode:
            axes = ['x','y','z']
        for axis in axes:
            for direction in ['-','+']:
                for part in fdtd.complex_parts:
                    if direction in self.boundary[axis]:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['tfsf_e'](*(fdtd.engine.kernel_args['tfsf']['e'][axis][direction][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['tfsf_e'], fdtd.engine.kernel_args['tfsf']['e'][axis][direction][part])
                        elif  'cpu' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['tfsf_e'](*(fdtd.engine.kernel_args['tfsf']['e'][axis][direction][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd  = self.fdtd
        tfdtd = self.tfsf_fdtd
        evts = []
        if   '2D' in fdtd.mode:
            axes = ['x','y']
        elif '3D' in fdtd.mode:
            axes = ['x','y','z']
        for axis in axes:
            for direction in ['-','+']:
                for part in fdtd.complex_parts:
                    if direction in self.boundary[axis]:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['tfsf_h'](*(fdtd.engine.kernel_args['tfsf']['h'][axis][direction][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['tfsf_h'], fdtd.engine.kernel_args['tfsf']['h'][axis][direction][part])
                        elif  'cpu' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['tfsf_h'](*(fdtd.engine.kernel_args['tfsf']['h'][axis][direction][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts
