import numpy as np

from util import *
from units import *
from ndarray import Fields

#import pyopencl as cl

class PBC_1d:
    def __init__(self, fdtd, pbc_apply):
        self.fdtd = fdtd
        self.fdtd.cores['pbc'] = self
        self.fdtd.prgs['pbc_e'] = self.update_e
        self.fdtd.prgs['pbc_h'] = self.update_h
        self.pbc_apply = pbc_apply

    def update_e(self):
        if self.pbc_apply['z']: self.fdtd.ex[-1] = self.fdtd.ex[0]

    def update_h(self):
        if self.pbc_apply['z']: self.fdtd.hy[0] = self.fdtd.hy[-1]

class PBC_2d:
    def __init__(self, fdtd, pbc_apply):
        self.fdtd = fdtd
        self.fdtd.cores['pbc'] = self
        self.fdtd.engine.updates['pbc_e'] = self.update_e
        self.fdtd.engine.updates['pbc_h'] = self.update_h
        self.pbc_apply = pbc_apply

        self.setup()

    def setup(self):
        ef   = 0; hf   = 1;
        ax_x = 0; ax_y = 1;
        fdtd = self.fdtd
        code = template_to_code(fdtd.engine.templates['pbc'], fdtd.engine.code_prev, fdtd.engine.code_post)
        fdtd.engine.programs['pbc'] = fdtd.engine.build(code)
        fdtd.engine.kernel_args['pbc_e'] = {'x':{'real':None, 'imag':None}, 'y':{'real':None, 'imag':None}}
        fdtd.engine.kernel_args['pbc_h'] = {'x':{'real':None, 'imag':None}, 'y':{'real':None, 'imag':None}}

        if   'opencl' in fdtd.engine.name:
            gs_x = cl_global_size(fdtd.ny, fdtd.engine.ls)
            gs_y = cl_global_size(fdtd.nx, fdtd.engine.ls)
        else:
            gs_x = fdtd.ny
            gs_y = fdtd.nx

        if   fdtd.mode == '2DTE':
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['pbc_e']['x'][part] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                               np.int32(ef), np.int32(ax_x), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                               fdtd.ey.__dict__[part].data]
                fdtd.engine.kernel_args['pbc_e']['y'][part] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                               np.int32(ef), np.int32(ax_y), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                               fdtd.ex.__dict__[part].data]
                fdtd.engine.kernel_args['pbc_h']['x'][part] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                               np.int32(hf), np.int32(ax_x), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                               fdtd.hz.__dict__[part].data]
                fdtd.engine.kernel_args['pbc_h']['y'][part] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                               np.int32(hf), np.int32(ax_y), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                               fdtd.hz.__dict__[part].data]
        elif fdtd.mode == '2DTM':
            for part in fdtd.complex_parts:
                fdtd.engine.kernel_args['pbc_e']['x'][part] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                               np.int32(ef), np.int32(ax_x), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                               fdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['pbc_e']['y'][part] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                               np.int32(ef), np.int32(ax_y), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                               fdtd.ez.__dict__[part].data]
                fdtd.engine.kernel_args['pbc_h']['x'][part] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                               np.int32(hf), np.int32(ax_x), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                               fdtd.hy.__dict__[part].data]
                fdtd.engine.kernel_args['pbc_h']['y'][part] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                               np.int32(hf), np.int32(ax_y), \
                                                               np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                               fdtd.hx.__dict__[part].data]

        if   'opencl' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.programs['pbc'].update_2d
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.programs['pbc'].update_2d
        elif  'cuda' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.get_function(fdtd.engine.programs['pbc'], 'update_2d')
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.get_function(fdtd.engine.programs['pbc'], 'update_2d')
            fdtd.engine.prepare(fdtd.engine.kernels['pbc_e'], fdtd.engine.kernel_args['pbc_e']['x'][part])
            fdtd.engine.prepare(fdtd.engine.kernels['pbc_h'], fdtd.engine.kernel_args['pbc_h']['x'][part])
        elif   'cpu' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['pbc'].update_2d, fdtd.engine.kernel_args['pbc_e']['x'][part])
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['pbc'].update_2d, fdtd.engine.kernel_args['pbc_h']['x'][part])

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for ax in ['x', 'y']:
            if self.pbc_apply[ax]:
                for part in fdtd.complex_parts:
                    if   'opencl' in fdtd.engine.name:
                        evt = fdtd.engine.kernels['pbc_e'](*(fdtd.engine.kernel_args['pbc_e'][ax][part]))
                    elif 'cuda' in fdtd.engine.name:
                        evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pbc_e'], fdtd.engine.kernel_args['pbc_e'][ax][part])
                    elif  'cpu' in fdtd.engine.name:
                        evt = fdtd.engine.kernels['pbc_e'](*(fdtd.engine.kernel_args['pbc_e'][ax][part][3:]))
                    evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for ax in ['x', 'y']:
            if self.pbc_apply[ax]:
                for part in fdtd.complex_parts:
                    if   'opencl' in fdtd.engine.name:
                        evt = fdtd.engine.kernels['pbc_h'](*(fdtd.engine.kernel_args['pbc_h'][ax][part]))
                    elif 'cuda' in fdtd.engine.name:
                        evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pbc_h'], fdtd.engine.kernel_args['pbc_h'][ax][part])
                    elif  'cpu' in fdtd.engine.name:
                        evt = fdtd.engine.kernels['pbc_h'](*(fdtd.engine.kernel_args['pbc_h'][ax][part][3:]))
                    evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

class PBC_3d:
    def __init__(self, fdtd, pbc_apply):
        self.fdtd = fdtd
        self.pbc_apply = pbc_apply
        self.fdtd.cores['pbc'] = self
        self.fdtd.engine.updates['pbc_e'] = self.update_e
        self.fdtd.engine.updates['pbc_h'] = self.update_h

        self.setup()

    def setup(self):
        fdtd = self.fdtd
        ef   = 0; hf   = 1;
        ax_x = 0; ax_y = 1; ax_z = 2;
        code = template_to_code(fdtd.engine.templates['pbc'], fdtd.engine.code_prev, fdtd.engine.code_post)
        fdtd.engine.programs['pbc'] = fdtd.engine.build(code)

        fdtd.engine.kernel_args['pbc_e'] = {'x':{0:{'real':None, 'imag':None}, 1:{'real':None, 'imag':None}}, \
                                            'y':{0:{'real':None, 'imag':None}, 1:{'real':None, 'imag':None}}, \
                                            'z':{0:{'real':None, 'imag':None}, 1:{'real':None, 'imag':None}}}
        fdtd.engine.kernel_args['pbc_h'] = {'x':{0:{'real':None, 'imag':None}, 1:{'real':None, 'imag':None}}, \
                                            'y':{0:{'real':None, 'imag':None}, 1:{'real':None, 'imag':None}}, \
                                            'z':{0:{'real':None, 'imag':None}, 1:{'real':None, 'imag':None}}}

        if 'opencl' in fdtd.engine.name:
            gs_x = cl_global_size(fdtd.ny*fdtd.nz, fdtd.engine.ls)
            gs_y = cl_global_size(fdtd.nz*fdtd.nx, fdtd.engine.ls)
            gs_z = cl_global_size(fdtd.nx*fdtd.ny, fdtd.engine.ls)
        else:
            gs_x = fdtd.ny*fdtd.nz
            gs_y = fdtd.nz*fdtd.nx
            gs_z = fdtd.nx*fdtd.ny

        for part in fdtd.complex_parts:
            fdtd.engine.kernel_args['pbc_e']['x'][0][part] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                              np.int32(ef), np.int32(ax_x), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.ey.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_h']['x'][0][part] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                              np.int32(hf), np.int32(ax_x), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.hy.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_e']['x'][1][part] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                              np.int32(ef), np.int32(ax_x), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.ez.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_h']['x'][1][part] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                              np.int32(hf), np.int32(ax_x), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.hz.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_e']['y'][0][part] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                              np.int32(ef), np.int32(ax_y), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.ez.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_h']['y'][0][part] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                              np.int32(hf), np.int32(ax_y), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.hz.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_e']['y'][1][part] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                              np.int32(ef), np.int32(ax_y), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.ex.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_h']['y'][1][part] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                              np.int32(hf), np.int32(ax_y), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.hx.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_e']['z'][0][part] = [fdtd.engine.queue, (gs_z,), (fdtd.engine.ls,), \
                                                              np.int32(ef), np.int32(ax_z), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.ex.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_h']['z'][0][part] = [fdtd.engine.queue, (gs_z,), (fdtd.engine.ls,), \
                                                              np.int32(hf), np.int32(ax_z), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.hx.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_e']['z'][1][part] = [fdtd.engine.queue, (gs_z,), (fdtd.engine.ls,), \
                                                              np.int32(ef), np.int32(ax_z), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.ey.__dict__[part].data]
            fdtd.engine.kernel_args['pbc_h']['z'][1][part] = [fdtd.engine.queue, (gs_z,), (fdtd.engine.ls,), \
                                                              np.int32(hf), np.int32(ax_z), \
                                                              np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                              fdtd.hy.__dict__[part].data]

        if 'opencl' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.programs['pbc'].update_3d
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.programs['pbc'].update_3d
        elif 'cuda' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.get_function(fdtd.engine.programs['pbc'], 'update_3d')
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.get_function(fdtd.engine.programs['pbc'], 'update_3d')
            fdtd.engine.prepare(fdtd.engine.kernels['pbc_e'], fdtd.engine.kernel_args['pbc_e']['x'][0]['real'])
            fdtd.engine.prepare(fdtd.engine.kernels['pbc_h'], fdtd.engine.kernel_args['pbc_h']['x'][0]['real'])
        elif  'cpu' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['pbc'].update_3d, fdtd.engine.kernel_args['pbc_e']['x'][0]['real'])
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['pbc'].update_3d, fdtd.engine.kernel_args['pbc_h']['x'][0]['real'])

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for ax in ['x', 'y', 'z']:
            if self.pbc_apply[ax]:
                for i in xrange(2):
                    for part in fdtd.complex_parts:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pbc_e'](*(fdtd.engine.kernel_args['pbc_e'][ax][i][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pbc_e'], fdtd.engine.kernel_args['pbc_e'][ax][i][part])
                        elif  'cpu' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pbc_e'](*(fdtd.engine.kernel_args['pbc_e'][ax][i][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for ax in ['x', 'y', 'z']:
            if self.pbc_apply[ax]:
                for i in xrange(2):
                    for part in fdtd.complex_parts:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pbc_h'](*(fdtd.engine.kernel_args['pbc_h'][ax][i][part]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pbc_h'], fdtd.engine.kernel_args['pbc_h'][ax][i][part])
                        elif  'cpu' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pbc_h'](*(fdtd.engine.kernel_args['pbc_h'][ax][i][part][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

class BBC_2d:
    def __init__(self, fdtd, pbc_apply={'x':True, 'y':True}, klx=0., kly=0.):
        if not fdtd.is_complex:
            import exception
            raise exception.BoundaryConditionError, 'Set dtype of FDTD class to complex, not float'
        self.fdtd = fdtd
        self.fdtd.cores['pbc'] = self
        self.fdtd.engine.updates['pbc_e'] = self.update_e
        self.fdtd.engine.updates['pbc_h'] = self.update_h
        self.pbc_apply = pbc_apply

        self.klx = klx
        self.kly = kly

        self.sinklx = np.sin(klx)
        self.cosklx = np.cos(klx)

        self.sinkly = np.sin(kly)
        self.coskly = np.cos(kly)

        self.setup()

    def setup(self):
        ef   = 0; hf   = 1;
        ax_x = 0; ax_y = 1;
        fdtd = self.fdtd
        code = template_to_code(fdtd.engine.templates['bbc'], fdtd.engine.code_prev, fdtd.engine.code_post)
        fdtd.engine.programs['pbc'] = fdtd.engine.build(code)
        fdtd.engine.kernel_args['pbc_e'] = {'x':None, 'y':None}
        fdtd.engine.kernel_args['pbc_h'] = {'x':None, 'y':None}

        if 'opencl' in fdtd.engine.name:
            gs_x = cl_global_size(fdtd.ny, fdtd.engine.ls)
            gs_y = cl_global_size(fdtd.nx, fdtd.engine.ls)
        else:
            gs_x = fdtd.ny
            gs_y = fdtd.nx

        if fdtd.mode == '2DTE':
            fdtd.engine.kernel_args['pbc_e']['x'] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                     np.int32(ef), np.int32(ax_x), \
                                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                     comp_to_real(fdtd.dtype)(-self.sinklx), \
                                                     comp_to_real(fdtd.dtype)(+self.cosklx), \
                                                     fdtd.ey.real.data, fdtd.ey.imag.data]
            fdtd.engine.kernel_args['pbc_e']['y'] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                     np.int32(ef), np.int32(ax_y), \
                                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                     comp_to_real(fdtd.dtype)(-self.sinkly), \
                                                     comp_to_real(fdtd.dtype)(+self.coskly), \
                                                     fdtd.ex.real.data, fdtd.ex.imag.data]
            fdtd.engine.kernel_args['pbc_h']['x'] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                     np.int32(hf), np.int32(ax_x), \
                                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                     comp_to_real(fdtd.dtype)(+self.sinklx), \
                                                     comp_to_real(fdtd.dtype)(+self.cosklx), \
                                                     fdtd.hz.real.data, fdtd.hz.imag.data]
            fdtd.engine.kernel_args['pbc_h']['y'] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                     np.int32(hf), np.int32(ax_y), \
                                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                     comp_to_real(fdtd.dtype)(+self.sinkly), \
                                                     comp_to_real(fdtd.dtype)(+self.coskly), \
                                                     fdtd.hz.real.data, fdtd.hz.imag.data]
        if fdtd.mode == '2DTM':
            fdtd.engine.kernel_args['pbc_e']['x'] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                     np.int32(ef), np.int32(ax_x), \
                                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                     comp_to_real(fdtd.dtype)(-self.sinklx), \
                                                     comp_to_real(fdtd.dtype)(+self.cosklx), \
                                                     fdtd.ez.real.data, fdtd.ez.imag.data]
            fdtd.engine.kernel_args['pbc_e']['y'] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                     np.int32(ef), np.int32(ax_y), \
                                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                     comp_to_real(fdtd.dtype)(-self.sinkly), \
                                                     comp_to_real(fdtd.dtype)(+self.coskly), \
                                                     fdtd.ez.real.data, fdtd.ez.imag.data]
            fdtd.engine.kernel_args['pbc_h']['x'] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                     np.int32(hf), np.int32(ax_x), \
                                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                     comp_to_real(fdtd.dtype)(+self.sinklx), \
                                                     comp_to_real(fdtd.dtype)(+self.cosklx), \
                                                     fdtd.hy.real.data, fdtd.hy.imag.data]
            fdtd.engine.kernel_args['pbc_h']['y'] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                     np.int32(hf), np.int32(ax_y), \
                                                     np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                                     comp_to_real(fdtd.dtype)(+self.sinkly), \
                                                     comp_to_real(fdtd.dtype)(+self.coskly), \
                                                     fdtd.hx.real.data, fdtd.hx.imag.data]

        if 'opencl' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.programs['pbc'].update_2d
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.programs['pbc'].update_2d
        elif 'cuda' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.get_function(fdtd.engine.programs['pbc'], 'update_2d')
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.get_function(fdtd.engine.programs['pbc'], 'update_2d')
            fdtd.engine.prepare(fdtd.engine.kernels['pbc_e'], fdtd.engine.kernel_args['pbc_e']['x'])
            fdtd.engine.prepare(fdtd.engine.kernels['pbc_h'], fdtd.engine.kernel_args['pbc_h']['x'])
        elif  'cpu' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['pbc'].update_2d, fdtd.engine.kernel_args['pbc_e']['x'])
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['pbc'].update_2d, fdtd.engine.kernel_args['pbc_h']['x'])

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for ax in ['x', 'y']:
            if self.pbc_apply[ax]:
                if   'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['pbc_e'](*(fdtd.engine.kernel_args['pbc_e'][ax]))
                elif 'cuda' in fdtd.engine.name:
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pbc_e'], fdtd.engine.kernel_args['pbc_e'][ax])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['pbc_e'](*(fdtd.engine.kernel_args['pbc_e'][ax][3:]))
                evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for ax in ['x', 'y']:
            if self.pbc_apply[ax]:
                if   'opencl' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['pbc_h'](*(fdtd.engine.kernel_args['pbc_h'][ax]))
                elif 'cuda' in fdtd.engine.name:
                    evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pbc_h'], fdtd.engine.kernel_args['pbc_h'][ax])
                elif  'cpu' in fdtd.engine.name:
                    evt = fdtd.engine.kernels['pbc_h'](*(fdtd.engine.kernel_args['pbc_h'][ax][3:]))
                evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    '''
    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if   self.fdtd.mode == '2DTE':
            for axis in ['x', 'y']:
                if self.pbc_apply[axis]:
                    fdtd.kernels['pbc_e']
                evt = self.prg.update_2d(fdtd.queue, (gs,), (fdtd.ls,), \
                                         np.int32(self.ef), np.int32(self.ax_x), \
                                         np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                         comp_to_real(fdtd.dtype)(-self.sinklx), \
                                         comp_to_real(fdtd.dtype)(+self.cosklx), \
                                         fdtd.ey.real.data, fdtd.ey.imag.data)
                evts.append(evt)
            if self.pbc_apply['y']:
                gs = cl_global_size(fdtd.nx, fdtd.ls)
                evt = self.prg.update_2d(fdtd.queue, (gs,), (fdtd.ls,), \
                                         np.int32(self.ef), np.int32(self.ax_y), \
                                         np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                         comp_to_real(fdtd.dtype)(-self.sinkly), \
                                         comp_to_real(fdtd.dtype)(+self.coskly), \
                                         fdtd.ex.real.data, fdtd.ex.imag.data)
                evts.append(evt)

        elif self.fdtd.mode == '2DTM':
            if self.pbc_apply['x']:
                gs = cl_global_size(fdtd.ny, fdtd.ls)
                evt = self.prg.update_2d(fdtd.queue, (gs,), (fdtd.ls,), \
                                         np.int32(self.ef), np.int32(self.ax_x), \
                                         np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                         comp_to_real(fdtd.dtype)(-self.sinklx), \
                                         comp_to_real(fdtd.dtype)(+self.cosklx), \
                                         fdtd.ez.real.data, fdtd.ez.imag.data)
                evts.append(evt)
            if self.pbc_apply['y']:
                gs = cl_global_size(fdtd.nx, fdtd.ls)
                evt = self.prg.update_2d(fdtd.queue, (gs,), (fdtd.ls,), \
                                         np.int32(self.ef), np.int32(self.ax_y), \
                                         np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                         comp_to_real(fdtd.dtype)(-self.sinkly), \
                                         comp_to_real(fdtd.dtype)(+self.coskly), \
                                         fdtd.ez.real.data, fdtd.ez.imag.data)
                evts.append(evt)
        if wait and len(evts) != 0: cl.wait_for_events(evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if   self.fdtd.mode == '2DTE':
            if self.pbc_apply['x']:
                gs = cl_global_size(fdtd.ny, fdtd.ls)
                evt = self.prg.update_2d(fdtd.queue, (gs,), (fdtd.ls,), \
                                         np.int32(self.hf), np.int32(self.ax_x), \
                                         np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                         comp_to_real(fdtd.dtype)(+self.sinklx), \
                                         comp_to_real(fdtd.dtype)(+self.cosklx), \
                                         fdtd.hz.real.data, fdtd.hz.imag.data)
                evts.append(evt)
            if self.pbc_apply['y']:
                gs = cl_global_size(fdtd.nx, fdtd.ls)
                evt = self.prg.update_2d(fdtd.queue, (gs,), (fdtd.ls,), \
                                         np.int32(self.hf), np.int32(self.ax_y), \
                                         np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                         comp_to_real(fdtd.dtype)(+self.sinkly), \
                                         comp_to_real(fdtd.dtype)(+self.coskly), \
                                         fdtd.hz.real.data, fdtd.hz.imag.data)
                evts.append(evt)

        elif self.fdtd.mode == '2DTM':
            if self.pbc_apply['x']:
                gs = cl_global_size(fdtd.ny, fdtd.ls)
                evt = self.prg.update_2d(fdtd.queue, (gs,), (fdtd.ls,), \
                                         np.int32(self.hf), np.int32(self.ax_x), \
                                         np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                         comp_to_real(fdtd.dtype)(+self.sinklx), \
                                         comp_to_real(fdtd.dtype)(+self.cosklx), \
                                         fdtd.hy.real.data, fdtd.hy.imag.data)
                evts.append(evt)
            if self.pbc_apply['y']:
                gs = cl_global_size(fdtd.nx, fdtd.ls)
                evt = self.prg.update_2d(fdtd.queue, (gs,), (fdtd.ls,), \
                                         np.int32(self.hf), np.int32(self.ax_y), \
                                         np.int32(fdtd.nx), np.int32(fdtd.ny), \
                                         comp_to_real(fdtd.dtype)(+self.sinkly), \
                                         comp_to_real(fdtd.dtype)(+self.coskly), \
                                         fdtd.hx.real.data, fdtd.hx.imag.data)
                evts.append(evt)
        if wait and len(evts) != 0: cl.wait_for_events(evts)
        return evts
    '''

class BBC_3d:
    def __init__(self, fdtd, pbc_apply={'x':True, 'y':True, 'z':True}, klx=0., kly=0., klz=0.):
        if not fdtd.is_complex:
            import exception
            raise exception.BoundaryConditionError, 'Set dtype of FDTD class to complex, not float'
        self.fdtd = fdtd
        self.fdtd.cores['pbc'] = self
        self.fdtd.engine.updates['pbc_e'] = self.update_e
        self.fdtd.engine.updates['pbc_h'] = self.update_h
        self.pbc_apply = pbc_apply

        self.klx = klx
        self.kly = kly
        self.klz = klz

        self.sinklx = np.sin(klx)
        self.cosklx = np.cos(klx)

        self.sinkly = np.sin(kly)
        self.coskly = np.cos(kly)

        self.sinklz = np.sin(klz)
        self.cosklz = np.cos(klz)

        self.setup()

    def setup(self):
        ef   = 0; hf   = 1;
        ax_x = 0; ax_y = 1; ax_z = 2;
        fdtd = self.fdtd
        code = template_to_code(fdtd.engine.templates['bbc'], fdtd.engine.code_prev, fdtd.engine.code_post)
        fdtd.engine.programs['pbc'] = fdtd.engine.build(code)
        fdtd.engine.kernel_args['pbc_e'] = {'x':[None,None], 'y':[None,None], 'z':[None,None]}
        fdtd.engine.kernel_args['pbc_h'] = {'x':[None,None], 'y':[None,None], 'z':[None,None]}

        if 'opencl' in fdtd.engine.name:
            gs_x = cl_global_size(fdtd.ny*fdtd.nz, fdtd.engine.ls)
            gs_y = cl_global_size(fdtd.nz*fdtd.nx, fdtd.engine.ls)
            gs_z = cl_global_size(fdtd.nx*fdtd.ny, fdtd.engine.ls)
        else:
            gs_x = fdtd.ny*fdtd.nz
            gs_y = fdtd.nz*fdtd.nx
            gs_z = fdtd.nx*fdtd.ny

        fdtd.engine.kernel_args['pbc_e']['x'][0] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                    np.int32(ef), np.int32(ax_x), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(-self.sinklx), \
                                                    comp_to_real(fdtd.dtype)(+self.cosklx), \
                                                    fdtd.ey.real.data, fdtd.ey.imag.data]
        fdtd.engine.kernel_args['pbc_e']['x'][1] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                    np.int32(ef), np.int32(ax_x), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(-self.sinklx), \
                                                    comp_to_real(fdtd.dtype)(+self.cosklx), \
                                                    fdtd.ez.real.data, fdtd.ez.imag.data]
        fdtd.engine.kernel_args['pbc_e']['y'][0] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                    np.int32(ef), np.int32(ax_y), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(-self.sinkly), \
                                                    comp_to_real(fdtd.dtype)(+self.coskly), \
                                                    fdtd.ez.real.data, fdtd.ez.imag.data]
        fdtd.engine.kernel_args['pbc_e']['y'][1] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                    np.int32(ef), np.int32(ax_y), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(-self.sinkly), \
                                                    comp_to_real(fdtd.dtype)(+self.coskly), \
                                                    fdtd.ex.real.data, fdtd.ex.imag.data]
        fdtd.engine.kernel_args['pbc_e']['z'][0] = [fdtd.engine.queue, (gs_z,), (fdtd.engine.ls,), \
                                                    np.int32(ef), np.int32(ax_z), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(-self.sinklz), \
                                                    comp_to_real(fdtd.dtype)(+self.cosklz), \
                                                    fdtd.ex.real.data, fdtd.ex.imag.data]
        fdtd.engine.kernel_args['pbc_e']['z'][1] = [fdtd.engine.queue, (gs_z,), (fdtd.engine.ls,), \
                                                    np.int32(ef), np.int32(ax_z), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(-self.sinklz), \
                                                    comp_to_real(fdtd.dtype)(+self.cosklz), \
                                                    fdtd.ey.real.data, fdtd.ey.imag.data]

        fdtd.engine.kernel_args['pbc_h']['x'][0] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                    np.int32(hf), np.int32(ax_x), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(+self.sinklx), \
                                                    comp_to_real(fdtd.dtype)(+self.cosklx), \
                                                    fdtd.hy.real.data, fdtd.hy.imag.data]
        fdtd.engine.kernel_args['pbc_h']['x'][1] = [fdtd.engine.queue, (gs_x,), (fdtd.engine.ls,), \
                                                    np.int32(hf), np.int32(ax_x), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(+self.sinklx), \
                                                    comp_to_real(fdtd.dtype)(+self.cosklx), \
                                                    fdtd.hz.real.data, fdtd.hz.imag.data]
        fdtd.engine.kernel_args['pbc_h']['y'][0] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                    np.int32(hf), np.int32(ax_y), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(+self.sinkly), \
                                                    comp_to_real(fdtd.dtype)(+self.coskly), \
                                                    fdtd.hz.real.data, fdtd.hz.imag.data]
        fdtd.engine.kernel_args['pbc_h']['y'][1] = [fdtd.engine.queue, (gs_y,), (fdtd.engine.ls,), \
                                                    np.int32(hf), np.int32(ax_y), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(+self.sinkly), \
                                                    comp_to_real(fdtd.dtype)(+self.coskly), \
                                                    fdtd.hx.real.data, fdtd.hx.imag.data]
        fdtd.engine.kernel_args['pbc_h']['z'][0] = [fdtd.engine.queue, (gs_z,), (fdtd.engine.ls,), \
                                                    np.int32(hf), np.int32(ax_z), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(+self.sinklz), \
                                                    comp_to_real(fdtd.dtype)(+self.cosklz), \
                                                    fdtd.hx.real.data, fdtd.hx.imag.data]
        fdtd.engine.kernel_args['pbc_h']['z'][1] = [fdtd.engine.queue, (gs_z,), (fdtd.engine.ls,), \
                                                    np.int32(hf), np.int32(ax_z), \
                                                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                                    comp_to_real(fdtd.dtype)(+self.sinklz), \
                                                    comp_to_real(fdtd.dtype)(+self.cosklz), \
                                                    fdtd.hy.real.data, fdtd.hy.imag.data]
        if 'opencl' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.programs['pbc'].update_3d
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.programs['pbc'].update_3d
        elif 'cuda' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.get_function(fdtd.engine.programs['pbc'], 'update_3d')
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.get_function(fdtd.engine.programs['pbc'], 'update_3d')
            fdtd.engine.prepare(fdtd.engine.kernels['pbc_e'], fdtd.engine.kernel_args['pbc_e']['x'][0])
            fdtd.engine.prepare(fdtd.engine.kernels['pbc_h'], fdtd.engine.kernel_args['pbc_h']['x'][0])
        elif  'cpu' in fdtd.engine.name:
            fdtd.engine.kernels['pbc_e'] = fdtd.engine.set_kernel(fdtd.engine.programs['pbc'].update_3d, fdtd.engine.kernel_args['pbc_e']['x'][0])
            fdtd.engine.kernels['pbc_h'] = fdtd.engine.set_kernel(fdtd.engine.programs['pbc'].update_3d, fdtd.engine.kernel_args['pbc_h']['x'][0])

    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for ax in ['x', 'y', 'z']:
            if self.pbc_apply[ax]:
                for i in xrange(2):
                    for part in fdtd.complex_parts:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pbc_e'](*(fdtd.engine.kernel_args['pbc_e'][ax][i]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pbc_e'], fdtd.engine.kernel_args['pbc_e'][ax][i])
                        elif  'cpu' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pbc_e'](*(fdtd.engine.kernel_args['pbc_e'][ax][i][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        for ax in ['x', 'y', 'z']:
            if self.pbc_apply[ax]:
                for i in xrange(2):
                    for part in fdtd.complex_parts:
                        if 'opencl' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pbc_h'](*(fdtd.engine.kernel_args['pbc_h'][ax][i]))
                        elif 'cuda' in fdtd.engine.name:
                            evt = fdtd.engine.enqueue_kernel(fdtd.engine.kernels['pbc_h'], fdtd.engine.kernel_args['pbc_h'][ax][i])
                        elif  'cpu' in fdtd.engine.name:
                            evt = fdtd.engine.kernels['pbc_h'](*(fdtd.engine.kernel_args['pbc_h'][ax][i][3:]))
                        evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        return evts

    '''
    def update_e(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if self.pbc_apply['x']:
            gs = cl_global_size(fdtd.ny*fdtd.nz, fdtd.ls)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.ef), np.int32(self.ax_x), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(-self.sinklx), \
                                     comp_to_real(fdtd.dtype)(+self.cosklx), \
                                     fdtd.ey.real.data, fdtd.ey.imag.data)
            evts.append(evt)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.ef), np.int32(self.ax_x), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(-self.sinklx), \
                                     comp_to_real(fdtd.dtype)(+self.cosklx), \
                                     fdtd.ez.real.data, fdtd.ez.imag.data)
            evts.append(evt)
        if self.pbc_apply['y']:
            gs = cl_global_size(fdtd.nz*fdtd.nx, fdtd.ls)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.ef), np.int32(self.ax_y), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(-self.sinkly), \
                                     comp_to_real(fdtd.dtype)(+self.coskly), \
                                     fdtd.ez.real.data, fdtd.ez.imag.data)
            evts.append(evt)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.ef), np.int32(self.ax_y), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(-self.sinkly), \
                                     comp_to_real(fdtd.dtype)(+self.coskly), \
                                     fdtd.ex.real.data, fdtd.ex.imag.data)
            evts.append(evt)
        if self.pbc_apply['z']:
            gs = cl_global_size(fdtd.nx*fdtd.ny, fdtd.ls)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.ef), np.int32(self.ax_z), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(-self.sinklz), \
                                     comp_to_real(fdtd.dtype)(+self.cosklz), \
                                     fdtd.ex.real.data, fdtd.ex.imag.data)
            evts.append(evt)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.ef), np.int32(self.ax_z), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(-self.sinklz), \
                                     comp_to_real(fdtd.dtype)(+self.cosklz), \
                                     fdtd.ey.real.data, fdtd.ey.imag.data)
            evts.append(evt)
        if wait and len(evts) != 0: cl.wait_for_events(evts)
        return evts

    def update_h(self, wait=True):
        fdtd = self.fdtd
        evts = []
        if self.pbc_apply['x']:
            gs = cl_global_size(fdtd.ny*fdtd.nz, fdtd.ls)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.hf), np.int32(self.ax_x), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(+self.sinklx), \
                                     comp_to_real(fdtd.dtype)(+self.cosklx), \
                                     fdtd.hy.real.data, fdtd.hy.imag.data)
            evts.append(evt)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.hf), np.int32(self.ax_x), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(+self.sinklx), \
                                     comp_to_real(fdtd.dtype)(+self.cosklx), \
                                     fdtd.hz.real.data, fdtd.hz.imag.data)
            evts.append(evt)
        if self.pbc_apply['y']:
            gs = cl_global_size(fdtd.nz*fdtd.nx, fdtd.ls)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.hf), np.int32(self.ax_y), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(+self.sinkly), \
                                     comp_to_real(fdtd.dtype)(+self.coskly), \
                                     fdtd.hz.real.data, fdtd.hz.imag.data)
            evts.append(evt)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.hf), np.int32(self.ax_y), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(+self.sinkly), \
                                     comp_to_real(fdtd.dtype)(+self.coskly), \
                                     fdtd.hx.real.data, fdtd.hx.imag.data)
            evts.append(evt)
        if self.pbc_apply['z']:
            gs = cl_global_size(fdtd.nx*fdtd.ny, fdtd.ls)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.hf), np.int32(self.ax_z), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(+self.sinklz), \
                                     comp_to_real(fdtd.dtype)(+self.cosklz), \
                                     fdtd.hx.real.data, fdtd.hx.imag.data)
            evts.append(evt)
            evt = self.prg.update_3d(fdtd.queue, (gs,), (fdtd.ls,), \
                                     np.int32(self.hf), np.int32(self.ax_z), \
                                     np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                                     comp_to_real(fdtd.dtype)(+self.sinklz), \
                                     comp_to_real(fdtd.dtype)(+self.cosklz), \
                                     fdtd.hy.real.data, fdtd.hy.imag.data)
            evts.append(evt)
        if wait and len(evts) != 0: cl.wait_for_events(evts)
        return evts
    '''

def PBC(fdtd, pbc_apply):
    if   fdtd.mode == '1D':
        return PBC_1d(fdtd, pbc_apply)
    elif fdtd.mode in ['2DTE', '2DTM']:
        return PBC_2d(fdtd, pbc_apply)
    elif fdtd.mode == '3D':
        return PBC_3d(fdtd, pbc_apply)

def BBC(fdtd, pbc_apply, klx=0., kly=0., klz=0.):
    if   fdtd.mode in ['2DTE', '2DTM']:
        return BBC_2d(fdtd, pbc_apply, klx=klx, kly=kly)
    elif fdtd.mode == '3D':
        return BBC_3d(fdtd, pbc_apply, klx=klx, kly=kly, klz=klz)

