# utility functions

import sys
import types
import numpy as np
import scipy as sc
import multiprocessing as mproc
import subprocess as sproc
import doctest
import unittest

from datetime import datetime as dtm

int_types     = (    int,     np.int32,      np.int64)
float_types   = (  float,   np.float32,    np.float64)
complex_types = (complex, np.complex64, np.complex128)
numpy_float_types   = (  np.float32,    np.float64)
numpy_complex_types = (np.complex64, np.complex128)

# cuda get global index 'GridDim', 'BlockDim'
cuda_get_global_index_1d_1d = 'blockIdx.x*blockDim.x + threadIdx.x'
cuda_get_global_index_1d_2d = 'blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x'
cuda_get_global_index_1d_3d = 'blockIdx.x*blockDim.x*blockDim.y*blockDim.z + threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x'
cuda_get_global_index_2d_1d = '(blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x'
cuda_get_global_index_2d_2d = '(blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x'
cuda_get_global_index_2d_3d = '(blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y*blockDim.z + threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x'

cuda_get_local_index_1d_1d = 'threadIdx.x'
cuda_get_local_index_1d_2d = 'threadIdx.y*blockDim.x + threadIdx.x'
cuda_get_local_index_1d_3d = 'threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x'
cuda_get_local_index_2d_1d = 'threadIdx.x'
cuda_get_local_index_2d_2d = 'threadIdx.y*blockDim.x + threadIdx.x'
cuda_get_local_index_2d_3d = 'threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x'

class FakeEvent:
    def __init__(self):
        pass
    def wait(self):
        pass

class FakeBuffer:
    def __init__(self, value=0.):
        self.value = value
    def release(self):
        pass
    def free(self):
        pass
    def __coerce__(self, other):
        return self.value, other

class Ext_executor(mproc.Process):
    def __init__(self, cmd_queue, out_queue, err_queue):
        mproc.Process.__init__(self)
        self.cmd_queue = cmd_queue
        self.out_queue = out_queue
        self.err_queue = err_queue

    def run(self):
        while True:
            try:
                cmd = self.cmd_queue.get()
                if cmd is None:
                    out, err = '', ''
                    break
                else:
                    out, err = sproc.Popen(cmd.split(), stdout=sproc.PIPE, stderr=sproc.PIPE).communicate()
                self.out_queue.put(out)
                self.err_queue.put(err)
            except KeyboardInterrupt:            
                break
        return

def wait_for_events(fdtd, evts):
    if 'opencl' in fdtd.engine_name and len(evts) != 0:
        fdtd.engines[0].cl.wait_for_events(evts)
    elif 'cuda' in fdtd.engine_name and len(evts) != 0:
        for evt in evts:
            evt.wait()
#        for engine in fdtd.engines:
#            engine.ctx.push()
#            engine.stream.synchronize()

def dtype_to_bytes(dtype):
    to_bytes = {    int:4,     np.int32:4,      np.int64:8,\
                  float:4,   np.float32:4,    np.float64:8,\
                complex:4, np.complex64:4, np.complex128:8}
    return to_bytes[dtype]

def comp_to_real(comp):
    to_real = {  float:float,   np.float32:np.float32,    np.float64:np.float64, \
               complex:float, np.complex64:np.float32, np.complex128:np.float64}
    return to_real[comp]

def real_to_comp(real):
    to_comp = {  float:complex,   np.float32:np.complex64,    np.float64:np.complex128, \
               complex:complex, np.complex64:np.complex64, np.complex128:np.complex128}
    return to_comp[real]

def DFT(t_dist, freqs, times):
    nax = np.newaxis
    dt = times[1] - times[0]
    try:
        f_dist_t = (dt/np.sqrt(2.*np.pi))*t_dist[nax,:]*np.exp(-2.j*np.pi*freqs[:,nax]*times[nax,:])
        f_dist = f_dist_t.sum(1)
    except MemoryError:
        f_dist = np.zeros(freqs.size, dtype=real_to_comp(t_dist.dtype))
        for f, freq in enumerate(freqs):
            f_dist[f] = (dt/np.sqrt(2.*np.pi))*t_dist*np.exp(-2.j*np.pi*freq*times).sum()
    return f_dist

def iDFT(f_dist, freqs, times):
    nax = np.newaxis
    df = freqs[1] - freqs[0]
    try:
        t_dist_f = (df/np.sqrt(2.*np.pi))*f_dist[nax,:]*np.exp(+2.j*np.pi*freqs[nax,:]*times[:,nax])
        t_dist = t_dist_f.sum(1)
    except MemoryError:
        t_dist = np.zeros(times.size, dtype=real_to_comp(f_dist.dtype))
        for t, time in enumerate(times):
            t_dist[t] = (df/np.sqrt(2.*np.pi))*f_dist*np.exp(+2.j*np.pi*freqs*time).sum()
    return t_dist

def shape_to_size(shape):
    size = 1
    for n in shape:
        if n <= 0:
            raise ValueError, 'Attribute \'shape\' of %s objects must have natural number' % str(ndarray)
        size *= n
    return size

def convert_to_tuple(arg):
    """
    Return the tuple which is converted from the arbitrary argument

    >>> convert_to_tuple(3)
    (3,)
    >>> convert_to_tuple(['a', 'b'])
    ('a', 'b')
    """

    if isinstance(arg, (list, tuple)):
        return tuple(arg)
    else:
        return (arg,)

def check_type(arg_name, arg, arg_type, element_type=None):
    """
    Check the type of the argument
    If the type is mismatch, the TypeError exception is raised.

    When the 'arg's type is a list or a tuple,
    each element's type is also checked.

    >>> check_type('arg_name', 2, int)
    >>> check_type('arg_name', 3.4, (int, float))
    >>> check_type('arg_name', 'xy', str)
    >>> check_type('arg_name', (1.2, 2.3), tuple, float)
    >>> check_type('arg_name', ['a', 'b'], (list, tuple), str)
    """

    if not isinstance(arg, arg_type):
        raise TypeError("argument '%s' type must be a %s : %s is given" % \
                (arg_name, repr(arg_type), type(arg)) )

    if isinstance(arg, (list, tuple)):
        if element_type == None:
            raise TypeError( \
                "\n\tWhen the 'arg's type is a list or a tuple, \
                \n\targument 'element_type' must be specified." )

        for element in arg:
            if not isinstance(element, element_type):
                raise TypeError("argument '%s's element type must be a %s : %s is given" % \
                        (arg_name, repr(element_type), type(element)) )

def check_value(arg_name, arg, value):
    """
    Check if the argument is one of the values
    If the value is mismatch, the ValueError exception is raised.

    >>> check_value('arg_name', 'a', ('a', 'b', 'ab'))
    """

    if not arg in convert_to_tuple(value):
        repr_val = repr(value)
        if isinstance(value, (list, tuple)) and len(repr_val) > 40:
            repr_val = str(value[:2] + ['...'] + value[-2:]).replace("'", '')

        raise ValueError("argument '%s' value must be one of %s : %s is given" % \
                (arg_name, repr_val, repr(arg)) )

def binary_prefix_nbytes(nbytes):
    """
    Return a (converted nbytes, binary prefix) pair for the nbytes

    >>> binary_prefix_nbytes(2e9)
    (1.862645149230957, 'GiB')
    >>> binary_prefix_nbytes(2e6)
    (1.9073486328125, 'MiB')
    >>> binary_prefix_nbytes(2e3)
    (1.953125, 'KiB')
    >>> binary_prefix_nbytes(2)
    (2, 'Bytes')
    """

    check_type('nbytes', nbytes, (int, float))

    if nbytes >= 1024**3:
        value = float(nbytes)/(1024**3)
        prefix_str = 'GiB'

    elif nbytes >= 1024**2:
        value = float(nbytes)/(1024**2)
        prefix_str = 'MiB'

    elif nbytes >= 1024:
        value = float(nbytes)/1024
        prefix_str = 'KiB'

    else:
        value = nbytes
        prefix_str = 'Bytes'

    return value, prefix_str

def cl_global_size(gs, ls):
    return int(gs + (ls - (gs%ls)))

def cuda_grid(gs, ls):
    b_size, rest = divmod(gs, ls)
    if rest > 0:
        b_size += 1
    grid_x       =    int(np.sqrt(b_size))
    grid_y, rest = divmod(b_size, grid_x)
    if rest > 0:
        grid_y += 1
    return (int(grid_x), int(grid_y))


#def cuda_grid(gs, ls):
#    return (65535, 2)

def divide_nx(nx, div_num, i):
    residue = nx % div_num
    if i < residue-1: return (nx + div_num)/div_num + 1
    else: return (nx + div_num)/div_num

def template_to_code(template, str0, str1):
    if type(template) is not str:
        raise TypeError, 'template must have type \'string\''
    if type(str0) is not type(str1):
        raise TypeError, 'input variables must have the same type'
    if   type(str0) is str:
        template = template.replace(str0, str1)
    elif type(str0) in [list, tuple]:
        for i in xrange(len(str0)):
            template = template.replace(str0[i], str1[i])
    return template

def rotate(axis, angle, vec):
    if len(axis) != 3:
        raise ValueError, 'rotation axis vector must have 3 dimension'
    if np.linalg.norm(axis) == 0.:
        raise ValueError, 'rotation axis vector must be nonzero vector'
    axis = np.array(axis)/np.linalg.norm(np.array(axis))

    ux, uy, uz = axis
    cost = np.cos(angle)
    sint = np.sin(angle)
    uxx = ux*ux
    uyy = uy*uy
    uzz = uz*uz
    uxy = ux*uy
    uyz = uy*uz
    uzx = uz*ux
    rot_mat = np.matrix([[cost+uxx*(1-cost), uxy*(1-cost)-uz*sint, uzx*(1-cost)+uy*sint], \
                         [uxy*(1-cost)+uz*sint, cost+uyy*(1-cost), uyz*(1-cost)-ux*sint], \
                         [uzx*(1-cost)-uy*sint, uyz*(1-cost)+ux*sint, cost+uzz*(1-cost)]] )
    rot_vec = rot_mat*np.matrix(vec).T
    return np.array(rot_vec.T)[0]

def rotate_jones(inc_angle, jones):
    tht = inc_angle[0] # radians
    phi = inc_angle[1] # radians
    sint = np.sin(tht)
    cost = np.cos(tht)
    sinp = np.sin(phi)
    cosp = np.cos(phi)
    rot_angle = tht
    rot_axis  = (-sint*sinp, sint*cost, 0)
    jones = (jones[0], jones[1], 0.)
    rot_jones = rotate(rot_axis, rot_angle, jones)
    return rot_jones

def jones_to_amp_pha(jones3d):
    jones3d = np.array(jones3d)/np.linalg.norm(np.array(jones3d))
    amp = np.array([np.abs(jones3d[i]) for i in xrange(3)])
    pha = []
    for i in xrange(3):
        try:
            pha.append(np.real(-1.j*np.log(np.complex(jones3d[i])/np.abs(jones3d[i]))))
        except ZeroDivisionError:
            pha.append(0.)
    return amp, np.array(pha)

def GridCalculator(space_grid):
    check_type('space_grid', space_grid, (list, tuple), np.ndarray)
    if len(space_grid) == 1:
        x       = [space_grid[i].cumsum() for i in xrange(1)]
    if len(space_grid) == 2:
        x, y    = [space_grid[i].cumsum() for i in xrange(2)]
    if len(space_grid) == 3:
        x, y, z = [space_grid[i].cumsum() for i in xrange(3)]

def GridDisplay(*space_grid):
    import matplotlib.pyplot as plt
    check_type('space_grid', space_grid, (list, tuple), np.ndarray)
    if len(space_grid) == 1:
        x       = space_grid[0].cumsum()
        y       = np.zeros(1)
        fig = plt.figure(figsize=(10,10))
        xy  = fig.add_subplot(111)

        xy.set_title('Grid XY-Plane')
        for j in xrange(y.size):
            xy.plot(x, np.array([y[j] for i in xrange(x.size)]), 'ro-')

        plt.show()

    if len(space_grid) == 2:
        x, y    = [space_grid[i].cumsum() for i in xrange(2)]
        fig = plt.figure(figsize=(10,10))
        xy  = fig.add_subplot(111)

        xy.set_title('Grid XY-Plane')
        for i in xrange(x.size):
            xy.plot(np.array([x[i] for j in xrange(y.size)]), y, 'ro-', markersize=2.)
        for j in xrange(y.size):
            xy.plot(x, np.array([y[j] for i in xrange(x.size)]), 'ro-', markersize=.1)

        plt.show()

    if len(space_grid) == 3:
        x, y, z = [space_grid[i].cumsum() for i in xrange(3)]
        fig = plt.figure(figsize=(15,5))
        xy  = fig.add_subplot(131)
        yz  = fig.add_subplot(132)
        zx  = fig.add_subplot(133)

        xy.set_title('Grid XY-Plane')
        for i in xrange(x.size):
            xy.plot(np.array([x[i] for j in xrange(y.size)]), y, 'ro-', markersize=.1)
        for j in xrange(y.size):
            xy.plot(x, np.array([y[j] for i in xrange(x.size)]), 'ro-', markersize=.1)

        yz.set_title('Grid YZ-Plane')
        for j in xrange(y.size):
            yz.plot(np.array([y[j] for k in xrange(z.size)]), z, 'go-', markersize=.1)
        for k in xrange(z.size):
            yz.plot(y, np.array([z[k] for j in xrange(y.size)]), 'go-', markersize=.1)

        zx.set_title('Grid ZX-Plane')
        for k in xrange(z.size):
            zx.plot(np.array([z[k] for i in xrange(x.size)]), x, 'bo-', markersize=.1)
        for i in xrange(x.size):
            zx.plot(z, np.array([x[i] for k in xrange(z.size)]), 'bo-', markersize=.1)

        plt.show()

def search_idx(arr, val, opt='interval'):
    check_type('array', arr, np.ndarray)
    check_value('opt', opt, ('interval', 'near'))
    if opt == 'interval':
        if   arr[0] > val:
            return -1, 0
        for i in xrange(arr.size-1):
            if arr[i] <= val and arr[i+1] > val:
                return i, i+1
        return arr.size-1, arr.size
    elif opt == 'near':
        if arr[0] > val:
            return 0
        for i in xrange(arr.size-1):
            if arr[i] <= val and arr[i+1] > val:
                if abs(arr[i]-val) < abs(arr[i+1]-val):
                    return i
                else:
                    return i+1
        return arr.size-1

def check_uniform_grid(space_grid):
    check_type('space_grid', space_grid, (list, tuple), np.ndarray)
    ndim = len(space_grid)
    for grid in space_grid:
        if (grid/grid.min()).std() > 0.:
            return False
    grid_mins = np.array([grid.min() for grid in space_grid])
    if (grid_mins/grid_mins.min()).std() > 0.:
        return False
    else:
        return True

def divide_xgrid(x_grid, num):
    nx = x_grid.size + 1
    nx_group, x_strt_group, x_stop_group = \
        [np.zeros(num, dtype=np.int) for i in xrange(3)]
    for i in xrange(num):
        sub_nx = divide_nx(nx, num, i)
        nx_group[i] = sub_nx
        if i != num - 1:
            x_strt_group[i+1:] += sub_nx - 1
        x_stop_group[i:] += sub_nx - 1

    return nx_group, x_strt_group, x_stop_group

def set_coordinate(fdtd, x_offset=0.):
    fdtd.x_SI_pts, fdtd.y_SI_pts, fdtd.z_SI_pts = \
        [np.array([0.] + list(np.array(fdtd.space_grid[i]).cumsum())) for i in xrange(3)]
    fdtd.x_SI_pts += x_offset
    fdtd.x_SI_cel, fdtd.y_SI_cel, fdtd.z_SI_cel = \
        [np.zeros(fdtd.space_grid[i].size+1, dtype=comp_to_real(fdtd.dtype)) for i in xrange(3)]
    fdtd.x_SI_cel[:-1] = (fdtd.x_SI_pts[:-1] + fdtd.x_SI_pts[1:])*.5
    fdtd.y_SI_cel[:-1] = (fdtd.y_SI_pts[:-1] + fdtd.y_SI_pts[1:])*.5
    fdtd.z_SI_cel[:-1] = (fdtd.z_SI_pts[:-1] + fdtd.z_SI_pts[1:])*.5
    fdtd.x_SI_cel[ -1] =  fdtd.x_SI_cel[ -2] + fdtd.space_grid[0][-1]
    fdtd.y_SI_cel[ -1] =  fdtd.y_SI_cel[ -2] + fdtd.space_grid[1][-1]
    fdtd.z_SI_cel[ -1] =  fdtd.z_SI_cel[ -2] + fdtd.space_grid[2][-1]

    fdtd.x = fdtd.x_SI_cel
    fdtd.y = fdtd.y_SI_cel
    fdtd.z = fdtd.z_SI_cel

    fdtd.ex.x = fdtd.x_SI_cel; fdtd.ex.y = fdtd.y_SI_pts; fdtd.ex.z = fdtd.z_SI_pts;
    fdtd.ey.x = fdtd.x_SI_pts; fdtd.ey.y = fdtd.y_SI_cel; fdtd.ey.z = fdtd.z_SI_pts;
    fdtd.ez.x = fdtd.x_SI_pts; fdtd.ez.y = fdtd.y_SI_pts; fdtd.ez.z = fdtd.z_SI_cel;

    fdtd.hx.x = fdtd.x_SI_pts; fdtd.hx.y = fdtd.y_SI_cel; fdtd.hx.z = fdtd.z_SI_cel;
    fdtd.hy.x = fdtd.x_SI_cel; fdtd.hy.y = fdtd.y_SI_pts; fdtd.hy.z = fdtd.z_SI_cel;
    fdtd.hz.x = fdtd.x_SI_cel; fdtd.hz.y = fdtd.y_SI_cel; fdtd.hz.z = fdtd.z_SI_pts;

def divide_array(arr, region, num, divided, x_strt_group, x_stop_group, nx_group):
    if region is None:
        return None
    else:
        check_type('region', region, (tuple, list), (tuple, list))
        if len(region) > 2:
            raise ValueError, 'region must contain no more than 2 tuple or list'
        for i in xrange(len(region)):
            check_type('region[%d]' % i, region[i], (tuple, list), int_types)

        x_strt = x_strt_group[num]
        x_stop = x_stop_group[num]
        num_nx =     nx_group[num]

        num_x = len(x_strt_group)
        regions = []
        for i in xrange(num_x):
            x_strt_node = x_strt_group[i]
            x_stop_node = x_stop_group[i]
            if   region[0][0] == 0:
                i_strt = 0
            elif region[0][0] >  x_strt_node and region[0][0] <= x_stop_node:
                i_strt = i
            if   region[1][0] >= x_strt_node and region[1][0] <= x_stop_node:
                i_stop = i

        if   num < i_strt:
            return None
        elif num > i_stop:
            return None
        else:
            if divided:
                if   num == i_strt and num == i_stop:
                    arr_x_strt = None
                    arr_x_stop = None
                    slcs = [slice(arr_x_strt, arr_x_stop  , None)]
                elif num == i_strt:
                    arr_x_strt = None
                    arr_x_stop = x_stop - region[0][0]
                    slcs = [slice(arr_x_strt, arr_x_stop+1, None)]
                elif num == i_stop:
                    arr_x_strt = x_strt - region[0][0]
                    arr_x_stop = None
                    slcs = [slice(arr_x_strt, arr_x_stop  , None)]
                else:
                    arr_x_strt = x_strt - region[0][0]
                    arr_x_stop = x_stop - region[0][0]
                    slcs = [slice(arr_x_strt, arr_x_stop+1, None)]
            else:
                slcs = [slice(None, None, None)]
            slc  = slice(None,None,None)
            for i in xrange(arr.ndim-1):
                slcs.append(slc)
            return arr.__getitem__(tuple(slcs))

def divide_region(region, x_strt_group, x_stop_group, nx_group):
    if region is None:
        regions = [None for i in xrange(len(x_strt_group))]
        return regions
    else:
        check_type('region', region, (tuple, list), (tuple, list))
        if len(region) > 2:
            raise ValueError, 'region must contain no more than 2 tuple or list'
        for i in xrange(len(region)):
            check_type('region[%d]' % i, region[i], (tuple, list), int_types)

        num_x = len(x_strt_group)
        regions = []
        for i in xrange(num_x):
            x_strt_node = x_strt_group[i]
            x_stop_node = x_stop_group[i]
            if   region[0][0] == 0:
                i_strt = 0
            elif region[0][0] >  x_strt_node and region[0][0] <= x_stop_node:
                i_strt = i
            if   region[1][0] >= x_strt_node and region[1][0] <= x_stop_node:
                i_stop = i

        for i in xrange(num_x):
            x_strt_node = x_strt_group[i]
            x_stop_node = x_stop_group[i]
            sub_nx      =     nx_group[i]
            if i >= i_strt and i <= i_stop:
                if   i == i_strt and i == i_stop:
                    x_strt = region[0][0] - x_strt_node
                    x_stop = region[1][0] - x_strt_node
                elif i == i_strt:
                    x_strt = region[0][0] - x_strt_node
                    x_stop =                 sub_nx - 1
                elif i == i_stop:
                    x_strt =                          0
                    x_stop = region[1][0] - x_strt_node
                else:
                    x_strt =                          0
                    x_stop =                 sub_nx - 1

                regions.append(((x_strt, region[0][1], region[0][2]),(x_stop, region[1][1], region[1][2])))
            else:
                regions.append(None)
        return regions

def correct_kvec(k_prev, s, ds, nl, kvec, iter_n):
    a = 0.5*ds*kvec[0]
    b = 0.5*ds*kvec[1]
    c = 0.5*ds*kvec[2]
    d = (1./(s**2))*(np.sin(np.pi*s/nl)**2)
    sin2 = np.sin(a*k_prev)**2 + np.sin(b*k_prev)**2 + np.sin(c*k_prev)**2
    res = (sin2 - d)/(a*np.sin(2*a*k_prev)+b*np.sin(2*b*k_prev)+c*np.sin(2*c*k_prev))
    k_temp = k_prev - res
    if iter_n == 0:
        return k_temp 
    else:
        return correct_kvec(k_temp, s, ds, nl, kvec, iter_n-1)

def correct_phase_velocity_tfsf1d(k0, s, ds, nl, kvec, iter_n):
# return 1/vp (rvp)
    return k0/correct_kvec(k0, s, ds, nl, kvec, iter_n)

def field_coordinate(fdtd, coord, field_name):
    check_type('coord', coord, tuple, int_types)
    check_value('field_name', field_name, ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', \
                                           'ex', 'ey', 'ez', 'hx', 'hy', 'hz') )
    if   '2D' in fdtd.mode:
        i, j = coord
        if field_name in ['Ex', 'ex']:
            return fdtd.x_SI_cel[i], fdtd.y_SI_pts[j]
        if field_name in ['Ey', 'ey']:
            return fdtd.x_SI_pts[i], fdtd.y_SI_cel[j]
        if field_name in ['Ez', 'ez']:
            return fdtd.x_SI_pts[i], fdtd.y_SI_pts[j]
        if field_name in ['Hx', 'hx']:
            return fdtd.x_SI_pts[i], fdtd.y_SI_cel[j]
        if field_name in ['Hy', 'hy']:
            return fdtd.x_SI_cel[i], fdtd.y_SI_pts[j]
        if field_name in ['Hz', 'hz']:
            return fdtd.x_SI_cel[i], fdtd.y_SI_cel[j]
    elif '3D' in fdtd.mode:
        i, j, k = coord
        if field_name in ['Ex', 'ex']:
            return fdtd.x_SI_cel[i], fdtd.y_SI_pts[j], fdtd.z_SI_pts[k]
        if field_name in ['Ey', 'ey']:
            return fdtd.x_SI_pts[i], fdtd.y_SI_cel[j], fdtd.z_SI_pts[k]
        if field_name in ['Ez', 'ez']:
            return fdtd.x_SI_pts[i], fdtd.y_SI_pts[j], fdtd.z_SI_cel[k]
        if field_name in ['Hx', 'hx']:
            return fdtd.x_SI_pts[i], fdtd.y_SI_cel[j], fdtd.z_SI_cel[k]
        if field_name in ['Hy', 'hy']:
            return fdtd.x_SI_cel[i], fdtd.y_SI_pts[j], fdtd.z_SI_cel[k]
        if field_name in ['Hz', 'hz']:
            return fdtd.x_SI_cel[i], fdtd.y_SI_cel[j], fdtd.z_SI_pts[k]

def symmetric_central_dense_grid(central_length, total_length, min_ds, max_ds, max_ratio):
    n_center = int(central_length/min_ds)
    n_float  = sc.logn(max_ratio, max_ds/min_ds)
    n_geo    = int(n_float+1)
    ratio    = (max_ds/min_ds)**(1./n_geo)

    inc_geo_ds = np.array([min_ds*ratio**idx for idx in xrange(n_geo)])
    dec_geo_ds = inc_geo_ds[::-1]

    geo_length = min_ds*(ratio**n_geo - 1)/(ratio - 1)

    res_length = 0.5*(total_length - central_length) - geo_length

    n_res = int(res_length/max_ds + 1)

    n_total = n_center + 2*(n_geo + n_res)

    grid = np.ones(n_total, dtype=np.float64)*min_ds

    indices = np.array([0, n_res, n_geo, n_center, n_geo, n_res]).cumsum()

    grid[indices[0]:indices[1]] = max_ds
    grid[indices[1]:indices[2]] = dec_geo_ds
    grid[indices[2]:indices[3]] = min_ds
    grid[indices[3]:indices[4]] = inc_geo_ds
    grid[indices[4]:indices[5]] = max_ds

    return grid

def remain_time(tmax, tstep, time_per_tstep):
    return (tmax-tstep)*time_per_tstep

class Timer:
    def __init__(self):
        self.init()

    def record(self):
        self.time_start = self.time_end
        self.time_end = dtm.now()
        self.num_record += 1
        if self.num_record != 1:
            self.time_intervals.append(self.time_end-self.time_start)

    def record_start(self):
        self.time_start = dtm.now()
        if self.recording == True:
            raise ValueError, 'recording is not ended'
        else:
            self.recording = True

    def record_end(self):
        self.time_end = dtm.now()
        self.num_record += 1
        if self.recording == False:
            raise ValueError, 'recording is not started'
        else:
            self.recording = False
        self.time_intervals.append(self.time_end-self.time_start)

    def mean(self):
        total_interval = self.time_intervals[0]
        for i in xrange(1, self.num_record):
            total_interval += self.time_intervals[i]
        return total_interval/self.num_record

    def init(self):
        self.time_intervals = []
        self.time_start = None
        self.time_end   = None
        self.num_record = 0
        self.recording  = False



if __name__ == '__main__':
    pass
