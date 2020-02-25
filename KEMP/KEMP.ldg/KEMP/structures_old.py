from __future__ import division
from mainfdtd import *
import numpy as np
import os

from util import *
from ndarray import Fields

_manual_2d = """
-*-  class instance -*-
-*- 2D strucrures -*-

structures.Polygon_2d(MATERIAL, coordinate)
    MATERIAL : FDTD.materials.{Dielectric, Dimagnetic, Dielectromagnetic}
               or FDTD.Dispersive.{Gold,Silver,Aluminium ...}
    coordinate[(n,2) shaped Tuple] : coordinates of Polygon edges
    n: number of Polygon edges

structures.Rectangle(MATERIAL, coordinate[(2,2) shaped Tuple])
structures.Ellipse(MATERIAL,
"""

def manual():
    print _manual_2d

def rotation_matrix(axis_vec, angle):
    axis_vec = np.array(axis_vec)
    mag = ((axis_vec)**2).sum()
    if mag == 0:
        raise ValueError, 'Invaild rotation axis vector'
    if mag != 1:
        axis_vec /= mag
    ang_rad = angle
    sin = np.sin(-ang_rad)
    cos = np.cos(-ang_rad)
    ux  = axis_vec[0]
    uy  = axis_vec[1]
    uz  = axis_vec[2]
    mat = np.array([cos+(ux**2)*(1-cos), ux*uy*(1-cos)-uz*sin, uz*ux*(1-cos)+uy*sin, \
                    ux*uy*(1-cos)+uz*sin, cos+(uy**2)*(1-cos), uy*uz*(1-cos)-ux*sin, \
                    uz*ux*(1-cos)-uy*sin, uy*uz*(1-cos)+ux*sin, cos+(uz**2)*(1-cos)], \
                    dtype=np.float64)
    return mat

class Structure:
    def __init__(self):
        self.group = 'structure'
        self.code_prev = [', __COEFFICIENT_FIELDS__', ', __MARK__', '__SET_MARK__', '__SET_MATERIAL__']
        self.code_list = {'dielectric':[', __GLOBAL__ __FLOAT__* ce1s, __GLOBAL__ __FLOAT__* ce2s', \
                                        '', \
                                        '', \
                                        'set_ces(idx, material_params, \
                                                 ce1s, ce2s)'], \
                          'dimagnetic':[', __GLOBAL__ __FLOAT__* ch1s, __GLOBAL__ __FLOAT__* ch2s', \
                                        '', \
                                        '', \
                                        'set_chs(idx, material_params, \
                                                 ch1s, ch2s)'], \
                          'dielectromagnetic':[', __GLOBAL__ __FLOAT__* ce1s, __GLOBAL__ __FLOAT__* ce2s \
                                                , __GLOBAL__ __FLOAT__* ch1s, __GLOBAL__ __FLOAT__* ch2s', \
                                               '', \
                                               '', \
                                               'set_cehs(idx, material_params, \
                                                         ce1s, ce2s, ch1s, ch2s)'], \
                          'electric dispersive':[', __GLOBAL__ __FLOAT__* ce1s, __GLOBAL__ __FLOAT__* ce2s \
                                                  , __GLOBAL__ int* mark', \
                                                 ', __GLOBAL__ int* mark', \
                                                 'mark[idx] = mrk',
                                                 'set_ces(idx, material_params, \
                                                  ce1s, ce2s, mark)'], \
                          'magnetic dispersive':[', __GLOBAL__ __FLOAT__* ch1s, __GLOBAL__ __FLOAT__* ch2s \
                                                  , __GLOBAL__ int* mark', \
                                                 ', __GLOBAL__ int* mark', \
                                                 'mark[idx] = mrk',
                                                 'set_chs(idx, material_params, \
                                                  ch1s, ch2s, mark)'], \
                          'electromagnetic dispersive':[', __GLOBAL__ __FLOAT__* ce1s, __GLOBAL__ __FLOAT__* ce2s \
                                                         , __GLOBAL__ __FLOAT__* ch1s, __GLOBAL__ __FLOAT__* ch2s \
                                                         , __GLOBAL__ int* mark', \
                                                        ', __GLOBAL__ int* mark', \
                                                        'mark[idx] = mrk',
                                                        'set_cehs(idx, material_params, \
                                                         ce1s, ce2s, ch1s, ch2s, mark)']}
        self.type_to_num = {'dielectric':0, 'dimagnetic':1, 'dielectromagnetic':2, \
                            'electric dispersive':3, 'magnetic dispersive':4, 'electromagnetic dispersive':5}

    def set_fdtd(self, fdtd):
        self.fdtd = fdtd
        self.coefs_lists   = self.coefs_list(fdtd)
        self.grids_lists   = self.grids_list(fdtd)
        self.material_type = self.material.classification
#        self.material_num  = self.type_to_num[fdtd.materials_classification]
#        self.code_post     = self.code_list[fdtd.materials_classification]

#        template = fdtd.engine.templates['structures']
#        # OpenMP settings
#        if 'cpu' in fdtd.engine.name:
#            self.code_prev += ['__COEFFICIENTS__']
#            self.code_post += ['ce1s, ce2s']
#        code     = template_to_code(template, self.code_prev+fdtd.engine.code_prev, self.code_post+fdtd.engine.code_post)
#        prg = fdtd.engine.build(code)

    def coefs_list(self, fdtd):
        if   fdtd.materials_classification == 'dielectric':
            fields_lists = [[fdtd.ce1x.data, fdtd.ce2x.data], \
                            [fdtd.ce1y.data, fdtd.ce2y.data], \
                            [fdtd.ce1z.data, fdtd.ce2z.data]  ]
        elif fdtd.materials_classification == 'electric dispersive':
            fields_lists = [[fdtd.ce1x.data, fdtd.ce2x.data, fdtd.mkex.data], \
                            [fdtd.ce1y.data, fdtd.ce2y.data, fdtd.mkey.data], \
                            [fdtd.ce1z.data, fdtd.ce2z.data, fdtd.mkez.data]  ]
        elif fdtd.materials_classification == 'dimagnetic':
            fields_lists = [[fdtd.ch1x.data, fdtd.ch2x.data], \
                            [fdtd.ch1y.data, fdtd.ch2y.data], \
                            [fdtd.ch1z.data, fdtd.ch2z.data]  ]
        elif fdtd.materials_classification == 'magnetic dispersive':
            fields_lists = [[fdtd.ch1x.data, fdtd.ch2x.data, fdtd.mkhx.data], \
                            [fdtd.ch1y.data, fdtd.ch2y.data, fdtd.mkhy.data], \
                            [fdtd.ch1z.data, fdtd.ch2z.data, fdtd.mkhz.data]  ]
        elif fdtd.materials_classification == 'dielectromagnetic':
            fields_lists = [[fdtd.ce1x.data, fdtd.ce2x.data, fdtd.ch1x.data, fdtd.ch2x.data], \
                            [fdtd.ce1y.data, fdtd.ce2y.data, fdtd.ch1y.data, fdtd.ch2y.data], \
                            [fdtd.ce1z.data, fdtd.ce2z.data, fdtd.ch1z.data, fdtd.ch2z.data]  ]
        elif fdtd.materials_classification == 'electromagnetic dispersive':
            fields_lists = [[fdtd.ce1x.data, fdtd.ce2x.data, fdtd.ch1x.data, fdtd.ch2x.data, fdtd.mkex.data, fdtd.mkhx.data], \
                            [fdtd.ce1y.data, fdtd.ce2y.data, fdtd.ch1y.data, fdtd.ch2y.data, fdtd.mkey.data, fdtd.mkhy.data], \
                            [fdtd.ce1z.data, fdtd.ce2z.data, fdtd.ch1z.data, fdtd.ch2z.data, fdtd.mkez.data, fdtd.mkhz.data]  ]
        return fields_lists

    def grids_list(self, fdtd):
        if   '2D' in fdtd.mode:
            grids_lists = [[fdtd.x_cel_NU.data, fdtd.y_pts_NU.data], \
                           [fdtd.x_pts_NU.data, fdtd.y_cel_NU.data], \
                           [fdtd.x_pts_NU.data, fdtd.y_pts_NU.data]  ]
        elif '3D' in fdtd.mode:
            grids_lists = [[fdtd.x_cel_NU.data, fdtd.y_pts_NU.data, fdtd.z_pts_NU.data], \
                           [fdtd.x_pts_NU.data, fdtd.y_cel_NU.data, fdtd.z_pts_NU.data], \
                           [fdtd.x_pts_NU.data, fdtd.y_pts_NU.data, fdtd.z_cel_NU.data]  ]
        return grids_lists

# 2D structures
class Polygon_2d(Structure):
    def __init__(self, material, poly, rotate=0., origin=None):
        Structure.__init__(self)
        self.name = 'polygon'
        self.poly = np.array(poly, dtype=np.float64)
        self.np   = len(self.poly)
        self.rotate = rotate
        self.sin    = np.sin(rotate)
        self.cos    = np.cos(rotate)
        self.com    = np.zeros(2, dtype=np.float64)
        self.material = material

        for i in xrange(self.np):
            self.com[0] += self.poly[i][0]/self.np
            self.com[1] += self.poly[i][1]/self.np
        if origin==None: self.origin = self.com
        else           : self.origin = origin

    # Determine if a point is inside a given polygon or not
    # Polygon is a list of (x,y) pairs. This fuction
    # returns True or False.  The algorithm is called
    # "Ray Casting Method".
    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg    = fdtd._structure_prg
        sin    = fdtd.dtype(self.sin)
        cos    = fdtd.dtype(self.cos)
        poly   = Fields(fdtd, (self.np, 2), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.poly  /fdtd.min_ds)
        origin = Fields(fdtd, (      2,  ), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.origin/fdtd.min_ds)

        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    poly.data, \
                    self.grids_lists[i][0], self.grids_lists[i][1], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(self.np), \
                    origin.data, \
                    sin, cos, \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt  = prg.set_struc_poly(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_poly')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_poly, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del poly, origin, material_params
        return evts
        '''
        poly = self.poly
        n = len(poly)
        p1x,p1y = poly[0]
        inside = False
        for i in xrange(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y) and y<= max(p1y,p2y) and x < max(p1x, p2x):
                sign = (p2y-p1y)*((y-p1y)*(p2x-p1x) - (x-p1x)*(p2y-p1y))
                if p1x == p2x or sign >= 0.: inside = not inside
            p1x,p1y = p2x,p2y
        return inside
        '''

class Rectangle(Structure):
    def __init__(self, material, coord):
        Structure.__init__(self)
        point_ld, point_ru = coord # left_down, right_up
        self.point_ld = np.array(point_ld)
        self.point_ru = np.array(point_ru)
        self.points = np.zeros(4, dtype=np.float64)
        self.points[:2] = point_ld[:]
        self.points[2:] = point_ru[:]
        self.material = material

    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg    = fdtd._structure_prg
        points   = Fields(fdtd, (4,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.points/fdtd.min_ds)
        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    points.data, \
                    self.grids_lists[i][0], self.grids_lists[i][1], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt  = prg.set_struc_rect(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_rect')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_rect, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del points, material_params
        return evts

class Ellipse(Structure):
    def __init__(self, material, com, radius, rotate=0., origin=None):
        Structure.__init__(self)
        self.name = 'ellipse'
        self.rotate = rotate
        self.sin    = np.sin(rotate)
        self.cos    = np.cos(rotate)
        self.com    = np.array(com, dtype=np.float64)
        self.radius = np.array(radius, dtype=np.float64)
        self.material = material

        if origin==None: self.origin = self.com
        else           : self.origin = origin


    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg    = fdtd._structure_prg
        sin    = fdtd.dtype(self.sin)
        cos    = fdtd.dtype(self.cos)
        com    = Fields(fdtd, (2,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.com   /fdtd.min_ds)
        radius = Fields(fdtd, (2,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.radius/fdtd.min_ds)
        origin = Fields(fdtd, (2,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.origin/fdtd.min_ds)

        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    com.data, radius.data, \
                    self.grids_lists[i][0], self.grids_lists[i][1], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), \
                    origin.data, \
                    sin, cos, \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt  = prg.set_struc_ellp2d(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_ellp2d')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_ellp2d, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del com, radius, origin, material_params
        return evts

class Circle(Ellipse):
    def __init__(self, material, com, radius, rotate=0., origin=None):
        Ellipse.__init__(self, material, com, (radius, radius), rotate, origin)

class Regular_polygon(Polygon_2d):
    def __init__(self, material, point_ct, radius, n, rotate=0.):
        poly = []
        self.x0, self.y0 = point_ct
        self.radius = radius
        self.angle = 2.*np.pi/n
        for i in xrange(n):
            x = self.x0 + self.radius*np.cos(self.angle*i + np.pi/2.)
            y = self.y0 + self.radius*np.sin(self.angle*i + np.pi/2.)
            poly.append(np.array([x, y]))
        poly = np.array(poly)
        Polygon_2d.__init__(self, material, poly, rotate)

# 3D structures
class Ellipsoid(Structure):
    def __init__(self, material, com, radius, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Structure.__init__(self)
        self.name    = 'ellipsoid'
        self.com     = np.array(com, dtype=np.float64)
        self.radius  = np.array(radius, dtype=np.float64)
        self.rot_mat = rotation_matrix(rot_axis, angle)
        if origin==None: self.origin = self.com
        else           : self.origin = origin
        self.material = material

    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg    = fdtd._structure_prg
        com     = Fields(fdtd, (3,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.com   /fdtd.min_ds)
        radius  = Fields(fdtd, (3,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.radius/fdtd.min_ds)
        origin  = Fields(fdtd, (3,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.origin/fdtd.min_ds)
        rot_mat = Fields(fdtd, (9,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.rot_mat           )

        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    com.data, radius.data, \
                    self.grids_lists[i][0], self.grids_lists[i][1], self.grids_lists[i][2], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                    origin.data, rot_mat.data, \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt = prg.set_struc_ellp3d(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_ellp3d')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_ellp3d, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del com, radius, origin, rot_mat, material_params
        return evts

class Sphere(Ellipsoid):
    def __init__(self, material, com, radius, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Ellipsoid.__init__(self, material, com, (radius, radius, radius), rot_axis, angle, origin)

class Elliptic_cylinder(Structure):
    def __init__(self, material, base_com, radius, height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Structure.__init__(self)
        self.name = 'elliptic_cylinder'
        self.base_com = np.array(base_com, dtype=np.float64)
        self.radius   = np.array(radius, dtype=np.float64)
        self.rot_mat  = rotation_matrix(rot_axis, angle)
        self.height   = height
        axtonum       = {'x':0, 'y':1, 'z':2}
        self.axis     = axtonum[axis]
        self.com      = np.array(base_com, dtype=np.float64)
        self.com[axtonum[axis]] += np.float64(height*.5)
        if origin==None: self.origin = self.com
        else           : self.origin = origin
        self.material = material

    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg    = fdtd._structure_prg
        height  = comp_to_real(fdtd.dtype)(self.height/fdtd.min_ds)
        com     = Fields(fdtd, (3,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.com   /fdtd.min_ds)
        radius  = Fields(fdtd, (2,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.radius/fdtd.min_ds)
        origin  = Fields(fdtd, (3,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.origin/fdtd.min_ds)
        rot_mat = Fields(fdtd, (9,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.rot_mat           )

        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    com.data, radius.data, height, np.int32(self.axis), \
                    self.grids_lists[i][0], self.grids_lists[i][1], self.grids_lists[i][2], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                    origin.data, rot_mat.data, \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt  = prg.set_struc_elcd(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_elcd')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_elcd, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del com, radius, origin, rot_mat, material_params
        return evts

class Cylinder(Elliptic_cylinder):
    def __init__(self, material, base_com, radius, height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Elliptic_cylinder.__init__(self, material, base_com, (radius, radius), height, axis, rot_axis, angle, origin)

Circular_cylinder = Cylinder

class Polyprism(Structure):
    def __init__(self, material, base_poly, height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Structure.__init__(self)
        self.name = 'polyprism'
        self.base_poly = np.array(base_poly, dtype=np.float64)
        self.np        = len(self.base_poly)
        self.height    = height
        axtonum        = {'x':0, 'y':1, 'z':2}
        self.axis      = axtonum[axis]
        self.com       = np.zeros(3, dtype=np.float64)
        self.rot_mat   = rotation_matrix(rot_axis, angle)
        for i in xrange(self.np):
            self.com[0] += self.base_poly[i][0]/self.np
            self.com[1] += self.base_poly[i][1]/self.np
            self.com[2] += self.base_poly[i][2]/self.np
        self.com[axtonum[axis]] += np.float64(self.height*.5)
        if origin==None: self.origin = self.com
        else           : self.origin = origin
        self.material = material

    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg       = fdtd._structure_prg
        height    = comp_to_real(fdtd.dtype)(self.height/fdtd.min_ds)
        base_poly = Fields(fdtd, (self.np, 3), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=np.round(self.base_poly/fdtd.min_ds,1))
        origin    = Fields(fdtd, (      3,  ), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=np.round(self.origin   /fdtd.min_ds,1))
        rot_mat   = Fields(fdtd, (      9,  ), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.rot_mat                          )

        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    base_poly.data, height, np.int32(self.np), np.int32(self.axis), \
                    self.grids_lists[i][0], self.grids_lists[i][1], self.grids_lists[i][2], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                    origin.data, rot_mat.data, \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt  = prg.set_struc_plpm(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_plpm')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_plpm, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del base_poly, origin, rot_mat, material_params
        return evts

class Regular_polyprism(Polyprism):
    def __init__(self, material, base_com, radius, n, height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        poly = []
        self.x0, self.y0, self.z0 = base_com
        self.radius = radius
        self.angle = 2.*np.pi/n
        for i in xrange(n):
            if axis == 'x':
                y = self.y0 + self.radius*np.cos(self.angle*i + np.pi/2.)
                z = self.z0 + self.radius*np.sin(self.angle*i + np.pi/2.)
                poly.append(np.array([self.x0, y, z]))
            if axis == 'y':
                z = self.z0 + self.radius*np.cos(self.angle*i + np.pi/2.)
                x = self.x0 + self.radius*np.sin(self.angle*i + np.pi/2.)
                poly.append(np.array([x, self.y0, z]))
            if axis == 'z':
                x = self.x0 + self.radius*np.cos(self.angle*i + np.pi/2.)
                y = self.y0 + self.radius*np.sin(self.angle*i + np.pi/2.)
                poly.append(np.array([x, y, self.z0]))
        poly = np.array(poly, dtype=np.float64)
        Polyprism.__init__(self, material, poly, height, axis, rot_axis, angle, origin)

class Box(Structure):
    def __init__(self, material, coord):
        Structure.__init__(self)
        point_ld, point_ru = coord # left_down, right_up
        self.point_ld = np.array(point_ld)
        self.point_ru = np.array(point_ru)
        self.points = np.zeros(6, dtype=np.float64)
        self.points[:3] = point_ld[:]
        self.points[3:] = point_ru[:]
        self.material = material

    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg      = fdtd._structure_prg
        points   = Fields(fdtd, (6,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.points/fdtd.min_ds)

        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    points.data, \
                    self.grids_lists[i][0], self.grids_lists[i][1], self.grids_lists[i][2], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt  = prg.set_struc_boxs(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_boxs')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_boxs, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del points, material_params
        return evts

class Elliptic_cone(Structure):
    def __init__(self, material, base_com, radius, height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Structure.__init__(self)
        self.name = 'elliptic_cone'
        self.base_com = np.array(base_com, dtype=np.float64)
        self.radius   = np.array(radius, dtype=np.float64)
        self.height   = height
        axtonum       = {'x':0, 'y':1, 'z':2}
        self.axis     = axtonum[axis]
        self.com      = np.array(base_com, dtype=np.float64)
        self.com[axtonum[axis]] += np.float64(height*1./3.)
        self.rot_mat  = rotation_matrix(rot_axis, angle)
        if origin==None: self.origin = self.com
        else           : self.origin = origin
        self.material = material

    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg     = fdtd._structure_prg
        height  = comp_to_real(fdtd.dtype)(self.height/fdtd.min_ds)
        com     = Fields(fdtd, (3,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.com   /fdtd.min_ds)
        radius  = Fields(fdtd, (2,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.radius/fdtd.min_ds)
        origin  = Fields(fdtd, (3,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.origin/fdtd.min_ds)
        rot_mat = Fields(fdtd, (9,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.rot_mat           )

        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    com.data, radius.data, height, np.int32(self.axis), \
                    self.grids_lists[i][0], self.grids_lists[i][1], self.grids_lists[i][2], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                    origin.data, rot_mat.data, \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt  = prg.set_struc_elcn(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_elcn')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_elcn, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del com, radius, origin, rot_mat, material_params
        return evts

class Circular_cone(Elliptic_cone):
    def __init__(self, material, base_com, radius, height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Elliptic_cone.__init__(self, material, base_com, (radius, radius), height, axis, rot_axis, angle, origin)

class Elliptic_truncated_cone(Structure):
    def __init__(self, material, base_com, radius, height, truncated_height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Structure.__init__(self)
        self.name = 'elliptic_cone'
        self.base_com = np.array(base_com, dtype=np.float64)
        self.radius   = np.array(radius, dtype=np.float64)
        self.height   = height
        self.t_height = truncated_height
        axtonum       = {'x':0, 'y':1, 'z':2}
        self.axis     = axtonum[axis]
        self.com      = np.array(base_com, dtype=np.float64)
        self.com[axtonum[axis]] += np.float64(height*1./3.)
        self.rot_mat      = rotation_matrix(rot_axis, angle)
        if origin==None: self.origin = self.com
        else           : self.origin = origin
        self.material = material

    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg     = fdtd._structure_prg
        height  = comp_to_real(fdtd.dtype)(self.height  /fdtd.min_ds)
        t_height= comp_to_real(fdtd.dtype)(self.t_height/fdtd.min_ds)
        com     = Fields(fdtd, (3,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.com   /fdtd.min_ds)
        radius  = Fields(fdtd, (2,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.radius/fdtd.min_ds)
        origin  = Fields(fdtd, (3,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.origin/fdtd.min_ds)
        rot_mat = Fields(fdtd, (9,), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.rot_mat           )

        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    com.data, radius.data, height, t_height, np.int32(self.axis), \
                    self.grids_lists[i][0], self.grids_lists[i][1], self.grids_lists[i][2], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                    origin.data, rot_mat.data, \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt = prg.set_struc_eltrcn(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_eltrcn')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_eltrcn, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del com, radius, origin, rot_mat, material_params
        return evts

class Circular_truncated_cone(Elliptic_truncated_cone):
    def __init__(self, material, base_com, radius, height, truncated_height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Elliptic_truncated_cone.__init__(self, material, base_com, (radius, radius), height, truncated_height, axis, rot_axis, angle, origin)

class Polypyramid(Structure):
    def __init__(self, material, base_poly, height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        Structure.__init__(self)
        self.name = 'polypyramid'
        self.base_poly = np.array(base_poly, dtype=np.float64)
        self.np        = len(self.base_poly)
        self.height    = height
        axtonum        = {'x':0, 'y':1, 'z':2}
        self.axis      = axtonum[axis]
        self.com = np.zeros(3, dtype=np.float64)
        for i in xrange(self.np):
            self.com[0] += self.base_poly[i][0]/self.np
            self.com[1] += self.base_poly[i][1]/self.np
            self.com[2] += self.base_poly[i][2]/self.np
        self.com[axtonum[axis]] += np.float64(self.height*.5)
        self.rot_mat   = rotation_matrix(rot_axis, angle)
        if origin==None: self.origin = self.com
        else           : self.origin = origin
        self.material = material

    def set_structure(self, fdtd, wait=True):
        self.set_fdtd(fdtd)
        prg       = fdtd._structure_prg
        height    = comp_to_real(fdtd.dtype)(self.height/fdtd.min_ds)
        com       = Fields(fdtd, (      3,  ), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.com      /fdtd.min_ds)
        base_poly = Fields(fdtd, (self.np, 3), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.base_poly/fdtd.min_ds)
        origin    = Fields(fdtd, (      3,  ), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.origin   /fdtd.min_ds)
        rot_mat   = Fields(fdtd, (      9,  ), dtype=comp_to_real(fdtd.dtype), mem_flag='r', init_value=self.rot_mat  )

        evts = []
        for i in xrange(3):
            material_params = Fields(fdtd, (len(self.material.params[i]),), dtype=comp_to_real(fdtd.dtype), mem_flag='r', \
                                     init_value=np.array(self.material.params[i], dtype=comp_to_real(fdtd.dtype)))
            args = [fdtd.engine.queue, (fdtd.engine.gs,), (fdtd.engine.ls,), \
                    base_poly.data, com.data, height, np.int32(self.np), np.int32(self.axis), \
                    self.grids_lists[i][0], self.grids_lists[i][1], self.grids_lists[i][2], \
                    np.int32(fdtd.nx), np.int32(fdtd.ny), np.int32(fdtd.nz), \
                    origin.data, rot_mat.data, \
                    material_params.data] + self.coefs_lists[i]
            if 'opencl' in fdtd.engine.name:
                evt  = prg.set_struc_plpy(*args)
            elif 'cuda' in fdtd.engine.name:
                func = fdtd.engine.get_function(prg, 'set_struc_plpy')
                fdtd.engine.prepare(func, args)
                evt  = fdtd.engine.enqueue_kernel(func, args, False)
            elif  'cpu' in fdtd.engine.name:
                func = fdtd.engine.set_kernel(prg.set_struc_plpy, args)
                evt  = func(*(args[3:]))
            evts.append(evt)
        if wait: wait_for_events(fdtd, evts)
        del com, base_poly, origin, rot_mat, material_params
        return evts

class Regular_pyramid(Polypyramid):
    def __init__(self, material, base_com, radius, n, height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        poly = []
        self.x0, self.y0, self.z0 = base_com
        self.radius = radius
        self.angle = 2.*np.pi/n
        for i in xrange(n):
            if axis == 'x':
                y = self.y0 + self.radius*np.cos(self.angle*i + np.pi/2.)
                z = self.z0 + self.radius*np.sin(self.angle*i + np.pi/2.)
                poly.append(np.array([self.x0, y, z]))
            if axis == 'y':
                z = self.z0 + self.radius*np.cos(self.angle*i + np.pi/2.)
                x = self.x0 + self.radius*np.sin(self.angle*i + np.pi/2.)
                poly.append(np.array([x, self.y0, z]))
            if axis == 'z':
                x = self.x0 + self.radius*np.cos(self.angle*i + np.pi/2.)
                y = self.y0 + self.radius*np.sin(self.angle*i + np.pi/2.)
                poly.append(np.array([x, y, self.z0]))
        poly = np.array(poly, dtype=np.float64)
        Polypyramid.__init__(self, material, poly, height, axis, rot_axis, angle, origin)

class Rectangular_pyramid(Polypyramid):
    def __init__(self, material, coord, height, axis, rot_axis=(0.,0.,1.), angle=0., origin=None):
        point_ld, point_ru = coord # left_down, right_up
        pornt_ld = np.array(point_ld)
        pornt_ru = np.array(point_ru)
        p0 = np.array([point_ld[0], point_ld[1], point_ld[2]])
        p1 = np.array([point_ld[0], point_ru[1], point_ld[2]])
        p2 = np.array([point_ru[0], point_ru[1], point_ld[2]])
        p3 = np.array([point_ru[0], point_ld[1], point_ld[2]])
        poly     = np.array([p0, p1, p2, p3], dtype=np.float64)
        Polypyramid.__init__(self, material, poly, height, axis, rot_axis, angle, origin)

if __name__ == '__main__':
    pass
