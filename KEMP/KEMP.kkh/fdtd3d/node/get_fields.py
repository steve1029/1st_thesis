# Author  : Ki-Hwan Kim
# Purpose : Class to get the subdomains of fields
# Target  : Node
# Created : 2012-02-22
# Modified: 

import numpy as np

from kemp.fdtd3d.util import common

from fields import Fields


class GetFields:
    def __init__(self, node_fields, str_f, pt0, pt1, process=''):
        """
        """

        common.check_type('node_fields', node_fields, Fields)
        common.check_type('str_f', str_f, (str, list, tuple), str)
        common.check_type('pt0', pt0, (list, tuple), (int, float))
        common.check_type('pt1', pt1, (list, tuple), (int, float))

        pt0 = list( common.convert_indices(node_fields.ns, pt0) )
        pt1 = list( common.convert_indices(node_fields.ns, pt1) )
        
        # local variables
        nodef = node_fields
        str_fs = common.convert_to_tuple(str_f)
        mainf_list = nodef.mainf_list
        anx = nodef.accum_nx_list

        for strf in str_fs:
            strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
            common.check_value('str_f', strf, strf_list)

        for axis, n, p0, p1 in zip(['x', 'y', 'z'], nodef.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))

        # allocation
        shape = common.shape_two_points(pt0, pt1, len(str_fs))
        dummied_shape = common.shape_two_points(pt0, pt1, is_dummy=True)
        host_array = np.zeros(shape, dtype=nodef.dtype)

        split_host_array = np.split(host_array, len(str_fs))
        split_host_array_dict = dict( zip(str_fs, split_host_array) ) 

        device_type_list = [f.device_type for f in nodef.updatef_list]
        if 'gpu' in device_type_list:
            from kemp.fdtd3d import gpu
            getclass = getattr(gpu, 'GetFields')
        if 'cpu' in device_type_list:
            from kemp.fdtd3d import cpu
            getclass = getattr(cpu, 'GetFields')

        getf_list = []
        slices_list = []
        for i, mainf in enumerate(mainf_list):
            nx0 = anx[i]
            nx1 = anx[i+1]-1 if i < len(mainf_list)-1 else anx[i+1]
            overlap = common.overlap_two_lines((nx0, nx1), (pt0[0], pt1[0]))

            if overlap != None:
                x0, y0, z0 = pt0
                x1, y1, z1 = pt1

                slice_pt0 = (overlap[0]-x0, 0, 0)
                slice_pt1 = (overlap[1]-x0, y1-y0, z1-z0)
                slices = []
                for j, p0, p1 in zip([0, 1, 2], slice_pt0, slice_pt1):
                    if dummied_shape[j] != 1:
                        slices.append( slice(p0, p1+1) ) 
                slices_list.append(slices if slices!=[] else [slice(0, 1)] )

                local_pt0 = (overlap[0]-nx0, y0, z0)
                local_pt1 = (overlap[1]-nx0, y1, z1)
                getf_list.append( getclass(mainf, str_fs, local_pt0, local_pt1, process) )

        # global variables
        self.str_fs = str_fs
        self.host_array = host_array
        self.split_host_array_dict = split_host_array_dict
        self.getf_list = getf_list
        self.slices_list = slices_list


    def enqueue_kernel(self):
        for getf in self.getf_list:
            getf.enqueue_kernel()


    def wait(self, exec_kernel=True):
        event_list = [getf.get_event(exec_kernel) for getf in self.getf_list]

        for event in event_list:
            event.wait()


    def get_fields(self, strf=''):
        for getf, slices in zip(self.getf_list, self.slices_list):
            for str_f in self.str_fs:
                self.split_host_array_dict[str_f][slices] = getf.get_fields(str_f)

        if strf == '':
            return self.host_array
        else:
            return self.split_host_array_dict[strf]
