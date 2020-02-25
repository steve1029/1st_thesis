# Author  : Ki-Hwan Kim
# Purpose : Update function for PBC (Periodic Boundary Condition) with normal incident source
# Target  : Node
# Created : 2012-02-22
# Modified: 

import numpy as np

from kemp.fdtd3d.util import common, common_buffer
from kemp.fdtd3d.node import Fields


class Pbc:
    def __init__(self, node_fields, axes):
        """
        """

        common.check_type('node_fields', node_fields, Fields)
        common.check_type('axes', axes, str)

        assert len( set(axes).intersection(set('xyz')) ) > 0, 'axes option is wrong: %s is given' % repr(axes)

        # local variables
        nodef = node_fields
        mainf_list = nodef.mainf_list
        buffer_dict = nodef.buffer_dict

        device_type_list = [f.device_type for f in nodef.updatef_list]
        if 'gpu' in device_type_list:
            from kemp.fdtd3d import gpu
            self.gpu = gpu
        if 'cpu' in device_type_list:
            from kemp.fdtd3d import cpu
            self.cpu = cpu

        # create Pbc instances
        if len(mainf_list) == 1:
            f0 = mainf_list[0]
            getattr(self, f0.device_type).Pbc(f0, axes)
            
        elif len(mainf_list) > 1:
            f0 = mainf_list[0]
            f1 = mainf_list[-1]

            if 'x' in axes:
                nx, ny, nz = f1.ns

                self.getf_e = getattr(self, f0.device_type).GetFields(f0, ['ey', 'ez'], (0, 0, 0), (0, -1, -1) )
                self.setf_e = getattr(self, f1.device_type).SetFields(f1, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

                self.getf_h = getattr(self, f1.device_type).GetFields(f1, ['hy', 'hz'], (-1, 0, 0), (-1, -1, -1) )
                self.setf_h = getattr(self, f0.device_type).SetFields(f0, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True )
                
                self.endf_is_cpu = True if f1.device_type == 'cpu' else False

                # append to the update list
                self.priority_type = 'pbc'
                nodef.append_instance(self)

            axs = axes.strip('x')
            if len( set(axs).intersection(set('yz')) ) > 0:
                for f in mainf_list:
                    getattr(self, f.device_type).Pbc(f, axs)

        # for buffer fields
        for direction, buf in buffer_dict.items():
            axs = axes.strip( direction[0] )
            if axs != '':
                cpu.Pbc(buf, axs)


    def update_e(self):
        if self.endf_is_cpu:
            self.getf_e.get_event().wait()
            self.setf_e.set_fields( self.getf_e.get_fields() )
        else:
            self.setf_e.set_fields(self.getf_e.get_fields(), [self.getf_e.get_event()])
            


    def update_h(self):
        self.setf_h.set_fields(self.getf_h.get_fields(), [self.getf_h.get_event()])
