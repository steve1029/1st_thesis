# Author  : Ki-Hwan Kim
# Purpose : Setup update functions for the core
# Target  : Node
# Created : 2012-02-22
# Modified: 

from kemp.fdtd3d.util import common
from fields import Fields


class Core:
    def __init__(self, node_fields):

        common.check_type('node_fields', node_fields, Fields)

        # create Core instances
        f_list = node_fields.updatef_list

        device_type_list = [f.device_type for f in f_list]
        if 'gpu' in device_type_list:
            from kemp.fdtd3d import gpu
            self.gpu = gpu
        if 'cpu' in device_type_list:
            from kemp.fdtd3d import cpu
            self.cpu = cpu

        for f in f_list:
            getattr(self, f.device_type).Core(f)
