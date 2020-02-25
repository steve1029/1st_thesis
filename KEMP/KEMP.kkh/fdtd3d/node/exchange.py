# Author  : Ki-Hwan Kim
# Purpose : Update function to exchange the boundaries
# Target  : Node
# Created : 2012-02-22
# Modified: 

import numpy as np
import unittest

from kemp.fdtd3d.util import common, common_exchange
from fields import Fields


class ExchangeNode:
    def __init__(self, node_fields):
        """
        """

        common.check_type('node_fields', node_fields, Fields)

        # local variables
        nodef = node_fields
        mainf_list = nodef.mainf_list

        # global variables
        device_type_list = [f.device_type for f in nodef.mainf_list]
        if 'gpu' in device_type_list:
            from kemp.fdtd3d import gpu
            self.gpu = gpu
        if 'cpu' in device_type_list:
            from kemp.fdtd3d import cpu
            self.cpu = cpu

        self.getf_dict = {'e': [], 'h': []}
        self.setf_dict = {'e': [], 'h': []}
        self.getf_block_dict = {'e': [], 'h': []}
        self.setf_block_dict = {'e': [], 'h': []}

        for f0, f1 in zip(mainf_list[:-1], mainf_list[1:]):
            for eh in ['e', 'h']:
                self.put_getf_setf_list(eh, f0, f1)

        # append to the update list
        self.priority_type = 'exchange'
        nodef.append_instance(self)


    def put_getf_setf_list(self, eh, f0, f1):
        gf, sf = (f1, f0) if eh == 'e' else (f0, f1)
        strfs = common_exchange.str_fs_dict['x'][eh]

        gpt0 = common_exchange.pt0_dict(*gf.ns)['x'][eh]['get']
        gpt1 = common_exchange.pt1_dict(*gf.ns)['x'][eh]['get']
        spt0 = common_exchange.pt0_dict(*sf.ns)['x'][eh]['set']
        spt1 = common_exchange.pt1_dict(*sf.ns)['x'][eh]['set']

        gtype = getattr(self, gf.device_type)
        stype = getattr(self, sf.device_type)
        getf = gtype.GetFields(gf, strfs, gpt0, gpt1)
        setf = stype.SetFields(sf, strfs, spt0, spt1, True)

        if gtype == 'cpu' and stype == 'gpu':
            self.getf_block_dict[eh].append(getf)
            self.setf_block_dict[eh].append(setf)
        else:
            self.getf_dict[eh].append(getf)
            self.setf_dict[eh].append(setf)
            

    def update(self, eh):
        setf_list = self.setf_dict[eh]
        getf_list = self.getf_dict[eh]
        for setf, getf in zip(setf_list, getf_list):
            setf.set_fields(getf.get_fields(), [getf.get_event()])

        setf_list = self.setf_block_dict[eh]
        getf_list = self.getf_block_dict[eh]
        for setf, getf in zip(setf_list, getf_list):
            getf.get_event().wait()
            setf.set_fields( getf.get_fields() )


    def update_e(self):
        self.update('e')


    def update_h(self):
        self.update('h')
