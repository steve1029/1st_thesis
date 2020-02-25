# Author  : Ki-Hwan Kim
# Purpose : Update function for PML (Perfectly Matched Layer)
# Target  : Node
# Created : 2012-02-22
# Modified: 

import numpy as np

from kemp.fdtd3d.util import common
from kemp.fdtd3d.node import Fields


class Pml:
    def __init__(self, node_fields, directions, npml=10, sigma_max=4, kappa_max=1, alpha_max=0, m_sigma=3, m_alpha=1):
        """
        """

        common.check_type('node_fields', node_fields, Fields)
        common.check_type('directions', directions, (list, tuple), str)
        common.check_type('npml', npml, int)
        common.check_type('sigma_max', sigma_max, (int, float))
        common.check_type('kappa_max', kappa_max, (int, float))
        common.check_type('alpha_max', alpha_max, (int, float))
        common.check_type('m_sigma', m_sigma, (int, float))
        common.check_type('m_alpha', m_alpha, (int, float))

        assert len(directions) == 3
        for axis in directions:
            assert axis in ['+', '-', '+-', '']

        # local variables
        nodef = node_fields
        mainf_list = nodef.mainf_list
        buffer_dict = nodef.buffer_dict
        
        pml_x = directions[0]
        pml_y = directions[1]
        pml_z = directions[2]
        pml_params = (npml, sigma_max, kappa_max, alpha_max, m_sigma, m_alpha)

        device_type_list = [f.device_type for f in nodef.updatef_list]
        if 'gpu' in device_type_list:
            from kemp.fdtd3d import gpu
            self.gpu = gpu
        if 'cpu' in device_type_list:
            from kemp.fdtd3d import cpu
            self.cpu = cpu

        # create Pml instances
        if len(mainf_list) == 1:
            f = mainf_list[0]
            getattr(self, f.device_type).Pml(f, directions, *pml_params)
            
        elif len(mainf_list) > 1:
            if '-' in pml_x:
                f = mainf_list[0]
                getattr(self, f.device_type).Pml(f, ('-', pml_y, pml_z), *pml_params)
                
            if '+' in pml_x:
                f = mainf_list[-1]
                getattr(self, f.device_type).Pml(f, ('+', pml_y, pml_z), *pml_params)

            if pml_x == '':
                for f in (mainf_list[0], mainf_list[-1]):
                    getattr(self, f.device_type).Pml(f, ('', pml_y, pml_z), *pml_params)

            if not (pml_y == '' and pml_z == ''):
                for f in mainf_list[1:-1]:
                    getattr(self, f.device_type).Pml(f, ('', pml_y, pml_z), *pml_params)
            
        # for buffer fields
        for direction, buf in buffer_dict.items():
            if 'x' in direction and not (pml_y == '' and pml_z == ''):
                cpu.Pml(buf, ('', pml_y, pml_z), *pml_params)
            elif 'y' in direction and not (pml_x == '' and pml_z == ''):
                cpu.Pml(buf, (pml_x, '', pml_z), *pml_params)
            elif 'z' in direction and not (pml_x == '' and pml_y == ''):
                cpu.Pml(buf, (pml_x, pml_y, ''), *pml_params)
