#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, create_out_fname, \
    write_csv, list_to_dict, read_csv_to_dict
from qm_utils.xyz_cluster2 import get_coordinates_xyz

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'xyz_cluster')
TS_DATA_DIR = os.path.join(SUB_DATA_DIR, 'TS_data')
PM3MM_DATA_DIR = os.path.join(SUB_DATA_DIR,'bxyl_data')

# Input files #
INPUT_XYZ_COORDS_OXANE = os.path.join(TS_DATA_DIR,'oxane-1c4-freeze_b3lyp-TS_b3lyp.xyz')
INPUT_XYZ_COORDS_BXYL  = os.path.join(PM3MM_DATA_DIR,'beta-xylose2-TS_pm3mm.xyz')

class TestXYZFunctions(unittest.TestCase):
    def testGetCoordinatesXYZ_OXANE(self):
        total_num_atoms, xyz_atoms, xyz_coords, atoms_ring_order, xyz_coords_ring, list_atoms = \
            get_coordinates_xyz('oxane-1c4-freeze_b3lyp-TS_b3lyp.xyz', TS_DATA_DIR, '5,0,1,2,3,4')

    def testGetCoordinatesXYZ_BXYL(self):
        total_num_atoms, xyz_atoms, xyz_coords, atoms_ring_order, xyz_coords_ring, list_atoms = \
            get_coordinates_xyz('beta-xylose2-TS_pm3mm.xyz', PM3MM_DATA_DIR, '1,5,8,9,13,17')

