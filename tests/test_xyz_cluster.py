#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""

import unittest
import os
from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, warning
from qm_utils.xyz_cluster import main, hartree_sum_pucker_cluster, compare_rmsd_xyz
import logging



__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'xyz_cluster')

# Input files #

OXANE_HARTREE_SUM_FILE = os.path.join(SUB_DATA_DIR, 'm02X-test.csv')
OXANE_1c4_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-1c4-freeze_B3LYP-relax_B3LYP.xyz')
OXANE_1e_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-1e-freeze_B3LYP-relax_B3LYP.xyz')
OXANE_4c1_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-4c1-freeze_B3LYP-relax_B3LYP.xyz')
OXANE_HARTREE_CLUSTER_FILE = os.path.join(SUB_DATA_DIR, 'xyz_cluster-sampleout.txt')

# Good output
XYZ_TOL = 1.0e-12
RMSD_KABSCH_SIMILAR_GOOD = (0.000300503125935, 3, 6)
RMSD_KABSCH_SIMILAR_1c4to4c1 = (0.45429783853700906, 3, 6)
PUCK_2SO = '2so'
PUCK_2SO_FILES = ['25Bm062xconstb3lypbigb3lrelm062x.log', 'E4m062xconstb3lypbigcon1b3ltsm062x.log']


class TestFailWell(unittest.TestCase):
    def testHelp(self):
        test_input = ['-h']
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertFalse(output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)

    def testNoSuchFile(self):
        test_input = ["-s", "ghost"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue("Could not find" in output)


class TestMain(unittest.TestCase):
    def testReadHartreeSummary(self):
        hartree_dict, pucker_filename_dict = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_FILE)
        self.assertTrue(pucker_filename_dict[PUCK_2SO], PUCK_2SO_FILES)

    def testTwoFiles_similar(self):
        rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE,
                                                                                 OXANE_1e_INPUT_FILE)
        self.assertTrue(abs(rmsd_kabsch - RMSD_KABSCH_SIMILAR_GOOD[0]) < XYZ_TOL)

    def testTwoFiles_1c4to4c1(self):
        rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE,
                                                                                 OXANE_4c1_INPUT_FILE)
        self.assertTrue(abs(rmsd_kabsch - RMSD_KABSCH_SIMILAR_1c4to4c1[0]) < XYZ_TOL)

    def testTwoFiles_PrintFeature(self):
        compare_rmsd_xyz(OXANE_1c4_INPUT_FILE, OXANE_1e_INPUT_FILE, print_status='on')
        with capture_stdout(compare_rmsd_xyz, OXANE_1c4_INPUT_FILE, OXANE_1e_INPUT_FILE, print_status='on') as output:
            self.assertTrue("Rmsd" in output)
            self.assertTrue(len(output) > 100)
