#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import csv
import unittest
import os
from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, warning, create_out_fname, \
    write_csv
from qm_utils.xyz_cluster import main, hartree_sum_pucker_cluster, compare_rmsd_xyz, test_clusters, dict_to_csv_writer
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

OXANE_HARTREE_SUM_HEATHER_FILE = os.path.join(SUB_DATA_DIR, 'b3lyp-test.csv')
OXANE_1c4_INPUT_FILE = os.path.join(SUB_DATA_DIR, "oxane-1c4-freeze_B3LYP-relax_B3LYP.log")
OXANE_1e_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-1e-freeze_B3LYP-relax_B3LYP.log')
OXANE_4c1_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-4c1-freeze_B3LYP-relax_B3LYP.log')
OXANE_HARTREE_CLUSTER_FILE = os.path.join(SUB_DATA_DIR, 'xyz_cluster-sampleout.txt')
OXANE_HARTREE_SUM_B3LYP_FILE = os.path.join(SUB_DATA_DIR, 'B3LYP_hartree_sum-cpsnap.csv')
OXANE_HARTREE_DICT_CLUSTER_FILE = os.path.join(SUB_DATA_DIR, 'oxane_hartree_sum_B3LYP_hartree_dict_GOOD.csv')

# Good output
XYZ_TOL = 1.0e-12
RMSD_KABSCH_SIMILAR_GOOD = (0.000300503125935, 3, 6)
RMSD_KABSCH_SIMILAR_1c4to4c1 = (0.45429783853700906, 3, 6)
PUCK_2SO = '2so'
PUCK_2SO_FILES = ['25Bm062xconstb3lypbigb3lrelm062x.log', 'E4m062xconstb3lypbigcon1b3ltsm062x.log']
TEST_XYZ_TOL = '0.000001'


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
        hartree_dict, pucker_filename_dict = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_HEATHER_FILE)
        self.assertTrue(pucker_filename_dict[PUCK_2SO], PUCK_2SO_FILES)

    def testTwoFiles_similar(self):
        rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE,
                                                                                 OXANE_1e_INPUT_FILE,
                                                                                 SUB_DATA_DIR)
        self.assertTrue(abs(rmsd_kabsch - RMSD_KABSCH_SIMILAR_GOOD[0]) < XYZ_TOL)

    def testTwoFiles_1c4to4c1(self):
        rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE,
                                                                                 OXANE_4c1_INPUT_FILE,
                                                                                 SUB_DATA_DIR)
        self.assertTrue(abs(rmsd_kabsch - RMSD_KABSCH_SIMILAR_1c4to4c1[0]) < XYZ_TOL)

    def testTwoFiles_PrintFeature(self):
        # compare_rmsd_xyz(OXANE_1c4_INPUT_FILE, OXANE_1e_INPUT_FILE, print_status='on')
        with capture_stdout(compare_rmsd_xyz, OXANE_1c4_INPUT_FILE, OXANE_1e_INPUT_FILE, SUB_DATA_DIR,
                                                                                        print_option='on') as output:
            self.assertTrue("Rmsd" in output)
            self.assertTrue(len(output) > 100)

# TODO discussion with Heather.. need to make sure that all of the lines are aligned properly
    def testwrite_csv(self):
        try:
            hartree_dict, pucker_filename_dict = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
            out_fname = 'oxane_hartree_sum_B3LYP_hartree_dict.csv'
            out_filename_path = os.path.join(SUB_DATA_DIR,out_fname)
            dict_to_csv_writer(pucker_filename_dict,out_fname, SUB_DATA_DIR)
            self.assertTrue(diff_lines(out_filename_path,OXANE_HARTREE_DICT_CLUSTER_FILE))
        finally:
            silent_remove(os.path.join(SUB_DATA_DIR,out_fname))

#    def testMain(self):
#        test_input = ["-s", OXANE_HARTREE_SUM_HEATHER_FILE, "-t", TEST_XYZ_TOL]
#        main(test_input)


