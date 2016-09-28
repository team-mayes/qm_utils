#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""

import unittest
import os
from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout
from qm_utils.xyz_cluster import main, process_hartree_sum, compare_rmsd_xyz
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
RMSD_KABSCH_SIMILAR_GOOD = (0.000300503125935, 3, 6)
RMSD_KABSCH_SIMILAR_1c4to4c1 = (0.45429783853700906, 3, 6)

OXANE_HARTREE_CLUSTER_FILE = os.path.join(SUB_DATA_DIR, 'xyz_cluster-sampleout.txt')


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
        try:
            process_hartree_sum(OXANE_HARTREE_SUM_FILE)
        finally:
            #print("\n The following test worked")
            pass

# TODO determine how I would want my tests to end (right now they pass)
    def testTwoFiles_similar(self):
        try:
            rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE, OXANE_1e_INPUT_FILE)
            #TODO at a way to compare the xyz coordinates from the various good files
        finally:
            if rmsd_kabsch == RMSD_KABSCH_SIMILAR_GOOD[0]:
                pass

    def testTwoFiles_1c4to4c1(self):
        try:
            rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE, OXANE_4c1_INPUT_FILE)
        finally:
            if rmsd_kabsch == RMSD_KABSCH_SIMILAR_1c4to4c1[0]:
                pass

    def testTwoFiles_PrintFeature(self):
        try:
            rmsd_kabsch_similar= compare_rmsd_xyz(OXANE_1c4_INPUT_FILE, OXANE_1e_INPUT_FILE, print_status='on')
        finally:
            # TODO: add a way to catch the output to the screen to verify that something is actually printing here
            pass


