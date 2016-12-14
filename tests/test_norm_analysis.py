#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.norm_analysis import read_puckering_information, analyze_first_normal_mode, split_ring_index, main

from qm_utils.qm_common import diff_lines, silent_remove, capture_stderr, capture_stdout

__author__ = 'SPVicchio'


# Constants #

DEF_RING_ORDER = '8,1,9,13,17,5'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'norm_analysis')

# Input Files #

NORM_SAMPLE_FILE = os.path.join(SUB_DATA_DIR, 'bxyl_1e_32-TS_am1_norm-norm.txt')
BAD_SAMPLE_FILE = os.path.join(SUB_DATA_DIR, 'bxyl_not_TS_structure.txt')
NORM_SAMPLE_FILE_ONE = os.path.join(SUB_DATA_DIR, 'bxyl_eo_64-TS_am1_norm-norm.txt')
INPUT_MAIN_NORM_FILE = os.path.join(SUB_DATA_DIR, 'z_hartree_norm_analysis_am1.txt')

# Output Files #

GOOD_OUTPUT_NORM_READ_TABLE = ['         (  1,   3): 1.50', '         (  1,   8): 15.40', '         (  1,   9): 12.40', '         (  5,   8): 19.60', '         (  5,  17): 33.80']
GOOD_OUTPUT_NORM_READ_FILENAME_BOTH = 'bxyl_1e_32-TS_am1_norm.log'
GOOD_OUTPUT_RING_ATOMS_INDEX = [1, 5, 8, 9, 13, 17]
GOOD_OUTPUT_RING_TS_STATUS_BOTH = 'yes'
GOOD_OUTPUT_RING_TS_STATUS_ONE = 'no'
GOOD_OUTPUT_PERCENTAGE = 81.2
GOOD_OUTPUT_FILE_LIST_EXO = os.path.join(SUB_DATA_DIR,'z_norm-analysis_TS_exo_puckers_z_hartree_norm_analysis_am1-good.txt')
GOOD_OUTPUT_FILE_LIST_PUCK = os.path.join(SUB_DATA_DIR,'z_norm-analysis_TS_ring_puckers_z_hartree_norm_analysis_am1-good.txt')
OUT_FILE_LIST_PUCK = os.path.join(SUB_DATA_DIR,'z_norm-analysis_TS_ring_puckers_z_hartree_norm_analysis_am1.txt')
OUT_FILE_LIST_EXO = os.path.join(SUB_DATA_DIR,'z_norm-analysis_TS_exo_puckers_z_hartree_norm_analysis_am1.txt')

# Tests #
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


class TestNORMFunctions(unittest.TestCase):
    def testReadNormTxtFile(self):
        out_filename, dihedral_table = read_puckering_information(NORM_SAMPLE_FILE, SUB_DATA_DIR)
        self.assertEquals(GOOD_OUTPUT_NORM_READ_FILENAME_BOTH, out_filename)
        self.assertEquals(GOOD_OUTPUT_NORM_READ_TABLE, dihedral_table)

    def testAnalyzeFirstNormalMode(self):
        filename, out_percent = analyze_first_normal_mode(NORM_SAMPLE_FILE, GOOD_OUTPUT_NORM_READ_TABLE, GOOD_OUTPUT_RING_ATOMS_INDEX)
        self.assertEquals(out_percent, GOOD_OUTPUT_PERCENTAGE)

    def testRingIndexSplit(self):
        sorted_ring_atoms_index = split_ring_index(DEF_RING_ORDER)
        self.assertEquals(GOOD_OUTPUT_RING_ATOMS_INDEX,sorted_ring_atoms_index)

class TestMain(unittest.TestCase):
    def testMain(self):
        try:
            test_input = ["-s", INPUT_MAIN_NORM_FILE, "-r", DEF_RING_ORDER]
            main(test_input)
            self.assertFalse(diff_lines(OUT_FILE_LIST_PUCK, GOOD_OUTPUT_FILE_LIST_PUCK))
            self.assertFalse(diff_lines(OUT_FILE_LIST_EXO, GOOD_OUTPUT_FILE_LIST_EXO))
        finally:
            pass
            silent_remove(OUT_FILE_LIST_PUCK)
            silent_remove(OUT_FILE_LIST_EXO)
