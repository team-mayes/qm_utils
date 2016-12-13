#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.norm_analysis import read_puckering_information, create_dihedral, split_ring_index, \
    identifying_ring_pucker_di, id_key_structures

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
NORM_SAMPLE_FILE_ONE = os.path.join(SUB_DATA_DIR, 'bxyl_eo_64-TS_am1_norm-norm.txt')

# Output Files #
GOOD_OUTPUT_NORM_READ_TABLE = ['(  1,   3) |   7  | 62.70 | 309.41', '(  1,   8) |   1  | 15.40 | -159.64', '(  1,   9) |   2  | 26.00 | 50.34', '(  5,   8) |   1  | 19.60 | -159.64', '(  5,  17) |   1  | 33.80 | -159.64', '(  9,  11) |   6  | 46.30 | 269.30', '(  9,  13) |   3  | 22.60 | 131.52', '( 13,  15) |  10  | 49.20 | 410.93', '( 13,  17) |   2  | 23.00 | 50.34', '( 17,  19) |  13  | 51.80 | 514.55']
GOOD_OUTPUT_NORM_READ_FILENAME_BOTH = 'bxyl_1e_32-TS_am1_norm.log'
GOOD_OUTPUT_FIRST_MODE_DATA_BOTH = [['(  1,   8) ', 1, ' 15.40 ', -159.64], ['(  5,   8) ', 1, ' 19.60 ', -159.64], ['(  5,  17) ', 1, ' 33.80 ', -159.64]]
GOOD_OUTPUT_RING_ATOMS_INDEX = [1, 5, 8, 9, 13, 17]
GOOD_OUTPUT_RING_TS_STATUS_BOTH = [['(  1,   8) ', 'Both'], ['(  5,   8) ', 'Both'], ['(  5,  17) ', 'Both']]
GOOD_OUTPUT_RING_TS_STATUS_ONE = [['( 17,  19) ', 'First']]


# Tests #

class TestNORMFunctions(unittest.TestCase):
    def testReadNormTxtFile(self):
        out_filename, dihedral_table = read_puckering_information(NORM_SAMPLE_FILE, SUB_DATA_DIR)
        self.assertEquals(GOOD_OUTPUT_NORM_READ_FILENAME_BOTH, out_filename)
        self.assertEquals(GOOD_OUTPUT_NORM_READ_TABLE, dihedral_table)

    def testCreateDihedralLines(self):
        out_filename, first_mode_information =\
            create_dihedral(GOOD_OUTPUT_NORM_READ_FILENAME_BOTH, GOOD_OUTPUT_NORM_READ_TABLE)
        self.assertEquals(GOOD_OUTPUT_FIRST_MODE_DATA_BOTH, first_mode_information)
        self.assertEquals(GOOD_OUTPUT_NORM_READ_FILENAME_BOTH, out_filename)

    def testRingIndexSplit(self):
        sorted_ring_atoms_index = split_ring_index(DEF_RING_ORDER)
        self.assertEquals(GOOD_OUTPUT_RING_ATOMS_INDEX,sorted_ring_atoms_index)

    def testIdentifyingRingPuckerDiBoth(self):
        out_filename, status_ring_puckering = identifying_ring_pucker_di(GOOD_OUTPUT_NORM_READ_FILENAME_BOTH, GOOD_OUTPUT_FIRST_MODE_DATA_BOTH,
                                   GOOD_OUTPUT_RING_ATOMS_INDEX)
        self.assertEquals(GOOD_OUTPUT_RING_TS_STATUS_BOTH, status_ring_puckering)
        self.assertEquals(GOOD_OUTPUT_NORM_READ_FILENAME_BOTH, out_filename)

    def testIdentifyingRingPuckerDiOne(self):
        out_filename, dihedral_table = read_puckering_information(NORM_SAMPLE_FILE_ONE, SUB_DATA_DIR)
        out_filename, first_mode_information =\
            create_dihedral(out_filename, dihedral_table)
        out_filename, status_ring_puckering = identifying_ring_pucker_di(out_filename, first_mode_information,
                                   GOOD_OUTPUT_RING_ATOMS_INDEX)
        self.assertEquals(GOOD_OUTPUT_RING_TS_STATUS_ONE, status_ring_puckering)

    def testIdentifyingKeyStructures(self):
        id_key_structures(GOOD_OUTPUT_NORM_READ_FILENAME_BOTH,GOOD_OUTPUT_RING_TS_STATUS_BOTH)
