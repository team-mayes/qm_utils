#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest
from qm_utils.gen_puck_table import read_hartree_files, create_pucker_gibbs_dict, rel_energy_values, \
    creating_level_dict_of_dict, main, check_same_puckers_lmirc_and_lm
from qm_utils.qm_common import diff_lines, silent_remove

__author__ = 'SPVicchio'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'gen_puck_table')

# Sample Files#
SAMPLE_HARTREE_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-optall-oxane-b3lyp.csv')


# Tests #

class TestGenPuckerTableFunctions(unittest.TestCase):
    def testReadHartreeFiles(self):
        hartree_headers, hartree_dict, job_type, qm_method = read_hartree_files(SAMPLE_HARTREE_FILE, SUB_DATA_DIR)
        self.assertEquals(qm_method,'b3lyp-lm')
