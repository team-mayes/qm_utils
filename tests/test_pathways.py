#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest
from qm_utils.pathways import read_pathway_information, main
from qm_utils.qm_common import diff_lines, silent_remove

__author__ = 'SPVicchio'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'pathways')

# Sample Files#
INPUT_BXYL_PATHWAYS = os.path.join(SUB_DATA_DIR, 'a_pathway_listbxyl_am1.txt')
INPUT_BXYL_CSV = os.path.join(SUB_DATA_DIR, 'z_pathways-am1.csv')
# Good Outputs #


# Tests #

class TestPathwayFunctions(unittest.TestCase):
    def testReadPathwayInformation(self):
        read_pathway_information(INPUT_BXYL_PATHWAYS, INPUT_BXYL_CSV)

class TestMain(unittest.TestCase):
    def testMain(self):
        input_info = ["-f", INPUT_BXYL_PATHWAYS]
        main(input_info)
