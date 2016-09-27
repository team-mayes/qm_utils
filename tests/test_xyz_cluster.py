#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""

import unittest
import os
from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout
from qm_utils.xyz_cluster import main, process_hartree_sum
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

OXANE_HARTREE_SUM_FILE = os.path.join(SUB_DATA_DIR, 'xyz_cluster-sampleout.txt')
OXANE_1c4_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-1c4-freeze_B3LYP-relax_B3LYP.xyz')
OXANE_1e_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-1e-freeze_B3LYP-relax_B3LYP.xyz')

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
    def testSDF(self):
        try:
            main(["-s", OXANE_HARTREE_SUM_FILE])
            process_hartree_sum(OXANE_HARTREE_SUM_FILE)
        finally:
            print("We're writing tests!")
    def testTwoFiles(self):
        try:
            main(["-f1", OXANE_1c4_INPUT_FILE, "-f2", OXANE_1e_INPUT_FILE])




