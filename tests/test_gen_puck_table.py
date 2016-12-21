#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.qm_common import capture_stderr, capture_stdout
from qm_utils.gen_pucker_table import main, load_cluster_file

__author__ = 'SPVicchio'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'gen_pucker_table')

SAMPLE_CLUSTER_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster_B3LYP_hartree_sum-cpsnap_good.csv')


class TestFailWell(unittest.TestCase):
    def testHelp(self):
        test_input = ['-h']
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertFalse(output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)


class TestGenPuckerTableFunctions(unittest.TestCase):

    def testLoadClusterFile(self):
        input_file = SAMPLE_CLUSTER_FILE
        cluster_dict = load_cluster_file(input_file)
        print(cluster_dict)
