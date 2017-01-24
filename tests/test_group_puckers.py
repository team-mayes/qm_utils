#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_read_pdb
----------------------------------

Tests for `read_pdb` module.
"""

import unittest
import os
from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'group_puckers')

# Input files #

BXYL_B3LYP_MIN_HARTREE = os.path.join(SUB_DATA_DIR, 'bxylose_b3lyp_min_hartree.csv')
BXYL_B3LYP_MIN_HARTREE_DIFF = os.path.join(SUB_DATA_DIR, 'bxylose_b3lyp_min_hartree_diff.csv')
BXYL_B3LYP_TS_HARTREE = os.path.join(SUB_DATA_DIR, 'bxylose_b3lyp_ts_hartree.csv')

# Tests #



class TestMain(unittest.TestCase):

    def testMain(self):
        I
