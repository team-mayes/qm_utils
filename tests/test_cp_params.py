#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_cp
----------------------------------

Tests for `cp` module.
"""

import unittest
import os
from qm_utils.cp_params import main
from qm_utils.qm_common import diff_lines, silent_remove, capture_stderr, capture_stdout
import logging

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'read_sdf')

# Input & corresponding output files #

IN_FILE = os.path.join(SUB_DATA_DIR, 'cp_good.inp')
OUT_FILE = os.path.join(SUB_DATA_DIR, 'cp.out')
OUT_FILE_GOOD = os.path.join(SUB_DATA_DIR, 'cp_good.out')

MISS_VALS_IN_FILE = os.path.join(SUB_DATA_DIR, 'cp_missing_vals.inp')


class TestMain(unittest.TestCase):
    def testGoodInp(self):
        try:
            main([IN_FILE, "-o", OUT_FILE])
            self.assertFalse(diff_lines(OUT_FILE, OUT_FILE_GOOD))
        finally:
            silent_remove(OUT_FILE, disable=DISABLE_REMOVE)


class TestFailWell(unittest.TestCase):
    def testNoArgs(self):
        with capture_stdout(main, []) as output:
            self.assertTrue('optional arguments' in output)

    def testMissVals(self):
        if logger.isEnabledFor(logging.DEBUG):
            main([MISS_VALS_IN_FILE])
        with capture_stderr(main, [MISS_VALS_IN_FILE]) as output:
            self.assertTrue('Expected exactly' in output)
