#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_read_sdf
----------------------------------

Tests for `read_sdf` module.
"""

import unittest
import os
from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr
from qm_utils.read_sdf import main
import logging

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'read_sdf')

# Input files #

SDF_FILE = os.path.join(SUB_DATA_DIR, '14b.sdf')
COM_FILE = os.path.join(SUB_DATA_DIR, '14b.com')
COM_FILE_GOOD = os.path.join(SUB_DATA_DIR, '14b_good.com')
COM_FILE2 = os.path.join(SUB_DATA_DIR, '1c4.com')
COM_FILE3 = os.path.join(SUB_DATA_DIR, '3e.com')
CP_FILE = os.path.join(TEST_DIR, 'cp.inp')
CP_FILE_GOOD = os.path.join(SUB_DATA_DIR, 'cp_good.inp')

SDF_WRONG_ORDER = os.path.join(SUB_DATA_DIR, '14b_wrong_order.txt')
COM_FILE4 = os.path.join(SUB_DATA_DIR, '14b_wrong_order.com')


class TestMain(unittest.TestCase):
    def testDefInp(self):
        try:
            main([])
            self.assertFalse(diff_lines(COM_FILE, COM_FILE_GOOD))
            self.assertFalse(diff_lines(CP_FILE, CP_FILE_GOOD))
        finally:
            for o_file in [COM_FILE, COM_FILE2, COM_FILE3, CP_FILE]:
                silent_remove(o_file, disable=DISABLE_REMOVE)


class TestFailWell(unittest.TestCase):
    def testWrongOrderAtoms(self):
        try:
            with capture_stderr(main, ['-f', '14b_wrong_order.txt']) as output:
                self.assertTrue('Expected the first five atoms to be carbons' in output)
                self.assertTrue('Expected the 6th atom to be an oxygen' in output)
        finally:
            for o_file in [COM_FILE4, CP_FILE]:
                silent_remove(o_file, disable=DISABLE_REMOVE)

    def testUnrecArg(self):
        with capture_stderr(main, ['-@']) as output:
            self.assertTrue('unrecognized arguments' in output)
