#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_read_sdf
----------------------------------

Tests for `read_sdf` module.
"""

import unittest
import os
from qm_utils.qm_common import silent_remove, diff_lines
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
CP_FILE = os.path.join(TEST_DIR, 'cp.inp')
CP_FILE_GOOD = os.path.join(SUB_DATA_DIR, 'cp_good.inp')


class TestMain(unittest.TestCase):
    def testDefInp(self):
        try:
            main([])
            self.assertFalse(diff_lines(COM_FILE, COM_FILE_GOOD))
            self.assertFalse(diff_lines(CP_FILE, CP_FILE_GOOD))
        finally:
            silent_remove(COM_FILE, disable=DISABLE_REMOVE)
            silent_remove(CP_FILE, disable=DISABLE_REMOVE)
