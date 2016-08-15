#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_cp
----------------------------------

Tests for `cp` module.
"""

import unittest
import os
from qm_utils.cp import main
from qm_utils.qm_common import diff_lines, silent_remove

__author__ = 'hmayes'

# Directories #

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'read_sdf')

# Input files #

IN_FILE = os.path.join(SUB_DATA_DIR, 'cp_good.inp')
OUT_FILE = os.path.join(SUB_DATA_DIR, 'cp.out')
OUT_FILE_GOOD = os.path.join(SUB_DATA_DIR, 'cp_good.out')


class TestMain(unittest.TestCase):
    def testDefInp(self):
        try:
            main([IN_FILE, "-o", OUT_FILE])
            self.assertFalse(diff_lines(OUT_FILE, OUT_FILE_GOOD))
        finally:
            silent_remove(OUT_FILE)
