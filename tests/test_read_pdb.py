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
from qm_utils.read_pdb import main
import logging

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'read_pdb')
SDF_DATA_DIR = os.path.join(DATA_DIR, 'read_sdf')

# Input files #

PDB_INI = os.path.join(SUB_DATA_DIR, 'read_pdb.ini')
PDB_FILE = os.path.join(SUB_DATA_DIR, '1c4.pdb')
COM_FILE1 = os.path.join(SUB_DATA_DIR, '1c4.com')
COM_FILE_GOOD = os.path.join(SUB_DATA_DIR, '1c4_good.com')
COM_FILE2 = os.path.join(SUB_DATA_DIR, 'e3.com')
COM_FILE3 = os.path.join(SUB_DATA_DIR, '4c1.com')
CP_FILE = os.path.join(SUB_DATA_DIR, 'cp.inp')
CP_FILE_GOOD = os.path.join(SUB_DATA_DIR, 'cp_good.inp')

PDB_WRONG_ORDER = os.path.join(SUB_DATA_DIR, '1c4_wrong_order.txt')
COM_FILE4 = os.path.join(SUB_DATA_DIR, '1c4_wrong_order.com')


class TestMain(unittest.TestCase):
    def testDefInp(self):
        try:
            main(["-c", PDB_INI, "-o", SUB_DATA_DIR])
            self.assertFalse(diff_lines(COM_FILE1, COM_FILE_GOOD))
            self.assertFalse(diff_lines(CP_FILE, CP_FILE_GOOD))
        finally:
            for o_file in [COM_FILE1, COM_FILE2, COM_FILE3]:
                silent_remove(o_file, disable=DISABLE_REMOVE)
            silent_remove(CP_FILE, disable=DISABLE_REMOVE)


class TestFailWell(unittest.TestCase):
    def testWrongOrderAtoms(self):
        try:
            if logger.isEnabledFor(logging.DEBUG):
                main(['-f', '1c4_wrong_order.txt', "-c", PDB_INI, "-o", SUB_DATA_DIR])
            with capture_stderr(main, ['-f', '1c4_wrong_order.txt', "-c", PDB_INI, "-o", SUB_DATA_DIR]) as output:
                self.assertTrue("Expected atom 1 to have type 'C'. Found 'O'" in output)
                self.assertTrue("Expected atom 6 to have type 'O'. Found 'C'" in output)
        finally:
            for o_file in [COM_FILE4, CP_FILE]:
                silent_remove(o_file, disable=DISABLE_REMOVE)

    def testNoIni(self):
        with capture_stderr(main, []) as output:
            self.assertTrue('Could not read file: read_pdb.ini' in output)

    def testUnrecArg(self):
        with capture_stderr(main, ['-@', "-c", PDB_INI]) as output:
            self.assertTrue('unrecognized arguments' in output)
