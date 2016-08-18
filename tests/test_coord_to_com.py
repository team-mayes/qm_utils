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
from qm_utils.coord_to_com import main
import logging

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'coord_to_com')

# Input files #

PDB_INI = os.path.join(SUB_DATA_DIR, 'read_pdb.ini')
PDB_FILE = os.path.join(SUB_DATA_DIR, '1c4.pdb')
COM_1C4 = os.path.join(SUB_DATA_DIR, '1c4.com')
COM_1C4_GOOD = os.path.join(SUB_DATA_DIR, '1c4_good.com')
COM_1C4_FREEZE = os.path.join(SUB_DATA_DIR, '1c4_freeze.com')
COM_E3_FILE = os.path.join(SUB_DATA_DIR, 'e3.com')
COM_4C1_FILE = os.path.join(SUB_DATA_DIR, '4c1.com')
CP_FILE = os.path.join(SUB_DATA_DIR, 'cp.inp')
CP_FILE_GOOD = os.path.join(SUB_DATA_DIR, 'cp_good.inp')

PDB_WRONG_ORDER = os.path.join(SUB_DATA_DIR, '1c4_wrong_order.txt')
COM_WRONG_ORDER = os.path.join(SUB_DATA_DIR, '1c4_wrong_order.com')

PDB_TOO_FEW = os.path.join(SUB_DATA_DIR, '1c4_too_few.txt')
COM_TOO_FEW = os.path.join(SUB_DATA_DIR, '1c4_too_few.com')

SDF_FILE = os.path.join(SUB_DATA_DIR, '14b.sdf')
COM_14B_FILE = os.path.join(SUB_DATA_DIR, '14b.com')
COM_14B_FILE_GOOD = os.path.join(SUB_DATA_DIR, '14b_good.com')
COM_3E_FILE = os.path.join(SUB_DATA_DIR, '3e.com')
COM_O3B_FILE = os.path.join(SUB_DATA_DIR, 'o3b.com')
CP_SDF_FILE_GOOD = os.path.join(SUB_DATA_DIR, 'cp_sdf_good.inp')

PDB_TOO_FEW_RING_INI = os.path.join(SUB_DATA_DIR, 'read_pdb_too_few_ring_atoms.ini')


class TestMain(unittest.TestCase):
    def testDefInp(self):
        try:
            test_input = ["-o", SUB_DATA_DIR]
            if logger.isEnabledFor(logging.DEBUG):
                main(test_input)
            with capture_stderr(main, test_input) as output:
                self.assertTrue('Will use only default values' in output)
            self.assertFalse(diff_lines(COM_1C4, COM_1C4_GOOD))
            self.assertFalse(diff_lines(CP_FILE, CP_FILE_GOOD))
        finally:
            for o_file in [COM_1C4, COM_E3_FILE, COM_4C1_FILE]:
                silent_remove(o_file, disable=DISABLE_REMOVE)
            silent_remove(CP_FILE, disable=DISABLE_REMOVE)

    def testAsTemplate(self):
        main(["-c", PDB_INI, "-o", SUB_DATA_DIR])
        self.assertFalse(diff_lines(COM_1C4, COM_1C4_FREEZE))
        for o_file in [COM_1C4, COM_E3_FILE, COM_4C1_FILE]:
            silent_remove(o_file, disable=DISABLE_REMOVE)
        silent_remove(CP_FILE, disable=DISABLE_REMOVE)

    def testSDF(self):
        try:
            main(["-f", "*sdf", "-o", SUB_DATA_DIR])
            self.assertFalse(diff_lines(COM_14B_FILE, COM_14B_FILE_GOOD))
            self.assertFalse(diff_lines(CP_FILE, CP_SDF_FILE_GOOD))
        finally:
            for o_file in [COM_14B_FILE, COM_1C4, COM_3E_FILE, COM_4C1_FILE, COM_O3B_FILE]:
                silent_remove(o_file, disable=DISABLE_REMOVE)
            silent_remove(CP_FILE, disable=DISABLE_REMOVE)


class TestFailWell(unittest.TestCase):
    def testWrongOrderAtoms(self):
        try:
            test_input = ['-f', '1c4_wrong_order.txt', "-o", SUB_DATA_DIR, "-t", "pdb"]
            if logger.isEnabledFor(logging.DEBUG):
                main(test_input)
            with capture_stderr(main, test_input) as output:
                self.assertTrue("Expected atom 1 to have type 'C'. Found 'O'" in output)
                self.assertTrue("Expected atom 6 to have type 'O'. Found 'C'" in output)
        finally:
            for o_file in [COM_WRONG_ORDER, CP_FILE]:
                silent_remove(o_file, disable=DISABLE_REMOVE)

    def testWrongOrderSDF(self):
        try:
            test_input = ['-f', '14b_wrong_order.txt', "-o", SUB_DATA_DIR, "-t", "sdf"]
            if logger.isEnabledFor(logging.DEBUG):
                main(test_input)
            with capture_stderr(main, test_input) as output:
                self.assertTrue("Expected atom 1 to have type 'C'. Found 'O'" in output)
                self.assertTrue("Expected atom 6 to have type 'O'. Found 'C'" in output)
        finally:
            for o_file in [COM_WRONG_ORDER, CP_FILE]:
                silent_remove(o_file, disable=DISABLE_REMOVE)

    def testTooFewAtomsFound(self):
        try:
            test_input = ['-f', '1c4_too_few.txt', "-o", SUB_DATA_DIR, "-t", "pdb"]
            if logger.isEnabledFor(logging.DEBUG):
                main(test_input)
            with capture_stderr(main, test_input) as output:
                self.assertTrue("Did not find the expected six ring atoms" in output)
        finally:
            for o_file in [COM_TOO_FEW, CP_FILE]:
                silent_remove(o_file, disable=DISABLE_REMOVE)

    def testTooFewRingAtomsListed(self):
        try:
            test_input = ['-f', '1c4.pdb', "-o", SUB_DATA_DIR, "-c", PDB_TOO_FEW_RING_INI]
            if logger.isEnabledFor(logging.DEBUG):
                main(test_input)
            with capture_stderr(main, test_input) as output:
                self.assertTrue("To print cp_params input, enter" in output)
            #     self.assertTrue("Did not find any data for cp_params input" in output)
        finally:
            for o_file in [COM_1C4, CP_FILE]:
                silent_remove(o_file, disable=DISABLE_REMOVE)

    def testUnrecArg(self):
        with capture_stderr(main, ['-@']) as output:
            self.assertTrue('unrecognized arguments' in output)

    def testWrongFileType(self):
        with capture_stderr(main, ['-f', '14b_wrong_order.txt']) as output:
            self.assertTrue('This program currently reads only pdb and sdf file formats' in output)
