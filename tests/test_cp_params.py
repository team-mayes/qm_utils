#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_cp
----------------------------------

Tests for `cp` module. (CP = Cremer-Pople)
"""

import unittest
import os
from qm_utils.cp_params import main
from qm_utils.qm_common import diff_lines, silent_remove, capture_stderr, capture_stdout, TOL
import logging

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'cp_params')

# Input & corresponding output files #

IN_FILE = os.path.join(SUB_DATA_DIR, 'cp_good.inp')
OUT_FILE = os.path.join(SUB_DATA_DIR, 'cp.out')
OUT_FILE_GOOD = os.path.join(SUB_DATA_DIR, 'cp_good.out')

IN_FILE_SHORT = os.path.join(SUB_DATA_DIR, 'cp_short.inp')
OUT_FILE_SHORT = os.path.join(SUB_DATA_DIR, 'cp_short.out')

MISS_VALS_IN_FILE = os.path.join(SUB_DATA_DIR, 'cp_missing_vals.inp')
EMPTY_PUCKER_DICT = os.path.join(SUB_DATA_DIR, 'pucker_dict_empty.csv')


class TestMain(unittest.TestCase):
    """
    This is not a straightforward test because it is possible to get different values for the CP parameter theta for
    1c4 which are insignificant. If that is the only difference, let the test pass.
    """
    def testGoodInp(self):
        try:
            # start by not passing, and pass if one of two tests are true
            pass_test = False
            main([IN_FILE, "-o", OUT_FILE])
            diffs = diff_lines(OUT_FILE, OUT_FILE_GOOD)

            if len(diffs) == 0:
                pass_test = True
            elif len(diffs) == 2:
                diff0 = diffs[0].split()
                diff1 = diffs[1].split()
                if diff0[1] == '1c4.sdf' and len(diff0) == len(diff1) and len(diff0) == 6:
                    pass_test = True
                    for index in [1, 3, 4, 5]:
                        if index == 4:
                            float0 = float(diff0[index])
                            float1 = float(diff1[index])
                            float_diff = abs(float0 - float1)
                            calc_tol = max(TOL * max(abs(float0), abs(float1)), TOL)
                            if float_diff > calc_tol:
                                pass_test = False
                                break
                        elif diff0[index] != diff1[index]:
                            pass_test = False
                            break
            self.assertTrue(pass_test)
        finally:
            silent_remove(OUT_FILE, disable=DISABLE_REMOVE)

    def testToStdOut(self):
        test_input = [IN_FILE_SHORT, "-s"]
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        try:
            with capture_stdout(main, test_input) as output:
                self.assertTrue('14b.sdf       240.000   90.000    0.835 14b\n'
                                '3e.sdf          0.517  125.167    0.477 3e\n'
                                'Wrote file: ' in output)
        finally:
            silent_remove(OUT_FILE_SHORT, disable=DISABLE_REMOVE)


class TestFailWell(unittest.TestCase):
    def testNoArgs(self):
        with capture_stdout(main, []) as output:
            self.assertTrue('optional arguments' in output)

    def testMissVals(self):
        if logger.isEnabledFor(logging.DEBUG):
            main([MISS_VALS_IN_FILE])
        with capture_stderr(main, [MISS_VALS_IN_FILE]) as output:
            self.assertTrue('Expected exactly' in output)

    def testEmptyPuckerDict(self):
        test_in = [IN_FILE, "-o", OUT_FILE, "-p", EMPTY_PUCKER_DICT]
        if logger.isEnabledFor(logging.DEBUG):
            main(test_in)
        with capture_stderr(main, test_in) as output:
            self.assertTrue('Did not find a closest pucker' in output)
