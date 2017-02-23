#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_cp
----------------------------------

Tests for `cp` module. (CP = Cremer-Pople)
"""

import unittest
import os
from qm_utils.group_puckers import main, find_TS_for_each_min, finding_H_and_pairing, make_dict
from qm_utils.qm_common import diff_lines, silent_remove
import logging
import pandas as pd

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'group_puckers')

# Input & corresponding output files #
BXYLOSE_B3LYP_LOCAL_MIN_HARTREE = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-localmin-bxyl-am1.csv')
BXYLOSE_B3LYP_MIN_HARTREE = os.path.join(SUB_DATA_DIR, 'z_hartree_raw_lmirc_am1.csv')
BXYLOSE_B3LYP_TS_HARTREE = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-TS-bxyl-am1.csv')
BXYLOSE_B3LYP_MIN_HARTREE_DIFF = os.path.join(SUB_DATA_DIR, 'bxylose_b3lyp_min_hartree_diff.csv')
OUTPUT_EMPTY_PUCKER = os.path.join(SUB_DATA_DIR, 'group_puckers_output.csv')
OXANE_MIN = os.path.join(SUB_DATA_DIR, 'z_hartree_out-unsorted-lmirc-oxane-am1.csv')
OXANE_LOCAL_MIN = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-optall-oxane-am1.csv')
OXANE_TS = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-TS-oxane-am1.csv')

# Good Output

GOOD_OUTPUT_EMPTY_PUCKER = os.path.join(SUB_DATA_DIR, 'group_puckers_output_good.csv')

#Testing the functions in group_puckers.py
class TestFunctions(unittest.TestCase):
    def testFindTSForEachMin(self):
        ts = pd.read_csv(BXYLOSE_B3LYP_MIN_HARTREE)
        TS_point = find_TS_for_each_min(ts, '_norm-ircf_am1-minIRC_am1.log', '_norm-ircr_am1-minIRC_am1.log')
        self.assertEqual(TS_point[1].return_min(), '1c4')
        self.assertEqual(TS_point[1].return_H(), -0.222527)
        self.assertEqual(TS_point[1].return_name(), 'bxyl_1e_52-TS_am1')

    def testFinding_H_and_pairing(self):
        min = pd.read_csv(BXYLOSE_B3LYP_TS_HARTREE)
        ts = pd.read_csv(BXYLOSE_B3LYP_MIN_HARTREE)
        TS_point = find_TS_for_each_min(ts, '_norm-ircf_am1-minIRC_am1.log', '_norm-ircr_am1-minIRC_am1.log')
        paths = finding_H_and_pairing(TS_point, min)

        self.assertEqual(paths[0].return_pucker(), '5ho')
        self.assertEqual("{0:.6f}".format(paths[0].return_H()),'-0.218374')
        self.assertEqual(paths[0].return_name(), 'bxyl_1e_63-TS_am1.log')

    def testMake_Dict(self):
        min = pd.read_csv(BXYLOSE_B3LYP_TS_HARTREE)
        ts = pd.read_csv(BXYLOSE_B3LYP_MIN_HARTREE)
        TS_point = find_TS_for_each_min(ts, '_norm-ircf_am1-minIRC_am1.log', '_norm-ircr_am1-minIRC_am1.log')
        paths = finding_H_and_pairing(TS_point, min)
        test_dict = make_dict(paths[0])
        self.assertEqual(test_dict["File name"], 'bxyl_1e_63-TS_am1.log')
        self.assertEqual(test_dict["Minimum1"],'bo3')
        self.assertEqual("{0:.6f}".format(test_dict["Delta H1"]), '1.091239')
        self.assertEqual(test_dict["Transition Pucker"], '5ho')
        self.assertEqual(test_dict["Minimum2"],'1c4')
        self.assertEqual("{0:.6f}".format(test_dict["Delta H2"]), '-0.553463')

#testing the main of group_puckers.py
class TestMain(unittest.TestCase):
    def testMain(self):
        try:
            test_input = ['-m', OXANE_MIN,
                          '-s', OXANE_TS,
                          '-o', OUTPUT_EMPTY_PUCKER,
                          '-l', OXANE_LOCAL_MIN,
                          '-for', 'am1-ircf_am1-minIRC_am1.log',
                          '-rev', 'am1-ircr_am1-minIRC_am1.log']
            main(test_input)
            self.assertFalse(diff_lines(OUTPUT_EMPTY_PUCKER, GOOD_OUTPUT_EMPTY_PUCKER))

        finally:
            silent_remove(OUTPUT_EMPTY_PUCKER)

