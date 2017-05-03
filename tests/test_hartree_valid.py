import os
import unittest

import pandas as pd

from qm_utils.old_scripts.hartree_valid import verify_same_cols, verify_transition_state, verify_local_minimum

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data', 'group_puckers')

# Test Files #

GOOD_MIN = os.path.join(DATA_DIR, 'bxylose_b3lyp_min_hartree.csv')
GOOD_TS = os.path.join(DATA_DIR, 'bxylose_b3lyp_ts_hartree.csv')
DIFF_COLS = os.path.join(DATA_DIR, 'bxylose_b3lyp_min_hartree_diff.csv')


class TestSameCols(unittest.TestCase):
    def testSame(self):
        data = pd.read_csv(GOOD_MIN)
        self.assertEqual(0, len(verify_same_cols(data)))

    def testDiff(self):
        data = pd.read_csv(DIFF_COLS)
        self.assertEqual(2, len(verify_same_cols(data)))


class TestTsVerify(unittest.TestCase):
    def testTrueForTs(self):
        data = pd.read_csv(GOOD_TS)
        self.assertTrue(verify_transition_state(data))

    def testFalseForMin(self):
        data = pd.read_csv(GOOD_MIN)
        self.assertFalse(verify_transition_state(data))


class TestMinVerify(unittest.TestCase):
    def testTrueForMin(self):
        data = pd.read_csv(GOOD_MIN)
        self.assertTrue(verify_local_minimum(data))

    def testFalseForTs(self):
        data = pd.read_csv(GOOD_TS)
        self.assertFalse(verify_local_minimum(data))
