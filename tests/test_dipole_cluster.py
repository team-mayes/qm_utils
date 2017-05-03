#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.old_scripts.dipole_cluster import main, hartree_sum_pucker_cluster, check_before_after_sorting
from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, create_out_fname, \
    write_csv, list_to_dict, read_csv_to_dict

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #
TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'dipole_cluster')

# Input Parameters #
XYZ_TOL = 0.1
RMSD_KABSCH_SIMILAR_GOOD = (0.000300503125935, 3, 6)
RMSD_KABSCH_SIMILAR_1c4to4c1 = (0.45429783853700906, 3, 6)
PUCK_2SO = '2so'
PUCK_2SO_FILES = ['25Bm062xconstb3lypbigb3lrelm062x.log', 'E4m062xconstb3lypbigcon1b3ltsm062x.log']

# Input files #
BXYL_HARTREE_PM3MM_TS_FILE = os.path.join(SUB_DATA_DIR, 'z_hartree-unsorted-TS-bxyl-pm3mm.csv')

# Output Files #
OUT_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster_z_hartree-unsorted-TS-bxyl-pm3mm.csv')
OUT_FILE_LIST_0p05 = os.path.join(SUB_DATA_DIR, 'z_files_list_freq_runsz_hartree-unsorted-TS-bxyl-pm3mm.txt')
OUT_FILE_0p1 = os.path.join(SUB_DATA_DIR, 'z_cluster_z_hartree-unsorted-TS-bxyl-pm3mm_0p1.csv')
OUT_FILE_LIST_0p1 = os.path.join(SUB_DATA_DIR, 'z_files_list_freq_runsz_hartree-unsorted-TS-bxyl-pm3mm_0p1.txt')

# Good Output Files #
GOOD_OUT_FILE_0p05 = os.path.join(SUB_DATA_DIR, 'z_dcluster-sorted-TS-bxyl-pm3mm_good.csv')
BAD_OUT_FILE = os.path.join(SUB_DATA_DIR, 'z_dcluster-sorted-TS-bxyl-pm3mm_bad.csv')
GOOD_OUT_FILE_0p1 = os.path.join(SUB_DATA_DIR, 'z_cluster_z_hartree-unsorted-TS-bxyl-pm3mm_0p1.csv')

# Hartree Headers #
FILE_NAME = 'File Name'


# Tests #

class TestFailWell(unittest.TestCase):
    def testHelp(self):
        test_input = ['-h']
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertEquals(output, 'WARNING:  0\n')
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)

    def testNoSuchFile(self):
        test_input = ["-s", "ghost"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue("Could not find" in output)


class TestDIPOLEFunctions(unittest.TestCase):
    def testReadHartreeSummary(self):
        hartree_dict, pucker_filename_dict, \
        hartree_headers = hartree_sum_pucker_cluster(BXYL_HARTREE_PM3MM_TS_FILE)
        self.assertTrue(pucker_filename_dict[PUCK_2SO], PUCK_2SO_FILES)

    def testHartreeSumPuckerCluster(self):
        try:
            hartree_list, pucker_filename_dict, hartree_headers \
                = hartree_sum_pucker_cluster(BXYL_HARTREE_PM3MM_TS_FILE)
            out_f_name = create_out_fname(SUB_DATA_DIR, prefix='hartree_list_', suffix='_output',
                                          ext='.csv')
            write_csv(hartree_list, out_f_name, hartree_headers, extrasaction="ignore")
            self.assertFalse(diff_lines(out_f_name, BXYL_HARTREE_PM3MM_TS_FILE))
        finally:
            silent_remove(out_f_name)

    def testListToDict(self):
        hartree_list, pucker_filename_dict, hartree_headers \
            = hartree_sum_pucker_cluster(BXYL_HARTREE_PM3MM_TS_FILE)
        hartree_dict = list_to_dict(hartree_list, FILE_NAME)
        len(hartree_dict)
        self.assertEqual(len(hartree_dict), len(hartree_list))

    def testCheckBeforeAfterSortingGood(self):
        try:
            file_unsorted = BXYL_HARTREE_PM3MM_TS_FILE
            file_sorted = GOOD_OUT_FILE_0p05
            list_pucker_missing = check_before_after_sorting(file_unsorted, file_sorted)
        finally:
            self.assertEquals(list_pucker_missing, [])

    def testCheckBeforeAfterSortingBad(self):
        try:
            file_unsorted = BXYL_HARTREE_PM3MM_TS_FILE
            file_sorted = BAD_OUT_FILE
            list_pucker_missing = check_before_after_sorting(file_unsorted, file_sorted)
        finally:
            self.assertEquals(list_pucker_missing, ['4c1'])


class TestMain(unittest.TestCase):
    def testMain0p05(self):
        try:
            test_input = ["-s", BXYL_HARTREE_PM3MM_TS_FILE, "-t", '0.05']
            main(test_input)
            hartree_dict_test = read_csv_to_dict(OUT_FILE, mode='rU')
            hartree_dict_good = read_csv_to_dict(GOOD_OUT_FILE_0p05, mode='rU')

            for row1 in hartree_dict_test:
                for row2 in hartree_dict_good:
                    if len(hartree_dict_test) != len(hartree_dict_good):
                        self.assertEqual(1, 0)
                    else:
                        if row1[FILE_NAME] == row2[FILE_NAME]:
                            self.assertEqual(row1, row2)
        finally:
            silent_remove(OUT_FILE)
            silent_remove(OUT_FILE_LIST_0p05)

    def testMain0p1(self):
        try:
            test_input = ["-s", BXYL_HARTREE_PM3MM_TS_FILE, "-t", '0.1']
            main(test_input)
            hartree_dict_test = read_csv_to_dict(OUT_FILE, mode='rU')
            hartree_dict_good = read_csv_to_dict(GOOD_OUT_FILE_0p1, mode='rU')

            for row1 in hartree_dict_test:
                for row2 in hartree_dict_good:
                    if len(hartree_dict_test) != len(hartree_dict_good):
                        self.assertEqual(1, 0)
                    else:
                        if row1[FILE_NAME] == row2[FILE_NAME]:
                            self.assertEqual(row1, row2)
        finally:
            silent_remove(OUT_FILE)
            silent_remove(OUT_FILE_LIST_0p1)
