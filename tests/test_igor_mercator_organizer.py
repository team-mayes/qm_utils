#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_igor_mercator_organizer
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, create_out_fname, \
    write_csv, list_to_dict, read_csv_to_dict
from qm_utils.igor_mercator_organizer import main, reading_all_csv_input_files, creating_dict_of_dict, \
    sorting_dict_of_dict, write_file_data_dict, creating_igor_pathway

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'igor_mercator')
NEW_DATA_DIR = os.path.join(SUB_DATA_DIR, 'new_data')

# Input files #
OXANE_AM1_LM_UNSORTED = os.path.join(SUB_DATA_DIR, 'z_hartree_out-unsorted-lm-oxane-am1.csv')
OXANE_AM1_LMIRC_UNSORTED = os.path.join(SUB_DATA_DIR, 'z_hartree_out-unsorted-lmirc-oxane-am1.csv')
OXANE_AM1_TS_UNSORTED = os.path.join(SUB_DATA_DIR, 'z_hartree_out-unsorted-TS-oxane-am1.csv')
OXANE_AM1_TS_SORTED = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-TS-oxane-am1.csv')
OXANE_AM1_LM_SORTED = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-optall-oxane-am1.csv')
LIST_FILES = [  OXANE_AM1_LM_UNSORTED,
                OXANE_AM1_LMIRC_UNSORTED,
                OXANE_AM1_TS_UNSORTED,
                OXANE_AM1_TS_SORTED,
                OXANE_AM1_LM_SORTED]

BXYL_AM1_LM_SORTED = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-lm-bxyl-am1.csv')
BXYL_AM1_LMIRC_UNSORTED = os.path.join(SUB_DATA_DIR, 'z_hartree-unsorted-lmirc-bxyl-am1.csv')
BXYL_AM1_TS_SORTED = os.path.join(SUB_DATA_DIR, 'z_cluster_ring_pucker-sorted-TS-bxyl-am1.csv')

NEW_OXANE_AM1_LMIRC_UNSORTED = os.path.join(NEW_DATA_DIR, 'z_hartree-unsorted-lmirc-oxane-am1.csv')
NEW_OXANE_AM1_TS_SORTED = os.path.join(NEW_DATA_DIR, 'z_cluster-sorted-TS-oxane-am1.csv')
NEW_OXANE_AM1_LM_SORTED = os.path.join(NEW_DATA_DIR, 'z_cluster-sorted-lm-oxane-am1.csv')

# Good output

OXANE_AM1_LM_UNSORTED_DICT = '1.5081'
OXANE_FINAL_CSV_GOOD = os.path.join(SUB_DATA_DIR, 'igor_df_oxane_am1_good.csv')

# Hartree Headers #
THETA = 'theta'
PHI = 'phi'


# Tests #

# class TestFailWell(unittest.TestCase):
#     def testHelp(self):
#         test_input = ['-h']
#         if logger.isEnabledFor(logging.DEBUG):
#             main(test_input)
#         # with capture_stderr(main, test_input) as output:
#         #     self.assertFalse(output)
#         # with capture_stdout(main, test_input) as output:
#         #     self.assertTrue("optional arguments" in output)


class TestIgorMercator(unittest.TestCase):
    def testReadingAllInputFiles(self):
        method, job_type, sort_status, dict = reading_all_csv_input_files(OXANE_AM1_LM_UNSORTED, 'oxane')
        self.assertEqual(method, 'am1')
        self.assertEqual(job_type,'lm')
        self.assertEqual(sort_status,'unsorted')
        self.assertEqual(dict[1]['dipole'], OXANE_AM1_LM_UNSORTED_DICT)

    def testCreatingDataDict(self):
        dict_of_dicts, method = creating_dict_of_dict(LIST_FILES, 'oxane')
        data_dict = sorting_dict_of_dict(dict_of_dicts)
        csv_file_information = read_csv_to_dict(OXANE_AM1_TS_UNSORTED, mode='r')
        self.assertEqual(data_dict['am1-TS-unsorted-theta'][0], csv_file_information[0][THETA])
        self.assertEqual(data_dict['am1-TS-unsorted-phi'][0],csv_file_information[0][PHI])
        self.assertEqual(data_dict['am1-TS-unsorted-theta'][-1], csv_file_information[-1][THETA])
        self.assertEqual(data_dict['am1-TS-unsorted-phi'][-1],csv_file_information[-1][PHI])

    def testCreatingIgorPathway(self):
        dict_of_dicts, method = creating_dict_of_dict(LIST_FILES, 'oxane')
        creating_igor_pathway(dict_of_dicts)

class TestMain(unittest.TestCase):
    def testMainOxane(self):
        try:
            test_input = [ "-m", 'oxane',
                           "-raw_lm", OXANE_AM1_LM_UNSORTED,
                           "-raw_lmirc", OXANE_AM1_LMIRC_UNSORTED,
                           "-raw_ts",OXANE_AM1_TS_UNSORTED,
                           "-ts", OXANE_AM1_TS_SORTED,
                           "-lm", OXANE_AM1_LM_SORTED]
            main(test_input)
            OXANE_AM1_OUTPUT = os.path.join(SUB_DATA_DIR, 'igor_df_oxane_am1.csv')
        finally:
            self.assertFalse(diff_lines(OXANE_AM1_OUTPUT, OXANE_FINAL_CSV_GOOD))
            silent_remove(OXANE_AM1_OUTPUT)

    def testMainBxyl(self):
        try:
            test_input = [ "-m", 'bxyl',
                           "-raw_lmirc", BXYL_AM1_LMIRC_UNSORTED,
                           "-ts", BXYL_AM1_TS_SORTED,
                           "-lm", BXYL_AM1_LM_SORTED]
            main(test_input)

        finally:
            print('hi')
        #     OXANE_AM1_OUTPUT = os.path.join(SUB_DATA_DIR, 'igor_df_oxane_am1.csv')
        # finally:
        #     self.assertFalse(diff_lines(OXANE_AM1_OUTPUT, OXANE_FINAL_CSV_GOOD))
        #     silent_remove(OXANE_AM1_OUTPUT)

    # def testMainOxane(self):
    #     test_input = [ "-m", 'oxane',
    #                    "-raw_lmirc", NEW_OXANE_AM1_LMIRC_UNSORTED,
    #                    "-ts", NEW_OXANE_AM1_TS_SORTED,
    #                    "-lm", NEW_OXANE_AM1_LM_SORTED]
    #     main(test_input)
