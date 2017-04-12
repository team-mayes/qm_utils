#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test structure_pairing
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.qm_common import read_csv_to_dict
from qm_utils.structure_pairing import create_new_cp_params, comparing_across_methods, \
    sorting_for_matching_values, boltzmann_weighting_group, main

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Headers #
GID = 'group ID'



# Directories #
TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'structure_pairing')

# Input files #
FILE_SAMPLE_B3LYP_LM = os.path.join(SUB_DATA_DIR, 'z_lm-b3lyp_howsugarspucker.csv')
FILE_SAMPLE_B3LYP_TS = os.path.join(SUB_DATA_DIR, 'z_TS-b3lyp_howsugarspucler.csv')
FILE_HARTREE_SAMPLE_DFTB_TS = os.path.join(SUB_DATA_DIR, 'z_pathways-dftb.csv')
LIST_OF_CSV_FILES_BXYL = os.path.join(SUB_DATA_DIR, 'a_list_csv_files.txt')

# Good output #


# class TestFailWell(unittest.TestCase):
#     def testHelp(self):
#         test_input = ['-h']
#         if logger.isEnabledFor(logging.DEBUG):
#             main(test_input)
#         with capture_stderr(main, test_input) as output:
#             self.assertEquals(output,'WARNING:  0\n')
#         with capture_stdout(main, test_input) as output:
#             self.assertTrue("optional arguments" in output)
#
#     def testNoSuchFile(self):
#         test_input = ["-s", "ghost"]
#         with capture_stderr(main, test_input) as output:
#             self.assertTrue("Could not find" in output)


# noinspection PyUnboundLocalVariable
class TestStructurePairingFunctions(unittest.TestCase):
    # def testComputeRmsdBetweenPuckers(self):
    #     phi = 37
    #     theta = 91
    #     compute_rmsd_between_puckers(phi, theta)

    def testCreateNewCPParams(self):
        data_dict_lm = read_csv_to_dict(FILE_SAMPLE_B3LYP_TS, mode='r')
        structure_dict_lm, phi_mean_lm, theta_mean_lm = create_new_cp_params(data_dict_lm)

        data_dict_ts = read_csv_to_dict(FILE_SAMPLE_B3LYP_TS, mode='r')
        structure_dict_ts, phi_mean_ts, theta_mean_ts = create_new_cp_params(data_dict_ts)

    def testComparingAcrossMethods(self):
        data_dict_ts = read_csv_to_dict(FILE_SAMPLE_B3LYP_TS, mode='r')
        structure_dict_ts, phi_mean_ts, theta_mean_ts = create_new_cp_params(data_dict_ts)
        method_dict = read_csv_to_dict(FILE_HARTREE_SAMPLE_DFTB_TS, mode='r')

        updated_method_dict, group_file_dict, ungrouped_files = comparing_across_methods(method_dict, structure_dict_ts)
        sorting_for_matching_values(updated_method_dict, print_status='off')

class TestMain(unittest.TestCase):
    def testMainBxyl(self):
        test_input = ["-s", LIST_OF_CSV_FILES_BXYL, "-d", SUB_DATA_DIR, "-m", 'bxyl']
        main(test_input)
