#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test structure_pairing
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.method_comparison import main, Local_Minima_Compare
from qm_utils.qm_common import capture_stderr, capture_stdout, read_csv_to_dict
from qm_utils.spherical_kmeans_voronoi import Local_Minima, read_csv_data, read_csv_canonical_designations, Plots, Local_Minima_Cano

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)



# # # Directories # # #
TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'method_comparison')

SUB_DATA_DIR_SV = os.path.join(DATA_DIR, 'spherical_kmeans_voronoi')

# # # Input files # # #
LIST_OF_DATASET_FILES_BXYL = os.path.join(SUB_DATA_DIR, "a_list_dataset_bxyl.txt")
DATASET_FILE_LM_AM1 = os.path.join(SUB_DATA_DIR, 'z_dataset-bxyl-LM-am1.csv')
HSP_LOCAL_MIN = 'z_lm-b3lyp_howsugarspucker.csv'

# # # Good output # # #





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


class TestMain(unittest.TestCase):
    def testMainBxyl(self):
        # test_input = ["-s", LIST_OF_DATASET_FILES_BXYL, "-d", SUB_DATA_DIR, "-m", "bxyl"]
        # main(test_input)

        number_clusters = 9
        data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, SUB_DATA_DIR_SV)
        dict_cano = read_csv_canonical_designations('CP_params.csv', SUB_DATA_DIR_SV)

        lm_class = Local_Minima(number_clusters, data_points, dict_cano, phi_raw, theta_raw, energy)

        lm_class_cano = Local_Minima_Cano(dict_cano)

        method_list_dicts = read_csv_to_dict(DATASET_FILE_LM_AM1, mode='r')

        lm_comp_cano_class = Local_Minima_Compare('AM1', method_list_dicts, lm_class_cano)
        lm_comp_cano_class.save_all_figures()

        # lm_comp_class = Local_Minima_Compare('AM1', method_list_dicts, lm_class)
        # lm_comp_class.save_all_figures()
