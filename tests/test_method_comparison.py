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
from qm_utils.spherical_kmeans_voronoi import Transition_States, Local_Minima, read_csv_data, read_csv_data_TS, read_csv_canonical_designations, Plots, Local_Minima_Cano

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
HSP_DATA_DIR = os.path.join(SUB_DATA_DIR_SV, 'images_local_min')

# # # Input files # # #
LIST_OF_DATASET_FILES_BXYL = os.path.join(SUB_DATA_DIR, "a_list_dataset_bxyl.txt")
DATASET_FILE_LM_AM1 = os.path.join(SUB_DATA_DIR, 'z_dataset-bxyl-LM-am1.csv')
HSP_LOCAL_MIN = 'z_lm-b3lyp_howsugarspucker.csv'
DATASET_FILE_LM_HSP = os.path.join(HSP_DATA_DIR, HSP_LOCAL_MIN)
HSP_LOCAL_MIN = 'z_bxyl_lm-b3lyp_howsugarspucker.csv'


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
        # initialization info for local minimum clustering for specific molecule
        number_clusters = 15
        dict_cano = read_csv_canonical_designations('CP_params.csv', SUB_DATA_DIR_SV)
        data_points, phi_raw, theta_raw, energy = read_csv_data(
            'z_aglc_lm-b3lyp_howsugarspucker.csv',
            SUB_DATA_DIR_SV)
        lm_class = Local_Minima(number_clusters, data_points, dict_cano, phi_raw, theta_raw, energy)

        lm_class.plot_all_vor_sec()
        lm_class.plot_local_min_sizes()
        lm_class.plot_cano()

        ts_data_dict = read_csv_data_TS('z_aglc_TS-b3lyp_howsugarspucker.csv',
                                        SUB_DATA_DIR_SV)[3]
        ts_class = Transition_States(ts_data_dict, lm_class)

        ts_class.plot_cano()
        ts_class.plot_all_2d()
        ts_class.show()
