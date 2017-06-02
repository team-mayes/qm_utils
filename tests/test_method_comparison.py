#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test structure_pairing
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.method_comparison import Local_Minima_Compare
from qm_utils.qm_common import capture_stderr, capture_stdout, read_csv_to_dict
from qm_utils.spherical_kmeans_voronoi import Transition_States, Local_Minima, read_csv_data, read_csv_data_TS, read_csv_canonical_designations, Plots, Local_Minima_Cano

__author__ = 'JHuber/SPVicchio'

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

# # # Local_Minima_Compare unit tests # # #
#region


#endregion

# # # Transition_State_Comapre unit tests # # #
#region

#endregion


# # # run unit tests # # #
def main():
    # # # Local_Minima_Compare unit tests # # #
    # region


    # endregion

    # # # Transition_State_Comapre unit tests # # #
    # region

    # endregion

    return
