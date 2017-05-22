#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test structure_pairing
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.igor_mercator_organizer import write_file_data_dict
from qm_utils.qm_common import read_csv_to_dict, create_out_fname, diff_lines
import qm_utils.structure_pairing
from qm_utils.reference_structures import local_min_reference_points

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Headers #
GID = 'group ID'


# # Updated CP Params # #

BXYL_LM_PARAMS = {
    'group_11': {'phi': [328.3, 328.8], 'mean phi': 328.55, 'Boltz Weight Gibbs': 8.3503, 'files': ['puck24', 'puck26'],
                 'G298 (Hartrees)': 8.36, 'closest group puck': ['os2'], 'mean theta': 92.45,
                 'Boltz Weight Enth': 8.5705,
                 'theta': [92.4, 92.5]},
    'group_00': {'phi': [14.7, 51.3], 'mean phi': 33.0, 'Boltz Weight Gibbs': 0.1643, 'files': ['puck15', 'puck16'],
                 'G298 (Hartrees)': 0.16, 'closest group puck': ['4c1'], 'mean theta': 1.7, 'Boltz Weight Enth': 0.1643,
                 'theta': [2.1, 1.3]},
    'group_12': {'phi': [339.4], 'mean phi': 339.4, 'Boltz Weight Gibbs': 8.6, 'files': ['puck25'],
                 'G298 (Hartrees)': 8.6,
                 'closest group puck': ['os2'], 'mean theta': 90.8, 'Boltz Weight Enth': 8.4, 'theta': [90.8]},
    'group_03': {'phi': [45.2, 48.8], 'mean phi': 47.0, 'Boltz Weight Gibbs': 9.5833, 'files': ['puck19', 'puck20'],
                 'G298 (Hartrees)': 9.58, 'closest group puck': ['b14'], 'mean theta': 86.55,
                 'Boltz Weight Enth': 10.5249,
                 'theta': [86.3, 86.8]},
    'group_04': {'phi': [96.7, 94.4], 'mean phi': 95.55, 'Boltz Weight Gibbs': 8.0615, 'files': ['puck17', 'puck18'],
                 'G298 (Hartrees)': 8.06, 'closest group puck': ['5s1'], 'mean theta': 90.2,
                 'Boltz Weight Enth': 8.7512,
                 'theta': [88.1, 92.3]},
    'group_06': {'phi': [155.5], 'mean phi': 155.5, 'Boltz Weight Gibbs': 3.9, 'files': ['puck12'],
                 'G298 (Hartrees)': 3.9,
                 'closest group puck': ['2so'], 'mean theta': 86.8, 'Boltz Weight Enth': 4.2, 'theta': [86.8]},
    'group_08': {'phi': [249.1], 'mean phi': 249.1, 'Boltz Weight Gibbs': 11.0, 'files': ['puck1'],
                 'G298 (Hartrees)': 11.0, 'closest group puck': ['14b'], 'mean theta': 91.8, 'Boltz Weight Enth': 12.4,
                 'theta': [91.8]},
    'group_05': {'phi': [113.5], 'mean phi': 113.5, 'Boltz Weight Gibbs': 7.9, 'files': ['puck11'],
                 'G298 (Hartrees)': 7.9,
                 'closest group puck': ['25b'], 'mean theta': 91.6, 'Boltz Weight Enth': 8.8, 'theta': [91.6]},
    'group_01': {'phi': [5.5, 7.1], 'mean phi': 6.3, 'Boltz Weight Gibbs': 7.8833, 'files': ['puck22', 'puck23'],
                 'G298 (Hartrees)': 7.88, 'closest group puck': ['o3b'], 'mean theta': 89.65,
                 'Boltz Weight Enth': 8.0416,
                 'theta': [89.6, 89.7]},
    'group_07': {'phi': [195.7, 194.7], 'mean phi': 195.2, 'Boltz Weight Gibbs': 5.9458, 'files': ['puck6', 'puck21'],
                 'G298 (Hartrees)': 5.95, 'closest group puck': ['1s3'], 'mean theta': 86.2,
                 'Boltz Weight Enth': 6.8542,
                 'theta': [86.2, 86.2]},
    'group_13': {'phi': [327.3, 30.0, 66.5, 44.9], 'mean phi': 117.175, 'Boltz Weight Gibbs': 2.4546,
                 'files': ['puck2', 'puck3', 'puck4', 'puck5'], 'G298 (Hartrees)': 2.45, 'closest group puck': ['1c4'],
                 'mean theta': 177.5, 'Boltz Weight Enth': 1.5752, 'theta': [177.9, 177.8, 176.6, 177.7]},
    'group_09': {'phi': [264.8], 'mean phi': 264.8, 'Boltz Weight Gibbs': 7.9, 'files': ['puck9'],
                 'G298 (Hartrees)': 7.9,
                 'closest group puck': ['1s5'], 'mean theta': 92.6, 'Boltz Weight Enth': 8.3, 'theta': [92.6]},
    'group_02': {'phi': [19.0, 17.0], 'mean phi': 18.0, 'Boltz Weight Gibbs': 7.8128, 'files': ['puck13', 'puck14'],
                 'G298 (Hartrees)': 7.82, 'closest group puck': ['3s1'], 'mean theta': 90.35,
                 'Boltz Weight Enth': 7.7504,
                 'theta': [89.9, 90.8]},
    'group_10': {'phi': [274.2, 272.4, 278.3], 'mean phi': 274.966667, 'Boltz Weight Gibbs': 7.4681,
                 'files': ['puck7', 'puck8', 'puck10'], 'G298 (Hartrees)': 7.47, 'closest group puck': ['1s5'],
                 'mean theta': 89.8, 'Boltz Weight Enth': 7.7383, 'theta': [89.4, 89.2, 90.8]}}

BXYL_TS_PARAMS = {
    'group_39': {'phi': [327.9, 332.3], 'mean phi': 330.1, 'Boltz Weight Enth': 11.8752, 'group_lm2': ['group_02'],
                 'files': ['puck45', 'puck46'], 'Boltz Weight Gibbs': 12.8128, 'closest group puck': ['3h2'],
                 'mean theta': 115.25, 'group_lm1': ['group_13'], 'theta': [116.5, 114.0]},
    'group_11': {'phi': [327.2], 'mean phi': 327.2, 'Boltz Weight Enth': 16.4, 'group_lm2': ['group_11'],
                 'files': ['puck88'], 'Boltz Weight Gibbs': 16.3, 'closest group puck': ['oh5'], 'mean theta': 63.6,
                 'group_lm1': ['group_00'], 'theta': [63.6]},
    'group_33': {'phi': [231.5], 'mean phi': 231.5, 'Boltz Weight Enth': 15.3, 'group_lm2': ['group_08'],
                 'files': ['puck36'], 'Boltz Weight Gibbs': 15.9, 'closest group puck': ['1e'], 'mean theta': 123.4,
                 'group_lm1': ['group_13'], 'theta': [123.4]},
    'group_25': {'phi': [315.7], 'mean phi': 315.7, 'Boltz Weight Enth': 9.3, 'group_lm2': ['group_01'],
                 'files': ['puck90'], 'Boltz Weight Gibbs': 10.2, 'closest group puck': ['os2'], 'mean theta': 90.6,
                 'group_lm1': ['group_10'], 'theta': [90.6]},
    'group_18': {'phi': [235.9, 243.1], 'mean phi': 239.5, 'Boltz Weight Enth': 9.6349, 'group_lm2': ['group_06'],
                 'files': ['puck28', 'puck27'], 'Boltz Weight Gibbs': 10.4349, 'closest group puck': ['14b'],
                 'theta': [84.5, 84.1], 'group_lm1': ['group_10'], 'mean theta': 84.3},
    'group_32': {'phi': [145.3, 142.7], 'mean phi': 144.0, 'Boltz Weight Enth': 14.9851, 'group_lm2': ['group_13'],
                 'files': ['puck66', 'puck67'], 'Boltz Weight Gibbs': 15.5645, 'closest group puck': ['5ho'],
                 'theta': [124.7, 127.2], 'group_lm1': ['group_06'], 'mean theta': 125.95},
    'group_26': {'phi': [35.6, 25.7], 'mean phi': 30.65, 'Boltz Weight Enth': 14.0663, 'group_lm2': ['group_13'],
                 'files': ['puck52', 'puck53'], 'Boltz Weight Gibbs': 14.3598, 'closest group puck': ['3h4'],
                 'mean theta': 116.95, 'group_lm1': ['group_06'], 'theta': [116.6, 117.3]},
    'group_08': {'phi': [206.0, 196.1, 199.4], 'mean phi': 200.5, 'Boltz Weight Enth': 10.7411,
                 'group_lm2': ['group_00'],
                 'files': ['puck54', 'puck55', 'puck56'], 'Boltz Weight Gibbs': 10.8697, 'closest group puck': ['4h3'],
                 'mean theta': 54.366667, 'group_lm1': ['group_06'], 'theta': [54.6, 54.6, 53.9]},
    'group_21': {'phi': [314.5], 'mean phi': 314.5, 'Boltz Weight Enth': 9.8, 'group_lm2': ['group_02'],
                 'files': ['puck69'], 'Boltz Weight Gibbs': 10.7, 'closest group puck': ['b25'], 'mean theta': 90.3,
                 'group_lm1': ['group_10'], 'theta': [90.3]},
    'group_05': {'phi': [148.6], 'mean phi': 148.6, 'Boltz Weight Enth': 10.2, 'group_lm2': ['group_06'],
                 'files': ['puck42'], 'Boltz Weight Gibbs': 10.2, 'closest group puck': ['2h3'], 'mean theta': 62.0,
                 'group_lm1': ['group_00'], 'theta': [62.0]},
    'group_00': {'phi': [12.3], 'mean phi': 12.3, 'Boltz Weight Enth': 15.9, 'group_lm2': ['group_02'],
                 'files': ['puck81'], 'Boltz Weight Gibbs': 16.1, 'closest group puck': ['oe'], 'mean theta': 61.6,
                 'group_lm1': ['group_00'], 'theta': [61.6]},
    'group_10': {'phi': [333.4], 'mean phi': 333.4, 'Boltz Weight Enth': 16.3, 'group_lm2': ['group_00'],
                 'files': ['puck89'], 'Boltz Weight Gibbs': 16.4, 'closest group puck': ['oh5'], 'mean theta': 61.1,
                 'group_lm1': ['group_01'], 'theta': [61.1]},
    'group_17': {'phi': [237.5], 'mean phi': 237.5, 'Boltz Weight Enth': 12.2, 'group_lm2': ['group_07'],
                 'files': ['puck29'], 'Boltz Weight Gibbs': 12.6, 'closest group puck': ['14b'], 'mean theta': 94.0,
                 'group_lm1': ['group_09'], 'theta': [94.0]},
    'group_13': {'phi': [1.4], 'mean phi': 1.4, 'Boltz Weight Enth': 9.4, 'group_lm2': ['group_02'],
                 'files': ['puck79'],
                 'Boltz Weight Gibbs': 10.9, 'closest group puck': ['o3b'], 'mean theta': 89.2,
                 'group_lm1': ['group_12'],
                 'theta': [89.2]},
    'group_09': {'phi': [290.4, 280.4], 'mean phi': 285.4, 'Boltz Weight Enth': 14.7643, 'group_lm2': ['group_00'],
                 'files': ['puck78', 'puck57'], 'Boltz Weight Gibbs': 14.7643, 'closest group puck': ['e5'],
                 'theta': [63.0, 63.4], 'group_lm1': ['group_10'], 'mean theta': 63.2},
    'group_02': {'phi': [29.3, 38.3, 31.6], 'mean phi': 33.0667, 'Boltz Weight Enth': 14.2176,
                 'group_lm2': ['group_00'],
                 'files': ['puck84', 'puck85', 'puck82'], 'Boltz Weight Gibbs': 14.6437, 'closest group puck': ['oh1'],
                 'theta': [65.5, 62.1, 62.3], 'group_lm1': ['group_02'], 'mean theta': 63.3},
    'group_16': {'phi': [184.1], 'mean phi': 184.1, 'Boltz Weight Enth': 9.8, 'group_lm2': ['group_06'],
                 'files': ['puck73'], 'Boltz Weight Gibbs': 10.5, 'closest group puck': ['bo3'], 'mean theta': 86.9,
                 'group_lm1': ['group_07'], 'theta': [86.9]},
    'group_35': {'phi': [255.0, 264.5, 257.0], 'mean phi': 258.8333, 'Boltz Weight Enth': 11.1275,
                 'group_lm2': ['group_09'], 'files': ['puck32', 'puck39', 'puck38'], 'Boltz Weight Gibbs': 12.098,
                 'closest group puck': ['1h2'], 'theta': [121.3, 120.7, 120.8], 'group_lm1': ['group_13'],
                 'mean theta': 120.9333},
    'group_06': {'phi': [136.6], 'mean phi': 136.6, 'Boltz Weight Enth': 13.6, 'group_lm2': ['group_05'],
                 'files': ['puck43'], 'Boltz Weight Gibbs': 13.3, 'closest group puck': ['2h3'], 'mean theta': 60.0,
                 'group_lm1': ['group_00'], 'theta': [60.0]},
    'group_31': {'phi': [135.9], 'mean phi': 135.9, 'Boltz Weight Enth': 16.5, 'group_lm2': ['group_13'],
                 'files': ['puck65'], 'Boltz Weight Gibbs': 16.6, 'closest group puck': ['5e'], 'mean theta': 118.4,
                 'group_lm1': ['group_07'], 'theta': [118.4]},
    'group_30': {'phi': [112.2, 113.3], 'mean phi': 112.75, 'Boltz Weight Enth': 14.4663, 'group_lm2': ['group_13'],
                 'files': ['puck62', 'puck63'], 'Boltz Weight Gibbs': 14.8349, 'closest group puck': ['5e'],
                 'theta': [124.7, 125.4], 'group_lm1': ['group_04'], 'mean theta': 125.05},
    'group_01': {'phi': [31.6, 49.7, 37.2], 'mean phi': 39.5, 'Boltz Weight Enth': 14.3115, 'group_lm2': ['group_00'],
                 'files': ['puck86', 'puck76', 'puck83'], 'Boltz Weight Gibbs': 14.586, 'closest group puck': ['oh1'],
                 'theta': [65.0, 63.2, 61.6], 'group_lm1': ['group_03'], 'mean theta': 63.2667},
    'group_20': {'phi': [239.0], 'mean phi': 239.0, 'Boltz Weight Enth': 15.5, 'group_lm2': ['group_07'],
                 'files': ['puck31'], 'Boltz Weight Gibbs': 15.6, 'closest group puck': ['14b'], 'mean theta': 88.3,
                 'group_lm1': ['group_11'], 'theta': [88.3]},
    'group_34': {'phi': [225.4, 228.8, 224.6, 227.9], 'mean phi': 226.675, 'Boltz Weight Enth': 13.7426,
                 'group_lm2': ['group_07'], 'files': ['puck33', 'puck34', 'puck40', 'puck35'],
                 'Boltz Weight Gibbs': 14.4194, 'closest group puck': ['1e'], 'theta': [123.6, 123.5, 126.0, 126.4],
                 'group_lm1': ['group_13'], 'mean theta': 124.875},
    'group_04': {'phi': [46.0, 48.4], 'mean phi': 47.2, 'Boltz Weight Enth': 14.4598, 'group_lm2': ['group_04'],
                 'files': ['puck74', 'puck75'], 'Boltz Weight Gibbs': 14.5598, 'closest group puck': ['e1'],
                 'theta': [63.1, 62.2], 'group_lm1': ['group_00'], 'mean theta': 62.65},
    'group_03': {'phi': [43.0], 'mean phi': 43.0, 'Boltz Weight Enth': 16.5, 'group_lm2': ['group_01'],
                 'files': ['puck87'], 'Boltz Weight Gibbs': 16.9, 'closest group puck': ['e1'], 'mean theta': 65.0,
                 'group_lm1': ['group_00'], 'theta': [65.0]},
    'group_24': {'phi': [291.7], 'mean phi': 291.7, 'Boltz Weight Enth': 11.2, 'group_lm2': ['group_10'],
                 'files': ['puck72'], 'Boltz Weight Gibbs': 12.3, 'closest group puck': ['b25'], 'mean theta': 91.0,
                 'group_lm1': ['group_11'], 'theta': [91.0]},
    'group_07': {'phi': [174.1], 'mean phi': 174.1, 'Boltz Weight Enth': 8.6, 'group_lm2': ['group_06'],
                 'files': ['puck77'], 'Boltz Weight Gibbs': 8.6, 'closest group puck': ['e3'], 'mean theta': 57.2,
                 'group_lm1': ['group_00'], 'theta': [57.2]},
    'group_19': {'phi': [237.7], 'mean phi': 237.7, 'Boltz Weight Enth': 16.2, 'group_lm2': ['group_06'],
                 'files': ['puck30'], 'Boltz Weight Gibbs': 16.0, 'closest group puck': ['14b'], 'mean theta': 86.6,
                 'group_lm1': ['group_11'], 'theta': [86.6]},
    'group_36': {'phi': [280.0], 'mean phi': 280.0, 'Boltz Weight Enth': 13.1, 'group_lm2': ['group_13'],
                 'files': ['puck37'], 'Boltz Weight Gibbs': 13.5, 'closest group puck': ['1h2'], 'mean theta': 125.9,
                 'group_lm1': ['group_10'], 'theta': [125.9]},
    'group_14': {'phi': [59.3], 'mean phi': 59.3, 'Boltz Weight Enth': 12.7, 'group_lm2': ['group_02'],
                 'files': ['puck68'], 'Boltz Weight Gibbs': 13.0, 'closest group puck': ['b14'], 'mean theta': 87.7,
                 'group_lm1': ['group_04'], 'theta': [87.7]},
    'group_22': {'phi': [303.2], 'mean phi': 303.2, 'Boltz Weight Enth': 10.4, 'group_lm2': ['group_10'],
                 'files': ['puck71'], 'Boltz Weight Gibbs': 11.4, 'closest group puck': ['b25'], 'mean theta': 92.5,
                 'group_lm1': ['group_01'], 'theta': [92.5]},
    'group_27': {'phi': [35.7], 'mean phi': 35.7, 'Boltz Weight Enth': 12.5, 'group_lm2': ['group_13'],
                 'files': ['puck51'], 'Boltz Weight Gibbs': 13.0, 'closest group puck': ['3h4'], 'mean theta': 115.0,
                 'group_lm1': ['group_02'], 'theta': [115.0]},
    'group_37': {'phi': [330.9, 331.5], 'mean phi': 331.2, 'Boltz Weight Enth': 11.493, 'group_lm2': ['group_01'],
                 'files': ['puck44', 'puck47'], 'Boltz Weight Gibbs': 12.5598, 'closest group puck': ['3h2'],
                 'mean theta': 115.25, 'group_lm1': ['group_13'], 'theta': [114.9, 115.6]},
    'group_15': {'phi': [134.4], 'mean phi': 134.4, 'Boltz Weight Enth': 9.3, 'group_lm2': ['group_04'],
                 'files': ['puck41'], 'Boltz Weight Gibbs': 8.9, 'closest group puck': ['25b'], 'mean theta': 86.4,
                 'group_lm1': ['group_06'], 'theta': [86.4]},
    'group_29': {'phi': [113.9], 'mean phi': 113.9, 'Boltz Weight Enth': 14.8, 'group_lm2': ['group_05'],
                 'files': ['puck64'], 'Boltz Weight Gibbs': 15.5, 'closest group puck': ['5e'], 'mean theta': 126.7,
                 'group_lm1': ['group_13'], 'theta': [126.7]},
    'group_12': {'phi': [5.8], 'mean phi': 5.8, 'Boltz Weight Enth': 11.7, 'group_lm2': ['group_03'],
                 'files': ['puck80'],
                 'Boltz Weight Gibbs': 12.7, 'closest group puck': ['o3b'], 'mean theta': 89.8,
                 'group_lm1': ['group_11'],
                 'theta': [89.8]},
    'group_38': {'phi': [328.3, 327.8, 331.6], 'mean phi': 329.233333, 'Boltz Weight Enth': 12.8611,
                 'group_lm2': ['group_13'], 'files': ['puck48', 'puck49', 'puck50'], 'Boltz Weight Gibbs': 13.5661,
                 'closest group puck': ['3h2'], 'mean theta': 120.033333, 'group_lm1': ['group_11'],
                 'theta': [120.3, 119.8, 120.0]},
    'group_23': {'phi': [302.6], 'mean phi': 302.6, 'Boltz Weight Enth': 10.4, 'group_lm2': ['group_10'],
                 'files': ['puck70'], 'Boltz Weight Gibbs': 11.5, 'closest group puck': ['b25'], 'mean theta': 92.9,
                 'group_lm1': ['group_02'], 'theta': [92.9]},
    'group_28': {'phi': [120.6, 126.3, 120.8, 123.1], 'mean phi': 122.7, 'Boltz Weight Enth': 11.2457,
                 'group_lm2': ['group_13'], 'files': ['puck60', 'puck58', 'puck59', 'puck61'],
                 'Boltz Weight Gibbs': 12.099, 'closest group puck': ['5e'], 'theta': [126.7, 125.2, 125.8, 121.0],
                 'group_lm1': ['group_06'], 'mean theta': 124.675}}


# Directories #
TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'structure_pairing')

# Input files #
FILE_SAMPLE_B3LYP_LM = os.path.join(SUB_DATA_DIR, 'z_bxyl_lm-b3lyp_howsugarspucker.csv')
FILE_SAMPLE_B3LYP_TS = os.path.join(SUB_DATA_DIR, 'z_bxyl_TS-b3lyp_howsugarspucker.csv')
FILE_HARTREE_SAMPLE_DFTB_TS = os.path.join(SUB_DATA_DIR, 'z_pathways-dftb.csv')
FILE_HARTREE_SAMPLE_DFTB_ALL = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-ALL-bxyl-dftb.csv')
LIST_OF_CSV_FILES_BXYL = os.path.join(SUB_DATA_DIR, 'a_list_csv_files.txt')

# Good output #
GOOD_STRUCTURE_DICT_TS       = 14.46
GOOD_COMPARING_TS_ARC_LENGTH = {'group_34': 0.06504834517869058, 'group_33': 0.11535002302291417}
GOOD_TS_PATHWAYS             = os.path.join(SUB_DATA_DIR, 'igor_info_testing_DFTB_good.csv')
# GOOD_UPDATED_METHOD_DICT =
# GOOD_GROUP_FILE_DICT     =



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
    #     phi = 150
    #     theta = 51
    #     top_pucker, difference = qm_utils.structure_pairing.compute_rmsd_between_puckers(phi, theta)
    #     self.assertEqual(top_pucker, '2h3')
    #     self.assertEqual(difference, 0.0)


    def testSeparatingTSandIRCInformation(self):
        try:
            method_dict = read_csv_to_dict(FILE_HARTREE_SAMPLE_DFTB_ALL, mode='r')
            lm_dict, ts_dict = qm_utils.structure_pairing.separating_TS_and_IRC_information(method_dict)
        finally:
            self.assertEqual(len(lm_dict), 196)
            self.assertEqual(len(ts_dict), 53)

    def testCreatingLMUniqueStructures(self):
        try:
            method_dict = read_csv_to_dict(FILE_HARTREE_SAMPLE_DFTB_ALL, mode='r')
            lm_dict, ts_dict = qm_utils.structure_pairing.separating_TS_and_IRC_information(method_dict)
            new_ts_dict = qm_utils.structure_pairing.comparing_TS_structures_arc_length(ts_dict, BXYL_TS_PARAMS)
        finally:
            self.assertEqual(new_ts_dict[4]['TS_compare_values'], GOOD_COMPARING_TS_ARC_LENGTH)

    def testComparingTSPathways(self):
        try:
            method_dict = read_csv_to_dict(FILE_HARTREE_SAMPLE_DFTB_ALL, mode='r')
            lm_dict, ts_dict = qm_utils.structure_pairing.separating_TS_and_IRC_information(method_dict)
            new_ts_dict = qm_utils.structure_pairing.comparing_TS_structures_arc_length(ts_dict, BXYL_TS_PARAMS)
            dict_one, matching_ts_dict, missing_ts_dict = qm_utils.structure_pairing.comparing_TS_pathways(new_ts_dict, lm_dict, BXYL_TS_PARAMS, BXYL_LM_PARAMS)
            data_dict = qm_utils.structure_pairing.generate_matching_ts_dict_pathways_full(matching_ts_dict)
            output_filename_pathway = create_out_fname('igor_info_' + 'testing' + '_' + str('DFTB'), base_dir=SUB_DATA_DIR, ext='.csv')
            write_file_data_dict(data_dict, output_filename_pathway)


            qm_utils.structure_pairing.grouping_and_weighting_TS(matching_ts_dict, 'testing')


            # polar_dict = qm_utils.structure_pairing.generate_igor_hemi_plots(matching_ts_dict, BXYL_LM_PARAMS)
            #
            # output_filename_polar = create_out_fname('igor_polar_' + 'testing' + '_' + str('DFTB'), base_dir=MET_COMP_DIR, ext='.csv')
            # write_file_data_dict(polar_dict, output_filename_polar)

            new_lm_dict, hsp_puckering = qm_utils.structure_pairing.comparing_LM_structures_arc_length(lm_dict, BXYL_LM_PARAMS)

            contribution_dict, qm_method = qm_utils.structure_pairing.boltzmann_weighting_group(new_lm_dict, 'testing')


        finally:
            self.assertFalse(diff_lines(output_filename_pathway,GOOD_TS_PATHWAYS))



    # def testComparingAcrossMethods(self):
            #     try:
            #         method_dict = read_csv_to_dict(FILE_HARTREE_SAMPLE_DFTB_TS, mode='r')
            #         updated_method_dict, group_file_dict, ungrouped_files = qm_utils.structure_pairing.comparing_across_methods(method_dict, BXYL_TS_PARAMS)
            #         phi_values_good, theta_values_good, phi_values_ungrouped, theta_values_ungrouped = qm_utils.structure_pairing.sorting_for_matching_values(updated_method_dict, print_status='off')
            #     finally:
            #         pass

class TestMain(unittest.TestCase):
    def testMainBxyl(self):
        test_input = ["-s", LIST_OF_CSV_FILES_BXYL, "-d", SUB_DATA_DIR, "-m", 'bxyl']
        qm_utils.structure_pairing.main(test_input)
