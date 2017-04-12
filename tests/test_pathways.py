#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest
from qm_utils.pathways import read_pathway_information, main, perform_pucker_boltzmann_weighting_gibbs, \
    find_hartree_csv_file, id_key_pathways, comparing_pathways_between_methods
from qm_utils.qm_common import diff_lines, silent_remove

__author__ = 'SPVicchio'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'pathways')

# Sample Files#
INPUT_BXYL_PATHWAYS = os.path.join(SUB_DATA_DIR, 'a_pathway_listbxyl_am1.txt')
INPUT_BXYL_CSV      = os.path.join(SUB_DATA_DIR, 'z_pathways-am1.csv')
INPUT_BXYL_LIST_FILES    = os.path.join(SUB_DATA_DIR, 'a_pathway_list_all.txt')

INPUT_BXYL_PATH_B3LYP = os.path.join(SUB_DATA_DIR, 'a_pathway_listbxyl_b3lyp.txt')
INPUT_BXYL_CSV_B3LYP  = os.path.join(SUB_DATA_DIR, 'z_TS-b3lyp.csv')


# Function Inputs #
INPUT_GIBBS_LIST    = [3.357175825, 4.750246915, 4.80672277, 2.77359199, 4.587094445]
INPUT_ENTH_LIST     = [3.884283805, 5.47188284, 5.47188284, 3.244224115, 5.327555655]

# Good Outputs #
GOOD_PATHWAY_DICT_LM1  = 3.0058
GOOD_PATHWAY_DICT_DUPE = '2'
GOOD_WEIGHT            = 3.5717


# Tests #
class TestPathwayFunctions(unittest.TestCase):
    def testReadPathwayInformation(self):
        qm_method, pathway_dict, method_dict, main_dict = read_pathway_information(INPUT_BXYL_PATHWAYS, INPUT_BXYL_CSV)
        self.assertEqual(qm_method, 'am1')
        self.assertEqual(pathway_dict['1c4-b14-os2']['lm1'], GOOD_PATHWAY_DICT_LM1)
        self.assertEqual(pathway_dict['bo3-5ho-1c4']['dupe'],GOOD_PATHWAY_DICT_DUPE)

    def testPerformPuckerBoltzmannWeightingGibbs(self):
        weight = perform_pucker_boltzmann_weighting_gibbs(INPUT_GIBBS_LIST, INPUT_ENTH_LIST)
        self.assertEqual(weight, GOOD_WEIGHT)


    def testComparingPathways(self):
        try:
            hartree_file = find_hartree_csv_file(INPUT_BXYL_PATHWAYS, SUB_DATA_DIR)
            method, pathway_dict, method_dict, main_dict = \
                read_pathway_information(INPUT_BXYL_PATHWAYS, os.path.join(SUB_DATA_DIR, hartree_file))
            path_interest_am1, multiple_pathways_am1 = id_key_pathways(main_dict, method_dict, pucker_interest='4c1')

            hartree_file = find_hartree_csv_file(INPUT_BXYL_PATH_B3LYP, SUB_DATA_DIR)
            method_b3lyp, pathway_dict_b3lyp, method_dict_b3lyp, main_dict_b3lyp = \
            read_pathway_information(INPUT_BXYL_PATH_B3LYP, os.path.join(SUB_DATA_DIR, hartree_file))
            path_interest_b3lyp, multiple_pathways_b3lyp = id_key_pathways(main_dict_b3lyp, method_dict_b3lyp, pucker_interest='4c1')

        finally:
            comparing_pathways_between_methods(path_interest_b3lyp, multiple_pathways_b3lyp, method_dict_b3lyp,
                                               path_interest_am1, multiple_pathways_am1, method_dict)
            print('working')



class TestMain(unittest.TestCase):
    def testMainSingle(self):
        input_info = ["-f", INPUT_BXYL_PATHWAYS, "-d", SUB_DATA_DIR]
        main(input_info)

    def testMainSingleB3LYP(self):
        input_info = ["-f", INPUT_BXYL_PATH_B3LYP, "-d", SUB_DATA_DIR]
        main(input_info)

    def testMainMultiple(self):
        input_info = ["-s", INPUT_BXYL_LIST_FILES, "-d", SUB_DATA_DIR]
        main(input_info)

