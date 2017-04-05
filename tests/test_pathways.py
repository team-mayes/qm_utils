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
    distance_between_puckers
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
        qm_method, pathway_dict, method_dict = read_pathway_information(INPUT_BXYL_PATHWAYS, INPUT_BXYL_CSV)
        self.assertEqual(qm_method, 'am1')
        self.assertEqual(pathway_dict['1c4-b14-os2']['lm1'], GOOD_PATHWAY_DICT_LM1)
        self.assertEqual(pathway_dict['bo3-5ho-1c4']['dupe'],GOOD_PATHWAY_DICT_DUPE)

    def testPerformPuckerBoltzmannWeightingGibbs(self):
        weight = perform_pucker_boltzmann_weighting_gibbs(INPUT_GIBBS_LIST, INPUT_ENTH_LIST)
        self.assertEqual(weight, GOOD_WEIGHT)

    def testDistanceBetweenPuckersSim(self):
        try:
            qm_method, pathway_dict, method_dict = read_pathway_information(INPUT_BXYL_PATHWAYS, INPUT_BXYL_CSV)
            distance_between_puckers(method_dict['beta-xylose38-TS_am1-ircr-am1.log'],
                                     method_dict['beta-xylose39-TS_am1-ircr-am1.log'])
        finally:
            self.assertEqual(qm_method, 'am1')

    def testDistanceBetweenPuckersDiff(self):
        try:
            qm_method, pathway_dict, method_dict = read_pathway_information(INPUT_BXYL_PATHWAYS, INPUT_BXYL_CSV)
            distance_between_puckers(method_dict['beta-xylose40-TS_am1-ircr-am1.log'],
                                     method_dict['beta-xylose68-TS_am1-ircf-am1.log']) # bo3, os2
        finally:
            self.assertEqual(qm_method, 'am1')




class TestMain(unittest.TestCase):
    def testMainSingle(self):
        input_info = ["-f", INPUT_BXYL_PATHWAYS, "-d", SUB_DATA_DIR]
        main(input_info)

    def testMainMultiple(self):
        input_info = ["-s", INPUT_BXYL_LIST_FILES, "-d", SUB_DATA_DIR]
        main(input_info)

