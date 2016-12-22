#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.gen_puck_table import read_hartree_files, create_pucker_gibbs_dict, rel_energy_values, \
    creating_level_dict_of_dict, main

__author__ = 'SPVicchio'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'gen_puck_table')

# Sample Files#
SAMPLE_HARTREE_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-optall-oxane-b3lyp.csv')
SAMPLE_HARTREE_FILE_MISSING = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-optall-oxane-am1.csv')
SAMPLE_HARTREE_FILE_TS = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-TS-oxane-am1.csv')
SAMPLE_PUCKER_DICT_NOT_REL = {'4c1': -170479.2753145665, '3s1': -170473.79213655548, 'os2': -170473.80719678348,
                              '5s1': -170473.176549736, '2so': -170473.81786444498, '1s3': -170473.791509046,
                              '1c4': -170479.27594207603, '1s5': -170473.17466720752}
LIST_OF_CSV_FILES = os.path.join(SUB_DATA_DIR, 'a_list_csv_files.txt')


# Good Outputs #

GOOD_PUCKERING_DICT = {'4c1': -170479.2753145665, '1s3': -170473.791509046, '1c4': -170479.27594207603,
                       '3s1': -170473.79213655548, '1s5': -170473.17466720752, '5s1': -170473.176549736,
                       '2so': -170473.81786444498, 'os2': -170473.80719678348}
GOOD_REL_PUCKERING_DICT = {'5s1': 6.1, '1s5': 6.1, '1s3': 5.5, '3s1': 5.5,
                           'os2': 5.5, '2so': 5.5, '4c1': 0.0, '1c4': 0.0}
GOOD_QM_METHOD = 'B3LYP'
GOOD_DICT_OF_DICTS = {'B3LYP': {'2so': 5.5, 'os2': 5.5, '1s5': 6.1, '5s1': 6.1, '1s3': 5.5,
                                '1c4': 0.0, '4c1': 0.0, '3s1': 5.5}}

class TestGenPuckerTableFunctions(unittest.TestCase):
    def testReadHartreeFiles(self):
        hartree_headers, hartree_dict, job_type, qm_method = read_hartree_files(SAMPLE_HARTREE_FILE, SUB_DATA_DIR)
        self.assertEquals(qm_method,'B3LYP')

    def testReadHartreeFilesMissing(self):
        hartree_headers, hartree_dict, job_type, qm_method = read_hartree_files(SAMPLE_HARTREE_FILE_MISSING, SUB_DATA_DIR)
        self.assertEquals(qm_method,'am1')
        self.assertEquals(job_type,'Local Min')

    def testReadHartreeFilesTS(self):
        hartree_headers, hartree_dict, job_type, qm_method = read_hartree_files(SAMPLE_HARTREE_FILE_TS, SUB_DATA_DIR)
        self.assertEquals(job_type,'TS')

    def testCreatingPuckerGibbsDict(self):
        hartree_headers, hartree_dict, job_type, qm_method = read_hartree_files(SAMPLE_HARTREE_FILE, SUB_DATA_DIR)
        puckering_dict, qm_method_n = create_pucker_gibbs_dict(hartree_dict, job_type, qm_method)
        self.assertEquals(puckering_dict,GOOD_PUCKERING_DICT)

    def testRelEnergyValues(self):
        lowest_energy_puckering = rel_energy_values(SAMPLE_PUCKER_DICT_NOT_REL)
        self.assertEquals(lowest_energy_puckering, GOOD_REL_PUCKERING_DICT)

    def testCreatingLevelDictOfDicts(self):
        level_of_theory_dict = creating_level_dict_of_dict(GOOD_REL_PUCKERING_DICT, GOOD_QM_METHOD)
        self.assertEquals(level_of_theory_dict,GOOD_DICT_OF_DICTS)


#    def testMain(self):
#            test_input = ["-s", LIST_OF_CSV_FILES]
#            main(test_input)
