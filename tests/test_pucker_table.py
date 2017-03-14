#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest
from qm_utils.pucker_table import read_hartree_files_lowest_energy, sorting_job_types, boltzmann_weighting, main

__author__ = 'SPVicchio'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'gen_puck_table')

# Sample Files #
SAMPLE_HARTREE_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-optall-oxane-b3lyp.csv')
SAMPLE_HARTREE_FILE_TS = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-TS-oxane-b3lyp.csv')
SAMPLE_HARTREE_FILE_BOTH = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-ALL-bxyl-am1.csv')
LIST_OF_CSV_FILES_BXYL = os.path.join(SUB_DATA_DIR, 'a_list_csv_files2.txt')
LIST_OF_CSV_FILES_OXANE = os.path.join(SUB_DATA_DIR, 'a_list_csv_files_oxane_ALL.txt')
ENERGY_FILE = os.path.join(SUB_DATA_DIR, 'z_energies_CCSDT-B3LYP_bxyl.csv')


# GOOD OUTPUTS #
GOOD_LM_CONTRIBUTION_DICT = {'b25': 4.53, 'bo3': 1.64, 'os2': 2.86, '4c1': 0.88, '1s3': 3.75, '1c4': 0.0,
                             '25b': 1.48, '2so': 0.89, 'o3b': 2.23, '1s5': 2.22}

GOOD_TS_CONTRIBUTION_DICT = {'e1': 8.41, 'oh1': 7.11, 'o3b': 2.98, '1c4': 0.64, '3s1': 6.39, '14b': 4.96, 'b14': 7.91,
                             '5e': 5.5, '4e': 5.12, '1s5': 3.97, '4h3': 4.67, 'b25': 4.26, '2so': 6.46, '4h5': 6.75,
                             '3h2': 5.35, '25b': 4.05, 'e5': 5.79, '1h2': 5.06, '5ho': 6.7, 'os2': 2.85,
                             'bo3': 1.98, '1s3': 5.38}

## Hartree Headers ##

DIPOLE = "dipole"


# Tests #

class TestGenPuckerTableFunctions(unittest.TestCase):
    def testReadHartreeFiles(self):
        hartree_headers, lowest_energy_dict, qm_method = read_hartree_files_lowest_energy(SAMPLE_HARTREE_FILE, SUB_DATA_DIR)
        self.assertEquals(qm_method,'b3lyp')
        self.assertEqual(lowest_energy_dict[0][DIPOLE],'1.6414')

    def testSortingJobTypesLMOnly(self):
        hartree_headers, lowest_energy_dict, qm_method = read_hartree_files_lowest_energy(SAMPLE_HARTREE_FILE, SUB_DATA_DIR)
        lm_jobs, ts_jobs, qm_method = sorting_job_types(lowest_energy_dict, qm_method)
        self.assertEquals(qm_method,'b3lyp')
        self.assertFalse(ts_jobs)

    def testSortingJobTypesTSOnly(self):
        hartree_headers, lowest_energy_dict, qm_method = read_hartree_files_lowest_energy(SAMPLE_HARTREE_FILE_TS, SUB_DATA_DIR)
        lm_jobs, ts_jobs, qm_method = sorting_job_types(lowest_energy_dict, qm_method)
        self.assertEquals(qm_method,'b3lyp')
        self.assertFalse(lm_jobs)

    def testSortingJobTypesBoth(self):
        hartree_headers, lowest_energy_dict, qm_method = read_hartree_files_lowest_energy(SAMPLE_HARTREE_FILE_BOTH, SUB_DATA_DIR)
        lm_jobs, ts_jobs, qm_method = sorting_job_types(lowest_energy_dict, qm_method)
        self.assertEqual(qm_method,'am1')
        self.assertEqual(len(lm_jobs),15)
        self.assertEqual(len(ts_jobs), 32)

    def testBoltzmannWeightingLM(self):
        hartree_headers, lowest_energy_dict, qm_method = read_hartree_files_lowest_energy(SAMPLE_HARTREE_FILE_BOTH, SUB_DATA_DIR)
        lm_jobs, ts_jobs, qm_method = sorting_job_types(lowest_energy_dict, qm_method)
        contribution_dict, qm_method = boltzmann_weighting(lm_jobs, qm_method)
        self.assertEqual(contribution_dict, GOOD_LM_CONTRIBUTION_DICT)

    def testBoltzmannWeightingTS(self):
        hartree_headers, lowest_energy_dict, qm_method = read_hartree_files_lowest_energy(SAMPLE_HARTREE_FILE_BOTH, SUB_DATA_DIR)
        lm_jobs, ts_jobs, qm_method = sorting_job_types(lowest_energy_dict, qm_method)
        contribution_dict, qm_method = boltzmann_weighting(ts_jobs, qm_method)
        self.assertEqual(contribution_dict, GOOD_TS_CONTRIBUTION_DICT)

class TestMain(unittest.TestCase):
    def testMainBxyl(self):
        test_input = ["-s", LIST_OF_CSV_FILES_BXYL, "-d", SUB_DATA_DIR, "-m", 'bxyl', "-c", ENERGY_FILE]
        main(test_input)

    def testMainOxane(self):
        test_input = ["-s", LIST_OF_CSV_FILES_OXANE, "-d", SUB_DATA_DIR, "-m", 'oxane']
        main(test_input)
