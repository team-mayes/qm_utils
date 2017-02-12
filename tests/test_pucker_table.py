#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest
from qm_utils.pucker_table import read_hartree_files_lowest_energy, sorting_job_types, boltzmann_weighting

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
SAMPLE_HARTREE_FILE_TS = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-TS-oxane-b3lyp.csv')
SAMPLE_HARTREE_FILE_BOTH = os.path.join(SUB_DATA_DIR, 'z_cluster-sorted-ALL-bxyl-am1.csv')

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

    def testBoltzmannWeighting(self):
        hartree_headers, lowest_energy_dict, qm_method = read_hartree_files_lowest_energy(SAMPLE_HARTREE_FILE_BOTH, SUB_DATA_DIR)
        lm_jobs, ts_jobs, qm_method = sorting_job_types(lowest_energy_dict, qm_method)
        boltzmann_weighting(lm_jobs,qm_method)


