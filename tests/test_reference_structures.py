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
from qm_utils.reference_structures import local_min_reference_points, assign_TS_localmin_reference_groups, \
    transition_state_reference_points, igor_pathway_creator, analyze_ts_structure_dict, group_meaningful_organizor, \
    boltzmann_weighting_reference

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Headers #
WEIGHT_GIBBS = 'Boltz Weight Gibbs'
WEIGHT_ENTH = 'Boltz Weight Enth'

# Directories #
TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'reference_structures')

# Input files #
FILE_SAMPLE_B3LYP_LM = os.path.join(SUB_DATA_DIR, 'z_lm-b3lyp_howsugarspucker.csv')
FILE_SAMPLE_B3LYP_TS = os.path.join(SUB_DATA_DIR, 'z_TS-b3lyp_howsugarspucker.csv')

# Good Output #

GOOD_LM_GROUP00 = {'theta': [2.1, 1.3], 'mean theta': 1.7, 'mean phi': 33.0, 'G298 (Hartrees)': 0.16, 'files': ['puck15', 'puck16'], 'closest group puck': ['4c1'], 'phi': [14.7, 51.3]}
GOOD_TS_GROUP13 = {'Boltz Weight Gibbs': 10.9, 'Boltz Weight Enth': 9.4}
# Tests #

class TestReferenceStructuresFunctions(unittest.TestCase):

    def testReferenceStructurePairingLM(self):
        try:
            data_dict_lm = read_csv_to_dict(FILE_SAMPLE_B3LYP_LM, mode='r')
            structure_dict_lm, phi_mean_lm, theta_mean_lm = local_min_reference_points(data_dict_lm)
            final_structure_dict_lm = group_meaningful_organizor(structure_dict_lm)
        finally:
            self.assertEqual(final_structure_dict_lm['group_00'], GOOD_LM_GROUP00)

    def testReferenceStructurePairingTS(self):
        try:
            data_dict_lm = read_csv_to_dict(FILE_SAMPLE_B3LYP_LM, mode='r')
            structure_dict_lm, phi_mean_lm, theta_mean_lm = local_min_reference_points(data_dict_lm)
            final_structure_dict_lm = group_meaningful_organizor(structure_dict_lm)
            data_dict_ts = read_csv_to_dict(FILE_SAMPLE_B3LYP_TS, mode='r')
            updated_ts_data_dict = assign_TS_localmin_reference_groups(data_dict_ts, final_structure_dict_lm)

            structure_dict, phi_mean, theta_mean = transition_state_reference_points(updated_ts_data_dict)
            structure_dict_ts, phi_ts_mean, theta_ts_mean = analyze_ts_structure_dict(structure_dict)
            final_structure_dict_ts = group_meaningful_organizor(structure_dict_ts)
            pathway_dict = igor_pathway_creator(final_structure_dict_lm, final_structure_dict_ts)

            final_weighted_lm = boltzmann_weighting_reference(final_structure_dict_lm, data_dict_lm, 'HPS')
            final_weighted_ts = boltzmann_weighting_reference(final_structure_dict_ts, data_dict_ts, 'HPS')

            data_dict = {}
            data_dict['HPS_phi_lm'] = phi_mean_lm
            data_dict['HPS_theta_lm'] = theta_mean_lm
            data_dict['HPS_phi_ts'] = phi_ts_mean
            data_dict['HPS_theta_ts'] = theta_ts_mean

            output_filename_pathway = create_out_fname('igor_pathway_' + str('bxyl') + '_' + str('HPS'), base_dir=SUB_DATA_DIR, ext='.csv')
            output_filename = create_out_fname('igor_df_' + str('bxyl') + '_' + str('HPS'), base_dir=SUB_DATA_DIR, ext='.csv')

            write_file_data_dict(pathway_dict, output_filename_pathway)
            write_file_data_dict(data_dict,output_filename)

        finally:
            self.assertEqual(final_structure_dict_ts['group_13'][WEIGHT_GIBBS], GOOD_TS_GROUP13[WEIGHT_GIBBS])
            self.assertEqual(final_structure_dict_ts['group_13'][WEIGHT_ENTH], GOOD_TS_GROUP13[WEIGHT_ENTH])
