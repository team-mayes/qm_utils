#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest
import pandas as pd

from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, create_out_fname, \
    write_csv, list_to_dict, read_csv_to_dict
from qm_utils.spherical_kmeans_voronoi import read_csv_data, spherical_kmeans_voronoi, \
    matplotlib_printing_size_bxyl_lm, matplotlib_printing_normal, read_csv_canonical_designations, \
    organizing_information_from_spherical_kmeans, matplotlib_printing_group_labels, read_csv_data_TS
from qm_utils.xyz_cluster import main, hartree_sum_pucker_cluster, compare_rmsd_xyz, test_clusters, \
    check_ring_ordering, read_ring_atom_ids, check_before_after_sorting

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

TEST_DIR     = os.path.dirname(__file__)
DATA_DIR     = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'spherical_kmeans_voronoi')

# Input files #
HSP_LOCAL_MIN = 'z_lm-b3lyp_howsugarspucker.csv'
HSP_TRANS_STA = 'z_TS-b3lyp_howsugarspucker.csv'
CANO_DESIGN   = 'CP_params.csv'

# Good output
PHI_RAW_GOOD    = [249.1, 327.3, 30.0, 66.5, 44.9, 195.7, 274.2, 272.4, 264.8, 278.3, 113.5, 155.5, 19.0, 17.0, 14.7,
                  51.3, 96.7, 94.4, 45.2, 48.8, 194.7, 5.5, 7.1, 328.3, 339.4, 328.8]
THETA_RAW_GOOD  = [91.8, 177.9, 177.8, 176.6, 177.7, 86.2, 89.4, 89.2, 92.6, 90.8, 91.6, 86.8, 89.9, 90.8, 2.1, 1.3,
                  88.1, 92.3, 86.3, 86.8, 86.2, 89.6, 89.7, 92.4, 90.8, 92.5]
NUMBER_CLUSTERS = 7
SKM_LABELS_GOOD = 'array([4, 2, 2, 2, 2, 3, 4, 4, 4, 4, 0, 3, 1, 1, 5, 5, 0, 0, 1, 1, 3, 1, 1, 6, 6, 6], dtype=int32)'
HSP_LM_DF_GOOD  = os.path.join(SUB_DATA_DIR, 'a_HSP_lm_reference_groups_good.csv')

# noinspection PyUnboundLocalVariable
class TestSphereicalKmeansVoronoi(unittest.TestCase):
    def testReadCSVData(self):
        try:
            data_points, phi_raw, theta_raw = read_csv_data(HSP_LOCAL_MIN, SUB_DATA_DIR)
        finally:
            self.assertEqual(phi_raw, PHI_RAW_GOOD)
            self.assertEqual(theta_raw, THETA_RAW_GOOD)

    def testSphericalKmeansVoronoi(self):
        try:
            data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, SUB_DATA_DIR)
            data_dict = spherical_kmeans_voronoi(7, data_points, phi_raw, theta_raw, energy)
        finally:
            self.assertEqual(data_dict['number_clusters'], NUMBER_CLUSTERS)

    def testImportCanonicalDesignation(self):
        try:
            pucker, phi_cano, theta_cano = read_csv_canonical_designations(CANO_DESIGN, SUB_DATA_DIR)
        finally:
            self.assertEqual(pucker[0], '1c4')

    def testMatplotlibPrinting(self):
        data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, SUB_DATA_DIR)
        data_dict = spherical_kmeans_voronoi(7, data_points, phi_raw, theta_raw, energy)
        matplotlib_printing_normal(data_dict, SUB_DATA_DIR, save_status='no')
        matplotlib_printing_size_bxyl_lm(data_dict, SUB_DATA_DIR, save_status='no')

    def testGroupAssignment(self):
        data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, SUB_DATA_DIR)
        data_dict = spherical_kmeans_voronoi(9, data_points, phi_raw, theta_raw, energy)
        final_groups = organizing_information_from_spherical_kmeans(data_dict)
        matplotlib_printing_group_labels(final_groups, dir_=SUB_DATA_DIR, save_status='off')

class MainRun(unittest.TestCase):
    def testMainRun(self):
        try:
            # Input Parameters #
            number_clusters = 9
            save_status = 'on'
        finally:
            # Running #
            data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, SUB_DATA_DIR)
            data_dict = spherical_kmeans_voronoi(number_clusters, data_points, phi_raw, theta_raw, energy)
            final_groups = organizing_information_from_spherical_kmeans(data_dict)
            df = pd.DataFrame.from_dict(final_groups)
            out_file_name = create_out_fname('a_HSP_lm_reference_groups', base_dir=SUB_DATA_DIR, ext='.csv')
            df.to_csv(out_file_name)

            # Testing #
            self.assertFalse(diff_lines(out_file_name, HSP_LM_DF_GOOD))
            silent_remove(out_file_name)

            # Plotting Commands #
            matplotlib_printing_normal(data_dict, SUB_DATA_DIR, save_status=save_status)
            matplotlib_printing_size_bxyl_lm(data_dict, SUB_DATA_DIR, save_status=save_status)
            matplotlib_printing_group_labels(final_groups, dir_=SUB_DATA_DIR, save_status=save_status)
            print('\n Done \n ')
            pass

    def testMainRunTS(self):
        try:
            # Input Parameters #
            number_cluster = 25
            save_status = 'yes'
            hsp_lm_dict = pd.read_csv(HSP_LM_DF_GOOD)

        finally:
            # Running #
            read_csv_data_TS(HSP_TRANS_STA, SUB_DATA_DIR)
        #     data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_TRANS_STA, SUB_DATA_DIR)
        #     data_dict = spherical_kmeans_voronoi(number_clusters, data_points, phi_raw, theta_raw, energy)
        #     final_ts_groups = organizing_information_from_spherical_kmeans(data_dict)
        #     matplotlib_printing_normal(data_dict, SUB_DATA_DIR, save_status=save_status, voronoi_status='no', ts_status='yes')
        #     print('\n Done \n ')
        #     pass
