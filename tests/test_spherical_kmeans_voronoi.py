#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, create_out_fname, \
    write_csv, list_to_dict, read_csv_to_dict
from qm_utils.spherical_kmeans_voronoi import read_csv_data, spherical_kmeans_voronoi, \
    matplotlib_printing_size, matplotlib_printing_normal, read_csv_canonical_designations
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
CANO_DESIGN   = 'CP_params.csv'

# Good output
PHI_RAW_GOOD    = [249.1, 327.3, 30.0, 66.5, 44.9, 195.7, 274.2, 272.4, 264.8, 278.3, 113.5, 155.5, 19.0, 17.0, 14.7,
                  51.3, 96.7, 94.4, 45.2, 48.8, 194.7, 5.5, 7.1, 328.3, 339.4, 328.8]
THETA_RAW_GOOD  = [91.8, 177.9, 177.8, 176.6, 177.7, 86.2, 89.4, 89.2, 92.6, 90.8, 91.6, 86.8, 89.9, 90.8, 2.1, 1.3,
                  88.1, 92.3, 86.3, 86.8, 86.2, 89.6, 89.7, 92.4, 90.8, 92.5]
NUMBER_CLUSTERS = 7
SKM_LABELS_GOOD = 'array([4, 2, 2, 2, 2, 3, 4, 4, 4, 4, 0, 3, 1, 1, 5, 5, 0, 0, 1, 1, 3, 1, 1, 6, 6, 6], dtype=int32)'

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
        matplotlib_printing_size(data_dict, SUB_DATA_DIR, save_status='no')


class MainRun(unittest.TestCase):
    def testMainRun(self):
        try:
            # Input
            number_clusters = 8
        finally:
            # Running
            data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, SUB_DATA_DIR)
            data_dict = spherical_kmeans_voronoi(number_clusters, data_points, phi_raw, theta_raw, energy)
            # matplotlib_printing_normal(data_dict, SUB_DATA_DIR, save_status='no')
            matplotlib_printing_size(data_dict, SUB_DATA_DIR, save_status='no')
