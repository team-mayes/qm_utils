#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest
from qm_utils.spherical_boltzmann_kmeans import SphericalKMeans_boltz


from qm_utils.spherical_kmeans_voronoi import Local_Minima, read_csv_data, read_csv_canonical_designations

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

TEST_DIR     = os.path.dirname(__file__)
DATA_DIR     = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'spherical_kmeans_voronoi')
LOCAL_MIN_IMAGES = os.path.join(SUB_DATA_DIR, 'images_local_min')
TRANS_STA_IMAGES = os.path.join(SUB_DATA_DIR, 'image_transition_state')
TS_PATHWAYS = os.path.join(TRANS_STA_IMAGES, 'pathways')

# Input files #
HSP_LOCAL_MIN = 'z_bxyl_lm-b3lyp_howsugarspucker.csv'
HSP_TRANS_STA = 'z_bxyl_TS-b3lyp_howsugarspucker.csv'
CANO_DESIGN   = 'CP_params.csv'



class MainRun(unittest.TestCase):
    def TestLocMin(self):
        try:
            save_status = True
            storage_spot = LOCAL_MIN_IMAGES
            number_clusters = 9
            data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, SUB_DATA_DIR)
            dict_cano = read_csv_canonical_designations('CP_params.csv', SUB_DATA_DIR)
        finally:
            data = Local_Minima(number_clusters, data_points, dict_cano ,phi_raw, theta_raw, energy)

            skm_boltz = SphericalKMeans_boltz(n_clusters=number_clusters, init='k-means++', n_init=30)
            # skm_boltz.fit(data_points, energy)

            skm_boltz.fit(data_points)

            pass


