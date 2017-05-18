#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os

from qm_utils.spherical_kmeans_voronoi import Transition_States, Local_Minima, read_csv_data, \
    read_csv_canonical_designations, read_csv_data_TS, Plots

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

TEST_DIR     = os.path.dirname(__file__)
LOCAL_MIN_IMAGES = os.path.join(TEST_DIR, 'images_local_min')
TRANS_STA_IMAGES = os.path.join(TEST_DIR, 'image_transition_state')
TS_PATHWAYS = os.path.join(TRANS_STA_IMAGES, 'pathways')

# Input files #
HSP_LOCAL_MIN = 'z_lm-b3lyp_howsugarspucker.csv'
HSP_TRANS_STA = 'z_TS-b3lyp_howsugarspucker.csv'
CANO_DESIGN   = 'CP_params.csv'

print('\nStarting to run script...\n')

# # # Local Min # # #
# try:
#     save_status = False
#     storage_spot = LOCAL_MIN_IMAGES
#     number_clusters = 9
#     data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, TEST_DIR)
#     dict_cano = read_csv_canonical_designations('CP_params.csv', TEST_DIR)
# finally:
#     data = Local_Minima(number_clusters, data_points, dict_cano ,phi_raw, theta_raw, energy)
#     data.plot_local_min(directory=storage_spot, save_status=save_status)
#     data.plot_group_labels(directory=storage_spot, save_status=save_status)
#     data.plot_local_min_sizes(directory=storage_spot, save_status=save_status)



# # # Transition States # # #
number_clusters = 9
data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, TEST_DIR)
dict_cano = read_csv_canonical_designations('CP_params.csv', TEST_DIR)
data = Local_Minima(number_clusters, data_points, dict_cano, phi_raw, theta_raw, energy)
try:
    save_status = False
    storage_spot = TRANS_STA_IMAGES
    # loading in the local minima information
    data_points_ts, phi_raw_ts, theta_raw_ts, data_dict_ts = read_csv_data_TS(HSP_TRANS_STA, TEST_DIR)

finally:
    ts_class = Transition_States(data_dict_ts, data)

    plot_test = Plots()
    ax_rect = plot_test.ax_rect
    ax_circ = plot_test.ax_circ_north
    ax_spher = plot_test.ax_spher

    ts_class.plot_all_2d(ax_rect, ax_circ)
    ts_class.plot_all_3d(ax_spher)

    # ts_class.plot_loc_min_group_2d(ax_rect, ax_circ, '00_02')
    # ts_class.plot_loc_min_group_3d(ax_spher, '00_02')

    plot_test.show()




