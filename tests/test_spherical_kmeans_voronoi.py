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
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi

from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, create_out_fname, \
    write_csv, list_to_dict, read_csv_to_dict
from qm_utils.spherical_kmeans_voronoi import read_csv_data, spherical_kmeans_voronoi, \
    matplotlib_printing_size_bxyl_lm, matplotlib_printing_normal, read_csv_canonical_designations, \
    organizing_information_from_spherical_kmeans, matplotlib_printing_group_labels, read_csv_data_TS, \
    assign_groups_to_TS_LM, matplotlib_printing_ts_local_min, matplotlib_printing_ts_raw_local_mini, arc_coords, \
    matplotlib_edge_printing, sorting_TS_into_groups, plot_regions, multiple_plots
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

    def testArcCoords(self):
        def plot(vert_1, vert_2):
            mpl.rcParams['legend.fontsize'] = 10

            ax = fig.gca(projection='3d')

            # endpts of the line
            x_0 = vert_1[0]
            y_0 = vert_1[1]
            z_0 = vert_1[2]
            x_f = vert_2[0]
            y_f = vert_2[1]
            z_f = vert_2[2]

            # polar coords to be changed to cartesian
            raw_coords = arc_coords(vert_1, vert_2)

            print(raw_coords)

            # converts the polar coords to cartesian with r = 1
            def get_arc_coord(phi, theta):
                phi = np.deg2rad(phi)
                theta = np.deg2rad(theta)

                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)

                return (x, y, z)

            # initializes the cartesian coordinates for the arclength
            vec_x = [x_0]
            vec_y = [y_0]
            vec_z = [z_0]

            # increments over the raw coords to get cartesian coords
            for i in range(len(raw_coords[0])):
                arc_coord = get_arc_coord(raw_coords[0][i], raw_coords[1][i])

                # pushes coords into the arclength
                vec_x.append(arc_coord[0])
                vec_y.append(arc_coord[1])
                vec_z.append(arc_coord[2])

                i += 1

            # pushes final coord into the arclength
            vec_x.append(x_f)
            vec_y.append(y_f)
            vec_z.append(z_f)

            # plots line
            ax.plot([x_0, x_f], [y_0, y_f], [z_0, z_f], label='parametric line')
            # plots arclength
            ax.plot(vec_x, vec_y, vec_z, label='arclength')
            ax.legend()
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_zlim([-1,1])

            # plots wireframe sphere
            theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
            THETA, PHI = np.meshgrid(theta, phi)
            R = 1.0
            X = R * np.sin(PHI) * np.cos(THETA)
            Y = R * np.sin(PHI) * np.sin(THETA)
            Z = R * np.cos(PHI)
            ax.plot_wireframe(X, Y, Z, color="lightblue")

        # simple case for 2D
        fig = plt.figure()
        plot([0, 0, 1], [0, 1, 0])
        plt.show()

        # edge case for actual data
        fig = plt.figure()
        plot([-0.5911, -0.5402, -0.5990], [0.2574, 0.7223, -.6419])
        plt.show()

        # edge case for actual data
        fig = plt.figure()
        plot([0.3666, -0.6591, -0.6567], [0.2574, 0.7223, -.6419])
        plt.show()

        fig = plt.figure()
        plot([0.1531, 0.7313, 0.6647], [-0.5932, 0.5122, 0.6210])
        plot([-0.6509, 0.4637, -0.6011], [0.2574, 0.7223, -0.6419])
        plot([0.1531, 0.7313, 0.6647], [0.2574, 0.7223, -0.6419])
        plot([-0.6509, 0.4637, -0.6011], [-0.5932, 0.5122, 0.6210])
        plt.show()


    # def testCanonicalDesignationCenters(self):
    #
    #     # Canonical Designations
    #     pucker, phi_cano, theta_cano = read_csv_canonical_designations('CP_params.csv', dir_)
    #
    #     #### TEST PURPOSES ####
    #     cano_centers = []
    #
    #     # converts strings to ints
    #     for i in range(len(phi_cano)):
    #         phi_cano[i] = float(phi_cano[i])
    #         theta_cano[i] = float(theta_cano[i])
    #
    #     # creating cartesian cano_centers
    #     for i in range(len(phi_cano)):
    #         vert_test = pol2cart([phi_cano[i], theta_cano[i], 1])
    #         vert_test = np.asarray(vert_test)
    #
    #         cano_centers.append(vert_test)
    #
    #     # Default parameters for spherical voronoi
    #     radius = 1
    #     center = np.array([0, 0, 0])
    #
    #     cano_centers = np.asarray(cano_centers)
    #
    #     # Spherical Voronoi for the centers
    #
    #     sv_test = SphericalVoronoi(cano_centers, radius, center)
    #     sv_test.sort_vertices_of_regions()
    #     test_dict = {}
    #
    #     test_dict['number_clusters'] = len(phi_cano)
    #     test_dict['vertices_sv_xyz'] = sv_test.vertices
    #     test_dict['regions_sv_labels'] = sv_test.regions
    #
    #     plot_regions(ax_3d, ax, test_dict)
    #
    #     #### TEST PURPOSES ####


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

            # Plotting Commands #
            # matplotlib_printing_normal(data_dict, SUB_DATA_DIR, save_status=save_status)
            # matplotlib_printing_size_bxyl_lm(data_dict, SUB_DATA_DIR, save_status=save_status)
            # matplotlib_printing_group_labels(final_groups, dir_=SUB_DATA_DIR, save_status=save_status)
            matplotlib_edge_printing(data_dict, SUB_DATA_DIR, save_status='no')

            # Testing #
            if number_clusters == 9:
                self.assertFalse(diff_lines(out_file_name, HSP_LM_DF_GOOD))
                silent_remove(out_file_name)
            else:
                silent_remove(out_file_name)
                pass


    def testMainRunTS(self):
        try:
            data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, SUB_DATA_DIR)
            data_dict = spherical_kmeans_voronoi(9, data_points, phi_raw, theta_raw, energy)

            # Input Parameters #
            number_cluster = 25
            save_status = False
            hsp_lm_dict = pd.read_csv(HSP_LM_DF_GOOD)
        finally:
            # Comparing the LM structures #
            data_points_ts, phi_raw_ts, theta_raw_ts, data_dict_ts = read_csv_data_TS(HSP_TRANS_STA, SUB_DATA_DIR)
            assigned_lm, hsp_lm_dict, phi_ts_lm, theta_ts_lm = assign_groups_to_TS_LM(data_dict_ts, hsp_lm_dict)
            # matplotlib_printing_ts_local_min(hsp_lm_dict, phi_ts_lm, theta_ts_lm, data_dict, SUB_DATA_DIR, save_status=save_status)
            # matplotlib_printing_ts_raw_local_mini(hsp_lm_dict, phi_ts_lm, theta_ts_lm, data_dict, SUB_DATA_DIR, save_status=save_status)

            # Grouping the TS #
            sorted_data_dict_ts = sorting_TS_into_groups(number_cluster, data_points_ts, data_dict_ts, phi_raw_ts, theta_raw_ts)
            # matplotlib_printing_normal(sorted_data_dict_ts, SUB_DATA_DIR, save_status=save_status, voronoi_status=False, ts_status=True)


            multiple_plots(sorted_data_dict_ts)

            pass

        #     data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_TRANS_STA, SUB_DATA_DIR)

        #     final_ts_groups = organizing_information_from_spherical_kmeans(data_dict)
        #     matplotlib_printing_normal(data_dict, SUB_DATA_DIR, save_status=save_status, voronoi_status='no', ts_status='yes')
        #     print('\n Done \n ')
        #     pass
