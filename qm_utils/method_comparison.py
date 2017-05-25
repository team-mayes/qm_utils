#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this script is to make comparisons for a particular QM method to the reference set of HSP.
"""

# # # import # # #
from __future__ import print_function

import argparse
import os
import statistics as st
import sys

import csv
import pandas as pd
import math
import numpy as np

from prettytable import PrettyTable
from collections import OrderedDict
from operator import itemgetter

import matplotlib.pyplot as plt

from qm_utils.igor_mercator_organizer import write_file_data_dict
from qm_utils.pucker_table import read_hartree_files_lowest_energy, sorting_job_types

from qm_utils.qm_common import (GOOD_RET, create_out_fname, warning, IO_ERROR, InvalidDataError, INVALID_DATA,
                                INPUT_ERROR, arc_length_calculator, read_csv_to_dict)
from qm_utils.spherical_kmeans_voronoi import Local_Minima, Transition_States,\
                                              read_csv_canonical_designations, read_csv_data, read_csv_data_TS,\
                                              pol2cart, plot_on_circle, plot_line, plot_arc

# # # Header Stuff # # #
#region
try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

# # Default Parameters # #
HARTREE_TO_KCALMOL = 627.5095
TOL_ARC_LENGTH = 0.1
TOL_ARC_LENGTH_CROSS = 0.2  # THIS WAS THE ORGINAL TOLERANCE6
DEFAULT_TEMPERATURE = 298.15
K_B = 0.001985877534  # Boltzmann Constant in kcal/mol K
#endregion

# # Pucker Keys # #
#region
FILE_NAME = 'File Name'
PUCKER = 'Pucker'
ENERGY_ELECTRONIC = 'Energy (A.U.)'
THETA = 'theta'
PHI = 'phi'
Q_VAL = 'Q'
GIBBS = 'G298 (Hartrees)'
ENTH = "H298 (Hartrees)"
MPHI = 'mean phi'
MTHETA = 'mean theta'
GID = 'group ID'
WEIGHT_GIBBS = 'Boltz Weight Gibbs'
WEIGHT_ENTH = 'Boltz Weight Enth'
FREQ = 'Freq 1'
#endregion

# # # Directories # # #
#region
QM_1_DIR = os.path.dirname(__file__)

# root of project
QM_0_DIR = os.path.dirname(QM_1_DIR)

TEST_DIR = os.path.join(QM_0_DIR, 'tests')
TEST_DATA_DIR = os.path.join(TEST_DIR, 'test_data')

MET_COMP_DIR = os.path.join(TEST_DATA_DIR, 'method_comparison')
MOL_DIR = os.path.join(MET_COMP_DIR, 'bxyl')
LM_DIR = os.path.join(MOL_DIR, 'local_minimum')

SV_DIR = os.path.join(TEST_DATA_DIR, 'spherical_kmeans_voronoi')
#endregion

# # # Classes # # #
#region
class Local_Minima_Compare():
    """
    class for organizing the local minima information
    """
    def __init__(self, method_in, lm_dataset_in, lm_class_in, lm_dir_in):
        self.hartree_data = []
        self.lm_class = lm_class_in
        self.group_data = []
        self.overall_data = {}
        self.overall_data['method'] = method_in
        self.group_rows = []
        self.lm_dir = lm_dir_in
        self.lm_dataset = lm_dataset_in

        self.fix_hartrees()
        self.populate_hartree_data()
        self.populate_groupings()
        self.do_calcs()

    # # # __init__ functions # # #
    #region
    def fix_hartrees(self):
        # converting hartrees to kcal/mol
        for i in range(len(self.lm_dataset)):
            self.lm_dataset[i]['G298 (Hartrees)'] = 627.509 * float(self.lm_dataset[i]['G298 (Hartrees)'])

        min_G298 = self.lm_dataset[0]['G298 (Hartrees)']

        for i in range(len(self.lm_dataset)):
            if self.lm_dataset[i]['G298 (Hartrees)'] < min_G298:
                min_G298 = self.lm_dataset[i]['G298 (Hartrees)']

        for i in range(len(self.lm_dataset)):
            self.lm_dataset[i]['G298 (Hartrees)'] -= min_G298

    def populate_hartree_data(self):
        for i in range(len(self.lm_dataset)):
            self.hartree_data.append({})

            self.hartree_data[i]['G298 (Hartrees)'] = float(self.lm_dataset[i]['G298 (Hartrees)'])
            self.hartree_data[i]['pucker'] = self.lm_dataset[i]['Pucker']
            self.hartree_data[i]['phi'] = float(self.lm_dataset[i]['phi'])
            self.hartree_data[i]['theta'] = float(self.lm_dataset[i]['theta'])

            # list for 3 shortest arclengths and their lm_groups
            arc_lengths = {}

            har_phi = float(self.hartree_data[i]['phi'])
            har_theta = float(self.hartree_data[i]['theta'])

            for j in range(len(self.lm_class.sv_kmeans_dict['regions_sv_labels'])):
                skm_phi = self.lm_class.sv_kmeans_dict['phi_skm_centers'][j]
                skm_theta = self.lm_class.sv_kmeans_dict['theta_skm_centers'][j]

                arc_lengths[j] = arc_length_calculator(har_phi, har_theta, skm_phi, skm_theta)

            ordered_arc_lengths = OrderedDict(sorted(arc_lengths.items(), key=itemgetter(1), reverse=False))
            ordered_list = []
            three_shortest_list = []

            for key, val in ordered_arc_lengths.items():
                ordered_list.append([key, val])

            for k in range(3):
                three_shortest_list.append(ordered_list[k])

            self.hartree_data[i]['arc_lengths'] = three_shortest_list

        return

    def populate_groupings(self):
        for i in range(len(self.lm_class.sv_kmeans_dict['regions_sv_labels'])):
            self.group_data.append({})
            self.group_data[i]['method'] = self.overall_data['method']
            self.group_data[i]['points'] = {}

            self.group_data[i]['name'] = self.lm_class.groups_dict[i]['name']

            for j in range(len(self.hartree_data)):
                if self.hartree_data[j]['arc_lengths'][0][0] == i:
                    self.group_data[i]['points'][j] = self.hartree_data[j]

        return
    #endregion

    # # # do_calc functions # # #
    #region
    def do_calcs(self):
        for i in range(len(self.group_data)):
            self.calc_WSS(i)
            self.calc_weighting(i)
            self.calc_WWSS(i)
            self.calc_group_RMSD(i)
            self.calc_group_WRMSD(i)

        self.calc_SSE()
        self.calc_WSSE()
        self.calc_RMSD()
        self.calc_WRMSD()

    # finds Boltzmann weighted Gibb's free energy
    def calc_weighting(self, group):
        total_boltz = 0

        for key in self.group_data[group]['points']:
            e_val = self.group_data[group]['points'][key]['G298 (Hartrees)']
            component = math.exp(-e_val / (K_B * DEFAULT_TEMPERATURE))
            self.group_data[group]['points'][key]['ind_boltz'] = component
            total_boltz += component

        wt_gibbs = 0
        for key in self.group_data[group]['points']:
            if self.group_data[group]['points'][key]['ind_boltz'] == 0:
                wt_gibbs += 0
                self.group_data[group]['points'][key]['weighting'] = 0
            else:
                wt_gibbs += (self.group_data[group]['points'][key]['ind_boltz'] / total_boltz) * self.group_data[group]['points'][key]['G298 (Hartrees)']
                self.group_data[group]['points'][key]['weighting'] = self.group_data[group]['points'][key]['ind_boltz'] / total_boltz

        self.group_data[group]['weighted_gibbs'] = round(wt_gibbs, 3)

    def calc_WSS(self, group):
        WSS = 0

        for key in self.group_data[group]['points']:
            arc_length = self.group_data[group]['points'][key]['arc_lengths'][0][1]
            WSS += arc_length**2

        self.group_data[group]['WSS'] = round(WSS, 5)

    def calc_WWSS(self, group):
        WWSS = 0

        for key in self.group_data[group]['points']:
            arc_length = self.group_data[group]['points'][key]['arc_lengths'][0][1]
            weighting = self.group_data[group]['points'][key]['weighting']
            WWSS += (arc_length ** 2) * weighting

        self.group_data[group]['WWSS'] = round(WWSS, 5)

    def calc_group_RMSD(self, group):
        size = len(self.group_data[group]['points'])
        if(size == 0):
            RMSD = 'n/a'
            self.group_data[group]['group_RMSD'] = RMSD
        else:
            RMSD = (self.group_data[group]['WSS'] / size) ** 0.5
            self.group_data[group]['group_RMSD'] = round(RMSD, 5)

    def calc_group_WRMSD(self, group):
        size = len(self.group_data[group]['points'])

        if (size == 0):
            WRMSD = 'n/a'
            self.group_data[group]['group_WRMSD'] = WRMSD
        else:
            WRMSD = (self.group_data[group]['WWSS'] / size) ** 0.5
            self.group_data[group]['group_WRMSD'] = round(WRMSD, 5)

    def calc_SSE(self):
        SSE = 0

        for i in range(len(self.group_data)):
            SSE += self.group_data[i]['WSS']

        self.overall_data['SSE'] = round(SSE, 5)

    def calc_WSSE(self):
        WSSE = 0

        for i in range(len(self.group_data)):
            WSSE += self.group_data[i]['WWSS']

        self.overall_data['WSSE'] = round(WSSE, 5)

    def calc_RMSD(self):
        RMSD = (self.overall_data['SSE'] / len(self.group_data)) ** 0.5
        self.overall_data['RMSD'] = round(RMSD, 5)

    def calc_WRMSD(self):
        WRMSD = (self.overall_data['WSSE'] / len(self.group_data)) ** 0.5
        self.overall_data['WRMSD'] = round(WRMSD, 5)
    #endregion

    # # # plotting functions # # #
    #region
    def plot_grouping(self, grouping):
        phi = []
        theta = []

        for key in self.group_data[grouping]['points']:
            phi.append(self.group_data[grouping]['points'][key]['phi'])
            theta.append(self.group_data[grouping]['points'][key]['theta'])

        group_phi = self.lm_class.sv_kmeans_dict['phi_skm_centers'][grouping]
        group_theta = self.lm_class.sv_kmeans_dict['theta_skm_centers'][grouping]

        self.lm_class.plot.ax_rect.scatter(group_phi, group_theta, s=30, c='red', marker='o', edgecolor='face', zorder=10)
        self.lm_class.plot.ax_rect.scatter(phi, theta, s=15, c='blue', marker='o', edgecolor='face', zorder = 10)


        self.lm_class.plot_vor_sec(grouping)

        return


    def plot_method_data_raw(self, grouping):
        phi = []
        theta = []

        for key in self.group_data[grouping]['points']:
            phi.append(self.group_data[grouping]['points'][key]['phi'])
            theta.append(self.group_data[grouping]['points'][key]['theta'])

        self.lm_class.plot.ax_rect.scatter(phi, theta, s=15, c='blue', marker='o', edgecolor='face', zorder = 10)
        self.lm_class.plot_vor_sec(grouping)


    def plot_groupings_raw(self, key):
        phi = []
        theta = []

        phi.append(self.lm_class.groups_dict[key]['phi'])
        theta.append(self.lm_class.groups_dict[key]['theta'])

        self.lm_class.plot.ax_rect.scatter(phi, theta, s=60, c='black', marker='o', edgecolor='face', zorder=10)


    def plot_window(self, grouping):
        border = 5

        indexes = self.lm_class.sv_kmeans_dict['regions_sv_labels'][grouping]

        min_phi = self.lm_class.sv_kmeans_dict['phi_sv_vertices'][indexes[0]]
        max_phi = self.lm_class.sv_kmeans_dict['phi_sv_vertices'][indexes[0]]
        min_theta = self.lm_class.sv_kmeans_dict['theta_sv_vertices'][indexes[0]]
        max_theta = self.lm_class.sv_kmeans_dict['theta_sv_vertices'][indexes[0]]

        for i in range(len(indexes)):
            phi = self.lm_class.sv_kmeans_dict['phi_sv_vertices'][indexes[i]]
            theta = self.lm_class.sv_kmeans_dict['theta_sv_vertices'][indexes[i]]

            if phi < min_phi:
                min_phi = phi
            elif phi > max_phi:
                max_phi = phi

            if theta < min_theta:
                min_theta = theta
            elif theta > max_theta:
                max_theta = theta

        min_phi -= border
        max_phi += border

        min_theta -= border
        max_theta += border

        if grouping == 0:
            min_phi = -border
            max_phi = 360 + border

            min_theta = -border

        if grouping == len(self.lm_class.sv_kmeans_dict['regions_sv_labels']) - 1:
            min_phi = -border
            max_phi = 360 + border

            max_theta = 180 + border

        self.lm_class.plot.ax_rect.set_xlim([min_phi, max_phi])
        self.lm_class.plot.ax_rect.set_ylim([max_theta, min_theta])

        self.plot_all_groupings()

        return


    def plot_all_groupings(self):
        for i in range(len(self.group_data)):
            self.plot_grouping(i)

        self.lm_class.plot_group_names()


    def plot_method_data(self):
        for i in range(len(self.group_data)):
            self.plot_method_data_raw(i)


    def plot_all_groupings_raw(self):
        for key in self.lm_class.groups_dict.keys():
            self.plot_groupings_raw(key)
        return


    def set_title_and_legend(self, artist_list, label_list):
        self.lm_class.plot.ax_rect.legend(artist_list,
                                          label_list,
                                          scatterpoints=1, fontsize=8, frameon=False, framealpha=0.75,
                                          bbox_to_anchor=(0.5, -0.15), loc=9, borderaxespad=0, ncol=4).set_zorder(100)

        plt.title(self.overall_data['method'], loc='left')


    def show(self):
        self.lm_class.show()
    #endregion

    # # # saving functions # # #
    #region
    def save_all_figures(self, mol_name):
        # Create custom artist
        size_scaling = 1
        ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30*size_scaling, c='red', marker='o', edgecolor='face')
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=15*size_scaling, c='blue', marker='o', edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=45*size_scaling, c='black', marker='+', edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')

        artist_list = [ref_lm_Artist, met_lm_Artist, path_Artist, cano_lm_Artist]
        label_list = ['Reference LM', 'Method LM', 'Voronoi Edge', 'Canonical Designation']

        base_name = "z_dataset-" + mol_name + "-LM-" + self.overall_data['method']

        if not os.path.exists(os.path.join(self.lm_dir, self.overall_data['method'])):
            os.makedirs(os.path.join(self.lm_dir, self.overall_data['method']))

        met_data_dir = os.path.join(self.lm_dir, self.overall_data['method'])

        overall_dir = os.path.join(met_data_dir, 'overall')
        # checks if directory exists, and creates it if not
        if not os.path.exists(overall_dir):
            os.makedirs(overall_dir)

        # saves a plot of all groupings
        self.plot_all_groupings()
        self.lm_class.plot_cano()

        self.set_title_and_legend(artist_list, label_list)

        self.lm_class.plot.save(base_name + '-all_groupings', overall_dir)
        self.lm_class.wipe_plot()

        for i in range(len(self.group_data)):
            # saves a plot of each group individually plotted
            self.plot_grouping(i)
            self.lm_class.plot_cano()
            # checks if directory exists, and creates it if not
            if not os.path.exists(os.path.join(met_data_dir, 'groups')):
                os.makedirs(os.path.join(met_data_dir, 'groups'))

            groups_dir = os.path.join(met_data_dir, 'groups')

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name + '-group_' + str(i), groups_dir)
            self.lm_class.wipe_plot()

            # # saves a plot of a focused view of each group
            # self.plot_window(i)
            # self.lm_class.plot_cano()
            # WINDOWED_DIR = os.path.join(MET_DATA_DIR, 'groups_windowed')
            # # checks if directory exists, and creates it if not
            # if not os.path.exists(WINDOWED_DIR):
            #     os.makedirs(WINDOWED_DIR)
            #
            # self.set_title_and_legend(artist_list, label_list)
            #
            # self.lm_class.plot.save(base_name + '-group_' + str(i) + '-windowed', WINDOWED_DIR)
            # self.lm_class.wipe_plot()

    def save_all_figures_raw(self, mol_name):
        # Create custom artists
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=15*size_scaling, c='blue', marker='o', edgecolor='face')
        raw_ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60*size_scaling, c='black', marker='o', edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60*size_scaling, c='black', marker='+', edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')

        artist_list = [raw_ref_lm_Artist, met_lm_Artist, path_Artist, cano_lm_Artist]
        label_list = ['Raw Reference LM', 'Method LM', 'Voronoi Edge', 'Canonical Designation']

        base_name = "z_dataset-" + mol_name + "-LM-" + self.overall_data['method']

        if not os.path.exists(os.path.join(self.lm_dir, self.overall_data['method'])):
            os.makedirs(os.path.join(self.lm_dir, self.overall_data['method']))

        met_data_dir = os.path.join(self.lm_dir, self.overall_data['method'])

        overall_dir = os.path.join(met_data_dir, 'overall')
        # checks if directory exists, and creates it if not
        if not os.path.exists(overall_dir):
            os.makedirs(overall_dir)

        # saves plot of all groupings with the raw group data
        # self.plot_all_groupings_raw()
        self.plot_method_data()
        self.lm_class.plot_cano()

        self.set_title_and_legend(artist_list, label_list)

        self.lm_class.plot.save(base_name + '-all_method_raw_data', overall_dir)
        self.lm_class.wipe_plot()
    #endregion

    def arb(self):
        return

class Transition_State_Compare():
    """
    class for organizing the transition state information
    """
    def  __init__(self, method_in, ts_dataset_in, lm_class_in, ts_class_in, ts_dir_in):
        self.lm_class = lm_class_in
        self.ts_class = ts_class_in
        self.ts_dataset = ts_dataset_in
        self.method = method_in

        self.ts_dir = ts_dir_in

        self.hartree_data = []
        self.path_group_data = {}
        self.ref_path_group_data = {}
        self.overall_data = {}
        self.overall_data['method'] = self.method

        self.fix_hartrees()
        self.populate_hartree_data()
        self.populate_ref_path_group_data()
        self.populate_path_group_data()
        self.populate_ts_groups()

        self.do_calcs()
        self.assign_closest_puckers()
        self.assign_group_name()

        self.circ_groups_init()

    # # # __init__ functions # # #
    # region
    def fix_hartrees(self):
        # converting hartrees to kcal/mol
        for i in range(len(self.ts_dataset)):
            self.ts_dataset[i]['G298 (Hartrees)'] = 627.509 * float(self.ts_dataset[i]['G298 (Hartrees)'])

        min_G298 = self.ts_dataset[0]['G298 (Hartrees)']

        for i in range(len(self.ts_dataset)):
            if self.ts_dataset[i]['G298 (Hartrees)'] < min_G298:
                min_G298 = self.ts_dataset[i]['G298 (Hartrees)']

        for i in range(len(self.ts_dataset)):
            self.ts_dataset[i]['G298 (Hartrees)'] -= min_G298

    def populate_hartree_data(self):
        for i in range(len(self.ts_dataset)):
            self.hartree_data.append({})

            self.hartree_data[i]['G298 (Hartrees)'] = float(self.ts_dataset[i]['G298 (Hartrees)'])
            self.hartree_data[i]['pucker'] = self.ts_dataset[i]['Pucker']
            self.hartree_data[i]['phi'] = float(self.ts_dataset[i]['phi'])
            self.hartree_data[i]['theta'] = float(self.ts_dataset[i]['theta'])
            self.hartree_data[i]['lm1'] = {}
            self.hartree_data[i]['lm2'] = {}
            self.hartree_data[i]['lm1']['phi'] = float(self.ts_dataset[i]['phi_lm1'])
            self.hartree_data[i]['lm1']['theta'] = float(self.ts_dataset[i]['theta_lm1'])
            self.hartree_data[i]['lm2']['phi'] = float(self.ts_dataset[i]['phi_lm2'])
            self.hartree_data[i]['lm2']['theta'] = float(self.ts_dataset[i]['theta_lm2'])

            # # # assigning lm groups # # #
            #region
            # list for 3 shortest arclengths and their lm_groups
            arc_lengths_lm1 = {}
            arc_lengths_lm2 = {}

            lm1_phi = self.hartree_data[i]['lm1']['phi']
            lm1_theta = self.hartree_data[i]['lm1']['theta']

            lm2_phi = self.hartree_data[i]['lm2']['phi']
            lm2_theta =self.hartree_data[i]['lm2']['theta']

            # calculate the closest ref groups
            for j in range(len(self.lm_class.sv_kmeans_dict['regions_sv_labels'])):
                skm_phi = self.lm_class.sv_kmeans_dict['phi_skm_centers'][j]
                skm_theta = self.lm_class.sv_kmeans_dict['theta_skm_centers'][j]

                arc_lengths_lm1[j] = arc_length_calculator(lm1_phi, lm1_theta, skm_phi, skm_theta)
                arc_lengths_lm2[j] = arc_length_calculator(lm2_phi, lm2_theta, skm_phi, skm_theta)

            ordered_arc_lengths_lm1 = OrderedDict(sorted(arc_lengths_lm1.items(), key=itemgetter(1), reverse=False))
            ordered_list_lm1 = []
            three_shortest_list_lm1 = []

            ordered_arc_lengths_lm2 = OrderedDict(sorted(arc_lengths_lm2.items(), key=itemgetter(1), reverse=False))
            ordered_list_lm2 = []
            three_shortest_list_lm2 = []

            for key, val in ordered_arc_lengths_lm1.items():
                ordered_list_lm1.append([key, val])
            for k in range(3):
                three_shortest_list_lm1.append(ordered_list_lm1[k])
            self.hartree_data[i]['lm1']['arc_lengths'] = three_shortest_list_lm1
            self.hartree_data[i]['lm1']['group'] = self.hartree_data[i]['lm1']['arc_lengths'][0][0]

            for key, val in ordered_arc_lengths_lm2.items():
                ordered_list_lm2.append([key, val])
            for k in range(3):
                three_shortest_list_lm2.append(ordered_list_lm2[k])
            self.hartree_data[i]['lm2']['arc_lengths'] = three_shortest_list_lm2
            self.hartree_data[i]['lm2']['group'] = self.hartree_data[i]['lm2']['arc_lengths'][0][0]
            #endregion

    def populate_ref_path_group_data(self):
        for key in self.ts_class.ts_groups:
            ref_key = str(int(key.split('_')[0])) + '_' + str(int(key.split('_')[1]))

            if ref_key not in self.ref_path_group_data:
                self.ref_path_group_data[ref_key] = []

            for i in range(len(self.ts_class.ts_groups[key])):
                ref_data = {}

                ref_data['name'] = self.ts_class.ts_groups[key][i]['name']

                ref_data['phi'] = self.ts_class.ts_groups[key][i]['ts_vert_pol'][0]
                ref_data['theta'] = self.ts_class.ts_groups[key][i]['ts_vert_pol'][1]
                ref_data['lm1'] = {}
                ref_data['lm2'] = {}
                ref_data['lm1']['phi'] = self.lm_class.groups_dict[int(key.split('_')[0])]['mean_phi']
                ref_data['lm1']['theta'] = self.lm_class.groups_dict[int(key.split('_')[0])]['mean_theta']
                ref_data['lm2']['phi'] = self.lm_class.groups_dict[int(key.split('_')[1])]['mean_phi']
                ref_data['lm2']['theta'] = self.lm_class.groups_dict[int(key.split('_')[1])]['mean_theta']

                ref_data['G298 (Hartrees)'] = self.ts_class.ts_groups[key][i]['G298 (Hartrees)']

                self.ref_path_group_data[ref_key].append(ref_data)

        return

    def populate_path_group_data(self):
        for i in range(len(self.hartree_data)):
            first = self.hartree_data[i]['lm1']['group']
            second = self.hartree_data[i]['lm2']['group']

            if first < second:
                key = str(first) + '_' + str(second)
            else:
                key = str(second) + '_' + str(first)

            if key not in self.path_group_data:
                self.path_group_data[key] = []

            self.path_group_data[key].append(self.hartree_data[i])

    def populate_ts_groups(self):
        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                self.ref_path_group_data[key][i]['points'] = []
                self.ref_path_group_data[key][i]['WSS'] = 0
                self.ref_path_group_data[key][i]['group_RMSD'] = 'n/a'

        for key in self.path_group_data:
            if key in self.ref_path_group_data:
                for i in range(len(self.path_group_data[key])):
                    # list for shortest arclengths
                    arc_lengths = {}

                    ts_phi = float(self.hartree_data[i]['phi'])
                    ts_theta = float(self.hartree_data[i]['theta'])

                    for j in range(len(self.ref_path_group_data[key])):
                        ref_phi = self.ref_path_group_data[key][j]['phi']
                        ref_theta = self.ref_path_group_data[key][j]['theta']

                        arc_lengths[j] = arc_length_calculator(ts_phi, ts_theta, ref_phi, ref_theta)

                    ordered_arc_lengths = OrderedDict(sorted(arc_lengths.items(), key=itemgetter(1), reverse=False))
                    ordered_list = []

                    for arc_key, val in ordered_arc_lengths.items():
                        ordered_list.append([arc_key, val])

                    self.path_group_data[key][i]['arc_lengths'] = ordered_list

                for i in range(len(self.ref_path_group_data[key])):
                    for j in range(len(self.path_group_data[key])):
                        if self.path_group_data[key][j]['arc_lengths'][0][0] == i:
                            self.ref_path_group_data[key][i]['points'].append(self.path_group_data[key][j])

        return

    def assign_closest_puckers(self):
        for group_key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[group_key])):
                for j in range(len(self.ref_path_group_data[group_key][i]['points'])):
                    # list for 3 shortest arclengths and their lm_groups
                    arc_lengths = {}

                    ts_phi = float(self.ref_path_group_data[group_key][i]['points'][j]['phi'])
                    ts_theta = float(self.ref_path_group_data[group_key][i]['points'][j]['theta'])

                    for k in range(len(self.lm_class.cano_points['pucker'])):
                        pucker_phi = float(self.lm_class.cano_points['phi_cano'][k])
                        pucker_theta = float(self.lm_class.cano_points['theta_cano'][k])

                        arc_lengths[self.lm_class.cano_points['pucker'][k]] = arc_length_calculator(ts_phi, ts_theta,
                                                                                                    pucker_phi,
                                                                                                    pucker_theta)

                    ordered_arc_lengths = OrderedDict(sorted(arc_lengths.items(), key=itemgetter(1), reverse=False))
                    ordered_list = []
                    three_shortest_list = []

                    for key, val in ordered_arc_lengths.items():
                        ordered_list.append([key, val])

                    for k in range(3):
                        three_shortest_list.append(ordered_list[k])

                    self.ref_path_group_data[group_key][i]['points'][j]['closest_puckers'] = three_shortest_list

        return

    def assign_group_name(self):
        for group_key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[group_key])):
                for j in range(len(self.ref_path_group_data[group_key][i]['points'])):
                    tolerance = 0.25

                    first_arc = self.ref_path_group_data[group_key][i]['points'][j]['closest_puckers'][0][1]
                    second_arc = self.ref_path_group_data[group_key][i]['points'][j]['closest_puckers'][1][1]

                    total_arc = first_arc + second_arc

                    first_weight = first_arc / total_arc

                    if first_weight < tolerance:
                        self.ref_path_group_data[group_key][i]['points'][j]['name'] = self.ref_path_group_data[group_key][i]['points'][j]['closest_puckers'][0][0]
                    else:
                        self.ref_path_group_data[group_key][i]['points'][j]['name'] = self.ref_path_group_data[group_key][i]['points'][j]['closest_puckers'][0][0] \
                                                       + '/' \
                                                       + self.ref_path_group_data[group_key][i]['points'][j]['closest_puckers'][1][0]
    # endregion

    # # # do_calc functions # # #
    # region
    # all calcs are on the ts pts
    def do_calcs(self):
        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                self.calc_WSS(key, i)
                self.calc_weighting(key, i)
                self.calc_WWSS(key, i)
                self.calc_group_RMSD(key, i)
                self.calc_group_WRMSD(key, i)

        self.calc_SSE()
        self.calc_WSSE()
        self.calc_RMSD()
        self.calc_WRMSD()

    def calc_weighting(self, lm_group, ts_group):
        total_boltz = 0

        for i in range(len(self.ref_path_group_data[lm_group][ts_group]['points'])):
            e_val = self.ref_path_group_data[lm_group][ts_group]['points'][i]['G298 (Hartrees)']
            component = math.exp(-e_val / (K_B * DEFAULT_TEMPERATURE))
            self.ref_path_group_data[lm_group][ts_group]['points'][i]['ind_boltz'] = component
            total_boltz += component

        wt_gibbs = 0
        for i in range(len(self.ref_path_group_data[lm_group][ts_group]['points'])):
            if self.ref_path_group_data[lm_group][ts_group]['points'][i]['ind_boltz'] == 0:
                wt_gibbs += 0
                self.ref_path_group_data[lm_group][ts_group]['points'][i]['weighting'] = 0
            else:
                wt_gibbs += (self.ref_path_group_data[lm_group][ts_group]['points'][i]['ind_boltz'] / total_boltz)\
                            * self.ref_path_group_data[lm_group][ts_group]['points'][i]['G298 (Hartrees)']
                self.ref_path_group_data[lm_group][ts_group]['points'][i]['weighting'] = \
                    self.ref_path_group_data[lm_group][ts_group]['points'][i]['ind_boltz'] / total_boltz

        self.ref_path_group_data[lm_group][ts_group]['G298 (Hartrees)'] = round(wt_gibbs, 3)

    def calc_WSS(self, lm_group, ts_group):
        WSS = 0

        for i in range(len(self.ref_path_group_data[lm_group][ts_group]['points'])):
            arc_length = self.ref_path_group_data[lm_group][ts_group]['points'][i]['arc_lengths'][0][1]
            WSS += arc_length**2

            self.ref_path_group_data[lm_group][ts_group]['WSS'] = round(WSS, 5)

    def calc_WWSS(self, lm_group, ts_group):
        WWSS = 0

        for i in range(len(self.ref_path_group_data[lm_group][ts_group]['points'])):
            arc_length = self.ref_path_group_data[lm_group][ts_group]['points'][i]['arc_lengths'][0][1]
            weighting = self.ref_path_group_data[lm_group][ts_group]['points'][i]['weighting']
            WWSS += (arc_length ** 2) * weighting

        self.ref_path_group_data[lm_group][ts_group]['WWSS'] = round(WWSS, 5)

    def calc_group_RMSD(self, lm_group, ts_group):
        size = len( self.ref_path_group_data[lm_group][ts_group]['points'])
        if (size == 0):
            RMSD = 'n/a'
            self.ref_path_group_data[lm_group][ts_group]['group_RMSD'] = RMSD
        else:
            RMSD = ( self.ref_path_group_data[lm_group][ts_group]['WSS'] / size) ** 0.5
            self.ref_path_group_data[lm_group][ts_group]['group_RMSD'] = round(RMSD, 5)

    def calc_group_WRMSD(self, lm_group, ts_group):
        size = len(self.ref_path_group_data[lm_group][ts_group]['points'])

        if (size == 0):
            WRMSD = 'n/a'
            self.ref_path_group_data[lm_group][ts_group]['group_WRMSD'] = WRMSD
        else:
            WRMSD = (self.ref_path_group_data[lm_group][ts_group]['WWSS'] / size) ** 0.5
            self.ref_path_group_data[lm_group][ts_group]['group_WRMSD'] = round(WRMSD, 5)

    def calc_SSE(self):
        SSE = 0

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                SSE +=  self.ref_path_group_data[key][i]['WSS']

        self.overall_data['SSE'] = round(SSE, 5)

    def calc_WSSE(self):
        WSSE = 0

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                WSSE += self.ref_path_group_data[key][i]['WWSS']

        self.overall_data['WSSE'] = round(WSSE, 5)

    def calc_RMSD(self):
        size = 0

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                size += 1

        RMSD = (self.overall_data['SSE'] / size) ** 0.5
        self.overall_data['RMSD'] = round(RMSD, 5)

    def calc_WRMSD(self):
        size = 0

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                size += 1

        WRMSD = (self.overall_data['WSSE'] / size) ** 0.5
        self.overall_data['WRMSD'] = round(WRMSD, 5)
    # endregion

    # # # plotting functions # # #
    # region
    # get group keys associated with north, south, and equatorial
    def circ_groups_init(self):
        self.north_groups = []
        self.south_groups = []
        self.equat_groups = []

        key_list = list(self.path_group_data.keys())
        key_list.sort()

        for key in self.path_group_data:
            if int(key.split("_")[0]) == int(key_list[0].split("_")[0]):
                self.north_groups.append(key)
            elif int(key.split("_")[1]) == int(key_list[-1].split("_")[1]):
                self.south_groups.append(key)
            else:
                self.equat_groups.append(key)

        return

    def plot_group_names(self, group):
        if group in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[group])):
                for j in range(len(self.ref_path_group_data[group][i]['points'])):
                    point = self.ref_path_group_data[group][i]['points'][j]

                    if float(point['theta']) < 30 or float(point['phi']) < 25:
                        self.ts_class.plot.ax_rect.annotate(point['name'], xy=(float(point['phi']),
                                                            float(point['theta'])),
                                                            xytext=(float(point['phi']) - 10,
                                                            float(point['theta']) + 15),
                                                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), )
                    else:
                        self.ts_class.plot.ax_rect.annotate(point['name'], xy=(point['phi'],
                                                            point['theta']),
                                                            xytext=(point['phi'] - 10,
                                                            float(point['theta']) - 15),
                                                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), )

    def plot_path_group_raw(self, path_group):
        for i in range(len(self.path_group_data[path_group])):
            ts_vert = pol2cart([self.path_group_data[path_group][i]['phi'], self.path_group_data[path_group][i]['theta']])
            lm1_vert = pol2cart([self.path_group_data[path_group][i]['lm1']['phi'], self.path_group_data[path_group][i]['lm1']['theta']])
            lm2_vert = pol2cart([self.path_group_data[path_group][i]['lm2']['phi'], self.path_group_data[path_group][i]['lm2']['theta']])

            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')

            if self.north_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')
            elif self.south_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')

    def plot_path_group_single(self, path_group):
        lm1_key = int(path_group.split('_')[0])
        lm2_key = int(path_group.split('_')[1])

        lm1_data = self.lm_class.groups_dict[lm1_key]
        lm2_data = self.lm_class.groups_dict[lm2_key]

        lm1_vert = pol2cart([float(lm1_data['mean_phi']), float(lm1_data['mean_theta'])])
        lm2_vert = pol2cart([float(lm2_data['mean_phi']), float(lm2_data['mean_theta'])])

        for i in range(len(self.path_group_data[path_group])):
            ts_vert = pol2cart([self.path_group_data[path_group][i]['phi'], self.path_group_data[path_group][i]['theta']])

            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')

            if self.north_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')
            elif self.south_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')

    def plot_ref_path_group_single(self, path_group):
        lm1_key = int(path_group.split('_')[0])
        lm2_key = int(path_group.split('_')[1])

        lm1_data = self.lm_class.groups_dict[lm1_key]
        lm2_data = self.lm_class.groups_dict[lm2_key]

        lm1_vert = pol2cart([float(lm1_data['mean_phi']), float(lm1_data['mean_theta'])])
        lm2_vert = pol2cart([float(lm2_data['mean_phi']), float(lm2_data['mean_theta'])])

        for i in range(len(self.ref_path_group_data[path_group])):
            ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'], self.ref_path_group_data[path_group][i]['theta']])

            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'gray', 30], [lm1_vert, 'green', 60], 'gray')
            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'gray', 30], [lm2_vert, 'green', 60], 'gray')

            if self.north_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'gray', 30], [lm1_vert, 'green', 60], 'gray')
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'gray', 30], [lm2_vert, 'green', 60], 'gray')
            elif self.south_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'gray', 30], [lm1_vert, 'green', 60], 'gray')
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'gray', 30], [lm2_vert, 'green', 60], 'gray')

    def plot_all_path_groups_raw(self):
        for key in self.path_group_data:
            self.plot_path_group_raw(key)

        # self.lm_class.plot_group_names()

    def plot_all_path_groups_single(self):
        for key in self.path_group_data:
            self.plot_path_group_single(key)

    def set_title_and_legend(self, artist_list, label_list):
        self.ts_class.plot.ax_rect.legend(artist_list,
                                          label_list,
                                          scatterpoints=1, fontsize=8, frameon=False, framealpha=0.75,
                                          bbox_to_anchor=(0.5, -0.3), loc=9, borderaxespad=0, ncol=4).set_zorder(100)

    def show(self):
        self.lm_class.show()
    # endregion

    # # # saving functions # # #
    def save_all_figures_raw(self, mol_name):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')

        artist_list = [met_lm_Artist, met_ts_Artist, path_Artist, cano_lm_Artist]
        label_list = [self.method + ' LM', self.method + ' TS', 'Pathway', 'Canonical Designation']

        base_name = "z_dataset-" + mol_name + "-TS-" + self.method

        if not os.path.exists(os.path.join(self.ts_dir, self.method)):
            os.makedirs(os.path.join(self.ts_dir, self.method))

        met_data_dir = os.path.join(self.ts_dir, self.method)

        if not os.path.exists(os.path.join(met_data_dir, 'raw_LMs')):
            os.makedirs(os.path.join(met_data_dir, 'raw_LMs'))

        raw_data_dir = os.path.join(met_data_dir, 'raw_LMs')

        for key in self.path_group_data:
            # saves a plot of each group individually plotted
            self.plot_path_group_raw(key)
            self.ts_class.plot_cano()
            self.plot_group_names(key)

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name + '-' + key, raw_data_dir)
            self.ts_class.wipe_plot()

    def save_all_figures_single(self, mol_name):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')
        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', marker='s')

        artist_list = [met_lm_Artist, met_ts_Artist, path_Artist, ref_path_Artist, cano_lm_Artist]
        label_list = [self.method + ' LM', self.method + ' TS', 'Pathway', 'Reference pathway', 'Canonical Designation']

        base_name = "z_dataset-" + mol_name + "-TS-" + self.method

        if not os.path.exists(os.path.join(self.ts_dir, self.method)):
            os.makedirs(os.path.join(self.ts_dir, self.method))

        met_data_dir = os.path.join(self.ts_dir, self.method)

        if not os.path.exists(os.path.join(met_data_dir, 'single_LMs')):
            os.makedirs(os.path.join(met_data_dir, 'single_LMs'))

        avg_data_dir = os.path.join(met_data_dir, 'single_LMs')

        for key in self.path_group_data:
            # saves a plot of each group individually plotted
            self.plot_path_group_single(key)
            if key in self.ref_path_group_data:
                self.plot_ref_path_group_single(key)
            self.ts_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name + '-' + key, avg_data_dir)
            self.ts_class.wipe_plot()
    # endregion

class Compare_All_Methods:
    def __init__(self, methods_lm_data_in, methods_ts_data_in, lm_dir_in, ts_dir_in):
        self.methods_lm_data = methods_lm_data_in
        self.lm_dir = lm_dir_in
        self.group_RMSD_vals = {}
        self.group_WRMSD_vals = {}

        self.methods_ts_data = methods_ts_data_in
        self.ts_dir = ts_dir_in

    def write_to_txt(self, do_print):
        tables = []

        for i in range(len(self.methods_lm_data)):
            for j in range(len(self.methods_lm_data[0].group_rows)):
                header = []
                header.append('group_' + str(j))
                header.append('group_RMSD')
                header.append('group_WRMSD')
                header.append('WSS')
                header.append('WWSS')

                if len(tables) < len(self.methods_lm_data[0].group_rows):
                    tables.append(PrettyTable(header))

                tables[j].add_row(self.methods_lm_data[i].group_rows[j])

            if len(tables) < len(self.methods_lm_data[0].group_rows) + 1:
                header = []
                header.append('overall')
                header.append('   RMSD   ')
                header.append('   WRMSD   ')
                header.append('SSE')
                header.append('WSSE')

                tables.append(PrettyTable(header))

            tables[len(tables) - 1].add_row(self.methods_lm_data[i].overall_row)

        filename = os.path.join(LM_DIR, 'comparison_data.txt')
        with open(filename, 'w') as file:
            file.write('')

        for i in range(len(tables)):
            if do_print:
                print(tables[i])

            table_txt = tables[i].get_string()

            with open(filename, 'a') as file:
                file.write(table_txt)
                file.write('\n')

    def write_lm_to_csv(self):
        group_RMSD_dict = {}
        group_WRMSD_dict = {}
        WSS_dict = {}
        WWSS_dict = {}

        group_RMSD_dict['group'] = []
        group_WRMSD_dict['group'] = []
        WSS_dict['group'] = []
        WWSS_dict['group'] = []

        group_RMSD_dict['pucker'] = []
        group_WRMSD_dict['pucker'] = []
        WSS_dict['pucker'] = []
        WWSS_dict['pucker'] = []

        # listing group names
        for i in range(len(self.methods_lm_data[0].group_data)):
            group_RMSD_dict['group'].append(i)
            group_WRMSD_dict['group'].append(i)
            WSS_dict['group'].append(i)
            WWSS_dict['group'].append(i)

        # listing group pucker
        for i in range(len(self.methods_lm_data[0].group_data)):
            group_RMSD_dict['pucker'].append(self.methods_lm_data[0].group_data[i]['name'])
            group_WRMSD_dict['pucker'].append(self.methods_lm_data[0].group_data[i]['name'])
            WSS_dict['pucker'].append(self.methods_lm_data[0].group_data[i]['name'])
            WWSS_dict['pucker'].append(self.methods_lm_data[0].group_data[i]['name'])

        # filling method data for each dict
        for i in range(len(self.methods_lm_data)):
            method = self.methods_lm_data[i].overall_data['method']

            group_RMSD_dict[method] = []
            group_WRMSD_dict[method] = []
            WSS_dict[method] = []
            WWSS_dict[method] = []

            for j in range(len(self.methods_lm_data[i].group_data)):
                group_RMSD_val = self.methods_lm_data[i].group_data[j]['group_RMSD']
                group_WRMSD_val = self.methods_lm_data[i].group_data[j]['group_WRMSD']
                WSS_val = self.methods_lm_data[i].group_data[j]['WSS']
                WWSS_val = self.methods_lm_data[i].group_data[j]['WWSS']

                group_RMSD_dict[method].append(group_RMSD_val)
                group_WRMSD_dict[method].append(group_WRMSD_val)
                WSS_dict[method].append(WSS_val)
                WWSS_dict[method].append(WWSS_val)

        group_RMSD_csv = os.path.join(self.lm_dir, 'group_RMSD.csv')
        group_WRMSD_csv = os.path.join(self.lm_dir, 'group_WRMSD.csv')
        WSS_csv = os.path.join(self.lm_dir, 'WSS.csv')
        WWSS_csv = os.path.join(self.lm_dir, 'WWSS.csv')

        with open(group_RMSD_csv, 'w', newline='') as file:
            w = csv.writer(file)
            w.writerow(group_RMSD_dict.keys())
            w.writerows(zip(*group_RMSD_dict.values()))
        with open(group_WRMSD_csv, 'w', newline='') as file:
            w = csv.writer(file)
            w.writerow(group_WRMSD_dict.keys())
            w.writerows(zip(*group_WRMSD_dict.values()))
        with open(WSS_csv, 'w', newline='') as file:
            w = csv.writer(file)
            w.writerow(WSS_dict.keys())
            w.writerows(zip(*WSS_dict.values()))
        with open(WWSS_csv, 'w', newline='') as file:
            w = csv.writer(file)
            w.writerow(WWSS_dict.keys())
            w.writerows(zip(*WWSS_dict.values()))

        return

    def write_ts_to_csv(self):
        group_RMSD_dict = {}
        group_WRMSD_dict = {}
        WSS_dict = {}
        WWSS_dict = {}

        group_RMSD_dict['group'] = []
        group_WRMSD_dict['group'] = []
        WSS_dict['group'] = []
        WWSS_dict['group'] = []

        group_RMSD_dict['pucker'] = []
        group_WRMSD_dict['pucker'] = []
        WSS_dict['pucker'] = []
        WWSS_dict['pucker'] = []

        # listing under the hood names
        for key in self.methods_ts_data[0].ref_path_group_data:
            for i in range(len(self.methods_ts_data[0].ref_path_group_data[key])):
                group_RMSD_dict['group'].append(key + '-' + str(i))
                group_WRMSD_dict['group'].append(key + '-' + str(i))
                WSS_dict['group'].append(key + '-' + str(i))
                WWSS_dict['group'].append(key + '-' + str(i))

        # listing group pucker
        for key in self.methods_ts_data[0].ref_path_group_data:
            for i in range(len(self.methods_ts_data[0].ref_path_group_data[key])):
                group_RMSD_dict['pucker'].append(self.methods_ts_data[0].ref_path_group_data[key][i]['name'])
                group_WRMSD_dict['pucker'].append(self.methods_ts_data[0].ref_path_group_data[key][i]['name'])
                WSS_dict['pucker'].append(self.methods_ts_data[0].ref_path_group_data[key][i]['name'])
                WWSS_dict['pucker'].append(self.methods_ts_data[0].ref_path_group_data[key][i]['name'])

        # filling method data for each dict
        for i in range(len(self.methods_ts_data)):
            method = self.methods_ts_data[i].overall_data['method']

            group_RMSD_dict[method] = []
            group_WRMSD_dict[method] = []
            WSS_dict[method] = []
            WWSS_dict[method] = []

            for key in self.methods_ts_data[i].ref_path_group_data:
                for j in range(len(self.methods_ts_data[i].ref_path_group_data[key])):
                    group_RMSD_val = self.methods_ts_data[i].ref_path_group_data[key][j]['group_RMSD']
                    group_WRMSD_val = self.methods_ts_data[i].ref_path_group_data[key][j]['group_WRMSD']
                    WSS_val = self.methods_ts_data[i].ref_path_group_data[key][j]['WSS']
                    WWSS_val = self.methods_ts_data[i].ref_path_group_data[key][j]['WWSS']

                    group_RMSD_dict[method].append(group_RMSD_val)
                    group_WRMSD_dict[method].append(group_WRMSD_val)
                    WSS_dict[method].append(WSS_val)
                    WWSS_dict[method].append(WWSS_val)

        group_RMSD_csv = os.path.join(self.ts_dir, 'group_RMSD.csv')
        group_WRMSD_csv = os.path.join(self.ts_dir, 'group_WRMSD.csv')
        WSS_csv = os.path.join(self.ts_dir, 'WSS.csv')
        WWSS_csv = os.path.join(self.ts_dir, 'WWSS.csv')

        with open(group_RMSD_csv, 'w', newline='') as file:
            w = csv.writer(file)
            w.writerow(group_RMSD_dict.keys())
            w.writerows(zip(*group_RMSD_dict.values()))
        with open(group_WRMSD_csv, 'w', newline='') as file:
            w = csv.writer(file)
            w.writerow(group_WRMSD_dict.keys())
            w.writerows(zip(*group_WRMSD_dict.values()))
        with open(WSS_csv, 'w', newline='') as file:
            w = csv.writer(file)
            w.writerow(WSS_dict.keys())
            w.writerows(zip(*WSS_dict.values()))
        with open(WWSS_csv, 'w', newline='') as file:
            w = csv.writer(file)
            w.writerow(WWSS_dict.keys())
            w.writerows(zip(*WWSS_dict.values()))

        return

    def organize_data_for_plotting(self, order_in):

        for j in range(len(self.methods_data[0].group_data)):
            self.group_RMSD_vals[str(j)] = []
            self.group_WRMSD_vals[str(j)] = []

            for k in range(len(order_in)):
                for i in range(len(self.methods_data)):
                    if self.methods_data[i].overall_data['method'] == order_in[k]:
                        for j in range(len(self.methods_data[i].group_data)):
                            if self.methods_data[i].group_data[j]['group_RMSD'] != 'n/a':
                                self.group_RMSD_vals[str(j)].append(self.methods_data[i].group_data[j]['group_RMSD'])
                                self.group_WRMSD_vals[str(j)].append(self.methods_data[i].group_data[j]['group_WRMSD'])
                            else:
                                self.group_RMSD_vals[str(j)].append(float(0))
                                self.group_WRMSD_vals[str(j)].append(float(0))

        return

    def ploting_all_method_information(self, order_in):

        pass
        #
        #
        # # self.fig, self.ax_rect = plt.subplots(facecolor='white')
        # for j in range(len(self.methods_data[0].group_data)):
        #     plt.figure(j, facecolor='white')
        #     objects = order_in
        #
        #     y_pos = np.arange(len(objects))
        #     performance = group_RMSD_vals[str(j)]
        #
        #     plt.bar(y_pos, performance, align='center', alpha=0.5)
        #     plt.xticks(y_pos, objects)
        #
        #     plt.ylim([0, 0.5])
        #     plt.ylabel('RMSD')
        #     plt.title('Group ' + str(j))
        #
        #     plt.show()
#endregion

# # # Helper Functions # # #
#region
# creates a .csv file in the form of the hsp ts .csv file
def rewrite_ts_hartree(ts_hartree_dict_list, method, molecule, dir):
    filename = 'z_dataset-' + molecule + '-TS-' + method + '.csv'

    ts_path_dict = {}

    ts_count = 0
    lm_count = 0

    # separate the dicts in terms of pathway
    for i in range(len(ts_hartree_dict_list)):
        ts_path = ts_hartree_dict_list[i]['File Name'].split('_')[0]

        if ts_path not in ts_path_dict:
            ts_path_dict[ts_path] = []

        ts_path_dict[ts_path].append(ts_hartree_dict_list[i])

        if float(ts_hartree_dict_list[i]['Freq 1']) < 0:
            ts_count += 1
        else:
            lm_count += 1

    assert(lm_count / ts_count == 2)

    new_ts_path_list = []

    for key in ts_path_dict:
        new_ts_path = {}

        for i in range(len(ts_path_dict[key])):
            # if the dict is the TS pt
            if float(ts_path_dict[key][i]['Freq 1']) < 0:
                new_ts_path['phi'] = ts_path_dict[key][i]['phi']
                new_ts_path['theta'] = ts_path_dict[key][i]['theta']
                new_ts_path['energy (A.U.)'] = ts_path_dict[key][i]['Energy (A.U.)']
                new_ts_path['G298 (Hartrees)'] = ts_path_dict[key][i]['G298 (Hartrees)']
                new_ts_path['Pucker'] = ts_path_dict[key][i]['Pucker']
            # else if it is a the forward lm
            elif 'ircf' in ts_path_dict[key][i]['File Name']:
                new_ts_path['phi_lm1'] = ts_path_dict[key][i]['phi']
                new_ts_path['theta_lm1'] = ts_path_dict[key][i]['theta']
            # else it is the reverse lm
            else:
                new_ts_path['phi_lm2'] = ts_path_dict[key][i]['phi']
                new_ts_path['theta_lm2'] = ts_path_dict[key][i]['theta']

        new_ts_path_list.append(new_ts_path)

    ts_paths_dict = {}

    for i in range(len(new_ts_path_list)):
        for key in new_ts_path_list[i]:
            if key not in ts_paths_dict:
                ts_paths_dict[key] = []

            ts_paths_dict[key].append(new_ts_path_list[i][key])

    full_filename = os.path.join(dir, filename)

    with open(full_filename, 'w', newline='') as file:
        w = csv.writer(file)
        w.writerow(ts_paths_dict.keys())
        w.writerows(zip(*ts_paths_dict.values()))

    return
#endregion

# # #  Main  # # #
#region
def main():
    save = True
    sv_all_mol_dir = os.path.join(SV_DIR, 'molecules')
    mol_list_dir = os.listdir(sv_all_mol_dir)

    num_clusters = [9, 8]

    # for each molecule, perform the comparisons
    for i in range(len(mol_list_dir)):
        molecule = mol_list_dir[i]

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(MET_COMP_DIR, mol_list_dir[i])):
            os.makedirs(os.path.join(MET_COMP_DIR, mol_list_dir[i]))

        comp_mol_dir = os.path.join(MET_COMP_DIR, mol_list_dir[i])

        sv_mol_dir = os.path.join(sv_all_mol_dir, mol_list_dir[i])

        # # # local minimum directory init # # #
        #region
        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(comp_mol_dir, 'local_minimum')):
            os.makedirs(os.path.join(comp_mol_dir, 'local_minimum'))

        comp_lm_dir = os.path.join(comp_mol_dir, 'local_minimum')

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(sv_mol_dir, 'z_datasets-LM')):
            os.makedirs(os.path.join(sv_mol_dir, 'z_datasets-LM'))

        lm_data_dir = os.path.join(sv_mol_dir, 'z_datasets-LM')
        #endregion

        # # # transition states directory init # # #
        #region
        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(comp_mol_dir, 'transitions_state')):
            os.makedirs(os.path.join(comp_mol_dir, 'transitions_state'))

        comp_ts_dir = os.path.join(comp_mol_dir, 'transitions_state')

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(sv_mol_dir, 'z_datasets-TS')):
            os.makedirs(os.path.join(sv_mol_dir, 'z_datasets-TS'))

        ts_data_dir = os.path.join(sv_mol_dir, 'z_datasets-TS')

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(sv_mol_dir, 'TS-unformatted')):
            os.makedirs(os.path.join(sv_mol_dir, 'TS-unformatted'))

        ts_unformatted_dir = os.path.join(sv_mol_dir, 'TS-unformatted')
        #endregion

        lm_comp_data_list = []
        ts_comp_data_list = []

        # initialization info for local minimum clustering for specific molecule
        number_clusters = num_clusters[i]
        dict_cano = read_csv_canonical_designations(mol_list_dir[i] + '-CP_params.csv', sv_mol_dir)
        data_points, phi_raw, theta_raw, energy = read_csv_data('z_' + mol_list_dir[i] + '_lm-b3lyp_howsugarspucker.csv',
                                                                sv_mol_dir)
        lm_class = Local_Minima(number_clusters, data_points, dict_cano, phi_raw, theta_raw, energy)

        ts_data_dict = read_csv_data_TS('z_' + mol_list_dir[i] + '_TS-b3lyp_howsugarspucker.csv',
                                                                sv_mol_dir)[3]
        ts_class = Transition_States(ts_data_dict, lm_class)

        # # # local minimum comparison data initialization # # #
        #region
        # for every local min data file in the directory perform the comparison calculations
        for filename in os.listdir(lm_data_dir):
            if filename.endswith(".csv"):
                method_hartree = read_csv_to_dict(os.path.join(lm_data_dir, filename), mode='r')
                method = (filename.split('-', 3)[3]).split('.')[0]
                lm_comp_class = Local_Minima_Compare(method, method_hartree, lm_class, comp_lm_dir)

                lm_comp_data_list.append(lm_comp_class)
        #endregion

        # # # transition state comparison data initialization # # #
        #region
        # for every ts data file in the directory reformat
        for filename in os.listdir(ts_unformatted_dir):
            if filename.endswith(".csv"):
                ts_hartree = read_csv_to_dict(os.path.join(ts_unformatted_dir, filename), mode='r')
                method = (filename.split('-', 3)[3]).split('.')[0]
                rewrite_ts_hartree(ts_hartree, method, molecule, ts_data_dir)

        # for every ts data file in the directory perform the comparison calculations
        for filename in os.listdir(ts_data_dir):
            if filename.endswith(".csv"):
                ts_hartree = read_csv_to_dict(os.path.join(ts_data_dir, filename), mode='r')
                method = (filename.split('-', 3)[3]).split('.')[0]
                ts_comp_class = Transition_State_Compare(method, ts_hartree, lm_class, ts_class, comp_ts_dir)

                ts_comp_data_list.append(ts_comp_class)
        #endregion

        comp_all_met = Compare_All_Methods(lm_comp_data_list, ts_comp_data_list, comp_lm_dir, comp_ts_dir)

        order_in = ['reference', 'b3lyp', 'dftb', 'am1', 'pm6', 'pm3mm', 'pm3']

        # comp_all_met_LM.organize_data_for_plotting(order_in)

        # comp_all_met_LM.ploting_all_method_information(order_in)

        if save:
            # save the comparison data
            comp_all_met.write_lm_to_csv()
            comp_all_met.write_ts_to_csv()

            # save all lm plots
            for j in range(len(lm_comp_data_list)):
                lm_comp_data_list[j].plot_all_groupings()
                lm_comp_data_list[j].save_all_figures(mol_list_dir[i])

                lm_comp_data_list[j].plot_all_groupings_raw()
                lm_comp_data_list[j].save_all_figures_raw(mol_list_dir[i])

            # save all ts plots
            for j in range(len(ts_comp_data_list)):
                ts_comp_data_list[j].save_all_figures_raw(mol_list_dir[i])
                ts_comp_data_list[j].save_all_figures_single(mol_list_dir[i])

    return

if __name__ == '__main__':
    status = main()
    sys.exit(status)
#endregion
