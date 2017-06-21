#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this script is to make comparisons for a particular QM method to the REFERENCE set of HSP.
"""

# # # import # # #
#region
from __future__ import print_function
import os

import csv
import math

import matplotlib
matplotlib.use('TkAgg')

from collections import OrderedDict
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from spherecluster import SphericalKMeans

from qm_utils.qm_common import arc_length_calculator
from qm_utils.spherical_kmeans_voronoi import pol2cart, plot_on_circle, plot_line, cart2pol, Plots
#endregion

# # # Header Stuff # # #
#region
try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'jhuber/SPVicchio'

# # Default Parameters # #
HARTREE_TO_KCALMOL = 627.5095
TOL_ARC_LENGTH = 0.1
TOL_ARC_LENGTH_CROSS = 0.2  # THIS WAS THE ORGINAL TOLERANCE6
DEFAULT_TEMPERATURE = 298.15
K_B = 0.001985877534  # Boltzmann Constant in kcal/mol K
#endregion

# # # Pucker Keys # # #
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

# # # Helper Functions # # #
#region
# bool for if the current color chosen isn't a color surrounding it
def is_excluded(color, list_of_excluded_colors):
    for i in range(len(list_of_excluded_colors)):
        if color == list_of_excluded_colors[i]:
            return True

    return False

# calculates WSS for the current number of clusters for added_kmeans for transition_state_compare
def calc_error(added_points):
    WSS = 0

    for i in range(len(added_points)):
        comp_val = added_points[i]['arc_to_added_skm']
        WSS += comp_val ** 2

    return round(WSS, 5)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_color_matrix(data_matrix, data_key, comp_key):
    color_matrix = []

    cmap = plt.get_cmap('RdYlGn')
    cmap = truncate_colormap(cmap, 0.15, 0.85)

    for i in range(len(data_matrix)):
        color_matrix.append([])

        for j in range(len(data_matrix[i])):
            if data_key == comp_key + '_comp':
                if data_matrix[i][j] == False:
                    color_matrix[i].append(cmap(0))
                elif data_matrix[i][j] == True:
                    color_matrix[i].append(cmap(0.99))
                else:
                    color_matrix[i].append('white')
            elif data_key == comp_key + '_group_WRMSD':
                if isinstance(data_matrix[1][j], str):
                    color = 'white'
                elif data_matrix[i][j] == 0:
                    comp_ratio = 0.99

                    color = cmap(comp_ratio)
                elif isinstance(data_matrix[i][j], str):
                    color = 'white'

                    if data_matrix[i][j] == 'n/a':
                        comp_ratio = 0

                        color = cmap(comp_ratio)
                elif data_matrix[1][j] == 0:
                    comp_ratio = 0.01 / data_matrix[i][j]

                    color = cmap(comp_ratio)
                else:
                    comp_ratio = data_matrix[1][j] / data_matrix[i][j]

                    color = cmap(comp_ratio)

                color_matrix[i].append(color)
            else:
                color = 'white'
                color_matrix[i].append(color)

    return color_matrix

def plot_diff_trend(plot, ref_line_energy, line_energy, x_points, size, ymax, title, method=''):
    plot.ax_rect.set_xlabel('Transition State')
    plot.ax_rect.set_ylabel('G298 (kcal/mol)')

    index = np.arange(size)
    bar_width = 0.8
    opacity = 0.8

    plt.bar(index, ref_line_energy, bar_width,
            alpha=opacity,
            align='center',
            color='black',
            label='REFERENCE')

    plt.bar(index, line_energy, bar_width,
            alpha=opacity,
            align='center',
            color='red',
            label=method)

    plt.title(title)
    plt.legend()

    plt.xticks(index)

    plot.ax_rect.set_ylim([0, ymax + 1])
    plot.ax_rect.set_xticks(index)
    plot.ax_rect.set_xticklabels(x_points, rotation=75)

    major_ticksy = np.arange(0, ymax + 5, 2)
    minor_ticksy = np.arange(0, ymax + 5, 1)

    plot.ax_rect.set_yticks(major_ticksy)
    plot.ax_rect.set_yticks(minor_ticksy, minor=True)
#endregion

# # # Classes # # #
#region
class Local_Minima_Compare():
    """
    class for organizing the local minima information
    """
    def __init__(self, molecule_in, method_in, lm_dataset_in, lm_class_in, lm_dir_in, lm_ref_in=None):
        # tolerance for comparisons between method and REFERENCE
        # used in self.calc_num_comp_lm()
        self.comp_tolerance = 0.1
        self.comp_cutoff = 0.1

        self.molecule = molecule_in
        self.method = method_in.upper()

        self.lm_class = lm_class_in
        self.lm_ref = lm_ref_in

        self.hartree_data = []
        self.group_data = []

        self.overall_data = {}

        self.lm_dir = lm_dir_in
        self.lm_dataset = lm_dataset_in

        self.fix_hartrees()

        self.populate_hartree_data()
        self.populate_groupings()

        self.do_calcs()

        self.dir_init()

    # # # __init__ functions # # #
    #region
    def fix_hartrees(self):
        # converting hartrees to kcal/mol
        for i in range(len(self.lm_dataset)):
            self.lm_dataset[i]['G298 (Hartrees)'] = 627.509 * float(self.lm_dataset[i]['G298 (Hartrees)'])

        # setting reference to be lowest energy in dataset
        self.min_G298 = self.lm_dataset[0]['G298 (Hartrees)']

        for i in range(len(self.lm_dataset)):
            if self.lm_dataset[i]['G298 (Hartrees)'] < self.min_G298:
                self.min_G298 = self.lm_dataset[i]['G298 (Hartrees)']

        for i in range(len(self.lm_dataset)):
            self.lm_dataset[i]['G298 (Hartrees)'] -= self.min_G298

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
            self.group_data[i]['method'] = self.method
            self.group_data[i]['points'] = {}

            self.group_data[i]['phi'] = self.lm_class.groups_dict[i]['mean_phi']
            self.group_data[i]['theta'] = self.lm_class.groups_dict[i]['mean_theta']

            self.group_data[i]['name'] = self.lm_class.groups_dict[i]['name']

            for j in range(len(self.hartree_data)):
                if self.hartree_data[j]['arc_lengths'][0][0] == i:
                    self.group_data[i]['points'][j] = self.hartree_data[j]

        return

    def dir_init(self):
        if not os.path.exists(os.path.join(self.lm_dir, self.method)):
            os.makedirs(os.path.join(self.lm_dir, self.method))

        self.met_data_dir = os.path.join(self.lm_dir, self.method)

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(self.met_data_dir, 'overall')):
            os.makedirs(os.path.join(self.met_data_dir, 'overall'))

        self.overall_dir = os.path.join(self.met_data_dir, 'overall')

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(self.lm_dir, 'overall')):
            os.makedirs(os.path.join(self.lm_dir, 'overall'))

        self.lm_overall_dir = os.path.join(self.lm_dir, 'overall')

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(self.met_data_dir, 'groups')):
            os.makedirs(os.path.join(self.met_data_dir, 'groups'))

        self.groups_dir = os.path.join(self.met_data_dir, 'groups')

        if not os.path.exists(os.path.join(self.met_data_dir, 'heatmaps')):
            os.makedirs(os.path.join(self.met_data_dir, 'heatmaps'))

        self.heatmap_data_dir = os.path.join(self.met_data_dir, 'heatmaps')

        if not os.path.exists(os.path.join(self.heatmap_data_dir, 'by_arclength')):
            os.makedirs(os.path.join(self.heatmap_data_dir, 'by_arclength'))

        self.arc_data_dir = os.path.join(self.heatmap_data_dir, 'by_arclength')

        if not os.path.exists(os.path.join(self.heatmap_data_dir, 'by_gibbs')):
            os.makedirs(os.path.join(self.heatmap_data_dir, 'by_gibbs'))

        self.gibbs_data_dir = os.path.join(self.heatmap_data_dir, 'by_gibbs')

        if not os.path.exists(os.path.join(self.lm_dir, 'final_comp')):
            os.makedirs(os.path.join(self.lm_dir, 'final_comp'))

        self.final_comp_dir = os.path.join(self.lm_dir, 'final_comp')

        if not os.path.exists(os.path.join(self.final_comp_dir, 'by_arclength')):
            os.makedirs(os.path.join(self.final_comp_dir, 'by_arclength'))

        self.arc_comp_dir = os.path.join(self.final_comp_dir, 'by_arclength')

        if not os.path.exists(os.path.join(self.final_comp_dir, 'by_gibbs')):
            os.makedirs(os.path.join(self.final_comp_dir, 'by_gibbs'))

        self.gibbs_comp_dir = os.path.join(self.final_comp_dir, 'by_gibbs')

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

            self.calc_gibbs_WSS(i)
            self.calc_gibbs_WWSS(i)
            self.calc_gibbs_group_RMSD(i)
            self.calc_gibbs_group_WRMSD(i)

        self.calc_SSE()
        self.calc_WSSE()
        self.calc_RMSD()
        self.calc_WRMSD()

        self.calc_gibbs_SSE()
        self.calc_gibbs_WSSE()
        self.calc_gibbs_RMSD()
        self.calc_gibbs_WRMSD()

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

    # # # by arclength # # #
    #region
    def calc_WSS(self, group):
        WSS = 0

        for key in self.group_data[group]['points']:
            arc_length = self.group_data[group]['points'][key]['arc_lengths'][0][1]
            WSS += arc_length ** 2

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

    def calc_num_comp_lm(self, comp_cutoff, comp_tolerance):
        comparable_paths = 0

        for i in range(len(self.group_data)):
            group_WRMSD = self.group_data[i]['group_WRMSD']
            ref_group_WRMSD = self.lm_ref.group_data[i]['group_WRMSD']

            if group_WRMSD != 'n/a' and (group_WRMSD < comp_cutoff or ref_group_WRMSD / group_WRMSD >= comp_tolerance):
                comparable_paths += 1

        return comparable_paths
    #endregion

    # # # by gibbs # # #
    #region
    def calc_gibbs_WSS(self, group):
        WSS = 0

        for key in self.group_data[group]['points']:
            ref_gibbs = self.group_data[group]['weighted_gibbs']
            curr_gibbs = self.group_data[group]['points'][key]['G298 (Hartrees)']

            gibbs_diff = ref_gibbs - curr_gibbs
            WSS += gibbs_diff ** 2

        self.group_data[group]['gibbs_WSS'] = round(WSS, 5)

    def calc_gibbs_WWSS(self, group):
        WWSS = 0

        for key in self.group_data[group]['points']:
            ref_gibbs = self.group_data[group]['weighted_gibbs']
            curr_gibbs = self.group_data[group]['points'][key]['G298 (Hartrees)']

            gibbs_diff = ref_gibbs - curr_gibbs

            weighting = self.group_data[group]['points'][key]['weighting']
            WWSS += (gibbs_diff ** 2) * weighting

        self.group_data[group]['gibbs_WWSS'] = round(WWSS, 5)

    def calc_gibbs_group_RMSD(self, group):
        size = len(self.group_data[group]['points'])

        if(size == 0):
            RMSD = 'n/a'
            self.group_data[group]['gibbs_group_RMSD'] = RMSD
        else:
            RMSD = (self.group_data[group]['gibbs_WSS'] / size) ** 0.5
            self.group_data[group]['gibbs_group_RMSD'] = round(RMSD, 5)

    def calc_gibbs_group_WRMSD(self, group):
        size = len(self.group_data[group]['points'])

        if (size == 0):
            WRMSD = 'n/a'
            self.group_data[group]['gibbs_group_WRMSD'] = WRMSD
        else:
            WRMSD = (self.group_data[group]['gibbs_WWSS'] / size) ** 0.5
            self.group_data[group]['gibbs_group_WRMSD'] = round(WRMSD, 5)

    def calc_gibbs_SSE(self):
        SSE = 0

        for i in range(len(self.group_data)):
            SSE += self.group_data[i]['gibbs_WSS']

        self.overall_data['gibbs_SSE'] = round(SSE, 5)

    def calc_gibbs_WSSE(self):
        WSSE = 0

        for i in range(len(self.group_data)):
            WSSE += self.group_data[i]['gibbs_WWSS']

        self.overall_data['gibbs_WSSE'] = round(WSSE, 5)

    def calc_gibbs_RMSD(self):
        RMSD = (self.overall_data['gibbs_SSE'] / len(self.group_data)) ** 0.5
        self.overall_data['gibbs_RMSD'] = round(RMSD, 5)

    def calc_gibbs_WRMSD(self):
        WRMSD = (self.overall_data['gibbs_WSSE'] / len(self.group_data)) ** 0.5
        self.overall_data['gibbs_WRMSD'] = round(WRMSD, 5)

    def calc_gibbs_num_comp_lm(self, comp_cutoff, comp_tolerance):
        comparable_paths = 0

        for i in range(len(self.group_data)):
            group_WRMSD = self.group_data[i]['gibbs_group_WRMSD']
            ref_group_WRMSD = self.lm_ref.group_data[i]['gibbs_group_WRMSD']

            if group_WRMSD != 'n/a' and (group_WRMSD < comp_cutoff or ref_group_WRMSD / group_WRMSD >= comp_tolerance):
                comparable_paths += 1

        return comparable_paths
    #endregion


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

    def plot_WRMSD_comp(self):
        # plotting heatmap
        for i in range(len(self.group_data)):
            # if no data matches to current ts group
            if self.group_data[i]['group_WRMSD'] == 'n/a':
                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='white',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)
            # if the data is within the tolerance
            elif self.group_data[i]['group_WRMSD'] >= self.comp_cutoff\
                and self.lm_ref.group_data[i]['group_WRMSD'] / self.group_data[i]['group_WRMSD'] < self.comp_tolerance:

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='black',
                                                   marker='o', edgecolor='face',
                                                   zorder=10)
            else:
                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='red',
                                                   marker='o', edgecolor='face',
                                                   zorder=10)

    def plot_WRMSD_heatmap(self):
        # plotting heatmap
        for i in range(len(self.group_data)):
            # if no data matches to current ts group
            if self.group_data[i]['group_WRMSD'] == 'n/a':
                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='white',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)
            else:
                if self.group_data[i]['group_WRMSD'] == 0:
                    lm_size = 1
                else:
                    lm_size = self.lm_ref.group_data[i]['group_WRMSD'] / self.group_data[i]['group_WRMSD']

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='black',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30 * lm_size, c='red',
                                                   marker='o', edgecolor='face',
                                                   zorder=10)

    def plot_RMSD_heatmap(self):
        # plotting heatmap
        for i in range(len(self.group_data)):
            # if no data matches to current ts group
            if self.group_data[i]['group_RMSD'] == 'n/a':
                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='white',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)
            else:
                if self.group_data[i]['group_RMSD'] == 0:
                    lm_size = 1
                else:
                    lm_size = self.lm_ref.group_data[i]['group_RMSD'] / self.group_data[i]['group_RMSD']

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='black',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30 * lm_size, c='red',
                                                   marker='o', edgecolor='face',
                                                   zorder=10)

    def plot_gibbs_WRMSD_comp(self):
        # plotting heatmap
        for i in range(len(self.group_data)):
            # if no data matches to current ts group
            if self.group_data[i]['gibbs_group_WRMSD'] == 'n/a':
                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='white',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)
            # if the data is within the tolerance
            elif self.group_data[i]['gibbs_group_WRMSD'] >= self.comp_cutoff\
                and self.lm_ref.group_data[i]['gibbs_group_WRMSD'] / self.group_data[i]['gibbs_group_WRMSD'] < self.comp_tolerance:

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='black',
                                                   marker='o', edgecolor='face',
                                                   zorder=10)
            else:
                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='red',
                                                   marker='o', edgecolor='face',
                                                   zorder=10)

    def plot_gibbs_WRMSD_heatmap(self):
        # plotting heatmap
        for i in range(len(self.group_data)):
            # if no data matches to current ts group
            if self.group_data[i]['gibbs_group_WRMSD'] == 'n/a':
                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='white',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)
            else:
                if self.group_data[i]['gibbs_group_WRMSD'] == 0:
                    lm_size = 1
                else:
                    lm_size = self.lm_ref.group_data[i]['gibbs_group_WRMSD'] / self.group_data[i]['gibbs_group_WRMSD']

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='black',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30 * lm_size, c='red',
                                                   marker='o', edgecolor='face',
                                                   zorder=10)

    def plot_gibbs_RMSD_heatmap(self):
        # plotting heatmap
        for i in range(len(self.group_data)):
            # if no data matches to current ts group
            if self.group_data[i]['gibbs_group_RMSD'] == 'n/a':
                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='white',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)
            else:
                if self.group_data[i]['gibbs_group_RMSD'] == 0:
                    lm_size = 1
                else:
                    lm_size = self.lm_ref.group_data[i]['gibbs_group_RMSD'] / self.group_data[i]['gibbs_group_RMSD']

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30, c='black',
                                                   marker='o', edgecolor='black',
                                                   zorder=10)

                self.lm_class.plot.ax_rect.scatter(self.group_data[i]['phi'], self.group_data[i]['theta'],
                                                   s=30 * lm_size, c='red',
                                                   marker='o', edgecolor='face',
                                                   zorder=10)

    def set_title_and_legend(self, artist_list, label_list):
        self.lm_class.plot.ax_rect.legend(artist_list,
                                          label_list,
                                          scatterpoints=1, fontsize=8, frameon=False, framealpha=0.75,
                                          bbox_to_anchor=(0.5, -0.15), loc=9, borderaxespad=0, ncol=4).set_zorder(100)

        self.lm_class.plot.ax_rect.set_title(self.method)

    def show(self):
        self.lm_class.show()
    #endregion

    # # # saving functions # # #
    #region
    def save_all_figures(self, overwrite):
        # Create custom artist
        size_scaling = 1
        ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30*size_scaling, c='red', marker='o', edgecolor='face')
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=15*size_scaling, c='blue', marker='o', edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=45*size_scaling, c='black', marker='+', edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')

        artist_list = [ref_lm_Artist, met_lm_Artist, path_Artist, cano_lm_Artist]
        label_list = ['LM Kmeans Center', self.method + ' LM', 'Voronoi Edge', 'Canonical Designation']

        base_name = "z_dataset-" + self.molecule + "-LM-" + self.method

        # if file either doesn't exist or needs to be overwritten
        if not os.path.exists(os.path.join(self.overall_dir, base_name + '-all_groupings' + '.png')) or overwrite:
            # saves a plot of all groupings
            self.plot_all_groupings()
            self.lm_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name + '-all_groupings', self.overall_dir)
            self.lm_class.wipe_plot()

        for i in range(len(self.group_data)):
            if not os.path.exists(os.path.join(self.groups_dir, base_name + '-group_' + str(i) + '.png')) or overwrite:
                # saves a plot of each group individually plotted
                self.plot_grouping(i)
                self.lm_class.plot_cano()

                self.set_title_and_legend(artist_list, label_list)

                self.lm_class.plot.save(base_name + '-group_' + str(i), self.groups_dir)
                self.lm_class.wipe_plot()

    def save_all_groupings(self, overwrite):
        # Create custom artist
        size_scaling = 1
        ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30*size_scaling, c='red', marker='o', edgecolor='face')
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=15*size_scaling, c='blue', marker='o', edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=45*size_scaling, c='black', marker='+', edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')

        artist_list = [ref_lm_Artist, met_lm_Artist, path_Artist, cano_lm_Artist]
        label_list = ['LM Kmeans Center', self.method + ' LM', 'Voronoi Edge', 'Canonical Designation']

        base_name = "z_dataset-" + self.molecule + "-LM-" + self.method

        # if file either doesn't exist or needs to be overwritten
        if not os.path.exists(os.path.join(self.lm_overall_dir, base_name + '-all_groupings' + '.png')) or overwrite:
            # saves a plot of all groupings
            self.plot_all_groupings()
            self.lm_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name + '-all_groupings', self.lm_overall_dir)
            self.lm_class.wipe_plot()

    def save_all_figures_raw(self, overwrite):
        # Create custom artists
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=15*size_scaling, c='blue', marker='o', edgecolor='face')
        raw_ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60*size_scaling, c='black', marker='o', edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60*size_scaling, c='black', marker='+', edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')

        artist_list = [raw_ref_lm_Artist, met_lm_Artist, path_Artist, cano_lm_Artist]
        label_list = ['LM Kmeans Center', self.method + ' LM', 'Voronoi Edge', 'Canonical Designation']

        base_name = "z_dataset-" + self.molecule + "-LM-" + self.method

        if not os.path.exists(os.path.join(self.overall_dir, base_name + '-all_method_raw_data' + '.png')) or overwrite:
            # saves plot of all groupings with the raw group data
            self.plot_method_data()
            self.lm_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name + '-all_method_raw_data', self.overall_dir)
            self.lm_class.wipe_plot()

    def save_WRMSD_heatmap(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='red', marker='o',
                                    edgecolor='face')
        full_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='o',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')
        ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='o',
                                    edgecolor='black')

        artist_list = [ref_lm_Artist, cano_lm_Artist, full_lm_Artist, met_lm_Artist, path_Artist]
        label_list = ['No LM found', 'Canonical Designation', 'LM Kmeans Center', self.method + ' LM',
                      'Voronoi Tessellation']

        base_name = "z_dataset-" + self.molecule + "-lm-WRMSD-heatmap-" + self.method

        if not os.path.exists(os.path.join(self.arc_data_dir, base_name + '.png')) or overwrite:
            self.plot_WRMSD_heatmap()
            self.lm_class.plot_cano()
            self.lm_class.plot_all_vor_sec()

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name, self.arc_data_dir)
            self.lm_class.wipe_plot()

    def save_RMSD_heatmap(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='red', marker='o',
                                    edgecolor='face')
        full_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='o',
                                     edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')
        ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='o',
                                    edgecolor='black')

        artist_list = [ref_lm_Artist, cano_lm_Artist, full_lm_Artist, met_lm_Artist, path_Artist]
        label_list = ['No LM found', 'Canonical Designation', 'LM Kmeans Center', self.method + ' LM',
                      'Voronoi Tessellation']

        base_name = "z_dataset-" + self.molecule + "-lm-RMSD-heatmap-" + self.method

        if not os.path.exists(os.path.join(self.arc_data_dir, base_name + '.png')) or overwrite:
            self.plot_RMSD_heatmap()
            self.lm_class.plot_cano()
            self.lm_class.plot_all_vor_sec()

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name, self.arc_data_dir)
            self.lm_class.wipe_plot()

    def save_WRMSD_comp(self, overwrite):
        # Create custom artist
        size_scaling = 1
        comp_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='red', marker='o',
                                    edgecolor='face')
        uncomp_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='o',
                                     edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')
        no_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='o',
                                    edgecolor='black')

        artist_list = [no_lm_Artist, cano_lm_Artist, uncomp_lm_Artist, comp_lm_Artist, path_Artist]
        label_list = ['No LM found', 'Canonical Designation', 'uncomparable LM', 'comparable LM',
                      'Voronoi Tessellation']

        base_name = "z_dataset-" + self.molecule + "-lm-WRMSD-comp-" + self.method

        if not os.path.exists(os.path.join(self.arc_comp_dir, base_name + '.png')) or overwrite:
            self.plot_WRMSD_comp()
            self.lm_class.plot_cano()
            self.lm_class.plot_all_vor_sec()

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name, self.arc_comp_dir)
            self.lm_class.wipe_plot()

    def save_gibbs_WRMSD_heatmap(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='red', marker='o',
                                    edgecolor='face')
        full_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='o',
                                     edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')
        ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='o',
                                    edgecolor='black')

        artist_list = [ref_lm_Artist, cano_lm_Artist, full_lm_Artist, met_lm_Artist, path_Artist]
        label_list = ['No LM found', 'Canonical Designation', 'LM Kmeans Center', self.method + ' LM',
                      'Voronoi Tessellation']

        base_name = "z_dataset-" + self.molecule + "-lm-WRMSD-heatmap-" + self.method

        if not os.path.exists(os.path.join(self.gibbs_data_dir, base_name + '.png')) or overwrite:
            self.plot_gibbs_WRMSD_heatmap()
            self.lm_class.plot_cano()
            self.lm_class.plot_all_vor_sec()

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name, self.gibbs_data_dir)
            self.lm_class.wipe_plot()

    def save_gibbs_RMSD_heatmap(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='red', marker='o',
                                    edgecolor='face')
        full_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='o',
                                     edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')
        ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='o',
                                    edgecolor='black')

        artist_list = [ref_lm_Artist, cano_lm_Artist, full_lm_Artist, met_lm_Artist, path_Artist]
        label_list = ['No LM found', 'Canonical Designation', 'LM Kmeans Center', self.method + ' LM',
                      'Voronoi Tessellation']

        base_name = "z_dataset-" + self.molecule + "-lm-RMSD-heatmap-" + self.method

        if not os.path.exists(os.path.join(self.gibbs_data_dir, base_name + '.png')) or overwrite:
            self.plot_gibbs_RMSD_heatmap()
            self.lm_class.plot_cano()
            self.lm_class.plot_all_vor_sec()

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name, self.gibbs_data_dir)
            self.lm_class.wipe_plot()

    def save_gibbs_WRMSD_comp(self, overwrite):
        # Create custom artist
        size_scaling = 1
        comp_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='red', marker='o',
                                     edgecolor='face')
        uncomp_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='o',
                                       edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')
        no_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='o',
                                   edgecolor='black')

        artist_list = [no_lm_Artist, cano_lm_Artist, uncomp_lm_Artist, comp_lm_Artist, path_Artist]
        label_list = ['No LM found', 'Canonical Designation', 'uncomparable LM', 'comparable LM',
                      'Voronoi Tessellation']

        base_name = "z_dataset-" + self.molecule + "-lm-WRMSD-comp-" + self.method

        if not os.path.exists(os.path.join(self.gibbs_comp_dir, base_name + '.png')) or overwrite:
            self.plot_gibbs_WRMSD_comp()
            self.lm_class.plot_cano()
            self.lm_class.plot_all_vor_sec()

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name, self.gibbs_comp_dir)
            self.lm_class.wipe_plot()
    #endregion

    pass

class Transition_State_Compare():
    """
    class for organizing the transition state information
    """
    def __init__(self, molecule_in, method_in, ts_dataset_in, lm_class_in, ts_class_in, ts_dir_in, min_G298_in=None, ts_ref_in=None, ts_ref_added_in=None):
        # tolerances for comparability
        self.comp_tolerance = 0.1
        self.comp_cutoff = 0.1

        self.min_G298 = min_G298_in

        # ts and lm classes with REFERENCE information
        self.lm_class = lm_class_in
        self.ts_class = ts_class_in

        # REFERENCE ts_comp and added REFERENCE ts_comp objects
        self.ts_ref = ts_ref_in
        self.ts_ref_added = ts_ref_added_in

        # method data
        self.ts_dataset = ts_dataset_in

        self.molecule = molecule_in
        self.method = method_in.upper()

        # dir for saving
        self.ts_dir = ts_dir_in

        # init objects
        self.hartree_data = []
        self.path_group_data = {}
        self.ref_path_group_data = {}
        self.overall_data = {}
        self.overall_data['method'] = self.method

        # init functions
        self.fix_hartrees()
        self.populate_hartree_data()
        self.populate_ref_path_group_data()
        self.populate_path_group_data()
        self.populate_ts_groups()

        self.do_calcs()

        self.assign_closest_puckers()
        self.assign_group_name()

        # calculating added kmeans centers
        self.calc_min_ref_dist()
        self.calc_added_kmeans_dict()

        if ts_ref_in is not None:
            self.assign_comp_orig(self.comp_cutoff, self.comp_tolerance, 'arc')
            self.assign_comp_orig(self.comp_cutoff, self.comp_tolerance, 'gibbs')

        # calculating the average method data for plotting purposes
        self.populate_avg_ts()

        # initializing the lm groups for the north and south hemisphere plots
        self.circ_groups_init()

        self.dir_init()

    # # # __init__ functions # # #
    # region
    def fix_hartrees(self):
        for i in range(len(self.ts_dataset)):
            self.ts_dataset[i]['G298 (Hartrees)'] = float(self.ts_dataset[i]['G298 (Hartrees)'])
            self.ts_dataset[i]['G298 (Hartrees)'] -= self.min_G298 / 627.509

        # converting hartrees to kcal/mol
        for i in range(len(self.ts_dataset)):
            self.ts_dataset[i]['G298 (Hartrees)'] = 627.509 * float(self.ts_dataset[i]['G298 (Hartrees)'])

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

                ref_data['lm1']['name'] = self.lm_class.groups_dict[int(key.split('_')[0])]['name']
                ref_data['lm2']['name'] = self.lm_class.groups_dict[int(key.split('_')[1])]['name']

                ref_data['G298 (Hartrees)'] = self.ts_class.ts_groups[key][i]['G298 (Hartrees)']

                self.ref_path_group_data[ref_key].append(ref_data)

        self.ref_path_group_data = OrderedDict(sorted(self.ref_path_group_data.items(), key=itemgetter(0), reverse=False))

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
                self.path_group_data[key] = {}

            self.path_group_data[key][i] = self.hartree_data[i]

    def populate_ts_groups(self):
        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                self.ref_path_group_data[key][i]['points'] = []
                self.ref_path_group_data[key][i]['WSS'] = 0
                self.ref_path_group_data[key][i]['group_RMSD'] = 'n/a'

        for key in self.path_group_data:
            if key in self.ref_path_group_data:
                for i in self.path_group_data[key]:
                    # list for shortest arclengths
                    arc_lengths = {}

                    ts_phi = float(self.path_group_data[key][i]['phi'])
                    ts_theta = float(self.path_group_data[key][i]['theta'])

                    for j in range(len(self.ref_path_group_data[key])):
                        ref_phi = self.ref_path_group_data[key][j]['phi']
                        ref_theta = self.ref_path_group_data[key][j]['theta']

                        arc_lengths[j] = arc_length_calculator(ts_phi, ts_theta, ref_phi, ref_theta)

                    ordered_arc_lengths = OrderedDict(sorted(arc_lengths.items(), key=itemgetter(1), reverse=False))
                    ordered_list = []

                    for arc_key, val in ordered_arc_lengths.items():
                        ordered_list.append([arc_key, val])

                    self.path_group_data[key][i]['arc_lengths'] = ordered_list
                    self.path_group_data[key][i]['arc'] = self.path_group_data[key][i]['arc_lengths'][0][1]

                for i in range(len(self.ref_path_group_data[key])):
                    for j in self.path_group_data[key]:
                        if self.path_group_data[key][j]['arc_lengths'][0][0] == i:
                            self.ref_path_group_data[key][i]['points'].append(self.path_group_data[key][j])

        return

    def calc_min_ref_dist(self):
        # to assure the first distance will be less
        min_dist = 100

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                for key2 in self.ref_path_group_data:
                    for j in range(len(self.ref_path_group_data[key2])):
                        point1 = self.ref_path_group_data[key][i]
                        point2 = self.ref_path_group_data[key2][j]

                        theta1 = point1['theta']
                        theta2 = point2['theta']

                        phi1 = point1['phi']
                        phi2 = point2['phi']

                        curr_dist = arc_length_calculator(phi1, theta1, phi2, theta2)

                        if curr_dist < min_dist:
                            min_dist = curr_dist

        self.min_ref_dist = min_dist

    def calc_added_kmeans_dict(self):
        self.added_points = {}

        for key in self.ref_path_group_data:
            self.added_points[key] = []

            for i in range(len(self.ref_path_group_data[key])):
                points = self.ref_path_group_data[key][i]['points']

                for j in range(len(points)):
                    if points[j]['arc'] > self.min_ref_dist:
                        self.added_points[key].append(points[j])

    def assign_comp_added(self, comp_cutoff, comp_tolerance, comp_key):
        for path_group in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[path_group])):
                group_WRMSD = self.ref_path_group_data[path_group][i][comp_key + '_group_WRMSD']
                added_ref_group_WRMSD = self.ts_ref_added.ref_path_group_data[path_group][i][comp_key + '_group_WRMSD']

                if group_WRMSD != 'n/a' and added_ref_group_WRMSD != 'n/a' and (group_WRMSD < comp_cutoff
                                                    or added_ref_group_WRMSD / group_WRMSD >= comp_tolerance):

                    self.ref_path_group_data[path_group][i][comp_key + '_comp'] = True
                else:
                    self.ref_path_group_data[path_group][i][comp_key + '_comp'] = False

    def assign_comp_orig(self, comp_cutoff, comp_tolerance, comp_key):
        for path_group in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[path_group])):
                group_WRMSD = self.ref_path_group_data[path_group][i][comp_key + '_group_WRMSD']
                ref_group_WRMSD = self.ts_ref.ref_path_group_data[path_group][i][comp_key + '_group_WRMSD']

                if group_WRMSD != 'n/a' and ref_group_WRMSD != 'n/a' and (group_WRMSD < comp_cutoff
                                                    or ref_group_WRMSD / group_WRMSD >= comp_tolerance):

                    self.ref_path_group_data[path_group][i][comp_key + '_comp'] = True
                else:
                    self.ref_path_group_data[path_group][i][comp_key + '_comp'] = False

    def assign_closest_puckers(self):
        for group_key in self.path_group_data:
            for i in self.path_group_data[group_key]:
                # list for 3 shortest arclengths and their lm_groups
                arc_lengths = {}

                ts_phi = float(self.path_group_data[group_key][i]['phi'])
                ts_theta = float(self.path_group_data[group_key][i]['theta'])

                for k in range(len(self.lm_class.cano_points['pucker'])):
                    pucker_phi = float(self.lm_class.cano_points['phi_cano'][k])
                    pucker_theta = float(self.lm_class.cano_points['theta_cano'][k])

                    arc_lengths[self.lm_class.cano_points['pucker'][k]] = arc_length_calculator(
                        ts_phi, ts_theta,
                        pucker_phi,
                        pucker_theta)

                ordered_arc_lengths = OrderedDict(
                    sorted(arc_lengths.items(), key=itemgetter(1), reverse=False))
                ordered_list = []
                three_shortest_list = []

                for key, val in ordered_arc_lengths.items():
                    ordered_list.append([key, val])

                for k in range(3):
                    three_shortest_list.append(ordered_list[k])

                self.path_group_data[group_key][i]['closest_puckers'] = three_shortest_list

        return

    def assign_group_name(self):
        for group_key in self.path_group_data:
            for i in self.path_group_data[group_key]:
                tolerance = 0.25

                first_arc = \
                self.path_group_data[group_key][i]['closest_puckers'][0][1]
                second_arc = \
                self.path_group_data[group_key][i]['closest_puckers'][1][1]

                total_arc = first_arc + second_arc

                first_weight = first_arc / total_arc

                if first_weight < tolerance:
                    self.path_group_data[group_key][i]['name'] = \
                    self.path_group_data[group_key][i]['closest_puckers'][0][0]
                else:
                    self.path_group_data[group_key][i]['name'] = \
                    self.path_group_data[group_key][i]['closest_puckers'][0][0] \
                    + '/' \
                    + self.path_group_data[group_key][i]['closest_puckers'][1][0]

        return

    def populate_avg_ts(self):
        for path_group in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[path_group])):
                mean_phi = 0
                mean_theta = 0

                for j in range(len(self.ref_path_group_data[path_group][i]['points'])):
                    point = self.ref_path_group_data[path_group][i]['points'][j]

                    mean_phi += point['weighting'] * point['phi']
                    mean_theta += point['weighting'] * point['theta']

                self.ref_path_group_data[path_group][i]['mean_phi'] = mean_phi
                self.ref_path_group_data[path_group][i]['mean_theta'] = mean_theta

    # assign group keys associated with north and south
    def circ_groups_init(self):
        self.north_groups = []
        self.south_groups = []

        self.ref_north_groups = []
        self.ref_south_groups = []

        first_group = 0
        last_group = len(self.lm_class.groups_dict) - 1

        for key in self.path_group_data:
            if int(key.split("_")[0]) == first_group:
                self.north_groups.append(key)
            elif int(key.split("_")[1]) == last_group:
                self.south_groups.append(key)

        for key in self.ref_path_group_data:
            if int(key.split("_")[0]) == first_group:
                self.ref_north_groups.append(key)
            elif int(key.split("_")[1]) == last_group:
                self.ref_south_groups.append(key)

    # create all the necessary directories
    def dir_init(self):
        if not os.path.exists(os.path.join(self.ts_dir, 'circ_and_rect_plots')):
            os.makedirs(os.path.join(self.ts_dir, 'circ_and_rect_plots'))

        self.plot_save_dir = os.path.join(self.ts_dir, 'circ_and_rect_plots')

        if not os.path.exists(os.path.join(self.plot_save_dir, self.method)):
            os.makedirs(os.path.join(self.plot_save_dir, self.method))

        self.met_data_dir = os.path.join(self.plot_save_dir, self.method)

        if not os.path.exists(os.path.join(self.met_data_dir, 'raw_LMs')):
            os.makedirs(os.path.join(self.met_data_dir, 'raw_LMs'))

        self.raw_data_dir = os.path.join(self.met_data_dir, 'raw_LMs')

        if not os.path.exists(os.path.join(self.met_data_dir, 'single_LMs')):
            os.makedirs(os.path.join(self.met_data_dir, 'single_LMs'))

        self.single_data_dir = os.path.join(self.met_data_dir, 'single_LMs')

        if not os.path.exists(os.path.join(self.plot_save_dir, 'all_groupings')):
            os.makedirs(os.path.join(self.plot_save_dir, 'all_groupings'))

        self.all_groupings_dir = os.path.join(self.plot_save_dir, 'all_groupings')

        if not os.path.exists(os.path.join(self.met_data_dir, 'group_comp')):
            os.makedirs(os.path.join(self.met_data_dir, 'group_comp'))

        self.group_comp_dir = os.path.join(self.met_data_dir, 'group_comp')

        if not os.path.exists(os.path.join(self.plot_save_dir, 'heatmaps')):
            os.makedirs(os.path.join(self.plot_save_dir, 'heatmaps'))

        self.heatmap_data_dir = os.path.join(self.plot_save_dir, 'heatmaps')

        if not os.path.exists(os.path.join(self.heatmap_data_dir, 'by_arclength')):
            os.makedirs(os.path.join(self.heatmap_data_dir, 'by_arclength'))

        self.arc_data_dir = os.path.join(self.heatmap_data_dir, 'by_arclength')

        if not os.path.exists(os.path.join(self.heatmap_data_dir, 'by_gibbs')):
            os.makedirs(os.path.join(self.heatmap_data_dir, 'by_gibbs'))

        self.gibbs_data_dir = os.path.join(self.heatmap_data_dir, 'by_gibbs')

        if not os.path.exists(os.path.join(self.plot_save_dir, 'final_comp')):
            os.makedirs(os.path.join(self.plot_save_dir, 'final_comp'))

        self.final_comp_dir = os.path.join(self.plot_save_dir, 'final_comp')

        if not os.path.exists(os.path.join(self.final_comp_dir, 'by_arclength')):
            os.makedirs(os.path.join(self.final_comp_dir, 'by_arclength'))

        self.arc_comp_dir = os.path.join(self.final_comp_dir, 'by_arclength')

        if not os.path.exists(os.path.join(self.final_comp_dir, 'by_gibbs')):
            os.makedirs(os.path.join(self.final_comp_dir, 'by_gibbs'))

        self.gibbs_comp_dir = os.path.join(self.final_comp_dir, 'by_gibbs')

        if not os.path.exists(os.path.join(self.final_comp_dir, 'all_groups')):
            os.makedirs(os.path.join(self.final_comp_dir, 'all_groups'))

        self.all_groups_comp_dir = os.path.join(self.final_comp_dir, 'all_groups')

        if not os.path.exists(os.path.join(self.ts_dir, 'diff_trend')):
            os.makedirs(os.path.join(self.ts_dir, 'diff_trend'))

        self.diff_trend_dir = os.path.join(self.ts_dir, 'diff_trend')

        return

    def populate_delta_gibbs(self):


        return

    # endregion

    # # # do_calc functions # # #
    # region
    # all calcs are on the ts pts
    def do_calcs(self):
        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                self.calc_weighting(key, i)

            if self.ts_ref is None:
                weighted_gibbs = 0

                # calc weighted gibbs to compare methods to
                for j in range(len(self.ref_path_group_data[key][i]['points'])):
                    ref_point = self.ref_path_group_data[key][i]['points'][j]
                    weighted_gibbs += ref_point['G298 (Hartrees)'] * ref_point['weighting']

                self.ref_path_group_data[key][i]['G298 (Hartrees)'] = weighted_gibbs

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                weighted_gibbs = 0

                # calc weighted gibbs to compare methods to
                if self.ts_ref is not None:
                    for j in range(len(self.ts_ref.ref_path_group_data[key][i]['points'])):
                        ref_point = self.ts_ref.ref_path_group_data[key][i]['points'][j]
                        weighted_gibbs += ref_point['G298 (Hartrees)'] * ref_point['weighting']
                else:
                    for j in range(len(self.ref_path_group_data[key][i]['points'])):
                        ref_point = self.ref_path_group_data[key][i]['points'][j]
                        weighted_gibbs += ref_point['G298 (Hartrees)'] * ref_point['weighting']

                for j in range(len(self.ref_path_group_data[key][i]['points'])):
                    point = self.ref_path_group_data[key][i]['points'][j]

                    point['gibbs'] = point['G298 (Hartrees)'] - weighted_gibbs

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                self.calc_WSS(key, i, 'arc')
                self.calc_group_RMSD(key, i, 'arc')
                self.calc_WSS(key, i, 'gibbs')
                self.calc_group_RMSD(key, i, 'gibbs')

                self.calc_WWSS(key, i, 'arc')
                self.calc_WWSS(key, i, 'gibbs')
                self.calc_group_WRMSD(key, i, 'arc')
                self.calc_group_WRMSD(key, i, 'gibbs')

        self.calc_SSE('arc')
        self.calc_RMSD('arc')
        self.calc_SSE('gibbs')
        self.calc_RMSD('gibbs')

        self.calc_WSSE('arc')
        self.calc_WRMSD('arc')
        self.calc_WSSE('gibbs')
        self.calc_WRMSD('gibbs')

    def calc_WSS(self, lm_group, ts_group, comp_key):
        WSS = 0

        for i in range(len(self.ref_path_group_data[lm_group][ts_group]['points'])):
            comp_val = self.ref_path_group_data[lm_group][ts_group]['points'][i][comp_key]
            WSS += comp_val**2

        self.ref_path_group_data[lm_group][ts_group][comp_key + '_WSS'] = round(WSS, 5)

    def calc_group_RMSD(self, lm_group, ts_group, comp_key):
        size = len( self.ref_path_group_data[lm_group][ts_group]['points'])

        if (size == 0):
            RMSD = 'n/a'
            self.ref_path_group_data[lm_group][ts_group][comp_key + '_group_RMSD'] = RMSD
        else:
            RMSD = ( self.ref_path_group_data[lm_group][ts_group][comp_key + '_WSS'] / size) ** 0.5
            self.ref_path_group_data[lm_group][ts_group][comp_key + '_group_RMSD'] = round(RMSD, 5)

    def calc_SSE(self, comp_key):
        SSE = 0

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                SSE +=  self.ref_path_group_data[key][i][comp_key + '_WSS']

        self.overall_data[comp_key + '_SSE'] = round(SSE, 5)

    def calc_RMSD(self, comp_key):
        size = 0

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                size += 1

        RMSD = (self.overall_data[comp_key + '_SSE'] / size) ** 0.5
        self.overall_data[comp_key + '_RMSD'] = round(RMSD, 5)


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

        # self.ref_path_group_data[lm_group][ts_group]['G298 (Hartrees)'] = round(wt_gibbs, 3)

    def calc_WWSS(self, lm_group, ts_group, comp_key):
        WWSS = 0

        points = self.ref_path_group_data[lm_group][ts_group]['points']

        # calculating each point's contribution
        for i in range(len(points)):
            comp_val = points[i][comp_key]
            weighting = points[i]['weighting']

            WWSS += (comp_val ** 2) * weighting

        self.ref_path_group_data[lm_group][ts_group][comp_key + '_WWSS'] = round(WWSS, 5)

    def calc_group_WRMSD(self, lm_group, ts_group, comp_key):
        points = self.ref_path_group_data[lm_group][ts_group]['points']

        size = len(points)

        if (size == 0):
            WRMSD = 'n/a'
            self.ref_path_group_data[lm_group][ts_group][comp_key + '_group_WRMSD'] = WRMSD
        else:
            WRMSD = (self.ref_path_group_data[lm_group][ts_group][comp_key + '_WWSS'] / size) ** 0.5
            self.ref_path_group_data[lm_group][ts_group][comp_key + '_group_WRMSD'] = round(WRMSD, 5)

    def calc_WSSE(self, comp_key):
        WSSE = 0

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                WSSE += self.ref_path_group_data[key][i][comp_key + '_WWSS']

        self.overall_data[comp_key + '_WSSE'] = round(WSSE, 5)

    def calc_WRMSD(self, comp_key):
        size = 0

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                size += 1

        WRMSD = (self.overall_data[comp_key + '_WSSE'] / size) ** 0.5
        self.overall_data[comp_key + '_WRMSD'] = round(WRMSD, 5)

    def calc_num_comp_paths(self, comp_cutoff, comp_tolerance, comp_key):
        comparable_paths = 0

        for path_group in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[path_group])):
                group_WRMSD = self.ref_path_group_data[path_group][i][comp_key + '_group_WRMSD']
                ref_group_WRMSD = self.ts_ref.ref_path_group_data[path_group][i][comp_key + '_group_WRMSD']

                if group_WRMSD != 'n/a' and ref_group_WRMSD != 'n/a' and (
                        group_WRMSD < comp_cutoff or ref_group_WRMSD / group_WRMSD >= comp_tolerance):

                    comparable_paths += 1
                    self.ref_path_group_data[path_group][i]['comparable'] = True


        return comparable_paths
    # endregion

    # # # plotting functions # # #
    # region

    # # # raw data plotting # # #
    #region
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
        for i in self.path_group_data[path_group]:
            ts_vert = pol2cart([self.path_group_data[path_group][i]['phi'], self.path_group_data[path_group][i]['theta']])
            lm1_vert = pol2cart([self.path_group_data[path_group][i]['lm1']['phi'], self.path_group_data[path_group][i]['lm1']['theta']])
            lm2_vert = pol2cart([self.path_group_data[path_group][i]['lm2']['phi'], self.path_group_data[path_group][i]['lm2']['theta']])

            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')

            if self.north_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')
            if self.south_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')

    def plot_path_group_single(self, path_group):
        lm1_key = int(path_group.split('_')[0])
        lm2_key = int(path_group.split('_')[1])

        lm1_data = self.lm_class.groups_dict[lm1_key]
        lm2_data = self.lm_class.groups_dict[lm2_key]

        lm1_vert = pol2cart([float(lm1_data['mean_phi']), float(lm1_data['mean_theta'])])
        lm2_vert = pol2cart([float(lm2_data['mean_phi']), float(lm2_data['mean_theta'])])

        for i in self.path_group_data[path_group]:
            ts_vert = pol2cart([self.path_group_data[path_group][i]['phi'], self.path_group_data[path_group][i]['theta']])

            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')

            if self.north_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'blue', 60], [lm1_vert, 'green', 60], 'red')
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'blue', 60], [lm2_vert, 'green', 60], 'red')
            if self.south_groups.count(path_group) == 1:
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

            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'gray', 30], [lm1_vert, 'green', 60], 'gray', '-.')
            plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'gray', 30], [lm2_vert, 'green', 60], 'gray', '-.')

            if self.ref_north_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'gray', 30], [lm1_vert, 'green', 60], 'gray', '-.')
                plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'gray', 30], [lm2_vert, 'green', 60], 'gray', '-.')
            if self.ref_south_groups.count(path_group) == 1:
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'gray', 30], [lm1_vert, 'green', 60], 'gray', '-.')
                plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'gray', 30], [lm2_vert, 'green', 60], 'gray', '-.')

    def plot_all_path_groups_raw(self):
        for key in self.path_group_data:
            self.plot_path_group_raw(key)

    def plot_all_path_groups_single(self):
        for key in self.path_group_data:
            self.plot_path_group_single(key)

    def plot_all_ref_path_groups_single(self):
        for key in self.ref_path_group_data:
            self.plot_ref_path_group_single(key)
    #endregion

    # # # group comp plotting # # #
    #region
    def plot_group_comp(self, path_group, color_coord_dict, comp_key):
        coord_tolerance = 30

        lm1_key = int(path_group.split('_')[0])
        lm2_key = int(path_group.split('_')[1])

        lm1_data = self.lm_class.groups_dict[lm1_key]
        lm2_data = self.lm_class.groups_dict[lm2_key]

        lm1_vert = pol2cart([float(lm1_data['mean_phi']), float(lm1_data['mean_theta'])])
        lm2_vert = pol2cart([float(lm2_data['mean_phi']), float(lm2_data['mean_theta'])])

        for i in range(len(self.ref_path_group_data[path_group])):
            point = self.ref_path_group_data[path_group][i]
            color_phi = point['mean_phi']

            cmap = plt.get_cmap('Paired')
            seed_num = 0
            color = cmap(seed_num)

            list_of_excluded_colors = []

            for comp_phi in color_coord_dict:
                if np.abs(color_phi - comp_phi) < coord_tolerance:
                    list_of_excluded_colors.append(color_coord_dict[comp_phi])


            count = 0
            while is_excluded(color, list_of_excluded_colors):
                seed_num += 0.085

                if seed_num == 0.85:
                    seed_num += 0.085

                color = cmap(seed_num)

                count += 1

                if count > 10:
                    break

            color_coord_dict[color_phi] = color

            # if no data matches to current ts group
            if point[comp_key + '_group_WRMSD'] == 'n/a':
                ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'],
                                    self.ref_path_group_data[path_group][i]['theta']])

                plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30], [lm1_vert, 'green', 60], 'gray', ':',
                          'gray')
                plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30], [lm2_vert, 'green', 60], 'gray', ':',
                          'gray')

                if self.ref_north_groups.count(path_group) == 1:
                    plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30],
                                   [lm1_vert, 'green', 60], 'gray', ':', 'gray')
                    plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30],
                                   [lm2_vert, 'green', 60], 'gray', ':', 'gray')
                if self.ref_south_groups.count(path_group) == 1:
                    plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30],
                                   [lm1_vert, 'green', 60], 'gray', ':', 'gray')
                    plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30],
                                   [lm2_vert, 'green', 60], 'gray', ':', 'gray')
            else:
                ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'],
                                          self.ref_path_group_data[path_group][i]['theta']])

                group_ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['mean_phi'],
                                    self.ref_path_group_data[path_group][i]['mean_theta']])

                plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30], [lm1_vert, 'green', 60], color, '--', color, 20)
                plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30], [lm2_vert, 'green', 60], color, '--', color, 20)

                plot_line(self.ts_class.plot.ax_rect, [group_ts_vert, color, 30], [lm1_vert, 'green', 60],
                          color, '-', 'face', 30)
                plot_line(self.ts_class.plot.ax_rect, [group_ts_vert, color, 30], [lm2_vert, 'green', 60],
                          color, '-', 'face', 30)

                if self.north_groups.count(path_group) == 1:
                    plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30],
                                   [lm1_vert, 'green', 60], color, '--', color, 20)
                    plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30],
                                   [lm2_vert, 'green', 60], color, '--', color, 20)

                    plot_on_circle(self.ts_class.plot.ax_circ_north, [group_ts_vert, color, 30],
                                   [lm1_vert, 'green', 60], color, '-', 'face', 30)
                    plot_on_circle(self.ts_class.plot.ax_circ_north, [group_ts_vert, color, 30],
                                   [lm2_vert, 'green', 60], color, '-', 'face', 30)
                if self.south_groups.count(path_group) == 1:
                    plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30],
                                   [lm1_vert, 'green', 60], color, '--', color, 20)
                    plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30],
                                   [lm2_vert, 'green', 60], color, '--', color, 20)

                    plot_on_circle(self.ts_class.plot.ax_circ_south, [group_ts_vert, color, 30],
                                   [lm1_vert, 'green', 60], color, '-', 'face', 30)
                    plot_on_circle(self.ts_class.plot.ax_circ_south, [group_ts_vert, color, 30],
                                   [lm2_vert, 'green', 60], color, '-', 'face', 30)
    #endregion

    # # # comparison plotting # # #
    #region
    def plot_WRMSD_comp(self, comp_key):
        # plotting heatmap
        for path_group in self.ref_path_group_data:
            lm1_key = int(path_group.split('_')[0])
            lm2_key = int(path_group.split('_')[1])

            lm1_data = self.lm_class.groups_dict[lm1_key]
            lm2_data = self.lm_class.groups_dict[lm2_key]

            lm1_vert = pol2cart([float(lm1_data['mean_phi']), float(lm1_data['mean_theta'])])
            lm2_vert = pol2cart([float(lm2_data['mean_phi']), float(lm2_data['mean_theta'])])

            for i in range(len(self.ref_path_group_data[path_group])):
                point = self.ref_path_group_data[path_group][i]
                ref_point = self.ts_ref.ref_path_group_data[path_group][i]

                # if no data matches to current ts group
                if point[comp_key + '_group_WRMSD'] == 'n/a':
                    ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'],
                                        self.ref_path_group_data[path_group][i]['theta']])

                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30], [lm1_vert, 'green', 60], 'gray', '-.', 'black')
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30], [lm2_vert, 'green', 60], 'gray', '-.', 'black')

                    if self.ref_north_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30], [lm1_vert, 'green', 60], 'gray', '-.', 'black')
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30], [lm2_vert, 'green', 60], 'gray', '-.', 'black')
                    if self.ref_south_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30], [lm1_vert, 'green', 60], 'gray', '-.', 'black')
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30], [lm2_vert, 'green', 60], 'gray', '-.', 'black')
                elif point[comp_key + '_group_WRMSD'] > self.comp_cutoff and float(ref_point[comp_key + '_group_WRMSD']) / point[comp_key + '_group_WRMSD'] < self.comp_tolerance:
                    ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'],
                                        self.ref_path_group_data[path_group][i]['theta']])

                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'black', 30], [lm1_vert, 'green', 60], 'black', '-.')
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'black', 30], [lm2_vert, 'green', 60], 'black', '-.')

                    if self.north_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'black', 30],
                                       [lm1_vert, 'green', 60], 'black', '-.')
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'black', 30],
                                       [lm2_vert, 'green', 60], 'black', '-.')
                    if self.south_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'black', 30],
                                       [lm1_vert, 'green', 60], 'black', '-.')
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'black', 30],
                                       [lm2_vert, 'green', 60], 'black', '-.')
                else:
                    ts_color = 'blue'

                    ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'], self.ref_path_group_data[path_group][i]['theta']])

                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, ts_color, 30], [lm1_vert, 'green', 60], 'red', '-')
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, ts_color, 30], [lm2_vert, 'green', 60], 'red', '-')

                    if self.north_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, ts_color, 30],
                                       [lm1_vert, 'green', 60], 'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, ts_color, 30],
                                       [lm2_vert, 'green', 60], 'red', '-')
                    if self.south_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, ts_color, 30],
                                       [lm1_vert, 'green', 60], 'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, ts_color, 30],
                                       [lm2_vert, 'green', 60], 'red', '-')

    def plot_diff_trend(self, plot):
        plot.ax_rect.set_xlabel('Transition State')
        plot.ax_rect.set_ylabel('G298 (kcal/mol)')

        size = 0

        line_energy = []
        ref_line_energy = []

        x_points = []

        ymax = 0

        for key in self.ref_path_group_data:
            for i in range(len(self.ref_path_group_data[key])):
                ref_point = self.ref_path_group_data[key][i]

                y_val = 0

                for j in range(len(self.ref_path_group_data[key][i]['points'])):
                    point = self.ref_path_group_data[key][i]['points'][j]

                    y_val += point['weighting'] * point['G298 (Hartrees)']

                if y_val > ymax:
                    ymax = y_val

                if ref_point['G298 (Hartrees)'] > ymax:
                    ymax = ref_point['G298 (Hartrees)']

                size += 1

                line_energy.append(y_val)
                ref_line_energy.append(ref_point['G298 (Hartrees)'])

                x_points.append(ref_point['name'])

        index = np.arange(size)
        bar_width = 0.8
        opacity = 0.8

        plt.bar(index, ref_line_energy, bar_width,
                alpha=opacity,
                align='center',
                color='black',
                label='REFERENCE')

        plt.bar(index, line_energy, bar_width,
                alpha=opacity,
                align='center',
                color='red',
                label=self.method)

        plt.title('Energy Trends')
        plt.legend()

        plt.xticks(index)

        plot.ax_rect.set_ylim([0, ymax])
        plot.ax_rect.set_xticks(index)
        plot.ax_rect.set_xticklabels(x_points, rotation=90)

        major_ticksy = np.arange(0, ymax, 5)
        minor_ticksy = np.arange(0, ymax, 2.5)

        plot.ax_rect.set_yticks(major_ticksy)
        plot.ax_rect.set_yticks(minor_ticksy, minor=True)
    #endregion

    # # # heatmap plotting # # #
    #region
    def plot_WRMSD_heatmap(self, comp_key):
        # plotting heatmap
        for path_group in self.ref_path_group_data:
            lm1_key = int(path_group.split('_')[0])
            lm2_key = int(path_group.split('_')[1])

            lm1_data = self.lm_class.groups_dict[lm1_key]
            lm2_data = self.lm_class.groups_dict[lm2_key]

            lm1_vert = pol2cart([float(lm1_data['mean_phi']), float(lm1_data['mean_theta'])])
            lm2_vert = pol2cart([float(lm2_data['mean_phi']), float(lm2_data['mean_theta'])])

            for i in range(len(self.ref_path_group_data[path_group])):
                point = self.ref_path_group_data[path_group][i]
                ref_point = self.ts_ref.ref_path_group_data[path_group][i]

                # if no data matches to current ts group
                if point[comp_key + '_group_WRMSD'] == 'n/a':
                    ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'],
                                        self.ref_path_group_data[path_group][i]['theta']])

                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30], [lm1_vert, 'green', 60], 'gray', '-.', 'black')
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30], [lm2_vert, 'green', 60], 'gray', '-.', 'black')

                    if self.ref_north_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30], [lm1_vert, 'green', 60], 'gray', '-.', 'black')
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30], [lm2_vert, 'green', 60], 'gray', '-.', 'black')
                    if self.ref_south_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30], [lm1_vert, 'green', 60], 'gray', '-.', 'black')
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30], [lm2_vert, 'green', 60], 'gray', '-.', 'black')
                else:
                    ts_color = 'blue'

                    if point[comp_key + '_group_WRMSD'] == 0:
                        ts_size = 1
                    else:
                        ts_size = ref_point[comp_key + '_group_WRMSD'] / point[comp_key + '_group_WRMSD']

                    ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'], self.ref_path_group_data[path_group][i]['theta']])

                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'black', 30 ], [lm1_vert, 'green', 60], 'red', '-')
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'black', 30], [lm2_vert, 'green', 60], 'red', '-')

                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, ts_color, 30 * ts_size], [lm1_vert, 'green', 60], 'red', '-')
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, ts_color, 30 * ts_size], [lm2_vert, 'green', 60], 'red', '-')

                    if self.north_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'black', 30],
                                       [lm1_vert, 'green', 60], 'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'black', 30],
                                       [lm2_vert, 'green', 60], 'red', '-')

                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, ts_color, 30 * ts_size],
                                       [lm1_vert, 'green', 60], 'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, ts_color, 30 * ts_size],
                                       [lm2_vert, 'green', 60], 'red', '-')
                    if self.south_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'black', 30],
                                       [lm1_vert, 'green', 60], 'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'black', 30],
                                       [lm2_vert, 'green', 60], 'red', '-')

                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, ts_color, 30 * ts_size],
                                       [lm1_vert, 'green', 60], 'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, ts_color, 30 * ts_size],
                                       [lm2_vert, 'green', 60], 'red', '-')

    def plot_RMSD_heatmap(self, comp_key):
        # plotting heatmap
        for path_group in self.ref_path_group_data:
            lm1_key = int(path_group.split('_')[0])
            lm2_key = int(path_group.split('_')[1])

            lm1_data = self.lm_class.groups_dict[lm1_key]
            lm2_data = self.lm_class.groups_dict[lm2_key]

            lm1_vert = pol2cart([float(lm1_data['mean_phi']), float(lm1_data['mean_theta'])])
            lm2_vert = pol2cart([float(lm2_data['mean_phi']), float(lm2_data['mean_theta'])])

            for i in range(len(self.ref_path_group_data[path_group])):
                point = self.ref_path_group_data[path_group][i]
                ref_point = self.ts_ref.ref_path_group_data[path_group][i]

                # if no data matches to current ts group
                if point[comp_key + '_group_RMSD'] == 'n/a':
                    ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'],
                                        self.ref_path_group_data[path_group][i]['theta']])

                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30],
                              [lm1_vert, 'green', 60], 'gray', '-.', 'black')
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'white', 30],
                              [lm2_vert, 'green', 60], 'gray', '-.', 'black')

                    if self.ref_north_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30],
                                   [lm1_vert, 'green', 60], 'gray', '-.', 'black')
                        plot_on_circle(self.ts_class.plot.ax_circ_north, [ts_vert, 'white', 30],
                                       [lm2_vert, 'green', 60], 'gray', '-.', 'black')
                    if self.ref_south_groups.count(path_group) == 1:
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30],
                                       [lm1_vert, 'green', 60], 'gray', '-.', 'black')
                        plot_on_circle(self.ts_class.plot.ax_circ_south, [ts_vert, 'white', 30],
                                       [lm2_vert, 'green', 60], 'gray', '-.', 'black')
                else:
                    ts_color = 'blue'

                    if point[comp_key + '_group_RMSD'] == 0:
                        ts_size = 1
                    else:
                        ts_size = ref_point[comp_key + '_group_RMSD'] / point[comp_key + '_group_RMSD']

                    ts_vert = pol2cart([self.ref_path_group_data[path_group][i]['phi'],
                                        self.ref_path_group_data[path_group][i]['theta']])

                    # ref pt
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'black', 30],
                              [lm1_vert, 'green', 60], 'red', '-')
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, 'black', 30],
                              [lm2_vert, 'green', 60], 'red', '-')

                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, ts_color, 30 * ts_size],
                              [lm1_vert, 'green', 60], 'red', '-')
                    plot_line(self.ts_class.plot.ax_rect, [ts_vert, ts_color, 30 * ts_size],
                              [lm2_vert, 'green', 60], 'red', '-')

                    if self.north_groups.count(path_group) == 1:
                        # ref pt
                        plot_on_circle(self.ts_class.plot.ax_circ_north,
                                       [ts_vert, 'black', 30], [lm1_vert, 'green', 60],
                                       'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_north,
                                       [ts_vert, 'black', 30], [lm2_vert, 'green', 60],
                                       'red', '-')

                        plot_on_circle(self.ts_class.plot.ax_circ_north,
                                       [ts_vert, ts_color, 30 * ts_size], [lm1_vert, 'green', 60],
                                       'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_north,
                                       [ts_vert, ts_color, 30 * ts_size], [lm2_vert, 'green', 60],
                                       'red', '-')
                    if self.south_groups.count(path_group) == 1:
                        # ref pt
                        plot_on_circle(self.ts_class.plot.ax_circ_south,
                                       [ts_vert, 'black', 30], [lm1_vert, 'green', 60],
                                       'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_south,
                                       [ts_vert, 'black', 30], [lm2_vert, 'green', 60],
                                       'red', '-')

                        plot_on_circle(self.ts_class.plot.ax_circ_south,
                                       [ts_vert, ts_color, 30 * ts_size], [lm1_vert, 'green', 60],
                                       'red', '-')
                        plot_on_circle(self.ts_class.plot.ax_circ_south,
                                       [ts_vert, ts_color, 30 * ts_size], [lm2_vert, 'green', 60],
                                       'red', '-')
    #endregion

    def set_title_and_legend(self, artist_list, label_list):
        self.ts_class.plot.ax_rect.legend(artist_list,
                                          label_list,
                                          scatterpoints=1, fontsize=8, frameon=False, framealpha=0.75,
                                          bbox_to_anchor=(0.5, -0.3), loc=9, borderaxespad=0, ncol=4).set_zorder(100)

    def show(self):
        self.ts_class.show()
    # endregion

    # # # organization functions # # #
    #region
    # returns whether a pair of local mins are connected by a transition state in the REFERENCE data
    def is_a_ref_path(self, path_group):
        if path_group in self.ref_path_group_data:
            return True
        else:
            return False
    #endregion

    # # # saving functions # # #
    #region
    def save_all_figures_raw(self, overwrite):
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

        base_name = "z_dataset-" + self.molecule + "-TS-" + self.method

        for key in self.path_group_data:
            if not os.path.exists(os.path.join(self.raw_data_dir, base_name + '-' + key + '.png')) or overwrite:
                # saves a plot of each group individually plotted
                self.plot_path_group_raw(key)
                self.ts_class.plot_cano()

                self.set_title_and_legend(artist_list, label_list)

                self.ts_class.plot.save(base_name + '-' + key, self.raw_data_dir)
                self.ts_class.wipe_plot()

    def save_all_figures_single(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')
        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', marker='s', linestyle='-.')

        artist_list = [ref_path_Artist, cano_lm_Artist, met_lm_Artist, met_ts_Artist, path_Artist]
        label_list = ['REFERENCE pathway', 'Canonical Designation', 'LM Kmeans Center', self.method + ' TS', 'Pathway']

        base_name = "z_dataset-" + self.molecule + "-TS-" + self.method

        for key in self.path_group_data:
            if not os.path.exists(os.path.join(self.single_data_dir, base_name + '-' + key + '.png')) or overwrite:
                # saves a plot of each group individually plotted
                self.plot_path_group_single(key)
                if key in self.ref_path_group_data:
                    self.plot_ref_path_group_single(key)
                self.ts_class.plot_cano()

                self.set_title_and_legend(artist_list, label_list)

                self.ts_class.plot.save(base_name + '-' + key, self.single_data_dir)
                self.ts_class.wipe_plot()

    def save_all_groupings(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')
        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', marker='s', linestyle='-.')

        artist_list = [ref_path_Artist, cano_lm_Artist, met_lm_Artist, met_ts_Artist, path_Artist]
        label_list = ['REFERENCE pathway', 'Canonical Designation', 'LM Kmeans Center', self.method + ' TS', 'Pathway']

        base_name = "z_dataset-" + self.molecule + "-TS-" + self.method

        if not os.path.exists(os.path.join(self.all_groupings_dir, base_name + '.png')) or overwrite:
            # saves a plot of each group individually plotted
            self.plot_all_path_groups_single()
            self.plot_all_ref_path_groups_single()
            self.ts_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name, self.all_groupings_dir)
            self.ts_class.wipe_plot()

    def save_group_comp(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')

        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='black')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                    edgecolor='black')

        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='black', linestyle='--')
        ref_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                    edgecolor='black')

        no_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', linestyle=':')
        no_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                   edgecolor='gray')

        artist_list = [(no_path_Artist, no_ts_Artist), cano_lm_Artist, met_lm_Artist, (ref_path_Artist, ref_ts_Artist),
                       (path_Artist, met_ts_Artist)]
        label_list = ['no pathway', 'Canonical Designation', 'LM Kmeans Center', 'REFERENCE pathway',
                      self.method + ' pathway']

        for path_group in self.ref_path_group_data:
            base_name = "z_dataset-" + self.molecule + "-TS-" + path_group + "-comp-" + self.method

            color_coord_dict = {}

            if not os.path.exists(os.path.join(self.group_comp_dir, base_name + '.png')) or overwrite:
                self.plot_group_comp(path_group, color_coord_dict, 'arc')
                self.ts_class.plot_cano()

                self.set_title_and_legend(artist_list, label_list)

                self.ts_class.plot.save(base_name, self.group_comp_dir)
                self.ts_class.wipe_plot()

    def save_all_groups_comp(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')

        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='black')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                    edgecolor='black')

        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='black', linestyle='--')
        ref_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                       edgecolor='black')

        no_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', linestyle=':')
        no_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                    edgecolor='gray')

        artist_list = [(no_path_Artist, no_ts_Artist), cano_lm_Artist, met_lm_Artist, (ref_path_Artist, ref_ts_Artist), (path_Artist, met_ts_Artist)]
        label_list = ['no pathway', 'Canonical Designation', 'LM Kmeans Center', 'REFERENCE pathway', self.method + ' pathway']

        base_name = "z_dataset-" + self.molecule + "-TS-" + "-all_groups_comp-" + self.method

        if not os.path.exists(os.path.join(self.all_groups_comp_dir, base_name + '.png')) or overwrite:
            color_coord_dict = {}
            for path_group in self.ref_path_group_data:
                self.plot_group_comp(path_group, color_coord_dict, 'arc')

            self.ts_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name, self.all_groups_comp_dir)
            self.ts_class.wipe_plot()

    def save_WRMSD_comp(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        uncomp_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                    edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')
        no_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', linestyle='--')
        uncomp_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='black', linestyle='--')
        no_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                    edgecolor='black')

        artist_list = [(no_path_Artist, no_ts_Artist), cano_lm_Artist, met_lm_Artist, (uncomp_ts_Artist, uncomp_path_Artist), (path_Artist, met_ts_Artist)]
        label_list = ['no pathway', 'Canonical Designation', 'LM Kmeans Center', 'uncomparable pathway', 'comparable pathway']

        base_name = "z_dataset-" + self.molecule + "-TS-WRMSD-comp-" + self.method

        if not os.path.exists(os.path.join(self.arc_comp_dir, base_name + '.png')) or overwrite:
            self.plot_WRMSD_comp('arc')
            self.ts_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name, self.arc_comp_dir)
            self.ts_class.wipe_plot()

    def save_WRMSD_heatmap(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        ref_met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                    edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')
        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', linestyle='-.')
        ref_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                    edgecolor='black')

        artist_list = [(ref_path_Artist, ref_ts_Artist), cano_lm_Artist, met_lm_Artist, ref_met_ts_Artist, met_ts_Artist, path_Artist]
        label_list = ['No pathway found', 'Canonical Designation', 'LM Kmeans Center', 'REFERENCE TS', self.method + ' TS', 'Pathway']

        base_name = "z_dataset-" + self.molecule + "-TS-WRMSD-heatmap-" + self.method

        if not os.path.exists(os.path.join(self.arc_data_dir, base_name + '.png')) or overwrite:
            self.plot_WRMSD_heatmap('arc')
            self.ts_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name, self.arc_data_dir)
            self.ts_class.wipe_plot()

    def save_RMSD_heatmap(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        ref_met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                        edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')
        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', linestyle='-.')
        ref_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                    edgecolor='black')

        artist_list = [(ref_path_Artist, ref_ts_Artist), cano_lm_Artist, met_lm_Artist, ref_met_ts_Artist,
                       met_ts_Artist, path_Artist]
        label_list = ['No pathway found', 'Canonical Designation', 'LM Kmeans Center', 'REFERENCE TS',
                      self.method + ' TS', 'Pathway']

        base_name = "z_dataset-" + self.molecule + "-TS-RMSD-heatmap-" + self.method

        if not os.path.exists(os.path.join(self.arc_data_dir, base_name + '.png')) or overwrite:
            self.plot_RMSD_heatmap('arc')
            self.ts_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name, self.arc_data_dir)
            self.ts_class.wipe_plot()

    def save_gibbs_WRMSD_comp(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        uncomp_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                       edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')
        no_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', linestyle='--')
        uncomp_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='black', linestyle='--')
        no_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                   edgecolor='black')

        artist_list = [(no_path_Artist, no_ts_Artist), cano_lm_Artist, met_lm_Artist,
                       (uncomp_ts_Artist, uncomp_path_Artist), (path_Artist, met_ts_Artist)]
        label_list = ['no pathway', 'Canonical Designation', 'LM Kmeans Center', 'uncomparable pathway',
                      'comparable pathway']

        base_name = "z_dataset-" + self.molecule + "-TS-WRMSD-comp-" + self.method

        if not os.path.exists(os.path.join(self.gibbs_comp_dir, base_name + '.png')) or overwrite:
            self.plot_WRMSD_comp('gibbs')
            self.ts_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name, self.gibbs_comp_dir)
            self.ts_class.wipe_plot()

    def save_gibbs_WRMSD_heatmap(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        ref_met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                        edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')
        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', linestyle='-.')
        ref_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                    edgecolor='black')

        artist_list = [(ref_path_Artist, ref_ts_Artist), cano_lm_Artist, met_lm_Artist, ref_met_ts_Artist,
                       met_ts_Artist, path_Artist]
        label_list = ['No pathway found', 'Canonical Designation', 'LM Kmeans Center', 'REFERENCE TS',
                      self.method + ' TS', 'Pathway']

        base_name = "z_dataset-" + self.molecule + "-TS-WRMSD-heatmap-" + self.method

        if not os.path.exists(os.path.join(self.gibbs_data_dir, base_name + '.png')) or overwrite:
            self.plot_WRMSD_heatmap('gibbs')
            self.ts_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name, self.gibbs_data_dir)
            self.ts_class.wipe_plot()

    def save_gibbs_RMSD_heatmap(self, overwrite):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        ref_met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                        edgecolor='face')
        met_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='blue', marker='s',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='red')
        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='gray', linestyle='-.')
        ref_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='white', marker='s',
                                    edgecolor='black')

        artist_list = [(ref_path_Artist, ref_ts_Artist), cano_lm_Artist, met_lm_Artist, ref_met_ts_Artist,
                       met_ts_Artist, path_Artist]
        label_list = ['No pathway found', 'Canonical Designation', 'LM Kmeans Center', 'REFERENCE TS',
                      self.method + ' TS', 'Pathway']

        base_name = "z_dataset-" + self.molecule + "-TS-RMSD-heatmap-" + self.method

        if not os.path.exists(os.path.join(self.gibbs_data_dir, base_name + '.png')) or overwrite:
            self.plot_RMSD_heatmap('gibbs')
            self.ts_class.plot_cano()

            self.set_title_and_legend(artist_list, label_list)

            self.ts_class.plot.save(base_name, self.gibbs_data_dir)
            self.ts_class.wipe_plot()

    def save_diff_trend(self, overwrite):
        base_name = self.molecule + "-diff-trend-" + self.method

        if not os.path.exists(os.path.join(self.diff_trend_dir, base_name + '.png')) or overwrite:
            plot = Plots(rect_arg=True)
            self.plot_diff_trend(plot)

            plot.save(base_name, self.diff_trend_dir)
    # endregion

    pass

class Compare_All_Methods:
    def __init__(self, methods_ts_data_in, ts_dir_in, methods_lm_data_in=None, lm_dir_in=None):
        self.methods_lm_data = methods_lm_data_in

        self.lm_dir = lm_dir_in
        self.group_RMSD_vals = {}
        self.group_WRMSD_vals = {}

        self.methods_ts_data = methods_ts_data_in

        self.reorg_ts_methods()

        if self.methods_lm_data is not None:
            self.reorg_lm_methods()

        self.ts_dir = ts_dir_in
        self.dir_init()
    # # # init functions # # #
    #region
    def reorg_ts_methods(self):
        aux_dict = {}
        aux_list = []

        for i in range(len(self.methods_ts_data)):
            method = self.methods_ts_data[i].method

            if method == 'REFERENCE':
                aux_dict[0] = self.methods_ts_data[i]
            elif method == 'B3LYP':
                aux_dict[1] = self.methods_ts_data[i]
            elif method == 'DFTB':
                aux_dict[2] = self.methods_ts_data[i]
            elif method == 'AM1':
                aux_dict[3] = self.methods_ts_data[i]
            elif method == 'PM3':
                aux_dict[4] = self.methods_ts_data[i]
            elif method == 'PM3MM':
                aux_dict[5] = self.methods_ts_data[i]
            elif method == 'PM6':
                aux_dict[6] = self.methods_ts_data[i]

        for i in range(len(self.methods_ts_data)):
            aux_list.append(aux_dict[i])

        self.methods_ts_data = aux_list

    def reorg_lm_methods(self):
        aux_dict = {}
        aux_list = []

        for i in range(len(self.methods_lm_data)):
            method = self.methods_ts_data[i].method

            if method == 'REFERENCE':
                aux_dict[0] = self.methods_lm_data[i]
            elif method == 'B3LYP':
                aux_dict[1] = self.methods_lm_data[i]
            elif method == 'DFTB':
                aux_dict[2] = self.methods_lm_data[i]
            elif method == 'AM1':
                aux_dict[3] = self.methods_lm_data[i]
            elif method == 'PM3':
                aux_dict[4] = self.methods_lm_data[i]
            elif method == 'PM3MM':
                aux_dict[5] = self.methods_lm_data[i]
            elif method == 'PM6':
                aux_dict[6] = self.methods_lm_data[i]

        for i in range(len(self.methods_lm_data)):
            aux_list.append(aux_dict[i])

        self.methods_lm_data = aux_list

    def dir_init(self):
        if not os.path.exists(os.path.join(self.methods_ts_data[0].ts_dir, 'csv_data')):
            os.makedirs(os.path.join(self.methods_ts_data[0].ts_dir, 'csv_data'))

        self.ts_csv_data_dir = os.path.join(self.methods_ts_data[0].ts_dir, 'csv_data')

        if self.methods_lm_data is not None:
            if not os.path.exists(os.path.join(self.methods_lm_data[0].lm_dir, 'csv_data')):
                os.makedirs(os.path.join(self.methods_lm_data[0].lm_dir, 'csv_data'))

            self.lm_csv_data_dir = os.path.join(self.methods_lm_data[0].lm_dir, 'csv_data')
    #endregion

    # # # csv writing functions # # #
    #region
    # writing to csv the extra kmeans centers
    def calc_added_kmeans_dict(self):
        self.added_skm_dict = {}
        self.added_skm_dict['File Name'] = []
        self.added_skm_dict['Pucker'] = []
        self.added_skm_dict['phi'] = []
        self.added_skm_dict['theta'] = []
        self.added_skm_dict['q_ts'] = []
        self.added_skm_dict['energy (A.U.)'] = []
        self.added_skm_dict['H298 (Hartrees)'] = []
        self.added_skm_dict['G298 (Hartrees)'] = []
        self.added_skm_dict['freq1'] = []
        self.added_skm_dict['phi_lm1'] = []
        self.added_skm_dict['theta_lm1'] = []
        self.added_skm_dict['phi_lm2'] = []
        self.added_skm_dict['theta_lm2'] = []

        self.added_points = {}

        # aggregating each lm_group's added points
        for key in self.methods_ts_data[0].ref_path_group_data:
            self.added_points[key] = []

            for i in range(len(self.methods_ts_data)):
                for j in range(len(self.methods_ts_data[i].added_points[key])):
                    self.added_points[key].append(self.methods_ts_data[i].added_points[key][j])

        for key in self.methods_ts_data[0].ref_path_group_data:
            data_points = []

            for i in range(len(self.added_points[key])):
                phi = self.added_points[key][i]['phi']
                theta = self.added_points[key][i]['theta']

                data_points.append(pol2cart([phi, theta]))

            number_clusters = 0
            # initialize so while loop happens at least once
            error = 1
            if len(data_points) > 0:
                # increasing number of clusters until error is low
                # self.methods_ts_data[0].comp_tolerance
                while error > 0.15:
                    number_clusters += 1

                    # Uses packages to calculate the k-means spherical centers
                    skm = SphericalKMeans(n_clusters=number_clusters, init='k-means++', n_init=30)
                    skm.fit(data_points)
                    skm_centers = skm.cluster_centers_
                    skm_centers = skm_centers.tolist()

                    coord_list = []

                    for i in range(len(self.added_points[key])):
                        phi1 = self.added_points[key][i]['phi']
                        theta1 = self.added_points[key][i]['theta']

                        # initialized to large value so min_dist is at least the first iteration
                        min_dist = 100

                        # finding closest skm center to current point
                        for j in range(len(skm_centers)):
                            skm_center = skm_centers[j]
                            skm_center = cart2pol(skm_center)

                            phi2 = skm_center[0]
                            theta2 = skm_center[1]

                            curr_dist = arc_length_calculator(phi1, theta1, phi2, theta2)

                            if curr_dist < min_dist:
                                min_dist = curr_dist
                                added_skm = j

                                curr_coord = [phi2, theta2]

                        coord_list.append(curr_coord)

                        self.added_points[key][i]['arc_to_added_skm'] = min_dist
                        self.added_points[key][i]['added_skm'] = added_skm

                    error = calc_error(self.added_points[key])

                for j in range(len(skm_centers)):
                    # calculate weightings
                    total_boltz = 0

                    for i in range(len(self.added_points[key])):
                        if self.added_points[key][i]['added_skm'] == j:
                            e_val = self.added_points[key][i]['G298 (Hartrees)']
                            component = math.exp(-e_val / (K_B * DEFAULT_TEMPERATURE))
                            self.added_points[key][i]['ind_boltz_added_skm'] = component
                            total_boltz += component

                    wt_gibbs = 0
                    for i in range(len(self.added_points[key])):
                        if self.added_points[key][i]['added_skm'] == j:
                            wt_gibbs += (self.added_points[key][i]['ind_boltz_added_skm'] / total_boltz) * \
                                        self.added_points[key][i]['G298 (Hartrees)']

                    skm_center = skm_centers[j]
                    skm_center = cart2pol(skm_center)

                    skm_phi = skm_center[0]
                    skm_theta = skm_center[1]

                    phi_lm1 = self.added_points[key][0]['lm1']['phi']
                    theta_lm1 = self.added_points[key][0]['lm1']['theta']
                    phi_lm2 = self.added_points[key][0]['lm2']['phi']
                    theta_lm2 = self.added_points[key][0]['lm2']['theta']

                    min_G298 = self.methods_ts_data[0].ts_ref.min_G298 * 627.509

                    self.added_skm_dict['File Name'].append('n/a')
                    self.added_skm_dict['Pucker'].append('n/a')
                    self.added_skm_dict['phi'].append(skm_phi)
                    self.added_skm_dict['theta'].append(skm_theta)
                    self.added_skm_dict['q_ts'].append('n/a')
                    self.added_skm_dict['energy (A.U.)'].append('n/a')
                    self.added_skm_dict['H298 (Hartrees)'].append('n/a')
                    self.added_skm_dict['G298 (Hartrees)'].append(round(wt_gibbs, 5) + min_G298)
                    self.added_skm_dict['freq1'].append('n/a')
                    self.added_skm_dict['phi_lm1'].append(phi_lm1)
                    self.added_skm_dict['theta_lm1'].append(theta_lm1)
                    self.added_skm_dict['phi_lm2'].append(phi_lm2)
                    self.added_skm_dict['theta_lm2'].append(theta_lm2)

        return

    def add_to_csv(self, filename):
        self.calc_added_kmeans_dict()

        # add dict of kmeans points to the .csv file
        with open(filename, 'a', newline='') as file:
            w = csv.writer(file)
            w.writerows(zip(*self.added_skm_dict.values()))

        return


    def write_lm_to_csv(self):
        if len(os.listdir(self.lm_csv_data_dir)) < 4:
            molecule = self.methods_lm_data[0].molecule

            group_RMSD_csv = os.path.join(self.lm_csv_data_dir, molecule + '-lm-group_RMSD.csv')
            group_WRMSD_csv = os.path.join(self.lm_csv_data_dir, molecule + '-lm-group_WRMSD.csv')
            WSS_csv = os.path.join(self.lm_csv_data_dir, molecule + '-lm-WSS.csv')
            WWSS_csv = os.path.join(self.lm_csv_data_dir, molecule + '-lm-WWSS.csv')

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
                method = self.methods_lm_data[i].method

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

    def write_ts_to_csv(self, comp_key):
        if len(os.listdir(self.ts_csv_data_dir)) < 9:
            molecule = self.methods_ts_data[0].molecule

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
                        group_RMSD_val = self.methods_ts_data[i].ref_path_group_data[key][j][comp_key + '_group_RMSD']
                        group_WRMSD_val = self.methods_ts_data[i].ref_path_group_data[key][j][comp_key + '_group_WRMSD']
                        WSS_val = self.methods_ts_data[i].ref_path_group_data[key][j][comp_key + '_WSS']
                        WWSS_val = self.methods_ts_data[i].ref_path_group_data[key][j][comp_key + '_WWSS']

                        group_RMSD_dict[method].append(group_RMSD_val)
                        group_WRMSD_dict[method].append(group_WRMSD_val)
                        WSS_dict[method].append(WSS_val)
                        WWSS_dict[method].append(WWSS_val)

            group_RMSD_csv = os.path.join(self.ts_csv_data_dir, molecule + '-' + comp_key + '-ts-group_RMSD.csv')
            group_WRMSD_csv = os.path.join(self.ts_csv_data_dir, molecule + '-' + comp_key + '-ts-group_WRMSD.csv')
            WSS_csv = os.path.join(self.ts_csv_data_dir, molecule + '-' + comp_key + '-ts-WSS.csv')
            WWSS_csv = os.path.join(self.ts_csv_data_dir, molecule + '-' + comp_key + '-ts-WWSS.csv')

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

    # debug information about W/RMSD calcs
    def write_debug_lm_to_csv(self):
        molecule = self.methods_lm_data[0].molecule

        debug_lm_dict = {}

        for k in range(len(self.methods_lm_data)):
            method = self.methods_lm_data[k].method

            debug_lm_dict['group'] = []
            debug_lm_dict['arclength'] = []
            debug_lm_dict['gibbs'] = []
            debug_lm_dict['weighting'] = []

            # filling the dict
            for i in range(len(self.methods_lm_data[k].group_data)):
                for key in self.methods_lm_data[k].group_data[i]['points']:
                    point = self.methods_lm_data[k].group_data[i]['points'][key]

                    debug_lm_dict['group'].append(str(i) + '_' + str(key))
                    debug_lm_dict['arclength'].append(self.methods_lm_data[k].hartree_data[key]['arc_lengths'][0][1])
                    debug_lm_dict['gibbs'].append(point['G298 (Hartrees)'])
                    debug_lm_dict['weighting'].append(point['weighting'])

            debug_lm_csv = os.path.join(os.path.join(self.lm_dir, method), molecule + '-' + method + '-debug_lm.csv')

            if not os.path.exists(os.path.join(self.lm_dir, method)):
                os.makedirs(os.path.join(self.lm_dir, method))

            with open(debug_lm_csv, 'w', newline='') as file:
                w = csv.writer(file)
                w.writerow(debug_lm_dict.keys())
                w.writerows(zip(*debug_lm_dict.values()))

        return

    def write_debug_ts_to_csv(self):
        molecule = self.methods_ts_data[0].molecule

        debug_ts_dict = {}

        debug_ts_dict['group'] = []
        debug_ts_dict['arclength'] = []
        debug_ts_dict['gibbs'] = []
        debug_ts_dict['weighting'] = []

        for k in range(len(self.methods_ts_data)):
            method = self.methods_ts_data[k].method

            # filling the dict
            for key in self.methods_ts_data[k].ref_path_group_data:
                for i in range(len(self.methods_ts_data[k].ref_path_group_data[key])):
                    for j in range(len(self.methods_ts_data[k].ref_path_group_data[key][i]['points'])):
                        point = self.methods_ts_data[k].ref_path_group_data[key][i]['points'][j]

                        debug_ts_dict['group'].append(key + '-' + str(i) + '-' + str(j))
                        debug_ts_dict['arclength'].append(point['arc_lengths'][0][1])
                        debug_ts_dict['gibbs'].append(point['G298 (Hartrees)'])
                        debug_ts_dict['weighting'].append(point['weighting'])

            debug_ts_csv = os.path.join(os.path.join(self.ts_dir, method), molecule + '-' + method + '-debug_ts.csv')

            if not os.path.exists(os.path.join(self.ts_dir, method)):
                os.makedirs(os.path.join(self.ts_dir, method))

            with open(debug_ts_csv, 'w', newline='') as file:
                w = csv.writer(file)
                w.writerow(debug_ts_dict.keys())
                w.writerows(zip(*debug_ts_dict.values()))

        return

    # writing to csv the pathways from each method which are connected but are not connected in the REFERENCE
    def write_uncompared_to_csv(self):
        molecule = self.methods_ts_data[0].molecule
        uncompared_csv = os.path.join(self.ts_csv_data_dir, molecule + '-uncompared.csv')

        if not os.path.exists(uncompared_csv):
            uncompared_dict = {}

            uncompared_dict['group'] = []
            uncompared_dict['pucker'] = []
            uncompared_dict['method'] = []

            for i in range(len(self.methods_ts_data)):
                # listing under the hood names & pucker names
                for key in self.methods_ts_data[i].path_group_data:
                    if key not in self.methods_ts_data[i].ref_path_group_data:
                        for j in self.methods_ts_data[i].path_group_data[key]:
                            uncompared_dict['group'].append(key + '-' + str(j))
                            uncompared_dict['pucker'].append(self.methods_ts_data[i].path_group_data[key][j]['name'])
                            uncompared_dict['method'].append(self.methods_ts_data[i].method)

            with open(uncompared_csv, 'w', newline='') as file:
                w = csv.writer(file)
                w.writerow(uncompared_dict.keys())
                w.writerows(zip(*uncompared_dict.values()))

    # writing to csv the number of comparable pathways for each method using both arc and gibbs comparisons
    def write_num_comp_lm_to_csv(self):
        molecule = self.methods_lm_data[0].molecule

        for i in range(len(self.methods_lm_data)):
            method = self.methods_lm_data[i].method

            num_comp_csv = os.path.join(self.methods_lm_data[i].met_data_dir, molecule + '-' + method + '-num_comp_lm.csv')

            if not os.path.exists(num_comp_csv):
                num_comp_dict = {}

                num_comp_dict['tolerance'] = []
                num_comp_dict['num_comp_lm'] = []
                num_comp_dict['accuracy'] = []

                tolerance = 0
                increment = 0.02

                while tolerance <= 1:
                    num_comp_paths = self.methods_lm_data[i].calc_num_comp_lm(0.1, tolerance)
                    ref_nump_comp_paths = self.methods_lm_data[-1].calc_num_comp_lm(0.1, tolerance)

                    num_comp_dict['tolerance'].append(tolerance)
                    num_comp_dict['num_comp_lm'].append(num_comp_paths)
                    num_comp_dict['accuracy'].append(round(num_comp_paths / ref_nump_comp_paths, 3))

                    tolerance += increment

                with open(num_comp_csv, 'w', newline='') as file:
                    w = csv.writer(file)
                    w.writerow(num_comp_dict.keys())
                    w.writerows(zip(*num_comp_dict.values()))

        return

    def write_gibbs_num_comp_lm_to_csv(self):
        molecule = self.methods_lm_data[0].molecule

        for i in range(len(self.methods_lm_data)):
            method = self.methods_lm_data[i].method

            num_comp_csv = os.path.join(self.methods_lm_data[i].met_data_dir,
                                        molecule + '-' + method + '-gibbs_num_comp_lm.csv')

            if not os.path.exists(num_comp_csv):
                num_comp_dict = {}

                num_comp_dict['tolerance'] = []
                num_comp_dict['num_comp_lm'] = []
                num_comp_dict['accuracy'] = []

                tolerance = 0
                increment = 0.02

                while tolerance <= 1:
                    num_comp_paths = self.methods_lm_data[i].calc_gibbs_num_comp_lm(0.1, tolerance)
                    ref_nump_comp_paths = self.methods_lm_data[-1].calc_gibbs_num_comp_lm(0.1, tolerance)

                    num_comp_dict['tolerance'].append(tolerance)
                    num_comp_dict['num_comp_lm'].append(num_comp_paths)
                    num_comp_dict['accuracy'].append(round(num_comp_paths / ref_nump_comp_paths, 3))

                    tolerance += increment

                with open(num_comp_csv, 'w', newline='') as file:
                    w = csv.writer(file)
                    w.writerow(num_comp_dict.keys())
                    w.writerows(zip(*num_comp_dict.values()))

    def write_num_comp_paths_to_csv(self, comp_key, ref_type):
        molecule = self.methods_ts_data[0].molecule

        for i in range(len(self.methods_ts_data)):
            method = self.methods_ts_data[i].method

            num_comp_csv = os.path.join(self.methods_ts_data[i].met_data_dir,
                                        molecule + '-' + method + '-' + comp_key + '-num_comp_paths_' + ref_type + '.csv')

            if not os.path.exists(num_comp_csv):
                num_comp_dict = {}

                num_comp_dict['tolerance'] = []
                num_comp_dict['num_comp_paths'] = []
                num_comp_dict['accuracy'] = []

                tolerance = 0
                increment = 0.02

                while tolerance <= 1:
                    num_comp_paths = self.methods_ts_data[i].calc_num_comp_paths(0.1, tolerance, comp_key)
                    ref_nump_comp_paths = self.methods_ts_data[0].calc_num_comp_paths(0.1, tolerance, comp_key)

                    num_comp_dict['tolerance'].append(tolerance)
                    num_comp_dict['num_comp_paths'].append(num_comp_paths)
                    num_comp_dict['accuracy'].append(round(num_comp_paths / ref_nump_comp_paths, 3))

                    tolerance += increment

                with open(num_comp_csv, 'w', newline='') as file:
                    w = csv.writer(file)
                    w.writerow(num_comp_dict.keys())
                    w.writerows(zip(*num_comp_dict.values()))
    #endregion

    # # # plotting functions # # #
    #region
    # plotting functions for plots with table included
    def plot_comp_table(self, path_group, data_key, ax, comp_key):
        lm1_key = int(path_group.split('_')[0])
        lm2_key = int(path_group.split('_')[1])

        lm1_data = self.methods_ts_data[0].lm_class.groups_dict[lm1_key]
        lm2_data = self.methods_ts_data[0].lm_class.groups_dict[lm2_key]

        lm1_vert = pol2cart([float(lm1_data['mean_phi']), float(lm1_data['mean_theta'])])
        lm2_vert = pol2cart([float(lm2_data['mean_phi']), float(lm2_data['mean_theta'])])

        lm1_name = lm1_data['name']
        lm2_name = lm2_data['name']

        table_rows = []
        colors = []
        row_labels = []

        colors.append('white')

        row = []

        # storing the row names
        row_labels.append(lm1_name + '-' + lm2_name)

        for i in range(len(self.methods_ts_data)):
            row_labels.append(self.methods_ts_data[i].method)

        # storing header row
        for i in range(len(self.methods_ts_data[0].ref_path_group_data[path_group])):
            row.append(path_group + '-' + str(i))

        table_rows.append(row)

        list_of_colors = []

        # plotting data and storing colors
        for i in range(len(self.methods_ts_data[0].ref_path_group_data[path_group])):
            cmap = plt.get_cmap('Paired')
            seed_num = 0
            color = cmap(seed_num)

            count = 0
            while is_excluded(color, list_of_colors):
                seed_num += 0.085

                if seed_num == 0.85:
                    seed_num += 0.085

                color = cmap(seed_num)

                count += 1

                if count > 10:
                    print('WARNING: more than 10 colors in use, some will be used twice going forward!')
                    break

            list_of_colors.append(color)

            # accounting for the potential use of the added kmeans centers
            if len(self.methods_ts_data) == 8:
                group_ts_vert = pol2cart([self.methods_ts_data[1].ref_path_group_data[path_group][i]['phi'],
                                          self.methods_ts_data[1].ref_path_group_data[path_group][i]['theta']])
            else:
                group_ts_vert = pol2cart([self.methods_ts_data[0].ref_path_group_data[path_group][i]['phi'],
                                          self.methods_ts_data[0].ref_path_group_data[path_group][i]['theta']])

            if self.methods_ts_data[0].north_groups.count(path_group) == 1 or self.methods_ts_data[0].south_groups.count(path_group) == 1:
                plot_on_circle(ax, [group_ts_vert, color, 30], [lm1_vert, 'green', 60], color, '-', 'face', 30)
                plot_on_circle(ax, [group_ts_vert, color, 30], [lm2_vert, 'green', 60], color, '-', 'face', 30)

                left_offset = 1.5
            else:
                plot_line(ax, [group_ts_vert, color, 30], [lm1_vert, 'green', 60], color, '-', 'face', 30)
                plot_line(ax, [group_ts_vert, color, 30], [lm2_vert, 'green', 60], color, '-', 'face', 30)

                left_offset = 1.2

        for j in range(len(self.methods_ts_data)):
            row = []

            for i in range(len(self.methods_ts_data[0].ref_path_group_data[path_group])):
                row.append(self.methods_ts_data[j].ref_path_group_data[path_group][i][data_key])

            table_rows.append(row)

        num_row = len(self.methods_ts_data) + 2
        num_col = len(self.methods_ts_data[0].ref_path_group_data[path_group])

        col_widths = []

        if self.methods_ts_data[0].north_groups.count(path_group) == 1\
            or self.methods_ts_data[0].south_groups.count(path_group) == 1:

            width = 0.2
        else:
            width = 0.1

        # setting column width
        for i in range(num_row):
            for j in range(num_col):
                col_widths.append(width)

        color_matrix = get_color_matrix(table_rows, data_key, comp_key)

        # bbox is [left, bottom, width, height]
        ax.table(cellText=table_rows,
                 cellLoc='center',
                 cellColours=color_matrix,
                 rowLabels=row_labels,
                 colColours=list_of_colors,
                 colWidths=col_widths,
                 loc='right',
                 bbox=[left_offset, 0, width * num_col, 1])

    def save_comp_table(self, data_key, comp_key):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')

        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='black', linestyle='--')
        ref_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                    edgecolor='black')

        artist_list = [cano_lm_Artist, met_lm_Artist, (ref_path_Artist, ref_ts_Artist)]
        label_list = ['Canonical Designation', 'LM Kmeans Center', 'pathway']

        if not os.path.exists(os.path.join(self.methods_ts_data[0].ts_dir, data_key)):
            os.makedirs(os.path.join(self.methods_ts_data[0].ts_dir, data_key))

        self.comp_data_dir = os.path.join(self.methods_ts_data[0].ts_dir, data_key)

        for path_group in self.methods_ts_data[0].ref_path_group_data:
            base_name = "z_dataset-" + self.methods_ts_data[0].molecule + "-TS-" + path_group + "-" + data_key + "-table"

            if not os.path.exists(os.path.join(self.comp_data_dir, base_name + '.png')):
                if self.methods_ts_data[0].north_groups.count(path_group) == 1:
                    plot = Plots(False, False, False, False, True, False)
                    ax = plot.ax_circ_north

                    self.plot_comp_table(path_group, data_key, ax, comp_key)
                elif self.methods_ts_data[0].south_groups.count(path_group) == 1:
                    plot = Plots(False, False, False, False, False, True)
                    ax = plot.ax_circ_south

                    self.plot_comp_table(path_group, data_key, ax, comp_key)
                else:
                    plot = Plots(False, False, False, True, False, False)

                    ax = plot.ax_rect

                    ax.scatter(self.methods_ts_data[0].lm_class.cano_points['phi_cano'],
                                self.methods_ts_data[0].lm_class.cano_points['theta_cano'],
                                s=60, c='black',
                                marker='+',
                                edgecolor='face')

                    self.plot_comp_table(path_group, data_key, ax, comp_key)

                    # must set the limits after plotting or set autoscaling off
                    ax.set_xlim([-5, 365])
                    ax.set_ylim([125, 55])

                ax.legend(artist_list,
                           label_list,
                           scatterpoints=1, fontsize=8, frameon=False, framealpha=0.75,
                           bbox_to_anchor=(0.5, -0.2), loc=9, borderaxespad=0, ncol=4).set_zorder(100)

                plot.save(base_name, self.comp_data_dir)

        self.methods_ts_data[0].ts_class.wipe_plot()

    def is_physical(self, path_group, i):
        if self.methods_ts_data[0].ref_path_group_data[path_group][i]['arc_group_WRMSD'] == 'n/a':
            return False
        else:
            return True

    def plot_all_comp_table(self, data_key, comp_key):
        table_rows = []
        row_labels = []

        # storing the row names
        row_labels.append('Paths:')

        for i in range(len(self.methods_ts_data)):
            row_labels.append(self.methods_ts_data[i].method)

        # initializing column names
        row = []

        row.append('unphysical')

        for path_group in self.methods_ts_data[0].ref_path_group_data:
            lm1_key = int(path_group.split('_')[0])
            lm2_key = int(path_group.split('_')[1])

            lm1_data = self.methods_ts_data[0].lm_class.groups_dict[lm1_key]
            lm2_data = self.methods_ts_data[0].lm_class.groups_dict[lm2_key]

            lm1_name = lm1_data['name']
            lm2_name = lm2_data['name']

            path_name = lm1_name + '-' + lm2_name

            # storing header row
            for i in range(len(self.methods_ts_data[0].ref_path_group_data[path_group])):
                if self.is_physical(path_group, i):
                    row.append(path_name + '_' + str(i))

        table_rows.append(row)

        list_of_colors = []
        list_of_colors.append('gray')

        cmap = plt.get_cmap('Vega20')
        # allows for incrementing over 20 colors
        increment = 0.0524
        seed_num = 0
        count = 0

        # plotting pathways
        for path_group in self.methods_ts_data[0].ref_path_group_data:
            lm1_key = int(path_group.split('_')[0])
            lm2_key = int(path_group.split('_')[1])

            lm1_data = self.methods_ts_data[0].lm_class.groups_dict[lm1_key]
            lm2_data = self.methods_ts_data[0].lm_class.groups_dict[lm2_key]

            lm1_vert = pol2cart([float(lm1_data['mean_phi']), float(lm1_data['mean_theta'])])
            lm2_vert = pol2cart([float(lm2_data['mean_phi']), float(lm2_data['mean_theta'])])

            # plotting data and storing colors
            for i in range(len(self.methods_ts_data[0].ref_path_group_data[path_group])):
                # accounting for the potential use of the added kmeans centers
                if len(self.methods_ts_data) == 8:
                    group_ts_vert = pol2cart([self.methods_ts_data[1].ref_path_group_data[path_group][i]['phi'],
                                              self.methods_ts_data[1].ref_path_group_data[path_group][i]['theta']])
                else:
                    group_ts_vert = pol2cart([self.methods_ts_data[0].ref_path_group_data[path_group][i]['phi'],
                                              self.methods_ts_data[0].ref_path_group_data[path_group][i]['theta']])

                linestyle = '-'

                # if pathway isn't physical, make it gray and dashed
                if not self.is_physical(path_group, i):
                    color = 'darkgray'
                    linestyle = '--'
                else:
                    if count > 20:
                        cmap = plt.get_cmap('Vega20b')
                        seed_num = 0
                        count = 0

                    color = cmap(seed_num)
                    seed_num += increment
                    count += 1

                    list_of_colors.append(color)

                plot_line(self.methods_ts_data[0].ts_class.plot.ax_rect, [group_ts_vert, color, 30],
                          [lm1_vert, 'green', 60], color, linestyle)
                plot_line(self.methods_ts_data[0].ts_class.plot.ax_rect, [group_ts_vert, color, 30],
                          [lm2_vert, 'green', 60], color, linestyle)

                if self.methods_ts_data[0].ref_north_groups.count(path_group) == 1:
                    plot_on_circle(self.methods_ts_data[0].ts_class.plot.ax_circ_north, [group_ts_vert, color, 30],
                                   [lm1_vert, 'green', 60], color, linestyle)
                    plot_on_circle(self.methods_ts_data[0].ts_class.plot.ax_circ_north, [group_ts_vert, color, 30],
                                   [lm2_vert, 'green', 60], color, linestyle)
                if self.methods_ts_data[0].ref_south_groups.count(path_group) == 1:
                    plot_on_circle(self.methods_ts_data[0].ts_class.plot.ax_circ_south, [group_ts_vert, color, 30],
                                   [lm1_vert, 'green', 60], color, linestyle)
                    plot_on_circle(self.methods_ts_data[0].ts_class.plot.ax_circ_south, [group_ts_vert, color, 30],
                                   [lm2_vert, 'green', 60], color, linestyle)

        for j in range(len(self.methods_ts_data)):
            row = []

            not_physical_count = 0

            # getting the number of non physical paths
            #need to factor non-ref lm groups
            for path_group in self.methods_ts_data[j].ref_path_group_data:
                for i in range(len(self.methods_ts_data[0].ref_path_group_data[path_group])):
                    if not self.is_physical(path_group, i)\
                        and self.methods_ts_data[j].ref_path_group_data[path_group][i][comp_key + '_group_WRMSD'] != 'n/a':

                        not_physical_count += 1

            # creating each row of the table
            row.append(not_physical_count)

            for path_group in self.methods_ts_data[j].ref_path_group_data:
                for i in range(len(self.methods_ts_data[0].ref_path_group_data[path_group])):
                    if self.is_physical(path_group, i):
                        val = self.methods_ts_data[j].ref_path_group_data[path_group][i][data_key]

                        if data_key == 'G298 (Hartrees)':
                            val = 0

                            for k in range(len(self.methods_ts_data[j].ref_path_group_data[path_group][i]['points'])):
                                point = self.methods_ts_data[j].ref_path_group_data[path_group][i]['points'][k]

                                val += point['G298 (Hartrees)'] * point['weighting']

                        if isinstance(val, float):
                            val = round(val, 3)

                        row.append(val)

            table_rows.append(row)

        num_row = len(self.methods_ts_data) + 2
        num_col = 1

        for path_group in self.methods_ts_data[0].ref_path_group_data:
            num_col += len(self.methods_ts_data[0].ref_path_group_data[path_group])

        col_widths = []

        width = 0.1

        # setting column width
        for i in range(num_row):
            for j in range(num_col):
                col_widths.append(width)

        color_matrix = get_color_matrix(table_rows, data_key, comp_key)

        # setting the first color of each row to white
        for i in range(num_row - 1):
            color_matrix[i][0] = 'white'

        # bbox is [left, bottom, width, height]
        self.methods_ts_data[0].ts_class.plot.ax_rect.table(cellText=table_rows,
                 cellLoc='center',
                 cellColours=color_matrix,
                 rowLabels=row_labels,
                 colColours=list_of_colors,
                 colWidths=col_widths,
                 loc='right',
                 bbox=[1.1, 0, width * num_col, 2])

        return

    def save_all_comp_table(self, data_key, comp_key):
        # Create custom artist
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='green', marker='o',
                                    edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60 * size_scaling, c='black', marker='+',
                                     edgecolor='face')

        ref_path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='black', linestyle='--')
        ref_ts_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30 * size_scaling, c='black', marker='s',
                                    edgecolor='black')

        artist_list = [cano_lm_Artist, met_lm_Artist, (ref_path_Artist, ref_ts_Artist)]
        label_list = ['Canonical Designation', 'LM Kmeans Center', 'pathway']

        base_name = "z_dataset-" + self.methods_ts_data[0].molecule + "-TS-all-groups-" + data_key + "-table"

        if data_key == 'G298 (Hartrees)':
            if not os.path.exists(os.path.join(self.methods_ts_data[0].ts_dir, data_key.split(' ')[1])):
                os.makedirs(os.path.join(self.methods_ts_data[0].ts_dir, data_key.split(' ')[1]))

            save_dir = os.path.join(self.methods_ts_data[0].ts_dir, data_key.split(' ')[1])
        else:
            if not os.path.exists(os.path.join(self.methods_ts_data[0].ts_dir, data_key.split('_')[1])):
                os.makedirs(os.path.join(self.methods_ts_data[0].ts_dir, data_key.split('_')[1]))

            save_dir = os.path.join(self.methods_ts_data[0].ts_dir, data_key.split('_')[1])

        if not os.path.exists(os.path.join(save_dir, base_name + '.png')) or\
            not os.path.exists(os.path.join(self.ts_dir, base_name + '.png')):

            self.plot_all_comp_table(data_key, comp_key)
            self.methods_ts_data[0].ts_class.plot_cano()

            self.methods_ts_data[0].ts_class.plot.ax_rect.legend(artist_list,
                       label_list,
                       scatterpoints=1, fontsize=8, frameon=False, framealpha=0.75,
                       bbox_to_anchor=(0.5, -0.3), loc=9, borderaxespad=0, ncol=4).set_zorder(100)

            self.methods_ts_data[0].ts_class.plot.save(base_name, save_dir)
            self.methods_ts_data[0].ts_class.plot.save(base_name, self.ts_dir)

    def plot_diff_trend_by_path(self, plot, path_group, i):
        size = 0

        line_energy = []
        ref_line_energy = []

        x_points = []

        ymax = 0

        for j in range(len(self.methods_ts_data)):
            ref_val = 0

            for k in range(len(self.methods_ts_data[0].ts_ref.ref_path_group_data[path_group][i]['points'])):
                point = self.methods_ts_data[0].ts_ref.ref_path_group_data[path_group][i]['points'][k]

                ref_val += point['weighting'] * point['G298 (Hartrees)']

            met_val = 0

            for k in range(len(self.methods_ts_data[j].ref_path_group_data[path_group][i]['points'])):
                point = self.methods_ts_data[j].ref_path_group_data[path_group][i]['points'][k]

                met_val += point['weighting'] * point['G298 (Hartrees)']

            if met_val > ymax:
                ymax = met_val

            if ref_val > ymax:
                ymax = ref_val

            size += 1

            line_energy.append(met_val)
            ref_line_energy.append(ref_val)

            x_points.append(self.methods_ts_data[j].method)

        title = 'Energy Trends - ' + self.methods_ts_data[0].ref_path_group_data[path_group][i]['name']

        plot_diff_trend(plot, ref_line_energy, line_energy, x_points, size, ymax, title)

    def save_diff_trend_by_path(self):
        if not os.path.exists(os.path.join(self.methods_ts_data[0].diff_trend_dir, 'by_path')):
            os.makedirs(os.path.join(self.methods_ts_data[0].diff_trend_dir, 'by_path'))

        by_path_dir = os.path.join(self.methods_ts_data[0].diff_trend_dir, 'by_path')

        for path_group in self.methods_ts_data[0].ref_path_group_data:
            for i in range(len(self.methods_ts_data[0].ref_path_group_data[path_group])):
                base_name = self.methods_ts_data[0].molecule + "-diff-trend-" + path_group + '-' + str(i)

                if not os.path.exists(os.path.join(by_path_dir, base_name + '.png')):
                    plot = Plots(rect_arg=True)
                    self.plot_diff_trend_by_path(plot, path_group, i)

                    plot.save(base_name, by_path_dir)

    def plot_diff_trend_by_met(self, plot, j):
        size = 0

        line_energy = []
        ref_line_energy = []

        x_points = []

        ymax = 0

        for path_group in self.methods_ts_data[0].ref_path_group_data:
            for i in range(len(self.methods_ts_data[0].ref_path_group_data[path_group])):
                if(self.is_physical(path_group, i)):
                    ref_val = 0

                    for k in range(len(self.methods_ts_data[0].ts_ref.ref_path_group_data[path_group][i]['points'])):
                        point = self.methods_ts_data[0].ts_ref.ref_path_group_data[path_group][i]['points'][k]

                        ref_val += point['weighting'] * point['G298 (Hartrees)']

                    met_val = 0

                    for k in range(len(self.methods_ts_data[j].ref_path_group_data[path_group][i]['points'])):
                        point = self.methods_ts_data[j].ref_path_group_data[path_group][i]['points'][k]

                        met_val += point['weighting'] * point['G298 (Hartrees)']

                    if met_val > ymax:
                        ymax = met_val

                    if ref_val > ymax:
                        ymax = ref_val

                    size += 1

                    line_energy.append(met_val)
                    ref_line_energy.append(ref_val)

                    x_points.append(self.methods_ts_data[0].ref_path_group_data[path_group][i]['name'])

        title = 'Energy Trends - ' + self.methods_ts_data[j].method

        plot_diff_trend(plot, ref_line_energy, line_energy, x_points, size, ymax, title, self.methods_ts_data[j].method)

    def save_diff_trend_by_met(self):
        if not os.path.exists(os.path.join(self.methods_ts_data[0].diff_trend_dir, 'by_met')):
            os.makedirs(os.path.join(self.methods_ts_data[0].diff_trend_dir, 'by_met'))

        by_met_dir = os.path.join(self.methods_ts_data[0].diff_trend_dir, 'by_met')

        for j in range(len(self.methods_ts_data)):
            base_name = self.methods_ts_data[0].molecule + "-diff-trend-" + self.methods_ts_data[j].method

            if not os.path.exists(os.path.join(by_met_dir, base_name + '.png')):
                plot = Plots(rect_arg=True)
                self.plot_diff_trend_by_met(plot, j)

                plot.save(base_name, by_met_dir)
    #endregion

    pass
#endregion
