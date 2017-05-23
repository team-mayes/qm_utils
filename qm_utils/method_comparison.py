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
from qm_utils.spherical_kmeans_voronoi import Local_Minima, read_csv_canonical_designations, read_csv_data

# # # header stuff # # #
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


# # # Number of Clusters # # #
NUM_CLUSTERS_BXYL = 9

# # # Classes # # #
#region
class Local_Minima_Compare():
    """
    class for organizing the local minima information
    """
    def __init__(self, method_in, parsed_hartree, lm_class_in):
        self.hartree_data = []
        self.lm_class = lm_class_in
        self.group_data = []
        self.overall_data = {}
        self.overall_data['method'] = method_in
        self.group_rows = []

        # converting hartrees to kcal/mol
        for i in range(len(parsed_hartree)):
            parsed_hartree[i]['G298 (Hartrees)'] = 627.509 * float(parsed_hartree[i]['G298 (Hartrees)'])

        self.populate_hartree_data(parsed_hartree)
        self.populate_groupings()
        self.do_calcs()
        self.populate_print_data()

    # # # __init__ functions # # #
    #region
    def populate_hartree_data(self, parsed_hartree):
        for i in range(len(parsed_hartree)):
            self.hartree_data.append({})

            self.hartree_data[i]['G298 (Hartrees)'] = float(parsed_hartree[i]['G298 (Hartrees)'])
            self.hartree_data[i]['pucker'] = parsed_hartree[i]['Pucker']
            self.hartree_data[i]['phi'] = float(parsed_hartree[i]['phi'])
            self.hartree_data[i]['theta'] = float(parsed_hartree[i]['theta'])

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

            for j in range(len(self.hartree_data)):
                if self.hartree_data[j]['arc_lengths'][0][0] == i:
                    self.group_data[i]['points'][j] = self.hartree_data[j]

        return

    def populate_print_data(self):
        for i in range(len(self.group_data)):
            row = []
            row.append(self.group_data[i]['method'])
            row.append(self.group_data[i]['group_RMSD'])
            row.append(self.group_data[i]['group_WRMSD'])
            row.append(self.group_data[i]['WSS'])
            row.append(self.group_data[i]['WWSS'])

            self.group_rows.append(row)

        overall_row = []
        overall_row.append(self.overall_data['method'])
        overall_row.append(self.overall_data['RMSD'])
        overall_row.append(self.overall_data['WRMSD'])
        overall_row.append(self.overall_data['SSE'])
        overall_row.append(self.overall_data['WSSE'])

        self.overall_row = overall_row
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
                                            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0).set_zorder(100)

        plt.title(self.overall_data['method'], loc='left')


    def show(self):
        self.lm_class.show()
    #endregion

    # # # saving functions # # #
    def save_all_figures(self):
        # Create custom artist
        size_scaling = 1
        ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=30*size_scaling, c='red', marker='o', edgecolor='face')
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=15*size_scaling, c='blue', marker='o', edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60*size_scaling, c='black', marker='+', edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')

        artist_list = [ref_lm_Artist, met_lm_Artist, path_Artist, cano_lm_Artist]
        label_list = ['Reference LM', 'Method LM', 'Voronoi Edge', 'Canonical Designation']

        base_name = "z_dataset-bxyl-LM-" + self.overall_data['method']
        MET_DATA_DIR = os.path.join(LM_DIR, self.overall_data['method'])

        OVERALL_DIR = os.path.join(MET_DATA_DIR, 'overall')
        # checks if directory exists, and creates it if not
        if not os.path.exists(OVERALL_DIR):
            os.makedirs(OVERALL_DIR)

        # saves a plot of all groupings
        self.plot_all_groupings()
        self.lm_class.plot_cano()

        self.set_title_and_legend(artist_list, label_list)

        self.lm_class.plot.save(base_name + '-all_groupings', OVERALL_DIR)
        self.lm_class.wipe_plot()

        for i in range(len(self.group_data)):
            # saves a plot of each group individually plotted
            self.plot_grouping(i)
            self.lm_class.plot_cano()
            GROUPS_DIR = os.path.join(MET_DATA_DIR, 'groups')
            # checks if directory exists, and creates it if not
            if not os.path.exists(GROUPS_DIR):
                os.makedirs(GROUPS_DIR)

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name + '-group_' + str(i), GROUPS_DIR)
            self.lm_class.wipe_plot()

            # saves a plot of a focused view of each group
            self.plot_window(i)
            self.lm_class.plot_cano()
            WINDOWED_DIR = os.path.join(MET_DATA_DIR, 'groups_windowed')
            # checks if directory exists, and creates it if not
            if not os.path.exists(WINDOWED_DIR):
                os.makedirs(WINDOWED_DIR)

            self.set_title_and_legend(artist_list, label_list)

            self.lm_class.plot.save(base_name + '-group_' + str(i) + '-windowed', WINDOWED_DIR)
            self.lm_class.wipe_plot()

#TODO: streamline the plot to also include just the raw data

    def save_all_figures_raw(self):
        # Create custom artists
        size_scaling = 1
        met_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=15*size_scaling, c='blue', marker='o', edgecolor='face')
        raw_ref_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60*size_scaling, c='black', marker='o', edgecolor='face')
        cano_lm_Artist = plt.scatter((5000, 5000), (4999, 4999), s=60*size_scaling, c='black', marker='+', edgecolor='face')
        path_Artist = plt.Line2D((5000, 5000), (4999, 4999), c='green')

        artist_list = [raw_ref_lm_Artist, met_lm_Artist, path_Artist, cano_lm_Artist]
        label_list = ['Raw Reference LM', 'Method LM', 'Voronoi Edge', 'Canonical Designation']

        base_name = "z_dataset-bxyl-LM-" + self.overall_data['method']
        MET_DATA_DIR = os.path.join(LM_DIR, self.overall_data['method'])

        OVERALL_DIR = os.path.join(MET_DATA_DIR, 'overall')
        # checks if directory exists, and creates it if not
        if not os.path.exists(OVERALL_DIR):
            os.makedirs(OVERALL_DIR)

        # saves plot of all groupings with the raw group data
        # self.plot_all_groupings_raw()
        self.plot_method_data()
        self.lm_class.plot_cano()

        self.set_title_and_legend(artist_list, label_list)

        self.lm_class.plot.save(base_name + '-all_method_raw_data', OVERALL_DIR)
        self.lm_class.wipe_plot()

class Compare_All_Methods_LM:
    def __init__(self, methods_data_in, lm_dir_in):
        self.methods_data = methods_data_in
        self.lm_dir = lm_dir_in

    def write_to_txt(self, do_print):
        tables = []

        for i in range(len(self.methods_data)):
            for j in range(len(self.methods_data[0].group_rows)):
                header = []
                header.append('group_' + str(j))
                header.append('group_RMSD')
                header.append('group_WRMSD')
                header.append('WSS')
                header.append('WWSS')

                if len(tables) < len(self.methods_data[0].group_rows):
                    tables.append(PrettyTable(header))

                tables[j].add_row(self.methods_data[i].group_rows[j])

            if len(tables) < len(self.methods_data[0].group_rows) + 1:
                header = []
                header.append('overall')
                header.append('   RMSD   ')
                header.append('   WRMSD   ')
                header.append('SSE')
                header.append('WSSE')

                tables.append(PrettyTable(header))

            tables[len(tables) - 1].add_row(self.methods_data[i].overall_row)

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

    def write_to_csv(self):
        group_RMSD_dict = {}
        group_WRMSD_dict = {}
        WSS_dict = {}
        WWSS_dict = {}

        group_RMSD_dict['group'] = []
        group_WRMSD_dict['group'] = []
        WSS_dict['group'] = []
        WWSS_dict['group'] = []

        # listing group names
        for i in range(len(self.methods_data[0].group_data)):
            group_RMSD_dict['group'].append(i)
            group_WRMSD_dict['group'].append(i)
            WSS_dict['group'].append(i)
            WWSS_dict['group'].append(i)

        # filling method data for each dict
        for i in range(len(self.methods_data)):
            method = self.methods_data[i].overall_data['method']

            group_RMSD_dict[method] = []
            group_WRMSD_dict[method] = []
            WSS_dict[method] = []
            WWSS_dict[method] = []

            for j in range(len(self.methods_data[i].group_data)):
                group_RMSD_val = self.methods_data[i].group_data[j]['group_RMSD']
                group_WRMSD_val = self.methods_data[i].group_data[j]['group_WRMSD']
                WSS_val = self.methods_data[i].group_data[j]['WSS']
                WWSS_val = self.methods_data[i].group_data[j]['WWSS']

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

class Transition_State_Compare():
    """
    class for organizing the transition state information
    """
    def  __init__(self, list_of_dicts, method, hsp_ts_groups):
        print(method)
        self.ts_list_of_dicts = []
        self.irc_list_of_dicts = []

        self.separate_TS_files(list_of_dicts)

    def separate_TS_files(self, list_of_dicts):
        """
        separates out the TS files for further processing
        :param list_of_dicts:
        :return:
        """
        for row in list_of_dicts:
            if float(row[FREQ]) < 0:
                self.ts_list_of_dicts.append(row)
            elif float(row[FREQ]) > 0:
                self.irc_list_of_dicts.append(row)

        if len(self.irc_list_of_dicts) / 2 != len(self.ts_list_of_dicts):
            print('\nThere are {} TS files and {} IRC files...THERE IS A PROBLEM.\n'.
                  format(len(self.ts_list_of_dicts),len(self.irc_list_of_dicts)))

        return


        # Perform the following operations on the transition state data set:
        # (1) isolate the TS and IRC files by 'Freq 1' values and naming convention (also link files together) -- DONE.
        # (2) assign each of the transition states to a particular HSP reference group
        #       (assigning pathway will need to be on both the transition state AND the local minima connecting them)
        # (3) within each of the assign group, perform RMSD calculations on the arc length and gibbs free energies
        # (4) develop a similar plotting strategy (more thought needed)
#endregion

# # # Helper Functions # # #
#region

#endregion


# # #  Main  # # #
#region
def main():
    save = True
    mol_list_dir = os.listdir(MET_COMP_DIR)

    # for each molecule, perform the comparisons
    for i in range(len(mol_list_dir)):
        comp_mol_dir  = os.path.join(MET_COMP_DIR, mol_list_dir[i])
        sv_mol_dir = os.path.join(os.path.join(SV_DIR, 'molecules'), mol_list_dir[i])

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(comp_mol_dir, 'local_minimum')):
            os.makedirs(os.path.join(comp_mol_dir, 'local_minimum'))

        comp_lm_dir = os.path.join(comp_mol_dir, 'local_minimum')

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(comp_mol_dir, 'transitions_state')):
            os.makedirs(os.path.join(comp_mol_dir, 'transitions_state'))

        ts_dir = os.path.join(comp_mol_dir, 'transitions_state')

        # checks if directory exists, and creates it if not
        if not os.path.exists(os.path.join(comp_lm_dir, 'z_datasets-LM')):
            os.makedirs(os.path.join(comp_lm_dir, 'z_datasets-LM'))

        lm_data_dir = os.path.join(comp_lm_dir, 'z_datasets-LM')

        methods_data_list = []

        # initialization info for local minimum clustering for specific molecule
        number_clusters = NUM_CLUSTERS_BXYL
        dict_cano = read_csv_canonical_designations(mol_list_dir[i] + '-CP_params.csv', sv_mol_dir)
        data_points, phi_raw, theta_raw, energy = read_csv_data('z_' + mol_list_dir[i] + '_lm-b3lyp_howsugarspucker.csv',
                                                                sv_mol_dir)
        lm_class = Local_Minima(number_clusters, data_points, dict_cano, phi_raw, theta_raw, energy)

        # for every local min data file in the directory perform the comparison calculations
        for filename in os.listdir(lm_data_dir):
            if filename.endswith(".csv"):
                method_hartree = read_csv_to_dict(os.path.join(lm_data_dir, filename), mode='r')

                method = (filename.split('-', 3)[3]).split('.')[0]

                lm_comp_class = Local_Minima_Compare(method, method_hartree, lm_class)

                methods_data_list.append(lm_comp_class)

        comp_all_met_LM = Compare_All_Methods_LM(methods_data_list, comp_lm_dir)

        if save:
            # save the comparison data
            comp_all_met_LM.write_to_csv()

            # save all plots
            for i in range(len(methods_data_list)):
                methods_data_list[i].plot_all_groupings()
                methods_data_list[i].save_all_figures()

                #TODO: plots for the HSP raw data and the method data
                methods_data_list[i].plot_all_groupings_raw()
                methods_data_list[i].save_all_figures_raw()

    return

if __name__ == '__main__':
    status = main()
    sys.exit(status)
#endregion
