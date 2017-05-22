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
LM_DIR = os.path.join(MET_COMP_DIR, 'local_minimum')
LM_DATA_DIR = os.path.join(LM_DIR, 'z_datasets-LM')
AM1_DATA_DIR = os.path.join(LM_DIR, 'am1')

SV_DIR = os.path.join(TEST_DATA_DIR, 'spherical_kmeans_voronoi')
#endregion

NUM_CLUSTERS_BXYL = 9

# # # Input Files # # #
#region
HSP_LOCAL_MIN = 'z_lm-b3lyp_howsugarspucker.csv'
#endregion

# # # Helper Functions # # #
#region




#endregion

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

        self.lm_class.plot.ax_rect.scatter(phi, theta, s=15, c='blue', marker='o', edgecolor='face', zorder = 10)
        self.lm_class.plot.ax_rect.scatter(group_phi, group_theta, s=30, c='red', marker='s', edgecolor='face', zorder=10)

        self.lm_class.plot_vor_sec(grouping)

        return

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

    def show(self):
        self.lm_class.show()
    #endregion

    # # # saving functions # # #
    def save_all_figures(self):
        base_name = "z_dataset-bxyl-LM-" + self.overall_data['method']
        MET_DATA_DIR = os.path.join(LM_DIR, self.overall_data['method'])

        # saves a plot of all groupings
        self.plot_all_groupings()
        self.lm_class.plot_cano()
        OVERALL_DIR = os.path.join(MET_DATA_DIR, 'overall')
        # checks if directory exists, and creates it if not
        if not os.path.exists(OVERALL_DIR):
            os.makedirs(OVERALL_DIR)
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
            self.lm_class.plot.save(base_name + '-group_' + str(i), GROUPS_DIR)
            self.lm_class.wipe_plot()

            # saves a plot of a focused view of each group
            self.plot_window(i)
            self.lm_class.plot_cano()
            WINDOWED_DIR = os.path.join(MET_DATA_DIR, 'groups_windowed')
            # checks if directory exists, and creates it if not
            if not os.path.exists(WINDOWED_DIR):
                os.makedirs(WINDOWED_DIR)
            self.lm_class.plot.save(base_name + '-group_' + str(i) + '-windowed', WINDOWED_DIR)
            self.lm_class.wipe_plot()

class Compare_All_Methods_LM:
    def __init__(self, methods_data_in):
        self.methods_data = methods_data_in

    def print(self):
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
                header.append('RMSD')
                header.append('WRMSD')
                header.append('SSE')
                header.append('WSSE')

                tables.append(PrettyTable(header))

            tables[len(tables) - 1].add_row(self.methods_data[i].overall_row)

        for i in range(len(tables)):
            print(tables[i])

            table_txt = tables[i].get_string()
            with open('output.txt', 'w') as file:
                file.write(table_txt)

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

# # #  Main  # # #
#region
def main():
    methods_data_list = []

    number_clusters = NUM_CLUSTERS_BXYL
    dict_cano_bxyl = read_csv_canonical_designations('CP_params.csv', SV_DIR)
    data_points, phi_raw, theta_raw, energy = read_csv_data(HSP_LOCAL_MIN, SV_DIR)
    lm_class = Local_Minima(number_clusters, data_points, dict_cano_bxyl, phi_raw, theta_raw, energy)

    # for every local min data file in the directory perform the comparison calculations
    for filename in os.listdir(LM_DATA_DIR):
        if filename.endswith(".csv"):
            method_hartree = read_csv_to_dict(os.path.join(LM_DATA_DIR, filename), mode='r')

            # converting hartrees to kcal/mol
            for i in range(len(method_hartree)):
                method_hartree[i]['G298 (Hartrees)'] = 627.509 * float(method_hartree[i]['G298 (Hartrees)'])

            method = (filename.split('-', 3)[3]).split('.')[0]

            lm_comp_class = Local_Minima_Compare(method, method_hartree, lm_class)

            methods_data_list.append(lm_comp_class)

    comp_all_met_LM = Compare_All_Methods_LM(methods_data_list)
    comp_all_met_LM.print()

    # # save all plots
    # for i in range(len(methods_data_list)):
    #     methods_data_list[i].plot_all_groupings()
    #     methods_data_list[i].save_all_figures()

    return

if __name__ == '__main__':
    status = main()
    sys.exit(status)
#endregion


# # # # Command Line Parse # # #
# #region
# def parse_cmdline(argv):
#     """
#     Returns the parsed argument list and return code.
#     `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
#     """
#     if argv is None:
#         argv = sys.argv[1:]
#
#     # initialize the parser object:
#     parser = argparse.ArgumentParser(description="The gen_puck_table.py script is designed to combine hartree output "
#                                                  "files to compare different properties across different levels of "
#                                                  "theory. The hartree input files for a variety of levels of theory "
#                                                  "are combined to produce a new data table.")
#
#     parser.add_argument('-s', "--sum_file", help="List of csv files to read.", default=None)
#     parser.add_argument('-d', "--dir_hartree", help="The directory where the hartree files can be found.",
#                         default=None)
#     parser.add_argument('-p', "--pattern", help="The file pattern you are looking for (example: '.csv').",
#                         default=None)
#     parser.add_argument('-m', "--molecule", help="The type of molecule that is currently being studied")
#     parser.add_argument('-c', "--ccsdt", help="The CCSD(T) file for the molecule being studied",
#                         default=None)
#
#     args = None
#     try:
#         args = parser.parse_args(argv)
#         if args.sum_file is None:
#             raise InvalidDataError("Input files are required. Missing hartree input or two-file inputs")
#         elif not os.path.isfile(args.sum_file):
#             raise IOError("Could not find specified hartree summary file: {}".format(args.sum_file))
#         # Finally, if the summary file is there, and there is no dir_xyz provided
#         if args.dir_hartree is None:
#             args.dir_hartree = os.path.dirname(args.sum_file)
#         # if a  dir_xyz is provided, ensure valid
#         elif not os.path.isdir(args.dir_hartree):
#             raise InvalidDataError("Invalid path provided for '{}': ".format('-d, --dir_hartree', args.dir_hartree))
#
#     except (KeyError, InvalidDataError) as e:
#         warning(e)
#         parser.print_help()
#         return args, INPUT_ERROR
#     except IOError as e:
#         warning(e)
#         parser.print_help()
#         return args, IO_ERROR
#     except (ValueError, SystemExit) as e:
#         if e.message == 0:
#             return args, GOOD_RET
#         warning(e)
#         parser.print_help()
#         return args, INPUT_ERROR
#
#     return args, GOOD_RET
# #endregion
#
# # # # Main # # #
# #region
# def main(argv=None):
#     """
#     Runs the main program
#     :param argv: The command line arguments.
#     :return: The return code for the program's termination.
#     """
#     args, ret = parse_cmdline(argv)
#     if ret != GOOD_RET or args is None:
#         return args, ret
#     try:
#         #TODO: import the class information from spherical_kmean_voronoi.py based on the molecule type
#         # Need to come up with a simple way to local in the necessary HSP reference information based on args.molecule (database? HSP reference files?)
#
#         hsp_lm_groups = 1
#         hsp_ts_groups = 1
#
#         with open(args.sum_file) as f:
#             for csv_file_read_newline in f:
#                 csv_file_read = csv_file_read_newline.strip("\n")
#                 method_list_dicts = read_csv_to_dict(os.path.join(args.dir_hartree, csv_file_read), mode='r')
#                 qm_method = csv_file_read.split('-')[3].split('.')[0]
#
#                 if csv_file_read.split('-')[1] != args.molecule:
#                     print('\nERROR: THE MOLECULE TYPE DOES NOT MATCH UP.\n')
#                     break
#
#                 if csv_file_read.split('-')[2] == 'LM':
#                     print(csv_file_read)
#
#                     data_lm = Local_Minima_Compare(method_list_dicts, qm_method, hsp_lm_groups)
#
#                     #TODO: run a local min class on this data set...
#                 elif csv_file_read.split('-')[2] == 'TS':
#                     print(csv_file_read)
#                     #TODO: run a transition state class on this data set...
#
#                     data_ts = Transition_State_Compare(method_list_dicts, qm_method, hsp_ts_groups)
#                 else:
#                     print('WARNING: NOT SURE WHAT TYPE OF JOB THIS IS')
#
#             #TODO: come up with a method for storing the information from each run so that later on
#             #A class with the important data from the above calculations is probably the best way to go about it?
#             #Also...should write out the data to a csv file for later processing if necessary.
#
#
#
#     except IOError as e:
#         warning(e)
#         return IO_ERROR
#     except (InvalidDataError, KeyError) as e:
#         warning(e)
#         return INVALID_DATA
#
#     return GOOD_RET  # success
#
#
# if __name__ == '__main__':
#     status = main()
#     sys.exit(status)
# #endregion
