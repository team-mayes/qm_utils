#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this script is to organize the pathway information for the various puckering interconversion pathways
"""

from __future__ import print_function

import argparse
import math
import os
import sys
import fnmatch
import numpy as np
import pandas as pd

from qm_utils.pucker_table import read_hartree_files_lowest_energy

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

# Constants #
HARTREE_TO_KCALMOL = 627.5095
K_B = 0.001985877534  # Boltzmann Constant in kcal/mol K

# Defaults #

DEFAULT_TEMPERATURE = 298.15

# Field Headers #
FILE_NAME = 'File Name'
PUCKER    = 'Pucker'
ENERGY_ELECTRONIC = 'Energy (A.U.)'
THETA = 'theta'
PHI   = 'phi'
GIBBS = 'G298 (Hartrees)'
ENTH  = "H298 (Hartrees)"
Q_VAL = 'Q'

# Puckers #
LIST_PUCKER = ['4c1', '14b', '25b', 'o3b', '1h2', '2h3', '3h4', '4h5', '5ho', 'oh1', '1s3', '1s5', '2so',  '1e',  '2e',
                '3e',  '4e',  '5e',  'oe', '1c4', 'b14', 'b25', 'bo3', '2h1', '3h2', '4h3', '5h4', 'oh5', '1ho', '3s1',
               '5s1', 'os2',  'e1',  'e2',  'e3',  'e4',  'e5',  'eo']

# Functions #

def read_pathway_information(file, csv_filename):
    """ Organizes and collects all of the pathway information! If there are multiple pathways with the same
        three puckers, then they are grouped together and a weighted boltzman distribution is computed for them.

    :param file: the text file containing the pathway file information
    :param csv_filename: a hartree CSV file containing all of the lmirc and TS structures for analysis
    :return: the qm method used and a dictionary of all of the pathways
    """

    method_dict = {}
    hartree_headers, lowest_energy_dict, qm_method = \
        read_hartree_files_lowest_energy(csv_filename, os.path.dirname(csv_filename))
    for row in lowest_energy_dict:
        method_dict[row[FILE_NAME]] = row

    main_dict = {}
    pathway_list = {}
    pathway_multiple = []

    f = open(file, mode='r')
    count = 0
    for line in f:
        if count == 0:
            method = line.split('#')
            count += 1
        else:
            line_info = line.split('#')
            pathway = str(line_info[0]) + '-' + str(line_info[2]) + '-' + str(line_info[4])
            filenames = str(line_info[1]) + '#' + str(line_info[3]) + '#' + str(line_info[5]).strip('\n')

            if pathway not in pathway_list:
                pathway_list[pathway] = int(0)
                mini_dict = {}
                mini_dict[pathway] = pathway
                mini_dict['files'] = filenames
                main_dict[pathway + '$' + str(pathway_list[pathway])] = mini_dict
            else:
                if pathway_list[pathway] == 0:
                    pathway_multiple.append(pathway)
                pathway_list[pathway] += int(1)
                mini_dict = {}
                mini_dict[pathway] = pathway
                mini_dict['files'] = filenames
                main_dict[pathway + '$' + str(pathway_list[pathway])] = mini_dict

    pathway_dict = {}

    for dupe_pathway in pathway_multiple:
        gibbs_lm1 = []
        gibbs_ts = []
        gibbs_lm2 = []
        enth_lm1 = []
        enth_ts = []
        enth_lm2 = []
        for large_dict_keys in main_dict.keys():
            if large_dict_keys.split('$')[0] == dupe_pathway:
                ind_pathway = {}
                files = main_dict[large_dict_keys]['files'].split('#')
                gibbs_lm1.append(method_dict[files[0]][GIBBS])
                gibbs_ts.append(method_dict[files[1]][GIBBS])
                gibbs_lm2.append(method_dict[files[2]][GIBBS])
                enth_lm1.append(method_dict[files[0]][ENTH])
                enth_ts.append(method_dict[files[1]][ENTH])
                enth_lm2.append(method_dict[files[2]][ENTH])
                if len(gibbs_lm2) == pathway_list[dupe_pathway] + 1:
                    ind_pathway['lm1'] = perform_pucker_boltzmann_weighting_gibbs(gibbs_lm1, enth_lm1)
                    ind_pathway['ts'] = perform_pucker_boltzmann_weighting_gibbs(gibbs_ts, enth_ts)
                    ind_pathway['lm2'] = perform_pucker_boltzmann_weighting_gibbs(gibbs_lm2, enth_lm2)
                    ind_pathway['dupe'] = str(pathway_list[dupe_pathway] + 1)

        pathway_dict[dupe_pathway] = ind_pathway

    for large_dict_keys in main_dict.keys():
        ind_pathway = {}
        if large_dict_keys.split('$')[0] not in pathway_multiple:
            files = main_dict[large_dict_keys]['files'].split('#')
            ind_pathway['lm1'] = round(method_dict[files[0]][ENTH], 4)
            ind_pathway['ts'] = round(method_dict[files[1]][ENTH], 4)
            ind_pathway['lm2'] = round(method_dict[files[2]][ENTH], 4)
            ind_pathway['dupe'] = str('0')
            pathway_dict[large_dict_keys.split('$')[0]] = ind_pathway

    return method[0], pathway_dict, method_dict, main_dict


def perform_pucker_boltzmann_weighting_gibbs(gibbs_list, enth_list):
    """ The script focuses on performing the boltzmann weighting for the puckering pathways. The boltzmann weighting
        is completed based on Gibbs free energy (with the weight applied to Enthalpy).

    :param gibbs_list: list of the energies associated with a particular pucker
    :param enth_list: list of enthalpies associated with a particular pucker
    :return: the weighted energy for the pathway
    """
    pucker_total_weight_gibbs = 0
    weight_gibbs = []
    puckering_weight = []

    for energy in gibbs_list:
        try:
            boltz_weight = math.exp(-energy / (DEFAULT_TEMPERATURE * K_B))
        finally:
            weight_gibbs.append(boltz_weight)
            pucker_total_weight_gibbs += boltz_weight

    for value in weight_gibbs:
        try:
            individual_weight = value / pucker_total_weight_gibbs
        finally:
            puckering_weight.append(individual_weight)

    if abs(sum(puckering_weight) - 1.0000) > 0.01:
        print('The sum of the boltzmann weights equals {} (should equal 1.0).'
              .format(abs(sum(puckering_weight))))

    weighted_value = round(sum(np.array(puckering_weight) * np.array(enth_list)), 4)

    return weighted_value


def find_hartree_csv_file(file, directory):
    """ This script just helps find the correct hartree file for the pathway file"

    :param file: the filename containing all of the pathways
    :param directory: the directory that the files are located in
    :return: the hartree file corresponding to the pathway file
    """
    base = os.path.basename(file).split('_')
    method_single_file = base[len(base)-1].split('.')[0]
    hartree_file = fnmatch.filter(os.listdir(directory), '*' + str(method_single_file) + '.csv')

    return hartree_file[0]


def id_key_pathways(main_dict, method_dict, pucker_interest):
    """ This script finds all of the pathways containing the loacl min that is in being studied.

    :param main_dict: the main dict of all of the pathways (including duplicates)
    :param method_dict: the hartree output dict where the key is the hartree input file
    :param pucker_interest: the preferred pucker of interest (must be lowest local min in pathway)
    :return: a list of the interesting pathways, and a list of all of the pathways (including multiples)

    """
    path_interest = []
    multiple_pathways = []

    for pathway in main_dict:
        count = pathway.split('$')
        puckers = count[0].split('-')
        if puckers[0] == pucker_interest:
            path_interest.append(pathway)
            if count[1] != 0 and count[0] not in multiple_pathways:
                multiple_pathways.append(count[0])

    return path_interest, multiple_pathways

def comparing_pathways_between_methods(path_interest_b3lyp, multiple_pathways_b3lyp, method_dict_b3lyp,
                                       path_interest_compare, multiple_pathways_compare, method_dict_compare):

    print(path_interest_b3lyp)

    for pathway_b3lyp in path_interest_b3lyp:
        pathway_b3lyp_puckers = pathway_b3lyp.split('$')[0].split('-')
        for pathway_compa in path_interest_compare:
            pathway_compa_puckers = pathway_compa.split('$')[0].split('-')
            if pathway_b3lyp_puckers[0] != pathway_compa_puckers[0]:
                print('The initial puckers are different! Something doesn\'t seem right.')
            else:
                if pathway_b3lyp_puckers[1] == pathway_compa_puckers[1]:
                    print(pathway_b3lyp_puckers)
                    print(pathway_compa_puckers)


            #
            # print(path_interest_b3lyp)
            # print(path_interest_b3lyp.split('-')[1])
            # if path_interest_b3lyp.split('-')[1] == path_interest_compare.split('-')[1]:
            #     print('hi')

    # for row in path_interest:
    #     for rows in main_dict:
    #         if row == rows:
    #             files = main_dict[row]['files'].split('#')
    #             lm1 = files[0]
    #             ts0 = files[1]
    #             lm2 = files[2]
    #
    #             params_lm1 = [method_dict[lm1][PUCKER], method_dict[lm1][PHI], method_dict[lm1][THETA]]
    #             params_ts0 = [method_dict[ts0][PUCKER], method_dict[ts0][PHI], method_dict[ts0][THETA]]
    #             params_lm2 = [method_dict[lm2][PUCKER], method_dict[lm2][PHI], method_dict[lm2][THETA]]

    return


def create_minimum_pathways():
    pass

# Command Line Parser #

def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="This script is designed to combine the pathway information for all "
                                                 "conformational landscapes generated by different methods for a "
                                                 "particular molecule. The input is a list of the pucker "
                                                 "interconversion pathways that are generated by IRC calculations "
                                                 "using gaussian.")
    parser.add_argument('-f', "--single_file", help="A single pathway file generated from igor_meractor.py",
                        default=None)
    parser.add_argument('-s', "--sum_file", help="List of the .txt files that need to be read into the script",
                        default=None)
    parser.add_argument('-d', "--dir", help="The dictory that contains all of the pathway files.",
                        default=None)
    parser.add_argument('-c', "--ccsdt", help="The file containing the CCSDT information for the pathways.")

    args = None
    args = parser.parse_args(argv)

    if args.dir and args.sum_file is None:
        if args.single_file is not None:
            GOOD_RET = 'good'
        else:
            GOOD_RET = 'No'
    else:
        GOOD_RET = 'good'

    return args, GOOD_RET


# Main #
def main(argv=None):
    """
    Runs the main program
    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """

    args, ret = parse_cmdline(argv)

    if ret != 'No':
        if args.single_file is not None:

            hartree_file = find_hartree_csv_file(args.single_file, args.dir)
            method, pathway_dict, method_dict, main_dict = read_pathway_information(args.single_file,
                                                                         os.path.join(args.dir, hartree_file))


            path_interest, multiple_pathways = id_key_pathways(main_dict, method_dict, pucker_interest='4c1')


            # print(pd.DataF rame(pathway_dict))

        else:
            list = open(args.sum_file, mode='r')
            method_dict = {}
            for row in list:
                hartree_file = find_hartree_csv_file(row.strip('\n'), args.dir)
                method, pathway_dict, single_method_dict, main_dict = read_pathway_information(os.path.join(args.dir,
                                                            row.strip('\n')), os.path.join(args.dir, hartree_file))
                method_dict[method] = pathway_dict

            overall_pathways = {}

            for method_keys in method_dict.keys():
                for pathway_keys in method_dict[method_keys].keys():
                    if pathway_keys not in overall_pathways:
                        overall_pathways[pathway_keys] = [method_keys]
                    elif pathway_keys in overall_pathways:
                        overall_pathways[pathway_keys].append(method_keys)


            for overall_pathways_keys in overall_pathways.keys():
                # print(overall_pathways_keys, overall_pathways[overall_pathways_keys])

                for row in LIST_PUCKER:
                    if row == overall_pathways_keys.split('-')[1]:
                        print(overall_pathways_keys, overall_pathways[overall_pathways_keys])

                # if 'am1' and 'am1full' in overall_pathways[overall_pathways_keys]:
                #     print(overall_pathways_keys, overall_pathways[overall_pathways_keys])


        # print(pd.DataFrame(pathway_dict))

if __name__ == '__main__':
    status = main()
    sys.exit(status)
