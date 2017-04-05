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

    return method[0], pathway_dict, method_dict


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


def distance_between_puckers(method_dict_info1, method_dict_info2):
    """"""
    print('phi: {}, theta: {}, Q: {}'.format(method_dict_info1[PHI], method_dict_info1[THETA], method_dict_info1[Q_VAL]))
    print('phi: {}, theta: {}, Q: {}\n'.format(method_dict_info2[PHI], method_dict_info2[THETA], method_dict_info2[Q_VAL]))


# Values for the first set of spherical coordiantes
    phi1   = 0*math.pi/180 #float(method_dict_info1[PHI]) * math.pi/180
    theta1 = 90*math.pi/180 #float(method_dict_info1[THETA]) * math.pi/180
    q_val1 = 1 #float(method_dict_info1[Q_VAL])

# Values for the second set of spherical coordinates
    phi2   = 180*math.pi/180 #float(method_dict_info2[PHI]) * math.pi/180
    theta2 = 0*math.pi/180 # float(method_dict_info2[THETA]) * math.pi/180
    q_val2 = 1 #float(method_dict_info2[Q_VAL])

# Altering spherical coordinates to cartesian coordinates
    q_val = (q_val1 + q_val2)/2
    x1 = round(q_val * math.sin(theta1) * math.cos(phi1),4)
    x2 = round(q_val * math.sin(theta2) * math.cos(phi2),4)

    y1 = round(q_val * math.sin(theta1) * math.sin(phi1),4)
    y2 = round(q_val * math.sin(theta2) * math.sin(phi2),4)

    z1 = round(q_val * math.cos(theta1),4)
    z2 = round(q_val * math.cos(theta2),4)

    print('x1: {}, y1: {}, z1: {}'.format(x1, y1, z1))
    print('x2: {}, y2: {}, z2: {}'.format(x2, y2, z2))



    distance = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2) + math.pow((z1-z2),2))

    arc_length =  q_val * math.asin((distance/(2 * q_val)) * (math.sqrt(4*math.pow(q_val,2) - math.pow(distance,2))))

    print('\n{}'.format(distance))
    print('\n{}'.format(arc_length))


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
            method, pathway_dict, method_dict = read_pathway_information(args.single_file,
                                                                         os.path.join(args.dir, hartree_file))

            print(pd.DataFrame(pathway_dict))

        else:
            list = open(args.sum_file, mode='r')
            method_dict = {}
            for row in list:
                hartree_file = find_hartree_csv_file(row.strip('\n'), args.dir)
                method, pathway_dict, single_method_dict = read_pathway_information(os.path.join(args.dir,
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
