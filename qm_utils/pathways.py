#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this script is to organize the pathway information for the various puckering interconversion pathways
"""

from __future__ import print_function

import argparse
import fnmatch
import os
import sys
import pandas as pd
import math
import numpy as np

from qm_utils.pucker_table import read_hartree_files_lowest_energy

from qm_utils.qm_common import (GOOD_RET, create_out_fname, warning, IO_ERROR,
                                InvalidDataError, INVALID_DATA, INPUT_ERROR, read_csv_to_dict, get_csv_fieldnames)

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

# Constants #
HARTREE_TO_KCALMOL = 627.5095
K_B = 0.001985877534 # Boltzmann Constant in kcal/mol K

# Defaults #

DEFAULT_TEMPERATURE = 298.15


# Field Headers #
FILE_NAME = 'File Name'
PUCKER = 'Pucker'
ENERGY_ELECTRONIC = 'Energy (A.U.)'
THETA = 'theta'
PHI = 'phi'
GIBBS = 'G298 (Hartrees)'
ENTH = "H298 (Hartrees)"

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


    dict = {}
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
                dict[pathway + '$' + str(pathway_list[pathway])] = mini_dict
            else:
                if pathway_list[pathway] == 0:
                    pathway_multiple.append(pathway)
                pathway_list[pathway] += int(1)
                mini_dict = {}
                mini_dict[pathway] = pathway
                mini_dict['files'] = filenames
                dict[pathway + '$' + str(pathway_list[pathway])] = mini_dict

    pathway_dict = {}

    for dupe_pathway in pathway_multiple:
        gibbs_lm1 = []
        gibbs_ts  = []
        gibbs_lm2 = []
        enth_lm1 = []
        enth_ts  = []
        enth_lm2 = []
        for large_dict_keys in dict.keys():
            if large_dict_keys.split('$')[0] == dupe_pathway:
                ind_pathway = {}
                files = dict[large_dict_keys]['files'].split('#')
                gibbs_lm1.append(method_dict[files[0]][GIBBS])
                gibbs_ts.append(method_dict[files[1]][GIBBS])
                gibbs_lm2.append(method_dict[files[2]][GIBBS])
                enth_lm1.append(method_dict[files[0]][ENTH])
                enth_ts.append(method_dict[files[1]][ENTH])
                enth_lm2.append(method_dict[files[2]][ENTH])
                if len(gibbs_lm2) == pathway_list[dupe_pathway]+1:
                    ind_pathway['lm1'] = perform_pucker_boltzmann_weighting_gibbs(gibbs_lm1, enth_lm1)
                    ind_pathway['ts']  = perform_pucker_boltzmann_weighting_gibbs(gibbs_ts , enth_ts)
                    ind_pathway['lm2'] = perform_pucker_boltzmann_weighting_gibbs(gibbs_lm2, enth_lm2)
                    ind_pathway['dupe'] = str(pathway_list[dupe_pathway ]+ 1)

        pathway_dict[dupe_pathway] = ind_pathway

    for large_dict_keys in dict.keys():
        ind_pathway = {}
        if large_dict_keys.split('$')[0] not in pathway_multiple:
            files = dict[large_dict_keys]['files'].split('#')
            ind_pathway['lm1'] = round(method_dict[files[0]][ENTH],4)
            ind_pathway['ts']  = round(method_dict[files[1]][ENTH],4)
            ind_pathway['lm2'] = round(method_dict[files[2]][ENTH],4)
            ind_pathway['dupe'] = str('0')
            pathway_dict[large_dict_keys.split('$')[0]] = ind_pathway

    return method[0], pathway_dict


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
            boltz_weight = math.exp(-energy/ (DEFAULT_TEMPERATURE * K_B))
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

    weighted_value = round(sum(np.array(puckering_weight) * np.array(enth_list)),4)

    return weighted_value

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
    parser.add_argument('-f, "--single_file', help="A single pathway file generated from igor_meractor.py",
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
        print('hi')


if __name__ == '__main__':
    status = main()
    sys.exit(status)
