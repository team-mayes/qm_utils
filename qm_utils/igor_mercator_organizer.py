#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this python script is to create csv files that will be uploaded into Igor Pro for data analysis. The
coordinates that are required are phi and theta.
"""

from __future__ import print_function

import argparse
import os

import pandas as pd
import numpy as np
import sys
from qm_utils.qm_common import read_csv_to_dict
from qm_utils.qm_common import (GOOD_RET, list_to_dict, create_out_fname, write_csv, list_to_file, warning, IO_ERROR,
                                InvalidDataError, INVALID_DATA, read_csv_to_dict, get_csv_fieldnames, INPUT_ERROR)

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

# Constants #

# Hartree field headers
FILE_NAME = 'File Name'
PUCKER = 'Pucker'
ENERGY_GIBBS = 'G298 (Hartrees)'
ENERGY_ELECTRONIC = 'Energy (A.U.)'
THETA = 'theta'
PHI = 'phi'

# Functions #


def reading_all_csv_input_files(file_read, molecule):
    ''''''

    dict = {}

    dict = read_csv_to_dict(file_read, mode='r')
    # Reads the correct information about the test
    pathway = os.path.dirname(file_read)
    file_read = file_read.replace(pathway,'')
    filename_split = file_read.split("-")
    len_file = len(filename_split)
    method = filename_split[len_file-1].split(".")

    if method[1] != 'csv':
       print("The input file isn't named correctly.")
    else:
        if filename_split[len_file-2] == str(molecule) and filename_split[len_file-3] == 'lm':
            job_type = filename_split[len_file-3]
        elif filename_split[len_file-2] == str(molecule) and filename_split[len_file-3] == 'TS':
            job_type = filename_split[len_file-3]
        elif filename_split[len_file-2] == str(molecule) and filename_split[len_file-3] == 'lmirc':
            job_type = filename_split[len_file-3]
        elif filename_split[len_file-2] == str(molecule) and filename_split[len_file-3] == 'optall':
            job_type = filename_split[len_file-3]
        else:
            print('Something is wrong...the job type is not found.')

        if filename_split[len_file-4] == 'unsorted':
            sort_status = filename_split[len_file-4]
        elif filename_split[len_file-4] == 'sorted':
            sort_status = filename_split[len_file-4]
        else:
            print('Some is wrong... not sure whether the file is sorted or not')

    return method[0], job_type, sort_status, dict


def creating_dict_of_dict(list_files, molecule):
    ''''''

    dict_of_dicts = {}
    for file in list_files:
        if file is not None:
            method, job_type, sort_status, dict = reading_all_csv_input_files(file, molecule)
            dict_id = method + '-' + job_type + '-' + sort_status
            dict_of_dicts[dict_id] = dict

    return dict_of_dicts, method


def sorting_dict_of_dict(dict_of_dicts):

    data_dict = {}

    for key in dict_of_dicts.keys():
        # creates the headers for the Igor Overwrite
        info = key.split('-')
        overwrite_phi = 'ov' + info[1] + '-' + info[2] + 'PHI'
        overwrite_theta = 'ov' + info[1] + '-' + info[2] + 'THETA'

        # creates the headers for level of theory in Igor
        phi_data = []
        theta_data =[]
        phi = key + '-' + PHI
        theta = key + '-' + THETA

        for rows in dict_of_dicts[key]:
            phi_data.append(rows[PHI])
            theta_data.append(rows[THETA])

        data_dict[phi] = phi_data
        data_dict[theta] = theta_data
        data_dict[overwrite_phi] = phi_data
        data_dict[overwrite_theta] = theta_data

    return data_dict


def write_file_data_dict(data_dict, out_filename):

    max_length = 0

    for job in data_dict.keys():
        job_length = len(data_dict[job])
        if job_length > max_length:
            max_length = job_length

    for job_1 in data_dict.keys():
        if len(data_dict[job_1]) != max_length:
            fill_length = max_length - len(data_dict[job_1])
            for num in range(0, fill_length):
                data_dict[job_1].append(np.nan)

    df_data_dict = pd.DataFrame(data_dict)

    df_data_dict.to_csv(out_filename)

    return


def creating_igor_pathway(dict_of_dicts):
    ''''''

    ts_dict = {}
    irc_dict = {}
    pathway_phi = []
    pathway_theta = []
    pathway_dict = {}

    for job_type_keys in dict_of_dicts.keys():
        type_split = job_type_keys.split('-')
        qm_method = type_split[0]
        if type_split[1] == 'TS' and type_split[2] == 'sorted':
            ts_dict = dict_of_dicts[job_type_keys]
        elif type_split[1] == 'lmirc' and type_split[2] == 'unsorted':
            irc_dict = dict_of_dicts[job_type_keys]
        else:
            pass

    for row_ts in ts_dict:
        ircf_file = 'missing'
        ircr_file = 'missing'
        ts_file_name = row_ts[FILE_NAME].split('-')

        for row_irc in irc_dict:
            irc_file_name = row_irc[FILE_NAME]
            if ts_file_name[0] in irc_file_name and ts_file_name[1] in irc_file_name:
                if 'ircf' in irc_file_name:
                   ircf_file = irc_file_name
                   ircf_theta = row_irc[THETA]
                   ircf_phi = row_irc[PHI]
                elif 'ircr' in irc_file_name:
                    ircr_file = irc_file_name
                    ircr_theta = row_irc[THETA]
                    ircr_phi = row_irc[PHI]



        if abs(float(ircf_phi) - float(row_ts[PHI])) > 340:
            pathway_phi.append(ircr_phi)
            pathway_theta.append('')

            pathway_phi.append(ircr_phi)
            pathway_theta.append(ircr_theta)

            pathway_phi.append(row_ts[PHI])
            pathway_theta.append(row_ts[THETA])

            pathway_phi.append(ircf_phi)
            pathway_theta.append('')

            pathway_phi.append(ircf_phi)
            pathway_theta.append(ircf_theta)

            pathway_phi.append(0)
            pathway_theta.append(0.5*(float(row_ts[THETA]) + float(ircf_theta)))

            pathway_phi.append(ircf_phi)
            pathway_theta.append('')

            pathway_phi.append(ircf_phi)
            pathway_theta.append(ircf_theta)


        elif abs(float(ircr_phi) - float(row_ts[PHI])) > 340:
            pathway_phi.append(ircr_phi)
            pathway_theta.append('')

            pathway_phi.append(ircr_phi)
            pathway_theta.append(ircr_theta)

            pathway_phi.append(0)
            pathway_theta.append(row_ts[THETA])

            pathway_phi.append(ircf_phi)
            pathway_theta.append('')

            pathway_phi.append(ircf_phi)
            pathway_theta.append(ircf_theta)

            pathway_phi.append(row_ts[PHI])
            pathway_theta.append(row_ts[THETA])
        else:
            pathway_phi.append(ircr_phi)
            pathway_theta.append('')
            pathway_phi.append(ircr_phi)
            pathway_theta.append(ircr_theta)
            pathway_phi.append(row_ts[PHI])
            pathway_theta.append(row_ts[THETA])
            pathway_phi.append(ircf_phi)
            pathway_theta.append('')
            pathway_phi.append(ircf_phi)
            pathway_theta.append(ircf_theta)
            pathway_phi.append(row_ts[PHI])
            pathway_theta.append(row_ts[THETA])

    pathway_dict[str(qm_method) +'path_phi'] = pathway_phi
    pathway_dict[str(qm_method) +'path_theta'] = pathway_theta

    return pathway_dict



# Command Line Parser #

def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="This script creates simple CSV files to be easily loaded into Igor "
                                                 "Pro. The input are Hartree Files for local minimas (lm and lmirc) "
                                                 "and the transition state structures.")

    parser.add_argument('-ts', "--ts_file", help="The hartree output for the transition states.",
                        default=None)
    parser.add_argument('-lm', "--lm_file", help="The hartree output for the combined low energy lm and lmirc "
                                                "structures.",
                        default=None)
    parser.add_argument('-raw_lm', "--raw_lm", help="The hartree output for just the lm structures (not sorted via"
                                                    " xyz_cluster.",
                        default=None)
    parser.add_argument('-raw_lmirc', "--raw_lmirc", help="The hartree output for just the lmirc structures.",
                        default=None)
    parser.add_argument('-raw_ts', "--raw_ts", help="The raw hartree output for TS structures.",
                        default=None)
    parser.add_argument('-d', "--dir", help="The directory where all hartree files are located.,",
                        default=None)
    parser.add_argument('-m', "--mole", help="The molecule currently being studied.",
                        default=None)


    args = None
    args = parser.parse_args(argv)

    if args.dir is None:
        GOOD_RET = 'Okay'
        if args.ts_file is not None:
            args.dir = os.path.dirname(args.ts_file)
        elif args.lm_file is not None:
            args.dir = os.path.dirname(args.lm_file)
        elif args.raw_lm is not None:
            args.dir = os.path.dirname(args.raw_lm)
        elif args.raw_ts is not None:
            args.dir = os.path.dirname(args.raw_ts)
        elif args.raw_lmirc is not None:
            args.dir = os.path.dirname(args.raw_lmirc)
        else:
            parser.print_help()
            GOOD_RET = 'No'

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

        list_files = [args.ts_file, args.lm_file, args.raw_lm, args.raw_lmirc, args.raw_ts]

        dict_of_dicts, method = creating_dict_of_dict(list_files, str(args.mole))
        data_dict = sorting_dict_of_dict(dict_of_dicts)
        pathway_dict = creating_igor_pathway(dict_of_dicts)

        output_filename = create_out_fname('igor_df_' + str(args.mole) + '_' + str(method), base_dir=args.dir, ext='.csv')
        output_filename_pathway = create_out_fname('igor_pathway_' + str(args.mole) + '_' + str(method), base_dir=args.dir, ext='.csv')
        write_file_data_dict(data_dict,output_filename)
        write_file_data_dict(pathway_dict, output_filename_pathway)


if __name__ == '__main__':
    status = main()
    sys.exit(status)
