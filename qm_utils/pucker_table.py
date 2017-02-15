#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this srcipt is to take hartree output files and generate a new CSV file with the desired information.
The hartree output files contain information such as file name, pucker, energy, etc. The script will generate
information based on pucker and level of theory.
"""

from __future__ import print_function

import argparse
import fnmatch
import os
import sys
import pandas as pd
import math

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
FUNCTIONAL = 'Functional'
MISSING_FUNCTIONAL = 'N/A'
PUCKER = 'Pucker'
GIBBS = 'G298 (Hartrees)'
ENTH = "H298 (Hartrees)"
JOB_TYPE = 'Pucker Status'
PUCKERING_NEW = 'Boltz Pucker Num'
FILE_NAME = 'File Name'
FREQ = "Freq 1"
WEIGHT_GIBBS = 'Boltz Weight Gibbs'
WEIGHT_ENTH = 'Boltz Weight Enth'
LIST_PUCKER = ['4c1',
               '14b',
               '25b',
               'o3b',
               '1h2',
               '2h3',
               '3h4',
               '4h5',
               '5ho',
               'oh1',
               '1s3',
               '1s5',
               '2so',
               '1e',
               '2e',
               '3e',
               '4e',
               '5e',
               'oe',
               '1c4',
               'b14',
               'b25',
               'bo3',
               '2h1',
               '3h2',
               '4h3',
               '5h4',
               'oh5',
               '1ho',
               '3s1',
               '5s1',
               'os2',
               'e1',
               'e2',
               'e3',
               'e4',
               'e5',
               'eo']


## Functions ##

def read_hartree_files_lowest_energy(filename, hartree_dir):
    ''' Loads the hartree files containing all of the methods for a given level of theory and finds the lowest energy
        structure for the given QM method.

    :param filename: the filename for the CSV that contains all of the information
    :param hartree_dir: the dictory that the files are located in
    :return: returns the hartree headers, the lowest energy dict, and the type of method used to solve it.
    '''
    hartree_file_path = create_out_fname(filename, base_dir=hartree_dir, ext='.csv')
    hartree_dict = read_csv_to_dict(hartree_file_path, mode='rU')
    hartree_headers = get_csv_fieldnames(hartree_file_path, mode='rU')
    base_filename = os.path.split(filename)[1]
    split_info = base_filename.split('-')
    qm_method = split_info[len(split_info)-1].split('.')[0]

    lowest_energy_enth_val = 1000000
    lowest_energy_gibbs_val = 1000000
    lowest_energy_puck_enth = []
    lowest_energy_puck_gibbs = []
    lowest_energy_dict = []

    for row in hartree_dict:
        energy_row_enth = row[ENTH]
        energy_row_gibbs = row[GIBBS]
        if float(energy_row_enth) < float(lowest_energy_enth_val):
            lowest_energy_enth_val = energy_row_enth
            lowest_energy_puck_enth = row[PUCKER]
        if float(energy_row_gibbs) < float(lowest_energy_gibbs_val):
            lowest_energy_gibbs_val = energy_row_gibbs
            lowest_energy_puck_gibbs = row[PUCKER]

    if lowest_energy_puck_gibbs == lowest_energy_puck_enth:
        print('The lowest energy pucker was : {}'.format(lowest_energy_puck_enth))

    for row in hartree_dict:
        row[ENTH] = round(float(row[ENTH]) - float(lowest_energy_enth_val), 5) * HARTREE_TO_KCALMOL
        row[GIBBS] = round(float(row[GIBBS]) - float(lowest_energy_gibbs_val), 5) * HARTREE_TO_KCALMOL
        lowest_energy_dict.append(row)

    return hartree_headers, lowest_energy_dict, qm_method


def sorting_job_types(lowest_energy_duct,qm_method):
    ''' Based on the frequency, the combined hartree file sorts the data into lm and TS jobs for future processing

    :param lowest_energy_duct: the lowest energy dict for the given qm_method
    :param qm_method: the qm_method used to generate the structures in the hartree file
    :return: list of dicts for the lm_jobs and the ts_jobs along with the qm_method
    '''

    lm_jobs = []
    ts_jobs = []

    for row in lowest_energy_duct:
        row_freq = row[FREQ]
        if float(row_freq) < 0:
            ts_jobs.append(row)
        elif float(row_freq) > 0:
            lm_jobs.append(row)
        else:
            print('Something is wrong...why is the 1st frequency not positive (lm) or negative (ts)?')

    return lm_jobs, ts_jobs, qm_method

def boltzmann_weighting(low_energy_job_dict, qm_method):

    list_puckers = []
    isolation_dict = {}
    dict_of_dict = {}
    pucker_total_weight_gibbs = {}
    contribution_dict = {}


    for row in low_energy_job_dict:
        row_pucker = row[PUCKER]
        row_filename = row[FILE_NAME]
        dict_of_dict[row_filename] = row
        if row_pucker in list_puckers:
            isolation_dict[row_pucker].append(row_filename)
        elif row_pucker not in list_puckers:
            list_puckers.append(row_pucker)
            isolation_dict[row_pucker] = []
            isolation_dict[row_pucker].append(row_filename)
            pucker_total_weight_gibbs[row_pucker] = float(0)
            contribution_dict[row_pucker] = float(0)


    for pucker_key in isolation_dict.keys():
        for pucker_file in isolation_dict[pucker_key]:
            for main_file in low_energy_job_dict:
                if pucker_file == main_file[FILE_NAME]:
                    pucker_type = main_file[PUCKER]
                    try:
                        gibbs_energy = float(main_file[GIBBS])
                        enth_energy = float(main_file[ENTH])
                        weight_gibbs = math.exp(-gibbs_energy/(DEFAULT_TEMPERATURE*K_B))
                        weight_enth = math.exp(-enth_energy/(DEFAULT_TEMPERATURE*K_B))
                        main_file[WEIGHT_GIBBS] = weight_gibbs
                        main_file[WEIGHT_ENTH] = weight_enth
                    finally:
                        pucker_total_weight_gibbs[pucker_type] = pucker_total_weight_gibbs[pucker_type] + weight_gibbs

    for pucker_key in isolation_dict.keys():
        total_weight = pucker_total_weight_gibbs[pucker_key]
        for main_file in low_energy_job_dict:
            if main_file[PUCKER] == pucker_key:
                contribution_dict[pucker_key] = \
                    round(contribution_dict[pucker_key] + (main_file[WEIGHT_GIBBS]/total_weight)*main_file[GIBBS],2)


    return contribution_dict, qm_method


def writing_xlsx_files(lm_table_dict, ts_table_dict, output_filename):
    """ utilizes panda dataframes to write the local min and transition state dict of dicts

    :param lm_table_dict: dictionary corresponding to the local mins
    :param ts_table_dict: dictional corresponding to the transition state structures
    :param output_filename: output filename for the excel file
    :return: excel file with the required information
    """

    df_lm = pd.DataFrame(lm_table_dict, index=LIST_PUCKER)
    df_ts = pd.DataFrame(ts_table_dict, index=LIST_PUCKER)
    writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')
    df_lm.to_excel(writer, sheet_name='local min')
    df_ts.to_excel(writer, sheet_name='transition state')

    workbook = writer.book

    format_lm = workbook.add_format({'font_color': '#008000'})
    format_ts = workbook.add_format({'font_color': '#4F81BD'})

    worksheet_lm = writer.sheets['local min']
    worksheet_ts = writer.sheets['transition state']


    # TODO: figure out why my conditional formatting is no longer working
    worksheet_lm.conditional_format('B2:P39', {'type': 'cell',
                                               'criteria': '>=',
                                               'value': 50,
                                               'format': format_lm})
    worksheet_ts.conditional_format('B2:P39', {'type': 'cell',
                                               'criteria': '>=',
                                               'value': 50,
                                               'format': format_ts})

    writer.save
    writer.close()

    return

def writing_csv_files(lm_table_dict, ts_table_dict, molecule, sum_file_location):
    """

    :param lm_table_dict:  dict of dicts containing the local min information
    :param ts_table_dict: dict of dicts containing the TS information
    :param molecule: the molecule currently being studied
    :param sum_file_location: the loation of the file
    :return: two csv files (one for the local min and one for the TS) containing the boltzmann weighted energies
    """

    prefix_lm = 'a_csv_lm_' + str(molecule)
    prefix_ts = 'a_csv_ts_' + str(molecule)

    path_lm = create_out_fname(sum_file_location, prefix=prefix_lm, remove_prefix='a_list_csv_files', ext='.csv')
    path_ts = create_out_fname(sum_file_location, prefix=prefix_ts, remove_prefix='a_list_csv_files', ext='.csv')

    df_lm = pd.DataFrame(lm_table_dict, index=LIST_PUCKER)
    df_ts = pd.DataFrame(ts_table_dict, index=LIST_PUCKER)

    df_lm.to_csv(path_lm, index=LIST_PUCKER)
    df_ts.to_csv(path_ts, index=LIST_PUCKER)


 ## Command Line Parse ##


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="The gen_puck_table.py script is designed to combine hartree output "
                                                 "files to compare different properties across different levels of "
                                                 "theory. The hartree input files for a variety of levels of theory "
                                                 "are combined to produce a new data table.")

    parser.add_argument('-s', "--sum_file", help="List of csv files to read.", default=None)
    parser.add_argument('-d', "--dir_hartree", help="The directory where the hartree files can be found.",
                        default=None)
    parser.add_argument('-p', "--pattern", help="The file pattern you are looking for (example: '.csv').",
                        default=None)
    parser.add_argument('-m', "--molecule", help="The type of molecule that is currently being studied")

    args = None
    try:
        args = parser.parse_args(argv)
        if args.sum_file is None:
            raise InvalidDataError("Input files are required. Missing hartree input or two-file inputs")
        elif not os.path.isfile(args.sum_file):
            raise IOError("Could not find specified hartree summary file: {}".format(args.sum_file))
        # Finally, if the summary file is there, and there is no dir_xyz provided
        if args.dir_hartree is None:
            args.dir_hartree = os.path.dirname(args.sum_file)
        # if a  dir_xyz is provided, ensure valid
        elif not os.path.isdir(args.dir_hartree):
            raise InvalidDataError("Invalid path provided for '{}': ".format('-d, --dir_hartree', args.dir_hartree))

    except (KeyError, InvalidDataError) as e:
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    except IOError as e:
        warning(e)
        parser.print_help()
        return args, IO_ERROR
    except (ValueError, SystemExit) as e:
        if e.message == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR

    return args, GOOD_RET


def main(argv=None):
    """
    Runs the main program
    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret
    try:

        lm_level_dict = {}
        ts_level_dict = {}
        overall_level_dict = {}

        with open(args.sum_file) as f:
            for csv_file_read_newline in f:
                csv_file_read = csv_file_read_newline.strip("\n")
                hartree_headers, lowest_energy_dict, qm_method = read_hartree_files_lowest_energy( csv_file_read,
                                                                                                   args.dir_hartree)
                lm_jobs, ts_jobs, qm_method = sorting_job_types(lowest_energy_dict, qm_method)
                contribution_dict_lm, qm_method = boltzmann_weighting(lm_jobs, qm_method)
                contribution_dict_ts, qm_method = boltzmann_weighting(ts_jobs, qm_method)

                lm_level_dict[qm_method + "-lm"] = contribution_dict_lm
                ts_level_dict[qm_method + "-ts"] = contribution_dict_ts
                overall_level_dict[qm_method + "-ts"] = contribution_dict_ts
                overall_level_dict[qm_method + "-lm"] = contribution_dict_lm


        prefix = 'a_table_lm-ts_' + str(args.molecule)

        list_f_name = create_out_fname(args.sum_file, prefix=prefix, remove_prefix='a_list_csv_files',
                                               base_dir=os.path.dirname(args.sum_file), ext='.xlsx')



        writing_csv_files(lm_level_dict, ts_level_dict, args.molecule, args.sum_file)
        writing_xlsx_files(lm_level_dict, ts_level_dict, list_f_name)

    except IOError as e:
        warning(e)
        return IO_ERROR
    except (InvalidDataError, KeyError) as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
