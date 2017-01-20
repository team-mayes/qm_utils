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
import itertools
import os
import sys
import csv
import pandas as pd
import numpy as np
from qm_utils.qm_common import (GOOD_RET, create_out_fname, list_to_file, warning, IO_ERROR,
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

# Field Headers #
FUNCTIONAL = 'Functional'
MISSING_FUNCTIONAL = 'N/A'
PUCKER = 'Pucker'
GIBBS = 'G298 (Hartrees)'
JOB_TYPE = 'Pucker Status'
LIST_PUCKER = [ '4c1',
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

# Functions #


def read_hartree_files(filename, hartree_dir):

    hartree_file_path = create_out_fname(filename, base_dir=hartree_dir, ext='.csv')
    hartree_dict = read_csv_to_dict(hartree_file_path, mode='rU')
    hartree_headers = get_csv_fieldnames(hartree_file_path, mode='rU')
    base_filename = os.path.split(filename)[1]
    split_info = base_filename[17:].split('-')
    job_type = split_info[0]

    if job_type == 'optall':
        job_type = '-lm'
    elif job_type == 'TS':
        job_type = '-ts'
    else:
        print('Job Type is not in the system!!!!!!')

    if hartree_dict[0][FUNCTIONAL] == MISSING_FUNCTIONAL:
        qm_method = split_info[2].split('.')[0] + job_type
        print('Collected level of theory from filename: {} level matches {}?'
                                .format(qm_method,base_filename))
    else:
        qm_method = hartree_dict[0][FUNCTIONAL] + job_type

    return hartree_headers, hartree_dict, job_type ,qm_method

def create_pucker_gibbs_dict(dict, job_type, qm_method):

    puckering_dict = {}
    #puckering_dict[JOB_TYPE] = job_type
    for row in dict:
        pucker = row[PUCKER]
        gibbs = float(row[GIBBS])*HARTREE_TO_KCALMOL
        puckering_dict[pucker] = gibbs

    #TODO: need to have a Boltzmann function for when multiple local minimum structures are present

    #TODO: brainstorm a way to organize the information so that it is meaninful.... need to have all structures ID as local min or TS structures...

    puckering_dict

    return puckering_dict, qm_method


def rel_energy_values(pucker_dict1, method_1, pucker_dict2, method_2):

    lowest_energy_value = 1000000
    lowest_energy_puck = []
    lowest_energy_puckering_1 = {}
    lowest_energy_puckering_2 = {}

    for check_puck in pucker_dict1:
        if pucker_dict1[check_puck] < lowest_energy_value:
            lowest_energy_value = pucker_dict1[check_puck]
            lowest_energy_puck = check_puck

    for check_puck in pucker_dict2:
        if pucker_dict2[check_puck] < lowest_energy_value:
            lowest_energy_value = pucker_dict2[check_puck]
            lowest_energy_puck = check_puck

    print('The lowest energy pucker was {} ({})'.format(lowest_energy_puck, lowest_energy_value))

    for pucker in pucker_dict1:
        lowest_energy_puckering_1[pucker] = round((pucker_dict1[pucker] - lowest_energy_value), 1)
        #print(pucker, lowest_energy_puckering[pucker])

    for pucker in pucker_dict2:
        lowest_energy_puckering_2[pucker] = round((pucker_dict2[pucker] - lowest_energy_value), 1)
        #print(pucker, lowest_energy_puckering[pucker])

        #level_of_theory_dict = creating_level_dict_of_dict(lowest_energy_puckering_1, method_1)
        #level_of_theory_dict = creating_level_dict_of_dict(lowest_energy_puckering_2, method_2)

        #print(level_of_theory_dict)
    return lowest_energy_puckering_1, lowest_energy_puckering_2


def creating_level_dict_of_dict(energy_dict, qm_method):

    level_of_theory_dict = {}
    level_of_theory_dict[qm_method] = energy_dict

    return level_of_theory_dict

def creating_lowest_energy_dict_of_dict(level_of_theory_dict):

    finished_keys = []
    level_theory_lowest_energy_dict = {}

    for level_keys_1 in level_of_theory_dict.keys():
        level_keys_1_info = level_keys_1.split("-")
        if level_keys_1_info[0] not in finished_keys:
            finished_keys.append(level_keys_1_info[0])
            for level_keys_2 in level_of_theory_dict.keys():
                level_keys_2_info = level_keys_2.split("-")
                if (level_keys_1_info[0] == level_keys_2_info[0] and level_keys_1_info[1] != level_keys_2_info[1]):
                    lowest_energy_pucker_dict_1, lowest_energy_pucker_dict_2 = \
                                          rel_energy_values(level_of_theory_dict[level_keys_1], level_keys_1,
                                                            level_of_theory_dict[level_keys_2], level_keys_2)

                    level_theory_lowest_energy_dict[level_keys_1] = lowest_energy_pucker_dict_1
                    level_theory_lowest_energy_dict[level_keys_2] = lowest_energy_pucker_dict_2

    return level_theory_lowest_energy_dict

def find_files_by_dir(tgt_dir, pat):
    """Recursively searches the target directory tree for files matching the given pattern.
    The results are returned as a dict with a list of found files keyed by the absolute
    directory name.
    @param tgt_dir: The target base directory.
    @param pat: The file pattern to search for.
    @return: A dict where absolute directory names are keys for lists of found file names
        that match the given pattern.
    """
    match_dirs = {}
    for root, dirs, files in os.walk(tgt_dir):
        matches = [match for match in files if fnmatch.fnmatch(match, pat)]
        if matches:
            match_dirs[os.path.abspath(root)] = matches
    return match_dirs


def creating_puckering_tables(level_theory_dict):
    ''''''

    lm_table_dict = {}
    ts_table_dict = {}

    for method_keys in level_theory_dict.keys():
        lm_individual_dict = {}
        ts_individual_dict = {}
        level_keys_info = method_keys.split("-")
        pucker_data = level_theory_dict[method_keys]
        if level_keys_info[1] == 'lm':
            for pucker_keys in pucker_data.keys():
                for pucker_list in LIST_PUCKER:
                    if pucker_keys == pucker_list:
                        pucker_energy = str(pucker_data[pucker_keys])
                        lm_individual_dict[pucker_list] = pucker_energy

            for list_pucker_keys in LIST_PUCKER:
                if list_pucker_keys not in lm_individual_dict.keys():
                    lm_individual_dict[list_pucker_keys] = ''

            lm_table_dict[level_keys_info[0]] = lm_individual_dict

        elif level_keys_info[1] =='ts':
            for pucker_keys in pucker_data.keys():
                for pucker_list in LIST_PUCKER:
                    if pucker_keys == pucker_list:
                        pucker_energy = str(pucker_data[pucker_keys])
                        ts_individual_dict[pucker_list] = pucker_energy

            for list_pucker_keys in LIST_PUCKER:
                if list_pucker_keys not in ts_individual_dict.keys():
                    ts_individual_dict[list_pucker_keys] = ''

            ts_table_dict[level_keys_info[0]] = ts_individual_dict

    return lm_table_dict, ts_table_dict


def writing_xlsx_files(lm_table_dict, ts_table_dict, output_filename):
    ''''''

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
    #worksheet_testing = writer.sheets['testing']

    worksheet_lm.conditional_format('B2:P39', {'type': 'cell',
                                               'criteria': '>=',
                                               'value': 50,
                                               'format': format_lm})
    worksheet_ts.conditional_format('B2:P39', {'type': 'cell',
                                               'criteria': '>=',
                                               'value': 50,
                                               'format': format_ts})

    return

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

        level_of_theory_dict = {}

        with open(args.sum_file) as f:
            for csv_file_read_newline in f:
                csv_file_read = csv_file_read_newline.strip("\n")
                hartree_headers, hartree_dict, job_type, qm_method =\
                    read_hartree_files(csv_file_read, args.dir_hartree)
                puckering_dict, qm_method = create_pucker_gibbs_dict(hartree_dict, job_type, qm_method)
                level_of_theory_dict[qm_method] = puckering_dict

        level_of_theory_dict_final = creating_lowest_energy_dict_of_dict(level_of_theory_dict)
        lm_table_dict, ts_table_dict = creating_puckering_tables(level_of_theory_dict_final)

        list_f_name = create_out_fname(args.sum_file, prefix='z_oxane_table', base_dir=args.dir_hartree, ext='.xlsx')

        writing_xlsx_files(lm_table_dict, ts_table_dict, list_f_name)



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
