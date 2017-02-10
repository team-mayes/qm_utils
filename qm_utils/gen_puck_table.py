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

# Field Headers #
FUNCTIONAL = 'Functional'
MISSING_FUNCTIONAL = 'N/A'
PUCKER = 'Pucker'
GIBBS = 'G298 (Hartrees)'
JOB_TYPE = 'Pucker Status'
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
    elif job_type == 'lmirc':
        job_type = '-lmirc'
    else:
        print('Job Type is not in the system!!!!!!')

    if hartree_dict[0][FUNCTIONAL] == MISSING_FUNCTIONAL:
        qm_method = split_info[2].split('.')[0] + job_type
        print('Collected level of theory from filename: {} level matches {}?'
              .format(qm_method, base_filename))
    else:
        qm_method = hartree_dict[0][FUNCTIONAL] + job_type

    return hartree_headers, hartree_dict, job_type, qm_method


def create_pucker_gibbs_dict(dict_input, qm_method):
    puckering_dict = {}
    # puckering_dict[JOB_TYPE] = job_type
    for row in dict_input:
        pucker = row[PUCKER]
        gibbs = float(row[GIBBS]) * HARTREE_TO_KCALMOL
        puckering_dict[pucker] = gibbs

    # TODO: need to have a Boltzmann function for when multiple local minimum structures are present

    return puckering_dict, qm_method


def rel_energy_values(pucker_dict1, pucker_dict2):
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
        # print(pucker, lowest_energy_puckering[pucker])

    for pucker in pucker_dict2:
        lowest_energy_puckering_2[pucker] = round((pucker_dict2[pucker] - lowest_energy_value), 1)
        # print(pucker, lowest_energy_puckering[pucker])

    return lowest_energy_puckering_1, lowest_energy_puckering_2


def creating_level_dict_of_dict(energy_dict, qm_method):
    """ creates a dict of dict based on the level of theory-type of job that is currently being processed

    :param energy_dict:
    :param qm_method:
    :return:
    """
    level_of_theory_dict = {}
    level_of_theory_dict[qm_method] = energy_dict

    return level_of_theory_dict


def creating_lowest_energy_dict_of_dict(level_of_theory_dict):
    """ takes the level of theory dict and IDs the lowest energy structures to modify the dictionary so that everything
        is compared to the lowest energy structure.

    :param level_of_theory_dict: dict of dicts containing all of the necessary information on the lm and TS structure
    :return: level_theory_lowest_energy_dict: dict of dicts that has been modified to be comapred to the lowest energy
                structure for each method.
    """

    finished_keys = []
    level_theory_lowest_energy_dict = {}

    for level_keys_1 in level_of_theory_dict.keys():
        level_keys_1_info = level_keys_1.split("-")
        if level_keys_1_info[0] not in finished_keys:
            finished_keys.append(level_keys_1_info[0])
            for level_keys_2 in level_of_theory_dict.keys():
                level_keys_2_info = level_keys_2.split("-")
                if level_keys_1_info[0] == level_keys_2_info[0] and level_keys_1_info[1] != level_keys_2_info[1]:
                    lowest_energy_pucker_dict_1, lowest_energy_pucker_dict_2 = \
                        rel_energy_values(level_of_theory_dict[level_keys_1], level_of_theory_dict[level_keys_2])

                    level_theory_lowest_energy_dict[level_keys_1] = lowest_energy_pucker_dict_1
                    level_theory_lowest_energy_dict[level_keys_2] = lowest_energy_pucker_dict_2

    return level_theory_lowest_energy_dict

def check_same_puckers_lmirc_and_lm(dict_1, job_type1, dict_2, job_type2):
    ''''''

    dict_1_puckers = []
    dict_2_puckers = []

    if job_type1.split("-")[0] == job_type2.split("-")[0] and job_type1.split("-")[1] != job_type2.split("-")[1]:
        for dict_row1 in dict_1:
            dict_1_puckers.append(dict_row1[PUCKER])
        for dict_row2 in dict_2:
            dict_2_puckers.append(dict_row2[PUCKER])


        print(dict_1_puckers)
        print(dict_2_puckers)

        intersecting_puckers = set(dict_2_puckers).intersection(dict_1_puckers)

        print(intersecting_puckers)

        for intersect_puck in intersecting_puckers:
            if intersect_puck == dict_1_puckers and intersect_puck == dict_2_puckers:
                print ('mama we made it')



















    return


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
    """ takes the dict of dict that contains both information on the local min and transition states structure
        and separates them.

    :param level_theory_dict: dict of dicts containing both the local min and transition state information
    :return: lm_table_dict: dict of dicts just for the local min
    :return: ts_table_dict: dict of dicts just for the transition state information
    """

    lm_table_dict = {}
    ts_table_dict = {}
    lmirc_table_dict = {}

    for method_keys in level_theory_dict.keys():
        lm_individual_dict = {}
        ts_individual_dict = {}
        lmirc_individual_dict = {}
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

        elif level_keys_info[1] == 'ts':
            for pucker_keys in pucker_data.keys():
                for pucker_list in LIST_PUCKER:
                    if pucker_keys == pucker_list:
                        pucker_energy = str(pucker_data[pucker_keys])
                        ts_individual_dict[pucker_list] = pucker_energy

            for list_pucker_keys in LIST_PUCKER:
                if list_pucker_keys not in ts_individual_dict.keys():
                    ts_individual_dict[list_pucker_keys] = ''

            ts_table_dict[level_keys_info[0]] = ts_individual_dict

        elif level_keys_info[1] == 'lmirc':
            for pucker_keys in pucker_data.keys():
                for pucker_list in LIST_PUCKER:
                    if pucker_keys == pucker_list:
                        pucker_energy = str(pucker_data[pucker_keys])
                        lmirc_individual_dict[pucker_list] = pucker_energy

                for list_pucker_keys in LIST_PUCKER:
                    if list_pucker_keys not in lmirc_individual_dict.keys():
                        lmirc_individual_dict[list_pucker_keys] = ''

                lmirc_table_dict[level_keys_info[0]] = lmirc_individual_dict

    return lm_table_dict, ts_table_dict


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

    worksheet_lm.conditional_format('B2:P39', {'type': 'cell',
                                               'criteria': '>=',
                                               'value': 50,
                                               'format': format_lm})
    worksheet_ts.conditional_format('B2:P39', {'type': 'cell',
                                               'criteria': '>=',
                                               'value': 50,
                                               'format': format_ts})

    return


def writing_csv_files(lm_table_dict, ts_table_dict, molecule, sum_file_location):
    """"""

    prefix_lm = 'a_csv_lm_' + str(molecule)
    prefix_ts = 'a_csv_ts_' + str(molecule)

    path_lm = create_out_fname(sum_file_location, prefix=prefix_lm, remove_prefix='a_list_csv_files', ext='.csv')
    path_ts = create_out_fname(sum_file_location, prefix=prefix_ts, remove_prefix='a_list_csv_files', ext='.csv')

    df_lm = pd.DataFrame(lm_table_dict, index=LIST_PUCKER)
    df_ts = pd.DataFrame(ts_table_dict, index=LIST_PUCKER)

    df_lm.to_csv(path_lm, index=LIST_PUCKER)
    df_ts.to_csv(path_ts, index=LIST_PUCKER)


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

        level_of_theory_dict = {}

        with open(args.sum_file) as f:
            for csv_file_read_newline in f:
                csv_file_read = csv_file_read_newline.strip("\n")
                hartree_headers, hartree_dict, job_type, qm_method = \
                    read_hartree_files(csv_file_read, args.dir_hartree)
                puckering_dict, qm_method = create_pucker_gibbs_dict(hartree_dict, qm_method)
                level_of_theory_dict[qm_method] = puckering_dict

        level_of_theory_dict_final = creating_lowest_energy_dict_of_dict(level_of_theory_dict)
        lm_table_dict, ts_table_dict = creating_puckering_tables(level_of_theory_dict_final)

        prefix = 'a_table_lm-ts_' + str(args.molecule)

        list_f_name = create_out_fname(args.sum_file, prefix=prefix, remove_prefix='a_list_csv_files',
                                       base_dir=args.dir_hartree, ext='.xlsx')

        writing_xlsx_files(lm_table_dict, ts_table_dict, list_f_name)
        writing_csv_files(lm_table_dict, ts_table_dict, args.molecule, args.sum_file)

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
