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


# Functions #


def read_hartree_files(filename, hartree_dir):

    hartree_file_path = create_out_fname(filename, base_dir=hartree_dir, ext='.csv')
    hartree_dict = read_csv_to_dict(hartree_file_path, mode='rU')
    hartree_headers = get_csv_fieldnames(hartree_file_path, mode='rU')
    base_filename = os.path.split(filename)[1]
    split_info = base_filename[17:].split('-')
    job_type = split_info[0]

    if job_type == 'optall':
        job_type = 'Local Min'

    if hartree_dict[0][FUNCTIONAL] == MISSING_FUNCTIONAL:
        qm_method = split_info[2].split('.')[0]
        print('Collected level of theory from filename: {} level matches {}?'
                                .format(qm_method,base_filename))
    else:
        qm_method = hartree_dict[0][FUNCTIONAL]

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


def rel_energy_values(pucker_dict):

    lowest_energy_value = 0
    lowest_energy_puck = []
    lowest_energy_puckering = {}

    for check_puck in pucker_dict:
        if pucker_dict[check_puck] < lowest_energy_value:
            lowest_energy_value = pucker_dict[check_puck]
            lowest_energy_puck = check_puck

    print('The lowest energy pucker was {} ({})'.format(lowest_energy_puck, lowest_energy_value))

    for pucker in pucker_dict:
        lowest_energy_puckering[pucker] = round((pucker_dict[pucker] - lowest_energy_value),1)
#        print(pucker, lowest_energy_puckering[pucker])

    return lowest_energy_puckering


def creating_level_dict_of_dict(lowest_energy_dict, qm_method):

    level_of_theory_dict = {}
    level_of_theory_dict[qm_method] = lowest_energy_dict

    return level_of_theory_dict


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

        with open(args.sum_file) as f:

            for csv_file_read in f:
                hartree_headers, hartree_dict, qm_method= read_hartree_files(csv_file_read, args.dir_hartree)
                puckering_dict, qm_method = create_pucker_gibbs_dict(hartree_headers, hartree_dict, qm_method)
                level_of_theory_dict = creating_level_dict_of_dict(puckering_dict, qm_method)

        print(level_of_theory_dict)


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
