#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this python script to analyze hartree norm output files for TS structures.
The output txt files from hartree contain meaningful information on the dihedral angles and can be used
to ID meaningful ring puckering TS structures and less significant encyclical TS structures. The output is a list
of meaningful TS structures that should have IRC calculations performed on them.
"""

from __future__ import print_function

import argparse
import itertools
import os
import sys

from qm_common import (GOOD_RET, create_out_fname, list_to_file, warning, IO_ERROR,
                       InvalidDataError, INVALID_DATA, INPUT_ERROR)

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

# Constants #

DEF_RING_ORDER = '8,1,9,13,17,5'
FIRST_NORMAL_MODE = '   1  '
RING_PUCKER_TOL = 25.0

# Field Headers
NORM_FILE_END = '=== Normal mode   2 ==='
REMOVE_BEGINNING_STRING = 'Normal mode summary for file '


def read_puckering_information(filename, norm_dir):
    """ The purpose of this script is to read a hartree normal input file to capture the "Highest DoF percentages by
        dihedral table" for further analysis." The table contains the top 10 highest percentages of all modes and their
        frequency.

    :param filename: the name of the hartree text file that will be analyzed
    :param norm_dir: the directory of the hartree files for analysis
    :return: The name of the Gaussian Log File that hartree analyzed and the table of the Highest DoF percentages by
        dihedral table" for further analysis."
    """
    highest_dihedral_table = []
    norm_file_path = create_out_fname(filename, base_dir=norm_dir, ext='.txt')
    with open(norm_file_path, mode='r') as file_reading:
        log_file_information = file_reading.next().strip('\n').replace(REMOVE_BEGINNING_STRING, '')
        for lines in itertools.islice(file_reading, 20, 25):
            lines = lines.strip('\n')
            if not lines:
                break
            else:
                highest_dihedral_table.append(lines)
    file_reading.close()

    return log_file_information, highest_dihedral_table


def split_ring_index(ring_order):
    """ Takes a list of strings, and converts them into a list of integers that are sorted.

    :param ring_order: list of strings that index the atoms in the rings
    :return: a sorted list of integers for the atoms in the ring
    """
    ring_atom_index = map(int, ring_order.split(','))
    sorted_ring_atom_index = sorted(ring_atom_index)

    return sorted_ring_atom_index


def analyze_first_normal_mode(filename, first_di_normal_mode_info, sorted_ring_atom_index):
    total_di_percent = 0.00
    for line in first_di_normal_mode_info:
        split_info = line.split(":")
        percent_line = float(split_info[1])
        ordered_pair = split_info[0].split(',')
        first_di_atom_in_pair = int(ordered_pair[0].strip('         ('))
        second_di_atom_in_pair = int(ordered_pair[1].strip(')'))
        if first_di_atom_in_pair in sorted_ring_atom_index and second_di_atom_in_pair in sorted_ring_atom_index:
            total_di_percent += percent_line

    return filename, total_di_percent


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="norm_analysis.py parses through the hartree output files based on "
                                                 "running Gaussian's normal analysis. Gaussian is able to identify the "
                                                 "contributions for each imaginary and normal frequencies with respect"
                                                 "bonds, angles, and most important dihedrals. In order to "
                                                 "differentiate "
                                                 "ring pucker and exocyclic group TS, large contributions to the "
                                                 "imaginery frequency must come from dihedral angles with the ring "
                                                 "atoms "
                                                 "at the middle pair.")

    parser.add_argument('-d', "--dir_norm", help="The directory where the hartree norm files can be found.",
                        default=None)
    parser.add_argument('-s', "--sum_file", help="List of the files complete in Hartree norm.",
                        default=None)
    parser.add_argument('-r', "--ring_order", help="List of the atom ids in any order.")
    parser.add_argument('-t', "--tol", help="The percentage tolerance for deciding if the dihedral percentage is"
                                            " sufficient or not.")

    args = None
    try:
        args = parser.parse_args(argv)
        if args.sum_file is None:
            raise InvalidDataError("Input files are required. Missing hartree input or two-file inputs")
        elif not os.path.isfile(args.sum_file):
            raise IOError("Could not find specified hartree summary file: {}".format(args.sum_file))
        # Finally, if the summary file is there, and there is no dir_xyz provided
        if args.dir_norm is None:
            args.dir_norm = os.path.dirname(args.sum_file)
        # if a  dir_xyz is provided, ensure valid
        elif not os.path.isdir(args.dir_norm):
            raise InvalidDataError("Invalid path provided for '{}': ".format('-d, --dir_norm', args.dir_norm))

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

        sorted_ring_order = split_ring_index(args.ring_order)

        ring_pucker_ts_list = []
        exo_pucker_ts_list = []

        with open(args.sum_file) as f:
            list_of_gaussian_norm = f.read().splitlines()
        for hartree_filename in list_of_gaussian_norm:
            out_filename, first_mode_di_info = read_puckering_information(hartree_filename, args.dir_norm)
            out_filename, file_percentage = analyze_first_normal_mode(out_filename, first_mode_di_info,
                                                                      sorted_ring_order)

            if file_percentage > RING_PUCKER_TOL:
                ring_pucker_ts_list.append([out_filename, file_percentage])
            elif file_percentage < RING_PUCKER_TOL:
                exo_pucker_ts_list.append([out_filename, file_percentage])

        filename_ring_ts = create_out_fname(args.sum_file, prefix='z_norm-analysis_TS_ring_puckers_',
                                            base_dir=args.dir_norm, ext='.txt')

        filename_exo_ts = create_out_fname(args.sum_file, prefix='z_norm-analysis_TS_exo_puckers_',
                                           base_dir=args.dir_norm, ext='.txt')

        list_to_file(ring_pucker_ts_list, filename_ring_ts, list_format=None, delimiter=' ', mode='w',
                     print_message=True)

        list_to_file(exo_pucker_ts_list, filename_exo_ts, list_format=None, delimiter=' ', mode='w', print_message=True)

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
