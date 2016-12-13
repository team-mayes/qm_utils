#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this python script to analyze hartree norm output files for TS structures.
The output txt files from hartree contain meaningful information on the dihedral angles and can be used
to ID meaningful ring puckering TS structures and less significant exocyclic TS strctures. The output is a list
of meaningful TS structures that should have IRC calculations performed on them.
"""

from __future__ import print_function

import argparse
import os
import sys
from shutil import copyfile

import itertools
import numpy as np
from qm_common import (GOOD_RET, list_to_dict, create_out_fname, write_csv, list_to_file, warning, IO_ERROR,
                       InvalidDataError, INVALID_DATA, read_csv_to_dict, get_csv_fieldnames, INPUT_ERROR)

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
        log_file_information = file_reading.next().strip('\n').replace(REMOVE_BEGINNING_STRING,'')
        for lines in itertools.islice(file_reading,4,14):
            lines = lines.strip('\n')
            highest_dihedral_table.append(lines)
    file_reading.close()

    return log_file_information, highest_dihedral_table

def create_dihedral(log_file_information, highest_dihedral_table):
    """ Parses the dihedral table to capture only the normal mode 1 information. Mode 1 corresponds to the first normal
        mode (which for TS structures is negative).

    :param log_file_information: the name of the log file that the orginal normal mode analysis was performed
    :param highest_dihedral_table: the highest DoF dihedral table
    :return: The name of the Gaussian Log File and the information corresponding to the first normal mode.
    """
    ts_first_modes_dihedral_info = []
    for table_line in highest_dihedral_table:
        table_split = table_line.split("|")
        pair = table_split[0]
        mode = int(table_split[1])
        percent = table_split[2]
        freq = float(table_split[3])
        if mode == 1:
            if freq > 0:
                print("The 1st normal mode has a frequency greater than 0! "
                      "Please files to make sure that these structures are TS.")
            elif freq < 0:
                ts_first_modes_dihedral_info.append([pair, mode, percent, freq])
    return log_file_information, ts_first_modes_dihedral_info

def split_ring_index(ring_order):
    """ Takes a list of strings, and converts them into a list of integers that are sorted.

    :param ring_order: list of strings that index the atoms in the rings
    :return: a sorted list of integers for the atoms in the ring
    """
    ring_atom_index = map(int, ring_order.split(','))
    sorted_ring_atom_index = sorted(ring_atom_index)

    return sorted_ring_atom_index

def identifying_ring_pucker_di(filename, first_mode_di_info, sorted_ring_atom_index):

    status_first_normal = []

    for line in first_mode_di_info:
        pair_1 = line[0].strip("(")
        pair_2 = pair_1.strip(") ")
        first_di_atom = map(int, pair_2.split(','))[0]
        second_di_atom = map(int, pair_2.split(','))[1]

        if first_di_atom in sorted_ring_atom_index:
            if second_di_atom in sorted_ring_atom_index:
                status_first_normal.append([line[0], 'Both'])
            else:
                print("For {}, the second atom ({}) is not located in the ring."
                      .format(filename, second_di_atom))
                status_first_normal.append([line[0], 'First'])
        else:
            print("For {}, the first atom {} is not located in the ring.".
                  format(filename, first_di_atom))
            status_first_normal.append([line[0], 'Neither'])

    return filename, status_first_normal

def id_key_structures(filename, mode_status):
    number_TS_dihedrals = len(mode_status)
    match_count = 0
    for line in mode_status:
        if line[1] == 'Both':
            match_count = match_count + 1

    if match_count == number_TS_dihedrals:
        ring_pucker = 'yes'
    elif (number_TS_dihedrals-match_count) == 1:
        ring_pucker = 'yes'
    else:
        ring_pucker = 'no'

    return filename, ring_pucker


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
                                                 "bonds, angles, and most important dihedrals. In order to differentiate "
                                                 "ring pucker and exocyclic group TS, large contributions to the "
                                                 "imaginery frequency must come from dihedral angles with the ring atoms"
                                                 "at the middle pair.")

    parser.add_argument('-d', "--dir_norm", help="The directory where the hartree norm files can be found.",
                        default=None)
    parser.add_argument('-s', "--sum_file", help="List of the files complete in Hartree norm.",
                        default=None)
    parser.add_argument('-r', "--ring_order", help="List of the atom ids in any order.")



    args = None
    try:
        args = parser.parse_args(argv)
        if args.sum_file is None:
            raise InvalidDataError("Input files are required. Missing hartree input or two-file inputs")
        elif not os.path.isfile(args.sum_file):
            raise IOError("Could not find specified hartree summary file: {}".format(args.sum_file))
        # Finally, if the summary file is there, and there is no dir_xyz provided
        if args.dir_xyz is None:
            args.dir_xyz = os.path.dirname(args.sum_file)
        # if a  dir_xyz is provided, ensure valid
        elif not os.path.isdir(args.dir_xyz):
            raise InvalidDataError("Invalid path provided for '{}': ".format('-d, --dir_xyz', args.dir_xyz))

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

        list_of_gaussian_norm = read_csv_to_dict(args.sum_file, mode='r')

        for hartree_filename in list_of_gaussian_norm:
            out_filename, dihedral_table = read_puckering_information(hartree_filename, args.dir_norm)
            out_filename, first_mode_information = create_dihedral(out_filename, dihedral_table)
            out_filename, status_ring_puckering = identifying_ring_pucker_di(out_filename,
                                                        first_mode_information,args.ring_order)
            out_filename, ring_pucker_status =

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
