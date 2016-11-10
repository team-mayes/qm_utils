#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this python script to align xyz coordinate files so that the structures
can be clustered. The script calculates the rmsd for structures in a given hartree cluster designation, and then
determines the lowest energy structure.
"""

from __future__ import print_function

import argparse
import os
import sys
import numpy as np
from qm_common import (GOOD_RET, INVALID_DATA, warning, InvalidDataError, IO_ERROR, INPUT_ERROR, list_to_file,
                       read_csv_to_dict, create_out_fname, list_to_dict, get_csv_fieldnames, write_csv)

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

def load_cluster_file(sum_file):
    """ Inputs the cluster hartree csv files to story CSV information as a dict

    :param sum_file: xyz_cluster - hartree file of the lowest energy structures
    :return:
    """
    hartree_dict = read_csv_to_dict(sum_file, mode='rU')
    hartree_headers = get_csv_fieldnames(sum_file, mode='rU')

    pucker_filename_dict = {}

    for row in hartree_dict:
        pucker_name = row[PUCKER]
        file_name = row[FILE_NAME]
        if pucker_name in pucker_filename_dict:
            pucker_filename_dict[pucker_name].append(file_name)
        else:
            pucker_filename_dict[pucker_name] = [file_name]

    return hartree_dict, pucker_filename_dict, hartree_headers


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="This script is designed to create a table for the puckering project "
                                                 "similar to the table with each of the puckering geometries on the left "
                                                 " and the different levels of theory accross the top. The table values "
                                                 "are the relative energies of the conformation to the 4c1 gibbs free "
                                                 "energy.")

    parser.add_argument('-d', "--dir_files", help="The directory where the clustered output files can be found. The "
                                                  "files must be sorted by xyz_cluster so that only the lowest energy "
                                                  "puckering conformations are selected.",
                        default=None)

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
    except SystemExit as e:
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
        print('hi')
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
