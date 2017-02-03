#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this python script is to create csv files that will be uploaded into Igor Pro for data analysis. The
coordinates that are required are phi and theta.
"""

from __future__ import print_function

import argparse
import os

import numpy as np
import sys

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
    parser.add_argument('lm', "--lm_file", help="The hartree output for the combined low energy lm and lmirc "
                                                "structures.",
                        default=None)
    parser.add_argument('-raw_lm', "--raw_lm", help="The hartree output for just the lm structures (not sorted via"
                                                    " xyz_cluster.",
                        default=None)
    parser.add_argument('raw_lmirc', "--raw_lmirc", help="The hartree output for just the lmirc structures.",
                        default=None)
    parser.add_argument('-m', "--mole", help="The molecule currently being studied.")



    # args = None
    # try:
    #     args = parser.parse_args(argv)
    #     if args.sum_file is None:
    #         raise InvalidDataError("Input files are required. Missing hartree input or two-file inputs")
    #     elif not os.path.isfile(args.sum_file):
    #         raise IOError("Could not find specified hartree summary file: {}".format(args.sum_file))
    #     # Finally, if the summary file is there, and there is no dir_xyz provided
    #     if args.dir_xyz is None:
    #         args.dir_xyz = os.path.dirname(args.sum_file)
    #     # if a  dir_xyz is provided, ensure valid
    #     elif not os.path.isdir(args.dir_xyz):
    #         raise InvalidDataError("Invalid path provided for '{}': ".format('-d, --dir_xyz', args.dir_xyz))
    #
    # except (KeyError, InvalidDataError) as e:
    #     warning(e)
    #     parser.print_help()
    #     return args, INPUT_ERROR
    # except IOError as e:
    #     warning(e)
    #     parser.print_help()
    #     return args, IO_ERROR
    # except (ValueError, SystemExit) as e:
    #     if e.message == 0:
    #         return args, GOOD_RET
    #     warning(e)
    #     parser.print_help()
    #     return args, INPUT_ERROR

    return args, GOOD_RET


# Main #
