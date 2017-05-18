#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this script is to make comparisons for a particular QM method to the reference set of HSP.
"""

from __future__ import print_function

import argparse
import os
import statistics as st
import sys

import csv
import pandas as pd
import math
import numpy as np

from qm_utils.igor_mercator_organizer import write_file_data_dict
from qm_utils.pucker_table import read_hartree_files_lowest_energy, sorting_job_types

from qm_utils.qm_common import (GOOD_RET, create_out_fname, warning, IO_ERROR, InvalidDataError, INVALID_DATA,
                                INPUT_ERROR, arc_length_calculator, read_csv_to_dict)

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

# # Default Parameters # #
HARTREE_TO_KCALMOL = 627.5095
TOL_ARC_LENGTH = 0.1
TOL_ARC_LENGTH_CROSS = 0.2  # THIS WAS THE ORGINAL TOLERANCE6
DEFAULT_TEMPERATURE = 298.15
K_B = 0.001985877534  # Boltzmann Constant in kcal/mol K

# # Pucker Keys # #
FILE_NAME = 'File Name'
PUCKER = 'Pucker'
ENERGY_ELECTRONIC = 'Energy (A.U.)'
THETA = 'theta'
PHI = 'phi'
Q_VAL = 'Q'
GIBBS = 'G298 (Hartrees)'
ENTH = "H298 (Hartrees)"
MPHI = 'mean phi'
MTHETA = 'mean theta'
GID = 'group ID'
WEIGHT_GIBBS = 'Boltz Weight Gibbs'
WEIGHT_ENTH = 'Boltz Weight Enth'
FREQ = 'Freq 1'


# # # Classes # # #
class Local_Minima_Compare():
    """
    class for organizing the local minima information
    """
    def __init__(self, list_of_dicts, method, hsp_lm_groups):
        print(method)

        # Perform the following operations on the local min data set:
        # (1) assign each of the unique local min to a particular HSP reference group
        #       (be sure to store the arc_length value to the closest local min AND the group ID)
        # (2) plot HSP reference groups and the data points associated with each (save all files into a specific folder)
        #       a) overall plot
        #       b) folder for each HSP reference group
        # (3) within each group, perform RMSD calculations on arc_length? and gibbs free energies




class Transition_State_Compare():
    """
    class for organizing the transition state information
    """
    def  __init__(self, list_of_dicts, method, hsp_ts_groups):
        print(method)
        self.ts_list_of_dicts = []
        self.irc_list_of_dicts = []

        self.separate_TS_files(list_of_dicts)

    def separate_TS_files(self, list_of_dicts):
        """
        separates out the TS files for further processing
        :param list_of_dicts:
        :return:
        """
        for row in list_of_dicts:
            if float(row[FREQ]) < 0:
                self.ts_list_of_dicts.append(row)
            elif float(row[FREQ]) > 0:
                self.irc_list_of_dicts.append(row)

        if len(self.irc_list_of_dicts) / 2 != len(self.ts_list_of_dicts):
            print('\nThere are {} TS files and {} IRC files...THERE IS A PROBLEM.\n'.
                  format(len(self.ts_list_of_dicts),len(self.irc_list_of_dicts)))

        return


        # Perform the following operations on the transition state data set:
        # (1) isolate the TS and IRC files by 'Freq 1' values and naming convention (also link files together) -- DONE.
        # (2) assign each of the transition states to a particular HSP reference group
        #       (assigning pathway will need to be on both the transition state AND the local minima connecting them)
        # (3) within each of the assign group, perform RMSD calculations on the arc length and gibbs free energies
        # (4) develop a similar plotting strategy (more thought needed)


# # # Command Line Parse # # #

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
    parser.add_argument('-c', "--ccsdt", help="The CCSD(T) file for the molecule being studied",
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


# # # Main # # #
def main(argv=None):
    """
    Runs the main program
    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return args, ret
    try:

        #TODO: import the class information from spherical_kmean_voronoi.py based on the molecule type

        # Need to come up with a simple way to local in the necessary HSP reference information based on args.molecule (database? HSP reference files?)

        hsp_lm_groups = 1
        hsp_ts_groups = 1

        with open(args.sum_file) as f:
            for csv_file_read_newline in f:
                csv_file_read = csv_file_read_newline.strip("\n")
                method_list_dicts = read_csv_to_dict(os.path.join(args.dir_hartree, csv_file_read), mode='r')
                qm_method = csv_file_read.split('-')[3].split('.')[0]

                if csv_file_read.split('-')[1] != args.molecule:
                    print('\nERROR: THE MOLECULE TYPE DOES NOT MATCH UP.\n')
                    break

                if csv_file_read.split('-')[2] == 'LM':
                    print(csv_file_read)

                    data_lm = Local_Minima_Compare(method_list_dicts, qm_method, hsp_lm_groups)

                    #TODO: run a local min class on this data set...


                elif csv_file_read.split('-')[2] == 'TS':
                    print(csv_file_read)
                    #TODO: run a transition state class on this data set...

                    data_ts = Transition_State_Compare(method_list_dicts, qm_method, hsp_ts_groups)



                else:
                    print('WARNING: NOT SURE WHAT TYPE OF JOB THIS IS')

            #TODO: come up with a method for storing the information from each run so that later on

            # A class with the important data from the above calculations is probably the best way to go about it?
            # Also...should write out the data to a csv file for later processing if necessary.



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
