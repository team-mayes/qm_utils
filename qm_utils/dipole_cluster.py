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

TOL_centroid = [0.001, 0.001, 0.001]
DEF_TOL_CLUSTER = 0.05
DEF_RING_ORDER = '5,0,1,2,3,4'
num_atoms_ring = 6
ACCEPT_AS_TRUE = ['T', 't', 'true', 'TRUE', 'True']
HARTREE_TO_KCALMOL = 627.5095
STRUCTURE_COMPARE_TOL = 5.0
TRIGGER_WARN_TOL = 2.50

# Hartree field headers
FILE_NAME = 'File Name'
PUCKER = 'Pucker'
ENERGY_GIBBS = 'G298 (Hartrees)'
ENERGY_ELECTRONIC = 'Energy (A.U.)'
DIPOLE = 'dipole'


def hartree_sum_pucker_cluster(sum_file, print_status='off'):
    """
    Reads the hartree output file and creates a dictionary of all hartree output and clusters based on pucker

    :param print_status: turns the print status on and off
    :param sum_file: name of hartree output file
    :return: lists of dicts for each row of hartree, and a dictionary of puckers (keys) and file_names,
        and a list of headers
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

        if print_status != 'off':
            print("Hartree Pucker: {} --> {}".format(row[PUCKER], row[FILE_NAME]))

    return hartree_dict, pucker_filename_dict, hartree_headers


def test_clusters(pucker_filename_dict, hartree_dict, ok_tol):
    """ Clusters the puckers based on their initial arrangement and RMSD. The puckers initially constructed from Hartree
    are further expanded to ensure the cluster is consistent.

    :param hartree_dict: the hartree dict with the outer key as the filename
    :param pucker_filename_dict: lists of dicts for each row of hartree, and a dictionary of puckers (keys) and
        file_names, and a list of headers
    :param ok_tol: the tolerance for when grouping two different structures
    :return: returns a dict (keys being the puckering geometries w/ potential duplicates) of lists (containing the
        clustered file names)
    """

    process_cluster_dict = {}

    for pucker, file_list in pucker_filename_dict.items():
        pucker_cluster = 0
        cluster_name = pucker + '-' + str(pucker_cluster)
        process_cluster_dict[cluster_name] = [file_list[0]]
        raw_cluster_len = len(file_list)
        # print('{} -- {}'.format(raw_cluster_len, pucker))
        # print(file_list)
        # print('')

        if raw_cluster_len == 1:
            pass
        else:
            for file_id in range(1, raw_cluster_len):
                file_name = file_list[file_id]
                not_assigned = True
                for assigned_cluster_name in process_cluster_dict:
                    # print(assigned_cluster_name.split('-')[0])
                    # print(hartree_dict[file_name][PUCKER])

                    if assigned_cluster_name.split('-')[0] == hartree_dict[file_name][PUCKER]:

                        dipole_difference = abs(float(hartree_dict[file_name][DIPOLE]) -
                                                float(hartree_dict[process_cluster_dict[assigned_cluster_name][0]][
                                                          DIPOLE]))
                        if dipole_difference < ok_tol:
                            process_cluster_dict[assigned_cluster_name].append(file_name)
                            not_assigned = False
                            break
                if not_assigned:
                    pucker_cluster += 1
                    cluster_name = pucker + "-" + str(pucker_cluster)
                    process_cluster_dict[cluster_name] = [file_name]

    return process_cluster_dict


def read_clustered_keys_in_hartree(process_cluster_dict, hartree_dict):
    """ Select only one file name from each cluster (based on the lowest energy)

    :param process_cluster_dict: returns a dict (keys being the puckering geometries w/ potential duplicates) of lists
        (containing the clustered file names)
    :param hartree_dict: a dict of dicts (where the outer key is the file name form the inner key) and the inner dict is
        with the keys and the corresponding value from Hartree
    :return: a list containing all of the low energy files and information
    """
    low_e_per_cluster = []
    low_e_per_cluster_filename_list = []

    for cluster_keys, cluster_file_names in process_cluster_dict.items():
        cluster_low_filename = cluster_file_names[0]
        cluster_low_e = float(hartree_dict[cluster_low_filename][ENERGY_ELECTRONIC]) * HARTREE_TO_KCALMOL

        # THIS IS WHERE THE ENERGY ERROR WARNING IS PRINTED
        for selected_file_cluster in cluster_file_names[1:]:
            test_cluster_dict = hartree_dict[selected_file_cluster]
            cluster_test_energy = float(test_cluster_dict[ENERGY_ELECTRONIC]) * HARTREE_TO_KCALMOL
            if abs(cluster_test_energy - cluster_low_e) > TRIGGER_WARN_TOL:
                print("Energy difference ({}) within cluster '{}' is greater than {}."
                      "Check files: {}, {}".format(abs(cluster_test_energy - cluster_low_e), cluster_keys,
                                                   TRIGGER_WARN_TOL, selected_file_cluster, cluster_low_filename))
            if cluster_test_energy < cluster_low_e:
                cluster_low_filename = selected_file_cluster
                cluster_low_e = cluster_test_energy

        low_e_per_cluster.append(hartree_dict[cluster_low_filename])
        low_e_per_cluster_filename_list.append(cluster_low_filename)

    return low_e_per_cluster, low_e_per_cluster_filename_list


def check_before_after_sorting(hartree_unsorted, hartree_sorted):
    """ Function checks to make sure that there is no information lose before and after the sorting process. For a TS
    hartree run, there was an issue where not all of the puckers before sorting were found after sorting."

    :param hartree_unsorted: hartree output of the unsorted puckers
    :param hartree_sorted: xyz_cluster output (in a hartree format) of the sorted puckers
    :return:
    """

    hartree_dict_unsorted = read_csv_to_dict(hartree_unsorted, mode='rU')
    hartree_dict_sorted = read_csv_to_dict(hartree_sorted, mode='rU')

    list_puckers_unsorted = []
    list_puckers_sorted = []
    list_puck_missing = []

    for row_unsorted in hartree_dict_unsorted:
        list_puckers_unsorted.append(row_unsorted[PUCKER])
    list_no_duplicates_unsorted = list(set(list_puckers_unsorted))

    for row_sorted in hartree_dict_sorted:
        list_puckers_sorted.append(row_sorted[PUCKER])

    list_puckers_both = list(set(list_puckers_sorted).intersection(set(list_puckers_unsorted)))

    for pucker_sorted in list_no_duplicates_unsorted:
        if pucker_sorted not in list_puckers_both:
            list_puck_missing.append(pucker_sorted)
            print('Something is not right! Puckers before and after are not the same.')
            print('The following puckers have been lost: {}.'.format(pucker_sorted))

    return list_puck_missing


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="Creates a list of the lowest energy pucker in each pucker grouping "
                                                 "from a Hartree input file. The script calculates the rmsd between "
                                                 "two sets of xyz coordinates based on the 6 membered ring to verify "
                                                 "that all structures belong to the same pucker. Next, the script "
                                                 "compares the lowest energy of each pucker group to select the final "
                                                 "structures for further analysis. The output is a condensed csv file "
                                                 "that follows the same form as Hartree.")
    parser.add_argument('-s', "--sum_file", help="The summary file from Hartree.",
                        default=None)
    parser.add_argument('-t', "--tol", help="Tolerance (allowable RMSD) for coordinates in the same cluster.",
                        default=DEF_TOL_CLUSTER, type=float)

    args = None
    try:
        args = parser.parse_args(argv)
        if args.sum_file is None:
            raise InvalidDataError("Input files are required. Missing hartree input or two-file inputs")
        elif not os.path.isfile(args.sum_file):
            raise IOError("Could not find specified hartree summary file: {}".format(args.sum_file))

    except (KeyError, InvalidDataError) as e:
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    except IOError as e:
        warning(e)
        parser.print_help()
        return args, IO_ERROR
    except (ValueError, SystemExit) as e:
        if e.args == 0:
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
        hartree_list, pucker_filename_dict, hartree_headers = hartree_sum_pucker_cluster(args.sum_file)
        hartree_dict = list_to_dict(hartree_list, FILE_NAME)
        process_cluster_dict = test_clusters(pucker_filename_dict, hartree_dict, args.tol)
        filtered_cluster_list, filtered_cluster_filename_list \
            = read_clustered_keys_in_hartree(process_cluster_dict, hartree_dict)
        out_f_name = create_out_fname(args.sum_file, prefix='z_cluster_', base_dir=os.path.dirname(args.sum_file),
                                      ext='.csv')
        write_csv(filtered_cluster_list, out_f_name, hartree_headers, extrasaction="ignore")

        list_f_name = create_out_fname(args.sum_file, prefix='z_files_list_freq_runs',
                                       base_dir=os.path.dirname(args.sum_file),
                                       ext='.txt')

        list_to_file(filtered_cluster_filename_list, list_f_name, list_format=None, delimiter=' ', mode='w',
                     print_message=True)

        list_puckers_missing = check_before_after_sorting(args.sum_file, out_f_name)

        if list_puckers_missing != []:
            print('')
            print('Warning! The following puckers have been dropped: {}.'.format(list_puckers_missing))
            print('')

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
