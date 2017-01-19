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

TOL_centroid = [0.001, 0.001, 0.001]
DEF_TOL_CLUSTER = 0.001
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


def get_coordinates_xyz(filename, xyz_dir, ring_atom_order):
    """This function is designed to upload xyz coordinates from .xyz files. The .xyz file format should contain the
    number of atoms on the first line followed by the filename on the second line. After the second line, the lines
    should be organized as followed:

    atom_type number         x coordinate        y coordinate        z coordinate

    This function stores the the xyz coordinates along with the atom_type xyz_coords.

    :param filename: The input file must be an xyz file
    :param xyz_dir: The directory that the xyz files are located is needed
    @return: A list of coordinates associated with the atom_type xyz_coords and
       a list of lists containing the xyz coordinates for the atoms.
    """
    xyz_file_path = create_out_fname(filename, base_dir=xyz_dir, ext='.xyz')

    f = open(xyz_file_path, mode='r')

    xyz_atoms = []
    atoms_ring_order = [None] * 6
    total_num_atoms = 0
    atom_num = 0
    # Read the first line to obtain the number of atoms read
    try:
        total_num_atoms = int(f.next())
    except ValueError:
        exit("Could not obtain the number of atoms in the .xyz file.")
    # Skip the title line
    f.next()

    xyz_coords_ring = np.full((num_atoms_ring, 3), np.nan)
    # creates an array that will be populated later with xyz coordinates
    xyz_coords = np.full((total_num_atoms, 3), np.nan)

    for line in f:
        if atom_num == total_num_atoms:
            break
        atom_type, coor_x, coor_y, coor_z = line.split()
        # map to take all of the coordinates and turn them into xyz_coords using float option
        coord_floats = map(float, [coor_x, coor_y, coor_z])
        if len(coord_floats) == 3:
            xyz_coords[atom_num] = coord_floats
            xyz_atoms.append(atom_type)

        if atom_num in ring_atom_order:
            ring_index = ring_atom_order.index(atom_num)
            xyz_coords_ring[ring_index] = coord_floats
            atoms_ring_order[ring_index] = atom_type
        atom_num += 1
    f.close()

    list_atoms = xyz_atoms

    xyz_atoms = np.array(xyz_atoms)

    return total_num_atoms, xyz_atoms, xyz_coords, atoms_ring_order, xyz_coords_ring, list_atoms


def translate_centroid_all(xyz_coords):
    """ Calculates the centroid of the xyz coordinates based on all the atoms. Once the centroid is found, then the
    xyz coordinates are translated so that the centroid is at the origin.

    :param xyz_coords: the xyz coordinates for the molecular structure (centroid not aligned)
    :return: outputs the xyz coordinates for the molecular structure with the centroid located at the origin.
    """

    centroid_xyz = sum(xyz_coords) / len(xyz_coords)

    xyz_coords_translate = np.array([xyz_coords - centroid_xyz])

    xyz_coords_translate = np.array(np.mat(xyz_coords_translate))
    return xyz_coords_translate


def translate_centroid_ring(xyz_coords, xyz_coords_ring):
    """This script is designed to calculate the centroid of the xyz coordinates based on only the atoms in the ring.
    Once the centroid of the ring is found, then the all xyz coordinates are translated so that the centroid of the ring
    is at the origin.

    :param xyz_coords: the xyz coordinates for the molecular structure (centroid not aligned)
    :param xyz_coords_ring: the xyz coordinates for the ring of the molecular structure
    :return: outputs the xyz coordinates for the molecular structure with the ring centroid located at the origin.
    """

    centroid_ring_xyz = sum(xyz_coords_ring) / len(xyz_coords_ring)

    xyz_coords_all_translate = np.array([xyz_coords - centroid_ring_xyz])
    xyz_coords_ring_translate = np.array([xyz_coords_ring - centroid_ring_xyz])

    xyz_coords_all_translate = np.array(np.mat(xyz_coords_all_translate))
    xyz_coords_ring_translate = np.array(np.mat(xyz_coords_ring_translate))

    return xyz_coords_all_translate, xyz_coords_ring_translate


def rmsd(coords1, coords2):
    """Calculates the root-mean-square deviation from two sets of vectors xyz_1 and xyz_2 (both of which are xyz
    coordinates for different molecules)

    :param coords1: input xyz coordinates of a molecule
    :param coords2: input xyz coordinates of a molecule
    :return: the root-mean-square deviation for the two molecules
    """

    num_dim = len(coords1[0])  # number of dimensions in system
    num_atoms = len(coords2)  # number of atoms in system
    rmsd_val = 0.0  # initial value of the rmsd

    for v, w in zip(coords1, coords2):
        rmsd_val += sum([(v[i] - w[i]) ** 2.0 for i in range(num_dim)])

    rmsd_value = np.sqrt(rmsd_val / num_atoms)

    return rmsd_value, num_dim, num_atoms


def kabsch_algorithm(xyz_coords1, xyz_coords2):
    """ Method for calculating the optimal rotation matrix that minimizes the RMSD between two paired sets of points.
    The method rotates the structures and ensures the correct orientation of the structure.

    More general information can be found at:
    https://en.wikipedia.org/wiki/Kabsch_algorithm

    More complex information on the algorithm can be found at:
    http://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures

    :param xyz_coords1: xyz coordinates for the first structure
    :param xyz_coords2: xyz coordinates for the second structure
    :return: the rmsd using the kabsch algorithm
    """

    # calculate the covariance matrix between the two sets of xyz coordinates
    covariance_matrix = np.dot(np.transpose(xyz_coords1), xyz_coords2)

    # calculate the singular value decomposition (svd) of the covariance matrix using numpy linear algebra
    v, s, w = np.linalg.svd(covariance_matrix)

    # check if the systems needs to be rotated to ensure a right-handed coordinate system
    direction = (np.linalg.det(v) * np.linalg.det(w))

    if direction < 0.0:
        s[-1] = -s[-1]
        v[:, -1] = - v[:, -1]

    rotation_matrix = np.dot(v, w)

    rotated_xyz_coords1 = np.dot(xyz_coords1, rotation_matrix)

    kabsch_rsmd = rmsd(rotated_xyz_coords1, xyz_coords2)

    return kabsch_rsmd


def check_ring_ordering(atoms_ring_order1, atoms_ring_order2):
    """ This script ensure that the atom arrangement is the same. Arranging the atoms differently between two xyz files
    could result in differences between in the calculations. Must compare the oxygen atoms to the oxygen atoms and the
    carbon atoms to the carbon atoms.

    :param atoms_ring_order1: input a list (in order) of the atoms of the ring
    :param atoms_ring_order2: input a list (in order) of the atoms of the ring
    :return: message saying the status of the ordering
    """

    if atoms_ring_order1 != atoms_ring_order2:
        print('The atoms in the ring are not aligned the same!')
        check_value = 1
    else:
        check_value = 0
    return check_value


def print_xyz_coords(to_print_xyz_coords, to_print_atoms, file_sum):
    """ Prints the xyz coordinates of the updates structure for further analysis (the structures can be input into VMD)

    :param to_print_xyz_coords: the xyz coordinates of the structure that you are going to print
    :param to_print_atoms: the atoms in the same ordering as xyz coordinates that you want to print
    :param file_sum: summary information on what file these coordinates originated from
    :return:
    """

    num_atoms = str(len(to_print_atoms))

    to_print = [num_atoms, file_sum]
    to_print2 = list(to_print)

    for atom_type, atom_xyz in zip(to_print_atoms, to_print_xyz_coords):
        to_print2.append([atom_type] + atom_xyz.tolist())

    list_to_file(to_print2, file_sum)

    return


def compare_rmsd_xyz(input_file1, input_file2, xyz_dir, ring_atom_order, print_option='off'):
    """ calculates the rmsd both using the standard method and rotating the structures method

    :param input_file1: xyz coordinates for the first molecular structure
    :param input_file2: xyz coordinates for the second molecular structure
    :param xyz_dir: the location of the xyz coordinates that are going to be printed
    :param ring_atom_order: user input of index of atoms in the ring
    :param print_option: has the ability to print out the output..or not
    :return: rmsd using the kabsch method, coordinates of the centered rings
    """
    atom_ordering = None
    n_atoms1, atoms1, xyz_coords1, atoms_ring_order1, xyz_coords_ring1, list_atoms1 \
        = get_coordinates_xyz(input_file1, xyz_dir, ring_atom_order)
    n_atoms2, atoms2, xyz_coords2, atoms_ring_order2, xyz_coords_ring2, list_atoms2 = \
        get_coordinates_xyz(input_file2, xyz_dir, ring_atom_order)

    if n_atoms1 != n_atoms2:
        exit("Error in the number of atoms! The number of atoms doesn't match!")

    check_value = check_ring_ordering(atoms_ring_order1, atoms_ring_order2)

    if check_value != 0:
        exit("The atoms alignment isn't the same!")
    elif check_value == 0:
        check_value_all = check_ring_ordering(list_atoms1, list_atoms2)
        if check_value_all == 0:
            atom_ordering = list_atoms1

    center_xyz1 = translate_centroid_all(xyz_coords1)
    center_xyz2 = translate_centroid_all(xyz_coords2)

    [center_ring_all_xyz1, center_ring_ring_xyz1] = translate_centroid_ring(xyz_coords1, xyz_coords_ring1)
    [center_ring_all_xyz2, center_ring_ring_xyz2] = translate_centroid_ring(xyz_coords2, xyz_coords_ring2)

    if print_option == 'on':
        print("""Now print the different cases:
        Rmsd (all align, standard): {}
        Rmsd (ring align, standard): {}
        Rmsd (all align, kabsch): {}
        Rmsd (ring align, kabsch): {}
        """.format(rmsd(center_xyz1, center_xyz2), rmsd(center_ring_ring_xyz1, center_ring_ring_xyz2),
                   kabsch_algorithm(center_xyz1, center_xyz2),
                   kabsch_algorithm(center_ring_ring_xyz1, center_ring_ring_xyz2)))

    rmsd_kabsch = kabsch_algorithm(center_ring_ring_xyz1, center_ring_ring_xyz2)[0]

    return rmsd_kabsch, center_ring_all_xyz1, center_ring_all_xyz2, atom_ordering


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


def test_clusters(pucker_filename_dict, xyz_dir, ok_tol, ring_num_list, print_option='off'):
    """ Clusters the puckers based on their initial arrangement and RMSD. The puckers initially constructed from Hartree
    are further expanded to ensure the cluster is consistent.

    :param ring_num_list: list of atom numbers in the ring (in order O, C1, C2, C3, C4, C5)
    :param print_option: turns on and off the print option
    :param pucker_filename_dict: lists of dicts for each row of hartree, and a dictionary of puckers (keys) and
        file_names, and a list of headers
    :param xyz_dir: the location of the directory in which all of the files are located (must contain xyz and log files)
    :param ok_tol: the tolerance for when grouping two different structures
    :return: returns a dict (keys being the puckering geometries w/ potential duplicates) of lists (containing the
        clustered file names)
    """
    process_cluster_dict = {}
    xyz_coords_dict = {}
    atoms_order = None
    for pucker, file_list in pucker_filename_dict.items():
        pucker_cluster = 0
        cluster_name = pucker + "_" + str(pucker_cluster)
        process_cluster_dict[cluster_name] = [file_list[0]]
        raw_cluster_len = len(file_list)

        if raw_cluster_len == 1:
            file_name = file_list[0]
            num_atoms, xyz_atoms, xyz_coords, atoms_ring_order, xyz_coords_ring, list_atoms \
                = get_coordinates_xyz(file_name, xyz_dir, ring_num_list)

            xyz_coords_all_translate, xyz_coords_ring_translate = translate_centroid_ring(xyz_coords, xyz_coords_ring)

            xyz_coords_dict[file_name] = xyz_coords_all_translate

        else:

            for file_id in range(1, raw_cluster_len):
                file_name = file_list[file_id]
                not_assigned = True

    #TODO create a check to see if the puckers, when being compared to other puckers are very similar.

                for assigned_cluster_name in process_cluster_dict:
                    (rmsd_kabsch, ctr_ring_all_xyz1, ctr_ring_all_xyz2, atoms_order) = \
                        compare_rmsd_xyz(file_name, process_cluster_dict[assigned_cluster_name][0], xyz_dir, ring_num_list)
                    xyz_coords_dict[file_name] = ctr_ring_all_xyz1
                    xyz_coords_dict[process_cluster_dict[assigned_cluster_name][0]] = ctr_ring_all_xyz2
                    if rmsd_kabsch < ok_tol:
                        process_cluster_dict[assigned_cluster_name].append(file_name)
                        not_assigned = False
                        break
                if not_assigned:
                    pucker_cluster += 1
                    cluster_name = pucker + "_" + str(pucker_cluster)
                    process_cluster_dict[cluster_name] = [file_name]

    if print_option != 'off':
        for cluster_key, cluster_values in process_cluster_dict.items():
            print("Cluster Key: {} Cluster Files: {}".format(cluster_key, cluster_values))
    return process_cluster_dict, xyz_coords_dict, atoms_order


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


def read_ring_atom_ids(atom_str):
    """
    Read entry for the list of atom numbers and convert to a list of ints
    :param atom_str: a string that is ideally a comma-separated list of 6 integers.
    :return: int list
    """
    try:
        int_list = [int(atom_num) for atom_num in atom_str.split(',')]
        if len(int_list) != 6:
            raise ValueError
    except ValueError:
        raise ValueError("Expected a comma-separated list of 6 integers. Read: {}".format(atom_str))
    return int_list


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

    parser.add_argument('-d', "--dir_xyz", help="The directory where the xyz files can be found. The default is the "
                                                "directory where the Hartree summary file can be found.",
                        default=None)
    parser.add_argument('-s', "--sum_file", help="The summary file from Hartree.",
                        default=None)
    parser.add_argument('-t', "--tol", help="Tolerance (allowable RMSD) for coordinates in the same cluster.",
                        default=DEF_TOL_CLUSTER, type=float)
    parser.add_argument('-r', "--ring_order", help="List of the atom ids in the order C1,C2,C3,C4,C5,O which define "
                                                   "the six-membered ring. The default is: {}.".format(DEF_RING_ORDER),
                        default=DEF_RING_ORDER, type=read_ring_atom_ids)
    parser.add_argument('-p', "--xyz_print", help='Prints the xyz coordinates of the aligned structures in the finally '
                                                  'output file from xyz_cluster. To print coordinates please use: '
                                                  ' -p \'true\'',
                        default='false')

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
        hartree_list, pucker_filename_dict, hartree_headers = hartree_sum_pucker_cluster(args.sum_file)
        hartree_dict = list_to_dict(hartree_list, FILE_NAME)
        process_cluster_dict, xyz_coords_dict, atom_order \
            = test_clusters(pucker_filename_dict, args.dir_xyz, args.tol, args.ring_order, print_option='off')
        filtered_cluster_list, filtered_cluster_filename_list \
            = read_clustered_keys_in_hartree(process_cluster_dict, hartree_dict)
        out_f_name = create_out_fname(args.sum_file, prefix='z_cluster_', base_dir=args.dir_xyz, ext='.csv')
        write_csv(filtered_cluster_list, out_f_name, hartree_headers, extrasaction="ignore")

        list_f_name = create_out_fname(args.sum_file, prefix='z_files_list_freq_runs', base_dir=args.dir_xyz,
                                       ext='.txt')

        list_to_file(filtered_cluster_filename_list, list_f_name, list_format=None, delimiter=' ', mode='w',
                     print_message=True)

        list_puckers_missing = check_before_after_sorting(args.sum_file, out_f_name)

        if list_puckers_missing != []:
            print('')
            print('Warning! The following puckers have been dropped: {}.'.format(list_puckers_missing))
            print('')

        if args.xyz_print == 'true':
            for row in filtered_cluster_list:
                filename_written_coords = row[FILE_NAME]
                coords_need_writing = xyz_coords_dict[filename_written_coords]
                filename_xyz_coords = create_out_fname(filename_written_coords, prefix="xyz_",
                                                       suffix="-xyz_updated", base_dir=args.dir_xyz,
                                                       ext=".xyz")
                print_xyz_coords(coords_need_writing, atom_order, filename_xyz_coords)


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
