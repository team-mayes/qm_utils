#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this python script to align xyz coordinate files so that the structures
can be clustered.
"""

from __future__ import print_function

import argparse
import os
import sys
import numpy as np
# TODO check error message that I am receiving only when running on my computer
from qm_common import GOOD_RET, INVALID_DATA, warning, InvalidDataError, IO_ERROR, INPUT_ERROR, list_to_file, read_csv_to_dict

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

# Constants #

TOL_centroid = [0.001, 0.001, 0.001]
num_atoms_ring = 6

ACCEPT_AS_TRUE = ['T', 't', 'true', 'TRUE', 'True']

# Hartree field headers
FILE_NAME = 'File Name'
PUCKER = 'Pucker'


def get_coordinates_xyz(filename):
    """This function is designed to upload xyz coordinates from .xyz files. The .xyz file format should contain the
    number of atoms on the first line followed by the filename on the second line. After the second line, the lines
    should be organized as followed:

    atom number         x coordinate        y coordinate        z coordinate

    This function stores the the xyz coordinates along with the atom numbers.

    :param filename:
    @return: A list of coordinates associated with the atom numbers and a list of lists containing the xyz coordinates
        for the atoms.
    """

    f = open(filename, mode='r')

    xyz_atoms = []
    atoms_ring_order = []
    num_atoms = 0
    lines_read = 0
    lines_read_ring = 0
    # Read the first line to obtain the number of atoms read
    try:
        num_atoms = int(f.next())
    except ValueError:
        exit("Could not obtain the number of atoms in the .xyz file.")
    # Skip the title line
    f.next()

    xyz_coords_ring = np.full((num_atoms_ring, 3), np.nan)
    xyz_coords = np.full((num_atoms, 3), np.nan)  # creates an array that will be populated later with xyz coordinates

    for line in f:
        if lines_read == num_atoms:
            break
        atom, coor_x, coor_y, coor_z = line.split()
        # map to take all of the coordinates and turn them into numbers using float option
        numbers = map(float, [coor_x, coor_y, coor_z])
        if len(numbers) == 3:
            xyz_coords[lines_read] = numbers
            xyz_atoms.append(atom)

        if atom != '1':
            xyz_coords_ring[lines_read_ring] = numbers
            atoms_ring_order.append(atom)
            lines_read_ring += 1

        lines_read += 1
    f.close()

    xyz_atoms = np.array(xyz_atoms)

    return num_atoms, xyz_atoms, xyz_coords, atoms_ring_order, xyz_coords_ring


def print_xyz_coord_info(num_atoms, xyz_atoms, xyz_coords):
    """ Prints the information from the get_coordinates_xyz
    :param num_atoms: the number of atoms in the molecule
    :param xyz_atoms: prints the order of the atoms (used to know which coordinates are which in xyz_cords)
    :param xyz_coords: prints the xyz coordinates (the row correspondence can be found from xyz_atoms)
    :return:
    """
    print("Your molecule currently has {} atoms.".format(num_atoms))
    print("\nThe atom ordering is:\n {}".format(xyz_atoms))
    print("\nThe XYZ coordinates are:\n {}".format(xyz_coords))


def centroid(xyz_coords):
    """ Calculates the centroid (geometric center) of the structure. The centroid
    is defined as the mean position of all the points in all of the coordinate directions.

    :param xyz_coords: xyz coordinates for the given molecular structure
    :return: the xyz coordinates for the centroid of the structure
    """
    centroid_xyz = sum(xyz_coords) / len(xyz_coords)
    return centroid_xyz


def translate_centroid_all(xyz_coords):
    """ Calculates the centroid of the xyz coordinates based on all the atoms. Once the centroid is found, then the
    xyz coordinates are translated so that the centroid is at the origin.

    :param xyz_coords: the xyz coordinates for the molecular structure (centroid not aligned)
    :return: outputs the xyz coordinates for the molecular structure with the centroid located at the origin.
    """

    centroid_xyz = sum(xyz_coords) / len(xyz_coords)

    xyz_coords_translate = np.array([xyz_coords - centroid_xyz])

    centroid_xyz_check = sum(xyz_coords_translate) / len(xyz_coords_translate)

    if centroid_xyz_check.all > TOL_centroid:
        exit("\nCould not properly align the centroid to the origin\n.")

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
        exit("The atoms in the ring are not aligned the same!")
    else:
        check_value = 0
        return check_value


def print_xyz_coords(to_print_xyz_coords, to_print_atoms, file_sum):
    """

    :param to_print_xyz_coords:
    :param to_print_atoms:
    :param file_sum:
    :return:
    """

    # num_atoms = len(to_print_atoms)
    num_atoms = '16'

    to_print = [num_atoms, file_sum]
    to_print2 = list(to_print)

    #    for line_id in range(len(atoms1)):
    #        to_print.append([atoms1[line_id]] + xyz_coords1[line_id].tolist())

    for atom_type, atom_xyz in zip(to_print_atoms, to_print_xyz_coords):
        to_print2.append([atom_type] + atom_xyz.tolist())

    list_to_file(to_print2, file_sum)

    return


def compare_rmsd_xyz(input_file1, input_file2, print_status='off'):
    """ calculates the rmsd both using the standard method and rotating the structures

    :param input_file1: xyz coordinates for the first molecular structure
    :param input_file2: xyz coordinates for the second molecular structure
    :param print_status:
    :return: returns all of the
    """
    n_atoms1, atoms1, xyz_coords1, atoms_ring_order1, xyz_coords_ring1 = get_coordinates_xyz(input_file1)
    n_atoms2, atoms2, xyz_coords2, atoms_ring_order2, xyz_coords_ring2 = get_coordinates_xyz(input_file2)

    if n_atoms1 != n_atoms2:
        exit("Error in the number of atoms! The number of atoms doesn't match!")

    check_value = check_ring_ordering(atoms_ring_order1, atoms_ring_order2)

    if check_value != 0:
        exit("The atoms alignment isn't the same!")

    center_xyz1 = translate_centroid_all(xyz_coords1)
    center_xyz2 = translate_centroid_all(xyz_coords2)

    [center_ring_all_xyz1, center_ring_ring_xyz1] = translate_centroid_ring(xyz_coords1, xyz_coords_ring1)
    [center_ring_all_xyz2, center_ring_ring_xyz2] = translate_centroid_ring(xyz_coords2, xyz_coords_ring2)

    if print_status == 'on':
        print("""Now print the different cases:
        Rmsd (all align, standard): {}
        Rmsd (ring align, standard): {}
        Rmsd (all align, kabsch): {}
        Rmsd (ring align, kabsch): {}
        """.format(rmsd(center_xyz1, center_xyz2), rmsd(center_ring_ring_xyz1, center_ring_ring_xyz2),
                   kabsch_algorithm(center_xyz1, center_xyz2),
                   kabsch_algorithm(center_ring_ring_xyz1, center_ring_ring_xyz2)))

    rmsd_kabsch = kabsch_algorithm(center_ring_ring_xyz1, center_ring_ring_xyz2)[0]

    return rmsd_kabsch, center_ring_all_xyz1, center_ring_all_xyz2


def hartree_sum_pucker_cluster(sum_file, print_status='off'):
    """
    Reads the hartree output file and creates a dictionary of all hartree output and clusters based on pucker
    :param sum_file: name of hartree output file
    :return: lists of dicts for each row of hartree, and a dictionary of puckers (keys) and file_names
    """
    # TODO: make clusters based on puckers
    hartree_dict = read_csv_to_dict(sum_file, mode='rU')
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

    return hartree_dict, pucker_filename_dict


def test_clusters(pucker_filename_dict):
    """
    What I do
    :param pucker_filename_dict:
    :return:
    """

    # TODO need to figure out why the clusters are arranged so that nothing else is being added or overwritten
    #ok_tol = 0.0000000000000000000000001
    ok_tol = 0.1
    process_cluster_dict = {}
    for pucker, file_list in pucker_filename_dict.items():
        pucker_cluster = 0 # initial pucker list count is 0
        cluster_name = pucker + "_" + str(pucker_cluster) # creates new name for cluster key
        process_cluster_dict[cluster_name] = [file_list[0]]  # adds new cluster key and first file into items of key
        raw_cluster_len = len(file_list) # calculates the length of the file list (how many files in hartree clustering)

        for file_id in range(1, raw_cluster_len ): # looks at all the files in the initial clustering
            file_name = file_list[file_id] # looks at a specific filename
            # TODO Figure out why I am not going into this for loop...since pucker cluster is empty
#            hi = range(file_id)
#            print(range(file_id)) # prints the number of clusters that needs to be further analyzed
            for cluster_id in range(pucker_cluster):
                rmsd_kabsch, ctr_ring_all_xyz1, ctr_ring_all_xyz2 = compare_rmsd_xyz(file_name,
                                                                                     process_cluster_dict[cluster_id][0])
                # calculates the rmsd by rotating and translating the rings so that they align properly

                if rmsd_kabsch < ok_tol:
                    process_cluster_dict[cluster_id].append(file_name) # add the file to the current key
                else:
                    pucker_cluster += 1 # say the criteria isn't met so another key is needed
                    cluster_name = pucker + "_" + str(pucker_cluster) # creates the new name for cluster key
                    process_cluster_dict[cluster_name] = [file_name] # adds the filename to the new cluster key


    print(process_cluster_dict)


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="Aligns xyz coordinates.")
    parser.add_argument('-s', "--sum_file", help="The summary file from hartree.",
                        default=None)

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
    except SystemExit as e:
        if e.message == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR

    return args, GOOD_RET


def main(argv=None):
    # type: (object) -> object
    """
    Runs the main program
    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret
    try:
        print("\nThe following Hartree file has been found: {}\n".format(args.sum_file))

        hartree_dict, pucker_filename_dict = hartree_sum_pucker_cluster(args.sum_file, print_status='off')
        test_clusters(pucker_filename_dict)
    except IOError as e:
        warning(e)
        return IO_ERROR
    except InvalidDataError as e:
        warning(e)
        return INVALID_DATA

    # print_xyz_coords(center_xyz1, atoms1, 'xyz_coords_all-align_1e.xyz')
    # print_xyz_coords(center_ring_all_xyz1, atoms1, 'xyz_coords_ring-align_1e.xyz')

    # print_xyz_coords(center_xyz2, atoms2, 'xyz_coords_all-align_1c4.xyz')
    # print_xyz_coords(center_ring_all_xyz2, atoms2, 'xyz_coords_ring-align_1c4.xyz')

    #
    # print("\n The rmsd without aligning and rotating the structures is {}\n".format(rmsd(center_xyz1, center_xyz2)))
    #
    # print("\n The rmsd from the Kabsch method is: {}\n".format(kabsch_algorithm(center_xyz1, xyz_coords2)))
    #
    # print("\n\n\n")

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
