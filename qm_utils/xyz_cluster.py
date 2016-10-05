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
import csv
import numpy as np
# TODO check error message that I am receiving only when running on my computer
from pip.utils import splitext

from qm_common import GOOD_RET, INVALID_DATA, warning, InvalidDataError, IO_ERROR, INPUT_ERROR, list_to_file, read_csv_to_dict, \
    create_out_fname, list_to_dict

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
num_atoms_ring = 6
ACCEPT_AS_TRUE = ['T', 't', 'true', 'TRUE', 'True']
HARTREE_TO_KCALMOL = 627.5095
STRUCTURE_COMPARE_TOL = 5.0

# Hartree field headers
FILE_NAME = 'File Name'
PUCKER = 'Pucker'
ENERGY_GIBBS = 'G298 (Hartrees)'
ENERGY_ELECTRONIC = 'Energy (A.U.)'

def get_coordinates_xyz(filename, xyz_dir):
    """This function is designed to upload xyz coordinates from .xyz files. The .xyz file format should contain the
    number of atoms on the first line followed by the filename on the second line. After the second line, the lines
    should be organized as followed:

    atom number         x coordinate        y coordinate        z coordinate

    This function stores the the xyz coordinates along with the atom numbers.

    :param filename:
    :param xyz_dir:
    @return: A list of coordinates associated with the atom numbers and a list of lists containing the xyz coordinates
        for the atoms.
    """
    xyz_file_path = create_out_fname(filename, base_dir=xyz_dir, ext='.xyz')

    f = open(xyz_file_path, mode='r')

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


def compare_rmsd_xyz(input_file1, input_file2, xyz_dir, print_option='off'):
    """ calculates the rmsd both using the standard method and rotating the structures

    :param input_file1: xyz coordinates for the first molecular structure
    :param input_file2: xyz coordinates for the second molecular structure
    :param xyz_dir:
    :param print_opt:
    :return: returns all of the
    """

    # TODO add a function that takes an imput log file but accesses the xyz
    n_atoms1, atoms1, xyz_coords1, atoms_ring_order1, xyz_coords_ring1 = get_coordinates_xyz(input_file1, xyz_dir)
    n_atoms2, atoms2, xyz_coords2, atoms_ring_order2, xyz_coords_ring2 = get_coordinates_xyz(input_file2, xyz_dir)

    if n_atoms1 != n_atoms2:
        exit("Error in the number of atoms! The number of atoms doesn't match!")

    check_value = check_ring_ordering(atoms_ring_order1, atoms_ring_order2)

    if check_value != 0:
        exit("The atoms alignment isn't the same!")

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

    return rmsd_kabsch, center_ring_all_xyz1, center_ring_all_xyz2


def hartree_sum_pucker_cluster(sum_file, print_status='off'):
    """
    Reads the hartree output file and creates a dictionary of all hartree output and clusters based on pucker
    :param sum_file: name of hartree output file
    :return: lists of dicts for each row of hartree, and a dictionary of puckers (keys) and file_names
    """
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


def test_clusters(pucker_filename_dict, xyz_dir, ok_tol=DEF_TOL_CLUSTER,print_option ='off'):
    """
    What I do
    :param pucker_filename_dict:
    :param xyz_dir:
    :return:
    """
    process_cluster_dict = {}
    for pucker, file_list in pucker_filename_dict.items():
        pucker_cluster = 0 # initial pucker list count is 0
        cluster_name = pucker + "_" + str(pucker_cluster) # creates new name for cluster key
        process_cluster_dict[cluster_name] = [file_list[0]]  # adds new cluster key and first file into items of key
        raw_cluster_len = len(file_list) # calculates the length of the file list (how many files in hartree clustering)

        for file_id in range(1, raw_cluster_len): # looks at all the files in the initial clustering
            # looks at a specific filename
            file_name = file_list[file_id]
            not_assigned = True

            for assigned_cluster_name in process_cluster_dict:
                # calculates the rmsd by rotating and translating the rings so that they align properly
                rmsd_kabsch, ctr_ring_all_xyz1, ctr_ring_all_xyz2 = compare_rmsd_xyz(file_name,
                                                                                     process_cluster_dict[assigned_cluster_name][0],
                                                                                     xyz_dir)

                if rmsd_kabsch < ok_tol:
                    # add the file to the current key
                    process_cluster_dict[assigned_cluster_name].append(file_name)
                    not_assigned = False
                    break
            if not_assigned:
                # say the criteria isn't met so another key is needed
                # creates the new name for cluster key
                pucker_cluster += 1
                cluster_name = pucker + "_" + str(pucker_cluster)
                # adds the filename to the new cluster key
                process_cluster_dict[cluster_name] = [file_name]

    if print_option != 'off':
        for cluster_key, cluster_values in process_cluster_dict.items():
            print("Cluster Key: {} Cluster Files: {}".format(cluster_key,cluster_values))
    return process_cluster_dict


def read_clustered_keys_in_hartree(process_cluster_dict, hartree_dict):

    # TODO need to still look at the files and make it so that it read multiple files and compares
    low_e_per_cluster = []

    for cluster_keys, clusters in process_cluster_dict.items():

        num_files_in_cluster = len(clusters)
        if num_files_in_cluster == 1:
            low_e_per_cluster.append(clusters[0])
        elif num_files_in_cluster > 1:

            cluster1_filename = clusters[0]
            cluster2_filename = clusters[1]

            for num_cluster in range(0, num_files_in_cluster):

                #print("First file: {}".format(cluster1_filename))
                #print("Second file: {}".format(cluster2_filename))

                for row in hartree_dict:
                    file_name = row[FILE_NAME]

                    if file_name == cluster1_filename:
                        cluster1_energy = float(row[ENERGY_ELECTRONIC])*HARTREE_TO_KCALMOL
                        #print('found the first file')
                    elif file_name == cluster2_filename:
                        cluster2_energy = float(row[ENERGY_ELECTRONIC])*HARTREE_TO_KCALMOL
                        #print('found the second file')

        #            if abs(cluster1_energy-cluster2_energy) < STRUCTURE_COMPARE_TOL:
        #                low_e_per_cluster.append(cluster1_filename)
        #                low_e_per_cluster.append(cluster2_filename)

                if cluster1_energy > cluster2_energy:
                #elif cluster1_energy > cluster2_energy:
                    #print('Number 1 Wins!')
                    low_energy_cluster_filename = cluster2_filename
                    low_e_per_cluster.append(low_energy_cluster_filename)
                    cluster1_filename = cluster2_filename
                    cluster2_filename = clusters[num_cluster]
                elif cluster1_energy < cluster2_energy:
                    #print('Number 2 Wins!')
                    low_energy_cluster_filename = cluster1_filename
                    low_e_per_cluster.append(low_energy_cluster_filename)
                    cluster1_filename = cluster1_filename
                    cluster2_filename = clusters[num_cluster]

        else:
            print("There is something seriously wrong if your code....")


    print(low_e_per_cluster)
    print(len(low_e_per_cluster))
    print(len(cluster_keys))
    # RIGHT NOW.... if low_e_per_cluster == clsuter_keys then I know that my code is working properly!
    return

def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="Aligns xyz coordinates.")
    parser.add_argument('-d', "--dir_xyz", help="The directory where the xyz files can be found. The default is the"
                                                "directory where the Hartree summary file can be found.",
                        default=None)
    parser.add_argument('-s', "--sum_file", help="The summary file from hartree.",
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


def dict_to_csv_writer(dict_to_write, out_filename, xyz_dir):

    correct_filename = os.path.join(xyz_dir,out_filename)

    with open (correct_filename, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_to_write.items():
            writer.writerow([key, value])





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
        hartree_list, pucker_filename_dict = hartree_sum_pucker_cluster(args.sum_file)
        hartree_dict = list_to_dict(hartree_list, FILE_NAME)
        process_cluster_dict = test_clusters(pucker_filename_dict, args.dir_xyz, print_option = 'off')
        out_filename = os.path.join(args.dir_xyz,'oxane_cont-clustered_B3LYP.csv')
#        dict_to_csv_writer(process_cluster_dict, out_filename,args.dir_xyz)
        read_clustered_keys_in_hartree(process_cluster_dict,hartree_dict)
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
