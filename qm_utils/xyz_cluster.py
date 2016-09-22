#!/usr/bin/env python

"""
(insert a description about the code here)

The purpose of this python script to align xyz coordinate files so that the structures
can be clustered.

"""

# Libraries
from __future__ import print_function
from sys import argv
from qm_common import find_files_by_dir
import numpy as np
import re

__author__ = 'SPVicchio'


# Constants #

TOL_centroid = [0.001, 0.001, 0.001]

# Defaults #


# Functions #

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

    f = open(filename, 'r')

    xyz_atoms = []
    num_atoms = 0
    lines_read = 0

    # Read the first line to obtain the number of atoms read
    try:
        num_atoms = int(f.next())
    except ValueError:
        exit("Could not obtain the number of atoms in the .xyz file.")
    # Skip the title line
    f.next()

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
        lines_read += 1
    f.close()

    xyz_atoms = np.array(xyz_atoms)

    return num_atoms, xyz_atoms, xyz_coords


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


def centroid(X):
    """ Calculates the centroid (geometric center) of the structure. The centroid
    is defined as the mean position of all the points in all of the coordinate directions.

    :param X: xyz coordinates for the given molecular structure
    :return: the xyz coordinates for the centroid of the structure
    """
    centroid_xyz = sum(X) / len(X)
    return centroid_xyz


def translate_centroid_all(xyz_coords):
    """ Calculates the centroid of the XYZ coordinates based on all the atoms. Once the cetroid is found, then the
    xyz corodinates are translated so that the centroid is at the origin.

    :param xyz_coords: the xyz coordinates for the molecular structure (centroid not aligned)
    :return: outputs the xyz coordiantes for the molecular structure with the centroid at zero
    """

    centroid_xyz = sum(xyz_coords) / len(xyz_coords)

    xyz_coords_translate = [xyz_coords - centroid_xyz]

    centroid_xyz_check = sum(xyz_coords_translate) / len(xyz_coords_translate)

    if centroid_xyz_check.all > TOL_centroid:
        exit("\nCould not properly align the centroid to the origin\n.")

    return xyz_coords_translate


script, input_file1, input_file2 = argv

n_atoms1, atoms1, xyz_coords1 = get_coordinates_xyz(input_file1)
n_atoms2, atoms2, xyz_coords2 = get_coordinates_xyz(input_file2)

if n_atoms1 != n_atoms2:
    print("Error in the number of atoms")

# c1 = centroid(xyz_coords1)
# c2 = centroid(xyz_coords2)

# print("{},{}".format(c1, c2))

# print("This is the translated verison\n {}".format([xyz_coords1 - c1]))
# print("\n This is the untranslated verison\n {}".format(xyz_coords1))

# print("{}".format(centroid(xyz_coords1 - c1)))


A = translate_centroid_all(xyz_coords1)

print("{}".format(translate_centroid_all(xyz_coords1)))
