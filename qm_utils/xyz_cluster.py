#!/usr/bin/env python

"""
(insert a description about the code here)

The purpose of this python script to align xyz coordinate files so that the structures
can be clustered.

"""

# Libraries
from __future__ import print_function
from sys import argv
from qm_common import find_files_by_dir, list_to_file
import numpy as np
import re

__author__ = 'SPVicchio'


# Constants #

TOL_centroid = [0.001, 0.001, 0.001]
num_atoms_ring = 6

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
            lines_read_ring += 1

        lines_read += 1
    f.close()

    xyz_atoms = np.array(xyz_atoms)

    return num_atoms, xyz_atoms, xyz_coords, xyz_coords_ring

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
    """ Calculates the centroid of the xyz coordinates based on all the atoms. Once the cetroid is found, then the
    xyz corodinates are translated so that the centroid is at the origin.

    :param xyz_coords: the xyz coordinates for the molecular structure (centroid not aligned)
    :return: outputs the xyz coordiantes for the molecular structure with the centroid located at the origin.
    """

    centroid_xyz = sum(xyz_coords) / len(xyz_coords)

    xyz_coords_translate = np.array([xyz_coords - centroid_xyz])

    centroid_xyz_check = sum(xyz_coords_translate) / len(xyz_coords_translate)

    if centroid_xyz_check.all > TOL_centroid:
        exit("\nCould not properly align the centroid to the origin\n.")

    xyz_coords_translate = np.array(np.mat(xyz_coords_translate))
    return xyz_coords_translate

def translate_centroid_ring(xyz_coords,xyz_atoms):
    """This script is designed to calculate the centroid of the xyz coordinates based on only the atoms in the ring.
    Once the centroid of the ring is found, then the all xyz coordinates are translated so that the centroid of the ring
    is at the origin.

    :param xyz_coords: the xyz coordinates for the molecular structure (centroid not aligned)
    :param xyz_atoms: the atoms of the ring
    :return: outputs the xyz coordinates for the molecular structure with the ring centroid located at the origin.
    """

    #for

    # for len(xyz_coords)

    centroid_xyz = sum(xyz_coords) / len(xyz_coords)

    xyz_coords_translate = [xyz_coords - centroid_xyz]

    centroid_xyz_check = sum(xyz_coords_translate) / len(xyz_coords_translate)

    if centroid_xyz_check.all > TOL_centroid:
        exit("\nCould not properly align the centroid to the origin\n.")

    return xyz_coords_translate

def rmsd(V, W):
    """Calculates the root-mean-square deviation from two sets of vectors xyz_1 and xyz_2 (both of which are xyz
    coordintes for different molecules)

    :param V: input xyz coordinates of a molecule
    :param W: input xyz coordinates of a molecule
    :return: the root-mean-square deviation for the two molecules
    """

    D = len(V[0]) # number of dimensions in system
    N = len(W) # number of atoms in system
    rmsd = 0.0 # initial value of the rmsd

    for v, w in zip(V, W):
        rmsd += sum([(v[i]-w[i])**2.0 for i in range(D)])

    rmsd_value = np.sqrt(rmsd/N)

    return rmsd_value, D, N

def kabsch_algorithm(xyz_coords1, xyz_coords2):

    # calculate the covariance matrix between the two sets of xyz coordinates
    covariance_matrix = np.dot(np.transpose(xyz_coords1),xyz_coords2)

    # calculate the singular value decomposition (svd) of the covariance matrix using numpy linear algebra
    V, S, W = np.linalg.svd(covariance_matrix)

    # check if the systems needs to be rotated to ensure a right-handed coordinate system
    direction = (np.linalg.det(V)*np.linalg.det(W))

    if direction < 0.0:
        S[-1] = -S[-1]
        V[:,-1] = - V[:,-1]

    rotation_matrix = np.dot(V,W)

    rotated_xyz_coords1 = np.dot(xyz_coords1,rotation_matrix)

    kabsch_rsmd = rmsd(rotated_xyz_coords1,xyz_coords2)

    list_to_file(['18'], 'test_me.txt')
    list_to_file(['testing'], 'test_me.txt', mode='a')
    list_to_file(rotated_xyz_coords1, 'test_me.txt', mode='a')


    return kabsch_rsmd



##### Loading Files
script, input_file1, input_file2 = argv

n_atoms1, atoms1, xyz_coords1, xyz_coords_ring1 = get_coordinates_xyz(input_file1)
n_atoms2, atoms2, xyz_coords2, xyz_coords_ring2 = get_coordinates_xyz(input_file2)

to_print = ['18', 'testing']
to_print2 = list(to_print)

for line_id in range(len(atoms1)):
    print("line_id", line_id)
    to_print.append([atoms1[line_id]] + xyz_coords1[line_id].tolist())


for atom_type, atom_xyz in zip(atoms1, xyz_coords1):
    to_print2.append([atom_type] + atom_xyz.tolist())

list_to_file(to_print2, 'test3.txt')

if n_atoms1 != n_atoms2:
    exit("Error in the number of atoms! The number of atoms doesn't match!")
#####

center_xyz1 = translate_centroid_all(xyz_coords1)
center_xyz2 = translate_centroid_all(xyz_coords2)

print("\n The rmsd without aligning and rotating the structures is {}\n".format(rmsd(center_xyz1,center_xyz2)))

print("\n The rmsd from the Kabsch method is: {}\n".format(kabsch_algorithm(center_xyz1,xyz_coords2)))

#print("{},\n\n{}".format(center_xyz1,center_xyz2))

#print("{}".format(kabsch_algorithm(center_xyz1,center_xyz2)))

#print("{}".format(xyz_coords1[2,:]))


# c1 = centroid(xyz_coords1)
# c2 = centroid(xyz_coords2)

# print("{},{}".format(c1, c2))

# print("This is the translated verison\n {}".format([xyz_coords1 - c1]))
# print("\n This is the untranslated verison\n {}".format(xyz_coords1))

# print("{}".format(centroid(xyz_coords1 - c1)))

#A = translate_centroid_all(xyz_coords1)

#print("{}".format(translate_centroid_all(xyz_coords1)))
