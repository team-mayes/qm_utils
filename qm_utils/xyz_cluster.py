#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
(insert a description about the code here)

The purpose of this python script to align xyz coordinate files so that the structures
can be clustered.

"""

from sys import argv

__author__ = 'SPVicchio'

# Libraries
from sys import argv
import numpy as np
import re

# Constants #


# Defaults #


# Functions #

def get_coordinates_xyz(filename):
    """This function is designed to upload xyz coordinates from .xyz files. The .xyz file format should contain the
    number of atoms on the first line followed by the filename on the second line. After the second line, the lines
    should be organized as followed:

    atom number         x coordinate        y coordinate        z coordinate

    This function stores the the xyz information

    :param filename:
    :return:
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
    xyz_coords = np.full((num_atoms, 3), np.nan)
    for line in f:

        if lines_read == num_atoms:
            break

        atom, coor_x, coor_y, coor_z = line.split()
        numbers = map(float, [coor_x, coor_y, coor_z])

        if len(numbers) == 3:
            xyz_coords[lines_read] = numbers
            xyz_atoms.append(atom)

        lines_read += 1


    f.close()
    xyz_atoms = np.array(xyz_atoms)
    return num_atoms, xyz_atoms, xyz_coords


script, input_file = argv

n_atoms, atoms, V = get_coordinates_xyz(input_file)

print "Your molecule currently has %d." % n_atoms
print "\nThe atom ordering is:\n %r" % atoms
print "\nThe XYZ coordinates are:\n %r" % V
