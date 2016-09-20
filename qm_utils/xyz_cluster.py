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

import numpy as np
import re

# Constants #


# Defaults #


# Functions #

def get_coordinates_xyz(filename):

    f = open(filename, 'r')
    V = []
    atoms = []
    n_atoms = 0
    lines_read = 0

    # Read the first line to obtain the number of atoms read
    try:
        n_atoms = int(f.next())
    except ValueError:
        exit("Could not obtain the number of atoms in the .xyz file.")

        # Skip the title line
    f.next()

    for line in f:

        atom, coor_x, coor_y, coor_z = line.split()

        numbers = coor_x, coor_y, coor_z

        if len(numbers) == 3:

            V.append(numbers)
            atoms.append(atom)

    f.close()
    return n_atoms, atoms, V


from sys import argv

script, input_file = argv

n_atoms, atoms, V = get_coordinates_xyz(input_file)

print "Your molecule currently has %d." % n_atoms
print "\nThe atom ordering is:\n %r" % atoms
print "\nThe XYZ coordinates are:\n %r" % V
