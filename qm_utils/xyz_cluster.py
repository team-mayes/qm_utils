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

        # Use the number of atoms to not read beyond the end of the file
        for line in f:

            if lines_read == n_atoms:
                break

            print f.readline()

            #if len(numbers) == 4:

            #    V.append(np.array(numbers))
            #    atoms.append(atom)

            lines_read += 1

            print lines_read

    f.close()
    V = np.array(V)
    return atoms, V


from sys import argv

script, input_file = argv

get_coordinates_xyz(input_file)


