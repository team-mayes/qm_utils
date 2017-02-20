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
    file_info = f.readlines()

    xyz_atoms = []
    atoms_ring_order = [None] * 6
    total_num_atoms = 0
    atom_num = 0
    index = 0

    # Read the first line to obtain the number of atoms read

    for line in file_info:
        index += 1
        coord_floats = []
        if index == 1:
            try:
                total_num_atoms = int(line)
            except ValueError:
                exit("Could not obtain the number of atoms in the .xyz file.")
        elif index == 2:
            xyz_coords_ring = np.full((num_atoms_ring, 3), np.nan)
            # creates an array that will be populated later with xyz coordinates
            xyz_coords = np.full((total_num_atoms, 3), np.nan)
        else:
            if atom_num == total_num_atoms:
                break
            atom_type, coor_x, coor_y, coor_z = line.split()
            # map to take all of the coordinates and turn them into xyz_coords using float option
            coords_list = list([coor_x, coor_y, coor_z])
            for item in coords_list:
                coord_floats.append(float(item))

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
