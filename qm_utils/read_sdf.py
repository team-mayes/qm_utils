#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reads an sdf ("structure-data file") file and creates gaussian input files and files to compute Cremer-Pople parameters.
"""

from __future__ import print_function
import argparse
import os
import sys
from qm_common import GOOD_RET, INVALID_DATA, warning, find_files_by_dir, create_out_fname, list_to_file

__author__ = 'hbmayes'

# Constants #

# Defaults

DEF_FILE_PAT = '*sdf'


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="Reads in sdf ('structure-data file') files and creates gaussian "
                                                 "input files and files to compute Cremer-Pople parameters.")
    parser.add_argument("-d", "--base_dir", help="The starting point for a summary file search "
                                                 "(defaults to current directory)",
                        default=os.getcwd())
    parser.add_argument('-o', "--out_dir", help="The output directory (defaults to the same directory as the "
                                                "template file)",
                        default=None)
    # parser.add_argument('-p', "--pdb_file", help="Flag to produce a pdb file.",
    #                     action='store_true')
    parser.add_argument('-f', "--file_pattern", help="The file pattern to search for to identify input sdf files"
                                                     "(defaults to '{}')".format(DEF_FILE_PAT),
                        default=DEF_FILE_PAT)

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        warning(e)
        return [], INVALID_DATA

    return args, GOOD_RET


def process_file(file_path, out_dir, cp_data):
    """
    Read sdf (Spartan file) information and create alternate format
    :param file_path: path of file to be read
    :param out_dir: output directory
    :param cp_data: data to be gathered to calculate Cremer-Pople Parameters
    :return:
    """
    with open(file_path, 'r') as f:
        sdf_data = f.readlines()

    gaussian_keywords = '#Put Keywords Here, check Charge and Multiplicity\n'

    file_name = os.path.basename(file_path)
    descrip = "From '{}': {}\n".format(file_name, sdf_data[0].strip())

    charge = 0
    mult = 1

    sdf_line_3 = sdf_data[3].strip().split()

    com_data = [[gaussian_keywords], [descrip], ['{}  {}'.format(charge, mult)], ]
    raw_cp_data = file_name + ' '

    num_atoms = int(sdf_line_3[0])
    num_bonds = int(sdf_line_3[1])

    ring_xyz = []
    for row in range(4, 4+num_atoms):
        atom_row = sdf_data[row].strip().split()
        xyz = [float(num) for num in atom_row[0:3]]
        element = atom_row[3]
        if row < 10:
            if row == 9:
                if element != 'O':
                    warning("Expected the 6th atom to be an oxygen. Found '{}'".format(element))
            elif element != 'C':
                warning("Expected the first five atoms to be carbons. Found '{}' for atom {}".format(element, row - 3))
            ring_xyz.append(xyz)
        com_data.append(['{:3}'.format(element)] + ['{:15.5f}'.format(num) for num in xyz])

    for ring_atom in [5, 0, 1, 2, 3, 4]:
        raw_cp_data += ' '.join(['{:6.3f} '.format(num) for num in ring_xyz[ring_atom]])
    # make sure to add empty line at end of com file
    com_data.append([])

    cp_data.append(raw_cp_data)
    com_file = create_out_fname(file_path, base_dir=out_dir, ext='.com')
    list_to_file(com_data, com_file)


def main(argv=None):
    """
    Runs the main program
    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET:
        return ret

    try:
        found_files = find_files_by_dir(args.base_dir, args.file_pattern)
        cp_data = []
        print("Found {} files to process".format(len(found_files)))
        for f_dir, files in found_files.iteritems():
            for file_path in ([os.path.join(f_dir, tgt) for tgt in files]):
                process_file(file_path, args.out_dir, cp_data)
        cp_file = create_out_fname('cp.inp', base_dir=args.out_dir)
        list_to_file(cp_data, cp_file)
    finally:
        pass

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
