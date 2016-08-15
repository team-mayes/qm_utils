#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reads a pdb (protein data bank") file and creates gaussian input files and files to compute Cremer-Pople parameters.
"""

from __future__ import print_function
import argparse
import os
import sys
import numpy as np
from qm_common import GOOD_RET, INVALID_DATA, warning, find_files_by_dir, create_out_fname, list_to_file, process_cfg

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'hbmayes'

# Constants #

# Defaults

DEF_FILE_PAT = '*pdb'

# PDB file info
PDB_LINE_TYPE_LAST_CHAR = 'pdb_line_type_last_char'
PDB_ATOM_NUM_LAST_CHAR = 'pdb_atom_num_last_char'
PDB_ATOM_TYPE_LAST_CHAR = 'pdb_atom_type_last_char'
PDB_RES_TYPE_LAST_CHAR = 'pdb_res_type_last_char'
PDB_MOL_NUM_LAST_CHAR = 'pdb_mol_num_last_char'
PDB_X_LAST_CHAR = 'pdb_x_last_char'
PDB_Y_LAST_CHAR = 'pdb_y_last_char'
PDB_Z_LAST_CHAR = 'pdb_z_last_char'
PDB_LAST_T_CHAR = 'pdb_last_temp_char'
PDB_LAST_ELEM_CHAR = 'pdb_last_element_char'
RING_ATOMS = 'ring_atom_nums'
RING_ATOM_TYPES = 'ring_atom_types'

# Config File Sections
MAIN_SEC = 'main'

# Defaults
DEF_CFG_FILE = 'read_pdb.ini'
DEF_ELEM_DICT_FILE = os.path.join(os.path.dirname(__file__), 'cfg', 'charmm36_atoms_elements.txt')
DEF_CFG_VALS = {RING_ATOMS: [6, 1, 2, 3, 4, 5],
                RING_ATOM_TYPES: ['O', 'C', 'C', 'C', 'C', 'C'],
                PDB_LINE_TYPE_LAST_CHAR: 6,
                PDB_ATOM_NUM_LAST_CHAR: 11,
                PDB_ATOM_TYPE_LAST_CHAR: 17,
                PDB_RES_TYPE_LAST_CHAR: 22,
                PDB_MOL_NUM_LAST_CHAR: 28,
                PDB_X_LAST_CHAR: 38,
                PDB_Y_LAST_CHAR: 46,
                PDB_Z_LAST_CHAR: 54,
                PDB_LAST_T_CHAR: 76,
                PDB_LAST_ELEM_CHAR: 78,
                }
REQ_KEYS = {}


def read_cfg(f_loc, cfg_proc=process_cfg):
    """
    Reads the given configuration file, returning a dict with the converted values supplemented by default values.

    :param f_loc: The location of the file to read.
    :param cfg_proc: The processor to use for the raw configuration values.  Uses default values when the raw
        value is missing.
    :return: A dict of the processed configuration file's data.
    """
    config = ConfigParser()
    good_files = config.read(f_loc)
    if not good_files:
        raise IOError('Could not read file: {}'.format(f_loc))
    main_proc = cfg_proc(dict(config.items(MAIN_SEC)), DEF_CFG_VALS, REQ_KEYS)

    return main_proc


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="Reads in pdb ('structure-data file') files and creates gaussian "
                                                 "input files and an input file to compute Cremer-Pople parameters.")
    parser.add_argument("-c", "--config", help="The location of the configuration file in ini format. "
                                               "The default file name is {}, located in the "
                                               "base directory where the program as run.".format(DEF_CFG_FILE),
                        default=DEF_CFG_FILE, type=read_cfg)
    parser.add_argument("-d", "--base_dir", help="The starting point for a summary file search "
                                                 "(defaults to current directory)",
                        default=os.getcwd())
    parser.add_argument('-o', "--out_dir", help="The output directory (defaults to the same directory as the "
                                                "template file)",
                        default=None)
    parser.add_argument('-f', "--file_pattern", help="The file pattern to search for to identify input sdf files"
                                                     "(defaults to '{}')".format(DEF_FILE_PAT),
                        default=DEF_FILE_PAT)

    try:
        args = parser.parse_args(argv)
    except (IOError, SystemExit) as e:
        warning(e)
        parser.print_help()
        return [], INVALID_DATA

    return args, GOOD_RET


def process_file(file_path, out_dir, cp_data, cfg):
    """
    Read sdf (Spartan file) information and create alternate format
    :param file_path: path of file to be read
    :param out_dir: output directory
    :param cp_data: data to be gathered to calculate Cremer-Pople Parameters
    :param cfg: configuration for this run
    :return:
    """

    gaussian_keywords = '#Put Keywords Here, check Charge and Multiplicity\n'
    file_name = os.path.basename(file_path)
    descrip = "From '{}'\n".format(file_name)
    charge = 0
    mult = 1
    com_data = [[gaussian_keywords], [descrip], ['{}  {}'.format(charge, mult)], ]
    raw_cp_data = file_name + ' '

    ring_xyz = [[np.nan] * 3] * 6

    with open(file_path, 'r') as f:
        atom_count = 0
        for line in f:
            line = line.strip()
            line_len = len(line)
            if line_len == 0:
                continue
            line_head = line[:cfg[PDB_LINE_TYPE_LAST_CHAR]]
            # head_content to contain Everything before 'Atoms' section
            # also capture the number of atoms
            if line_head == 'REMARK' or line_head == 'CRYST1':
                continue

            # atoms_content to contain everything but the xyz
            elif line_head == 'ATOM  ' or line_head == 'HETATM':
                # PDB may have ***** after atom_count 99999, or not numbered. Thus, I'm renumbering. Otherwise:
                # atom_num = line[cfg[PDB_LINE_TYPE_LAST_CHAR]:cfg[PDB_ATOM_NUM_LAST_CHAR]]
                # For renumbering, making sure prints in the correct format, including num of characters:
                atom_count += 1

                xyz = [float(num) for num in (line[cfg[PDB_MOL_NUM_LAST_CHAR]:cfg[PDB_X_LAST_CHAR]],
                                              line[cfg[PDB_X_LAST_CHAR]:cfg[PDB_Y_LAST_CHAR]],
                                              line[cfg[PDB_Y_LAST_CHAR]:cfg[PDB_Z_LAST_CHAR]])]

                element = line[cfg[PDB_LAST_T_CHAR]:cfg[PDB_LAST_ELEM_CHAR]]

                if atom_count in cfg[RING_ATOMS]:
                    ring_index = cfg[RING_ATOMS].index(atom_count)
                    ring_xyz[ring_index] = xyz
                    expected_atom_type = cfg[RING_ATOM_TYPES][ring_index]
                    if element.strip() != expected_atom_type:
                        warning("Expected atom {} to have type '{}'. Found '{}'".format(atom_count,
                                                                                        expected_atom_type,
                                                                                        element.strip()))
                com_data.append(['{:3}'.format(element)] + ['{:15.5f}'.format(num) for num in xyz])

    for ring_atom in ring_xyz:
        raw_cp_data += ' '.join(['{:6.3f} '.format(num) for num in ring_atom])
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

    # Read template and dump files
    cfg = args.config

    try:
        found_files = find_files_by_dir(args.base_dir, args.file_pattern)
        cp_data = []
        print("Searching in {} directories for files to process".format(len(found_files)))
        # noinspection PyCompatibility
        for f_dir, files in found_files.iteritems():
            for file_path in ([os.path.join(f_dir, tgt) for tgt in files]):
                process_file(file_path, args.out_dir, cp_data, cfg)
        cp_file = create_out_fname('cp.inp', base_dir=args.out_dir)
        list_to_file(cp_data, cp_file)
    finally:
        pass

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
