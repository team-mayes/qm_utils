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
import six

from qm_common import (GOOD_RET, INVALID_DATA, warning, find_files_by_dir, create_out_fname, list_to_file, process_cfg,
                       InvalidDataError, IO_ERROR, INPUT_ERROR, prep_string)

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'hbmayes'

# Constants #

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

TOT_CHARGE = 'total_charge'
TOT_MULT = 'multiplicity'
GAUSS_KEYWORDS = 'gaussian_keywords'
GAUSS_DESCRIP = 'description'
GAUSS_FOOTER = 'gaussian_footer'

# Config File Sections
MAIN_SEC = 'main'

# Defaults
DEF_FILE_PAT = '*pdb'
DEF_CFG_FILE = 'read_pdb.ini'
DEF_GAUSS_KEYWORDS = '# Put Keywords Here, check Charge and Multiplicity\n'
DEF_ELEM_DICT_FILE = os.path.join(os.path.dirname(__file__), 'cfg', 'charmm36_atoms_elements.txt')
DEF_CFG_VALS = {RING_ATOMS: [6, 1, 2, 3, 4, 5],
                RING_ATOM_TYPES: ['O', 'C', 'C', 'C', 'C', 'C'],
                GAUSS_KEYWORDS: None,
                GAUSS_DESCRIP: None,
                TOT_CHARGE: 0,
                TOT_MULT: 1,
                GAUSS_FOOTER: '',
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

OUT_DIR = 'out_dir'
FILE_TYPE = 'file_type'
PRINT_CP_INPUT = 'print_cp_input'


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
        warning('Will use only default values since not find file: {}'.format(f_loc))
        raw_cfg = {}
    else:
        raw_cfg = dict(config.items(MAIN_SEC))
    main_proc = cfg_proc(raw_cfg, DEF_CFG_VALS, REQ_KEYS)

    return main_proc


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="Reads in a pdb (protein data bank) or sdf (structure-data file) "
                                                 "and creates gaussian input files and an input file to "
                                                 "compute Cremer-Pople parameters.")
    parser.add_argument("-c", "--config", help="The location of the configuration file in ini format. "
                                               "The default file name is {}, located in the "
                                               "base directory where the program as run.".format(DEF_CFG_FILE),
                        default=DEF_CFG_FILE, type=read_cfg)
    parser.add_argument("-d", "--base_dir", help="The starting point for a summary file search "
                                                 "(defaults to the current directory)",
                        default=os.getcwd())
    parser.add_argument('-f', "--file_pattern", help="The file pattern to search for to identify input files"
                                                     "(defaults to '{}')".format(DEF_FILE_PAT),
                        default=DEF_FILE_PAT)
    parser.add_argument('-o', "--out_dir", help="The output directory (defaults to the current directory)",
                        default=os.getcwd())
    parser.add_argument('-s', "--skip_cp", help="Flag to skip creating an input file for cp_params. This action "
                                                "will also be skipped if ring atoms are not found.",
                        action='store_true')

    parser.add_argument('-t', "--file_type", help="The file type (pdb or sdf) of the input file. If not specified, "
                                                  "it is assumed based on the file extension.",
                        default=None)

    args = None
    try:
        args = parser.parse_args(argv)
    except KeyError as e:

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

    if not args.skip_cp:
        if len(args.config[RING_ATOMS]) != 6 or len(args.config[RING_ATOM_TYPES]) != 6:
            warning('To print cp_params input, enter 6 atom numbers and 6 atom types (found {}, and {}, '
                    'respectively).\nProgram will continue but skip creating cp_params input.'
                    ''.format(args.config[RING_ATOMS], args.config[RING_ATOM_TYPES]))
            args.skip_cp = True

    return args, GOOD_RET


def process_file(file_path, cp_data, cfg):
    """
    Read sdf (Spartan file) information and create alternate format
    :param file_path: path of file to be read
    :param cp_data: data to be gathered to calculate Cremer-Pople Parameters
    :param cfg: configuration for this run
    :return: will create a com file for each file read and gather data for CP parameters
    """

    file_name = os.path.basename(file_path)
    if cfg[FILE_TYPE] is None:
        file_type = file_name.split('.')[-1]
    else:
        file_type = cfg[FILE_TYPE]

    # begin building up com file output
    if cfg[GAUSS_KEYWORDS] is None:
        com_data = [[DEF_GAUSS_KEYWORDS]]
    else:
        com_data = [prep_string(cfg[GAUSS_KEYWORDS])]

    if cfg[GAUSS_DESCRIP] is None:
        descrip = "From '{}'\n".format(file_name)
    else:
        descrip = cfg[GAUSS_DESCRIP] + "\n"

    charge = cfg[TOT_CHARGE]
    mult = cfg[TOT_MULT]
    com_data += [[descrip], ['{}  {}'.format(charge, mult)]]

    # prepared list so that values can be entered in the correct location independent of whether read in order
    ring_xyz = [[np.nan] * 3] * 6
    if file_type == 'pdb':
        process_pdb(cfg, com_data, ring_xyz, file_path)
    elif file_type == 'sdf':
        process_sdf(cfg, com_data, ring_xyz, file_path)
    else:
        raise InvalidDataError("Found file format '{}'. This program currently reads only pdb and sdf file formats. "
                               "If the file is one of these types, but has a different extension, specify the type "
                               "with the '-t' command-line argument.".format(file_type))

    # make sure to add empty line before footer
    com_data.append(['\n' + prep_string(cfg[GAUSS_FOOTER])])
    com_file = create_out_fname(file_path, base_dir=cfg[OUT_DIR], ext='.com')
    list_to_file(com_data, com_file)

    if cfg[PRINT_CP_INPUT]:
        # gather header + 18 floats (6*xyz) for each row of cp input
        raw_cp_data = file_name + ' '
        for ring_atom in ring_xyz:
            raw_cp_data += ' '.join(['{:6.3f} '.format(num) for num in ring_atom])
        if 'nan' in raw_cp_data:
            warning('Did not find the expected six ring atoms. '
                    'For cp_params input, skipping file: {}'.format(file_path))
        else:
            cp_data.append(raw_cp_data)


def process_sdf(cfg, com_data, ring_xyz, file_path):
    """
    Reads data from a sdf file needed to create a gaussian input file and a cp_params input file
    :param cfg: configuration for the run
    :param com_data: formatted data being collected to output the gaussian input ('com') file
    :param ring_xyz: a list to store xyz coordinates of the ring atoms
    :param file_path: file being read
    :return: the com_data and ring_xyz files will have data added to them from the PDB file
    """
    with open(file_path, 'r') as f:
        sdf_data = f.readlines()

    sdf_line_3 = sdf_data[3].strip().split()

    num_atoms = int(sdf_line_3[0])
    atom_count = 0
    for row in range(4, 4 + num_atoms):
        atom_count += 1
        atom_row = sdf_data[row].strip().split()
        xyz = [float(num) for num in atom_row[0:3]]
        element = atom_row[3]
        com_data.append(['{:3}'.format(element)] + ['{:15.5f}'.format(num) for num in xyz])
        if cfg[PRINT_CP_INPUT]:
            if atom_count in cfg[RING_ATOMS]:
                ring_index = cfg[RING_ATOMS].index(atom_count)
                ring_xyz[ring_index] = xyz
                expected_atom_type = cfg[RING_ATOM_TYPES][ring_index]
                if element.strip() != expected_atom_type:
                    warning("Expected atom {} to have type '{}'. Found '{}'".format(atom_count,
                                                                                    expected_atom_type,
                                                                                    element.strip()))


def process_pdb(cfg, com_data, ring_xyz, file_path):
    """
    Reads data from a pdb file needed to create a gaussian input file and a cp_params input file
    :param cfg: configuration for the run
    :param com_data: formatted data being collected to output the gaussian input ('com') file
    :param ring_xyz: a list to store xyz coordinates of the ring atoms
    :param file_path: file being read
    :return: the com_data and ring_xyz files will have data added to them from the PDB file
    """
    with open(file_path, 'r') as f:
        atom_count = 0
        for line in f:
            line = line.strip()
            line_len = len(line)
            if line_len == 0:
                continue
            line_head = line[:cfg[PDB_LINE_TYPE_LAST_CHAR]]
            # Only need atom information, so can ignore lines with remark, title, etc.
            if line_head == 'ATOM  ' or line_head == 'HETATM':
                # PDB may have ***** after atom_count 99999, or not numbered. Thus, I'm renumbering
                atom_count += 1

                xyz = [float(num) for num in (line[cfg[PDB_MOL_NUM_LAST_CHAR]:cfg[PDB_X_LAST_CHAR]],
                                              line[cfg[PDB_X_LAST_CHAR]:cfg[PDB_Y_LAST_CHAR]],
                                              line[cfg[PDB_Y_LAST_CHAR]:cfg[PDB_Z_LAST_CHAR]])]

                element = line[cfg[PDB_LAST_T_CHAR]:cfg[PDB_LAST_ELEM_CHAR]]
                com_data.append(['{:3}'.format(element)] + ['{:15.5f}'.format(num) for num in xyz])

                if cfg[PRINT_CP_INPUT]:
                    if atom_count in cfg[RING_ATOMS]:
                        ring_index = cfg[RING_ATOMS].index(atom_count)
                        ring_xyz[ring_index] = xyz
                        expected_atom_type = cfg[RING_ATOM_TYPES][ring_index]
                        if element.strip() != expected_atom_type:
                            warning("Expected atom {} to have type '{}'. Found '{}'".format(atom_count,
                                                                                            expected_atom_type,
                                                                                            element.strip()))


def main(argv=None):
    """
    Runs the main program
    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret

    cfg = args.config
    cfg[FILE_TYPE] = args.file_type
    cfg[OUT_DIR] = args.out_dir
    cfg[PRINT_CP_INPUT] = not args.skip_cp

    try:
        found_files = find_files_by_dir(args.base_dir, args.file_pattern)
        cp_data = []
        for f_dir, files in six.iteritems(found_files):
            for file_path in ([os.path.join(f_dir, tgt) for tgt in files]):
                process_file(file_path, cp_data, cfg)
        if cfg[PRINT_CP_INPUT]:
            if len(cp_data) == 0:
                warning('Did not find any data for cp_params input; a cp_param input file will not be created')
            else:
                cp_file = create_out_fname('cp.inp', base_dir=args.out_dir)
                # because the file search provide a different orders on different machines, sort for consistency
                cp_data.sort()
                list_to_file(cp_data, cp_file)
    except IOError as e:
        warning(e)
        return IO_ERROR
    except InvalidDataError as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
