#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
From 2007Hill_Supporting.pdf

cp.py
Usage:
./cp.py file.dat
This is a script that reads a list of atomic coordinates
for several 6-member rings and outputs Cremer-Pople
puckering coordinates.
Input format:
name1 18x(%6.3f )
name2 18x(%6.3f )
name3 18x(%6.3f )
....
Each line contains a name and Cartesian coordinates
for a unique 6-member ring. The order for the atomic
coordinates is:
O5(or C6) C1 C2 C3 C4 C5
with x, y, and z coordinates specified for each atom
in that order
"""

from __future__ import print_function

import csv
import os

import numpy as np
import argparse
import sys
import math
from qm_common import GOOD_RET, INVALID_DATA, warning, read_csv_to_dict, create_out_fname, write_csv

__author__ = 'hbmayes'

# Constants #


# Defaults

PUCKER_DICT_FILE = os.path.join(os.path.dirname(__file__), 'cfg', 'pucker_degrees.csv')
NAME_KEY = 'name'
PHI_KEY = 'phi'
THETA_KEY = 'theta'
Q_KEY = 'q'
PUCKER_KEY = 'closest_pucker'
OUT_KEYS = [NAME_KEY, PHI_KEY, THETA_KEY, Q_KEY, PUCKER_KEY]


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description='Reads in space-separated lists of atomic coordinates for 6-membered '
                                                 'rings and outputs Cremer-Pople puckering coordinates. The input '
                                                 'format is: \n'
                                                 'name1 18x(float )\n'
                                                 'name2 18x(float )\n'
                                                 'name3 18x(float ) ....\n'
                                                 'where each line contains a name and Cartesian coordinates for a '
                                                 'unique 6-member ring. The order for the atomic coordinates is: O5 '
                                                 '(or C6) C1 C2 C3 C4 C5 with x, y, and z coordinates specified for'
                                                 'each atom in that order.')
    parser.add_argument("file", help="The input file to process")
    parser.add_argument('-o', "--out_file", help="The output file name (defaults to input file name with the '.out' "
                                                 "extension)",
                        default=None)
    parser.add_argument('-d', "--out_dir", help="The output directory (defaults to the same directory as the "
                                                "input file)",
                        default=None)
    parser.add_argument('-p', "--pucker_dict", help="A csv file specifying the Cremer-Pople puckering parameters (in "
                                                    "degrees) for each IUPAC specified puckering designation. The"
                                                    "default file is: {}".format(PUCKER_DICT_FILE),
                        default=PUCKER_DICT_FILE)

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        warning(e)
        return [], INVALID_DATA

    return args, GOOD_RET


def to_float_list(a):
    b = []
    for i in a:
        b.append(float(i))
    return b


def find_closest_pucker(phi1, theta1, pucker_dict):
    closest_pucker = 'nan'
    closest_dist = np.inf
    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)

    for pucker in pucker_dict:
        sin_phi2 = math.sin(pucker_dict[pucker][0])
        cos_phi2 = math.cos(pucker_dict[pucker][0])
        theta2 = pucker_dict[pucker][1]
        cos_angle = (sin_phi1 * sin_phi2 * math.cos(theta1 - theta2) + cos_phi1 * cos_phi2)
        current_dist = math.acos(cos_angle)
        if current_dist < closest_dist:
            closest_dist = current_dist
            closest_pucker = pucker

    return closest_pucker


def process_file(input_file, pucker_dict):
    """
    :param input_file: path of file to be read
    :param pucker_dict: dictionary of pucker_ids
    :return:
    """

    pucker_results = []
    atoms = np.zeros((6, 3), dtype='float64')

    with open(input_file, 'r') as f:

        for line in f:
            s_line = line.split()
            header = s_line[0]
            for i in range(1, 19, 3):
                atoms[(i - 1) / 3] = to_float_list(s_line[i:i + 3])
            center = np.sum(atoms) / 6.
            atoms -= center

            r1a = np.zeros(3, dtype='float64')
            r2a = np.zeros(3, dtype='float64')
            for j, i in enumerate(atoms[0:6]):
                r1a += i * math.sin(2.*math.pi * j / 6.)
                r2a += i * math.cos(2.*math.pi * j / 6.)
            n = np.cross(r1a, r2a)
            n /= np.linalg.norm(n)

            z = np.dot(atoms, n)
            q2cosphi = 0.
            q2sinphi = 0.
            q1cosphi = 0.
            q1sinphi = 0.
            q3 = 0.
            big_q = 0.
            sqrt_2 = math.sqrt(2.)
            inv_sqrt_6 = math.sqrt(1. / 6.)
            for j, i in enumerate(z):
                q2cosphi += i * math.cos(2.*math.pi * 2.*j / 6.)
                q2sinphi -= i * math.sin(2.*math.pi * 2.*j / 6.)
                q1cosphi += i * math.cos(2.*math.pi * j / 6.)
                q1sinphi -= i * math.sin(2.*math.pi * j / 6.)
                q3 += i * math.cos(j * math.pi)
                big_q += i * i
            q2cosphi *= sqrt_2 * inv_sqrt_6
            q2sinphi *= sqrt_2 * inv_sqrt_6
            q3 *= inv_sqrt_6
            q2 = math.sqrt(q2cosphi * q2cosphi + q2sinphi * q2sinphi)
            # q1 = math.sqrt(q1cosphi * q1cosphi + q1sinphi * q1sinphi)
            big_q = math.sqrt(big_q)

            if q2cosphi > 0.:
                if q2sinphi > 0.:
                    phi = math.degrees(math.atan(q2sinphi / q2cosphi))
                else:
                    phi = 360. - abs(math.degrees(math.atan(q2sinphi / q2cosphi)))
            else:
                if q2sinphi > 0.:
                    phi = 180. - abs(math.degrees(math.atan(q2sinphi / q2cosphi)))
                else:
                    phi = 180. + abs(math.degrees(math.atan(q2sinphi / q2cosphi)))
            # theta = math.degrees(math.atan(q2 / q3))

            if q3 > 0.:
                if q2 > 0.:
                    theta = math.degrees(math.atan(q2 / q3))
                else:
                    theta = 360. - abs(math.degrees(math.atan(q2 / q3)))
            else:
                if q2 > 0.:
                    theta = 180. - abs(math.degrees(math.atan(q2 / q3)))
                else:
                    theta = 180. + abs(math.degrees(math.atan(q2 / q3)))
                    # bigQ2 = np.array([q1,q2,q3],dtype='float64')
            # bigQ2 = math.sqrt((bigQ2*bigQ2).sum())
            closest_pucker = find_closest_pucker(np.deg2rad(phi), np.deg2rad(theta), pucker_dict)

            pucker_results.append({NAME_KEY: header, PHI_KEY: phi, THETA_KEY: theta, Q_KEY: big_q,
                                   PUCKER_KEY: closest_pucker})

    return pucker_results


def create_pucker_dict(pucker_dict_file):
    """
    Reads in a list of pucker ids and their phi, theta in degrees and returns a dict of pucker id with [phi, theta]
    in radians
    :param pucker_dict_file: location of the csv with "pucker_id","phi,degrees","theta,degrees"
    :return:
    """
    raw_dict = read_csv_to_dict(pucker_dict_file, quote_style=csv.QUOTE_NONNUMERIC)

    rad_dict = {}
    for line in raw_dict:
        rad_dict[line["pucker_id"]] = [np.deg2rad(line["phi,degrees"]), np.deg2rad(line["theta,degrees"])]
    return rad_dict


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
        pucker_dict = create_pucker_dict(args.pucker_dict)
        result_dict = process_file(args.file, pucker_dict)
        if args.out_file is None:
            args.out_file = create_out_fname(args.file, base_dir=args.out_dir, ext='.out')
        write_csv(result_dict, args.out_file, OUT_KEYS)
    finally:
        pass

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
