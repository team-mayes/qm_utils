#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
From 2007Hill_Supporting.pdf

cp_params.py
Usage:
./cp_params.py file.dat
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
from qm_common import (GOOD_RET, INVALID_DATA, warning, read_csv_to_dict, create_out_fname, write_csv, InvalidDataError,
                       INPUT_ERROR)

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

PHI_RAD = 'phi_rad'
THETA_RAD = 'theta_rad'
XYZ = 'cart_xyz'


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description='Reads in space-separated lists of atomic coordinates for 6-membered '
                                                 'rings and \noutputs Cremer-Pople puckering coordinates. The input '
                                                 'format is: \n' +
                                                 '   name1 18x(float)\n'
                                                 '   name2 18x(float)\n'
                                                 '   name3 18x(float)...\n'
                                                 'where each line contains a name and Cartesian coordinates for a '
                                                 'unique \n6-membered ring. The order for the atomic coordinates is '
                                                 'the x,y,z coordinates \nfor O5 (or C6) followed by those for C1, C2,'
                                                 ' C3, C4, and then C5.\n\n'
                                                 'This script also returns the closest IUPAC puckering designation '
                                                 'determined by \nthe arc length along the Cremer-Pople sphere.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("file", help="The input file to process")
    parser.add_argument('-o', "--out_file", help="The output file name (defaults to input file name with the '.out' "
                                                 "extension)",
                        default=None)
    parser.add_argument('-d', "--out_dir", help="The output directory (defaults to the same directory as the "
                                                "input file)",
                        default=None)
    parser.add_argument('-p', "--pucker_dict", help="A csv file specifying the Cremer-Pople puckering parameters (in "
                                                    "degrees) \nfor each IUPAC specified puckering designation. The "
                                                    "default dictionary is \nlocated at: {}".format(PUCKER_DICT_FILE),
                        default=PUCKER_DICT_FILE)
    parser.add_argument('-s', "--to_stdout", help="Flag to display output to standard out (screen) in addition to "
                                                  "saving \nthe output to a file. The default is false.",
                        action='store_true')

    args = None
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        if e.message == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return [], INPUT_ERROR

    return args, GOOD_RET


def to_float_list(a):
    """
    Given an interable, returns a list of its contents converted to floats
    :param a: interable
    :return: list of floats
    """
    return [float(i) for i in a]


def angles_to_xyz(phi, theta):
    """
    Calculates the distance on a unit sphere for Cremer-Pople coordinates
    Note!!! CP uses theta for latitude!! this is opposite of much of math convention
    :param phi: longitude in radians
    :param theta: latitude in radians
    :return: a numpy array of Cartesian coordinates on the unit sphere
    """
    sin_theta = math.sin(theta)
    return np.array((sin_theta * math.cos(phi),
                     sin_theta * math.sin(phi),
                     math.cos(theta)))


def find_closest_pucker(phi, theta, pucker_dict):
    """
    Calculated based on cord length:
    delta_x = sin(phi2) * cos(theta2) - cos(ph1)*cos(theta1)
    delta_y = sin(phi2) * sin(theta2) - cos(phi1)*sin(theta1)
    delta_z = cos(phi2) - cos(phi1)
    chord_length = sqrt( delta_x^2 + delta_y^2 + delta_z^2)
       (aka norm of the vectors)
    central angle (aka delta_sigma) = 2 * arcsin( chord / 2)
    d = r * delta_sigma
    :param phi: phi of the pucker to identify
    :param theta: second CP parameter of the pucker to identify
    :param pucker_dict: dictionary of names and CP params of IUPAC puckers
    :return: closest_pucker: (string) closest IUPAC pucker name
    """
    closest_pucker = None
    closest_dist = np.inf
    xyz = angles_to_xyz(phi, theta)

    for pucker in pucker_dict:
        chord = np.linalg.norm(xyz-pucker_dict[pucker][XYZ])
        current_dist = 2. * math.asin(chord / 2.)
        if current_dist < closest_dist:
            closest_dist = current_dist
            closest_pucker = pucker

    if closest_pucker is None:
        warning("Did not find a closest pucker. Check puckering dictionary.")

    return closest_pucker


def process_file(input_file, pucker_dict, print_to_stdout):
    """
    :param input_file: path of file to be read
    :param pucker_dict: dictionary of pucker_ids
    :param print_to_stdout: boolean to display output to screen as it is calculated
    :return: pucker_results: a list of dicts containing the header, CP parameters, and closest pucker
    """

    pucker_results = []
    atoms = np.zeros((6, 3), dtype='float64')

    with open(input_file, 'r') as f:

        for line in f:
            s_line = line.split()
            if len(s_line) != 19:
                raise InvalidDataError("Expected exactly 19 space-separated entries per line: a header "
                                       "followed by 18 float values. \n"
                                       "However, found {} on line: {}".format(len(s_line), line))
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
                common_term = 2.*math.pi * j / 6.
                q2cosphi += i * math.cos(2.*common_term)
                q2sinphi -= i * math.sin(2.*common_term)
                q1cosphi += i * math.cos(common_term)
                q1sinphi -= i * math.sin(common_term)
                q3 += i * math.cos(j * math.pi)
                big_q += i * i
            q2cosphi *= sqrt_2 * inv_sqrt_6
            q2sinphi *= sqrt_2 * inv_sqrt_6
            q3 *= inv_sqrt_6
            q2 = math.sqrt(q2cosphi * q2cosphi + q2sinphi * q2sinphi)
            big_q = math.sqrt(big_q)

            if q2cosphi > 0.:
                if q2sinphi > 0.:
                    phi = math.degrees(math.atan(q2sinphi / q2cosphi))
                else:
                    if q2cosphi == 0:
                        phi = 270.
                    else:
                        phi = 360. - abs(math.degrees(math.atan(q2sinphi / q2cosphi)))
            else:
                if q2sinphi > 0.:
                    phi = 180. - abs(math.degrees(math.atan(q2sinphi / q2cosphi)))
                else:
                    if q2cosphi == 0:
                        phi = 270.
                    else:
                        phi = 180. + abs(math.degrees(math.atan(q2sinphi / q2cosphi)))

            if q3 > 0.:
                if q2 > 0.:
                    theta = math.degrees(math.atan(q2 / q3))
                else:
                    theta = 360. - abs(math.degrees(math.atan(q2 / q3)))
            else:
                if q2 > 0.:
                    if q3 == 0:
                        theta = 90.
                    else:
                        theta = 180. - abs(math.degrees(math.atan(q2 / q3)))
                else:
                    if q3 == 0:
                        theta = 270.
                    else:
                        theta = 180. + abs(math.degrees(math.atan(q2 / q3)))

            closest_pucker = find_closest_pucker(np.deg2rad(phi), np.deg2rad(theta), pucker_dict)
            if print_to_stdout:
                print("{:12} {:8.3f} {:8.3f} {:8.3f} {}".format(header, phi, theta, big_q, closest_pucker))
            pucker_results.append({NAME_KEY: header, PHI_KEY: phi, THETA_KEY: theta, Q_KEY: big_q,
                                   PUCKER_KEY: closest_pucker})

    return pucker_results


def create_pucker_dict(pucker_dict_file):
    """
    Reads in a list of pucker ids and their phi, theta in degrees and returns a dict of pucker id with [phi, theta]
    in radians
    :param pucker_dict_file: location of the csv with "pucker_id","phi,degrees","theta,degrees"
    :return: rad_dict: a dictionary with IUPAC pucker names as the key mapping to a dictionary designating its
        CP phi, CP theta, and Cartesian coordinates (XYZ key)
    """
    raw_dict = read_csv_to_dict(pucker_dict_file, quote_style=csv.QUOTE_NONNUMERIC)

    rad_dict = {}
    for line in raw_dict:
        phi = np.deg2rad(line["phi,degrees"])
        theta = np.deg2rad(line["theta,degrees"])
        rad_dict[line["pucker_id"]] = {PHI_RAD: phi,
                                       THETA_RAD: theta,
                                       XYZ: angles_to_xyz(phi, theta),
                                       }
    return rad_dict


def main(argv=None):
    """
    Runs the main program
    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret

    try:
        pucker_dict = create_pucker_dict(args.pucker_dict)
        result_dict = process_file(args.file, pucker_dict, args.to_stdout)
        if args.out_file is None:
            args.out_file = create_out_fname(args.file, base_dir=args.out_dir, ext='.out')
        write_csv(result_dict, args.out_file, OUT_KEYS)
    except InvalidDataError as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
