from __future__ import print_function

from collections import OrderedDict
from collections import defaultdict

import pandas as pd

from qm_utils.hartree_valid import verify_local_minimum, verify_transition_state
from qm_utils.qm_common import warning, create_out_fname, InvalidDataError, write_csv

"""
Groups output from Hartree by pucker (for local minima) or path (for transition states).
"""

import sys
import argparse


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description='INSERT DESCRIPTION')

    parser.add_argument('-i', "--in_file", help='Input file')
    parser.add_argument('-o', "--out_dir", help='Input file')

    # parser.add_argument("-i", "--input_rates", help="The location of the input rates file",
    #                     default=DEF_IRATE_FILE, type=read_input_rates)

    args = None
    try:
        args = parser.parse_args(argv)
    except IOError as e:
        warning("Problems reading file:", e)
        parser.print_help()
        return args, 2
    print(args)
    return args, 0


def main(argv=None):
    """
        Runs the main program
    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """

    args, ret = parse_cmdline(argv)

    if ret != 0:
        return ret
    sorting = True
    hold_lines = []
    with open(args.in_file, 'r') as text_file:
        for row in text_file:
            hold_lines.append(row)
    sorting = True
    int i =0
    if line in ['\n', '\r\n']:
    while sorting:
        file_name =  hold_lines[i] + ".txt"
        hold_new_lines = []
        for lines in
        if left < 300:
            while count < left:
                hold_new_lines.append(hold_lines[line_count])
                count += 1
                line_count += 1
            sorting = False
        else:
            while count < 300:
                hold_new_lines.append(hold_lines[line_count])
                count += 1
                line_count += 1
        outer_count += 1
        with open(file_name, 'w') as next_file:
            for row in hold_new_lines:
                next_file.write(row)

    return 0  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)