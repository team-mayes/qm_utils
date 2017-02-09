from __future__ import print_function

from collections import OrderedDict
from collections import defaultdict

import pandas as pd
import os
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
    parser.add_argument('-d', "--out_directory", help='Output directory')

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
    hold_lines = []
    with open(args.in_file, 'r') as text_file:
        for row in text_file:
            hold_lines.append(row)
    i =0
    last = 0
    new_array = []
    hold_new_lines = []
    file_names = []


    for lines in hold_lines:
        if lines in ['\n', '\r\n']:
            new_array = hold_lines[last:i];
            new_array.insert(0, str(i-last-1)+'\n')
            hold_new_lines.append(new_array)
            string_name = hold_lines[last][:(len(hold_lines[last])-1)] + ".xyz"
            string_name = string_name.replace(" \\$","")
            file_names.append(string_name)
            last = i+1;
        i = i+1

    for i in range(len(hold_new_lines)):
        complete_path = os.path.join(args.out_directory, file_names[i])
        with open(complete_path, 'w') as next_file:
            for row in hold_new_lines[i]:
                next_file.write(str(row))

    return 0  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
