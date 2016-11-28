from __future__ import print_function

#!/usr/bin/env python
import pandas as pd

from qm_utils.hartree_valid import verify_local_minimum, verify_transition_state
from qm_utils.qm_common import warning

"""
Groups output from Hartree by pucker (for local minima) or path (for transition states).
"""

import sys
import argparse

__author__ = 'cmayes'

# Constants #

DT_TS = 'ts'
DT_LM = 'lm'
DATA_NAMES = {DT_TS: "transition state", DT_LM: "local minima"}
GROUP_PUCKER = 'pucker'
GROUP_PATH = 'path'
GROUP_COL = 'group_name'

# Logic


def create_dframes(inputs):
    dframes = {}
    for fname in inputs:
        dframes[fname] = pd.read_csv(fname)
    return dframes

# Command Processing #


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input_rates", help="The location of the input rates file",
    #                     default=DEF_IRATE_FILE, type=read_input_rates)
    parser.add_argument("type", help="The type of data to group", choices=DATA_NAMES.keys())
    parser.add_argument("input", help="The input files to process", nargs='+')
    parser.add_argument("-g", "--group_type", help="The type of grouping to perform",
                        default=GROUP_PUCKER, choices=[GROUP_PUCKER, GROUP_PATH])

    args = None
    try:
        args = parser.parse_args(argv)
    except IOError as e:
        warning("Problems reading file:", e)
        parser.print_help()
        return args, 2

    return args, 0


def group_by_pucker(dframe):
    dframe[GROUP_COL] = dframe['Pucker']
    return dframe


def group_by_path(dframe):
    pass


def main(argv=None):
    args, ret = parse_cmdline(argv)

    if ret != 0:
        return ret

    dframes = create_dframes(args.input)

    invalids = []
    if args.type == DT_LM:
        for dname, dframe in dframes.items():
            if not verify_local_minimum(dframe):
                invalids.append(dname)
    elif args.type == DT_TS:
        for dname, dframe in dframes.items():
            if not verify_transition_state(dframe):
                invalids.append(dname)
    else:
        warning("Unhandled data type '", args.type, "'")
        exit(1)

    if len(invalids) > 0:
        warning("File(s) do not match criteria for", DATA_NAMES[args.type], ":", " ,".join(invalids))
        exit(3)


    if args.group_type == GROUP_PUCKER:
        grouped_dframe = group_by_pucker(pd.concat(dframes.values()))
    elif args.group_type == GROUP_PATH:
        # TODO: Verify TS
        grouped_dframe = group_by_path(pd.concat(dframes.values()))
    else:
        warning("Unhandled group type '", args.group_type, "'")
        exit(4)

    # TODO: create a better outfile name
    grouped_dframe.to_csv("test_result.csv")

    exit(0)  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
