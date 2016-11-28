from __future__ import print_function

#!/usr/bin/env python
from collections import OrderedDict

import pandas as pd

from qm_utils.hartree_valid import verify_local_minimum, verify_transition_state
from qm_utils.qm_common import warning, create_out_fname

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
    # We use the first input for out default out_file, so we use an OrderedDict here
    dframes = OrderedDict()
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
    parser.add_argument("-o", "--out_file", help="The name of the out file.  Defaults "
                                                 "to the first input file name with the "
                                                 "suffix of the data type")

    args = None
    try:
        args = parser.parse_args(argv)
    except IOError as e:
        warning("Problems reading file:", e)
        parser.print_help()
        return args, 2

    return args, 0


def group_by_pucker(dframes):
    dframe = pd.concat(dframes.values())
    dframe[GROUP_COL] = dframe['Pucker']
    return dframe


def group_by_path(dframes):
    return pd.concat(dframes.values())


def get_out_file_name(out_file, first_in_file, group_type):
    if out_file is not None:
        return out_file
    return create_out_fname(first_in_file, suffix="_" + group_type)


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
        grouped_dframe = group_by_pucker(dframes)
    elif args.group_type == GROUP_PATH:
        # TODO: Verify TS && multiple
        if args.type != DT_TS or len(dframes) < 2:
            warning("Cannot group by path without multiple transition state files")
            exit(5)

        grouped_dframe = group_by_path(dframes)
    else:
        warning("Unhandled group type '", args.group_type, "'")
        exit(4)

    # TODO: create a better outfile name
    grouped_dframe.to_csv(get_out_file_name(args.out_file, dframes.keys()[0], args.group_type), index=False)

    return 0  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
