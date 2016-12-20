from __future__ import print_function

from collections import OrderedDict
from collections import defaultdict

import pandas as pd

from qm_utils.hartree_valid import verify_local_minimum, verify_transition_state
from qm_utils.qm_common import warning, create_out_fname, InvalidDataError

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
# Boltzmann Constant in kcal/mol K
K_B = 0.001985877534

# Defaults #

DEFAULT_TEMPERATURE = 298.15


# Logic #


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
    parser.add_argument("-o", "--out_file", help="The name of the out file.  Defaults "
                                                 "to the first input file name with the "
                                                 "suffix of the data type")
    parser.add_argument("-t", "--temperature", help="The temperature to use for the KBT "
                                                    "calculation (defaults to {})".format(DEFAULT_TEMPERATURE),
                        default=DEFAULT_TEMPERATURE)

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


def bin_dframes_by_type(dframes):
    binned_frames = defaultdict(list)
    for dframe in dframes:
        if (verify_local_minimum(dframe)):
            binned_frames[DT_LM].append(dframe)
        elif (verify_transition_state(dframe)):
            binned_frames[DT_TS].append(dframe)
        else:
            raise InvalidDataError("Data frame is neither a local minimum nor a transition state")
    return binned_frames


def calc_boltz(dframes):

    pass


def main(argv=None):
    args, ret = parse_cmdline(argv)

    if ret != 0:
        return ret

    dframes = create_dframes(args.input)

    try:
        binned_dframes = bin_dframes_by_type(dframes)
    except InvalidDataError as e:
        warning("Invalid input: ", e)
        return 6

    if len(binned_dframes[DT_LM]) == 0:
        warning("You must specify at least one local minimum file")
        return 7

    if args.type == DT_TS and len(binned_dframes[DT_TS]) == 0:
        warning("Type", DT_TS, "requires at least one transition state file")
        return 8

    calc_boltz(dframes)

    # TODO: create a better outfile name
    grouped_dframe.to_csv(get_out_file_name(args.out_file, dframes.keys()[0], args.group_type), index=False)

    return 0  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
