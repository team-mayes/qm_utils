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
        if verify_local_minimum(dframe):
            binned_frames[DT_LM].append(dframe)
        elif verify_transition_state(dframe):
            binned_frames[DT_TS].append(dframe)
        else:
            raise InvalidDataError("Data frame is neither a local minimum nor a transition state")
    return binned_frames


def calc_boltz(dframes):

    pass

#REQUIRES: A DataFrame object that corresponds to a single input CSV file
#RETURNS: an array that contains "TS_Storage" objects - each correspond to a line in the CSV file
#ASSUMPTIONS: Ethalpy is held in column 11, Pucker is held in column 18
#EFFECTS: Reads in the "minimum" CSV file and returns all of the minimums that were found and what the name ("string")
#of the corresponding transition state should be (along with the pucker and enthalpy - for future use)
def find_TS_for_each_min(binned_frames):
    TS_points = []
    for ir in binned_frames.itertuples():
        name = ir[1]
        forward = "_norm-ircf_am1-minIRC_am1.log"
        reverse = "_norm-ircr_am1-minIRC_am1.log"
        reverse_index = name.find(reverse)
        forward_index = name.find(forward)
        if reverse_index != -1:
            ts_name = name[:reverse_index]
            new = TS_storage(ts_name, 0 , 1,ir[11],ir[18])
            TS_points.append(new)
            continue
        elif forward_index != -1:
            ts_name = name[:forward_index]
            new = TS_storage(ts_name, 1, 0, ir[11], ir[18])
            TS_points.append(new)
            continue
    return TS_points



def finding_H_and_pairing(TS_points,binned_frames):
    all_paths = []
    for value in TS_points:
        for ir in binned_frames.itertuples():
            already_done = False
            if value.return_name() in ir[1]:
                #print("found matching words " + ir[1] + " " + value.return_name())
                for thing in all_paths:
                    if ir[1] == thing.return_name():
                        #print("appending double word"+ ir[1] + " " + thing.return_name())
                        thing.add_minimum(value.forward,value.reverse, value.H, value.minimum)
                        already_done= True


                if(not already_done):
                    #print("appendeding first " + " " + ir[1])
                    all_paths.append(TS_final(ir[1],ir[11],ir[18]))
                    all_paths[-1].add_minimum(value.forward,value.reverse, value.H, value.minimum)

    return all_paths

class TS_storage:
    def __init__(self,name, forward,reverse,H, minimum):
        self.name = name
        self.forward = forward
        self.reverse = reverse
        self.H = H
        self.minimum = minimum

    def return_name(self):
         return self.name
    def return_min(self):
         return self.minimum
    def return_H(self):
         return self.H

def make_dict(TS_final):
    new_dictionary = {}
    minimums = TS_final.return_mins()
    new_dictionary["File name"] = TS_final.return_name()
    new_dictionary["Minimum1"] = minimums[0]["pucker"]
    new_dictionary["Delta H1"] = (-minimums[0]["enthalpy"] + TS_final.return_H())*627.5095
    new_dictionary["Transition Pucker"] = TS_final.return_pucker()
    new_dictionary["Minimum2"] = minimums[1]["pucker"]
    new_dictionary["Delta H2"] = (-TS_final.return_H()+ minimums[1]["enthalpy"])*627.5095

    return new_dictionary




class TS_final:
    def __init__(self,name,enthalpy,pucker):
        self.name = name
        self.forward = 0
        self.H = enthalpy
        self.reverse = 0
        self.pucker = pucker
        self.minimums = []

    def return_name(self):
        return self.name

    def return_H(self):
        return self.H

    def return_pucker(self):
        return self.pucker


    def return_mins(self):
        return self.minimums

    def add_minimum(self, forward, reverse, H, pucker):
        if forward:
            self.forward = 1
        else:
            self.reverse = 1
        min = {'pucker': pucker, "enthalpy": H}
        (self.minimums).append(min)


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
    #grouped_dframe.to_csv(get_out_file_name(args.out_file, dframes.keys()[0], args.group_type), index=False)

    ts_points = []
    states = []
    ts_points = find_TS_for_each_min(min)
    states = finding_H_and_pairing(ts_points, ts)
    list_of_dicts = []
    for all in states:
        list_of_dicts.append(make_dict(all))
    headers = ["File name", "Minimum1", "Delta H1", "Transition Pucker", "Delta H2", "Minimum2"]
    write_csv(list_of_dicts, "/Users/SteveJobs/Desktop/research/TESTS/sam_output.csv", headers)
    return 0  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
