from __future__ import print_function

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
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

## Class ##

## Command Line Parse ##

def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description='INSERT DESCRIPTION')

    parser.add_argument('-c', "--file_CCSDT", help='File continaing CCDS info')
    parser.add_argument('-m', "--file_min", help='file with min information')
    parser.add_argument('-s', "--file_ts", help="file with ts information")
    parser.add_argument('-o', "--out_file", help="testing")
    parser.add_argument('-p', "--print", help="If you would like a graph put True, otherwise leave blank")


    args = None
    try:
        args = parser.parse_args(argv)
    except IOError as e:
        warning("Problems reading file:", e)
        parser.print_help()
        return args, 2
    print(args)
    return args, 0


## Main ##

def main(argv=None):
    """
        Runs the main program
    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """

    args, ret = parse_cmdline(argv)

    if ret != 0:
        return ret

        #CCSDT_TS = {}
        #CCSDT_MIN = {}


        # with open("z_energies_CCSDT-B3LYP_bxyl.csv", 'rU') as CCSDTCSV:
        #    reader = csv.reader(CCSDTCSV)
        #   for row in reader:
        #      if row[1] is not '':
        #         CCSDT_MIN[row[0]] = row[1]
        #    if row[2] is not '':
        #       CCSDT_TS[row[0]] =row[2]

    with open("a_csv_ts_bxyla_list_file.csv", 'rU') as LOCAL_MIN_CSV:
        headers = next(LOCAL_MIN_CSV)
        delim = "," if "," == headers[0] else " "
        headers = list(filter(None, headers.rstrip().split(delim)))

        headers = headers[1:]

        min_reader = csv.reader(LOCAL_MIN_CSV,skipinitialspace=True,delimiter=delim)
        sums_min = np.array([0]*(len(headers)),dtype=np.float)
        amount_min = np.array([0]*(len(headers)),dtype=np.float)
        max_diff_min = np.array([0]*(len(headers)),dtype=np.float)
        for row in min_reader:
            j=0
            for i in range(len(headers)):
                if row[i+2] is '':
                    j = j + 1
                    continue;
                diff = float(row[1]) - float(row[i+2])
                square = diff **2
                if max_diff_min[j] < square:
                    max_diff_min[j] = square
                sums_min[j] = square + sums_min[j]
                amount_min[j] = amount_min[j] + 1
                j = j+1
    for i in range(len(sums_min)):
        sums_min[i] = sums_min[i]/amount_min[i]
        sums_min[i] = math.sqrt(sums_min[i])
        max_diff_min[i] = math.sqrt(max_diff_min[i])

    with open("a_csv_lm_bxyla_list_file.csv", 'rU') as LOCAL_MIN_CSV:
        headers = next(LOCAL_MIN_CSV)
        delim = "," if "," == headers[0] else " "
        headers = list(filter(None, headers.rstrip().split(delim)))
        headers = headers[1:]
        min_reader = csv.reader(LOCAL_MIN_CSV, skipinitialspace=True, delimiter=delim)
        sums = np.array([0]*len(headers),dtype=np.float)
        amount = np.array([0]*len(headers),dtype=np.float)
        max_diff = np.array([0]*len(headers),dtype=np.float)
        for row in min_reader:


            j = 0
            for i in range(len(headers)):
                if row[i + 2] is '':
                    j = j + 1
                    continue;
                diff = float(row[1]) - float(row[i + 2])
                square = diff ** 2
                if max_diff[j] < square:
                    max_diff[j] = square
                sums[j] = square + sums[j]
                amount[j] = amount[j] + 1
                j = j + 1
    for i in range(len(sums)):
        sums[i] = sums[i] / amount[i]
        sums[i] = math.sqrt(sums[i])
        max_diff[i] = math.sqrt(max_diff[i])



    Max_number = amount + amount_min;
    tot_RMSD = np.sqrt(((sums_min*sums_min*amount_min)+(sums*sums*amount))/Max_number)
    tot_max_diff = np.maximum(max_diff_min,max_diff)


    in_rows = [headers, sums_min,sums,tot_RMSD,tot_max_diff]
    in_rows = zip(*in_rows)
    Titles = ["Method", "RMSD Local Min", "RMSD Transition States","RMSD Total","Max Difference"]
    with open('output.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([Titles])
        for thing in in_rows:
            writer.writerows([thing])



    fig, ax = plt.subplots()

    index = np.arange(len(sums))
    bar_width = 0.2
    opacity = 0.8
    rects1 = plt.bar(index, sums_min, bar_width,
                     alpha=opacity,
                     color='g',
                     label='RMSD (local minima)')

    rects2 = plt.bar(index + bar_width, sums, bar_width,
                     alpha=opacity,
                     color='r',
                     label='RMSD (Transition state)')
    rects3 = plt.bar(index+ 2*bar_width, tot_RMSD, bar_width,
                     alpha=opacity,
                     color='b',
                     label='RMSD (Total)')

    rects4 = plt.bar(index + 3*bar_width, tot_max_diff, bar_width,
                     alpha=opacity,
                     color='y',
                     label='Max difference')

    plt.xlabel('Method')
    plt.ylabel('Gibbs Free Energy (kcal/ mol)')
    high = max(tot_max_diff)
    low = min(sums_min)
    plt.ylim(0, math.ceil(high + 1 * (high - low)))
    plt.xticks(index + bar_width, headers)
    plt.legend()
    plt.tight_layout()
    plt.show()






    return 0  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
