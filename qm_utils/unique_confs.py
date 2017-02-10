#!/usr/bin/env python
# coding=utf-8

"""
Uses cp_snap and vmd clustering output to create a list of unique conformations
"""

from __future__ import print_function
from qm_utils.qm_common import (read_csv_old, conv_map_old, write_csv_old, GOOD_RET, warning, IO_ERROR,
                                InvalidDataError,
                                INVALID_DATA)
import operator
from optparse import OptionParser
import sys

# Columns
FNAME_COL = "File Name"
ENERGY_COL = "Energy (A.U.)"
EN_DIFF = "Energy Difference (kcal/mol)"
SOLV_COL = "Solvent type"
STOI_COL = "Stoichiometry"
CHG_COL = "Charge"
MLT_COL = "Mult"
FUN_COL = "Functional"
BSET_COL = "Basis Set"
DIP_COL = "dipole"
ZPE_COL = "ZPE (kcal/mol)"
H298_COL = "H298 (Hartrees)"
G298_COL = "G298 (Hartrees)"
F1_COL = "Freq 1"
F2_COL = "Freq 2"
PHI_COL = "phi"
THETA_COL = "theta"
Q_COL = "Q"
PUCK_COL = "Pucker"
COL_ORDER = [FNAME_COL, SOLV_COL, STOI_COL, CHG_COL, MLT_COL, FUN_COL,
             BSET_COL, ENERGY_COL, DIP_COL, ZPE_COL, H298_COL, G298_COL, F1_COL, F2_COL,
             PHI_COL, THETA_COL, Q_COL, PUCK_COL, EN_DIFF]

# Constants
HARTREE_KCAL_CONV = 627.5094709
DEF_L_ENERGY = tuple(['default', 0])


def col_conv_map():
    convs = dict()
    convs[FNAME_COL] = str
    convs[SOLV_COL] = str
    convs[STOI_COL] = str
    convs[CHG_COL] = int
    convs[MLT_COL] = int
    convs[FUN_COL] = str
    convs[BSET_COL] = str
    convs[ENERGY_COL] = float
    convs[DIP_COL] = float
    convs[ZPE_COL] = float
    convs[H298_COL] = float
    convs[G298_COL] = float
    convs[F1_COL] = float
    convs[F2_COL] = float
    convs[PHI_COL] = float
    convs[THETA_COL] = float
    convs[Q_COL] = float
    convs[PUCK_COL] = str
    convs[EN_DIFF] = float
    return convs


def read_clusters(clusterfile):
    clusters = list()
    for line in open(clusterfile):
        clusters.append([rawf.replace('.xyz', '') for rawf in line.split()])
    return clusters


def find_l_energy(clusters, val_map):
    l_energies = dict()
    for clow in clusters:
        lenergy = DEF_L_ENERGY
        for fname in clow:
            en_val = val_map[fname][ENERGY_COL]
            if en_val < lenergy[1]:
                lenergy = tuple([fname, en_val])
        if lenergy == DEF_L_ENERGY:
            sys.stderr.write("No energies found for %s\n" % (clow))
        else:
            l_energies[lenergy[0]] = lenergy[1]

    return l_energies


def collect_idx_vals(slenfnames, val_idx):
    slen_vals = list()
    for fname in slenfnames:
        slen_vals.append(val_idx[fname])
    return slen_vals


def add_energy_diff(lenerval, slenfnames, val_idx):
    orig_vals = collect_idx_vals(slenfnames, val_idx)
    kcal_diff_vals = list()
    for val in orig_vals:
        # noinspection PyTypeChecker
        conv_eng = (val[ENERGY_COL] - lenerval) * HARTREE_KCAL_CONV
        energy_val = dict(val)
        energy_val[EN_DIFF] = conv_eng
        kcal_diff_vals.append(energy_val)
    return kcal_diff_vals


def parse_cmdline(sysargs):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    usage = "usage: %prog [options] infile"
    parser = OptionParser(usage=usage)
    (options, p_args) = parser.parse_args(sysargs)
    if len(p_args) < 1:
        sys.stderr.write("Must specify infile")
        parser.print_help()
        parser.exit(-1)

    return options, p_args

    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(
        description='Combines outputReads in best output file and generates new input files.')
    parser.add_argument("-f", "--file", help="The fitevb output file to read, if some values are to be obtained from "
                                             "a previous fitEVB run.",
                        default=None)
    parser.add_argument("-c", "--config", help="The location of the configuration file in ini format. "
                                               "The default file name is {}, located in the "
                                               "base directory where the program as run. See example files in the test "
                                               "directory ({}). Note that in FitEVB, 'ARQ' is the same ARQ as in "
                                               "the evb parameter file and corresponds to the off-diagonal term from "
                                               "Maupin 2006 (http://pubs.acs.org/doi/pdf/10.1021/jp053596r). "
                                               "'ARQ2' corresponds to 'PT' with "
                                               "option 1 ('1-symmetric') and no exchange charges."
                                               "".format(DEF_CFG_FILE, 'tests/test_data/fitevb'),
                        default=DEF_CFG_FILE, type=read_cfg)
    parser.add_argument("-v", "--vii_fit", help="Flag to specify fitting the VII term. The default value "
                                                "is {}.".format(DEF_FIT_VII),
                        default=DEF_FIT_VII)
    parser.add_argument("-s", "--summary_file", help="If a summary file name is specified, the program will append "
                                                     "results to a summary file and specify parameter value changes.",
                        default=False)
    args = None
    try:
        args = parser.parse_args(argv)
    except IOError as e:
        warning("Problems reading file:", e)
        parser.print_help()
        return args, IO_ERROR
    except KeyError as e:
        warning("Input data missing:", e)
        parser.print_help()
        return args, INPUT_ERROR
    except SystemExit as e:
        if e.message == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR

    if args.file is not None and not os.path.isfile(args.file):
        if args.file == DEF_BEST_FILE:
            warning("Problems reading specified default fitevb output file ({}) in current directory. "
                    "A different name or directory can be specified with the optional "
                    "-f or --file arguments".format(args.file))
        else:
            warning("Problems reading specified fitevb output file: {}".format(args.file))
        parser.print_help()
        return args, IO_ERROR

    return args, GOOD_RET


# To process the "cluster out data" and hartree output to get a list of unique conformations,
# and a list of the name associated with checkpoints that can be tossed. Each line of the
# "cluster out data" contains file names (+ ".xyz") with molecules in the same conformation.
# The hartree output has the file names and the energies.  In each row, pick the file name
# with the lowest energy.  Print the whole hartree output line to a new file.  Separately, keep
# a list of the reject file names.
# Steps:
# 1) Read in CSV of all the energies; I'll be mainly using:
#     a) "File Name"
#     b) "Energy (A.U.)"
# 2) For each line in the "cluster out" file,
#     a) Read the first file name (minus ".xyz"). Keep it.
#     b) Use the map from 1) to get the energy. Keep it.
#     c) While there is another file name, get the energy.
#         i) If the energy is lower than the previously kept energy, move the previously
#            kept name to the reject list. Keep the new name and new energy.
#         ii) Else, move the new name to the reject list.
#     d) Print the entire hartree output for the kept line to a the new output file.


def process_files(run_name):
    """
    """
    hartree_file = run_name + 'cpsnap.csv'
    cluster_file = run_name + 'clusterout.txt'
    winner_file = run_name + 'cpsnapwinners.csv'
    raw_map = read_csv_old(hartree_file)
    orig_vals = conv_map_old(col_conv_map(), raw_map)
    val_idx = dict([(cur_val[FNAME_COL], cur_val) for cur_val in orig_vals])
    lenergies = find_l_energy(read_clusters(cluster_file), val_idx)
    slenergies = sorted(lenergies.items(), key=operator.itemgetter(1))
    if len(slenergies) == 0:
        sys.stderr.write("No energies in lowest energy map.  Stopping.")
        return
    energy_diff = add_energy_diff(slenergies[0][1],
                                  [slen[0] for slen in slenergies], val_idx)

    write_csv_old(open(winner_file, 'wb'), energy_diff, COL_ORDER)


def main(argv=None):
    # Read input
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret
    cfg = args.config

    try:
        process_files(cfg)
    except IOError as e:
        warning("IOError:", e)
        return IO_ERROR
    except InvalidDataError as e:
        warning("Invalid data:", e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
