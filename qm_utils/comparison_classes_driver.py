"""
The purpose of this script is to make comparisons for a particular QM method to the reference set of HSP.
"""

# # # import # # #
#region
from __future__ import print_function

import os
import sys

import csv

import matplotlib
matplotlib.use('TkAgg')

import qm_utils.comparison_classes as cc


def main():
    comp_met = cc.Compare_Methods('oxane')

    for method in comp_met.Method_Pathways_dict:
        comp_met.save_raw_data(method)
        comp_met.save_connectivity(method)

    comp_met.save_raw_data()
    comp_met.save_connectivity()
    comp_met.save_tessellation()

    comp_met.write_csvs()

    return 0

if __name__ == '__main__':
    status = main()
    sys.exit(status)
#endregion
