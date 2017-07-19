#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this python script is to create csv files that will be uploaded into Igor Pro for data analysis. The
coordinates that are required are phi and theta.
"""

from __future__ import print_function

import argparse
import os
import statistics as st
import sys

import csv
import pandas as pd
import math
import numpy as np

from qm_utils.igor_mercator_organizer import write_file_data_dict
from qm_utils.pucker_table import read_hartree_files_lowest_energy, sorting_job_types

from qm_utils.qm_common import (GOOD_RET, create_out_fname, warning, IO_ERROR, InvalidDataError, INVALID_DATA,
                                INPUT_ERROR, arc_length_calculator, read_csv_to_dict)

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'SPVicchio'

# # Default Parameters # #
HARTREE_TO_KCALMOL = 627.5095
TOL_ARC_LENGTH = 0.1
TOL_ARC_LENGTH_CROSS = 0.2  # THIS WAS THE ORGINAL TOLERANCE6
DEFAULT_TEMPERATURE = 298.15
K_B = 0.001985877534  # Boltzmann Constant in kcal/mol K

# # Pucker Keys # #
FILE_NAME = 'File Name'
PUCKER = 'Pucker'
ENERGY_ELECTRONIC = 'Energy (A.U.)'
THETA = 'theta'
PHI = 'phi'
Q_VAL = 'Q'
GIBBS = 'G298 (Hartrees)'
ENTH = "H298 (Hartrees)"
MPHI = 'mean phi'
MTHETA = 'mean theta'
GID = 'group ID'
WEIGHT_GIBBS = 'Boltz Weight Gibbs'
WEIGHT_ENTH = 'Boltz Weight Enth'
CPK = 'closest group puck'
FREQ = 'Freq 1'

LM1_GROUP = 'group_lm1'
LM2_GROUP = 'group_lm2'

IRCF = 'ircf'
IRCR = 'ircr'

LMIRC_p1 = 'lm irc phi1'
LMIRC_t1 = 'lm irc theta1'
LMIRC_p2 = 'lm irc phi2'
LMIRC_t2 = 'lm irc theta2'

HSP_LM1 = 'HSP TS ref group 1'
HSP_LM2 = 'HSP TS ref group 2'
HSP_TS = 'HSP TS ref'

HSP_LM_ID = 'LM_compare_values'
MISSING = 'missing'

# # Default CP Params # #

CP_PARAMS = [{'PHI': '180', 'THETA': '180', 'Pucker': '1c4', 'Q': '0.57'},
             {'PHI': '180', 'THETA': '0', 'Pucker': '4c1', 'Q': '0.57'},
             {'PHI': '240', 'THETA': '90', 'Pucker': '14b', 'Q': '0.76'},
             {'PHI': '60', 'THETA': '90', 'Pucker': 'b14', 'Q': '0.76'},
             {'PHI': '120', 'THETA': '90', 'Pucker': '25b', 'Q': '0.76'},
             {'PHI': '300', 'THETA': '90', 'Pucker': 'b25', 'Q': '0.76'},
             {'PHI': '0', 'THETA': '90', 'Pucker': '3ob', 'Q': '0.76'},
             {'PHI': '180', 'THETA': '90', 'Pucker': 'b03', 'Q': '0.76'},
             {'PHI': '270', 'THETA': '129', 'Pucker': '1h2', 'Q': '0.42'},
             {'PHI': '90', 'THETA': '51', 'Pucker': '2h1', 'Q': '0.42'},
             {'PHI': '150', 'THETA': '51', 'Pucker': '2h3', 'Q': '0.42'},
             {'PHI': '330', 'THETA': '129', 'Pucker': '3h2', 'Q': '0.42'},
             {'PHI': '30', 'THETA': '129', 'Pucker': '3h4', 'Q': '0.42'},
             {'PHI': '210', 'THETA': '51', 'Pucker': '4h3', 'Q': '0.42'},
             {'PHI': '270', 'THETA': '51', 'Pucker': '4h5', 'Q': '0.42'},
             {'PHI': '90', 'THETA': '129', 'Pucker': '5h4', 'Q': '0.42'},
             {'PHI': '150', 'THETA': '129', 'Pucker': '5ho', 'Q': '0.42'},
             {'PHI': '330', 'THETA': '51', 'Pucker': 'oh5', 'Q': '0.42'},
             {'PHI': '30', 'THETA': '51', 'Pucker': 'oh1', 'Q': '0.42'},
             {'PHI': '210', 'THETA': '129', 'Pucker': '1ho', 'Q': '0.42'},
             {'PHI': '210', 'THETA': '88', 'Pucker': '1s3', 'Q': '0.62'},
             {'PHI': '30', 'THETA': '92', 'Pucker': '3s1', 'Q': '0.62'},
             {'PHI': '90', 'THETA': '92', 'Pucker': '5s1', 'Q': '0.62'},
             {'PHI': '270', 'THETA': '88', 'Pucker': '1s5', 'Q': '0.62'},
             {'PHI': '330', 'THETA': '88', 'Pucker': 'os2', 'Q': '0.62'},
             {'PHI': '150', 'THETA': '92', 'Pucker': '2so', 'Q': '0.62'},
             {'PHI': '240', 'THETA': '125', 'Pucker': '1e', 'Q': '0.45'},
             {'PHI': '60', 'THETA': '55', 'Pucker': 'e1', 'Q': '0.45'},
             {'PHI': '120', 'THETA': '55', 'Pucker': '2e', 'Q': '0.45'},
             {'PHI': '300', 'THETA': '125', 'Pucker': 'e2', 'Q': '0.45'},
             {'PHI': '360', 'THETA': '125', 'Pucker': '3e', 'Q': '0.45'},
             {'PHI': '180', 'THETA': '55', 'Pucker': 'e3', 'Q': '0.45'},
             {'PHI': '240', 'THETA': '55', 'Pucker': '4e', 'Q': '0.45'},
             {'PHI': '60', 'THETA': '125', 'Pucker': 'e4', 'Q': '0.45'},
             {'PHI': '120', 'THETA': '125', 'Pucker': '5e', 'Q': '0.45'},
             {'PHI': '300', 'THETA': '55', 'Pucker': 'e5', 'Q': '0.45'},
             {'PHI': '360', 'THETA': '55', 'Pucker': 'oe', 'Q': '0.45'},
             {'PHI': '180', 'THETA': '125', 'Pucker': 'eo', 'Q': '0.45'},
             {'PHI': '0', 'THETA': '55', 'Pucker': 'oeD', 'Q': '0.45'},
             {'PHI': '360', 'THETA': '90', 'Pucker': '3obD', 'Q': '0.76'},
             {'PHI': '0', 'THETA': '125', 'Pucker': '3eD', 'Q': '0.45'}]

# # HSP REFERENCE POINTS # #

BXYL_LM_PARAMS = {
    'group_11': {'phi': [328.3, 328.8], 'mean phi': 328.55, 'Boltz Weight Gibbs': 8.3503, 'files': ['puck24', 'puck26'],
                 'G298 (Hartrees)': 8.36, 'closest group puck': ['os2'], 'mean theta': 92.45,
                 'Boltz Weight Enth': 8.5705,
                 'theta': [92.4, 92.5]},
    'group_00': {'phi': [14.7, 51.3], 'mean phi': 33.0, 'Boltz Weight Gibbs': 0.1643, 'files': ['puck15', 'puck16'],
                 'G298 (Hartrees)': 0.16, 'closest group puck': ['4c1'], 'mean theta': 1.7, 'Boltz Weight Enth': 0.1643,
                 'theta': [2.1, 1.3]},
    'group_12': {'phi': [339.4], 'mean phi': 339.4, 'Boltz Weight Gibbs': 8.6, 'files': ['puck25'],
                 'G298 (Hartrees)': 8.6,
                 'closest group puck': ['os2'], 'mean theta': 90.8, 'Boltz Weight Enth': 8.4, 'theta': [90.8]},
    'group_03': {'phi': [45.2, 48.8], 'mean phi': 47.0, 'Boltz Weight Gibbs': 9.5833, 'files': ['puck19', 'puck20'],
                 'G298 (Hartrees)': 9.58, 'closest group puck': ['b14'], 'mean theta': 86.55,
                 'Boltz Weight Enth': 10.5249,
                 'theta': [86.3, 86.8]},
    'group_04': {'phi': [96.7, 94.4], 'mean phi': 95.55, 'Boltz Weight Gibbs': 8.0615, 'files': ['puck17', 'puck18'],
                 'G298 (Hartrees)': 8.06, 'closest group puck': ['5s1'], 'mean theta': 90.2,
                 'Boltz Weight Enth': 8.7512,
                 'theta': [88.1, 92.3]},
    'group_06': {'phi': [155.5], 'mean phi': 155.5, 'Boltz Weight Gibbs': 3.9, 'files': ['puck12'],
                 'G298 (Hartrees)': 3.9,
                 'closest group puck': ['2so'], 'mean theta': 86.8, 'Boltz Weight Enth': 4.2, 'theta': [86.8]},
    'group_08': {'phi': [249.1], 'mean phi': 249.1, 'Boltz Weight Gibbs': 11.0, 'files': ['puck1'],
                 'G298 (Hartrees)': 11.0, 'closest group puck': ['14b'], 'mean theta': 91.8, 'Boltz Weight Enth': 12.4,
                 'theta': [91.8]},
    'group_05': {'phi': [113.5], 'mean phi': 113.5, 'Boltz Weight Gibbs': 7.9, 'files': ['puck11'],
                 'G298 (Hartrees)': 7.9,
                 'closest group puck': ['25b'], 'mean theta': 91.6, 'Boltz Weight Enth': 8.8, 'theta': [91.6]},
    'group_01': {'phi': [5.5, 7.1], 'mean phi': 6.3, 'Boltz Weight Gibbs': 7.8833, 'files': ['puck22', 'puck23'],
                 'G298 (Hartrees)': 7.88, 'closest group puck': ['o3b'], 'mean theta': 89.65,
                 'Boltz Weight Enth': 8.0416,
                 'theta': [89.6, 89.7]},
    'group_07': {'phi': [195.7, 194.7], 'mean phi': 195.2, 'Boltz Weight Gibbs': 5.9458, 'files': ['puck6', 'puck21'],
                 'G298 (Hartrees)': 5.95, 'closest group puck': ['1s3'], 'mean theta': 86.2,
                 'Boltz Weight Enth': 6.8542,
                 'theta': [86.2, 86.2]},
    'group_13': {'phi': [327.3, 30.0, 66.5, 44.9], 'mean phi': 117.175, 'Boltz Weight Gibbs': 2.4546,
                 'files': ['puck2', 'puck3', 'puck4', 'puck5'], 'G298 (Hartrees)': 2.45, 'closest group puck': ['1c4'],
                 'mean theta': 177.5, 'Boltz Weight Enth': 1.5752, 'theta': [177.9, 177.8, 176.6, 177.7]},
    'group_09': {'phi': [264.8], 'mean phi': 264.8, 'Boltz Weight Gibbs': 7.9, 'files': ['puck9'],
                 'G298 (Hartrees)': 7.9,
                 'closest group puck': ['1s5'], 'mean theta': 92.6, 'Boltz Weight Enth': 8.3, 'theta': [92.6]},
    'group_02': {'phi': [19.0, 17.0], 'mean phi': 18.0, 'Boltz Weight Gibbs': 7.8128, 'files': ['puck13', 'puck14'],
                 'G298 (Hartrees)': 7.82, 'closest group puck': ['3s1'], 'mean theta': 90.35,
                 'Boltz Weight Enth': 7.7504,
                 'theta': [89.9, 90.8]},
    'group_10': {'phi': [274.2, 272.4, 278.3], 'mean phi': 274.966667, 'Boltz Weight Gibbs': 7.4681,
                 'files': ['puck7', 'puck8', 'puck10'], 'G298 (Hartrees)': 7.47, 'closest group puck': ['1s5'],
                 'mean theta': 89.8, 'Boltz Weight Enth': 7.7383, 'theta': [89.4, 89.2, 90.8]}}

BXYL_TS_PARAMS = {
    'group_39': {'phi': [327.9, 332.3], 'mean phi': 330.1, 'Boltz Weight Enth': 11.8752, 'group_lm2': ['group_02'],
                 'files': ['puck45', 'puck46'], 'Boltz Weight Gibbs': 12.8128, 'closest group puck': ['3h2'],
                 'mean theta': 115.25, 'group_lm1': ['group_13'], 'theta': [116.5, 114.0]},
    'group_11': {'phi': [327.2], 'mean phi': 327.2, 'Boltz Weight Enth': 16.4, 'group_lm2': ['group_11'],
                 'files': ['puck88'], 'Boltz Weight Gibbs': 16.3, 'closest group puck': ['oh5'], 'mean theta': 63.6,
                 'group_lm1': ['group_00'], 'theta': [63.6]},
    'group_33': {'phi': [231.5], 'mean phi': 231.5, 'Boltz Weight Enth': 15.3, 'group_lm2': ['group_08'],
                 'files': ['puck36'], 'Boltz Weight Gibbs': 15.9, 'closest group puck': ['1e'], 'mean theta': 123.4,
                 'group_lm1': ['group_13'], 'theta': [123.4]},
    'group_25': {'phi': [315.7], 'mean phi': 315.7, 'Boltz Weight Enth': 9.3, 'group_lm2': ['group_01'],
                 'files': ['puck90'], 'Boltz Weight Gibbs': 10.2, 'closest group puck': ['os2'], 'mean theta': 90.6,
                 'group_lm1': ['group_10'], 'theta': [90.6]},
    'group_18': {'phi': [235.9, 243.1], 'mean phi': 239.5, 'Boltz Weight Enth': 9.6349, 'group_lm2': ['group_06'],
                 'files': ['puck28', 'puck27'], 'Boltz Weight Gibbs': 10.4349, 'closest group puck': ['14b'],
                 'theta': [84.5, 84.1], 'group_lm1': ['group_10'], 'mean theta': 84.3},
    'group_32': {'phi': [145.3, 142.7], 'mean phi': 144.0, 'Boltz Weight Enth': 14.9851, 'group_lm2': ['group_13'],
                 'files': ['puck66', 'puck67'], 'Boltz Weight Gibbs': 15.5645, 'closest group puck': ['5ho'],
                 'theta': [124.7, 127.2], 'group_lm1': ['group_06'], 'mean theta': 125.95},
    'group_26': {'phi': [35.6, 25.7], 'mean phi': 30.65, 'Boltz Weight Enth': 14.0663, 'group_lm2': ['group_13'],
                 'files': ['puck52', 'puck53'], 'Boltz Weight Gibbs': 14.3598, 'closest group puck': ['3h4'],
                 'mean theta': 116.95, 'group_lm1': ['group_06'], 'theta': [116.6, 117.3]},
    'group_08': {'phi': [206.0, 196.1, 199.4], 'mean phi': 200.5, 'Boltz Weight Enth': 10.7411,
                 'group_lm2': ['group_00'],
                 'files': ['puck54', 'puck55', 'puck56'], 'Boltz Weight Gibbs': 10.8697, 'closest group puck': ['4h3'],
                 'mean theta': 54.366667, 'group_lm1': ['group_06'], 'theta': [54.6, 54.6, 53.9]},
    'group_21': {'phi': [314.5], 'mean phi': 314.5, 'Boltz Weight Enth': 9.8, 'group_lm2': ['group_02'],
                 'files': ['puck69'], 'Boltz Weight Gibbs': 10.7, 'closest group puck': ['b25'], 'mean theta': 90.3,
                 'group_lm1': ['group_10'], 'theta': [90.3]},
    'group_05': {'phi': [148.6], 'mean phi': 148.6, 'Boltz Weight Enth': 10.2, 'group_lm2': ['group_06'],
                 'files': ['puck42'], 'Boltz Weight Gibbs': 10.2, 'closest group puck': ['2h3'], 'mean theta': 62.0,
                 'group_lm1': ['group_00'], 'theta': [62.0]},
    'group_00': {'phi': [12.3], 'mean phi': 12.3, 'Boltz Weight Enth': 15.9, 'group_lm2': ['group_02'],
                 'files': ['puck81'], 'Boltz Weight Gibbs': 16.1, 'closest group puck': ['oe'], 'mean theta': 61.6,
                 'group_lm1': ['group_00'], 'theta': [61.6]},
    'group_10': {'phi': [333.4], 'mean phi': 333.4, 'Boltz Weight Enth': 16.3, 'group_lm2': ['group_00'],
                 'files': ['puck89'], 'Boltz Weight Gibbs': 16.4, 'closest group puck': ['oh5'], 'mean theta': 61.1,
                 'group_lm1': ['group_01'], 'theta': [61.1]},
    'group_17': {'phi': [237.5], 'mean phi': 237.5, 'Boltz Weight Enth': 12.2, 'group_lm2': ['group_07'],
                 'files': ['puck29'], 'Boltz Weight Gibbs': 12.6, 'closest group puck': ['14b'], 'mean theta': 94.0,
                 'group_lm1': ['group_09'], 'theta': [94.0]},
    'group_13': {'phi': [1.4], 'mean phi': 1.4, 'Boltz Weight Enth': 9.4, 'group_lm2': ['group_02'],
                 'files': ['puck79'],
                 'Boltz Weight Gibbs': 10.9, 'closest group puck': ['o3b'], 'mean theta': 89.2,
                 'group_lm1': ['group_12'],
                 'theta': [89.2]},
    'group_09': {'phi': [290.4, 280.4], 'mean phi': 285.4, 'Boltz Weight Enth': 14.7643, 'group_lm2': ['group_00'],
                 'files': ['puck78', 'puck57'], 'Boltz Weight Gibbs': 14.7643, 'closest group puck': ['e5'],
                 'theta': [63.0, 63.4], 'group_lm1': ['group_10'], 'mean theta': 63.2},
    'group_02': {'phi': [29.3, 38.3, 31.6], 'mean phi': 33.0667, 'Boltz Weight Enth': 14.2176,
                 'group_lm2': ['group_00'],
                 'files': ['puck84', 'puck85', 'puck82'], 'Boltz Weight Gibbs': 14.6437, 'closest group puck': ['oh1'],
                 'theta': [65.5, 62.1, 62.3], 'group_lm1': ['group_02'], 'mean theta': 63.3},
    'group_16': {'phi': [184.1], 'mean phi': 184.1, 'Boltz Weight Enth': 9.8, 'group_lm2': ['group_06'],
                 'files': ['puck73'], 'Boltz Weight Gibbs': 10.5, 'closest group puck': ['bo3'], 'mean theta': 86.9,
                 'group_lm1': ['group_07'], 'theta': [86.9]},
    'group_35': {'phi': [255.0, 264.5, 257.0], 'mean phi': 258.8333, 'Boltz Weight Enth': 11.1275,
                 'group_lm2': ['group_09'], 'files': ['puck32', 'puck39', 'puck38'], 'Boltz Weight Gibbs': 12.098,
                 'closest group puck': ['1h2'], 'theta': [121.3, 120.7, 120.8], 'group_lm1': ['group_13'],
                 'mean theta': 120.9333},
    'group_06': {'phi': [136.6], 'mean phi': 136.6, 'Boltz Weight Enth': 13.6, 'group_lm2': ['group_05'],
                 'files': ['puck43'], 'Boltz Weight Gibbs': 13.3, 'closest group puck': ['2h3'], 'mean theta': 60.0,
                 'group_lm1': ['group_00'], 'theta': [60.0]},
    'group_31': {'phi': [135.9], 'mean phi': 135.9, 'Boltz Weight Enth': 16.5, 'group_lm2': ['group_13'],
                 'files': ['puck65'], 'Boltz Weight Gibbs': 16.6, 'closest group puck': ['5e'], 'mean theta': 118.4,
                 'group_lm1': ['group_07'], 'theta': [118.4]},
    'group_30': {'phi': [112.2, 113.3], 'mean phi': 112.75, 'Boltz Weight Enth': 14.4663, 'group_lm2': ['group_13'],
                 'files': ['puck62', 'puck63'], 'Boltz Weight Gibbs': 14.8349, 'closest group puck': ['5e'],
                 'theta': [124.7, 125.4], 'group_lm1': ['group_04'], 'mean theta': 125.05},
    'group_01': {'phi': [31.6, 49.7, 37.2], 'mean phi': 39.5, 'Boltz Weight Enth': 14.3115, 'group_lm2': ['group_00'],
                 'files': ['puck86', 'puck76', 'puck83'], 'Boltz Weight Gibbs': 14.586, 'closest group puck': ['oh1'],
                 'theta': [65.0, 63.2, 61.6], 'group_lm1': ['group_03'], 'mean theta': 63.2667},
    'group_20': {'phi': [239.0], 'mean phi': 239.0, 'Boltz Weight Enth': 15.5, 'group_lm2': ['group_07'],
                 'files': ['puck31'], 'Boltz Weight Gibbs': 15.6, 'closest group puck': ['14b'], 'mean theta': 88.3,
                 'group_lm1': ['group_11'], 'theta': [88.3]},
    'group_34': {'phi': [225.4, 228.8, 224.6, 227.9], 'mean phi': 226.675, 'Boltz Weight Enth': 13.7426,
                 'group_lm2': ['group_07'], 'files': ['puck33', 'puck34', 'puck40', 'puck35'],
                 'Boltz Weight Gibbs': 14.4194, 'closest group puck': ['1e'], 'theta': [123.6, 123.5, 126.0, 126.4],
                 'group_lm1': ['group_13'], 'mean theta': 124.875},
    'group_04': {'phi': [46.0, 48.4], 'mean phi': 47.2, 'Boltz Weight Enth': 14.4598, 'group_lm2': ['group_04'],
                 'files': ['puck74', 'puck75'], 'Boltz Weight Gibbs': 14.5598, 'closest group puck': ['e1'],
                 'theta': [63.1, 62.2], 'group_lm1': ['group_00'], 'mean theta': 62.65},
    'group_03': {'phi': [43.0], 'mean phi': 43.0, 'Boltz Weight Enth': 16.5, 'group_lm2': ['group_01'],
                 'files': ['puck87'], 'Boltz Weight Gibbs': 16.9, 'closest group puck': ['e1'], 'mean theta': 65.0,
                 'group_lm1': ['group_00'], 'theta': [65.0]},
    'group_24': {'phi': [291.7], 'mean phi': 291.7, 'Boltz Weight Enth': 11.2, 'group_lm2': ['group_10'],
                 'files': ['puck72'], 'Boltz Weight Gibbs': 12.3, 'closest group puck': ['b25'], 'mean theta': 91.0,
                 'group_lm1': ['group_11'], 'theta': [91.0]},
    'group_07': {'phi': [174.1], 'mean phi': 174.1, 'Boltz Weight Enth': 8.6, 'group_lm2': ['group_06'],
                 'files': ['puck77'], 'Boltz Weight Gibbs': 8.6, 'closest group puck': ['e3'], 'mean theta': 57.2,
                 'group_lm1': ['group_00'], 'theta': [57.2]},
    'group_19': {'phi': [237.7], 'mean phi': 237.7, 'Boltz Weight Enth': 16.2, 'group_lm2': ['group_06'],
                 'files': ['puck30'], 'Boltz Weight Gibbs': 16.0, 'closest group puck': ['14b'], 'mean theta': 86.6,
                 'group_lm1': ['group_11'], 'theta': [86.6]},
    'group_36': {'phi': [280.0], 'mean phi': 280.0, 'Boltz Weight Enth': 13.1, 'group_lm2': ['group_13'],
                 'files': ['puck37'], 'Boltz Weight Gibbs': 13.5, 'closest group puck': ['1h2'], 'mean theta': 125.9,
                 'group_lm1': ['group_10'], 'theta': [125.9]},
    'group_14': {'phi': [59.3], 'mean phi': 59.3, 'Boltz Weight Enth': 12.7, 'group_lm2': ['group_02'],
                 'files': ['puck68'], 'Boltz Weight Gibbs': 13.0, 'closest group puck': ['b14'], 'mean theta': 87.7,
                 'group_lm1': ['group_04'], 'theta': [87.7]},
    'group_22': {'phi': [303.2], 'mean phi': 303.2, 'Boltz Weight Enth': 10.4, 'group_lm2': ['group_10'],
                 'files': ['puck71'], 'Boltz Weight Gibbs': 11.4, 'closest group puck': ['b25'], 'mean theta': 92.5,
                 'group_lm1': ['group_01'], 'theta': [92.5]},
    'group_27': {'phi': [35.7], 'mean phi': 35.7, 'Boltz Weight Enth': 12.5, 'group_lm2': ['group_13'],
                 'files': ['puck51'], 'Boltz Weight Gibbs': 13.0, 'closest group puck': ['3h4'], 'mean theta': 115.0,
                 'group_lm1': ['group_02'], 'theta': [115.0]},
    'group_37': {'phi': [330.9, 331.5], 'mean phi': 331.2, 'Boltz Weight Enth': 11.493, 'group_lm2': ['group_01'],
                 'files': ['puck44', 'puck47'], 'Boltz Weight Gibbs': 12.5598, 'closest group puck': ['3h2'],
                 'mean theta': 115.25, 'group_lm1': ['group_13'], 'theta': [114.9, 115.6]},
    'group_15': {'phi': [134.4], 'mean phi': 134.4, 'Boltz Weight Enth': 9.3, 'group_lm2': ['group_04'],
                 'files': ['puck41'], 'Boltz Weight Gibbs': 8.9, 'closest group puck': ['25b'], 'mean theta': 86.4,
                 'group_lm1': ['group_06'], 'theta': [86.4]},
    'group_29': {'phi': [113.9], 'mean phi': 113.9, 'Boltz Weight Enth': 14.8, 'group_lm2': ['group_05'],
                 'files': ['puck64'], 'Boltz Weight Gibbs': 15.5, 'closest group puck': ['5e'], 'mean theta': 126.7,
                 'group_lm1': ['group_13'], 'theta': [126.7]},
    'group_12': {'phi': [5.8], 'mean phi': 5.8, 'Boltz Weight Enth': 11.7, 'group_lm2': ['group_03'],
                 'files': ['puck80'],
                 'Boltz Weight Gibbs': 12.7, 'closest group puck': ['o3b'], 'mean theta': 89.8,
                 'group_lm1': ['group_11'],
                 'theta': [89.8]},
    'group_38': {'phi': [328.3, 327.8, 331.6], 'mean phi': 329.233333, 'Boltz Weight Enth': 12.8611,
                 'group_lm2': ['group_13'], 'files': ['puck48', 'puck49', 'puck50'], 'Boltz Weight Gibbs': 13.5661,
                 'closest group puck': ['3h2'], 'mean theta': 120.033333, 'group_lm1': ['group_11'],
                 'theta': [120.3, 119.8, 120.0]},
    'group_23': {'phi': [302.6], 'mean phi': 302.6, 'Boltz Weight Enth': 10.4, 'group_lm2': ['group_10'],
                 'files': ['puck70'], 'Boltz Weight Gibbs': 11.5, 'closest group puck': ['b25'], 'mean theta': 92.9,
                 'group_lm1': ['group_02'], 'theta': [92.9]},
    'group_28': {'phi': [120.6, 126.3, 120.8, 123.1], 'mean phi': 122.7, 'Boltz Weight Enth': 11.2457,
                 'group_lm2': ['group_13'], 'files': ['puck60', 'puck58', 'puck59', 'puck61'],
                 'Boltz Weight Gibbs': 12.099, 'closest group puck': ['5e'], 'theta': [126.7, 125.2, 125.8, 121.0],
                 'group_lm1': ['group_06'], 'mean theta': 124.675}}


# # Script Functions # #

def compute_rmsd_between_puckers(phi, theta, new_cp_params=None, q_val=1):
    """
    This script computes the RMSD between a particular ordered pair of phi and theta values, and then IDs the arc length
    to the closest pucker.
    :param phi: phi value (0 to 360)
    :param theta: theta value (180 to 0)
    :param new_cp_params: any new REFERENCE points that you might want to include in the analysis.
    :param q_val: the value of the radius on the surface of the sphere. Note that if that radius changes your results
                    might change
    :return: the pucker that is closely associated with the particular input structure, and the arc_length value
    """
    arc_length_dict = {}
    min_arc_length = 100

    if new_cp_params is not None:
        CP_PARAMS.append(new_cp_params)

    for row in CP_PARAMS:
        value_arc_length = arc_length_calculator(phi, theta, float(row['PHI']), float(row['THETA']))
        arc_length_dict[row['Pucker']] = value_arc_length
        if value_arc_length < min_arc_length:
            min_arc_length = value_arc_length
            min_arc_length_pucker = row['Pucker']

    for arc_keys in arc_length_dict.keys():
        arc_length_dict[arc_keys] = arc_length_dict[arc_keys] - min_arc_length

    top_difference = (sorted(arc_length_dict, key=arc_length_dict.get, reverse=False)[:5])

    for row in top_difference:
        print(row, arc_length_dict[row])

    return top_difference[0], arc_length_dict[top_difference[0]]


def separating_TS_and_IRC_information(method_dict):
    ts_dict = []
    lm_dict = []

    low_energy_value = 100000000000

    for row in method_dict:
        row_filename = row[FILE_NAME]
        if float(row[FREQ]) < 0:
            ts_dict.append(row)
            if float(row[GIBBS]) < low_energy_value:
                low_energy_value = float(row[GIBBS])
        elif float(row[FREQ]) > 0:
            lm_dict.append(row)
            if float(row[GIBBS]) < low_energy_value:
                low_energy_value = float(row[GIBBS])

    for ts_row in ts_dict:
        ts_row[GIBBS] = (float(ts_row[GIBBS]) - low_energy_value) * HARTREE_TO_KCALMOL

    for lm_row in lm_dict:
        lm_row[GIBBS] = (float(lm_row[GIBBS]) - low_energy_value) * HARTREE_TO_KCALMOL

    return lm_dict, ts_dict


def comparing_TS_structures_arc_length(ts_dict, reference_ts):
    new_ts_dict = []

    for row in ts_dict:
        p1 = float(row[PHI])
        t1 = float(row[THETA])
        arc_length_dict = {}
        for key, key_val in reference_ts.items():
            p2 = float(key_val[MPHI])
            t2 = float(key_val[MTHETA])
            arc_length_dict[key] = arc_length_calculator(p1, t1, p2, t2)
        top_difference = (sorted(arc_length_dict, key=arc_length_dict.get, reverse=False)[:5])

        lowest_arc_length = {}
        if arc_length_dict[top_difference[0]] > TOL_ARC_LENGTH_CROSS:
            row['TS_compare_values'] = 'NONE'
        else:
            for top_row in top_difference:
                if arc_length_dict[top_row] < TOL_ARC_LENGTH_CROSS:
                    lowest_arc_length[top_row] = arc_length_dict[top_row]

            row['TS_compare_values'] = lowest_arc_length

        new_ts_dict.append(row)

    if len(new_ts_dict) != len(ts_dict):
        print('The length of the new TS dict does not equal the length of the old TS dict.')

    return new_ts_dict


def comparing_TS_pathways(ts_dict, lm_dict, reference_ts, reference_lm):
    """
    This script completes the sorting for TS states while looking at pathways. The sorting is based on the following
    criteria:
    (1) For a particular TS, which HSP REFERENCE TS are within a predetermined arc length tolerance. All REFERENCE TSs
    that meet this criteria are kept.
    (2) Now, the TS associated with each HSP REFERENCE TS and the TS associated with a particular TS are compared.
    Again, arc length calculations are computed and compared to the respective TS. If both TS are within a certain
    arc length tolerance, then the HSP REFERENCE group is assigned as a match for the particular TS. This process is
    repeated for all TS.
    (3) If there exist multiple matches (a particular TS has multiple HSP REFERENCE TS), then the pathway with the
    smaller arc length value is selected.

    :param ts_dict: the TS dict associated for a particular method
    :param lm_dict: the TS dict assiciated for a particular method
    :param reference_ts: the HSP REFERENCE TS dict
    :param reference_lm: the HSP REFERENCE TS dict
    :return: overall_ts_dict (all information), matching_ts_dict (matching only), missing_ts_dict (missing only) ...
    (note that the all of these dicts are updated with new information pertaining to their pathways and such).
    """
    status_ircf = False
    status_ircr = False
    overall_ts_dict = []
    matching_ts_dict = []
    missing_ts_dict = []

    for row_ts in ts_dict:
        filename = row_ts[FILE_NAME]
        ind_dict = {}
        for row_lm in lm_dict:
            if row_ts[FILE_NAME].split('_')[0] in row_lm[FILE_NAME] and row_ts[FILE_NAME].split('_')[1] in row_lm[FILE_NAME]:
                if IRCF in row_lm[FILE_NAME]:
                    lm_ircf_phi = float(row_lm[PHI])
                    lm_ircf_theta = float(row_lm[THETA])
                    status_ircf = True
                elif IRCR in row_lm[FILE_NAME]:
                    lm_ircr_phi = float(row_lm[PHI])
                    lm_ircr_theta = float(row_lm[THETA])
                    status_ircr = True

                if status_ircf is True and status_ircr is True:
                    status_ircf = False
                    status_ircr = False
                    break

        match_status = {}
        match_info = {}
        hsp_ts_arc_groups = row_ts['TS_compare_values']
        if hsp_ts_arc_groups == 'NONE':
            row_ts[HSP_TS] = 'missing'
            missing_ts_dict.append(row_ts)
        elif hsp_ts_arc_groups != 'NONE':
            for arc_group_key in hsp_ts_arc_groups.keys():
                match_cp_params = {}
                hsp_pathway_lm1 = reference_ts[arc_group_key][LM1_GROUP][0]
                hsp_pathway_lm2 = reference_ts[arc_group_key][LM2_GROUP][0]

                hsp_path_lm1_phi = float(reference_lm[hsp_pathway_lm1][MPHI])
                hsp_path_lm1_theta = float(reference_lm[hsp_pathway_lm1][MTHETA])
                hsp_path_lm2_phi = float(reference_lm[hsp_pathway_lm2][MPHI])
                hsp_path_lm2_theta = float(reference_lm[hsp_pathway_lm2][MTHETA])

                # for IRCF
                lm_ircf_lm1 = arc_length_calculator(lm_ircf_phi, lm_ircf_theta, hsp_path_lm1_phi, hsp_path_lm1_theta)
                lm_ircf_lm2 = arc_length_calculator(lm_ircf_phi, lm_ircf_theta, hsp_path_lm2_phi, hsp_path_lm2_theta)

                # for IRCR
                lm_ircr_lm1 = arc_length_calculator(lm_ircr_phi, lm_ircr_theta, hsp_path_lm1_phi, hsp_path_lm1_theta)
                lm_ircr_lm2 = arc_length_calculator(lm_ircr_phi, lm_ircr_theta, hsp_path_lm2_phi, hsp_path_lm2_theta)

                if lm_ircf_lm1 < TOL_ARC_LENGTH_CROSS and lm_ircr_lm2 < TOL_ARC_LENGTH_CROSS:
                    match_status[arc_group_key] = 'match'
                    match_cp_params[LMIRC_p1] = lm_ircf_phi
                    match_cp_params[LMIRC_t1] = lm_ircf_theta
                    match_cp_params[LMIRC_p2] = lm_ircr_phi
                    match_cp_params[LMIRC_t2] = lm_ircr_theta
                    match_cp_params[HSP_LM1] = reference_ts[arc_group_key][LM1_GROUP][0]
                    match_cp_params[HSP_LM2] = reference_ts[arc_group_key][LM2_GROUP][0]
                    match_info[arc_group_key] = match_cp_params
                elif lm_ircf_lm2 < TOL_ARC_LENGTH_CROSS and lm_ircr_lm1 < TOL_ARC_LENGTH_CROSS:
                    match_status[arc_group_key] = 'match'
                    match_cp_params[LMIRC_p1] = lm_ircr_phi
                    match_cp_params[LMIRC_t1] = lm_ircr_theta
                    match_cp_params[LMIRC_p2] = lm_ircf_phi
                    match_cp_params[LMIRC_t2] = lm_ircf_theta
                    match_cp_params[HSP_LM1] = reference_ts[arc_group_key][LM1_GROUP][0]
                    match_cp_params[HSP_LM2] = reference_ts[arc_group_key][LM2_GROUP][0]
                    match_info[arc_group_key] = match_cp_params
                else:
                    match_status[arc_group_key] = 'missing'

            list_status_val = []
            for status_keys, status_val in match_status.items():
                list_status_val.append(status_val)

            num_match = list_status_val.count('match')

            if num_match == 0:
                row_ts[HSP_TS] = status_val
                missing_ts_dict.append(row_ts)

            elif num_match == 1:
                for status_keys, status_val in match_status.items():
                    if status_val == 'match':
                        row_ts[HSP_TS] = status_keys
                        row_ts.update(match_info[status_keys])
                        matching_ts_dict.append(row_ts)
                        break

            elif num_match > 1:
                arc_length_lm2_dict = {}
                top_diff = None
                for status_keys, status_val in match_status.items():
                    value_storage = {}

                    hsp_pathway_lm1 = reference_ts[status_keys][LM1_GROUP][0]
                    hsp_pathway_lm2 = reference_ts[status_keys][LM2_GROUP][0]

                    hsp_path_lm1_phi = float(reference_lm[hsp_pathway_lm1][MPHI])
                    hsp_path_lm1_theta = float(reference_lm[hsp_pathway_lm1][MTHETA])
                    hsp_path_lm2_phi = float(reference_lm[hsp_pathway_lm2][MPHI])
                    hsp_path_lm2_theta = float(reference_lm[hsp_pathway_lm2][MTHETA])

                    if status_val == 'match':
                        p1_lm1 = float(match_info[status_keys][LMIRC_p1])
                        t1_lm1 = float(match_info[status_keys][LMIRC_t1])
                        p2_lm2 = float(match_info[status_keys][LMIRC_p2])
                        t2_lm2 = float(match_info[status_keys][LMIRC_t2])

                        arc_length_lm1 = arc_length_calculator(p1_lm1, t1_lm1, hsp_path_lm1_phi, hsp_path_lm1_theta)
                        arc_length_lm2 = arc_length_calculator(p2_lm2, t2_lm2, hsp_path_lm2_phi, hsp_path_lm2_theta)

                        arc_length_lm2_dict[status_keys] = arc_length_lm2

                top_diff = (sorted(arc_length_lm2_dict, key=arc_length_lm2_dict.get, reverse=False)[:1])
                row_ts[HSP_TS] = top_diff[0]
                row_ts.update(match_info[top_diff[0]])
                matching_ts_dict.append(row_ts)

        overall_ts_dict.append(row_ts)

    error_status = check_matching_missing_dicts(overall_ts_dict, matching_ts_dict, missing_ts_dict)

    if error_status is True:
        print('Something wrong happened! There is an error in the sorting of TS and matching TSs...')

    return overall_ts_dict, matching_ts_dict, missing_ts_dict


def check_matching_missing_dicts(overall_ts_dict, match_ts_dict, missing_ts_dict):
    """
    The only purpose of this script is to check the overall, matching, and missing TS dict to ensure that they were
    formulated properly. There was an issue that too many TS were being found (there existed duplicates).

    :param overall_ts_dict: all of the TS structures
    :param match_ts_dict: only the matching TS structures (contains the matching pathway)
    :param missing_ts_dict: the missing TS structures (missing the matching pathway)
    :return: error_status to see if any issues arise...
    """

    error_status = None

    list_match = []
    list_missi = []

    for row_match in match_ts_dict:
        list_match.append(row_match[FILE_NAME])

    for row_miss in missing_ts_dict:
        list_missi.append(row_miss[FILE_NAME])

    for row in overall_ts_dict:
        filename = row[FILE_NAME]
        if filename in list_match and filename in list_missi:
            error_status = True

    if (len(overall_ts_dict) - len(match_ts_dict) - len(missing_ts_dict)) != 0:
        error_status = True

    return error_status


def generate_matching_ts_dict_pathways_full(matching_ts_dict):
    """
    Creates the Igor CSV file so that the file can easily be created in Igor for to visualize the pathways.
    :param matching_ts_dict: The matching_ts_dict parameters.
    :return: the data dict for this particular method
    """
    pathway_phi = []
    pathway_theta = []
    ts_phi = []
    ts_theta = []
    lm_phi = []
    lm_theta = []

    for row in matching_ts_dict:
        ts0_phi = round(float(row[PHI]), 2)
        ts0_theta = round(float(row[THETA]), 2)
        lm1_phi = round(float(row[LMIRC_p1]), 2)
        lm1_theta = round(float(row[LMIRC_t1]), 2)
        lm2_phi = round(float(row[LMIRC_p2]), 2)
        lm2_theta = round(float(row[LMIRC_t2]), 2)

        # Creates a list of all of the points so that they can be analyzed in Igor
        ts_phi.append(ts0_phi)
        ts_theta.append(ts0_theta)
        lm_phi.append(lm1_phi)
        lm_theta.append(lm1_theta)
        lm_phi.append(lm2_phi)
        lm_theta.append(lm2_theta)

        # Creates all of the pathways in a way for Igor to udnerstand the vectors
        pathway_phi.append(lm1_phi)
        pathway_theta.append('')

        pathway_phi.append(lm1_phi)
        pathway_theta.append(lm1_theta)

        pathway_phi.append(ts0_phi)
        pathway_theta.append(ts0_theta)

        pathway_phi.append(lm2_phi)
        pathway_theta.append('')

        pathway_phi.append(lm2_phi)
        pathway_theta.append(lm2_theta)

        pathway_phi.append(ts0_phi)
        pathway_theta.append(ts0_theta)

    data_dict = create_datadict(pathway_phi, pathway_theta, ts_phi, ts_theta, lm_phi, lm_theta)

    return data_dict


def generate_igor_hemi_plots(matching_ts_dict, reference_lm):
    """"""

    group_keys = []

    for key_lm, val_lm in reference_lm.items():
        if val_lm[CPK][0] == '4c1':
            group_keys.append(key_lm)
        elif val_lm[CPK][0] == '1c4':
            group_keys.append(key_lm)

    polar_dict = {}
    for hemi_points in group_keys:
        lm_phi_values = []
        lm_proj_values = []
        ts_phi_values = []
        ts_proj_values = []
        pathways_phi = []
        pathways_proj = []
        for row in matching_ts_dict:
            if row[HSP_LM1] == hemi_points or row[HSP_LM2] == hemi_points:
                lm1_phi = float(row[LMIRC_p1])
                lm2_phi = float(row[LMIRC_p2])
                tso_phi = float(row[THETA])

                lm1_proj = np.sin(math.radians(float(row[LMIRC_t1])))
                lm2_proj = np.sin(math.radians(float(row[LMIRC_t2])))
                tso_proj = np.sin(math.radians(float(row[THETA])))

                lm_phi_values.append(lm1_phi)
                lm_phi_values.append(lm2_phi)
                lm_proj_values.append(lm1_proj)
                lm_proj_values.append(lm2_proj)

                ts_phi_values.append(tso_phi)
                ts_proj_values.append(tso_proj)

                pathways_phi.append(lm1_phi)
                pathways_proj.append('')

                pathways_phi.append(lm1_phi)
                pathways_proj.append(lm1_proj)
                pathways_phi.append(tso_phi)
                pathways_proj.append(tso_proj)

                pathways_phi.append(lm2_phi)
                pathways_proj.append('')

                pathways_phi.append(lm2_phi)
                pathways_proj.append(lm2_proj)
                pathways_phi.append(tso_phi)
                pathways_proj.append(tso_proj)

        polar_dict[str('lm_phi_values_') + str(hemi_points)] = lm_phi_values
        polar_dict[str('lm_proj_values_') + str(hemi_points)] = lm_proj_values
        polar_dict[str('ts_phi_values_') + str(hemi_points)] = ts_phi_values
        polar_dict[str('ts_proj_values_') + str(hemi_points)] = ts_proj_values
        polar_dict[str('pathways_phi_') + str(hemi_points)] = pathways_phi
        polar_dict[str('pathways_proj_') + str(hemi_points)] = pathways_proj

    return polar_dict


def create_datadict(pathway_phi, pathway_theta, ts_phi, ts_theta, lm_phi, lm_theta):
    """
    This script simply creates the information for Igor to plot the phi and theta values associated with the pathways,
    the TS, and the LMs.

    :param pathway_phi: list of values for the phi pathways
    :param pathway_theta: list of values for the theta pathways
    :param ts_phi: list of values for the phi ts
    :param ts_theta: list of values for the theta ts
    :param lm_phi: list of values for the phi lm
    :param lm_theta: list of values for the theta lm
    :return:
    """
    data_dict = {}

    data_dict['compare_path_phi'] = pathway_phi
    data_dict['compare_path_theta'] = pathway_theta
    data_dict['compare_ts_phi'] = ts_phi
    data_dict['compare_ts_theta'] = ts_theta
    data_dict['compare_lm_phi'] = lm_phi
    data_dict['compare_lm_theta'] = lm_theta

    return data_dict


def comparing_LM_structures_arc_length(lm_dict, reference_lm):

    new_lm_dict = []
    hsp_puckering = []

    for row in lm_dict:
        p1 = float(row[PHI])
        t1 = float(row[THETA])
        arc_length_dict = {}
        for key, key_val in reference_lm.items():
            p2 = float(key_val[MPHI])
            t2 = float(key_val[MTHETA])
            arc_length_dict[key] = arc_length_calculator(p1, t1, p2, t2)

        top_difference = (sorted(arc_length_dict, key=arc_length_dict.get, reverse=False)[:1])

        if arc_length_dict[top_difference[0]] > TOL_ARC_LENGTH_CROSS:
            row[GID] = MISSING
        elif arc_length_dict[top_difference[0]] < TOL_ARC_LENGTH_CROSS:
            row[GID] = top_difference[0]
            if top_difference[0] not in hsp_puckering:
                hsp_puckering.append(top_difference[0])

        new_lm_dict.append(row)

    hsp_puckering.sort()

    if len(new_lm_dict) != len(lm_dict):
        print('The length of the new TS dict does not equal the length of the old TS dict.')

    low_energy_value = 100000000000

    for row in new_lm_dict:
        row_filename = row[FILE_NAME]
        if float(row[FREQ]) < 0:
            print('ERROR - a TS Freq is less than 0.')
        elif float(row[FREQ]) > 0:
            lm_dict.append(row)
            if float(row[GIBBS]) < low_energy_value:
                low_energy_value = float(row[GIBBS])

    for lm_row in new_lm_dict:
        lm_row[GIBBS] = (float(lm_row[GIBBS]) - low_energy_value) * HARTREE_TO_KCALMOL


    return new_lm_dict, hsp_puckering


def boltzmann_weighting_group(low_energy_job_dict, qm_method):
    list_groupings = []
    isolation_dict = {}
    dict_of_dict = {}
    group_total_weight_gibbs = {}
    contribution_dict = {}

    for row in low_energy_job_dict:
        row_grouping = row[GID]
        if row_grouping != MISSING:
            row_filename = row[FILE_NAME]
            dict_of_dict[row_filename] = row
            if row_grouping in list_groupings:
                isolation_dict[row_grouping].append(row_filename)
            elif row_grouping not in list_groupings:
                list_groupings.append(row_grouping)
                isolation_dict[row_grouping] = []
                isolation_dict[row_grouping].append(row_filename)
                group_total_weight_gibbs[row_grouping] = float(0)
                contribution_dict[row_grouping] = float(0)

    for group_key in isolation_dict.keys():
        if group_key != MISSING:
            for group_file in isolation_dict[group_key]:
                for main_file in low_energy_job_dict:
                    if group_file == main_file[FILE_NAME]:
                        group_type = main_file[GID]
                        try:
                            gibbs_energy = float(main_file[GIBBS])
                            enth_energy = float(main_file[ENTH])
                            weight_gibbs = math.exp(-gibbs_energy / (DEFAULT_TEMPERATURE * K_B))
                            weight_enth = math.exp(-enth_energy / (DEFAULT_TEMPERATURE * K_B))
                            main_file[WEIGHT_GIBBS] = weight_gibbs
                            main_file[WEIGHT_ENTH] = weight_enth
                        finally:
                            group_total_weight_gibbs[group_type] += weight_gibbs

    for group_key in isolation_dict.keys():
        if group_key != 'NONE':
            total_weight = group_total_weight_gibbs[group_key]
            for main_file in low_energy_job_dict:
                if main_file[GID] == group_key:
                    contribution_dict[group_key] = \
                        round(contribution_dict[group_key] + (main_file[WEIGHT_GIBBS] / total_weight)
                              * float(main_file[GIBBS]), 2)

    return contribution_dict, qm_method


def grouping_and_weighting_TS(low_energy_job_dict, qm_method):


    file_dict = {}
    TS_group_dict = {}
    initial = None

    for row in low_energy_job_dict:
        file_dict[row[FILE_NAME]] = row
        hsp_ts_group_num = row[HSP_TS]
        if initial is None:
            TS_group_dict[hsp_ts_group_num] = [row[FILE_NAME]]
            initial = 'good'
        else:
            if hsp_ts_group_num not in TS_group_dict.keys():
                TS_group_dict[hsp_ts_group_num] = [row[FILE_NAME]]
            elif hsp_ts_group_num in TS_group_dict.keys():
                TS_group_dict[hsp_ts_group_num].append(row[FILE_NAME])

    count = -1
    unique_ts_pathway = {}
    for group_keys, group_files in TS_group_dict.items():
        match_status = {}
        status = 'good'
        if len(group_files) == 1:
            count += 1
            # unique_ts_pathway[str('pathway_' + str(count).rjust(2, '0'))] = [group_files[0]]
            unique_ts_pathway[group_keys] = [group_files[0]]
        elif len(group_files) > 1:
            for i in range(0, len(group_files)):
                for j in range(i + 1, len(group_files)):
                    if file_dict[group_files[i]][HSP_TS] != file_dict[group_files[j]][HSP_TS]:
                        print('SOMETHING IS WRONG HERE!!')
                    elif file_dict[group_files[i]][HSP_TS] == file_dict[group_files[j]][HSP_TS]:
                        i_lm1 = file_dict[group_files[i]][HSP_LM1]
                        i_lm2 = file_dict[group_files[i]][HSP_LM2]
                        j_lm1 = file_dict[group_files[j]][HSP_LM1]
                        j_lm2 = file_dict[group_files[j]][HSP_LM2]
                        if i_lm1 == j_lm1 and i_lm2 == j_lm2:
                            match_status[str(str(i) + '-' + str(j))] = 'match'
                        else:
                            match_status[str(str(i) + '-' + str(j))] = 'missing'
                            status = 'bad'

            if status != 'bad':
                count += 1
                # unique_ts_pathway[str('pathway_' + str(count).rjust(2, '0'))] = group_files
                unique_ts_pathway[group_keys] = group_files

                # # # STARTS THE BOLTZMANN WEIGHTING FOR THE TS PATHWAYS # # #

    group_total_weight_gibbs = {}
    contribution_dict = {}

    for unique_group in unique_ts_pathway.keys():
        group_total_weight_gibbs[unique_group] = float(0)
        contribution_dict[unique_group] = float(0)

    for group_key in unique_ts_pathway.keys():
        if group_key != MISSING:
            for group_file in unique_ts_pathway[group_key]:
                for main_file in low_energy_job_dict:
                    if group_file == main_file[FILE_NAME]:
                        try:
                            gibbs_energy = float(main_file[GIBBS])
                            enth_energy = float(main_file[ENTH])
                            weight_gibbs = math.exp(-gibbs_energy / (DEFAULT_TEMPERATURE * K_B))
                            weight_enth = math.exp(-enth_energy / (DEFAULT_TEMPERATURE * K_B))
                            main_file[WEIGHT_GIBBS] = weight_gibbs
                            main_file[WEIGHT_ENTH] = weight_enth
                        finally:
                            group_total_weight_gibbs[group_key] += weight_gibbs

    for pathway_key in unique_ts_pathway.keys():
        if pathway_key != 'NONE':
            total_weight = group_total_weight_gibbs[pathway_key]
            for main_file in unique_ts_pathway[pathway_key]:
                contribution_dict[pathway_key] = round(
                    contribution_dict[pathway_key] + (file_dict[main_file][WEIGHT_GIBBS] / total_weight)
                    * float(file_dict[main_file][GIBBS]), 2)

    return contribution_dict, unique_ts_pathway


def job_id(file_name):
    file_parts = file_name.split('-')

    if file_parts[0] != 'z_dataset':
        status = 'error'

    job_type = file_parts[2]

    method = file_parts[3].split('.')[0]

    return job_type, method


def generating_lm_structures(dict):

    data_dict = {}
    lm_phi_match = []
    lm_theta_match = []
    lm_phi_missi = []
    lm_theta_missi = []


    for row in dict:
        if row[GID] is 'missing':
            lm_phi_missi.append(row[PHI])
            lm_theta_missi.append(row[THETA])
        else:
            lm_phi_match.append(row[PHI])
            lm_theta_match.append(row[THETA])

    data_dict['lm_match_phi'] = lm_phi_match
    data_dict['lm_match_theta'] = lm_theta_match
    data_dict['lm_missing_phi'] = lm_phi_missi
    data_dict['lm_mising_theta'] = lm_theta_missi

    return data_dict


def rmsd_calculator(rmsd_dict, lm_dict, ts_dict):
    """

    Performs all of the RMSD between HSP REFERENCE grouping and the method in question.

    :param rmsd_dict: the dict containing all of the relative Gibbs free energies for the group
    :param lm_dict: the REFERENCE HSP TS group
    :param ts_dict: the REFERENCE HSP TS group
    :return: the RMSD information for each method-job type combination
    """

    lm_rmsd_val = {}
    ts_rmsd_val = {}

    overall_dict = {}

    for rmsd_job_id, rmsd_data in rmsd_dict.items():
        rmsd_info = rmsd_job_id.split('-')
        diff_squared = 0
        ind_dict = {}
        max_diff = -10
        if rmsd_info[0] in 'TS':
            for rmsd_key, rmsd_energy in rmsd_data.items():
                for hsp_key in lm_dict.keys():
                    if rmsd_key == hsp_key:
                        diff_squared += math.pow(float(rmsd_energy) - float(lm_dict[hsp_key][WEIGHT_GIBBS]), 2)
                        diff = abs(float(rmsd_energy) - float(lm_dict[hsp_key][WEIGHT_GIBBS]))
                        if diff > max_diff:
                            max_diff = diff
                        break
            rmsd_val = math.sqrt(diff_squared/len(rmsd_data))
            ind_dict['RMSD val'] = round(rmsd_val, 1)
            ind_dict['diff_squared'] = round(diff_squared, 1)
            ind_dict['match number'] = len(rmsd_data)
            ind_dict['max_diff'] = round(max_diff, 1)

            overall_dict[rmsd_job_id] = ind_dict

        elif rmsd_info[0] in 'TS':
            pass
            rmsd_info = rmsd_job_id.split('-')
            diff_squared = 0
            for rmsd_key, rmsd_energy in rmsd_data.items():
                for hsp_key in ts_dict.keys():
                    if rmsd_key == hsp_key:
                        diff_squared += math.pow(float(rmsd_energy) - float(ts_dict[hsp_key][WEIGHT_GIBBS]), 2)
                        diff = abs(float(rmsd_energy) - float(ts_dict[hsp_key][WEIGHT_GIBBS]))
                        if diff > max_diff:
                            max_diff = diff
                        break
            rmsd_val = math.sqrt(diff_squared/len(rmsd_data))
            ind_dict['RMSD val'] = round(rmsd_val, 1)
            ind_dict['diff_squared'] = round(diff_squared, 1)
            ind_dict['match number'] = len(rmsd_data)
            ind_dict['max_diff'] = round(max_diff, 1)

            overall_dict[rmsd_job_id] = ind_dict


    return overall_dict


def writer_rmsd_information(overall_dict, dir_data):
    """
    This script writes the information so things can easily be manipulated into Igor for further analysis.

    :param overall_dict: the RMSD information for each method-job type combination
    :param dir_data: the location to write the output file to
    :return: outputs a file containing all of the RMSD information for Igor
    """

    output_filename_ = create_out_fname('igor_overall_rmsd', base_dir=dir_data, ext='.csv')

    list_method = []

    for keys in overall_dict.keys():
        info = keys.split('-')
        if info[1] not in list_method:
            list_method.append(info[1])

    list_method.sort()

    Titles = ["rmsd_method", "rmsd_lm", "rmsd_ts", "rmsd_total", "rmsd_max_diff"]

    with open(output_filename_, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([Titles])
        list_info = []

        for method_eval in list_method:
            max_diff = 0
            total_diff = 0
            match_count = 0
            job_complete = []
            for data_key, data_val in overall_dict.items():
                if data_key.split('-')[1] == method_eval and data_key.split('-')[0] not in job_complete:
                    job_complete.append(data_key.split('-')[0])

                    if data_key.split('-')[0] == 'TS':
                        rmsd_lm = data_val['RMSD val']
                        total_diff += data_val['diff_squared']
                        match_count += data_val['match number']
                        if data_val['max_diff'] > max_diff:
                            max_diff = data_val['max_diff']
                    elif data_key.split('-')[0] == 'TS':
                        rmsd_ts = data_val['RMSD val']
                        total_diff += data_val['diff_squared']
                        match_count += data_val['match number']
                        if data_val['max_diff'] > max_diff:
                            max_diff = data_val['max_diff']

            total_rmsd = round(math.sqrt((total_diff/match_count)),1)

            list_info.append([method_eval, rmsd_lm, rmsd_ts, total_rmsd, max_diff])


        writer.writerows(list_info)

    return

# # # # OLDER FUNCTIONS # # # #
def comparing_across_methods(method_dict, reference_dict, arc_tol=TOL_ARC_LENGTH_CROSS):
    """
    This script compared the structures generated from one particular method to the REFERENCE set of structures from HSP
    :param method_dict: the list of dicts for a particular method
    :param reference_dict: the REFERENCE dict to compare the puckers too
    :param arc_tol: the tolerance for arc length (if below tolerance, the structures are then grouped together)
    :return: updated_method_dict (contains the grouping solely based on arc length)
    """

    arc_length_key_dict = {}
    grouping_dict = {}
    updated_method_dict = []
    group_file_dict = {}
    ungrouped_files = []

    for method_row in method_dict:
        p1 = float(method_row[PHI])
        t1 = float(method_row[THETA])
        for reference_key in reference_dict.keys():
            p2 = reference_dict[reference_key][MPHI]
            t2 = reference_dict[reference_key][MTHETA]

            arc_length = arc_length_calculator(p1, t1, p2, t2)
            arc_length_key_dict[reference_key] = arc_length

        top_difference = (sorted(arc_length_key_dict, key=arc_length_key_dict.get, reverse=False)[:1])

        for row in top_difference:
            if arc_length_key_dict[row] < arc_tol:
                method_row[GID] = row
                if row not in group_file_dict.keys():
                    group_file_dict[row] = [method_row[FILE_NAME]]
                else:
                    group_file_dict[row].append(method_row[FILE_NAME])
            else:
                method_row[GID] = 'NONE'
                ungrouped_files.append(method_row[FILE_NAME])

        updated_method_dict.append(method_row)

    return updated_method_dict, group_file_dict, ungrouped_files


def sorting_for_matching_values(updated_method_dict, print_status='off'):
    """
    This script is designed to sort through the matching values and return the phi and theta values for the
    structures that are good (grouped) and ungrouped.
    :param updated_method_dict: the updated method dict (contains the group pucker information)
    :param print_status: the status of whether or not you want to print the out CP params
    :return:
    """
    phi_values_good = []
    theta_values_good = []
    phi_values_ungrouped = []
    theta_values_ungrouped = []

    for row in updated_method_dict:
        if row[GID] is not 'NONE':
            phi_values_good.append(row['phi'])
            theta_values_good.append(row['theta'])
        else:
            phi_values_ungrouped.append(row['phi'])
            theta_values_ungrouped.append(row['theta'])

    if print_status != 'off':
        print(phi_values_good)
        print(theta_values_good)
        print(phi_values_ungrouped)
        print(theta_values_ungrouped)

    return phi_values_good, theta_values_good, phi_values_ungrouped, theta_values_ungrouped


def modifying_contribution_dict(contribution_dict, reference_groups):
    final_contribution_dict = {}

    for contr_key in contribution_dict.keys():
        for reference_key in reference_groups.keys():
            if contr_key in reference_key:
                key = str(reference_key + ' (' + str(reference_groups[reference_key][CPK][0]) + ')')
                final_contribution_dict[key] = contribution_dict[contr_key]

    return final_contribution_dict


def writing_xlsx_files(lm_table_dict, ts_table_dict, output_filename):
    """ utilizes panda dataframes to write the local min and transition state dict of dicts

    :param lm_table_dict: dictionary corresponding to the local mins
    :param ts_table_dict: dictional corresponding to the transition state structures
    :param output_filename: output filename for the excel file
    :return: excel file with the required information
    """

    df_lm = pd.DataFrame(lm_table_dict)  # , index=LIST_PUCKER)
    df_ts = pd.DataFrame(ts_table_dict)  # , index=LIST_PUCKER)
    writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')
    df_lm.to_excel(writer, sheet_name='local min')
    df_ts.to_excel(writer, sheet_name='transition state')

    workbook = writer.book

    format_lm = workbook.add_format({'font_color': '#008000'})
    format_ts = workbook.add_format({'font_color': '#4F81BD'})

    worksheet_lm = writer.sheets['local min']
    worksheet_ts = writer.sheets['transition state']

    # TODO: figure out why my conditional formatting is no longer working
    worksheet_lm.conditional_format('B2:P39', {'type': 'cell',
                                               'criteria': '>=',
                                               'value': 50,
                                               'format': format_lm})
    worksheet_ts.conditional_format('B2:P39', {'type': 'cell',
                                               'criteria': '>=',
                                               'value': 50,
                                               'format': format_ts})

    writer.save()
    writer.close()

    return


def writing_csv_files(lm_table_dict, ts_table_dict, molecule, sum_file_location):
    """

    :param lm_table_dict:  dict of dicts containing the local min information
    :param ts_table_dict: dict of dicts containing the TS information
    :param molecule: the molecule currently being studied
    :param sum_file_location: the loation of the file
    :return: two csv files (one for the local min and one for the TS) containing the boltzmann weighted energies
    """

    prefix_lm = 'a_csv_lm_' + str(molecule)
    prefix_ts = 'a_csv_ts_' + str(molecule)

    path_lm = create_out_fname(sum_file_location, prefix=prefix_lm, remove_prefix='a_list_csv_files', ext='.csv')
    path_ts = create_out_fname(sum_file_location, prefix=prefix_ts, remove_prefix='a_list_csv_files', ext='.csv')

    df_lm = pd.DataFrame(lm_table_dict)  # , index=LIST_PUCKER)
    df_ts = pd.DataFrame(ts_table_dict)  # , index=LIST_PUCKER)

    df_lm.to_csv(path_lm)  # , index=LIST_PUCKER)
    df_ts.to_csv(path_ts)  # , index=LIST_PUCKER)


## Command Line Parse ##

def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="The gen_puck_table.py script is designed to combine hartree output "
                                                 "files to compare different properties across different levels of "
                                                 "theory. The hartree input files for a variety of levels of theory "
                                                 "are combined to produce a new data table.")

    parser.add_argument('-s', "--sum_file", help="List of csv files to read.", default=None)
    parser.add_argument('-d', "--dir_hartree", help="The directory where the hartree files can be found.",
                        default=None)
    parser.add_argument('-p', "--pattern", help="The file pattern you are looking for (example: '.csv').",
                        default=None)
    parser.add_argument('-m', "--molecule", help="The type of molecule that is currently being studied")
    parser.add_argument('-c', "--ccsdt", help="The CCSD(T) file for the molecule being studied",
                        default=None)

    args = None
    try:
        args = parser.parse_args(argv)
        if args.sum_file is None:
            raise InvalidDataError("Input files are required. Missing hartree input or two-file inputs")
        elif not os.path.isfile(args.sum_file):
            raise IOError("Could not find specified hartree summary file: {}".format(args.sum_file))
        # Finally, if the summary file is there, and there is no dir_xyz provided
        if args.dir_hartree is None:
            args.dir_hartree = os.path.dirname(args.sum_file)
        # if a  dir_xyz is provided, ensure valid
        elif not os.path.isdir(args.dir_hartree):
            raise InvalidDataError("Invalid path provided for '{}': ".format('-d, --dir_hartree', args.dir_hartree))

    except (KeyError, InvalidDataError) as e:
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    except IOError as e:
        warning(e)
        parser.print_help()
        return args, IO_ERROR
    except (ValueError, SystemExit) as e:
        if e.message == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR

    return args, GOOD_RET


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
        lm_level_dict = {}
        ts_level_dict = {}
        overall_level_dict = {}

        lm_dict = {}
        ts_dict = {}
        if 'bxyl' == args.molecule:
            HSP_LM_REFERENCE = BXYL_LM_PARAMS
            HSP_TS_REFERENCE = BXYL_TS_PARAMS
            for row in BXYL_LM_PARAMS.keys():
                lm_key = str(str(row) + ' (' + str(BXYL_LM_PARAMS[row][CPK][0]) + ')')
                lm_dict[lm_key] = round(BXYL_LM_PARAMS[row]['Boltz Weight Gibbs'], 2)

            for row in BXYL_TS_PARAMS.keys():
                ts_key = str(str(row) + ' (' + str(BXYL_TS_PARAMS[row][CPK][0]) + ')')
                ts_dict[ts_key] = round(BXYL_TS_PARAMS[row]['Boltz Weight Gibbs'], 2)

        lm_level_dict['CCSDT' + "-lm"] = lm_dict
        ts_level_dict['CCSDT' + "-ts"] = ts_dict

        with open(args.sum_file) as f:

            RMSD_data = {}

            for csv_file_read_newline in f:
                csv_file_read = csv_file_read_newline.strip("\n")

                # creates a dictionary of all of the meaninful information
                method_dict = read_csv_to_dict(os.path.join(args.dir_hartree, csv_file_read), mode='r')

                job_type, method = job_id(csv_file_read)

                if job_type == 'TS':

                    # splits the information based on TS and TS structures
                    lm_irc_dict, ts_dict = separating_TS_and_IRC_information(method_dict)

                    # compares all of the TS structures (contains the HSP reference TS that is within tolerance values)
                    new_ts_dict = comparing_TS_structures_arc_length(ts_dict, HSP_TS_REFERENCE)

                    # compares different TS to find whether or not the pathways are included
                    dict_one, matching_ts_dict, missing_ts_dict = comparing_TS_pathways(new_ts_dict, lm_irc_dict,
                                                                                        HSP_TS_REFERENCE,
                                                                                        HSP_LM_REFERENCE)

                    # generates the information associated with each pathway
                    data_dict = generate_matching_ts_dict_pathways_full(matching_ts_dict)

                    # performs the boltzmann weighting for the TS
                    contribution_dict, unique_ts_pathway = grouping_and_weighting_TS(matching_ts_dict, method)

                    # output the pathways for the local min and TS structures
                    output_filename_pathway = create_out_fname('igor_info_' + job_type + '_' + str(method), base_dir=args.dir_hartree, ext='.csv')
                    write_file_data_dict(data_dict, output_filename_pathway)

                if job_type == 'TS':

                    # assigns the local min groups for each of the unique local min structures
                    new_lm_dict, hsp_puckering = comparing_LM_structures_arc_length(method_dict, HSP_LM_REFERENCE)

                    # performs the Boltzmann weight for the local min structures
                    contribution_dict, qm_method = boltzmann_weighting_group(new_lm_dict, method)

                    # generates the information associated with each pathway
                    data_dict = generating_lm_structures(new_lm_dict)

                    # outputs the phi and theta files for matching and missing local min structures
                    output_filename_pathway = create_out_fname('igor_info_' + job_type + '_' + str(method), base_dir=args.dir_hartree, ext='.csv')
                    write_file_data_dict(data_dict, output_filename_pathway)

                RMSD_data[str(job_type + '-' + method)] = contribution_dict

        rmsd_overall_dict = rmsd_calculator(RMSD_data, HSP_LM_REFERENCE, HSP_TS_REFERENCE)


        writer_rmsd_information(rmsd_overall_dict, args.dir_hartree)


    except IOError as e:
        warning(e)
        return IO_ERROR
    except (InvalidDataError, KeyError) as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)

