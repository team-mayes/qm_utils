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
import pandas as pd
import math

from qm_utils.pucker_table import read_hartree_files_lowest_energy, sorting_job_types

from qm_utils.qm_common import (GOOD_RET, create_out_fname, warning, IO_ERROR, InvalidDataError, INVALID_DATA,
                                INPUT_ERROR, arc_length_calculator)

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
TOL_ARC_LENGTH = 0.1
TOL_ARC_LENGTH_CROSS = 0.2
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

# # Updated CP Params # #

BXYL_LM_PARAMS ={
  'group_00': {'theta': [91.6], 'mean phi': 113.5, 'mean theta': 91.6, 'G298 (Hartrees)': 7.9, 'phi': [113.5], 'closest group puck': ['25b'], 'files': ['puck11']},
  'group_01': {'theta': [90.8], 'mean phi': 339.4, 'mean theta': 90.8, 'G298 (Hartrees)': 8.6, 'phi': [339.4], 'closest group puck': ['os2'], 'files': ['puck25']},
  'group_02': {'theta': [177.9, 177.8, 176.6, 177.7], 'mean phi': 117.175, 'mean theta': 177.5, 'G298 (Hartrees)': 2.45, 'phi': [327.3, 30.0, 66.5, 44.9], 'closest group puck': ['1c4'], 'files': ['puck2', 'puck3', 'puck4', 'puck5']},
  'group_03': {'theta': [86.8], 'mean phi': 155.5, 'mean theta': 86.8, 'G298 (Hartrees)': 3.9, 'phi': [155.5], 'closest group puck': ['2so'], 'files': ['puck12']},
  'group_04': {'theta': [91.8], 'mean phi': 249.1, 'mean theta': 91.8, 'G298 (Hartrees)': 11.0, 'phi': [249.1], 'closest group puck': ['14b'], 'files': ['puck1']},
  'group_05': {'theta': [2.1, 1.3], 'mean phi': 33.0, 'mean theta': 1.7, 'G298 (Hartrees)': 0.16, 'phi': [14.7, 51.3], 'closest group puck': ['4c1'], 'files': ['puck15', 'puck16']},
  'group_06': {'theta': [88.1, 92.3], 'mean phi': 95.55, 'mean theta': 90.2, 'G298 (Hartrees)': 8.06, 'phi': [96.7, 94.4], 'closest group puck': ['5s1'], 'files': ['puck17', 'puck18']},
  'group_07': {'theta': [92.4, 92.5], 'mean phi': 328.55, 'mean theta': 92.45, 'G298 (Hartrees)': 8.36, 'phi': [328.3, 328.8], 'closest group puck': ['os2'], 'files': ['puck24', 'puck26']},
  'group_08': {'theta': [89.9, 90.8], 'mean phi': 18.0, 'mean theta': 90.35, 'G298 (Hartrees)': 7.82, 'phi': [19.0, 17.0], 'closest group puck': ['3s1'], 'files': ['puck13', 'puck14']},
  'group_09': {'theta': [89.6, 89.7], 'mean phi': 6.3, 'mean theta': 89.65, 'G298 (Hartrees)': 7.88, 'phi': [5.5, 7.1], 'closest group puck': ['3ob'], 'files': ['puck22', 'puck23']},
  'group_10': {'theta': [86.2, 86.2], 'mean phi': 195.2, 'mean theta': 86.2, 'G298 (Hartrees)': 5.95, 'phi': [195.7, 194.7], 'closest group puck': ['1s3'], 'files': ['puck6', 'puck21']},
  'group_11': {'theta': [92.6], 'mean phi': 264.8, 'mean theta': 92.6, 'G298 (Hartrees)': 7.9, 'phi': [264.8], 'closest group puck': ['1s5'], 'files': ['puck9']},
  'group_12': {'theta': [89.4, 89.2, 90.8], 'mean phi': 274.966667, 'mean theta': 89.8, 'G298 (Hartrees)': 7.47, 'phi': [274.2, 272.4, 278.3], 'closest group puck': ['1s5'], 'files': ['puck7', 'puck8', 'puck10']},
  'group_13': {'theta': [86.3, 86.8], 'mean phi': 47.0, 'mean theta': 86.55, 'G298 (Hartrees)': 9.58, 'phi': [45.2, 48.8], 'closest group puck': ['b14'], 'files': ['puck19', 'puck20']}
 }


BXYL_TS_PARAMS = {'group_08': {'closest group puck': ['1h2'], 'G298 (Hartrees)': 12.02, 'phi': [255.0, 257.0], 'files': ['puck32', 'puck38'], 'mean theta': 121.05, 'theta': [121.3, 120.8], 'mean phi': 256.0}, 'group_17': {'closest group puck': ['14b'], 'G298 (Hartrees)': 12.6, 'phi': [237.5], 'files': ['puck29'], 'mean theta': 94.0, 'theta': [94.0], 'mean phi': 237.5}, 'group_02': {'closest group puck': ['e1'], 'G298 (Hartrees)': 14.62, 'phi': [46.0, 48.4, 49.7, 43.0], 'files': ['puck74', 'puck75', 'puck76', 'puck87'], 'mean theta': 63.375, 'theta': [63.1, 62.2, 63.2, 65.0], 'mean phi': 46.775}, 'group_31': {'closest group puck': ['4h3'], 'G298 (Hartrees)': 11.39, 'phi': [196.1, 199.4], 'files': ['puck55', 'puck56'], 'mean theta': 54.25, 'theta': [54.6, 53.9], 'mean phi': 197.75}, 'group_23': {'closest group puck': ['25b'], 'G298 (Hartrees)': 8.9, 'phi': [134.4], 'files': ['puck41'], 'mean theta': 86.4, 'theta': [86.4], 'mean phi': 134.4}, 'group_16': {'closest group puck': ['os2'], 'G298 (Hartrees)': 10.35, 'phi': [314.5, 315.7], 'files': ['puck69', 'puck90'], 'mean theta': 90.45, 'theta': [90.3, 90.6], 'mean phi': 315.1}, 'group_28': {'closest group puck': ['b03'], 'G298 (Hartrees)': 10.5, 'phi': [184.1], 'files': ['puck73'], 'mean theta': 86.9, 'theta': [86.9], 'mean phi': 184.1}, 'group_15': {'closest group puck': ['oh5'], 'G298 (Hartrees)': 16.3, 'phi': [327.2], 'files': ['puck88'], 'mean theta': 63.6, 'theta': [63.6], 'mean phi': 327.2}, 'group_30': {'closest group puck': ['oh1'], 'G298 (Hartrees)': 14.6, 'phi': [31.6, 37.2, 38.3, 31.6, 29.3], 'files': ['puck82', 'puck83', 'puck85', 'puck86', 'puck84'], 'mean theta': 63.3, 'theta': [62.3, 61.6, 62.1, 65.0, 65.5], 'mean phi': 33.6}, 'group_11': {'closest group puck': ['5ho'], 'G298 (Hartrees)': 15.56, 'phi': [145.3, 142.7], 'files': ['puck66', 'puck67'], 'mean theta': 125.95, 'theta': [124.7, 127.2], 'mean phi': 144.0}, 'group_07': {'closest group puck': ['4h5'], 'G298 (Hartrees)': 14.6, 'phi': [280.4], 'files': ['puck57'], 'mean theta': 63.4, 'theta': [63.4], 'mean phi': 280.4}, 'group_20': {'closest group puck': ['3h4'], 'G298 (Hartrees)': 13.14, 'phi': [35.7, 35.6], 'files': ['puck51', 'puck52'], 'mean theta': 115.8, 'theta': [115.0, 116.6], 'mean phi': 35.65}, 'group_13': {'closest group puck': ['5e'], 'G298 (Hartrees)': 16.6, 'phi': [135.9], 'files': ['puck65'], 'mean theta': 118.4, 'theta': [118.4], 'mean phi': 135.9}, 'group_19': {'closest group puck': ['oh5'], 'G298 (Hartrees)': 16.4, 'phi': [333.4], 'files': ['puck89'], 'mean theta': 61.1, 'theta': [61.1], 'mean phi': 333.4}, 'group_26': {'closest group puck': ['e5'], 'G298 (Hartrees)': 15.3, 'phi': [290.4], 'files': ['puck78'], 'mean theta': 63.0, 'theta': [63.0], 'mean phi': 290.4}, 'group_00': {'closest group puck': ['2h3'], 'G298 (Hartrees)': 13.3, 'phi': [136.6], 'files': ['puck43'], 'mean theta': 60.0, 'theta': [60.0], 'mean phi': 136.6}, 'group_18': {'closest group puck': ['b25'], 'G298 (Hartrees)': 11.45, 'phi': [302.6, 303.2], 'files': ['puck70', 'puck71'], 'mean theta': 92.7, 'theta': [92.9, 92.5], 'mean phi': 302.9}, 'group_14': {'closest group puck': ['3h2'], 'G298 (Hartrees)': 12.77, 'phi': [330.9, 327.9, 332.3, 331.5, 328.3, 327.8, 331.6], 'files': ['puck44', 'puck45', 'puck46', 'puck47', 'puck48', 'puck49', 'puck50'], 'mean theta': 117.3, 'theta': [114.9, 116.5, 114.0, 115.6, 120.3, 119.8, 120.0], 'mean phi': 330.042857}, 'group_01': {'closest group puck': ['5e'], 'G298 (Hartrees)': 14.94, 'phi': [112.2, 113.3, 113.9], 'files': ['puck62', 'puck63', 'puck64'], 'mean theta': 125.6, 'theta': [124.7, 125.4, 126.7], 'mean phi': 113.133333}, 'group_04': {'closest group puck': ['5e'], 'G298 (Hartrees)': 12.1, 'phi': [126.3, 120.8, 120.6, 123.1], 'files': ['puck58', 'puck59', 'puck60', 'puck61'], 'mean theta': 124.675, 'theta': [125.2, 125.8, 126.7, 121.0], 'mean phi': 122.7}, 'group_10': {'closest group puck': ['b14'], 'G298 (Hartrees)': 13.0, 'phi': [59.3], 'files': ['puck68'], 'mean theta': 87.7, 'theta': [87.7], 'mean phi': 59.3}, 'group_32': {'closest group puck': ['4h3'], 'G298 (Hartrees)': 10.6, 'phi': [206.0], 'files': ['puck54'], 'mean theta': 54.6, 'theta': [54.6], 'mean phi': 206.0}, 'group_21': {'closest group puck': ['2h3'], 'G298 (Hartrees)': 10.2, 'phi': [148.6], 'files': ['puck42'], 'mean theta': 62.0, 'theta': [62.0], 'mean phi': 148.6}, 'group_03': {'closest group puck': ['b25'], 'G298 (Hartrees)': 12.3, 'phi': [291.7], 'files': ['puck72'], 'mean theta': 91.0, 'theta': [91.0], 'mean phi': 291.7}, 'group_09': {'closest group puck': ['3obD'], 'G298 (Hartrees)': 10.98, 'phi': [1.4, 5.8], 'files': ['puck79', 'puck80'], 'mean theta': 89.5, 'theta': [89.2, 89.8], 'mean phi': 3.6}, 'group_25': {'closest group puck': ['1e'], 'G298 (Hartrees)': 14.46, 'phi': [225.4, 228.8, 227.9, 231.5, 224.6], 'files': ['puck33', 'puck34', 'puck35', 'puck36', 'puck40'], 'mean theta': 124.58, 'theta': [123.6, 123.5, 126.4, 123.4, 126.0], 'mean phi': 227.64}, 'group_05': {'closest group puck': ['14b'], 'G298 (Hartrees)': 10.3, 'phi': [243.1], 'files': ['puck27'], 'mean theta': 84.1, 'theta': [84.1], 'mean phi': 243.1}, 'group_29': {'closest group puck': ['14b'], 'G298 (Hartrees)': 10.7, 'phi': [235.9, 237.7, 239.0], 'files': ['puck28', 'puck30', 'puck31'], 'mean theta': 86.466667, 'theta': [84.5, 86.6, 88.3], 'mean phi': 237.533333}, 'group_22': {'closest group puck': ['1h2'], 'G298 (Hartrees)': 13.5, 'phi': [280.0], 'files': ['puck37'], 'mean theta': 125.9, 'theta': [125.9], 'mean phi': 280.0}, 'group_27': {'closest group puck': ['3h4'], 'G298 (Hartrees)': 14.8, 'phi': [25.7], 'files': ['puck53'], 'mean theta': 117.3, 'theta': [117.3], 'mean phi': 25.7}, 'group_06': {'closest group puck': ['1h2'], 'G298 (Hartrees)': 13.6, 'phi': [264.5], 'files': ['puck39'], 'mean theta': 120.7, 'theta': [120.7], 'mean phi': 264.5}, 'group_12': {'closest group puck': ['e3'], 'G298 (Hartrees)': 8.6, 'phi': [174.1], 'files': ['puck77'], 'mean theta': 57.2, 'theta': [57.2], 'mean phi': 174.1}, 'group_24': {'closest group puck': ['oeD'], 'G298 (Hartrees)': 16.1, 'phi': [12.3], 'files': ['puck81'], 'mean theta': 61.6, 'theta': [61.6], 'mean phi': 12.3}}

# # Script Functions # #

def compute_rmsd_between_puckers(phi, theta, new_cp_params=None, q_val=1):
    """"""
    # Puckers in the middle of four puckers...
    # phi = 43.76
    # theta = 73.16

    # Coordinates in the middle of two puckers...


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

    return


def create_reference_cp_params(data_dict, arc_tol=TOL_ARC_LENGTH, print_status='off'):
    """
    This script created different puckering groups for further analyze based on arc length calculations. If the
    difference in arc length between two structures is small, then the structures are grouped together. Before, all
    structures were grouped together simply based on their CP pucking designations. Now, the grouping are completed by

    :param data_dict:
    :return:
    """
    structure_dict = {}
    ind_dict = {}

    for i in range(0, len(data_dict)):
        if i == 0:
            p1 = float(data_dict[i][PHI])
            t1 = float(data_dict[i][THETA])
            ind_dict[PHI] = [p1]
            ind_dict[THETA] = [t1]
            ind_dict['files'] = [data_dict[i][FILE_NAME]]
            ind_dict['mean phi'] = st.mean(ind_dict[PHI])
            ind_dict['mean theta'] = st.mean(ind_dict[THETA])
            structure_dict['group' + '_' + str(0)] = ind_dict
            pucker = data_dict[i][PUCKER]
        else:
            ind_dict = {}
            for j in range(0, len(structure_dict)):
                p1 = float(data_dict[i][PHI])
                t1 = float(data_dict[i][THETA])

                p2 = structure_dict['group' + '_' + str(j)]['mean phi']
                t2 = structure_dict['group' + '_' + str(j)]['mean theta']

                arc_length = arc_length_calculator(p1, t1, p2, t2)
                if arc_length < arc_tol:
                    structure_dict['group' + '_' + str(j)][PHI].append(p1)
                    structure_dict['group' + '_' + str(j)][THETA].append(t1)
                    structure_dict['group' + '_' + str(j)]['files'].append(data_dict[i][FILE_NAME])
                    structure_dict['group' + '_' + str(j)]['mean phi'] = round(
                        st.mean(structure_dict['group' + '_' + str(j)][PHI]), 6)
                    structure_dict['group' + '_' + str(j)]['mean theta'] = round(
                        st.mean(structure_dict['group' + '_' + str(j)][THETA]), 6)
                    break
                elif j == len(structure_dict) - 1:
                    ind_dict[PHI] = [p1]
                    ind_dict[THETA] = [t1]
                    ind_dict['files'] = [data_dict[i][FILE_NAME]]
                    ind_dict['mean phi'] = round(st.mean(ind_dict[PHI]), 6)
                    ind_dict['mean theta'] = round(st.mean(ind_dict[THETA]), 6)
                    structure_dict['group' + '_' + str(len(structure_dict))] = ind_dict


    list_redo_groups = []
    for s in range(0, len(data_dict)):
        arc_length_dict = {}
        p1 = float(data_dict[s][PHI])
        t1 = float(data_dict[s][THETA])
        for pucker_group_keys in structure_dict.keys():
            p2 = structure_dict[pucker_group_keys]['mean phi']
            t2 = structure_dict[pucker_group_keys]['mean theta']
            arc_length_dict[pucker_group_keys] = arc_length_calculator(p1, t1, p2, t2)
            if data_dict[s][FILE_NAME] in structure_dict[pucker_group_keys]['files']:
                assigned_group = pucker_group_keys

        top_difference = (sorted(arc_length_dict, key=arc_length_dict.get, reverse=False)[:2])

        if arc_length_dict[top_difference[0]] < arc_tol and arc_length_dict[top_difference[1]] < arc_tol:
            if top_difference[0] not in list_redo_groups:
                list_redo_groups.append(top_difference[0])
            elif top_difference[1] not in list_redo_groups:
                list_redo_groups.append(top_difference[1])

    list_redo_files = []
    list_redo_data = []
    for x in range(0, len(list_redo_groups)):
        hi = list_redo_groups[x]
        redo = structure_dict[list_redo_groups[x]]['files']
        for y in redo:
            list_redo_files.append(y)
            for row in data_dict:
                if y == row[FILE_NAME]:
                    list_redo_data.append(row)
        structure_dict.pop(list_redo_groups[x], None)


    mod_structure_dict = {}
    ind_dict = {}
    for m in range(0, len(list_redo_files)):
        if m == 0:
            p1 = float(list_redo_data[m][PHI])
            t1 = float(list_redo_data[m][THETA])
            ind_dict[PHI] = [p1]
            ind_dict[THETA] = [t1]
            ind_dict['files'] = [list_redo_data[m][FILE_NAME]]
            ind_dict['mean phi'] = st.mean(ind_dict[PHI])
            ind_dict['mean theta'] = st.mean(ind_dict[THETA])
            mod_structure_dict['group' + '_' + str(0)] = ind_dict
            pucker = data_dict[i][PUCKER]
        else:
            ind_dict = {}
            for n in range(0, len(mod_structure_dict)):
                p1 = float(list_redo_data[m][PHI])
                t1 = float(list_redo_data[m][THETA])

                p2 = mod_structure_dict['group' + '_' + str(n)]['mean phi']
                t2 = mod_structure_dict['group' + '_' + str(n)]['mean theta']

                arc_length = arc_length_calculator(p1, t1, p2, t2)
                if arc_length < arc_tol:
                    mod_structure_dict['group' + '_' + str(n)][PHI].append(p1)
                    mod_structure_dict['group' + '_' + str(n)][THETA].append(t1)
                    mod_structure_dict['group' + '_' + str(n)]['files'].append(list_redo_data[m][FILE_NAME])
                    mod_structure_dict['group' + '_' + str(n)]['mean phi'] = round(
                        st.mean(mod_structure_dict['group' + '_' + str(n)][PHI]), 6)
                    mod_structure_dict['group' + '_' + str(n)]['mean theta'] = round(
                        st.mean(mod_structure_dict['group' + '_' + str(n)][THETA]), 6)
                    break
                elif n == len(mod_structure_dict) - 1:
                    ind_dict[PHI] = [p1]
                    ind_dict[THETA] = [t1]
                    ind_dict['files'] = [list_redo_data[m][FILE_NAME]]
                    ind_dict['mean phi'] = round(st.mean(ind_dict[PHI]), 6)
                    ind_dict['mean theta'] = round(st.mean(ind_dict[THETA]), 6)
                    mod_structure_dict['group' + '_' + str(len(mod_structure_dict))] = ind_dict


    count = 0
    final_structure_dict = {}
    for struct_key in structure_dict.keys():
        if count < 10:
            new_key = 'group_0' + str(count)
        else:
            new_key = 'group_' + str(count)
        final_structure_dict[new_key] = structure_dict[struct_key]
        count += 1
    for struct_key in mod_structure_dict.keys():
        new_key = 'group_' + str(count)
        final_structure_dict[new_key] = mod_structure_dict[struct_key]
        count += 1


    phi_mean = []
    theta_mean = []
    for structure_key in final_structure_dict.keys():
        phi_mean.append(final_structure_dict[structure_key]['mean phi'])
        theta_mean.append(final_structure_dict[structure_key]['mean theta'])
        p1 = float(final_structure_dict[structure_key]['mean phi'])
        t1 = float(final_structure_dict[structure_key]['mean theta'])
        arc_length_key_dict = {}
        for row in CP_PARAMS:
            p2 = float(row['PHI'])
            t2 = float(row['THETA'])
            arc_length = arc_length_calculator(p1, t1, p2, t2)
            arc_length_key_dict[row[PUCKER]] = arc_length
            top_difference = (sorted(arc_length_key_dict, key=arc_length_key_dict.get, reverse=False)[:1])
            final_structure_dict[structure_key]['closest group puck'] = top_difference

    lowest_energy_dict = []
    for row in data_dict:
        for final_structure_keys in final_structure_dict.keys():
            for file in final_structure_dict[final_structure_keys]['files']:
                if row[FILE_NAME] == file:
                    row[GID] = final_structure_keys

        lowest_energy_dict.append(row)


    boltzmann, qm_method = boltzmann_weighting_group(lowest_energy_dict, 'CCSDT')

    for boltz_key in boltzmann.keys():
        for final_structure_keys in final_structure_dict.keys():
            if boltz_key == final_structure_keys:
                final_structure_dict[boltz_key][GIBBS] = boltzmann[final_structure_keys]

    if print_status != 'off':
        print(phi_mean)
        print(theta_mean)

    return final_structure_dict, phi_mean, theta_mean


def comparing_across_methods(method_dict, reference_dict, arc_tol=TOL_ARC_LENGTH_CROSS):
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


def boltzmann_weighting_group(low_energy_job_dict, qm_method):
    list_groupings = []
    isolation_dict = {}
    dict_of_dict = {}
    group_total_weight_gibbs = {}
    contribution_dict = {}

    for row in low_energy_job_dict:
        row_grouping = row[GID]
        if row_grouping != 'NONE':
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
        if group_key != 'NONE':
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

def modifying_contribution_dict(contribution_dict, reference_groups):

    final_contribution_dict = {}

    for contr_key in contribution_dict.keys():
        for reference_key in reference_groups.keys():
            if contr_key in reference_key:
                key = str(reference_key + ' (' + str(reference_groups[reference_key][CPK][0]) +')')
                final_contribution_dict[key] = contribution_dict[contr_key]

    return final_contribution_dict


def writing_xlsx_files(lm_table_dict, ts_table_dict, output_filename):
    """ utilizes panda dataframes to write the local min and transition state dict of dicts

    :param lm_table_dict: dictionary corresponding to the local mins
    :param ts_table_dict: dictional corresponding to the transition state structures
    :param output_filename: output filename for the excel file
    :return: excel file with the required information
    """

    df_lm = pd.DataFrame(lm_table_dict)#, index=LIST_PUCKER)
    df_ts = pd.DataFrame(ts_table_dict)#, index=LIST_PUCKER)
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

    df_lm = pd.DataFrame(lm_table_dict)#, index=LIST_PUCKER)
    df_ts = pd.DataFrame(ts_table_dict)#, index=LIST_PUCKER)

    df_lm.to_csv(path_lm)#, index=LIST_PUCKER)
    df_ts.to_csv(path_ts)#, index=LIST_PUCKER)


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
            for row in BXYL_LM_PARAMS.keys():
                lm_key = str(str(row) + ' (' + str(BXYL_LM_PARAMS[row][CPK][0]) + ')')
                lm_dict[lm_key] = round(BXYL_LM_PARAMS[row]['G298 (Hartrees)'],2)
            for row in BXYL_TS_PARAMS.keys():
                ts_key = str(str(row) + ' (' + str(BXYL_TS_PARAMS[row][CPK][0]) + ')')
                ts_dict[ts_key] = round(BXYL_TS_PARAMS[row]['G298 (Hartrees)'],2)

        lm_level_dict['CCSDT' + "-lm"] = lm_dict
        ts_level_dict['CCSDT' + "-ts"] = ts_dict

        with open(args.sum_file) as f:
            for csv_file_read_newline in f:
                csv_file_read = csv_file_read_newline.strip("\n")
                hartree_headers, lowest_energy_dict, qm_method = read_hartree_files_lowest_energy(csv_file_read,
                                                                                                  args.dir_hartree)
                lm_jobs, ts_jobs, qm_method = sorting_job_types(lowest_energy_dict, qm_method)

                lm_jobs_updated, lm_group_file_dict, lm_ungrouped_files = comparing_across_methods(lm_jobs,
                                                                                                   BXYL_LM_PARAMS)
                ts_jobs_updated, ts_group_file_dict, ts_ungrouped_files = comparing_across_methods(ts_jobs,
                                                                                                   BXYL_TS_PARAMS)

                contribution_dict_lm, qm_method = boltzmann_weighting_group(lm_jobs_updated, qm_method)
                contribution_dict_ts, qm_method = boltzmann_weighting_group(ts_jobs_updated, qm_method)

                final_contribution_dict_lm = modifying_contribution_dict(contribution_dict_lm, BXYL_LM_PARAMS)
                final_contribution_dict_ts = modifying_contribution_dict(contribution_dict_ts, BXYL_TS_PARAMS)

                lm_level_dict[qm_method + "-lm"] = final_contribution_dict_lm
                ts_level_dict[qm_method + "-ts"] = final_contribution_dict_ts
                overall_level_dict[qm_method + "-ts"] = final_contribution_dict_ts
                overall_level_dict[qm_method + "-lm"] = final_contribution_dict_lm

            prefix = 'a_table_lm-ts_' + str(args.molecule)

            list_f_name = create_out_fname(args.sum_file, prefix=prefix, remove_prefix='a_list_csv_files',
                                           base_dir=os.path.dirname(args.sum_file), ext='.xlsx')

            writing_csv_files(lm_level_dict, ts_level_dict, args.molecule, args.sum_file)
            writing_xlsx_files(lm_level_dict, ts_level_dict, list_f_name)

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
