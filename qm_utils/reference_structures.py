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

import itertools
import pandas as pd
import math

from qm_utils.structure_pairing import boltzmann_weighting_group

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


PLM1 = 'phi_lm1'
TLM1 = 'theta_lm1'
PTS = 'phi_ts'
TTS = 'theta_ts'
PLM2 = 'phi_lm2'
TLM2 = 'theta_lm2'
LM1_GROUP = 'group_lm1'
LM2_GROUP = 'group_lm2'



# # Default CP Params # #

CP_PARAMS = [{'PHI': '180', 'THETA': '180', 'Pucker': '1c4', 'Q': '0.57'},
             {'PHI': '180', 'THETA': '0', 'Pucker': '4c1', 'Q': '0.57'},
             {'PHI': '240', 'THETA': '90', 'Pucker': '14b', 'Q': '0.76'},
             {'PHI': '60', 'THETA': '90', 'Pucker': 'b14', 'Q': '0.76'},
             {'PHI': '120', 'THETA': '90', 'Pucker': '25b', 'Q': '0.76'},
             {'PHI': '300', 'THETA': '90', 'Pucker': 'b25', 'Q': '0.76'},
             {'PHI': '0', 'THETA': '90', 'Pucker': 'o3b', 'Q': '0.76'},
             {'PHI': '180', 'THETA': '90', 'Pucker': 'bo3', 'Q': '0.76'},
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
             {'PHI': '0', 'THETA': '55', 'Pucker': 'oe', 'Q': '0.45'},
             {'PHI': '360', 'THETA': '90', 'Pucker': '3ob', 'Q': '0.76'},
             {'PHI': '0', 'THETA': '125', 'Pucker': '3e', 'Q': '0.45'}]

# # Puckers # #

LIST_PUCKER = ['4c1',  'oe', 'oh1',  'e1', '2h1',  '2e', '2h3',  'e3', '4h3',  '4e', '4h5',  'e5', 'oh5', 'o3b', '3s1',
               'b14', '5s1', '25b', '2so', 'bo3', '1s3', '14b', '1s5', 'b25', 'os2',  '3e', '3h4',  'e4', '5h4',  '5e',
               '5ho',  'eo', '1ho',  '1e', '1h2',  'e2', '3h2', '1c4']

# # Script Functions # #

def local_min_reference_points(data_dict, arc_tol=TOL_ARC_LENGTH, print_status='off'):
    """
    This script generates the local min reference points by calculating groups based on
    arc length calculations between the respective points.

    :param data_dict: the local min data dict
    :param arc_tol: the tolerance for the arc length grouping
    :param print_status: whether or not to pring the print the phi and theta parameters for the reference group
    :return: a dict containing the local min reference groups and the phi and theta values
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


def assign_TS_localmin_reference_groups(ts_data_dict, lm_final_structure_dict):

    regroup_list = []
    update_ts_data_dict = []

    for row in ts_data_dict:
        p1_lm1 = float(row[PLM1])
        t1_lm1 = float(row[TLM1])
        p1_lm2 = float(row[PLM2])
        t1_lm2 = float(row[TLM2])
        arc_length_lm1 = {}
        arc_length_lm2 = {}
        for lm_final_key in lm_final_structure_dict.keys():
            p2 = float(lm_final_structure_dict[lm_final_key][MPHI])
            t2 = float(lm_final_structure_dict[lm_final_key][MTHETA])
            arc_length_lm1[lm_final_key] = arc_length_calculator(p1_lm1, t1_lm1, p2, t2)
            arc_length_lm2[lm_final_key] = arc_length_calculator(p1_lm2, t1_lm2, p2, t2)

        top_diff_lm1 = (sorted(arc_length_lm1, key=arc_length_lm1.get, reverse=False)[:1])
        top_diff_lm2 = (sorted(arc_length_lm2, key=arc_length_lm2.get, reverse=False)[:1])

        # if not arc_length_lm1[top_diff_lm1[0]] < TOL_ARC_LENGTH_CROSS:
        #     print('------------------------------------------')
        #     print('PLM1: {}\n'
        #           'TLM1: {}\n'
        #           '  P2: {}\n'
        #           '  T2: {}\n'
        #           ' ARC: {}\n'
        #           '------------------------------------------\n'.format(row[PLM1], row[TLM1],
        #                               round(lm_final_structure_dict[top_diff_lm1[0]][MPHI],2),
        #                               round(lm_final_structure_dict[top_diff_lm1[0]][MTHETA],2),
        #                               round(arc_length_lm1[top_diff_lm1[0]],2)))
        #
        #
        # elif not arc_length_lm2[top_diff_lm2[0]] < TOL_ARC_LENGTH_CROSS:
        #     print('------------------------------------------')
        #     print('PLM2: {}\n'
        #           'TLM2: {}\n'
        #           '  P2: {}\n'
        #           '  T2: {}\n'
        #           ' ARC: {}\n'
        #           '------------------------------------------\n'.format(row[PLM2], row[TLM2],
        #                                 round(lm_final_structure_dict[top_diff_lm2[0]][MPHI],2),
        #                                 round(lm_final_structure_dict[top_diff_lm2[0]][MTHETA],2),
        #                                 round(arc_length_lm2[top_diff_lm2[0]],2)))
        # else:

        row[LM1_GROUP] = top_diff_lm1[0]
        row[LM2_GROUP] = top_diff_lm2[0]

        update_ts_data_dict.append(row)

    return update_ts_data_dict


def transition_state_reference_points(updated_ts_data_dict, arc_tol=0.25):


    data_dict = updated_ts_data_dict
    ind_dict = {}
    structure_dict = {}

    for i in range(0, len(data_dict)):
        if i == 0:
            p1 = float(data_dict[i][PTS])
            t1 = float(data_dict[i][TTS])
            ind_dict[PHI] = [p1]
            ind_dict[THETA] = [t1]
            ind_dict['files'] = [data_dict[i][FILE_NAME]]
            ind_dict['mean phi'] = st.mean(ind_dict[PHI])
            ind_dict['mean theta'] = st.mean(ind_dict[THETA])
            ind_dict[LM1_GROUP] = [data_dict[i][LM1_GROUP]]
            ind_dict[LM2_GROUP] = [data_dict[i][LM2_GROUP]]
            structure_dict['group' + '_' + str(0) + str(0)] = ind_dict
            pucker = data_dict[i][PUCKER]
        else:
            ind_dict = {}
            for j in range(0, len(structure_dict)):
                j_real = str(j).rjust(2, '0')

                p1 = float(data_dict[i][PTS])
                t1 = float(data_dict[i][TTS])

                p2 = structure_dict['group' + '_' + str(j_real)]['mean phi']
                t2 = structure_dict['group' + '_' + str(j_real)]['mean theta']

                arc_length = arc_length_calculator(p1, t1, p2, t2)

                if arc_length < arc_tol:
                    if [data_dict[i][LM1_GROUP]] == structure_dict['group' + '_' + str(j_real)][LM1_GROUP] and structure_dict['group' + '_' + str(j_real)][LM2_GROUP] == [data_dict[i][LM2_GROUP]]:
                        structure_dict['group' + '_' + str(j_real)][PHI].append(p1)
                        structure_dict['group' + '_' + str(j_real)][THETA].append(t1)
                        structure_dict['group' + '_' + str(j_real)]['files'].append(data_dict[i][FILE_NAME])
                        structure_dict['group' + '_' + str(j_real)]['mean phi'] = round(
                            st.mean(structure_dict['group' + '_' + str(j_real)][PHI]), 6)
                        structure_dict['group' + '_' + str(j_real)]['mean theta'] = round(
                            st.mean(structure_dict['group' + '_' + str(j_real)][THETA]), 6)
                        break
                if j == len(structure_dict) - 1:
                    ind_dict[PHI] = [p1]
                    ind_dict[THETA] = [t1]
                    ind_dict['files'] = [data_dict[i][FILE_NAME]]
                    ind_dict['mean phi'] = round(st.mean(ind_dict[PHI]), 6)
                    ind_dict['mean theta'] = round(st.mean(ind_dict[THETA]), 6)
                    ind_dict[LM1_GROUP] = [data_dict[i][LM1_GROUP]]
                    ind_dict[LM2_GROUP] = [data_dict[i][LM2_GROUP]]
                    structure_dict['group' + '_' + str(len(structure_dict)).rjust(2, '0')] = ind_dict

    phi_mean = []
    theta_mean = []
    for structure_key in structure_dict.keys():
        phi_mean.append(structure_dict[structure_key]['mean phi'])
        theta_mean.append(structure_dict[structure_key]['mean theta'])
        p1 = float(structure_dict[structure_key]['mean phi'])
        t1 = float(structure_dict[structure_key]['mean theta'])
        arc_length_key_dict = {}
        for row in CP_PARAMS:
            p2 = float(row['PHI'])
            t2 = float(row['THETA'])
            arc_length = arc_length_calculator(p1, t1, p2, t2)
            arc_length_key_dict[row[PUCKER]] = arc_length
            top_difference = (sorted(arc_length_key_dict, key=arc_length_key_dict.get, reverse=False)[:1])
            structure_dict[structure_key]['closest group puck'] = top_difference

    return structure_dict, phi_mean, theta_mean


def analyze_ts_structure_dict(structure_dict):

    dict_same_groups = None
    list_same_groups = []

    for ts1_key in structure_dict.keys():
        for ts2_key in structure_dict.keys():
            if ts1_key != ts2_key:
                ts1_lm1 = structure_dict[ts1_key][LM1_GROUP]
                ts1_lm2 = structure_dict[ts1_key][LM2_GROUP]
                ts2_lm1 = structure_dict[ts2_key][LM1_GROUP]
                ts2_lm2 = structure_dict[ts2_key][LM2_GROUP]
                if ts1_lm1 == ts2_lm2 and ts1_lm2 == ts2_lm1:
                    p1 = structure_dict[ts1_key][MPHI]
                    t1 = structure_dict[ts1_key][MTHETA]
                    p2 = structure_dict[ts2_key][MPHI]
                    t2 = structure_dict[ts2_key][MTHETA]
                    arc_length = arc_length_calculator(p1, t1, p2, t2)
                    if arc_length < 0.2:
                        if dict_same_groups is None:
                            dict_same_groups = {}
                            dict_same_groups[ts1_key] = ts2_key
                            list_same_groups.append(ts1_key)
                            list_same_groups.append(ts2_key)
                        else:
                            assign = False
                            for dict_keys in dict_same_groups.keys():
                                if ts1_key == dict_keys and ts2_key == dict_same_groups[ts1_key]:
                                    assign = True
                                    break
                                elif ts2_key == dict_keys and ts1_key == dict_same_groups[ts2_key]:
                                    assign = True
                                    break
                            if assign is False:
                                dict_same_groups[ts1_key] = ts2_key
                                list_same_groups.append(ts1_key)
                                list_same_groups.append(ts2_key)

    new_dict = {}
    key_count = 0

    for ts_key, key_val in structure_dict.items():
        if ts_key not in list_same_groups:
            new_dict['group' + '_' + str(key_count).rjust(2, '0')] = key_val
            key_count += 1

    for ts1_key, ts2_key in dict_same_groups.items():
        ind_dict = {}
        phi = []
        theta = []
        files = []

        for row in structure_dict[ts1_key][PHI]:
            phi.append(row)
        for row in structure_dict[ts1_key][THETA]:
            theta.append(row)
        for row in structure_dict[ts2_key][PHI]:
            phi.append(row)
        for row in structure_dict[ts2_key][THETA]:
            theta.append(row)
        for row in structure_dict[ts1_key]['files']:
            files.append(row)
        for row in structure_dict[ts2_key]['files']:
            files.append(row)

        ind_dict[PHI] = phi
        ind_dict[THETA] = theta
        ind_dict['mean phi'] = round(st.mean(phi),4)
        ind_dict['mean theta'] = round(st.mean(theta),4)

        ind_dict[CPK] = structure_dict[ts2_key][CPK]
        ind_dict['group_lm1'] = structure_dict[ts1_key][LM1_GROUP]
        ind_dict['group_lm2'] = structure_dict[ts1_key][LM2_GROUP]
        ind_dict['files'] = files

        new_dict['group' + '_' + str(key_count).rjust(2, '0')] = ind_dict
        key_count += 1

    phi_mean = []
    theta_mean = []

    for structure_key in new_dict.keys():
        phi_mean.append(new_dict[structure_key]['mean phi'])
        theta_mean.append(new_dict[structure_key]['mean theta'])
        p1 = float(new_dict[structure_key]['mean phi'])
        t1 = float(new_dict[structure_key]['mean theta'])
        arc_length_key_dict = {}
        for row in CP_PARAMS:
            p2 = float(row['PHI'])
            t2 = float(row['THETA'])
            arc_length = arc_length_calculator(p1, t1, p2, t2)
            arc_length_key_dict[row[PUCKER]] = arc_length
            top_difference = (sorted(arc_length_key_dict, key=arc_length_key_dict.get, reverse=False)[:1])
            new_dict[structure_key]['closest group puck'] = top_difference

    return new_dict, phi_mean, theta_mean


def group_meaningful_organizor(structure_dict):

    updated_organized_dict = {}
    key_count = 0

    for row in LIST_PUCKER:
        for key, key_val in structure_dict.items():
            if key_val[CPK][0] == row:
                updated_organized_dict['group' + '_' + str(key_count).rjust(2, '0')] = key_val
                key_count += 1

    return updated_organized_dict


def igor_pathway_creator(lm_dict, ts_dict):

    pathway_phi = []
    pathway_theta = []
    pathway_dict = {}
    pathway_grouping = []

    for ts_key in ts_dict.keys():
        lm1_group = ts_dict[ts_key][LM1_GROUP][0]
        lm2_group = ts_dict[ts_key][LM2_GROUP][0]
        ts_phi   = round(ts_dict[ts_key][MPHI],2)
        ts_theta = round(ts_dict[ts_key][MTHETA],2)
        for lm_key in lm_dict.keys():
            if lm_key == lm1_group:
                lm1_phi   = round(lm_dict[lm_key][MPHI],2)
                lm1_theta = round(lm_dict[lm_key][MTHETA],2)
            elif lm_key == lm2_group:
                lm2_phi   = round(lm_dict[lm_key][MPHI],2)
                lm2_theta = round(lm_dict[lm_key][MTHETA],2)

        pathway_grouping.append(str(lm1_group) + '#' + str(ts_key) + '#' + str(lm2_group))

        pathway_phi.append(lm1_phi)
        pathway_theta.append('')

        pathway_phi.append(lm1_phi)
        pathway_theta.append(lm1_theta)

        pathway_phi.append(ts_phi)
        pathway_theta.append(ts_theta)

        pathway_phi.append(lm2_phi)
        pathway_theta.append('')

        pathway_phi.append(lm2_phi)
        pathway_theta.append(lm2_theta)

        pathway_phi.append(ts_phi)
        pathway_theta.append(ts_theta)

    pathway_dict['HPS' + '_path_phi'] = pathway_phi
    pathway_dict['HPS' + '_path_theta'] = pathway_theta

    return pathway_dict


def boltzmann_weighting_reference(final_structure_dict, initial_data_dict, qm_method):

    weighted_dict = {}

    for key, key_val in final_structure_dict.items():
        if len(key_val['files']) == 1:
            for row_files in key_val['files']:
                for row_intial in initial_data_dict:
                    if row_intial[FILE_NAME] == row_files:
                        key_val[WEIGHT_GIBBS] = float(row_intial[GIBBS])
                        key_val[WEIGHT_ENTH]  = float(row_intial[ENTH])
        else:
            total_weight_gibbs = 0
            list_gibbs_energy = []
            list_enths_energy = []
            list_gibbs_weight = []
            list_enths_weight = []
            for row_files in key_val['files']:
                for row_intial in initial_data_dict:
                    if row_intial[FILE_NAME] == row_files:
                        gibbs_energy = float(row_intial[GIBBS])
                        enths_energy = float(row_intial[ENTH])
                        list_gibbs_energy.append(gibbs_energy)
                        list_enths_energy.append(enths_energy)

                        weight_gibbs = math.exp(-gibbs_energy / (DEFAULT_TEMPERATURE * K_B))
                        weight_enth = math.exp(-enths_energy / (DEFAULT_TEMPERATURE * K_B))

                        list_gibbs_weight.append(weight_gibbs)
                        list_enths_weight.append(weight_enth)

                        total_weight_gibbs = total_weight_gibbs + weight_gibbs

            total_energy_gibbs = 0
            total_energy_enths = 0
            for i in range(0,len(list_gibbs_weight)):
                total_energy_gibbs = total_energy_gibbs + list_gibbs_energy[i] * (list_gibbs_weight[i] / total_weight_gibbs)
                total_energy_enths = total_energy_enths + list_enths_energy[i] * (list_gibbs_weight[i] / total_weight_gibbs)

            key_val[WEIGHT_GIBBS] = round(total_energy_gibbs,4)
            key_val[WEIGHT_ENTH]  = round(total_energy_enths,4)

        weighted_dict[key] = key_val

    return weighted_dict
