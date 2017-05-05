#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose if this script is to perform spherical k-means and spherical voronoi to determine the reference structures
for HSP. The output will serve as the foundation for making all comparisons across different methods.

"""

from __future__ import print_function

import os
import sys

import csv
import numpy as np
from qm_utils.qm_common import read_csv_to_dict, create_out_fname
from spherecluster import SphericalKMeans
from scipy.spatial import SphericalVoronoi
import statistics as st
import math
from collections import OrderedDict

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors
import matplotlib.pyplot as plt

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


# Hartree field headers





def read_csv_data(filename, dir_):
    """
    Reads the CSV file with the information
    :param filename: the filename of the CSV file
    :param dir_: the directory
    :return: data points (np array containing the x, y, z coordinates), phi_raw (phi coords), theta_raw (theta coords)
    """

    file_sample_lm = os.path.join(dir_, filename)
    data_dict_lm = read_csv_to_dict(file_sample_lm, mode='r')

    data_points = []
    phi_raw = []
    theta_raw = []
    energy = []

    for i in range(0, len(data_dict_lm)):
        info = data_dict_lm[i]
        phi = float(info['phi'])
        theta = float(info['theta'])
        gibbs = float(info['G298 (Hartrees)'])

        phi_raw.append(phi)
        theta_raw.append(theta)
        energy.append(gibbs)

        x = np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        y = np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = np.cos(np.deg2rad(theta))

        data_points.append(np.array([x, y, z]))

    return data_points, phi_raw, theta_raw, energy


def read_csv_canonical_designations(filename, dir_):
    """
    This script reads in the canonical designations from a csv file
    :param filename: filename containing the canonical designations
    :param dir_: the directory where the file is located
    :return: the puckers, the phi values, and the theta values
    """

    file_cano = os.path.join(dir_, filename)

    phi_cano = []
    theta_cano = []
    pucker = []

    with open(file_cano, 'r') as csvfile:
        data = csv.reader(csvfile)
        for row in data:
            phi_cano.append(row[1])
            theta_cano.append(row[2])
            pucker.append(row[0])

    return pucker, phi_cano, theta_cano


def spherical_kmeans_voronoi(number_clusters, data_points, phi_raw, theta_raw, energy):
    """
    Performs the spherical kmeans and voronoi for the data set
    :param number_clusters: number of clusters that are necessary here
    :param data_points: the set of xyz coordinates that represent the data set
    :return: all of the phi and theta coordinates for vertices and centers
    """
    # Generating the important lists
    ind_dict = {}
    phi_centers = []
    theta_centers = []

    #
    ind_dict['phi_raw'] = phi_raw
    ind_dict['theta_raw'] = theta_raw
    ind_dict['energy'] = energy

    # Uses packages to calculate the k-means spherical centers
    skm = SphericalKMeans(n_clusters=number_clusters, init='k-means++', n_init=20)
    skm.fit(data_points)
    skm_centers = skm.cluster_centers_
    ind_dict['number_clusters'] = number_clusters

    ind_dict['skm_centers_xyz'] = skm_centers

    # Converting the skm centers to phi and theta coordinates
    for center_coord in skm_centers:
        r = np.sqrt(center_coord[0] ** 2 + center_coord[1] ** 2 + center_coord[2] ** 2)
        theta_new = np.rad2deg(np.arctan2(np.sqrt(center_coord[0] ** 2 + center_coord[1] ** 2), center_coord[2]))
        phi_new = np.rad2deg(np.arctan2(center_coord[1], center_coord[0]))
        if phi_new < 0:
            phi_new += 360

        phi_centers.append(round(phi_new, 1))
        theta_centers.append(round(theta_new, 1))

    ind_dict['phi_skm_centers'] = phi_centers
    ind_dict['theta_skm_centers'] = theta_centers
    ind_dict['labels_skm_centers'] = skm.labels_

    # Default parameters for spherical voronoi
    radius = 1
    center = np.array([0, 0, 0])

    # Spherical Voronoi for the centers
    sv = SphericalVoronoi(skm_centers, radius, center)

    # Generating the important base datasets for spherical voronoi
    r_vertices = []
    phi_vertices = []
    theta_vertices = []

    # Computing the Spherical Voronoi vertices to spherical coordinates
    for value in sv.vertices:
        r = np.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)
        theta_new = np.rad2deg(np.arctan2(np.sqrt(value[0] ** 2 + value[1] ** 2), value[2]))
        phi_new = np.rad2deg(np.arctan2(value[1], value[0]))
        if phi_new < 0:
            phi_new += 360

        r_vertices.append(round(r, 1))
        phi_vertices.append(round(phi_new, 1))
        theta_vertices.append(round(theta_new, 1))

    ind_dict['phi_sv_vertices'] = phi_vertices
    ind_dict['theta_sv_vertices'] = theta_vertices
    ind_dict['vertices_sv_xyz'] = sv.vertices

    return ind_dict


def organizing_information_from_spherical_kmeans(data_dict):
    """
    This script is designed to organize all of the raw data information into the correct group
    :param data_dict: the data dict that is output from spherical k-means and voronoi script
    :return: an organized data dict
    """

    temp_groups = {}
    groups = {}

    for g in range(0, data_dict['number_clusters']):
        temp_dict = {}
        temp_dict['assignment_key'] = g
        temp_groups[str('temp_group_' + str(g))] = temp_dict

    for key, key_value in temp_groups.items():
        temp_dict = {}
        energies = []
        theta = []
        phi = []
        mean_phi = None
        mean_theta = None
        for i in range(0, len(data_dict['labels_skm_centers'])):
            if key_value['assignment_key'] == data_dict['labels_skm_centers'][i]:
                energies.append(data_dict['energy'][i])
                theta.append(data_dict['theta_raw'][i])
                phi.append(data_dict['phi_raw'][i])
                mean_phi = data_dict['phi_skm_centers'][key_value['assignment_key']]
                mean_theta = data_dict['theta_skm_centers'][key_value['assignment_key']]

        boltzman_weighted_energies = boltzmann_weighting_mini(energies)

        temp_dict['energies'] = energies
        temp_dict['mean_phi'] = str(mean_phi)
        temp_dict['mean_theta'] = str(mean_theta)
        temp_dict['phi'] = phi
        temp_dict['theta'] = theta
        temp_dict['weighted_gibbs'] = boltzman_weighted_energies

        groups[key] = temp_dict

    final_groups = correcting_group_order(groups)

    return final_groups


def correcting_group_order(groups):
    """
    Organizes in a logical manner
    :param groups: the dict with all of the necessary information
    :return: the final dict the groups properly in order
    """
    final_dict = {}

    phi_groups = []
    theta_groups = []

    groups_top = {}
    dict_top = {}
    groups_mid = {}
    dict_mid = {}
    groups_bot = {}
    dict_bot = {}

    for key, key_val in groups.items():
        phi_groups.append(key_val['mean_phi'])
        theta_groups.append(key_val['mean_theta'])

    for i in range(0, len(phi_groups)):
        if float(theta_groups[i]) < 60:
            for old_id, val in groups.items():
                if theta_groups[i] == val['mean_theta']:
                    groups_top[old_id] = val
                    dict_top[str(val['mean_theta'])] = float(val['mean_phi'])
                    break
        elif float(theta_groups[i]) > 120:
            for old_id, val in groups.items():
                if theta_groups[i] == val['mean_theta']:
                    groups_bot[old_id] = val
                    dict_bot[str(val['mean_theta'])] = float(val['mean_phi'])
                    break
        else:
            for old_id, val in groups.items():
                if theta_groups[i] == val['mean_theta']:
                    groups_mid[old_id] = val
                    dict_mid[str(val['mean_theta'])] = float(val['mean_phi'])
                    break

    group_count = 0

    order_top = sorted(dict_top, key=dict_top.get, reverse=False)
    order_mid = sorted(dict_mid, key=dict_mid.get, reverse=False)
    order_bot = sorted(dict_bot, key=dict_bot.get, reverse=False)

    for row in order_top:
        for key, key_val in groups.items():
            if row == key_val['mean_theta']:
                final_dict['group_' + str(group_count).rjust(2, '0')] = key_val
                group_count += 1

    for row in order_mid:
        for key, key_val in groups.items():
            if row == key_val['mean_theta']:
                final_dict['group_' + str(group_count).rjust(2, '0')] = key_val
                group_count += 1

    for row in order_bot:
        for key, key_val in groups.items():
            if row == key_val['mean_theta']:
                final_dict['group_' + str(group_count).rjust(2, '0')] = key_val
                group_count += 1

    return final_dict


def boltzmann_weighting_mini(energies):
    """
    Performs boltzmann weighting on the groups
    :param energies: the relative gibbs free energies
    :return: the weighted boltzmann parameter
    """
    e_val_list = []
    ind_boltz = []
    total_botlz = 0
    for e_val in energies:
        component = math.exp(-float(e_val) / (K_B * DEFAULT_TEMPERATURE))
        ind_boltz.append(component)
        total_botlz += component

    weighted_gibbs_free_energy = 0
    for i in range(0, len(energies)):
        weighted_gibbs_free_energy += (ind_boltz[i] / total_botlz) * energies[i]

    return round(weighted_gibbs_free_energy,3)


def read_csv_data_TS(filename, dir_):
    """
    Reads the CSV file with the information for the TS structures
    :param filename: the filename of the CSV file
    :param dir_: the directory
    :return: data points (np array containing the x, y, z coordinates), phi_raw (phi coords), theta_raw (theta coords)
    """

    file_sample_ts = os.path.join(dir_, filename)
    data_dict_ts = read_csv_to_dict(file_sample_ts, mode='r')

    data_points = []
    phi_raw = []
    theta_raw = []
    energy = []

    for i in range(0, len(data_dict_ts)):
        info = data_dict_ts[i]
        phi = float(info['phi'])
        theta = float(info['theta'])

        phi_raw.append(phi)
        theta_raw.append(theta)

        x = np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        y = np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = np.cos(np.deg2rad(theta))

        data_points.append(np.array([x, y, z]))

    return data_points, phi_raw, theta_raw, data_dict_ts


def assign_groups_to_TS_LM(data_dict_ts, hsp_lm_groups):
    """"""

    for structure in data_dict_ts:
        print(structure)


    return


# Plotting Functions #

def matplotlib_printing_normal(data_dict, dir, save_status='no', voronoi_status='yes', ts_status='no'):
    # The data from the previous
    phi_raw = data_dict['phi_raw']
    theta_raw = data_dict['theta_raw']
    phi_centers = data_dict['phi_skm_centers']
    theta_centers = data_dict['theta_skm_centers']
    phi_vertices = data_dict['phi_sv_vertices']
    theta_vertices = data_dict['theta_sv_vertices']

    # Canonical Designations
    pucker, phi_cano, theta_cano = read_csv_canonical_designations('CP_params.csv', dir)

    fig, ax = plt.subplots(facecolor='white')

    major_ticksx = np.arange(0, 372, 60)
    minor_ticksx = np.arange(0, 372, 12)
    ax.set_xticks(major_ticksx)
    ax.set_xticks(minor_ticksx, minor=True)

    major_ticksy = np.arange(0, 182, 30)
    minor_ticksy = np.arange(0, 182, 10)
    ax.set_yticks(major_ticksy)
    ax.set_yticks(minor_ticksy, minor=True)

    ax.set_xlim([-10, 370])
    ax.set_ylim([185, -5])
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Theta (degrees)')

    hsp = ax.scatter(phi_raw, theta_raw, s=60, c='blue', marker='o')
    kmeans = ax.scatter(phi_centers, theta_centers, s=60, c='red', marker='h')
    if voronoi_status == 'yes':
        voronoi = ax.scatter(phi_vertices, theta_vertices, s=60, c='green', marker='s')
    cano = ax.scatter(phi_cano, theta_cano, s=60, c='black', marker='+')


    if voronoi_status =='yes':
        leg = ax.legend((hsp, kmeans, voronoi, cano),
                        ('HSP local minima', 'k-means center (k = ' + str(data_dict['number_clusters']) + ')',
                         'voronoi vertice',
                         'canonical designation'),
                        scatterpoints=1, fontsize=12, frameon='false')
    else:
        leg = ax.legend((hsp, kmeans, cano),
                        ('HSP local minima', 'k-means center (k = ' + str(data_dict['number_clusters']) + ')',
                         'canonical designation'),
                        scatterpoints=1, fontsize=12, frameon='false')

    leg.get_frame().set_linewidth(0.0)

    if save_status != 'no':
        if ts_status == 'no':
            filename = create_out_fname('bxyl-k' + str(data_dict['number_clusters']) + '-normal.png', base_dir=dir)
            plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
        elif ts_status == 'yes':
            filename = create_out_fname('bxyl-TS-k' + str(data_dict['number_clusters']) + '-normal.png', base_dir=dir)
            plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
    else:
        plt.show()

    return


def matplotlib_printing_size_bxyl_lm(data_dict, dir, save_status='no'):
    # The data from the previous
    phi_raw = data_dict['phi_raw']
    theta_raw = data_dict['theta_raw']
    phi_centers = data_dict['phi_skm_centers']
    theta_centers = data_dict['theta_skm_centers']
    phi_vertices = data_dict['phi_sv_vertices']
    theta_vertices = data_dict['theta_sv_vertices']
    energy = data_dict['energy']

    # Generating the marker size based on energy
    max_energy = float(max(energy))
    med_energy = float(st.median(energy))
    size = []

    for row in energy:
        size.append(80 * (1 - (float(row) / max_energy)))
        # size.append(60 * float(row) / med_energy)

    # Canonical Designations
    pucker, phi_cano, theta_cano = read_csv_canonical_designations('CP_params.csv', dir)

    fig, ax = plt.subplots(facecolor='white')

    major_ticksx = np.arange(0, 372, 60)
    minor_ticksx = np.arange(0, 372, 12)
    ax.set_xticks(major_ticksx)
    ax.set_xticks(minor_ticksx, minor=True)

    major_ticksy = np.arange(0, 182, 30)
    minor_ticksy = np.arange(0, 182, 10)
    ax.set_yticks(major_ticksy)
    ax.set_yticks(minor_ticksy, minor=True)

    ax.set_xlim([-10, 370])
    ax.set_ylim([185, -5])
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Theta (degrees)')

    hsp = ax.scatter(phi_raw, theta_raw, s=size, c='blue', marker='o')
    kmeans = ax.scatter(phi_centers, theta_centers, s=60, c='red', marker='h')
    cano = ax.scatter(phi_cano, theta_cano, s=60, c='black', marker='+')

    for i, txt in enumerate(pucker):
        if float(theta_cano[i]) < 120 and float(theta_cano[i]) > 60 and float(phi_cano[i]) < 355:
            ax.annotate(txt, xy=(phi_cano[i], theta_cano[i]),
                        xytext=(float(phi_cano[i]) - 7, float(theta_cano[i]) + 12))

    ax.annotate(str(energy[5]) + r' ${\frac{kcal}{mol}}$', xy=(phi_raw[5], theta_raw[5]),
                xytext=(float(phi_raw[5]) + 18, float(theta_raw[5]) - 15), arrowprops=dict(arrowstyle="->",
                                                                                           connectionstyle="arc3"), )
    ax.annotate(str(energy[20]) + r' ${\frac{kcal}{mol}}$', xy=(phi_raw[20], theta_raw[20]),
                xytext=(float(phi_raw[20]) - 20, float(theta_raw[20]) - 15), arrowprops=dict(arrowstyle="->",
                                                                                             connectionstyle="arc3"), )
    ax.annotate(str(energy[11]) + r' ${\frac{kcal}{mol}}$', xy=(phi_raw[11], theta_raw[11]),
                xytext=(float(phi_raw[11]) - 20, float(theta_raw[11]) - 15), arrowprops=dict(arrowstyle="->",
                                                                                             connectionstyle="arc3"), )
    ax.annotate(str(energy[24]) + r' ${\frac{kcal}{mol}}$', xy=(phi_raw[24], theta_raw[24]),
                xytext=(float(phi_raw[24]) - 20, float(theta_raw[24]) - 15), arrowprops=dict(arrowstyle="->",
                                                                                             connectionstyle="arc3"), )

    leg = ax.legend((hsp, kmeans),
                    ('HSP local minima', 'k-means center (k = ' + str(data_dict['number_clusters']) + ')',
                     'voronoi vertice', 'canonical designation'),
                    scatterpoints=1, fontsize=12, frameon='false')

    leg.get_frame().set_linewidth(0.0)

    if save_status != 'no':
        filename = create_out_fname('bxyl-k' + str(data_dict['number_clusters']) + '-size.png', base_dir=dir)
        plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
    else:
        plt.show()

    return


def matplotlib_printing_group_labels(groups, dir_, save_status='no'):

    phi_values = []
    theta_values = []

    for key, key_val in groups.items():
        phi_values.append(key_val['mean_phi'])
        theta_values.append(key_val['mean_theta'])

    # The plotting for this function is completed below.
    fig, ax = plt.subplots(facecolor='white')

    major_ticksx = np.arange(0, 372, 60)
    minor_ticksx = np.arange(0, 372, 12)
    ax.set_xticks(major_ticksx)
    ax.set_xticks(minor_ticksx, minor=True)

    major_ticksy = np.arange(0, 182, 30)
    minor_ticksy = np.arange(0, 182, 10)
    ax.set_yticks(major_ticksy)
    ax.set_yticks(minor_ticksy, minor=True)

    ax.set_xlim([-10, 370])
    ax.set_ylim([185, -5])
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Theta (degrees)')

    # for key, value in group_dict.items():
    kmeans = ax.scatter(phi_values, theta_values, s=60, c='red', marker='h')

    for key, value in groups.items():
        if float(value['mean_theta']) < 30:
            ax.annotate('G-' + key.split('_')[1], xy=(float(value['mean_phi']), float(value['mean_theta'])),
                        xytext=(float(value['mean_phi']) - 10, float(value['mean_theta']) + 15),
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3"), )
        else:
            ax.annotate('G-' + key.split('_')[1], xy=(float(value['mean_phi']), float(value['mean_theta'])),
                        xytext=(float(value['mean_phi']) - 10, float(value['mean_theta']) - 15),
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3"), )



    if save_status != 'off':
        filename = create_out_fname('bxyl-k' + str(len(groups)) + '-groups.png', base_dir=dir_)
        plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
    else:
        plt.show()

    return
