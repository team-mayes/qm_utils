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
from qm_utils.qm_common import read_csv_to_dict, create_out_fname, arc_length_calculator
from spherecluster import SphericalKMeans
from scipy.spatial import SphericalVoronoi
import statistics as st
import math
from collections import OrderedDict

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib as mpl


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
PLM1 = 'phi_lm1'
TLM1 = 'theta_lm1'
PLM2 = 'phi_lm2'
TLM2 = 'theta_lm2'


########################################################################################################################

# # # Local Minima Functions # # #

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
    sv.sort_vertices_of_regions()

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
    ind_dict['regions_sv_labels'] = sv.regions

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


########################################################################################################################

# # # Justin's Functions # # #

def arc_coords(vert_1, vert_2):
    """
    REQUIRES: arclength < PI*radius
    MODIFIES: nothing
    EFFECTS: returns a vector of phi & theta values for the voronoi edges
    :param vert_1: one vertex of an edge
    :param vert_2: other vertex of an edge
    :return: returns a vector of phi & theta values for the voronoi edges
    """

    # desired number of pts in arclength & line
    NUM_PTS = 10

    # endpts of the line
    x_0 = vert_1[0]
    y_0 = vert_1[1]
    z_0 = vert_1[2]
    x_f = vert_2[0]
    y_f = vert_2[1]
    z_f = vert_2[2]

    # eqn for parametric eqns of x, y, & z respectively
    a = x_f - x_0
    b = y_f - y_0
    c = z_f - z_0

    # incrementing variable for parametric equations
    # normalized to allow for input to be number of desired pts
    # if clause to prevent division by zero
    if(a != 0):
        t_0 = abs(((x_f - x_0) / a) / NUM_PTS)
    elif(b != 0):
        t_0 = abs(((y_f - y_0) / b) / NUM_PTS)
    else:
        t_0 = abs(((z_f - z_0) / c) / NUM_PTS)

    t = t_0
    t_f = t_0 * NUM_PTS

    # converts the cartesian coords to polar
    def get_arc_coord(x, y, z):
        theta = np.rad2deg(np.arctan2(np.sqrt(x ** 2 + y ** 2), z))
        phi = np.rad2deg(np.arctan2(y, x))

        while theta < 0:
            theta += 360
        while phi < 0:
            phi += 360

        return(phi, theta)

    # initialize the theta and phi vectors
    coords = get_arc_coord(x_0, y_0, z_0)
    arc_coords = [[coords[0]], [coords[1]]]

    # increments over t to give desired number of pts
    while (t < t_f):
        # parametric eqns
        x = x_0 + t*a
        y = y_0 + t*b
        z = z_0 + t*c

        # pushes polar coords into the arclength
        coords = get_arc_coord(x, y, z)
        arc_coords[0].append(coords[0])
        arc_coords[1].append(coords[1])

        t += t_0

    # pushes final coords into the arclength
    coords = get_arc_coord(x_f, y_f, z_f)
    arc_coords[0].append(coords[0])
    arc_coords[1].append(coords[1])

    return arc_coords


def pol2cart(vert):
    """
    converts polar coords to cartesian
    :param vert:
    :return:
    """
    phi = np.deg2rad(vert[0])
    theta = np.deg2rad(vert[1])
    r = vert[2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return [x, y, z]


# # # Plotting # # #

# vector of edges
def vor_edges(data_dict):
    return get_regions(data_dict)


# gets lines of particular region
def get_vor_sec(verts):
    pairs = []

    for i in range(len(verts)):
        # first vertex gets paired to last vertex
        if i == len(verts) - 1:
            curr_pair = [verts[i], verts[0]]
        else:
            curr_pair = [verts[i], verts[i + 1]]

        pairs.append(curr_pair)

    lines = []

    for i in range(len(pairs)):
        # vector of phi & theta vectors
        edge = arc_coords(pairs[i][0], pairs[i][1])

        if(is_end(edge)):
            two_edges = split_in_two(edge)
            lines.append(two_edges[0])
            lines.append(two_edges[1])
        else:
            lines.append(edge)

    return lines


# gets lines of all regions
def get_regions(data_dict):
    lines = []

    for i in range(len(data_dict['regions_sv_labels'])):
        verts = []

        for j in range(len(data_dict['regions_sv_labels'][i])):
            verts.append(data_dict['vertices_sv_xyz'][data_dict['regions_sv_labels'][i][j]])

        lines.extend(get_vor_sec(verts))

    return lines


# plots 2D and 3D voronoi edges
def matplotlib_edge_printing(data_dict, dir_, save_status ='no'):
    # The data from the previous
    phi_raw = data_dict['phi_raw']
    theta_raw = data_dict['theta_raw']
    phi_centers = data_dict['phi_skm_centers']
    theta_centers = data_dict['theta_skm_centers']
    phi_vertices = data_dict['phi_sv_vertices']
    theta_vertices = data_dict['theta_sv_vertices']

    # Canonical Designations
    pucker, phi_cano, theta_cano = read_csv_canonical_designations('CP_params.csv', dir_)

    fig, ax = plt.subplots(facecolor='white')
    fig_3d = plt.figure()
    ax_3d = fig_3d.gca(projection='3d')

    major_ticksx = np.arange(0, 372, 60)
    minor_ticksx = np.arange(0, 372, 12)
    ax.set_xticks(major_ticksx)
    ax.set_xticks(minor_ticksx, minor=True)

    major_ticksy = np.arange(0, 182, 30)
    minor_ticksy = np.arange(0, 182, 10)
    ax.set_yticks(major_ticksy)
    ax.set_yticks(minor_ticksy, minor=True)

    ax.set_xlim([-5, 365])
    ax.set_ylim([185, -5])
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Theta (degrees)')

    hsp = ax.scatter(phi_raw, theta_raw, s=60, c='blue', marker='o')
    kmeans = ax.scatter(phi_centers, theta_centers, s=60, c='red', marker='h')
    voronoi = ax.scatter(phi_vertices, theta_vertices, s=60, c='green', marker='s')
    cano = ax.scatter(phi_cano, theta_cano, s=60, c='black', marker='+')


    #### TEST PURPOSES ####
    cano_centers = []

    # converts strings to ints
    for i in range(len(phi_cano)):
        phi_cano[i] = float(phi_cano[i])
        theta_cano[i] = float(theta_cano[i])

    # creating cartesian cano_centers
    for i in range(len(phi_cano)):
        vert_test = pol2cart([phi_cano[i], theta_cano[i], 1])
        vert_test = np.asarray(vert_test)

        cano_centers.append(vert_test)

    # Default parameters for spherical voronoi
    radius = 1
    center = np.array([0, 0, 0])

    cano_centers = np.asarray(cano_centers)

    # Spherical Voronoi for the centers

    sv_test = SphericalVoronoi(cano_centers, radius, center)
    sv_test.sort_vertices_of_regions()
    test_dict = {}

    test_dict['number_clusters'] = len(phi_cano)
    test_dict['vertices_sv_xyz'] = sv_test.vertices
    test_dict['regions_sv_labels'] = sv_test.regions

    plot_regions(ax_3d, ax, test_dict)

    #### TEST PURPOSES ####

    # plots wireframe sphere
    theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
    THETA, PHI = np.meshgrid(theta, phi)
    R = 1.0
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    ax_3d.plot_wireframe(X, Y, Z, color="lightblue")

    # settings for 3d graph
    ax_3d.legend()
    ax_3d.set_xlim([-1, 1])
    ax_3d.set_ylim([-1, 1])
    ax_3d.set_zlim([-1, 1])

    plot_regions(ax_3d, ax, data_dict)

    leg = ax.legend((hsp, kmeans, voronoi, cano),
                    ('HSP local minima', 'k-means center (k = ' + str(data_dict['number_clusters']) + ')', 'voronoi vertice',
                     'canonical designation'),
                    scatterpoints = 1, fontsize = 12, frameon = 'false')

    leg.get_frame().set_linewidth(0.0)

    if save_status != 'no':
        filename = create_out_fname('bxyl-k' + str(data_dict['number_clusters']) + '-normal.png', base_dir=dir_)
        plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
    else:
        plt.show()

    return


def plot_vor_sec(ax_3d, ax, verts):
    """
    helper function for plotting a single voronoi section (input is the vertices of the section)
    :param ax: plot being added to
    :param verts: all vertices of the voronoi section
    :return: nothing
    """

    pairs = []

    for i in range(len(verts)):
        # first vertex gets paired to last vertex
        if i == len(verts) - 1:
            curr_pair = [verts[i], verts[0]]
        else:
            curr_pair = [verts[i], verts[i + 1]]

        pairs.append(curr_pair)

    for i in range(len(pairs)):
        # vector of phi & theta vectors
        edge = arc_coords(pairs[i][0], pairs[i][1])

        plot_3d(ax_3d, pairs[i][0], pairs[i][1])

        if(is_end(edge)):
            two_edges = split_in_two(edge)

            ax.plot(two_edges[0][0], two_edges[0][1], color='green')
            ax.plot(two_edges[1][0], two_edges[1][1], color='green')
        else:
            ax.plot(edge[0], edge[1], color='green')

    return


def plot_regions(ax_3d, ax, data_dict):
    """
    plots all voronoi sections
    :param ax_3d:
    :param ax:
    :param data_dict:
    :return:
    """
    for i in range(len(data_dict['regions_sv_labels'])):
        verts = []

        for j in range(len(data_dict['regions_sv_labels'][i])):
            verts.append(data_dict['vertices_sv_xyz'][data_dict['regions_sv_labels'][i][j]])

        plot_vor_sec(ax_3d, ax, verts)

    return


def plot_3d(ax_3d, vert_1, vert_2):
    """
    plots lines individually on a sphere
    :param ax_3d:
    :param vert_1:
    :param vert_2:
    :return:
    """
    mpl.rcParams['legend.fontsize'] = 10

    # endpts of the line
    x_0 = vert_1[0]
    y_0 = vert_1[1]
    z_0 = vert_1[2]
    x_f = vert_2[0]
    y_f = vert_2[1]
    z_f = vert_2[2]

    # polar coords to be changed to cartesian
    raw_coords = arc_coords(vert_1, vert_2)

    # converts the polar coords to cartesian with r = 1
    def get_arc_coord(phi, theta):
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return (x, y, z)

    # initializes the cartesian coordinates for the arclength
    vec_x = [x_0]
    vec_y = [y_0]
    vec_z = [z_0]

    # increments over the raw coords to get cartesian coords
    for i in range(len(raw_coords[0])):
        arc_coord = get_arc_coord(raw_coords[0][i], raw_coords[1][i])

        # pushes coords into the arclength
        vec_x.append(arc_coord[0])
        vec_y.append(arc_coord[1])
        vec_z.append(arc_coord[2])

        i += 1

    # pushes final coord into the arclength
    vec_x.append(x_f)
    vec_y.append(y_f)
    vec_z.append(z_f)

    # plots line
    #ax_3d.plot([x_0, x_f], [y_0, y_f], [z_0, z_f], label='parametric line', color='green')
    # plots arclength
    ax_3d.plot(vec_x, vec_y, vec_z, label='arclength', color='green')


def not_in_pairs(pairs, curr_pair):
    """
    helper function for plot_vor_sec
    returns a bool for if a coord pair is not already in the set of coord pairs
    :param pairs:
    :param curr_pair:
    :return:
    """


    for i in range(len(pairs)):
        if curr_pair == pairs[i] or \
            (curr_pair[0] == pairs[i][1] and curr_pair[1] == pairs[i][0]):

            return False

    return True


def is_end(edge):
    """
    helper function to determine if an edge goes across the end
    (i.e - crosses from 0 to 360)
    :param edge:
    :return:
    """
    has_0 = False
    has_360 = False

    for i in range(len(edge[0])):
        if edge[0][i] < 5:
            has_0 = True
        if edge[0][i] > 355:
            has_360 = True

    return has_0 and has_360


def split_in_two(edge):
    """
    helper function to split an edge into two based on the 0 / 360 degree split
    :param edge: [[phi], [theta]]
    :return: two edges
    """
    edge_one_phi = []
    edge_one_theta = []
    edge_two_phi = []
    edge_two_theta = []

    for i in range(len(edge[0])):
        if edge[0][i] < 180:
            edge_one_phi.append(edge[0][i])
            edge_one_theta.append(edge[1][i])
        elif edge[0][i] >= 180:
            edge_two_phi.append(edge[0][i])
            edge_two_theta.append(edge[1][i])

    edge_one = [edge_one_phi, edge_one_theta]
    edge_two = [edge_two_phi, edge_two_theta]

    two_edges = [edge_one, edge_two]

    return two_edges

########################################################################################################################

# # # Transition State Functions # # #

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
    """
    This script assigned each of the local minima associated with a TS to one of the TS local minima groups
    :param data_dict_ts: the data containing the TS information
    :param hsp_lm_groups: the imported data from the HSP lm file
    :return: TS with the local min assignments (list of dicts) and a workable dict of dict for lm groups
    """

    # Recreating the local minima dict for further processing
    raw_hsp_lm_dict = hsp_lm_groups.to_dict(orient='dict')
    number_dict = raw_hsp_lm_dict['Unnamed: 0']
    hsp_lm_dict = {}

    for group, group_dict in raw_hsp_lm_dict.items():
        ind_dict = {}
        if group != 'Unnamed: 0':
            for num_key, value in group_dict.items():
                correct_key = number_dict[num_key]
                ind_dict[correct_key] = value
            hsp_lm_dict[group] = ind_dict

    for key, key_val in hsp_lm_dict.items():

        phi_values = []
        theta_values = []
        energy_values = []

        phi_redo = key_val['phi'].split(',')
        theta_redo = key_val['theta'].split(',')
        enery_redo = key_val['energies'].split(',')

        for i in range(0, len(phi_redo)):
            phi_values.append(str(phi_redo[i].replace("[", "").replace("]", "")))
            theta_values.append(str(theta_redo[i].replace("[", "").replace("]", "")))
            energy_values.append(str(enery_redo[i].replace("[","").replace("]", "")))

        key_val['phi'] = phi_values
        key_val['theta'] = theta_values
        key_val['energies'] = energy_values

    # Assigning each local minima to a lm group
    assigned_lm = []
    phi_ts_lm = []
    theta_ts_lm = []
    for structure in data_dict_ts:
        p1 = float(structure[PLM1])
        t1 = float(structure[TLM1])
        p2 = float(structure[PLM2])
        t2 = float(structure[TLM2])

        phi_ts_lm.append(p1)
        phi_ts_lm.append(p2)
        theta_ts_lm.append(t1)
        theta_ts_lm.append(t2)

        lm1_arc_dict = {}
        lm2_arc_dict = {}
        for lm_key, lm_val in hsp_lm_dict.items():
            group_phi   = float(lm_val['mean_phi'])
            group_theta = float(lm_val['mean_theta'])
            lm1_arc_dict[lm_key] = arc_length_calculator(p1, t1, group_phi, group_theta, radius=1)
            lm2_arc_dict[lm_key] = arc_length_calculator(p2, t2, group_phi, group_theta, radius=1)
        lm1_assignment = (sorted(lm1_arc_dict, key=lm1_arc_dict.get, reverse=False)[:1])
        lm2_assignment = (sorted(lm2_arc_dict, key=lm2_arc_dict.get, reverse=False)[:1])
        structure['assign_lm1'] = lm1_assignment[0]
        structure['arc_lm1'] = str(round(lm1_arc_dict[lm1_assignment[0]],3))
        structure['assign_lm2'] = lm2_assignment[0]
        structure['arc_lm2'] = str(round(lm2_arc_dict[lm2_assignment[0]],3))
        assigned_lm.append(structure)

    return assigned_lm, hsp_lm_dict, phi_ts_lm, theta_ts_lm




########################################################################################################################

 # # # Plotting Functions # # #

def matplotlib_printing_normal(data_dict, dir_, save_status='no', voronoi_status='yes', ts_status='no'):
    # The data from the previous
    phi_raw = data_dict['phi_raw']
    theta_raw = data_dict['theta_raw']
    phi_centers = data_dict['phi_skm_centers']
    theta_centers = data_dict['theta_skm_centers']
    phi_vertices = data_dict['phi_sv_vertices']
    theta_vertices = data_dict['theta_sv_vertices']

    # Canonical Designations
    pucker, phi_cano, theta_cano = read_csv_canonical_designations('CP_params.csv', dir_)

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

    hsp = ax.scatter(phi_raw, theta_raw, s=60, c='blue', marker='o', edgecolor='face')
    kmeans = ax.scatter(phi_centers, theta_centers, s=60, c='red', marker='h', edgecolor='face')
    if voronoi_status == 'yes':
        voronoi = ax.scatter(phi_vertices, theta_vertices, s=60, c='green', marker='s', edgecolor='face')
    cano = ax.scatter(phi_cano, theta_cano, s=60, c='black', marker='+', edgecolor='face')


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
            filename = create_out_fname('bxyl-k' + str(data_dict['number_clusters']) + '-normal.png', base_dir=dir_)
            plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
        elif ts_status == 'yes':
            filename = create_out_fname('bxyl-TS-k' + str(data_dict['number_clusters']) + '-normal.png', base_dir=dir_)
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

    hsp = ax.scatter(phi_raw, theta_raw, s=size, c='blue', marker='o', edgecolor='face')
    kmeans = ax.scatter(phi_centers, theta_centers, s=60, c='red', marker='h', edgecolor='face')
    cano = ax.scatter(phi_cano, theta_cano, s=60, c='black', marker='+', edgecolor='face')

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
    kmeans = ax.scatter(phi_values, theta_values, s=60, c='red', marker='h', edgecolor='face')

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


def matplotlib_printing_ts_local_min(groups, phi_ts_lm, theta_ts_lm, voronoi_info, dir_, save_status=False):

    phi_sv = voronoi_info['phi_sv_vertices']
    theta_sv = voronoi_info['theta_sv_vertices']


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
    raw_data = ax.scatter(phi_ts_lm, theta_ts_lm, s=60, c='cyan', marker='o', edgecolor='face')
    kmeans = ax.scatter(phi_values, theta_values, s=60, c='red', marker='h', edgecolor='face')
    voronoi = ax.scatter(phi_sv, theta_sv, s=60, c='green', marker='s', edgecolor='face')

    leg = ax.legend((raw_data, kmeans, voronoi),
                    ('HSP LM (from IRCs)', 'k-means center',
                     'voronoi vertice'),
                    scatterpoints=1, fontsize=12, frameon='false')

    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.75)

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

    if save_status is True:
        filename = create_out_fname('bxyl-k' + str(len(groups)) + '-show_TSLMs.png', base_dir=dir_)
        plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
    else:
        plt.show()

    return


def matplotlib_printing_ts_raw_local_mini(groups, phi_ts_lm, theta_ts_lm, voronoi_info, dir_, save_status=False):

    #TODO: add in the edge's from Justin work to complete the image...

    phi_sv = voronoi_info['phi_sv_vertices']
    theta_sv = voronoi_info['theta_sv_vertices']

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
    raw_data = ax.scatter(phi_ts_lm, theta_ts_lm, s=60, c='cyan', marker='o', edgecolor='face')
    kmeans = ax.scatter(phi_values, theta_values, s=60, c='red', marker='h', edgecolor='face')
    #voronoi = ax.scatter(phi_sv, theta_sv, s=60, c='green', marker='o', edgecolor='face')
    voronoi_edges = vor_edges(voronoi_info)

    for i in range(len(voronoi_edges)):
        voronoi = ax.plot(voronoi_edges[i][0], voronoi_edges[i][1], color='green')

    for key, key_val in groups.items():
        phi_group_val = list(map(float, key_val['phi']))
        theta_group_val = list(map(float, key_val['theta']))

        lm_sv_data = ax.scatter(phi_group_val, theta_group_val, s=25, c='blue', marker='o', edgecolor='face')

    leg = ax.legend((raw_data, lm_sv_data, kmeans, voronoi),
                    ('HSP LM (from IRCs)', 'HSP LM (from LM opt)','k-means center',
                     'voronoi vertice'),
                    scatterpoints=1, fontsize=12, frameon='false')

    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.75)


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

    if save_status is True:
        filename = create_out_fname('bxyl-k' + str(len(groups)) + '-comparing_TSLMs_LMs.png', base_dir=dir_)
        plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
    else:
        plt.show()

    return



