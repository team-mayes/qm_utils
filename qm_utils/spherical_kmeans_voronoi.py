#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose if this script is to perform spherical k-means and spherical voronoi to determine the reference structures
for HSP. The output will serve as the foundation for making all comparisons across different methods.

"""

from __future__ import print_function

import os

import csv
import numpy as np
from qm_utils.qm_common import read_csv_to_dict, create_out_fname, arc_length_calculator
from spherecluster import SphericalKMeans
from scipy.spatial import SphericalVoronoi
import math
from collections import OrderedDict
from operator import itemgetter

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# # # Header Stuff # # #
# region
try:
    # noinspection PyCompatibility
    from ConfigParser import ConfigParser
except ImportError:
    # noinspection PyCompatibility
    from configparser import ConfigParser

__author__ = 'jhuber/SPVicchio'

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
# endregion

# # # Helper Functions # # #
# region
# converts a vertex from polar to cartesian
def pol2cart(vert):
    """
    converts polar coords to cartesian
    :param vert:
    :return:
    """
    phi = np.deg2rad(vert[0])
    theta = np.deg2rad(vert[1])
    r = 1

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return [x, y, z]

# converts a vertex from cartesian to polar
def cart2pol(vert):
    theta = np.rad2deg(np.arctan2(np.sqrt(vert[0] ** 2 + vert[1] ** 2), vert[2]))
    phi = np.rad2deg(np.arctan2(vert[1], vert[0]))

    while theta < 0:
        theta += 360
    while phi < 0:
        phi += 360

    return [phi, theta]

# gets the phi and theta values between two vertices
def get_pol_coords(vert_1, vert_2):
    """
    REQUIRES: arclength < PI*radius
    MODIFIES: nothing
    EFFECTS: returns a vector of phi & theta values for the voronoi edges
    :param vert_1: one vertex of an edge
    :param vert_2: other vertex of an edge
    :return: returns a vector of phi & theta values for the voronoi edges
    """

    # desired number of pts in arclength & line
    NUM_PTS = 100

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
    if (a != 0):
        t_0 = abs(((x_f - x_0) / a) / NUM_PTS)
    elif (b != 0):
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

        return (phi, theta)

    # initialize the theta and phi vectors
    coords = get_arc_coord(x_0, y_0, z_0)
    arc_coords = [[coords[0]], [coords[1]]]

    # increments over t to give desired number of pts
    while (t < t_f):
        # parametric eqns
        x = x_0 + t * a
        y = y_0 + t * b
        z = z_0 + t * c

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

# plots a line on a rectangular plot (2D)
# vert_n: [vert in polar, color, size]
# vert_1 always ts, vert_2 always lm
def plot_line(ax, vert_1, vert_2, line_color, line_style='-', edge_color='face', zorder=10):
    ax.set_ylim(105, 75)

    line = get_pol_coords(vert_1[0], vert_2[0])

    if (is_end(line)):
        two_edges = split_in_two(line)

        ax.plot(two_edges[0][0], two_edges[0][1], color=line_color, linestyle=line_style, zorder=1)
        ax.plot(two_edges[1][0], two_edges[1][1], color=line_color, linestyle=line_style, zorder=1)
    else:
        ax.plot(line[0], line[1], color=line_color, linestyle=line_style)

    ax.scatter(line[0][0], line[1][0], s=vert_1[2], c=vert_1[1], marker='s', edgecolor=edge_color, zorder=zorder)
    ax.scatter(line[0][-1], line[1][-1], s=vert_2[2], c=vert_2[1], marker='o', edgecolor='face', zorder=zorder)

    return

# plots a line on a rectangular plot (2D)
def plot_vor_line(ax, vert_1, vert_2, line_color):
    line = get_pol_coords(vert_1, vert_2)

    if (is_end(line)):
        two_edges = split_in_two(line)

        ax.plot(two_edges[0][0], two_edges[0][1], color=line_color, zorder=1)
        ax.plot(two_edges[1][0], two_edges[1][1], color=line_color, zorder=1)
    else:
        ax.plot(line[0], line[1], color=line_color)

    ax.scatter([line[0][0], line[0][-1]], [line[1][0], line[1][-1]], s=30, c='green', marker='s', edgecolor='face',
               zorder=10)

    return

# plots a line on a spherical plot (3D)
def plot_arc(ax_3d, vert_1, vert_2, color_in):
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
    raw_coords = get_pol_coords(vert_1, vert_2)

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

    # plots arclength
    ax_3d.plot(vec_x, vec_y, vec_z, label='arclength', color=color_in, zorder=1)

    # plots verts
    ax_3d.scatter(vert_1[0], vert_1[1], vert_1[2], s=60, c='blue', marker='s', edgecolor='face', zorder=10)
    ax_3d.scatter(vert_2[0], vert_2[1], vert_2[2], s=60, c='green', marker='o', edgecolor='face', zorder=1)

    return raw_coords

# plots a line on a circular plot (2D)
# vert_n: [vert in polar, color, size]
# vert_1 always ts, vert_2 always lm
def plot_on_circle(ax_circ, vert_1, vert_2, line_color='black', line_style='-', edge_color='face', zorder=10):
    pol_coords = get_pol_coords(vert_1[0], vert_2[0])

    # theta
    r = []

    # phi
    theta = pol_coords[0]

    for i in range(len(pol_coords[1])):
        r.append(abs(math.sin(np.radians(pol_coords[1][i]))))
        theta[i] = np.radians(pol_coords[0][i])

    theta[0] = theta[1]

    ax_circ.plot(theta, r, color=line_color, linestyle=line_style, zorder=1)

    ax_circ.scatter(theta[0], r[0], s=vert_1[2], c=vert_1[1], marker='s', edgecolor=edge_color, zorder=zorder)
    ax_circ.scatter(theta[-1], r[-1], s=vert_2[2], c=vert_2[1], marker='o', edgecolor='face', zorder=zorder)

    return

# creates a file for given plot & figure
def make_file_from_plot(filename, fig, dir_):
    filename1 = create_out_fname(filename, base_dir=dir_, ext='.png')
    fig.savefig(filename1, facecolor=fig.get_facecolor(), transparent=True)

# checks if the edge crosses from 0 to 360
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

# splits an edge that crosses from 0 to 360 in half
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
# endregion

# # # Classes # # #
# region
# class for local minima
class Local_Minima():
    def __init__(self, number_clusters_in, data_points_in, cano_points_in, phi_raw_in, theta_raw_in, energy):
        self.sv_kmeans_dict = {}
        self.groups_dict = {}
        self.cano_points = cano_points_in
        self.plot = Plots(False, False, True)

        self.populate_sv_kmeans_dict(number_clusters_in, data_points_in, phi_raw_in, theta_raw_in, energy)
        self.populate_groups_dict()
        self.calc_arc_lengths()

        self.organize_regions()

        self.assign_closest_puckers()
        self.assign_group_name()

    def populate_sv_kmeans_dict(self, number_clusters, data_points, phi_raw, theta_raw, energy):
        # Generating the important lists
        phi_centers = []
        theta_centers = []

        self.sv_kmeans_dict['phi_raw'] = phi_raw
        self.sv_kmeans_dict['theta_raw'] = theta_raw
        if energy is not None:
            self.sv_kmeans_dict['energy'] = energy

        # Uses packages to calculate the k-means spherical centers
        skm = SphericalKMeans(n_clusters=number_clusters, init='k-means++', n_init=30)
        skm.fit(data_points)
        skm_centers = skm.cluster_centers_
        self.sv_kmeans_dict['number_clusters'] = number_clusters

        self.sv_kmeans_dict['skm_centers_xyz'] = skm_centers

        # Converting the skm centers to phi and theta coordinates
        for center_coord in skm_centers:
            r = np.sqrt(center_coord[0] ** 2 + center_coord[1] ** 2 + center_coord[2] ** 2)
            theta_new = np.rad2deg(
                np.arctan2(np.sqrt(center_coord[0] ** 2 + center_coord[1] ** 2), center_coord[2]))
            phi_new = np.rad2deg(np.arctan2(center_coord[1], center_coord[0]))
            if phi_new < 0:
                phi_new += 360

            phi_centers.append(round(phi_new, 1))
            theta_centers.append(round(theta_new, 1))

        self.sv_kmeans_dict['phi_skm_centers'] = phi_centers
        self.sv_kmeans_dict['theta_skm_centers'] = theta_centers
        self.sv_kmeans_dict['labels_skm_centers'] = skm.labels_

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

        self.sv_kmeans_dict['phi_sv_vertices'] = phi_vertices
        self.sv_kmeans_dict['theta_sv_vertices'] = theta_vertices
        self.sv_kmeans_dict['vertices_sv_xyz'] = sv.vertices
        self.sv_kmeans_dict['regions_sv_labels'] = sv.regions

        return

    def populate_groups_dict(self):
        temp_groups = {}
        groups = {}

        for g in range(0, self.sv_kmeans_dict['number_clusters']):
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
            for i in range(0, len(self.sv_kmeans_dict['labels_skm_centers'])):
                if key_value['assignment_key'] == self.sv_kmeans_dict['labels_skm_centers'][i]:
                    if len(self.sv_kmeans_dict['energy']) != 0:
                        energies.append(self.sv_kmeans_dict['energy'][i])
                    theta.append(self.sv_kmeans_dict['theta_raw'][i])
                    phi.append(self.sv_kmeans_dict['phi_raw'][i])
                    mean_phi = self.sv_kmeans_dict['phi_skm_centers'][key_value['assignment_key']]
                    mean_theta = self.sv_kmeans_dict['theta_skm_centers'][key_value['assignment_key']]

            boltzman_weighted_energies = boltzmann_weighting_mini(energies)

            temp_dict['energies'] = energies
            temp_dict['mean_phi'] = str(mean_phi)
            temp_dict['mean_theta'] = str(mean_theta)
            temp_dict['phi'] = phi
            temp_dict['theta'] = theta
            temp_dict['weighted_gibbs'] = boltzman_weighted_energies

            groups[key] = temp_dict

        # dict for the class info
        self.groups_dict = correcting_group_order(groups)

        return

    def calc_arc_lengths(self):
        for group_key in self.groups_dict:
            arc_lengths = []

            group_phi = float(self.groups_dict[group_key]['mean_phi'])
            group_theta = float(self.groups_dict[group_key]['mean_theta'])

            for i in range(len(self.groups_dict[group_key]['phi'])):
                raw_phi = self.groups_dict[group_key]['phi'][i]
                raw_theta = self.groups_dict[group_key]['theta'][i]

                arc_lengths.append(arc_length_calculator(group_phi, group_theta, raw_phi, raw_theta))

            self.groups_dict[group_key]['arc_lengths'] = arc_lengths

    def assign_closest_puckers(self):
        for group_key in self.groups_dict:
            # list for 3 shortest arclengths and their lm_groups
            arc_lengths = {}

            group_phi = float(self.groups_dict[group_key]['mean_phi'])
            group_theta = float(self.groups_dict[group_key]['mean_theta'])

            for j in range(len(self.cano_points['pucker'])):
                pucker_phi = float(self.cano_points['phi_cano'][j])
                pucker_theta = float(self.cano_points['theta_cano'][j])

                arc_lengths[self.cano_points['pucker'][j]] = arc_length_calculator(group_phi, group_theta, pucker_phi, pucker_theta)

            ordered_arc_lengths = OrderedDict(sorted(arc_lengths.items(), key=itemgetter(1), reverse=False))
            ordered_list = []
            three_shortest_list = []

            for key, val in ordered_arc_lengths.items():
                ordered_list.append([key, val])

            for k in range(3):
                three_shortest_list.append(ordered_list[k])

            self.groups_dict[group_key]['closest_puckers'] = three_shortest_list

        return

    def assign_group_name(self):
        for group_key in self.groups_dict:
            tolerance = 0.25

            first_arc = self.groups_dict[group_key]['closest_puckers'][0][1]
            second_arc = self.groups_dict[group_key]['closest_puckers'][1][1]

            total_arc = first_arc + second_arc

            first_weight = first_arc / total_arc

            if first_weight < tolerance:
                self.groups_dict[group_key]['name'] = self.groups_dict[group_key]['closest_puckers'][0][0]
            else:
                self.groups_dict[group_key]['name'] = self.groups_dict[group_key]['closest_puckers'][0][0]\
                                                      + '/'\
                                                      + self.groups_dict[group_key]['closest_puckers'][1][0]

    # # # Plotting Functions # # #
    #region
    def plot_group_names(self):
        for key, value in self.groups_dict.items():
            if float(value['mean_theta']) < 30 or float(value['mean_phi']) < 25:
                self.plot.ax_rect.annotate(value['name'], xy=(float(value['mean_phi']), float(value['mean_theta'])),
                            xytext=(float(value['mean_phi']) - 10, float(value['mean_theta']) + 15),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), )
            else:
                self.plot.ax_rect.annotate(value['name'], xy=(float(value['mean_phi']), float(value['mean_theta'])),
                            xytext=(float(value['mean_phi']) - 10, float(value['mean_theta']) - 15),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), )

    def plot_cano(self):
        self.plot.ax_rect.scatter(self.cano_points['phi_cano'], self.cano_points['theta_cano'], s=60, c='black',
                                  marker='+',
                                  edgecolor='face')

    def show(self):
        self.plot.show()

    def plot_cano_vor(self):
        cano_plot = Plots(False, False, True)

        cano_centers = []

        # converts strings to ints
        for i in range(len(self.cano_points['phi_cano'])):
            self.cano_points['phi_cano'][i] = float(self.cano_points['phi_cano'][i])
            self.cano_points['theta_cano'][i] = float(self.cano_points['theta_cano'][i])

        # creating cartesian cano_centers
        for i in range(len(self.cano_points['phi_cano'])):
            cano_vert = pol2cart([self.cano_points['phi_cano'][i], self.cano_points['theta_cano'][i], 1])
            cano_vert = np.asarray(cano_vert)

            cano_centers.append(cano_vert)

        # Default parameters for spherical voronoi
        radius = 1
        center = np.array([0, 0, 0])

        cano_centers = np.asarray(cano_centers)

        # Spherical Voronoi for the centers

        sv_cano = SphericalVoronoi(cano_centers, radius, center)
        sv_cano.sort_vertices_of_regions()
        cano_dict = {}

        cano_dict['number_clusters'] = len(self.cano_points['phi_cano'])
        cano_dict['vertices_sv_xyz'] = sv_cano.vertices
        cano_dict['regions_sv_labels'] = sv_cano.regions

        for k in range(len(cano_dict['regions_sv_labels'])):
            verts = []

            for j in range(len(cano_dict['regions_sv_labels'][k])):
                verts.append(cano_dict['vertices_sv_xyz'][cano_dict['regions_sv_labels'][k][j]])

            pairs = []

            for i in range(len(verts)):
                # first vertex gets paired to last vertex
                if i == len(verts) - 1:
                    curr_pair = [verts[i], verts[0]]
                else:
                    curr_pair = [verts[i], verts[i + 1]]

                pairs.append(curr_pair)

            for i in range(len(pairs)):
                # plot_arc(ax_3d, pairs[i][0], pairs[i][1])

                plot_vor_line(cano_plot.ax_rect, pairs[i][0], pairs[i][1], 'green')

        cano_plot.ax_rect.scatter(self.cano_points['phi_cano'], self.cano_points['theta_cano'], s=60, c='black',
                                  marker='+',
                                  edgecolor='face')

        cano_plot.show()

        return

    def plot_vor_sec(self, lm_group):
        """
        helper function for plotting a single voronoi section (input is the vertices of the section)
        :param ax: plot being added to
        :param verts: all vertices of the voronoi section
        :return: nothing
        """

        verts = []

        for j in range(len(self.sv_kmeans_dict['regions_sv_labels'][int(lm_group)])):
            verts.append(
                self.sv_kmeans_dict['vertices_sv_xyz'][self.sv_kmeans_dict['regions_sv_labels'][int(lm_group)][j]])

        pairs = []

        for i in range(len(verts)):
            # first vertex gets paired to last vertex
            if i == len(verts) - 1:
                curr_pair = [verts[i], verts[0]]
            else:
                curr_pair = [verts[i], verts[i + 1]]

            pairs.append(curr_pair)

        for i in range(len(pairs)):
            # plot_arc(ax_3d, pairs[i][0], pairs[i][1])

            plot_vor_line(self.plot.ax_rect, pairs[i][0], pairs[i][1], 'green')

        return

    def plot_all_vor_sec(self):
        for i in range(len(self.groups_dict)):
            self.plot_vor_sec(i)

        return

    def wipe_plot(self):
        self.plot = Plots(False, False, True)
    #endregion

    def organize_regions(self):
        org_sv_reg_dict = {}
        org_phi_skm_dict = {}
        org_theta_skm_dict = {}
        org_sv_reg_list = []
        org_phi_skm_list = []
        org_theta_skm_list = []

        for group_key in self.groups_dict:
            for i in range(len(self.sv_kmeans_dict['phi_skm_centers'])):
                if float(self.groups_dict[group_key]['mean_phi']) == float(self.sv_kmeans_dict['phi_skm_centers'][i]) \
                    and float(self.groups_dict[group_key]['mean_theta']) == float(
                        self.sv_kmeans_dict['theta_skm_centers'][i]):
                    org_sv_reg_dict[group_key] = self.sv_kmeans_dict['regions_sv_labels'][i]
                    org_phi_skm_dict[group_key] = self.sv_kmeans_dict['phi_skm_centers'][i]
                    org_theta_skm_dict[group_key] = self.sv_kmeans_dict['theta_skm_centers'][i]

        for key in org_sv_reg_dict:
            org_sv_reg_list.append(org_sv_reg_dict[key])
            org_phi_skm_list.append(org_phi_skm_dict[key])
            org_theta_skm_list.append(org_theta_skm_dict[key])

        self.sv_kmeans_dict['regions_sv_labels'] = org_sv_reg_list
        self.sv_kmeans_dict['phi_skm_centers'] = org_phi_skm_list
        self.sv_kmeans_dict['theta_skm_centers'] = org_theta_skm_list

        return

# class for transition states and their pathways
class Transition_States():
    def __init__(self, ts_data_in, lm_class_obj):
        # groups by the unique transition state paths
        __unorg_groups = sorting_TS_into_groups(ts_data_in, lm_class_obj)

        self.ts_groups = self.reorg_groups(__unorg_groups, lm_class_obj)
        self.lm_class = lm_class_obj

        self.assign_closest_puckers()
        self.assign_group_name()

        self.circ_groups_init()

        self.plot = Plots(True, False, False)
    # # # init functions # # #
    #region
    # reorganizes the data structure
    def reorg_groups(self, unorg_groups, lm_class_obj):
        temp_ts_groups = {}

        # creating new dict of lm_groups
        for lm_key in unorg_groups:
            curr_lm_group = unorg_groups[lm_key]
            temp_ts_group = {}

            for i in range(curr_lm_group['num_clusters']):
                # storing the kmeans ts vertex
                ts_vert = [curr_lm_group['center_phi'][i],
                           curr_lm_group['center_theta'][i]]

                temp_ts_group[i] = {}
                temp_ts_group[i]['ts_vert'] = np.asarray(pol2cart(ts_vert))

                lm_keys = lm_key.split("_")
                lm1_key = int(lm_keys[0])
                lm2_key = int(lm_keys[1])

                lm1_phi = lm_class_obj.groups_dict[lm1_key]['mean_phi']
                lm1_theta = lm_class_obj.groups_dict[lm1_key]['mean_theta']

                lm1_vert = [float(lm1_phi), float(lm1_theta)]

                lm2_phi = lm_class_obj.groups_dict[lm2_key]['mean_phi']
                lm2_theta = lm_class_obj.groups_dict[lm2_key]['mean_theta']

                lm2_vert = [float(lm2_phi), float(lm2_theta)]

                temp_ts_group[i]['lm1_vert_cart'] = np.asarray(pol2cart(lm1_vert))
                temp_ts_group[i]['lm2_vert_cart'] = np.asarray(pol2cart(lm2_vert))
                temp_ts_group[i]['ts_vert_cart'] = np.asarray(pol2cart(ts_vert))

                temp_ts_group[i]['lm1_vert_pol'] = np.asarray(lm1_vert)
                temp_ts_group[i]['lm2_vert_pol'] = np.asarray(lm2_vert)
                temp_ts_group[i]['ts_vert_pol'] = np.asarray(ts_vert)

                # creating a dict of uniq ts's
                temp_ts_group[i]['ts_group'] = {}

                # creating a dict for each unique transition state
                for j in range(len(curr_lm_group['gibbs_energy'])):
                    if curr_lm_group['skm_labels'][j] == i:
                        uniq_ts = {}

                        uniq_ts['energy (A.U.)'] = curr_lm_group['gibbs_energy'][j]
                        uniq_ts['uniq_ts_vert_cart'] = np.asarray(pol2cart([curr_lm_group['ts_vals_phi'][j],
                                                                            curr_lm_group['ts_vals_theta'][j]]))
                        # storing the lm verts for specific ts
                        uniq_ts['uniq_lm1_vert_cart'] = np.asarray(pol2cart([curr_lm_group['lm_vals_phi'][j][0],
                                                                             curr_lm_group['lm_vals_theta'][j][0]]))
                        uniq_ts['uniq_lm2_vert_cart'] = np.asarray(pol2cart([curr_lm_group['lm_vals_phi'][j][1],
                                                                             curr_lm_group['lm_vals_theta'][j][1]]))

                        uniq_ts['uniq_ts_vert_pol'] = np.asarray([curr_lm_group['ts_vals_phi'][j],
                                                                  curr_lm_group['ts_vals_theta'][j]])
                        # storing the lm verts for specific ts
                        uniq_ts['uniq_lm1_vert_pol'] = np.asarray([curr_lm_group['lm_vals_phi'][j][0],
                                                                   curr_lm_group['lm_vals_theta'][j][0]])
                        uniq_ts['uniq_lm2_vert_pol'] = np.asarray([curr_lm_group['lm_vals_phi'][j][1],
                                                                   curr_lm_group['lm_vals_theta'][j][1]])

                        temp_ts_group[i]['ts_group'][j] = uniq_ts

                wt_gibbs = 0
                total_boltz = 0

                # finding Boltzmann weighted Gibb's free energy
                for key in temp_ts_group[i]['ts_group']:
                    e_val = temp_ts_group[i]['ts_group'][key]['energy (A.U.)']
                    component = math.exp(-float(e_val) / (K_B * DEFAULT_TEMPERATURE))
                    temp_ts_group[i]['ts_group'][key]['ind_boltz'] = component
                    total_boltz += component

                for key in temp_ts_group[i]['ts_group']:
                    wt_gibbs += (temp_ts_group[i]['ts_group'][key]['ind_boltz'] / total_boltz) * temp_ts_group[i]['ts_group'][key][
                        'energy (A.U.)']

                temp_ts_group[i]['G298 (Hartrees)'] = round(wt_gibbs, 3)

            temp_ts_groups[lm_key] = temp_ts_group

        return temp_ts_groups

    def assign_closest_puckers(self):
        for group_key in self.ts_groups:
            for i in self.ts_groups[group_key]:
                # list for 3 shortest arclengths and their lm_groups
                arc_lengths = {}

                ts_phi = float(self.ts_groups[group_key][i]['ts_vert_pol'][0])
                ts_theta = float(self.ts_groups[group_key][i]['ts_vert_pol'][1])

                for k in range(len(self.lm_class.cano_points['pucker'])):
                    pucker_phi = float(self.lm_class.cano_points['phi_cano'][k])
                    pucker_theta = float(self.lm_class.cano_points['theta_cano'][k])

                    arc_lengths[self.lm_class.cano_points['pucker'][k]] = arc_length_calculator(ts_phi, ts_theta,
                                                                                                pucker_phi,
                                                                                                pucker_theta)

                ordered_arc_lengths = OrderedDict(sorted(arc_lengths.items(), key=itemgetter(1), reverse=False))
                ordered_list = []
                three_shortest_list = []

                for key, val in ordered_arc_lengths.items():
                    ordered_list.append([key, val])

                for k in range(3):
                    three_shortest_list.append(ordered_list[k])

                self.ts_groups[group_key][i]['closest_puckers'] = three_shortest_list

        return

    def assign_group_name(self):
        for group_key in self.ts_groups:
            for i in range(len(self.ts_groups[group_key])):
                tolerance = 0.25

                first_arc = self.ts_groups[group_key][i]['closest_puckers'][0][1]
                second_arc = self.ts_groups[group_key][i]['closest_puckers'][1][1]

                total_arc = first_arc + second_arc

                first_weight = first_arc / total_arc

                if first_weight < tolerance:
                    self.ts_groups[group_key][i]['name'] = self.ts_groups[group_key][i]['closest_puckers'][0][0]
                else:
                    self.ts_groups[group_key][i]['name'] = self.ts_groups[group_key][i]['closest_puckers'][0][0] \
                                                 + '/' \
                                                 + self.ts_groups[group_key][i]['closest_puckers'][1][0]

    # get group keys associated with north, south, and equatorial
    def circ_groups_init(self):
        self.north_groups = []
        self.south_groups = []
        self.equat_groups = []

        lm_key_list = list(self.lm_class.groups_dict.keys())

        lm_key_list.sort()

        for lm_key in self.ts_groups:
            if lm_key.split("_")[0] == lm_key_list[0]:
                self.north_groups.append(lm_key)
            elif lm_key.split("_")[1] == lm_key_list[-1]:
                self.south_groups.append(lm_key)
            else:
                self.equat_groups.append(lm_key)

        return
    #endregion

    # # # plotting functions # # #
    #region
    # plots desired local minimum group pathways for all uniq ts pts
    def plot_loc_min_group_2d(self, lm_key_in):
        for lm_key in self.ts_groups:
            # if the key is the local min group, plot it
            if (lm_key == lm_key_in):
                for ts_group_key in self.ts_groups[lm_key]:
                    path = self.ts_groups[lm_key][ts_group_key]

                    plot_line(self.plot.ax_rect, [path['ts_vert_cart'], 'blue', 60], [path['lm1_vert_cart'], 'green', 60], 'red')
                    plot_line(self.plot.ax_rect, [path['ts_vert_cart'], 'blue', 60], [path['lm2_vert_cart'], 'green', 60], 'red')

                    if self.north_groups.count(lm_key) == 1:
                        plot_on_circle(self.plot.ax_circ_north, [path['ts_vert_cart'], 'blue', 60], [path['lm1_vert_cart'], 'green', 60], 'red')
                        plot_on_circle(self.plot.ax_circ_north, [path['ts_vert_cart'], 'blue', 60], [path['lm2_vert_cart'], 'green', 60], 'red')
                    elif self.south_groups.count(lm_key) == 1:
                        plot_on_circle(self.plot.ax_circ_south, [path['ts_vert_cart'], 'blue', 60], [path['lm1_vert_cart'], 'green', 60], 'red')
                        plot_on_circle(self.plot.ax_circ_south, [path['ts_vert_cart'], 'blue', 60], [path['lm2_vert_cart'], 'green', 60], 'red')

    # plots desired local minimum group pathways for all uniq ts pts
    def plot_loc_min_group_3d(self, plot, lm_key_in):
        for lm_key in self.ts_groups:
            # if the key is the local min group, plot it
            if (lm_key == lm_key_in):
                for ts_group_key in self.ts_groups[lm_key]:
                    path = self.ts_groups[lm_key][ts_group_key]

                    plot_arc(plot.ax_spher, path['ts_vert_cart'], path['lm1_vert_cart'], 'red')
                    plot_arc(plot.ax_spher, path['ts_vert_cart'], path['lm2_vert_cart'], 'red')

        plot.show()

        return

    # plots canonical designations
    def plot_cano(self):
        self.plot.ax_rect.scatter(self.lm_class.cano_points['phi_cano'], self.lm_class.cano_points['theta_cano'], s=60, c='black',
                                  marker='+',
                                  edgecolor='face')

    # plots all pathways in 2d
    def plot_all_2d(self):
        for lm_key in self.ts_groups:
            self.plot_loc_min_group_2d(lm_key)

    # plots all pathways in 3d
    def plot_all_3d(self):
        plot = Plots(False, True, False)

        for lm_key in self.ts_groups:
            self.plot_loc_min_group_3d(plot, lm_key)

        plot.show()

    def plot_vor_sec(self, ax, verts):
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
            # plot_arc(ax_3d, pairs[i][0], pairs[i][1])

            plot_vor_line(ax, pairs[i][0], pairs[i][1], 'green')

        return

    def plot_regions(self, ax):
        for i in range(len(self.lm_class.sv_kmeans_dict['regions_sv_labels'])):
            verts = []

            for j in range(len(self.lm_class.sv_kmeans_dict['regions_sv_labels'][i])):
                verts.append(self.lm_class.sv_kmeans_dict['vertices_sv_xyz'][
                                 self.lm_class.sv_kmeans_dict['regions_sv_labels'][i][j]])

            self.plot_vor_sec(ax, verts)

        return

    def show(self):
        self.plot.show()

    def wipe_plot(self):
        self.plot = Plots(True, False, False)
    #endregion

    pass

# class for initializing plots
class Plots():
    # arguments are bools for creating the 2d and 3d plots
    def __init__(self, twoD_arg=False, threeD_arg=False, merc_arg=False, ts_merc_arg=False, north_pol_arg=False, south_pol_arg=False):
        self.fig = plt.figure()

        if twoD_arg:
            gs = gridspec.GridSpec(2, 2)
            self.ax_rect = self.fig.add_subplot(gs[1:, :], facecolor='white')
            self.ax_circ_north = self.fig.add_subplot(gs[0, 0], projection='polar')
            self.ax_circ_south = self.fig.add_subplot(gs[0, 1], projection='polar')

            self.ax_rect.set_xlim([-5, 365])
            self.ax_rect.set_ylim([105, 75])

            # initializing settings for the 2d plot
            self.twoD_init()

        if threeD_arg:
            self.ax_spher = self.fig.gca(projection='3d')

            # settings for spherical plot
            self.threeD_init()

        if merc_arg:
            self.fig, self.ax_rect = plt.subplots(facecolor='white')

            self.ax_rect.set_xlim([-5, 365])
            self.ax_rect.set_ylim([185, -5])

            self.ax_rect_init()

        if ts_merc_arg:
            gs = gridspec.GridSpec(1, 1)
            self.ax_rect = self.fig.add_subplot(gs[0, 0])

            self.ax_rect_init()

        if north_pol_arg:
            gs = gridspec.GridSpec(1, 1)
            self.ax_circ_north = self.fig.add_subplot(gs[0, 0], projection='polar')

            self.ax_circ_north_init()

        if south_pol_arg:
            gs = gridspec.GridSpec(1, 1)
            self.ax_circ_south = self.fig.add_subplot(gs[0, 0], projection='polar')

            self.ax_circ_south_init()

    def twoD_init(self):
        self.ax_rect_init()
        self.ax_circ_north_init()
        self.ax_circ_south_init()

    def ax_rect_init(self):
        major_ticksx = np.arange(0, 372, 60)
        minor_ticksx = np.arange(0, 372, 12)
        self.ax_rect.set_xticks(major_ticksx)
        self.ax_rect.set_xticks(minor_ticksx, minor=True)

        major_ticksy = np.arange(0, 182, 30)
        minor_ticksy = np.arange(0, 182, 10)
        self.ax_rect.set_yticks(major_ticksy)
        self.ax_rect.set_yticks(minor_ticksy, minor=True)

        self.ax_rect.set_xlabel('Phi (degrees)')
        self.ax_rect.set_ylabel('Theta (degrees)')

    def threeD_init(self):
        # plots wireframe sphere
        theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
        THETA, PHI = np.meshgrid(theta, phi)
        R = 1.0
        X = R * np.sin(PHI) * np.cos(THETA)
        Y = R * np.sin(PHI) * np.sin(THETA)
        Z = R * np.cos(PHI)
        self.ax_spher.plot_wireframe(X, Y, Z, color="lightblue")

        # settings for 3d graph
        self.ax_spher.legend()
        self.ax_spher.set_xlim([-1, 1])
        self.ax_spher.set_ylim([-1, 1])
        self.ax_spher.set_zlim([-1, 1])

    def ax_circ_north_init(self):
        thetaticks = np.arange(0, 360, 30)

        self.ax_circ_north.set_rlim([0, 1])
        self.ax_circ_north.set_rticks([0.5])  # less radial ticks
        self.ax_circ_north.set_rlabel_position(-22.5)  # get radial labels away from plotted line
        self.ax_circ_north.set_title("Northern", ha='right', va='bottom', loc='left', fontsize=12)
        self.ax_circ_north.set_theta_zero_location("N")
        self.ax_circ_north.set_yticklabels([])
        self.ax_circ_north.set_thetagrids(thetaticks, frac=1.25, fontsize=12)
        self.ax_circ_north.set_theta_direction(-1)

    def ax_circ_south_init(self):
        thetaticks = np.arange(0, 360, 30)

        self.ax_circ_south.set_rlim([0, 1])
        self.ax_circ_south.set_rticks([0.5])  # less radial ticks
        self.ax_circ_south.set_rlabel_position(-22.5)  # get radial labels away from plotted line
        self.ax_circ_south.set_title("Southern", ha='right', va='bottom', loc='left', fontsize=12)
        self.ax_circ_south.set_theta_zero_location("N")
        self.ax_circ_south.set_yticklabels([])
        self.ax_circ_south.set_thetagrids(thetaticks, frac=1.25, fontsize=12)
        self.ax_circ_south.set_theta_direction(-1)

    # shows all plots
    def show(self):
        plt.show()

    def save(self, filename, dir_):
        filename1 = create_out_fname(filename, base_dir=dir_, ext='.png')
        self.fig.set_size_inches(9, 5)
        self.fig.savefig(filename1, facecolor=self.fig.get_facecolor(), transparent=True, dpi=300, bbox_inches='tight')
# endregion

# # # Local Minima Functions # # #
# region
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

    dict = {}
    dict['pucker'] = pucker
    dict['phi_cano'] = phi_cano
    dict['theta_cano'] = theta_cano

    return dict

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
                final_dict[group_count] = key_val
                group_count += 1

    for row in order_mid:
        for key, key_val in groups.items():
            if row == key_val['mean_theta']:
                final_dict[group_count] = key_val
                group_count += 1

    for row in order_bot:
        for key, key_val in groups.items():
            if row == key_val['mean_theta']:
                final_dict[group_count] = key_val
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

    return round(weighted_gibbs_free_energy, 3)
# endregion

# # # Transition State Functions # # #
# region
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
        for lm_key, lm_val in hsp_lm_groups.items():
            group_phi = float(lm_val['mean_phi'])
            group_theta = float(lm_val['mean_theta'])
            lm1_arc_dict[lm_key] = arc_length_calculator(p1, t1, group_phi, group_theta, radius=1)
            lm2_arc_dict[lm_key] = arc_length_calculator(p2, t2, group_phi, group_theta, radius=1)
        lm1_assignment = (sorted(lm1_arc_dict, key=lm1_arc_dict.get, reverse=False)[:1])
        lm2_assignment = (sorted(lm2_arc_dict, key=lm2_arc_dict.get, reverse=False)[:1])
        structure['assign_lm1'] = lm1_assignment[0]
        structure['arc_lm1'] = str(round(lm1_arc_dict[lm1_assignment[0]], 3))
        structure['assign_lm2'] = lm2_assignment[0]
        structure['arc_lm2'] = str(round(lm2_arc_dict[lm2_assignment[0]], 3))
        assigned_lm.append(structure)

    return


def sorting_TS_into_groups(data_points, lm_class_obj, show_status=False):
    local_min_structure = {}

    assign_groups_to_TS_LM(data_points, lm_class_obj.groups_dict)

    for row in data_points:
        first, second = return_lowest_value(row['assign_lm1'], row['assign_lm2'])

        if str(first) + '_' + str(second) not in local_min_structure.keys():
            inner_dict = {}
            inner_dict['origin_files'] = [row]
            local_min_structure[str(first) + '_' + str(second)] = inner_dict
        else:
            local_min_structure[str(first) + '_' + str(second)]['origin_files'].append(row)

    # Check to makesure that the files are loaded properly
    count = 0
    for key, key_val in local_min_structure.items():
        count += len(key_val['origin_files'])
    if count != len(data_points):
        print('WARNING: THE NUMBER OF FILES CURRENTLY BEING STUDIED HAS AN ISSUE.')

    lm_lm_dict = {}

    for group_key, group_info in local_min_structure.items():
        ts_phi_vals = []
        ts_theta_vals = []
        lm_phi_vals = []
        lm_theta_vals = []
        xyz_data = []
        energies = []
        for row in group_info['origin_files']:
            ts_phi_vals.append(float(row['phi']))
            ts_theta_vals.append(float(row['theta']))

            lm_phi_vals.append(np.array([float(row['phi_lm1']), float(row['phi_lm2'])]))
            lm_theta_vals.append(np.array([float(row['theta_lm1']), float(row['theta_lm2'])]))
            energies.append(float(row['G298 (Hartrees)']))

            ts_phi = float(row['phi'])
            ts_theta = float(row['theta'])

            x = np.sin(np.deg2rad(ts_theta)) * np.cos(np.deg2rad(ts_phi))
            y = np.sin(np.deg2rad(ts_theta)) * np.sin(np.deg2rad(ts_phi))
            z = np.cos(np.deg2rad(ts_theta))

            xyz_data.append(np.array([x, y, z]))

        # Determining the correct number of k-meand centers using RMSD tolerance criteria
        for number_clusters in range(1, len(xyz_data) + 1):
            skm = SphericalKMeans(n_clusters=number_clusters, init='k-means++', n_init=30)
            skm.fit(xyz_data)
            phi_centers = []
            theta_centers = []
            for center_coord in skm.cluster_centers_:
                r = np.sqrt(center_coord[0] ** 2 + center_coord[1] ** 2 + center_coord[2] ** 2)
                theta_new = np.rad2deg(
                    np.arctan2(np.sqrt(center_coord[0] ** 2 + center_coord[1] ** 2), center_coord[2]))
                phi_new = np.rad2deg(np.arctan2(center_coord[1], center_coord[0]))
                if phi_new < 0:
                    phi_new += 360
                phi_centers.append(round(phi_new, 1))
                theta_centers.append(round(theta_new, 1))

            arc_length_diff = 0
            for i in range(0, skm.n_clusters):
                center_phi = phi_centers[i]
                center_theta = theta_centers[i]
                for k in range(0, len(skm.labels_)):
                    if i == skm.labels_[k]:
                        arc_length_diff += math.pow(
                            arc_length_calculator(center_phi, center_theta, ts_phi_vals[k], ts_theta_vals[k]), 2)
            rmsd = math.sqrt(arc_length_diff / len(skm.labels_))
            if rmsd < 0.1:
                if show_status is True:
                    pass
                    # TODO: plot each of the TS group assignments to visualize
                    # matplotlib_printing_localmin_transition(lm_phi_vals, lm_theta_vals, ts_phi_vals, ts_theta_vals,
                    #                                         phi_centers, theta_centers, group_key)

                break

        inner_dict = {}
        inner_dict['center_phi'] = phi_centers
        inner_dict['center_theta'] = theta_centers
        inner_dict['lm_vals_phi'] = lm_phi_vals
        inner_dict['lm_vals_theta'] = lm_theta_vals
        inner_dict['ts_vals_phi'] = ts_phi_vals
        inner_dict['ts_vals_theta'] = ts_theta_vals
        inner_dict['skm_labels'] = skm.labels_
        inner_dict['num_clusters'] = skm.n_clusters
        inner_dict['gibbs_energy'] = energies

        lm_lm_dict[group_key] = inner_dict

    num_ts = checking_accurate_sorting(lm_lm_dict)
    if num_ts != len(data_points):
        print('ERROR: THERE IS A TS MISSING FORM THE LM_LM_DICT')

    return lm_lm_dict


def checking_accurate_sorting(lm_lm_dict):
    count = 0
    for key_path, val_path in lm_lm_dict.items():
        count += len(val_path['ts_vals_phi'])

    return count


def return_lowest_value(value1, value2):
    if float(value1) < float(value2):
        lowest_val = value1
        highest_val = value2
    elif float(value2) < float(value1):
        lowest_val = value2
        highest_val = value1
    elif float(value1) == float(value2):
        lowest_val = value1
        highest_val = value2

    return lowest_val, highest_val
# endregion
