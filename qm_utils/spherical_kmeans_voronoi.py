#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose if this script is to perform spherical k-means and spherical voronoi to determine the reference structures
for HSP. The output will serve as the foundation for making all comparisons across different methods.

"""

from __future__ import print_function


import os
import sys
import numpy as np
from qm_utils.qm_common import read_csv_to_dict, create_out_fname
from spherecluster import SphericalKMeans
from scipy.spatial import SphericalVoronoi
import statistics as st

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

# Constants #



# Hartree field headers





def read_csv_data(filename, dir):
    """
    Reads the CSV file with the information
    :param filename: the filename of the CSV file
    :param dir: the directory
    :return: data points (np array containing the x, y, z coordinates), phi_raw (phi coords), theta_raw (theta coords)
    """

    file_sample_lm = os.path.join(dir, filename)
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

    ind_dict['phi_sv_centers'] = phi_vertices
    ind_dict['theta_sv_centers'] = theta_vertices

    return ind_dict


def matplotlib_printing_normal(data_dict, dir, save_status='no'):

    # The data from the previous
    phi_raw        = data_dict['phi_raw']
    theta_raw      = data_dict['theta_raw']
    phi_centers    = data_dict['phi_skm_centers']
    theta_centers  = data_dict['theta_skm_centers']
    phi_vertices   = data_dict['phi_sv_centers']
    theta_vertices = data_dict['theta_sv_centers']

    fig, ax = plt.subplots(facecolor='white')


    major_ticks = np.arange(0, 372, 60)
    minor_ticks = np.arange(0, 372, 12)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    hsp = ax.scatter(phi_raw, theta_raw, s=60, c='blue', marker='o')
    kmeans = ax.scatter(phi_centers, theta_centers, s=60, c='red', marker='h')
    voronoi = ax.scatter(phi_vertices, theta_vertices, s=60, c='green', marker='s')
    ax.set_ylim([180, 0])
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Theta (degrees)')

    leg = ax.legend((hsp, kmeans, voronoi),
              ('HSP local minima', 'k-means center (k = ' + str(data_dict['number_clusters']) + ')', 'voronoi vertice'),
                scatterpoints = 1, fontsize = 12, frameon='false')

    leg.get_frame().set_linewidth(0.0)

    if save_status != 'no':
        filename = create_out_fname('bxyl-k' + str(data_dict['number_clusters']) + '-normal.png', base_dir=dir)
        plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
    else:
        plt.show()


    return


def matplotlib_printing_size(data_dict, dir, save_status='no'):

    # The data from the previous
    phi_raw        = data_dict['phi_raw']
    theta_raw      = data_dict['theta_raw']
    phi_centers    = data_dict['phi_skm_centers']
    theta_centers  = data_dict['theta_skm_centers']
    phi_vertices   = data_dict['phi_sv_centers']
    theta_vertices = data_dict['theta_sv_centers']
    energy         = data_dict['energy']

    max_energy = float(max(energy))
    med_energy = float(st.median(energy))
    size = []

    for row in energy:
        size.append(80 * (1 - (float(row) / max_energy)))
        # size.append(60 * float(row) / med_energy)

    fig, ax = plt.subplots(facecolor='white')


    major_ticks = np.arange(0, 372, 60)
    minor_ticks = np.arange(0, 372, 12)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)



    kmeans = ax.scatter(phi_centers, theta_centers, s=60, c='red', marker='h')
    hsp = ax.scatter(phi_raw, theta_raw, s=size, c='blue', marker='o')
    ax.set_ylim([180, 0])
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Theta (degrees)')

    leg = ax.legend((hsp, kmeans),
                    ('HSP local minima', 'k-means center (k = ' + str(data_dict['number_clusters']) + ')', 'voronoi vertice'),
                    scatterpoints = 1, fontsize = 12, frameon='false')

    leg.get_frame().set_linewidth(0.0)

    if save_status != 'no':
        filename = create_out_fname('bxyl-k' + str(data_dict['number_clusters']) + '-size.png', base_dir=dir)
        plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
    else:
        plt.show()


    return
