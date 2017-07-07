import os

import numpy as np
from spherecluster import SphericalKMeans
from scipy.spatial import SphericalVoronoi
import math

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import csv

import copy

from qm_utils.qm_common import create_out_fname

from qm_utils.spherical_kmeans_voronoi import read_csv_to_dict, read_csv_canonical_designations,\
                                                pol2cart, cart2pol, plot_line, arc_length_calculator,\
                                                split_in_two, is_end, get_pol_coords

###################################### Directories ######################################
#                                                                                       #
#########################################################################################
#region
QM_1_DIR = os.path.dirname(__file__)

# root of project
QM_0_DIR = os.path.dirname(QM_1_DIR)

PROG_DATA_DIR = os.path.join(QM_0_DIR, 'pucker_prog_data')

COMP_CLASSES_DIR = os.path.join(PROG_DATA_DIR, 'comparison_classes')
SV_DIR = os.path.join(PROG_DATA_DIR, 'spherical_kmeans_voronoi')
#endregion

#check freq in parsing#

NUM_CLUSTERS = 38
REGION_THRESHOLD = 30
DEFAULT_TEMPERATURE = 298.15
K_B = 0.001985877534  # Boltzmann Constant in kcal/mol K
HART2KCAL = 627.509

REFERENCE = 'REFERENCE'

################################### Helper Functions#####################################
#                                                                                       #
#########################################################################################
#region
def is_south(pathway):
    theta_threshold = 180 - REGION_THRESHOLD

    if pathway.TS.theta > theta_threshold or pathway.LM1.theta > theta_threshold or pathway.LM2.theta > theta_threshold:
        return True
    else:
        return False

def is_north(pathway):
    theta_threshold = REGION_THRESHOLD

    ts = pathway.TS.theta
    lm1 = pathway.LM1.theta
    lm2 = pathway.LM2.theta

    if pathway.TS.theta < theta_threshold or pathway.LM1.theta < theta_threshold or pathway.LM2.theta < theta_threshold:
        return True
    else:
        return False

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

#endregion

# class for initializing plots
class Plots():
    # arguments are bools for creating the 2d and 3d plots
    def __init__(self, twoD_arg=False, threeD_arg=False, merc_arg=False, ts_merc_arg=False, north_pol_arg=False, south_pol_arg=False, rect_arg=False, rxn_arg=False):
        self.fig = plt.figure()

        if rect_arg:
            self.fig, self.ax_rect = plt.subplots(facecolor='white')

        if twoD_arg:
            norm_param = 1

            gs = gridspec.GridSpec(2 * norm_param, 2 * norm_param)
            self.ax_rect = self.fig.add_subplot(gs[(1 * norm_param):, :], facecolor='white')
            self.ax_circ_north = self.fig.add_subplot(gs[0, 0], projection='polar')
            self.ax_circ_south = self.fig.add_subplot(gs[0, 1 * norm_param], projection='polar')

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

        if rxn_arg:
            self.fig, self.ax_rect = plt.subplots(facecolor='white')

            self.rxn_init()

    def rxn_init(self):
        major_ticksy = np.arange(0, 20, 2)
        minor_ticksy = np.arange(0, 20, 1)
        self.ax_rect.set_yticks(major_ticksy)
        self.ax_rect.set_yticks(minor_ticksy, minor=True)

        self.ax_rect.set_xlabel('Reaction Coordinate')
        self.ax_rect.set_ylabel(r'$\Delta$' + 'G (kcal/mol)')

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
        self.ax_circ_north.set_thetagrids(thetaticks, frac=1.25, fontsize=12, zorder=-100)
        self.ax_circ_north.set_theta_direction(-1)

    def ax_circ_south_init(self):
        thetaticks = np.arange(0, 360, 30)

        self.ax_circ_south.set_rlim([0, 1])
        self.ax_circ_south.set_rticks([0.5])  # less radial ticks
        self.ax_circ_south.set_rlabel_position(-22.5)  # get radial labels away from plotted line
        self.ax_circ_south.set_title("Southern", ha='right', va='bottom', loc='left', fontsize=12)
        self.ax_circ_south.set_theta_zero_location("N")
        self.ax_circ_south.set_yticklabels([])
        self.ax_circ_south.set_thetagrids(thetaticks, frac=1.25, fontsize=12, zorder=-100)
        self.ax_circ_south.set_theta_direction(-1)

    # shows all plots
    def show(self):
        plt.show()

    def save(self, filename, dir_):
        filename1 = create_out_fname(filename, base_dir=dir_, ext='.png')
        self.fig.set_size_inches(9, 5)
        self.fig.savefig(filename1, facecolor=self.fig.get_facecolor(), transparent=True, dpi=300, bbox_inches='tight')

# # # Structures # # #
#region
class Structure():
    def __init__(self, phi, theta, gibbs=None, name=None):
        self.phi = phi
        self.theta = theta
        self.gibbs = gibbs
        self.name = name

class Local_Minimum(Structure):
    def __init__(self, phi, theta, gibbs=None, name=None):
        Structure.__init__(self, phi, theta, gibbs, name)

class Transition_State(Structure):
    def __init__(self, phi, theta, gibbs, name):
        Structure.__init__(self, phi, theta, gibbs, name)
#endregion

# # # Pathways # # #
#region
class Pathway():
    def __init__(self, TS, LM1, LM2):
        self.TS = TS
        self.LM1 = LM1
        self.LM2 = LM2

class Method_Pathways():
    def __init__(self, LM_csv_filename, TS_csv_filename, IRC_csv_filename, method):
        self.method = method

        self.parse_LM_csv(LM_csv_filename)
        self.parse_TS_csv(TS_csv_filename)
        self.parse_IRC_csv(IRC_csv_filename)

        self.create_structure_list()
        self.normalize_energies()

        self.create_Pathways()
        self.check_energies()

    def check_energies(self):
        for i in range(len(self.Pathways)):
            TS_gibbs = self.Pathways[i].TS.gibbs
            LM1_gibbs = self.Pathways[i].LM1.gibbs
            LM2_gibbs = self.Pathways[i].LM2.gibbs

            try:
                assert(TS_gibbs > LM1_gibbs and TS_gibbs > LM2_gibbs)
            except AssertionError:
                print('TS energy not greater than both LMs')
                print('TS.gibbs = ' + TS_gibbs)
                print('LM1.gibbs = ' + LM1_gibbs)
                print('LM2.gibbs = ' + LM2_gibbs)
        pass

    def parse_LM_csv(self, LM_csv_filename):
        self.LM_csv_list = []

        LM_csv_dict = read_csv_to_dict(LM_csv_filename, mode='r')

        for i in range(len(LM_csv_dict)):
            info = LM_csv_dict[i]
            phi = float(info['phi'])
            theta = float(info['theta'])
            gibbs = float(info['G298 (Hartrees)'])
            name = info['Pucker']

            try:
                assert(float(info['Freq 1']) > 0)
            except AssertionError:
                print('Local Minimum has a negative Freq 1')
                exit(1)

            self.LM_csv_list.append(Local_Minimum(phi, theta, gibbs, name))

    def parse_TS_csv(self, TS_csv_filename):
        self.TS_csv_list = []

        TS_csv_dict = read_csv_to_dict(TS_csv_filename, mode='r')

        for i in range(len(TS_csv_dict)):
            info = TS_csv_dict[i]
            phi = float(info['phi'])
            theta = float(info['theta'])
            gibbs = float(info['G298 (Hartrees)'])
            name = info['File Name'].split('-')[1].split('-')[0]

            try:
                assert(float(info['Freq 1']) < 0)
            except AssertionError:
                print('Transition State has a non-negative Freq 1')
                exit(1)

            self.TS_csv_list.append(Transition_State(phi, theta, gibbs, name))

    def parse_IRC_csv(self, IRC_csv_filename):
        self.IRC_csv_list = []

        IRC_csv_dict = read_csv_to_dict(IRC_csv_filename, mode='r')

        for i in range(len(IRC_csv_dict)):
            info = IRC_csv_dict[i]
            phi = float(info['phi'])
            theta = float(info['theta'])
            gibbs = float(info['G298 (Hartrees)'])

            try:
                assert(float(info['Freq 1']) > 0)
            except AssertionError:
                print('IRC Local Minimum has a negative Freq 1')
                exit(1)

            if 'ircr' in info['File Name']:
                direction = 'ircr'
            else:
                direction = 'ircf'

            pucker = info['File Name'].split('-')[1].split('-')[0]

            name = pucker + '_' + direction

            self.IRC_csv_list.append(Local_Minimum(phi, theta, gibbs, name))

    def create_structure_list(self):
        self.structure_list = []

        for i in range(len(self.TS_csv_list)):
            self.structure_list.append(self.TS_csv_list[i])

        for i in range(len(self.IRC_csv_list)):
            self.structure_list.append(self.IRC_csv_list[i])

        for i in range(len(self.LM_csv_list)):
            self.structure_list.append(self.LM_csv_list[i])

    def normalize_energies(self):
        self.min_gibbs = self.structure_list[0].gibbs

        for i in range(len(self.structure_list)):
            curr_gibbs = self.structure_list[i].gibbs

            if curr_gibbs < self.min_gibbs:
                self.min_gibbs = curr_gibbs

        for i in range(len(self.structure_list)):
            self.structure_list[i].gibbs -= self.min_gibbs
            self.structure_list[i].gibbs *= HART2KCAL

    def create_Pathways(self):
        self.Pathways = []

        for i in range(len(self.TS_csv_list)):
            TS = self.TS_csv_list[i]
            LM1 = None
            LM2 = None

            j = 0
            while LM1 is None or LM2 is None:
                if TS.name in self.IRC_csv_list[j].name:
                    if 'ircf' in self.IRC_csv_list[j].name:
                        LM1 = self.IRC_csv_list[j]
                    elif 'ircr' in self.IRC_csv_list[j].name:
                        LM2 = self.IRC_csv_list[j]

                j += 1

            self.Pathways.append(Pathway(TS, LM1, LM2))

class Reference_Pathways():
    def __init__(self, LM_csv_filename, TS_csv_filename):
        self.method = 'REFERENCE'

        self.parse_LM_csv(LM_csv_filename)
        self.parse_TS_csv(TS_csv_filename)

        self.create_structure_list()

    def check_energies(self):
        for i in range(len(self.Pathways)):
            TS_gibbs = self.Pathways[i].TS.gibbs
            LM1_gibbs = self.Pathways[i].LM1.gibbs
            LM2_gibbs = self.Pathways[i].LM2.gibbs

            try:
                assert(TS_gibbs > LM1_gibbs and TS_gibbs > LM2_gibbs)
            except AssertionError:
                print('TS energy not greater than both LMs')
                print('TS.gibbs = ' + TS_gibbs)
                print('LM1.gibbs = ' + LM1_gibbs)
                print('LM2.gibbs = ' + LM2_gibbs)
        pass

    def parse_LM_csv(self, LM_csv_filename):
        self.LM_csv_list = []

        LM_csv_dict = read_csv_to_dict(LM_csv_filename, mode='r')

        for i in range(len(LM_csv_dict)):
            info = LM_csv_dict[i]
            phi = float(info['phi'])
            theta = float(info['theta'])
            gibbs = float(info['G298 (Hartrees)'])
            name = info['Pucker']

            self.LM_csv_list.append(Local_Minimum(phi, theta, gibbs, name))

    def parse_TS_csv(self, TS_csv_filename):
        TS_csv_dict = read_csv_to_dict(TS_csv_filename, mode='r')

        self.Pathways = []

        for i in range(len(TS_csv_dict)):
            info = TS_csv_dict[i]
            phi = float(info['phi'])
            theta = float(info['theta'])
            gibbs = float(info['G298 (Hartrees)'])
            name = info['Pucker']

            phi_lm1 = float(info['phi_lm1'])
            theta_lm1 = float(info['theta_lm1'])
            phi_lm2 = float(info['phi_lm2'])
            theta_lm2 = float(info['theta_lm2'])

            try:
                assert(float(info['Freq 1']) < 0)
            except AssertionError:
                print('Transition State has a non-negative Freq 1')
                exit(1)

            TS = Transition_State(phi, theta, gibbs, name)
            LM1 = Local_Minimum(phi=phi_lm1,
                                theta=theta_lm1)
            LM2 = Local_Minimum(phi=phi_lm2,
                                theta=theta_lm2)

            self.Pathways.append(Pathway(TS, LM1, LM2))

    def create_structure_list(self):
        self.structure_list = []

        for i in range(len(self.Pathways)):
            self.structure_list.append(self.Pathways[i].TS)

        for i in range(len(self.LM_csv_list)):
            self.structure_list.append(self.LM_csv_list[i])

    def normalize_energies(self):
        self.min_gibbs = self.structure_list[0].gibbs

        for i in range(len(self.structure_list)):
            curr_gibbs = self.structure_list[i].gibbs

            if curr_gibbs < self.min_gibbs:
                self.min_gibbs = curr_gibbs

        for i in range(len(self.LM_csv_list)):
            self.LM_csv_list[i].gibbs -= self.min_gibbs
            self.LM_csv_list[i].gibbs *= HART2KCAL

        for i in range(len(self.Pathways)):
            TS = self.Pathways[i].TS
            LM1 = self.Pathways[i].LM1
            LM2 = self.Pathways[i].LM2

            TS.gibbs -= self.min_gibbs
            TS.gibbs *= HART2KCAL

            LM1.gibbs -= self.min_gibbs
            LM1.gibbs *= HART2KCAL

            LM2.gibbs -= self.min_gibbs
            LM2.gibbs *= HART2KCAL
#endregion

class Reference_Landscape():
    # # # Init # # #
    #region
    def __init__(self, LM_csv_filename, TS_csv_filename):
        self.method = 'REFERENCE'

        self.Reference_Pathways = Reference_Pathways(LM_csv_filename, TS_csv_filename)

        self.Pathways = self.Reference_Pathways.Pathways
        self.Local_Minima = self.Reference_Pathways.LM_csv_list
        self.Reference_Structures = self.get_Reference_Structures()

        self.canonical_designations = read_csv_canonical_designations('CP_params.csv', SV_DIR)
        self.reorg_canonical()
        self.populate_unbinned_canos()

        self.tessellate(NUM_CLUSTERS)
        self.assign_region_names()
        self.assign_skm_labels()

        self.placeholder_IRC_energies()
        self.Reference_Pathways.normalize_energies()
        self.Reference_Pathways.check_energies()

    def get_avg_LM_energy(self, LM):
        energy = 0
        num_points = 0

        for j in range(len(self.Local_Minima)):
            if self.Local_Minima[j].closest_skm == LM.closest_skm:
                energy += self.Local_Minima[j].gibbs
                num_points += 1

        avg_energy = energy / num_points

        LM.gibbs = avg_energy

    def placeholder_IRC_energies(self):
        for i in range(len(self.Reference_Pathways.Pathways)):
            LM1 = self.Reference_Pathways.Pathways[i].LM1
            LM2 = self.Reference_Pathways.Pathways[i].LM2

            self.get_avg_LM_energy(LM1)
            self.get_avg_LM_energy(LM2)


    def get_Reference_Structures(self):
        structures_list = []

        for i in range(len(self.Pathways)):
            structures_list.append(self.Pathways[i].TS)
            structures_list.append(self.Pathways[i].LM1)
            structures_list.append(self.Pathways[i].LM2)

        for i in range(len(self.Local_Minima)):
            structures_list.append(self.Local_Minima[i])

        return structures_list

    def reorg_canonical(self):
        aux_list = []

        for i in range(len(self.canonical_designations['pucker'])):
            name = self.canonical_designations['pucker'][i]
            phi = float(self.canonical_designations['phi_cano'][i])
            theta = float(self.canonical_designations['theta_cano'][i])

            aux_list.append(Structure(phi=phi,
                                      theta=theta,
                                      name=name))

        self.canonical_designations = aux_list

    def populate_unbinned_canos(self):
        self.unbinned_canos = []

        for i in range(len(self.canonical_designations)):
            unbinned = True

            for j in range(len(self.Reference_Structures)):
                structure = self.Reference_Structures[j]

                if self.get_closest_cano(structure.phi, structure.theta) == self.canonical_designations[i].name:
                    unbinned = False

            if unbinned:
                self.unbinned_canos.append(self.canonical_designations[i])

    def tessellate(self, number_clusters):
        centers = []

        # populate all centers to be used in voronoi tessellation
        for i in range(len(self.Reference_Structures)):
            structure = self.Reference_Structures[i]
            center = pol2cart([structure.phi, structure.theta])

            centers.append(center)

        for i in range(len(self.unbinned_canos)):
            structure = self.unbinned_canos[i]

            center = pol2cart([float(structure.phi), float(structure.theta)])

            centers.append(center)

        # Uses packages to calculate the k-means spherical centers
        self.skm = SphericalKMeans(n_clusters=number_clusters, init='k-means++', n_init=30)
        self.skm.fit(centers)
        skm_centers = self.skm.cluster_centers_

        self.reorg_clusters()

        # Default parameters for spherical voronoi
        radius = 1
        center = np.array([0, 0, 0])

        # Spherical Voronoi for the centers
        self.sv = SphericalVoronoi(skm_centers, radius, center)
        self.sv.sort_vertices_of_regions()

    def reorg_clusters(self):
        exchanges = True

        passnum = len(self.skm.cluster_centers_) - 1
        while passnum > 0 and exchanges:
            exchanges = False

            for i in range(passnum):
                vert = cart2pol(self.skm.cluster_centers_[i])
                vert2 = cart2pol(self.skm.cluster_centers_[i + 1])

                theta = vert[1]
                theta2 = vert2[1]

                if theta > theta2:
                    exchanges = True

                    aux_vert = copy.deepcopy(self.skm.cluster_centers_[i + 1])
                    self.skm.cluster_centers_[i + 1] = self.skm.cluster_centers_[i]
                    self.skm.cluster_centers_[i] = aux_vert

            passnum = passnum - 1

    def get_closest_cano(self, phi, theta):
        min_dist = 100

        for i in range(len(self.canonical_designations)):
            phi2 = self.canonical_designations[i].phi
            theta2 = self.canonical_designations[i].theta

            curr_dist = arc_length_calculator(phi1=phi, theta1=theta,
                                              phi2=phi2, theta2=theta2)

            if curr_dist < min_dist:
                min_dist = curr_dist
                name = self.canonical_designations[i].name

        return name

    def assign_region_names(self):
        self.skm_name_list = []

        for i in range(len(self.skm.cluster_centers_)):
            vert = cart2pol(self.skm.cluster_centers_[i])

            phi = vert[0]
            theta = vert[1]

            name = self.get_closest_cano(phi, theta)

            self.skm_name_list.append(name)

    def assign_skm_labels(self):
        for i in range(len(self.Pathways)):
            TS = self.Pathways[i].TS
            LM1 = self.Pathways[i].LM1
            LM2 = self.Pathways[i].LM2

            self.calc_closest_skm(TS)
            self.calc_closest_skm(LM1)
            self.calc_closest_skm(LM2)

        for i in range(len(self.Local_Minima)):
            self.calc_closest_skm(self.Local_Minima[i])

    def calc_closest_skm(self, structure):
        min_dist = 100

        phi1 = structure.phi
        theta1 = structure.theta

        for i in range(len(self.skm.cluster_centers_)):
            center = cart2pol(self.skm.cluster_centers_[i])

            phi2 = center[0]
            theta2 = center[1]

            curr_dist = arc_length_calculator(phi1=phi1, theta1=theta1,
                                              phi2=phi2, theta2=theta2)

            if curr_dist < min_dist:
                min_dist = curr_dist
                skm_index = i

        structure.closest_skm = skm_index
    # endregion

    # # # Plotting # # #
    #region
    def plot_voronoi_regions(self, plot):
        color = 'lightgray'

        for i in range(len(self.sv.regions)):
            for j in range(len(self.sv.regions[i])):
                if j == len(self.sv.regions[i]) - 1:
                    index1 = self.sv.regions[i][j]
                    index2 = self.sv.regions[i][0]

                    vert1 = self.sv.vertices[index1]
                    vert2 = self.sv.vertices[index2]

                    plot_line(plot.ax_rect, [vert1, color, 0], [vert2, color, 0], color)
                else:
                    index1 = self.sv.regions[i][j]
                    index2 = self.sv.regions[i][j + 1]

                    vert1 = self.sv.vertices[index1]
                    vert2 = self.sv.vertices[index2]

                    plot_line(plot.ax_rect, [vert1, color, 0], [vert2, color, 0], color)

        self.plot_regions_names(plot)

    # gets highest (lowest) theta of a region
    def get_highest_theta(self, region):
        theta_vals = []

        for i in range(len(region)):
            vert_index = region[i]
            vert = cart2pol(self.sv.vertices[vert_index])

            theta_vals.append(vert[1])

        min_theta = 360

        for i in range(len(theta_vals)):
            if theta_vals[i] < min_theta:
                min_theta = theta_vals[i]

        return min_theta

    def plot_regions_names(self, plot):
        for i in range(len(self.skm_name_list)):
            name = self.skm_name_list[i]

            region = self.sv.regions[i]

            theta = self.get_highest_theta(region) + 7
            phi = cart2pol(self.skm.cluster_centers_[i])[0] - 5

            if i == 0:
                theta = 20
                phi = 180

            plot.ax_rect.annotate(name, xy=(phi, theta), xytext=(phi, theta), fontsize=8)

    def plot_skm_centers(self, plot):
        phi_vals = []
        theta_vals = []

        for i in range(len(self.Pathways)):
            TS_skm = self.Pathways[i].TS.closest_skm
            LM1_skm = self.Pathways[i].LM1.closest_skm
            LM2_skm = self.Pathways[i].LM2.closest_skm

            TS_vert = cart2pol(self.skm.cluster_centers_[TS_skm])
            phi_vals.append(TS_vert[0])
            theta_vals.append(TS_vert[1])

            LM1_vert = cart2pol(self.skm.cluster_centers_[LM1_skm])
            phi_vals.append(LM1_vert[0])
            theta_vals.append(LM1_vert[1])

            LM2_vert = cart2pol(self.skm.cluster_centers_[LM2_skm])
            phi_vals.append(LM2_vert[0])
            theta_vals.append(LM2_vert[1])

        plot.ax_rect.scatter(phi_vals, theta_vals, c='red', marker='x', s=60)

    def plot_cano(self, plot):
        phi_vals = []
        theta_vals = []

        for i in range(len(self.canonical_designations)):
            cano = self.canonical_designations[i]

            phi_vals.append(cano.phi)
            theta_vals.append(cano.theta)

        plot.ax_rect.scatter(phi_vals, theta_vals, c='black', marker='+', s=60)
    #endregion

    pass

class Compare_Methods():
    # # # Init # # #
    #region
    def __init__(self, molecule):
        self.molecule = molecule
        self.dir_init()

        self.reference_landscape_init()

        self.Method_Pathways_dict = {}
        self.Method_Pathways_init()

        self.assign_structure_names()

        self.assign_met_colors_and_markers()
        self.assign_skm_labels()

        self.pathway_groupings_init()

        self.populate_skm_groupings(REFERENCE)
        self.populate_pathway_groupings(REFERENCE)
        self.populate_local_minima(REFERENCE)
        self.do_calcs(REFERENCE)

        for method in self.Method_Pathways_dict:
            self.populate_skm_groupings(method)
            self.populate_pathway_groupings(method)
            self.populate_local_minima(method)
            self.do_calcs(method)

            self.normalize_pathways(method)

        pass

    def normalize_pathways(self, method):
        for key in self.Method_Pathways_dict[method].pathway_groupings:
            ref_pathways = len(self.Method_Pathways_dict[REFERENCE].pathway_groupings[key]['pathways'])
            met_pathways = len(self.Method_Pathways_dict[method].pathway_groupings[key]['pathways'])

            self.Method_Pathways_dict[method].pathway_groupings[key]['norm_pathways'] = np.abs(met_pathways - ref_pathways)

    def normalize_comp_metrics(self, method, comp_key):
        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            ref_metric = self.Method_Pathways_dict[REFERENCE].skm_groupings[i][comp_key + '_group_WRMSD']
            met_metric = self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_WRMSD']

            if ref_metric != 'n/a' and met_metric != 'n/a':
                if ref_metric == 0:
                    self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_norm_group_WRMSD'] = met_metric
                else:
                    self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_norm_group_WRMSD'] = met_metric / ref_metric
            else:
                self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_norm_group_WRMSD'] = 'n/a'


    def dir_init(self):
        self.MOL_DATA_DIR = make_dir(os.path.join(SV_DIR, self.molecule))
        self.MOL_SAVE_DIR = make_dir(os.path.join(COMP_CLASSES_DIR, self.molecule))

        self.IRC_DATA_DIR = make_dir(os.path.join(self.MOL_DATA_DIR, 'IRC'))
        self.LM_DATA_DIR = make_dir(os.path.join(self.MOL_DATA_DIR, 'LM'))
        self.TS_DATA_DIR = make_dir(os.path.join(self.MOL_DATA_DIR, 'TS'))

        self.IRC_DATA_dir_list = os.listdir(os.path.join(self.MOL_DATA_DIR, 'IRC'))

    def reference_landscape_init(self):
        ref_LM_csv_filename = os.path.join(self.MOL_DATA_DIR, 'z_oxane_LM-b3lyp_howsugarspucker.csv')
        ref_TS_csv_filename = os.path.join(self.MOL_DATA_DIR, 'z_oxane_TS-b3lyp_howsugarspucker.csv')

        self.reference_landscape = Reference_Landscape(LM_csv_filename=ref_LM_csv_filename,
                                                       TS_csv_filename=ref_TS_csv_filename)

    def assign_structure_names(self):
        for method in self.Method_Pathways_dict:
            for i in range(len(self.Method_Pathways_dict[method].Pathways)):
                TS = self.Method_Pathways_dict[method].Pathways[i].TS
                LM1 = self.Method_Pathways_dict[method].Pathways[i].LM1
                LM2 = self.Method_Pathways_dict[method].Pathways[i].LM2

                TS.name = self.reference_landscape.get_closest_cano(TS.phi, TS.theta)
                LM1.name = self.reference_landscape.get_closest_cano(LM1.phi, LM1.theta)
                LM2.name = self.reference_landscape.get_closest_cano(LM2.phi, LM2.theta)

    def Method_Pathways_init(self):
        for i in range(len(self.IRC_DATA_dir_list)):
            method = self.IRC_DATA_dir_list[i].split('-')[3].split('.')[0]

            IRC_csv_filename = os.path.join(self.IRC_DATA_DIR, 'z_dataset-oxane-IRC-' + method + '.csv')
            LM_csv_filename = os.path.join(self.LM_DATA_DIR, 'z_dataset-oxane-LM-' + method + '.csv')
            TS_csv_filename = os.path.join(self.TS_DATA_DIR, 'z_dataset-oxane-TS-' + method + '.csv')

            self.Method_Pathways_dict[method] = (Method_Pathways(LM_csv_filename=LM_csv_filename,
                                                             TS_csv_filename=TS_csv_filename,
                                                             IRC_csv_filename=IRC_csv_filename,
                                                             method=method.upper()))

            self.Method_Pathways_dict[method].comp_metrics = {}

        self.Method_Pathways_dict['REFERENCE'] = self.reference_landscape.Reference_Pathways
        self.Method_Pathways_dict['REFERENCE'].comp_metrics = {}


    # creates a list of structures grouped by skm
    def populate_skm_groupings(self, method):
        self.Method_Pathways_dict[method].skm_groupings = []

        for i in range(len(self.reference_landscape.skm.cluster_centers_)):
            self.Method_Pathways_dict[method].skm_groupings.append({})
            self.Method_Pathways_dict[method].skm_groupings[i]['structures'] = []
            structures = self.Method_Pathways_dict[method].skm_groupings[i]['structures']

            for j in range(len(self.Method_Pathways_dict[method].Pathways)):
                TS = self.Method_Pathways_dict[method].Pathways[j].TS
                LM1 = self.Method_Pathways_dict[method].Pathways[j].LM1
                LM2 = self.Method_Pathways_dict[method].Pathways[j].LM2

                if TS.closest_skm == i:
                    structures.append(TS)
                if LM1.closest_skm == i:
                    structures.append(LM1)
                if LM2.closest_skm == i:
                    structures.append(LM2)

            for j in range(len(self.Method_Pathways_dict[method].LM_csv_list)):
                LM = self.Method_Pathways_dict[method].LM_csv_list[j]

                if LM.closest_skm == i:
                    structures.append(LM)


    def pathway_groupings_init(self):
        self.pathway_groupings = {}
        pathway_groupings = self.pathway_groupings

        for method in self.Method_Pathways_dict:
            for i in range(len(self.Method_Pathways_dict[method].Pathways)):
                TS = self.Method_Pathways_dict[method].Pathways[i].TS
                LM1 = self.Method_Pathways_dict[method].Pathways[i].LM1
                LM2 = self.Method_Pathways_dict[method].Pathways[i].LM2

                if LM1.closest_skm < LM2.closest_skm:
                    lm_grouping = str(LM1.closest_skm) + '_' + str(LM2.closest_skm)
                else:
                    lm_grouping = str(LM2.closest_skm) + '_' + str(LM1.closest_skm)

                key = lm_grouping + '-' + str(TS.closest_skm)

                if key not in self.pathway_groupings:
                    pathway_groupings[key] = {}
                    pathway_groupings[key]['pathways'] = []

    # creates a dict of raw pathway data
    def populate_pathway_groupings(self, method):
        self.Method_Pathways_dict[method].pathway_groupings = copy.deepcopy(self.pathway_groupings)
        pathway_groupings = self.Method_Pathways_dict[method].pathway_groupings

        for i in range(len(self.Method_Pathways_dict[method].Pathways)):
            TS = self.Method_Pathways_dict[method].Pathways[i].TS
            LM1 = self.Method_Pathways_dict[method].Pathways[i].LM1
            LM2 = self.Method_Pathways_dict[method].Pathways[i].LM2

            if LM1.closest_skm < LM2.closest_skm:
                lm_grouping = str(LM1.closest_skm) + '_' + str(LM2.closest_skm)
            else:
                lm_grouping = str(LM2.closest_skm) + '_' + str(LM1.closest_skm)

            key = lm_grouping + '-' + str(TS.closest_skm)

            if key not in self.Method_Pathways_dict[method].pathway_groupings:
                pathway_groupings[key] = {}
                pathway_groupings[key]['pathways'] = []

            pathway_groupings[key]['pathways'].append(self.Method_Pathways_dict[method].Pathways[i])

    # creates a dict of raw local minima data
    def populate_local_minima(self, method):
        IRC_lm_list = []

        pathways = self.Method_Pathways_dict[method].Pathways

        for i in range(len(pathways)):
            IRC_lm_list.append(pathways[i].LM1)
            IRC_lm_list.append(pathways[i].LM2)

        LM_list = self.Method_Pathways_dict[method].LM_csv_list


    def assign_skm_labels(self):
        for method in self.Method_Pathways_dict:
            pathways = self.Method_Pathways_dict[method].Pathways

            for item in list(pathways):
                TS = item.TS
                LM1 = item.LM1
                LM2 = item.LM2

                TS.comp_metrics = {}
                LM1.comp_metrics = {}
                LM2.comp_metrics = {}

                self.calc_closest_skm(TS)
                self.calc_closest_skm(LM1)
                self.calc_closest_skm(LM2)

                if LM1.closest_skm == LM2.closest_skm:
                    pathways.remove(item)

                    print('A pathway was not included since its LM1 and LM2 are the same structure.')
                    print('class: Compare_Methods')
                    print('function: assign_skm_labels')
                    print('method: ' + method)

            for i in range(len(self.Method_Pathways_dict[method].LM_csv_list)):
                LM = self.Method_Pathways_dict[method].LM_csv_list[i]
                LM.comp_metrics = {}
                self.calc_closest_skm(LM)

    def calc_closest_skm(self, structure):
        min_dist = 100

        phi1 = structure.phi
        theta1 = structure.theta

        for i in range(len(self.reference_landscape.skm.cluster_centers_)):
            center = cart2pol(self.reference_landscape.skm.cluster_centers_[i])

            phi2 = center[0]
            theta2 = center[1]

            curr_dist = arc_length_calculator(phi1=phi1, theta1=theta1,
                                              phi2=phi2, theta2=theta2)

            if curr_dist < min_dist:
                min_dist = curr_dist
                skm_index = i

        structure.comp_metrics['arc'] = min_dist
        structure.closest_skm = skm_index

    def calc_gibbs_diff(self, method):
        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
                structure = self.Method_Pathways_dict[method].skm_groupings[i]['structures'][j]
                ref_structure_gibbs = self.Method_Pathways_dict[REFERENCE].skm_groupings[i]['weighted_gibbs']

                structure.comp_metrics['gibbs'] = structure.gibbs - ref_structure_gibbs


    def assign_met_colors_and_markers(self):
        cmap = plt.get_cmap('Vega20')
        # allows for incrementing over 20 colors
        increment = 0.0524
        seed_num = 0
        i = 0

        self.met_colors_dict = {}
        self.met_ts_markers_dict = {}
        self.met_lm_markers_dict = {}

        for method in self.Method_Pathways_dict:
            color = cmap(seed_num)
            seed_num += increment
            self.met_colors_dict[method] = color

            ts_marker = mpl.markers.MarkerStyle.filled_markers[i]
            lm_marker = mpl.markers.MarkerStyle.filled_markers[i]
            i += 1
            self.met_ts_markers_dict[method] = ts_marker
            self.met_lm_markers_dict[method] = lm_marker


    # # # do_calc # # #
    # region
    def do_calcs(self, method):
        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            self.calc_weighting(method, i)

        self.calc_gibbs_diff(method)

        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            self.calc_weighting(method, i)
            self.calc_WSS(method, i, 'arc')
            self.calc_group_RMSD(method, i, 'arc')
            self.calc_WSS(method, i, 'gibbs')
            self.calc_group_RMSD(method, i, 'gibbs')

            self.calc_WWSS(method, i, 'arc')
            self.calc_WWSS(method, i, 'gibbs')
            self.calc_group_WRMSD(method, i, 'arc')
            self.calc_group_WRMSD(method, i, 'gibbs')

        self.calc_SSE(method=method,
                       comp_key='arc')
        self.calc_RMSD(method=method,
                       comp_key='arc')
        self.calc_SSE(method=method,
                       comp_key='gibbs')
        self.calc_RMSD(method=method,
                       comp_key='gibbs')

        self.calc_WSSE(method=method,
                       comp_key='arc')
        self.calc_WRMSD(method=method,
                       comp_key='arc')
        self.calc_WSSE(method=method,
                       comp_key='gibbs')
        self.calc_WRMSD(method=method,
                       comp_key='gibbs')

    def calc_weighting(self, method, i):
        total_boltz = 0

        for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
            structure = self.Method_Pathways_dict[method].skm_groupings[i]['structures'][j]
            e_val = structure.gibbs

            component = math.exp(-e_val / (K_B * DEFAULT_TEMPERATURE))
            structure.ind_bolts = component
            total_boltz += component

        wt_gibbs = 0

        for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
            structure = self.Method_Pathways_dict[method].skm_groupings[i]['structures'][j]

            if structure.ind_bolts == 0:
                structure.weighting = 0
                wt_gibbs += 0
            else:
                structure.weighting = structure.ind_bolts / total_boltz
                wt_gibbs += structure.gibbs * structure.weighting

        self.Method_Pathways_dict[method].skm_groupings[i]['weighted_gibbs'] = wt_gibbs

    def calc_WSS(self, method, i, comp_key):
        WSS = 0

        for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
            structure = self.Method_Pathways_dict[method].skm_groupings[i]['structures'][j]

            comp_val = structure.comp_metrics[comp_key]
            WSS += comp_val ** 2

        self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WSS'] = round(WSS, 5)

    def calc_group_RMSD(self, method, i, comp_key):
        size = len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])

        if (size == 0):
            RMSD = 'n/a'
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_RMSD'] = RMSD
        else:
            RMSD = (self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WSS'] / size) ** 0.5
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_RMSD'] = round(RMSD, 5)

    def calc_SSE(self, method, comp_key):
        SSE = 0

        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            SSE += self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WSS']

        self.Method_Pathways_dict[method].comp_metrics[comp_key + '_SSE'] = round(SSE, 5)

    def calc_RMSD(self, method, comp_key):
        size = 0

        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
                size += 1

        RMSD = (self.Method_Pathways_dict[method].comp_metrics[comp_key + '_SSE'] / size) ** 0.5
        self.Method_Pathways_dict[method].comp_metrics[comp_key + '_RMSD'] = round(RMSD, 5)

    def calc_WWSS(self, method, i, comp_key):
        WWSS = 0

        # calculating each point's contribution
        for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
            structure = self.Method_Pathways_dict[method].skm_groupings[i]['structures'][j]

            comp_val = structure.comp_metrics[comp_key]
            weighting = structure.weighting

            WWSS += (comp_val ** 2) * weighting

        self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WWSS'] = round(WWSS, 5)

    def calc_group_WRMSD(self, method, i, comp_key):
        size = len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])

        if (size == 0):
            WRMSD = 'n/a'
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_WRMSD'] = WRMSD
        else:
            WRMSD = (self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WWSS'] / size) ** 0.5
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_WRMSD'] = round(WRMSD, 5)

    def calc_WSSE(self, method, comp_key):
        WSSE = 0

        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
                WSSE += self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WWSS']

        self.Method_Pathways_dict[method].comp_metrics[comp_key + '_WSSE'] = round(WSSE, 5)

    def calc_WRMSD(self, method, comp_key):
        size = 0

        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
                size += 1

        WRMSD = (self.Method_Pathways_dict[method].comp_metrics[comp_key + '_WSSE'] / size) ** 0.5
        self.Method_Pathways_dict[method].comp_metrics[comp_key + '_WRMSD'] = round(WRMSD, 5)
    #endregion
    #endregion

    # # # Plotting # # #
    #region
    def get_artist(self, method):
        ts_artist = plt.scatter((5000, 5000), (4999, 4999), s=30, c='', marker=self.met_ts_markers_dict[method],
                                    edgecolor=self.met_colors_dict[method])

        lm_artist = plt.scatter((5000, 5000), (4999, 4999), s=30, c=self.met_colors_dict[method], marker=self.met_lm_markers_dict[method],
                             edgecolor=self.met_colors_dict[method])

        return ts_artist, lm_artist

    def plot_raw_data(self, plot, method):
        size = 30

        ts_phi_vals = []
        ts_theta_vals = []

        lm_phi_vals = []
        lm_theta_vals = []

        pathways = self.Method_Pathways_dict[method].Pathways

        for j in range(len(pathways)):
            TS = pathways[j].TS
            LM1 = pathways[j].LM1
            LM2 = pathways[j].LM2

            TS_vert = [TS.phi, TS.theta]
            ts_phi_vals.append(TS_vert[0])
            ts_theta_vals.append(TS_vert[1])

            LM1_vert = [LM1.phi, LM1.theta]
            lm_phi_vals.append(LM1_vert[0])
            lm_theta_vals.append(LM1_vert[1])

            LM2_vert = [LM2.phi, LM2.theta]
            lm_phi_vals.append(LM2_vert[0])
            lm_theta_vals.append(LM2_vert[1])

        for j in range(len(self.Method_Pathways_dict[method].LM_csv_list)):
            LM = self.Method_Pathways_dict[method].LM_csv_list[j]

            LM_vert = [LM.phi, LM.theta]
            lm_phi_vals.append(LM_vert[0])
            lm_theta_vals.append(LM_vert[1])

        plot.ax_rect.scatter(ts_phi_vals, ts_theta_vals, c='',
                             edgecolor=self.met_colors_dict[method],
                             marker=self.met_ts_markers_dict[method],
                             s=size)
        plot.ax_rect.scatter(lm_phi_vals, lm_theta_vals, c=self.met_colors_dict[method],
                             edgecolor=self.met_colors_dict[method],
                             marker=self.met_lm_markers_dict[method],
                             s=size)

        # # if input plot is twoD
        # if plot.ax_circ_south is not None:
        #     ts_phi_north_vals = []
        #     ts_theta_north_vals = []
        #
        #     lm_phi_north_vals = []
        #     lm_theta_north_vals = []
        #
        #     ts_phi_south_vals = []
        #     ts_theta_south_vals = []
        #
        #     lm_phi_south_vals = []
        #     lm_theta_south_vals = []
        #
        #     for j in range(len(pathways)):
        #         TS = pathways[j].TS
        #         LM1 = pathways[j].LM1
        #         LM2 = pathways[j].LM2
        #
        #         TS_vert = [TS.phi, TS.theta]
        #         LM1_vert = [LM1.phi, LM1.theta]
        #         LM2_vert = [LM2.phi, LM2.theta]
        #
        #         if is_north(pathways[j]):
        #             ts_phi_north_vals.append(TS_vert[0])
        #             ts_theta_north_vals.append(abs(math.sin(np.radians(TS_vert[1]))))
        #
        #             lm_phi_north_vals.append(LM1_vert[0])
        #             lm_theta_north_vals.append(abs(math.sin(np.radians(LM1_vert[1]))))
        #
        #             lm_phi_north_vals.append(LM2_vert[0])
        #             lm_theta_north_vals.append(abs(math.sin(np.radians(LM2_vert[1]))))
        #         elif is_south(pathways[j]):
        #             ts_phi_south_vals.append(TS_vert[0])
        #             ts_theta_south_vals.append(abs(math.sin(np.radians(TS_vert[1]))))
        #
        #             lm_phi_south_vals.append(LM1_vert[0])
        #             lm_theta_south_vals.append(abs(math.sin(np.radians(LM1_vert[1]))))
        #
        #             lm_phi_south_vals.append(LM2_vert[0])
        #             lm_theta_south_vals.append(abs(math.sin(np.radians(LM2_vert[1]))))
        #
        #     plot.ax_circ_north.scatter(ts_phi_north_vals, ts_theta_north_vals, c='',
        #                                edgecolor=self.met_colors_dict[method],
        #                                marker=self.met_ts_markers_dict[method],
        #                                s=size)
        #     plot.ax_circ_north.scatter(lm_phi_north_vals, lm_theta_north_vals, c=self.met_colors_dict[method],
        #                                edgecolor=self.met_colors_dict[method],
        #                                marker=self.met_lm_markers_dict[method],
        #                                s=size)
        #     plot.ax_circ_south.scatter(ts_phi_south_vals, ts_theta_south_vals, c='',
        #                                edgecolor=self.met_colors_dict[method],
        #                                marker=self.met_ts_markers_dict[method],
        #                                s=size)
        #     plot.ax_circ_south.scatter(lm_phi_south_vals, lm_theta_south_vals, c=self.met_colors_dict[method],
        #                                edgecolor=self.met_colors_dict[method],
        #                                marker=self.met_lm_markers_dict[method],
        #                                s=size)
        #
        #     pass

    def save_raw_data(self, method='ALL'):
        filename = self.molecule + '-' + method + '-raw_data'
        dir = make_dir(os.path.join(self.MOL_SAVE_DIR, 'plots'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.reference_landscape.plot_voronoi_regions(plot=plot)
            self.reference_landscape.plot_skm_centers(plot=plot)
            self.reference_landscape.plot_cano(plot=plot)

            artist_list = []
            label_list = []

            if method == 'ALL':
                for method in self.Method_Pathways_dict:
                    self.plot_raw_data(plot=plot, method=method)

                    ts_artist, lm_artist = self.get_artist(method)

                    artist_list.append(ts_artist)
                    artist_list.append(lm_artist)

                    label_list.append('TS ' + method)
                    label_list.append('LM ' + method)

                    plot.ax_rect.legend(artist_list,
                                        label_list,
                                        scatterpoints=1, fontsize=8, frameon=True,
                                        framealpha=0.75,
                                        bbox_to_anchor=(1.2, 0.5), loc='right', borderaxespad=0,
                                        ncol=1).set_zorder(100)
            else:
                self.plot_raw_data(plot=plot, method=method)

                ts_artist, lm_artist = self.get_artist(method)

                artist_list.append(ts_artist)
                artist_list.append(lm_artist)

                label_list.append('TS ' + method)
                label_list.append('LM ' + method)

                plot.ax_rect.legend(artist_list,
                                     label_list,
                                     scatterpoints=1, fontsize=8, frameon=False,
                                     framealpha=0.75,
                                     bbox_to_anchor=(0.5, -0.15), loc=9, borderaxespad=0,
                                     ncol=len(self.met_colors_dict)).set_zorder(100)

            plot.ax_rect.set_ylim(185, -5)
            plot.ax_rect.set_xlim(-5, 365)

            plot.save(dir_=dir, filename=filename)

    def plot_line(self, plot, TS_vert, LM_vert, method, zorder=10):
        size = 30
        color = self.met_colors_dict[method]

        line = get_pol_coords(pol2cart(TS_vert), pol2cart(LM_vert))

        if (is_end(line)):
            two_edges = split_in_two(line)

            plot.ax_rect.plot(two_edges[0][0], two_edges[0][1], color=color, linestyle='-', zorder=1)
            plot.ax_rect.plot(two_edges[1][0], two_edges[1][1], color=color, linestyle='-', zorder=1)
        else:
            plot.ax_rect.plot(line[0], line[1], color=color, linestyle='-')

        plot.ax_rect.scatter(TS_vert[0], TS_vert[1], c='white',
                             edgecolor=color,
                             marker=self.met_ts_markers_dict[method],
                             s=size, zorder=zorder)
        plot.ax_rect.scatter(LM_vert[0], LM_vert[1], c=color,
                             edgecolor=color,
                             marker=self.met_lm_markers_dict[method],
                             s=size, zorder=zorder)

    def plot_connectivity(self, plot, method):
        pathways = self.Method_Pathways_dict[method].Pathways

        for j in range(len(pathways)):
            TS = pathways[j].TS
            LM1 = pathways[j].LM1
            LM2 = pathways[j].LM2

            TS_vert = [TS.phi, TS.theta]
            LM1_vert = [LM1.phi, LM1.theta]
            LM2_vert = [LM2.phi, LM2.theta]

            self.plot_line(plot, TS_vert, LM1_vert, method)
            self.plot_line(plot, TS_vert, LM2_vert, method)

    def save_connectivity(self, method='ALL'):
        filename = self.molecule + '-' + method + '-connectivity'
        dir = make_dir(os.path.join(self.MOL_SAVE_DIR, 'plots'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.reference_landscape.plot_voronoi_regions(plot=plot)
            self.reference_landscape.plot_skm_centers(plot=plot)
            self.reference_landscape.plot_cano(plot=plot)

            artist_list = []
            label_list = []

            if method == 'ALL':
                for method in self.Method_Pathways_dict:
                    self.plot_connectivity(plot=plot, method=method)

                    ts_artist, lm_artist = self.get_artist(method)

                    artist_list.append(ts_artist)
                    artist_list.append(lm_artist)

                    label_list.append('TS ' + method)
                    label_list.append('LM ' + method)

                    plot.ax_rect.legend(artist_list,
                                        label_list,
                                        scatterpoints=1, fontsize=8, frameon=True,
                                        framealpha=0.75,
                                        bbox_to_anchor=(1.2, 0.5), loc='right', borderaxespad=0,
                                        ncol=1).set_zorder(100)
            else:
                self.plot_connectivity(plot=plot, method=method)

                ts_artist, lm_artist = self.get_artist(method)

                artist_list.append(ts_artist)
                artist_list.append(lm_artist)

                label_list.append('TS ' + method)
                label_list.append('LM ' + method)

                plot.ax_rect.legend(artist_list,
                                     label_list,
                                     scatterpoints=1, fontsize=8, frameon=False,
                                     framealpha=0.75,
                                     bbox_to_anchor=(0.5, -0.15), loc=9, borderaxespad=0,
                                     ncol=len(self.met_colors_dict)).set_zorder(100)

            plot.ax_rect.set_xlim(-5, 365)
            plot.ax_rect.set_ylim(185, -5)

            plot.save(dir_=dir, filename=filename)


    def save_tessellation(self):
        filename = self.molecule + '-tessellation'
        dir = make_dir(os.path.join(self.MOL_SAVE_DIR, 'plots'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.reference_landscape.plot_voronoi_regions(plot=plot)
            self.reference_landscape.plot_regions_names(plot=plot)
            plot.ax_rect.set_ylim(185, -5)
            plot.ax_rect.set_xlim(-5, 365)

            plot.save(dir_=dir, filename=filename)
    #endregion

    # # # Writing # # #
    #region
    def format_skm_dict_for_csv(self, val):
        csv_dict = {}

        csv_dict[''] = []

        for name in list(self.reference_landscape.skm_name_list):
            csv_dict[''].append(name)

        for method in self.Method_Pathways_dict:
            csv_dict[method] = []

            for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
                skm_group = self.Method_Pathways_dict[method].skm_groupings[i]

                csv_dict[method].append(skm_group[val])

        return csv_dict

    def format_pathway_dict_for_csv(self):
        csv_dict = {}

        csv_dict['pathway'] = []

        for key in self.pathway_groupings:
            TS_skm_index = int(key.split('-')[1])
            LM1_skm_index = int(key.split('_')[0])
            LM2_skm_index = int(key.split('_')[1].split('-')[0])

            TS_name = self.reference_landscape.skm_name_list[TS_skm_index]
            LM1_name = self.reference_landscape.skm_name_list[LM1_skm_index]
            LM2_name = self.reference_landscape.skm_name_list[LM2_skm_index]

            name = LM1_name + '_' + LM2_name + '-' + TS_name

            csv_dict['pathway'].append(name)

        for method in self.Method_Pathways_dict:
            csv_dict[method] = []

            for key in self.Method_Pathways_dict[method].pathway_groupings:
                pathways = self.Method_Pathways_dict[method].pathway_groupings[key]['pathways']

                csv_dict[method].append(len(pathways))

        return csv_dict

    def format_norm_pathway_dict_for_csv(self):
        csv_dict = {}

        csv_dict['pathway'] = []

        for key in self.pathway_groupings:
            TS_skm_index = int(key.split('-')[1])
            LM1_skm_index = int(key.split('_')[0])
            LM2_skm_index = int(key.split('_')[1].split('-')[0])

            TS_name = self.reference_landscape.skm_name_list[TS_skm_index]
            LM1_name = self.reference_landscape.skm_name_list[LM1_skm_index]
            LM2_name = self.reference_landscape.skm_name_list[LM2_skm_index]

            name = LM1_name + '_' + LM2_name + '-' + TS_name

            csv_dict['pathway'].append(name)

        for method in self.Method_Pathways_dict:
            csv_dict[method] = []

            for key in self.Method_Pathways_dict[method].pathway_groupings:
                norm_pathways = self.Method_Pathways_dict[method].pathway_groupings[key]['norm_pathways']

                csv_dict[method].append(norm_pathways)

        return csv_dict

    def format_RMSD_dict_for_csv(self, comp_key):
        csv_dict = {}

        csv_dict['method'] = []

        for method in self.Method_Pathways_dict:
            csv_dict['method'].append(method)

        csv_dict[comp_key + '_RMSD'] = []

        for method in self.Method_Pathways_dict:
            csv_dict[comp_key + '_RMSD'].append(self.Method_Pathways_dict[method].comp_metrics[comp_key + '_RMSD'])

        return csv_dict


    def write_dict_to_csv(self, dict, name):
        csv_filename = self.molecule + '-' + name + '.csv'
        dir = make_dir(os.path.join(self.MOL_SAVE_DIR, 'tables'))

        if not os.path.exists(os.path.join(dir, csv_filename)):
            with open(os.path.join(dir, csv_filename), 'w', newline='') as file:
                w = csv.writer(file)
                w.writerow(dict.keys())
                w.writerows(zip(*dict.values()))

    def write_csvs(self):
        self.write_dict_to_csv(self.format_skm_dict_for_csv('gibbs_group_RMSD'), 'gibbs_group_RMSD')
        self.write_dict_to_csv(self.format_RMSD_dict_for_csv('gibbs'), 'gibbs_RMSD')

        self.write_dict_to_csv(self.format_skm_dict_for_csv('arc_group_WRMSD'), 'arc_group_WRMSD')
        self.write_dict_to_csv(self.format_RMSD_dict_for_csv('arc'), 'arc_RMSD')

        self.write_dict_to_csv(self.format_skm_dict_for_csv('weighted_gibbs'), 'weighted_gibbs')

        self.write_dict_to_csv(self.format_pathway_dict_for_csv(), 'pathways')
        self.write_dict_to_csv(self.format_norm_pathway_dict_for_csv(), 'norm_pathways')
    #endregion

    pass
