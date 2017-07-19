import os

import numpy as np
from spherecluster import SphericalKMeans
from scipy.spatial import SphericalVoronoi
import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import csv

import copy

from qm_utils.qm_common import create_out_fname

from qm_utils.spherical_kmeans_voronoi import read_csv_to_dict, read_csv_canonical_designations,\
                                                pol2cart, cart2pol, plot_line, arc_length_calculator,\
                                                split_in_two, is_end, get_pol_coords

##################################################### Directories ######################################################
#                                                                                                                      #
##################################################### Directories ######################################################
#region
QM_1_DIR = os.path.dirname(__file__)

# root of project
QM_0_DIR = os.path.dirname(QM_1_DIR)

PROG_DATA_DIR = os.path.join(QM_0_DIR, 'pucker_prog_data')

COMP_CLASSES_DIR = os.path.join(PROG_DATA_DIR, 'comparison_classes')
SV_DIR = os.path.join(PROG_DATA_DIR, 'spherical_kmeans_voronoi')
SV_MOL_DIR = os.path.join(SV_DIR, 'molecules')
#endregion

###################################################### Constants #######################################################
#                                                                                                                      #
###################################################### Constants #######################################################
#region
NUM_CLUSTERS = 38
REGION_THRESHOLD = 30
DEFAULT_TEMPERATURE = 298.15
K_B = 0.001985877534  # Boltzmann Constant in kcal/mol K
HART2KCAL = 627.509

REFERENCE = 'REFERENCE'
#endregion

################################################### Helper Functions ###################################################
#                                                                                                                      #
################################################### Helper Functions ###################################################
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

def is_IRC(structure):
    if structure.type == 'IRC':
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

    def save(self, filename, dir_, width=9, height=5):
        filename1 = create_out_fname(filename, base_dir=dir_, ext='.png')
        self.fig.set_size_inches(width, height)
        self.fig.savefig(filename1, facecolor=self.fig.get_facecolor(), transparent=True, dpi=300, bbox_inches='tight')

# # # Structures # # #
#region
class Structure():
    def __init__(self, phi, theta, gibbs=None, name=None, type=None):
        self.phi = phi
        self.theta = theta
        self.gibbs = gibbs
        self.name = name
        self.type = type

class Local_Minimum(Structure):
    def __init__(self, phi, theta, gibbs=None, name=None, type=None):
        Structure.__init__(self, phi, theta, gibbs, name, type)

class Transition_State(Structure):
    def __init__(self, phi, theta, gibbs, name, type=None):
        Structure.__init__(self, phi, theta, gibbs, name, type)
#endregion

# # # Pathways # # #
#region
class Pathway():
    def __init__(self, TS, LM1, LM2):
        self.TS = TS
        self.LM1 = LM1
        self.LM2 = LM2

class Method_Pathways():
    def __init__(self, LM_csv_filename, TS_csv_filename, IRC_csv_filename, method, molecule):
        self.method = method
        self.molecule = molecule

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

            self.LM_csv_list.append(Local_Minimum(phi, theta, gibbs, name, 'LM'))

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

            self.TS_csv_list.append(Transition_State(phi, theta, gibbs, name, 'TS'))

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

            self.IRC_csv_list.append(Local_Minimum(phi, theta, gibbs, name, 'IRC'))

        try:
            assert (len(self.IRC_csv_list) == len(self.TS_csv_list) * 2)
        except AssertionError:
            print('ERROR parsing ' + self.method + ' in ' + self.molecule)
            print('Unmatching number of IRC to TS (2 IRCs to 1 TS)')
            print('IRCs: ' + str(len(self.IRC_csv_list)))
            print('TSs: ' + str(len(self.TS_csv_list)))
            exit(1)

    def create_structure_list(self):
        self.structure_list = []

        for i in range(len(self.TS_csv_list)):
            self.structure_list.append(self.TS_csv_list[i])

        for i in range(len(self.LM_csv_list)):
            self.structure_list.append(self.LM_csv_list[i])

    def normalize_energies(self):
        self.min_gibbs = 100

        for i in range(len(self.structure_list)):
            if not is_IRC(self.structure_list[i]):
                curr_gibbs = self.structure_list[i].gibbs
            else:
                curr_gibbs = 100

            if curr_gibbs < self.min_gibbs:
                self.min_gibbs = curr_gibbs

        for i in range(len(self.structure_list)):
            self.structure_list[i].gibbs -= self.min_gibbs
            self.structure_list[i].gibbs *= HART2KCAL

        self.max_gibbs = self.min_gibbs

        for i in range(len(self.structure_list)):
            if not is_IRC(self.structure_list[i]):
                curr_gibbs = self.structure_list[i].gibbs
            else:
                curr_gibbs = 0

            if curr_gibbs > self.max_gibbs:
                self.max_gibbs = curr_gibbs

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

    # outdated
    def check_energies(self):
        for i in range(len(self.Pathways)):
            TS_gibbs = self.Pathways[i].TS.gibbs
            LM1_gibbs = self.Pathways[i].LM1.gibbs
            LM2_gibbs = self.Pathways[i].LM2.gibbs

            try:
                assert(TS_gibbs > LM1_gibbs and TS_gibbs > LM2_gibbs)
            except AssertionError:
                print('TS energy not greater than both LMs')
                print('TS.gibbs = ' + str(TS_gibbs))
                print('LM1.gibbs = ' + str(LM1_gibbs))
                print('LM2.gibbs = ' + str(LM2_gibbs))
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

            self.LM_csv_list.append(Local_Minimum(phi, theta, gibbs, name, 'LM'))

    def parse_TS_csv(self, TS_csv_filename):
        TS_csv_dict = read_csv_to_dict(TS_csv_filename, mode='r')
        self.TS_csv_list = []

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

            TS = Transition_State(phi, theta, gibbs, name, 'TS')
            LM1 = Local_Minimum(phi=phi_lm1,
                                theta=theta_lm1,
                                type='IRC')
            LM2 = Local_Minimum(phi=phi_lm2,
                                theta=theta_lm2,
                                type='IRC')

            self.TS_csv_list.append(TS)
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

            TS.gibbs -= self.min_gibbs
            TS.gibbs *= HART2KCAL

        self.max_gibbs = self.min_gibbs

        for i in range(len(self.structure_list)):
            curr_gibbs = self.structure_list[i].gibbs

            if curr_gibbs > self.max_gibbs:
                self.max_gibbs = curr_gibbs
#endregion


class Tessellation():
    def __init__(self, centers, number_clusters, n_init, max_iter, type):
        self.methods = {}
        self.type = type

        # Uses packages to calculate the k-means spherical centers
        self.skm = SphericalKMeans(n_clusters=number_clusters, init='k-means++', n_init=n_init, max_iter=max_iter)

        if len(centers) == number_clusters:
            self.skm.cluster_centers_ = np.asarray(centers)
        else:
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


class Reference_Landscape():
    # # # Init # # #
    #region
    def __init__(self, LM_csv_filename, TS_csv_filename, molecule, LM_clusters=None, TS_clusters=None):
        self.method = 'REFERENCE'
        self.molecule = molecule
        self.MOL_SAVE_DIR = make_dir(os.path.join(COMP_CLASSES_DIR, self.molecule))

        self.Reference_Pathways = Reference_Pathways(LM_csv_filename, TS_csv_filename)

        self.Pathways = self.Reference_Pathways.Pathways
        self.Local_Minima = self.Reference_Pathways.LM_csv_list

        self.canonical_designations = read_csv_canonical_designations('CP_params.csv', SV_DIR)
        self.reorg_canonical()

        if LM_clusters is not None:
            self.LM_input = True

            number_clusters = len(LM_clusters)
            n_init = 30
            max_iter = 300

            self.LM_Tessellation = Tessellation(LM_clusters, number_clusters, n_init, max_iter, 'LM')
        else:
            self.LM_input = False
            self.tessellate_LM()

        if TS_clusters is not None:
            self.TS_input = True
            number_clusters = len(TS_clusters)
            n_init = 30
            max_iter = 300

            self.TS_Tessellation = Tessellation(TS_clusters, number_clusters, n_init, max_iter, 'TS')
        else:
            self.TS_input = False
            self.tessellate_TS()

        self.LM_skm_name_list = []
        self.TS_skm_name_list = []

        self.assign_region_names(tessellation=self.LM_Tessellation,
                                 skm_name_list=self.LM_skm_name_list)
        self.assign_region_names(tessellation=self.TS_Tessellation,
                                 skm_name_list=self.TS_skm_name_list)
        self.assign_skm_labels()

        self.Reference_Pathways.normalize_energies()

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

    def get_unbinned_canos(self, structures=None):
        if structures == None:
            structures = self.Reference_Pathways.structure_list

        unbinned_canos = []

        for i in range(len(self.canonical_designations)):
            unbinned = True

            for j in range(len(structures)):
                structure = structures[j]

                if self.get_closest_cano(structure.phi, structure.theta) == self.canonical_designations[i].name:
                    unbinned = False

            if unbinned:
                unbinned_canos.append(pol2cart([self.canonical_designations[i].phi, self.canonical_designations[i].theta]))

        return unbinned_canos

    def tessellate_LM(self):
        number_clusters = 38
        n_init = 30
        max_iter = 300

        centers = []
        for i in range(len(self.canonical_designations)):
            phi = self.canonical_designations[i].phi
            theta = self.canonical_designations[i].theta
            centers.append(pol2cart([phi, theta]))

        # populate all centers to be used in voronoi tessellation
        for i in range(len(self.Reference_Pathways.LM_csv_list)):
            structure = self.Reference_Pathways.LM_csv_list[i]
            center = pol2cart([structure.phi, structure.theta])

            centers.append(center)

        self.LM_Tessellation = Tessellation(centers, number_clusters, n_init, max_iter, 'LM')

    def tessellate_TS(self):
        number_clusters = 38
        n_init = 30
        max_iter = 300

        centers = self.get_unbinned_canos(self.Reference_Pathways.TS_csv_list)

        centers = []
        for i in range(len(self.canonical_designations)):
            phi = self.canonical_designations[i].phi
            theta = self.canonical_designations[i].theta
            centers.append(pol2cart([phi, theta]))

        # populate all centers to be used in voronoi tessellation
        for i in range(len(self.Reference_Pathways.TS_csv_list)):
            structure = self.Reference_Pathways.TS_csv_list[i]
            center = pol2cart([structure.phi, structure.theta])

            centers.append(center)

        self.TS_Tessellation = Tessellation(centers, number_clusters, n_init, max_iter, 'TS')


    def check_for_dup_names(self, i, skm_name_list):
        name = skm_name_list[i]
        name_list = [i]

        for j in range(len(skm_name_list)):
            curr_name = skm_name_list[j]

            if curr_name == name and i != j:
                name_list.append(j)

        if len(name_list) > 1:
            for j in range(len(name_list)):
                skm_name_list[name_list[j]] = name + '_' + str(j)

    def assign_region_names(self, tessellation, skm_name_list):
        for i in range(len(tessellation.skm.cluster_centers_)):
            vert = cart2pol(tessellation.skm.cluster_centers_[i])

            phi = vert[0]
            theta = vert[1]

            name = self.get_closest_cano(phi, theta)

            skm_name_list.append(name)

        for i in range(len(skm_name_list)):
            self.check_for_dup_names(i, skm_name_list)

    def assign_skm_labels(self):
        for i in range(len(self.Pathways)):
            TS = self.Pathways[i].TS
            LM1 = self.Pathways[i].LM1
            LM2 = self.Pathways[i].LM2

            self.calc_closest_skm(TS, self.TS_Tessellation)
            self.calc_closest_skm(LM1, self.LM_Tessellation)
            self.calc_closest_skm(LM2, self.LM_Tessellation)

        for i in range(len(self.Local_Minima)):
            self.calc_closest_skm(self.Local_Minima[i], self.LM_Tessellation)

    def calc_closest_skm(self, structure, tessellation):
        min_dist = 100
        skm_index = 0

        phi1 = structure.phi
        theta1 = structure.theta

        for i in range(len(tessellation.skm.cluster_centers_)):
            center = cart2pol(tessellation.skm.cluster_centers_[i])

            phi2 = center[0]
            theta2 = center[1]

            curr_dist = arc_length_calculator(phi1=phi1, theta1=theta1,
                                              phi2=phi2, theta2=theta2)

            if curr_dist < min_dist:
                min_dist = curr_dist
                skm_index = i

        next_min_dist = 100
        next_skm_index = 0

        for i in range(len(tessellation.skm.cluster_centers_)):
            center = cart2pol(tessellation.skm.cluster_centers_[i])

            phi2 = center[0]
            theta2 = center[1]

            curr_dist = arc_length_calculator(phi1=phi1, theta1=theta1,
                                              phi2=phi2, theta2=theta2)

            if curr_dist < next_min_dist and curr_dist > min_dist:
                next_skm_index = i
                next_min_dist = curr_dist

        structure.closest_skm = skm_index
        structure.next_closest_skm = next_skm_index
    # endregion

    # # # Plotting # # #
    #region
    def plot_voronoi_regions(self, plot, tessellation):
        color = 'lightgray'

        for i in range(len(tessellation.sv.regions)):
            for j in range(len(tessellation.sv.regions[i])):
                if j == len(tessellation.sv.regions[i]) - 1:
                    index1 = tessellation.sv.regions[i][j]
                    index2 = tessellation.sv.regions[i][0]

                    vert1 = tessellation.sv.vertices[index1]
                    vert2 = tessellation.sv.vertices[index2]

                    plot_line(plot.ax_rect, [vert1, color, 0], [vert2, color, 0], color, zorder=0)
                else:
                    index1 = tessellation.sv.regions[i][j]
                    index2 = tessellation.sv.regions[i][j + 1]

                    vert1 = tessellation.sv.vertices[index1]
                    vert2 = tessellation.sv.vertices[index2]

                    plot_line(plot.ax_rect, [vert1, color, 0], [vert2, color, 0], color, zorder=0)

    def plot_regions_names(self, plot, tessellation):
        if tessellation.type == 'TS':
            self.skm_name_list = self.LM_skm_name_list
        else:
            self.skm_name_list = self.TS_skm_name_list

        for i in range(len(self.skm_name_list)):
            name = self.skm_name_list[i]

            vert = cart2pol(tessellation.skm.cluster_centers_[i])

            phi = vert[0]
            theta = vert[1] - 5

            if i == 0:
                theta = 20
                phi = 180

            plot.ax_rect.annotate(name, xy=(phi, theta), xytext=(phi, theta),  ha="center", va="center", fontsize=8, zorder=100)

    def plot_regions_coords(self, plot, tessellation):
        for i in range(len(tessellation.skm.cluster_centers_)):
            vert = cart2pol(tessellation.skm.cluster_centers_[i])

            phi = vert[0]
            theta = vert[1] - 5

            if i == 0:
                theta = 20
                phi = 180

            plot.ax_rect.annotate(str(round(vert[0],1)) + '\n' + str(round(vert[1],1)),
                                  xy=(phi, theta), xytext=(phi, theta),
                                  ha="center", va="center", fontsize=5, zorder=100)

    def plot_skm_centers(self, plot, tessellation):
        for i in range(len(tessellation.skm.cluster_centers_)):
            vert = cart2pol(tessellation.skm.cluster_centers_[i])

            phi = vert[0]
            theta = vert[1]

            if len(tessellation.methods[REFERENCE]['skm_groupings'][i]['structures']) > 0:
                plot.ax_rect.scatter(phi, theta, c='green', marker='x', s=60, zorder=10)
            else:
                plot.ax_rect.scatter(phi, theta, c='red', marker='x', s=60, zorder=10)

    def plot_cano(self, plot):
        phi_vals = []
        theta_vals = []

        for i in range(len(self.canonical_designations)):
            cano = self.canonical_designations[i]

            phi_vals.append(cano.phi)
            theta_vals.append(cano.theta)

        plot.ax_rect.scatter(phi_vals, theta_vals, c='black', marker='+', s=60, zorder=10)


    def save_tessellation(self, tessellation):
        filename = self.molecule + '-tessellation-' + tessellation.type + '-coords'
        dir = make_dir(os.path.join(self.MOL_SAVE_DIR, 'plots'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.plot_voronoi_regions(plot=plot, tessellation=tessellation)
            self.plot_regions_coords(plot=plot, tessellation=tessellation)

            plot.ax_rect.set_ylim(185, -5)
            plot.ax_rect.set_xlim(-5, 365)

            plot.save(dir_=dir, filename=filename)

    def write_tessellation_to_csv(self, tessellation):
        dict = {}
        dict['phi'] = []
        dict['theta'] = []

        for i in range(len(tessellation.skm.cluster_centers_)):
            vert = cart2pol(tessellation.skm.cluster_centers_[i])

            phi = vert[0]
            theta = vert[1]

            dict['phi'].append(round(phi,1))
            dict['theta'].append(round(theta,1))

        name = tessellation.type

        csv_filename = self.molecule + '-' + name + '_clusters.csv'
        dir = make_dir(os.path.join(SV_MOL_DIR, self.molecule))

        if not os.path.exists(os.path.join(dir, csv_filename)):
            with open(os.path.join(dir, csv_filename), 'w', newline='') as file:
                w = csv.writer(file)
                w.writerow(dict.keys())
                w.writerows(zip(*dict.values()))


    #endregion

    pass

class Compare_Methods():
    # # # Init # # #
    #region
    def __init__(self,
                 molecule,
                 met_colors_dict,
                 met_ts_markers_dict,
                 met_lm_markers_dict):

        self.molecule = molecule
        self.dir_init()

        self.reference_landscape_init()

        self.Method_Pathways_dict = {}
        self.Method_Pathways_init()

        self.assign_structure_names()

        self.met_colors_dict = met_colors_dict
        self.met_ts_markers_dict = met_ts_markers_dict
        self.met_lm_markers_dict = met_lm_markers_dict

        self.assign_skm_labels()

        self.pathway_groupings_init()

        # REFERENCE data initialization
        self.populate_skm_groupings(self.reference_landscape.LM_Tessellation,
                                    REFERENCE,
                                    self.reference_landscape.Reference_Pathways.LM_csv_list)
        self.populate_skm_groupings(self.reference_landscape.TS_Tessellation,
                                    REFERENCE,
                                    self.reference_landscape.Reference_Pathways.TS_csv_list)

        self.populate_pathway_groupings(REFERENCE)
        self.populate_local_minima(REFERENCE)
        self.do_init_calcs(REFERENCE)

        if not self.reference_landscape.LM_input:
            self.retessellate(self.reference_landscape.LM_Tessellation)
        if not self.reference_landscape.TS_input:
            self.retessellate(self.reference_landscape.TS_Tessellation)

        self.recalc_skms(self.reference_landscape.LM_Tessellation, REFERENCE)
        self.recalc_skms(self.reference_landscape.TS_Tessellation, REFERENCE)

        self.assign_skm_labels()

        self.populate_skm_again(self.reference_landscape.LM_Tessellation,
                                REFERENCE,
                                self.reference_landscape.Reference_Pathways.LM_csv_list)
        self.populate_skm_again(self.reference_landscape.TS_Tessellation,
                                REFERENCE,
                                self.reference_landscape.Reference_Pathways.TS_csv_list)

        self.do_init_calcs(REFERENCE)

        for method in self.Method_Pathways_dict:
            self.populate_skm_groupings(self.reference_landscape.LM_Tessellation,
                                        method,
                                        self.Method_Pathways_dict[method].LM_csv_list)
            self.populate_skm_groupings(self.reference_landscape.TS_Tessellation,
                                        method,
                                        self.Method_Pathways_dict[method].TS_csv_list)

            self.populate_pathway_groupings(method)
            self.populate_local_minima(method)
            self.do_init_calcs(method)

            self.assign_skm_labels()

            self.populate_skm_again(self.reference_landscape.LM_Tessellation,
                                    method,
                                    self.Method_Pathways_dict[method].LM_csv_list)
            self.populate_skm_again(self.reference_landscape.TS_Tessellation,
                                    method,
                                    self.Method_Pathways_dict[method].TS_csv_list)

            self.do_init_calcs(method)

            self.populate_pathway_groupings(method)
            self.populate_local_minima(method)
            #self.do_calcs(method)

        pass

    # # # init # # #
    #region
    def retessellate(self, tessellation):
        centers = []
        structures = []

        for i in range(len(tessellation.skm.cluster_centers_)):
            region_structures = []

            if tessellation.type == 'LM':
                for j in range(len(self.reference_landscape.Reference_Pathways.structure_list)):
                    structure = self.reference_landscape.Reference_Pathways.structure_list[j]

                    if structure.closest_skm == i and structure.type == 'LM':
                        region_structures.append(structure)
            elif tessellation.type == 'TS':
                for j in range(len(self.reference_landscape.Reference_Pathways.structure_list)):
                    structure = self.reference_landscape.Reference_Pathways.structure_list[j]

                    if structure.closest_skm == i and structure.type == 'TS':
                        region_structures.append(structure)

            if len(region_structures) > 0:
                avg_phi = 0
                avg_theta = 0

                for j in range(len(region_structures)):
                    phi = region_structures[j].phi
                    theta = region_structures[j].theta

                    if tessellation.type == 'LM':
                        weighting = region_structures[j].LM_weighting
                    elif tessellation.type == 'TS':
                        weighting = region_structures[j].TS_weighting

                    avg_phi += phi * weighting
                    avg_theta += theta * weighting

                centers.append(pol2cart([avg_phi, avg_theta]))
                structures.append(Structure(phi=avg_phi, theta=avg_theta))

        unbinned_cano_centers = self.reference_landscape.get_unbinned_canos(structures)

        if tessellation.type == 'LM':
            unbinned_cano_centers = self.reference_landscape.get_unbinned_canos(self.reference_landscape.Reference_Pathways.LM_csv_list)
        elif tessellation.type == 'TS':
            unbinned_cano_centers = self.reference_landscape.get_unbinned_canos(self.reference_landscape.Reference_Pathways.TS_csv_list)

        final_centers = centers + unbinned_cano_centers

        number_clusters = 38

        if len(final_centers) < number_clusters:
            number_clusters = len(final_centers)

        if tessellation.type == 'LM':
            self.reference_landscape.LM_Tessellation = Tessellation(final_centers, number_clusters, 300, 300, 'LM')
        elif tessellation.type == 'TS':
            self.reference_landscape.TS_Tessellation = Tessellation(final_centers, number_clusters, 300, 300, 'TS')

    def is_end(self, structures):
        has_0 = False
        has_360 = False

        for i in range(len(structures)):
            if structures[i].phi < 60:
                has_0 = True
            if structures[i].phi > 300:
                has_360 = True

        return has_0 and has_360

    def recalc_skms(self, tessellation, method):
        for i in range(len(tessellation.skm.cluster_centers_)):
            if tessellation.type == 'LM':
                skm_grouping = self.reference_landscape.LM_Tessellation.methods[method]['skm_groupings'][i]
            elif tessellation.type == 'TS':
                skm_grouping = self.reference_landscape.TS_Tessellation.methods[method]['skm_groupings'][i]

            if len(skm_grouping['structures']) > 0:
                avg_phi = 0
                avg_theta = 0

                is_end = self.is_end(skm_grouping['structures'])

                for j in range(len(skm_grouping['structures'])):
                    structure = skm_grouping['structures'][j]
                    phi = structure.phi
                    theta = structure.theta

                    if is_end and phi < 180:
                        phi += 360

                    if tessellation.type == 'LM':
                        weighting = structure.LM_weighting
                    elif tessellation.type == 'TS':
                        weighting = structure.TS_weighting

                    avg_phi += phi * weighting
                    avg_theta += theta * weighting

                if avg_phi > 360:
                    avg_phi -= 360

                tessellation.skm.cluster_centers_[i] = pol2cart([avg_phi, avg_theta])

        # Default parameters for spherical voronoi
        radius = 1
        center = np.array([0, 0, 0])

        # Spherical Voronoi for the centers
        tessellation.sv = SphericalVoronoi(tessellation.skm.cluster_centers_, radius, center)
        tessellation.sv.sort_vertices_of_regions()


    def get_dist_between_structures(self, struct1, struct2):
        return arc_length_calculator(struct1.phi, struct1.theta, struct2.phi, struct2.theta)

    def energy_is_close(self, tessellation, structure, i):
        max_gibbs = 0
        min_gibbs = 100

        for j in range(len(tessellation.skm_groupings[i]['structures'])):
            gibbs = tessellation.skm_groupings[i]['structures'][j].gibbs

            if gibbs < min_gibbs:
                min_gibbs = gibbs

            if gibbs > max_gibbs:
                max_gibbs = gibbs

        max_gibbs_diff = abs(max_gibbs - min_gibbs)

        if tessellation.skm_groupings[i][tessellation.type + '_weighted_gibbs'] == None:
            return False

        lower_bound = tessellation.skm_groupings[i][tessellation.type + '_weighted_gibbs'] - max_gibbs_diff
        upper_bound = tessellation.skm_groupings[i][tessellation.type + '_weighted_gibbs'] + max_gibbs_diff

        if lower_bound < structure.gibbs and upper_bound > structure.gibbs:
            return True
        else:
            return False

    def weighting_is_significant(self, tessellation, structure, i, method):
        wt_tol = 0.1

        structures = tessellation.methods[method]['skm_groupings'][i]['structures']
        weighting = self.calc_weighting_added(structures, structure)

        if weighting > wt_tol and weighting < 1 - wt_tol / 2:
            return True
        else:
            return False

    def calc_weighting_added(self, structures, structure):
        total_boltz = 0

        for j in range(len(structures)):
            if len(structures[j].type.split('_')) == 1:
                e_val = structures[j].gibbs

                component = math.exp(-e_val / (K_B * DEFAULT_TEMPERATURE))
                total_boltz += component

        e_val = structure.gibbs

        component = math.exp(-e_val / (K_B * DEFAULT_TEMPERATURE))
        ind_boltz = component
        total_boltz += component

        weighting = ind_boltz / total_boltz

        return weighting

    def get_max_raw_dist(self, tessellation, method, i):
        structures = tessellation.methods[method]['skm_groupings'][i]['structures']
        skm_vert = cart2pol(tessellation.skm.cluster_centers_[i])

        max_dist = 0

        for j in range(len(structures)):
            curr_dist = arc_length_calculator(structures[j].phi, structures[j].theta, skm_vert[0], skm_vert[1])

            if curr_dist > max_dist:
                max_dist = curr_dist

        return max_dist

    # creates a list of structures grouped by skm
    def populate_skm_again(self, tessellation, method, structure_list):
        gibbs_diff_tol = 1

        for i in range(len(tessellation.skm.cluster_centers_)):
            if tessellation.methods[REFERENCE]['skm_groupings'][i][tessellation.type + '_weighted_gibbs'] is not None:
                structures = tessellation.methods[method]['skm_groupings'][i]['structures']

                for j in range(len(structure_list)):
                    structure = structure_list[j]

                    if structure.next_closest_skm == i:
                        skm_1 = cart2pol(tessellation.skm.cluster_centers_[i])
                        skm_2 = cart2pol(tessellation.skm.cluster_centers_[structure.closest_skm])

                        skm_dist = arc_length_calculator(skm_1[0], skm_1[1], skm_2[0], skm_2[1])

                        arc = structure.comp_metrics['arc']
                        next_arc = structure.comp_metrics['next_arc']

                        dist_tol = 1.2

                        if arc == 0:
                            dist_ratio = dist_tol
                        else:
                            dist_ratio = next_arc / arc

                        if len(structures) == 0:
                            max_raw_dist = skm_dist * 0.75 / 3
                        else:
                            max_raw_dist = self.get_max_raw_dist(tessellation, method, i)

                        if structure.comp_metrics['next_arc'] < max_raw_dist * 3:
                            if structure.comp_metrics['next_arc'] < skm_dist * 0.75 or dist_ratio < dist_tol:
                                if structure.gibbs is not None and tessellation.methods[method]['skm_groupings'][i][tessellation.type + '_weighted_gibbs'] is None\
                                    or abs(structure.gibbs - tessellation.methods[method]['skm_groupings'][i][tessellation.type + '_weighted_gibbs']) < gibbs_diff_tol\
                                    or self.weighting_is_significant(tessellation, structure, i, method):

                                    structure_copy = Structure(structure.phi,
                                                               structure.theta,
                                                               structure.gibbs,
                                                               structure.name,
                                                               structure.type + '_added')

                                    structure_copy.closest_skm = structure.next_closest_skm
                                    structure_copy.next_closest_skm = structure.closest_skm
                                    structure_copy.comp_metrics = {}
                                    structure_copy.comp_metrics['arc'] = structure.comp_metrics['next_arc']
                                    structure_copy.comp_metrics['next_arc'] = structure.comp_metrics['arc']

                                    structures.append(structure_copy)


    def normalize_pathways(self, method):
        for key in self.Method_Pathways_dict[method].pathway_groupings:
            met_pathways = len(self.Method_Pathways_dict[method].pathway_groupings[key]['pathways'])

            if met_pathways > 0:
                self.Method_Pathways_dict[method].pathway_groupings[key]['norm_pathways'] = 1
            else:
                self.Method_Pathways_dict[method].pathway_groupings[key]['norm_pathways'] = 0

    def normalize_comp_metrics(self, method, comp_key):
        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            ref_metric = self.Method_Pathways_dict[REFERENCE].skm_groupings[i][comp_key + '_group_WRMSD']
            met_metric = self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_WRMSD']

            if ref_metric != None and met_metric != None:
                if ref_metric == 0:
                    self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_norm_group_WRMSD'] = met_metric
                else:
                    self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_norm_group_WRMSD'] = met_metric / ref_metric
            else:
                self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_norm_group_WRMSD'] = None


    def dir_init(self):
        self.MOL_DATA_DIR = make_dir(os.path.join(SV_MOL_DIR, self.molecule))
        self.MOL_SAVE_DIR = make_dir(os.path.join(COMP_CLASSES_DIR, self.molecule))

        self.IRC_DATA_DIR = make_dir(os.path.join(self.MOL_DATA_DIR, 'IRC'))
        self.LM_DATA_DIR = make_dir(os.path.join(self.MOL_DATA_DIR, 'LM'))
        self.TS_DATA_DIR = make_dir(os.path.join(self.MOL_DATA_DIR, 'TS'))

        self.IRC_DATA_dir_list = os.listdir(os.path.join(self.MOL_DATA_DIR, 'IRC'))

    def parse_clusters_csv(self, clusters_filename):
        clusters = []

        clusters_dict = read_csv_to_dict(clusters_filename, mode='r')

        for i in range(len(clusters_dict)):
            info = clusters_dict[i]
            phi = float(info['phi'])
            theta = float(info['theta'])

            clusters.append(pol2cart([phi, theta]))

        return clusters

    def reference_landscape_init(self):
        ref_LM_csv_filename = os.path.join(self.MOL_DATA_DIR, 'z_' + self.molecule + '_LM-b3lyp_howsugarspucker.csv')
        ref_TS_csv_filename = os.path.join(self.MOL_DATA_DIR, 'z_' + self.molecule + '_TS-b3lyp_howsugarspucker.csv')

        LM_clusters_filename = os.path.join(self.MOL_DATA_DIR, self.molecule + '-LM_clusters.csv')
        TS_clusters_filename = os.path.join(self.MOL_DATA_DIR, self.molecule + '-TS_clusters.csv')

        if os.path.exists(LM_clusters_filename) and os.path.exists(TS_clusters_filename):
            LM_clusters = self.parse_clusters_csv(LM_clusters_filename)
            TS_clusters = self.parse_clusters_csv(TS_clusters_filename)
        else:
            LM_clusters = []
            TS_clusters = []

        if len(LM_clusters) == 0:
            LM_clusters = None

        if len(TS_clusters) == 0:
            TS_clusters = None

        self.reference_landscape = Reference_Landscape(LM_csv_filename=ref_LM_csv_filename,
                                                       TS_csv_filename=ref_TS_csv_filename,
                                                       molecule=self.molecule,
                                                       LM_clusters=LM_clusters,
                                                       TS_clusters=TS_clusters)

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

            IRC_csv_filename = os.path.join(self.IRC_DATA_DIR, 'z_dataset-' + self.molecule + '-IRC-' + method + '.csv')
            LM_csv_filename = os.path.join(self.LM_DATA_DIR, 'z_dataset-' + self.molecule + '-LM-' + method + '.csv')
            TS_csv_filename = os.path.join(self.TS_DATA_DIR, 'z_dataset-' + self.molecule + '-TS-' + method + '.csv')

            self.Method_Pathways_dict[method] = (Method_Pathways(LM_csv_filename=LM_csv_filename,
                                                                 TS_csv_filename=TS_csv_filename,
                                                                 IRC_csv_filename=IRC_csv_filename,
                                                                 method=method.upper(),
                                                                 molecule=self.molecule))

            self.Method_Pathways_dict[method].comp_metrics = {}

        self.Method_Pathways_dict['REFERENCE'] = self.reference_landscape.Reference_Pathways
        self.Method_Pathways_dict['REFERENCE'].comp_metrics = {}

    # creates a list of structures grouped by skm
    def populate_skm_groupings(self, tessellation, method, structure_list):
        tessellation.methods[method] = {}
        tessellation.methods[method]['skm_groupings'] = []

        for i in range(len(tessellation.skm.cluster_centers_)):
            tessellation.methods[method]['skm_groupings'].append({})
            tessellation.methods[method]['skm_groupings'][i]['structures'] = []
            structures = tessellation.methods[method]['skm_groupings'][i]['structures']

            for j in range(len(structure_list)):
                structure = structure_list[j]

                if structure.closest_skm == i:
                    structures.append(structure)

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

                self.calc_closest_TS_skm(TS)
                self.calc_closest_LM_skm(LM1)
                self.calc_closest_LM_skm(LM2)

                if LM1.closest_skm == LM2.closest_skm:
                    pathways.remove(item)

                    print('A pathway was not included since its LM1 and LM2 are the same structure.')
                    print('class: Compare_Methods')
                    print('function: assign_skm_labels')
                    print('method: ' + method)

            for i in range(len(self.Method_Pathways_dict[method].LM_csv_list)):
                LM = self.Method_Pathways_dict[method].LM_csv_list[i]
                LM.comp_metrics = {}
                self.calc_closest_LM_skm(LM)

    def calc_closest_LM_skm(self, structure):
        min_dist = 100
        skm_index = 0

        phi1 = structure.phi
        theta1 = structure.theta

        for i in range(len(self.reference_landscape.LM_Tessellation.skm.cluster_centers_)):
            center = cart2pol(self.reference_landscape.LM_Tessellation.skm.cluster_centers_[i])

            phi2 = center[0]
            theta2 = center[1]

            curr_dist = arc_length_calculator(phi1=phi1, theta1=theta1,
                                              phi2=phi2, theta2=theta2)

            if curr_dist < min_dist:
                min_dist = curr_dist
                skm_index = i

        next_min_dist = 100
        next_skm_index = 0

        for i in range(len(self.reference_landscape.LM_Tessellation.skm.cluster_centers_)):
            center = cart2pol(self.reference_landscape.LM_Tessellation.skm.cluster_centers_[i])

            phi2 = center[0]
            theta2 = center[1]

            curr_dist = arc_length_calculator(phi1=phi1, theta1=theta1,
                                              phi2=phi2, theta2=theta2)

            if curr_dist < next_min_dist and curr_dist > min_dist:
                next_skm_index = i
                next_min_dist = curr_dist

        structure.comp_metrics['arc'] = min_dist
        structure.closest_skm = skm_index

        structure.comp_metrics['next_arc'] = next_min_dist
        structure.next_closest_skm = next_skm_index

    def calc_closest_TS_skm(self, structure):
        min_dist = 100
        skm_index = 0

        phi1 = structure.phi
        theta1 = structure.theta

        for i in range(len(self.reference_landscape.TS_Tessellation.skm.cluster_centers_)):
            center = cart2pol(self.reference_landscape.TS_Tessellation.skm.cluster_centers_[i])

            phi2 = center[0]
            theta2 = center[1]

            curr_dist = arc_length_calculator(phi1=phi1, theta1=theta1,
                                              phi2=phi2, theta2=theta2)

            if curr_dist < min_dist:
                min_dist = curr_dist
                skm_index = i

        next_min_dist = 100
        next_skm_index = 0

        for i in range(len(self.reference_landscape.TS_Tessellation.skm.cluster_centers_)):
            center = cart2pol(self.reference_landscape.TS_Tessellation.skm.cluster_centers_[i])

            phi2 = center[0]
            theta2 = center[1]

            curr_dist = arc_length_calculator(phi1=phi1, theta1=theta1,
                                              phi2=phi2, theta2=theta2)

            if curr_dist < next_min_dist and curr_dist > min_dist:
                next_skm_index = i
                next_min_dist = curr_dist

        structure.comp_metrics['arc'] = min_dist
        structure.closest_skm = skm_index

        structure.comp_metrics['next_arc'] = next_min_dist
        structure.next_closest_skm = next_skm_index

    def calc_gibbs_diff(self, method):
        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
                structure = self.Method_Pathways_dict[method].skm_groupings[i]['structures'][j]
                ref_structure_gibbs = self.Method_Pathways_dict[REFERENCE].skm_groupings[i]['weighted_gibbs']

                if ref_structure_gibbs == None:
                    structure.comp_metrics['gibbs'] = None
                else:
                    structure.comp_metrics['gibbs'] = structure.gibbs - ref_structure_gibbs
    #endregion

    # # # do_calcs # # #
    # region
    def do_init_calcs(self, method):
        for i in range(len(self.reference_landscape.LM_Tessellation.methods[method]['skm_groupings'])):
            self.calc_LM_weighting(i, method)

        for i in range(len(self.reference_landscape.TS_Tessellation.methods[method]['skm_groupings'])):
            self.calc_TS_weighting(i, method)

    def calc_LM_weighting(self, i, method):
        tessellation = self.reference_landscape.LM_Tessellation

        total_boltz = 0

        for j in range(len(tessellation.methods[method]['skm_groupings'][i]['structures'])):
            structure = tessellation.methods[method]['skm_groupings'][i]['structures'][j]
            e_val = structure.gibbs

            component = math.exp(-e_val / (K_B * DEFAULT_TEMPERATURE))
            structure.LM_ind_bolts = component
            total_boltz += component

        wt_gibbs = 0

        for j in range(len(tessellation.methods[method]['skm_groupings'][i]['structures'])):
            structure = tessellation.methods[method]['skm_groupings'][i]['structures'][j]

            if structure.LM_ind_bolts == 0:
                structure.LM_weighting = 0
                wt_gibbs += 0
            else:
                structure.LM_weighting = structure.LM_ind_bolts / total_boltz
                wt_gibbs += structure.gibbs * structure.LM_weighting

        if len(tessellation.methods[method]['skm_groupings'][i]['structures']) == 0:
            tessellation.methods[method]['skm_groupings'][i]['LM_weighted_gibbs'] = None
        else:
            tessellation.methods[method]['skm_groupings'][i]['LM_weighted_gibbs'] = wt_gibbs

    def calc_TS_weighting(self, i, method):
        tessellation = self.reference_landscape.TS_Tessellation

        total_boltz = 0

        for j in range(len(tessellation.methods[method]['skm_groupings'][i]['structures'])):
            structure = tessellation.methods[method]['skm_groupings'][i]['structures'][j]
            e_val = structure.gibbs

            component = math.exp(-e_val / (K_B * DEFAULT_TEMPERATURE))
            structure.TS_ind_bolts = component
            total_boltz += component

        wt_gibbs = 0

        for j in range(len(tessellation.methods[method]['skm_groupings'][i]['structures'])):
            structure = tessellation.methods[method]['skm_groupings'][i]['structures'][j]

            if structure.TS_ind_bolts == 0:
                structure.TS_weighting = 0
                wt_gibbs += 0
            else:
                structure.TS_weighting = structure.TS_ind_bolts / total_boltz
                wt_gibbs += structure.gibbs * structure.TS_weighting

        if len(tessellation.methods[method]['skm_groupings'][i]['structures']) == 0:
            tessellation.methods[method]['skm_groupings'][i]['TS_weighted_gibbs'] = None
        else:
            tessellation.methods[method]['skm_groupings'][i]['TS_weighted_gibbs'] = wt_gibbs


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

        if len(self.Method_Pathways_dict[method].skm_groupings[i]['structures']) == 0:
            self.Method_Pathways_dict[method].skm_groupings[i]['weighted_gibbs'] = None
        else:
            self.Method_Pathways_dict[method].skm_groupings[i]['weighted_gibbs'] = wt_gibbs

    def calc_WSS(self, method, i, comp_key):
        WSS = 0

        for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
            structure = self.Method_Pathways_dict[method].skm_groupings[i]['structures'][j]

            comp_val = structure.comp_metrics[comp_key]
            if comp_val is not None:
                WSS += comp_val ** 2

        if len(self.Method_Pathways_dict[method].skm_groupings[i]['structures']) == 0:
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WSS'] = None
        else:
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WSS'] = round(WSS, 5)

    def calc_group_RMSD(self, method, i, comp_key):
        size = len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])

        if (size == 0):
            RMSD = None
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_RMSD'] = RMSD
        else:
            RMSD = (self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WSS'] / size) ** 0.5
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_RMSD'] = round(RMSD, 5)

    def calc_SSE(self, method, comp_key):
        SSE = 0

        unphys = True

        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            if self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WSS'] is not None:
                unphys = False
                SSE += self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WSS']

        if unphys:
            self.Method_Pathways_dict[method].comp_metrics[comp_key + '_SSE'] = None
        else:
            self.Method_Pathways_dict[method].comp_metrics[comp_key + '_SSE'] = round(SSE, 5)

    def calc_RMSD(self, method, comp_key):
        size = 0

        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
                size += 1

        if self.Method_Pathways_dict[method].comp_metrics[comp_key + '_SSE'] is None:
            self.Method_Pathways_dict[method].comp_metrics[comp_key + '_RMSD'] = None

        RMSD = (self.Method_Pathways_dict[method].comp_metrics[comp_key + '_SSE'] / size) ** 0.5
        self.Method_Pathways_dict[method].comp_metrics[comp_key + '_RMSD'] = round(RMSD, 5)

    def calc_WWSS(self, method, i, comp_key):
        WWSS = 0

        # calculating each point's contribution
        for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
            structure = self.Method_Pathways_dict[method].skm_groupings[i]['structures'][j]

            comp_val = structure.comp_metrics[comp_key]
            weighting = structure.weighting
            if comp_val is not None:
                WWSS += (comp_val ** 2) * weighting

        if len(self.Method_Pathways_dict[method].skm_groupings[i]['structures']) == 0:
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WWSS'] = None
        else:
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WWSS'] = round(WWSS, 5)

    def calc_group_WRMSD(self, method, i, comp_key):
        size = len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])

        if (size == 0):
            WRMSD = None
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_WRMSD'] = WRMSD
        else:
            WRMSD = (self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WWSS'] / size) ** 0.5
            self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_group_WRMSD'] = round(WRMSD, 5)

    def calc_WSSE(self, method, comp_key):
        WSSE = 0

        unphys = True

        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            if self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WWSS'] is not None:
                unphys = False
                WSSE += self.Method_Pathways_dict[method].skm_groupings[i][comp_key + '_WWSS']

        if unphys:
            self.Method_Pathways_dict[method].comp_metrics[comp_key + '_WSSE'] = None
        else:
            self.Method_Pathways_dict[method].comp_metrics[comp_key + '_WSSE'] = round(WSSE, 5)

    def calc_WRMSD(self, method, comp_key):
        size = 0

        for i in range(len(self.Method_Pathways_dict[method].skm_groupings)):
            for j in range(len(self.Method_Pathways_dict[method].skm_groupings[i]['structures'])):
                size += 1

        if self.Method_Pathways_dict[method].comp_metrics[comp_key + '_WSSE'] is None:
            self.Method_Pathways_dict[method].comp_metrics[comp_key + '_WRMSD'] = None

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


    def plot_raw_data_LM(self, plot, method):
        size = 30

        lm_phi_vals = []
        lm_theta_vals = []

        irc_phi_vals = []
        irc_theta_vals = []

        for j in range(len(self.Method_Pathways_dict[method].LM_csv_list)):
            LM = self.Method_Pathways_dict[method].LM_csv_list[j]

            LM_vert = [LM.phi, LM.theta]
            lm_phi_vals.append(LM_vert[0])
            lm_theta_vals.append(LM_vert[1])

        pathways = self.Method_Pathways_dict[method].Pathways

        for j in range(len(pathways)):
            LM1 = pathways[j].LM1

            LM1_vert = [LM1.phi, LM1.theta]
            irc_phi_vals.append(LM1_vert[0])
            irc_theta_vals.append(LM1_vert[1])

            LM2 = pathways[j].LM2

            LM2_vert = [LM2.phi, LM2.theta]
            irc_phi_vals.append(LM2_vert[0])
            irc_theta_vals.append(LM2_vert[1])

        plot.ax_rect.scatter(irc_phi_vals, irc_theta_vals, c=self.met_colors_dict[method],
                             edgecolor='black',
                             marker=self.met_lm_markers_dict[method],
                             s=size / 2, zorder=10)

        plot.ax_rect.scatter(lm_phi_vals, lm_theta_vals, c=self.met_colors_dict[method],
                             edgecolor=self.met_colors_dict[method],
                             marker=self.met_lm_markers_dict[method],
                             s=size, zorder=10)

    def save_raw_data_LM(self, method='ALL'):
        filename = self.molecule + '-' + method + '-raw_data_LM'

        dir1 = make_dir(os.path.join(self.MOL_SAVE_DIR, 'plots'))
        dir = make_dir(os.path.join(dir1, 'raw_data'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.reference_landscape.plot_voronoi_regions(plot=plot,
                                                          tessellation=self.reference_landscape.LM_Tessellation)
            self.reference_landscape.plot_skm_centers(plot=plot,
                                                      tessellation=self.reference_landscape.LM_Tessellation)
            self.reference_landscape.plot_cano(plot=plot)

            artist_list = []
            label_list = []

            if method == 'ALL':
                for method in self.Method_Pathways_dict:
                    self.plot_raw_data_LM(plot=plot, method=method)

                    ts_artist, lm_artist = self.get_artist(method)

                    artist_list.append(lm_artist)
                    label_list.append('TS ' + method)

                    plot.ax_rect.legend(artist_list,
                                        label_list,
                                        scatterpoints=1, fontsize=8, frameon=True,
                                        framealpha=0.75,
                                        bbox_to_anchor=(1.2, 0.5), loc='right', borderaxespad=0,
                                        ncol=1).set_zorder(100)
            else:
                self.plot_raw_data_LM(plot=plot, method=REFERENCE)
                self.plot_raw_data_LM(plot=plot, method=method)

                ts_artist, lm_artist = self.get_artist(method)
                ref_ts_artist, ref_lm_artist = self.get_artist(REFERENCE)

                artist_list.append(lm_artist)
                artist_list.append(ref_lm_artist)
                label_list.append('TS ' + method)
                label_list.append('TS ' + REFERENCE)

                plot.ax_rect.legend(artist_list,
                                     label_list,
                                     scatterpoints=1, fontsize=8, frameon=False,
                                     framealpha=0.75,
                                     bbox_to_anchor=(0.5, -0.15), loc=9, borderaxespad=0,
                                     ncol=len(self.met_colors_dict)).set_zorder(100)

            plot.ax_rect.set_ylim(185, -5)
            plot.ax_rect.set_xlim(-5, 365)

            plot.save(dir_=dir, filename=filename)

    def plot_raw_data_TS(self, plot, method):
        size = 30

        ts_phi_vals = []
        ts_theta_vals = []

        pathways = self.Method_Pathways_dict[method].Pathways

        for j in range(len(pathways)):
            TS = pathways[j].TS

            TS_vert = [TS.phi, TS.theta]
            ts_phi_vals.append(TS_vert[0])
            ts_theta_vals.append(TS_vert[1])

        plot.ax_rect.scatter(ts_phi_vals, ts_theta_vals, c='white',
                             edgecolor=self.met_colors_dict[method],
                             marker=self.met_ts_markers_dict[method],
                             s=size, zorder=10)

    def save_raw_data_TS(self, method='ALL'):
        filename = self.molecule + '-' + method + '-raw_data_TS'

        dir1 = make_dir(os.path.join(self.MOL_SAVE_DIR, 'plots'))
        dir = make_dir(os.path.join(dir1, 'raw_data'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.reference_landscape.plot_voronoi_regions(plot=plot,
                                                          tessellation=self.reference_landscape.TS_Tessellation)
            self.reference_landscape.plot_skm_centers(plot=plot,
                                                      tessellation=self.reference_landscape.TS_Tessellation)
            self.reference_landscape.plot_cano(plot=plot)

            artist_list = []
            label_list = []

            if method == 'ALL':
                for method in self.Method_Pathways_dict:
                    self.plot_raw_data_TS(plot=plot, method=method)

                    ts_artist, lm_artist = self.get_artist(method)

                    artist_list.append(ts_artist)
                    label_list.append('TS ' + method)

                    plot.ax_rect.legend(artist_list,
                                        label_list,
                                        scatterpoints=1, fontsize=8, frameon=True,
                                        framealpha=0.75,
                                        bbox_to_anchor=(1.2, 0.5), loc='right', borderaxespad=0,
                                        ncol=1).set_zorder(100)
            else:
                self.plot_raw_data_TS(plot=plot, method=REFERENCE)
                self.plot_raw_data_TS(plot=plot, method=method)

                ts_artist, lm_artist = self.get_artist(method)
                ref_ts_artist, ref_lm_artist = self.get_artist(REFERENCE)

                artist_list.append(ts_artist)
                artist_list.append(ref_ts_artist)
                label_list.append('TS ' + method)
                label_list.append('TS ' + REFERENCE)

                plot.ax_rect.legend(artist_list,
                                     label_list,
                                     scatterpoints=1, fontsize=8, frameon=False,
                                     framealpha=0.75,
                                     bbox_to_anchor=(0.5, -0.15), loc=9, borderaxespad=0,
                                     ncol=len(self.met_colors_dict)).set_zorder(100)

            plot.ax_rect.set_ylim(185, -5)
            plot.ax_rect.set_xlim(-5, 365)

            plot.save(dir_=dir, filename=filename)

    # # # norm plotting # # #
    #region
    def plot_raw_data_norm_LM(self, plot, method, connect_to_skm=False, plot_criteria=False):
        skm_groupings = self.reference_landscape.LM_Tessellation.methods[method]['skm_groupings']

        amt = 0.5
        scaling = 1

        for i in range(len(skm_groupings)):
            for j in range(len(skm_groupings[i]['structures'])):
                LM = skm_groupings[i]['structures'][j]

                if LM.gibbs > amt:
                    LM_size = LM.gibbs * scaling
                else:
                    LM_size = amt * scaling

                skm_vert = cart2pol(self.reference_landscape.LM_Tessellation.skm.cluster_centers_[LM.closest_skm])
                LM_vert = [LM.phi, LM.theta]

                if connect_to_skm:
                    if LM.type == 'LM_added':
                        linestyle = '-.'
                    else:
                        linestyle = '-'

                    self.plot_line(plot, LM_vert, skm_vert, method, line_style=linestyle, plot_TS_vert=False, plot_LM_vert=False)

                if plot_criteria:
                    tessellation = self.reference_landscape.LM_Tessellation
                    structure = LM

                    skm_1 = cart2pol(tessellation.skm.cluster_centers_[i])
                    skm_2 = cart2pol(tessellation.skm.cluster_centers_[structure.next_closest_skm])

                    skm_dist = arc_length_calculator(skm_1[0], skm_1[1], skm_2[0], skm_2[1])

                    arc = structure.comp_metrics['arc']
                    next_arc = structure.comp_metrics['next_arc']

                    dist_tol = 1.2
                    gibbs_diff_tol = 1

                    if arc == 0:
                        dist_ratio = dist_tol
                    else:
                        dist_ratio = next_arc / arc

                    self.plot_line(plot, [structure.phi, structure.theta],
                                   cart2pol(tessellation.skm.cluster_centers_[structure.next_closest_skm]), method,
                                   line_style=':',
                                   plot_LM_vert=False, plot_TS_vert=False)

                    if LM.type != 'LM_added':
                        amt = 10
                        width = 2

                        if structure.comp_metrics['next_arc'] < skm_dist * 0.75 or dist_ratio < dist_tol:
                            if len(tessellation.methods[REFERENCE]['skm_groupings'][structure.next_closest_skm]['structures']) > 0:
                                if structure.comp_metrics['next_arc'] < skm_dist * 0.75:
                                    plot.ax_rect.scatter(LM.phi, LM.theta, c='',
                                                         edgecolor='black',
                                                         marker=self.met_lm_markers_dict[method],
                                                         s=LM_size * (amt + width), zorder=10)

                                if dist_ratio < dist_tol:
                                    plot.ax_rect.scatter(LM.phi, LM.theta, c='',
                                                         edgecolor='red',
                                                         marker=self.met_lm_markers_dict[method],
                                                         s=LM_size * amt, zorder=10)

                                if structure.gibbs is not None and tessellation.methods[method]['skm_groupings'][structure.next_closest_skm][
                                    tessellation.type + '_weighted_gibbs'] is None and\
                                    tessellation.methods[REFERENCE]['skm_groupings'][structure.next_closest_skm][tessellation.type + '_weighted_gibbs'] is not None:

                                    plot.ax_rect.scatter(LM.phi, LM.theta, c='',
                                                         edgecolor='orange',
                                                         marker=self.met_lm_markers_dict[method],
                                                         s=LM_size * (amt - 1 * width), zorder=10)
                                elif abs(structure.gibbs - tessellation.methods[method]['skm_groupings'][structure.next_closest_skm][
                                        tessellation.type + '_weighted_gibbs']) < gibbs_diff_tol:
                                    plot.ax_rect.scatter(LM.phi, LM.theta, c='',
                                                         edgecolor='yellow',
                                                         marker=self.met_lm_markers_dict[method],
                                                         s=LM_size * (amt - 2 * width), zorder=10)

                                if self.weighting_is_significant(tessellation, structure, structure.next_closest_skm, method):
                                    plot.ax_rect.scatter(LM.phi, LM.theta, c='',
                                                         edgecolor='green',
                                                         marker=self.met_lm_markers_dict[method],
                                                         s=LM_size * (amt - 3 * width), zorder=10)

                plot.ax_rect.scatter(LM.phi, LM.theta, c='white',
                                     edgecolor=self.met_colors_dict[method],
                                     marker=self.met_lm_markers_dict[method],
                                     s=LM_size, zorder=10)
                plot.ax_rect.annotate(str(round(LM.gibbs, 1)),
                                      xy=(LM.phi, LM.theta), xytext=(LM.phi, LM.theta),
                                      ha="center", va="center", fontsize=3 * scaling, zorder=100)

    def save_raw_data_norm_LM(self, method='ALL', plot_ref=True, connect_to_skm=False, plot_criteria=False):
        filename = self.molecule + '-' + method + '-raw_data_norm_LM'
        dir1 = make_dir(os.path.join(os.path.join(self.MOL_SAVE_DIR, 'plots'), 'raw_data_norm'))
        dir = make_dir(os.path.join(dir1, 'LM'))

        if plot_criteria:
            dir = make_dir(os.path.join(dir, 'plot_criteria'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.reference_landscape.plot_voronoi_regions(plot=plot, tessellation=self.reference_landscape.LM_Tessellation)
            self.reference_landscape.plot_cano(plot=plot)
            self.reference_landscape.plot_skm_centers(plot=plot, tessellation=self.reference_landscape.LM_Tessellation)

            artist_list = []
            label_list = []

            if method == 'ALL':
                for method in self.Method_Pathways_dict:
                    self.plot_raw_data_norm_LM(plot=plot, method=method, connect_to_skm=connect_to_skm, plot_criteria=plot_criteria)

                    ts_artist, lm_artist = self.get_artist(method)

                    artist_list.append(lm_artist)
                    label_list.append('LM ' + method)

                    plot.ax_rect.legend(artist_list,
                                        label_list,
                                        scatterpoints=1, fontsize=8, frameon=True,
                                        framealpha=0.75,
                                        bbox_to_anchor=(1.2, 0.5), loc='right', borderaxespad=0,
                                        ncol=1).set_zorder(100)
            else:
                if plot_ref:
                    self.plot_raw_data_LM(plot=plot, method=REFERENCE)

                self.plot_raw_data_norm_LM(plot=plot, method=method, connect_to_skm=connect_to_skm, plot_criteria=plot_criteria)

                ts_artist, lm_artist = self.get_artist(method)
                ref_ts_artist, ref_lm_artist = self.get_artist(REFERENCE)

                artist_list.append(lm_artist)
                if plot_ref:
                    artist_list.append(ref_lm_artist)

                label_list.append('LM ' + method)
                if plot_ref:
                    label_list.append('LM ' + REFERENCE)

                plot.ax_rect.legend(artist_list,
                                    label_list,
                                    scatterpoints=1, fontsize=8, frameon=False,
                                    framealpha=0.75,
                                    bbox_to_anchor=(0.5, -0.15), loc=9, borderaxespad=0,
                                    ncol=len(self.met_colors_dict)).set_zorder(100)

            plot.ax_rect.set_ylim(185, -5)
            plot.ax_rect.set_xlim(-5, 365)

            plot.save(dir_=dir, filename=filename, width=18, height=10)

    def plot_raw_data_norm_TS(self, plot, method, connect_to_skm=False, plot_criteria=False):
        skm_groupings = self.reference_landscape.TS_Tessellation.methods[method]['skm_groupings']

        amt = 0.5
        scaling = 1

        for i in range(len(skm_groupings)):
            for j in range(len(skm_groupings[i]['structures'])):
                TS = skm_groupings[i]['structures'][j]

                if TS.gibbs > amt:
                    TS_size = TS.gibbs * scaling
                else:
                    TS_size = amt * scaling

                skm_vert = cart2pol(self.reference_landscape.TS_Tessellation.skm.cluster_centers_[TS.closest_skm])
                TS_vert = [TS.phi, TS.theta]

                if connect_to_skm:
                    if TS.type == 'TS_added':
                        linestyle = '-.'
                    else:
                        linestyle = '-'

                    self.plot_line(plot, TS_vert, skm_vert, method, line_style=linestyle, plot_TS_vert=False,  plot_LM_vert=False)

                if plot_criteria:
                    tessellation = self.reference_landscape.TS_Tessellation
                    structure = TS

                    skm_1 = cart2pol(tessellation.skm.cluster_centers_[i])
                    skm_2 = cart2pol(tessellation.skm.cluster_centers_[structure.next_closest_skm])

                    skm_dist = arc_length_calculator(skm_1[0], skm_1[1], skm_2[0], skm_2[1])

                    arc = structure.comp_metrics['arc']
                    next_arc = structure.comp_metrics['next_arc']

                    dist_tol = 1.2
                    gibbs_diff_tol = 1

                    if arc == 0:
                        dist_ratio = dist_tol
                    else:
                        dist_ratio = next_arc / arc

                    self.plot_line(plot, [structure.phi, structure.theta],
                                   cart2pol(tessellation.skm.cluster_centers_[structure.next_closest_skm]), method,
                                   line_style=':',
                                   plot_LM_vert=False, plot_TS_vert=False)

                    if TS.type != 'TS_added':
                        amt = 10
                        width = 2

                        if len(tessellation.methods[method]['skm_groupings'][i]['structures']) == 0:
                            max_raw_dist = skm_dist * 0.75 / 3
                        else:
                            max_raw_dist = self.get_max_raw_dist(tessellation, method, i)

                        if structure.comp_metrics['next_arc'] < max_raw_dist*3:
                            plot.ax_rect.scatter(TS.phi, TS.theta, c='',
                                                 edgecolor='gray',
                                                 marker=self.met_ts_markers_dict[method],
                                                 s=TS_size * (amt + 2*width), zorder=10)

                            if structure.comp_metrics['next_arc'] < skm_dist * 0.75 or dist_ratio < dist_tol:
                                if len(tessellation.methods[REFERENCE]['skm_groupings'][structure.next_closest_skm]['structures']) > 0:
                                    if structure.comp_metrics['next_arc'] < skm_dist * 0.75:
                                        plot.ax_rect.scatter(TS.phi, TS.theta, c='',
                                                             edgecolor='black',
                                                             marker=self.met_ts_markers_dict[method],
                                                             s=TS_size * (amt + width), zorder=10)

                                    if dist_ratio < dist_tol:
                                        plot.ax_rect.scatter(TS.phi, TS.theta, c='',
                                                             edgecolor='red',
                                                             marker=self.met_ts_markers_dict[method],
                                                             s=TS_size * amt, zorder=10)

                                    if structure.gibbs is not None and tessellation.methods[method]['skm_groupings'][structure.next_closest_skm][
                                        tessellation.type + '_weighted_gibbs'] is None and\
                                        tessellation.methods[REFERENCE]['skm_groupings'][structure.next_closest_skm][tessellation.type + '_weighted_gibbs'] is not None:

                                        plot.ax_rect.scatter(TS.phi, TS.theta, c='',
                                                             edgecolor='orange',
                                                             marker=self.met_ts_markers_dict[method],
                                                             s=TS_size * (amt - 1 * width), zorder=10)

                                    if abs(structure.gibbs - tessellation.methods[method]['skm_groupings'][structure.next_closest_skm][
                                            tessellation.type + '_weighted_gibbs']) < gibbs_diff_tol:
                                        plot.ax_rect.scatter(TS.phi, TS.theta, c='',
                                                             edgecolor='yellow',
                                                             marker=self.met_ts_markers_dict[method],
                                                             s=TS_size * (amt - 2 * width), zorder=10)

                                    if self.weighting_is_significant(tessellation, structure, structure.next_closest_skm, method):
                                        plot.ax_rect.scatter(TS.phi, TS.theta, c='',
                                                             edgecolor='green',
                                                             marker=self.met_ts_markers_dict[method],
                                                             s=TS_size * (amt - 3 * width), zorder=10)

                plot.ax_rect.scatter(TS.phi, TS.theta, c='white',
                                     edgecolor=self.met_colors_dict[method],
                                     marker=self.met_ts_markers_dict[method],
                                     s=TS_size, zorder=10)
                plot.ax_rect.annotate(str(round(TS.gibbs, 1)),
                                      xy=(TS.phi, TS.theta), xytext=(TS.phi, TS.theta),
                                      ha="center", va="center", fontsize=3 * scaling, zorder=100)

    def save_raw_data_norm_TS(self, method='ALL', plot_ref=True, connect_to_skm=False, plot_criteria=False):
        filename = self.molecule + '-' + method + '-raw_data_norm_TS'
        dir1 = make_dir(os.path.join(os.path.join(self.MOL_SAVE_DIR, 'plots'), 'raw_data_norm'))
        dir = make_dir(os.path.join(dir1, 'TS'))

        if plot_criteria:
            dir = make_dir(os.path.join(dir, 'plot_criteria'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.reference_landscape.plot_voronoi_regions(plot=plot,
                                                          tessellation=self.reference_landscape.TS_Tessellation)
            self.reference_landscape.plot_cano(plot=plot)
            self.reference_landscape.plot_skm_centers(plot=plot, tessellation=self.reference_landscape.TS_Tessellation)

            artist_list = []
            label_list = []

            if method == 'ALL':
                for method in self.Method_Pathways_dict:
                    self.plot_raw_data_norm_TS(plot=plot, method=method, connect_to_skm=connect_to_skm, plot_criteria=plot_criteria)

                    ts_artist, lm_artist = self.get_artist(method)

                    artist_list.append(ts_artist)
                    label_list.append('TS ' + method)

                    plot.ax_rect.legend(artist_list,
                                        label_list,
                                        scatterpoints=1, fontsize=8, frameon=True,
                                        framealpha=0.75,
                                        bbox_to_anchor=(1.2, 0.5), loc='right', borderaxespad=0,
                                        ncol=1).set_zorder(100)
            else:
                if plot_ref:
                    self.plot_raw_data_TS(plot=plot, method=REFERENCE)

                self.plot_raw_data_norm_TS(plot=plot, method=method, connect_to_skm=connect_to_skm, plot_criteria=plot_criteria)

                ts_artist, lm_artist = self.get_artist(method)
                ref_ts_artist, ref_lm_artist = self.get_artist(REFERENCE)

                artist_list.append(ts_artist)
                if plot_ref:
                    artist_list.append(ref_ts_artist)

                label_list.append('TS ' + method)
                if plot_ref:
                    label_list.append('TS ' + REFERENCE)

                plot.ax_rect.legend(artist_list,
                                    label_list,
                                    scatterpoints=1, fontsize=8, frameon=False,
                                    framealpha=0.75,
                                    bbox_to_anchor=(0.5, -0.15), loc=9, borderaxespad=0,
                                    ncol=len(self.met_colors_dict)).set_zorder(100)

            plot.ax_rect.set_ylim(185, -5)
            plot.ax_rect.set_xlim(-5, 365)

            plot.save(dir_=dir, filename=filename, width=18, height=10)
    #endregion

    def plot_line(self, plot, TS_vert, LM_vert, method, zorder=10, line_style='-', plot_TS_vert=True, plot_LM_vert=True):
        size = 30
        color = self.met_colors_dict[method]

        line = get_pol_coords(pol2cart(TS_vert), pol2cart(LM_vert))

        if (is_end(line)):
            two_edges = split_in_two(line)

            plot.ax_rect.plot(two_edges[0][0], two_edges[0][1], color=color, linestyle=line_style, zorder=1)
            plot.ax_rect.plot(two_edges[1][0], two_edges[1][1], color=color, linestyle=line_style, zorder=1)
        else:
            plot.ax_rect.plot(line[0], line[1], color=color, linestyle=line_style)

        if plot_TS_vert:
            plot.ax_rect.scatter(TS_vert[0], TS_vert[1], c='white',
                                 edgecolor=color,
                                 marker=self.met_ts_markers_dict[method],
                                 s=size, zorder=zorder)

        if plot_LM_vert:
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

            if LM1.closest_skm < LM2.closest_skm:
                lm_grouping = str(LM1.closest_skm) + '_' + str(LM2.closest_skm)
            else:
                lm_grouping = str(LM2.closest_skm) + '_' + str(LM1.closest_skm)

            key = lm_grouping + '-' + str(TS.closest_skm)

            # if self.Method_Pathways_dict[method].pathway_groupings[key]['norm_pathways']\
            #     == self.Method_Pathways_dict[REFERENCE].pathway_groupings[key]['norm_pathways']:
            #
            #     linestyle = '-'
            # else:
            #     linestyle = '--'

            linestyle = '-'

            TS_vert = [TS.phi, TS.theta]
            LM1_vert = [LM1.phi, LM1.theta]
            LM2_vert = [LM2.phi, LM2.theta]

            self.plot_line(plot, TS_vert, LM1_vert, method, line_style=linestyle)
            self.plot_line(plot, TS_vert, LM2_vert, method, line_style=linestyle)

    def save_connectivity(self, method='ALL'):
        filename = self.molecule + '-' + method + '-connectivity'
        dir1 = make_dir(os.path.join(self.MOL_SAVE_DIR, 'plots'))
        dir = make_dir(os.path.join(dir1, 'connectivity'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.reference_landscape.plot_voronoi_regions(plot=plot, tessellation=self.reference_landscape.TS_Tessellation)
            self.reference_landscape.plot_skm_centers(plot=plot, tessellation=self.reference_landscape.TS_Tessellation)

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
                    label_list.append('TS ' + method)

                    plot.ax_rect.legend(artist_list,
                                        label_list,
                                        scatterpoints=1, fontsize=8, frameon=True,
                                        framealpha=0.75,
                                        bbox_to_anchor=(1.2, 0.5), loc='right', borderaxespad=0,
                                        ncol=1).set_zorder(100)
            else:
                self.plot_connectivity(plot=plot, method=REFERENCE)
                self.plot_connectivity(plot=plot, method=method)

                ts_artist, lm_artist = self.get_artist(method)
                ref_ts_artist, ref_lm_artist = self.get_artist(REFERENCE)

                artist_list.append(ts_artist)
                artist_list.append(lm_artist)
                artist_list.append(ref_ts_artist)
                artist_list.append(ref_lm_artist)

                label_list.append('TS ' + method)
                label_list.append('TS ' + method)
                label_list.append('TS ' + REFERENCE)
                label_list.append('TS ' + REFERENCE)

                plot.ax_rect.legend(artist_list,
                                     label_list,
                                     scatterpoints=1, fontsize=8, frameon=False,
                                     framealpha=0.75,
                                     bbox_to_anchor=(0.5, -0.15), loc=9, borderaxespad=0,
                                     ncol=len(self.met_colors_dict)).set_zorder(100)

            plot.ax_rect.set_xlim(-5, 365)
            plot.ax_rect.set_ylim(185, -5)

            plot.save(dir_=dir, filename=filename)


    def save_tessellation(self, tessellation):
        filename = self.molecule + '-tessellation-' + tessellation.type
        dir = make_dir(os.path.join(self.MOL_SAVE_DIR, 'plots'))

        if not os.path.exists(os.path.join(dir, filename + '.png')):
            plot = Plots(rect_arg=True)

            self.reference_landscape.plot_voronoi_regions(plot=plot, tessellation=tessellation)
            self.reference_landscape.plot_regions_names(plot=plot, tessellation=tessellation)
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

    def format_WRMSD_dict_for_csv(self, comp_key):
        csv_dict = {}

        csv_dict['method'] = []

        for method in self.Method_Pathways_dict:
            csv_dict['method'].append(method)

        csv_dict[comp_key + '_WRMSD'] = []

        for method in self.Method_Pathways_dict:
            csv_dict[comp_key + '_WRMSD'].append(self.Method_Pathways_dict[method].comp_metrics[comp_key + '_WRMSD'])

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
        self.write_dict_to_csv(self.format_WRMSD_dict_for_csv('arc'), 'arc_WRMSD')

        self.write_dict_to_csv(self.format_skm_dict_for_csv('weighted_gibbs'), 'weighted_gibbs')

        self.write_dict_to_csv(self.format_pathway_dict_for_csv(), 'pathways')
        self.write_dict_to_csv(self.format_norm_pathway_dict_for_csv(), 'norm_pathways')
    #endregion

    pass
