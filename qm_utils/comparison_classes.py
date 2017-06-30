import os

import numpy as np
from spherecluster import SphericalKMeans
from scipy.spatial import SphericalVoronoi

from qm_utils.spherical_kmeans_voronoi import read_csv_to_dict, read_csv_canonical_designations,\
                                                pol2cart, cart2pol, plot_line, Plots

###################################### Directories ######################################
#                                                                                       #
#########################################################################################
#region
QM_1_DIR = os.path.dirname(__file__)

# root of project
QM_0_DIR = os.path.dirname(QM_1_DIR)

PROG_DATA_DIR = os.path.join(QM_0_DIR, 'pucker_prog_data')

MET_COMP_DIR = os.path.join(PROG_DATA_DIR, 'method_comparison')
SV_DIR = os.path.join(PROG_DATA_DIR, 'spherical_kmeans_voronoi')
#endregion

### IMPORTANT: Check that all units are read in correctly from .csv files ###
#check freq in parsing#

class Structure():
    def __init__(self, phi, theta, gibbs=None, name=None):
        self.phi = phi
        self.theta = theta
        self.gibbs = gibbs
        self.name = name

class Local_Minimum(Structure):
    def __init__(self, phi, theta, gibbs, name):
        Structure.__init__(self, phi, theta, gibbs, name)

class Transition_State(Structure):
    def __init__(self, phi, theta, gibbs, name):
        Structure.__init__(self, phi, theta, gibbs, name)

class Pathway():
    def __init__(self, TS, LM1, LM2, Kmeans_TS=None, Kmeans_LM1=None, Kmeans_LM2=None):
        self.TS = TS
        self.LM1 = LM1
        self.LM2 = LM2

        self.Kmeans_TS = Kmeans_TS
        self.Kmeans_LM1 = Kmeans_LM1
        self.Kmeans_LM2 = Kmeans_LM2

        self.check_energies()

    # checks that the TS energy is higher than either LM energy
    def check_energies(self):
        return

class Method_Pathways():
    def __init__(self, LM_csv_filename, TS_csv_filename, IRC_csv_filename):
        self.parse_LM_csv(LM_csv_filename)
        self.parse_TS_csv(TS_csv_filename)
        self.parse_IRC_csv(IRC_csv_filename)

        self.create_Pathways()

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
        self.TS_csv_list = []

        TS_csv_dict = read_csv_to_dict(TS_csv_filename, mode='r')

        for i in range(len(TS_csv_dict)):
            info = TS_csv_dict[i]
            phi = float(info['phi'])
            theta = float(info['theta'])
            gibbs = float(info['G298 (Hartrees)'])
            name = info['File Name'].split('-')[0].split('-')[0]

            self.TS_csv_list.append(Transition_State(phi, theta, gibbs, name))

    def parse_IRC_csv(self, IRC_csv_filename):
        self.IRC_csv_list = []

        IRC_csv_dict = read_csv_to_dict(IRC_csv_filename, mode='r')

        for i in range(len(IRC_csv_dict)):
            info = IRC_csv_dict[i]
            phi = float(info['phi'])
            theta = float(info['theta'])
            gibbs = float(info['G298 (Hartrees)'])

            if 'ircr' in info['File Name']:
                direction = 'ircr'
            else:
                direction = 'ircf'

            pucker = info['File Name'].split('-')[0].split('-')[0]

            name = pucker + '_' + direction

            self.IRC_csv_list.append(Local_Minimum(phi, theta, gibbs, name))

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

class Reference_Landscape():
    def __init__(self, LM_csv_filename, TS_csv_filename, IRC_csv_filename):
        self.Reference_Pathways = Method_Pathways(LM_csv_filename, TS_csv_filename, IRC_csv_filename).Pathways
        self.Reference_Structures = self.get_Reference_Structures()

        self.cano_designations = read_csv_canonical_designations('CP_params.csv', SV_DIR)
        self.reorg_canonical()
        self.populate_unbinned_canos()

        self.tessellate(36)

    def get_Reference_Structures(self):
        structures_list = []

        for i in range(len(self.Reference_Pathways)):
            structures_list.append(self.Reference_Pathways[i].TS)
            structures_list.append(self.Reference_Pathways[i].LM1)
            structures_list.append(self.Reference_Pathways[i].LM2)

        return structures_list

    def reorg_canonical(self):
        aux_list = []

        for i in range(len(self.cano_designations['pucker'])):
            name = self.cano_designations['pucker'][i]
            phi = self.cano_designations['phi_cano'][i]
            theta = self.cano_designations['theta_cano'][i]

            aux_list.append(Structure(phi=phi,
                                      theta=theta,
                                      name=name))

        self.canonical_designations = aux_list

    def populate_unbinned_canos(self):
        self.unbinned_canos = []

        for i in range(len(self.canonical_designations)):
            unbinned = True

            for j in range(len(self.Reference_Structures)):
                if self.canonical_designations[i].name == self.Reference_Structures[j].name:
                    unbinned = False

            if unbinned:
                self.unbinned_canos.append(self.canonical_designations)

    def tessellate(self, number_clusters):
        centers = []

        # populate all centers to be used in voronoi tessellation
        for i in range(len(self.Reference_Structures)):
            structure = self.Reference_Structures[i]
            center = pol2cart([structure.phi, structure.theta])

            centers.append(center)

        for i in range(len(self.unbinned_canos[0])):
            structure = self.unbinned_canos[0][i]

            center = pol2cart([float(structure.phi), float(structure.theta)])

            centers.append(center)

        # Uses packages to calculate the k-means spherical centers
        self.skm = SphericalKMeans(n_clusters=number_clusters, init='k-means++', n_init=30)
        self.skm.fit(centers)
        skm_centers = self.skm.cluster_centers_

        # Default parameters for spherical voronoi
        radius = 1
        center = np.array([0, 0, 0])

        # Spherical Voronoi for the centers
        self.sv = SphericalVoronoi(skm_centers, radius, center)
        self.sv.sort_vertices_of_regions()

    def plot_voronoi_regions(self, plot):
        for i in range(len(self.sv.regions)):
            for j in range(len(self.sv.regions[i])):
                if j == len(self.sv.regions[i]) - 1:
                    index1 = self.sv.regions[i][j]
                    index2 = self.sv.regions[i][0]

                    vert1 = self.sv.vertices[index1]
                    vert2 = self.sv.vertices[index2]

                    plot_line(plot.ax_rect, [vert1, 'green', 30], [vert2, 'green', 30], 'green')
                else:
                    index1 = self.sv.regions[i][j]
                    index2 = self.sv.regions[i][j + 1]

                    vert1 = self.sv.vertices[index1]
                    vert2 = self.sv.vertices[index2]

                    plot_line(plot.ax_rect, [vert1, 'green', 30], [vert2, 'green', 30], 'green')

    def plot_skm_centers(self, plot):
        phi_vals = []
        theta_vals = []
        for i in range(len(self.skm.cluster_centers_)):
            vert = cart2pol(self.skm.cluster_centers_[i])

            phi_vals.append(vert[0])
            theta_vals.append(vert[1])

        plot.ax_rect.scatter(phi_vals, theta_vals)


class Compare_Methods():
    def __init__(self):
        pass


def run_test():
    plot = Plots(merc_arg=True)

    OX_DIR = os.path.join(SV_DIR, 'oxane')

    LM_csv_filename = os.path.join(OX_DIR, 'z_dataset-oxane-LM-B3LYP.csv')
    TS_csv_filename = os.path.join(OX_DIR, 'z_dataset-oxane-TS-B3LYP.csv')
    IRC_csv_filename = os.path.join(OX_DIR, 'z_dataset-oxane-IRC-B3LYP.csv')

    reference_landscape = Reference_Landscape(LM_csv_filename, TS_csv_filename, IRC_csv_filename)

    reference_landscape.plot_voronoi_regions(plot)
    reference_landscape.plot_skm_centers(plot)

    plot.ax_rect.set_ylim(185, -5)

    plot.show()
