"""
The purpose of this script is to make comparisons for a particular QM method to the reference set of HSP.
"""

# # # import # # #
#region
from __future__ import print_function

import os
import sys

import csv

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import qm_utils.comparison_classes as cc


def main():
    do_main()

    return 0

def do_main():
    methods_list = []

    methods_list.append('REF')
    methods_list.append('B3LYP')
    methods_list.append('APFD')
    methods_list.append('BMK')
    methods_list.append('M06L')
    methods_list.append('PBEPBE')
    methods_list.append('DFTB')
    methods_list.append('DFTB3')
    methods_list.append('AM1')
    methods_list.append('PM3')
    methods_list.append('PM3MM')
    methods_list.append('PM6')
    methods_list.append('PM7')

    cmap = plt.get_cmap('Paired')
    met_colors_dict = {}

    met_colors_dict['REF'] = cmap.colors[1]
    met_colors_dict['B3LYP'] = cmap.colors[0]
    met_colors_dict['APFD'] = cmap.colors[7]
    met_colors_dict['BMK'] = cmap.colors[2]
    met_colors_dict['M06L'] = cmap.colors[6]
    met_colors_dict['M062X'] = 'sienna'
    met_colors_dict['PBEPBE'] = cmap.colors[3]
    met_colors_dict['DFTB'] = cmap.colors[4]
    met_colors_dict['DFTB3'] = 'teal'
    met_colors_dict['AM1'] = 'mediumvioletred'
    met_colors_dict['PM3'] = cmap.colors[8]
    met_colors_dict['PM3MM'] = cmap.colors[9]
    met_colors_dict['PM6'] = cmap.colors[11]
    met_colors_dict['PM7'] = 'rosybrown'

    met_ts_markers_dict = {}
    met_lm_markers_dict = {}

    i = 0

    for method in list(methods_list):
        ts_marker = 'o'
        lm_marker = 'o'

        met_ts_markers_dict[method] = ts_marker
        met_lm_markers_dict[method] = lm_marker

    mol_list = ['bglc', 'bxyl', 'oxane']

    energy_format = 'H298 (Hartrees)'

    for i in range(len(mol_list)):
        comp_met = cc.Compare_Methods(mol_list[i],
                                      met_colors_dict,
                                      met_ts_markers_dict,
                                      met_lm_markers_dict,
                                      energy_format)

        comp_met.write_csvs()

        for method in comp_met.Method_Pathways_dict:
            comp_met.write_ind_grouping_csv(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                            method=method, energy_type=energy_format)
            comp_met.write_ind_grouping_csv(tessellation=comp_met.reference_landscape.LM_Tessellation,
                                            method=method, energy_type=energy_format)

        comp_met.save_circ_paths('REF', 'N', True)
        comp_met.save_circ_paths('REF', 'S', True)

        for method in comp_met.Method_Pathways_dict:
            comp_met.save_circ_paths(method, 'N')
            comp_met.save_circ_paths(method, 'S')

        for method in comp_met.Method_Pathways_dict:
            comp_met.save_raw_data_norm_LM(method, connect_to_skm=True, plot_criteria=True)
            comp_met.save_raw_data_norm_TS(method, connect_to_skm=True, plot_criteria=True)

        for method in comp_met.Method_Pathways_dict:
            comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                       method=method,
                                       type='raw')
            comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                       method=method,
                                       type='skm')

            comp_met.save_raw_data_norm_LM(method, connect_to_skm=False)
            comp_met.save_raw_data_norm_TS(method, connect_to_skm=False)

        dehyd_filename = 'C:/Users/justi/Desktop/Winter 2017/Research/codes/python/qm_utils/pucker_prog_data/spherical_kmeans_voronoi/molecules/bglc/z_dataset-dehy.csv'
        comp_met.save_dehyd(dehyd_filename=dehyd_filename)

    energy_format = 'G298 (Hartrees)'

    for i in range(len(mol_list)):
        comp_met = cc.Compare_Methods(mol_list[i],
                                      met_colors_dict,
                                      met_ts_markers_dict,
                                      met_lm_markers_dict,
                                      energy_format)

        comp_met.write_csvs()

        for method in comp_met.Method_Pathways_dict:
            comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                       method=method,
                                       type='raw')
            comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                       method=method,
                                       type='skm')

            comp_met.save_raw_data_norm_LM(method, connect_to_skm=True)
            comp_met.save_raw_data_norm_TS(method, connect_to_skm=True)

            comp_met.save_raw_data_norm_LM(method, connect_to_skm=True, plot_criteria=True)
            comp_met.save_raw_data_norm_TS(method, connect_to_skm=True, plot_criteria=True)

            comp_met.save_raw_data_norm_LM(method, connect_to_skm=False)
            comp_met.save_raw_data_norm_TS(method, connect_to_skm=False)

        comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                   method='DFT',
                                   type='raw')
        comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                   method='DFT',
                                   type='skm')

        comp_met.save_raw_data_norm_LM('DFT', connect_to_skm=False)
        comp_met.save_raw_data_norm_TS('DFT', connect_to_skm=False)

        comp_met.save_raw_data_norm_LM('DFT', connect_to_skm=True)
        comp_met.save_raw_data_norm_TS('DFT', connect_to_skm=True)

        comp_met.save_tessellation(comp_met.reference_landscape.LM_Tessellation)
        comp_met.save_tessellation(comp_met.reference_landscape.TS_Tessellation)

if __name__ == '__main__':
    status = main()
    sys.exit(status)
#endregion
