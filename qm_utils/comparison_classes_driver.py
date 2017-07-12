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
    #cluster()
    #do_main()
    save_plots()

    return 0

def cluster():
    comp_met = cc.Compare_Methods('bxyl', [36, 900, 1000], {}, {}, {})
    comp_met.reference_landscape.save_tessellations()
    comp_met.reference_landscape.save_tessellations_LM()
    comp_met.reference_landscape.save_tessellations_TS()

def save_plots():
    methods_list = []

    methods_list.append('REFERENCE')
    methods_list.append('B3LYP')
    methods_list.append('APFD')
    methods_list.append('BMK')
    methods_list.append('M06L')
    methods_list.append('PBEPBE')
    methods_list.append('DFTB')
    methods_list.append('AM1')
    methods_list.append('PM3')
    methods_list.append('PM3MM')
    methods_list.append('PM6')

    cmap = plt.get_cmap('Vega20')
    # allows for incrementing over 20 colors
    increment = 0.0524
    seed_num = 0
    i = 0

    met_colors_dict = {}
    met_ts_markers_dict = {}
    met_lm_markers_dict = {}

    for method in list(methods_list):
        color = cmap(seed_num)
        seed_num += increment
        met_colors_dict[method] = color

        ts_marker = mpl.markers.MarkerStyle.filled_markers[i]
        lm_marker = mpl.markers.MarkerStyle.filled_markers[i]
        i += 1
        met_ts_markers_dict[method] = ts_marker
        met_lm_markers_dict[method] = lm_marker

    mol_list = ['bglc', 'bxyl', 'oxane']

    # [c, s, i]
    skm_params = [[40, 900, 900], [40, 1200, 300], [38, 100, 200]]
    skm_params = [[38, 1000, 1000], [38, 1000, 1000], [38, 1000, 1000]]
    LM_clusters = [13, 9, 8]
    TS_clusters = [20, 20, 20]

    for i in range(len(mol_list)):
        comp_met = cc.Compare_Methods(mol_list[i],
                                      skm_params[i],
                                      met_colors_dict,
                                      met_ts_markers_dict,
                                      met_lm_markers_dict)

        comp_met.save_tessellation()

        for method in comp_met.Method_Pathways_dict:
            comp_met.save_raw_data(method, plot_IRC=False)
            comp_met.save_raw_data_LM(method)
            comp_met.save_raw_data_TS(method)
            comp_met.save_connectivity(method)

        comp_met.save_raw_data()
        comp_met.save_connectivity()

def do_main():
    methods_list = []

    methods_list.append('REFERENCE')
    methods_list.append('B3LYP')
    methods_list.append('APFD')
    methods_list.append('BMK')
    methods_list.append('M06L')
    methods_list.append('PBEPBE')
    methods_list.append('DFTB')
    methods_list.append('AM1')
    methods_list.append('PM3')
    methods_list.append('PM3MM')
    methods_list.append('PM6')

    cmap = plt.get_cmap('Vega20')
    # allows for incrementing over 20 colors
    increment = 0.0524
    seed_num = 0
    i = 0

    met_colors_dict = {}
    met_ts_markers_dict = {}
    met_lm_markers_dict = {}

    for method in list(methods_list):
        color = cmap(seed_num)
        seed_num += increment
        met_colors_dict[method] = color

        ts_marker = mpl.markers.MarkerStyle.filled_markers[i]
        lm_marker = mpl.markers.MarkerStyle.filled_markers[i]
        i += 1
        met_ts_markers_dict[method] = ts_marker
        met_lm_markers_dict[method] = lm_marker

    mol_list = ['bglc', 'bxyl', 'oxane']

    # [c, s, i]
    skm_params = [[40, 900, 900], [40, 1200, 300], [38, 100, 200]]
    LM_clusters = [13, 9, 8]
    TS_clusters = [20, 20, 20]

    for i in range(len(mol_list)):
        comp_met = cc.Compare_Methods(mol_list[i],
                                      skm_params[i],
                                      met_colors_dict,
                                      met_ts_markers_dict,
                                      met_lm_markers_dict,
                                      LM_clusters[i],
                                      TS_clusters[i])

        comp_met.save_raw_data_norm('REFERENCE', plot_ref=False, plot_IRC=True)
        comp_met.save_raw_data_norm('REFERENCE', plot_ref=False, plot_IRC=False)

        comp_met.save_raw_data_norm_LM('REFERENCE', plot_ref=False, plot_IRC=True)
        comp_met.save_raw_data_norm_LM('REFERENCE', plot_ref=False, plot_IRC=False)

        comp_met.save_raw_data_norm_TS('REFERENCE', plot_ref=False)

        comp_met.save_tessellation()
        # comp_met.save_tessellation_LM()
        # comp_met.save_tessellation_TS()

        for method in comp_met.Method_Pathways_dict:
            comp_met.save_raw_data(method)
            comp_met.save_connectivity(method)

            comp_met.save_raw_data_LM(method)
            comp_met.save_raw_data_TS(method)

        comp_met.save_raw_data()
        comp_met.save_connectivity()

        #comp_met.write_csvs()

if __name__ == '__main__':
    status = main()
    sys.exit(status)
#endregion
