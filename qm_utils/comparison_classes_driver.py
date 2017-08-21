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
        # if color is red
        if seed_num == increment * 6:
            seed_num += increment

        color = cmap(seed_num)

        if method == 'M06L':
            color = cmap(increment * 16)
        elif method == 'PBEPBE':
            color = cmap(increment * 18)

        seed_num += increment
        met_colors_dict[method] = color

        ts_marker = mpl.markers.MarkerStyle.filled_markers[i]
        lm_marker = mpl.markers.MarkerStyle.filled_markers[i]
        i += 1
        met_ts_markers_dict[method] = ts_marker
        met_lm_markers_dict[method] = lm_marker

    mol_list = ['bglc', 'bxyl', 'oxane']

    for i in range(len(mol_list)):
        comp_met = cc.Compare_Methods(mol_list[i],
                                      met_colors_dict,
                                      met_ts_markers_dict,
                                      met_lm_markers_dict)

        comp_met.write_csvs()

        for method in comp_met.Method_Pathways_dict:
            comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                       method=method,
                                       type='raw')
            comp_met.save_connectivity(tessellation=comp_met.reference_landscape.LM_Tessellation,
                                       method=method,
                                       type='raw')
            comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                       method=method,
                                       type='skm')
            comp_met.save_connectivity(tessellation=comp_met.reference_landscape.LM_Tessellation,
                                       method=method,
                                       type='skm')

            comp_met.save_raw_data_norm_LM(method, connect_to_skm=True)
            comp_met.save_raw_data_norm_TS(method, connect_to_skm=True)

            comp_met.save_raw_data_norm_LM(method, connect_to_skm=True, plot_criteria=True)
            comp_met.save_raw_data_norm_TS(method, connect_to_skm=True, plot_criteria=True)

            comp_met.save_raw_data_norm_LM(method, connect_to_skm=False)
            comp_met.save_raw_data_norm_TS(method, connect_to_skm=False)

        # comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
        #                            method='ALL',
        #                            type='raw')
        # comp_met.save_connectivity(tessellation=comp_met.reference_landscape.LM_Tessellation,
        #                            method='ALL',
        #                            type='raw')
        # comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
        #                            method='ALL',
        #                            type='skm')
        # comp_met.save_connectivity(tessellation=comp_met.reference_landscape.LM_Tessellation,
        #                            method='ALL',
        #                            type='skm')

        comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                   method='DFT',
                                   type='raw')
        comp_met.save_connectivity(tessellation=comp_met.reference_landscape.LM_Tessellation,
                                   method='DFT',
                                   type='raw')
        comp_met.save_connectivity(tessellation=comp_met.reference_landscape.TS_Tessellation,
                                   method='DFT',
                                   type='skm')
        comp_met.save_connectivity(tessellation=comp_met.reference_landscape.LM_Tessellation,
                                   method='DFT',
                                   type='skm')

        # comp_met.save_raw_data_norm_LM('ALL', connect_to_skm=True)
        # comp_met.save_raw_data_norm_TS('ALL', connect_to_skm=True)

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
