
"""
The purpose of this script is to make comparisons for a particular QM method to the reference set of HSP.
"""

# # # import # # #
#region
from __future__ import print_function

import os
import sys

import csv

import matplotlib
matplotlib.use('TkAgg')

from shutil import copyfile

from qm_utils.qm_common import read_csv_to_dict
from qm_utils.spherical_kmeans_voronoi import Local_Minima, Transition_States,\
                                              read_csv_canonical_designations, read_csv_data, read_csv_data_TS
from qm_utils.method_comparison import Local_Minima_Compare, Transition_State_Compare, Compare_All_Methods
#endregion

# # # Directories # # #
#region
QM_1_DIR = os.path.dirname(__file__)

# root of project
QM_0_DIR = os.path.dirname(QM_1_DIR)

TEST_DIR = os.path.join(QM_0_DIR, 'tests')
TEST_DATA_DIR = os.path.join(TEST_DIR, 'test_data')

MET_COMP_DIR = os.path.join(TEST_DATA_DIR, 'method_comparison')
SV_DIR = os.path.join(TEST_DATA_DIR, 'spherical_kmeans_voronoi')
#endregion

# # # Helper Functions # # #
#region
# creates a .csv file in the form of the hsp ts .csv file
def rewrite_ts_hartree(ts_hartree_dict_list, method, molecule, dir):
    filename = 'z_dataset-' + molecule + '-TS-' + method + '.csv'

    ts_path_dict = {}

    ts_count = 0
    lm_count = 0

    # separate the dicts in terms of pathway
    for i in range(len(ts_hartree_dict_list)):
        ts_path = ts_hartree_dict_list[i]['File Name'].split('_')[0]

        if ts_path not in ts_path_dict:
            ts_path_dict[ts_path] = []

        ts_path_dict[ts_path].append(ts_hartree_dict_list[i])

        if float(ts_hartree_dict_list[i]['Freq 1']) < 0:
            ts_count += 1
        else:
            lm_count += 1

    assert(lm_count / ts_count == 2)

    new_ts_path_list = []

    for key in ts_path_dict:
        new_ts_path = {}

        for i in range(len(ts_path_dict[key])):
            # if the dict is the TS pt
            if float(ts_path_dict[key][i]['Freq 1']) < 0:
                new_ts_path['phi'] = ts_path_dict[key][i]['phi']
                new_ts_path['theta'] = ts_path_dict[key][i]['theta']
                new_ts_path['energy (A.U.)'] = ts_path_dict[key][i]['Energy (A.U.)']
                new_ts_path['G298 (Hartrees)'] = ts_path_dict[key][i]['G298 (Hartrees)']
                new_ts_path['Pucker'] = ts_path_dict[key][i]['Pucker']
            # else if it is a the forward lm
            elif 'ircf' in ts_path_dict[key][i]['File Name']:
                new_ts_path['phi_lm1'] = ts_path_dict[key][i]['phi']
                new_ts_path['theta_lm1'] = ts_path_dict[key][i]['theta']
            # else it is the reverse lm
            else:
                new_ts_path['phi_lm2'] = ts_path_dict[key][i]['phi']
                new_ts_path['theta_lm2'] = ts_path_dict[key][i]['theta']

        new_ts_path_list.append(new_ts_path)

    ts_paths_dict = {}

    for i in range(len(new_ts_path_list)):
        for key in new_ts_path_list[i]:
            if key not in ts_paths_dict:
                ts_paths_dict[key] = []

            ts_paths_dict[key].append(new_ts_path_list[i][key])

    full_filename = os.path.join(dir, filename)

    with open(full_filename, 'w', newline='') as file:
        w = csv.writer(file)
        w.writerow(ts_paths_dict.keys())
        w.writerows(zip(*ts_paths_dict.values()))

    return

def check_lm_running(comp_lm_dir, lm_data_dir, molecule):
    count = 0

    for d in os.listdir(comp_lm_dir):
        if os.path.isdir(os.path.join(comp_lm_dir, d)):
            count += 1

    if (count - 3!= len(os.listdir(lm_data_dir))):
        print('Warning: not all methods may have run for ' + molecule + '! (lm)')
        print(comp_lm_dir + ' should contain the following directories:')
        for l in range(len(os.listdir(lm_data_dir))):
            print(os.listdir(lm_data_dir)[l].split('-')[3].split('.')[0])
        print('final_comp')
        print('overall')
        print('csv_data')

def check_ts_running(ts_working_dir, ts_data_dir, molecule):
    if (len(os.listdir(ts_working_dir)) - 3 != len(os.listdir(ts_data_dir))):
        print('Warning: not all methods may have run for ' + molecule + '! (ts)')
        print(ts_working_dir + ' should contain the following directories:')
        for l in range(len(os.listdir(ts_data_dir))):
            print(os.listdir(ts_data_dir)[l].split('-')[3].split('.')[0])
        print('final_comp')
        print('all_groupings')
        print('heatmaps')

def save_comp_all_met_data(comp_all_met):
    comp_all_met.write_uncompared_to_csv()

    comp_all_met.write_ts_to_csv('arc')
    comp_all_met.write_ts_to_csv('gibbs')

    comp_all_met.save_diff_trend_by_met()
    comp_all_met.save_diff_trend_by_path()

    comp_all_met.write_num_comp_paths_to_csv('arc', 'added')
    comp_all_met.write_num_comp_paths_to_csv('gibbs', 'added')

    comp_all_met.save_comp_table('arc_comp', 'arc')
    comp_all_met.save_comp_table('arc_group_WRMSD', 'arc')

    comp_all_met.save_comp_table('gibbs_comp', 'gibbs')
    comp_all_met.save_comp_table('gibbs_group_WRMSD', 'gibbs')

    comp_all_met.save_all_comp_table('arc_comp', 'arc')
    comp_all_met.save_all_comp_table('arc_group_WRMSD', 'arc')

    comp_all_met.save_all_comp_table('gibbs_comp', 'gibbs')
    comp_all_met.save_all_comp_table('gibbs_group_WRMSD', 'gibbs')

def save_lm_comp_class_data(lm_comp_class, overwrite):
    lm_comp_class.save_all_figures(overwrite)
    lm_comp_class.save_all_groupings(overwrite)
    lm_comp_class.save_all_figures_raw(overwrite)

    lm_comp_class.save_WRMSD_heatmap(overwrite)
    lm_comp_class.save_RMSD_heatmap(overwrite)
    lm_comp_class.save_WRMSD_comp(overwrite)

    lm_comp_class.save_gibbs_WRMSD_heatmap(overwrite)
    lm_comp_class.save_gibbs_RMSD_heatmap(overwrite)
    lm_comp_class.save_gibbs_WRMSD_comp(overwrite)

def save_ts_comp_class_data(ts_comp_class, overwrite, write_individual):
    ts_comp_class.save_all_groups_comp(overwrite)

    if write_individual:
        ts_comp_class.save_group_comp(overwrite)

        ts_comp_class.save_all_figures_raw(overwrite)
        ts_comp_class.save_all_figures_single(overwrite)
        ts_comp_class.save_all_groupings(overwrite)

        ts_comp_class.save_WRMSD_comp(overwrite)
        ts_comp_class.save_WRMSD_heatmap(overwrite)
        ts_comp_class.save_RMSD_heatmap(overwrite)

        ts_comp_class.save_gibbs_WRMSD_comp(overwrite)
        ts_comp_class.save_gibbs_WRMSD_heatmap(overwrite)
        ts_comp_class.save_gibbs_RMSD_heatmap(overwrite)
#endregion

# # # Main # # #
#region
def main():
    # # # save init # # #
    #region
    save = True
    # overwrite existing plots, True is resource intensive
    overwrite = False

    # write the info for lm and/or ts
    write_lm = True
    write_ts = True

    write_individual = False

    # run calcs for specific molecule
    do_aglc = 0
    do_bglc = 0
    do_bxyl = 1
    do_oxane = 1

    do_molecule = [do_aglc, do_bglc, do_bxyl, do_oxane]

    debug = False # has those CSV files
    #endregion

    # # # init stuff # # #
    #region
    sv_all_mol_dir = os.path.join(SV_DIR, 'molecules')
    mol_list_dir = os.listdir(sv_all_mol_dir)

    num_clusters = [15, 13, 9, 8]
    #endregion

    # for each molecule, perform the comparisons
    for i in range(len(mol_list_dir)):
        if do_molecule[i]:
            # # # calcs # # #
            #region
            molecule = mol_list_dir[i]

            # # # directory init # # #
            #region
            # checks if directory exists, and creates it if not
            if not os.path.exists(os.path.join(MET_COMP_DIR, mol_list_dir[i])):
                os.makedirs(os.path.join(MET_COMP_DIR, mol_list_dir[i]))

            comp_mol_dir = os.path.join(MET_COMP_DIR, mol_list_dir[i])

            sv_mol_dir = os.path.join(sv_all_mol_dir, mol_list_dir[i])
            #endregion

            # # # local minimum directory init # # #
            #region
            # checks if directory exists, and creates it if not
            if not os.path.exists(os.path.join(comp_mol_dir, 'local_minimum')):
                os.makedirs(os.path.join(comp_mol_dir, 'local_minimum'))

            comp_lm_dir = os.path.join(comp_mol_dir, 'local_minimum')

            # checks if directory exists, and creates it if not
            if not os.path.exists(os.path.join(sv_mol_dir, 'z_datasets-LM')):
                os.makedirs(os.path.join(sv_mol_dir, 'z_datasets-LM'))

            lm_data_dir = os.path.join(sv_mol_dir, 'z_datasets-LM')
            #endregion

            # # # transition states directory init # # #
            #region
            # checks if directory exists, and creates it if not
            if not os.path.exists(os.path.join(comp_mol_dir, 'transitions_state')):
                os.makedirs(os.path.join(comp_mol_dir, 'transitions_state'))

            comp_ts_dir = os.path.join(comp_mol_dir, 'transitions_state')

            # checks if directory exists, and creates it if not
            if not os.path.exists(os.path.join(sv_mol_dir, 'z_datasets-TS')):
                os.makedirs(os.path.join(sv_mol_dir, 'z_datasets-TS'))

            ts_data_dir = os.path.join(sv_mol_dir, 'z_datasets-TS')

            # checks if directory exists, and creates it if not
            if not os.path.exists(os.path.join(sv_mol_dir, 'TS-unformatted')):
                os.makedirs(os.path.join(sv_mol_dir, 'TS-unformatted'))

            ts_unformatted_dir = os.path.join(sv_mol_dir, 'TS-unformatted')
            #endregion

            # # # comparison data initialization # # #
            #region
            lm_comp_data_list = []
            ts_comp_data_list = []

            # initialization info for local minimum clustering for specific molecule
            number_clusters = num_clusters[i]
            dict_cano = read_csv_canonical_designations('CP_params.csv', SV_DIR)
            data_points, phi_raw, theta_raw, energy = read_csv_data('z_' + mol_list_dir[i] + '_lm-b3lyp_howsugarspucker.csv',
                                                                    sv_mol_dir)
            lm_class = Local_Minima(number_clusters, data_points, dict_cano, phi_raw, theta_raw, energy)

            ts_data_dict = read_csv_data_TS('z_' + mol_list_dir[i] + '_TS-b3lyp_howsugarspucker.csv',
                                                                    sv_mol_dir)[3]

            hsp_file = os.path.join(sv_mol_dir, 'z_' + mol_list_dir[i] + '_TS-b3lyp_howsugarspucker.csv')
            hsp_added_kmeans_file = os.path.join(sv_mol_dir, 'z_' + mol_list_dir[i] + '_TS-b3lyp_hsp_added_kmeans.csv')

            ts_class = Transition_States(ts_data_dict, lm_class)
            #endregion

            # # # local minimum comparison data initialization # # #
            #region
            # for every local min data file in the directory perform the comparison calculations
            for filename in os.listdir(lm_data_dir):
                if filename.endswith(".csv"):
                    lm_hartree = read_csv_to_dict(os.path.join(lm_data_dir, filename), mode='r')
                    method = (filename.split('-', 3)[3]).split('.')[0]

                    ref_lm_hartree = read_csv_to_dict(os.path.join(lm_data_dir, 'z_dataset-' + molecule + '-LM-reference.csv'), mode='r')

                    ref_lm_comp_class = Local_Minima_Compare(molecule, 'reference', ref_lm_hartree, lm_class, comp_lm_dir)
                    lm_comp_class = Local_Minima_Compare(molecule, method, lm_hartree, lm_class, comp_lm_dir, ref_lm_comp_class)

                    if save and write_lm:
                        save_lm_comp_class_data(lm_comp_class, overwrite)

                    lm_comp_data_list.append(lm_comp_class)
            #endregion

            # # # transition state comparison data initialization # # #
            #region
            copyfile(hsp_file, hsp_added_kmeans_file)

            # for every ts data file in the directory reformat
            for filename in os.listdir(ts_unformatted_dir):
                if filename.endswith(".csv"):
                    ts_hartree = read_csv_to_dict(os.path.join(ts_unformatted_dir, filename), mode='r')
                    method = (filename.split('-', 3)[3]).split('.')[0]
                    rewrite_ts_hartree(ts_hartree, method, molecule, ts_data_dir)

            # for every ts data file in the directory perform the comparison calculations
            for filename in os.listdir(ts_data_dir):
                if filename.endswith(".csv"):
                    ts_hartree = read_csv_to_dict(os.path.join(ts_data_dir, filename), mode='r')
                    method = (filename.split('-', 3)[3]).split('.')[0]

                    ref_ts_hartree = read_csv_to_dict(os.path.join(ts_data_dir, 'z_dataset-' + molecule + '-TS-reference.csv'), mode='r')

                    ref_ts_comp_class = Transition_State_Compare(molecule, 'reference', ref_ts_hartree, lm_class,
                                                                 ts_class, comp_ts_dir)

                    ts_comp_class = Transition_State_Compare(molecule, method, ts_hartree, lm_class,
                                                             ts_class, comp_ts_dir, ref_ts_comp_class)

                    if save and write_ts:
                        save_ts_comp_class_data(ts_comp_class, overwrite, write_individual)

                    ts_comp_data_list.append(ts_comp_class)
            #endregion

            comp_all_met = Compare_All_Methods(ts_comp_data_list, comp_ts_dir, lm_comp_data_list, comp_lm_dir)

            comp_all_met.add_to_csv(hsp_added_kmeans_file)
            #endregion

            if debug:
                comp_all_met.write_debug_lm_to_csv()
                comp_all_met.write_debug_ts_to_csv()

            if save:
                # save the comparison data
                if write_lm:
                    comp_all_met.write_lm_to_csv()

                    comp_all_met.write_num_comp_lm_to_csv()
                    comp_all_met.write_gibbs_num_comp_lm_to_csv()

                if write_ts:
                    save_comp_all_met_data(comp_all_met)

            check_lm_running(comp_lm_dir, lm_data_dir, molecule)
            check_ts_running(ts_comp_class.plot_save_dir, ts_data_dir, molecule)

    if (len(os.listdir(MET_COMP_DIR)) != len(mol_list_dir)):
        print('Warning: the seed molecule directory and the populated molecule directory are not the same size!')
        print('This could be due to stray files/folders in either directory.')
        print('It could also be due to certain molecules not running.')
        print('The following molecules should have run:')
        for i in range(len(mol_list_dir)):
            print(i)

    # for each molecule, perform the comparisons
    for i in range(len(mol_list_dir)):
        if do_molecule[i]:
            molecule = mol_list_dir[i]
            comp_mol_dir = os.path.join(MET_COMP_DIR, mol_list_dir[i])
            sv_mol_dir = os.path.join(sv_all_mol_dir, mol_list_dir[i])
            ts_data_dir = os.path.join(sv_mol_dir, 'z_datasets-TS')

            # checks if directory exists, and creates it if not
            if not os.path.exists(os.path.join(comp_mol_dir, 'transitions_state_added')):
                os.makedirs(os.path.join(comp_mol_dir, 'transitions_state_added'))

            comp_ts_dir = os.path.join(comp_mol_dir, 'transitions_state_added')
            # # # calcs # # #
            # region
            # # # comparison data initialization # # #
            # region
            ts_comp_data_list = []

            # initialization info for local minimum clustering for specific molecule
            number_clusters = num_clusters[i]
            dict_cano = read_csv_canonical_designations('CP_params.csv', SV_DIR)
            data_points, phi_raw, theta_raw, energy = read_csv_data('z_' + mol_list_dir[i] + '_lm-b3lyp_howsugarspucker.csv',
                                                                    sv_mol_dir)
            lm_class = Local_Minima(number_clusters, data_points, dict_cano, phi_raw, theta_raw, energy)

            ts_data_dict = read_csv_data_TS('z_' + mol_list_dir[i] + '_TS-b3lyp_hsp_added_kmeans.csv',
                                            sv_mol_dir)[3]
            ts_class = Transition_States(ts_data_dict, lm_class)
            # endregion

            # # # transition state comparison data initialization # # #
            # region
            # for every ts data file in the directory perform the comparison calculations
            for filename in os.listdir(ts_data_dir):
                if filename.endswith(".csv"):
                    ts_hartree = read_csv_to_dict(os.path.join(ts_data_dir, filename), mode='r')
                    method = (filename.split('-', 3)[3]).split('.')[0]

                    ref_ts_hartree = read_csv_to_dict(
                        os.path.join(ts_data_dir, 'z_dataset-' + molecule + '-TS-reference.csv'), mode='r')
                    ref_ts_comp_class = Transition_State_Compare(molecule, method, ref_ts_hartree, lm_class,
                                                                 ts_class, comp_ts_dir)

                    added_ref_ts_hartree = read_csv_to_dict(
                        os.path.join(ts_data_dir, 'z_dataset-' + molecule + '-TS-addedref.csv'), mode='r')
                    added_ref_ts_comp_class = Transition_State_Compare(molecule, method, added_ref_ts_hartree, lm_class,
                                                                 ts_class, comp_ts_dir)

                    ts_comp_class = Transition_State_Compare(molecule, method, ts_hartree, lm_class,
                                                             ts_class, comp_ts_dir, ref_ts_comp_class, added_ref_ts_comp_class)

                    if save and write_ts:
                        save_ts_comp_class_data(ts_comp_class, overwrite, write_individual)

                    ts_comp_data_list.append(ts_comp_class)
            # endregion

            comp_all_met = Compare_All_Methods(ts_comp_data_list, comp_ts_dir)
            # endregion

            if debug:
                comp_all_met.write_debug_ts_to_csv()

            if save:
                # save the comparison data
                if write_ts:
                    save_comp_all_met_data(comp_all_met)

            check_ts_running(ts_comp_class.plot_save_dir, ts_data_dir, molecule)

    if (len(os.listdir(MET_COMP_DIR)) != len(mol_list_dir)):
        print('Warning: the seed molecule directory and the populated molecule directory are not the same size!')
        print('This could be due to stray files/folders in either directory.')
        print('It could also be due to certain molecules not running.')
        print('The following molecules should have run:')
        for i in range(len(mol_list_dir)):
            print(mol_list_dir(i))

    return 0

if __name__ == '__main__':
    status = main()
    sys.exit(status)
#endregion
