import unittest

import os
from qm_utils.comparison_classes import Compare_Methods
from qm_utils.comparison_classes import Structure

import matplotlib.pyplot as plt


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

class TestCompare_Methods(unittest.TestCase):
    def setUp(self):
        methods_list = []

        methods_list.append('REF')
        methods_list.append('B3LYP')
        methods_list.append('AM1')

        cmap = plt.get_cmap('Paired')
        met_colors_dict = {}

        met_colors_dict['REF'] = cmap.colors[1]
        met_colors_dict['B3LYP'] = cmap.colors[0]
        met_colors_dict['AM1'] = 'mediumvioletred'

        met_ts_markers_dict = {}
        met_lm_markers_dict = {}

        for method in list(methods_list):
            ts_marker = 'o'
            lm_marker = 'o'

            met_ts_markers_dict[method] = ts_marker
            met_lm_markers_dict[method] = lm_marker

        energy_format = 'H298 (Hartrees)'

        self.comp_met = Compare_Methods('testmol',
                                      met_colors_dict,
                                      met_ts_markers_dict,
                                      met_lm_markers_dict,
                                      energy_format)
        pass

    # tests dir_init and make_dir
    def test_dir_init(self):
        self.assertTrue(os.path.exists(os.path.join(SV_MOL_DIR, self.comp_met.molecule)))
        self.assertTrue(os.path.exists(os.path.join(COMP_CLASSES_DIR, self.comp_met.molecule)))
        self.assertTrue(os.path.exists(os.path.join(self.comp_met.MOL_SAVE_DIR,
                                                    self.comp_met.energy_format.split(' ')[0])))
        self.assertTrue(os.path.exists(os.path.join(self.comp_met.MOL_DATA_DIR, 'IRC')))
        self.assertTrue(os.path.exists(os.path.join(self.comp_met.MOL_DATA_DIR, 'LM')))
        self.assertTrue(os.path.exists(os.path.join(self.comp_met.MOL_DATA_DIR, 'TS')))
        self.assertTrue(os.path.exists(os.path.join(self.comp_met.MOL_DATA_DIR, 'IRC')))

    # tests calc_closest_LM_skm with arbitrary structure coords
    def test_calc_closest_LM_skm(self):
        structure = Structure(phi=100,theta=240)
        self.comp_met.calc_closest_LM_skm(structure)

        self.assertTrue(abs(structure.comp_metrics['arc'] - 0.212) < 0.001)
        self.assertTrue(abs(structure.comp_metrics['next_arc'] - 0.306) < 0.001)

        self.assertTrue(structure.closest_skm == 32)
        self.assertTrue(structure.next_closest_skm == 29)

    # tests calc_closest_TS_skm with arbitrary structure coords
    def test_calc_closest_TS_skm(self):
        structure = Structure(phi=100, theta=240)
        self.comp_met.calc_closest_TS_skm(structure)

        self.assertTrue(abs(structure.comp_metrics['arc'] - 0.212) < 0.001)
        self.assertTrue(abs(structure.comp_metrics['next_arc'] - 0.306) < 0.001)

        self.assertTrue(structure.closest_skm == 33)
        self.assertTrue(structure.next_closest_skm == 31)

    def test_assign_skm_labels(self):
        self.assertTrue(len(self.comp_met.Method_Pathways_dict['AM1'].Pathways) == 10)
        self.assertTrue(len(self.comp_met.Method_Pathways_dict['AM1'].IRC_csv_list) == 20)

    def test_are_equal(self):
        struct1 = Structure(0, 15, 10)
        struct2 = Structure(180, 15, 10)
        struct3 = Structure(0, 15, 15)

        struct4 = Structure(0, 165, 10)
        struct5 = Structure(180, 165, 10)

        # case where theta is small enough where phi doesn't matter
        self.assertTrue(self.comp_met.are_equal(struct1, struct2))
        # case where theta is large enough where phi doesn't matter
        self.assertTrue(self.comp_met.are_equal(struct4, struct5))
        # case where coords are similar enough, but energies aren't
        self.assertFalse(self.comp_met.are_equal(struct1, struct3))

    def test_get_ref_IRC_skms(self):
        skm_list = self.comp_met.get_ref_IRC_skms(Structure(180, 90))
        pass
        self.assertTrue(len(skm_list) == 1)

    def test_pathway_groupings_init(self):
        self.fail()

    def test_populate_pathway_groupings(self):
        self.fail()


    def test_do_path_calcs(self):
        self.fail()

    def test_do_boltz_calcs(self):
        self.fail()

    def test_do_init_calcs(self):
        self.fail()

    def test_calc_LM_weighting(self):
        self.fail()

    def test_calc_TS_weighting(self):
        self.fail()

    def test_calc_gibbs_diff(self):
        self.fail()

    def test_do_calcs(self):
        self.fail()

    def test_calc_weighting(self):
        self.fail()

    def test_calc_WSS(self):
        self.fail()

    def test_calc_group_RMSD(self):
        self.fail()

    def test_calc_SSE(self):
        self.fail()

    def test_calc_RMSD(self):
        self.fail()

    def test_calc_WWSS(self):
        self.fail()

    def test_calc_group_WRMSD(self):
        self.fail()

    def test_calc_WSSE(self):
        self.fail()

    def test_calc_WRMSD(self):
        self.fail()

    def test_calc_phys_SSE(self):
        self.fail()

    def test_calc_phys_RMSD(self):
        self.fail()

    def test_calc_phys_WSSE(self):
        self.fail()

    def test_calc_phys_WRMSD(self):
        self.fail()


    def test_pos_theta_conflicts(self):
        self.fail()

    def test_pos_r_conflicts(self):
        self.fail()

    def test_neg_theta_conflicts(self):
        self.fail()

    def test_neg_r_conflicts(self):
        self.fail()

    def test_get_label_theta_and_r(self):
        self.fail()


if __name__ == '__main__':
    unittest.main()
