#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, create_out_fname, \
    write_csv, list_to_dict, read_csv_to_dict
from qm_utils.xyz_cluster import main, hartree_sum_pucker_cluster, compare_rmsd_xyz, test_clusters, \
    check_ring_ordering, read_ring_atom_ids, check_before_after_sorting

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'xyz_cluster')
TS_DATA_DIR = os.path.join(SUB_DATA_DIR, 'TS_data')
PM3MM_DATA_DIR = os.path.join(SUB_DATA_DIR,'bxyl_data')

# Input files #

OXANE_HARTREE_SUM_HEATHER_FILE = os.path.join(SUB_DATA_DIR, 'b3lyp-test.csv')
OXANE_1c4_INPUT_FILE = os.path.join(SUB_DATA_DIR, "oxane-1c4-freeze_b3lyp-optall_b3lyp.log")
OXANE_1e_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-1e-freeze_b3lyp-optall_b3lyp.log')
OXANE_4c1_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-4c1-freeze_b3lyp-optall_b3lyp.log')
OXANE_HARTREE_CLUSTER_FILE = os.path.join(SUB_DATA_DIR, 'xyz_cluster-sampleout.txt')
OXANE_HARTREE_SUM_B3LYP_FILE = os.path.join(SUB_DATA_DIR, 'z_hartree_out-unsorted-oxane-b3lyp.csv')
OXANE_HARTREE_DICT_CLUSTER_FILE = os.path.join(SUB_DATA_DIR, 'oxane_hartree_sum_b3lyp_hartree_dict_good.csv')
OXANE_XYZ_COORDS_WRITE_FILE_1s3 = os.path.join(SUB_DATA_DIR, 'xyz_oxane-1s3-freeze_b3lyp-optall_b3lyp-xyz_updated.xyz')
OXANE_XYZ_COORDS_WRITE_FILE_3s1 = os.path.join(SUB_DATA_DIR, 'xyz_oxane-3s1-freeze_b3lyp-optall_b3lyp-xyz_updated.xyz')
OXANE_XYZ_COORDS_WRITE_FILE_5e = os.path.join(SUB_DATA_DIR, 'xyz_oxane-5e-freeze_b3lyp-optall_b3lyp-xyz_updated.xyz')
OXANE_XYZ_COORDS_WRITE_FILE_25b = os.path.join(SUB_DATA_DIR, 'xyz_oxane-25b-freeze_b3lyp-optall_b3lyp-xyz_updated.xyz')
OXANE_XYZ_COORDS_WRITE_FILE_b25 = os.path.join(SUB_DATA_DIR, 'xyz_oxane-b25-freeze_b3lyp-optall_b3lyp-xyz_updated.xyz')
OXANE_XYZ_COORDS_WRITE_FILE_e5 = os.path.join(SUB_DATA_DIR, 'xyz_oxane-e5-freeze_b3lyp-optall_b3lyp-xyz_updated.xyz')
OXANE_XYZ_COORDS_WRITE_FILE_1s5 = os.path.join(SUB_DATA_DIR, 'xyz_oxane-1s5-freeze_b3lyp-optall_b3lyp-xyz_updated.xyz')
OXANE_XYZ_COORDS_WRITE_FILE_5s1 = os.path.join(SUB_DATA_DIR, 'xyz_oxane-5s1-freeze_b3lyp-optall_b3lyp-xyz_updated.xyz')
GLUCOSE_XYZ_COORDS_HEATHER_03b_1 = os.path.join(SUB_DATA_DIR, 'bglc_03b_1.log.xyz')
GLUCOSE_XYZ_COORDS_HEATHER_03b_2 = os.path.join(SUB_DATA_DIR, 'bglc_03b_2.log.xyz')
FILE_1s3_TO_1s3 = os.path.join(SUB_DATA_DIR, 'oxane-1s3-freeze_b3lyp-optall_b3lyp.log')
FILE_1s5_TO_1s5 = os.path.join(SUB_DATA_DIR, 'oxane-1s5-freeze_b3lyp-optall_b3lyp.log')
FILE_3s1_TO_3s1 = os.path.join(SUB_DATA_DIR, 'oxane-3s1-freeze_b3lyp-optall_b3lyp.log')
FILE_5e_TO_1c4 = os.path.join(SUB_DATA_DIR, 'oxane-5e-freeze_b3lyp-optall_b3lyp.log')
FILE_5s1_TO_5s1 = os.path.join(SUB_DATA_DIR, 'oxane-5s1-freeze_b3lyp-optall_b3lyp.log')
FILE_25b_TO_2so = os.path.join(SUB_DATA_DIR, 'oxane-25b-freeze_b3lyp-optall_b3lyp.log')
FILE_b25_TO_os2 = os.path.join(SUB_DATA_DIR, 'oxane-b25-freeze_b3lyp-optall_b3lyp.log')
FILE_e5_TO_4c1 = os.path.join(SUB_DATA_DIR, 'oxane-e5-freeze_b3lyp-optall_b3lyp.log')
FILE_NEW_PUCK_LIST = os.path.join(SUB_DATA_DIR, 'z_files_list_new_puck_b3lyp_hartree_sum-cpsnap.txt')
ATOMS_RING_ORDER2 = ['1', '6', '6', '6', '6', '6', '8', '1', '1', '1', '1', '1', '1', '1', '1', '1']
ATOMS_RING_ORDER1 = ['6', '6', '6', '6', '6', '8', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
ATOMS_RING_ORDER3 = ['6', '6', '6', '6', '6', '8', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
OXANE_RING_ATOM_ORDER = '5,0,1,2,3,4'

OUT_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster_z_hartree_out-unsorted-oxane-b3lyp.csv')
OUT_FILE_LIST = os.path.join(SUB_DATA_DIR, 'z_files_list_freq_runsb3lyp_hartree_sum-cpsnap.txt')
GOOD_OUT_FILE_LIST = os.path.join(SUB_DATA_DIR, 'z_files_list_freq_runsb3lyp_hartree_sum-cpsnap_good.txt')
GOOD_OUT_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster_sorted-oxane-b3lyp.csv')
GOOD_OUT_FILE2 = os.path.join(SUB_DATA_DIR, 'z_cluster_sorted-oxane-b3lyp.csv')
BAD_OUT_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster_b3lyp_hartree_sum-cpsnap_bad.csv')
GOOD_OXANE_XYZ_COORDS_WRITE_FILE_1s3 = os.path.join(SUB_DATA_DIR,
                                                    'xyz_oxane-1s3-freeze_b3lyp-optall_b3lyp-xyz_updated_good.xyz')
GOOD_OXANE_XYZ_COORDS_WRITE_FILE_3s1 = os.path.join(SUB_DATA_DIR,
                                                    'xyz_oxane-3s1-freeze_b3lyp-optall_b3lyp-xyz_updated_good.xyz')
GOOD_OXANE_XYZ_COORDS_WRITE_FILE_5e = os.path.join(SUB_DATA_DIR,
                                                   'xyz_oxane-5e-freeze_b3lyp-optall_b3lyp-xyz_updated_good.xyz')
GOOD_OXANE_XYZ_COORDS_WRITE_FILE_25b = os.path.join(SUB_DATA_DIR,
                                                    'xyz_oxane-25b-freeze_b3lyp-optall_b3lyp-xyz_updated_good.xyz')
GOOD_OXANE_XYZ_COORDS_WRITE_FILE_b25 = os.path.join(SUB_DATA_DIR,
                                                    'xyz_oxane-b25-freeze_b3lyp-optall_b3lyp-xyz_updated_good.xyz')
GOOD_OXANE_XYZ_COORDS_WRITE_FILE_e5 = os.path.join(SUB_DATA_DIR,
                                                   'xyz_oxane-e5-freeze_b3lyp-optall_b3lyp-xyz_updated_good.xyz')
GOOD_OXANE_XYZ_COORDS_WRITE_FILE_1s5 = os.path.join(SUB_DATA_DIR,
                                                    'xyz_oxane-1s5-freeze_b3lyp-optall_b3lyp-xyz_updated_good.xyz')
GOOD_OXANE_XYZ_COORDS_WRITE_FILE_5s1 = os.path.join(SUB_DATA_DIR,
                                                    'xyz_oxane-5s1-freeze_b3lyp-optall_b3lyp-xyz_updated_good.xyz')
GOOD_1s3_TO_1s3 = os.path.join(SUB_DATA_DIR, 'oxane-1s3-freeze_b3lyp-optall_b3lyp.log')
GOOD_1s5_TO_1s5 = os.path.join(SUB_DATA_DIR, 'oxane-1s5-freeze_b3lyp-optall_b3lyp.log')
GOOD_3s1_TO_3s1 = os.path.join(SUB_DATA_DIR, 'oxane-3s1-freeze_b3lyp-optall_b3lyp.log')
GOOD_5e_TO_1c4 = os.path.join(SUB_DATA_DIR, 'oxane-5e-freeze_b3lyp-optall_b3lyp.log')
GOOD_5s1_TO_5s1 = os.path.join(SUB_DATA_DIR, 'oxane-5s1-freeze_b3lyp-optall_b3lyp.log')
GOOD_25b_TO_2so = os.path.join(SUB_DATA_DIR, 'oxane-25b-freeze_b3lyp-optall_b3lyp.log')
GOOD_b25_TO_os2 = os.path.join(SUB_DATA_DIR, 'oxane-b25-freeze_b3lyp-optall_b3lyp.log')
GOOD_e5_TO_4c1 = os.path.join(SUB_DATA_DIR, 'oxane-e5-freeze_b3lyp-optall_b3lyp.log')
GOOD_NEW_PUCKER_LIST = os.path.join(SUB_DATA_DIR, 'z_files_list_new_puck_b3lyp_hartree_sum-cpsnap_good.txt')


OXANE_HARTREE_SUM_TS_B3LYP_FILE = os.path.join(TS_DATA_DIR, 'z_hartree_out-unsorted-oxane-b3lyp.csv')
BXYL_HARTREE_PM3MM_TS_FILE = os.path.join(PM3MM_DATA_DIR, 'z_hartree-unsorted-TS-pm3mm.csv')

# Good output
XYZ_TOL = 0.1
RMSD_KABSCH_SIMILAR_GOOD = (0.000300503125935, 3, 6)
RMSD_KABSCH_SIMILAR_1c4to4c1 = (0.45429783853700906, 3, 6)
PUCK_2SO = '2so'
PUCK_2SO_FILES = ['25Bm062xconstb3lypbigb3lrelm062x.log', 'E4m062xconstb3lypbigcon1b3ltsm062x.log']
TEST_XYZ_TOL = '0.000001'
FILE_NAME = 'File Name'
CLUSTER_DICT_NUM_PUCKER_GROUPS = 8
TOTAL_NUM_OXANE_CLUSTER = 38


class TestFailWell(unittest.TestCase):
    def testHelp(self):
        test_input = ['-h']
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertEquals(output,'WARNING:  0\n')
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)

    def testNoSuchFile(self):
        test_input = ["-s", "ghost"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue("Could not find" in output)


# noinspection PyUnboundLocalVariable
class TestXYZFunctions(unittest.TestCase):
    def testAtomRingOrderingWrong(self):
        status = check_ring_ordering(ATOMS_RING_ORDER1, ATOMS_RING_ORDER2)
        # self.assertEqual(status, 0)
        self.assertEqual(status, 1)

    def testAtomRingOrderingCorrect(self):
        status = check_ring_ordering(ATOMS_RING_ORDER1, ATOMS_RING_ORDER3)
        self.assertEqual(status, 0)

    def testReadHartreeSummary(self):
        hartree_dict, pucker_filename_dict, \
        hartree_headers = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_HEATHER_FILE)
        self.assertTrue(pucker_filename_dict[PUCK_2SO], PUCK_2SO_FILES)

    def testTwoFiles_similar(self):
        atoms_order = read_ring_atom_ids(OXANE_RING_ATOM_ORDER)
        rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar, \
        atom_order = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE, OXANE_1e_INPUT_FILE,
                                      SUB_DATA_DIR, atoms_order)
        self.assertTrue(abs(rmsd_kabsch - RMSD_KABSCH_SIMILAR_GOOD[0]) < XYZ_TOL)

    def testTwoFiles_1c4to4c1(self):
        atoms_order = read_ring_atom_ids(OXANE_RING_ATOM_ORDER)
        rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar, \
        atom_order = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE, OXANE_4c1_INPUT_FILE,
                                      SUB_DATA_DIR, atoms_order)
        self.assertTrue(abs(rmsd_kabsch - RMSD_KABSCH_SIMILAR_1c4to4c1[0]) < 0.5)

    def testTwoFiles_PrintFeature(self):
        atoms_order = read_ring_atom_ids(OXANE_RING_ATOM_ORDER)
        with capture_stdout(compare_rmsd_xyz, OXANE_1c4_INPUT_FILE, OXANE_1e_INPUT_FILE, SUB_DATA_DIR, atoms_order,
                            print_option='on') as output:
            self.assertTrue("Rmsd" in output)
            self.assertTrue(len(output) > 100)

    def testHartreeSumPuckerCluster(self):
        try:
            hartree_list, pucker_filename_dict, hartree_headers \
                = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
            out_f_name = create_out_fname(SUB_DATA_DIR, prefix='hartree_list_', suffix='_output',
                                          ext='.csv')
            write_csv(hartree_list, out_f_name, hartree_headers, extrasaction="ignore")
            self.assertFalse(diff_lines(out_f_name, OXANE_HARTREE_SUM_B3LYP_FILE))
        finally:
            silent_remove(out_f_name)

    def testListToDict(self):
        hartree_list, pucker_filename_dict, hartree_headers \
            = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
        hartree_dict = list_to_dict(hartree_list, FILE_NAME)
        len(hartree_dict)
        if len(hartree_dict) == len(hartree_list):
            pass

    def testTestClustersLowTol(self):
        low_tol = 0.00001
        atoms_order = read_ring_atom_ids(OXANE_RING_ATOM_ORDER)
        hartree_list, pucker_filename_dict, hartree_headers \
            = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
        hartree_dict = list_to_dict(hartree_list, FILE_NAME)
        process_cluster_dict, xyz_coords_dict, atom_order \
            = test_clusters(pucker_filename_dict, hartree_dict, SUB_DATA_DIR, low_tol, atoms_order, print_option='off')
        self.assertEquals(len(process_cluster_dict), len(hartree_dict))

    def testTestClustersHighTol(self):
        high_tol = 100
        atoms_order = read_ring_atom_ids(OXANE_RING_ATOM_ORDER)
        hartree_list, pucker_filename_dict, hartree_headers \
            = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
        hartree_dict = list_to_dict(hartree_list, FILE_NAME)
        process_cluster_dict, xyz_coords_dict, atom_order \
            = test_clusters(pucker_filename_dict, hartree_dict, SUB_DATA_DIR, high_tol, atoms_order, print_option='off')

        self.assertEqual(CLUSTER_DICT_NUM_PUCKER_GROUPS, len(process_cluster_dict))
        self.assertEqual(TOTAL_NUM_OXANE_CLUSTER, len(hartree_dict))

    def testCheckBeforeAfterSortingGood(self):
        try:
            file_unsorted = OXANE_HARTREE_SUM_B3LYP_FILE
            file_sorted = GOOD_OUT_FILE
            list_pucker_missing = check_before_after_sorting(file_unsorted, file_sorted)
        finally:
            self.assertEquals(list_pucker_missing, [])

    def testCheckBeforeAfterSortingBad(self):
        try:
            file_unsorted = OXANE_HARTREE_SUM_B3LYP_FILE
            file_sorted = BAD_OUT_FILE
            list_pucker_missing = check_before_after_sorting(file_unsorted, file_sorted)
        finally:
            self.assertEquals(list_pucker_missing, ['4c1'])


class TestMain(unittest.TestCase):
    def testMain(self):
        try:
            test_input = ["-s", OXANE_HARTREE_SUM_B3LYP_FILE, "-t", '0.1']
            main(test_input)
            hartree_dict_test = read_csv_to_dict(OUT_FILE, mode='rU')
            hartree_dict_good = read_csv_to_dict(GOOD_OUT_FILE2, mode='rU')

            for row1 in hartree_dict_test:
                for row2 in hartree_dict_good:
                    if row1[FILE_NAME] == row2[FILE_NAME]:
                        self.assertEqual(row1, row2)
        finally:
            silent_remove(OUT_FILE)
            silent_remove(OUT_FILE_LIST)
            silent_remove(FILE_NEW_PUCK_LIST)

    # def testMainPrintXYZCoords(self):
    #     try:
    #         test_input = ["-s", OXANE_HARTREE_SUM_B3LYP_FI LE, "-t", '0.1', "-p", "true"]
    #         main(test_input)
    #         self.assertFalse(diff_lines(OXANE_XYZ_COORDS_WRITE_FILE_1s3, GOOD_OXANE_XYZ_COORDS_WRITE_FILE_1s3))
    #         self.assertFalse(diff_lines(OXANE_XYZ_COORDS_WRITE_FILE_3s1, GOOD_OXANE_XYZ_COORDS_WRITE_FILE_3s1))
    #         self.assertFalse(diff_lines(OXANE_XYZ_COORDS_WRITE_FILE_5e, GOOD_OXANE_XYZ_COORDS_WRITE_FILE_5e))
    #         self.assertFalse(diff_lines(OXANE_XYZ_COORDS_WRITE_FILE_25b, GOOD_OXANE_XYZ_COORDS_WRITE_FILE_25b))
    #         self.assertFalse(diff_lines(OXANE_XYZ_COORDS_WRITE_FILE_b25, GOOD_OXANE_XYZ_COORDS_WRITE_FILE_b25))
    #         self.assertFalse(diff_lines(OXANE_XYZ_COORDS_WRITE_FILE_e5, GOOD_OXANE_XYZ_COORDS_WRITE_FILE_e5))
    #         self.assertFalse(diff_lines(OXANE_XYZ_COORDS_WRITE_FILE_1s5, GOOD_OXANE_XYZ_COORDS_WRITE_FILE_1s5))
    #         self.assertFalse(diff_lines(OXANE_XYZ_COORDS_WRITE_FILE_5s1, GOOD_OXANE_XYZ_COORDS_WRITE_FILE_5s1))
    #     finally:
    #         silent_remove(OXANE_XYZ_COORDS_WRITE_FILE_1s3)
    #         silent_remove(OXANE_XYZ_COORDS_WRITE_FILE_3s1)
    #         silent_remove(OXANE_XYZ_COORDS_WRITE_FILE_5e)
    #         silent_remove(OXANE_XYZ_COORDS_WRITE_FILE_25b)
    #         silent_remove(OXANE_XYZ_COORDS_WRITE_FILE_b25)
    #         silent_remove(OXANE_XYZ_COORDS_WRITE_FILE_e5)
    #         silent_remove(OXANE_XYZ_COORDS_WRITE_FILE_1s5)
    #         silent_remove(OXANE_XYZ_COORDS_WRITE_FILE_5s1)
    #         silent_remove(OUT_FILE)
    #         silent_remove(OUT_FILE_LIST)
    #         silent_remove(FILE_NEW_PUCK_LIST)

    def testMainNotPrintXYZCoords(self):
        try:
            silent_remove(OXANE_XYZ_COORDS_WRITE_FILE_1s3)
            test_input = ["-s", OXANE_HARTREE_SUM_B3LYP_FILE, "-t", '0.1']
            main(test_input)
            self.assertFalse(os.path.isfile(OXANE_XYZ_COORDS_WRITE_FILE_1s3))
        finally:
            silent_remove(OUT_FILE)
            silent_remove(FILE_NEW_PUCK_LIST)


    def testTranstionStateMainScript(self):
        test_input = ["-s", OXANE_HARTREE_SUM_TS_B3LYP_FILE, "-t", '0.1']
        main(test_input)
        with capture_stdout(main, test_input) as output:
            self.assertFalse("Warning! The following puckers have been dropped: ['eo']." in output)


    def testTranstionStateMainScript2(self):
        test_input = ["-s", OXANE_HARTREE_SUM_TS_B3LYP_FILE, "-t", '0.005']
        with capture_stdout(main, test_input) as output:
            self.assertFalse("Warning! The following puckers have been dropped:" in output)

    def testTransitionStateMain3(self):
        test_input = ["-s", BXYL_HARTREE_PM3MM_TS_FILE, "-t", '0.1', '-r', '7,4,16,12,8,0']
        main(test_input)
        # with capture_stdout(main, test_input) as output:
        #     self.assertFalse("Warning! The following puckers have been dropped:" in output)

