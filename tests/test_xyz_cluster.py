#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_xyz_cluster
----------------------------------

"""
import csv
import unittest
import os
from qm_utils.qm_common import silent_remove, diff_lines, capture_stderr, capture_stdout, warning, create_out_fname, \
    write_csv, list_to_dict
from qm_utils.xyz_cluster import main, hartree_sum_pucker_cluster, compare_rmsd_xyz, test_clusters, dict_to_csv_writer, \
    read_clustered_keys_in_hartree
import logging



__author__ = 'SPVicchio'

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'xyz_cluster')

# Input files #

OXANE_HARTREE_SUM_HEATHER_FILE = os.path.join(SUB_DATA_DIR, 'b3lyp-test.csv')
OXANE_1c4_INPUT_FILE = os.path.join(SUB_DATA_DIR, "oxane-1c4-freeze_B3LYP-relax_B3LYP.log")
OXANE_1e_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-1e-freeze_B3LYP-relax_B3LYP.log')
OXANE_4c1_INPUT_FILE = os.path.join(SUB_DATA_DIR, 'oxane-4c1-freeze_B3LYP-relax_B3LYP.log')
OXANE_HARTREE_CLUSTER_FILE = os.path.join(SUB_DATA_DIR, 'xyz_cluster-sampleout.txt')
OXANE_HARTREE_SUM_B3LYP_FILE = os.path.join(SUB_DATA_DIR, 'B3LYP_hartree_sum-cpsnap.csv')
OXANE_HARTREE_DICT_CLUSTER_FILE = os.path.join(SUB_DATA_DIR, 'oxane_hartree_sum_B3LYP_hartree_dict_good.csv')


OUT_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster_B3LYP_hartree_sum-cpsnap.csv')
GOOD_OUT_FILE = os.path.join(SUB_DATA_DIR, 'z_cluster_B3LYP_hartree_sum-cpsnap_good.csv')

# Good output
XYZ_TOL = 1.0e-12
RMSD_KABSCH_SIMILAR_GOOD = (0.000300503125935, 3, 6)
RMSD_KABSCH_SIMILAR_1c4to4c1 = (0.45429783853700906, 3, 6)
PUCK_2SO = '2so'
PUCK_2SO_FILES = ['25Bm062xconstb3lypbigb3lrelm062x.log', 'E4m062xconstb3lypbigcon1b3ltsm062x.log']
TEST_XYZ_TOL = '0.000001'
FILE_NAME = 'File Name'



class TestFailWell(unittest.TestCase):
    def testHelp(self):
        test_input = ['-h']
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertFalse(output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)

    def testNoSuchFile(self):
        test_input = ["-s", "ghost"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue("Could not find" in output)


class TestXYZFunctions(unittest.TestCase):
    def testReadHartreeSummary(self):
        hartree_dict, pucker_filename_dict,\
            hartree_headers = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_HEATHER_FILE)
        self.assertTrue(pucker_filename_dict[PUCK_2SO], PUCK_2SO_FILES)

    def testTwoFiles_similar(self):
        rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE,
                                                                                 OXANE_1e_INPUT_FILE,
                                                                                 SUB_DATA_DIR)
        self.assertTrue(abs(rmsd_kabsch - RMSD_KABSCH_SIMILAR_GOOD[0]) < XYZ_TOL)

    def testTwoFiles_1c4to4c1(self):
        rmsd_kabsch, xyz_coords1_similar, xyz_coords2_similar = compare_rmsd_xyz(OXANE_1c4_INPUT_FILE,
                                                                                 OXANE_4c1_INPUT_FILE,
                                                                                 SUB_DATA_DIR)
        self.assertTrue(abs(rmsd_kabsch - RMSD_KABSCH_SIMILAR_1c4to4c1[0]) < XYZ_TOL)

    def testTwoFiles_PrintFeature(self):
        with capture_stdout(compare_rmsd_xyz, OXANE_1c4_INPUT_FILE, OXANE_1e_INPUT_FILE, SUB_DATA_DIR,
                                                                                        print_option='on') as output:
            self.assertTrue("Rmsd" in output)
            self.assertTrue(len(output) > 100)

    def testHartreeSumPuckerCluster(self):
        try:
            hartree_list, pucker_filename_dict, hartree_headers\
                = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
            out_f_name = create_out_fname(SUB_DATA_DIR, prefix='hartree_list_', suffix='_output',
                                                    ext='.csv')
            write_csv(hartree_list, out_f_name, hartree_headers, extrasaction="ignore")
            self.assertFalse(diff_lines(out_f_name,OXANE_HARTREE_SUM_B3LYP_FILE))
        finally:
            silent_remove(out_f_name)

    def testListToDict(self):
        hartree_list, pucker_filename_dict, hartree_headers \
              = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
        hartree_dict = list_to_dict(hartree_list,FILE_NAME)
        len(hartree_dict)
        if len(hartree_dict) == len(hartree_list):
            pass

    def testTestClustersNormalTol(self):
        try:
            hartree_list, pucker_filename_dict, hartree_headers \
                = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
            hartree_dict = list_to_dict(hartree_list,FILE_NAME)
            process_cluster_dict = test_clusters(pucker_filename_dict, SUB_DATA_DIR,0.1, print_option='off')

# TODO still need to write this...
            print(process_cluster_dict)

        finally:
            pass

    def testTestClustersLowTol(self):
        low_tol = 0.00001
        hartree_list, pucker_filename_dict, hartree_headers \
            = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
        hartree_dict = list_to_dict(hartree_list,FILE_NAME)
        process_cluster_dict = test_clusters(pucker_filename_dict, SUB_DATA_DIR,low_tol, print_option='off')

        self.assertEquals()

        if len(process_cluster_dict)-len(hartree_list) == 0:
            pass

    def testTestClustersHighTol(self):
        high_tol = 100
        hartree_list, pucker_filename_dict, hartree_headers \
            = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
        hartree_dict = list_to_dict(hartree_list,FILE_NAME)
        process_cluster_dict = test_clusters(pucker_filename_dict, SUB_DATA_DIR,high_tol, print_option='off')


        out_f_name = create_out_fname(SUB_DATA_DIR, prefix='z_test_clusters_high_tol', suffix='_output_good',
                                      ext='.csv')

        write_csv(process_cluster_dict.keys(),out_f_name,'NEEDED',extrasaction="ignore")


    def testReadClusters(self):
        pass

    def testMainWrongWay(self):
        try:
            hartree_list, pucker_filename_dict, hartree_headers \
                = hartree_sum_pucker_cluster(OXANE_HARTREE_SUM_B3LYP_FILE)
            hartree_dict = list_to_dict(hartree_list,FILE_NAME)
            process_cluster_dict = test_clusters(pucker_filename_dict, SUB_DATA_DIR,0.1, print_option='off')
            filtered_clustered_list = read_clustered_keys_in_hartree(process_cluster_dict, hartree_dict)
            out_f_name = create_out_fname(OUT_FILE, ext='.csv')
            write_csv(filtered_clustered_list, out_f_name, hartree_headers, extrasaction="ignore")
            self.assertFalse(diff_lines(out_f_name, GOOD_OUT_FILE))
        finally:
            silent_remove(out_f_name)


class TestMain(unittest.TestCase):
    def testMain(self):
        try:
            test_input = ["-s", OXANE_HARTREE_SUM_B3LYP_FILE, "-t", '0.1']
            main(test_input)
            self.assertFalse(diff_lines(GOOD_OUT_FILE,OUT_FILE))
        finally:
            silent_remove(OUT_FILE)

    def testMainPrintXYZCoords(self):
        try:
            test_input = ["-s", OXANE_HARTREE_SUM_B3LYP_FILE, "-t", '0.1',"-p"]
            main(test_input)

        finally:
            silent_remove(OUT_FILE)
            pass
