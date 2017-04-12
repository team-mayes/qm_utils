#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_read_sdf
----------------------------------

Tests for `read_sdf` module.
"""

import unittest
import os
import math
from qm_utils.qm_common import (list_to_file, diff_lines, silent_remove, write_csv, capture_stdout, create_out_fname,
                                process_cfg, InvalidDataError, dequote, capture_stderr, arc_length_calculator)


__author__ = 'hmayes'

# Directories #

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'common')

# OUTPUT files #

LIST_OUT = os.path.join(SUB_DATA_DIR, 'list.txt')
LIST_OUT_GOOD = os.path.join(SUB_DATA_DIR, 'list_good.txt')
LIST_APPENDED_GOOD = os.path.join(SUB_DATA_DIR, 'list_appended_good.txt')

# to check creating names
TEST_LIST_OUT_GOOD = os.path.join(SUB_DATA_DIR, 'test_list_good.txt')

# Checking diff_lines fails well
LIST_OPP_ORDER = os.path.join(SUB_DATA_DIR, 'list_opp_order.txt')
LIST_WORD_DIFF = os.path.join(SUB_DATA_DIR, 'list_word_diff.txt')

LIST_TO_PRINT = ['Word', ["a", 1, 4.55]]
DICT_TO_PRINT = [{'name': 'qm_common', 'row': 0}]
FIELD_NAMES = ['name', 'row']
DICT_OUT_GOOD = os.path.join(SUB_DATA_DIR, 'dict_good.csv')
DICT_APPENDED_GOOD = os.path.join(SUB_DATA_DIR, 'dict_appended_good.csv')
DICT_FLOAT_DIFF = os.path.join(SUB_DATA_DIR, 'dict_float_diff.csv')

# data for testing reading config
NAME_KEY = 'name'
RANK_KEY = 'rank'
INTS_KEY = 'ints'
FLOATS_KEY = 'floats'
COLORS_KEY = 'colors'
BEV_KEY = 'beverage'
BOOL_KEY = 'bool_key'
BOOL_FALSE = 'bool_key_false'
RAW_CFG = {RANK_KEY: 1, COLORS_KEY: 'blue, green', INTS_KEY: '5,6', BOOL_FALSE: 'T'}
RAW_CFG_DIFF_BOOL = {NAME_KEY: 'team', RANK_KEY: 1, COLORS_KEY: 'blue, green', INTS_KEY: '5,6', BOOL_FALSE: 10}
RAW_CFG_WRONG_TYPE = {NAME_KEY: 'team', BOOL_FALSE: 'F', RANK_KEY: 'ghost'}
RAW_CFG_GOOD = {NAME_KEY: 'team', COLORS_KEY: 'blue, green', INTS_KEY: '5,6', FLOATS_KEY: '2, 12.5, 75',
                BOOL_KEY: 'T', BOOL_FALSE: 'False', RANK_KEY: 1}
PROC_CFG_GOOD = {NAME_KEY: 'team', COLORS_KEY: ['blue', 'green'], INTS_KEY: [5, 6], FLOATS_KEY: [2.0, 12.5, 75.0],
                 BEV_KEY: 'coffee', BOOL_KEY: True, BOOL_FALSE: False, RANK_KEY: 1}
DEF_CFG = {COLORS_KEY: [], INTS_KEY: [1, 2], BEV_KEY: 'coffee', FLOATS_KEY: [], BOOL_KEY: False, }
REQ_CFG = {NAME_KEY: str, BOOL_FALSE: bool, RANK_KEY: int}


class TestDiffLines(unittest.TestCase):
    def testWrongOrder(self):
        self.assertEqual(len(diff_lines(LIST_OUT_GOOD, LIST_OPP_ORDER)), 4)

    def testWordDiff(self):
        diffs = diff_lines(LIST_OUT_GOOD, LIST_WORD_DIFF, delimiter=" ")
        self.assertEqual(len(diffs), 2)

    def testFloatDiff(self):
        diffs = diff_lines(DICT_OUT_GOOD, DICT_FLOAT_DIFF)
        self.assertFalse(diffs)


class TestCreateOutName(unittest.TestCase):
    # create_out_fname(src_file, prefix='', suffix='', remove_prefix=None, base_dir=None, ext=None):
    def testAddSuffix(self):
        f_name = create_out_fname(LIST_OUT, suffix = "_good")
        self.assertEqual(LIST_OUT_GOOD, f_name)

    def testAddPrefixSuffix(self):
        f_name = create_out_fname(LIST_OUT, prefix="test_", suffix = "_good")
        self.assertEqual(TEST_LIST_OUT_GOOD, f_name)

    def testRemovePrefix(self):
        f_name = create_out_fname(TEST_LIST_OUT_GOOD, remove_prefix="test_")
        self.assertEqual(LIST_OUT_GOOD, f_name)

    def testRemovePrefixNotThere(self):
        f_name = create_out_fname(LIST_OUT, remove_prefix="test_", suffix = "_good")
        self.assertEqual(LIST_OUT_GOOD, f_name)


class TestListToFile(unittest.TestCase):
    def testMixedList(self):
        # Tests that writes and appends correctly
        try:
            with capture_stdout(list_to_file, LIST_TO_PRINT, LIST_OUT) as output:
                self.assertTrue('Wrote file' in output)
            self.assertFalse(diff_lines(LIST_OUT, LIST_OUT_GOOD))
            with capture_stdout(list_to_file, LIST_TO_PRINT, LIST_OUT, mode='a') as output:
                self.assertTrue('Appended' in output)
            self.assertFalse(diff_lines(LIST_OUT, LIST_APPENDED_GOOD))
        finally:
            silent_remove(LIST_OUT)


class TestWriteCSV(unittest.TestCase):
    def testDictToCSV(self):
        # Tests that it writes correctly and appends correctly
        try:
            with capture_stdout(write_csv, DICT_TO_PRINT, LIST_OUT, FIELD_NAMES) as output:
                self.assertTrue('Wrote file' in output)
            self.assertFalse(diff_lines(LIST_OUT, DICT_OUT_GOOD))
            with capture_stdout(write_csv, DICT_TO_PRINT, LIST_OUT, FIELD_NAMES, mode='a') as output:
                self.assertTrue('Appended' in output)
            self.assertFalse(diff_lines(LIST_OUT, DICT_APPENDED_GOOD))
        finally:
            silent_remove(LIST_OUT)


class TestArcLengthCalculator(unittest.TestCase):
    def testArcLengthOneRadian(self):
        try:
            p1 = 179 # around equator (ranges from 0 to 360)
            t1 = 90 # vertical (ranges from 0 to 180)
            p2 = 180 # around equator (ranges fom 0 to 360)
            t2 = 90 # vertical (ranges form 0 to 180)
            arc_length = arc_length_calculator(p1, t1, p2, t2)
        finally:
            self.assertEqual(round(arc_length,4), round(math.pi / 180,4))

    def testArcLengthPiRadians(self):
        try:
            p1 = 0 # around equator (ranges from 0 to 360)
            t1 = 90 # vertical (ranges from 0 to 180)
            p2 = 180 # around equator (ranges fom 0 to 360)
            t2 = 90 # vertical (ranges form 0 to 180)
            arc_length = arc_length_calculator(p1, t1, p2, t2)
        finally:
            self.assertEqual(round(arc_length,4), round(math.pi,4))

    def testArcLengthZeroRadians(self):
        try:
            p1 = 360 # around equator (ranges from 0 to 360)
            t1 = 55 # vertical (ranges from 0 to 180)
            p2 = 0 # around equator (ranges fom 0 to 360)
            t2 = 55 # vertical (ranges form 0 to 180)
            arc_length = arc_length_calculator(p1, t1, p2, t2)
        finally:
            self.assertEqual(round(arc_length,4), round(0.000000,4))

    def testArcLength1e1hoRadians(self):
        try:
            p1 = 240 # around equator (ranges from 0 to 360)
            t1 = 125 # vertical (ranges from 0 to 180)
            p2 = 210 # around equator (ranges fom 0 to 360)
            t2 = 129 # vertical (ranges form 0 to 180)
            arc_length1 = arc_length_calculator(p1, t1, p2, t2)
            arc_length2 = arc_length_calculator(p1, t1, 270, 129) # puckers for 1h2
        finally:
            self.assertEqual(round(arc_length1,4), round(arc_length2,4))

    def testArcLength1etoe1Radians(self):
        try:
            p1 = 240 # around equator (ranges from 0 to 360)
            t1 = 125 # vertical (ranges from 0 to 180)
            p2 = 60 # around equator (ranges fom 0 to 360)
            t2 = 55 # vertical (ranges form 0 to 180)
            arc_length1 = arc_length_calculator(p1, t1, p2, t2)
        finally:
            self.assertEqual(round(arc_length1,4), round(math.pi,4))

class TestProcessCfg(unittest.TestCase):
    def testEmptyCfg(self):
        self.assertEqual(len(process_cfg({})), 0)

    def testExtraKey(self):
        try:
            process_cfg(RAW_CFG, DEF_CFG)
        except InvalidDataError as e:
            self.assertTrue('Unexpected key ' in e.message)

    def testBoolWarning(self):
        with capture_stderr(process_cfg, RAW_CFG_DIFF_BOOL, DEF_CFG, REQ_CFG) as output:
            self.assertTrue('when expecting a boolean input' in output)

    def testMissReq(self):
        # In addition to catching that there is a missing required config value, catches warning that '10' will
        # be interpreted as false
        try:
            process_cfg(RAW_CFG, DEF_CFG, REQ_CFG)
        except KeyError as e:
            self.assertTrue("Missing config val for key 'name'" in e.message)

    def testWrongType(self):
        try:
            process_cfg(RAW_CFG_WRONG_TYPE, DEF_CFG, REQ_CFG)
        except InvalidDataError as e:
            self.assertTrue('Problem with config vals on key ' in e.message)

    def testGoodInput(self):
        # has required key, default filled in, all good!
        self.assertEqual(PROC_CFG_GOOD, process_cfg(RAW_CFG_GOOD, DEF_CFG, REQ_CFG))


class TestDeQuote(unittest.TestCase):
    def testDequote(self):
        self.assertTrue(dequote('"(0, 1)"') == '(0, 1)')

    def testNoDequoteNeeded(self):
        self.assertTrue(dequote("(0, 1)") == '(0, 1)')

    def testDequoteUnmatched(self):
        self.assertTrue(dequote('"' + '(0, 1)') == '"(0, 1)')
