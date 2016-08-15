#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_read_sdf
----------------------------------

Tests for `read_sdf` module.
"""

import unittest
import os
from qm_utils.qm_common import list_to_file, diff_lines, silent_remove, write_csv

__author__ = 'hmayes'

# Directories #

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'common')

# OUTPUT files #

LIST_OUT = os.path.join(SUB_DATA_DIR, 'list.txt')
LIST_OUT_GOOD = os.path.join(SUB_DATA_DIR, 'list_good.txt')

# Checking fails well
LIST_OPP_ORDER = os.path.join(SUB_DATA_DIR, 'list_opp_order.txt')
LIST_WORD_DIFF = os.path.join(SUB_DATA_DIR, 'list_word_diff.txt')

LIST_TO_PRINT = ['Word', ["a", 1, 4.55]]
DICT_TO_PRINT = [{'name': 'qm_common', 'row': 0}]
FIELD_NAMES = ['name', 'row']
DICT_OUT_GOOD = os.path.join(SUB_DATA_DIR, 'dict_good.csv')
DICT_FLOAT_DIFF = os.path.join(SUB_DATA_DIR, 'dict_float_diff.csv')


class TestListToFile(unittest.TestCase):
    def testMixedList(self):
        try:
            list_to_file(LIST_TO_PRINT, LIST_OUT)
            self.assertFalse(diff_lines(LIST_OUT, LIST_OUT_GOOD))
        finally:
            silent_remove(LIST_OUT)


class TestDiffLines(unittest.TestCase):
    def testWrongOrder(self):
        self.assertEqual(len(diff_lines(LIST_OUT_GOOD, LIST_OPP_ORDER)), 4)

    def testWordDiff(self):
        diffs = diff_lines(LIST_OUT_GOOD, LIST_WORD_DIFF, delimiter=" ")
        self.assertEqual(len(diffs), 2)

    def testFloatDiff(self):
        diffs = diff_lines(DICT_OUT_GOOD, DICT_FLOAT_DIFF)
        self.assertFalse(diffs)


class TestWriteCSV(unittest.TestCase):
    def testMixedList(self):
        try:
            write_csv(DICT_TO_PRINT, LIST_OUT, FIELD_NAMES)
            self.assertFalse(diff_lines(LIST_OUT, DICT_OUT_GOOD))
        finally:
            silent_remove(LIST_OUT)
