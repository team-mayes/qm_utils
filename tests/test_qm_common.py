#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_read_sdf
----------------------------------

Tests for `read_sdf` module.
"""

import unittest
import os
from qm_utils.qm_common import list_to_file, diff_lines, silent_remove

__author__ = 'hmayes'

# Directories #

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'common')

# OUTPUT files #

LIST_OUT = os.path.join(SUB_DATA_DIR, 'list.txt')
LIST_OUT_GOOD = os.path.join(SUB_DATA_DIR, 'list_good.txt')


class TestListToFile(unittest.TestCase):
    def testMixedList(self):
        try:
            to_print = ['Word', ["a", 1, 4.55]]
            list_to_file(to_print, LIST_OUT)
            self.assertFalse(diff_lines(LIST_OUT, LIST_OUT_GOOD))
        finally:
            silent_remove(LIST_OUT)
