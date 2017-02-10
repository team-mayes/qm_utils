#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_cp
----------------------------------

Tests for `cp` module. (CP = Cremer-Pople)
"""

import unittest
import os
from qm_utils.break_up_file import main

__author__ = 'SPV'

# Directories #

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'break_up_file')

# Input & corresponding output files #

INPUT_TEXT_FILE = os.path.join(SUB_DATA_DIR, 'puckersupportXYZ_only.txt')
OUTPUT_DIRECTORY = os.path.dirname(INPUT_TEXT_FILE)

# TESTS #
main_input = ["-i", INPUT_TEXT_FILE, '-d', OUTPUT_DIRECTORY]

print(main_input)
main(main_input)
