#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test structure_pairing
----------------------------------

"""
import logging
import os
import unittest

from qm_utils.method_comparison import main
from qm_utils.qm_common import capture_stderr, capture_stdout

__author__ = 'SPVicchio'

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)



# # # Directories # # #
TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'method_comparison')

# # # Input files # # #
LIST_OF_DATASET_FILES_BXYL = os.path.join(SUB_DATA_DIR, "a_list_dataset_bxyl.txt")

# # # Good output # # #





# class TestFailWell(unittest.TestCase):
#     def testHelp(self):
#         test_input = ['-h']
#         if logger.isEnabledFor(logging.DEBUG):
#             main(test_input)
#         with capture_stderr(main, test_input) as output:
#             self.assertEquals(output,'WARNING:  0\n')
#         with capture_stdout(main, test_input) as output:
#             self.assertTrue("optional arguments" in output)
#
#     def testNoSuchFile(self):
#         test_input = ["-s", "ghost"]
#         with capture_stderr(main, test_input) as output:
#             self.assertTrue("Could not find" in output)


class TestMain(unittest.TestCase):
    def testMainBxyl(self):
        test_input = ["-s", LIST_OF_DATASET_FILES_BXYL, "-d", SUB_DATA_DIR, "-m", "bxyl"]
        main(test_input)
