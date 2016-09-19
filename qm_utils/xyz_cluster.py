#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
(insert a description about the code here)

The purpose of this python script to align xyz coordinate files so that the structures
can be clustered.

"""

from sys import argv

__author__ = 'SPVicchio'


# Constants #


# Defaults #


# Functions #

def print_a_line(line_count, f):
    print line_count, f.readline()


def get_coordinates_xyz(filename):
    print "Currently looking at %r." % filename





    xyz_raw_file = open(filename)
    xyz_raw_data = xyz_raw_file.read()

    print "The input file is %d bytes long" % len(xyz_raw_data)

    #    print "%r" % xyz_raw_data

    line_count = 1


#    print line_count, xyz_raw_data.readline()
#    print_a_line(4,xyz_raw_data)



from sys import argv

script, input_file = argv

get_coordinates_xyz(input_file)
