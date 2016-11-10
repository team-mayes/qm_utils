from __future__ import print_function

#!/usr/bin/env python
from qm_utils.qm_common import warning

"""
Module docstring.
"""

import sys
import argparse

__author__ = 'cmayes'

def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input_rates", help="The location of the input rates file",
    #                     default=DEF_IRATE_FILE, type=read_input_rates)
    args = None
    try:
        args = parser.parse_args(argv)
    except IOError as e:
        warning("Problems reading file:", e)
        parser.print_help()
        return args, 2

    return args, 0



def main(argv=None):
    args, ret = parse_cmdline(argv)
    if ret != 0:
        return ret
    return 0  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
