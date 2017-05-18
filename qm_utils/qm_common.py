#!/usr/bin/env python
# coding=utf-8

"""
Common methods for this project.
"""

from __future__ import print_function
import collections
import csv
import difflib
import os
import sys
import fnmatch
import errno
import six
from contextlib import contextmanager
import math

ACCEPT_AS_TRUE = ['T', 't', 'true', 'TRUE', 'True']

GOOD_RET = 0
INPUT_ERROR = 1
IO_ERROR = 2
INVALID_DATA = 3

# Tolerance initially based on double standard machine precision of 5 × 10−16 for float64 (decimal64)
# found to be too stringent
TOL = 0.00000000001

NA = 'N/A'


# Error checking including testing scripts

# Exceptions #

class QmError(Exception):
    pass


class InvalidDataError(QmError):
    pass


# From http://schinckel.net/2013/04/15/capture-and-test-sys.stdout-sys.stderr-in-unittest.testcase/
@contextmanager
def capture_stdout(command, *args, **kwargs):
    # noinspection PyCallingNonCallable
    out, sys.stdout = sys.stdout, six.StringIO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out


@contextmanager
def capture_stderr(command, *args, **kwargs):
    # noinspection PyCallingNonCallable
    err, sys.stderr = sys.stderr, six.StringIO()
    command(*args, **kwargs)
    sys.stderr.seek(0)
    yield sys.stderr.read()
    sys.stderr = err


def warning(*objs):
    """Writes a message to stderr."""
    print("WARNING: ", *objs, file=sys.stderr)


def diff_lines(floc1, floc2, delimiter=","):
    """
    Determine all lines in a file are equal.
    If not, test if the line is a csv that has floats and the difference is due to machine precision.
    If not, return all lines with differences.
    @param floc1: file location 1
    @param floc2: file location 1
    @param delimiter: defaults to CSV
    @return: a list of the lines with differences
    """
    diff_lines_list = []
    # Save diffs to strings to be converted to use csv parser
    output_plus = ""
    output_neg = ""
    with open(floc1, 'r') as file1:
        with open(floc2, 'r') as file2:
            diff = difflib.ndiff(file1.read().splitlines(), file2.read().splitlines())
    for line in diff:
        if line.startswith('-') or line.startswith('+'):
            diff_lines_list.append(line)
            if line.startswith('-'):
                output_neg += line[2:]+'\n'
            elif line.startswith('+'):
                output_plus += line[2:]+'\n'
    try:
        # noinspection PyCallingNonCallable
        diff_plus_lines = list(csv.reader(six.StringIO(output_plus), delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC))
        # noinspection PyCallingNonCallable
        diff_neg_lines = list(csv.reader(six.StringIO(output_neg), delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC))
    except ValueError:
        diff_plus_lines = output_plus.split('\n')
        diff_neg_lines = output_neg.split('\n')
        for diff_list in [diff_plus_lines, diff_neg_lines]:
            for line_id in range(len(diff_list)):
                diff_list[line_id] = diff_list[line_id].split(delimiter)

    if len(diff_plus_lines) == len(diff_neg_lines):
        # if the same number of lines, there is a chance that the difference is only due to difference in
        # floating point precision. Check each value of the line, split on whitespace or comma
        diff_lines_list = []
        for line_plus, line_neg in zip(diff_plus_lines, diff_neg_lines):
            # if they are the same, then they are out of order.
            if len(line_plus) == len(line_neg) and line_plus != line_neg:
                print("Checking for differences between: ", line_neg, line_plus)
                for item_plus, item_neg in zip(line_plus, line_neg):
                    if isinstance(item_plus, float) and isinstance(item_neg, float):
                        # if difference greater than the tolerance, the difference is not just precision
                        float_diff = abs(item_plus - item_neg)
                        calc_tol = max(TOL * max(abs(item_plus), abs(item_neg)), TOL)
                        if float_diff > calc_tol:
                            warning("Values {} and {} differ by {}, which is greater than the calculated tolerance ({})"
                                    "".format(item_plus, item_neg, float_diff, calc_tol))
                            diff_lines_list.append("- " + " ".join(map(str, line_neg)))
                            diff_lines_list.append("+ " + " ".join(map(str, line_plus)))
                            return diff_lines_list
                    else:
                        # not floats, so the difference is not just precision
                        if item_plus != item_neg:
                            diff_lines_list.append("- " + " ".join(map(str, line_neg)))
                            diff_lines_list.append("+ " + " ".join(map(str, line_plus)))
                            return diff_lines_list
            # Not the same number of items in the lines
            else:
                diff_lines_list.append("- " + " ".join(map(str, line_neg)))
                diff_lines_list.append("+ " + " ".join(map(str, line_plus)))
    return diff_lines_list


def silent_remove(filename, disable=False):
    """
    Removes the target file name, catching and ignoring errors that indicate that the
    file does not exist.

    @param filename: The file to remove.
    @param disable: boolean to flag if want to disable removal
    """
    if not disable:
        try:
            os.remove(filename)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise


# Conversions #

def dequote(s):
    """
    from: http://stackoverflow.com/questions/3085382/python-how-can-i-strip-first-and-last-double-quotes
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found, return the string unchanged.
    """
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s


def to_list(raw_val):
    """
    converts a comma-separated string into a list of integers if possible.
    Otherwise, a list of floats, or, if that fails, a list of strings.
    :param raw_val: a comma-separated string
    :return: a list of ints, floats, or strings (depending on what is possible)
    """
    try:
        return [int(val.strip()) for val in raw_val.split(',')]
    except ValueError:
        try:
            return [float(val.strip()) for val in raw_val.split(',')]
        except ValueError:
            return [val.strip() for val in raw_val.split(',')]


def conv_raw_val(param, def_val):
    """
    Converts the given parameter into the given type (default returns the raw value).  Returns the default value
    if the param is None.
    @param param: The value to convert.
    @param def_val: The value that determines the type to target.
    @return: The converted parameter value.
    """
    if param is None:
        return def_val
    if isinstance(def_val, bool):
        return check_bool(param)
    if isinstance(def_val, int):
        return int(param)
    if isinstance(def_val, float):
        return float(param)
    if isinstance(def_val, list):
        return to_list(param)
    return param


def check_bool(param):
    if param in ACCEPT_AS_TRUE:
        return True
    else:
        if param not in ['F', 'f', 'false', 'FALSE', 'False']:
            warning("Read '{}' when expecting a boolean input. Since the input is not in {}, it will be "
                    "interpreted as False.".format(param, ACCEPT_AS_TRUE))
        return False


def process_cfg(raw_cfg, def_cfg_vals=None, req_keys=None):
    """
    Converts the given raw configuration, filling in defaults and converting the specified value (if any) to the
    default value's type.
    @param raw_cfg: The configuration map.
    @param def_cfg_vals: dictionary of default values
    @param req_keys: dictionary of required types
    @return: The processed configuration.
    """
    if req_keys is None:
        req_keys = {}
    if def_cfg_vals is None:
        def_cfg_vals = {}
    proc_cfg = {}
    for key in raw_cfg:
        if not (key in def_cfg_vals or key in req_keys):
            raise InvalidDataError("Unexpected key '{}' in configuration ('ini') file.".format(key))
    key = None
    try:
        for key, def_val in def_cfg_vals.items():
            proc_cfg[key] = conv_raw_val(raw_cfg.get(key), def_val)
        for key, type_func in req_keys.items():
            if type_func == bool:
                proc_cfg[key] = check_bool(raw_cfg[key])
            else:
                proc_cfg[key] = type_func(raw_cfg[key])
    except KeyError as e:
        raise KeyError("Missing config val for key '{}'".format(key, e))
    except Exception as e:
        raise InvalidDataError("Problem with config vals on key '{}': {}".format(key, e))

    return proc_cfg


# Input and output

def find_files_by_dir(tgt_dir, pat):
    """Recursively searches the target directory tree for files matching the given pattern.
    The results are returned as a dict with a list of found files keyed by the absolute
    directory name.
    @param tgt_dir: The target base directory.
    @param pat: The file pattern to search for.
    @return: A dict where absolute directory names are keys for lists of found file names
        that match the given pattern.
    """
    match_dirs = {}
    for root, dirs, files in os.walk(tgt_dir):
        matches = [match for match in files if fnmatch.fnmatch(match, pat)]
        if matches:
            match_dirs[os.path.abspath(root)] = matches
    return match_dirs


def create_out_fname(src_file, prefix='', suffix='', remove_prefix=None, base_dir=None, ext=None):
    """Creates an outfile name for the given source file.

    @param remove_prefix: string to remove at the beginning of file name
    @param src_file: The file to process.
    @param prefix: The file prefix to add, if specified.
    @param suffix: The file suffix to append, if specified.
    @param base_dir: The base directory to use; defaults to `src_file`'s directory.
    @param ext: The extension to use instead of the source file's extension;
        defaults to the `scr_file`'s extension.
    @return: The output file name.
    """

    if base_dir is None:
        base_dir = os.path.dirname(src_file)

    file_name = os.path.basename(src_file)
    base_name = os.path.splitext(file_name)[0]
    if remove_prefix is not None and file_name.startswith(remove_prefix):
        base_name = base_name[len(remove_prefix):]

    if ext is None:
        ext = os.path.splitext(file_name)[1]

    return os.path.abspath(os.path.join(base_dir, prefix + base_name + suffix + ext))


def list_to_file(list_to_print, fname, list_format=None, delimiter=' ', mode='w', print_message=True):
    """
    Writes the list of sequences to the given file in the specified format for a PDB.

    @param list_to_print: A list of lines to print. The list may be a list of lists, list of strings, or a mixture.
    @param fname: The location of the file to write.
    @param list_format: Specified formatting for the line if the line is  list.
    @param delimiter: If no format is given and the list contains lists, the delimiter will join items in the list.
    @param print_message: boolean to determine whether to write to output if the file is printed or appended
    @param mode: write by default; can be changed to allow appending to file.
    """
    with open(fname, mode) as w_file:
        for line in list_to_print:
            if isinstance(line, six.string_types):
                w_file.write(line + '\n')
            elif isinstance(line, collections.Iterable):
                if list_format is None:
                    w_file.write(delimiter.join(map(str, line)) + "\n")
                else:
                    w_file.write(list_format.format(*line) + '\n')
    if print_message:
        if mode == 'w':
            print("Wrote file: {}".format(fname))
        elif mode == 'a':
            print("  Appended: {}".format(fname))


def read_csv_to_dict(src_file, mode='r', quote_style=csv.QUOTE_MINIMAL):
    """
    Reads the given CSV (comma-separated with a first-line header row) and returns a list of
    dicts where each dict contains a row's data keyed by the header row.

    @param src_file: The CSV to read.
    @param quote_style: how to read the dictionary
    @param mode: default is 'r'; now can specify 'rb' or 'rU'
    @return: A list of dicts containing the file's data.
    """
    result = []

    with open(src_file, mode) as csv_file:
        csv_reader = csv.DictReader(csv_file, quoting=quote_style)
        for line in csv_reader:
            result.append(convert_dict_line(line))
    return result


def get_csv_fieldnames(src_file, mode='r', quote_style=csv.QUOTE_MINIMAL):
    """
    Get fieldnames in preserved order (list)
    :param src_file:
    :param quote_style:
    @param mode: default is 'r'; now can specify 'rb' or 'rU'
    :return: list of fieldnames from CSV
    """
    with open(src_file, mode) as csv_file:
        csv_reader = csv.DictReader(csv_file, quoting=quote_style)
        return csv_reader.fieldnames


def convert_dict_line(line):
    s_dict = {}
    for s_key, s_val in line.items():
        s_dict[s_key] = s_val
    return s_dict


def list_to_dict(list_of_dicts, key_for_dict):
    """
    Convert a list of dicts to a dict of dicts
    :param list_of_dicts: a list of dicts
    :param key_for_dict: key in inner dict, whose value will be the key for the outer dict
    :return : a dict of dicts, with the keys being based on key_for_dict, and the values being the "inner" dict
    """
    dict_of_dicts = {}
    for row in list_of_dicts:
        if row[key_for_dict] in dict_of_dicts:
            warning("Will overwrite dictionary entry for key '{}'".format(row[key_for_dict]))
        dict_of_dicts[row[key_for_dict]] = row
    return dict_of_dicts


def write_csv(data, out_fname, fieldnames, extrasaction="raise", mode='w', quote_style=csv.QUOTE_NONNUMERIC):
    """
    Writes the given data to the given file location.

    @param data: The data to write (list of dicts).
    @param out_fname: The name of the file to write to.
    @param fieldnames: The sequence of field names to use for the header.
    @param extrasaction: What to do when there are extra keys.  Acceptable
        values are "raise" or "ignore".
    @param mode: default mode is to overwrite file
    @param quote_style: dictates csv output style
    """
    with open(out_fname, mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames, extrasaction=extrasaction, quoting=quote_style)
        if mode == 'w':
            writer.writeheader()
        writer.writerows(data)
    if mode == 'w':
        print("Wrote file: {}".format(out_fname))
    elif mode == 'a':
        print("  Appended: {}".format(out_fname))


def write_csv_old(tgt, write_list, cols):
    try:
        writer = csv.writer(tgt)
        writer.writerow(cols)
        for write_row in write_list:
            writer.writerow([write_row[col] for col in cols])
    finally:
        tgt.close()


def conv_map_old(conv_dict, raw_vals):
    """
    Takes a dictionary mapping column name to conversion function
    and a list of dictionaries containing unconverted data.
    """
    conv_vals = list()
    for raw_row in raw_vals:
        conv_row = dict()
        for (col, cfunc) in conv_dict.items():
            if (col in raw_row and (raw_row[col] != NA)):
                conv_row[col] = cfunc(raw_row[col])
            else:
                conv_row[col] = ''

        conv_vals.append(conv_row)
    return conv_vals


def read_csv_old(file_loc):
    """
    Reads the given file, mapping data rows to their respective
    header row.
    """
    reader = csv.reader(open(file_loc, 'r'))
    is_first = True
    rows = list()
    for row in reader:
        if is_first:
            header = row
            is_first = False
            continue
        rows.append(dict(zip(header,row)))
    return rows


def prep_string(raw_string):
    """
    Reads potentially multi-line raw string and returns string that will be correctly outputted
    :param raw_string: a string which may have "\n" to denote line breaks and may be quoted
    :return: a string ready to print to the com file, including extra return
    """
    if len(raw_string) < 1:
        return raw_string
    else:
        return '\n'.join(dequote(raw_string).split('\\n'))


def arc_length_calculator(phi1, theta1, phi2, theta2, radius=1):
    """
    calculates the arc length between two points on the surface of a sphere using the haversine
    formula (https://en.wikipedia.org/wiki/Great-circle_distance)
    :param phi1: phi (deg) of structure 1
    :param theta1: theta (deg) of structure 1
    :param phi2: phi (deg) of structure 2
    :param theta2: theta (deg) of structure 2
    :param radius: radius used for the surface of sphere (default value is one)
    :return:
    """

    dp = abs(phi2 - phi1) * math.pi/180
    dt = abs(theta2 - theta1) * math.pi/180

    term1 = pow(math.sin(dt/2),2)
    term2 = math.sin(theta1 * math.pi/180) * math.sin(theta2 * math.pi/180) * pow(math.sin(dp/2),2)

    central_angle = 2 * math.asin(math.sqrt(term1 + term2))

    arc_length = radius * central_angle

    return arc_length

