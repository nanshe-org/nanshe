"""
The module ``search`` provides glob paths to search for content in a HDF5 file.

===============================================================================
Overview
===============================================================================
The module implements a strategy similar to Python's |glob|_ module for HDF5
files. In short, it uses regex patterns to match as many possible paths as it
can.

.. |glob| replace:: ``glob``
.. _glob: http://docs.python.org/2/library/glob.html

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 18, 2014 20:06:44 EDT$"


import re
import collections
import itertools

import h5py


# Need in order to have logging information no matter what.
from nanshe.util import prof


# Get the logger
trace_logger = prof.getTraceLogger(__name__)


@prof.log_call(trace_logger)
def get_matching_paths(a_filehandle, a_path_pattern):
    """
        Looks for existing paths that match the full provide pattern path.
        Returns a list of matches for the given file handle.

        Args:
            a_filehandle(h5py.File):        an HDF5 file.
            a_path_pattern(str):            an internal path (with patterns for
                                            each group) for the HDF5 file.

        Returns:
            (list):                         a list of matching paths.
    """

    current_pattern_group_matches = []

    if (isinstance(a_filehandle, h5py.Group) and a_path_pattern):
        current_group = a_filehandle

        a_path_pattern = a_path_pattern.strip("/")

        to_split = a_path_pattern.find("/")
        if to_split != -1:
            current_path = a_path_pattern[:to_split]
            next_path = a_path_pattern[1 + to_split:]
        else:
            current_path, next_path = a_path_pattern, ""

        current_pattern_group_regex = re.compile("/" + current_path + "/")

        for each_group in current_group:
            if current_pattern_group_regex.match("/" + each_group + "/") is not None:
                next_group = current_group[each_group]

                next_pattern_group_matches = get_matching_paths(
                    next_group, next_path
                )

                for each_next_pattern_group_match in next_pattern_group_matches:
                    current_pattern_group_matches.append(
                        "/" + each_group + each_next_pattern_group_match
                    )
    else:
        current_pattern_group_matches = [""]

    return(current_pattern_group_matches)


@prof.log_call(trace_logger)
def get_matching_paths_groups(a_filehandle, a_path_pattern):
    """
        Looks for parts of the path pattern and tries to match them in order.
        Returns a list of matches that can be combined to yield acceptable
        matches for the given file handle.

        Note:
            This works best when a tree structure is created systematically in
            HDF5. Then, this will recreate what the tree structure could and
            may contain.

        Args:
            a_filehandle(h5py.File):        an HDF5 file.
            a_path_pattern(str):            an internal path (with patterns for
                                            each group) for the HDF5 file.

        Returns:
            (list):                         a list of matching paths.
    """

    def get_matching_paths_groups_recursive(a_filehandle, a_path_pattern):
        current_pattern_group_matches = []

        if (isinstance(a_filehandle, h5py.Group) and a_path_pattern):
            current_pattern_group_matches.append(collections.OrderedDict())

            current_group = a_filehandle

            a_path_pattern = a_path_pattern.strip("\b").strip("/")

            to_split = a_path_pattern.find("/")
            if to_split != -1:
                current_path = a_path_pattern[:to_split]
                next_path = a_path_pattern[1 + to_split:]
            else:
                current_path, next_path = a_path_pattern, ""

            current_pattern_group_regex = re.compile("/" + current_path + "/")

            for each_group in current_group:
                if current_pattern_group_regex.match("/" + each_group + "/") is not None:
                    next_group = current_group[each_group]

                    next_pattern_group_matches = get_matching_paths_groups_recursive(
                        next_group, next_path
                    )

                    current_pattern_group_matches[0][each_group] = None

                    while (len(current_pattern_group_matches) - 1) < len(next_pattern_group_matches):
                        current_pattern_group_matches.append(
                            collections.OrderedDict()
                        )

                    for i, each_next_pattern_group_matches in enumerate(
                            next_pattern_group_matches, start=1
                    ):
                        for each_next_pattern_group_match in each_next_pattern_group_matches:
                            current_pattern_group_matches[i][each_next_pattern_group_match] = None
        else:
            current_pattern_group_matches = []

        return(current_pattern_group_matches)

    groups = get_matching_paths_groups_recursive(a_filehandle, a_path_pattern)

    new_groups = []
    for i in xrange(len(groups)):
        new_groups.append(list(groups[i]))

    groups = new_groups

    return(groups)


@prof.log_call(trace_logger)
def get_matching_grouped_paths(a_filehandle, a_path_pattern):
    """
        Looks for existing paths that match the full provide pattern path.
        Returns a list of matches as keys and whether they are found in the
        HDF5 file or not.

        Args:
            a_filehandle(h5py.File):        an HDF5 file.
            a_path_pattern(str):            an internal path (with patterns for
                                            each group) for the HDF5 file.

        Returns:
            (list):                         an ordered dictionary with possible
                                            paths that fit the pattern and
                                            whether they are found.
    """

    paths_found = collections.OrderedDict()

    for each_path_components in itertools.product(
            *get_matching_paths_groups(a_filehandle, a_path_pattern)
    ):
        each_path = "/" + "/".join([_ for _ in each_path_components])

        paths_found[each_path] = None

    paths_found = paths_found.keys()

    return(paths_found)


@prof.log_call(trace_logger)
def get_matching_grouped_paths_found(a_filehandle, a_path_pattern):
    """
        Looks for existing paths that match the full provide pattern path.
        Returns a list of matches as keys and whether they are found in the
        HDF5 file or not.

        Args:
            a_filehandle(h5py.File):        an HDF5 file.
            a_path_pattern(str):            an internal path (with patterns for
                                            each group) for the HDF5 file.

        Returns:
            (collections.OrderedDict):      an ordered dictionary with possible
                                            paths that fit the pattern and
                                            whether they are found.
    """

    paths_found = collections.OrderedDict()

    for each_path_components in itertools.product(
            *get_matching_paths_groups(a_filehandle, a_path_pattern)
    ):
        each_path = "/" + "/".join([_ for _ in each_path_components])

        paths_found[each_path] = (each_path in a_filehandle)

    return(paths_found)
