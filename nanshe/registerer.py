"""
The ``registerer`` module allows the registration algorithm to be run.

===============================================================================
Overview
===============================================================================
The ``main`` function actually starts the algorithm and can be called
externally. Configuration files for the registerer are provided in the
examples_ and are entitled registerer. Any attributes on the raw dataset are
copied to the registered dataset.

.. _examples: http://github.com/nanshe-org/nanshe/tree/master/examples

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Feb 20, 2015 13:00:51 EST$"


import itertools
import os

import h5py

from nanshe.util import iters, prof
from nanshe.io import xjson
from nanshe.util.pathHelpers import PathComponents
from nanshe.imp import registration



# Get the logger
trace_logger = prof.getTraceLogger(__name__)



@prof.log_call(trace_logger)
def main(*argv):
    """
        Simple main function (like in C). Takes all arguments (as from
        sys.argv) and returns an exit status.

        Args:
            argv(list):     arguments (includes command line call).

        Returns:
            int:            exit code (0 if success)
    """

    # Only necessary if running main (normally if calling command line). No
    # point in importing otherwise.
    import argparse

    argv = list(argv)

    # Creates command line parser
    parser = argparse.ArgumentParser(
        description="Parses input from the command line " +
                    "for a registration job."
    )

    parser.add_argument("config_filename",
                        metavar="CONFIG_FILE",
                        type=str,
                        help="JSON file that provides configuration options " +
                             "for how to import TIFF(s)."
    )
    parser.add_argument("input_filenames",
                        metavar="INPUT_FILE",
                        type=str,
                        nargs=1,
                        help="HDF5 file to import (this should include a " +
                             "path to where the internal dataset should be " +
                             "stored)."
    )

    parser.add_argument("output_filenames",
                        metavar="OUTPUT_FILE",
                        type=str,
                        nargs=1,
                        help="HDF5 file to export (this should include a " +
                             "path to where the internal dataset should be " +
                             "stored)."
    )

    # Results of parsing arguments
    # (ignore the first one as it is the command line call).
    parsed_args = parser.parse_args(argv[1:])

    # Go ahead and stuff in parameters with the other parsed_args
    parsed_args.parameters = xjson.read_parameters(parsed_args.config_filename)

    parsed_args.input_file_components = []
    for each_input_filename in parsed_args.input_filenames:
        parsed_args.input_file_components.append(
            PathComponents(each_input_filename)
        )

    parsed_args.output_file_components = []
    for each_output_filename in parsed_args.output_filenames:
        parsed_args.output_file_components.append(
            PathComponents(each_output_filename)
        )

    for each_input_filename_components, each_output_filename_components in iters.izip(
            parsed_args.input_file_components, parsed_args.output_file_components):
        with h5py.File(each_input_filename_components.externalPath, "r") as input_file:
            with h5py.File(each_output_filename_components.externalPath, "a") as output_file:
                data = input_file[each_input_filename_components.internalPath]
                result_filename = registration.register_mean_offsets(
                    data, to_truncate=True, **parsed_args.parameters
                )
                with h5py.File(result_filename, "r") as result_file:
                    result_file.copy(
                        "reg_frames",
                        output_file[each_output_filename_components.internalDirectory],
                        name=each_output_filename_components.internalDatasetName
                    )

                    if parsed_args.parameters.get("include_shift", False):
                        result_file.copy(
                            "space_shift",
                            output_file[each_output_filename_components.internalDirectory],
                            name=each_output_filename_components.internalDatasetName + "_shift"
                        )

                # Copy all attributes from raw data to the final result.
                output = output_file[
                    each_output_filename_components.internalDatasetName
                ]
                for each_attr_name in data.attrs:
                    output.attrs[each_attr_name] = data.attrs[each_attr_name]

                # Only remove the directory if our input or output files are
                # not stored there.
                os.remove(result_filename)
                in_out_dirnames = set(
                    os.path.dirname(os.path.abspath(_.filename)) for _ in [
                        input_file, output_file
                    ]
                )
                result_dirname = os.path.dirname(result_filename)
                if result_dirname not in in_out_dirnames:
                    os.rmdir(result_dirname)

    return(0)
