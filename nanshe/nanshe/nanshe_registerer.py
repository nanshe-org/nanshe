#!/usr/bin/env python

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Feb 20, 2015 13:00:51 EST$"


import itertools
import os

import h5py

from nanshe.util import prof, xjson
from nanshe.util.pathHelpers import PathComponents
import registration



# Get the logger
logger = prof.logging.getLogger(__name__)



@prof.log_call(logger)
def main(*argv):
    """
        Simple main function (like in C). Takes all arguments (as from sys.argv) and returns an exit status.

        Args:
            argv(list):     arguments (includes command line call).

        Returns:
            int:            exit code (0 if success)
    """

    # Only necessary if running main (normally if calling command line). No point in importing otherwise.
    import argparse

    argv = list(argv)

    # Creates command line parser
    parser = argparse.ArgumentParser(description = "Parses input from the command line for a registration job.")

    parser.add_argument("config_filename",
                        metavar = "CONFIG_FILE",
                        type = str,
                        help = "JSON file that provides configuration options for how to import TIFF(s)."
    )
    parser.add_argument("input_filenames",
                        metavar = "INPUT_FILE",
                        type = str,
                        nargs = "+",
                        help = "HDF5 file to import (this should include a path to where the internal dataset should be stored)."
    )

    parser.add_argument("output_filenames",
                        metavar = "OUTPUT_FILE",
                        type = str,
                        nargs = 1,
                        help = "HDF5 file to export (this should include a path to where the internal dataset should be stored)."
    )

    # Results of parsing arguments (ignore the first one as it is the command line call).
    parsed_args = parser.parse_args(argv[1:])

    # Go ahead and stuff in parameters with the other parsed_args
    parsed_args.parameters = xjson.read_parameters(parsed_args.config_filename)

    parsed_args.input_file_components = []
    for each_input_filename in parsed_args.input_filenames:
        parsed_args.input_file_components.append(PathComponents(each_input_filename))

    parsed_args.output_file_components = []
    for each_output_filename in parsed_args.output_filenames:
        parsed_args.output_file_components.append(PathComponents(each_output_filename))


    for each_input_filename_components, each_output_filename_components in itertools.izip(parsed_args.input_file_components, parsed_args.output_file_components):
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
                os.remove(result_filename)
                os.removedirs(os.path.dirname(result_filename))

    return(0)


if __name__ == "__main__":
    # only necessary if running main (normally if calling command line). no point in importing otherwise.
    import sys

    # call main if the script is loaded from command line. otherwise, user can import package without main being called.
    sys.exit(main(*sys.argv))
