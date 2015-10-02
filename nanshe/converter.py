"""
The module ``converter`` supports translation of TIFF files to HDF5 files.

===============================================================================
Overview
===============================================================================
The ``main`` function actually starts the algorithm and can be called
externally. Configuration files for the learner are provided in the examples_
and are entitled learner.

.. _examples: http://github.com/nanshe-org/nanshe/tree/master/examples

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 26, 2014 17:33:37 EDT$"


from nanshe.util import prof
from nanshe.io import xjson, xtiff


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

    # Only necessary if running main (normally if calling command line).
    # No point in importing otherwise.
    import argparse

    argv = list(argv)

    # Creates command line parser
    parser = argparse.ArgumentParser(
        description="Parses input from the command line for a batch job."
    )

    parser.add_argument(
        "format",
        choices=["tiff"],
        help="Format to convert from to HDF5.",
    )

    parser.add_argument(
        "config_filename",
        metavar="CONFIG_FILE",
        type=str,
        help="JSON file that provides configuration options" +
             " for how to import TIFF(s)."
    )
    parser.add_argument(
        "input_files",
        metavar="INPUT_FILE",
        type=str,
        nargs="+",
        help="TIFF file paths (with optional regex e.g. \"./*.tif\")."
    )

    parser.add_argument(
        "output_file",
        metavar="OUTPUT_FILE",
        type=str,
        nargs=1,
        help="HDF5 file to export" +
             " (this should include a path to" +
             " where the internal dataset should be stored)."
    )

    # Results of parsing arguments
    # (ignore the first one as it is the command line call).
    parsed_args = parser.parse_args(argv[1:])

    # Go ahead and stuff in parameters with the other parsed_args
    parsed_args.parameters = xjson.read_parameters(parsed_args.config_filename)

    if parsed_args.format == "tiff":
        xtiff.convert_tiffs(
            parsed_args.input_files,
            parsed_args.output_file[0],
            **parsed_args.parameters
        )

    return(0)
