#!/usr/bin/env python

__author__ = "John Kirkham"
__date__ = "$Apr 9, 2014 4:00:40PM$"


import os

# Generally useful and fast to import so done immediately.
import numpy

import h5py

import lazyflow.utility.pathHelpers

# Need in order to have logging information no matter what.
import debugging_tools

import HDF5_recorder

# Short function to process image data.
import advanced_image_processing

# For IO. Right now, just includes read_parameters for reading a config.json file.
import read_config

import HDF5_serializers


# Get the logger
logger = debugging_tools.logging.getLogger(__name__)



@debugging_tools.log_call(logger)
def generate_neurons_io_handler(input_filename, output_filename, parameters_filename):
    """
        Uses generate_neurons to process a input_filename (HDF5 dataset) and outputs results to an output_filename (HDF5
        dataset). Also,

        Args:
            input_filename          HDF5 filename to read from (should be a path to a h5py.Dataset)
            output_filename         HDF5 filename to write to (should be a path to a h5py.Group)
            parameters_filename     JSON filename with parameters.
    """


    # Extract and validate file extensions.

    # Parse input filename and validate that the name is acceptable
    input_filename_details = lazyflow.utility.pathHelpers.PathComponents(input_filename)
    # Clean up the extension so it fits the standard.
    input_filename_details.extension = input_filename_details.extension.lower()
    input_filename_details.extension = input_filename_details.extension.lstrip(os.extsep)
    if ( input_filename_details.extension not in ["h5", "hdf5", "he5"] ):
        raise Exception("Input file with filename: \"" + input_filename + "\"" + " provided with an unknown file extension: \"" + input_filename_details.extension + "\". If it is a supported format, please run the given file through HDF5_importer first before proceeding.")

    # Parse output filename and validate that the name is acceptable
    output_filename_details = lazyflow.utility.pathHelpers.PathComponents(output_filename)
    # Clean up the extension so it fits the standard.
    output_filename_details.extension = output_filename_details.extension.lower()
    output_filename_details.extension = output_filename_details.extension.lstrip(os.extsep)
    if ( output_filename_details.extension not in ["h5", "hdf5", "he5"] ):
        raise Exception("Output file with filename: \"" + input_filename + "\"" + " provided with an unknown file extension: \"" + output_filename_details.extension + "\". If it is a supported format, please run the given file through HDF5_importer first before proceeding.")

    # Parse parameter filename and validate that the name is acceptable
    parameters_filename_details = lazyflow.utility.pathHelpers.PathComponents(parameters_filename)
    # Clean up the extension so it fits the standard.
    parameters_filename_details.extension = parameters_filename_details.extension.lower()
    parameters_filename_details.extension = parameters_filename_details.extension.lstrip(os.extsep)
    if ( parameters_filename_details.extension not in ["json"] ):
        raise Exception("Parameter file with filename: \"" + parameters_filename + "\"" + " provided with an unknown file extension: \"" + output_filename_details.extension + "\". If it is a supported format, please run the given file through HDF5_importer first before proceeding.")


    # Store useful values

    # Parse the parameters from the json file.
    parameters = read_config.read_parameters(parameters_filename)

    # Grab the debug value from the parameters. Let it default to false if it is not present.
    debug = parameters.get("debug", False)

    # Where the original images are.
    input_dataset_name = input_filename_details.internalPath

    # Name of the group where all data will be stored.
    output_group_name = output_filename_details.internalPath


    # Read the input data.
    original_images = None
    with h5py.File(input_filename_details.externalPath, "r") as input_file_handle:
        original_images_object = HDF5_serializers.read_numpy_structured_array_from_HDF5(input_file_handle, input_dataset_name)

        # TODO: Refactor into HDF5_serializers.read_numpy_structured_array_from_HDF5.
        # Read the original images in and also handle the case of a reference or region reference.
        if isinstance(original_images_object, numpy.ndarray):
            original_images = original_images_object
        elif isinstance(original_images_object, h5py.Reference):
            original_images = input_file_handle[original_images_object]
        elif isinstance(original_images_object, h5py.RegionReference):
            original_images = input_file_handle[original_images_object]
        else:
            raise Exception("Unknown type of, \"" + repr(type(original_images_object)) + "\", in the HDF5 input file named, \"" + input_filename + "\".")


    # Write out the output.
    with h5py.File(output_filename_details.externalPath, "a") as output_file_handle:
        # Create a new output directory if doesn't exists.
        if output_group_name not in output_file_handle:
            output_file_handle.create_group(output_group_name)

        # Group where all data will be stored.
        output_group = output_file_handle[output_group_name]

        # Create a soft link to the original images. But use the appropriate type of soft link depending on whether
        # the input and output file are the same.
        if input_dataset_name not in output_group:
            if input_filename_details.externalPath == output_filename_details.externalPath:
                output_group["original_images"] = h5py.SoftLink(input_dataset_name)
            else:
                output_group["original_images"] = h5py.ExternalLink(input_filename_details.externalPath, input_dataset_name)

        # Get a debug logger for the HDF5 file (if needed)
        array_debug_recorder = HDF5_recorder.generate_HDF5_array_recorder(output_group,
                                                                          group_name = "debug",
                                                                          enable = debug,
                                                                          overwrite_group = False)

        # Saves intermediate result to make resuming easier
        resume_logger = HDF5_recorder.generate_HDF5_array_recorder(output_group, allow_overwrite_dataset = True)

        # Generate the neurons and attempt to resume if possible
        generate_neurons(original_images = original_images, resume_logger = resume_logger, array_debug_recorder = array_debug_recorder, **parameters["generate_neurons"])

        # Save the configuration parameters in the attributes as a string.
        if "parameters" not in output_group.attrs:
            # Write the configuration parameters in the attributes as a string.
            output_group.attrs["parameters"] = repr(parameters)


@debugging_tools.log_call(logger)
def generate_neurons(original_images, run_stage = "all", resume_logger = HDF5_recorder.EmptyArrayRecorder(), array_debug_recorder = HDF5_recorder.EmptyArrayRecorder(), **parameters):
    if "original_images_max_projection" not in array_debug_recorder:
        array_debug_recorder("original_images_max_projection", original_images.max(axis = 0))

    if "original_images_mean_projection" not in array_debug_recorder:
        array_debug_recorder("original_images_mean_projection", original_images.mean(axis = 0))

    # Preprocess images
    new_preprocessed_images = resume_logger.get("preprocessed_images", None)
    if (new_preprocessed_images is None) or (run_stage == "preprocessing") or (run_stage == "all"):
        new_preprocessed_images = advanced_image_processing.preprocess_data(original_images,
                                                                            array_debug_recorder = array_debug_recorder,
                                                                            **parameters["preprocess_data"])
        resume_logger("preprocessed_images", new_preprocessed_images)

        if "preprocessed_images_max_projection" not in array_debug_recorder:
            array_debug_recorder("preprocessed_images_max_projection", new_preprocessed_images.max(axis = 0))

    if run_stage == "preprocessing":
        return

    # Find the dictionary
    new_dictionary = resume_logger.get("dictionary", None)
    if (new_dictionary is None) or (run_stage == "dictionary") or (run_stage == "all"):
        new_dictionary = advanced_image_processing.generate_dictionary(new_preprocessed_images,
                                                                       array_debug_recorder = array_debug_recorder,
                                                                       **parameters["generate_dictionary"])
        resume_logger("dictionary", new_dictionary)

        if "dictionary_max_projection" not in array_debug_recorder:
            array_debug_recorder("dictionary_max_projection", new_dictionary.max(axis = 0))

    if run_stage == "dictionary":
        return

    # Find the neurons
    new_neurons = None
    new_neurons = resume_logger.get("neurons", None)
    if (new_neurons is None) or (run_stage == "postprocessing") or (run_stage == "all"):
        new_neurons = advanced_image_processing.postprocess_data(new_dictionary,
                                                                 array_debug_recorder,
                                                                 **parameters["postprocess_data"])
        resume_logger("neurons", new_neurons)

    if new_neurons.size == 0:
        logger.warning("No neurons were found in the data.")


@debugging_tools.log_call(logger)
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
    parser = argparse.ArgumentParser(description = "Parses input from the command line for a batch job.")

    # Takes a config file and then a series of one or more HDF5 files.
    parser.add_argument("config_filename", metavar = "CONFIG_FILE", type = str,
                        help = "JSON file that provides configuration options for how to use dictionary learning on the input files.")
    parser.add_argument("input_file", metavar = "INPUT_FILE", type = str, nargs = 1,
                        help = "HDF5 file with an array of images. A single dataset or video will be expected at the internal path. Time must be the first dimension.")
    parser.add_argument("output_file", metavar = "OUTPUT_FILE", type = str, nargs = 1,
                            help = "HDF5 file(s) to write output. If a specific group is desired, that should be included in the filename.")

    # Results of parsing arguments (ignore the first one as it is the command line call).
    parsed_args = parser.parse_args(argv[1:])

    # Remove args from singleton lists
    parsed_args.input_file = parsed_args.input_file[0]
    parsed_args.output_file = parsed_args.output_file[0]

    # Runs the dictionary learning algorithm on each file with the given parameters
    # and saves the results in the given files.
    generate_neurons_io_handler(parsed_args.input_file, parsed_args.output_file, parsed_args.config_filename)

    return(0)


if __name__ == "__main__":
    # only necessary if running main (normally if calling command line). no point in importing otherwise.
    import sys

    # call main if the script is loaded from command line. otherwise, user can import package without main being called.
    sys.exit(main(*sys.argv))
