#!/usr/bin/env python

__author__ = "John Kirkham"
__date__ = "$Apr 9, 2014 4:00:40PM$"

import os

# Generally useful and fast to import so done immediately.
import numpy

# Need in order to have logging information no matter what.
import debugging_tools

import HDF5_logger

# Short function to process image data.
import advanced_image_processing

# For IO. Right now, just includes read_parameters for reading a config.json file.
import read_config

import HDF5_serializers

import vigra
import vigra.impex

# Get the logger
logger = debugging_tools.logging.getLogger(__name__)



@debugging_tools.log_call(logger)
def batch_generate_save_neurons(new_filenames, parameters):
    """
        Uses generate_save_neurons to process a list of filename (HDF5 files) with the given parameters for trainDL.
        Results will be saved in each file.
        
        Args:
            new_filenames     names of the files to read.
            parameters        passed directly to generate_save_neurons.
    """

    # simple. iterates over each call to generate and save results in given HDF5 file.
    for each_new_filename in new_filenames:
        # runs each one and saves results in each file
        generate_save_neurons(each_new_filename, **parameters)


@debugging_tools.log_call(logger)
def generate_save_neurons(new_filename, debug = False, resume = False, run_stage = "all", **parameters):
    """
        Uses advanced_image_processing.generate_dictionary to process a given filename (HDF5 files)
        with the given parameters for trainDL.
        
        Args:
            new_filenames     name of the internal file to read (should be a Dataset)
            parameters        passed directly to advanced_image_processing.generate_dictionary.
    """

    # No need unless loading data.
    # thus, won't be loaded if only using numpy arrays with advanced_image_processing.generate_dictionary.
    import h5py

    # Need in order to read h5py path. Otherwise unneeded.
    #import lazyflow.utility.pathHelpers as pathHelpers # Use this when merged into the ilastik framework.
    import pathHelpers


    new_filename_details = pathHelpers.PathComponents(new_filename)

    new_filename_ext = new_filename_details.extension
    new_filename_ext = new_filename_ext.lower()
    new_filename_ext = new_filename_ext.replace(os.path.extsep, "", 1)

    if ( (new_filename_ext == "h5") or (new_filename_ext == "hdf5") or (new_filename_ext == "he5") ):
        # HDF5 file. Nothing to do here.
        new_hdf5_filepath = new_filename
    else:
        raise Exception("File with filename: \"" + new_filename + "\"" + " provided with an unknown file extension: \"" + new_filename_ext + "\". If it is a supported format, please run the given file through HDF5_importer first before proceeding.")


    # Inspect path name to get where the file is and its internal path
    new_hdf5_filepath_details = pathHelpers.PathComponents(new_hdf5_filepath)

    # The name of the data without the its path
    new_hdf5_filepath_details.internalDatasetName = new_hdf5_filepath_details.internalDatasetName.strip("/")

    with h5py.File(new_hdf5_filepath_details.externalPath, "a") as new_file:
        # Must contain the internal path in question
        if new_hdf5_filepath_details.internalPath not in new_file:
            raise Exception( "The given data file \"" + new_filename + "\" does not contain \"" + new_hdf5_filepath_details.internalPath + "\".")

        # Must be a path to a h5py.Dataset not a h5py.Group (would be nice to relax this constraint)
        elif not isinstance(new_file[new_hdf5_filepath_details.internalPath], h5py.Dataset):
            raise Exception("The given data file \"" + new_filename + "\" does not contain a dataset at location \"" + new_hdf5_filepath_details.internalPath + "\".")

        # Where to read data files from
        input_directory = new_hdf5_filepath_details.internalDirectory.rstrip("/")

        # Where the results will be saved to
        output_directory = ""
        if input_directory == "":
            # if we are at the root
            output_directory = "/ADINA_results" + "/" + new_hdf5_filepath_details.internalDatasetName.rstrip("/")
        else:
            # otherwise (not at that the root)
            output_directory = input_directory + "_ADINA_results" + "/" + new_hdf5_filepath_details.internalDatasetName.rstrip("/")

        # Delete the old output directory if it exists.
        if (not resume) and (output_directory in new_file):
            # Purge the output directory.
            del new_file[output_directory]
            new_file.create_group(output_directory)
        elif output_directory not in new_file:
            # Create a new output directory.
            new_file.create_group(output_directory)

        output_group = new_file[output_directory]

        # Create a hardlink (does not copy the original data)
        if "original_images" not in new_file[output_directory]:
            output_group["original_images"] = new_file[new_hdf5_filepath_details.internalPath]

        # Copy out images for manipulation in memory
        new_images = output_group["original_images"][:]

        # Get a debug logger for the HDF5 file (if needed)
        array_debug_logger = HDF5_logger.generate_HDF5_array_logger(output_group,
                                                                    group_name = "debug",
                                                                    enable = debug,
                                                                    overwrite_group = False)

        # Saves intermediate result to make resuming easier
        resume_logger = HDF5_logger.generate_HDF5_array_logger(output_group, allow_overwrite_dataset = True)

        if "original_images_max_projection" not in output_group:
            array_debug_logger("original_images_max_projection", new_images.max(axis = 0))

        # Preprocess images
        new_preprocessed_images = None
        if ("preprocessed_images" in resume_logger) and (run_stage != "preprocessing"):
            new_preprocessed_images = resume_logger["preprocessed_images"]
        else:
            new_preprocessed_images = advanced_image_processing.preprocess_data(new_images,
                                                                                array_debug_logger = array_debug_logger,
                                                                                **parameters["preprocess_data"])
            resume_logger("preprocessed_images", new_preprocessed_images)

            if "preprocessed_images_max_projection" not in array_debug_logger:
                array_debug_logger("preprocessed_images_max_projection", new_preprocessed_images.max(axis = 0))

        if run_stage == "preprocessing":
            return

        # Find the dictionary
        new_dictionary = None
        if ("dictionary" in resume_logger) and (run_stage != "dictionary"):
            new_dictionary = resume_logger["dictionary"]
        else:
            new_dictionary = advanced_image_processing.generate_dictionary(new_preprocessed_images,
                                                                           array_debug_logger = array_debug_logger,
                                                                           **parameters["generate_dictionary"])
            resume_logger("dictionary", new_dictionary)

            if "dictionary_max_projection" not in array_debug_logger:
                array_debug_logger("dictionary_max_projection", new_dictionary.max(axis = 0))

        if run_stage == "dictionary":
            return

        # Find the neurons
        new_neurons = None
        if ("neurons" in resume_logger) and (run_stage != "postprocessing"):
            new_neurons = resume_logger["neurons"]
        else:
            new_neurons = advanced_image_processing.postprocess_data(new_dictionary,
                                                                     array_debug_logger,
                                                                     **parameters["postprocess_data"])
            resume_logger("neurons", new_neurons)

        if new_neurons.size == 0:
            logger.warning("No neurons were found in the data.")

        # Save the configuration parameters in the attributes as a string.
        if "parameters" not in output_group.attrs:
            # Write the configuration parameters in the attributes as a string.
            output_group.attrs["parameters"] = repr(parameters)


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
    parser.add_argument("input_files", metavar = "INPUT_FILE", type = str, nargs = '+',
                        help = "HDF5 file(s) to process (a single dataset or video will be expected in /images (time must be the first dimension) the results will be placed in /results (will overwrite old data) of the respective file with attribute tags related to the parameters used).")

    # Results of parsing arguments (ignore the first one as it is the command line call).
    parsed_args = parser.parse_args(argv[1:])

    # Go ahead and stuff in parameters with the other parsed_args
    parsed_args.parameters = read_config.read_parameters(parsed_args.config_filename)

    # Runs the dictionary learning algorithm on each file with the given parameters
    # and saves the results in the given files.
    batch_generate_save_neurons(parsed_args.input_files, parsed_args.parameters)

    return(0)


if __name__ == "__main__":
    # only necessary if running main (normally if calling command line). no point in importing otherwise.
    import sys

    # call main if the script is loaded from command line. otherwise, user can import package without main being called.
    sys.exit(main(*sys.argv))
