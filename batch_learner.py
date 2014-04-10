#!/usr/bin/env python



"""

@package        batch_learner

@date           Created on Wed  Apr 9, 2014  16:00:40

@author         John Kirkham

"""


# on my system need macports at present
###########!/opt/local/bin/python



# -*- coding: utf-8 -*-




# Generally useful and fast to import so done immediately.
import numpy



def read_parameters(config_filename):
    """
        @brief  Reads the contents of a json config file and returns the parameters.

        @param  config_filename     name of the file to read.
    """

    # only relevant if reading parameter file.
    import json

    # gets parameters out of the file and dumps them in the dictionary. just that simple.
    parameters = {}
    with open(config_filename, 'r') as fp:
        # will just give a dictionary. just that simple
        parameters = json.load(fp)

    return(parameters)



def batch_generate_save_dictionary(*new_filenames, **parameters):
    """
        @brief  Uses generate_save_dictionary to process a list of filename (HDF5 files) with the given parameters for trainDL. Results will be saved in each file

        @param  new_filenames     name of the files to read.
        @param  parameters        name of the file to read.
    """
    
    # simple. iterates over each call to generate and save results in given HDF5 file.
    for each_new_filename in new_filenames:
        # runs each one and saves results in each file
        generate_save_dictionary(each_new_filename, **parameters)



def generate_save_dictionary(new_filename, **parameters):
    """
        @brief  Uses generate_dictionary to process a given filename (HDF5 files) with the given parameters for trainDL.

        @param  new_filename     name of the file to read.
        @param  parameters       name of the file to read.
    """

    # no need unless loading data. thus, won't be loaded if only using numpy arrays with generate_dictionary.
    import h5py

    with h5py.File(new_filename, "a") as new_file:
        if "/images" not in new_file:
            # need /images in order to process (would be good to relax this constraint somehow
            raise IOError("The given data file \"" + new_filename + "\" does not contain \"/images\".")
        if "/results" in new_file:
            # if there are results, we will overwrite them (perhaps could use a date stamp instead).
            del new_file["/results"]

        # copy out images for manipulation in memory
        new_data = new_file["/images"][:]
        
        # generates dictionary and stores results
        new_file["/results"] = generate_dictionary(new_data, **parameters)

        # stores all parameters used to generate the dictionary in results
        for parameter_key, parameter_value in parameters.items():
            new_file["/results"].attrs[parameter_key] = parameter_value



def generate_dictionary(new_data, **parameters):
    """
        @brief  Uses generate_dictionary to process a given filename (HDF5 files) with the given parameters for trainDL.

        @param  new_filename     name of the file to read.
        @param  parameters       name of the file to read.
        
        @todo   look into move data normalization into separate method (have method chosen by config file).
    """
    
    # it takes a loooong time to load spams. so, we shouldn't do this until we are sure that we are ready to generate the dictionary (i.e. the user supplied a bad config file, /images does not exist, etc.). note it caches the import so subsequent calls should not make it any slower.
    import spams

    # maybe should copy data so as not to change the original
    # new_data_processed = new_data[:]
    new_data_processed = new_data

    # reshape data into a matrix (each image is now a column vector)
    new_data_processed = numpy.asmatrix(numpy.reshape(new_data_processed, [new_data_processed.shape[0], -1])).transpose()

    # remove the mean of each row vector
    new_data_processed -= new_data_processed.mean(axis = 0)

    # renormalize each row vector using L_2
    new_data_processed /= numpy.tile(numpy.linalg.norm(new_data_processed, axis = 0).reshape((1, -1)), (new_data_processed.shape[0], 1))

    # spams requires all matrices to be fortran
    new_data_processed = numpy.asfortranarray(new_data_processed)
    
    # simply trains the dictionary (does not return sparse code, need to look into generating the sparse code given the dictionary, spams.nmf? (may be too slow))
    new_dictionary = spams.trainDL(new_data_processed, **parameters)

    # fix dictionary so that the first index will be the particular image and the rest will be the shape of an image (same as input shape)
    new_dictionary = new_dictionary.transpose()
    new_dictionary = numpy.asarray(new_dictionary).reshape((parameters["K"],) + new_data.shape[1:])[:]

    return(new_dictionary)



def main(*argv):
    """
        @brief  Simple main function (like in C). Takes all arguments (as from sys.argv) and returns an exit status.

        @param  argv     arguments (includes command line call).
    """
    
    # only necessary if running main (normally if calling command line). no point in importing otherwise.
    import argparse

    # creates command line parser
    parser = argparse.ArgumentParser(description = "Parses input from the command line for a batch job.")

    # Takes a config file and then a series of one or more HDF5 files.
    parser.add_argument("config_filename", metavar="CONFIG_FILE", type = str, help = "JSON file that provides configuration options for how to use dictionary learning on the input files.")
    parser.add_argument("input_files", metavar="INPUT_FILE", type = str, nargs='+', help = "HDF5 file(s) to process (a single dataset or video will be expected in /images (time must be the first dimension) the results will be placed in /results (will overwrite old data) of the respective file with attribute tags related to the parameters used).")

    # Results of parsing arguments (ignore the first one as it is the command line call).
    parsed_args = parser.parse_args(argv[1:])

    # Go ahead and stuff in parameters with the other parsed_args (a little risky if parsed_args may later contain a parameters variable due to changing the main file or argparse changing behavior; however, this keeps all arguments in the same place.)
    parsed_args.parameters = read_parameters(parsed_args.config_filename)

    # Runs the dictionary learning algorithm on each file with the given parameters and saves the results in the given files.
    batch_generate_save_dictionary(*parsed_args.input_files, **parsed_args.parameters)

    return(0)



if __name__ == "__main__":
    # only necessary if running main (normally if calling command line). no point in importing otherwise.
    import sys
    
    # call main if the script is loaded from command line. otherwise, user can import package without main being called.
    sys.exit(main(*sys.argv))