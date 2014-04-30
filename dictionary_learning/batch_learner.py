#!/usr/bin/env python

__author__="John Kirkham"
__date__ ="$Apr 9, 2014 4:00:40 PM$"


# Generally useful and fast to import so done immediately.
import numpy

# Need in order to have logging information no matter what.
import advanced_debugging

# For IO. Right now, just includes read_parameters for reading a config.json file.
import read_config


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)



@advanced_debugging.log_call(logger, print_args = True)
def batch_generate_save_dictionary(*new_filenames, **parameters):
    """
        Uses generate_save_dictionary to process a list of filename (HDF5 files) with the given parameters for trainDL.
        Results will be saved in each file.
        
        Args:
            new_filenames     names of the files to read.
            parameters        passed directly to generate_save_dictionary.
    """
    
    # simple. iterates over each call to generate and save results in given HDF5 file.
    for each_new_filename in new_filenames:
        # runs each one and saves results in each file
        generate_save_dictionary(each_new_filename, **parameters)


@advanced_debugging.log_call(logger, print_args = True)
def generate_save_dictionary(new_filename, **parameters):
    """
        Uses generate_dictionary to process a given filename (HDF5 files) with the given parameters for trainDL.
        
        Args:
            new_filenames     name of the internal file to read (should be a Dataset)
            parameters        passed directly to generate_dictionary.
    """
    
    # No need unless loading data. thus, won't be loaded if only using numpy arrays with generate_dictionary.
    import h5py
    
    # Need in order to read h5py path. Otherwise unneeded.
    #import lazyflow.utility.pathHelpers as pathHelpers # Use this when merged into the ilastik framework.
    import pathHelpers
    
    # Inspect path name to get where the file is an its internal path
    new_filename_details = pathHelpers.PathComponents(new_filename)
    
    # The name of the data without the its path
    new_filename_details.internalDatasetName = new_filename_details.internalDatasetName.strip("/")
    
    with h5py.File(new_filename_details.externalPath, "a") as new_file:
        # Must contain the internal path in question
        if new_filename_details.internalPath not in new_file:
            raise Exception("The given data file \"" + new_filename + "\" does not contain \"" + new_filename_details.internalPath + "\".")
        
        # Must be a path to a h5py.Dataset not a h5py.Group (would be nice to relax this constraint)
        elif not isinstance(new_file[new_filename_details.internalPath], h5py.Dataset):
            raise Exception("The given data file \"" + new_filename + "\" does not not contain a dataset for \"" + new_filename_details.internalPath + "\".")
        
        # Where to read data files from
        input_directory = new_filename_details.internalDirectory.rstrip("/")
        
        # Where the results will be saved to
        output_directory = ""
        
        if input_directory == "":
            # if we are at the root
            output_directory = "/ADINA_results" + "/" + new_filename_details.internalDatasetName.rstrip("/")
        else:
            # otherwise (not at that the root)
            output_directory = input_directory + "_ADINA_results" + "/" + new_filename_details.internalDatasetName.rstrip("/")
        
        # If the group does not exist, make it.
        if not output_directory in new_file:
            new_file.create_group(output_directory)
        
        # Remove a link to the original data if it already exists
        if "original_data" in new_file[output_directory]:
            del new_file[output_directory]["original_data"]
        
        # Remove a the old dictionary data if it already exists
        if "dictionary" in new_file[output_directory]:
            del new_file[output_directory]["dictionary"]
        
        # Create a hardlink (does not copy) the original data
        new_file[output_directory]["original_data"] = new_file[new_filename_details.internalPath]
        
        # Copy out images for manipulation in memory
        new_data = new_file[output_directory]["original_data"][:]
        
        # generates dictionary and stores results
        new_file[output_directory]["dictionary"] = generate_dictionary(new_data, **parameters)

        # stores all parameters used to generate the dictionary in results
        for parameter_key, parameter_value in parameters.items():
            new_file[output_directory]["dictionary"].attrs[parameter_key] = parameter_value


@advanced_debugging.log_call(logger)
def zeroed_mean_images(new_numpy_array):
    """
        Takes and finds the mean for each image. Where each image is new_numpy_array[i] with some index i.
        
        Args:
            new_numpy_array(numpy.ndarray):     array images with time as the first index
        
        Returns:
            result(numpy.ndarray):              The same array with each images mean removed. Where means[i] = mean(new_numpy_array[i])
        
        
        Examples:
            >>> import numpy; zeroed_mean_images(numpy.array([[0,0],[0,0]]))
            array([[ 0.,  0.],
                   [ 0.,  0.]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[6,0],[0,0]]))
            array([[ 3., -3.],
                   [ 0.,  0.]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[0,0],[0,4]]))
            array([[ 0.,  0.],
                   [-2.,  2.]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[6,0],[0,4]]))
            array([[ 3., -3.],
                   [-2.,  2.]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[1,2],[3,4]]))
            array([[-0.5,  0.5],
                   [-0.5,  0.5]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[1,2],[3,4]]))
            array([[-0.5,  0.5],
                   [-0.5,  0.5]])
    """
    
    # start with means having the same contents as the given images
    means = new_numpy_array
    
    # take the mean while we haven't gotten one mean for each image.
    while means.ndim > 1:
        means = means.mean(axis = 1)
    
    # reshape means until it has the right number of dimensions to broadcast.
    while means.ndim < new_numpy_array.ndim:
        means = means.reshape(means.shape + (1,))
    
    # broadcast and subtract the means so that the mean of all values in result[i] is zero
    result = new_numpy_array - means
    
    return(result)


@advanced_debugging.log_call(logger)
def renormalized_images(new_numpy_array, ord = 2):
    """
        Takes and finds the mean for each image. Where each image is new_numpy_array[i] with some index i.
        
        Args:
            new_numpy_array(numpy.ndarray):     array images with time as the first index
        
        Returns:
            result(numpy.ndarray):              The same array with each images mean removed. Where means[i] = mean(new_numpy_array[i])
        
        
        Examples:
            >>> import numpy; renormalized_images(numpy.array([[0,1],[1,0]]))
            array([[0, 1],
                   [1, 0]])
                   
            >>> import numpy; renormalized_images(numpy.array([[0.,2.],[1.,0.]]))
            array([[ 0.,  1.],
                   [ 1.,  0.]])
                   
            >>> import numpy; renormalized_images(numpy.array([[2.,2.],[1.,0.]]))
            array([[ 0.70710678,  0.70710678],
                   [ 1.        ,  0.        ]])
                   
            >>> import numpy; renormalized_images(numpy.array([[1.,2.],[3.,4.]]))
            array([[ 0.4472136 ,  0.89442719],
                   [ 0.6       ,  0.8       ]])
                   
            >>> import numpy; renormalized_images(numpy.array([[1.,2.],[3.,4.]]), ord = 1)
            array([[ 0.33333333,  0.66666667],
                   [ 0.42857143,  0.57142857]])
    """
    
    # must copy as we will change contents
    result = new_numpy_array.copy()
    
    # take each image at each time turn the image into a vector and find the norm.
    # divide each image by this norm. (required for spams.trainDL)
    for i in xrange(result.shape[0]):
        result[i] /= numpy.linalg.norm(result[i].ravel(), ord = ord)
    
    return(result)



@advanced_debugging.log_call(logger, print_args = True)
def generate_dictionary(new_data, **parameters):
    """
        Generates a dictionary using the data and parameters given for trainDL.
        
        Args:
            new_data(numpy.ndarray):      array of data for generating a dictionary (first axis is time).
            parameters(dict):             passed directly to spams.trainDL.
        
        Note:
            Todo
            Look into move data normalization into separate method (have method chosen by config file).
        
        Returns:
            dict: the dictionary found.
    """
    
    # It takes a loooong time to load spams. so, we shouldn't do this until we are sure that we are ready to generate the dictionary
    # (i.e. the user supplied a bad config file, /images does not exist, etc.).
    # Note it caches the import so subsequent calls should not make it any slower.
    import spams

    # Maybe should copy data so as not to change the original.
    # new_data_processed = new_data.copy()
    new_data_processed = new_data
    
    # Remove the mean of each row vector
    new_data_processed = zeroed_mean_images(new_data_processed)

    # Renormalize each row vector using L_2
    # Unfortunately our version of numpy's function numpy.linalg.norm does not support the axis keyword. So, we must use a for loop.
    new_data_processed = renormalized_images(new_data_processed, ord = 2)

    # Reshape data into a matrix (each image is now a column vector)
    new_data_processed = numpy.reshape(new_data_processed, [new_data_processed.shape[0], -1])
    new_data_processed = numpy.asmatrix(new_data_processed)
    new_data_processed = new_data_processed.transpose()

    # Spams requires all matrices to be fortran.
    new_data_processed = numpy.asfortranarray(new_data_processed)
    
    # Simply trains the dictionary. Does not return sparse code.
    # Need to look into generating the sparse code given the dictionary, spams.nmf? (may be too slow))
    new_dictionary = spams.trainDL(new_data_processed, **parameters)

    # Fix dictionary so that the first index will be the particular image.
    # The rest will be the shape of an image (same as input shape).
    new_dictionary = new_dictionary.transpose()
    new_dictionary = numpy.asarray(new_dictionary).reshape((parameters["K"],) + new_data.shape[1:])[:]

    return(new_dictionary)


@advanced_debugging.log_call(logger, print_args = True)
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

    # Creates command line parser
    parser = argparse.ArgumentParser(description = "Parses input from the command line for a batch job.")

    # Takes a config file and then a series of one or more HDF5 files.
    parser.add_argument("config_filename", metavar="CONFIG_FILE", type = str, help = "JSON file that provides configuration options for how to use dictionary learning on the input files.")
    parser.add_argument("input_files", metavar="INPUT_FILE", type = str, nargs='+', help = "HDF5 file(s) to process (a single dataset or video will be expected in /images (time must be the first dimension) the results will be placed in /results (will overwrite old data) of the respective file with attribute tags related to the parameters used).")

    # Results of parsing arguments (ignore the first one as it is the command line call).
    parsed_args = parser.parse_args(argv[1:])

    # Go ahead and stuff in parameters with the other parsed_args
    # A little risky if parsed_args may later contain a parameters variable due to changing the main file
    # or argparse changing behavior; however, this keeps all arguments in the same place.
    parsed_args.parameters = read_config.read_parameters(parsed_args.config_filename)

    # Runs the dictionary learning algorithm on each file with the given parameters and saves the results in the given files.
    batch_generate_save_dictionary(*parsed_args.input_files, **parsed_args.parameters)

    return(0)



if __name__ == "__main__":
    # only necessary if running main (normally if calling command line). no point in importing otherwise.
    import sys
    
    # call main if the script is loaded from command line. otherwise, user can import package without main being called.
    sys.exit(main(*sys.argv))
