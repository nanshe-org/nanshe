# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__="John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ ="$Apr 30, 2014 5:23:37 PM$"



# Generally useful and fast to import so done immediately.
import numpy

# For image processing.
import scipy
import scipy.ndimage
import scipy.spatial
import scipy.spatial.distance

# More image processing...
import skimage
import skimage.measure
import skimage.feature
import skimage.morphology
import skimage.segmentation

# To allow for more advanced iteration pattersn
import itertools

# Allows for deep and shallow copies.
import copy

# Need for opening
import vigra
import vigra.filters

# Need in order to have logging information no matter what.
import advanced_debugging

import advanced_numpy

# Short function to process image data.
import simple_image_processing

# To remove noise from the basis images
import denoising

# Wavelet transformation operations
import wavelet_transform


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)


@advanced_debugging.log_call(logger)
def normalize_data(new_data, **parameters):
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
    
    # TODO: Add preprocessing step wavelet transform, F_0, etc.
    
    # Maybe should copy data so as not to change the original.
    # new_data_processed = new_data.copy()
    new_data_processed = new_data
    
    # Remove the mean of each row vector
    new_data_processed = simple_image_processing.zeroed_mean_images(new_data_processed)
    
    # Renormalize each row vector using some specified normalization
    new_data_processed = simple_image_processing.renormalized_images(new_data_processed, **parameters["renormalized_images"])
    
    return(new_data_processed)


@advanced_debugging.log_call(logger)
def run_multiprocessing_queue_spams_trainDL(out_queue, *args, **kwargs):
    """
        Designed to run spams.trainDL in a seperate process.
        
        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the 
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.
        
        This particular version uses a multiprocessing.Queue to return the resulting dictionary.
        
        
        Args:
            out_queue(multiprocessing.Queue):       what will take the returned dictionary from spams.trainDL.
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            *kwargs(dict):                          a dictionary of keyword arguments to pass to spams.trainDL.
        
        Note:
            Todo
            Look into having the raw data for input for spams.trainDL copied in.
    """
    
    # It is not needed outside of calling spams.trainDL.
    # Also, it takes a long time to load this module.
    import spams
    
    result = spams.trainDL(*args, **kwargs)
    out_queue.put(result)
    

@advanced_debugging.log_call(logger)
def call_multiprocessing_queue_spams_trainDL(*args, **kwargs):
    """
        Designed to start spams.trainDL in a seperate process and handle the result in an unnoticeably different way.
        
        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the 
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.
        
        This particular version uses a multiprocessing.Queue to return the resulting dictionary.
        
        
        Args:
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            *kwargs(dict):                          a dictionary of keyword arguments to pass to spams.trainDL.
        
        Note:
            Todo
            Look into having the raw data for input for spams.trainDL copied in.
        
        Returns:
            result(numpy.matrix): the dictionary found
    """
    
    # Only necessary for dealing with SPAMS
    import multiprocessing
    
    out_queue = multiprocessing.Queue()
    
    p = multiprocessing.Process(target = run_multiprocessing_queue_spams_trainDL, args = (out_queue,) + args, kwargs = kwargs)
    p.start()
    result = out_queue.get()
    p.join()
    
    return(result)


@advanced_debugging.log_call(logger)
def run_multiprocessing_array_spams_trainDL(output_array, *args, **kwargs):
    """
        Designed to start spams.trainDL in a seperate process and handle the result in an unnoticeably different way.
        
        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the 
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.
        
        This particular version uses a multiprocessing.Array to share memory to return the resulting dictionary.
        
        
        Args:
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            *kwargs(dict):                          a dictionary of keyword arguments to pass to spams.trainDL.
        
        Note:
            This is somewhat faster than using multiprocessing.Queue.
            
            Todo
            Need to deal with return_model case.
            Look into having the raw data for input for spams.trainDL copied in.
    """
    
    # Just to make sure this exists in the new process. Shouldn't be necessary.
    import numpy
    # Just to make sure this exists in the new process. Shouldn't be necessary.
    # Also, it is not needed outside of calling this function.
    import spams
    
    # Create a numpy.ndarray that uses the shared buffer.
    result = numpy.frombuffer(output_array.get_obj(), dtype = float).reshape((-1, kwargs["K"]))
    result = numpy.asmatrix(result)
    
    result[:] = spams.trainDL(*args, **kwargs)
    

@advanced_debugging.log_call(logger)
def call_multiprocessing_array_spams_trainDL(X, *args, **kwargs):
    """
        Designed to start spams.trainDL in a seperate process and handle the result in an unnoticeably different way.
        
        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the 
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.
        
        This particular version uses a multiprocessing.Array to share memory to return the resulting dictionary.
        
        
        Args:
            X(numpy.matrix)                         a Fortran order NumPy Matrix with the same name as used by spams.trainDL (so if someone tries to use it as a keyword argument...).
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            **kwargs(dict):                         a dictionary of keyword arguments to pass to spams.trainDL.
        
        Note:
            This is somewhat faster than using multiprocessing.Queue.
            
            Todo
            Need to deal with return_model case.
            Look into having the raw data for input for spams.trainDL copied in.
    """
    
    # Only necessary for dealing with SPAMS
    import multiprocessing
    # Only necessary for dealing with multiprocessing.Array for SPAMS
    import ctypes
    # Just to make sure this exists in the new process. Shouldn't be necessary.
    import numpy
    
    output_array_size = X.shape[0] * kwargs["K"]
    output_array = multiprocessing.Array(ctypes.c_double, output_array_size)
    
    p = multiprocessing.Process(target = run_multiprocessing_array_spams_trainDL, args = (output_array, X,) + args, kwargs = kwargs)
    p.start()
    p.join()
    
    result = numpy.frombuffer(output_array.get_obj(), dtype = float).reshape((-1, kwargs["K"]))
    result = result.copy()
    
    return(result)


@advanced_debugging.log_call(logger)
def call_spams_trainDL(*args, **kwargs):
    """
        Encapsulates call to spams.trainDL. Ensures copy of results occur just in case.
        Designed to be like the multiprocessing calls.
        
        Args:
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            **kwargs(dict):                         a dictionary of keyword arguments to pass to spams.trainDL.
        
        Note:
            For legacy.
    """
    
    result = spams.trainDL(*args, **kwargs)
    result = result.copy()
    
    return(result)
    

@advanced_debugging.log_call(logger)
def generate_dictionary(new_data, **parameters):
    """
        Generates a dictionary using the data and parameters given for trainDL.
        
        Args:
            new_data(numpy.ndarray):      array of data for generating a dictionary (first axis is time).
            parameters(dict):             passed directly to spams.trainDL.
        
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

    # Reshape data into a matrix (each image is now a column vector)
    new_data_processed = numpy.reshape(new_data_processed, [new_data_processed.shape[0], -1])
    new_data_processed = numpy.asmatrix(new_data_processed)
    new_data_processed = new_data_processed.transpose()

    # Spams requires all matrices to be fortran.
    new_data_processed = numpy.asfortranarray(new_data_processed)
    
    # Simply trains the dictionary. Does not return sparse code.
    # Need to look into generating the sparse code given the dictionary, spams.nmf? (may be too slow))
    new_dictionary = call_multiprocessing_array_spams_trainDL(new_data_processed, **parameters["spams_trainDL"])

    # Fix dictionary so that the first index will be the particular image.
    # The rest will be the shape of an image (same as input shape).
    new_dictionary = new_dictionary.transpose()
    new_dictionary = numpy.asarray(new_dictionary).reshape((parameters["spams_trainDL"]["K"],) + new_data.shape[1:])
    
    return(new_dictionary)


@advanced_debugging.log_call(logger)
def region_properties(new_label_image, *args, **kwargs):
    """
        Grabs region properties from a label .
        
        Args:
            new_data(numpy.ndarray):      array of data for generating a dictionary (first axis is time).
            parameters(dict):             passed directly to spams.trainDL.
        
        Returns:
            dict: the dictionary found.
        
        
        Examples:
            
            >>> region_properties(numpy.zeros((2,2), dtype=int))
            array([], 
                  dtype=[('Centroid', 'O'), ('Area', 'O')])
            
            >>> region_properties(numpy.ones((2,2), dtype=int))
            array([(4.0, [0.5, 0.5], 1)], 
                  dtype=[('Area', '<f8'), ('Centroid', '<f8', (2,)), ('Label', '<i8')])
            
            >>> region_properties(numpy.ones((3,3), dtype=int))
            array([(9.0, [1.0, 1.0], 1)], 
                  dtype=[('Area', '<f8'), ('Centroid', '<f8', (2,)), ('Label', '<i8')])
    """
    
    logger.debug(repr(new_label_image))
    logger.debug(repr(numpy.unique(new_label_image)))
    
    fixed_shape_array_values = [ "BoundingBox", "CentralMoments", "Centroid", "HuMoments", "Moments", "NormalizedMoments", "WeightedCentralMoments", "WeightedCentroid", "WeightedHuMoments", "WeightedMoments", "WeightedNormalizedMoments" ]
    all_array_values = [ "BoundingBox", "CentralMoments", "Centroid", "ConvexImage", "Coordinates", "FilledImage", "HuMoments", "Image", "Moments", "NormalizedMoments", "WeightedCentralMoments", "WeightedCentroid", "WeightedHuMoments", "WeightedMoments", "WeightedNormalizedMoments" ]
    
    new_label_image_props = None
    new_label_image_props_with_arrays = None
    new_label_image_props_with_arrays_values = None
    new_label_image_props_with_arrays_dtype = None
    
    properties = None
    if (len(args)) and (args[0]):
        properties = args[0]
        args = args[1:]
    elif (len(kwargs)) and ("properties" in kwargs):
        properties = kwargs["properties"]
        del kwargs["properties"]
    else:
        properties = ["Area", "Centroid"]
    
    if properties == "all":
        properties = ["Area",
                      "Coordinates",
                      "ConvexArea",
                      "Centroid",
                      "EquivDiameter",
                      "Perimeter",
                      "CentralMoments",
                      "Solidity",
                      "EulerNumber",
                      "Extent",
                      "NormalizedMoments",
                      "Eccentricity",
                      "ConvexImage",
                      "FilledImage",
                      "Orientation",
                      "MajorAxisLength",
                      "Moments",
                      "Image",
                      "FilledArea",
                      "BoundingBox",
                      "MinorAxisLength",
                      "HuMoments"]
    
    properties = ["Label"] + properties
    
    if new_label_image.size:
        # This gives a list of dictionaries. However, this is not very usable. So, we will convert this to a structured NumPy array.
        new_label_image_props = skimage.measure.regionprops(new_label_image, properties, *args, **kwargs)
        
        new_label_image_props_with_arrays = []
        for i in xrange(len(new_label_image_props)):
            new_label_image_props_with_arrays.append({})
            
            for each_key in properties:
                if each_key in all_array_values:
                    new_label_image_props_with_arrays[i][each_key] = numpy.array(new_label_image_props[i][each_key])
                else:
                    new_label_image_props_with_arrays[i][each_key] = new_label_image_props[i][each_key]
        
        #print(repr(new_label_image_props_with_arrays))
        #print("")
        
        # Holds the values from props.
        new_label_image_props_with_arrays_values = []

        # Holds the types from props.
        new_label_image_props_with_arrays_dtype = []

        if len(new_label_image_props_with_arrays):
            # Get types for all properties as a dictionary
            for each_name in properties:
                each_sample_value = new_label_image_props_with_arrays[0][each_name]
                each_type = type(each_sample_value)
                each_shape = tuple()
                
                if (each_type is numpy.ndarray) and (each_name in fixed_shape_array_values):
                    each_type = each_sample_value.dtype
                    each_shape = each_sample_value.shape
                else:
                    each_type = numpy.dtype(each_type)
                    
                new_label_image_props_with_arrays_dtype.append( (each_name, each_type, each_shape) )

            # Store the values to place in NumPy structured array in order.
            new_label_image_props_with_arrays_values = []
            for j in xrange(len(new_label_image_props_with_arrays)):
                # Add all values in order of keys from the dictionary.
                new_label_image_props_with_arrays_values.append([])
                for each_new_label_image_props_with_arrays_dtype in new_label_image_props_with_arrays_dtype:
                    
                    each_name, each_type, each_shape = each_new_label_image_props_with_arrays_dtype
                    
                    if each_shape:
                        new_label_image_props_with_arrays_values[j].append(new_label_image_props_with_arrays[j][each_name].tolist())
                    else:
                        new_label_image_props_with_arrays_values[j].append(new_label_image_props_with_arrays[j][each_name])

                # NumPy will expect a tuple for each set of values.
                new_label_image_props_with_arrays_values[j] = tuple(new_label_image_props_with_arrays_values[j])

    if (not new_label_image.size) or (not len(new_label_image_props_with_arrays)):
        new_label_image_props_with_arrays_dtype = []
        for each_key in properties:
            if each_key in [ "BoundingBox", "Centroid", "HuMoments", "WeightedCentroid", "WeightedHuMoments" ]:
                new_label_image_props_with_arrays_dtype.append( (each_key, numpy.dtype(numpy.object), new_label_image.ndim) )
            else:
                new_label_image_props_with_arrays_dtype.append( (each_key, numpy.dtype(numpy.object)) )
    
    new_label_image_props_with_arrays_dtype = numpy.dtype(new_label_image_props_with_arrays_dtype)

    # Replace the properties with the structured array.
    new_label_image_props_with_arrays = numpy.array(new_label_image_props_with_arrays_values, dtype = new_label_image_props_with_arrays_dtype)
    
    return(new_label_image_props_with_arrays)


@advanced_debugging.log_call(logger)
def get_neuron_dtype(new_image):
    neurons_dtype = [("mask", bool, new_image.shape),
                    #("image", new_image.dtype, new_image.shape),
                    ("image_original", new_image.dtype, new_image.shape),
                    #("segmentation", new_wavelet_image_denoised.dtype, new_wavelet_image_denoised.shape),
                    ("area", float),
                    ("max_F", float),
                    ("gaussian_mean", float, (new_image.ndim,)),
                    ("gaussian_cov", float, (new_image.ndim, new_image.ndim,)),
                    ("centroid", new_image.dtype, (new_image.ndim,))]
    
    return(neurons_dtype)

@advanced_debugging.log_call(logger)
def get_empty_neuron(new_image):
    neurons_dtype = get_neuron_dtype(new_image)
    neurons = numpy.zeros((0,), dtype = neurons_dtype)
    
    return(neurons)

@advanced_debugging.log_call(logger)
def get_one_neuron(new_image):
    neurons_dtype = get_neuron_dtype(new_image)
    neurons = numpy.zeros((1,), dtype = neurons_dtype)
    
    return(neurons)

class LabelImageCentroidProps(object):
    @advanced_debugging.log_call(logger)
    def __init__(self, new_intensity_image, new_label_image, **parameters):
        self.intensity_image = new_intensity_image
        self.label_image = new_label_image
        self.props = None
        self.count = None
    
        logger.debug("Found new label image.")

        logger.debug("Finding the local maxima...")

        # Would be good to use peak_local_max as it has more features and is_local_maximum is removed in later versions,
        # but it requires skimage 0.8.0 minimum.
        local_maxima_neighborhood = numpy.ones((2 * parameters["local_max_neighborhood_size"] + 1,) * new_intensity_image.ndim)
        local_maxima_mask = skimage.feature.peak_local_max(new_intensity_image, footprint = local_maxima_neighborhood, labels = (new_intensity_image > 0).astype(int), indices = False)

        logger.debug("Found the local maxima.")

        logger.debug("Labeling the local maxima...")

        # Group local maxima. Also, we don't care about differentiating them. If there are several local maxima touching, we only want one.
        # Note, if they are touching, they must be on a plateau (i.e. all have the same value).
        local_maxima_labeled = scipy.ndimage.label(local_maxima_mask.astype(int))[0]
        # Renumber all labels sequentially
        print(repr(numpy.unique(local_maxima_labeled)))
        local_maxima_labeled = skimage.segmentation.relabel_sequential(local_maxima_labeled)[0].copy()
        print(repr(numpy.unique(local_maxima_labeled)))


        logger.debug("Labeled the local maxima.")

        logger.debug("Extracting properties from the local maxima.")

        # Extract the centroids.

        print("local_maxima_labeled = " + repr(local_maxima_labeled))

        self.props = region_properties(local_maxima_labeled, properties = ["Centroid"])

        print("local_maxima_labeled = " + repr(local_maxima_labeled))
        print("self.props = " + repr(self.props))

        logger.debug("Extracted properties from the local maxima.")

        logger.debug("Refinining properties for local maxima...")

        # We want to have a few more type present in our NumPy structured array. To do this, we collect the existing types into
        # a list and then add our new types onto the end. Finally, we make the new structured array type from the list we have.
        local_maxima_labeled_props_dtype = []

        #for each_name in self.props.dtype.names:
        #    local_maxima_labeled_props_dtype.append( (each_name, self.props[each_name].dtype, self.props[each_name].shape[1:]) )

        local_maxima_labeled_props_dtype.append( ("Label", int) )
        local_maxima_labeled_props_dtype.append( ("Centroid", float, new_intensity_image.ndim) )
        local_maxima_labeled_props_dtype.append( ("IntCentroid", int, new_intensity_image.ndim) )
        local_maxima_labeled_props_dtype.append( ("IntCentroidWaveletValue", new_intensity_image.dtype) )

        local_maxima_labeled_props_dtype = numpy.dtype(local_maxima_labeled_props_dtype)

        # Makes a new properties array that contains enough entries to hold the old one and has all the types we desire.
        new_local_maxima_labeled_props = numpy.zeros(self.props.shape, dtype = local_maxima_labeled_props_dtype)

        # Copy over the old values.
        for each_name in self.props.dtype.names:
            new_local_maxima_labeled_props[each_name] = self.props[each_name].copy()

        # Replace the old structured array with the enlarged version.
        self.props = new_local_maxima_labeled_props


        # Get integers close to local max
        self.props["IntCentroid"] = self.props["Centroid"].round().astype(int)

        # Stores the value from wavelet denoising at the centroid for easy retrieval
        if self.props["IntCentroidWaveletValue"].size:
            self.props["IntCentroidWaveletValue"] = new_label_image[ tuple(self.props["IntCentroid"].T) ]

        # Overwrite the label parameter as it holds no information as it is always 1, Now, is the label from wavelet mask label image.
        #if self.props["Label"].size:
        #    self.props["Label"] = new_wavelet_mask_labeled[ tuple(self.props["IntCentroid"].T) ]

        if (numpy.any(self.props["Label"] == 0)):
            # There shouldn't be any maximums in the background. This should never happen.
            logger.warning("Maximum found where Label is 0.")

        logger.debug("self.props = " + repr(self.props))

        logger.debug("Refinined properties for local maxima.")

        print(self.props)
        
        # Overwrite the label parameter as it holds no information. Now, is the label from wavelet mask label image.
        if self.props["Label"].size:
            self.props["Label"] = new_label_image[ self.get_centroid_index_array() ]

        if (numpy.any(self.props["Label"] == 0)):
            # There shouldn't be any maximums in the background. This should never happen.
            logger.warning("Maximum found where Label is 0.")
        
        # Stores the number of times a particular label maxima appears.
        self.count = numpy.zeros( (new_label_image.max(),), dtype = [("Label", int), ("Count", int)] )
        # Get all the labels used in the label image
        self.count["Label"] = numpy.arange(1, new_label_image.max() + 1)
        # Get the count of those labels (excluding zero as it is background)
        self.count["Count"] = numpy.bincount(self.props["Label"], minlength = len(self.count) + 1)[1:]
        
        logger.debug("self.props = " + repr(self.props))
        logger.debug("self.count = " + repr(self.count))

        if numpy.any(self.count["Count"] == 0):
            # All labels should have a local maximum. If they don't, this could be a problem.

            failed_labels_list = self.count["Label"][self.count["Count"] == 0].tolist()
            failed_labels_list = [str(_) for _ in failed_labels_list]
            failed_label_msg = "Label(s) not found in local maxima. For labels = " + repr(failed_labels_list) + "."

            logger.warning(failed_label_msg)
        
        logger.debug("Refinined properties for local maxima.")
    
    @advanced_debugging.log_call(logger)
    def get_centroid_index_array(self):
        return(tuple(self.props["IntCentroid"].T))
    
    @advanced_debugging.log_call(logger)
    def get_centroid_mask(self):
        # Returns a label image containing each centroid and its labels.
        new_centroid_mask = numpy.zeros(self.label_image.shape, dtype = self.label_image.dtype)
        
        # Set the given centroids to be the same as their labels
        new_centroid_mask[self.get_centroid_index_array()] = True
        
        return(new_centroid_mask)
    
    @advanced_debugging.log_call(logger)
    def get_centroid_label_image(self):
        # Returns a label image containing each centroid and its labels.
        new_centroid_label_image = numpy.zeros(self.label_image.shape, dtype = self.label_image.dtype)
        
        # Set the given centroids to be the same as their labels
        new_centroid_label_image[self.get_centroid_index_array()] = self.props["Label"]
        
        return(new_centroid_label_image)
    
    @advanced_debugging.log_call(logger)
    def get_active_label_image(self):
        # Returns a label image containing the labels, which still have centroids.
        
        # Mask over self.count of labels that still have centroid(s)
        active_label_count_mask = self.count["Count"] > 0
        # Labels that still have centroid(s)
        active_labels = self.count["Label"][active_label_count_mask]
        # Mask for each label as to whether it appears in the self.label_image
        #every_active_label_mask = advanced_numpy.all_permutations_equal(active_labels, self.label_image)
        # Mask for self.label_image that matches any relevant label.
        #active_label_mask = every_active_label_mask.any(axis = 0)
        active_label_mask = advanced_numpy.contains(self.label_image, active_labels)
        # The label image of relevant labels
        #new_active_label_image = active_label_mask * self.label_image
        
        
        new_active_label_image = skimage.morphology.watershed(active_label_mask.astype(int), self.get_centroid_label_image(), mask = active_label_mask)
                
        
        return(new_active_label_image)
    
    @advanced_debugging.log_call(logger)
    def remove_prop_mask(self, remove_prop_indices_mask):
        # Get the labels to remove
        remove_labels = self.props["Label"][remove_prop_indices_mask]
        # Get how many of each label to remove
        label_count_to_remove = numpy.bincount(remove_labels, minlength = len(self.count) + 1)[1:]
        
        print("remove_labels = " + repr(remove_labels))
        print("label_count_to_remove = " + repr(label_count_to_remove))
        
        # Take a subset of the label props that does not include the removal mask
        self.props = self.props[ ~remove_prop_indices_mask ].copy()
        # Reduce the count by the number of each label
        self.count["Count"] -= label_count_to_remove
    
    @advanced_debugging.log_call(logger)
    def remove_prop_indices(self, *i):
        # A mask of the indices to remove
        remove_prop_indices_mask = numpy.zeros( (len(self.props),), dtype = bool )
        remove_prop_indices_mask[numpy.array(i)] = True
        
        self.remove_prop_mask(remove_prop_indices_mask)


@advanced_debugging.log_call(logger)
def remove_low_intensity_local_maxima(local_maxima, **parameters):
    # Deleting local maxima that does not exceed the 90th percentile of the pixel intensities
    low_intensities__local_maxima_label_mask__to_remove = numpy.zeros(local_maxima.props.shape, dtype = bool)
    for i in xrange(len(local_maxima.props)):
        # Get the region with the label matching the maximum
        each_region_image_wavelet_mask = (local_maxima.label_image == local_maxima.props["Label"][i])
        each_region_image_wavelet = local_maxima.intensity_image[each_region_image_wavelet_mask]

        # Get the number of pixels in that region
        each_region_image_wavelet_num_pixels = float(each_region_image_wavelet.size)

        # Get the value of the max for that region
        each_region_image_wavelet_centroid_value = local_maxima.props["IntCentroidWaveletValue"][i]

        # Get a mask of the pixels below that max for that region
        each_region_image_wavelet_num_pixels_below_max = float((each_region_image_wavelet < each_region_image_wavelet_centroid_value).sum())

        # Get a ratio of the number of pixels below that max for that region
        each_region_image_wavelet_ratio_pixels = each_region_image_wavelet_num_pixels_below_max / each_region_image_wavelet_num_pixels

        # If the ratio clears our threshhold, keep this label. Otherwise, eliminate it.
        low_intensities__local_maxima_label_mask__to_remove[i] = (each_region_image_wavelet_ratio_pixels < parameters["percentage_pixels_below_max"])

    new_local_maxima = copy.deepcopy(local_maxima)

    new_local_maxima.remove_prop_mask(low_intensities__local_maxima_label_mask__to_remove)
    
    return(new_local_maxima)


@advanced_debugging.log_call(logger)
def remove_too_close_local_maxima(local_maxima, **parameters):
    # Deleting close local maxima below 16 pixels
    too_close__local_maxima_label_mask__to_remove = numpy.zeros(local_maxima.props.shape, dtype = bool)

    # Find the distance between every centroid (efficiently)
    local_maxima_pairs = numpy.array(list(itertools.combinations(xrange(len(local_maxima.props)), 2)))
    local_maxima_centroid_distance = scipy.spatial.distance.pdist(local_maxima.props["Centroid"], metric = "euclidean")

    too_close_local_maxima_labels_mask = local_maxima_centroid_distance < parameters["min_centroid_distance"]
    too_close_local_maxima_pairs = local_maxima_pairs[too_close_local_maxima_labels_mask]

    for each_too_close_local_maxima_pairs in too_close_local_maxima_pairs:
        first_props_index, second_props_index = each_too_close_local_maxima_pairs

        if (local_maxima.props["Label"][first_props_index] == local_maxima.props["Label"][second_props_index]):
            if local_maxima.props["IntCentroidWaveletValue"][first_props_index] < local_maxima.props["IntCentroidWaveletValue"][second_props_index]:
                too_close__local_maxima_label_mask__to_remove[first_props_index] = True
            else:
                too_close__local_maxima_label_mask__to_remove[second_props_index] = True

    new_local_maxima = copy.deepcopy(local_maxima)

    new_local_maxima.remove_prop_mask(too_close__local_maxima_label_mask__to_remove)
    
    return(new_local_maxima)


@advanced_debugging.log_call(logger)
def wavelet_denoising(new_image, debug = False, **parameters):
    """
        Performs wavelet denoising on the given dictionary.
        
        Args:
            new_data(numpy.ndarray):      array of data for generating a dictionary (first axis is time).
            parameters(dict):             passed directly to spams.trainDL.
        
        Returns:
            dict: the dictionary found.
    """
    ######## TODO: Break up into several simpler functions with unit/doctests. Debug further to find memory leak?
    
    neurons = get_empty_neuron(new_image)
    
    if debug:
        centroid_label_image_0 = numpy.zeros(new_image.shape, dtype = int)
        centroid_label_image_1 = numpy.zeros(new_image.shape, dtype = int)
        centroid_label_image_2 = numpy.zeros(new_image.shape, dtype = int)
        
        centroid_active_label_image_0 = numpy.zeros(new_image.shape, dtype = int)
        centroid_active_label_image_1 = numpy.zeros(new_image.shape, dtype = int)
        centroid_active_label_image_2 = numpy.zeros(new_image.shape, dtype = int)
    
    new_wavelet_image_denoised_segmentation = None
    
    logger.debug("Started wavelet denoising.")
    logger.debug("Removing noise...")
    
    # Contains a bool array with significant values True and noise False.
    new_image_noise_estimate = denoising.estimate_noise(new_image, significance_threshhold = parameters["significance_threshhold"])
    
    # Dictionary with wavelet transform applied. Wavelet transform is the first index.
    #new_wavelet_transformed_image = numpy.zeros((parameters["scale"],) + new_image.shape)
    #new_wavelet_transformed_image_intermediates = numpy.zeros((parameters["scale"] + 1,) + new_image.shape)
    #new_wavelet_transformed_image[:], new_wavelet_transformed_image_intermediates[:] = wavelet_transform.wavelet_transform(new_image, parameters["scale"], True)
    new_wavelet_transformed_image = numpy.zeros((parameters["scale"],) + new_image.shape)
    new_wavelet_transformed_image[:] = wavelet_transform.wavelet_transform(new_image, scale = parameters["scale"])
    
    # Contains a bool array with significant values True and noise False for all wavelet transforms.
    new_wavelet_transformed_image_significant_mask = denoising.significant_mask(new_wavelet_transformed_image, noise_estimate = new_image_noise_estimate, noise_threshhold = parameters["noise_threshhold"])
    new_wavelet_image_mask = new_wavelet_transformed_image_significant_mask[-1].copy()
    
    # Creates a new dictionary without the noise
    new_wavelet_image_denoised = new_wavelet_transformed_image[-1].copy()
    new_wavelet_image_denoised *= new_wavelet_image_mask
    
    logger.debug("Noise removed.")
    
    if new_wavelet_image_denoised.any():
        logger.debug("Frame has content other than noise.")
        
        logger.debug("Finding the label image...")
        
        # For holding the label image
        new_wavelet_image_denoised_labeled = scipy.ndimage.label(new_wavelet_image_denoised)[0]
        # Renumber all labels sequentially
        new_wavelet_image_denoised_labeled = skimage.segmentation.relabel_sequential(new_wavelet_image_denoised_labeled)[0]

        logger.debug("Found the label image.")
        logger.debug("Determining the properties of the label image...")
        
        # For holding the label image properties
        new_wavelet_image_denoised_labeled_props = region_properties(new_wavelet_image_denoised_labeled, properties = parameters["accepted_region_shape_constraints"].keys())
        
        logger.debug("Determined the properties of the label image.")
        
        logger.debug("Finding regions that fail to meet some shape constraints...")

        not_within_bound = numpy.zeros(new_wavelet_image_denoised_labeled_props.shape, dtype = bool)

        # Go through each property and make sure they are within the bounds
        for each_prop in parameters["accepted_region_shape_constraints"]:
            # Get lower and upper bounds for the current property
            lower_bound = parameters["accepted_region_shape_constraints"][each_prop]["min"]
            upper_bound = parameters["accepted_region_shape_constraints"][each_prop]["max"]

            # Determine whether lower or upper bound is satisfied
            is_lower_bounded = (lower_bound <= new_wavelet_image_denoised_labeled_props[each_prop])
            is_upper_bounded = (new_wavelet_image_denoised_labeled_props[each_prop] <= upper_bound)

            # See whether both or neither bound is satisified.
            is_within_bound = is_lower_bounded & is_upper_bounded
            is_not_within_bound = ~is_within_bound
            
            # Collect the unbounded ones
            not_within_bound |= is_not_within_bound
        
        logger.debug("Found regions that fail to meet some shape constraints.")
        
        logger.debug("Reducing wavelet transform on regions outside of constraints...")
        
        # Get labels of the unbounded ones
        labels_not_within_bound = new_wavelet_image_denoised_labeled_props["Label"][not_within_bound]
        #labels_not_within_bound = new_wavelet_image_denoised_labeled_props[not_within_bound]
        #if labels_not_within_bound.size:
        #    labels_not_within_bound = labels_not_within_bound["Label"]
        
        # Iterate over the unbounded ones to fix any errors.
        for each_labels_not_within_bound in labels_not_within_bound:
            # Get a mask for the current label
            current_label_mask = ( new_wavelet_image_denoised_labeled == each_labels_not_within_bound )

            # Get a lower wavelet mask
            lower_wavelet_mask = new_wavelet_transformed_image_significant_mask[-2]

            # Replacement mask
            replacement_mask = current_label_mask & lower_wavelet_mask

            # Zero everything that is not in the replacement region and then use the lower transformed wavelet.
            new_wavelet_image_denoised_replacement = new_wavelet_transformed_image[-2].copy()
            new_wavelet_image_denoised_replacement *= replacement_mask

            # Overwrite the area in our old labeled mask to match this lower wavelet transform
            new_wavelet_image_mask[current_label_mask] = lower_wavelet_mask[current_label_mask]

            # However, overwrite the previously labeled area completely (will push more things into the background)
            new_wavelet_image_denoised[current_label_mask] = new_wavelet_image_denoised_replacement[current_label_mask]
        
        logger.debug("Reduced wavelet transform on regions outside of constraints...")
        
        logger.debug("Finding new label image...")

        new_wavelet_mask_labeled = scipy.ndimage.label(new_wavelet_image_mask)[0]
        # Renumber all labels sequentially
        new_wavelet_mask_labeled = skimage.segmentation.relabel_sequential(new_wavelet_mask_labeled)[0]
        
        logger.debug("Found new label image.")
        
        local_maxima = LabelImageCentroidProps(new_wavelet_image_denoised, new_wavelet_mask_labeled, **parameters["LabelImageCentroidProps"])
        
        centroid_label_image_0 = local_maxima.get_centroid_label_image()
        centroid_active_label_image_0 = local_maxima.get_active_label_image()
        
        local_maxima = remove_low_intensity_local_maxima(local_maxima, **parameters["remove_low_intensity_local_maxima"])
        
        centroid_label_image_1 = local_maxima.get_centroid_label_image()
        centroid_active_label_image_1 = local_maxima.get_active_label_image()

        local_maxima = remove_too_close_local_maxima(local_maxima, **parameters["remove_too_close_local_maxima"])
        
        centroid_label_image_2 = local_maxima.get_centroid_label_image()
        centroid_active_label_image_2 = local_maxima.get_active_label_image()
        
        logger.debug("Removed local maxima that are too close.")
        
        logger.debug("Removing regions without local maxima...")

        # Deleting regions without local maxima
        # As we have been decreasing the count by removing maxima, it is possible that some regions should no longer exist as they have no maxima.
        # Find all these labels that no longer have maxima and create a mask that includes them.
        no_maxima__local_maxima_label_count_mask__to_remove = (local_maxima.count["Count"] == 0)
        no_maxima__local_maxima_labels__to_remove = local_maxima.count["Label"][no_maxima__local_maxima_label_count_mask__to_remove]
        no_maxima__local_maxima_labels__to_remove_labels_mask = numpy.in1d(new_wavelet_mask_labeled, no_maxima__local_maxima_labels__to_remove).reshape(new_wavelet_mask_labeled.shape)
        #no_maxima__local_maxima_label_count_mask__to_remove = (local_maxima_labeled_count["Count"] == 0)
        #no_maxima__local_maxima_labels__to_remove = local_maxima_labeled_count["Label"][no_maxima__local_maxima_label_count_mask__to_remove]
        #no_maxima__local_maxima_labels__to_remove_labels_mask = numpy.in1d(new_wavelet_mask_labeled, no_maxima__local_maxima_labels__to_remove).reshape(new_wavelet_mask_labeled.shape)
        
        # Set all of these regions without maxima to the background
        new_wavelet_image_mask[no_maxima__local_maxima_labels__to_remove_labels_mask] = 0
        new_wavelet_mask_labeled[no_maxima__local_maxima_labels__to_remove_labels_mask] = 0
        new_wavelet_image_denoised[no_maxima__local_maxima_labels__to_remove_labels_mask] = 0
        
        logger.debug("Removed regions without local maxima.")

        if local_maxima.props.size:
            if parameters["use_watershed"]:
                ################ TODO: Revisit to make sure all of Ferran's algorithm is implemented and this is working properly.

                print("local_maxima.props = " + repr(local_maxima.props))

                # Perform the watershed segmentation.

                # First perform disc opening on the image. (Actually, we don't do this.)
                #new_wavelet_image_denoised_opened = vigra.filters.discOpening(new_wavelet_image_denoised.astype(numpy.float32), radius = 1)

                #new_wavelet_image_denoised_maxima = skimage.feature.peak_local_max(new_wavelet_image_denoised, footprint = numpy.ones((3, 3)), labels = (new_wavelet_image_denoised > 0).astype(int), indices = False)

                # We could look for seeds using local maxima. However, we already know what these should be as these are the centroids we have found.
                new_wavelet_image_denoised_maxima = local_maxima.get_centroid_label_image()
                
                print("new_wavelet_image_denoised_maxima = " + repr(new_wavelet_image_denoised_maxima))

                # Segment with watershed on minimum image
                # Use seeds from centroids of local minima
                # Also, include mask
                new_wavelet_image_denoised_segmentation = skimage.morphology.watershed(-new_wavelet_image_denoised, new_wavelet_image_denoised_maxima, mask = (new_wavelet_image_denoised > 0))
                # Renumber all labels sequentially
                new_wavelet_image_denoised_segmentation = skimage.segmentation.relabel_sequential(new_wavelet_image_denoised_segmentation)[0]

                # Get the regions created in segmentation (drop zero as it is the background)
                new_wavelet_image_denoised_segmentation_regions = numpy.unique(new_wavelet_image_denoised_segmentation[new_wavelet_image_denoised_segmentation != 0])

                ## Drop the first two as 0's are the region edges and 1's are the background. Not necessary in our code as seeds for the watershed are set.
                #new_wavelet_image_denoised_segmentation[new_wavelet_image_denoised_segmentation == 1] = 0
                #new_wavelet_image_denoised_segmentation_regions = new_wavelet_image_denoised_segmentation_regions[2:]
                
                ## TODO: Replace with numpy.bincount.

                # Renumber all labels sequentially.
                new_wavelet_image_denoised_segmentation, forward_label_mapping, reverse_label_mapping = skimage.segmentation.relabel_sequential(new_wavelet_image_denoised_segmentation)

                # Find properties of all regions
                new_wavelet_image_denoised_segmentation_props = region_properties(new_wavelet_image_denoised_segmentation, properties = ["Centroid"] + parameters["accepted_neuron_shape_constraints"].keys())
                
                print("new_wavelet_image_denoised_segmentation_props[\"Label\"] = " + repr(new_wavelet_image_denoised_segmentation_props["Label"]))
                print("reverse_label_mapping = " + repr(reverse_label_mapping))

                new_wavelet_image_denoised_segmentation_props["Label"] = reverse_label_mapping[ new_wavelet_image_denoised_segmentation_props["Label"] ]


                new_wavelet_image_denoised_segmentation_props_labels_count = numpy.bincount(new_wavelet_image_denoised_segmentation_props["Label"])[1:]

                print("new_wavelet_image_denoised_segmentation_props[\"Label\"] = " + repr(new_wavelet_image_denoised_segmentation_props["Label"]))
                print("new_wavelet_image_denoised_segmentation_props_labels_count = " + repr(new_wavelet_image_denoised_segmentation_props_labels_count))

                new_wavelet_image_denoised_segmentation_props_labels_duplicates_mask = (new_wavelet_image_denoised_segmentation_props_labels_count > 1)
                new_wavelet_image_denoised_segmentation_props_labels_duplicates = numpy.unique(new_wavelet_image_denoised_segmentation_props["Label"][new_wavelet_image_denoised_segmentation_props_labels_duplicates_mask])

                # Determine a mask that represents all labels to be set to zero in the watershed (new_wavelet_image_denoised_segmentation_props_labels_non_duplicates_watershed_mask)
                # The first index now corresponds to the same index used to denote each duplicate. The rest for the image.
                #new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_all_masks = advanced_numpy.all_permutations_equal(new_wavelet_image_denoised_segmentation_props_labels_duplicates, new_wavelet_image_denoised_segmentation)
                # If any of the masks contain a point to remove then it should be included for removal. Only points in none of the stacks should not.
                #new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_mask = new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_all_masks.any(axis = 0)
                new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_mask = advanced_numpy.contains(new_wavelet_image_denoised_segmentation, new_wavelet_image_denoised_segmentation_props_labels_duplicates)
                
                # Zero the labels that need to be removed.
                new_wavelet_image_denoised_segmentation[new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_mask] = 0

                # Toss the region props that we don't want.
                new_wavelet_image_denoised_segmentation_props = new_wavelet_image_denoised_segmentation_props[~new_wavelet_image_denoised_segmentation_props_labels_duplicates_mask]

                # Renumber all labels sequentially.
                new_wavelet_image_denoised_segmentation, forward_label_mapping, reverse_label_mapping = skimage.segmentation.relabel_sequential(new_wavelet_image_denoised_segmentation)
                new_wavelet_image_denoised_segmentation_props["Label"] = reverse_label_mapping[ new_wavelet_image_denoised_segmentation_props["Label"] ]


























    #            new_wavelet_image_denoised_segmentation_props_labels_match = advanced_numpy.all_permutations_equal(new_wavelet_image_denoised_segmentation_props["Label"], new_wavelet_image_denoised_segmentation_regions)
    #            
    #            #print("new_wavelet_image_denoised_segmentation_props_labels_match = ")
    #            #print(repr(new_wavelet_image_denoised_segmentation_props_labels_match))
    #
    #            if (new_wavelet_image_denoised_segmentation_props_labels_match.ndim != 2):
    #                raise Exception("There is no reason this should happen. Someone changed something they shouldn't have. The dimensions of this match should be 2 exactly.")
    #
    #            if (new_wavelet_image_denoised_segmentation_props_labels_match.shape[0] < new_wavelet_image_denoised_segmentation_props_labels_match.shape[1]):
    #                raise Exception("There is no reason this should happen. There are less labeled regions than there are unique labels?!")
    #            elif (new_wavelet_image_denoised_segmentation_props_labels_match.shape[0] > new_wavelet_image_denoised_segmentation_props_labels_match.shape[1]):
    #                # So, we have some labels represented more than once. We will simply eliminate these.
    #                # Find all labels that repeat
    #                new_wavelet_image_denoised_segmentation_props_labels_duplicates_mask = (new_wavelet_image_denoised_segmentation_props_labels_match.sum(axis = 0) > 1)
    #                new_wavelet_image_denoised_segmentation_props_labels_duplicates = numpy.unique(new_wavelet_image_denoised_segmentation_props["Label"][new_wavelet_image_denoised_segmentation_props_labels_duplicates_mask])
    #
    #                # Determine a mask that represents all labels to be set to zero in the watershed (new_wavelet_image_denoised_segmentation_props_labels_non_duplicates_watershed_mask)
    #                # The first index now corresponds to the same index used to denote each duplicate. The rest for the image.
    #                new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_all_masks = advanced_numpy.all_permutations_equal(new_wavelet_image_denoised_segmentation_props_labels_duplicates, new_wavelet_image_denoised_segmentation)
    #                
    #                # If any of the masks contain a point to remove then it should be included for removal. Only points in none of the stacks should not.
    #                new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_mask = new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_all_masks.any(axis = 0)
    #
    #                # Zero the labels that need to be removed.
    #                new_wavelet_image_denoised_segmentation[new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_mask] = 0
    #
    #                # Toss the region props that we don't want.
    #                new_wavelet_image_denoised_segmentation_props = new_wavelet_image_denoised_segmentation_props[~new_wavelet_image_denoised_segmentation_props_labels_duplicates_mask]
    #                
    #                # Renumber all labels sequentially.
    #                new_wavelet_image_denoised_segmentation, forward_label_mapping, reverse_label_mapping = skimage.segmentation.relabel_sequential(new_wavelet_image_denoised_segmentation)
    #                new_wavelet_image_denoised_segmentation_props["Label"] = reverse_label_mapping[ new_wavelet_image_denoised_segmentation_props["Label"] ]

                # Just go ahead and toss the regions. The same information already exists through new_wavelet_image_denoised_segmentation_props["Label"].
                del new_wavelet_image_denoised_segmentation_regions

                not_within_bound = numpy.zeros(new_wavelet_image_denoised_segmentation_props.shape, dtype = bool)


                # Go through each property and make sure they are within the bounds
                for each_prop in parameters["accepted_neuron_shape_constraints"]:
                    # Get lower and upper bounds for the current property
                    lower_bound = parameters["accepted_neuron_shape_constraints"][each_prop]["min"]
                    upper_bound = parameters["accepted_neuron_shape_constraints"][each_prop]["max"]

                    # Determine whether lower or upper bound is satisfied
                    is_lower_bounded = lower_bound <= new_wavelet_image_denoised_segmentation_props[each_prop]
                    is_upper_bounded = new_wavelet_image_denoised_segmentation_props[each_prop] <= upper_bound

                    # See whether both or neither bound is satisified.
                    is_within_bound = is_lower_bounded & is_upper_bounded
                    is_not_within_bound = ~is_within_bound

                    # Collect the unbounded ones
                    not_within_bound |= is_not_within_bound

                # Get labels outside of bounds
                new_wavelet_image_denoised_segmentation_props_unbounded_labels = new_wavelet_image_denoised_segmentation_props["Label"][not_within_bound]

                # Get a mask of the locations in the watershed where these must be zeroed
                #new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_all_masks = advanced_numpy.all_permutations_equal(new_wavelet_image_denoised_segmentation_props_unbounded_labels, new_wavelet_image_denoised_segmentation)
                #new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_mask = numpy.any(new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_all_masks, axis = 0)
                new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_mask = advanced_numpy.contains(new_wavelet_image_denoised_segmentation, new_wavelet_image_denoised_segmentation_props_unbounded_labels)
                
                # Zero them
                new_wavelet_image_denoised_segmentation[new_wavelet_image_denoised_segmentation_regions_labels_duplicates_watershed_mask] = 0

                # Remove the corresponding properties.
                new_wavelet_image_denoised_segmentation_props = new_wavelet_image_denoised_segmentation_props[~not_within_bound]

                if new_wavelet_image_denoised_segmentation_props.size:
                    # Creates a NumPy structure array to store
                    neurons = numpy.zeros(len(new_wavelet_image_denoised_segmentation_props), dtype = neurons.dtype)

                    # Get masks for all cells
                    neurons["mask"] = advanced_numpy.all_permutations_equal(new_wavelet_image_denoised_segmentation_props["Label"], new_wavelet_image_denoised_segmentation)

                    #neurons["image"] = new_wavelet_image_denoised * neurons["mask"]

                    neurons["image_original"] = new_image * neurons["mask"]

                    neurons["area"] = new_wavelet_image_denoised_segmentation_props["Area"]

                    neurons["max_F"] = neurons["image_original"].reshape( (neurons["image_original"].shape[0], -1) ).max(axis = 1)

                    for i in xrange(len(neurons)):
                        neuron_mask_i_points = numpy.array(neurons["mask"][i].nonzero())

                        neurons["gaussian_mean"][i] = neuron_mask_i_points.mean(axis = 1)
                        neurons["gaussian_cov"][i] = numpy.cov(neuron_mask_i_points)

                    neurons["centroid"] = new_wavelet_image_denoised_segmentation_props["Centroid"]
                    
                    if len(neurons) > 1:
                        logger.debug("Extracted neurons. Found " + str(len(neurons)) + " neurons.")
                    else:
                        logger.debug("Extracted a neuron. Found " + str(len(neurons)) + " neuron.")
                    
                    logger.debug("neurons = " + repr(neurons) + "")
            else:
                #################### Some other kind of segmentation??? Talked to Ferran and he said don't worry about implementing this for now. Does not seem to give noticeably better results.
                raise Exception("No other form of segmentation is implemented.")
        else:
            logger.debug("No local maxima left that are acceptable neurons.")
    else:
        logger.debug("Frame is only noise.")
    
    
    
    #print("Done with making neurons.")
    if debug:
        return((neurons, centroid_label_image_0, centroid_label_image_1, centroid_label_image_2, centroid_active_label_image_0, centroid_active_label_image_1, centroid_active_label_image_2,))
    else:
        return(neurons)


@advanced_debugging.log_call(logger)
def fuse_neurons(neuron_1, neuron_2, **parameters):
    """
        Merges the two neurons into one (treats the first with preference).
        
        Args:
            neuron_1(numpy.ndarray):      first neuron (prefered for tie breaking).
            neuron_2(numpy.ndarray):      second neuron (one to merge in).
            parameters(dict):             dictionary of parameters
        
        Note:
            Todo
            Look into move data normalization into separate method (have method chosen by config file).
        
        Returns:
            dict: the dictionary found.
    """
    
    print neuron_1.shape
    print neuron_2.shape
    assert(neuron_1.shape == neuron_2.shape == tuple())
    assert(neuron_1.dtype == neuron_2.dtype)
    
    mean_neuron = numpy.array([neuron_1["image_original"], neuron_2["image_original"]]).mean(axis = 0)
    mean_neuron_mask = mean_neuron > (parameters["fraction_mean_neuron_max_threshold"] * mean_neuron.max())
    
    print("neuron_1[\"image_original\"].shape = " + repr(neuron_1["image_original"].shape))
    
    print("mean_neuron = " + repr(mean_neuron))
    
    print("mean_neuron.shape = " + repr(mean_neuron.shape))
    
    print("mean_neuron_mask = " + repr(mean_neuron_mask))
    
    print("mean_neuron_mask.shape = " + repr(mean_neuron_mask.shape))
    
    # Gaussian mixture model ??? Skipped this.
    
    # Creates a NumPy structure array to store
    new_neuron = numpy.zeros(neuron_1.shape, dtype = neuron_1.dtype)

    new_neuron["mask"] = mean_neuron_mask

    #new_neuron["image"] = mean_neuron * new_neuron["mask"]

    ##### TODO: Revisit whether this correct per Ferran's code.
    new_neuron["image_original"] = neuron_1["image_original"]

    new_neuron["area"] = (new_neuron["mask"] > 0).sum()

    new_neuron["max_F"] = new_neuron["image"].max()

    new_neuron_mask_points = numpy.array(new_neuron["mask"].nonzero())
        
    print("new_neuron_mask_points = " + repr(new_neuron_mask_points))

    print("new_neuron_mask_points.mean(axis = 1) = " + repr(new_neuron_mask_points.mean(axis = 1)))

    new_neuron["gaussian_mean"] = new_neuron_mask_points.mean(axis = 1)
    new_neuron["gaussian_cov"] = numpy.cov(new_neuron_mask_points)
    
    #    for i in xrange(len(new_neuron)):
    #        new_neuron_mask_points = numpy.array(new_neuron["mask"][i].nonzero())
    #        
    #        print("new_neuron_mask_points = " + repr(new_neuron_mask_points))
    #
    #        print("new_neuron_mask_points.mean(axis = 1) = " + repr(new_neuron_mask_points.mean(axis = 1)))
    #        
    #        new_neuron["gaussian_mean"][i] = new_neuron_mask_points.mean(axis = 1)
    #        new_neuron["gaussian_cov"][i] = numpy.cov(new_neuron_mask_points)

    new_neuron["centroid"] = new_neuron["gaussian_mean"]
    
    return(new_neuron)


@advanced_debugging.log_call(logger)
def merge_neuron_sets(new_neuron_set_1, new_neuron_set_2, **parameters):
    """
        Merges the two sets of neurons into one (treats the first with preference).
        
        Args:
            neuron_1(numpy.ndarray):      first neuron (prefered for tie breaking).
            neuron_2(numpy.ndarray):      second neuron (one to merge in).
            parameters(dict):             dictionary of parameters
        
        Note:
            Todo
            Look into move data normalization into separate method (have method chosen by config file).
        
        Returns:
            dict: the dictionary found.
    """
    
    assert(new_neuron_set_1.dtype == new_neuron_set_2.dtype)
    
    #print(repr(new_neuron_set_1))
    #print(repr(new_neuron_set_2))
    
    #print(len(new_neuron_set_1))
    #print(len(new_neuron_set_2))
    
    #try:
    #    print(new_neuron_set_1.dtype)
    #    
    #    for each_name in new_neuron_set_1.dtype.names:
    #        print("new_neuron_set_1[" + each_name + "]")
    #        print(repr(new_neuron_set_1[each_name]))
    #except AttributeError:
    #    print(type(new_neuron_set_1))
    #
    #try:
    #    print(new_neuron_set_2.dtype)
    #    
    #    for each_name in new_neuron_set_2.dtype.names:
    #        print("new_neuron_set_2[" + each_name + "]")
    #        print(repr(new_neuron_set_2[each_name]))
    #except AttributeError:
    #    print(type(new_neuron_set_2))
    
    
    # TODO: Reverse if statement so it is not nots
    if len(new_neuron_set_1) and len(new_neuron_set_2):
        logger.debug("Have 2 sets of neurons to merge.")
        
        new_neuron_set = new_neuron_set_1.copy()

        new_neuron_set_1_flattened = new_neuron_set_1["image_original"].reshape(new_neuron_set_1["image_original"].shape[0], -1)
        new_neuron_set_2_flattened = new_neuron_set_2["image_original"].reshape(new_neuron_set_2["image_original"].shape[0], -1)

        new_neuron_set_1_flattened_mask = new_neuron_set_1["mask"].reshape(new_neuron_set_1["mask"].shape[0], -1)
        new_neuron_set_2_flattened_mask = new_neuron_set_2["mask"].reshape(new_neuron_set_2["mask"].shape[0], -1)
        
        print("new_neuron_set_1_flattened.nonzero() = " + repr(new_neuron_set_1_flattened.nonzero()))
        print("new_neuron_set_2_flattened.nonzero() = " + repr(new_neuron_set_2_flattened.nonzero()))
        
        # Measure the dot product between any two neurons (i.e. related to the angle of separation)
        new_neuron_set_angle = 1 - scipy.spatial.distance.cdist(new_neuron_set_1_flattened,
                                                                new_neuron_set_2_flattened,
                                                                "cosine")

        # Measure the normalized Hamming distance (0, the same; 1, exact opposites) between the two
        # Need 1 minus to calculate the number of similarities. Now, (1, the same; 0, exact opposites)
        new_neuron_set_masks_overlayed = 1 - scipy.spatial.distance.cdist(new_neuron_set_1_flattened_mask,
                                                                          new_neuron_set_2_flattened_mask,
                                                                          "hamming")

        # Rescale the normalized Hamming distance to a non-normalized Hamming distance (i.e. the Hamming distance).
        new_neuron_set_masks_overlayed *= (new_neuron_set_1_flattened_mask.shape[1])

        # Find the number of true values in each mask for each neuron
        new_neuron_set_1_masks_count = new_neuron_set_1["area"]
        new_neuron_set_2_masks_count = new_neuron_set_2["area"]
        
        # Expand the counts to the size new_neuron_set_masks_overlayed. This solves any broadcasting bug.
        new_neuron_set_1_masks_count_expanded = advanced_numpy.expand_view(new_neuron_set_1_masks_count, reps_after = (new_neuron_set_masks_overlayed.shape[1],))
        new_neuron_set_2_masks_count_expanded = advanced_numpy.expand_view(new_neuron_set_2_masks_count, reps_before = (new_neuron_set_masks_overlayed.shape[0],))
        
        # Normalizes each set of masks by the count
        new_neuron_set_masks_overlayed_1 = new_neuron_set_masks_overlayed / new_neuron_set_1_masks_count_expanded
        new_neuron_set_masks_overlayed_2 = new_neuron_set_masks_overlayed / new_neuron_set_2_masks_count_expanded

        print("new_neuron_set_angle = " + repr(new_neuron_set_angle))
        print("new_neuron_set_masks_overlayed = " + repr(new_neuron_set_masks_overlayed))
        print("new_neuron_set_masks_overlayed_1 = " + repr(new_neuron_set_masks_overlayed_1))
        print("new_neuron_set_masks_overlayed_2 = " + repr(new_neuron_set_masks_overlayed_2))

        # Now that the three measures for the correlation method have been found, we want to know,
        # which are the best correlated neurons between the two sets using these measures.
        # This done to find the neuron in new_neuron_set_1 that best matches each neuron in new_neuron_set_2.
        new_neuron_set_angle_optimal = new_neuron_set_angle.argmax(axis = 0)
        new_neuron_set_masks_overlayed_1_optimal = new_neuron_set_masks_overlayed_1.argmax(axis = 0)
        new_neuron_set_masks_overlayed_2_optimal = new_neuron_set_masks_overlayed_2.argmax(axis = 0)

        print("new_neuron_set_angle_optimal = " + repr(new_neuron_set_angle_optimal))
        print("new_neuron_set_masks_overlayed_1_optimal = " + repr(new_neuron_set_masks_overlayed_1_optimal))
        print("new_neuron_set_masks_overlayed_2_optimal = " + repr(new_neuron_set_masks_overlayed_2_optimal))

        # Fuse or add each neuron from new_neuron_set_2 to the new_neuron_set composed of new_neuron_set_1
        for j in xrange(new_neuron_set_2.shape[0]):
            new_neuron_set_angle_i = new_neuron_set_angle_optimal[j]
            new_neuron_set_masks_overlayed_1_i = new_neuron_set_masks_overlayed_1_optimal[j]
            new_neuron_set_masks_overlayed_2_i = new_neuron_set_masks_overlayed_2_optimal[j]

            print("new_neuron_set_angle_i = " + repr(new_neuron_set_angle_i))
            print("new_neuron_set_masks_overlayed_1_i = " + repr(new_neuron_set_masks_overlayed_1_i))
            print("new_neuron_set_masks_overlayed_2_i = " + repr(new_neuron_set_masks_overlayed_2_i))
            
            new_neuron_set_angle_max = new_neuron_set_angle[ new_neuron_set_angle_i, j ]
            new_neuron_set_masks_overlayed_1_max = new_neuron_set_masks_overlayed_1[ new_neuron_set_masks_overlayed_1_i, j ]
            new_neuron_set_masks_overlayed_2_max = new_neuron_set_masks_overlayed_2[ new_neuron_set_masks_overlayed_2_i, j ]
            
            print(repr(new_neuron_set_masks_overlayed_2_max))
            
            if new_neuron_set_angle_max > parameters["alignment_min_threshold"]:
                new_neuron_set[new_neuron_set_angle_i] = fuse_neurons(new_neuron_set_1[new_neuron_set_angle_i], new_neuron_set_2[j], **parameters["fuse_neurons"])
            else:
                if new_neuron_set_masks_overlayed_2_max > parameters["overlap_min_threshold"]:
                    new_neuron_set[new_neuron_set_masks_overlayed_2_i] = fuse_neurons(new_neuron_set_1[new_neuron_set_masks_overlayed_2_i], new_neuron_set_2[j], **parameters["fuse_neurons"])
                elif new_neuron_set_masks_overlayed_1_max > parameters["overlap_min_threshold"]:
                    new_neuron_set[new_neuron_set_masks_overlayed_1_i] = fuse_neurons(new_neuron_set_1[new_neuron_set_masks_overlayed_1_i], new_neuron_set_2[j], **parameters["fuse_neurons"])
                else:
                    numpy.append(new_neuron_set, new_neuron_set_2[j].copy())

        #print("new_neuron_set = ")
        #print(repr(new_neuron_set))
        
    elif not len(new_neuron_set_1):
        logger.debug("Have 1 sets of neurons to merge. Only the first set has neurons.")
        new_neuron_set = new_neuron_set_2
    elif not len(new_neuron_set_2):
        logger.debug("Have 1 sets of neurons to merge. Only the second set has neurons.")
        new_neuron_set = new_neuron_set_1
    else:
        logger.debug("Have 0 sets of neurons to merge.")
        new_neuron_set = new_neuron_set_1
    
    return(new_neuron_set)


@advanced_debugging.log_call(logger)
def generate_neurons(new_images, debug = False, **parameters):
    """
        Generates the neurons.
        
        Args:
            neuron_1(numpy.ndarray):      first neuron (prefered for tie breaking).
            neuron_2(numpy.ndarray):      second neuron (one to merge in).
            parameters(dict):             dictionary of parameters
        
        Note:
            Todo
            Look into move data normalization into separate method (have method chosen by config file).
        
        Returns:
            dict: the dictionary found.
    """
    
    new_preprocessed_images = normalize_data(new_images, **parameters["normalize_data"])
    
    new_dictionary = generate_dictionary(new_preprocessed_images, **parameters["generate_dictionary"])
    
    
    if debug:
        centroid_label_image_0 = numpy.zeros(new_dictionary.shape, dtype = int)
        centroid_label_image_1 = numpy.zeros(new_dictionary.shape, dtype = int)
        centroid_label_image_2 = numpy.zeros(new_dictionary.shape, dtype = int)
        
        centroid_active_label_image_0 = numpy.zeros(new_dictionary.shape, dtype = int)
        centroid_active_label_image_1 = numpy.zeros(new_dictionary.shape, dtype = int)
        centroid_active_label_image_2 = numpy.zeros(new_dictionary.shape, dtype = int)
    
    # Get all neurons for all images
    new_neurons_set = get_empty_neuron(new_images[0])
    for i, each_new_dictionary_image in enumerate(new_dictionary):
        if debug:
            (each_new_neuron_set, centroid_label_image_0[i][:], centroid_label_image_1[i][:], centroid_label_image_2[i][:], centroid_active_label_image_0[i][:], centroid_active_label_image_1[i][:], centroid_active_label_image_2[i][:],) = wavelet_denoising(each_new_dictionary_image, debug = debug, **parameters["wavelet_denoising"])
        else:
            each_new_neuron_set = wavelet_denoising(each_new_dictionary_image, debug = debug, **parameters["wavelet_denoising"])
        
        logger.debug("Denoised a set of neurons from frame " + str(i + 1) + " of " + str(len(new_dictionary)) + ".")
        
        new_neurons_set = merge_neuron_sets(new_neurons_set, each_new_neuron_set, **parameters["merge_neuron_sets"])
        
        logger.debug("Merged a set of neurons from frame " + str(i + 1) + " of " + str(len(new_dictionary)) + ".")
    
    if debug:
        return((new_dictionary, new_neurons_set, centroid_label_image_0, centroid_label_image_1, centroid_label_image_2, centroid_active_label_image_0, centroid_active_label_image_1, centroid_active_label_image_2))
    else:
        return(new_neurons_set)
