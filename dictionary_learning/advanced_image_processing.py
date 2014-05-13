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

# To allow for more advanced iteration pattersn
import itertools

# Need for opening
import vigra
import vigra.filters

# Need in order to have logging information no matter what.
import advanced_debugging

# Short function to process image data.
import simple_image_processing

# To remove noise from the basis images
import denoising

# Wavelet transformation operations
import wavelet_transform


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)


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
    new_data_processed = simple_image_processing.zeroed_mean_images(new_data_processed)
    
    # Renormalize each row vector using L_2
    # Unfortunately our version of numpy's function numpy.linalg.norm does not support the axis keyword. So, we must use a for loop.
    new_data_processed = simple_image_processing.renormalized_images(new_data_processed, ord = 2)

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
    new_dictionary = numpy.asarray(new_dictionary).reshape((parameters["K"],) + new_data.shape[1:])
    new_dictionary = new_dictionary.copy()
    
    return(new_dictionary)


@advanced_debugging.log_call(logger, print_args = True)
def region_properties(new_label_image, *args, **kwargs):
    """
        Grabs region properties from a label .
        
        Args:
            new_data(numpy.ndarray):      array of data for generating a dictionary (first axis is time).
            parameters(dict):             passed directly to spams.trainDL.
        
        Note:
            Todo
            Look into move data normalization into separate method (have method chosen by config file).
        
        Returns:
            dict: the dictionary found.
        
        
        Examples:
            
            >>> region_properties(numpy.zeros((2,2), dtype=int))
            array([], 
                  dtype=[('Centroid', 'O'), ('Area', 'O')])
            
            >>> region_properties(numpy.ones((2,2), dtype=int))
            array([(1, (0.5, 0.5), 4.0)], 
                  dtype=[('Label', '<i8'), ('Centroid', 'O'), ('Area', '<f8')])
            
            >>> region_properties(numpy.ones((3,3), dtype=int))
            array([(1, (1.0, 1.0), 9.0)], 
                  dtype=[('Label', '<i8'), ('Centroid', 'O'), ('Area', '<f8')])
    """
    
    # This gives a list of dictionaries. However, this is not very usable. So, we will convert this to a structured NumPy array.
    new_label_image_props = skimage.measure.regionprops(new_label_image, *args, **kwargs)
    
    for each_new_label_image_prop in new_label_image_props:
        for each_key in [ "BoundingBox", "Centroid", "HuMoments", "WeightedCentroid", "WeightedHuMoments" ]:
            if each_key in each_new_label_image_prop:
                each_new_label_image_prop[each_key] = numpy.array(each_new_label_image_prop[each_key])
    
    # Holds the values from props.
    new_label_image_props_values = []
    
    # Holds the types from props.
    new_label_image_props_dtype = []
    
    if new_label_image_props:
        # Get types for all properties as a dictionary
        for each_name, each_sample_value in new_label_image_props[0].items():
            each_type = type(each_sample_value)
            
            if each_type is numpy.ndarray:
                new_label_image_props_dtype.append( (each_name, each_sample_value.dtype, each_sample_value.shape) )
            else:
                new_label_image_props_dtype.append( (each_name, each_type) )

        # Store the values to place in NumPy structured array in order.
        new_label_image_props_values = []
        for j in xrange(len(new_label_image_props)):
            # Add all values in order of keys from the dictionary.
            new_label_image_props_values.append([])
            for each_new_label_image_props_dtype in new_label_image_props_dtype:
                each_dtype_key = each_new_label_image_props_dtype[0]
                new_label_image_props_values[j].append(new_label_image_props[j][each_dtype_key])

            # NumPy will expect a tuple for each set of values.
            new_label_image_props_values[j] = tuple(new_label_image_props_values[j])
            
    else:
        if "properties" not in kwargs:
            kwargs["properties"] = ["Area", "Centroid"]
        
        new_label_image_props_dtype = dict([(_k, numpy.object)  for _k in kwargs["properties"]])
    
    # Replace the properties with the structured array.
    new_label_image_props = numpy.array(new_label_image_props_values, dtype = new_label_image_props_dtype)
    
    return(new_label_image_props)


@advanced_debugging.log_call(logger, print_args = True)
def wavelet_denoising(new_image, **parameters):
    """
        Preforms wavelet denoising on the given dictionary.
        
        Args:
            new_data(numpy.ndarray):      array of data for generating a dictionary (first axis is time).
            parameters(dict):             passed directly to spams.trainDL.
        
        Note:
            Todo
            Rest of steps
        
        Returns:
            dict: the dictionary found.
    """
    ######## TODO: Break up into several simpler functions with unit/doctests.
    
    # Contains a bool array with significant values True and noise False.
    new_image_noise_estimate = denoising.estimate_noise(new_image, significance_threshhold = parameters["significance_threshhold"])
    
    # Dictionary with wavelet transform applied. Wavelet transform is the first index.
    new_wavelet_transformed_image = numpy.zeros((parameters["scale"],) + new_image.shape)
    new_wavelet_transformed_image_intermediates = numpy.zeros((parameters["scale"] + 1,) + new_image.shape)
    new_wavelet_transformed_image, new_wavelet_transformed_image_intermediates = wavelet_transform.wavelet_transform(new_image, scale = parameters["scale"])
    
    # Contains a bool array with significant values True and noise False for all wavelet transforms.
    new_wavelet_transformed_image_significant_mask = denoising.significant_mask(new_wavelet_transformed_image, noise_estimate = new_image_noise_estimate, noise_threshhold = parameters["noise_threshhold"])
    new_wavelet_image_mask = new_wavelet_transformed_image_significant_mask[-1].copy()
    
    # Creates a new dictionary without the noise
    new_wavelet_image_denoised = new_wavelet_transformed_image[-1].copy()
    new_wavelet_image_denoised *= new_wavelet_image_mask
    
    # For holding the label image
    new_wavelet_image_denoised_labeled = skimage.morphology.label(new_wavelet_image_denoised)[0]
    
    # For holding the label image properties
    new_wavelet_image_denoised_labeled_props = []
    
    # Get the types for the label image properties structured array
    new_wavelet_image_denoised_labeled_props_dtypes = []
    for each_key in parameters["regionprops"].keys():
        each_type = numpy.dtype(parameters["regionprops"][each_key]["max"])
        new_wavelet_image_denoised_labeled_props_dtypes.append( (each_key, each_type) )
    
    # For holding the label image properties
    new_wavelet_image_denoised_labeled_props = region_properties(new_wavelet_image_denoised_labeled , properties = parameters["regionprops"].keys())
    
    
    
    not_within_bound = numpy.zeros(new_wavelet_image_denoised_labeled_props.shape, dtype = bool)
    
    # Go through each property and make sure they are within the bounds
    for each_prop in parameters["regionprops"]:
        # Get lower and upper bounds for the current property
        lower_bound = parameters["regionprops"][each_prop]["min"]
        upper_bound = parameters["regionprops"][each_prop]["max"]

        # Determine whether lower or upper bound is satisfied
        is_lower_bounded = lower_bound < new_wavelet_image_denoised_labeled_props[each_prop]
        is_upper_bounded = new_wavelet_image_denoised_labeled_props[each_prop] < upper_bound

        # See whether both or neither bound is satisified.
        is_within_bound = is_lower_bounded & is_upper_bounded
        is_not_within_bound = ~is_within_bound

        # Collect the unbounded ones
        not_within_bound |= is_not_within_bound

    # Get labels of the unbounded ones
    labels_not_within_bound = new_wavelet_image_denoised_labeled_props[not_within_bound]["Label"]

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
    
    
    new_wavelet_mask_labeled = skimage.morphology.label(new_wavelet_image_mask)[0]
    
    
    
    
    # Would be good to use peak_local_max as it has more features and is_local_maximum is removed in later versions,
    # but it requires skimage 0.8.0 minimum.
    local_maxima_mask = skimage.morphology.is_local_maximum(new_wavelet_image_denoised, footprint = numpy.ones((3,) * new_wavelet_image_denoised.ndim), labels = (new_wavelet_image_denoised > 0).astype(int))
    #local_maxima = skimage.feature.peak_local_max(new_wavelet_image_denoised, footprint = numpy.ones((3,) * new_wavelet_image_denoised.ndim), labels = (new_wavelet_image_denoised > 0).astype(int), indices = False)

    # Group local maxima. Also, we don't care about differentiating them. If there are several local maxima touching, we only want one.
    # Note, if they are touching, they must be on a plateau (i.e. all have the same value).
    local_maxima_labeled = label(local_maxima_mask.astype(int))[0]
    local_maxima_labeled_binary = (local_maxima_labeled > 0).astype(int)

    # Extract the centroids.
    local_maxima_labeled_props = region_properties(local_maxima_labeled_binary, properties = ["Centroid"])
    
    # These should not be used agan. So, we drop them.
    # This way, if they are used, we get an error.
    del local_maxima_mask
    del local_maxima_labeled
    del local_maxima_labeled_binary

    # Stores the number of times a particular label appears.
    local_maxima_labeled_count = numpy.zeros( (len(local_maxima_labeled_props),), dtype = [("Label", int), ("Count", int)] )
    local_maxima_labeled_count["Label"] = numpy.arange(1,len(local_maxima_labeled_count)+1)
    
    # We want to have a few more type present in our NumPy structured array. To do this, we collect the existing types into
    # a list and then add our new types onto the end. Finally, we make the new structured array type from the list we have.
    local_maxima_labeled_props_dtype = []
    
    for each_name in local_maxima_labeled_props.dtype.names:
        local_maxima_labeled_props_dtype.append( (each_name, local_maxima_labeled_props[each_name].dtype) )
    
    local_maxima_labeled_props_dtype.append( ("IntCentroid", local_maxima_labeled_props["Centroid"].dtype, local_maxima_labeled_props["Centroid"].shape[1:]) )
    local_maxima_labeled_props_dtype.append( ("IntCentroidWaveletValue", new_wavelet_image_denoised.dtype) )
    
    local_maxima_labeled_props_dtype = numpy.dtype(local_maxima_labeled_props_dtype)
    
    # Makes a new properties array that contains enough entries to hold the old one and has all the types we desire.
    new_local_maxima_labeled_props = numpy.zeros(local_maxima_labeled_props.shape, dtype = local_maxima_labeled_props_dtype)
    
    # Copy ove the old values.
    for each_name in local_maxima_labeled_props.dtype.names:
        new_local_maxima_labeled_props[each_name] = local_maxima_labeled_props[each_name].copy()
    
    # Replace the old structured array with the enlarged version.
    local_maxima_labeled_props = new_local_maxima_labeled_props
    
    
    # Get integers close to local max
    local_maxima_labeled_props["IntCentroid"] = local_maxima_labeled_props["Centroid"].round().astype(int)
    
    # Stores the value from wavelet denoising at the centroid for easy retrieval
    local_maxima_labeled_props["IntCentroidWaveletValue"] = new_wavelet_image_denoised[ tuple(local_maxima_labeled_props["IntCentroid"].T) ]
    
    # Overwrite the label parameter as it holds no information as it is always 1, Now, is the label from wavelet mask label image.
    local_maxima_labeled_props["Label"] = new_wavelet_mask_labeled[ tuple(local_maxima_labeled_props["IntCentroid"].T) ]
    
    if (numpy.any(local_maxima_labeled_props["Label"] == 0)):
        # There shouldn't be any maximums in the background. This should never happen.
        logger.debug("Maximum found where Label is 0.")
    
    # Increase the count of the matching label
    local_maxima_labeled_count["Count"] += (local_maxima_labeled_count["Label"].reshape(-1,1) == local_maxima_labeled_props["Label"].reshape(1,-1)).sum(axis=1)
    
    
    if numpy.any(local_maxima_labeled_count["Count"] == 0):
        # All labels should have a local maximum. If they don't, this could be a problem
        
        failed_labels_list = local_maxima_labeled_count["Label"][local_maxima_labeled_count["Count"] == 0].tolist()
        failed_labels_list = [str(_) for _ in failed_labels_list]
        failed_label_str = ", ".join(failed_labels_list)
        failed_label_msg = "Label(s) not found in local maxima. For labels = [ " + failed_label_str + " ]."
        
        logger.debug(failed_label_msg)
    
    # Deleting local maxima that does not exceed the 90th percentile of the pixel intensities
    low_intensities__local_maxima_labels__to_remove = numpy.array(local_maxima_labeled_props.shape, dtype = bool)
    for i in xrange(len(local_maxima_labeled_props)):
        # Get the region with the label matching the maximum
        each_region_image_wavelet_mask = (new_wavelet_mask_labeled == local_maxima_labeled_props[i]["Label"])
        each_region_image_wavelet = new_wavelet_image_denoised[each_region_image_wavelet_mask]
        
        # Get the number of pixels in that region
        each_region_image_wavelet_num_pixels = float(each_region_image_wavelet.size)
        
        # Get the value of the max for that region
        each_region_image_wavelet_centroid_value = local_maxima_labeled_props[i]["IntCentroidWaveletValue"]
        
        # Get a mask of the pixels below that max for that region
        each_region_image_wavelet_num_pixels_below_max = float((each_region_image_wavelet < each_region_image_wavelet_centroid_value).sum())
        
        # Get a ratio of the number of pixels below that max for that region
        each_region_image_wavelet_ratio_pixels = each_region_image_wavelet_num_pixels_below_max / each_region_image_wavelet_num_pixels
        
        # If the ratio clears our threshhold, keep this label. Otherwise, eliminate it.
        low_intensities__local_maxima_labels__to_remove[i] = (each_region_image_wavelet_ratio_pixels < parameters["percentage_pixels_below_max"])
    
    
    local_maxima_mask[tuple(local_maxima_labeled_props["IntCentroidWaveletValue"].T)]
    # Take a subset of the label props and reduce the count
    local_maxima_labeled_props = local_maxima_labeled_props[ ~low_intensities__local_maxima_labels__to_remove ].copy()
    local_maxima_labeled_count["Count"] -= low_intensities__local_maxima_labels__to_remove
    
    
    
    # Deleting close local maxima below 16 pixels
    too_close__local_maxima_labels__to_remove = numpy.array(local_maxima_labeled_props.shape, dtype = bool)
    
    # Find the distance between every centroid (efficiently)
    local_maxima_pairs = numpy.array(list(itertools.combinations(xrange(len(local_maxima_labeled_props)), 2)))
    local_maxima_centroid_distance = distance.pdist(local_maxima_labeled_props["Centroid"], metric = "euclidean")
    
    too_close_local_maxima_labels_mask = local_maxima_centroid_distance < parameters["min_centroid_distance"]
    too_close_local_maxima_pairs = local_maxima_pairs[too_close_local_maxima_labels_mask]
    
    for each_too_close_local_maxima_pairs in too_close_local_maxima_pairs:
        first_props_index, second_props_index = each_too_close_local_maxima_pairs
        
        ############################# Shouldn't we check for the same label. Ask Ferran. He ok'd. Probably not necessary, but won't hurt.
        if (local_maxima_labeled_props["Label"][first_props_index] == local_maxima_labeled_props["Label"][second_props_index]):
            if local_maxima_labeled_props["IntCentroidWaveletValue"][first_props_index] < local_maxima_labeled_props["IntCentroidWaveletValue"][second_props_index]:
                too_close__local_maxima_labels__to_remove[first_props_index] = True
            else:
                too_close__local_maxima_labels__to_remove[second_props_index] = True
    
    # Take a subset of the label props and reduce the count
    local_maxima_labeled_props = local_maxima_labeled_props[ ~too_close__local_maxima_labels__to_remove ].copy()
    local_maxima_labeled_count["Count"] -= too_close__local_maxima_labels__to_remove
    
    # Deleting regions without local maxima
    # As we have been decreasing the count by removing maxima, it is possible that some regions should no longer exist as they have no maxima.
    # Find all these labels that no longer have maxima and create a mask that includes them.
    no_maxima__local_maxima_labels__to_remove = (local_maxima_labeled_count["Count"] == 0)
    no_maxima__local_maxima_labels__to_remove_labels = local_maxima_labeled_props["Label"][no_maxima__local_maxima_labels__to_remove]
    no_maxima__local_maxima_labels__to_remove_labels_mask = numpy.in1d(new_wavelet_mask_labeled, no_maxima__local_maxima_labels__to_remove_labels).reshape(new_wavelet_mask_labeled.shape)
    
    # Set all of these regions without maxima to the background
    new_wavelet_image_mask[no_maxima__local_maxima_labels__to_remove_labels_mask] = 0
    new_wavelet_image_denoised[no_maxima__local_maxima_labels__to_remove_labels_mask] = 0
    
    if parameters["use_watershed"]:
        ################ TODO: Revisit to make sure all of Ferran's algorithm is implemented and this is working properly.
        
        # Preform the watershed segmentation.
        
        # First preform disc opening on the image.
        new_wavelet_image_denoised_opened = vigra.filters.discOpening(new_wavelet_image_denoised, radius = 1)
        
        # Would be good to use peak_local_max as it has more features and is_local_maximum is removed in later versions,
        # but it requires skimage 0.8.0 minimum.
        #new_wavelet_image_denoised_opened_maxima = skimage.morphology.is_local_maximum(new_wavelet_image_denoised_opened, footprint = numpy.ones((3, 3)), labels = (new_wavelet_image_denoised_opened > 0).astype(int))
        #new_wavelet_image_denoised_opened_maxima = skimage.feature.peak_local_max(new_wavelet_image_denoised_opened, footprint = numpy.ones((3, 3)), labels = (new_wavelet_image_denoised_opened > 0).astype(int), indices = False)
        
        # We could look for seeds using local minima. However, we already know what these should be as these are the centroids we have found.
        
        
        # Segment with watershed on minimum image
        # Use seeds from centroids of local minima
        # Also, include mask
        new_wavelet_image_denoised_opened_segmentation = skimage.morphology.watershed(-new_wavelet_image_denoised_opened, new_wavelet_image_denoised_opened_maxima)
        
        # Get the regions created in segmentation
        new_wavelet_image_denoised_opened_segmentation_regions = numpy.unique(new_wavelet_image_denoised_opened_segmentation)
        
        # Drop the first two as 0's are the region edges and 1's are the background.
        new_wavelet_image_denoised_opened_segmentation[new_wavelet_image_denoised_opened_segmentation == 1] = 0
        new_wavelet_image_denoised_opened_segmentation_regions = new_wavelet_image_denoised_opened_segmentation_regions[2:]
        
    else:
        #################### Some other kind of segmentation??? Talked to Ferran and he said don't worry about implementing this for now. Does not seem to give noticeably better results.
        raise Exception("No other form of segmentation is implemented.")
    
    return()