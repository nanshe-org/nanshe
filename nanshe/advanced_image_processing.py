# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 30, 2014 5:23:37PM$"



# Generally useful and fast to import so done immediately.
import numpy

# For image processing.
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.spatial
import scipy.spatial.distance

# More image processing...
import skimage
import skimage.measure
import skimage.feature
import skimage.morphology
import skimage.segmentation

import advanced_iterators

# To allow for more advanced iteration patterns
import itertools

# Allows for deep and shallow copies.
import copy

# Need for opening
import vigra
import vigra.filters
import vigra.analysis

# Need in order to have logging information no matter what.
import advanced_debugging

import advanced_numpy

# Short function to process image data.
import simple_image_processing

# To remove noise from the basis images
import denoising

# Wavelet transformation operations
import wavelet_transform

import HDF5_logger


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)



@advanced_debugging.log_call(logger)
def removing_lines(new_data, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **parameters):
    """
        Due to registration errors, there will sometimes be lines that are zero. To correct this, we find an interpolated
        value and

        Args:
            new_data(numpy.ndarray):      array of raw data.
            parameters(dict):             essentially unused.

        Returns:
            numpy.ndarray:                a new array with the lines interpolated away.
    """

    result = numpy.zeros(new_data.shape)

    # Get an outline of the region around the parts of the image that contain zeros
    erosion_structure = numpy.ones(tuple(parameters["erosion_shape"]))
    dilation_structure = numpy.ones(tuple(parameters["dilation_shape"]))

    points = numpy.array(numpy.meshgrid(*[numpy.arange(_) for _ in new_data.shape[1:]], indexing="ij"))

    for i in xrange(new_data.shape[0]):
        new_data_i = new_data[i]
        zero_mask = (new_data_i == 0)

        zero_mask_dilated = skimage.morphology.binary_dilation(zero_mask, dilation_structure).astype(bool)
        zero_mask_eroded = skimage.morphology.binary_erosion(zero_mask, erosion_structure).astype(bool)
        zero_mask_outline = zero_mask_dilated - zero_mask_eroded

        # Get the points that correspond to those
        zero_mask_outline_points = points[:, i, zero_mask_outline]

        new_data_i_zero_mask_outline_interpolation = numpy.zeros(new_data_i.shape)
        if zero_mask_outline.any():
            new_data_i_zero_mask_outline_interpolation = scipy.interpolate.griddata(zero_mask_outline_points, new_data_i[zero_mask_outline], tuple(points), method = "linear")

            # Only need to check for nan in our case.
            new_data_i_zero_mask_outline_interpolation = numpy.where((new_data_i_zero_mask_outline_interpolation == numpy.nan),
                                                                     new_data_i_zero_mask_outline_interpolation,
                                                                     0)

        result[i] = numpy.where(zero_mask, new_data_i_zero_mask_outline_interpolation, new_data_i)

    return(result)


@advanced_debugging.log_call(logger)
def extract_f0(new_data, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **params):
    @advanced_debugging.log_call(logger)
    def extract_quantile(new_data, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **params):
        if (params["step_size"] > new_data.shape[0]):
            raise Exception("The step size provided, " + params["step_size"] + ", is larger than the number of frames in the data, " + new_data.shape[0] + ".")

        window_centers = numpy.arange(0, new_data.shape[0], params["step_size"])

        if window_centers[-1] != new_data.shape[0]:
            window_centers = numpy.append(window_centers, new_data.shape[0])

        def window_shape_iterator(window_centers = window_centers):
            for each_window_center in window_centers:
                each_window_lower = max(window_centers[0], each_window_center - params["half_window_size"])
                each_window_upper = min(window_centers[-1], each_window_center + params["half_window_size"])

                yield( (each_window_lower, each_window_center, each_window_upper) )

        which_quantile = advanced_numpy.get_quantiles(params["which_quantile"])

        window_quantiles = numpy.zeros( (window_centers.shape[0],) + which_quantile.shape + new_data.shape[1:] )
        for i, (each_window_lower, each_window_center, each_window_upper) in enumerate(window_shape_iterator()):
            new_data_i = new_data[each_window_lower:each_window_upper]

            each_quantile = advanced_numpy.quantile(new_data_i.reshape(new_data_i.shape[0], -1), params["which_quantile"], axis=0)
            each_quantile = each_quantile.reshape(each_quantile.shape[0], *new_data_i.shape[1:])

            # Are there bad values in our result (shouldn't be if axis=None)
            if each_quantile.mask.any():
                msg = "Found erroneous regions in quantile calculation. Dropping in HDF5 logger."

                logger.error(msg)
                array_debug_logger("each_quantile_" + repr(i), each_quantile)
                raise Exception(msg)
            else:
                each_quantile = each_quantile.data

            window_quantiles[i] = each_quantile

        # Should only be one quantile. Drop the singleton dimension.
        window_quantiles = window_quantiles[:, 0]

        new_data_quantiled = scipy.interpolate.interp1d(window_centers, window_quantiles, axis=0)(numpy.arange(new_data.shape[0]))

        return(new_data_quantiled)


    temporal_smoothing_gaussian_filter = vigra.filters.Kernel1D()
    # TODO: Check to see if norm is acceptable as 1.0 or if it must be 0.0.
    temporal_smoothing_gaussian_filter.initGaussian(params["temporal_smoothing_gaussian_filter_stdev"], 1.0, 5 * params["temporal_smoothing_gaussian_filter_stdev"])
    # TODO: Check what border treatment to use
    temporal_smoothing_gaussian_filter.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_REFLECT)

    new_data_temporally_smoothed = vigra.filters.convolveOneDimension(new_data.astype(numpy.float32), 0, temporal_smoothing_gaussian_filter)

    new_data_quantiled = extract_quantile(new_data_temporally_smoothed, array_debug_logger, **params["extract_quantile"])

    spatial_smoothing_gaussian_filter = vigra.filters.Kernel1D()
    # TODO: Check to see if norm is acceptable as 1.0 or if it must be 0.0.
    spatial_smoothing_gaussian_filter.initGaussian(params["spatial_smoothing_gaussian_filter_stdev"], 1.0, 5 * params["spatial_smoothing_gaussian_filter_stdev"])
    # TODO: Check what border treatment to use
    spatial_smoothing_gaussian_filter.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_REFLECT)

    new_data_spatialy_smoothed = new_data_quantiled.copy()
    for d in xrange(1, new_data_spatialy_smoothed.ndim):
        new_data_spatialy_smoothed = vigra.filters.convolveOneDimension(new_data_spatialy_smoothed.astype(numpy.float32), d, spatial_smoothing_gaussian_filter)

    new_data_baselined = (new_data - new_data_spatialy_smoothed) / new_data_spatialy_smoothed

    return(new_data_baselined)


@advanced_debugging.log_call(logger)
def normalize_data(new_data, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **parameters):
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

    # Remove the mean of each row vector
    new_data_mean_zeroed = simple_image_processing.zeroed_mean_images(new_data)

    # Renormalize each row vector using some specified normalization
    new_data_renormalized = simple_image_processing.renormalized_images(new_data_mean_zeroed,
                                                                        **parameters["simple_image_processing.renormalized_images"])

    return(new_data_renormalized)


@advanced_debugging.log_call(logger)
def preprocess_data(new_data, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **parameters):
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

    # TODO: Add preprocessing step wavelet transform, F_0, remove lines, etc.

    # Remove lines
    if "removing_lines" in parameters:
        new_data_lines_removed = removing_lines(new_data, array_debug_logger, **parameters["removing_lines"])
        array_debug_logger("images_lines_removed", new_data_lines_removed)
    else:
        new_data_lines_removed = new_data

    # Add the bias param
    if "bias" in parameters:
        new_data_bias = new_data_lines_removed + parameters["bias"]
        array_debug_logger("images_biased", new_data_bias)
    else:
        new_data_bias = new_data_lines_removed

    if "extract_f0" in parameters:
        new_data_f0_result = extract_f0(new_data_bias, array_debug_logger, **parameters["extract_f0"])
        array_debug_logger("images_f0", new_data_f0_result)
    else:
        new_data_f0_result = new_data_bias

    if "wavelet_transform" in parameters:
        new_data_wavelet_result = wavelet_transform.wavelet_transform(new_data_f0_result, array_debug_logger, **parameters["wavelet_transform"])[-1]
        array_debug_logger("images_wavelet_transformed", new_data_wavelet_result)
    else:
        new_data_wavelet_result = new_data_f0_result

    new_data_normalized = normalize_data(new_data_wavelet_result, array_debug_logger, **parameters["normalize_data"])
    array_debug_logger("images_normalized", new_data_normalized)

    return(new_data_normalized)


@advanced_debugging.log_call(logger)
def generate_dictionary(new_data, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **parameters):
    """
        Generates a dictionary using the data and parameters given for trainDL.
        
        Args:
            new_data(numpy.ndarray):      array of data for generating a dictionary (first axis is time).
            parameters(dict):             passed directly to spams.trainDL.
        
        Returns:
            dict: the dictionary found.
    """

    import spams_sandbox

    # It takes a loooong time to load spams. so, we shouldn't do this until we are sure that we are ready to generate the dictionary
    # (i.e. the user supplied a bad config file, /images does not exist, etc.).
    # Note it caches the import so subsequent calls should not make it any slower.
    import spams

    # Maybe should copy data so as not to change the original.
    # new_data_processed = new_data.copy()
    new_data_processed = new_data

    # Reshape data into a matrix (each image is now a column vector)
    new_data_processed = numpy.reshape(new_data_processed, (new_data_processed.shape[0], -1))
    new_data_processed = numpy.asmatrix(new_data_processed)
    new_data_processed = new_data_processed.transpose()

    # Spams requires all matrices to be fortran.
    new_data_processed = numpy.asfortranarray(new_data_processed)

    # Simply trains the dictionary. Does not return sparse code.
    # Need to look into generating the sparse code given the dictionary, spams.nmf? (may be too slow))
    new_dictionary = spams_sandbox.spams_sandbox.call_multiprocessing_array_spams_trainDL(new_data_processed, **parameters["spams.trainDL"])

    # Fix dictionary so that the first index will be the particular image.
    # The rest will be the shape of an image (same as input shape).
    new_dictionary = new_dictionary.transpose()
    new_dictionary = numpy.asarray(new_dictionary).reshape((parameters["spams.trainDL"]["K"],) + new_data.shape[1:])

    return(new_dictionary)


@advanced_debugging.log_call(logger)
def region_properties(new_label_image, *args, **kwargs):
    """
        Grabs region properties from a label .
        
        Args:
            new_label_image(numpy.ndarray):      label image used for generating properties.
            args(list):                          additional position arguments to pass skimage.measure.regionprops.
            parameters(dict):                    additional keyword arguments to pass skimage.measure.regionprops.
        
        Returns:
            dict: the dictionary found.
        
        
        Examples:
            
            >>> region_properties(numpy.zeros((2,2), dtype=int))
            array([], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])
            
            >>> region_properties(numpy.ones((2,2), dtype=int))
            array([(1, 4.0, [0.5, 0.5])], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])
            
            >>> region_properties(numpy.ones((3,3), dtype=int))
            array([(1, 9.0, [1.0, 1.0])], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])
    """

    region_properties_type_dict = {
        "area": numpy.float64,
        "bbox": numpy.int64,
        "centroid": numpy.float64,
        "convex_area": numpy.int64,
        "convex_image": numpy.bool_,
        "coords": numpy.int64,
        "eccentricity": numpy.float64,
        "equivalent_diameter": numpy.float64,
        "euler_number": numpy.int64,
        "filled_area": numpy.int64,
        "filled_image": numpy.bool_,
        "image": numpy.bool_,
        "inertia_tensor": numpy.float64,
        "inertia_tensor_eigvals": numpy.float64,
        "intensity_image": numpy.float64,
        "label": numpy.int64,
        "local_centroid": numpy.float64,
        "major_axis_length": numpy.float64,
        "max_intensity": numpy.float64,
        "mean_intensity": numpy.float64,
        "min_intensity": numpy.float64,
        "minor_axis_length": numpy.float64,
        "moments": numpy.float64,
        "moments_central": numpy.float64,
        "moments_hu": numpy.float64,
        "moments_normalized": numpy.float64,
        "orientation": numpy.float64,
        "perimeter": numpy.float64,
        "solidity": numpy.float64,
        "weighted_centroid": numpy.float64,
        "weighted_local_centroid": numpy.float64,
        "weighted_moments": numpy.float64,
        "weighted_moments_central": numpy.float64,
        "weighted_moments_hu": numpy.float64,
        "weighted_moments_normalized": numpy.float64
    }

    region_properties_shape_dict = {
        "area": (),
        "bbox": (4,),
        "centroid": (new_label_image.ndim,),
        "convex_area": (),
        "convex_image": (-1, -1),
        "coords": (-1, 2),
        "eccentricity": (),
        "equivalent_diameter": (),
        "euler_number": (),
        "filled_area": (),
        "filled_image": (-1, -1),
        "image": (-1, -1),
        "inertia_tensor": (2, 2),
        "inertia_tensor_eigvals": (2,),
        "intensity_image": (-1, -1),
        "label": (),
        "local_centroid": (2,),
        "major_axis_length": (),
        "max_intensity": (),
        "mean_intensity": (),
        "min_intensity": (),
        "minor_axis_length": (),
        "moments": (4, 4),
        "moments_central": (4, 4),
        "moments_hu": (7,),
        "moments_normalized": (4, 4),
        "orientation": (),
        "perimeter": (),
        "solidity": (),
        "weighted_centroid": (2,),
        "weighted_local_centroid": (2,),
        "weighted_moments": (4, 4),
        "weighted_moments_central": (4, 4),
        "weighted_moments_hu": (7,),
        "weighted_moments_normalized": (4, 4)
    }

    region_properties_ndim_dict = {
        "area": 0,
        "bbox": 1,
        "centroid": 1,
        "convex_area": 0,
        "convex_image": 2,
        "coords": 2,
        "eccentricity": 0,
        "equivalent_diameter": 0,
        "euler_number": 0,
        "filled_area": 0,
        "filled_image": 2,
        "image": 2,
        "inertia_tensor": 2,
        "inertia_tensor_eigvals": 1,
        "intensity_image": 2,
        "label": 0,
        "local_centroid": 1,
        "major_axis_length": 0,
        "max_intensity": 0,
        "mean_intensity": 0,
        "min_intensity": 0,
        "minor_axis_length": 0,
        "moments": 2,
        "moments_central": 2,
        "moments_hu": 1,
        "moments_normalized": 2,
        "orientation": 0,
        "perimeter": 0,
        "solidity": 0,
        "weighted_centroid": 1,
        "weighted_local_centroid": 1,
        "weighted_moments": 2,
        "weighted_moments_central": 2,
        "weighted_moments_hu": 1,
        "weighted_moments_normalized": 2
    }

    array_properties = [_k for _k, _v in region_properties_ndim_dict.items() if _v > 0]
    fixed_shape_properties = [_k for _k, _v in region_properties_shape_dict.items() if -1 not in _v]
    varied_shape_properties = [_k for _k, _v in region_properties_shape_dict.items() if -1 in _v]

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
        properties = ["area", "centroid"]

    if ( (properties == "all") or (properties == None) ):
        properties = region_properties_type_dict.keys()

    intensity_image = None
    if (len(args)) and (args[0]):
        intensity_image = args[0]
        args = args[1:]
    elif (len(kwargs)) and ("intensity_image" in kwargs):
        intensity_image = kwargs["intensity_image"]
        del kwargs["intensity_image"]
    else:
        pass

    # Remove duplicates and make sure label is at the front.
    properties = set(properties)
    allowed_properties = set(region_properties_type_dict.keys())
    disallowed_properties = properties.difference(allowed_properties)

    if len(disallowed_properties):
        disallowed_properties = sorted(disallowed_properties)
        raise Exception("Recieved \"" + repr(
            len(disallowed_properties)) + "\" properties that are not allowed, which are \"" + repr(
            disallowed_properties) + "\".")

    properties.discard("label")
    properties = ["label"] + sorted(properties)

    if new_label_image.size:
        # This gives a list of dictionaries. However, this is not very usable. So, we will convert this to a structured NumPy array.
        # In future versions, the properties argument will be removed. It does not need to be passed to retain functionality of this function.
        # new_label_image_props = skimage.measure.regionprops(new_label_image, intensity_image, *args, **kwargs)
        new_label_image_props = skimage.measure.regionprops(label_image = new_label_image,
                                                            properties = properties,
                                                            intensity_image = intensity_image,
                                                            *args, **kwargs)

        new_label_image_props_with_arrays = []
        for i in xrange(len(new_label_image_props)):
            new_label_image_props_with_arrays.append({})

            for each_key in properties:
                if each_key in array_properties:
                    new_label_image_props_with_arrays[i][each_key] = numpy.array(new_label_image_props[i][each_key])
                else:
                    new_label_image_props_with_arrays[i][each_key] = new_label_image_props[i][each_key]

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

                if isinstance(each_sample_value, numpy.ndarray) and (each_name in fixed_shape_properties):
                    each_type = each_sample_value.dtype
                    each_shape = each_sample_value.shape
                else:
                    each_type = numpy.dtype(each_type)

                new_label_image_props_with_arrays_dtype.append((each_name, each_type, each_shape))

            # Store the values to place in NumPy structured array in order.
            new_label_image_props_with_arrays_values = []
            for j in xrange(len(new_label_image_props_with_arrays)):
                # Add all values in order of keys from the dictionary.
                new_label_image_props_with_arrays_values.append([])
                for each_new_label_image_props_with_arrays_dtype in new_label_image_props_with_arrays_dtype:

                    each_name, each_type, each_shape = each_new_label_image_props_with_arrays_dtype

                    if each_shape:
                        new_label_image_props_with_arrays_values[j].append(
                            new_label_image_props_with_arrays[j][each_name].tolist())
                    else:
                        new_label_image_props_with_arrays_values[j].append(
                            new_label_image_props_with_arrays[j][each_name])

                # NumPy will expect a tuple for each set of values.
                new_label_image_props_with_arrays_values[j] = tuple(new_label_image_props_with_arrays_values[j])

    if (not new_label_image.size) or (not len(new_label_image_props_with_arrays)):
        new_label_image_props_with_arrays_dtype = []
        for each_key in properties:
            each_type = region_properties_type_dict[each_key]
            each_shape = region_properties_shape_dict[each_key]

            if each_key in varied_shape_properties:
                each_type = numpy.object_
                each_shape = tuple()

            new_label_image_props_with_arrays_dtype.append((each_key, each_type, each_shape))

    new_label_image_props_with_arrays_dtype = numpy.dtype(new_label_image_props_with_arrays_dtype)

    # Replace the properties with the structured array.
    new_label_image_props_with_arrays = numpy.array(new_label_image_props_with_arrays_values,
                                                    dtype = new_label_image_props_with_arrays_dtype)

    return(new_label_image_props_with_arrays)


@advanced_debugging.log_call(logger)
def get_neuron_dtype(new_image):
    neurons_dtype = [("mask", bool, new_image.shape),
                     ("contour", bool, new_image.shape),
                     # ("image", new_image.dtype, new_image.shape),
                     ("image", new_image.dtype, new_image.shape),
                     # ("segmentation", new_wavelet_image_denoised.dtype, new_wavelet_image_denoised.shape),
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


@advanced_debugging.log_call(logger)
def generate_local_maxima_vigra(new_intensity_image):
    """
        Creates a mask the same size as the intensity image with local maxima as True and background False.
        Uses vigra's vigra.analysis.extendedLocalMaxima.
        
        Args:
            new_intensity_image(numpy.ndarray):     The image to find local maxima for.
        
        Returns:
            numpy.ndarray:  A mask of the local maxima.
    """

    local_maxima_mask = vigra.analysis.extendedLocalMaxima(new_intensity_image.astype(numpy.float32)).astype(bool)

    return(local_maxima_mask)


@advanced_debugging.log_call(logger)
def generate_local_maxima_scikit_image(new_intensity_image, local_max_neighborhood_size = 1):
    """
        Creates a mask the same size as the intensity image with local maxima as True and background False.
        Uses scikit image's skimage.feature.peak_local_max.
        
        Args:
            new_intensity_image(numpy.ndarray):     The image to find local maxima for.
            neighborhood_size(int):                 Size of the neighborhood to check for a local maxima.
        
        Returns:
            numpy.ndarray:  A mask of the local maxima.
    """

    local_maxima_neighborhood = numpy.ones((2 * local_max_neighborhood_size + 1,) * new_intensity_image.ndim)
    local_maxima_mask = skimage.feature.peak_local_max(new_intensity_image, footprint = local_maxima_neighborhood,
                                                       labels = (new_intensity_image > 0).astype(int), indices = False)

    return(local_maxima_mask)


@advanced_debugging.log_call(logger)
def generate_local_maxima(new_intensity_image):
    """
        Creates a mask the same size as the intensity image with local maxima as True and background False.
        
        Args:
            new_intensity_image(numpy.ndarray):     The image to find local maxima for.
        
        Returns:
            numpy.ndarray:  A mask of the local maxima.
    """

    return(generate_local_maxima_vigra(new_intensity_image))


@advanced_debugging.log_call(logger)
def extended_region_local_maxima_properties(new_intensity_image, new_label_image = None, new_label_image_threshhold = 0,
                                            **kwargs):
    """
        Generates local maxima along with other properties for each labeled region (therefore at least one entry per label).
        Gets a label image if not provided by using the threshhold (if not provided is zero).
        
        
    """

    if new_label_image is None:
        new_label_image = scipy.ndimage.label(new_label_image > new_label_image_threshhold)[0]

    # Remove the background
    new_image_mask = (new_label_image != 0)
    new_intensity_image_masked = ( ~new_image_mask * new_intensity_image.min() ) + (
        new_image_mask * new_intensity_image )

    # Get a mask of the local maxima (that only includes local maxima not in the background)
    local_maxima_mask = generate_local_maxima(new_intensity_image_masked)

    # Count the local maxima and give them different labels
    local_maxima_labeled = scipy.ndimage.label(local_maxima_mask.astype(int))[0]

    # Generate the properties of the labeled regions
    labeled_props = region_properties(new_label_image, **kwargs)

    # Generate the properties of the local maxima
    local_maxima_props = region_properties(local_maxima_labeled, properties = ["label", "centroid"])

    # We want to have a few more type present in our NumPy structured array. To do this, we collect the existing types into
    # a list and then add our new types onto the end. Finally, we make the new structured array type from the list we have.
    props_dtype = []

    # Place on the label props dtype first
    labeled_props_dtype = advanced_numpy.numpy_array_dtype_list(labeled_props)
    props_dtype.extend(labeled_props_dtype)

    # Then add new fields.
    props_dtype.append(("local_max", int, new_intensity_image.ndim))
    props_dtype.append(("intensity", new_intensity_image.dtype))

    # Makes a new properties array that contains enough entries to hold the old one and has all the types we desire.
    new_local_maxima_props = numpy.zeros(local_maxima_props.shape, dtype = numpy.dtype(props_dtype))

    # Take the centroids and old labels
    new_local_maxima_props["label"] = local_maxima_props["label"]
    new_local_maxima_props["local_max"] = local_maxima_props["centroid"].round().astype(int)

    # Replace the old structured array with the enlarged version.
    local_maxima_props = new_local_maxima_props

    # Stores the value from wavelet denoising at the centroid for easy retrieval
    # Replace the labels by using the values from the label image
    if local_maxima_props.size:
        local_maxima_props["intensity"] = new_intensity_image[tuple(local_maxima_props["local_max"].T)]
        local_maxima_props["label"] = new_label_image[tuple(local_maxima_props["local_max"].T)]

    # Now, we want to merge the other properties in with our local maxima
    # But, we will skip it if there are no local maxima.
    if local_maxima_props.size:
        for each_i, each_label in enumerate(labeled_props["label"]):
            each_label_props_mask = (local_maxima_props["label"] == each_label)

            for each_new_prop_name, _, __ in labeled_props_dtype:
                local_maxima_props[each_new_prop_name][each_label_props_mask] = labeled_props[each_new_prop_name][
                    each_i]

    return(local_maxima_props)


class ExtendedRegionProps(object):
    @advanced_debugging.log_call(logger)
    def __init__(self, new_intensity_image, new_label_image, array_debug_logger = HDF5_logger.EmptyArrayLogger(), properties = ["centroid"]):
        self.intensity_image = new_intensity_image
        self.label_image = new_label_image
        self.image_mask = (self.label_image > 0)
        self.props = None
        self.count = None
        self.array_debug_logger = array_debug_logger

        logger.debug("Finding the local maxima and properties...")

        self.props = extended_region_local_maxima_properties(self.intensity_image, self.label_image,
                                                             properties = properties)

        logger.debug("Found the local maxima and properties.")

        # Remove maxima in the background
        background_maxima_mask = (self.props["label"] == 0)
        if (numpy.any(background_maxima_mask)):
            # There shouldn't be any maximums in the background. This should never happen.
            logger.warning(
                "Found \"" + (background_maxima_mask).sum() + "\" maximum(s) found in the background (label 0).")
            # Remove the 0 labels
            self.props = self.props[background_maxima_mask]

        del background_maxima_mask

        # Stores the number of times a particular label maxima appears.
        self.count = numpy.zeros((self.label_image.max(),), dtype = [("label", int), ("count", int)])
        # Get all the labels used in the label image
        self.count["label"] = numpy.arange(1, self.label_image.max() + 1)
        # Get the count of those labels (excluding zero as it is background)
        self.count["count"] = numpy.bincount(self.props["label"], minlength = len(self.count) + 1)[1:]

        if self.count.size and numpy.any(self.count["count"] == 0):
            # All labels should have a local maximum. If they don't, this could be a problem.

            failed_labels = self.count["label"][self.count["count"] == 0]
            failed_labels_list = failed_labels.tolist()
            failed_label_msg = "Label(s) not found in local maxima. For labels = " + repr(failed_labels_list) + "."

            logger.warning(failed_label_msg)

            self.array_debug_logger("intensity_image", self.intensity_image)
            self.array_debug_logger("label_image", self.label_image)

            if self.props.size:
                self.array_debug_logger("props", self.props)

            self.array_debug_logger("count", self.count)
            self.array_debug_logger("masks", advanced_numpy.all_permutations_equal(failed_labels, self.label_image))
            self.array_debug_logger("masks_labels", failed_labels)

            # with h5py.File("missing_labels.h5", "a") as fid:
            # curtime = time.time()
            #                curtime_str = str(curtime)
            #
            #                fid.create_group(curtime_str)
            #                fid[curtime_str]["intensity_image"] = self.intensity_image
            #                fid[curtime_str]["label_image"] = self.label_image
            #
            #                if self.props.size:
            #                    HDF5_serializers.write_numpy_structured_array_to_HDF5(fid, curtime_str + "/props", self.props)
            #
            #                HDF5_serializers.write_numpy_structured_array_to_HDF5(fid, curtime_str + "/count", self.count)
            #
            #                fid[curtime_str]["masks"] = advanced_numpy.all_permutations_equal(failed_labels, self.label_image)
            #
            #                for i, each_label in enumerate(failed_labels):
            #                    fid[curtime_str]["masks"].attrs[repr(i)] = repr(each_label)

            # Renumber labels. This way there are no labels without local maxima.
            self.renumber_labels()

        logger.debug("Refinined properties for local maxima.")


    @advanced_debugging.log_call(logger)
    def get_centroid_index_array(self):
        return(tuple(self.props["local_max"].T))


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
        new_centroid_label_image[self.get_centroid_index_array()] = self.props["label"]

        return(new_centroid_label_image)


    @advanced_debugging.log_call(logger)
    def remove_prop_mask(self, remove_prop_indices_mask):
        # Get the labels to remove
        remove_labels = self.props["label"][remove_prop_indices_mask]
        # Get how many of each label to remove
        label_count_to_remove = numpy.bincount(remove_labels, minlength = len(self.count) + 1)[1:]

        # Take a subset of the label props that does not include the removal mask
        # (copying may not be necessary as the mask may be as effective)
        self.props = self.props[~remove_prop_indices_mask].copy()
        # Reduce the count by the number of each label
        self.count["count"] -= label_count_to_remove

        # Mask over self.count to find labels that do not have centroid(s)
        inactive_label_count_mask = (self.count["count"] == 0)

        # Are there labels that do not exist now? If so, we will dump them.
        if inactive_label_count_mask.any():
            # Find the labels to remove from the label image and mask and remove them
            labels_to_remove = self.count["label"][inactive_label_count_mask]
            labels_to_remove_mask = advanced_numpy.contains(self.label_image, labels_to_remove)
            self.label_image[labels_to_remove_mask] = 0
            self.image_mask[labels_to_remove_mask] = 0
            self.intensity_image[labels_to_remove_mask] = 0

            # Renumber all labels sequentially starting with the label image
            self.renumber_labels()


    @advanced_debugging.log_call(logger)
    def remove_prop_indices(self, *i):
        # A mask of the indices to remove
        remove_prop_indices_mask = numpy.zeros((len(self.props),), dtype = bool)
        remove_prop_indices_mask[numpy.array(i)] = True

        self.remove_prop_mask(remove_prop_indices_mask)

    @advanced_debugging.log_call(logger)
    def renumber_labels(self):
        # Renumber all labels sequentially starting with the label image
        self.label_image[:], forward_label_mapping, reverse_label_mapping = skimage.segmentation.relabel_sequential(
            self.label_image)
        # new_label_image, forward_label_mapping, reverse_label_mapping = advanced_numpy.renumber_label_image(self.label_image)

        # Remove zero from the mappings as it is background and remains the same
        forward_label_mapping = forward_label_mapping[forward_label_mapping != 0]
        reverse_label_mapping = reverse_label_mapping[reverse_label_mapping != 0]

        # Find which of the old labels appear in self.props["label"] (skip 0)
        props_reverse_mapped = advanced_numpy.all_permutations_equal(reverse_label_mapping, self.props["label"])
        # Get the new labels by noting they must range from 0 to the length of reverse_label_mapping.
        # Skip zero as it is not necessary to check for it.
        new_labels = numpy.arange(1, len(reverse_label_mapping) + 1)
        # Expand new_labels into a view of the same size as props_reverse_mapped
        new_labels_expanded = advanced_numpy.expand_view(new_labels, reps_after = self.props["label"].shape)
        # Replace the labels with the matches and combine them
        self.props["label"] = (props_reverse_mapped * new_labels_expanded).sum(axis = 0)

        # Get a mask over the labels to find what is contained
        new_count_mask = advanced_numpy.contains(self.count["label"], reverse_label_mapping)
        # Move the values of the count into the proper lower labels and zero everything else
        self.count["count"][:len(reverse_label_mapping)] = self.count["count"][new_count_mask]
        self.count["count"][len(reverse_label_mapping):] = 0

        # (copying may not be necessary as the slice may be as effective)
        self.count = self.count[:len(reverse_label_mapping)].copy()


@advanced_debugging.log_call(logger)
def remove_low_intensity_local_maxima(local_maxima, **parameters):
    # Deleting local maxima that does not exceed the 90th percentile of the pixel intensities
    low_intensities__local_maxima_label_mask__to_remove = numpy.zeros(local_maxima.props.shape, dtype = bool)
    for i in xrange(len(local_maxima.props)):
        # Get the region with the label matching the maximum
        each_region_image_wavelet_mask = (local_maxima.label_image == local_maxima.props["label"][i])
        each_region_image_wavelet = local_maxima.intensity_image[each_region_image_wavelet_mask]

        # Get the number of pixels in that region
        each_region_image_wavelet_num_pixels = float(each_region_image_wavelet.size)

        # Get the value of the max for that region
        each_region_image_wavelet_centroid_value = local_maxima.props["intensity"][i]

        # Get a mask of the pixels below that max for that region
        each_region_image_wavelet_num_pixels_below_max = float(
            (each_region_image_wavelet < each_region_image_wavelet_centroid_value).sum())

        # Get a ratio of the number of pixels below that max for that region
        each_region_image_wavelet_ratio_pixels = each_region_image_wavelet_num_pixels_below_max / each_region_image_wavelet_num_pixels

        # If the ratio clears our threshhold, keep this label. Otherwise, eliminate it.
        low_intensities__local_maxima_label_mask__to_remove[i] = (
            each_region_image_wavelet_ratio_pixels < parameters["percentage_pixels_below_max"])

    new_local_maxima = copy.copy(local_maxima)

    new_local_maxima.remove_prop_mask(low_intensities__local_maxima_label_mask__to_remove)

    logger.debug("Removed low intensity maxima that are too close.")

    return(new_local_maxima)


@advanced_debugging.log_call(logger)
def remove_too_close_local_maxima(local_maxima, **parameters):
    # Deleting close local maxima below 16 pixels
    too_close__local_maxima_label_mask__to_remove = numpy.zeros(local_maxima.props.shape, dtype = bool)

    # Find the distance between every centroid (efficiently)
    local_maxima_pairs = numpy.array(list(itertools.combinations(xrange(len(local_maxima.props)), 2)))
    local_maxima_centroid_distance = scipy.spatial.distance.pdist(local_maxima.props["local_max"], metric = "euclidean")

    too_close_local_maxima_labels_mask = local_maxima_centroid_distance < parameters["min_centroid_distance"]
    too_close_local_maxima_pairs = local_maxima_pairs[too_close_local_maxima_labels_mask]

    for each_too_close_local_maxima_pairs in too_close_local_maxima_pairs:
        first_props_index, second_props_index = each_too_close_local_maxima_pairs

        if (local_maxima.props["label"][first_props_index] == local_maxima.props["label"][second_props_index]):
            if local_maxima.props["intensity"][first_props_index] < local_maxima.props["intensity"][second_props_index]:
                too_close__local_maxima_label_mask__to_remove[first_props_index] = True
            else:
                too_close__local_maxima_label_mask__to_remove[second_props_index] = True

    new_local_maxima = copy.copy(local_maxima)

    new_local_maxima.remove_prop_mask(too_close__local_maxima_label_mask__to_remove)

    logger.debug("Removed local maxima that are too close.")

    return(new_local_maxima)


@advanced_debugging.log_call(logger)
def wavelet_denoising(new_image, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **parameters):
    """
        Performs wavelet denoising on the given dictionary.
        
        Args:
            new_data(numpy.ndarray):      array of data for generating a dictionary (first axis is time).
            parameters(dict):             passed directly to spams.trainDL.
        
        Returns:
            dict: the dictionary found.
    """

    neurons = get_empty_neuron(new_image)

    new_wavelet_image_denoised_segmentation = None

    logger.debug("Started wavelet denoising.")
    logger.debug("Removing noise...")

    # Contains a bool array with significant values True and noise False.
    new_image_noise_estimate = denoising.estimate_noise(new_image,
                                                        **parameters["denoising.estimate_noise"])

    # Dictionary with wavelet transform applied. Wavelet transform is the first index.
    new_wavelet_transformed_image = wavelet_transform.wavelet_transform(new_image,
                                                                        **parameters["wavelet_transform.wavelet_transform"])

    for i in xrange(len(new_wavelet_transformed_image)):
        array_debug_logger("new_wavelet_transformed_image_" + repr(i), new_wavelet_transformed_image[i])

    # Contains a bool array with significant values True and noise False for all wavelet transforms.
    new_wavelet_transformed_image_significant_mask = denoising.significant_mask(new_wavelet_transformed_image,
                                                                                noise_estimate = new_image_noise_estimate,
                                                                                noise_threshhold = parameters[
                                                                                    "noise_threshhold"])

    for i in xrange(len(new_wavelet_transformed_image_significant_mask)):
        array_debug_logger("new_wavelet_transformed_image_significant_mask_" + repr(i), new_wavelet_transformed_image_significant_mask[i])

    new_wavelet_image_mask = new_wavelet_transformed_image_significant_mask[-1].copy()

    # Creates a new dictionary without the noise
    new_wavelet_image_denoised = new_wavelet_transformed_image[-1].copy()
    new_wavelet_image_denoised *= new_wavelet_image_mask

    array_debug_logger("new_wavelet_image_denoised_0", new_wavelet_image_denoised)

    logger.debug("Noise removed.")

    if new_wavelet_image_denoised.any():
        logger.debug("Frame has content other than noise.")

        logger.debug("Finding the label image...")

        # For holding the label image
        new_wavelet_image_denoised_labeled = scipy.ndimage.label(new_wavelet_image_denoised)[0]

        logger.debug("Found the label image.")
        logger.debug("Determining the properties of the label image...")

        # For holding the label image properties
        new_wavelet_image_denoised_labeled_props = region_properties(new_wavelet_image_denoised_labeled,
                                                                     properties = parameters[
                                                                         "accepted_region_shape_constraints"].keys())

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
        labels_not_within_bound = new_wavelet_image_denoised_labeled_props["label"][not_within_bound]

        # Iterate over the unbounded ones to fix any errors.
        for each_labels_not_within_bound in labels_not_within_bound:
            # Get a mask for the current label
            current_label_mask = ( new_wavelet_image_denoised_labeled == each_labels_not_within_bound )

            # Get a lower wavelet mask
            lower_wavelet_mask = new_wavelet_transformed_image_significant_mask[-2]

            # Replacement mask
            replacement_mask = current_label_mask & lower_wavelet_mask

            # Zero everything that is not in the replacement region and then use the lower transformed wavelet.
            new_wavelet_image_denoised_replacement = new_wavelet_transformed_image[-2] * replacement_mask

            # Overwrite the area in our old labeled mask to match this lower wavelet transform
            new_wavelet_image_mask[current_label_mask] = lower_wavelet_mask[current_label_mask]

            # However, overwrite the previously labeled area completely (will push more things into the background)
            new_wavelet_image_denoised[current_label_mask] = new_wavelet_image_denoised_replacement[current_label_mask]

        logger.debug("Reduced wavelet transform on regions outside of constraints...")

        logger.debug("Finding new label image...")

        new_wavelet_image_denoised_label_image = scipy.ndimage.label(new_wavelet_image_denoised > 0)[0]

        logger.debug("Found new label image.")

        extended_region_props_0_array_debug_logger = HDF5_logger.create_subgroup_HDF5_array_logger(
            "extended_region_props_0", array_debug_logger)

        local_maxima = ExtendedRegionProps(new_wavelet_image_denoised, new_wavelet_image_denoised_label_image,
                                           array_debug_logger = extended_region_props_0_array_debug_logger)

        local_maxima.label_image
        array_debug_logger("local_maxima_label_image_0", local_maxima.label_image)
        array_debug_logger("local_maxima_label_image_contours_0", advanced_numpy.generate_labeled_contours(local_maxima.label_image > 0))

        local_maxima = remove_low_intensity_local_maxima(local_maxima, **parameters["remove_low_intensity_local_maxima"])

        array_debug_logger("local_maxima_label_image_1", local_maxima.label_image)
        array_debug_logger("local_maxima_label_image_contours_1", advanced_numpy.generate_labeled_contours(local_maxima.label_image > 0))

        local_maxima = remove_too_close_local_maxima(local_maxima, **parameters["remove_too_close_local_maxima"])

        array_debug_logger("local_maxima_label_image_2", local_maxima.label_image)
        array_debug_logger("local_maxima_label_image_contours_2", advanced_numpy.generate_labeled_contours(local_maxima.label_image > 0))

        if local_maxima.props.size:
            if parameters["use_watershed"]:
                # ############### TODO: Revisit to make sure all of Ferran's algorithm is implemented and this is working properly.

                # Perform the watershed segmentation.

                # First perform disc opening on the image. (Actually, we don't do this.)
                # new_wavelet_image_denoised_opened = vigra.filters.discOpening(new_wavelet_image_denoised.astype(numpy.float32), radius = 1)

                # We could look for seeds using local maxima. However, we already know what these should be as these are the centroids we have found.
                new_wavelet_image_denoised_maxima = local_maxima.get_centroid_label_image()

                # Segment with watershed on minimum image
                # Use seeds from centroids of local minima
                # Also, include mask
                new_wavelet_image_denoised_segmentation = skimage.morphology.watershed(local_maxima.intensity_image,
                                                                                       new_wavelet_image_denoised_maxima,
                                                                                       mask = (local_maxima.intensity_image > 0))

                array_debug_logger("watershed_segmentation", new_wavelet_image_denoised_segmentation)
                array_debug_logger("watershed_segmentation_contours", advanced_numpy.generate_labeled_contours(new_wavelet_image_denoised_segmentation))

                extended_region_props_1_array_debug_logger = HDF5_logger.create_subgroup_HDF5_array_logger(
                    "extended_region_props_1", array_debug_logger)

                watershed_local_maxima = ExtendedRegionProps(local_maxima.intensity_image,
                                                             new_wavelet_image_denoised_segmentation,
                                                             array_debug_logger = extended_region_props_1_array_debug_logger,
                                                             properties = ["centroid"] + parameters["accepted_neuron_shape_constraints"].keys())

                array_debug_logger("watershed_local_maxima_label_image_0", watershed_local_maxima.label_image)
                array_debug_logger("watershed_local_maxima_label_image_contours_0", advanced_numpy.generate_labeled_contours(watershed_local_maxima.label_image > 0))

                array_debug_logger("watershed_local_maxima_props_0", watershed_local_maxima.props)
                array_debug_logger("watershed_local_maxima_count_0", watershed_local_maxima.count)

                # if watershed_local_maxima.props.size:
                #     array_debug_logger("watershed_local_maxima_props_0", watershed_local_maxima.props)
                # if watershed_local_maxima.count.size:
                #     array_debug_logger("watershed_local_maxima_count_0", watershed_local_maxima.count)


                # Renumber all labels sequentially
                #new_wavelet_image_denoised_segmentation = skimage.segmentation.relabel_sequential(new_wavelet_image_denoised_segmentation)[0]

                # Get the regions created in segmentation (drop zero as it is the background)
                #new_wavelet_image_denoised_segmentation_regions = numpy.unique(new_wavelet_image_denoised_segmentation[new_wavelet_image_denoised_segmentation != 0])
                #new_wavelet_image_denoised_segmentation_regions = watershed_local_maxima.count[watershed_local_maxima.count["count"] > 0]

                ## Drop the first two as 0's are the region edges and 1's are the background. Not necessary in our code as seeds for the watershed are set.
                #new_wavelet_image_denoised_segmentation[new_wavelet_image_denoised_segmentation == 1] = 0
                #new_wavelet_image_denoised_segmentation_regions = new_wavelet_image_denoised_segmentation_regions[2:]

                ## TODO: Replace with numpy.bincount.

                # Renumber all labels sequentially.
                #new_wavelet_image_denoised_segmentation, forward_label_mapping, reverse_label_mapping = skimage.segmentation.relabel_sequential(new_wavelet_image_denoised_segmentation)

                # Find properties of all regions
                #new_wavelet_image_denoised_segmentation_props = region_properties(new_wavelet_image_denoised_segmentation, properties = ["centroid"] + parameters["accepted_neuron_shape_constraints"].keys())

                #new_wavelet_image_denoised_segmentation_props["label"] = reverse_label_mapping[ new_wavelet_image_denoised_segmentation_props["label"] ]


                #new_wavelet_image_denoised_segmentation_props_labels_count = numpy.bincount(new_wavelet_image_denoised_segmentation_props["label"])[1:]

                #new_wavelet_image_denoised_segmentation_props_labels_duplicates_mask = (new_wavelet_image_denoised_segmentation_props_labels_count > 1)
                #new_wavelet_image_denoised_segmentation_props_labels_duplicates = numpy.unique(new_wavelet_image_denoised_segmentation_props["label"][new_wavelet_image_denoised_segmentation_props_labels_duplicates_mask])

                new_watershed_local_maxima_count_duplicates_mask = (watershed_local_maxima.count["count"] > 1)
                new_watershed_local_maxima_count_duplicate_labels = watershed_local_maxima.count["label"][
                    new_watershed_local_maxima_count_duplicates_mask]
                new_watershed_local_maxima_props_duplicates_mask = advanced_numpy.contains(
                    watershed_local_maxima.props["label"], new_watershed_local_maxima_count_duplicate_labels)
                watershed_local_maxima.remove_prop_mask(new_watershed_local_maxima_props_duplicates_mask)

                array_debug_logger("watershed_local_maxima_label_image_1", watershed_local_maxima.label_image)
                array_debug_logger("watershed_local_maxima_label_image_contours_1", advanced_numpy.generate_labeled_contours(watershed_local_maxima.label_image > 0))

                if watershed_local_maxima.props.size:
                    array_debug_logger("watershed_local_maxima_props_1", watershed_local_maxima.props)
                if watershed_local_maxima.count.size:
                    array_debug_logger("watershed_local_maxima_count_1", watershed_local_maxima.count)

                not_within_bound = numpy.zeros(watershed_local_maxima.props.shape, dtype = bool)

                # Go through each property and make sure they are within the bounds
                for each_prop in parameters["accepted_neuron_shape_constraints"]:
                    # Get lower and upper bounds for the current property
                    lower_bound = parameters["accepted_neuron_shape_constraints"][each_prop]["min"]
                    upper_bound = parameters["accepted_neuron_shape_constraints"][each_prop]["max"]

                    # Determine whether lower or upper bound is satisfied
                    is_lower_bounded = lower_bound <= watershed_local_maxima.props[each_prop]
                    is_upper_bounded = watershed_local_maxima.props[each_prop] <= upper_bound

                    # See whether both or neither bound is satisfied.
                    is_within_bound = is_lower_bounded & is_upper_bounded
                    is_not_within_bound = ~is_within_bound

                    # Collect the unbounded ones
                    not_within_bound |= is_not_within_bound

                # Get labels outside of bounds and remove them
                #new_wavelet_image_denoised_segmentation_props_unbounded_labels = watershed_local_maxima.props["label"][not_within_bound]
                watershed_local_maxima.remove_prop_mask(not_within_bound)

                array_debug_logger("watershed_local_maxima_label_image_2", watershed_local_maxima.label_image)
                array_debug_logger("watershed_local_maxima_label_image_contours_2", advanced_numpy.generate_labeled_contours(watershed_local_maxima.label_image > 0))

                if watershed_local_maxima.props.size:
                    array_debug_logger("watershed_local_maxima_props_2", watershed_local_maxima.props)
                if watershed_local_maxima.count.size:
                    array_debug_logger("watershed_local_maxima_count_2", watershed_local_maxima.count)

                if watershed_local_maxima.props.size:
                    # Creates a NumPy structure array to store
                    neurons = numpy.zeros(len(watershed_local_maxima.props), dtype = neurons.dtype)

                    # Get masks for all cells
                    neurons["mask"] = advanced_numpy.all_permutations_equal(watershed_local_maxima.props["label"],
                                                                            new_wavelet_image_denoised_segmentation)

                    #neurons["image"] = new_wavelet_image_denoised * neurons["mask"]

                    neurons["image"] = new_image * neurons["mask"]

                    neurons["area"] = watershed_local_maxima.props["area"]

                    neurons["max_F"] = neurons["image"].reshape((neurons["image"].shape[0], -1)).max(axis = 1)

                    for i in xrange(len(neurons)):
                        neuron_mask_i_points = numpy.array(neurons["mask"][i].nonzero())

                        neurons["contour"][i] = advanced_numpy.generate_contour(neurons["mask"][i])
                        neurons["gaussian_mean"][i] = neuron_mask_i_points.mean(axis = 1)
                        neurons["gaussian_cov"][i] = numpy.cov(neuron_mask_i_points)

                    neurons["centroid"] = watershed_local_maxima.props["centroid"]

                    if len(neurons) > 1:
                        logger.debug("Extracted neurons. Found " + str(len(neurons)) + " neurons.")
                    else:
                        logger.debug("Extracted a neuron. Found " + str(len(neurons)) + " neuron.")

                    array_debug_logger("new_neuron_set", neurons)
            else:
                # ################### Some other kind of segmentation??? Talked to Ferran and he said don't worry about implementing this for now. Does not seem to give noticeably better results.
                raise Exception("No other form of segmentation is implemented.")
        else:
            logger.debug("No local maxima left that are acceptable neurons.")
    else:
        logger.debug("Frame is only noise.")

    logger.debug("Finished making neurons for the current frame.")
    return(neurons)


@advanced_debugging.log_call(logger)
def fuse_neurons(neuron_1, neuron_2, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **parameters):
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

    array_debug_logger("neuron_1", neuron_1)
    array_debug_logger("neuron_2", neuron_2)

    assert (neuron_1.shape == neuron_2.shape == tuple())
    assert (neuron_1.dtype == neuron_2.dtype)

    mean_neuron = numpy.array([neuron_1["image"], neuron_2["image"]]).mean(axis = 0)
    mean_neuron_mask = mean_neuron > (parameters["fraction_mean_neuron_max_threshold"] * mean_neuron.max())

    # Gaussian mixture model ??? Skipped this.

    # Creates a NumPy structure array to store
    new_neuron = numpy.zeros(neuron_1.shape, dtype = neuron_1.dtype)

    new_neuron["mask"] = mean_neuron_mask

    new_neuron["contour"] = advanced_numpy.generate_contour(new_neuron["mask"])

    new_neuron["image"] = mean_neuron * new_neuron["mask"]

    new_neuron["area"] = (new_neuron["mask"] > 0).sum()

    new_neuron["max_F"] = new_neuron["image"].max()

    new_neuron_mask_points = numpy.array(new_neuron["mask"].nonzero())

    new_neuron["gaussian_mean"] = new_neuron_mask_points.mean(axis = 1)
    new_neuron["gaussian_cov"] = numpy.cov(new_neuron_mask_points)

    #    for i in xrange(len(new_neuron)):
    #        new_neuron_mask_points = numpy.array(new_neuron["mask"][i].nonzero())
    #        
    #        new_neuron["gaussian_mean"][i] = new_neuron_mask_points.mean(axis = 1)
    #        new_neuron["gaussian_cov"][i] = numpy.cov(new_neuron_mask_points)

    new_neuron["centroid"] = new_neuron["gaussian_mean"]

    array_debug_logger("new_neuron", new_neuron)

    return(new_neuron)


@advanced_debugging.log_call(logger)
def merge_neuron_sets(new_neuron_set_1, new_neuron_set_2, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **parameters):
    """
        Merges the two sets of neurons into one (treats the first with preference).
        
        Args:
            neuron_1(numpy.ndarray):      first neuron (prefered for tie breaking).
            neuron_2(numpy.ndarray):      second neuron (one to merge in).
            parameters(dict):             dictionary of parameters
        
        Returns:
            dict: the dictionary found.
    """

    if new_neuron_set_1.size:
        array_debug_logger("new_neuron_set_1", new_neuron_set_1)

    if new_neuron_set_2.size:
        array_debug_logger("new_neuron_set_2", new_neuron_set_2)

    assert (new_neuron_set_1.dtype == new_neuron_set_2.dtype)


    # TODO: Reverse if statement so it is not nots
    if len(new_neuron_set_1) and len(new_neuron_set_2):
        logger.debug("Have 2 sets of neurons to merge.")

        new_neuron_set = new_neuron_set_1.copy()

        new_neuron_set_1_flattened = new_neuron_set_1["image"].reshape(
            new_neuron_set_1["image"].shape[0], -1)
        new_neuron_set_2_flattened = new_neuron_set_2["image"].reshape(
            new_neuron_set_2["image"].shape[0], -1)

        array_debug_logger("new_neuron_set_1_flattened", new_neuron_set_1_flattened)
        array_debug_logger("new_neuron_set_2_flattened", new_neuron_set_2_flattened)

        new_neuron_set_1_flattened_mask = new_neuron_set_1["mask"].reshape(new_neuron_set_1["mask"].shape[0], -1)
        new_neuron_set_2_flattened_mask = new_neuron_set_2["mask"].reshape(new_neuron_set_2["mask"].shape[0], -1)

        array_debug_logger("new_neuron_set_1_flattened_mask", new_neuron_set_1_flattened_mask)
        array_debug_logger("new_neuron_set_2_flattened_mask", new_neuron_set_2_flattened_mask)

        # Measure the normalized dot product between any two neurons (i.e. related to the angle of separation)
        new_neuron_set_angle = advanced_numpy.dot_product_L2_normalized(new_neuron_set_1_flattened,
                                                                        new_neuron_set_2_flattened)

        array_debug_logger("new_neuron_set_angle", new_neuron_set_angle)

        # Measure the distance between the two masks (note distance relative to the total mask content of each mask individually)
        new_neuron_set_masks_overlaid_1, new_neuron_set_masks_overlaid_2 = advanced_numpy.dot_product_partially_normalized(
            new_neuron_set_1_flattened_mask, new_neuron_set_2_flattened_mask, ord = 1)

        array_debug_logger("new_neuron_set_masks_overlaid_1", new_neuron_set_masks_overlaid_1)
        array_debug_logger("new_neuron_set_masks_overlaid_2", new_neuron_set_masks_overlaid_2)

        # Now that the three measures for the correlation method have been found, we want to know,
        # which are the best correlated neurons between the two sets using these measures.
        # This done to find the neuron in new_neuron_set_1 that best matches each neuron in new_neuron_set_2.
        new_neuron_set_angle_all_optimal_i = new_neuron_set_angle.argmax(axis = 0)
        new_neuron_set_masks_overlaid_1_all_optimal_i = new_neuron_set_masks_overlaid_1.argmax(axis = 0)
        new_neuron_set_masks_overlaid_2_all_optimal_i = new_neuron_set_masks_overlaid_2.argmax(axis = 0)

        array_debug_logger("new_neuron_set_angle_all_optimal_i", new_neuron_set_angle_all_optimal_i)
        array_debug_logger("new_neuron_set_masks_overlaid_1_all_optimal_i",
                           new_neuron_set_masks_overlaid_1_all_optimal_i)
        array_debug_logger("new_neuron_set_masks_overlaid_2_all_optimal_i",
                           new_neuron_set_masks_overlaid_2_all_optimal_i)

        # Get all the j indices
        new_neuron_set_all_j = numpy.arange(len(new_neuron_set_2))

        array_debug_logger("new_neuron_set_all_j", new_neuron_set_all_j)

        # Get the maximum corresponding to the best matched paris from before
        new_neuron_set_angle_maxes = new_neuron_set_angle[(new_neuron_set_angle_all_optimal_i, new_neuron_set_all_j,)]
        new_neuron_set_masks_overlaid_1_maxes = new_neuron_set_masks_overlaid_1[
            (new_neuron_set_masks_overlaid_1_all_optimal_i, new_neuron_set_all_j,)]
        new_neuron_set_masks_overlaid_2_maxes = new_neuron_set_masks_overlaid_2[
            (new_neuron_set_masks_overlaid_2_all_optimal_i, new_neuron_set_all_j,)]

        array_debug_logger("new_neuron_set_angle_maxes", new_neuron_set_angle_maxes)
        array_debug_logger("new_neuron_set_masks_overlaid_1_maxes", new_neuron_set_masks_overlaid_1_maxes)
        array_debug_logger("new_neuron_set_masks_overlaid_2_maxes", new_neuron_set_masks_overlaid_2_maxes)

        # Store a list of the optimal neurons in the existing set to fuse with (by default set all values to -1)
        new_neuron_set_all_optimal_i = numpy.zeros((len(new_neuron_set_2),), dtype = int)
        new_neuron_set_all_optimal_i -= 1

        array_debug_logger("new_neuron_set_all_optimal_i_0", new_neuron_set_all_optimal_i)

        # Create the masks to use for getting the proper indices
        new_neuron_set_angle_maxes_significant = numpy.zeros((len(new_neuron_set_2),), dtype = bool)
        new_neuron_set_masks_overlaid_1_maxes_significant = numpy.zeros((len(new_neuron_set_2),), dtype = bool)
        new_neuron_set_masks_overlaid_2_maxes_significant = numpy.zeros((len(new_neuron_set_2),), dtype = bool)

        array_debug_logger("new_neuron_set_angle_maxes_significant_0", new_neuron_set_angle_maxes_significant)
        array_debug_logger("new_neuron_set_masks_overlaid_1_maxes_significant_0",
                           new_neuron_set_masks_overlaid_1_maxes_significant)
        array_debug_logger("new_neuron_set_masks_overlaid_2_maxes_significant_0",
                           new_neuron_set_masks_overlaid_2_maxes_significant)

        # Get masks that indicate which measurements have the best matching neuron
        new_neuron_set_angle_maxes_significant[
            new_neuron_set_angle_maxes > parameters["alignment_min_threshold"]] = True
        new_neuron_set_masks_overlaid_1_maxes_significant[~new_neuron_set_angle_maxes_significant & (
            new_neuron_set_masks_overlaid_2_maxes > parameters["overlap_min_threshold"])] = True
        new_neuron_set_masks_overlaid_2_maxes_significant[
            ~new_neuron_set_angle_maxes_significant & ~new_neuron_set_masks_overlaid_1_maxes_significant & (
                new_neuron_set_masks_overlaid_1_maxes > parameters["overlap_min_threshold"])] = True

        array_debug_logger("new_neuron_set_angle_maxes_significant_1", new_neuron_set_angle_maxes_significant)
        array_debug_logger("new_neuron_set_masks_overlaid_1_maxes_significant_1",
                           new_neuron_set_masks_overlaid_1_maxes_significant)
        array_debug_logger("new_neuron_set_masks_overlaid_2_maxes_significant_1",
                           new_neuron_set_masks_overlaid_2_maxes_significant)

        # Using the masks construct the best match neuron index for each case
        # After doing these three, new_neuron_set_all_optimal_i will contain either
        # the index of the neuron to fuse with in new_neuron_set for each 
        new_neuron_set_all_optimal_i[new_neuron_set_angle_maxes_significant] = new_neuron_set_angle_all_optimal_i[new_neuron_set_angle_maxes_significant]

        array_debug_logger("new_neuron_set_all_optimal_i_1", new_neuron_set_all_optimal_i)

        new_neuron_set_all_optimal_i[new_neuron_set_masks_overlaid_1_maxes_significant] = new_neuron_set_masks_overlaid_1_all_optimal_i[new_neuron_set_masks_overlaid_1_maxes_significant]

        array_debug_logger("new_neuron_set_all_optimal_i_2", new_neuron_set_all_optimal_i)

        new_neuron_set_all_optimal_i[new_neuron_set_masks_overlaid_2_maxes_significant] = new_neuron_set_masks_overlaid_2_all_optimal_i[new_neuron_set_masks_overlaid_2_maxes_significant]

        array_debug_logger("new_neuron_set_all_optimal_i_3", new_neuron_set_all_optimal_i)


        # Separate all the best matches that were found from those that were not.
        # Also, remove the -1 as they have served their purpose.
        new_neuron_set_all_optimal_i_found = (new_neuron_set_all_optimal_i != -1)
        new_neuron_set_all_j_fuse = new_neuron_set_all_j[new_neuron_set_all_optimal_i_found]
        new_neuron_set_all_j_append = new_neuron_set_all_j[~new_neuron_set_all_optimal_i_found]
        new_neuron_set_all_optimal_i = new_neuron_set_all_optimal_i[new_neuron_set_all_optimal_i_found]

        array_debug_logger("new_neuron_set_all_optimal_i_found", new_neuron_set_all_optimal_i_found)

        if new_neuron_set_all_j_fuse.size:
            array_debug_logger("new_neuron_set_all_j_fuse", new_neuron_set_all_j_fuse)

        if new_neuron_set_all_j_append.size:
            array_debug_logger("new_neuron_set_all_j_append", new_neuron_set_all_j_append)

        if new_neuron_set_all_optimal_i.size:
            array_debug_logger("new_neuron_set_all_optimal_i_4", new_neuron_set_all_optimal_i)

        # Fuse all the neurons that can be from new_neuron_set_2 to the new_neuron_set (composed of new_neuron_set_1)
        for i, j in itertools.izip(new_neuron_set_all_optimal_i, new_neuron_set_all_j_fuse):
            new_fusing_neurons_array_debug_logger = HDF5_logger.create_subgroup_HDF5_array_logger("__".join(["fusing_neurons",
                                                                                                                   "new_neuron_set_1_" + str(i),
                                                                                                                   "new_neuron_set_2_" + str(j)]),
                                                                                                        array_debug_logger)

            new_neuron_set[i] = fuse_neurons(new_neuron_set_1[i], new_neuron_set_2[j], new_fusing_neurons_array_debug_logger, **parameters["fuse_neurons"])

        logger.debug("Fused \"" + repr(len(new_neuron_set_all_j_fuse)) + "\" neurons to the existing set.")

        # Tack on the ones that must be appended
        new_neuron_set = numpy.hstack([new_neuron_set, new_neuron_set_2[new_neuron_set_all_j_append]])

        logger.debug("Added \"" + repr(len(new_neuron_set_all_j_append)) + "\" new neurons to the existing set.")

    elif len(new_neuron_set_1):
        logger.debug("Have 1 set of neurons to merge. Only the first set has neurons.")
        new_neuron_set = new_neuron_set_1
    elif len(new_neuron_set_2):
        logger.debug("Have 1 set of neurons to merge. Only the second set has neurons.")
        new_neuron_set = new_neuron_set_2
    else:
        logger.debug("Have 0 sets of neurons to merge.")
        new_neuron_set = new_neuron_set_1

    if new_neuron_set.size:
        array_debug_logger("new_merged_neurons_set", new_neuron_set)

    return(new_neuron_set)


@advanced_debugging.log_call(logger)
def postprocess_data(new_dictionary, array_debug_logger = HDF5_logger.EmptyArrayLogger(), **parameters):
    """
        Generates neurons from the dictionary.
        
        Args:
            new_dictionary(numpy.ndarray):        dictionary of basis images to analyze.
            array_debug_logger(callable):         logger for array debug output.
            parameters(dict):                     dictionary of parameters
        
        Returns:
            numpy.ndarray:                        structured array with relevant information for each neuron.
    """


    new_neurons_set = None


    # Puts each dictionary basis debug log into a separate group depending on which basis image it was a part of.
    def array_debug_logger_enumerator(new_list):
        neuron_sets_array_debug_logger = HDF5_logger.create_subgroup_HDF5_array_logger("neuron_sets",
                                                                                             array_debug_logger)

        for i, i_str, each in advanced_iterators.filled_stringify_enumerate(new_list):
            yield ( (i, each, HDF5_logger.create_subgroup_HDF5_array_logger(i_str, neuron_sets_array_debug_logger)) )

    # Get all neurons for all images
    new_neurons_set = get_empty_neuron(new_dictionary[0])
    unmerged_neuron_set = get_empty_neuron(new_dictionary[0])
    for i, each_new_dictionary_image, each_array_debug_logger in array_debug_logger_enumerator(new_dictionary):
        each_new_neuron_set = wavelet_denoising(each_new_dictionary_image, array_debug_logger = each_array_debug_logger,
                                                **parameters["wavelet_denoising"])

        logger.debug("Denoised a set of neurons from frame " + str(i + 1) + " of " + str(len(new_dictionary)) + ".")

        unmerged_neuron_set = numpy.hstack([unmerged_neuron_set, each_new_neuron_set])
        new_neurons_set = merge_neuron_sets(new_neurons_set, each_new_neuron_set, array_debug_logger = each_array_debug_logger,
                                            **parameters["merge_neuron_sets"])

        logger.debug("Merged a set of neurons from frame " + str(i + 1) + " of " + str(len(new_dictionary)) + ".")

    array_debug_logger("unmerged_neuron_set", unmerged_neuron_set)

    array_debug_logger("new_neurons_set", new_neurons_set)

    unmerged_neuron_set_contours = unmerged_neuron_set["contour"]
    unmerged_neuron_set_contours = (unmerged_neuron_set_contours * advanced_numpy.expand_view(numpy.arange(1, 1 + len(unmerged_neuron_set_contours)), reps_after = unmerged_neuron_set_contours.shape[1:])).max(axis = 0)

    array_debug_logger("unmerged_neuron_set_contours", unmerged_neuron_set_contours)

    new_neurons_set_contours = new_neurons_set["contour"]
    new_neurons_set_contours = (new_neurons_set_contours * advanced_numpy.expand_view(numpy.arange(1, 1 + len(new_neurons_set_contours)), reps_after = new_neurons_set_contours.shape[1:])).max(axis = 0)

    array_debug_logger("new_neurons_set_contours", new_neurons_set_contours)


    return(new_neurons_set)
