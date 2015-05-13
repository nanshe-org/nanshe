"""
``random_dictionary_learning_data`` supports generating synthetic data.

===============================================================================
Overview
===============================================================================
The module ``random_dictionary_learning_data`` provides a way of generating
synthetic data for testing the segmentation algorithm against. Moving forward
the useful content in here will be refactored and moved into the ``data``
module. So, depending on this module is unwise.

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "Apr 17, 2014 11:27:08 EDT$"


import warnings

warnings.warn(
    "The module `random_dictionary_learning_data` is deprecated." +
    "Please consider using `data` instead." +
    "Relevant content from this module is being refactored and moved there.",
    DeprecationWarning
)

import numpy
import scipy
import scipy.ndimage

import nanshe.util.prof
import nanshe.util.iters


# Get the logger
trace_logger = nanshe.util.prof.getTraceLogger(__name__)



class MappingDiscreteUniformDistributionGenerator(object):
    """
        Given a bunch of arguments. This will create a random element generator
        that returns one or many.
    """

    def __init__(self, *args):
        """
            Builds a random element generator.

            Args:
                *args:  Anything that needs to be drawn from.
        """

        self.args = args

    def __call__(self, size=1):
        """
            Draws a certain number of elements with equal likelihood and
            returns them in a list.

            Args:
                size(int):       Number of elements to draw (one if
                                 unspecified).

            Returns:
                results(list):   a list of arguments drawn (None if no
                                 arguments)
        """

        results = []

        if len(self.args) > 0:
            indices = numpy.random.randint(0, len(self.args), size)

            results = [self.args[_] for _ in indices]
        else:
            results = [None for _ in xrange(size)]

        return(results)


class NumpyRandomArrayDiscreteUniformDistributionGenerator(object):
    """
        Creates a random numpy array (with type bool) generator that will set
        a certain number of random positions in the array to True (like
        Bernoulli random distribution)
    """

    def __init__(self, shape):
        """
            Creates a random numpy array generator that returns arrays of a
            certain shape.

            Args:
                shape(tuple):   A tuple permissible for setting the size of a
                                numpy array.
        """

        self.shape = shape

    def __call__(self, size=1):
        """
            Generates a numpy array.

            Args:
                size(int):                Number of Trues to have in the numpy
                                          array.

            Returns:
                results(numpy.ndarray):   a boolean numpy array with a fixed
                                          number of randomly placed Trues
        """

        # A completely empty numpy array
        results = numpy.zeros(self.shape, dtype=bool)

        # Gets a set of random indices that need to be non-zero
        indices = tuple([
            numpy.random.randint(0, each_dim, size) for each_dim in self.shape
        ])

        # Makes them non-zero
        results[indices] = True

        return(results)


class MappingDiscreteGeometricDistributionGenerator(object):
    """
        A random generator of groups. Each group has a size that is
        geometrically distributed. However, the individuals chosen for the
        group are all equally likely.
    """

    def __init__(self, *args):
        """
            Sets the arguments for use to compose the groups.

            Args:
                *args:  Any variety of useful items for drawing.
        """

        self.args = args

    def __call__(self, p, size=1):
        """
            Generates a number of groups equal to size with each group size
            being distributed geometrically by p.

            Args:
                p(float)         the probability of success for a geometric
                                 distribution (starts with 1 so has mean 1/p).

                size(int)        the number of groups to make

            Returns:
                results(list):   a list of groups of arguments drawn (None if
                                 no arguments)
        """

        # Get a uniform distribution over the elements to fill each group.
        uni_gen = MappingDiscreteUniformDistributionGenerator(*self.args)

        # Draw the sizes for each group
        group_sizes = numpy.random.geometric(p, size)

        # Using the sizes draw element to fill groups up to the right size
        results = [uni_gen(group_sizes[i]) for i in xrange(size)]

        return(results)





class DictionaryLearningRandomDataSample(object):
    """
        Essentially a struct with its values set at runtime by
        DictionaryLearningRandomDataGenerator calls.
    """
    def __init__(self):
        """
            Default constructor just to establish values.
        """

        self.points = None
        self.centroid_activation_frames = None
        self.noiseless_frames = None
        self.frames = None


class DictionaryLearningRandomDataGenerator(object):
    """
        A Random Generator that build pseudo-data similar in nature to that
        which the ADINA algorithm is run.
    """

    def __init__(self,
                 frame_shape,
                 num_objects,
                 num_groups,
                 num_frames,
                 mean_group_size,
                 object_spread,
                 object_max_intensity,
                 object_min_intensity,
                 background_noise_intensity):
        """
            Builds a DictionaryLearningRandomDataGenerator for draws.

            Args:
                frame_shape(tuple)                  a tuple of ints for
                                                    constructing a numpy array

                num_objects(int)                    the number of objects that
                                                    can possible be active
                                                    (i.e. neurons present
                                                    whether active or not)

                num_groups(int)                     number of groups of objects
                                                    that will be active (i.e.
                                                    number of groups of neurons
                                                    seen to be active)

                num_frames(int)                     number of frames for any
                                                    group to be active in the
                                                    pseudo-video

                mean_group_size(float)              average group size (average
                                                    for a geometric
                                                    distribution)

                object_spread(float)                how big an object is on
                                                    average

                object_max_intensity(float)         the highest intensity
                                                    possible

                object_min_intensity(float)         the lowest intensity
                                                    possible

                background_noise_intensity(float)   how much noise there is in
                                                    the background.
        """

        self.frame_shape = frame_shape
        self.num_objects = num_objects
        self.num_groups = num_groups
        self.num_frames = num_frames
        self.mean_group_size = mean_group_size
        self.object_spread = object_spread
        self.object_max_intensity = object_max_intensity
        self.object_min_intensity = object_min_intensity
        self.background_noise_intensity = background_noise_intensity

        self.object_intensity_range = self.object_max_intensity - \
                                      self.object_min_intensity

    def __call__(self, num_runs=1, seed=None):
        """
            Constructs a series of pseudo-videos.

            Args:
                num_runs(int):          number of pseudo-videos to generate
                seed(int):              uses the seed for numpy.random.seed if
                                        provided.

            Returns:
                results(list):          a list of
                                        DictionaryLearningRandomDataSample
                                        instances with relevant data from
                                        generation included.
        """

        # Use the seed provided.
        numpy.random.seed(seed)

        # A list of DictionaryLearningRandomDataSample instances
        results = []

        for i in xrange(num_runs):
            # Where the result will be stored
            each_result = DictionaryLearningRandomDataSample()

            # Generates a numpy array that has a shape of self.frame_shape with
            # a fixed number of randomly selected (equally likely) non-zero
            # entries
            each_result.points = NumpyRandomArrayDiscreteUniformDistributionGenerator(
                self.frame_shape)(self.num_objects).astype(float)

            # Creates a point generator that selects from the non-zero points
            # generated for activation to create groups
            # as an index array (tuple of 1D numpy.ndarrays)
            selected_points = each_result.points.nonzero()
            # convert to a single numpy.ndarrays
            selected_points = numpy.array(selected_points)
            # simpler, lightweight way of doing zip(*selected_points)
            selected_points = selected_points.T
            selected_points = selected_points.tolist()
            point_groups_gen = MappingDiscreteGeometricDistributionGenerator(
                *selected_points
            )

            # Using a mean group size and the number of groups creates point
            # groups (these should in someway relate to the basis images)
            point_groups = point_groups_gen(
                1.0 / float(self.mean_group_size), self.num_groups)

            # Will store the essential frames that indicate which points will
            # be active in each frame
            each_result.centroid_activation_frames = []
            for each_point_group in point_groups:
                # Get an index array
                each_point_group_index_array = nanshe.util.iters.list_indices_to_index_array(
                    each_point_group
                )

                # Create an empty activation frame
                each_centroid_activation_frame = numpy.zeros(self.frame_shape)

                # Set the active points to be randomly distributed
                each_centroid_activation_frame_points_shape = each_centroid_activation_frame[each_point_group_index_array].shape

                # Set the active points to be randomly distributed
                each_centroid_activation_frame[each_point_group_index_array] = numpy.random.random(
                    each_centroid_activation_frame_points_shape
                )

                # Rescale the active points
                each_centroid_activation_frame[each_point_group_index_array] *= self.object_intensity_range

                # Translate the active points
                each_centroid_activation_frame[each_point_group_index_array] += self.object_min_intensity

                # add to the stack of centroid activations
                each_result.centroid_activation_frames.append(
                    each_centroid_activation_frame
                )

            # convert to numpy array
            each_result.centroid_activation_frames = numpy.array(
                each_result.centroid_activation_frames
            )

            # Holds the frames without noise
            each_result.noiseless_frames = []

            # Takes each centroid activation frame and creates objects that dim
            # over time
            for each_centroid_activation_frame in each_result.centroid_activation_frames:
                # Determines how much to spread each active point
                # (self.object_spread is like the average spread)
                sigma = 2 * self.object_spread * numpy.random.random()
                for each_frame_num in xrange(self.num_frames):
                    # Determines a linear rescaling of each image (where they
                    # slowly become dimmer)
                    rescale = float(
                        self.num_frames - each_frame_num
                    ) / float(self.num_frames)
                    # Convolves each frame to generate a frame with objects
                    # (uses the same spread for each simply dims over time)
                    each_matrix_convolved = scipy.ndimage.filters.gaussian_filter(
                        rescale * each_centroid_activation_frame, sigma
                    )
                    # Adds to the stack of frames
                    each_result.noiseless_frames.append(each_matrix_convolved)

            # Converts the form of the noiseless frames
            each_result.noiseless_frames = numpy.array(
                each_result.noiseless_frames
            )

            # Creates frames that contain some background noise from a normal
            # distribution
            each_result.frames = each_result.noiseless_frames.copy()
            each_result.frames += numpy.random.normal(
                scale=self.background_noise_intensity,
                size=each_result.frames.shape
            )

            # Append to our list of results
            results.append(each_result)


        return(results)
