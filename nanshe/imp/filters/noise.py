"""
The ``noise`` module provides support for handling noise in images.

===============================================================================
Overview
===============================================================================
Provides a way of estimating noise based on what falls out of some multiple of
the standard deviation and generate a mask that excludes the noise or the
non-noise.

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 01, 2014 14:23:45 EDT$"


import numpy


# Need in order to have logging information no matter what.
from nanshe.util import prof


# Get the logger
trace_logger = prof.getTraceLogger(__name__)


@prof.log_call(trace_logger)
def estimate_noise(input_array, significance_threshold=3.0):
    """
        Estimates the noise in the given array.

        Using the array finds what the standard deviation is of some values in
        the array, which are within the standard deviation of the whole array
        times the significance threshold.

        Args:
            input_array(numpy.ndarray):         the array to estimate noise of.
            significance_threshold(float):      the number of standard
                                                deviations (of the whole
                                                array), below which must be
                                                noise.

        Returns:
            noise(float):                       The standard deviation of the
                                                noise.


        Examples:
            >>> estimate_noise(numpy.eye(2))
            0.5

            >>> estimate_noise(numpy.eye(2), 3)
            0.5

            >>> round(estimate_noise(numpy.eye(3), 3), 3)
            0.471

            >>> numpy.random.seed(10)
            >>> round(estimate_noise(numpy.random.random((2000,2000)), 1), 3)
            0.167

            >>> numpy.random.seed(10)
            >>> round(estimate_noise(numpy.random.random((2000,2000)), 2), 3)
            0.289

            >>> numpy.random.seed(10)
            >>> round(estimate_noise(numpy.random.random((2000,2000)), 3), 3)
            0.289
    """

    mean = input_array.mean()
    stddev = input_array.std()

    # Find cells that are inside an acceptable range
    # (3 std devs from the mean by default)
    input_array_devs = numpy.abs(input_array - mean)
    insignificant_mask = input_array_devs < significance_threshold * stddev

    # Those cells have noise. Estimate the standard deviation on them.
    # That will be our noise unit size.
    noise = input_array[insignificant_mask].std()

    return(noise)


@prof.log_call(trace_logger)
def significant_mask(input_array, noise_threshold=6.0, noise_estimate=None):
    """
        Using estimate_noise, creates a mask that selects the non-noise and
        suppresses noise.

        Args:
            input_array(numpy.ndarray):         the array, which needs noise
                                                removed.

            noise_threshold(float):             the estimated noise times this
                                                value determines the max value
                                                to consider as noise (to zero).

            in_place(bool):                     whether to modify input_array
                                                directly or to return a copy
                                                instead.

        Returns:
            result(numpy.ndarray): a numpy array with noise zeroed.


        Examples:

            >>> significant_mask(numpy.eye(2), 6.0)
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> significant_mask(numpy.eye(2), 6.0, 0.5)
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> significant_mask(numpy.eye(2), 1.0, 0.5)
            array([[ True,  True],
                   [ True,  True]], dtype=bool)

            >>> significant_mask(numpy.eye(3), 2.0, 0.47140452079103173)
            array([[False, False, False],
                   [False, False, False],
                   [False, False, False]], dtype=bool)

            >>> significant_mask(numpy.eye(3), 1.0, 0.47140452079103173)
            array([[ True, False, False],
                   [False,  True, False],
                   [False, False,  True]], dtype=bool)
    """

    mean = input_array.mean()

    # Estimate noise with the default estimate_noise function
    # if a value is not provided.
    if noise_estimate is None:
        noise_estimate = estimate_noise(input_array)

    # Get all the noisy points in a mask and toss them.
    input_array_devs = numpy.abs(input_array - mean)
    significant_mask = input_array_devs >= noise_threshold * noise_estimate

    return(significant_mask)


@prof.log_call(trace_logger)
def noise_mask(input_array, noise_threshold=6.0, noise_estimate=None):
    """
        Using estimate_noise, creates a mask that selects the noise and
        suppresses non-noise.

        Args:
            input_array(numpy.ndarray):         the array to use for generating
                                                the noise mask.

            significance_threshold(float):      the number of standard
                                                deviations (of the whole
                                                array), below which must be
                                                noise.

            noise_threshold(float):             the estimated noise times this
                                                value determines the max value
                                                to consider as noise (to zero).

        Returns:
            result(numpy.ndarray):              a numpy array with noise
                                                zeroed.


        Examples:
            >>> noise_mask(numpy.eye(2), 6.0, 0.5)
            array([[ True,  True],
                   [ True,  True]], dtype=bool)

            >>> noise_mask(numpy.eye(2), 1.0, 0.5)
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> noise_mask(numpy.eye(3), 2.0, 0.47140452079103173)
            array([[ True,  True,  True],
                   [ True,  True,  True],
                   [ True,  True,  True]], dtype=bool)

            >>> noise_mask(numpy.eye(3), 1.0, 0.47140452079103173)
            array([[False,  True,  True],
                   [ True, False,  True],
                   [ True,  True, False]], dtype=bool)
    """

    # Get all the significant points in a mask.
    noisy_mask = significant_mask(
        input_array,
        noise_threshold=noise_threshold,
        noise_estimate=noise_estimate
    )

    # Invert the maske
    numpy.logical_not(noisy_mask, noisy_mask)

    return(noisy_mask)
