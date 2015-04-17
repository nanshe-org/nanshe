"""
The ``renorm`` module provides ways to standardize frames of an image stack.

===============================================================================
Overview
===============================================================================
The ``renorm`` (or **renorm**\ alization) module provides a few simple utility
methods for standardizing a set of images. The first is to recompute each frame
so that its mean is 0. The second is to normalize each frame using an
appropriate |L^p norm|_. Both have the ability of working in-place.

.. |L^p norm| replace:: L\ :sup:`p` norm
.. _`L^p norm`: http://en.wikipedia.org/wiki/Lp_space

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 30, 2014 17:14:50 EDT$"


# Generally useful and fast to import so done immediately.
import numpy

# Need in order to have logging information no matter what.
from nanshe.util import prof


# Get the logger
trace_logger = prof.getTraceLogger(__name__)


@prof.log_call(trace_logger)
def zeroed_mean_images(input_array, output_array=None):
    """
        Takes and finds the mean for each image. Where each image is
        new_numpy_array[i] with some index i.

        Args:
            new_numpy_array(numpy.ndarray):     array images with time as the
                                                first index

            output_array(numpy.ndarray):        provides a location to store
                                                the result (optional)

        Returns:
            result(numpy.ndarray):              The same array with each images
                                                mean removed. Where
                                                means[i] = mean(new_numpy_array[i])


        Examples:
            >>> zeroed_mean_images(numpy.array([[0.,0.],[0.,0.]]))
            array([[ 0.,  0.],
                   [ 0.,  0.]])

            >>> zeroed_mean_images(numpy.array([[6.,0.],[0.,0.]]))
            array([[ 3., -3.],
                   [ 0.,  0.]])

            >>> zeroed_mean_images(numpy.array([[0.,0.],[0.,4.]]))
            array([[ 0.,  0.],
                   [-2.,  2.]])

            >>> zeroed_mean_images(numpy.array([[6.,0],[0.,4.]]))
            array([[ 3., -3.],
                   [-2.,  2.]])

            >>> zeroed_mean_images(numpy.array([[1.,2.],[3.,4.]]))
            array([[-0.5,  0.5],
                   [-0.5,  0.5]])

            >>> zeroed_mean_images(
            ...     numpy.array([[1,2],[3,4]])
            ... ) #doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            AssertionError

            >>> a = numpy.array([[1.,2.],[3.,4.]])
            >>> zeroed_mean_images(a, output_array=a)
            array([[-0.5,  0.5],
                   [-0.5,  0.5]])
            >>> a
            array([[-0.5,  0.5],
                   [-0.5,  0.5]])

            >>> a = numpy.array([[1.,2.],[3.,4.]]); b = numpy.zeros_like(a)
            >>> zeroed_mean_images(a, output_array=b)
            array([[-0.5,  0.5],
                   [-0.5,  0.5]])
            >>> b
            array([[-0.5,  0.5],
                   [-0.5,  0.5]])

            >>> zeroed_mean_images(
            ...     numpy.array([[1,2],[3,4]]).astype(numpy.float32)
            ... )
            array([[-0.5,  0.5],
                   [-0.5,  0.5]], dtype=float32)

            >>> a = numpy.array([[1.,2.],[3.,4.]])
            >>> numpy.all(a != zeroed_mean_images(a))
            True

            >>> a = numpy.array([[1.,2.],[3.,4.]])
            >>> numpy.all(a == zeroed_mean_images(a, output_array=a))
            True
    """

    assert issubclass(input_array.dtype.type, numpy.floating)

    if output_array is None:
        output_array = input_array.copy()
    elif id(input_array) != id(output_array):
        assert issubclass(output_array.dtype.type, numpy.floating)

        assert (input_array.shape == output_array.shape)

        output_array[:] = input_array

    # start with means having the same contents as the given images
    means = output_array

    # take the mean while we haven't gotten one mean for each image.
    while means.ndim > 1:
        means = means.mean(axis=1)

    # reshape means until it has the right number of dimensions to broadcast.
    means = means.reshape(means.shape + (output_array.ndim - means.ndim)*(1,))

    # broadcast and subtract the means so that the mean of all values is zero
    output_array[:] -= means

    return(output_array)


@prof.log_call(trace_logger)
def renormalized_images(input_array, ord=2, output_array=None):
    """
        Takes and divide each image by its norm. Where each image is
        new_numpy_array[i] with some index i.

        Args:
            new_numpy_array(numpy.ndarray):     array images with time as the
                                                first index

            ord(int):                           Which norm to use. (L_2 or
                                                Euclidean is default)

            output_array(numpy.ndarray):        provides a location to store
                                                the result (optional)

        Returns:
            result(numpy.ndarray):              The same array with each images
                                                normalized.


        Examples:
            >>> renormalized_images(numpy.array([[0.,1.],[1.,0.]]))
            array([[ 0.,  1.],
                   [ 1.,  0.]])

            >>> renormalized_images(
            ...     numpy.array([[0,1],[1,0]])
            ... ) #doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            AssertionError

            >>> renormalized_images(
            ...     numpy.array([[0,1],[1,0]], dtype=numpy.float32)
            ... )
            array([[ 0.,  1.],
                   [ 1.,  0.]], dtype=float32)

            >>> renormalized_images(numpy.array([[0.,2.],[1.,0.]]))
            array([[ 0.,  1.],
                   [ 1.,  0.]])

            >>> renormalized_images(numpy.array([[2.,2.],[1.,0.]]))
            array([[ 0.70710678,  0.70710678],
                   [ 1.        ,  0.        ]])

            >>> renormalized_images(numpy.array([[1.,2.],[3.,4.]]))
            array([[ 0.4472136 ,  0.89442719],
                   [ 0.6       ,  0.8       ]])

            >>> renormalized_images(numpy.array([[1.,2.],[3.,4.]]), ord=1)
            array([[ 0.33333333,  0.66666667],
                   [ 0.42857143,  0.57142857]])

            >>> a = numpy.array([[1.,2.],[3.,4.]])
            >>> numpy.all(a != renormalized_images(a))
            True

            >>> a = numpy.array([[1.,2.],[3.,4.]])
            >>> numpy.all(a == renormalized_images(a, output_array=a))
            True

            >>> a = numpy.array([[1.,2.],[3.,4.]]); b = numpy.zeros_like(a)
            >>> numpy.all(b == renormalized_images(a, output_array=b))
            True

            >>> renormalized_images(numpy.zeros((2,3,)))
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  0.]])
    """

    assert issubclass(input_array.dtype.type, numpy.floating)

    if output_array is None:
        output_array = input_array.copy()
    elif id(input_array) != id(output_array):
        assert issubclass(output_array.dtype.type, numpy.floating)

        assert (input_array.shape == output_array.shape)

        output_array[:] = input_array

    # Unfortunately, our version of numpy's function numpy.linalg.norm
    # does not support the axis keyword. So, we must use a for loop.
    # Take each image at each time and turn the image into a vector.
    # Then, find the norm and divide each image by this norm.
    for i in xrange(output_array.shape[0]):
        output_array_i = output_array[i]
        output_array_i_norm = numpy.linalg.norm(
            output_array_i.ravel(), ord=ord
        )

        if output_array_i_norm != 0:
            output_array_i /= output_array_i_norm

    return(output_array)
