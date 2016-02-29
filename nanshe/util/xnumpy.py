"""
The module ``xnumpy`` provides useful functions in combination with ``numpy``.

===============================================================================
Overview
===============================================================================
The module ``xnumpy`` provides some addition useful functions that are useful
in conjunction with  |numpy|_. The functions provided vary from handling view,
adding calculations, combining arrays in interesting ways, handling masks, etc.

.. |numpy| replace:: ``numpy``
.. _numpy: http://www.numpy.org/

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 20, 2014 09:46:45 EDT$"


import collections
import functools
import itertools
import operator
import warnings

import numpy
import scipy

import scipy.misc
import scipy.ndimage
import scipy.ndimage.morphology
import scipy.spatial
import scipy.stats
import scipy.stats.mstats

import bottleneck

import mahotas

import vigra


from nanshe.util import iters


# Need in order to have logging information no matter what.
from nanshe.util import prof


# Get the logger
trace_logger = prof.getTraceLogger(__name__)


@prof.log_call(trace_logger)
def info(a_type):
    """
        Takes a ``numpy.dtype`` or any type that can be converted to a
        ``numpy.dtype`` and returns its info.

        Args:
            a_type(type):                  the type to find info for.

        Returns:
            (np.core.getlimits.info):      info about the type.

        Examples:
            >>> info(float)
            finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)

            >>> info(numpy.float64)
            finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)

            >>> info(numpy.float32)
            finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

            >>> info(complex)
            finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)

            >>> info(int)
            iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)
    """

    a_type = numpy.dtype(a_type).type

    if issubclass(a_type, numpy.integer):
        return(numpy.iinfo(a_type))
    else:
        return(numpy.finfo(a_type))


@prof.log_call(trace_logger)
def to_ctype(a_type):
    """
        Takes a numpy.dtype or any type that can be converted to a numpy.dtype
        and returns its equivalent ctype.

        Args:
            a_type(type):      the type to find an equivalent ctype to.

        Returns:
            (ctype):           the ctype equivalent to the dtype provided.

        Examples:
            >>> to_ctype(float)
            <class 'ctypes.c_double'>

            >>> to_ctype(numpy.float64)
            <class 'ctypes.c_double'>

            >>> to_ctype(numpy.float32)
            <class 'ctypes.c_float'>

            >>> to_ctype(numpy.dtype(numpy.float32))
            <class 'ctypes.c_float'>

            >>> to_ctype(int)
            <class 'ctypes.c_long'>
    """

    return(type(numpy.ctypeslib.as_ctypes(numpy.array(0, dtype=a_type))))



@prof.log_call(trace_logger)
def renumber_label_image(new_array):
    """
        Takes a label image with non-consecutive numbering and renumbers it to
        be consecutive. Returns the relabeled image, a mapping from the old
        labels (by index) to the new ones, and a mapping from the new labels
        back to the old labels.

        Args:
            new_array(numpy.ndarray):                               the label
                                                                    image.

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray):          the
                                                                    relabeled
                                                                    label
                                                                    image,
                                                                    the
                                                                    forward
                                                                    label
                                                                    mapping and
                                                                    the reverse
                                                                    label
                                                                    mapping

        Examples:
            >>> renumber_label_image(numpy.array([1, 2, 3]))
            (array([1, 2, 3]), array([0, 1, 2, 3]), array([0, 1, 2, 3]))

            >>> renumber_label_image(numpy.array([1, 2, 4]))
            (array([1, 2, 3]), array([0, 1, 2, 0, 3]), array([0, 1, 2, 4]))

            >>> renumber_label_image(numpy.array([0, 1, 2, 3]))
            (array([0, 1, 2, 3]), array([0, 1, 2, 3]), array([0, 1, 2, 3]))

            >>> renumber_label_image(numpy.array([0, 1, 2, 4]))
            (array([0, 1, 2, 3]), array([0, 1, 2, 0, 3]), array([0, 1, 2, 4]))
    """

    # Get the set of reverse label mapping
    # (ensure the background is always included)
    reverse_label_mapping = numpy.unique(
        numpy.array([0] + numpy.unique(new_array).tolist())
    )

    # Get the set of old labels excluding background
    old_labels = reverse_label_mapping[reverse_label_mapping != 0]

    # Get the set of new labels in order
    new_labels = numpy.arange(1, len(old_labels) + 1)

    # Get the forward label mapping (ensure the background is included)
    forward_label_mapping = numpy.zeros(
        (reverse_label_mapping.max() + 1,), dtype=new_array.dtype
    )
    forward_label_mapping[old_labels] = new_labels

    # Get masks for each old label
    new_array_label_masks = all_permutations_equal(old_labels, new_array)

    # Create tiled where each label is expanded to the size of the new_array
    new_labels_tiled_view = expand_view(new_labels, new_array.shape)

    # Take every mask and make sure it has the appropriate sequential label
    # Then combine each of these parts of the label image together into a new
    # sequential label image
    new_array_relabeled = (
        new_array_label_masks * new_labels_tiled_view
    ).sum(axis=0)

    return((new_array_relabeled, forward_label_mapping, reverse_label_mapping))


@prof.log_call(trace_logger)
def index_axis_at_pos(new_array, axis, pos):
    """
        Indexes an arbitrary axis to the given position, which may be an index,
        a slice, or any other NumPy allowed indexing type. This will return a
        view.

        Args:
            new_array(numpy.ndarray):            array to add the singleton
                                                 axis to.

            axis(int):                           position for the axis to be in
                                                 the final array.
            pos(int or slice):                   how to index at the given
                                                 axis.

        Returns:
            (numpy.ndarray):                     a numpy array view of the
                                                 original array.

        Examples:
            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> index_axis_at_pos(a, 0, 0).shape
            (2, 3, 4)

            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> index_axis_at_pos(a, 1, 0).shape
            (1, 3, 4)

            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> index_axis_at_pos(a, 2, 0).shape
            (1, 2, 4)

            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> index_axis_at_pos(a, 3, 0).shape
            (1, 2, 3)

            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> (index_axis_at_pos(a, 3, 0) == a[:,:,:,0]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> (index_axis_at_pos(a, -1, 0) == a[:,:,:,0]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> (index_axis_at_pos(a, -1, 2) == a[:,:,:,2]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> (index_axis_at_pos(a, 1, 1) == a[:,1,:,:]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> (index_axis_at_pos(a, 2, slice(None,None,2)) == a[:,:,::2,:]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4))
            >>> index_axis_at_pos(a, 2, 2)[0, 1, 3] = 19; a[0, 1, 2, 3] == 19
            True
    """

    # Rescale axis inside the bounds
    axis %= new_array.ndim

    # Swaps the first with the desired axis (returns a view)
    new_array_swapped = new_array.swapaxes(0, axis)
    # Index to pos at the given axis
    new_subarray = new_array_swapped[pos]

    # Check to see if the chosen axis still exists (if pos were a slice)
    if new_subarray.ndim == new_array.ndim:
        # Transpose our selection to that ordering.
        new_subarray = new_subarray.swapaxes(0, axis)
    else:
        new_subarray = new_subarray[None]
        new_subarray = new_subarray.swapaxes(0, axis)
        new_subarray = numpy.squeeze(new_subarray, axis)

    return(new_subarray)


@prof.log_call(trace_logger)
def add_singleton_axis_pos(a_array, axis=0):
    """
        Adds a singleton axis to the given position.

        Allows negative values for axis. Also, automatically bounds axis in an
        acceptable regime if it is not already.

        Args:
            a_array(numpy.ndarray):            array to add the singleton axis
                                               to.
            axis(int):                         position for the axis to be in
                                               the final array (defaults to
                                               zero).

        Returns:
            (numpy.ndarray):                   a numpy array with the singleton
                                               axis added (should be a view).

        Examples:
            >>> add_singleton_axis_pos(numpy.ones((7,9,6))).shape
            (1, 7, 9, 6)

            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), 0).shape
            (1, 7, 9, 6)

            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), axis = 0).shape
            (1, 7, 9, 6)

            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), 1).shape
            (7, 1, 9, 6)

            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), 2).shape
            (7, 9, 1, 6)

            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), 3).shape
            (7, 9, 6, 1)

            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), -1).shape
            (7, 9, 6, 1)

            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), -2).shape
            (7, 9, 1, 6)
    """

    # Clean up axis to be within the allowable range.
    axis %= (a_array.ndim + 1)

    # Constructing the current ordering of axis and the singleton dime
    new_array_shape = list(iters.irange(1, a_array.ndim + 1))
    new_array_shape.insert(axis, 0)
    new_array_shape = tuple(new_array_shape)

    # Adds singleton dimension at front.
    # Then changes the order so it is elsewhere.
    new_array = a_array[None]
    new_array = new_array.transpose(new_array_shape)

    return(new_array)


@prof.log_call(trace_logger)
def add_singleton_axis_beginning(new_array):
    """
        Adds a singleton axis to the beginning of the array.

        Args:
            new_array(numpy.ndarray):            array to add the singleton
                                                 axis to.

        Returns:
            (numpy.ndarray):                     a numpy array with the
                                                 singleton axis added at the
                                                 end (should be view)

        Examples:
            >>> add_singleton_axis_beginning(numpy.ones((7,9,6))).shape
            (1, 7, 9, 6)

            >>> add_singleton_axis_beginning(numpy.eye(3)).shape
            (1, 3, 3)
    """

    # return( new_array[None] )
    return(add_singleton_axis_pos(new_array, 0))


@prof.log_call(trace_logger)
def add_singleton_axis_end(new_array):
    """
        Adds a singleton axis to the end of the array.

        Args:
            new_array(numpy.ndarray):            array to add the singleton
                                                 axis to.

        Returns:
            (numpy.ndarray):                     a numpy array with the
                                                 singleton axis added at the
                                                 end (should be view)

        Examples:
            >>> add_singleton_axis_end(numpy.ones((7,9,6))).shape
            (7, 9, 6, 1)

            >>> add_singleton_axis_end(numpy.eye(3)).shape
            (3, 3, 1)
    """

    # return( numpy.rollaxis(new_array[None], 0, new_array.ndim + 1) )
    return(add_singleton_axis_pos(new_array, new_array.ndim))


@prof.log_call(trace_logger)
def squish(new_array, axis=None, keepdims=False):
    """
        Moves the given axes to the last dimensions of the array and then
        squishes them into one dimension.

        Note:
            Returns a view if possible. However, if the axes provided are not
            consecutive integers when placed in the range [0, new_array.ndim),
            due to reshaping, the returned array will be a copy.

        Args:
            new_array(numpy.ndarray):           array to find the max (subject
                                                to the absolute value).

            axis(int or collection of ints):    desired axes to squish.
            keepdims(bool):                     ensure the number of dimensions
                                                is the same plus one by
                                                inserting singleton dimensions
                                                at all the axes squished.

        Returns:
            (numpy.ndarray):                    an array with one dimension at
                                                the end containing all the
                                                given axes.

        Examples:
            >>> a = numpy.arange(24).reshape(2,3,4).copy(); a
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> a.base is None
            True

            >>> b = squish(a); b
            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23])
            >>> b.base is a
            True

            >>> b = squish(a, axis=0); b
            array([[[ 0, 12],
                    [ 1, 13],
                    [ 2, 14],
                    [ 3, 15]],
            <BLANKLINE>
                   [[ 4, 16],
                    [ 5, 17],
                    [ 6, 18],
                    [ 7, 19]],
            <BLANKLINE>
                   [[ 8, 20],
                    [ 9, 21],
                    [10, 22],
                    [11, 23]]])
            >>> b.base is a
            True

            >>> b = squish(a, axis=2); b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> b.base is a
            True

            >>> b = squish(a, axis=-1); b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> b.base is a
            True

            >>> b = squish(a, axis=(0,1)); b
            array([[ 0,  4,  8, 12, 16, 20],
                   [ 1,  5,  9, 13, 17, 21],
                   [ 2,  6, 10, 14, 18, 22],
                   [ 3,  7, 11, 15, 19, 23]])
            >>> b.base is a
            True

            >>> b = squish(a, axis=(1, 0)); b
            array([[ 0, 12,  4, 16,  8, 20],
                   [ 1, 13,  5, 17,  9, 21],
                   [ 2, 14,  6, 18, 10, 22],
                   [ 3, 15,  7, 19, 11, 23]])
            >>> b.base is a
            False

            >>> b = squish(a, axis=(0, 2)); b
            array([[ 0,  1,  2,  3, 12, 13, 14, 15],
                   [ 4,  5,  6,  7, 16, 17, 18, 19],
                   [ 8,  9, 10, 11, 20, 21, 22, 23]])
            >>> b.base is a
            False

            >>> b = squish(a, axis=(0, 2), keepdims=True); b
            array([[[[ 0,  1,  2,  3, 12, 13, 14, 15]],
            <BLANKLINE>
                    [[ 4,  5,  6,  7, 16, 17, 18, 19]],
            <BLANKLINE>
                    [[ 8,  9, 10, 11, 20, 21, 22, 23]]]])

            >>> b = squish(a, axis=(1, 0), keepdims=True); b
            array([[[[ 0, 12,  4, 16,  8, 20],
                     [ 1, 13,  5, 17,  9, 21],
                     [ 2, 14,  6, 18, 10, 22],
                     [ 3, 15,  7, 19, 11, 23]]]])
    """

    # Convert the axes into a standard format that we can work with.
    axes = axis
    if axes is None:
        axes = list(iters.irange(new_array.ndim))
    else:
        # If axes is some kind of iterable, convert it to a list.
        # If not assume, it is a single value.
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]

        # Correct axes to be within the range [0, new_array.ndim).
        for i in iters.irange(len(axes)):
            axes[i] %= new_array.ndim

    axes = tuple(axes)

    # Put all axes not part of the group in front and
    # stuff the rest at the back.
    axis_order = tuple(iters.xrange_with_skip(new_array.ndim, to_skip=axes)) + axes
    result = new_array.transpose(axis_order)

    # Squash the axes at the end into one dimension.
    # If the axes aren't consecutive, this will force a copy.
    if len(axes) > 1:
        result_shape = result.shape[:result.ndim-len(axes)] + (-1,)
        result = result.reshape(result_shape)

    # Add back singleton dimensions as needed.
    if keepdims:
        # Need to sort or they won't be at the right place.
        sorted_axes = sorted(axes)

        # Add in singleton dimensions
        for i in sorted_axes:
            result = add_singleton_axis_pos(result, axis=i)

    return(result)


@prof.log_call(trace_logger)
def unsquish(new_array, shape, axis=None):
    """
        Inverts the squish operation given the shape and the axis/axes to
        extract from the last dimension.

        Args:
            new_array(numpy.ndarray):           array to find the max (subject
                                                to the absolute value).

            shape(collection of ints):          should be the shape of the
                                                result array (or the array
                                                before squishing).

            axis(int or collection of ints):    desired axes to remove from the
                                                last axis.

        Returns:
            (numpy.ndarray):                    an array with the shape
                                                provided and the axes removed
                                                from the end.

        Examples:
            >>> a = numpy.arange(24).reshape(2,3,4).copy(); a
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> a.base is None
            True

            >>> a.reshape(-1)
            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23])

            >>> b = unsquish(a.reshape(-1), (2,3,4)); b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> b.base is a
            True

            >>> a.transpose(1,2,0)
            array([[[ 0, 12],
                    [ 1, 13],
                    [ 2, 14],
                    [ 3, 15]],
            <BLANKLINE>
                   [[ 4, 16],
                    [ 5, 17],
                    [ 6, 18],
                    [ 7, 19]],
            <BLANKLINE>
                   [[ 8, 20],
                    [ 9, 21],
                    [10, 22],
                    [11, 23]]])

            >>> b = unsquish(a.transpose(1,2,0), (2,3,4), axis=0); b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> b.base is a
            True

            >>> b = unsquish(a, (2,3,4), axis=2); b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> b.base is a
            True

            >>> b = unsquish(a, (2,3,4), axis=-1); b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> b.base is a
            True

            >>> a.transpose(2,0,1).reshape(a.shape[2], -1)
            array([[ 0,  4,  8, 12, 16, 20],
                   [ 1,  5,  9, 13, 17, 21],
                   [ 2,  6, 10, 14, 18, 22],
                   [ 3,  7, 11, 15, 19, 23]])

            >>> b = unsquish(
            ...     a.transpose(2,0,1).reshape(a.shape[2], -1),
            ...     (2,3,4),
            ...     axis=(0,1)
            ... )
            >>> b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> b.base is a
            True

            >>> a.transpose(2, 1, 0).reshape(a.shape[2], -1)
            array([[ 0, 12,  4, 16,  8, 20],
                   [ 1, 13,  5, 17,  9, 21],
                   [ 2, 14,  6, 18, 10, 22],
                   [ 3, 15,  7, 19, 11, 23]])

            >>> b = unsquish(
            ...     a.transpose(2, 1, 0).reshape(a.shape[2], -1),
            ...     (2,3,4),
            ...     axis=(1,0)
            ... )
            >>> b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> b.base is a
            False

            >>> a.transpose(1, 0, 2).reshape(a.shape[1], -1)
            array([[ 0,  1,  2,  3, 12, 13, 14, 15],
                   [ 4,  5,  6,  7, 16, 17, 18, 19],
                   [ 8,  9, 10, 11, 20, 21, 22, 23]])

            >>> b = unsquish(
            ...     a.transpose(1, 0, 2).reshape(a.shape[1], -1),
            ...     (2,3,4),
            ...     axis=(0,2)
            ... )
            >>> b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
            >>> b.base is a
            False

            >>> a.transpose(1, 0, 2).reshape(a.shape[1], -1)[None]
            array([[[ 0,  1,  2,  3, 12, 13, 14, 15],
                    [ 4,  5,  6,  7, 16, 17, 18, 19],
                    [ 8,  9, 10, 11, 20, 21, 22, 23]]])

            >>> b = unsquish(
            ...     a.transpose(1, 0, 2).reshape(a.shape[1], -1)[None],
            ...     (2,3,4),
            ...     axis=(0,2)
            ... )
            >>> b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])

            >>> a.transpose(2, 1, 0).reshape(a.shape[2], -1)[None, None]
            array([[[[ 0, 12,  4, 16,  8, 20],
                     [ 1, 13,  5, 17,  9, 21],
                     [ 2, 14,  6, 18, 10, 22],
                     [ 3, 15,  7, 19, 11, 23]]]])

            >>> b = unsquish(
            ...     a.transpose(2, 1, 0).reshape(a.shape[2], -1)[None, None],
            ...     (2,3,4),
            ...     axis=(1,0)
            ... )
            >>> b
            array([[[ 0,  1,  2,  3],
                    [ 4,  5,  6,  7],
                    [ 8,  9, 10, 11]],
            <BLANKLINE>
                   [[12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23]]])
    """

    # Ensure shape is a tuple.
    shape = tuple(shape)

    # Convert the axes into a standard format that we can work with.
    axes = axis
    if axes is None:
        axes = list(iters.irange(0, len(shape)))
    else:
        # If axes is some kind of iterable, convert it to a list.
        # If not assume, it is a single value.
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]

        # Correct axes to be within the range [0, len(shape)).
        for i in iters.irange(len(axes)):
            axes[i] %= len(shape)

    axes = tuple(axes)

    result = new_array

    # Reshape the array to get the original shape (wrong axis order).
    # This will also eliminate singleton axes
    # that weren't part of the original shape (i.e. squish keepdim=True).
    shape_transposed = tuple()
    # Get how the axis order was changed
    old_axis_order_iter = itertools.chain(
        iters.xrange_with_skip(len(shape), to_skip=axes), axes
    )

    for i in old_axis_order_iter:
        shape_transposed += shape[i:i+1]

    result = result.reshape(shape_transposed)

    # Find out how the axes will need to be transposed
    # to return to the original order.
    if axis is not None:
        # Get how the axis order was changed
        old_axis_order_iter = itertools.chain(
            iters.xrange_with_skip(len(shape), to_skip=axes), axes
        )
        # Get the current axis order (i.e. in order)
        current_axis_order_iter = iters.irange(len(shape))

        # Find how the old order relates to the new one
        axis_order_map = dict(
            iters.izip(old_axis_order_iter, current_axis_order_iter)
        )

        # Export how the new order will be changed
        # (as the old axis order will be how to transform the axes).
        axis_order = tuple(axis_order_map.values())

        # Put all axes not part of the group in front and
        # stuff the rest at the back.
        result = result.transpose(axis_order)

    return(result)


@prof.log_call(trace_logger)
def add_singleton_op(op, new_array, axis):
    """
        Performs an operation on the given array on the specified axis, which
        otherwise would have eliminated the axis in question. This function
        will instead ensure that the given axis remains after the operation as
        a singleton.

        Note:
            The operation must be able to take only two arguments where the
            first is the array and the second is the axis to apply the
            operation along.

        Args:
            op(callable):                 callable that takes a numpy.ndarray
                                          and an int in order.

            new_array(numpy.ndarray):     array to perform operation on and add
                                          singleton axis too.

            axis(int):                    the axis to apply the operation along
                                          and turn into a singleton.

        Returns:
            (numpy.ndarray):              the array with the operation
                                          performed.

        Examples:
            >>> add_singleton_op(numpy.max, numpy.ones((7,9,6)), 0).shape
            (1, 9, 6)

            >>> add_singleton_op(numpy.max, numpy.ones((7,9,6)), -1).shape
            (7, 9, 1)

            >>> add_singleton_op(numpy.min, numpy.ones((7,9,6)), 1).shape
            (7, 1, 6)

            >>> add_singleton_op(numpy.mean, numpy.ones((7,9,6)), 1).shape
            (7, 1, 6)
    """

    return(add_singleton_axis_pos(op(new_array, axis), axis))


@prof.log_call(trace_logger)
def roll(new_array, shift, out=None, to_mask=False):
    """
        Like numpy.roll, but generalizes to include a roll for each axis of
        new_array.

        Note:
            Right shift occurs with a positive and left occurs with a negative.

        Args:
            new_array(numpy.ndarray):     array to roll axes of.
            shift(container of ints):     some sort of container (list, tuple,
                                          array) of ints specifying how much to
                                          roll each axis.

            out(numpy.ndarray):           array to store the results in.

            to_mask(bool):                Makes the result a masked array with
                                          the portion that rolled off masked.

        Returns:
            out(numpy.ndarray):           result of the roll.

        Examples:
            >>> roll(numpy.arange(20), numpy.array([0]))
            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19])

            >>> roll(numpy.arange(20), numpy.array([1]))
            array([19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                   16, 17, 18])

            >>> roll(numpy.arange(20), numpy.array([2]))
            array([18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                   15, 16, 17])

            >>> roll(numpy.arange(20), numpy.array([3]))
            array([17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                   14, 15, 16])

            >>> roll(numpy.arange(20), numpy.array([4]))
            array([16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                   13, 14, 15])

            >>> roll(numpy.arange(20), numpy.array([5]))
            array([15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                   12, 13, 14])

            >>> roll(numpy.arange(20), numpy.array([6]))
            array([14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                   11, 12, 13])

            >>> roll(numpy.arange(20), numpy.array([8]))
            array([12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,
                    9, 10, 11])

            >>> roll(numpy.arange(20), numpy.array([-1]))
            array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19,  0])

            >>> roll(
            ...     numpy.arange(10).reshape(2,5),
            ...     numpy.ones((2,), dtype=int)
            ... )
            array([[9, 5, 6, 7, 8],
                   [4, 0, 1, 2, 3]])

            >>> roll(numpy.arange(10).reshape(2,5), numpy.arange(2))
            array([[4, 0, 1, 2, 3],
                   [9, 5, 6, 7, 8]])

            >>> roll(numpy.arange(10).reshape(2,5), numpy.arange(1, 3))
            array([[8, 9, 5, 6, 7],
                   [3, 4, 0, 1, 2]])

            >>> roll(numpy.arange(10).reshape(2,5), numpy.array([2, 5]))
            array([[0, 1, 2, 3, 4],
                   [5, 6, 7, 8, 9]])

            >>> roll(numpy.arange(10).reshape(2,5), numpy.array([5, 3]))
            array([[7, 8, 9, 5, 6],
                   [2, 3, 4, 0, 1]])

            >>> roll(numpy.arange(10).reshape(2,5), numpy.array([-1, -1]))
            array([[6, 7, 8, 9, 5],
                   [1, 2, 3, 4, 0]])

            >>> roll(
            ...     numpy.arange(10).reshape(2,5),
            ...     numpy.array([1, -1]),
            ...     to_mask=True
            ... )
            masked_array(data =
             [[-- -- -- -- --]
             [1 2 3 4 --]],
                         mask =
             [[ True  True  True  True  True]
             [False False False False  True]],
                   fill_value = 999999)
            <BLANKLINE>

            >>> roll(
            ...     numpy.arange(10).reshape(2,5),
            ...     numpy.array([0, -1]),
            ...     to_mask=True
            ... )
            masked_array(data =
             [[1 2 3 4 --]
             [6 7 8 9 --]],
                         mask =
             [[False False False False  True]
             [False False False False  True]],
                   fill_value = 999999)
            <BLANKLINE>

            >>> a = numpy.ma.arange(10).reshape(2,5).copy()
            >>> roll(a, numpy.array([0, -1]), to_mask=True, out=a)
            masked_array(data =
             [[1 2 3 4 --]
             [6 7 8 9 --]],
                         mask =
             [[False False False False  True]
             [False False False False  True]],
                   fill_value = 999999)
            <BLANKLINE>
            >>> a
            masked_array(data =
             [[1 2 3 4 --]
             [6 7 8 9 --]],
                         mask =
             [[False False False False  True]
             [False False False False  True]],
                   fill_value = 999999)
            <BLANKLINE>

            >>> a = numpy.ma.arange(10).reshape(2,5).copy(); b = a[:, 1:-1]
            >>> roll(b, numpy.array([0, -1]), to_mask=True, out=b)
            masked_array(data =
             [[2 3 --]
             [7 8 --]],
                         mask =
             [[False False  True]
             [False False  True]],
                   fill_value = 999999)
            <BLANKLINE>
            >>> b
            masked_array(data =
             [[2 3 --]
             [7 8 --]],
                         mask =
             [[False False  True]
             [False False  True]],
                   fill_value = 999999)
            <BLANKLINE>
            >>> a # this should work, but it doesn't. # doctest: +SKIP
            masked_array(data =
             [[0 2 3 -- 4]
             [5 7 8 -- 9]],
                         mask =
             [[False False False  True False]
             [False False False  True False]],
                   fill_value = 999999)
            <BLANKLINE>

            >>> a = numpy.ma.arange(10).reshape(2,5).copy(); b = a[:, 1:-1]
            >>> roll(b, numpy.array([0, -1]), to_mask=True, out=b)
            masked_array(data =
             [[2 3 --]
             [7 8 --]],
                         mask =
             [[False False  True]
             [False False  True]],
                   fill_value = 999999)
            <BLANKLINE>
            >>> b
            masked_array(data =
             [[2 3 --]
             [7 8 --]],
                         mask =
             [[False False  True]
             [False False  True]],
                   fill_value = 999999)
            <BLANKLINE>
            >>> a.mask = numpy.ma.getmaskarray(a); a[:, 1:-1] = b; a
            masked_array(data =
             [[0 2 3 -- 4]
             [5 7 8 -- 9]],
                         mask =
             [[False False False  True False]
             [False False False  True False]],
                   fill_value = 999999)
            <BLANKLINE>

            >>> a = numpy.arange(10).reshape(2,5)
            >>> b = a.copy()
            >>> roll(a, numpy.arange(1, 3), b)
            array([[8, 9, 5, 6, 7],
                   [3, 4, 0, 1, 2]])
            >>> a
            array([[0, 1, 2, 3, 4],
                   [5, 6, 7, 8, 9]])
            >>> b
            array([[8, 9, 5, 6, 7],
                   [3, 4, 0, 1, 2]])

            >>> a = numpy.arange(10).reshape(2,5); a
            array([[0, 1, 2, 3, 4],
                   [5, 6, 7, 8, 9]])
            >>> roll(a, numpy.arange(1, 3), a)
            array([[8, 9, 5, 6, 7],
                   [3, 4, 0, 1, 2]])
            >>> a
            array([[8, 9, 5, 6, 7],
                   [3, 4, 0, 1, 2]])
    """

    shift = numpy.array(shift)

    assert (len(shift) == new_array.ndim)
    assert issubclass(shift.dtype.type, numpy.integer)

    if out is None:
        out = new_array.copy()

        if to_mask:
            out = out.view(numpy.ma.MaskedArray)
            out.mask = numpy.ma.getmaskarray(out)
    elif id(out) != id(new_array):
        out[:] = new_array

        if to_mask:
            if not isinstance(out, numpy.ma.MaskedArray):
                warnings.warn(
                    "Provided an array for `out` that is not a MaskedArray " +
                    " when requesting to mask the result. A view of `out` " +
                    "will be used so all changes are propagated to `out`. " +
                    "However, the mask may not be available " +
                    "(i.e. if the array is a view). To get the mask, " +
                    "either provide a MaskedArray as input or simply use " +
                    "the returned result.",
                    RuntimeWarning
                )

                out = out.view(numpy.ma.MaskedArray)
                out.mask = numpy.ma.getmaskarray(out)
            elif out.mask is numpy.ma.nomask:
                warnings.warn(
                    "Provided an array for `out` that is a MaskedArray, " +
                    "but has a trivial mask (i.e. nomask). A nontrivial " +
                    "mask will generated for `out` so that masking of `out` " +
                    "will work properly. However, the mask will not be " +
                    "available through the array provided. To get the mask, " +
                    "either provide a MaskedArray with a nontrivial mask as " +
                    "input or simply use the returned result.",
                    RuntimeWarning
                )

                out.mask = numpy.ma.getmaskarray(out)
    else:
        if to_mask:
            if not isinstance(out, numpy.ma.MaskedArray):
                warnings.warn(
                    "Provided an array for `new_array`/`out` that is not a " +
                    "MaskedArray when requesting to mask the result. A view " +
                    "of `new_array`/`out` will be used so all changes are " +
                    "propagated to `new_array`/`out`. However, the mask may " +
                    "not be available (i.e. if the array is a view). To get " +
                    "the mask, either provide a MaskedArray as input or " +
                    "simply use the returned result.",
                    RuntimeWarning
                )

                out = out.view(numpy.ma.MaskedArray)
                out.mask = numpy.ma.getmaskarray(out)
            elif out.mask is numpy.ma.nomask:
                warnings.warn(
                    "Provided an array for `new_array`/`out` that is a " +
                    "MaskedArray, but has a trivial mask (i.e. nomask). " +
                    "A nontrivial mask will generated for `new_array`/`out` " +
                    "so that masking of `new_array`/`out` will work " +
                    "properly. However, the mask will not be available " +
                    "through the array provided. To get the mask, either " +
                    "provide a MaskedArray with a nontrivial mask as input " +
                    "or simply use the returned result.",
                    RuntimeWarning
                )

                out.mask = numpy.ma.getmaskarray(out)

    for i in iters.irange(len(shift)):
        if (shift[i] != 0):
            out[:] = numpy.roll(out, shift[i], i)

            # If fill is specified, fill the portion that rolled over.
            if to_mask:
                slice_start = shift[i] if shift[i] < 0 else None
                slice_end = shift[i] if shift[i] > 0 else None
                shift_slice = slice(slice_start, slice_end)

                index_axis_at_pos(out.mask, i, shift_slice)[:] = True

    return(out)


@prof.log_call(trace_logger)
def contains(new_array, to_contain):
    """
        Gets a mask array that is true every time something from to_contain
        appears in new_array.

        Args:
            new_array(numpy.ndarray):            array to check for matches.
            to_contain(array_like):              desired matches to find.

        Returns:
            (numpy.ndarray):                     a mask for new_array that
                                                 selects values from
                                                 ``to_contain``.

        Examples:
            >>> contains(numpy.zeros((2,2)), 0)
            array([[ True,  True],
                   [ True,  True]], dtype=bool)

            >>> contains(numpy.zeros((2,2)), 1)
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> contains(numpy.zeros((2,2)), [1])
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> contains(numpy.zeros((2,2)), numpy.array([1]))
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> contains(numpy.arange(4).reshape((2,2)), numpy.array([0]))
            array([[ True, False],
                   [False, False]], dtype=bool)

            >>> contains(numpy.arange(4).reshape((2,2)), numpy.array([1]))
            array([[False,  True],
                   [False, False]], dtype=bool)

            >>> contains(numpy.arange(4).reshape((2,2)), numpy.array([2]))
            array([[False, False],
                   [ True, False]], dtype=bool)

            >>> contains(numpy.arange(4).reshape((2,2)), numpy.array([3]))
            array([[False, False],
                   [False,  True]], dtype=bool)
    """
    return(numpy.in1d(new_array, to_contain).reshape(new_array.shape))


@prof.log_call(trace_logger)
def min_abs(new_array, axis=None, keepdims=False, return_indices=False):
    """
        Takes the min of the given array subject to the absolute value
        (magnitude for complex numbers).

        Args:
            new_array(numpy.ndarray):            array to find the min (subject
                                                 to the absolute value).

            axis(int):                           desired matches to find.

            keepdims(bool):                      ensure the number of
                                                 dimensions is the same by
                                                 inserting singleton dimensions
                                                 at all the axes squished
                                                 (excepting the last one).

            return_indices(bool):                whether to return the indices
                                                 of the mins in addition to the
                                                 mins.

        Returns:
            (tuple of numpy.ndarray):            an array or value that is the
                                                 smallest (subject to the
                                                 absolute value) or if
                                                 ``return_indices`` the indices
                                                 corresponding to the smallest
                                                 value(s), as well.

        Examples:
            >>> min_abs(numpy.arange(10))
            0

            >>> min_abs(numpy.arange(10).reshape(2,5))
            0

            >>> min_abs(numpy.arange(10).reshape(2,5), axis=0)
            array([0, 1, 2, 3, 4])

            >>> min_abs(numpy.arange(10).reshape(2,5), axis=1)
            array([0, 5])

            >>> min_abs(numpy.arange(10).reshape(2,5), axis=-1)
            array([0, 5])

            >>> min_abs(numpy.arange(10).reshape(2,5), axis=-1, keepdims=True)
            array([[0],
                   [5]])

            >>> min_abs(numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]))
            0j

            >>> min_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]),
            ...     axis=0
            ... )
            array([ 0.+0.j,  0.+1.j,  2.+1.j])

            >>> min_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]),
            ...     axis=1
            ... )
            array([ 1.+0.j,  0.+0.j])

            >>> min_abs(numpy.arange(24).reshape(2,3,4), axis=(1,2))
            array([ 0, 12])

            >>> min_abs(numpy.arange(24).reshape(2,3,4), axis=(0,2))
            array([0, 4, 8])

            >>> min_abs(
            ...     numpy.arange(24).reshape(2,3,4),
            ...     axis=(0,2),
            ...     keepdims=True
            ... )
            array([[[0],
                    [4],
                    [8]]])

            >>> min_abs(numpy.arange(24).reshape(2,3,4), axis=(2,0))
            array([0, 4, 8])

            >>> min_abs(
            ...     numpy.arange(24).reshape(2,3,4),
            ...     axis=(2,0),
            ...     return_indices=True
            ... )
            (array([0, 4, 8]), (array([0, 0, 0]), array([0, 1, 2]), array([0, 0, 0])))

            >>> min_abs(numpy.array([numpy.nan, -2, 3]))
            nan
    """

    # Squish array to ensure all axes to be operated on
    # are at the end in one axis.
    new_array_refolded = squish(new_array, axis=axis, keepdims=keepdims)

    # Add singleton dimensions at the end
    # where the axes to be operated on now is.
    result_shape = new_array_refolded.shape[:-1] + (1,)

    # Get indices for the result and strip off the singleton axis (last dim).
    result_indices = numpy.indices(result_shape)[..., 0]

    # Get the indices that correspond to argmin for the given axis.
    result_indices[-1] = numpy.argmin(numpy.abs(new_array_refolded), axis=-1)

    # Make into index array.
    result_indices = tuple(result_indices)

    # Slice out relevant results
    result = new_array_refolded[result_indices]

    if not return_indices:
        return(result)
    else:
        # Create a mask.
        # This is required to remap the indices to the old array.
        result_mask = numpy.zeros(new_array_refolded.shape, dtype=bool)
        result_mask[result_indices] = True
        result_mask = unsquish(result_mask, new_array.shape, axis)
        result_indices = result_mask.nonzero()

        return(result, result_indices)


@prof.log_call(trace_logger)
def nanmin_abs(new_array, axis=None, keepdims=False, return_indices=False):
    """
        Takes the min of the given array subject to the absolute value
        (magnitude for complex numbers).

        Args:
            new_array(numpy.ndarray):            array to find the min (subject
                                                 to the absolute value).

            axis(int):                           desired matches to find.
            keepdims(bool):                      ensure the number of
                                                 dimensions is the same by
                                                 inserting singleton dimensions
                                                 at all the axes squished
                                                 (excepting the last one).

            return_indices(bool):                whether to return the indices
                                                 of the mins in addition to the
                                                 mins.

        Returns:
            (tuple of numpy.ndarray):            an array or value that is the
                                                 smallest (subject to the
                                                 absolute value) or if
                                                 ``return_indices`` the indices
                                                 corresponding to the smallest
                                                 value(s), as well.

        Examples:
            >>> nanmin_abs(numpy.arange(10))
            0

            >>> nanmin_abs(numpy.arange(10).reshape(2,5))
            0

            >>> nanmin_abs(numpy.arange(10).reshape(2,5), axis=0)
            array([0, 1, 2, 3, 4])

            >>> nanmin_abs(numpy.arange(10).reshape(2,5), axis=1)
            array([0, 5])

            >>> nanmin_abs(numpy.arange(10).reshape(2,5), axis=-1)
            array([0, 5])

            >>> nanmin_abs(
            ...     numpy.arange(10).reshape(2,5),
            ...     axis=-1,
            ...     keepdims=True
            ... )
            array([[0],
                   [5]])

            >>> nanmin_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]),
            ... )
            0j

            >>> nanmin_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]),
            ...     axis=0
            ... )
            array([ 0.+0.j,  0.+1.j,  2.+1.j])

            >>> nanmin_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]),
            ...     axis=1
            ... )
            array([ 1.+0.j,  0.+0.j])

            >>> nanmin_abs(numpy.arange(24).reshape(2,3,4), axis=(1,2))
            array([ 0, 12])

            >>> nanmin_abs(numpy.arange(24).reshape(2,3,4), axis=(0,2))
            array([0, 4, 8])

            >>> nanmin_abs(
            ...     numpy.arange(24).reshape(2,3,4),
            ...     axis=(0,2),
            ...     keepdims=True
            ... )
            array([[[0],
                    [4],
                    [8]]])

            >>> nanmin_abs(
            ...     numpy.arange(24).reshape(2,3,4),
            ...     axis=(2,0)
            ... )
            array([0, 4, 8])

            >>> nanmin_abs(
            ...     numpy.arange(24).reshape(2,3,4),
            ...     axis=(2,0),
            ...     return_indices=True
            ... )
            (array([0, 4, 8]), (array([0, 0, 0]), array([0, 1, 2]), array([0, 0, 0])))

            >>> nanmin_abs(numpy.array([numpy.nan, -2, 3]))
            -2.0
    """

    # Squish array to ensure all axes to be operated on
    # are at the end in one axis.
    new_array_refolded = squish(new_array, axis=axis, keepdims=keepdims)

    # Add singleton dimensions at the end
    # where the axes to be operated on now is.
    result_shape = new_array_refolded.shape[:-1] + (1,)

    # Get indices for the result and strip off the singleton axis (last dim).
    result_indices = numpy.indices(result_shape)[..., 0]

    # Get the indices that correspond to argmin (ignoring nan)
    # for the given axis.
    result_indices[-1] = bottleneck.nanargmin(
        numpy.abs(new_array_refolded), axis=-1
    )

    # Make into index array.
    result_indices = tuple(result_indices)

    # Slice out relevant results
    result = new_array_refolded[result_indices]

    if not return_indices:
        return(result)
    else:
        # Create a mask.
        # This is required to remap the indices to the old array.
        result_mask = numpy.zeros(new_array_refolded.shape, dtype=bool)
        result_mask[result_indices] = True
        result_mask = unsquish(result_mask, new_array.shape, axis)
        result_indices = result_mask.nonzero()

        return(result, result_indices)


@prof.log_call(trace_logger)
def max_abs(new_array, axis=None, keepdims=False, return_indices=False):
    """
        Takes the max of the given array subject to the absolute value
        (magnitude for complex numbers).

        Args:
            new_array(numpy.ndarray):            array to find the max (subject
                                                 to the absolute value).

            axis(int):                           desired matches to find.

            keepdims(bool):                      ensure the number of
                                                 dimensions is the same by
                                                 inserting singleton
                                                 dimensions at all the axes
                                                 squished (excepting the last
                                                 one).

            return_indices(bool):                whether to return the indices
                                                 of the maxes in addition to
                                                 the maxes.

        Returns:
            (tuple of numpy.ndarray):            an array or value that is the
                                                 largest (subject to the
                                                 absolute value) or if
                                                 ``return_indices`` the indices
                                                 corresponding to the largest
                                                 value(s), as well.

        Examples:
            >>> max_abs(numpy.arange(10))
            9

            >>> max_abs(numpy.arange(10).reshape(2,5))
            9

            >>> max_abs(numpy.arange(10).reshape(2,5), axis=0)
            array([5, 6, 7, 8, 9])

            >>> max_abs(numpy.arange(10).reshape(2,5), axis=1)
            array([4, 9])

            >>> max_abs(numpy.arange(10).reshape(2,5), axis=-1)
            array([4, 9])

            >>> max_abs(numpy.arange(10).reshape(2,5), axis=-1, keepdims=True)
            array([[4],
                   [9]])

            >>> max_abs(numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]))
            (1+3j)

            >>> max_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]),
            ...     axis=0
            ... )
            array([ 1.+0.j,  1.+1.j,  1.+3.j])

            >>> max_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]),
            ...     axis=1
            ... )
            array([ 2.+1.j,  1.+3.j])

            >>> max_abs(numpy.arange(24).reshape(2,3,4), axis=(1,2))
            array([11, 23])

            >>> max_abs(numpy.arange(24).reshape(2,3,4), axis=(0,2))
            array([15, 19, 23])

            >>> max_abs(
            ...     numpy.arange(24).reshape(2,3,4),
            ...     axis=(0,2),
            ...     keepdims=True
            ... )
            array([[[15],
                    [19],
                    [23]]])

            >>> max_abs(numpy.arange(24).reshape(2,3,4), axis=(2,0))
            array([15, 19, 23])

            >>> max_abs(
            ...     numpy.arange(24).reshape(2,3,4),
            ...     axis=(2,0),
            ...     return_indices=True
            ... )
            (array([15, 19, 23]), (array([1, 1, 1]), array([0, 1, 2]), array([3, 3, 3])))

            >>> max_abs(numpy.array([numpy.nan, -2, 3]))
            nan
    """

    # Squish array to ensure all axes to be operated on
    # are at the end in one axis.
    new_array_refolded = squish(new_array, axis=axis, keepdims=keepdims)

    # Add singleton dimensions at the end
    # where the axes to be operated on now is.
    result_shape = new_array_refolded.shape[:-1] + (1,)

    # Get indices for the result and strip off the singleton axis (last dim).
    result_indices = numpy.indices(result_shape)[..., 0]

    # Get the indices that correspond to argmax for the given axis.
    result_indices[-1] = numpy.argmax(numpy.abs(new_array_refolded), axis=-1)

    # Make into index array.
    result_indices = tuple(result_indices)

    # Slice out relevant results
    result = new_array_refolded[result_indices]

    if not return_indices:
        return(result)
    else:
        # Create a mask.
        # This is required to remap the indices to the old array.
        result_mask = numpy.zeros(new_array_refolded.shape, dtype=bool)
        result_mask[result_indices] = True
        result_mask = unsquish(result_mask, new_array.shape, axis)
        result_indices = result_mask.nonzero()

        return(result, result_indices)


@prof.log_call(trace_logger)
def nanmax_abs(new_array, axis=None, keepdims=False, return_indices=False):
    """
        Takes the max of the given array subject to the absolute value
        (magnitude for complex numbers).

        Args:
            new_array(numpy.ndarray):            array to find the max (subject
                                                 to the absolute value).

            axis(int):                           desired matches to find.

            keepdims(bool):                      ensure the number of
                                                 dimensions is the same by
                                                 inserting singleton dimensions
                                                 at all the axes squished
                                                 (excepting the last one).

            return_indices(bool):                whether to return the indices
                                                 of the maxes in addition to
                                                 the maxes.

        Returns:
            (tuple of numpy.ndarray):            an array or value that is the
                                                 largest (subject to the
                                                 absolute value) or if
                                                 ``return_indices`` the indices
                                                 corresponding to the largest
                                                 value(s), as well.

        Examples:
            >>> nanmax_abs(numpy.arange(10))
            9

            >>> nanmax_abs(numpy.arange(10).reshape(2,5))
            9

            >>> nanmax_abs(numpy.arange(10).reshape(2,5), axis=0)
            array([5, 6, 7, 8, 9])

            >>> nanmax_abs(numpy.arange(10).reshape(2,5), axis=1)
            array([4, 9])

            >>> nanmax_abs(numpy.arange(10).reshape(2,5), axis=-1)
            array([4, 9])

            >>> nanmax_abs(
            ...     numpy.arange(10).reshape(2,5),
            ...     axis=-1,
            ...     keepdims=True
            ... )
            array([[4],
                   [9]])

            >>> nanmax_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]])
            ... )
            (1+3j)

            >>> nanmax_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]),
            ...     axis=0,
            ... )
            array([ 1.+0.j,  1.+1.j,  1.+3.j])

            >>> nanmax_abs(
            ...     numpy.array([[1+0j, 0+1j, 2+1j], [0+0j, 1+1j, 1+3j]]),
            ...     axis=1,
            ... )
            array([ 2.+1.j,  1.+3.j])

            >>> nanmax_abs(numpy.arange(24).reshape(2,3,4), axis=(1,2))
            array([11, 23])

            >>> nanmax_abs(numpy.arange(24).reshape(2,3,4), axis=(0,2))
            array([15, 19, 23])

            >>> nanmax_abs(
            ...     numpy.arange(24).reshape(2,3,4),
            ...     axis=(0,2),
            ...     keepdims=True
            ... )
            array([[[15],
                    [19],
                    [23]]])

            >>> nanmax_abs(numpy.arange(24).reshape(2,3,4), axis=(2,0))
            array([15, 19, 23])

            >>> nanmax_abs(
            ...     numpy.arange(24).reshape(2,3,4),
            ...     axis=(2,0),
            ...     return_indices=True
            ... )
            (array([15, 19, 23]), (array([1, 1, 1]), array([0, 1, 2]), array([3, 3, 3])))

            >>> nanmax_abs(numpy.array([numpy.nan, -2, 3]))
            3.0
    """

    # Squish array to ensure all axes to be operated on
    # are at the end in one axis.
    new_array_refolded = squish(new_array, axis=axis, keepdims=keepdims)

    # Add singleton dimensions at the end
    # where the axes to be operated on now is.
    result_shape = new_array_refolded.shape[:-1] + (1,)

    # Get indices for the result and strip off the singleton axis (last dim).
    result_indices = numpy.indices(result_shape)[..., 0]

    # Get the indices that correspond to argmax
    # (ignoring nan) for the given axis.
    result_indices[-1] = bottleneck.nanargmax(
        numpy.abs(new_array_refolded), axis=-1
    )

    # Make into index array.
    result_indices = tuple(result_indices)

    # Slice out relevant results
    result = new_array_refolded[result_indices]

    if not return_indices:
        return(result)
    else:
        # Create a mask.
        # This is required to remap the indices to the old array.
        result_mask = numpy.zeros(new_array_refolded.shape, dtype=bool)
        result_mask[result_indices] = True
        result_mask = unsquish(result_mask, new_array.shape, axis)
        result_indices = result_mask.nonzero()

        return(result, result_indices)


@prof.log_call(trace_logger)
def array_to_matrix(a):
    """
        Flattens an array so that the row is the only original shape kept.

        Args:
            a(numpy.ndarray):                The array to flatten, partially.

        Returns:
            (numpy.ndarray):                 The matrix version of the array
                                             after flattening.

        Examples:
            >>> array_to_matrix(numpy.eye(3))
            array([[ 1.,  0.,  0.],
                   [ 0.,  1.,  0.],
                   [ 0.,  0.,  1.]])

            >>> array_to_matrix(numpy.arange(24).reshape(4, 3, 2))
            array([[ 0,  1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10, 11],
                   [12, 13, 14, 15, 16, 17],
                   [18, 19, 20, 21, 22, 23]])

            >>> array_to_matrix(numpy.arange(24).reshape(4, 6))
            array([[ 0,  1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10, 11],
                   [12, 13, 14, 15, 16, 17],
                   [18, 19, 20, 21, 22, 23]])

            >>> array_to_matrix(numpy.zeros((0, 4, 3, 2)))
            array([], shape=(0, 24), dtype=float64)
    """

    return(a.reshape(a.shape[0], functools.reduce(operator.mul, a.shape[1:])))


@prof.log_call(trace_logger)
def index_array_to_bool_array(index_array, shape):
    """
        Creates a bool array mask that has each value from the index array as
        True. All other values are False. Requires a shape be specified to
        create the bool array.

        Args:
            index_array(numpy.ndarray of ints):     The index array with the
                                                    indices to use.

            shape(tuple of ints):                   The shape to give the bool
                                                    array.

        Returns:
            (numpy.ndarray):                        The bool array with
                                                    selected indices as True
                                                    and the rest are False.

        Examples:
            >>> index_array_to_bool_array((numpy.arange(5),), (5,))
            array([ True,  True,  True,  True,  True], dtype=bool)

            >>> index_array_to_bool_array((numpy.arange(1, 4),), (5,))
            array([False,  True,  True,  True, False], dtype=bool)

            >>> index_array_to_bool_array(
            ...     (numpy.arange(3), numpy.arange(3)), (3, 3)
            ... )
            array([[ True, False, False],
                   [False,  True, False],
                   [False, False,  True]], dtype=bool)
    """

    bool_array = numpy.zeros(shape, dtype=bool)
    bool_array[index_array] = True

    return(bool_array)


@prof.log_call(trace_logger)
def expand_view(new_array, reps_after=tuple(), reps_before=tuple()):
    """
        Behaves like NumPy tile except that it always returns a view and not a
        copy. Though, it differs in that additional dimensions are added for
        repetition as opposed to repeating in the same one. Also, it allows
        repetitions to be specified before or after unlike tile. Though, will
        behave identical to tile if the keyword is not specified.

        Uses strides to trick NumPy into providing a view.

        Args:
            new_array(numpy.ndarray):            array to tile.

            reps_after(tuple):                   repetitions dimension size to
                                                 add before (if int will turn
                                                 into tuple).

            reps_before(tuple):                  repetitions dimension size to
                                                 add after (if int will turn
                                                 into tuple).

        Returns:
            (numpy.ndarray):                     a view of a numpy array with
                                                 tiling in various dimension.

        Examples:
            >>> numpy.arange(6).reshape(2,3)
            array([[0, 1, 2],
                   [3, 4, 5]])

            >>> expand_view(numpy.arange(6).reshape(2,3))
            array([[0, 1, 2],
                   [3, 4, 5]])

            >>> a = numpy.arange(6).reshape(2,3); a is expand_view(a)
            False

            >>> expand_view(numpy.arange(6).reshape(2,3), 1)
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])

            >>> expand_view(numpy.arange(6).reshape(2,3), (1,))
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])

            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_after=1)
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])

            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_after=(1,))
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])

            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_before=1)
            array([[[0, 1, 2],
                    [3, 4, 5]]])

            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_before=(1,))
            array([[[0, 1, 2],
                    [3, 4, 5]]])

            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_before=(3,))
            array([[[0, 1, 2],
                    [3, 4, 5]],
            <BLANKLINE>
                   [[0, 1, 2],
                    [3, 4, 5]],
            <BLANKLINE>
                   [[0, 1, 2],
                    [3, 4, 5]]])

            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_after=(4,))
            array([[[0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2]],
            <BLANKLINE>
                   [[3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5]]])

            >>> expand_view(
            ...     numpy.arange(6).reshape((2,3)),
            ...     reps_before=(3,),
            ...     reps_after=(4,)
            ... )
            array([[[[0, 0, 0, 0],
                     [1, 1, 1, 1],
                     [2, 2, 2, 2]],
            <BLANKLINE>
                    [[3, 3, 3, 3],
                     [4, 4, 4, 4],
                     [5, 5, 5, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 0, 0, 0],
                     [1, 1, 1, 1],
                     [2, 2, 2, 2]],
            <BLANKLINE>
                    [[3, 3, 3, 3],
                     [4, 4, 4, 4],
                     [5, 5, 5, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 0, 0, 0],
                     [1, 1, 1, 1],
                     [2, 2, 2, 2]],
            <BLANKLINE>
                    [[3, 3, 3, 3],
                     [4, 4, 4, 4],
                     [5, 5, 5, 5]]]])

            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_after = (4,3))
            array([[[[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
            <BLANKLINE>
                    [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]],
            <BLANKLINE>
                    [[2, 2, 2],
                     [2, 2, 2],
                     [2, 2, 2],
                     [2, 2, 2]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[3, 3, 3],
                     [3, 3, 3],
                     [3, 3, 3],
                     [3, 3, 3]],
            <BLANKLINE>
                    [[4, 4, 4],
                     [4, 4, 4],
                     [4, 4, 4],
                     [4, 4, 4]],
            <BLANKLINE>
                    [[5, 5, 5],
                     [5, 5, 5],
                     [5, 5, 5],
                     [5, 5, 5]]]])

            >>> expand_view(
            ...     numpy.arange(6).reshape((2,3)),
            ...     reps_before=(4,3),
            ... )
            array([[[[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]]]])
    """

    if not isinstance(reps_after, tuple):
        reps_after = (reps_after,)

    if not isinstance(reps_before, tuple):
        reps_before = (reps_before,)

    return(numpy.lib.stride_tricks.as_strided(
        new_array,
        reps_before + new_array.shape + reps_after,
        (0,) * len(reps_before) + new_array.strides + (0,) * len(reps_after)
    ))


def expand_arange(start,
                  stop=None,
                  step=1,
                  dtype=numpy.int64,
                  reps_before=tuple(),
                  reps_after=tuple()):
    """
        Much like ``numpy.arange`` except that it applies expand_view
        afterwards to get a view of the same range in a larger hyperrectangle.

        This is very useful for situations where broadcasting is desired.

        Args:
            start(int):                          starting point (or stopping
                                                 point if only one is
                                                 specified).

            stop(int):                           stopping point (if the
                                                 starting point is specified)
                                                 (0 by default).

            step(int):                           size of steps to take between
                                                 value (1 by default).

            reps_after(tuple):                   repetitions dimension size to
                                                 add before (if int will turn
                                                 into tuple).

            reps_before(tuple):                  repetitions dimension size to
                                                 add after (if int will turn
                                                 into tuple).

        Returns:
            (numpy.ndarray):                     a view of a numpy arange with
                                                 tiling in various dimension.

        Examples:
            >>> expand_arange(3, reps_before=3)
            array([[0, 1, 2],
                   [0, 1, 2],
                   [0, 1, 2]])

            >>> expand_arange(3, reps_after=3)
            array([[0, 0, 0],
                   [1, 1, 1],
                   [2, 2, 2]])

            >>> expand_arange(4, reps_before=3)
            array([[0, 1, 2, 3],
                   [0, 1, 2, 3],
                   [0, 1, 2, 3]])

            >>> expand_arange(4, reps_after=3)
            array([[0, 0, 0],
                   [1, 1, 1],
                   [2, 2, 2],
                   [3, 3, 3]])

            >>> expand_arange(4, reps_before=3, reps_after=2)
            array([[[0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 3]],
            <BLANKLINE>
                   [[0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 3]],
            <BLANKLINE>
                   [[0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 3]]])
    """

    if (stop is None):
        stop = start
        start = 0

    an_arange = numpy.arange(start=start, stop=stop, step=step, dtype=dtype)

    an_arange = expand_view(
        an_arange, reps_before=reps_before, reps_after=reps_after
    )

    return(an_arange)


def expand_enumerate(new_array, axis=0, start=0, step=1):
    """
        Builds on expand_arange, which has the same shape as the original
        array. Specifies the increments to occur along the given axis, which by
        default is the zeroth axis.

        Provides mechanisms for changing the starting value and also the
        increment.

        Args:
            new_array(numpy.ndarray):            array to enumerate

            axis(int):                           axis to enumerate along (0 by
                                                 default).
            start(int):                          starting point (0 by default).

            step(int):                           size of steps to take between
                                                 value (1 by default).

        Returns:
            (numpy.ndarray):                     a view of a numpy arange with
                                                 tiling in various dimension.

        Examples:
            >>> expand_enumerate(numpy.ones((4,5)))
            array([[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3]], dtype=uint64)

            >>> expand_enumerate(numpy.ones((4,5)), axis=0)
            array([[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3]], dtype=uint64)

            >>> expand_enumerate(numpy.ones((4,5)), axis=0, start=1)
            array([[1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4]], dtype=uint64)

            >>> expand_enumerate(numpy.ones((4,5)), axis=0, start=1, step=2)
            array([[1, 1, 1, 1, 1],
                   [3, 3, 3, 3, 3],
                   [5, 5, 5, 5, 5],
                   [7, 7, 7, 7, 7]], dtype=uint64)

            >>> expand_enumerate(numpy.ones((4,5)), axis=1)
            array([[0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4]], dtype=uint64)

            >>> expand_enumerate(numpy.ones((4,5)), axis=1, start=1)
            array([[1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5]], dtype=uint64)

            >>> expand_enumerate(numpy.ones((4,5)), axis=1, start=1, step=2)
            array([[1, 3, 5, 7, 9],
                   [1, 3, 5, 7, 9],
                   [1, 3, 5, 7, 9],
                   [1, 3, 5, 7, 9]], dtype=uint64)
    """

    an_enumeration = expand_arange(
        start=start,
        stop=start + step * new_array.shape[axis],
        step=step,
        dtype=numpy.uint64,
        reps_before=new_array.shape[:axis],
        reps_after=new_array.shape[(axis+1):]
    )

    return(an_enumeration)


def enumerate_masks(new_masks, axis=0):
    """
        Takes a mask stack and replaces them by an enumerated stack. In other
        words, each mask is replaced by a consecutive integer (starts with 1
        and proceeds to the length of the given axis (0 by default)).

        Note:
            The masks could be recreated by finding the values not equal to
            zero.

        Args:
            new_masks(numpy.ndarray):            masks to enumerate
            axis(int):                           axis to enumerate along (0 by
                                                 default).

        Returns:
            (numpy.ndarray):                     an enumerated stack.

        Examples:
            >>> enumerate_masks(numpy.ones((3,3,3), dtype=bool))
            array([[[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]],
            <BLANKLINE>
                   [[2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2]],
            <BLANKLINE>
                   [[3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3]]], dtype=uint64)

            >>> enumerate_masks(
            ...     numpy.array([[[ True, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False, False]],
            ...
            ...                  [[False, False, False, False],
            ...                   [False,  True, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False, False]],
            ...
            ...                  [[False, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False,  True, False],
            ...                   [False, False, False, False]],
            ...
            ...                  [[False, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False,  True]]], dtype=bool)
            ... )
            array([[[1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]],
            <BLANKLINE>
                   [[0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]],
            <BLANKLINE>
                   [[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 3, 0],
                    [0, 0, 0, 0]],
            <BLANKLINE>
                   [[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 4]]], dtype=uint64)
    """

    new_enumerated_masks = new_masks * expand_enumerate(
        new_masks, axis=axis, start=1, step=1
    )

    return(new_enumerated_masks)


def enumerate_masks_max(new_masks, axis=0):
    """
        Takes a mask stack and replaces them by the max of an enumerated stack.
        In other words, each mask is replaced by a consecutive integer (starts
        with 1 and proceeds to the length of the given axis (0 by default)).
        Afterwards, the max is taken along the given axis. However, a singleton
        dimension is left on the original axis.

        Note:
            The masks could be recreated by finding the values not equal to
            zero.

        Args:
            new_masks(numpy.ndarray):            masks to enumerate

            axis(int):                           axis to enumerate along (0 by
                                                 default).

        Returns:
            (numpy.ndarray):                     an enumerated stack.

        Examples:
            >>> enumerate_masks_max(numpy.ones((3,3,3), dtype=bool))
            array([[[3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3]]], dtype=uint64)

            >>> enumerate_masks_max(
            ...     numpy.array([[[ True, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False, False]],
            ...
            ...                  [[False, False, False, False],
            ...                   [False,  True, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False, False]],
            ...
            ...                  [[False, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False,  True, False],
            ...                   [False, False, False, False]],
            ...
            ...                  [[False, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False, False],
            ...                   [False, False, False,  True]]], dtype=bool)
            ... )
            array([[[1, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 3, 0],
                    [0, 0, 0, 4]]], dtype=uint64)
    """

    axis %= new_masks.ndim

    new_enumerated_masks_max = numpy.zeros(
        new_masks.shape[:axis] + (1,) + new_masks.shape[axis+1:],
        dtype=numpy.uint64
    )

    for i in iters.irange(new_masks.shape[axis]):
        i = new_enumerated_masks_max.dtype.type(i)
        one = new_enumerated_masks_max.dtype.type(1)
        numpy.maximum(
            new_enumerated_masks_max,
            (i + one) * add_singleton_axis_pos(
                            index_axis_at_pos(new_masks, axis, i),
                            axis
                        ),
            out=new_enumerated_masks_max
        )

    return(new_enumerated_masks_max)


@prof.log_call(trace_logger)
def cartesian_product(arrays):
    """
        Takes the cartesian product between the elements in each array.

        Args:
            arrays(collections.Sequence of numpy.ndarrays):     A sequence of
                                                                1-D arrays or a
                                                                2-D array.

        Returns:
            (numpy.ndarray):                                    an array
                                                                containing the
                                                                result of the
                                                                cartesian
                                                                product of each
                                                                array.

        Examples:
            >>> cartesian_product([numpy.arange(2), numpy.arange(3)])
            array([[0, 0],
                   [0, 1],
                   [0, 2],
                   [1, 0],
                   [1, 1],
                   [1, 2]])

            >>> cartesian_product([
            ...     numpy.arange(2, dtype=float),
            ...     numpy.arange(3)
            ... ])
            array([[ 0.,  0.],
                   [ 0.,  1.],
                   [ 0.,  2.],
                   [ 1.,  0.],
                   [ 1.,  1.],
                   [ 1.,  2.]])

            >>> cartesian_product([
            ...     numpy.arange(2),
            ...     numpy.arange(3),
            ...     numpy.arange(4)
            ... ])
            array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 0, 2],
                   [0, 0, 3],
                   [0, 1, 0],
                   [0, 1, 1],
                   [0, 1, 2],
                   [0, 1, 3],
                   [0, 2, 0],
                   [0, 2, 1],
                   [0, 2, 2],
                   [0, 2, 3],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 0, 2],
                   [1, 0, 3],
                   [1, 1, 0],
                   [1, 1, 1],
                   [1, 1, 2],
                   [1, 1, 3],
                   [1, 2, 0],
                   [1, 2, 1],
                   [1, 2, 2],
                   [1, 2, 3]])

            >>> cartesian_product(numpy.diag((1, 2, 3)))
            array([[1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 3],
                   [1, 2, 0],
                   [1, 2, 0],
                   [1, 2, 3],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 3],
                   [0, 2, 0],
                   [0, 2, 0],
                   [0, 2, 3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 3],
                   [0, 2, 0],
                   [0, 2, 0],
                   [0, 2, 3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 3]])
    """

    for i in iters.irange(len(arrays)):
        assert (arrays[
                i].ndim == 1), "Must provide only 1D arrays to this function or a single 2D array."

    array_shapes = tuple(len(arrays[i]) for i in iters.irange(len(arrays)))

    result_shape = [0, 0]
    result_shape[0] = numpy.product(array_shapes)
    result_shape[1] = len(arrays)
    result_shape = tuple(result_shape)

    result_dtype = numpy.find_common_type(
        [arrays[i].dtype for i in iters.irange(result_shape[1])], []
    )

    result = numpy.empty(result_shape, dtype=result_dtype)
    for i in iters.irange(result.shape[1]):
        repeated_array_i = expand_view(
            arrays[i],
            reps_before=array_shapes[:i],
            reps_after=array_shapes[i+1:]
        )
        for j, repeated_array_i_j in enumerate(repeated_array_i.flat):
            result[j, i] = repeated_array_i_j

    return(result)


@prof.log_call(trace_logger)
def truncate_masked_frames(shifted_frames):
    """
        Takes frames that have been shifted and truncates out the portion,
        which is an intact rectangular shape.

        Args:
            shifted_frames(numpy.ma.masked_array):      Image stack to register
                                                        (time is the first
                                                        dimension uses C-order
                                                        tyx or tzyx).

        Returns:
            (numpy.ndarray):                            an array containing a
                                                        subsection of the stack
                                                        that has no mask.

        Examples:
            >>> a = numpy.arange(60).reshape(3,5,4)
            >>> a = numpy.ma.masked_array(
            ...     a, mask=numpy.zeros(a.shape, dtype=bool), shrink=False
            ... )
            >>> a[0, :1, :] = numpy.ma.masked; a[0, :, -1:] = numpy.ma.masked
            >>> a[1, :2, :] = numpy.ma.masked; a[1, :, :0] = numpy.ma.masked
            >>> a[2, :0, :] = numpy.ma.masked; a[2, :, :1] = numpy.ma.masked
            >>> a
            masked_array(data =
             [[[-- -- -- --]
              [4 5 6 --]
              [8 9 10 --]
              [12 13 14 --]
              [16 17 18 --]]
            <BLANKLINE>
             [[-- -- -- --]
              [-- -- -- --]
              [28 29 30 31]
              [32 33 34 35]
              [36 37 38 39]]
            <BLANKLINE>
             [[-- 41 42 43]
              [-- 45 46 47]
              [-- 49 50 51]
              [-- 53 54 55]
              [-- 57 58 59]]],
                         mask =
             [[[ True  True  True  True]
              [False False False  True]
              [False False False  True]
              [False False False  True]
              [False False False  True]]
            <BLANKLINE>
             [[ True  True  True  True]
              [ True  True  True  True]
              [False False False False]
              [False False False False]
              [False False False False]]
            <BLANKLINE>
             [[ True False False False]
              [ True False False False]
              [ True False False False]
              [ True False False False]
              [ True False False False]]],
                   fill_value = 999999)
            <BLANKLINE>

            >>> truncate_masked_frames(a)
            array([[[ 9, 10],
                    [13, 14],
                    [17, 18]],
            <BLANKLINE>
                   [[29, 30],
                    [33, 34],
                    [37, 38]],
            <BLANKLINE>
                   [[49, 50],
                    [53, 54],
                    [57, 58]]])
    """

    # Find the mask to slice out the relevant data from all frames
    shifted_frames_mask = ~numpy.ma.getmaskarray(shifted_frames).max(axis=0)

    # Find the shape
    shifted_frames_mask_shape = tuple(shifted_frames_mask.sum(
        axis=_i).max() for _i in iters.irange(shifted_frames_mask.ndim)
    )
    shifted_frames_mask_shape = (len(shifted_frames),) + shifted_frames_mask_shape

    # Assert that this is an acceptable mask
    #shifted_frames_mask_upper_offset = tuple(
    #    (shifted_frames_mask.sum(axis=_i) != 0).argmax() for _i in reversed(iters.irange(shifted_frames_mask.ndim))
    #)
    #shifted_frames_mask_lower_offset = tuple(
    #    numpy.array(shifted_frames_mask.shape) - \
    #    numpy.array(shifted_frames_mask_shape[1:]) - \
    #    numpy.array(shifted_frames_mask_upper_offset)
    #)
    #
    #shifted_frames_mask_reconstructed = numpy.pad(
    #    numpy.ones(shifted_frames_mask_shape[1:], dtype=bool),
    #    [(_d, _e) for _d, _e in iters.izip(shifted_frames_mask_upper_offset, shifted_frames_mask_lower_offset)],
    #    "constant"
    #)
    #assert(
    #    (shifted_frames_mask_reconstructed == shifted_frames_mask).all(),
    #    "The masked array provide has a mask that does not reduce to a square when max projected."
    #)

    # Slice out the relevant data from the frames
    truncated_shifted_frames = shifted_frames[:, shifted_frames_mask].reshape(
        shifted_frames_mask_shape
    )
    truncated_shifted_frames = truncated_shifted_frames.data

    return(truncated_shifted_frames)


@prof.log_call(trace_logger)
def all_permutations_operation(new_op, new_array_1, new_array_2):
    """
        Takes two arrays and constructs a new array that contains the result
        of new_op on every permutation of elements in each array (like
        broadcasting).

        Suppose that new_result contained the result, then one would find that
        the result of the following operation on the specific indices.

        new_op( new_array_1[ i_1_1, i_1_2, ... ],
        new_array_2[ i_2_1, i_2_2, ... ] )

        would be found in new_result as shown

        new_result[ i_1_1, i_1_2, ..., i_2_1, i_2_2, ... ]


        Args:
            new_array(numpy.ndarray):            array to add the singleton
                                                 axis to.

        Returns:
            (numpy.ndarray):                     a numpy array with the
                                                 singleton axis added at the
                                                 end.

        Examples:
            >>> all_permutations_operation(
            ...     operator.add, numpy.ones((1,3)), numpy.eye(2)
            ... ).shape
            (1, 3, 2, 2)

            >>> all_permutations_operation(
            ...     operator.add, numpy.ones((2,2)), numpy.eye(2)
            ... ).shape
            (2, 2, 2, 2)

            >>> all_permutations_operation(
            ...     operator.add, numpy.ones((2,2)), numpy.eye(2)
            ... )
            array([[[[ 2.,  1.],
                     [ 1.,  2.]],
            <BLANKLINE>
                    [[ 2.,  1.],
                     [ 1.,  2.]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[ 2.,  1.],
                     [ 1.,  2.]],
            <BLANKLINE>
                    [[ 2.,  1.],
                     [ 1.,  2.]]]])

            >>> all_permutations_operation(
            ...     operator.sub, numpy.ones((2,2)), numpy.eye(2)
            ... )
            array([[[[ 0.,  1.],
                     [ 1.,  0.]],
            <BLANKLINE>
                    [[ 0.,  1.],
                     [ 1.,  0.]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[ 0.,  1.],
                     [ 1.,  0.]],
            <BLANKLINE>
                    [[ 0.,  1.],
                     [ 1.,  0.]]]])

            >>> all_permutations_operation(
            ...     operator.sub, numpy.ones((2,2)), numpy.fliplr(numpy.eye(2))
            ... )
            array([[[[ 1.,  0.],
                     [ 0.,  1.]],
            <BLANKLINE>
                    [[ 1.,  0.],
                     [ 0.,  1.]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[ 1.,  0.],
                     [ 0.,  1.]],
            <BLANKLINE>
                    [[ 1.,  0.],
                     [ 0.,  1.]]]])

            >>> all_permutations_operation(
            ...     operator.sub, numpy.zeros((2,2)), numpy.eye(2)
            ... )
            array([[[[-1.,  0.],
                     [ 0., -1.]],
            <BLANKLINE>
                    [[-1.,  0.],
                     [ 0., -1.]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[-1.,  0.],
                     [ 0., -1.]],
            <BLANKLINE>
                    [[-1.,  0.],
                     [ 0., -1.]]]])
    """

    new_array_1_tiled = expand_view(new_array_1, reps_after=new_array_2.shape)
    new_array_2_tiled = expand_view(new_array_2, reps_before=new_array_1.shape)

    return(new_op(new_array_1_tiled, new_array_2_tiled))


@prof.log_call(trace_logger)
def all_permutations_equal(new_array_1, new_array_2):
    """
        Takes two arrays and constructs a new array that contains the result
        of equality comparison on every permutation of elements in each array
        (like broadcasting).

        Suppose that new_result contained the result, then one would find that
        the result of the following operation on the specific indices

        new_op( new_array_1[ i_1_1, i_1_2, ... ],
        new_array_2[ i_2_1, i_2_2, ... ] )

        would be found in new_result as shown

        new_result[ i_1_1, i_1_2, ..., i_2_1, i_2_2, ... ]


        Args:
            new_array(numpy.ndarray):            array to add the singleton
                                                 axis to.

        Returns:
            (numpy.ndarray):                     a numpy array with the
                                                 singleton axis added at the
                                                 end.

        Examples:
            >>> all_permutations_equal(numpy.ones((1,3)), numpy.eye(2)).shape
            (1, 3, 2, 2)

            >>> all_permutations_equal(numpy.ones((2,2)), numpy.eye(2)).shape
            (2, 2, 2, 2)

            >>> all_permutations_equal(numpy.ones((2,2)), numpy.eye(2))
            array([[[[ True, False],
                     [False,  True]],
            <BLANKLINE>
                    [[ True, False],
                     [False,  True]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[ True, False],
                     [False,  True]],
            <BLANKLINE>
                    [[ True, False],
                     [False,  True]]]], dtype=bool)

            >>> all_permutations_equal(
            ...     numpy.ones((2,2)), numpy.fliplr(numpy.eye(2))
            ... )
            array([[[[False,  True],
                     [ True, False]],
            <BLANKLINE>
                    [[False,  True],
                     [ True, False]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[False,  True],
                     [ True, False]],
            <BLANKLINE>
                    [[False,  True],
                     [ True, False]]]], dtype=bool)

            >>> all_permutations_equal(numpy.zeros((2,2)), numpy.eye(2))
            array([[[[False,  True],
                     [ True, False]],
            <BLANKLINE>
                    [[False,  True],
                     [ True, False]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[False,  True],
                     [ True, False]],
            <BLANKLINE>
                    [[False,  True],
                     [ True, False]]]], dtype=bool)

            >>> all_permutations_equal(numpy.zeros((2,2)), numpy.eye(3))
            array([[[[False,  True,  True],
                     [ True, False,  True],
                     [ True,  True, False]],
            <BLANKLINE>
                    [[False,  True,  True],
                     [ True, False,  True],
                     [ True,  True, False]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[False,  True,  True],
                     [ True, False,  True],
                     [ True,  True, False]],
            <BLANKLINE>
                    [[False,  True,  True],
                     [ True, False,  True],
                     [ True,  True, False]]]], dtype=bool)

            >>> all_permutations_equal(
            ...     numpy.arange(4).reshape((2,2)),
            ...     numpy.arange(2,6).reshape((2,2))
            ... )
            array([[[[False, False],
                     [False, False]],
            <BLANKLINE>
                    [[False, False],
                     [False, False]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[ True, False],
                     [False, False]],
            <BLANKLINE>
                    [[False,  True],
                     [False, False]]]], dtype=bool)
    """

    return(all_permutations_operation(operator.eq, new_array_1, new_array_2))


class NotNumPyStructuredArrayType(Exception):
    """
        Designed for being thrown if a NumPy Structured Array is received.
    """
    pass


@prof.log_call(trace_logger)
def numpy_structured_array_dtype_generator(new_array):
    """
        Takes a NumPy structured array and returns a generator that goes over
        each name in the structured array and yields the name, type, and shape
        (for the given name).

        Args:
            new_array(numpy.ndarray):       the array to get the info dtype
                                            from.

        Raises:
            (NotNumPyStructuredArrayType):  if it is a normal NumPy array.

        Returns:
            (iterator):                     An iterator yielding tuples.
    """

    # Test to see if this is a NumPy Structured Array
    if not new_array.dtype.names: raise NotNumPyStructuredArrayType(
        "Not a NumPy structured array."
    )

    # Go through each name
    for each_name in new_array.dtype.names:
        # Get the type (want the actual type, not a str or dtype object)
        each_dtype = new_array[each_name].dtype.type
        # Get the shape (will be an empty tuple if no shape, which numpy.dtype
        # accepts)
        each_shape = new_array.dtype[each_name].shape

        yield ((each_name, each_dtype, each_shape))


@prof.log_call(trace_logger)
def numpy_structured_array_dtype_list(new_array):
    """
        Takes any NumPy array and returns either a list for a NumPy structured
        array via numpy_structured_array_dtype_generator or if it is a normal
        NumPy array it returns the type used.

        Args:
            new_array(numpy.ndarray):       the array to get the dtype info
                                            from.

        Returns:
            (list or type):                 something that can be given to
                                            numpy.dtype to obtain the
                                            new_array.dtype, but is more
                                            malleable than a numpy.dtype.
    """

    return(list(numpy_structured_array_dtype_generator(new_array)))


@prof.log_call(trace_logger)
def blocks_split(space_shape, block_shape, block_halo=None):
    """
        Return a list of slicings to cut each block out of an array or other.

        Takes an array with ``space_shape`` and ``block_shape`` for every
        dimension and a ``block_halo`` to extend each block on each side. From
        this, it can compute slicings to use for cutting each block out from
        the original array, HDF5 dataset or other.

        Note:
            Blocks on the boundary that cannot extend the full range will
            be truncated to the largest block that will fit. This will raise
            a warning, which can be converted to an exception, if needed.

        Args:
            space_shape(numpy.ndarray):    Shape of array to slice
            block_shape(numpy.ndarray):    Size of each block to take
            block_halo(numpy.ndarray):     Halo to tack on to each block

        Returns:
            collections.Sequence of \
            tuples of slices:              Provides tuples of slices for \
                                           retrieving blocks.

        Examples:
            >>> blocks_split(
            ...     (2,), (1,)
            ... )  #doctest: +NORMALIZE_WHITESPACE
            ([(slice(0, 1, 1),), (slice(1, 2, 1),)],
            <BLANKLINE>
             [(slice(0, 1, 1),), (slice(1, 2, 1),)],
            <BLANKLINE>
             [(slice(0, 1, 1),), (slice(0, 1, 1),)])

            >>> blocks_split((2,), (-1,))
            ([(slice(0, 2, 1),)], [(slice(0, 2, 1),)], [(slice(0, 2, 1),)])

            >>> blocks_split(
            ...     (2, 3,), (1, 1,)
            ... )  #doctest: +NORMALIZE_WHITESPACE
            ([(slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(1, 2, 1)),
              (slice(0, 1, 1), slice(2, 3, 1)),
              (slice(1, 2, 1), slice(0, 1, 1)),
              (slice(1, 2, 1), slice(1, 2, 1)),
              (slice(1, 2, 1), slice(2, 3, 1))],
            <BLANKLINE>
             [(slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(1, 2, 1)),
              (slice(0, 1, 1), slice(2, 3, 1)),
              (slice(1, 2, 1), slice(0, 1, 1)),
              (slice(1, 2, 1), slice(1, 2, 1)),
              (slice(1, 2, 1), slice(2, 3, 1))],
            <BLANKLINE>
             [(slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1))])

            >>> blocks_split(
            ...     (2, 3,), (1, 1,), (0, 0,)
            ... )  #doctest: +NORMALIZE_WHITESPACE
            ([(slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(1, 2, 1)),
              (slice(0, 1, 1), slice(2, 3, 1)),
              (slice(1, 2, 1), slice(0, 1, 1)),
              (slice(1, 2, 1), slice(1, 2, 1)),
              (slice(1, 2, 1), slice(2, 3, 1))],
            <BLANKLINE>
             [(slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(1, 2, 1)),
              (slice(0, 1, 1), slice(2, 3, 1)),
              (slice(1, 2, 1), slice(0, 1, 1)),
              (slice(1, 2, 1), slice(1, 2, 1)),
              (slice(1, 2, 1), slice(2, 3, 1))],
            <BLANKLINE>
             [(slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(0, 1, 1))])

            >>> blocks_split(
            ...     (2, 3,), (1, 1,), (1, 1,)
            ... )  #doctest: +NORMALIZE_WHITESPACE
            ([(slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(1, 2, 1)),
              (slice(0, 1, 1), slice(2, 3, 1)),
              (slice(1, 2, 1), slice(0, 1, 1)),
              (slice(1, 2, 1), slice(1, 2, 1)),
              (slice(1, 2, 1), slice(2, 3, 1))],
            <BLANKLINE>
             [(slice(0, 2, 1), slice(0, 2, 1)),
              (slice(0, 2, 1), slice(0, 3, 1)),
              (slice(0, 2, 1), slice(1, 3, 1)),
              (slice(0, 2, 1), slice(0, 2, 1)),
              (slice(0, 2, 1), slice(0, 3, 1)),
              (slice(0, 2, 1), slice(1, 3, 1))],
            <BLANKLINE>
             [(slice(0, 1, 1), slice(0, 1, 1)),
              (slice(0, 1, 1), slice(1, 2, 1)),
              (slice(0, 1, 1), slice(1, 2, 1)),
              (slice(1, 2, 1), slice(0, 1, 1)),
              (slice(1, 2, 1), slice(1, 2, 1)),
              (slice(1, 2, 1), slice(1, 2, 1))])


            >>> blocks_split(
            ...     (10, 12,), (3, 2,), (4, 3,)
            ... )  #doctest: +NORMALIZE_WHITESPACE
            ([(slice(0, 3, 1), slice(0, 2, 1)),
              (slice(0, 3, 1), slice(2, 4, 1)),
              (slice(0, 3, 1), slice(4, 6, 1)),
              (slice(0, 3, 1), slice(6, 8, 1)),
              (slice(0, 3, 1), slice(8, 10, 1)),
              (slice(0, 3, 1), slice(10, 12, 1)),
              (slice(3, 6, 1), slice(0, 2, 1)),
              (slice(3, 6, 1), slice(2, 4, 1)),
              (slice(3, 6, 1), slice(4, 6, 1)),
              (slice(3, 6, 1), slice(6, 8, 1)),
              (slice(3, 6, 1), slice(8, 10, 1)),
              (slice(3, 6, 1), slice(10, 12, 1)),
              (slice(6, 9, 1), slice(0, 2, 1)),
              (slice(6, 9, 1), slice(2, 4, 1)),
              (slice(6, 9, 1), slice(4, 6, 1)),
              (slice(6, 9, 1), slice(6, 8, 1)),
              (slice(6, 9, 1), slice(8, 10, 1)),
              (slice(6, 9, 1), slice(10, 12, 1)),
              (slice(9, 10, 1), slice(0, 2, 1)),
              (slice(9, 10, 1), slice(2, 4, 1)),
              (slice(9, 10, 1), slice(4, 6, 1)),
              (slice(9, 10, 1), slice(6, 8, 1)),
              (slice(9, 10, 1), slice(8, 10, 1)),
              (slice(9, 10, 1), slice(10, 12, 1))],
            <BLANKLINE>
             [(slice(0, 7, 1), slice(0, 5, 1)),
              (slice(0, 7, 1), slice(0, 7, 1)),
              (slice(0, 7, 1), slice(1, 9, 1)),
              (slice(0, 7, 1), slice(3, 11, 1)),
              (slice(0, 7, 1), slice(5, 12, 1)),
              (slice(0, 7, 1), slice(7, 12, 1)),
              (slice(0, 10, 1), slice(0, 5, 1)),
              (slice(0, 10, 1), slice(0, 7, 1)),
              (slice(0, 10, 1), slice(1, 9, 1)),
              (slice(0, 10, 1), slice(3, 11, 1)),
              (slice(0, 10, 1), slice(5, 12, 1)),
              (slice(0, 10, 1), slice(7, 12, 1)),
              (slice(2, 10, 1), slice(0, 5, 1)),
              (slice(2, 10, 1), slice(0, 7, 1)),
              (slice(2, 10, 1), slice(1, 9, 1)),
              (slice(2, 10, 1), slice(3, 11, 1)),
              (slice(2, 10, 1), slice(5, 12, 1)),
              (slice(2, 10, 1), slice(7, 12, 1)),
              (slice(5, 10, 1), slice(0, 5, 1)),
              (slice(5, 10, 1), slice(0, 7, 1)),
              (slice(5, 10, 1), slice(1, 9, 1)),
              (slice(5, 10, 1), slice(3, 11, 1)),
              (slice(5, 10, 1), slice(5, 12, 1)),
              (slice(5, 10, 1), slice(7, 12, 1))],
            <BLANKLINE>
              [(slice(0, 3, 1), slice(0, 2, 1)),
               (slice(0, 3, 1), slice(2, 4, 1)),
               (slice(0, 3, 1), slice(3, 5, 1)),
               (slice(0, 3, 1), slice(3, 5, 1)),
               (slice(0, 3, 1), slice(3, 5, 1)),
               (slice(0, 3, 1), slice(3, 5, 1)),
               (slice(3, 6, 1), slice(0, 2, 1)),
               (slice(3, 6, 1), slice(2, 4, 1)),
               (slice(3, 6, 1), slice(3, 5, 1)),
               (slice(3, 6, 1), slice(3, 5, 1)),
               (slice(3, 6, 1), slice(3, 5, 1)),
               (slice(3, 6, 1), slice(3, 5, 1)),
               (slice(4, 7, 1), slice(0, 2, 1)),
               (slice(4, 7, 1), slice(2, 4, 1)),
               (slice(4, 7, 1), slice(3, 5, 1)),
               (slice(4, 7, 1), slice(3, 5, 1)),
               (slice(4, 7, 1), slice(3, 5, 1)),
               (slice(4, 7, 1), slice(3, 5, 1)),
               (slice(4, 7, 1), slice(0, 2, 1)),
               (slice(4, 7, 1), slice(2, 4, 1)),
               (slice(4, 7, 1), slice(3, 5, 1)),
               (slice(4, 7, 1), slice(3, 5, 1)),
               (slice(4, 7, 1), slice(3, 5, 1)),
               (slice(4, 7, 1), slice(3, 5, 1))])

    """

    space_shape = numpy.array(space_shape)
    block_shape = numpy.array(block_shape)

    if block_halo is not None:
        block_halo = numpy.array(block_halo)

        assert (space_shape.ndim == block_shape.ndim == block_halo.ndim == 1),\
            "There should be no more than 1 dimension for " + \
            "`space_shape`, `block_shape`, and `block_halo`."
        assert (len(space_shape) == len(block_shape) == len(block_halo)), \
            "The dimensions of `space_shape`, `block_shape`, and " + \
            "`block_halo` should be the same."
    else:
        assert (space_shape.ndim == block_shape.ndim == 1), \
            "There should be no more than 1 dimension for " + \
            "`space_shape` and `block_shape`."
        assert (len(space_shape) == len(block_shape)), \
            "The dimensions of `space_shape` and `block_shape` " + \
            "should be the same."

        block_halo = numpy.zeros_like(space_shape)

    uneven_block_division = (space_shape % block_shape != 0)

    if uneven_block_division.any():
        uneven_block_division_str = uneven_block_division.nonzero()[0].tolist()
        uneven_block_division_str = [str(_) for _ in uneven_block_division_str]
        uneven_block_division_str = ", ".join(uneven_block_division_str)

        warnings.warn(
            "Blocks will not evenly divide the array." +
            " The following dimensions will be unevenly divided: %s." %
            uneven_block_division_str,
            RuntimeWarning
        )

    ranges_per_dim = []
    haloed_ranges_per_dim = []
    trimmed_halos_per_dim = []

    for each_dim in iters.irange(len(space_shape)):
        # Construct each block using the block size given. Allow to spill over.
        if block_shape[each_dim] == -1:
            block_shape[each_dim] = space_shape[each_dim]

        a_range = numpy.arange(0, space_shape[each_dim], block_shape[each_dim])
        a_range = expand_view(a_range, reps_before=2).copy()
        a_range[1] += block_shape[each_dim]

        # Add the halo to each block on both sides.
        a_range_haloed = a_range.copy()
        a_range_haloed[1] += block_halo[each_dim]
        a_range_haloed[0] -= block_halo[each_dim]
        a_range_haloed.clip(0, space_shape[each_dim], out=a_range_haloed)

        # Compute how to trim the halo off of each block.
        # Clip each block to the boundaries.
        a_trimmed_halo = numpy.empty_like(a_range)
        a_trimmed_halo[...] = a_range - a_range_haloed[0]
        a_range.clip(0, space_shape[each_dim], out=a_range)

        # Transpose to allow for iteration over each block's dimension.
        a_range = a_range.T.copy()
        a_range_haloed = a_range_haloed.T.copy()
        a_trimmed_halo = a_trimmed_halo.T.copy()

        # Convert all ranges to slices for easier use.
        a_range = iters.reformat_slices([
            slice(*a_range[i]) for i in iters.irange(len(a_range))
        ])
        a_range_haloed = iters.reformat_slices([
            slice(*a_range_haloed[i]) for i in iters.irange(len(a_range_haloed))
        ])
        a_trimmed_halo = iters.reformat_slices([
            slice(*a_trimmed_halo[i]) for i in iters.irange(len(a_trimmed_halo))
        ])

        # Collect all blocks
        ranges_per_dim.append(a_range)
        haloed_ranges_per_dim.append(a_range_haloed)
        trimmed_halos_per_dim.append(a_trimmed_halo)

    # Take all combinations of all ranges to get blocks.
    blocks = list(itertools.product(*ranges_per_dim))
    haloed_blocks = list(itertools.product(*haloed_ranges_per_dim))
    trimmed_halos = list(itertools.product(*trimmed_halos_per_dim))

    return(blocks, haloed_blocks, trimmed_halos)


@prof.log_call(trace_logger)
def dot_product(new_vector_set_1, new_vector_set_2):
    """
        Determines the dot product between the two pairs of vectors from each
        set.

        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.

        Returns:
            (numpy.ndarray):                      an array with the distances
                                                  between each pair of vectors.

        Examples:
            >>> (dot_product(numpy.eye(2), numpy.eye(2)) == numpy.eye(2)).all()
            True

            >>> (dot_product(numpy.eye(10), numpy.eye(10)) == numpy.eye(10)).all()
            True

            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[ 1,  0]]))
            array([[ 1.]])

            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[ 0,  1]]))
            array([[ 0.]])

            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[-1,  0]]))
            array([[-1.]])

            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[ 0, -1]]))
            array([[ 0.]])

            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[ 1,  1]]))
            array([[ 1.]])

            >>> dot_product(
            ...     numpy.array([[ True,  False]]),
            ...     numpy.array([[ True,  True]])
            ... )
            array([[ 1.]])
    """

    if not issubclass(new_vector_set_1.dtype.type, numpy.floating):
        new_vector_set_1 = new_vector_set_1.astype(numpy.float64)
    if not issubclass(new_vector_set_2.dtype.type, numpy.floating):
        new_vector_set_2 = new_vector_set_2.astype(numpy.float64)

    # Measure the dot product between any two neurons
    # (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(
        new_vector_set_1, new_vector_set_2.T
    )

    return(vector_pairs_dot_product)


@prof.log_call(trace_logger)
def pair_dot_product(new_vector_set):
    """
        Determines the dot product between the vectors in the set.

        Args:
            new_vector_set(numpy.ndarray):        set of vectors.

        Returns:
            (numpy.ndarray):                      an array with the distances
                                                  between each pair of vectors.

        Examples:
            >>> (pair_dot_product(numpy.eye(2)) == numpy.eye(2)).all()
            True

            >>> (pair_dot_product(numpy.eye(10)) == numpy.eye(10)).all()
            True

            >>> pair_dot_product(numpy.array([[ 1,  0]]))
            array([[ 1.]])

            >>> pair_dot_product(numpy.array([[ 1.,  0.]]))
            array([[ 1.]])

            >>> pair_dot_product(numpy.array([[-1,  0]]))
            array([[ 1.]])

            >>> pair_dot_product(numpy.array([[ 0,  1]]))
            array([[ 1.]])

            >>> pair_dot_product(numpy.array([[ 1,  1]]))
            array([[ 2.]])

            >>> pair_dot_product(numpy.array([[ True,  False]]))
            array([[ 1.]])
    """

    return(dot_product(new_vector_set, new_vector_set))


@prof.log_call(trace_logger)
def norm(new_vector_set, ord=2):
    """
        Determines the norm of a vector or a set of vectors.

        Args:
            new_vector_set(numpy.ndarray):        either a single vector or a
                                                  set of vectors (matrix).

            ord(optional):                        basically the same arguments
                                                  as numpy.linalg.norm
                                                  (though some are redundant
                                                  here).

        Returns:
            (numpy.ndarray):                      an array with the norms of
                                                  all vectors in the set.

        Examples:
            >>> norm(numpy.array([ 1,  0]), 2).ndim
            0

            >>> norm(numpy.array([[ 1,  0]]), 2).ndim
            1

            >>> norm(numpy.array([ 1,  0]), 2)
            array(1.0)

            >>> norm(numpy.array([ 1,  0]), 1)
            array(1.0)

            >>> norm(numpy.array([[ 1,  0]]), 2)
            array([ 1.])

            >>> norm(numpy.array([[ 1,  1]]), 1)
            array([ 2.])

            >>> norm(numpy.array([[ 1,  1]]), 2)
            array([ 1.41421356])

            >>> norm(numpy.array([[ True,  False]]), 1)
            array([ 1.])

            >>> norm(numpy.array([[ True,  False]]), 2)
            array([ 1.])

            >>> norm(numpy.array([[ True,  True]]), 1)
            array([ 2.])

            >>> norm(numpy.array([[ True,  True]]), 2)
            array([ 1.41421356])

            >>> norm(numpy.array([[ 1,  1,  1], [ 1,  0,  1]]), 1)
            array([ 3.,  2.])

            >>> norm(numpy.array([[ 1,  1,  1], [ 1,  0,  1]]), 2)
            array([ 1.73205081,  1.41421356])

            >>> norm(numpy.array([ 0,  1,  2]))
            array(2.23606797749979)

            >>> norm(numpy.zeros((0, 2,)))
            array([], shape=(0, 2), dtype=float64)

            >>> norm(numpy.zeros((2, 0,)))
            array([], shape=(2, 0), dtype=float64)
    """

    assert ord > 0

    # Needs to have at least one vector
    assert (new_vector_set.ndim >= 1)

    # Must be double.
    if not issubclass(new_vector_set.dtype.type, numpy.floating):
        new_vector_set = new_vector_set.astype(numpy.float64)

    result = None

    # Return a scalar NumPy array in the case of a single vector
    # Always return type float as the result.
    if 0 in new_vector_set.shape:
        result = new_vector_set
    else:
        result = numpy.linalg.norm(new_vector_set, ord=ord, axis=-1)

        if new_vector_set.ndim:
            result = numpy.array(result)

    return(result)


@prof.log_call(trace_logger)
def threshold_array(an_array, threshold, include_below=True, is_closed=True):
    """
        Given a threshold, this function compares the given array to it to see
        which entries match.

        Args:
            an_array(numpy.ndarray):                          an array to
                                                              threshold.

            threshold(int or float or numpy.ndarray):         something to
                                                              compare to.

            include_below(bool):                              whether values
                                                              below the
                                                              threshold count
                                                              or ones above it.

            is_closed(bool):                                  whether to
                                                              include values
                                                              equal to the
                                                              threshold

        Returns:
            out(numpy.ndarray):                               a mask of entries
                                                              reflecting the
                                                              threshold.

        Examples:
            >>> threshold_array(
            ...     numpy.arange(6).reshape(2,3),
            ...     0,
            ...     include_below=True,
            ...     is_closed=False
            ... )
            array([[False, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_array(
            ...     numpy.arange(6).reshape(2,3),
            ...     0,
            ...     include_below=True,
            ...     is_closed=True
            ... )
            array([[ True, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_array(
            ...     numpy.arange(6).reshape(2,3),
            ...     0,
            ...     include_below=False,
            ...     is_closed=True
            ... )
            array([[ True,  True,  True],
                   [ True,  True,  True]], dtype=bool)

            >>> threshold_array(
            ...     numpy.arange(6).reshape(2,3),
            ...     0,
            ...     include_below=False,
            ...     is_closed=False
            ... )
            array([[False,  True,  True],
                   [ True,  True,  True]], dtype=bool)

            >>> threshold_array(
            ...     numpy.arange(6).reshape(2,3),
            ...     5,
            ...     include_below=True,
            ...     is_closed=True
            ... )
            array([[ True,  True,  True],
                   [ True,  True,  True]], dtype=bool)

            >>> threshold_array(
            ...     numpy.arange(6).reshape(2,3),
            ...     6,
            ...     include_below=True,
            ...     is_closed=False
            ... )
            array([[ True,  True,  True],
                   [ True,  True,  True]], dtype=bool)

            >>> threshold_array(
            ...     numpy.arange(6).reshape(2,3),
            ...     2*numpy.ones((2,3)),
            ...     include_below=True,
            ...     is_closed=True
            ... )
            array([[ True,  True,  True],
                   [False, False, False]], dtype=bool)

            >>> threshold_array(
            ...     numpy.arange(6).reshape(2,3) % 2,
            ...     1,
            ...     include_below=False,
            ...     is_closed=True
            ... )
            array([[False,  True, False],
                   [ True, False,  True]], dtype=bool)

            >>> threshold_array(
            ...     numpy.arange(6).reshape(2,3) % 2,
            ...     1,
            ...     include_below=True,
            ...     is_closed=False
            ... )
            array([[ True, False,  True],
                   [False,  True, False]], dtype=bool)
    """

    accepted = numpy.zeros(an_array.shape, dtype=bool)

    if include_below and is_closed:
        accepted[:] = (an_array <= threshold)
    elif not include_below and is_closed:
        accepted[:] = (an_array >= threshold)
    elif include_below and not is_closed:
        accepted[:] = (an_array < threshold)
    else:
        accepted[:] = (an_array > threshold)

    return(accepted)


@prof.log_call(trace_logger)
def unique_mapping(mapping, out=None):
    """
        Take a binary mapping between two sets and excludes portions of the
        mapping that are not one-to-one.

        Args:
            mapping(numpy.ndarray):      bool array mapping between to sets.
            out(numpy.ndarray):          where to store the results.

        Returns:
            out(numpy.ndarray):          the results returned.

        Examples:
            >>> unique_mapping(numpy.zeros((2,2), dtype=bool))
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> unique_mapping(numpy.ones((2,2), dtype=bool))
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> unique_mapping(numpy.eye(2, dtype=bool))
            array([[ True, False],
                   [False,  True]], dtype=bool)

            >>> unique_mapping(
            ...     numpy.array([[ True,  True],
            ...                  [ True, False]], dtype=bool)
            ... )
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> unique_mapping(
            ...     numpy.array([[ True, False],
            ...                  [False, False]], dtype=bool)
            ... )
            array([[ True, False],
                   [False, False]], dtype=bool)

            >>> a = numpy.ones((2,2), dtype=bool); b = a.copy(); a
            array([[ True,  True],
                   [ True,  True]], dtype=bool)

            >>> unique_mapping(a, out=b)
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> b
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> unique_mapping(a, out=a)
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> a
            array([[False, False],
                   [False, False]], dtype=bool)
    """

    assert issubclass(mapping.dtype.type, numpy.bool_)
    assert (mapping.ndim == 2)

    if out is None:
        out = mapping.copy()
    elif id(mapping) != id(out):
        out[:] = mapping

    injective_into = list()
    for i in iters.irange(mapping.ndim):
        injective_into_i = (add_singleton_op(numpy.sum, mapping, i) == 1)
        injective_into.append(injective_into_i)

    for i in iters.irange(mapping.ndim):
        out *= injective_into[i]

    return(out)


@prof.log_call(trace_logger)
def threshold_metric(a_metric, threshold, include_below=True, is_closed=True):
    """
        Given a threshold, this function finds which entries uniquely match
        given the threshold.

        Args:
            a_metric(numpy.ndarray):                          an array to
                                                              threshold.

            threshold(int or float or numpy.ndarray):         something to
                                                              compare to.

            include_below(bool):                              whether values
                                                              below the
                                                              threshold count
                                                              or ones above it.

            is_closed(bool):                                  whether to
                                                              include values
                                                              equal to the
                                                              threshold

        Returns:
            out(numpy.ndarray):                               a mapping of
                                                              unique matches.

        Examples:
            >>> threshold_metric(
            ...     numpy.arange(6).reshape(2,3),
            ...     0,
            ...     include_below=True,
            ...     is_closed=False
            ... )
            array([[False, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_metric(
            ...     numpy.arange(6).reshape(2,3),
            ...     0,
            ...     include_below=True,
            ...     is_closed=True
            ... )
            array([[ True, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_metric(
            ...     numpy.arange(6).reshape(2,3),
            ...     0,
            ...     include_below=False,
            ...     is_closed=True
            ... )
            array([[False, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_metric(
            ...     numpy.arange(6).reshape(2,3),
            ...     0,
            ...     include_below=False,
            ...     is_closed=False
            ... )
            array([[False, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_metric(
            ...     numpy.arange(6).reshape(2,3),
            ...     5,
            ...     include_below=True,
            ...     is_closed=True
            ... )
            array([[False, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_metric(
            ...     numpy.arange(6).reshape(2,3),
            ...     6,
            ...     include_below=True,
            ...     is_closed=False
            ... )
            array([[False, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_metric(
            ...     numpy.arange(6).reshape(2,3),
            ...     2*numpy.ones((2,3)),
            ...     include_below=True,
            ...     is_closed=True
            ... )
            array([[False, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_metric(
            ...     numpy.arange(6).reshape(2,3),
            ...     numpy.arange(0, 12, 2).reshape(2,3),
            ...     include_below=True,
            ...     is_closed=True
            ... )
            array([[False, False, False],
                   [False, False, False]], dtype=bool)

            >>> threshold_metric(
            ...     numpy.arange(6).reshape(2,3) % 2,
            ...     1,
            ...     include_below=True,
            ...     is_closed=False
            ... )
            array([[False, False, False],
                   [False,  True, False]], dtype=bool)
    """

    assert (a_metric.ndim == 2)

    accepted = unique_mapping(threshold_array(
        a_metric, threshold, include_below=include_below, is_closed=is_closed))

    return(accepted)


@prof.log_call(trace_logger)
def compute_mapping_matches(mapping):
    """
        Given a mapping this function computes number of matches and mismatches
        found.

        In the array returned, first value is the number of true positives (or
        matches) and then mismatches along each dimension of the mapping in
        order.

        If axis 0 is the ground truth, then the second and third values are the
        number of false negatives (or misses) and false positives (or false
        alarm)

        Args:
            mapping(numpy.ndarray):                           a 2D bool array
                                                              mapping
                                                              intersections
                                                              between 2 groups.

        Returns:
            out(numpy.ndarray):                               Counts of the
                                                              number of matches
                                                              and mismatches.

        Examples:
            >>> compute_mapping_matches(numpy.arange(6).reshape(2,3) < 0)
            array([0, 2, 3], dtype=uint64)

            >>> compute_mapping_matches(numpy.arange(6).reshape(2,3) <= 0)
            array([1, 1, 2], dtype=uint64)

            >>> compute_mapping_matches(
            ...     (numpy.arange(6).reshape(2,3) % 2) == 1
            ... )
            array([1, 1, 2], dtype=uint64)

            >>> compute_mapping_matches(numpy.eye(2, dtype=bool))
            array([2, 0, 0], dtype=uint64)

            >>> compute_mapping_matches(numpy.fliplr(numpy.eye(2, dtype=bool)))
            array([2, 0, 0], dtype=uint64)
    """

    assert issubclass(mapping.dtype.type, numpy.bool_)
    assert (mapping.ndim == 2)

    stats = numpy.zeros((3,), dtype=numpy.uint64)

    stats[:] = unique_mapping(mapping).sum()
    mapping_shape = numpy.array(mapping.shape, dtype=numpy.uint64)
    stats[1:] = mapping_shape - stats[1:]

    return(stats)


@prof.log_call(trace_logger)
def compute_mapping_relevance(mapping):
    """
        Given a mapping this function computes the recall and precision.

        If axis 0 is the ground truth, then the returned values are the recall
        and precision in order. If axis 1 is the ground truth, then they are
        flipped.

        Args:
            mapping(numpy.ndarray):            a 2D bool array mapping
                                               intersections between 2 groups.

        Returns:
            relevance(tuple of floats):        relevance - a combination of
                                                           recall and precision
                                                   recall    - the ratio of
                                                               relevant
                                                               predicted
                                                               positives out of
                                                               all relevant
                                                               positives.
                                                   precision - the ratio of
                                                               relevant
                                                               predicted
                                                               positives out of
                                                               all predicted
                                                               positives.

        Examples:
            >>> compute_mapping_relevance(numpy.arange(6).reshape(2,3) < 0)
            array([ 0.,  0.])

            >>> compute_mapping_relevance(numpy.arange(6).reshape(2,3) <= 0)
            array([ 0.5       ,  0.33333333])

            >>> compute_mapping_relevance(
            ...     (numpy.arange(6).reshape(2,3) % 2) == 1
            ... )
            array([ 0.5       ,  0.33333333])

            >>> compute_mapping_relevance(numpy.eye(2, dtype=bool))
            array([ 1.,  1.])

            >>> compute_mapping_relevance(
            ...     numpy.fliplr(numpy.eye(2, dtype=bool))
            ... )
            array([ 1.,  1.])
    """

    assert issubclass(mapping.dtype.type, numpy.bool_)
    assert (mapping.ndim == 2)

    matches = compute_mapping_matches(mapping)
    mapping_shape = numpy.array(mapping.shape)

    relevance = numpy.zeros((2,), dtype=float)
    relevance[:] = matches[0]
    relevance /= mapping_shape

    return(relevance)


@prof.log_call(trace_logger)
def find_relative_offsets(points, center=None, out=None):
    """
        Given a series of points, find the relative points from the mean.

        Args:
            points(numpy.ndarray):       a set of integer points (NxD) where N
                                         is the number of points and D is the
                                         dimensionality, in which they lay.

            center(numpy.ndarray):       an integer point (D) where D is the
                                         dimensionality, in which they lay.
                                         Defaults to the mean of the points.

            out(numpy.ndarray):          another set of points relative to
                                         their mean.

        Returns:
            out(numpy.ndarray):          the results returned.

        Examples:
            >>> find_relative_offsets(numpy.zeros((3,2), dtype=int))
            array([[0, 0],
                   [0, 0],
                   [0, 0]])

            >>> find_relative_offsets(numpy.ones((3,2), dtype=int))
            array([[0, 0],
                   [0, 0],
                   [0, 0]])

            >>> find_relative_offsets(
            ...     numpy.ones((3,2), dtype=int), -numpy.ones((2,), dtype=int)
            ... )
            array([[2, 2],
                   [2, 2],
                   [2, 2]])

            >>> find_relative_offsets(numpy.arange(6).reshape(2,3).T % 2)
            array([[ 0,  0],
                   [ 1, -1],
                   [ 0,  0]])

            >>> a = numpy.arange(6).reshape(2,3).T % 2; b = numpy.zeros_like(a)
            >>> find_relative_offsets(a, out=b)
            array([[ 0,  0],
                   [ 1, -1],
                   [ 0,  0]])
            >>> b
            array([[ 0,  0],
                   [ 1, -1],
                   [ 0,  0]])

            >>> a = numpy.arange(6).reshape(2,3).T % 2
            >>> find_relative_offsets(a, out=a)
            array([[ 0,  0],
                   [ 1, -1],
                   [ 0,  0]])
            >>> a
            array([[ 0,  0],
                   [ 1, -1],
                   [ 0,  0]])
    """

    if out is None:
        out = points.copy()
    elif id(points) != id(out):
        out[:] = points

    if center is None:
        center = numpy.round(
            out.mean(axis=0)
        ).astype(int)

    out -= center[None]

    return(out)


@prof.log_call(trace_logger)
def find_shortest_wraparound(points, shape, out=None):
    """
        Compute the smallest values for the points given periodic boundary
        conditions.

        Args:
            points(numpy.ndarray):       a set of integer points (NxD) where N
                                         is the number of points and D is the
                                         dimensionality, in which they lay.

            shape(numpy.ndarray):        the shape to use for wrapping (D).

            out(numpy.ndarray):          another set of points relative to
                                         their mean.

        Returns:
            out(numpy.ndarray):          the results returned.

        Examples:
            >>> find_shortest_wraparound(
            ...     numpy.zeros((3, 2), dtype=int),
            ...     (4, 8)
            ... )
            array([[0, 0],
                   [0, 0],
                   [0, 0]])

            >>> find_shortest_wraparound(
            ...     numpy.ones((3, 2), dtype=int),
            ...     (4, 8)
            ... )
            array([[1, 1],
                   [1, 1],
                   [1, 1]])

            >>> find_shortest_wraparound(
            ...     4 * numpy.ones((3, 2), dtype=int),
            ...     (4, 8)
            ... )
            array([[0, 4],
                   [0, 4],
                   [0, 4]])

            >>> find_shortest_wraparound(
            ...     8 * (numpy.arange(6).reshape(3, 2) % 2),
            ...     (4, 8)
            ... )
            array([[0, 0],
                   [0, 0],
                   [0, 0]])

            >>> find_shortest_wraparound(
            ...     7 * (numpy.arange(6).reshape(3, 2) % 2),
            ...     (4, 8)
            ... )
            array([[ 0, -1],
                   [ 0, -1],
                   [ 0, -1]])

            >>> find_shortest_wraparound(
            ...     -numpy.ones((3, 2), dtype=int),
            ...     (4, 8)
            ... )
            array([[-1, -1],
                   [-1, -1],
                   [-1, -1]])

            >>> find_shortest_wraparound(
            ...     -4 * numpy.ones((3, 2), dtype=int),
            ...     (4, 8)
            ... )
            array([[ 0, -4],
                   [ 0, -4],
                   [ 0, -4]])

            >>> find_shortest_wraparound(
            ...     -7 * (numpy.arange(6).reshape(3, 2) % 2),
            ...     (4, 8)
            ... )
            array([[0, 1],
                   [0, 1],
                   [0, 1]])

            >>> a = -7 * (numpy.arange(6).reshape(3, 2) % 2)
            >>> b = numpy.zeros_like(a)
            >>> find_shortest_wraparound(
            ...     a,
            ...     (4, 8),
            ...     out=b
            ... )
            array([[0, 1],
                   [0, 1],
                   [0, 1]])
            >>> b
            array([[0, 1],
                   [0, 1],
                   [0, 1]])

            >>> a = -7 * (numpy.arange(6).reshape(3, 2) % 2)
            >>> find_shortest_wraparound(
            ...     a,
            ...     (4, 8),
            ...     out=a
            ... )
            array([[0, 1],
                   [0, 1],
                   [0, 1]])
            >>> a
            array([[0, 1],
                   [0, 1],
                   [0, 1]])
    """

    if out is None:
        out = points.copy()
    elif id(points) != id(out):
        out[:] = points

    shape = numpy.array(shape)
    half_shape = numpy.trunc(shape / 2.0).astype(int)

    points_mask_above = (
        out > half_shape[None]
    )
    if points_mask_above.any():
        out -= points_mask_above * shape
    points_mask_below = (
        out < -half_shape[None]
    )
    if points_mask_below.any():
        out += points_mask_below * shape

    return(out)


@prof.log_call(trace_logger)
def matrix_reduced_op(a, b, op):
    """
        Sort of like numpy.dot. However, it will use the first axis with both
        arrays. This means they may not need to be matrices. However, they must
        have the same number of dimensions. Generally, though not explicitly
        required, the operator will likely expect every dimension other than
        the first to be the same shape.

        Args:
            a(numpy.ndarray):      first array.
            b(numpy.ndarray):      second array.
            op(callable):          an operator that will take a[i] and b[j] as
                                   arguments and return a scalar.

        Returns:
            out(numpy.ndarray):    an array (matrix) with the shape (len(a),
                                   len(b)) with each element out[i, j] the
                                   result of op(a[i], b[j]).

        Examples:
            >>> matrix_reduced_op(
            ...     numpy.arange(6).reshape(2,3),
            ...     numpy.arange(0,12,2).reshape(2,3),
            ...     op=numpy.dot
            ... )
            array([[ 10,  28],
                   [ 28, 100]])

            >>> numpy.dot(
            ...     numpy.arange(6).reshape(2,3),
            ...     numpy.arange(0,12,2).reshape(2,3).T,
            ... )
            array([[ 10,  28],
                   [ 28, 100]])

            >>> matrix_reduced_op(
            ...     numpy.arange(8).reshape(2,4),
            ...     numpy.arange(0,16,2).reshape(2,4),
            ...     op=numpy.dot
            ... )
            array([[ 28,  76],
                   [ 76, 252]])

            >>> numpy.dot(
            ...     numpy.arange(8).reshape(2,4),
            ...     numpy.arange(0,16,2).reshape(2,4).T,
            ... )
            array([[ 28,  76],
                   [ 76, 252]])

            >>> matrix_reduced_op(numpy.eye(2).astype(bool),
            ...                   numpy.ones((2,2), dtype=bool),
            ...                   op=lambda _a, _b: (_a & _b).sum())
            array([[ True,  True],
                   [ True,  True]], dtype=bool)

            >>> matrix_reduced_op(numpy.eye(2).astype(bool),
            ...                   numpy.ones((2,), dtype=bool),
            ...                   op=lambda _a, _b: (_a & _b).sum())
            array([[ True],
                   [ True]], dtype=bool)

            >>> matrix_reduced_op(numpy.ones((2,), dtype=bool),
            ...                   numpy.eye(2).astype(bool),
            ...                   op=lambda _a, _b: (_a & _b).sum())
            array([[ True,  True]], dtype=bool)

            >>> matrix_reduced_op(numpy.ones((2,), dtype=bool),
            ...                    numpy.ones((2,), dtype=bool),
            ...                    op=lambda _a, _b: (_a & _b).sum())
            array([[ True]], dtype=bool)

            >>> matrix_reduced_op(numpy.eye(2).astype(bool),
            ...                   numpy.zeros((2,2), dtype=bool),
            ...                   op=lambda _a, _b: (_a & _b).sum())
            array([[False, False],
                   [False, False]], dtype=bool)

            >>> matrix_reduced_op(numpy.eye(2).astype(bool),
            ...                   numpy.zeros((2,2), dtype=bool),
            ...                   op=lambda _a, _b: (_a | _b).sum())
            array([[ True,  True],
                   [ True,  True]], dtype=bool)
    """

    if a.ndim == 1:
        a = a[None]

    if b.ndim == 1:
        b = b[None]

    assert (a.ndim == b.ndim)

    for i in iters.irange(1, a.ndim):
        assert (a.shape[i] == b.shape[i])

    out = numpy.empty(
        (len(a), len(b)), dtype=numpy.promote_types(a.dtype, b.dtype)
    )

    for i, j in itertools.product(iters.irange(out.shape[0]), iters.irange(out.shape[1])):
        out[i, j] = op(a[i], b[j])

    return(out)


@prof.log_call(trace_logger)
def masks_intersection(a, b):
    """
        Generates a mask that contains the points share by both masks.

        Args:
            a(numpy.ndarray):      first mask.
            b(numpy.ndarray):      second mask.

        Returns:
            out(numpy.ndarray):    a mask that is only True where both masks
                                   are.

        Examples:
            >>> masks_intersection(numpy.eye(2).astype(bool),
            ...                    numpy.ones((2,2), dtype=bool))
            array([[1, 1],
                   [1, 1]], dtype=uint64)

            >>> masks_intersection(numpy.eye(2).astype(bool),
            ...                    numpy.ones((2,), dtype=bool))
            array([[1],
                   [1]], dtype=uint64)

            >>> masks_intersection(numpy.ones((2,), dtype=bool),
            ...                    numpy.eye(2).astype(bool))
            array([[1, 1]], dtype=uint64)

            >>> masks_intersection(numpy.ones((2,), dtype=bool),
            ...                    numpy.ones((2,), dtype=bool))
            array([[2]], dtype=uint64)

            >>> masks_intersection(numpy.eye(2).astype(bool),
            ...                    numpy.zeros((2,2), dtype=bool))
            array([[0, 0],
                   [0, 0]], dtype=uint64)

            >>> (numpy.arange(6).reshape(2,3) % 2).astype(bool)
            array([[False,  True, False],
                   [ True, False,  True]], dtype=bool)

            >>> numpy.arange(6).reshape(2,3) == 4
            array([[False, False, False],
                   [False,  True, False]], dtype=bool)

            >>> masks_intersection(
            ...     (numpy.arange(6).reshape(2,3) % 2).astype(bool),
            ...     (numpy.arange(6).reshape(2,3) == 4)
            ... )
            array([[0, 1],
                   [0, 0]], dtype=uint64)
    """

    assert issubclass(a.dtype.type, numpy.bool_)
    assert issubclass(b.dtype.type, numpy.bool_)

    if a.ndim == 1:
        a = a[None]

    if b.ndim == 1:
        b = b[None]

    assert (a.ndim == b.ndim == 2)
    assert (a.shape[1] == b.shape[1])

    out = numpy.empty((len(a), len(b)), dtype=numpy.uint64)

    for i, j in itertools.product(iters.irange(out.shape[0]), iters.irange(out.shape[1])):
        out[i, j] = (a[i] & b[j]).sum()

    return(out)


@prof.log_call(trace_logger)
def masks_union(a, b):
    """
        A mask that contains point contained in either mask.

        Args:
            a(numpy.ndarray):      first mask.
            b(numpy.ndarray):      second mask.

        Returns:
            out(numpy.ndarray):    a mask that is True where either mask is.

        Examples:
            >>> masks_union(numpy.eye(2).astype(bool),
            ...             numpy.ones((2,2), dtype=bool))
            array([[2, 2],
                   [2, 2]], dtype=uint64)

            >>> masks_union(numpy.eye(2).astype(bool),
            ...             numpy.ones((2,), dtype=bool))
            array([[2],
                   [2]], dtype=uint64)

            >>> masks_union(numpy.ones((2,), dtype=bool),
            ...             numpy.eye(2).astype(bool))
            array([[2, 2]], dtype=uint64)

            >>> masks_union(numpy.ones((2,), dtype=bool),
            ...             numpy.ones((2,), dtype=bool))
            array([[2]], dtype=uint64)

            >>> masks_union(numpy.eye(2).astype(bool),
            ...             numpy.zeros((2,2), dtype=bool))
            array([[1, 1],
                   [1, 1]], dtype=uint64)

            >>> (numpy.arange(6).reshape(2,3) % 2).astype(bool)
            array([[False,  True, False],
                   [ True, False,  True]], dtype=bool)

            >>> numpy.arange(6).reshape(2,3) == 4
            array([[False, False, False],
                   [False,  True, False]], dtype=bool)

            >>> masks_union((numpy.arange(6).reshape(2,3) % 2).astype(bool),
            ...             (numpy.arange(6).reshape(2,3) == 4))
            array([[1, 1],
                   [2, 3]], dtype=uint64)
    """

    assert issubclass(a.dtype.type, numpy.bool_)
    assert issubclass(b.dtype.type, numpy.bool_)

    if a.ndim == 1:
        a = a[None]

    if b.ndim == 1:
        b = b[None]

    assert (a.ndim == b.ndim == 2)
    assert (a.shape[1] == b.shape[1])

    out = numpy.empty((len(a), len(b)), dtype=numpy.uint64)

    for i, j in itertools.product(iters.irange(out.shape[0]), iters.irange(out.shape[1])):
        out[i, j] = (a[i] | b[j]).sum()

    return(out)


@prof.log_call(trace_logger)
def masks_overlap_normalized(a, b):
    """
        The area of intersection of the masks divided by the area of their
        union.

        Args:
            a(numpy.ndarray):      first mask.
            b(numpy.ndarray):      second mask.

        Returns:
            out(numpy.ndarray):    ratio of the areas of the masks'
                                   intersection and union.

        Examples:
            >>> masks_overlap_normalized(numpy.eye(2).astype(bool),
            ...                          numpy.ones((2,2), dtype=bool))
            array([[ 0.5,  0.5],
                   [ 0.5,  0.5]])

            >>> masks_overlap_normalized(numpy.eye(2).astype(bool),
            ...                          numpy.ones((2,), dtype=bool))
            array([[ 0.5],
                   [ 0.5]])

            >>> masks_overlap_normalized(numpy.ones((2,), dtype=bool),
            ...                          numpy.eye(2).astype(bool))
            array([[ 0.5,  0.5]])

            >>> masks_overlap_normalized(numpy.ones((2,), dtype=bool),
            ...                          numpy.ones((2,), dtype=bool))
            array([[ 1.]])

            >>> masks_overlap_normalized(numpy.eye(2).astype(bool),
            ...                          numpy.zeros((2,2), dtype=bool))
            array([[ 0.,  0.],
                   [ 0.,  0.]])

            >>> masks_overlap_normalized(numpy.zeros((2,2), dtype=bool),
            ...                          numpy.zeros((2,2), dtype=bool))
            array([[ 0.,  0.],
                   [ 0.,  0.]])

            >>> (numpy.arange(6).reshape(2,3) % 2).astype(bool)
            array([[False,  True, False],
                   [ True, False,  True]], dtype=bool)

            >>> numpy.arange(6).reshape(2,3) == 4
            array([[False, False, False],
                   [False,  True, False]], dtype=bool)

            >>> masks_overlap_normalized(
            ...     (numpy.arange(6).reshape(2,3) % 2).astype(bool),
            ...     (numpy.arange(6).reshape(2,3) == 4)
            ... )
            array([[ 0.,  1.],
                   [ 0.,  0.]])
    """

    assert issubclass(a.dtype.type, numpy.bool_)
    assert issubclass(b.dtype.type, numpy.bool_)

    if a.ndim == 1:
        a = a[None]

    if b.ndim == 1:
        b = b[None]

    assert (a.ndim == b.ndim == 2)
    assert (a.shape[1] == b.shape[1])

    out = masks_intersection(a, b).astype(numpy.float64)
    out /= masks_union(a, b)

    out[numpy.isnan(out)] = 0

    return(out)


@prof.log_call(trace_logger)
def dot_product_partially_normalized(new_vector_set_1,
                                     new_vector_set_2,
                                     ord=2):
    """
        Determines the dot product between the two pairs of vectors from each
        set and creates a tuple with the dot product divided by one norm or the
        other.

        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.

            new_vector_set_2(numpy.ndarray):      second set of vectors.

            ord(optional):                        basically the same arguments
                                                  as numpy.linalg.norm

        Returns:
            (numpy.ndarray):                      an array with the normalized
                                                  distances between each pair
                                                  of vectors.

        Examples:
            >>> (numpy.array(dot_product_partially_normalized(numpy.eye(2), numpy.eye(2), 2)) == numpy.array((numpy.eye(2), numpy.eye(2),))).all()
            True

            >>> (numpy.array(dot_product_partially_normalized(numpy.eye(10), numpy.eye(10), 2)) == numpy.array((numpy.eye(10), numpy.eye(10),))).all()
            True

            >>> dot_product_partially_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 1,  0]]),
            ...     2
            ... )
            (array([[ 1.]]), array([[ 1.]]))

            >>> dot_product_partially_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 0,  1]]),
            ...     2
            ... )
            (array([[ 0.]]), array([[ 0.]]))

            >>> dot_product_partially_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[-1,  0]]),
            ...     2
            ... )
            (array([[-1.]]), array([[-1.]]))

            >>> dot_product_partially_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 0, -1]]),
            ...     2
            ... )
            (array([[ 0.]]), array([[ 0.]]))

            >>> dot_product_partially_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 1,  1]]),
            ...     2
            ... )
            (array([[ 1.]]), array([[ 0.70710678]]))

            >>> dot_product_partially_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 1,  1]]),
            ...     1
            ... )
            (array([[ 1.]]), array([[ 0.5]]))

            >>> dot_product_partially_normalized(
            ...     numpy.array([[ True,  False]]),
            ...     numpy.array([[ True,  True]]),
            ...     2
            ... )
            (array([[ 1.]]), array([[ 0.70710678]]))

            >>> dot_product_partially_normalized(
            ...     numpy.array([[ True,  False]]),
            ...     numpy.array([[ True,  True]]),
            ...     1
            ... )
            (array([[ 1.]]), array([[ 0.5]]))

            >>> dot_product_partially_normalized(
            ...     numpy.arange(6).reshape((2,3)),
            ...     numpy.arange(5, 17).reshape((4,3)),
            ...     2
            ... )  #doctest: +NORMALIZE_WHITESPACE
            (array([[  8.94427191,  12.96919427,  16.99411663,  21.01903899],
                    [ 10.46518036,  15.55634919,  20.64751801,  25.73868684]]),
             array([[ 1.90692518,  1.85274204,  1.82405837,  1.80635674],
                    [ 7.05562316,  7.02764221,  7.00822427,  6.99482822]]))
    """

    assert ord > 0

    # Must be double.
    if not issubclass(new_vector_set_1.dtype.type, numpy.floating):
        new_vector_set_1 = new_vector_set_1.astype(numpy.float64)
    if not issubclass(new_vector_set_2.dtype.type, numpy.floating):
        new_vector_set_2 = new_vector_set_2.astype(numpy.float64)

    # Gets all of the norms
    new_vector_set_1_norms = norm(new_vector_set_1, ord)
    new_vector_set_2_norms = norm(new_vector_set_2, ord)

    # Expand the norms to have a shape equivalent to vector_pairs_dot_product
    new_vector_set_1_norms_expanded = expand_view(
        new_vector_set_1_norms, reps_after=new_vector_set_2.shape[0])
    new_vector_set_2_norms_expanded = expand_view(
        new_vector_set_2_norms, reps_before=new_vector_set_1.shape[0])

    # Measure the dot product between any two neurons
    # (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(
        new_vector_set_1, new_vector_set_2.T)

    vector_pairs_dot_product_1_normalized = vector_pairs_dot_product / \
        new_vector_set_1_norms_expanded
    vector_pairs_dot_product_2_normalized = vector_pairs_dot_product / \
        new_vector_set_2_norms_expanded

    return(
        (vector_pairs_dot_product_1_normalized,
         vector_pairs_dot_product_2_normalized)
    )


@prof.log_call(trace_logger)
def pair_dot_product_partially_normalized(new_vector_set, ord=2, float_type=None):
    """
        Determines the dot product between the two pairs of vectors from each
        set and creates a tuple with the dot product divided by one norm or the
        other.

        Args:
            new_vector_set(numpy.ndarray):        set of vectors.

            ord(optional):                        basically the same argument
                                                  as numpy.linalg.norm

            float_type(type):                     some form of float

        Returns:
            (numpy.ndarray):                      an array with the normalized
                                                  distances between each pair
                                                  of vectors.

        Examples:
            >>> (pair_dot_product_partially_normalized(numpy.eye(2), 2) == numpy.eye(2)).all()
            True

            >>> (pair_dot_product_partially_normalized(numpy.eye(10), 2) == numpy.eye(10)).all()
            True

            >>> pair_dot_product_partially_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     2
            ... )
            array([[ 1.]])

            >>> pair_dot_product_partially_normalized(
            ...     numpy.array([[-1,  0]]),
            ...     2
            ... )
            array([[ 1.]])

            >>> pair_dot_product_partially_normalized(
            ...     numpy.array([[ 1,  1]]),
            ...     2
            ... )
            array([[ 1.41421356]])

            >>> pair_dot_product_partially_normalized(
            ...     numpy.array([[ 1,  1]]),
            ...     1
            ... )
            array([[ 1.]])

            >>> pair_dot_product_partially_normalized(
            ...     numpy.array([[ True,  False]]),
            ...     2
            ... )
            array([[ 1.]])

            >>> pair_dot_product_partially_normalized(
            ...     numpy.array([[ True,  True]]),
            ...     1
            ... )
            array([[ 1.]])

            >>> pair_dot_product_partially_normalized(
            ...     numpy.arange(6).reshape((2,3)),
            ...     2
            ... )
            array([[ 2.23606798,  6.26099034],
                   [ 1.97989899,  7.07106781]])

            >>> pair_dot_product_partially_normalized(
            ...     numpy.array([[ True, False, False],
            ...                  [ True, False,  True]]),
            ...     3
            ... )
            array([[ 1.        ,  1.        ],
                   [ 0.79370053,  1.58740105]])
    """

    assert ord > 0

    is_bool = (new_vector_set.dtype == bool)

    if float_type is not None:
        float_type = numpy.dtype(float_type).type
        assert issubclass(float_type, numpy.floating)

        if not issubclass(new_vector_set.dtype.type, float_type):
            new_vector_set = new_vector_set.astype(float_type)
    else:
        float_type = numpy.float64
        if not issubclass(new_vector_set.dtype.type, numpy.floating):
            new_vector_set = new_vector_set.astype(float_type)
        else:
            float_type = new_vector_set.dtype.type

    # Measure the dot product between any two neurons
    # (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(
        new_vector_set, new_vector_set.T
    )

    # Gets all of the norms
    if is_bool:
        new_vector_set_norms = vector_pairs_dot_product.diagonal()
        if ord != numpy.inf:
            inv_ord = 1.0 / float_type(ord)
            new_vector_set_norms = new_vector_set_norms ** inv_ord
        else:
            new_vector_set_norms = (new_vector_set_norms > 0).astype(float_type)
    elif ord == 2:
        new_vector_set_norms = numpy.sqrt(vector_pairs_dot_product.diagonal())
    else:
        new_vector_set_norms = norm(new_vector_set, ord=ord)

    # Expand the norms to have a shape equivalent to vector_pairs_dot_product
    new_vector_set_norms_expanded = expand_view(
        new_vector_set_norms, reps_after=new_vector_set.shape[0]
    )

    # Measure the dot product between any two neurons
    # (i.e. related to the angle of separation)
    vector_pairs_dot_product_normalized = vector_pairs_dot_product / new_vector_set_norms_expanded

    return(vector_pairs_dot_product_normalized)


@prof.log_call(trace_logger)
def dot_product_normalized(new_vector_set_1, new_vector_set_2, ord=2):
    """
        Determines the dot product between a pair of vectors from each set and
        divides them by the norm of the two.

        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.

            new_vector_set_2(numpy.ndarray):      second set of vectors.

            ord(optional):                        basically the same arguments
                                                  as numpy.linalg.norm.

        Returns:
            (numpy.ndarray):                      an array with the normalized
                                                  distances between each pair
                                                  of vectors.

        Examples:
            >>> (dot_product_normalized(numpy.eye(2), numpy.eye(2), 2) == numpy.eye(2)).all()
            True

            >>> (dot_product_normalized(numpy.eye(10), numpy.eye(10), 2) == numpy.eye(10)).all()
            True

            >>> dot_product_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 1,  0]]),
            ...     2
            ... )
            array([[ 1.]])

            >>> dot_product_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 0,  1]]),
            ...     2
            ... )
            array([[ 0.]])

            >>> dot_product_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[-1,  0]]),
            ...     2
            ... )
            array([[-1.]])

            >>> dot_product_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 0, -1]]),
            ...     2
            ... )
            array([[ 0.]])

            >>> dot_product_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 1,  1]]),
            ...     2
            ... )
            array([[ 0.70710678]])

            >>> dot_product_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 1,  1]]),
            ...     1
            ... )
            array([[ 0.5]])

            >>> dot_product_normalized(
            ...     numpy.array([[ True,  False]]),
            ...     numpy.array([[ True,   True]]),
            ...     2
            ... )
            array([[ 0.70710678]])

            >>> dot_product_normalized(
            ...     numpy.array([[ True, False]]),
            ...     numpy.array([[ True,  True]]),
            ...     1
            ... )
            array([[ 0.5]])

            >>> dot_product_normalized(
            ...     numpy.arange(6).reshape((2,3)),
            ...     numpy.arange(5, 17).reshape((4,3)),
            ...     2
            ... )
            array([[ 0.85280287,  0.82857143,  0.8157437 ,  0.80782729],
                   [ 0.9978158 ,  0.99385869,  0.99111258,  0.98921809]])
    """

    assert ord > 0

    # Must be double.
    if not issubclass(new_vector_set_1.dtype.type, numpy.floating):
        new_vector_set_1 = new_vector_set_1.astype(numpy.float64)
    if not issubclass(new_vector_set_2.dtype.type, numpy.floating):
        new_vector_set_2 = new_vector_set_2.astype(numpy.float64)

    # Gets all of the norms
    new_vector_set_1_norms = norm(new_vector_set_1, ord=ord)
    new_vector_set_2_norms = norm(new_vector_set_2, ord=ord)

    # Finds the product of each combination for normalization
    norm_products = all_permutations_operation(
        operator.mul, new_vector_set_1_norms, new_vector_set_2_norms
    )

    # Measure the dot product between any two neurons
    # (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(
        new_vector_set_1, new_vector_set_2.T
    )

    # Measure the dot product between any two neurons
    # (i.e. related to the angle of separation)
    vector_pairs_dot_product_normalized = vector_pairs_dot_product / norm_products

    return(vector_pairs_dot_product_normalized)


@prof.log_call(trace_logger)
def pair_dot_product_normalized(new_vector_set, ord=2, float_type=None):
    """
        Determines the dot product between a pair of vectors from each set and
        divides them by the norm of the two.

        Args:
            new_vector_set(numpy.ndarray):        set of vectors.

            ord(optional):                        basically the same arguments
                                                  as numpy.linalg.norm.

            float_type(type):                     some form of float

        Returns:
            (numpy.ndarray):                      an array with the normalized
                                                  distances between each pair
                                                  of vectors.

        Examples:
            >>> (pair_dot_product_normalized(numpy.eye(2)) == numpy.eye(2)).all()
            True

            >>> (pair_dot_product_normalized(numpy.eye(10)) == numpy.eye(10)).all()
            True

            >>> pair_dot_product_normalized(
            ...     numpy.array([[ 1,  0]])
            ... )
            array([[ 1.]])

            >>> pair_dot_product_normalized(
            ...     numpy.array([[ 1.,  0.]])
            ... )
            array([[ 1.]])

            >>> pair_dot_product_normalized(
            ...     numpy.array([[-1,  0]]),
            ... )
            array([[ 1.]])

            >>> pair_dot_product_normalized(
            ...     numpy.array([[ 0,  1]]),
            ... )
            array([[ 1.]])

            >>> pair_dot_product_normalized(
            ...     numpy.array([[ 1,  1]]),
            ... )
            array([[ 1.]])

            >>> pair_dot_product_normalized(
            ...     numpy.array([[ True,  False]]),
            ... )
            array([[ 1.]])

            >>> pair_dot_product_normalized(
            ...     numpy.arange(6).reshape((2,3)),
            ...     2
            ... )
            array([[ 1.        ,  0.88543774],
                   [ 0.88543774,  1.        ]])

            >>> pair_dot_product_normalized(
            ...     numpy.array([[ True, False, False],
            ...                  [ True, False,  True]]),
            ...     3
            ... )
            array([[ 1.        ,  0.79370053],
                   [ 0.79370053,  1.25992105]])
    """

    assert ord > 0

    is_bool = (new_vector_set.dtype == bool)

    if float_type is not None:
        float_type = numpy.dtype(float_type).type
        assert issubclass(float_type, numpy.floating)

        if not issubclass(new_vector_set.dtype.type, float_type):
            new_vector_set = new_vector_set.astype(float_type)
    else:
        float_type = numpy.float64
        if not issubclass(new_vector_set.dtype.type, numpy.floating):
            new_vector_set = new_vector_set.astype(float_type)
        else:
            float_type = new_vector_set.dtype.type

    # Measure the dot product between any two neurons
    # (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(
        new_vector_set, new_vector_set.T
    )

    # Gets all of the norms
    if is_bool:
        new_vector_set_norms = vector_pairs_dot_product.diagonal()
        if ord != numpy.inf:
            inv_ord = 1.0 / float_type(ord)
            new_vector_set_norms = new_vector_set_norms ** inv_ord
        else:
            new_vector_set_norms = (new_vector_set_norms > 0).astype(float_type)
    elif ord == 2:
        new_vector_set_norms = numpy.sqrt(vector_pairs_dot_product.diagonal())
    else:
        new_vector_set_norms = norm(new_vector_set, ord=ord)

    # Finds the product of each combination for normalization
    norm_products = all_permutations_operation(
        operator.mul, new_vector_set_norms, new_vector_set_norms
    )

    # Measure the dot product between any two neurons
    # (i.e. related to the angle of separation)
    vector_pairs_dot_product_normalized = vector_pairs_dot_product / norm_products

    return(vector_pairs_dot_product_normalized)


@prof.log_call(trace_logger)
def dot_product_L2_normalized(new_vector_set_1, new_vector_set_2):
    """
        Determines the dot product between a pair of vectors from each set and
        divides them by the L_2 norm of the two.

        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.

        Returns:
            (numpy.ndarray):                      an array with the distances
                                                  between each pair of vectors.

        Examples:
            >>> (dot_product_L2_normalized(numpy.eye(2), numpy.eye(2)) == numpy.eye(2)).all()
            True

            >>> (dot_product_L2_normalized(numpy.eye(10), numpy.eye(10)) == numpy.eye(10)).all()
            True

            >>> dot_product_L2_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 1,  0]]),
            ... )
            array([[ 1.]])

            >>> dot_product_L2_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 0,  1]]),
            ... )
            array([[ 0.]])

            >>> dot_product_L2_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[-1,  0]]),
            ... )
            array([[-1.]])

            >>> dot_product_L2_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 0, -1]]),
            ... )
            array([[ 0.]])

            >>> dot_product_L2_normalized(
            ...     numpy.array([[ 1,  0]]),
            ...     numpy.array([[ 1,  1]]),
            ... )
            array([[ 0.70710678]])

            >>> dot_product_L2_normalized(
            ...     numpy.array([[ True,  False]]),
            ...     numpy.array([[ True,   True]]),
            ... )
            array([[ 0.70710678]])

            >>> dot_product_L2_normalized(
            ...     numpy.arange(6).reshape((2,3)),
            ...     numpy.arange(5, 17).reshape((4,3)),
            ... )
            array([[ 0.85280287,  0.82857143,  0.8157437 ,  0.80782729],
                   [ 0.9978158 ,  0.99385869,  0.99111258,  0.98921809]])
    """

    if not issubclass(new_vector_set_1.dtype.type, numpy.floating):
        new_vector_set_1 = new_vector_set_1.astype(numpy.float64)
    if not issubclass(new_vector_set_2.dtype.type, numpy.floating):
        new_vector_set_2 = new_vector_set_2.astype(numpy.float64)

    # Measure the angle between any two neurons
    # (i.e. related to the angle of separation)
    vector_pairs_cosine_angle = 1 - scipy.spatial.distance.cdist(new_vector_set_1,
                                                                 new_vector_set_2,
                                                                 "cosine")

    return(vector_pairs_cosine_angle)


def generate_contour_fast(a_image):
    """
        Takes an image and extracts labeled contours from the mask.

        Args:
            a_image(numpy.ndarray):            takes an image.

        Returns:
            (numpy.ndarray):                   an array with the labeled
                                               contours.

        Examples:
            >>> a = numpy.array([[ True,  True, False],
            ...                  [False, False, False],
            ...                  [ True,  True,  True]], dtype=bool)
            >>> generate_contour(a)
            array([[ True,  True, False],
                   [False, False, False],
                   [ True,  True,  True]], dtype=bool)

            >>> generate_contour(numpy.eye(3))
            array([[ 1.,  0.,  0.],
                   [ 0.,  1.,  0.],
                   [ 0.,  0.,  1.]])

            >>> a = numpy.array([
            ...     [False, False,  True, False, False, False,  True],
            ...     [ True, False, False, False,  True, False, False],
            ...     [ True,  True, False,  True,  True, False,  True],
            ...     [ True, False, False,  True,  True, False, False],
            ...     [ True, False, False, False, False, False, False],
            ...     [False,  True, False, False, False, False,  True],
            ...     [False,  True,  True, False, False, False, False]
            ... ], dtype=bool)
            >>> generate_contour(a)
            array([[False, False,  True, False, False, False,  True],
                   [ True, False, False, False,  True, False, False],
                   [ True,  True, False,  True,  True, False,  True],
                   [ True, False, False,  True,  True, False, False],
                   [ True, False, False, False, False, False, False],
                   [False,  True, False, False, False, False,  True],
                   [False,  True,  True, False, False, False, False]], dtype=bool)
    """

    structure = numpy.ones((3,)*a_image.ndim, dtype=bool)

    a_mask = (a_image != 0)
    a_mask ^= mahotas.erode(
            a_image, structure
    )

    a_image_contours = a_image * a_mask

    return(a_image_contours)


def generate_contour(a_image, separation_distance=1.0, margin=1.0):
    """
        Takes an image and extracts labeled contours from the mask using some
        minimum distance from the mask edge and some margin.

        Args:
            a_image(numpy.ndarray):            takes an image.

            separation_distance(float):        a separation distance from the
                                               edge of the mask for the center
                                               of the contour.

            margin(float):                     the width of contour.

        Returns:
            (numpy.ndarray):                   an array with the labeled
                                               contours.

        Examples:
            >>> a = numpy.array([[ True,  True, False],
            ...                  [False, False, False],
            ...                  [ True,  True,  True]], dtype=bool)
            >>> generate_contour(a)
            array([[ True,  True, False],
                   [False, False, False],
                   [ True,  True,  True]], dtype=bool)

            >>> generate_contour(numpy.eye(3))
            array([[ 1.,  0.,  0.],
                   [ 0.,  1.,  0.],
                   [ 0.,  0.,  1.]])

            >>> a = numpy.array([
            ...     [False, False,  True, False, False, False,  True],
            ...     [ True, False, False, False,  True, False, False],
            ...     [ True,  True, False,  True,  True, False,  True],
            ...     [ True, False, False,  True,  True, False, False],
            ...     [ True, False, False, False, False, False, False],
            ...     [False,  True, False, False, False, False,  True],
            ...     [False,  True,  True, False, False, False, False]
            ... ], dtype=bool)
            >>> generate_contour(a)
            array([[False, False,  True, False, False, False,  True],
                   [ True, False, False, False,  True, False, False],
                   [ True,  True, False,  True,  True, False,  True],
                   [ True, False, False,  True,  True, False, False],
                   [ True, False, False, False, False, False, False],
                   [False,  True, False, False, False, False,  True],
                   [False,  True,  True, False, False, False, False]], dtype=bool)
    """

    half_thickness = margin / 2

    lower_threshold = separation_distance - half_thickness
    upper_threshold = separation_distance + half_thickness

    a_mask_transformed = mahotas.distance(
            a_image,
            "euclidean"
    )

    above_lower_threshold = (lower_threshold <= a_mask_transformed)
    below_upper_threshold = (a_mask_transformed <= upper_threshold)

    a_mask_transformed_thresholded = (
        above_lower_threshold & below_upper_threshold
    )

    a_image_contours = a_image * a_mask_transformed_thresholded

    return(a_image_contours)


def generate_labeled_contours(a_mask, separation_distance=1.0, margin=1.0):
    """
        Takes a bool mask and extracts labeled contours from the mask using
        some minimum distance from the mask edge and some margin.

        Args:
            a_mask(numpy.ndarray):             takes a bool mask.

            separation_distance(float):        a separation distance from the
                                               edge of the mask for the center
                                               of the contour.

            margin(float):                     the width of contour.

        Returns:
            (numpy.ndarray):                   an array with the labeled
                                               contours.

        Examples:
            >>> a = numpy.array([[ True,  True, False],
            ...                  [False, False, False],
            ...                  [ True,  True,  True]], dtype=bool)
            >>> generate_labeled_contours(a)
            array([[1, 1, 0],
                   [0, 0, 0],
                   [2, 2, 2]], dtype=int32)

            >>> generate_labeled_contours(numpy.eye(3))
            array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]], dtype=int32)

            >>> a = numpy.array([
            ...     [False, False,  True, False, False, False,  True],
            ...     [ True, False, False, False,  True, False, False],
            ...     [ True,  True, False,  True,  True, False,  True],
            ...     [ True, False, False,  True,  True, False, False],
            ...     [ True, False, False, False, False, False, False],
            ...     [False,  True, False, False, False, False,  True],
            ...     [False,  True,  True, False, False, False, False]
            ... ], dtype=bool)
            >>> generate_labeled_contours(a)
            array([[0, 0, 1, 0, 0, 0, 2],
                   [3, 0, 0, 0, 4, 0, 0],
                   [3, 3, 0, 4, 4, 0, 5],
                   [3, 0, 0, 4, 4, 0, 0],
                   [3, 0, 0, 0, 0, 0, 0],
                   [0, 3, 0, 0, 0, 0, 6],
                   [0, 3, 3, 0, 0, 0, 0]], dtype=int32)
    """

    a_mask_contoured = generate_contour(
        a_mask, separation_distance=separation_distance, margin=margin
    )

    a_mask_contoured_labeled = scipy.ndimage.label(
        a_mask_contoured, structure=numpy.ones((3,) * a_mask.ndim)
    )[0]

    return(a_mask_contoured_labeled)


def get_quantiles(probs):
    """
        Determines the probabilites for quantiles for given data much like
        MATLAB's function

        Args:
            data(numpy.ndarray):                        to find the quantiles
                                                        of.

            probs(int or float or numpy.ndarray):       either some sort of
                                                        integer for the number
                                                        of quantiles or a
                                                        single float specifying
                                                        which quantile to get
                                                        or an array of floats
                                                        specifying the division
                                                        for each quantile in
                                                        the the range (0, 1).

            axis(int or None):                          the axis to perform the
                                                        calculation on (if
                                                        default (None) then
                                                        all, otherwise only on
                                                        a particular axis.

        Returns:
            (numpy.ma.MaskedArray):                     an array with the
                                                        quantiles (the first
                                                        dimension will be the
                                                        same length as probs).

        Examples:
            >>> get_quantiles(0)
            array([], dtype=float64)

            >>> get_quantiles(1)
            array([ 0.5])

            >>> get_quantiles(3)
            array([ 0.25,  0.5 ,  0.75])

            >>> get_quantiles(0.5)
            array([ 0.5])

            >>> get_quantiles([0.25, 0.75])
            array([ 0.25,  0.75])

            >>> get_quantiles(numpy.array([0.25, 0.75]))
            array([ 0.25,  0.75])
    """

    probs_type = collections.Sequence
    if not isinstance(probs, collections.Sequence):
        probs_type = numpy.dtype(type(probs)).type

    probs_array = None
    if issubclass(probs_type, numpy.integer):
        num_quantiles = probs
        probs_array = numpy.linspace(0, 1, num_quantiles + 2)[1:-1]
    elif issubclass(probs_type, numpy.floating):
        a_quantile = probs
        probs_array = numpy.array([a_quantile])
    else:
        probs_array = numpy.array(probs)
        probs_array.sort()


    if not ((0 < probs_array) & (probs_array < 1)).all(): raise Exception(
        "Cannot pass values that are not within the range (0, 1)."
    )


    return(probs_array)


def quantile(data, probs, axis=None):
    """
        Determines the quantiles for given data much like MATLAB's function.

        Args:
            data(numpy.ndarray):                        to find the quantiles
                                                        of.

            probs(int or float or numpy.ndarray):       either some sort of
                                                        integer for the number
                                                        of quantiles or a
                                                        single float specifying
                                                        which quantile to get
                                                        or an array of floats
                                                        specifying the division
                                                        for each quantile in
                                                        the range (0, 1).

            axis(int or None):                          the axis to perform the
                                                        calculation on (if
                                                        default (None) then
                                                        all, otherwise only on
                                                        a particular axis.

        Returns:
            (numpy.ma.MaskedArray):                     an array with the
                                                        quantiles (the first
                                                        dimension will be the
                                                        same length as probs).

        Examples:
            >>> quantile(numpy.array([ 1.,  2.,  3.]), 2)
            masked_array(data = [ 1.5  2.5],
                         mask = False,
                   fill_value = nan)
            <BLANKLINE>

            >>> quantile(numpy.array([ 1.,  2.,  3.]), 3)
            masked_array(data = [ 1.25  2.    2.75],
                         mask = False,
                   fill_value = nan)
            <BLANKLINE>

            >>> quantile(
            ...     numpy.array([ 1.,  2.,  3.]),
            ...     numpy.array([ 0.25,  0.5,  0.75])
            ... )
            masked_array(data = [ 1.25  2.    2.75],
                         mask = False,
                   fill_value = nan)
            <BLANKLINE>

            >>> quantile(numpy.array([ 1.,  2.,  3.]), 0.5)
            masked_array(data = [ 2.],
                         mask = False,
                   fill_value = nan)
            <BLANKLINE>

            >>> a = numpy.array([[-1.1176, -0.0679, -0.3031,  0.8261],
            ...                  [ 1.2607, -0.1952,  0.023 ,  1.527 ],
            ...                  [ 0.6601, -0.2176,  0.0513,  0.4669]])
            >>> quantile(a, 2, axis = 0)
            masked_array(data =
             [[-0.22875 -0.2064  -0.14005  0.6465 ]
             [ 0.9604  -0.13155  0.03715  1.17655]],
                         mask =
             False,
                   fill_value = nan)
            <BLANKLINE>
    """

    probs_array = get_quantiles(probs)

    new_quantiles = scipy.stats.mstats.mquantiles(
        data, probs_array, alphap=0.5, betap=0.5, axis=axis
    )

    if not isinstance(new_quantiles, numpy.ma.MaskedArray):
        new_quantiles = numpy.ma.MaskedArray(new_quantiles)

    new_quantiles.set_fill_value(numpy.nan)

    return(new_quantiles)


@prof.log_call(trace_logger)
def binomial_coefficients(n):
    """
        Generates a row in Pascal's triangle (binomial coefficients).

        Args:
            n(int):                 which row of Pascal's triangle to return.

        Returns:
            cs(numpy.ndarray):      a numpy array containing the row of
                                    Pascal's triangle.


        Examples:
            >>> binomial_coefficients(-25)
            array([], dtype=int64)

            >>> binomial_coefficients(-1)
            array([], dtype=int64)

            >>> binomial_coefficients(0)
            array([1])

            >>> binomial_coefficients(1)
            array([1, 1])

            >>> binomial_coefficients(2)
            array([1, 2, 1])

            >>> binomial_coefficients(4)
            array([1, 4, 6, 4, 1])

            >>> binomial_coefficients(4.0)
            array([1, 4, 6, 4, 1])
    """

    # Must be integer
    n = int(n)

    # Below -1 is all the same.
    if n < -1:
        n = -1

    # Get enough repeats of n to get each coefficent
    ns = numpy.repeat(n, n + 1)
    # Get all relevant k's
    ks = numpy.arange(n + 1)

    # Get all the coefficents in order
    cs = scipy.misc.comb(ns, ks)
    cs = numpy.around(cs)
    cs = cs.astype(int)

    return(cs)


@prof.log_call(trace_logger)
def line_filter(shape, dim=-1):
    """
        Creates a boolean array mask for a line. This mask has size for the
        length of the line and number of empty lines beside it in any
        orthogonal direction. The mask has dimensions equal to ndims and the
        line is placed along dimension ``dim``.

        Args:
            shape(tuple of ints):   the distance from the center of the filter
                                    to the nearest edge for each dimension.

            dim(int):               the dimension to put the line along.

        Returns:
            (numpy.ndarray):        a boolean array to use as the filter.

        Examples:
            >>> line_filter((1,1))
            array([[False, False, False],
                   [ True,  True,  True],
                   [False, False, False]], dtype=bool)

            >>> line_filter((1,1), dim = -1)
            array([[False, False, False],
                   [ True,  True,  True],
                   [False, False, False]], dtype=bool)

            >>> line_filter((1,1), dim = 1)
            array([[False, False, False],
                   [ True,  True,  True],
                   [False, False, False]], dtype=bool)

            >>> line_filter((2,1))
            array([[False, False, False],
                   [False, False, False],
                   [ True,  True,  True],
                   [False, False, False],
                   [False, False, False]], dtype=bool)

            >>> line_filter((1, 1, 1))
            array([[[False, False, False],
                    [False, False, False],
                    [False, False, False]],
            <BLANKLINE>
                   [[False, False, False],
                    [ True,  True,  True],
                    [False, False, False]],
            <BLANKLINE>
                   [[False, False, False],
                    [False, False, False],
                    [False, False, False]]], dtype=bool)
    """

    line = numpy.zeros([(2*_+1) for _ in shape], dtype=bool)

    line_loc = list(shape)
    line_loc[dim] = slice(None)
    line[line_loc] = 1

    return(line)


@prof.log_call(trace_logger)
def symmetric_line_filter(size, ndims=2, dim=-1):
    """
        Creates a boolean array mask for a line. This mask has size for the
        length of the line and number of empty lines beside it in any
        orthogonal direction. The mask has dimensions equal to ndims and the
        line is placed along dimension dim.

        Args:
            size(int):          the distance from the center of the filter to
                                the nearest edge.

            ndims(int):         the number of dimensions for the filter.

            dim(int):           the dimension to put the line along.

        Returns:
            (numpy.ndarray):    a boolean array to use as the filter.

        Examples:
            >>> symmetric_line_filter(1)
            array([[False, False, False],
                   [ True,  True,  True],
                   [False, False, False]], dtype=bool)

            >>> symmetric_line_filter(1, ndims = 2, dim = -1)
            array([[False, False, False],
                   [ True,  True,  True],
                   [False, False, False]], dtype=bool)

            >>> symmetric_line_filter(1, ndims = 2, dim = 1)
            array([[False, False, False],
                   [ True,  True,  True],
                   [False, False, False]], dtype=bool)

            >>> symmetric_line_filter(1, ndims = 3)
            array([[[False, False, False],
                    [False, False, False],
                    [False, False, False]],
            <BLANKLINE>
                   [[False, False, False],
                    [ True,  True,  True],
                    [False, False, False]],
            <BLANKLINE>
                   [[False, False, False],
                    [False, False, False],
                    [False, False, False]]], dtype=bool)
    """

    assert (size > 0)
    assert (ndims > 0)
    assert (-ndims <= dim < ndims)

    line = numpy.zeros(ndims * (2*size+1, ), dtype=bool)

    line_loc = ndims * [size]
    line_loc[dim] = slice(None)
    line[line_loc] = 1

    return(line)


@prof.log_call(trace_logger)
def tagging_reorder_array(new_array,
                          from_axis_order="tzyxc",
                          to_axis_order="tzyxc",
                          to_copy=False):
    """
        Transforms one axis ordering to another giving a view of the array
        (unless otherwise specified).

        Args:
            new_array(numpy.ndarray):                   the array to reorder

            from_axis_order(str or list of str):        current labeled axis
                                                        order.

            to_axis_order(str or list of str):          desired labeled axis
                                                        order

            to_copy(bool):                              whether to return a
                                                        view or a copy

        Returns:
            (numpy.ndarray):                            an array with the axis
                                                        order specified (view).

        Examples:
            >>> tagging_reorder_array(numpy.ones((1,2,3,4,5))).shape
            (1, 2, 3, 4, 5)

            >>> tagging_reorder_array(
            ...     numpy.ones((1,2,3,4,5)),
            ...     from_axis_order = "tzyxc"
            ... ).shape
            (1, 2, 3, 4, 5)

            >>> tagging_reorder_array(
            ...     numpy.ones((1,2,3,4,5)),
            ...     to_axis_order = "tzyxc"
            ... ).shape
            (1, 2, 3, 4, 5)

            >>> tagging_reorder_array(
            ...     numpy.ones((1,2,3,4,5)),
            ...     from_axis_order = "tzyxc",
            ...     to_axis_order = "tzyxc"
            ... ).shape
            (1, 2, 3, 4, 5)

            >>> tagging_reorder_array(
            ...     numpy.ones((1,2,3,4,5)),
            ...     from_axis_order = "txyzc"
            ... ).shape
            (1, 4, 3, 2, 5)

            >>> tagging_reorder_array(
            ...     numpy.ones((1,2,3,4,5)),
            ...     from_axis_order = "ctxyz"
            ... ).shape
            (2, 5, 4, 3, 1)

            >>> tagging_reorder_array(
            ...     numpy.ones((1,2,3,4,5)),
            ...     to_axis_order = "txyzc"
            ... ).shape
            (1, 4, 3, 2, 5)

            >>> tagging_reorder_array(
            ...     numpy.ones((1,2,3,4,5)),
            ...     from_axis_order = ["c","t","x","y","z"]
            ... ).shape
            (2, 5, 4, 3, 1)

            >>> tagging_reorder_array(
            ...     numpy.ones((1,2,3,4,5)),
            ...     to_axis_order = ["t","x","y","z","c"]
            ... ).shape
            (1, 4, 3, 2, 5)
    """

    from_axis_order = "".join(from_axis_order)
    to_axis_order = "".join(to_axis_order)

    if (from_axis_order != to_axis_order):
        # Change view to the specified one
        new_array = vigra.taggedView(new_array, from_axis_order)
        # Reorder to the user specified one
        new_array = new_array.withAxes(*to_axis_order)
        # Dump the VIGRA array as we do not care
        new_array = new_array.view(numpy.ndarray)

    if to_copy:
        new_array = new_array.copy()

    return(new_array)
