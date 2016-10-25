"""
The module ``iters`` provides support working with and creating generators.

===============================================================================
Overview
===============================================================================
The module ``iters`` provides a variety of functions targeted at working with
or creating generators. It also borrows some functions from ``kenjutsu``
targeted at working with slices and tuples of slices. However, this borrowed
API should be considered deprecated and is subject to removal.

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 17, 2014 13:43:56 EDT$"


import itertools
import math
import warnings

import numpy

import yail
import yail.core

from kenjutsu.kenjutsu import *

# Need in order to have logging information no matter what.
from nanshe.util import prof

# Import Python 2/3 compatibility functions.
from yail.core import (
    range as irange,
    map as imap,
    zip as izip,
    zip_longest as izip_longest,
)

# Import replacement functions for compatibility.
from yail.core import (
    accumulate as cumulative_generator,
    cycles as cycle_generator,
    disperse,
    duplicate as repeat_generator,
    indices as index_generator,
    subrange,
    sliding_window,
    sliding_window_filled,
)


# Get the logger
trace_logger = prof.getTraceLogger(__name__)


@prof.log_call(trace_logger)
def index_enumerator(*sizes):
    """
        Takes an argument list of sizes and iterates through them from 0 up to
        (but not including) each size (i.e. like index_generator). However,
        also included is an index corresponding to how many elements have been
        seen.

        Args:
            *sizes(int):            an argument list of ints for the max sizes
                                    in each index.

        Returns:
            chain_gen(generator):   a generator over every possible coordinated


        Examples:
            >>> index_enumerator(0) #doctest: +ELLIPSIS
            <enumerate object at 0x...>

            >>> list(index_enumerator(0))
            []

            >>> list(index_enumerator(0, 2))
            []

            >>> list(index_enumerator(2, 0))
            []

            >>> list(index_enumerator(2, 1))
            [(0, (0, 0)), (1, (1, 0))]

            >>> list(index_enumerator(1, 2))
            [(0, (0, 0)), (1, (0, 1))]

            >>> list(index_enumerator(3, 2))
            [(0, (0, 0)), (1, (0, 1)), (2, (1, 0)), (3, (1, 1)), (4, (2, 0)), (5, (2, 1))]
    """

    return(enumerate(index_generator(*sizes)))


@prof.log_call(trace_logger)
def list_indices_to_index_array(list_indices):
    """
        Converts a list of tuple indices to numpy index array.

        Args:
            list_indices(list):    a list of indices corresponding to some
                                   array object

        Returns:
            chain_gen(tuple):      a tuple containing a numpy array for each
                                   index

        Examples:
            >>> list_indices_to_index_array([])
            ()

            >>> list_indices_to_index_array([(1,2)])
            (array([1]), array([2]))

            >>> list_indices_to_index_array([(1, 2), (5, 7), (33, 2)])
            (array([ 1,  5, 33]), array([2, 7, 2]))
    """

    # Combines the indices so that one dimension is represented by each list.
    # Then converts this to a tuple numpy.ndarrays.
    return(tuple(numpy.array(list(izip(*list_indices)))))


@prof.log_call(trace_logger)
def list_indices_to_numpy_bool_array(list_indices, shape):
    """
        Much like list_indices_to_index_array except that it constructs
        numpy.ndarray with dtype of bool. All indices in list_indices are set
        to True in the numpy.ndarray. The rest are False by default.

        Args:
            list_indices(list):      a list of indices corresponding to some
                                     numpy.ndarray object

            shape(tuple):            a tuple used to set the shape of the
                                     numpy.ndarray to return

        Returns:
            result(numpy.ndarray):   a numpy.ndarray with dtype bool (True for
                                     indices in list_indices and False
                                     otherwise).

        Examples:
            >>> list_indices_to_numpy_bool_array([], ())
            array(False, dtype=bool)

            >>> list_indices_to_numpy_bool_array([], (0))
            array([], dtype=bool)

            >>> list_indices_to_numpy_bool_array([], (0,0))
            array([], shape=(0, 0), dtype=bool)

            >>> list_indices_to_numpy_bool_array([], shape=(1))
            array([False], dtype=bool)

            >>> list_indices_to_numpy_bool_array([(0,0)], (1,1))
            array([[ True]], dtype=bool)

            >>> list_indices_to_numpy_bool_array(
            ...     [(2,3), (0,0), (0,2), (1,1)], (3,4)
            ... )
            array([[ True, False,  True, False],
                   [False,  True, False, False],
                   [False, False, False,  True]], dtype=bool)
    """

    # Constructs the numpy.ndarray with False everywhere
    result = numpy.zeros(shape, dtype=bool)

    # Gets the index array
    # Done first to make sure that if list_indices is this [], or this (), or this [()]
    # will be converted to this ().
    index_array = list_indices_to_index_array(list_indices)

    # Sets the given indices to True
    if index_array != ():
        result[index_array] = True

    return(result)


@prof.log_call(trace_logger)
def iter_with_skip_indices(a_iter, to_skip=None):
    """
        Behaves as a normal iterator except allows for skipping arbitrary
        values, as well. These values to be skipped should be specified by
        their indices using some iterable.

        Args:
            a_iter(iter):          an iterator that will skip some values

            to_skip(iter):         some form of iterable or list of indices to
                                   skip (can be a single value as well).

        Returns:
            (generator object):    a generator that skips some values with
                                   indices in to_skip.

        Examples:
            >>> iter_with_skip_indices(irange(10)) #doctest: +ELLIPSIS
            <generator object iter_with_skip_indices at 0x...>

            >>> list(iter_with_skip_indices(irange(10)))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_indices(irange(10), to_skip = 2))
            [0, 1, 3, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_indices(irange(1, 10), to_skip = 2))
            [1, 2, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_indices(irange(10), to_skip = [2, 7]))
            [0, 1, 3, 4, 5, 6, 8, 9]

            >>> list(iter_with_skip_indices(irange(10), to_skip = [0]))
            [1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_indices(irange(1, 10), to_skip = [0]))
            [2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_indices(irange(10), to_skip = [9]))
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """

    full = iter(a_iter)
    full_enum = enumerate(full)

    if to_skip is None:
        to_skip = iter([])
    else:
        try:
            to_skip = iter(sorted(set(to_skip)))
        except TypeError:
            to_skip = iter([to_skip])

    next_to_skip = next(to_skip, None)

    for i, each in full_enum:
        if i != next_to_skip:
            yield(each)
        else:
            next_to_skip = next(to_skip, None)


@prof.log_call(trace_logger)
def iter_with_skip_values(a_iter, to_skip=None):
    """
        Behaves as a normal iterator except allows for skipping arbitrary
        values, as well. These values to be skipped should be specified by
        their indices using some iterable.

        Args:
            a_iter(iter):          an iterator that will skip some values

            to_skip(iter):         some form of iterable or list of indices to
                                   skip (can be a single value as well).

        Returns:
            (generator object):    a generator that skips some values with
                                   indices in to_skip.

        Examples:
            >>> iter_with_skip_values(irange(10)) #doctest: +ELLIPSIS
            <generator object iter_with_skip_values at 0x...>

            >>> list(iter_with_skip_values(irange(10)))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_values(irange(10)))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_values(irange(1, 10)))
            [1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_values(irange(0, 10, 2)))
            [0, 2, 4, 6, 8]

            >>> list(iter_with_skip_values(irange(1, 10, 2)))
            [1, 3, 5, 7, 9]

            >>> list(iter_with_skip_values(irange(10), to_skip = 2))
            [0, 1, 3, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_values(irange(1, 10), to_skip = 2))
            [1, 3, 4, 5, 6, 7, 8, 9]

            >>> list(iter_with_skip_values(irange(0, 10, 2), to_skip = [2,6]))
            [0, 4, 8]
    """

    full = iter(a_iter)

    if to_skip is None:
        to_skip = []
    else:
        try:
            to_skip = sorted(set(to_skip))
        except TypeError:
            to_skip = [to_skip]

    for each in full:
        if each not in to_skip:
            yield(each)


@prof.log_call(trace_logger)
def xrange_with_skip(start, stop=None, step=None, to_skip=None):
    """
        Behaves as irange does except allows for skipping arbitrary values, as
        well. These values to be skipped should be specified using some
        iterable.

        Args:
            start(int):            start for irange or if stop is not specified
                                   this will be stop.

            stop(int):             stop for irange.

            stop(int):             step for irange.

            to_skip(iter):         some form of iterable or list of elements to
                                   skip (can be a single value as well).

        Returns:
            (generator object):    an irange-like generator that skips some
                                   values.

        Examples:
            >>> xrange_with_skip(10) #doctest: +ELLIPSIS
            <generator object xrange_with_skip at 0x...>

            >>> list(xrange_with_skip(10))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(xrange_with_skip(0, 10))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(xrange_with_skip(1, 10))
            [1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(xrange_with_skip(0, 10, 2))
            [0, 2, 4, 6, 8]

            >>> list(xrange_with_skip(1, 10, 2))
            [1, 3, 5, 7, 9]

            >>> list(xrange_with_skip(10, to_skip = 2))
            [0, 1, 3, 4, 5, 6, 7, 8, 9]

            >>> list(xrange_with_skip(10, to_skip = [2, 7]))
            [0, 1, 3, 4, 5, 6, 8, 9]

            >>> list(xrange_with_skip(10, to_skip = [0]))
            [1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(xrange_with_skip(1, 10, to_skip = [0]))
            [1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> list(xrange_with_skip(10, to_skip = [9]))
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """

    full = None

    if (stop is None):
        full = iter(irange(start))
    elif (step is None):
        full = iter(irange(start, stop))
    else:
        full = iter(irange(start, stop, step))

    if to_skip is None:
        to_skip = iter([])
    else:
        try:
            to_skip = iter(sorted(set(to_skip)))
        except TypeError:
            to_skip = iter([to_skip])

    next_to_skip = next(to_skip, None)

    for each in full:
        if each != next_to_skip:
            yield(each)
        else:
            next_to_skip = next(to_skip, None)


splitting_xrange = lambda a, *args: disperse(irange(a, *args))


@prof.log_call(trace_logger)
def reverse_each_element(new_iter):
    """
        Takes each element yielded by new_iter and reverses it using reversed.

        Args:
            new_iter(iter):        an iterator or something that can be turned
                                   into an iterator.

        Returns:
            (generator object):    an iterator over the reversed elements.

        Examples:
            >>> reverse_each_element(
            ...     zip(irange(5, 11), irange(5))
            ... ) #doctest: +ELLIPSIS
            <generator object reverse_each_element at 0x...>

            >>> list(reverse_each_element(zip(irange(5, 11), irange(5))))
            [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

            >>> list(reverse_each_element(iter([[5]])))
            [[5]]

            >>> list(reverse_each_element(iter([[5,2,3], [1, 7, 9]])))
            [[3, 2, 5], [9, 7, 1]]
    """

    new_iter = iter(new_iter)

    for each in new_iter:
        yield (type(each)(reversed(each)))


@prof.log_call(trace_logger)
def lagged_generators(new_iter, n=2):
    """
        Creates a tuple of generators with each next generator one step ahead
        of the previous generator.

        Args:
            new_iter(iter):                 an iterator or something that can
                                            be turned into an iterator

            n(int):                         number of generators to create as
                                            lagged

        Returns:
            (tuple of generator objects):   a tuple of iterators with each one
                                            step in front of the others.

        Examples:
            >>> lagged_generators(irange(5), 1) #doctest: +ELLIPSIS
            (<itertools... object at 0x...>,)

            >>> list(izip(*lagged_generators(irange(5), 1)))
            [(0,), (1,), (2,), (3,), (4,)]

            >>> list(izip(*lagged_generators(irange(5), 2)))
            [(0, 1), (1, 2), (2, 3), (3, 4)]

            >>> list(izip(*lagged_generators(irange(5))))
            [(0, 1), (1, 2), (2, 3), (3, 4)]

            >>> list(izip(*lagged_generators(irange(5), 3)))
            [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

            >>> list(izip_longest(*lagged_generators(irange(5))))
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, None)]

            >>> list(izip_longest(*lagged_generators(irange(5), 3)))
            [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, None), (4, None, None)]
    """

    warnings.warn(
        "Please use `lagged_generators_zipped` instead.", DeprecationWarning
    )

    assert (n >= 0), \
        "Only a positive semi-definite number of generators can be created."

    # Where they will be stored
    all_iters = tuple()

    # If some positive definite number of generators is requested, then fill
    # the list.
    if n > 0:
        # Convert to the same type
        next_iter = itertools.tee(new_iter, 1)[0]
        for i in irange(1, n):
            # Duplicate the iterator
            prev_iter, next_iter = itertools.tee(next_iter, 2)

            # Store the copy of the old one
            all_iters += (prev_iter,)

            # Attempt to advance the next one
            # If it fails, create an empty iterator.
            try:
                next(next_iter)
            except StopIteration:
                next_iter = itertools.tee([], 1)[0]

        # Add the last one. If n == 1, the last one is the only one.
        all_iters += (next_iter,)

    return(all_iters)


lagged_generators_zipped = lambda new_iter, \
                                  n=2, \
                                  longest=False, \
                                  fillvalue=None: \
    sliding_window_filled(new_iter, n, False, longest, fillvalue)


@prof.log_call(trace_logger)
def filled_stringify_numbers(new_iter, include_numbers=False):
    """
        Like enumerate except it also returns a string with the number from
        enumeration with left padding by zero.

        Args:
            new_iter(iter):        an iterator to use for enumeration over.

        Returns:
            (generator object):    an iterator over the reversed elements.

        Examples:
            >>> filled_stringify_numbers([5, 7]) #doctest: +ELLIPSIS
            <generator object filled_stringify_numbers at 0x...>

            >>> list(filled_stringify_numbers([]))
            []

            >>> list(filled_stringify_numbers(irange(5)))
            ['0', '1', '2', '3', '4']

            >>> list(filled_stringify_numbers([5]))
            ['5']

            >>> list(filled_stringify_numbers([5, 7]))
            ['5', '7']

            >>> list(filled_stringify_numbers([5, 7, 11]))
            ['05', '07', '11']

            >>> list(
            ...     filled_stringify_numbers([5, 7, 11], include_numbers=True)
            ... )
            [(5, '05'), (7, '07'), (11, '11')]

            >>> list(
            ...     filled_stringify_numbers(
            ...         iter([5, 7, 11]),
            ...         include_numbers=True
            ...     )
            ... )
            [(5, '05'), (7, '07'), (11, '11')]
    """

    new_list = new_iter
    new_list_len = None
    new_list_max = None
    try:
        new_list_len = len(new_list)
        new_list_max = max(new_list) if new_list_len else None
    except TypeError:
        new_list = list(new_list)
        new_list_len = len(new_list)
        new_list_max = max(new_list) if new_list_len else None

    if new_list_len:
        if new_list_max:
            digits = int(numpy.floor(numpy.log10(new_list_max))) + 1
        else:
            digits = 1

    if include_numbers:
        for each in new_list:
            yield((each, str(each).zfill(digits)))
    else:
        for each in new_list:
            yield(str(each).zfill(digits))


@prof.log_call(trace_logger)
def filled_stringify_xrange(new_iter):
    """
        Takes each element yielded by new_iter and reverses it using reversed.

        Args:
            new_iter(iter):        an iterator to use for enumeration over.

        Returns:
            (generator object):    an iterator over the reversed elements.

        Examples:
            >>> filled_stringify_xrange([5, 7]) #doctest: +ELLIPSIS
            <generator object filled_stringify_xrange at 0x...>

            >>> list(filled_stringify_xrange([]))
            []

            >>> list(filled_stringify_xrange(irange(5)))
            [(0, '0'), (1, '1'), (2, '2'), (3, '3'), (4, '4')]

            >>> list(filled_stringify_xrange(irange(2, 5)))
            [(0, '0'), (1, '1'), (2, '2')]

            >>> list(filled_stringify_xrange([5]))
            [(0, '0')]

            >>> list(filled_stringify_xrange([5, 7]))
            [(0, '0'), (1, '1')]

            >>> list(filled_stringify_xrange(iter([5, 7])))
            [(0, '0'), (1, '1')]

            >>> list(filled_stringify_xrange(list(irange(11))))
            [(0, '00'), (1, '01'), (2, '02'), (3, '03'), (4, '04'), (5, '05'), (6, '06'), (7, '07'), (8, '08'), (9, '09'), (10, '10')]
    """

    new_list = new_iter
    new_list_index_gen = None
    try:
        new_list_index_gen = irange(len(new_list))
    except TypeError:
        new_list = list(new_list)
        new_list_index_gen = irange(len(new_list))

    new_list_index_gen_stringified = filled_stringify_numbers(
        new_list_index_gen, include_numbers=True
    )

    for i, i_str in new_list_index_gen_stringified:
        yield ((i, i_str))


@prof.log_call(trace_logger)
def filled_stringify_enumerate(new_iter):
    """
        Takes each element yielded by new_iter and reverses it using reversed.

        Args:
            new_iter(iter):        an iterator to use for enumeration over.

        Returns:
            (generator object):    an iterator over the reversed elements.

        Examples:
            >>> filled_stringify_enumerate([5, 7]) #doctest: +ELLIPSIS
            <generator object filled_stringify_enumerate at 0x...>

            >>> list(filled_stringify_enumerate([]))
            []

            >>> list(filled_stringify_enumerate(irange(5)))
            [(0, '0', 0), (1, '1', 1), (2, '2', 2), (3, '3', 3), (4, '4', 4)]

            >>> list(filled_stringify_enumerate(irange(2, 5)))
            [(0, '0', 2), (1, '1', 3), (2, '2', 4)]

            >>> list(filled_stringify_enumerate([5]))
            [(0, '0', 5)]

            >>> list(filled_stringify_enumerate([5, 7]))
            [(0, '0', 5), (1, '1', 7)]

            >>> list(filled_stringify_enumerate(iter([5, 7])))
            [(0, '0', 5), (1, '1', 7)]

            >>> list(filled_stringify_enumerate(range(11)))
            [(0, '00', 0), (1, '01', 1), (2, '02', 2), (3, '03', 3), (4, '04', 4), (5, '05', 5), (6, '06', 6), (7, '07', 7), (8, '08', 8), (9, '09', 9), (10, '10', 10)]
    """

    new_list = new_iter
    new_list_index_gen = None
    try:
        new_list_index_gen = irange(len(new_list))
    except TypeError:
        new_list = list(new_list)
        new_list_index_gen = irange(len(new_list))

    new_list_index_gen_stringified = filled_stringify_numbers(
        new_list_index_gen, include_numbers=True
    )

    for (i, i_str), each in izip(new_list_index_gen_stringified, new_list):
        yield ((i, i_str, each))
