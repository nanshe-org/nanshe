import itertools
import numpy
import math


# Need in order to have logging information no matter what.
import debugging_tools

# Get the logger
logger = debugging_tools.logging.getLogger(__name__)


@debugging_tools.log_call(logger)
def index_generator(*sizes):
    """
        Takes an argument list of sizes and iterates through them from 0 up to (but not including) each size.
        
        Args:
            *sizes(int):            an argument list of ints for the max sizes in each index.
        
        Returns:
            chain_gen(generator):   a generator over every possible coordinated
        
        
        Examples:
            >>> index_generator(0) #doctest: +ELLIPSIS
            <itertools.product object at 0x...>
            
            >>> list(index_generator(0))
            []
            
            >>> list(index_generator(0, 2))
            []
            
            >>> list(index_generator(2, 0))
            []
            
            >>> list(index_generator(2, 1))
            [(0, 0), (1, 0)]
            
            >>> list(index_generator(1, 2))
            [(0, 0), (0, 1)]
            
            >>> list(index_generator(3, 2))
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    """

    # Creates a list of xrange generator objects over each respective dimension of sizes
    gens = [xrange(_) for _ in sizes]

    # Combines the generators to a single generator of indices that go throughout sizes
    chain_gen = itertools.product(*gens)

    return(chain_gen)


@debugging_tools.log_call(logger)
def index_enumerator(*sizes):
    """
        Takes an argument list of sizes and iterates through them from 0 up to (but not including) each size (i.e. like
        index_generator). However, also included is an index corresponding to how many elements have been seen.

        Args:
            *sizes(int):            an argument list of ints for the max sizes in each index.

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

    # Creates a list of xrange generator objects over each respective dimension of sizes
    gens = [xrange(_) for _ in sizes]

    # Combines the generators to a single generator of indices that go throughout sizes
    chain_gen = enumerate(itertools.product(*gens))

    return(chain_gen)


@debugging_tools.log_call(logger)
def list_indices_to_index_array(list_indices):
    """
        Converts a list of tuple indices to numpy index array.
        
        Args:
            list_indices(list):    a list of indices corresponding to some array object
        
        Returns:
            chain_gen(tuple):       a tuple containing a numpy array in for each index
            
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
    return(tuple(numpy.array(zip(*list_indices))))


@debugging_tools.log_call(logger)
def list_indices_to_numpy_bool_array(list_indices, shape):
    """
        Much like list_indices_to_index_array except that it constructs a numpy.ndarray with dtype of bool.
        All indices in list_indices are set to True in the numpy.ndarray. The rest are False by default.
        
        Args:
            list_indices(list):      a list of indices corresponding to some numpy.ndarray object
            shape(tuple):            a tuple used to set the shape of the numpy.ndarray to return 
        
        Returns:
            result(numpy.ndarray):   a numpy.ndarray with dtype bool (True for indices in list_indices and False otherwise).
        
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
            
            >>> list_indices_to_numpy_bool_array([(2,3), (0,0), (0,2), (1,1)], (3,4))
            array([[ True, False,  True, False],
                   [False,  True, False, False],
                   [False, False, False,  True]], dtype=bool)
            
    """

    # Constructs the numpy.ndarray with False everywhere
    result = numpy.zeros(shape, dtype = bool)

    # Gets the index array
    # Done first to make sure that if list_indices is this [], or this (), or this [()]
    # will be converted to this ().
    index_array = list_indices_to_index_array(list_indices)

    # Sets the given indices to True
    if index_array != ():
        result[index_array] = True

    return(result)


@debugging_tools.log_call(logger)
def xrange_with_skip(start, stop = None, step = None, to_skip = None):
    """
        Behaves as xrange does except allows for skipping arbitrary values as well.
        These values to be skipped should be specified using some iterable.
        
        Args:
            start(int):            start for xrange or if stop is not specified this will be stop.
            stop(int):             stop for xrange.
            stop(int):             step for xrange.
            to_skip(iter):         some form of iterable or list of elements to skip (can be a single value as well).
        
        Returns:
            (generator object):    an xrange-like generator that skips some values.
        
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
        full = iter(xrange(start))
    elif (step is None):
        full = iter(xrange(start, stop))
    else:
        full = iter(xrange(start, stop, step))

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
            yield (each)
        else:
            next_to_skip = next(to_skip, None)


@debugging_tools.log_call(logger)
def cumulative_generator(new_op, new_iter):
    """
        Takes each value from new_iter and applies new_op to it with the result
        of previous values.
        
        For instance cumulative_generator(op.mul, xrange(1,5)) will return all
        factorials up to and including the factorial of 4 (24).
        
        Args:
            new_op(callabel):      something that can be called on two values and return a result with a type that is a
                                   permissible argument.

            new_iter(iter):        an iterator or something that can be turned into an iterator.
        
        Returns:
            (generator object):    an iterator over the intermediate results.
        
        Examples:
            >>> import operator; cumulative_generator(operator.add, 10) #doctest: +ELLIPSIS
            <generator object cumulative_generator at 0x...>
            
            >>> import operator; list(cumulative_generator(operator.add, xrange(1,5)))
            [1, 3, 6, 10]

            >>> import operator; list(cumulative_generator(operator.add, xrange(5)))
            [0, 1, 3, 6, 10]

            >>> import operator; list(cumulative_generator(operator.mul, xrange(5)))
            [0, 0, 0, 0, 0]

            >>> import operator; list(cumulative_generator(operator.mul, xrange(1,5)))
            [1, 2, 6, 24]
        
    """

    new_iter = iter(new_iter)

    cur = next(new_iter)
    yield (cur)

    for each in new_iter:
        cur = new_op(cur, each)
        yield (cur)


@debugging_tools.log_call(logger)
def reverse_each_element(new_iter):
    """
        Takes each element yielded by new_iter and reverses it using reversed.
        
        Args:
            new_iter(iter):        an iterator or something that can be turned into an iterator.
        
        Returns:
            (generator object):    an iterator over the reversed elements.
        
        Examples:
            >>> reverse_each_element(zip(xrange(5, 11), xrange(5))) #doctest: +ELLIPSIS
            <generator object reverse_each_element at 0x...>
            
            >>> list(reverse_each_element(zip(xrange(5, 11), xrange(5))))
            [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
            
            >>> list(reverse_each_element(iter([[5]])))
            [[5]]
            
            >>> list(reverse_each_element(iter([[5,2,3], [1, 7, 9]])))
            [[3, 2, 5], [9, 7, 1]]
    """

    new_iter = iter(new_iter)

    for each in new_iter:
        yield ( type(each)(reversed(each)) )


@debugging_tools.log_call(logger)
def lagged_generators(new_iter, n = 2):
    """
        Creates a tuple of generators with each next generator one step ahead of the previous generator.

        Args:
            new_iter(iter):                 an iterator or something that can be turned into an iterator
            n(int):                         number of generators to create as lagged

        Returns:
            (tuple of generator objects):   a tuple of iterators with each one step in front of the others.

        Examples:
            >>> lagged_generators(xrange(5), 1) #doctest: +ELLIPSIS
            (<itertools.tee object at 0x...>,)

            >>> zip(*lagged_generators(xrange(5), 1))
            [(0,), (1,), (2,), (3,), (4,)]

            >>> zip(*lagged_generators(xrange(5), 2))
            [(0, 1), (1, 2), (2, 3), (3, 4)]

            >>> zip(*lagged_generators(xrange(5)))
            [(0, 1), (1, 2), (2, 3), (3, 4)]

            >>> zip(*lagged_generators(xrange(5), 3))
            [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

            >>> list(itertools.izip_longest(*lagged_generators(xrange(5))))
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, None)]

            >>> list(itertools.izip_longest(*lagged_generators(xrange(5), 3)))
            [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, None), (4, None, None)]
    """

    # Only a positive semi-definite number of generators can be created
    assert(n >= 0)

    # Where they will be stored
    all_iters = tuple()

    # If some positive definite number of generators is requested, then fill the list.
    if n > 0:
        # Convert to the same type
        next_iter = itertools.tee(new_iter, 1)[0]
        for i in xrange(1, n):
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


@debugging_tools.log_call(logger)
def filled_stringify_numbers(new_iter, include_numbers = False):
    """
        Like enumerate except it also returns a string with the number from enumeration with left padding by zero.
        
        Args:
            new_iter(iter):        a list or an iterator to use for enumeration over.
        
        Returns:
            (generator object):    an iterator over the reversed elements.
        
        Examples:
            >>> filled_stringify_numbers([5, 7]) #doctest: +ELLIPSIS
            <generator object filled_stringify_numbers at 0x...>

            >>> list(filled_stringify_numbers([]))
            []

            >>> list(filled_stringify_numbers([5]))
            ['5']

            >>> list(filled_stringify_numbers([5, 7]))
            ['5', '7']

            >>> list(filled_stringify_numbers([5, 7, 11]))
            ['05', '07', '11']

            >>> list(filled_stringify_numbers([5, 7, 11], include_numbers = True))
            [(5, '05'), (7, '07'), (11, '11')]
    """

    new_list = new_iter
    if not isinstance(new_list, list):
        new_list = list(new_list)

    new_array = numpy.array(new_list)

    if len(new_array):
        new_array_max = new_array.max()
        if new_array_max:
            digits = int(numpy.floor(numpy.log10(new_array.max()))) + 1
        else:
            digits = 1

    if include_numbers:
        for each in new_array:
            yield( (each, str(each).zfill(digits)) )
    else:
        for each in new_array:
            yield( str(each).zfill(digits) )


@debugging_tools.log_call(logger)
def filled_stringify_enumerate(new_iter):
    """
        Takes each element yielded by new_iter and reverses it using reversed.

        Args:
            new_iter(iter):        a list or an iterator to use for enumeration over.

        Returns:
            (generator object):    an iterator over the reversed elements.

        Examples:
            >>> filled_stringify_enumerate([5, 7]) #doctest: +ELLIPSIS
            <generator object filled_stringify_enumerate at 0x...>

            >>> list(filled_stringify_enumerate([]))
            []

            >>> list(filled_stringify_enumerate([5]))
            [(0, '0', 5)]

            >>> list(filled_stringify_enumerate([5, 7]))
            [(0, '0', 5), (1, '1', 7)]

            >>> list(filled_stringify_enumerate(range(11)))
            [(0, '00', 0), (1, '01', 1), (2, '02', 2), (3, '03', 3), (4, '04', 4), (5, '05', 5), (6, '06', 6), (7, '07', 7), (8, '08', 8), (9, '09', 9), (10, '10', 10)]

    """

    new_list = new_iter
    if not isinstance(new_list, list):
        new_list = list(new_list)

    new_list_index_gen = xrange(len(new_list))
    new_list_index_gen_stringified = filled_stringify_numbers(new_list_index_gen, include_numbers = True)

    for (i, i_str), each in itertools.izip(new_list_index_gen_stringified, new_list):
        yield ( (i, i_str, each) )


@debugging_tools.log_call(logger)
def reformat_slice(a_slice, a_length = None):
    """
        Takes a slice and reformats it to fill in as many undefined values as possible.

        Args:
            a_slice(slice):        a slice to reformat.
            a_length(int):         a length to fill for stopping if not provided.

        Returns:
            (slice):               a new slice with as many values filled in as possible.

        Examples:
            >>> reformat_slice(slice(None))
            slice(0, None, 1)

            >>> reformat_slice(slice(None), 10)
            slice(0, 10, 1)

            >>> reformat_slice(slice(2, None))
            slice(2, None, 1)

            >>> reformat_slice(slice(2, None), 10)
            slice(2, 10, 1)

            >>> reformat_slice(slice(2, None, None))
            slice(2, None, 1)

            >>> reformat_slice(slice(2, None, None), 10)
            slice(2, 10, 1)

            >>> range(10)[reformat_slice(slice(None))] == range(10)[:]
            True

            >>> range(10)[reformat_slice(slice(2, None))] == range(10)[2:]
            True

            >>> range(10)[reformat_slice(slice(2, 6))] == range(10)[2:6]
            True

            >>> range(10)[reformat_slice(slice(2, 6, 3))] == range(10)[2:6:3]
            True

            >>> range(10)[reformat_slice(slice(2, None, 3))] == range(10)[2::3]
            True

            >>> range(10)[reformat_slice(slice(None), 10)] == range(10)[:]
            True

            >>> range(10)[reformat_slice(slice(2, None), 10)] == range(10)[2:]
            True

            >>> range(10)[reformat_slice(slice(2, 6), 10)] == range(10)[2:6]
            True

            >>> range(10)[reformat_slice(slice(2, 6, 3), 10)] == range(10)[2:6:3]
            True

            >>> range(10)[reformat_slice(slice(2, None, 3), 10)] == range(10)[2::3]
            True
    """
    assert a_slice != None, "err"

    new_slice_stop = a_slice.stop
    if (new_slice_stop is None) or ((a_length is not None) and (new_slice_stop > a_length)):
        new_slice_stop = a_length

    new_slice_start = a_slice.start
    if new_slice_start is None:
        new_slice_start = 0

    new_slice_step = a_slice.step
    if new_slice_step is None:
        new_slice_step = 1

    new_slice = slice(new_slice_start, new_slice_stop, new_slice_step)

    return(new_slice)


@debugging_tools.log_call(logger)
def reformat_slices(slices, lengths = None):
    """
        Takes a tuple of slices and reformats them to fill in as many undefined values as possible.

        Args:
            slices(tuple(slice)):        a tuple of slices to reformat.
            lengths(tuple(int)):         a tuple of lengths to fill.

        Returns:
            (slice):                     a tuple of slices with all default values filled if possible.
    """

    new_lengths = lengths
    if new_lengths is None:
        new_lengths = [None] * len(slices)

    new_slices = list(slices)
    for i, each_length in enumerate(new_lengths):
        new_slices[i] = reformat_slice(new_slices[i], each_length)

    new_slices = tuple(new_slices)

    return(new_slices)


class UnknownSliceLengthException(Exception):
    """
        Raised if a slice does not have a known length.
    """

    pass


@debugging_tools.log_call(logger)
def len_slice(a_slice, a_length = None):
    """
        Determines how many elements a slice will contain.

        Raises:
            UnknownSliceLengthException: Will raise an exception if a_slice.stop and a_length is None.

        Args:
            a_slice(slice):        a slice to reformat.
            a_length(int):         a length to fill for stopping if not provided.

        Returns:
            (slice):               a new slice with as many values filled in as possible.

        Examples:
            >>> len_slice(slice(None)) #doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            UnknownSliceLengthException: ...

            >>> len_slice(slice(None), 10)
            10

            >>> len_slice(slice(None), 10) == len(range(10)[:])
            True

            >>> len_slice(slice(2, None), 10)
            8

            >>> len_slice(slice(2, None), 10) == len(range(10)[2:])
            True

            >>> len_slice(slice(2, None, None), 10)
            8

            >>> len_slice(slice(2, None, None), 10) == len(range(10)[2:])
            True

            >>> len_slice(slice(2, 6))
            4

            >>> len_slice(slice(2, 6), 1000)
            4

            >>> len_slice(slice(2, 6), 10) == len(range(10)[2:6])
            True

            >>> len_slice(slice(2, 6, 3))
            2

            >>> len_slice(slice(2, 6, 3), 10) == len(range(10)[2:6:3])
            True
    """

    new_slice = reformat_slice(a_slice, a_length)

    if new_slice.stop is None:
        raise UnknownSliceLengthException("Cannot determine slice length without a defined end point. The reformatted slice was " + repr(new_slice) + ".")

    new_slice_diff = float(new_slice.stop - new_slice.start)

    new_slice_size = int(math.ceil(new_slice_diff / new_slice.step))

    return(new_slice_size)


@debugging_tools.log_call(logger)
def len_slices(slices, lengths = None):
    """
        Takes a tuple of slices and reformats them to fill in as many undefined values as possible.

        Args:
            slices(tuple(slice)):        a tuple of slices to reformat.
            lengths(tuple(int)):         a tuple of lengths to fill.

        Returns:
            (slice):                     a tuple of slices with all default values filled if possible.
    """

    new_slices = reformat_slices(slices, lengths)

    lens = []

    for each_slice in new_slices:
        lens.append(len_slice(each_slice))

    lens = tuple(lens)

    return(lens)