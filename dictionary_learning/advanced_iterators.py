import itertools
import numpy

def index_generator(*sizes):
    """
        Takes an argument list of sizes and iterates through them from 0 up to (but including) each size.
        
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
    
    # Combines the generators to a single generator of indicies that go throughout sizes
    chain_gen = itertools.product(*gens)
    
    return(chain_gen)


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
            yield(each)
        else:
            next_to_skip = next(to_skip, None)
