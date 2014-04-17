import itertools
import numpy

def index_generator(*sizes):
    """
        Takes an argument list of sizes and iterates through them from 0 up to (but including) each size.
        
        Args:
            *sizes(int):            an argument list of ints for the max sizes in each index.
        
        Returns:
            chain_gen(generator):   a generator over every possible coordinated
    """
    
    gens = [xrange(_) for _ in sizes]
    
    chain_gen = itertools.product(*gens)
    
    return(chain_gen)


def list_indices_to_index_array(list_indices):
    """
        Converts a list of tuple indices to numpy index array.
        
        Args:
            list_indices(list):    a list of indices corresponding to some array object
        
        Returns:
            chain_gen(tuple):       a tuple containing a numpy array in for each index
    """
    
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
    """
    
    result = numpy.zeros(shape, dtype = bool)
    
    result[list_indices_to_index_array(list_indices)] = True
    
    return(result)