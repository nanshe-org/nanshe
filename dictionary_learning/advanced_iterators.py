import itertools
import numpy

def index_generator(*sizes):
    gens = [xrange(_) for _ in sizes]
    
    chain_gen = itertools.product(*gens)
    
    return(chain_gen)


def list_indices_to_index_array(list_indices):
    return(tuple(numpy.array(zip(*list_indices))))


def list_indices_to_numpy_bool_array(list_indices, shape):
    result = numpy.zeros(shape, dtype = bool)
    
    result[list_indices_to_index_array(list_indices)] = True
    
    return(result)