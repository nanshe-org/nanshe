# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__="John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ ="$May 20, 2014 9:46:45 AM$"



import numpy
import operator



# Need in order to have logging information no matter what.
import advanced_debugging


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)


@advanced_debugging.log_call(logger)
def add_singleton_axis_pos(new_array, new_pos = 0):
    """
        Adds a singleton axis to the given position.
        Allows negative values for new_pos.
        Also, automatically bounds new_pos in an acceptable regime if it is not already.
        
        Args:
            new_array(numpy.ndarray):            array to add the singleton axis to.
            new_pos(int):                        position for the axis to be in the final array (defaults to zero ).
        
        Returns:
            (numpy.ndarray):                     a numpy array with the singleton axis added.
        
        Examples:
            >>> add_singleton_axis_pos(numpy.ones((2,3,4))).shape
            (1, 2, 3, 4)
            
            >>> add_singleton_axis_pos(numpy.ones((2,3,4)), 0).shape
            (1, 2, 3, 4)
            
            >>> add_singleton_axis_pos(numpy.ones((2,3,4)), new_pos = 0).shape
            (1, 2, 3, 4)
            
            >>> add_singleton_axis_pos(numpy.ones((2,3,4)), new_pos = 1).shape
            (2, 1, 3, 4)
            
            >>> add_singleton_axis_pos(numpy.ones((2,3,4)), new_pos = 2).shape
            (2, 3, 1, 4)
            
            >>> add_singleton_axis_pos(numpy.ones((2,3,4)), new_pos = 3).shape
            (2, 3, 4, 1)
            
            >>> add_singleton_axis_pos(numpy.ones((2,3,4)), new_pos = -1).shape
            (2, 3, 4, 1)
            
    """
    
    new_pos %= (new_array.ndim + 1)
    if new_pos < 0:
        new_pos += new_array.ndim + 1
        
    return( numpy.rollaxis(new_array[None], 0, new_pos + 1) )



@advanced_debugging.log_call(logger)
def add_singleton_axis_beginning(new_array):
    """
        Adds a singleton axis to the beginning of the array.
        
        Args:
            new_array(numpy.ndarray):            array to add the singleton axis to.
        
        Returns:
            (numpy.ndarray):                     a numpy array with the singleton axis added at the end.
        
        Examples:
            >>> add_singleton_axis_beginning(numpy.ones((2,3,4))).shape
            (1, 2, 3, 4)
            
            >>> add_singleton_axis_beginning(numpy.eye(3)).shape
            (1, 3, 3)
            
    """
    
    #return( new_array[None] )
    return( add_singleton_axis_pos(new_array, new_pos = 0) )



@advanced_debugging.log_call(logger)
def add_singleton_axis_end(new_array):
    """
        Adds a singleton axis to the end of the array.
        
        Args:
            new_array(numpy.ndarray):            array to add the singleton axis to.
        
        Returns:
            (numpy.ndarray):                     a numpy array with the singleton axis added at the end.
        
        Examples:
            >>> add_singleton_axis_end(numpy.ones((2,3,4))).shape
            (2, 3, 4, 1)
            
            >>> add_singleton_axis_end(numpy.eye(3)).shape
            (3, 3, 1)
            
    """
    
    #return( numpy.rollaxis(new_array[None], 0, new_array.ndim + 1) )
    return( add_singleton_axis_pos(new_array, new_pos = new_array.ndim) )



@advanced_debugging.log_call(logger)
def expand_view(new_array, reps_after = tuple(), reps_before = tuple()):
    """
        Behaves like NumPy tile except that it always returns a view and not a copy.
        Though, it differs in that additional dimensions are added for repetion as
        opposed to repeating in the same one. Also, it allows repetitions to be
        specified before or after unlike tile. Though, will behave identical to
        tile if the keyword is not specified.
        
        Uses strides to trick NumPy into providing a view.
        
        Args:
            new_array(numpy.ndarray):            array to tile.
            reps_after(tuple):                   repetitions dimension size to add before (if an int will turn into a tuple).
            reps_before(tuple):                  repetitions dimension size to add after (if an int will turn into a tuple).
        
        Returns:
            (numpy.ndarray):                     a view of a numpy array with tiling in various dimension.
        
        Examples:
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
            
            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_after = 1)
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])
            
            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_after = (1,))
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])
            
            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_before = 1)
            array([[[0, 1, 2],
                    [3, 4, 5]]])
            
            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_before = (1,))
            array([[[0, 1, 2],
                    [3, 4, 5]]])
            
            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_before = (3,))
            array([[[0, 1, 2],
                    [3, 4, 5]],
            <BLANKLINE>
                   [[0, 1, 2],
                    [3, 4, 5]],
            <BLANKLINE>
                   [[0, 1, 2],
                    [3, 4, 5]]])
            
            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_after = (4,))
            array([[[0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2]],
            <BLANKLINE>
                   [[3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5]]])
            
            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_before = (3,), reps_after = (4,))
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
            
            >>> expand_view(numpy.arange(6).reshape((2,3)), reps_before = (4,3))
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
    
    
    if type(reps_after) is not tuple:
        reps_after = (reps_after,)
        
    if type(reps_before) is not tuple:
        reps_before = (reps_before,)
        
    if (not reps_after) and (not reps_before):
        raise Exception("expand_view() requires reps_after or reps_before to specified.")
    
    return( numpy.lib.stride_tricks.as_strided(new_array, reps_before + new_array.shape + reps_after, len(reps_before) * (0,) + new_array.strides + len(reps_after) * (0,)) )



@advanced_debugging.log_call(logger)
def all_permutations_operation(new_op, new_array_1, new_array_2):
    """
        Takes two arrays and constructs a new array that contains the result
        of new_op on every permutation of elements in each array (like broadcasting).
        
        Suppose that new_result contained the result, then one would find that
        the result of the following operation on the specific indicies
        
        new_op( new_array_1[ i_1_1, i_1_2, ... ], new_array_2[ i_2_1, i_2_2, ... ] )
        
        would be found in new_result as shown
        
        new_result[ i_1_1, i_1_2, ..., i_2_1, i_2_2, ... ]
        
        
        Args:
            new_array(numpy.ndarray):            array to add the singleton axis to.
        
        Returns:
            (numpy.ndarray):                     a numpy array with the singleton axis added at the end.
        
        Examples:
            >>> all_permutations_operation(operator.add, numpy.ones((1,3)), numpy.eye(2)).shape
            (1, 3, 2, 2)
            
            >>> all_permutations_operation(operator.add, numpy.ones((2,2)), numpy.eye(2)).shape
            (2, 2, 2, 2)
        
            >>> all_permutations_operation(operator.add, numpy.ones((2,2)), numpy.eye(2))
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
            
            >>> all_permutations_operation(operator.sub, numpy.ones((2,2)), numpy.eye(2))
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
            
            >>> all_permutations_operation(operator.sub, numpy.ones((2,2)), numpy.fliplr(numpy.eye(2)))
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
            
            >>> all_permutations_operation(operator.sub, numpy.zeros((2,2)), numpy.eye(2))
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
    
    new_array_1_tiled = expand_view(new_array_1, reps_after = new_array_2.shape)
    new_array_2_tiled = expand_view(new_array_2, reps_before = new_array_1.shape)
    
    return( new_op(new_array_1_tiled, new_array_2_tiled) )


@advanced_debugging.log_call(logger)
def all_permutations_equal(new_array_1, new_array_2):
    """
        Takes two arrays and constructs a new array that contains the result
        of equality comparison on every permutation of elements in each array (like broadcasting).
        
        Suppose that new_result contained the result, then one would find that
        the result of the following operation on the specific indicies
        
        new_op( new_array_1[ i_1_1, i_1_2, ... ], new_array_2[ i_2_1, i_2_2, ... ] )
        
        would be found in new_result as shown
        
        new_result[ i_1_1, i_1_2, ..., i_2_1, i_2_2, ... ]
        
        
        Args:
            new_array(numpy.ndarray):            array to add the singleton axis to.
        
        Returns:
            (numpy.ndarray):                     a numpy array with the singleton axis added at the end.
        
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
            
            >>> all_permutations_equal(numpy.ones((2,2)), numpy.fliplr(numpy.eye(2)))
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
            
            >>> all_permutations_equal(numpy.arange(4).reshape((2,2)), numpy.arange(2,6).reshape((2,2)))
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
    
    return( all_permutations_operation(operator.eq, new_array_1, new_array_2) )


