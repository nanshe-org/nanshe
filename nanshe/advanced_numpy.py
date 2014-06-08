# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 20, 2014 9:46:45 AM$"

import numpy
import scipy
import operator

import scipy.spatial

# Need in order to have logging information no matter what.
import advanced_debugging


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)


@advanced_debugging.log_call(logger)
def renumber_label_image(new_array):
    """
        Takes a label image with non-consecutive numbering and renumbers it to be consecutive.
        Returns the relabeled image, a mapping from the old labels (by index) to the new ones,
        and a mapping from the new labels back to the old labels.
        
        Args:
            new_array(numpy.ndarray):                               the label image.
            
        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray):          the relabeled label image, the forward label mapping, and the reverse label mapping
        
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

    # Get the set of reverse label mapping (ensure the background is always included)
    reverse_label_mapping = numpy.unique(numpy.array([0] + numpy.unique(new_array).tolist()))

    # Get the set of old labels excluding background
    old_labels = reverse_label_mapping[reverse_label_mapping != 0]

    # Get the set of new labels in order
    new_labels = numpy.arange(1, len(old_labels) + 1)

    # Get the forward label mapping (ensure the background is included)
    forward_label_mapping = numpy.zeros((reverse_label_mapping.max() + 1,), dtype = new_array.dtype)
    forward_label_mapping[old_labels] = new_labels

    # Get masks for each old label
    new_array_label_masks = all_permutations_equal(old_labels, new_array)

    # Create tiled where each label is expanded to the size of the new_array
    new_labels_tiled_view = expand_view(new_labels, new_array.shape)

    # Take every mask and make sure it has the appropriate sequential label
    # Then combine each of these parts of the label image together into a new sequential label image
    new_array_relabeled = (new_array_label_masks * new_labels_tiled_view).sum(axis = 0)

    return((new_array_relabeled, forward_label_mapping, reverse_label_mapping))


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

    # return( new_array[None] )
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

    # return( numpy.rollaxis(new_array[None], 0, new_array.ndim + 1) )
    return( add_singleton_axis_pos(new_array, new_pos = new_array.ndim) )


@advanced_debugging.log_call(logger)
def contains(new_array, to_contain):
    """
        Gets a mask array that is true every time something from to_contain appears in new_array.
        
        Args:
            new_array(numpy.ndarray):            array to check for matches.
            to_contain(array_like):              desired matches to find.
        
        Returns:
            (numpy.ndarray):                     a mask for new_array that selects values from to_contain.
        
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

    return(numpy.lib.stride_tricks.as_strided(new_array, reps_before + new_array.shape + reps_after,
                                               len(reps_before) * (0,) + new_array.strides + len(reps_after) * (0,)) )


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


class NotNumPyStructuredArrayType(Exception):
    """
        Designed for being thrown if a NumPy Structured Array is recieved.
    """
    pass


def numpy_structured_array_dtype_generator(new_array):
    """
        Takes a NumPy structured array and returns a generator that goes over
        each name in the structured array and yields the name, type, and shape
        (for the given name).
        
        Args:
            new_array(numpy.ndarray):       the array to get the info dtype from.
        
        Raises:
            (NotNumPyStructuredArrayType):  if it is a normal NumPy array.
        
        Returns:
            (iterator):                     An iterator yielding tuples.
    """

    # Test to see if this is a NumPy Structured Array
    if new_array.dtype.names:
        # Go through each name
        for each_name in new_array.dtype.names:
            # Get the type (want the actual type, not a str or dtype object)
            each_dtype = new_array[each_name].dtype.type
            # Get the shape (will be an empty tuple if no shape, which numpy.dtype accepts)
            each_shape = new_array.dtype[each_name].shape

            yield ( (each_name, each_dtype, each_shape) )
    else:
        raise NotNumPyStructuredArrayType("Not a NumPy structured array.")


def numpy_array_dtype_list(new_array):
    """
        Takes any NumPy array and returns either a list for a NumPy structured array
        via numpy_structured_array_dtype_generator or if it is a normal NumPy array
        it returns the type used.
        
        Args:
            new_array(numpy.ndarray):       the array to get the dtype info from.
        
        Returns:
            (list or type):                 something that can be given to numpy.dtype
                                            to obtain the new_array.dtype, but is more
                                            malleable than a numpy.dtype. 
    """

    try:
        return(list(numpy_structured_array_dtype_generator(new_array)))
    except NotNumPyStructuredArrayType:
        return(new_array.dtype.type)


def dot_product(new_vector_set_1, new_vector_set_2):
    """
        Determines the dot product between the two pairs of vectors from each set.
        
        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.
        
        Returns:
            (numpy.ndarray):                      an array with the distances between each pair of vectors from the first and second set.
        
        Examples:
            >>> (dot_product(numpy.eye(2), numpy.eye(2)) == numpy.eye(2)).all()
            True
            
            >>> (dot_product(numpy.eye(10), numpy.eye(10)) == numpy.eye(10)).all()
            True
            
            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[ 1,  0]]))
            array([[1]])
            
            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[ 0,  1]]))
            array([[0]])
            
            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[-1,  0]]))
            array([[-1]])
            
            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[ 0, -1]]))
            array([[0]])
            
            >>> dot_product(numpy.array([[ 1,  0]]), numpy.array([[ 1,  1]]))
            array([[1]])
    """

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(new_vector_set_1, new_vector_set_2.T)

    return(vector_pairs_dot_product)


def norm(new_vector_set, ord = 2):
    """
        Determines the norm of a vector or a set of vectors.
        
        Args:
            new_vector_set(numpy.ndarray):        either a single vector or a set of vectors (matrix).
            ord(optional):                        basically the same arguments as numpy.linalg.norm (though some are redundant here).
        
        Returns:
            (numpy.ndarray):                      an array with .
        
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
            
            >>> norm(numpy.array([[ 1,  1,  1], [ 1,  0,  1]]), 1)
            array([ 3.,  2.])
            
            >>> norm(numpy.array([[ 1,  1,  1], [ 1,  0,  1]]), 2)
            array([ 1.73205081,  1.41421356])
            
            >>> norm(numpy.array([ 0,  1,  2]))
            array(2.23606797749979)
    """

    new_vector_set = new_vector_set.astype(float)

    # Wrap the order parameter so as to avoid passing through numpy.apply_along_axis
    # and risk having it break. Also, makes sure the same function can be used in the
    # two cases.
    def wrapped_norm(new_vector):
        return(numpy.linalg.norm(new_vector, ord = ord))

    # Return a scalar NumPy array in the case of a single vector
    # Always return type float as the result.
    if new_vector_set.ndim == 1:
        return(numpy.array(wrapped_norm(new_vector_set)).astype(float))
    else:
        return(numpy.apply_along_axis(wrapped_norm, 1, new_vector_set).astype(float))


def dot_product_partially_normalized(new_vector_set_1, new_vector_set_2, ord = 2):
    """
        Determines the dot product between the two pairs of vectors from each set and creates a tuple with the dot product divided by one norm or the other.
        
        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.
            ord(optional):                        basically the same arguments as numpy.linalg.norm (though some are redundant here).
        
        Returns:
            (numpy.ndarray):                      an array with the normalized distances between each pair of vectors from the first and second set.
        
        Examples:
            >>> (numpy.array(dot_product_partially_normalized(numpy.eye(2), numpy.eye(2), 2)) == numpy.array((numpy.eye(2), numpy.eye(2),))).all()
            True
            
            >>> (numpy.array(dot_product_partially_normalized(numpy.eye(10), numpy.eye(10), 2)) == numpy.array((numpy.eye(10), numpy.eye(10),))).all()
            True
            
            >>> dot_product_partially_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 1,  0]]), 2)
            (array([[ 1.]]), array([[ 1.]]))
            
            >>> dot_product_partially_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 0,  1]]), 2)
            (array([[ 0.]]), array([[ 0.]]))
            
            >>> dot_product_partially_normalized(numpy.array([[ 1,  0]]), numpy.array([[-1,  0]]), 2)
            (array([[-1.]]), array([[-1.]]))
            
            >>> dot_product_partially_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 0, -1]]), 2)
            (array([[ 0.]]), array([[ 0.]]))
            
            >>> dot_product_partially_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 1,  1]]), 2)
            (array([[ 1.]]), array([[ 0.70710678]]))
            
            >>> dot_product_partially_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 1,  1]]), 1)
            (array([[ 1.]]), array([[ 0.5]]))
            
            >>> dot_product_partially_normalized( numpy.arange(6).reshape((2,3)), numpy.arange(5, 17).reshape((4,3)), 2 )  #doctest: +NORMALIZE_WHITESPACE
            (array([[  8.94427191,  12.96919427,  16.99411663,  21.01903899],
                   [ 10.46518036,  15.55634919,  20.64751801,  25.73868684]]),
             array([[ 1.90692518,  1.85274204,  1.82405837,  1.80635674],
                   [ 7.05562316,  7.02764221,  7.00822427,  6.99482822]]))
    """

    # Gets all of the norms
    new_vector_set_1_norms = norm(new_vector_set_1.astype(float), ord)
    new_vector_set_2_norms = norm(new_vector_set_2.astype(float), ord)

    # Expand the norms to have a shape equivalent to vector_pairs_dot_product
    new_vector_set_1_norms_expanded = expand_view(new_vector_set_1_norms, reps_after = new_vector_set_2.shape[0])
    new_vector_set_2_norms_expanded = expand_view(new_vector_set_2_norms, reps_before = new_vector_set_1.shape[0])

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(new_vector_set_1, new_vector_set_2.T)

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product_1_normalized = vector_pairs_dot_product / new_vector_set_1_norms_expanded
    vector_pairs_dot_product_2_normalized = vector_pairs_dot_product / new_vector_set_2_norms_expanded

    return( (vector_pairs_dot_product_1_normalized, vector_pairs_dot_product_2_normalized) )


def dot_product_normalized(new_vector_set_1, new_vector_set_2, ord = 2):
    """
        Determines the dot product between the two pairs of vectors from each set and divides them by the norm of the two.
        
        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.
            ord(optional):                        basically the same arguments as numpy.linalg.norm (though some are redundant here).
        
        Returns:
            (numpy.ndarray):                      an array with the normalized distances between each pair of vectors from the first and second set.
        
        Examples:
            >>> (dot_product_normalized(numpy.eye(2), numpy.eye(2), 2) == numpy.eye(2)).all()
            True
            
            >>> (dot_product_normalized(numpy.eye(10), numpy.eye(10), 2) == numpy.eye(10)).all()
            True
            
            >>> dot_product_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 1,  0]]), 2)
            array([[ 1.]])
            
            >>> dot_product_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 0,  1]]), 2)
            array([[ 0.]])
            
            >>> dot_product_normalized(numpy.array([[ 1,  0]]), numpy.array([[-1,  0]]), 2)
            array([[-1.]])
            
            >>> dot_product_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 0, -1]]), 2)
            array([[ 0.]])
            
            >>> dot_product_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 1,  1]]), 2)
            array([[ 0.70710678]])
            
            >>> dot_product_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 1,  1]]), 1)
            array([[ 0.5]])
            
            >>> dot_product_normalized( numpy.arange(6).reshape((2,3)), numpy.arange(5, 17).reshape((4,3)), 2 )
            array([[ 0.85280287,  0.82857143,  0.8157437 ,  0.80782729],
                   [ 0.9978158 ,  0.99385869,  0.99111258,  0.98921809]])
    """

    # Gets all of the norms
    new_vector_set_1_norms = norm(new_vector_set_1.astype(float), ord)
    new_vector_set_2_norms = norm(new_vector_set_2.astype(float), ord)

    # Finds the product of each combination for normalization
    norm_products = all_permutations_operation(operator.mul, new_vector_set_1_norms, new_vector_set_2_norms)

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(new_vector_set_1, new_vector_set_2.T)

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product_normalized = vector_pairs_dot_product / norm_products

    return(vector_pairs_dot_product_normalized)


def dot_product_L2_normalized(new_vector_set_1, new_vector_set_2):
    """
        Determines the dot product between the two pairs of vectors from each set and divides them by the L_2 norm of the two.
        
        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.
        
        Returns:
            (numpy.ndarray):                      an array with the distances between each pair of vectors from the first and second set.
        
        Examples:
            >>> (dot_product_L2_normalized(numpy.eye(2), numpy.eye(2)) == numpy.eye(2)).all()
            True
            
            >>> (dot_product_L2_normalized(numpy.eye(10), numpy.eye(10)) == numpy.eye(10)).all()
            True
            
            >>> dot_product_L2_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 1,  0]]))
            array([[ 1.]])
            
            >>> dot_product_L2_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 0,  1]]))
            array([[ 0.]])
            
            >>> dot_product_L2_normalized(numpy.array([[ 1,  0]]), numpy.array([[-1,  0]]))
            array([[-1.]])
            
            >>> dot_product_L2_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 0, -1]]))
            array([[ 0.]])
            
            >>> dot_product_L2_normalized(numpy.array([[ 1,  0]]), numpy.array([[ 1,  1]]))
            array([[ 0.70710678]])
            
            >>> dot_product_normalized( numpy.arange(6).reshape((2,3)), numpy.arange(5, 17).reshape((4,3)) )
            array([[ 0.85280287,  0.82857143,  0.8157437 ,  0.80782729],
                   [ 0.9978158 ,  0.99385869,  0.99111258,  0.98921809]])
    """

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_cosine_angle = 1 - scipy.spatial.distance.cdist(new_vector_set_1,
                                                                 new_vector_set_2,
                                                                 "cosine")

    return(vector_pairs_cosine_angle)