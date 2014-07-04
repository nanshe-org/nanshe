# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 20, 2014 9:46:45 AM$"

import numpy
import scipy
import operator

import scipy.spatial
import scipy.ndimage
import scipy.ndimage.morphology
import scipy.stats
import scipy.stats.mstats

import vigra

# Need in order to have logging information no matter what.
import debugging_tools


# Get the logger
logger = debugging_tools.logging.getLogger(__name__)


@debugging_tools.log_call(logger)
def renumber_label_image(new_array):
    """
        Takes a label image with non-consecutive numbering and renumbers it to be consecutive.
        Returns the relabeled image, a mapping from the old labels (by index) to the new ones,
        and a mapping from the new labels back to the old labels.
        
        Args:
            new_array(numpy.ndarray):                               the label image.
            
        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray):          the relabeled label image, the forward label mapping
                                                                    and the reverse label mapping
        
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


@debugging_tools.log_call(logger)
def index_axis_at_pos(new_array, axis, pos):
    """
        Indexes an arbitrary axis to the given position, which may be an index, a slice, or any other NumPy allowed
        indexing type. This will return a view.

        Args:
            new_array(numpy.ndarray):            array to add the singleton axis to.
            axis(int):                           position for the axis to be in the final array.
            pos(int or slice):                   how to index at the given axis.

        Returns:
            (numpy.ndarray):                     a numpy array view of the original array.

        Examples:
            >>> a = numpy.arange(24).reshape((1,2,3,4)); index_axis_at_pos(a, 0, 0).shape
            (2, 3, 4)

            >>> a = numpy.arange(24).reshape((1,2,3,4)); index_axis_at_pos(a, 1, 0).shape
            (1, 3, 4)

            >>> a = numpy.arange(24).reshape((1,2,3,4)); index_axis_at_pos(a, 2, 0).shape
            (1, 2, 4)

            >>> a = numpy.arange(24).reshape((1,2,3,4)); index_axis_at_pos(a, 3, 0).shape
            (1, 2, 3)

            >>> a = numpy.arange(24).reshape((1,2,3,4)); (index_axis_at_pos(a, 3, 0) == a[:,:,:,0]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4)); (index_axis_at_pos(a, -1, 0) == a[:,:,:,0]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4)); (index_axis_at_pos(a, -1, 2) == a[:,:,:,2]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4)); (index_axis_at_pos(a, 1, 1) == a[:,1,:,:]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4)); (index_axis_at_pos(a, 2, slice(None,None,2)) == a[:,:,::2,:]).all()
            True

            >>> a = numpy.arange(24).reshape((1,2,3,4)); index_axis_at_pos(a, 2, 2)[0, 1, 3] = 19; a[0, 1, 2, 3] == 19
            True

    """

    import additional_generators

    # Rescale axis inside the bounds
    axis %= (new_array.ndim)
    if axis < 0:
        axis += new_array.ndim


    # Ordering of the axes to generate
    axis_new_ordering = []
    # Place the chosen axis first (as all axes are positive semi-definite) and then 0 (if it is different from our axis)
    axis_new_ordering += sorted(set([axis, 0]), reverse = True)
    # Skip generating 0 or the chosen axis, but generate all others in normal order
    axis_new_ordering += list(additional_generators.xrange_with_skip(new_array.ndim, to_skip = [0, axis]))

    # Swaps the first with the desired axis (returns a view)
    new_array_swapped = new_array.transpose(axis_new_ordering)
    # Index to pos at the given axis
    new_subarray = new_array_swapped[pos]

    # Check to see if the chosen axis still exists (if pos were a slice)
    if new_subarray.ndim == new_array.ndim:
        # If so, generate the old ordering.
        axis_old_ordering = range(1, axis + 1) + [0] + range(axis + 1, new_array.ndim)
        # Transpose our selction to that ordering.
        new_subarray = new_subarray.transpose(axis_old_ordering)


    return( new_subarray )


@debugging_tools.log_call(logger)
def add_singleton_axis_pos(a_array, new_axis = 0):
    """
        Adds a singleton axis to the given position.
        Allows negative values for new_axis.
        Also, automatically bounds new_axis in an acceptable regime if it is not already.
        
        Args:
            a_array(numpy.ndarray):            array to add the singleton axis to.
            new_axis(int):                     position for the axis to be in the final array (defaults to zero).
        
        Returns:
            (numpy.ndarray):                   a numpy array with the singleton axis added (should be a view).
        
        Examples:
            >>> add_singleton_axis_pos(numpy.ones((7,9,6))).shape
            (1, 7, 9, 6)
            
            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), 0).shape
            (1, 7, 9, 6)
            
            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), new_axis = 0).shape
            (1, 7, 9, 6)
            
            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), new_axis = 1).shape
            (7, 1, 9, 6)
            
            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), new_axis = 2).shape
            (7, 9, 1, 6)
            
            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), new_axis = 3).shape
            (7, 9, 6, 1)
            
            >>> add_singleton_axis_pos(numpy.ones((7,9,6)), new_axis = -1).shape
            (7, 9, 6, 1)
            
    """

    # Clean up new_axis to be within the allowable range.
    new_axis %= (a_array.ndim + 1)
    if new_axis < 0:
        new_axis += a_array.ndim + 1

    # Constructing the current ordering of axis and the singleton dime
    new_array_shape = range(1, a_array.ndim + 1)
    new_array_shape.insert(new_axis, 0)
    new_array_shape = tuple(new_array_shape)

    # Adds singleton dimension at front.
    # Then changes the order so it is elsewhere.
    new_array = a_array[None]
    new_array = new_array.transpose(new_array_shape)

    return( new_array )


@debugging_tools.log_call(logger)
def add_singleton_axis_beginning(new_array):
    """
        Adds a singleton axis to the beginning of the array.
        
        Args:
            new_array(numpy.ndarray):            array to add the singleton axis to.
        
        Returns:
            (numpy.ndarray):                     a numpy array with the singleton axis added at the end (should be view)
        
        Examples:
            >>> add_singleton_axis_beginning(numpy.ones((7,9,6))).shape
            (1, 7, 9, 6)
            
            >>> add_singleton_axis_beginning(numpy.eye(3)).shape
            (1, 3, 3)
            
    """

    # return( new_array[None] )
    return( add_singleton_axis_pos(new_array, new_axis = 0) )


@debugging_tools.log_call(logger)
def add_singleton_axis_end(new_array):
    """
        Adds a singleton axis to the end of the array.
        
        Args:
            new_array(numpy.ndarray):            array to add the singleton axis to.
        
        Returns:
            (numpy.ndarray):                     a numpy array with the singleton axis added at the end (should be view)
        
        Examples:
            >>> add_singleton_axis_end(numpy.ones((7,9,6))).shape
            (7, 9, 6, 1)
            
            >>> add_singleton_axis_end(numpy.eye(3)).shape
            (3, 3, 1)
            
    """

    # return( numpy.rollaxis(new_array[None], 0, new_array.ndim + 1) )
    return( add_singleton_axis_pos(new_array, new_axis = new_array.ndim) )


@debugging_tools.log_call(logger)
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


@debugging_tools.log_call(logger)
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
            reps_after(tuple):                   repetitions dimension size to add before (if int will turn into tuple).
            reps_before(tuple):                  repetitions dimension size to add after (if int will turn into tuple).
        
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

    if not isinstance(reps_after, tuple):
        reps_after = (reps_after,)

    if not isinstance(reps_before, tuple):
        reps_before = (reps_before,)

    if (not reps_after) and (not reps_before):
        return(new_array.view())

    return(numpy.lib.stride_tricks.as_strided(new_array, reps_before + new_array.shape + reps_after,
                                               len(reps_before) * (0,) + new_array.strides + len(reps_after) * (0,)) )


def expand_arange(start, stop = None, step = 1, dtype=numpy.int64, reps_before = tuple(), reps_after = tuple()):
    """
        Much like numpy.arange except that it applies expand_view afterwards to get a view of the same arange in a
        larger cube.

        This is very useful for situations where broadcasting is desired.

        Args:
            start(int):                          starting point (or stopping point if only one is specified).
            stop(int):                           stopping point (if the starting point is specified) (0 by default).
            step(int):                           size of steps to take between value (1 by default).
            reps_after(tuple):                   repetitions dimension size to add before (if int will turn into tuple).
            reps_before(tuple):                  repetitions dimension size to add after (if int will turn into tuple).

        Returns:
            (numpy.ndarray):                     a view of a numpy arange with tiling in various dimension.

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

    an_arange = numpy.arange(start = start, stop = stop, step = step, dtype = dtype)

    an_arange = expand_view(an_arange, reps_before=reps_before, reps_after=reps_after)

    return(an_arange)


def expand_enumerate(new_array, axis = 0, start = 0, step = 1):
    """
        Builds on expand_arange, which has the same shape as the original array. Specifies the increments to occur along
        the given axis, which by default is the zeroth axis.

        Provides mechanisms for changing the starting value and also the increment.

        Args:
            new_array(numpy.ndarray):            array to enumerate
            axis(int):                           axis to enumerate along (0 by default).
            start(int):                          starting point (0 by default).
            step(int):                           size of steps to take between value (1 by default).

        Returns:
            (numpy.ndarray):                     a view of a numpy arange with tiling in various dimension.

        Examples:
            >>> expand_enumerate(numpy.ones((4,5)))
            array([[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3]])

            >>> expand_enumerate(numpy.ones((4,5)), axis=0)
            array([[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3]])

            >>> expand_enumerate(numpy.ones((4,5)), axis=0, start=1)
            array([[1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4]])

            >>> expand_enumerate(numpy.ones((4,5)), axis=0, start=1, step=2)
            array([[1, 1, 1, 1, 1],
                   [3, 3, 3, 3, 3],
                   [5, 5, 5, 5, 5],
                   [7, 7, 7, 7, 7]])

            >>> expand_enumerate(numpy.ones((4,5)), axis=1)
            array([[0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4]])

            >>> expand_enumerate(numpy.ones((4,5)), axis=1, start=1)
            array([[1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5]])

            >>> expand_enumerate(numpy.ones((4,5)), axis=1, start=1, step=2)
            array([[1, 3, 5, 7, 9],
                   [1, 3, 5, 7, 9],
                   [1, 3, 5, 7, 9],
                   [1, 3, 5, 7, 9]])

    """

    an_enumeration = expand_arange(start = start, stop = start + step * new_array.shape[axis], step = step,
                                   reps_before = new_array.shape[:axis], reps_after = new_array.shape[(axis+1):])

    return(an_enumeration)


@debugging_tools.log_call(logger)
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


@debugging_tools.log_call(logger)
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


@debugging_tools.log_call(logger)
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


@debugging_tools.log_call(logger)
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


@debugging_tools.log_call(logger)
def dot_product(new_vector_set_1, new_vector_set_2):
    """
        Determines the dot product between the two pairs of vectors from each set.
        
        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.
        
        Returns:
            (numpy.ndarray):                      an array with the distances between each pair of vectors.
        
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

            >>> dot_product(numpy.array([[ True,  False]]), numpy.array([[ True,  True]]))
            array([[ 1.]])
    """

    new_vector_set_1_float = new_vector_set_1.astype(float)
    new_vector_set_2_float = new_vector_set_2.astype(float)

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(new_vector_set_1_float, new_vector_set_2_float.T)

    return(vector_pairs_dot_product)


@debugging_tools.log_call(logger)
def norm(new_vector_set, ord = 2):
    """
        Determines the norm of a vector or a set of vectors.
        
        Args:
            new_vector_set(numpy.ndarray):        either a single vector or a set of vectors (matrix).
            ord(optional):                        basically the same arguments as numpy.linalg.norm
                                                  (though some are redundant here).
        
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
    """

    new_vector_set_float = new_vector_set.astype(float)

    # Wrap the order parameter so as to avoid passing through numpy.apply_along_axis
    # and risk having it break. Also, makes sure the same function can be used in the
    # two cases.
    def wrapped_norm(new_vector):
        return(numpy.linalg.norm(new_vector, ord = ord))

    result = None

    # Return a scalar NumPy array in the case of a single vector
    # Always return type float as the result.
    if new_vector_set.ndim == 1:
        result = numpy.array(wrapped_norm(new_vector_set_float)).astype(float)
    else:
        result = numpy.apply_along_axis(wrapped_norm, 1, new_vector_set_float).astype(float)

    return(result)


@debugging_tools.log_call(logger)
def dot_product_partially_normalized(new_vector_set_1, new_vector_set_2, ord = 2):
    """
        Determines the dot product between the two pairs of vectors from each set and creates a tuple
        with the dot product divided by one norm or the other.
        
        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.
            ord(optional):                        basically the same arguments as numpy.linalg.norm
        
        Returns:
            (numpy.ndarray):                      an array with the normalized distances between each pair of vectors.
        
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

            >>> dot_product_partially_normalized(numpy.array([[ True,  False]]), numpy.array([[ True,  True]]), 2)
            (array([[ 1.]]), array([[ 0.70710678]]))

            >>> dot_product_partially_normalized(numpy.array([[ True,  False]]), numpy.array([[ True,  True]]), 1)
            (array([[ 1.]]), array([[ 0.5]]))
            
            >>> dot_product_partially_normalized( numpy.arange(6).reshape((2,3)), numpy.arange(5, 17).reshape((4,3)), 2 )  #doctest: +NORMALIZE_WHITESPACE
            (array([[  8.94427191,  12.96919427,  16.99411663,  21.01903899],
                   [ 10.46518036,  15.55634919,  20.64751801,  25.73868684]]),
             array([[ 1.90692518,  1.85274204,  1.82405837,  1.80635674],
                   [ 7.05562316,  7.02764221,  7.00822427,  6.99482822]]))
    """

    new_vector_set_1_float = new_vector_set_1.astype(float)
    new_vector_set_2_float = new_vector_set_2.astype(float)

    # Gets all of the norms
    new_vector_set_1_norms = norm(new_vector_set_1_float, ord)
    new_vector_set_2_norms = norm(new_vector_set_2_float, ord)

    # Expand the norms to have a shape equivalent to vector_pairs_dot_product
    new_vector_set_1_norms_expanded = expand_view(new_vector_set_1_norms, reps_after = new_vector_set_2_float.shape[0])
    new_vector_set_2_norms_expanded = expand_view(new_vector_set_2_norms, reps_before = new_vector_set_1_float.shape[0])

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(new_vector_set_1_float, new_vector_set_2_float.T)

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product_1_normalized = vector_pairs_dot_product / new_vector_set_1_norms_expanded
    vector_pairs_dot_product_2_normalized = vector_pairs_dot_product / new_vector_set_2_norms_expanded

    return( (vector_pairs_dot_product_1_normalized, vector_pairs_dot_product_2_normalized) )


@debugging_tools.log_call(logger)
def dot_product_normalized(new_vector_set_1, new_vector_set_2, ord = 2):
    """
        Determines the dot product between a pair of vectors from each set and divides them by the norm of the two.
        
        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.
            ord(optional):                        basically the same arguments as numpy.linalg.norm.
        
        Returns:
            (numpy.ndarray):                      an array with the normalized distances between each pair of vectors.
        
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

            >>> dot_product_normalized(numpy.array([[ True,  False]]), numpy.array([[ True,  True]]), 2)
            array([[ 0.70710678]])

            >>> dot_product_normalized(numpy.array([[ True,  False]]), numpy.array([[ True,  True]]), 1)
            array([[ 0.5]])
            
            >>> dot_product_normalized( numpy.arange(6).reshape((2,3)), numpy.arange(5, 17).reshape((4,3)), 2)
            array([[ 0.85280287,  0.82857143,  0.8157437 ,  0.80782729],
                   [ 0.9978158 ,  0.99385869,  0.99111258,  0.98921809]])
    """

    new_vector_set_1_float = new_vector_set_1.astype(float)
    new_vector_set_2_float = new_vector_set_2.astype(float)

    # Gets all of the norms
    new_vector_set_1_norms = norm(new_vector_set_1_float, ord = ord)
    new_vector_set_2_norms = norm(new_vector_set_2_float, ord = ord)

    if not new_vector_set_1_norms.shape:
        new_vector_set_1_norms = numpy.array([new_vector_set_1_norms])

    if not new_vector_set_2_norms.shape:
        new_vector_set_2_norms = numpy.array([new_vector_set_2_norms])

    # Finds the product of each combination for normalization
    norm_products = all_permutations_operation(operator.mul, new_vector_set_1_norms, new_vector_set_2_norms)

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product = numpy.dot(new_vector_set_1_float, new_vector_set_2_float.T)

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_dot_product_normalized = vector_pairs_dot_product / norm_products

    return(vector_pairs_dot_product_normalized)


@debugging_tools.log_call(logger)
def dot_product_L2_normalized(new_vector_set_1, new_vector_set_2):
    """
        Determines the dot product between a pair of vectors from each set and divides them by the L_2 norm of the two.
        
        Args:
            new_vector_set_1(numpy.ndarray):      first set of vectors.
            new_vector_set_2(numpy.ndarray):      second set of vectors.
        
        Returns:
            (numpy.ndarray):                      an array with the distances between each pair of vectors.
        
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

            >>> dot_product_L2_normalized(numpy.array([[ True,  False]]), numpy.array([[ True,  True]]))
            array([[ 0.70710678]])
            
            >>> dot_product_L2_normalized( numpy.arange(6).reshape((2,3)), numpy.arange(5, 17).reshape((4,3)) )
            array([[ 0.85280287,  0.82857143,  0.8157437 ,  0.80782729],
                   [ 0.9978158 ,  0.99385869,  0.99111258,  0.98921809]])
    """

    new_vector_set_1_float = new_vector_set_1.astype(float)
    new_vector_set_2_float = new_vector_set_2.astype(float)

    # Measure the dot product between any two neurons (i.e. related to the angle of separation)
    vector_pairs_cosine_angle = 1 - scipy.spatial.distance.cdist(new_vector_set_1_float,
                                                                 new_vector_set_2_float,
                                                                 "cosine")

    return(vector_pairs_cosine_angle)


def generate_contour(a_image, separation_distance = 1.0, margin = 1.0):
    """
        Takes an image and extracts labeled contours from the mask using some minimum distance from the mask edge
        and some margin.

        Args:
            a_image(numpy.ndarray):            takes an image.
            separation_distance(float):        a separation distance from the edge of the mask for the center of the contour.
            margin(float):                     the width of contour.

        Returns:
            (numpy.ndarray):                   an array with the labeled contours.

        Examples:
            >>> a = numpy.array([[ True,  True, False],
            ...                  [False, False, False],
            ...                  [ True,  True,  True]], dtype=bool);
            >>> generate_contour(a)
            array([[ True,  True, False],
                   [False, False, False],
                   [ True,  True,  True]], dtype=bool)

            >>> generate_contour(numpy.eye(3))
            array([[ 1.,  0.,  0.],
                   [ 0.,  1.,  0.],
                   [ 0.,  0.,  1.]])

            >>> a = numpy.array([[False, False,  True, False, False, False,  True],
            ...                  [ True, False, False, False,  True, False, False],
            ...                  [ True,  True, False,  True,  True, False,  True],
            ...                  [ True, False, False,  True,  True, False, False],
            ...                  [ True, False, False, False, False, False, False],
            ...                  [False,  True, False, False, False, False,  True],
            ...                  [False,  True,  True, False, False, False, False]], dtype=bool)
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

    a_mask_transformed = scipy.ndimage.morphology.distance_transform_edt(a_image)

    above_lower_threshold = (lower_threshold <= a_mask_transformed)
    below_upper_threshold = (a_mask_transformed <= upper_threshold)

    a_mask_transformed_thresholded = (above_lower_threshold & below_upper_threshold)

    a_image_contours = a_image * a_mask_transformed_thresholded

    return(a_image_contours)


def generate_labeled_contours(a_mask, separation_distance = 1.0, margin = 1.0):
    """
        Takes a bool mask and extracts labeled contours from the mask using some minimum distance from the mask edge
        and some margin.

        Args:
            a_mask(numpy.ndarray):             takes a bool mask.
            separation_distance(float):        a separation distance from the edge of the mask for the center of the contour.
            margin(float):                     the width of contour.

        Returns:
            (numpy.ndarray):                   an array with the labeled contours.

        Examples:
            >>> a = numpy.array([[ True,  True, False],
            ...                  [False, False, False],
            ...                  [ True,  True,  True]], dtype=bool);
            >>> generate_labeled_contours(a)
            array([[1, 1, 0],
                   [0, 0, 0],
                   [2, 2, 2]], dtype=int32)

            >>> generate_labeled_contours(numpy.eye(3))
            array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]], dtype=int32)

            >>> a = numpy.array([[False, False,  True, False, False, False,  True],
            ...                  [ True, False, False, False,  True, False, False],
            ...                  [ True,  True, False,  True,  True, False,  True],
            ...                  [ True, False, False,  True,  True, False, False],
            ...                  [ True, False, False, False, False, False, False],
            ...                  [False,  True, False, False, False, False,  True],
            ...                  [False,  True,  True, False, False, False, False]], dtype=bool);
            >>> generate_labeled_contours(a)
            array([[0, 0, 1, 0, 0, 0, 2],
                   [3, 0, 0, 0, 4, 0, 0],
                   [3, 3, 0, 4, 4, 0, 5],
                   [3, 0, 0, 4, 4, 0, 0],
                   [3, 0, 0, 0, 0, 0, 0],
                   [0, 3, 0, 0, 0, 0, 6],
                   [0, 3, 3, 0, 0, 0, 0]], dtype=int32)
    """

    a_mask_contoured = generate_contour(a_mask, separation_distance = separation_distance, margin = margin)

    a_mask_contoured_labeled = scipy.ndimage.label(a_mask_contoured, structure = numpy.ones( (3,) * a_mask.ndim ))[0]

    return(a_mask_contoured_labeled)


def get_quantiles(probs):
    """
        Determines the probabilites for quantiles for given data much like MATLAB's function

        Args:
            data(numpy.ndarray):                        to find the quantiles of.

            probs(int or float or numpy.ndarray):       either some sort of integer for the number of quantiles
                                                            or a single float specifying which quantile to get
                                                            or an array of floats specifying the division for
                                                            each quantile in the the range (0, 1).

            axis(int or None):                          the axis to perform the calculation on (if default (None) then
                                                            all, otherwise only on a particular axis.

        Returns:
            (numpy.ma.MaskedArray):                     an array with the quantiles (the first dimension will be
                                                        the same length as probs).

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

    probs_array = None
    if isinstance(probs, (numpy.int, numpy.int_, numpy.int8, numpy.int16, numpy.int32, numpy.int64)):
        num_quantiles = probs
        probs_array = numpy.linspace(0, 1, num_quantiles + 2)[1:-1]
    elif isinstance(probs, (numpy.float, numpy.float_, numpy.float128, numpy.float16, numpy.float32, numpy.float64)):
        a_quantile = probs
        probs_array = numpy.array([a_quantile])
    else:
        probs_array = numpy.array(probs)
        probs_array.sort()


    if not ((0 < probs_array) & (probs_array < 1)).all():
        raise Exception("Cannot pass values that are not within the range (0, 1).")


    return(probs_array)


def quantile(data, probs, axis = None):
    """
        Determines the quantiles for given data much like MATLAB's function

        Args:
            data(numpy.ndarray):                        to find the quantiles of.

            probs(int or float or numpy.ndarray):       either some sort of integer for the number of quantiles
                                                            or a single float specifying which quantile to get
                                                            or an array of floats specifying the division for
                                                            each quantile in the the range (0, 1).

            axis(int or None):                          the axis to perform the calculation on (if default (None) then
                                                            all, otherwise only on a particular axis.

        Returns:
            (numpy.ma.MaskedArray):                     an array with the quantiles (the first dimension will be the
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

            >>> quantile(numpy.array([ 1.,  2.,  3.]), numpy.array([ 0.25,  0.5,  0.75]))
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

    new_quantiles = scipy.stats.mstats.mquantiles(data, probs_array, alphap=0.5, betap=0.5, axis=axis)

    if not isinstance(new_quantiles, numpy.ma.MaskedArray):
        new_quantiles = numpy.ma.MaskedArray(new_quantiles)

    new_quantiles.set_fill_value(numpy.nan)

    return(new_quantiles)


@debugging_tools.log_call(logger)
def tagging_reorder_array(new_array, from_axis_order = "tzyxc", to_axis_order = "tzyxc", to_copy = False):
    """
        Transforms one axis ordering to another giving a view of the array (unless otherwise specified).

        Args:
            new_array(numpy.ndarray):                   the array to reorder

            from_axis_order(str or list of str):        current labeled axis order.

            to_axis_order(str or list of str):          desired labeled axis order

            to_copy(bool):                              whether to return a view or a copy

        Returns:
            (numpy.ndarray):                            an array with the axis order specified (view).
    """

    if not isinstance(from_axis_order, str):
        from_axis_order = "".join([_ for _ in from_axis_order])

    if not isinstance(to_axis_order, str):
        to_axis_order = "".join([_ for _ in to_axis_order])

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