"""
The ``registration`` module helps bring image stacks into alignment.

===============================================================================
Overview
===============================================================================
The registration algorithm developed by Wenzhi Sun. The strategy uses an
area-based method for registration. Namely, it takes a template image (the mean
projection) and finds the translations required for each frame to overlap
optimally with the template image. Then a number of adjustments are made to the
shifts to ensure that unnecessary translations are not preformed.

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jan 28, 2015 11:25:47 EST$"


import itertools
import os
import tempfile
import warnings

import h5py
import numpy


try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

try:
    import pyfftw.interfaces.numpy_fft as fft
except Exception as e:
    warnings.warn(str(e) + ". Falling back to NumPy FFTPACK.", ImportWarning)
    import numpy.fft as fft

from nanshe.util import iters, xnumpy
from nanshe.io import hdf5

# Need in order to have logging information no matter what.
from nanshe.util import prof


# Get the loggers
trace_logger = prof.getTraceLogger(__name__)
logger = prof.logging.getLogger(__name__)



@prof.log_call(trace_logger)
def register_mean_offsets(frames2reg,
                          max_iters=-1,
                          block_frame_length=-1,
                          include_shift=False,
                          to_truncate=False,
                          float_type=numpy.dtype(float).type):
    """
        This algorithm registers the given image stack against its mean
        projection. This is done by computing translations needed to put each
        frame in alignment. Then the translation is performed and new
        translations are computed. This is repeated until no further
        improvement can be made.

        The code for translations can be found in find_mean_offsets.

        Notes:
            Adapted from code provided by Wenzhi Sun with speed improvements
            provided by Uri Dubin.

        Args:
            frames2reg(numpy.ndarray):           Image stack to register (time
                                                 is the first dimension uses
                                                 C-order tyx or tzyx).

            max_iters(int):                      Number of iterations to allow
                                                 before forcing termination if
                                                 stable point is not found yet.
                                                 Set to -1 if no limit.
                                                 (Default -1)

            block_frame_length(int):             Number of frames to work with
                                                 at a time. By default all.
                                                 (Default -1)

            include_shift(bool):                 Whether to return the shifts
                                                 used, as well. (Default False)

            to_truncate(bool):                   Whether to truncate the frames
                                                 to remove all masked portions.
                                                 (Default False)

            float_type(type):                    Type of float to use for
                                                 calculation. (Default
                                                 numpy.float64).

        Returns:
            (numpy.ndarray):                     an array containing the
                                                 translations to apply to each
                                                 frame.

        Examples:
            >>> a = numpy.zeros((5, 3, 4)); a[:,0] = 1; a[2,0] = 0; a[2,2] = 1
            >>> a
            array([[[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.],
                    [ 1.,  1.,  1.,  1.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]]])

            >>> register_mean_offsets(a, include_shift=True)
            (masked_array(data =
             [[[1.0 1.0 1.0 1.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]
            <BLANKLINE>
             [[1.0 1.0 1.0 1.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]
            <BLANKLINE>
             [[-- -- -- --]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]
            <BLANKLINE>
             [[1.0 1.0 1.0 1.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]
            <BLANKLINE>
             [[1.0 1.0 1.0 1.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]],
                         mask =
             [[[False False False False]
              [False False False False]
              [False False False False]]
            <BLANKLINE>
             [[False False False False]
              [False False False False]
              [False False False False]]
            <BLANKLINE>
             [[ True  True  True  True]
              [False False False False]
              [False False False False]]
            <BLANKLINE>
             [[False False False False]
              [False False False False]
              [False False False False]]
            <BLANKLINE>
             [[False False False False]
              [False False False False]
              [False False False False]]],
                   fill_value = 0.0)
            , array([[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 0],
                   [0, 0]]))
    """

    float_type = numpy.dtype(float_type).type

    # Must be of type float and must be at least 32-bit (smallest complex type
    # uses two 32-bit floats).
    assert issubclass(float_type, numpy.floating)
    assert numpy.dtype(float_type).itemsize >= 4

    # Sadly, there is no easier way to map the two types; so, this is it.
    float_complex_mapping = {
        numpy.float32 : numpy.complex64,
        numpy.float64 : numpy.complex128,
        numpy.float128 : numpy.complex256
    }
    complex_type = float_complex_mapping[float_type]

    if block_frame_length == -1:
        block_frame_length = len(frames2reg)

    tempdir_name = ""
    temporaries_filename = ""
    if isinstance(frames2reg, h5py.Dataset):
        tempdir_name, temporaries_filename = os.path.split(
            os.path.abspath(frames2reg.file.filename)
        )

        temporaries_filename = os.path.splitext(temporaries_filename)[0]
        temporaries_filename += "_".join(
            [
                frames2reg.name.replace("/", "_"),
                "temporaries.h5"
            ]
        )
        temporaries_filename = os.path.join(
            tempdir_name,
            temporaries_filename
        )
    elif (block_frame_length != len(frames2reg)):
        tempdir_name = tempfile.mkdtemp()
        temporaries_filename = os.path.join(tempdir_name, "temporaries.h5")

    frames2reg_fft = None
    space_shift = None
    this_space_shift = None
    if tempdir_name:
        temporaries_file = h5py.File(temporaries_filename, "w")

        frames2reg_fft = temporaries_file.create_dataset(
            "frames2reg_fft", shape=frames2reg.shape, dtype=complex_type
        )
        space_shift = temporaries_file.create_dataset(
            "space_shift",
            shape=(len(frames2reg), len(frames2reg.shape)-1),
            dtype=int
        )
        this_space_shift = temporaries_file.create_dataset(
            "this_space_shift",
            shape=space_shift.shape,
            dtype=space_shift.dtype
        )
    else:
        frames2reg_fft = numpy.empty(frames2reg.shape, dtype=complex_type)
        space_shift = numpy.zeros(
            (len(frames2reg), len(frames2reg.shape)-1), dtype=int
        )
        this_space_shift = numpy.empty_like(space_shift)

    for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
        frames2reg_fft[range_ij] = fft.fftn(
            frames2reg[range_ij], axes=range(1, len(frames2reg.shape))
        )

    template_fft = numpy.empty(frames2reg.shape[1:], dtype=complex_type)

    this_space_shift_mean = numpy.empty(
        this_space_shift.shape[1:],
        dtype=this_space_shift.dtype
    )

    # Repeat shift calculation until there is no further adjustment.
    num_iters = 0
    squared_magnitude_delta_space_shift = 1.0
    while (squared_magnitude_delta_space_shift != 0.0):
        squared_magnitude_delta_space_shift = 0.0

        template_fft[:] = 0
        for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
            frames2reg_shifted_fft_ij = translate_fourier(
                frames2reg_fft[range_ij] / len(frames2reg),
                space_shift[range_ij]
            )
            template_fft += frames2reg_shifted_fft_ij.sum(axis=0)

        for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
            this_space_shift[range_ij] = find_offsets(
                frames2reg_fft[range_ij], template_fft
            )

        # Remove global shifts.
        this_space_shift_mean[...] = 0
        for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
            this_space_shift_mean += this_space_shift[range_ij].sum(axis=0)
        this_space_shift_mean[...] = numpy.round(
            this_space_shift_mean.astype(float_type) / len(this_space_shift)
        ).astype(this_space_shift_mean.dtype)
        for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
            this_space_shift[range_ij] = xnumpy.find_relative_offsets(
                this_space_shift[range_ij],
                this_space_shift_mean
            )

        # Find the shortest roll possible (i.e. if it is going over halfway
        # switch direction so it will go less than half).
        # Note all indices by definition were positive semi-definite and upper
        # bounded by the shape. This change will make them bound by
        # the half shape, but with either sign.
        for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
            this_space_shift[range_ij] = xnumpy.find_shortest_wraparound(
                this_space_shift[range_ij],
                frames2reg_fft.shape[1:]
            )

        for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
            delta_space_shift_ij = this_space_shift[range_ij] - \
                                   space_shift[range_ij]
            squared_magnitude_delta_space_shift += numpy.dot(
                delta_space_shift_ij, delta_space_shift_ij.T
            ).sum()

        for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
            space_shift[range_ij] = this_space_shift[range_ij]

        num_iters += 1
        logger.info(
            "Completed iteration, %i, " %
            num_iters
            + "where the L_2 norm squared of the relative shift was, %f." %
            squared_magnitude_delta_space_shift
        )
        if (max_iters != -1) and (num_iters >= max_iters):
            logger.info("Hit maximum number of iterations.")
            break

    reg_frames_shape = frames2reg.shape
    if to_truncate:
        space_shift_max = numpy.zeros(
            space_shift.shape[1:], dtype=space_shift.dtype
        )
        space_shift_min = numpy.zeros(
            space_shift.shape[1:], dtype=space_shift.dtype
        )
        for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
            numpy.maximum(
                space_shift_max,
                space_shift[range_ij].max(axis=0),
                out=space_shift_max
            )
            numpy.minimum(
                space_shift_min,
                space_shift[range_ij].min(axis=0),
                out=space_shift_min
            )
        reg_frames_shape = numpy.asarray(reg_frames_shape)
        reg_frames_shape[1:] -= space_shift_max
        reg_frames_shape[1:] += space_shift_min
        reg_frames_shape = tuple(reg_frames_shape)

        space_shift_max = tuple(space_shift_max)
        space_shift_min = space_shift_min.astype(object)
        space_shift_min[space_shift_min == 0] = None
        space_shift_min = tuple(space_shift_min)
        reg_frames_slice = tuple(
            slice(_1, _2) for _1, _2 in itertools.izip(
                space_shift_max, space_shift_min
            )
        )

    # Adjust the registered frames using the translations found.
    # Mask rolled values.
    reg_frames = None
    if tempdir_name:
        if to_truncate:
            reg_frames = temporaries_file.create_dataset(
                "reg_frames",
                shape=reg_frames_shape,
                dtype=frames2reg.dtype,
                chunks=True
            )
        else:
            reg_frames = temporaries_file.create_group("reg_frames")
            reg_frames = hdf5.serializers.HDF5MaskedDataset(
                reg_frames, shape=frames2reg.shape, dtype=frames2reg.dtype
            )
    else:
        if to_truncate:
            reg_frames = numpy.empty(reg_frames_shape, dtype=frames2reg.dtype)
        else:
            reg_frames = numpy.ma.empty_like(frames2reg)
            reg_frames.mask = numpy.ma.getmaskarray(reg_frames)
            reg_frames.set_fill_value(reg_frames.dtype.type(0))

    for range_ij in iters.subrange(0, len(frames2reg), block_frame_length):
        for k in range_ij:
            if to_truncate:
                reg_frames[k] = xnumpy.roll(
                    frames2reg[k], space_shift[k]
                )[reg_frames_slice]
            else:
                reg_frames[k] = xnumpy.roll(
                    frames2reg[k], space_shift[k], to_mask=True
                )

    result = None
    results_filename = ""
    if tempdir_name:
        result = results_filename
        results_filename = os.path.join(tempdir_name, "results.h5")
        results_file = h5py.File(results_filename, "w")
        if to_truncate:
            temporaries_file.copy(reg_frames.name, results_file)
        else:
            temporaries_file.copy(reg_frames.group, results_file)
        if include_shift:
            temporaries_file.copy(space_shift.name, results_file)
        frames2reg_fft = None
        reg_frames = None
        space_shift = None
        this_space_shift = None
        temporaries_file.close()
        os.remove(temporaries_filename)
        temporaries_filename = ""
        result = results_filename
    else:
        result = reg_frames
        if include_shift:
            result = (reg_frames, space_shift)

    if tempdir_name:
        results_file.close()
        results_file = None

    return(result)


@prof.log_call(trace_logger)
@lru_cache(maxsize=2)
def generate_unit_phase_shifts(shape, float_type=float):
    """
        Computes the complex phase shift's angle due to a unit spatial shift.

        This is meant to be a helper function for ``register_mean_offsets``. It
        does this by computing a table of the angle of the phase of a unit
        shift in each dimension (with a factor of :math:`2\pi`).

        This allows arbitrary phase shifts to be made in each dimensions by
        multiplying these angles by the size of the shift and added to the
        existing angle to induce the proper phase shift in fourier space, which
        is equivalent to the spatial translation.

        Args:
            shape(tuple of ints):       shape of the data to be shifted.

            float_type(real type):      phase type (default numpy.float64)

        Returns:
            (numpy.ndarray):            an array containing the angle of the
                                        complex phase shift to use for each
                                        dimension.

        Examples:
            >>> generate_unit_phase_shifts((2,4))
            array([[[-0.        , -0.        , -0.        , -0.        ],
                    [-3.14159265, -3.14159265, -3.14159265, -3.14159265]],
            <BLANKLINE>
                   [[-0.        , -1.57079633, -3.14159265, -4.71238898],
                    [-0.        , -1.57079633, -3.14159265, -4.71238898]]])
    """

    # Convert to `numpy`-based type if not done already.
    float_type = numpy.dtype(float_type).type

    # Must be of type float.
    assert issubclass(float_type, numpy.floating)
    assert numpy.dtype(float_type).itemsize >= 4

    # Get the negative wave vector
    negative_wave_vector = numpy.asarray(shape, dtype=float_type)
    numpy.reciprocal(negative_wave_vector, out=negative_wave_vector)
    negative_wave_vector *= 2*numpy.pi
    numpy.negative(negative_wave_vector, out=negative_wave_vector)

    # Get the indices for each point in the selected space.
    indices = xnumpy.cartesian_product([numpy.arange(_) for _ in shape])

    # Determine the phase offset for each point in space.
    complex_angle_unit_shift = indices * negative_wave_vector
    complex_angle_unit_shift = complex_angle_unit_shift.T.copy()
    complex_angle_unit_shift = complex_angle_unit_shift.reshape(
        (len(shape),) + shape
    )

    return(complex_angle_unit_shift)


@prof.log_call(trace_logger)
def translate_fourier(frame_fft, shift):
    """
        Translates frame(s) of data in Fourier space using the shift(s) given.

        Args:
            frame_fft(complex array):   Either a single frame with C-order axes
                                        or multiple frames with time on the 0th
                                        axis.

            shift(array of ints):       Either the shift for each dimension
                                        with C-ordered values or multiple
                                        frames with time on the 0th axis.

        Returns:
            (numpy.ndarray):            The frame(s) shifted.

        Examples:
            >>> a = numpy.arange(12).reshape(3,4).astype(float)
            >>> a
            array([[  0.,   1.,   2.,   3.],
                   [  4.,   5.,   6.,   7.],
                   [  8.,   9.,  10.,  11.]])
            >>> af = fft.fftn(a, axes=tuple(xrange(a.ndim)))
            >>> numpy.around(af, decimals=10)
            array([[ 66. +0.j        ,  -6. +6.j        ,  -6. +0.j        ,  -6. -6.j        ],
                   [-24.+13.85640646j,   0. +0.j        ,   0. +0.j        ,   0. +0.j        ],
                   [-24.-13.85640646j,   0. +0.j        ,   0. +0.j        ,   0. +0.j        ]])

            >>> s = numpy. array([1, -1])

            >>> atf = translate_fourier(af, s)
            >>> numpy.around(atf, decimals=10)
            array([[ 66. +0.j        ,  -6. -6.j        ,   6. -0.j        ,  -6. +6.j        ],
                   [ 24.+13.85640646j,   0. +0.j        ,   0. +0.j        ,  -0. +0.j        ],
                   [ 24.-13.85640646j,   0. -0.j        ,   0. +0.j        ,   0. +0.j        ]])

            >>> fft.ifftn(
            ...     atf, axes=tuple(xrange(a.ndim))
            ... ).real.round().astype(int).astype(float)
            array([[  9.,  10.,  11.,   8.],
                   [  1.,   2.,   3.,   0.],
                   [  5.,   6.,   7.,   4.]])

            >>> a = a[None]; af = af[None]; s = s[None]
            >>> atf = translate_fourier(af, s)
            >>> numpy.around(atf, decimals=10)
            array([[[ 66. +0.j        ,  -6. -6.j        ,   6. -0.j        ,  -6. +6.j        ],
                    [ 24.+13.85640646j,   0. +0.j        ,   0. +0.j        ,  -0. +0.j        ],
                    [ 24.-13.85640646j,   0. -0.j        ,   0. +0.j        ,   0. +0.j        ]]])


            >>> fft.ifftn(
            ...     atf, axes=tuple(xrange(1, a.ndim))
            ... ).real.round().astype(int).astype(float)
            array([[[  9.,  10.,  11.,   8.],
                    [  1.,   2.,   3.,   0.],
                    [  5.,   6.,   7.,   4.]]])

    """

    add_frame_axis = False
    if (len(shift.shape) == 1) and (len(shift) == len(frame_fft.shape)):
        add_frame_axis = True
        shift = shift[None]
        frame_fft = frame_fft[None]

    assert (
        (len(shift.shape) == 2) and
        (shift.shape[1] == (len(frame_fft.shape) - 1))
    ), "Shapes are incompatible." + \
       ("`shift.shape = %s`" % repr(shift.shape)) + \
       (" and `frame_fft.shape = %s`." % repr(frame_fft.shape))

    # Sadly, there is no easier way to map the two types; so, this is it.
    complex_type = frame_fft.dtype.type
    complex_float_mapping = {
        numpy.complex64 : numpy.float32,
        numpy.complex128 : numpy.float64,
        numpy.complex256 : numpy.float128
    }
    float_type = complex_float_mapping[complex_type]
    J = complex_type(1j)

    # Get unit translations in all directions as the complex phase's angle.
    unit_space_shift_fft = generate_unit_phase_shifts(
        frame_fft.shape[1:], float_type=float_type
    )

    # Compute phase adjustment in complex.
    frame_fft_shifted = numpy.exp(
        J * numpy.tensordot(
                shift,
                unit_space_shift_fft,
                axes=[-1, 0]
            )
    )
    frame_fft_shifted *= frame_fft

    if add_frame_axis:
        frame_fft_shifted = frame_fft_shifted[0]

    return(frame_fft_shifted)


@prof.log_call(trace_logger)
def find_offsets(frames2reg_fft, template_fft):
    """
        Computes the convolution of the template with the frames by taking
        advantage of their FFTs for faster computation that an ordinary
        convolution ( O(N*lg(N)) vs O(N^2) )
        < http://ccrma.stanford.edu/~jos/ReviewFourier/FFT_Convolution_vs_Direct.html >.
        Once computed the maximum of the convolution is found to determine the
        best overlap of each frame with the template, which provides the needed
        offset. Some corrections are performed to make reasonable offsets.

        Notes:
            Adapted from code provided by Wenzhi Sun with speed improvements
            provided by Uri Dubin.

        Args:
            frames2reg(numpy.ndarray):           image stack to register (time
                                                 is the first dimension uses
                                                 C-order tyx or tzyx).

            template_fft(numpy.ndarray):         what to register the image
                                                 stack against (single frame
                                                 using C-order yx or zyx).

        Returns:
            (numpy.ndarray):                     an array containing the
                                                 translations to apply to each
                                                 frame.

        Examples:
            >>> a = numpy.zeros((5, 3, 4)); a[:,0] = 1; a[2,0] = 0; a[2,2] = 1
            >>> a
            array([[[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.],
                    [ 1.,  1.,  1.,  1.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]]])

            >>> af = numpy.fft.fftn(a, axes=range(1, a.ndim)); af
            array([[[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]],
            <BLANKLINE>
                   [[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]],
            <BLANKLINE>
                   [[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [-2.+3.46410162j,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [-2.-3.46410162j,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]],
            <BLANKLINE>
                   [[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]],
            <BLANKLINE>
                   [[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]]])

            >>> tf = numpy.fft.fftn(a.mean(axis=0)); tf
            array([[ 4.0+0.j        ,  0.0+0.j        ,  0.0+0.j        ,  0.0+0.j        ],
                   [ 2.8+0.69282032j,  0.0+0.j        ,  0.0+0.j        ,  0.0+0.j        ],
                   [ 2.8-0.69282032j,  0.0+0.j        ,  0.0+0.j        ,  0.0+0.j        ]])

            >>> find_offsets(af, tf)
            array([[ 0,  0],
                   [ 0,  0],
                   [-2,  0],
                   [ 0,  0],
                   [ 0,  0]])
    """

    # If there is only one frame, add a singleton axis to indicate this.
    frames2reg_fft_added_singleton = (frames2reg_fft.ndim == template_fft.ndim)
    if frames2reg_fft_added_singleton:
        frames2reg_fft = frames2reg_fft[None]

    # Compute the product of the two FFTs (i.e. the convolution of the regular
    # versions).
    frames2reg_template_conv_fft = frames2reg_fft * template_fft.conj()[None]

    # Find the FFT inverse (over all spatial dimensions) to return to the
    # convolution.
    frames2reg_template_conv = fft.ifftn(
        frames2reg_template_conv_fft, axes=range(1, frames2reg_fft.ndim)
    )

    # Find where the convolution is maximal. Will have the most things in
    # common between the template and frames.
    frames2reg_template_conv_max, frames2reg_template_conv_max_indices = xnumpy.max_abs(
        frames2reg_template_conv,
        axis=range(1, frames2reg_fft.ndim),
        return_indices=True
    )

    # First index is just the frame, which will be in sequential order. We
    # don't need this so we drop it.
    frames2reg_template_conv_max_indices = frames2reg_template_conv_max_indices[1:]

    # Convert indices into an array for easy manipulation.
    frames2reg_template_conv_max_indices = numpy.array(
        frames2reg_template_conv_max_indices
    ).T.copy()

    # Shift will have to be in the opposite direction to bring everything to
    # the center.
    numpy.negative(
        frames2reg_template_conv_max_indices,
        out=frames2reg_template_conv_max_indices
    )

    return(frames2reg_template_conv_max_indices)
